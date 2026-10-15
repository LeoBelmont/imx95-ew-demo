import numpy as np
from time import time
import cv2
import gi
import threading
import queue as Q

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstApp, GstVideo

import tflite_runtime.interpreter as tflite


MODEL_PATH = "yolov8n_saved_model/my_model_optimized2.tflite"
VIDEO_PATH = "videos/supremis.mp4"
USE_HW_ACCELERATED_INFERENCE = True
MINIMUM_SCORE = 0.5
FRAMERATE = 30
N_THREADS = 12

# COCO class IDs and their BGR colors — only these classes are drawn
CLASS_CONFIG = {
    0: {"name": "person", "color": (0, 255, 0)},    # green
    2: {"name": "car",    "color": (255, 80, 0)},    # blue
    5: {"name": "bus",    "color": (0, 165, 255)},   # orange
}


# Load delegate
delegates = (
    [tflite.load_delegate("/usr/lib/libneutron_delegate.so")]
    if USE_HW_ACCELERATED_INFERENCE
    else []
)

# Load interpreter
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=delegates,
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_size = input_details[0]["shape"][1]
is_floating_model = input_details[0]["dtype"] == np.float32


def _build_decode_pipeline(path: str):
    """Build a GStreamer NV12 appsink pipeline.
    Returns (pipeline, appsink, width, height) or raises on failure."""
    pipeline_str = (
        f"filesrc location={path} ! "
        f"qtdemux ! queue ! h265parse ! "
        f"v4l2h265dec capture-io-mode=4 ! "
        f"video/x-raw,format=NV12 ! "
        f"appsink name=sink emit-signals=false sync=true "
        f"max-buffers=4 drop=true"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)

    # Wait until the sink pad has negotiated caps so we can read w/h
    import time as _time
    pad = sink.get_static_pad("sink")
    for _ in range(200):          # up to 2 s
        caps = pad.get_current_caps()
        if caps and not caps.is_empty() and not caps.is_any():
            break
        _time.sleep(0.01)
    else:
        raise RuntimeError("Decode pipeline failed to negotiate caps")

    s = caps.get_structure(0)
    width  = s.get_int("width").value
    height = s.get_int("height").value
    print(f"Decode pipeline ready: {width}x{height} NV12")
    return pipeline, sink, width, height


def _pull_bgr(sink, width: int, height: int):
    """Pull one NV12 sample and return a (height, width, 3) BGR ndarray.
    Uses GstVideoMeta plane offsets so HW-padding is never a problem."""
    sample = sink.emit("pull-sample")
    if sample is None:
        return None

    buf = sample.get_buffer()

    # GstVideoMeta carries the exact byte offsets the HW decoder used,
    # regardless of any row-alignment padding applied to the Y plane.
    meta = GstVideo.buffer_get_video_meta(buf)
    if meta is not None:
        y_off  = meta.offset[0]
        uv_off = meta.offset[1]
        y_str  = meta.stride[0]   # bytes per Y row (may be > width)
        uv_str = meta.stride[1]   # bytes per UV row
    else:
        # No meta — assume tightly packed NV12 (no padding)
        y_off, uv_off = 0, width * height
        y_str = uv_str = width

    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    data = np.frombuffer(mapinfo.data, dtype=np.uint8)

    uv_rows = height // 2

    # Build tight NV12 by extracting row-by-row using correct stride
    y_plane = np.stack([
        data[y_off  + r * y_str  : y_off  + r * y_str  + width]
        for r in range(height)
    ])                                         # (H, W)
    uv_plane = np.stack([
        data[uv_off + r * uv_str : uv_off + r * uv_str + width]
        for r in range(uv_rows)
    ])                                         # (H/2, W)

    buf.unmap(mapinfo)
    # explicit del so GStreamer can recycle the buffer immediately
    del sample

    nv12 = np.vstack([y_plane, uv_plane])      # (H*3//2, W) — no padding
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


def _reader_thread(frame_q: Q.Queue, stop: threading.Event, video_path: str,
                   width: int, height: int,
                   dec_pipeline, sink):
    """Decode NV12 via GStreamer appsink, convert to BGR, feed the queue."""
    # dec_pipeline / sink are already PLAYING (opened by main)

    while not stop.is_set():
        frame = _pull_bgr(sink, width, height)
        if frame is None:
            # EOF — restart
            dec_pipeline.set_state(Gst.State.NULL)
            try:
                dec_pipeline, sink, _, _ = _build_decode_pipeline(video_path)
            except RuntimeError:
                break
            continue

        if frame_q.full():
            try:
                frame_q.get_nowait()
            except Q.Empty:
                pass
        frame_q.put_nowait(frame)

    dec_pipeline.set_state(Gst.State.NULL)


def main():
    Gst.init(None)

    try:
        dec_pipeline, sink, width, height = _build_decode_pipeline(VIDEO_PATH)
    except RuntimeError as e:
        print(f"Failed to open video pipeline: {e}")
        return

    # Output pipeline for display (Wayland).
    # do-timestamp=true  → GStreamer stamps each buffer from its own clock,
    #                       so manually-computed pts/dts are no longer needed.
    # framerate=0/1      → variable-rate appsrc; no clock-driven back-pressure.
    # block=true         → push-buffer blocks when the queue is full instead
    #                       of silently dropping buffers.
    pipeline_str = (
        f"appsrc name=source is-live=true format=time "
        f"do-timestamp=true block=true "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate=0/1 ! "
        f"queue max-size-buffers=2 leaky=downstream ! "
        f"videoconvert n-threads={N_THREADS} ! "
        f"waylandsink sync=false"
    )

    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("source")
    pipeline.set_state(Gst.State.PLAYING)

    # Decode frames in a background thread so the decoder is never stalled
    # waiting for inference to finish.
    frame_q: Q.Queue = Q.Queue(maxsize=4)
    stop_event = threading.Event()
    reader = threading.Thread(
        target=_reader_thread,
        args=(frame_q, stop_event, VIDEO_PATH, width, height, dec_pipeline, sink),
        daemon=True,
    )
    reader.start()

    try:
        while True:
            # ---------------- Get next decoded frame ----------------
            tA = time()
            try:
                frame = frame_q.get(timeout=2.0)
            except Q.Empty:
                print("Frame queue empty — stopping")
                break
            tB = time()

            # ---------------- Pre-process ----------------
            image = cv2.resize(frame, (input_size, input_size))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb, axis=0)

            if is_floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = (input_data.astype(np.int16) - 128).astype(np.int8)

            interpreter.set_tensor(input_details[0]["index"], input_data)
            tC = time()

            # ---------------- Inference ----------------
            interpreter.invoke()
            tD = time()

            # ---------------- Post-process / decode ----------------
            raw = interpreter.get_tensor(output_details[0]["index"])[0]

            scale, zero_point = output_details[0]["quantization"]
            if scale != 0.0:
                raw = (raw.astype(np.float32) - zero_point) * scale

            if raw.shape[0] < raw.shape[1]:
                raw = raw.T

            box_raw = raw[:, :4]
            class_scores = raw[:, 4:]

            det_scores = class_scores.max(axis=1)
            det_classes = class_scores.argmax(axis=1)

            mask = det_scores > MINIMUM_SCORE
            det_scores = det_scores[mask]
            det_classes = det_classes[mask]
            box_raw = box_raw[mask]

            # Convert normalized cx,cy,w,h → pixel x,y,w,h for NMS
            nms_boxes = []
            for cx, cy, bw, bh in box_raw:
                x = int((cx - bw / 2) * width)
                y = int((cy - bh / 2) * height)
                nms_boxes.append([x, y, int(bw * width), int(bh * height)])

            indices = cv2.dnn.NMSBoxes(
                nms_boxes, det_scores.tolist(), MINIMUM_SCORE, 0.45
            )
            if len(indices) > 0:
                indices = indices.flatten()
            else:
                indices = []

            for i in indices:
                cls_id = int(det_classes[i])
                if cls_id not in CLASS_CONFIG:
                    continue
                cfg = CLASS_CONFIG[cls_id]
                x, y, bw, bh = nms_boxes[i]
                color = cfg["color"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                label = f"{cfg['name']} {det_scores[i]:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x, y - th - 6), (x + tw + 2, y), color, -1)
                cv2.putText(frame, label, (x + 1, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            tE = time()

            # ---------------- Overlay ----------------
            cv2.putText(
                frame,
                f"Inf:{(tD-tC)*1000:.1f}ms",
                (10, 720 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            # ---------------- Push to Wayland ----------------
            # Single copy: numpy → Python bytes, wrapped without extra GLib.Bytes.
            tF = time()
            buf = Gst.Buffer.new_wrapped(frame.tobytes())
            tG = time()
            appsrc.emit("push-buffer", buf)
            tH = time()

            print(
                f"queue:{(tB-tA)*1000:.1f}ms "
                f"pre:{(tC-tB)*1000:.1f}ms "
                f"invoke:{(tD-tC)*1000:.1f}ms "
                f"post:{(tE-tD)*1000:.1f}ms "
                f"tobytes:{(tG-tF)*1000:.1f}ms "
                f"push:{(tH-tG)*1000:.1f}ms "
                f"total:{(tH-tA)*1000:.1f}ms"
            )

    finally:
        stop_event.set()
        reader.join(timeout=2.0)
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
