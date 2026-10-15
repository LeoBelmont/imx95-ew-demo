import numpy as np
from time import time
import cv2
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import tflite_runtime.interpreter as tflite


MODEL_PATH = "yolov8n_saved_model/yolov8n_full_integer_quant_converted.tflite"
VIDEO_PATH = "videos/input10fpsFinal.mp4"
USE_HW_ACCELERATED_INFERENCE = True
MINIMUM_SCORE = 0.4
FRAMERATE = 10
N_THREADS = 6  # number of threads for videoconvert

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


def open_capture(path: str) -> cv2.VideoCapture:
    """Single optimized pipeline with HW decoder."""
    pipeline = (
        f"filesrc location={path} ! "
        f"qtdemux ! queue ! h265parse ! "
        f"v4l2h265dec capture-io-mode=4 ! "
        f"videoconvert n-threads={N_THREADS} ! "
        f"video/x-raw,format=BGR ! "
        f"appsink sync=true"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if cap.isOpened():
        print("Pipeline opened successfully")
        return cap
    else:
        cap.release()
        return None


def main():
    Gst.init(None)

    cap_inference = open_capture(VIDEO_PATH)
    if cap_inference is None:
        print("Failed to open video pipeline")
        return

    width = int(cap_inference.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_inference.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = int((1.0 / FRAMERATE) * Gst.SECOND)

    # Output pipeline for display (Wayland)
    pipeline_str = (
        f"appsrc name=source is-live=true format=time "
        f"caps=video/x-raw,format=BGR,width={width},height={height} ! "
        f"queue max-size-buffers=1 leaky=downstream ! "
        f"videoconvert n-threads={N_THREADS} ! "
        f"waylandsink sync=false"
    )

    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("source")
    pipeline.set_state(Gst.State.PLAYING)

    number_frames = 0

    ret, frame = cap_inference.read()
    if not ret:
        print("Failed to read first frame")
        return

    while True:
        # ---------------- Inference ----------------

        image = cv2.resize(frame, (input_size, input_size))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, axis=0)

        if is_floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = (input_data.astype(np.int16) - 128).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], input_data)

        t1 = time()
        interpreter.invoke()
        t2 = time()

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

        cv2.putText(
            frame,
            f"Inference: {(t2 - t1):.3f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # ---------------- Push to Wayland ----------------

        data = GLib.Bytes.new_take(frame.tobytes())
        buf = Gst.Buffer.new_wrapped_bytes(data)

        buf.duration = duration
        timestamp = number_frames * duration
        buf.pts = buf.dts = timestamp
        buf.offset = number_frames

        number_frames += 1

        appsrc.emit("push-buffer", buf)

        # ---------------- Read next frame ----------------

        ret, frame = cap_inference.read()
        if not ret:
            cap_inference.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap_inference.read()
            if not ret:
                break

        # Crop possible decoder padding
        frame = frame[:height, :width]

    pipeline.set_state(Gst.State.NULL)
    cap_inference.release()


if __name__ == "__main__":
    main()

