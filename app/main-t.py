import numpy as np
from time import time
import cv2
import gi

gi.require_version('Gst', '1.0')

from gi.repository import Gst, GLib
import tflite_runtime.interpreter as tflite

MODEL_PATH = "yolov8s-seg_full_integer_quant_converted.tflite"
VIDEO_PATH = "input.mp4"

USE_HW_ACCELERATED_INFERENCE = True
MINIMUM_SCORE = 0.8
FRAMERATE = 30

# Load delegate
if USE_HW_ACCELERATED_INFERENCE:
    delegates = [tflite.load_delegate("/usr/lib/libneutron_delegate.so")]
else:
    delegates = []

# Load interpreter
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=delegates
)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]
is_floating_model = input_details[0]['dtype'] == np.float32


def open_capture(path: str) -> cv2.VideoCapture:
    """Try pipeline variants in order, return the first that opens successfully."""
    pipelines = [
        # i.MX HW decoder – explicit caps required for v4l2h264dec
        (f'filesrc location={path} ! qtdemux ! queue ! h264parse ! '
         f'v4l2h264dec capture-io-mode=4 ! videoconvert ! '
         f'video/x-raw,format=BGR ! appsink sync=false'),
        # i.MX HW decoder without capture-io-mode (older BSP)
        (f'filesrc location={path} ! qtdemux ! queue ! h264parse ! '
         f'v4l2h264dec ! videoconvert ! '
         f'video/x-raw,format=BGR ! appsink sync=false'),
        # Generic fallback via decodebin (software decode)
        (f'filesrc location={path} ! decodebin ! videoconvert ! '
         f'video/x-raw,format=BGR ! appsink sync=false'),
    ]

    for pl in pipelines:
        cap = cv2.VideoCapture(pl, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"Pipeline opened: {pl[:60]}...")
            return cap
        cap.release()

    return None

def main():
    Gst.init(None)

    cap = open_capture(VIDEO_PATH)
    if cap is None:
        print("Failed to open video: no working GStreamer pipeline found")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        return

    # Read the true container dimensions from the opened capture.
    # Hardware decoders pad height to multiples of 16 (e.g. 820 → 832), so we
    # use CAP_PROP values (reported before HW padding) and crop every frame.
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Crop decoded frame to true dimensions in case the HW decoder added padding
    frame = frame[:height, :width]
    duration = int((1.0 / FRAMERATE) * Gst.SECOND)

    # Output display pipeline via waylandsink
    pipeline_str = (
        f'appsrc name=source is-live=true format=time '
        f'caps=video/x-raw,format=BGR,width={width},height={height},framerate={FRAMERATE}/1 ! '
        f'videoconvert ! waylandsink sync=false'
    )

    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name('source')
    pipeline.set_state(Gst.State.PLAYING)

    number_frames = 0

    while True:
        # --- Inference (commented out for GStreamer pipeline test) ---
        image = cv2.resize(frame, (input_size, input_size))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, axis=0)

        if is_floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = (input_data.astype(np.int16) - 128).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        t1 = time()
        interpreter.invoke()
        t2 = time()

        raw = interpreter.get_tensor(output_details[0]['index'])[0]
        scale, zero_point = output_details[0]['quantization']
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
        for i in range(len(det_scores)):
            cx, cy, w, h = box_raw[i]
            left   = int((cx - w / 2) * width)
            right  = int((cx + w / 2) * width)
            top    = int((cy - h / 2) * height)
            bottom = int((cy + h / 2) * height)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {(t2 - t1):.3f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # --- End inference ---

        # Push raw frame to waylandsink via appsrc
        data = GLib.Bytes.new_take(frame.tobytes())
        buf = Gst.Buffer.new_wrapped_bytes(data)
        buf.duration = duration
        timestamp = number_frames * duration
        buf.pts = buf.dts = timestamp
        buf.offset = number_frames
        number_frames += 1
        appsrc.emit('push-buffer', buf)

        # Read next frame, loop video on end
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        # Strip any HW-decoder padding
        frame = frame[:height, :width]

    pipeline.set_state(Gst.State.NULL)
    cap.release()


if __name__ == "__main__":
    main()

