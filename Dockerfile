# Base: the image originally committed on the board (demo-imx95:gi from
# demo-imx95.tar), pushed as lbornia/demo-imx95:base.
# Build & push:
#   docker build --platform linux/arm64 -t lbornia/demo-imx95:latest . && docker push lbornia/demo-imx95:latest
FROM lbornia/demo-imx95:base

WORKDIR /app

# The demo's own files; the rest of /app (Python venv, old experiments)
# comes from lbornia/demo-imx95:base.
COPY app/main-f.py app/entrypoint.sh /app/
COPY app/videos/sample.mp4 /app/videos/
COPY app/yolov8n_saved_model/my_model_optimized.tflite /app/yolov8n_saved_model/

CMD ["bash", "/app/entrypoint.sh"]
