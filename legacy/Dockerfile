FROM lbornia/npu-imx95:latest

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY yolov8s-q-neutron.tflite .
COPY labelmap.txt .
COPY requirements.txt .
COPY main.py .
COPY input.mp4 .

RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "main.py"]
