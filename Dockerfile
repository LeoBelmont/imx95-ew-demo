# Base: the image originally committed on the board (demo-imx95:gi from
# demo-imx95.tar), pushed as lbornia/demo-imx95:base.
# Build & push:
#   docker build --platform linux/arm64 -t lbornia/demo-imx95:latest . && docker push lbornia/demo-imx95:latest
FROM lbornia/demo-imx95:base

WORKDIR /app

# Only the files that differ from the base image; the rest of /app (venv,
# model, old experiments) comes from lbornia/demo-imx95:base.
COPY app/main-f.py app/entrypoint.sh /app/
COPY app/videos/didi.mp4 /app/videos/

CMD ["bash", "/app/entrypoint.sh"]
