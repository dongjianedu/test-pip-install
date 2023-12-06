ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.07-py3
FROM $BASE_IMAGE

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "GO BRRR!" \
    && apt-get update \
    && apt-get install -y  wget git zip ffmpeg \
    && git clone https://github.com/Flode-Labs/vid2densepose.git \
    && cd vid2densepose \
    && pip install -r requirements.txt \
    && pip install  opencv-python==4.5.1.48 \
    && git clone https://github.com/facebookresearch/detectron2.git \
    && python main.py


