# ---------------------------------------------------------------------------- #
#                         Stage 1: Download the models                         #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.36.2 as download

COPY builder/clone.sh /clone.sh

# Clone the repos and clean unnecessary files
RUN . /clone.sh taming-transformers https://github.com/CompVis/taming-transformers.git 24268930bf1dce879235a7fddd0b2355b84d7ea6 && \
    rm -rf data assets **/*.ipynb

RUN . /clone.sh stable-diffusion-stability-ai https://github.com/Stability-AI/stablediffusion.git 47b6b607fdd31875c9279cd2f4f16b92e4ea958e && \
    rm -rf assets data/**/*.png data/**/*.jpg data/**/*.gif

RUN . /clone.sh CodeFormer https://github.com/sczhou/CodeFormer.git c5b4593074ba6214284d6acd5f1719b6c5d739af && \
    rm -rf assets inputs

RUN . /clone.sh BLIP https://github.com/salesforce/BLIP.git 48211a1594f1321b00f14c9f7a5b4813144b2fb9 && \
    . /clone.sh k-diffusion https://github.com/crowsonkb/k-diffusion.git 5b3af030dd83e0297272d861c19477735d0317ec && \
    . /clone.sh clip-interrogator https://github.com/pharmapsychotic/clip-interrogator 2486589f24165c8e3b303f84e9dbbea318df83e8 && \
    . /clone.sh generative-models https://github.com/Stability-AI/generative-models 45c443b316737a4ab6e40413d7794a7f5657c19f

#RUN apk add --no-cache wget && \
#    wget  -O /model.safetensors "https://civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com/model/2253457/classicV2ByStable.JaA9.safetensors?X-Amz-Expires=86400&response-content-disposition=attachment%3B%20filename%3D%22classicBYSTABLEYOGI_v20.safetensors%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=e01358d793ad6966166af8b3064953ad/20231225/us-east-1/s3/aws4_request&X-Amz-Date=20231225T040453Z&X-Amz-SignedHeaders=host&X-Amz-Signature=58a649383098736e288cb9c81973262de13f4570c6734613223ef0308e6e1157"



# ---------------------------------------------------------------------------- #
#                        Stage 3: Build the final image                        #
# ---------------------------------------------------------------------------- #
#FROM python:3.10.9-slim as build_final_image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

WORKDIR /

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential libssl-dev libasound2  cmake -y &&\
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install python3.10-dev python3.10-venv python3-pip -y --no-install-recommends  && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

ARG SHA=5ef669de080814067961f28357256e8fe27544f4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    LD_PRELOAD=libtcmalloc.so \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN export COMMANDLINE_ARGS="--skip-torch-cuda-test --precision full --no-half"
RUN export TORCH_COMMAND='pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.6'

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync  jq moreutils aria2  libgoogle-perftools-dev procps  libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/cache --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${SHA} && \
    cd ${ROOT}/extensions && \
    git clone https://github.com/Mikubill/sd-webui-controlnet.git
#&& \ pip install -r requirements_versions.txt

COPY --from=download /repositories/ ${ROOT}/repositories/
#COPY --from=download /model.safetensors /model.safetensors
RUN mkdir ${ROOT}/interrogate && cp ${ROOT}/repositories/clip-interrogator/data/* ${ROOT}/interrogate
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${ROOT}/repositories/CodeFormer/requirements.txt

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
COPY builder/fetch_sd_models.py /fetch_models.py
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt && \
    cd / && \
    python /fetch_models.py && \
    mkdir ${ROOT}/models/ControlNet && \
    pwd && \
    ls -la && \
    ln -s /models/majicmixRealistic_v7.safetensors ${ROOT}/models/Stable-diffusion/majicmixRealistic_v7.safetensors && \
    ln -s /models/control_v11f1p_sd15_depth.pth ${ROOT}/models/ControlNet/control_v11f1p_sd15_depth.pth && \
    ln -s /models/control_v11p_sd15_openpose.pth ${ROOT}/models/ControlNet/control_v11p_sd15_openpose.pth && \
    mkdir -p ${ROOT}/extensions/sd-webui-controlnet/annotator/downloads/openpose && \
    ln -s /models/downloads/openpose/body_pose_model.pth ${ROOT}/extensions/sd-webui-controlnet/annotator/downloads/openpose/body_pose_model.pth && \
    ln -s /models/downloads/openpose/facenet.pth ${ROOT}/extensions/sd-webui-controlnet/annotator/downloads/openpose/facenet.pth && \
    ln -s /models/downloads/openpose/hand_pose_model.pth ${ROOT}/extensions/sd-webui-controlnet/annotator/downloads/openpose/hand_pose_model.pth && \
    mkdir -p ${ROOT}/extensions/sd-webui-controlnet/annotator/downloads/midas && \
    ln -s /models/downloads/midas/dpt_hybrid-midas-501f0c75.pt ${ROOT}/extensions/sd-webui-controlnet/annotator/downloads/midas/dpt_hybrid-midas-501f0c75.pt && \
    mkdir -p ${ROOT}/extensions/sd-webui-controlnet/annotator/downloads/mlsd && \
    ln -s /models/downloads/mlsd/mlsd_large_512_fp32.pth ${ROOT}/extensions/sd-webui-controlnet/annotator/downloads/mlsd/mlsd_large_512_fp32.pth



ADD src .

COPY builder/cache.py ${ROOT}/cache.py
RUN cd ${ROOT} && python cache.py --use-cpu=all --ckpt /models/majicmixRealistic_v7.safetensors

# Cleanup section (Worker Template)
RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Set permissions and specify the command to run
#RUN chmod +x /start.sh
#CMD /start.sh
