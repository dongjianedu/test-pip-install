ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.07-py3
FROM $BASE_IMAGE

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "GO BRRR!" \
    && sudo df -h \
    && apt-get update \
    && apt-get install -y  wget git  ffmpeg

RUN conda init bash && \
    git clone https://github.com/magic-research/magic-animate.git && \
    cd magic-animate && \
    conda env create -f environment.yaml && \
    conda activate manimate && \
    conda list


