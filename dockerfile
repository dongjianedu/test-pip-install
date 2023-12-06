ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.07-py3
FROM $BASE_IMAGE

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "GO BRRR!" \
    && apt-get update \
    && apt-get install -y  wget git  ffmpeg

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo "ls /opt/conda" && \
    ls /opt/conda && \
    echo "ls /opt/conda/bin" && \
    ls /opt/conda/bin && \
    conda init bash && \
    git clone https://github.com/magic-research/magic-animate.git && \
    cd magic-animate && \
    conda env create -f environment.yaml && \
    conda activate manimate && \
    conda list


