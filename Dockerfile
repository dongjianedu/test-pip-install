FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

WORKDIR /

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential libssl-dev libasound2  cmake  ldconfig -y &&\
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
    rm get-pip.py && \
    pip install  git+https://github.com/runpod/runpod-python.git




ADD src .

# Set permissions and specify the command to run
RUN chmod +x /start-gpu.sh
CMD /start-gpu.sh
