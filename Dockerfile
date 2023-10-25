# Use specific version of nvidia cuda image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common  cmake -y &&\
    wget https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate && \
    chmod 777 Anaconda3-2022.10-Linux-x86_64.sh && \
    ENV CONDA_DIR /usr/share/anaconda3 && \
    bash Anaconda3-2022.10-Linux-x86_64.sh -b -p /usr/share/anaconda3 && \
    rm Anaconda3-2022.10-Linux-x86_64.sh && \
    NV PATH=$CONDA_DIR/bin:$PATH &&\
    conda init bash && \
    conda update -n base -c defaults conda && \
    conda create -n geneface python=3.9.16 -y && \
    conda activate geneface && \
    conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath && \
    conda install -c bottler nvidiacub -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* 



# Add the deadsnakes PPA and install Python 3.10
#RUN add-apt-repository ppa:deadsnakes/ppa -y && \
#    apt-get install python3.10-dev python3.10-venv python3-pip -y --no-install-recommends  && \
#    ln -s /usr/bin/python3.10 /usr/bin/python && \
#    rm /usr/bin/python3 && \
#    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
#    apt-get autoremove -y && \
#    apt-get clean -y && \
#    rm -rf /var/lib/apt/lists/*

# Download and install pip
#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#    python get-pip.py && \
#    rm get-pip.py

# Install Python dependencies (Worker Template)
#COPY builder/requirements.txt /requirements.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install --upgrade pip && \
#    pip install -r /requirements.txt --no-cache-dir   && \
#    rm /requirements.txt


# Copy source code into image
COPY src .

# Set default command
CMD python -u /main.py
