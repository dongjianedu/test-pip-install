# Use specific version of nvidia cuda image
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /
# Update and upgrade the system packages (Worker Template)
RUN df -h

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
COPY builder/requirements.txt /requirements.txt
RUN echo "ls /opt/conda" && \
    ls /opt/conda && \
    echo "ls /opt/conda/bin" && \
    ls /opt/conda/bin && \
    conda init bash && \
    git clone https://github.com/magic-research/magic-animate.git && \
    cd magic-animate && \
    conda env create -f environment.yaml && \
    conda activate manimate && \
    conda list && \
    df -h

