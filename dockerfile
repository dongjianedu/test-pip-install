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

