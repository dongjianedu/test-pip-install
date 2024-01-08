FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

WORKDIR /


ADD src .




# Set permissions and specify the command to run
RUN chmod +x /start-gpu.sh
CMD /start-gpu.sh
