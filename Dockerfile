FROM timpietruskyblibla/runpod-worker-comfy:latest

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

WORKDIR /

# Set permissions and specify the command to run
RUN chmod +x /start.sh
CMD /start.sh
