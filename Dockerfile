# Use Nvidia CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && df -h

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
#RUN cd /comfyui/custom_nodes/ \
#    && git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
#    && git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git \
#    && git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-Gemini.git \
#    && git clone https://github.com/shadowcz007/comfyui-mixlab-nodes.git \
#    && git clone https://github.com/Gourieff/comfyui-reactor-node.git\
#    && git clone https://github.com/jamesWalker55/comfyui-various.git \
#    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
#    && git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git \
#    && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git\
#    && git clone https://github.com/comfyanonymous/ComfyUI_experiments.git \
#    && git clone https://github.com/kinfolk0117/ComfyUI_GradientDeepShrink.git \
#    && git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID.git \
#    && cd /comfyui/custom_nodes/ComfyUI-InstantID \
#    && git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
#
#RUN cd /comfyui/custom_nodes/ComfyUI-Manager \
#    && pip3 install -r requirements.txt \
#    && cd /comfyui/custom_nodes/ComfyUI-Gemini \
#    && pip3 install -r requirements.txt \
#    && cd /comfyui/custom_nodes/comfyui-mixlab-nodes \
#    && pip3 install -r requirements.txt \
#    && cd /comfyui/custom_nodes/comfyui-reactor-node \
#    && pip3 install -r requirements.txt \
#    && cd /comfyui/custom_nodes/ComfyUI-VideoHelperSuite \
#    && pip3 install -r requirements.txt \
#    && cd /comfyui/custom_nodes/comfyui_controlnet_aux \
#    && pip3 install -r requirements.txt \
#    && cd /comfyui/custom_nodes/ComfyUI-InstantID \
#    && pip3 install -r requirements.txt \
#    &&  cd /comfyui \
#    && rm -fr /root/.cache/pip

RUN df -h

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && rm -fr /root/.cache/pip \
    && df -h \
    && du -h --max-depth=1


RUN pip3 install --no-cache-dir xformers==0.0.21 \
    && rm -fr /root/.cache/pip \
    && df -h
RUN pip3 install -r requirements.txt \
    && rm -fr /root/.cache/pip \
    && df -h \
    && du -h --max-depth=1

# Install runpod
RUN pip3 install runpod requests \
    && rm -fr /root/.cache/pip \
    && df -h \
    && du -h --max-depth=1

# Download checkpoints/vae/LoRA to include in image
#RUN wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
#RUN wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
#RUN wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors
#RUN wget -O models/loras/xl_more_art-full_v1.safetensors https://civitai.com/api/download/models/152309

# Example for adding specific models into image
# ADD models/checkpoints/sd_xl_base_1.0.safetensors models/checkpoints/
# ADD models/vae/sdxl_vae.safetensors models/vae/

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/comfy/start.sh src/comfy/rp_handler.py src/comfy/test_input.json ./
RUN chmod +x /start.sh


# Start the container
CMD /start.sh