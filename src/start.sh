#!/bin/bash

cd /GroundingDINO
echo "reinstall GroundingDINO"
pip install -e .

echo "Worker Initiated"

echo "Starting WebUI API"
cd /stable-diffusion-webui
python webui.py --port 3000 --api  --listen  --no-download-sd-model &
cd /
python sam-server.py &
#python /stable-diffusion-webui/webui.py --skip-python-version-check --skip-torch-cuda-test --skip-install --ckpt /model.safetensors --lowram --opt-sdp-attention --disable-safe-unpickle --port 3000 --api --nowebui --skip-version-check  --no-hashing --no-download-sd-model &

#echo "Starting RunPod Handler"
python -u /rp_handler.py
