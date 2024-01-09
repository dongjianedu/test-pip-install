#!/bin/bash

cd /usr/lib/x86_64-linux-gnu/
libnvidia=$(find . -name "libnvidia-ml.so*" | grep -v "libnvidia-ml.so.1")
version=${libnvidia#*libnvidia-ml.so.}
echo $version

cp libcuda.so libcuda.so.backup
rm libcuda.so
ln -s libcuda.so.1 libcuda.so

# 建立软链接 libcuda.so.1 > libcuda.so.450.80.02
cp libcuda.so.1 libcuda.so.1.backup
rm libcuda.so.1
cp libcuda.so.$version libcuda.so.1

# 建立软链接 libnvidia-ml.so.1 > libnvidia-ml.so.450.80.02
cp libnvidia-ml.so.1 libnvidia-ml.so.1.backup
rm libnvidia-ml.so.1
ln -s libnvidia-ml.so.$version libnvidia-ml.so.1

nvidia-smi

echo "Starting WebUI API"
cd /stable-diffusion-webui
python webui.py --port 3000 --api  --listen  --no-download-sd-model &
cd /
python sam-server.py &
#python /stable-diffusion-webui/webui.py --skip-python-version-check --skip-torch-cuda-test --skip-install --ckpt /model.safetensors --lowram --opt-sdp-attention --disable-safe-unpickle --port 3000 --api --nowebui --skip-version-check  --no-hashing --no-download-sd-model &

#echo "Starting RunPod Handler"
python -u /rp_handler.py
