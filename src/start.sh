#!/bin/bash

cd /GroundingDINO
echo "reinstall GroundingDINO"
pip install -e .

# cache hug model
DIRECTORY=/root/.cache/huggingface/hub/models--deejac--zhanyin

if [ -d "$DIRECTORY" ]; then
    # 如果目录存在，执行a
    echo "Directory exists. creat link"
    cd /stable-diffusion-webui/models/ControlNet
    ln -s /root/.cache/huggingface/hub/models--deejac--zhanyin/blobs/761077ffe369fe8cf16ae353f8226bd4ca29805b161052f82c0170c7b50f1d99  ./control_v11f1p_sd15_depth.pth
    ln -s /root/.cache/huggingface/hub/models--deejac--zhanyin/blobs/db97becd92cd19aff71352a60e93c2508decba3dee64f01f686727b9b406a9dd  ./control_v11p_sd15_openpose.pth
    cd /stable-diffusion-webui/models/Stable-diffusion
    ln -s /root/.cache/huggingface/hub/models--deejac--zhanyin/blobs/7c819b6d13663ed720c2254f4fe18373107dfef2448d337913c8fc545640881e  ./majicmixRealistic_v7.safetensors
    mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose
    cd /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose
    ln -s /root/.cache/huggingface/hub/models--deejac--zhanyin/blobs/25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746  ./body_pose_model.pth
    ln -s /root/.cache/huggingface/hub/models--deejac--zhanyin/blobs/8beb52e548624ffcc4aed12af7aee7dcbfaeea420c75609fee999fe7add79d43  ./facenet.pth
    ln -s /root/.cache/huggingface/hub/models--deejac--zhanyin/blobs/b76b00d1750901abd07b9f9d8c98cc3385b8fe834a26d4b4f0aad439e75fc600  ./hand_pose_model.pth
    mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/midas
    cd /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/midas
    ln -s /root/.cache/huggingface/hub/models--deejac--zhanyin/blobs/501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853 ./dpt_hybrid-midas-501f0c75.pt
    mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/mlsd
    cd /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/mlsd
    ln -s /root/.cache/huggingface/hub/models--deejac--zhanyin/blobs/5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082 ./mlsd_large_512_fp32.pth
    cp  /workspace/detection_Resnet50_Final.pth  /stable-diffusion-webui/repositories/CodeFormer/facelib/detection_Resnet50_Final.pth
	  cp  /workspace/codeformer-v0.1.0.pth  /stable-diffusion-webui/models/Codeformer/codeformer-v0.1.0.pth
	  cp  /workspace/parsing_parsenet.pth   /stable-diffusion-webui/repositories/CodeFormer/weights/facelib/parsing_parsenet.pth
	  cp  /workspace/detection_Resnet50_Final.pth    /stable-diffusion-webui/repositories/CodeFormer/weights/facelib/detection_Resnet50_Final.pth
else
    # 如果目录不存在，执行b
    echo "Directory not  exists. download models"
    python python /fetch_models.py
    mkdir /stable-diffusion-webui/models/ControlNet
    ln -s /models/majicmixRealistic_v7.safetensors /stable-diffusion-webui/models/Stable-diffusion/majicmixRealistic_v7.safetensors
    ln -s /models/control_v11f1p_sd15_depth.pth /stable-diffusion-webui/models/ControlNet/control_v11f1p_sd15_depth.pth
    ln -s /models/control_v11p_sd15_openpose.pth /stable-diffusion-webui/models/ControlNet/control_v11p_sd15_openpose.pth
    mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose
    ln -s /models/downloads/openpose/body_pose_model.pth /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose/body_pose_model.pth
    ln -s /models/downloads/openpose/facenet.pth /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose/facenet.pth
    ln -s /models/downloads/openpose/hand_pose_model.pth /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose/hand_pose_model.pth
    mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/midas
    ln -s /models/downloads/midas/dpt_hybrid-midas-501f0c75.pt /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/midas/dpt_hybrid-midas-501f0c75.pt
    mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/mlsd
    ln -s /models/downloads/mlsd/mlsd_large_512_fp32.pth /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/mlsd/mlsd_large_512_fp32.pth
fi



echo "Worker Initiated"

echo "Starting WebUI API"
cd /stable-diffusion-webui
python webui.py --port 3000 --api  --listen  --no-download-sd-model &
sleep 100
cd /
python sam-server.py &
sleep 200
#python /stable-diffusion-webui/webui.py --skip-python-version-check --skip-torch-cuda-test --skip-install --ckpt /model.safetensors --lowram --opt-sdp-attention --disable-safe-unpickle --port 3000 --api --nowebui --skip-version-check  --no-hashing --no-download-sd-model &

#echo "Starting RunPod Handler"
python -u /rp_handler.py
