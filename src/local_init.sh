#!/bin/bash

cd /GroundingDINO
echo "reinstall GroundingDINO"
pip install -e .

pip install flask --index https://pypi.tuna.tsinghua.edu.cn/simple
# cache hug model
DIRECTORY=/workspace/models--deejac--zhanyin
echo "start copy"
cp ${DIRECTORY}/blobs/761077ffe369fe8cf16ae353f8226bd4ca29805b161052f82c0170c7b50f1d99  /stable-diffusion-webui/models/ControlNet/control_v11f1p_sd15_depth.pth
cp ${DIRECTORY}/blobs/db97becd92cd19aff71352a60e93c2508decba3dee64f01f686727b9b406a9dd  /stable-diffusion-webui/models/ControlNet/control_v11p_sd15_openpose.pth
cp ${DIRECTORY}/blobs/7c819b6d13663ed720c2254f4fe18373107dfef2448d337913c8fc545640881e  /stable-diffusion-webui/models/Stable-diffusion/majicmixRealistic_v7.safetensors
mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose
cp ${DIRECTORY}/blobs/25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746  /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose/body_pose_model.pth
cp ${DIRECTORY}/blobs/8beb52e548624ffcc4aed12af7aee7dcbfaeea420c75609fee999fe7add79d43  /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose/facenet.pth
cp ${DIRECTORY}/blobs/b76b00d1750901abd07b9f9d8c98cc3385b8fe834a26d4b4f0aad439e75fc600  /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/openpose/hand_pose_model.pth

mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/midas
cp ${DIRECTORY}/blobs/501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853  /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/midas/dpt_hybrid-midas-501f0c75.pt
mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/mlsd  
cp ${DIRECTORY}/blobs/5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082  /stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/mlsd/mlsd_large_512_fp32.pth

cp  /workspace/detection_Resnet50_Final.pth  /stable-diffusion-webui/repositories/CodeFormer/facelib/detection_Resnet50_Final.pth
cp  /workspace/codeformer-v0.1.0.pth  /stable-diffusion-webui/models/Codeformer/codeformer-v0.1.0.pth 
cp  /workspace/parsing_parsenet.pth   /stable-diffusion-webui/repositories/CodeFormer/weights/facelib/parsing_parsenet.pth
cp  /workspace/detection_Resnet50_Final.pth    /stable-diffusion-webui/repositories/CodeFormer/weights/facelib/detection_Resnet50_Final.pth

echo "end copy"


 

    


 
	



