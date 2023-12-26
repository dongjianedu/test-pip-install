from huggingface_hub import hf_hub_download
from transformers import CLIPModel as _CLIPModel

REPO_ID = "deejac/zhanyin"
FILENAME = "models/majicmixRealistic_v7.safetensors"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "models/control_v11f1p_sd15_depth.pth"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "models/control_v11p_sd15_openpose.pth"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "models/downloads/openpose/body_pose_model.pth"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "models/downloads/openpose/facenet.pth"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "models/downloads/openpose/hand_pose_model.pth"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "models/downloads/mlsd/mlsd_large_512_fp32.pth"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "models/downloads/midas/dpt_hybrid-midas-501f0c75.pt"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")


_CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

