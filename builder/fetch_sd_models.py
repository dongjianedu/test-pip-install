from huggingface_hub import hf_hub_download


REPO_ID = "deejac/zhanyin/model"
FILENAME = "majicmixRealistic_v7.safetensors"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "control_v11f1p_sd15_depth.pth"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "control_v11p_sd15_openpose.pth"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")



