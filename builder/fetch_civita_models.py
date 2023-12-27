from huggingface_hub import hf_hub_download
from transformers import CLIPTokenizer
REPO_ID = "deejac/zhanyin"
FILENAME = "models/majicmixRealistic_v7.safetensors"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")




