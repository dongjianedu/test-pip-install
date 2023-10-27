from huggingface_hub import hf_hub_download


REPO_ID = "deejac/zhanyin"
FILENAME = "01_MorphableModel.mat"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "BFM_model_front.mat"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "Exp_Pca.bin"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")

FILENAME = "epoch_20.pth"
hf_hub_download(repo_id=REPO_ID, filename=FILENAME,local_dir="./",repo_type="model")



