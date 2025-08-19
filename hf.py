#Code for downloading a hf model, checking size in MB and deleting later 

from huggingface_hub import snapshot_download
import os
import shutil


def download_model(model_id: str, local_dir: str = "./hf_model") -> None:
    snapshot_download(repo_id=model_id, local_dir=local_dir)
    print("Model downloaded")


def size_checker(path: str = "./hf_model") -> int:
    size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            filepath = os.path.join(dirpath, file)
            size += os.path.getsize(filepath)
    
    size_mb = size / (1024 * 1024)  

    return size_mb



model_id = "distilbert/distilgpt2"

download_model(model_id)

print(size_checker())

shutil.rmtree("./hf_model")
print("Model directory deleted")




