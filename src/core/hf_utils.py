from huggingface_hub import HfApi,snapshot_download
from loguru import logger
import os
import time

api = HfApi()

def download_lora_config(repo_id: str, revision: str) -> bool:
    try:
        api.hf_hub_download(
            repo_id=repo_id,
            filename="adapter_config.json",
            local_dir="lora",
            revision=revision,
        )
    except Exception as e:
        if "adapter_config.json" in str(e):
            logger.info("No adapter_config.json found in the repo, assuming full model")
            return False
        else:
            raise  # Re-raise the exception if it's not related to the missing file
    return True


def download_lora_repo(repo_id: str, revision: str) -> None:
    os.makedirs("lora", exist_ok=True)
    for attempt in range(6):
        try:
            print(f"download the adapter weights(attempt:{attempt + 1})...")
            snapshot_download(
                repo_id=repo_id,
                local_dir="lora",
                revision=revision,
                allow_patterns=["*.bin", "*.safetensors", "*.json"],
                max_workers=4
            )
            break
        except Exception as e:
            print(f"download adapter weights failed : {e}")
            time.sleep(3)