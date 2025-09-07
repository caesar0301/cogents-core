
from huggingface_hub import snapshot_download

# Download all files from repo to a local folder
local_dir = snapshot_download("ggml-org/gemma-3-270m-it-GGUF")
print(f"Model downloaded to: {local_dir}")
