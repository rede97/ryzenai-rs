from pathlib import Path

from huggingface_hub import hf_hub_download

# Download Yolov8 model from Ryzen AI model zoo. Registration is required before download.
hf_hub_download(repo_id="amd/yolov8m", filename="yolov8m.onnx",
                local_dir=str(Path.cwd()))
