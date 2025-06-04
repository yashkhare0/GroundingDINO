from pathlib import Path

import cv2  # noqa: F401
import torch

from groundingdino.util.inference import load_model

# Print debug information

# Initialize model paths
model_path = Path("/opt/program/weights/groundingdino_swint_ogc.pth")

# Try to find the config file
try:
    import groundingdino

    groundingdino_root = Path(groundingdino.__file__).parent
    config_path = groundingdino_root / "config" / "GroundingDINO_SwinT_OGC.py"
except Exception:
    # Fallback paths
    config_path = Path(
        "/opt/conda/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )

# Check if paths exist

# Load model
try:
    model = load_model(str(config_path), str(model_path))

    # Check CUDA availability and try to move model to GPU
    if torch.cuda.is_available():
        model = model.to("cuda:0")

except Exception:
    import traceback

    traceback.print_exc()
