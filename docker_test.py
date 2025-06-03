import os
import sys
from pathlib import Path

import cv2  # noqa: F401
import torch

from groundingdino.util.inference import annotate, load_image, load_model, predict

# Print debug information
print("Python version:", sys.version)
print("CUDA available:", torch.cuda.is_available())
print("Current directory:", os.getcwd())
print("Directory contents:", os.listdir())

# Initialize model paths
model_path = Path("/opt/program/weights/groundingdino_swint_ogc.pth")

# Try to find the config file
try:
    import groundingdino
    groundingdino_root = Path(groundingdino.__file__).parent
    config_path = groundingdino_root / "config" / "GroundingDINO_SwinT_OGC.py"
    print(f"Discovered config path: {config_path}")
except Exception as e:
    print(f"Error finding groundingdino config path: {e}")
    # Fallback paths
    config_path = Path("/opt/conda/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# Check if paths exist
print(f"Model path: {model_path}, exists: {model_path.exists()}")
print(f"Config path: {config_path}, exists: {config_path.exists()}")

# Load model
try:
    print("Loading model...")
    model = load_model(str(config_path), str(model_path))
    print("Model loaded successfully!")
    
    # Check CUDA availability and try to move model to GPU
    if torch.cuda.is_available():
        print("Moving model to CUDA...")
        model = model.to('cuda:0')
        print("Model moved to CUDA successfully!")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print('DONE!')
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()