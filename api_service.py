import base64
import io
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Import from groundingdino-py package
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T  # noqa: N812

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/log/groundingdino.log"),
    ],
)
logger = logging.getLogger("groundingdino")

# Initialize model paths - using pre-downloaded weights
MODEL_PATH = Path("/opt/program/weights/groundingdino_swint_ogc.pth")
CONFIG_PATH = Path("/opt/conda/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# Determine device - with graceful CUDA detection
try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    else:
        DEVICE = "cpu"
        logger.warning("CUDA not available, using CPU")
except Exception as e:
    logger.warning(f"Error checking CUDA availability: {e}. Falling back to CPU.")
    DEVICE = "cpu"

logger.info(f"Using device: {DEVICE}")

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Initialize and manage the FastAPI application lifecycle.

    Args:
        app (FastAPI): The FastAPI application instance to manage.

    Raises:
        FileNotFoundError: If the model weights or config files are not found.
    """
    # Load model on startup
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        if not MODEL_PATH.exists():
            error_msg = f"Model weights file not found: {MODEL_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not CONFIG_PATH.exists():
            error_msg = f"Model config file not found: {CONFIG_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            # Try to load the model with the selected device
            logger.info(f"Loading model with {DEVICE}")
            app.state.model = load_model(str(CONFIG_PATH), str(MODEL_PATH), device=DEVICE)
            logger.info("GroundingDINO model loaded successfully")
        except Exception as device_error:
            # If loading with CUDA fails, try CPU as fallback
            if DEVICE == "cuda":
                logger.warning(f"Failed to load model with CUDA: {device_error}. Trying CPU instead.")
                app.state.model = load_model(str(CONFIG_PATH), str(MODEL_PATH), device="cpu")
                logger.info("GroundingDINO model loaded successfully with CPU")
            else:
                raise device_error
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        logger.error(f"Model path exists: {MODEL_PATH.exists()}")
        logger.error(f"Config path exists: {CONFIG_PATH.exists()}")
        raise e
    yield
    # Clean up resources when shutting down
    logger.info("Shutting down API service")

app = FastAPI(title="GroundingDINO API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionRequest(BaseModel):
    """Request model for object detection."""

    image: str  # base64 encoded image
    text_prompt: str
    box_threshold: float = 0.35
    text_threshold: float = 0.25

class BoundingBox(BaseModel):
    """Bounding box model for GroundingDINO results."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float
    class_name: str

class DetectionResponse(BaseModel):
    """Response model for object detection."""

    boxes: List[BoundingBox]

# Transform for preprocessing images
transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def preprocess_image(image_pil: Image.Image) -> torch.Tensor:
    """
    Preprocess PIL Image for the model.
    
    Args:
        image_pil: PIL Image
        
    Returns:
        Preprocessed image tensor
    """
    image_transformed, _ = transform(image_pil, None)
    return image_transformed

@app.get("/health")
async def health_check() -> dict:
    """Check the health of the API service."""
    if not hasattr(app.state, "model") or app.state.model is None:
        logger.error("Health check failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # If CUDA is being used, check CUDA health
    if DEVICE == "cuda":
        try:
            # Simple CUDA operation to verify GPU access
            x = torch.zeros(1).cuda()
            del x
            logger.debug("CUDA health check passed")
        except Exception as e:
            logger.error(f"CUDA health check failed: {e}")
            raise HTTPException(status_code=503, detail=f"CUDA health check failed: {e}")
    
    logger.debug("Health check passed")
    return {"status": "healthy", "device": DEVICE}

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest) -> Union[DetectionResponse, HTTPException]:
    """Detect objects in an image based on a text prompt."""
    if not hasattr(app.state, "model") or app.state.model is None:
        logger.error("Detection request failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Processing detection request with prompt: {request.text_prompt}")
        
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.debug(f"Image size: {image_pil.size}")
        
        # Prepare image for model
        image_tensor = preprocess_image(image_pil)
        
        # Perform prediction
        logger.info("Running model prediction")
        try:
            boxes, logits, phrases = predict(
                model=app.state.model,
                image=image_tensor,
                caption=request.text_prompt,
                box_threshold=request.box_threshold,
                text_threshold=request.text_threshold,
                device=DEVICE,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("CUDA out of memory, falling back to CPU")
                boxes, logits, phrases = predict(
                    model=app.state.model,
                    image=image_tensor,
                    caption=request.text_prompt,
                    box_threshold=request.box_threshold,
                    text_threshold=request.text_threshold,
                    device="cpu",
                )
            else:
                raise e
        
        # Convert to response format
        response_boxes = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            logit = logits[i]
            phrase = phrases[i]
            
            response_boxes.append(
                BoundingBox(
                    x_min=float(box[0]),
                    y_min=float(box[1]),
                    x_max=float(box[2]),
                    y_max=float(box[3]),
                    score=float(logit),
                    class_name=phrase,
                ),
            )
        
        logger.info(f"Detection complete, found {len(response_boxes)} objects")
        return DetectionResponse(boxes=response_boxes)
    
    except Exception as e:
        logger.exception(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e!s}") from e

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting GroundingDINO API service")
    uvicorn.run(app, host="0.0.0.0", port=8080)  # noqa: S104
