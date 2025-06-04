import base64
import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("groundingdino")


# Debug information
logger.info(f"Python version: {sys.version}")
logger.info(
    f"Environment variables: { {k: v for k, v in os.environ.items() if k.startswith(('CUDA', 'TORCH', 'PATH'))} }",
)
logger.info(f"Current directory: {Path.cwd()}")
logger.info(f"Directory contents: {os.listdir()}")
if Path("/opt/program/weights").exists():
    logger.info(f"Weights directory contents: {os.listdir('/opt/program/weights')}")
else:
    logger.warning("Weights directory does not exist!")

# Import torch FIRST to ensure C extensions are loaded properly
import torch  # noqa: E402
import torchvision  # noqa: E402

logger.info(f"Torch version: {torch.__version__}")
logger.info(f"Torchvision version: {torchvision.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# Import from groundingdino-py package AFTER torch is initialized
try:
    # Import from groundingdino-py package
    import groundingdino.datasets.transforms as T  # noqa: N812
    from groundingdino.util.inference import load_model, predict

    logger.info("Successfully imported GroundingDINO")
except Exception as e:
    logger.error(f"Error importing GroundingDINO: {e}")
    sys.exit(1)

# Create log directory if it doesn't exist
Path("/var/log").mkdir(parents=True, exist_ok=True)

# Update logging configuration to include file handler
file_handler = logging.FileHandler("/var/log/groundingdino.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger.addHandler(file_handler)

# Initialize model paths - using pre-downloaded weights
MODEL_PATH = Path("/opt/program/weights/groundingdino_swint_ogc.pth")

# Try to find the config file in multiple possible locations
try:
    import groundingdino

    groundingdino_root = Path(groundingdino.__file__).parent
    CONFIG_PATH = groundingdino_root / "config" / "GroundingDINO_SwinT_OGC.py"
    logger.info(f"Discovered config path: {CONFIG_PATH}")
except Exception as e:
    logger.error(f"Error finding groundingdino config path: {e}")
    # Fallback paths
    CONFIG_PATH = Path(
        "/opt/conda/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )

# Check if config path exists, try alternative paths if not
if not CONFIG_PATH.exists():
    logger.warning(f"Config path {CONFIG_PATH} does not exist, trying alternatives")

    alternative_paths = [
        Path(
            "/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ),
        Path("/app/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        Path.cwd() / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
    ]

    for path in alternative_paths:
        logger.info(f"Trying alternative path: {path}")
        if path.exists():
            CONFIG_PATH = path
            logger.info(f"Found config at: {CONFIG_PATH}")
            break
    else:
        logger.error("Could not find config file in any location!")

# Log paths for debugging
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Config path: {CONFIG_PATH}")
logger.info(f"Model path exists: {MODEL_PATH.exists()}")
logger.info(f"Config path exists: {CONFIG_PATH.exists()}")

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
            app.state.model = load_model(
                str(CONFIG_PATH),
                str(MODEL_PATH),
                device=DEVICE,
            )
            logger.info("GroundingDINO model loaded successfully")
        except Exception as device_error:
            # If loading with CUDA fails, try CPU as fallback
            if DEVICE == "cuda":
                logger.warning(
                    f"Failed to load model with CUDA: {device_error}. Trying CPU instead.",
                )
                app.state.model = load_model(
                    str(CONFIG_PATH),
                    str(MODEL_PATH),
                    device="cpu",
                )
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
# Remove the custom ResizeForGroundingDINO class and use the official transforms
transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ],
)


def preprocess_image(image_pil: Image.Image) -> torch.Tensor:
    """
    Preprocess PIL Image for the model.

    Args:
        image_pil: PIL Image

    Returns:
        Preprocessed image tensor
    """
    # Following the exact pattern used in GroundingDINO's inference.py
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
            raise HTTPException(
                status_code=503,
                detail=f"CUDA health check failed: {e}",
            ) from e

    logger.debug("Health check passed")
    return {"status": "healthy", "device": DEVICE}


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    request: DetectionRequest,
) -> Union[DetectionResponse, HTTPException]:
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

            # GroundingDINO returns boxes in (center_x, center_y, width, height) format
            # Convert to (x_min, y_min, x_max, y_max) format
            center_x, center_y, width, height = box[0], box[1], box[2], box[3]

            x_min = center_x - width / 2
            y_min = center_y - height / 2
            x_max = center_x + width / 2
            y_max = center_y + height / 2

            response_boxes.append(
                BoundingBox(
                    x_min=float(x_min),
                    y_min=float(y_min),
                    x_max=float(x_max),
                    y_max=float(y_max),
                    score=float(logit),
                    class_name=phrase,
                ),
            )

        logger.info(f"Detection complete, found {len(response_boxes)} objects")
        return DetectionResponse(boxes=response_boxes)

    except Exception as e:
        logger.exception(f"Error during detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {e!s}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("KIZUNA_LOCATOR_GROUNDINGDINO_PORT", 8080))  # noqa: PLW1508
    logger.info(f"Starting GroundingDINO API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)  # noqa: S104
