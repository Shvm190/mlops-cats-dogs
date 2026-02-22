"""
src/api/main.py
===============
FastAPI inference service for Cats vs Dogs classifier.

Endpoints:
  GET  /health         → Health check + model status
  GET  /ready          → Readiness check (model must be loaded)
  POST /predict        → Predict from image upload
  GET  /metrics        → Prometheus metrics
  GET  /model/info     → Model metadata
  POST /predict/batch  → Batch prediction (up to 10 images)
"""

import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from PIL import Image

from src.api.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.models.inference import ModelLoader, predict_from_bytes
from src.monitoring.metrics import (
    REQUEST_COUNTER,
    get_metrics_output,
    record_error,
    record_prediction,
    set_model_loaded,
)
from src.monitoring.logger import (
    configure_logging,
    get_logger,
    log_request,
    log_prediction,
)

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/models/model_torchscript.pt")
METADATA_PATH = os.getenv("METADATA_PATH", "artifacts/models/model_metadata.json")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

configure_logging(level=LOG_LEVEL, json_logs=False)
logger = get_logger(__name__)

# ─── App Lifecycle ───────────────────────────────────────────────────────────

model_loader: ModelLoader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model_loader
    logger.info("Starting inference service...")
    try:
        model_loader = ModelLoader.get_instance(MODEL_PATH, METADATA_PATH)
        model_loader.load()
        set_model_loaded(True)
        logger.info("Model loaded successfully ✓")
    except FileNotFoundError as e:
        set_model_loaded(False)
        logger.warning(f"Model not found at startup: {e}. Will retry on first request.")
    except Exception as e:
        set_model_loaded(False)
        logger.error(f"Failed to load model: {e}")
    yield
    logger.info("Shutting down inference service.")


# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Binary image classification API for pet adoption platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_ready() -> bool:
    return model_loader is not None and model_loader.is_loaded


def _model_meta() -> dict:
    return model_loader.metadata if _model_ready() else {}


def _ensure_model_loaded():
    global model_loader
    if _model_ready():
        return

    try:
        model_loader = ModelLoader.get_instance(MODEL_PATH, METADATA_PATH)
        model_loader.load()
        set_model_loaded(True)
    except Exception as e:
        set_model_loaded(False)
        record_error("model_unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {str(e)}",
        )


async def _read_and_validate_image(file: UploadFile) -> tuple[bytes, int, int]:
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {content_type}. Expected image/*",
        )

    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {size_mb:.1f}MB (max {MAX_UPLOAD_SIZE_MB}MB)",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupt image file",
        )

    return image_bytes, width, height


# ─── Middleware: Request Logging ──────────────────────────────────────────────


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with timing."""
    start = time.perf_counter()
    REQUEST_COUNTER.labels(method=request.method, path=request.url.path).inc()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    log_request(
        method=request.method,
        path=str(request.url.path),
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns service status and model readiness.
    """
    global model_loader

    model_ready = _model_ready()
    model_meta = _model_meta()

    return HealthResponse(
        status="healthy" if model_ready else "degraded",
        model_loaded=model_ready,
        model_architecture=model_meta.get("architecture", "unknown"),
        model_version=model_meta.get("run_id", "unknown"),
        service_version="1.0.0",
    )


@app.get("/ready", tags=["System"])
async def readiness_check():
    """Readiness endpoint used by probes and load balancers."""
    if not _model_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return {"status": "ready"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return model metadata and configuration."""
    global model_loader

    if not _model_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    meta = _model_meta()
    return ModelInfoResponse(
        architecture=meta.get("architecture", "unknown"),
        num_classes=meta.get("num_classes", 2),
        classes=meta.get("classes", ["cat", "dog"]),
        image_size=meta.get("image_size", 224),
        test_accuracy=meta.get("test_acc"),
        test_f1=meta.get("test_f1"),
        mlflow_run_id=meta.get("run_id"),
        model_format=meta.get("model_format", "torchscript"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP)")
):
    """
    Classify an uploaded image as 'cat' or 'dog'.

    Returns:
        - **label**: Predicted class ('cat' or 'dog')
        - **confidence**: Model confidence for predicted class (0-1)
        - **probabilities**: Per-class probability scores
        - **latency_ms**: Inference latency in milliseconds
    """
    image_bytes, img_width, img_height = await _read_and_validate_image(file)
    _ensure_model_loaded()

    # Run inference
    try:
        result = predict_from_bytes(model_loader, image_bytes)
    except Exception as e:
        record_error("inference_failure")
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed",
        )

    # Record metrics
    record_prediction(label=result["label"], latency_ms=result["latency_ms"])

    log_prediction(
        filename=file.filename,
        label=result["label"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
    )

    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        latency_ms=result["latency_ms"],
        filename=file.filename,
        image_size=f"{img_width}x{img_height}",
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    files: list[UploadFile] = File(..., description="Up to 10 image files")
):
    """
    Classify multiple images in one request (max 10).
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files: {len(files)} (max {MAX_BATCH_SIZE})",
        )

    _ensure_model_loaded()

    results = []
    for file in files:
        try:
            image_bytes, _, _ = await _read_and_validate_image(file)
            result = predict_from_bytes(model_loader, image_bytes)
            record_prediction(label=result["label"], latency_ms=result["latency_ms"])
            log_prediction(
                filename=file.filename,
                label=result["label"],
                confidence=result["confidence"],
                latency_ms=result["latency_ms"],
            )
            result["filename"] = file.filename
            result["error"] = None
            results.append(result)
        except HTTPException as e:
            record_error(f"http_{e.status_code}")
            results.append(
                {
                    "filename": file.filename,
                    "label": None,
                    "confidence": None,
                    "probabilities": None,
                    "latency_ms": None,
                    "error": e.detail,
                }
            )
        except Exception as e:
            record_error("batch_inference_failure")
            results.append(
                {
                    "filename": file.filename,
                    "label": None,
                    "confidence": None,
                    "probabilities": None,
                    "latency_ms": None,
                    "error": str(e),
                }
            )

    return BatchPredictionResponse(results=results, count=len(results))


@app.get("/metrics", response_class=PlainTextResponse, tags=["System"])
async def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics_output()


@app.get("/", tags=["System"])
async def root():
    """Redirect to docs."""
    return {
        "message": "Cats vs Dogs Classifier API",
        "docs": "/docs",
        "health": "/health",
    }
