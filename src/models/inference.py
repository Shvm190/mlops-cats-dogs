"""
src/models/inference.py
========================
Inference utilities: model loading, image preprocessing, prediction.
Used by the FastAPI inference service.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import get_inference_transforms

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "artifacts/models/model_torchscript.pt"
DEFAULT_METADATA_PATH = "artifacts/models/model_metadata.json"
IMAGE_SIZE = 224
CLASS_NAMES = ["cat", "dog"]


# ─── Model Loader ────────────────────────────────────────────────────────────

class ModelLoader:
    """
    Singleton-style model loader that caches the model in memory.
    Supports TorchScript (.pt) and state-dict (.pth) formats.
    """

    _instance: Optional["ModelLoader"] = None

    def __init__(self, model_path: str, metadata_path: str = None):
        self.model_path = Path(model_path)
        if metadata_path:
            self.metadata_path = Path(metadata_path)
        else:
            # Prefer colocated metadata when only a model path is supplied.
            self.metadata_path = self.model_path.with_name("model_metadata.json")
        self.model = None
        self.metadata: Dict = {}
        self.device = self._get_device()
        self.transform = get_inference_transforms(IMAGE_SIZE)
        self._loaded = False

    @classmethod
    def get_instance(
        cls,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
    ) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = cls(model_path, metadata_path)
        return cls._instance

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_metadata(self):
        if self.metadata_path and self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
            return

        fallback_path = Path(DEFAULT_METADATA_PATH)
        if fallback_path.exists() and fallback_path != self.metadata_path:
            with open(fallback_path) as f:
                self.metadata = json.load(f)
            return

        self.metadata = {}

    @staticmethod
    def _extract_state_dict(state_obj: object) -> Dict[str, torch.Tensor]:
        """Handle both raw state dicts and training checkpoint wrappers."""
        if isinstance(state_obj, dict) and "model_state_dict" in state_obj:
            model_state = state_obj["model_state_dict"]
            if isinstance(model_state, dict):
                return model_state
            raise ValueError("Invalid checkpoint format: 'model_state_dict' is not a dict")

        if isinstance(state_obj, dict) and "state_dict" in state_obj:
            model_state = state_obj["state_dict"]
            if isinstance(model_state, dict):
                return model_state
            raise ValueError("Invalid checkpoint format: 'state_dict' is not a dict")

        if isinstance(state_obj, dict):
            return state_obj

        raise ValueError("Invalid checkpoint format: expected a state_dict-like dictionary")

    def load(self) -> "ModelLoader":
        """Load model from disk into memory."""
        if self._loaded:
            return self

        logger.info(f"Loading model from: {self.model_path}")
        start = time.time()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Metadata is needed before state-dict loading to recover architecture.
        self._load_metadata()

        suffix = self.model_path.suffix.lower()
        if suffix == ".pt":
            # TorchScript model
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        elif suffix == ".pth":
            # State dict — requires architecture info from metadata
            from src.models.architecture import build_model

            arch = self.metadata.get("architecture", "mobilenet_v2")
            num_classes = int(self.metadata.get("num_classes", 2))
            dropout = float(self.metadata.get("dropout", 0.3))
            self.model = build_model(
                architecture=arch,
                num_classes=num_classes,
                dropout=dropout,
                pretrained=False,
            )
            raw_state = torch.load(str(self.model_path), map_location=self.device)
            state = self._extract_state_dict(raw_state)
            self.model.load_state_dict(state)
            self.model = self.model.to(self.device)
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        self.model.eval()

        elapsed = time.time() - start
        self._loaded = True
        logger.info(f"Model loaded in {elapsed:.2f}s on {self.device}")
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference.

    Args:
        image: Input PIL Image (any mode/size).
        image_size: Target square size.

    Returns:
        Tensor of shape (1, 3, image_size, image_size).
    """
    transform = get_inference_transforms(image_size)
    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb)
    return tensor.unsqueeze(0)  # Add batch dimension


def preprocess_image_bytes(image_bytes: bytes, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Preprocess raw image bytes for model inference.

    Args:
        image_bytes: Raw image file bytes (JPEG, PNG, etc.).
        image_size: Target square size.

    Returns:
        Tensor of shape (1, 3, image_size, image_size).
    """
    import io
    image = Image.open(io.BytesIO(image_bytes))
    return preprocess_image(image, image_size)


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """
    Run inference and return structured prediction result.

    Args:
        model: Loaded PyTorch model (eval mode).
        tensor: Preprocessed image tensor (1, C, H, W).
        device: Device to run inference on.
        class_names: List of class name strings.

    Returns:
        Dict with 'label', 'confidence', 'probabilities', 'latency_ms'.
    """
    start = time.perf_counter()

    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    latency_ms = (time.perf_counter() - start) * 1000

    probs_np = probs.squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs_np))
    pred_label = class_names[pred_idx]
    confidence = float(probs_np[pred_idx])

    return {
        "label": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(class_names, probs_np)},
        "latency_ms": round(latency_ms, 2),
        "predicted_class_index": pred_idx,
    }


def predict_from_bytes(
    model_loader: ModelLoader,
    image_bytes: bytes,
    image_size: int = IMAGE_SIZE,
) -> Dict:
    """
    End-to-end prediction from raw image bytes.

    Args:
        model_loader: Loaded ModelLoader instance.
        image_bytes: Raw image bytes.
        image_size: Target image size.

    Returns:
        Prediction result dict.
    """
    if not model_loader.is_loaded:
        model_loader.load()

    tensor = preprocess_image_bytes(image_bytes, image_size)
    result = predict(model_loader.model, tensor, model_loader.device)
    return result
