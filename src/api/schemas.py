"""
src/api/schemas.py
==================
Pydantic request/response models for the inference API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' or 'degraded'")
    model_loaded: bool
    model_architecture: str
    model_version: str
    service_version: str


class ModelInfoResponse(BaseModel):
    architecture: str
    num_classes: int
    classes: List[str]
    image_size: int
    test_accuracy: Optional[float] = None
    test_f1: Optional[float] = None
    mlflow_run_id: Optional[str] = None
    model_format: str


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class: 'cat' or 'dog'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    probabilities: Dict[str, float] = Field(..., description="Per-class probabilities")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    filename: Optional[str] = None
    image_size: Optional[str] = None


class BatchPredictionItem(BaseModel):
    filename: Optional[str]
    label: Optional[str]
    confidence: Optional[float]
    probabilities: Optional[Dict[str, float]]
    latency_ms: Optional[float]
    error: Optional[str]


class BatchPredictionResponse(BaseModel):
    results: List[BatchPredictionItem]
    count: int


class BatchPredictionRequest(BaseModel):
    pass  # File upload handled via form
