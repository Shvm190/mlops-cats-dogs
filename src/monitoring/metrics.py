"""
src/monitoring/metrics.py
==========================
Prometheus metrics for the inference service.
Tracks: request counts, prediction counts, latency histograms.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest

# ─── Metrics Definitions ─────────────────────────────────────────────────────

REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "path"],
)

PREDICTION_COUNTER = Counter(
    "predictions_total",
    "Total predictions made",
    ["label"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Inference latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
)

MODEL_LOAD_GAUGE = Gauge(
    "model_loaded",
    "Whether the model is currently loaded (1=yes, 0=no)",
)

ERROR_COUNTER = Counter(
    "prediction_errors_total",
    "Total prediction errors",
    ["error_type"],
)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def record_prediction(label: str, latency_ms: float):
    """Record a successful prediction."""
    PREDICTION_COUNTER.labels(label=label).inc()
    PREDICTION_LATENCY.observe(latency_ms)


def record_error(error_type: str = "unknown"):
    """Record a prediction error."""
    ERROR_COUNTER.labels(error_type=error_type).inc()


def set_model_loaded(loaded: bool):
    """Update model-loaded gauge."""
    MODEL_LOAD_GAUGE.set(1 if loaded else 0)


def get_metrics_output() -> str:
    """Return Prometheus text format metrics."""
    return generate_latest().decode("utf-8")
