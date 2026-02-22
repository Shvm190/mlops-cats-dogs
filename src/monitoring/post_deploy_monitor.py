"""
src/monitoring/post_deploy_monitor.py
=====================================
Post-deployment model performance monitoring.
- Collects predictions on a labeled batch (real or simulated)
- Computes accuracy, F1, confidence distribution
- Logs metrics to MLflow for tracking
- Flags potential data drift
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Default Config ───────────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8080"
CONFIDENCE_THRESHOLD = 0.7    # Flag predictions below this
ACCURACY_THRESHOLD = 0.85     # Alert if accuracy drops below this


# ─── Simulated Evaluation Batch ──────────────────────────────────────────────

def generate_simulated_batch(
    test_dir: str = "data/processed/test",
    max_samples: int = 100,
) -> List[Dict]:
    """
    Build a labeled batch from the test set for post-deploy evaluation.

    Returns:
        List of dicts: {'image_path': ..., 'true_label': ...}
    """
    batch = []
    test_path = Path(test_dir)

    for cls in ["cat", "dog"]:
        cls_dir = test_path / cls
        if not cls_dir.exists():
            logger.warning(f"Test directory not found: {cls_dir}")
            continue

        images = list(cls_dir.glob("*.jpg"))[:max_samples // 2]
        for img_path in images:
            batch.append({"image_path": str(img_path), "true_label": cls})

    np.random.shuffle(batch)
    logger.info(f"Generated evaluation batch: {len(batch)} samples")
    return batch


# ─── Prediction Collector ─────────────────────────────────────────────────────

def run_predictions_on_batch(
    batch: List[Dict],
    api_url: str = API_BASE_URL,
    timeout: int = 10,
) -> List[Dict]:
    """
    Send each sample to the /predict endpoint and collect results.

    Args:
        batch: List of {'image_path': ..., 'true_label': ...}.
        api_url: Base URL of the inference service.
        timeout: Per-request timeout in seconds.

    Returns:
        List of result dicts with true_label and predicted_label.
    """
    results = []
    predict_url = f"{api_url}/predict"

    for i, sample in enumerate(batch):
        image_path = Path(sample["image_path"])
        true_label = sample["true_label"]

        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    predict_url,
                    files={"file": (image_path.name, f, "image/jpeg")},
                    timeout=timeout,
                )
            response.raise_for_status()
            pred = response.json()

            results.append({
                "true_label": true_label,
                "predicted_label": pred["label"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"],
                "latency_ms": pred["latency_ms"],
                "correct": pred["label"] == true_label,
            })
        except Exception as e:
            logger.warning(f"Request failed for {image_path.name}: {e}")
            results.append({
                "true_label": true_label,
                "predicted_label": None,
                "confidence": None,
                "probabilities": None,
                "latency_ms": None,
                "correct": False,
                "error": str(e),
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(batch)} samples...")

    return results


# ─── Metrics Computation ─────────────────────────────────────────────────────

def compute_performance_metrics(results: List[Dict]) -> Dict:
    """
    Compute accuracy, F1, confidence stats from batch results.

    Returns:
        Metrics dictionary.
    """
    valid = [r for r in results if r["predicted_label"] is not None]
    errors = len(results) - len(valid)

    if not valid:
        return {"error": "No valid predictions to evaluate"}

    true_labels = [r["true_label"] for r in valid]
    pred_labels = [r["predicted_label"] for r in valid]
    confidences = [r["confidence"] for r in valid if r["confidence"] is not None]
    latencies = [r["latency_ms"] for r in valid if r["latency_ms"] is not None]

    label_map = {"cat": 0, "dog": 1}
    y_true = [label_map[l] for l in true_labels]
    y_pred = [label_map[l] for l in pred_labels]

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
    report = classification_report(y_true, y_pred, target_names=["cat", "dog"])

    low_confidence = [r for r in valid if r["confidence"] < CONFIDENCE_THRESHOLD]
    pct_low_confidence = len(low_confidence) / len(valid) * 100

    metrics = {
        "total_samples": len(results),
        "valid_predictions": len(valid),
        "errors": errors,
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "std_confidence": round(float(np.std(confidences)), 4),
        "min_confidence": round(float(np.min(confidences)), 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "pct_low_confidence": round(pct_low_confidence, 2),
        "classification_report": report,
        "alerts": [],
    }

    # Drift alerts
    if accuracy < ACCURACY_THRESHOLD:
        alert = f"⚠️  Accuracy {accuracy:.3f} below threshold {ACCURACY_THRESHOLD}"
        metrics["alerts"].append(alert)
        logger.warning(alert)

    if pct_low_confidence > 20:
        alert = f"⚠️  {pct_low_confidence:.1f}% low-confidence predictions (>{CONFIDENCE_THRESHOLD:.0%} threshold)"
        metrics["alerts"].append(alert)
        logger.warning(alert)

    return metrics


# ─── Main Monitoring Run ──────────────────────────────────────────────────────

def run_monitoring(
    api_url: str = API_BASE_URL,
    test_dir: str = "data/processed/test",
    max_samples: int = 100,
    output_path: str = "artifacts/monitoring_report.json",
    log_to_mlflow: bool = False,
):
    """Run full post-deployment monitoring cycle."""
    logger.info("=" * 60)
    logger.info("Starting post-deployment monitoring run")
    logger.info(f"  API URL    : {api_url}")
    logger.info(f"  Max samples: {max_samples}")
    logger.info("=" * 60)

    # Check API health first
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        health.raise_for_status()
        health_data = health.json()
        logger.info(f"API health: {health_data['status']}")
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return None

    # Generate batch
    batch = generate_simulated_batch(test_dir=test_dir, max_samples=max_samples)
    if not batch:
        logger.error("No samples to evaluate")
        return None

    # Run predictions
    logger.info(f"Running predictions on {len(batch)} samples...")
    results = run_predictions_on_batch(batch, api_url=api_url)

    # Compute metrics
    metrics = compute_performance_metrics(results)

    logger.info("\n📊 Monitoring Results:")
    logger.info(f"  Accuracy         : {metrics.get('accuracy', 'N/A')}")
    logger.info(f"  F1 Score         : {metrics.get('f1_score', 'N/A')}")
    logger.info(f"  Mean Confidence  : {metrics.get('mean_confidence', 'N/A')}")
    logger.info(f"  Mean Latency     : {metrics.get('mean_latency_ms', 'N/A')} ms")
    logger.info(f"  P95 Latency      : {metrics.get('p95_latency_ms', 'N/A')} ms")
    logger.info(f"  Low Confidence % : {metrics.get('pct_low_confidence', 'N/A')}%")
    if metrics.get("alerts"):
        logger.warning(f"  ALERTS: {metrics['alerts']}")
    logger.info(f"\n{metrics.get('classification_report', '')}")

    # Save report
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        report_data = {k: v for k, v in metrics.items() if k != "classification_report"}
        json.dump(report_data, f, indent=2)
    logger.info(f"Report saved: {output}")

    # Optional: log to MLflow
    if log_to_mlflow:
        try:
            import mlflow
            with mlflow.start_run(run_name="post_deploy_monitoring"):
                mlflow.log_metrics({
                    "monitor_accuracy": metrics["accuracy"],
                    "monitor_f1": metrics["f1_score"],
                    "monitor_mean_confidence": metrics["mean_confidence"],
                    "monitor_mean_latency_ms": metrics["mean_latency_ms"],
                    "monitor_pct_low_confidence": metrics["pct_low_confidence"],
                })
                mlflow.log_artifact(str(output))
            logger.info("Metrics logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Post-deployment model monitoring")
    parser.add_argument("--api-url", default=API_BASE_URL)
    parser.add_argument("--test-dir", default="data/processed/test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", default="artifacts/monitoring_report.json")
    parser.add_argument("--mlflow", action="store_true", help="Log to MLflow")
    args = parser.parse_args()

    run_monitoring(
        api_url=args.api_url,
        test_dir=args.test_dir,
        max_samples=args.max_samples,
        output_path=args.output,
        log_to_mlflow=args.mlflow,
    )
