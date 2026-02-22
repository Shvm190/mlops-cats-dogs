"""
src/models/evaluate.py
=======================
Standalone model evaluation script.
Can be run directly or as a DVC stage.
"""

import json
import logging
from pathlib import Path
from typing import Dict

# Allow running as either `python -m src.models.evaluate` or `python src/models/evaluate.py`.
if __package__ is None or __package__ == "":
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
import click

from src.models.inference import ModelLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_model(
    model_path: str,
    test_dir: str,
    output_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate a trained model on the test set.

    Returns:
        Dictionary of evaluation metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    loader = ModelLoader(model_path=model_path)
    loader.load()
    model = loader.model.to(device)
    model.eval()

    # Load test data
    from src.data.dataset import CatsDogsDataset, get_eval_transforms
    dataset = CatsDogsDataset(
        root_dir=test_dir,
        transform=get_eval_transforms(image_size),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # dog probability for AUC

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
    precision = precision_score(y_true, y_pred, average="binary", pos_label=1)
    recall = recall_score(y_true, y_pred, average="binary", pos_label=1)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["cat", "dog"])

    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "roc_auc": round(float(auc), 4),
    }

    # Save metrics
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    with open(output_path / "confusion_matrix.json", "w") as f:
        json.dump({"matrix": cm.tolist(), "labels": ["cat", "dog"]}, f, indent=2)

    # Save classification report
    with open(output_path / "classification_report.txt", "w") as f:
        f.write(report)

    logger.info(f"\n📊 Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k:15s}: {v}")
    logger.info(f"\n{report}")
    logger.info(f"\n✅ Results saved to {output_path}")

    return metrics


@click.command()
@click.option("--model-path", required=True, help="Path to model (.pt or .pth)")
@click.option("--test-dir", default="data/processed/test", help="Test set directory")
@click.option("--output-dir", default="artifacts/evaluation", help="Output directory")
@click.option("--image-size", default=224, help="Input image size")
@click.option("--batch-size", default=32, help="Batch size")
def main(model_path, test_dir, output_dir, image_size, batch_size):
    """Evaluate trained model on test set."""
    evaluate_model(
        model_path=model_path,
        test_dir=test_dir,
        output_dir=output_dir,
        image_size=image_size,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
