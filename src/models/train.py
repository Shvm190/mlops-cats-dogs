"""
src/models/train.py
====================
Training script for Cats vs Dogs classifier.
Logs all experiments, metrics, and artifacts to MLflow.

Usage:
    python -m src.models.train --config configs/train_config.yaml
    python -m src.models.train --architecture simple_cnn --epochs 10
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

# Allow running as either `python -m src.models.train` or `python src/models/train.py`.
if __package__ is None or __package__ == "":
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import yaml
import click
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from src.data.dataset import get_dataloaders
from src.models.architecture import build_model, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Training Utilities ───────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def get_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("learning_rate", 1e-3)
    wd = cfg.get("weight_decay", 1e-4)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt_name == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer, cfg: dict):
    sched_name = cfg.get("scheduler", "cosine").lower()
    params = cfg.get("scheduler_params", {})

    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=params.get("T_max", 20))
    elif sched_name == "step":
        return StepLR(optimizer, step_size=params.get("step_size", 7), gamma=params.get("gamma", 0.1))
    elif sched_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    return None


def configure_mlflow_tracking(mlflow_uri: str) -> str:
    """
    Configure MLflow tracking with graceful fallback.
    If a remote URI is unreachable, fallback to local './mlruns'.
    """
    target_uri = mlflow_uri or "mlruns"
    mlflow.set_tracking_uri(target_uri)

    if target_uri.startswith(("http://", "https://")):
        try:
            MlflowClient().search_experiments(max_results=1)
        except Exception as e:
            fallback_uri = "mlruns"
            logger.warning(
                "MLflow server '%s' is unreachable (%s). Falling back to local '%s'.",
                target_uri,
                e,
                fallback_uri,
            )
            mlflow.set_tracking_uri(fallback_uri)
            return fallback_uri

    return target_uri


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-3, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.triggered = False

    def step(self, metric: float) -> bool:
        improved = (
            metric > self.best + self.min_delta if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ─── Epoch Functions ─────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool = True,
    grad_clip: float = 1.0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run one epoch (train or eval).

    Returns:
        (avg_loss, accuracy, all_preds, all_labels)
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(cfg: dict):
    """Full training pipeline with MLflow tracking."""

    # ── Setup ──
    device = get_device()
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    artifact_cfg = cfg.get("artifacts", {})
    project_cfg = cfg.get("project", {})

    # ── MLflow ──
    mlflow_uri = project_cfg.get("mlflow_tracking_uri", "mlruns")
    experiment_name = project_cfg.get("mlflow_experiment", "cats-dogs")
    effective_mlflow_uri = configure_mlflow_tracking(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {effective_mlflow_uri}")

    # ── Data ──
    loaders = get_dataloaders(
        processed_dir=data_cfg.get("processed_dir", "data/processed"),
        image_size=data_cfg.get("image_size", 224),
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=4,
    )
    required_splits = {"train", "val", "test"}
    missing_splits = sorted(required_splits - set(loaders.keys()))
    if missing_splits:
        processed_dir = data_cfg.get("processed_dir", "data/processed")
        raise ValueError(
            f"Missing required data splits: {missing_splits}. "
            f"Expected directories under '{processed_dir}'."
        )

    logger.info(f"Train: {len(loaders['train'].dataset)} samples")
    logger.info(f"Val  : {len(loaders['val'].dataset)} samples")

    # ── Model ──
    model = build_model(
        architecture=model_cfg.get("architecture", "mobilenet_v2"),
        num_classes=model_cfg.get("num_classes", 2),
        dropout=model_cfg.get("dropout", 0.3),
        pretrained=model_cfg.get("pretrained", True),
    ).to(device)

    param_counts = count_parameters(model)
    logger.info(f"Parameters: {param_counts}")

    # Freeze backbone initially for transfer learning
    freeze_epochs = model_cfg.get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        logger.info(f"Backbone frozen for first {freeze_epochs} epochs")

    # ── Training Infra ──
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, train_cfg)
    scheduler = get_scheduler(optimizer, train_cfg)
    early_stop_cfg = train_cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=early_stop_cfg.get("patience", 5),
        min_delta=early_stop_cfg.get("min_delta", 1e-3),
    ) if early_stop_cfg.get("enabled", True) else None

    # ── Artifact Directories ──
    checkpoint_dir = Path(artifact_cfg.get("checkpoint_dir", "artifacts/checkpoints"))
    model_dir = Path(artifact_cfg.get("model_dir", "artifacts/models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow Run ──
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_params({
            "architecture": model_cfg.get("architecture"),
            "pretrained": model_cfg.get("pretrained"),
            "epochs": train_cfg.get("epochs"),
            "batch_size": train_cfg.get("batch_size"),
            "learning_rate": train_cfg.get("learning_rate"),
            "optimizer": train_cfg.get("optimizer"),
            "scheduler": train_cfg.get("scheduler"),
            "image_size": data_cfg.get("image_size"),
            "dropout": model_cfg.get("dropout"),
            "total_params": param_counts["total"],
            "trainable_params": param_counts["trainable"],
            "mlflow_tracking_uri": effective_mlflow_uri,
        })

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = float("-inf")
        best_model_path = checkpoint_dir / "best_model.pth"

        epochs = train_cfg.get("epochs", 20)
        grad_clip = train_cfg.get("gradient_clip", 1.0)

        for epoch in range(1, epochs + 1):
            t_start = time.time()

            # Unfreeze backbone after N epochs
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                if hasattr(model, "unfreeze_backbone"):
                    model.unfreeze_backbone()
                    optimizer = get_optimizer(model, train_cfg)
                    logger.info(f"Epoch {epoch}: Backbone unfrozen, optimizer reset")

            # Train
            train_loss, train_acc, _, _ = run_epoch(
                model, loaders["train"], criterion, optimizer, device,
                is_train=True, grad_clip=grad_clip
            )

            # Validate
            val_loss, val_acc, val_preds, val_labels = run_epoch(
                model, loaders["val"], criterion, None, device, is_train=False
            )

            # Scheduler step
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            elapsed = time.time() - t_start
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"LR={current_lr:.2e} | {elapsed:.1f}s"
            )

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            }, step=epoch)

            # Checkpoint best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": cfg,
                }, best_model_path)
                logger.info(f"  ✓ New best val_acc={val_acc:.4f} → saved checkpoint")

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Early stopping
            if early_stopper and early_stopper.step(val_acc):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # ── Final Evaluation on Test Set ──
        logger.info("\n🧪 Running evaluation on test set...")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_loss, test_acc, test_preds, test_labels = run_epoch(
            model, loaders["test"], criterion, None, device, is_train=False
        )

        f1 = f1_score(test_labels, test_preds, average="binary", pos_label=1)
        precision = precision_score(test_labels, test_preds, average="binary", pos_label=1)
        recall = recall_score(test_labels, test_preds, average="binary", pos_label=1)
        cm = confusion_matrix(test_labels, test_preds)
        report = classification_report(
            test_labels, test_preds, target_names=["cat", "dog"]
        )

        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy : {test_acc:.4f}")
        logger.info(f"  F1 Score : {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall   : {recall:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"\nConfusion Matrix:\n{cm}")

        mlflow.log_metrics({
            "test_acc": test_acc,
            "test_loss": test_loss,
            "test_f1": f1,
            "test_precision": precision,
            "test_recall": recall,
            "best_val_acc": best_val_acc,
        })

        # ── Save Artifacts ──
        # Save confusion matrix as JSON
        cm_path = model_dir / "confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump({
                "matrix": cm.tolist(),
                "labels": ["cat", "dog"],
                "test_acc": test_acc,
            }, f, indent=2)

        # Save classification report
        report_path = model_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # Save training history
        history_path = model_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(history_path))

        # ── Export Model ──
        # Save as .pth
        final_model_path = model_dir / "model.pth"
        torch.save(model.state_dict(), final_model_path)

        # Save as TorchScript for production
        if artifact_cfg.get("export_torchscript", True):
            model.eval()
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            with torch.no_grad():
                traced = torch.jit.trace(model, dummy_input)
            ts_path = model_dir / "model_torchscript.pt"
            traced.save(str(ts_path))
            mlflow.log_artifact(str(ts_path))
            logger.info(f"TorchScript model saved: {ts_path}")

        # Save model metadata
        metadata = {
            "run_id": run_id,
            "architecture": model_cfg.get("architecture"),
            "num_classes": 2,
            "image_size": data_cfg.get("image_size", 224),
            "classes": ["cat", "dog"],
            "test_acc": float(test_acc),
            "test_f1": float(f1),
            "best_val_acc": float(best_val_acc),
            "pretrained": model_cfg.get("pretrained"),
            "dropout": float(model_cfg.get("dropout", 0.3)),
            "model_format": "torchscript",
            "mlflow_tracking_uri": effective_mlflow_uri,
        }
        meta_path = model_dir / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(meta_path))

        # Log model to MLflow Model Registry
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="cats-dogs-classifier",
        )

        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Best Val Acc: {best_val_acc:.4f}")
        logger.info(f"   Test Acc    : {test_acc:.4f}")
        logger.info(f"   MLflow Run  : {run_id}")
        logger.info(f"   Models saved: {model_dir}")

        return metadata


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--config", default="configs/train_config.yaml", help="Config file path")
@click.option("--architecture", default=None, help="Override model architecture")
@click.option("--epochs", default=None, type=int, help="Override epochs")
@click.option("--batch-size", default=None, type=int, help="Override batch size")
@click.option("--lr", default=None, type=float, help="Override learning rate")
@click.option("--mlflow-uri", default=None, help="Override MLflow tracking URI")
def main(config, architecture, epochs, batch_size, lr, mlflow_uri):
    """Train Cats vs Dogs classifier with MLflow experiment tracking."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if architecture:
        cfg["model"]["architecture"] = architecture
    if epochs:
        cfg["training"]["epochs"] = epochs
    if batch_size:
        cfg["training"]["batch_size"] = batch_size
    if lr:
        cfg["training"]["learning_rate"] = lr
    if mlflow_uri:
        cfg["project"]["mlflow_tracking_uri"] = mlflow_uri

    train(cfg)


if __name__ == "__main__":
    main()
