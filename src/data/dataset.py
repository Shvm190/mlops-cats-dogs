"""
src/data/dataset.py
===================
PyTorch Dataset classes and DataLoader factories for
Cats vs Dogs classification.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ─── Label Mapping ──────────────────────────────────────────────────────────

CLASS_TO_IDX = {"cat": 0, "dog": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Build augmented transform pipeline for training."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Build deterministic transform pipeline for validation/test/inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """Alias for eval transforms; used by inference service."""
    return get_eval_transforms(image_size)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CatsDogsDataset(Dataset):
    """
    Dataset for Cats vs Dogs classification.

    Expects directory structure:
        root/
            cat/  *.jpg
            dog/  *.jpg
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        """
        Args:
            root_dir: Path to split directory (e.g., data/processed/train).
            transform: Optional transform to apply to each image.
            extensions: Valid image file extensions.
        """
        self.root = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        self.samples: list[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Scan directory and build (path, label) pairs."""
        for cls_name, label in CLASS_TO_IDX.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in self.extensions:
                    self.samples.append((p, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {self.root}. "
                f"Expected subdirs: {list(CLASS_TO_IDX.keys())}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def class_counts(self) -> Dict[str, int]:
        """Return per-class sample counts."""
        counts = {cls: 0 for cls in CLASS_TO_IDX}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(
    processed_dir: str = "data/processed",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders.

    Args:
        processed_dir: Root of preprocessed data.
        image_size: Input image size.
        batch_size: Batch size for all loaders.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfers.

    Returns:
        Dictionary with keys 'train', 'val', 'test'.
    """
    loaders = {}

    transform_map = {
        "train": get_train_transforms(image_size),
        "val": get_eval_transforms(image_size),
        "test": get_eval_transforms(image_size),
    }

    for split, transform in transform_map.items():
        split_dir = Path(processed_dir) / split
        if not split_dir.exists():
            continue

        dataset = CatsDogsDataset(root_dir=str(split_dir), transform=transform)
        shuffle = split == "train"
        drop_last = split == "train" and len(dataset) >= batch_size

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    return loaders
