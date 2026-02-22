"""
src/data/preprocess.py
======================
Data preprocessing pipeline for Cats vs Dogs dataset.
- Downloads/validates raw data
- Resizes to 224x224 RGB
- Splits into train/val/test (80/10/10)
- Saves manifests for DVC tracking
"""

import random
import logging
import json
from pathlib import Path
from typing import Tuple, Dict, List

# Allow running as either `python -m src.data.preprocess` or `python src/data/preprocess.py`.
if __package__ is None or __package__ == "":
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from PIL import Image, ImageOps
import yaml
import click
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

CLASSES = ["cat", "dog"]
SPLITS = ["train", "val", "test"]
IMAGE_SIZE = (224, 224)


# ─── Core Preprocessing Functions ────────────────────────────────────────────


def load_and_validate_image(image_path: Path) -> Image.Image:
    """
    Load an image, convert to RGB, and validate it's not corrupt.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the image cannot be opened or is corrupt.
    """
    try:
        img = Image.open(image_path)
        img.verify()  # Check for corruption
        img = Image.open(image_path)  # Re-open after verify (verify closes)
        img = ImageOps.exif_transpose(img)  # Fix EXIF rotation
        img = img.convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Cannot load image {image_path}: {e}") from e


def resize_image(
    img: Image.Image, target_size: Tuple[int, int] = IMAGE_SIZE
) -> Image.Image:
    """
    Resize image to target size using high-quality Lanczos resampling.

    Args:
        img: PIL Image to resize.
        target_size: (width, height) tuple.

    Returns:
        Resized PIL Image.
    """
    return img.resize(target_size, Image.Resampling.LANCZOS)


def process_single_image(
    src_path: Path, dst_path: Path, target_size: Tuple[int, int] = IMAGE_SIZE
) -> bool:
    """
    Load, validate, resize, and save a single image.

    Args:
        src_path: Source image path.
        dst_path: Destination path to save processed image.
        target_size: Target (width, height).

    Returns:
        True if successful, False if image was skipped (corrupt/invalid).
    """
    try:
        img = load_and_validate_image(src_path)
        img = resize_image(img, target_size)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, format="JPEG", quality=95, optimize=True)
        return True
    except (ValueError, OSError, Exception) as e:
        logger.warning(f"Skipping {src_path.name}: {e}")
        return False


def split_file_list(
    file_list: List[Path],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split a list of files into train/val/test sets.

    Args:
        file_list: List of file paths to split.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set (test = 1 - train - val).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_files, val_files, test_files).
    """
    random.seed(seed)
    shuffled = file_list.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return train, val, test


# ─── Pipeline ────────────────────────────────────────────────────────────────


def discover_raw_images(raw_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover images in the raw directory.

    Expected structure (Kaggle dataset):
        raw/
            PetImages/
                Cat/  *.jpg
                Dog/  *.jpg
    OR flat:
        raw/
            cat/  *.jpg
            dog/  *.jpg

    Returns:
        Dict mapping class name -> list of image paths.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    discovered: Dict[str, List[Path]] = {cls: [] for cls in CLASSES}

    # Try PetImages structure (Kaggle)
    pet_images_dir = raw_dir / "PetImages"
    if pet_images_dir.exists():
        for cls in CLASSES:
            cls_dir = pet_images_dir / cls.capitalize()
            if cls_dir.exists():
                discovered[cls] = [
                    p
                    for p in sorted(cls_dir.iterdir())
                    if p.suffix.lower() in image_extensions
                ]
        logger.info(f"Discovered PetImages structure: {pet_images_dir}")
    else:
        # Try flat class directories
        for cls in CLASSES:
            for subdir_name in [cls, cls.capitalize()]:
                cls_dir = raw_dir / subdir_name
                if cls_dir.exists():
                    discovered[cls] = [
                        p
                        for p in sorted(cls_dir.iterdir())
                        if p.suffix.lower() in image_extensions
                    ]
                    break

    for cls, files in discovered.items():
        logger.info(f"Found {len(files):,} raw {cls} images")

    return discovered


def run_preprocessing_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    image_size: int = 224,
    seed: int = 42,
    max_per_class: int = None,
) -> Dict:
    """
    Full preprocessing pipeline: discover → split → resize → save → manifest.

    Returns:
        Dictionary with dataset statistics.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    target_size = (image_size, image_size)

    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info(f"  Raw dir      : {raw_path}")
    logger.info(f"  Processed dir: {processed_path}")
    logger.info(f"  Image size   : {image_size}x{image_size}")
    logger.info(
        f"  Split        : {train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%}"
    )
    logger.info("=" * 60)

    # Discover
    class_images = discover_raw_images(raw_path)

    stats = {
        "image_size": image_size,
        "splits": {},
        "class_counts": {},
        "skipped": 0,
        "total_processed": 0,
    }

    manifest = {}

    for cls in CLASSES:
        images = class_images[cls]
        if max_per_class:
            images = images[:max_per_class]

        train_files, val_files, test_files = split_file_list(
            images, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )

        split_map = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, split_files in split_map.items():
            processed_count = 0
            skipped_count = 0

            for src_path in tqdm(split_files, desc=f"  {split_name}/{cls}", unit="img"):
                dst_path = processed_path / split_name / cls / src_path.name
                if dst_path.suffix.lower() not in {".jpg", ".jpeg"}:
                    dst_path = dst_path.with_suffix(".jpg")

                success = process_single_image(src_path, dst_path, target_size)
                if success:
                    processed_count += 1
                    manifest[str(dst_path.relative_to(processed_path))] = {
                        "class": cls,
                        "split": split_name,
                        "source": str(src_path),
                    }
                else:
                    skipped_count += 1

            logger.info(
                f"  [{split_name}/{cls}] Processed: {processed_count}, Skipped: {skipped_count}"
            )
            stats["skipped"] += skipped_count
            stats["total_processed"] += processed_count

    # Compute split counts
    for split_name in SPLITS:
        split_dir = processed_path / split_name
        counts = {}
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                counts[cls] = len(list(cls_dir.glob("*.jpg")))
            else:
                counts[cls] = 0
        stats["splits"][split_name] = counts

    # Save manifest
    manifest_path = processed_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Save stats
    stats_path = processed_path / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n📊 Dataset Statistics:")
    for split_name, counts in stats["splits"].items():
        total = sum(counts.values())
        logger.info(f"  {split_name:8s}: {counts} = {total:,} total")
    logger.info(f"  Skipped (corrupt): {stats['skipped']}")
    logger.info(f"  Total processed  : {stats['total_processed']:,}")
    logger.info(f"\n✅ Manifest saved: {manifest_path}")

    return stats


# ─── CLI ─────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--raw-dir", default="data/raw", help="Raw data directory")
@click.option("--processed-dir", default="data/processed", help="Output directory")
@click.option("--image-size", default=224, help="Target image size (square)")
@click.option("--train-ratio", default=0.80, help="Train split ratio")
@click.option("--val-ratio", default=0.10, help="Validation split ratio")
@click.option("--seed", default=42, help="Random seed")
@click.option(
    "--max-per-class",
    default=None,
    type=int,
    help="Limit images per class (for testing)",
)
@click.option("--config", default=None, help="Path to YAML config (overrides CLI args)")
def main(
    raw_dir,
    processed_dir,
    image_size,
    train_ratio,
    val_ratio,
    seed,
    max_per_class,
    config,
):
    """Preprocess Cats vs Dogs dataset for training."""
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get("data", {})
        raw_dir = data_cfg.get("raw_dir", raw_dir)
        processed_dir = data_cfg.get("processed_dir", processed_dir)
        image_size = data_cfg.get("image_size", image_size)
        splits = data_cfg.get("splits", {})
        train_ratio = splits.get("train", train_ratio)
        val_ratio = splits.get("val", val_ratio)
        seed = data_cfg.get("random_seed", seed)

    run_preprocessing_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        image_size=image_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        max_per_class=max_per_class,
    )


if __name__ == "__main__":
    main()
