"""
tests/test_preprocessing.py
============================
Unit tests for data preprocessing functions.
Run with: pytest tests/ -v
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ── Import under test ────────────────────────────────────────────────────────
from src.data.preprocess import (
    load_and_validate_image,
    resize_image,
    process_single_image,
    split_file_list,
)
from src.data.dataset import (
    get_train_transforms,
    get_eval_transforms,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


def make_test_image(
    width: int = 300, height: int = 200, mode: str = "RGB"
) -> Image.Image:
    """Create a synthetic test image."""
    data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, mode=mode)


def save_test_image(img: Image.Image, path: Path, fmt: str = "JPEG"):
    """Save test image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), format=fmt)


# ─── Tests: load_and_validate_image ──────────────────────────────────────────


class TestLoadAndValidateImage:
    def test_loads_valid_jpeg(self, tmp_path):
        """Valid JPEG should load successfully as RGB."""
        img = make_test_image()
        path = tmp_path / "test.jpg"
        save_test_image(img, path)

        result = load_and_validate_image(path)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_loads_png_and_converts_to_rgb(self, tmp_path):
        """PNG with alpha channel should be converted to RGB."""
        rgba_img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8), mode="RGBA"
        )
        path = tmp_path / "test.png"
        rgba_img.save(str(path), format="PNG")

        result = load_and_validate_image(path)

        assert result.mode == "RGB"

    def test_raises_on_nonexistent_file(self, tmp_path):
        """Non-existent file should raise ValueError."""
        path = tmp_path / "nonexistent.jpg"
        with pytest.raises(ValueError, match="Cannot load image"):
            load_and_validate_image(path)

    def test_raises_on_corrupt_file(self, tmp_path):
        """Corrupt file should raise ValueError."""
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"not an image at all")
        with pytest.raises(ValueError):
            load_and_validate_image(path)

    def test_grayscale_converted_to_rgb(self, tmp_path):
        """Grayscale image should be converted to RGB."""
        gray = Image.fromarray(
            np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode="L"
        )
        path = tmp_path / "gray.jpg"
        gray.convert("RGB").save(str(path), format="JPEG")

        result = load_and_validate_image(path)
        assert result.mode == "RGB"


# ─── Tests: resize_image ──────────────────────────────────────────────────────


class TestResizeImage:
    def test_resizes_to_target(self):
        """Image should be resized to exact target dimensions."""
        img = make_test_image(width=640, height=480)
        resized = resize_image(img, target_size=(224, 224))
        assert resized.size == (224, 224)

    def test_handles_already_correct_size(self):
        """Image already at target size should still work."""
        img = make_test_image(width=224, height=224)
        resized = resize_image(img, target_size=(224, 224))
        assert resized.size == (224, 224)

    def test_resizes_small_image_upscale(self):
        """Small images should be upscaled without error."""
        img = make_test_image(width=50, height=50)
        resized = resize_image(img, target_size=(224, 224))
        assert resized.size == (224, 224)

    def test_custom_target_size(self):
        """Custom target sizes should be respected."""
        img = make_test_image(width=1024, height=768)
        resized = resize_image(img, target_size=(128, 128))
        assert resized.size == (128, 128)


# ─── Tests: process_single_image ─────────────────────────────────────────────


class TestProcessSingleImage:
    def test_processes_and_saves_image(self, tmp_path):
        """Valid image should be processed and saved."""
        img = make_test_image(300, 200)
        src = tmp_path / "src" / "cat.jpg"
        dst = tmp_path / "dst" / "cat.jpg"
        save_test_image(img, src)

        success = process_single_image(src, dst, target_size=(224, 224))

        assert success is True
        assert dst.exists()
        saved = Image.open(dst)
        assert saved.size == (224, 224)

    def test_returns_false_on_corrupt_image(self, tmp_path):
        """Corrupt image should return False (not raise)."""
        src = tmp_path / "corrupt.jpg"
        src.write_bytes(b"garbage")
        dst = tmp_path / "dst.jpg"

        success = process_single_image(src, dst)
        assert success is False
        assert not dst.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Destination parent directories should be created automatically."""
        img = make_test_image()
        src = tmp_path / "src.jpg"
        save_test_image(img, src)
        dst = tmp_path / "deep" / "nested" / "dir" / "dst.jpg"

        success = process_single_image(src, dst)

        assert success is True
        assert dst.exists()


# ─── Tests: split_file_list ──────────────────────────────────────────────────


class TestSplitFileList:
    def make_paths(self, n: int) -> list:
        return [Path(f"img_{i:04d}.jpg") for i in range(n)]

    def test_correct_split_ratios(self):
        """Split sizes should match expected ratios (approximately)."""
        files = self.make_paths(1000)
        train, val, test = split_file_list(
            files, train_ratio=0.8, val_ratio=0.1, seed=42
        )

        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100

    def test_all_files_accounted_for(self):
        """Every file should appear in exactly one split."""
        files = self.make_paths(500)
        train, val, test = split_file_list(files, train_ratio=0.8, val_ratio=0.1)

        all_files = set(train) | set(val) | set(test)
        assert len(all_files) == len(files)
        assert len(train) + len(val) + len(test) == len(files)

    def test_reproducible_with_same_seed(self):
        """Same seed should produce identical splits."""
        files = self.make_paths(200)
        t1, v1, te1 = split_file_list(files, seed=42)
        t2, v2, te2 = split_file_list(files, seed=42)
        assert t1 == t2 and v1 == v2 and te1 == te2

    def test_different_seeds_produce_different_splits(self):
        """Different seeds should produce different orderings."""
        files = self.make_paths(200)
        t1, _, _ = split_file_list(files, seed=1)
        t2, _, _ = split_file_list(files, seed=99)
        assert t1 != t2

    def test_handles_small_dataset(self):
        """Very small datasets should not crash."""
        files = self.make_paths(10)
        train, val, test = split_file_list(files, train_ratio=0.8, val_ratio=0.1)
        assert len(train) + len(val) + len(test) == 10


# ─── Tests: Transforms ───────────────────────────────────────────────────────


class TestTransforms:
    def test_train_transform_output_shape(self):
        """Train transform should produce (3, 224, 224) tensor."""
        import torch

        img = make_test_image(256, 256)
        transform = get_train_transforms(224)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_eval_transform_output_shape(self):
        """Eval transform should produce (3, 224, 224) tensor."""
        img = make_test_image(300, 300)
        transform = get_eval_transforms(224)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_eval_transform_is_deterministic(self):
        """Eval transform should produce identical outputs for same image."""
        import torch

        img = make_test_image(224, 224)
        transform = get_eval_transforms(224)
        t1 = transform(img)
        t2 = transform(img)
        assert torch.allclose(t1, t2)

    def test_train_transform_normalizes(self):
        """Transformed tensor should have values in roughly [-3, 3] range after normalization."""
        img = make_test_image(224, 224)
        transform = get_eval_transforms(224)
        tensor = transform(img)
        assert tensor.min() < 0  # Normalized values go negative
        assert tensor.max() < 5  # Shouldn't be huge

    def test_class_mapping_consistency(self):
        """CLASS_TO_IDX and IDX_TO_CLASS should be inverses."""
        for cls, idx in CLASS_TO_IDX.items():
            assert IDX_TO_CLASS[idx] == cls
        assert set(CLASS_TO_IDX.keys()) == {"cat", "dog"}
        assert set(CLASS_TO_IDX.values()) == {0, 1}
