# tests/conftest.py
"""
Shared pytest fixtures for all test modules.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """Create temporary data directory structure."""
    for split in ["train", "val", "test"]:
        for cls in ["cat", "dog"]:
            (tmp_path / split / cls).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Return valid JPEG bytes."""
    data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(data, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def populated_data_dir(tmp_data_dir) -> Path:
    """Create data dir with synthetic images (5 per class per split)."""
    for split in ["train", "val", "test"]:
        for cls in ["cat", "dog"]:
            for i in range(5):
                data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(data, "RGB")
                img.save(
                    str(tmp_data_dir / split / cls / f"img_{i:04d}.jpg"), format="JPEG"
                )
    return tmp_data_dir
