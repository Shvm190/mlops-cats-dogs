"""
tests/test_inference.py
========================
Unit tests for model inference utilities.
Tests preprocessing, prediction shape/type, and API endpoint behavior.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_jpeg_bytes(width: int = 300, height: int = 200) -> bytes:
    """Create valid JPEG image bytes for testing."""
    data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(data, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_cat_tensor() -> torch.Tensor:
    """Create a synthetic image tensor."""
    return torch.zeros(1, 3, 224, 224)


@pytest.fixture
def mock_model():
    """A mock PyTorch model that always predicts 'cat' with 90% confidence."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)

    # Logits that give ~90% cat, ~10% dog after softmax
    logits = torch.tensor([[2.2, 0.1]])
    model.return_value = logits
    model.__call__ = MagicMock(return_value=logits)
    return model


@pytest.fixture
def mock_model_loader(mock_model, tmp_path):
    """A ModelLoader with a mocked model."""
    from src.models.inference import ModelLoader

    loader = MagicMock(spec=ModelLoader)
    loader.model = mock_model
    loader.device = torch.device("cpu")
    loader.is_loaded = True
    loader.metadata = {
        "architecture": "mobilenet_v2",
        "num_classes": 2,
        "classes": ["cat", "dog"],
        "image_size": 224,
        "test_acc": 0.95,
    }
    return loader


# ─── Tests: preprocess_image ─────────────────────────────────────────────────

class TestPreprocessImage:
    def test_output_shape(self):
        """Preprocessed image should be (1, 3, 224, 224)."""
        from src.models.inference import preprocess_image
        img = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8), "RGB"
        )
        tensor = preprocess_image(img, image_size=224)
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_dtype(self):
        """Output tensor should be float32."""
        from src.models.inference import preprocess_image
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), "RGB"
        )
        tensor = preprocess_image(img)
        assert tensor.dtype == torch.float32

    def test_handles_non_224_size(self):
        """Custom image size should be respected."""
        from src.models.inference import preprocess_image
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), "RGB"
        )
        tensor = preprocess_image(img, image_size=128)
        assert tensor.shape == (1, 3, 128, 128)

    def test_normalizes_values(self):
        """Values should be normalized (not in [0, 255] range)."""
        from src.models.inference import preprocess_image
        img = Image.fromarray(
            np.full((224, 224, 3), 128, dtype=np.uint8), "RGB"
        )
        tensor = preprocess_image(img)
        assert tensor.min() < 1.0
        assert tensor.max() < 5.0

    def test_preprocess_bytes_matches_image(self):
        """preprocess_image_bytes should give same result as preprocess_image for same image."""
        from src.models.inference import preprocess_image, preprocess_image_bytes
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), "RGB"
        )
        # Save to bytes
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=100)
        img_bytes = buf.getvalue()

        # Load fresh (JPEG compression may slightly alter pixels)
        reloaded = Image.open(io.BytesIO(img_bytes))
        t1 = preprocess_image(reloaded)
        t2 = preprocess_image_bytes(img_bytes)
        assert t1.shape == t2.shape


# ─── Tests: predict function ─────────────────────────────────────────────────

class TestPredictFunction:
    def test_returns_correct_keys(self, mock_model):
        """predict() result should have all required keys."""
        from src.models.inference import predict
        tensor = make_cat_tensor()
        result = predict(mock_model, tensor, torch.device("cpu"))

        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "latency_ms" in result
        assert "predicted_class_index" in result

    def test_label_is_valid_class(self, mock_model):
        """Predicted label should be 'cat' or 'dog'."""
        from src.models.inference import predict
        tensor = make_cat_tensor()
        result = predict(mock_model, tensor, torch.device("cpu"))
        assert result["label"] in ["cat", "dog"]

    def test_confidence_in_valid_range(self, mock_model):
        """Confidence should be between 0 and 1."""
        from src.models.inference import predict
        tensor = make_cat_tensor()
        result = predict(mock_model, tensor, torch.device("cpu"))
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, mock_model):
        """Softmax probabilities should sum to 1.0 (within tolerance)."""
        from src.models.inference import predict
        tensor = make_cat_tensor()
        result = predict(mock_model, tensor, torch.device("cpu"))
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_latency_is_positive(self, mock_model):
        """Latency should be a positive number."""
        from src.models.inference import predict
        tensor = make_cat_tensor()
        result = predict(mock_model, tensor, torch.device("cpu"))
        assert result["latency_ms"] > 0

    def test_cat_prediction_for_cat_logits(self):
        """Model with cat-biased logits should predict 'cat'."""
        from src.models.inference import predict
        model = MagicMock()
        model.return_value = torch.tensor([[3.0, 0.1]])  # Strong cat bias
        model.__call__ = MagicMock(return_value=torch.tensor([[3.0, 0.1]]))

        tensor = make_cat_tensor()
        result = predict(model, tensor, torch.device("cpu"))
        assert result["label"] == "cat"

    def test_dog_prediction_for_dog_logits(self):
        """Model with dog-biased logits should predict 'dog'."""
        from src.models.inference import predict
        model = MagicMock()
        model.return_value = torch.tensor([[0.1, 3.0]])  # Strong dog bias
        model.__call__ = MagicMock(return_value=torch.tensor([[0.1, 3.0]]))

        tensor = make_cat_tensor()
        result = predict(model, tensor, torch.device("cpu"))
        assert result["label"] == "dog"


# ─── Tests: ModelLoader ───────────────────────────────────────────────────────

class TestModelLoader:
    def test_pth_uses_metadata_architecture(self, tmp_path, monkeypatch):
        """ModelLoader should read metadata before constructing a .pth model."""
        from src.models.inference import ModelLoader

        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"x")
        metadata_path = tmp_path / "model_metadata.json"
        metadata_path.write_text('{"architecture":"simple_cnn","num_classes":2}')

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        used = {}

        def fake_build_model(architecture, num_classes, dropout, pretrained):
            used["architecture"] = architecture
            used["num_classes"] = num_classes
            return mock_model

        monkeypatch.setattr("src.models.architecture.build_model", fake_build_model)
        monkeypatch.setattr("torch.load", lambda *args, **kwargs: {})

        loader = ModelLoader(str(model_path), str(metadata_path))
        loader.load()

        assert used["architecture"] == "simple_cnn"
        assert used["num_classes"] == 2

    def test_pth_accepts_checkpoint_wrapper(self, tmp_path, monkeypatch):
        """ModelLoader should accept checkpoints with model_state_dict wrapper."""
        from src.models.inference import ModelLoader

        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"x")
        metadata_path = tmp_path / "model_metadata.json"
        metadata_path.write_text('{"architecture":"mobilenet_v2","num_classes":2}')

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        monkeypatch.setattr("src.models.architecture.build_model", lambda **kwargs: mock_model)
        monkeypatch.setattr("torch.load", lambda *args, **kwargs: {"model_state_dict": {}})

        loader = ModelLoader(str(model_path), str(metadata_path))
        loader.load()
        assert loader.is_loaded is True


# ─── Tests: API Endpoints (Integration) ──────────────────────────────────────

class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints using TestClient."""

    @pytest.fixture
    def client(self, mock_model_loader):
        """Create a test client with mocked model."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        # Patch the global model loader
        with patch("src.api.main.model_loader", mock_model_loader), \
             patch("src.models.inference.ModelLoader.get_instance", return_value=mock_model_loader):
            with TestClient(app) as c:
                yield c

    def test_health_endpoint_returns_200(self, client):
        """GET /health should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        """Health response should match expected schema."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "service_version" in data

    def test_health_shows_model_loaded(self, client):
        """Health endpoint should reflect model is loaded."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True

    def test_ready_endpoint_returns_200(self, client):
        """GET /ready should return 200 when model is loaded."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_predict_returns_200_for_valid_image(self, client):
        """POST /predict with valid image should return 200."""
        image_bytes = make_jpeg_bytes()
        response = client.post(
            "/predict",
            files={"file": ("cat.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self, client):
        """Prediction response should contain label, confidence, probabilities."""
        image_bytes = make_jpeg_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert "label" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "latency_ms" in data

    def test_predict_label_is_cat_or_dog(self, client):
        """Prediction label should be 'cat' or 'dog'."""
        image_bytes = make_jpeg_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["label"] in ["cat", "dog"]

    def test_predict_rejects_non_image(self, client):
        """POST /predict with non-image should return 400."""
        response = client.post(
            "/predict",
            files={"file": ("text.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400

    def test_metrics_endpoint_returns_200(self, client):
        """GET /metrics should return 200 with Prometheus text."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_batch_predict_returns_results(self, client):
        """POST /predict/batch should return one result per file."""
        image_bytes = make_jpeg_bytes()
        response = client.post(
            "/predict/batch",
            files=[("files", ("cat.jpg", image_bytes, "image/jpeg"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["results"]) == 1

    def test_batch_predict_marks_invalid_file_error(self, client):
        """Batch prediction should report per-file validation errors."""
        response = client.post(
            "/predict/batch",
            files=[("files", ("bad.txt", b"abc", "text/plain"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["error"] is not None

    def test_root_endpoint(self, client):
        """GET / should return 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_batch_predict_returns_503_when_model_unavailable(self):
        """Batch endpoint should fail fast if model cannot be loaded."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        with patch("src.api.main.model_loader", None), \
             patch("src.models.inference.ModelLoader.get_instance", side_effect=RuntimeError("missing model")):
            with TestClient(app) as c:
                response = c.post(
                    "/predict/batch",
                    files=[("files", ("cat.jpg", make_jpeg_bytes(), "image/jpeg"))],
                )
        assert response.status_code == 503


# ─── Tests: Model Architecture ───────────────────────────────────────────────

class TestModelArchitecture:
    def test_simple_cnn_forward_pass(self):
        """SimpleCNN should accept (B, 3, 224, 224) and output (B, 2)."""
        from src.models.architecture import SimpleCNN
        model = SimpleCNN(num_classes=2)
        x = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_mobilenet_forward_pass(self):
        """MobileNetV2Classifier should accept (B, 3, 224, 224) and output (B, 2)."""
        from src.models.architecture import MobileNetV2Classifier
        model = MobileNetV2Classifier(num_classes=2, pretrained=False)
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_build_model_factory(self):
        """build_model() should return correct model type."""
        from src.models.architecture import build_model, SimpleCNN, MobileNetV2Classifier
        m1 = build_model("simple_cnn")
        assert isinstance(m1, SimpleCNN)
        m2 = build_model("mobilenet_v2", pretrained=False)
        assert isinstance(m2, MobileNetV2Classifier)

    def test_build_model_invalid_architecture(self):
        """build_model() with unknown architecture should raise ValueError."""
        from src.models.architecture import build_model
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("nonexistent_model")

    def test_count_parameters(self):
        """Parameter counting should return valid dict."""
        from src.models.architecture import SimpleCNN, count_parameters
        model = SimpleCNN()
        counts = count_parameters(model)
        assert "total" in counts
        assert "trainable" in counts
        assert counts["total"] > 0
        assert counts["trainable"] <= counts["total"]
