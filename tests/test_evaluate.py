import json

import torch

from src.models.evaluate import evaluate_model


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, images):
        batch_size = images.shape[0]
        # Deterministic logits with class-1 favored.
        return torch.tensor([[0.1, 2.0]] * batch_size, dtype=torch.float32)


class _FakeLoader:
    def __init__(self, model_path):
        self.model = _FakeModel()
        self.model_path = model_path

    def load(self):
        return self


def test_evaluate_model_writes_artifacts_and_metrics(tmp_path, monkeypatch):
    # Stub dataset + dataloader so evaluation is fast and deterministic.
    class _FakeDataset:
        def __init__(self, root_dir, transform):
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return 4

    images = torch.zeros(4, 3, 224, 224)
    labels = torch.tensor([1, 1, 1, 1], dtype=torch.long)

    monkeypatch.setattr("src.models.evaluate.ModelLoader", _FakeLoader)
    monkeypatch.setattr("src.data.dataset.CatsDogsDataset", _FakeDataset)
    monkeypatch.setattr("src.data.dataset.get_eval_transforms", lambda size: None)
    monkeypatch.setattr(
        "torch.utils.data.DataLoader",
        lambda dataset, batch_size, shuffle, num_workers: [(images, labels)],
    )

    out_dir = tmp_path / "eval_out"
    metrics = evaluate_model(
        model_path="artifacts/models/model_torchscript.pt",
        test_dir=str(tmp_path / "test"),
        output_dir=str(out_dir),
        image_size=224,
        batch_size=2,
    )

    assert set(metrics.keys()) == {
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
    }

    metrics_path = out_dir / "metrics.json"
    cm_path = out_dir / "confusion_matrix.json"
    report_path = out_dir / "classification_report.txt"
    assert metrics_path.exists()
    assert cm_path.exists()
    assert report_path.exists()

    saved_metrics = json.loads(metrics_path.read_text())
    assert saved_metrics["accuracy"] == 1.0
    saved_cm = json.loads(cm_path.read_text())
    assert saved_cm["labels"] == ["cat", "dog"]
    assert isinstance(saved_cm["matrix"], list)
