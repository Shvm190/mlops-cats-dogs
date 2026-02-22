import json

from src.monitoring.post_deploy_monitor import (
    compute_performance_metrics,
    generate_simulated_batch,
    run_monitoring,
    run_predictions_on_batch,
)


def test_generate_simulated_batch_collects_labels(tmp_path):
    for cls in ["cat", "dog"]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            (cls_dir / f"img_{i}.jpg").write_bytes(b"fake")

    batch = generate_simulated_batch(test_dir=str(tmp_path), max_samples=4)

    assert len(batch) == 4
    labels = {item["true_label"] for item in batch}
    assert labels == {"cat", "dog"}


def test_compute_performance_metrics_no_valid_predictions():
    result = compute_performance_metrics(
        [
            {
                "predicted_label": None,
                "true_label": "cat",
                "confidence": None,
                "latency_ms": None,
            }
        ]
    )
    assert result == {"error": "No valid predictions to evaluate"}


def test_compute_performance_metrics_with_alerts():
    results = [
        {
            "true_label": "cat",
            "predicted_label": "dog",
            "confidence": 0.2,
            "probabilities": {"cat": 0.2, "dog": 0.8},
            "latency_ms": 10.0,
            "correct": False,
        },
        {
            "true_label": "dog",
            "predicted_label": "cat",
            "confidence": 0.3,
            "probabilities": {"cat": 0.7, "dog": 0.3},
            "latency_ms": 12.0,
            "correct": False,
        },
    ]
    metrics = compute_performance_metrics(results)
    assert metrics["accuracy"] == 0.0
    assert metrics["valid_predictions"] == 2
    assert len(metrics["alerts"]) >= 1


def test_run_predictions_on_batch_success_and_failure(tmp_path, monkeypatch):
    good_img = tmp_path / "good.jpg"
    bad_img = tmp_path / "bad.jpg"
    good_img.write_bytes(b"good")
    bad_img.write_bytes(b"bad")

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "label": "cat",
                "confidence": 0.9,
                "probabilities": {"cat": 0.9, "dog": 0.1},
                "latency_ms": 5.2,
            }

    def _fake_post(url, files, timeout):
        filename = files["file"][0]
        if filename == "bad.jpg":
            raise RuntimeError("network error")
        return _Resp()

    monkeypatch.setattr("src.monitoring.post_deploy_monitor.requests.post", _fake_post)

    batch = [
        {"image_path": str(good_img), "true_label": "cat"},
        {"image_path": str(bad_img), "true_label": "dog"},
    ]
    outputs = run_predictions_on_batch(batch, api_url="http://fake:8080")
    assert outputs[0]["predicted_label"] == "cat"
    assert outputs[1]["predicted_label"] is None
    assert "error" in outputs[1]


def test_run_monitoring_success_writes_report(tmp_path, monkeypatch):
    report_path = tmp_path / "monitoring.json"

    class _HealthResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "healthy"}

    monkeypatch.setattr(
        "src.monitoring.post_deploy_monitor.requests.get",
        lambda url, timeout: _HealthResp(),
    )
    monkeypatch.setattr(
        "src.monitoring.post_deploy_monitor.generate_simulated_batch",
        lambda test_dir, max_samples: [
            {"image_path": "a.jpg", "true_label": "cat"},
            {"image_path": "b.jpg", "true_label": "dog"},
        ],
    )
    monkeypatch.setattr(
        "src.monitoring.post_deploy_monitor.run_predictions_on_batch",
        lambda batch, api_url: [
            {
                "true_label": "cat",
                "predicted_label": "cat",
                "confidence": 0.95,
                "probabilities": {"cat": 0.95, "dog": 0.05},
                "latency_ms": 7.2,
                "correct": True,
            },
            {
                "true_label": "dog",
                "predicted_label": "dog",
                "confidence": 0.93,
                "probabilities": {"cat": 0.07, "dog": 0.93},
                "latency_ms": 8.1,
                "correct": True,
            },
        ],
    )

    metrics = run_monitoring(
        api_url="http://fake:8080",
        test_dir=str(tmp_path),
        max_samples=1,
        output_path=str(report_path),
        log_to_mlflow=False,
    )

    assert metrics is not None
    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert "classification_report" not in saved
    assert saved["accuracy"] == 1.0


def test_run_monitoring_returns_none_when_health_fails(monkeypatch, tmp_path):
    def _raise_health(url, timeout):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(
        "src.monitoring.post_deploy_monitor.requests.get", _raise_health
    )

    result = run_monitoring(
        api_url="http://fake:8080",
        test_dir=str(tmp_path),
        max_samples=2,
        output_path=str(tmp_path / "report.json"),
    )
    assert result is None
