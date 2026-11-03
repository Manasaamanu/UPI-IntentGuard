import json
from pathlib import Path


def test_trained_model_exists():
    model_path = Path("artifacts/intentguard.pt")
    assert model_path.exists()


def test_metrics_file_exists():
    metrics_path = Path("artifacts/metrics.json")
    assert metrics_path.exists()


def test_model_metrics():
    metrics_path = Path("artifacts/metrics.json")

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics