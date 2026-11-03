
import json
from pathlib import Path

import mlflow

import src.training.train as training
from config import ART, REPORTS


# Configure MLflow with SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("UPI-IntentGuard-Experiments")


def main():

    with mlflow.start_run(run_name="UPI-Fraud-Detection-Training"):

        # 1. Log training parameters
        mlflow.log_params({
            "model": "IntentGuardNet",
            "framework": "PyTorch",
            "training_function": "train",
            "random_seed": 42,
            "tracking_backend": "SQLite"
        })

        print("\nStarting UPI-IntentGuard training with MLflow...")

        # 2. Execute original training pipeline
        training.train()

        # 3. Load evaluation metrics
        metrics_path = Path(ART) / "metrics.json"

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {metrics_path}"
            )

        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        # 4. Log numerical metrics
        numeric_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        }

        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        # 5. Log trained model checkpoint
        model_path = Path(ART) / "intentguard.pt"

        if model_path.exists():
            mlflow.log_artifact(
                str(model_path),
                artifact_path="model"
            )

        # 6. Log evaluation metrics file
        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="reports"
        )

        # 7. Log generated reports
        report_files = [
            "classification_report.txt",
            "confusion_matrix.csv"
        ]

        for filename in report_files:

            report_path = Path(REPORTS) / filename

            if report_path.exists():
                mlflow.log_artifact(
                    str(report_path),
                    artifact_path="reports"
                )

        print("\nMLflow tracking completed successfully!")
        print("Experiment: UPI-IntentGuard-Experiments")
        print("Model, parameters, metrics and reports logged.")


if __name__ == "__main__":
    main()
