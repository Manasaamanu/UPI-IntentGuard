import numpy as np
import pandas as pd
import torch

from config import ART, DEVICE, FEATURES, SEQ_LEN
from src.models import IntentGuardNet
from src.features import build_feature_frame
from src.scoring import intent_drift_score, action


class Predictor:
    def __init__(self):
        checkpoint_path = ART / "intentguard.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {checkpoint_path}. Run "
                "`python -m src.training.train` first."
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=False
        )
        if checkpoint.get("features") != FEATURES:
            raise ValueError(
                "Checkpoint feature order does not match config.FEATURES. "
                "Retrain the model using the updated training script."
            )

        self.rv = checkpoint["rv"]
        self.scaler = checkpoint["scaler"]
        self.recipient_counts = checkpoint["recipient_counts"]
        self.global_median = float(checkpoint["global_median"])
        self.edge = checkpoint["edge"].to(DEVICE)

        self.model = IntentGuardNet(
            checkpoint["n"], checkpoint["dim"]
        ).to(DEVICE)
        self.model.load_state_dict(checkpoint["state"])
        self.model.eval()

    def predict(self, amount, hour, recipient, category, device, location, history):
        history = history.copy() if history is not None else pd.DataFrame()
        required = [
            "amount", "timestamp", "recipient_id",
            "category", "device_type", "location"
        ]

        if len(history):
            missing = [column for column in required if column not in history.columns]
            if missing:
                raise ValueError(f"History is missing columns: {missing}")
            history = history[required].copy()
            history["user_id"] = "LIVE_DEMO_USER"
            history["is_fraud"] = 0
            history["timestamp"] = pd.to_datetime(history["timestamp"])
            history = history.sort_values("timestamp")

            last_timestamp = history["timestamp"].iloc[-1]
            hour_gap = (int(hour) - int(last_timestamp.hour)) % 24
            if hour_gap == 0:
                hour_gap = 24
            candidate_timestamp = last_timestamp + pd.Timedelta(hours=hour_gap)
        else:
            history = pd.DataFrame(columns=required + ["user_id", "is_fraud"])
            candidate_timestamp = pd.Timestamp(
                year=2025, month=1, day=1, hour=int(hour)
            )

        candidate = pd.DataFrame([{
            "user_id": "LIVE_DEMO_USER",
            "amount": float(amount),
            "timestamp": candidate_timestamp,
            "recipient_id": str(recipient),
            "category": str(category),
            "device_type": str(device),
            "location": str(location),
            "is_fraud": 0,
        }])

        combined = pd.concat([history, candidate], ignore_index=True)
        combined["user_id"] = "LIVE_DEMO_USER"
        featured = build_feature_frame(
            combined,
            recipient_counts=self.recipient_counts,
            global_median=self.global_median,
        )

        raw_features = featured[FEATURES].to_numpy(dtype=np.float32)
        scaled_features = self.scaler.transform(raw_features).astype(np.float32)
        scaled_features = scaled_features[-SEQ_LEN:]

        padded = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)
        padded[-len(scaled_features):] = scaled_features
        context = padded[-1].copy()

        recipient_index = self.rv.get(str(recipient), 0)
        seq_tensor = torch.tensor(
            padded[None], dtype=torch.float32, device=DEVICE
        )
        ctx_tensor = torch.tensor(
            context[None], dtype=torch.float32, device=DEVICE
        )
        rec_tensor = torch.tensor(
            [recipient_index], dtype=torch.long, device=DEVICE
        )

        with torch.no_grad():
            logit = self.model(
                seq_tensor, ctx_tensor, rec_tensor, self.edge
            )
            probability = float(torch.sigmoid(logit).item())

        # Behavioural indicators are interpretable feature-based signals.
        avg_amount = (
            float(pd.to_numeric(history["amount"]).mean())
            if len(history) else self.global_median
        )
        known_recipients = (
            set(history["recipient_id"].astype(str))
            if len(history) else set()
        )
        recipient_unfamiliarity = float(str(recipient) not in known_recipients)
        amount_deviation = min(
            abs(float(amount) / (avg_amount + 1.0) - 1.0) / 4.0, 1.0
        )
        device_changed = float(
            len(history) > 0 and str(device) != str(history.iloc[-1]["device_type"])
        )
        location_changed = float(
            len(history) > 0 and str(location) != str(history.iloc[-1]["location"])
        )
        behaviour = min(
            0.45 * amount_deviation
            + 0.20 * recipient_unfamiliarity
            + 0.20 * device_changed
            + 0.15 * location_changed,
            1.0,
        )
        if len(history):
            median_hour = float(pd.to_datetime(history["timestamp"]).dt.hour.median())
        else:
            median_hour = 12.0
        timing = min(abs(int(hour) - median_hour) / 12.0, 1.0)
        prior_categories = set(history["category"].astype(str)) if len(history) else set()
        context_risk = (
            1.0 if category in ["investment", "urgent_transfer"]
            else (0.5 if str(category) not in prior_categories else 0.1)
        )

        ids, parts = intent_drift_score(
            amount_deviation, behaviour, recipient_unfamiliarity,
            timing, context_risk
        )
        fraud_risk = round(
            100.0 * (0.65 * probability + 0.35 * (ids / 100.0)), 1
        )
        decision, message = action(fraud_risk, ids)

        return {
            "risk": fraud_risk,
            "ids": ids,
            "probability": round(100.0 * probability, 4),
            "action": decision,
            "message": message,
            "parts": parts,
            "behaviour_risk": round(behaviour * 100, 1),
            "recipient_risk": round(recipient_unfamiliarity * 100, 1),
            "sequence_anomaly": round(
                min(1.0, 0.55 * timing + 0.45 * behaviour) * 100, 1
            ),
        }
