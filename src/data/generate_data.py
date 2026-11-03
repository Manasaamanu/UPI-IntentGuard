"""
UPI IntentGuard 2.0 - Improved Synthetic Transaction Generator

Purpose
-------
Generate a more challenging synthetic dataset for a research/demo project.
Fraudulent and genuine transactions intentionally overlap in amount, time,
recipient, category, device, and location. No single feature is a guaranteed
fraud indicator.

Output columns are kept compatible with the existing project:
transaction_id, user_id, recipient_id, amount, timestamp, category,
device_type, location, scenario, is_fraud

IMPORTANT
---------
This is synthetic data for experimentation only. It is not real UPI data and
must not be used to claim real-world fraud-detection performance.
"""

import numpy as np
import pandas as pd

from config import RAW


def _choose(rng, values, probabilities=None):
    """Choose one value, optionally using a probability distribution."""
    return rng.choice(values, p=probabilities)


def generate(
    random_seed: int = 42,
    n_users: int = 350,
    transactions_per_user: int = 28,
    fraud_rate: float = 0.06,
):
    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)

    recipients = [f"R{i:03d}" for i in range(250)]
    devices = [f"D{i}" for i in range(7)]
    locations = [f"L{i}" for i in range(22)]
    categories = [
        "food", "groceries", "utilities", "shopping",
        "travel", "transfer", "investment", "urgent_transfer",
    ]
    fraud_scenarios = [
        "fake_support_call", "investment_scam", "impersonation"
    ]

    rows = []

    for user_number in range(n_users):
        user_id = f"U{user_number:04d}"

        # Each user has a personal but imperfectly stable spending pattern.
        typical_amount = float(rng.uniform(250, 4500))
        typical_hour = float(rng.uniform(8, 21))
        preferred_recipients = list(
            rng.choice(recipients, size=8, replace=False)
        )
        preferred_device = f"D{int(rng.integers(0, 4))}"
        preferred_location = f"L{int(rng.integers(0, 15))}"

        current_time = (
            pd.Timestamp("2025-01-01")
            + pd.Timedelta(days=int(rng.integers(0, 60)))
        )

        for transaction_number in range(transactions_per_user):
            # Irregular transaction intervals.
            current_time += pd.Timedelta(
                minutes=int(rng.integers(45, 60 * 30))
            )

            is_fraud = int(rng.random() < fraud_rate)

            # Start from the user's ordinary behaviour. Fraud cases modify
            # some, but not all, of these attributes.
            amount = float(
                rng.lognormal(mean=np.log(typical_amount), sigma=0.65)
            )
            amount = float(np.clip(amount, 30, 30000))

            hour = int(
                np.clip(rng.normal(typical_hour, 3.2), 0, 23)
            )

            if rng.random() < 0.90:
                recipient = str(rng.choice(preferred_recipients))
            else:
                recipient = str(rng.choice(recipients))

            device = (
                preferred_device
                if rng.random() < 0.94
                else str(rng.choice(devices))
            )
            location = (
                preferred_location
                if rng.random() < 0.94
                else str(rng.choice(locations))
            )
            category = str(
                rng.choice(
                    categories,
                    p=[0.16, 0.15, 0.10, 0.18, 0.12, 0.17, 0.06, 0.06],
                )
            )
            scenario = "normal"

            if is_fraud:
                scenario = str(rng.choice(fraud_scenarios))

                # Fraud signals are probabilistic, not deterministic.
                # Some fraudulent transactions deliberately look ordinary.
                fraud_style = str(
                    rng.choice(
                        ["recipient", "amount", "context", "mixed", "subtle"],
                        p=[0.24, 0.20, 0.16, 0.28, 0.12],
                    )
                )

                if fraud_style in ("recipient", "mixed"):
                    if rng.random() < 0.72:
                        recipient = str(
                            rng.choice(
                                [r for r in recipients
                                 if r not in preferred_recipients]
                            )
                        )

                if fraud_style in ("amount", "mixed"):
                    # Multipliers overlap with genuine large payments.
                    multiplier = float(rng.choice(
                        [1.4, 2.0, 3.0, 4.5],
                        p=[0.25, 0.35, 0.28, 0.12],
                    ))
                    amount = float(
                        np.clip(
                            typical_amount * multiplier
                            * rng.lognormal(0, 0.35),
                            50, 35000
                        )
                    )

                if fraud_style in ("context", "mixed"):
                    if rng.random() < 0.48:
                        hour = int(rng.choice([0, 1, 2, 3, 22, 23]))
                    if rng.random() < 0.40:
                        device = str(rng.choice(devices))
                    if rng.random() < 0.35:
                        location = str(rng.choice(locations))

                if fraud_style == "subtle":
                    # Keep all attributes close to normal behaviour.
                    amount = float(
                        np.clip(
                            rng.lognormal(np.log(typical_amount), 0.75),
                            30, 18000
                        )
                    )
                    hour = int(
                        np.clip(rng.normal(typical_hour, 3.5), 0, 23)
                    )
                    if rng.random() < 0.82:
                        recipient = str(rng.choice(preferred_recipients))

                # Scam categories overlap with normal categories.
                if scenario == "investment_scam" and rng.random() < 0.60:
                    category = str(rng.choice(
                        ["investment", "transfer", "shopping", "utilities"]
                    ))
                elif scenario == "fake_support_call" and rng.random() < 0.60:
                    category = str(rng.choice(
                        ["transfer", "shopping", "utilities", "food"]
                    ))
                elif scenario == "impersonation" and rng.random() < 0.60:
                    category = str(rng.choice(
                        ["transfer", "shopping", "travel", "groceries"]
                    ))

            else:
                # Genuine transactions sometimes contain anomalies too:
                # new recipients, high amounts, unusual hours, new devices,
                # and unfamiliar locations.
                if rng.random() < 0.07:
                    recipient = str(
                        rng.choice(
                            [r for r in recipients
                             if r not in preferred_recipients]
                        )
                    )

                if rng.random() < 0.045:
                    amount = float(
                        np.clip(
                            typical_amount * rng.uniform(2.0, 5.0),
                            50, 35000
                        )
                    )

                if rng.random() < 0.10:
                    hour = int(rng.choice([0, 1, 2, 3, 22, 23]))

                if rng.random() < 0.045:
                    device = str(rng.choice(devices))

                if rng.random() < 0.045:
                    location = str(rng.choice(locations))

                # Genuine users can make investment or urgent transfers.
                if rng.random() < 0.035:
                    category = str(rng.choice(
                        ["investment", "urgent_transfer"]
                    ))

            timestamp = current_time.replace(hour=int(hour))

            rows.append({
                "transaction_id": f"T{user_number}_{transaction_number}",
                "user_id": user_id,
                "recipient_id": str(recipient),
                "amount": round(float(amount), 2),
                "timestamp": timestamp.isoformat(),
                "category": str(category),
                "device_type": str(device),
                "location": str(location),
                "scenario": scenario,
                "is_fraud": int(is_fraud),
            })

    df = (
        pd.DataFrame(rows)
        .sort_values(["user_id", "timestamp"])
        .reset_index(drop=True)
    )

    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW, index=False)

    print(f"Created {len(df):,} synthetic transactions at {RAW}")
    print("\nFraud label distribution:")
    print(df["is_fraud"].value_counts().rename(
        index={0: "Genuine", 1: "Simulated fraud"}
    ))
    print("\nFraud rate by scenario:")
    print(
        df.groupby("scenario")["is_fraud"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "fraud_rate"})
        .to_string()
    )
    print("\nDataset saved. This is synthetic demo data only.")

    return df


if __name__ == "__main__":
    generate()
