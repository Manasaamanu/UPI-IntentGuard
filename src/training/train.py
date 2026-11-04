
import copy
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

from config import RAW, ART, REPORTS, DEVICE, FEATURES
from src.data.generate_data import generate
from src.features import (
    fit_preprocessor,
    transform_arrays,
    build_recipient_edges,
)
from src.models import IntentGuardNet


def train():
    torch.manual_seed(42)
    np.random.seed(42)

    ART.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate dataset
    df = pd.read_csv(RAW) if RAW.exists() else generate()

    if df.empty or "user_id" not in df or "is_fraud" not in df:
        raise ValueError("Dataset must contain user_id and is_fraud columns.")

    if df["user_id"].isna().any():
        raise ValueError("Dataset contains missing user_id values.")

    # 2. Split by user: 70% train, 15% validation, 15% test
    split1 = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=42
    )
    train_val_idx, test_idx = next(
        split1.split(df, groups=df["user_id"])
    )

    train_val = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    split2 = GroupShuffleSplit(
        n_splits=1, test_size=0.1765, random_state=42
    )
    train_idx, val_idx = next(
        split2.split(train_val, groups=train_val["user_id"])
    )

    train_df = train_val.iloc[train_idx].copy()
    val_df = train_val.iloc[val_idx].copy()

    # Confirm that users do not overlap
    train_users = set(train_df["user_id"])
    val_users = set(val_df["user_id"])
    test_users = set(test_df["user_id"])

    assert not (train_users & val_users)
    assert not (train_users & test_users)
    assert not (val_users & test_users)

    # 3. Fit preprocessing using training data only
    prep = fit_preprocessor(train_df)
    edge = build_recipient_edges(train_df, prep["rv"])

    seq_tr, ctx_tr, rec_tr, y_tr = transform_arrays(train_df, prep)
    seq_val, ctx_val, rec_val, y_val = transform_arrays(val_df, prep)
    seq_te, ctx_te, rec_te, y_te = transform_arrays(test_df, prep)

    for name, labels in [
        ("Training", y_tr),
        ("Validation", y_val),
        ("Test", y_te),
    ]:
        if len(np.unique(labels)) < 2:
            raise ValueError(
                f"{name} dataset must contain both genuine and fraud classes."
            )

    to_tensor = lambda x, dtype=torch.float32: torch.tensor(
        x, dtype=dtype, device=DEVICE
    )

    edge_tensor = to_tensor(edge, torch.long)

    # 4. Build model and handle class imbalance
    model = IntentGuardNet(
        len(prep["rv"]), len(FEATURES)
    ).to(DEVICE)

    positives = max(int(y_tr.sum()), 1)
    negatives = max(len(y_tr) - positives, 1)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=to_tensor([negatives / positives])
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    dataset = TensorDataset(
        to_tensor(seq_tr),
        to_tensor(ctx_tr),
        to_tensor(rec_tr, torch.long),
        to_tensor(y_tr),
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
    )

    # Validation tensors
    val_seq = to_tensor(seq_val)
    val_ctx = to_tensor(ctx_val)
    val_rec = to_tensor(rec_val, torch.long)

    # 5. Early stopping based on validation PR-AUC
    max_epochs = 30
    patience = 5
    best_val_pr_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    history = []

    for epoch in range(max_epochs):
        model.train()
        losses = []

        for seq_batch, ctx_batch, rec_batch, labels in loader:
            optimizer.zero_grad()

            logits = model(
                seq_batch,
                ctx_batch,
                rec_batch,
                edge_tensor,
            )

            loss = loss_fn(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()
            losses.append(float(loss.item()))

        # Neutral embedding for unseen recipients
        with torch.no_grad():
            if model.graph.emb.weight.shape[0] > 1:
                model.graph.emb.weight[0].copy_(
                    model.graph.emb.weight[1:].mean(dim=0)
                )

        # Evaluate on validation data
        model.eval()

        with torch.no_grad():
            val_logits = model(
                val_seq,
                val_ctx,
                val_rec,
                edge_tensor,
            )
            val_probabilities = torch.sigmoid(
                val_logits
            ).cpu().numpy()

        val_pr_auc = average_precision_score(
            y_val, val_probabilities
        )
        val_roc_auc = roc_auc_score(
            y_val, val_probabilities
        )

        train_loss = float(np.mean(losses))

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_pr_auc": float(val_pr_auc),
            "validation_roc_auc": float(val_roc_auc),
        })

        print(
            f"Epoch {epoch + 1:02d}/{max_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Validation PR-AUC: {val_pr_auc:.4f} | "
            f"Validation ROC-AUC: {val_roc_auc:.4f}"
        )

        if val_pr_auc > best_val_pr_auc + 1e-5:
            best_val_pr_auc = float(val_pr_auc)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")

    # 6. Restore best model
    model.load_state_dict(best_state)
    model.eval()

    # 7. Select classification threshold using validation data only
    with torch.no_grad():
        val_logits = model(
            val_seq,
            val_ctx,
            val_rec,
            edge_tensor,
        )
        val_probabilities = torch.sigmoid(
            val_logits
        ).cpu().numpy()

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_probabilities
    )

    if len(thresholds) == 0:
        selected_threshold = 0.5
    else:
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-10)
        )
        selected_threshold = float(
            thresholds[int(np.argmax(f1_values))]
        )

    # 8. Final evaluation on untouched test data
    with torch.no_grad():
        test_logits = model(
            to_tensor(seq_te),
            to_tensor(ctx_te),
            to_tensor(rec_te, torch.long),
            edge_tensor,
        )
        test_probabilities = torch.sigmoid(
            test_logits
        ).cpu().numpy()

    predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_te, predictions)),
        "precision": float(
            precision_score(y_te, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_te, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_te, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_te, test_probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_te, test_probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_te, test_probabilities)
        ),
        "threshold_selected_on_validation": selected_threshold,
        "best_validation_pr_auc": best_val_pr_auc,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "test_samples": int(len(y_te)),
        "train_users": int(len(train_users)),
        "validation_users": int(len(val_users)),
        "test_users": int(len(test_users)),
        "confusion_matrix": {
            "tn": int(confusion_matrix(y_te, predictions, labels=[0, 1])[0, 0]),
            "fp": int(confusion_matrix(y_te, predictions, labels=[0, 1])[0, 1]),
            "fn": int(confusion_matrix(y_te, predictions, labels=[0, 1])[1, 0]),
            "tp": int(confusion_matrix(y_te, predictions, labels=[0, 1])[1, 1]),
        },
        "note": (
            "Synthetic academic prototype only. User-level train, validation "
            "and test split. Threshold selected on validation data. "
            "Not evidence of real-world UPI fraud performance."
        ),
    }

    # 9. Save model checkpoint using the existing app-compatible structure
    checkpoint = {
        "state": model.state_dict(),
        "n": len(prep["rv"]),
        "dim": len(FEATURES),
        "rv": prep["rv"],
        "scaler": prep["scaler"],
        "recipient_counts": prep["recipient_counts"],
        "global_median": prep["global_median"],
        "edge": torch.as_tensor(edge, dtype=torch.long).cpu(),
        "features": FEATURES,
    }

    torch.save(checkpoint, ART / "intentguard.pt")

    # Save metrics and reports
    (ART / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(history).to_csv(
        REPORTS / "training_history.csv",
        index=False,
    )

    (REPORTS / "classification_report.txt").write_text(
        classification_report(
            y_te,
            predictions,
            target_names=["Genuine", "Simulated manipulation"],
            zero_division=0,
        ),
        encoding="utf-8",
    )

    cm = confusion_matrix(y_te, predictions, labels=[0, 1])

    pd.DataFrame(
        cm,
        index=["Actual genuine", "Actual simulated manipulation"],
        columns=["Predicted genuine", "Predicted simulated manipulation"],
    ).to_csv(REPORTS / "confusion_matrix.csv")

    pd.DataFrame([metrics]).to_csv(
        REPORTS / "test_metrics_summary.csv",
        index=False,
    )

    (REPORTS / "metrics_summary.txt").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nTraining completed successfully!")
    print("Best epoch:", best_epoch)
    print("Selected threshold:", round(selected_threshold, 4))
    print("Model saved:", ART / "intentguard.pt")
    print("\nFinal test metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
