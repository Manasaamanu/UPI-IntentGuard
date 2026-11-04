import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from config import RAW, ART, REPORTS, DEVICE, FEATURES
from src.data.generate_data import generate
from src.features import fit_preprocessor, transform_arrays, build_recipient_edges
from src.models import IntentGuardNet


def train():
    torch.manual_seed(42)
    np.random.seed(42)

    ART.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW) if RAW.exists() else generate()
    if df.empty or "user_id" not in df or "is_fraud" not in df:
        raise ValueError("Dataset must contain user_id and is_fraud columns.")

    # Split by user before fitting preprocessing to prevent user overlap and scaler leakage.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(
        splitter.split(df, df["is_fraud"], groups=df["user_id"])
    )
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    prep = fit_preprocessor(train_df)
    edge = build_recipient_edges(train_df, prep["rv"])

    seq_tr, ctx_tr, rec_tr, y_tr = transform_arrays(train_df, prep)
    seq_te, ctx_te, rec_te, y_te = transform_arrays(test_df, prep)

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        raise ValueError(
            "The user-level split does not contain both classes in train and test. "
            "Check the dataset's fraud distribution or adjust the split."
        )

    to_tensor = lambda x, dtype=torch.float32: torch.tensor(x, dtype=dtype, device=DEVICE)
    edge_tensor = to_tensor(edge, torch.long)

    model = IntentGuardNet(len(prep["rv"]), len(FEATURES)).to(DEVICE)
    positives = max(int(y_tr.sum()), 1)
    negatives = max(len(y_tr) - positives, 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=to_tensor([negatives / positives])
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    dataset = TensorDataset(
        to_tensor(seq_tr), to_tensor(ctx_tr),
        to_tensor(rec_tr, torch.long), to_tensor(y_tr)
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    epochs = 12
    for epoch in range(epochs):
        model.train()
        losses = []
        for seq_batch, ctx_batch, rec_batch, labels in loader:
            optimizer.zero_grad()
            logits = model(seq_batch, ctx_batch, rec_batch, edge_tensor)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"Epoch {epoch + 1:02d}/{epochs} loss {np.mean(losses):.4f}")

    # Index 0 is the reserved unseen-recipient node. Give it a neutral
    # embedding derived from trained recipient embeddings instead of random weights.
    with torch.no_grad():
        if model.graph.emb.weight.shape[0] > 1:
            model.graph.emb.weight[0].copy_(
                model.graph.emb.weight[1:].mean(dim=0)
            )

    model.eval()
    with torch.no_grad():
        logits = model(
            to_tensor(seq_te), to_tensor(ctx_te),
            to_tensor(rec_te, torch.long), edge_tensor
        )
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_te, predictions)),
        "precision": float(precision_score(y_te, predictions, zero_division=0)),
        "recall": float(recall_score(y_te, predictions, zero_division=0)),
        "f1": float(f1_score(y_te, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, probabilities)),
        "test_samples": int(len(y_te)),
        "train_users": int(train_df["user_id"].nunique()),
        "test_users": int(test_df["user_id"].nunique()),
        "note": (
            "Synthetic demo only. User-level holdout split; these results do not "
            "represent real-world UPI fraud performance."
        ),
    }

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
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS / "classification_report.txt").write_text(
        classification_report(y_te, predictions, zero_division=0)
    )
    pd.DataFrame(
        confusion_matrix(y_te, predictions),
        index=["Actual genuine", "Actual simulated manipulation"],
        columns=["Predicted genuine", "Predicted simulated manipulation"],
    ).to_csv(REPORTS / "confusion_matrix.csv")

    print("\nModel saved successfully:", ART / "intentguard.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
