import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from config import FEATURES, SEQ_LEN


def build_feature_frame(df, recipient_counts=None, global_median=None):
    """Create causal per-user features. Input rows must contain transaction columns."""
    if df.empty:
        return df.copy()

    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    if global_median is None:
        global_median = float(pd.to_numeric(d["amount"]).median())
    if not np.isfinite(global_median):
        global_median = 1.0

    d["hour"] = d["timestamp"].dt.hour.fillna(12).astype(int)
    d["log_amount"] = np.log1p(pd.to_numeric(d["amount"]).clip(lower=0))
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)

    grouped = d.groupby("user_id", sort=False)
    previous_avg = grouped["amount"].transform(
        lambda s: s.shift().expanding().mean()
    )
    d["avg"] = previous_avg.fillna(global_median)
    d["amount_deviation"] = (
        pd.to_numeric(d["amount"]) / (d["avg"] + 1.0)
    ).clip(0, 20)

    d["new_recipient"] = grouped["recipient_id"].transform(
        lambda s: (~s.duplicated()).astype(float)
    )
    d["device_changed"] = grouped["device_type"].transform(
        lambda s: s.ne(s.shift()).fillna(False).astype(float)
    )
    d["location_changed"] = grouped["location"].transform(
        lambda s: s.ne(s.shift()).fillna(False).astype(float)
    )
    gap_hours = grouped["timestamp"].diff().dt.total_seconds() / 3600.0
    d["velocity"] = (1.0 / gap_hours.replace(0, np.nan)).clip(0, 10).fillna(0)

    d["category_unusual"] = grouped["category"].transform(
        lambda s: (~s.duplicated()).astype(float)
    )

    if recipient_counts is None:
        recipient_counts = d["recipient_id"].astype(str).value_counts().to_dict()
    counts = d["recipient_id"].astype(str).map(recipient_counts).fillna(1.0)
    d["recipient_prior"] = (1.0 / counts.clip(lower=1)).astype(float)

    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d


def fit_preprocessor(train_df):
    """Fit feature scaler and lookup statistics on training users only."""
    global_median = float(pd.to_numeric(train_df["amount"]).median())
    recipient_counts = train_df["recipient_id"].astype(str).value_counts().to_dict()
    train_features = build_feature_frame(
        train_df, recipient_counts=recipient_counts, global_median=global_median
    )
    scaler = RobustScaler()
    scaler.fit(train_features[FEATURES].to_numpy(dtype=np.float32))

    # Reserve index 0 for recipients not seen during training.
    recipients = sorted(train_df["recipient_id"].astype(str).unique())
    rv = {"<UNK>": 0}
    rv.update({recipient: i + 1 for i, recipient in enumerate(recipients)})

    prep = {
        "scaler": scaler,
        "recipient_counts": recipient_counts,
        "global_median": global_median,
        "rv": rv,
    }
    return prep


def transform_arrays(df, prep, edge=None):
    """Transform transactions using a previously fitted training-only preprocessor."""
    d = build_feature_frame(
        df,
        recipient_counts=prep["recipient_counts"],
        global_median=prep["global_median"],
    )
    scaled = prep["scaler"].transform(d[FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)
    d.loc[:, FEATURES] = scaled

    rv = prep["rv"]
    seq, ctx, rec, y = [], [], [], []
    for _, group in d.groupby("user_id", sort=False):
        values = group[FEATURES].to_numpy(dtype=np.float32)
        for i in range(len(group)):
            hist = values[max(0, i - SEQ_LEN + 1):i + 1]
            padded = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)
            padded[-len(hist):] = hist
            seq.append(padded)
            ctx.append(values[i])
            rec.append(rv.get(str(group.iloc[i]["recipient_id"]), 0))
            if "is_fraud" in group.columns:
                y.append(int(group.iloc[i]["is_fraud"]))

    if not seq:
        return (
            np.zeros((0, SEQ_LEN, len(FEATURES)), dtype=np.float32),
            np.zeros((0, len(FEATURES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.asarray(seq, dtype=np.float32),
        np.asarray(ctx, dtype=np.float32),
        np.asarray(rec, dtype=np.int64),
        np.asarray(y, dtype=np.int64),
    )


def build_recipient_edges(train_df, rv):
    """Build recipient co-occurrence edges from training users only."""
    edges = set()
    for _, group in train_df.groupby("user_id", sort=False):
        nodes = list(dict.fromkeys(
            rv.get(str(value), 0) for value in group["recipient_id"]
        ))
        nodes = [node for node in nodes if node != 0]
        for source in nodes:
            for target in nodes:
                if source != target:
                    edges.add((source, target))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64).T


# Backward-compatible helper for notebooks/scripts that still call arrays(df).
# This helper fits on the supplied dataframe and should not be used for final evaluation.
def arrays(df):
    prep = fit_preprocessor(df)
    edge = build_recipient_edges(df, prep["rv"])
    seq, ctx, rec, y = transform_arrays(df, prep, edge=edge)
    return seq, ctx, rec, y, edge, prep["rv"], prep["scaler"]
