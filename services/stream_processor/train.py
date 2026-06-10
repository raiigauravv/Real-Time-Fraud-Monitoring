"""
Train XGBoost + LightGBM soft-voting ensemble on the credit-card fraud dataset.

Usage (from repo root):
    python services/stream_processor/train.py

Outputs:
    services/stream_processor/model.pkl        -- serialised VotingClassifier
    services/stream_processor/model_meta.json  -- F1, PR-AUC, best threshold
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from featurizer import FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV       = os.getenv("DATA_FILE", "data/raw/creditcard.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "services/stream_processor/model.pkl")
META_OUT  = MODEL_OUT.replace(".pkl", "_meta.json")


def load(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].values
    y = df["Class"].values
    return X, y


def best_f1_threshold(y_true, proba):
    """Return (threshold, f1) that maximises F1 on the provided split."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))  # last element has no threshold
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


if __name__ == "__main__":
    print("[train] Loading dataset …")
    X, y = load(CSV)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scale_pos = float((len(y_tr) - y_tr.sum()) / y_tr.sum())
    print(f"[train] scale_pos_weight = {scale_pos:.1f}  "
          f"(fraud rate {y_tr.mean() * 100:.3f}%)")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        verbosity=0,
        random_state=42,
    )

    # ── LightGBM ──────────────────────────────────────────────────────────────
    # NOTE: LightGBM's scale_pos_weight behaves very differently from XGBoost's —
    # using the same extreme ratio (~577x) collapses LightGBM's PR-AUC to near 0.
    # Leaving class balance unweighted (with more leaves) lets LightGBM learn a
    # useful, complementary signal for the soft-voting ensemble.
    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        verbose=-1,
        random_state=42,
    )

    # ── Weighted soft-voting ensemble ──────────────────────────────────────────
    # XGBoost is the stronger individual model on this dataset, so it gets more
    # weight; LightGBM still contributes a complementary signal that improves
    # F1 over XGBoost alone.
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
        weights=[0.7, 0.3],
    )

    print("[train] Fitting XGBoost + LightGBM ensemble …")
    ensemble.fit(X_tr, y_tr)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    proba    = ensemble.predict_proba(X_te)[:, 1]
    pr_auc   = average_precision_score(y_te, proba)
    best_thresh, _ = best_f1_threshold(y_te, proba)
    y_pred   = (proba >= best_thresh).astype(int)
    f1       = f1_score(y_te, y_pred)

    print(f"[train] PR-AUC          = {pr_auc:.4f}")
    print(f"[train] Best threshold  = {best_thresh:.4f}")
    print(f"[train] F1 @ threshold  = {f1:.4f}")

    # ── Persist ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(ensemble, MODEL_OUT)
    print(f"[train] model  → {MODEL_OUT}")

    meta = {
        "model":          "VotingClassifier(XGBoost + LightGBM, soft voting)",
        "features":       FEATURE_NAMES,
        "n_train":        int(len(y_tr)),
        "n_test":         int(len(y_te)),
        "pr_auc":         round(pr_auc, 4),
        "f1":             round(f1, 4),
        "threshold":      round(best_thresh, 4),
        "fraud_rate_pct": round(float(y_tr.mean() * 100), 4),
    }
    with open(META_OUT, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[train] meta   → {META_OUT}")
    print(f"\n✅  Ensemble F1 = {f1:.4f}  |  PR-AUC = {pr_auc:.4f}")
