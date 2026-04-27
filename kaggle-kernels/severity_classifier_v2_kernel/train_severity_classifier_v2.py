"""MH-2026-023 — Aegis Severity Classifier v2 (Human-Label Retrain).

Key differences from v1 (MH-2026-022):
- Dataset: corpus_v1.jsonl (189 human-labeled findings) instead of mixed 503-row CSV
- Labels: ONLY human_label as target — no rule_severity fallback
  Rationale: v1 was trained with 90% rule_severity (auto-derived) labels.
  Real accuracy against human labels was 46.9% (vs 96.6% CV on rule-derived).
  v2 aims to close this gap by training purely on human judgment.
- Holdout eval: 40-sample holdout from corpus_v1 (last 40 rows, CTO triage)
  v1 accuracy on same holdout: 46.9%
- Target: CV accuracy >= 0.60 (realistic given MEDIUM class only 12 samples)
- Status: shadow — 7d shadow observation before promote to production

Architecture (same as v1):
    Features: TF-IDF on (title + evidence + module + asset) — 5000 max_features
              + module one-hot
              + rule_severity ordinal (kept as feature even though not used as label)
    Model:    XGBoost multi-class classifier
    Eval:     5-fold StratifiedKFold CV

Dataset:
    /kaggle/input/aegis-severity-corpus-v2/corpus_v1.jsonl
    189 rows — all human-labeled
    Format: JSONL {"title": ..., "module": ..., "asset": ..., "evidence": ...,
                   "rule_severity": ..., "human_label": ..., "features": {...}}

Output:
    /kaggle/working/model.pkl       — trained model bundle
    /kaggle/working/result.json     — full results for modelhub poller (req_no: MH-2026-023)
    /kaggle/working/metrics.json    — shorthand metrics
    /kaggle/working/holdout_eval.json — 40-sample holdout vs v1 comparison
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATASET_PATH = Path("/kaggle/input/aegis-severity-corpus-v2/corpus_v1.jsonl")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
TARGET_ACCURACY = 0.60   # Realistic target for 189 samples, human labels
N_FOLDS = 5
SEED = 42
HOLDOUT_SIZE = 40        # Last 40 rows reserved for holdout eval

# v1 baseline for comparison
V1_BASELINE_ACCURACY = 0.469

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
print("Loading dataset (corpus_v1.jsonl)...")
rows: list[dict] = []
with DATASET_PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"  Total rows: {len(rows)}")

# v2: ONLY human_label — skip rows without it
valid_rows = [r for r in rows if r.get("human_label", "").strip()]
print(f"  Rows with human_label: {len(valid_rows)}")

if len(valid_rows) < 20:
    raise ValueError(
        f"Insufficient human-labeled samples: {len(valid_rows)}. "
        "Upload corpus_v1.jsonl as Kaggle dataset 'aegis-severity-corpus-v2'."
    )

# Split holdout (last HOLDOUT_SIZE rows) before any shuffling
holdout_rows = valid_rows[-HOLDOUT_SIZE:]
train_rows = valid_rows[:-HOLDOUT_SIZE]
print(f"  Train: {len(train_rows)}, Holdout: {len(holdout_rows)}")

# Label distribution
label_dist = Counter(r["human_label"] for r in valid_rows)
train_label_dist = Counter(r["human_label"] for r in train_rows)
print(f"  Label distribution (all): {dict(sorted(label_dist.items()))}")
print(f"  Label distribution (train): {dict(sorted(train_label_dist.items()))}")


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────

def build_text_feature(row: dict) -> str:
    """Combine title + evidence text + module + asset."""
    title = row.get("title", "")
    module = row.get("module", "")
    asset = row.get("asset", "")
    # Use pre-computed features.text if available (CorpusBuilder output)
    feat = row.get("features", {})
    if isinstance(feat, dict) and feat.get("text"):
        return feat["text"]
    evidence_text = ""
    try:
        ev = row.get("evidence") or {}
        if isinstance(ev, str):
            ev = json.loads(ev)
        ev_parts = []
        for k, v in ev.items():
            if isinstance(v, (str, int, float)):
                ev_parts.append(f"{k} {v}")
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        ev_parts.append(item.get("service", ""))
                        ev_parts.append(item.get("product", ""))
        evidence_text = " ".join(ev_parts)
    except Exception:
        evidence_text = str(row.get("evidence", ""))[:200]
    return f"{title} {module} {asset} {evidence_text}".strip()


def build_features(
    subset: list[dict],
    tfidf: TfidfVectorizer | None = None,
    module_to_idx: dict[str, int] | None = None,
    fit: bool = False,
) -> tuple:
    """Build feature matrix. Returns (X, tfidf, module_to_idx)."""
    texts = [build_text_feature(r) for r in subset]

    if fit:
        # Fit TF-IDF on training data only
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        X_tfidf = tfidf.fit_transform(texts)

        all_modules = sorted(set(r.get("module", "") for r in subset))
        module_to_idx = {m: i for i, m in enumerate(all_modules)}
    else:
        X_tfidf = tfidf.transform(texts)

    n_modules = len(module_to_idx)
    module_matrix = np.zeros((len(subset), n_modules), dtype=np.float32)
    for i, row in enumerate(subset):
        idx = module_to_idx.get(row.get("module", ""), -1)
        if idx >= 0:
            module_matrix[i, idx] = 1.0

    # Rule severity ordinal (kept as feature)
    rule_sev_ordinal = np.array(
        [SEVERITY_ORDER.get(r.get("rule_severity", "MEDIUM").strip(), 2) for r in subset],
        dtype=np.float32,
    ).reshape(-1, 1)

    X = hstack([X_tfidf, csr_matrix(module_matrix), csr_matrix(rule_sev_ordinal)])
    return X, tfidf, module_to_idx


print("\nBuilding training features...")
X_train_full, tfidf, module_to_idx = build_features(train_rows, fit=True)
print(f"  Training feature shape: {X_train_full.shape}")

# Label encoding
effective_labels_train = [r["human_label"] for r in train_rows]
le = LabelEncoder()
y_train_full = le.fit_transform(effective_labels_train)
classes = le.classes_
print(f"  Classes: {list(classes)}")

# Class weights (handle imbalanced: INFO dominant)
class_weights_arr = compute_class_weight("balanced", classes=np.unique(y_train_full), y=y_train_full)
weight_map = dict(enumerate(class_weights_arr))
sample_weights_train = np.array([weight_map[yi] for yi in y_train_full])
print(f"  Class weights: {dict(zip(classes, class_weights_arr.round(3)))}")


# ──────────────────────────────────────────────
# 5-fold CV on training set
# ──────────────────────────────────────────────
print(f"\n{N_FOLDS}-fold Stratified CV (on {len(train_rows)} training samples)...")
X_arr = X_train_full.toarray()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_accuracies: list[float] = []
fold_f1s: list[float] = []

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_train_full)):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_train_full[tr_idx], y_train_full[val_idx]
    sw_tr = sample_weights_train[tr_idx]

    xgb_fold = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        objective="multi:softprob",
        num_class=len(classes),
    )
    xgb_fold.fit(X_tr, y_tr, sample_weight=sw_tr)
    y_pred = xgb_fold.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    fold_accuracies.append(acc)
    fold_f1s.append(f1)
    print(f"  Fold {fold_idx + 1}: accuracy={acc:.4f}, f1_weighted={f1:.4f}")

cv_mean_acc = float(np.mean(fold_accuracies))
cv_std_acc = float(np.std(fold_accuracies))
cv_mean_f1 = float(np.mean(fold_f1s))
print(f"\n  CV accuracy: {cv_mean_acc:.4f} +/- {cv_std_acc:.4f}")
print(f"  CV f1_weighted: {cv_mean_f1:.4f}")
print(f"  Target accuracy: {TARGET_ACCURACY}")
pass_fail_cv = "pass" if cv_mean_acc >= TARGET_ACCURACY else "fail"
print(f"  CV Result: {pass_fail_cv.upper()}")


# ──────────────────────────────────────────────
# Final model (fit on all training data)
# ──────────────────────────────────────────────
print("\nFitting final model on full training set...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
    objective="multi:softprob",
    num_class=len(classes),
)
xgb.fit(X_arr, y_train_full, sample_weight=sample_weights_train)

# Train-set metrics (expected high — for reference)
y_pred_train = xgb.predict(X_arr)
train_report = classification_report(
    y_train_full, y_pred_train,
    target_names=classes,
    output_dict=True,
    zero_division=0,
)
print("\nPer-class metrics (train set, expected high):")
for cls in classes:
    p = train_report[cls]["precision"]
    r = train_report[cls]["recall"]
    f = train_report[cls]["f1-score"]
    s = int(train_report[cls]["support"])
    print(f"  {cls:<10} precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}")


# ──────────────────────────────────────────────
# Holdout evaluation (40-sample, real accuracy)
# ──────────────────────────────────────────────
print(f"\nHoldout evaluation ({len(holdout_rows)} samples)...")
X_holdout, _, _ = build_features(holdout_rows, tfidf=tfidf, module_to_idx=module_to_idx, fit=False)
y_holdout_true_labels = [r["human_label"] for r in holdout_rows]
# Handle labels not seen during training (map to closest)
y_holdout = []
for lbl in y_holdout_true_labels:
    if lbl in le.classes_:
        y_holdout.append(le.transform([lbl])[0])
    else:
        y_holdout.append(-1)  # OOV

valid_holdout_mask = [i for i, v in enumerate(y_holdout) if v != -1]
X_ho_valid = X_holdout[valid_holdout_mask]
y_ho_valid = np.array([y_holdout[i] for i in valid_holdout_mask])

if len(y_ho_valid) > 0:
    y_ho_pred = xgb.predict(X_ho_valid.toarray())
    holdout_accuracy = accuracy_score(y_ho_valid, y_ho_pred)
    holdout_f1 = f1_score(y_ho_valid, y_ho_pred, average="weighted", zero_division=0)
    holdout_report = classification_report(
        y_ho_valid, y_ho_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    print(f"  Holdout accuracy: {holdout_accuracy:.4f}")
    print(f"  Holdout F1-weighted: {holdout_f1:.4f}")
    print(f"  v1 baseline accuracy: {V1_BASELINE_ACCURACY:.4f}")
    print(f"  Delta vs v1: {holdout_accuracy - V1_BASELINE_ACCURACY:+.4f}")
    print("\nPer-class holdout metrics:")
    for cls in classes:
        if cls in holdout_report:
            p = holdout_report[cls]["precision"]
            r = holdout_report[cls]["recall"]
            f = holdout_report[cls]["f1-score"]
            s = int(holdout_report[cls]["support"])
            print(f"  {cls:<10} precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}")
else:
    holdout_accuracy = None
    holdout_f1 = None
    holdout_report = {}
    print("  WARNING: No valid holdout samples (all OOV classes)")


# ──────────────────────────────────────────────
# Save model
# ──────────────────────────────────────────────
model_bundle = {
    "tfidf": tfidf,
    "xgb": xgb,
    "label_encoder": le,
    "module_to_idx": module_to_idx,
    "severity_order": SEVERITY_ORDER,
    "classes": list(classes),
    "feature_shape": X_train_full.shape,
    "version": "v2",
    "req_no": "MH-2026-023",
    "training_label_type": "human_only",
}

model_path = OUTPUT_DIR / "model.pkl"
with model_path.open("wb") as f:
    pickle.dump(model_bundle, f)
print(f"\nModel saved: {model_path} ({model_path.stat().st_size // 1024} KB)")


# ──────────────────────────────────────────────
# Save result.json (modelhub poller format)
# ──────────────────────────────────────────────
per_class_metrics: dict[str, dict] = {}
for cls in classes:
    per_class_metrics[cls] = {
        "train_precision": round(train_report[cls]["precision"], 4),
        "train_recall": round(train_report[cls]["recall"], 4),
        "train_f1": round(train_report[cls]["f1-score"], 4),
        "train_support": int(train_report[cls]["support"]),
        "holdout_f1": round(holdout_report.get(cls, {}).get("f1-score", 0.0), 4),
        "holdout_support": int(holdout_report.get(cls, {}).get("support", 0)),
    }

result = {
    "req_no": "MH-2026-023",
    "product": "Aegis",
    "arch": "xgboost-text",
    "model_type": "text-classifier",
    "dataset": "aegis-severity-corpus-v2",
    "num_samples": len(valid_rows),
    "train_samples": len(train_rows),
    "holdout_samples": len(holdout_rows),
    "label_type": "human_only",
    "label_sources": {"human": len(valid_rows), "rule": 0},
    "label_distribution": {k: int(v) for k, v in label_dist.items()},
    "classes": list(classes),
    "cv_folds": N_FOLDS,
    "cv_accuracy_mean": round(cv_mean_acc, 4),
    "cv_accuracy_std": round(cv_std_acc, 4),
    "cv_f1_weighted_mean": round(cv_mean_f1, 4),
    "target_accuracy": TARGET_ACCURACY,
    "pass_fail": pass_fail_cv,
    "holdout_accuracy": round(holdout_accuracy, 4) if holdout_accuracy is not None else None,
    "holdout_f1_weighted": round(holdout_f1, 4) if holdout_f1 is not None else None,
    "v1_baseline_accuracy": V1_BASELINE_ACCURACY,
    "accuracy_delta_vs_v1": (
        round(holdout_accuracy - V1_BASELINE_ACCURACY, 4)
        if holdout_accuracy is not None else None
    ),
    "per_class_metrics": per_class_metrics,
    "map50": None,          # N/A for text classifiers
    "map50_95": None,
    "accuracy": round(cv_mean_acc, 4),
    "model_path": str(model_path),
    "model_size_bytes": model_path.stat().st_size,
    "model_status": "shadow",  # 7d shadow observation before promote
}

result_path = OUTPUT_DIR / "result.json"
with result_path.open("w") as f:
    json.dump(result, f, indent=2)
print(f"result.json saved: {result_path}")

# holdout_eval.json (separate detail)
holdout_eval = {
    "req_no": "MH-2026-023",
    "holdout_size": len(holdout_rows),
    "v1_baseline_accuracy": V1_BASELINE_ACCURACY,
    "v2_holdout_accuracy": round(holdout_accuracy, 4) if holdout_accuracy is not None else None,
    "accuracy_delta": (
        round(holdout_accuracy - V1_BASELINE_ACCURACY, 4)
        if holdout_accuracy is not None else None
    ),
    "improvement": (
        "IMPROVED" if holdout_accuracy and holdout_accuracy > V1_BASELINE_ACCURACY
        else "REGRESSED" if holdout_accuracy and holdout_accuracy < V1_BASELINE_ACCURACY
        else "UNKNOWN"
    ),
    "per_class_holdout": {
        cls: {
            "f1": round(holdout_report.get(cls, {}).get("f1-score", 0.0), 4),
            "precision": round(holdout_report.get(cls, {}).get("precision", 0.0), 4),
            "recall": round(holdout_report.get(cls, {}).get("recall", 0.0), 4),
            "support": int(holdout_report.get(cls, {}).get("support", 0)),
        }
        for cls in classes
    },
}
holdout_path = OUTPUT_DIR / "holdout_eval.json"
with holdout_path.open("w") as f:
    json.dump(holdout_eval, f, indent=2)
print(f"holdout_eval.json saved: {holdout_path}")

# metrics.json
metrics = {
    "accuracy": round(cv_mean_acc, 4),
    "f1_weighted": round(cv_mean_f1, 4),
    "cv_accuracy_std": round(cv_std_acc, 4),
    "holdout_accuracy": round(holdout_accuracy, 4) if holdout_accuracy is not None else None,
    "pass_fail": pass_fail_cv,
    "num_samples": len(valid_rows),
    "classes": list(classes),
}
metrics_path = OUTPUT_DIR / "metrics.json"
with metrics_path.open("w") as f:
    json.dump(metrics, f, indent=2)
print(f"metrics.json saved: {metrics_path}")


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("MH-2026-023 Training Summary — Severity Classifier v2")
print("=" * 60)
print(f"  Training samples: {len(train_rows)}")
print(f"  Holdout samples:  {len(holdout_rows)}")
print(f"  Label type:       human_only")
print(f"  CV Accuracy:      {cv_mean_acc:.4f} ± {cv_std_acc:.4f}")
print(f"  CV F1-weighted:   {cv_mean_f1:.4f}")
print(f"  Holdout Accuracy: {holdout_accuracy:.4f}" if holdout_accuracy else "  Holdout: N/A")
print(f"  v1 Baseline:      {V1_BASELINE_ACCURACY:.4f}")
if holdout_accuracy:
    delta = holdout_accuracy - V1_BASELINE_ACCURACY
    print(f"  Delta vs v1:      {delta:+.4f} ({'IMPROVED' if delta > 0 else 'REGRESSED'})")
print(f"  Target:           {TARGET_ACCURACY}")
print(f"  CV PASS/FAIL:     {pass_fail_cv.upper()}")
print(f"  Status:           shadow (7d observation)")
print("=" * 60)
