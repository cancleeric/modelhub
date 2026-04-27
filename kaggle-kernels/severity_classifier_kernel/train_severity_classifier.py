"""MH-2026-022 — Aegis Severity Auto-classifier (XGBoost + TF-IDF).

Architecture:
    Features: TF-IDF on (title + evidence + module + asset) — 5000 max_features
              + module one-hot (11 modules)
              + severity ordinal from rule_severity
    Model:    XGBoost multi-class classifier (5 classes: CRITICAL/HIGH/MEDIUM/LOW/INFO)
    Eval:     5-fold StratifiedKFold cross-validation
    Target:   accuracy >= 0.85

Dataset:
    /kaggle/input/aegis-severity-corpus/severity_corpus.csv
    503 rows: 50 human_label + 453 rule_severity (mixed labels)
    Columns: id, asset, module, title, evidence, rule_severity, human_label

Output:
    /kaggle/working/model.pkl       — trained model (Pipeline: TF-IDF + XGBoost)
    /kaggle/working/result.json     — full results for modelhub poller
    /kaggle/working/metrics.json    — shorthand metrics
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATASET_PATH = Path("/kaggle/input/aegis-severity-corpus/severity_corpus.csv")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
TARGET_ACCURACY = 0.85
N_FOLDS = 5
SEED = 42

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
print("Loading dataset...")
rows: list[dict] = []
with DATASET_PATH.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"  Total rows: {len(rows)}")

# Build effective label: human_label if available, else rule_severity
effective_labels: list[str] = []
label_sources: dict[str, int] = {"human": 0, "rule": 0}
for row in rows:
    hl = row.get("human_label", "").strip()
    if hl:
        effective_labels.append(hl)
        label_sources["human"] += 1
    else:
        effective_labels.append(row["rule_severity"].strip())
        label_sources["rule"] += 1

print(f"  Label sources: human={label_sources['human']}, rule={label_sources['rule']}")

from collections import Counter
label_dist = Counter(effective_labels)
print(f"  Label distribution: {dict(sorted(label_dist.items()))}")


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────

def build_text_feature(row: dict) -> str:
    """Combine title + evidence + module + asset into single text."""
    title = row.get("title", "")
    module = row.get("module", "")
    asset = row.get("asset", "")
    # Parse evidence JSON for key terms
    evidence_text = ""
    try:
        ev = json.loads(row.get("evidence") or "{}")
        # Flatten important values: service names, ports, versions
        ev_parts = []
        for k, v in ev.items():
            if isinstance(v, (str, int, float)):
                ev_parts.append(f"{k} {v}")
            elif isinstance(v, list) and len(v) < 20:
                # For open_ports list, extract service names
                for item in v:
                    if isinstance(item, dict):
                        ev_parts.append(item.get("service", ""))
                        ev_parts.append(item.get("product", ""))
        evidence_text = " ".join(ev_parts)
    except Exception:
        evidence_text = str(row.get("evidence", ""))[:200]

    return f"{title} {module} {asset} {evidence_text}".strip()


print("\nBuilding features...")
texts = [build_text_feature(r) for r in rows]

# Module one-hot encoding
all_modules = sorted(set(r["module"] for r in rows))
module_to_idx = {m: i for i, m in enumerate(all_modules)}
print(f"  Modules ({len(all_modules)}): {all_modules}")

module_matrix = np.zeros((len(rows), len(all_modules)), dtype=np.float32)
for i, row in enumerate(rows):
    idx = module_to_idx.get(row["module"], -1)
    if idx >= 0:
        module_matrix[i, idx] = 1.0

# Rule severity ordinal feature (1D)
rule_sev_ordinal = np.array(
    [SEVERITY_ORDER.get(r["rule_severity"].strip(), 2) for r in rows],
    dtype=np.float32,
).reshape(-1, 1)

# ──────────────────────────────────────────────
# Label encoding
# ──────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(effective_labels)
classes = le.classes_
print(f"  Classes: {list(classes)}")

# Compute class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
weight_map = dict(enumerate(class_weights))
sample_weights = np.array([weight_map[yi] for yi in y])
print(f"  Class weights: {dict(zip(classes, class_weights.round(3)))}")

# ──────────────────────────────────────────────
# TF-IDF vectorizer
# ──────────────────────────────────────────────
print("\nFitting TF-IDF vectorizer...")
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
print(f"  TF-IDF shape: {X_tfidf.shape}")

# Combine all features
X_module = csr_matrix(module_matrix)
X_rule_sev = csr_matrix(rule_sev_ordinal)
X = hstack([X_tfidf, X_module, X_rule_sev])
print(f"  Combined feature shape: {X.shape}")

# ──────────────────────────────────────────────
# XGBoost model
# ──────────────────────────────────────────────
print("\nTraining XGBoost classifier...")
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

# ──────────────────────────────────────────────
# 5-fold CV
# ──────────────────────────────────────────────
print(f"\n{N_FOLDS}-fold Stratified CV...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_accuracies: list[float] = []
fold_f1s: list[float] = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X.toarray(), y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    sw_train = sample_weights[train_idx]

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
    xgb_fold.fit(X_train, y_train, sample_weight=sw_train)
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
pass_fail = "pass" if cv_mean_acc >= TARGET_ACCURACY else "fail"
print(f"  Result: {pass_fail.upper()}")

# ──────────────────────────────────────────────
# Final model (fit on all data)
# ──────────────────────────────────────────────
print("\nFitting final model on full dataset...")
xgb.fit(X.toarray(), y, sample_weight=sample_weights)

# Per-class metrics on full training set (train-set eval, expected high — for reference only)
y_pred_full = xgb.predict(X.toarray())
report = classification_report(
    y, y_pred_full,
    target_names=classes,
    output_dict=True,
    zero_division=0,
)
print("\nPer-class metrics (full train set, for reference):")
for cls in classes:
    p = report[cls]["precision"]
    r = report[cls]["recall"]
    f = report[cls]["f1-score"]
    s = int(report[cls]["support"])
    print(f"  {cls:<10} precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}")

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
    "feature_shape": X.shape,
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
        "precision": round(report[cls]["precision"], 4),
        "recall": round(report[cls]["recall"], 4),
        "f1": round(report[cls]["f1-score"], 4),
        "support": int(report[cls]["support"]),
    }

result = {
    "req_no": "MH-2026-022",
    "product": "Aegis",
    "arch": "xgboost-text",
    "model_type": "text-classifier",
    "dataset": "aegis-severity-corpus",
    "num_samples": len(rows),
    "label_sources": label_sources,
    "label_distribution": {k: int(v) for k, v in label_dist.items()},
    "classes": list(classes),
    "cv_folds": N_FOLDS,
    "cv_accuracy_mean": round(cv_mean_acc, 4),
    "cv_accuracy_std": round(cv_std_acc, 4),
    "cv_f1_weighted_mean": round(cv_mean_f1, 4),
    "target_accuracy": TARGET_ACCURACY,
    "pass_fail": pass_fail,
    "per_class_metrics": per_class_metrics,
    "map50": None,          # N/A for text classifiers — modelhub poller compat
    "map50_95": None,       # N/A
    "accuracy": round(cv_mean_acc, 4),
    "model_path": str(model_path),
    "model_size_bytes": model_path.stat().st_size,
}

result_path = OUTPUT_DIR / "result.json"
with result_path.open("w") as f:
    json.dump(result, f, indent=2)
print(f"result.json saved: {result_path}")

# metrics.json (shorthand)
metrics = {
    "accuracy": round(cv_mean_acc, 4),
    "f1_weighted": round(cv_mean_f1, 4),
    "cv_accuracy_std": round(cv_std_acc, 4),
    "pass_fail": pass_fail,
    "num_samples": len(rows),
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
print("MH-2026-022 Training Summary")
print("=" * 60)
print(f"  Samples:        {len(rows)}")
print(f"  Human labels:   {label_sources['human']}")
print(f"  Rule labels:    {label_sources['rule']}")
print(f"  CV Accuracy:    {cv_mean_acc:.4f} ± {cv_std_acc:.4f}")
print(f"  CV F1-weighted: {cv_mean_f1:.4f}")
print(f"  Target:         {TARGET_ACCURACY}")
print(f"  PASS/FAIL:      {pass_fail.upper()}")
print("=" * 60)
