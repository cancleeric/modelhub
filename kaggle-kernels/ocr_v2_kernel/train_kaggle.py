# MH-2026-029 Engineering OCR v2 — TrOCR-base, 50 epochs + augmentation
# CEO Anderson 授權 2026-04-27
# MH-009 v1 CER=0.4227（太高）。v2 改進：
#   - epochs: 30 → 50（longer training）
#   - augmentation: RandomResizedCrop + ColorJitter
#   - learning rate: cosine warmup
#   - result.json 帶 cer + per-script accuracy
# Target: CER <= 0.05 (pass); CER <= 0.10 (baseline)
# Estimated: ~4hr on T4

import os
import sys
import time
import json
import subprocess
import random
from pathlib import Path

# torch cu121 for T4
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
    "--index-url", "https://download.pytorch.org/whl/cu121",
    "--no-deps",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.35", "sentencepiece", "Pillow",
])

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)

WORK_DIR = "/kaggle/working/"
BEST_PATH = os.path.join(WORK_DIR, "ocr_v2_best")
CKPT_DIR = os.path.join(WORK_DIR, "ocr_v2_checkpoints")

# Dataset path
_DATA_CANDIDATES = [
    "/kaggle/input/aicad-engineering-ocr",
    "/kaggle/input/datasets/boardgamegroup/aicad-engineering-ocr",
]
DATA_ROOT = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])

import glob as _glob
if not os.path.exists(DATA_ROOT):
    _found = _glob.glob("/kaggle/input/**/labels.txt", recursive=True)
    if _found:
        DATA_ROOT = os.path.dirname(_found[0])
    print(f"[DIAG] fallback DATA_ROOT = {DATA_ROOT}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device       : {device}")
print(f"torch version: {torch.__version__}")
print(f"DATA_ROOT    : {DATA_ROOT}")
print(f"Files        : {sorted(os.listdir(DATA_ROOT)) if os.path.exists(DATA_ROOT) else '路徑不存在！'}")

# Find images directory
_candidates = [
    os.path.join(DATA_ROOT, "images", "images"),
    os.path.join(DATA_ROOT, "images"),
]
IMAGES_DIR = None
for c in _candidates:
    if os.path.exists(c):
        _n = len([f for f in os.listdir(c) if f.endswith(".png")])
        if _n > 100:
            IMAGES_DIR = c
            break
if IMAGES_DIR is None:
    _found = _glob.glob(f"{DATA_ROOT}/**/angle_0000.png", recursive=True)
    if _found:
        IMAGES_DIR = os.path.dirname(_found[0])
print(f"IMAGES_DIR   : {IMAGES_DIR}")
print(f"images count : {len(os.listdir(IMAGES_DIR)) if IMAGES_DIR else 0}")

# Read labels.txt
LABELS_FILE = os.path.join(DATA_ROOT, "labels.txt")
samples = []
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            fname, label = parts[0].strip(), parts[1].strip()
            img_path = os.path.join(IMAGES_DIR, fname)
            if os.path.exists(img_path):
                samples.append((img_path, label))

print(f"Total samples: {len(samples)}")
print(f"Sample labels: {[s[1] for s in samples[:5]]}")

# Detect per-script info（文字分類：numeric/latin/mixed）
def detect_script(text: str) -> str:
    """簡單偵測 script type."""
    text = text.strip()
    if text.replace(".", "").replace("-", "").replace(",", "").isnumeric():
        return "numeric"
    elif text.isascii():
        return "latin"
    else:
        return "mixed"

sample_scripts = [detect_script(s[1]) for s in samples]
from collections import Counter
script_dist = Counter(sample_scripts)
print(f"Script distribution: {dict(script_dist)}")

# Load TrOCR-base
MODEL_NAME = "microsoft/trocr-base-printed"
print(f"\nLoading model: {MODEL_NAME}")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
print(f"Model loaded, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ---------------------------------------------------------------------------
# v2 Augmentation: RandomResizedCrop + ColorJitter（training 用）
# ---------------------------------------------------------------------------
_aug_transform = T.Compose([
    T.RandomResizedCrop(size=(384, 384), scale=(0.85, 1.0), ratio=(2.5, 4.5)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
])


class OCRDataset(Dataset):
    def __init__(self, samples, processor, max_target_len=32, augment=False):
        self.samples = samples
        self.processor = processor
        self.max_target_len = max_target_len
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.augment:
            img = _aug_transform(img)
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            label,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels, "_script": detect_script(label)}


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
shuffled = samples[:]
random.shuffle(shuffled)
n_val = int(len(shuffled) * 0.15)
n_train = len(shuffled) - n_val
train_samples = shuffled[n_train:]
val_samples = shuffled[:n_val]

# augment=True for training（v2 新增）
train_ds = OCRDataset(train_samples, processor, augment=True)
val_ds = OCRDataset(val_samples, processor, augment=False)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------
def compute_cer(pred_str, label_str):
    from difflib import SequenceMatcher
    if len(label_str) == 0:
        return 0.0 if len(pred_str) == 0 else 1.0
    sm = SequenceMatcher(None, pred_str, label_str)
    edit_ops = len(pred_str) + len(label_str) - 2 * sum(b.size for b in sm.get_matching_blocks())
    return edit_ops / len(label_str)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred_ids = logits.argmax(-1) if hasattr(logits, 'argmax') else logits
    pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_strs = processor.batch_decode(labels, skip_special_tokens=True)
    cer_list = [compute_cer(p, l) for p, l in zip(pred_strs, label_strs)]
    mean_cer = float(np.mean(cer_list))
    exact_match = float(np.mean([p.strip() == l.strip() for p, l in zip(pred_strs, label_strs)]))
    return {"cer": mean_cer, "exact_match": exact_match}


# ---------------------------------------------------------------------------
# Training — v2: 50 epochs + cosine warmup LR
# ---------------------------------------------------------------------------
EPOCHS = int(os.getenv("MH029_EPOCHS", "50"))  # v2: 30→50
BATCH_SIZE = 16
WARMUP_STEPS = 300  # cosine warmup（v2 新增）

print(f"\n[TRAIN] epochs={EPOCHS} batch={BATCH_SIZE} warmup_steps={WARMUP_STEPS} aug=True", flush=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=CKPT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    logging_dir=os.path.join(WORK_DIR, "logs"),
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=(device.type == "cuda"),
    dataloader_num_workers=2,
    report_to="none",
    # v2: cosine LR schedule
    lr_scheduler_type="cosine",
    learning_rate=5e-5,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=processor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

print(f"\n=== Training Start (TrOCR-base, {EPOCHS} epochs, cosine warmup) ===", flush=True)
t_start = time.time()
trainer.train()
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] train elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

# Final eval
print("\n=== Final Evaluation ===", flush=True)
eval_results = trainer.evaluate()
final_cer = eval_results.get("eval_cer", 1.0)
exact_match = eval_results.get("eval_exact_match", 0.0)
print(f"Final CER        : {final_cer:.4f}  (target <= 0.05)")
print(f"Exact Match Rate : {exact_match:.4f}")

# Per-script accuracy（v2 新增）
per_script_results = {}
try:
    model.eval()
    script_groups: dict[str, list] = {"numeric": [], "latin": [], "mixed": []}
    for img_path, label in val_samples:
        script = detect_script(label)
        img = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        cer = compute_cer(pred.strip(), label.strip())
        exact = 1 if pred.strip() == label.strip() else 0
        script_groups[script].append((cer, exact))

    for script, vals in script_groups.items():
        if vals:
            mean_cer = float(np.mean([v[0] for v in vals]))
            mean_exact = float(np.mean([v[1] for v in vals]))
            per_script_results[script] = {
                "count": len(vals),
                "mean_cer": round(mean_cer, 4),
                "exact_match": round(mean_exact, 4),
            }
    print(f"Per-script: {per_script_results}", flush=True)
except Exception as _pse:
    print(f"[WARN] per-script eval failed: {_pse}", flush=True)

# Save best model
trainer.save_model(BEST_PATH)
processor.save_pretrained(BEST_PATH)
print(f"Best model saved -> {BEST_PATH}", flush=True)

# Result JSON
if final_cer <= 0.05:
    verdict, tier = "pass", "達目標 CER<=0.05"
elif final_cer <= 0.10:
    verdict, tier = "baseline", "CER<=0.10 可接受"
else:
    verdict, tier = "fail", "未達 baseline"

result = {
    "req_no": "MH-2026-029",
    "run": "kaggle_v2_trocr_base_longer",
    "arch": "TrOCR-base-printed",
    "device": str(device),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "warmup_steps": WARMUP_STEPS,
    "lr_scheduler": "cosine",
    "augmentation": ["RandomResizedCrop", "ColorJitter"],
    "train_seconds": train_seconds,
    "n_train": len(train_ds),
    "n_val": len(val_ds),
    "cer": round(final_cer, 4),
    "exact_match": round(exact_match, 4),
    "per_script_accuracy": per_script_results,
    "verdict": verdict,
    "tier": tier,
    "target": "CER <= 0.05",
    "baseline": "CER <= 0.10",
    "comparison": {
        "v1_cer": 0.4227,
        "v2_cer": round(final_cer, 4),
        "delta": round(final_cer - 0.4227, 4),
    },
    "best_path": BEST_PATH,
    "note": "v2: epochs 30→50, cosine warmup LR, RandomResizedCrop+ColorJitter augmentation",
}
result_path = os.path.join(WORK_DIR, "result.json")
with open(result_path, "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)
