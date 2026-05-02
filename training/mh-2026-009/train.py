"""
MH-2026-009 工程尺寸 OCR — TrOCR-small fine-tune（本機 Mac MPS 版）

目標：CER <= 0.05（baseline -0.05: CER <= 0.10 亦可交付）
資料：5000 patch @ ~/HurricaneCore/docker-data/modelhub/datasets/engineering_ocr
設備：Mac MPS（torch 2.8）
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# ---------------------------------------------------------------------------
# 路徑 + 設備
# ---------------------------------------------------------------------------
REQ_NO = "MH-2026-009"
DATA_ROOT = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/engineering_ocr")
OUT_DIR = Path(f"/Users/yinghaowang/HurricaneCore/modelhub/training/{REQ_NO.lower()}")
BEST_PATH = OUT_DIR / "best"
LOG_PATH = OUT_DIR / "train.log"
RESULT_PATH = OUT_DIR / "result.json"

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] req={REQ_NO} device={device} torch={torch.__version__}")

LABELS_FILE = DATA_ROOT / "labels.txt"
IMAGES_DIR = DATA_ROOT / "images"
assert LABELS_FILE.exists(), f"labels.txt not found: {LABELS_FILE}"
assert IMAGES_DIR.exists(), f"images/ not found: {IMAGES_DIR}"

# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
samples = []
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        fname, label = parts[0].strip(), parts[1].strip()
        img_path = IMAGES_DIR / fname
        if img_path.exists():
            samples.append((str(img_path), label))

print(f"[DATA] total={len(samples)} | first={[s[1] for s in samples[:5]]}")
assert len(samples) > 100

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/trocr-small-printed"
print(f"[MODEL] loading {MODEL_NAME}")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
print(f"[MODEL] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class OCRDataset(Dataset):
    def __init__(self, samples, processor, max_target_len=32):
        self.samples = samples
        self.processor = processor
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            label,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


SEED = 42
torch.manual_seed(SEED)
# 80/10/10 split（test 作為 holdout，trainer 用 train+val）
n_total = len(samples)
n_test = int(n_total * 0.10)
n_val = int(n_total * 0.10)
n_train = n_total - n_val - n_test
train_s, val_s, test_s = random_split(
    samples, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED),
)
train_s, val_s, test_s = list(train_s), list(val_s), list(test_s)
print(f"[SPLIT] train={len(train_s)} val={len(val_s)} test={len(test_s)}")

train_ds = OCRDataset(train_s, processor)
val_ds = OCRDataset(val_s, processor)
test_ds = OCRDataset(test_s, processor)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_cer(pred, label):
    from difflib import SequenceMatcher
    if not label:
        return 0.0 if not pred else 1.0
    sm = SequenceMatcher(None, pred, label)
    ops = len(pred) + len(label) - 2 * sum(b.size for b in sm.get_matching_blocks())
    return ops / len(label)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # predict_with_generate=True: predictions are already generated token IDs (not raw logits)
    # Do NOT call argmax — that would pick the single highest token ID per sequence
    pred_ids = logits if (hasattr(logits, "ndim") and logits.ndim == 2) else logits
    pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_strs = processor.batch_decode(labels, skip_special_tokens=True)
    cer_list = [compute_cer(p, l) for p, l in zip(pred_strs, label_strs)]
    em_list = [p.strip() == l.strip() for p, l in zip(pred_strs, label_strs)]
    return {"cer": float(np.mean(cer_list)), "exact_match": float(np.mean(em_list))}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS = int(os.getenv("MH009_EPOCHS", "8"))   # Mac MPS 8 epochs 約 3-4h
BATCH_SIZE = int(os.getenv("MH009_BATCH", "8"))
print(f"[TRAIN] epochs={EPOCHS} batch={BATCH_SIZE}")

BEST_PATH.mkdir(parents=True, exist_ok=True)
training_args = Seq2SeqTrainingArguments(
    output_dir=str(BEST_PATH),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=str(OUT_DIR / "logs"),
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    ignore_data_skip=True,   # resume 時跳過已處理過的 batch（加速）
    predict_with_generate=True,
    fp16=False,  # MPS 不支援 fp16 training
    dataloader_num_workers=0,  # MPS 有時 worker>0 會掛
    report_to="none",
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

t0 = time.time()
# 自動偵測 checkpoint 以便 resume
import glob as _glob
import re as _re
ckpts = _glob.glob(str(BEST_PATH / "checkpoint-*"))
# 按 checkpoint 數字排序，取最大步數
ckpts.sort(key=lambda p: int(_re.search(r"checkpoint-(\d+)", p).group(1)))
resume_ckpt = ckpts[-1] if ckpts else None
if resume_ckpt:
    print(f"\n=== Resume from {resume_ckpt} ===")
else:
    print("\n=== Training Start ===")
trainer.train(resume_from_checkpoint=resume_ckpt)
train_seconds = int(time.time() - t0)
print(f"[DONE] train elapsed {train_seconds}s = {train_seconds/3600:.2f}h")

# ---------------------------------------------------------------------------
# 最終 holdout 評估
# ---------------------------------------------------------------------------
print("\n=== Holdout Test Evaluation ===")
trainer.eval_dataset = test_ds
test_results = trainer.evaluate()
test_cer = test_results.get("eval_cer", 1.0)
test_em = test_results.get("eval_exact_match", 0.0)

trainer.eval_dataset = val_ds
val_results = trainer.evaluate()
val_cer = val_results.get("eval_cer", 1.0)

# 儲存
trainer.save_model(str(BEST_PATH))
processor.save_pretrained(str(BEST_PATH))

# 結論分級
if test_cer <= 0.05:
    verdict = "pass"
    tier = "達標"
elif test_cer <= 0.10:
    verdict = "baseline"
    tier = "baseline 可交付（未達 0.05 但優於 0.10）"
else:
    verdict = "fail"
    tier = "未達 baseline，需再訓練"

result = {
    "req_no": REQ_NO,
    "device": device,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "train_seconds": train_seconds,
    "n_train": len(train_s),
    "n_val": len(val_s),
    "n_test": len(test_s),
    "val_cer": round(val_cer, 4),
    "test_cer": round(test_cer, 4),
    "test_exact_match": round(test_em, 4),
    "verdict": verdict,
    "tier": tier,
    "target_cer": 0.05,
    "baseline_cer": 0.10,
    "best_path": str(BEST_PATH),
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"\n結論：{tier}")
