# MH-2026-009 Engineering OCR — TrOCR-base fine-tune (Kaggle GPU v2)
# 工程圖面數字/文字辨識
# 資料集：/kaggle/input/aicad-engineering-ocr/
# 目標：CER <= 0.05（字元錯誤率 5% 以下）
# 架構：TrOCR-base（由 small 升級，Kaggle T4 可跑，預估 6hr 以內）
# 每 epoch save checkpoint

import os, sys, time, subprocess

# ---------------------------------------------------------------------------
# 1. 安裝
# ---------------------------------------------------------------------------
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.35", "sentencepiece", "Pillow",
])
# 不降 NumPy：Kaggle Python 3.12 + torch 2.10+cu128 與 NumPy 2.x 相容

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# ---------------------------------------------------------------------------
# 2. 路徑設定
# ---------------------------------------------------------------------------
WORK_DIR   = "/kaggle/working/"
BEST_PATH  = os.path.join(WORK_DIR, "ocr_best")
CKPT_DIR   = os.path.join(WORK_DIR, "ocr_checkpoints")

# 資料集路徑候選
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

# ---------------------------------------------------------------------------
# 3. 找 images 目錄
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 4. 讀取 labels.txt
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 5. 載入 TrOCR-base（由 small 升級，表達能力更強）
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/trocr-base-printed"
print(f"\nLoading model: {MODEL_NAME}")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model     = model.to(device)

# 設定 decoder config
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size

print(f"Model loaded, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ---------------------------------------------------------------------------
# 6. Dataset
# ---------------------------------------------------------------------------
class OCRDataset(Dataset):
    def __init__(self, samples, processor, max_target_len=32):
        self.samples        = samples
        self.processor      = processor
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
n_val   = int(len(samples) * 0.15)
n_train = len(samples) - n_val
train_samples, val_samples = random_split(
    samples, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)
train_samples = list(train_samples)
val_samples   = list(val_samples)

train_ds = OCRDataset(train_samples, processor)
val_ds   = OCRDataset(val_samples,   processor)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ---------------------------------------------------------------------------
# 7. CER 計算
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
    pred_strs  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_strs = processor.batch_decode(labels, skip_special_tokens=True)
    cer_list = [compute_cer(p, l) for p, l in zip(pred_strs, label_strs)]
    mean_cer = float(np.mean(cer_list))
    exact_match = float(np.mean([p.strip() == l.strip() for p, l in zip(pred_strs, label_strs)]))
    return {"cer": mean_cer, "exact_match": exact_match}

# ---------------------------------------------------------------------------
# 8. 訓練（TrOCR-base，30 epochs，每 epoch save checkpoint）
# ---------------------------------------------------------------------------
EPOCHS     = int(os.getenv("MH009_EPOCHS", "30"))
BATCH_SIZE = 16

print(f"\n[TRAIN] epochs={EPOCHS} batch={BATCH_SIZE} model=trocr-base device={device}", flush=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=CKPT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir=os.path.join(WORK_DIR, "logs"),
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,          # 保留最後 3 個 checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=(device.type == "cuda"),
    dataloader_num_workers=2,
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

print("\n=== Training Start (TrOCR-base, 30 epochs) ===", flush=True)
t_start = time.time()
trainer.train()
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] train elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

# ---------------------------------------------------------------------------
# 9. 最終評估
# ---------------------------------------------------------------------------
print("\n=== Final Evaluation ===", flush=True)
eval_results = trainer.evaluate()
final_cer = eval_results.get("eval_cer", 1.0)
exact_match = eval_results.get("eval_exact_match", 0.0)

print(f"Final CER        : {final_cer:.4f}  (target <= 0.05)")
print(f"Exact Match Rate : {exact_match:.4f}")

# 儲存 best model
trainer.save_model(BEST_PATH)
processor.save_pretrained(BEST_PATH)
print(f"Best model saved -> {BEST_PATH}", flush=True)

# ---------------------------------------------------------------------------
# 10. 結果輸出
# ---------------------------------------------------------------------------
import json

if final_cer <= 0.05:
    verdict, tier = "pass", "達目標 CER<=0.05"
elif final_cer <= 0.10:
    verdict, tier = "baseline", "CER<=0.10 可接受"
else:
    verdict, tier = "fail", "未達 baseline"

result = {
    "req_no": "MH-2026-009",
    "run": "kaggle_v2_trocr_base",
    "arch": "TrOCR-base-printed",
    "device": str(device),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "train_seconds": train_seconds,
    "n_train": len(train_ds),
    "n_val": len(val_ds),
    "cer": round(final_cer, 4),
    "exact_match": round(exact_match, 4),
    "verdict": verdict,
    "tier": tier,
    "target": "CER <= 0.05",
    "baseline": "CER <= 0.10",
    "best_path": BEST_PATH,
}
result_path = os.path.join(WORK_DIR, "result.json")
with open(result_path, "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)
