# MH-2026-009 Engineering OCR — TrOCR-small fine-tune
# 工程圖面數字/文字辨識
# 資料集：/kaggle/input/datasets/boardgamegroup/aicad-engineering-ocr/
# 目標：CER <= 0.05（字元錯誤率 5% 以下）
# 預估：< 3 hr

import os, sys, time, subprocess

# ---------------------------------------------------------------------------
# 1. 安裝（P100 compatible）
# ---------------------------------------------------------------------------
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.2.2+cu118", "torchvision==0.17.2+cu118",
    "--index-url", "https://download.pytorch.org/whl/cu118",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.35", "sentencepiece", "Pillow",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy==1.26.4", "--force-reinstall",
])

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
DATA_ROOT  = "/kaggle/input/datasets/boardgamegroup/aicad-engineering-ocr/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device       : {device}")
print(f"torch version: {torch.__version__}")

# 診斷路徑
import glob as _glob
if not os.path.exists(DATA_ROOT):
    # fallback: 搜尋 labels.txt
    _found = _glob.glob("/kaggle/input/**/labels.txt", recursive=True)
    if _found:
        DATA_ROOT = os.path.dirname(_found[0]) + "/"
    print(f"[DIAG] fallback DATA_ROOT = {DATA_ROOT}")
print(f"DATA_ROOT    : {DATA_ROOT}")
print(f"Files        : {sorted(os.listdir(DATA_ROOT)) if os.path.exists(DATA_ROOT) else '路徑不存在！'}")

# ---------------------------------------------------------------------------
# 3. 解壓 images.zip（若有）
# ---------------------------------------------------------------------------
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
ZIP_PATH   = os.path.join(DATA_ROOT, "images.zip")

if not os.path.exists(IMAGES_DIR) and os.path.exists(ZIP_PATH):
    print("[UNZIP] Extracting images.zip ...")
    import zipfile
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_ROOT)
    print(f"[UNZIP] Done. images/ has {len(os.listdir(IMAGES_DIR))} files")
else:
    print(f"images/ has {len(os.listdir(IMAGES_DIR)) if os.path.exists(IMAGES_DIR) else 0} files")

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
# 5. 載入 TrOCR-small（printed）
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/trocr-small-printed"
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
        # Resize to TrOCR expected 384x384 (processor handles this)
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            label,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        # Replace pad token with -100 for loss masking
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
    """Simple character error rate."""
    from difflib import SequenceMatcher
    if len(label_str) == 0:
        return 0.0 if len(pred_str) == 0 else 1.0
    sm = SequenceMatcher(None, pred_str, label_str)
    edit_ops = len(pred_str) + len(label_str) - 2 * sum(b.size for b in sm.get_matching_blocks())
    return edit_ops / len(label_str)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred_ids = logits.argmax(-1) if hasattr(logits, 'argmax') else logits
    # Decode
    pred_strs  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_strs = processor.batch_decode(labels, skip_special_tokens=True)
    cer_list = [compute_cer(p, l) for p, l in zip(pred_strs, label_strs)]
    mean_cer = float(np.mean(cer_list))
    exact_match = float(np.mean([p.strip() == l.strip() for p, l in zip(pred_strs, label_strs)]))
    return {"cer": mean_cer, "exact_match": exact_match}

# ---------------------------------------------------------------------------
# 8. 訓練
# ---------------------------------------------------------------------------
EPOCHS     = 15
BATCH_SIZE = 16

training_args = Seq2SeqTrainingArguments(
    output_dir=BEST_PATH,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=os.path.join(WORK_DIR, "logs"),
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=True,
    dataloader_num_workers=2,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor.feature_extractor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

print("\n=== Training Start ===")
trainer.train()

# ---------------------------------------------------------------------------
# 9. 最終評估
# ---------------------------------------------------------------------------
print("\n=== Final Evaluation ===")
results = trainer.evaluate()
final_cer = results.get("eval_cer", 1.0)
exact_match = results.get("eval_exact_match", 0.0)

print(f"Final CER        : {final_cer:.4f}  (target <= 0.05)")
print(f"Exact Match Rate : {exact_match:.4f}")

# 儲存 best model
trainer.save_model(BEST_PATH)
processor.save_pretrained(BEST_PATH)
print(f"Best model saved → {BEST_PATH}")

if final_cer <= 0.05:
    print("✅ 目標達成 (CER <= 0.05)")
elif final_cer <= 0.10:
    print(f"⚠️  CER={final_cer:.4f}，未達標但可接受（<= 0.10），繼續調參")
else:
    print(f"❌ CER={final_cer:.4f}，需繼續訓練")
