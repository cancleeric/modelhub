"""
MH-2026-009 — 修正 generation_config 後重新評估既有 checkpoint
不重訓，只 reload 模型 + 設對 generate config + 跑 eval
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DATA_ROOT = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/engineering_ocr")
OUT_DIR = Path("/Users/yinghaowang/HurricaneCore/modelhub/training/mh-2026-009")
BEST_PATH = OUT_DIR / "best"
RESULT_PATH = OUT_DIR / "result_fixed.json"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INIT] device={device}")

# 載入相同的 split（seed=42）
LABELS_FILE = DATA_ROOT / "labels.txt"
IMAGES_DIR = DATA_ROOT / "images"
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

torch.manual_seed(42)
n_total = len(samples)
n_test = int(n_total * 0.10)
n_val = int(n_total * 0.10)
n_train = n_total - n_val - n_test
train_s, val_s, test_s = random_split(
    samples, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42),
)
test_s = list(test_s)
val_s = list(val_s)
print(f"[SPLIT] val={len(val_s)} test={len(test_s)}")

# 載入 best 模型 + processor
print(f"[MODEL] loading from {BEST_PATH}")
processor = TrOCRProcessor.from_pretrained(str(BEST_PATH), use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(str(BEST_PATH)).to(device)

# === 關鍵修正：設對 generation_config ===
print("[FIX] before generation_config:")
print(f"  decoder_start_token_id={model.generation_config.decoder_start_token_id}")
print(f"  eos_token_id={model.generation_config.eos_token_id}")
print(f"  pad_token_id={model.generation_config.pad_token_id}")
print(f"  max_length={model.generation_config.max_length}")

model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id  # 0
model.generation_config.bos_token_id = processor.tokenizer.cls_token_id  # 0
model.generation_config.eos_token_id = processor.tokenizer.sep_token_id  # 2
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id  # 1
model.generation_config.max_new_tokens = 32
model.generation_config.num_beams = 4
model.generation_config.early_stopping = True
model.generation_config.no_repeat_ngram_size = 3

print("[FIX] after generation_config:")
print(f"  decoder_start_token_id={model.generation_config.decoder_start_token_id}")
print(f"  eos_token_id={model.generation_config.eos_token_id}")
print(f"  pad_token_id={model.generation_config.pad_token_id}")
print(f"  max_new_tokens={model.generation_config.max_new_tokens}")
print(f"  num_beams={model.generation_config.num_beams}")

model.eval()

# CER
def compute_cer(pred, label):
    from difflib import SequenceMatcher
    if not label:
        return 0.0 if not pred else 1.0
    sm = SequenceMatcher(None, pred, label)
    ops = len(pred) + len(label) - 2 * sum(b.size for b in sm.get_matching_blocks())
    return ops / len(label)

def eval_split(name, ds_samples):
    print(f"\n=== {name} (n={len(ds_samples)}) ===")
    cer_list, em_list = [], []
    bad_examples, good_examples = [], []
    t0 = time.time()
    BATCH = 16
    for i in range(0, len(ds_samples), BATCH):
        batch = ds_samples[i:i+BATCH]
        imgs = [Image.open(p).convert("RGB") for p, _ in batch]
        labels = [l for _, l in batch]
        pixel_values = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(pixel_values)
        preds = processor.batch_decode(generated, skip_special_tokens=True)
        for p, l in zip(preds, labels):
            c = compute_cer(p, l)
            cer_list.append(c)
            em = (p.strip() == l.strip())
            em_list.append(em)
            if em and len(good_examples) < 5:
                good_examples.append((p, l))
            if not em and c > 0.5 and len(bad_examples) < 5:
                bad_examples.append((p, l, c))
        if (i // BATCH) % 5 == 0:
            print(f"  progress {i+len(batch)}/{len(ds_samples)} elapsed={time.time()-t0:.1f}s mean_cer={np.mean(cer_list):.4f}")
    cer = float(np.mean(cer_list))
    em = float(np.mean(em_list))
    print(f"\n  [{name}] CER={cer:.4f} EM={em:.4f} elapsed={time.time()-t0:.1f}s")
    print(f"  good examples (pred=label):")
    for p, l in good_examples:
        print(f"    pred={p!r} label={l!r}")
    print(f"  bad examples (cer>0.5):")
    for p, l, c in bad_examples:
        print(f"    pred={p!r} label={l!r} cer={c:.3f}")
    return cer, em

val_cer, val_em = eval_split("VAL", val_s)
test_cer, test_em = eval_split("TEST", test_s)

# 結論
if test_cer <= 0.05:
    verdict, tier = "pass", "達標"
elif test_cer <= 0.10:
    verdict, tier = "baseline", "baseline 可交付"
else:
    verdict, tier = "fail", "未達 baseline"

result = {
    "req_no": "MH-2026-009",
    "method": "fixed_generation_config_only_no_retrain",
    "device": device,
    "n_val": len(val_s),
    "n_test": len(test_s),
    "val_cer": round(val_cer, 4),
    "val_exact_match": round(val_em, 4),
    "test_cer": round(test_cer, 4),
    "test_exact_match": round(test_em, 4),
    "verdict": verdict,
    "tier": tier,
    "target_cer": 0.05,
    "baseline_cer": 0.10,
    "fix_applied": {
        "decoder_start_token_id": int(model.generation_config.decoder_start_token_id),
        "eos_token_id": int(model.generation_config.eos_token_id) if isinstance(model.generation_config.eos_token_id, int) else model.generation_config.eos_token_id,
        "max_new_tokens": int(model.generation_config.max_new_tokens),
        "num_beams": int(model.generation_config.num_beams),
    }
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"\n結論：{tier}")
