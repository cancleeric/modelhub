"""
MH-2026-010 工程符號語意分類器 — MobileNetV2（本機 Mac MPS 版）

目標：val_accuracy >= 0.90（misc_instrument >= 0.85）
資料：11 class × 200 imgs @ ~/HurricaneCore/docker-data/modelhub/datasets/pid_symbols
設備：Mac MPS（torch 2.8）
"""
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

REQ_NO = "MH-2026-010"
DATA_DIR = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/pid_symbols")
OUT_DIR = Path(f"/Users/yinghaowang/HurricaneCore/modelhub/training/{REQ_NO.lower()}")
BEST_PATH = OUT_DIR / "pid_symbols_best.pth"
RESULT_PATH = OUT_DIR / "result.json"

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] req={REQ_NO} device={device} torch={torch.__version__}")

NUM_CLASSES = 11
EPOCHS = int(os.getenv("MH010_EPOCHS", "40"))
BATCH_SIZE = int(os.getenv("MH010_BATCH", "64"))
LR = 1e-3
SEED = 42

assert DATA_DIR.exists(), f"data dir not found: {DATA_DIR}"
print(f"[DATA] root={DATA_DIR}")

train_tf = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
val_tf = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# 兩套 transform 同一資料集，先做 split
full_ds = datasets.ImageFolder(str(DATA_DIR))
classes = full_ds.classes
print(f"[CLASSES] {classes}")
print(f"[TOTAL] images={len(full_ds)}")
n_val = int(len(full_ds) * 0.2)
n_train = len(full_ds) - n_val
train_idx, val_idx = random_split(
    range(len(full_ds)),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)

train_ds_full = datasets.ImageFolder(str(DATA_DIR), transform=train_tf)
val_ds_full = datasets.ImageFolder(str(DATA_DIR), transform=val_tf)
train_ds = Subset(train_ds_full, train_idx.indices)
val_ds = Subset(val_ds_full, val_idx.indices)
print(f"[SPLIT] train={len(train_ds)} val={len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 用 torchvision 的 mobilenet_v2（避免 timm 依賴）
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
print(f"[MODEL] loading mobilenet_v2 (pretrained ImageNet)")
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)
print(f"[MODEL] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
best_epoch = 0


def run_epoch(loader, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


t_start = time.time()
print("\n=== Training Start ===")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    vl_loss, vl_acc = run_epoch(val_loader, training=False)
    scheduler.step()
    elapsed = time.time() - t0
    print(f"Epoch [{epoch:02d}/{EPOCHS}]  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  |  val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}  |  {elapsed:.1f}s")
    if vl_acc > best_acc:
        best_acc = vl_acc
        best_epoch = epoch
        torch.save(model.state_dict(), BEST_PATH)
        print(f"  ** Best saved  val_acc={best_acc:.4f} -> {BEST_PATH.name}")

train_seconds = int(time.time() - t_start)
print(f"\n[DONE] train elapsed {train_seconds}s = {train_seconds/3600:.2f}h")

# Final evaluation with per-class
print("\n=== Final Evaluation (best checkpoint) ===")
model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.eval()
class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES
conf_matrix = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for t, p in zip(labels, preds):
            conf_matrix[t.item()][p.item()] += 1
            class_total[t.item()] += 1
            if t == p:
                class_correct[t.item()] += 1

print("Per-class accuracy:")
per_class = {}
misc_acc = 0.0
for i, cls in enumerate(classes):
    acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    per_class[cls] = round(acc, 4)
    flag = " ← misc" if cls == "misc_instrument" else ""
    if cls == "misc_instrument":
        misc_acc = acc
    print(f"  {cls:24s}: {acc:.4f}{flag}")

final_acc = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0
print(f"\nFinal val_acc : {final_acc:.4f}")
print(f"Best  val_acc : {best_acc:.4f}")
print(f"misc_instrument acc: {misc_acc:.4f} (target >= 0.85)")

# 結論
if best_acc >= 0.90 and misc_acc >= 0.85:
    verdict, tier = "pass", "達標"
elif best_acc >= 0.85:
    verdict, tier = "baseline", "baseline 可交付（val_acc >= 0.85 但未達 0.90 或 misc < 0.85）"
else:
    verdict, tier = "fail", "未達 baseline，需再訓練"

result = {
    "req_no": REQ_NO,
    "device": device,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "train_seconds": train_seconds,
    "n_train": len(train_ds),
    "n_val": len(val_ds),
    "classes": classes,
    "best_val_acc": round(best_acc, 4),
    "best_epoch": best_epoch,
    "final_val_acc": round(final_acc, 4),
    "misc_instrument_acc": round(misc_acc, 4),
    "per_class_acc": per_class,
    "verdict": verdict,
    "tier": tier,
    "target": "val_acc >= 0.90 AND misc_instrument >= 0.85",
    "best_path": str(BEST_PATH),
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"\n結論：{tier}")
