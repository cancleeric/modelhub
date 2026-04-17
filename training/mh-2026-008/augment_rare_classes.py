"""
MH-2026-008 稀少類別資料增補（Sprint 13 P1-C 步驟 1）

對 gate_valve、panel、safety_valve 現有訓練圖做重度 augmentation，
目標：每類補到 >= 30 張 augmented 圖（含原始）。

使用 OpenCV 實作（避免額外安裝 albumentations 依賴）：
- flip (LR / UD)
- rotate (90 / 180 / 270)
- brightness / contrast 調整
- Gaussian blur

輸出到 training/mh-2026-008/augmented_dataset/
（YOLO 格式：images/ + labels/，可直接合併進 yolo_dataset/train/）

用法：
    python3 training/mh-2026-008/augment_rare_classes.py
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

OUT_DIR = Path("/Users/yinghaowang/HurricaneCore/modelhub/training/mh-2026-008")
YOLO_DIR = OUT_DIR / "yolo_dataset"
AUG_DIR = OUT_DIR / "augmented_dataset"
TRAIN_IMAGES = YOLO_DIR / "train" / "images"
TRAIN_LABELS = YOLO_DIR / "train" / "labels"

# 要重點增補的稀少類別（class_id 參照 CLASS_NAMES）
CLASS_NAMES = [
    "ball_valve", "check_valve", "cock_valve", "flow_meter", "gate_valve",
    "globe_valve", "misc_instrument", "panel", "regulator", "safety_valve",
    "solenoid_valve",
]
RARE_CLASSES = {"gate_valve": 4, "panel": 7, "safety_valve": 9}
TARGET_PER_CLASS = 30   # 每類至少達到此數量（含原始）


# ---------------------------------------------------------------------------
# Augmentation 函式
# ---------------------------------------------------------------------------

def aug_flip_lr(img: np.ndarray, labels: list) -> tuple:
    """水平翻轉"""
    img_out = cv2.flip(img, 1)
    labels_out = []
    for line in labels:
        parts = line.split()
        if len(parts) == 5:
            cid, cx, cy, bw, bh = parts
            cx_new = 1.0 - float(cx)
            labels_out.append(f"{cid} {cx_new:.6f} {cy} {bw} {bh}")
        else:
            labels_out.append(line)
    return img_out, labels_out


def aug_flip_ud(img: np.ndarray, labels: list) -> tuple:
    """垂直翻轉"""
    img_out = cv2.flip(img, 0)
    labels_out = []
    for line in labels:
        parts = line.split()
        if len(parts) == 5:
            cid, cx, cy, bw, bh = parts
            cy_new = 1.0 - float(cy)
            labels_out.append(f"{cid} {cx} {cy_new:.6f} {bw} {bh}")
        else:
            labels_out.append(line)
    return img_out, labels_out


def aug_rotate90(img: np.ndarray, labels: list, k: int = 1) -> tuple:
    """旋轉 k*90 度（逆時針）"""
    img_out = np.rot90(img, k=k)
    labels_out = []
    for line in labels:
        parts = line.split()
        if len(parts) == 5:
            cid = parts[0]
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # 每旋轉 90 度，座標做對應轉換
            for _ in range(k % 4):
                cx, cy, bw, bh = cy, 1.0 - cx, bh, bw
            labels_out.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        else:
            labels_out.append(line)
    return img_out, labels_out


def aug_brightness(img: np.ndarray, labels: list, factor: float = 1.3) -> tuple:
    """亮度調整"""
    img_float = img.astype(np.float32) * factor
    img_out = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_out, labels


def aug_contrast(img: np.ndarray, labels: list, factor: float = 1.4) -> tuple:
    """對比度調整"""
    mean = np.mean(img.astype(np.float32))
    img_float = (img.astype(np.float32) - mean) * factor + mean
    img_out = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_out, labels


def aug_blur(img: np.ndarray, labels: list, ksize: int = 3) -> tuple:
    """Gaussian blur"""
    img_out = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img_out, labels


AUG_OPS = [
    ("flipLR", aug_flip_lr),
    ("flipUD", aug_flip_ud),
    ("rot90",  lambda img, lbs: aug_rotate90(img, lbs, k=1)),
    ("rot180", lambda img, lbs: aug_rotate90(img, lbs, k=2)),
    ("rot270", lambda img, lbs: aug_rotate90(img, lbs, k=3)),
    ("bright", lambda img, lbs: aug_brightness(img, lbs, factor=1.3)),
    ("dark",   lambda img, lbs: aug_brightness(img, lbs, factor=0.7)),
    ("contrast", lambda img, lbs: aug_contrast(img, lbs, factor=1.4)),
    ("blur",   lambda img, lbs: aug_blur(img, lbs, ksize=3)),
]


# ---------------------------------------------------------------------------
# 掃描訓練集，找出各稀少類別的圖片
# ---------------------------------------------------------------------------

def find_images_with_class(class_id: int, images_dir: Path, labels_dir: Path) -> list:
    """回傳含有特定 class_id 的圖片路徑清單"""
    result = []
    for txt_file in labels_dir.glob("*.txt"):
        try:
            content = txt_file.read_text().strip()
            if not content:
                continue
            for line in content.splitlines():
                parts = line.split()
                if parts and int(parts[0]) == class_id:
                    # 找對應圖片
                    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                        img_path = images_dir / (txt_file.stem + ext)
                        if img_path.exists():
                            result.append(img_path)
                            break
                    break  # 一個 txt 只加一次
        except Exception:
            continue
    return result


def write_aug_pair(img: np.ndarray, labels: list, out_images: Path, out_labels: Path, stem: str):
    """寫出 augmented 圖片與對應 label"""
    out_img_path = out_images / f"{stem}.jpg"
    out_lbl_path = out_labels / f"{stem}.txt"
    cv2.imwrite(str(out_img_path), img)
    out_lbl_path.write_text("\n".join(labels) + "\n" if labels else "")


def main():
    if not TRAIN_IMAGES.exists():
        print(f"[ERROR] 訓練圖片目錄不存在：{TRAIN_IMAGES}", file=sys.stderr)
        print(f"  請先執行 train.py 建立 yolo_dataset 目錄結構", file=sys.stderr)
        sys.exit(1)

    # 建立輸出目錄
    aug_images = AUG_DIR / "images"
    aug_labels = AUG_DIR / "labels"
    aug_images.mkdir(parents=True, exist_ok=True)
    aug_labels.mkdir(parents=True, exist_ok=True)

    report = {}

    for class_name, class_id in RARE_CLASSES.items():
        source_images = find_images_with_class(class_id, TRAIN_IMAGES, TRAIN_LABELS)
        original_count = len(source_images)
        print(f"\n[{class_name}] class_id={class_id} 原始圖片數={original_count}", flush=True)

        if original_count == 0:
            print(f"  [WARN] {class_name} 無訓練圖片，跳過", flush=True)
            report[class_name] = {"original": 0, "augmented": 0, "total": 0}
            continue

        # 複製原始圖片到 augmented_dataset（不重複）
        aug_count = 0
        for img_path in source_images:
            lbl_path = TRAIN_LABELS / (img_path.stem + ".txt")
            dst_img = aug_images / img_path.name
            dst_lbl = aug_labels / lbl_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if lbl_path.exists() and not dst_lbl.exists():
                shutil.copy2(lbl_path, dst_lbl)

        # 計算需要多少 augmented 圖
        needed = max(0, TARGET_PER_CLASS - original_count)
        print(f"  需要補充 {needed} 張 augmented 圖（目標 {TARGET_PER_CLASS} 張）", flush=True)

        op_idx = 0
        for img_path in source_images:
            if aug_count >= needed:
                break
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            lbl_path = TRAIN_LABELS / (img_path.stem + ".txt")
            try:
                labels = lbl_path.read_text().strip().splitlines() if lbl_path.exists() else []
            except Exception:
                labels = []

            for op_name, op_fn in AUG_OPS:
                if aug_count >= needed:
                    break
                try:
                    aug_img, aug_lbs = op_fn(img, labels[:])
                    stem = f"{img_path.stem}_aug_{op_name}_{op_idx:04d}"
                    write_aug_pair(aug_img, aug_lbs, aug_images, aug_labels, stem)
                    aug_count += 1
                    op_idx += 1
                except Exception as e:
                    print(f"  [WARN] aug op {op_name} 失敗: {e}", flush=True)

        total = original_count + aug_count
        report[class_name] = {"original": original_count, "augmented": aug_count, "total": total}
        print(f"  完成：原始 {original_count} + augmented {aug_count} = {total} 張", flush=True)

    # 寫 report.json
    report_path = AUG_DIR / "augment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n=== Augmentation 完成 ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"輸出目錄：{AUG_DIR}", flush=True)
    print(f"Report：{report_path}", flush=True)


if __name__ == "__main__":
    main()
