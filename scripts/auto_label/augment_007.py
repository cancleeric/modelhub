"""
MH-2026-007 Dataset Augmentation Script
步驟：
1. 合成 300 張工程線圖 + 精確 mask（用 OpenCV 隨機畫線）
2. 將 raw/ 下已有的 500 張真實圖 + mask 複製到 processed/
3. 對真實圖若有缺 mask 則用 Canny+dilate 自動產
4. 統計並輸出 audit_sample_50.json（隨機抽 50 組）

用法：
    python3 augment_007.py [--raw DIR] [--out DIR] [--synth-count 300]

驗收：
- processed/{images,masks}/ 各有 >= 200 組
- audit_sample_50.json 含 50 組路徑
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

DATASETS_BASE = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/line_segmentation")
DEFAULT_RAW = DATASETS_BASE / "raw"
DEFAULT_OUT = DATASETS_BASE / "processed"
SYNTH_DIR = DATASETS_BASE / "synthetic"
AUDIT_JSON = DATASETS_BASE / "audit_sample_50.json"

SYNTH_SIZE = 512  # 合成圖尺寸


def synthesize_line_image(idx: int, seed: int) -> tuple:
    """
    在白底 512x512 上隨機畫直線/折線/虛線/箭頭，
    同步產生精確的 binary mask（白=線像素）。
    回傳 (image_bgr, mask_gray) numpy arrays。
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    img = np.ones((SYNTH_SIZE, SYNTH_SIZE, 3), dtype=np.uint8) * 255  # 白底
    mask = np.zeros((SYNTH_SIZE, SYNTH_SIZE), dtype=np.uint8)

    # 背景輕微雜訊（模擬紙質）
    noise = np_rng.integers(0, 15, (SYNTH_SIZE, SYNTH_SIZE, 3), dtype=np.uint8)
    img = cv2.subtract(img, noise)

    n_lines = rng.randint(2, 12)
    line_types = ["solid", "dashed", "center", "phantom"]

    for _ in range(n_lines):
        ltype = rng.choice(line_types)
        thickness = rng.randint(1, 5)
        # 線條顏色（深色，模擬工程圖）
        color_val = rng.randint(0, 80)
        color = (color_val, color_val, color_val)

        x1 = rng.randint(5, SYNTH_SIZE - 5)
        y1 = rng.randint(5, SYNTH_SIZE - 5)
        x2 = rng.randint(5, SYNTH_SIZE - 5)
        y2 = rng.randint(5, SYNTH_SIZE - 5)

        if ltype == "solid":
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

        elif ltype == "dashed":
            # 虛線：每段長 10px，間隔 8px
            length = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
            if length == 0:
                continue
            n_segs = length // 18 + 1
            for seg in range(n_segs):
                t_start = seg * 18 / length
                t_end = min((seg * 18 + 10) / length, 1.0)
                sx = int(x1 + (x2 - x1) * t_start)
                sy = int(y1 + (y2 - y1) * t_start)
                ex = int(x1 + (x2 - x1) * t_end)
                ey = int(y1 + (y2 - y1) * t_end)
                cv2.line(img, (sx, sy), (ex, ey), color, thickness)
                cv2.line(mask, (sx, sy), (ex, ey), 255, thickness)

        elif ltype == "center":
            # 中心線：長段-短段交替（12px-4px）
            length = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
            if length == 0:
                continue
            t = 0
            seg_pattern = [12, 4]
            seg_idx = 0
            while t < length:
                seg_len = seg_pattern[seg_idx % 2]
                t2 = min(t + seg_len, length)
                if seg_idx % 2 == 0:
                    sx = int(x1 + (x2 - x1) * t / length)
                    sy = int(y1 + (y2 - y1) * t / length)
                    ex = int(x1 + (x2 - x1) * t2 / length)
                    ey = int(y1 + (y2 - y1) * t2 / length)
                    cv2.line(img, (sx, sy), (ex, ey), color, thickness)
                    cv2.line(mask, (sx, sy), (ex, ey), 255, thickness)
                t = t2 + 3
                seg_idx += 1

        elif ltype == "phantom":
            # 假想線：細線加兩點
            thin = max(1, thickness - 1)
            cv2.line(img, (x1, y1), (x2, y2), color, thin)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thin)
            cx, cy_ = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx - 5, cy_), 2, color, -1)
            cv2.circle(img, (cx + 5, cy_), 2, color, -1)
            cv2.circle(mask, (cx - 5, cy_), 2, 255, -1)
            cv2.circle(mask, (cx + 5, cy_), 2, 255, -1)

        # 有機率加折線
        if rng.random() < 0.3:
            xm = rng.randint(5, SYNTH_SIZE - 5)
            ym = rng.randint(5, SYNTH_SIZE - 5)
            cv2.line(img, (x1, y1), (xm, ym), color, thickness)
            cv2.line(mask, (x1, y1), (xm, ym), 255, thickness)

    return img, mask


def generate_mask_from_image(img_path: Path) -> np.ndarray:
    """
    對真實圖用 Canny + dilate 產 mask（白=線條，黑=背景）。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((512, 512), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    # Dilate 讓細線更明顯
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    # 去除點狀雜訊：面積 < 20px 的連通域濾掉
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    mask = np.zeros_like(dilated)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= 20:
            mask[labels == label_id] = 255
    return mask


def main():
    parser = argparse.ArgumentParser(description="MH-2026-007 dataset augmentation")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synth-dir", type=Path, default=SYNTH_DIR)
    parser.add_argument("--synth-count", type=int, default=300)
    parser.add_argument("--audit-json", type=Path, default=AUDIT_JSON)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_images = args.out / "images"
    out_masks = args.out / "masks"
    synth_images = args.synth_dir / "images"
    synth_masks = args.synth_dir / "masks"
    for d in [out_images, out_masks, synth_images, synth_masks]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    all_pairs = []  # [(image_path, mask_path), ...]

    # === Step 1: 合成圖 ===
    print(f"\n[STEP 1] Generating {args.synth_count} synthetic images...")
    for i in range(args.synth_count):
        seed_i = args.seed + i
        img_arr, mask_arr = synthesize_line_image(i, seed_i)
        img_name = f"synth_{i:04d}.png"
        img_save = synth_images / img_name
        mask_save = synth_masks / img_name
        cv2.imwrite(str(img_save), img_arr)
        cv2.imwrite(str(mask_save), mask_arr)
        # 同時複製到 processed
        out_img_path = out_images / img_name
        out_mask_path = out_masks / img_name
        shutil.copy2(img_save, out_img_path)
        shutil.copy2(mask_save, out_mask_path)
        all_pairs.append((str(out_img_path), str(out_mask_path)))
        if (i + 1) % 100 == 0:
            print(f"  generated {i+1}/{args.synth_count}")

    print(f"[STEP 1] Done. Synthetic images: {len(all_pairs)}")

    # === Step 2: 處理 raw 真實圖 ===
    print(f"\n[STEP 2] Processing raw images from {args.raw}...")
    raw_images_dir = args.raw / "images"
    raw_masks_dir = args.raw / "masks"

    real_img_files = []
    if raw_images_dir.exists():
        real_img_files = sorted([
            f for f in raw_images_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
    elif args.raw.exists():
        # 若 raw/ 直接放圖
        real_img_files = sorted([
            f for f in args.raw.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])

    print(f"  Found {len(real_img_files)} real images in raw/")

    for img_path in real_img_files:
        img_name = img_path.stem + ".png"
        # 找對應 mask
        mask_path = None
        if raw_masks_dir.exists():
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = raw_masks_dir / (img_path.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break

        out_img_path = out_images / img_name
        out_mask_path = out_masks / img_name

        # 複製或轉換圖
        if img_path.suffix.lower() == ".png":
            shutil.copy2(img_path, out_img_path)
        else:
            img_arr = cv2.imread(str(img_path))
            if img_arr is not None:
                cv2.imwrite(str(out_img_path), img_arr)
            else:
                continue

        # 複製或自動產 mask
        if mask_path is not None:
            if mask_path.suffix.lower() == ".png":
                shutil.copy2(mask_path, out_mask_path)
            else:
                mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_arr is not None:
                    cv2.imwrite(str(out_mask_path), mask_arr)
                else:
                    mask_arr = generate_mask_from_image(out_img_path)
                    cv2.imwrite(str(out_mask_path), mask_arr)
        else:
            # 用 Canny 自動產 mask
            mask_arr = generate_mask_from_image(out_img_path)
            cv2.imwrite(str(out_mask_path), mask_arr)

        all_pairs.append((str(out_img_path), str(out_mask_path)))

    print(f"[STEP 2] Done. Real images processed: {len(real_img_files)}")

    # === Step 3: 驗證配對完整性 ===
    print(f"\n[STEP 3] Verifying pairs...")
    valid_pairs = []
    for img_p, mask_p in all_pairs:
        if Path(img_p).exists() and Path(mask_p).exists():
            valid_pairs.append((img_p, mask_p))
        else:
            print(f"  [WARN] Missing pair: {img_p}")

    print(f"  Valid pairs: {len(valid_pairs)}")

    # === Step 4: 產 audit_sample_50.json ===
    print(f"\n[STEP 4] Generating audit sample (50 pairs)...")
    random.seed(args.seed + 1)
    n_sample = min(50, len(valid_pairs))
    sample = random.sample(valid_pairs, n_sample)
    audit_data = [
        {"image": img_p, "mask": mask_p}
        for img_p, mask_p in sample
    ]
    with open(args.audit_json, "w") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"  Audit sample written to: {args.audit_json}")

    # === Summary ===
    n_processed_images = len(list(out_images.glob("*.png")))
    n_processed_masks = len(list(out_masks.glob("*.png")))
    print(f"\n=== MH-2026-007 Augmentation Report ===")
    print(f"Synthetic images      : {args.synth_count}")
    print(f"Real images processed : {len(real_img_files)}")
    print(f"Total processed images: {n_processed_images}")
    print(f"Total processed masks : {n_processed_masks}")
    print(f"Audit sample size     : {len(audit_data)}")
    print(f"Output images dir     : {out_images}")
    print(f"Output masks dir      : {out_masks}")
    print(f"Audit JSON            : {args.audit_json}")

    target_met = n_processed_images >= 200 and len(audit_data) == 50
    verdict = "PASS" if target_met else "FAIL"
    print(f"Verdict               : {verdict}")
    if n_processed_images < 200:
        print(f"[WARN] Only {n_processed_images} images, target >= 200")
    if len(audit_data) < 50:
        print(f"[WARN] Audit sample has only {len(audit_data)} pairs, need 50")

    report = {
        "req_no": "MH-2026-007",
        "synthetic_count": args.synth_count,
        "real_count": len(real_img_files),
        "total_processed": n_processed_images,
        "mask_count": n_processed_masks,
        "audit_sample_count": len(audit_data),
        "audit_json": str(args.audit_json),
        "verdict": verdict,
    }
    report_path = args.out / "augment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
