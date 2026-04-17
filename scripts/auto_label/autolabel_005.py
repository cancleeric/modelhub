"""
MH-2026-005 Auto-Labeling Script
策略：用 MH-2026-010 訓練的 MobileNetV2 classifier 做 sliding window，
      在 270 張儀器規格圖上產生 YOLO 格式標籤。

classifier 是 11-class 符號分類器（非 detector），
用 sliding window + 信心分數過濾後輸出 YOLO bbox。

用法：
    python3 autolabel_005.py [--input DIR] [--output DIR] [--threshold 0.5]

驗收門檻：auto-label 覆蓋率 >= 80%（270 張中至少 216 張有 bbox）
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ----------- 設定 -----------
DEFAULT_INPUT = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/instrument_drawings/raw")
DEFAULT_OUTPUT = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/instrument_drawings/labels_auto")
MODEL_PATH = Path("/Users/yinghaowang/HurricaneCore/modelhub/training/mh-2026-010/pid_symbols_best.pth")
NUM_CLASSES = 11
CLASSES = [
    "ball_valve", "check_valve", "cock_valve", "flow_meter", "gate_valve",
    "globe_valve", "misc_instrument", "panel", "regulator", "safety_valve",
    "solenoid_valve",
]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# sliding window 參數
# 對大圖先縮至 MAX_DIM，再做 sliding window 加速
MAX_DIM = 640
WINDOW_SIZES = [64, 96]        # 縮圖後的 patch 尺寸（減少尺度降低計算量）
STRIDE_RATIO = 0.75            # stride = window_size * ratio（增大 stride 加速）
RESIZE_TO = 96                 # classifier input size
BATCH_SIZE = 32                # 批次推論


def load_model(model_path: Path, device: str) -> torch.nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    import torch.nn as nn
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def preprocess_patch(patch: np.ndarray, resize_to: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
    elif patch.shape[2] == 4:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGRA2RGB)
    else:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return tf(patch).unsqueeze(0)


def nms_boxes(boxes: list, iou_threshold: float = 0.3) -> list:
    """Simple NMS on [x1, y1, x2, y2, conf, cls] boxes."""
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    while boxes_sorted:
        best = boxes_sorted.pop(0)
        kept.append(best)
        remaining = []
        for b in boxes_sorted:
            inter_x1 = max(best[0], b[0])
            inter_y1 = max(best[1], b[1])
            inter_x2 = min(best[2], b[2])
            inter_y2 = min(best[3], b[3])
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            area_best = (best[2] - best[0]) * (best[3] - best[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_best + area_b - inter_area
            iou = inter_area / union if union > 0 else 0
            if iou < iou_threshold:
                remaining.append(b)
        boxes_sorted = remaining
    return kept


def sliding_window_label(
    img_path: Path,
    model: torch.nn.Module,
    device: str,
    threshold: float,
) -> list:
    """
    Return list of (class_id, cx_norm, cy_norm, w_norm, h_norm, conf) tuples.
    使用批次推論 + 圖片縮放加速。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    orig_h, orig_w = img.shape[:2]

    # 縮圖到 MAX_DIM（加速 sliding window）
    scale = 1.0
    if max(orig_h, orig_w) > MAX_DIM:
        scale = MAX_DIM / max(orig_h, orig_w)
        img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
    h, w = img.shape[:2]

    all_patches = []   # tensor list
    all_coords = []    # [x1, y1, x2, y2] in scaled coords

    for win_size in WINDOW_SIZES:
        stride = max(1, int(win_size * STRIDE_RATIO))
        for y in range(0, h - win_size + 1, stride):
            for x in range(0, w - win_size + 1, stride):
                patch = img[y:y + win_size, x:x + win_size]
                all_patches.append(preprocess_patch(patch, RESIZE_TO))
                all_coords.append([x, y, x + win_size, y + win_size])

    if not all_patches:
        return []

    # 批次推論
    raw_boxes = []
    with torch.no_grad():
        for i in range(0, len(all_patches), BATCH_SIZE):
            batch = torch.cat(all_patches[i:i + BATCH_SIZE], dim=0).to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            confs, cls_ids = probs.max(1)
            for j, (conf, cls_id) in enumerate(zip(confs.cpu().tolist(), cls_ids.cpu().tolist())):
                if conf >= threshold:
                    x1, y1, x2, y2 = all_coords[i + j]
                    raw_boxes.append([x1, y1, x2, y2, conf, cls_id])

    kept = nms_boxes(raw_boxes, iou_threshold=0.3)

    # 反縮回原始座標
    yolo_lines = []
    for bx1, by1, bx2, by2, conf, cls_id in kept:
        # 轉回原始圖尺寸
        if scale != 1.0:
            bx1, by1 = bx1 / scale, by1 / scale
            bx2, by2 = bx2 / scale, by2 / scale
        cx = (bx1 + bx2) / 2 / orig_w
        cy = (by1 + by2) / 2 / orig_h
        bw = (bx2 - bx1) / orig_w
        bh = (by2 - by1) / orig_h
        yolo_lines.append((int(cls_id), cx, cy, bw, bh, conf))
    return yolo_lines


def main():
    parser = argparse.ArgumentParser(description="MH-2026-005 PID instrument auto-labeler")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INIT] device={device} threshold={args.threshold}")
    print(f"[INIT] model={args.model}")
    print(f"[INIT] input={args.input}")
    print(f"[INIT] output={args.output}")

    model = load_model(args.model, device)
    print(f"[MODEL] loaded OK, classes={CLASSES}")

    img_files = sorted([f for f in args.input.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
    print(f"[DATA] found {len(img_files)} images")

    covered = 0
    low_conf_images = []
    results_summary = []

    for i, img_path in enumerate(img_files):
        boxes = sliding_window_label(img_path, model, device, args.threshold)
        txt_path = args.output / (img_path.stem + ".txt")
        if boxes:
            with open(txt_path, "w") as f:
                for cls_id, cx, cy, bw, bh, conf in boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            covered += 1
        else:
            # 空檔案表示此圖無標注（仍建立，符合 YOLO 慣例）
            txt_path.write_text("")
            low_conf_images.append(str(img_path.name))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(img_files)}] covered={covered}")

    total = len(img_files)
    coverage_pct = covered / total * 100 if total > 0 else 0

    print(f"\n=== MH-2026-005 Auto-Label Report ===")
    print(f"Total images   : {total}")
    print(f"Covered (>= 1 bbox): {covered}")
    print(f"Coverage rate  : {coverage_pct:.1f}% (target >= 80%)")
    verdict = "PASS" if coverage_pct >= 80 else "FAIL"
    print(f"Verdict        : {verdict}")
    print(f"Low-conf images: {len(low_conf_images)}")
    print(f"Labels written to: {args.output}")

    # 輸出 needs_review.txt
    needs_review_path = args.output / "needs_review.txt"
    with open(needs_review_path, "w") as f:
        f.write("\n".join(low_conf_images))
    print(f"Needs review list: {needs_review_path}")

    # 輸出 coverage_report.json
    report = {
        "req_no": "MH-2026-005",
        "total": total,
        "covered": covered,
        "coverage_pct": round(coverage_pct, 2),
        "threshold": args.threshold,
        "verdict": verdict,
        "low_conf_count": len(low_conf_images),
        "labels_dir": str(args.output),
    }
    report_path = args.output / "coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Coverage report: {report_path}")

    if coverage_pct < 80:
        print(f"\n[WARN] Coverage {coverage_pct:.1f}% < 80% target.")
        print("       Try lowering --threshold (e.g., 0.3) and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
