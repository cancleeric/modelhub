"""
MH-2026-006 Auto-Labeling Script
策略：用 OpenCV-based text region detection 偵測工程圖中的文字框，
      輸出 YOLO bbox（class 0 = text）。

優先使用 EasyOCR（輕量）；若未安裝，fallback 到 OpenCV MSER text detector。

用法：
    python3 autolabel_006.py [--input DIR] [--output DIR] [--min-confidence 0.3]

驗收門檻：auto-label 覆蓋率 >= 85%
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_INPUT = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/text_detection/raw")
DEFAULT_OUTPUT = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/text_detection/labels_auto")
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# 嘗試載入 easyocr
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
    print("[INIT] EasyOCR available")
except ImportError:
    _EASYOCR_AVAILABLE = False
    print("[INIT] EasyOCR not available, using OpenCV MSER fallback")


def nms_boxes_cv(boxes: list, iou_threshold: float = 0.5) -> list:
    """boxes: list of [x1, y1, x2, y2, conf]. Returns filtered list."""
    if not boxes:
        return []
    rects = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    scores = [b[4] for b in boxes]
    indices = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []
    return [boxes[i] for i in indices.flatten()]


def detect_text_easyocr(img_path: Path, reader, min_confidence: float) -> list:
    """Return list of [x1, y1, x2, y2, conf] in pixel coords."""
    results = reader.readtext(str(img_path), detail=1)
    boxes = []
    for bbox_pts, text, conf in results:
        if conf < min_confidence:
            continue
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        boxes.append([x1, y1, x2, y2, float(conf)])
    return boxes


def detect_text_mser(img_path: Path) -> list:
    """
    Fallback: OpenCV MSER-based text region detection.
    Works well on engineering drawings with printed text.
    Returns list of [x1, y1, x2, y2, conf=0.7].
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 方法1：MSER
    mser = cv2.MSER_create(5, 50, 14400)
    regions, _ = mser.detectRegions(gray)
    boxes_raw = []
    for region in regions:
        x, y, bw, bh = cv2.boundingRect(region.reshape(-1, 1, 2))
        ar = bw / max(bh, 1)
        if ar < 0.1 or ar > 15:
            continue
        if bw < 8 or bh < 6:
            continue
        boxes_raw.append([x, y, x + bw, y + bh, 0.7])

    # 方法2：adaptive threshold + contour（補充）
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        ar = bw / max(bh, 1)
        if area < 80 or area > w * h * 0.1:
            continue
        if ar < 0.15 or ar > 20:
            continue
        boxes_raw.append([x, y, x + bw, y + bh, 0.6])

    kept = nms_boxes_cv(boxes_raw, iou_threshold=0.5)
    return kept


def label_image(
    img_path: Path,
    reader,
    min_confidence: float,
) -> list:
    """Return YOLO lines as strings."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    if _EASYOCR_AVAILABLE and reader is not None:
        raw_boxes = detect_text_easyocr(img_path, reader, min_confidence)
    else:
        raw_boxes = detect_text_mser(img_path)

    if not raw_boxes:
        return []

    # NMS 去重
    final_boxes = nms_boxes_cv(raw_boxes, iou_threshold=0.5)

    yolo_lines = []
    for x1, y1, x2, y2, conf in final_boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return yolo_lines


def main():
    parser = argparse.ArgumentParser(description="MH-2026-006 text region auto-labeler")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # 初始化 reader
    reader = None
    if _EASYOCR_AVAILABLE:
        print("[INIT] Loading EasyOCR (English)...")
        reader = easyocr.Reader(["en"], gpu=False)
        print("[INIT] EasyOCR loaded")
    else:
        print("[INIT] Using OpenCV MSER text detector")

    img_files = sorted([f for f in args.input.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
    print(f"[DATA] found {len(img_files)} images")

    covered = 0
    low_conf_images = []

    for i, img_path in enumerate(img_files):
        lines = label_image(img_path, reader, args.min_confidence)
        txt_path = args.output / (img_path.stem + ".txt")
        if lines:
            with open(txt_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            covered += 1
        else:
            txt_path.write_text("")
            low_conf_images.append(str(img_path.name))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(img_files)}] covered={covered}")

    total = len(img_files)
    coverage_pct = covered / total * 100 if total > 0 else 0

    print(f"\n=== MH-2026-006 Auto-Label Report ===")
    print(f"Total images   : {total}")
    print(f"Covered (>= 1 bbox): {covered}")
    print(f"Coverage rate  : {coverage_pct:.1f}% (target >= 85%)")
    verdict = "PASS" if coverage_pct >= 85 else "FAIL"
    print(f"Verdict        : {verdict}")

    report = {
        "req_no": "MH-2026-006",
        "total": total,
        "covered": covered,
        "coverage_pct": round(coverage_pct, 2),
        "detector": "easyocr" if _EASYOCR_AVAILABLE else "opencv_mser",
        "min_confidence": args.min_confidence,
        "verdict": verdict,
        "low_conf_count": len(low_conf_images),
        "labels_dir": str(args.output),
    }
    report_path = args.output / "coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Coverage report: {report_path}")

    needs_review_path = args.output / "needs_review.txt"
    with open(needs_review_path, "w") as f:
        f.write("\n".join(low_conf_images))

    if coverage_pct < 85:
        print(f"\n[WARN] Coverage {coverage_pct:.1f}% < 85% target.")
        print("       Consider installing EasyOCR: pip install easyocr")
        sys.exit(1)


if __name__ == "__main__":
    main()
