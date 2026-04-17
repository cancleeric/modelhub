"""
MH-2026-007 Audit Viewer
用 Flask 提供一個簡易 image+mask overlay 瀏覽頁，
方便 CPO/CEO 快速抽檢 50 組樣本品質。

用法：
    python3 audit_viewer.py [--audit-json PATH] [--port 7777]
    然後開瀏覽器：http://localhost:7777

依賴：flask, opencv-python（已在系統 Python 中）
"""
import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

AUDIT_JSON_DEFAULT = Path(
    "/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/line_segmentation/audit_sample_50.json"
)


def load_audit_samples(json_path: Path) -> list:
    with open(json_path) as f:
        return json.load(f)


def make_overlay_b64(img_path: str, mask_path: str) -> str:
    """回傳 image+mask overlay 的 base64 PNG。"""
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(blank, "FILE NOT FOUND", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        _, buf = cv2.imencode(".png", blank)
        return base64.b64encode(buf.tobytes()).decode()

    # resize mask 到和 img 一樣大
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # overlay：mask 白色像素塗半透明紅色
    overlay = img.copy()
    overlay[mask > 127] = (0, 0, 200)
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    # resize 縮圖以加速顯示
    h, w = blended.shape[:2]
    max_size = 512
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        blended = cv2.resize(blended, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".png", blended)
    return base64.b64encode(buf.tobytes()).decode()


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MH-2026-007 Audit Viewer</title>
<style>
body {{ font-family: sans-serif; background: #f5f5f5; }}
h1 {{ padding: 16px; background: #1e293b; color: white; margin: 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; padding: 16px; }}
.card {{ background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.12); padding: 12px; }}
.card img {{ width: 100%; border-radius: 4px; }}
.card .idx {{ font-size: 12px; color: #666; margin-bottom: 4px; }}
.card .paths {{ font-size: 10px; color: #999; word-break: break-all; margin-top: 6px; }}
.summary {{ padding: 12px 16px; background: #e0f2fe; font-size: 14px; }}
</style>
</head>
<body>
<h1>MH-2026-007 Audit Viewer — {total} samples</h1>
<div class="summary">
  紅色覆蓋區 = mask（線條像素）。目測確認 mask 正確覆蓋線條、無大量誤標即算通過。<br>
  驗收門檻：50 組中 >= 40 組通過。
</div>
<div class="grid">
{cards}
</div>
</body>
</html>"""

CARD_TEMPLATE = """<div class="card">
  <div class="idx">#{idx} / {total}</div>
  <img src="data:image/png;base64,{b64}" alt="sample {idx}">
  <div class="paths">
    img: {img_path}<br>
    mask: {mask_path}
  </div>
</div>"""


def build_html(samples: list) -> str:
    cards = []
    total = len(samples)
    for i, s in enumerate(samples):
        b64 = make_overlay_b64(s["image"], s["mask"])
        card = CARD_TEMPLATE.format(
            idx=i + 1,
            total=total,
            b64=b64,
            img_path=s["image"],
            mask_path=s["mask"],
        )
        cards.append(card)
        if (i + 1) % 10 == 0:
            print(f"  Rendered {i+1}/{total}")
    return HTML_TEMPLATE.format(total=total, cards="\n".join(cards))


def main():
    parser = argparse.ArgumentParser(description="MH-2026-007 audit viewer")
    parser.add_argument("--audit-json", type=Path, default=AUDIT_JSON_DEFAULT)
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--static", action="store_true", help="輸出靜態 HTML 到 audit_viewer.html 而非啟動 server")
    args = parser.parse_args()

    samples = load_audit_samples(args.audit_json)
    print(f"[INIT] Loaded {len(samples)} audit samples")

    if args.static:
        print("[RENDER] Building static HTML...")
        html = build_html(samples)
        out_path = args.audit_json.parent / "audit_viewer.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"[DONE] Static HTML: {out_path}")
        return

    # Flask server
    try:
        from flask import Flask, Response
    except ImportError:
        print("[ERROR] Flask not installed. Try: pip install flask")
        print("        Or use --static flag to generate a static HTML file instead.")
        import sys; sys.exit(1)

    app = Flask(__name__)

    @app.route("/")
    def index():
        print("[RENDER] Building overlay HTML...")
        html = build_html(samples)
        return Response(html, mimetype="text/html")

    print(f"[SERVER] Starting audit viewer at http://localhost:{args.port}")
    print("         Press Ctrl+C to stop.")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
