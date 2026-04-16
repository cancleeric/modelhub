#!/usr/bin/env python3
"""
ModelHub 模型拉取腳本
用法：python pull_latest_model.py --product AICAD --model pid --output ./models/pid_model.pt

環境變數：
  MODELHUB_URL      ModelHub 服務 URL（預設：http://localhost:8950）
  MODELHUB_API_KEY  API Key（從 Hurricane Vault: hurricanecore/dev/MODELHUB_API_KEY）
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

MODELHUB_URL = os.getenv("MODELHUB_URL", "http://localhost:8950")
MODELHUB_API_KEY = os.getenv("MODELHUB_API_KEY", "")
HEADERS = {"X-Api-Key": MODELHUB_API_KEY}


def get_latest(product: str, model_name: str) -> dict:
    url = f"{MODELHUB_URL}/api/registry/latest"
    r = requests.get(url, params={"product": product, "model_name": model_name}, headers=HEADERS, timeout=10)
    if r.status_code == 404:
        print(f"[ModelHub] 找不到 product={product} model_name={model_name} 的已通過版本", file=sys.stderr)
        sys.exit(1)
    r.raise_for_status()
    return r.json()


def download_model(version_id: int, output_path: str) -> None:
    url = f"{MODELHUB_URL}/api/registry/{version_id}/download"
    r = requests.get(url, headers=HEADERS, stream=True, timeout=60)
    if r.status_code == 404:
        print(f"[ModelHub] 模型檔案不存在於伺服器（id={version_id}）", file=sys.stderr)
        sys.exit(1)
    r.raise_for_status()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)
    print(f"[ModelHub] 下載完成，{total / 1024 / 1024:.1f} MB -> {output_path}")


def load_cache(cache_path: str) -> dict:
    p = Path(cache_path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache_path: str, cache: dict) -> None:
    Path(cache_path).write_text(json.dumps(cache, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="從 ModelHub 拉取最新模型")
    parser.add_argument("--product", required=True, help="產品代碼，如 AICAD")
    parser.add_argument("--model", required=True, help="模型名稱，如 pid")
    parser.add_argument("--output", required=True, help="輸出路徑，如 ./models/pid_model.pt")
    parser.add_argument("--cache-meta", default=".modelhub_cache.json", help="本地 cache 記錄檔（預設：.modelhub_cache.json）")
    parser.add_argument("--force", action="store_true", help="強制重新下載，忽略 cache")
    args = parser.parse_args()

    if not MODELHUB_API_KEY:
        print("[ModelHub] 錯誤：未設定 MODELHUB_API_KEY 環境變數", file=sys.stderr)
        print("           請執行：export MODELHUB_API_KEY=$(hvault get hurricanecore/dev/MODELHUB_API_KEY --show -q)", file=sys.stderr)
        sys.exit(1)

    print(f"[ModelHub] 查詢 product={args.product} model={args.model}")
    latest = get_latest(args.product, args.model)

    version = latest.get("version", "unknown")
    accepted_at = latest.get("accepted_at", "")
    version_id = latest["id"]

    print(f"[ModelHub] 最新版本: {version} (id={version_id}, accepted_at={accepted_at})")

    cache = load_cache(args.cache_meta)
    cache_key = f"{args.product}/{args.model}"

    if not args.force and cache.get(cache_key) == accepted_at and Path(args.output).exists():
        print(f"[ModelHub] 已是最新版本（{accepted_at}），跳過下載")
        return

    print(f"[ModelHub] 下載新版本 {version}...")
    download_model(version_id, args.output)

    cache[cache_key] = accepted_at
    save_cache(args.cache_meta, cache)
    print(f"[ModelHub] 已更新至 {args.output}")


if __name__ == "__main__":
    main()
