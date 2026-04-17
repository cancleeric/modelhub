"""
validators.py — Sprint 6 提交驗證（rule-based warnings，非 block）
"""
import logging
from typing import List

import httpx

logger = logging.getLogger("modelhub.validators")


async def validate_submission(payload) -> List[str]:
    """回傳 warning string list，非空即前端應顯示黃色 banner。"""
    warnings: List[str] = []

    # 6.5a: Kaggle dataset URL 可達性
    url = getattr(payload, "kaggle_dataset_url", None)
    if url:
        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                resp = await client.head(url)
                if resp.status_code == 405:  # Kaggle 拒 HEAD，退 GET
                    resp = await client.get(url)
                if resp.status_code >= 400:
                    warnings.append(
                        f"Kaggle dataset URL 回應 {resp.status_code}，請確認網址正確且為公開資料集。"
                    )
        except Exception as e:
            logger.info("dataset url check failed (%s): %s", url, e)
            warnings.append(f"Kaggle dataset URL 無法驗證（{type(e).__name__}）。")

    # 6.5b: class_list vs class_count 一致性
    class_list_raw = getattr(payload, "class_list", None)
    class_count = getattr(payload, "class_count", None)
    if class_list_raw and class_count:
        items = [s.strip() for s in class_list_raw.split(",") if s.strip()]
        if len(items) != class_count:
            warnings.append(
                f"class_list 有 {len(items)} 個類別，但 class_count 標示為 {class_count}，請確認一致。"
            )

    # 6.5c: mAP target 合理範圍
    for field, label in [("map50_target", "mAP50 目標"), ("map50_95_target", "mAP50-95 目標")]:
        val = getattr(payload, field, None)
        if val is not None and not (0.0 <= val <= 1.0):
            warnings.append(f"{label} 應介於 0.0 ~ 1.0（目前 {val}）。")

    # Sprint 13 P1-A: 整體資料量及每類別平均樣本數警告
    dataset_train_count = getattr(payload, "dataset_train_count", None)
    class_count_val = getattr(payload, "class_count", None)

    if dataset_train_count is not None and dataset_train_count > 0:
        if dataset_train_count < 100:
            warnings.append(
                f"訓練集總量僅 {dataset_train_count} 張，建議 ≥ 100 張以確保基本訓練品質。"
            )
        if class_count_val is not None and class_count_val > 0:
            per_class_avg = dataset_train_count / class_count_val
            if per_class_avg < 20:
                warnings.append(
                    f"平均每類僅 {per_class_avg:.1f} 張（{dataset_train_count} 張 / {class_count_val} 類），"
                    f"建議 ≥ 50 才能達到合理 mAP。強烈建議補充資料或減少類別數量。"
                )
            elif per_class_avg < 50:
                warnings.append(
                    f"平均每類 {per_class_avg:.1f} 張（{dataset_train_count} 張 / {class_count_val} 類），"
                    f"建議每類 ≥ 50 張以達到合理 mAP。"
                )

    # Sprint 13 P2-B: 本機 dataset 路徑存在時掃描類別分佈
    dataset_path_val = getattr(payload, "dataset_path", None)
    if dataset_path_val:
        import os
        from pathlib import Path
        labels_train_dir = Path(dataset_path_val) / "labels" / "train"
        if labels_train_dir.exists() and labels_train_dir.is_dir():
            class_instance_counts: dict = {}
            try:
                for txt_file in labels_train_dir.glob("*.txt"):
                    with open(txt_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if parts:
                                cid = parts[0]
                                class_instance_counts[cid] = class_instance_counts.get(cid, 0) + 1
                for cid, count in sorted(class_instance_counts.items(), key=lambda x: x[1]):
                    if count < 20:
                        warnings.append(
                            f"class {cid} 僅有 {count} 個 instance，建議補充標記資料。"
                        )
            except Exception as _e:
                logger.info("dataset label scan failed: %s", _e)

    return warnings
