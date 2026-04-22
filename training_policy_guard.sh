#!/bin/bash
# training_policy_guard.sh — 訓練資源合規檢查
# 呼叫任何訓練腳本前先執行這個
#
# 用法：source training_policy_guard.sh  （作為前置檢查）
#   或：./training_policy_guard.sh <腳本> [args...]  （包裝執行）
#
# 規則（CLAUDE.md 集團規範）：
#   1. 優先用 Kaggle 免費 GPU（30 hr/週）
#   2. Kaggle 配額耗盡 → Lightning AI / SSH 內網
#   3. 本機 MPS 是最後手段，且同時只能跑一個任務（用 training_queue.sh）
#
# 若 Kaggle 配額未耗盡，腳本會提示使用 Kaggle，不直接執行本機訓練。
# 若已確認 Kaggle 配額耗盡，請帶環境變數執行：
#   KAGGLE_QUOTA_EXHAUSTED=1 ./training_policy_guard.sh <腳本> [args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAGGLE_KERNELS_DIR="${SCRIPT_DIR}/kaggle-kernels"

# ─── 輔助函式 ────────────────────────────────────────────────────────────────

check_kaggle_quota() {
    # 嘗試用 neritic 或直接查 Kaggle API 確認本週剩餘配額
    # 若無法查詢，保守假設「還有配額」
    if [ "${KAGGLE_QUOTA_EXHAUSTED:-0}" = "1" ]; then
        return 1  # 已宣告耗盡
    fi

    # 嘗試查詢 Kaggle 使用紀錄（需要 kaggle CLI 已設定）
    if command -v kaggle >/dev/null 2>&1 && \
       [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
        # 實際查詢方式：解析 kaggle kernels list 的 GPU 使用量
        # 此處使用保守策略：預設假設有剩餘配額，除非明確宣告耗盡
        echo "[policy_guard] Kaggle CLI 可用，假設本週仍有配額。"
        echo "[policy_guard] 若配額確實耗盡，請設定 KAGGLE_QUOTA_EXHAUSTED=1"
        return 0  # 有配額
    fi

    # 無法確認，保守假設有配額（促使使用者先確認 Kaggle）
    echo "[policy_guard] 無法查詢 Kaggle 配額（CLI 未設定或缺少 KAGGLE_USERNAME/KAGGLE_KEY）"
    echo "[policy_guard] 保守假設：Kaggle 仍有配額"
    return 0  # 保守假設有配額
}

list_kaggle_kernels() {
    if [ -d "$KAGGLE_KERNELS_DIR" ]; then
        echo "  現有 Kaggle kernel："
        ls "$KAGGLE_KERNELS_DIR" 2>/dev/null | sed 's/^/    - /'
    else
        echo "  （尚無 kaggle-kernels/ 目錄）"
    fi
}

# ─── 主邏輯 ──────────────────────────────────────────────────────────────────

main() {
    local script="${1:-}"
    shift 2>/dev/null || true
    local args=("$@")

    echo "========================================================"
    echo " ModelHub 訓練資源合規檢查（集團規範）"
    echo "========================================================"
    echo ""
    echo "資源優先序："
    echo "  1. Kaggle 免費 GPU（30 hr/週）← 優先"
    echo "  2. Lightning AI（22 hr/月）/ SSH 內網 GPU"
    echo "  3. 本機 MPS（最後手段）"
    echo ""

    if check_kaggle_quota; then
        # Kaggle 有配額 → 提示使用 Kaggle，不執行本機
        echo "[policy_guard] Kaggle 配額未耗盡 → 請使用 Kaggle Kernel 訓練"
        echo ""
        list_kaggle_kernels
        echo ""
        echo "使用方式："
        echo "  1. 確認 kaggle-kernels/<kernel>/ 目錄下有對應 kernel"
        echo "  2. 執行：kaggle kernels push -p kaggle-kernels/<kernel>/"
        echo "  3. 在 ModelHub 後台 attach-kernel 綁定工單"
        echo ""
        echo "若確認 Kaggle 配額已耗盡，請這樣執行本機訓練："
        echo "  KAGGLE_QUOTA_EXHAUSTED=1 $0 ${script:-<腳本>} ${args[*]:-[args]}"
        echo ""
        echo "[policy_guard] 中止本機訓練（Kaggle 配額充足時不允許直接跑本機）"
        exit 2
    fi

    # Kaggle 配額耗盡 → 允許本機執行（透過 training_queue.sh 序列化）
    echo "[policy_guard] KAGGLE_QUOTA_EXHAUSTED=1 已確認，允許本機訓練"
    echo "[policy_guard] 透過 training_queue.sh 序列化執行（確保同時只跑一個任務）"
    echo ""

    if [ -z "$script" ]; then
        echo "錯誤：請提供訓練腳本路徑" >&2
        echo "用法：KAGGLE_QUOTA_EXHAUSTED=1 $0 <腳本> [args...]" >&2
        exit 1
    fi

    # 交給 training_queue.sh 處理序列化 + device 檢查
    exec "${SCRIPT_DIR}/training_queue.sh" "$script" "${args[@]:-}"
}

main "$@"
