#!/bin/bash
# training_queue.sh — 本機訓練序列化控制
# 確保同一時間只有一個本機訓練任務執行
#
# 用法：./training_queue.sh <訓練腳本> [args...]
# 範例：./training_queue.sh training/train_local.py --epochs 50 --device mps
#
# 機制：flock /tmp/modelhub-training.lock 確保序列化執行
# 若有任務在跑，此腳本會等待（不丟棄），直到前一個任務完成。
# Ctrl+C 會中斷當前等待或執行，並釋放 lock。

set -euo pipefail

LOCK_FILE="/tmp/modelhub-training.lock"
LOCK_FD=9
QUEUE_LOG="/tmp/modelhub-training-queue.log"

if [ $# -eq 0 ]; then
    echo "用法：$0 <訓練腳本> [args...]" >&2
    echo "範例：$0 training/train_local.py --epochs 50 --device mps" >&2
    exit 1
fi

SCRIPT="$1"
shift
ARGS=("$@")

# 取得目前排隊位置（用 log 估算）
PID=$$
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

cleanup() {
    echo "[training_queue] [$$] 中斷訊號收到，釋放 lock 並退出" | tee -a "$QUEUE_LOG"
    # flock 會在 FD 關閉時自動釋放，此處確保退出
    exit 130
}
trap cleanup INT TERM

# 開啟 lock FD
eval "exec ${LOCK_FD}>\"${LOCK_FILE}\""

echo "[training_queue] [${PID}] ${TIMESTAMP} 等待 lock：${SCRIPT} ${ARGS[*]:-}" | tee -a "$QUEUE_LOG"

# flock -x：排他鎖，若 lock 被持有則等待（不加 -n，允許排隊）
# flock 釋放時自動喚醒排隊中的下一個
if ! flock -x "${LOCK_FD}"; then
    echo "[training_queue] [${PID}] 無法取得 lock，退出" >&2
    exit 1
fi

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "[training_queue] [${PID}] ${START_TIME} 取得 lock，開始執行：${SCRIPT} ${ARGS[*]:-}" | tee -a "$QUEUE_LOG"

# 檢查規範：本機訓練必須用 --device mps，禁止 --device cpu
for arg in "${ARGS[@]:-}"; do
    if [ "$arg" = "cpu" ] && [ "${PREV_ARG:-}" = "--device" ]; then
        echo "[training_queue] 錯誤：禁止使用 --device cpu（集團規範），請改用 --device mps" >&2
        exit 1
    fi
    PREV_ARG="$arg"
done

EXIT_CODE=0
# 執行訓練腳本
if python3 "$SCRIPT" "${ARGS[@]:-}"; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
if [ $EXIT_CODE -eq 0 ]; then
    echo "[training_queue] [${PID}] ${END_TIME} 訓練完成：${SCRIPT}" | tee -a "$QUEUE_LOG"
else
    echo "[training_queue] [${PID}] ${END_TIME} 訓練失敗（exit=${EXIT_CODE}）：${SCRIPT}" | tee -a "$QUEUE_LOG"
fi

# lock 在 FD 關閉時自動釋放
eval "exec ${LOCK_FD}>&-"

exit $EXIT_CODE
