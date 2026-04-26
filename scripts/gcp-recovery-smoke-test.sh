#!/usr/bin/env bash
# M15 R4 — GCP Cloud Run 恢復 Smoke Test
#
# 依序 curl 4 個 GCP 服務健康檢查 URL：
#   1. brain-api  /v1/health
#   2. brain-cloud /health
#   3. brain-cloud /health/detailed（需 X-Internal-Key）
#   4. anemone-server /health
#
# 所有端點通過 → exit 0
# 任一失敗 → 列出失敗項 → exit 1
#
# 所有 URL 從環境變數讀取（不含 GCP secret）：
#   BRAIN_API_URL      — brain-api base URL（e.g. https://brain-api-production-xxx.run.app）
#   BRAIN_CLOUD_URL    — brain-cloud base URL（e.g. https://brain-cloud-production-xxx.run.app）
#   ANEMONE_URL        — anemone-server base URL（e.g. https://anemone-api-production-xxx.run.app）
#   BRAIN_API_KEY      — X-Internal-Key for brain-cloud internal endpoints
#
# 本機 dev 測試模式：設 SMOKE_TEST_LOCAL=1 改用本機 URL
#
# 用法：
#   BRAIN_API_URL=https://... BRAIN_CLOUD_URL=https://... ANEMONE_URL=https://... \
#   BRAIN_API_KEY=xxx bash scripts/gcp-recovery-smoke-test.sh

set -euo pipefail

# ── 設定 ─────────────────────────────────────────────────────────────────────

TIMEOUT="${SMOKE_TEST_TIMEOUT:-10}"  # 每個 curl 超時（秒）

if [[ "${SMOKE_TEST_LOCAL:-0}" == "1" ]]; then
  # 本機 dev 模式（mock server 測試）
  BRAIN_API_URL="${BRAIN_API_URL:-http://localhost:8931}"
  BRAIN_CLOUD_URL="${BRAIN_CLOUD_URL:-http://localhost:8932}"
  ANEMONE_URL="${ANEMONE_URL:-http://localhost:8920}"
  BRAIN_API_KEY="${BRAIN_API_KEY:-test-internal-key-32plus-chars-aaaa}"
else
  # GCP prod 模式 — 必須設定
  : "${BRAIN_API_URL:?請設定 BRAIN_API_URL 環境變數（brain-api GCP URL）}"
  : "${BRAIN_CLOUD_URL:?請設定 BRAIN_CLOUD_URL 環境變數（brain-cloud GCP URL）}"
  : "${ANEMONE_URL:?請設定 ANEMONE_URL 環境變數（anemone-server GCP URL）}"
  : "${BRAIN_API_KEY:?請設定 BRAIN_API_KEY 環境變數}"
fi

BRAIN_API_URL="${BRAIN_API_URL%/}"
BRAIN_CLOUD_URL="${BRAIN_CLOUD_URL%/}"
ANEMONE_URL="${ANEMONE_URL%/}"

FAILED_ITEMS=()

# ── 顏色輸出 ──────────────────────────────────────────────────────────────────

_green() { echo -e "\033[0;32m$*\033[0m"; }
_red()   { echo -e "\033[0;31m$*\033[0m"; }
_yellow(){ echo -e "\033[0;33m$*\033[0m"; }

echo "=============================="
echo " GCP Recovery Smoke Test"
echo " 時間：$(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================="
echo ""

# ── 健康檢查函數 ──────────────────────────────────────────────────────────────

check_endpoint() {
  local name="$1"
  local url="$2"
  shift 2

  echo -n "  [$name] $url ... "

  local http_code
  if [[ $# -gt 0 ]]; then
    http_code=$(curl -sf --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" \
      -o /dev/null -w "%{http_code}" \
      "$@" \
      "$url" 2>/dev/null) || http_code="ERR"
  else
    http_code=$(curl -sf --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" \
      -o /dev/null -w "%{http_code}" \
      "$url" 2>/dev/null) || http_code="ERR"
  fi

  if [[ "$http_code" == "200" ]]; then
    _green "OK (HTTP $http_code)"
    return 0
  else
    _red "FAIL (HTTP $http_code)"
    FAILED_ITEMS+=("$name: $url (HTTP $http_code)")
    return 1
  fi
}

# ── Smoke Tests ───────────────────────────────────────────────────────────────

echo "Step 1: brain-api LLM Gateway"
check_endpoint "brain-api /v1/health" "${BRAIN_API_URL}/v1/health" || true

echo ""
echo "Step 2: brain-cloud 智腦管理層"
check_endpoint "brain-cloud /health" "${BRAIN_CLOUD_URL}/health" || true
check_endpoint "brain-cloud /health/db" "${BRAIN_CLOUD_URL}/health/db" \
  -H "X-Internal-Key: ${BRAIN_API_KEY}" || true

echo ""
echo "Step 3: brain-cloud /health/detailed（含 DB + WS 狀態）"
check_endpoint "brain-cloud /health/detailed" "${BRAIN_CLOUD_URL}/health/detailed" \
  -H "X-Internal-Key: ${BRAIN_API_KEY}" || true

echo ""
echo "Step 4: anemone-server 中繼層"
check_endpoint "anemone-server /health" "${ANEMONE_URL}/health" || true

echo ""
echo "=============================="

# ── 結果摘要 ─────────────────────────────────────────────────────────────────

if [[ ${#FAILED_ITEMS[@]} -eq 0 ]]; then
  _green " 全部通過！GCP 服務恢復正常"
  echo "=============================="
  exit 0
else
  _red " 失敗項目（${#FAILED_ITEMS[@]} 個）："
  for item in "${FAILED_ITEMS[@]}"; do
    echo "  - $item"
  done
  echo ""
  _yellow "請檢查以上服務狀態，確認 Cloud Run 是否已正常啟動。"
  _yellow "可用 'gcloud run services describe <service>' 查看詳情。"
  echo "=============================="
  exit 1
fi
