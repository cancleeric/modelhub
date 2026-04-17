#!/usr/bin/env bash
# inference-service.sh — ModelHub 推論 server launchd 管理腳本（Sprint 8.1）
#
# 使用方式：
#   ./scripts/inference-service.sh install    # 安裝並啟動
#   ./scripts/inference-service.sh start      # 啟動
#   ./scripts/inference-service.sh stop       # 停止
#   ./scripts/inference-service.sh status     # 查狀態
#   ./scripts/inference-service.sh uninstall  # 卸載
#   ./scripts/inference-service.sh logs       # 顯示最新 log

set -euo pipefail

LABEL="tw.hurricanecore.modelhub.inference"
PLIST_SRC="$(dirname "$0")/../launchd/${LABEL}.plist"
PLIST_DST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_OUT="$HOME/HurricaneCore/modelhub/logs/inference_server.log"
LOG_ERR="$HOME/HurricaneCore/modelhub/logs/inference_server_err.log"

CMD="${1:-status}"

case "$CMD" in
  install)
    echo "Installing $LABEL to LaunchAgents..."
    mkdir -p "$HOME/Library/LaunchAgents"
    mkdir -p "$(dirname "$LOG_OUT")"
    cp -f "$PLIST_SRC" "$PLIST_DST"
    launchctl load -w "$PLIST_DST"
    echo "Done. Service loaded."
    ;;
  start)
    launchctl start "$LABEL"
    echo "Service started."
    ;;
  stop)
    launchctl stop "$LABEL"
    echo "Service stopped."
    ;;
  restart)
    launchctl stop "$LABEL" 2>/dev/null || true
    sleep 2
    launchctl start "$LABEL"
    echo "Service restarted."
    ;;
  status)
    echo "=== launchctl list ==="
    launchctl list | grep modelhub || echo "(not loaded)"
    echo ""
    echo "=== health check ==="
    curl -sf http://localhost:8951/health && echo "" || echo "health check FAILED"
    ;;
  uninstall)
    launchctl unload -w "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "Service uninstalled."
    ;;
  logs)
    echo "=== stdout ==="
    tail -50 "$LOG_OUT" 2>/dev/null || echo "(no log)"
    echo "=== stderr ==="
    tail -20 "$LOG_ERR" 2>/dev/null || echo "(no log)"
    ;;
  *)
    echo "Usage: $0 {install|start|stop|restart|status|uninstall|logs}"
    exit 1
    ;;
esac
