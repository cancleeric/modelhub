#!/usr/bin/env bash
# start.sh — modelhub 啟動腳本，自動從 Hurricane Vault 注入 key
set -e
cd "$(dirname "$0")"
DATABASE_URL="${DATABASE_URL:-sqlite:////app/data/modelhub.db}" \
hvault run --prefix "hurricanecore/dev/" -- docker-compose up -d "$@"
