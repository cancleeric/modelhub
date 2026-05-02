#!/usr/bin/env bash
# Sprint 7.5 — 版本 bump 工具
# 用法：./scripts/bump-version.sh [patch|minor|major]
# 預設 patch

set -euo pipefail
cd "$(dirname "$0")/.."

LEVEL="${1:-patch}"
NEW=$(python3 backend/version.py bump "$LEVEL")
FRONT="frontend/package.json"

# 前端 package.json 同步（透過 jq，若無 jq 則改 python）
if command -v jq >/dev/null 2>&1; then
  TMP=$(mktemp)
  jq --arg v "$NEW" '.version = $v' "$FRONT" > "$TMP" && mv "$TMP" "$FRONT"
else
  python3 -c "import json,sys; p='$FRONT'; d=json.load(open(p)); d['version']='$NEW'; json.dump(d, open(p,'w'), indent=2)"
fi

echo "Bumped to $NEW (VERSION + package.json)"
echo "$NEW"
