#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; }

info "Starting Image-to-CSV (GLM-OCR)..."
sudo docker compose up -d --build

# ── Cloudflare Tunnel ─────────────────────────────────────────────────────────
pkill cloudflared 2>/dev/null || true
sleep 1
cloudflared tunnel --config ~/.cloudflared/config.yml run > /tmp/cloudflared.log 2>&1 &
info "Cloudflare tunnel started (csv.synopustech.com → localhost:9001)"

echo ""
info "Containers starting. vLLM model may take a few minutes to load."
info "App available locally:  http://localhost:9001"
info "App available publicly: https://csv.synopustech.com"
echo ""
sudo docker compose ps
