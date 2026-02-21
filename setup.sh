#!/bin/bash
# ──────────────────────────────────────────────────────
# Setup script for Offline AI Code Generator
# Run this ONCE with internet to download everything.
# After that, the tool works fully offline.
# ──────────────────────────────────────────────────────

set -e

echo "═══════════════════════════════════════════════════"
echo "  Offline AI Code Generator — Setup"
echo "═══════════════════════════════════════════════════"

# 1. Create virtual environment
echo ""
echo "[1/4] Creating virtual environment …"
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "[2/4] Installing dependencies …"
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create directories
echo "[3/4] Creating directories …"
mkdir -p models/codellama-7b models/embeddings data/chroma_db knowledge_base generated_projects

# 4. Download model
echo "[4/4] Downloading CodeLlama-7B model (this may take a while) …"
python main.py --download

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Setup complete! Run with: python main.py"
echo "═══════════════════════════════════════════════════"
