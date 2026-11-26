#!/usr/bin/env bash
set -euo pipefail

export PROBCOUNT_ENABLE_EXACT=true
export PROBCOUNT_USE_MEMORY_STORAGE=true
export PROBCOUNT_API_URL="http://127.0.0.1:8000"

python scripts/run_demo.py
