#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync
uv run python handcoded/lettertrace.py task=count word_len=6 mod=2 steps=8000
