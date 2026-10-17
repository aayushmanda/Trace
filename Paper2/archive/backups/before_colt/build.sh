#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
tectonic -X compile main.tex --outdir build --keep-logs
cp build/main.pdf build/trace.pdf
