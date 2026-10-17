#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build
NAME="${1:-long_horizon_search_poster}"
tectonic -X compile "$NAME.tex" --outdir build --keep-logs
cp "build/$NAME.pdf" "$NAME.pdf"
pdftoppm -singlefile -scale-to 2600 -png "$NAME.pdf" "build/$NAME"
