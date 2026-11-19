#!/usr/bin/env bash
# Run the whole comparison: generate the data, fit it with every tool, and
# write results/RESULTS.md. Needs a Python with numpy, scipy, iminuit and zfit
# (PYTHON, default `python3`), and a Python with PyROOT (ROOT_PYTHON, default
# `python3`) — see README.md.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
ROOT_PYTHON=${ROOT_PYTHON:-python3}

"$PYTHON" make_data.py data
mkdir -p results
cargo run -q --release -p oxifit --example fit_comparison -- data > results/oxifit.jsonl
cargo run -q --release -p oxifit --features rayon --example fit_comparison -- data > results/oxifit_rayon.jsonl
"$PYTHON" fit_iminuit.py data > results/iminuit.jsonl
"$PYTHON" fit_scipy.py data > results/scipy.jsonl
"$PYTHON" fit_zfit.py data 2>/dev/null > results/zfit.jsonl
"$ROOT_PYTHON" fit_root.py data > results/root.jsonl
"$PYTHON" report.py results/*.jsonl > results/RESULTS.md
echo "wrote results/RESULTS.md"
