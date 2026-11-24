#!/usr/bin/env bash
set -e
python scripts/run_experiments.py --mode quickbert --config configs/default.yaml "$@"
