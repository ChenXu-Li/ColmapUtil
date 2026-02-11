#!/bin/bash

set -e

echo "[Setup] Activate environment (optional)"
# 按需启用你的环境，例如：
# source ~/miniconda3/bin/activate pc_align

echo "[Deps] Install Python packages (open3d, pyyaml, numpy, tqdm) if missing"
pip install -q open3d pyyaml numpy tqdm

mkdir -p output

echo "[Run] align.py with config.yaml"
python align.py

