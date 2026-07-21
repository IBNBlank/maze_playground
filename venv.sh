#!/usr/bin/env bash
set -Eeuo pipefail
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-04-28
################################################################

CUR_DIR="$(pwd)"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "CUR_DIR: $CUR_DIR"
echo "SCRIPT_DIR: $SCRIPT_DIR"

cd $SCRIPT_DIR

if ! command -v uv >/dev/null 2>&1; then
	echo "Error: uv not found. Please install uv first." >&2
	exit 1
fi

if [ ! -d .venv ]; then
	uv venv --python 3.10
fi
source .venv/bin/activate

# Install 
uv pip install torch torchvision numpy opencv-python scikit-image tyro tqdm

cd $CUR_DIR
