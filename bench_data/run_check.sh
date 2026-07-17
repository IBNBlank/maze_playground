#!/usr/bin/env bash
###############################################################################
# Wrapper for data_check.py — visualize the same leading maps as preview.
#
# Runs once per entry in NUM_ROUTES_LIST. Each run reads
# ${DATASET_DIR}_r${num_routes} and writes check_robots.png inside it.
#
# Usage:
#   ./run_check.sh
#   SIZE=128 NUM_ROUTES_LIST="2 4 6" ./run_check.sh
#   ./run_check.sh --preview-count 8
#
# Extra CLI args are forwarded to data_check.py. Loop-controlled flags
# (--dataset-dir) are applied last and win.
#
# Tunables (env vars):
#   PYTHON           : python interpreter (default: repo .venv)
#   SIZE             : map side length used in dataset name (default: 256)
#   NUM_ROUTES_LIST  : space-separated route counts (default: 2 3 4 5 6)
#   DATASET_DIR      : dataset base dir (default: ../dataset/genplan${SIZE});
#                      each run uses ${DATASET_DIR}_r${num_routes}
#   PREVIEW_COUNT    : number of leading maps to visualize (default: 16)
###############################################################################
set -euo pipefail

cd "$(dirname "$0")" || exit 1

PYTHON="${PYTHON:-../.venv/bin/python}"
SIZE="${SIZE:-256}"
NUM_ROUTES_LIST="${NUM_ROUTES_LIST:-2 3 4 5 6}"
DATASET_DIR="${DATASET_DIR:-../dataset/genplan${SIZE}}"
PREVIEW_COUNT="${PREVIEW_COUNT:-16}"

if [ ! -x "${PYTHON}" ] && ! command -v "${PYTHON}" >/dev/null 2>&1; then
	echo "[run_check] python not found: ${PYTHON}" >&2
	echo "[run_check] create the venv with ../venv.sh or set PYTHON=..." >&2
	exit 1
fi

for num_routes in ${NUM_ROUTES_LIST}; do
	dataset_dir="${DATASET_DIR}_r${num_routes}"

	echo "######################################################################"
	echo "[run_check] dataset_dir=${dataset_dir}"
	echo "[run_check] preview_count=${PREVIEW_COUNT}"
	echo "######################################################################"

	if [ ! -f "${dataset_dir}/manifest.json" ]; then
		echo "[run_check] skip: manifest not found -> ${dataset_dir}" >&2
		continue
	fi

	"${PYTHON}" data_check.py \
		--preview-count "${PREVIEW_COUNT}" \
		"$@" \
		--dataset-dir "${dataset_dir}"
	code=$?

	if [ "${code}" -eq 0 ]; then
		echo "[run_check] num_routes=${num_routes}: finished cleanly -> ${dataset_dir}"
	elif [ "${code}" -eq 130 ]; then
		echo "[run_check] interrupted by user (Ctrl-C)."
		exit 130
	else
		echo "[run_check] num_routes=${num_routes}: exited abnormally (exit ${code})." >&2
		exit "${code}"
	fi
done

echo "[run_check] all num_routes finished. done."
