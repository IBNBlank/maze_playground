#!/usr/bin/env bash
###############################################################################
# Wrapper for data_gen.py — generate multimodal planning map shards.
#
# Runs once per entry in NUM_ROUTES_LIST. Each run writes a separate dataset
# with seed = num_routes * 14.
#
# Usage:
#   ./run_gen.sh
#   NUM_MAPS=500 SIZE=128 NUM_ROUTES_LIST="2 4 6" ./run_gen.sh
#   ./run_gen.sh --preview-count 8
#
# Extra CLI args are forwarded to data_gen.py. Loop-controlled flags
# (--num-routes, --seed, --output-dir) are applied last and win.
#
# Tunables (env vars):
#   PYTHON           : python interpreter (default: active venv, else repo .venv)
#   NUM_MAPS         : number of maps (default: 1000)
#   SIZE             : map side length (default: 256)
#   NUM_ROUTES_LIST  : space-separated route counts (default: 2 3 4 5 6)
#   ACTION_HORIZON   : fixed action chunk length (default: 72); trailing zeros after goal
#   OUTPUT_DIR       : output base dir (default: data/genplan${SIZE});
#                      each run uses ${OUTPUT_DIR}_r${num_routes}
#   SHARD_SIZE       : maps per NPZ shard (default: 100)
#   ROBOT_RADIUS     : clearance inflation radius (default: 2)
#   PREVIEW_COUNT    : preview collage tiles (default: 16)
#   MAX_MAP_ATTEMPTS : retries per accepted map (default: 80)
###############################################################################
set -euo pipefail

cd "$(dirname "$0")" || exit 1

if [ -n "${PYTHON:-}" ]; then
	:
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
	PYTHON="${VIRTUAL_ENV}/bin/python"
else
	PYTHON="../.venv/bin/python"
fi
NUM_MAPS="${NUM_MAPS:-5000}"
SIZE="${SIZE:-256}"
NUM_ROUTES_LIST="${NUM_ROUTES_LIST:-2}"
ACTION_HORIZON="${ACTION_HORIZON:-72}"
OUTPUT_DIR="${OUTPUT_DIR:-../demons/genplan${SIZE}}"
SHARD_SIZE="${SHARD_SIZE:-100}"
ROBOT_RADIUS="${ROBOT_RADIUS:-5}"
PREVIEW_COUNT="${PREVIEW_COUNT:-16}"
MAX_MAP_ATTEMPTS="${MAX_MAP_ATTEMPTS:-80}"

if [ ! -x "${PYTHON}" ] && ! command -v "${PYTHON}" >/dev/null 2>&1; then
	echo "[run_gen] python not found: ${PYTHON}" >&2
	echo "[run_gen] create the venv with ../venv.sh or set PYTHON=..." >&2
	exit 1
fi

for num_routes in ${NUM_ROUTES_LIST}; do
	seed=$((num_routes * 14))
	out_dir="${OUTPUT_DIR}_r${num_routes}"

	echo "######################################################################"
	echo "[run_gen] num_maps=${NUM_MAPS} size=${SIZE} num_routes=${num_routes}"
	echo "[run_gen] output_dir=${out_dir} seed=${seed}"
	echo "######################################################################"

	"${PYTHON}" data_gen.py \
		--num-maps "${NUM_MAPS}" \
		--size "${SIZE}" \
		--action-horizon "${ACTION_HORIZON}" \
		--shard-size "${SHARD_SIZE}" \
		--robot-radius "${ROBOT_RADIUS}" \
		--preview-count "${PREVIEW_COUNT}" \
		--max-map-attempts "${MAX_MAP_ATTEMPTS}" \
		"$@" \
		--num-routes "${num_routes}" \
		--seed "${seed}" \
		--output-dir "${out_dir}"
	code=$?

	if [ "${code}" -eq 0 ]; then
		echo "[run_gen] num_routes=${num_routes}: finished cleanly -> ${out_dir}"
	elif [ "${code}" -eq 130 ]; then
		echo "[run_gen] interrupted by user (Ctrl-C)."
		exit 130
	else
		echo "[run_gen] num_routes=${num_routes}: exited abnormally (exit ${code})." >&2
		exit "${code}"
	fi
done

echo "[run_gen] all num_routes finished. done."
