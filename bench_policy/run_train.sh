#!/usr/bin/env bash
###############################################################################
# Batch training wrapper for train.py (TrainArgs CLI).
#
# Nested schedule order: seed -> dataset -> algo.
#
# Usage:
#   ./run_train.sh
#   DATASET_NAME=genplan256_r4 MAZE_ALGOS="bc" MAZE_SEEDS="1 2" \
#     EPOCHS=50 ./run_train.sh
#
# Tunables (env vars):
#   PYTHON                 : python interpreter (default: active venv, else repo .venv)
#   DATASET_NAME           : if set, run that single dataset only
#   MAZE_ALGOS             : space-separated algos (default: bc)
#   MAZE_SEEDS             : space-separated seeds (default: 42)
#   EPOCHS                 : training epochs (default: 50)
#   EVAL_FREQ              : eval every N epochs (default: 5)
#   MAX_CONSECUTIVE_FAILS  : abort after this many hard crashes (default: 5)
#   EXTRA_ARGS             : extra CLI args forwarded to train.py
###############################################################################
set -u

cd "$(dirname "$0")" || exit 1

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${PYTHON:-}" ]; then
	:
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
	PYTHON="${VIRTUAL_ENV}/bin/python"
else
	PYTHON="${REPO_DIR}/.venv/bin/python"
fi

read -r -a MAZE_ALGOS <<< "${MAZE_ALGOS:-bc}"
MAZE_SEEDS="${MAZE_SEEDS:-42}"
EPOCHS="${EPOCHS:-50}"
EVAL_FREQ="${EVAL_FREQ:-5}"
MAX_CONSECUTIVE_FAILS="${MAX_CONSECUTIVE_FAILS:-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -n "${DATASET_NAME:-}" ]; then
	DATASETS=("${DATASET_NAME}")
else
	DATASETS=(
		"genplan256_r2"
		"genplan256_r3"
		"genplan256_r4"
		"genplan256_r5"
		"genplan256_r6"
	)
fi

echo "[run_train] datasets=${#DATASETS[@]} algos=${MAZE_ALGOS[*]} seeds=${MAZE_SEEDS}"
echo "[run_train] epochs=${EPOCHS} eval_freq=${EVAL_FREQ}"
echo "[run_train] loop order: seed -> dataset -> algo"

for seed in ${MAZE_SEEDS}; do
	for dataset in "${DATASETS[@]}"; do
		DATASET_DIR="${REPO_DIR}/dataset/${dataset}"

		if [ ! -f "${DATASET_DIR}/manifest.json" ]; then
			echo "[run_train] error: missing ${DATASET_DIR}/manifest.json"
			echo "[run_train] build it first via bench_data/run_gen.sh"
			exit 1
		fi

		for algo in "${MAZE_ALGOS[@]}"; do
			echo "######################################################################"
			echo "[run_train] === seed=${seed} dataset=${dataset} algo=${algo} ==="
			echo "######################################################################"

			fails=0
			# shellcheck disable=SC2086
			"${PYTHON}" train.py \
				--algo "${algo}" \
				--dataset-name "${dataset}" \
				--seed "${seed}" \
				--epochs "${EPOCHS}" \
				--eval-freq "${EVAL_FREQ}" \
				${EXTRA_ARGS}
			code=$?

			if [ "${code}" -eq 0 ]; then
				echo "[run_train] seed=${seed} dataset=${dataset}" \
					"algo=${algo}: finished cleanly."
			elif [ "${code}" -eq 130 ]; then
				echo "[run_train] interrupted by user (Ctrl-C). stopping."
				exit 130
			else
				fails=$((fails + 1))
				echo "[run_train] seed=${seed} dataset=${dataset}" \
					"algo=${algo}: exited abnormally (exit ${code});" \
					"consecutive failures=${fails}/${MAX_CONSECUTIVE_FAILS}."
				if [ "${fails}" -ge "${MAX_CONSECUTIVE_FAILS}" ]; then
					echo "[run_train] too many consecutive failures; aborting."
					exit "${code}"
				fi
			fi
		done
	done
done

echo "[run_train] all jobs finished. done."
