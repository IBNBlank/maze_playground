#!/usr/bin/env bash
###############################################################################
# Batch training wrapper for train.py + notify_train.py.
#
# Nested schedule order: seed -> dataset -> algo -> use_class.
#
# After the sweep, sends one Feishu summary card.
#
# Usage:
#   ./run_train.sh
#   DATASET_NAME=genplan256_r2 MAZE_ALGOS="bc" MAZE_SEEDS="1 2" \
#     EPOCHS=50 ./run_train.sh
#
# Tunables (env vars):
#   PYTHON                 : python interpreter (default: active venv, else repo .venv)
#   DATASET_NAME           : if set, run that single dataset only
#   MAZE_ALGOS             : space-separated algos (default: bc act dp fm)
#   MAZE_SEEDS             : space-separated seeds (default: 14 28 42)
#   EPOCHS                 : training epochs (default: 500)
#   EVAL_FREQ              : eval every N epochs (default: 5)
#   NUM_EVAL_EPISODES      : mid-train eval episodes; 0 = full epoch
#                            (default: 500)
#   GOAL_TOL               : pixel L2 success threshold (default: 2.0)
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

read -r -a MAZE_ALGOS <<< "${MAZE_ALGOS:-bc act dp fm}"
MAZE_SEEDS="${MAZE_SEEDS:-14 28 42}"
EPOCHS="${EPOCHS:-500}"
EVAL_FREQ="${EVAL_FREQ:-5}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-500}"
GOAL_TOL="${GOAL_TOL:-2.0}"
MAX_CONSECUTIVE_FAILS="${MAX_CONSECUTIVE_FAILS:-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
USE_CLASS_VALUES=(0 1)

if [ -n "${DATASET_NAME:-}" ]; then
	DATASETS=("${DATASET_NAME}")
else
	DATASETS=(
		# "genplan256_mix"
		"genplan256_r2"
	)
fi

echo "[run_train] datasets=${#DATASETS[@]} algos=${MAZE_ALGOS[*]} seeds=${MAZE_SEEDS}"
echo "[run_train] epochs=${EPOCHS} eval_freq=${EVAL_FREQ} use_class=${USE_CLASS_VALUES[*]}"
echo "[run_train] loop order: seed -> dataset -> algo -> use_class"

for seed in ${MAZE_SEEDS}; do
	for dataset in "${DATASETS[@]}"; do
		DATASET_DIR="${REPO_DIR}/datasets/${dataset}"

		if [ ! -f "${DATASET_DIR}/dataset.json" ]; then
			echo "[run_train] error: missing ${DATASET_DIR}/dataset.json"
			echo "[run_train] build it first via bench_data/run_set.sh"
			exit 1
		fi

		for algo in "${MAZE_ALGOS[@]}"; do
			for use_class in "${USE_CLASS_VALUES[@]}"; do
				if [ "${use_class}" -eq 1 ]; then
					USE_CLASS_FLAG="--use-class"
				else
					USE_CLASS_FLAG="--no-use-class"
				fi

				echo "######################################################################"
				echo "[run_train] === seed=${seed} dataset=${dataset} algo=${algo} use_class=${use_class} ==="
				echo "######################################################################"

				fails=0
				# shellcheck disable=SC2086
				"${PYTHON}" train.py \
					--algo "${algo}" \
					--dataset-name "${dataset}" \
					--seed "${seed}" \
					--epochs "${EPOCHS}" \
					--eval-freq "${EVAL_FREQ}" \
					--num-eval "${NUM_EVAL_EPISODES}" \
					--goal-tol "${GOAL_TOL}" \
					${USE_CLASS_FLAG} \
					${EXTRA_ARGS}
				code=$?

				if [ "${code}" -eq 0 ]; then
					fails=0
					echo "[run_train] seed=${seed} dataset=${dataset}" \
						"algo=${algo} use_class=${use_class}: finished cleanly."
				elif [ "${code}" -eq 130 ]; then
					echo "[run_train] interrupted by user (Ctrl-C). stopping."
					exit 130
				else
					fails=$((fails + 1))
					echo "[run_train] seed=${seed} dataset=${dataset}" \
						"algo=${algo} use_class=${use_class}: exited abnormally (exit ${code});" \
						"consecutive failures=${fails}/${MAX_CONSECUTIVE_FAILS}."
					if [ "${fails}" -ge "${MAX_CONSECUTIVE_FAILS}" ]; then
						echo "[run_train] too many consecutive failures; aborting."
						exit "${code}"
					fi
				fi
			done
		done
	done
done

echo "######################################################################"
echo "[run_train] sending Feishu notification..."
echo "######################################################################"
# shellcheck disable=SC2086
"${PYTHON}" notify_train.py \
	--seeds ${MAZE_SEEDS} \
	--algos "${MAZE_ALGOS[@]}" \
	--datasets "${DATASETS[@]}"

echo "[run_train] all jobs finished. done."
