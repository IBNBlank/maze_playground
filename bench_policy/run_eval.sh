#!/usr/bin/env bash
###############################################################################
# Batch evaluation wrapper for eval.py + notify_eval.py.
#
# Nested schedule order: seed -> dataset -> algo -> use_class. Evaluates each
# trained policy under runs/[priv_]seed{seed}_{dataset_name}_{algo}/; writes
# results to runs/.../eval/. After the sweep, sends one Feishu summary card.
#
# Usage:
#   ./run_eval.sh
#   DATASET_NAME=genplan256_r2 MAZE_ALGOS="bc" MAZE_SEEDS="42" ./run_eval.sh
#
# Tunables (env vars):
#   PYTHON                 : python interpreter (default: active venv, else repo .venv)
#   DATASET_NAME           : if set, run that single dataset only
#   MAZE_ALGOS             : space-separated algos (default: bc act dp fm)
#   MAZE_SEEDS             : space-separated training seeds (default: 14 28 42)
#   MAZE_CKPT_NAME         : checkpoint file under the run dir
#                            (default: best_success_ckpt.pt)
#   NUM_EVAL_EPISODES      : episodes per job; 0 = full epoch / all samples
#                            (default: 0)
#   GOAL_TOL               : pixel L2 success threshold (default: 2.0)
#   EXTRA_ARGS             : extra CLI args forwarded to eval.py
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
MAZE_CKPT_NAME="${MAZE_CKPT_NAME:-best_success_ckpt.pt}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-0}"
GOAL_TOL="${GOAL_TOL:-2.0}"
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

echo "[run_eval] datasets=${#DATASETS[@]} algos=${MAZE_ALGOS[*]} seeds=${MAZE_SEEDS}"
if [ "${NUM_EVAL_EPISODES}" -eq 0 ]; then
	echo "[run_eval] ckpt=${MAZE_CKPT_NAME} episodes=all use_class=${USE_CLASS_VALUES[*]}"
else
	echo "[run_eval] ckpt=${MAZE_CKPT_NAME} episodes=${NUM_EVAL_EPISODES}" \
		"use_class=${USE_CLASS_VALUES[*]}"
fi
echo "[run_eval] loop order: seed -> dataset -> algo -> use_class"

for seed in ${MAZE_SEEDS}; do
	for dataset in "${DATASETS[@]}"; do
		for algo in "${MAZE_ALGOS[@]}"; do
			for use_class in "${USE_CLASS_VALUES[@]}"; do
				if [ "${use_class}" -eq 1 ]; then
					USE_CLASS_FLAG="--use-class"
					run_name="priv_seed${seed}_${dataset}_${algo}"
				else
					USE_CLASS_FLAG="--no-use-class"
					run_name="seed${seed}_${dataset}_${algo}"
				fi
				ckpt="runs/${run_name}/${MAZE_CKPT_NAME}"
				echo "######################################################################"
				echo "[run_eval] === seed=${seed} dataset=${dataset}" \
					"algo=${algo} use_class=${use_class} ==="
				echo "######################################################################"
				if [ ! -f "${ckpt}" ]; then
					echo "[run_eval] skip: ckpt not found: ${ckpt}"
					continue
				fi

				# shellcheck disable=SC2086
				"${PYTHON}" eval.py \
					--algo "${algo}" \
					--dataset-name "${dataset}" \
					--seed "${seed}" \
					--ckpt-name "${MAZE_CKPT_NAME}" \
					--num-eval "${NUM_EVAL_EPISODES}" \
					--goal-tol "${GOAL_TOL}" \
					${USE_CLASS_FLAG} \
					${EXTRA_ARGS}
				code=$?

				if [ "${code}" -eq 0 ]; then
					echo "[run_eval] seed=${seed} dataset=${dataset}" \
						"algo=${algo} use_class=${use_class}: finished cleanly."
				elif [ "${code}" -eq 130 ]; then
					echo "[run_eval] interrupted by user (Ctrl-C). stopping."
					exit 130
				else
					echo "[run_eval] seed=${seed} dataset=${dataset}" \
						"algo=${algo} use_class=${use_class}: exited abnormally" \
						"(exit ${code}); continuing."
				fi
			done
		done
	done
done

echo "######################################################################"
echo "[run_eval] summarizing results and sending Feishu notification..."
echo "######################################################################"
# shellcheck disable=SC2086
"${PYTHON}" notify_eval.py \
	--seeds ${MAZE_SEEDS} \
	--algos "${MAZE_ALGOS[@]}" \
	--datasets "${DATASETS[@]}"

echo "[run_eval] all jobs finished. done."
