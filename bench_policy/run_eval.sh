#!/usr/bin/env bash
###############################################################################
# Batch evaluation wrapper for eval.py (EvalArgs CLI).
#
# Nested schedule order: seed -> dataset -> algo. Evaluates each trained
# policy under runs/Seed{seed}_{dataset_name}_{algo}/.
#
# Metrics: collision_rate / success_rate (percent), success_average_steps.
#
# Usage:
#   ./run_eval.sh
#   DATASET_NAME=genplan256_r4 MAZE_ALGOS="bc" MAZE_SEEDS="42" ./run_eval.sh
#
# Tunables (env vars):
#   PYTHON                 : python interpreter (default: active venv, else repo .venv)
#   DATASET_NAME           : if set, run that single dataset only
#   MAZE_ALGOS             : space-separated algos (default: bc)
#   MAZE_SEEDS             : space-separated training seeds (default: 42)
#   MAZE_CKPT_NAME         : checkpoint file under the run dir
#                            (default: best_success_ckpt.pt)
#   NUM_EVAL_EPISODES      : episodes per job (default: 100)
#   GOAL_TOL               : pixel L2 success threshold (default: 1.0)
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

read -r -a MAZE_ALGOS <<< "${MAZE_ALGOS:-bc}"
MAZE_SEEDS="${MAZE_SEEDS:-42}"
MAZE_CKPT_NAME="${MAZE_CKPT_NAME:-best_success_ckpt.pt}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-100}"
GOAL_TOL="${GOAL_TOL:-1.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -n "${DATASET_NAME:-}" ]; then
	DATASETS=("${DATASET_NAME}")
else
	DATASETS=(
		"genplan256_mix"
	)
fi

echo "[run_eval] datasets=${#DATASETS[@]} algos=${MAZE_ALGOS[*]} seeds=${MAZE_SEEDS}"
echo "[run_eval] ckpt=${MAZE_CKPT_NAME} episodes=${NUM_EVAL_EPISODES} goal_tol=${GOAL_TOL}"
echo "[run_eval] loop order: seed -> dataset -> algo"

for seed in ${MAZE_SEEDS}; do
	for dataset in "${DATASETS[@]}"; do
		for algo in "${MAZE_ALGOS[@]}"; do
			run_name="Seed${seed}_${dataset}_${algo}"
			ckpt="runs/${run_name}/${MAZE_CKPT_NAME}"
			echo "######################################################################"
			echo "[run_eval] === seed=${seed} dataset=${dataset} algo=${algo} ==="
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
				${EXTRA_ARGS}
			code=$?

			if [ "${code}" -eq 0 ]; then
				echo "[run_eval] seed=${seed} dataset=${dataset}" \
					"algo=${algo}: finished cleanly."
			elif [ "${code}" -eq 130 ]; then
				echo "[run_eval] interrupted by user (Ctrl-C). stopping."
				exit 130
			else
				echo "[run_eval] seed=${seed} dataset=${dataset}" \
					"algo=${algo}: exited abnormally (exit ${code}); continuing."
			fi
		done
	done
done

echo "[run_eval] all jobs finished. done."
