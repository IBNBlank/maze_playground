#!/usr/bin/env bash
###############################################################################
# Build flat map/state/action_chunk training shards from demons (data_set.py).
#
# Reads demons/{id}/ for every listed demons id, expands each expert route into
# one sample, and writes ../datasets/{dataset_name}/dataset/data_*.npz
# (2048 samples per shard) plus dataset.json and idx/epoch_*.npy (300 epochs).
#
# Usage:
#   ./run_set.sh
#   DATASET_NAME=genplan256_r2 DEMONS_IDS="genplan256_r2" ./run_set.sh
#   SIZE=128 ./run_set.sh
#   SHARD_SIZE=1024 NUM_IDX_PERMS=100 ./run_set.sh
#
# Tunables (env vars):
#   PYTHON        : python interpreter (default: active venv, else repo .venv)
#   SIZE          : map side length used in default job names (default: 256)
#   DEMONS_IDS    : space-separated demons subdir names; when set with
#                   DATASET_NAME, overrides to a single custom job
#   DATASET_NAME  : output subdir under ../datasets/ (used with DEMONS_IDS)
#   DEMONS_ROOT   : demons root directory (default: ../demons)
#   SHARD_SIZE    : samples per shard npz (default: 2048)
#   NUM_IDX_PERMS : shuffled idx/epoch_XXX.npy count (default: 300)
#   IDX_PERM_SEED : base seed for epoch permutations (default: 0)
#   ACTION_HORIZON: expected action chunk length (default: 72)
###############################################################################
set -u

cd "$(dirname "$0")" || exit 1

if [ -n "${PYTHON:-}" ]; then
	:
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
	PYTHON="${VIRTUAL_ENV}/bin/python"
else
	PYTHON="../.venv/bin/python"
fi

DEMONS_ROOT="${DEMONS_ROOT:-../demons}"
SHARD_SIZE="${SHARD_SIZE:-2048}"
NUM_IDX_PERMS="${NUM_IDX_PERMS:-300}"
IDX_PERM_SEED="${IDX_PERM_SEED:-0}"
ACTION_HORIZON="${ACTION_HORIZON:-72}"

# Each job: "dataset_name|demons_id [demons_id ...]"
# DEMONS_IDS / DATASET_NAME override to a single custom job when set.
if [ -n "${DEMONS_IDS:-}" ]; then
	DATASET_JOBS=("${DATASET_NAME:-default}|${DEMONS_IDS}")
else
	DATASET_JOBS=(
		# "genplan256_r2|genplan256_r2"
		# "genplan256_r3|genplan256_r3"
		# "genplan256_r4|genplan256_r4"
		"genplan256_mix|genplan256_r2 genplan256_r3 genplan256_r4"
	)
fi

if [ ! -x "${PYTHON}" ] && ! command -v "${PYTHON}" >/dev/null 2>&1; then
	echo "[run_set] python not found: ${PYTHON}" >&2
	echo "[run_set] create the venv with ../venv.sh or set PYTHON=..." >&2
	exit 1
fi

for job in "${DATASET_JOBS[@]}"; do
	DATASET_NAME="${job%%|*}"
	# shellcheck disable=SC2206
	DEMONS_LIST=(${job#*|})

	echo "######################################################################"
	echo "[run_set] dataset_name=${DATASET_NAME}"
	echo "[run_set] demons_ids=${DEMONS_LIST[*]}"
	echo "[run_set] demons_root=${DEMONS_ROOT}"
	echo "[run_set] shard_size=${SHARD_SIZE}"
	echo "[run_set] num_idx_perms=${NUM_IDX_PERMS} idx_perm_seed=${IDX_PERM_SEED}"
	echo "[run_set] action_horizon=${ACTION_HORIZON}"
	echo "[run_set] out=../datasets/${DATASET_NAME}/"
	echo "######################################################################"

	missing=0
	for demons_id in "${DEMONS_LIST[@]}"; do
		if [ ! -f "${DEMONS_ROOT}/${demons_id}/manifest.json" ]; then
			echo "[run_set] warn: missing ${DEMONS_ROOT}/${demons_id}/manifest.json"
			missing=$((missing + 1))
		fi
	done
	if [ "${missing}" -eq "${#DEMONS_LIST[@]}" ]; then
		echo "[run_set] error: no manifest.json found for any demons_id."
		exit 1
	fi

	"${PYTHON}" data_set.py \
		--demons-ids "${DEMONS_LIST[@]}" \
		--demons-root "${DEMONS_ROOT}" \
		--dataset-name "${DATASET_NAME}" \
		--shard-size "${SHARD_SIZE}" \
		--num-idx-perms "${NUM_IDX_PERMS}" \
		--idx-perm-seed "${IDX_PERM_SEED}" \
		--action-horizon "${ACTION_HORIZON}"
	code=$?

	if [ "${code}" -eq 0 ]; then
		echo "[run_set] finished ${DATASET_NAME}."
	elif [ "${code}" -eq 130 ]; then
		echo "[run_set] interrupted by user (Ctrl-C). stopping."
		exit 130
	else
		echo "[run_set] ${DATASET_NAME} exited abnormally (exit ${code})."
		exit "${code}"
	fi
done

echo "[run_set] all ${#DATASET_JOBS[@]} dataset(s) finished cleanly."
