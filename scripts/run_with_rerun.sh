#!/bin/bash
# InstantSplat with Rerun 3D Visualization
# Usage: ./scripts/run_with_rerun.sh <DATA_ROOT> <SCENE> [N_VIEWS] [ITERATIONS]

DATA_ROOT="${1:?Usage: $0 <DATA_ROOT> <SCENE> [N_VIEWS] [ITERATIONS]}"
SCENE="${2:?Usage: $0 <DATA_ROOT> <SCENE> [N_VIEWS] [ITERATIONS]}"
N_VIEWS="${3:-3}"
GS_TRAIN_ITER="${4:-1000}"

SOURCE_PATH="${DATA_ROOT}/${SCENE}/"
MODEL_PATH="./output_rerun/${SCENE}/${N_VIEWS}_views"
mkdir -p "${MODEL_PATH}"

echo "=== InstantSplat + Rerun ==="
echo "Scene: ${SCENE} | Views: ${N_VIEWS} | Iters: ${GS_TRAIN_ITER}"

# (1) Geometry Initialization
python -W ignore ./init_geo.py \
    -s "${SOURCE_PATH}" \
    -m "${MODEL_PATH}" \
    --n_views ${N_VIEWS} \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking \
    --rerun

# (2) Training
python ./train.py \
    -s "${SOURCE_PATH}" \
    -m "${MODEL_PATH}" \
    -r 1 \
    --n_views ${N_VIEWS} \
    --iterations ${GS_TRAIN_ITER} \
    --pp_optimizer \
    --optim_pose \
    --rerun \
    --rerun_log_freq 100

echo "=== Done. Check Rerun viewer. ==="
