#!/bin/bash
# SubZero: gpt2-xl on QNLI
# Model: gpt2-xl
# Task: qnli (GLUE)
# Algorithm: SubZero
# Hyperparameters are shared unchanged across GPT-2-XL tasks.


# Change to repository root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.." || exit 1


# GPU Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Hugging Face Mirror (for faster downloads in China)
unset http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
MODEL="gpt2-xl"
TASK="qnli"
ALGORITHM="SubZero"

# Shared GPT-2-XL hyperparameters
LR=5e-6
EPOCHS=1000
NUM_QUERIES=6
BATCH_SIZE=64
BATCH_SIZE_LIMIT=64
SAMPLE_SIZE=512
SAMPLE_SIZE_VAL=256
MAX_SEQ_LENGTH=128

# SubZero-specific parameters
RANK=8
STEP_INTERVAL=1000

# Other settings
DEVICE=0
TRIAL=0
SEED=42

python main.py \
    --epochs $EPOCHS \
    --model_name $MODEL \
    --task $TASK \
    --algorithm $ALGORITHM \
    --lr $LR \
    --weight_decay 0.0 \
    --num_queries $NUM_QUERIES \
    --perturbation_scale 1e-3 \
    --rank $RANK \
    --step_interval $STEP_INTERVAL \
    --samplesize $SAMPLE_SIZE \
    --samplesize_validation $SAMPLE_SIZE_VAL \
    --batchsize $BATCH_SIZE \
    --batchsize_limit $BATCH_SIZE_LIMIT \
    --full_parameter \
    --max_seq_length $MAX_SEQ_LENGTH \
    --device $DEVICE \
    --save_every 10000000 \
    --logging "wandb+csv" \
    --init_seed $SEED \
    --trial $TRIAL \
    --early_stopping --patience 1000
