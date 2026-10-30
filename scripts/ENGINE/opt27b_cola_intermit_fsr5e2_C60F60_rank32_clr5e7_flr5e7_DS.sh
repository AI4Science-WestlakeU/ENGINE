#!/bin/bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

export WANDB_MODE="${WANDB_MODE:-offline}"

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.." || exit 1


LOG_FILE="/tmp/opt27b_cola_intermit_fsr5e2_C60F60_rank32_clr5e7_flr5e7_DS_$(date +%Y%m%d_%H%M%S).log"


python main.py \
    --algorithm ENGINE \
    --model_name facebook/opt-2.7b \
    --task cola \
    --lr 5e-7 \
    --weight_decay 0.0 \
    --num_queries 6 \
    --perturbation_scale 1e-3 \
    --damping 0.1 \
    --lr_cov 1e-3 \
    --samplesize 512 \
    --samplesize_validation 256 \
    --batchsize 64 \
    --batchsize_limit 64 \
    --full_parameter \
    --max_seq_length 128 \
    --device 0 \
    --logging "wandb+csv" \
    --alignment_diag \
    --alignment_diag_interval 100 \
    --alignment_fd_radius 1e-3 \
    --alignment_kappa_delta 1e-12 \
    --init_seed 42 \
    --epochs 1000 \
    --eval_every 1 \
    --lora_rank 32 \
    --ema_gamma 0.9 \
    --inject_method intermit \
    --fine_sample_ratio 0.05 \
    --tau_scheme adaptive \
    --tau_min 0.3 \
    --tau_max 0.8 \
    --tau_warmup_epochs 30 \
    --vcycle_coarse_epochs 60 \
    --vcycle_fine_epochs 60 \
    --phase3_lr 5e-7 \
    --checkpoint_interval 1000 \
    --early_stopping --patience 1000 \
    --curvature_multigrid --curvature_mg_beta 0.3 \
    --basis_method reuse \
    --no_coarse_isotropic \
    --checkpoint_dir results_demo/facebook_opt-2.7b/cola/ckpt_intermit_fsr5e2_C60F60_rank32_clr5e7_flr5e7_DS \
    --no_checkpoint \
    --deep_search \
    2>&1 | tee "$LOG_FILE"

echo "Done. Log: $LOG_FILE"
