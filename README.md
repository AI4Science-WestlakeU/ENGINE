## ENGINE: Endogenous Variational MultiScale Optimization for Zeroth-Order LLM Fine-Tuning

This repository provides the official implementation of **ENGINE**, a zeroth-order LLM fine-tuning method that brings the **Variational MultiScale (VMS)** view from computational mechanics into the LoRA–LOREN setting. ENGINE achieves **higher accuracy** and **faster convergence** than existing ZO methods.



**Authors:** Zhuoli Ouyang, Changxi Chi, Siyuan Li, and Tailin Wu.  
**Accepted at NeurIPS 2026.**  
**Corresponding author:** Tailin Wu (`wutailin@westlake.edu.cn`).

[Paper](https://arxiv.org/abs/2611.03293) · [Project webpage](https://dominator-index.github.io/ENGINE-website/)

## Installation

Use a Python environment with a CUDA-compatible PyTorch installation for LLM experiments.

```bash
git clone https://github.com/AI4Science-WestlakeU/ENGINE.git
cd ENGINE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiment scripts read `CUDA_VISIBLE_DEVICES`, `HF_HOME`, and the Hugging Face cache variables from the environment. Their defaults use GPU 0 and the standard user cache. Downloads are enabled by default and WandB runs offline. Override these variables for an existing offline cache or an online WandB account. For gated models, obtain access and supply your Hugging Face credentials through `HF_TOKEN`.

## Overview

ENGINE treats the LoRA subspace as the resolved coarse scale and the full parameter space as the unresolved fine scale: the **VMS-EO correction** reinjects the fine-grid curvature that a LoRA-only restriction would discard, and a V-cycle schedule with bidirectional state transfer keeps the endogenous Hessian fresh between coarse phases. 




## Models

| Family | Model                                   |
| :----- | :-------------------------------------- |
| GPT    | `gpt2-xl`                               |
| OPT    | `facebook/opt-2.7b`, `facebook/opt-13b` |
| LLaMA  | `meta-llama/Meta-Llama-3-8B`            |

## Tasks

**GLUE:** `mnli`, `qnli`, `sst2`, `cola`  
**SuperGLUE:** `rte`, `boolq`, `wic`, `cb`



## Algorithms
| Algorithm | Description                                         | Reference                                                    |
| :-------- | :-------------------------------------------------- | :----------------------------------------------------------- |
| `MeZO`    | Baseline memory-efficient ZO                        | [Malladi et al., 2023, *Fine-Tuning Language Models with Just Forward Passes*](https://proceedings.neurips.cc/paper_files/paper/2023/file/a627810151be4d13f907ac898ff7e948-Paper-Conference.pdf) |
| `ZOAdam`  | MeZO with Adam's adaptive update                    |                                                              |
| `ZOSVRG`  | MeZO with stochastic variance-reduced gradients     | [Gautam et al., 2024, *Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models*](https://openreview.net/pdf?id=yc758jO4i5) |
| `LOZO`    | Low-rank ZO gradient                                | [Chen et al., 2025, *Enhancing Zeroth-Order Fine-Tuning for Language Models with Low-Rank Structures*](https://openreview.net/pdf?id=9BiVepgmWW) |
| `HiZOO`   | Hessian informed ZO                                 | [Zhao et al., 2025, *Second-Order Fine-Tuning without Pain for LLMs: A Hessian Informed Zeroth-Order Optimizer*](https://openreview.net/forum?id=bEqI61iBue) |
| `LOREN`   | Low-rank curvature-aware ZO with variance reduction | Seung et al., 2026, *Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning* |
| `ENGINE`(ours)   | ZO with VMS-EO correction | Ouyang et al., NeurIPS 2026 |



## How to Run

Run the supplied GPT-2-XL/QNLI configuration from the repository root:

```bash
bash scripts/ENGINE/gpt2xl_qnli_intermit_fsr5e2_C60F60_rank16_clr1e5_flr1e5_DS.sh
```

For a configurable command, set these shell variables first. These values illustrate the GPT-2-XL/QNLI configuration:

```bash
epochs=1000
model=gpt2-xl
task=qnli
fine_lr=1e-5
coarse_lr=1e-5
num_queries=6
bsz=64
bsz_lim=64
lora_rank=16
fsr=0.05
c_ep=60
f_ep=60
```

Fine-tune with ENGINE:

``` shell
python main.py \
    --epochs $epochs \
    --algorithm ENGINE \
    --model_name $model \
    --task $task \
    --lr $fine_lr \
    --weight_decay 0.0 \
    --num_queries $num_queries \
    --perturbation_scale 1e-3 \
    --damping 0.1 \
    --lr_cov 1e-3 \
    --samplesize 512 \
    --samplesize_validation 256 \
    --batchsize $bsz \
    --batchsize_limit $bsz_lim \
    --full_parameter \
    --max_seq_length 128 \
    --device 0 \
    --logging "wandb+csv" \
    --init_seed 42 \
    --eval_every 1 \
    --lora_rank $lora_rank \
    --ema_gamma 0.9 \
    --inject_method intermit \
    --fine_sample_ratio $fsr \
    --tau_scheme adaptive \
    --tau_min 0.3 \
    --tau_max 0.8 \
    --tau_warmup_epochs 30 \
    --vcycle_coarse_epochs $c_ep \
    --vcycle_fine_epochs $f_ep \
    --phase3_lr $coarse_lr \
    --checkpoint_interval 1000 \
    --early_stopping --patience 1000 \
    --curvature_multigrid --curvature_mg_beta 0.3 \
    --basis_method reuse \
    --no_coarse_isotropic \
    --checkpoint_dir results_demo \
    --no_checkpoint \
    --deep_search 

```



To run a baseline, set `algorithm`, `lr`, and `trial` along with the model, task, and training variables above. For example, `algorithm=LOREN`, `lr=1e-5`, and `trial=0`. The baseline implementation builds on [LOREN](https://github.com/hseung88/loren):

``` shell
python main.py\
    --epochs $epochs \
    --model_name $model \
    --task $task \
    --algorithm $algorithm \
    --lr $lr \
    --weight_decay 0.0 \
    --num_queries $num_queries \
    --perturbation_scale 1e-3  \
    --lr_mezosvrg_mb 1e-6 --q 2 --anneal 5 \
    --eps 1e-8 \
    --rank 2 --step_interval 50 \
    --hessian_smooth_type constant1e-8 \
    --lr_cov 1e-3 --damping 0.1 \
    --samplesize 512 --samplesize_validation 256 \
    --batchsize $bsz \
    --batchsize_limit $bsz_lim \
    --full_parameter \
    --max_seq_length 128 \
    --device 0 --save_every 10000000 --logging csv \
    --init_seed 42 --trial $trial \
    --early_stopping --patience 1000
```



## Evaluation metrics

The ENGINE trainer selects MCC automatically for CoLA and logs the raw coefficient in `val_mcc` on the scale [-1, 1]. Multiply it by 100 to compare with the paper's MCC scores. Other tasks use accuracy in percent. The separate baseline trainer currently logs accuracy. See [CoLA-MCC.md](CoLA-MCC.md) for the exact evaluation behavior and validation-subset settings.

## Tests

The existing unit tests use toy models and do not download LLM checkpoints:

```bash
python -m unittest discover -s tests -v
```

## Citation

If you find ENGINE useful for your research, please cite our paper.

```bibtex
@misc{ouyang2026engineendogenousvariationalmultiscale,
      title={ENGINE: Endogenous Variational MultiScale Optimization for Zeroth-Order LLM Fine-Tuning},
      author={Zhuoli Ouyang and Changxi Chi and Siyuan Li and Tailin Wu},
      year={2026},
      eprint={2611.03293},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.03293},
}
```
