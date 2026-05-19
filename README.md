# LionMuon

Code accompanying the paper [LionMuon: Alternating Spectral and Sign Descent for Efficient Training (arXiv placeholder)](https://arxiv.org/abs/PLACEHOLDER).

This repository is a fork of [Andrei Semenov's `llm-baselines`](https://github.com/Niccolo-Ajroldi/llm-baselines), extended with **LionMuon** and **SignMuon** optimizers and the experiments reported in the paper.

## Algorithm

For a single 2D parameter $W \in \mathbb{R}^{m \times n}$, **LionMuon** uses one momentum buffer $M_t$, forms a Lion-style interpolated direction $\hat{G}_t$, and every $P$-th step replaces $\mathrm{sign}(\hat{G}_t)$ with a Newton-Schulz orthogonalization $\mathrm{NS}_{K_{\mathrm{NS}}}(\hat{G}_t)$:

**Inputs:** horizon $T$, period $P \in \{1, 2, \dots\} \cup \{\infty\}$, learning rates $\eta_M$ (Muon) and $\eta_L$ (Lion), betas $\beta_1, \beta_2 \in [0, 1)$, weight decay $\lambda \ge 0$, NS steps $K_{\mathrm{NS}}$, initial $W_0$ and $M_{-1} = 0$.

**For** $t = 0, 1, \dots, T - 1$:

1. $G_t = \nabla_W \mathcal{L}_t$ &nbsp; (stochastic gradient)
2. $\hat{G}_t = \beta_1 M_{t-1} + (1 - \beta_1)\, G_t$ &nbsp; (Lion interpolation)
3. **If** $t \bmod P = 0$: &nbsp; $W_{t+1} = W_t - \eta_M \bigl(\mathrm{NS}_{K_{\mathrm{NS}}}(\hat{G}_t) + \lambda W_t\bigr)$ &nbsp; (Muon step)
4. **Else**: &nbsp; $W_{t+1} = W_t - \eta_L \bigl(\mathrm{sign}(\hat{G}_t) + \lambda W_t\bigr)$ &nbsp; (Lion step)
5. $M_t = \beta_2 M_{t-1} + (1 - \beta_2)\, G_t$ &nbsp; (momentum update, every step)

**SignMuon** is the special case $\beta_1 = \beta_2$: the dual-EMA collapses to a single Signum-style buffer, but the alternation between Newton-Schulz and elementwise sign is preserved. Pure **Signum**, **Lion**, and **Muon** are recovered as further special cases of the same algorithm:

| Optimizer | Momentum | Period |
| --- | --- | --- |
| `Signum` (Bernstein et al., 2018) | $\beta_1 = \beta_2$ | $P = \infty$ |
| `Lion` (Chen et al., 2024) | $\beta_1 \ne \beta_2$ (dual-EMA) | $P = \infty$ |
| `Muon` (Jordan et al., 2024) | $\beta_1 = \beta_2$ | $P = 1$ |
| `SignMuon` **(this work)** | $\beta_1 = \beta_2$ | any $P$ |
| `LionMuon` **(this work)** | $\beta_1 \ne \beta_2$ (dual-EMA) | any $P$ |

1D parameters (biases, LayerNorm/RMSNorm gains) fall back to AdamW with a small fixed learning rate, following the standard Muon-hybrid convention. The optimizer state therefore matches Lion / Muon and is exactly half of AdamW.

Reference implementations live in [src/optim/lion_muon.py](src/optim/lion_muon.py) and [src/optim/sign_muon.py](src/optim/sign_muon.py); Newton-Schulz orthogonalization is in [src/optim/muon.py](src/optim/muon.py).

## Install

```bash
pip install -r requirements.txt
```

## Running experiments

A single training run is launched through `src/main.py`:

```bash
python ./src/main.py --config_format base --opt lion_muon --dataset fineweb --model base
```

The batched scripts under `scripts/` reproduce the paper's experiments. They write checkpoints and logs to `./exps/<experiment_name>/`.

```bash
# All baselines on FineWeb (GPT-base + LLaMA)
bash scripts/train_fineweb.sh 0

# A single dataset/model combination
bash scripts/train_baselines.sh 0 --dataset fineweb --model base --algo-set regular

# Scaling runs
bash scripts/train_fineweb_355m.sh 0
bash scripts/train_fineweb_720m.sh 0

# Hyperparameter tuning sweeps
bash scripts/train_tuning.sh 0
bash scripts/train_tuning_720m.sh 0
bash scripts/train_tuning_llama.sh 0
```

Shared hyperparameters (architecture, batch size, schedule, weight decay) live in [scripts/common_config.sh](scripts/common_config.sh). Per-experiment learning rates are at the top of each `train_*.sh` script.

Multi-GPU runs use `torchrun`:

```bash
torchrun --nproc_per_node=4 ./src/main.py --config_format base --distributed_backend nccl \
    --opt lion_muon --dataset fineweb --model base
```

### Key flags

| Flag | Meaning |
| --- | --- |
| `--opt {lion_muon, sign_muon}` | Select the alternating optimizer |
| `--lr` | AdamW backup LR $\eta_{\mathrm{AdamW}}$ for 1D parameters |
| `--muon_lr_factor` | Muon learning rate $\eta_M$ for 2D parameters |
| `--sign_lr` | Lion / sign learning rate $\eta_L$ for 2D parameters |
| `--muon_every_k K` | Period $P$ (`K=1`: pure Muon; `K=∞`: pure Lion/Signum) |
| `--beta1`, `--beta2` | $\beta_1, \beta_2$ (set equal $\Rightarrow$ SignMuon) |
| `--muon_ns_steps` | Newton-Schulz iterations $K_{\mathrm{NS}}$ |
| `--weight_decay` | Decoupled weight decay $\lambda$ |
| `--srank_alpha` | Adaptive Muon trigger via stable-rank ratio (overrides `--muon_every_k`) |

## Directory layout

```
llm-baselines/
  src/
    main.py              # entry point: picks dataset, model, optimizer, training loop
    config/base.py       # all CLI flags
    data/                # dataset loaders (FineWeb, SlimPajama, WikiText, ...)
    models/              # base GPT and LLaMA architectures
    optim/
      lion_muon.py       # LionMuon  (this paper)
      sign_muon.py       # SignMuon  (this paper)
      muon.py            # Muon baseline + Newton-Schulz primitive
      lion.py            # Lion baseline
      ...                # AdamW, SOAP, Shampoo, Sophia, AdEMAMix, MARS, ...
    distributed/         # DDP wrappers
  scripts/               # bash scripts to reproduce paper experiments
  exps/                  # output checkpoints + logs (created on run)
  requirements.txt
```

## Logging with WandB

Pass `--wandb --wandb_project <name>` to log to your account. Set `WANDB_API_KEY` in the environment for headless runs.

## License

See [LICENSE](LICENSE). The original `llm-baselines` license is preserved.
