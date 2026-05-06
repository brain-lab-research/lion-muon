# Per-iteration norm-ratio diagnostics

This pipeline records 7 norm ratios at each optimizer step and plots their
trajectories across training. It is wired into `LionMuon` (and used as-is for
SignMuon, since SignMuon is the `β1=β2` special case of LionMuon).

The 7 ratios per iteration:

| key                  | meaning                                                                |
|----------------------|------------------------------------------------------------------------|
| `grad_nuc_over_1`    | $\|G\|_{\mathrm{nuc}} / \|G\|_{1}$ — gradient nuclear vs.\ $\ell_1$    |
| `param_2_over_inf`   | $\|W\|_{2}    / \|W\|_{\infty}$ — parameter spectral vs.\ $\ell_\infty$|
| `update_2_over_inf`  | $\|U\|_{2}    / \|U\|_{\infty}$ — post-LMO update (sign or NS)         |
| `errmom_F_over_nuc`  | $\|G-M\|_{F} / \|G-M\|_{\mathrm{nuc}}$ — momentum-residual            |
| `errmom_F_over_1`    | $\|G-M\|_{F} / \|G-M\|_{1}$ — momentum-residual                       |
| `smooth_nuc_over_2`  | $\|\nabla f(W_{t+1})-\nabla f(W_{t})\|_{\mathrm{nuc}} / \|W_{t+1}-W_{t}\|_{2}$ — smoothness proxy in $L_2$ pairing |
| `smooth_1_over_inf`  | $\|\nabla f(W_{t+1})-\nabla f(W_{t})\|_{1}   / \|W_{t+1}-W_{t}\|_{\infty}$ — smoothness proxy in $L_\infty$ pairing |

Each entry stores the **median across all 2D matrices** in the model at that
iteration. Smoothness is a noisy single-batch estimate (different mini-batches
at $t$ and $t{+}1$); plot smoothing in the plotter mitigates this.

---

## How to record

Add `--norm_diag` to the training command. The diagnostic JSON lands at
`<exp_dir>/norm_diag.json` (atomically rewritten every 10 records ≈ every 500
steps with default settings).

Optional flags:
- `--norm_diag_every_k 50` — record every 50 optimizer steps (default).
- `--norm_diag_path /custom/path.json` — override location.

### Headline FineWeb run (LionMuon $P=2$)

```bash
conda activate optim
cd llm-baselines/src
python main.py \
  --opt lion_muon --dataset fineweb --model base \
  --muon_every_k 2 --muon_lr_factor 0.001 --sign_lr 5e-5 --lr 0.001 \
  --beta1 0.9 --beta2 0.99 \
  --batch_size 32 --sequence_length 512 \
  --iterations 64000 --warmup_steps 3000 \
  --muon_ns_steps 5 --weight_decay 0.1 --grad_clip 0.5 \
  --exp_name fw_base_lionmuon_k2_normdiag \
  --norm_diag --norm_diag_every_k 50
```

### Headline WikiText-103 run (LionMuon $P=1$)

```bash
python main.py \
  --opt lion_muon --dataset wikitext --model base \
  --muon_every_k 1 --muon_lr_factor 7e-4 --lr 0.001 \
  --beta1 0.9 --beta2 0.99 \
  --batch_size 32 --sequence_length 512 \
  --iterations 64000 --warmup_steps 3000 \
  --muon_ns_steps 5 --weight_decay 0.1 --grad_clip 0.5 \
  --exp_name wt_base_lionmuon_k1_normdiag \
  --norm_diag --norm_diag_every_k 50
```

(64,000 steps × 50-step interval = 1,280 records per run, well under the file-size limit.)

---

## How to plot

```bash
cd <project root>
python llm-baselines/scripts/plotting/plot_norm_diagnostics.py \
  llm-baselines/exps/fw_base_lionmuon_k2_normdiag/norm_diag.json:FineWeb \
  llm-baselines/exps/wt_base_lionmuon_k1_normdiag/norm_diag.json:WikiText-103 \
  --out paper/figures/norm_diag.png \
  --smooth_window 5
```

The output PNG has 7 rows (one per ratio) × N columns (one per run).
`--smooth_window K` applies a K-record running median per series; set to 1 to disable.

---

## Cost notes

- Diagnostic recording adds an extra forward-free SVD per 2D matrix per logged
  step. At 124M with ~50 matrices and `every_k=50`, that's ~50 SVDs every 50
  steps — negligible against forward+backward.
- Cloning `(p, g, m, u)` per logged matrix on logged steps only — also small.
- Disabling: omit `--norm_diag` (zero overhead path; hook is gated by
  `norm_diag_active`).
