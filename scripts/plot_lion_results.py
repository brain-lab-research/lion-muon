"""Plot Lion / LionMuon tuning results vs AdamW / Muon / SignMuon / Signum baselines."""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb

api = wandb.Api()


def fetch_project(project):
    """Return {run_name: {'losses': [...], 'cfg': {...}}} for a WandB project."""
    runs = {}
    for r in api.runs(project):
        hist = r.history(keys=["val/loss"], x_axis="_step", pandas=False)
        losses = [row["val/loss"] for row in hist if "val/loss" in row]
        if len(losses) >= 20:
            runs[r.name] = {"losses": losses, "cfg": r.config}
    return runs


def best_run(runs, prefix):
    """Return (name, losses) for the run with given prefix and minimum val_loss."""
    candidates = {k: v for k, v in runs.items() if k.startswith(prefix)}
    if not candidates:
        return None, []
    best = min(candidates, key=lambda k: min(candidates[k]["losses"]))
    return best, candidates[best]["losses"]


def compute_flops(n_layer, n_embd, seq_len, batch_size, iterations, opt_type, K=1, ns_steps=6):
    n_params = n_layer * (4 * n_embd**2 + 8 * n_embd**2)
    model_flops = 6 * n_params * seq_len * batch_size
    d = n_embd
    ns_flops = 0
    for m, n in [(d, 3*d), (d, d), (d, 4*d), (4*d, d)]:
        dmin, dmax = min(m, n), max(m, n)
        ns_flops += ns_steps * 2 * dmin**2 * (3*dmax + dmin)
    ns_flops *= n_layer
    if opt_type == "adamw":
        return iterations * model_flops
    elif opt_type == "muon":
        return iterations * (model_flops + ns_flops)
    elif opt_type in ("sign_muon", "lion_muon"):
        return iterations * (model_flops + ns_flops / K)
    else:
        return iterations * model_flops


# ── consistent style ─────────────────────────────────────────────────────────
COLORS = {
    "AdamW":        "#4c72b0",
    "Signum":       "#dd8452",
    "Lion":         "#8172b2",
    "Muon":         "#55a868",
    "SignMuon K=5": "#c44e52",
    "LionMuon K=5": "#e377c2",
}
ORDER = ["AdamW", "Signum", "Lion", "Muon", "SignMuon K=5", "LionMuon K=5"]


def main():
    print("Fetching sign-muon-tuning-v2 …")
    all_runs = fetch_project("sign-muon-tuning-v2")

    methods = {
        "AdamW":        best_run(all_runs, "adam_lr"),
        "Signum":       best_run(all_runs, "signum_lr"),
        "Muon":         best_run(all_runs, "muon_lr"),
        "SignMuon K=5": best_run(all_runs, "signmuon_lr"),
        "Lion":         best_run(all_runs, "lion_lr"),
        "LionMuon K=5": best_run(all_runs, "lionmuon_lr"),
    }

    print("\n── Best val_loss per method ──────────────────")
    for label in ORDER:
        name, losses = methods[label]
        if losses:
            print(f"  {label:<16s}  {name:<40s}  best={min(losses):.4f}  ppl={math.exp(min(losses)):.1f}")
        else:
            print(f"  {label:<16s}  NO DATA")

    # ── LionMuon hyperparameter grid ─────────────────────────────────────────
    base_lrs = ["1e-3", "2e-3", "5e-3"]
    lion_lrs = ["1e-5", "5e-5", "1e-4", "5e-4"]
    grid = np.full((len(base_lrs), len(lion_lrs)), np.nan)
    for i, blr in enumerate(base_lrs):
        for j, llr in enumerate(lion_lrs):
            key = f"lionmuon_lr{blr}_llr{llr}"
            if key in all_runs:
                grid[i, j] = min(all_runs[key]["losses"])

    # ── Figure: 3 panels ─────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.35)

    iters_per_eval = 100

    # Panel 1 — val_loss curves, clipped to [4.7, 6.2] to drop early divergence
    for label in ORDER:
        name, losses = methods[label]
        if not losses:
            continue
        xs = [(i + 1) * iters_per_eval for i in range(len(losses))]
        vl = min(losses)
        ax1.plot(xs, losses, label=f"{label}  ({vl:.4f})",
                 linewidth=2.0, color=COLORS[label])
    ax1.set_ylim(4.7, 6.2)
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Val Loss", fontsize=12)
    ax1.set_title("Val Loss — best LR per method", fontsize=13)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2 — FLOPs vs perplexity scatter
    n_layer, n_embd, seq_len, batch_size, iters = 6, 384, 256, 32, 3000
    flop_map = {
        "AdamW":        compute_flops(n_layer, n_embd, seq_len, batch_size, iters, "adamw"),
        "Signum":       compute_flops(n_layer, n_embd, seq_len, batch_size, iters, "adamw"),
        "Lion":         compute_flops(n_layer, n_embd, seq_len, batch_size, iters, "adamw"),
        "Muon":         compute_flops(n_layer, n_embd, seq_len, batch_size, iters, "muon"),
        "SignMuon K=5": compute_flops(n_layer, n_embd, seq_len, batch_size, iters, "sign_muon", K=5),
        "LionMuon K=5": compute_flops(n_layer, n_embd, seq_len, batch_size, iters, "lion_muon", K=5),
    }
    for label in ORDER:
        name, losses = methods[label]
        if not losses:
            continue
        vl = min(losses)
        ppl = math.exp(vl)
        ax2.scatter(flop_map[label], ppl, s=180, color=COLORS[label],
                    label=f"{label} ({vl:.4f})", zorder=5)
        ax2.annotate(label, (flop_map[label], ppl),
                     textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    ax2.set_xlabel("Total FLOPs", fontsize=12)
    ax2.set_ylabel("Perplexity", fontsize=12)
    ax2.set_title("FLOPs vs Perplexity", fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Panel 3 — LionMuon hyperparameter grid heatmap
    vmin = np.nanmin(grid)
    vmax = min(np.nanmax(grid), 6.5)
    im = ax3.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)
    ax3.set_xticks(range(len(lion_lrs)))
    ax3.set_xticklabels([f"llr={x}" for x in lion_lrs], fontsize=9, rotation=30)
    ax3.set_yticks(range(len(base_lrs)))
    ax3.set_yticklabels([f"lr={x}" for x in base_lrs], fontsize=9)
    ax3.set_title("LionMuon K=5 — val_loss grid", fontsize=13)
    plt.colorbar(im, ax=ax3, label="Val Loss")
    for i in range(len(base_lrs)):
        for j in range(len(lion_lrs)):
            v = grid[i, j]
            if not np.isnan(v):
                txt_color = "white" if v > 5.8 else "black"
                ax3.text(j, i, f"{v:.3f}", ha="center", va="center",
                         fontsize=9, color=txt_color, fontweight="bold")

    fig.suptitle("Tuning Results — 6L/384d, 3K iters, WSD, WD=0.1", fontsize=14, y=1.01)
    out = "/home/arman.bolatov/Desktop/llm-baselines/lion_tuning_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")

    print("\n── Summary ───────────────────────────────────")
    for label in ORDER:
        name, losses = methods[label]
        if losses:
            vl = min(losses)
            print(f"  {label:<16s}  val_loss={vl:.4f}  ppl={math.exp(vl):.1f}")


if __name__ == "__main__":
    main()
