#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunData:
    name: str
    label: str
    summary_path: str
    args: Dict
    val_loss: List[float]
    iterations_axis: List[int]
    best_val_loss: float
    opt: str
    muon_every_k: int
    muon_ns_steps: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot RMSspectral variants vs AdamW/Muon/Lion with FLOPs.")
    p.add_argument("--exps-dir", default="exps", help="Experiments root folder")
    p.add_argument("--dataset", default="fw", choices=["fw", "fineweb", "spj", "slimpajama"])
    p.add_argument("--model", default="base", choices=["base", "llama", "gpt"])
    p.add_argument("--out", default="results/fw_base_rmsspectral_vs_baselines.png", help="Output plot path")
    return p.parse_args()


def normalize_prefix(dataset_raw: str, model_raw: str) -> str:
    if dataset_raw in ("fw", "fineweb"):
        ds = "fw"
    else:
        ds = "spj"

    if model_raw in ("base", "gpt"):
        model = "base"
    else:
        model = "llama"

    return f"{ds}_{model}_"


def load_summary(summary_path: str) -> Optional[Dict]:
    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def discover_runs(exps_dir: str, prefix: str) -> List[RunData]:
    targets = [
        ("adamw", "AdamW"),
        ("muon", "Muon"),
        ("lion", "Lion"),
        ("rmsspectral_pre", "RMSspectral pre"),
        ("rmsspectral_pre_ema", "RMSspectral pre_ema"),
        ("rmsspectral_post_orth", "RMSspectral post_orth"),
        ("rmsspectral_split", "RMSspectral split"),
    ]

    runs: List[RunData] = []
    for short_name, label in targets:
        exp_name = prefix + short_name
        summary_path = os.path.join(exps_dir, exp_name, "summary.json")
        if not os.path.isfile(summary_path):
            continue

        data = load_summary(summary_path)
        if not data:
            continue

        val_loss = data.get("val_loss", [])
        args = data.get("args", {}) if isinstance(data.get("args", {}), dict) else {}
        if not isinstance(val_loss, list) or len(val_loss) == 0:
            continue

        eval_interval = int(args.get("eval_interval", 500))
        iterations_axis = list(range(eval_interval, eval_interval * (len(val_loss) + 1), eval_interval))

        runs.append(
            RunData(
                name=exp_name,
                label=label,
                summary_path=summary_path,
                args=args,
                val_loss=[float(x) for x in val_loss],
                iterations_axis=iterations_axis,
                best_val_loss=float(min(val_loss)),
                opt=str(args.get("opt", "")),
                muon_every_k=int(args.get("muon_every_k", 1) or 1),
                muon_ns_steps=int(args.get("muon_ns_steps", 5) or 5),
            )
        )
    return runs


def estimate_params(n_layer: int, n_embd: int, seq_len: int, vocab_size: int = 50304) -> int:
    # Same structure as the repository plotting code: attn + mlp + embeddings.
    params_per_layer = 12 * (n_embd ** 2)
    embed_params = vocab_size * n_embd + seq_len * n_embd
    return n_layer * params_per_layer + embed_params


def ns_flops_per_iter(n_layer: int, n_embd: int, ns_steps: int) -> float:
    # Approximate NS cost on 4 major 2D blocks per layer.
    d = n_embd
    total = 0.0
    for m, n in ((d, 3 * d), (d, d), (d, 4 * d), (4 * d, d)):
        dmin, dmax = min(m, n), max(m, n)
        total += ns_steps * 2.0 * (dmin ** 2) * (2 * dmax + dmin)
    return n_layer * total


def flops_per_iter(run: RunData) -> float:
    a = run.args
    n_layer = int(a.get("n_layer", 12))
    n_embd = int(a.get("n_embd", 768))
    seq_len = int(a.get("sequence_length", 512))
    batch_size = int(a.get("batch_size", 32))
    acc_steps = int(a.get("acc_steps", 1))

    # 6 * params * tokens is a common forward+backward transformer training approximation.
    params = estimate_params(n_layer, n_embd, seq_len)
    tokens_per_iter = seq_len * batch_size * acc_steps
    model_cost = 6.0 * params * tokens_per_iter

    opt = run.opt
    if opt in ("adamw",):
        return model_cost

    if opt in ("sign_muon", "lion_muon"):
        ns_cost = ns_flops_per_iter(n_layer, n_embd, run.muon_ns_steps)
        k = max(1, run.muon_every_k)
        return model_cost + ns_cost / float(k)

    if opt in ("rmsspectral",):
        ns_cost = ns_flops_per_iter(n_layer, n_embd, run.muon_ns_steps)
        return model_cost + ns_cost

    return model_cost


def human_flops(x: float) -> str:
    if x >= 1e18:
        return f"{x/1e18:.2f} EF"
    if x >= 1e15:
        return f"{x/1e15:.2f} PF"
    if x >= 1e12:
        return f"{x/1e12:.2f} TF"
    if x >= 1e9:
        return f"{x/1e9:.2f} GF"
    return f"{x:.2e}"


def make_plot(runs: List[RunData], out_path: str, title_prefix: str) -> None:
    styles = {
        "AdamW": {"color": "#555555", "ls": "-", "marker": "s"},
        "Muon": {"color": "#8b0000", "ls": "-", "marker": "^"},
        "Lion": {"color": "#0f4c81", "ls": "-", "marker": "o"},
        "RMSspectral pre": {"color": "#2a9d8f", "ls": "-", "marker": "D"},
        "RMSspectral pre_ema": {"color": "#1d7f73", "ls": "--", "marker": "D"},
        "RMSspectral post_orth": {"color": "#e76f51", "ls": "-.", "marker": "P"},
        "RMSspectral split": {"color": "#a44a3f", "ls": ":", "marker": "P"},
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    ranking: List[Tuple[str, float, float]] = []
    for r in runs:
        st = styles.get(r.label, {"color": "#333333", "ls": "-", "marker": "o"})
        fpi = flops_per_iter(r)
        total = (r.iterations_axis[-1] * fpi) if r.iterations_axis else 0.0
        ranking.append((r.label, r.best_val_loss, total))

        ax.plot(r.iterations_axis, r.val_loss, label=r.label, color=st["color"], ls=st["ls"], lw=2.0)

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Validation loss", fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(3.5, 6)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=11, frameon=True)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved plot: {out_path}")
    print("\nBest val_loss ranking:")
    for i, (label, best, total) in enumerate(sorted(ranking, key=lambda t: t[1]), 1):
        print(f"{i:2d}. {label:24s} best_val_loss={best:.6f}  total_flops={human_flops(total)}")



def main() -> None:
    args = parse_args()
    prefix = normalize_prefix(args.dataset, args.model)
    runs = discover_runs(args.exps_dir, prefix)

    if not runs:
        raise SystemExit(f"No matching summary.json files found for prefix '{prefix}' in {args.exps_dir}")

    # Keep a stable, readable order in plot/legend.
    order = [
        "AdamW",
        "Muon",
        "Lion",
        "RMSspectral pre",
        "RMSspectral pre_ema",
        "RMSspectral post_orth",
        "RMSspectral split",
    ]
    runs.sort(key=lambda r: order.index(r.label) if r.label in order else 999)

    title_prefix = f"{prefix[:-1]}"
    make_plot(runs, args.out, title_prefix)


if __name__ == "__main__":
    main()
