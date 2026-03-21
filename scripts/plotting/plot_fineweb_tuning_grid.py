#!/usr/bin/env python3
"""
Generate a heatmap showing validation loss across different RMSspectral variants
and learning rates for FineWeb + GPT experiments.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Plot FineWeb RMSspectral LR tuning grid")
    p.add_argument("--exps-dir", default="exps_tuning_rmsspectral_fw_gpt")
    p.add_argument("--output", default="results/fineweb_tuning_grid.png")
    return p.parse_args()


def load_variant_runs(exps_dir: Path):
    """Load validation loss data for each variant."""
    # Variants to include
    variants = {
        "rmsspectral_pre": "Pre",
        "rmsspectral_pre_ema": "PreEMA",
        "rmsspectral_post_orth": "PostOrth",
        "rmsspectral_split": "Split",
    }
    
    data = {}
    for prefix, label in variants.items():
        # Find all runs for this variant
        lrs = {}
        for d in sorted(exps_dir.iterdir()):
            if not d.is_dir():
                continue
            if not d.name.startswith(prefix + "_lr"):
                continue
            
            # Extract LR from directory name
            match = re.search(rf"{re.escape(prefix)}_lr([\de\-\.]+)$", d.name)
            if not match:
                continue
            
            lr_str = match.group(1)
            summary_path = d / "summary.json"
            if not summary_path.exists():
                continue
            
            try:
                with open(summary_path) as f:
                    run_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            
            val_loss = run_data.get("val_loss", [])
            if val_loss:
                lrs[lr_str] = min(val_loss)
        
        if lrs:
            data[label] = lrs
    
    return data


def sort_numeric_str(vals):
    """Sort numeric strings by their float value."""
    return sorted(vals, key=lambda s: float(s))


def plot_variants_heatmap(data: dict, out_path: Path):
    """Create a heatmap showing LR tuning across variants."""
    if not data:
        raise ValueError("No data found")
    
    # Get all unique LRs
    all_lrs = set()
    for lrs_dict in data.values():
        all_lrs.update(lrs_dict.keys())
    
    lrs = sort_numeric_str(all_lrs)
    variants = list(data.keys())
    
    # Build grid: rows = variants, columns = LRs
    z = np.full((len(variants), len(lrs)), np.nan)
    for i, variant in enumerate(variants):
        for j, lr in enumerate(lrs):
            if lr in data[variant]:
                z[i, j] = data[variant][lr]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="RdYlGn_r")
    
    ax.set_xticks(np.arange(len(lrs)))
    ax.set_xticklabels(lrs, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels(variants, fontsize=11)
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Variant", fontsize=12)
    
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Best Validation Loss", fontsize=11)
    
    # Add text annotations
    for i in range(len(variants)):
        for j in range(len(lrs)):
            if not np.isnan(z[i, j]):
                text = ax.text(j, i, f"{z[i, j]:.2f}",
                              ha="center", va="center", color="black", fontsize=8)
    
    fig.tight_layout()
    fig.subplots_adjust(right=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    exps_dir = (repo_root / args.exps_dir).resolve()
    out_path = (repo_root / args.output).resolve()
    
    if not exps_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {exps_dir}")
    
    data = load_variant_runs(exps_dir)
    if not data:
        raise ValueError(f"No matching runs found in {exps_dir}")
    
    plot_variants_heatmap(data, out_path)


if __name__ == "__main__":
    main()
