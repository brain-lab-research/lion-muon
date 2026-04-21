import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_points(exps_dir: Path, prefix: str):
    pat = re.compile(rf"^{re.escape(prefix)}_lr([^_]+)_slr([^_]+)$")
    points = []
    for d in sorted(exps_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        val_loss = data.get("val_loss", [])
        if not val_loss:
            continue
        lr_s, slr_s = m.group(1), m.group(2)
        points.append((lr_s, slr_s, min(val_loss)))
    return points


def sort_numeric_str(vals):
    return sorted(vals, key=lambda s: float(s))


def build_grid(points):
    lrs = sort_numeric_str({p[0] for p in points})
    slrs = sort_numeric_str({p[1] for p in points})
    lr_to_i = {v: i for i, v in enumerate(lrs)}
    slr_to_i = {v: i for i, v in enumerate(slrs)}

    z = np.full((len(slrs), len(lrs)), np.nan)
    for lr_s, slr_s, best in points:
        z[slr_to_i[slr_s], lr_to_i[lr_s]] = best
    return lrs, slrs, z


def plot_heatmap(exps_dir: Path, out_path: Path):
    prefixes = [
        "signmuon_fixed_k2", "signmuon_fixed_k5", "signmuon_fixed_k20", "signmuon_fixed_k100",
        "lionmuon_k2", "lionmuon_k5", "lionmuon_k20", "lionmuon_k100",
    ]
    labels = {
        "signmuon_fixed_k2": "SignMuon k=2",
        "signmuon_fixed_k5": "SignMuon k=5",
        "signmuon_fixed_k20": "SignMuon k=20",
        "signmuon_fixed_k100": "SignMuon k=100",
        "lionmuon_k2": "LionMuon k=2",
        "lionmuon_k5": "LionMuon k=5",
        "lionmuon_k20": "LionMuon k=20",
        "lionmuon_k100": "LionMuon k=100",
    }

    data = {}
    all_vals = []
    for prefix in prefixes:
        pts = load_points(exps_dir, prefix)
        if not pts:
            continue
        lrs, slrs, z = build_grid(pts)
        data[prefix] = (lrs, slrs, z)
        all_vals.extend([v for v in z.ravel() if not np.isnan(v)])

    if not data:
        raise SystemExit(f"No matching runs found in {exps_dir} for SignMuon/LiMuon k grids")

    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=False, constrained_layout=True)
    im_for_cbar = None
    for ax, prefix in zip(axes.ravel(), prefixes):
        if prefix not in data:
            ax.axis("off")
            continue
        lrs, slrs, z = data[prefix]
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        im_for_cbar = im
        ax.set_xticks(np.arange(len(lrs)))
        ax.set_xticklabels(lrs, rotation=35, ha="right", fontsize=15)
        ax.set_yticks(np.arange(len(slrs)))
        ax.set_yticklabels(slrs, fontsize=15)
        ax.set_xlabel("lr", fontsize=17)
        ax.set_ylabel("sign_lr", fontsize=17)
        ax.set_title(labels[prefix], fontsize=19)

    if im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, ax=axes.ravel().tolist(), pad=0.01, shrink=0.92)
        cbar.set_label("best val_loss", fontsize=17)
        cbar.ax.tick_params(labelsize=15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    exps_dir = (repo_root / "exps_tuning_llama").resolve()
    out_path = (repo_root / "results/tuning_heatmap.png").resolve()
    plot_heatmap(exps_dir, out_path)


if __name__ == "__main__":
    main()
