import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_points(exps_dir: Path, prefix: str, name_prefix: str = ""):
    pat_full = re.compile(rf"^{re.escape(name_prefix)}{re.escape(prefix)}_lr([^_]+)_slr([^_]+)$")
    pat_lr_only = re.compile(rf"^{re.escape(name_prefix)}{re.escape(prefix)}_lr([^_]+)$")
    points = []
    for d in sorted(exps_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pat_full.match(d.name)
        slr_s = None
        if m:
            lr_s, slr_s = m.group(1), m.group(2)
        else:
            m = pat_lr_only.match(d.name)
            if not m:
                continue
            lr_s = m.group(1)
            slr_s = "n/a"
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
        points.append((lr_s, slr_s, min(val_loss)))
    return points


def sort_numeric_str(vals):
    return sorted(vals, key=lambda s: float(s))


def build_grid(points):
    lrs = sort_numeric_str({p[0] for p in points})
    slr_vals = {p[1] for p in points}
    if slr_vals == {"n/a"}:
        slrs = ["n/a"]
    else:
        slrs = sort_numeric_str(slr_vals)
    lr_to_i = {v: i for i, v in enumerate(lrs)}
    slr_to_i = {v: i for i, v in enumerate(slrs)}

    z = np.full((len(slrs), len(lrs)), np.nan)
    for lr_s, slr_s, best in points:
        z[slr_to_i[slr_s], lr_to_i[lr_s]] = best
    return lrs, slrs, z


def plot_heatmap(exps_dir: Path, out_path: Path, name_prefix: str = ""):
    prefixes = [
        "signmuon_fixed_k1", "signmuon_fixed_k2", "signmuon_fixed_k5",
        "lionmuon_k1", "lionmuon_k2", "lionmuon_k5",
    ]
    labels = {
        "signmuon_fixed_k1": "Muon",
        "signmuon_fixed_k2": "SignMuon k=2",
        "signmuon_fixed_k5": "SignMuon k=5",
        "lionmuon_k1": "LionMuon k=1",
        "lionmuon_k2": "LionMuon k=2",
        "lionmuon_k5": "LionMuon k=5",
    }

    data = {}
    all_vals = []
    for prefix in prefixes:
        pts = load_points(exps_dir, prefix, name_prefix)
        if not pts:
            continue
        lrs, slrs, z = build_grid(pts)
        data[prefix] = (lrs, slrs, z)
        all_vals.extend([v for v in z.ravel() if not np.isnan(v)])

    if not data:
        raise SystemExit(f"No matching runs found in {exps_dir} for SignMuon/LionMuon k grids")

    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False, constrained_layout=True)
    im_for_cbar = None
    for ax, prefix in zip(axes.ravel(), prefixes):
        if prefix not in data:
            ax.axis("off")
            continue
        lrs, slrs, z = data[prefix]
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        im_for_cbar = im
        ax.set_xticks(np.arange(len(lrs)))
        ax.set_xticklabels(lrs, rotation=35, ha="right", fontsize=11)
        ax.set_yticks(np.arange(len(slrs)))
        ax.set_yticklabels(slrs, fontsize=11)
        ax.set_xlabel("lr", fontsize=13)
        ax.set_ylabel("sign_lr", fontsize=13)
        ax.set_title(labels[prefix], fontsize=14)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if not np.isnan(z[i, j]):
                    ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color="black")
    # turn off any unused axes
    for ax, prefix in zip(axes.ravel(), prefixes + [None] * (axes.size - len(prefixes))):
        if prefix is None:
            ax.axis("off")

    if im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, ax=axes.ravel().tolist(), pad=0.01, shrink=0.92)
        cbar.set_label("best val_loss", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    exps_dir = (repo_root / "exps_tuning_720m").resolve()
    out_path = (repo_root / "results/tuning_heatmap.png").resolve()
    plot_heatmap(exps_dir, out_path, name_prefix="fw_base_720m_tune_")


if __name__ == "__main__":
    main()
