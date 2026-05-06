"""Plot per-iteration norm-ratio diagnostics from norm_diag.json files.

Usage examples:
    # Plot one run, write PNG to paper/figures/
    python plot_norm_diagnostics.py llm-baselines/exps/fw_base_lionmuon_k2/norm_diag.json \\
        --label "FineWeb / GPT-2, LionMuon P=2" --out paper/figures/norm_diag_fineweb.png

    # Plot multiple runs side by side (one panel each)
    python plot_norm_diagnostics.py \\
        llm-baselines/exps/fw_base_lionmuon_k2/norm_diag.json:FineWeb \\
        llm-baselines/exps/wt_base_lionmuon_k1/norm_diag.json:WikiText-103 \\
        --out paper/figures/norm_diag.png
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt


SERIES = [
    ('grad_nuc_over_1',     r'$\|G\|_{\mathrm{nuc}}\,/\,\|G\|_{1}$',                'grad nuc/1'),
    ('param_2_over_inf',    r'$\|W\|_{2}\,/\,\|W\|_{\infty}$',                       'param 2/inf'),
    ('update_2_over_inf',   r'$\|U\|_{2}\,/\,\|U\|_{\infty}$',                       'update 2/inf (post-LMO)'),
    ('errmom_F_over_nuc',   r'$\|G-M\|_{F}\,/\,\|G-M\|_{\mathrm{nuc}}$',             'momentum-residual F/nuc'),
    ('errmom_F_over_1',     r'$\|G-M\|_{F}\,/\,\|G-M\|_{1}$',                        'momentum-residual F/1'),
    ('smooth_nuc_over_2',   r'$\|\Delta G\|_{\mathrm{nuc}}/\|\Delta W\|_{2}$',       r'smoothness $\widehat L_2$'),
    ('smooth_1_over_inf',   r'$\|\Delta G\|_{1}/\|\Delta W\|_{\infty}$',             r'smoothness $\widehat L_\infty$'),
]


def _parse_run(arg: str) -> tuple[str, str]:
    if ':' in arg and not (len(arg) > 2 and arg[1] == ':'):  # not a Windows drive letter
        path, label = arg.rsplit(':', 1)
    else:
        path, label = arg, os.path.splitext(os.path.basename(os.path.dirname(arg)) or arg)[0]
    return path, label


def _load(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data['records'] if 'records' in data else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='norm_diag.json paths, optionally suffixed with :LABEL')
    ap.add_argument('--out', default='paper/figures/norm_diag.png', help='output PNG path')
    ap.add_argument('--smooth_window', type=int, default=5, help='median-smoothing window over records')
    args = ap.parse_args()

    runs = [_parse_run(a) for a in args.runs]
    n_runs = len(runs)
    n_panels = len(SERIES)

    fig, axes = plt.subplots(
        n_panels, n_runs,
        figsize=(5.0 * n_runs, 2.0 * n_panels),
        sharex='col', squeeze=False,
    )

    def _smooth(xs):
        w = args.smooth_window
        if w <= 1:
            return xs
        out = []
        for i in range(len(xs)):
            lo, hi = max(0, i - w // 2), min(len(xs), i + w // 2 + 1)
            window = [v for v in xs[lo:hi] if v == v]  # drop NaN
            out.append(sorted(window)[len(window) // 2] if window else float('nan'))
        return out

    for col, (path, label) in enumerate(runs):
        records = _load(path)
        if not records:
            print(f'No records in {path}', file=sys.stderr)
            continue
        steps = [r['step'] for r in records]
        for row, (key, ylabel, _) in enumerate(SERIES):
            ax = axes[row][col]
            ys = [r.get(key, float('nan')) for r in records]
            ys = _smooth(ys)
            ax.plot(steps, ys, lw=1.4)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(ylabel, fontsize=10)
            if row == 0:
                ax.set_title(label, fontsize=12)
            if row == n_panels - 1:
                ax.set_xlabel('Iteration', fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {args.out}')


if __name__ == '__main__':
    main()
