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
    ('grad_1_over_nuc',     r'$\|G\|_{1}\,/\,\|G\|_{\mathrm{nuc}}$',                 'grad 1/nuc'),
    ('errmom_nuc_over_F',   r'$\|G-M\|_{\mathrm{nuc}}\,/\,\|G-M\|_{F}$',             'momentum-residual nuc/F'),
    ('errmom_1_over_F',     r'$\|G-M\|_{1}\,/\,\|G-M\|_{F}$',                        'momentum-residual 1/F'),
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
    records = data['records'] if 'records' in data else data
    
    # Add inverted ratios for the ones we flipped
    for rec in records:
        if 'grad_nuc_over_1' in rec and rec['grad_nuc_over_1'] == rec['grad_nuc_over_1']:  # not NaN
            rec['grad_1_over_nuc'] = 1.0 / rec['grad_nuc_over_1'] if rec['grad_nuc_over_1'] > 1e-12 else float('nan')
        if 'errmom_F_over_nuc' in rec and rec['errmom_F_over_nuc'] == rec['errmom_F_over_nuc']:
            rec['errmom_nuc_over_F'] = 1.0 / rec['errmom_F_over_nuc'] if rec['errmom_F_over_nuc'] > 1e-12 else float('nan')
        if 'errmom_F_over_1' in rec and rec['errmom_F_over_1'] == rec['errmom_F_over_1']:  # not NaN
            rec['errmom_1_over_F'] = 1.0 / rec['errmom_F_over_1'] if rec['errmom_F_over_1'] > 1e-12 else float('nan')
    
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='norm_diag.json paths, optionally suffixed with :LABEL')
    ap.add_argument('--out', default='paper/figures/norm_diag.png', help='output PNG path')
    ap.add_argument('--smooth_window', type=int, default=5, help='median-smoothing window over records')
    args = ap.parse_args()

    runs = [_parse_run(a) for a in args.runs]
    n_runs = len(runs)
    
    # 5 panels: 3 on top, 2 on bottom spanning full width
    fig = plt.figure(figsize=(12.0, 4.0))
    gs = fig.add_gridspec(2, 6)
    
    axes_flat = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6])
    ]
    colors = plt.cm.tab10(range(n_runs))

    def _smooth(xs):
        w = args.smooth_window
        if w <= 1:
            return xs
        out = []
        for i in range(len(xs)):
            lo, hi = max(0, i - w // 2), min(len(xs), i + w // 2 + 1)
            window = [v for v in xs[lo:hi] if v == v]
            out.append(sorted(window)[len(window) // 2] if window else float('nan'))
        return out

    for row, (key, ylabel, _) in enumerate(SERIES):
        ax = axes_flat[row]
        for col, (path, label) in enumerate(runs):
            records = _load(path)
            if not records:
                print(f'No records in {path}', file=sys.stderr)
                continue
            steps = [r['step'] for r in records]
            ys = [r.get(key, float('nan')) for r in records]
            ys = _smooth(ys)
            
            # Calculate mean of last 1000 iterations
            valid_ys = [y for y in ys if y == y]  # drop NaN
            if valid_ys:
                mean_val = sum(valid_ys[-1000:]) / len(valid_ys[-1000:])
                label_with_mean = f'{label} (μ={mean_val:.3f})'
            else:
                label_with_mean = label
            
            ax.plot(steps, ys, lw=1.8, label=label_with_mean, color=colors[col])
        
        ax.grid(True, alpha=0.3)
        if key in ['grad_1_over_nuc', 'errmom_nuc_over_F', 'smooth_nuc_over_2', 'smooth_1_over_inf']:
            ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {args.out}')


if __name__ == '__main__':
    main()
