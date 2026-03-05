"""Plot results for optimizer comparison experiments.

Usage:
    python scripts/plotting/plot_results.py fineweb
    python scripts/plotting/plot_results.py slimpajama
    python scripts/plotting/plot_results.py all
"""

import re
import os
import sys
import json
import math
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = '/home/arman/llm-baselines'
WANDB_DIR = os.path.join(BASE_DIR, 'wandb')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

DATASETS = {
    'fineweb': {
        'wandb_project': 'sign-muon-main',
        'prefix': '',
        'title': 'FineWeb',
    },
    'slimpajama': {
        'wandb_project': 'sign-muon-slimpajama',
        'prefix': 'spj_',
        'title': 'SlimPajama',
    },
}

# Display names: experiment key -> label
# Conceptual families:
#   SignMuon family (SGD momentum): Muon(K=1) -> SignMuon K=2,5,20,100 -> Signum(K=inf)
#   LiMuon family (dual-EMA):      LiMuon K=1 -> K=2,5,20,100 -> Lion(K=inf)
#   AdamW: standalone baseline
NAMES = {
    'adamw':          'AdamW',
    'muon':           'Muon (K=1)',
    'signmuon_k2':    'SignMuon K=2',
    'signmuon_k5':    'SignMuon K=5',
    'signmuon_k20':   'SignMuon K=20',
    'signmuon_k100':  'SignMuon K=100',
    'signum':         'Signum (K=\u221e)',
    'lionmuon_k1':    'LiMuon K=1',
    'lionmuon_k2':    'LiMuon K=2',
    'lionmuon_k5':    'LiMuon K=5',
    'lionmuon_k20':   'LiMuon K=20',
    'lionmuon_k100':  'LiMuon K=100',
    'lion':           'Lion (K=\u221e)',
}

# --- Visual style ---
# SignMuon family: red gradient, darker = lower K (more NS), triangle marker
# LiMuon family: blue gradient, darker = lower K, circle marker
# AdamW: gray, square marker
# Darker color = more frequent NS steps (lower K)

STYLE = {
    'AdamW':              {'color': '#888888', 'ls': '-',  'lw': 1.8, 'marker': 's'},
    # SignMuon family: dark red (K=1) -> light pink (K=inf)
    'Muon (K=1)':         {'color': '#8b0000', 'ls': '-',        'lw': 2.2, 'marker': '^'},
    'SignMuon K=2':       {'color': '#cc1111', 'ls': '--',       'lw': 2.2, 'marker': '^'},
    'SignMuon K=5':       {'color': '#e04040', 'ls': '-.',       'lw': 2.2, 'marker': '^'},
    'SignMuon K=20':      {'color': '#e87070', 'ls': ':',        'lw': 2.2, 'marker': '^'},
    'SignMuon K=100':     {'color': '#f0a0a0', 'ls': (0,(1,3)),  'lw': 2.2, 'marker': '^'},
    'Signum (K=\u221e)':  {'color': '#f5c8c8', 'ls': (0,(5,10)), 'lw': 2.2, 'marker': '^'},
    # LiMuon family: dark blue (K=1) -> light blue (K=inf)
    'LiMuon K=1':        {'color': '#00008b', 'ls': '-',        'lw': 2.2, 'marker': 'o'},
    'LiMuon K=2':        {'color': '#1144cc', 'ls': '--',       'lw': 2.2, 'marker': 'o'},
    'LiMuon K=5':        {'color': '#3377ee', 'ls': '-.',       'lw': 2.2, 'marker': 'o'},
    'LiMuon K=20':       {'color': '#6699ee', 'ls': ':',        'lw': 2.2, 'marker': 'o'},
    'LiMuon K=100':      {'color': '#99bbff', 'ls': (0,(1,3)),  'lw': 2.2, 'marker': 'o'},
    'Lion (K=\u221e)':   {'color': '#c8d8f5', 'ls': (0,(5,10)), 'lw': 2.2, 'marker': 'o'},
}

# Plot order: SignMuon family high-K first (behind), then LiMuon family
ORDER = [
    'adamw',
    'signum', 'signmuon_k100', 'signmuon_k20', 'signmuon_k5', 'signmuon_k2', 'muon',
    'lion', 'lionmuon_k100', 'lionmuon_k20', 'lionmuon_k5', 'lionmuon_k2', 'lionmuon_k1',
]


def parse_log(logfile):
    pat = re.compile(r'>Eval: Iter=(\d+).*?val_loss=([\d.]+)')
    iters, losses = [], []
    with open(logfile) as f:
        for line in f:
            m = pat.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return iters, losses


def get_runs(wandb_project, prefix):
    runs = {}
    for entry in sorted(os.listdir(WANDB_DIR)):
        run_dir = os.path.join(WANDB_DIR, entry)
        if not entry.startswith('run-') or not os.path.isdir(run_dir):
            continue
        cfg_f = os.path.join(run_dir, 'files/config.yaml')
        log_f = os.path.join(run_dir, 'files/output.log')
        summary_f = os.path.join(run_dir, 'files/wandb-summary.json')
        if not os.path.exists(cfg_f) or not os.path.exists(log_f):
            continue
        cfg = yaml.safe_load(open(cfg_f))
        if cfg.get('wandb_project', {}).get('value', '') != wandb_project:
            continue
        exp = cfg.get('experiment_name', {}).get('value', '')
        opt = cfg.get('opt', {}).get('value', '')
        K = cfg.get('muon_every_k', {}).get('value', 1)
        ns = cfg.get('muon_ns_steps', {}).get('value', 6)
        iters, losses = parse_log(log_f)
        if not losses:
            continue

        runtime = None
        if os.path.exists(summary_f):
            summary = json.load(open(summary_f))
            runtime = summary.get('_runtime', None)

        key = exp[len(prefix):] if prefix and exp.startswith(prefix) else exp
        best_loss = min(losses)
        runs[key] = {
            'exp': exp, 'opt': opt, 'K': K, 'ns_steps': ns,
            'iters': iters, 'losses': losses,
            'best_loss': best_loss,
            'runtime': runtime,
        }
    return runs


def compute_flops(n_layer, n_embd, seq_len, batch_size, iterations, opt, K=1, ns_steps=6,
                  vocab_size=50304):
    params_per_layer = 4 * n_embd**2 + 8 * n_embd**2
    embed_params = vocab_size * n_embd + seq_len * n_embd
    n_params = n_layer * params_per_layer + embed_params
    model_flops_per_iter = 6 * n_params * seq_len * batch_size

    d = n_embd
    ns_flops_per_layer = 0
    for (m, n) in [(d, 3*d), (d, d), (d, 4*d), (4*d, d)]:
        dmin, dmax = min(m, n), max(m, n)
        ns_flops_per_layer += ns_steps * 2 * dmin**2 * (2*dmax + dmin)
    ns_flops_per_iter = n_layer * ns_flops_per_layer

    if opt in ('adamw', 'signum', 'lion'):
        return iterations * model_flops_per_iter
    elif opt in ('sign_muon', 'lion_muon'):
        if K <= 1:
            return iterations * (model_flops_per_iter + ns_flops_per_iter)
        return iterations * (model_flops_per_iter + ns_flops_per_iter / K)
    return iterations * model_flops_per_iter


def plot_dataset(dataset_name):
    ds = DATASETS[dataset_name]
    runs = get_runs(ds['wandb_project'], ds['prefix'])
    if not runs:
        print(f"No runs found for {dataset_name}!")
        return

    print(f"\n{ds['title']} results (sorted by best val_loss):")
    print(f"  {'Optimizer':<18s} {'best_loss':>10s} {'ppl':>8s} {'wall(min)':>10s}")
    print("  " + "-" * 50)
    for key, r in sorted(runs.items(), key=lambda x: x[1]['best_loss']):
        label = NAMES.get(key, key)
        wall_min = r['runtime'] / 60 if r['runtime'] else 0
        print(f"  {label:<18s} {r['best_loss']:>10.4f} {math.exp(r['best_loss']):>8.1f} {wall_min:>10.1f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # Plot 1: val_loss vs iteration
    for key in ORDER:
        if key not in runs:
            continue
        r = runs[key]
        label = NAMES.get(key, key)
        s = STYLE.get(label, {})
        ax1.plot(r['iters'], r['losses'],
                 label=f"{label} ({r['best_loss']:.3f})",
                 color=s.get('color', '#333'), ls=s.get('ls', '-'), lw=s.get('lw', 2))
    ax1.set_yscale('log')
    ax1.set_ylim(None, 5)
    ax1.set_xlabel('Iteration', fontsize=16)
    ax1.set_ylabel('Val Loss', fontsize=16)
    ax1.set_title(f'Val Loss vs Iteration \u2014 124M, {ds["title"]}, 20K iters', fontsize=16)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=13)

    # Labels that should be placed to the left of their point
    LEFT_LABELS = {'Muon (K=1)', 'LiMuon K=1'}

    def smart_annotate(ax, label, x, y, fontsize=11):
        if label in LEFT_LABELS:
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(-8, 5), fontsize=fontsize, ha='right')
        else:
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(8, 5), fontsize=fontsize, ha='left')

    # Plot 2: Best val loss vs wall time
    wall_data = []
    for key in ORDER:
        if key not in runs:
            continue
        r = runs[key]
        label = NAMES.get(key, key)
        s = STYLE.get(label, {})
        wall_min = r['runtime'] / 60 if r['runtime'] else 0
        ax2.scatter(wall_min, r['best_loss'], s=180, zorder=5,
                    color=s.get('color', '#333'), marker=s.get('marker', 'o'),
                    edgecolors='black', linewidths=0.5)
        wall_data.append((label, wall_min, r['best_loss']))
    ax2.set_xlabel('Wall Time (min)', fontsize=16)
    ax2.set_ylabel('Best Val Loss', fontsize=16)
    ax2.set_title('Best Val Loss vs Wall Time', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=13)
    for label, x, y in wall_data:
        smart_annotate(ax2, label, x, y)

    # Plot 3: Best val loss vs FLOPs
    n_layer, n_embd, seq_len, batch_size, iterations = 12, 768, 512, 32, 20000
    flops_data = []
    for key in ORDER:
        if key not in runs:
            continue
        r = runs[key]
        label = NAMES.get(key, key)
        s = STYLE.get(label, {})
        flops = compute_flops(n_layer, n_embd, seq_len, batch_size, iterations,
                              r['opt'], K=r['K'], ns_steps=r['ns_steps'])
        ax3.scatter(flops, r['best_loss'], s=180, zorder=5,
                    color=s.get('color', '#333'), marker=s.get('marker', 'o'),
                    edgecolors='black', linewidths=0.5)
        flops_data.append((label, flops, r['best_loss']))
    ax3.set_xlabel('Total Training FLOPs', fontsize=16)
    ax3.set_ylabel('Best Val Loss', fontsize=16)
    ax3.set_title('Best Val Loss vs FLOPs', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=13)
    ax3.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    for label, x, y in flops_data:
        smart_annotate(ax3, label, x, y)

    # Fit loss(K) = a - b/(c+K) + d/(e+K),  a = L_∞ fixed from data
    #   - b/(c+K): NS benefit (spectral alignment, half-life c)
    #   - d/(e+K): cost of NS overhead / alternation penalty (half-life e)
    #   - At K→∞: loss → a = L_∞
    import numpy as np  # noqa: E402
    from scipy.optimize import curve_fit  # noqa: E402
    c_base = compute_flops(n_layer, n_embd, seq_len, batch_size, iterations, 'adamw')

    SIGNMUON_KEYS = ['muon', 'signmuon_k2', 'signmuon_k5', 'signmuon_k20', 'signmuon_k100', 'signum']
    LIMUON_KEYS = ['lionmuon_k1', 'lionmuon_k2', 'lionmuon_k5', 'lionmuon_k20', 'lionmuon_k100', 'lion']
    K_INF = 1e6

    def loss_model(K, a, b, c, d, e):
        return a - b / (c + K) + d / (e + K)

    for family_keys, inf_key, color, fname in [
        (SIGNMUON_KEYS, 'signum', '#cc1111', 'SignMuon'),
        (LIMUON_KEYS, 'lion', '#1144cc', 'LiMuon'),
    ]:
        k_pts, l_pts = [], []
        for key in family_keys:
            if key not in runs:
                continue
            r = runs[key]
            k = K_INF if key == inf_key else float(r['K'])
            k_pts.append(k)
            l_pts.append(r['best_loss'])
        if len(k_pts) >= 4:
            K_arr = np.array(k_pts)
            L_arr = np.array(l_pts)
            L_inf = runs[inf_key]['best_loss'] if inf_key in runs else max(l_pts)

            def fit_fn(K, b, c, d, e):
                return loss_model(K, L_inf, b, c, d, e)

            try:
                popt, _ = curve_fit(fit_fn, K_arr, L_arr, p0=[1, 5, 0.1, 0.1],
                                    bounds=([0, 0.01, 0, 0.001], [100, 1000, 100, 1000]),
                                    maxfev=50000)
                b, c, d, e = popt

                # Plot on FLOPs axis
                k_smooth = np.logspace(0, 6, 500)
                l_smooth = loss_model(k_smooth, L_inf, b, c, d, e)
                c_ns_total = c_base * (compute_flops(n_layer, n_embd, seq_len, batch_size,
                                        iterations, 'sign_muon', K=1) / c_base - 1)
                f_smooth = c_base + c_ns_total / k_smooth
                eq = (f'L={L_inf:.2f} \u2212 {b:.2f}/({c:.1f}+K)'
                      f' + {d:.2f}/({e:.1f}+K)')
                ax3.plot(f_smooth, l_smooth, color=color, ls='--', lw=1.5, alpha=0.6,
                         label=f'{fname}: {eq}')

                l_pred = loss_model(K_arr, L_inf, b, c, d, e)
                rmse = np.sqrt(np.mean((L_arr - l_pred)**2))
                print(f"  {fname}: L={L_inf:.3f} - {b:.2f}/({c:.1f}+K) + {d:.2f}/({e:.1f}+K)  RMSE={rmse:.4f}")
            except RuntimeError as e:
                print(f"  {fname}: fit failed: {e}")

    ax3.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f'{dataset_name}_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <{'|'.join(DATASETS)}|all>")
        sys.exit(1)

    target = sys.argv[1]
    if target == 'all':
        for ds in DATASETS:
            plot_dataset(ds)
    elif target in DATASETS:
        plot_dataset(target)
    else:
        print(f"Unknown dataset: {target}. Choose from: {', '.join(DATASETS)}, all")
        sys.exit(1)


if __name__ == '__main__':
    main()
