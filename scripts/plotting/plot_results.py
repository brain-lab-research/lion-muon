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

NAMES = {
    'adamw':          'AdamW',
    'signum':         'Signum',
    'muon':           'Muon',
    'lion':           'Lion',
    'signmuon_k2':    'SignMuon K=2',
    'signmuon_k5':    'SignMuon K=5',
    'signmuon_k20':   'SignMuon K=20',
    'lionmuon_k1':    'LionMuon K=1',
    'lionmuon_k2':    'LionMuon K=2',
    'lionmuon_k5':    'LionMuon K=5',
    'lionmuon_k20':   'LionMuon K=20',
}

# --- Visual style ---
# Baselines: muted distinct colors, solid thin
# Muon: green, solid thick
# SignMuon: red family, dashed (different dash patterns per K)
# LionMuon: blue family, solid (different shades per K)

STYLE = {
    'AdamW':           {'color': '#888888', 'ls': '-',      'lw': 1.5, 'marker': 's'},
    'Signum':          {'color': '#cc8800', 'ls': '-',      'lw': 1.5, 'marker': 'v'},
    'Lion':            {'color': '#9467bd', 'ls': '-',      'lw': 1.5, 'marker': 'D'},
    'Muon':            {'color': '#2ca02c', 'ls': '-',      'lw': 2.5, 'marker': 'P'},
    'SignMuon K=2':    {'color': '#e03030', 'ls': '--',     'lw': 2.0, 'marker': '^'},
    'SignMuon K=5':    {'color': '#cc5555', 'ls': '-.',     'lw': 2.0, 'marker': '^'},
    'SignMuon K=20':   {'color': '#aa7777', 'ls': ':',      'lw': 2.0, 'marker': '^'},
    'LionMuon K=1':    {'color': '#0055cc', 'ls': '-',      'lw': 2.5, 'marker': 'o'},
    'LionMuon K=2':    {'color': '#1177ee', 'ls': '--',     'lw': 2.5, 'marker': 'o'},
    'LionMuon K=5':    {'color': '#3399ff', 'ls': '-.',     'lw': 2.5, 'marker': 'o'},
    'LionMuon K=20':   {'color': '#77bbff', 'ls': ':',      'lw': 2.5, 'marker': 'o'},
}

ORDER = [
    'adamw', 'signum', 'lion', 'muon',
    'signmuon_k20', 'signmuon_k5', 'signmuon_k2',
    'lionmuon_k20', 'lionmuon_k5', 'lionmuon_k2',
    'lionmuon_k1',
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

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
    ax1.set_ylim(None, 6)
    ax1.set_xlabel('Iteration', fontsize=13)
    ax1.set_ylabel('Val Loss', fontsize=13)
    ax1.set_title(f'Val Loss vs Iteration — GPT-2 124M, {ds["title"]}, 20K iters', fontsize=13)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Best val loss vs wall time
    for key in ORDER:
        if key not in runs:
            continue
        r = runs[key]
        label = NAMES.get(key, key)
        s = STYLE.get(label, {})
        wall_min = r['runtime'] / 60 if r['runtime'] else 0
        ax2.scatter(wall_min, r['best_loss'], s=150, zorder=5,
                    color=s.get('color', '#333'), marker=s.get('marker', 'o'))
        ax2.annotate(label, (wall_min, r['best_loss']),
                     textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax2.set_xlabel('Wall Time (min)', fontsize=13)
    ax2.set_ylabel('Best Val Loss', fontsize=13)
    ax2.set_title('Best Val Loss vs Wall Time', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Best val loss vs FLOPs
    n_layer, n_embd, seq_len, batch_size, iterations = 12, 768, 512, 32, 20000
    for key in ORDER:
        if key not in runs:
            continue
        r = runs[key]
        label = NAMES.get(key, key)
        s = STYLE.get(label, {})
        flops = compute_flops(n_layer, n_embd, seq_len, batch_size, iterations,
                              r['opt'], K=r['K'], ns_steps=r['ns_steps'])
        ax3.scatter(flops, r['best_loss'], s=150, zorder=5,
                    color=s.get('color', '#333'), marker=s.get('marker', 'o'))
        ax3.annotate(label, (flops, r['best_loss']),
                     textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax3.set_xlabel('Total Training FLOPs', fontsize=13)
    ax3.set_ylabel('Best Val Loss', fontsize=13)
    ax3.set_title('Best Val Loss vs FLOPs', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f'{dataset_name}_results.png')
    plt.savefig(out_path, dpi=150)
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
