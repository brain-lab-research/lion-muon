"""Plot results for optimizer comparison experiments.

Usage:
    python scripts/plotting/plot_results.py fineweb
    python scripts/plotting/plot_results.py slimpajama
    python scripts/plotting/plot_results.py all
"""

import re
import os
import sys
import glob
import json
import math
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = '/home/arman/llm-baselines'
LOCAL_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '')
WANDB_DIR = os.path.join(BASE_DIR, 'wandb')
RESULTS_DIR = os.path.join(LOCAL_BASE_DIR, 'results')

DATASETS = {
    'fineweb': {
        'wandb_project': 'sign-muon-main',
        'prefix': '',
        'local_prefix': 'fw_',
        'title': 'FineWeb',
    },
    'slimpajama': {
        'wandb_project': 'sign-muon-slimpajama',
        'prefix': 'spj_',
        'local_prefix': 'spj_',
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


def get_local_runs(exps_dir, prefix):
    """Load runs from local exps/ folder (summary.json files)."""
    runs = {}
    for summary_path in sorted(glob.glob(os.path.join(exps_dir, '*/summary.json'))):
        exp = os.path.basename(os.path.dirname(summary_path))
        try:
            with open(summary_path) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue  # still running / mid-write
        val_loss = d.get('val_loss', [])
        args = d.get('args', {})
        if not val_loss:
            continue
        eval_interval = args.get('eval_interval', 500)
        total_iters = args.get('iterations', 0)
        expected_evals = total_iters // eval_interval
        if expected_evals and len(val_loss) < expected_evals:
            continue  # still running
        # Only include runs whose name starts with the given prefix
        if prefix and not exp.startswith(prefix):
            continue
        opt = args.get('opt', '')
        K = args.get('muon_every_k', 1)
        ns = args.get('muon_ns_steps', 6)
        # type=bool in argparse doesn't parse 'False' correctly; use name instead
        nesterov = 'nonesterov' not in exp
        iters = list(range(eval_interval, eval_interval * (len(val_loss) + 1), eval_interval))
        key = exp[len(prefix):] if prefix and exp.startswith(prefix) else exp
        runs[key] = {
            'exp': exp, 'opt': opt, 'K': K, 'ns_steps': ns,
            'nesterov': nesterov,
            'iters': iters, 'losses': val_loss,
            'best_loss': min(val_loss),
            'runtime': d.get('wall_time', None),
            'iterations': total_iters or len(val_loss) * eval_interval,
        }
    return runs


def get_runs(wandb_project, prefix):
    runs = {}
    if not os.path.isdir(WANDB_DIR):
        return runs
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
        nesterov = 'nonesterov' not in exp
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
            'nesterov': nesterov,
            'iters': iters, 'losses': losses,
            'best_loss': best_loss,
            'runtime': runtime,
            'iterations': cfg.get('iterations', {}).get('value', len(losses) * 500),
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

    if opt in ('adamw', 'lion'):
        return iterations * model_flops_per_iter
    elif opt in ('muon', 'sign_muon', 'lion_muon'):
        if K <= 1:
            return iterations * (model_flops_per_iter + ns_flops_per_iter)
        return iterations * (model_flops_per_iter + ns_flops_per_iter / K)
    return iterations * model_flops_per_iter


def plot_dataset(dataset_name):
    ds = DATASETS[dataset_name]
    runs = get_runs(ds['wandb_project'], ds['prefix'])
    if not runs:
        local_exps = os.path.join(LOCAL_BASE_DIR, 'exps')
        runs = get_local_runs(local_exps, ds['local_prefix'])
    if not runs:
        print(f"No runs found for {dataset_name}!")
        return

    def bkey(key):
        return key.replace('_nonesterov', '')

    print(f"\n{ds['title']} results (sorted by best val_loss):")
    print(f"  {'Optimizer':<30s} {'nesterov':>8s} {'best_loss':>10s} {'ppl':>8s} {'wall(min)':>10s}")
    print("  " + "-" * 70)
    for key, r in sorted(runs.items(), key=lambda x: x[1]['best_loss']):
        label = NAMES.get(bkey(key), key)
        nes_str = 'yes' if r.get('nesterov', True) else 'no'
        wall_min = r['runtime'] / 60 if (r['runtime'] and r['runtime'] > 0) else 0
        print(f"  {label:<30s} {nes_str:>8s} {r['best_loss']:>10.4f} {math.exp(r['best_loss']):>8.1f} {wall_min:>10.1f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    LEFT_LABELS = {'Muon (K=1)', 'LiMuon K=1'}

    def smart_annotate(ax, label, x, y, nesterov=True, fontsize=10):
        text = label if nesterov else f"{label}*"
        if label in LEFT_LABELS:
            ax.annotate(text, (x, y), textcoords="offset points",
                        xytext=(-8, 5), fontsize=fontsize, ha='right')
        else:
            ax.annotate(text, (x, y), textcoords="offset points",
                        xytext=(8, 5), fontsize=fontsize, ha='left')

    n_layer, n_embd, seq_len, batch_size = 12, 768, 512, 32

    # Plot both nesterov and non-nesterov variants together
    # Iterate ORDER twice: first nesterov (solid), then non-nesterov (faded/dashed)
    for nesterov_pass in [True, False]:
        for key in ORDER:
            rkey = key if nesterov_pass else key + '_nonesterov'
            if rkey not in runs:
                continue
            r = runs[rkey]
            if r.get('nesterov', True) != nesterov_pass:
                continue
            label = NAMES.get(bkey(rkey), rkey)
            s = STYLE.get(label, {})
            color = s.get('color', '#333')
            alpha = 1.0 if nesterov_pass else 0.45
            lw = s.get('lw', 2) if nesterov_pass else s.get('lw', 2) * 0.8
            ls = s.get('ls', '-') if nesterov_pass else (0, (3, 2))
            legend_label = f"{label} ({r['best_loss']:.3f})" if nesterov_pass else f"{label}* no-Nes ({r['best_loss']:.3f})"

            # Plot 1: val loss curve
            ax1.plot(r['iters'], r['losses'], label=legend_label,
                     color=color, ls=ls, lw=lw, alpha=alpha)

    ax1.set_yscale('log')
    ax1.set_ylim(None, 5)
    ax1.set_xlabel('Iteration', fontsize=16)
    ax1.set_ylabel('Val Loss', fontsize=16)
    ax1.set_title(f'Val Loss vs Iteration — 124M, {ds["title"]}', fontsize=16)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=13)

    # Plots 2 & 3: scatter
    wall_data, flops_data = [], []
    for nesterov_pass in [True, False]:
        for key in ORDER:
            rkey = key if nesterov_pass else key + '_nonesterov'
            if rkey not in runs:
                continue
            r = runs[rkey]
            if r.get('nesterov', True) != nesterov_pass:
                continue
            label = NAMES.get(bkey(rkey), rkey)
            s = STYLE.get(label, {})
            color = s.get('color', '#333')
            alpha = 1.0 if nesterov_pass else 0.45
            marker = s.get('marker', 'o') if nesterov_pass else 'x'
            wall_min = r['runtime'] / 60 if (r['runtime'] and r['runtime'] > 0) else 0
            flops = compute_flops(n_layer, n_embd, seq_len, batch_size,
                                  r.get('iterations', 64000),
                                  r['opt'], K=r['K'], ns_steps=r['ns_steps'])
            ec = {} if marker == 'x' else {'edgecolors': 'black', 'linewidths': 0.5}
            ax2.scatter(wall_min, r['best_loss'], s=180, zorder=5,
                        color=color, marker=marker, alpha=alpha, **ec)
            ax3.scatter(flops, r['best_loss'], s=180, zorder=5,
                        color=color, marker=marker, alpha=alpha, **ec)
            wall_data.append((label, wall_min, r['best_loss'], nesterov_pass))
            flops_data.append((label, flops, r['best_loss'], nesterov_pass))

    for label, x, y, nes in wall_data:
        smart_annotate(ax2, label, x, y, nesterov=nes)
    for label, x, y, nes in flops_data:
        smart_annotate(ax3, label, x, y, nesterov=nes)

    ax2.set_xlabel('Wall Time (min)', fontsize=16)
    ax2.set_ylabel('Best Val Loss', fontsize=16)
    ax2.set_title('Best Val Loss vs Wall Time', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=13)

    ax3.set_xlabel('Total Training FLOPs', fontsize=16)
    ax3.set_ylabel('Best Val Loss', fontsize=16)
    ax3.set_title('Best Val Loss vs FLOPs', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=13)
    ax3.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    # Legend note for non-nesterov
    has_nonesterov = any(not v.get('nesterov', True) for v in runs.values())
    if has_nonesterov:
        fig.text(0.5, 0.01, '* = no Nesterov (faded)', ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.03 if has_nonesterov else 0, 1, 1])
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
