"""Plot results for optimizer comparison experiments.

Usage:
    python scripts/plotting/plot_results.py fineweb base
    python scripts/plotting/plot_results.py fineweb llama
    python scripts/plotting/plot_results.py slimpajama base
    python scripts/plotting/plot_results.py slimpajama llama
    python scripts/plotting/plot_results.py all base
    python scripts/plotting/plot_results.py
"""

import re
import os
import sys
import glob
import json
import math
from collections import defaultdict
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = '/home/arman/llm-baselines'
LOCAL_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '')
WANDB_DIR = os.path.join(BASE_DIR, 'wandb')
RESULTS_DIR = os.path.join(LOCAL_BASE_DIR, 'results')

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
    'wikitext': {
        'wandb_project': None,
        'prefix': 'wt_',
        'title': 'WikiText-103',
    },
}

MODELS = {'base', 'llama'}

# Display names: experiment key -> label
# Conceptual families:
#   SignMuon family (SGD momentum): Muon(K=1) -> SignMuon K=2,5,20,100 -> Signum(K=inf)
#   LionMuon family (dual-EMA):      LionMuon K=1 -> K=2,5,20,100 -> Lion(K=inf)
#   AdamW: standalone baseline
NAMES = {
    'adamw':          'AdamW',
    'muon_fixed':     'Muon (P=1)',
    'signmuon_fixed_k2':    'SignMuon P=2',
    'signmuon_fixed_k5':    'SignMuon P=5',
    'signmuon_fixed_k20':   'SignMuon P=20',
    'signmuon_fixed_k100':  'SignMuon P=100',
    'signum_fixed':   'Signum (P=\u221e)',
    'lionmuon_k1':    'LionMuon P=1',
    'lionmuon_k2':    'LionMuon P=2',
    'lionmuon_k5':    'LionMuon P=5',
    'lionmuon_k20':   'LionMuon P=20',
    'lionmuon_k100':  'LionMuon P=100',
    'lion':           'Lion (P=\u221e)',
}

# --- Visual style ---
# SignMuon family: red gradient, darker = lower K (more NS), triangle marker
# LionMuon family: blue gradient, darker = lower K, circle marker
# AdamW: gray, square marker
# Darker color = more frequent NS steps (lower K)

STYLE = {
    'AdamW':              {'color': '#888888', 'ls': '-',  'lw': 1.8, 'marker': 's'},
    # SignMuon family: dark red (K=1) -> light pink (K=inf)
    'Muon (P=1)':         {'color': '#8b0000', 'ls': '-',        'lw': 2.2, 'marker': '^'},
    'SignMuon P=2':       {'color': '#cc1111', 'ls': '--',       'lw': 2.2, 'marker': '^'},
    'SignMuon P=5':       {'color': '#e04040', 'ls': '-.',       'lw': 2.2, 'marker': '^'},
    'SignMuon P=20':      {'color': '#e87070', 'ls': ':',        'lw': 2.2, 'marker': '^'},
    'SignMuon P=100':     {'color': '#f0a0a0', 'ls': (0,(1,3)),  'lw': 2.2, 'marker': '^'},
    'Signum (P=\u221e)':  {'color': '#f5c8c8', 'ls': (0,(5,10)), 'lw': 2.2, 'marker': '^'},
    # LionMuon family: dark blue (K=1) -> light blue (K=inf)
    'LionMuon P=1':        {'color': '#00008b', 'ls': '-',        'lw': 2.2, 'marker': 'o'},
    'LionMuon P=2':        {'color': '#1144cc', 'ls': '--',       'lw': 2.2, 'marker': 'o'},
    'LionMuon P=5':        {'color': '#3377ee', 'ls': '-.',       'lw': 2.2, 'marker': 'o'},
    'LionMuon P=20':       {'color': '#6699ee', 'ls': ':',        'lw': 2.2, 'marker': 'o'},
    'LionMuon P=100':      {'color': '#99bbff', 'ls': (0,(1,3)),  'lw': 2.2, 'marker': 'o'},
    'Lion (P=\u221e)':   {'color': '#c8d8f5', 'ls': (0,(5,10)), 'lw': 2.2, 'marker': 'o'},
}

# Plot order: SignMuon family high-K first (behind), then LionMuon family
ORDER = [
    'adamw',
    'signum_fixed', 'signmuon_fixed_k100', 'signmuon_fixed_k20', 'signmuon_fixed_k5', 'signmuon_fixed_k2', 'muon_fixed',
    'lion', 'lionmuon_k100', 'lionmuon_k20', 'lionmuon_k5', 'lionmuon_k2', 'lionmuon_k1',
]

ALLOWED_BASE_KEYS = {
    'adamw',
    'muon_fixed',
    'signmuon_fixed_k2', 'signmuon_fixed_k5', 'signmuon_fixed_k20', 'signmuon_fixed_k100',
    'signum_fixed',
    'lionmuon_k1', 'lionmuon_k2', 'lionmuon_k5', 'lionmuon_k20', 'lionmuon_k100',
    'lion',
}

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


def _normalize_dataset_tag(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _dataset_prefix_candidates(dataset_name):
    norm = _normalize_dataset_tag(dataset_name)
    if norm == 'fineweb':
        return ['fw', norm]
    if norm == 'slimpajama':
        return ['spj', norm]
    if norm == 'wikitext':
        return ['wt', norm]
    # For unknown datasets, use normalized name directly.
    return [norm]


def get_local_prefix(dataset_name, model_name):
    tag = _dataset_prefix_candidates(dataset_name)[0]
    return f"{tag}_{model_name}_"


def get_legacy_local_prefix(dataset_name):
    tag = _dataset_prefix_candidates(dataset_name)[0]
    return f"{tag}_"


def get_local_runs(exps_dir, prefix, model_name=None):
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
        opt_diag = d.get('opt_diagnostics', {}) or {}
        opt_diag_hist = d.get('opt_diag_history', []) or []
        opt_step_diag_hist = d.get('opt_step_diag_history', []) or []
        if model_name and args.get('model') and args.get('model') != model_name:
            continue
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
        iters = list(range(0, eval_interval * len(val_loss), eval_interval))
        key = exp[len(prefix):] if prefix and exp.startswith(prefix) else exp
        runs[key] = {
            'exp': exp, 'opt': opt, 'K': K, 'ns_steps': ns,
            'opt_diag': opt_diag,
            'opt_diag_hist': opt_diag_hist,
            'opt_step_diag_hist': opt_step_diag_hist,
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


def _is_excluded_run_key(key):
    return "signdmuon" in key or "lidmuon" in key or "rmsspectral" in key


def compute_flops(n_layer, n_embd, seq_len, batch_size, iterations, opt, K=1, ns_steps=6,
                  vocab_size=50304, opt_diag=None):
    params_per_layer = 4 * n_embd**2 + 8 * n_embd**2
    embed_params = vocab_size * n_embd + seq_len * n_embd
    n_params = n_layer * params_per_layer + embed_params
    model_flops_per_iter = 6 * n_params * seq_len * batch_size

    # If exact per-run optimizer diagnostics are present, use them.
    if isinstance(opt_diag, dict) and opt_diag:
        ns_total = float(opt_diag.get('ns_flops', 0.0) or 0.0)
        sign_total = float(opt_diag.get('sign_flops', 0.0) or 0.0)
        if (ns_total + sign_total) > 0.0:
            total = iterations * model_flops_per_iter + ns_total + sign_total
            return total, total

    d = n_embd
    ns_flops_per_layer = 0
    matvec_mn_sum_per_layer = 0
    for (m, n) in [(d, 3*d), (d, d), (d, 4*d), (4*d, d)]:
        dmin, dmax = min(m, n), max(m, n)
        ns_flops_per_layer += ns_steps * 2 * dmin**2 * (2*dmax + dmin)
        matvec_mn_sum_per_layer += m * n
    ns_flops_per_iter = n_layer * ns_flops_per_layer
    sign_flops_per_iter = n_layer * matvec_mn_sum_per_layer

    if opt in ('adamw', 'lion'):
        total = iterations * model_flops_per_iter
        return total, total
    elif opt in ('muon', 'sign_muon', 'lion_muon'):
        if K <= 1:
            total = iterations * (model_flops_per_iter + ns_flops_per_iter)
            return total, total
        p_muon = 1.0 / K
        extra = p_muon * ns_flops_per_iter + (1.0 - p_muon) * sign_flops_per_iter
        total = iterations * (model_flops_per_iter + extra)
        return total, total
    total = iterations * model_flops_per_iter
    return total, total


def _base_key(key):
    bkey = key.replace('_nonesterov', '')
    # Keep compatibility with newer run names.
    if bkey == 'muon':
        return 'muon_fixed'
    if bkey == 'signum':
        return 'signum_fixed'
    return bkey


def _display_name_for_key(key):
    bkey = _base_key(key)
    if bkey in NAMES:
        return NAMES[bkey]
    return bkey


def _iter_plot_runs(runs, ordered_keys):
    for nesterov_pass in (True, False):
        for key in ordered_keys:
            rkey = key if nesterov_pass else key + '_nonesterov'
            r = runs.get(rkey)
            if not r or r.get('nesterov', True) != nesterov_pass:
                continue
            label = _display_name_for_key(rkey)
            style = STYLE.get(label, {})
            yield rkey, r, label, style, nesterov_pass


def _normalize_run_keys(runs):
    """Map newer run names to canonical keys expected by ORDER/NAMES."""
    out = {}
    for k, v in runs.items():
        nk = k
        if k == 'muon':
            nk = 'muon_fixed'
        elif k == 'signum':
            nk = 'signum_fixed'
        elif k == 'muon_nonesterov':
            nk = 'muon_fixed_nonesterov'
        elif k == 'signum_nonesterov':
            nk = 'signum_fixed_nonesterov'
        out[nk] = v
    return out





def plot_dataset(dataset_name, model_name, scale=None):
    ds = DATASETS.get(dataset_name, {
        'wandb_project': None,
        'prefix': '',
        'title': dataset_name,
    })

    runs = {}
    model_size_label = '124M'
    n_layer, n_embd, seq_len, batch_size = 12, 768, 512, 32
    if ds['wandb_project']:
        runs = get_runs(ds['wandb_project'], ds['prefix'])

    if not runs:
        want_720m = (scale is None or scale == '720m')
        want_355m = (scale is None or scale == '355m')
        want_124m = (scale is None or scale == '124m')

        # Prefer dedicated 720M fineweb/base runs when present.
        if want_720m and dataset_name == 'fineweb' and model_name == 'base':
            local_exps_720m = os.path.join(LOCAL_BASE_DIR, 'exps_720m')
            runs = get_local_runs(local_exps_720m, 'fw_base_720m_', model_name=model_name)
            if runs:
                model_size_label = '720M'
                n_layer, n_embd, seq_len, batch_size = 12, 2048, 512, 1984  # eff. batch = 64 * acc_steps=31

        # Prefer dedicated 355M fineweb/base runs when present.
        if want_355m and not runs and dataset_name == 'fineweb' and model_name == 'base':
            local_exps_355m = os.path.join(LOCAL_BASE_DIR, 'exps_355m')
            runs = get_local_runs(local_exps_355m, 'fw_base_355m_', model_name=model_name)
            if runs:
                model_size_label = '355M'
                n_layer, n_embd, seq_len, batch_size = 24, 1024, 1024, 512  # eff. batch = 64 * acc_steps=8


        if want_124m and not runs:
            local_exps = os.path.join(LOCAL_BASE_DIR, 'exps')
            auto_prefix = get_local_prefix(dataset_name, model_name)
            runs = get_local_runs(local_exps, auto_prefix, model_name=model_name)
            # Fallback for older naming patterns that used dataset-only prefix.
            if not runs:
                legacy_prefix = get_legacy_local_prefix(dataset_name)
                runs = get_local_runs(local_exps, legacy_prefix, model_name=model_name)
    if not runs:
        print(f"No runs found for {dataset_name} ({model_name})!")
        return

    runs = {
        k: v for k, v in runs.items()
        if not _is_excluded_run_key(k) and _base_key(k) in ALLOWED_BASE_KEYS
    }
    runs = _normalize_run_keys(runs)

    dynamic_order = ORDER

    model_title = 'Base' if model_name == 'base' else 'Llama'
    print(f"\n{ds['title']} ({model_title}) results (sorted by best val_loss):")
    print(f"  {'Optimizer':<30s} {'nesterov':>8s} {'best_loss':>10s} {'ppl':>8s} {'wall(min)':>10s} {'FLOPs':>14s}")
    print("  " + "-" * 86)
    for key, r in sorted(runs.items(), key=lambda x: x[1]['best_loss']):
        label = _display_name_for_key(key)
        nes_str = 'yes' if r.get('nesterov', True) else 'no'
        wall_min = r['runtime'] / 60 if (r['runtime'] and r['runtime'] > 0) else 0
        low, high = compute_flops(
            n_layer, n_embd, seq_len, batch_size,
            r.get('iterations', 64000),
            r['opt'], K=r['K'], ns_steps=r['ns_steps'],
            opt_diag=r.get('opt_diag'),
        )
        flops_str = f"{low/1e18:.3f}" if abs(high - low) < 1e-9 else f"{low/1e18:.3f}-{high/1e18:.3f}"
        print(f"  {label:<30s} {nes_str:>8s} {r['best_loss']:>10.4f} {math.exp(r['best_loss']):>8.1f} {wall_min:>10.1f}  {flops_str:>13s}e18")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    has_nonesterov = any(not v.get('nesterov', True) for v in runs.values())

    for rkey, r, label, s, nesterov_pass in _iter_plot_runs(runs, dynamic_order):
        color = s.get('color', '#333')
        alpha = 1.0 if nesterov_pass else 0.45
        lw = s.get('lw', 2) if nesterov_pass else s.get('lw', 2) * 0.8
        ls = s.get('ls', '-') if nesterov_pass else (0, (3, 2))
        suffix = '' if nesterov_pass else '* no-Nes'
        legend_label = f"{label}{suffix} ({r['best_loss']:.3f})"

        # Left panel: val loss vs iteration.
        ax1.plot(r['iters'], r['losses'], label=legend_label,
                 color=color, ls=ls, lw=lw, alpha=alpha)

        # Right panel: val loss vs cumulative FLOPs (same curve, different x-axis).
        flops_curve = [compute_flops(n_layer, n_embd, seq_len, batch_size,
                                     it, r['opt'], K=r['K'], ns_steps=r['ns_steps'],
                                     opt_diag=r.get('opt_diag'))[0]
                       for it in r['iters']]
        ax2.plot(flops_curve, r['losses'], label=legend_label,
                 color=color, ls=ls, lw=lw, alpha=alpha)

    y_min_data = min(min(r['losses']) for r in runs.values() if r.get('losses'))
    y_max = {'355M': 4.0, '720M': 5.0}.get(model_size_label, 4.5)
    y_min = max(1e-6, y_min_data * 0.98)

    title_suffix = f'{model_size_label}, {ds["title"]} ({model_title})'

    ax1.set_yscale('log')
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('Iteration', fontsize=16)
    ax1.set_ylabel('Val Loss', fontsize=16)
    ax1.set_title(f'Val Loss vs Iteration — {title_suffix}', fontsize=16)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=13)

    ax2.set_yscale('log')
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('Total Training FLOPs', fontsize=16)
    ax2.set_ylabel('Val Loss', fontsize=16)
    ax2.set_title(f'Val Loss vs FLOPs — {title_suffix}', fontsize=16)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(labelsize=13)
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    if has_nonesterov:
        fig.text(0.5, 0.01, '* = no Nesterov (faded)', ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.03 if has_nonesterov else 0, 1, 1])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if model_size_label == '355M':
        suffix = '_355m'
    elif model_size_label == '720M':
        suffix = '_720m'
    else:
        suffix = ''
    out_path = os.path.join(RESULTS_DIR, f'{dataset_name}_{model_name}{suffix}_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")

def discover_available_dataset_model_pairs():
    """Discover (dataset, model) pairs from local summary files."""
    local_exps_dirs = [
        os.path.join(LOCAL_BASE_DIR, 'exps_355m'),
        os.path.join(LOCAL_BASE_DIR, 'exps'),
    ]
    pairs = set()
    for local_exps in local_exps_dirs:
        if not os.path.isdir(local_exps):
            continue
        for summary_path in glob.glob(os.path.join(local_exps, '*/summary.json')):
            try:
                with open(summary_path) as f:
                    d = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            args = d.get('args', {}) or {}
            dataset = args.get('dataset')
            model = args.get('model')

            if dataset in DATASETS and model in MODELS:
                pairs.add((dataset, model))

    return sorted(pairs)


def main():
    args = [a for a in sys.argv[1:]]
    scale = None
    for s in ('124m', '355m', '720m'):
        if s in args:
            scale = s
            args.remove(s)

    if not args:
        pairs = discover_available_dataset_model_pairs()
        if not pairs:
            print("No local results found in exps/ to plot.")
            sys.exit(0)
        print("Auto-discovered result groups:")
        for ds, model_name in pairs:
            print(f"  - {ds} {model_name}")
            plot_dataset(ds, model_name, scale=scale)
        return

    if len(args) < 2:
        print(f"Usage: {sys.argv[0]} <dataset|all> <base|llama> [124m|355m|720m]")
        print(f"   or: {sys.argv[0]}   # auto-plot all available local results")
        sys.exit(1)

    target = args[0]
    model_name = args[1].lower()
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Choose from: {', '.join(sorted(MODELS))}")
        sys.exit(1)

    if target == 'all':
        for ds in DATASETS:
            plot_dataset(ds, model_name, scale=scale)
    else:
        # Allow unknown datasets for local-only plotting via automatic prefix.
        plot_dataset(target, model_name, scale=scale)


if __name__ == '__main__':
    main()
