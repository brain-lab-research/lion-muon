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
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing import event_accumulator

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
    'lionmuon_srank_a0.05': 'LiMuon sRank a=0.05',
    'lionmuon_srank_a0.1':  'LiMuon sRank a=0.10',
    'lionmuon_srank_a0.2':  'LiMuon sRank a=0.20',
    'lionmuon_srank_a0.5':  'LiMuon sRank a=0.50',
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
    'LiMuon sRank a=0.05': {'color': '#006d5b', 'ls': '-',        'lw': 2.4, 'marker': 'D'},
    'LiMuon sRank a=0.10': {'color': '#008f78', 'ls': '--',       'lw': 2.4, 'marker': 'D'},
    'LiMuon sRank a=0.20': {'color': '#2fa892', 'ls': '-.',       'lw': 2.4, 'marker': 'D'},
    'LiMuon sRank a=0.50': {'color': '#67c7b7', 'ls': (0,(1,3)),  'lw': 2.4, 'marker': 'D'},
}

# Plot order: SignMuon family high-K first (behind), then LiMuon family
ORDER = [
    'adamw',
    'signum', 'signmuon_k100', 'signmuon_k20', 'signmuon_k5', 'signmuon_k2', 'muon',
    'lion', 'lionmuon_k100', 'lionmuon_k20', 'lionmuon_k5', 'lionmuon_k2', 'lionmuon_k1',
    'lionmuon_srank_a0.5', 'lionmuon_srank_a0.2', 'lionmuon_srank_a0.1', 'lionmuon_srank_a0.05',
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
        srank_alpha = args.get('srank_alpha', 0.0)
        # type=bool in argparse doesn't parse 'False' correctly; use name instead
        nesterov = 'nonesterov' not in exp
        iters = list(range(0, eval_interval * len(val_loss), eval_interval))
        key = exp[len(prefix):] if prefix and exp.startswith(prefix) else exp
        runs[key] = {
            'exp': exp, 'opt': opt, 'K': K, 'ns_steps': ns,
            'srank_alpha': srank_alpha,
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
        srank_alpha = cfg.get('srank_alpha', {}).get('value', 0.0)
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
            'srank_alpha': srank_alpha,
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
                  srank_alpha=0.0, vocab_size=50304, opt_diag=None):
    params_per_layer = 4 * n_embd**2 + 8 * n_embd**2
    embed_params = vocab_size * n_embd + seq_len * n_embd
    n_params = n_layer * params_per_layer + embed_params
    model_flops_per_iter = 6 * n_params * seq_len * batch_size

    # If exact per-run optimizer diagnostics are present, use them.
    if isinstance(opt_diag, dict) and opt_diag:
        ns_total = float(opt_diag.get('ns_flops', 0.0) or 0.0)
        sign_total = float(opt_diag.get('sign_flops', 0.0) or 0.0)
        srank_total = float(opt_diag.get('srank_gate_flops', 0.0) or 0.0)
        if (ns_total + sign_total + srank_total) > 0.0:
            total = iterations * model_flops_per_iter + ns_total + sign_total + srank_total
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

    # Stable-rank gate overhead in adaptive mode (srank_alpha > 0).
    # From srank_wants_muon(update, alpha):
    # - frob_sq = sum(update^2): ~2mn
    # - 3x power iterations: each has (A @ v, A^T @ u): ~4mn
    # - sigma1_sq = sum((A @ v)^2): ~2mn
    # Total ~16mn per matrix, summed over Muon matrices per transformer layer.
    srank_check_flops_per_iter = n_layer * (16 * matvec_mn_sum_per_layer)

    if opt in ('adamw', 'lion'):
        total = iterations * model_flops_per_iter
        return total, total
    elif opt in ('muon', 'sign_muon', 'lion_muon'):
        if srank_alpha and srank_alpha > 0:
            if isinstance(opt_diag, dict) and opt_diag.get('srank_count', 0) > 0:
                p_muon = float(opt_diag['ns_count']) / float(opt_diag['srank_count'])
                extra = p_muon * ns_flops_per_iter + (1.0 - p_muon) * sign_flops_per_iter + srank_check_flops_per_iter
                total = iterations * (model_flops_per_iter + extra)
                return total, total

            # Exact interval without per-run branch telemetry:
            # low  => all layers choose sign after srank check
            # high => all layers choose Muon after srank check
            low = iterations * (model_flops_per_iter + srank_check_flops_per_iter + sign_flops_per_iter)
            high = iterations * (model_flops_per_iter + srank_check_flops_per_iter + ns_flops_per_iter)
            return low, high
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
    return key.replace('_nonesterov', '')


def _display_name_for_key(key):
    bkey = _base_key(key)
    if bkey in NAMES:
        return NAMES[bkey]
    m = re.match(r'^lionmuon_srank_a(.+)$', bkey)
    if m:
        return f"LiMuon sRank a={m.group(1)}"
    return bkey


def _iter_plot_runs(runs, ordered_keys):
    import colorsys
    srank_alphas = []
    for k, r in runs.items():
        if 'srank_alpha' in r and r['srank_alpha'] > 0:
            srank_alphas.append(r['srank_alpha'])
    srank_alphas = sorted(list(set(srank_alphas)))
    
    for nesterov_pass in (True, False):
        for key in ordered_keys:
            rkey = key if nesterov_pass else key + '_nonesterov'
            r = runs.get(rkey)
            if not r or r.get('nesterov', True) != nesterov_pass:
                continue
            label = _display_name_for_key(rkey)
            style = STYLE.get(label, {})
            if not style and 'sRank' in label:
                a = r.get('srank_alpha', 0.0)
                # Map alpha to a pure green gradient (lowest a = lighter, highest a = darker)
                try:
                    idx = srank_alphas.index(a)
                    ratio = idx / max(1, len(srank_alphas) - 1)
                    hue = 0.33  # stick to green
                    sat = 0.4 + 0.6 * ratio   # 0.4 to 1.0 (pale/light to deep/saturated)
                    val = 0.9 - 0.5 * ratio   # 0.9 to 0.4 (bright to dark)
                except:
                    hue, sat, val = 0.33, 0.8, 0.7
                rgb = colorsys.hsv_to_rgb(hue, sat, val)
                hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                style = {'color': hex_color, 'ls': '-', 'lw': 2.4, 'marker': 'D'}
            yield rkey, r, label, style, nesterov_pass



def get_tb_srank_metrics(exp_name):
    import os
    import math
    import copy
    try:
        from tensorboard.backend.event_processing import event_accumulator
        tb_dir = os.path.join("exps", "tb_logs", exp_name)
        if not os.path.exists(tb_dir):
            return None, None, None
        ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={'histograms': 10, 'scalars': 5000})
        ea.Reload()
        tags = ea.Tags()
        
        means, vars_, hist = [], [], None
        if "train/srank_mean" in tags.get('scalars', []):
            means = [(e.step, e.value) for e in ea.Scalars("train/srank_mean")]
        if "train/srank_var" in tags.get('scalars', []):
            vars_ = [(e.step, e.value) for e in ea.Scalars("train/srank_var")]
        if "train/srank_dist" in tags.get('histograms', []):
            hists = ea.Histograms("train/srank_dist")
            if hists:
                last_hist = hists[-1].histogram_value
                counts = list(last_hist.bucket)
                limits = list(last_hist.bucket_limit)
                hist = (limits, counts)
        return means, vars_, hist
    except Exception as e:
        print(f"Error reading TB info: {e}")
        pass
    return None, None, None

def smooth_curve(points, window=10):
    if not points: return []
    import numpy as np
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    if len(y) < window:
        window = max(1, len(y) // 2)
    y_padded = np.pad(y, (window//2, window-1-window//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones(window)/window, mode='valid')
    return list(zip(x, y_smooth))

def plot_adaptive_diagnostics(dataset_name, model_name, runs):
    """Diagnostics for adaptive LiMuon runs (Mean, Variance, Histogram)."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import colorsys
    
    adaptive_items = []
    srank_alphas = []
    for key, r in runs.items():
        a = float(r.get('srank_alpha', 0.0) or 0.0)
        if a > 0:
            adaptive_items.append((key, r, a))
            srank_alphas.append(a)

    if not adaptive_items:
        return
        
    srank_alphas = sorted(list(set(srank_alphas)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    for key, r, a in sorted(adaptive_items, key=lambda x: x[2]):
        label = _display_name_for_key(key)
        
        # Get exact color from the green spectrum
        try:
            idx = srank_alphas.index(a)
            ratio = idx / max(1, len(srank_alphas) - 1)
            hue = 0.33
            sat = 0.4 + 0.6 * ratio
            val = 0.9 - 0.5 * ratio
        except:
            hue, sat, val = 0.33, 0.8, 0.7
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        means, vars_, hist = get_tb_srank_metrics(r['exp'])
        
        if means:
            s_means = smooth_curve(means, window=50) # Very smooth
            x_m = [p[0] for p in s_means]
            y_m = [p[1] for p in s_means]
            ax1.plot(x_m, y_m, lw=2.2, label=label, color=color)
            
            if vars_:
                s_vars = smooth_curve(vars_, window=50)
                x_v = [p[0] for p in s_vars]
                y_v = [p[1] for p in s_vars]
                ax2.plot(x_v, y_v, lw=2.2, label=label, color=color)
                
                # Interpolate exactly for filling shaded region
                std_interp = np.interp(x_m, x_v, np.sqrt(np.maximum(y_v, 0)))
                ax1.fill_between(x_m, np.array(y_m) - std_interp, np.array(y_m) + std_interp, color=color, alpha=0.15)
                
            if hist:
                limits, counts = hist
                # Since buckets represent ranges, len(counts) == len(limits).
                # We can plot as step using 'post' where limits[0] is X[0], counts[0] is Y[0].
                mids = [(limits[i-1] + limits[i])/2.0 for i in range(1, len(limits))]
                if len(mids) == len(counts):
                    ax3.plot(mids, counts, marker='o', ls='-', lw=2.2, label=label, color=color)
                else: 
                     ax3.step(limits, counts, where='post', lw=2.2, label=label, color=color)
                     
    ax1.set_xlabel('Iteration', fontsize=13)
    ax1.set_ylabel('Mean Stable Rank', fontsize=13)
    ax1.set_title('Stable Rank Mean ± 1 Std (Smoothed Window=50)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.set_xlabel('Iteration', fontsize=13)
    ax2.set_ylabel('Variance of Stable Rank', fontsize=13)
    ax2.set_title('Stable Rank Variance', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Stable Rank Value', fontsize=13)
    ax3.set_ylabel('Matrix Count', fontsize=13)
    ax3.set_title('Final Gradient Histogram', fontsize=14)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f'{dataset_name}_{model_name}_adaptive_diag.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Adaptive diagnostics plot saved to {out_path}")





def plot_dataset(dataset_name, model_name):
    ds = DATASETS.get(dataset_name, {
        'wandb_project': None,
        'prefix': '',
        'title': dataset_name,
    })

    runs = {}
    if ds['wandb_project']:
        runs = get_runs(ds['wandb_project'], ds['prefix'])

    if not runs:
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

    runs = {k: v for k, v in runs.items() if not _is_excluded_run_key(k)}

    dynamic_order = ORDER + [k for k in sorted(runs.keys()) if _base_key(k) not in ORDER]

    model_title = 'Base' if model_name == 'base' else 'Llama'
    print(f"\n{ds['title']} ({model_title}) results (sorted by best val_loss):")
    print(f"  {'Optimizer':<30s} {'nesterov':>8s} {'best_loss':>10s} {'ppl':>8s} {'wall(min)':>10s} {'FLOPs':>14s}")
    print("  " + "-" * 86)
    for key, r in sorted(runs.items(), key=lambda x: x[1]['best_loss']):
        label = _display_name_for_key(key)
        nes_str = 'yes' if r.get('nesterov', True) else 'no'
        wall_min = r['runtime'] / 60 if (r['runtime'] and r['runtime'] > 0) else 0
        low, high = compute_flops(
            12, 768, 512, 32,
            r.get('iterations', 64000),
            r['opt'], K=r['K'], ns_steps=r['ns_steps'],
            srank_alpha=r.get('srank_alpha', 0.0),
            opt_diag=r.get('opt_diag'),
        )
        flops_str = f"{low/1e18:.3f}" if abs(high - low) < 1e-9 else f"{low/1e18:.3f}-{high/1e18:.3f}"
        print(f"  {label:<30s} {nes_str:>8s} {r['best_loss']:>10.4f} {math.exp(r['best_loss']):>8.1f} {wall_min:>10.1f}  {flops_str:>13s}e18")

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 7))

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

    for rkey, r, label, s, nesterov_pass in _iter_plot_runs(runs, dynamic_order):
        color = s.get('color', '#333')
        alpha = 1.0 if nesterov_pass else 0.45
        lw = s.get('lw', 2) if nesterov_pass else s.get('lw', 2) * 0.8
        ls = s.get('ls', '-') if nesterov_pass else (0, (3, 2))
        legend_label = f"{label} ({r['best_loss']:.3f})" if nesterov_pass else f"{label}* no-Nes ({r['best_loss']:.3f})"
        ax1.plot(r['iters'], r['losses'], label=legend_label,
                 color=color, ls=ls, lw=lw, alpha=alpha)

    ax1.set_yscale('log')
    y_max = 4 if dataset_name == 'wikitext' else 5
    ax1.set_ylim(None, y_max)
    ax1.set_xlabel('Iteration', fontsize=16)
    ax1.set_ylabel('Val Loss', fontsize=16)
    ax1.set_title(f'Val Loss vs Iteration — 124M, {ds["title"]} ({model_title})', fontsize=16)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=13)

    # Plot 2: FLOPs scatter
    flops_data = []
    for rkey, r, label, s, nesterov_pass in _iter_plot_runs(runs, dynamic_order):
        color = s.get('color', '#333')
        alpha = 1.0 if nesterov_pass else 0.45
        marker = s.get('marker', 'o') if nesterov_pass else 'x'
        flops_lo, flops_hi = compute_flops(n_layer, n_embd, seq_len, batch_size,
                                           r.get('iterations', 64000),
                                           r['opt'], K=r['K'], ns_steps=r['ns_steps'],
                                           srank_alpha=r.get('srank_alpha', 0.0),
                                           opt_diag=r.get('opt_diag'))
        # Keep scatter simple: plot the midpoint for adaptive FLOPs ranges.
        flops = 0.5 * (flops_lo + flops_hi)
        ec = {} if marker == 'x' else {'edgecolors': 'black', 'linewidths': 0.5}
        ax3.scatter(flops, r['best_loss'], s=180, zorder=5,
                    color=color, marker=marker, alpha=alpha, **ec)
        flops_data.append((label, flops, r['best_loss'], nesterov_pass))

    for label, x, y, nes in flops_data:
        smart_annotate(ax3, label, x, y, nesterov=nes)

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
    out_path = os.path.join(RESULTS_DIR, f'{dataset_name}_{model_name}_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")

    # Additional simple diagnostics for adaptive LiMuon runs.
    plot_adaptive_diagnostics(dataset_name, model_name, runs)


def discover_available_dataset_model_pairs():
    """Discover (dataset, model) pairs from local summary files in exps/."""
    local_exps = os.path.join(LOCAL_BASE_DIR, 'exps')
    if not os.path.isdir(local_exps):
        return []

    pairs = set()
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
    if len(sys.argv) == 1:
        pairs = discover_available_dataset_model_pairs()
        if not pairs:
            print("No local results found in exps/ to plot.")
            sys.exit(0)
        print("Auto-discovered result groups:")
        for ds, model_name in pairs:
            print(f"  - {ds} {model_name}")
            plot_dataset(ds, model_name)
        return

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <dataset|all> <base|llama>")
        print(f"   or: {sys.argv[0]}   # auto-plot all available local results")
        sys.exit(1)

    target = sys.argv[1]
    model_name = sys.argv[2].lower()
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Choose from: {', '.join(sorted(MODELS))}")
        sys.exit(1)

    if target == 'all':
        for ds in DATASETS:
            plot_dataset(ds, model_name)
    else:
        # Allow unknown datasets for local-only plotting via automatic prefix.
        plot_dataset(target, model_name)


if __name__ == '__main__':
    main()
