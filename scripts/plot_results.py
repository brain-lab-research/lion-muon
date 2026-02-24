"""Plot val_loss curves and FLOPs-vs-perplexity for sign-muon-tuning experiments."""

import re
import os
import glob
import yaml
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(logfile):
    """Extract (iter, val_loss) pairs from output.log."""
    pattern = re.compile(r'>Eval: Iter=(\d+).*?val_loss=([\d.]+)')
    iters, losses = [], []
    with open(logfile) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return iters, losses


def get_runs(project):
    """Get all completed runs from a wandb project."""
    runs = []
    for run_dir in sorted(glob.glob('/home/arman.bolatov/Desktop/llm-baselines/wandb/run-*/')):
        config_f = os.path.join(run_dir, 'files/config.yaml')
        summary_f = os.path.join(run_dir, 'files/wandb-summary.json')
        log_f = os.path.join(run_dir, 'files/output.log')
        if not all(os.path.exists(f) for f in [config_f, summary_f, log_f]):
            continue
        cfg = yaml.safe_load(open(config_f))
        s = json.load(open(summary_f))
        if cfg.get('wandb_project', {}).get('value', '') != project:
            continue
        steps = s.get('_step', 0)
        if steps < 20:
            continue

        val = s.get('val/loss', None)
        if isinstance(val, dict):
            val = val.get('min', None)

        exp = cfg.get('experiment_name', {}).get('value', 'unknown')
        opt = cfg.get('opt', {}).get('value', '')
        runs.append({
            'name': exp, 'opt': opt, 'val_loss': val,
            'log': log_f, 'cfg': cfg, 'summary': s,
        })
    return runs


def compute_flops(n_layer, n_embd, seq_len, batch_size, iterations, opt_type, K=1, ns_steps=6):
    """Compute total training FLOPs."""
    # Model params (approximate)
    n_params = n_layer * (4 * n_embd**2 + 8 * n_embd**2)  # attn + mlp
    # Forward + backward: ~6N per token
    model_flops_per_iter = 6 * n_params * seq_len * batch_size

    # NS FLOPs per Muon step (all 2D params)
    # Per layer: c_attn(d,3d), c_proj(d,d), c_fc(d,4d), mlp_proj(4d,d)
    d = n_embd
    ns_flops_per_layer = 0
    for (m, n) in [(d, 3*d), (d, d), (d, 4*d), (4*d, d)]:
        dmin, dmax = min(m, n), max(m, n)
        # Per NS step: ~2*dmin^2*(3*dmax + dmin) FLOPs
        ns_flops_per_layer += ns_steps * 2 * dmin**2 * (3*dmax + dmin)
    ns_flops_per_iter = n_layer * ns_flops_per_layer

    if opt_type == 'adamw':
        total = iterations * model_flops_per_iter
    elif opt_type == 'muon':
        total = iterations * (model_flops_per_iter + ns_flops_per_iter)
    elif opt_type == 'sign_muon':
        # NS only every K steps
        total = iterations * (model_flops_per_iter + ns_flops_per_iter / K)
    else:
        total = iterations * model_flops_per_iter

    return total


def main():
    runs = get_runs('sign-muon-tuning')
    if not runs:
        print("No runs found for project 'sign-muon-tuning'")
        return

    # Find best run per method
    best = {}
    for r in runs:
        if r['val_loss'] is None:
            continue
        name = r['name']
        if name.startswith('adam_'):
            key = 'AdamW'
        elif name.startswith('muon_'):
            key = 'Muon'
        elif name.startswith('signmuon_'):
            key = 'SignMuon K=5'
        else:
            continue
        if key not in best or r['val_loss'] < best[key]['val_loss']:
            best[key] = r

    print("Best runs:")
    for key, r in sorted(best.items()):
        print(f"  {key}: {r['name']} val_loss={r['val_loss']:.4f}")

    # Print all results sorted
    print("\nAll results:")
    for r in sorted(runs, key=lambda x: x['val_loss'] or 999):
        print(f"  {r['name']:<35s} val_loss={r['val_loss']}")

    if not best:
        return

    # Plot 1: val_loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {'AdamW': 'C0', 'Muon': 'C2', 'SignMuon K=5': 'C3'}
    for key in ['AdamW', 'Muon', 'SignMuon K=5']:
        if key not in best:
            continue
        r = best[key]
        iters, losses = parse_log(r['log'])
        label = f"{key} ({r['name']})"
        ax1.plot(iters, losses, label=label, linewidth=2, color=colors[key])

    ax1.set_xlabel('Iteration', fontsize=13)
    ax1.set_ylabel('Val Loss', fontsize=13)
    ax1.set_title('Val Loss Curves (best LR per method)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: FLOPs vs best perplexity
    n_layer, n_embd, seq_len, batch_size = 6, 384, 256, 32
    iterations = 3000

    flops_data = []
    for key in ['AdamW', 'Muon', 'SignMuon K=5']:
        if key not in best:
            continue
        r = best[key]
        if key == 'AdamW':
            flops = compute_flops(n_layer, n_embd, seq_len, batch_size, iterations, 'adamw')
        elif key == 'Muon':
            flops = compute_flops(n_layer, n_embd, seq_len, batch_size, iterations, 'muon')
        else:
            flops = compute_flops(n_layer, n_embd, seq_len, batch_size, iterations, 'sign_muon', K=5)
        ppl = math.exp(r['val_loss'])
        flops_data.append((key, flops, ppl, r['val_loss']))

    for key, flops, ppl, vl in flops_data:
        ax2.scatter(flops, ppl, s=150, label=f"{key} (loss={vl:.3f})", color=colors[key], zorder=5)

    ax2.set_xlabel('Total FLOPs', fontsize=13)
    ax2.set_ylabel('Perplexity', fontsize=13)
    ax2.set_title('FLOPs vs Perplexity', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/arman.bolatov/Desktop/llm-baselines/tuning_results.png', dpi=150)
    print(f"\nPlot saved to tuning_results.png")


if __name__ == '__main__':
    main()
