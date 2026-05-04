"""Multi-panel best-val-loss vs total-FLOPs scatter for the main text.

Produces two figures from one shared codepath:
  - combined_flops_loss.png            (2x3, 124M only, all six dataset/arch combos)
  - fineweb_base_355m_720m_pareto.png  (1x2, FineWeb/GPT-2 at 355M and 720M)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plot_results import (  # noqa: E402
    LOCAL_BASE_DIR, ALLOWED_BASE_KEYS, ORDER, compute_flops,
    get_local_runs, get_local_prefix, get_legacy_local_prefix,
    _is_excluded_run_key, _normalize_run_keys, _base_key, _iter_plot_runs,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Per-scale dims (matched to actual runs) and where to look for them.
DIMS = {
    '124m': dict(n_layer=12, n_embd=768,  seq_len=512,  batch_size=32),
    '355m': dict(n_layer=24, n_embd=1024, seq_len=1024, batch_size=512),
    '720m': dict(n_layer=12, n_embd=2048, seq_len=512,  batch_size=1984),
}
EXP_SUBDIR = {'124m': 'exps', '355m': 'exps_355m', '720m': 'exps_720m'}
EXPLICIT_PREFIX = {'355m': 'fw_base_355m_', '720m': 'fw_base_720m_'}

DATASET_TITLES = {'fineweb': 'FineWeb', 'slimpajama': 'SlimPajama', 'wikitext': 'WikiText-103'}
ARCH_TITLES = {'base': 'GPT-2', 'llama': 'LLaMA'}
SCALE_TITLES = {
    '355m': r'355M ($1\times$ Chinchilla, $\sim$23 TPP)',
    '720m': r'720M ($1/4$ Chinchilla, $\sim$5 TPP)',
}


# Figure specifications.
FIGURES = [
    {
        'out': 'combined_flops_loss.png',
        'figsize': (13, 6.5),
        'grid': (2, 3),
        'shared_legend': True,
        'marker_size': 80, 'title_fs': 12, 'label_fs': 11, 'tick_fs': 9, 'legend_fs': 9,
        'panels': [
            ('124m', 'fineweb',    'base'),
            ('124m', 'slimpajama', 'base'),
            ('124m', 'wikitext',   'base'),
            ('124m', 'fineweb',    'llama'),
            ('124m', 'slimpajama', 'llama'),
            ('124m', 'wikitext',   'llama'),
        ],
    },
    {
        'out': 'fineweb_base_355m_720m_pareto.png',
        'figsize': (14, 5),
        'grid': (1, 2),
        'shared_legend': False,
        'marker_size': 180, 'title_fs': 15, 'label_fs': 15, 'tick_fs': 12, 'legend_fs': 10,
        'panels': [
            ('355m', 'fineweb', 'base'),
            ('720m', 'fineweb', 'base'),
        ],
    },
]


def _panel_title(scale, dataset, arch):
    if scale in SCALE_TITLES:
        return SCALE_TITLES[scale]
    return f'{DATASET_TITLES[dataset]} / {ARCH_TITLES[arch]}'


def _load_runs(scale, dataset, arch):
    exps_dir = os.path.join(LOCAL_BASE_DIR, EXP_SUBDIR[scale])
    prefix = EXPLICIT_PREFIX.get(scale) or get_local_prefix(dataset, arch)
    runs = get_local_runs(exps_dir, prefix, model_name=arch)
    if not runs and scale == '124m':
        runs = get_local_runs(exps_dir, get_legacy_local_prefix(dataset), model_name=arch)
    runs = {k: v for k, v in runs.items()
            if not _is_excluded_run_key(k) and _base_key(k) in ALLOWED_BASE_KEYS}
    return _normalize_run_keys(runs)


def _draw_panel(ax, panel, spec, collected):
    scale, dataset, arch = panel
    runs = _load_runs(scale, dataset, arch)
    if not runs:
        ax.text(0.5, 0.5, f'No runs: {scale} {dataset} {arch}',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        return
    dims = DIMS[scale]
    for _, r, label, s, nesterov_pass in _iter_plot_runs(runs, ORDER):
        color = s.get('color', '#333')
        marker = s.get('marker', 'o') if nesterov_pass else 'x'
        edge_color = 'black' if marker != 'x' else color
        edge_width = 0.5 if marker != 'x' else 1.0
        flops_lo, flops_hi = compute_flops(
            dims['n_layer'], dims['n_embd'], dims['seq_len'], dims['batch_size'],
            r.get('iterations', 64000),
            r['opt'], K=r['K'], ns_steps=r['ns_steps'],
            opt_diag=r.get('opt_diag'),
        )
        flops = 0.5 * (flops_lo + flops_hi)
        ax.scatter(flops, r['best_loss'], s=spec['marker_size'], zorder=5,
                   color=color, marker=marker, alpha=0.9,
                   edgecolors=edge_color, linewidths=edge_width)
        legend_text = f"{label}{'' if nesterov_pass else '*'}"
        if legend_text not in collected:
            collected[legend_text] = Line2D(
                [0], [0], marker=marker, linestyle='None',
                markerfacecolor=color, markeredgecolor=edge_color,
                markeredgewidth=edge_width, alpha=0.9, markersize=8,
                label=legend_text,
            )

    ax.set_title(_panel_title(scale, dataset, arch), fontsize=spec['title_fs'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=spec['tick_fs'])
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))


def _build_figure(spec):
    nrows, ncols = spec['grid']
    fig, axes = plt.subplots(nrows, ncols, figsize=spec['figsize'])
    axes = list(axes.flat) if nrows * ncols > 1 else [axes]

    collected = {}
    for ax, panel in zip(axes, spec['panels']):
        _draw_panel(ax, panel, spec, collected)

    # Axis labels on outer panels only.
    for i, ax in enumerate(axes):
        r, c = divmod(i, ncols)
        if c == 0:
            ax.set_ylabel('Best Val Loss', fontsize=spec['label_fs'])
        if r == nrows - 1:
            ax.set_xlabel('Total Training FLOPs', fontsize=spec['label_fs'])

    handles = list(collected.values())
    if spec['shared_legend']:
        fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.0, 0.5),
                   fontsize=spec['legend_fs'], title='Optimizer',
                   title_fontsize=spec['legend_fs'] + 1, frameon=True)
        plt.tight_layout(rect=[0, 0, 0.86, 1])
    else:
        for ax in axes:
            ax.legend(handles=handles, loc='upper right',
                      fontsize=spec['legend_fs'], title='Run',
                      title_fontsize=spec['legend_fs'], frameon=True)
        plt.tight_layout()

    project_root = os.path.dirname(os.path.normpath(LOCAL_BASE_DIR))
    out_dir = os.path.join(project_root, 'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, spec['out'])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    for spec in FIGURES:
        _build_figure(spec)


if __name__ == '__main__':
    main()
