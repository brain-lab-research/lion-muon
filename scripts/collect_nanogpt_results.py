#!/usr/bin/env python3
"""Collect optimizer comparison results from WandB.

For each optimizer, finds the best LR run based on final val_acc,
then exports final metrics to CSV, creates LaTeX tables, and plots training loss curves.
Works with both NanoGPT and Llama experiments.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def collect_results(project_name, entity=None, output_dir="results", model_type="nanogpt"):
    """Collect results from WandB project.
    
    Args:
        project_name: WandB project name
        entity: WandB entity (username or team)
        output_dir: Output directory for results
        model_type: Model type ('nanogpt' or 'llama')
    """
    api = wandb.Api()
    
    # Get runs from project
    if entity:
        runs = api.runs(f"{entity}/{project_name}")
    else:
        runs = api.runs(project_name)
    
    # Filter for comparison runs
    optimizers = ["adamw", "muon", "rmsspectral", "adamuon", "rmsspectral-sania"]
    
    # Group runs by optimizer
    runs_by_opt = {opt: [] for opt in optimizers}
    
    print(f"Fetching runs from {project_name}...")
    print(f"Total runs found: {len(list(runs))}")
    
    # Reload runs iterator
    if entity:
        runs = api.runs(f"{entity}/{project_name}")
    else:
        runs = api.runs(project_name)
    
    for run in runs:
        print(f"Checking run: {run.name}")
        matched = False
        
        # Extract opt value from run name (format: opt-<optimizer>__)
        import re
        opt_match = re.search(r'opt-([a-zA-Z0-9\-]+)__', run.name)
        if opt_match:
            opt_from_name = opt_match.group(1)
            if opt_from_name in optimizers:
                runs_by_opt[opt_from_name].append(run)
                print(f"  -> Matched to {opt_from_name}")
                matched = True
        
        if not matched:
            # Fallback: Check if run is a comparison run for the specified model type
            if "compare_" in run.name and model_type in run.name:
                for opt in optimizers:
                    if opt.replace("-", "_") in run.name:
                        runs_by_opt[opt].append(run)
                        print(f"  -> Matched to {opt}")
                        break
    
    # Find best run per optimizer based on final val_acc
    best_runs = {}
    for opt, opt_runs in runs_by_opt.items():
        if not opt_runs:
            print(f"Warning: No runs found for {opt}")
            continue
        
        best_run = None
        best_val_acc = -float('inf')
        
        for run in opt_runs:
            # Get final val_acc
            summary = run.summary
            if 'val/acc' in summary:
                val_acc = summary['val/acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_run = run
        
        if best_run:
            best_runs[opt] = best_run
            print(f"{opt}: best val_acc = {best_val_acc:.4f} (run: {best_run.name})")
    
    # Collect final metrics for CSV
    results_data = []
    for opt, run in best_runs.items():
        summary = run.summary
        config = run.config
        
        results_data.append({
            'optimizer': opt,
            'run_name': run.name,
            'lr': config.get('lr', 'N/A'),
            'final_train_loss': summary.get('train/loss', None),
            'final_val_loss': summary.get('val/loss', None),
            'final_val_acc': summary.get('val/acc', None),
            'iterations': summary.get('iter', config.get('iterations', 'N/A')),
        })
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results_data)
    csv_path = output_path / f"{model_type}_best_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))
    
    # Create LaTeX table
    latex_path = output_path / f"{model_type}_results.tex"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Optimizer & LR & Train Loss & Val Loss & Val Acc \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            opt = row['optimizer']
            lr = f"{float(row['lr']):.0e}" if row['lr'] != 'N/A' else 'N/A'
            train_loss = f"{row['final_train_loss']:.4f}" if pd.notna(row['final_train_loss']) else 'N/A'
            val_loss = f"{row['final_val_loss']:.4f}" if pd.notna(row['final_val_loss']) else 'N/A'
            val_acc = f"{row['final_val_acc']:.4f}" if pd.notna(row['final_val_acc']) else 'N/A'
            f.write(f"{opt} & {lr} & {train_loss} & {val_loss} & {val_acc} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{model_type.upper()} Optimizer Comparison Results}}\n")
        f.write(f"\\label{{tab:{model_type}_results}}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_path}")
    
    return df, best_runs


def plot_combined_losses(nanogpt_runs, llama_runs, output_dir="results"):
    """Plot training loss curves for both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        'adamw': '#1f77b4',
        'muon': '#ff7f0e',
        'rmsspectral': '#2ca02c',
        'adamuon': '#d62728',
        'rmsspectral-sania': '#9467bd',
    }
    
    # Plot NanoGPT
    ax = axes[0]
    for opt, run in nanogpt_runs.items():
        history = run.history(keys=['iter', 'train/loss'], samples=10000)
        if not history.empty and 'train/loss' in history.columns:
            history = history.dropna(subset=['train/loss'])
            ax.plot(
                history['iter'],
                history['train/loss'],
                label=opt,
                color=colors.get(opt, None),
                linewidth=2,
                alpha=0.8
            )
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('NanoGPT', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot Llama
    ax = axes[1]
    for opt, run in llama_runs.items():
        history = run.history(keys=['iter', 'train/loss'], samples=10000)
        if not history.empty and 'train/loss' in history.columns:
            history = history.dropna(subset=['train/loss'])
            ax.plot(
                history['iter'],
                history['train/loss'],
                label=opt,
                color=colors.get(opt, None),
                linewidth=2,
                alpha=0.8
            )
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Llama 124M', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plot_path = output_path / "combined_train_loss.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nCombined plot saved to {plot_path}")
    plt.close()


def create_combined_latex_table(nanogpt_df, llama_df, output_dir="results"):
    """Create a combined LaTeX table for both models."""
    output_path = Path(output_dir)
    latex_path = output_path / "combined_results.tex"
    
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lccccccccc}\n")
        f.write("\\toprule\n")
        f.write("& \\multicolumn{4}{c}{NanoGPT} & \\multicolumn{4}{c}{Llama 124M} \\\\\n")
        f.write("\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\n")
        f.write("Optimizer & LR & Train Loss & Val Loss & Val Acc & LR & Train Loss & Val Loss & Val Acc \\\\\n")
        f.write("\\midrule\n")
        
        # Merge dataframes on optimizer
        optimizers = ["adamw", "muon", "rmsspectral", "adamuon", "rmsspectral-sania"]
        
        for opt in optimizers:
            nano_row = nanogpt_df[nanogpt_df['optimizer'] == opt]
            llama_row = llama_df[llama_df['optimizer'] == opt]
            
            # NanoGPT values
            if not nano_row.empty:
                nano_lr = f"{float(nano_row.iloc[0]['lr']):.0e}" if nano_row.iloc[0]['lr'] != 'N/A' else 'N/A'
                nano_train = f"{nano_row.iloc[0]['final_train_loss']:.4f}" if pd.notna(nano_row.iloc[0]['final_train_loss']) else 'N/A'
                nano_val = f"{nano_row.iloc[0]['final_val_loss']:.4f}" if pd.notna(nano_row.iloc[0]['final_val_loss']) else 'N/A'
                nano_acc = f"{nano_row.iloc[0]['final_val_acc']:.4f}" if pd.notna(nano_row.iloc[0]['final_val_acc']) else 'N/A'
            else:
                nano_lr = nano_train = nano_val = nano_acc = 'N/A'
            
            # Llama values
            if not llama_row.empty:
                llama_lr = f"{float(llama_row.iloc[0]['lr']):.0e}" if llama_row.iloc[0]['lr'] != 'N/A' else 'N/A'
                llama_train = f"{llama_row.iloc[0]['final_train_loss']:.4f}" if pd.notna(llama_row.iloc[0]['final_train_loss']) else 'N/A'
                llama_val = f"{llama_row.iloc[0]['final_val_loss']:.4f}" if pd.notna(llama_row.iloc[0]['final_val_loss']) else 'N/A'
                llama_acc = f"{llama_row.iloc[0]['final_val_acc']:.4f}" if pd.notna(llama_row.iloc[0]['final_val_acc']) else 'N/A'
            else:
                llama_lr = llama_train = llama_val = llama_acc = 'N/A'
            
            f.write(f"{opt} & {nano_lr} & {nano_train} & {nano_val} & {nano_acc} & ")
            f.write(f"{llama_lr} & {llama_train} & {llama_val} & {llama_acc} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Optimizer Comparison Results for NanoGPT and Llama 124M}\n")
        f.write("\\label{tab:combined_results}\n")
        f.write("\\end{table}\n")
    
    print(f"Combined LaTeX table saved to {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect optimizer comparison results from WandB")
    parser.add_argument(
        "--nanogpt_project",
        type=str,
        default="llm-baselines-nanogpt",
        help="WandB project name for NanoGPT experiments"
    )
    parser.add_argument(
        "--llama_project",
        type=str,
        default="llm-baselines-llama",
        help="WandB project name for Llama experiments"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity (username or team)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Collect results for both models
    print("=" * 60)
    print("Collecting NanoGPT results...")
    print("=" * 60)
    nanogpt_df, nanogpt_runs = collect_results(args.nanogpt_project, args.entity, args.output, "nanogpt")
    
    print("\n" + "=" * 60)
    print("Collecting Llama results...")
    print("=" * 60)
    llama_df, llama_runs = collect_results(args.llama_project, args.entity, args.output, "llama")
    
    # Create combined LaTeX table
    print("\n" + "=" * 60)
    print("Creating combined LaTeX table...")
    print("=" * 60)
    create_combined_latex_table(nanogpt_df, llama_df, args.output)
    
    # Plot combined training loss curves
    print("\n" + "=" * 60)
    print("Generating combined training loss plot...")
    print("=" * 60)
    plot_combined_losses(nanogpt_runs, llama_runs, args.output)
    
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
