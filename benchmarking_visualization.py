"""
Benchmarking Visualisation Script
===================================
Reads two Excel files and produces:
  1. One boxplot figure per varying parameter  (5 figures)
  2. A barplot of average combined score per parameter configuration
  3. A barplot comparing best runs across tools  (when OTHER_TOOL_FILE is set)

Run:
    python benchmarking_visualization.py

Edit the CONFIG section below to match your setup.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# File paths (relative to this script, or absolute)
AGGREGATE_FILE   = "/home/mnxsybi_lumc/spatial-multiomics-benchmarking/benchmarks/results/many_parameters_24062026/results_metrics.xlsx"   # contains Continuity Score, Marker score, etc.
METRICS_FILE = "/home/mnxsybi_lumc/spatial-multiomics-benchmarking/benchmarks/results/many_parameters_24062026/results.xlsx"           # contains PCC, CosineSim, ARI, etc.

# Columns used to join the two files (parameter identifiers + run index)
JOIN_COLS = [
    'n_latent', 'hidden_stacks', 'lr', 'max_epoch',
    'n_top_genes', 'n_top_metabolites', 'kl_weight', 'mmd_weight',
    'iteration_number',
]

# Score columns from AGGREGATE_FILE to include in the boxplots
SCORE_COLS = ['Biological Conservation', 'Marker score', 'Continuity Score', 'Reconstruction Accuracy']

# Extra metric columns from METRICS_FILE to include in the boxplots
EXTRA_METRIC_COLS = ['ARI', 'NMI', 'runtime', 'peak_memory']

# All metrics shown in the boxplots (order determines subplot order)
ALL_METRICS = ['Combined Score'] + SCORE_COLS + EXTRA_METRIC_COLS

# Parameters that are varied one at a time; all others held at BASELINE
VARIED_PARAMS = ['n_latent', 'n_top_genes', 'n_top_metabolites', 'kl_weight', 'mmd_weight']

# Baseline parameter values (the reference / "control" condition)
BASELINE = dict(
    n_latent=10,
    n_top_genes=2000,
    n_top_metabolites=800,
    kl_weight=1.0,
    mmd_weight=1.0,
)

# ── Combined score ────────────────────────────────────────────────────────────
# Computed per row as the mean of these four metrics (all in METRICS_FILE).
# Used to rank configurations and select the best run for the tool comparison.
COMBINED_SCORE_COLS = [
    'Continuity Score', 'Marker score',
    'Biological Conservation', 'Reconstruction Accuracy',
]
COMBINED_SCORE_COL = 'Combined Score'

# ── Tool comparison (Plot 3) ──────────────────────────────────────────────────
# Set OTHER_TOOL_FILE to the path of the other tool's results file to enable.
OTHER_TOOL_FILE = None    # e.g. "other_tool_results.xlsx"
OTHER_TOOL_NAME = None    # e.g. "DestVI"
THIS_TOOL_NAME  = "MOMO"

# Metrics shown in the tool comparison barplot
COMPARISON_METRICS = SCORE_COLS + EXTRA_METRIC_COLS

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "visualizations_benchmarking"
OUTPUT_DIR.mkdir(exist_ok=True)
DPI = 150

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE  = ['#2C7BB6', '#D7191C', '#1A9641', '#FDAE61', '#9B59B6', '#E67E22']
GRID_CLR = '#EBEBEB'

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': GRID_CLR,
    'grid.linewidth': 0.7,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(metrics_file: str = METRICS_FILE,
              aggregate_file: str = AGGREGATE_FILE) -> pd.DataFrame:
    """
    Merge the two result files on JOIN_COLS.
    METRICS_FILE  → Continuity Score, Marker score, Biological Conservation,
                    Reconstruction Accuracy, and EXTRA_METRIC_COLS
    AGGREGATE_FILE → PCC, CosineSim, ARI, NMI, etc. (SCORE_COLS)
    Both files share JOIN_COLS but also both contain runtime/memory columns;
    we only pull SCORE_COLS from AGGREGATE_FILE to avoid _x/_y collisions.
    """
    metrics = pd.read_excel(metrics_file)
    agg     = pd.read_excel(aggregate_file)
 
    # Only bring score columns + join keys from the aggregate file
    merged = metrics.merge(agg[JOIN_COLS + SCORE_COLS], on=JOIN_COLS, how='left')
 
    # Compute combined score per row
    merged[COMBINED_SCORE_COL] = merged[COMBINED_SCORE_COLS].mean(axis=1)
    return merged

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – BOXPLOTS: effect of each parameter on all metrics
# ══════════════════════════════════════════════════════════════════════════════

def make_boxplot_figure(df: pd.DataFrame, sweep_param: str) -> plt.Figure:
    """One figure per parameter sweep; subplots = one per metric."""
    mask = pd.Series(True, index=df.index)
    for k, v in BASELINE.items():
        if k != sweep_param:
            mask &= (df[k] == v)
    sub = df[mask].copy()
    order = [str(v) for v in sorted(sub[sweep_param].unique(), key=float)]
    sub['_g'] = sub[sweep_param].astype(str)

    n_cols = 5
    n_rows = int(np.ceil(len(ALL_METRICS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.6, n_rows * 4.2),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(order)}

    for ax, metric in zip(axes, ALL_METRICS):
        data = [sub.loc[sub['_g'] == g, metric].dropna().values for g in order]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color='white', linewidth=2.2, solid_capstyle='round'),
            whiskerprops=dict(linewidth=1.1, color='#555'),
            capprops=dict(linewidth=1.1, color='#555'),
            flierprops=dict(marker='o', markersize=3.5, alpha=0.45, markeredgewidth=0.5),
            boxprops=dict(linewidth=1.1),
        )
        for patch, g in zip(bp['boxes'], order):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.80)
        for flier, g in zip(bp['fliers'], order):
            flier.set(markerfacecolor=colors[g], markeredgecolor=colors[g])

        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, fontsize=8.5)
        ax.set_title(metric, fontsize=9.5, fontweight='bold', pad=4)
        ax.set_ylabel('Value', fontsize=8)
        ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=8)

    for ax in axes[len(ALL_METRICS):]:
        ax.set_visible(False)

    fig.suptitle(f'Effect of  {sweep_param}  on benchmarking metrics - SpatialMETA',
                 fontsize=13, fontweight='bold', y=1.03)
    legend_handles = [mpatches.Patch(facecolor=colors[g], alpha=0.80, label=g) for g in order]
    fig.legend(handles=legend_handles, title=sweep_param, loc='lower center',
               ncol=len(order), bbox_to_anchor=(0.5, -0.06),
               frameon=False, fontsize=9, title_fontsize=9.5)
    return fig



# ══════════════════════════════════════════════════════════════════════════════
# LR × MAX_EPOCH SWEEP – data loading + boxplots
# ══════════════════════════════════════════════════════════════════════════════

def load_lr_epoch_data(metrics_file: str, aggregate_file: str) -> pd.DataFrame:
    """
    Load the lr × max_epoch sweep files.
    Same two-file structure as the main experiment:
      metrics_file   → Continuity Score, Marker score, Biological Conservation,
                       Reconstruction Accuracy  (+ JOIN_COLS_LR)
      aggregate_file → PCC, CosineSim, ARI, etc.  (+ JOIN_COLS_LR)
    """
    join_cols = ['n_latent', 'hidden_stacks', 'lr', 'max_epoch', 'iteration_number']

    metrics = pd.read_excel(metrics_file)
    agg     = pd.read_excel(aggregate_file)

    merged = metrics.merge(agg[join_cols + SCORE_COLS], on=join_cols, how='left')
    merged[COMBINED_SCORE_COL] = merged[COMBINED_SCORE_COLS].mean(axis=1)
    return merged


def make_lr_epoch_boxplots(df: pd.DataFrame) -> plt.Figure:
    """
    Boxplots of all metrics for every lr × max_epoch combination.
    Each group on the x-axis is labelled "lr=<value> / epoch=<value>".
    Groups are arranged first by lr, then by max_epoch within each lr.
    """
    df = df.copy()
    df['_combo'] = df.apply(
        lambda r: f"lr={r['lr']:.4g}, ep={int(r['max_epoch'])}", axis=1
    )

    # Order: sort by lr first, then max_epoch
    order = (
        df[['lr', 'max_epoch', '_combo']]
        .drop_duplicates()
        .sort_values(['lr', 'max_epoch'])['_combo']
        .tolist()
    )

    n_cols = 3
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5.0, n_rows * 4.2),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    # Colour by lr value so groups with same lr share a hue
    lr_values = sorted(df['lr'].unique())
    lr_palette = ['#2C7BB6', '#D7191C', '#1A9641', '#9B59B6']
    lr_color   = {lr: lr_palette[i % len(lr_palette)] for i, lr in enumerate(lr_values)}

    # Lighter shade per max_epoch within each lr
    epoch_values = sorted(df['max_epoch'].unique())
    alphas = [0.55, 0.75, 0.95]  # lightest → darkest for increasing epochs
    epoch_alpha = {ep: alphas[i % len(alphas)] for i, ep in enumerate(epoch_values)}

    combo_color = {}
    combo_alpha = {}
    for _, row in df[['lr', 'max_epoch', '_combo']].drop_duplicates().iterrows():
        combo_color[row['_combo']] = lr_color[row['lr']]
        combo_alpha[row['_combo']] = epoch_alpha[row['max_epoch']]

    for ax, metric in zip(axes, ALL_METRICS):
        data = [df.loc[df['_combo'] == g, metric].dropna().values for g in order]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color='white', linewidth=2.0, solid_capstyle='round'),
            whiskerprops=dict(linewidth=1.1, color='#555'),
            capprops=dict(linewidth=1.1, color='#555'),
            flierprops=dict(marker='o', markersize=3.5, alpha=0.45, markeredgewidth=0.5),
            boxprops=dict(linewidth=1.1),
        )
        for patch, g in zip(bp['boxes'], order):
            patch.set_facecolor(combo_color[g])
            patch.set_alpha(combo_alpha[g])
        for flier, g in zip(bp['fliers'], order):
            flier.set(markerfacecolor=combo_color[g], markeredgecolor=combo_color[g])

        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, fontsize=7.5, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_title(metric, fontsize=9.5, fontweight='bold', pad=4)
        ax.set_ylabel('Value', fontsize=8)
        ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=8)

    for ax in axes[len(ALL_METRICS):]:
        ax.set_visible(False)

    fig.suptitle('Effect of learning rate and training epochs on benchmarking metrics - SpatialMETA',
                 fontsize=13, fontweight='bold', y=1.03)

    # Legend: one patch per lr (colour) × epoch (shade)
    legend_handles = []
    for lr in lr_values:
        for ep in epoch_values:
            combo_label = f"lr={lr:.4g} / epoch={int(ep)}"
            legend_handles.append(
                mpatches.Patch(
                    facecolor=lr_color[lr],
                    alpha=epoch_alpha[ep],
                    label=combo_label,
                )
            )
    fig.legend(handles=legend_handles, title='lr / epoch',
               loc='lower center',
               ncol=len(epoch_values),
               bbox_to_anchor=(0.5, -0.15),
               frameon=False, fontsize=8.5, title_fontsize=9.5)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – COMBINED SCORE BARPLOT: one bar per parameter configuration
# ══════════════════════════════════════════════════════════════════════════════

def make_combined_score_barplot(df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal barplot: one bar per parameter configuration,
    y = average combined score across all iterations.
    Bars are sorted by score; the baseline config is coloured grey.
    """
    param_cols = [c for c in JOIN_COLS if c != 'iteration_number']

    avg = (df.groupby(param_cols)[COMBINED_SCORE_COL]
             .mean()
             .reset_index())

    def config_label(row):
        diffs = [f'{p}={row[p]}' for p in VARIED_PARAMS if row[p] != BASELINE[p]]
        if not diffs:
            return 'baseline'
        return diffs[0] if len(diffs) == 1 else ', '.join(diffs)

    avg['_label'] = avg.apply(config_label, axis=1)
    avg = avg.sort_values(COMBINED_SCORE_COL, ascending=True)

    colors = ['#888888' if lbl == 'baseline' else PALETTE[0] for lbl in avg['_label']]

    fig, ax = plt.subplots(figsize=(7, max(4, len(avg) * 0.45 + 1.2)),
                           constrained_layout=True)
    bars = ax.barh(avg['_label'], avg[COMBINED_SCORE_COL],
                   color=colors, alpha=0.85, height=0.65)

    for bar, val in zip(bars, avg[COMBINED_SCORE_COL]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=8.5)

    ax.set_xlabel('Average Combined Score', fontsize=10)
    ax.set_title('Average combined score per parameter configuration - SpatialMETA',
                 fontsize=12, fontweight='bold')
    ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – TOOL COMPARISON BARPLOT
# ══════════════════════════════════════════════════════════════════════════════
 
# Metrics included in the tool comparison barplot.
TOOL_COMPARISON_METRICS = [
    'NMI',
    'ARI',
    'Continuity Score',
    'Marker score',
    'Biological Conservation',
    'Combined Score',
]
 
 
def get_spatial_meta_median_run(df: pd.DataFrame) -> pd.Series:
    """
    From spatialMETA data, pick the parameter configuration with the highest
    average combined score, then return the single iteration whose combined
    score is closest to the median of that config.
    """
    param_cols = [c for c in JOIN_COLS if c != 'iteration_number']
    avg = df.groupby(param_cols)[COMBINED_SCORE_COL].mean()
    best_params = avg.idxmax()
 
    mask = pd.Series(True, index=df.index)
    for col, val in zip(param_cols, best_params if isinstance(best_params, tuple) else [best_params]):
        mask &= (df[col] == val)
    best_df = df[mask]
 
    median_val = best_df[COMBINED_SCORE_COL].median()
    median_idx = (best_df[COMBINED_SCORE_COL] - median_val).abs().idxmin()
    return best_df.loc[median_idx]
 
 
def make_tool_comparison_figure(spatial_meta_metrics_file: str,
                                spatial_meta_aggregate_file: str,
                                miso_3m_file: str,
                                miso_2m_file: str) -> plt.Figure:
    """
    Grouped horizontal barplot comparing spatialMETA (median run of best config)
    with MISO 3m and MISO 2m across all shared benchmarking metrics.
    Each metric is one row; bars are grouped by tool.
    """
    df_tool = load_data(spatial_meta_metrics_file, spatial_meta_aggregate_file)
    miso3 = pd.read_csv(miso_3m_file).iloc[0]
    miso2 = pd.read_csv(miso_2m_file).iloc[0]
 
    # Determine which combined-score components are available across all tools,
    # then recompute Combined Score consistently for all three using only those.
    shared_combined = [c for c in COMBINED_SCORE_COLS if c in miso3.index and c in miso2.index]
    for miso in [miso3, miso2]:
        miso[COMBINED_SCORE_COL] = sum(miso[c] for c in shared_combined) / len(shared_combined)
 
    meta = get_spatial_meta_median_run(df_tool).copy()
    meta[COMBINED_SCORE_COL] = sum(meta[c] for c in shared_combined) / len(shared_combined)
 
    tool_data = {
        'spatialMETA': meta,
        'MISO 3m':     miso3,
        'MISO 2m':     miso2,
    }
    tools     = list(tool_data.keys())
    metrics   = TOOL_COMPARISON_METRICS
    n_tools   = len(tools)
    n_metrics = len(metrics)
    bar_h     = 0.7 / n_tools
    y_pos     = np.arange(n_metrics)
 
    fig, ax = plt.subplots(figsize=(9, n_metrics * 0.75 + 1.5), constrained_layout=True)
 
    for i, tool in enumerate(tools):
        offsets = (i - (n_tools - 1) / 2) * bar_h
        values  = [tool_data[tool][m] for m in metrics]
        bars = ax.barh(
            y_pos + offsets, values,
            height=bar_h * 0.88,
            color=PALETTE[i % len(PALETTE)],
            label=tool,
            alpha=0.88,
        )
        for bar, val in zip(bars, values):
            x = bar.get_width()
            if x > 0.08:
                ax.text(x * 0.97, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', ha='right',
                        fontsize=7.5, color='white', fontweight='bold')
            else:
                ax.text(x + 0.005, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', ha='left',
                        fontsize=7.5, color='#333')
 
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=10)
    ax.set_xlabel('Score', fontsize=10)
    ax.set_title('Tool comparison — shared benchmarking metrics',
                 fontsize=13, fontweight='bold')
    ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.legend(title='Tool', frameon=False, fontsize=9,
              title_fontsize=9.5, loc='lower right')
    return fig
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    df = load_data()
 
    # ── lr × max_epoch sweep (optional) ──────────────────────────────────────
    # Set these to the paths of the lr/epoch sweep files to generate those plots.
    LR_EPOCH_AGGREGATE_FILE = "/home/mnxsybi_lumc/spatial-multiomics-benchmarking/benchmarks/results/learning_rate_max_epoch_23062026/automated/results_metrics.xlsx"   # e.g. "results_metrics.xlsx" for lr/epoch run
    LR_EPOCH_METRICS_FILE = "/home/mnxsybi_lumc/spatial-multiomics-benchmarking/benchmarks/results/learning_rate_max_epoch_23062026/automated/results.xlsx"   # e.g. "results.xlsx" for lr/epoch run
 
    if LR_EPOCH_METRICS_FILE and LR_EPOCH_AGGREGATE_FILE:
        df_lr = load_lr_epoch_data(LR_EPOCH_METRICS_FILE, LR_EPOCH_AGGREGATE_FILE)
        fig   = make_lr_epoch_boxplots(df_lr)
        path  = OUTPUT_DIR / 'boxplot_lr_epoch.png'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f'  {path}')
    else:
        print('lr × epoch plots skipped — set LR_EPOCH_METRICS_FILE and LR_EPOCH_AGGREGATE_FILE in main().')
 
    # ── Main parameter sweep ───────────────────────────────────────────────────
    # Plot 1 – boxplots: one figure per varied parameter
    for param in VARIED_PARAMS:
        fig  = make_boxplot_figure(df, param)
        path = OUTPUT_DIR / f'boxplot_{param}.png'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f'  {path}')
 
    # Plot 2 – combined score barplot across all parameter configurations
    fig  = make_combined_score_barplot(df)
    path = OUTPUT_DIR / 'barplot_combined_score.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')
 
    # Plot 3 – tool comparison
    # Set these paths to generate the tool comparison barplot.
    SPATIAL_META_METRICS_FILE   = "/home/mnxsybi_lumc/spatial-multiomics-benchmarking/benchmarks/results/last_best_28062026/results.xlsx"   # e.g. "path/to/results_metrics.xlsx"
    SPATIAL_META_AGGREGATE_FILE = "/home/mnxsybi_lumc/spatial-multiomics-benchmarking/benchmarks/results/last_best_28062026/results_metrics.xlsx"   # e.g. "path/to/results.xlsx"
    MISO_3M_FILE = "/home/mnxsybi_lumc/spatial-multiomics-benchmarking/benchmarks_miso/MISO_3m_results.csv"   # e.g. "MISO_3m_results.csv"
    MISO_2M_FILE = "/home/mnxsybi_lumc/spatial-multiomics-benchmarking/benchmarks_miso/MISO_2m_results.csv"   # e.g. "MISO_2m_results.csv"
 
    df = load_data(SPATIAL_META_METRICS_FILE, SPATIAL_META_AGGREGATE_FILE)
    best_run = get_spatial_meta_median_run(df)

    print("Best spatialMETA run:")
    for col in [c for c in JOIN_COLS if c != 'iteration_number'] + ['iteration_number']:
        print(f"  {col}: {best_run[col]}")
    print(f"  {COMBINED_SCORE_COL}: {best_run[COMBINED_SCORE_COL]:.4f}")


    if SPATIAL_META_METRICS_FILE and SPATIAL_META_AGGREGATE_FILE and MISO_3M_FILE and MISO_2M_FILE:
        fig  = make_tool_comparison_figure(SPATIAL_META_METRICS_FILE, SPATIAL_META_AGGREGATE_FILE, MISO_3M_FILE, MISO_2M_FILE)
        path = OUTPUT_DIR / 'barplot_tool_comparison.png'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f'  {path}')
    else:
        print('Tool comparison skipped — set SPATIAL_META_METRICS_FILE, SPATIAL_META_AGGREGATE_FILE, MISO_3M_FILE and MISO_2M_FILE in main().')
 
 
if __name__ == '__main__':
    main()
