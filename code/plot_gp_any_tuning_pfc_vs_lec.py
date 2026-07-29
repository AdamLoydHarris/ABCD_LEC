"""Bar chart: proportion of neurons tuned to goal progress (any) — PFC vs LEC.

For each region we load the cached ``distance_gp_filtered`` GLM fit (the filtered,
30 s transition-duration variant that also carries the joint ``gp_any`` group) and
count, per neuron, whether it is significantly tuned to *goal progress (any)* — i.e.
time-based OR path-distance goal progress. Significance follows the repo convention:
the real joint F exceeds the 95th percentile of its permutation null (one-sided
permutation F-test, p < 0.05), matching the ``gp_any`` cells in
``LEC_glm_tuning_variables.ipynb`` and ``compute_tuning_arrays``.

The unit of analysis (n) is a mouse_recday: each point is one recday's proportion,
the bar is the mean across recdays, and the error bar is ±SEM.

Caveat: LEC's fit includes ``head_direction`` (9 base regressors) while PFC's does
not (8). The ``gp_any`` test is computed within each region's own model; strict
regressor-matching would require a refit and is out of scope for this figure.

Run:  python code/plot_gp_any_tuning_pfc_vs_lec.py
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from glm_analysis_v2 import apply_gridmaze_style, load_glm_results  # noqa: E402

# --- Config -----------------------------------------------------------------
SECTION = 'distance_gp_filtered'
SAVE_DIRS = {
    'LEC': os.path.join(REPO_ROOT, 'data', 'glm_outputs', 'LEC'),
    'PFC': os.path.join(REPO_ROOT, 'mFC_data', 'glm_outputs', 'PFC'),
}
# Very Peri / Saffron — strong luminance contrast, greyscale-robust (gridmaze-colors).
REGION_COLORS = {'LEC': '#6667AB', 'PFC': '#FFA500'}
FIG_DIR = os.path.join(HERE, 'figures')
FIG_STEM = os.path.join(FIG_DIR, 'gp_any_tuning_pfc_vs_lec')


def gp_any_proportions(permutation_results, min_neurons=2):
    """Per-recday proportion of neurons significantly tuned to goal progress (any).

    Returns ``{mouse_recday: proportion}``. Neurons missing the ``gp_any`` key are
    skipped; recdays with fewer than ``min_neurons`` valid neurons are omitted
    (single-neuron recdays give a degenerate 0/1 proportion).
    """
    props = {}
    for mouse_recday, neuron_dict in permutation_results.items():
        sig = []
        for f_real, f_perm in neuron_dict.values():
            gp_f = f_real.get('gp_any')
            gp_p = f_perm.get('gp_any')
            if gp_f is None or gp_p is None:
                continue
            sig.append(gp_f > np.percentile(gp_p, 95))
        if len(sig) >= min_neurons:
            props[mouse_recday] = float(np.mean(sig))
    return props


def plot_gp_any_tuning(region_props, ax=None, out_path=None):
    """Bar chart of mean ± SEM proportion tuned to goal progress (any), per region.

    ``region_props`` maps region name -> {mouse_recday: proportion}. Bars are the
    mean across recdays, error bars ±SEM, and each recday is drawn as a jittered
    point. Returns the ``matplotlib.figure.Figure``.
    """
    regions = list(region_props)
    rng = np.random.default_rng(0)

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(2.6, 3.2))
    else:
        fig = ax.figure

    means, sems, all_vals = [], [], []
    for x, region in enumerate(regions):
        vals = np.array(sorted(region_props[region].values()))
        all_vals.append(vals)
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals))
        means.append(mean)
        sems.append(sem)
        color = REGION_COLORS[region]

        ax.bar(x, mean, yerr=sem, width=0.6, color=color, alpha=0.7,
               edgecolor='black', linewidth=0.8, capsize=3,
               error_kw={'lw': 1.0, 'zorder': 4})
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x + jitter, vals, s=18, color=color, alpha=0.95,
                   edgecolors='white', linewidths=0.4, zorder=3)

    # Optional light between-region test (Mann-Whitney U on per-recday proportions).
    if len(regions) == 2:
        u, p = st.mannwhitneyu(all_vals[0], all_vals[1], alternative='two-sided')
        y_top = max(v.max() for v in all_vals)
        y_bar = y_top + 0.06
        ax.plot([0, 0, 1, 1], [y_bar, y_bar + 0.015, y_bar + 0.015, y_bar],
                color='black', lw=0.8)
        stars = ('***' if p < 1e-3 else '**' if p < 1e-2 else
                 '*' if p < 0.05 else 'n.s.')
        ax.text(0.5, y_bar + 0.02, f'{stars} (p = {p:.1e})',
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([f'{r}\n(n={len(region_props[r])})' for r in regions])
    ax.set_ylabel('Prop. tuned to goal progress (any)')
    ax.set_ylim(0, 1.05)
    ax.margins(x=0.25)

    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                             'pdf.fonttype': 42, 'ps.fonttype': 42}):
            fig.savefig(out_path, bbox_inches=None)
    return fig


def main():
    apply_gridmaze_style()

    region_props = {}
    for region, save_dir in SAVE_DIRS.items():
        perm = load_glm_results(save_dir, SECTION)['permutation_results']
        region_props[region] = gp_any_proportions(perm)

    print(f'Goal progress (any) tuning — section {SECTION!r}')
    for region, props in region_props.items():
        vals = np.array(list(props.values()))
        sem = vals.std(ddof=1) / np.sqrt(len(vals))
        print(f'  {region}: mean={vals.mean():.3f}  SEM={sem:.3f}  '
              f'n_recdays={len(vals)}')

    os.makedirs(FIG_DIR, exist_ok=True)
    fig = plot_gp_any_tuning(region_props, out_path=FIG_STEM + '.pdf')
    fig.savefig(FIG_STEM + '.png', dpi=200)
    print(f'Saved {FIG_STEM}.pdf and .png')


if __name__ == '__main__':
    main()
