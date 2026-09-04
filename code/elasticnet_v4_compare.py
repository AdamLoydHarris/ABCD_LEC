"""LEC vs PFC comparison for the V4 anchoring regression.

Reads the exported `.npz` files that `elasticnet_regression_v4.export_regression_outputs`
writes, so it needs neither `data_dic` nor a re-fit -- point it at one output directory per
dataset per lag direction and it does the rest.

    python code/elasticnet_v4_compare.py                       # newest run of each
    python code/elasticnet_v4_compare.py --lec-root data/figures \\
                                         --pfc-root mFC_data/data/figures

The **unit of analysis is a mouse_recday**: each point is one recday's value, the bar is the
mean across recdays and the error bar is +/- SEM, matching
`plot_gp_any_tuning_pfc_vs_lec.py`. `min_neurons` drops degenerate recdays --
`me10_20122021_21122021` has exactly one neuron, which can only ever give a 0/1 proportion.

Read the caveat table (`diagnostics_table`) alongside the result. On both datasets the
selection criterion carries a large baseline: the non-zero-lag test fires on ~22% of pure
Poisson noise, and the state-tuning filter is confounded by leg duration (see ELASTICNET_V4.md).
A difference in selection rate between datasets is only interpretable against those.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import elasticnet_regression_v4 as v4                                    # noqa: E402
from glm_analysis_v2 import apply_gridmaze_style                         # noqa: E402

DEFAULT_ROOTS = {
    'LEC': os.path.join(REPO_ROOT, 'data', 'figures'),
    'PFC': os.path.join(REPO_ROOT, 'mFC_data', 'data', 'figures'),
}
#: Very Peri / Saffron -- strong luminance contrast, greyscale-robust (gridmaze-colors), and
#: already the repo's region pair in `plot_gp_any_tuning_pfc_vs_lec.py` / `ccgp_state_pairs.py`.
REGION_COLORS = {'LEC': '#6667AB', 'PFC': '#FFA500'}
#: Past / future -- warm retrospective, cool prospective (gridmaze-colors).
DIRECTION_COLORS = {'past': '#C03030', 'future': '#2A6FB5'}
STONE = '#B4B2A9'          # CIs, null lines, non-data ink
CAVIAR = '#2C2C2A'
FIG_DIR = os.path.join(HERE, 'figures')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def resolve_latest(root, direction, estimator=None):
    """Newest `{estimator}_v4_{direction}_*` directory under `root`.

    `estimator` is 'poisson' / 'elasticnet' / 'linear', or None for whichever is newest.
    Also matches the legacy `elasticnet_v4_*` layout, which was written under that name even
    for Poisson runs -- check `run_config.json` if an old folder looks surprising.
    """
    pattern = f'{estimator or "*"}_v4_{direction}_*'
    hits = sorted(glob.glob(os.path.join(root, pattern)))
    hits = [h for h in hits if os.path.isdir(h)]
    if not hits:
        raise FileNotFoundError(f'no {pattern} directory under {root}')
    return hits[-1]


def read_manifest(out_dir):
    """`run_config.json` for a run directory, or None for a run that predates it."""
    import json
    path = os.path.join(out_dir, 'run_config.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def check_comparable(dirs, keys=('estimator', 'num_lags', 'alpha_mode', 'elasticnet_alpha',
                                 'state_reduce', 'state_tuning_statistic',
                                 'pref_phase_source', 'restrict_to_pref_phase',
                                 'nonzero_lag_min', 'nonzero_lag_max')):
    """Warn when the runs being compared were not produced with the same settings.

    Comparing an ElasticNet LEC run against a Poisson PFC run is not a region difference, and
    with the folders now named by estimator that mistake should be visible -- but the manifest
    is what actually proves it. Returns a list of mismatch descriptions (empty if consistent,
    or if the runs predate `run_config.json`).
    """
    seen = {}
    for dataset, by_dir in dirs.items():
        for direction, path in by_dir.items():
            man = read_manifest(path)
            if man is None:
                continue
            flat = {'estimator': man.get('estimator'), **(man.get('config') or {})}
            for k in keys:
                seen.setdefault(k, {})[f'{dataset}/{direction}'] = flat.get(k)
    problems = []
    for k, vals in seen.items():
        distinct = {repr(v) for v in vals.values()}
        if len(distinct) > 1:
            problems.append(f'{k}: ' + ', '.join(f'{n}={v!r}' for n, v in sorted(vals.items())))
    if problems:
        print('WARNING - these runs were not produced with the same settings:')
        for p in problems:
            print(f'    {p}')
        print()
    return problems


def load_run(out_dir):
    """`{recday: results dict}` for every `*_arrays.npz` in an export directory."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, '*_arrays.npz'))):
        res = v4.load_regression_outputs(path)
        runs[res['mouse_recday']] = res
    if not runs:
        raise FileNotFoundError(f'no *_arrays.npz in {out_dir}')
    return runs


# ---------------------------------------------------------------------------
# Per-recday statistics -- the unit of analysis
# ---------------------------------------------------------------------------

def per_recday_stats(runs, min_neurons=2):
    """One row per recday: selection rate, mean r, peak lag, and the caveat diagnostics.

    Diagnostic fields are optional -- a run exported before they were added simply reports NaN
    rather than failing, so an in-flight sweep does not have to be redone.
    """
    import pandas as pd
    rows = []
    for recday, res in sorted(runs.items()):
        sel = (res['nonzero_lag_mask'] & res['state_tuned_mask']
               & ~np.isnan(res['mean_corrs']))
        n_units = len(sel)
        if n_units < min_neurons:
            continue
        nb = res.get('n_nonzero_betas')
        fitted = np.isfinite(nb) if nb is not None else None
        rows.append({
            'recday': recday,
            'mouse': recday.split('_')[0],
            'n_units': n_units,
            'n_folds': int(res.get('num_sessions', len(res['used_sessions']))),
            'n_selected': int(sel.sum()),
            'selection_rate': float(sel.mean()),
            'mean_r': float(np.nanmean(res['mean_corrs'][sel])) if sel.any() else np.nan,
            'mean_r_nonzero': (float(np.nanmean(res['mean_corrs_nonzero'][sel]))
                               if sel.any() else np.nan),
            'median_peak_lag': (float(np.nanmedian(res['peak_lags'][sel]))
                                if sel.any() else np.nan),
            'state_tuned_rate': float(res['state_tuned_mask'].mean()),
            'nonzero_lag_rate': float(res['nonzero_lag_mask'].mean()),
            'frac_allzero_fits': (float(np.mean(nb[fitted] == 0))
                                  if fitted is not None and fitted.any() else np.nan),
            'frac_pref_phase_flips': float(np.mean(
                [len(set(r[r >= 0].tolist())) > 1 for r in res['pref_phases']])),
            'state_duration_ratio': _scalar(res.get('state_duration_ratio')),
            'frac_pref_state_is_shortest': _scalar(res.get('frac_pref_state_is_shortest')),
        })
    return pd.DataFrame(rows)


def _scalar(v):
    """Optional exported scalar -> float, NaN when the run predates the field."""
    if v is None:
        return np.nan
    return float(np.asarray(v).reshape(-1)[0])


def selected_peak_lags(runs, min_neurons=2):
    """Peak lags of the selected units, pooled across recdays."""
    out = []
    for res in runs.values():
        sel = (res['nonzero_lag_mask'] & res['state_tuned_mask']
               & ~np.isnan(res['mean_corrs']))
        if len(sel) < min_neurons:
            continue
        out.append(res['peak_lags'][sel])
    return np.concatenate(out) if out else np.array([])


def prospective_index(runs_past, runs_future, min_neurons=2):
    """Per-recday mean of (r_future - r_past) / (|r_future| + |r_past|) over units selected
    in either direction. Positive = better explained prospectively."""
    import pandas as pd
    rows = []
    for recday in sorted(set(runs_past) & set(runs_future)):
        rp, rf = runs_past[recday], runs_future[recday]
        if len(rp['mean_corrs']) < min_neurons or len(rp['mean_corrs']) != len(rf['mean_corrs']):
            continue
        sp = rp['nonzero_lag_mask'] & rp['state_tuned_mask'] & ~np.isnan(rp['mean_corrs'])
        sf = rf['nonzero_lag_mask'] & rf['state_tuned_mask'] & ~np.isnan(rf['mean_corrs'])
        keep = sp | sf
        if not keep.any():
            continue
        a, b = rp['mean_corrs'][keep], rf['mean_corrs'][keep]
        denom = np.abs(a) + np.abs(b)
        idx = np.where(denom > 0, (b - a) / np.where(denom > 0, denom, 1), np.nan)
        rows.append({'recday': recday, 'mouse': recday.split('_')[0],
                     'n_units': int(keep.sum()), 'pro_index': float(np.nanmean(idx))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _bar_by_dataset(ax, stats, column, ylabel, title, jitter=0.13, seed=0):
    """Grouped bars: x = lag direction, colour = dataset; one point per recday."""
    rng = np.random.default_rng(seed)
    directions = ['past', 'future']
    datasets = [d for d in ('LEC', 'PFC') if d in stats]
    width = 0.36
    for di, dataset in enumerate(datasets):
        offset = (di - (len(datasets) - 1) / 2) * width
        means, sems, xs = [], [], []
        for xi, direction in enumerate(directions):
            v = stats[dataset][direction][column].dropna().to_numpy()
            means.append(v.mean() if v.size else np.nan)
            sems.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan)
            xs.append(xi + offset)
            if v.size:
                ax.plot(xi + offset + rng.uniform(-jitter, jitter, v.size), v, 'o',
                        mfc='none', mec=STONE, ms=2.6, lw=0, zorder=2)
        ax.bar(xs, means, width * 0.88, yerr=sems, capsize=2,
               color=REGION_COLORS[dataset], edgecolor=CAVIAR, linewidth=0.6,
               error_kw=dict(lw=0.8, ecolor=CAVIAR), zorder=1,
               label=f'{dataset} (n={len(stats[dataset]["past"])} recdays)')
    ax.set_xticks(range(len(directions)))
    ax.set_xticklabels([d.capitalize() for d in directions])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_lec_vs_pfc(stats, peak_lags, pro, out_path=None):
    """Four-panel LEC vs PFC comparison. Returns the Figure.

    `stats`     {dataset: {direction: per_recday_stats DataFrame}}
    `peak_lags` {dataset: {direction: 1-D array of selected peak lags}}
    `pro`       {dataset: prospective_index DataFrame}
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    apply_gridmaze_style()
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.6))

    _bar_by_dataset(axes[0], stats, 'selection_rate',
                    'Selected units (fraction)', 'Selection rate')
    # The non-zero-lag criterion fires on ~22% of pure Poisson noise, so the bars are read
    # against this line, not against zero.
    axes[0].axhline(0.22, color=STONE, ls='--', lw=0.8, zorder=0)
    axes[0].set_ylim(0, max(0.26, axes[0].get_ylim()[1]))
    axes[0].text(0.98, 0.22, 'noise', transform=axes[0].get_yaxis_transform(),
                 ha='right', va='bottom', fontsize=6, color=CAVIAR)
    axes[0].legend(frameon=False, loc='upper left', fontsize=6)

    _bar_by_dataset(axes[1], stats, 'mean_r',
                    'Mean r (selected units)', 'Predicted vs actual')
    axes[1].axhline(0, color=CAVIAR, lw=0.8, zorder=0)

    ax = axes[2]
    for dataset in [d for d in ('LEC', 'PFC') if d in peak_lags]:
        for direction, ls in (('past', '-'), ('future', '--')):
            v = peak_lags[dataset][direction]
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            counts, edges = np.histogram(v, bins=np.arange(-0.5, 12.5), density=True)
            ax.step(edges[:-1] + 0.5, counts, where='mid', ls=ls, lw=1.2,
                    color=REGION_COLORS[dataset], label=f'{dataset} {direction}')
    for lo in (2, 10):
        ax.axvline(lo, color=STONE, ls=':', lw=0.8, zorder=0)
    ax.set_xlabel('Peak lag (phase-steps)')
    ax.set_ylabel('Fraction of selected units')
    ax.set_title('Peak lag')
    ax.legend(frameon=False, fontsize=6)

    ax = axes[3]
    rng = np.random.default_rng(1)
    datasets = [d for d in ('LEC', 'PFC') if d in pro]
    for xi, dataset in enumerate(datasets):
        v = pro[dataset]['pro_index'].dropna().to_numpy()
        if v.size:
            ax.plot(xi + rng.uniform(-0.13, 0.13, v.size), v, 'o', mfc='none', mec=STONE,
                    ms=2.6, lw=0, zorder=2)
            ax.bar([xi], [v.mean()], 0.5,
                   yerr=[v.std(ddof=1) / np.sqrt(v.size)] if v.size > 1 else None,
                   capsize=2, color=REGION_COLORS[dataset], edgecolor=CAVIAR, linewidth=0.6,
                   error_kw=dict(lw=0.8, ecolor=CAVIAR), zorder=1)
    ax.axhline(0, color=CAVIAR, lw=0.8, zorder=0)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel('(r$_{future}$ - r$_{past}$) / (|r$_f$| + |r$_p$|)')
    ax.set_title('Prospective vs retrospective')

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
        with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                             'pdf.fonttype': 42, 'ps.fonttype': 42}):
            for ext in ('svg', 'png'):
                fig.savefig(f'{out_path}.{ext}', bbox_inches=None, dpi=300)
    return fig


# ---------------------------------------------------------------------------
# The caveat table
# ---------------------------------------------------------------------------

def diagnostics_table(stats):
    """Side-by-side caveats, so a dataset difference can be read against them.

    None of these are cosmetic: `frac_allzero_fits` is the alpha dropout (a firing-rate cut),
    `state_duration_ratio` drives the state-tuning false-positive rate (~5% at 1x, ~95% at 2x),
    and `frac_pref_phase_flips` is the share of neurons whose beta coordinate frame changes
    between folds.
    """
    import pandas as pd
    rows = []
    for dataset, by_dir in stats.items():
        for direction, df in by_dir.items():
            if df.empty:
                continue
            rows.append({
                'dataset': dataset, 'direction': direction,
                'n_recdays': len(df), 'n_units': int(df.n_units.sum()),
                'median_folds': float(df.n_folds.median()),
                'selection_rate': round(float(df.selection_rate.mean()), 3),
                'state_tuned_rate': round(float(df.state_tuned_rate.mean()), 3),
                'nonzero_lag_rate': round(float(df.nonzero_lag_rate.mean()), 3),
                'frac_allzero_fits': round(float(df.frac_allzero_fits.mean()), 3),
                'frac_pref_phase_flips': round(float(df.frac_pref_phase_flips.mean()), 3),
                'leg_duration_ratio': round(float(df.state_duration_ratio.median()), 2),
                'pref_state_is_shortest': round(float(df.frac_pref_state_is_shortest.mean()), 3),
            })
    return pd.DataFrame(rows)


def compare_datasets(dirs, min_neurons=2, out_path=None, verbose=True):
    """`dirs` = {dataset: {direction: export directory}}. Returns (stats, table, fig)."""
    from scipy import stats as st

    check_comparable(dirs)
    runs = {ds: {d: load_run(p) for d, p in by_dir.items()} for ds, by_dir in dirs.items()}
    stats = {ds: {d: per_recday_stats(r, min_neurons) for d, r in by_dir.items()}
             for ds, by_dir in runs.items()}
    peak_lags = {ds: {d: selected_peak_lags(r, min_neurons) for d, r in by_dir.items()}
                 for ds, by_dir in runs.items()}
    pro = {ds: prospective_index(by_dir['past'], by_dir['future'], min_neurons)
           for ds, by_dir in runs.items() if {'past', 'future'} <= set(by_dir)}

    table = diagnostics_table(stats)
    if verbose:
        print(table.to_string(index=False))
        for direction in ('past', 'future'):
            have = [ds for ds in stats if direction in stats[ds] and not stats[ds][direction].empty]
            if len(have) == 2:
                a = stats[have[0]][direction].selection_rate
                b = stats[have[1]][direction].selection_rate
                t, p = st.ttest_ind(a, b, equal_var=False)
                print(f'\n{direction}: selection rate {have[0]} {a.mean():.3f} vs '
                      f'{have[1]} {b.mean():.3f}  (Welch t={t:.2f}, p={p:.2g}, '
                      f'n={len(a)} vs {len(b)} recdays)')
        for ds, df in pro.items():
            v = df.pro_index.dropna()
            if len(v) > 1:
                t, p = st.ttest_1samp(v, 0)
                print(f'{ds}: prospective index mean={v.mean():+.3f} '
                      f'(t={t:.2f}, p={p:.2g}, n={len(v)} recdays)')

    fig = plot_lec_vs_pfc(stats, peak_lags, pro, out_path=out_path)
    return stats, table, fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lec-root', default=DEFAULT_ROOTS['LEC'])
    ap.add_argument('--pfc-root', default=DEFAULT_ROOTS['PFC'])
    ap.add_argument('--min-neurons', type=int, default=2)
    ap.add_argument('--estimator', default=None,
                    choices=['poisson', 'elasticnet', 'linear'],
                    help='pick a specific estimator\'s run; default = newest of any')
    ap.add_argument('--out', default=None,
                    help='figure stem; defaults to figures/{estimator}_v4_lec_vs_pfc')
    args = ap.parse_args()
    out = args.out or os.path.join(FIG_DIR, f'{args.estimator or "any"}_v4_lec_vs_pfc')

    dirs = {}
    for name, root in (('LEC', args.lec_root), ('PFC', args.pfc_root)):
        try:
            dirs[name] = {d: resolve_latest(root, d, args.estimator)
                          for d in ('past', 'future')}
        except FileNotFoundError as exc:
            print(f'{name}: {exc}')
    if not dirs:
        return 1
    for name, by_dir in dirs.items():
        for d, p in by_dir.items():
            print(f'{name} {d}: {p}')
    print()
    compare_datasets(dirs, min_neurons=args.min_neurons, out_path=out)
    print(f'\nfigure -> {out}.svg / .png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
