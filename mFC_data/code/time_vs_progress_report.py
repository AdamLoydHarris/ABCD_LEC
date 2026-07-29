"""time_vs_progress_report.py — one report that adjudicates time vs goal progress.

Companion / orchestrator for the encoding-bias package. It combines four pieces
into a single answer, in order of evidential weight:

  ADJUDICATING TEST (decisive)
    The out-of-distribution dissociation in ``time_vs_progress_dissociation.py``:
    train on short legs, predict held-out long legs (and reverse). Only duration
    variability can break the within-leg time↔progress collinearity, so this is
    the test that actually decides which frame a cell uses. The pooled signed
    delta (``population_delta_test``) is the headline.

  DESCRIPTIVE / ROBUSTNESS (supporting)
    1. CV-CPD across encoding schemes (``glm_cv_cpd``): is the descriptive
       time-vs-GP CPD ordering stable when the binning is made flexibility-fair
       and cross-validated? If it flips with scheme, the original CPD result was
       a binning artefact — which is the whole point.
    2. Ground-truth simulation (``encoding_bias_simulation``): how diagonal is the
       misattribution matrix under each scheme?
    3. De-circularised peak-binning (``tuning_peak_binning``): the PI's adaptive
       bins, scored without double-dipping.

The CPD GLM is presented as *descriptive*; the OOD test is what adjudicates.
"""

from __future__ import annotations

import numpy as np

import glm_cv_cpd as cv
import glm_analysis_v2 as glm
import time_vs_progress_dissociation as tvp


# regressor groups that constitute each "frame" for the descriptive CPD contrast
_TIME = ['time_from_reward', 'time_to_reward']
_DIST = ['distance_from_reward', 'distance_to_reward']
_GP_T = 'goal_progress'
_GP_D = 'goal_progress_distance'


# ============================================================================
# Descriptive: CV-CPD across schemes
# ============================================================================

def compare_schemes_cpd(tables, *, schemes=('glm_onehot', 'matched_linear',
                                            'matched_quantile', 'matched_rc'),
                        n_folds=5, n_bins=10, seed=0, verbose=True):
    """Run cross-validated CPD for each encoding scheme and summarise the
    time-vs-goal-progress contrast.

    Returns
    -------
    pooled_by_scheme : {scheme: pooled-dict from glm_cv_cpd.run_cv_cpd}
    summary : list of per-scheme dicts with median CPD for GP / time / distance
              and the GP-minus-time contrast (median over neurons).
    """
    pooled_by_scheme, summary = {}, []
    for sc in schemes:
        if verbose:
            print(f"\n=== scheme: {sc} ===")
        _, pooled = cv.run_cv_cpd(tables, scheme=sc, n_folds=n_folds,
                                  n_bins=n_bins, seed=seed, verbose=verbose)
        pooled_by_scheme[sc] = pooled

        # Per-sample single-neuron held-out CPD is ~0 for the (majority) untuned
        # neurons, so a grand median over all neurons is uninformative. Summarise
        # the TUNED subpopulation (positive full-model held-out R^2) as well, since
        # that is where a time-vs-GP difference could exist at all.
        r2 = pooled['r2']
        tuned = np.isfinite(r2) & (r2 > 0)
        gp_arr = pooled['cpd'].get(_GP_T, np.full_like(r2, np.nan))
        time_arr = np.nanmax(np.vstack([pooled['cpd'][t] for t in _TIME]), axis=0)
        dist_arr = np.nanmax(np.vstack([pooled['cpd'][d] for d in _DIST]), axis=0)
        contrast = gp_arr - time_arr

        def _med(a, mask=None):
            a = a if mask is None else a[mask]
            return float(np.nanmedian(a)) if np.any(np.isfinite(a)) else np.nan

        summary.append({
            'scheme': sc,
            'frac_tuned': float(np.mean(tuned)),
            'median_full_r2': _med(r2),
            # tuned-subpopulation medians (the informative comparison)
            'GP_tuned': _med(gp_arr, tuned), 'time_tuned': _med(time_arr, tuned),
            'dist_tuned': _med(dist_arr, tuned),
            'GP_minus_time_tuned': _med(contrast, tuned),
            # all-neuron means (robust to the untuned mass dragging the median to 0)
            'GP_mean': float(np.nanmean(gp_arr)), 'time_mean': float(np.nanmean(time_arr)),
            'GP_minus_time_mean': float(np.nanmean(contrast)),
        })
    return pooled_by_scheme, summary


def plot_scheme_comparison(pooled_by_scheme, ax=None):
    """Grouped bar: held-out CPD for GP(t), time-from/to, dist-from/to under each
    scheme, averaged over the TUNED subpopulation (full-model held-out R^2 > 0).
    Stable bar heights across schemes => the descriptive ordering is not a binning
    artefact."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()

    regs = [_GP_T, 'time_from_reward', 'time_to_reward',
            'distance_from_reward', 'distance_to_reward']
    short = ['GP(t)', 'time-from', 'time-to', 'dist-from', 'dist-to']
    schemes = list(pooled_by_scheme)
    x = np.arange(len(regs))
    w = 0.8 / len(schemes)
    colors = ['crimson', 'steelblue', 'darkorange', 'seagreen']

    ax = ax or plt.subplots(figsize=(6.5, 3.2))[1]
    for k, sc in enumerate(schemes):
        p = pooled_by_scheme[sc]
        tuned = np.isfinite(p['r2']) & (p['r2'] > 0)
        vals = [np.nanmean(p['cpd'][r][tuned]) if r in p['cpd'] else np.nan for r in regs]
        ax.bar(x + k * w, vals, w, label=sc, color=colors[k % len(colors)])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(short, rotation=30, ha='right')
    ax.set_ylabel('mean held-out CPD (tuned cells)')
    ax.set_title('descriptive CPD by encoding scheme (cross-validated)')
    ax.legend(frameon=False, fontsize=6)
    return ax


# ============================================================================
# Adjudicating: OOD dissociation summary
# ============================================================================

def ood_summary(tables, *, contrast='temporal', region='', r2_floor=0.0, **kw):
    """Run the OOD dissociation across recdays and return (df, classified, test).

    ``df``        per-neuron OOD frame (delta_ood, R^2s, bootstrap CI)
    ``classified`` adds the conservative per-cell class
    ``test``      population_delta_test — the headline signed-delta summary
    """
    df = tvp.run_population_ood(tables, contrast, region=region, **kw)
    if df.empty:
        return df, df, df
    classified = tvp.classify(df, r2_floor=r2_floor)
    test = tvp.population_delta_test(df)
    return df, classified, test


# ============================================================================
# Full report
# ============================================================================

def build_report(tables, *, region='', contrast='temporal',
                 schemes=('glm_onehot', 'matched_linear', 'matched_quantile'),
                 run_simulation=True, run_peak_binning=True,
                 n_folds=5, n_bins=10, seed=0, ood_kw=None, sim_kw=None):
    """Assemble the full time-vs-progress report.

    Returns a dict with keys: 'ood' (df/classified/test), 'cpd' (pooled_by_scheme,
    summary), 'simulation' (per-scheme matrices, optional), 'peak_binning'
    (optional), and 'figure' (a multi-panel summary).
    """
    ood_kw = ood_kw or {}
    sim_kw = sim_kw or {}
    out = {}

    # --- adjudicating test ---
    df, classified, test = ood_summary(tables, contrast=contrast, region=region, **ood_kw)
    out['ood'] = {'df': df, 'classified': classified, 'test': test}

    # --- descriptive CPD across schemes ---
    pooled_by_scheme, summary = compare_schemes_cpd(
        tables, schemes=schemes, n_folds=n_folds, n_bins=n_bins, seed=seed)
    out['cpd'] = {'pooled_by_scheme': pooled_by_scheme, 'summary': summary}

    # --- simulation (one representative recday) ---
    if run_simulation:
        import encoding_bias_simulation as sim
        rep = next(iter(tables.values()))
        out['simulation'] = sim.run_simulation(rep, schemes=schemes, plot=False, **sim_kw)

    # --- de-circularised peak-binning (one representative recday) ---
    if run_peak_binning:
        import tuning_peak_binning as pk
        rep = next(iter(tables.values()))
        out['peak_binning'] = pk.run_peak_binning(rep, n_folds=n_folds,
                                                  n_bins=n_bins, seed=seed)

    out['figure'] = plot_report(out, region=region, contrast=contrast)
    print_summary(out, region=region, contrast=contrast)
    return out


def plot_report(report, *, region='', contrast='temporal'):
    """Multi-panel: OOD delta histogram (decisive) + CPD-by-scheme (robustness)."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    df = report['ood']['classified']
    if df is not None and not df.empty:
        tvp.plot_delta_hist(df, contrast, ax=axes[0])
    axes[0].set_title(f'{region} OOD delta (decisive): >0 time, <0 progress')
    plot_scheme_comparison(report['cpd']['pooled_by_scheme'], ax=axes[1])
    fig.tight_layout()
    return fig


def print_summary(report, *, region='', contrast='temporal'):
    """Print the headline: OOD adjudication + CPD robustness across schemes."""
    print('\n' + '=' * 70)
    print(f'TIME vs GOAL PROGRESS — {region or "(region)"} / {contrast}')
    print('=' * 70)

    test = report['ood']['test']
    print('\nADJUDICATING TEST — OOD generalisation (the decisive result):')
    if test is not None and not test.empty:
        for _, r in test.iterrows():
            lean = 'ELAPSED TIME' if r['median_delta'] > 0 else 'GOAL PROGRESS'
            print(f"  {r['region'] or region}: n={int(r['n'])}, median delta={r['median_delta']:+.3f} "
                  f"-> leans {lean}  (Wilcoxon p={r['wilcoxon_p']:.1e}, "
                  f"frac leaning time={r['frac_leaning_time']:.2f})")
    else:
        print('  (no OOD result — too few legs?)')

    print('\nDESCRIPTIVE — cross-validated CPD, robustness across binning schemes:')
    print('  Tuned subpopulation (held-out full-model R^2 > 0); per-sample single-neuron')
    print('  CPD is ~0 for untuned cells so the grand median is uninformative.')
    print('  If GP-minus-time flips sign across schemes, the in-sample CPD ordering was a')
    print('  binning artefact and only the OOD test should be trusted.')
    for s in report['cpd']['summary']:
        print(f"  {s['scheme']:>16}: tuned={s['frac_tuned']:.0%}  "
              f"GP={s['GP_tuned']:.3f}  time={s['time_tuned']:.3f}  dist={s['dist_tuned']:.3f}  "
              f"GP-time(tuned)={s['GP_minus_time_tuned']:+.3f}  "
              f"GP-time(mean)={s['GP_minus_time_mean']:+.4f}")

    if 'simulation' in report:
        print('\nBIAS CHECK — simulation misattribution (mean GP->time/dist leak):')
        for sc, r in report['simulation'].items():
            if sc.startswith('_'):
                continue
            lk = r['leakage'].get(_GP_T, {})
            print(f"  {sc:>16}: GP correct CPD={lk.get('correct_cpd', float('nan')):.3f}, "
                  f"leaked to time/dist={lk.get('leaked_to_time_distance', float('nan')):.3f}, "
                  f"frac correct={lk.get('fraction_correct', float('nan')):.2f}")

    if 'peak_binning' in report:
        pb = report['peak_binning']
        gp = float(np.nanmedian(pb['cpd'].get(_GP_T, np.array([np.nan]))))
        tt = float(np.nanmedian(np.nanmax(
            np.vstack([pb['cpd'][t] for t in _TIME]), axis=0)))
        print('\nPI PEAK-BINNING (de-circularised, held-out CPD):')
        print(f"  median GP={gp:.3f}  median time={tt:.3f}  GP-time={gp - tt:+.3f}")
    print('=' * 70 + '\n')
