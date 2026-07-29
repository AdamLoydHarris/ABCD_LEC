"""encoding_bias_simulation.py — ground-truth test for binning-driven CPD bias.

Companion to ``glm_cv_cpd.py``. Build first; it *validates* every other fix.

The question
------------
Does the mismatched binning in ``glm_analysis_v2`` (quantile deciles for
``time_*`` / ``distance_*`` vs equal-width bins for ``goal_progress``) make a
genuinely goal-progress-tuned neuron *look* time- or distance-tuned?

The method
----------
We know the answer for a *synthetic* neuron because we built its tuning. So:

1. Take the real pooled covariates for one recday (``build_design_tables``), which
   carry the true behavioural sampling / occupancy and the real time↔progress↔distance
   collinearity.
2. Synthesise neurons whose firing depends on **exactly one** continuous variable
   (e.g. temporal goal progress), with a known tuning shape and additive noise. The
   firing is generated from the *continuous* variable, so no encoding scheme gets an
   unfair exact match to the generative process.
3. Push the synthetic firing through the GLM design + CPD under several encoding
   schemes (``glm_cv_cpd``), in-sample and cross-validated.
4. Build the **misattribution matrix**: rows = true variable, columns = regressor the
   CPD was attributed to. A perfectly unbiased pipeline is diagonal. Off-diagonal mass
   in the ``time_*`` / ``distance_*`` columns of the ``goal_progress`` row is exactly
   the bias the concern is about.

Use this to (a) confirm the bias exists under ``glm_onehot`` and (b) show that
``matched_*`` schemes + cross-validation move the matrix toward the diagonal.
"""

from __future__ import annotations

import numpy as np

import glm_cv_cpd as cv
import glm_analysis_v2 as glm


# ============================================================================
# Tuning shapes (defined on a normalised support x in [0, 1])
# ============================================================================

def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def tuning_curve(kind, x):
    """Return a tuning profile in [0, 1] over normalised values ``x`` in [0, 1].

    'bimodal' (peaks near both ends) is the case the concern highlights: a cell
    with more structure early and late, where decile binning of time gives the
    densest resolution.
    """
    x = np.asarray(x, float)
    if kind == 'ramp_up':
        t = x
    elif kind == 'ramp_down':
        t = 1.0 - x
    elif kind == 'early':
        t = _gauss(x, 0.15, 0.10)
    elif kind == 'mid':
        t = _gauss(x, 0.50, 0.12)
    elif kind == 'late':
        t = _gauss(x, 0.85, 0.10)
    elif kind == 'bimodal':
        t = _gauss(x, 0.10, 0.08) + _gauss(x, 0.90, 0.08)
    else:
        raise ValueError(f"unknown tuning kind {kind!r}")
    # nan-safe normalisation: a single NaN in x (e.g. phase_dist when the animal
    # was stationary) must not collapse the whole curve to zero. NaN entries are
    # zeroed by the caller's finite mask anyway.
    lo, hi = np.nanmin(t), np.nanmax(t)
    rng = hi - lo
    out = (t - lo) / rng if rng > 0 else np.zeros_like(t)
    return np.nan_to_num(out)


DEFAULT_KINDS = ('ramp_up', 'ramp_down', 'early', 'mid', 'late', 'bimodal')
DEFAULT_TRUE_VARS = ('goal_progress', 'goal_progress_distance',
                     'time_from_reward', 'time_to_reward',
                     'distance_from_reward', 'distance_to_reward')


# ============================================================================
# Synthetic firing
# ============================================================================

def _normalised_value(table, var):
    """Continuous values for ``var`` mapped to [0, 1], plus a finite mask.

    Unit-range variables (goal progress) are used as-is; others are min/max
    scaled by their 1st–99th percentile (matches the encoders' clipping).
    """
    vals = np.asarray(table[cv._CONT_VARS[var]], float)
    finite = np.isfinite(vals)
    if var in cv._UNIT_RANGE_VARS:
        x = np.clip(vals, 0, 1)
    else:
        lo, hi = np.percentile(vals[finite], [1, 99])
        hi = hi if hi > lo else lo + 1e-6
        x = np.clip((vals - lo) / (hi - lo), 0, 1)
    return x, finite


def synthesize_neuron(table, true_var, kind, *, gain=1.0, base=0.2, snr=1.0,
                      noise='gaussian', seed=0):
    """One synthetic neuron tuned only to ``true_var`` with shape ``kind``.

    snr scales the noise relative to the signal sd. Samples where ``true_var`` is
    undefined (e.g. phase_dist when the animal did not move) fire at baseline.
    """
    rng = np.random.default_rng(seed)
    x, finite = _normalised_value(table, true_var)
    signal = gain * tuning_curve(kind, x)
    signal[~finite] = 0.0
    rate = base + signal
    if noise == 'poisson':
        return rng.poisson(np.maximum(rate, 1e-3)).astype(float)
    sd = np.std(signal)
    sigma = (sd / snr) if (sd > 0 and snr > 0) else (base / max(snr, 1e-6))
    return rate + rng.normal(0.0, sigma, size=rate.shape)


def simulate_population(table, *, true_vars=DEFAULT_TRUE_VARS, kinds=DEFAULT_KINDS,
                        n_per_cell=20, gain=1.0, base=0.2, snr=1.0,
                        noise='gaussian', seed=0):
    """Build a synthetic firing matrix and the per-neuron ground-truth label.

    Returns
    -------
    FR     : (n_neurons, T) synthetic firing
    labels : (n_neurons,) array of true_var names
    """
    T = len(table['t'])
    cols, labels = [], []
    s = seed
    for var in true_vars:
        for kind in kinds:
            for _ in range(n_per_cell):
                cols.append(synthesize_neuron(table, var, kind, gain=gain,
                                               base=base, snr=snr, noise=noise, seed=s))
                labels.append(var)
                s += 1
    return np.asarray(cols), np.asarray(labels, dtype=object)


# ============================================================================
# Misattribution matrix
# ============================================================================

def misattribution_matrix(table, scheme, *, true_vars=DEFAULT_TRUE_VARS,
                          kinds=DEFAULT_KINDS, n_per_cell=20, gain=1.0, base=0.2,
                          snr=1.0, noise='gaussian', cv_folds=0, n_bins=10, seed=0):
    """Mean CPD attributed to each regressor, per true generative variable.

    ``cv_folds=0`` uses in-sample CPD (the GLM's metric); ``cv_folds>=2`` uses
    held-out CPD via :func:`glm_cv_cpd.cpd_cv`.

    Returns
    -------
    M       : (len(true_vars), len(cols)) mean-CPD matrix
    rows    : list of true_var names (matrix row order)
    cols    : list of regressor names (matrix column order)
    """
    FR, labels = simulate_population(table, true_vars=true_vars, kinds=kinds,
                                     n_per_cell=n_per_cell, gain=gain, base=base,
                                     snr=snr, noise=noise, seed=seed)
    sim_table = dict(table)
    sim_table['FR'] = FR

    if cv_folds and cv_folds >= 2:
        res = cv.cpd_cv(sim_table, scheme=scheme, n_folds=cv_folds, n_bins=n_bins, seed=seed)
        cpd, cols = res['cpd'], res['names']
    else:
        enc = cv.fit_encoders(sim_table, cv.make_scheme(scheme) if isinstance(scheme, str)
                              else scheme, n_bins=n_bins)
        X, groups, cols = cv.apply_encoders(sim_table, enc, n_bins=n_bins)
        cpd, _ = cv.cpd_insample(X, FR.T, groups)

    rows = list(true_vars)
    M = np.full((len(rows), len(cols)), np.nan)
    for i, var in enumerate(rows):
        sel = labels == var
        for j, name in enumerate(cols):
            M[i, j] = np.nanmean(cpd[name][sel])
    return M, rows, cols


def leakage_summary(M, rows, cols):
    """For each true variable, fraction of attributed CPD on the correct regressor
    vs leaked to the time / distance competitors. Returns a dict of dicts."""
    out = {}
    time_dist = [c for c in cols if c.startswith('time_') or c.startswith('distance_')]
    for i, var in enumerate(rows):
        total = np.nansum(np.clip(M[i], 0, None))
        correct = max(M[i, cols.index(var)], 0.0) if var in cols else np.nan
        leaked_td = float(np.nansum([max(M[i, cols.index(c)], 0.0) for c in time_dist
                                     if c != var]))
        out[var] = {
            'correct_cpd': float(correct),
            'leaked_to_time_distance': leaked_td,
            'fraction_correct': float(correct / total) if total > 0 else np.nan,
        }
    return out


# ============================================================================
# Plotting
# ============================================================================

_SHORT = {
    'place': 'place', 'head_direction': 'HD',
    'goal_progress': 'GP(t)', 'goal_progress_distance': 'GP(d)',
    'speed': 'speed', 'acceleration': 'acc',
    'time_from_reward': 'time-from', 'time_to_reward': 'time-to',
    'distance_from_reward': 'dist-from', 'distance_to_reward': 'dist-to',
}


def plot_misattribution(M, rows, cols, ax=None, title=None, vmax=None,
                        drop=('place', 'head_direction')):
    """Heatmap of mean CPD (true variable × attributed regressor).

    The diagonal is correct attribution; off-diagonal mass in the time/distance
    columns of a goal-progress row is the binning bias. ``place``/``HD`` columns
    are dropped by default (nuisance, never the generative variable here).
    """
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()

    keep = [j for j, c in enumerate(cols) if c not in drop]
    Mp = M[:, keep]
    cols_p = [cols[j] for j in keep]
    if vmax is None:
        vmax = np.nanpercentile(np.clip(Mp, 0, None), 99) or 1e-3

    if ax is None:
        _, ax = plt.subplots(figsize=(0.55 * len(cols_p) + 1.5, 0.45 * len(rows) + 1.2))
    im = ax.imshow(Mp, cmap='magma', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(cols_p)))
    ax.set_xticklabels([_SHORT.get(c, c) for c in cols_p], rotation=45, ha='right')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_SHORT.get(r, r) for r in rows])
    ax.set_xlabel('attributed to (CPD)')
    ax.set_ylabel('true tuning')
    for i in range(Mp.shape[0]):
        for j in range(Mp.shape[1]):
            if np.isfinite(Mp[i, j]):
                ax.text(j, i, f'{Mp[i, j]:.02f}', ha='center', va='center',
                        color='white' if Mp[i, j] < 0.6 * vmax else 'black', fontsize=6)
    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='mean CPD')
    return ax


def run_simulation(table, *, schemes=('glm_onehot', 'matched_linear', 'matched_quantile'),
                   cv_folds=0, plot=True, **kw):
    """Compute the misattribution matrix under several schemes and (optionally)
    plot them side by side. Returns {scheme: (M, rows, cols)} plus leakage dicts.
    """
    results = {}
    for sc in schemes:
        M, rows, cols = misattribution_matrix(table, sc, cv_folds=cv_folds, **kw)
        results[sc] = {'M': M, 'rows': rows, 'cols': cols,
                       'leakage': leakage_summary(M, rows, cols)}

    if plot:
        import matplotlib.pyplot as plt
        glm.apply_gridmaze_style()
        vmax = max(np.nanpercentile(np.clip(r['M'], 0, None), 99) for r in results.values())
        fig, axes = plt.subplots(1, len(schemes),
                                 figsize=(5.0 * len(schemes), 3.6), squeeze=False)
        tag = 'held-out' if cv_folds and cv_folds >= 2 else 'in-sample'
        for ax, sc in zip(axes[0], schemes):
            r = results[sc]
            plot_misattribution(r['M'], r['rows'], r['cols'], ax=ax,
                                title=f'{sc} ({tag})', vmax=vmax)
        fig.tight_layout()
        results['_figure'] = fig
    return results
