"""tuning_peak_binning.py — the PI's data-adaptive binning, de-circularised.

Companion to ``glm_cv_cpd.py`` and ``encoding_bias_simulation.py``.

The PI's idea
-------------
Rather than placing time bins where *behaviour* is dense (quantile deciles) or
uniformly (equal width), place them where the *neural tuning* actually has
structure: build per-neuron tuning curves on a fine uniform grid, look at where
tuning concentrates across the population, and give those regions finer bins.
Apply the same recipe to every variable so the comparison is symmetric.

Two fixes to make it valid
--------------------------
1. **Avoid circularity (double-dipping).** Deriving bin edges from the same
   neurons/samples you then score inflates CPD exactly where you added
   resolution (Kriegeskorte et al. 2009). We split legs into two disjoint halves:
   derive edges on one half, score held-out CPD on the *other* half (and swap).
   No sample informs both the edges and its own score.
2. **Be robust to ramps.** Time/progress cells are often monotonic ramps with no
   meaningful "peak" — argmax lands noisily at an edge. We place resolution by the
   population density of |tuning gradient| (where tuning *changes* fastest), which
   captures ramps and bumps alike. The argmax-of-|tuning| distribution is still
   returned, as a diagnostic for the PI's original framing.

The derived edges are fed back through :func:`glm_cv_cpd.cpd_cv` with
``encoders=`` fixed, so only the regression betas are cross-validated.
"""

from __future__ import annotations

import numpy as np

import glm_cv_cpd as cv
import glm_analysis_v2 as glm


# ============================================================================
# Per-neuron tuning curves on a fine uniform grid
# ============================================================================

def _norm_and_range(table, var):
    """Normalised value x in [0, 1], finite mask, and the (lo, hi) used so edges
    can be mapped back to raw units for :func:`glm_analysis_v2.apply_onehot`."""
    vals = np.asarray(table[cv._CONT_VARS[var]], float)
    finite = np.isfinite(vals)
    if var in cv._UNIT_RANGE_VARS:
        return np.clip(vals, 0, 1), finite, (0.0, 1.0)
    lo, hi = np.percentile(vals[finite], [1, 99])
    hi = hi if hi > lo else lo + 1e-6
    return np.clip((vals - lo) / (hi - lo), 0, 1), finite, (lo, hi)


def tuning_curves(table, var, sample_mask, *, n_grid=50, min_count=5):
    """Mean firing per fine x-bin for every neuron, over ``sample_mask`` rows.

    Returns
    -------
    tuning  : (N, n_grid) mean firing per grid bin (NaN where under-sampled)
    centers : (n_grid,) bin centres in normalised [0, 1]
    rng     : (lo, hi) raw range used for normalisation
    """
    x, finite, rng = _norm_and_range(table, var)
    use = sample_mask & finite
    grid = np.linspace(0, 1, n_grid + 1)
    idx = np.clip(np.digitize(x, grid) - 1, 0, n_grid - 1)
    FR = np.asarray(table['FR'], float)
    N = FR.shape[0]
    tuning = np.full((N, n_grid), np.nan)
    for b in range(n_grid):
        m = use & (idx == b)
        if m.sum() >= min_count:
            tuning[:, b] = FR[:, m].mean(1)
    centers = 0.5 * (grid[:-1] + grid[1:])
    return tuning, centers, rng


def structure_density(tuning):
    """Population density of tuning *change* across the grid.

    Each neuron's curve is z-scored (so firing scale is irrelevant), NaNs
    interpolated, and |gradient| averaged across neurons. High where tuning
    changes fast — bumps and ramps both register. Returns a length-n_grid vector.
    """
    N, G = tuning.shape
    grad = np.zeros((N, G))
    for n in range(N):
        c = tuning[n].copy()
        good = np.isfinite(c)
        if good.sum() < 3:
            continue
        c = np.interp(np.arange(G), np.flatnonzero(good), c[good])
        sd = c.std()
        if sd > 0:
            grad[n] = np.abs(np.gradient((c - c.mean()) / sd))
    return grad.mean(0)


def peak_distribution(tuning):
    """Diagnostic: argmax bin of |z-scored tuning| per neuron (the PI's 'peak')."""
    N, G = tuning.shape
    peaks = np.full(N, np.nan)
    for n in range(N):
        c = tuning[n]
        good = np.isfinite(c)
        if good.sum() < 3 or c[good].std() == 0:
            continue
        z = (c - np.nanmean(c)) / np.nanstd(c)
        peaks[n] = np.nanargmax(np.abs(z))
    return peaks


# ============================================================================
# Density -> bin edges
# ============================================================================

def edges_from_density(density, centers, rng, n_bins, *, floor=0.2):
    """Place ``n_bins`` one-hot bins so each carries equal cumulative tuning
    density: narrow bins where tuning changes fast, wide where it is flat.

    ``floor`` mixes in a uniform component (fraction of mean density) so empty
    regions still get *some* coverage and edges stay monotonic.
    """
    d = np.asarray(density, float)
    d = d + floor * (d.mean() if d.mean() > 0 else 1.0)
    cdf = np.concatenate([[0.0], np.cumsum(d)])
    cdf /= cdf[-1]
    grid = np.linspace(0, 1, len(centers) + 1)              # normalised bin edges
    targets = np.linspace(0, 1, n_bins + 1)
    norm_edges = np.interp(targets, cdf, grid)              # equal-density edges in [0,1]
    lo, hi = rng
    raw_edges = lo + norm_edges * (hi - lo)
    raw_edges[0], raw_edges[-1] = -np.inf, np.inf
    return raw_edges


def derive_adaptive_edges(table, variables, derive_mask, *, n_bins=10, n_grid=50):
    """Derive data-adaptive one-hot edges for each variable from ``derive_mask``
    rows. Returns (edges_dict, diagnostics_dict)."""
    edges, diag = {}, {}
    for var in variables:
        tuning, centers, rng = tuning_curves(table, var, derive_mask, n_grid=n_grid)
        dens = structure_density(tuning)
        edges[var] = edges_from_density(dens, centers, rng, n_bins)
        diag[var] = {'density': dens, 'centers': centers, 'range': rng,
                     'peaks': peak_distribution(tuning)}
    return edges, diag


def adaptive_encoders(adaptive_edges, *, others='matched_linear', table=None,
                      fit_mask=None, n_bins=10):
    """Assemble an encoders dict: data-adaptive one-hot edges for the adaptive
    variables, and a baseline scheme for the rest (speed/acc, and any continuous
    variable not given adaptive edges)."""
    base_scheme = cv.make_scheme(others)
    rest = {v: spec for v, spec in base_scheme.items() if v not in adaptive_edges}
    enc = cv.fit_encoders(table, rest, fit_mask=fit_mask, n_bins=n_bins) if rest else {}
    for var, e in adaptive_edges.items():
        enc[var] = ('onehot', e)
    return enc


# ============================================================================
# Split-half driver
# ============================================================================

DEFAULT_ADAPTIVE_VARS = ('goal_progress', 'goal_progress_distance',
                         'time_from_reward', 'time_to_reward',
                         'distance_from_reward', 'distance_to_reward')


def _half_split(iid, seed):
    legs = np.unique(iid)
    np.random.default_rng(seed).shuffle(legs)
    h1 = set(legs[: len(legs) // 2])
    return np.array([i in h1 for i in iid])


def run_peak_binning(table, *, variables=DEFAULT_ADAPTIVE_VARS, n_bins=10,
                     n_grid=50, n_folds=5, ridge=1e-6, seed=0, others='matched_linear'):
    """De-circularised peak-binning CPD.

    Derive adaptive edges on half-1 legs, score held-out CPD on half-2 (and swap),
    then average the two per-neuron CPD estimates. Returns:
      'cpd'    : {regressor: (N,) CPD}     (mean of the two disjoint halves)
      'r2'     : (N,) mean held-out full-model R^2
      'names'  : regressor names
      'edges'  : {('h1'|'h2'): adaptive_edges}
      'diag'   : {('h1'|'h2'): diagnostics}  (density / peaks per variable)
      'scheme' : 'adaptive_peak'
    """
    iid = np.asarray(table['interval_id'])
    in_h1 = _half_split(iid, seed)
    halves = {'h1': in_h1, 'h2': ~in_h1}

    cpd_acc, r2_acc, names, edges_out, diag_out, n_used = None, [], None, {}, {}, 0
    # derive on one half, score on the other
    for derive_key, score_key in (('h1', 'h2'), ('h2', 'h1')):
        d_mask, s_mask = halves[derive_key], halves[score_key]
        edges, diag = derive_adaptive_edges(table, variables, d_mask,
                                            n_bins=n_bins, n_grid=n_grid)
        edges_out[derive_key], diag_out[derive_key] = edges, diag

        sub = cv_subset_table(table, s_mask)
        enc = adaptive_encoders(edges, others=others, table=sub, n_bins=n_bins)
        res = cv.cpd_cv(sub, scheme=others, n_folds=n_folds, n_bins=n_bins,
                        ridge=ridge, seed=seed, encoders=enc)
        if cpd_acc is None:
            names = res['names']
            cpd_acc = {g: np.zeros_like(res['r2']) for g in names}
        for g in names:
            cpd_acc[g] = cpd_acc[g] + np.nan_to_num(res['cpd'][g])
        r2_acc.append(res['r2'])
        n_used += 1

    cpd = {g: cpd_acc[g] / n_used for g in names}
    r2 = np.nanmean(np.vstack(r2_acc), axis=0)
    return {'cpd': cpd, 'r2': r2, 'names': names, 'edges': edges_out,
            'diag': diag_out, 'scheme': 'adaptive_peak'}


def cv_subset_table(table, mask):
    """Return a copy of ``table`` keeping only ``mask`` samples (FR is (N, T))."""
    out = {}
    for k, v in table.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(mask):
            out[k] = v[mask]
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == len(mask):
            out[k] = v[:, mask]
        else:
            out[k] = v
    return out


# ============================================================================
# Plotting
# ============================================================================

def plot_adaptive_bins(diag, var, ax=None, half='h1'):
    """Show the tuning-structure density, the derived bin edges, and the peak
    histogram for one variable, so the PI can see where resolution was placed."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()

    d = diag[half][var]
    dens, centers, rng = d['density'], d['centers'], d['range']
    edges = edges_from_density(dens, centers, rng, n_bins=10)
    # map edges back to normalised [0,1] for plotting against centers
    lo, hi = rng
    norm_edges = np.clip((edges[1:-1] - lo) / (hi - lo), 0, 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 2.4))
    ax.plot(centers, dens / dens.max(), color='steelblue', lw=1.2, label='tuning-change density')
    for e in norm_edges:
        ax.axvline(e, color='crimson', lw=0.8, alpha=0.7)
    peaks = d['peaks']
    peaks = peaks[np.isfinite(peaks)] / (len(centers) - 1)
    if peaks.size:
        ax.hist(peaks, bins=20, range=(0, 1), color='0.8', density=True,
                alpha=0.6, label='peak (argmax) distribution')
    ax.set_xlabel(f'{var} (normalised)')
    ax.set_ylabel('density (a.u.)')
    ax.set_title(f'adaptive bins — {var}  (derived on {half})')
    ax.legend(frameon=False, fontsize=6)
    return ax
