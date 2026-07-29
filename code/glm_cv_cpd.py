"""glm_cv_cpd.py — cross-validated, fair-basis Coefficient of Partial Determination.

Companion to ``glm_analysis_v2.py`` and ``time_vs_progress_dissociation.py``.

Why this module exists
----------------------
The headline time-vs-goal-progress (GP) comparison in
``glm_analysis_v2.run_glm_analysis`` rests on a per-neuron CPD / nested-F that has
**three coupled problems**:

1. **Binning-placement asymmetry.** ``time_*`` and ``distance_*`` are quantile-binned
   (deciles, :func:`glm_analysis_v2.compute_decile_edges`) while ``goal_progress`` is
   equal-width. Quantile binning concentrates resolution where a variable's samples
   are dense, so the two frames get *different effective flexibility* — which biases the
   in-sample variance split between collinear regressors.
2. **No cross-validation.** CPD is computed on the full in-sample fit
   (``glm_analysis_v2.py`` L1149), so a better-placed / finer basis *mechanically*
   reduces RSS and inflates CPD. This is the larger driver and re-binning alone does
   not fix it.
3. **Within-leg collinearity.** GP ≈ time/D within a leg, so no binning scheme can
   cleanly *adjudicate* the two; only duration variability can (that is the job of
   ``time_vs_progress_dissociation.py``).

This module attacks (1) and (2). It builds the *same* design matrix the GLM builds
(reusing the ``glm_analysis_v2`` encoding primitives) but

* lets every continuous variable share **one** encoding scheme (``matched_*``), so the
  comparison is flexibility-fair, and
* computes CPD on **held-out interval folds** (cross-validated), so finer bins cannot
  win mechanically.

It operates on the long-format ``table`` produced by
:func:`time_vs_progress_dissociation.build_design_tables` (one per recday), so the same
pooled / filtered / downsampled samples feed every analysis in this package
(simulation, CV-CPD, peak-binning, OOD report).

Public API
----------
``make_scheme(name)``           -> per-variable encoding spec
``fit_encoders(table, scheme)`` -> fitted edges/ranges (fit on train rows only)
``apply_encoders(table, enc)``  -> (X, groups, names)
``cpd_insample(X, Y, groups)``  -> in-sample CPD (what the GLM reports)
``cpd_cv(table, scheme=...)``   -> cross-validated CPD per neuron per regressor group
``run_cv_cpd(tables, scheme)``  -> loop recdays, return per-recday + pooled results
"""

from __future__ import annotations

import numpy as np

import glm_analysis_v2 as glm


# ============================================================================
# Variable inventory
# ============================================================================
# Continuous regressors and the `table` key holding their raw continuous values.
# `goal_progress` / `goal_progress_distance` are read from phase_time / phase_dist,
# which are already normalised to [0, 1] (NaN in phase_dist where the animal did
# not move). All others are read in their native units.
_CONT_VARS = {
    'goal_progress':          'phase_time',
    'goal_progress_distance': 'phase_dist',
    'speed':                  'speed',
    'acceleration':           'acc',
    'time_from_reward':       't',
    'time_to_reward':         'time_to',
    'distance_from_reward':   'dist_from',
    'distance_to_reward':     'dist_to',
}

# Variables already on a fixed [0, 1] support (do not percentile-clip these).
_UNIT_RANGE_VARS = {'goal_progress', 'goal_progress_distance'}

# Canonical column order (matches glm_analysis_v2._CANONICAL_ORDER). Distance-GP
# sits next to time-GP; spatial / kinematic nuisances first.
_CANONICAL_ORDER = [
    'place', 'head_direction',
    'goal_progress', 'goal_progress_distance',
    'speed', 'acceleration',
    'time_from_reward', 'time_to_reward',
    'distance_from_reward', 'distance_to_reward',
]

N_PLACE = 21   # nodes 1..21
N_HD = 36      # 10-degree bins


# ============================================================================
# Encoding schemes
# ============================================================================
# A scheme maps each continuous variable -> (kind, spacing):
#   kind='onehot', spacing='quantile' : decile (equal-occupancy) one-hot bins
#   kind='onehot', spacing='linear'   : equal-width one-hot bins
#   kind='rc',     spacing='linear'   : raised-cosine bumps, evenly spaced
#   kind='rc',     spacing='log'      : raised-cosine bumps, log-spaced (fine near 0)
#
# Named presets:
#   'glm_onehot'   reproduces the current GLM default — MISMATCHED (deciles for
#                  time/dist/speed/acc, equal-width for GP). This is the baseline the
#                  bias concern is about.
#   'glm_rc'       reproduces continuous_basis='raised_cosine' — still asymmetric
#                  spacing (log for reward-relative variables, linear for GP).
#   'matched_linear'   every continuous variable gets equal-width one-hot bins.
#   'matched_quantile' every continuous variable gets decile one-hot bins.
#   'matched_rc'       every continuous variable gets linear raised-cosine bumps.
# The three matched_* schemes are flexibility-fair: GP and time/distance are encoded
# identically, so neither can win the variance split on binning placement alone.

_GLM_ONEHOT = {
    'goal_progress':          ('onehot', 'linear'),
    'goal_progress_distance': ('onehot', 'linear'),
    'speed':                  ('onehot', 'quantile'),
    'acceleration':           ('onehot', 'quantile'),
    'time_from_reward':       ('onehot', 'quantile'),
    'time_to_reward':         ('onehot', 'quantile'),
    'distance_from_reward':   ('onehot', 'quantile'),
    'distance_to_reward':     ('onehot', 'quantile'),
}

_GLM_RC = {
    'goal_progress':          ('rc', 'linear'),
    'goal_progress_distance': ('rc', 'linear'),
    'speed':                  ('rc', 'linear'),
    'acceleration':           ('rc', 'linear'),
    'time_from_reward':       ('rc', 'log'),
    'time_to_reward':         ('rc', 'log'),
    'distance_from_reward':   ('rc', 'log'),
    'distance_to_reward':     ('rc', 'log'),
}

_NAMED_SCHEMES = {
    'glm_onehot':       _GLM_ONEHOT,
    'glm_rc':           _GLM_RC,
    'matched_linear':   {v: ('onehot', 'linear')   for v in _CONT_VARS},
    'matched_quantile': {v: ('onehot', 'quantile') for v in _CONT_VARS},
    'matched_rc':       {v: ('rc', 'linear')       for v in _CONT_VARS},
}


def make_scheme(name):
    """Return a {variable: (kind, spacing)} dict for a named preset.

    See module docstring for the catalogue. Raises on unknown names.
    """
    if name not in _NAMED_SCHEMES:
        raise ValueError(f"unknown scheme {name!r}; choose from {list(_NAMED_SCHEMES)}")
    return dict(_NAMED_SCHEMES[name])


def list_schemes():
    return list(_NAMED_SCHEMES)


# ============================================================================
# Encoder fitting / application
# ============================================================================

def _linear_edges(values, n_bins, unit_range):
    """Equal-width one-hot edges. Open at both ends so out-of-range clips in."""
    if unit_range:
        lo, hi = 0.0, 1.0
    else:
        v = values[np.isfinite(values)]
        if v.size < n_bins * 2:
            return None
        lo, hi = np.percentile(v, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def fit_encoders(table, scheme, *, fit_mask=None, n_bins=10):
    """Fit per-variable bin edges / basis ranges from (training) rows only.

    Quantile edges and percentile ranges are computed from ``fit_mask`` rows so
    that cross-validation never leaks test-set statistics into the encoding.
    ``place`` and ``head_direction`` need no fitting (fixed bins) and are omitted.

    Returns a dict {variable: encoder}, where encoder is one of
      ('onehot', edges)                          for one-hot variables
      ('rc', spacing, (lo, hi))                  for raised-cosine variables
    """
    enc = {}
    for var, (kind, spacing) in scheme.items():
        key = _CONT_VARS[var]
        vals = np.asarray(table[key], float)
        v_fit = vals if fit_mask is None else vals[fit_mask]
        unit = var in _UNIT_RANGE_VARS

        if kind == 'onehot':
            if spacing == 'quantile':
                edges = glm.compute_decile_edges(v_fit, n_bins=n_bins)
            elif spacing == 'linear':
                edges = _linear_edges(v_fit, n_bins, unit)
            else:
                raise ValueError(f"onehot spacing must be quantile|linear, got {spacing!r}")
            if edges is None:                      # too few finite samples
                edges = np.linspace(0, 1, n_bins + 1)
                edges[0], edges[-1] = -np.inf, np.inf
            enc[var] = ('onehot', edges)

        elif kind == 'rc':
            if unit:
                rng = (0.0, 1.0)
            else:
                finite = v_fit[np.isfinite(v_fit)]
                rng = tuple(np.percentile(finite, [1, 99])) if finite.size else (0.0, 1.0)
            enc[var] = ('rc', spacing, rng)
        else:
            raise ValueError(f"kind must be onehot|rc, got {kind!r}")
    return enc


def apply_encoders(table, encoders, *, n_bins=10, include_hd=None):
    """Build the design matrix from fitted encoders.

    No intercept; all bins kept per block (matches the GLM 'all_bins'
    parameterisation, rank-deficient but min-norm / ridge solved). NaN samples
    are zeroed (a NaN row contributes no signal, e.g. phase_dist when the animal
    did not move).

    Returns
    -------
    X       : (T, p) design matrix
    groups  : {regressor_name: [column indices]}
    names   : list of regressor names actually included (canonical order)
    """
    if include_hd is None:
        include_hd = bool(table.get('has_hd', np.isfinite(np.asarray(table['hd'])).any()))

    locs = np.asarray(table['locs'], int)
    place = (locs[:, None] == np.arange(1, N_PLACE + 1)).astype(float)

    blocks = {'place': place}
    if include_hd:
        hd = np.asarray(table['hd'], float)
        idx = np.clip(np.floor((hd % 360) / 10).astype(int), 0, N_HD - 1)
        H = (idx[:, None] == np.arange(N_HD)).astype(float)
        H[~np.isfinite(hd)] = 0.0
        blocks['head_direction'] = H

    for var, encoder in encoders.items():
        vals = np.asarray(table[_CONT_VARS[var]], float)
        if encoder[0] == 'onehot':
            blocks[var] = glm.apply_onehot(vals, encoder[1])
        else:                                       # ('rc', spacing, range)
            _, spacing, rng = encoder
            blocks[var] = glm.make_raised_cosine_basis(
                vals, n_basis=n_bins, spacing=spacing, value_range=rng)

    names = [n for n in _CANONICAL_ORDER if n in blocks]
    cols, groups, cursor = [], {}, 0
    for n in names:
        b = blocks[n]
        groups[n] = list(range(cursor, cursor + b.shape[1]))
        cols.append(b)
        cursor += b.shape[1]
    X = np.nan_to_num(np.column_stack(cols))
    return X, groups, names


# ============================================================================
# Solvers
# ============================================================================

def _solve(X, Y, ridge=0.0):
    """Least squares (ridge optional). Y may be (T,) or (T, N). Returns beta.

    The all-bins design is rank-deficient; ridge=0 returns the min-norm lstsq
    solution (matches the GLM). A small ridge stabilises out-of-fold prediction
    in cross-validation without materially shrinking the fit.
    """
    if ridge and ridge > 0:
        A = X.T @ X
        A[np.diag_indices_from(A)] += ridge * (np.trace(A) / A.shape[0] + 1e-12)
        return np.linalg.solve(A, X.T @ Y)
    return np.linalg.lstsq(X, Y, rcond=None)[0]


# ============================================================================
# In-sample CPD (reproduces the GLM's metric on a chosen scheme)
# ============================================================================

def cpd_insample(X, Y, groups, *, ridge=0.0):
    """In-sample CPD per regressor group, vectorised over neurons.

    CPD_g = (RSS_reduced(g) - RSS_full) / RSS_reduced(g), the fraction of the
    reduced model's residual variance uniquely explained by group ``g``.

    Parameters
    ----------
    X : (T, p) design matrix
    Y : (T, N) firing rates (one column per neuron)
    groups : {name: [col indices]}

    Returns
    -------
    cpd : {name: (N,) array}
    r2_full : (N,) full-model R^2
    """
    Y = np.asarray(Y, float)
    beta = _solve(X, Y, ridge)
    rss_full = ((Y - X @ beta) ** 2).sum(0)
    tss = ((Y - Y.mean(0)) ** 2).sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2_full = np.where(tss > 0, 1.0 - rss_full / tss, 0.0)

    cpd = {}
    for name, idx in groups.items():
        Xr = np.delete(X, idx, axis=1)
        br = _solve(Xr, Y, ridge)
        rss_r = ((Y - Xr @ br) ** 2).sum(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            cpd[name] = np.where(rss_r > 0, (rss_r - rss_full) / rss_r, 0.0)
    return cpd, r2_full


# ============================================================================
# Cross-validated CPD (interval folds)
# ============================================================================

def _interval_folds(iid, n_folds, rng):
    """Assign each sample to a fold by its interval (leg), so whole legs are
    held out together (samples within a leg are strongly autocorrelated)."""
    legs = np.unique(iid)
    rng.shuffle(legs)
    leg_fold = {leg: k % n_folds for k, leg in enumerate(legs)}
    return np.array([leg_fold[i] for i in iid])


def cpd_cv(table, scheme='matched_linear', *, n_folds=5, n_bins=10,
           ridge=1e-6, seed=0, include_hd=None, neurons=None, encoders=None):
    """Cross-validated CPD per neuron per regressor group.

    For each interval-level fold: fit encoders on the TRAIN rows only (quantile
    edges / ranges never see the test leg), fit the full + per-group reduced
    models on train, predict the held-out test rows, and accumulate test squared
    error. CPD/R^2 are computed from squared error pooled across folds.

    ``encoders`` : if given, these FIXED encoders are used for every fold instead
    of refitting per fold. Use this when the bin edges were derived from data
    disjoint from ``table`` (e.g. the de-circularised peak-binning), so they must
    not be re-estimated from the scoring rows. Only the regression betas are then
    cross-validated.

    Returns a dict:
      'cpd'    : {regressor_name: (N,) held-out CPD}
      'r2'     : (N,) held-out R^2 of the full model
      'names'  : regressor names (canonical order)
      'scheme' : scheme name
    A negative held-out R^2 (full model worse than the mean) is reported as-is.
    """
    if isinstance(scheme, str):
        scheme_name, scheme = scheme, make_scheme(scheme)
    else:
        scheme_name = '<custom>'
    fixed = encoders is not None

    Y = np.asarray(table['FR'], float).T               # (T, N)
    iid = np.asarray(table['interval_id'])
    rng = np.random.default_rng(seed)
    fold_of = _interval_folds(iid, n_folds, rng)

    if neurons is not None:
        Y = Y[:, neurons]
    N = Y.shape[1]

    # Probe column layout once (encoders fitted on all rows) to size accumulators.
    enc0 = encoders if fixed else fit_encoders(table, scheme, n_bins=n_bins)
    _, groups0, names = apply_encoders(table, enc0, n_bins=n_bins, include_hd=include_hd)

    sse_full = np.zeros(N)
    sse_red = {g: np.zeros(N) for g in names}
    sse_null = np.zeros(N)

    for k in range(n_folds):
        tr = fold_of != k
        te = ~tr
        if te.sum() == 0 or tr.sum() < 50:
            continue
        # fit_encoders reads full-length arrays + a mask, so pass the original
        # table and the train mask (keeps NaN handling identical). Skip refitting
        # when fixed encoders were supplied (edges derived from disjoint data).
        enc = enc0 if fixed else fit_encoders(table, scheme, fit_mask=tr, n_bins=n_bins)
        X, groups, _ = apply_encoders(table, enc, n_bins=n_bins, include_hd=include_hd)
        Xtr, Xte = X[tr], X[te]
        Ytr, Yte = Y[tr], Y[te]

        beta = _solve(Xtr, Ytr, ridge)
        sse_full += ((Yte - Xte @ beta) ** 2).sum(0)
        sse_null += ((Yte - Ytr.mean(0)) ** 2).sum(0)   # train-mean predictor
        for g, idx in groups.items():
            Xr_tr = np.delete(Xtr, idx, axis=1)
            Xr_te = np.delete(Xte, idx, axis=1)
            br = _solve(Xr_tr, Ytr, ridge)
            sse_red[g] += ((Yte - Xr_te @ br) ** 2).sum(0)

    with np.errstate(divide='ignore', invalid='ignore'):
        cpd = {g: np.where(sse_red[g] > 0, (sse_red[g] - sse_full) / sse_red[g], 0.0)
               for g in names}
        r2 = np.where(sse_null > 0, 1.0 - sse_full / sse_null, np.nan)

    return {'cpd': cpd, 'r2': r2, 'names': names, 'scheme': scheme_name}


# ============================================================================
# Recday loop / pooling
# ============================================================================

def run_cv_cpd(tables, scheme='matched_linear', *, n_folds=5, n_bins=10,
               ridge=1e-6, seed=0, verbose=True):
    """Run :func:`cpd_cv` for every recday and pool neurons across recdays.

    Returns
    -------
    per_recday : {recday: cpd_cv(...) result}
    pooled     : {'cpd': {name: (Ntot,) array}, 'r2': (Ntot,), 'recday': (Ntot,) labels,
                  'names': [...], 'scheme': name}
    """
    per_recday = {}
    pooled_cpd, pooled_r2, pooled_lbl = None, [], []
    names_ref = None

    for mr, table in tables.items():
        res = cpd_cv(table, scheme=scheme, n_folds=n_folds, n_bins=n_bins,
                     ridge=ridge, seed=seed)
        per_recday[mr] = res
        if names_ref is None:
            names_ref = res['names']
            pooled_cpd = {g: [] for g in names_ref}
        for g in names_ref:
            pooled_cpd[g].append(res['cpd'].get(g, np.full_like(res['r2'], np.nan)))
        pooled_r2.append(res['r2'])
        pooled_lbl.append(np.full(res['r2'].shape[0], mr, dtype=object))
        if verbose:
            n = res['r2'].shape[0]
            print(f"  {mr}: {n} neurons, median full-model held-out R^2 = "
                  f"{np.nanmedian(res['r2']):.3f}")

    pooled = {
        'cpd': {g: np.concatenate(v) for g, v in pooled_cpd.items()},
        'r2': np.concatenate(pooled_r2),
        'recday': np.concatenate(pooled_lbl),
        'names': names_ref,
        'scheme': scheme if isinstance(scheme, str) else '<custom>',
    }
    return per_recday, pooled
