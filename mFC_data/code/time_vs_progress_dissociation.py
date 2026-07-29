"""
Dissociating elapsed-time from goal-progress coding by out-of-distribution
(cross-condition) generalization.

Companion to glm_analysis_v2.py. The in-sample CPD / nested-F machinery there
cannot cleanly adjudicate elapsed-time vs goal-progress because the two frames
are collinear within an interval and CPD splits shared variance arbitrarily.
This module adds the decisive test: fit a reference frame on SHORT inter-reward
legs and predict held-out LONG legs (and the reverse). Whichever frame
extrapolates across durations is the cell's code. Interval-duration variability
(LEC ~6x, PFC ~4.7x p90/p10) is what gives the leverage.

Two contrasts (per the user's request):
  - 'temporal'  : absolute elapsed time   vs  temporal phase  (t / D)
  - 'spatial'   : absolute elapsed time   vs  spatial progress (distance fraction)

Two reference frames per contrast:
  - ABS         : raised-cosine bases on time_from_reward AND time_to_reward.
                  Both alignments are included so fixed-latency-after-reward and
                  fixed-latency-before-reward fields are both representable, and
                  so the short->long extrapolation is symmetric (a pre-reward
                  field late in a long leg sits at a small time-to-reward, which
                  is in the short-leg training range).
  - PHASE       : raised-cosine bases on t/D in [0,1]  (contrast 'temporal'),
                  or on distance-fraction in [0,1]     (contrast 'spatial').

Both frames sit on top of a shared nuisance block K (place, head direction,
speed, acceleration, peri-reward event kernels; plus distance bases for the
'temporal' contrast only) so neither frame can win by proxying a nuisance.

Headline per cell: a held-out pseudo-R^2 for each frame under the OOD split, and
the signed difference  delta = R2_abs - R2_phase  (>0 favours elapsed time,
<0 favours progress), with an interval-bootstrap CI.

Population object (contrast 'temporal' only): a continuous warp
  c_alpha = t / D**alpha     (alpha=0 -> elapsed time, alpha=1 -> temporal phase)
fit by grid search on cross-validated held-out fit. The per-region histogram of
alpha* (+ a Hartigan dip test) says whether cells form a time/phase dichotomy or
a continuum.

Gaussian (least-squares on spike counts; fast, vectorised across neurons) is the
workhorse; Poisson (ridge-IRLS, log link) is provided as a per-neuron cross-check
on the headline. Spike counts are integer per 25-ms bin in both regions, so both
likelihoods are valid; the constant exposure is absorbed by the intercept.

Reuses from glm_analysis_v2 (same directory): get_sessions_for_glm,
prepare_session_data, truncate_all_arrays, downsample_session_data,
make_raised_cosine_basis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from glm_analysis_v2 import (
    get_sessions_for_glm,
    prepare_session_data,
    truncate_all_arrays,
    downsample_session_data,
    make_raised_cosine_basis,
)


# ============================================================================
# Palette (GridMaze Colors skill)
# ============================================================================

FRAME_COLORS = {
    'abs':        '#C03030',   # elapsed time (warm)
    'phase_time': '#2A6FB5',   # temporal phase (cool)
    'phase_dist': '#88B04B',   # spatial progress (green)
}
CLASS_COLORS = {
    'time':             '#BE3455',
    'temporal_phase':   '#0F4C81',
    'spatial_progress': '#88B04B',
    'ambiguous':        '#B4B2A9',   # Stone
}
REGION_COLORS = {'LEC': '#6667AB', 'PFC': '#FFA500'}   # Very Peri / Saffron
NULL_GREY = '#555555'
NEUTRAL   = '#2C2C2A'   # Caviar


# ============================================================================
# A. Long-format design tables
# ============================================================================

def build_design_tables(mouse_recdays, data_dic, *,
                        downsample_factor=10,
                        filter_correct_paths=False,
                        max_transition_seconds=60,
                        bin_size_ms=25,
                        min_sessions=2,
                        verbose=True):
    """Pool every recday into one long-format table of per-sample covariates.

    Mirrors the pooling logic in `glm_analysis_v2.run_glm_analysis`: per session,
    `prepare_session_data` -> `truncate_all_arrays` -> `downsample_session_data`,
    keep nodes (Locs <= 21) intersected with the transition-filter mask, then
    concatenate across sessions. Adds a per-sample `interval_id` (which inter-
    reward leg the sample belongs to) for interval-level CV / bootstrap splits.

    Returns
    -------
    tables : dict {recday: table_dict}
        table_dict keys (all length-T 1-D arrays unless noted):
          t           elapsed time since last reward (s)
          time_to     time until next reward (s)
          D           leg duration (s) = t + time_to
          phase_time  t / D in [0, 1]
          phase_dist  distance fraction to next goal in [0, 1] (NaN if no motion)
          dist_from   path length since last reward
          dist_to     path length to next reward
          speed, acc  kinematics
          locs        node id (1..21)
          hd          head direction (deg) or NaN array
          interval_id globally-unique leg index
          FR          spike counts [n_neurons x T]   *** 2-D ***
          has_hd      bool (False when HD is entirely NaN, e.g. PFC)
          bin_size_ms int
    """
    binsec = bin_size_ms / 1000.0
    tables = {}

    for mr in tqdm(mouse_recdays, desc='Building tables', disable=not verbose):
        if mr not in data_dic:
            continue
        sessions, _ = get_sessions_for_glm(data_dic[mr])
        if len(sessions) < min_sessions:
            continue

        any_filter = filter_correct_paths or max_transition_seconds is not None
        cols = defaultdict(list)
        fr_blocks = []
        iid_offset = 0

        for sess in sessions:
            sd = data_dic[mr][sess]
            prep = prepare_session_data(
                sd,
                task=sd.get('Task') if any_filter else None,
                filter_correct_paths=filter_correct_paths,
                max_transition_seconds=max_transition_seconds,
                bin_size_ms=bin_size_ms,
            )
            prep = truncate_all_arrays(prep)
            prep = downsample_session_data(prep, downsample_factor)

            tf = np.asarray(prep['time_from_reward'], float)
            tt = np.asarray(prep['time_to_reward'], float)

            # interval_id over the FULL (pre node-filter) session: a new leg
            # starts wherever elapsed time fails to increase.
            resets = np.r_[True, np.diff(tf) <= 0]
            iid = np.cumsum(resets) - 1

            nf = np.asarray(prep['Locs']) <= 21
            vtm = prep.get('valid_transition_mask')
            if vtm is not None:
                vtm = np.asarray(vtm, bool)
                if len(vtm) == len(nf):
                    nf = nf & vtm
                elif len(vtm) > len(nf):
                    nf = nf & vtm[:len(nf)]
                else:
                    pad = np.zeros(len(nf), bool); pad[:len(vtm)] = vtm
                    nf = nf & pad

            D = tf + tt
            valid = nf & (D > 0)

            cols['t'].append(tf[valid] * binsec)
            cols['time_to'].append(tt[valid] * binsec)
            cols['D'].append(D[valid] * binsec)
            cols['phase_time'].append(np.clip(tf[valid] / D[valid], 0, 1))
            cols['phase_dist'].append(np.asarray(prep['GP_dist_continuous'], float)[valid])
            cols['dist_from'].append(np.asarray(prep['dist_from_reward'], float)[valid])
            cols['dist_to'].append(np.asarray(prep['dist_to_reward'], float)[valid])
            cols['speed'].append(np.asarray(prep['Speed'], float)[valid])
            cols['acc'].append(np.asarray(prep['Acc'], float)[valid])
            cols['locs'].append(np.asarray(prep['Locs'], float)[valid])
            cols['hd'].append(np.asarray(prep['HD'], float)[valid])
            cols['interval_id'].append(iid[valid] + iid_offset)
            iid_offset = int(iid[valid].max()) + 1 + iid_offset if valid.any() else iid_offset

            fr_blocks.append(np.asarray(prep['FR'], float)[:, valid])

        if not fr_blocks or sum(b.shape[1] for b in fr_blocks) == 0:
            continue

        table = {k: np.concatenate(v) for k, v in cols.items()}
        table['FR'] = np.concatenate(fr_blocks, axis=1)
        table['has_hd'] = bool(np.isfinite(table['hd']).any())
        table['bin_size_ms'] = bin_size_ms
        tables[mr] = table
        if verbose:
            print(f"  {mr}: {table['FR'].shape[0]} neurons x {table['FR'].shape[1]} "
                  f"samples, {len(np.unique(table['interval_id']))} legs, "
                  f"has_hd={table['has_hd']}")

    return tables


# ============================================================================
# B. Design-matrix blocks
# ============================================================================

def _cosine(values, n_basis, spacing, value_range=None):
    return make_raised_cosine_basis(values, n_basis=n_basis, spacing=spacing,
                                    value_range=value_range)


def _event_basis(value_sec, window_s, n_basis):
    """Peri-event cosine kernel over [0, window_s]; zero outside the window so a
    sharp reward-locked transient is captured without leaking into the slow
    temporal/phase component."""
    B = _cosine(np.clip(value_sec, 0, window_s), n_basis, 'linear',
                value_range=(0.0, window_s))
    B[np.asarray(value_sec, float) > window_s] = 0.0
    return B


def _pct_range(values, fit_idx, lo=1, hi=99):
    v = np.asarray(values, float)[fit_idx]
    v = v[np.isfinite(v)]
    if v.size < 4:
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
    a, b = np.percentile(v, [lo, hi])
    if b <= a:
        b = a + 1e-6
    return (a, b)


def build_blocks(table, fit_idx=None, *, n_time=8, n_phase=8, n_kin=6,
                 n_event=4, event_window_s=1.0):
    """Build every named design block, evaluated on ALL samples but with the
    percentile RANGES of the continuous cosine bases taken from `fit_idx` only.

    Passing the training rows as `fit_idx` is what makes the ABS frame genuinely
    extrapolate: trained on short legs its time bases span only short-leg time,
    so long-leg samples beyond that range clip onto the last bump (a flat
    "fixed-latency" extrapolation) rather than getting fresh free parameters.

    Returns a dict of (T, k) arrays. Block groups used by `_FRAME_NAMES`:
      nuisance : place, head_direction(if has_hd), speed, acc, event_post,
                 event_pre, dist_from, dist_to
      abs      : abs_from, abs_to
      phase    : phase_time, phase_dist
    """
    T = len(table['t'])
    if fit_idx is None:
        fit_idx = np.arange(T)
    fit_idx = np.asarray(fit_idx)
    b = {}

    # --- nuisance: spatial / kinematic ---
    locs = np.asarray(table['locs'], int)
    b['place'] = (locs[:, None] == np.arange(1, 22)).astype(float)
    if table['has_hd']:
        hd = np.asarray(table['hd'], float)
        idx = np.clip(np.floor((hd % 360) / 10).astype(int), 0, 35)
        H = (idx[:, None] == np.arange(36)).astype(float)
        H[~np.isfinite(hd)] = 0.0
        b['head_direction'] = H
    b['speed'] = _cosine(table['speed'], n_kin, 'linear', _pct_range(table['speed'], fit_idx))
    b['acc']   = _cosine(table['acc'],   n_kin, 'linear', _pct_range(table['acc'],   fit_idx))

    # --- nuisance: peri-reward event kernels ---
    b['event_post'] = _event_basis(table['t'],       event_window_s, n_event)
    b['event_pre']  = _event_basis(table['time_to'], event_window_s, n_event)

    # --- nuisance: distance (temporal contrast only; see _FRAME_NAMES) ---
    b['dist_from'] = _cosine(table['dist_from'], n_kin, 'log', _pct_range(table['dist_from'], fit_idx))
    b['dist_to']   = _cosine(table['dist_to'],   n_kin, 'log', _pct_range(table['dist_to'],   fit_idx))

    # --- ABS frame: elapsed-time bases (both alignments), log-spaced ---
    b['abs_from'] = _cosine(table['t'],       n_time, 'log', _pct_range(table['t'],       fit_idx))
    b['abs_to']   = _cosine(table['time_to'], n_time, 'log', _pct_range(table['time_to'], fit_idx))

    # --- PHASE frames in [0, 1] ---
    b['phase_time'] = _cosine(table['phase_time'], n_phase, 'linear', (0.0, 1.0))
    b['phase_dist'] = _cosine(table['phase_dist'], n_phase, 'linear', (0.0, 1.0))
    return b


def _nuisance_names(table, contrast):
    names = ['place']
    if table['has_hd']:
        names.append('head_direction')
    names += ['speed', 'acc', 'event_post', 'event_pre']
    if contrast == 'temporal':
        names += ['dist_from', 'dist_to']   # distance is nuisance only here
    return names


_FRAME_BLOCKS = {
    'temporal': {'abs': ['abs_from', 'abs_to'], 'phase': ['phase_time']},
    'spatial':  {'abs': ['abs_from', 'abs_to'], 'phase': ['phase_dist']},
}


def _assemble(blocks, names):
    X = np.column_stack([blocks[n] for n in names]) if names else np.empty((len(blocks['place']), 0))
    return np.nan_to_num(X)


def _design(blocks, nuisance_names, frame_names):
    """Stack nuisance + frame blocks and prepend an intercept (column 0)."""
    X = _assemble(blocks, nuisance_names + frame_names)
    return np.column_stack([np.ones(len(X)), X])


# ============================================================================
# C. Fitting backends
# ============================================================================

def _fit_gaussian(X_tr, Y_tr):
    """Least squares with matrix RHS: solves all neurons at once. Y_tr [n, N]."""
    beta, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)
    return beta                      # [p, N]


def _fit_poisson_irls(X, y, ridge=1e-3, max_iter=100, tol=1e-7):
    """Ridge-regularised Poisson IRLS (log link). Intercept assumed at column 0
    and left unpenalised. The ridge keeps the partition-of-unity cosine /
    one-hot blocks from making X'WX singular. Returns beta."""
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(y.mean(), 1e-3))
    R = ridge * np.eye(p); R[0, 0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)
        w = np.maximum(mu, 1e-6)
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            beta_new = np.linalg.solve(XtW @ X + R, XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _poisson_predict(X, beta):
    return np.exp(np.clip(X @ beta, -20, 20))


def _gaussian_sqerr(Y_te, pred):
    return (Y_te - pred) ** 2                       # [n_te, N]


def _poisson_dev_rows(y_te, mu):
    eps = 1e-9
    mu = np.clip(mu, eps, None)
    return 2 * (y_te * np.log((y_te + eps) / mu) - (y_te - mu))   # [n_te]


# ============================================================================
# D. Out-of-distribution cross-condition test  (Gaussian, vectorised)
# ============================================================================

def _interval_split_by_duration(table, idx):
    """Median-duration split of the intervals present in `idx`. Returns
    (short_intervals, long_intervals, D_per_interval_dict)."""
    iid = table['interval_id'][idx]
    D = table['D'][idx]
    uniq = np.unique(iid)
    Dper = {iv: float(np.median(D[iid == iv])) for iv in uniq}
    med = np.median(list(Dper.values()))
    short = np.array([iv for iv in uniq if Dper[iv] <= med])
    long = np.array([iv for iv in uniq if Dper[iv] > med])
    return short, long, Dper


def _interval_folds(iid_values, n_folds, rng):
    uniq = np.unique(iid_values)
    rng.shuffle(uniq)
    fold = {iv: i % n_folds for i, iv in enumerate(uniq)}
    return np.array([fold[iv] for iv in iid_values])


def run_ood_gaussian(table, contrast, *, n_boot=200, n_folds=5, seed=0,
                     block_kw=None):
    """Per-neuron OOD (short<->long) and in-distribution CV held-out pseudo-R^2
    for both frames, vectorised across all neurons of one recday.

    Returns a DataFrame (one row per neuron) with:
      r2_abs_ood, r2_phase_ood, delta_ood (= r2_abs - r2_phase),
      ci_lo, ci_hi (interval-bootstrap 95% CI on delta_ood),
      r2_abs_cv, r2_phase_cv, delta_cv, n_samples.
    delta_ood > 0 favours elapsed time; < 0 favours progress.
    """
    block_kw = block_kw or {}
    rng = np.random.default_rng(seed)
    Y = table['FR'].T                                  # [T, N]
    N = Y.shape[1]

    # contrast 'spatial' needs finite phase_dist; restrict the analysed samples
    if contrast == 'spatial':
        idx = np.where(np.isfinite(table['phase_dist']))[0]
    else:
        idx = np.arange(len(table['t']))

    nui = _nuisance_names(table, contrast)
    fb = _FRAME_BLOCKS[contrast]
    iid_all = table['interval_id']

    short, long, _ = _interval_split_by_duration(table, idx)

    # ---- OOD: accumulate per-interval squared error for each test interval ----
    # Each interval is a test interval in exactly one direction.
    per_int_sq = {'abs': {}, 'phase': {}, 'null': {}}
    for train_iv, test_iv in [(short, long), (long, short)]:
        train_rows = idx[np.isin(iid_all[idx], train_iv)]
        test_rows = idx[np.isin(iid_all[idx], test_iv)]
        if len(train_rows) < 20 or len(test_rows) < 20:
            continue
        blocks = build_blocks(table, fit_idx=train_rows, **block_kw)
        ymean = Y[train_rows].mean(axis=0)             # [N] held-out null
        for frame in ('abs', 'phase'):
            X = _design(blocks, nui, fb[frame])
            beta = _fit_gaussian(X[train_rows], Y[train_rows])
            pred = X[test_rows] @ beta
            sq = _gaussian_sqerr(Y[test_rows], pred)   # [n_te, N]
            te_iid = iid_all[test_rows]
            for iv in np.unique(te_iid):
                m = te_iid == iv
                per_int_sq[frame].setdefault(iv, sq[m].sum(axis=0))
        sq_null = (Y[test_rows] - ymean[None, :]) ** 2
        te_iid = iid_all[test_rows]
        for iv in np.unique(te_iid):
            m = te_iid == iv
            per_int_sq['null'].setdefault(iv, sq_null[m].sum(axis=0))

    test_ivs = np.array(sorted(per_int_sq['null'].keys()))
    if len(test_ivs) < 4:
        return pd.DataFrame()
    A = np.stack([per_int_sq['abs'][iv] for iv in test_ivs])    # [n_int, N]
    P = np.stack([per_int_sq['phase'][iv] for iv in test_ivs])
    Z = np.stack([per_int_sq['null'][iv] for iv in test_ivs])

    def _delta(sel):
        nullsum = Z[sel].sum(0)
        good = nullsum > 0
        r2a = np.where(good, 1 - A[sel].sum(0) / nullsum, np.nan)
        r2p = np.where(good, 1 - P[sel].sum(0) / nullsum, np.nan)
        return r2a, r2p

    r2_abs_ood, r2_phase_ood = _delta(np.arange(len(test_ivs)))
    delta_ood = r2_abs_ood - r2_phase_ood

    boot = np.empty((n_boot, N))
    for bi in range(n_boot):
        sel = rng.integers(0, len(test_ivs), len(test_ivs))
        ra, rp = _delta(sel)
        boot[bi] = ra - rp
    ci_lo = np.nanpercentile(boot, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot, 97.5, axis=0)

    # ---- in-distribution CV (interval folds) ----
    fold_of = _interval_folds(iid_all[idx], n_folds, rng)
    cvA = np.zeros(N); cvP = np.zeros(N); cvZ = np.zeros(N)
    for f in range(n_folds):
        tr = idx[fold_of != f]; te = idx[fold_of == f]
        if len(tr) < 20 or len(te) < 20:
            continue
        blocks = build_blocks(table, fit_idx=tr, **block_kw)
        ymean = Y[tr].mean(axis=0)
        for frame, acc in (('abs', 'cvA'), ('phase', 'cvP')):
            X = _design(blocks, nui, fb[frame])
            beta = _fit_gaussian(X[tr], Y[tr])
            sq = _gaussian_sqerr(Y[te], X[te] @ beta).sum(0)
            if frame == 'abs':
                cvA += sq
            else:
                cvP += sq
        cvZ += ((Y[te] - ymean[None, :]) ** 2).sum(0)
    good = cvZ > 0
    r2_abs_cv = np.where(good, 1 - cvA / cvZ, np.nan)
    r2_phase_cv = np.where(good, 1 - cvP / cvZ, np.nan)

    return pd.DataFrame({
        'neuron': np.arange(N),
        'r2_abs_ood': r2_abs_ood, 'r2_phase_ood': r2_phase_ood,
        'delta_ood': delta_ood, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'r2_abs_cv': r2_abs_cv, 'r2_phase_cv': r2_phase_cv,
        'delta_cv': r2_abs_cv - r2_phase_cv,
        'n_samples': len(idx),
    })


def run_ood_poisson(table, contrast, *, n_folds=5, seed=0, block_kw=None,
                    neurons=None):
    """Per-neuron OOD Poisson cross-check (deviance pseudo-R^2). Loops neurons
    (IRLS is per-column). Returns DataFrame with r2_abs_ood/r2_phase_ood/
    delta_ood only — the sign is what we compare against the Gaussian result."""
    block_kw = block_kw or {}
    Y = table['FR']
    N = Y.shape[0]
    neurons = range(N) if neurons is None else neurons
    if contrast == 'spatial':
        idx = np.where(np.isfinite(table['phase_dist']))[0]
    else:
        idx = np.arange(len(table['t']))
    nui = _nuisance_names(table, contrast)
    fb = _FRAME_BLOCKS[contrast]
    iid_all = table['interval_id']
    short, long, _ = _interval_split_by_duration(table, idx)

    rows = []
    dirs = []   # (train_rows, test_rows, {frame: design}) per duration direction
    for train_iv, test_iv in [(short, long), (long, short)]:
        tr = idx[np.isin(iid_all[idx], train_iv)]
        te = idx[np.isin(iid_all[idx], test_iv)]
        if len(tr) < 20 or len(te) < 20:
            continue
        blocks = build_blocks(table, fit_idx=tr, **block_kw)
        dirs.append((tr, te, {fr: _design(blocks, nui, fb[fr]) for fr in ('abs', 'phase')}))

    if not dirs:
        return pd.DataFrame()

    for n in neurons:
        devA = devP = devZ = 0.0
        for tr, te, Xs in dirs:
            ytr, yte = Y[n, tr], Y[n, te]
            mu0 = max(ytr.mean(), 1e-6)
            devZ += _poisson_dev_rows(yte, np.full(len(yte), mu0)).sum()
            for fr, acc in (('abs', 'A'), ('phase', 'P')):
                beta = _fit_poisson_irls(Xs[fr][tr], ytr)
                mu = _poisson_predict(Xs[fr][te], beta)
                d = _poisson_dev_rows(yte, mu).sum()
                if fr == 'abs':
                    devA += d
                else:
                    devP += d
        if devZ <= 0:
            continue
        r2a = 1 - devA / devZ
        r2p = 1 - devP / devZ
        rows.append({'neuron': n, 'r2_abs_ood': r2a, 'r2_phase_ood': r2p,
                     'delta_ood': r2a - r2p})
    return pd.DataFrame(rows)


# ============================================================================
# E. Continuous warp alpha  (contrast 'temporal' only, Gaussian)
# ============================================================================

def run_alpha_gaussian(table, *, alpha_grid=None, n_folds=5, seed=0,
                       n_phase=12, n_boot=200, block_kw=None):
    """Fit the warp c_alpha = t / D**alpha for every neuron of a recday.

    For each alpha: build a cosine basis on c_alpha (range from the fold's
    training rows), full model = nuisance + basis, score by interval-fold CV
    held-out R^2 (all neurons at once). alpha* = argmax over the grid.

    The CI and the support statistics come from a LEG-LEVEL bootstrap. This is
    deliberate: within-leg 250-ms bins are strongly autocorrelated, so an
    in-sample profile-likelihood CI (which treats every bin as independent) is
    wildly over-powered — it would report alpha CIs of zero width and reject a
    true time cell's alpha=0. Resampling whole legs respects the real unit of
    independence. The per-leg held-out errors from the CV folds are cached, so
    the bootstrap just re-aggregates them (no refitting).

    Returns DataFrame: neuron, alpha_star, alpha_lo, alpha_hi (95% CI),
    support_time (boot frac with alpha*<=0), support_phase (frac with
    alpha*>=1), r2_at_star.
    """
    if alpha_grid is None:
        alpha_grid = np.round(np.arange(-0.5, 1.5 + 1e-9, 0.1), 3)
    block_kw = dict(block_kw or {})
    block_kw.pop('n_phase', None)
    rng = np.random.default_rng(seed)
    Y = table['FR'].T                                   # [T, N]
    T, N = Y.shape
    idx = np.arange(T)
    nui = _nuisance_names(table, 'temporal')
    iid = table['interval_id']
    legs = np.unique(iid)
    n_legs = len(legs)
    leg_pos = {lg: i for i, lg in enumerate(legs)}
    g = np.array([leg_pos[lg] for lg in iid])            # per-sample leg index
    fold_of = _interval_folds(iid, n_folds, rng)

    n_a = len(alpha_grid)
    t = table['t']; D = np.maximum(table['D'], 1e-6)
    # Warp bases built once (coordinate system from all data — no response
    # leakage; held-out predictions still use train-fitted betas only).
    W_all = [np.nan_to_num(_cosine(t / np.power(D, a), n_phase, 'linear',
                                   _pct_range(t / np.power(D, a), idx)))
             for a in alpha_grid]

    # Per-leg held-out squared error for each alpha [n_legs, n_a, N], plus the
    # alpha-independent per-leg null error. Nuisance built once per fold.
    dev_leg = np.zeros((n_legs, n_a, N))
    null_leg = np.zeros((n_legs, N))
    folds_ok = True
    for f in range(n_folds):
        tr = idx[fold_of != f]; te = idx[fold_of == f]
        if len(tr) < 20 or len(te) < 20:
            folds_ok = False; break
        blocks = build_blocks(table, fit_idx=tr, **block_kw)
        Xnui = _design(blocks, nui, [])                  # intercept + K
        g_te = g[te]
        np.add.at(null_leg, g_te, (Y[te] - Y[tr].mean(0)[None, :]) ** 2)
        sq_stack = np.empty((len(te), n_a, N))
        for ai in range(n_a):
            X = np.column_stack([Xnui, W_all[ai]])
            beta = _fit_gaussian(X[tr], Y[tr])
            sq_stack[:, ai, :] = (Y[te] - X[te] @ beta) ** 2
        np.add.at(dev_leg, g_te, sq_stack)

    if not folds_ok:
        return pd.DataFrame()

    def _r2_curve(sel):
        nullsum = null_leg[sel].sum(0)                   # [N]
        good = nullsum > 0
        devsum = dev_leg[sel].sum(0)                     # [n_a, N]
        return np.where(good[None, :], 1 - devsum / np.where(good, nullsum, np.nan), np.nan)

    all_legs = np.arange(n_legs)
    r2 = _r2_curve(all_legs)
    star_i = np.nanargmax(np.where(np.isnan(r2), -np.inf, r2), axis=0)
    alpha_star = alpha_grid[star_i]
    r2_at_star = r2[star_i, np.arange(N)]

    boot_star = np.empty((n_boot, N))
    for b in range(n_boot):
        sel = rng.integers(0, n_legs, n_legs)
        rb = _r2_curve(sel)
        boot_star[b] = alpha_grid[np.nanargmax(np.where(np.isnan(rb), -np.inf, rb), axis=0)]
    alpha_lo = np.nanpercentile(boot_star, 2.5, axis=0)
    alpha_hi = np.nanpercentile(boot_star, 97.5, axis=0)

    return pd.DataFrame({
        'neuron': np.arange(N),
        'alpha_star': alpha_star,
        'alpha_lo': alpha_lo, 'alpha_hi': alpha_hi,
        'support_time': np.mean(boot_star <= 0, axis=0),
        'support_phase': np.mean(boot_star >= 1, axis=0),
        'r2_at_star': r2_at_star,
    })


# ============================================================================
# F. Nulls
# ============================================================================

def temporal_tuning_pvalue(table, contrast, *, n_shuffle=100, n_folds=5, seed=0,
                           block_kw=None):
    """Circular-shift null gating "is there ANY temporal/phase structure".

    Per neuron: CV held-out R^2 of the union temporal model (nuisance + ABS +
    PHASE). Null = same statistic after globally rolling each neuron's spikes
    against the covariates (preserves rate + autocorrelation, destroys the
    time/phase relationship). Returns DataFrame: neuron, r2_real, p_value.
    """
    block_kw = block_kw or {}
    rng = np.random.default_rng(seed)
    Y = table['FR'].T
    T, N = Y.shape
    if contrast == 'spatial':
        idx = np.where(np.isfinite(table['phase_dist']))[0]
    else:
        idx = np.arange(T)
    nui = _nuisance_names(table, contrast)
    fb = _FRAME_BLOCKS[contrast]
    frame_names = fb['abs'] + fb['phase']
    fold_of = _interval_folds(table['interval_id'][idx], n_folds, rng)

    def _cv_r2(Ymat):
        dev = np.zeros(N); null = np.zeros(N)
        for f in range(n_folds):
            tr = idx[fold_of != f]; te = idx[fold_of == f]
            if len(tr) < 20 or len(te) < 20:
                continue
            blocks = build_blocks(table, fit_idx=tr, **block_kw)
            X = _design(blocks, nui, frame_names)
            beta = _fit_gaussian(X[tr], Ymat[tr])
            dev += ((Ymat[te] - X[te] @ beta) ** 2).sum(0)
            null += ((Ymat[te] - Ymat[tr].mean(0)[None, :]) ** 2).sum(0)
        good = null > 0
        return np.where(good, 1 - dev / null, np.nan)

    r2_real = _cv_r2(Y)
    null_ge = np.zeros(N)
    for _ in range(n_shuffle):
        shift = rng.integers(1, T)
        r2_null = _cv_r2(np.roll(Y, shift, axis=0))
        null_ge += (r2_null >= r2_real).astype(float)
    p = (null_ge + 1) / (n_shuffle + 1)
    return pd.DataFrame({'neuron': np.arange(N), 'r2_real': r2_real, 'p_value': p})


def hartigan_dip(x):
    """Self-contained Hartigan dip statistic (the `diptest` package is not
    installed). Returns the dip value; larger => more bimodal. Compares the
    empirical CDF to the closest unimodal CDF via the greatest-convex-minorant /
    least-concave-majorant construction."""
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return np.nan
    cdf = np.arange(1, n + 1) / n

    def gcm(xs, ys):
        # greatest convex minorant of (xs, ys)
        idx = [0]
        for i in range(1, len(xs)):
            while len(idx) >= 2:
                a, b = idx[-2], idx[-1]
                if (ys[i] - ys[b]) * (xs[b] - xs[a]) <= (ys[b] - ys[a]) * (xs[i] - xs[b]):
                    idx.pop()
                else:
                    break
            idx.append(i)
        return np.interp(xs, xs[idx], ys[idx])

    lower = gcm(x, cdf)
    upper = 1 - gcm(-x[::-1], (1 - cdf)[::-1])[::-1]
    return float(np.max(np.abs(np.minimum(np.maximum(cdf, lower), upper) - cdf)))


def dip_pvalue(x, n_boot=500, seed=0):
    """Bootstrap p-value for the dip vs a uniform-null reference of equal n."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 8:
        return np.nan, np.nan
    d = hartigan_dip(x)
    rng = np.random.default_rng(seed)
    n = x.size
    null = np.array([hartigan_dip(rng.uniform(0, 1, n)) for _ in range(n_boot)])
    return d, float((np.sum(null >= d) + 1) / (n_boot + 1))


# ============================================================================
# Population runners (loop recdays, tag region/mouse)
# ============================================================================

def run_population_ood(tables, contrast, *, family='gaussian', region='', **kw):
    """Run the OOD test across all recdays in `tables`; concatenate to one
    DataFrame with `recday`, `mouse`, `region` columns."""
    runner = run_ood_gaussian if family == 'gaussian' else run_ood_poisson
    out = []
    for mr, table in tqdm(tables.items(), desc=f'OOD {contrast}/{family}'):
        df = runner(table, contrast, **kw)
        if df.empty:
            continue
        df['recday'] = mr
        df['mouse'] = mr.split('_')[0]
        df['region'] = region
        df['contrast'] = contrast
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_population_alpha(tables, *, region='', **kw):
    out = []
    for mr, table in tqdm(tables.items(), desc='alpha warp'):
        df = run_alpha_gaussian(table, **kw)
        if df.empty:
            continue
        df['recday'] = mr; df['mouse'] = mr.split('_')[0]; df['region'] = region
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def classify(df, *, r2_floor=0.0, gate_p=None):
    """Add a `class` column from the bootstrap CI on delta_ood:
       ci_lo>0 -> 'time'; ci_hi<0 -> progress class; else 'ambiguous'.
    The progress class is 'temporal_phase' or 'spatial_progress' per contrast.

    Two gates keep an untuned cell out of the time/progress bins:
      - `r2_floor`: the WINNING frame's held-out OOD R^2 must exceed this. A
        near-zero delta can have a tight bootstrap CI (both frames resampled on
        the same intervals), so a CI excluding 0 is necessary but not sufficient
        — the winner must also explain real held-out variance. Default 0.0
        (winner must be positive); 0.01-0.02 is a stricter choice.
      - `gate_p` / a `p_value` column (from `temporal_tuning_pvalue`): cells with
        p>0.05 for any temporal/phase structure are forced to 'ambiguous'.
    """
    df = df.copy()
    prog = 'temporal_phase' if (df['contrast'].iloc[0] == 'temporal') else 'spatial_progress'
    winner_r2 = np.where(df['delta_ood'] > 0, df['r2_abs_ood'], df['r2_phase_ood'])
    cls = np.where(df['ci_lo'] > 0, 'time',
                   np.where(df['ci_hi'] < 0, prog, 'ambiguous'))
    cls = np.where(winner_r2 <= r2_floor, 'ambiguous', cls)
    if gate_p is not None:
        cls = np.where(np.asarray(gate_p) > 0.05, 'ambiguous', cls)
    elif 'p_value' in df.columns:
        cls = np.where(df['p_value'].values > 0.05, 'ambiguous', cls)
    df['class'] = cls
    return df


def population_delta_test(df):
    """Per-region summary of the OOD delta distribution — the headline answer.

    Per-cell classification is deliberately conservative (a leg-level bootstrap
    CI must exclude 0), so on a single recday most cells read 'ambiguous'. The
    region-level question "does this area lean elapsed time or progress" is
    answered by the SIGN of the pooled delta distribution, which aggregates the
    weak per-cell signal across all neurons. Returns median/mean delta, the
    fraction leaning toward time (delta>0), and a Wilcoxon signed-rank p-value
    against a zero median.
    """
    from scipy.stats import wilcoxon
    out = []
    for region, sub in df.groupby('region'):
        d = sub['delta_ood'].dropna().values
        if len(d) < 5:
            continue
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = np.nan
        out.append({'region': region, 'n': len(d),
                    'median_delta': float(np.median(d)), 'mean_delta': float(np.mean(d)),
                    'frac_leaning_time': float(np.mean(d > 0)), 'wilcoxon_p': p})
    return pd.DataFrame(out)


# ============================================================================
# G. Plots (GridMaze style — call apply_gridmaze_style() first)
# ============================================================================

def plot_ood_scatter(df, contrast, ax=None):
    prog_lbl = 'temporal phase' if contrast == 'temporal' else 'spatial progress'
    ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
    for region, sub in df.groupby('region'):
        ax.scatter(sub['r2_abs_ood'], sub['r2_phase_ood'], s=12, alpha=0.5,
                   color=REGION_COLORS.get(region, NEUTRAL), label=region or 'all')
    lo = float(min(df['r2_abs_ood'].min(), df['r2_phase_ood'].min(), 0))
    hi = float(max(df['r2_abs_ood'].max(), df['r2_phase_ood'].max(), 0.01))
    ax.plot([lo, hi], [lo, hi], '--', color=NULL_GREY, lw=1)
    ax.set_xlabel('held-out $R^2$  (elapsed time)')
    ax.set_ylabel(f'held-out $R^2$  ({prog_lbl})')
    ax.set_title(f'OOD generalization: time vs {prog_lbl}')
    ax.legend(fontsize=7, title='region')
    return ax


def plot_delta_hist(df, contrast, ax=None):
    prog_lbl = 'temporal phase' if contrast == 'temporal' else 'spatial progress'
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    regions = sorted(df['region'].unique())
    for region in regions:
        sub = df[df['region'] == region]['delta_ood'].dropna()
        ax.hist(sub, bins=40, alpha=0.55, color=REGION_COLORS.get(region, NEUTRAL),
                label=f'{region or "all"} (med={np.median(sub):+.3f})')
    ax.axvline(0, color=NEUTRAL, lw=1.2, ls='--')
    ax.set_xlabel(f'$\\Delta$ held-out $R^2$  (elapsed time $-$ {prog_lbl})')
    ax.set_ylabel('neurons')
    ax.set_title('< favours progress      |      favours time >')
    ax.legend(fontsize=7)
    return ax


def plot_alpha_hist(df, ax=None):
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    for region in sorted(df['region'].unique()):
        sub = df[df['region'] == region]['alpha_star'].dropna()
        d, p = dip_pvalue(sub)
        ax.hist(sub, bins=np.arange(-0.55, 1.6, 0.1), alpha=0.55,
                color=REGION_COLORS.get(region, NEUTRAL),
                label=f'{region or "all"} (dip p={p:.3f})')
    ax.axvline(0, color=FRAME_COLORS['abs'], lw=1.5, label='pure time ($\\alpha$=0)')
    ax.axvline(1, color=FRAME_COLORS['phase_time'], lw=1.5, label='pure phase ($\\alpha$=1)')
    ax.set_xlabel('warp $\\alpha^*$   (0 = elapsed time, 1 = temporal phase)')
    ax.set_ylabel('neurons')
    ax.set_title('Time$\\leftrightarrow$phase continuum')
    ax.legend(fontsize=7)
    return ax


def plot_classification(df, contrast, ax=None):
    prog = 'temporal_phase' if contrast == 'temporal' else 'spatial_progress'
    order = ['time', prog, 'ambiguous']
    regions = sorted(df['region'].unique())
    ax = ax or plt.subplots(figsize=(1.6 * len(regions) + 2, 4))[1]
    bottom = np.zeros(len(regions))
    for cls in order:
        fr = [np.mean(df[df['region'] == r]['class'] == cls) * 100 for r in regions]
        ax.bar(range(len(regions)), fr, bottom=bottom,
               color=CLASS_COLORS[cls], edgecolor='white', lw=0.6,
               label=cls.replace('_', ' '))
        bottom += np.array(fr)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([f'{r}\n(n={int((df["region"]==r).sum())})' for r in regions])
    ax.set_ylabel('% of neurons')
    ax.set_ylim(0, 100)
    ax.set_title(f'Classification ({contrast})')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    return ax


def plot_gaussian_vs_poisson(df_g, df_p, ax=None, eff_thresh=0.02):
    """Gaussian vs Poisson delta scatter. Sign agreement is reported BOTH over
    all cells and over cells with a non-trivial Gaussian effect
    (|delta_g| > eff_thresh): near-zero cells have an essentially random sign so
    they drag the overall agreement toward chance — the cross-check is only
    meaningful where there is an effect to agree about."""
    m = df_g.merge(df_p, on=['recday', 'neuron'], suffixes=('_g', '_p'))
    ax = ax or plt.subplots(figsize=(4.2, 4.2))[1]
    same = np.sign(m['delta_ood_g']) == np.sign(m['delta_ood_p'])
    agree_all = 100 * np.mean(same)
    eff = np.abs(m['delta_ood_g']) > eff_thresh
    agree_eff = 100 * np.mean(same[eff]) if eff.any() else np.nan
    ax.scatter(m['delta_ood_g'][~eff], m['delta_ood_p'][~eff], s=8, alpha=0.3, color=NULL_GREY)
    ax.scatter(m['delta_ood_g'][eff], m['delta_ood_p'][eff], s=12, alpha=0.6, color=NEUTRAL)
    ax.axhline(0, color=NULL_GREY, lw=0.8); ax.axvline(0, color=NULL_GREY, lw=0.8)
    ax.set_xlabel('$\\Delta R^2$ Gaussian'); ax.set_ylabel('$\\Delta R^2$ Poisson')
    ax.set_title(f'sign agree: {agree_all:.0f}% all, '
                 f'{agree_eff:.0f}% for $|\\Delta|>${eff_thresh} (n={int(eff.sum())})')
    return ax


def plot_peak_latency_vs_duration(table, neuron, *, n_terciles=3, frame='abs',
                                  block_kw=None, ax=None):
    """Model-derived peak latency vs leg duration (the interpretable figure).

    Fits the ABS frame within each duration tercile (trial-count matched by
    subsampling to the smallest tercile), takes the latency of the fitted
    elapsed-time tuning peak. A flat slope => time cell; a slope tracking a
    fixed fraction of D => progress cell. Latency is read off the FITTED curve,
    never an argmax of the raw PSTH.
    """
    block_kw = block_kw or {}
    ax = ax or plt.subplots(figsize=(4.2, 3.6))[1]
    iid = table['interval_id']
    uniq = np.unique(iid)
    Dper = np.array([np.median(table['D'][iid == iv]) for iv in uniq])
    terc = np.quantile(Dper, np.linspace(0, 1, n_terciles + 1))
    y = table['FR'][neuron]
    nui = _nuisance_names(table, 'temporal')
    durations, peaks = [], []
    counts = []
    groups = []
    for k in range(n_terciles):
        sel_iv = uniq[(Dper >= terc[k]) & (Dper <= terc[k + 1])]
        rows = np.where(np.isin(iid, sel_iv))[0]
        groups.append(rows); counts.append(len(rows))
    nmin = min(counts)
    rng = np.random.default_rng(0)
    tgrid = np.linspace(0, np.percentile(table['t'], 95), 60)
    for k, rows in enumerate(groups):
        rows = rng.choice(rows, nmin, replace=False)
        blocks = build_blocks(table, fit_idx=rows, **block_kw)
        X = _design(blocks, nui, ['abs_from'])
        beta = _fit_gaussian(X[rows], y[rows][:, None])[:, 0]
        # evaluate the abs_from tuning over a time grid holding nuisance at mean
        base = blocks['abs_from'][rows].mean(0)
        cols = []
        rng_t = _pct_range(table['t'], rows)
        B = _cosine(tgrid, block_kw.get('n_time', 8), 'log', rng_t)
        # locate abs_from columns in X (after intercept + nuisance)
        n_nui = _design(blocks, nui, []).shape[1] - 1
        af = beta[1 + n_nui: 1 + n_nui + B.shape[1]]
        curve = B @ af
        peak_t = tgrid[int(np.argmax(curve))]
        durations.append(np.median(table['D'][rows])); peaks.append(peak_t)
    ax.plot(durations, peaks, 'o-', color=FRAME_COLORS['abs'])
    ax.plot([min(durations), max(durations)],
            [0.5 * min(durations), 0.5 * max(durations)],
            '--', color=FRAME_COLORS['phase_time'], lw=1, label='phase (peak $\\propto$ D)')
    ax.set_xlabel('leg duration D (s)'); ax.set_ylabel('fitted peak latency (s)')
    ax.set_title(f'neuron {neuron}: peak latency vs duration')
    ax.legend(fontsize=7)
    return ax


# ============================================================================
# Synthetic positive controls (validation)
# ============================================================================

def append_synthetic_neurons(table, *, seed=0, peak_t=3.0, phi_star=0.6,
                             field_sigma_t=0.8, field_sigma_phi=0.12, gain=3.0,
                             base=0.1):
    """Return a copy of `table` with two simulated neurons appended to FR:
      row -2 : a TIME cell  (Gaussian field at fixed elapsed time peak_t)
      row -1 : a PHASE cell (Gaussian field at fixed temporal phase phi_star)
    Use to confirm the OOD test recovers delta_ood>0 / <0 and alpha*~0 / ~1.
    Counts are Poisson draws so both backends apply.
    """
    rng = np.random.default_rng(seed)
    t = table['t']; D = np.maximum(table['D'], 1e-6); phi = t / D
    rate_time = base + gain * np.exp(-0.5 * ((t - peak_t) / field_sigma_t) ** 2)
    rate_phase = base + gain * np.exp(-0.5 * ((phi - phi_star) / field_sigma_phi) ** 2)
    new = np.vstack([rng.poisson(rate_time), rng.poisson(rate_phase)]).astype(float)
    out = dict(table)
    out['FR'] = np.vstack([table['FR'], new])
    return out
