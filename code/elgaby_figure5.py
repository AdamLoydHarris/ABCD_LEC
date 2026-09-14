"""El-Gaby Figure 5 scoring semantics, applied to V4/V5 exports.

**This module never modifies `elasticnet_regression_v5`.** It imports it read-only, constructs
`RegressionConfigV5` instances, and re-scores stored betas. Nothing here changes the V4
defaults, its outputs, or the analysis you are running.

Why it exists
-------------
Our V4 headline and the paper's Figure 5 differ in the *scoring*, not the fit, in two ways that
turn out to dominate the neuron count:

1. **The lag set.** His `close_to_anchor_bins_30 = [0, 11]`; V4's default window excludes
   {0, 1, 11}. Dropping lag 1 roughly doubles the count.
2. **Per-fold vs per-neuron.** He has no neuron mask: he sets *that fold's* correlation to NaN
   when a top-3 beta lands in the excluded set, then `nanmean`s the survivors — so a neuron
   counts if **>= 1 fold** passes. V4 applies a per-neuron mask with a majority vote.

Both are re-derivable from a finished run: `cv_coeffs` is exported, and the test regressors are
cheap to rebuild (~0.2 s/session), so any lag set can be scored without re-fitting anything.

    import elgaby_figure5 as eg
    runs   = eg.load_run('.../elasticnet_v4_past_20260904_154516')
    table  = eg.compare_semantics(data_dic, runs)      # the reconciliation table

The correspondence with his three saved arrays:

    corrs_all             <- Predicted_Actual_correlation_mean          (no lag exclusion)
    corrs_nozero          <- Predicted_Actual_correlation_nonzero_mean          lags {0,11}
    corrs_nozero_strict   <- Predicted_Actual_correlation_nonzero_strict_mean   {0,1,2,9,10,11}

`n = 329` in the paper is the middle one.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import elasticnet_regression_v5 as v5                                    # noqa: E402

#: His exclusion sets, verbatim from cell 26.
BINS_30 = (0, 11)                          # close_to_anchor_bins_30, both branches
BINS_90 = (0, 1, 2, 9, 10, 11)             # close_to_anchor_bins_90, limited=True
NUM_MAX = 3                                # Num_max


def bins_90_for(num_lags):
    """His 90-degree set. In the `_beyond` branch cell 26 overwrites it:

        close_to_anchor_bins_90 = np.arange(12)

    so with 24 lags the strict variant excludes the ENTIRE first loop (lags 0-11) and keeps
    only the second loop back -- a materially different analysis from the 12-lag one, not a
    rescaling of it. `close_to_anchor_bins_30` is left at [0, 11] in both branches.
    """
    return tuple(range(12)) if num_lags > 12 else BINS_90


def load_run(out_dir):
    """`{recday: results dict}` from a V4 export directory."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, '*_arrays.npz'))):
        res = v5.load_regression_outputs(path)
        runs[res['mouse_recday']] = res
    if not runs:
        raise FileNotFoundError(f'no *_arrays.npz in {out_dir}')
    return runs


def _config_from(results, **overrides):
    """Rebuild the `RegressionConfigV5` a run was produced with, plus overrides.

    `results['config']` is a plain dict when the run came back from `load_regression_outputs`
    (the npz stores it as one) but the live `RegressionConfigV5` object when the results are
    still in memory from `run_cross_validated_regression_v4`. Accept both.
    """
    cfg = results.get('config')
    stored = dict(cfg.to_dict()) if hasattr(cfg, 'to_dict') else dict(cfg or {})
    init = {k: stored[k] for k in (
        'num_locations', 'num_goal_progress_bins', 'num_task_states', 'num_lags',
        'use_poisson', 'regularize', 'poisson_alpha', 'elasticnet_alpha', 'l1_ratio',
        'positive', 'num_bins_per_state', 'state_tuning_p_threshold', 'max_iter',
        'lag_direction', 'restrict_to_pref_phase', 'pref_phase_source',
        'drop_untracked_bins', 'alpha_mode', 'alpha_frac', 'state_reduce',
        'state_tuning_statistic', 'nonzero_lag_zero_lags', 'nonzero_lag_min',
        'nonzero_lag_max', 'require_positive_top3', 'nz_per_fold') if k in stored}
    init.update(overrides)
    return v5.RegressionConfigV5(**init)


def _top_n_indices(coeffs, num_max=NUM_MAX):
    """His `topN_indices`: the `num_max` largest coefficients, NaNs dropped.

    Verbatim from cell 26:
        indices_sorted = np.flip(np.argsort(coeffs))
        indices_sorted_nonan = indices_sorted[~np.isnan(coeffs[indices_sorted])]
        topN_indices = indices_sorted_nonan[:Num_max]
    """
    order = np.flip(np.argsort(coeffs))
    order = order[~np.isnan(coeffs[order])]
    return order[:num_max]


def _prepare_test_sessions(data_dic, recday, results, config):
    """Rebuild the per-fold test regressors and trial times the original fit used.

    Cheap (`generate_regressors_raw` is ~0.2 s/session) and exact, because it goes through the
    same `v5._prepare_session`. Validated by `verify_against_v4`.
    """
    preps = {}
    for s in results['used_sessions']:
        s = int(s)
        p = v5._prepare_session(data_dic[recday][s], config)
        if p is not None:
            preps[s] = p
    return preps


def score_elgaby(data_dic, recday, results, lag_bins=BINS_30, num_max=NUM_MAX, config=None,
                 preps=None):
    """Cell 26's scoring, with his semantics, on this run's stored betas.

    Per (neuron, fold): rebuild the prediction with the `lag_bins` columns removed, normalise,
    take the n=4 preferred-phase state means, correlate. Separately mark the fold excluded when
    any of the top-`num_max` betas sits in those columns.

    Returns both, so a single pass serves both semantics:
      `corrs_raw`    (n, folds)  every fold scored, no exclusion applied
      `excluded`     (n, folds)  True where a top-N beta is in the excluded columns
      `corrs_nozero` (n, folds)  `corrs_raw` with `excluded` set to NaN  -- his array
      `mean_nozero`  (n,)        nanmean of the survivors                -- his per-neuron value
      `counts`       (n,)        surviving folds; he counts a neuron when this is > 0

    `actual` comes from the stored `cv_actual_tuning` rather than being recomputed: it does not
    depend on the betas, and `verify_against_v4` proves the stored curves and the rebuilt
    regressors belong to the same fit.
    """
    config = config or _config_from(results)
    nbps, nstates = config.num_bins_per_state, config.num_task_states
    num_lags, n_reg = config.num_lags, config.num_regressors
    phase_norm = v5._phase_label_per_norm_bin(config)

    lag_axis = np.arange(n_reg) % num_lags
    excluded_cols = np.flatnonzero(np.isin(lag_axis, np.asarray(lag_bins)))

    if preps is None:
        preps = _prepare_test_sessions(data_dic, recday, results, config)
    sessions = [int(s) for s in results['used_sessions']]
    coeffs_all = results['cv_coeffs']
    prefs = results['pref_phases']
    actual_all = results['cv_actual_tuning']
    n_neurons, n_folds, _ = coeffs_all.shape

    raw = np.full((n_neurons, n_folds), np.nan)
    excl = np.zeros((n_neurons, n_folds), dtype=bool)
    for fold, sess in enumerate(sessions):
        if sess not in preps:
            continue
        test = preps[sess]
        for ni in range(n_neurons):
            c = coeffs_all[ni, fold]
            if np.all(np.isnan(c)) or prefs[ni, fold] < 0:
                continue
            if num_max and np.isin(_top_n_indices(c, num_max), excluded_cols).any():
                excl[ni, fold] = True
            actual = np.asarray(actual_all[ni, fold], dtype=float)
            if np.all(np.isnan(actual)):
                continue
            c_nz = np.array(c, dtype=float, copy=True)
            c_nz[excluded_cols] = 0.0                    # == his NaN + nansum
            pred = v5.raw_to_norm(test['regressors'] @ c_nz, test['trial_times'], config)
            if pred is None:
                continue
            r_m, r_x, *_ = v5._state_pref_corr(actual, pred, phase_norm, int(prefs[ni, fold]),
                                               nbps, nstates)
            raw[ni, fold] = r_m if config.state_reduce == 'mean' else r_x

    nozero = np.where(excl, np.nan, raw)
    with np.errstate(all='ignore'):
        mean = np.nanmean(np.where(np.isfinite(nozero), nozero, np.nan), axis=1)
        mean_raw = np.nanmean(np.where(np.isfinite(raw), raw, np.nan), axis=1)
    return {'corrs_raw': raw, 'excluded': excl, 'corrs_nozero': nozero,
            'mean_nozero': mean, 'mean_raw': mean_raw,
            'counts': np.sum(np.isfinite(nozero), axis=1)}


def score_ours(results, lag_bins=BINS_30, config=None, num_max=NUM_MAX, per_fold=True):
    """The per-neuron-mask counterpart, from the same betas.

    Implemented here rather than through `v5.identify_nonzero_lag_neurons` because that takes a
    contiguous `[min, max]` window, and the reference's sets are not contiguous complements
    once `num_lags = 24`: excluding {0, 11} of 24 leaves {1..10, 12..23}. This applies the same
    top-N test to an arbitrary excluded set, and for a contiguous case it agrees with V4 exactly
    (asserted by `verify_mask_against_v4`).

    `per_fold=True` mirrors V4's default: test each fold's own betas, majority vote, with folds
    that produced no positive beta abstaining.
    """
    config = config or _config_from(results)
    num_lags = config.num_lags
    coeffs = results['cv_coeffs']
    n_neurons, n_folds, _ = coeffs.shape
    excluded = set(int(b) % num_lags for b in lag_bins)

    def passes(c):
        c = np.nan_to_num(np.asarray(c, dtype=float), nan=0.0)
        if config.require_positive_top3:
            idx = np.flatnonzero(c > 0)
            if idx.size == 0:
                return None
            top = idx[np.argsort(c[idx])[-num_max:]]
        else:
            if np.all(c == 0):
                return None
            top = np.argsort(c)[-num_max:]
        return not any((int(t) % num_lags) in excluded for t in top)

    mask = np.zeros(n_neurons, dtype=bool)
    peak = np.full(n_neurons, np.nan)
    for ni in range(n_neurons):
        if per_fold:
            votes = [v for v in (passes(coeffs[ni, fi]) for fi in range(n_folds))
                     if v is not None]
            mask[ni] = bool(votes) and np.mean(votes) > 0.5
        else:
            v = passes(np.nanmean(coeffs[ni], axis=0))
            mask[ni] = bool(v)
        c = np.nan_to_num(np.nanmean(coeffs[ni], axis=0), nan=0.0)
        if np.any(c > 0):
            peak[ni] = float(np.argmax(c) % num_lags)
    return {'mask': mask, 'peak_lags': peak}


def verify_mask_against_v4(results, lag_bins=BINS_30):
    """Gate: on a contiguous case our arbitrary-set mask must equal V4's own.

    Only meaningful when the complement is contiguous -- i.e. the 12-lag sets -- which is
    exactly where V4's window API can express the same thing.
    """
    config = _config_from(results)
    keep = sorted(set(range(config.num_lags)) - set(lag_bins))
    if keep != list(range(keep[0], keep[-1] + 1)):
        return None                                   # not expressible in V4's API; skip
    cfg = _config_from(results, nonzero_lag_min=keep[0], nonzero_lag_max=keep[-1],
                       nonzero_lag_zero_lags=tuple(lag_bins))
    theirs, _ = v5.identify_nonzero_lag_neurons(results, cfg)
    ours = score_ours(results, lag_bins, config=config)['mask']
    n_diff = int((ours != theirs).sum())
    if n_diff:
        raise AssertionError(f'{results["mouse_recday"]}: arbitrary-set mask differs from V4 '
                             f'on {n_diff} neurons for lag_bins={lag_bins}')
    return n_diff


def verify_against_v4(data_dic, recday, results, tol=1e-5):
    """Gate: re-derived test regressors must reproduce V4's own `corrs_nonzero`.

    Scores with `lag_bins` equal to the run's own `nonzero_lag_zero_lags` and *without* the
    fold exclusion, which is precisely what V4 stored. Any mismatch means the rebuilt
    regressors are not the ones the original fit used, and every number below is void.

    `tol` is 1e-5 because `export_regression_outputs` downcasts the `corrs*` arrays to float32,
    so the round trip alone costs ~1e-6 on correlations near 1 (observed worst: 5.3e-07 over 25
    PFC recdays from disk, 1.1e-06 on one recday scored in memory). A genuine regressor
    mismatch is of order 0.1-1, i.e. four orders of magnitude clear of this.
    """
    config = _config_from(results)
    got = score_elgaby(data_dic, recday, results,
                       lag_bins=tuple(config.nonzero_lag_zero_lags),
                       num_max=0, config=config)['corrs_raw']
    want = np.asarray(results['corrs_nonzero'], dtype=float)
    both = np.isfinite(got) & np.isfinite(want)
    if not both.any():
        return np.nan, 0
    diff = float(np.max(np.abs(got[both] - want[both])))
    if diff > tol:
        raise AssertionError(
            f'{recday}: re-derived corrs_nonzero differs from the stored one by {diff:.3g} '
            f'(> tol={tol:g}) over {int(both.sum())} cells; the rebuilt test regressors do not '
            f'match the fitted ones, so nothing downstream is valid.')
    return diff, int(both.sum())


def compare_semantics(data_dic, runs, lag_sets=((0, 11), (0, 1, 11)),
                      tuning_key='state_tuned_mask', verbose=True):
    """The reconciliation table: n, mean r, frac>0 and p under each lag set x semantics."""
    import pandas as pd
    from scipy import stats as st

    rows = []
    prep_cache = {}
    for lag_bins in lag_sets:
        n_ours, r_ours, n_eg, r_eg = 0, [], 0, []
        for recday, res in runs.items():
            cfg = _config_from(res)
            if recday not in prep_cache:
                prep_cache[recday] = _prepare_test_sessions(data_dic, recday, res, cfg)
            # ONE pass gives both: `mean_raw` (no fold exclusion) for the per-neuron-mask
            # semantics, `mean_nozero` + `counts` for his.
            sc = score_elgaby(data_dic, recday, res, lag_bins=lag_bins, config=cfg,
                              preps=prep_cache[recday])
            m = score_ours(res, lag_bins)['mask']
            sel = m & res[tuning_key] & np.isfinite(sc['mean_raw'])
            n_ours += int(sel.sum())
            r_ours.append(sc['mean_raw'][sel])

            sel_eg = (sc['counts'] > 0) & res[tuning_key] & np.isfinite(sc['mean_nozero'])
            n_eg += int(sel_eg.sum())
            r_eg.append(sc['mean_nozero'][sel_eg])
        for label, n, rr in (('per-neuron mask', n_ours, r_ours),
                             ('El-Gaby per-fold', n_eg, r_eg)):
            r = np.concatenate(rr) if rr else np.array([])
            r = r[np.isfinite(r)]
            t, p = st.ttest_1samp(r, 0) if len(r) > 1 else (np.nan, np.nan)
            rows.append({'lag_set': '{' + ','.join(map(str, lag_bins)) + '}',
                         'semantics': label, 'n': n,
                         'mean_r': round(float(r.mean()), 4) if len(r) else np.nan,
                         'frac_pos': round(float(np.mean(r > 0)), 3) if len(r) else np.nan,
                         'p': float(p)})
    table = pd.DataFrame(rows)
    if verbose:
        print(table.to_string(index=False))
        print("\nEl-Gaby reports n = 329 for the 'nonzero' (30 deg) quantity, lags {0,11}")
    return table
