"""Why do the LEC GLMs explain so little, and what would fix it?

Cross-validating the existing GLM turned up a result that needed explaining before any
refit was worth launching: on five recdays (one per mouse), **place is the only regressor
with a positive cross-validated CPD**. Every other regressor -- goal_progress, task_state,
time_from/to_reward, head_direction, speed, acceleration, the reward distances -- has a
NEGATIVE held-out CPD, meaning dropping it *improves* held-out prediction.

This module holds the diagnostics that were run to explain that, so every number quoted in
`ANATOMY_SPLIT.md` is reproducible. Each one is a candidate explanation, and each is either
confirmed or ruled out:

  sparsity   Is the input simply too thin? `Neuron_raw` is integer counts per 25 ms bin, and
             `downsample_session_data(mode='stride')` keeps every 10th bin and DISCARDS the
             other nine. Measures how many spikes actually reach the GLM.
  affine     Would normalising the firing rates help? Proves that a GLOBAL z-score (or any
             global shift/scale) leaves R^2 and CPD algebraically unchanged, so only a
             PER-SESSION adjustment can move them. Pure synthetic; needs no data.
  ridge      Are the negative CPDs overfitting that shrinkage would fix? Sweeps the ridge
             penalty and asks whether any value turns them positive while place survives.
  families   Is it collinearity? goal_progress / time_* / distance_* are near-redundant
             within a leg, so each may show no UNIQUE variance while the family carries
             signal. Drops whole families jointly to test that.
  density    Does binned aggregation beat strided subsampling? Same model, same folds, same
             sample count and spacing -- only the number of spikes per sample changes.
  grid       2x2 of {stride, bin} x {linear, Poisson}. Separates the data-density question
             from the likelihood question, on one fixed neuron set.
  binning    Decile vs uniform bin placement. The main GLM bins speed/acceleration/time_*/
             distance_* by QUANTILE but goal_progress/goal_progress_distance/progress_since_A
             by EQUAL WIDTH, so the two frames get different effective flexibility. This also
             tests a leak: the decile edges in `run_glm_analysis` are computed from pooled
             data that has already seen the held-out fold, whereas equal-width edges cannot
             leak -- and the quantile-binned variables are exactly the ones binning rescued.
             Uses `glm_cv_cpd`, which fits encoders on TRAIN rows only, so both issues are
             settled by the same run.

Run:
    python code/w1_encoding_diagnostics.py                    # all of them
    python code/w1_encoding_diagnostics.py --only ridge families
    python code/w1_encoding_diagnostics.py --only affine      # no data load needed
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = os.path.join(REPO, 'data', 'processed_data', 'w1_encoding_diagnostics.pkl')

#: The backbone section's regressors (same as `w1_refit.SECTIONS['distance_gp_state_filtered']`
#: minus goal_progress_distance, which is NaN-heavy and not at issue here).
REG = ['place', 'task_state', 'head_direction', 'goal_progress', 'speed', 'acceleration',
       'time_from_reward', 'time_to_reward', 'distance_from_reward', 'distance_to_reward']

#: The full set, matching `w1_refit.SECTIONS['all_regressors']` -- what the refit will
#: actually fit. Adds the three families the backbone omits, each of which bears on a
#: conclusion drawn from the backbone run:
#:   goal_progress_distance : the PATH-LENGTH analogue of goal_progress. The backbone only
#:       tested the time-based version, so "goal progress does not survive CV" is currently
#:       a claim about time-framed progress only -- and the distance-framed reward variables
#:       are the ones that binning rescued, so this is the live alternative.
#:   poke_rewarded / poke_unrewarded : collinear with early time_from_reward BY CONSTRUCTION
#:       (a rewarded poke's entry bin IS the reward bin). Without them, consumption-period
#:       activity has nowhere to go but early time_from_reward, so the backbone's rescue of
#:       that regressor is partly attributable to drinking.
#:   time_since_A / time_to_A / progress_since_A : the loop-anchored family.
#: Including pokes forces parameterization='all_bins' (a single-column regressor cannot be
#: reference-coded), which is rank-deficient but leaves RSS/R2/CPD untouched.
REG_FULL = ['place', 'task_state', 'poke_rewarded', 'poke_unrewarded',
            'head_direction', 'goal_progress', 'goal_progress_distance',
            'speed', 'acceleration', 'time_from_reward', 'time_to_reward',
            'time_since_A', 'time_to_A', 'progress_since_A',
            'distance_from_reward', 'distance_to_reward']

#: Set by --regressor-set; every diagnostic reads this.
ACTIVE_REG = list(REG)

#: Whole-family joint drops, for the collinearity test.
FAMILIES = [
    ('within_leg_any', ['goal_progress', 'time_from_reward', 'time_to_reward',
                        'distance_from_reward', 'distance_to_reward']),
    ('time_any', ['time_from_reward', 'time_to_reward']),
    ('reward_dist_any', ['distance_from_reward', 'distance_to_reward']),
    ('gp_plus_time', ['goal_progress', 'time_from_reward', 'time_to_reward']),
    ('kinematics_any', ['speed', 'acceleration']),
]

NSUB = 25          # neurons per recday for the Poisson arms (IRLS is 9-22 s/neuron)
SUBSET_SEED = 0    # see `poisson_subset` -- the subset MUST NOT be arange(NSUB)
RIDGE_LAMBDAS = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def one_recday_per_mouse(data_dic):
    """One recday from each mouse.

    The first smoke test used the first three recdays alphabetically, which are all ah08 --
    the lowest-firing mouse in the cohort (1.07 Hz median) and therefore the worst case. It
    could not speak for the cohort, and the per-mouse spread turned out to be large.
    """
    seen, out = set(), []
    for r in sorted(data_dic):
        m = r.split('_')[0]
        if m not in seen:
            seen.add(m)
            out.append(r)
    return out


def poisson_subset(n_neurons, n=NSUB, seed=SUBSET_SEED):
    """A RANDOM neuron subset, never the first `n`.

    Neuron index is `Neuron_raw` row order, which is almost perfectly depth-ordered along the
    probe (corr of row index with `y_um` is +0.976 to +0.988 across mice). So `arange(25)` is
    "the 25 most superficial units" -- in ah10 that was 18 ENTl-deep and 2 ENTl-sup with ZERO
    of its 29 SUB/ProS and 21 CA1/HPF units, and in ly07 it caught 3 of 28 SUB/ProS. Since
    SUB/ProS fires ~3x faster than ENTl-deep and binning helps most where firing is sparsest,
    a superficial-biased subset systematically overstates the density benefit.
    """
    rng = np.random.default_rng(seed)
    if n >= n_neurons:
        return np.arange(n_neurons)
    return np.sort(rng.choice(n_neurons, n, replace=False))


def per_neuron_frame(recday, cpd_cv, extra=None, neuron_index=None):
    """Per-neuron CPDs as a long-format record, so a result can be split by region LATER.

    Diagnostics that collapse straight to a median can never be re-examined anatomically,
    which is a bad property in a project whose whole purpose is the regional split. Row k
    carries `neuron` = the `Neuron_raw` row index, which is exactly the key
    `anatomy_split.join_regions` needs.
    """
    idx = (np.arange(len(next(iter(cpd_cv.values())))) if neuron_index is None
           else np.asarray(neuron_index))
    rec = {'recday': recday, 'neuron': idx}
    for g, v in cpd_cv.items():
        rec[f'cpd__{g}'] = np.asarray(v, dtype=float)
    if extra:
        for k, v in extra.items():
            rec[k] = np.asarray(v, dtype=float)
    return rec


def capture_designs(data_dic, recdays, downsample_mode='stride', joint_specs=()):
    """Return `[(X, FR, session_ids, groups), ...]`, one per recday.

    `run_glm_analysis` builds the design matrix inside its own loop and does not return it,
    so this intercepts `glm_cv.cv_scores` (which it calls with exactly those arguments) to
    grab them. A testing device, deliberately local to this module: the alternative is a
    `return_design` flag on the fitting path, which is more API surface than a diagnostics
    script justifies. `cv_only=True` skips the in-sample per-neuron loop, which costs ~680 s
    per recday against ~20 s for the CV path.
    """
    import glm_analysis_v2 as glm
    import glm_cv as cv
    import w1_refit

    param = w1_refit.choose_parameterization(ACTIVE_REG)
    store = []
    original = cv.cv_scores

    def hook(X, FR, sid, groups, **kw):
        store.append((X.copy(), FR.copy(), sid.copy(), dict(groups)))
        return original(X, FR, sid, groups, **kw)

    cv.cv_scores = hook
    try:
        glm.run_glm_analysis(
            recdays, data_dic, num_permutations=1, regressors_to_include=ACTIVE_REG,
            joint_drop_groups=list(joint_specs), compute_cpd=True,
            parameterization=param, cross_validate=True,
            cv_n_perm=0, cv_only=True, downsample_mode=downsample_mode,
        )
    finally:
        cv.cv_scores = original
    return store


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diag_sparsity(**_):
    """How many spikes actually reach the GLM?"""
    print('=' * 78)
    print('SPARSITY — what the GLM is actually fed')
    print('=' * 78)
    cands = sorted(glob.glob(os.path.join(
        REPO, 'data', 'processed_data', 'neuron_raw_*', 'Neuron_raw_*.npy')))
    if not cands:
        print('  no raw Neuron_raw arrays found on disk; skipping')
        return None
    a = np.load(cands[0])
    integral = bool(np.allclose(a, np.rint(a)))
    res = {
        'file': os.path.basename(cands[0]), 'shape': a.shape, 'dtype': str(a.dtype),
        'is_integer_counts': integral, 'mean_per_bin': float(a.mean()),
        'frac_zero': float(np.mean(a == 0)), 'max': float(a.max()),
        'implied_hz_at_25ms': float(a.mean() / 0.025),
    }
    print(f"  {res['file']}  shape {a.shape}  dtype {a.dtype}")
    print(f"  integer counts: {integral}   mean {res['mean_per_bin']:.4f}/bin   "
          f"max {res['max']:.0f}   zeros {res['frac_zero']:.3f}")
    print(f"  implied rate at 25 ms bins: {res['implied_hz_at_25ms']:.2f} Hz")
    print(f"\n  With downsample_factor=10 and mode='stride', 9 of every 10 bins are")
    print(f"  DISCARDED: ~{100 / 10:.0f}% of spikes reach the GLM, each retained sample")
    print(f"  still spanning 25 ms and carrying ~{res['mean_per_bin']:.2f} spikes.")
    return res


def diag_affine(**_):
    """A global z-score cannot change R^2 or CPD. Only a per-session one can."""
    print('=' * 78)
    print('AFFINE INVARIANCE — can normalising the firing rates help?')
    print('=' * 78)
    rng = np.random.default_rng(0)
    T, p = 4000, 20
    X = np.column_stack([np.ones(T), rng.normal(size=(T, p))])
    y = X[:, 1:4] @ rng.normal(size=3) + rng.normal(size=T) * 3
    gi = [1, 2, 3]

    def scores(yy):
        Xr = np.delete(X, gi, axis=1)
        f = yy - X @ np.linalg.pinv(X) @ yy
        r = yy - Xr @ np.linalg.pinv(Xr) @ yy
        return 1 - (f @ f) / ((yy - yy.mean()) ** 2).sum(), ((r @ r) - (f @ f)) / (r @ r)

    base = scores(y)
    sid = np.repeat(np.arange(5), T // 5)
    y_ps = y.copy()
    for s in np.unique(sid):
        y_ps[sid == s] -= y[sid == s].mean()

    rows = [('raw', y), ('global z-score', (y - y.mean()) / y.std()),
            ('global shift', y - 5.0), ('global scale', y / 7.3),
            ('per-session centre', y_ps)]
    res = {}
    for tag, yy in rows:
        r2, cpd = scores(yy)
        same = abs(r2 - base[0]) < 1e-12 and abs(cpd - base[1]) < 1e-12
        res[tag] = {'r2': r2, 'cpd': cpd, 'identical_to_raw': same}
        print(f"  {tag:20s} r2 {r2:.10f}  cpd {cpd:.10f}   "
              f"{'unchanged' if same else 'CHANGED'}")
    print('\n  R2 and CPD are ratios of sums of squares, and OLS with an intercept is')
    print('  equivariant under any affine transform of the target. So a global z-score')
    print('  returns identical numbers -- it is not a bad idea, it simply has no effect.')
    print('  Only a per-session adjustment alters the data rather than its units.')
    return res


def diag_ridge(designs, **_):
    """Is the negative held-out CPD overfitting that shrinkage would fix?"""
    import glm_cv as cv
    print('=' * 78)
    print('RIDGE SWEEP — does shrinkage turn the negative CPDs into signal?')
    print('=' * 78)
    show = ['place', 'goal_progress', 'task_state', 'time_from_reward']
    print(f"{'lambda':>10s} " + ' '.join(f'{g[:12]:>13s}' for g in show))
    res = {}
    for lam in RIDGE_LAMBDAS:
        acc = {g: [] for g in ACTIVE_REG}
        for X, FR, sid, groups in designs:
            wf = cv.within_session_folds(sid)
            r = cv.cv_scores(X, FR, sid, groups, center_within_sessions=True,
                             fold_ids=wf, ridge=lam)
            for g in ACTIVE_REG:
                acc[g].append(np.nanmedian(r['cpd_cv'][g]))
        res[lam] = {g: float(np.nanmean(v)) for g, v in acc.items()}
        print(f'{lam:10.0f} ' + ' '.join(f'{res[lam][g]:+13.5f}' for g in show))
    print('\n  Shrinkage drives the negatives toward ZERO, not into signal. The only lambda')
    print('  where they turn nominally positive is also where place collapses -- everything')
    print('  is shrunk to nothing from both directions. Overfitting is confirmed as the')
    print('  source of the negatives, and ruled out as something regularisation can rescue.')
    return res


def diag_families(designs_j, **_):
    """Is it collinearity? Test whole families, not just unique variance."""
    import glm_cv as cv
    print('=' * 78)
    print('FAMILY DROPS — is the signal hidden by collinearity?')
    print('=' * 78)
    print('  CPD measures UNIQUE variance. goal_progress, time_* and distance_* are')
    print('  near-redundant within a leg, so each could look empty while the family')
    print('  carries signal. Dropping whole families together tests that.\n')
    single, joint = {}, {}
    for X, FR, sid, groups in designs_j:
        wf = cv.within_session_folds(sid)
        r = cv.cv_scores(X, FR, sid, groups, center_within_sessions=True, fold_ids=wf,
                         joint_specs=FAMILIES)
        for g, v in r['cpd_cv'].items():
            (joint if g in dict(FAMILIES) else single).setdefault(g, []).append(
                np.nanmedian(v))
    for label, d in [('SINGLE (unique variance)', single), ('FAMILY (joint drop)', joint)]:
        print(f'  {label}')
        for g, v in sorted(d.items()):
            print(f'    {g:24s} {np.nanmean(v):+.5f}')
    print('\n  NOTE: place stays in the model in every one of these fits, so this does NOT')
    print('  test whether place absorbs progress-correlated variance. That is a separate')
    print('  question and needs a design with place excluded.')
    return {'single': {k: float(np.nanmean(v)) for k, v in single.items()},
            'joint': {k: float(np.nanmean(v)) for k, v in joint.items()}}


def diag_density(data_dic, recdays, **_):
    """Strided subsampling vs binned aggregation, same model and folds."""
    import glm_cv as cv
    print('=' * 78)
    print('DENSITY — strided subsample vs binned aggregation')
    print('=' * 78)
    res = {}
    for mode in ('stride', 'bin'):
        designs = capture_designs(data_dic, recdays, downsample_mode=mode)
        acc = {g: [] for g in ACTIVE_REG}
        r2s, dens, per_neuron = [], [], []
        for rd, (X, FR, sid, groups) in zip(recdays, designs):
            wf = cv.within_session_folds(sid)
            r = cv.cv_scores(X, FR, sid, groups, center_within_sessions=True, fold_ids=wf)
            for g in ACTIVE_REG:
                acc[g].append(np.nanmedian(r['cpd_cv'][g]))
            r2s.append(np.nanmedian(r['r2_cv']))
            dens.append((float(FR.mean()), float(np.mean(FR == 0))))
            per_neuron.append(per_neuron_frame(
                rd, r['cpd_cv'], extra={'r2_cv': r['r2_cv'],
                                        'mean_count': FR.mean(axis=1)}))
        res[mode] = {'cpd': {g: float(np.nanmean(v)) for g, v in acc.items()},
                     'r2_cv': float(np.nanmean(r2s)),
                     'counts_per_sample': float(np.mean([d[0] for d in dens])),
                     'frac_zero': float(np.mean([d[1] for d in dens])),
                     'per_neuron': per_neuron}
        print(f"  [{mode}] counts/sample {res[mode]['counts_per_sample']:.4f}  "
              f"zeros {res[mode]['frac_zero']:.3f}  r2_cv {res[mode]['r2_cv']:+.5f}")
    print(f"\n{'regressor':24s} {'stride':>11s} {'bin':>11s} {'change':>11s}")
    for g in ACTIVE_REG:
        a, b = res['stride']['cpd'][g], res['bin']['cpd'][g]
        print(f'  {g:22s} {a:+11.5f} {b:+11.5f} {b - a:+11.5f}')
    return res


def diag_binning(data_dic, recdays, **_):
    """Decile vs uniform bin placement, with encoders fit on train rows only."""
    import glm_cv_cpd as gcc
    import time_vs_progress_dissociation as tvp

    print('=' * 78)
    print('BINNING — decile vs uniform bin placement')
    print('=' * 78)
    print('  The live GLM bins speed/acc/time_*/distance_* by QUANTILE and the progress')
    print('  family by EQUAL WIDTH. `matched_linear` makes everything equal-width and')
    print('  `matched_quantile` makes everything quantile, so the asymmetry is isolated.')
    print('  All three fit bin edges on TRAIN rows only, which also removes the edge leak')
    print('  present in run_glm_analysis (its deciles see the held-out fold).\n')

    res = {}
    for mode in ('stride', 'bin'):
        tables = tvp.build_design_tables(recdays, data_dic, downsample_mode=mode,
                                         verbose=False)
        for scheme in ('glm_onehot', 'matched_linear', 'matched_quantile'):
            _per, pooled = gcc.run_cv_cpd(tables, scheme=scheme, verbose=False)
            res[(mode, scheme)] = {
                'cpd': {g: float(np.nanmedian(np.concatenate(
                    [np.atleast_1d(x) for x in pooled['cpd'][g]]))) for g in pooled['names']},
                'r2': float(np.nanmedian(pooled['r2'])),
            }
            print(f"  [{mode}/{scheme}] median r2_cv {res[(mode, scheme)]['r2']:+.5f}",
                  flush=True)

    names = sorted({g for k in res for g in res[k]['cpd']})
    for mode in ('stride', 'bin'):
        print(f"\n  --- {mode} ---")
        print(f"  {'regressor':24s} {'glm(asym)':>11s} {'uniform':>11s} {'quantile':>11s}")
        for g in names:
            row = [res[(mode, s)]['cpd'].get(g, np.nan)
                   for s in ('glm_onehot', 'matched_linear', 'matched_quantile')]
            print(f'    {g:22s} ' + ' '.join(f'{v:+11.5f}' for v in row))
    print('\n  If uniform and quantile agree, bin placement was not driving the result.')
    print('  If the quantile-binned variables lose ground under `matched_linear`, part of')
    print('  their advantage was placement and/or the edge leak, not encoding.')
    return {f'{m}__{s}': v for (m, s), v in res.items()}


def diag_grid(data_dic, recdays, **_):
    """2x2: {stride, bin} x {linear, Poisson}, one fixed neuron set."""
    import glm_cv as cv
    print('=' * 78)
    print(f'2x2 GRID — density x likelihood ({NSUB} neurons/recday)')
    print('=' * 78)
    print('  Poisson cannot be mean-centred (it needs non-negative integers), so the linear')
    print('  arm is centred within session and the Poisson arm is not. Compare WITHIN a')
    print('  likelihood (stride vs bin); across likelihoods, r2 and d2 are different')
    print('  currencies and are not directly comparable.\n')
    res = {}
    for mode in ('stride', 'bin'):
        designs = capture_designs(data_dic, recdays, downsample_mode=mode)
        lin = {g: [] for g in ACTIVE_REG}
        poi = {g: [] for g in ACTIVE_REG}
        lr2, pd2, times, per_neuron = [], [], [], []
        for rd, (X, FR, sid, groups) in zip(recdays, designs):
            wf = cv.within_session_folds(sid)
            sub = poisson_subset(FR.shape[0])
            L = cv.cv_scores(X, FR, sid, groups, center_within_sessions=True, fold_ids=wf)
            t0 = time.time()
            P = cv.cv_scores_poisson(X, FR, sid, groups, fold_ids=wf, neuron_subset=sub)
            times.append(time.time() - t0)
            for g in ACTIVE_REG:
                lin[g].append(np.nanmedian(np.asarray(L['cpd_cv'][g])[sub]))
                poi[g].append(np.nanmedian(P['cpd_cv'][g]))
            lr2.append(np.nanmedian(np.asarray(L['r2_cv'])[sub]))
            pd2.append(np.nanmedian(P['d2_cv']))
            per_neuron.append(per_neuron_frame(
                rd, {g: np.asarray(L['cpd_cv'][g])[sub] for g in ACTIVE_REG},
                extra={'r2_cv': np.asarray(L['r2_cv'])[sub],
                       'd2_cv_poisson': P['d2_cv']}, neuron_index=sub))
        res[mode] = {
            'linear': {g: float(np.nanmean(v)) for g, v in lin.items()},
            'poisson': {g: float(np.nanmean(v)) for g, v in poi.items()},
            'lin_r2_cv': float(np.nanmean(lr2)), 'poi_d2_cv': float(np.nanmean(pd2)),
            'poisson_seconds_per_neuron': float(np.mean(times) / NSUB),
            'per_neuron': per_neuron,
            'neuron_subset': 'random (see poisson_subset); NOT arange, which is depth-ordered',
        }
        print(f"  [{mode}] poisson {np.mean(times):.0f}s for {NSUB} neurons "
              f"({res[mode]['poisson_seconds_per_neuron']:.1f} s/neuron)")
    print(f"\n{'regressor':22s} {'stride/lin':>11s} {'stride/pois':>12s} "
          f"{'bin/lin':>11s} {'bin/pois':>11s}")
    for g in ACTIVE_REG:
        v = [res[m][k][g] for m in ('stride', 'bin') for k in ('linear', 'poisson')]
        print(f'  {g:20s} {v[0]:+11.5f} {v[1]:+12.5f} {v[2]:+11.5f} {v[3]:+11.5f}')
    print(f"\n  {'held-out r2 / d2':20s} {res['stride']['lin_r2_cv']:+11.5f} "
          f"{res['stride']['poi_d2_cv']:+12.5f} {res['bin']['lin_r2_cv']:+11.5f} "
          f"{res['bin']['poi_d2_cv']:+11.5f}")
    return res


DIAGNOSTICS = {
    'sparsity': (diag_sparsity, False),
    'affine': (diag_affine, False),
    'ridge': (diag_ridge, True),
    'families': (diag_families, True),
    'density': (diag_density, True),
    'grid': (diag_grid, True),
    'binning': (diag_binning, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None, choices=list(DIAGNOSTICS))
    ap.add_argument('--recdays', type=int, default=None,
                    help='limit recdays (default: one per mouse)')
    ap.add_argument('--regressor-set', choices=('backbone', 'full'), default='backbone',
                    help="'backbone' = the 10-regressor design; 'full' = all 16, matching "
                         "w1_refit's `all_regressors` section (adds goal_progress_distance, "
                         "the pokes, and the loop-anchored family)")
    args = ap.parse_args()

    global ACTIVE_REG
    ACTIVE_REG = list(REG_FULL if args.regressor_set == 'full' else REG)
    print(f"regressor set: {args.regressor_set} ({len(ACTIVE_REG)} regressors)")

    names = args.only or list(DIAGNOSTICS)
    needs_data = any(DIAGNOSTICS[n][1] for n in names)

    data_dic = recdays = designs = designs_j = None
    if needs_data:
        import glm_analysis_v2 as glm
        data_dic = glm.load_data_dic(validate=False, apply_exclusions=True, verbose=True)
        recdays = one_recday_per_mouse(data_dic)
        if args.recdays:
            recdays = recdays[:args.recdays]
        print(f'\nrecdays ({len(recdays)}, one per mouse): {recdays}\n')

    results = {}
    for n in names:
        fn, wants = DIAGNOSTICS[n]
        if wants and designs is None and n in ('ridge',):
            designs = capture_designs(data_dic, recdays)
        if wants and designs_j is None and n in ('families',):
            designs_j = capture_designs(data_dic, recdays, joint_specs=FAMILIES)
        print()
        results[n] = fn(designs=designs, designs_j=designs_j,
                        data_dic=data_dic, recdays=recdays)
        print()

    prev = {}
    if os.path.exists(OUT):
        with open(OUT, 'rb') as fh:
            prev = pickle.load(fh)
    tag = '' if args.regressor_set == 'backbone' else '__full'
    prev.update({k + tag: v for k, v in results.items() if v is not None})
    prev['_recdays'] = recdays
    prev['_regressors' + tag] = ACTIVE_REG
    with open(OUT, 'wb') as fh:
        pickle.dump(prev, fh)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
