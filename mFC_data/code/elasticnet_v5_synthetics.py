"""Synthetic controls for the V4 anchoring regression, run through the REAL pipeline.

Repo practice: a synthetic must enter at the same door the data does. Here that means emitting
a `data_dic`-shaped dict of `Neuron_raw` / `Locs_raw` / `Trial_times` and letting
`elasticnet_regression_v4.run_cross_validated_regression_v5` build the regressors, pick the
preferred phase, fit, cross-validate and apply the non-zero-lag criterion -- rather than
handing the analysis a ready-made design matrix.

    python elasticnet_v4_synthetics.py [--quick]

| # | control | what a failure would mean |
|---|---|---|
| 1 | the bump loop matches an independent segment-based reference at lags 1..11 | the regressor builder is wrong |
| 2 | 'future' regressors equal the past ones on a time-reversed session | the reversal trick is wrong |
| 3 | a cell anchored 5 steps in the PAST is recovered at past lag 5, and not at a future intermediate lag | past/future are not separable |
| 4 | a cell anchored 5 steps in the FUTURE is the mirror image | ditto, in the other direction |
| 5 | a pure place cell lands at lag 0 and is REJECTED -- and is DETECTED once lag 0 is admitted | the criterion measures the wrong thing, or the test is vacuous |
| 6 | Poisson noise: r ~ 0, and the false-positive rates stay in their measured range | the metric has a positive bias |
| 7 | every per-fold beta lies on the (pref -/+ lag) stripe | the phase/lag coupling is broken |
| 8 | v5 with the v3 legacy flags reproduces v3 exactly | the refactor changed the numbers |
| 8b | v5 with the v4 legacy flags reproduces v4 exactly | v5's new defaults leaked into the legacy path |
| 9 | El-Gaby's state statistic ('pref_phase_mean') stays near nominal FPR on constant-rate cells with 3x unequal legs, where 'max' saturates | his statistic is duration-confounded after all, or 'max' is not (the control is vacuous) |
| 10 | the two preferred-phase rules agree on a cell with a broad early-third field and DISAGREE, in the documented direction, on a cell with a narrow late-third peak over a higher early-third mean | one of the rules is not what it claims |
| 11 | `add_fold_semantics` reproduces `elgaby_figure5.score_elgaby` (his per-fold NaN + nanmean) on the same betas | the post hoc semantics differ from the reference implementation |
| 12 | `y_scaling='zscore_recday'` leaves the state test, the preferred phases and the actual tuning curves BIT IDENTICAL and moves only the betas | the scaling leaked out of the fit into the neuron selection or the readout |
| 13 | `alpha_max(y/s) == alpha_max(y)/s`, and `fit(y/s; alpha, rho)` equals `fit(y; alpha*rho*s + alpha*(1-rho), ...)/s` | the documented penalty algebra -- the reason a fixed alpha is a firing-rate filter -- is wrong |
| 14 | the known anchored cells are still recovered with a z-scored target; the low-rate admission is measured | z-scoring destroys the signal it was meant to expose |
| 15 | Poisson + `y_scaling` raises, and a neuron silent across the recday scores NaN rather than raising | the guards are missing, so a bad run fails hours in or silently divides by zero |
| 16 | glum with `l1_ratio=0` reproduces sklearn's `PoissonRegressor` at the same alpha (coefficients to solver tolerance, same non-zero-lag decisions and peak lags) | the two solvers' alpha conventions differ, so every sweep alpha is mislabelled |
| 17 | the lasso (glum, `l1_ratio=1`) recovers the planted past-anchored cell at its lag and still rejects the place cell, and gets sparser as alpha grows; the all-zero-fit fraction is printed | the L1 branch fits something other than the anchoring model, or the penalty is not applied |
| 18 | `poisson_l1_ratio > 0` / `poisson_positive` on the sklearn solver raise at construction; `run_tag` names lasso and elastic-net runs distinctly and leaves the L2 name unchanged | a sweep point silently runs L2, or overwrites the L2 reference directory |

Control 5 is stated in both directions on purpose: a rejection test that never accepts anything
is not evidence. Controls 8 and 8b must pin EVERY flag whose default has moved, or they silently
stop testing equivalence and start testing the new defaults.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import elasticnet_regression_v5 as v5                      # noqa: E402

GRID = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1),
        6: (1, 2), 7: (2, 0), 8: (2, 1), 9: (2, 2)}
NEIGHBOURS = {n: [m for m in GRID
                  if abs(GRID[m][0] - GRID[n][0]) + abs(GRID[m][1] - GRID[n][1]) == 1]
              for n in GRID}


# ---------------------------------------------------------------------------
# An independent, segment-based definition of the anchoring regressors
# ---------------------------------------------------------------------------

def segment_structure(locs, phases, num_locs=9):
    """Split a session into phase-segments and record which nodes each one visited."""
    T = len(locs)
    change = np.flatnonzero(np.diff(phases) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, T]
    seg_of = np.zeros(T, dtype=int)
    for g, (a, b) in enumerate(zip(starts, ends)):
        seg_of[a:b] = g
    nodes = np.full(T, -1, dtype=int)
    ok = (~np.isnan(locs)) & (locs >= 1) & (locs <= num_locs)
    nodes[ok] = locs[ok].astype(int) - 1
    visited = np.zeros((len(starts), num_locs), dtype=bool)
    for t in range(T):
        if nodes[t] >= 0:
            visited[seg_of[t], nodes[t]] = True
    return starts, ends, seg_of, phases[starts], visited


def segment_reference_regressors(locs, phases, config, direction='past'):
    """(T, 324) anchoring regressors defined directly, with no bump/roll book-keeping.

    Anchor (loc, ap) at lag k is on during segment g iff segment g-k (past) / g+k (future) had
    phase `ap` and the animal visited `loc` in it. This is the definition the bump loop is
    supposed to implement, written independently so it can check it.
    """
    nloc, nph, nlag = config.num_locations, config.num_goal_progress_bins, config.num_lags
    starts, ends, _, seg_phase, visited = segment_structure(locs, phases, nloc)
    G = len(starts)
    X = np.zeros((len(locs), nloc, nph, nlag), dtype=np.float32)
    for g in range(G):
        for k in range(nlag):
            h = g - k if direction == 'past' else g + k
            if 0 <= h < G:
                X[starts[g]:ends[g], :, seg_phase[h], k] = visited[h]
    return X.reshape(len(locs), -1)


# ---------------------------------------------------------------------------
# A synthetic recday that enters through the same door as the data
# ---------------------------------------------------------------------------

def _route(a, b, rng, detour_p=0.45):
    """A walk from node a to node b: greedy on the grid, with random detours so that routes
    differ between trials. Without that variability past lag k and future lag 12-k would be
    the same regressor and the direction comparison would be vacuous."""
    path, cur, guard = [a], a, 0
    while cur != b and guard < 24:
        guard += 1
        if rng.random() < detour_p:
            cur = rng.choice(NEIGHBOURS[cur])
        else:
            ta, tb = GRID[cur], GRID[b]
            opts = [m for m in NEIGHBOURS[cur]
                    if abs(GRID[m][0] - tb[0]) + abs(GRID[m][1] - tb[1])
                    < abs(ta[0] - tb[0]) + abs(ta[1] - tb[1])]
            cur = rng.choice(opts) if opts else rng.choice(NEIGHBOURS[cur])
        path.append(int(cur))
    if path[-1] != b:
        path.append(b)
    return path


def make_behaviour(task, n_trials, rng, dwell=(18, 55), edge_p=0.25, untracked_p=0.03):
    """Locs_raw (with edges as 10..21 and untracked as 0) plus contiguous Trial_times."""
    locs, bounds = [], [0]
    for _ in range(n_trials):
        for s in range(len(task)):
            a, b = task[s], task[(s + 1) % len(task)]
            for i, node in enumerate(_route(a, b, rng)):
                locs += [node] * int(rng.integers(*dwell))
                if i and rng.random() < edge_p:                 # time spent on a corridor
                    locs += [int(rng.integers(10, 22))] * int(rng.integers(3, 12))
            bounds.append(len(locs))
    locs = np.asarray(locs, dtype=float)
    drop = rng.random(len(locs)) < untracked_p                  # SLEAP dropout -> code 0
    locs[drop] = 0
    b = np.asarray(bounds, dtype=int)
    trial_times = np.stack([b[i * 4:i * 4 + 5] for i in range(n_trials)])
    return locs, trial_times


def anchored_rate(locs, phases, config, loc0, anchor_phase, lag, direction, amp, base):
    """Firing driven by ONE anchor: 'node `loc0`+1 was/will be visited `lag` phase-steps
    away, in a segment of phase `anchor_phase`'. Built from the segment reference, so it does
    not inherit any quirk of the bump loop it is used to test."""
    X = segment_reference_regressors(locs, phases, config, direction=direction)
    col = (loc0 * config.num_goal_progress_bins + anchor_phase) * config.num_lags + lag
    return base + amp * X[:, col]


def make_recday(n_sessions=4, n_trials=14, n_noise=6, seed=0, amp=6.0, base=0.15,
                anchor_loc=4, anchor_phase=1, anchor_lag=5, config=None):
    """A `data_dic`-shaped recday whose neuron identities are known by construction.

    Neuron order: 0 = past-anchored, 1 = future-anchored, 2 = pure place cell,
    3.. = Poisson noise.
    """
    config = config or v5.RegressionConfigV5()
    rng = np.random.default_rng(seed)
    tasks = [list(rng.permutation(np.arange(1, 10))[:4]) for _ in range(n_sessions)]
    sessions = {}
    for si, task in enumerate(tasks):
        locs, tt = make_behaviour(task, n_trials, rng)
        phases, _ = v5.compute_phase_state_raw(tt, config.num_goal_progress_bins,
                                               config.num_task_states)
        L = min(len(locs), len(phases))
        locs, phases = locs[:L], phases[:L]

        rates = [
            anchored_rate(locs, phases, config, anchor_loc, anchor_phase, anchor_lag,
                          'past', amp, base),
            anchored_rate(locs, phases, config, anchor_loc, anchor_phase, anchor_lag,
                          'future', amp, base),
            # pure place cell: current node only, no lag and no phase preference
            base + amp * (locs == anchor_loc + 1).astype(float),
        ]
        rates += [np.full(L, base + amp / 3) for _ in range(n_noise)]
        neuron = rng.poisson(np.clip(np.asarray(rates), 0, None)).astype(float)

        sessions[si] = {'Neuron_raw': neuron, 'Locs_raw': locs.astype(int),
                        'Trial_times': tt, 'Task': np.asarray(task)}
    return {'synthetic_recday': sessions}, tasks


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

PAST_CELL, FUTURE_CELL, PLACE_CELL = 0, 1, 2
_results = []


def check(name, passed, detail=''):
    _results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ''))
    return passed


def control_1_2_regressors(data_dic, config):
    """The bump loop vs the independent segment definition, and the reversal trick."""
    print('\n[1] bump loop vs segment-based reference')
    sess = data_dic['synthetic_recday'][0]
    locs = np.asarray(sess['Locs_raw'], dtype=float)
    tt = sess['Trial_times']
    phases, _ = v5.compute_phase_state_raw(tt, config.num_goal_progress_bins,
                                           config.num_task_states)
    L = min(len(locs), len(phases))
    locs, phases = locs[:L].copy(), phases[:L]
    locs[locs > config.num_locations] = np.nan
    locs[locs < 1] = np.nan

    nloc, nph, nlag = config.num_locations, config.num_goal_progress_bins, config.num_lags
    for direction in ('past', 'future'):
        cfg = copy.copy(config)
        cfg.lag_direction = direction
        got = v5.generate_regressors_raw(locs, phases, cfg).reshape(L, nloc, nph, nlag)
        ref = segment_reference_regressors(locs, phases, cfg, direction).reshape(L, nloc, nph, nlag)
        agree = [(got[..., k] == ref[..., k]).mean() for k in range(nlag)]
        worst = min(agree[1:])
        check(f'[1] {direction}: bump loop == segment reference at lags 1..{nlag - 1}',
              worst > 0.999,
              f'min agreement {worst:.4f}; lag 0 agreement {agree[0]:.3f} '
              f'(expected < 1: lag 0 accumulates causally, the reference uses the whole segment)')

    print('\n[2] the future regressors are the past ones on a reversed session')
    cfg_p, cfg_f = copy.copy(config), copy.copy(config)
    cfg_p.lag_direction, cfg_f.lag_direction = 'past', 'future'
    fwd = v5.generate_regressors_raw(locs[::-1].copy(), phases[::-1].copy(), cfg_p)
    fut = v5.generate_regressors_raw(locs, phases, cfg_f)
    check('[2] future(X) == reverse(past(reverse(X)))', np.array_equal(fwd[::-1], fut))


def run_direction(data_dic, config, direction):
    cfg = copy.copy(config)
    cfg.lag_direction = direction
    return cfg, v5.run_cross_validated_regression_v5(
        data_dic, 'synthetic_recday', cfg, valid_sessions=list(data_dic['synthetic_recday']))


def control_3_4_5_6(data_dic, config):
    out = {}
    for direction in ('past', 'future'):
        cfg, res = run_direction(data_dic, config, direction)
        out[direction] = (cfg, res)

    print('\n[3/4] anchored cells are recovered in their own direction only')
    for cell, own, other in ((PAST_CELL, 'past', 'future'), (FUTURE_CELL, 'future', 'past')):
        cfg_own, res_own = out[own]
        cfg_oth, res_oth = out[other]
        lag_own = res_own['peak_lags'][cell]
        lag_oth = res_oth['peak_lags'][cell]
        label = 'past-anchored' if cell == PAST_CELL else 'future-anchored'
        check(f'[3/4] {label} cell peaks at {own} lag {config_anchor_lag}',
              lag_own == config_anchor_lag,
              f'peak lag = {lag_own} (r = {res_own["mean_corrs"][cell]:.3f})')
        check(f'[3/4] {label} cell is flagged non-zero-lag in the {own} run',
              bool(res_own['nonzero_lag_mask'][cell]))
        check(f'[3/4] {label} cell does NOT peak at the mirrored {other} lag',
              lag_oth != config_anchor_lag,
              f'{other} peak lag = {lag_oth}')

    print('\n[5] the pure place cell lands at lag 0 and is rejected -- but IS found once lag 0 counts')
    for direction in ('past', 'future'):
        cfg, res = out[direction]
        lag = res['peak_lags'][PLACE_CELL]
        check(f'[5] place cell peaks at {direction} lag 0', lag == 0, f'peak lag = {lag}')
        check(f'[5] place cell rejected by the lag {cfg.nonzero_lag_min}-{cfg.nonzero_lag_max} window',
              not bool(res['nonzero_lag_mask'][PLACE_CELL]))
        # the other direction of the gate: widen the window and it must be accepted, or the
        # rejection above is vacuous
        wide = copy.copy(cfg)
        wide.nonzero_lag_min, wide.nonzero_lag_max = 0, cfg.num_lags - 1
        mask_wide, _ = v5.identify_nonzero_lag_neurons(res, wide)
        check(f'[5] place cell IS detected in {direction} once lag 0 is admitted (gate is not vacuous)',
              bool(mask_wide[PLACE_CELL]))

    print('\n[6] Poisson noise sits at chance')
    print('    The non-zero-lag criterion is a SHAPE DESCRIPTOR of the beta vector, not a')
    print('    statistical test. Measured on pure noise under the reference-matched defaults')
    print('    (30 cells x 4 seeds x 2 directions): 30-33% fire the 30-degree criterion and')
    print('    8-17% survive the full selection, against 0.8% / <1% for the strict 90-degree')
    print('    one. What has to be unbiased is the CORRELATION -- that is what the headline')
    print('    t-test runs on -- and it is (mean r = +0.04).')
    for direction in ('past', 'future'):
        cfg, res = out[direction]
        noise = np.arange(3, res['cv_coeffs'].shape[0])
        nz_rate = res['nonzero_lag_mask'][noise].mean()
        nz_strict = res['nonzero_lag_mask_strict'][noise].mean()
        sel = res['nonzero_lag_mask'] & res['state_tuned_mask'] & ~np.isnan(res['mean_corrs'])
        mean_r = np.nanmean(res['mean_corrs'][noise])
        check(f'[6] {direction}: noise mean r ~ 0 (the metric is unbiased)',
              not np.isfinite(mean_r) or abs(mean_r) < 0.25, f'mean r = {mean_r:.3f}')
        # bound set from the measurement above (max 0.43 over any single run), not invented
        check(f'[6] {direction}: 30-degree false-positive rate within the measured range',
              nz_rate <= 0.6,
              f'{nz_rate:.0%} of {len(noise)} noise cells flagged NZ-lag, '
              f'{sel[noise].mean():.0%} survive the full selection')
        check(f'[6] {direction}: the STRICT 90-degree criterion is near noise-free',
              nz_strict <= 0.15, f'{nz_strict:.0%} of noise cells flagged strict-NZ-lag')

    print('\n[7] every per-fold beta lies on the stripe')
    for direction in ('past', 'future'):
        cfg, res = out[direction]
        try:
            checked, bad = v5.assert_beta_stripe(res, cfg)
            check(f'[7] {direction}: stripe invariant holds', True, f'{checked} beta vectors checked')
        except AssertionError as exc:
            check(f'[7] {direction}: stripe invariant holds', False, str(exc))
    return out


def _v4_legacy(config):
    """The three v5 flags whose defaults moved away from v4, pinned back to v4's values."""
    legacy = copy.copy(config)
    legacy.state_tuning_statistic = 'max'          # v5 default 'pref_phase_mean' (his code)
    legacy.state_tuning_nan_policy = 'omit'        # v5 default 'propagate' (his code)
    legacy.pref_phase_method = 'raw_mean'          # v5 default 'elgaby_peak' (his code)
    legacy.require_positive_mean_prediction = False
    return legacy


def control_8_v3_equivalence(data_dic, config):
    """v5 in legacy mode must reproduce v3 bit for bit."""
    print('\n[8] v5 with the v3 legacy flags == v3')
    try:
        import elasticnet_regression_v3 as v3
    except Exception as exc:
        check('[8] v3 importable', False, str(exc))
        return

    # Every setting whose DEFAULT has since moved must be pinned here explicitly, or this
    # control silently stops testing v3 equivalence and starts testing the new defaults.
    legacy = _v4_legacy(config)
    legacy.lag_direction = 'past'
    legacy.pref_phase_source = 'train'   # default is now 'test' (the reference's leakage)
    legacy.restrict_to_pref_phase = True
    legacy.drop_untracked_bins = False
    legacy.nonzero_lag_zero_lags = (0,)  # v3 zeroed lag 0 only
    legacy.nonzero_lag_min = 2           # v3's window was 2..num_lags-2
    legacy.nonzero_lag_max = config.num_lags - 2
    legacy.state_tuning_min_fraction = 0.0   # v3 OR-ed across sessions (tuned in >= 1 task)
    legacy.poisson_link = 'linear'       # v3 always used the linear predictor
    legacy.require_positive_top3 = False
    legacy.nz_per_fold = False
    legacy.state_reduce = 'mean'
    legacy.alpha_mode = 'fixed'          # v3 has no relative-alpha branch

    sessions = list(data_dic['synthetic_recday'])
    r4 = v5.run_cross_validated_regression_v5(data_dic, 'synthetic_recday', legacy,
                                              valid_sessions=sessions)
    c3 = v3.RegressionConfigV3(
        num_locations=config.num_locations, num_goal_progress_bins=config.num_goal_progress_bins,
        num_task_states=config.num_task_states, num_lags=config.num_lags,
        use_poisson=config.use_poisson, regularize=config.regularize,
        poisson_alpha=config.poisson_alpha, elasticnet_alpha=config.elasticnet_alpha,
        l1_ratio=config.l1_ratio, positive=config.positive,
        num_bins_per_state=config.num_bins_per_state,
        state_tuning_p_threshold=config.state_tuning_p_threshold, max_iter=config.max_iter)
    r3 = v3.run_cross_validated_regression_v3(data_dic, 'synthetic_recday', c3,
                                              valid_sessions=sessions)

    for key in ('cv_coeffs', 'corrs', 'corrs_nonzero', 'mean_corrs', 'state_tuned_mask',
                'cv_tuning_correlations'):
        a, b = np.asarray(r4[key], dtype=float), np.asarray(r3[key], dtype=float)
        check(f'[8] {key} identical', a.shape == b.shape and np.allclose(a, b, equal_nan=True),
              f'max |diff| = {np.nanmax(np.abs(a - b)):.3g}' if a.shape == b.shape else
              f'shape {a.shape} vs {b.shape}')

    m4, l4 = v5.identify_nonzero_lag_neurons(r4, legacy)
    m3, l3 = v3.identify_nonzero_lag_neurons(r3, c3)
    check('[8] non-zero-lag mask identical', np.array_equal(m4, m3),
          f'{int((m4 != m3).sum())} neurons differ')


def control_8b_v4_equivalence(data_dic, config):
    """v5 with the v4 flags must reproduce v4 bit for bit (v4 is frozen for this purpose)."""
    print('\n[8b] v5 with the v4 legacy flags == v4')
    try:
        import elasticnet_regression_v4 as v4
    except Exception as exc:
        check('[8b] v4 importable', False, str(exc))
        return
    legacy = _v4_legacy(config)
    sessions = list(data_dic['synthetic_recday'])
    r5 = v5.run_cross_validated_regression_v5(data_dic, 'synthetic_recday', legacy,
                                              valid_sessions=sessions)
    allowed = v4.RegressionConfigV4.__init__.__code__.co_varnames
    c4 = v4.RegressionConfigV4(**{k: v for k, v in legacy.to_dict().items()
                                  if k in allowed and k != 'self'})
    r4 = v4.run_cross_validated_regression_v4(data_dic, 'synthetic_recday', c4,
                                              valid_sessions=sessions)
    for key in ('cv_coeffs', 'corrs', 'corrs_nonzero', 'corrs_nonzero_strict', 'mean_corrs',
                'pref_phases', 'state_tuned_mask', 'state_tuned_mask_alt',
                'state_tuned_fraction', 'nonzero_lag_mask', 'nonzero_lag_mask_strict',
                'peak_lags', 'cv_tuning_correlations', 'cv_tuning_correlations_pref',
                'cv_actual_mean_state4', 'cv_predicted_nz_mean_state4'):
        a, b = np.asarray(r5[key], dtype=float), np.asarray(r4[key], dtype=float)
        check(f'[8b] {key} identical', a.shape == b.shape and np.allclose(a, b, equal_nan=True),
              f'max |diff| = {np.nanmax(np.abs(a - b)):.3g}' if a.shape == b.shape else
              f'shape {a.shape} vs {b.shape}')


def control_9_duration_confound(config, n_cells=300, ratio=3.0, rate_hz=8.0, n_trials=30,
                                seed=0):
    """Constant-rate Poisson cells with one leg `ratio` times longer than the others: the
    false-positive rate of each state-tuning statistic at p<0.05.

    The claim under test is that El-Gaby's statistic (the mean over the preferred-phase bins,
    Figure2 cell 46) is NOT confounded by leg duration, whereas the paper's-text statistic
    (the peak) is. Both directions are asserted: 'max' must saturate (or the control would be
    vacuous), 'pref_phase_mean' must stay near nominal.
    """
    print(f'\n[9] duration confound: {n_cells} constant-rate cells, longest:shortest = {ratio}x')
    rng = np.random.default_rng(seed)
    short = 60
    durs = np.array([short, short, short, int(round(short * ratio))])
    bounds = np.concatenate([[0], np.cumsum(np.tile(durs, n_trials))])
    tt = np.stack([bounds[i * 4:i * 4 + 5] for i in range(n_trials)])
    T = int(bounds[-1])
    neuron = rng.poisson(rate_hz * 0.025, size=(n_cells, T)).astype(float)
    prefs = rng.integers(0, config.num_goal_progress_bins, n_cells)
    fpr = {}
    for stat, policy in (('max', 'omit'), ('mean', 'omit'), ('pref_phase_mean', 'propagate'),
                         ('pref_phase_mean', 'omit')):
        tuned = v5.identify_state_tuned_neurons_raw(neuron, tt, config, statistic=stat,
                                                    nan_policy=policy, pref_phases=prefs)
        fpr[(stat, policy)] = float(tuned.mean())
        print(f'    {stat:16s} nan_policy={policy:9s} FPR = {fpr[(stat, policy)]:.3f}')
    check(f"[9] 'max' saturates at a {ratio}x duration ratio (control is not vacuous)",
          fpr[('max', 'omit')] > 0.5, f"FPR = {fpr[('max', 'omit')]:.3f}")
    # nominal is 0.05; the 0.15 bound leaves room for the selection-then-test bias that every
    # argmax-then-t-test carries, which is small (measured ~0.05-0.10 at equal legs)
    check(f"[9] his 'pref_phase_mean' stays near nominal at {ratio}x (propagate)",
          fpr[('pref_phase_mean', 'propagate')] < 0.15,
          f"FPR = {fpr[('pref_phase_mean', 'propagate')]:.3f}")
    check(f"[9] his 'pref_phase_mean' stays near nominal at {ratio}x (omit)",
          fpr[('pref_phase_mean', 'omit')] < 0.15,
          f"FPR = {fpr[('pref_phase_mean', 'omit')]:.3f}")
    return fpr


def control_10_pref_phase_rules(config, seed=0):
    """The two preferred-phase rules, on cells built to agree and to disagree.

    Cell A: a broad field over the whole early third of every state -> both rules give 0.
    Cell B: a flat 6 Hz early third and a late third that is 1 Hz except for a 3-bin 60 Hz
    burst at its start. Time-weighted mean rate is highest in the early third ('raw_mean' -> 0)
    but the peak of the trial-averaged normalised curve is in the late third
    ('elgaby_peak' -> 2). That is the documented direction of disagreement: his rule follows
    peaks, ours follows mass.
    """
    print('\n[10] preferred-phase rules: agreement and the documented disagreement')
    rng = np.random.default_rng(seed)
    nph = config.num_goal_progress_bins
    task = [1, 5, 9, 3]
    locs, tt = make_behaviour(task, 12, rng, dwell=(30, 60))
    phases, states = v5.compute_phase_state_raw(tt, nph, config.num_task_states)
    L = min(len(locs), len(phases))
    locs, phases = locs[:L], phases[:L]
    # position within the current phase segment, in bins
    seg_start = np.r_[0, np.flatnonzero(np.diff(phases) != 0) + 1]
    pos = np.arange(L) - np.repeat(seg_start, np.diff(np.r_[seg_start, L]))
    rate_a = np.where(phases == 0, 8.0, 1.0)
    rate_b = np.where(phases == 0, 6.0, np.where((phases == 2) & (pos < 3), 60.0, 1.0))
    neuron = rng.poisson(np.stack([rate_a, rate_b]) * 0.025).astype(float)
    locs_nan = locs.copy()
    locs_nan[(locs_nan < 1) | (locs_nan > config.num_locations)] = np.nan
    raw = v5.session_pref_phases(neuron, tt, config, locs=locs_nan, phases=phases,
                                 method='raw_mean')
    peak = v5.session_pref_phases(neuron, tt, config, method='elgaby_peak')
    check('[10] broad early field: raw_mean -> 0', raw[0] == 0, f'got {raw[0]}')
    check('[10] broad early field: elgaby_peak -> 0', peak[0] == 0, f'got {peak[0]}')
    check("[10] narrow late burst over a higher early mean: raw_mean -> 0 (follows mass)",
          raw[1] == 0, f'got {raw[1]}')
    check("[10] narrow late burst over a higher early mean: elgaby_peak -> 2 (follows the peak)",
          peak[1] == 2, f'got {peak[1]}')


def control_11_fold_semantics(data_dic, config):
    """`add_fold_semantics` (post hoc, from stored arrays) == `elgaby_figure5.score_elgaby`
    (his cell-26 semantics re-implemented from the regressors) on the same betas.

    Run with `require_positive_top3=False` so both use his `argsort` top-3, otherwise the tie
    rule among exact zeros makes them legitimately differ on sparse folds.
    """
    print('\n[11] post hoc fold semantics == elgaby_figure5.score_elgaby')
    # `elgaby_figure5.py` lives in the LEC tree only -- this file is mirrored byte-identically
    # into mFC_data/code/, where the reference implementation is absent. Absent is a SKIP;
    # present-but-unimportable is a real failure and must still be reported as one.
    if not os.path.exists(os.path.join(HERE, 'elgaby_figure5.py')):
        print(f'  [SKIP] elgaby_figure5.py is not in {HERE} (it lives in the LEC tree only)')
        return
    try:
        import elgaby_figure5 as eg
    except Exception as exc:
        check('[11] elgaby_figure5 importable', False, str(exc))
        return
    cfg = copy.copy(config)
    cfg.require_positive_top3 = False
    sessions = list(data_dic['synthetic_recday'])
    res = v5.run_cross_validated_regression_v5(data_dic, 'synthetic_recday', cfg,
                                               valid_sessions=sessions)
    for lag_bins, ours_key in ((tuple(cfg.nonzero_lag_zero_lags), 'mean_corrs_nonzero_passing'),
                               (tuple(cfg.nonzero_lag_zero_lags_strict),
                                'mean_corrs_nonzero_strict_passing')):
        his = eg.score_elgaby(data_dic, 'synthetic_recday', res, lag_bins=lag_bins, config=cfg)
        ours = np.asarray(res[ours_key], dtype=float)
        theirs = np.asarray(his['mean_nozero'], dtype=float)
        both = np.isfinite(ours) & np.isfinite(theirs)
        same_support = np.array_equal(np.isfinite(ours), np.isfinite(theirs))
        diff = float(np.max(np.abs(ours[both] - theirs[both]))) if both.any() else 0.0
        check(f'[11] lags {lag_bins}: same neurons counted', same_support,
              f'{int((np.isfinite(ours) != np.isfinite(theirs)).sum())} neurons differ')
        check(f'[11] lags {lag_bins}: passing-fold means identical', diff < 1e-5,
              f'max |diff| = {diff:.3g} over {int(both.sum())} neurons')

    # --- 11b: the POOL, not just the per-neuron values -------------------------------------
    # This is what the original control missed. It compared values neuron by neuron, which
    # matched, while `three_panel_summary`'s 'elgaby' pool carried an extra "finite r in every
    # fold" requirement that his cell 38 does not have -- so the panel COUNTS were wrong by ~4x
    # for two years' worth of reported numbers. Assert the counts his way.
    table = v5.build_unit_table({'synthetic_recday': res}, cfg, data_dic=None,
                                require_anatomy=False)
    tuned = table.state_tuned.to_numpy()
    got = v5.three_panel_summary(table, semantics='elgaby', links='', verbose=False)
    exp = [int((tuned & np.isfinite(np.asarray(res[k], dtype=float))).sum())
           for k in ('mean_corrs', 'mean_corrs_nonzero_passing',
                     'mean_corrs_nonzero_strict_passing')]
    for (label, n_got), n_exp in zip(got[['panel', 'n']].itertuples(index=False), exp):
        check(f"[11b] 'elgaby' pool is per-panel: {label[:34]}", int(n_got) == n_exp,
              f'summary n={int(n_got)} vs state_tuned & finite(panel) = {n_exp}')

    # The superseded rule must still be reachable, and must be a SUBSET -- strictly smaller
    # whenever some fold produced an all-zero fit, or the two rules would be indistinguishable
    # and this control vacuous.
    old = v5.three_panel_summary(table, semantics='elgaby_everyfold', links='', verbose=False)
    n_new, n_old = got.n.to_numpy(), old.n.to_numpy()
    check('[11b] everyfold counts <= corrected counts', bool(np.all(n_old <= n_new)),
          f'{list(n_old)} vs {list(n_new)}')
    # Non-vacuity. This synthetic has only a handful of state-tuned cells and they all fit in
    # every fold, so the two rules coincide above and the subset assertion alone proves nothing.
    # Rather than hope a random population separates them, construct the discriminating case:
    # take a state-tuned neuron that IS finite everywhere and blank one of its folds, which is
    # what his `nanmean(prediction) > 0` gate does on real data (it drops 51% of PFC Poisson
    # folds, and is exactly why that pool read 140 instead of 573). The corrected rule must
    # still count the neuron; the superseded one must drop it.
    n_folds = int(res['num_sessions'])
    corrs = np.asarray(res['corrs'], dtype=float)
    tuned_arr = np.asarray(res['state_tuned_mask'], dtype=bool)
    cand = [i for i in range(len(tuned_arr))
            if tuned_arr[i] and np.isfinite(corrs[i]).sum() == n_folds]
    if not cand:
        check('[11b] a neuron was available to perturb', False,
              'no state-tuned neuron is finite in every fold')
    else:
        j = cand[0]
        pert = dict(res)
        pert['corrs'] = corrs.copy()
        pert['corrs'][j, 0] = np.nan          # one fold blanked, the rest still finite
        for k in ('n_finite_folds', 'n_passing_folds', 'n_passing_folds_strict'):
            pert.pop(k, None)                 # force add_fold_semantics to recompute
        ptab = v5.build_unit_table({'synthetic_recday': pert}, cfg, data_dic=None,
                                   require_anatomy=False)
        pnew = v5.three_panel_summary(ptab, semantics='elgaby', links='',
                                      verbose=False).n.to_numpy()
        pold = v5.three_panel_summary(ptab, semantics='elgaby_everyfold', links='',
                                      verbose=False).n.to_numpy()
        print(f'    perturbed neuron {j}: fold 0 blanked, {n_folds - 1}/{n_folds} folds finite')
        in_new = bool(ptab.selected_all_eg.to_numpy()[j])
        in_old = bool(ptab.selected_all_eg_everyfold.to_numpy()[j])
        check('[11b] corrected rule still counts a neuron missing one fold', in_new,
              f'selected_all_eg[{j}] = {in_new}')
        check('[11b] superseded every-fold rule drops it (the rules SEPARATE)', not in_old,
              f'selected_all_eg_everyfold[{j}] = {in_old}')
        check('[11b] and that shows up as a panel-count difference of 1',
              int(pnew[0] - pold[0]) == 1, f'corrected {list(pnew)} vs everyfold {list(pold)}')

    # --- 11c: with no positive-mean gate, Poisson makes the two rules identical -------------
    # Poisson never zeroes a coefficient, so every fold yields a finite r and "every fold" ==
    # ">= 1 fold". The PFC Poisson pool collapsed to 140 only because that run applied his
    # `nanmean(prediction) > 0` gate.
    pcfg = copy.copy(cfg)
    pcfg.use_poisson = True
    pcfg.regularize = True
    pcfg.require_positive_mean_prediction = False
    pres = v5.run_cross_validated_regression_v5(data_dic, 'synthetic_recday', pcfg,
                                                valid_sessions=sessions)
    ptab = v5.build_unit_table({'synthetic_recday': pres}, pcfg, data_dic=None,
                               require_anatomy=False)
    a = v5.three_panel_summary(ptab, semantics='elgaby', links='', verbose=False).n.to_numpy()
    b = v5.three_panel_summary(ptab, semantics='elgaby_everyfold', links='',
                               verbose=False).n.to_numpy()
    check('[11c] Poisson without the gate: both pool rules agree', bool(np.array_equal(a, b)),
          f'corrected {list(a)} vs everyfold {list(b)}')


def control_12_zscore_invariance(data_dic, config):
    """`y_scaling='zscore_recday'` must touch the FIT and nothing else.

    The state test (z across states, then a t-test), the preferred-phase rules and the n=4
    state-mean Pearson readout are all invariant to a positive per-neuron affine transform, and
    the transform is applied to `yfit` only -- so every one of those arrays must come back bit
    identical to the raw run, while `cv_coeffs` must move (otherwise the flag did nothing).
    """
    print('\n[12] y_scaling="zscore_recday" changes the betas and nothing upstream')
    sessions = list(data_dic['synthetic_recday'])
    raw_cfg = copy.copy(config)
    raw_cfg.y_scaling = 'none'
    z_cfg = copy.copy(config)
    z_cfg.y_scaling = 'zscore_recday'
    r_raw = v5.run_cross_validated_regression_v5(data_dic, 'synthetic_recday', raw_cfg,
                                                 valid_sessions=sessions)
    r_z = v5.run_cross_validated_regression_v5(data_dic, 'synthetic_recday', z_cfg,
                                               valid_sessions=sessions)

    check('[12] used sessions identical', r_raw['used_sessions'] == r_z['used_sessions'],
          f"{r_raw['used_sessions']} vs {r_z['used_sessions']}")
    for key in ('state_tuned_mask', 'state_tuned_mask_any', 'state_tuned_fraction',
                'state_tuned_per_session', 'tuning_pref_state', 'pref_phases',
                'session_pref_phase_raw', 'session_pref_phase_elgaby',
                'cv_actual_tuning', 'cv_actual_mean_state4', 'y_scale_sd', 'n_spikes_recday'):
        a, b = np.asarray(r_raw[key], dtype=float), np.asarray(r_z[key], dtype=float)
        check(f'[12] {key} identical', a.shape == b.shape and np.allclose(a, b, equal_nan=True),
              f'max |diff| = {np.nanmax(np.abs(a - b)):.3g}' if a.shape == b.shape else
              f'shape {a.shape} vs {b.shape}')
    a = np.asarray(r_raw['cv_coeffs'], dtype=float)
    b = np.asarray(r_z['cv_coeffs'], dtype=float)
    check('[12] the betas actually change (the flag is not a no-op)',
          not np.allclose(a, b, equal_nan=True),
          f'max |diff| = {np.nanmax(np.abs(a - b)):.3g}')
    # the fixed alpha zeroes fewer neurons once the target is scaled
    nz_raw = np.nanmean(np.asarray(r_raw['n_nonzero_betas'], dtype=float) == 0)
    nz_z = np.nanmean(np.asarray(r_z['n_nonzero_betas'], dtype=float) == 0)
    print(f'    all-zero fits: raw {nz_raw:.3f} -> z-scored {nz_z:.3f}'
          f'  (alpha_mode={config.alpha_mode!r}, so a relative alpha makes this a no-op)')

    check('[12] the run directory is tagged', '_zscore_v5_' in v5.run_dir_name(z_cfg, stamp='S'),
          v5.run_dir_name(z_cfg, stamp='S'))
    check('[12] the raw run is NOT tagged', '_zscore' not in v5.run_dir_name(raw_cfg, stamp='S'),
          v5.run_dir_name(raw_cfg, stamp='S'))
    check("[12] the config records it", z_cfg.to_dict().get('y_scaling') == 'zscore_recday',
          repr(z_cfg.to_dict().get('y_scaling')))
    check("[12] results record it", r_z.get('y_scaling') == 'zscore_recday',
          repr(r_z.get('y_scaling')))
    return r_raw, r_z


def control_13_zscore_penalty_algebra(data_dic, config):
    """The exact price of dividing y by s, which is the whole point of the flag.

    `alpha_max` scales by 1/s -- that is the claim behind "a fixed alpha is a firing-rate
    filter". But the fit is NOT a pure per-neuron alpha rescale at l1_ratio < 1: in sklearn's
    parameterisation, fitting `y/s` at `(alpha, rho)` gives `1/s` times the fit of raw `y` with
    L1 weight `alpha*rho*s` and L2 weight `alpha*(1-rho)` -- i.e. at
    `alpha' = alpha*rho*s + alpha*(1-rho)`, `rho' = alpha*rho*s / alpha'`. Only the L1 term
    follows s. Asserting the identity keeps the docstring honest.
    """
    print('\n[13] penalty algebra: alpha_max scales by 1/sd; the fit is not a pure alpha rescale')
    from sklearn.linear_model import ElasticNet

    sess = data_dic['synthetic_recday'][0]
    prep, reason = v5._prepare_session_with_reason(sess, config)
    if prep is None:
        check('[13] session prepared', False, str(reason))
        return
    keep = ~np.isnan(prep['locs'])
    X = prep['regressors'][keep]
    N = prep['neuron'][keep]
    # A real design matrix, but only its live columns and a bounded number of rows: coordinate
    # descent at the tolerance this identity needs costs minutes on the full 324-column problem
    # and the algebra does not depend on the size.
    X = X[:, X.any(axis=0)]
    if X.shape[0] > 800:
        X, N = X[:800], N[:800]
    print(f'    design {X.shape[0]} rows x {X.shape[1]} live columns')
    rho = config.l1_ratio
    alpha = 0.01                       # the reference's fixed value, the case that matters

    n_checked = 0
    for ni in range(min(3, N.shape[1])):
        y = N[:, ni].astype(float)
        s = float(y.std())
        if not s > 0:
            continue
        n_checked += 1
        am_raw = v5.elasticnet_alpha_max(X, y, rho)
        am_z = v5.elasticnet_alpha_max(X, y / s, rho)
        check(f'[13] neuron {ni}: alpha_max(y/s) == alpha_max(y)/s',
              abs(am_z - am_raw / s) <= 1e-12 * max(1.0, am_raw / s),
              f'{am_z:.6g} vs {am_raw / s:.6g}  (sd = {s:.4g})')

        kw = dict(positive=config.positive, fit_intercept=True, max_iter=100000, tol=1e-11)
        w_z = ElasticNet(alpha=alpha, l1_ratio=rho, **kw).fit(X, y / s).coef_
        a2 = alpha * rho * s + alpha * (1 - rho)
        r2 = alpha * rho * s / a2
        w_eq = ElasticNet(alpha=a2, l1_ratio=r2, **kw).fit(X, y).coef_ / s
        # and it is NOT the same as simply scaling alpha by s (the tempting shortcut)
        w_naive = ElasticNet(alpha=alpha * s, l1_ratio=rho, **kw).fit(X, y).coef_ / s
        scale = max(1e-12, float(np.abs(w_z).max()))
        d_eq = float(np.abs(w_z - w_eq).max())
        d_naive = float(np.abs(w_z - w_naive).max())
        # 1e-3 relative, fixed in advance: these are two DIFFERENT optimisation problems that
        # coincide only at their common optimum, and coordinate descent stops on a duality gap,
        # so the residual is a convergence bound and not an exactness claim (measured 1e-6 to
        # 5e-5 relative). What makes the control non-vacuous is the second assertion: the naive
        # alpha*sd rescale misses by orders of magnitude more.
        check(f'[13] neuron {ni}: fit(y/s; alpha, rho) == fit(y; alpha\', rho\')/s',
              d_eq <= 1e-3 * scale + 1e-9,
              f'max |diff| = {d_eq:.3g} ({d_eq / scale:.1e} relative), max |beta| = {scale:.3g}')
        check(f'[13] neuron {ni}: alpha*sd alone is NOT the equivalent fit',
              d_naive > 100 * max(d_eq, 1e-12),
              f'naive miss {d_naive:.3g} vs exact residual {d_eq:.3g}')
        print(f'    neuron {ni}: sd {s:.4g} | alpha_max {am_raw:.4g} -> {am_z:.4g} | '
              f'exact-vs-naive-alpha*sd max |diff| = {d_naive:.3g}')
    check('[13] at least one neuron was testable', n_checked > 0, f'{n_checked} neurons')


def control_14_zscore_recovery(data_dic, config, seed=0):
    """The known anchored cells must still be recovered with a z-scored target, and the
    low-spike admission that scaling buys is MEASURED (printed, not asserted).

    Only recovery is asserted: whether admitting slow cells is good or bad is a scientific
    judgement, but it has to be quantified somewhere, and `n_spikes_recday` in the unit table is
    what makes it checkable on the real runs.
    """
    print('\n[14] anchored cells survive z-scoring; low-rate admission measured')
    for direction, neuron, lag_name in (('past', 0, 'past'), ('future', 1, 'future')):
        cfg = copy.copy(config)
        cfg.lag_direction = direction
        cfg.y_scaling = 'zscore_recday'
        res = v5.run_cross_validated_regression_v5(
            data_dic, 'synthetic_recday', cfg,
            valid_sessions=list(data_dic['synthetic_recday']))
        mask, lags = v5.identify_nonzero_lag_neurons(res, cfg)
        check(f'[14] {lag_name}-anchored cell recovered under z-scoring', bool(mask[neuron]),
              f'peak lag {lags[neuron]} (built at {config_anchor_lag}), '
              f"r = {res['mean_corrs_nonzero'][neuron]:.3f}")
        check(f'[14] {lag_name}: the pure place cell is still rejected', not bool(mask[2]),
              f'peak lag {lags[2]}')

    # Constant-rate noise cells at three rates, raw vs z-scored, through the real pipeline.
    rng = np.random.default_rng(seed)
    rates = (0.5, 2.0, 8.0)
    dd = {'synthetic_recday': {}}
    for s, sess in data_dic['synthetic_recday'].items():
        T = np.asarray(sess['Neuron_raw']).shape[1]
        neuron = np.vstack([rng.poisson(r * 0.025, size=T) for r in rates]).astype(float)
        dd['synthetic_recday'][s] = {**sess, 'Neuron_raw': neuron}
    # alpha_mode is forced to 'fixed' below -- the reproduction's regime, and the only one in
    # which sd(y) decides whether a neuron gets any beta at all.
    print(f'    {len(rates)} constant-rate cells at {rates} Hz '
          f"(alpha_mode='fixed', elasticnet_alpha={config.elasticnet_alpha})")
    for ysc in ('none', 'zscore_recday'):
        cfg = copy.copy(config)
        cfg.y_scaling = ysc
        cfg.alpha_mode = 'fixed'          # the reproduction's regime: this is where sd matters
        res = v5.run_cross_validated_regression_v5(dd, 'synthetic_recday', cfg,
                                                   valid_sessions=list(dd['synthetic_recday']))
        nzb = np.asarray(res['n_nonzero_betas'], dtype=float)
        mask, _ = v5.identify_nonzero_lag_neurons(res, cfg)
        for i, r in enumerate(rates):
            print(f'      {ysc:14s} {r:4.1f} Hz: spikes {res["n_spikes_recday"][i]:6.0f}, '
                  f'any-beta folds {np.mean(nzb[i] > 0):.2f}, '
                  f'mean r {np.nanmean(res["corrs"][i]):+.3f}, '
                  f'non-zero-lag {bool(mask[i])}')


def control_15_zscore_guards(config):
    """The two ways this flag could fail silently: a Poisson run, and a dead neuron."""
    print('\n[15] guards: Poisson rejects scaling; a silent neuron yields NaN, not an exception')
    try:
        v5.RegressionConfigV5(use_poisson=True, y_scaling='zscore_recday')
        check('[15] Poisson + zscore_recday raises', False, 'no exception')
    except ValueError as exc:
        check('[15] Poisson + zscore_recday raises', True, str(exc)[:70])
    try:
        v5.RegressionConfigV5(y_scaling='z')
        check('[15] an unknown y_scaling raises', False, 'no exception')
    except ValueError as exc:
        check('[15] an unknown y_scaling raises', True, str(exc)[:70])

    rng = np.random.default_rng(0)
    dd = {'synthetic_recday': {}}
    for s in range(3):
        locs, tt = make_behaviour([1, 5, 9, 3], 10, rng)
        T = len(locs)
        neuron = np.vstack([rng.poisson(6.0 * 0.025, size=T),
                            np.zeros(T)]).astype(float)          # neuron 1 never fires
        dd['synthetic_recday'][s] = {'Neuron_raw': neuron, 'Locs_raw': locs.astype(float),
                                     'Trial_times': tt, 'Task': np.array([1, 5, 9, 3]) + s,
                                     'num_trials': 10}
    cfg = copy.copy(config)
    cfg.y_scaling = 'zscore_recday'
    res = v5.run_cross_validated_regression_v5(dd, 'synthetic_recday', cfg,
                                               valid_sessions=list(dd['synthetic_recday']))
    check('[15] the silent neuron is counted as sd = 0', res['n_neurons_sd_zero'] >= 1,
          f"n_neurons_sd_zero = {res['n_neurons_sd_zero']}")
    check('[15] the silent neuron scores NaN in every fold',
          bool(np.all(np.isnan(np.asarray(res['corrs'], dtype=float)[1]))),
          f"corrs[1] = {np.asarray(res['corrs'], dtype=float)[1]}")


def _glum_available():
    try:
        import glum  # noqa: F401
        return True
    except ImportError:
        return False


def _poisson_cfg(config, solver, l1_ratio, alpha):
    cfg = copy.copy(config)
    cfg.use_poisson, cfg.regularize, cfg.alpha_mode = True, True, 'fixed'
    cfg.poisson_solver, cfg.poisson_l1_ratio, cfg.poisson_alpha = solver, float(l1_ratio), alpha
    cfg.poisson_positive = False
    return cfg


def control_16_glum_l2_equivalence(data_dic, config):
    """glum's objective is 1/(2n) deviance + alpha [l1 ||w||_1 + (1 - l1)/2 ||w||^2]; with
    l1_ratio=0 that is sklearn's PoissonRegressor objective, so the two solvers must agree at
    the same alpha (to solver tolerance -- sklearn is lbfgs, glum is irls-cd) and reach the
    same non-zero-lag decisions. A disagreement means the alpha conventions differ, and every
    alpha in the sweep would be mislabelled relative to the L2 reference run."""
    print('\n[16] glum(l1_ratio=0) reproduces sklearn PoissonRegressor at the same alpha')
    if not _glum_available():
        check('[16] glum importable (pip install glum into maze_ephys)', False, 'ImportError')
        return
    alpha = 0.01   # small enough for the betas to be non-trivial (median max|beta| 0.007 at alpha=1)
    _, res_sk = run_direction(data_dic, _poisson_cfg(config, 'sklearn', 0.0, alpha), 'past')
    _, res_gl = run_direction(data_dic, _poisson_cfg(config, 'glum', 0.0, alpha), 'past')
    a = np.asarray(res_sk['cv_coeffs'], dtype=float)
    b = np.asarray(res_gl['cv_coeffs'], dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    d = float(np.max(np.abs(a[both] - b[both]))) if both.any() else np.nan
    scale = float(np.max(np.abs(a[both]))) if both.any() else np.nan
    check('[16] same set of finite fits', bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
          f'{int(np.sum(np.isfinite(a) != np.isfinite(b)))} coefficient slots differ in finiteness')
    check('[16] coefficients agree to solver tolerance (max|diff| <= 2% of max|beta|, or 1e-3)',
          np.isfinite(d) and d <= max(2e-2 * scale, 1e-3),
          f'max|diff| = {d:.2e}, max|beta| = {scale:.2e}')
    # near-tied top-3 betas on a noise cell can flip the argsort under solver tolerance
    # (1 of 23 cells, 2026-09-14); planted cells must agree, the flip rate must stay small
    planted = [PAST_CELL, PLACE_CELL]
    differ = np.flatnonzero(res_sk['nonzero_lag_mask'] != res_gl['nonzero_lag_mask'])
    n_cells = len(res_sk['nonzero_lag_mask'])
    check('[16] planted cells get the same non-zero-lag decision',
          not any(c in differ for c in planted), f'differing cells: {differ.tolist()}')
    check('[16] non-zero-lag decisions differ on <= 5% of cells (solver-tolerance flips only)',
          len(differ) <= max(1, int(0.05 * n_cells)),
          f'{len(differ)}/{n_cells} cells differ: {differ.tolist()}')
    check('[16] same peak lags on the planted cells',
          all(res_sk['peak_lags'][c] == res_gl['peak_lags'][c] for c in planted),
          f"sklearn {list(res_sk['peak_lags'][planted])} vs glum {list(res_gl['peak_lags'][planted])}")


def control_17_lasso_recovery(data_dic, config):
    """The lasso must keep finding what the L2 fit finds -- the planted past-anchored cell at
    its lag, the place cell rejected -- and must get sparser as alpha grows. The all-zero-fit
    fraction is printed at every alpha because under an L1 penalty a fixed alpha is a
    firing-rate filter (control 13's algebra), so a sweep's n changes with alpha."""
    print('\n[17] lasso Poisson recovers the planted cells and gets sparser with alpha')
    if not _glum_available():
        check('[17] glum importable (pip install glum into maze_ephys)', False, 'ImportError')
        return
    alphas = (0.001, 0.003, 0.01)
    mean_nnz = {}
    for alpha in alphas:
        _, res = run_direction(data_dic, _poisson_cfg(config, 'glum', 1.0, alpha), 'past')
        coefs = np.asarray(res['cv_coeffs'], dtype=float)
        fitted = np.isfinite(coefs).all(axis=2)
        nnz = (np.abs(coefs) > 1e-8).sum(axis=2)
        mean_nnz[alpha] = float(nnz[fitted].mean()) if fitted.any() else np.nan
        zero_frac = float(((nnz == 0) & fitted).sum() / max(int(fitted.sum()), 1))
        print(f'    alpha={alpha:g}: mean non-zero betas per fit {mean_nnz[alpha]:.1f}, '
              f'all-zero fits {zero_frac:.0%} of {int(fitted.sum())}')
        if alpha <= 0.003:
            lag = res['peak_lags'][PAST_CELL]
            check(f'[17] lasso alpha={alpha:g}: past-anchored cell peaks at lag {config_anchor_lag}',
                  lag == config_anchor_lag,
                  f"peak lag {lag}, r = {res['mean_corrs'][PAST_CELL]:.3f}")
            check(f'[17] lasso alpha={alpha:g}: past-anchored cell flagged non-zero-lag',
                  bool(res['nonzero_lag_mask'][PAST_CELL]))
            check(f'[17] lasso alpha={alpha:g}: place cell rejected',
                  not bool(res['nonzero_lag_mask'][PLACE_CELL]),
                  f"peak lag {res['peak_lags'][PLACE_CELL]}")
    check('[17] sparsity is monotone in alpha',
          mean_nnz[0.001] >= mean_nnz[0.003] >= mean_nnz[0.01],
          ', '.join(f'{a:g}: {v:.1f}' for a, v in mean_nnz.items()))


def control_18_solver_guards():
    print('\n[18] guards: an L1 ratio or positivity on the sklearn solver raises; run_tag names the penalty')
    for kw in (dict(poisson_l1_ratio=0.5), dict(poisson_positive=True), dict(poisson_solver='irls'),
               dict(poisson_solver='glum', poisson_l1_ratio=1.5)):
        try:
            v5.RegressionConfigV5(use_poisson=True, **kw)
            check(f'[18] {kw} raises', False, 'no exception')
        except ValueError as exc:
            check(f'[18] {kw} raises', True, str(exc)[:70])
    lasso = v5.RegressionConfigV5(use_poisson=True, poisson_solver='glum', poisson_l1_ratio=1.0)
    enet = v5.RegressionConfigV5(use_poisson=True, poisson_solver='glum', poisson_l1_ratio=0.5)
    ref = v5.RegressionConfigV5(use_poisson=True)
    check("[18] run_tag is '_l1' for the lasso", v5.run_tag(lasso) == '_l1', repr(v5.run_tag(lasso)))
    check("[18] run_tag is '_en0.5' for the elastic net", v5.run_tag(enet) == '_en0.5', repr(v5.run_tag(enet)))
    check('[18] the L2 reference keeps its unchanged directory name', v5.run_tag(ref) == '',
          repr(v5.run_tag(ref)))
    check('[18] new fields are in to_dict (so run_config.json records them)',
          all(k in lasso.to_dict() for k in ('poisson_solver', 'poisson_l1_ratio', 'poisson_positive')))


config_anchor_lag = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='fewer sessions/trials')
    ap.add_argument('--poisson', action='store_true', help='use the Poisson branch')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    config = v5.RegressionConfigV5(
        use_poisson=args.poisson, regularize=True,
        # relative alpha, so the synthetic is not silently zeroed by the fixed-alpha default
        alpha_mode='fixed' if args.poisson else 'relative', alpha_frac=0.05)
    n_sessions = 3 if args.quick else 4
    n_trials = 8 if args.quick else 14

    print(f'Building synthetic recday ({n_sessions} sessions x {n_trials} trials), '
          f'anchor = node {5}, phase 1, lag {config_anchor_lag}')
    # 20 noise cells, not the default 6: a false-positive RATE estimated from 6 cells has an
    # SD of ~0.19, which is what made this control flicker.
    data_dic, tasks = make_recday(n_sessions=n_sessions, n_trials=n_trials, n_noise=20,
                                  seed=args.seed, anchor_lag=config_anchor_lag, config=config)
    shapes = {s: d['Neuron_raw'].shape for s, d in data_dic['synthetic_recday'].items()}
    print(f'  Neuron_raw shapes: {shapes}')

    control_1_2_regressors(data_dic, config)
    control_3_4_5_6(data_dic, config)
    control_8_v3_equivalence(data_dic, config)
    control_8b_v4_equivalence(data_dic, config)
    control_9_duration_confound(config, n_cells=100 if args.quick else 300)
    control_10_pref_phase_rules(config)
    control_11_fold_semantics(data_dic, config)
    control_12_zscore_invariance(data_dic, config)
    control_13_zscore_penalty_algebra(data_dic, config)
    control_14_zscore_recovery(data_dic, config, seed=args.seed)
    control_15_zscore_guards(config)
    control_16_glum_l2_equivalence(data_dic, config)
    control_17_lasso_recovery(data_dic, config)
    control_18_solver_guards()

    n_pass = sum(p for _, p in _results)
    print(f'\n{"=" * 70}\n{n_pass}/{len(_results)} controls passed')
    for name, passed in _results:
        if not passed:
            print(f'  FAILED: {name}')
    return 0 if n_pass == len(_results) else 1


if __name__ == '__main__':
    sys.exit(main())
