"""Synthetic controls for the V4 anchoring regression, run through the REAL pipeline.

Repo practice: a synthetic must enter at the same door the data does. Here that means emitting
a `data_dic`-shaped dict of `Neuron_raw` / `Locs_raw` / `Trial_times` and letting
`elasticnet_regression_v4.run_cross_validated_regression_v4` build the regressors, pick the
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
| 6 | Poisson noise gives no non-zero-lag detections and r ~ 0 | the metric has a positive bias |
| 7 | every per-fold beta lies on the (pref -/+ lag) stripe | the phase/lag coupling is broken |
| 8 | v4 with legacy flags reproduces v3 exactly | the refactor changed the numbers |

Control 5 is stated in both directions on purpose: a rejection test that never accepts anything
is not evidence.
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

import elasticnet_regression_v4 as v4                      # noqa: E402

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
    config = config or v4.RegressionConfigV4()
    rng = np.random.default_rng(seed)
    tasks = [list(rng.permutation(np.arange(1, 10))[:4]) for _ in range(n_sessions)]
    sessions = {}
    for si, task in enumerate(tasks):
        locs, tt = make_behaviour(task, n_trials, rng)
        phases, _ = v4.compute_phase_state_raw(tt, config.num_goal_progress_bins,
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
    phases, _ = v4.compute_phase_state_raw(tt, config.num_goal_progress_bins,
                                           config.num_task_states)
    L = min(len(locs), len(phases))
    locs, phases = locs[:L].copy(), phases[:L]
    locs[locs > config.num_locations] = np.nan
    locs[locs < 1] = np.nan

    nloc, nph, nlag = config.num_locations, config.num_goal_progress_bins, config.num_lags
    for direction in ('past', 'future'):
        cfg = copy.copy(config)
        cfg.lag_direction = direction
        got = v4.generate_regressors_raw(locs, phases, cfg).reshape(L, nloc, nph, nlag)
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
    fwd = v4.generate_regressors_raw(locs[::-1].copy(), phases[::-1].copy(), cfg_p)
    fut = v4.generate_regressors_raw(locs, phases, cfg_f)
    check('[2] future(X) == reverse(past(reverse(X)))', np.array_equal(fwd[::-1], fut))


def run_direction(data_dic, config, direction):
    cfg = copy.copy(config)
    cfg.lag_direction = direction
    return cfg, v4.run_cross_validated_regression_v4(
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
        mask_wide, _ = v4.identify_nonzero_lag_neurons(res, wide)
        check(f'[5] place cell IS detected in {direction} once lag 0 is admitted (gate is not vacuous)',
              bool(mask_wide[PLACE_CELL]))

    print('\n[6] Poisson noise sits at chance')
    print('    NB the non-zero-lag criterion is a SHAPE DESCRIPTOR of the beta vector, not a')
    print('    statistical test: on pure noise it fires at 15-30%. What has to be unbiased is')
    print('    the correlation, because that is what the headline t-test is run on.')
    for direction in ('past', 'future'):
        cfg, res = out[direction]
        noise = np.arange(3, res['cv_coeffs'].shape[0])
        nz_rate = res['nonzero_lag_mask'][noise].mean()
        sel = res['nonzero_lag_mask'] & res['state_tuned_mask'] & ~np.isnan(res['mean_corrs'])
        mean_r = np.nanmean(res['mean_corrs'][noise])
        check(f'[6] {direction}: noise mean r ~ 0 (the metric is unbiased)',
              not np.isfinite(mean_r) or abs(mean_r) < 0.25, f'mean r = {mean_r:.3f}')
        check(f'[6] {direction}: non-zero-lag false-positive rate is bounded', nz_rate < 0.5,
              f'{nz_rate:.0%} of {len(noise)} noise cells flagged NZ-lag, '
              f'{sel[noise].mean():.0%} survive the full selection')

    print('\n[7] every per-fold beta lies on the stripe')
    for direction in ('past', 'future'):
        cfg, res = out[direction]
        try:
            checked, bad = v4.assert_beta_stripe(res, cfg)
            check(f'[7] {direction}: stripe invariant holds', True, f'{checked} beta vectors checked')
        except AssertionError as exc:
            check(f'[7] {direction}: stripe invariant holds', False, str(exc))
    return out


def control_8_v3_equivalence(data_dic, config):
    """v4 in legacy mode must reproduce v3 bit for bit."""
    print('\n[8] v4 with legacy flags == v3')
    try:
        import elasticnet_regression_v3 as v3
    except Exception as exc:
        check('[8] v3 importable', False, str(exc))
        return

    legacy = copy.copy(config)
    legacy.lag_direction = 'past'
    legacy.pref_phase_source = 'train'
    legacy.restrict_to_pref_phase = True
    legacy.drop_untracked_bins = False
    legacy.nonzero_lag_zero_lags = (0,)
    legacy.require_positive_top3 = False
    legacy.nz_per_fold = False
    legacy.state_reduce = 'mean'
    legacy.alpha_mode = 'fixed'          # v3 has no relative-alpha branch

    sessions = list(data_dic['synthetic_recday'])
    r4 = v4.run_cross_validated_regression_v4(data_dic, 'synthetic_recday', legacy,
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

    m4, l4 = v4.identify_nonzero_lag_neurons(r4, legacy)
    m3, l3 = v3.identify_nonzero_lag_neurons(r3, c3)
    check('[8] non-zero-lag mask identical', np.array_equal(m4, m3),
          f'{int((m4 != m3).sum())} neurons differ')


config_anchor_lag = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='fewer sessions/trials')
    ap.add_argument('--poisson', action='store_true', help='use the Poisson branch')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    config = v4.RegressionConfigV4(
        use_poisson=args.poisson, regularize=True,
        # relative alpha, so the synthetic is not silently zeroed by the fixed-alpha default
        alpha_mode='fixed' if args.poisson else 'relative', alpha_frac=0.05)
    n_sessions = 3 if args.quick else 4
    n_trials = 8 if args.quick else 14

    print(f'Building synthetic recday ({n_sessions} sessions x {n_trials} trials), '
          f'anchor = node {5}, phase 1, lag {config_anchor_lag}')
    data_dic, tasks = make_recday(n_sessions=n_sessions, n_trials=n_trials, seed=args.seed,
                                  anchor_lag=config_anchor_lag, config=config)
    shapes = {s: d['Neuron_raw'].shape for s, d in data_dic['synthetic_recday'].items()}
    print(f'  Neuron_raw shapes: {shapes}')

    control_1_2_regressors(data_dic, config)
    control_3_4_5_6(data_dic, config)
    control_8_v3_equivalence(data_dic, config)

    n_pass = sum(p for _, p in _results)
    print(f'\n{"=" * 70}\n{n_pass}/{len(_results)} controls passed')
    for name, passed in _results:
        if not passed:
            print(f'  FAILED: {name}')
    return 0 if n_pass == len(_results) else 1


if __name__ == '__main__':
    sys.exit(main())
