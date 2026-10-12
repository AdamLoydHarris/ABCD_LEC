"""Synthetic controls for anchor_distribution, entering through V5's door.

    python anchor_distribution_synthetics.py [--quick]

Each synthetic recday is behaviour + spikes from the V5 synthetics' own builders
(`make_behaviour`, `anchored_rate`) with the tasks fixed by construction. It is fitted by
`run_cross_validated_regression_v5` with the spec-12 config, exported to npz, and read back by
the same `load_v5` / `v5_fold_anchors` the data uses; Task_data, Num_trials and a visit table go
to a scratch run tree. Exit 0 only if every check passes, including the accept case.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

import anchor_distribution as ad
import elasticnet_regression_v5 as v5
import elasticnet_v5_synthetics as syn

ALWAYS_GOAL = 3        # node that is a goal in every task
NEVER_GOAL = 5         # centre node: on many routes, never a goal
LAG = 5
N_SESSIONS = 6
N_TRIALS = 14


def spec12_config():
    runs_dir = ad.V5_RUNS['lec']
    import json
    stored = json.load(open(os.path.join(runs_dir, 'run_config.json')))['config']
    init = {k: v for k, v in stored.items()
            if k in v5.RegressionConfigV5.__init__.__code__.co_varnames and k != 'self'}
    return v5.RegressionConfigV5(**init)


def make_tasks(rng):
    """Every task holds ALWAYS_GOAL, none holds NEVER_GOAL."""
    pool = [n for n in range(1, 10) if n not in (ALWAYS_GOAL, NEVER_GOAL)]
    return [list(rng.permutation([ALWAYS_GOAL] + list(rng.choice(pool, 3, replace=False))))
            for _ in range(N_SESSIONS)]


def behaviour(tasks, config, rng):
    out = []
    for task in tasks:
        locs, tt = syn.make_behaviour(task, N_TRIALS, rng)
        phases, _ = v5.compute_phase_state_raw(tt, config.num_goal_progress_bins,
                                               config.num_task_states)
        L = min(len(locs), len(phases))
        out.append((locs[:L], phases[:L], tt))
    return out


def visits(beh):
    """(S, 3, 9) bins at each (phase, node): the synthetic stand-in for num_passes_all."""
    v = np.zeros((len(beh), 3, 9))
    for s, (locs, phases, _) in enumerate(beh):
        ok = (locs >= 1) & (locs <= 9) & np.isfinite(phases)
        np.add.at(v[s], (phases[ok].astype(int), locs[ok].astype(int) - 1), 1)
    return v


def build(rd, anchors, tasks, beh, config, rng, io_dir, run_dir, amp=6.0, base=0.15):
    """anchors: list of (node 1-9, phase) planted at LAG; fit, export, write the task files."""
    sessions = {}
    for s, (locs, phases, tt) in enumerate(beh):
        rates = [syn.anchored_rate(locs, phases, config, node - 1, ph, LAG, 'past', amp, base)
                 for node, ph in anchors]
        spikes = rng.poisson(np.clip(np.asarray(rates), 0, None)).astype(float)
        sessions[s] = {'Neuron_raw': spikes, 'Locs_raw': locs.astype(int), 'Trial_times': tt,
                       'Task': np.asarray(tasks[s])}
    res = v5.run_cross_validated_regression_v5({rd: sessions}, rd, config,
                                               valid_sessions=list(sessions))
    res['mouse_recday'] = rd
    v5.export_regression_outputs(res, config, run_dir)
    np.save(os.path.join(io_dir, f'Task_data_{rd}.npy'), np.asarray(tasks))
    np.save(os.path.join(io_dir, f'Num_trials_{rd}.npy'), np.full(len(tasks), N_TRIALS))
    np.save(os.path.join(io_dir, f'Spatial_anchoring_num_passes_all_{rd}.npy'), visits(beh))


def draw_by_visits(v, k, rng):
    """k (node, phase) anchors drawn in proportion to the recday's visits."""
    p = v.sum(axis=0).ravel()
    idx = rng.choice(p.size, k, p=p / p.sum())
    return [(int(i % 9) + 1, int(i // 9)) for i in idx]


CHECKS = []


def check(name, ok, detail=''):
    CHECKS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}  {detail}')


def main(quick):
    rng = np.random.default_rng(0)
    config = spec12_config()
    io_dir = tempfile.mkdtemp(prefix='ad_syn_io_')
    run_dir = tempfile.mkdtemp(prefix='ad_syn_run_')
    k = 12 if quick else 30

    print('(a) single planted anchors are scored against the training tasks')
    tasks = make_tasks(rng)
    beh = behaviour(tasks, config, rng)
    build('synA_20250101_20250102', [(ALWAYS_GOAL, 0)] * 3 + [(NEVER_GOAL, 1)] * 3,
          tasks, beh, config, rng, io_dir, run_dir)

    print('(b) population drawn by visits, and one 60% reward-anchored')
    tasks_b = make_tasks(rng)
    beh_b = behaviour(tasks_b, config, rng)
    v_b = visits(beh_b)
    build('synB_20250101_20250102', draw_by_visits(v_b, k, rng), tasks_b, beh_b, config, rng,
          io_dir, run_dir)
    n_rew = int(round(0.6 * k))
    build('synC_20250101_20250102',
          [(ALWAYS_GOAL, 0)] * n_rew + draw_by_visits(v_b, k - n_rew, rng),
          tasks_b, beh_b, config, rng, io_dir, run_dir)

    runs, cfg = ad.load_v5(run_dir)
    folds = ad.v5_fold_anchors(runs, cfg, 'lec', io_dir)
    neur = ad.neuron_scores(folds)

    a = neur[neur.recday.str.startswith('synA')].set_index('neuron')
    fa = folds[folds.recday.str.startswith('synA')]
    rew = fa[fa.neuron < 3]
    check('reward-anchored cells recovered at the always-goal node, phase 0',
          ((rew['loc'] == ALWAYS_GOAL - 1) & (rew.phase == 0)).mean() > 0.8,
          f'{((rew["loc"] == ALWAYS_GOAL - 1) & (rew.phase == 0)).mean():.2f} of folds; '
          f'lags {sorted(set(rew.lag))}')
    check('their reward_frac is 1', (a.loc[[0, 1, 2], 'reward_frac'] > 0.8).all(),
          str(a.loc[[0, 1, 2], 'reward_frac'].round(2).tolist()))
    never = fa[fa.neuron >= 3]
    check('never-goal cells recovered at the centre node',
          (never['loc'] == NEVER_GOAL - 1).mean() > 0.8,
          f'{(never["loc"] == NEVER_GOAL - 1).mean():.2f} of folds')
    check('their goal_frac is 0', (a.loc[[3, 4, 5], 'goal_frac'] < 0.2).all(),
          str(a.loc[[3, 4, 5], 'goal_frac'].round(2).tolist()))

    for rd, want, label in (('synB', False, 'visit-drawn population: occupancy null accepts'),
                            ('synC', True, '60% reward-anchored: occupancy null rejects')):
        keys = neur[neur.recday.str.startswith(rd)].set_index(['recday', 'neuron']).index
        t = ad.occupancy_null(folds, keys, 'reward_frac', n_null=500 if quick else 2000)
        p = float(t.p.iloc[0])
        check(label, (p < 0.05) == want,
              f'observed {t.observed.iloc[0]:.3f} vs null {t.null_mean.iloc[0]:.3f}, p {p:.3f}')

    n_fail = sum(not ok for _, ok in CHECKS)
    print(f'\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    sys.exit(main(ap.parse_args().quick))
