"""Synthetic controls for lagged_spatial_maps: planted maps through the real loader.

    python lagged_spatial_maps_synthetics.py [--quick]

Each case writes a data_dic-free recday in El-Gaby's file layout (Phase_shifted_maps,
Task_data, Num_trials, State_, Neuron_raw, combined_ABCDonly_days) to a scratch
Intermediate_objects dir and reads it back with `load_recday`, so the door is the one the data
uses. Exit 0 only if every check passes, including the ones that must NOT reject.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import scipy.stats as st

import lagged_spatial_maps as lsm

S_TASKS = 6          # distinct tasks; plus one repeat and one zero-trial session appended
LAG = 5              # planted lag, outside both exclusion sets


def make_tasks(rng, n):
    return np.stack([rng.choice(np.arange(1, 10), 4, replace=False) for _ in range(n)])


def write_recday(io_dir, rd, maps_task, tasks, state):
    """maps_task (N, S_TASKS, 12, 21) -> files for S_TASKS + repeat + zero-trial sessions."""
    N = maps_task.shape[0]
    rng = np.random.default_rng(sum(map(ord, rd)))    # str hash is salted per process
    # repeat of task 0 (excluded by non_repeat) and a zero-trial session with NaN maps
    rep = maps_task[:, :1] + rng.normal(0, 5, maps_task[:, :1].shape)
    empty = np.full_like(maps_task[:, :1], np.nan)
    maps = np.concatenate([maps_task, rep, empty], axis=1)
    all_tasks = np.concatenate([tasks, tasks[:1], make_tasks(rng, 1)])
    num_trials = np.r_[np.full(S_TASKS, 10), 10, 0]
    f = lambda name: os.path.join(io_dir, name)            # noqa: E731
    np.save(f(f'Spatial_anchoring_Phase_shifted_maps_{rd}.npy'), maps)
    np.save(f(f'Task_data_{rd}.npy'), all_tasks)
    np.save(f(f'Num_trials_{rd}.npy'), num_trials)
    np.save(f(f'State_{rd}.npy'), state)
    for s in range(maps.shape[1]):
        np.save(f(f'Neuron_raw_{rd}_{s}.npy'), rng.poisson(0.1, (N, 400)).astype(float))


def noise_maps(rng, N):
    return rng.gamma(2.0, 0.5, (N, S_TASKS, lsm.N_LAGS, 21))


def planted_place(rng, N, lag, snr=3.0):
    """Same spatial pattern in every task at `lag`; independent noise elsewhere."""
    m = noise_maps(rng, N)
    P = rng.gamma(2.0, 0.5, (N, 1, 21)) * snr
    m[:, :, lag] = P + rng.gamma(2.0, 0.5, (N, S_TASKS, 21))
    return m


def planted_goal(rng, N, lag, tasks, profile=(3.0, 1.0, 0.0, 0.0)):
    """Reward-following: at `lag` each task's A-D nodes carry the same letter profile."""
    m = noise_maps(rng, N)
    for s in range(S_TASKS):
        for k, node in enumerate(tasks[s]):
            m[:, s, lag, node - 1] += profile[k] * 2.0
    return m


def run_case(io_dir, rd, maps, tasks, state=None):
    state = np.ones(maps.shape[0], bool) if state is None else state
    write_recday(io_dir, rd, maps, tasks, state)
    return lsm.load_recday('lec', rd, io_dir=io_dir)


CHECKS = []


def check(name, ok, detail=''):
    CHECKS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}  {detail}')


def main(quick):
    rng = np.random.default_rng(0)
    io_dir = tempfile.mkdtemp(prefix='lsm_syn_')
    N = 30 if quick else 100
    tasks = make_tasks(rng, S_TASKS)

    print('(a) consistent map at lag 5 only')
    rec = run_case(io_dir, 'syn_a', planted_place(rng, N, LAG), tasks)
    check('loader drops the repeat and the zero-trial session',
          list(rec['non_repeat']) == list(range(S_TASKS)), str(rec['non_repeat']))
    h = lsm.recday_heldout(rec)
    frac5 = np.mean(h['max_lags'] == LAG)
    check('best lag recovered in held-out folds', frac5 > 0.9, f'{frac5:.2f} of folds at lag 5')
    check('held-out r high', np.nanmean(h['r_all']) > 0.5, f'mean r_all {np.nanmean(h["r_all"]):.2f}')
    check('lag 5 kept in the nonzero and strict sets',
          np.isfinite(h['r_nonzero']).mean() > 0.9 and np.isfinite(h['r_strict']).mean() > 0.9,
          f'{np.isfinite(h["r_nonzero"]).mean():.2f} / {np.isfinite(h["r_strict"]).mean():.2f}')

    print('(b) consistent map at lag 0')
    rec = run_case(io_dir, 'syn_b', planted_place(rng, N, 0), tasks)
    h = lsm.recday_heldout(rec)
    check('in the all-tuned set', np.nanmean(h['r_all']) > 0.5, f'mean r_all {np.nanmean(h["r_all"]):.2f}')
    check('excluded from nonzero and strict by the modal-lag rule',
          np.isnan(h['r_nonzero']).all() and np.isnan(h['r_strict']).all())

    print('(c) maps independent across tasks: the t-test must not reject')
    n_pop = 10 if quick else 20
    rejects, means = 0, []
    for i in range(n_pop):
        rec = run_case(io_dir, f'syn_c{i}', noise_maps(rng, N), make_tasks(rng, S_TASKS))
        h = lsm.recday_heldout(rec)
        v = h['r_all'][np.isfinite(h['r_all'])]
        means.append(v.mean())
        rejects += st.ttest_1samp(v, 0).pvalue < 0.05
    check('false-positive rate near 5%', rejects <= max(2, int(0.15 * n_pop)),
          f'{rejects}/{n_pop} populations rejected; mean r {np.mean(means):+.3f}')

    print('(d) NaN nodes inside a used session pass through')
    m = planted_place(rng, N, LAG)
    m[:, 2, :, 3] = np.nan
    rec = run_case(io_dir, 'syn_d', m, tasks)
    h = lsm.recday_heldout(rec)
    check('finite scores with a missing node', np.isfinite(h['r_all']).all())

    print('(e) cell-34 pooling respects the state mask and recday order')
    state = np.zeros(N, bool)
    state[: N // 2] = True
    rec = run_case(io_dir, 'syn_e', planted_place(rng, N, LAG), tasks, state)
    h = lsm.recday_heldout(rec)
    out = lsm.cell34({'syn_e': h}, ['syn_e'], {'syn_e': state})
    check('n equals the tuned finite count', out['all'][0] == int(np.isfinite(h['r_all'][state]).sum()),
          f'n={out["all"][0]}')

    n_null = 100 if quick else 200
    print('(f) frame test: place-anchored cells')
    rec = run_case(io_dir, 'syn_f', planted_place(rng, N, LAG), tasks)
    df = lsm.recday_frame_test(rec, n_null=n_null)
    fp, fg = (df.p_place < 0.05).mean(), (df.p_goal < 0.05).mean()
    check('place frame rejects', fp > 0.8, f'{fp:.2f} of cells p_place < 0.05')
    check('goal frame mostly does not', fg < 0.3, f'{fg:.2f} of cells p_goal < 0.05')

    print('(g) frame test: reward-following cells')
    rec = run_case(io_dir, 'syn_g', planted_goal(rng, N, LAG, tasks), tasks)
    df = lsm.recday_frame_test(rec, n_null=n_null)
    fp, fg = (df.p_place < 0.05).mean(), (df.p_goal < 0.05).mean()
    check('goal frame rejects', fg > 0.8, f'{fg:.2f} of cells p_goal < 0.05')
    check('place frame mostly does not', fp < 0.3, f'{fp:.2f} of cells p_place < 0.05')
    check('r_goal > r_place', (df.r_goal > df.r_place).mean() > 0.8,
          f'{(df.r_goal > df.r_place).mean():.2f} of cells')

    print('(h) frame test: noise cells must not reject in either frame')
    rec = run_case(io_dir, 'syn_h', noise_maps(rng, 3 * N), tasks)
    df = lsm.recday_frame_test(rec, n_null=n_null)
    fp, fg = (df.p_place < 0.05).mean(), (df.p_goal < 0.05).mean()
    check('place frame near 5%', fp < 0.12, f'{fp:.3f}')
    check('goal frame near 5%', fg < 0.12, f'{fg:.3f}')

    n_fail = sum(not ok for _, ok in CHECKS)
    print(f'\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    sys.exit(main(ap.parse_args().quick))
