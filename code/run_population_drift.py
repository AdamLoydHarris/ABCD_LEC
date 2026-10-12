"""Kanter population measures on real data: one recday per job, a stamped merge (M2, WP3 part 1).

    python code/run_population_drift.py --dataset lec --recday ah10_20250616_20250617
    python code/run_population_drift.py --dataset lec --merge [--min-trials 10]
    sbatch --cpus-per-task=8 ... run_population_drift.py --dataset lec --gate   # planted populations

Per recday: the analysis sessions (the Stage 1 rule) with counts from recording start. Unit sets:
all units; 20 random units (20 draws); per region, the units >= 50 um from a region boundary, all
of them and 20 random (20 draws), when a region has >= 20 such units. Unit order is depth order
on the probes, so sets are random draws, never the first N. Per set and session: distance vs lag
(1-min epochs of 10 s bins), kNN epoch decoding within session (5-fold) and on the held-out
session, with confusion matrices, and trajectory acceleration (1 s bins) at the first A, each
reward A, every reward and synchrony onsets. Entry is bin 0 and has no pre-window, so no
acceleration. Shards and merge as `run_event_kernels` (same refusals). The gate needs 8 CPUs, as
the M1 reference: at 2 BLAS threads two distance curves differed from it by 6.9e-18 (exact at 8).

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import population_drift as pdft                                        # noqa: E402
import population_drift_synthetics as pds                              # noqa: E402
import run_event_kernels as rek                                        # noqa: E402

IS_PFC = 'mFC_data' in HERE
N_MATCH, N_DRAWS = 20, 20
KNN = dict(bin_s=10.0, epoch_s=60.0, max_epochs=18, k=10, n_folds=5)
ACC = dict(bin_s=1.0, window=5)
EVENTS = ('first_A', 'A', 'reward')
SKIP_GROUPS = ('fibre/other',)
PLANTED = ('noise', 'locked', 'drift', 'boundary')
N_PLANTED = 60                                 # as the full P-gate, so its populations are reproduced exactly
DRIFT_DIR = os.path.join(ek.REPO, 'data', 'population_drift')


def arm_name(dataset, min_trials, synthetic=None) -> str:
    a = f'{dataset}_mt{min_trials}'
    return f'synthetic-{synthetic}_{a}' if synthetic else a


def stamp(dataset, min_trials, seed, synthetic=None) -> dict:
    return dict(runner='run_population_drift', dataset=dataset, min_trials=int(min_trials), seed=int(seed),
                synthetic=synthetic, n_match=N_MATCH, n_draws=N_DRAWS, knn=dict(KNN), acc=dict(ACC),
                events=list(EVENTS), boundary_margin_um=50.0, skip_groups=list(SKIP_GROUPS),
                sha1_population_drift=rek._sha1(pdft.__file__), sha1_event_kernels=rek._sha1(ek.__file__),
                sha1_runner=rek._sha1(os.path.abspath(__file__)), inputs=rek.input_versions(dataset))


def knn_predictions(sessions_counts, seed=0, bin_s=10.0, epoch_s=60.0, max_epochs=18, k=10, n_folds=5):
    """`population_drift.knn_epochs` keeping its predictions: the same rng sequence, so the same
    calls. Returns ([{'y', 'within', 'across'} per session], chance MAE)."""
    rng = np.random.default_rng(seed)
    data = []
    for c in sessions_counts:
        X = pdft.soft_normalize(pdft.rebin(c, bin_s)).T
        y = pdft.epoch_labels(len(X), bin_s, epoch_s)
        keep = y < max_epochs
        data.append((X[keep], y[keep]))
    out = []
    for i, (X, y) in enumerate(data):
        order = rng.permutation(len(y))
        pred = np.empty_like(y)
        for f in np.array_split(order, n_folds):
            tr = np.setdiff1d(order, f)
            pred[f] = pdft._knn_predict(X[tr], y[tr], X[f], k)
        others = [j for j in range(len(data)) if j != i]
        across = (pdft._knn_predict(np.vstack([data[j][0] for j in others]),
                                    np.concatenate([data[j][1] for j in others]), X, k) if others else None)
        out.append(dict(y=y, within=pred, across=across))
    yy = np.concatenate([d[1] for d in data])
    return out, float(np.mean(np.abs(rng.permutation(yy) - yy)))


def unit_sets(units: pd.DataFrame, seed: int) -> list:
    """[(set, group, draw, unit indices)]: all; all_n20 draws; per region (off-boundary) full + draws."""
    n = len(units)
    sets = [('all', 'all', 0, np.arange(n))]
    if n >= N_MATCH:
        sets += [('all_n20', 'all', d, idx) for d, idx in enumerate(pdft.subsample(n, N_MATCH, N_DRAWS, seed=seed))]
    groups = [g for g in sorted(units['group'].unique()) if g not in SKIP_GROUPS]
    if len(groups) > 1:
        for k, g in enumerate(groups):
            elig = np.flatnonzero((units['group'] == g).to_numpy() & ~units['near_boundary'].to_numpy(bool))
            if len(elig) < N_MATCH:
                continue
            sets.append(('region_full', g, 0, elig))
            sets += [('region', g, d, elig[idx])
                     for d, idx in enumerate(pdft.subsample(len(elig), N_MATCH, N_DRAWS, seed=seed + 1 + k))]
    return sets


def measure(counts: list, sessions: list, idx: np.ndarray, seed: int) -> dict:
    """Every measure for one unit set, per session."""
    C = [c[idx] for c in counts]
    preds, chance_mae = knn_predictions(C, seed=seed, **KNN)
    nc = KNN['max_epochs']
    conf_w, conf_a = np.zeros((nc, nc), np.int32), np.zeros((nc, nc), np.int32)
    rows = []
    for s, r in zip(sessions, preds):
        y = r['y']
        row = dict(session=s['session'], n_bins=len(y), within_acc=float(np.mean(r['within'] == y)),
                   within_mae=float(np.mean(np.abs(r['within'] - y))), chance_acc=1.0 / len(np.unique(y)),
                   across_acc=np.nan, across_mae=np.nan)
        np.add.at(conf_w, (y, r['within']), 1)
        if r['across'] is not None:
            row.update(across_acc=float(np.mean(r['across'] == y)), across_mae=float(np.mean(np.abs(r['across'] - y))))
            np.add.at(conf_a, (y, r['across']), 1)
        rows.append(row)
    dist = np.array([pdft.distance_vs_lag(c, bin_s=KNN['bin_s'], epoch_s=KNN['epoch_s']) for c in C])
    acc = []
    for s, c in zip(sessions, C):
        evs = {e: ek.event_bins(s, e) for e in EVENTS}
        evs['sync'] = pdft.synchrony_onsets(c, bin_s=ACC['bin_s'])
        for e, b in evs.items():
            a = pdft.event_acceleration(c, b, bin_s=ACC['bin_s'], window=ACC['window'])
            acc.append(dict(session=s['session'], event=e, n=int(a['n']), z=a['z'], mean=a['mean']))
    return dict(knn=pd.DataFrame(rows), chance_mae=chance_mae, conf_within=conf_w, conf_across=conf_a,
                dist=dist, acc=acc)


def fit_recday(dataset, recday, *, min_trials, seed, synthetic=None) -> dict:
    t0 = time.time()
    rec = ek.load_recday(recday, dataset)
    sessions = ek.analysis_sessions(rec, min_trials)
    if synthetic:
        rng = np.random.default_rng(seed)                      # one stream in the P-gate's order
        pops = {k: pds.planted(k, sessions, N_PLANTED, rng) for k in PLANTED}
        counts = [np.vstack([pops[k][i] for k in PLANTED]) for i in range(len(sessions))]
        units = pd.DataFrame(dict(unit=np.arange(len(PLANTED) * N_PLANTED),
                                  group=np.repeat(PLANTED, N_PLANTED), near_boundary=False))
        units['recday'], units['mouse'] = recday, recday.split('_')[0]
    else:
        counts = [s['counts'] for s in sessions]
        units = rek.unit_table(rec)
    data = dict(recday=recday, dataset=dataset, units=units, sessions=rek.session_table(sessions), skipped=None,
                host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if not sessions:
        data['skipped'] = f'no analysis session at min_trials {min_trials}'
    else:
        data['sets'] = [dict(set=name, group=g, draw=d, units=idx, **measure(counts, sessions, idx, seed))
                        for name, g, d, idx in unit_sets(units, seed)]
    data['elapsed_s'] = time.time() - t0
    return data


def _nanmean_rows(a):
    with np.errstate(invalid='ignore'):
        return np.nanmean(a, axis=0) if len(a) else np.full(2 * ACC['window'] + 1, np.nan)


def merge_dir(shard_dir: str, expect: int = rek.N_RECDAYS) -> dict:
    """Long tables: kNN per session, distance vs lag and acceleration z per recday (session means)."""
    cfg, shards = rek.load_shards(shard_dir, key=lambda d: d['recday'])
    if expect and len(shards) != expect:
        raise ValueError(f'{len(shards)} recdays, expected {expect}; pass --expect-recdays 0 to merge anyway')
    knn, dist, acc, conf = [], [], [], {}
    for d in shards:
        if d['skipped']:
            continue
        mouse = d['recday'].split('_')[0]
        for r in d['sets']:
            base = dict(recday=d['recday'], mouse=mouse, set=r['set'], group=r['group'], draw=r['draw'],
                        n_units=len(r['units']))
            knn.append(r['knn'].assign(**base, chance_mae=r['chance_mae']))
            with np.errstate(invalid='ignore'):
                curve = np.nanmean(r['dist'], axis=0)
            dist += [dict(base, lag_min=L, distance=float(v)) for L, v in enumerate(curve)]
            a = pd.DataFrame(r['acc'])
            for e, sub in a.groupby('event', sort=False):
                ok = sub[sub['n'] > 0]
                z = _nanmean_rows(np.stack(ok['z'].to_numpy())) if len(ok) else _nanmean_rows([])
                lags = np.arange(-ACC['window'], ACC['window'] + 1)
                acc += [dict(base, event=e, lag_s=int(L) * ACC['bin_s'], z=float(v), n_events=int(sub['n'].sum()),
                             n_sessions=len(ok)) for L, v in zip(lags, z)]
            conf[(d['recday'], r['set'], r['group'], r['draw'])] = (r['conf_within'], r['conf_across'])
    summary = pd.DataFrame([dict(recday=d['recday'], n_units=len(d['units']), n_sessions=len(d['sessions']),
                                 n_sets=len(d.get('sets', [])), skipped=d['skipped'], elapsed_s=d.get('elapsed_s'),
                                 slurm_job_id=d.get('slurm_job_id')) for d in shards])
    return dict(config=cfg, knn=pd.concat(knn, ignore_index=True) if knn else pd.DataFrame(),
                dist=pd.DataFrame(dist), acc=pd.DataFrame(acc), confusion=conf,
                summary=summary.sort_values('recday').reset_index(drop=True),
                units={d['recday']: d['units'] for d in shards})


def write_merged(M: dict, arm: str) -> str:
    p = rek.versioned(os.path.join(DRIFT_DIR, f'drift_{arm}.pkl'))
    with open(p, 'wb') as fh:
        pickle.dump(M, fh, protocol=pickle.HIGHEST_PROTOCOL)
    stem = p[:-len('.pkl')]
    for k in ('knn', 'dist', 'acc', 'summary'):
        M[k].to_csv(f'{stem}_{k}.csv', index=False)
    return p


def p_verdicts(M: dict, set_name: str) -> pd.DataFrame:
    """The P1-P4 rules of `population_drift_synthetics`, read from merged tables, per planted kind."""
    k = M['knn'][M['knn']['set'] == set_name].groupby('group')[['within_acc', 'across_acc', 'chance_acc']].mean()
    dl = M['dist'][M['dist']['set'] == set_name].groupby(['group', 'lag_min'])['distance'].mean().unstack()
    a = M['acc'][(M['acc']['set'] == set_name) & (M['acc']['event'] == 'reward')]
    z = a.groupby(['group', 'lag_s'])['z'].mean().unstack()
    ch = k['chance_acc']
    rows = [('P1', 'noise within <= chance + 0.05', k.loc['noise', 'within_acc'], k.loc['noise', 'within_acc'] <= ch['noise'] + 0.05),
            ('P1', 'noise across <= chance + 0.05', k.loc['noise', 'across_acc'], k.loc['noise', 'across_acc'] <= ch['noise'] + 0.05),
            ('P1', 'noise |distance lag10 - lag1| < 0.01', dl.loc['noise', 10] - dl.loc['noise', 1], abs(dl.loc['noise', 10] - dl.loc['noise', 1]) < 0.01),
            ('P2', 'locked across >= 3x chance', k.loc['locked', 'across_acc'] / ch['locked'], k.loc['locked', 'across_acc'] >= 3 * ch['locked']),
            ('P3', 'drift within >= 3x chance', k.loc['drift', 'within_acc'] / ch['drift'], k.loc['drift', 'within_acc'] >= 3 * ch['drift']),
            ('P3', 'drift across <= 1.5x chance', k.loc['drift', 'across_acc'] / ch['drift'], k.loc['drift', 'across_acc'] <= 1.5 * ch['drift']),
            ('P3', 'drift distance grows with lag', dl.loc['drift', 10] - dl.loc['drift', 1], dl.loc['drift', 10] > dl.loc['drift', 1]),
            ('P4', 'boundary z at lag 0 or 1 > 3', max(z.loc['boundary', 0.0], z.loc['boundary', 1.0]), max(z.loc['boundary', 0.0], z.loc['boundary', 1.0]) > 3)]
    rows += [('P4', f'{g} |z| at rewards < 3', float(np.nanmax(np.abs(z.loc[g]))), float(np.nanmax(np.abs(z.loc[g]))) < 3)
             for g in ('noise', 'locked', 'drift')]
    return pd.DataFrame(rows, columns=['gate', 'check', 'value', 'passed'])


def gate(dataset: str, seed: int = 0) -> pd.DataFrame:
    """Planted P1-P4 populations through this path: equal to the M1 full P-gate pickle (same
    populations, same kNN calls), P verdicts from the merged tables, and the merge refusal."""
    rd = pds.DEFAULT_RECDAY[dataset]
    ref_path = rek._latest_pickle(os.path.join(pds.OUT_DIR, f'population_drift_synthetics_{dataset}_{rd}'))
    with open(ref_path, 'rb') as fh:
        ref = pickle.load(fh)
    root = os.path.join(DRIFT_DIR, 'runner_gate', f"{dataset}_{time.strftime('%Y%m%d_%H%M%S')}")
    arm = arm_name(dataset, ek.MIN_TRIALS, synthetic='planted')
    st = stamp(dataset, ek.MIN_TRIALS, seed, synthetic='planted')
    adir = os.path.join(root, arm)
    rows = []

    def add(gid, check, value, ok):
        rows.append(dict(gate=gid, check=check, value=value, passed=bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {gid} {check}: {value}", flush=True)

    print(f'drift runner gate {dataset} {rd} vs {os.path.basename(ref_path)}', flush=True)
    data = fit_recday(dataset, rd, min_trials=ek.MIN_TRIALS, seed=seed, synthetic='planted')
    rek.write_atomic(os.path.join(adir, f'{rd}.pkl'), {'config': st, 'data': data})
    M = merge_dir(adir, expect=1)
    for kind in PLANTED:
        r = next(x for x in data['sets'] if x['set'] == 'region_full' and x['group'] == kind)
        g = ref['results'][kind]
        dk = max(float(np.max(np.abs(r['knn'][c].to_numpy() - g['knn'][c]))) for c in ('within_acc', 'across_acc', 'chance_acc'))
        add('eq', f'{kind}: kNN within / across / chance per session, max |runner - M1|', dk, dk == 0.0)
        add('eq', f'{kind}: chance MAE, |runner - M1|', abs(r['chance_mae'] - g['knn']['chance_mae']),
            r['chance_mae'] == g['knn']['chance_mae'])
        with np.errstate(invalid='ignore'):
            dd = float(np.nanmax(np.abs(np.nanmean(r['dist'], axis=0) - g['dist_lag'])))
        add('eq', f'{kind}: distance vs lag, max |runner - M1|', dd, dd == 0.0)
        a = M['acc'][(M['acc']['set'] == 'region_full') & (M['acc']['group'] == kind) & (M['acc']['event'] == 'reward')]
        dz = float(np.nanmax(np.abs(a.sort_values('lag_s')['z'].to_numpy() - g['acc_z'])))
        add('eq', f'{kind}: reward acceleration z, max |runner - M1|', dz, dz == 0.0)
    for _, v in p_verdicts(M, 'region_full').iterrows():
        add(v['gate'], f"{v['check']} (60 units)", round(float(v['value']), 3), v['passed'])
    for _, v in p_verdicts(M, 'region').iterrows():
        rows.append(dict(gate=v['gate'], check=f"{v['check']} (20 units, reported)", value=float(v['value']), passed=True))
        print(f"  [info] {v['gate']} {v['check']} (20 units): {float(v['value']):.3f} ({'holds' if v['passed'] else 'does not hold'})")
    neg = os.path.join(root, 'negative_stamp')
    rek.write_atomic(os.path.join(neg, f'{rd}.pkl'), {'config': st, 'data': data})
    rek.write_atomic(os.path.join(neg, 'zz_other.pkl'), {'config': {**st, 'n_match': N_MATCH + 1},
                                                          'data': {**data, 'recday': 'zz_other'}})
    try:
        merge_dir(neg, expect=0)
        add('neg', 'merge refuses a shard with one stamp field changed', 'merged', False)
    except ValueError as exc:
        add('neg', 'merge refuses a shard with one stamp field changed', str(exc)[:60], 'different stamp' in str(exc))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(root, 'runner_gate.csv'), index=False)
    print(f'-> {root}', flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=('lec', 'pfc'), default='pfc' if IS_PFC else 'lec')
    ap.add_argument('--recday')
    ap.add_argument('--min-trials', type=int, default=ek.MIN_TRIALS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--expect-recdays', type=int, default=rek.N_RECDAYS)
    ap.add_argument('--gate', action='store_true', help='runner gate on planted populations (exit 0 = all pass)')
    args = ap.parse_args(argv)
    if args.gate:
        out = gate(args.dataset, args.seed)
        return 0 if out['passed'].all() else 1
    arm = arm_name(args.dataset, args.min_trials)
    if args.merge:
        M = merge_dir(os.path.join(DRIFT_DIR, 'shards', arm), expect=args.expect_recdays)
        p = write_merged(M, arm)
        print(M['summary'].to_string(index=False))
        print(f"{len(M['summary'])} shards -> {p}")
        return 0
    if not args.recday:
        ap.error('--recday is required for a fit')
    path = os.path.join(DRIFT_DIR, 'shards', arm, f'{args.recday}.pkl')
    st = stamp(args.dataset, args.min_trials, args.seed)
    if rek.check_or_skip(path, st):
        print(f'{path} already holds this stamp; nothing to do')
        return 0
    d = fit_recday(args.dataset, args.recday, min_trials=args.min_trials, seed=args.seed)
    rek.write_atomic(path, {'config': st, 'data': d})
    if d['skipped']:
        print(f"{args.recday}: skipped ({d['skipped']}) -> {path}")
        return 0
    allset = next(r for r in d['sets'] if r['set'] == 'all')
    k = allset['knn'][['within_acc', 'across_acc', 'chance_acc']].mean()
    print(f"{args.recday}: {len(d['sessions'])} sessions, {len(d['units'])} units, {len(d['sets'])} unit sets, "
          f"all-unit kNN within {k['within_acc']:.3f} / across {k['across_acc']:.3f} / chance {k['chance_acc']:.3f}, "
          f"{d['elapsed_s']:.0f} s -> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
