"""Synthetic controls for `population_drift`, on one real recday's session lengths and rewards.

    python code/population_drift_synthetics.py --dataset lec [--quick]

| # | planted population | must hold |
|---|---|---|
| P1 | Poisson noise | kNN within and across <= chance + 0.05; flat distance vs lag; |z| < 3 at rewards |
| P2 | locked clock: the same smooth function of time since entry in every session | across >= 3x chance |
| P3 | OU drift, tau 3 min, a fresh path per session | within >= 3x chance, across <= 1.5x chance, distance grows with lag |
| P4 | boundary ensembles: a different random 20% of units bursts for 1 s at each reward | z > 3 at lag 0 or +1; P1-P3 populations |z| < 3 |

M3 part 2 (2026-10-14): `planted_part2` adds the populations of `run_population_drift.py --gate-part2`.
`planted` is unchanged, so P1-P4 reproduce. The shift is the transition into the 0-1 s bin.

| # | planted population | must hold |
|---|---|---|
| P5 | fixed boundary ensemble: the same 50% of units burst at every reward, lognormal amplitudes | shift preferred >= 2x chance (1/6); PV correlation of up-modulated units >= 0.3; window speed peaks at the burst's onset or offset |
| P6 | `boundary` above: a different 20% at each reward | PV correlation of up-modulated units <= 0.1 |
| P7 | noise | shift preferred within 1/6 +/- 0.05 |
| P8 | OU drift, no boundaries | reported: loop-number decoding error vs the shuffle |
| P9 | noise | loop-number error >= 0.9x the shuffle |
| P10 | leg-locked bumps in the 6 s before each reward plus independent OU drift | leg vs session offset >= 60 deg, both stabilities <= 30 deg and below the offset |
| P11 | one within-leg ramp and one OU drift on a shared unit axis | leg vs session offset <= 30 deg |
| P12 | OU drift in 25% of units | drift growth after top-25%-Fano removal <= 0.5x that after random removal |
| P13 | OU drift in every unit (`drift`) | that ratio >= 0.6 |
| P14 | noise | median log Fano within 0.1 of 0 |
| P15 | noise | P10's criterion fails |

Revised 2026-10-14 after the full part-2 gates (3766749 LEC, 3766752 PFC):
- P5: a 1 s burst has equal onset and offset jumps, so the window speed peaked at the offset in LEC
  (transition 3) and the onset in PFC (2); either is the planted shift.
- P10/P15: noise met "offset above both stabilities" (LEC 84.6 vs 68.3 / 68.0 deg; PFC 81.2 vs 50.4 /
  63.2): random axes in 60 dimensions are near-orthogonal and unstable. Stable axes are now required.
- P13: ranking by realised Fano selects each session's largest drift excursions even when every
  unit drifts (ratio 0.77 PFC, 0.80 LEC); the test is the contrast with P12 (0.00).

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import population_drift as pdft                                        # noqa: E402

IS_PFC = 'mFC_data' in HERE
DEFAULT_RECDAY = {'lec': 'ah10_20250616_20250617', 'pfc': 'me11_05122021_06122021'}
OUT_DIR = os.path.join(ek.REPO, 'data', 'event_kernels', 'synthetics')


def _ou(T, tau_s, rng):
    a = np.exp(-pdft.BIN_S / tau_s)
    e = rng.standard_normal(T) * np.sqrt(1 - a ** 2)
    x = np.empty(T)
    x[0] = rng.standard_normal()
    for t in range(1, T):
        x[t] = a * x[t - 1] + e[t]
    return x


def planted(kind, sessions, n_units, rng):
    """List of (units, T) counts per session for one population."""
    T_max = max(s['counts'].shape[1] for s in sessions)
    t = np.arange(T_max) * pdft.BIN_S
    if kind == 'locked':
        curves = np.zeros((n_units, T_max))
        for u in range(n_units):
            for _ in range(3):
                period = rng.uniform(120, 1200)
                curves[u] += np.cos(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        curves /= curves.std(axis=1, keepdims=True)
    out = []
    for s in sessions:
        T = s['counts'].shape[1]
        if kind == 'locked':
            f = 0.8 * curves[:, :T]
        elif kind == 'drift':
            f = 0.8 * np.stack([_ou(T, 180.0, rng) for _ in range(n_units)])
        elif kind == 'boundary':
            f = np.zeros((n_units, T))
            for e in ek.event_bins(s, 'reward'):
                sub = rng.random(n_units) < 0.2
                f[sub, e:min(e + 40, T)] = 1.5
        else:
            f = np.zeros((n_units, T))
        lam = 2.0 * pdft.BIN_S * np.exp(np.clip(f, -5, 5))
        out.append(rng.poisson(lam).astype(np.uint16))
    return out


def _time_to_next_reward(sess, T):
    """(T,) seconds until the next reward arrival at each 25 ms bin; NaN after the last."""
    ev = np.sort(ek.event_bins(sess, 'reward'))
    t = np.arange(T)
    j = np.searchsorted(ev, t, side='left')
    out = np.full(T, np.nan)
    ok = j < len(ev)
    out[ok] = (ev[j[ok]] - t[ok]) * pdft.BIN_S
    return out


PART2_KINDS = ('noise', 'boundary', 'fixed_boundary', 'drift', 'leg_drift', 'shared_axis', 'subset_drift')


def planted_part2(kind, sessions, n_units, rng):
    """M3 part 2 populations: list of (units, T) counts per session. Kinds `planted` makes are
    delegated to it with the same stream."""
    if kind in ('noise', 'locked', 'drift', 'boundary'):
        return planted(kind, sessions, n_units, rng)
    if kind == 'fixed_boundary':
        amp = np.where(rng.random(n_units) < 0.5, rng.lognormal(np.log(1.5), 0.4, n_units), 0.0)
    elif kind == 'leg_drift':
        centre, width = rng.uniform(0.5, 5.5, n_units), rng.uniform(0.5, 1.5, n_units)
    elif kind == 'shared_axis':
        w = rng.lognormal(0.0, 0.5, n_units) * rng.choice([-1.0, 1.0], n_units)
    elif kind == 'subset_drift':
        members = np.zeros(n_units, bool)
        members[rng.choice(n_units, n_units // 4, replace=False)] = True
    else:
        raise ValueError(kind)
    out = []
    for s in sessions:
        T = s['counts'].shape[1]
        if kind == 'fixed_boundary':
            f = np.zeros((n_units, T))
            for e in ek.event_bins(s, 'reward'):
                f[:, e:min(e + 40, T)] = amp[:, None]
        elif kind == 'leg_drift':
            ttr = _time_to_next_reward(s, T)
            with np.errstate(invalid='ignore'):
                leg = np.exp(-0.5 * ((ttr[None, :] - centre[:, None]) / width[:, None]) ** 2)
                leg = np.where(ttr[None, :] <= 6.0, leg, 0.0)
            f = np.nan_to_num(leg) + 0.8 * np.stack([_ou(T, 180.0, rng) for _ in range(n_units)])
        elif kind == 'shared_axis':
            ttr = _time_to_next_reward(s, T)
            with np.errstate(invalid='ignore'):
                g = np.where(ttr <= 6.0, 1.0 - ttr / 6.0, 0.0)
            f = w[:, None] * (np.nan_to_num(g) + 0.8 * _ou(T, 180.0, rng))[None, :]
        else:
            f = np.zeros((n_units, T))
            for u in np.flatnonzero(members):
                f[u] = 0.8 * _ou(T, 180.0, rng)
        lam = 2.0 * pdft.BIN_S * np.exp(np.clip(f, -5, 5))
        out.append(rng.poisson(lam).astype(np.uint16))
    return out


def run(dataset, recday, quick, seed=0):
    rng = np.random.default_rng(seed)
    rec = ek.load_recday(recday, dataset)
    sessions = ek.analysis_sessions(rec)
    n = 30 if quick else 60
    res, rows = {}, []
    for kind in ('noise', 'locked', 'drift', 'boundary'):
        counts = planted(kind, sessions, n, rng)
        knn = pdft.knn_epochs(counts, seed=seed)
        dl = np.nanmean([pdft.distance_vs_lag(c) for c in counts], axis=0)
        acc = [pdft.event_acceleration(c, ek.event_bins(s, 'reward')) for c, s in zip(counts, sessions)]
        z = np.nanmean([a['z'] for a in acc if a['n']], axis=0)
        lags = acc[0]['lags']
        res[kind] = dict(knn=knn, dist_lag=dl, acc_z=z)
        rows.append(dict(kind=kind, within=float(knn['within_acc'].mean()), across=float(knn['across_acc'].mean()),
                         chance=float(knn['chance_acc'].mean()), d_lag1=dl[1], d_lag10=dl[10],
                         z_lag0=float(z[lags == 0][0]), z_lag1=float(z[lags == 1][0]),
                         z_max_abs=float(np.nanmax(np.abs(z)))))
    tab = pd.DataFrame(rows).set_index('kind')
    g = []

    def add(gid, desc, val, ok):
        g.append(dict(gate=gid, check=desc, value=val, passed=bool(ok)))

    ch = tab['chance']
    add('P1', 'noise within', tab.loc['noise', 'within'], tab.loc['noise', 'within'] <= ch['noise'] + 0.05)
    add('P1', 'noise across', tab.loc['noise', 'across'], tab.loc['noise', 'across'] <= ch['noise'] + 0.05)
    add('P1', 'noise distance lag10 - lag1', tab.loc['noise', 'd_lag10'] - tab.loc['noise', 'd_lag1'],
        abs(tab.loc['noise', 'd_lag10'] - tab.loc['noise', 'd_lag1']) < 0.01)
    add('P2', 'locked across / chance', tab.loc['locked', 'across'] / ch['locked'],
        tab.loc['locked', 'across'] >= 3 * ch['locked'])
    add('P3', 'drift within / chance', tab.loc['drift', 'within'] / ch['drift'],
        tab.loc['drift', 'within'] >= 3 * ch['drift'])
    add('P3', 'drift across / chance', tab.loc['drift', 'across'] / ch['drift'],
        tab.loc['drift', 'across'] <= 1.5 * ch['drift'])
    add('P3', 'drift distance grows with lag', tab.loc['drift', 'd_lag10'] - tab.loc['drift', 'd_lag1'],
        tab.loc['drift', 'd_lag10'] > tab.loc['drift', 'd_lag1'])
    add('P4', 'boundary z at lag 0 or 1', max(tab.loc['boundary', 'z_lag0'], tab.loc['boundary', 'z_lag1']),
        max(tab.loc['boundary', 'z_lag0'], tab.loc['boundary', 'z_lag1']) > 3)
    for kind in ('noise', 'locked', 'drift'):
        add('P4', f'{kind} |z| at rewards', tab.loc[kind, 'z_max_abs'], tab.loc[kind, 'z_max_abs'] < 3)
    return dict(table=tab, gates=pd.DataFrame(g), results=res, recday=recday, dataset=dataset)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('lec', 'pfc'), default='pfc' if IS_PFC else 'lec')
    ap.add_argument('--recday', default=None)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args(argv)
    rd = args.recday or DEFAULT_RECDAY[args.dataset]
    out = run(args.dataset, rd, args.quick)
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, f"population_drift_synthetics_{args.dataset}_{rd}{'_quick' if args.quick else ''}")
    k = 0
    while os.path.exists(p := base + (f'.v{k}' if k else '') + '.pkl'):
        k += 1
    with open(p, 'wb') as fh:
        pickle.dump(out, fh)
    print(out['table'].round(3).to_string())
    print(out['gates'].to_string(index=False))
    print('->', p)
    return 0 if out['gates']['passed'].all() else 1


if __name__ == '__main__':
    sys.exit(main())
