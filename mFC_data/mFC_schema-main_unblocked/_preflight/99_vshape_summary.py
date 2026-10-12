#!/usr/bin/env python3
"""Summarise the Figure 7 V panel from the UNBLOCK-F7-05 bin report.

Cell 86 saves one curve: the unweighted np.mean over sleep stages of the per-stage binned
means. That estimator has two failure modes this script measures rather than argues about --
it propagates NaN, so one empty bin in one stage deletes that bin from the figure, and it
weights a stage holding 2 pairs the same as one holding 2000. His curve is reported first
and never replaced; the two alternatives are reported beside it.

    python 99_vshape_summary.py                 # read the report, print the tables
    python 99_vshape_summary.py --selftest      # synthetic accept/reject control, no data
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
RUN = os.path.join(ROOT, 'replication_run')
REPORT = os.path.join(RUN, 'Output_folder', 'ephys', 'vshape_bin_n.json')
LOGDIR = os.path.join(RUN, 'logs')

STAGE_ORDER = ['preX', 'post0', 'post1', 'post2', 'post3', 'post4', 'post5', 'preX2']


def estimators(means, counts):
    """means/counts: (n_bins, n_stages). Returns his curve and the two alternatives."""
    his = np.mean(means, axis=1)                       # cell 86, verbatim
    nan_tol = np.nanmean(np.where(np.isnan(means), np.nan, means), axis=1)
    w = np.where(np.isnan(means), 0.0, counts)
    num = np.nansum(np.where(np.isnan(means), 0.0, means) * w, axis=1)
    den = w.sum(axis=1)
    weighted = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    return dict(his_unweighted=his, nan_tolerant=nan_tol, n_weighted=weighted)


def v_score(curve):
    """Ends minus trough, in units of the curve. Positive means V-shaped.

    A V wants the two extreme bins high and the middle low; the dashed line cell 86 draws at
    3.5 is the vertex it expects, so the trough is taken over the two bins either side of it.
    """
    c = np.asarray(curve, float)
    if np.isnan(c[0]) or np.isnan(c[-1]):
        return np.nan, 'ends missing'
    mid = np.nanmin(c[len(c) // 2 - 1:len(c) // 2 + 1])
    if np.isnan(mid):
        return np.nan, 'trough missing'
    return float((c[0] + c[-1]) / 2 - mid), 'ok'


def load(path):
    rep = json.load(open(path))
    stages = [s for s in STAGE_ORDER if s in rep] + \
             [s for s in rep if s not in STAGE_ORDER and not s.startswith('_')]
    means = np.array([[np.nan if v is None else v for v in rep[s]['mean']] for s in stages]).T
    counts = np.array([rep[s]['n_per_bin'] for s in stages], float).T
    return rep, stages, means, counts


def report(path=REPORT, label='PFC'):
    if not os.path.exists(path):
        print(f'{label}: NOT RUN -- no report at {path}')
        return None
    rep, stages, means, counts = load(path)
    nb = means.shape[0]
    print(f'\n=== {label} === {path}')
    print(f'  stages {len(stages)}   bins {nb}   '
          f'co-anchored pairs per stage: {[int(rep[s]["n_total"]) for s in stages]}')
    print('\n  pairs per angle bin')
    print('  ' + 'stage'.ljust(8) + ''.join(f'{i:>8d}' for i in range(nb)) + '   recdays')
    for j, s in enumerate(stages):
        row = ''.join(f'{int(counts[i, j]):>8d}' for i in range(nb))
        print(f'  {s:8s}{row}   {rep[s]["n_recdays"]:>7d}')
    tot = counts.sum(axis=1).astype(int)
    print('  ' + 'TOTAL'.ljust(8) + ''.join(f'{v:>8d}' for v in tot))

    est = estimators(means, counts)
    dropped = [int(i) for i in np.where(np.isnan(est['his_unweighted']))[0]]
    print(f'\n  bins NaN in cell 86\'s saved curve: {dropped if dropped else "none"}'
          f'   (reported by the notebook: {rep.get("_aggregate", {}).get("bins_dropped_by_nan")})')
    print('\n  estimator                 ' + ''.join(f'{i:>9d}' for i in range(nb)) + '   V score')
    for k, c in est.items():
        s, why = v_score(c)
        cells = ''.join(('     nan' if np.isnan(v) else f'{v:9.4f}') for v in c)
        print(f'  {k:24s}{cells}   {"nan ("+why+")" if np.isnan(s) else f"{s:+.4f}"}')
    print('  (V score = mean of the two end bins minus the trough at the vertex; positive = V)')

    out = dict(path=path, stages=stages,
               n_per_bin={s: [int(v) for v in counts[:, j]] for j, s in enumerate(stages)},
               bins_nan_in_saved_curve=dropped,
               estimators={k: [None if np.isnan(v) else float(v) for v in c]
                           for k, c in est.items()},
               v_scores={k: (None if np.isnan(v_score(c)[0]) else v_score(c)[0])
                         for k, c in est.items()})
    return out


def selftest():
    """Accept a planted V, reject a planted flat curve, through the same estimators."""
    rng = np.random.default_rng(0)
    nb, ns = 8, 8
    ok = True

    # a V in every stage, but one near-empty stage with two missing bins
    true_v = np.array([1.0, .6, .3, .15, .15, .3, .6, 1.0])
    means = np.tile(true_v[:, None], (1, ns)) + rng.normal(0, .01, (nb, ns))
    counts = np.full((nb, ns), 300.0)
    means[1:3, -1] = np.nan
    counts[1:3, -1] = 0
    counts[:, -1] = np.where(np.isnan(means[:, -1]), 0, 2)
    est = estimators(means, counts)
    # the failure is bins DELETED from the saved curve, not a bad V score: bins 1-2 are not
    # read by v_score, which is why the LEC v4 figure looks broken while its V score survives
    lost = np.where(np.isnan(est['his_unweighted']))[0]
    print(f'  planted V, one 2-pair stage with bins 1-2 empty:')
    print(f'    his_unweighted drops bins {list(lost)}   <- expected [1, 2]')
    print(f'    nan_tolerant drops bins   '
          f'{list(np.where(np.isnan(est["nan_tolerant"]))[0])}   <- expected []')
    print(f'    n_weighted drops bins     '
          f'{list(np.where(np.isnan(est["n_weighted"]))[0])}   <- expected []')
    ok &= list(lost) == [1, 2]
    ok &= not np.isnan(est['nan_tolerant']).any()
    ok &= not np.isnan(est['n_weighted']).any()
    b = v_score(est['nan_tolerant'])[0]
    c = v_score(est['n_weighted'])[0]
    print(f'    V score: nan_tolerant {b:+.4f}, n_weighted {c:+.4f}'
          f'   <- expected ~{true_v[0]-true_v[3]:+.2f} for both')
    ok &= abs(b - 0.85) < 0.05 and abs(c - 0.85) < 0.05

    # REJECTION: flat truth must not produce a V under any estimator
    flat = np.full((nb, ns), 0.5) + rng.normal(0, .01, (nb, ns))
    estf = estimators(flat, np.full((nb, ns), 300.0))
    scores = {k: v_score(v)[0] for k, v in estf.items()}
    print(f'  planted FLAT curve: V scores ' +
          ', '.join(f'{k} {v:+.4f}' for k, v in scores.items()) + '   <- expected ~0')
    ok &= all(abs(v) < 0.05 for v in scores.values())

    # REJECTION: the weighted estimator must follow the many-pair stages, not the 2-pair one
    means2 = np.tile(true_v[:, None], (1, ns))
    means2[:, -1] = true_v[::-1] * 3          # one rogue stage, inverted and 3x
    counts2 = np.full((nb, ns), 300.0); counts2[:, -1] = 2
    e2 = estimators(means2, counts2)
    du, dw = v_score(e2['his_unweighted'])[0], v_score(e2['n_weighted'])[0]
    print(f'  rogue 2-pair stage inverted and x3: unweighted {du:+.4f}, n_weighted {dw:+.4f}'
          f'   <- weighting must stay near {true_v[0]-true_v[3]:+.2f}, unweighted must not')
    ok &= abs(dw - 0.85) < 0.05                       # weighting ignores the 2-pair stage
    ok &= abs(du - 0.85) > abs(dw - 0.85) + 0.1       # and the unweighted one is pulled off it

    print('\n  SELFTEST', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--report', default=REPORT)
    ap.add_argument('--label', default='PFC')
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    out = report(args.report, args.label)
    if out is None:
        return 1
    os.makedirs(LOGDIR, exist_ok=True)
    p = os.path.join(LOGDIR, 'vshape_summary.json')
    json.dump(out, open(p, 'w'), indent=2)
    print(f'\nwrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
