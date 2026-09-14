#!/usr/bin/env python3
"""Cross-check the pipeline's 360-bin normalised arrays against the ones in the LEC pickle.

`Neuron_<rd>_<s>.npy` is written by Figure2's UNBLOCK-F2-06 cell with the notebook's OWN
`raw_to_norm` (the handoff insists we let the pipeline build it rather than export
`data_dic[...]['Neurons_norm']`, so the arrays are consistent with cell 46's normalisation by
construction). This script compares the two afterwards: a large discrepancy is a real
finding about the two normalisations, not something to patch.

Run with the repo python (maze_ephys, numpy 2.x) -- it unpickles data_dic_lec.pkl.
Compares, per exported session: shapes; per-neuron Pearson r between the trial-averaged
360-bin curves on the common trials; the trial-count difference (raw_to_norm drops trailing
incomplete states). Writes lec_replication_run/logs/lec_neurons_norm_compare.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
MFC = os.path.join(ROOT, 'mFC_data')
MIRROR = os.path.join(MFC, 'lec_replication_run', 'data', 'Intermediate_objects')
LOGDIR = os.path.join(MFC, 'lec_replication_run', 'logs')
REPO_CODE = os.path.join(ROOT, 'code')


def main():
    sys.path.insert(0, REPO_CODE)
    from glm_analysis_v3 import load_data_dic
    data_dic = load_data_dic()

    manifest = json.load(open(os.path.join(LOGDIR, 'lec_export_manifest.json')))
    sessions = [(m['recday'], m['session']) for m in manifest['exported']]

    rows, missing = [], []
    all_r = []
    for rd, s in sessions:
        f = os.path.join(MIRROR, f'Neuron_{rd}_{s}.npy')
        if not os.path.exists(f):
            missing.append(f'{rd}_{s}')
            continue
        pipe = np.load(f, allow_pickle=True)
        sess = data_dic[rd][s]
        ours = sess.get('Neurons_norm')
        if ours is None:
            rows.append(dict(recday=rd, session=s, status='no Neurons_norm in data_dic',
                             pipe_shape=list(np.shape(pipe))))
            continue
        ours = np.asarray(ours, dtype=float)
        if pipe.dtype == object:
            rows.append(dict(recday=rd, session=s, status='pipeline array is ragged (object)',
                             pipe_shape=list(np.shape(pipe)), ours_shape=list(ours.shape)))
            continue
        pipe = np.asarray(pipe, dtype=float)
        if pipe.ndim != 3 or ours.ndim != 3 or pipe.shape[0] != ours.shape[0] or pipe.shape[2] != ours.shape[2]:
            rows.append(dict(recday=rd, session=s, status='shape mismatch',
                             pipe_shape=list(pipe.shape), ours_shape=list(ours.shape)))
            continue
        m = min(pipe.shape[1], ours.shape[1])
        a = np.nanmean(pipe[:, :m, :], axis=1)      # (n_neurons, 360) trial-averaged curves
        b = np.nanmean(ours[:, :m, :], axis=1)
        r = np.full(a.shape[0], np.nan)
        for i in range(a.shape[0]):
            x, y = a[i], b[i]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                r[i] = np.corrcoef(x[ok], y[ok])[0, 1]
        all_r.extend(r[np.isfinite(r)].tolist())
        rows.append(dict(recday=rd, session=s, status='compared',
                         n_neurons=int(a.shape[0]), trials_pipe=int(pipe.shape[1]),
                         trials_ours=int(ours.shape[1]), trials_common=int(m),
                         median_r=float(np.nanmedian(r)), min_r=float(np.nanmin(r)) if np.isfinite(r).any() else None,
                         frac_r_gt_0_99=float(np.mean(r[np.isfinite(r)] > 0.99)) if np.isfinite(r).any() else None,
                         mean_abs_diff=float(np.nanmean(np.abs(a - b))),
                         pipe_nan_frac=float(np.isnan(pipe[:, :m]).mean()),
                         ours_nan_frac=float(np.isnan(ours[:, :m]).mean())))

    compared = [r for r in rows if r['status'] == 'compared']
    print(f'sessions exported {len(sessions)}, Neuron_ files missing {len(missing)}, compared {len(compared)}')
    for st in sorted({r['status'] for r in rows}):
        print(f'  status {st!r}: {sum(r["status"] == st for r in rows)}')
    if all_r:
        q = np.percentile(all_r, [0, 5, 25, 50, 75, 95, 100])
        print(f'per-neuron r of trial-averaged curves, {len(all_r)} neurons: '
              f'min {q[0]:.4f}  p5 {q[1]:.4f}  p25 {q[2]:.4f}  median {q[3]:.4f}  p75 {q[4]:.4f}  '
              f'p95 {q[5]:.4f}  max {q[6]:.4f};  frac > 0.99: {np.mean(np.asarray(all_r) > 0.99):.4f}')
        dt = [r['trials_ours'] - r['trials_pipe'] for r in compared]
        print(f'trials (data_dic - pipeline): min {min(dt)}, median {np.median(dt)}, max {max(dt)}')
    if not compared:
        print('NOT RUN: nothing compared (run after Figure2 has written the Neuron_ files)')
    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, 'lec_neurons_norm_compare.json'), 'w') as f:
        json.dump(dict(rows=rows, missing=missing,
                       summary=dict(n_compared=len(compared), n_neurons=len(all_r),
                                    r_percentiles=(np.percentile(all_r, [0, 5, 50, 95, 100]).tolist() if all_r else None))),
                  f, indent=2)
    print(f'wrote {LOGDIR}/lec_neurons_norm_compare.json')
    return 0 if compared else 2


if __name__ == '__main__':
    sys.exit(main())
