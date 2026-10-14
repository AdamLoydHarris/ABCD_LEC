"""Diagnostic (2026-10-12): planted border x HD conjunctive cells in the 50 cm LEC box are called border in
0.55 of 20 (gate run 3752973, sigma 1 bin). Breaks the miss down by HD preference relative to the wall:
border score against the pooled threshold, split-half stability against the unit's own threshold, and the
time the mouse spends near the wall facing each way.

    sbatch ... sbatch_files/spatial.sbatch code/diag_box_conjunctive.py
"""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    import spatial_io as sio
    sio.set_dataset('lec')
    import spatial_tuning as st
    import spatial_tuning_synthetics as g
    cfg = copy.deepcopy(st.DEFAULT_CFG)
    cfg.update(n_shifts=200, n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    rng = np.random.default_rng(1)
    v = sio.load('lec', ['ah10_20250624_20250625'], 'v4')['ah10_20250624_20250625'][1]
    tr = g.trajectory(v)
    hd = tr['hd']
    near = (tr['xy'][:, 0] < 5.0) & np.isfinite(hd)
    sectors = np.arange(0, 360, 45)
    rel = np.degrees(hd[near]) % 360
    dwell = {s: float((np.abs((rel - s + 180) % 360 - 180) < 22.5).sum() * 0.025) for s in sectors}
    print(f'within 5 cm of the left wall: {near.sum() * 0.025:.0f} s; seconds per 45 deg HD sector '
          f'(0 = facing away from the wall, 180 = facing it): {dwell}', flush=True)
    mus = np.repeat(sectors, 5)
    for lam in (5.0, 10.0):
        R = np.array([g.r_border(tr, ('left',), lam, 1.0) * g.r_hd(tr, np.deg2rad(m), 3, 1.0) * 10 for m in mus])
        res, idx = g.run({4: tr}, {4: R}, cfg, rng)
        ss = res['sessions'][4]
        pool = {int(u): p for p, mem in ss['pools'].items() for u in np.asarray(mem, int)}
        thr = np.array([ss['thresholds'][pool[int(u)]]['border'] for u in idx])
        b = np.asarray(ss['real']['border'])[idx]
        stab, own = np.asarray(ss['real']['stab'])[idx], np.asarray(ss['own']['stab'])[idx]
        cb, ch = np.asarray(ss['classes']['border'])[idx], np.asarray(ss['classes']['hd'])[idx]
        print(f'lam {lam:.0f} cm: called border {cb.mean():.2f}, HD {ch.mean():.2f} of {len(idx)}', flush=True)
        for m in sectors:
            k = mus == m
            print(f'  HD pref {m:3d} deg (dwell {dwell[m]:5.0f} s): border median {np.nanmedian(b[k]):.2f} vs threshold '
                  f'{np.nanmedian(thr[k]):.2f}, above {np.mean(b[k] > thr[k]):.1f}; stable {np.mean(stab[k] > own[k]):.1f}; '
                  f'called border {cb[k].mean():.1f}, HD {ch[k].mean():.1f}', flush=True)


if __name__ == '__main__':
    main()
