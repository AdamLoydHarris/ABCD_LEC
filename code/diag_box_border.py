"""Diagnostic (2026-10-12): why planted two-wall border cells (decay 5 cm) miss the border class in the 50 cm
LEC box while one-wall cells pass (gate run 3752932). Prints per planted cell the Solstad terms and the
pooled threshold, for one- and two-wall fields at decay 3-6 cm, with and without the session's objects.

    sbatch ... sbatch_files/spatial.sbatch code/diag_box_border.py
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
    print('objects', [(o['label'], np.round(o['xy_cm'], 1).tolist()) for o in v['objects']])
    for sigma in (2.0, 1.5, 1.0):
        c_ = copy.deepcopy(cfg)
        c_['sigma_bins'] = sigma
        for name, maker in (('left lam 5', lambda: g.r_border(tr, ('left',), 5, rng.uniform(5, 15))),
                            ('bottom+right lam 5', lambda: g.r_border(tr, ('bottom', 'right'), 5, rng.uniform(5, 15))),
                            ('near-wall place', lambda: g.r_place(tr, np.array([5.0, 25.0]) + rng.normal(0, 2, 2), rng.uniform(3, 6), rng.uniform(5, 15))),
                            ('centre place', lambda: g.r_place(tr, np.array([25.0, 25.0]) + rng.normal(0, 2, 2), rng.uniform(3, 6), rng.uniform(5, 15)))):
            R = np.array([maker() for _ in range(10)])
            res, idx = g.run({1: tr}, {1: R}, c_, rng, n_noise=30)
            ss = res['sessions'][1]
            pb = np.asarray(ss['pools']['broad'], int)
            nb = np.asarray(ss['null']['border'])[:, pb]
            thr = ss['thresholds']['broad']['border']
            b = np.asarray(ss['real']['border'])
            called = np.asarray(ss['classes']['border'])
            noise = np.setdiff1d(np.arange(len(b)), idx)
            print(f"sigma {sigma} {name:20s}: planted border median {np.nanmedian(b[idx]):.3f}, called {called[idx].mean():.2f} | "
                  f"noise units finite real {np.isfinite(b[noise]).mean():.2f}, called {called[noise].mean():.2f} | "
                  f"pooled p99 {thr:.3f}, finite shuffles {np.isfinite(nb).mean():.3f}", flush=True)


if __name__ == '__main__':
    main()
