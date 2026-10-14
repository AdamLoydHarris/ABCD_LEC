"""Diagnostic (2026-10-12): planted object-vector cells were found in 0.73 of 15 on ly06 s5->s9 (full LEC gate run
3755102; ah10 1.00). Prints per planted vector cell the translated-map statistic against its shift p99 and
whether the field sat outside the band in both sessions (offset_ok), the two terms of the class.

    sbatch ... sbatch_files/spatial.sbatch code/diag_move_vector_ly06.py
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
    cfg['rotation'].update(n_perm=200, n_boot=500, n_shift_object=100)
    rng = np.random.default_rng(5)
    for rd, s_a, s_b in (('ly06_20250624_20250625', 5, 9), ('ah10_20250624_20250625', 1, 4)):
        vm = sio.load('lec', [rd], 'v4')[rd]
        ta, tb = g.trajectory(vm[s_a]), g.trajectory(vm[s_b])
        side = ta['side']
        mvc = st.arena_cfg(cfg, vm[s_a]['arena'])['move']
        oa = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=6.0) for o in vm[s_a]['objects']]
        ob = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=6.0) for o in vm[s_b]['objects']]
        lay = st.object_layout(oa, ob, side, mvc)
        ro = lay['roles']
        V, S_ = ro['V']['xy'], ro['S']['xy']
        slider = next(l_ for l_, m_ in lay['movers'].items() if m_['frm'] == 'V')
        wall_vs = min(('left', 'right', 'bottom', 'top'), key=lambda w: g.wall_d(np.array([(V + S_) / 2]), side, w)[0])
        normal = {'left': (1.0, 0.0), 'right': (-1.0, 0.0), 'bottom': (0.0, 1.0), 'top': (0.0, -1.0)}[wall_vs]
        pos = {s_a: {o['label']: o['xy'] for o in oa}, s_b: {o['label']: o['xy'] for o in ob}}
        m = 15
        rates = {s_a: [], s_b: []}
        for _ in range(m):
            pk = rng.uniform(12, 20)
            for s_, tr in ((s_a, ta), (s_b, tb)):
                rates[s_].append(g.r_place(tr, pos[s_][slider] + 17.0 * np.asarray(normal), 3.0, pk))
        res, idx = g.run({s_a: ta, s_b: tb}, {s_: np.array(r_) for s_, r_ in rates.items()}, cfg, rng)
        mvr = res['rotations'][(s_a, s_b)]['object_move']
        v = mvr['vector'][slider]
        cls = np.asarray(mvr['classes'][f'vector[{slider}]'])
        fa = pos[s_a][slider] + 17.0 * np.asarray(normal)
        fb = pos[s_b][slider] + 17.0 * np.asarray(normal)
        print(f"{rd} s{s_a}->s{s_b}: slider {slider} along the {wall_vs} wall, field at {np.round(fa, 1).tolist()} -> "
              f"{np.round(fb, 1).tolist()}, shift {v['shift_bins']} bins; detected {cls[idx].mean():.2f}", flush=True)
        for u in idx:
            print(f"  unit {u:3d}: stat {v['stat'][u]:6.3f} vs p99 {v['p99'][u]:6.3f} {'ok ' if v['stat'][u] > v['p99'][u] else 'MISS'} | "
                  f"field outside the band in both {bool(v['offset_ok'][u])} | eligible {bool(res['sessions'][s_a]['eligible'][u] & res['sessions'][s_b]['eligible'][u])}",
                  flush=True)


if __name__ == '__main__':
    main()
