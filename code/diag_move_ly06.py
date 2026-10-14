"""Diagnostic (2026-10-12): in the full LEC gate (run 3755102) the ly06 s5->s9 day-pair found planted trace cells in
0 of 15 and place-at-V cells in 0 of 15 (ah10: 1.00), and object-vector cells in 0.73. Prints, per session and
corner, the band design behind P: moving samples in the band, sectors that pass min_sector, rest samples,
and the occupancy of each corner band.

    sbatch ... sbatch_files/spatial.sbatch code/diag_move_ly06.py
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
    for rd, s_a, s_b in (('ah10_20250624_20250625', 1, 4), ('ly06_20250624_20250625', 5, 9), ('ly06_20250624_20250625', 1, 4)):
        vm = sio.load('lec', [rd], 'v4')[rd]
        cfg = st.arena_cfg(copy.deepcopy(st.DEFAULT_CFG), vm[s_a]['arena'])
        mv = cfg['move']
        trs = {s: g.trajectory(vm[s]) for s in (s_a, s_b)}
        side = trs[s_a]['side']
        oa = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=3.0) for o in vm[s_a]['objects']]
        ob = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=3.0) for o in vm[s_b]['objects']]
        lay = st.object_layout(oa, ob, side, mv)
        roles, r_out = lay['roles'], lay['r_out']
        print(f"{rd} s{s_a}->s{s_b}: roles {[(k, np.round(v['xy'], 1).tolist()) for k, v in roles.items()]}, movers "
              f"{[(l_, m_['frm'] + '->' + m_['to']) for l_, m_ in lay['movers'].items()]}", flush=True)
        for s in (s_a, s_b):
            tr = trs[s]
            spd = np.nan_to_num(tr['speed'])
            moving = spd > cfg['min_speed']
            xy = tr['xy']
            rest = moving.copy()
            for L in roles.values():
                rest &= np.hypot(xy[:, 0] - L['xy'][0], xy[:, 1] - L['xy'][1]) >= r_out
            line = [f'  s{s}: moving {moving.mean():.2f} of {len(xy)} samples, rest {rest.sum()}']
            for L, R_ in roles.items():
                rel = xy - R_['xy']
                d = np.hypot(rel[:, 0], rel[:, 1])
                band = moving & (d >= R_['r_in']) & (d < r_out)
                sec = np.floor(np.mod(np.arctan2(rel[band, 1], rel[band, 0]), 2 * np.pi) / (2 * np.pi) * mv['n_sectors']).astype(int)
                nb = np.bincount(np.clip(sec, 0, mv['n_sectors'] - 1), minlength=mv['n_sectors'])
                line.append(f"{L}: band {band.sum()} (min {mv['min_band']}), sectors >= {mv['min_sector']}: {int((nb >= mv['min_sector']).sum())}/8 "
                            f"{nb.tolist()}, all samples within {r_out:.0f} cm {(d < r_out).sum()}")
            print('\n    '.join(line), flush=True)


if __name__ == '__main__':
    main()
