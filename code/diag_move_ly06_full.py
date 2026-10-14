"""Diagnostic (2026-10-12): the full LEC gate (run 3755102) found ly06 s5->s9 object-vector cells in 0.73 of 15, but
alone they were found in 15 of 15 (job 3755356). Replicates the full gate's A17L population on ly06 with 1000
shifts and prints the vector terms per planted vector cell, and the engine classes of every kind.

    sbatch ... sbatch_files/spatial.sbatch code/diag_move_fire.py
"""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STASH = {}


def main():
    import spatial_io as sio
    sio.set_dataset('lec')
    import spatial_tuning as st
    import spatial_tuning_synthetics as g
    orig = st.object_move

    def sector_ratios(counts, des, k):
        """(N, n_sectors) log band / matched-rest ratio per sector, NaN where the sector is thin."""
        N = counts.shape[0]
        if des is None or not des['ok'].any():
            return np.full((N, 8), np.nan)
        c = np.roll(counts, int(k), axis=1) if k else counts
        band = np.asarray(des['B'].T.dot(c.T)).T / np.maximum(des['n_band'], 1)[None] / 0.025
        with np.errstate(invalid='ignore', divide='ignore'):
            rr = np.asarray(des['R'].T.dot(c.T)).T / des['n_rest'][None] / 0.025
        matched = np.nan_to_num(rr) @ des['W'].T
        r = np.log((band + 0.1) / (matched + 0.1))
        r[:, ~des['ok']] = np.nan
        return r

    def wrapped(ctxs, sessions, s_a, s_b, rot=None):
        out = orig(ctxs, sessions, s_a, s_b, rot)
        ca, cb = ctxs[s_a], ctxs[s_b]
        mv = ca['cfg']['move']
        lay = st.object_layout(ca['objs'], cb['objs'], ca['side'], mv)
        roles, r_out = lay['roles'], lay['r_out']
        des = {}
        for s, ctx in ((s_a, ca), (s_b, cb)):
            code = st._strata(ctx, mv)
            rest = ctx['moving'].copy()
            for L in roles.values():
                rest &= np.hypot(ctx['xy'][:, 0] - L['xy'][0], ctx['xy'][:, 1] - L['xy'][1]) >= r_out
            des[s] = {L: st._location_design(ctx, roles[L]['xy'], roles[L]['r_in'], r_out, rest, code, mv) for L in roles}
        sha, shb = sessions[s_a]['shifts'], sessions[s_b]['shifts']
        keys = [(0, 0)] + [(int(a), int(b)) for a, b in zip(sha, shb)]
        STASH['Ra'] = {L: np.array([sector_ratios(ca['counts'], des[s_a][L], ka) for ka, _ in keys]) for L in roles}
        STASH['Rb'] = {L: np.array([sector_ratios(cb['counts'], des[s_b][L], kb) for _, kb in keys]) for L in roles}
        STASH['lay'] = lay
        STASH['classes'] = out.get('classes')
        return out
    st.object_move = wrapped

    cfg = copy.deepcopy(st.DEFAULT_CFG)
    cfg.update(n_shifts=1000, n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    cfg['rotation'].update(n_perm=1000, n_boot=500, n_shift_object=200)
    rng = np.random.default_rng(17)
    rd, s_a, s_b = 'ly06_20250624_20250625', 5, 9
    vm = sio.load('lec', [rd], 'v4')[rd]
    ta, tb = g.trajectory(vm[s_a]), g.trajectory(vm[s_b])
    side = ta['side']
    mvc = st.arena_cfg(cfg, vm[s_a]['arena'])['move']
    oa = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=6.0) for o in vm[s_a]['objects']]
    ob = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=6.0) for o in vm[s_b]['objects']]
    lay = st.object_layout(oa, ob, side, mvc)
    ro = lay['roles']
    V, S_, N_ = (ro[k]['xy'] for k in 'VSN')
    mov = lay['movers']
    slider = next(l_ for l_, m_ in mov.items() if m_['frm'] == 'V')
    diag = next(l_ for l_, m_ in mov.items() if m_['frm'] == 'S')
    cen = np.array([side / 2, side / 2])

    def inward(p_, dist):
        u_ = cen - p_
        return p_ + dist * u_ / np.linalg.norm(u_)
    pos = {s_a: {o['label']: o['xy'] for o in oa}, s_b: {o['label']: o['xy'] for o in ob}}
    wall_vs = min(('left', 'right', 'bottom', 'top'), key=lambda w: g.wall_d(np.array([(V + S_) / 2]), side, w)[0])
    wall_vn = min(('left', 'right', 'bottom', 'top'), key=lambda w: g.wall_d(np.array([(V + N_) / 2]), side, w)[0])
    vn_dir = np.arctan2(*(((V + N_) / 2 - cen)[::-1]))
    v_dir = np.arctan2(*((V - cen)[::-1]))
    normal = {'left': (1.0, 0.0), 'right': (-1.0, 0.0), 'bottom': (0.0, 1.0), 'top': (0.0, -1.0)}[wall_vs]
    kinds = ['follow_slider', 'follow_diag', 'trace', 'generic', 'vector', 'place_V', 'place_S', 'border_VS', 'border_VN',
             'hd_VN', 'hd_V', 'slow', 'gain', 'anchor']
    m = 15
    rates = {s_a: [], s_b: []}
    lab = []
    for kind in kinds:
        for _ in range(m):
            pk = rng.uniform(12, 20)
            for s_, tr in ((s_a, ta), (s_b, tb)):
                P_ = pos[s_]
                if kind == 'follow_slider':
                    r_ = g.r_place(tr, inward(P_[slider], 9), 2.5, pk)
                elif kind == 'follow_diag':
                    r_ = g.r_place(tr, inward(P_[diag], 9), 2.5, pk)
                elif kind == 'trace':
                    r_ = g.r_place(tr, inward(V, 9), 2.5, pk) if s_ == s_b else np.full(tr['T'], 0.2)
                elif kind == 'generic':
                    r_ = sum(g.r_place(tr, inward(q_, 9), 2.5, pk) - 0.2 for q_ in P_.values()) + 0.2
                elif kind == 'vector':
                    r_ = g.r_place(tr, P_[slider] + 17.0 * np.asarray(normal), 3.0, pk)
                elif kind == 'place_V':
                    r_ = g.r_place(tr, inward(V, 9), 2.5, pk)
                elif kind == 'place_S':
                    r_ = g.r_place(tr, inward(S_, 9), 2.5, pk)
                elif kind == 'border_VS':
                    r_ = g.r_border(tr, (wall_vs,), 4, pk)
                elif kind == 'border_VN':
                    r_ = g.r_border(tr, (wall_vn,), 4, pk)
                elif kind == 'hd_VN':
                    r_ = g.r_hd(tr, vn_dir, 3, pk)
                elif kind == 'hd_V':
                    r_ = g.r_hd(tr, v_dir, 3, pk)
                elif kind == 'slow':
                    r_ = 1.0 + pk * np.exp(-np.nan_to_num(tr['speed']) / 5.0)
                elif kind == 'gain':
                    r_ = g.r_place(tr, cen + rng.normal(0, 3, 2), 6, pk) * (2.0 if s_ == s_b else 1.0)
                else:
                    r_ = g.r_place(tr, cen + rng.normal(0, 3, 2), 6, pk)
                rates[s_].append(r_)
            lab.append(kind)
    res, idx = g.run({s_a: ta, s_b: tb}, {s_: np.array(r_) for s_, r_ in rates.items()}, cfg, rng)
    lab = np.array(lab)
    mvr = res['rotations'][(s_a, s_b)]['object_move']
    v = mvr['vector'][slider]
    elig = res['sessions'][s_a]['eligible'] & res['sessions'][s_b]['eligible']
    print(f"replicating the full gate on {rd} s{s_a}->s{s_b}: shift bins {v['shift_bins']}, {mvr['n_shifts']} shifts", flush=True)
    for u in idx[lab == 'vector']:
        print(f"  vector unit {u:3d}: stat {v['stat'][u]:6.3f} vs p99 {v['p99'][u]:6.3f} {'ok ' if v['stat'][u] > v['p99'][u] else 'MISS'} | "
              f"outside the band in both {bool(v['offset_ok'][u])} | eligible {bool(elig[u])} | rate a/b "
              f"{res['sessions'][s_a]['rate_hz'][u]:.2f}/{res['sessions'][s_b]['rate_hz'][u]:.2f} Hz", flush=True)
    for k_, c_ in mvr['classes'].items():
        c_ = np.asarray(c_)[idx]
        print(f'  class {k_:28s}: ' + ', '.join(f'{kd} {c_[lab == kd].mean():.2f}' for kd in kinds if c_[lab == kd].any()), flush=True)

if __name__ == '__main__':
    main()
