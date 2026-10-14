"""Diagnostic 5 (2026-10-12): generic object cells are found in 0.37-0.40 of planted cells with the engine's
statistic (jobs 3753163, 3753986); a sector-matched change was worse. Compares, on every A17L planted type, the
log-ratio regulariser (0.1 vs 0.5 Hz) and the test level (every term at p99 vs p95).

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

    def sector_ratios(counts, des, k, c0=0.1):
        """(N, n_sectors) log band / matched-rest ratio per sector, NaN where the sector is thin."""
        N = counts.shape[0]
        if des is None or not des['ok'].any():
            return np.full((N, 8), np.nan)
        c = np.roll(counts, int(k), axis=1) if k else counts
        band = np.asarray(des['B'].T.dot(c.T)).T / np.maximum(des['n_band'], 1)[None] / 0.025
        with np.errstate(invalid='ignore', divide='ignore'):
            rr = np.asarray(des['R'].T.dot(c.T)).T / des['n_rest'][None] / 0.025
        matched = np.nan_to_num(rr) @ des['W'].T
        r = np.log((band + c0) / (matched + c0))
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
        for c0 in (0.1, 0.5):
            STASH[('Ra', c0)] = {L: np.array([sector_ratios(ca['counts'], des[s_a][L], ka, c0) for ka, _ in keys]) for L in roles}
            STASH[('Rb', c0)] = {L: np.array([sector_ratios(cb['counts'], des[s_b][L], kb, c0) for _, kb in keys]) for L in roles}
        STASH['lay'] = lay
        STASH['classes'] = out.get('classes')
        return out
    st.object_move = wrapped

    cfg = copy.deepcopy(st.DEFAULT_CFG)
    cfg.update(n_shifts=200, n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    cfg['rotation'].update(n_perm=200, n_boot=500, n_shift_object=100)
    rng = np.random.default_rng(13)
    rd, s_a, s_b = 'ah10_20250624_20250625', 1, 4
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
    kinds = ['follow_slider', 'follow_diag', 'trace', 'generic', 'place_V', 'place_S', 'border_VS', 'border_VN',
             'hd_VN', 'hd_V', 'slow', 'gain', 'anchor']
    m = 20
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
    Ntot = STASH[('Ra', 0.1)]['V'].shape[1]
    lab_u = np.full(Ntot, 'noise', dtype=object)
    lab_u[idx] = lab

    def over(x, q):
        """(n+1, N) -> (N,) real row above each unit's own q-th percentile of the shift rows."""
        return x[0] > np.nanpercentile(x[1:], q, axis=0)
    m_ = lay['movers']
    fs, fd = m_[slider], m_[diag]
    for c0 in (0.1, 0.5):
        Ra, Rb = STASH[('Ra', c0)], STASH[('Rb', c0)]
        with np.errstate(invalid='ignore'):
            P_a = {L: np.nanmax(Ra[L], axis=2) for L in Ra}
            P_b = {L: np.nanmax(Rb[L], axis=2) for L in Rb}
        for q in (99, 95):
            up = {L: over(P_b[L] - P_a[L], q) for L in Ra}
            dn = {L: over(P_a[L] - P_b[L], q) for L in Ra}
            sa, sb = {L: over(P_a[L], q) for L in P_a}, {L: over(P_b[L], q) for L in P_b}
            cl = dict(follow_slider=sa[fs['frm']] & dn[fs['frm']] & up[fs['to']] & sb[fs['to']],
                      follow_diag=sa[fd['frm']] & dn[fd['frm']] & up[fd['to']] & sb[fd['to']],
                      trace=up['V'] & sb['V'], generic=sa['V'] & dn['V'] & up['N'] & sb['N'])
            print(f'regulariser {c0} Hz, every term at p{q}', flush=True)
            for kind in kinds + ['noise']:
                k = lab_u == kind
                row = {c_: float(v[k].mean()) for c_, v in cl.items()}
                anyp = float(np.any([v[k] for v in cl.values()], axis=0).mean())
                print(f'  {kind:14s} n {k.sum():3d}: ' + ', '.join(f'{c_} {v:.2f}' for c_, v in row.items()) + f' | any {anyp:.2f}', flush=True)
            g_ = lab_u == 'generic'
            print(f"  generic terms: fires a(V) {sa['V'][g_].mean():.2f}, drop V {dn['V'][g_].mean():.2f}, rise N {up['N'][g_].mean():.2f}, "
                  f"fires b(N) {sb['N'][g_].mean():.2f}", flush=True)

if __name__ == '__main__':
    main()
