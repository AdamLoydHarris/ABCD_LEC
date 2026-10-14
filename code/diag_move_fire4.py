"""Diagnostic 4 (2026-10-12): generic object cells pass the rise-at-N test in 0.57 of 30 (job 3753163); the change
statistic differences each session's best sector, which can be two different sectors. Compares, on every A17L
planted type, the engine's statistic with a sector-matched change (best sector of b - a at one corner) and the
fires-there test at p99 / p95.

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
    cfg.update(n_shifts=200, n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    cfg['rotation'].update(n_perm=200, n_boot=500, n_shift_object=100)
    rng = np.random.default_rng(11)
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
    Ntot = STASH['Ra']['V'].shape[1]
    lab_u = np.full(Ntot, 'noise', dtype=object)
    lab_u[idx] = lab
    Ra, Rb = STASH['Ra'], STASH['Rb']

    def over(x, q):
        """(n+1, N) -> (N,) real row above each unit's own q-th percentile of the shift rows."""
        return x[0] > np.nanpercentile(x[1:], q, axis=0)
    with np.errstate(invalid='ignore'):
        P_a = {L: np.nanmax(Ra[L], axis=2) for L in Ra}
        P_b = {L: np.nanmax(Rb[L], axis=2) for L in Rb}
        dmat = {L: Rb[L] - Ra[L] for L in Ra}
        up_m = {L: over(np.nanmax(dmat[L], axis=2), 99) for L in Ra}
        dn_m = {L: over(np.nanmax(-dmat[L], axis=2), 99) for L in Ra}
    up_0 = {L: over(P_b[L] - P_a[L], 99) for L in Ra}
    dn_0 = {L: over(P_a[L] - P_b[L], 99) for L in Ra}
    m_ = lay['movers']
    fs, fd = m_[slider], m_[diag]
    variants = {}
    for name, up, dn, q in (('now: max-sector difference, fires p99', up_0, dn_0, 99),
                            ('sector-matched difference, fires p99', up_m, dn_m, 99),
                            ('sector-matched difference, fires p95', up_m, dn_m, 95)):
        sa, sb = {L: over(P_a[L], q) for L in P_a}, {L: over(P_b[L], q) for L in P_b}
        variants[name] = dict(follow_slider=sa[fs['frm']] & dn[fs['frm']] & up[fs['to']] & sb[fs['to']],
                              follow_diag=sa[fd['frm']] & dn[fd['frm']] & up[fd['to']] & sb[fd['to']],
                              trace=up['V'] & sb['V'], generic=sa['V'] & dn['V'] & up['N'] & sb['N'])
    # the engine's own classes for the first variant must match (checks the replication)
    eng = STASH['classes']
    same = all(np.array_equal(np.asarray(eng[k]) & True, variants['now: max-sector difference, fires p99'][c_] & np.asarray(eng[k]))
               for k, c_ in ((f'follow[{slider}]', 'follow_slider'), ('generic', 'generic'), ('trace', 'trace')))
    print(f'replication of the engine classes (eligible units): {same}', flush=True)
    for name, cl in variants.items():
        print(name, flush=True)
        for kind in kinds + ['noise']:
            k = lab_u == kind
            row = {c_: float(v[k].mean()) for c_, v in cl.items()}
            anyp = float(np.any([v[k] for v in cl.values()], axis=0).mean())
            print(f'  {kind:14s} n {k.sum():3d}: ' + ', '.join(f'{c_} {v:.2f}' for c_, v in row.items()) + f' | any {anyp:.2f}', flush=True)


if __name__ == '__main__':
    main()
