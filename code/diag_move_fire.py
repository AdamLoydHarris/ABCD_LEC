"""Diagnostic (2026-10-12): after the object-move classes required significant firing at the corner, planted
generic object cells were found in 0.50 of 10 (gate run 3752973) against 0.9-1.0 for object-following cells.
Prints, per planted type, how often each 'fires there' test passes at p99 / p95 / p90 and the class rates that
follow, with the change tests kept at p99.

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
            STASH[f'rest_frac_{s}'] = float(rest.mean())
        sha, shb = sessions[s_a]['shifts'], sessions[s_b]['shifts']
        keys = [(0, 0)] + [(int(a), int(b)) for a, b in zip(sha, shb)]
        STASH['Pa'] = {L: np.array([st._location_stat(ca['counts'], des[s_a][L], ka) for ka, _ in keys]) for L in roles}
        STASH['Pb'] = {L: np.array([st._location_stat(cb['counts'], des[s_b][L], kb) for _, kb in keys]) for L in roles}
        STASH['lay'] = lay
        return out
    st.object_move = wrapped

    cfg = copy.deepcopy(st.DEFAULT_CFG)
    cfg.update(n_shifts=200, n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    cfg['rotation'].update(n_perm=200, n_boot=500, n_shift_object=100)
    rng = np.random.default_rng(7)
    rd, s_a, s_b = 'ah10_20250624_20250625', 1, 4
    vm = sio.load('lec', [rd], 'v4')[rd]
    ta, tb = g.trajectory(vm[s_a]), g.trajectory(vm[s_b])
    side = ta['side']
    mvc = st.arena_cfg(cfg, vm[s_a]['arena'])['move']
    oa = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=6.0) for o in vm[s_a]['objects']]
    ob = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=6.0) for o in vm[s_b]['objects']]
    lay = st.object_layout(oa, ob, side, mvc)
    V = lay['roles']['V']['xy']
    slider = next(l_ for l_, m_ in lay['movers'].items() if m_['frm'] == 'V')
    cen = np.array([side / 2, side / 2])

    def inward(p_, dist):
        u_ = cen - p_
        return p_ + dist * u_ / np.linalg.norm(u_)
    pos = {s_a: {o['label']: o['xy'] for o in oa}, s_b: {o['label']: o['xy'] for o in ob}}
    kinds = ['follow_slider', 'generic', 'trace', 'place_V', 'gain', 'anchor']
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
                elif kind == 'generic':
                    r_ = sum(g.r_place(tr, inward(q_, 9), 2.5, pk) for q_ in P_.values())
                elif kind == 'trace':
                    r_ = g.r_place(tr, inward(V, 9), 2.5, pk) if s_ == s_b else np.full(tr['T'], 0.2)
                elif kind == 'place_V':
                    r_ = g.r_place(tr, inward(V, 9), 2.5, pk)
                elif kind == 'gain':
                    r_ = g.r_place(tr, cen + rng.normal(0, 3, 2), 6, pk) * (2.0 if s_ == s_b else 1.0)
                else:
                    r_ = g.r_place(tr, cen + rng.normal(0, 3, 2), 6, pk)
                rates[s_].append(r_)
            lab.append(kind)
    res, idx = g.run({s_a: ta, s_b: tb}, {s_: np.array(r_) for s_, r_ in rates.items()}, cfg, rng)
    lab = np.array(lab + ['noise'] * (STASH['Pa']['V'].shape[1] - len(lab)))
    order = np.r_[idx, np.setdiff1d(np.arange(len(lab)), idx)]
    lab_u = np.empty(len(lab), object)
    lab_u[order] = lab
    Pa, Pb = STASH['Pa'], STASH['Pb']
    print(f"roles {[(k, np.round(v['xy'], 1).tolist(), round(float(v['r_in']), 1)) for k, v in STASH['lay']['roles'].items()]}; "
          f"r_out {STASH['lay']['r_out']}; rest fraction a {STASH[f'rest_frac_{s_a}']:.2f}, b {STASH[f'rest_frac_{s_b}']:.2f}", flush=True)

    def over(x, q):
        return x[0] > np.nanpercentile(x[1:], q, axis=0)
    D = {L: Pb[L] - Pa[L] for L in Pa}
    up, down = {L: over(D[L], 99) for L in D}, {L: over(-D[L], 99) for L in D}
    frm, to = lay['movers'][slider]['frm'], lay['movers'][slider]['to']
    for kind in kinds + ['noise']:
        k = lab_u == kind
        line = [f'{kind:14s} n {k.sum():3d}: P_a(V) median {np.nanmedian(Pa["V"][0][k]):.2f} vs shift p99 '
                f'{np.nanmedian(np.nanpercentile(Pa["V"][1:], 99, axis=0)[k]):.2f}, p95 {np.nanmedian(np.nanpercentile(Pa["V"][1:], 95, axis=0)[k]):.2f}']
        for q in (99, 95, 90):
            sa, sb = {L: over(Pa[L], q) for L in Pa}, {L: over(Pb[L], q) for L in Pb}
            gen = sa['V'] & down['V'] & up['N'] & sb['N']
            fol = sa[frm] & down[frm] & up[to] & sb[to]
            line.append(f'q_fire {q}: sig_a(V) {sa["V"][k].mean():.2f}, sig_b(N) {sb["N"][k].mean():.2f}, '
                        f'generic {gen[k].mean():.2f}, follow {fol[k].mean():.2f}')
        line.append(f'change only: generic {(down["V"] & up["N"])[k].mean():.2f}, follow {(down[frm] & up[to])[k].mean():.2f}')
        print(' | '.join(line), flush=True)


if __name__ == '__main__':
    main()
