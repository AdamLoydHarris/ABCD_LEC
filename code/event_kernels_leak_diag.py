"""Where the A+reward joint's held-out gain for planted tau-2 s speed units comes from (G5 leak).

Rebuilds the full synthetic (seed 0) on one recday and splits the joint's leave-one-session-out
SSE gain by row type: tracked, untracked before the first A, after the last trial, other gaps.

    sbatch --job-name=ek_diag_pfc --time=0-01:00 sbatch_files/event_kernels.sbatch \
        code/event_kernels_leak_diag.py pfc me11_05122021_06122021
"""
import sys, numpy as np, pandas as pd
ds, rd = sys.argv[1], sys.argv[2]
tree = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/' + ('mFC_data/code' if ds == 'pfc' else 'code')
sys.path.insert(0, tree)
import event_kernels as ek, event_kernels_synthetics as syn, glm_cv
rec = ek.load_recday(rd, ds); ek.add_day_clock(rec)
sessions = ek.analysis_sessions(rec)
synth, truth = syn.make_synthetic(rec, sessions, syn.unit_specs(ds, quick=False), seed=0)
u = truth[(truth.kind == 'speed_history') & (truth.tau == 2)].unit.to_numpy()
pd.set_option('display.width', 200)
for kin in ('full', 'strict'):
    D = ek.build_design(synth, kin=kin)
    sids = D['session_ids']
    X = ek.center_within_sessions(D['X'], sids)
    Y = glm_cv.center_within_session(D['FR'][u], sids)
    cols = D['groups']['ev_A'] + D['groups']['ev_reward']
    Xr = np.delete(X, cols, axis=1)
    cat = np.empty(len(sids), dtype=object)
    for i, (s, b) in enumerate(zip(synth, D['blocks'])):
        m = sids == i
        t = b['t0_s']
        k_s = s['first_A_bin'] * ek.BIN_S
        last = s['trial_times'].max() * ek.BIN_S if len(s['trial_times']) else np.inf
        miss = D['kin_missing'][m]
        c = np.where(~miss, 'tracked', np.where(t < k_s, 'untracked pre-A', np.where(t >= last, 'untracked post', 'untracked gap')))
        cat[m] = c
    gain = np.zeros((len(u), len(sids)))
    for f in np.unique(sids):
        te, tr = sids == f, sids != f
        bf = glm_cv._pinv(X[tr]) @ Y[:, tr].T
        br = glm_cv._pinv(Xr[tr]) @ Y[:, tr].T
        ef = (Y[:, te].T - X[te] @ bf) ** 2
        er = (Y[:, te].T - Xr[te] @ br) ** 2
        gain[:, te] = (er - ef).T
    rows = []
    for c in ('tracked', 'untracked pre-A', 'untracked post', 'untracked gap'):
        m = cat == c
        rows.append(dict(rows=c, n=int(m.sum()), **{f'u{i}': float(gain[j, m].sum()) for j, i in enumerate(u)}))
    df = pd.DataFrame(rows)
    print(f'== {ds} {rd} kin={kin}: summed held-out SSE gain of the A+reward joint, per unit (positive = joint helps)')
    print(df.round(1).to_string(index=False))
