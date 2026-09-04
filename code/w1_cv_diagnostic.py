"""Why is held-out R2 negative? Separate the nuisance causes from the biology.

The first smoke test gave median held-out R2 of -0.004 to -0.007 against in-sample +0.010 to
+0.013, on three ah08 recdays. Three candidate explanations, and they call for different
responses, so they have to be separated before 125 cluster jobs are launched on this design:

  (1) NUISANCE -- session rate drift. The design carries ONE global intercept, so a neuron
      whose mean rate drifts between sessions has every held-out prediction offset, while
      `r2_cv`'s TSS baselines against the held-out session's OWN mean. The model is scored
      against a baseline it was never allowed to match. Fix: mean-centre within session.

  (2) BIOLOGY -- remapping. Leave-one-session-out folds are DIFFERENT TASKS. Cells remap in
      task space between tasks (that is what W3 measures), so a model trained on tasks 1-5
      genuinely cannot predict task 6. This is not a defect; it means LOSO answers "does
      this neuron encode V the SAME WAY across tasks", not "does it encode V".

  (3) NO SIGNAL in these particular recdays -- all three are ah08, the lowest-firing mouse
      (1.07 Hz median), so this may not generalise to the cohort.

The discriminating test for (1) vs (2)+(3) is whether centring rescues R2. The
discriminating test for (2) vs (3) is WITHIN-session CV: if a model cross-validates fine
inside a session but fails across sessions, that gap is remapping, not absence of tuning.

Run:  python code/w1_cv_diagnostic.py [--recdays N]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = os.path.join(REPO, 'data', 'processed_data', 'w1_cv_diagnostic.pkl')

SECTION_REGRESSORS = [
    'place', 'task_state', 'head_direction', 'goal_progress',
    'speed', 'acceleration',
    'time_from_reward', 'time_to_reward',
    'distance_from_reward', 'distance_to_reward',
]
JOINT_DROP = [('time_any', ['time_from_reward', 'time_to_reward']),
              ('gp_any', ['goal_progress'])]


def summarise(tag, cvres):
    rows = []
    for rd, r in cvres.items():
        r2 = np.asarray(r['r2_cv'], dtype=float)
        rows.append((rd, np.nanmedian(r2), np.nanmean(r2 > 0)))
    med = np.nanmean([r[1] for r in rows])
    frac = np.nanmean([r[2] for r in rows])
    print(f'  {tag:34s} median r2_cv {med:+.5f}   frac>0 {frac:.2f}')
    return {'rows': rows, 'median_r2': med, 'frac_pos': frac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recdays', type=int, default=4)
    ap.add_argument('--mice', nargs='*', default=None,
                    help='restrict to these mice (default: one recday from each mouse)')
    args = ap.parse_args()

    import glm_analysis_v2 as glm

    data_dic = glm.load_data_dic(validate=True, apply_exclusions=True, verbose=True)

    # One recday per mouse by default -- the first smoke test was all ah08, which is the
    # lowest-firing mouse and cannot speak for the cohort.
    all_rd = sorted(data_dic)
    if args.mice:
        recdays = [r for r in all_rd if r.split('_')[0] in args.mice][:args.recdays]
    else:
        seen, recdays = set(), []
        for r in all_rd:
            m = r.split('_')[0]
            if m not in seen:
                seen.add(m)
                recdays.append(r)
    print(f'\nrecdays ({len(recdays)}, one per mouse): {recdays}\n')

    common = dict(
        num_permutations=1,                       # unused: cv_only skips the in-sample loop
        regressors_to_include=SECTION_REGRESSORS,
        joint_drop_groups=JOINT_DROP,
        compute_cpd=True,
        parameterization='reference_coded',
        cross_validate=True, cv_n_perm=0, cv_only=True,
    )

    out = {}
    print('=' * 78)
    print('DIAGNOSTIC 1 — does a free per-session offset rescue held-out R2?')
    print('=' * 78)

    t0 = time.time()
    res_raw = glm.run_glm_analysis(recdays, data_dic,
                                   cv_center_within_sessions=False, **common)[-1]
    print(f'  [cv_only took {time.time() - t0:.1f}s for {len(recdays)} recdays]')
    out['loso_raw'] = summarise('LOSO, global intercept only', res_raw)

    res_ctr = glm.run_glm_analysis(recdays, data_dic,
                                   cv_center_within_sessions=True, **common)[-1]
    out['loso_centered'] = summarise('LOSO, centred within session', res_ctr)

    print('\n  If centring rescues R2, the negative values were session rate drift')
    print('  (a nuisance), not absence of tuning.')

    print('\n' + '=' * 78)
    print('CPD by regressor: LOSO raw vs LOSO centred')
    print('=' * 78)
    print(f"{'regressor':24s} {'LOSO raw':>10s} {'LOSO centred':>13s} {'frac>0 ctr':>11s}")
    groups = sorted(g for g in next(iter(res_raw.values()))['cpd_cv'] if not g.startswith('__'))
    tbl = {}
    for g in groups:
        a = np.nanmean([np.nanmedian(r['cpd_cv'][g]) for r in res_raw.values()])
        b = np.nanmean([np.nanmedian(r['cpd_cv'][g]) for r in res_ctr.values()])
        f = np.nanmean([np.nanmean(np.asarray(r['cpd_cv'][g]) > 0) for r in res_ctr.values()])
        tbl[g] = (a, b, f)
        print(f'{g:24s} {a:10.5f} {b:13.5f} {f:11.2f}')
    out['cpd_table'] = tbl

    print('\n' + '=' * 78)
    print('Per-mouse held-out R2 (centred) — is this an ah08 story?')
    print('=' * 78)
    for rd, r in res_ctr.items():
        r2 = np.asarray(r['r2_cv'], dtype=float)
        print(f'  {rd:32s} n={len(r2):4d}  median r2_cv {np.nanmedian(r2):+.5f}  '
              f'frac>0 {np.nanmean(r2 > 0):.2f}')

    # ---- DIAGNOSTIC 2: the decisive one -------------------------------------
    print('\n' + '=' * 78)
    print('DIAGNOSTIC 2 — remapping, or no signal? LOSO vs WITHIN-session folds')
    print('=' * 78)
    print('  LOSO folds are different TASKS, so a cell that remaps cannot be predicted')
    print('  across them. Within-session folds cut inside each session, so every fold')
    print('  spans every task. A regressor that scores well within-session and badly')
    print('  across-session is REMAPPING, not absent.\n')

    res_win = glm.run_glm_analysis(recdays, data_dic,
                                   cv_center_within_sessions=True,
                                   cv_within_session_folds=True, **common)[-1]
    out['within_session'] = summarise('within-session folds', res_win)

    res_z = glm.run_glm_analysis(recdays, data_dic,
                                 cv_zscore_within_sessions=True, **common)[-1]
    out['loso_zscored'] = summarise('LOSO, z-scored within session', res_z)

    print('\n' + '=' * 78)
    print('CPD by regressor: the remapping gap')
    print('=' * 78)
    print(f"{'regressor':24s} {'LOSO':>9s} {'within-sess':>12s} {'gap':>9s}  reading")
    for g in groups:
        a = np.nanmean([np.nanmedian(r['cpd_cv'][g]) for r in res_ctr.values()])
        b = np.nanmean([np.nanmedian(r['cpd_cv'][g]) for r in res_win.values()])
        gap = b - a
        reading = ('encodes it, but REMAPS' if b > 0.0002 and gap > 0.0002 else
                   'generalises across tasks' if b > 0.0002 else
                   'no reliable signal')
        print(f'{g:24s} {a:9.5f} {b:12.5f} {gap:+9.5f}  {reading}')

    print('\n' + '=' * 78)
    print('Per-mouse, within-session folds')
    print('=' * 78)
    for rd in res_win:
        r2 = np.asarray(res_win[rd]['r2_cv'], dtype=float)
        pl = np.asarray(res_win[rd]['cpd_cv']['place'], dtype=float)
        print(f'  {rd:32s} r2_cv {np.nanmedian(r2):+.5f} (frac>0 {np.nanmean(r2>0):.2f})'
              f'   place cpd {np.nanmedian(pl):+.5f}')

    with open(OUT, 'wb') as fh:
        pickle.dump({'raw': res_raw, 'centered': res_ctr, 'within': res_win,
                     'zscored': res_z, 'summary': out}, fh)
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
