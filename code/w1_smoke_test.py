"""W1 smoke test: time the cross-validated refit on 3 recdays before committing to 25.

The plan says not to schedule the full refit from a guess. This measures:

  * in-sample fit time (the existing path), for reference
  * LOSO CV time with no permutations
  * LOSO CV time with permutations, to price the CV null
  * Poisson CV time on a small neuron subset, to price the robustness check
  * agreement between in-sample CPD and held-out CPD, which is the scientific point

Run:  python code/w1_smoke_test.py [--recdays N] [--section NAME]
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

OUT = os.path.join(REPO, 'data', 'processed_data', 'w1_smoke_test.pkl')

#: The backbone section: place + goal_progress + HD + task_state in one design.
SECTION_REGRESSORS = [
    'place', 'task_state', 'head_direction', 'goal_progress',
    'speed', 'acceleration',
    'time_from_reward', 'time_to_reward',
    'distance_from_reward', 'distance_to_reward',
]
JOINT_DROP = [('time_any', ['time_from_reward', 'time_to_reward']),
              ('gp_any', ['goal_progress'])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recdays', type=int, default=3)
    ap.add_argument('--cv-perms', type=int, default=20)
    ap.add_argument('--poisson-neurons', type=int, default=5)
    ap.add_argument('--permutations', type=int, default=100)
    args = ap.parse_args()

    import glm_analysis_v2 as glm

    print('=' * 78)
    print('W1 SMOKE TEST — cross-validated GLM refit')
    print('=' * 78)

    data_dic = glm.load_data_dic(validate=True, apply_exclusions=True, verbose=True)
    recdays = sorted(data_dic)[:args.recdays]
    print(f'\nrecdays: {recdays}\n')

    results = {}

    # ---- 1. in-sample only (the existing path), for reference -------------
    print('--- [1/3] in-sample path (existing behaviour) ---')
    t0 = time.time()
    out_ins = glm.run_glm_analysis(
        recdays, data_dic,
        num_permutations=args.permutations,
        regressors_to_include=SECTION_REGRESSORS,
        joint_drop_groups=JOINT_DROP,
        compute_cpd=True,
        parameterization='reference_coded',
    )
    t_ins = time.time() - t0
    print(f'\n  in-sample: {t_ins:.1f}s for {len(recdays)} recdays '
          f'({t_ins / len(recdays):.1f}s each)')
    results['in_sample'] = {'elapsed_s': t_ins, 'cpd': out_ins[2]}

    # ---- 2. + LOSO CV, no CV permutations --------------------------------
    print('\n--- [2/3] in-sample + LOSO CV (no CV permutations) ---')
    t0 = time.time()
    out_cv = glm.run_glm_analysis(
        recdays, data_dic,
        num_permutations=args.permutations,
        regressors_to_include=SECTION_REGRESSORS,
        joint_drop_groups=JOINT_DROP,
        compute_cpd=True,
        parameterization='reference_coded',
        cross_validate=True, cv_n_perm=0,
    )
    t_cv0 = time.time() - t0
    cvres = out_cv[3]
    print(f'\n  in-sample + CV: {t_cv0:.1f}s  => CV overhead '
          f'{t_cv0 - t_ins:+.1f}s ({(t_cv0 - t_ins) / len(recdays):.1f}s per recday)')
    results['cv_noperm'] = {'elapsed_s': t_cv0, 'cv': cvres, 'cpd': out_cv[2]}

    # ---- 3. + CV permutations, and Poisson on a subset -------------------
    print(f'\n--- [3/3] LOSO CV with {args.cv_perms} permutations '
          f'+ Poisson on {args.poisson_neurons} neurons ---')
    t0 = time.time()
    out_cvp = glm.run_glm_analysis(
        recdays[:1], data_dic,
        num_permutations=args.permutations,
        regressors_to_include=SECTION_REGRESSORS,
        joint_drop_groups=JOINT_DROP,
        compute_cpd=True,
        parameterization='reference_coded',
        cross_validate=True, cv_n_perm=args.cv_perms,
        cv_poisson=True, cv_poisson_neurons=np.arange(args.poisson_neurons),
    )
    t_cvp = time.time() - t0
    results['cv_perm_poisson'] = {'elapsed_s': t_cvp, 'cv': out_cvp[3]}
    print(f'\n  1 recday with {args.cv_perms} CV perms + Poisson: {t_cvp:.1f}s')

    # ---- report ----------------------------------------------------------
    print('\n' + '=' * 78)
    print('TIMING SUMMARY (per recday)')
    print('=' * 78)
    per_ins = t_ins / len(recdays)
    per_cv = (t_cv0 - t_ins) / len(recdays)
    print(f'  in-sample fit + {args.permutations} perms : {per_ins:7.1f} s')
    print(f'  LOSO CV, no CV perms                     : {per_cv:7.1f} s')
    cvp = results['cv_perm_poisson']['cv'][recdays[0]]
    print(f'  LOSO CV, {args.cv_perms:3d} CV perms (measured)      : '
          f"{cvp['elapsed_s']:7.1f} s")
    if 'poisson' in cvp:
        print(f"  (of which Poisson on {args.poisson_neurons} neurons is included)")

    n_rd, n_sec = 25, 5
    print(f'\n  EXTRAPOLATION to {n_rd} recdays x {n_sec} sections:')
    for label, per in [('in-sample only', per_ins),
                       ('in-sample + CV (no CV perms)', per_ins + per_cv),
                       (f'in-sample + CV ({args.cv_perms} CV perms)',
                        per_ins + cvp['elapsed_s'])]:
        tot = per * n_rd * n_sec
        print(f'    {label:38s} {tot / 3600:6.2f} h  ({per * n_rd / 60:5.1f} min/section)')

    # ---- in-sample vs held-out CPD, the scientific point ------------------
    print('\n' + '=' * 78)
    print('IN-SAMPLE vs HELD-OUT CPD')
    print('=' * 78)
    rows = []
    for rd in recdays:
        if rd not in cvres:
            continue
        cpd_ins = out_cv[2][rd]
        cv = cvres[rd]
        for g in sorted(cv['cpd_cv']):
            if g.startswith('__'):
                continue
            ins = np.array([cpd_ins[n].get(g, np.nan)
                            for n in sorted(cpd_ins)], dtype=float)
            hel = np.asarray(cv['cpd_cv'][g], dtype=float)
            m = min(len(ins), len(hel))
            rows.append((rd, g, np.nanmedian(ins[:m]), np.nanmedian(hel[:m]),
                         float(np.nanmean(hel[:m] > 0))))
    print(f"{'regressor':24s} {'in-sample':>10s} {'held-out':>10s} {'frac>0':>8s}")
    import collections
    agg = collections.defaultdict(list)
    for _rd, g, i, h, f in rows:
        agg[g].append((i, h, f))
    for g, v in sorted(agg.items()):
        v = np.array(v)
        print(f'{g:24s} {np.nanmean(v[:,0]):10.5f} {np.nanmean(v[:,1]):10.5f} '
              f'{np.nanmean(v[:,2]):8.2f}')
    print('\n  A held-out CPD can be negative — dropping a group can IMPROVE held-out fit')
    print('  when the group was only fitting noise. That is signal, and is not clipped.')

    with open(OUT, 'wb') as fh:
        pickle.dump(results, fh)
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
