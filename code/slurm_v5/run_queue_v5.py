"""Sequential V5 run queue (one process, so the box is not oversubscribed):

  1. PFC  ElasticNet  future   reproduction config (MIN_TRIALS=1, test-session pref phase)
  2. PFC  Poisson     past     reproduction config + his nanmean(prediction) > 0 gate  (ED Fig 8d)
  3. LEC  Poisson     past     science config (MIN_TRIALS=5, training-session pref phase)
  4. LEC  Poisson     future   science config
  5. PFC  Poisson     future   reproduction config + gate (the prospective mirror of 2)
  6. PFC  ElasticNet  past     reproduction config, RECDAY-Z-SCORED target
  7. PFC  ElasticNet  future   reproduction config, RECDAY-Z-SCORED target
  8. LEC  ElasticNet  past     science config, RECDAY-Z-SCORED target
  9. LEC  ElasticNet  future   science config, RECDAY-Z-SCORED target
 10. LEC  ElasticNet  past     reproduction config, Z-SCORED  (`repro_` prefix)
 11. LEC  ElasticNet  future   reproduction config, Z-SCORED  (`repro_` prefix)
 12. LEC  Poisson     past     reproduction config + gate  (`repro_`) -- matches PFC spec 2
 13. LEC  Poisson     future   reproduction config + gate  (`repro_`) -- matches PFC spec 5
 14. PFC  Poisson     past     spec 2 with a LASSO penalty (glum, l1_ratio=1), alpha=0.01
 15. PFC  Poisson     past     lasso, alpha=0.003
 16. PFC  Poisson     past     lasso, alpha=0.001
 17. PFC  Poisson     past     elastic net (l1_ratio=0.5), alpha=0.003
     (14-17 are explicit-only and need glum; `alpha<a>_` prefix + `_l1` / `_en0.5` tag)

A Poisson run stores BOTH readouts (linear Xb, his code; exp(Xb+b), the paper's LNP) from one fit.

6-9 write `*_zscore_v5_*` directories and are NOT the reproduction: they divide each neuron's fit
target by its sd over the whole recday, because the reference's fixed alpha=0.01 on raw counts
zeroes 59% of neurons and does so as a function of firing rate. Poisson cannot take a z-scored
target (needs y >= 0), so the z-scored runs are ElasticNet only.
"""
import os, sys, time, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')
ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 else None      # e.g. "1,2"
T0 = time.time()


def hrs():
    return f'{(time.time() - T0) / 3600:.2f} h'


def dedup(data_dic, mouse_recdays, min_trials, v5):
    out = {}
    for mr in mouse_recdays:
        valid, tasks = [], []
        for s in sorted(k for k in data_dic[mr] if k != 'valid_sessions'):
            sd = data_dic[mr][s]
            if not isinstance(sd, dict) or sd.get('Task') is None or 'defaultdict' in str(sd.get('Task')):
                continue
            if sd['num_trials'] < min_trials:
                continue
            if not any(np.array_equal(sd['Task'], c) for c in tasks):
                tasks.append(sd['Task']); valid.append(s)
        out[mr] = valid
    return v5.apply_excluded_sessions(out, verbose=False)


def run(v5, data_dic, mouse_recdays, data_root, kind, direction, min_trials, pref_src, prefix,
        use_poisson, positive_gate, regions=None, make_pdfs=True, y_scaling='none',
        poisson_alpha=1.0, poisson_l1_ratio=0.0, poisson_solver='sklearn'):
    cfg = v5.RegressionConfigV5(use_poisson=use_poisson, regularize=True, lag_direction=direction,
                                alpha_mode='fixed', state_reduce='mean', pref_phase_source=pref_src,
                                require_positive_mean_prediction=positive_gate,
                                y_scaling=y_scaling, poisson_alpha=poisson_alpha,
                                poisson_l1_ratio=poisson_l1_ratio, poisson_solver=poisson_solver)
    if y_scaling != 'none':
        kind = f'{kind}_zscore'
    if use_poisson and poisson_alpha != 1.0:
        prefix = f'alpha{poisson_alpha:g}_{prefix}'   # alpha=1 keeps the unchanged name
    vsd = dedup(data_dic, mouse_recdays, min_trials, v5)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    # `mFC_data/data` was briefly read-only on 2026-09-07 (chmod -R by something outside this
    # pipeline; permissions restored the same day). The fallback stays as a cheap guard rather
    # than touching permissions on the published dataset.
    fig_root = os.path.join(data_root, 'figures')
    if not os.access(fig_root, os.W_OK):
        fig_root = os.path.join(os.path.dirname(data_root), 'figures_v5')
        os.makedirs(fig_root, exist_ok=True)
        print(f'  NOTE {os.path.join(data_root, "figures")} is read-only -> writing to {fig_root}',
              flush=True)
    out_dir = os.path.join(fig_root, v5.run_dir_name(cfg, stamp=stamp, prefix=prefix))
    print(f'\n{"=" * 70}\n[{hrs()}] {kind} {v5.estimator_name(cfg)} {direction} MIN_TRIALS={min_trials} '
          f'pref={pref_src} gate={positive_gate} y_scaling={y_scaling} '
          f'solver={poisson_solver} alpha={poisson_alpha:g} l1_ratio={poisson_l1_ratio:g} '
          f'-> {out_dir}\n{"=" * 70}', flush=True)
    results, pooled, diag = v5.run_and_summarise_all_mice_v5(
        data_dic, cfg, valid_sessions_dic=vsd, save_dir=out_dir, export_dir=out_dir,
        make_pdfs=make_pdfs, n_jobs=N_JOBS, verbose=True,
        manifest_extra={'min_trials': min_trials, 'run_kind': kind,
                        'launcher': 'code/slurm_v5/run_queue_v5.py'})
    table = v5.build_unit_table(results, cfg, regions=regions, data_dic=data_dic, strict=False,
                                require_anatomy=False)
    print(f'\n=== THREE PANELS [{kind} {v5.estimator_name(cfg)} {direction}] ===')
    out = v5.three_panel_summary(table)
    out.to_csv(os.path.join(out_dir, f'three_panel_summary_{direction}.csv'), index=False)
    print(f'[{hrs()}] finished {kind} {v5.estimator_name(cfg)} {direction} -> {out_dir}', flush=True)


#: spec -> (poisson_alpha, poisson_l1_ratio); l1_ratio 1.0 = lasso, 0.5 = elastic net
POISSON_SWEEP = {'14': (0.01, 1.0), '15': (0.003, 1.0), '16': (0.001, 1.0), '17': (0.003, 0.5)}


def pfc_jobs():
    sys.path.insert(0, os.path.join(ROOT, 'mFC_data', 'code'))
    import elasticnet_regression_v5 as v5
    from glm_analysis_v2 import build_data_dic_from_pfc
    data_root = os.path.join(ROOT, 'mFC_data', 'data')
    recdays = list(np.load(os.path.join(data_root, 'MetaData', 'combined_ABCDonly_days.npy')).astype(str))
    dd = build_data_dic_from_pfc(data_root, recdays, compute_norm=False, verbose=False)
    mrs = sorted(dd)
    print(f'[{hrs()}] PFC loaded: {len(mrs)} recdays', flush=True)
    if ONLY is None or '1' in ONLY:
        run(v5, dd, mrs, data_root, 'reproduction', 'future', 1, 'test', '', False, False)
    if ONLY is None or '2' in ONLY:
        run(v5, dd, mrs, data_root, 'reproduction', 'past', 1, 'test', '', True, True)
    if ONLY is None or '5' in ONLY:
        # 5. PFC Poisson future, reproduction config + gate (the prospective mirror of job 2)
        run(v5, dd, mrs, data_root, 'reproduction', 'future', 1, 'test', '', True, True)
    # 6/7: the same criterion as the raw PFC ElasticNet runs, z-scored target, so the two are
    # comparable neuron for neuron.
    if ONLY is None or '6' in ONLY:
        run(v5, dd, mrs, data_root, 'reproduction', 'past', 1, 'test', '', False, False,
            y_scaling='zscore_recday')
    if ONLY is None or '7' in ONLY:
        run(v5, dd, mrs, data_root, 'reproduction', 'future', 1, 'test', '', False, False,
            y_scaling='zscore_recday')
    # 14-17: spec 2 with a glum lasso / elastic-net penalty (spec 2 is the L2 reference).
    # Explicit-only: ONLY=None does not run the sweep.
    for spec, (alpha, l1) in POISSON_SWEEP.items():
        if ONLY is not None and spec in ONLY:
            run(v5, dd, mrs, data_root, 'reproduction', 'past', 1, 'test', '', True, True,
                poisson_alpha=alpha, poisson_l1_ratio=l1, poisson_solver='glum')
    del dd
    sys.path.pop(0)
    for m in [k for k in sys.modules if k.startswith(('elasticnet_regression_v5', 'glm_analysis_v2'))]:
        del sys.modules[m]


def lec_jobs():
    sys.path.insert(0, os.path.join(ROOT, 'code'))
    import elasticnet_regression_v5 as v5
    data_root = os.path.join(ROOT, 'data')
    with open(os.path.join(data_root, 'processed_data', 'data_dic_lec.pkl'), 'rb') as f:
        dd = pickle.load(f)
    mrs = sorted(k for k in dd if '_sb' not in k)
    regions = v5.load_unit_regions(required=False)
    print(f'[{hrs()}] LEC loaded: {len(mrs)} recdays', flush=True)
    if ONLY is None or '3' in ONLY:
        run(v5, dd, mrs, data_root, 'science', 'past', 5, 'train', '', True, False, regions=regions)
    if ONLY is None or '4' in ONLY:
        run(v5, dd, mrs, data_root, 'science', 'future', 5, 'train', '', True, False, regions=regions)
    if ONLY is None or '8' in ONLY:
        run(v5, dd, mrs, data_root, 'science', 'past', 5, 'train', '', False, False,
            regions=regions, y_scaling='zscore_recday')
    if ONLY is None or '9' in ONLY:
        run(v5, dd, mrs, data_root, 'science', 'future', 5, 'train', '', False, False,
            regions=regions, y_scaling='zscore_recday')
    # 10/11: the reproduction-config companions of 8/9 (his criterion: MIN_TRIALS=1 and the
    # held-out session's own preferred phase), so the LEC z-scored runs can be compared to the
    # paper the way the PFC ones are. `repro_` prefix, matching the raw companion run.
    if ONLY is None or '10' in ONLY:
        run(v5, dd, mrs, data_root, 'reproduction', 'past', 1, 'test', 'repro_', False, False,
            regions=regions, y_scaling='zscore_recday')
    if ONLY is None or '11' in ONLY:
        run(v5, dd, mrs, data_root, 'reproduction', 'future', 1, 'test', 'repro_', False, False,
            regions=regions, y_scaling='zscore_recday')
    # 12/13: LEC Poisson under HIS criterion -- MIN_TRIALS=1, held-out preferred phase AND his
    # `nanmean(prediction) > 0` gate. This is the matched counterpart of the PFC Poisson runs
    # (specs 2/5). Without it the only PFC result that reproduces ED 8a (90 deg, t 3.14) has no
    # LEC equivalent, so no regional comparison at 90 deg is possible: the existing LEC Poisson
    # runs (3/4) differ from it in min_trials, preferred-phase source AND the gate.
    if ONLY is None or '12' in ONLY:
        run(v5, dd, mrs, data_root, 'reproduction', 'past', 1, 'test', 'repro_', True, True,
            regions=regions)
    if ONLY is None or '13' in ONLY:
        run(v5, dd, mrs, data_root, 'reproduction', 'future', 1, 'test', 'repro_', True, True,
            regions=regions)


if ONLY is None or any(j in ONLY for j in ('1', '2', '5', '6', '7', *POISSON_SWEEP)):
    pfc_jobs()
if ONLY is None or any(j in ONLY for j in ('3', '4', '8', '9', '10', '11', '12', '13')):
    lec_jobs()
print(f'\nQUEUE DONE in {hrs()}')
