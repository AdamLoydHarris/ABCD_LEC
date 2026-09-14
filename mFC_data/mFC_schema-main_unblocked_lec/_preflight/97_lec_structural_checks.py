#!/usr/bin/env python3
"""Structural checks on the LEC run of El-Gaby's pipeline -- run these BEFORE trusting any
scientific number. They are the checks LEC_PORT_HANDOFF.md section 8 lists, made into a
script so they are reproducible rather than ad hoc:

  1. GLM_anchoring_prep_dic_regressors<rd> has 324 columns (9 nodes x 3 phases x 12 lags)
  2. Phases_raw2_ values are {0,1,2}, Phases_raw_ are {0..4}, equal length per session
  3. 100 % of non-zero fitted coefficients lie on the mod-3 stripe
       anchor_phase == (pref_phase - lag) mod 3
     for BOTH models. Poisson (L2 only) should also show a non-zero fraction of exactly
     12/36 = 1/3; ElasticNet (L1 + positive) may be sparser, so only purity is asserted.
     Fold k of the coefficient array pairs with non_repeat_ses[k]; the preferred phase is
     read from tuning_phase_boolean_max_<rd>[session], as cell 21 does.
  4. State_99 is a subset of State_95 per recday
  5. speed_ files exist for every exported session; neuron totals vs 2851
  6. (separate script, 98_compare_neurons_norm.py) pipeline Neuron_ vs data_dic Neurons_norm

Every check prints PASS / FAIL / NOT RUN, and a check that made zero comparisons is NOT RUN,
never PASS. Exit 1 on any FAIL. Results -> lec_replication_run/logs/lec_structural_checks.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np


def rec_dd():
    """Figure2 cell 80 pickles an EMPTY autovivified defaultdict(rec_dd) for every session
    cell 18 skipped (the 30 zero-trial / Object sessions). np.load of those files unpickles
    the factory by name from __main__, so it must exist here or the load raises."""
    return defaultdict(rec_dd)


ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
MIRROR = os.path.join(ROOT, 'lec_replication_run', 'data', 'Intermediate_objects')
LOGDIR = os.path.join(ROOT, 'lec_replication_run', 'logs')
NUM_NODES, NUM_PHASES, NUM_LAGS = 9, 3, 12
NUM_REGRESSORS = NUM_NODES * NUM_PHASES * NUM_LAGS
TOTAL_NEURONS = 2851
MODELS = [('Poisson_', 'Poisson'), ('', 'ElasticNet')]


def P(name):
    return os.path.join(MIRROR, name)


def load(name, **kw):
    return np.load(P(name), **kw)


def flatten(x):
    """Descend nested object arrays / lists to one flat float vector (the per-bin content
    that two concatenate_complex2 calls expose), independent of how numpy packed them."""
    out = []

    def rec(y):
        if isinstance(y, np.ndarray) and y.dtype != object and y.ndim <= 1:
            out.extend(np.asarray(y, dtype=float).ravel().tolist())
        elif isinstance(y, (list, tuple)) or isinstance(y, np.ndarray):
            for z in y:
                rec(z)
        else:
            out.append(float(y))
    rec(x)
    return np.asarray(out, dtype=float)


def non_repeat_ses_maker(Tasks, num_trials_day):
    """Verbatim transcription of the notebooks' selector."""
    non_repeat_bool_all = []
    for ses_ind in np.arange(len(Tasks)):
        if ses_ind == 0:
            nr = True
        else:
            nr = np.sum([np.array_equal(Tasks[ses_ind], Tasks[:ses_ind][jj])
                         for jj in range(len(Tasks[:ses_ind]))]) == 0
        non_repeat_bool_all.append(nr)
    non_repeat_bool_all = np.hstack((non_repeat_bool_all))
    return np.where(np.logical_and(non_repeat_bool_all, num_trials_day > 0))[0]


def verdict(name, ok, n, detail=''):
    tag = 'NOT RUN' if n == 0 else ('PASS' if ok else 'FAIL')
    print(f'[{tag:7s}] {name}  ({n} comparisons){"  " + detail if detail else ""}')
    return tag


def main():
    recdays = [str(r) for r in load('combined_ABCDonly_days.npy')]
    results, tags = {}, {}

    # ---- 1. regressor columns
    per, n = {}, 0
    for rd in recdays:
        f = P(f'GLM_anchoring_prep_dic_regressors{rd}.npy')
        if not os.path.exists(f):
            per[rd] = 'missing'
            continue
        a = np.load(f, allow_pickle=True)
        folds = list(a) if a.dtype == object else [a[i] for i in range(a.shape[0])]
        cols = sorted({int(np.asarray(x).shape[1]) for x in folds})
        per[rd] = dict(n_folds=len(folds), cols=cols)
        n += 1
    ok = n > 0 and all(isinstance(v, dict) and v['cols'] == [NUM_REGRESSORS] for v in per.values() if v != 'missing')
    tags['regressor_columns'] = verdict('1. regressor columns == 324', ok, n,
                                        f'{n}/{len(recdays)} recdays present')
    results['regressor_columns'] = per

    # ---- 2. phase arrays
    per, n, bad = {}, 0, []
    for rd in recdays:
        Tasks = load(f'Task_data_{rd}.npy'); nt = load(f'Num_trials_{rd}.npy')
        for s in range(len(Tasks)):
            f3, f5 = P(f'Phases_raw2_{rd}_{s}.npy'), P(f'Phases_raw_{rd}_{s}.npy')
            if not (os.path.exists(f3) and os.path.exists(f5)):
                continue
            p3 = np.load(f3, allow_pickle=True); p5 = np.load(f5, allow_pickle=True)
            if p3.dtype == object and p3.ndim == 0:      # pickled empty defaultdict
                per[f'{rd}_{s}'] = 'empty-defaultdict'
                if nt[s] > 0:
                    bad.append(f'{rd}_{s}: empty phase file for a live session')
                continue
            v3, v5 = flatten(p3), flatten(p5)
            u3, u5 = sorted(set(v3.tolist())), sorted(set(v5.tolist()))
            rec = dict(len3=len(v3), len5=len(v5), uniq3=u3, uniq5=u5)
            per[f'{rd}_{s}'] = rec
            n += 1
            if not (u3 == [0.0, 1.0, 2.0] and set(u5) <= {0.0, 1.0, 2.0, 3.0, 4.0}
                    and len(v3) == len(v5) and len(v3) > 0):
                bad.append(f'{rd}_{s}: {rec}')
    tags['phase_arrays'] = verdict('2. Phases_raw2_ is 3-bin, Phases_raw_ is 5-bin, equal length',
                                   not bad, n, f'{n} sessions; {len(bad)} bad')
    for b in bad[:10]:
        print('        ', b)
    results['phase_arrays'] = dict(n_sessions=n, bad=bad, per_session=per)

    # ---- 3. the mod-3 stripe, both models
    for prefix, label in MODELS:
        per, on_t, off_t, nz_t, fin_t, n = {}, 0, 0, 0, 0, 0
        fold_mismatch = []
        for rd in recdays:
            cf, tf = P(f'{prefix}GLM_anchoring_coeffs_all_{rd}.npy'), P(f'tuning_phase_boolean_max_{rd}.npy')
            if not (os.path.exists(cf) and os.path.exists(tf)):
                per[rd] = 'missing'
                continue
            coeffs = np.load(cf, allow_pickle=True)
            tpbm = np.load(tf, allow_pickle=True)
            nrs = non_repeat_ses_maker(load(f'Task_data_{rd}.npy'), load(f'Num_trials_{rd}.npy'))
            if coeffs.ndim != 3 or coeffs.shape[2] != NUM_REGRESSORS:
                per[rd] = f'unexpected coeff shape {coeffs.shape}'
                fold_mismatch.append(per[rd]); continue
            if coeffs.shape[1] != len(nrs):
                per[rd] = f'fold count {coeffs.shape[1]} != len(non_repeat_ses) {len(nrs)}'
                fold_mismatch.append(f'{rd}: {per[rd]}'); continue
            on = off = nz = fin = 0
            lag = np.arange(NUM_LAGS)[None, None, None, :]
            ap = np.arange(NUM_PHASES)[None, None, :, None]
            for k, ses in enumerate(nrs):
                pref = np.argmax(np.asarray(tpbm[int(ses)], dtype=float), axis=1)   # (n_neurons,)
                c = coeffs[:, k, :].reshape(-1, NUM_NODES, NUM_PHASES, NUM_LAGS)
                finite = np.isfinite(c)
                nonzero = finite & (c != 0)
                stripe = (ap == (pref[:, None, None, None] - lag) % NUM_PHASES)  # (n,1,3,12)
                on += int((nonzero & stripe).sum()); off += int((nonzero & ~stripe).sum())
                nz += int(nonzero.sum()); fin += int(finite.sum())
            per[rd] = dict(neurons=int(coeffs.shape[0]), folds=int(coeffs.shape[1]), on=on, off=off,
                           purity=(on / (on + off) if on + off else None),
                           nonzero_frac=(nz / fin if fin else None))
            on_t += on; off_t += off; nz_t += nz; fin_t += fin; n += 1
        purity = on_t / (on_t + off_t) if on_t + off_t else None
        frac = nz_t / fin_t if fin_t else None
        ok = n > 0 and not fold_mismatch and off_t == 0 and on_t > 0
        detail = (f'{n}/{len(recdays)} recdays; on-stripe {on_t}, off-stripe {off_t}, '
                  f'purity {purity if purity is None else round(100 * purity, 4)} %, '
                  f'non-zero fraction {frac if frac is None else round(frac, 4)}'
                  f'{" (Poisson expects 0.3333)" if prefix else " (<= 0.3333 expected)"}')
        tags[f'stripe_{label}'] = verdict(f'3. mod-3 stripe purity, {label}', ok, n, detail)
        for m in fold_mismatch:
            print('        ', m)
        results[f'stripe_{label}'] = dict(on=on_t, off=off_t, purity=purity, nonzero_frac=frac,
                                          fold_mismatch=fold_mismatch, per_recday=per)

    # ---- 4. State_99 subset of State_95 (combined recdays + the single-day cohort)
    per, n, bad = {}, 0, []
    singles = [str(r) for r in load('3_task_all_days.npy')] if os.path.exists(P('3_task_all_days.npy')) else []
    for rd in recdays + singles:
        f95, f99 = P(f'State_95{rd}.npy'), P(f'State_99{rd}.npy')
        if not (os.path.exists(f95) and os.path.exists(f99)):
            continue
        a95 = np.asarray(load(f'State_95{rd}.npy', allow_pickle=True), dtype=bool)
        a99 = np.asarray(load(f'State_99{rd}.npy', allow_pickle=True), dtype=bool)
        sub = a95.shape == a99.shape and bool(np.all(a99 <= a95))
        per[rd] = dict(n=int(a95.size), tuned95=int(a95.sum()), tuned99=int(a99.sum()), subset=sub)
        n += 1
        if not sub:
            bad.append(rd)
    tags['state_subset'] = verdict('4. State_99 subset of State_95', not bad, n,
                                   f'{n} recdays; tuned95={sum(v["tuned95"] for v in per.values())}, '
                                   f'tuned99={sum(v["tuned99"] for v in per.values())}, '
                                   f'neurons={sum(v["n"] for v in per.values())}')
    results['state_subset'] = per

    # ---- 5. coverage counts
    # cell 16 computes speed wherever XY_raw_ exists (combined AND single-day names, including
    # trial-less sessions with tracking), so the reference count is the XY_raw_ files.
    n_speed = sum(1 for f in os.listdir(MIRROR) if f.startswith('speed_') and f.endswith('.npy'))
    n_exported = sum(1 for f in os.listdir(MIRROR) if f.startswith('XY_raw_') and f.endswith('.npy'))
    fig5 = {}
    for prefix, label in MODELS:
        tot, tuned, found = 0, 0, 0
        for rd in recdays:
            f = P(f'{prefix}Predicted_Actual_correlation_mean_{rd}.npy')
            if not os.path.exists(f):
                continue
            v = load(f'{prefix}Predicted_Actual_correlation_mean_{rd}.npy')
            tot += int(len(v)); found += 1
            f95 = P(f'State_95{rd}.npy')
            if os.path.exists(f95):
                tuned += int(np.asarray(load(f'State_95{rd}.npy', allow_pickle=True), dtype=bool).sum())
        fig5[label] = dict(recdays_found=found, neurons=tot, state_tuned=tuned)
    ok = n_speed == n_exported
    tags['coverage'] = verdict('5. speed_ files == tracked sessions (XY_raw_ files)', ok, 1 if n_exported else 0,
                               f'speed {n_speed} vs XY_raw_ {n_exported}; Figure 5 entrants: {fig5}; '
                               f'total LEC neurons {TOTAL_NEURONS}')
    results['coverage'] = dict(speed_files=n_speed, exported_sessions=n_exported, figure5=fig5)

    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, 'lec_structural_checks.json'), 'w') as f:
        json.dump(dict(tags=tags, results=results), f, indent=2, default=str)
    print(f'\nwrote {LOGDIR}/lec_structural_checks.json')
    fails = [k for k, v in tags.items() if v == 'FAIL']
    notrun = [k for k, v in tags.items() if v == 'NOT RUN']
    print(f'FAIL: {fails or "none"};  NOT RUN: {notrun or "none"}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
