#!/usr/bin/env python3
"""Extract the Figure 5 panel statistics into machine-readable form.

Cell 38 of `Figure5_Regression.ipynb` prints its n / t / p to stdout and saves an SVG; it
writes no stats file. So the headline numbers live only inside the executed notebook's
outputs, which is useless for comparison. This recomputes them from the per-neuron arrays
that cell 26 *does* save, and writes JSON + CSV.

Recomputing rather than scraping is deliberate: it reproduces cell 38's selection exactly
(`use_tuned=True`, `use_strict=False` -> `State_95`, `remove_nan`), so a mismatch against
the notebook's printed values would itself be a red flag.

    python 96_extract_figure5_stats.py                 # both models if present
    python 96_extract_figure5_stats.py --model Poisson_
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy import stats as st

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
MIRROR = os.path.join(ROOT, 'lec_replication_run', 'data', 'Intermediate_objects')
LOGDIR = os.path.join(ROOT, 'lec_replication_run', 'logs')

PANELS = [
    ('Predicted_Actual_correlation_mean', 'all state-tuned'),
    ('Predicted_Actual_correlation_nonzero_mean', 'non-zero-lag 30 deg excluded'),
    ('Predicted_Actual_correlation_nonzero_strict_mean', 'non-zero-lag 90 deg excluded'),
]

# published / reference values, for context only
REFERENCE = {
    'paper':            {'all state-tuned': 489, 'non-zero-lag 30 deg excluded': 329,
                         'non-zero-lag 90 deg excluded': 224},
    'his_stored_beyond': {'all state-tuned': 482, 'non-zero-lag 30 deg excluded': 278,
                          'non-zero-lag 90 deg excluded': 69},
    'v5_band':          {'all state-tuned': '447-481', 'non-zero-lag 30 deg excluded': '246-287',
                         'non-zero-lag 90 deg excluded': '79-92'},
    # --- LEC context (this copy runs on LEC data) ---
    # his code on his PFC data, from ../replication_run (REPRODUCTION.md):
    'pfc_his_code_poisson':    {'all state-tuned': '581 (t=17.15)', 'non-zero-lag 30 deg excluded': '296 (t=6.76)',
                                'non-zero-lag 90 deg excluded': '79 (t=2.67)'},
    'pfc_his_code_elasticnet': {'all state-tuned': '623 (t=15.56)', 'non-zero-lag 30 deg excluded': '307 (t=2.87)',
                                'non-zero-lag 90 deg excluded': '86 (t=2.13)'},
    # V5 reimplementation on LEC (ElasticNet a=0.01, pref_phase_source='test'), corrected
    # El-Gaby-semantics pool from code/PANEL_POOL_CORRECTION.md -- the primary comparison:
    'lec_v5_corrected':        {'all state-tuned': '1170 (t=31.88)', 'non-zero-lag 30 deg excluded': '544 (t=5.34)',
                                'non-zero-lag 90 deg excluded': '203 (t=1.72)'},
    # the superseded every-fold pool quoted in LEC_PORT_PROMPT.md, kept for traceability:
    'lec_v5_superseded_everyfold': {'all state-tuned': '887 (t=35.0)', 'non-zero-lag 30 deg excluded': '392 (t=5.98)',
                                    'non-zero-lag 90 deg excluded': '129 (t=2.89)'},
}


def load_days():
    return [str(x) for x in np.load(os.path.join(MIRROR, 'combined_ABCDonly_days.npy'))]


def run(model_prefix, addition2=''):
    """model_prefix: 'Poisson_' or '' (ElasticNet). addition2: '' (12-lag) or '_beyond'."""
    days = load_days()
    # cell 32: which recdays have results
    found = [rd for rd in days
             if os.path.exists(os.path.join(
                 MIRROR, f'{addition2}{model_prefix}Predicted_Actual_correlation_mean_{rd}.npy'))]
    if not found:
        return None

    # cell 38: neurons_tuned = State_95 concatenated over the found recdays
    state = np.hstack([np.load(os.path.join(MIRROR, f'State_95{rd}.npy'), allow_pickle=True)
                       for rd in found]).astype(bool)
    phase = np.hstack([np.load(os.path.join(MIRROR, f'Phase_{rd}.npy'), allow_pickle=True)
                       for rd in found]).astype(bool)

    out = dict(model='Poisson' if model_prefix else 'ElasticNet',
               lag_set='12-lag' if addition2 == '' else '24-lag (_beyond)',
               recdays_found=len(found), recdays_total=len(days),
               n_neurons=int(state.size), n_state_tuned=int(state.sum()),
               n_phase_tuned=int(phase.sum()), panels={})

    for fam, label in PANELS:
        v = np.hstack([np.load(os.path.join(MIRROR, f'{addition2}{model_prefix}{fam}_{rd}.npy'),
                               allow_pickle=True) for rd in found])
        if v.size != state.size:
            out['panels'][label] = dict(error=f'length {v.size} != State_95 {state.size}')
            continue
        sel = v[state]                      # cell 38 indexes by the tuning mask first
        sel = sel[~np.isnan(sel)]           # then remove_nan
        t = st.ttest_1samp(sel, 0) if sel.size > 1 else None
        out['panels'][label] = dict(
            family=f'{addition2}{model_prefix}{fam}',
            n=int(sel.size),
            mean_r=float(np.mean(sel)) if sel.size else None,
            sem=float(np.std(sel, ddof=1) / np.sqrt(sel.size)) if sel.size > 1 else None,
            t=float(t.statistic) if t else None,
            p=float(t.pvalue) if t else None,
            df=int(t.df) if t else None,
            n_finite_all_neurons=int(np.sum(~np.isnan(v))),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help="'Poisson_' or '' ; default: both")
    args = ap.parse_args()
    models = [args.model] if args.model is not None else ['Poisson_', '']

    results = []
    for m in models:
        for a2 in ('', '_beyond', '_prospective'):   # _prospective: VARIANT-PRO (2026-09-14)
            r = run(m, a2)
            if r:
                results.append(r)

    if not results:
        print('no Predicted_Actual_correlation_* files found')
        return 1

    for r in results:
        print(f"\n=== {r['model']}, {r['lag_set']} ===")
        print(f"  recdays {r['recdays_found']}/{r['recdays_total']}   "
              f"neurons {r['n_neurons']}   state-tuned(State_95) {r['n_state_tuned']}")
        for label, d in r['panels'].items():
            if 'error' in d:
                print(f"  {label:32s} ERROR {d['error']}")
                continue
            print(f"  {label:32s} n={d['n']:4d}  mean r={d['mean_r']:+.4f}  "
                  f"t={d['t']:7.3f}  p={d['p']:.3e}")
            ref = {k: v[label] for k, v in REFERENCE.items() if label in v}
            print(f"  {'':32s} reference: paper {ref['paper']}, "
                  f"his stored (_beyond) {ref['his_stored_beyond']}, V5 {ref['v5_band']}")

    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, 'figure5_stats.json'), 'w') as f:
        json.dump(dict(results=results, reference=REFERENCE), f, indent=2)
    rows = []
    for r in results:
        for label, d in r['panels'].items():
            if 'error' in d:
                continue
            rows.append(dict(model=r['model'], lag_set=r['lag_set'], panel=label,
                             n=d['n'], mean_r=d['mean_r'], sem=d['sem'],
                             t=d['t'], p=d['p'], df=d['df'],
                             n_finite_all=d['n_finite_all_neurons'],
                             neurons=r['n_neurons'], state_tuned=r['n_state_tuned']))
    with open(os.path.join(LOGDIR, 'figure5_stats.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {LOGDIR}/figure5_stats.json and .csv")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
