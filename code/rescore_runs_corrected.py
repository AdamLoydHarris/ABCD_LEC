"""Re-score every finished V5 run under the corrected El-Gaby pool rule. No re-fits.

Why this exists: `three_panel_summary(semantics='elgaby')` used to require a finite r in EVERY
fold before a neuron entered any panel. His cell 38 applies `remove_nan` to each of the three
correlation arrays independently, so each panel pools on its own finiteness. The old rule
suppressed the non-zero-lag panels by ~4x in n and ~2x in t. See `PANEL_POOL_CORRECTION.md`.

Everything here is derived from the stored `*_arrays.npz`, so no run is repeated. Outputs are
written to NEW files only -- `*_corrected.csv` / `*_corrected.svg` -- and the md5 of every
pre-existing summary is recorded before and after, so "the originals are untouched" is checked
rather than asserted.

    python rescore_runs_corrected.py [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import elasticnet_regression_v5 as v5                                    # noqa: E402

ROOT = os.path.dirname(HERE)
FIG_ROOTS = {'LEC': os.path.join(ROOT, 'data', 'figures'),
             'PFC': os.path.join(ROOT, 'mFC_data', 'data', 'figures')}
#: both pool rules, so every corrected table carries the number it supersedes next to it
SEMANTICS = ('elgaby', 'elgaby_everyfold', 'v4')


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def existing_summaries(run_dir):
    """md5 of every summary artefact that predates this script, keyed by filename."""
    out = {}
    for pat in ('three_panel_summary*.csv', 'cross_mouse_v5_*_summary*.svg',
                'cross_mouse_v5_*_summary*.png'):
        for p in glob.glob(os.path.join(run_dir, pat)):
            if '_corrected' in os.path.basename(p):
                continue
            out[os.path.basename(p)] = md5(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true', help='report what would be written')
    args = ap.parse_args()

    rows, guard_before, guard_after, failures = [], {}, {}, []
    for dataset, root in FIG_ROOTS.items():
        if not os.path.isdir(root):
            print(f'{dataset}: no {root}')
            continue
        for run_dir in sorted(glob.glob(os.path.join(root, '*_v5_*'))):
            if not glob.glob(os.path.join(run_dir, '*_arrays.npz')):
                continue
            name = os.path.basename(run_dir)
            guard_before[f'{dataset}/{name}'] = existing_summaries(run_dir)
            man = os.path.join(run_dir, 'run_config.json')
            meta = json.load(open(man)) if os.path.exists(man) else {}
            cfg = (meta.get('config') or {})
            print(f'\n=== {dataset} {name} '
                  f'(kind={meta.get("run_kind")}, min_trials={meta.get("min_trials")}, '
                  f'pref={cfg.get("pref_phase_source")}, y_scaling={cfg.get("y_scaling", "none")}, '
                  f'gate={cfg.get("require_positive_mean_prediction")})', flush=True)
            try:
                table = v5.regenerate_summary(run_dir, save=not args.dry_run,
                                              suffix='_corrected', semantics=SEMANTICS)
            except Exception as exc:                       # a bad run must not stop the sweep
                print(f'  FAILED: {type(exc).__name__}: {exc}')
                failures.append(f'{dataset}/{name}: {exc}')
                continue
            table.insert(0, 'dataset', dataset)
            table.insert(1, 'run', name)
            table.insert(2, 'run_kind', meta.get('run_kind'))
            table.insert(3, 'min_trials', meta.get('min_trials'))
            table.insert(4, 'pref_phase_source', cfg.get('pref_phase_source'))
            table.insert(5, 'y_scaling', cfg.get('y_scaling', 'none'))
            table.insert(6, 'positive_mean_gate', cfg.get('require_positive_mean_prediction'))
            direction = cfg.get('lag_direction', 'past')
            if not args.dry_run:
                dest = os.path.join(run_dir, f'three_panel_summary_{direction}_corrected.csv')
                table.to_csv(dest, index=False)
                print(f'  -> {os.path.basename(dest)}')
            rows.append(table)
            guard_after[f'{dataset}/{name}'] = existing_summaries(run_dir)

    if not rows:
        print('\nno runs with exports found')
        return 1
    combined = pd.concat(rows, ignore_index=True)
    if not args.dry_run:
        for dataset, root in FIG_ROOTS.items():
            sub = combined[combined.dataset == dataset]
            if len(sub):
                dest = os.path.join(root, 'panel_summary_all_runs_corrected.csv')
                sub.to_csv(dest, index=False)
                print(f'\ncombined {dataset} -> {dest}')

    print('\n=== originals untouched? ===')
    changed = [k for k in guard_before
               if k in guard_after and guard_before[k] != guard_after[k]]
    print('OK -- every pre-existing summary has the same md5' if not changed
          else f'CHANGED: {changed}')
    if failures:
        print('\n=== failures ===')
        for f in failures:
            print(' ', f)

    print('\n=== headline: his corrected pool rule, past runs ===')
    show = combined[(combined.semantics == 'elgaby')
                    & combined.panel.str.contains('finite fold|passing fold')]
    cols = ['dataset', 'run', 'run_kind', 'y_scaling', 'link', 'panel', 'n', 'mean_r', 't']
    cols = [c for c in cols if c in show.columns]
    print(show[cols].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
