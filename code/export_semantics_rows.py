"""Export a table explaining the rows of a corrected `cross_mouse_v5_*_summary_corrected.svg`.

Those figures carry one row per (link x semantics). The three semantics differ only in
bookkeeping -- same fits, same betas, same state-tuning test, same per-fold top-3 lag test -- so
a reader needs the rule alongside the numbers. This writes both:

    semantics_rows_<direction>_definitions.{csv,md}   what distinguishes the rows
    semantics_rows_<direction>_numbers.{csv,md}       n / mean r / t per row x panel

    python export_semantics_rows.py <run dir> [<run dir> ...]
    python export_semantics_rows.py --all

Nothing is overwritten: outputs are new files named `semantics_rows_*`.
"""
from __future__ import annotations

import argparse
import glob
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
SEMANTICS = ('elgaby', 'elgaby_everyfold', 'v4')

#: The three bookkeeping steps, per semantics. This is the substance of the difference: every
#: other part of the pipeline is shared, so these rows are the same betas counted three ways.
DEFINITIONS = [
    {'row (semantics)': 'elgaby',
     'what it is': "El-Gaby's cell 38, as his code actually does it -- the reproduction claim",
     '1. who enters the pool': 'state-tuned AND >=1 fold gave a finite r FOR THAT PANEL '
                               '(each panel pools independently)',
     '2. is the neuron "anchored"': '>=1 passing fold (any-fold)',
     '3. value per neuron': 'mean over PASSING folds only',
     'use it for': 'comparing to the paper'},
    {'row (semantics)': 'elgaby_everyfold',
     'what it is': 'SUPERSEDED (2026-09-08). Our earlier reading of "with non-zero beta '
                   'coefficients"; see PANEL_POOL_CORRECTION.md',
     '1. who enters the pool': 'the above AND a finite r in EVERY fold',
     '2. is the neuron "anchored"': '>=1 passing fold (any-fold)',
     '3. value per neuron': 'mean over PASSING folds only',
     'use it for': 'tracing any number we reported before 2026-09-08'},
    {'row (semantics)': 'v4',
     'what it is': 'our own bookkeeping -- no selection on the quantity being averaged',
     '1. who enters the pool': 'state-tuned AND a finite fold-average r',
     '2. is the neuron "anchored"': 'MAJORITY of folds pass',
     '3. value per neuron': 'mean over ALL fitted folds',
     'use it for': 'a claim of our own'},
]

NOTE = (
    'All three rows come from the SAME fits and the SAME betas; nothing about the model, the '
    'regressors, the state-tuning test or the per-fold top-3 lag test differs between them. '
    'Step 1 is strictest in `elgaby_everyfold`, step 2 is loosest in the two `elgaby` rows, and '
    'step 3 conditions the average on the folds that passed the test (a selection on the '
    'quantity being averaged), which is why `v4` exists for claims of our own.'
)


def one_run(run_dir):
    runs, cfg = {}, None
    for path in sorted(glob.glob(os.path.join(run_dir, '*_arrays.npz'))):
        res = v5.load_regression_outputs(path)
        runs[res['mouse_recday']] = res
        cfg = cfg or v5._config_from_results(res)
    if not runs:
        raise FileNotFoundError(f'no *_arrays.npz in {run_dir}')
    table = v5.build_unit_table(runs, cfg, require_anatomy=False)
    got = v5.three_panel_summary(table, semantics=SEMANTICS, verbose=False)
    keep = [c for c in ('link', 'semantics', 'panel', 'n', 'mean_r', 'frac_pos', 't', 'P',
                        'effect_size', 'paper_n', 'paper_t') if c in got.columns]
    return cfg, got[keep]


def _md_table(df):
    """A GitHub-flavoured pipe table. Hand-rolled: `to_markdown` needs `tabulate`, which is not
    in the `maze_ephys` env, and this is not worth a new dependency."""
    def cell(x):
        if isinstance(x, float):
            return f'{x:.4g}'
        return str(x).replace('|', r'\|').replace('\n', ' ')
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cell(c) for c in cols) + ' |',
           '|' + '|'.join('---' for _ in cols) + '|']
    for row in df.itertuples(index=False):
        out.append('| ' + ' | '.join(cell(v) for v in row) + ' |')
    return '\n'.join(out)


def write(df, stem, note=None):
    df.to_csv(stem + '.csv', index=False)
    with open(stem + '.md', 'w') as f:
        f.write(_md_table(df) + '\n')
        if note:
            f.write('\n' + note + '\n')
    print(f'  -> {os.path.basename(stem)}.csv / .md')


#: What "reproduction" and "science" mean. Only two settings differ; everything else in the
#: 39-key config is identical, so these two lines are the whole difference.
CONFIG_DEFINITIONS = [
    {'setting': 'pref_phase_source', 'reproduction': "'test'", 'science': "'train'",
     'what it does': "which session's preferred goal-progress phase (which third of each state) "
                     'chooses the rows that are fitted and the bins that are scored',
     'why it matters': 'LEAKAGE. His cell 21 reads the HELD-OUT session, so the test task picks '
                       'the bins its own correlation is averaged over. 87% of units change '
                       "preferred phase between folds under 'test' vs 44% under 'train'"},
    {'setting': 'min_trials', 'reproduction': '1', 'science': '5',
     'what it does': 'minimum completed trials for a session to become a fold '
                     "(his `non_repeat_ses_maker` keeps any session with num_trials > 0)",
     'why it matters': 'admits four 1-4-trial LEC sessions (142 folds vs 138). Those folds are '
                       'noisy enough that the top-3 lag test passes near its random rate'},
]

CONFIG_NOTE = (
    'Everything else is identical between the two configs (checked by diffing `run_config.json`: '
    '37 of 39 keys match), including the state-tuning statistic, the preferred-phase rule, '
    'alpha, the lag sets and the top-3 criterion. Quote the reproduction run only for '
    '"under his exact criterion we get X"; quote the science run for a claim about the data. '
    'The `repro_` filename prefix marks the reproduction companion but is not derived from the '
    'config -- `run_kind` and `pref_phase_source` in the manifest are authoritative.'
)


def manifest(run_dir):
    import json
    p = os.path.join(run_dir, 'run_config.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def config_diff(run_dirs):
    """Which settings actually differ across the given runs, read from their manifests."""
    flat = {}
    for d in run_dirs:
        m = manifest(d)
        flat[os.path.basename(d)] = {'run_kind': m.get('run_kind'),
                                     'min_trials': m.get('min_trials'),
                                     **(m.get('config') or {})}
    keys = sorted(set().union(*[set(v) for v in flat.values()])) if flat else []
    rows = []
    for k in keys:
        vals = {n: v.get(k) for n, v in flat.items()}
        if len({repr(x) for x in vals.values()}) > 1:
            rows.append({'setting': k, **{n: repr(x) for n, x in vals.items()}})
    n_same = len(keys) - len(rows)
    return pd.DataFrame(rows), n_same


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dirs', nargs='*')
    ap.add_argument('--all', action='store_true', help='every run with exports, both datasets')
    ap.add_argument('--compare', nargs='+', metavar='RUN_DIR',
                    help='two or more runs: write a config diff (e.g. reproduction vs science) '
                         'and their panel numbers side by side')
    ap.add_argument('--out', default=None, help='directory for --compare output')
    args = ap.parse_args()

    if args.compare:
        out_dir = args.out or args.compare[0].rstrip('/')
        diff, n_same = config_diff([d.rstrip('/') for d in args.compare])
        print(f'\n=== settings that differ ({n_same} others identical) ===')
        print(diff.to_string(index=False) if len(diff) else '  (none)')
        write(pd.DataFrame(CONFIG_DEFINITIONS),
              os.path.join(out_dir, 'run_comparison_definitions'), CONFIG_NOTE)
        write(diff, os.path.join(out_dir, 'run_comparison_config_diff'),
              f'{n_same} further settings are identical across these runs.')
        frames = []
        for d in args.compare:
            d = d.rstrip('/')
            try:
                _, nums = one_run(d)
            except Exception as exc:
                print(f'  {os.path.basename(d)} FAILED: {exc}')
                continue
            m = manifest(d)
            nums.insert(0, 'run', os.path.basename(d))
            nums.insert(1, 'run_kind', m.get('run_kind'))
            nums.insert(2, 'min_trials', m.get('min_trials'))
            nums.insert(3, 'pref_phase_source', (m.get('config') or {}).get('pref_phase_source'))
            frames.append(nums)
        both = pd.concat(frames, ignore_index=True)
        write(both, os.path.join(out_dir, 'run_comparison_numbers'))
        print('\n=== panel numbers, both runs ===')
        print(both[both.semantics == 'elgaby'].to_string(index=False))
        return 0

    dirs = list(args.run_dirs)
    if args.all:
        for root in (os.path.join(ROOT, 'data', 'figures'),
                     os.path.join(ROOT, 'mFC_data', 'data', 'figures')):
            dirs += [d for d in sorted(glob.glob(os.path.join(root, '*_v5_*')))
                     if glob.glob(os.path.join(d, '*_arrays.npz'))]
    if not dirs:
        ap.error('give at least one run directory, or --all')

    defs = pd.DataFrame(DEFINITIONS)
    for run_dir in dirs:
        run_dir = run_dir.rstrip('/')
        print(f'\n=== {os.path.basename(run_dir)}')
        try:
            cfg, nums = one_run(run_dir)
        except Exception as exc:
            print(f'  FAILED: {type(exc).__name__}: {exc}')
            continue
        d = cfg.lag_direction
        write(defs, os.path.join(run_dir, f'semantics_rows_{d}_definitions'), NOTE)
        write(nums, os.path.join(run_dir, f'semantics_rows_{d}_numbers'))
        print(nums.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
