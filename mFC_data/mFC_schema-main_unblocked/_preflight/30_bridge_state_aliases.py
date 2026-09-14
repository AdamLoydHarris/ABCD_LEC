#!/usr/bin/env python3
"""Write `State_95<rd>.npy` / `State_99<rd>.npy`, which no deposited notebook produces.

Run AFTER Figure2.ipynb.

Note the filename shape: the consumers build it as `'State_' + tuning_percentile + rd`,
with **no separator** before the recday --

    State_95ah04_01122021_02122021.npy        <- what is read
    State_ah04_01122021_02122021.npy          <- what Figure2 cell 63 writes

so these are genuinely different files, and the deposit ships neither. Readers:
Figure3 cell 51; Figure5_Figure6 cells 26/28/31/37 (28/31/37 with NO try, so they
hard-crash without them); Figure5_Regression cell 38.

THE MAPPING IS EVIDENCED, NOT GUESSED. Figure2 cell 56 lines 82-85 set

    Tuned_dic2['State']['95'] = Tuned_dic['State_zmax_bool']
    Tuned_dic2['State']['99'] = Tuned_dic['State_zmax_bool_strict']

and cell 63 writes `State_<rd>.npy` from `State_zmax_bool` and `State_strict_<rd>.npy`
from `State_zmax_bool_strict`. Hence State_95 <- State_, State_99 <- State_strict_.
Figure5_Regression cell 38's own comment ("use_strict ... p=0.01 threshold for state
tuning") agrees.

Cell 48 derives both booleans from the same `State_zmax` p-value matrix at 0.05 and 0.01,
so `State_99` must be a SUBSET of `State_95`. That is asserted here per recday -- it is
the one structural check available on this mapping.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
MIRROR = os.path.join(ROOT, 'replication_run', 'data', 'Intermediate_objects')
LOGDIR = os.path.join(ROOT, 'replication_run', 'logs')

# (read name, written name, percentile)
PAIRS = [('State_', 'State_95', '95'), ('State_strict_', 'State_99', '99')]


def main():
    allow_empty = '--allow-empty' in sys.argv
    if not os.path.isdir(MIRROR):
        sys.exit(f'FATAL: {MIRROR} not found')

    day_lists = ['combined_ABCDonly_days.npy', '3_task_all_days.npy',
                 'combined_ABCDE_days.npy']
    recdays = []
    for dl in day_lists:
        p = os.path.join(MIRROR, dl)
        if os.path.exists(p):
            recdays += [str(x) for x in np.load(p)]
    recdays = list(dict.fromkeys(recdays))
    print(f'{len(recdays)} recdays from {", ".join(day_lists)}')

    written, missing, nested_fail, report = 0, [], [], {}
    for rd in recdays:
        arrs = {}
        for src, dst, pct in PAIRS:
            sp = os.path.join(MIRROR, f'{src}{rd}.npy')
            if not os.path.exists(sp):
                missing.append(f'{src}{rd}.npy')
                continue
            a = np.load(sp, allow_pickle=True)
            # Figure2 cell 63 can pickle an empty auto-vivified defaultdict when the
            # upstream dict entry was never populated; refuse to propagate that.
            if a.dtype == object and a.ndim == 0:
                missing.append(f'{src}{rd}.npy (empty defaultdict, not an array)')
                continue
            np.save(os.path.join(MIRROR, f'{dst}{rd}.npy'), a)
            arrs[pct] = a
            written += 1

        if '95' in arrs and '99' in arrs:
            a95 = np.asarray(arrs['95']).astype(bool).ravel()
            a99 = np.asarray(arrs['99']).astype(bool).ravel()
            if a95.shape != a99.shape:
                nested_fail.append(f'{rd}: shape {a95.shape} vs {a99.shape}')
            elif not np.all(a99 <= a95):
                nested_fail.append(f'{rd}: State_99 is NOT a subset of State_95 '
                                   f'({int(np.sum(a99 & ~a95))} neurons in 99 only)')
            report[rd] = dict(n=int(a95.size), n95=int(a95.sum()), n99=int(a99.sum()))

    print(f'\nwrote {written} files')
    if report:
        tot = sum(v['n'] for v in report.values())
        t95 = sum(v['n95'] for v in report.values())
        t99 = sum(v['n99'] for v in report.values())
        print(f'across {len(report)} recdays: {tot} neurons, '
              f'State_95 {t95}, State_99 {t99}')
    # A check that made zero comparisons must not report PASS -- that is how a
    # mis-ordered run (bridge before Figure2) looks like a success.
    if not report:
        print('nesting check (State_99 subset of State_95): NOT RUN (0 comparisons)')
    else:
        print(f'nesting check (State_99 subset of State_95): '
              f'{"PASS" if not nested_fail else "FAIL"} over {len(report)} recdays')
    for x in nested_fail:
        print(f'  {x}')
    if missing:
        print(f'\n{len(missing)} inputs absent (Figure2 did not produce them):')
        for m in missing[:20]:
            print(f'  {m}')
        if len(missing) > 20:
            print(f'  ... and {len(missing) - 20} more')

    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, 'bridge_state_aliases.json'), 'w') as f:
        json.dump(dict(written=written, per_recday=report, missing=missing,
                       nesting_failures=nested_fail), f, indent=2)
    if written == 0 and not allow_empty:
        print('\nFAIL: wrote nothing. Figure2 has not produced State_/State_strict_ yet --'
              '\n      run Figure2.ipynb first, or pass --allow-empty to probe.')
        return 1
    return 1 if nested_fail else 0


if __name__ == '__main__':
    sys.exit(main())
