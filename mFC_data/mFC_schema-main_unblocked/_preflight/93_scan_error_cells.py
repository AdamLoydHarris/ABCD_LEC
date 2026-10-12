#!/usr/bin/env python3
"""Report every executed notebook cell that raised, against an allow-list of known failures.

    python 93_scan_error_cells.py [--all] [--json-only]
    -> logs/error_cells.json  (exit 1 if any un-allowed error cell is found)

Why: `40_run_notebook.py --allow-errors` lets nbclient record the exception into the notebook
and carry on, so its summary line prints `0 failed` and the job exits 0 while cells have in
fact failed. Measured 2026-10-08: three notebooks in this project carried hidden error cells
behind a clean EXIT=0 -- PFC Figure2_UMAP (3), and two archived LEC Figure3_fast runs (6 and
1). The run log is not evidence; the executed notebook is.

Scans `executed/` by default, `--all` adds the archive subdirectories.

I/O: reads executed/**/*.ipynb; writes logs/error_cells.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REGION = 'pfc'                                   # the LEC copy says 'lec'
ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
RUN = 'replication_run' if REGION == 'pfc' else 'lec_replication_run'
EXECUTED = os.path.join(ROOT, RUN, 'executed')
LOGDIR = os.path.join(ROOT, RUN, 'logs')

# Known-by-design failures, each with the reason it is not a defect of this run. A cell not
# listed here is a finding. Keyed by notebook basename -> {cell index: reason}.
ALLOWED = {
    'Figure2_UMAP.ipynb': {
        26: 'ephys_mean_z_shuff_all_alliterations: kernel-resident shuffle bank, never deposited',
        28: 'ephys_mean_z_shuff_dic: same, never deposited',
        33: 'ephys_mean_z_shuff_dic: same, never deposited',
    },
    'Figure3_fast.ipynb': {
        89: "deposit defect: cell 80's ses_ind leaks from cell 59, so module_shuff_dic stays empty",
    },
    'Figure3.ipynb': {
        86: 'deposit defect, as Figure3_fast cell 89',
        87: 'deposit defect, as Figure3_fast cell 89',
        89: 'deposit defect, as Figure3_fast cell 89',
    },
}
# Notebooks kept only as a record of a failed attempt; their error IS the record. Same for
# anything under an archive subdirectory (attempt1_combined_only/, attempt2b_singleday_allrows/):
# those are superseded runs, and their failures are what the current run was built to fix.
ATTEMPT_RECORD = ('_attempt', '_crash', '_KeyError', '_IndexError', '_timeout')


def scan(path):
    try:
        cells = json.load(open(path))['cells']
    except Exception as e:                       # a truncated notebook is itself a finding
        return [dict(cell=None, ename='UNREADABLE', evalue=str(e)[:200])]
    out = []
    for i, c in enumerate(cells):
        for o in c.get('outputs', []):
            if o.get('output_type') == 'error':
                out.append(dict(cell=i, ename=o.get('ename', ''), evalue=(o.get('evalue', '') or '')[:160]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='include archive subdirectories')
    ap.add_argument('--json-only', action='store_true')
    args = ap.parse_args()

    pat = os.path.join(EXECUTED, '**', '*.ipynb') if args.all else os.path.join(EXECUTED, '*.ipynb')
    files = sorted(glob.glob(pat, recursive=args.all))
    if not files:
        print(f'NOT RUN: no executed notebooks under {EXECUTED}')
        return 1

    rows, n_unexpected, n_clean = [], 0, 0
    for f in files:
        base, rel = os.path.basename(f), os.path.relpath(f, EXECUTED)
        errs = scan(f)
        archived = os.sep in rel
        record = archived or any(t in base for t in ATTEMPT_RECORD)
        if not errs:
            n_clean += 1
            continue
        allowed = ALLOWED.get(base, {})
        for e in errs:
            ok = record or e['cell'] in allowed
            if not ok:
                n_unexpected += 1
            rows.append(dict(notebook=rel, status=('archived run' if archived else 'kept failure record') if record
                             else ('allowed' if ok else 'UNEXPECTED'),
                             reason=allowed.get(e['cell'], ''), **e))

    print(f'=== error cells in {RUN}/executed{"/**" if args.all else ""} ===')
    print(f'{len(files)} notebooks, {n_clean} with no error output\n')
    for r in rows:
        print(f"  {r['status']:19} {r['notebook'][:52]:52} c{str(r['cell']):>4}  {r['ename']}: {r['evalue'][:70]}")
        if r['reason']:
            print(f"  {'':19} -> {r['reason']}")
    verdict = 'PASS' if n_unexpected == 0 else f'FAIL: {n_unexpected} unexpected error cell(s)'
    print(f'\n{verdict}')

    os.makedirs(LOGDIR, exist_ok=True)
    json.dump(dict(region=REGION, scanned=len(files), clean=n_clean,
                   unexpected=n_unexpected, verdict=verdict, rows=rows),
              open(os.path.join(LOGDIR, 'error_cells.json'), 'w'), indent=2)
    if not args.json_only:
        print(f'wrote {LOGDIR}/error_cells.json')
    return 1 if n_unexpected else 0


if __name__ == '__main__':
    sys.exit(main())
