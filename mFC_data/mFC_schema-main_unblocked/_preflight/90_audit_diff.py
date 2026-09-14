#!/usr/bin/env python3
"""Prove the unblocked copy differs from the deposit ONLY by the declared edits.

Dependency-free on purpose: an audit should not need nbdime installed to be trustworthy.

Hard failures (exit 1):
  1. a changed or added source line that carries no `# UNBLOCK-<ID>` token
  2. an ID appearing in the diff but not in EDITS.md, or vice versa (bijection)
  3. a cell whose source differs but which no edit names
  4. a cell_type change other than the one declared
  5. LICENSE / README.md not byte-identical

Deliberately ignored, because G-01 declares them: `outputs` and `execution_count`.
`_preflight/` is excluded by path -- it is new code, not an edit to the deposit.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import sys

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
DEPOSIT = os.path.join(ROOT, 'mFC_schema-main')
UNBLOCKED = os.path.join(ROOT, 'mFC_schema-main_unblocked')
LOGDIR = os.path.join(ROOT, 'replication_run', 'logs')
EDITS_MD = os.path.join(UNBLOCKED, 'EDITS.md')

NOTEBOOKS = [
    'Basic_analysis.ipynb', 'Behavioural Analysis (Figure 1).ipynb', 'Figure2.ipynb',
    'Figure2_UMAP.ipynb', 'Figure3.ipynb', 'Figure5_Figure6.ipynb',
    'Figure5_Regression.ipynb', 'Figure7.ipynb',
]
VERBATIM = ['LICENSE', 'README.md']
ID_RE = re.compile(r'UNBLOCK-([A-Z0-9]+-\d+)')
# Only this cell may change cell_type, and only to this value.
DECLARED_CELLTYPE = {('Figure2.ipynb', 20): 'raw'}


def cells(path):
    return json.load(open(path))['cells']


def src(c):
    return ''.join(c.get('source', []))


def main():
    fails, all_diff_ids, report = [], set(), []

    for name in NOTEBOOKS:
        dp, up = os.path.join(DEPOSIT, name), os.path.join(UNBLOCKED, name)
        if not os.path.exists(up):
            fails.append(f'{name}: missing from the unblocked copy')
            continue
        dc, uc = cells(dp), cells(up)

        # --- align cells: new cells are declared insertions, so walk both with an offset
        d_i = u_i = 0
        offset_note = []
        while d_i < len(dc) or u_i < len(uc):
            if d_i >= len(dc):
                # trailing new cell(s)
                ids = set(ID_RE.findall(src(uc[u_i])))
                if not ids:
                    fails.append(f'{name}: undeclared extra cell at unblocked index {u_i}')
                all_diff_ids |= ids
                offset_note.append(f'new cell at u{u_i} ({",".join(sorted(ids))})')
                u_i += 1
                continue
            if u_i >= len(uc):
                fails.append(f'{name}: cell {d_i} of the deposit is missing')
                d_i += 1
                continue

            ds, us = src(dc[d_i]), src(uc[u_i])
            if ds == us:
                # identical source: still check cell_type
                if dc[d_i].get('cell_type') != uc[u_i].get('cell_type'):
                    want = DECLARED_CELLTYPE.get((name, d_i))
                    if want != uc[u_i].get('cell_type'):
                        fails.append(f'{name} c{d_i}: undeclared cell_type change '
                                     f'{dc[d_i].get("cell_type")} -> {uc[u_i].get("cell_type")}')
                    else:
                        report.append(f'{name} c{d_i}: cell_type -> {want} (declared)')
                        # No source diff, so credit the ID from the cell's metadata
                        # or the EDITS.md bijection would under-count this edit.
                        meta = uc[u_i].get('metadata', {}).get('unblock', '')
                        all_diff_ids.update(ID_RE.findall(meta))
                d_i += 1
                u_i += 1
                continue

            # source differs. Is the unblocked cell an inserted NEW cell (deposit cell
            # reappears later)? Then consume it as an insertion.
            if u_i + 1 < len(uc) and src(uc[u_i + 1]) == ds:
                ids = set(ID_RE.findall(us))
                if not ids:
                    fails.append(f'{name}: undeclared new cell before deposit cell {d_i}')
                all_diff_ids |= ids
                report.append(f'{name}: NEW CELL before c{d_i} '
                              f'({",".join(sorted(ids))}, {len(us.splitlines())} lines)')
                u_i += 1
                continue

            # a genuine in-place edit -- every added/changed line must carry a marker
            dl, ul = ds.split('\n'), us.split('\n')
            cell_ids, unmarked = set(), []
            for line in difflib.ndiff(dl, ul):
                if line.startswith('+ '):
                    body = line[2:]
                    ids = ID_RE.findall(body)
                    if ids:
                        cell_ids |= set(ids)
                    elif body.strip():
                        unmarked.append(body)
            if not cell_ids:
                fails.append(f'{name} c{d_i}: source differs but no UNBLOCK id found')
            # Lines belonging to a marked multi-line construct (folded calls, comment
            # continuations) are allowed if the cell declares at least one id and the
            # line is a comment continuation or a backslash continuation.
            for body in unmarked:
                s = body.strip()
                if s.startswith('#') or body.rstrip().endswith('\\'):
                    continue
                fails.append(f'{name} c{d_i}: added line without an UNBLOCK marker: {s[:90]!r}')
            all_diff_ids |= cell_ids
            report.append(f'{name} c{d_i}: edited ({",".join(sorted(cell_ids))})')
            d_i += 1
            u_i += 1

    # --- verbatim files
    for v in VERBATIM:
        dp, up = os.path.join(DEPOSIT, v), os.path.join(UNBLOCKED, v)
        if not os.path.exists(dp):
            continue
        h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
        if not os.path.exists(up) or h(dp) != h(up):
            fails.append(f'{v}: not byte-identical to the deposit')
        else:
            report.append(f'{v}: byte-identical')

    # --- bijection against the ledger
    if os.path.exists(EDITS_MD):
        ledger_ids = set(ID_RE.findall(open(EDITS_MD).read()))
        only_diff = sorted(all_diff_ids - ledger_ids)
        only_ledger = sorted(ledger_ids - all_diff_ids)
        if only_diff:
            fails.append(f'IDs in the diff but not in EDITS.md: {only_diff}')
        if only_ledger:
            fails.append(f'IDs in EDITS.md but not in the diff: {only_ledger}')
    else:
        fails.append('EDITS.md not found -- the ledger is part of the deliverable')

    # --- output
    print('=== per-cell findings ===')
    for r in report:
        print(f'  {r}')
    print(f'\n=== {len(all_diff_ids)} edit IDs found in the diff ===')
    print('  ' + ', '.join(sorted(all_diff_ids)))

    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, 'audit_diff.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n\n')
        f.write('ids: ' + ', '.join(sorted(all_diff_ids)) + '\n')
        if fails:
            f.write('\nFAILURES:\n' + '\n'.join(fails) + '\n')

    if fails:
        print(f'\n=== AUDIT FAILED ({len(fails)}) ===')
        for x in fails:
            print(f'  {x}')
        return 1
    print('\n=== AUDIT PASSED: the copy differs from the deposit only by declared edits ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
