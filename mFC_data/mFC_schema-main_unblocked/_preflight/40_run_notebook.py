#!/usr/bin/env python3
"""Execute one unblocked notebook cell by cell, with progress and inline gates.

Why not plain `jupyter nbconvert --execute`: several deposited cells swallow their own
failures in a bare `except`, so a run can "succeed" while producing empty dicts and
plausible-looking garbage downstream. This runner
  * prints one line per cell as it starts and finishes, so a multi-hour cell is visible,
  * stops on the first genuine exception (no --allow-errors by default),
  * runs declared gates in the live kernel between cells,
  * can skip cells by index (Figure5_Figure6 cell 9),
  * writes the executed notebook out so stored outputs can be compared to the deposit's.

`raw` cells are skipped by the kernel automatically -- that is how Figure2 cell 20 stays
out of a Run-All (edit UNBLOCK-F2-04).

Usage:
    python 40_run_notebook.py Figure2.ipynb [--skip 9,12] [--only 0-30] [--allow-errors]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
UNBLOCKED = os.path.join(ROOT, 'mFC_schema-main_unblocked')
LOGDIR = os.path.join(ROOT, 'replication_run', 'logs')
EXECUTED = os.path.join(ROOT, 'replication_run', 'executed')

# Gates run in the live kernel AFTER the cell whose source CONTAINS the given marker.
#
# Keyed by source text, not by index, on purpose: UNBLOCK-F2-06 inserts a cell at index
# 31, so every later cell shifts +1 relative to the deposit's numbering. An index-keyed
# gate for "deposit cell 46" would fire after unblocked cell 46 -- the wrong cell, and
# usually an empty spacer, so it would fail silently rather than loudly.
GATES = {
    ('Figure2.ipynb', '###calculating speed'): (
        "assert len(speed_dic) > 0, 'speed_dic is EMPTY after cell 16'\n"
        "_n = sum(len(v) for v in speed_dic.values())\n"
        "assert _n > 0, 'speed_dic has recdays but no sessions'\n"
        "print(f'GATE ok: speed_dic covers {len(speed_dic)} recdays, {_n} sessions')",
        'speed_dic must be non-empty, or cell 18 silently produces nothing '
        '(distances=speed_dic[rd][ses] autovivifies, distances[start:end] raises '
        "TypeError: unhashable type: 'slice', and cell 18's except swallows it)"),
    ('Figure2.ipynb', '###Making phase, state and time arrays'): (
        "assert len(Phases_raw_dic) > 0, 'Phases_raw_dic EMPTY after cell 18'\n"
        "_rd = next(iter(Phases_raw_dic))\n"
        "_s  = next(iter(Phases_raw_dic[_rd]))\n"
        "import numpy as _np\n"
        "_p5 = _np.concatenate([_np.asarray(x) for t in Phases_raw_dic[_rd][_s] for x in t])\n"
        "_p3 = _np.concatenate([_np.asarray(x) for t in Phases_raw_dic2[_rd][_s] for x in t])\n"
        "assert len(_p5) == len(_p3), f'5-bin/3-bin length mismatch {len(_p5)} vs {len(_p3)}'\n"
        "assert set(_np.unique(_p5)) <= {0,1,2,3,4}, f'5-bin values {set(_np.unique(_p5))}'\n"
        "assert set(_np.unique(_p3)) <= {0,1,2},     f'3-bin values {set(_np.unique(_p3))}'\n"
        "assert set(_np.unique(_p3)) == {0,1,2}, '3-bin array is not using all 3 bins'\n"
        "print(f'GATE ok: {len(Phases_raw_dic)} recdays; {_rd} ses {_s}: '\n"
        "      f'{len(_p5)} bins, 5-bin={sorted(set(_np.unique(_p5)))}, "
        "3-bin={sorted(set(_np.unique(_p3)))}')",
        'UNBLOCK-F2-03: Phases_raw_dic2 must now be a genuine 3-bin array of the same '
        'length as the 5-bin Phases_raw_dic'),
    ('Figure2.ipynb', '###GLM - across tasks/states'): (
        "import numpy as _np\n"
        "_k = GLM_dic2['mean_neuron_betas']\n"
        "_bad = [r for r, v in _k.items() if not isinstance(v, _np.ndarray)]\n"
        "assert len(_k) > 0, 'GLM_dic2[mean_neuron_betas] is EMPTY -- cell 23 stored nothing'\n"
        "assert not _bad, f'{len(_bad)} recdays hold a defaultdict not an array: {_bad[:3]}'\n"
        "print(f'GATE ok: mean_neuron_betas populated for {len(_k)} recdays, '\n"
        "      f'first shape {next(iter(_k.values())).shape}')",
        'UNBLOCK-F2-05: cell 23 must actually STORE its betas. Its loads sit at the outer '
        'per-recday try, so any FileNotFoundError makes it print "betas not calculated" for '
        'every recday and leave the dict empty -- which silently wastes cell 25 (4 h) and '
        'only surfaces at cell 56 as "TypeError: unhashable type: slice"'),
    ('Figure2.ipynb', '###computing state tuning using per trial zscore'): (
        "assert 'Tuned_dic' in dir(), 'Tuned_dic undefined -- UNBLOCK-F2-07 did not take'\n"
        "assert len(Tuned_dic) > 0, 'Tuned_dic is EMPTY: cell 46 silently discarded its work'\n"
        "print(f'GATE ok: Tuned_dic keys={sorted(Tuned_dic.keys())}, '\n"
        "      f'State_zmax recdays={len(Tuned_dic[\"State_zmax\"])}')",
        'UNBLOCK-F2-07: Tuned_dic must exist AND be populated -- cell 46 wraps its three '
        'assignments in a try with a bare except that prints "Not found"'),
    ('Figure2.ipynb', 'phase_bool_all=[]'): (
        "assert use_both is True, f'use_both is {use_both}, expected True'\n"
        "assert len(Tuned_dic2) > 0, 'Tuned_dic2 EMPTY after cell 56'\n"
        "print(f'GATE ok: use_both={use_both}, use_permuted={use_permuted}, '\n"
        "      f'Tuned_dic2 keys={sorted(Tuned_dic2.keys())}')",
        'UNBLOCK-F2-08: use_both/use_permuted must be defined and cell 56 must populate '
        'Tuned_dic2, which cell 63 reads'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('notebook')
    ap.add_argument('--skip', default='', help='comma-separated cell indices to skip')
    ap.add_argument('--only', default='', help='cell range to run, e.g. 0-30')
    ap.add_argument('--allow-errors', action='store_true')
    ap.add_argument('--timeout', type=int, default=86400)
    ap.add_argument('--out', default='', help='output filename (default: same as input)')
    ap.add_argument('--kernel', default='mfc_replication',
                    help='jupyter kernelspec name (pinned numpy 1.22 env)')
    args = ap.parse_args()

    out_name = args.out or args.notebook
    os.makedirs(EXECUTED, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)

    # Pin BLAS/OpenMP threads for the kernel we are about to start.
    # Measured on this box: one PoissonRegressor fit on (30000, 324) takes 0.11 s
    # single-threaded vs 0.14 s with 4 threads -- at these feature counts the thread
    # coordination costs more than it saves, and an unpinned kernel spawns ~33 threads.
    # It is also the polite setting on a shared node (load average reached 32 on 8 cores
    # with three other users' multi-threaded jobs resident).
    # Override with UNBLOCK_THREADS=N if a future cell genuinely benefits.
    _nthreads = os.environ.get('UNBLOCK_THREADS', '1')
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(_v, _nthreads)

    import nbformat
    from nbclient import NotebookClient

    path = os.path.join(UNBLOCKED, args.notebook)
    nb = nbformat.read(path, as_version=4)

    skip = {int(x) for x in args.skip.split(',') if x.strip()}
    lo, hi = 0, len(nb.cells) - 1
    if args.only:
        a, _, b = args.only.partition('-')
        lo, hi = int(a), int(b or a)

    client = NotebookClient(nb, timeout=args.timeout, allow_errors=args.allow_errors,
                            kernel_name=args.kernel)
    t_all = time.time()
    ran, skipped, failed = [], [], []

    print(f'=== executing {args.notebook}: {len(nb.cells)} cells, '
          f'range {lo}..{hi}, skipping {sorted(skip) or "none"} ===', flush=True)

    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                if cell.cell_type == 'raw':
                    print(f'[{i:3d}] RAW  -- skipped by design '
                          f'({cell.get("metadata", {}).get("unblock", "")[:60]})', flush=True)
                    skipped.append(i)
                continue
            if not ''.join(cell.source).strip():
                continue
            if i in skip:
                print(f'[{i:3d}] SKIP (requested)', flush=True)
                skipped.append(i)
                continue
            if not (lo <= i <= hi):
                continue

            head = ''.join(cell.source).strip().split('\n')[0][:70]
            t0 = time.time()
            print(f'[{i:3d}] RUN  {head}', flush=True)
            try:
                client.execute_cell(cell, i)
            except Exception as e:
                dt = time.time() - t0
                print(f'[{i:3d}] FAIL after {dt:.1f}s: {type(e).__name__}: '
                      f'{str(e)[:400]}', flush=True)
                failed.append(i)
                nbformat.write(nb, os.path.join(EXECUTED, out_name))
                if not args.allow_errors:
                    print(f'\n=== ABORTED at cell {i} ===', flush=True)
                    return 1
            else:
                print(f'[{i:3d}] done {time.time() - t0:.1f}s', flush=True)
                ran.append(i)

            gate = None
            _src = ''.join(cell.source)
            for (_nb, _marker), _g in GATES.items():
                if _nb == args.notebook and _marker in _src:
                    gate = _g
                    break
            if gate:
                code, why = gate
                print(f'[{i:3d}] GATE {why[:90]}', flush=True)
                g = nbformat.v4.new_code_cell(source=code)
                # Append the gate cell and execute it AT ITS OWN INDEX, then pop it.
                # Passing the real cell's index would make nbclient assign the gate's
                # outputs to nb.cells[i], destroying that cell's stdout -- and cell 18's
                # recday listing is one of the checkpoints we score against the deposit.
                nb.cells.append(g)
                gi = len(nb.cells) - 1
                try:
                    client.execute_cell(g, gi)
                    for o in g.get('outputs', []):
                        for line in (o.get('text', '') or '').splitlines():
                            print(f'       {line}', flush=True)
                        if o.get('output_type') == 'error':
                            raise AssertionError('\n'.join(o.get('traceback', [])[-3:]))
                except Exception as e:
                    print(f'[{i:3d}] GATE FAILED: {str(e)[:600]}', flush=True)
                    if nb.cells and nb.cells[-1] is g:
                        nb.cells.pop()
                    nbformat.write(nb, os.path.join(EXECUTED, out_name))
                    return 2
                finally:
                    if nb.cells and nb.cells[-1] is g:
                        nb.cells.pop()

    nbformat.write(nb, os.path.join(EXECUTED, out_name))
    print(f'\n=== {args.notebook}: {len(ran)} cells run, {len(skipped)} skipped, '
          f'{len(failed)} failed, {time.time() - t_all:.1f}s total ===', flush=True)
    print(f'executed notebook -> {os.path.join(EXECUTED, out_name)}', flush=True)
    return 1 if failed and not args.allow_errors else 0


if __name__ == '__main__':
    sys.exit(main())
