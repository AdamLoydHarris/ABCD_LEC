"""Execute a notebook headless and save the executed copy. Fails on the first cell error.

Two timeouts matter and both must be set: the SLURM walltime, and the per-cell `--timeout`
here. A notebook that outlives its per-cell timeout raises; one that outlives the walltime is
killed with no traceback. There is deliberately no `--allow-errors`: under it nbclient reports
"0 failed" while writing error outputs into the cells, which is not evidence of anything.

    python code/run_notebook.py code/LEC_glm_v3.ipynb --timeout 1800
    TSA_DATA_VERSION=v1 python code/run_notebook.py code/LEC_glm_v3.ipynb   # the v1 figures
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('notebook')
    ap.add_argument('--timeout', type=int, default=1800, help='per-cell seconds')
    ap.add_argument('--out', default=None,
                    help='executed copy; default <notebook>.executed_<version>.ipynb')
    ap.add_argument('--kernel', default='python3')
    args = ap.parse_args()

    import nbformat
    from nbclient import NotebookClient

    nb_path = os.path.abspath(args.notebook)
    workdir = os.path.dirname(nb_path)          # notebooks resolve REPO as os.path.abspath('..')
    version = os.environ.get('TSA_DATA_VERSION') or ''
    tag = f'.executed{"_" + version if version else ""}.ipynb'
    out = args.out or nb_path.replace('.ipynb', tag)

    print(f'notebook : {nb_path}')
    print(f'workdir  : {workdir}')
    print(f'version  : {version or "<registry default>"}')
    print(f'per-cell timeout: {args.timeout} s', flush=True)

    nb = nbformat.read(nb_path, as_version=4)
    # the kernel inherits the environment: spatial.sbatch's MPLBACKEND=Agg left executed copies without figures
    os.environ.pop('MPLBACKEND', None)
    client = NotebookClient(nb, timeout=args.timeout, kernel_name=args.kernel,
                            resources={'metadata': {'path': workdir}},
                            allow_errors=False)
    t0 = time.time()
    try:
        client.execute()
    finally:
        nbformat.write(nb, out)
        print(f'\nexecuted copy -> {out}')
    n_err = sum(1 for c in nb.cells for o in c.get('outputs', [])
                if o.get('output_type') == 'error')
    print(f'cells: {len(nb.cells)}   error outputs: {n_err}   '
          f'elapsed {(time.time() - t0) / 60:.1f} min')
    return 1 if n_err else 0


if __name__ == '__main__':
    sys.exit(main())
