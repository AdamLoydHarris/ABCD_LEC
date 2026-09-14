"""Batch driver for the registration QC (checks A-E, G) over the cohort.

Usage (histology env, from code/histology_refit/)::

    python run_registration_qc.py                       # all six, figures, table
    python run_registration_qc.py ah08 ly05 --no-figures
    python run_registration_qc.py --gate                # synthetic gate only

Each mouse is processed in turn and its caches cleared, so peak memory stays
around the one-slab level (~3-4 GB) plus the 4.3 GB transient of the gate's
``verify_contacts``.  Suitable for ``sbatch`` on a 16 GB allocation.
"""

from __future__ import annotations

import sys
import time
import argparse
import traceback

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, ".")
import pandas as pd

import registration_qc as rq


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mice", nargs="*", default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--gate", action="store_true", help="run the synthetic gate only")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)

    if args.gate:
        gate = rq.run_synthetic_controls("ah08")
        sys.exit(0 if gate.attrs["passed"] else 1)

    mice = args.mice or rq.pr.list_subjects()
    rows = []
    for m in mice:
        t = time.time()
        print(f"=== {m}", flush=True)
        try:
            g = rq.global_checks(m)
            l = rq.local_checks(m)
            row = {k: v for k, v in g.items() if not k.startswith("_")}
            row.update({k: v for k, v in l.items() if not k.startswith("_") and k != "subject"})
            rows.append(row)
            if not args.no_figures:
                rq.make_all_figures(m)
        except Exception as exc:
            print(f"!!! {m} failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        finally:
            rq.clear_caches(m)
        print(f"=== {m} done in {time.time() - t:.0f} s", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        out = rq.BRAINREG_DIR / "registration_qc.csv"
        df.to_csv(out, index=False)
        print(f"wrote {out}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
