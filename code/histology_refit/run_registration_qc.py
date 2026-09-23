"""Batch driver for the registration QC (checks A-E, G) over the cohort.

Usage (histology env, from code/histology_refit/)::

    python run_registration_qc.py                       # LEC: all six, figures, table
    python run_registration_qc.py ah08 ly05 --no-figures
    python run_registration_qc.py --gate                # synthetic gate only
    python run_registration_qc.py --dataset mec --global-only      # mEC brains, registration only
    python run_registration_qc.py --dataset lec --global-only ah08 # equivalence control for the mEC changes

`--dataset mec` points probe_refit / registration_qc at the mEC tree (TSA_BRAINREG_DIR,
TSA_HIST_DATASET) before they are imported. `--global-only` runs the checks that need no probe
fit (B, C over the whole brain, region tissue coverage, the void report) and the overview figures
(A overlays, atlas-space checker, L/R profile, mEC zoom); local checks D/E/G and the laminar and
slab-Jacobian figures are skipped. The table is written versioned (`registration_qc[_global][.vN].csv`).

Each mouse is processed in turn and its caches cleared, so peak memory stays
around the one-slab level (~3-4 GB) plus the 4.3 GB transient of the gate's
``verify_contacts``.  Suitable for ``sbatch`` on a 16-32 GB allocation.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--dataset", default="lec", choices=("lec", "mec"))
_args, _ = _pre.parse_known_args()
if _args.dataset == "mec":
    os.environ["TSA_BRAINREG_DIR"] = "/ceph/behrens/adam_harris/Taskspace_abstraction_mEC/data/preprocessed_data/brainreg"
    os.environ["TSA_HIST_DATASET"] = "mec"
    os.environ.setdefault("BRAINGLOBE_CONFIG_DIR", "/ceph/apps/application_data/brainglobe")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import registration_qc as rq


def _versioned(path: Path) -> Path:
    n = 0
    while (p := path.with_name(path.stem + (f".v{n}" if n else "") + path.suffix)).exists():
        n += 1
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mice", nargs="*", default=None)
    ap.add_argument("--dataset", default="lec", choices=("lec", "mec"))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--gate", action="store_true", help="run the synthetic gate only")
    ap.add_argument("--global-only", action="store_true", help="checks that need no probe fit; overview figures only")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    print(f"dataset {args.dataset}: brainreg dir {rq.BRAINREG_DIR}; figures {rq.FIGURE_DIR}", flush=True)

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
            row = {k: v for k, v in g.items() if not k.startswith("_")}
            if args.global_only:
                cov = rq.region_tissue_coverage(m)
                for _, r in cov.iterrows():
                    row[f"cov_{r['group']}_left"] = r["left_coverage"]
                    row[f"cov_{r['group']}_right"] = r["right_coverage"]
                row.update(rq.global_jacobian(m))
                print(cov.round(3).to_string(index=False), flush=True)
            else:
                l = rq.local_checks(m)
                row.update({k: v for k, v in l.items() if not k.startswith("_") and k != "subject"})
            rows.append(row)
            if not args.no_figures:
                import matplotlib.pyplot as plt
                if args.global_only:
                    for fn in (rq.plot_sample_overlays, rq.plot_atlas_space_checker, rq.plot_lr_profile):
                        plt.close(fn(m))
                    if rq.DATASET == "mec":
                        plt.close(rq.plot_mec_zoom(m))
                else:
                    rq.make_all_figures(m)
        except Exception as exc:
            print(f"!!! {m} failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        finally:
            rq.clear_caches(m)
        print(f"=== {m} done in {time.time() - t:.0f} s", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        out = _versioned(rq.BRAINREG_DIR / ("registration_qc_global.csv" if args.global_only else "registration_qc.csv"))
        df.to_csv(out, index=False)
        print(f"wrote {out}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
