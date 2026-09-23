"""
Regression survey of the frame clock: alignment fitter (frame_clock.py) vs the superseded
global-line fitter, on every session the matched-clock QC lists for a dataset.

Per session: both fitters' matched count, jumps, drop total, local residual, and the impact --
the largest and the median absolute difference (frames) between the two clocks over the
session's 25 ms bins, plus the same for the plain first-edge 60 fps line. Written to
`data/processed_data/frame_clock_survey_{dataset}.csv` (versioned, never overwritten).

Usage:  python frame_clock_survey.py --dataset lec
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
import sleap_alignment_copilot as cp                                  # noqa: E402
import frame_clock as fc                                              # noqa: E402

QC_DIRS = {"lec": ["processed_v2", "processed_v2_globalline_2026-09-18"],
           "mec": ["processed", "processed_globalline_2026-09-18"]}


def sessions() -> pd.DataFrame:
    frames = []
    for d in QC_DIRS[rr.DATASET]:
        for f in sorted((rr.REPO_ROOT / "data" / d).glob(f"sync_qc_{rr.DATASET}*.csv")):
            frames.append(pd.read_csv(f)[["mouse", "recday", "session", "pinstate"]])
    q = pd.concat(frames).drop_duplicates(["recday", "session"], keep="last")
    return q.reset_index(drop=True)


def pinstate_path(name: str) -> Path | None:
    hits = glob.glob(str(rr.REPO_ROOT / "data" / "raw" / "**" / name), recursive=True)
    return Path(hits[0]) if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    q = sessions()
    if args.limit:
        q = q.head(args.limit)
    rows = []
    for _, s in q.iterrows():
        t0 = time.time()
        row = dict(mouse=s.mouse, recday=s.recday, session=int(s.session), pinstate=s.pinstate)
        try:
            pin = pinstate_path(s.pinstate)
            pc = rr.pycontrol_file(s.recday, int(s.session))
            up = cp.rising_edges(cp.load_pinstate(pin)).astype(float)
            rs = cp.all_rsync_timestamps(cp.parse_pycontrol_file(pc))
        except Exception as exc:                                       # noqa: BLE001
            row["error"] = f"inputs: {exc}"
            rows.append(row)
            print(f"{s.recday} s{s.session}: inputs failed: {exc}")
            continue
        row.update(n_rsync=len(rs), n_edges=len(up))
        new = old = None
        try:
            new = fc.fit_frame_clock(up, rs)
            row.update({f"new_{k}": v for k, v in new.items() if not k.startswith("anchors")})
        except ValueError as exc:
            row["new_error"] = str(exc)
        try:
            old = cp.fit_frame_clock_globalline(up, rs)
            row.update({f"old_{k}": v for k, v in old.items() if not k.startswith("anchors")})
        except ValueError as exc:
            row["old_error"] = str(exc)
        if new is not None:
            t = np.arange(rs[0], rs[-1], 25.0)
            fn = fc.frames_at(new, t)
            first_edge_line = up[0] + (t - rs[0]) * 60.0 / 1000.0     # the legacy first-edge, 60 fps clock
            row["firstedge_vs_new_max_frames"] = float(np.abs(first_edge_line - fn).max())
            row["firstedge_vs_new_median_frames"] = float(np.median(np.abs(first_edge_line - fn)))
            if old is not None:
                fo = cp.frames_at(old, t)
                row["old_vs_new_max_frames"] = float(np.abs(fo - fn).max())
                row["old_vs_new_median_frames"] = float(np.median(np.abs(fo - fn)))
                row["old_vs_new_frac_bins_gt2"] = float((np.abs(fo - fn) > 2).mean())
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"{s.recday} s{s.session}: new {row.get('new_n_matched')}/{len(rs)} drop {row.get('new_total_drop_frames', float('nan')):.0f} fr"
              f" | old {row.get('old_n_matched')} | old-new max {row.get('old_vs_new_max_frames', float('nan')):.1f} fr"
              f" | firstedge-new max {row.get('firstedge_vs_new_max_frames', float('nan')):.1f} fr ({row['seconds']} s)")
    out = rr.REPO_ROOT / "data" / "processed_data" / f"frame_clock_survey_{rr.DATASET}.csv"
    n = 0
    while (p := out.with_name(out.stem + (f".v{n}" if n else "") + ".csv")).exists():
        n += 1
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
