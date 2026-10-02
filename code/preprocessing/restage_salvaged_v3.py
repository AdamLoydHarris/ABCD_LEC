"""
Re-stage the salvaged session ah10_20250618_20250619 s5 on the cleaning-v2 CSVs, into processed_v3.

That video's pinstate has 0 rising edges, so the copilot cannot fit a frame clock; its arrays were
salvaged on 2026-09-06 from poke-entry / tracked-head_back agreement (first rsync frame 2244, first
A_on frame 2556, 60 fps arithmetic; `salvage_tracking_offset.py`, record
`data/processed_data/salvage_staging/salvage_ah10_20250618_20250619_5.json`) and copied unchanged
into processed_v2. Under cleaning v2 the CSVs change (`maze_v2/`), so the arrays are rebuilt here
with the salvage's own arithmetic (`salvage_tracking_offset.build_tracking_arrays`, the pipeline's
60 fps resample) from the v2 CSVs, plus the v2 NaN propagation (a bin whose bracketing frames are
NaN is NaN, `sleap_alignment_copilot.untracked_bins`). Written beside a sidecar json; nothing is
overwritten (an existing array is left alone and reported).

    python restage_salvaged_v3.py [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
import sleap_alignment_copilot as sac                                 # noqa: E402
import salvage_tracking_offset as sto                                 # noqa: E402

RECDAY, SESSION, MOUSE, STAMP = "ah10_20250618_20250619", 5, "ah10", "2025-06-19-143522"
RECORD = rr.REPO_ROOT / "data" / "processed_data" / "salvage_staging" / f"salvage_{RECDAY}_{SESSION}.json"
SRC = rr.REPO_ROOT / "data" / "preprocessed" / "sleap" / "maze_v2" / MOUSE
DST = rr.REPO_ROOT / "data" / "processed_v3" / MOUSE / "20250618_20250619"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if rr.DATASET != "lec":
        sys.exit("LEC only")
    rec = json.loads(RECORD.read_text())
    f_a = int(rec["first_A_on_frame"])
    assert f_a == sto.first_A_on_frame_pipeline(rec["estimate"]["first_rsync_frame"], rec["first_A_on_ms"], rec["first_rsync_ms"])
    files = {"coordinates": SRC / f"{MOUSE}_{STAMP}.analysis_cleaned_coordinates.csv",
             "head_direction": SRC / f"{MOUSE}_{STAMP}.analysis_head_direction.csv",
             "rois": SRC / f"{MOUSE}_{STAMP}.analysis_ROIs.csv"}
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        sys.exit(f"maze_v2 CSVs missing for {STAMP}: {missing} (run sleap_to_rois.py --dataset lec --clean v2 --write first)")
    sleap = {k: pd.read_csv(p) for k, p in files.items()}
    arrays = sto.build_tracking_arrays(sleap, f_a)
    coords = sleap["coordinates"][["head_back.x", "head_back.y"]].to_numpy(float)
    n = len(arrays["XY"])
    frames = f_a + np.arange(n) * (sac.VIDEO_FPS / sac.TARGET_FPS)          # the 60 -> 40 fps resample grid
    untracked = sac.untracked_bins(coords[: len(coords)], frames)
    arrays["XY"] = arrays["XY"].astype(float)
    arrays["XY"][untracked] = np.nan
    hd = arrays["HD"].astype(float)
    hd_frames = np.clip(np.rint(frames).astype(int), 0, len(sleap["head_direction"]) - 1)
    arrays["HD"] = sleap["head_direction"].to_numpy(float)[hd_frames]          # nearest, carries the CSV NaN
    names = sto.output_names(RECDAY, SESSION)
    print(f"{RECDAY} s{SESSION}: first A_on frame {f_a}, {n} bins; XY NaN {np.isnan(arrays['XY']).any(1).mean():.4f}, "
          f"HD NaN {np.isnan(arrays['HD'][:, 0]).mean():.4f}, Locs None {np.mean(arrays['Locs'].astype(str) == 'None'):.4f}")
    side = dict(session=f"{RECDAY} s{SESSION} (video {STAMP})", restaged=datetime.now().strftime("%Y-%m-%d"),
                source="data/preprocessed/sleap/maze_v2/ (cleaning v2) through salvage_tracking_offset.build_tracking_arrays",
                first_rsync_frame=rec["estimate"]["first_rsync_frame"], first_A_on_frame=f_a,
                clock="poke-estimated first rsync frame + 60 fps arithmetic (as the 2026-09-06 salvage), NOT a fitted clock",
                nan_propagation="bins whose bracketing frames are NaN in the v2 CSV are NaN (sleap_alignment_copilot.untracked_bins)",
                n_bins=int(n), xy_nan_frac=float(np.isnan(arrays["XY"]).any(1).mean()))
    if not args.write:
        print("dry run; pass --write to stage into", DST)
        return
    DST.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        p = DST / names[key]
        if p.exists():
            print(f"  exists, left alone: {p.name}")
            continue
        np.save(p, arr)
        print(f"  wrote {p.name} {arr.shape}")
    (DST / f"RESTAGED_V3_{RECDAY}_{SESSION}.json").write_text(json.dumps(side, indent=1))
    print("  sidecar written")


if __name__ == "__main__":
    main()
