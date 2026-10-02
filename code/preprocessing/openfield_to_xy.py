"""
Open-field / object sessions: SLEAP h5 -> cleaned coordinates (cm) and head direction CSVs.

The open-field twin of `sleap_to_rois.py`: same cleaning (jump rejection at median + k MAD,
linear interpolation both ways, centred 5-frame rolling mean), but the pixel->cm step is the
arena calibration of `arena_calibration.py` (radial correction + homography for the mEC big
field; for the LEC box a per-session homography from that session's clicked corners to the
50 cm square, `session_calibration`, since the box shifts between sessions) and there are no
ROI masks, so no `*_ROIs.csv`.
Outputs, read by `sleap_alignment_copilot --sessions openfield`:

    data/preprocessed/sleap/openfield/{mouse}/{mouse}_{YYYY-MM-DD-HHMMSS}.analysis_cleaned_coordinates.csv
    data/preprocessed/sleap/openfield/{mouse}/{mouse}_{YYYY-MM-DD-HHMMSS}.analysis_head_direction.csv

Which sessions: every sheet row whose `recday_registry.session_type` is 'object' or 'openfield'
on a sorted block, matched to an h5 by the Tracking stamp (exact, else within 2 min). The
notebook used a jump multiplier of 4 for the open field (3 for the maze); kept.

Run with the `histology` env (cv2). Existing outputs are never overwritten.

`--clean v2` (2026-09-23) runs `tracking_clean.clean_v2` instead of the notebook cleaning (same
rules and constants as the maze; the 4-MAD multiplier is a v1 constant), writes to
`openfield_v2/` with a per-frame `*_track_flags.csv`, and takes head direction from the
cleaned track. `--compare-existing` recomputes `--clean v1` in memory and compares it with
`openfield/` (bytes, max |diff|, NaN pattern); exit 1 on any difference, nothing written.

Usage::
    python openfield_to_xy.py --dataset mec [--write] [--mouse bp01]
    python openfield_to_xy.py --dataset {mec,lec} --clean v1 --compare-existing
    python openfield_to_xy.py --dataset {mec,lec} --clean v2 --write
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
import arena_calibration as ac                                        # noqa: E402
from sleap_to_rois import load_tracks, drop_jumps, interpolate_smooth, head_direction, compare_outputs  # noqa: E402
import tracking_clean as tclean                                       # noqa: E402

OUT_ROOTS = {"v1": rr.REPO_ROOT / "data" / "preprocessed" / "sleap" / "openfield",
             "v2": rr.REPO_ROOT / "data" / "preprocessed" / "sleap" / "openfield_v2"}
OUT_ROOT = OUT_ROOTS["v1"]
CLEAN_VERSION = "v1"
RAW_DIR = {"mec": "openfield", "lec": "objects"}[rr.DATASET]
JUMP_MULTIPLIER = 4.0                # v1 only
LEC_CORNERS = rr.REPO_ROOT / "data" / "processed_data" / "arena_corners_lec.csv"

#: which arena calibration a mouse's open-field sessions use
def arena_for(mouse: str) -> str:
    if rr.DATASET == "mec":
        return "bigfield"
    return "openfield_box2" if mouse in ("ah08", "ly06") else "openfield_box1"


def session_calibration(base: ac.ArenaCalibration, mouse: str, stamp: str) -> tuple[ac.ArenaCalibration, str]:
    """The calibration for one session and the name of its frame.

    mEC: the base (the big field does not move; frame 'arena'). LEC: the box shifts between
    sessions and the maze affine shows it as a 4 % trapezoid (bottom side 51.2 cm, top 49.1 cm
    on all 20 sessions, camera tilt, 2026-09-18), so each session gets its own homography from
    its four clicked corners (`object_click.ipynb` -> arena_corners_lec.csv) to the 50 cm square:
    frame 'box', origin at the image-top-left corner, x right, y down. Without corners for the
    session the affine is used and the frame is 'maze', flagged in the QC.
    """
    if rr.DATASET != "lec":
        return base, "arena"
    if not LEC_CORNERS.exists():
        return base, "maze (no corners file)"
    d = pd.read_csv(LEC_CORNERS)
    if "stamp" not in d.columns:
        return base, "maze (per-box corner file, superseded)"
    c = d[(d["mouse"] == mouse) & (d["stamp"] == stamp)].sort_values("corner")
    if len(c) != 4:
        return base, f"maze (no corners for this session: {len(c)} rows)"
    return ac.apply_arena_corners_homography(base, c[["x_px", "y_px"]].to_numpy(float), ac.LEC_BOX_SIDE_CM), "box"


def find_h5(mouse: str, stamp: str) -> Path | None:
    """h5 for `{mouse}_{YYYY-MM-DD-HHMMSS}`; exact stamp first, else the unique one within 2 min."""
    d = rr.REPO_ROOT / "data" / "raw" / "sleap" / mouse / RAW_DIR
    exact = sorted(d.glob(f"*_{mouse}_{stamp}.analysis.h5"))
    if exact:
        return exact[0]
    want = datetime.strptime(stamp, "%Y-%m-%d-%H%M%S")
    near = []
    for p in d.glob(f"*_{mouse}_*.analysis.h5"):
        m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{6})", p.name)
        if m and abs((datetime.strptime(m.group(1), "%Y-%m-%d-%H%M%S") - want).total_seconds()) <= 120:
            near.append(p)
    return near[0] if len(near) == 1 else None


def openfield_sessions(mouse: str | None = None) -> list[tuple[str, int, str, str, str]]:
    """(recday, session, mouse, stamp, session_type) over merged recdays (single-day ones reuse the CSVs)."""
    out = []
    for recday in rr.all_recdays("merged"):
        m, _ = rr.recday_dates(recday)
        if mouse and m != mouse:
            continue
        rows = rr.task_session_rows(recday)
        for s in range(len(rows)):
            kind = rr.session_type(recday, s)
            if kind not in ("object", "openfield"):
                continue
            row = rows.iloc[s]
            tr = row["Tracking"]
            if pd.isna(tr) or str(tr).strip() in ("-", ""):
                continue
            d = str(row["_date"])
            stamp = f"{d[:4]}-{d[4:6]}-{d[6:]}-{str(int(float(tr))).zfill(6)}"
            out.append((recday, s, m, stamp, kind))
    return out


def process_one(h5: Path, mouse: str, stamp: str, cal: ac.ArenaCalibration, write: bool,
                compare_dir: Path | None = None) -> dict:
    raw, nodes = load_tracks(h5)
    flags, v2_report = None, None
    if CLEAN_VERSION == "v1":
        cleaned = interpolate_smooth(drop_jumps(raw, nodes, multiplier=JUMP_MULTIPLIER), nodes)
        cm = cleaned.copy()
        for bp in nodes:
            xy = cal.to_cm(cleaned[[f"{bp}.x", f"{bp}.y"]].to_numpy())
            cm[f"{bp}.x"], cm[f"{bp}.y"] = xy[:, 0], xy[:, 1]
    else:
        scores = tclean.load_scores(h5)
        cm, flags, v2_report = tclean.clean_v2(raw, nodes, scores, cal.to_cm)
    hd = head_direction(cm)
    out_dir = OUT_ROOT / mouse
    if write and CLEAN_VERSION != "v1" and OUT_ROOTS["v1"].resolve() in (out_dir.resolve(), *out_dir.resolve().parents):
        raise ValueError(f"refusing to write --clean {CLEAN_VERSION} output into the v1 tree: {out_dir}")
    names = {f"{mouse}_{stamp}.analysis_cleaned_coordinates.csv": cm,
             f"{mouse}_{stamp}.analysis_head_direction.csv": hd}
    if flags is not None:
        names[f"{mouse}_{stamp}.analysis_track_flags.csv"] = flags
    existing = [n for n in names if (out_dir / n).exists()]
    if write and not existing:
        out_dir.mkdir(parents=True, exist_ok=True)
        for n, frame in names.items():
            frame.to_csv(out_dir / n, index=False)
    hb = cm[["head_back.x", "head_back.y"]].to_numpy()
    walls = ac.straight_wall_test(hb)
    row = dict(n_frames=len(raw), raw_nan_frac=float(raw["head_back.x"].isna().mean()),
               x_min=float(np.nanmin(hb[:, 0])), x_max=float(np.nanmax(hb[:, 0])),
               y_min=float(np.nanmin(hb[:, 1])), y_max=float(np.nanmax(hb[:, 1])),
               wall_rms_cm={k: (None if v is None else round(v["rms_cm"], 2)) for k, v in walls.items()},
               clean_version=CLEAN_VERSION,
               status="exists_skipped" if existing else ("written" if write else "dry_run"))
    if v2_report is not None:
        row.update(tclean.report_row(v2_report))
        row["v2_hd_nan_frac"] = float(hd["back2mid_deg"].isna().mean())
    if compare_dir is not None:
        row.update(compare_outputs(names, compare_dir))
        row["status"] = "compared"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--mouse", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--clean", choices=("v1", "v2"), default="v1",
                    help="v1: notebook cleaning, openfield/ (default); v2: tracking_clean -> openfield_v2/")
    ap.add_argument("--compare-existing", action="store_true",
                    help="recompute in memory and compare with openfield/ (v1) or openfield_v2/ (v2); exit 1 on difference")
    args = ap.parse_args()

    global CLEAN_VERSION, OUT_ROOT
    CLEAN_VERSION = args.clean
    OUT_ROOT = OUT_ROOTS[CLEAN_VERSION]
    compare_root = OUT_ROOT if args.compare_existing else None
    if args.compare_existing and args.write:
        sys.exit("--compare-existing never writes; drop --write")
    print(f"clean {CLEAN_VERSION} -> {OUT_ROOT}" + (f" (comparing against it, nothing written)" if compare_root else ""))

    cals = {}
    rows = []
    seen = set()
    for recday, s, mouse, stamp, kind in openfield_sessions(args.mouse):
        if (mouse, stamp) in seen:
            continue
        seen.add((mouse, stamp))
        arena = arena_for(mouse)
        if arena not in cals:
            cals[arena] = ac.latest_calibration(rr.REPO_ROOT, rr.DATASET, arena)
            print(f"calibration {arena}: plane_factor={cals[arena].plane_factor} "
                  f"reproj {cals[arena].reprojection_rms_px:.2f} px")
        h5 = find_h5(mouse, stamp)
        if h5 is None:
            rows.append(dict(recday=recday, session=s, mouse=mouse, stamp=stamp, kind=kind, status="no h5"))
            print(f"{recday} s{s} {kind} {stamp}: NO SLEAP h5")
            continue
        cal, frame = session_calibration(cals[arena], mouse, stamp)
        r = process_one(h5, mouse, stamp, cal, args.write, compare_dir=(compare_root / mouse) if compare_root else None)
        r.update(recday=recday, session=s, mouse=mouse, stamp=stamp, kind=kind, h5=h5.name, arena=arena, frame=frame,
                 side_cm=cal.arena_side_cm)
        rows.append(r)
        extra = ""
        if compare_root:
            extra = (f"bytes_equal={r['cmp_bytes_equal']} max|diff|={r['cmp_max_abs_diff']:.2e} "
                     f"nan_mismatch={r['cmp_nan_mismatch']} missing={r['cmp_missing']}")
        elif CLEAN_VERSION == "v2":
            extra = f"v2 removed={r['v2_removed_by_rules']} bridged={r['v2_bridged']} leftNaN={r['v2_left_nan']} hdNaN={r['v2_hd_nan_frac']:.3f}"
        print(f"{r['status']:15s} {recday} s{s} {kind:9s} {stamp} [{frame}] frames={r['n_frames']} "
              f"x[{r['x_min']:.0f},{r['x_max']:.0f}] y[{r['y_min']:.0f},{r['y_max']:.0f}] cm "
              f"walls rms cm {r['wall_rms_cm']} {extra}", flush=True)
    table = pd.DataFrame(rows)
    if compare_root is not None and len(table):
        done = table[table.status == "compared"]
        ok = done["cmp_bytes_equal"].astype(bool)
        zero = (done["cmp_max_abs_diff"] == 0) & (done["cmp_nan_mismatch"] == 0) & (done["cmp_missing"] == 0)
        print(f"\nEQUIVALENCE: {int(ok.sum())}/{len(done)} byte-identical; {int((zero & ~ok).sum())} differ in bytes only; "
              f"{int((~zero).sum())} differ in values; {int((table.status == 'no h5').sum())} without h5")
        bad = done[~zero]
        if len(bad):
            print(bad[["mouse", "stamp", "cmp_max_abs_diff", "cmp_nan_mismatch", "cmp_missing"]].to_string(index=False))
        sys.exit(0 if len(bad) == 0 else 1)
    if args.write and len(table):
        qc = OUT_ROOT.parent / f"openfield_to_xy_qc_{rr.DATASET}{'' if CLEAN_VERSION == 'v1' else '_v2'}.csv"
        n = 0
        while (p := qc.with_name(qc.stem + (f".v{n}" if n else "") + ".csv")).exists():
            n += 1
        table.to_csv(p, index=False)
        print(f"QC -> {p}")


if __name__ == "__main__":
    main()
