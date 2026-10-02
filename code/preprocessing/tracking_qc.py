"""
Tracking QC: one row per SLEAP session of a dataset, every arena, from the raw h5.

Stage T1 of `docs/handoff/TRACKING_QC_PLAN.md` (revised 2026-09-23). Reads the raw SLEAP h5
(tracks + point/instance scores) for every maze and arena session, puts head_back in cm with
the session's geometry (`sleap_to_rois.geometry` + `affine` under --geometry v2 for the maze;
`openfield_to_xy.session_calibration` for the 50 cm LEC box and the 120 cm mEC big field),
and measures, per session:

- scores: point-score percentiles and low-score fractions per node, instance-score fractions;
  the fraction of low-score frames that sit OFF the trajectory of the good frames around them
  (> 3 cm from the chord between the flanking good frames) -- the number that decides whether
  a hard gate or the conditional gate is right for each SLEAP model;
- raw NaN runs (n, longest, runs > 2 s, frames inside them);
- motion: frame-to-frame displacement percentiles in cm/s, speed on a smooth-only track;
- the current v1 rule (`drop_jumps`, median + 3 MAD; 4 MAD in the arenas): fraction of frames
  it drops, split into moving (> 5 cm/s) and still frames; speed p90 and path length after v1;
- the v2 rule (`tracking_clean.clean_v2`, dry): removals per reason, bridged, left NaN, moving
  frames removed, speed p90 and path length after v2, R1 runs longer than 5 frames;
- bridged frames off the maze skeleton (beyond the kept frames' p99 distance to an edge
  centreline; node centres from the mask centroids at 5 / 23.5 / 42 cm);
- stuck plateaus, head_back-body_back length, out-of-arena, coverage of the 9 nodes and
  bounding-box span (a collapsed affine shows here), flags.

Outputs (never overwritten, `.vN`): `data/processed_data/tracking_qc_{dataset}.csv` in the
dataset's tree. `--gate` runs the planted synthetic h5 through the same function and checks the
counts exactly. Maze rows need only numpy/pandas/h5py (maze_ephys env); arena rows need cv2
(histology env) -- pick with --arena.

    python tracking_qc.py --gate
    python tracking_qc.py --dataset mec --arena maze --write          # sbatch mec_stage
    python tracking_qc.py --dataset lec --arena arena --write         # sbatch histology_stage
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
import sleap_to_rois as s2r                                           # noqa: E402
import tracking_clean as tc                                           # noqa: E402

PROCESSED_DATA = rr.REPO_ROOT / "data" / "processed_data"
FPS = 60.0
LONG_GAP = 120                       # 2 s
MOVING_CM_S = 5.0
ARENA_SIDE = {"maze": 47.0, "box": 50.0, "bigfield": 120.0}
V1_MULTIPLIER = {"maze": 3.0, "box": 4.0, "bigfield": 4.0}
#: Node centres in cm from the node-mask centroids under every good geometry (2026-09-23):
#: 5 / 23.5 / 42 cm, not 0 / 23.5 / 47 (PHYSICAL_PTS are the outer corners of the corner nodes).
MAZE_NODE_GRID = np.array([5.0, 23.5, 42.0])
MAZE_NODES = np.array([[x, y] for y in MAZE_NODE_GRID for x in MAZE_NODE_GRID])
MAZE_EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]
MAZE_SEGS = np.array([[MAZE_NODES[a], MAZE_NODES[b]] for a, b in MAZE_EDGES])
NODE_RADIUS_CM = 6.0                 # 'at a node' for coverage

FLAG_RULES = {
    "false_drop": ("v1_drop_moving_frac", ">", 0.20),
    "low_conf": ("inst_lt05", ">", 0.02),
    "long_gap": ("raw_gap_gt2s", ">", 0),
    "body_len": ("body_len_implausible_frac", ">", 0.01),
    "out_of_arena": ("out_of_arena_frac", ">", 0.01),
    # the Z-order maze-2 affine gave span_min_frac 0.03; a mouse that never left one corner in a
    # no-trial session gives 0.1-0.6 (coverage), which is behaviour, not geometry
    "degenerate_geometry": ("span_min_frac", "<", 0.10),
    "low_span": ("span_min_frac", "<", 0.60),
    "whole_session_nan": ("raw_nan_frac", ">", 0.90),
    "short": ("n_frames", "<", 3000),
}


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def dist_to_skeleton(pts: np.ndarray) -> np.ndarray:
    """Distance (cm) of each point to the nearest maze edge centreline segment; NaN in -> NaN out."""
    out = np.full(len(pts), np.nan)
    ok = ~np.isnan(pts).any(axis=1)
    if not ok.any():
        return out
    A = MAZE_SEGS[:, 0][None]
    B = MAZE_SEGS[:, 1][None]
    P = pts[ok][:, None, :]
    AB = B - A
    t = np.clip(((P - A) * AB).sum(-1) / (AB * AB).sum(-1), 0, 1)
    proj = A + t[..., None] * AB
    out[ok] = np.linalg.norm(P - proj, axis=-1).min(axis=1)
    return out


def maze_to_cm(corners: np.ndarray, check: bool = True):
    A, b = s2r.affine(corners, s2r.PHYSICAL_PTS, check=check)
    return lambda px: (A @ np.asarray(px, float).T).T + b


def path_length(xy: np.ndarray) -> float:
    d = np.diff(xy, axis=0)
    step = np.hypot(d[:, 0], d[:, 1])
    return float(np.nansum(step))


def consecutive_speed(xy: np.ndarray) -> np.ndarray:
    """cm/s between consecutive frames; NaN where either frame is NaN. Length T (last = NaN)."""
    d = np.diff(xy, axis=0)
    v = np.hypot(d[:, 0], d[:, 1]) * FPS
    return np.r_[v, np.nan]


def v1_threshold_px(raw: pd.DataFrame, first_node: str, multiplier: float) -> float:
    """The v1 jump threshold as drop_jumps computes it (first bodypart, px/frame)."""
    x, y = raw[f"{first_node}.x"].to_numpy(), raw[f"{first_node}.y"].to_numpy()
    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    good = dist[~np.isnan(dist)]
    if len(good) == 0:
        return float("nan")
    med = np.median(good)
    mad = np.median(np.abs(good - med))
    return float(med * multiplier if mad == 0 else med + multiplier * mad)


def runs_longer_than(mask: np.ndarray, n: int) -> int:
    s, e = tc.gap_runs(mask)
    return int(((e - s) > n).sum())


def pct(a: np.ndarray, q) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


# ---------------------------------------------------------------------------
# one session
# ---------------------------------------------------------------------------

def qc_session(h5: Path, to_cm, arena: str, side: float, base: dict) -> dict:
    """All QC metrics of one h5. `to_cm(px (T,2)) -> cm (T,2)` NaN-propagating."""
    row = dict(base)
    raw, nodes = s2r.load_tracks(h5)
    scores = tc.load_scores(h5)
    with h5py.File(h5, "r") as f:
        lp = f["labels_path"][()] if "labels_path" in f else b""
    lp = lp.decode() if isinstance(lp, bytes) else str(lp)
    row["model"] = Path(lp.replace("\\", "/")).name if lp else ""
    T = len(raw)
    row["n_frames"] = T
    row["duration_s"] = round(T / FPS, 1)
    ps, isc = scores["point_scores"], scores["instance_scores"]
    hb = nodes.index("head_back")
    xy_px = raw[["head_back.x", "head_back.y"]].to_numpy(float)
    xy = to_cm(xy_px)
    fin = ~np.isnan(xy).any(axis=1)
    row["raw_nan_frac"] = float((~fin).mean())

    # scores
    p = ps[hb][fin]
    row.update(pt_p1=pct(p, 1), pt_p5=pct(p, 5), pt_p50=pct(p, 50),
               pt_lt03=float(np.mean(p < 0.3)) if len(p) else np.nan,
               pt_lt05=float(np.mean(p < 0.5)) if len(p) else np.nan,
               pt_lt07=float(np.mean(p < 0.7)) if len(p) else np.nan)
    for k, bp in enumerate(nodes):
        pk = ps[k][~np.isnan(raw[f"{bp}.x"].to_numpy())]
        row[f"pt_lt05_{bp}"] = float(np.mean(pk < 0.5)) if len(pk) else np.nan
    i = isc[fin]
    row.update(inst_p1=pct(i, 1), inst_p5=pct(i, 5), inst_p50=pct(i, 50),
               inst_lt03=float(np.mean(i < 0.3)) if len(i) else np.nan,
               inst_lt05=float(np.mean(i < 0.5)) if len(i) else np.nan,
               inst_lt07=float(np.mean(i < 0.7)) if len(i) else np.nan)

    # raw gaps
    g = tc.describe_gaps(~fin, LONG_GAP)
    row.update(raw_gap_n=g["n_runs"], raw_gap_longest_s=round(g["longest"] / FPS, 2), raw_gap_gt15=g["n_gt15"],
               raw_gap_gt2s=g["n_gt_long"], raw_frames_in_gt2s_frac=g["frames_in_long"] / T)

    # low-score frames vs the trajectory of the good frames around them
    low = ((ps[hb] < tc.V2["point_min"]) | (isc < tc.V2["instance_min"])) & fin
    dv = tc.lowscore_deviation(xy, low)
    dev = dv["dev"]
    testable = np.isfinite(dev)
    gated = tc.lowscore_gate(xy, low, tc.V2["lowscore_thr_cm"], tc.V2["lowscore_chord_frac"],
                             tc.V2["lowscore_chord_frac_long"], tc.V2["lowscore_long_run"], tc.V2["lowscore_path_frac"])
    ls, le = tc.gap_runs(low)
    row.update(low_frac=float(low.mean()), low_testable_n=int(testable.sum()), low_runs_n=int(len(ls)),
               low_in_long_runs_frac=float((dv["run_len"][testable] > tc.V2["lowscore_long_run"]).mean()) if testable.any() else np.nan,
               low_dev_p50=pct(dev, 50), low_dev_p90=pct(dev, 90),
               low_off3_frac=float(np.mean(dev[testable] > 3.0)) if testable.any() else np.nan,
               low_off_tol_frac=float(gated[testable].mean()) if testable.any() else np.nan)

    # motion on the raw track and on a smooth-only reference
    v_raw = consecutive_speed(xy)
    row.update(disp_p50=pct(v_raw, 50), disp_p90=pct(v_raw, 90), disp_p99=pct(v_raw, 99),
               disp_p999=pct(v_raw, 99.9), disp_max=float(np.nanmax(v_raw)) if fin.sum() > 1 else np.nan)
    ref = tc.smooth_remask(xy, tc.V2["smooth_window"])
    v_ref = consecutive_speed(ref)
    moving = np.nan_to_num(v_ref) > MOVING_CM_S
    row.update(speed_ref_p50=pct(v_ref, 50), speed_ref_p90=pct(v_ref, 90), moving_frac=float(moving.mean()),
               path_ref_cm=round(path_length(ref), 0))

    # v1 rule
    mult = V1_MULTIPLIER[arena]
    v1 = s2r.drop_jumps(raw, nodes, multiplier=mult)
    dropped = np.isnan(v1["head_back.x"].to_numpy()) & fin
    n_mov, n_still = int((moving & fin).sum()), int((~moving & fin).sum())
    row.update(v1_threshold_px=round(v1_threshold_px(raw, nodes[0], mult), 3),
               v1_drop_frac=float(dropped.mean()),
               v1_drop_moving_frac=float((dropped & moving).sum() / n_mov) if n_mov else np.nan,
               v1_drop_still_frac=float((dropped & ~moving).sum() / n_still) if n_still else np.nan)
    g1 = tc.describe_gaps(np.isnan(v1["head_back.x"].to_numpy()), LONG_GAP)
    v1_cm = to_cm(s2r.interpolate_smooth(v1, nodes)[["head_back.x", "head_back.y"]].to_numpy(float))
    v_v1 = consecutive_speed(v1_cm)
    row.update(v1_gap_longest_s=round(g1["longest"] / FPS, 2), v1_gap_gt2s=g1["n_gt_long"],
               v1_speed_p90=pct(v_v1, 90),
               v1_path_ratio=round(path_length(v1_cm) / path_length(ref), 3) if path_length(ref) > 0 else np.nan)

    # v2 rule (dry)
    cm2, flags, rep = tc.clean_v2(raw, nodes, scores, to_cm)
    row.update(tc.report_row(rep))
    # per-node rule removals (a bodypart-specific failure shows here)
    for bp in nodes:
        row[f"v2_rules_{bp}"] = rep[bp]["removed_by_rules"]
    f = flags["head_back"].to_numpy()
    out, why = tc.outcome(f), tc.reason(f)
    removed = out != tc.KEPT
    xy2 = cm2[["head_back.x", "head_back.y"]].to_numpy(float)
    v_v2 = consecutive_speed(xy2)
    row.update(v2_removed_frac=float((removed & fin).mean()),
               v2_removed_moving_frac=float((removed & moving & fin).sum() / n_mov) if n_mov else np.nan,
               v2_left_nan_frac=float((out == tc.LEFT_NAN).mean()),
               v2_longest_left_s=round(rep["head_back"]["longest_left_nan"] / FPS, 2),
               v2_r1_run_gt5=runs_longer_than(why == tc.R_VELOCITY, 5),
               v2_speed_p90=pct(v_v2, 90),
               v2_path_ratio=round(path_length(xy2) / path_length(ref), 3) if path_length(ref) > 0 else np.nan)

    # bridged frames off the maze skeleton
    if arena == "maze":
        d2 = dist_to_skeleton(xy2)
        kept = out == tc.KEPT
        p99 = pct(d2[kept], 99)
        bridged = out == tc.BRIDGED
        row.update(skel_dist_kept_p50=pct(d2[kept], 50), skel_dist_kept_p99=round(p99, 2),
                   bridged_n=int(bridged.sum()),
                   bridged_off_maze_frac=float(np.mean(d2[bridged] > p99)) if bridged.any() else np.nan)
    else:
        row.update(skel_dist_kept_p50=np.nan, skel_dist_kept_p99=np.nan, bridged_n=int((out == tc.BRIDGED).sum()),
                   bridged_off_maze_frac=np.nan)

    # skeleton-shift jumps (cm): head_back moves > 4.5 cm in one frame (> 270 cm/s, beyond any
    # head motion: p99 of real displacement is ~42 cm/s on the maze, ~70 in the big field) and
    # lands within 1 cm of where another node was on the previous frame. The whole SLEAP
    # skeleton slides one segment backwards for 1-6 frames with point score 1.00; 0.6 % of
    # frames on the mEC maze-2 model vs 0.04-0.16 % elsewhere (2026-09-23, 6 sessions). In cm,
    # not px: the big field has ~8 px/cm against the maze's ~17.5.
    dcm = np.hypot(*np.diff(xy, axis=0).T)
    big = np.flatnonzero(np.nan_to_num(dcm) > 4.5)
    n_swap = 0
    if len(big):
        others = [bp for bp in nodes if bp != "head_back"]
        allcm = np.stack([to_cm(raw[[f"{bp}.x", f"{bp}.y"]].to_numpy(float)) for bp in others], axis=1)   # (T, n-1, 2)
        land = np.linalg.norm(allcm[big] - xy[big + 1][:, None, :], axis=-1)       # to other nodes' previous positions
        n_swap = int(np.nansum(np.nanmin(land, axis=1) < 1.0))
    row.update(jump45cm_n=int(len(big)), jump45cm_frac=float(len(big) / T), swap_like_n=n_swap,
               swap_like_frac_of_jumps=float(n_swap / len(big)) if len(big) else np.nan)

    # stuck plateaus (raw px): < 0.3 px/frame for >= 120 frames
    dpx = np.hypot(*np.diff(xy_px, axis=0).T)
    still_px = np.nan_to_num(dpx, nan=np.inf) < 0.3
    s, e = tc.gap_runs(still_px)
    L = e - s
    plateaus = [(a, b) for a, b, l in zip(s, e, L) if l >= LONG_GAP]
    row.update(plateau_n=len(plateaus),
               plateau_longest_s=round(L.max() / FPS, 1) if len(L) and L.max() >= LONG_GAP else 0.0,
               plateau_inst_mean=float(np.mean(np.concatenate([isc[a:b] for a, b in plateaus]))) if plateaus else np.nan)

    # body length head_back - body_back
    if "body_back" in nodes:
        bb = to_cm(raw[["body_back.x", "body_back.y"]].to_numpy(float))
        bl = np.hypot(*(xy - bb).T)
        med = np.nanmedian(bl)
        ok = np.isfinite(bl)
        row.update(body_len_p1=pct(bl, 1), body_len_p50=round(float(med), 2), body_len_p99=pct(bl, 99),
                   body_len_implausible_frac=float(np.mean((bl[ok] > 2 * med) | (bl[ok] < 0.4 * med))) if ok.any() else np.nan)

    # out of arena, coverage, span
    if fin.any():
        x, y = xy[fin, 0], xy[fin, 1]
        row["out_of_arena_frac"] = float(np.mean((x < -5) | (x > side + 5) | (y < -5) | (y > side + 5)))
        sx = np.percentile(x, 99.5) - np.percentile(x, 0.5)
        sy = np.percentile(y, 99.5) - np.percentile(y, 0.5)
        row.update(span_x_cm=round(float(sx), 1), span_y_cm=round(float(sy), 1),
                   span_min_frac=round(float(min(sx, sy) / side), 3))
        if arena == "maze":
            dn = np.linalg.norm(xy[fin][:, None, :] - MAZE_NODES[None], axis=-1)
            at = dn < NODE_RADIUS_CM
            row["nodes_visited"] = int((at.mean(axis=0) >= 0.005).sum())
    else:
        row.update(out_of_arena_frac=np.nan, span_x_cm=np.nan, span_y_cm=np.nan, span_min_frac=np.nan)

    flags_on = []
    for name, (col, op, thr) in FLAG_RULES.items():
        v = row.get(col, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if (op == ">" and v > thr) or (op == "<" and v < thr):
            flags_on.append(name)
    if arena == "maze" and row.get("session_type") in ("ABCD", "ABCDE") and row.get("nodes_visited", 9) < 7:
        flags_on.append("coverage")
    row["flags"] = ";".join(flags_on)
    return row


# ---------------------------------------------------------------------------
# session inventory
# ---------------------------------------------------------------------------

def sheet_sessions() -> dict[tuple[str, str], tuple[str, int, str]]:
    """{(mouse, stamp): (recday, session, session_type)} for every tracked sheet session on a sorted block."""
    out = {}
    for recday in rr.all_recdays("merged"):
        m, _ = rr.recday_dates(recday)
        rows = rr.task_session_rows(recday)
        for s in range(len(rows)):
            row = rows.iloc[s]
            tr = row["Tracking"]
            if pd.isna(tr) or str(tr).strip() in ("-", ""):
                continue
            d = str(row["_date"])
            stamp = f"{d[:4]}-{d[4:6]}-{d[6:]}-{str(int(float(tr))).zfill(6)}"
            out[(m, stamp)] = (recday, s, rr.session_type(recday, s))
    return out


def match_sheet(sheet: dict, mouse: str, stamp: str) -> tuple[str, int, str] | None:
    if (mouse, stamp) in sheet:
        return sheet[(mouse, stamp)]
    want = datetime.strptime(stamp, "%Y-%m-%d-%H%M%S")
    near = [v for (m, st), v in sheet.items() if m == mouse
            and abs((datetime.strptime(st, "%Y-%m-%d-%H%M%S") - want).total_seconds()) <= 120]
    return near[0] if len(near) == 1 else None


def stamp_of(h5: Path, mouse: str) -> str | None:
    m = re.search(rf"{mouse}_(\d{{4}}-\d{{2}}-\d{{2}}-\d{{6}})\.analysis\.h5$", h5.name)
    return m.group(1) if m else None


def maze_jobs(mice, sheet) -> list[tuple[Path, dict, str, float]]:
    jobs = []
    for mouse in mice:
        for h5, stem, d in s2r.session_files(mouse):
            stamp = stamp_of(h5, mouse)
            hit = match_sheet(sheet, mouse, stamp) if stamp else None
            mask_name, corners, note = s2r.geometry(mouse, d)
            base = dict(dataset=rr.DATASET, mouse=mouse, arena="maze", stem=stem, h5=h5.name, stamp=stamp,
                        recday=hit[0] if hit else "", session=hit[1] if hit else -1,
                        session_type=hit[2] if hit else "untracked_day", geometry=mask_name, geometry_note=note,
                        frame="maze")
            jobs.append((h5, base, "maze", ARENA_SIDE["maze"], corners))
    return jobs


def arena_jobs(mice, sheet) -> list:
    import openfield_to_xy as ofx                     # cv2 via arena_calibration
    import arena_calibration as ac
    jobs = []
    cals = {}
    for mouse in mice:
        arena_name = ofx.arena_for(mouse)
        if arena_name not in cals:
            cals[arena_name] = ac.latest_calibration(rr.REPO_ROOT, rr.DATASET, arena_name)
        arena = "bigfield" if arena_name == "bigfield" else "box"
        d = rr.REPO_ROOT / "data" / "raw" / "sleap" / mouse / ofx.RAW_DIR
        for h5 in sorted(d.glob("*.analysis.h5")):
            stamp = stamp_of(h5, mouse)
            hit = match_sheet(sheet, mouse, stamp) if stamp else None
            sheet_stamp = None
            if hit is None:
                # the sheet stamp may differ from the h5 stamp by more than 2 min (ah10 2025-06-25)
                kind = "no_sheet_session"
            else:
                kind = hit[2]
                sheet_stamp = [st for (m, st), v in sheet.items() if m == mouse and v == hit][0]
            cal, frame = ofx.session_calibration(cals[arena_name], mouse, sheet_stamp or stamp)
            side = cal.arena_side_cm or (ARENA_SIDE["bigfield"] if arena == "bigfield" else ARENA_SIDE["maze"])
            base = dict(dataset=rr.DATASET, mouse=mouse, arena=arena, stem=h5.name.split(".analysis")[0], h5=h5.name,
                        stamp=stamp, recday=hit[0] if hit else "", session=hit[1] if hit else -1, session_type=kind,
                        geometry=arena_name, geometry_note="", frame=frame)
            jobs.append((h5, base, arena, float(side), cal))
    return jobs


def run_jobs(jobs, limit=None, verbose=True) -> pd.DataFrame:
    rows = []
    for n, (h5, base, arena, side, geom) in enumerate(jobs[:limit] if limit else jobs):
        t0 = time.time()
        try:
            if arena == "maze":
                to_cm = maze_to_cm(geom, check=not str(base.get("geometry_note", "")).startswith("DEGENERATE"))
            else:
                to_cm = geom.to_cm
            row = qc_session(h5, to_cm, arena, side, base)
        except (OSError, KeyError, ValueError) as exc:
            row = dict(base, status="h5_unreadable" if isinstance(exc, OSError) else "error",
                       flags="h5_unreadable" if isinstance(exc, OSError) else "error", error=str(exc)[:120])
        row.setdefault("status", "ok")
        rows.append(row)
        if verbose:
            print(f"{n + 1:4d}/{len(jobs)} {row['stem']:32s} {row.get('session_type', ''):16s} {row.get('status')} "
                  f"nan={row.get('raw_nan_frac', np.nan):.3f} inst<.5={row.get('inst_lt05', np.nan):.3f} "
                  f"v1drop_mov={row.get('v1_drop_moving_frac', np.nan):.2f} v2rem_mov={row.get('v2_removed_moving_frac', np.nan):.3f} "
                  f"span={row.get('span_min_frac', np.nan):.2f} flags={row.get('flags', '')} ({time.time() - t0:.1f}s)",
                  flush=True)
    return pd.DataFrame(rows)


def write_versioned(table: pd.DataFrame, path: Path) -> Path:
    n = 0
    while (p := path.with_name(path.stem + (f".v{n}" if n else "") + path.suffix)).exists():
        n += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def gate() -> int:
    import tracking_clean_synthetics as syn
    fails = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    p = syn.planted_session(plant_low_off=False)
    with tempfile.TemporaryDirectory() as td:
        h5 = syn.write_synthetic_h5(Path(td) / "synthetic_2000-01-01-000000.analysis.h5", p["raw"],
                                    p["point_scores"], p["instance_scores"])
        base = dict(dataset="synthetic", mouse="syn", arena="maze", stem="syn", h5=h5.name, stamp="", recday="",
                    session=-1, session_type="ABCD", geometry="identity", geometry_note="", frame="maze")
        row = qc_session(h5, lambda a: np.asarray(a, float), "maze", 47.0, base)
    T = p["T"]
    n_spikes = row["v2_r1_velocity"] + row["v2_r2_spike"] + row["v2_r3_hampel"]
    check("QC counts planted spikes exactly", n_spikes == 50, f"{n_spikes} R1+R2+R3 removals for 50 spikes")
    check("QC raw gap count and lengths exact", row["raw_gap_n"] == 3 and row["raw_gap_gt2s"] == 1
          and abs(row["raw_gap_longest_s"] - 200 / FPS) < 0.02 and abs(row["raw_frames_in_gt2s_frac"] - 200 / T) < 1e-9,
          f"n={row['raw_gap_n']} longest={row['raw_gap_longest_s']} s gt2s={row['raw_gap_gt2s']}")
    check("QC low-score fraction exact", abs(row["pt_lt05"] - 100 / (T - 310)) < 1e-6 and abs(row["inst_lt05"] - 10 / (T - 310)) < 1e-6,
          f"pt<0.5 {row['pt_lt05']:.5f} (100 of {T - 310}), inst<0.5 {row['inst_lt05']:.5f}")
    check("QC v2 bridged / left-NaN exact", row["v2_bridged"] == 50 + 10 + 100 and row["v2_left_nan"] == 200,
          f"bridged {row['v2_bridged']} (50 spikes + 110 dropout frames), left NaN {row['v2_left_nan']}")
    check("QC on-trajectory low-score runs pass the gate", row["low_off_tol_frac"] == 0.0 and row["low_runs_n"] == 3,
          f"gated {row['low_off_tol_frac']}, runs {row['low_runs_n']}, dev p90 {row['low_dev_p90']:.2f} cm (curved path)")
    # moving-frame classification vs the planted truth, and the v1 drop fraction against it
    raw_df = syn.raw_dataframe(p["raw"])
    ref = tc.smooth_remask(p["raw"]["head_back"], 5)
    moving_qc = np.nan_to_num(consecutive_speed(ref)) > MOVING_CM_S
    agree = float(np.mean(moving_qc == p["moving"]))
    check("QC moving mask agrees with the planted speed profile", agree > 0.95, f"agreement {agree:.3f}")
    from sleap_to_rois import drop_jumps
    v1 = drop_jumps(raw_df, syn.NODES, multiplier=3.0)
    dropped = np.isnan(v1["head_back.x"].to_numpy()) & ~np.isnan(p["raw"]["head_back"][:, 0])
    truth_frac = (dropped & p["moving"]).sum() / p["moving"].sum()
    check("QC v1 moving-frame drop within 5 % of the planted truth", abs(row["v1_drop_moving_frac"] - truth_frac) < 0.05,
          f"QC {row['v1_drop_moving_frac']:.3f} vs truth {truth_frac:.3f}")
    check("v1 false-drop flag raised on the synthetic", "false_drop" in row["flags"], f"flags={row['flags']}")
    check("v2 moving-frame removal small", row["v2_removed_moving_frac"] < 0.02, f"{row['v2_removed_moving_frac']:.4f}")
    check("coverage / span sane on the synthetic", row["span_min_frac"] > 0.6 and "degenerate_geometry" not in row["flags"],
          f"span_min_frac {row['span_min_frac']}, nodes_visited {row.get('nodes_visited')}")
    print(f"\n{len(fails)} failed" + (": " + ", ".join(fails) if fails else ""))
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--arena", choices=("maze", "arena", "all"), default="maze",
                    help="maze rows (numpy only) or arena rows (box / big field, need cv2)")
    ap.add_argument("--mouse", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--gate", action="store_true", help="run the planted synthetic through the QC and check the counts")
    args = ap.parse_args()
    if args.gate:
        sys.exit(gate())

    s2r.GEOMETRY_VERSION = "v2"                       # cyclic maze-2 corners; the QC never writes CSVs
    sheet = sheet_sessions()
    mice = args.mouse or list(rr.EPHYS_MICE)
    jobs = []
    if args.arena in ("maze", "all"):
        jobs += maze_jobs(mice, sheet)
    if args.arena in ("arena", "all"):
        jobs += arena_jobs(mice, sheet)
    print(f"{len(jobs)} sessions ({rr.DATASET}, {args.arena}); {len(sheet)} sheet sessions with a tracking stamp")
    table = run_jobs(jobs, args.limit)
    if len(table):
        print("\nper mouse x arena (medians):")
        cols = [c for c in ("raw_nan_frac", "inst_lt05", "low_off3_frac", "v1_drop_moving_frac", "v2_removed_moving_frac",
                            "v2_left_nan_frac", "bridged_off_maze_frac", "span_min_frac") if c in table]
        print(table.groupby(["mouse", "arena"])[cols].median().round(3).to_string())
        print("\nflags:")
        print(table["flags"].fillna("").str.split(";").explode().replace("", "none").value_counts().to_string())
        if args.write:
            p = write_versioned(table, PROCESSED_DATA / f"tracking_qc_{rr.DATASET}_{args.arena}.csv")
            print(f"QC -> {p}")


if __name__ == "__main__":
    main()
