"""
Does the LEC box need a radial term? Measurement (2026-09-18).

With the maze affine alone the box walls come out 0.5-1.8 cm RMS from straight
(`openfield_to_xy_qc_lec.csv`), against 0.2-0.9 cm for the mEC field after its k1/k2 fit. The
box fills the frame, where barrel distortion is largest, so this scans one radial coefficient
k1 (principal point at the image centre, f = 930 px as for the mEC camera; any f works with a
compensating k1) and, for each value, refits the maze affine on the undistorted maze corners,
maps every object session's cleaned head_back track to cm and scores (a) the mean straight-wall
RMS over the four walls and all sessions of that camera, (b) the mean side spread and angle
error of the clicked per-session box corners. Prints the table per box and the k1 that
minimises the wall RMS, with per-session optima to show consistency. Nothing is written; if a
k1 is adopted, `write_lec_calibration_with_k1(box, k1)` saves a new calibration version.

Run with the histology env (cv2):  python lec_box_distortion.py [--write-k1 BOX K1]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["TSA_DATASET"] = "lec"

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
import arena_calibration as ac                                        # noqa: E402
import openfield_to_xy as ox                                          # noqa: E402
from sleap_to_rois import LEC_MAZE1, LEC_MAZE2, PHYSICAL_PTS, load_tracks, drop_jumps, interpolate_smooth  # noqa: E402

F_PX = 930.0
K = np.array([[F_PX, 0, ac.IMAGE_SIZE[0] / 2], [0, F_PX, ac.IMAGE_SIZE[1] / 2], [0, 0, 1.0]])
K1_GRID = np.round(np.arange(-0.45, 0.151, 0.025), 3)
PD = rr.REPO_ROOT / "data" / "processed_data"


def undistort(px: np.ndarray, k1: float) -> np.ndarray:
    import cv2
    px = np.asarray(px, float)
    out = np.full_like(px, np.nan)
    ok = ~np.isnan(px).any(axis=1)
    if ok.any():
        u = cv2.undistortPoints(px[ok].reshape(-1, 1, 2), K, np.array([k1, 0, 0, 0, 0.0]), P=K)
        out[ok] = u.reshape(-1, 2)
    return out


def calibration_with_k1(box: int, k1: float) -> ac.ArenaCalibration:
    maze_px = np.asarray(LEC_MAZE1 if box == 1 else LEC_MAZE2, float)
    fit = ac.fit_affine(undistort(maze_px, k1), np.asarray(PHYSICAL_PTS, float))
    cal = ac.ArenaCalibration("lec", f"openfield_box{box}", ac.IMAGE_SIZE, K=K.tolist(),
                              dist=[float(k1), 0.0, 0.0, 0.0, 0.0], H=fit["H"],
                              reprojection_rms_px=fit["reprojection_rms_px"], n_points=4,
                              source=f"maze-{box} corner affine on undistorted corners, 47 cm; k1 from the box walls (lec_box_distortion.py)",
                              notes=[f"k1={k1:+.3f} at f={F_PX:.0f} px, principal point = image centre; chosen to straighten the box walls"])
    return cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-k1", nargs=2, metavar=("BOX", "K1"), default=None)
    args = ap.parse_args()
    corners = pd.read_csv(PD / "arena_corners_lec.csv") if (PD / "arena_corners_lec.csv").exists() else None
    sessions = [(r, s, m, st, k) for r, s, m, st, k in ox.openfield_sessions() if ox.find_h5(m, st) is not None]
    tracks = {}
    for r, s, m, st, k in sessions:
        raw, nodes = load_tracks(ox.find_h5(m, st))
        cleaned = interpolate_smooth(drop_jumps(raw, nodes, multiplier=ox.JUMP_MULTIPLIER), nodes)
        tracks[(r, s)] = (m, st, cleaned[["head_back.x", "head_back.y"]].to_numpy())
    print(f"{len(tracks)} sessions loaded")
    for box, mice in ((1, ("ah10", "ly05", "ly07")), (2, ("ah08", "ly06"))):
        keys = [k for k, v in tracks.items() if v[0] in mice]
        rows = []
        per_session_best = {}
        for k1 in K1_GRID:
            cal = calibration_with_k1(box, k1)
            wall, spread, angerr = [], [], []
            for key in keys:
                m, st, px = tracks[key]
                cm = cal.to_cm(px)
                w = ac.straight_wall_test(cm)
                vals = [v["rms_cm"] for v in w.values() if v is not None]
                if vals:
                    wall.append(np.mean(vals))
                    per_session_best.setdefault(key, []).append((np.mean(vals), k1))
                if corners is not None:
                    c = corners[(corners.recday == key[0]) & (corners.session == key[1])].sort_values("corner")
                    if len(c) == 4:
                        ccm = cal.to_cm(c[["x_px", "y_px"]].to_numpy(float))
                        nxt = np.roll(ccm, -1, axis=0) - ccm
                        prv = np.roll(ccm, 1, axis=0) - ccm
                        sides = np.linalg.norm(nxt, axis=1)
                        ang = np.degrees(np.arccos(np.clip(np.sum(nxt * prv, axis=1) / (np.linalg.norm(nxt, axis=1) * np.linalg.norm(prv, axis=1)), -1, 1)))
                        spread.append(100 * (sides.max() - sides.min()) / sides.mean())
                        angerr.append(np.abs(ang - 90).max())
            rows.append(dict(k1=k1, wall_rms_cm=np.mean(wall), corner_side_spread_pct=np.mean(spread) if spread else np.nan,
                             corner_angle_err_deg=np.mean(angerr) if angerr else np.nan, maze_reproj_px=cal.reprojection_rms_px))
        t = pd.DataFrame(rows)
        best = t.loc[t.wall_rms_cm.idxmin()]
        print(f"\n=== box {box} ({', '.join(mice)}; {len(keys)} sessions)")
        print(t.round(3).to_string(index=False))
        print(f"best k1 by walls: {best.k1:+.3f} (wall rms {best.wall_rms_cm:.2f} cm vs {t[t.k1 == 0].wall_rms_cm.iloc[0]:.2f} at k1=0)")
        print("per-session best k1:", {f"{k[0][:4]}{k[0][4:12]} s{k[1]}": min(v)[1] for k, v in per_session_best.items() if k in keys})
    if args.write_k1:
        box, k1 = int(args.write_k1[0]), float(args.write_k1[1])
        cal = calibration_with_k1(box, k1)
        p = cal.save(ac.calibration_path(rr.REPO_ROOT, "lec", f"openfield_box{box}"))
        print(f"-> {p}")


if __name__ == "__main__":
    main()
