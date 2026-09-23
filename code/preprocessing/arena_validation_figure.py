"""
Distortion-correction validation figure for an arena (user request, 2026-09-18).

Panels: (1) the median open-field frame as recorded, with the clicked corners; (2) the same
frame undistorted with the fitted lens model (walls should straighten); (3) the frame warped
into the arena's cm frame through the full calibration (walls should be the square's edges);
(4) tracked head_back occupancy in cm for every open-field / object session that has cleaned
coordinates, with the arena square, clicked objects and the per-wall straightness residual.

Usage (histology env):  python arena_validation_figure.py --dataset mec --arena bigfield
Writes data/processed_data/arena_validation_{dataset}_{arena}.png (never overwritten).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
import arena_calibration as ac                                        # noqa: E402

PD = rr.REPO_ROOT / "data" / "processed_data"


def _latest(folder: Path, pattern: str) -> Path | None:
    fs = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return fs[-1] if fs else None


def _versioned_png(path: Path) -> Path:
    n = 0
    while (p := path.with_name(path.stem + (f".v{n}" if n else "") + ".png")).exists():
        n += 1
    return p


def lec_figures(scale: float = 4.0) -> None:
    """LEC (2026-09-20): one figure per box. The box shifts between sessions and every session has
    its own homography to the 50 cm square (`openfield_to_xy.session_calibration`), so the panels
    are: (1) one session's median frame with its clicked corners; (2) that frame warped through the
    session's homography into the box frame -- straight walls and a full square are the test;
    (3) every session's clicked corners in the MAZE frame, the shift and the 4 % trapezoid seen
    directly; (4) the numbers per session; then head_back occupancy per session in the box frame
    with the square and the clicked objects. Writes arena_validation_lec_{box}.png (versioned)."""
    import ast
    import cv2
    import matplotlib.pyplot as plt
    import openfield_to_xy as ox
    SLEAP = rr.REPO_ROOT / "data" / "preprocessed" / "sleap"
    corners = pd.read_csv(_latest(PD, "arena_corners_lec_cm*.csv"))
    qc = pd.read_csv(_latest(SLEAP, "openfield_to_xy_qc_lec*.csv"))
    objects = pd.read_csv(_latest(PD, "object_positions_lec_cm*.csv"))
    side = ac.LEC_BOX_SIDE_CM
    for box in sorted(corners.arena.unique()):
        sub = corners[corners.arena == box]
        sess = sub.drop_duplicates(["recday", "session"]).sort_values(["mouse", "stamp"]).reset_index(drop=True)
        base = ac.latest_calibration(rr.REPO_ROOT, "lec", box)
        n = len(sess)
        cols = 4
        rows_occ = int(np.ceil(n / cols))
        fig = plt.figure(figsize=(4.6 * cols, 4.0 * (1 + rows_occ)))
        gs = fig.add_gridspec(1 + rows_occ, cols)

        r0 = sess.iloc[0]
        c0 = sub[(sub.recday == r0.recday) & (sub.session == r0.session)].sort_values("corner")
        img = cv2.imread(str(PD / "object_frames" / r0.frame), cv2.IMREAD_GRAYSCALE)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img, cmap="gray")
        ax.plot(c0.x_px, c0.y_px, "r+", ms=12, mew=2)
        for _, cc in c0.iterrows():
            ax.text(cc.x_px + 12, cc.y_px - 12, cc.corner_name, color="r", fontsize=8)
        ax.set_title(f"1. {r0.mouse} {r0.stamp}: median frame + clicked corners", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        cal, _ = ox.session_calibration(base, r0.mouse, r0.stamp)
        H = np.asarray(cal.H, float)
        S = np.diag([scale, scale, 1.0])
        warped = cv2.warpPerspective(img, S @ H, (int(side * scale) + 1, int(side * scale) + 1))
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(warped, cmap="gray", extent=[0, side, side, 0])
        ax.add_patch(plt.Rectangle((0, 0), side, side, fill=False, color="r", lw=1))
        ax.set_title(f"2. warped through this session's homography into the {side:.0f} cm box", fontsize=9)
        ax.set_xlabel("cm"); ax.set_ylabel("cm")

        ax = fig.add_subplot(gs[0, 2])
        cmap = plt.get_cmap("viridis", max(n, 2))
        first_centre = {}
        shifts = {}
        for i, r in sess.iterrows():
            c = sub[(sub.recday == r.recday) & (sub.session == r.session)].sort_values("corner")
            pts = c[["x_maze_cm", "y_maze_cm"]].to_numpy()
            poly = np.vstack([pts, pts[:1]])
            ax.plot(poly[:, 0], poly[:, 1], "-", color=cmap(i), lw=1, label=f"{r.mouse} s{int(r.session)}")
            ax.plot(r.centre_maze_x_cm, r.centre_maze_y_cm, "o", color=cmap(i), ms=3)
            ref = first_centre.setdefault(r.mouse, (r.centre_maze_x_cm, r.centre_maze_y_cm))
            shifts[(r.recday, int(r.session))] = float(np.hypot(r.centre_maze_x_cm - ref[0], r.centre_maze_y_cm - ref[1]))
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xlabel("maze frame, cm"); ax.set_ylabel("cm")
        ax.set_title("3. clicked corners of every session in the maze frame\n(the shift between sessions; the 4 % trapezoid)", fontsize=9)
        ax.legend(fontsize=5, ncol=2, framealpha=0.6)

        ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
        lines = [f"lec / {box}: box {side:.0f} cm, per-session homography from 4 corners",
                 f"maze affine reprojection {base.reprojection_rms_px:.1f} px on the 4 maze corners", "",
                 "session         side(maze) trapez angle  shift  walls max (box cm)",
                 "                   cm        %     deg    cm    L/R/B/T"]
        for _, r in sess.iterrows():
            q = qc[(qc.mouse == r.mouse) & (qc.stamp == r.stamp)]
            walls = ast.literal_eval(q.wall_rms_cm.iloc[0]) if len(q) and isinstance(q.wall_rms_cm.iloc[0], str) else None
            wtxt = "/".join(f"{walls[k]:.1f}" for k in ("left", "right", "bottom", "top")) if walls else "no track"
            lines.append(f"{r.mouse} {r.stamp[5:]}  {r.side_maze_cm:5.1f}   {r.trapezoid_pct:+4.1f}  {r.maze_angle_err_deg:4.1f}  {shifts[(r.recday, int(r.session))]:4.1f}  {wtxt}")
        lines += ["", "side(maze), trapez, angle: the clicked corners under the maze affine",
                  "(what the homography corrects). shift: box centre vs the mouse's first",
                  "session. walls: RMS of the occupancy edge from straight, box frame."]
        ax.text(0, 1, "\n".join(lines), va="top", fontsize=7, family="monospace", transform=ax.transAxes)

        for i, (_, r) in enumerate(sess.iterrows()):
            ax = fig.add_subplot(gs[1 + i // cols, i % cols])
            csv = SLEAP / "openfield" / r.mouse / f"{r.mouse}_{r.stamp}.analysis_cleaned_coordinates.csv"
            if not csv.exists():
                ax.set_title(f"{r.mouse} {r.stamp[5:]}: no tracking", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
                continue
            xy = pd.read_csv(csv, usecols=["head_back.x", "head_back.y"]).to_numpy()
            xy = xy[~np.isnan(xy).any(axis=1)]
            ax.hist2d(xy[:, 0], xy[:, 1], bins=50, range=[[-5, side + 5], [-5, side + 5]], cmap="magma")
            ax.add_patch(plt.Rectangle((0, 0), side, side, fill=False, color="c", lw=1))
            ob = objects[(objects.mouse == r.mouse) & (objects.stamp == r.stamp)]
            for _, o in ob.iterrows():
                ax.plot(o.x_cm, o.y_cm, "w+", ms=12, mew=2)
                ax.text(o.x_cm, o.y_cm - 3, str(o.object_label).replace("object_", ""), color="w", fontsize=6, ha="center")
            q = qc[(qc.mouse == r.mouse) & (qc.stamp == r.stamp)]
            walls = ast.literal_eval(q.wall_rms_cm.iloc[0]) if len(q) and isinstance(q.wall_rms_cm.iloc[0], str) else {}
            wmax = max(v for v in walls.values() if v is not None) if walls else float("nan")
            ax.set_aspect("equal"); ax.invert_yaxis()
            ax.set_title(f"{r.mouse} s{int(r.session)} {r.stamp[5:]}  walls max {wmax:.1f} cm", fontsize=8)
            ax.set_xlabel("box cm"); ax.set_ylabel("cm")
        fig.suptitle(f"LEC {box}: per-session box calibration -- frame, warp, corner shift, occupancy", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = _versioned_png(PD / f"arena_validation_lec_{box}.png")
        fig.savefig(out, dpi=100, bbox_inches=None)
        plt.close(fig)
        print(f"-> {out}")


def main() -> None:
    import cv2
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--arena", default=None, help="mEC: the arena to draw (bigfield); LEC draws every box")
    ap.add_argument("--side-cm", type=float, default=None)
    args = ap.parse_args()
    if rr.DATASET == "lec":
        lec_figures()
        return
    if not args.arena:
        ap.error("--arena is required for the mEC dataset")

    cal = ac.latest_calibration(rr.REPO_ROOT, rr.DATASET, args.arena)
    side = args.side_cm or cal.arena_side_cm or 120.0
    corners = pd.read_csv(PD / f"arena_corners_{rr.DATASET}.csv")
    corners = corners[corners.arena == args.arena].sort_values("corner")
    frame_name = corners.frame.iloc[0]
    img = cv2.imread(str(PD / "object_frames" / frame_name), cv2.IMREAD_GRAYSCALE)
    K, dist = np.asarray(cal.K, float), np.asarray(cal.dist, float)
    und = cv2.undistort(img, K, dist, None, K) if cal.dist is not None else img
    # warp into the arena frame at 4 px per cm
    scale = 4.0
    H = np.asarray(cal.H, float)
    S = np.diag([scale, scale, 1.0])
    warped = cv2.warpPerspective(und, S @ H, (int(side * scale) + 1, int(side * scale) + 1))

    sessions = pd.read_csv(sorted((rr.REPO_ROOT / "data" / "preprocessed" / "sleap").glob(f"openfield_to_xy_qc_{rr.DATASET}*.csv"))[-1]) \
        if list((rr.REPO_ROOT / "data" / "preprocessed" / "sleap").glob(f"openfield_to_xy_qc_{rr.DATASET}*.csv")) else pd.DataFrame()
    objects = pd.read_csv(sorted(PD.glob(f"object_positions_{rr.DATASET}_cm*.csv"))[-1]) if list(PD.glob(f"object_positions_{rr.DATASET}_cm*.csv")) else pd.DataFrame()

    n_occ = len(sessions)
    cols = 4
    rows_occ = int(np.ceil(n_occ / cols)) if n_occ else 0
    fig = plt.figure(figsize=(4.4 * cols, 3.8 * (1 + rows_occ)))
    gs = fig.add_gridspec(1 + rows_occ, cols)

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(img, cmap="gray"); ax.plot(*np.asarray(cal.corners_px).T, "r+", ms=10, mew=2)
    ax.set_title("1. recorded median frame + clicked corners", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(und, cmap="gray")
    ax.plot(*cal.undistort(np.asarray(cal.corners_px)).T, "r+", ms=10, mew=2)
    ax.set_title(f"2. undistorted (k1={cal.dist[0]:.3f}, k2={cal.dist[1]:.3f})" if cal.dist else "2. no distortion model", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(warped, cmap="gray", extent=[0, side, side, 0])
    ax.add_patch(plt.Rectangle((0, 0), side, side, fill=False, color="r", lw=1))
    ax.set_title(f"3. warped to the {side:.0f} cm arena frame", fontsize=9); ax.set_xlabel("cm"); ax.set_ylabel("cm")
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    txt = [f"{rr.DATASET}/{args.arena}", f"reprojection on pegs: {cal.reprojection_rms_px:.2f} px" if cal.reprojection_rms_px else "",
           f"side lengths from corners: {np.round(cal.square_side_cm, 1).tolist()} cm" if cal.square_side_cm else "",
           "", "wall straightness (RMS of the occupancy edge, cm):"]
    if len(sessions):
        for _, s in sessions.iterrows():
            txt.append(f"  {s.stamp} {s.kind:9s} {s.wall_rms_cm}")
    ax.text(0, 1, "\n".join(txt), va="top", fontsize=7.5, family="monospace", transform=ax.transAxes)

    for i, (_, s) in enumerate(sessions.iterrows()):
        ax = fig.add_subplot(gs[1 + i // cols, i % cols])
        csv = rr.REPO_ROOT / "data" / "preprocessed" / "sleap" / "openfield" / s.mouse / f"{s.mouse}_{s.stamp}.analysis_cleaned_coordinates.csv"
        if not csv.exists():
            ax.set_title(f"{s.stamp}: no CSV", fontsize=8); continue
        xy = pd.read_csv(csv, usecols=["head_back.x", "head_back.y"]).to_numpy()
        xy = xy[~np.isnan(xy).any(axis=1)]
        ax.hist2d(xy[:, 0], xy[:, 1], bins=60, range=[[-10, side + 10], [-10, side + 10]], cmap="magma")
        ax.add_patch(plt.Rectangle((0, 0), side, side, fill=False, color="c", lw=1))
        ob = objects[(objects.mouse == s.mouse) & (objects.stamp == s.stamp)] if len(objects) else objects
        for _, o in ob.iterrows():
            ax.plot(o.x_cm, o.y_cm, "w+", ms=12, mew=2); ax.text(o.x_cm, o.y_cm - 5, o.object_label, color="w", fontsize=7, ha="center")
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_title(f"{s.mouse} {s.stamp} {s.kind}", fontsize=8); ax.set_xlabel("cm"); ax.set_ylabel("cm")
    out = PD / f"arena_validation_{rr.DATASET}_{args.arena}.png"
    n = 0
    while (p := out.with_name(out.stem + (f".v{n}" if n else "") + ".png")).exists():
        n += 1
    fig.savefig(p, dpi=100, bbox_inches=None)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
