"""
Object marking, after the clicks (Stage G3): arena corners -> calibration, objects -> cm, checks.

Reads the click notebook's two CSVs (`arena_corners_{dataset}.csv`, `object_positions_{dataset}.csv`)
and:

1. mEC: applies the arena's four clicked corners to its base calibration
   (`arena_calibration.apply_arena_corners_homography`, 120 cm floor) and saves it as a new
   calibration version. LEC (2026-09-18): the box shifts between sessions and the maze affine
   shows it as a 4 % trapezoid, so every session's four clicked corners give that session its
   own homography to the 50 cm square (`lec_session_corners`, written to
   `arena_corners_lec_cm.csv` in the box and the maze frame);
2. converts every clicked object to cm (`x_cm`, `y_cm`, `radius_cm`; LEC in the session's box
   frame plus `x_maze_cm`, `y_maze_cm`) and writes `object_positions_{dataset}_cm.csv`;
3. runs the checks and writes `object_checks_{dataset}.json`:
   a. round trip px -> cm -> px < 2 px;
   b. clicked corners form a square: side lengths within 2 %, angles within 1 deg;
   c. straight walls: `straight_wall_test` on every open-field/object session's `XY_raw`
      that exists (needs Stage E3 output); RMS < 1 cm per wall;
   d. occupancy hole: fraction of tracked bins within the object's radius (or 6 cm) against
      the arena-wide density, per object;
   e. LEC: every recday's four object sessions carry the same set of labels; the second
      session of a day differs from the first; a derived `config` (A = same as the day's first
      session) is compared with the sheet's `config1/config2` where typed;
   f. mEC: one object per session and its position changes across sessions.
4. draws `object_contact_sheet_{dataset}.png`.

Usage::
    python object_checks.py --dataset mec [--write]
"""

from __future__ import annotations

import argparse
import json
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
OBJ_CSV = PD / f"object_positions_{rr.DATASET}.csv"
ARENA_CSV = PD / f"arena_corners_{rr.DATASET}.csv"
FRAMES = PD / "object_frames"
DEFAULT_RADIUS_CM = 6.0
ARENA_SIDE_CM = {"bigfield": ac.BIG_FIELD_SIDE_CM}
#: straight-wall gate. mEC field after k1/k2: 0.2-0.9 cm. LEC box: 0.5-1.6 cm whatever the
#: calibration (no k1 lowers it, the per-session homography does not either: lec_box_distortion.py,
#: 2026-09-18), so for the 50 cm box that is the floor of the occupancy-edge method, not distortion.
WALL_RMS_MAX_CM = 2.0 if rr.DATASET == "lec" else 1.0
#: investigation ring around an object for the centring test. 8 cm: LEC objects stand 10-12 cm
#: from a wall, and a 12 cm ring pulled the centroid towards the wall-hugging occupancy.
RING_CM = 8.0
RING_OFFSET_MAX_CM = 3.0


def arena_of(mouse: str) -> str:
    return "bigfield" if rr.DATASET == "mec" else ("openfield_box2" if mouse in ("ah08", "ly06") else "openfield_box1")


CORNER_ORDER = ("TL", "TR", "BR", "BL")


def load_corners() -> pd.DataFrame:
    """Click rows. The first schema (mEC, per arena) had no session columns; they are added
    empty so both schemas read alike."""
    cols = ["arena", "recday", "session", "mouse", "stamp", "kind", "corner", "x_px", "y_px", "frame"]
    if not ARENA_CSV.exists():
        return pd.DataFrame(columns=cols)
    d = pd.read_csv(ARENA_CSV)
    for col in cols:
        if col not in d.columns:
            d[col] = np.nan
    return d


def calibrations_with_corners(write: bool) -> dict[str, ac.ArenaCalibration]:
    corners = load_corners()
    corners = corners[corners.recday.isna()]              # per-arena rows only (mEC)
    cals = {}
    for arena in sorted({arena_of(m) for m in rr.EPHYS_MICE}):
        base = ac.latest_calibration(rr.REPO_ROOT, rr.DATASET, arena)
        if rr.DATASET == "lec":
            # The box shifts between sessions (user, 2026-09-18): its corners are per session
            # (`lec_session_corners`) and the maze affine stays the frame of XY_raw and objects.
            cals[arena] = base
            continue
        c = corners[corners.arena == arena] if len(corners) else corners
        if len(c) == 4 and base.corners_px is not None and "homography refitted" not in " ".join(base.notes) \
                and ARENA_SIDE_CM.get(arena) is not None:
            # an earlier version applied the corners by rescale only; redo on the peg-fit base
            base = ac.ArenaCalibration.load(ac.calibration_path(rr.REPO_ROOT, rr.DATASET, arena))
        if len(c) == 4 and base.corners_px is None:
            px = c.sort_values("corner")[["x_px", "y_px"]].to_numpy()
            side = ARENA_SIDE_CM.get(arena)
            if side is not None:
                # mEC: the floor is above the peg plane -> refit the floor's own homography
                cal = ac.apply_arena_corners_homography(base, px, side_cm=side)
            else:
                # LEC: scale from the maze affine; corners set origin and rotation only
                cal = ac.apply_arena_corners(base, px, side_cm=float(np.mean(
                    np.linalg.norm(np.roll(base.to_cm(px), -1, axis=0) - base.to_cm(px), axis=1))), rescale=False)
                cal.notes.append("LEC: scale from the maze affine, corners set origin/rotation only")
            if write:
                p = cal.save(ac.calibration_path(rr.REPO_ROOT, rr.DATASET, arena))
                print(f"{arena}: corners applied -> {p.name}")
            cals[arena] = cal
        else:
            cals[arena] = base
            if base.corners_px is None:
                print(f"{arena}: no corners clicked yet ({len(c)} rows); calibration frame unchanged")
    return cals


def square_stats(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Side lengths and inner angles (deg) of four corners in TL,TR,BR,BL order."""
    nxt = np.roll(cm, -1, axis=0) - cm
    prv = np.roll(cm, 1, axis=0) - cm
    sides = np.linalg.norm(nxt, axis=1)
    cosang = np.sum(nxt * prv, axis=1) / (np.linalg.norm(nxt, axis=1) * np.linalg.norm(prv, axis=1))
    return sides, np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def box_frame(cm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points (cm, maze frame) -> the box's own frame: origin TL, x along TL->TR, y towards BL."""
    ex = cm[1] - cm[0]
    ex = ex / np.linalg.norm(ex)
    ey = np.array([-ex[1], ex[0]])
    if np.dot(cm[3] - cm[0], ey) < 0:
        ey = -ey
    d = np.asarray(pts, float) - cm[0]
    return np.stack([d @ ex, d @ ey], axis=1)


def lec_session_corners(cals: dict, checks: dict, write: bool) -> tuple[pd.DataFrame | None, dict]:
    """LEC: per-session box corners -> a per-session calibration, with checks.

    The maze affine shows every session's box as a 4 % trapezoid (bottom 51.2 cm, top 49.1 cm;
    camera tilt, not lens distortion: no k1 straightens the walls, `lec_box_distortion.py`),
    so each session's four clicked corners define their own homography to the 50 cm square
    (`arena_calibration.apply_arena_corners_homography`), frame 'box': origin image-top-left,
    x right, y down. Checks: the affine-measured mean side is within 3 % of 50 cm (scale
    sanity) and the same within a mouse; the box-centre shift between sessions (maze cm) is
    reported, not judged. Writes `arena_corners_lec_cm.csv` (versioned) with both frames.
    Returns (table, {(recday, session): calibration}).
    """
    per = load_corners()
    per = per[per.recday.notna()]
    if not len(per):
        print("LEC: no per-session corners clicked yet")
        return None, {}
    rows, sess_cals = [], {}
    for (recday, s), g in per.groupby(["recday", "session"]):
        if len(g) != 4:
            checks["problems"].append(f"{recday} s{int(s)}: {len(g)} corner rows, expected 4")
            continue
        mouse = str(g.mouse.iloc[0])
        base = cals[arena_of(mouse)]
        cal_s = ac.apply_arena_corners_homography(base, g[["x_px", "y_px"]].to_numpy(float), ac.LEC_BOX_SIDE_CM)
        px = np.asarray(cal_s.corners_px, float)             # TL, TR, BR, BL
        maze_cm = base.to_cm(px)
        box_cm = cal_s.to_cm(px)
        sides, ang = square_stats(maze_cm)
        scale_err = float(100 * (sides.mean() - ac.LEC_BOX_SIDE_CM) / ac.LEC_BOX_SIDE_CM)
        trapezoid = float(100 * (sides[2] - sides[0]) / sides.mean())      # bottom minus top
        if abs(scale_err) > 3:
            checks["problems"].append(f"{recday} s{int(s)}: box side under the maze affine is {sides.mean():.1f} cm, "
                                      f"expected {ac.LEC_BOX_SIDE_CM:.0f} ({scale_err:+.1f} %)")
        sess_cals[(recday, int(s))] = cal_s
        for k in range(4):
            rows.append(dict(recday=recday, session=int(s), mouse=mouse, stamp=g.stamp.iloc[0], kind=g.kind.iloc[0],
                             arena=arena_of(mouse), corner=k, corner_name=CORNER_ORDER[k],
                             x_px=px[k, 0], y_px=px[k, 1], x_cm=box_cm[k, 0], y_cm=box_cm[k, 1],
                             x_maze_cm=maze_cm[k, 0], y_maze_cm=maze_cm[k, 1],
                             side_cm=ac.LEC_BOX_SIDE_CM, side_maze_cm=float(sides.mean()), scale_err_pct=scale_err,
                             trapezoid_pct=trapezoid, maze_angle_err_deg=float(np.abs(ang - 90).max()),
                             centre_maze_x_cm=float(maze_cm[:, 0].mean()), centre_maze_y_cm=float(maze_cm[:, 1].mean()),
                             H=json.dumps(np.round(cal_s.H, 8).tolist()), frame=g.frame.iloc[0]))
    out = pd.DataFrame(rows)
    one = out.drop_duplicates(["recday", "session"])
    for mouse, g in one.groupby("mouse"):
        sides = g.side_maze_cm.to_numpy()
        sp = float(100 * (sides.max() - sides.min()) / sides.mean()) if len(sides) > 1 else 0.0
        c = g[["centre_maze_x_cm", "centre_maze_y_cm"]].to_numpy()
        shifts = np.linalg.norm(c - c[0], axis=1)
        checks["arenas"][f"{arena_of(mouse)}/{mouse}"] = dict(
            n_sessions=int(len(g)), side_maze_cm_mean=round(float(sides.mean()), 2), side_spread_pct=round(sp, 2),
            trapezoid_pct_mean=round(float(g.trapezoid_pct.mean()), 2),
            centre_shift_cm_max=round(float(shifts.max()), 2),
            centre_shift_cm=dict(zip((f"{r}_s{s}" for r, s in zip(g.recday, g.session)), shifts.round(2).tolist())))
        if sp > 3:
            checks["problems"].append(f"{mouse}: box side varies {sp:.1f} % across sessions (same box expected)")
    if write:
        p = PD / "arena_corners_lec_cm.csv"
        n = 0
        while (q := p.with_name(p.stem + (f".v{n}" if n else "") + ".csv")).exists():
            n += 1
        out.to_csv(q, index=False)
        print(f"per-session box corners (box and maze frames) -> {q}")
    return out, sess_cals


def load_xy(recday: str, session: int) -> np.ndarray | None:
    mouse, dates = rr.recday_dates(recday)
    for tag in ("processed_v2", "processed"):
        p = rr.REPO_ROOT / "data" / tag / mouse / "_".join(dates) / f"XY_raw_{recday}_{session}.npy"
        if p.exists():
            return np.load(p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cals = calibrations_with_corners(args.write)
    checks: dict = {"dataset": rr.DATASET, "arenas": {}, "objects": {}, "sessions": [], "problems": []}

    # b. square corners and a. round trip
    for arena, cal in cals.items():
        rec = dict(reprojection_rms_px=cal.reprojection_rms_px, plane_factor=cal.plane_factor)
        if cal.corners_px is not None:
            sides = np.array(cal.square_side_cm)
            rec.update(square_side_cm=sides.round(1).tolist(), square_angles_deg=np.round(cal.square_angles_deg, 2).tolist(),
                       side_spread_pct=float(100 * (sides.max() - sides.min()) / sides.mean()),
                       angle_error_deg=float(np.max(np.abs(np.array(cal.square_angles_deg) - 90))))
            if rec["side_spread_pct"] > 2 or rec["angle_error_deg"] > 1:
                checks["problems"].append(f"{arena}: corners not square (spread {rec['side_spread_pct']:.1f} %, angle err {rec['angle_error_deg']:.1f} deg)")
            px = np.asarray(cal.corners_px)
            rt = np.linalg.norm(cal.to_px(cal.to_cm(px)) - px, axis=1)
            rec["round_trip_px_max"] = float(rt.max())
            if rt.max() > 2:
                checks["problems"].append(f"{arena}: round trip {rt.max():.1f} px")
        checks["arenas"][arena] = rec
    sess_corners, sess_cals = lec_session_corners(cals, checks, args.write) if rr.DATASET == "lec" else (None, {})

    def cal_for(recday: str, session: int, mouse: str) -> ac.ArenaCalibration | None:
        """The session's calibration: mEC the arena's; LEC the session's box homography."""
        if rr.DATASET != "lec":
            return cals[arena_of(mouse)]
        return sess_cals.get((recday, int(session)))

    if not OBJ_CSV.exists():
        print("no object clicks yet")
    else:
        obj = pd.read_csv(OBJ_CSV)
        obj = obj[obj.object_label.astype(str) != "NONE"].copy()
        cm = np.full((len(obj), 2), np.nan)
        maze = np.full((len(obj), 2), np.nan)
        missing = set()
        for i, (r, s, m, x, y) in enumerate(zip(obj.recday, obj.session, obj.mouse, obj.x_px, obj.y_px)):
            cal = cal_for(r, s, m)
            if cal is None:
                missing.add((r, int(s)))
                continue
            cm[i] = cal.to_cm([[x, y]])[0]
            if rr.DATASET == "lec":
                maze[i] = cals[arena_of(m)].to_cm([[x, y]])[0]
        for r, s in sorted(missing):
            checks["problems"].append(f"{r} s{s}: objects clicked but no box corners for this session")
        obj["x_cm"], obj["y_cm"] = cm[:, 0], cm[:, 1]
        obj["coord_frame"] = "box (per-session homography, origin TL, x right, y down)" if rr.DATASET == "lec" else "arena"
        if rr.DATASET == "lec":
            obj["x_maze_cm"], obj["y_maze_cm"] = maze[:, 0], maze[:, 1]
            for r, s in set(sess_cals) - {(r, int(s)) for r, s in zip(obj.recday, obj.session)}:
                if rr.session_type(r, s) == "object":
                    checks["problems"].append(f"{r} s{s}: box corners clicked but no objects")
        # radius in cm from the edge click, else the default
        rad = []
        for r_, s_, m, x, y, r in zip(obj.recday, obj.session, obj.mouse, obj.x_px, obj.y_px, obj.radius_px):
            cal = cal_for(r_, s_, m)
            if pd.notna(r) and r > 0 and cal is not None:
                rad.append(float(np.linalg.norm(cal.to_cm([[x + r, y]])[0] - cal.to_cm([[x, y]])[0])))
            else:
                rad.append(DEFAULT_RADIUS_CM)
        obj["radius_cm"] = rad
        obj["arena"] = obj.mouse.map(arena_of)

        # d. occupancy hole, c. straight walls, per session with tracking
        for (recday, s), g in obj.groupby(["recday", "session"]):
            xy = load_xy(recday, int(s))
            row = dict(recday=recday, session=int(s), n_objects=len(g), labels=g.object_label.tolist(), has_xy=xy is not None)
            if xy is not None:
                ok = ~np.isnan(xy).any(axis=1)
                pts = xy[ok]
                area = np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])
                dens = len(pts) / max(area, 1e-6)
                holes = []
                for _, o in g.iterrows():
                    d = np.linalg.norm(pts - np.array([o.x_cm, o.y_cm]), axis=1)
                    inside = d < o.radius_cm
                    expected = dens * np.pi * o.radius_cm ** 2
                    # Mice investigate objects: head_back occupancy within the object radius is
                    # 0.5-4.4x uniform on every LEC object and 1.5-2.6x on the mEC bottles
                    # (2026-09-18), so an occupancy hole is not a usable position test. What is:
                    # the investigation ring is centred on the object, so the centroid of the
                    # points within RING_CM of the click must sit within RING_OFFSET_MAX_CM of it.
                    near = d < RING_CM
                    offset = float(np.linalg.norm(pts[near].mean(axis=0) - np.array([o.x_cm, o.y_cm]))) if near.sum() >= 50 else float("nan")
                    holes.append(dict(label=o.object_label, frac_inside=float(inside.mean()),
                                      ratio_to_uniform=float(inside.sum() / max(expected, 1e-9)),
                                      n_within_ring=int(near.sum()), ring_centroid_offset_cm=offset))
                    if offset == offset and offset > RING_OFFSET_MAX_CM:
                        checks["problems"].append(f"{recday} s{s} {o.object_label}: occupancy around the object is centred "
                                                  f"{offset:.1f} cm from the click (> {RING_OFFSET_MAX_CM} cm)")
                row["occupancy_hole"] = holes
                walls = ac.straight_wall_test(pts)
                row["wall_rms_cm"] = {k: (None if v is None else round(v["rms_cm"], 2)) for k, v in walls.items()}
                bad = [k for k, v in walls.items() if v is not None and v["rms_cm"] > WALL_RMS_MAX_CM]
                if bad:
                    checks["problems"].append(f"{recday} s{s}: walls not straight {row['wall_rms_cm']}")
            checks["sessions"].append(row)

        # e./f. per-dataset structure
        for recday, g in obj.groupby("recday"):
            per_sess = {int(s): sorted(x.object_label.tolist()) for s, x in g.groupby("session")}
            rec = dict(sessions=per_sess)
            if rr.DATASET == "lec":
                # Four physical objects per recday, two per session, each object in two of the
                # four sessions (clicked 2026-09-18: object_1..4 x 2 sessions for every mouse).
                for s, labs in per_sess.items():
                    if len(labs) != 2:
                        checks["problems"].append(f"{recday} s{s}: {len(labs)} objects, expected 2")
                counts = g.object_label.value_counts()
                rec["objects_per_recday"] = counts.to_dict()
                if len(counts) != 4 or set(counts.values) != {2}:
                    checks["problems"].append(f"{recday}: expected 4 objects each in 2 sessions, got {counts.to_dict()}")
                # day pairs: sessions on the same date; the configuration must change between
                # the two (different objects, or the same object moved by > 5 cm)
                rows_ = rr.task_session_rows(recday)
                by_day = {}
                for s in per_sess:
                    by_day.setdefault(str(rows_.iloc[s]["_date"]), []).append(s)
                config = {}
                for day, ss in by_day.items():
                    ss = sorted(ss)
                    first = g[g.session == ss[0]].set_index("object_label")[["x_cm", "y_cm"]]
                    for k, s in enumerate(ss):
                        cur = g[g.session == s].set_index("object_label")[["x_cm", "y_cm"]]
                        shared = first.index.intersection(cur.index)
                        moved = float(np.max(np.linalg.norm(cur.loc[shared].to_numpy() - first.loc[shared].to_numpy(), axis=1))) if (k and len(shared)) else 0.0
                        changed = bool(k and (set(cur.index) != set(first.index) or moved > 5))
                        config[int(s)] = dict(day=day, labels=sorted(cur.index), shared_with_first=sorted(shared),
                                              moved_from_first_cm=round(moved, 1), changed=changed,
                                              config="config1" if k == 0 else "config2",
                                              sheet=str(rows_.iloc[s].get("Structure", "")))
                        if k and not changed:
                            checks["problems"].append(f"{recday} s{s}: same objects at the same places as the day's first session")
                rec["config"] = config
            else:
                if any(len(v) != 1 for v in per_sess.values()):
                    checks["problems"].append(f"{recday}: expected one object per mEC session, got {per_sess}")
                pos = g.groupby("session")[["x_cm", "y_cm"]].first()
                if len(pos) > 1:
                    d = np.linalg.norm(np.diff(pos.to_numpy(), axis=0), axis=1)
                    rec["moves_cm"] = d.round(1).tolist()
            checks["objects"][recday] = rec

        if args.write:
            out = PD / f"object_positions_{rr.DATASET}_cm.csv"
            n = 0
            while (p := out.with_name(out.stem + (f".v{n}" if n else "") + ".csv")).exists():
                n += 1
            obj.to_csv(p, index=False)
            print(f"objects in cm -> {p}")
            # contact sheet
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            frames = obj.frame.unique()
            cols = 4
            rows_n = int(np.ceil(len(frames) / cols))
            fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.4 * max(rows_n, 1)))
            for a, f in zip(np.atleast_1d(axes).ravel(), frames):
                a.imshow(mpimg.imread(FRAMES / f))
                a.set_xticks([]); a.set_yticks([]); a.set_title(f.replace("_median.png", ""), fontsize=7)
                for _, r in obj[obj.frame == f].iterrows():
                    a.plot(r.x_px, r.y_px, "r+", ms=10, mew=1.5)
                    a.text(r.x_px, r.y_px - 20, f"{r.object_label} ({r.x_cm:.0f},{r.y_cm:.0f})", color="w", fontsize=6, ha="center")
            for a in np.atleast_1d(axes).ravel()[len(frames):]:
                a.axis("off")
            sheet = PD / f"object_contact_sheet_{rr.DATASET}.png"
            n = 0
            while (p := sheet.with_name(sheet.stem + (f".v{n}" if n else "") + ".png")).exists():
                n += 1
            fig.savefig(p, dpi=90, bbox_inches=None)
            print(f"contact sheet -> {p}")

    if args.write:
        out = PD / f"object_checks_{rr.DATASET}.json"
        n = 0
        while (p := out.with_name(out.stem + (f".v{n}" if n else "") + ".json")).exists():
            n += 1
        p.write_text(json.dumps(checks, indent=1, default=str))
        print(f"checks -> {p}")
    print(json.dumps(checks["arenas"], indent=1, default=str))
    print(f"{len(checks['problems'])} problem(s):")
    for pr in checks["problems"]:
        print("  -", pr)


if __name__ == "__main__":
    main()
