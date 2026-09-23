"""
Arena calibration: camera pixels -> cm on the open-field floor, with radial correction.

Two arenas, one interface:

- **mEC big field** (`arena='bigfield'`): a plain 120 cm square floor laid over the 7x7 peg
  maze. The only pixel<->world data is the 13-peg registration `maze_registration.htsv`
  (metres, 0.18 m pitch). `cv2.calibrateCamera` on that single planar view (fx=fy, principal
  point fixed at the image centre, k1 and k2 only -- 13 points do not constrain more), then
  `undistortPoints`, then ONE homography to cm. The pegs lie below the floor, so the scale is
  off by D/(D-h); `apply_arena_corners` fixes it from the clicked corners and the known 120 cm
  side, and re-origins the frame to the arena.
- **LEC box** (`arena='openfield_box1'|'openfield_box2'`): a square box standing on the maze
  under the same camera, so px->cm is that box's maze affine (47 cm, `sleap_to_rois` corners).
  No distortion term to start with; `fit_k1_from_walls` adds one if the straight-wall test fails.

The acceptance test for either is `straight_wall_test`: the wall-hugging edge of the tracked
occupancy, in cm, must be straight (RMS residual < 1 cm per wall) and the clicked corners must
form a square (sides within 2 %, angles within 1 deg). Numbers are stored in the JSON so the
data note can quote them.

I/O: JSON at `data/metadata/arena_calibration_{dataset}_{arena}.json` (never overwritten;
a changed calibration goes beside as .vN). `to_cm(px)`, `to_px(cm)` are the runtime API.
Needs cv2 (the `histology` conda env); numpy-only fallback for affine arenas.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BIG_FIELD_REGISTRATION = Path("/ceph/behrens/adam_harris/Taskspace_abstraction_mEC/mEC_local_upload/"
                              "mEC_preprocessing/maze_registration.htsv")
IMAGE_SIZE = (1280, 1024)
BIG_FIELD_SIDE_CM = 120.0          # user, 2026-09-18
LEC_MAZE_SIDE_CM = 47.0
#: LEC object box: 49.9-50.2 cm mean side under the maze affine on all 20 sessions (2026-09-18);
#: a 50 cm box. The affine shows it as a 4 % trapezoid (bottom 51.2 cm, top 49.1 cm: camera
#: tilt, not lens distortion -- no k1 straightens the walls, lec_box_distortion.py), so each
#: session gets its own homography from its clicked corners to this square.
LEC_BOX_SIDE_CM = 50.0


@dataclass
class ArenaCalibration:
    dataset: str
    arena: str
    image_size: tuple
    #: 3x3 camera matrix and distortion coefficients (k1, k2, p1, p2, k3); None = no distortion
    K: Optional[list] = None
    dist: Optional[list] = None
    #: 3x3 homography from UNDISTORTED pixels to cm in the calibration frame
    H: list = field(default_factory=list)
    #: rigid+scale correction from the calibration frame to the arena frame, set by
    #: apply_arena_corners: cm_arena = plane_factor * R @ (cm_calib - origin)
    plane_factor: Optional[float] = None
    arena_rotation_deg: Optional[float] = None
    arena_origin_cm: Optional[list] = None
    arena_side_cm: Optional[float] = None
    corners_px: Optional[list] = None
    corners_cm: Optional[list] = None
    #: quality numbers
    reprojection_rms_px: Optional[float] = None
    n_points: Optional[int] = None
    wall_rms_cm: Optional[dict] = None
    square_side_cm: Optional[list] = None
    square_angles_deg: Optional[list] = None
    source: str = ""
    notes: list = field(default_factory=list)

    # ---- runtime -------------------------------------------------------------------
    def undistort(self, px: np.ndarray) -> np.ndarray:
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        if self.dist is None:
            return px
        import cv2
        K = np.asarray(self.K, float)
        out = cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), K,
                                  np.asarray(self.dist, float), P=K)
        return out.reshape(-1, 2)

    def redistort(self, px_undist: np.ndarray) -> np.ndarray:
        px_undist = np.asarray(px_undist, dtype=float).reshape(-1, 2)
        if self.dist is None:
            return px_undist
        import cv2
        K = np.asarray(self.K, float)
        norm = np.linalg.solve(K, np.c_[px_undist, np.ones(len(px_undist))].T).T
        pts3 = np.c_[norm[:, :2] / norm[:, 2:3], np.ones(len(norm))]
        out, _ = cv2.projectPoints(pts3.astype(np.float64), np.zeros(3), np.zeros(3), K,
                                   np.asarray(self.dist, float))
        return out.reshape(-1, 2)

    def to_cm(self, px: np.ndarray) -> np.ndarray:
        """Pixels -> cm in the arena frame (calibration frame if no corners applied). NaN-safe."""
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        out = np.full_like(px, np.nan)
        ok = ~np.isnan(px).any(axis=1)
        if ok.any():
            u = self.undistort(px[ok])
            h = np.asarray(self.H, float) @ np.c_[u, np.ones(len(u))].T
            cm = (h[:2] / h[2]).T
            out[ok] = self._calib_to_arena(cm)
        return out

    def to_px(self, cm: np.ndarray) -> np.ndarray:
        cm = np.asarray(cm, dtype=float).reshape(-1, 2)
        calib = self._arena_to_calib(cm)
        h = np.linalg.inv(np.asarray(self.H, float)) @ np.c_[calib, np.ones(len(calib))].T
        return self.redistort((h[:2] / h[2]).T)

    def _calib_to_arena(self, cm):
        if self.plane_factor is None:
            return cm
        th = np.deg2rad(self.arena_rotation_deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return self.plane_factor * (R @ (cm - np.asarray(self.arena_origin_cm)).T).T

    def _arena_to_calib(self, cm):
        if self.plane_factor is None:
            return cm
        th = np.deg2rad(self.arena_rotation_deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return (R.T @ (cm / self.plane_factor).T).T + np.asarray(self.arena_origin_cm)

    # ---- I/O -----------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        text = json.dumps(asdict(self), indent=1)
        if path.exists():
            if path.read_text() == text:
                return path
            n = 1
            while (alt := path.with_name(f"{path.stem}.v{n}.json")).exists():
                n += 1
            path = alt
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    @classmethod
    def load(cls, path: Path) -> "ArenaCalibration":
        d = json.loads(Path(path).read_text())
        d["image_size"] = tuple(d["image_size"])
        return cls(**d)


def calibration_path(dataset_root: Path, dataset: str, arena: str) -> Path:
    return Path(dataset_root) / "data" / "metadata" / f"arena_calibration_{dataset}_{arena}.json"


def latest_calibration(dataset_root: Path, dataset: str, arena: str) -> ArenaCalibration:
    """The highest .vN version if any, else the base file."""
    base = calibration_path(dataset_root, dataset, arena)
    cands = sorted(base.parent.glob(f"{base.stem}*.json"),
                   key=lambda p: (0 if p == base else int(p.stem.rsplit(".v", 1)[1])))
    if not cands:
        raise FileNotFoundError(base)
    return ArenaCalibration.load(cands[-1])


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def read_big_field_registration(path: Path = BIG_FIELD_REGISTRATION) -> tuple[np.ndarray, np.ndarray, list]:
    """(pixel (13,2), world cm (13,2), labels) from the 7x7 peg registration (metres -> cm)."""
    t = pd.read_csv(path, sep="\t")
    px = np.array([ast.literal_eval(s) for s in t["pixel_coords"]], float)
    world = np.array([ast.literal_eval(s) for s in t["physical_coords"]], float) * 100.0
    return px, world, t["alignment_point"].tolist()


def fit_planar_with_distortion(px: np.ndarray, world_cm: np.ndarray, image_size=IMAGE_SIZE) -> dict:
    """Single-view planar calibration: fx=fy, fixed principal point, k1+k2 only; then H.

    13 points cannot support the notebook's free 5-coefficient model, so the model is
    restricted to what the data constrain. Returns K, dist, H and the reprojection RMS of
    world -> pixel through the full model.
    """
    import cv2
    obj = np.c_[world_cm, np.zeros(len(world_cm))].astype(np.float32).reshape(1, -1, 3)
    img = px.astype(np.float32).reshape(1, -1, 2)
    K0 = np.array([[0.8 * image_size[0], 0, image_size[0] / 2],
                   [0, 0.8 * image_size[0], image_size[1] / 2], [0, 0, 1]], float)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    rms, K, dist, _, _ = cv2.calibrateCamera([obj], [img], tuple(image_size), K0, np.zeros(5), flags=flags)
    und = cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(-1, 2)
    H, _ = cv2.findHomography(und.astype(np.float64), world_cm.astype(np.float64), 0)   # least squares
    cal = ArenaCalibration("", "", tuple(image_size), K=K.tolist(), dist=dist.ravel().tolist(), H=H.tolist())
    back = cal.to_px(world_cm)
    rep = float(np.sqrt(np.mean(np.sum((back - px) ** 2, axis=1))))
    return dict(K=K.tolist(), dist=dist.ravel().tolist(), H=H.tolist(), reprojection_rms_px=rep,
                calibrate_rms=float(rms))


def fit_affine(px: np.ndarray, world_cm: np.ndarray) -> dict:
    """Least-squares affine as a homography with a zero perspective row (the maze method)."""
    n = len(px)
    M = np.zeros((2 * n, 6))
    v = np.zeros(2 * n)
    for i, ((x, y), (wx, wy)) in enumerate(zip(px, world_cm)):
        M[2 * i, :3] = (x, y, 1.0)
        v[2 * i] = wx
        M[2 * i + 1, 3:] = (x, y, 1.0)
        v[2 * i + 1] = wy
    a11, a12, bx, a21, a22, by = np.linalg.lstsq(M, v, rcond=None)[0]
    H = [[a11, a12, bx], [a21, a22, by], [0.0, 0.0, 1.0]]
    cal = ArenaCalibration("", "", IMAGE_SIZE, H=H)
    back = cal.to_px(world_cm)
    return dict(H=H, reprojection_rms_px=float(np.sqrt(np.mean(np.sum((back - px) ** 2, axis=1)))))


def big_field_calibration() -> ArenaCalibration:
    px, world, labels = read_big_field_registration()
    fit = fit_planar_with_distortion(px, world)
    cal = ArenaCalibration("mec", "bigfield", IMAGE_SIZE, K=fit["K"], dist=fit["dist"], H=fit["H"],
                           reprojection_rms_px=fit["reprojection_rms_px"], n_points=len(px),
                           source=f"{BIG_FIELD_REGISTRATION.name}: 13 pegs of the 7x7 grid, metres x100; "
                                  "cv2.calibrateCamera fx=fy, fixed principal point, k1 k2; H least squares",
                           notes=[f"calibrateCamera rms {fit['calibrate_rms']:.2f} px",
                                  "scale is the PEG plane until apply_arena_corners sets plane_factor"])
    return cal


def lec_box_calibration(box: int) -> ArenaCalibration:
    from sleap_to_rois import LEC_MAZE1, LEC_MAZE2, PHYSICAL_PTS
    px = LEC_MAZE1 if box == 1 else LEC_MAZE2
    fit = fit_affine(px, PHYSICAL_PTS)
    return ArenaCalibration("lec", f"openfield_box{box}", IMAGE_SIZE, H=fit["H"],
                            reprojection_rms_px=fit["reprojection_rms_px"], n_points=4,
                            source=f"maze-{box} corner affine (sleap_to_rois), 47 cm; box stands on the maze, same camera",
                            notes=["no distortion term; add one with fit_k1_from_walls if the walls curve"])


# ---------------------------------------------------------------------------
# arena corners and checks
# ---------------------------------------------------------------------------

def apply_arena_corners_homography(cal: ArenaCalibration, corners_px: np.ndarray, side_cm: float) -> ArenaCalibration:
    """Replace the homography with the floor's own: undistorted clicked corners -> a `side_cm`
    square, keeping the lens model (K, dist) from the peg fit.

    The pegs lie below the open-field floor; a homography fitted on them maps the raised floor
    with a perspective error (the clicked corners came out 117-122 cm and 88-92 deg under a
    pure rescale). Four coplanar corners define the floor's homography exactly, so the corners
    are a square by construction and the calibration is judged by `straight_wall_test` and the
    tracked occupancy instead. Corner order in the arena frame: image top-left (0,0), top-right
    (L,0), bottom-right (L,L), bottom-left (0,L) -- image handedness, y down, like the video.
    """
    import cv2
    px = np.asarray(corners_px, float)
    c = px.mean(axis=0)
    ang = np.arctan2(px[:, 1] - c[1], px[:, 0] - c[0])            # image coords: y down
    order = np.argsort(ang)                                        # -pi..pi: TL? sort then rotate to start at TL
    px = px[order]
    tl = int(np.argmin(px[:, 0] + px[:, 1]))
    px = np.roll(px, -tl, axis=0)
    world = np.array([[0, 0], [side_cm, 0], [side_cm, side_cm], [0, side_cm]], float)
    und = cal.undistort(px)
    H, _ = cv2.findHomography(und.astype(np.float64), world, 0)
    new = ArenaCalibration(**{**asdict(cal)})
    new.H = H.tolist()
    new.plane_factor = None
    new.arena_rotation_deg = None
    new.arena_origin_cm = None
    new.arena_side_cm = float(side_cm)
    new.corners_px = px.tolist()
    new.corners_cm = new.to_cm(px).tolist()
    new.square_side_cm = np.linalg.norm(np.roll(new.to_cm(px), -1, axis=0) - new.to_cm(px), axis=1).tolist()
    new.square_angles_deg = [90.0] * 4
    new.notes = list(cal.notes) + ["homography refitted on the 4 clicked floor corners (undistorted) to a "
                                   f"{side_cm:.0f} cm square; lens model from the peg fit; peg-plane H replaced"]
    return new


def apply_arena_corners(cal: ArenaCalibration, corners_px: np.ndarray, side_cm: float,
                        rescale: bool = True) -> ArenaCalibration:
    """Fit a square to the four clicked corners (in calibration cm), set origin, rotation and,
    if `rescale`, the plane-height factor so the square's mean side equals `side_cm`.

    Corner order: any; they are sorted counter-clockwise from the lower-left in cm. Kept for
    the LEC boxes (scale from the maze affine); the mEC big field uses
    `apply_arena_corners_homography` because its pegs are on a lower plane.
    """
    cm = cal.to_cm(corners_px) if cal.plane_factor is None else cal._arena_to_calib(cal.to_cm(corners_px))
    c = cm.mean(axis=0)
    order = np.argsort(np.arctan2(cm[:, 1] - c[1], cm[:, 0] - c[0]))
    cm = cm[order]
    sides = np.linalg.norm(np.roll(cm, -1, axis=0) - cm, axis=1)
    vecs = np.roll(cm, -1, axis=0) - cm
    angles = [float(np.degrees(np.arccos(np.clip(np.dot(vecs[i], -vecs[i - 1]) /
                                                 (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i - 1])), -1, 1))))
              for i in range(4)]
    rot = -np.degrees(np.arctan2(vecs[0][1], vecs[0][0]))            # make side 0 the x axis
    factor = side_cm / float(sides.mean()) if rescale else 1.0
    new = ArenaCalibration(**{**asdict(cal)})
    new.plane_factor = float(factor)
    new.arena_rotation_deg = float(rot)
    new.arena_origin_cm = cm[0].tolist()
    new.arena_side_cm = float(side_cm)
    new.corners_px = np.asarray(corners_px, float)[order].tolist()
    new.square_side_cm = (sides * factor).tolist()
    new.square_angles_deg = angles
    new.corners_cm = new.to_cm(np.asarray(corners_px, float)[order]).tolist()
    new.notes = list(cal.notes) + [f"corners applied: measured sides {np.round(sides, 1).tolist()} cm in the "
                                    f"calibration frame, plane factor {factor:.4f}, rotation {rot:.2f} deg"]
    return new


def straight_wall_test(xy_cm: np.ndarray, band_frac: float = 0.04, n_slices: int = 40,
                       percentile: float = 1.0) -> dict:
    """RMS residual (cm) of a line fitted to the occupancy edge along each of the four walls.

    For the left wall: slice y into `n_slices` bins, take the `percentile`-th x in each slice
    (the wall-hugging edge), fit x = a*y + b, report the RMS residual. Same for the others.
    Only slices with >= 20 points count. A curved wall (uncorrected barrel distortion) shows as
    a residual of several cm; a straight one stays under ~1 cm at 60 fps occupancy.
    """
    xy = np.asarray(xy_cm, float)
    xy = xy[~np.isnan(xy).any(axis=1)]
    out = {}
    for name, axis in (("left", 0), ("right", 0), ("bottom", 1), ("top", 1)):
        other = 1 - axis
        lo, hi = np.percentile(xy[:, other], [2, 98])
        edges = np.linspace(lo, hi, n_slices + 1)
        pts = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (xy[:, other] >= a) & (xy[:, other] < b)
            if sel.sum() < 20:
                continue
            q = percentile if name in ("left", "bottom") else 100 - percentile
            pts.append(((a + b) / 2, np.percentile(xy[sel, axis], q)))
        if len(pts) < 5:
            out[name] = None
            continue
        p = np.array(pts)
        coef = np.polyfit(p[:, 0], p[:, 1], 1)
        resid = p[:, 1] - np.polyval(coef, p[:, 0])
        out[name] = dict(rms_cm=float(np.sqrt(np.mean(resid ** 2))), n_slices=len(p),
                         slope_deg=float(np.degrees(np.arctan(coef[0]))), intercept_cm=float(coef[1]))
    return out


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="fit and store the base arena calibrations")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mec_root = Path("/ceph/behrens/adam_harris/Taskspace_abstraction_mEC")
    lec_root = Path(__file__).resolve().parents[2]
    cals = [(mec_root, big_field_calibration()), (lec_root, lec_box_calibration(1)), (lec_root, lec_box_calibration(2))]
    for root, cal in cals:
        px, world, labels = (read_big_field_registration() if cal.arena == "bigfield" else (None, None, None))
        print(f"{cal.dataset}/{cal.arena}: reprojection rms {cal.reprojection_rms_px:.2f} px over {cal.n_points} pts"
              + (f"; dist k1,k2 = {cal.dist[0]:.4f}, {cal.dist[1]:.4f}; fx {cal.K[0][0]:.0f}" if cal.dist else ""))
        if cal.arena == "bigfield":
            back = cal.to_cm(px)
            err = np.linalg.norm(back - world, axis=1)
            print("   per-peg error cm:", dict(zip(labels, np.round(err, 2))))
        if args.write:
            p = cal.save(calibration_path(root, cal.dataset, cal.arena))
            print("   ->", p)
