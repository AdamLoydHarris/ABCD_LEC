"""
SLEAP h5 -> per-session ROI, cleaned-coordinate and head-direction CSVs (Stage B1 of the mEC port).

Headless version of the registration cell in `sleap_preprocess_lEC.ipynb` (itself the mEC
notebook's cell), with the same numerics, writing the three CSVs `sleap_alignment_copilot`
reads:

    data/preprocessed/sleap/maze/{mouse}/{mouse}_{YYYY-MM-DD-HHMMSS}.analysis_ROIs.csv
    ..._cleaned_coordinates.csv      (all bodyparts, in cm after the affine fit)
    ..._head_direction.csv           (back2mid_deg, earL2earR_deg)

Per file: read the single-instance SLEAP `tracks` (n_tracks, 2, n_nodes, T); keep a raw copy
for the mask lookup; drop frame-to-frame jumps above median + 3 MAD (threshold from the FIRST
bodypart, reused for the rest -- that is what the notebook did, because it assigned the
computed threshold back to its own argument); interpolate both directions; centred rolling
mean over 5 frames; least-squares affine from the four hand-picked maze corners to the
47 x 47 cm square, applied to every bodypart; head direction from the cm coordinates; and
the ROI label of every bodypart from the raw pixel position against the 21 TIFF masks.

Geometry per mouse is in `geometry()`. bp02 is given bp04's maze-2 camera geometry, validated
by `poke_calibration.py` (98 % frame-wise mask agreement). Existing outputs are never
overwritten, and nothing here writes under `data/raw/`.

Geometry versions (`--geometry`, default v1). The notebook listed the maze-2 corners in Z order
(node 1, 3, 7, 9) against the cyclic `PHYSICAL_PTS`, which no affine can realise: det(A) = -7e-5,
corner residual 23 cm, and every bp02/bp04 cm track had head_back.x pinned to 22.8-24.2 cm with
head direction at +-90 deg on 87-89 % of frames (measured 2026-09-23 on all 69 sessions).
`v1` reproduces those CSVs byte for byte in `maze/`; `v2` uses the cyclic order (node mask
centroids then land within 1 cm of the maze-1 and LEC positions) and writes to `maze_geom2/`.
`affine()` refuses a degenerate corner set unless `check=False`, which only the v1 path passes.

Cleaning versions (`--clean`, default v1):
  v1    the notebook numerics above, Locs from raw pixels. Byte-identical to `maze/`.
  none  no cleaning at all: raw pixels through the affine (what the LEC `maze/` CSVs are;
        TRACKING_QC_PLAN.md). Writing is refused; `--compare-existing` is its purpose.
  v2    `tracking_clean.clean_v2`: physical rules in cm (100 cm/s velocity cap, out-and-back
        spike, Hampel 11 / 3 cm), a conditional confidence gate on the SLEAP scores, straight
        bridging of gaps <= 2 s, re-masked 5-frame mean; head direction and the ROI labels from
        the CLEANED track (user, 2026-09-23; the cleaned cm track is inverted through the affine
        for the mask lookup); a per-frame provenance file `*_track_flags.csv` (one int8 per
        bodypart, codes in tracking_clean). Output `maze_v2/`; implies `--geometry v2`.
`--compare-existing` recomputes every session in memory and compares it with the CSVs in the
reference tree (v1 -> maze/, none -> maze/): bytes, max |diff|, NaN pattern, ROI labels; exit 1
on any difference. Nothing is written in that mode.

Usage::

    python sleap_to_rois.py --dataset mec                    # dry run: list what would be written
    python sleap_to_rois.py --dataset mec --write [--mouse bp04] [--limit 1]
    python sleap_to_rois.py --dataset mec --geometry v2 --write   # -> data/preprocessed/sleap/maze_geom2/
    python sleap_to_rois.py --dataset mec --clean v1 --compare-existing        # 102 files vs maze/
    python sleap_to_rois.py --dataset lec --clean none --compare-existing      # 461 files vs maze/
    python sleap_to_rois.py --dataset {mec,lec} --clean v2 --write            # -> maze_v2/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path

# `--dataset` must reach recday_registry before it is imported.
if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import h5py
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
import tracking_clean as tclean                                       # noqa: E402

RAW_SLEAP = rr.REPO_ROOT / "data" / "raw" / "sleap"
SLEAP_ROOT = rr.REPO_ROOT / "data" / "preprocessed" / "sleap"
#: Output root per geometry version; `maze/` is the tree every existing data_dic was built from.
OUT_ROOTS = {"v1": SLEAP_ROOT / "maze", "v2": SLEAP_ROOT / "maze_geom2"}
OUT_ROOT = OUT_ROOTS["v1"]
OUT_ROOT_CLEAN_V2 = SLEAP_ROOT / "maze_v2"
CLEAN_VERSION = "v1"                 # 'v1' | 'none' | 'v2', set by --clean
MASKS_ROOT = {"mec": rr.REPO_ROOT / "code" / "masks",
              "lec": rr.REPO_ROOT / "code" / "preprocessing" / "masks"}[rr.DATASET]

#: 'v1' reproduces the CSVs on disk (including the degenerate maze-2 affine); 'v2' is the fix.
GEOMETRY_VERSION = "v1"

#: Maze corners in cm, cyclic: node 1 (0,0), node 3 (47,0), node 9 (47,47), node 7 (0,47).
PHYSICAL_PTS = np.array([[0, 0], [47, 0], [47, 47], [0, 47]], dtype=float)

#: Hand-picked pixel corners, verbatim from `sleap_preprocess_mEC.ipynb`, in PHYSICAL_PTS order.
MAZE1_EARLY = np.array([[1208, 840], [989, 25], [184, 213], [376, 1030]], dtype=float)
MAZE1_LATE = np.array([[1109, 826], [908, -5], [72, 174], [254, 1028]], dtype=float)
LEC_MAZE1 = np.array([[1022, 974], [981, 136], [158, 169], [172, 1010]], dtype=float)
LEC_MAZE2 = np.array([[234, 945], [1036, 953], [1020, 158], [248, 169]], dtype=float)
#: Maze 2 as the notebook had it: Z order (BL, BR, TL, TR) against the cyclic PHYSICAL_PTS. The
#: least-squares affine degenerates (det -7e-5, corner residual 23 cm) and collapses x to a 1.3 cm
#: band; kept verbatim so `--geometry v1` reproduces `maze/`. Found 2026-09-23.
MAZE2_MEC_ZORDER_LEGACY = np.array([[256, 944], [1034, 944], [260, 182], [1009, 166]], dtype=float)
#: The same four pixels in cyclic order (det -3.8e-3, residual 0.52 cm; node mask centroids land
#: within 1 cm of the maze-1 and LEC positions).
MAZE2_MEC = np.array([[256, 944], [1034, 944], [1009, 166], [260, 182]], dtype=float)

#: bp01's camera moved between 2024-03-25 and 2024-03-28 (the 26th "is all over the place").
BP01_SWITCH = date(2024, 3, 25)

#: A corner set is degenerate when the fitted affine misses its own corners by more than this or
#: its Jacobian is (near) singular. Good sets: residual 0.38-0.55 cm, |det| 2.9e-3 to 3.8e-3.
AFFINE_MAX_RESID_CM = 2.0
AFFINE_MIN_ABS_DET = 1e-4


def geometry(mouse: str, session_date: date) -> tuple[str, np.ndarray, str]:
    """(mask directory name, pixel corners, note) for one session under GEOMETRY_VERSION."""
    if rr.DATASET == "mec":
        if mouse == "bp01":
            return (("mEC_maze1_early", MAZE1_EARLY, "") if session_date <= BP01_SWITCH
                    else ("mEC_maze1_late", MAZE1_LATE, ""))
        if mouse in ("bp04", "bp02"):
            if GEOMETRY_VERSION == "v1":
                corners, note = MAZE2_MEC_ZORDER_LEGACY, "DEGENERATE GEOMETRY (legacy Z-order corners, x collapsed)"
            else:
                corners, note = MAZE2_MEC, ""
            if mouse == "bp02":
                note = (note + "; " if note else "") + "bp04 geometry (poke_calibration: 98 % mask agreement)"
            return "mEC_maze2", corners, note
    if rr.DATASET == "lec":
        # Box 2 (ah08, ly06) and box 1 (ah10, ly05, ly07), as in openfield_to_xy.arena_for. The LEC
        # maze CSVs on disk are the raw affine with no cleaning (TRACKING_QC_PLAN.md); this branch
        # serves the QC and `--clean none|v2`, never a v1-cleaned rewrite of `maze/`.
        if mouse in ("ah08", "ly06"):
            return "lEC_maze2", LEC_MAZE2, ""
        if mouse in ("ah10", "ly05", "ly07"):
            return "lEC_maze1", LEC_MAZE1, ""
    raise KeyError(f"no geometry for {rr.DATASET}/{mouse}")


# ---------------------------------------------------------------------------
# numerics, matching the notebook
# ---------------------------------------------------------------------------

def load_tracks(h5_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Single-instance SLEAP tracks -> DataFrame of `{bp}.x`, `{bp}.y` columns in node order."""
    with h5py.File(h5_path, "r") as f:
        nodes = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]
        tracks = f["tracks"][:]                               # (n_tracks, 2, n_nodes, T)
    if tracks.shape[0] != 1:
        raise ValueError(f"{h5_path.name}: {tracks.shape[0]} tracks; expected single-instance output")
    cols = {}
    for i, bp in enumerate(nodes):
        cols[f"{bp}.x"] = tracks[0, 0, i, :]
        cols[f"{bp}.y"] = tracks[0, 1, i, :]
    return pd.DataFrame(cols), nodes


def drop_jumps(df: pd.DataFrame, nodes: list[str], multiplier: float = 3.0) -> pd.DataFrame:
    """NaN the destination frame of any jump above median + multiplier * MAD.

    The threshold is computed on the first bodypart and reused for all of them, reproducing
    the notebook (it wrote the computed value back into its `distance_threshold` argument).
    """
    out = df.copy()
    threshold = None
    for bp in nodes:
        x, y = out[f"{bp}.x"].to_numpy().copy(), out[f"{bp}.y"].to_numpy().copy()
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        if threshold is None:
            good = dist[~np.isnan(dist)]
            if len(good) == 0:
                continue
            med = np.median(good)
            mad = np.median(np.abs(good - med))
            threshold = med * multiplier if mad == 0 else med + multiplier * mad
        bad = np.where(dist > threshold)[0] + 1
        x[bad] = np.nan
        y[bad] = np.nan
        out[f"{bp}.x"], out[f"{bp}.y"] = x, y
    return out


def interpolate_smooth(df: pd.DataFrame, nodes: list[str], window: int = 5) -> pd.DataFrame:
    for bp in nodes:
        for c in (f"{bp}.x", f"{bp}.y"):
            df[c] = df[c].interpolate(method="linear", limit_direction="both")
            df[c] = df[c].rolling(window=window, min_periods=1, center=True).mean()
    return df


def affine(pixel_pts: np.ndarray, physical_pts: np.ndarray, check: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares 2x2 A and offset b with physical = A @ pixel + b.

    Raises ValueError on a degenerate corner set (see AFFINE_MAX_RESID_CM / AFFINE_MIN_ABS_DET)
    unless `check=False`; only the `--geometry v1` reproduction path passes that.
    """
    n = len(pixel_pts)
    M = np.zeros((2 * n, 6))
    v = np.zeros(2 * n)
    for i, ((px, py), (wx, wy)) in enumerate(zip(pixel_pts, physical_pts)):
        M[2 * i, :3] = (px, py, 1.0)
        v[2 * i] = wx
        M[2 * i + 1, 3:] = (px, py, 1.0)
        v[2 * i + 1] = wy
    a11, a12, bx, a21, a22, by = np.linalg.lstsq(M, v, rcond=None)[0]
    A, b = np.array([[a11, a12], [a21, a22]]), np.array([bx, by])
    if check:
        resid, det = affine_quality(A, b, pixel_pts, physical_pts)
        if resid > AFFINE_MAX_RESID_CM or abs(det) < AFFINE_MIN_ABS_DET:
            raise ValueError(f"degenerate maze affine: corner residual {resid:.1f} cm, det {det:.2e} "
                             f"(corners {pixel_pts.tolist()}); check the corner order against PHYSICAL_PTS")
    return A, b


def affine_quality(A: np.ndarray, b: np.ndarray, pixel_pts: np.ndarray,
                   physical_pts: np.ndarray) -> tuple[float, float]:
    """(max corner residual in cm, det(A)) of a fitted affine."""
    fit = (A @ np.asarray(pixel_pts, float).T).T + b
    return float(np.linalg.norm(fit - physical_pts, axis=1).max()), float(np.linalg.det(A))


def to_cm(df: pd.DataFrame, nodes: list[str], A: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    for bp in nodes:
        xy = df[[f"{bp}.x", f"{bp}.y"]].to_numpy()
        cm = (A @ xy.T).T + b
        df[f"{bp}.x"], df[f"{bp}.y"] = cm[:, 0], cm[:, 1]
    return df


def cm_to_px(df_cm: pd.DataFrame, nodes: list[str], A: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """Inverse of `to_cm` (exact for an affine), new DataFrame; used to label ROIs from a cleaned track."""
    Ainv = np.linalg.inv(A)
    out = {}
    for bp in nodes:
        cm = df_cm[[f"{bp}.x", f"{bp}.y"]].to_numpy(float)
        px = (Ainv @ (cm - b).T).T
        out[f"{bp}.x"], out[f"{bp}.y"] = px[:, 0], px[:, 1]
    return pd.DataFrame(out, index=df_cm.index)


def head_direction(df: pd.DataFrame) -> pd.DataFrame:
    """back2mid_deg = angle head_back -> head_mid; earL2earR_deg = angle ear_L -> ear_R."""
    need = ["head_mid", "head_back", "ear_L", "ear_R"]
    cols = [f"{p}{s}" for p in need for s in (".x", ".y")]
    nan_row = df[cols].isna().any(axis=1).to_numpy()
    b2m = np.degrees(np.arctan2(df["head_mid.y"] - df["head_back.y"], df["head_mid.x"] - df["head_back.x"]))
    l2r = np.degrees(np.arctan2(df["ear_R.y"] - df["ear_L.y"], df["ear_R.x"] - df["ear_L.x"]))
    out = pd.DataFrame({"back2mid_deg": b2m.to_numpy(), "earL2earR_deg": l2r.to_numpy()})
    out[nan_row] = np.nan
    return out


#: Which mask wins where a node mask and an edge mask overlap: sorted name order (`edge_*` before
#: `node_*`) for both datasets. The mEC CSVs were made this way. The LEC ROI CSVs (notebook) match
#: neither this nor node-first exactly -- 0.06-0.8 % of head_back frames per session differ,
#: all within 0.7 px of a mask edge (ah08 closer to node-first, ly05 closer to sorted), i.e. the
#: notebook iterated the TIFFs in filesystem order. Not reproducible, not worth reproducing:
#: `--clean v2` labels ROIs from the cleaned track anyway. `--clean none --compare-existing`
#: reports the gap as `cmp_roi_diff`.
MASK_PRIORITY = "sorted"


def label_image(mask_dir: Path, priority: str = MASK_PRIORITY) -> tuple[np.ndarray, list[str], int]:
    """Stack the 21 binary masks into one int label image.

    `priority='sorted'`: the first mask in sorted name order wins on overlap (edges before
    nodes); `'node_first'`: node masks are laid down first and win. Returns (labels, names,
    n_overlap_px). Label 0 = no mask; name index = label - 1, names in sorted order either way.
    """
    tifs = sorted(mask_dir.glob("*.tif*"))
    names = [t.stem for t in tifs]
    order = tifs if priority == "sorted" else ([t for t in tifs if t.stem.startswith("node")]
                                              + [t for t in tifs if not t.stem.startswith("node")])
    labels, overlap = None, 0
    for tif in order:
        mask = tifffile.imread(tif).astype(bool)
        if labels is None:
            labels = np.zeros(mask.shape, dtype=np.int16)
        claimed = mask & (labels > 0)
        overlap += int(claimed.sum())
        labels[mask & (labels == 0)] = names.index(tif.stem) + 1
    return labels, names, overlap


def roi_labels(raw: pd.DataFrame, nodes: list[str], labels: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-frame ROI name for every bodypart from its raw pixel position; 'None' if NaN/off-image/unmasked."""
    h, w = labels.shape
    lut = np.array(["None"] + names, dtype=object)
    out = {}
    for bp in nodes:
        x, y = raw[f"{bp}.x"].to_numpy(), raw[f"{bp}.y"].to_numpy()
        ok = ~(np.isnan(x) | np.isnan(y))
        xi = np.zeros(len(x), dtype=int)
        yi = np.zeros(len(y), dtype=int)
        xi[ok] = np.rint(x[ok]).astype(int)
        yi[ok] = np.rint(y[ok]).astype(int)
        inside = ok & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        lab = np.zeros(len(x), dtype=int)
        lab[inside] = labels[yi[inside], xi[inside]]
        out[bp] = lut[lab]
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def session_files(mouse: str) -> list[tuple[Path, str, date]]:
    """(h5, stem 'bp02_2024-03-24-105214', session date) for every raw SLEAP h5 of a mouse."""
    out = []
    for h5 in sorted((RAW_SLEAP / mouse / "maze").glob("*.analysis.h5")):
        m = re.search(rf"({mouse}_(\d{{4}})-(\d{{2}})-(\d{{2}})-\d{{6}})\.analysis\.h5$", h5.name)
        if not m:
            continue
        out.append((h5, m.group(1), date(int(m.group(2)), int(m.group(3)), int(m.group(4)))))
    return out


def guard_output_dir(out_dir: Path) -> None:
    """Refuse to write under data/raw, or into a tree that belongs to another geometry/cleaning mode."""
    out_dir = out_dir.resolve()
    if RAW_SLEAP.resolve() in (out_dir, *out_dir.parents):
        raise ValueError(f"refusing to write under the raw SLEAP tree: {out_dir}")
    v1_tree = OUT_ROOTS["v1"].resolve()
    if (GEOMETRY_VERSION != "v1" or CLEAN_VERSION != "v1") and v1_tree in (out_dir, *out_dir.parents):
        raise ValueError(f"refusing to write --geometry {GEOMETRY_VERSION} --clean {CLEAN_VERSION} output into "
                         f"the v1 tree: {out_dir}")
    if CLEAN_VERSION == "v2" and OUT_ROOTS["v2"].resolve() in (out_dir, *out_dir.parents):
        raise ValueError(f"refusing to write --clean v2 output into the geom2 tree: {out_dir}")
    if CLEAN_VERSION == "none":
        raise ValueError("--clean none never writes (provenance mode); use --compare-existing")


def compare_outputs(outputs: dict[str, pd.DataFrame], ref_dir: Path) -> dict:
    """Compare in-memory outputs with the CSVs of the same names in `ref_dir`.

    Returns cmp_bytes_equal (all files byte-identical to to_csv(index=False)), cmp_max_abs_diff
    (coordinates + head direction, NaN-aware), cmp_nan_mismatch (cells NaN in one and not the
    other), cmp_roi_diff (ROI cells that differ; an empty cell counts as 'None', which is how
    the notebook wrote untracked frames), cmp_missing (reference files absent)."""
    res = dict(cmp_bytes_equal=True, cmp_max_abs_diff=0.0, cmp_nan_mismatch=0, cmp_roi_diff=0, cmp_missing=0)
    for name, frame in outputs.items():
        if name.endswith("_track_flags.csv"):
            continue
        ref = ref_dir / name
        if not ref.exists():
            res["cmp_missing"] += 1
            res["cmp_bytes_equal"] = False
            continue
        mine = frame.to_csv(index=False).encode()
        if mine == ref.read_bytes():
            continue
        res["cmp_bytes_equal"] = False
        other = pd.read_csv(ref)
        if name.endswith("_ROIs.csv"):
            a = frame.astype(str).to_numpy()
            o = other.fillna("None").astype(str).to_numpy()
            res["cmp_roi_diff"] += int((a != o).sum()) if a.shape == o.shape else a.size
        else:
            a = frame.to_numpy(float)
            o = other.to_numpy(float)
            if a.shape != o.shape:
                res["cmp_max_abs_diff"] = float("inf")
                continue
            na, no = np.isnan(a), np.isnan(o)
            res["cmp_nan_mismatch"] += int((na != no).sum())
            both = ~na & ~no
            if both.any():
                res["cmp_max_abs_diff"] = max(res["cmp_max_abs_diff"], float(np.abs(a[both] - o[both]).max()))
    return res


def process_one(h5: Path, stem: str, mask_dir: Path, corners: np.ndarray, out_dir: Path,
                write: bool, allow_degenerate: bool = False, compare_dir: Path | None = None) -> dict:
    """One session under CLEAN_VERSION / GEOMETRY_VERSION; returns its QC row.

    v1 keeps the notebook call sequence exactly (byte-identity gate); none is the raw affine;
    v2 runs tracking_clean and labels ROIs from the cleaned track."""
    if write:
        guard_output_dir(out_dir)
    raw, nodes = load_tracks(h5)
    labels, names, overlap = label_image(mask_dir)
    if len(names) != 21:
        raise ValueError(f"{mask_dir}: {len(names)} masks, expected 21")
    A, b = affine(corners, PHYSICAL_PTS, check=not allow_degenerate)
    resid_cm, det = affine_quality(A, b, corners, PHYSICAL_PTS)

    flags, v2_report = None, None
    if CLEAN_VERSION == "v1":
        cleaned = interpolate_smooth(drop_jumps(raw, nodes), nodes)
        cleaned_cm = to_cm(cleaned, nodes, A, b)
        hd = head_direction(cleaned_cm)
        rois = roi_labels(raw, nodes, labels, names)
    elif CLEAN_VERSION == "none":
        cleaned_cm = to_cm(raw.copy(), nodes, A, b)
        hd = head_direction(cleaned_cm)
        rois = roi_labels(raw, nodes, labels, names)
    else:
        scores = tclean.load_scores(h5)
        cleaned_cm, flags, v2_report = tclean.clean_v2(raw, nodes, scores, lambda px: (A @ np.asarray(px, float).T).T + b)
        hd = head_direction(cleaned_cm)
        rois = roi_labels(cm_to_px(cleaned_cm, nodes, A, b), nodes, labels, names)

    outputs = {f"{stem}.analysis_cleaned_coordinates.csv": cleaned_cm,
               f"{stem}.analysis_head_direction.csv": hd,
               f"{stem}.analysis_ROIs.csv": rois}
    if flags is not None:
        outputs[f"{stem}.analysis_track_flags.csv"] = flags
    existing = [n for n in outputs if (out_dir / n).exists()]
    if write and not existing:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in outputs.items():
            frame.to_csv(out_dir / name, index=False)

    hb = rois["head_back"].to_numpy()
    hbx, hby = cleaned_cm["head_back.x"], cleaned_cm["head_back.y"]
    row = dict(stem=stem, n_frames=len(raw), masks=mask_dir.name, mask_overlap_px=overlap,
               head_back_none=float(np.mean(hb == "None")),
               head_back_node=float(np.mean(np.char.startswith(hb.astype(str), "node"))),
               head_back_edge=float(np.mean(np.char.startswith(hb.astype(str), "edge"))),
               raw_nan_frac=float(raw["head_back.x"].isna().mean()),
               geometry=GEOMETRY_VERSION, clean_version=CLEAN_VERSION,
               affine_resid_cm=round(resid_cm, 3), affine_det=det,
               hb_x_span_cm=round(float(np.nanmax(hbx) - np.nanmin(hbx)), 1),
               hb_y_span_cm=round(float(np.nanmax(hby) - np.nanmin(hby)), 1),
               status=("exists_skipped" if existing else "written" if write else "dry_run"))
    if v2_report is not None:
        row.update(tclean.report_row(v2_report))
        row["v2_hd_nan_frac"] = float(hd["back2mid_deg"].isna().mean())
    if compare_dir is not None:
        row.update(compare_outputs(outputs, compare_dir))
        row["status"] = "compared"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--mouse", nargs="*", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="process at most N files (smoke test)")
    ap.add_argument("--geometry", choices=("v1", "v2"), default="v1",
                    help="v1: corner sets as the notebook had them (reproduces maze/); "
                         "v2: cyclic maze-2 corners, output to maze_geom2/")
    ap.add_argument("--clean", choices=("v1", "none", "v2"), default="v1",
                    help="v1: notebook cleaning (default); none: raw affine, compare only; v2: tracking_clean -> maze_v2/")
    ap.add_argument("--compare-existing", action="store_true",
                    help="recompute in memory and compare with the reference tree; exit 1 on any difference")
    args = ap.parse_args()

    global GEOMETRY_VERSION, OUT_ROOT, CLEAN_VERSION
    CLEAN_VERSION = args.clean
    if CLEAN_VERSION == "v2" and args.geometry == "v1":
        print("--clean v2 implies --geometry v2 (cyclic maze-2 corners)")
        args.geometry = "v2"
    GEOMETRY_VERSION = args.geometry
    OUT_ROOT = OUT_ROOT_CLEAN_V2 if CLEAN_VERSION == "v2" else OUT_ROOTS[GEOMETRY_VERSION]
    compare_root = None
    if args.compare_existing:
        if args.write:
            sys.exit("--compare-existing never writes; drop --write")
        compare_root = OUT_ROOTS["v1"] if CLEAN_VERSION in ("v1", "none") else OUT_ROOT_CLEAN_V2
        print(f"comparing --clean {CLEAN_VERSION} --geometry {GEOMETRY_VERSION} against {compare_root}")
    else:
        print(f"geometry {GEOMETRY_VERSION}, clean {CLEAN_VERSION} -> {OUT_ROOT}")
    if CLEAN_VERSION == "none" and args.write:
        sys.exit("--clean none never writes (LEC provenance mode); use --compare-existing")
    if rr.DATASET == "lec" and args.write and CLEAN_VERSION == "v1":
        sys.exit("LEC maze CSVs on disk are the raw affine with no cleaning; a v1-cleaned rewrite of "
                 "maze/ is refused (use --clean none --compare-existing, or --clean v2 --write)")

    rows = []
    for mouse in args.mouse or rr.EPHYS_MICE:
        files = session_files(mouse)
        if args.limit:
            files = files[:args.limit]
        for h5, stem, d in files:
            mask_name, corners, note = geometry(mouse, d)
            degenerate = note.startswith("DEGENERATE")
            try:
                r = process_one(h5, stem, MASKS_ROOT / mask_name, corners, OUT_ROOT / mouse, args.write,
                                allow_degenerate=degenerate,
                                compare_dir=(compare_root / mouse) if compare_root else None)
            except OSError as exc:                      # truncated h5 (ly06 2025-06-19 15:40): skip, never repair
                r = dict(stem=stem, status="h5_unreadable", note=f"{note}; {str(exc)[:80]}", n_frames=0,
                         head_back_none=np.nan, head_back_node=np.nan, head_back_edge=np.nan, raw_nan_frac=np.nan,
                         affine_resid_cm=np.nan, hb_x_span_cm=np.nan, hb_y_span_cm=np.nan, geometry=GEOMETRY_VERSION,
                         clean_version=CLEAN_VERSION)
                rows.append(dict(r, mouse=mouse))
                print(f"h5_unreadable   {stem}: {str(exc)[:80]}")
                continue
            r.update(mouse=mouse, note=note)
            rows.append(r)
            extra = ""
            if compare_root:
                extra = (f"bytes_equal={r['cmp_bytes_equal']} max|diff|={r['cmp_max_abs_diff']:.2e} "
                         f"nan_mismatch={r['cmp_nan_mismatch']} roi_diff={r['cmp_roi_diff']} missing={r['cmp_missing']}")
            elif CLEAN_VERSION == "v2":
                extra = (f"v2 removed={r['v2_removed_by_rules']} bridged={r['v2_bridged']} leftNaN={r['v2_left_nan']} "
                         f"hdNaN={r['v2_hd_nan_frac']:.3f}")
            print(f"{r['status']:15s} {stem} masks={mask_name} frames={r['n_frames']} "
                  f"head_back none={r['head_back_none']:.3f} node={r['head_back_node']:.3f} "
                  f"edge={r['head_back_edge']:.3f} rawNaN={r['raw_nan_frac']:.3f} "
                  f"affine resid={r['affine_resid_cm']:.2f}cm span x={r['hb_x_span_cm']} y={r['hb_y_span_cm']} {extra} {note}",
                  flush=True)
    table = pd.DataFrame(rows)
    if len(table):
        print("\nper mouse (means over sessions):")
        print(table.groupby("mouse")[["n_frames", "head_back_none", "head_back_node", "head_back_edge",
                                       "raw_nan_frac", "affine_resid_cm", "hb_x_span_cm", "hb_y_span_cm"]]
              .mean().round(3).to_string())
        print(table.groupby(["mouse", "status"]).size().to_string())
        n_degen = int(table["note"].fillna("").str.startswith("DEGENERATE").sum())
        if n_degen:
            print(f"WARNING: {n_degen} sessions written with the legacy degenerate maze-2 affine "
                  f"(x collapsed); use --geometry v2 for usable cm coordinates")
        if compare_root is not None:
            ok = table["cmp_bytes_equal"].fillna(False).astype(bool)
            zero = (table["cmp_max_abs_diff"].fillna(np.inf) == 0) & (table["cmp_nan_mismatch"].fillna(1) == 0) \
                & (table["cmp_roi_diff"].fillna(1) == 0) & (table["cmp_missing"].fillna(1) == 0)
            n = int(ok.sum())
            print(f"\nEQUIVALENCE: {n}/{len(table)} byte-identical; {int((zero & ~ok).sum())} differ in bytes only "
                  f"(max|diff| 0, same NaN pattern, same ROIs); {int((~zero).sum())} differ in values; "
                  f"{int((table['status'] == 'h5_unreadable').sum())} unreadable h5")
            bad = table[~zero & (table["status"] != "h5_unreadable")]
            if len(bad):
                print(bad[["mouse", "stem", "cmp_max_abs_diff", "cmp_nan_mismatch", "cmp_roi_diff", "cmp_missing"]].to_string(index=False))
            sys.exit(0 if len(bad) == 0 else 1)
        suffix = {"v1": "" if GEOMETRY_VERSION == "v1" else "_geom2", "none": "_none", "v2": "_v2"}[CLEAN_VERSION]
        qc = SLEAP_ROOT / f"sleap_to_rois_qc_{rr.DATASET}{suffix}.csv"
        if args.write:
            n = 0
            while (p := qc.with_name(qc.stem + (f".v{n}" if n else "") + ".csv")).exists():
                n += 1
            table.to_csv(p, index=False)
            print(f"QC -> {p}")


if __name__ == "__main__":
    main()
