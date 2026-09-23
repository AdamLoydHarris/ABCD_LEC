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

Geometry per mouse is in GEOMETRY. bp02 is given bp04's maze-2 camera geometry
PROVISIONALLY: the two mice used the same box at different times of day and their SLEAP
footprints agree to ~40 px, but nothing has been measured yet -- `poke_calibration.py` is
the test. Existing outputs are never overwritten.

Usage::

    python sleap_to_rois.py --dataset mec                    # dry run: list what would be written
    python sleap_to_rois.py --dataset mec --write [--mouse bp04] [--limit 1]
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
import recday_registry as rr                                          # noqa: E402

RAW_SLEAP = rr.REPO_ROOT / "data" / "raw" / "sleap"
OUT_ROOT = rr.REPO_ROOT / "data" / "preprocessed" / "sleap" / "maze"
MASKS_ROOT = {"mec": rr.REPO_ROOT / "code" / "masks",
              "lec": rr.REPO_ROOT / "code" / "preprocessing" / "masks"}[rr.DATASET]

#: Maze corners in cm. Order matches the pixel arrays below (the notebook's convention).
PHYSICAL_PTS = np.array([[0, 0], [47, 0], [47, 47], [0, 47]], dtype=float)

#: Hand-picked pixel corners, verbatim from `sleap_preprocess_mEC.ipynb`.
MAZE1_EARLY = np.array([[1208, 840], [989, 25], [184, 213], [376, 1030]], dtype=float)
MAZE1_LATE = np.array([[1109, 826], [908, -5], [72, 174], [254, 1028]], dtype=float)
MAZE2_MEC = np.array([[256, 944], [1034, 944], [260, 182], [1009, 166]], dtype=float)
LEC_MAZE1 = np.array([[1022, 974], [981, 136], [158, 169], [172, 1010]], dtype=float)
LEC_MAZE2 = np.array([[234, 945], [1036, 953], [1020, 158], [248, 169]], dtype=float)

#: bp01's camera moved between 2024-03-25 and 2024-03-28 (the 26th "is all over the place").
BP01_SWITCH = date(2024, 3, 25)


def geometry(mouse: str, session_date: date) -> tuple[str, np.ndarray, str]:
    """(mask directory name, pixel corners, note) for one session."""
    if rr.DATASET == "mec":
        if mouse == "bp01":
            return (("mEC_maze1_early", MAZE1_EARLY, "") if session_date <= BP01_SWITCH
                    else ("mEC_maze1_late", MAZE1_LATE, ""))
        if mouse == "bp04":
            return "mEC_maze2", MAZE2_MEC, ""
        if mouse == "bp02":
            return "mEC_maze2", MAZE2_MEC, "PROVISIONAL: bp04 geometry, verify with poke_calibration"
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


def affine(pixel_pts: np.ndarray, physical_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares 2x2 A and offset b with physical = A @ pixel + b."""
    n = len(pixel_pts)
    M = np.zeros((2 * n, 6))
    v = np.zeros(2 * n)
    for i, ((px, py), (wx, wy)) in enumerate(zip(pixel_pts, physical_pts)):
        M[2 * i, :3] = (px, py, 1.0)
        v[2 * i] = wx
        M[2 * i + 1, 3:] = (px, py, 1.0)
        v[2 * i + 1] = wy
    a11, a12, bx, a21, a22, by = np.linalg.lstsq(M, v, rcond=None)[0]
    return np.array([[a11, a12], [a21, a22]]), np.array([bx, by])


def to_cm(df: pd.DataFrame, nodes: list[str], A: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    for bp in nodes:
        xy = df[[f"{bp}.x", f"{bp}.y"]].to_numpy()
        cm = (A @ xy.T).T + b
        df[f"{bp}.x"], df[f"{bp}.y"] = cm[:, 0], cm[:, 1]
    return df


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


def label_image(mask_dir: Path) -> tuple[np.ndarray, list[str], int]:
    """Stack the 21 binary masks into one int label image; first mask (sorted) wins on overlap.

    Returns (labels, names, n_overlap_px). Label 0 = no mask; name index = label - 1.
    """
    names, labels, overlap = [], None, 0
    for tif in sorted(mask_dir.glob("*.tif*")):
        mask = tifffile.imread(tif).astype(bool)
        if labels is None:
            labels = np.zeros(mask.shape, dtype=np.int16)
        names.append(tif.stem)
        claimed = mask & (labels > 0)
        overlap += int(claimed.sum())
        labels[mask & (labels == 0)] = len(names)
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


def process_one(h5: Path, stem: str, mask_dir: Path, corners: np.ndarray, out_dir: Path,
                write: bool) -> dict:
    raw, nodes = load_tracks(h5)
    labels, names, overlap = label_image(mask_dir)
    if len(names) != 21:
        raise ValueError(f"{mask_dir}: {len(names)} masks, expected 21")

    cleaned = interpolate_smooth(drop_jumps(raw, nodes), nodes)
    A, b = affine(corners, PHYSICAL_PTS)
    cleaned_cm = to_cm(cleaned, nodes, A, b)
    hd = head_direction(cleaned_cm)
    rois = roi_labels(raw, nodes, labels, names)

    outputs = {f"{stem}.analysis_cleaned_coordinates.csv": cleaned_cm,
               f"{stem}.analysis_head_direction.csv": hd,
               f"{stem}.analysis_ROIs.csv": rois}
    existing = [n for n in outputs if (out_dir / n).exists()]
    if write and not existing:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in outputs.items():
            frame.to_csv(out_dir / name, index=False)

    hb = rois["head_back"].to_numpy()
    return dict(stem=stem, n_frames=len(raw), masks=mask_dir.name, mask_overlap_px=overlap,
                head_back_none=float(np.mean(hb == "None")),
                head_back_node=float(np.mean(np.char.startswith(hb.astype(str), "node"))),
                head_back_edge=float(np.mean(np.char.startswith(hb.astype(str), "edge"))),
                raw_nan_frac=float(raw["head_back.x"].isna().mean()),
                status=("exists_skipped" if existing else "written" if write else "dry_run"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--mouse", nargs="*", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="process at most N files (smoke test)")
    args = ap.parse_args()

    rows = []
    for mouse in args.mouse or rr.EPHYS_MICE:
        files = session_files(mouse)
        if args.limit:
            files = files[:args.limit]
        for h5, stem, d in files:
            mask_name, corners, note = geometry(mouse, d)
            r = process_one(h5, stem, MASKS_ROOT / mask_name, corners, OUT_ROOT / mouse, args.write)
            r.update(mouse=mouse, note=note)
            rows.append(r)
            print(f"{r['status']:15s} {stem} masks={mask_name} frames={r['n_frames']} "
                  f"head_back none={r['head_back_none']:.3f} node={r['head_back_node']:.3f} "
                  f"edge={r['head_back_edge']:.3f} rawNaN={r['raw_nan_frac']:.3f} {note}")
    table = pd.DataFrame(rows)
    if len(table):
        print("\nper mouse (means over sessions):")
        print(table.groupby("mouse")[["n_frames", "head_back_none", "head_back_node", "head_back_edge",
                                       "raw_nan_frac"]].mean().round(3).to_string())
        print(table.groupby(["mouse", "status"]).size().to_string())
        qc = OUT_ROOT.parent / f"sleap_to_rois_qc_{rr.DATASET}.csv"
        if args.write:
            n = 0
            while (p := qc.with_name(qc.stem + (f".v{n}" if n else "") + ".csv")).exists():
                n += 1
            table.to_csv(p, index=False)
            print(f"QC -> {p}")


if __name__ == "__main__":
    main()
