"""
Salvage the camera -> pyControl offset of a session whose pinstate file is truncated.

`sleap_alignment_copilot.py` aligns SLEAP tracking to pyControl with a single anchor: the
frame of the first up-state in the camera's pinstate CSV is the time of the first pyControl
`rsync` event, and the video runs at VIDEO_FPS = 60. When the pinstate file is truncated
(ah10 2025-06-19-143522 has 24 lines for a 78,000-frame video) that anchor is lost and the
session gets no Locs_raw / XY_raw / HD_raw. This script recovers the anchor from behaviour.

Method
------
Under a candidate offset ``f0`` (video frame index at pyControl t = 0; frame = f0 + fps * t)
the tracked ``head_back`` ROI at each nose-poke ENTRY must be the poked port. The poke table
(`pokes_{recday}_{session}.npy`, columns [entry_bin, exit_bin, port, rewarded, state], 25 ms
bins from the first A_on) gives ~500 such constraints per session. Sweeping f0 gives an
agreement curve with a flat top (the animal is at the port for a while before and after the
poke registers), typically 10-30 frames wide, so agreement alone fixes f0 to within that
plateau. Three things then locate f0 precisely:

1. Which pokes. Many videos drop frames (the pinstate edges fall behind the pipeline's 60 fps
   model by up to ~300 frames by the end of a session, in steps). A dropped block shifts the
   constraints of every LATER poke downward, so the start-of-session anchor is set by the
   pokes before the first drop. The plateau is therefore computed on growing prefixes of the
   pokes (25%, 37.5%, ..., 100%) and the largest prefix whose plateau still overlaps the
   running intersection of the earlier ones is used; for a clean video that is all pokes.
2. Where in the plateau. The plateau's LOWER edge is set by the fastest arrivals at a port
   (crossing into the node ROI to breaking the beam, ~200 ms at running speed): a property of
   the rig, not of the animal's mood, and unaffected by dropped frames (which only pull the
   upper edge down). Its offset from the true f0, measured on the mouse's pinstate-backed
   sessions, has a per-mouse IQR of 2-3 frames (medians 8-14 frames). The primary estimate is
   f0 = plateau lower edge + the mouse's median offset.
3. Comparisons, always reported: the latency-matched f0 (the f0 in the plateau whose median
   arrival-to-entry latency equals the mouse's reference median -- the original proposal; the
   median latency turned out NOT to be stable, 1.9-3.4 s on ah10's first recording days vs
   0.42 s later) and the plateau centre.

Reliability. A session whose offset holds only over a prefix of its pokes (frames dropped
later), or whose plateau is narrower than MIN_PLATEAU_WIDTH or agrees with fewer than
MIN_AGREEMENT of the pokes, is flagged UNRELIABLE; salvage refuses to stage arrays for it
unless `--allow-unreliable` is given. (Pinstate-backed sessions with dropped frames are
misaligned late in the session in the existing pipeline output too -- see the `pipe_drift`
column of `--validate`.)

Frame rate: the pipeline hard-codes 60 fps; clean pinstate fits give 59.997. The estimate
uses the mouse's median fitted fps over clean sessions, but the salvaged arrays are cut with
the pipeline's own arithmetic (60 fps from the estimated first-rsync frame) so a salvaged
session is produced exactly as a pinstate-backed one would be.

Validation (`--validate`)
-------------------------
Every session of every mouse with a usable pinstate (>= 10 rising edges), a poke table and
SLEAP ROIs -- 181 sessions on 2026-09-06 (10 skipped for empty poke tables, 1 the broken
session). Truth: rsync pulses are matched to pinstate rising edges. Mice were run in pairs
(ah08 with ly07, ly05 with ly06) and the camera's sync line carries BOTH boxes' rsync
trains, so those pinstates hold ~2x the edges; the anchor is found by nearest-edge distance
over the first pulses and each later pulse is predicted from the previous matched edge, so
foreign, lost or extra edges do not derail the pairing. The first-rsync frame is the median
over the first ten matched pairs of edge - 60 * (t - t_first) -- the pipeline's
`first_up_frame` to within the 0-4 frames by which the paired box's pulse sometimes came
first. The behavioural estimate uses leave-one-session-out references (lower-edge offset,
latency, fps) from the same mouse. Gate: |first_rsync_frame_est - truth| <= 15 frames
(250 ms, ten 25 ms bins) on every session; failures are printed, split by the estimator's
own reliability flag. Result on 2026-09-06: median |err| 1.4 frames, p90 4.8, max 16.7,
180/181 within the gate; the one failure is flagged unreliable (offset held over 25% of its
pokes only). Latency-matched: median 3.1, max 28.7, 171/180. Plateau centre: 4.4, max 24.7,
174/181. ah10 alone (the mouse of the broken session): median 1.2, max 4.4 over 38 sessions.
The table printed by `--validate` (also `salvage_staging/salvage_validation.csv`) is the record.

Salvage (`--session MOUSE YYYY-MM-DD-HHMMSS`)
--------------------------------------------
Estimates the offset, produces Locs_raw / XY_raw / HD_raw exactly as
`process_session_by_metadata` would (same ROI column, same first-A_on cut, same 60 -> 40 Hz
resample, same dtypes and filenames) under `data/processed_data/salvage_staging/`, with a
JSON sidecar, and prints the `loc_at_goal` / `poke_loc` checks of `validate_alignment.py`
for the staged arrays. Nothing outside the staging directory is touched unless `--write`
is ALSO given, which copies the arrays into the pipeline's tracking folder and inserts them
into `data_dic_lec.pkl` after backing it up. Without `--write` the plan is printed only.

`--replicate MOUSE STAMP` re-cuts a pinstate-backed session with its own pinstate anchor
through this script's staging code and diffs it against the arrays on disk, proving the
staging code is the pipeline's.

Assumptions made explicit
-------------------------
* Poke entry time = start of its 25 ms bin, t_ms = (entry_bin + first_A_on_ms // 25) * 25.
  The <= 25 ms bias is identical in the reference sessions and cancels.
* Agreement counts a poke as valid when the tracked label is any known maze location
  (node or edge); an edge at poke time is a miss, an untracked frame is excluded.
* `head_back` is the tracked node, as in the pipeline. Location strings are mapped with
  `build_data_dic.LOCATION_MAPPING` (nodes 1-9, edges 10-21, unknown 0).
* Locs are resampled by index picking (`linspace(...).astype(int)`), XY and HD by the
  pipeline's `resample_data` (linear interpolation over valid points, extrapolating).
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import recday_registry as rr                                          # noqa: E402
from build_data_dic import LOCATION_MAPPING, REVERSE_MAPPING, locs_to_int  # noqa: E402
from extract_pokes import BIN_MS, parse_pycontrol                     # noqa: E402
import sleap_alignment_copilot as sac                                 # noqa: E402

REPO_ROOT = rr.REPO_ROOT
PROCESSED_DATA = REPO_ROOT / "data" / "processed_data"
TRIALTIMES_PATH = PROCESSED_DATA / "trialtimes_raw_mingyutest"
NEURON_PATH = PROCESSED_DATA / "neuron_raw_mingyutest"
TRACKING_PATH = REPO_ROOT / "data" / "processed"
STAGING_PATH = PROCESSED_DATA / "salvage_staging"
DATA_DIC_PATH = PROCESSED_DATA / "data_dic_lec.pkl"
ALIGNMENT_QC_PATH = PROCESSED_DATA / "alignment_qc.csv"

MICE = list(rr.EPHYS_MICE)
MIN_PINSTATE_EDGES = 10          # fewer rising edges than this: the pinstate is unusable
GATE_FRAMES = 15                 # validation gate on |first_rsync_frame_est - truth|
SEARCH_HALF_WIDTH_S = 150.0      # f0 searched over the clock-stamp offset +/- this
PLATEAU_TOL = 0.005              # plateau = agreement within this of the best value
MIN_POKE_FRACTION = 0.9          # a candidate f0 must keep this fraction of pokes in the video
MIN_LATENCY_POKES = 20           # pokes needed for a median latency to count
LATENCY_MATCH_FRAMES = 2.0       # latency match accepted if the reference is reached this closely
PREFIX_FRACTIONS = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)   # growing prefixes of the pokes
OVERLAP_TOL_FRAMES = 3.0         # a prefix plateau must overlap the earlier ones to within this
MIN_PLATEAU_WIDTH = 6            # narrower plateau than this -> unreliable
MIN_AGREEMENT = 0.99             # lower best agreement than this -> unreliable
CLEAN_RESID_SD = 0.5             # pinstate fit residual SD below which a video counts as clean
NODE_COLUMN = "head_back"


# ---------------------------------------------------------------------------
# Core estimator (pure functions, no I/O)
# ---------------------------------------------------------------------------

def poke_entry_times_ms(pokes: np.ndarray, first_A_on_ms: int) -> np.ndarray:
    """Poke entry bins -> pyControl ms (start of the 25 ms bin). Inverse of `extract_pokes.to_bins`."""
    return (np.asarray(pokes)[:, 0].astype(np.int64) + int(first_A_on_ms) // BIN_MS) * BIN_MS


def frames_at(f0_grid: np.ndarray, t_ms: np.ndarray, fps: float) -> np.ndarray:
    """(n_f0, n_pokes) video frame index of each pyControl time under each candidate f0."""
    return np.floor(np.asarray(f0_grid, float)[:, None]
                    + fps * np.asarray(t_ms, float)[None, :] / 1000.0).astype(np.int64)


def run_starts(codes: np.ndarray) -> np.ndarray:
    """For every frame, the index at which the current run of identical labels began."""
    codes = np.asarray(codes)
    starts = np.zeros(len(codes), dtype=np.int64)
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts[change] = change
    return np.maximum.accumulate(starts)


def agreement_curve(codes: np.ndarray, t_ms: np.ndarray, ports: np.ndarray, fps: float,
                    f0_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fraction of poke entries whose tracked location is the poked port, per candidate f0.

    Valid pokes are those whose frame is inside the video and whose label is a known maze
    location (code > 0). Returns (agreement, n_valid); agreement is NaN where n_valid == 0.
    """
    fr = frames_at(f0_grid, t_ms, fps)
    inside = (fr >= 0) & (fr < len(codes))
    c = codes[np.clip(fr, 0, len(codes) - 1)]
    valid = inside & (c > 0)
    hit = valid & (c == np.asarray(ports)[None, :])
    n = valid.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        agr = np.where(n > 0, hit.sum(1) / np.maximum(n, 1), np.nan)
    return agr, n


def latency_curve(codes: np.ndarray, starts: np.ndarray, t_ms: np.ndarray, ports: np.ndarray,
                  fps: float, f0_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median arrival-to-entry latency (ms) per candidate f0.

    For each poke whose mapped frame shows the poked port, the latency is the number of frames
    since that run of the port label began. Returns (median_ms, n_pokes_with_latency,
    latency matrix in ms with NaN where the poke is not at its port).
    """
    fr = frames_at(f0_grid, t_ms, fps)
    idx = np.clip(fr, 0, len(codes) - 1)
    at_port = (fr >= 0) & (fr < len(codes)) & (codes[idx] == np.asarray(ports)[None, :])
    lat_ms = np.where(at_port, (fr - starts[idx]) / fps * 1000.0, np.nan)
    n = at_port.sum(1)
    med = np.array([np.nanmedian(row) if k else np.nan for row, k in zip(lat_ms, n)])
    return med, n, lat_ms


def find_plateau(codes, t_ms, ports, fps, f0_grid, plateau_tol=PLATEAU_TOL,
                 min_poke_fraction=MIN_POKE_FRACTION) -> dict:
    """Contiguous run of candidate f0 around the best agreement, within `plateau_tol` of it."""
    agr, n_used = agreement_curve(codes, t_ms, ports, fps, f0_grid)
    ok = (n_used >= min_poke_fraction * len(t_ms)) & np.isfinite(agr)
    if not ok.any():
        raise ValueError("no candidate f0 keeps enough pokes inside the video; widen f0_range")
    agr_ok = np.where(ok, agr, -np.inf)
    i_best = int(np.argmax(agr_ok))
    best = float(agr_ok[i_best])
    on = ok & (agr >= best - plateau_tol)
    lo = i_best
    while lo > 0 and on[lo - 1]:
        lo -= 1
    hi = i_best
    while hi < len(f0_grid) - 1 and on[hi + 1]:
        hi += 1
    return {"lo": lo, "hi": hi, "best": best, "agr": agr, "n_used": n_used,
            "f0_lo": float(f0_grid[lo]), "f0_hi": float(f0_grid[hi]),
            "centre": float(0.5 * (f0_grid[lo] + f0_grid[hi]))}


def estimate_first_rsync_frame(node_per_frame, pokes, first_A_on_ms, first_rsync_ms, fps,
                               ref_median_latency_ms, f0_range, step: float = 1.0,
                               plateau_tol: float = PLATEAU_TOL,
                               min_poke_fraction: float = MIN_POKE_FRACTION,
                               min_latency_pokes: int = MIN_LATENCY_POKES,
                               prefix_fractions: Tuple[float, ...] = PREFIX_FRACTIONS,
                               n_quarters: int = 4,
                               ref_lower_offset_frames: Optional[float] = None) -> dict:
    """Estimate the video frame of the first pyControl rsync from poke/tracking agreement.

    Args:
        node_per_frame: (n_frames,) integer location code per video frame (nodes 1-9,
            edges 10-21, 0 unknown), i.e. `locs_to_int` of the head_back ROI column.
        pokes: (n_pokes, 5) table [entry_bin, exit_bin, port, rewarded, state].
        first_A_on_ms: pyControl time of the first A_on (the poke table's origin).
        first_rsync_ms: pyControl time of the first rsync event (0 if none, as the pipeline).
        fps: video frame rate to assume (the mouse's clean pinstate-fitted median, ~59.997).
        ref_median_latency_ms: reference median arrival-to-entry latency from pinstate-backed
            sessions of the same mouse. None -> no latency candidate.
        f0_range: (lo, hi) frames to sweep for f0, the frame at pyControl t = 0.
        ref_lower_offset_frames: reference offset of the true f0 above the plateau's lower edge
            (`lower_offset_reference`), from pinstate-backed sessions of the same mouse. When
            given it defines the primary estimate; otherwise the latency/centre hybrid does.

    Returns a dict:
        f0, first_rsync_frame          the primary estimate (frames; first_rsync_frame is the
                                       integer anchor the pipeline calls first_up_frame)
        method                         'lower-edge' | 'latency' | 'centre' (see module doc)
        f0_lower, first_rsync_frame_lower, ref_lower_offset_frames
                                       plateau lower edge + reference offset (NaN if no reference)
        reliable, flags                False + reasons when the offset holds only over a prefix
                                       of the pokes (frames dropped later), or the plateau is
                                       too narrow / agreement too low
        used_fraction, n_pokes_used    the prefix of pokes the estimate is based on (1.0 = all)
        f0_latency, f0_centre          the two candidates inside the chosen plateau (NaN if undefined)
        f0_all                         centre of the all-poke plateau (what a naive sweep gives)
        f0_early                       centre of the first-prefix (25%) plateau
        first_rsync_frame_latency / _centre / _all / _early   the same as integer anchors
        plateau_lo, plateau_hi, plateau_width          chosen plateau (frames); plateau_all likewise
        agreement, agreement_at_f0, agreement_all      best agreement of the chosen / all-poke plateau
        median_latency_ms, median_latency_centre_ms, ref_median_latency_ms
        latency_matchable, at_plateau_edge             whether the reference was reached
        n_pokes_with_latency, n_pokes
        prefix_plateaus                per-prefix (frac, n, t_end_s, lo, hi, best, centre)
        quarter_f0, quarter_drift      plateau centres per session quarter and their offset
                                       from the first quarter (diagnostic drift profile)
        fps                            echoed
        f0_grid, agreement_grid        the all-poke sweep, for plotting / diagnostics
    Validated accuracy: the table printed by `--validate` is the record.
    """
    codes = np.asarray(node_per_frame).astype(np.int64)
    pokes = np.asarray(pokes, dtype=np.int64)
    if len(pokes) == 0:
        raise ValueError("empty poke table")
    order = np.argsort(pokes[:, 0], kind="stable")
    pokes = pokes[order]
    t_ms = poke_entry_times_ms(pokes, first_A_on_ms)
    ports = pokes[:, 2]
    starts = run_starts(codes)

    def anchor(f0):
        return None if f0 is None or not np.isfinite(f0) else int(round(f0 + fps * float(first_rsync_ms) / 1000.0))

    # 1. plateau over ALL pokes on the wide grid; everything else is refined on a local grid
    f0_grid = np.arange(np.floor(float(f0_range[0])), float(f0_range[1]) + step / 2, step)
    pl_all = find_plateau(codes, t_ms, ports, fps, f0_grid, plateau_tol, min_poke_fraction)
    local = np.arange(pl_all["f0_lo"] - 400.0, pl_all["f0_hi"] + 400.0 + step / 2, step)

    # 2. plateaus of growing prefixes of the session's pokes. Dropped frames only ever move the
    #    constraints of LATER pokes downward, so the start-of-session anchor is set by the pokes
    #    before the first drop: the largest prefix whose plateau still overlaps the running
    #    intersection of the earlier ones is the one to trust. Clean video -> all pokes.
    prefixes = []
    for frac in prefix_fractions:
        n = int(round(frac * len(pokes)))
        if n < 10:
            continue
        try:
            q = find_plateau(codes, t_ms[:n], ports[:n], fps, local, plateau_tol, min_poke_fraction)
        except ValueError:
            continue
        prefixes.append({"frac": float(frac), "n": n, "t_end_s": float(t_ms[n - 1] / 1000.0),
                         "lo": q["f0_lo"], "hi": q["f0_hi"], "best": q["best"], "centre": q["centre"]})
    if not prefixes:
        raise ValueError("no prefix of the pokes yields a plateau")
    chosen = prefixes[0]
    L, U = chosen["lo"], chosen["hi"]
    for p in prefixes[1:]:
        if p["hi"] < L - OVERLAP_TOL_FRAMES or p["lo"] > U + OVERLAP_TOL_FRAMES or p["best"] < MIN_AGREEMENT:
            break
        chosen = p
        L, U = max(L, p["lo"]), min(U, p["hi"])
    n_used = chosen["n"]
    plateau_grid = np.arange(chosen["lo"], chosen["hi"] + step / 2, step)
    width = float(plateau_grid[-1] - plateau_grid[0])
    agr_local, _ = agreement_curve(codes, t_ms[:n_used], ports[:n_used], fps, plateau_grid)

    # 3. latency match inside the chosen plateau (over the same prefix of pokes)
    med, n_lat, _ = latency_curve(codes, starts, t_ms[:n_used], ports[:n_used], fps, plateau_grid)
    j_centre = int(np.argmin(np.abs(plateau_grid - chosen["centre"])))
    usable = n_lat >= min_latency_pokes
    f0_latency, j_lat, at_edge, matchable = np.nan, None, False, False
    if ref_median_latency_ms is not None and usable.any():
        d = np.where(usable, np.abs(med - float(ref_median_latency_ms)), np.inf)
        j_lat = int(np.lexsort((np.abs(np.arange(len(plateau_grid)) - j_centre), d))[0])
        f0_latency = float(plateau_grid[j_lat])
        at_edge = j_lat in (0, len(plateau_grid) - 1) and len(plateau_grid) > 1
        matchable = bool(d[j_lat] <= LATENCY_MATCH_FRAMES / fps * 1000.0) and not at_edge

    # 4. drift profile (diagnostic only): plateau centre per quarter of the session's pokes
    quarter_f0 = []
    for chunk in np.array_split(np.arange(len(pokes)), n_quarters):
        try:
            q = find_plateau(codes, t_ms[chunk], ports[chunk], fps, local, plateau_tol, min_poke_fraction) \
                if len(chunk) >= 10 else None
        except ValueError:
            q = None
        quarter_f0.append(q["centre"] if q else np.nan)
    quarter_f0 = np.asarray(quarter_f0, float)
    quarter_drift = quarter_f0 - quarter_f0[0] if np.isfinite(quarter_f0[0]) else np.full(n_quarters, np.nan)

    # 5. reliability and the primary estimate
    flags = []
    if chosen["frac"] < 1.0:
        flags.append(f"offset consistent only over the first {chosen['frac']:.0%} of pokes "
                     f"(to t = {chosen['t_end_s']:.0f} s); frames dropped later in the video")
    if width < MIN_PLATEAU_WIDTH:
        flags.append(f"plateau only {width:.0f} frames wide")
    if chosen["best"] < MIN_AGREEMENT:
        flags.append(f"best agreement {chosen['best']:.3f}")
    reliable = not flags
    f0_lower = np.nan
    if ref_lower_offset_frames is not None and np.isfinite(ref_lower_offset_frames):
        f0_lower = float(plateau_grid[0]) + float(ref_lower_offset_frames)
        f0, method = f0_lower, "lower-edge"
    elif matchable:
        f0, method = f0_latency, "latency"
    else:
        f0, method = chosen["centre"], "centre"
    agr_at_f0, _ = agreement_curve(codes, t_ms[:n_used], ports[:n_used], fps, np.array([f0]))
    j_f0 = int(np.argmin(np.abs(plateau_grid - f0)))

    return {
        "f0": float(f0), "first_rsync_frame": anchor(f0), "method": method,
        "reliable": bool(reliable), "flags": flags,
        "used_fraction": chosen["frac"], "n_pokes_used": int(n_used), "n_pokes": int(len(pokes)),
        "f0_lower": float(f0_lower), "first_rsync_frame_lower": anchor(f0_lower),
        "ref_lower_offset_frames": None if ref_lower_offset_frames is None else float(ref_lower_offset_frames),
        "f0_latency": float(f0_latency), "first_rsync_frame_latency": anchor(f0_latency),
        "f0_centre": float(chosen["centre"]), "first_rsync_frame_centre": anchor(chosen["centre"]),
        "f0_all": float(pl_all["centre"]), "first_rsync_frame_all": anchor(pl_all["centre"]),
        "f0_early": float(prefixes[0]["centre"]), "first_rsync_frame_early": anchor(prefixes[0]["centre"]),
        "plateau_lo": float(plateau_grid[0]), "plateau_hi": float(plateau_grid[-1]), "plateau_width": width,
        "plateau_all": (pl_all["f0_lo"], pl_all["f0_hi"]),
        "agreement": chosen["best"], "agreement_at_f0": float(agr_at_f0[0]), "agreement_all": pl_all["best"],
        "median_latency_ms": float(med[j_lat]) if j_lat is not None else np.nan,
        "median_latency_centre_ms": float(med[j_centre]),
        "ref_median_latency_ms": None if ref_median_latency_ms is None else float(ref_median_latency_ms),
        "latency_matchable": bool(matchable), "at_plateau_edge": bool(at_edge),
        "n_pokes_with_latency": int(n_lat[j_f0]),
        "prefix_plateaus": prefixes, "quarter_f0": quarter_f0, "quarter_drift": quarter_drift,
        "fps": float(fps), "f0_grid": f0_grid, "agreement_grid": pl_all["agr"],
    }


def lower_offset_of(est: dict, first_rsync_frame_true: float, fps: float, first_rsync_ms: int) -> float:
    """Offset (frames) of a pinstate-backed session's true f0 above its estimated plateau lower edge.

    The lower edge is set by the fastest arrivals at a port (the time from entering the node
    ROI to breaking the beam), which is a physical constant of the rig rather than a
    behavioural state, and dropped frames never move it; its median over a mouse's
    pinstate-backed sessions is the reference the primary estimate adds to the lower edge.
    """
    f0_true = float(first_rsync_frame_true) - fps * float(first_rsync_ms) / 1000.0
    return f0_true - est["plateau_lo"]


def latencies_at(node_per_frame, pokes, first_A_on_ms, fps, f0) -> np.ndarray:
    """Arrival-to-entry latencies (ms) of every poke at one offset; used to build the reference."""
    codes = np.asarray(node_per_frame).astype(np.int64)
    pokes = np.asarray(pokes, dtype=np.int64)
    _, _, lat = latency_curve(codes, run_starts(codes), poke_entry_times_ms(pokes, first_A_on_ms),
                              pokes[:, 2], fps, np.array([float(f0)]))
    return lat[0][np.isfinite(lat[0])]


# ---------------------------------------------------------------------------
# Pinstate truth
# ---------------------------------------------------------------------------

def rising_edges(pinstate: np.ndarray) -> np.ndarray:
    """Frame indices of low -> high transitions, using the pipeline's level detection."""
    high, low = sac.detect_pinstate_levels(pinstate)
    if high is None or low is None:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero((pinstate[1:] == high) & (pinstate[:-1] == low)) + 1


def match_pulses(up_frames, rsync_ms, fps_nominal: float = 60.0, tol_frames: float = 10.0,
                 jump_frames: float = 150.0, start_pulses: int = 12, n_candidate_edges: int = 60,
                 max_first_pulse: int = 15) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Pair pyControl rsync pulses with pinstate rising edges.

    The pinstate may hold MORE edges than this box's pulses: mice were run in pairs (ah08 with
    ly07, ly05 with ly06) and the camera's sync line carries both boxes' rsync trains, so
    ah08/ly05/ly06/ly07 pinstates have ~2x the edges and the foreign train is interleaved at
    random. Interval-pattern matching therefore fails; instead the anchor is found by trying
    each of the first `n_candidate_edges` edges as the frame of pulse j0 (j0 < max_first_pulse)
    and scoring the first `start_pulses` predicted frames (at `fps_nominal`) by their distance
    to the nearest edge. Then every later pulse is predicted from the previous matched edge and
    accepted if an edge lies within `tol_frames`; edges that belong to the other box are simply
    never claimed. Frames dropped between two pulses shift the prediction only once.

    Returns (matched_edge_frames, matched_pulse_ms, first_edge_index, first_pulse_index).
    """
    up = np.asarray(up_frames, float)
    rs = np.asarray(rsync_ms, float)
    if len(up) < 3 or len(rs) < 3:
        raise ValueError("too few pulses/edges to pair")

    def nearest_dist(pred):
        k = np.clip(np.searchsorted(up, pred), 1, len(up) - 1)
        return np.minimum(np.abs(up[k - 1] - pred), np.abs(up[k] - pred))

    def nearest(pred):
        k = int(np.clip(np.searchsorted(up, pred), 1, len(up) - 1))
        cands = up[k - 1:k + 1]
        return cands[int(np.argmin(np.abs(cands - pred)))]

    # Anchor: the EARLIEST pulse that fits the pinstate to within 2 frames wins; only if no early
    # pulse fits (camera started late, or frames dropped in the first minute) is the overall
    # best-scoring (pulse, edge) pair used.
    best = None
    for j0 in range(min(max_first_pulse, len(rs) - 3)):
        n = min(start_pulses, len(rs) - j0)
        rel = fps_nominal * (rs[j0:j0 + n] - rs[j0]) / 1000.0
        for i0, e0 in enumerate(up[:n_candidate_edges]):
            score = float(np.median(nearest_dist(e0 + rel)))
            if best is None or score < best[0]:
                best = (score, i0, j0)
        if best is not None and best[2] == j0 and best[0] <= 2.0:
            break
    if best is None or best[0] > tol_frames:
        raise ValueError(f"could not anchor the pulse train on the pinstate "
                         f"(median nearest-edge distance {best[0] if best else 'n/a'} frames)")
    _, i0, j0 = best
    edges, pulses = [up[i0]], [rs[j0]]
    e_prev, t_prev = up[i0], rs[j0]
    for j in range(j0 + 1, len(rs)):
        pred = e_prev + fps_nominal * (rs[j] - t_prev) / 1000.0
        e = nearest(pred)
        accept = abs(e - pred) <= tol_frames
        if not accept and abs(e - pred) <= jump_frames and j + 1 < len(rs):
            # A block of frames dropped (or duplicated) since the last pulse: accept the jump only
            # if the NEXT pulse confirms the new alignment, so a foreign edge cannot hijack the chain.
            pred2 = e + fps_nominal * (rs[j + 1] - rs[j]) / 1000.0
            accept = abs(nearest(pred2) - pred2) <= tol_frames
        if accept:
            edges.append(e)
            pulses.append(rs[j])
            e_prev, t_prev = e, rs[j]
    return np.asarray(edges), np.asarray(pulses), int(i0), int(j0)


def fit_pinstate_truth(up_frames: np.ndarray, rsync_ms: np.ndarray, first_up_frame: Optional[int] = None,
                       fps_pipeline: float = float(sac.VIDEO_FPS)) -> dict:
    """Pinstate truth for one session.

    first_rsync_frame: median, over the first ten matched pairs, of edge - 60 * (t - t_first);
        the frame of the first pulse, robust to a stray foreign edge, and equal to the
        pipeline's `first_up_frame` when the pinstate is healthy and this box's pulse came
        first. fps / f0 / resid_sd: the linear fit over all matched pairs. pipe_drift_max /
        pipe_drift_end: how far (frames) the matched edges stray from the pipeline's model
        first_rsync_frame + 60 * t, i.e. the misalignment the pipeline's own arrays carry
        within this session. anchor_diff: first_up_frame - first_rsync_frame (a few frames
        when the paired box's pulse came first).
    """
    edges, pulses, i0, j0 = match_pulses(up_frames, rsync_ms)
    rs = np.asarray(rsync_ms, float)
    if len(edges) < MIN_PINSTATE_EDGES:
        raise ValueError(f"only {len(edges)} pulses matched")
    k = min(10, len(edges))
    first_rsync_frame = float(np.median(edges[:k] - fps_pipeline * (pulses[:k] - rs[0]) / 1000.0))
    fps, f0 = np.polyfit(pulses / 1000.0, edges, 1)
    resid = edges - (f0 + fps * pulses / 1000.0)
    model60 = first_rsync_frame + fps_pipeline * (pulses - rs[0]) / 1000.0
    dev = edges - model60
    anchor_diff = None if first_up_frame is None else float(first_up_frame - first_rsync_frame)
    return {"first_rsync_frame": first_rsync_frame, "fps": float(fps), "f0": float(f0),
            "resid_sd": float(resid.std()), "resid_max": float(np.abs(resid).max()),
            "n_matched": int(len(edges)), "n_edges": int(len(up_frames)), "n_rsync": int(len(rs)),
            "coverage": float(len(edges) / len(rs)), "first_edge_index": i0, "first_pulse_index": j0,
            "first_up_frame": None if first_up_frame is None else int(first_up_frame),
            "anchor_diff": anchor_diff,
            "anchor_ok": None if anchor_diff is None else bool(abs(anchor_diff) <= 5),
            "pipe_drift_max": float(np.abs(dev).max()), "pipe_drift_end": float(dev[-1]),
            "span_s": float((pulses[-1] - pulses[0]) / 1000.0),
            "clean": bool(resid.std() < CLEAN_RESID_SD and len(edges) >= 0.9 * len(rs))}


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

@dataclass
class SessionRef:
    mouse: str
    recday: str
    session: int
    date: str                       # YYYYMMDD
    tracking_stamp: str             # YYYY-MM-DD-HHMMSS from the metadata Tracking column
    behaviour_stamp: str            # same, Behaviour column
    sleap_stamp: Optional[str]      # matched SLEAP file stamp
    pokes_path: Path
    pycontrol_path: Optional[Path]
    pinstate_path: Optional[Path]
    note: str = ""
    skip: str = ""                  # non-empty -> reason this session cannot be validated


def _stamp(date: str, hhmmss) -> Optional[str]:
    if hhmmss is None or (isinstance(hhmmss, float) and np.isnan(hhmmss)):
        return None
    s = str(hhmmss).strip()
    if s in ("-", "nan", ""):
        return None
    return f"{date[:4]}-{date[4:6]}-{date[6:]}-{int(float(s)):06d}"


def _stamp_dt(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d-%H%M%S")


def roi_file(mouse: str, stamp: str) -> Path:
    return sac.SLEAP_PATH / mouse / f"{mouse}_{stamp}.analysis_ROIs.csv"


def resolve_pinstate(mouse: str, tracking_stamp: str) -> Tuple[Optional[Path], int]:
    """The pinstate the pipeline would use (closest within 5 min, then other-mouse fallback)."""
    tracking_dt = _stamp_dt(tracking_stamp)
    result = sac.find_closest_pinstate_file(mouse, tracking_dt, max_diff_minutes=5)
    path = result[0] if result else None
    n_edges = len(rising_edges(sac.load_pinstate(path))) if path else 0
    if n_edges < MIN_PINSTATE_EDGES:
        fallback = sac.find_pinstate_file_fallback(tracking_stamp, exclude_mouse=mouse)
        if fallback:
            n_fb = len(rising_edges(sac.load_pinstate(fallback[0])))
            if n_fb > n_edges:
                path, n_edges = fallback[0], n_fb
    return path, n_edges


def session_ref(recday: str, session: int) -> SessionRef:
    """Resolve every input file of one (recday, session) the way the pipeline does."""
    mouse, date1, date2 = rr.split_recday(recday)
    pokes_path = TRIALTIMES_PATH / f"pokes_{recday}_{session}.npy"
    rows = rr.task_session_rows(recday)
    ref = SessionRef(mouse, recday, session, "", "", "", None, pokes_path, None, None)
    if session >= len(rows):
        ref.skip = "no metadata row for this session index"
        return ref
    row = rows.iloc[session]
    ref.date = str(row["_date"])
    ref.note = "" if pd.isna(row.get("Notes1")) else str(row.get("Notes1"))
    tracking = _stamp(ref.date, row.get("Tracking"))
    behaviour = _stamp(ref.date, row.get("Behaviour"))
    if tracking is None:
        ref.skip = "no Tracking time in metadata"
        return ref
    ref.tracking_stamp, ref.behaviour_stamp = tracking, behaviour or tracking
    if not pokes_path.exists():
        ref.skip = "no poke table"
        return ref
    try:
        ref.pycontrol_path = rr.pycontrol_file(recday, session)
    except (KeyError, FileNotFoundError, AssertionError) as exc:
        ref.skip = f"pycontrol unresolved ({exc})"
        return ref
    ref.sleap_stamp = sac.find_closest_sleap_datetime(mouse, tracking, max_diff_minutes=2)
    if ref.sleap_stamp is None or not roi_file(mouse, ref.sleap_stamp).exists():
        ref.skip = "no SLEAP ROI file"
        return ref
    ref.pinstate_path, n_edges = resolve_pinstate(mouse, tracking)
    if n_edges < MIN_PINSTATE_EDGES:
        ref.skip = (f"pinstate unusable ({n_edges} rising edges"
                    f"{'' if ref.pinstate_path else ', no file'}) -- a salvage candidate")
    return ref


def discover_sessions(mouse: str) -> List[SessionRef]:
    refs = []
    for path in sorted(TRIALTIMES_PATH.glob(f"pokes_{mouse}_*.npy")):
        _, m, d1, d2, s = path.stem.split("_")
        refs.append(session_ref(f"{m}_{d1}_{d2}", int(s)))
    return sorted(refs, key=lambda r: (r.recday, r.session))


def load_node_codes(mouse: str, stamp: str) -> np.ndarray:
    """Integer location code per video frame from the head_back ROI column."""
    df = pd.read_csv(roi_file(mouse, stamp), usecols=[NODE_COLUMN])
    return locs_to_int(df[NODE_COLUMN].values)


def load_pycontrol_times(path: Path) -> Tuple[Dict[str, np.ndarray], int, int]:
    """(times, first_A_on_ms, first_rsync_ms); first_rsync is 0 when absent, as the pipeline."""
    times, _ = parse_pycontrol(path)
    if "A_on" not in times or len(times["A_on"]) == 0:
        raise ValueError("no A_on in pycontrol file")
    first_rsync = int(times["rsync"][0]) if "rsync" in times and len(times["rsync"]) else 0
    return times, int(times["A_on"][0]), first_rsync


def nominal_f0_range(ref: SessionRef, fps: float) -> Tuple[float, float]:
    """Clock-stamp offset between behaviour and video start, +/- SEARCH_HALF_WIDTH_S."""
    offset_s = (_stamp_dt(ref.behaviour_stamp) - _stamp_dt(ref.tracking_stamp)).total_seconds()
    return ((offset_s - SEARCH_HALF_WIDTH_S) * fps, (offset_s + SEARCH_HALF_WIDTH_S) * fps)


# ---------------------------------------------------------------------------
# Reference sessions (pinstate-backed) and validation
# ---------------------------------------------------------------------------

@dataclass
class GoodSession:
    ref: SessionRef
    codes: np.ndarray
    pokes: np.ndarray
    first_A_on_ms: int
    first_rsync_ms: int
    truth: dict
    latencies_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))


def load_good_session(ref: SessionRef, verbose: bool = False) -> GoodSession:
    codes = load_node_codes(ref.mouse, ref.sleap_stamp)
    pokes = np.load(ref.pokes_path)
    if len(pokes) == 0:
        raise ValueError("empty poke table")
    times, first_A_on, first_rsync = load_pycontrol_times(ref.pycontrol_path)
    if "rsync" not in times:
        raise ValueError("no rsync events in pycontrol file")
    pinstate = sac.load_pinstate(ref.pinstate_path)
    truth = fit_pinstate_truth(rising_edges(pinstate), times["rsync"], sac.find_first_up_state(pinstate))
    good = GoodSession(ref, codes, pokes, first_A_on, first_rsync, truth)
    good.latencies_ms = latencies_at(codes, pokes, first_A_on, truth["fps"], truth["f0"])
    if verbose:
        t = truth
        print(f"  {ref.recday} s{ref.session} ({ref.sleap_stamp}): {len(codes)} frames, {len(pokes)} pokes; "
              f"pinstate: {t['n_matched']}/{t['n_rsync']} pulses matched among {t['n_edges']} edges "
              f"(pulse {t['first_pulse_index']} = edge {t['first_edge_index']}), fps {t['fps']:.4f}, "
              f"resid SD {t['resid_sd']:.2f}, first rsync frame {t['first_rsync_frame']:.1f} "
              f"(first_up_frame {t['first_up_frame']}), pipeline drift max {t['pipe_drift_max']:.0f} frames"
              f"{'' if t['clean'] else '  [DROPPED FRAMES]'}")
    return good


def collect_good_sessions(mouse: str, verbose: bool = True) -> Tuple[List[GoodSession], List[SessionRef]]:
    """All pinstate-backed sessions of a mouse, plus the skipped ones with reasons."""
    good, skipped = [], []
    for ref in discover_sessions(mouse):
        if ref.skip:
            skipped.append(ref)
            continue
        try:
            good.append(load_good_session(ref, verbose=verbose))
        except (ValueError, KeyError) as exc:
            ref.skip = f"could not fit pinstate truth ({exc})"
            skipped.append(ref)
    return good, skipped


def reference_from(sessions: List[GoodSession]) -> Tuple[Optional[float], Optional[float]]:
    """(pooled median latency ms, median fps over clean sessions) for a set of pinstate-backed sessions."""
    if not sessions:
        return None, None
    lat = np.concatenate([s.latencies_ms for s in sessions if len(s.latencies_ms)])
    clean = [s.truth["fps"] for s in sessions if s.truth["clean"]] or [s.truth["fps"] for s in sessions]
    return (float(np.median(lat)) if len(lat) else None), float(np.median(clean))


def lower_offsets(sessions: List[GoodSession], fps: float) -> List[float]:
    """Per-session offset of the true f0 above the estimated plateau lower edge (`lower_offset_of`)."""
    out = []
    for s in sessions:
        try:
            est = estimate_first_rsync_frame(s.codes, s.pokes, s.first_A_on_ms, s.first_rsync_ms, fps, None,
                                             nominal_f0_range(s.ref, fps))
        except ValueError:
            continue
        out.append(lower_offset_of(est, s.truth["first_rsync_frame"], fps, s.first_rsync_ms))
    return out


def validate(mice: List[str] = None, verbose: bool = True) -> pd.DataFrame:
    """Leave-one-session-out validation of the estimator against the pinstate truth."""
    mice = mice or MICE
    rows, skipped_all = [], []
    for mouse in mice:
        print(f"\n{'=' * 100}\n{mouse}\n{'=' * 100}")
        good, skipped = collect_good_sessions(mouse, verbose=verbose)
        skipped_all += skipped
        # Pass 1: plateau + candidates for every session with a leave-one-out latency / fps
        # reference; the lower-edge offset of each session against its own truth.
        firsts, offsets = {}, {}
        for s in good:
            others = [g for g in good if g is not s]
            ref_lat, ref_fps = reference_from(others)
            if ref_fps is None:
                ref_fps = s.truth["fps"]
            try:
                est = estimate_first_rsync_frame(
                    s.codes, s.pokes, s.first_A_on_ms, s.first_rsync_ms, ref_fps, ref_lat,
                    nominal_f0_range(s.ref, ref_fps))
            except ValueError as exc:
                firsts[id(s)] = (ref_lat, ref_fps, None, str(exc))
                continue
            firsts[id(s)] = (ref_lat, ref_fps, est, "")
            offsets[id(s)] = lower_offset_of(est, s.truth["first_rsync_frame"], ref_fps, s.first_rsync_ms)
        # Pass 2: the primary estimate with the leave-one-out lower-edge offset.
        for s in good:
            ref_lat, ref_fps, est, error = firsts[id(s)]
            truth_frame = s.truth["first_rsync_frame"]
            base = {"mouse": mouse, "recday": s.ref.recday, "session": s.ref.session,
                    "video": s.ref.sleap_stamp, "n_pokes": len(s.pokes),
                    "fps_true": s.truth["fps"], "fps_used": ref_fps, "resid_sd": s.truth["resid_sd"],
                    "n_matched": s.truth["n_matched"], "n_rsync": s.truth["n_rsync"], "n_edges": s.truth["n_edges"],
                    "first_pulse": s.truth["first_pulse_index"], "first_edge": s.truth["first_edge_index"],
                    "pipe_drift": s.truth["pipe_drift_max"], "clean_video": s.truth["clean"],
                    "truth": truth_frame, "first_up": s.truth["first_up_frame"],
                    "anchor_diff": s.truth["anchor_diff"]}
            if est is None:
                rows.append({**base, "error": f"estimate failed: {error}"})
                continue
            other_offsets = [v for k, v in offsets.items() if k != id(s)]
            ref_off = float(np.median(other_offsets)) if other_offsets else None
            if ref_off is not None:
                est = estimate_first_rsync_frame(
                    s.codes, s.pokes, s.first_A_on_ms, s.first_rsync_ms, ref_fps, ref_lat,
                    nominal_f0_range(s.ref, ref_fps), ref_lower_offset_frames=ref_off)
            agr_true, _ = agreement_curve(s.codes, poke_entry_times_ms(s.pokes, s.first_A_on_ms),
                                          s.pokes[:, 2], s.truth["fps"], np.array([s.truth["f0"]]))
            width = est["plateau_width"]

            def err(key):
                v = est[key]
                return np.nan if v is None else v - truth_frame

            rows.append({
                **base,
                "est": est["first_rsync_frame"], "err": err("first_rsync_frame"),
                "err_ms": err("first_rsync_frame") / s.truth["fps"] * 1000.0,
                "method": est["method"], "reliable": est["reliable"], "used_frac": est["used_fraction"],
                "own_off": offsets.get(id(s), np.nan), "ref_off": ref_off,
                "err_lower": err("first_rsync_frame_lower"),
                "err_lat": err("first_rsync_frame_latency"), "err_centre": err("first_rsync_frame_centre"),
                "err_all": err("first_rsync_frame_all"), "err_early": err("first_rsync_frame_early"),
                "plateau_w": width,
                "truth_pos": (s.truth["f0"] - est["plateau_lo"]) / width if width > 0 else np.nan,
                "agreement": est["agreement"], "agr_f0": est["agreement_at_f0"], "agr_true": float(agr_true[0]),
                "med_lat": est["median_latency_ms"], "ref_lat": ref_lat,
                "own_lat": float(np.median(s.latencies_ms)) if len(s.latencies_ms) else np.nan,
                "matchable": est["latency_matchable"],
                "drift": " ".join(f"{d:+.0f}" for d in est["quarter_drift"][1:]),
                "flags": "; ".join(est["flags"]), "error": "",
            })
    table = pd.DataFrame(rows)

    print(f"\n{'=' * 100}\nSKIPPED SESSIONS ({len(skipped_all)})\n{'=' * 100}")
    for ref in skipped_all:
        print(f"  {ref.recday} s{ref.session} [{ref.tracking_stamp or '?'}]: {ref.skip}")

    print(f"\n{'=' * 100}\nVALIDATION TABLE (leave-one-session-out reference within mouse; frames unless stated)\n"
          f"  truth/est = first-rsync frame (pinstate / behavioural); err = est - truth for the primary\n"
          f"  (plateau lower edge + the mouse's reference offset ref_off; own_off = this session's true\n"
          f"  offset); used_frac = prefix of pokes the plateau is based on; err_lat/err_centre = latency-\n"
          f"  matched / centre candidates in that plateau, err_all = centre of the all-poke plateau,\n"
          f"  err_early = centre of the 25% prefix plateau; pipe_drift = max departure of the pinstate\n"
          f"  edges from the pipeline's 60 fps model; drift = quarter-wise plateau centres relative to\n"
          f"  the first quarter\n{'=' * 100}")
    if table.empty:
        print("no sessions validated")
        return table
    show = ["recday", "session", "video", "n_pokes", "n_matched", "n_rsync", "fps_true", "resid_sd", "pipe_drift",
            "truth", "anchor_diff", "est", "err", "err_ms", "own_off", "ref_off", "used_frac", "reliable",
            "err_lat", "err_centre", "err_all", "err_early", "plateau_w", "agreement", "agr_f0", "agr_true",
            "own_lat", "ref_lat", "drift", "flags", "error"]
    fmt = {"fps_true": 4, "resid_sd": 2, "pipe_drift": 0, "truth": 1, "anchor_diff": 0, "err": 1, "err_ms": 0,
           "own_off": 1, "ref_off": 1, "used_frac": 3, "err_lat": 1, "err_centre": 1, "err_all": 1,
           "err_early": 1, "plateau_w": 0, "agreement": 3, "agr_f0": 3, "agr_true": 3, "own_lat": 0, "ref_lat": 0}
    with pd.option_context("display.width", 400, "display.max_rows", 500, "display.max_colwidth", 60):
        print(table[[c for c in show if c in table.columns]].round(fmt).to_string(index=False))

    ok = table[table["error"] == ""].copy()
    failed_est = table[table["error"] != ""]
    e = ok["err"].abs()
    print(f"\n{'=' * 100}\nSUMMARY over {len(ok)} sessions, {ok['mouse'].nunique()} mice\n{'=' * 100}")
    print(f"  primary estimate         : median |err| {e.median():.1f} frames, p90 {e.quantile(.9):.1f}, max |err| "
          f"{e.max():.1f} frames, {(e <= GATE_FRAMES).sum()}/{len(ok)} within {GATE_FRAMES} frames; signed mean "
          f"{ok['err'].mean():+.1f}, SD {ok['err'].std():.1f}")
    for key, label in (("err_lower", "lower edge + ref offset"), ("err_lat", "latency-matched        "),
                       ("err_centre", "chosen-plateau centre  "), ("err_all", "all-poke plateau centre"),
                       ("err_early", "25%-prefix centre      ")):
        v = ok[key].abs().dropna()
        print(f"  {label}  : median |err| {v.median():.1f}, p90 {v.quantile(.9):.1f}, max |err| {v.max():.1f}, "
              f"{(v <= GATE_FRAMES).sum()}/{len(v)} within {GATE_FRAMES} (n={len(v)})")
    print(f"  method used              : {ok['method'].value_counts().to_dict()}; "
          f"flagged unreliable: {(~ok['reliable']).sum()}; prefix truncated (used_frac < 1): "
          f"{(ok['used_frac'] < 1).sum()}")
    print(f"  lower-edge offset        : truth - plateau_lo median {ok['own_off'].median():.1f} frames "
          f"(IQR {ok['own_off'].quantile(.25):.1f}-{ok['own_off'].quantile(.75):.1f}); per mouse "
          + ", ".join(f"{m} {g['own_off'].median():.1f} (IQR {g['own_off'].quantile(.75) - g['own_off'].quantile(.25):.1f})"
                      for m, g in ok.groupby('mouse')))
    print(f"  plateau width            : median {ok['plateau_w'].median():.0f} frames "
          f"[{ok['plateau_w'].min():.0f}, {ok['plateau_w'].max():.0f}]")
    print(f"  agreement at truth       : median {ok['agr_true'].median():.3f}, min {ok['agr_true'].min():.3f}")
    print(f"  fitted fps               : median {ok['fps_true'].median():.4f} [{ok['fps_true'].min():.4f}, {ok['fps_true'].max():.4f}]; "
          f"videos with dropped frames (resid SD >= {CLEAN_RESID_SD} or pulses lost): {(~ok['clean_video']).sum()}; "
          f"pipeline drift > {GATE_FRAMES} frames in {(ok['pipe_drift'] > GATE_FRAMES).sum()} sessions (max {ok['pipe_drift'].max():.0f})")
    ad = ok["anchor_diff"].dropna()
    print(f"  pipeline anchor          : first_up_frame - first pulse frame: median {ad.median():+.1f}, "
          f"range [{ad.min():+.0f}, {ad.max():+.0f}] frames over {len(ad)} sessions "
          f"({(ad.abs() > 2).sum()} off by > 2 frames: the paired box's pulse came first); "
          f"pulse 0 missing from the pinstate in {(ok['first_pulse'] > 0).sum()} session(s)")
    bad_anchor = ok[ok["anchor_diff"].abs() > 5]
    for _, r in bad_anchor.iterrows():
        print(f"      {r['recday']} s{r['session']}: first_up_frame {r['first_up']} vs first pulse {r['truth']:.0f} "
              f"(pulse {r['first_pulse']} = edge {r['first_edge']}) -- the pipeline anchored on the wrong edge")
    for mouse, g in ok.groupby("mouse"):
        v = g["err"].abs()
        print(f"    {mouse}: {len(g)} sessions, median |err| {v.median():.1f}, max |err| {v.max():.1f}, "
              f"{(v <= GATE_FRAMES).sum()} within gate; ref latency {g['ref_lat'].median():.0f} ms "
              f"(own medians {g['own_lat'].min():.0f}-{g['own_lat'].max():.0f}); fps {g['fps_used'].median():.4f}")

    bad = ok[e > GATE_FRAMES]
    print()
    if len(bad) or len(failed_est):
        unflagged = bad[bad["reliable"]]
        flagged = bad[~bad["reliable"]]
        print(f"GATE FAILED: {len(bad)} session(s) with |err| > {GATE_FRAMES} frames "
              f"({len(unflagged)} NOT flagged by the estimator, {len(flagged)} flagged unreliable), "
              f"{len(failed_est)} with no estimate")
        for label, sub in (("UNFLAGGED", unflagged), ("flagged", flagged)):
            for _, r in sub.iterrows():
                print(f"  FAIL [{label}] {r['recday']} s{r['session']} ({r['video']}): err {r['err']:+.0f} frames "
                      f"({r['err_ms']:+.0f} ms) by {r['method']} on the first {r['used_frac']:.0%} of pokes "
                      f"(own offset {r['own_off']:.0f} vs ref {r['ref_off']:.0f}); other candidates lat "
                      f"{r['err_lat']:+.0f} / centre {r['err_centre']:+.0f} / all-poke {r['err_all']:+.0f} / 25% "
                      f"{r['err_early']:+.0f}; plateau {r['plateau_w']:.0f} wide, agreement {r['agreement']:.3f}; "
                      f"pinstate fps {r['fps_true']:.4f}, pipeline drift {r['pipe_drift']:.0f} frames; drift "
                      f"profile {r['drift']}; {r['flags'] or 'no flags'}")
        for _, r in failed_est.iterrows():
            print(f"  FAIL {r['recday']} s{r['session']}: {r['error']}")
    else:
        print(f"GATE PASSED: all {len(ok)} sessions within {GATE_FRAMES} frames")
    flagged_ok = ok[(~ok["reliable"]) & (e <= GATE_FRAMES)]
    if len(flagged_ok):
        print(f"  ({len(flagged_ok)} flagged-unreliable session(s) nevertheless within the gate)")

    STAGING_PATH.mkdir(parents=True, exist_ok=True)
    out = STAGING_PATH / "salvage_validation.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return table


# ---------------------------------------------------------------------------
# Producing the arrays exactly as the pipeline does
# ---------------------------------------------------------------------------

def first_A_on_frame_pipeline(first_rsync_frame: int, first_A_on_ms: int, first_rsync_ms: int) -> int:
    """`process_session_by_metadata` lines 1344-1346, verbatim arithmetic (60 fps, int truncation)."""
    time_to_A_on_ms = first_A_on_ms - first_rsync_ms
    return int(first_rsync_frame) + int(time_to_A_on_ms * sac.VIDEO_FPS / 1000)


def build_tracking_arrays(sleap_data: Dict[str, pd.DataFrame], first_A_on_frame: int) -> Dict[str, np.ndarray]:
    """XY / HD / Locs from loaded SLEAP frames, as `process_session_by_metadata` lines 1361-1406."""
    out = {}
    if "coordinates" in sleap_data:
        coords_df = sleap_data["coordinates"]
        head_back_cols = ["head_back.x", "head_back.y"]
        if all(col in coords_df.columns for col in head_back_cols):
            coords = coords_df[head_back_cols].values
        else:
            coords = coords_df.values[:, 4:6]
        if 0 <= first_A_on_frame < len(coords):
            out["XY"] = sac.resample_data(coords[first_A_on_frame:], sac.VIDEO_FPS, sac.TARGET_FPS)
    if "head_direction" in sleap_data:
        hd = sleap_data["head_direction"].values
        if 0 <= first_A_on_frame < len(hd):
            out["HD"] = sac.resample_data(hd[first_A_on_frame:], sac.VIDEO_FPS, sac.TARGET_FPS)
    if "rois" in sleap_data:
        rois_df = sleap_data["rois"]
        rois = rois_df["head_back"].values if "head_back" in rois_df.columns else rois_df.values[:, 2]
        if 0 <= first_A_on_frame < len(rois):
            rois_trimmed = rois[first_A_on_frame:]
            original_samples = len(rois_trimmed)
            target_samples = int(original_samples * sac.TARGET_FPS / sac.VIDEO_FPS)
            indices = np.linspace(0, original_samples - 1, target_samples).astype(int)
            out["Locs"] = rois_trimmed[indices]
    return out


def output_names(recday: str, session: int) -> Dict[str, str]:
    mouse, d1, d2 = rr.split_recday(recday)
    return {k: sac.create_output_filename(k, mouse, d1, d2, session) for k in ("XY", "HD", "Locs")}


def alignment_checks(locs_codes: np.ndarray, recday: str, session: int) -> dict:
    """`validate_alignment.check_session` checks 1 and 2 on an in-memory Locs array."""
    tt_path = TRIALTIMES_PATH / f"trialtimes_{recday}_{session}.npy"
    task_path = TRIALTIMES_PATH / f"Task_data_{recday}_{session}.npy"
    poke_path = TRIALTIMES_PATH / f"pokes_{recday}_{session}.npy"
    neuron_path = NEURON_PATH / f"Neuron_raw_{recday}_{session}.npy"
    out = {"loc_at_goal": np.nan, "n_goal_checks": 0, "poke_loc": np.nan, "n_poke_bins": 0,
           "n_locs": int(len(locs_codes)), "n_bins": None}
    if tt_path.exists() and task_path.exists():
        trial_times = np.load(tt_path).astype(int)
        task = np.load(task_path, allow_pickle=True)
        hits = total = 0
        for state in range(min(trial_times.shape[1] - 1, len(task))):
            for t in trial_times[:, state]:
                if 0 <= t < len(locs_codes):
                    total += 1
                    hits += int(locs_codes[t] == task[state])
        if total:
            out["loc_at_goal"], out["n_goal_checks"] = hits / total, total
    if poke_path.exists():
        pokes = np.load(poke_path)
        hits = total = 0
        for entry, exit_, port, _r, _s in pokes:
            lo, hi = min(int(entry), len(locs_codes) - 1), min(int(exit_), len(locs_codes) - 1)
            if hi < lo:
                continue
            seg = locs_codes[lo:hi + 1]
            hits += int((seg == port).sum())
            total += len(seg)
        if total:
            out["poke_loc"], out["n_poke_bins"] = hits / total, total
    if neuron_path.exists():
        out["n_bins"] = int(np.load(neuron_path, mmap_mode="r").shape[1])
    return out


def resolve_video_session(mouse: str, video_stamp: str) -> Tuple[str, int, SessionRef]:
    """(recday, session, ref) for a mouse + video timestamp 'YYYY-MM-DD-HHMMSS'."""
    date = video_stamp[:10].replace("-", "")
    target = _stamp_dt(video_stamp)
    for recday in rr.all_recdays():
        m, d1, d2 = rr.split_recday(recday)
        if m != mouse or date not in (d1, d2):
            continue
        rows = rr.task_session_rows(recday)
        for session, row in rows.iterrows():
            stamp = _stamp(str(row["_date"]), row.get("Tracking"))
            if stamp and abs(_stamp_dt(stamp) - target) <= timedelta(minutes=2):
                return recday, int(session), session_ref(recday, int(session))
    raise KeyError(f"no metadata session of {mouse} has a Tracking time within 2 min of {video_stamp}")


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items() if k not in ("f0_grid", "agreement_grid")}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def salvage_session(mouse: str, video_stamp: str, allow_unreliable: bool = False, verbose: bool = True) -> dict:
    """Estimate the offset of a broken session and stage its tracking arrays."""
    recday, session, ref = resolve_video_session(mouse, video_stamp)
    print(f"\n{'=' * 100}\nSALVAGE {recday} session {session}  video {video_stamp}\n{'=' * 100}")
    print(f"  metadata: Tracking {ref.tracking_stamp}, Behaviour {ref.behaviour_stamp}, note: {ref.note!r}")
    if ref.skip and not ref.skip.startswith("pinstate unusable"):
        raise RuntimeError(f"cannot salvage: {ref.skip}")
    if not ref.skip:
        print("  NOTE: this session has a usable pinstate; salvaging anyway (the pinstate truth is "
              "printed for comparison)")
    print(f"  pycontrol: {ref.pycontrol_path.name}; pinstate: "
          f"{ref.pinstate_path.name if ref.pinstate_path else 'none'} -> {ref.skip or 'usable'}")

    # Reference from this mouse's pinstate-backed sessions (excluding this one if it has one).
    print(f"\n  reference sessions for {mouse}:")
    good, _ = collect_good_sessions(mouse, verbose=verbose)
    good = [g for g in good if not (g.ref.recday == recday and g.ref.session == session)]
    ref_lat, ref_fps = reference_from(good)
    if ref_lat is None:
        raise RuntimeError("no pinstate-backed reference sessions for this mouse")
    same_day = [g for g in good if g.ref.date == ref.date]
    ref_lat_day, _ = reference_from(same_day)
    per_session_meds = [float(np.median(g.latencies_ms)) for g in good if len(g.latencies_ms)]
    offs = lower_offsets(good, ref_fps)
    ref_off = float(np.median(offs)) if offs else None
    print(f"\n  reference: {len(good)} sessions ({sum(g.truth['clean'] for g in good)} clean videos); median fps over "
          f"clean videos {ref_fps:.4f}")
    print(f"    lower-edge offset (truth f0 - plateau lower edge): median {ref_off:.1f} frames, IQR "
          f"{np.percentile(offs, 25):.1f}-{np.percentile(offs, 75):.1f}, range {min(offs):.1f}-{max(offs):.1f} "
          f"over {len(offs)} sessions  <- the primary estimate adds this to the plateau lower edge")
    print(f"    pooled median arrival latency {ref_lat:.0f} ms"
          + (f" (same-day sessions, n={len(same_day)}: {ref_lat_day:.0f} ms)" if ref_lat_day else "")
          + f"; per-session medians {np.round(per_session_meds).astype(int).tolist()} ms")

    # Inputs of the broken session.
    codes = load_node_codes(mouse, ref.sleap_stamp)
    pokes = np.load(ref.pokes_path)
    times, first_A_on, first_rsync = load_pycontrol_times(ref.pycontrol_path)
    f0_range = nominal_f0_range(ref, ref_fps)
    print(f"\n  session: {len(codes)} video frames ({np.mean(codes > 0):.2f} tracked), {len(pokes)} pokes, "
          f"first A_on {first_A_on} ms, first rsync {first_rsync} ms, {len(times.get('rsync', []))} rsync pulses "
          f"in pycontrol, last event {max(v.max() for v in times.values()) / 1000:.0f} s")
    print(f"  searching f0 in [{f0_range[0]:.0f}, {f0_range[1]:.0f}] frames (clock stamps say "
          f"{(_stamp_dt(ref.behaviour_stamp) - _stamp_dt(ref.tracking_stamp)).total_seconds():.0f} s)")

    est = estimate_first_rsync_frame(codes, pokes, first_A_on, first_rsync, ref_fps, ref_lat, f0_range,
                                     ref_lower_offset_frames=ref_off)
    print(f"\n  ESTIMATE ({est['method']}): f0 = {est['f0']:.0f} frames ({est['f0'] / ref_fps:.2f} s after video "
          f"start); first_rsync_frame = {est['first_rsync_frame']}")
    print(f"    plateau [{est['plateau_lo']:.0f}, {est['plateau_hi']:.0f}] ({est['plateau_width']:.0f} frames) over the "
          f"first {est['used_fraction']:.0%} of pokes ({est['n_pokes_used']}/{est['n_pokes']}), best agreement "
          f"{est['agreement']:.3f}, agreement at f0 {est['agreement_at_f0']:.3f}; all-poke plateau "
          f"[{est['plateau_all'][0]:.0f}, {est['plateau_all'][1]:.0f}] agreement {est['agreement_all']:.3f}")
    print("    prefix plateaus: " + "; ".join(f"{p['frac']:.0%} [{p['lo']:.0f},{p['hi']:.0f}] {p['best']:.3f}"
                                            for p in est["prefix_plateaus"]))
    print(f"    lower edge {est['plateau_lo']:.0f} + reference offset {ref_off:.1f} = f0 {est['f0_lower']:.1f}")
    print(f"    other candidates: latency-matched f0 {est['f0_latency']:.0f} (median latency "
          f"{est['median_latency_ms']:.0f} ms vs reference {ref_lat:.0f}; "
          f"{'matchable' if est['latency_matchable'] else 'NOT matchable inside the plateau'}"
          f"{', at plateau edge' if est['at_plateau_edge'] else ''}); plateau centre f0 {est['f0_centre']:.0f}; "
          f"all-poke centre {est['f0_all']:.0f}; 25%-prefix centre {est['f0_early']:.0f}")
    print(f"    drift profile (quarter centres relative to Q1): {' '.join(f'{d:+.0f}' for d in est['quarter_drift'][1:])} frames")
    print(f"    reliability: {'OK' if est['reliable'] else 'UNRELIABLE -- ' + '; '.join(est['flags'])}")
    if offs:
        lo_q, hi_q = np.percentile(offs, 25), np.percentile(offs, 75)
        print(f"    offset IQR of the reference maps to f0 in [{est['plateau_lo'] + lo_q:.0f}, {est['plateau_lo'] + hi_q:.0f}]")
    for fps_alt in (ref_fps - 0.005, ref_fps + 0.003):
        e2 = estimate_first_rsync_frame(codes, pokes, first_A_on, first_rsync, fps_alt, ref_lat, f0_range,
                                        ref_lower_offset_frames=ref_off)
        print(f"    fps sensitivity: at {fps_alt:.4f} fps -> f0 {e2['f0']:.0f}, plateau "
              f"[{e2['plateau_lo']:.0f}, {e2['plateau_hi']:.0f}], agreement {e2['agreement']:.3f}")
    grid, agr = est["f0_grid"], est["agreement_grid"]
    off = agr[(grid < est["plateau_lo"] - 120) | (grid > est["plateau_hi"] + 120)]
    off = off[np.isfinite(off)]
    if len(off):
        print(f"    agreement > 120 frames away from the plateau: max {off.max():.3f}, median {np.median(off):.3f} "
              f"(the plateau is unique)")

    truth = None
    if not ref.skip:  # pinstate exists: show the truth for comparison
        pinstate = sac.load_pinstate(ref.pinstate_path)
        truth = fit_pinstate_truth(rising_edges(pinstate), times["rsync"], sac.find_first_up_state(pinstate))
        print(f"    pinstate truth for comparison: first rsync frame {truth['first_rsync_frame']:.1f} -> "
              f"error {est['first_rsync_frame'] - truth['first_rsync_frame']:+.1f} frames")

    if not est["reliable"] and not allow_unreliable:
        raise RuntimeError("estimate flagged unreliable; nothing staged (pass --allow-unreliable to override)")

    # Cut and resample exactly as the pipeline would with first_up_frame := first_rsync_frame.
    first_A_on_frame = first_A_on_frame_pipeline(est["first_rsync_frame"], first_A_on, first_rsync)
    sleap_data, used_stamp = sac.load_sleap_files(mouse, ref.sleap_stamp)
    if not sleap_data or used_stamp != ref.sleap_stamp:
        raise RuntimeError(f"SLEAP files for {ref.sleap_stamp} not loaded")
    arrays = build_tracking_arrays(sleap_data, first_A_on_frame)
    names = output_names(recday, session)
    STAGING_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\n  first_A_on_frame = {first_A_on_frame} (pipeline arithmetic: {est['first_rsync_frame']} + "
          f"int({first_A_on - first_rsync} * 60 / 1000))")
    for key in ("XY", "HD", "Locs"):
        if key not in arrays:
            print(f"    {key}: NOT PRODUCED (missing SLEAP file or A_on frame outside video)")
            continue
        path = STAGING_PATH / names[key]
        np.save(path, arrays[key])
        a = arrays[key]
        print(f"    saved {path.name}: shape {a.shape}, dtype {a.dtype}")

    checks = {}
    if "Locs" in arrays:
        codes_staged = locs_to_int(arrays["Locs"])
        checks = alignment_checks(codes_staged, recday, session)
        n_bins = checks["n_bins"]
        print(f"\n  alignment checks on the staged Locs (as validate_alignment.py computes them):")
        print(f"    loc_at_goal = {checks['loc_at_goal']:.3f} over {checks['n_goal_checks']} goal times "
              f"(cohort floor 0.60; aligned sessions score 0.944-1.000)")
        print(f"    poke_loc    = {checks['poke_loc']:.3f} over {checks['n_poke_bins']} poke bins (tracking-quality measure)")
        if n_bins:
            print(f"    Locs bins {len(codes_staged)} vs Neuron_raw bins {n_bins}: tracking outruns neurons by "
                  f"{len(codes_staged) - n_bins} bins (normal sessions: +200..+4400)")
        if ALIGNMENT_QC_PATH.exists():
            qc = pd.read_csv(ALIGNMENT_QC_PATH)
            same = qc[(qc["recday"] == recday) & qc["loc_at_goal"].notna()]
            if len(same):
                print(f"    other sessions of {recday} on disk: loc_at_goal "
                      f"{same['loc_at_goal'].min():.3f}-{same['loc_at_goal'].max():.3f}, poke_loc "
                      f"{same['poke_loc'].min():.3f}-{same['poke_loc'].max():.3f}")

    sidecar = {
        "recday": recday, "session": session, "mouse": mouse, "video_stamp": video_stamp,
        "sleap_stamp": ref.sleap_stamp, "pycontrol_file": ref.pycontrol_path.name,
        "pinstate_file": ref.pinstate_path.name if ref.pinstate_path else None,
        "pinstate_status": ref.skip or "usable", "metadata_note": ref.note,
        "method": ("poke-entry / tracked-head_back agreement plateau over f0, on the largest prefix of "
                   "pokes whose plateaus stay mutually consistent (dropped frames only shrink the upper "
                   "edge of later pokes); f0 = plateau lower edge + the mouse's reference offset "
                   "(median over its pinstate-backed sessions of true f0 - plateau lower edge). "
                   "Latency-matched and plateau-centre candidates reported for comparison. Arrays cut "
                   "with the pipeline's 60 fps arithmetic from the estimated first-rsync frame."),
        "estimate": _json_ready(est),
        "pinstate_truth_if_any": _json_ready(truth),
        "reference": {"n_sessions": len(good), "sessions": [f"{g.ref.recday}_s{g.ref.session}" for g in good],
                      "lower_offset_frames_median": ref_off, "lower_offset_frames_per_session": offs,
                      "pooled_median_latency_ms": ref_lat, "median_fps_clean": ref_fps,
                      "same_day_median_latency_ms": ref_lat_day,
                      "per_session_median_latency_ms": per_session_meds},
        "first_A_on_ms": first_A_on, "first_rsync_ms": first_rsync,
        "first_A_on_frame": first_A_on_frame, "video_frames": int(len(codes)), "n_pokes": int(len(pokes)),
        "outputs": {k: {"file": names[k], "shape": list(arrays[k].shape), "dtype": str(arrays[k].dtype)}
                    for k in arrays},
        "alignment_checks": _json_ready(checks),
        "validated_accuracy": "python salvage_tracking_offset.py --validate (salvage_staging/salvage_validation.csv)",
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    sidecar_path = STAGING_PATH / f"salvage_{recday}_{session}.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\n  wrote sidecar {sidecar_path}")
    return sidecar


# ---------------------------------------------------------------------------
# Writing into the pipeline (never run automatically)
# ---------------------------------------------------------------------------

def write_to_pipeline(recday: str, session: int, dry_run: bool = True, force: bool = False) -> None:
    """Copy staged arrays into `data/processed/{mouse}/{d1}_{d2}/` and insert them into data_dic_lec.pkl.

    With `dry_run` (the default) only prints what would happen. The pickle is backed up first.
    Refuses to overwrite existing tracking files or an entry that already has tracking unless
    `force`.
    """
    mouse, d1, d2 = rr.split_recday(recday)
    names = output_names(recday, session)
    target_dir = TRACKING_PATH / mouse / f"{d1}_{d2}"
    sidecar = STAGING_PATH / f"salvage_{recday}_{session}.json"
    backup = DATA_DIC_PATH.with_name(DATA_DIC_PATH.name + f".PRE_salvage_{recday}_s{session}")
    staged = {k: STAGING_PATH / n for k, n in names.items()}

    print(f"\n{'=' * 100}\n{'DRY RUN: ' if dry_run else ''}--write plan for {recday} session {session}\n{'=' * 100}")
    missing = [str(p) for p in list(staged.values()) + [sidecar] if not p.exists()]
    print(f"  1. require staged files in {STAGING_PATH}/: {', '.join(p.name for p in staged.values())} + {sidecar.name}"
          + (f"\n     MISSING: {missing}" if missing else "\n     all present"))
    exists = [p.name for p in (target_dir / n for n in names.values()) if p.exists()]
    print(f"  2. copy them into {target_dir}/"
          + (f"\n     REFUSE: already present: {exists}" if exists and not force else "\n     none present there yet"))
    size_gb = DATA_DIC_PATH.stat().st_size / 1e9 if DATA_DIC_PATH.exists() else float("nan")
    print(f"  3. back up {DATA_DIC_PATH.name} ({size_gb:.2f} GB) -> {backup.name}")
    print(f"  4. load {DATA_DIC_PATH.name}; in data_dic['{recday}'][{session}] set "
          f"HD_raw (float64 (T,2)), XY_raw (float64 (T,2)), Locs_raw (locs_to_int -> int64 (T,)), "
          f"location_mapping, reverse_mapping; keep build_data_dic's key order; dump back "
          f"(refuse if the entry already has tracking)")
    print(f"     norm_neurons_dic / session_inds_dic / tasks_dic do not hold tracking and are untouched")
    print(f"  5. afterwards run: python validate_alignment.py --recdays {recday} --detail")
    if dry_run:
        print("\n  nothing written (pass --write together with --session to execute)")
        return
    if missing:
        raise FileNotFoundError(f"staged files missing: {missing}")
    if exists and not force:
        raise FileExistsError(f"tracking files already exist in {target_dir}: {exists}")
    with open(DATA_DIC_PATH, "rb") as f:
        data_dic = pickle.load(f)
    entry = data_dic.get(recday, {}).get(session)
    if entry is None:
        raise KeyError(f"data_dic has no entry {recday}[{session}]")
    if any(k in entry for k in ("HD_raw", "XY_raw", "Locs_raw")) and not force:
        raise FileExistsError(f"data_dic['{recday}'][{session}] already has tracking")

    shutil.copy2(DATA_DIC_PATH, backup)
    print(f"  backed up -> {backup}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for key, src in staged.items():
        shutil.copy2(src, target_dir / names[key])
        print(f"  copied {names[key]}")
    entry["HD_raw"] = np.load(target_dir / names["HD"], allow_pickle=True)
    entry["XY_raw"] = np.load(target_dir / names["XY"], allow_pickle=True)
    entry["Locs_raw"] = locs_to_int(np.load(target_dir / names["Locs"], allow_pickle=True))
    entry["location_mapping"] = LOCATION_MAPPING
    entry["reverse_mapping"] = REVERSE_MAPPING
    order = ["Neuron_raw", "Task", "Trial_times", "num_trials", "num_neurons",
             "Neurons_norm", "Neurons_mean", "Mean_norm", "Smoothed_norm",
             "Std_err_smooth", "HD_raw", "XY_raw", "Locs_raw",
             "location_mapping", "reverse_mapping"]
    data_dic[recday][session] = {k: entry[k] for k in order if k in entry}
    data_dic[recday][session].update({k: v for k, v in entry.items() if k not in order})
    with open(DATA_DIC_PATH, "wb") as f:
        pickle.dump(data_dic, f)
    print(f"  wrote {DATA_DIC_PATH}")


# ---------------------------------------------------------------------------
# Replication check
# ---------------------------------------------------------------------------

def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.allclose(a, b, equal_nan=True))
    if a.dtype.kind == "O":                       # object arrays: NaN != NaN, compare as text
        return bool(np.array_equal(a.astype(str), b.astype(str)))
    return bool(np.array_equal(a, b))


def replicate(mouse: str, video_stamp: str) -> bool:
    """Re-cut a pinstate-backed session with its own anchor and diff against the arrays on disk."""
    recday, session, ref = resolve_video_session(mouse, video_stamp)
    print(f"\n{'=' * 100}\nREPLICATION {recday} session {session}  video {video_stamp}\n{'=' * 100}")
    if ref.skip:
        print(f"  cannot replicate: {ref.skip}")
        return False
    times, first_A_on, first_rsync = load_pycontrol_times(ref.pycontrol_path)
    pinstate = sac.load_pinstate(ref.pinstate_path)
    first_up = int(sac.find_first_up_state(pinstate))
    frame = first_A_on_frame_pipeline(first_up, first_A_on, first_rsync)
    sleap_data, _ = sac.load_sleap_files(mouse, ref.sleap_stamp)
    arrays = build_tracking_arrays(sleap_data, frame)
    names = output_names(recday, session)
    d1, d2 = rr.split_recday(recday)[1:]
    ok = True
    for key in ("XY", "HD", "Locs"):
        disk_path = TRACKING_PATH / mouse / f"{d1}_{d2}" / names[key]
        if not disk_path.exists() or key not in arrays:
            print(f"  {key}: no comparison ({'not on disk' if not disk_path.exists() else 'not produced'})")
            ok = False
            continue
        disk = np.load(disk_path, allow_pickle=True)
        mine = arrays[key]
        same = _arrays_equal(disk, mine)
        ok &= same
        print(f"  {key}: disk {disk.shape} {disk.dtype} vs staged {mine.shape} {mine.dtype} -> "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
    print(f"  first_up_frame {first_up}, first_A_on_frame {frame}: {'replication exact' if ok else 'MISMATCH'}")
    return ok


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true",
                        help="leave-one-out validation on every pinstate-backed session")
    parser.add_argument("--mice", nargs="*", default=None, help="restrict --validate to these mice")
    parser.add_argument("--session", nargs=2, metavar=("MOUSE", "STAMP"),
                        help="salvage one session, e.g. --session ah10 2025-06-19-143522")
    parser.add_argument("--allow-unreliable", action="store_true",
                        help="with --session: stage arrays even if the estimate is flagged unreliable")
    parser.add_argument("--write", action="store_true",
                        help="with --session: copy the staged arrays into the pipeline and data_dic_lec.pkl "
                             "(backs the pickle up first). Without it the plan is printed only.")
    parser.add_argument("--force", action="store_true", help="with --write: overwrite existing tracking")
    parser.add_argument("--replicate", nargs=2, metavar=("MOUSE", "STAMP"),
                        help="re-cut a pinstate-backed session and diff against disk")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    if not (args.validate or args.session or args.replicate):
        parser.error("one of --validate, --session, --replicate is required")

    if args.replicate:
        if not replicate(*args.replicate):
            sys.exit(1)

    if args.validate:
        table = validate(args.mice, verbose=verbose)
        ok = table[table["error"] == ""] if len(table) else table
        if len(table) == 0 or (ok["err"].abs() > GATE_FRAMES).any() or (table["error"] != "").any():
            sys.exit(1)

    if args.session:
        mouse, stamp = args.session
        sidecar = salvage_session(mouse, stamp, allow_unreliable=args.allow_unreliable, verbose=verbose)
        write_to_pipeline(sidecar["recday"], sidecar["session"], dry_run=not args.write, force=args.force)


if __name__ == "__main__":
    main()
