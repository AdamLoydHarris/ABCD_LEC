"""
Tracking cleaning v2: physical rules in cm, a conditional confidence gate, run-limited bridging,
a re-masked rolling mean, and a per-frame provenance flag for every bodypart.

Shared by `sleap_to_rois.py --clean v2`, `openfield_to_xy.py --clean v2`, `tracking_qc.py` and
`tracking_clean_synthetics.py`. numpy / pandas / h5py only (no cv2), so it runs in both the
`maze_ephys` and `histology` envs.

Why v2 (measured 2026-09-20/23, `docs/handoff/TRACKING_QC_PLAN.md`, `TRACKING_QC.md`):
- v1 `drop_jumps` (median + 3 MAD of the frame-to-frame displacement, threshold from the first
  bodypart) is mobility-dependent: on maze sessions the mouse is mostly still, the threshold
  collapses to 0.3-1.1 px/frame and the rule deletes 58-97 % of the frames moving > 5 cm/s.
  v2 thresholds are physical (cm, cm/s) and never move with the mouse's behaviour.
- The SLEAP scores were never read. A hard gate at 0.5 would mostly remove frames that sit
  where interpolation puts them anyway (11 of 12 sessions: p50 deviation 0.0-0.6 cm), so the
  gate is conditional. Each run of low-score frames is judged against the good frames that
  flank it (chord = straight line between them):
    short runs (<= lowscore_long_run frames, 0.25 s): a frame more than thr + 0.25 x |chord|
        off the chord is removed -- over 0.25 s the chord is where the mouse was;
    long runs: a frame more than thr + 0.75 x |chord| off the chord is removed (a true path
        deviates by at most 0.5 x chord, the L-turn), and the WHOLE run is removed when the
        path length inside it is below 0.9 x |chord| - thr: a true trajectory always covers at
        least the chord, a static mislocalisation (SLEAP on the cable or an object while the
        mouse moved on) covers nothing. A U-turn under low score (chord ~ 0) is rejected by the
        first test and bridged -- what the hard gate would have done to every low-score frame.
  Frames that pass are kept and flagged. A Hampel window cannot do this: inside a 20-frame run
  the median IS the run.
- v1 interpolated every gap with no length limit. v2 bridges runs <= 120 frames (2 s) by a
  straight line and leaves longer runs NaN; genuinely missing tracking beyond 2 s is 0.03-0.55 %
  of frames in every arena, and the frame-clock mask downstream already writes NaN.

Rules, in order, per bodypart, on the cm track:
  gate   low-score = point score < point_min, or instance score < instance_min (every bodypart
         of that frame). Each run of low-score frames is compared with the chord between its
         flanking good frames (see above): off-chord frames, and long runs whose path falls
         short of the chord -> NaN (reason 7); runs touching the start or end of the video, or
         a NaN run, have no chord and are kept. First, so that a rejected run is never R1's
         reference. A spike sitting exactly on a flank would mislead the chord; R1-R3 still
         remove the spike, the run's verdict is then wrong for that one run (rare, accepted).
  R1     velocity: |p_i - p_j| > v_max * (i - j) / fps against the previous VALID frame j -> NaN
  R2     out-and-back spike: d_in, d_out > spike_thr_cm and the two-frame net move
         < 0.5 * min(d_in, d_out) -> the middle frame NaN
  R3     Hampel: |p_i - median(window)| > hampel_thr_cm (3 cm) -> NaN
  bridge np.interp across interior NaN runs <= max_gap_frames; longer runs and leading/trailing
         runs stay NaN
  smooth centred rolling mean (smooth_window, min_periods 1), then the pre-smoothing NaN mask is
         re-imposed so no gap shrinks

Flag codes (int8, one per frame per bodypart; tens digit = outcome, units digit = reason):
    0  kept, nothing to say            2  kept, low point score       3  kept, low instance score
   1r  removed for reason r and bridged      2r  removed for reason r and left NaN
   reasons r: 1 raw NaN, 4 R1 velocity, 5 R2 spike, 6 R3 Hampel, 7 low score AND off trajectory
`outcome(flags)` and `reason(flags)` split them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pandas as pd

#: Constants of cleaning v2. User decisions of 2026-09-20 (v_max, max_gap) and 2026-09-23
#: (conditional gate); the rest from the planning agent, to be confirmed by tracking_qc.py.
V2 = dict(
    clean_version="v2",
    fps=60.0,                    # nominal; cameras run 59.95-60.00, irrelevant for a 100 cm/s cap
    point_min=0.5,
    instance_min=0.5,
    v_max_cm_s=100.0,
    spike_thr_cm=1.0,
    hampel_window=11,
    hampel_min_periods=6,
    hampel_thr_cm=3.0,
    lowscore_thr_cm=1.5,
    lowscore_chord_frac=0.25,        # short runs
    lowscore_chord_frac_long=0.75,   # long runs (an L-turn deviates 0.5 x chord)
    lowscore_long_run=15,            # frames; 0.25 s
    lowscore_path_frac=0.9,          # long run removed when path < 0.9 x chord - thr
    max_gap_frames=120,
    smooth_window=5,
)

# flag components
KEPT, BRIDGED, LEFT_NAN = 0, 1, 2
R_NONE, R_RAW_NAN, R_LOW_POINT, R_LOW_INSTANCE, R_VELOCITY, R_SPIKE, R_HAMPEL, R_LOWSCORE_OFF = 0, 1, 2, 3, 4, 5, 6, 7
REASON_NAMES = {R_NONE: "ok", R_RAW_NAN: "raw_nan", R_LOW_POINT: "low_point_kept",
                R_LOW_INSTANCE: "low_instance_kept", R_VELOCITY: "r1_velocity", R_SPIKE: "r2_spike",
                R_HAMPEL: "r3_hampel", R_LOWSCORE_OFF: "lowscore_off_trajectory"}


def outcome(flags: np.ndarray) -> np.ndarray:
    """0 kept, 1 bridged, 2 left NaN."""
    return np.asarray(flags) // 10


def reason(flags: np.ndarray) -> np.ndarray:
    return np.asarray(flags) % 10


# ---------------------------------------------------------------------------
# h5 scores
# ---------------------------------------------------------------------------

def load_scores(h5_path: Path) -> dict:
    """{'point_scores': (n_nodes, T), 'instance_scores': (T,)} from a single-instance SLEAP h5.

    Read-only. Raises KeyError naming the file if either dataset is missing (older exports).
    """
    with h5py.File(h5_path, "r") as f:
        for key in ("point_scores", "instance_scores"):
            if key not in f:
                raise KeyError(f"{Path(h5_path).name}: no '{key}' dataset; re-export the SLEAP analysis h5")
        ps = f["point_scores"][:]
        isc = f["instance_scores"][:]
    if ps.shape[0] != 1 or isc.shape[0] != 1:
        raise ValueError(f"{Path(h5_path).name}: {ps.shape[0]} instances in scores; expected 1")
    return dict(point_scores=ps[0].astype(float), instance_scores=isc[0].astype(float))


# ---------------------------------------------------------------------------
# run helpers
# ---------------------------------------------------------------------------

def gap_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of True runs in a boolean vector; run k is mask[starts[k]:ends[k]]."""
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def describe_gaps(mask: np.ndarray, long_frames: int = 120) -> dict:
    """Counts of NaN runs: n, longest, > 15 / > 60 / > long_frames, frames inside runs > long."""
    s, e = gap_runs(mask)
    L = e - s
    return dict(n_runs=int(len(L)), longest=int(L.max()) if len(L) else 0,
                n_gt15=int((L > 15).sum()), n_gt60=int((L > 60).sum()), n_gt_long=int((L > long_frames).sum()),
                frames_in_long=int(L[L > long_frames].sum()) if len(L) else 0)


# ---------------------------------------------------------------------------
# the rules (each returns a boolean 'remove' mask over frames, for one bodypart)
# ---------------------------------------------------------------------------

def velocity_gate(xy: np.ndarray, v_max_cm_s: float, fps: float) -> np.ndarray:
    """R1: frame i is removed when its displacement from the previous VALID frame j exceeds
    v_max * (i - j) / fps. Sequential: a removed frame never becomes the reference."""
    x, y = xy[:, 0], xy[:, 1]
    valid = ~(np.isnan(x) | np.isnan(y))
    bad = np.zeros(len(x), bool)
    last = -1
    step = v_max_cm_s / fps
    for i in np.flatnonzero(valid):
        if last >= 0:
            if np.hypot(x[i] - x[last], y[i] - y[last]) > step * (i - last):
                bad[i] = True
                continue
        last = i
    return bad


def spike_gate(xy: np.ndarray, spike_thr_cm: float) -> np.ndarray:
    """R2: out-and-back single-frame spike -- d_in and d_out both > thr while the two-frame net
    move is < half the smaller of them. NaN neighbours never qualify."""
    p_prev, p, p_next = xy[:-2], xy[1:-1], xy[2:]
    d_in = np.hypot(*(p - p_prev).T)
    d_out = np.hypot(*(p_next - p).T)
    net = np.hypot(*(p_next - p_prev).T)
    with np.errstate(invalid="ignore"):
        mid = (d_in > spike_thr_cm) & (d_out > spike_thr_cm) & (net < 0.5 * np.minimum(d_in, d_out))
    bad = np.zeros(len(xy), bool)
    bad[1:-1] = np.nan_to_num(mid.astype(float)).astype(bool)
    return bad


def hampel_residual(xy: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """|p_i - median of the centred window| in cm, NaN-aware (NaN where too few neighbours)."""
    med = pd.DataFrame(xy).rolling(window, center=True, min_periods=min_periods).median().to_numpy()
    return np.hypot(xy[:, 0] - med[:, 0], xy[:, 1] - med[:, 1])


def hampel_gate(xy: np.ndarray, window: int, min_periods: int, thr_cm: float) -> np.ndarray:
    """R3: removed where the Hampel residual exceeds thr_cm (NaN residual never removes)."""
    resid = hampel_residual(xy, window, min_periods)
    with np.errstate(invalid="ignore"):
        off = resid > thr_cm
    return np.nan_to_num(off.astype(float)).astype(bool)


def lowscore_deviation(xy: np.ndarray, low_score: np.ndarray) -> dict:
    """Per-frame arrays for every low-score frame with a valid frame flanking its run on both
    sides: `dev` distance (cm) from the chord between the flanks, `chord` its length, `run_len`
    frames in the run, `run_path` path length (cm) over the run's valid frames. NaN where not
    testable (good frame, NaN position, run touching the video start/end or a NaN run). Used by
    the gate and by the QC's 'low-score frames off trajectory' columns."""
    valid = ~np.isnan(xy).any(axis=1)
    testable = low_score | ~valid                    # a low-score run may contain NaN frames
    T = len(xy)
    out = {k: np.full(T, np.nan) for k in ("dev", "chord", "run_len", "run_path")}
    starts, ends = gap_runs(testable)
    for s, e in zip(starts, ends):
        if s == 0 or e == T or not (valid[s - 1] and valid[e]):
            continue
        a, b = xy[s - 1], xy[e]
        t = (np.arange(s, e) - (s - 1)) / (e - (s - 1))
        ref = a[None, :] + t[:, None] * (b - a)[None, :]
        sel = low_score[s:e] & valid[s:e]
        d = np.hypot(*(xy[s:e] - ref).T)
        seg = xy[s:e][valid[s:e]]
        path = float(np.hypot(*np.diff(seg, axis=0).T).sum()) if len(seg) > 1 else 0.0
        idx = np.arange(s, e)[sel]
        out["dev"][idx] = d[sel]
        out["chord"][idx] = np.hypot(*(b - a))
        out["run_len"][idx] = e - s
        out["run_path"][idx] = path
    return out


def lowscore_gate(xy: np.ndarray, low_score: np.ndarray, thr_cm: float, chord_frac: float,
                  chord_frac_long: float, long_run: int, path_frac: float) -> np.ndarray:
    """The conditional confidence gate (module docstring): off-chord frames of any run, plus
    every frame of a long run whose path falls short of its chord. Untestable runs are kept."""
    d = lowscore_deviation(xy, low_score)
    with np.errstate(invalid="ignore"):
        is_long = d["run_len"] > long_run
        tol = thr_cm + np.where(is_long, chord_frac_long, chord_frac) * d["chord"]
        off = d["dev"] > tol
        static = is_long & (d["run_path"] < path_frac * d["chord"] - thr_cm)
    return np.nan_to_num((off | static).astype(float)).astype(bool)


# ---------------------------------------------------------------------------
# bridging and smoothing
# ---------------------------------------------------------------------------

def interpolate_limited(xy: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straight-line bridge of interior NaN runs <= max_gap_frames. Returns (xy, bridged, left_nan).

    Uses np.interp per run so a longer run is left whole (pandas' `limit=` would fill its first
    N frames). Leading and trailing runs stay NaN (v1 filled them with limit_direction='both')."""
    out = xy.copy()
    nan = np.isnan(out).any(axis=1)
    bridged = np.zeros(len(out), bool)
    left = np.zeros(len(out), bool)
    starts, ends = gap_runs(nan)
    for s, e in zip(starts, ends):
        if s == 0 or e == len(out) or (e - s) > max_gap_frames:
            left[s:e] = True
            continue
        t = np.arange(s, e)
        for c in (0, 1):
            out[s:e, c] = np.interp(t, [s - 1, e], [out[s - 1, c], out[e, c]])
        bridged[s:e] = True
    return out, bridged, left


def smooth_remask(xy: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean with min_periods=1, then the input's NaN mask re-imposed (pandas
    would otherwise shrink every gap by (window-1)/2 frames a side). The frames beside a gap
    are averaged over the available neighbours only, as in v1."""
    nan = np.isnan(xy).any(axis=1)
    # .copy(): under pandas copy-on-write (histology env) to_numpy() hands back a read-only view
    sm = pd.DataFrame(xy).rolling(window, center=True, min_periods=1).mean().to_numpy().copy()
    sm[nan] = np.nan
    return sm


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def clean_v2(raw_px: pd.DataFrame, nodes: list[str], scores: dict,
             to_cm: Callable[[np.ndarray], np.ndarray], cfg: dict = V2) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cleaning v2 of one session.

    raw_px: DataFrame with `{bp}.x`, `{bp}.y` columns in pixels (one row per frame); scores from
    `load_scores`; `to_cm(xy_px (T,2)) -> (T,2) cm`, NaN-propagating (maze affine or arena
    calibration). Returns (cleaned_cm DataFrame with the same columns, flags DataFrame int8 with
    one column per bodypart, report dict with per-bodypart counts and the constants).
    """
    T = len(raw_px)
    ps, isc = scores["point_scores"], scores["instance_scores"]
    if ps.shape != (len(nodes), T) or isc.shape != (T,):
        raise ValueError(f"scores shape {ps.shape}/{isc.shape} do not match {len(nodes)} nodes x {T} frames")
    low_inst = isc < cfg["instance_min"]

    cleaned = {}
    flags = {}
    report = dict(cfg)
    report["n_frames"] = T
    for k, bp in enumerate(nodes):
        xy = to_cm(raw_px[[f"{bp}.x", f"{bp}.y"]].to_numpy(float))
        raw_nan = np.isnan(xy).any(axis=1)
        low_point = (ps[k] < cfg["point_min"]) & ~raw_nan
        low = (low_point | low_inst) & ~raw_nan

        why = np.zeros(T, np.int8)
        why[raw_nan] = R_RAW_NAN

        # The gate goes first: a rejected low-score run must never serve as R1's reference,
        # otherwise R1 re-accepts the run after D / v_max frames and then eats the true frames
        # on the way back (synthetic control 7, 2026-09-23).
        low_off = lowscore_gate(xy, low, cfg["lowscore_thr_cm"], cfg["lowscore_chord_frac"],
                                cfg["lowscore_chord_frac_long"], cfg["lowscore_long_run"], cfg["lowscore_path_frac"])
        why[low_off & (why == 0)] = R_LOWSCORE_OFF
        xy0 = xy.copy()
        xy0[low_off] = np.nan

        r1 = velocity_gate(xy0, cfg["v_max_cm_s"], cfg["fps"])
        why[r1 & (why == 0)] = R_VELOCITY
        xy1 = xy0.copy()
        xy1[r1] = np.nan

        r2 = spike_gate(xy1, cfg["spike_thr_cm"])
        why[r2 & (why == 0)] = R_SPIKE
        xy2 = xy1.copy()
        xy2[r2] = np.nan

        r3 = hampel_gate(xy2, cfg["hampel_window"], cfg["hampel_min_periods"], cfg["hampel_thr_cm"])
        why[r3 & (why == 0)] = R_HAMPEL
        xy3 = xy2.copy()
        xy3[r3] = np.nan

        xy4, bridged, left = interpolate_limited(xy3, cfg["max_gap_frames"])
        xy5 = smooth_remask(xy4, cfg["smooth_window"])

        flag = np.zeros(T, np.int8)
        kept = why == 0
        flag[kept & low_point] = R_LOW_POINT
        flag[kept & low_inst] = R_LOW_INSTANCE          # instance wins over point when both
        flag[bridged] = 10 + why[bridged]
        flag[left] = 20 + why[left]

        cleaned[f"{bp}.x"], cleaned[f"{bp}.y"] = xy5[:, 0], xy5[:, 1]
        flags[bp] = flag
        gaps = describe_gaps(left, cfg["max_gap_frames"])
        report[bp] = dict(raw_nan=int(raw_nan.sum()), low_point=int(low_point.sum()),
                          low_instance=int(low_inst.sum()), low_kept=int((kept & low).sum()),
                          r1_velocity=int((why == R_VELOCITY).sum()), r2_spike=int((why == R_SPIKE).sum()),
                          r3_hampel=int((why == R_HAMPEL).sum()), lowscore_off=int((why == R_LOWSCORE_OFF).sum()),
                          removed=int((why > 0).sum()), removed_by_rules=int(((why > 0) & ~raw_nan).sum()),
                          bridged=int(bridged.sum()), left_nan=int(left.sum()),
                          n_bridged_runs=int(len(gap_runs(bridged)[0])), longest_left_nan=gaps["longest"],
                          n_left_runs=gaps["n_runs"])
    return pd.DataFrame(cleaned, index=raw_px.index), pd.DataFrame(flags, index=raw_px.index, dtype=np.int8), report


def clean_v1(raw_px: pd.DataFrame, nodes: list[str], to_cm_df: Callable[[pd.DataFrame], pd.DataFrame],
             multiplier: float = 3.0) -> pd.DataFrame:
    """The v1 numerics, unchanged: sleap_to_rois.drop_jumps -> interpolate_smooth -> cm.
    `to_cm_df` maps a whole DataFrame (the callers' affine / arena step). Imported lazily to
    avoid a circular import (sleap_to_rois imports this module for v2)."""
    from sleap_to_rois import drop_jumps, interpolate_smooth
    return to_cm_df(interpolate_smooth(drop_jumps(raw_px, nodes, multiplier=multiplier), nodes))


def report_row(report: dict, bp: str = "head_back") -> dict:
    """Flat QC-row view of one bodypart's counts plus the constants (for the stage QC CSVs)."""
    r = {f"v2_{k}": v for k, v in report[bp].items()}
    r.update({k: report[k] for k in ("clean_version", "point_min", "instance_min", "v_max_cm_s", "spike_thr_cm",
                                     "hampel_window", "hampel_thr_cm", "lowscore_thr_cm", "lowscore_chord_frac",
                                     "lowscore_chord_frac_long", "lowscore_long_run", "lowscore_path_frac",
                                     "max_gap_frames", "smooth_window")})
    return r
