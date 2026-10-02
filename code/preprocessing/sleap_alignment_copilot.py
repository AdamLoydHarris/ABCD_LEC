"""
SLEAP Alignment Preprocessing Script

This script aligns SLEAP tracking data with task states from pycontrol files.
It finds the frame where the first A_on state occurs (start of first trial),
removes everything before it, and resamples from 60Hz to 40Hz.

Workflow:
1. Find ephys metadata files (iterate through ~5 per mouse, 5 mice)
2. Use metadata to determine if session is 'awake' (task) or 'sleep'
3. For awake sessions, locate pinstate and pycontrol files
4. Handle case where pinstate has 0 up states (try another mouse's file)
5. First up state in pinstate = frame where first pycontrol rsync occurs
6. Find first A_on event timestamp, convert to video frame, cut SLEAP data
7. Resample from 60Hz to 40Hz and save
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta
import warnings

# `--dataset` must reach recday_registry before it is imported (default: lec, unchanged).
if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recday_registry as rr                                          # noqa: E402

# Base paths, from the registry so `--dataset mec` moves every one of them together
BASE_PATH = rr.REPO_ROOT
DATA_PATH = BASE_PATH / "data"
METADATA_PATH = DATA_PATH / "metadata"
RAW_PATH = DATA_PATH / "raw"
PREPROCESSED_PATH = DATA_PATH / "preprocessed"
PROCESSED_PATH = DATA_PATH / "processed"

# Input paths
SLEAP_PATH = PREPROCESSED_PATH / "sleap" / "maze"
SLEAP_PATH_OPENFIELD = PREPROCESSED_PATH / "sleap" / "openfield"   # openfield_to_xy.py output
EPHYS_PATH = PREPROCESSED_PATH / "ephys"

#: Which cleaned-track tree is read (`--tracking-clean`). 'v1' is the tree every existing
#: data_dic was built from. 'geom2' = `sleap_to_rois.py --geometry v2` (cyclic maze-2 corners;
#: the Z-order set collapsed bp02/bp04 x to a 1.3 cm band, 2026-09-23), arenas unchanged.
#: 'v2' = cleaning v2 (`tracking_clean.py`) for maze and arenas. Anything but 'v1' needs
#: --out-tag so data/processed/ is never rewritten.
TRACKING_CLEAN = "v1"
TRACKING_ROOTS = {"v1": ("maze", "openfield"),
                  "geom2": ("maze_geom2", "openfield"),
                  "v2": ("maze_v2", "openfield_v2")}
#: v2 CSVs carry NaN where the track is untracked for > 2 s (and at the video's ends). With this
#: on, a 25 ms bin whose bracketing frames are NaN is written NaN in XY_raw instead of being
#: bridged by `sample_at_frames`' interpolant (which fits valid frames only and extrapolates).
#: HD ('nearest') carries the NaN by itself; Locs come as 'None' -> 0 from the v2 ROI file.
PROPAGATE_TRACK_NAN = False


def set_tracking_clean(mode: str) -> None:
    """Rebind the two SLEAP roots to the tree of `mode` (see TRACKING_ROOTS); v2 propagates NaN."""
    global TRACKING_CLEAN, SLEAP_PATH, SLEAP_PATH_OPENFIELD, PROPAGATE_TRACK_NAN
    maze, arena = TRACKING_ROOTS[mode]
    TRACKING_CLEAN = mode
    SLEAP_PATH = PREPROCESSED_PATH / "sleap" / maze
    SLEAP_PATH_OPENFIELD = PREPROCESSED_PATH / "sleap" / arena
    PROPAGATE_TRACK_NAN = mode == "v2"
    if mode != "v1" and not SLEAP_PATH.exists():
        raise FileNotFoundError(f"--tracking-clean {mode}: {SLEAP_PATH} does not exist; run the stage script first")


def untracked_bins(data: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """True for every bin whose lower or upper bracketing frame is NaN in `data` (T, k)."""
    nan = np.isnan(data).any(axis=1) if data.ndim > 1 else np.isnan(data)
    lo = np.clip(np.floor(frames).astype(int), 0, len(nan) - 1)
    hi = np.clip(np.ceil(frames).astype(int), 0, len(nan) - 1)
    return nan[lo] | nan[hi]

#: Which sheet sessions to align: 'task' (ABCD-type, the LEC default), 'openfield' (object and
#: open-field sessions, anchored at pyControl rsync[0] like extract_neuron_raw's no-trials
#: branch; needs SYNC_METHOD 'matched') or 'all'.
SESSIONS = "task"


def set_output_tag(tag: str) -> None:
    """Redirect PROCESSED_PATH to data/processed_<tag> so a re-derivation never touches the
    arrays an existing data_dic was built from."""
    global PROCESSED_PATH
    PROCESSED_PATH = DATA_PATH / (f"processed_{tag}" if tag else "processed")


def session_kind(metadata_row: pd.Series) -> str:
    """'ABCD' | 'ABCDE' | 'object' | 'openfield' | raw string, from the sheet row."""
    raw = str(metadata_row.get("Structure_abstract", "")).strip()
    return rr._SESSION_TYPES.get(raw.lower(), raw)
PINSTATE_PATH = RAW_PATH / "pinstate"
PYCONTROL_PATH = RAW_PATH / "pycontrol"

# Constants
VIDEO_FPS = 60  # Original SLEAP video frame rate
TARGET_FPS = 40  # Target frame rate after resampling
PYCONTROL_SAMPLE_RATE = 1000  # Hz (pycontrol samples at 1000 Hz)
# Default pinstate values (will be auto-detected per file)
PINSTATE_HIGH_DEFAULT = 805306368  # Sync pulse ON value
PINSTATE_LOW_DEFAULT = 536870912  # Sync pulse OFF value

# List of mice to process
MICE = list(rr.EPHYS_MICE)

#: How video frames are put on the pyControl clock.
#:   'first_edge' (LEC, default): first pinstate rising edge = first pyControl rsync, then
#:                nominal VIDEO_FPS. Reproduces every LEC array on disk.
#:   'matched'   : fit frame = a * t_ms + b to the WHOLE pinstate train against the pyControl
#:                rsync list (`fit_frame_clock`), then sample each 25 ms bin at its fitted
#:                frame. Needed for mEC: the maze-2 camera runs at 59.948 fps (measured on
#:                13 sessions of bp02/bp04), which under 60 fps drifts ~1 s over a 20 min
#:                session; the maze-1 camera runs at 59.995 fps. The pinstate line also
#:                carries the paired box's pulses, so the first edge can be the partner's.
SYNC_METHOD = "first_edge"
SYNC_QC: List[Dict] = []          # one row per matched session, written by main_metadata
MAX_JUMP_FRAMES = 600             # 10 s of dropped frames is the most a fit is allowed to bridge


def rising_edges(pinstate: np.ndarray) -> np.ndarray:
    """Frame indices of pinstate rising edges (low -> high)."""
    high, _ = detect_pinstate_levels(pinstate)
    if high is None:
        return np.array([], dtype=int)
    is_high = pinstate == high
    return np.where(is_high[1:] & ~is_high[:-1])[0] + 1


def all_rsync_timestamps(pycontrol_data: Dict) -> np.ndarray:
    """Every rsync event time (ms) in a parsed pyControl file, in order."""
    code = pycontrol_data['events'].get('rsync')
    if code is None:
        return np.array([], dtype=float)
    return np.array([t for t, c in pycontrol_data['data'] if c == code], dtype=float)


def fit_frame_clock_globalline(rising_frames: np.ndarray, rsync_ms: np.ndarray,
                    fps_range: Tuple[float, float] = (58.5, 61.5), tol_frames: float = 1.5,
                    max_passes: int = 30) -> Dict:
    """SUPERSEDED 2026-09-18 by `frame_clock.fit_frame_clock` (kept for the survey): this
    global-line fitter lost 30-70 % of the pulses in sessions with bursts of dropped frames
    (ah08/ly07 2025-06-16 s3, bp04 2024-03-25 s4, 2024-03-27 s0) and drifted rate (ah10
    2025-06-13 s3), producing clocks worse than a plain 60 fps line.

    Piecewise-linear frame clock from pinstate edges and this mouse's pyControl rsyncs.

    The pinstate may hold two boxes' trains, so this is a search rather than interval
    matching: (1) the frame rate is the one at which the histogram of all edge-minus-rsync
    differences has the tallest 3-bin peak (a wrong rate smears the peak; the partner's
    pulses only add a flat floor); (2) at that rate the peak gives the offset and every
    rsync is paired to its nearest edge within `tol_frames`; (3) the prediction is then
    re-drawn as the piecewise-linear interpolation through the matched (t_ms, frame) anchors
    and the pairing repeated, and whenever no new pulses pair, the still-unmatched pulses
    are searched for a common shift -- a dropped-frame segment -- and shifted before the
    next pass. Videos do drop frames: bp04 2024-03-24 s4 loses 16 frames at 14.6 min and
    a single line pairs only 95/225 of its rsyncs; anchored interpolation follows it.

    Returns the anchors plus summary numbers (fps from the global line, residual of the
    anchors about that line, number and size of jumps). Use `frames_at(clock, t_ms)` to
    map pyControl times to frames. Raises ValueError when fewer than half the rsyncs pair.
    """
    up = np.asarray(rising_frames, dtype=float)
    rs = np.asarray(rsync_ms, dtype=float)
    if len(up) < 5 or len(rs) < 5:
        raise ValueError(f"too few pulses (edges {len(up)}, rsyncs {len(rs)})")

    def peak(fps: float, bin_ms: float = 20.0):
        d = (up[None, :] / fps * 1000.0 - rs[:, None]).ravel()
        edges = np.arange(d.min(), d.max() + bin_ms, bin_ms)
        h, _ = np.histogram(d, edges)
        k = int(h.argmax())
        return int(h[max(k - 1, 0):k + 2].sum()), (edges[k] + edges[k + 1]) / 2

    grid = np.arange(fps_range[0], fps_range[1], 0.005)
    coarse = grid[int(np.argmax([peak(f)[0] for f in grid]))]
    fine = np.arange(coarse - 0.01, coarse + 0.01, 0.0002)
    fps0 = fine[int(np.argmax([peak(f)[0] for f in fine]))]
    _, offset_ms = peak(fps0)
    slope0 = fps0 / 1000.0                                  # frames per ms

    pred = (rs + offset_ms) * slope0
    matched = np.zeros(len(rs), dtype=bool)
    frame_of = np.full(len(rs), np.nan)
    used = np.zeros(len(up), dtype=bool)

    def nearest_free(pred_sub):
        free_idx = np.where(~used)[0]
        if len(free_idx) == 0:
            return None, None
        free = up[free_idx]
        idx = np.clip(np.searchsorted(free, pred_sub), 1, len(free) - 1) if len(free) > 1 else np.zeros(len(pred_sub), int)
        left = np.maximum(idx - 1, 0)
        cand = np.where(np.abs(free[left] - pred_sub) < np.abs(free[idx] - pred_sub), left, idx)
        return free_idx[cand], np.abs(free[cand] - pred_sub)

    for _ in range(max_passes):
        um = np.where(~matched)[0]
        if len(um) == 0:
            break
        cand, dist = nearest_free(pred[um])
        if cand is None:
            break
        ok = dist <= tol_frames
        # one edge may only anchor one rsync: keep the closer claimant
        order = np.argsort(dist)
        taken = set()
        for j in order:
            if not ok[j] or cand[j] in taken:
                ok[j] = False
                continue
            taken.add(cand[j])
        if ok.any():
            matched[um[ok]] = True
            frame_of[um[ok]] = up[cand[ok]]
            used[cand[ok]] = True
            m_t, m_f = rs[matched], frame_of[matched]
            pred = np.interp(rs, m_t, m_f)
            pred[rs < m_t[0]] = m_f[0] + (rs[rs < m_t[0]] - m_t[0]) * slope0
            pred[rs > m_t[-1]] = m_f[-1] + (rs[rs > m_t[-1]] - m_t[-1]) * slope0
            continue
        # nothing new within tolerance: look for a common shift among the unmatched pulses.
        # After a dropped-frame segment the shift is one value plus the segment's own drift
        # (bp04 s4: -15 frames drifting to -23), so judge the 3-bin neighbourhood of 2-frame
        # bins and accept a fifth of the unmatched; each pass then anchors the central part
        # and the interpolation carries the rest.
        if len(um) < 5:
            break
        free = up[~used]
        if len(free) == 0:
            break
        d = (free[None, :] - pred[um][:, None]).ravel()
        edges = np.arange(d.min() - 1.0, d.max() + 3.0, 2.0)
        h, _ = np.histogram(d, edges)
        k = int(h.argmax())
        lo, hi = max(k - 1, 0), min(k + 2, len(h))
        if h[lo:hi].sum() < max(5, 0.2 * len(um)):
            break
        centres = (edges[lo:hi] + edges[lo + 1:hi + 1]) / 2
        pred[um] += float(np.average(centres, weights=h[lo:hi]))

    if matched.sum() < max(5, len(rs) // 2):
        raise ValueError(f"only {int(matched.sum())}/{len(rs)} rsyncs pair at {fps0:.3f} fps")
    m_t, m_f = rs[matched], frame_of[matched]
    keep = np.concatenate([[True], np.diff(m_f) > 0])       # anchors must be strictly increasing
    m_t, m_f = m_t[keep], m_f[keep]
    a, b = np.polyfit(m_t, m_f, 1)
    resid = m_f - (a * m_t + b)
    steps = np.diff(resid)
    jumps = steps[np.abs(steps) > 2.0]
    # A truncated pinstate (ah08 2025-06-20/23 s3: edges stop mid-session) lets the search pair
    # the remaining pulses to whatever edges exist and fit 55 fps with a 50k-frame jump. Refuse
    # anything a real camera and a few dropped frames cannot produce.
    fps = a * 1000.0
    if not 58.5 <= fps <= 61.5 or (len(jumps) and np.abs(jumps).max() > MAX_JUMP_FRAMES):
        raise ValueError(f"implausible frame clock: {fps:.2f} fps, largest jump "
                         f"{np.abs(jumps).max() if len(jumps) else 0:.0f} frames (pinstate truncated?)")
    return dict(anchors_ms=m_t, anchors_frames=m_f, a=float(a), b=float(b), fps=float(a * 1000.0),
                n_rsync=int(len(rs)), n_matched=int(len(m_t)), n_edges=int(len(up)),
                n_edges_unmatched=int(len(up) - len(m_t)),
                resid_rms_frames=float(np.sqrt(np.mean(resid ** 2))),
                n_jumps=int(len(jumps)), max_jump_frames=float(np.abs(jumps).max()) if len(jumps) else 0.0,
                first_rsync_frame=float(m_f[0] if matched[0] else a * rs[0] + b))


def frames_at(clock: Dict, t_ms) -> np.ndarray:
    """pyControl time (ms) -> fractional video frame through the fitted anchors, with
    linear extrapolation at the global slope beyond the first/last anchor."""
    t = np.asarray(t_ms, dtype=float)
    m_t, m_f, a = clock['anchors_ms'], clock['anchors_frames'], clock['a']
    f = np.interp(t, m_t, m_f)
    lo, hi = t < m_t[0], t > m_t[-1]
    f[lo] = m_f[0] + (t[lo] - m_t[0]) * a
    f[hi] = m_f[-1] + (t[hi] - m_t[-1]) * a
    return f


def ms_at_frame(clock: Dict, frame: float) -> float:
    """Inverse of `frames_at` for one frame index."""
    m_t, m_f, a = clock['anchors_ms'], clock['anchors_frames'], clock['a']
    if frame <= m_f[0]:
        return float(m_t[0] + (frame - m_f[0]) / a)
    if frame >= m_f[-1]:
        return float(m_t[-1] + (frame - m_f[-1]) / a)
    return float(np.interp(frame, m_f, m_t))


# The frame clock in use: the monotone-alignment fitter (frame_clock.py, 2026-09-18). It
# returns the same anchors/a/b dict, so frames_at / ms_at_frame above apply unchanged.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frame_clock as _frame_clock                                     # noqa: E402
fit_frame_clock = _frame_clock.fit_frame_clock
MIN_MATCHED_FRACTION = 0.9        # a session whose clock anchors fewer pulses is not written


def sample_at_frames(data: np.ndarray, frames: np.ndarray, kind: str) -> np.ndarray:
    """Sample per-frame data at fractional frame positions.

    kind 'linear': NaN-aware linear interpolation per column (as `resample_data` does);
    kind 'nearest': the nearest frame's value (labels, angles).
    """
    from scipy import interpolate
    if kind == 'nearest':
        return data[np.clip(np.rint(frames).astype(int), 0, len(data) - 1)]
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    out = np.full((len(frames), data.shape[1]), np.nan)
    old = np.arange(len(data))
    for j in range(data.shape[1]):
        valid = ~np.isnan(data[:, j])
        if valid.sum() >= 2:
            f = interpolate.interp1d(old[valid], data[valid, j], kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
            out[:, j] = f(frames)
    return out


def load_metadata(mouse_id: str) -> pd.DataFrame:
    """Load the metadata file for a given mouse."""
    metadata_file = METADATA_PATH / f"MetaData-{mouse_id}.csv"
    df = pd.read_csv(metadata_file)
    return df


def get_ephys_folders(mouse_id: str) -> List[Path]:
    """Get all ephys preprocessed folders for a mouse."""
    ephys_mouse_path = EPHYS_PATH / mouse_id
    if not ephys_mouse_path.exists():
        return []
    folders = [f for f in ephys_mouse_path.iterdir() if f.is_dir() and "_preprocessed" in f.name]
    return sorted(folders)


def load_ephys_metadata(ephys_folder: Path) -> pd.DataFrame:
    """Load the recording_sessions_in_concat.csv from an ephys folder."""
    csv_path = ephys_folder / "recording_sessions_in_concat.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Ephys metadata not found: {csv_path}")
    return pd.read_csv(csv_path)


def parse_ephys_folder_dates(folder_name: str) -> Tuple[str, str]:
    """Extract date range from an ephys folder name like '2025-06-13_2025-06-15_preprocessed'.

    A single-day block ('2024-03-21_preprocessed', mEC only) returns the same date twice, so
    the date-range filter below still works; `create_output_filename` collapses the pair
    back to one date in the recday name.
    """
    dates = parse_ephys_folder_all_dates(folder_name)
    if dates:
        return dates[0], dates[-1]
    return None, None


def parse_ephys_folder_all_dates(folder_name: str) -> List[str]:
    """Every date of a block folder: '2025-07-14_..._2025-07-19_preprocessed' -> six dates, the
    July ABCDE blocks (2026-09-28). Anchored at the end, so a renamed copy such as
    '*_preprocessed.failed_*' is not a block. Names come out through BLOCK_DATES below."""
    m = re.match(r"^((?:\d{4}-\d{2}-\d{2})(?:_\d{4}-\d{2}-\d{2})*)_preprocessed$", folder_name)
    return m.group(1).split("_") if m else []


#: Dates of the block being processed (set per ephys folder). A block of more than two dates is
#: named with all of them, `ah10_20250714_20250715_20250716_20250717_20250718_20250719`, the way
#: recday_registry names it and build_data_dic / validate_alignment look it up; one- and
#: two-date blocks keep the names every existing array has.
BLOCK_DATES: List[str] = []


def is_awake_session(metadata_row: pd.Series) -> bool:
    """Determine if a session is awake (task) vs sleep based on Maze column."""
    maze_value = str(metadata_row.get("Maze", ""))
    return maze_value.lower() != "sleepbox" and maze_value != "-"


def format_tracking_time(tracking_val) -> str:
    """Convert tracking time like 121535 to datetime string like '2025-06-13-121535'."""
    if pd.isna(tracking_val) or tracking_val == "-":
        return None
    tracking_str = str(int(tracking_val)).zfill(6)
    return tracking_str


def format_date_for_filename(date_str: str) -> str:
    """Convert date like '2025-6-13' to '2025-06-13'."""
    parts = date_str.split("-")
    if len(parts) == 3:
        year, month, day = parts
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return date_str


def find_pinstate_file(mouse_id: str, datetime_str: str) -> Optional[Path]:
    """
    Find pinstate file for given mouse and datetime.
    datetime_str should be like '2025-06-13-170056'
    """
    pattern = f"{mouse_id}_pinstate_{datetime_str}.csv"
    pinstate_file = PINSTATE_PATH / pattern
    if pinstate_file.exists():
        return pinstate_file
    return None


def get_available_pinstate_files(mouse_id: str) -> List[Tuple[datetime, Path]]:
    """
    Get all available pinstate files for a mouse.
    
    Returns list of (datetime, path) tuples sorted by datetime.
    """
    available = []
    for pinstate_file in PINSTATE_PATH.glob(f"{mouse_id}_pinstate_*.csv"):
        # Parse datetime from filename: ah08_pinstate_2025-06-24-114245.csv
        name = pinstate_file.stem
        parts = name.split('_pinstate_')
        if len(parts) == 2:
            datetime_str = parts[1]  # 2025-06-24-114245
            try:
                dt = datetime.strptime(datetime_str, "%Y-%m-%d-%H%M%S")
                available.append((dt, pinstate_file))
            except ValueError:
                continue
    
    return sorted(available, key=lambda x: x[0])


def find_closest_pinstate_file(mouse_id: str, target_dt: datetime, 
                                max_diff_minutes: int = 2) -> Optional[Tuple[Path, str]]:
    """
    Find the closest pinstate file to the target datetime.
    
    Args:
        mouse_id: Mouse identifier
        target_dt: Target datetime
        max_diff_minutes: Maximum allowed difference in minutes
    
    Returns:
        Tuple of (path, matched_datetime_str) or None if no match
    """
    available = get_available_pinstate_files(mouse_id)
    if not available:
        return None
    
    best_match = None
    best_dt_str = None
    best_diff = timedelta(days=999)
    
    for dt, path in available:
        diff = abs(target_dt - dt)
        if diff < best_diff:
            best_diff = diff
            best_match = path
            best_dt_str = dt.strftime("%Y-%m-%d-%H%M%S")
    
    if best_diff <= timedelta(minutes=max_diff_minutes):
        return best_match, best_dt_str
    
    return None


def find_pinstate_file_fallback(datetime_str: str, exclude_mouse: str = None) -> Optional[Tuple[Path, str]]:
    """
    Find pinstate file from any mouse with matching datetime (fallback when primary has no up states).
    Returns tuple of (file_path, mouse_id) or None.
    """
    for mouse_id in MICE:
        if mouse_id == exclude_mouse:
            continue
        pattern = f"{mouse_id}_pinstate_{datetime_str}.csv"
        pinstate_file = PINSTATE_PATH / pattern
        if pinstate_file.exists():
            return pinstate_file, mouse_id
    
    # Also try files with "-" prefix (seen in data)
    pattern = f"-_pinstate_{datetime_str}.csv"
    pinstate_file = PINSTATE_PATH / pattern
    if pinstate_file.exists():
        return pinstate_file, "-"
    
    return None


def find_pycontrol_file(mouse_id: str, date_str: str, behaviour_time: str) -> Optional[Path]:
    """
    Find pycontrol file for given mouse and datetime.
    date_str should be like '2025-06-13'
    behaviour_time should be like '170205'
    """
    datetime_str = f"{date_str}-{behaviour_time}"
    pattern = f"{mouse_id}-{datetime_str}.txt"
    pycontrol_file = PYCONTROL_PATH / pattern
    if pycontrol_file.exists():
        return pycontrol_file
    return None


def get_available_pycontrol_files(mouse_id: str) -> List[Tuple[datetime, Path]]:
    """
    Get all available pycontrol files for a mouse.
    
    Returns list of (datetime, path) tuples sorted by datetime.
    """
    available = []
    # .txt = pyControl 1 (ABCD task); .tsv = pyControl 2 (mEC open field, rsync only)
    for pycontrol_file in list(PYCONTROL_PATH.glob(f"{mouse_id}-*.txt")) + list(PYCONTROL_PATH.glob(f"{mouse_id}-*.tsv")):
        # Parse datetime from filename: ah08-2025-06-24-114301.txt
        name = pycontrol_file.stem
        parts = name.split('-', 1)
        if len(parts) == 2:
            datetime_str = parts[1]  # 2025-06-24-114301
            try:
                dt = datetime.strptime(datetime_str, "%Y-%m-%d-%H%M%S")
                available.append((dt, pycontrol_file))
            except ValueError:
                continue
    
    return sorted(available, key=lambda x: x[0])


def find_closest_pycontrol_file(mouse_id: str, target_dt: datetime, 
                                 max_diff_minutes: int = 2) -> Optional[Path]:
    """
    Find the closest pycontrol file to the target datetime.
    
    Args:
        mouse_id: Mouse identifier
        target_dt: Target datetime
        max_diff_minutes: Maximum allowed difference in minutes
    
    Returns:
        Path to the closest matching pycontrol file, or None if no match
    """
    available = get_available_pycontrol_files(mouse_id)
    if not available:
        return None
    
    best_match = None
    best_diff = timedelta(days=999)
    
    for dt, path in available:
        diff = abs(target_dt - dt)
        if diff < best_diff:
            best_diff = diff
            best_match = path
    
    if best_diff <= timedelta(minutes=max_diff_minutes):
        return best_match
    
    return None


def load_pinstate(pinstate_file: Path) -> np.ndarray:
    """Load pinstate file as numpy array."""
    df = pd.read_csv(pinstate_file, header=None)
    return df.values.flatten()


def detect_pinstate_levels(pinstate: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """
    Auto-detect the high and low values in a pinstate array.
    
    Assumes the pinstate contains two distinct values representing
    sync pulse ON (high) and OFF (low) states.
    
    Args:
        pinstate: 1D numpy array of pinstate values
    
    Returns:
        Tuple of (high_value, low_value) or (None, None) if detection fails
    """
    unique_values = np.unique(pinstate)
    
    if len(unique_values) == 0:
        return None, None
    
    if len(unique_values) == 1:
        # Only one value - can't determine high/low
        # Return None for high, the single value as low (no transitions)
        return None, unique_values[0]
    
    if len(unique_values) == 2:
        # Standard case: two values, higher one is the pulse ON
        low_val, high_val = sorted(unique_values)
        return high_val, low_val
    
    if len(unique_values) > 2:
        # More than 2 values - take the two most common
        # This handles edge cases with noise or intermediate values
        value_counts = {v: np.sum(pinstate == v) for v in unique_values}
        sorted_by_count = sorted(value_counts.keys(), key=lambda x: value_counts[x], reverse=True)
        top_two = sorted(sorted_by_count[:2])
        low_val, high_val = top_two
        return high_val, low_val
    
    return None, None


def find_first_up_state(pinstate: np.ndarray, high_value: Optional[int] = None) -> Optional[int]:
    """
    Find the frame index of the first 'up' (high) state in pinstate.
    
    Args:
        pinstate: 1D numpy array of pinstate values
        high_value: The value representing sync pulse ON. If None, auto-detect.
    
    Returns:
        Frame index of first high state, or None if not found
    """
    if high_value is None:
        high_value, _ = detect_pinstate_levels(pinstate)
    
    if high_value is None:
        return None
    
    high_indices = np.where(pinstate == high_value)[0]
    if len(high_indices) == 0:
        return None
    return high_indices[0]


def count_up_states(pinstate: np.ndarray, high_value: Optional[int] = None, low_value: Optional[int] = None) -> int:
    """
    Count the number of sync pulse transitions (rising edges).
    
    Args:
        pinstate: 1D numpy array of pinstate values
        high_value: The value representing sync pulse ON. If None, auto-detect.
        low_value: The value representing sync pulse OFF. If None, auto-detect.
    
    Returns:
        Number of rising edge transitions
    """
    if high_value is None or low_value is None:
        detected_high, detected_low = detect_pinstate_levels(pinstate)
        if high_value is None:
            high_value = detected_high
        if low_value is None:
            low_value = detected_low
    
    if high_value is None or low_value is None:
        return 0
    
    # Find rising edges (transitions from low to high)
    diff = np.diff(pinstate)
    # A rising edge occurs when going from low to high
    rising_edges = np.where(diff > 0)[0]
    return len(rising_edges)


def parse_pycontrol_file(pycontrol_file: Path) -> Dict:
    """
    Parse pycontrol file to extract state definitions and events.
    Returns dict with:
        - states: dict mapping state name to code
        - events: dict mapping event name to code
        - data: list of (timestamp, code) tuples for 'D' lines
    """
    states = {}
    events = {}
    data = []
    if Path(pycontrol_file).suffix == ".tsv":
        # pyControl 2 (mEC open field): events by name, seconds -> ms; presented in the same
        # {states, events, data} shape with synthetic codes so the rsync/A_on lookups work.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from extract_pokes import parse_pycontrol_v2
        times, _ = parse_pycontrol_v2(Path(pycontrol_file))
        events = {name: i + 1 for i, name in enumerate(sorted(times))}
        data = sorted((int(t), events[name]) for name, ts in times.items() for t in ts)
        return {'states': {}, 'events': events, 'data': data}
    
    with open(pycontrol_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('S '):
                # State definitions
                import json
                states = json.loads(line[2:])
            elif line.startswith('E '):
                # Event definitions
                import json
                events = json.loads(line[2:])
            elif line.startswith('D '):
                # Data line: D timestamp code
                parts = line[2:].split()
                if len(parts) >= 2:
                    timestamp = int(parts[0])
                    code = int(parts[1])
                    data.append((timestamp, code))
    
    return {'states': states, 'events': events, 'data': data}


def find_first_rsync_timestamp(pycontrol_data: Dict) -> Optional[int]:
    """Find the timestamp of the first rsync event in pycontrol data."""
    rsync_code = pycontrol_data['events'].get('rsync')
    if rsync_code is None:
        return None
    
    for timestamp, code in pycontrol_data['data']:
        if code == rsync_code:
            return timestamp
    return None


def find_first_A_on_timestamp(pycontrol_data: Dict) -> Optional[int]:
    """Find the timestamp of the first A_on state in pycontrol data."""
    a_on_code = pycontrol_data['states'].get('A_on')
    if a_on_code is None:
        return None
    
    for timestamp, code in pycontrol_data['data']:
        if code == a_on_code:
            return timestamp
    return None


def calculate_frame_from_pycontrol_time(
    pycontrol_timestamp: int,
    first_rsync_timestamp: int,
    first_rsync_frame: int,
    video_fps: int = VIDEO_FPS
) -> int:
    """
    Convert a pycontrol timestamp to a video frame number.
    
    Args:
        pycontrol_timestamp: The pycontrol event timestamp (in ms at 1000 Hz)
        first_rsync_timestamp: Timestamp of first rsync in pycontrol (ms)
        first_rsync_frame: Frame number of first rsync in video
        video_fps: Video frame rate (default 60 Hz)
    
    Returns:
        Frame number in video corresponding to the pycontrol timestamp
    """
    # Time difference in ms
    time_diff_ms = pycontrol_timestamp - first_rsync_timestamp
    # Convert to seconds
    time_diff_s = time_diff_ms / 1000.0
    # Convert to frames
    frame_diff = int(time_diff_s * video_fps)
    # Add to first rsync frame
    return first_rsync_frame + frame_diff


def get_available_sleap_datetimes(mouse_id: str, root: Optional[Path] = None) -> List[Tuple[datetime, str]]:
    """
    Get all available SLEAP tracking datetimes for a mouse.

    Returns list of (datetime, datetime_str) tuples sorted by datetime.
    """
    sleap_mouse_path = (root or SLEAP_PATH) / mouse_id
    if not sleap_mouse_path.exists():
        return []
    
    available = []
    for coords_file in sleap_mouse_path.glob(f"{mouse_id}_*.analysis_cleaned_coordinates.csv"):
        # Extract datetime string from filename: ah08_2025-06-24-114245.analysis_cleaned_coordinates.csv
        name = coords_file.stem  # ah08_2025-06-24-114245.analysis_cleaned_coordinates
        parts = name.split('_')
        if len(parts) >= 2:
            datetime_str = parts[1].replace('.analysis', '')
            try:
                dt = datetime.strptime(datetime_str, "%Y-%m-%d-%H%M%S")
                available.append((dt, datetime_str))
            except ValueError:
                continue
    
    return sorted(available, key=lambda x: x[0])


def find_closest_sleap_datetime(mouse_id: str, target_datetime_str: str,
                                 max_diff_minutes: int = 2, root: Optional[Path] = None) -> Optional[str]:
    """
    Find the closest SLEAP tracking datetime to the target datetime.
    
    Args:
        mouse_id: Mouse identifier
        target_datetime_str: Target datetime string like '2025-06-24-114301'
        max_diff_minutes: Maximum allowed difference in minutes
    
    Returns:
        The closest matching datetime string, or None if no match within threshold
    """
    try:
        target_dt = datetime.strptime(target_datetime_str, "%Y-%m-%d-%H%M%S")
    except ValueError:
        return None
    
    available = get_available_sleap_datetimes(mouse_id, root)
    if not available:
        return None

    best_match = None
    best_diff = timedelta(days=999)
    
    for dt, dt_str in available:
        diff = abs(target_dt - dt)
        if diff < best_diff:
            best_diff = diff
            best_match = dt_str
    
    if best_diff <= timedelta(minutes=max_diff_minutes):
        return best_match
    
    return None


def load_sleap_files(mouse_id: str, tracking_datetime: str, 
                     use_fuzzy_match: bool = True,
                     max_diff_minutes: int = 2,
                     verbose: bool = False,
                     root: Optional[Path] = None) -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
    """
    Load SLEAP coordinate, head direction, and ROI files.
    
    Args:
        mouse_id: Mouse identifier
        tracking_datetime: Expected datetime like '2025-06-13-170056'
        use_fuzzy_match: If True, search for closest match within time window
        max_diff_minutes: Maximum time difference for fuzzy matching
        verbose: Print debug info
    
    Returns:
        Tuple of (dict of DataFrames, actual_datetime_used)
    """
    sleap_mouse_path = (root or SLEAP_PATH) / mouse_id

    # First try exact match
    actual_datetime = tracking_datetime
    coords_file = sleap_mouse_path / f"{mouse_id}_{tracking_datetime}.analysis_cleaned_coordinates.csv"

    if not coords_file.exists() and use_fuzzy_match:
        # Try fuzzy match
        matched_datetime = find_closest_sleap_datetime(mouse_id, tracking_datetime, max_diff_minutes, root)
        if matched_datetime:
            actual_datetime = matched_datetime
            if verbose:
                print(f"    Fuzzy match: {tracking_datetime} → {matched_datetime}")
        else:
            if verbose:
                print(f"    No SLEAP files found within {max_diff_minutes} min of {tracking_datetime}")
            return {}, None
    
    files = {}
    
    # Coordinates file
    coords_file = sleap_mouse_path / f"{mouse_id}_{actual_datetime}.analysis_cleaned_coordinates.csv"
    if coords_file.exists():
        files['coordinates'] = pd.read_csv(coords_file)
    
    # Head direction file
    hd_file = sleap_mouse_path / f"{mouse_id}_{actual_datetime}.analysis_head_direction.csv"
    if hd_file.exists():
        files['head_direction'] = pd.read_csv(hd_file)
    
    # ROIs file
    roi_file = sleap_mouse_path / f"{mouse_id}_{actual_datetime}.analysis_ROIs.csv"
    if roi_file.exists():
        files['rois'] = pd.read_csv(roi_file)
    
    return files, actual_datetime if files else None


def resample_data(data: np.ndarray, original_fps: int, target_fps: int) -> np.ndarray:
    """
    Resample data from original_fps to target_fps using linear interpolation.
    
    This function handles NaN values properly by interpolating only valid data points.
    Note: scipy.signal.resample uses FFT which propagates NaN to entire output,
    so we use linear interpolation instead.
    
    Args:
        data: 1D or 2D numpy array (time on axis 0)
        original_fps: Original sampling rate
        target_fps: Target sampling rate
    
    Returns:
        Resampled array
    """
    from scipy import interpolate
    
    if len(data) == 0:
        return data
    
    original_samples = len(data)
    target_samples = int(original_samples * target_fps / original_fps)
    
    # Create old and new index arrays
    old_indices = np.arange(original_samples)
    new_indices = np.linspace(0, original_samples - 1, target_samples)
    
    if data.ndim == 1:
        # Handle 1D array
        valid_mask = ~np.isnan(data)
        if valid_mask.sum() < 2:
            # Not enough valid points to interpolate
            return np.full(target_samples, np.nan)
        f = interpolate.interp1d(
            old_indices[valid_mask], 
            data[valid_mask], 
            kind='linear', 
            bounds_error=False,
            fill_value='extrapolate'
        )
        resampled = f(new_indices)
    else:
        # Handle 2D array - interpolate each column
        resampled = np.zeros((target_samples, data.shape[1]))
        for col_idx in range(data.shape[1]):
            col = data[:, col_idx]
            valid_mask = ~np.isnan(col)
            if valid_mask.sum() < 2:
                # Not enough valid points to interpolate
                resampled[:, col_idx] = np.nan
            else:
                f = interpolate.interp1d(
                    old_indices[valid_mask],
                    col[valid_mask],
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                resampled[:, col_idx] = f(new_indices)
    
    return resampled


def _same_array(a: np.ndarray, b: np.ndarray) -> bool:
    """Equality that treats NaN as equal to NaN. XY/HD carry NaN (frame-clock mask, v2 track
    NaN) and Locs carry NaN objects (pandas reads the ROI files' 'None' as NaN), so plain
    np.array_equal called identical re-runs 'conflicts' (ly06 2026-09-28, 7 spurious files)."""
    if a.shape != b.shape:
        return False
    try:
        if np.array_equal(a, b, equal_nan=True):
            return True
    except TypeError:                                    # object dtype
        pass
    if a.dtype == object or b.dtype == object:
        return bool(np.array_equal(a.astype(str), b.astype(str)))
    return False


def save_new(path: Path, array: np.ndarray, verbose: bool = True) -> str:
    """Write unless an identical array is there already; a differing one is written beside.

    Nothing is ever overwritten (user's rule). Returns 'written', 'identical' or the
    conflict filename.
    """
    if path.exists():
        existing = np.load(path, allow_pickle=True)
        if _same_array(existing, array):
            return "identical"
        n = 1
        while (alt := path.with_name(f"{path.stem}.conflict{n}.npy")).exists():
            n += 1
        np.save(alt, array)
        if verbose:
            print(f"    CONFLICT: {path.name} differs from the file on disk; wrote {alt.name}")
        return alt.name
    np.save(path, array)
    return "written"


def create_output_filename(data_type: str, mouse_id: str, date1: str, date2: str, session_idx: int) -> str:
    """
    Create output filename following the naming convention.
    
    Args:
        data_type: 'XY', 'HD', or 'Locs'
        mouse_id: Mouse identifier (e.g., 'ah08')
        date1: First date string (e.g., '2025-06-13')
        date2: Second date string (e.g., '2025-06-15')
        session_idx: Index of the awake session (0-based)
    
    Returns:
        Filename like 'XY_raw_ah08_20250613_20250615_0.npy'
    """
    # Format dates without dashes; a single-day block carries the same date twice and is
    # named with it once (recday 'bp01_20240321'), matching recday_registry.block_to_recday.
    date1_fmt = date1.replace("-", "")
    date2_fmt = date2.replace("-", "")
    dates = date1_fmt if date2_fmt == date1_fmt else f"{date1_fmt}_{date2_fmt}"
    if len(BLOCK_DATES) > 2:
        dates = "_".join(d.replace("-", "") for d in BLOCK_DATES)
    return f"{data_type}_raw_{mouse_id}_{dates}_{session_idx}.npy"


def process_session(
    mouse_id: str,
    metadata_row: pd.Series,
    ephys_dates: Tuple[str, str],
    session_idx: int,
    output_dir: Path,
    verbose: bool = True
) -> bool:
    """
    Process a single awake session.
    
    Returns True if successful, False otherwise.
    """
    date1, date2 = ephys_dates
    
    # Get tracking and behaviour times from metadata
    date_str = format_date_for_filename(str(metadata_row['Date']))
    tracking_time = format_tracking_time(metadata_row.get('Tracking'))
    behaviour_time = str(int(metadata_row.get('Behaviour'))).zfill(6) if pd.notna(metadata_row.get('Behaviour')) else None
    
    if tracking_time is None or behaviour_time is None:
        if verbose:
            print(f"  Skipping session - missing tracking or behaviour time")
        return False
    
    tracking_datetime = f"{date_str}-{tracking_time}"
    
    if verbose:
        print(f"  Processing session: {tracking_datetime}")
    
    # Find pinstate file
    pinstate_file = find_pinstate_file(mouse_id, tracking_datetime)
    if pinstate_file is None:
        if verbose:
            print(f"    Pinstate file not found for {tracking_datetime}")
        return False
    
    # Load and check pinstate
    pinstate = load_pinstate(pinstate_file)
    first_up_frame = find_first_up_state(pinstate)
    
    if first_up_frame is None:
        if verbose:
            print(f"    No up states in pinstate, trying fallback...")
        # Try fallback
        fallback_result = find_pinstate_file_fallback(tracking_datetime, exclude_mouse=mouse_id)
        if fallback_result is None:
            if verbose:
                print(f"    No fallback pinstate found")
            return False
        pinstate_file, fallback_mouse = fallback_result
        pinstate = load_pinstate(pinstate_file)
        first_up_frame = find_first_up_state(pinstate)
        if first_up_frame is None:
            if verbose:
                print(f"    Fallback pinstate also has no up states")
            return False
        if verbose:
            print(f"    Using fallback pinstate from {fallback_mouse}")
    
    # Find pycontrol file
    pycontrol_file = find_pycontrol_file(mouse_id, date_str, behaviour_time)
    if pycontrol_file is None:
        if verbose:
            print(f"    Pycontrol file not found for {date_str}-{behaviour_time}")
        return False
    
    # Parse pycontrol file
    pycontrol_data = parse_pycontrol_file(pycontrol_file)
    
    # Find first rsync timestamp
    first_rsync_ts = find_first_rsync_timestamp(pycontrol_data)
    if first_rsync_ts is None:
        if verbose:
            print(f"    No rsync event found in pycontrol")
        return False
    
    # Find first A_on timestamp
    first_A_on_ts = find_first_A_on_timestamp(pycontrol_data)
    if first_A_on_ts is None:
        if verbose:
            print(f"    No A_on event found in pycontrol")
        return False
    
    # Calculate the frame where first A_on occurs
    first_A_on_frame = calculate_frame_from_pycontrol_time(
        first_A_on_ts, first_rsync_ts, first_up_frame
    )
    
    if verbose:
        print(f"    First rsync frame: {first_up_frame}")
        print(f"    First rsync pycontrol timestamp: {first_rsync_ts} ms")
        print(f"    First A_on pycontrol timestamp: {first_A_on_ts} ms")
        print(f"    First A_on frame: {first_A_on_frame}")
    
    # Load SLEAP files
    sleap_data = load_sleap_files(mouse_id, tracking_datetime)
    if not sleap_data:
        if verbose:
            print(f"    No SLEAP files found for {tracking_datetime}")
        return False
    
    # Process and save each data type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process coordinates (XY) - extract only head_back (columns 4, 5: head_back.x, head_back.y)
    if 'coordinates' in sleap_data:
        coords_df = sleap_data['coordinates']
        # Extract head_back.x and head_back.y columns
        head_back_cols = ['head_back.x', 'head_back.y']
        if all(col in coords_df.columns for col in head_back_cols):
            coords = coords_df[head_back_cols].values
        else:
            # Fallback to column indices 4, 5 if column names don't match
            coords = coords_df.values[:, 4:6]
        
        if first_A_on_frame >= 0 and first_A_on_frame < len(coords):
            # Cut data before first A_on
            coords_trimmed = coords[first_A_on_frame:]
            # Resample from 60Hz to 40Hz
            coords_resampled = resample_data(coords_trimmed, VIDEO_FPS, TARGET_FPS)
            # Save
            filename = create_output_filename('XY', mouse_id, date1, date2, session_idx)
            np.save(output_dir / filename, coords_resampled)
            if verbose:
                print(f"    Saved XY: {filename} (shape: {coords_resampled.shape})")
        else:
            if verbose:
                print(f"    Invalid A_on frame for coordinates: {first_A_on_frame}")
    
    # Process head direction (HD)
    if 'head_direction' in sleap_data:
        hd = sleap_data['head_direction'].values
        if first_A_on_frame >= 0 and first_A_on_frame < len(hd):
            hd_trimmed = hd[first_A_on_frame:]
            hd_resampled = resample_data(hd_trimmed, VIDEO_FPS, TARGET_FPS)
            filename = create_output_filename('HD', mouse_id, date1, date2, session_idx)
            np.save(output_dir / filename, hd_resampled)
            if verbose:
                print(f"    Saved HD: {filename} (shape: {hd_resampled.shape})")
        else:
            if verbose:
                print(f"    Invalid A_on frame for head direction: {first_A_on_frame}")
    
    # Process ROIs (Locs) - extract only head_back column (index 2)
    if 'rois' in sleap_data:
        rois_df = sleap_data['rois']
        # Extract head_back column
        if 'head_back' in rois_df.columns:
            rois = rois_df['head_back'].values
        else:
            # Fallback to column index 2 if column name doesn't match
            rois = rois_df.values[:, 2]
        
        if first_A_on_frame >= 0 and first_A_on_frame < len(rois):
            rois_trimmed = rois[first_A_on_frame:]
            # For categorical ROI data, use nearest neighbor resampling
            original_samples = len(rois_trimmed)
            target_samples = int(original_samples * TARGET_FPS / VIDEO_FPS)
            indices = np.linspace(0, original_samples - 1, target_samples).astype(int)
            rois_resampled = rois_trimmed[indices]
            filename = create_output_filename('Locs', mouse_id, date1, date2, session_idx)
            np.save(output_dir / filename, rois_resampled)
            if verbose:
                print(f"    Saved Locs: {filename} (shape: {rois_resampled.shape})")
        else:
            if verbose:
                print(f"    Invalid A_on frame for ROIs: {first_A_on_frame}")
    
    return True


def process_session_by_ephys_time(
    mouse_id: str,
    ephys_date: str,
    ephys_time: str,
    ephys_dates: Tuple[str, str],
    session_idx: int,
    output_dir: Path,
    max_diff_minutes: int = 2,
    verbose: bool = True
) -> bool:
    """
    Process a session using ephys timestamp to find matching files via fuzzy matching.
    
    This function finds SLEAP, pinstate, and pycontrol files by looking for the 
    closest timestamp match to the ephys recording time, which is more reliable
    than metadata timestamps that may have copy-paste errors.
    
    Args:
        mouse_id: Mouse identifier
        ephys_date: Date from ephys CSV (e.g., '2025-06-24')
        ephys_time: Time from ephys CSV (e.g., '11-43-01')
        ephys_dates: Tuple of (date1, date2) for the ephys date range
        session_idx: Session index for output filename
        output_dir: Directory to save output files
        max_diff_minutes: Maximum time difference for fuzzy matching
        verbose: Print debug output
    
    Returns:
        True if successful, False otherwise
    """
    date1, date2 = ephys_dates
    
    # Parse ephys timestamp
    ephys_time_fmt = ephys_time.replace('-', '')
    ephys_datetime_str = f"{ephys_date}-{ephys_time_fmt}"
    try:
        ephys_dt = datetime.strptime(f"{ephys_date} {ephys_time}", "%Y-%m-%d %H-%M-%S")
    except ValueError:
        if verbose:
            print(f"  Could not parse ephys timestamp: {ephys_date} {ephys_time}")
        return False
    
    if verbose:
        print(f"  Processing ephys session: {ephys_date} {ephys_time}")
    
    # Find closest SLEAP files
    sleap_result = find_closest_sleap_datetime(mouse_id, ephys_datetime_str, max_diff_minutes)
    if sleap_result is None:
        if verbose:
            print(f"    No SLEAP files found within {max_diff_minutes} min of {ephys_datetime_str}")
        return False
    
    actual_sleap_datetime = sleap_result
    if verbose and actual_sleap_datetime != ephys_datetime_str:
        print(f"    Matched SLEAP: {ephys_datetime_str} → {actual_sleap_datetime}")
    
    # Find closest pinstate file
    pinstate_result = find_closest_pinstate_file(mouse_id, ephys_dt, max_diff_minutes)
    if pinstate_result is None:
        if verbose:
            print(f"    No pinstate file found within {max_diff_minutes} min")
        return False
    
    pinstate_file, matched_pinstate_datetime = pinstate_result
    if verbose:
        print(f"    Matched pinstate: {matched_pinstate_datetime}")
    
    # Load and check pinstate - need actual pulses (rising edges), not just high values
    pinstate = load_pinstate(pinstate_file)
    n_pulses = count_up_states(pinstate)
    first_up_frame = find_first_up_state(pinstate) if n_pulses > 0 else None
    used_fallback_mouse = None
    
    if n_pulses == 0:
        if verbose:
            print(f"    No sync pulses in pinstate (0 rising edges), trying fallback...")
        # Try fallback from other mice recorded at same time
        for other_mouse in MICE:
            if other_mouse == mouse_id:
                continue
            other_result = find_closest_pinstate_file(other_mouse, ephys_dt, max_diff_minutes)
            if other_result:
                other_pinstate_file, _ = other_result
                other_pinstate = load_pinstate(other_pinstate_file)
                other_n_pulses = count_up_states(other_pinstate)
                if other_n_pulses > 0:
                    pinstate = other_pinstate
                    pinstate_file = other_pinstate_file
                    first_up_frame = find_first_up_state(pinstate)
                    used_fallback_mouse = other_mouse
                    if verbose:
                        print(f"    Using fallback pinstate from {other_mouse} ({other_n_pulses} pulses)")
                    break
        
        if first_up_frame is None:
            if verbose:
                print(f"    No valid pinstate found (all mice have 0 pulses)")
            return False
    else:
        if verbose:
            print(f"    Pinstate has {n_pulses} sync pulses")
    
    # Find closest pycontrol file
    pycontrol_file = find_closest_pycontrol_file(mouse_id, ephys_dt, max_diff_minutes)
    if pycontrol_file is None:
        if verbose:
            print(f"    No pycontrol file found within {max_diff_minutes} min")
        return False
    
    if verbose:
        print(f"    Matched pycontrol: {pycontrol_file.name}")
    
    # Parse pycontrol file
    pycontrol_data = parse_pycontrol_file(pycontrol_file)
    
    # Find first rsync timestamp
    first_rsync_ts = find_first_rsync_timestamp(pycontrol_data)
    if first_rsync_ts is None:
        if verbose:
            print(f"    No rsync event found in pycontrol")
        return False
    
    # Find first A_on timestamp
    first_A_on_ts = find_first_A_on_timestamp(pycontrol_data)
    if first_A_on_ts is None:
        if verbose:
            print(f"    No A_on event found in pycontrol - session may be sleep/test")
        return False
    
    # Calculate the frame where first A_on occurs
    first_A_on_frame = calculate_frame_from_pycontrol_time(
        first_A_on_ts, first_rsync_ts, first_up_frame
    )
    
    if verbose:
        print(f"    First rsync frame: {first_up_frame}")
        print(f"    First A_on frame: {first_A_on_frame}")
    
    # Load SLEAP files
    sleap_data, _ = load_sleap_files(mouse_id, actual_sleap_datetime, use_fuzzy_match=False, verbose=verbose)
    if not sleap_data:
        if verbose:
            print(f"    SLEAP files empty for {actual_sleap_datetime}")
        return False
    
    # Validate SLEAP data is properly populated
    coords_df = sleap_data.get('coordinates')
    if coords_df is None or len(coords_df) < 100:  # Require at least 100 frames
        if verbose:
            frames = len(coords_df) if coords_df is not None else 0
            print(f"    SLEAP data too short ({frames} frames)")
        return False
    
    # Process and save each data type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process coordinates (XY) - extract only head_back
    if 'coordinates' in sleap_data:
        coords_df = sleap_data['coordinates']
        head_back_cols = ['head_back.x', 'head_back.y']
        if all(col in coords_df.columns for col in head_back_cols):
            coords = coords_df[head_back_cols].values
        else:
            coords = coords_df.values[:, 4:6]
        
        if first_A_on_frame >= 0 and first_A_on_frame < len(coords):
            coords_trimmed = coords[first_A_on_frame:]
            coords_resampled = resample_data(coords_trimmed, VIDEO_FPS, TARGET_FPS)
            filename = create_output_filename('XY', mouse_id, date1, date2, session_idx)
            np.save(output_dir / filename, coords_resampled)
            if verbose:
                print(f"    Saved XY: {filename} (shape: {coords_resampled.shape})")
        else:
            if verbose:
                print(f"    Invalid A_on frame for coordinates: {first_A_on_frame}")
    
    # Process head direction (HD)
    if 'head_direction' in sleap_data:
        hd = sleap_data['head_direction'].values
        if first_A_on_frame >= 0 and first_A_on_frame < len(hd):
            hd_trimmed = hd[first_A_on_frame:]
            hd_resampled = resample_data(hd_trimmed, VIDEO_FPS, TARGET_FPS)
            filename = create_output_filename('HD', mouse_id, date1, date2, session_idx)
            np.save(output_dir / filename, hd_resampled)
            if verbose:
                print(f"    Saved HD: {filename} (shape: {hd_resampled.shape})")
    
    # Process ROIs (Locs) - extract only head_back
    if 'rois' in sleap_data:
        rois_df = sleap_data['rois']
        if 'head_back' in rois_df.columns:
            rois = rois_df['head_back'].values
        else:
            rois = rois_df.values[:, 2]
        
        if first_A_on_frame >= 0 and first_A_on_frame < len(rois):
            rois_trimmed = rois[first_A_on_frame:]
            original_samples = len(rois_trimmed)
            target_samples = int(original_samples * TARGET_FPS / VIDEO_FPS)
            indices = np.linspace(0, original_samples - 1, target_samples).astype(int)
            rois_resampled = rois_trimmed[indices]
            filename = create_output_filename('Locs', mouse_id, date1, date2, session_idx)
            np.save(output_dir / filename, rois_resampled)
            if verbose:
                print(f"    Saved Locs: {filename} (shape: {rois_resampled.shape})")
    
    return True


def match_metadata_to_ephys(
    metadata_df: pd.DataFrame, 
    ephys_df: pd.DataFrame, 
    date1: str, 
    date2: str
) -> List[Tuple[int, pd.Series]]:
    """
    Match metadata rows to ephys sessions based on date and ephys time.
    Only returns awake sessions.
    
    Returns list of (metadata_index, metadata_row) tuples for matching awake sessions.
    """
    matched_sessions = []
    
    # Convert date strings to comparable format
    def normalize_date(d):
        if isinstance(d, str):
            parts = d.split("-")
            if len(parts) == 3:
                return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        return str(d)
    
    # Get unique dates from ephys metadata
    ephys_dates = set(ephys_df['date'].apply(str).unique())
    
    for idx, row in metadata_df.iterrows():
        # Check if this is an awake session
        if not is_awake_session(row):
            continue
        
        # Get the date and ephys time from metadata
        meta_date = normalize_date(str(row['Date']))
        ephys_time = str(row.get('Ephys', '')).replace('-', ':') if pd.notna(row.get('Ephys')) else None
        
        # Check if this date is in the ephys date range
        if meta_date in ephys_dates:
            matched_sessions.append((idx, row))
    
    return matched_sessions


def process_mouse_ephys_folder(
    mouse_id: str, 
    ephys_folder: Path, 
    metadata_df: pd.DataFrame,
    verbose: bool = True
) -> int:
    """
    Process all awake sessions within an ephys folder for a mouse.
    Uses ephys timestamps directly to find matching files via fuzzy matching.
    
    Only sessions with A_on events (awake/task sessions) are indexed and saved.
    Session indices match the awake session ordering used by neuron files.
    
    Returns number of successfully processed sessions.
    """
    folder_name = ephys_folder.name
    date1, date2 = parse_ephys_folder_dates(folder_name)
    global BLOCK_DATES
    BLOCK_DATES = parse_ephys_folder_all_dates(folder_name)
    
    if date1 is None or date2 is None:
        if verbose:
            print(f"Could not parse dates from folder: {folder_name}")
        return 0
    
    if verbose:
        print(f"\nProcessing ephys folder: {folder_name}")
        print(f"  Date range: {date1} to {date2}")
    
    # Load ephys metadata
    try:
        ephys_df = load_ephys_metadata(ephys_folder)
    except FileNotFoundError as e:
        if verbose:
            print(f"  {e}")
        return 0
    
    if verbose:
        print(f"  Found {len(ephys_df)} ephys sessions in CSV")
    
    # Create output directory
    date1_fmt = date1.replace("-", "")
    date2_fmt = date2.replace("-", "")
    output_dir = PROCESSED_PATH / mouse_id / f"{date1_fmt}_{date2_fmt}"
    
    # Process each ephys session using timestamp-based matching
    # Only increment awake_session_idx for sessions that have A_on events (task sessions)
    # Skip duplicates where multiple ephys recordings match the same SLEAP+pycontrol
    successful = 0
    awake_session_idx = 0
    seen_combinations = set()  # Track processed (sleap, pycontrol) combos
    
    for _, row in ephys_df.iterrows():
        ephys_date = str(row['date'])
        ephys_time = str(row['time'])
        
        # Pre-check for duplicates before full processing
        ephys_time_fmt = ephys_time.replace('-', '')
        ephys_datetime_str = f"{ephys_date}-{ephys_time_fmt}"
        
        # Check if this would be a duplicate
        sleap_match = find_closest_sleap_datetime(mouse_id, ephys_datetime_str, max_diff_minutes=2)
        if sleap_match:
            try:
                ephys_dt = datetime.strptime(f"{ephys_date} {ephys_time}", "%Y-%m-%d %H-%M-%S")
                pycontrol = find_closest_pycontrol_file(mouse_id, ephys_dt, max_diff_minutes=2)
                if pycontrol:
                    combo_key = (sleap_match, pycontrol.name)
                    if combo_key in seen_combinations:
                        if verbose:
                            print(f"  Processing ephys session: {ephys_date} {ephys_time}")
                            print(f"    Skipping duplicate (same SLEAP+pycontrol as earlier session)")
                        continue
            except ValueError:
                pass
        
        result = process_session_by_ephys_time(
            mouse_id=mouse_id,
            ephys_date=ephys_date,
            ephys_time=ephys_time,
            ephys_dates=(date1, date2),
            session_idx=awake_session_idx,
            output_dir=output_dir,
            max_diff_minutes=2,
            verbose=verbose
        )
        if result:
            successful += 1
            awake_session_idx += 1
            # Record combo only on successful processing
            if sleap_match:
                try:
                    ephys_dt = datetime.strptime(f"{ephys_date} {ephys_time}", "%Y-%m-%d %H-%M-%S")
                    pycontrol = find_closest_pycontrol_file(mouse_id, ephys_dt, max_diff_minutes=2)
                    if pycontrol:
                        seen_combinations.add((sleap_match, pycontrol.name))
                except ValueError:
                    pass
    
    return successful


def process_mouse(mouse_id: str, verbose: bool = True) -> int:
    """
    Process all ephys folders for a given mouse.
    Uses ephys timestamps directly to find matching files.
    
    Returns total number of successfully processed sessions.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing mouse: {mouse_id}")
        print(f"{'='*60}")
    
    # Get all ephys folders
    ephys_folders = get_ephys_folders(mouse_id)
    if not ephys_folders:
        if verbose:
            print(f"No ephys folders found for {mouse_id}")
        return 0
    
    if verbose:
        print(f"Found {len(ephys_folders)} ephys folders")
    
    # Process each ephys folder
    total_successful = 0
    for ephys_folder in ephys_folders:
        successful = process_mouse_ephys_folder(
            mouse_id=mouse_id,
            ephys_folder=ephys_folder,
            metadata_df=None,  # Not needed anymore
            verbose=verbose
        )
        total_successful += successful
    
    return total_successful


def process_session_by_metadata(
    mouse_id: str,
    metadata_row: pd.Series,
    ephys_dates: Tuple[str, str],
    session_idx: int,
    output_dir: Path,
    max_diff_minutes: int = 2,
    verbose: bool = True
) -> bool:
    """
    Process a session using metadata tracking time to find SLEAP files.
    
    This uses the tracking time from metadata (which matches SLEAP filenames exactly)
    to ensure proper session ordering that aligns with trialtimes indices.
    
    Args:
        mouse_id: Mouse identifier
        metadata_row: Row from metadata DataFrame with session info
        ephys_dates: Tuple of (date1, date2) for the ephys date range
        session_idx: Session index for output filename (matches trialtimes)
        output_dir: Directory to save output files
        max_diff_minutes: Maximum time difference for fuzzy matching
        verbose: Print debug output
    
    Returns:
        True if successful, False otherwise
    """
    date1, date2 = ephys_dates

    # Extract info from metadata
    meta_date = str(metadata_row['Date'])
    tracking_time = str(metadata_row.get('Tracking', ''))
    behaviour_time = str(metadata_row.get('Behaviour', ''))

    # Object / open-field sessions have no A_on and their SLEAP CSVs live in a different
    # folder; they are aligned only under --sessions openfield|all and --sync matched.
    kind = session_kind(metadata_row)
    is_openfield = kind in ("object", "openfield")
    if is_openfield and SESSIONS == "task":
        return False
    if not is_openfield and SESSIONS == "openfield":
        return False
    if is_openfield and SYNC_METHOD != "matched":
        if verbose:
            print(f"    {kind} session needs --sync matched (anchored at rsync[0]); skipped")
        return False
    sleap_root = SLEAP_PATH_OPENFIELD if is_openfield else SLEAP_PATH

    if tracking_time == '-' or tracking_time == 'nan' or not tracking_time:
        if verbose:
            print(f"    No tracking time in metadata")
        return False
    
    # Normalize date format: 2025-6-13 -> 2025-06-13
    date_parts = meta_date.split('-')
    if len(date_parts) == 3:
        meta_date_fmt = f"{date_parts[0]}-{int(date_parts[1]):02d}-{int(date_parts[2]):02d}"
    else:
        meta_date_fmt = meta_date
    
    # Build tracking datetime string for SLEAP matching
    tracking_str = str(int(float(tracking_time))).zfill(6)
    tracking_datetime_str = f"{meta_date_fmt}-{tracking_str}"
    
    if verbose:
        print(f"  Processing metadata session: {meta_date} tracking={tracking_str}")
    
    # Find SLEAP files by tracking time (should match exactly or very close)
    sleap_datetime = find_closest_sleap_datetime(mouse_id, tracking_datetime_str, max_diff_minutes, sleap_root)
    if sleap_datetime is None:
        if verbose:
            print(f"    No SLEAP files found for tracking time {tracking_datetime_str}")
        return False
    
    if verbose and sleap_datetime != tracking_datetime_str:
        print(f"    Matched SLEAP: {tracking_datetime_str} → {sleap_datetime}")
    
    # Parse datetime for pycontrol/pinstate matching
    try:
        tracking_dt = datetime.strptime(f"{meta_date_fmt} {tracking_str[:2]}:{tracking_str[2:4]}:{tracking_str[4:6]}", 
                                        "%Y-%m-%d %H:%M:%S")
    except ValueError:
        if verbose:
            print(f"    Could not parse tracking time: {tracking_str}")
        return False
    
    # Find closest pinstate file
    pinstate_result = find_closest_pinstate_file(mouse_id, tracking_dt, max_diff_minutes=5)
    if pinstate_result is None:
        if verbose:
            print(f"    No pinstate file found")
        return False
    
    pinstate_file, matched_pinstate_datetime = pinstate_result
    
    # Load and check pinstate
    pinstate = load_pinstate(pinstate_file)
    n_pulses = count_up_states(pinstate)
    first_up_frame = find_first_up_state(pinstate) if n_pulses > 0 else None
    used_fallback_mouse = None
    
    if verbose:
        print(f"    Pinstate: {pinstate_file.name} ({n_pulses} pulses)")
    
    # If no pulses, try fallback to other mice
    if first_up_frame is None:
        fallback_result = find_pinstate_file_fallback(tracking_datetime_str, exclude_mouse=mouse_id)
        if fallback_result:
            fallback_file, fallback_mouse = fallback_result
            fallback_pinstate = load_pinstate(fallback_file)
            fallback_n_pulses = count_up_states(fallback_pinstate)
            if fallback_n_pulses > 0:
                pinstate = fallback_pinstate
                first_up_frame = find_first_up_state(pinstate)
                used_fallback_mouse = fallback_mouse
                if verbose:
                    print(f"    Using fallback pinstate from {fallback_mouse}: {fallback_file.name} ({fallback_n_pulses} pulses)")
    
    if first_up_frame is None:
        if verbose:
            print(f"    No sync pulses found in pinstate")
        return False
    
    # pyControl file. Under --sync matched: the sheet's Behaviour stamp through the registry
    # (the same mapping trialtimes and the spikes use; 60 s rule), because the nearest file to
    # the tracking time can be an aborted start -- ah10 2025-06-24 13:58:58 holds one rsync
    # and sits 25 s from the tracking stamp, the session's file is 13:59:48. Legacy path
    # unchanged: nearest file within 5 min.
    pycontrol_file = None
    if SYNC_METHOD == 'matched':
        import recday_registry as _rr
        recday = create_output_filename('X', mouse_id, date1, date2, 0)[6:-6]
        try:
            pycontrol_file = _rr.pycontrol_file(recday, session_idx)
        except Exception as exc:                                   # noqa: BLE001
            if verbose:
                print(f"    registry has no pyControl file for {recday} s{session_idx} ({exc}); using nearest")
            pycontrol_file = None
        if pycontrol_file is not None and not Path(pycontrol_file).exists():
            pycontrol_file = None
    if pycontrol_file is None:
        pycontrol_file = find_closest_pycontrol_file(mouse_id, tracking_dt, max_diff_minutes=5)
    if pycontrol_file is None:
        if verbose:
            print(f"    No pycontrol file found")
        return False
    if verbose:
        print(f"    pyControl: {Path(pycontrol_file).name}")

    # Parse pycontrol to get first A_on
    pycontrol_data = parse_pycontrol_file(pycontrol_file)
    first_rsync = find_first_rsync_timestamp(pycontrol_data)
    first_A_on = find_first_A_on_timestamp(pycontrol_data)
    
    if first_A_on is None:
        if (is_openfield or SYNC_METHOD == 'matched') and first_rsync is not None:
            # No task: bin 0 is pyControl rsync[0], exactly where extract_neuron_raw's
            # no-trials branch puts the spikes' origin. Under --sync matched this also
            # covers sheet-ABCD sessions in which no trial was started (mEC: bp01 03-24
            # 14:18, 03-25 13:36, 04-03 14:39, bp02 03-26 11:53 -- 10-20 min of free running
            # on the maze with spikes but, until now, no tracking). Legacy path unchanged.
            first_A_on = first_rsync
            if verbose and not is_openfield:
                print(f"    No A_on event in pycontrol: no trials; bin 0 = rsync[0] as for the spikes")
        else:
            if verbose:
                print(f"    No A_on event in pycontrol")
            return False
    
    if first_rsync is None:
        first_rsync = 0
    
    # Calculate frame offset
    time_to_A_on_ms = first_A_on - first_rsync
    frames_at_video_fps = int(time_to_A_on_ms * VIDEO_FPS / 1000)
    first_A_on_frame = first_up_frame + frames_at_video_fps

    if verbose:
        print(f"    First A_on frame: {first_A_on_frame} (rsync_offset={first_up_frame}, A_on_ms={time_to_A_on_ms})")

    # Load SLEAP files using the matched datetime (returns tuple: dict, actual_datetime)
    sleap_data, _ = load_sleap_files(mouse_id, sleap_datetime, root=sleap_root)
    if not sleap_data:
        if verbose:
            print(f"    Failed to load SLEAP files for {sleap_datetime}")
        return False

    # Process and save each data type
    output_dir.mkdir(parents=True, exist_ok=True)

    if SYNC_METHOD == 'matched':
        return _save_matched(mouse_id, date1, date2, session_idx, output_dir, sleap_data,
                             pinstate, pycontrol_data, first_A_on, used_fallback_mouse,
                             pinstate_file.name, verbose)

    # Process coordinates (XY) - extract only head_back
    if 'coordinates' in sleap_data:
        coords_df = sleap_data['coordinates']
        head_back_cols = ['head_back.x', 'head_back.y']
        if all(col in coords_df.columns for col in head_back_cols):
            coords = coords_df[head_back_cols].values
        else:
            coords = coords_df.values[:, 4:6]

        if first_A_on_frame >= 0 and first_A_on_frame < len(coords):
            coords_trimmed = coords[first_A_on_frame:]
            coords_resampled = resample_data(coords_trimmed, VIDEO_FPS, TARGET_FPS)
            filename = create_output_filename('XY', mouse_id, date1, date2, session_idx)
            status = save_new(output_dir / filename, coords_resampled, verbose)
            if verbose:
                print(f"    Saved XY: {filename} (shape: {coords_resampled.shape}) [{status}]")

    # Process head direction (HD)
    if 'head_direction' in sleap_data:
        hd = sleap_data['head_direction'].values
        if first_A_on_frame >= 0 and first_A_on_frame < len(hd):
            hd_trimmed = hd[first_A_on_frame:]
            hd_resampled = resample_data(hd_trimmed, VIDEO_FPS, TARGET_FPS)
            filename = create_output_filename('HD', mouse_id, date1, date2, session_idx)
            status = save_new(output_dir / filename, hd_resampled, verbose)
            if verbose:
                print(f"    Saved HD: {filename} (shape: {hd_resampled.shape}) [{status}]")

    # Process ROIs (Locs) - extract only head_back
    if 'rois' in sleap_data:
        rois_df = sleap_data['rois']
        if 'head_back' in rois_df.columns:
            rois = rois_df['head_back'].values
        else:
            rois = rois_df.values[:, 2]

        if first_A_on_frame >= 0 and first_A_on_frame < len(rois):
            rois_trimmed = rois[first_A_on_frame:]
            original_samples = len(rois_trimmed)
            target_samples = int(original_samples * TARGET_FPS / VIDEO_FPS)
            indices = np.linspace(0, original_samples - 1, target_samples).astype(int)
            rois_resampled = rois_trimmed[indices]
            filename = create_output_filename('Locs', mouse_id, date1, date2, session_idx)
            status = save_new(output_dir / filename, rois_resampled, verbose)
            if verbose:
                print(f"    Saved Locs: {filename} (shape: {rois_resampled.shape}) [{status}]")

    return True


def _save_matched(mouse_id, date1, date2, session_idx, output_dir, sleap_data, pinstate,
                  pycontrol_data, first_A_on, used_fallback_mouse, pinstate_name, verbose) -> bool:
    """`SYNC_METHOD == 'matched'`: sample XY / HD / Locs at the fitted frame of every 25 ms bin.

    Bin k starts at pyControl time `first_A_on + 25 k` ms and is read at frame
    `a * t + b` -- XY by linear interpolation, HD and Locs at the nearest frame. Bins run
    until the last video frame. Fitting fails loudly rather than falling back to the
    first-edge method, so a session is either on the fitted clock or absent.
    """
    if used_fallback_mouse is not None:
        if verbose:
            print(f"    matched sync refuses a fallback pinstate ({used_fallback_mouse}): "
                  "another box's camera line is not this mouse's clock")
        return False
    try:
        clock = fit_frame_clock(rising_edges(pinstate), all_rsync_timestamps(pycontrol_data))
    except ValueError as exc:
        if verbose:
            print(f"    matched sync failed: {exc}")
        return False
    if clock['n_matched'] < MIN_MATCHED_FRACTION * clock['n_rsync']:
        if verbose:
            print(f"    matched sync refused: only {clock['n_matched']}/{clock['n_rsync']} rsyncs anchored")
        return False
    n_frames = min(len(v) for v in sleap_data.values())
    bin_ms = 1000.0 / TARGET_FPS
    f_start = float(frames_at(clock, [first_A_on])[0])
    t_last = ms_at_frame(clock, n_frames - 1)              # pyControl time of the last video frame
    n_bins = int(np.floor((t_last - first_A_on) / bin_ms))
    if f_start < 0 or n_bins <= 0:
        if verbose:
            print(f"    first A_on maps to frame {f_start:.1f} of {n_frames}; nothing to save")
        return False
    t_bins = first_A_on + np.arange(n_bins) * bin_ms
    frames = frames_at(clock, t_bins)
    # Bins inside an anchor gap that hides dropped frames (frame_clock.uncertain_gaps) have no
    # frame index better than +-step, so they are written untracked (XY/HD NaN, Locs 0).
    uncertain = _frame_clock.uncertain_mask(clock, t_bins)
    SYNC_QC.append(dict(mouse=mouse_id, recday=create_output_filename('X', mouse_id, date1, date2, 0)[6:-6],
                        session=session_idx, pinstate=pinstate_name, first_A_on_frame=f_start,
                        n_bins=n_bins, n_frames=n_frames, n_bins_uncertain=int(uncertain.sum()),
                        tracking_clean=TRACKING_CLEAN,
                        **{k: v for k, v in clock.items() if not k.startswith('anchors')}))
    if verbose:
        print(f"    matched clock: {clock['fps']:.4f} fps, {clock['n_matched']}/{clock['n_rsync']} rsyncs "
              f"anchored ({clock['n_jumps']} jumps, max {clock['max_jump_frames']:.0f} fr, "
              f"{clock['total_drop_frames']:.0f} fr dropped, {int(uncertain.sum())} uncertain bins), "
              f"first A_on at frame {f_start:.1f}, {n_bins} bins")

    if 'coordinates' in sleap_data:
        df = sleap_data['coordinates']
        cols = ['head_back.x', 'head_back.y']
        coords = df[cols].values if all(c in df.columns for c in cols) else df.values[:, 4:6]
        arr = sample_at_frames(coords, frames, 'linear')
        arr[uncertain] = np.nan
        if PROPAGATE_TRACK_NAN:
            untracked = untracked_bins(coords, frames)
            arr[untracked] = np.nan
            SYNC_QC[-1]['n_bins_track_nan'] = int(untracked.sum())
        name = create_output_filename('XY', mouse_id, date1, date2, session_idx)
        status = save_new(output_dir / name, arr, verbose)
        if verbose:
            print(f"    Saved XY: {name} (shape: {arr.shape}) [{status}]")
    if 'head_direction' in sleap_data:
        arr = sample_at_frames(sleap_data['head_direction'].values, frames, 'nearest')
        arr = arr.astype(float)
        arr[uncertain] = np.nan
        name = create_output_filename('HD', mouse_id, date1, date2, session_idx)
        status = save_new(output_dir / name, arr, verbose)
        if verbose:
            print(f"    Saved HD: {name} (shape: {arr.shape}) [{status}]")
    if 'rois' in sleap_data:
        df = sleap_data['rois']
        rois = df['head_back'].values if 'head_back' in df.columns else df.values[:, 2]
        arr = sample_at_frames(rois, frames, 'nearest')
        if uncertain.any():
            arr = arr.astype(object) if arr.dtype.kind in 'OU' else arr
            arr[uncertain] = 0
        name = create_output_filename('Locs', mouse_id, date1, date2, session_idx)
        status = save_new(output_dir / name, arr, verbose)
        if verbose:
            print(f"    Saved Locs: {name} (shape: {arr.shape}) [{status}]")
    return True


def process_mouse_ephys_folder_by_metadata(
    mouse_id: str,
    ephys_folder: Path,
    metadata_df: pd.DataFrame,
    verbose: bool = True
) -> int:
    """
    Process all awake sessions within an ephys folder using metadata as source of truth.
    
    This ensures session indices align with trialtimes by iterating through
    awake sessions in metadata order and using tracking times for SLEAP matching.
    
    Returns number of successfully processed sessions.
    """
    folder_name = ephys_folder.name
    date1, date2 = parse_ephys_folder_dates(folder_name)
    global BLOCK_DATES
    BLOCK_DATES = parse_ephys_folder_all_dates(folder_name)
    
    if date1 is None or date2 is None:
        if verbose:
            print(f"Could not parse dates from folder: {folder_name}")
        return 0
    
    if verbose:
        print(f"\nProcessing ephys folder: {folder_name}")
        print(f"  Date range: {date1} to {date2}")
    
    # Convert date range to match metadata format
    start_date = datetime.strptime(date1, "%Y-%m-%d")
    end_date = datetime.strptime(date2, "%Y-%m-%d")
    
    # Create output directory (one date for a single-day block, as in the recday name)
    date1_fmt = date1.replace("-", "")
    date2_fmt = date2.replace("-", "")
    output_dir = PROCESSED_PATH / mouse_id / (date1_fmt if date2_fmt == date1_fmt
                                              else f"{date1_fmt}_{date2_fmt}")
    if len(BLOCK_DATES) > 2:
        output_dir = PROCESSED_PATH / mouse_id / "_".join(d.replace("-", "") for d in BLOCK_DATES)

    # Find awake sessions in metadata within date range
    awake_sessions = []
    for idx, row in metadata_df.iterrows():
        meta_date_str = str(row['Date'])
        # Parse metadata date (format: 2025-6-13)
        try:
            date_parts = meta_date_str.split('-')
            meta_date = datetime(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
        except:
            continue
        
        # Check if within date range and is awake session
        if start_date <= meta_date <= end_date and is_awake_session(row):
            awake_sessions.append(row)
    
    if verbose:
        print(f"  Found {len(awake_sessions)} awake sessions in metadata")
    
    # Process each awake session in order
    successful = 0
    for session_idx, row in enumerate(awake_sessions):
        result = process_session_by_metadata(
            mouse_id=mouse_id,
            metadata_row=row,
            ephys_dates=(date1, date2),
            session_idx=session_idx,
            output_dir=output_dir,
            max_diff_minutes=2,
            verbose=verbose
        )
        if result:
            successful += 1
    
    return successful


def process_mouse_by_metadata(mouse_id: str, verbose: bool = True, ephys_folder_filter: str = None) -> int:
    """
    Process all ephys folders for a mouse using metadata as source of truth.
    
    This ensures session indices align with trialtimes indices.
    
    Args:
        mouse_id: Mouse identifier
        verbose: Print detailed output
        ephys_folder_filter: If specified, only process folders containing this string
    
    Returns total number of successfully processed sessions.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing mouse (metadata-driven): {mouse_id}")
        print(f"{'='*60}")
    
    # Load metadata
    try:
        metadata_df = load_metadata(mouse_id)
    except FileNotFoundError:
        if verbose:
            print(f"No metadata file found for {mouse_id}")
        return 0
    
    # Get all ephys folders
    ephys_folders = get_ephys_folders(mouse_id)
    if not ephys_folders:
        if verbose:
            print(f"No ephys folders found for {mouse_id}")
        return 0
    
    # Filter ephys folders if specified
    if ephys_folder_filter:
        ephys_folders = [f for f in ephys_folders if ephys_folder_filter in str(f.name)]
    
    if verbose:
        print(f"Found {len(ephys_folders)} ephys folders")
    
    # Process each ephys folder
    total_successful = 0
    for ephys_folder in ephys_folders:
        successful = process_mouse_ephys_folder_by_metadata(
            mouse_id=mouse_id,
            ephys_folder=ephys_folder,
            metadata_df=metadata_df,
            verbose=verbose
        )
        total_successful += successful
    
    return total_successful


def main_metadata(animals: List[str] = None, ephys_folder_filter: str = None):
    """Main entry point using metadata-driven processing.
    
    Args:
        animals: List of mouse IDs to process (default: all MICE)
        ephys_folder_filter: If specified, only process folders containing this string
    """
    print("SLEAP Alignment Preprocessing Pipeline (Metadata-driven)")
    print("="*60)
    
    # Create output directory
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    
    # Determine which mice to process
    mice_to_process = animals if animals else MICE
    
    # Process each mouse using metadata
    total_sessions = 0
    for mouse_id in mice_to_process:
        sessions = process_mouse_by_metadata(
            mouse_id, 
            verbose=True, 
            ephys_folder_filter=ephys_folder_filter
        )
        total_sessions += sessions
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Processed {total_sessions} sessions total.")
    print(f"Output saved to: {PROCESSED_PATH}")

    if SYNC_QC:
        qc = pd.DataFrame(SYNC_QC)
        path = PROCESSED_PATH / f"sync_qc_{rr.DATASET}.csv"
        n = 1
        while path.exists():
            path = PROCESSED_PATH / f"sync_qc_{rr.DATASET}.v{n}.csv"
            n += 1
        qc.to_csv(path, index=False)
        print(f"matched-clock QC: {len(qc)} sessions, fps median {qc['fps'].median():.4f} "
              f"(min {qc['fps'].min():.4f}, max {qc['fps'].max():.4f}), rsyncs anchored median "
              f"{(qc['n_matched'] / qc['n_rsync']).median():.3f} (min {(qc['n_matched'] / qc['n_rsync']).min():.3f}), "
              f"sessions with dropped-frame jumps {int((qc['n_jumps'] > 0).sum())}, max jump "
              f"{qc['max_jump_frames'].max():.0f} frames -> {path}")


def validate_alignment(
    mouse_id: str,
    date1: str,
    date2: str,
    session_idx: int,
    trialtimes_path: Path = None,
    verbose: bool = True
) -> Tuple[int, int, List[Dict]]:
    """
    Validate alignment by checking if mouse is at expected location during task states.
    
    Uses existing trialtimes and task_data files to verify that the mouse position
    in our processed Locs file matches the expected port at each task state.
    
    Args:
        mouse_id: Mouse identifier (e.g., 'ah08')
        date1: First date string with no dashes (e.g., '20250613')
        date2: Second date string with no dashes (e.g., '20250615')
        session_idx: Session index
        trialtimes_path: Path to trialtimes directory (default: standard location)
        verbose: Print detailed output
    
    Returns:
        Tuple of (correct_count, total_count, mismatch_details)
    """
    from collections import Counter
    
    if trialtimes_path is None:
        trialtimes_path = DATA_PATH / "processed_data" / "trialtimes_raw_mingyutest"
    
    output_path = PROCESSED_PATH / mouse_id / f"{date1}_{date2}"
    
    def extract_port(loc):
        loc_str = str(loc)
        if loc_str.startswith('node_'):
            return int(loc_str.split('_')[1])
        return -1
    
    try:
        task_data = np.load(trialtimes_path / f"Task_data_{mouse_id}_{date1}_{date2}_{session_idx}.npy", allow_pickle=True)
        trialtimes = np.load(trialtimes_path / f"trialtimes_{mouse_id}_{date1}_{date2}_{session_idx}.npy", allow_pickle=True)
        locs = np.load(output_path / f"Locs_raw_{mouse_id}_{date1}_{date2}_{session_idx}.npy", allow_pickle=True)
    except FileNotFoundError as e:
        if verbose:
            print(f"File not found: {e}")
        return 0, 0, []
    
    port_map = np.array([extract_port(l) for l in locs])
    
    correct = 0
    total = 0
    mismatches = []
    state_names = ['A_on', 'B_on', 'C_on', 'D_on']
    
    for trial_idx in range(len(trialtimes)):
        trial = trialtimes[trial_idx]
        for state_idx in range(4):
            expected_port = task_data[state_idx]
            time_start = int(trial[state_idx])
            
            if time_start < len(port_map):
                window_start = max(0, time_start - 5)
                window_end = min(len(port_map), time_start + 15)
                actual_ports = port_map[window_start:window_end]
                port_counts = Counter(actual_ports)
                most_common = port_counts.most_common(1)[0][0] if port_counts else -1
                
                if most_common == expected_port:
                    correct += 1
                else:
                    mismatches.append({
                        'trial': trial_idx,
                        'state': state_names[state_idx],
                        'time': time_start,
                        'expected': expected_port,
                        'found': most_common,
                        'all_ports': dict(port_counts)
                    })
                total += 1
    
    if verbose:
        pct = (correct / total * 100) if total > 0 else 0
        status = '✓' if pct == 100 else '✗'
        print(f"Session {session_idx}: {correct}/{total} ({pct:.1f}%) {status}")
        if mismatches:
            for m in mismatches:
                print(f"  Mismatch: Trial {m['trial']}, {m['state']} (t={m['time']}): expected {m['expected']}, found {m['found']}")
    
    return correct, total, mismatches


def validate_all_sessions(
    mouse_id: str,
    date1: str,
    date2: str,
    trialtimes_path: Path = None,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Validate alignment for all sessions in a date range.
    
    Args:
        mouse_id: Mouse identifier
        date1: First date string (no dashes)
        date2: Second date string (no dashes)
        trialtimes_path: Path to trialtimes directory
        verbose: Print output
    
    Returns:
        Tuple of (total_correct, total_count)
    """
    if verbose:
        print(f"\n=== Validation Summary for {mouse_id} {date1} to {date2} ===\n")
    
    all_correct = 0
    all_total = 0
    session_idx = 0
    
    while True:
        correct, total, _ = validate_alignment(
            mouse_id, date1, date2, session_idx, 
            trialtimes_path, verbose=verbose
        )
        if total == 0:
            break
        all_correct += correct
        all_total += total
        session_idx += 1
    
    if verbose and all_total > 0:
        print(f"\nOverall: {all_correct}/{all_total} ({all_correct/all_total*100:.1f}%)")
    
    return all_correct, all_total


def validate_with_locations(
    mouse_id: str,
    date1: str,
    date2: str,
    session_idx: int,
    trialtimes_path: Path = None
) -> Dict:
    """
    Validate alignment and return detailed location info at each trialtime.
    
    Returns dict with:
        - locations_at_trialtimes: array of actual locations at each (trial, state)
        - expected_ports: expected port for each state
        - correct: number correct
        - total: total checked
        - accuracy: percentage
    """
    from collections import Counter
    
    if trialtimes_path is None:
        trialtimes_path = DATA_PATH / "processed_data" / "trialtimes_raw_mingyutest"
    
    output_path = PROCESSED_PATH / mouse_id / f"{date1}_{date2}"
    
    def extract_port(loc):
        loc_str = str(loc)
        if loc_str.startswith('node_'):
            return int(loc_str.split('_')[1])
        return -1
    
    try:
        task_data = np.load(trialtimes_path / f"Task_data_{mouse_id}_{date1}_{date2}_{session_idx}.npy", allow_pickle=True)
        trialtimes = np.load(trialtimes_path / f"trialtimes_{mouse_id}_{date1}_{date2}_{session_idx}.npy", allow_pickle=True)
        locs = np.load(output_path / f"Locs_raw_{mouse_id}_{date1}_{date2}_{session_idx}.npy", allow_pickle=True)
    except FileNotFoundError:
        return None
    
    port_map = np.array([extract_port(l) for l in locs])
    
    n_trials = len(trialtimes)
    locations_at_trialtimes = np.zeros((n_trials, 4), dtype=int)
    correct = 0
    total = 0
    
    for trial_idx in range(n_trials):
        trial = trialtimes[trial_idx]
        for state_idx in range(4):
            expected_port = task_data[state_idx]
            time_start = int(trial[state_idx])
            
            if time_start < len(port_map):
                window_start = max(0, time_start - 5)
                window_end = min(len(port_map), time_start + 15)
                actual_ports = port_map[window_start:window_end]
                port_counts = Counter(actual_ports)
                most_common = port_counts.most_common(1)[0][0] if port_counts else -1
                locations_at_trialtimes[trial_idx, state_idx] = most_common
                
                if most_common == expected_port:
                    correct += 1
                total += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    return {
        'locations_at_trialtimes': locations_at_trialtimes,
        'expected_ports': task_data,
        'correct': correct,
        'total': total,
        'accuracy': accuracy
    }


def main():
    """Main entry point for the preprocessing pipeline."""
    print("SLEAP Alignment Preprocessing Pipeline")
    print("="*60)
    
    # Create output directory
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    
    # Process each mouse
    total_sessions = 0
    for mouse_id in MICE:
        sessions = process_mouse(mouse_id, verbose=True)
        total_sessions += sessions
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Processed {total_sessions} sessions total.")
    print(f"Output saved to: {PROCESSED_PATH}")
    
    # Run validation with location output
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print("="*60)
    
    trialtimes_path = DATA_PATH / "processed_data" / "trialtimes_raw_mingyutest"
    grand_correct = 0
    grand_total = 0
    
    for mouse_id in MICE:
        mouse_output = PROCESSED_PATH / mouse_id
        if not mouse_output.exists():
            continue
        
        for date_folder in sorted(mouse_output.iterdir()):
            if not date_folder.is_dir():
                continue
            
            date_parts = date_folder.name.split('_')
            if len(date_parts) != 2:
                continue
            
            date1, date2 = date_parts
            print(f"\n--- {mouse_id} {date1} to {date2} ---")
            
            session_idx = 0
            while True:
                result = validate_with_locations(mouse_id, date1, date2, session_idx, trialtimes_path)
                if result is None:
                    break
                
                locs = result['locations_at_trialtimes']
                expected = result['expected_ports']
                acc = result['accuracy']
                correct = result['correct']
                total = result['total']
                
                print(f"\nSession {session_idx}: {correct}/{total} ({acc:.1f}%)")
                print(f"  Expected ports (A,B,C,D): {list(expected)}")
                print(f"  Locations at trialtimes (trial x state):")
                for trial_idx, row in enumerate(locs):
                    match_str = ''.join(['✓' if row[i] == expected[i] else '✗' for i in range(4)])
                    print(f"    Trial {trial_idx}: {list(row)} {match_str}")
                
                grand_correct += correct
                grand_total += total
                session_idx += 1
    
    if grand_total > 0:
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: {grand_correct}/{grand_total} ({grand_correct/grand_total*100:.1f}%)")
        print("="*60)


def run_validation_only(save_json: bool = True) -> Dict:
    """Run validation on all processed data and optionally save to JSON."""
    import json
    from datetime import datetime
    
    print("SLEAP Alignment Validation")
    print("="*60)
    
    trialtimes_path = DATA_PATH / "processed_data" / "trialtimes_raw_mingyutest"
    grand_correct = 0
    grand_total = 0
    
    all_results = {
        'run_timestamp': datetime.now().isoformat(),
        'mice': {}
    }
    
    for mouse_id in MICE:
        mouse_output = PROCESSED_PATH / mouse_id
        if not mouse_output.exists():
            continue
        
        all_results['mice'][mouse_id] = {}
        
        for date_folder in sorted(mouse_output.iterdir()):
            if not date_folder.is_dir():
                continue
            
            date_parts = date_folder.name.split('_')
            if len(date_parts) != 2:
                continue
            
            date1, date2 = date_parts
            date_key = f"{date1}_{date2}"
            all_results['mice'][mouse_id][date_key] = {'sessions': []}
            
            print(f"\n--- {mouse_id} {date1} to {date2} ---")
            
            session_idx = 0
            while True:
                result = validate_with_locations(mouse_id, date1, date2, session_idx, trialtimes_path)
                if result is None:
                    break
                
                locs = result['locations_at_trialtimes']
                expected = result['expected_ports']
                acc = result['accuracy']
                correct = result['correct']
                total = result['total']
                
                # Store session results
                session_data = {
                    'session_idx': session_idx,
                    'correct': correct,
                    'total': total,
                    'accuracy': acc,
                    'expected_ports': [int(x) for x in expected],
                    'locations_at_trialtimes': locs.tolist(),
                    'trial_matches': []
                }
                
                print(f"\nSession {session_idx}: {correct}/{total} ({acc:.1f}%)")
                print(f"  Expected ports (A,B,C,D): {list(expected)}")
                print(f"  Locations at trialtimes (trial x state):")
                for trial_idx, row in enumerate(locs):
                    matches = [bool(row[i] == expected[i]) for i in range(4)]
                    match_str = ''.join(['✓' if m else '✗' for m in matches])
                    print(f"    Trial {trial_idx}: {list(row)} {match_str}")
                    session_data['trial_matches'].append({
                        'trial': trial_idx,
                        'locations': [int(x) for x in row],
                        'matches': matches,
                        'all_correct': all(matches)
                    })
                
                all_results['mice'][mouse_id][date_key]['sessions'].append(session_data)
                grand_correct += correct
                grand_total += total
                session_idx += 1
    
    # Add summary
    all_results['summary'] = {
        'grand_correct': grand_correct,
        'grand_total': grand_total,
        'accuracy': (grand_correct / grand_total * 100) if grand_total > 0 else 0
    }
    
    if grand_total > 0:
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: {grand_correct}/{grand_total} ({grand_correct/grand_total*100:.1f}%)")
        print("="*60)
    
    # Save to JSON
    if save_json:
        json_path = PROCESSED_PATH / "validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nValidation results saved to: {json_path}")
    
    return all_results


if __name__ == "__main__":
    import sys
    import argparse

    # Mode flags are looked for anywhere in argv, not only at argv[1]: with `--dataset mec`
    # in front, `--metadata` used to be missed and the legacy ephys-time path ran instead
    # (2026-09-18, mEC; its output was quarantined as data/processed_LEGACY_*).
    if '--validate-only' in sys.argv:
        run_validation_only()
    elif '--metadata' in sys.argv:
        # Use metadata-driven processing (recommended for proper session ordering)
        parser = argparse.ArgumentParser(description='SLEAP alignment preprocessing (metadata-driven)')
        parser.add_argument('--dataset', default=rr.DATASET, choices=sorted(rr.DATASETS),
                            help='which tree to work on (read before import; default lec)')
        parser.add_argument('--metadata', action='store_true', help='Use metadata-driven processing')
        parser.add_argument('--animals', nargs='+', help='Specific animals to process')
        parser.add_argument('--ephys_folder', type=str, help='Filter by ephys folder name (partial match)')
        parser.add_argument('--sync', choices=('first_edge', 'matched'), default=SYNC_METHOD,
                            help="frame clock: first pinstate edge + nominal fps (LEC default) or a "
                                 "per-session fit of the whole pinstate train (mEC)")
        parser.add_argument('--sessions', choices=('task', 'openfield', 'all'), default=SESSIONS,
                            help="which sheet sessions to align (default task, as before)")
        parser.add_argument('--out-tag', default='',
                            help="write to data/processed_<tag>/ instead of data/processed/")
        parser.add_argument('--tracking-clean', choices=tuple(TRACKING_ROOTS), default=TRACKING_CLEAN,
                            help="cleaned-track tree to read: v1 (default, maze/), geom2 (maze_geom2/, "
                                 "cyclic maze-2 corners) or v2 (maze_v2/ + openfield_v2/); needs --out-tag")
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        args = parser.parse_args()
        SYNC_METHOD = args.sync
        SESSIONS = args.sessions
        if args.tracking_clean != 'v1' and not args.out_tag:
            sys.exit(f"--tracking-clean {args.tracking_clean} rewrites XY/HD/Locs; pass --out-tag so "
                     "data/processed/ stays the tree the existing data_dic was built from")
        if args.tracking_clean == 'v2' and args.sync != 'matched':
            sys.exit("--tracking-clean v2 propagates track NaN in _save_matched only; pass --sync matched")
        set_output_tag(args.out_tag)
        set_tracking_clean(args.tracking_clean)

        main_metadata(animals=args.animals, ephys_folder_filter=args.ephys_folder)
    elif '--dataset' in sys.argv:
        sys.exit("refusing to run the legacy ephys-time path with --dataset; pass --metadata")
    else:
        main()
