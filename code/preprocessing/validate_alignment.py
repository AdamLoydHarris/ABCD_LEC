"""
Alignment Validation

Checks that a session's neural, behavioural and tracking arrays are actually on the same
clock, using observables that only line up if they are.

The point of this script is that shapes prove nothing. `Neuron_raw` is binned onto the
behavioural timeline, so `trialtimes.max() == Neuron_raw.shape[1]` holds by construction no
matter which recording day the spikes came from -- which is exactly how
`docs/BUG_ly05_recday_mismatch.md` survived undetected. These checks compare *content*
across arrays produced by independent parts of the pipeline:

1. `loc_at_goal`  -- the animal's tracked location at each goal-poke time must be that
   task's goal node. Ties `Locs_raw` (SLEAP) to `Trial_times` (pyControl) to `Task_data`.
   The sharpest check available: 1.000 on well-tracked recdays, ~0.96 at the cohort floor.
2. `poke_loc`     -- during a poke at port p, the tracked location must be node p. Ties
   `pokes_*.npy` to `Locs_raw`. Dominated by tracking quality, not alignment: it runs
   0.63-0.96 on ly06's perfectly good data and 0.88-1.00 on ly05's. **Read it within
   mouse**, against that animal's other recdays -- never against an absolute threshold.
3. `task_ok`      -- stored `Task_data` equals the pyControl file's own `active_poke`,
   with the file resolved from the recday's own dates. This is the check that identified
   the mispaired recday's behaviour as genuinely 06-18/19.
4. `bins_ok`      -- `Neuron_raw.shape[1] == trialtimes.max()`, and the tracking arrays
   cover at least that many bins. Scored sessions only: a session with no completed
   trials is deliberately binned to the end of the recording rather than to a
   behavioural window, so its array routinely outruns the tracking by design.
5. `rsync_ok`     -- the TTL pairing behind the neural clock (QC of `extract_rsync_from_ttl`).

Checks 1-5 cannot see a spike clock that is off by a constant: in v2 every one of them passed
while 42 LEC task sessions had spikes 0.94-29.2 s late (docs/handoff/SPIKE_CLOCK_V4.md). The
probes below compare the spikes themselves with behaviour, and are hard gates:

6. `speed_*`      -- lag of the population rate against running speed (tracking, frame clock),
   searched over ±90 s; must sit within ±0.5 s of the mouse's own reference, the median over
   its sessions whose recording starts at sample 0. Tested where r > 0.2 and the peak beats
   everything more than 2 s away by 20 %: a flat cross-correlation has no lag to test. The
   reference is per mouse because it differs: -0.30 s for ah10, about +0.15 s for ah08.
7. `reward_*`     -- peak of the mean |z| reward-entry PSTH (250 ms blocks, ±30 s); must lie
   in [-2, +4] s. Tested where peak / median > 2.
8. `wallclock_s`  -- pyControl start stamp minus OpenEphys Software Time minus the clock's
   own t = 0 offset; must lie within the IQR ± 3 s of the same mouse's sessions from the same
   pyControl format. The maze `.txt` and open-field `.tsv` rigs' PC clocks differ by ~3.5 s, and
   `.txt` stamps are 1 s.
9. `early_ok`     -- a session whose first bin maps before ephys sample 0 passes only if the
   wall clock confirms it.

Reads the arrays on disk rather than `data_dic_lec.pkl`, so a recday can be checked before
it is built into the dictionary. `--rsync` names the train tree that made the arrays under
test (v1 `ephys_rsync_mingyutest`, v2 `ephys_rsync`, v4 `ephys_rsync_v4`); the probes 8-9
read its clock.

Usage::

    python validate_alignment.py                       # whole cohort, per-mouse summary
    python validate_alignment.py --recdays ly05_20250618_20250619 --detail
    python validate_alignment.py --neural v4 --tracking v3 --rsync ephys_rsync_v4
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

# `--dataset` must reach recday_registry before it is imported (default: lec, unchanged).
if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
from build_data_dic import locs_to_int                                # noqa: E402
from extract_pokes import parse_pycontrol                             # noqa: E402

PROCESSED_DATA = rr.REPO_ROOT / "data" / "processed_data"
# `--neural v2` / `--tracking v2` point at the fitted-clock re-derivations (Stage F); the
# defaults are the arrays the current data_dic was built from.
_NEURAL = sys.argv[sys.argv.index("--neural") + 1] if "--neural" in sys.argv else "mingyutest"
_TRACKING = sys.argv[sys.argv.index("--tracking") + 1] if "--tracking" in sys.argv else "processed"
NEURON_PATH = PROCESSED_DATA / ("neuron_raw_mingyutest" if _NEURAL == "mingyutest" else f"neuron_raw_{_NEURAL}")
TRIALTIMES_PATH = PROCESSED_DATA / "trialtimes_raw_mingyutest"
TRACKING_PATH = rr.REPO_ROOT / "data" / ("processed" if _TRACKING == "processed" else f"processed_{_TRACKING}")
# `--rsync`: the train tree behind the arrays under test, read by the wall-clock and early-start
# probes. Its own QC (`<tree>_qc.csv`) when it has one; the legacy caches have none, and fall back
# to the TTL-derived QC, whose pairing columns are the same in v2 and v4.
_RSYNC = sys.argv[sys.argv.index("--rsync") + 1] if "--rsync" in sys.argv else "ephys_rsync"
RSYNC_TREE = PROCESSED_DATA / _RSYNC
RSYNC_QC = PROCESSED_DATA / (f"{_RSYNC}_qc.csv" if list(PROCESSED_DATA.glob(f"{_RSYNC}_qc*.csv"))
                             else "ephys_rsync_qc.csv")

_suffix = "" if (rr.DATASET == "lec" and _NEURAL == "mingyutest" and _TRACKING == "processed"
                 and _RSYNC == "ephys_rsync") else (
    f"_{rr.DATASET}" + ("" if _NEURAL == "mingyutest" else f"_n{_NEURAL}") + ("" if _TRACKING == "processed" else f"_t{_TRACKING}")
    + ("" if _RSYNC == "ephys_rsync" else f"_r{_RSYNC.replace('ephys_rsync_', '')}"))
REPORT_PATH = PROCESSED_DATA / f"alignment_qc{_suffix}.csv"

BIN_MS = 25
SAMPLES_PER_MS = 30
LONDON = ZoneInfo("Europe/London")      # pyControl stamps are local wall time; OpenEphys logs UTC ms
SPEED_MAX_LAG_S = 90.0                  # mEC's largest first-sample offset is 67 s
SPEED_R_MIN = 0.2                       # below this the speed cross-correlation is multi-peaked noise
SPEED_PEAK_RATIO_MIN = 1.2              # peak / best value > 2 s away: 12 ly05 sessions with r 0.21-0.38
                                        # sit at 1.00-1.14 with lags scattered over +-70 s, also where
                                        # v4 == v2; 95 % of the other sessions are >= 1.24
SPEED_TOL_S = 0.5
REWARD_HALF_S = 30.0
REWARD_BLOCK_BINS = 10                  # 250 ms blocks
REWARD_WINDOW_S = (-2.0, 4.0)           # ah10 v1 peaks inside [-1, 4] in 38/38; 8 mEC sessions whose speed
                                        # probe passes peak at -1.25 to -1.5 s (activity before the entry)
REWARD_PTM_MIN = 2.0
WALLCLOCK_PAD_S = 3.0

#: Threshold for `loc_at_goal`, set from the measured separation rather than by taste.
#: Correctly aligned sessions score 0.944-1.000 across the whole cohort (90% score exactly
#: 1.000; the minimum is 2 misses out of 36 checks, i.e. tracking noise in a short session).
#: Deliberately mis-paired sessions -- each session's tracking against a neighbouring
#: session's trialtimes -- score mean 0.096, max 0.250. The two regimes do not overlap, so
#: the threshold sits in the gap: it separates misalignment from noise, and is not a
#: tracking-quality standard.
LOC_AT_GOAL_FLOOR = 0.60


def _session_arrays(recday: str, session: int):
    """(locs, trial_times, task, pokes, n_bins) for one session; None where absent."""
    mouse, dates = rr.recday_dates(recday)
    tracking = TRACKING_PATH / mouse / "_".join(dates) / f"Locs_raw_{recday}_{session}.npy"
    locs = locs_to_int(np.load(tracking, allow_pickle=True)) if tracking.exists() else None

    tt_path = TRIALTIMES_PATH / f"trialtimes_{recday}_{session}.npy"
    task_path = TRIALTIMES_PATH / f"Task_data_{recday}_{session}.npy"
    poke_path = TRIALTIMES_PATH / f"pokes_{recday}_{session}.npy"
    neuron_path = NEURON_PATH / f"Neuron_raw_{recday}_{session}.npy"

    trial_times = np.load(tt_path).astype(int) if tt_path.exists() else None
    task = np.load(task_path, allow_pickle=True) if task_path.exists() else None
    pokes = np.load(poke_path) if poke_path.exists() else None
    n_bins = np.load(neuron_path, mmap_mode="r").shape[1] if neuron_path.exists() else None
    return locs, trial_times, task, pokes, n_bins


def check_session(recday: str, session: int) -> dict:
    """Run every check for one session. NaN where the inputs for a check are absent."""
    locs, trial_times, task, pokes, n_bins = _session_arrays(recday, session)
    row = {"recday": recday, "mouse": recday[:4], "session": session,
           "n_trials": None if trial_times is None else len(trial_times),
           "n_bins": n_bins, "n_locs": None if locs is None else len(locs),
           "loc_at_goal": np.nan, "n_goal_checks": 0,
           "poke_loc": np.nan, "n_poke_bins": 0,
           "task_ok": None, "bins_ok": None, "notes": ""}

    # 1. location at each goal time == that task's goal node. Untracked bins (Locs 0: SLEAP
    #    gap, or a frame-clock gap hiding dropped frames, frame_clock.uncertain_gaps) are
    #    counted separately, not as misses: they say nothing about alignment.
    if locs is not None and trial_times is not None and task is not None and len(trial_times):
        hits = total = untracked = 0
        for state in range(min(trial_times.shape[1] - 1, len(task))):
            for t in trial_times[:, state]:
                if 0 <= t < len(locs):
                    if locs[t] == 0:
                        untracked += 1
                        continue
                    total += 1
                    hits += int(locs[t] == task[state])
        if total:
            row["loc_at_goal"] = hits / total
            row["n_goal_checks"] = total
        row["n_goal_untracked"] = untracked

    # 2. location during each poke == that poke's port
    if locs is not None and pokes is not None and len(pokes):
        hits = total = 0
        for entry, exit_, port, _rewarded, _state in pokes:
            lo, hi = min(int(entry), len(locs) - 1), min(int(exit_), len(locs) - 1)
            if hi < lo:
                continue
            segment = locs[lo:hi + 1]
            hits += int((segment == port).sum())
            total += len(segment)
        if total:
            row["poke_loc"] = hits / total
            row["n_poke_bins"] = total

    # 3. stored task == the day's own pyControl active_poke
    if task is not None:
        try:
            times, active = parse_pycontrol(rr.pycontrol_file(recday, session))
            row["task_ok"] = (active is not None
                              and tuple(int(x) for x in np.asarray(task).ravel())
                              == tuple(active))
        except (KeyError, FileNotFoundError, AssertionError) as exc:
            row["notes"] += f"pycontrol unresolved ({exc}); "

    # 4. length coherence.
    # Only meaningful for scored sessions. A session with no completed trials is
    # deliberately binned to the end of the recording rather than to a behavioural
    # window, so its array routinely outruns the tracking; that is the intended
    # geometry, not a defect, and every valid-session filter drops these anyway.
    if n_bins is not None and trial_times is not None and len(trial_times):
        row["bins_ok"] = bool(n_bins == int(trial_times.max())
                              and (locs is None or len(locs) >= n_bins))
    elif n_bins is not None:
        row["notes"] += "no trials: array spans the recording, not the trial window; "

    # 5. rsync pairing -- the neural clock's anchor. From the TTL-derived QC: how many of this
    #    mouse's pyControl pulses were found in the recording's TTL, and whether pulse 0 was.
    #    The legacy caches could not report this; for ah08/ly06 their first pulse was the
    #    partner box's (-64 ms) and for ly05 06-20/23 an unrelated train, invisible to 1-4.
    qc = _rsync_qc_row(recday, session)
    if qc is not None:
        # Denominator: this mouse's pulses that the recording could contain. ah10 2025-06-23
        # 17-35-04 (s7) was stopped after 50 of pyControl's 91 pulses; all 50 TTLs matched, so
        # the clock is sound although only 55 % of the pyControl train is in the recording.
        n_pc, n_ttl = int(qc["n_pycontrol"]), int(qc.get("n_ttl", qc["n_pycontrol"]))
        denom = n_pc
        if n_ttl < n_pc and qc["n_matched"] >= RSYNC_MATCH_FLOOR * n_ttl:
            denom = n_ttl
            row["notes"] += f"recording holds {n_ttl} of {n_pc} pyControl pulses (stopped early); "
        row["rsync_matched_frac"] = qc["n_matched"] / max(denom, 1)
        row["rsync_first_ok"] = bool(qc["first_pc_rsync_matched"])
        row["rsync_ok"] = row["rsync_matched_frac"] >= RSYNC_MATCH_FLOOR and row["rsync_first_ok"]
    else:
        row["rsync_matched_frac"] = np.nan
        row["rsync_first_ok"] = None
        row["rsync_ok"] = None

    # 6-7. behaviour-locked probes on the spikes themselves (module docstring). Gated per mouse
    #      afterwards by `probe_gates`, which needs every session of the mouse.
    row.update(ap_first_sample=np.nan if qc is None else qc.get("ap_first_sample", np.nan),
               speed_lag_s=np.nan, speed_r=np.nan, speed_peak_ratio=np.nan, reward_peak_s=np.nan, reward_ptm=np.nan,
               n_rewards=0, wallclock_s=np.nan, t0_dat_s=np.nan)
    neuron_path = NEURON_PATH / f"Neuron_raw_{recday}_{session}.npy"
    xy_path = _tracking_dir(recday) / f"XY_raw_{recday}_{session}.npy"
    if neuron_path.exists():
        neural = np.load(neuron_path)
        if xy_path.exists():
            row["speed_lag_s"], row["speed_r"], row["speed_peak_ratio"] = speed_probe(neural, np.load(xy_path))
        if pokes is not None:
            row["reward_peak_s"], row["reward_ptm"], row["n_rewards"] = reward_probe(neural, pokes)

    # 8-9. wall clock and early start, on the clock of the `--rsync` tree
    if qc is not None:
        try:
            pc_path = rr.pycontrol_file(recday, session)
            row["pycontrol_suffix"] = pc_path.suffix
            times, _ = parse_pycontrol(pc_path)
            pc_zero = _pc_zero_samples(recday, qc["recording"], times)
            sync = _sync_messages(recday, qc["recording"])
            pc_start = pycontrol_start(pc_path)
            if pc_zero is not None:
                a_on = times.get("A_on", np.array([], dtype=np.int64))
                t0 = ((int(a_on[0]) // BIN_MS) * BIN_MS if len(a_on) and trial_times is not None
                      and len(trial_times) else int(times["rsync"][0]))
                row["t0_dat_s"] = (pc_zero + SAMPLES_PER_MS * t0) / (1000 * SAMPLES_PER_MS)
                if sync is not None and pc_start is not None:
                    row["wallclock_s"] = wallclock_discrepancy_s(pc_start, oe_start(sync), pc_zero)
        except (KeyError, FileNotFoundError, AssertionError, ValueError) as exc:
            row["notes"] += f"wall clock unresolved ({exc}); "

    return row


def _tracking_dir(recday: str) -> Path:
    mouse, dates = rr.recday_dates(recday)
    return TRACKING_PATH / mouse / "_".join(dates)


def _z(x: np.ndarray) -> np.ndarray | None:
    s = x.std()
    return None if not np.isfinite(s) or s == 0 else (x - x.mean()) / s


def speed_probe(neuron_raw, xy_cm, max_lag_s: float = SPEED_MAX_LAG_S) -> tuple[float, float, float]:
    """(lag in s maximising corr(speed[t], pop[t + lag]), r there, r / best r more than 2 s away);
    negative lag = population leads.

    neuron_raw (units, bins), xy_cm (bins, >= 2). Population rate sigma 10 bins; speed from XY
    with NaNs interpolated, sigma 3 bins, then the gradient (the handoff probe exactly).
    """
    pop = gaussian_filter1d(np.asarray(neuron_raw).sum(axis=0).astype(float), 10)
    xy = np.asarray(xy_cm, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return np.nan, np.nan, np.nan
    xy = xy[:, :2].copy()
    for k in range(2):
        ok = np.isfinite(xy[:, k])
        if ok.sum() < 2000:
            return np.nan, np.nan, np.nan
        xy[:, k] = np.interp(np.arange(len(xy)), np.flatnonzero(ok), xy[ok, k])
    speed = np.hypot(*np.gradient(gaussian_filter1d(xy, 3, axis=0), axis=0).T)
    n = min(len(pop), len(speed))
    lead = int(np.argmax(pop[:n] > 0))      # a late ephys start leaves leading empty bins
    a, b = speed[lead + 40:n], pop[lead + 40:n]
    m = min(int(round(max_lag_s * 1000 / BIN_MS)), len(a) // 3)
    a, b = _z(a), _z(b)
    if a is None or b is None or m < 40:
        return np.nan, np.nan, np.nan
    c = fftconvolve(b, a[::-1], "full") / len(a)
    seg = c[len(a) - 1 - m:len(a) + m]
    k = int(np.argmax(seg))
    far = np.abs(np.arange(len(seg)) - k) > 2000 // BIN_MS
    ratio = float(seg[k] / seg[far].max()) if far.any() and seg[far].max() > 0 else np.inf
    return (k - m) * BIN_MS / 1000, float(seg[k]), ratio


def reward_probe(neuron_raw, pokes: np.ndarray, half_s: float = REWARD_HALF_S,
                 block_bins: int = REWARD_BLOCK_BINS) -> tuple[float, float, int]:
    """(peak lag in s, peak / median, n rewards) of the mean |z| across units of the PSTH
    around rewarded poke entries (`pokes[:, 3] == 1`, entry bin in column 0), in blocks of
    `block_bins` starting at entry + lag. Each unit is z-scored across lags before |.|."""
    counts = np.asarray(neuron_raw, dtype=np.float64)
    h = int(round(half_s * 1000 / (BIN_MS * block_bins)))
    entries = np.asarray(pokes)[np.asarray(pokes)[:, 3] == 1, 0].astype(np.int64) if len(pokes) else np.array([], int)
    entries = entries[(entries - h * block_bins >= 0) & (entries + (h + 1) * block_bins <= counts.shape[1])]
    if len(entries) < 5:
        return np.nan, np.nan, int(len(entries))
    cs = np.concatenate([np.zeros((counts.shape[0], 1)), np.cumsum(counts, axis=1)], axis=1)
    lags = np.arange(-h, h + 1)
    starts = entries[:, None] + lags[None, :] * block_bins
    psth = (cs[:, starts + block_bins] - cs[:, starts]).mean(axis=1)          # (units, lags)
    sd = psth.std(axis=1)
    if not (sd > 0).any():
        return np.nan, np.nan, int(len(entries))
    z = (psth[sd > 0] - psth[sd > 0].mean(axis=1, keepdims=True)) / sd[sd > 0, None]
    curve = np.abs(z).mean(axis=0)
    k = int(np.argmax(curve))
    return float(lags[k] * block_bins * BIN_MS / 1000), float(curve[k] / np.median(curve)), int(len(entries))


def pycontrol_start(path) -> datetime | None:
    """Aware start time of a pyControl file: `.txt` 'I Start date' (1 s), `.tsv` start_time (ms)."""
    path = Path(path)
    with open(path) as f:
        for i, line in enumerate(f):
            if path.suffix == ".txt" and line.startswith("I Start date"):
                t = datetime.strptime(line.split(":", 1)[1].strip(), "%Y/%m/%d %H:%M:%S")
                return t.replace(tzinfo=LONDON)
            if path.suffix == ".tsv" and "\tstart_time\t" in line:
                # UTC: read as London time, all 8 April mEC .tsv sessions sit at -3599.4 s and March ones do not
                return datetime.fromisoformat(line.rstrip("\n").split("\t")[-1]).replace(tzinfo=timezone.utc)
            if i > 50:                       # the header is the first ~10 lines
                break
    return None


def oe_start(sync_messages: Path) -> datetime:
    """Aware UTC time the recording started, from `sync_messages.txt` 'Software Time' (ms)."""
    for line in Path(sync_messages).read_text().splitlines():
        if line.startswith("Software Time"):
            return datetime.fromtimestamp(int(line.rsplit(":", 1)[1].strip()) / 1000, tz=timezone.utc)
    raise ValueError(f"{sync_messages}: no Software Time line")


def wallclock_discrepancy_s(pc_start: datetime, oe_t0: datetime, pc_zero_samples: float) -> float:
    """pyControl start minus OpenEphys start minus the clock's own pyControl t = 0 offset, in s.
    0 when the clock is right; v2's error shows as +(sn0 - origin) / 30 kHz."""
    return (pc_start - oe_t0).total_seconds() - pc_zero_samples / (1000 * SAMPLES_PER_MS)


@lru_cache(maxsize=None)
def _concat_manifest(recday: str) -> pd.DataFrame:
    return pd.read_csv(rr.kilosort_dir(recday).parent / "recording_sessions_in_concat.csv")


def _sync_messages(recday: str, tag: str) -> Path | None:
    """`sync_messages.txt` of the recording named `YYYY-MM-DD_HH-MM-SS`, via the concat manifest."""
    c = _concat_manifest(recday)
    hit = c[(c["date"].astype(str) == tag[:10]) & (c["time"].astype(str) == tag[11:])]
    if not len(hit):
        return None
    p = Path(hit.iloc[0]["sample_numbers_path"]).parents[2] / "sync_messages.txt"
    return p if p.exists() else None


def _pc_zero_samples(recday: str, tag: str, times: dict) -> float | None:
    """`.dat` index of pyControl t = 0 on the `--rsync` tree's train: train[0] - 30 * rsync[0].
    For a legacy cache that is `session_window`'s pc_zero; for a fitted tree it is b to < 1 ms."""
    mouse, _ = rr.recday_dates(recday)
    p = RSYNC_TREE / mouse / f"rsync_timestamps_{tag}.npy"
    rsync = times.get("rsync", np.array([]))
    if not p.exists() or not len(rsync):
        return None
    train = np.load(p)
    return float(train[0]) - SAMPLES_PER_MS * float(rsync[0]) if len(train) else None


RSYNC_MATCH_FLOOR = 0.9


def _rsync_qc_table() -> pd.DataFrame | None:
    global _RSYNC_QC_CACHE
    try:
        return _RSYNC_QC_CACHE
    except NameError:
        pass
    # Every QC file the rsync extractor wrote (partial re-runs add .vN files); the latest row
    # per recording wins.
    cands = sorted(RSYNC_QC.parent.glob(RSYNC_QC.stem + "*.csv"), key=lambda p: p.stat().st_mtime)
    if not cands:
        _RSYNC_QC_CACHE = None
    else:
        table = pd.concat([pd.read_csv(p) for p in cands], ignore_index=True)
        _RSYNC_QC_CACHE = table.drop_duplicates(subset=["mouse", "recording"], keep="last")
    return _RSYNC_QC_CACHE


def _rsync_qc_row(recday: str, session: int) -> dict | None:
    table = _rsync_qc_table()
    if table is None:
        return None
    rows = rr.task_session_rows(recday)
    if session >= len(rows):
        return None
    r = rows.iloc[session]
    d = str(r["_date"])
    tag = f"{d[:4]}-{d[4:6]}-{d[6:]}_{r['Ephys']}"
    mouse, _ = rr.recday_dates(recday)
    hit = table[(table["mouse"] == mouse) & (table["recording"] == tag) & (table["kind"] == "task")]
    return hit.iloc[0].to_dict() if len(hit) else None


def _speed_testable(df: pd.DataFrame) -> pd.Series:
    """A session's speed lag means something: r above the floor and a distinct peak."""
    ratio = df["speed_peak_ratio"] if "speed_peak_ratio" in df else pd.Series(np.inf, index=df.index)
    return (df["speed_r"] > SPEED_R_MIN) & (ratio >= SPEED_PEAK_RATIO_MIN)


def probe_gates(report: pd.DataFrame) -> pd.DataFrame:
    """Per-mouse references and pass flags for probes 6-9; adds *_ref and *_ok columns.

    References come from the mouse's sessions whose recording starts at sample 0, where every
    clock version agrees, so a shifted version cannot drag its own reference along.
    """
    report = report.copy()
    for col in ("speed_ref_s", "wallclock_lo_s", "wallclock_hi_s"):
        report[col] = np.nan
    report["ref_source"] = ""
    if "pycontrol_suffix" not in report:
        report["pycontrol_suffix"] = ""
    report["pycontrol_suffix"] = report["pycontrol_suffix"].fillna("")
    clean_all = report["ap_first_sample"] == 0
    cohort_sp = report[clean_all & _speed_testable(report)]["speed_lag_s"].dropna()
    for mouse, g in report.groupby("mouse"):
        clean = g["ap_first_sample"] == 0
        sp = g[clean & _speed_testable(g)]["speed_lag_s"].dropna()
        # Too few clean sessions falls back to the cohort's clean sessions, never to this mouse's
        # suspect ones: on a 2-recday subset that fallback put ah10's reference at +4.55 s.
        src = "speed mouse" if len(sp) >= 3 else "speed cohort"
        sp = sp if len(sp) >= 3 else cohort_sp
        idx = report["mouse"] == mouse
        report.loc[idx, "ref_source"] = src
        if len(sp):
            report.loc[idx, "speed_ref_s"] = float(sp.median())
    # wall-clock band per mouse and pyControl format (rig), with the same fallback rule
    for (mouse, suffix), g in report.groupby(["mouse", "pycontrol_suffix"]):
        wc = g[g["ap_first_sample"] == 0]["wallclock_s"].dropna()
        src = "wall mouse"
        if len(wc) < 4:
            wc = report[clean_all & (report["pycontrol_suffix"] == suffix)]["wallclock_s"].dropna()
            src = "wall cohort"
        idx = (report["mouse"] == mouse) & (report["pycontrol_suffix"] == suffix)
        report.loc[idx, "ref_source"] = report.loc[idx, "ref_source"] + f", {src}{suffix}"
        if len(wc):
            q1, q3 = np.percentile(wc, [25, 75])
            report.loc[idx, "wallclock_lo_s"] = q1 - WALLCLOCK_PAD_S
            report.loc[idx, "wallclock_hi_s"] = q3 + WALLCLOCK_PAD_S
    speed_tested = _speed_testable(report) & report["speed_ref_s"].notna()
    report["speed_ok"] = np.where(speed_tested, (report["speed_lag_s"] - report["speed_ref_s"]).abs()
                                  <= SPEED_TOL_S, None)
    reward_tested = report["reward_ptm"] > REWARD_PTM_MIN
    report["reward_ok"] = np.where(reward_tested, report["reward_peak_s"].between(*REWARD_WINDOW_S), None)
    wall_tested = report["wallclock_s"].notna() & report["wallclock_lo_s"].notna()
    report["wallclock_ok"] = np.where(wall_tested, report["wallclock_s"].between(
        report["wallclock_lo_s"], report["wallclock_hi_s"]), None)
    early = report["t0_dat_s"] < 0
    report["early_ok"] = np.where(early, report["wallclock_ok"] == True, None)    # noqa: E712
    return report


def check_recdays(recdays: List[str], verbose: bool = True) -> pd.DataFrame:
    rows = []
    for recday in recdays:
        for session in range(12):
            locs, trial_times, task, pokes, n_bins = _session_arrays(recday, session)
            if all(x is None for x in (locs, trial_times, task, pokes, n_bins)):
                continue
            rows.append(check_session(recday, session))
        if verbose:
            print(f"  {recday}: {sum(r['recday'] == recday for r in rows)} sessions")
    return pd.DataFrame(rows)


def summarise(report: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Per-recday summary, with each recday's `poke_loc` placed inside its mouse's range."""
    def verdict(series):
        """'ok' / 'FAIL' / '-' -- never report a vacuous pass as a pass.

        A check that had no inputs to run on is not a check that succeeded; conflating
        the two is how a silent gap becomes a green tick.
        """
        seen = series.dropna()
        if not len(seen):
            return "-"
        return "ok" if seen.all() else "FAIL"

    grouped = report.groupby(["mouse", "recday"], as_index=False).agg(
        sessions=("session", "count"),
        loc_at_goal_min=("loc_at_goal", "min"),
        loc_at_goal_mean=("loc_at_goal", "mean"),
        poke_loc_min=("poke_loc", "min"),
        poke_loc_mean=("poke_loc", "mean"),
        task_ok=("task_ok", verdict),
        bins_ok=("bins_ok", verdict),
        rsync_ok=("rsync_ok", verdict),
        rsync_matched_min=("rsync_matched_frac", "min"),
        speed_ok=("speed_ok", verdict),
        speed_lag_median=("speed_lag_s", "median"),
        reward_ok=("reward_ok", verdict),
        reward_peak_median=("reward_peak_s", "median"),
        wallclock_ok=("wallclock_ok", verdict),
        wallclock_median=("wallclock_s", "median"),
        early_ok=("early_ok", verdict),
    )
    # poke_loc is a tracking-quality measure, so judge it against the same mouse only.
    per_mouse = grouped.groupby("mouse")["poke_loc_mean"]
    grouped["poke_loc_vs_mouse"] = (grouped["poke_loc_mean"] - per_mouse.transform("median"))
    if verbose:
        print("\n" + grouped.round(3).to_string(index=False))
    return grouped


def failures(report: pd.DataFrame) -> List[str]:
    """Hard failures only -- things that cannot be explained by tracking quality."""
    bad = []
    low = report[report["loc_at_goal"].notna()
                 & (report["loc_at_goal"] < LOC_AT_GOAL_FLOOR)]
    for _, r in low.iterrows():
        bad.append(f"{r['recday']} s{r['session']}: loc_at_goal={r['loc_at_goal']:.3f} "
                   f"below the {LOC_AT_GOAL_FLOOR} cohort floor")
    for _, r in report[report["task_ok"] == False].iterrows():        # noqa: E712
        bad.append(f"{r['recday']} s{r['session']}: Task_data != pyControl active_poke")
    for _, r in report[report["bins_ok"] == False].iterrows():        # noqa: E712
        bad.append(f"{r['recday']} s{r['session']}: Neuron_raw bins disagree with "
                   f"trialtimes / tracking length")
    if "rsync_ok" in report:
        for _, r in report[report["rsync_ok"] == False].iterrows():   # noqa: E712
            bad.append(f"{r['recday']} s{r['session']}: rsync pairing weak or first pulse unmatched "
                       f"(matched {r['rsync_matched_frac']:.2f}, first {r['rsync_first_ok']})")
    if "speed_ok" in report:
        for _, r in report[report["speed_ok"] == False].iterrows():   # noqa: E712
            bad.append(f"{r['recday']} s{r['session']}: speed lag {r['speed_lag_s']:+.2f} s vs the mouse "
                       f"reference {r['speed_ref_s']:+.2f} s (r {r['speed_r']:.2f}, peak ratio {r['speed_peak_ratio']:.2f})")
        for _, r in report[report["reward_ok"] == False].iterrows():  # noqa: E712
            bad.append(f"{r['recday']} s{r['session']}: reward PSTH peak {r['reward_peak_s']:+.2f} s outside "
                       f"{list(REWARD_WINDOW_S)} (peak/median {r['reward_ptm']:.1f}, {r['n_rewards']} rewards)")
        for _, r in report[report["wallclock_ok"] == False].iterrows():   # noqa: E712
            bad.append(f"{r['recday']} s{r['session']}: wall clock {r['wallclock_s']:+.2f} s outside the mouse "
                       f"band [{r['wallclock_lo_s']:+.2f}, {r['wallclock_hi_s']:+.2f}] s")
        for _, r in report[report["early_ok"] == False].iterrows():   # noqa: E712
            bad.append(f"{r['recday']} s{r['session']}: first bin maps {-r['t0_dat_s']:.2f} s before ephys "
                       f"sample 0 and the wall clock does not confirm it")
    return bad


def probe_table(report: pd.DataFrame) -> pd.DataFrame:
    """Per mouse: sessions tested and passed for each probe, and the references."""
    rows = []
    for mouse, g in report.groupby("mouse"):
        row = {"mouse": mouse, "sessions": len(g), "speed_ref_s": g["speed_ref_s"].iloc[0],
               "wallclock_median_s": g.loc[g["ap_first_sample"] == 0, "wallclock_s"].median(),
               "ref_source": g["ref_source"].iloc[0]}
        for probe in ("speed", "reward", "wallclock", "early"):
            seen = g[f"{probe}_ok"].dropna()
            row[f"{probe}_tested"] = len(seen)
            row[f"{probe}_passed"] = int(seen.astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS),
                        help="which tree to check (read before import; default lec)")
    parser.add_argument("--recdays", nargs="*", default=None)
    parser.add_argument("--neural", default=_NEURAL, help="neuron_raw source: mingyutest (default) or v2")
    parser.add_argument("--tracking", default=_TRACKING, help="tracking source: processed (default) or v2")
    parser.add_argument("--rsync", default=_RSYNC,
                        help="train tree behind the arrays (read before import): ephys_rsync (default), "
                             "ephys_rsync_v4, ephys_rsync_mingyutest")
    parser.add_argument("--detail", action="store_true", help="print every session")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    recdays = args.recdays or rr.all_recdays()
    report = check_recdays(recdays, verbose=verbose)
    if report.empty:
        print("no sessions found")
        sys.exit(1)
    report = probe_gates(report)

    if args.detail:
        print("\n" + report.round(3).to_string(index=False))
    summarise(report, verbose=verbose)
    table = probe_table(report)
    print(f"\nbehaviour probes [neural={_NEURAL}, tracking={_TRACKING}, rsync={_RSYNC}]: tested / passed")
    print(table.round(2).to_string(index=False))
    # The gates are relative to each mouse's clean sessions, so a constant offset shared by all of
    # them would pass; these absolute bounds (ah10 -0.30 s, ah08 +0.35 s; wall clock -0.3..-0.7 s
    # per mouse in the handoff) flag it. Notes, not failures: the PCs' clocks are not ours.
    for _, r in table.iterrows():
        if pd.notna(r["speed_ref_s"]) and abs(r["speed_ref_s"]) > 1.0:
            print(f"  note: {r['mouse']} speed reference {r['speed_ref_s']:+.2f} s is beyond ±1 s")
        if pd.notna(r["wallclock_median_s"]) and abs(r["wallclock_median_s"]) > 2.0:
            print(f"  note: {r['mouse']} clean-session wall clock median {r['wallclock_median_s']:+.2f} s is beyond ±2 s")
    refs = table["speed_ref_s"].dropna()
    for _, r in table.iterrows():
        if pd.notna(r["speed_ref_s"]) and len(refs) > 1 and abs(r["speed_ref_s"] - refs.median()) > SPEED_TOL_S:
            print(f"  note: {r['mouse']} speed reference {r['speed_ref_s']:+.2f} s sits more than "
                  f"{SPEED_TOL_S} s from the cohort median {refs.median():+.2f} s")
    # never overwrite an earlier report (user rule): version it
    out = REPORT_PATH
    n = 0
    while (out := REPORT_PATH.with_name(REPORT_PATH.stem + (f".v{n}" if n else "") + ".csv")).exists():
        n += 1
    report.to_csv(out, index=False)
    print(f"\nwrote {len(report)} session rows -> {out}")

    bad = failures(report)
    if bad:
        print(f"\n{len(bad)} failure(s):")
        for line in bad:
            print(f"  FAIL {line}")
        sys.exit(1)
    print("all alignment checks passed")


if __name__ == "__main__":
    main()
