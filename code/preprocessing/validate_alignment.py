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

Reads the arrays on disk rather than `data_dic_lec.pkl`, so a recday can be checked before
it is built into the dictionary.

Usage::

    python validate_alignment.py                       # whole cohort, per-mouse summary
    python validate_alignment.py --recdays ly05_20250618_20250619 --detail
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

# `--dataset` must reach recday_registry before it is imported (default: lec, unchanged).
if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import numpy as np
import pandas as pd

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
RSYNC_QC = PROCESSED_DATA / "ephys_rsync_qc.csv"          # extract_rsync_from_ttl.py, TTL-derived trains

_suffix = "" if (rr.DATASET == "lec" and _NEURAL == "mingyutest" and _TRACKING == "processed") else (
    f"_{rr.DATASET}" + ("" if _NEURAL == "mingyutest" else f"_n{_NEURAL}") + ("" if _TRACKING == "processed" else f"_t{_TRACKING}"))
REPORT_PATH = PROCESSED_DATA / f"alignment_qc{_suffix}.csv"

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

    # 6. speed x population-rate cross-correlation: running speed comes from the tracking (on
    #    the pyControl clock through the frame fit), population rate from the spikes (through
    #    the rsync fit). Movement-related firing puts the peak near lag 0 when both clocks are
    #    right; a neural offset shows as a displaced peak. Reported per session, not gated.
    row["speed_lag_bins"] = np.nan
    row["speed_lag_r"] = np.nan
    neuron_path = NEURON_PATH / f"Neuron_raw_{recday}_{session}.npy"
    xy_path = _tracking_dir(recday) / f"XY_raw_{recday}_{session}.npy"
    if neuron_path.exists() and xy_path.exists():
        lag, r = speed_rate_lag(np.load(neuron_path, mmap_mode="r"), np.load(xy_path))
        row["speed_lag_bins"], row["speed_lag_r"] = lag, r

    return row


def _tracking_dir(recday: str) -> Path:
    mouse, dates = rr.recday_dates(recday)
    return TRACKING_PATH / mouse / "_".join(dates)


def speed_rate_lag(neuron_raw, xy_cm: np.ndarray, max_lag: int = 80, smooth_bins: int = 4):
    """(lag in bins at the peak |cross-correlation| of speed vs population rate, r at that lag).

    Both signals z-scored and lightly smoothed (box of `smooth_bins`). Positive lag means the
    rate leads the speed. Sessions with < 2000 common bins return NaN.
    """
    pop = np.asarray(neuron_raw).sum(axis=0).astype(float)
    xy = np.asarray(xy_cm, dtype=float)
    n = min(len(pop), len(xy))
    if n < 2000:
        return np.nan, np.nan
    speed = np.hypot(*np.diff(xy[:n], axis=0).T)
    speed = np.append(speed, speed[-1])
    ok = ~np.isnan(speed)
    if ok.sum() < 2000:
        return np.nan, np.nan
    speed = np.where(ok, speed, np.nanmedian(speed))
    k = np.ones(smooth_bins) / smooth_bins
    a = np.convolve(speed, k, mode="same")
    b = np.convolve(pop[:n], k, mode="same")
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.array([np.dot(a[max(0, -l):n - max(0, l)], b[max(0, l):n - max(0, -l)]) / (n - abs(l)) for l in lags])
    i = int(np.argmax(np.abs(cc)))
    return int(lags[i]), float(cc[i])


RSYNC_MATCH_FLOOR = 0.9
REWARD_LAG_MAX = 80        # bins (2 s) either side


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


def reward_response_lag(neuron_raw, pokes: np.ndarray, max_lag: int = REWARD_LAG_MAX):
    """(peak lag in bins, peak / median of the event-triggered population rate).

    Population rate = spike counts summed over units. Averaged around every rewarded poke
    entry (`pokes[:, 3] == 1`) over lags -max_lag..+max_lag; the argmax is the lag reported.
    """
    pop = np.asarray(neuron_raw).sum(axis=0).astype(float)
    entries = pokes[pokes[:, 3] == 1, 0].astype(int)
    entries = entries[(entries >= max_lag) & (entries < len(pop) - max_lag)]
    if len(entries) < 5:
        return np.nan, np.nan
    lags = np.arange(-max_lag, max_lag + 1)
    curve = np.array([pop[entries + l].mean() for l in lags])
    return int(lags[int(np.argmax(curve))]), float(curve.max() / max(np.median(curve), 1e-9))


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
        speed_lag_median=("speed_lag_bins", "median"),
        speed_lag_r_median=("speed_lag_r", "median"),
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
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS),
                        help="which tree to check (read before import; default lec)")
    parser.add_argument("--recdays", nargs="*", default=None)
    parser.add_argument("--neural", default=_NEURAL, help="neuron_raw source: mingyutest (default) or v2")
    parser.add_argument("--tracking", default=_TRACKING, help="tracking source: processed (default) or v2")
    parser.add_argument("--detail", action="store_true", help="print every session")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    recdays = args.recdays or rr.all_recdays()
    report = check_recdays(recdays, verbose=verbose)
    if report.empty:
        print("no sessions found")
        sys.exit(1)

    if args.detail:
        print("\n" + report.round(3).to_string(index=False))
    summarise(report, verbose=verbose)
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
