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
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
from build_data_dic import locs_to_int                                # noqa: E402
from extract_pokes import parse_pycontrol                             # noqa: E402

PROCESSED_DATA = rr.REPO_ROOT / "data" / "processed_data"
NEURON_PATH = PROCESSED_DATA / "neuron_raw_mingyutest"
TRIALTIMES_PATH = PROCESSED_DATA / "trialtimes_raw_mingyutest"
TRACKING_PATH = rr.REPO_ROOT / "data" / "processed"

REPORT_PATH = PROCESSED_DATA / "alignment_qc.csv"

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
    mouse, date1, date2 = rr.split_recday(recday)
    tracking = TRACKING_PATH / mouse / f"{date1}_{date2}" / f"Locs_raw_{recday}_{session}.npy"
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

    # 1. location at each goal time == that task's goal node
    if locs is not None and trial_times is not None and task is not None and len(trial_times):
        hits = total = 0
        for state in range(min(trial_times.shape[1] - 1, len(task))):
            for t in trial_times[:, state]:
                if 0 <= t < len(locs):
                    total += 1
                    hits += int(locs[t] == task[state])
        if total:
            row["loc_at_goal"] = hits / total
            row["n_goal_checks"] = total

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

    return row


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
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recdays", nargs="*", default=None)
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
    report.to_csv(REPORT_PATH, index=False)
    print(f"\nwrote {len(report)} session rows -> {REPORT_PATH}")

    bad = failures(report)
    if bad:
        print(f"\n{len(bad)} failure(s):")
        for line in bad:
            print(f"  FAIL {line}")
        sys.exit(1)
    print("all alignment checks passed")


if __name__ == "__main__":
    main()
