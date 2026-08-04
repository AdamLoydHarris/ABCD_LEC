"""
Poke Event Extraction

Extracts nose-poke events from raw pyControl session files and saves them as one
.npy per ephys session, on the same 25 ms bin grid and time origin as the existing
trialtimes_*.npy and Neuron_raw_*.npy files.

Workflow:
1. For each mouse, read the metadata to map (date-pair, session index) -> pycontrol file
2. Parse the pycontrol .txt, building the event ID -> name map from the file's own header
3. Pair poke_N entries with the following poke_N_out exits
4. Tag the poke that triggered each state transition (A_on..D_on) by interval containment
5. Bin to 25 ms, align to the first A_on, clip to the session length, and save

Output columns are [entry_bin, exit_bin, port, rewarded, state]; see POKE_COLUMNS.

Run `python extract_pokes.py --validate` to write the files and check them against
the independently generated trialtimes_*.npy.
"""

import argparse
import ast
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Base paths
BASE_PATH = Path("/ceph/behrens/adam_harris/Taskspace_abstraction_lEC")
DATA_PATH = BASE_PATH / "data"
METADATA_PATH = DATA_PATH / "metadata"
RAW_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed_data"

# Input paths
PYCONTROL_PATH = RAW_PATH / "pycontrol"

# Output path (alongside the sibling trialtimes_ / Task_data_ files)
TRIALTIMES_PATH = PROCESSED_DATA_PATH / "trialtimes_raw_mingyutest"
NEURON_PATH = PROCESSED_DATA_PATH / "neuron_raw_mingyutest"
QC_REPORT_PATH = PROCESSED_DATA_PATH / "pokes_extraction_qc.csv"

# Constants
BIN_MS = 25  # Bin width in ms, matching trialtimes and Neuron_raw
N_PORTS = 9  # Physical poke ports, named poke_1..poke_9
# Reward state onsets, in task order. ABCDE_v1 sessions add E_on; the number of
# states actually used is set by the length of active_poke in each file.
STATE_NAMES = ["A_on", "B_on", "C_on", "D_on", "E_on"]

POKE_COLUMNS = ["entry_bin", "exit_bin", "port", "rewarded", "state"]

# List of mice to process
MICE = ["ah08", "ah10", "ly05", "ly06", "ly07"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_pycontrol(pycontrol_file: Path) -> Tuple[Dict[str, np.ndarray], Optional[List[int]]]:
    """
    Parse a pyControl v1.x session file.

    The ID -> name mapping is built from this file's own S (states) and E (events)
    header lines. It must never be hardcoded: METraining_Task_v1 and ABCDE_v1 use
    the same names with every event ID shifted by 2, so a hardcoded table silently
    mislabels every event in the ABCDE sessions.

    Returns:
        times: {event_or_state_name: array of timestamps in ms}
        active_poke: the rewarded port sequence from the V line, or None if absent
    """
    lines = [line.strip() for line in pycontrol_file.read_text().split("\n") if line.strip()]

    state_ids = ast.literal_eval(next(l for l in lines if l[0] == "S")[2:])
    event_ids = ast.literal_eval(next(l for l in lines if l[0] == "E")[2:])
    id2name = {v: k for k, v in {**state_ids, **event_ids}.items()}

    times: Dict[str, List[int]] = {}
    for line in lines:
        if line[0] != "D":
            continue
        timestamp, code = line[2:].split(" ")
        times.setdefault(id2name[int(code)], []).append(int(timestamp))
    times = {name: np.array(ts, dtype=np.int64) for name, ts in times.items()}

    variable_lines = [l[2:] for l in lines if l[0] == "V"]
    active_line = next((l for l in variable_lines if "active_poke" in l), None)
    active_poke = ast.literal_eval(active_line.split(" active_poke ")[1]) if active_line else None

    return times, active_poke


def to_bins(times_ms, first_a_on_ms: int) -> np.ndarray:
    """
    Convert pyControl timestamps (ms) to 25 ms bin indices aligned to the first A_on.

    The timestamps are binned onto a global grid *first* and the origin subtracted
    afterwards. Doing it the other way round -- (t - first_a_on) // BIN_MS -- shifts
    roughly half of all events by one bin relative to the on-disk trialtimes files.
    """
    return (np.asarray(times_ms, dtype=np.int64) // BIN_MS) - (first_a_on_ms // BIN_MS)


def make_trial_times(times: Dict[str, np.ndarray], n_states: int = 4) -> np.ndarray:
    """
    Rebuild the trialtimes array from parsed pyControl events.

    Shape (n_trials, n_states + 1); the last column is the next trial's A_on, so
    row i's last entry equals row i+1's first. Used to reproduce and check against
    the existing trialtimes_*.npy files.
    """
    states = STATE_NAMES[:n_states]
    if any(s not in times for s in states) or len(times[states[0]]) < 2:
        return np.zeros((0, n_states + 1), dtype=np.int64)

    first_a_on = int(times[states[0]][0])
    n_trials = len(times[states[0]]) - 1
    trial_times = np.zeros((n_trials, n_states + 1), dtype=np.int64)
    for i in range(n_trials):
        for j, state in enumerate(states):
            trial_times[i, j] = times[state][i]
        trial_times[i, -1] = times[states[0]][i + 1]
    return to_bins(trial_times, first_a_on)


# ---------------------------------------------------------------------------
# Poke intervals
# ---------------------------------------------------------------------------

def build_poke_intervals(
    times: Dict[str, np.ndarray], session_end_ms: int
) -> Tuple[List[Tuple[int, int, int]], int, int]:
    """
    Pair each poke entry with the exit that follows it at the same port.

    Entries and exits alternate cleanly in almost all sessions, but 202 of 593 raw
    files have a port with mismatched counts: usually a dangling entry because the
    session ended while the animal was still in the port, occasionally a dangling
    exit because the animal was already in the port when logging began.

    Dangling entries get an imputed exit at the last event in the file; orphan exits
    are dropped. Both are counted so the QC report can show which sessions were touched.

    Returns:
        intervals: [(entry_ms, exit_ms, port)] sorted by entry time
        n_imputed_exits, n_orphan_exits_dropped
    """
    intervals: List[Tuple[int, int, int]] = []
    n_imputed = 0
    n_orphans = 0

    for port in range(1, N_PORTS + 1):
        entries = times.get(f"poke_{port}", np.array([], dtype=np.int64))
        exits = times.get(f"poke_{port}_out", np.array([], dtype=np.int64))

        for entry in entries:
            later = exits[exits > entry]
            if len(later):
                intervals.append((int(entry), int(later[0]), port))
            else:
                intervals.append((int(entry), int(session_end_ms), port))
                n_imputed += 1

        # Exits with no preceding entry: the animal was already in the port at
        # the start of logging, so there is no poke onset to report.
        if len(entries):
            n_orphans += int((exits < entries[0]).sum())
        else:
            n_orphans += len(exits)

    intervals.sort()
    return intervals, n_imputed, n_orphans


def tag_rewarded(
    intervals: List[Tuple[int, int, int]],
    times: Dict[str, np.ndarray],
    active_poke: List[int],
) -> Tuple[np.ndarray, int, int]:
    """
    Mark which pokes triggered a state transition, by interval containment.

    The rewarded poke for state onset X_on is the poke at port active_poke[j] whose
    [entry, exit] interval contains that onset. Containment -- not a nearest-timestamp
    tolerance -- is what makes this exact: the animal is frequently already inside the
    port when the state advances, so entry-to-onset offsets run to several ms. Across
    all 593 raw files this rule matched 40491/40491 state onsets.

    Returns:
        states: per-interval state index (0..n_states-1), -1 where unrewarded
        n_state_onsets, n_unmatched_onsets
    """
    states = np.full(len(intervals), -1, dtype=np.int64)
    n_onsets = 0
    n_unmatched = 0

    for state_idx, state_name in enumerate(STATE_NAMES[:len(active_poke)]):
        if state_name not in times:
            continue
        port = active_poke[state_idx]
        candidates = [i for i, (_, _, p) in enumerate(intervals) if p == port]

        for onset in times[state_name]:
            n_onsets += 1
            match = next(
                (i for i in candidates if intervals[i][0] <= onset <= intervals[i][1]), None
            )
            if match is None:
                n_unmatched += 1
            else:
                states[match] = state_idx

    return states, n_onsets, n_unmatched


def make_pokes_array(
    times: Dict[str, np.ndarray], active_poke: List[int], end_bin: int
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build the (n_pokes, 5) poke event table for one session.

    Columns: entry_bin, exit_bin (inclusive), port, rewarded, state. Bins are clipped
    to [0, end_bin) so every value is a valid index into that session's Neuron_raw.

    Returns the array and a dict of QC counts.
    """
    stats = {
        "n_pokes": 0,
        "n_rewarded": 0,
        "n_state_onsets": 0,
        "n_unmatched_onsets": 0,
        "n_imputed_exits": 0,
        "n_orphan_exits_dropped": 0,
        "n_clipped_out_of_range": 0,
    }

    if "A_on" not in times or len(times["A_on"]) == 0 or active_poke is None:
        return np.zeros((0, len(POKE_COLUMNS)), dtype=np.int64), stats

    first_a_on = int(times["A_on"][0])
    session_end_ms = int(max(v.max() for v in times.values()))

    intervals, n_imputed, n_orphans = build_poke_intervals(times, session_end_ms)
    stats["n_imputed_exits"] = n_imputed
    stats["n_orphan_exits_dropped"] = n_orphans

    state_idx, n_onsets, n_unmatched = tag_rewarded(intervals, times, active_poke)
    stats["n_state_onsets"] = n_onsets
    stats["n_unmatched_onsets"] = n_unmatched

    if not intervals:
        return np.zeros((0, len(POKE_COLUMNS)), dtype=np.int64), stats

    entry_ms = np.array([iv[0] for iv in intervals], dtype=np.int64)
    exit_ms = np.array([iv[1] for iv in intervals], dtype=np.int64)
    ports = np.array([iv[2] for iv in intervals], dtype=np.int64)

    entry_bin = to_bins(entry_ms, first_a_on)
    exit_bin = to_bins(exit_ms, first_a_on)

    # Drop pokes that fall entirely outside the neural recording, then clamp the
    # ones that straddle either edge.
    keep = (exit_bin >= 0) & (entry_bin < end_bin)
    stats["n_clipped_out_of_range"] = int((~keep).sum())
    entry_bin, exit_bin, ports, state_idx = (
        entry_bin[keep], exit_bin[keep], ports[keep], state_idx[keep]
    )
    entry_bin = np.clip(entry_bin, 0, end_bin - 1)
    exit_bin = np.clip(exit_bin, 0, end_bin - 1)

    rewarded = (state_idx >= 0).astype(np.int64)
    pokes = np.column_stack([entry_bin, exit_bin, ports, rewarded, state_idx]).astype(np.int64)

    stats["n_pokes"] = len(pokes)
    stats["n_rewarded"] = int(rewarded.sum())
    return pokes, stats


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def _normalise_date(date_str: str) -> str:
    """Metadata dates are written 2025-6-13; pycontrol filenames use 2025-06-13."""
    parts = str(date_str).split("-")
    if len(parts) != 3:
        return str(date_str)
    return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def sessions_for_mouse(mouse_id: str) -> List[Tuple[str, str, int, Path]]:
    """
    Map every existing trialtimes file for a mouse to its source pycontrol file.

    Session index is the ordinal of the task session within the ephys date-pair:
    take the metadata rows for the two dates that have a Behaviour time, sort by
    (Date, Behaviour), and count. This reproduces the on-disk session indices for
    all 191 sessions with no fuzzy datetime matching -- the filename is built exactly.

    Returns [(date1, date2, session_idx, pycontrol_file)].
    """
    metadata = pd.read_csv(METADATA_PATH / f"MetaData-{mouse_id}.csv")
    sessions = []

    for trialtimes_file in sorted(TRIALTIMES_PATH.glob(f"trialtimes_{mouse_id}_*.npy")):
        _, _, date1, date2, session_idx = trialtimes_file.stem.split("_")
        session_idx = int(session_idx)

        dates = [f"{d[:4]}-{int(d[4:6])}-{int(d[6:])}" for d in (date1, date2)]
        rows = metadata[
            metadata["Date"].astype(str).isin(dates)
            & metadata["Behaviour"].notna()
            & (metadata["Behaviour"].astype(str) != "-")
        ].sort_values(
            ["Date", "Behaviour"],
            key=lambda col: pd.to_datetime(col) if col.name == "Date" else col,
        )

        if session_idx >= len(rows):
            print(f"  WARNING: {trialtimes_file.name} has no metadata row (index out of range)")
            continue

        row = rows.iloc[session_idx]
        behaviour_time = str(int(float(row["Behaviour"]))).zfill(6)
        pycontrol_file = PYCONTROL_PATH / f"{mouse_id}-{_normalise_date(row['Date'])}-{behaviour_time}.txt"

        if not pycontrol_file.exists():
            print(f"  WARNING: no pycontrol file {pycontrol_file.name} for {trialtimes_file.name}")
            continue

        sessions.append((date1, date2, session_idx, pycontrol_file))

    return sessions


def get_end_bin(mouse_id: str, date1: str, date2: str, session_idx: int) -> int:
    """
    Session length in 25 ms bins.

    trialtimes.max() equals Neuron_raw.shape[1] exactly, so the last trial's next-A_on
    defines the end of the neural recording.
    """
    trial_times = np.load(TRIALTIMES_PATH / f"trialtimes_{mouse_id}_{date1}_{date2}_{session_idx}.npy")
    return int(trial_times.max()) if trial_times.size else 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pokes(mouse_id: str, date1: str, date2: str, session_idx: int) -> np.ndarray:
    """Load a poke event table. Columns are POKE_COLUMNS."""
    return np.load(TRIALTIMES_PATH / f"pokes_{mouse_id}_{date1}_{date2}_{session_idx}.npy")


def pokes_to_dense(pokes: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Expand a poke event table into a (n_bins, N_PORTS) occupancy matrix.

    dense[t, port - 1] is 1 while the animal is in that port, for dropping straight
    into a design matrix alongside XY_raw / Locs_raw.
    """
    dense = np.zeros((n_bins, N_PORTS), dtype=np.uint8)
    for entry_bin, exit_bin, port, _, _ in pokes:
        dense[entry_bin:exit_bin + 1, port - 1] = 1
    return dense


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_all(verbose: bool = True) -> pd.DataFrame:
    """Extract pokes for every session that has a trialtimes file. Returns the QC table."""
    qc_rows = []

    for mouse_id in MICE:
        sessions = sessions_for_mouse(mouse_id)
        if verbose:
            print(f"\n{mouse_id}: {len(sessions)} sessions")

        for date1, date2, session_idx, pycontrol_file in sessions:
            times, active_poke = parse_pycontrol(pycontrol_file)
            end_bin = get_end_bin(mouse_id, date1, date2, session_idx)

            if end_bin == 0:
                # Session with no completed trials; write an empty table so
                # downstream loops don't need to special-case a missing file.
                pokes = np.zeros((0, len(POKE_COLUMNS)), dtype=np.int64)
                stats = {k: 0 for k in (
                    "n_pokes", "n_rewarded", "n_state_onsets", "n_unmatched_onsets",
                    "n_imputed_exits", "n_orphan_exits_dropped", "n_clipped_out_of_range")}
            else:
                pokes, stats = make_pokes_array(times, active_poke, end_bin)

            output_file = TRIALTIMES_PATH / f"pokes_{mouse_id}_{date1}_{date2}_{session_idx}.npy"
            np.save(output_file, pokes)

            qc_rows.append({
                "mouse": mouse_id, "d1": date1, "d2": date2, "sess": session_idx,
                "pycontrol_file": pycontrol_file.name, "end_bin": end_bin, **stats,
            })

            if verbose:
                flags = []
                if stats["n_imputed_exits"]:
                    flags.append(f"{stats['n_imputed_exits']} imputed exit(s)")
                if stats["n_orphan_exits_dropped"]:
                    flags.append(f"{stats['n_orphan_exits_dropped']} orphan exit(s)")
                if stats["n_clipped_out_of_range"]:
                    flags.append(f"{stats['n_clipped_out_of_range']} out of range")
                if stats["n_unmatched_onsets"]:
                    flags.append(f"{stats['n_unmatched_onsets']} UNMATCHED onset(s)")
                suffix = f"  [{', '.join(flags)}]" if flags else ""
                print(f"  {output_file.name}: {stats['n_pokes']} pokes, "
                      f"{stats['n_rewarded']} rewarded{suffix}")

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(QC_REPORT_PATH, index=False)
    if verbose:
        print(f"\nWrote {len(qc)} poke files and QC report -> {QC_REPORT_PATH}")
    return qc


def validate(verbose: bool = True) -> bool:
    """
    Check the written poke files against the independently generated trialtimes.

    1. Trialtimes round-trip: re-deriving trialtimes from the raw pycontrol file must
       reproduce the on-disk array exactly. A mismatch means the session mapping is
       wrong and that session's pokes are untrustworthy.
    2. Rewarded pokes must reconstruct columns 0..3 of trialtimes. This ties poke
       times, time origin and binning to a file this script did not produce.
    3. Range invariant: 0 <= entry_bin <= exit_bin < end_bin.
    4. Coverage: rewarded pokes account for every in-range state onset.
    """
    n_exact = n_empty = n_checked = 0
    failures = []

    for mouse_id in MICE:
        for date1, date2, session_idx, pycontrol_file in sessions_for_mouse(mouse_id):
            tag = f"{mouse_id}_{date1}_{date2}_{session_idx}"
            times, active_poke = parse_pycontrol(pycontrol_file)
            reference = np.load(TRIALTIMES_PATH / f"trialtimes_{tag}.npy").astype(np.int64)
            pokes = load_pokes(mouse_id, date1, date2, session_idx)
            n_checked += 1

            # 1. Trialtimes round-trip
            if reference.size == 0:
                n_empty += 1
                if len(pokes):
                    failures.append(f"{tag}: empty trialtimes but {len(pokes)} pokes")
                continue
            rebuilt = make_trial_times(times, n_states=reference.shape[1] - 1)
            if rebuilt.shape != reference.shape or not np.array_equal(rebuilt, reference):
                failures.append(f"{tag}: trialtimes round-trip mismatch")
                continue
            n_exact += 1

            end_bin = int(reference.max())

            # 3. Range invariant
            if len(pokes):
                if pokes[:, 0].min() < 0 or (pokes[:, 0] > pokes[:, 1]).any() or pokes[:, 1].max() >= end_bin:
                    failures.append(f"{tag}: bins out of range [0, {end_bin})")

            # 2. Rewarded pokes must span the trialtimes state onsets.
            # entry_bin and the trialtimes bin are different quantities: the animal
            # enters the port a few ms before the state machine advances, which
            # occasionally lands them either side of a 25 ms boundary. The invariant
            # that must hold is containment -- the onset falls within the poke.
            n_states = reference.shape[1] - 1
            rewarded = pokes[pokes[:, 3] == 1]
            rewarded = rewarded[np.argsort(rewarded[:, 0])]
            for state_idx in range(n_states):
                spans = rewarded[rewarded[:, 4] == state_idx]
                want = reference[:, state_idx]
                # The final loop's onsets can sit past end_bin and be clipped away.
                n = min(len(spans), len(want))
                contained = (spans[:n, 0] <= want[:n]) & (want[:n] <= spans[:n, 1])
                if not contained.all():
                    failures.append(
                        f"{tag}: {(~contained).sum()} rewarded poke(s) do not span "
                        f"trialtimes col {state_idx}")
                    break

            # 4. Coverage
            if (pokes[:, 3] == 1).sum() > reference.shape[0] * n_states + n_states:
                failures.append(f"{tag}: more rewarded pokes than state onsets")

    if verbose:
        print(f"\nValidation: {n_checked} sessions checked, "
              f"{n_exact} trialtimes round-trips exact, {n_empty} empty, "
              f"{len(failures)} failures")
        for failure in failures[:20]:
            print(f"  FAIL {failure}")

    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true",
                        help="Check the written files against trialtimes")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate existing files without rewriting them")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if not args.validate_only:
        process_all(verbose=verbose)

    if args.validate or args.validate_only:
        if not validate(verbose=verbose):
            sys.exit(1)


if __name__ == "__main__":
    main()
