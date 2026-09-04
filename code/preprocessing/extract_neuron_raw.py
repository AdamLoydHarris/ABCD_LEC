"""
Neuron_raw Extraction

Bins the QC'd single units of each sorted block into per-session spike-count arrays on the
25 ms grid the rest of the pipeline uses, and writes them as
`Neuron_raw_{mouse}_{d1}_{d2}_{sess}.npy` (task sessions) and `..._sb_{j}.npy` (sleep box).

Written to replace the untracked script that produced `neuron_raw_mingyutest/`, which
paired sorted blocks to recday names **by position**. ly05 has five sorted blocks but only
four were extracted, so every block after the skipped one landed under the previous block's
name: `ly05_20250618_20250619` ended up holding the 06-20/23 sorting against 06-18/19
behaviour, and 06-18/19's own 109 units never entered the dataset at all. Nothing detected
it for months, because the neural data is binned onto the behavioural timeline and so every
shape and length check passes; only the unit count differs. See
`docs/BUG_ly05_recday_mismatch.md`.

Here every lookup is by date:

    recday name -> sorted block          recday_registry.recday_to_block  (date match)
    session idx -> concat recording      the metadata Ephys timestamp     (exact match)
    session idx -> pycontrol file        recday_registry.pycontrol_file   (date checked)

Workflow:
1. Read `recording_sessions_in_concat.csv` and turn `dat_size` into sample offsets
2. Map each metadata task row to its concat recording via the Ephys timestamp; the
   unmatched recordings are the sleep-box sessions, in order
3. Bin the QC units of that block into 25 ms bins, in `QC_single_units.npy` row order
4. Anchor and truncate each session (see `session_window`)

Run `python extract_neuron_raw.py --validate-only` to re-derive every array that already
exists and assert it is byte-identical. That gate is the point: the extractor has to
reproduce known-good output before it is trusted to write anything new. Two recordings
have no cached rsync train (ah08 2025-06-18/19, ah10 2025-06-16/17); their origins are
recovered from the stored arrays instead, and reported as such — see `recover_origin`.
"""

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recday_registry as rr                                          # noqa: E402

# Output path (alongside the sibling trialtimes_ / pokes_ files' consumers)
NEURON_PATH = rr.REPO_ROOT / "data" / "processed_data" / "neuron_raw_mingyutest"
TRIALTIMES_PATH = rr.REPO_ROOT / "data" / "processed_data" / "trialtimes_raw_mingyutest"
RSYNC_PATH = rr.REPO_ROOT / "data" / "processed_data" / "ephys_rsync_mingyutest"

# Constants
BIN_MS = 25                      # bin width, matching trialtimes and the tracking arrays
SAMPLE_RATE_HZ = 30_000
SAMPLES_PER_MS = SAMPLE_RATE_HZ // 1000
SAMPLES_PER_BIN = BIN_MS * SAMPLES_PER_MS          # 750
BYTES_PER_SAMPLE = 768           # 384 channels x int16, for dat_size -> n_samples
NEURON_DTYPE = np.uint16


# ---------------------------------------------------------------------------
# Session geometry
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def concat_table(recday: str) -> pd.DataFrame:
    """Recordings making up a block's concatenated timeline, with sample offsets.

    Cached: `session_window` needs it once per session, and re-parsing the CSV each time
    dominated the validation run. Callers only read from the frame.

    Columns: date (YYYYMMDD), time, start (first sample in the concatenated stream),
    n_samples. Offsets come from `dat_size`, not from the `duration` column, because the
    concatenation is byte-exact and `duration` is rounded to two decimals.
    """
    csv = pd.read_csv(rr.kilosort_dir(recday).parent / "recording_sessions_in_concat.csv")
    n_samples = (csv["dat_size"] // BYTES_PER_SAMPLE).to_numpy()
    return pd.DataFrame({
        "date": [str(d).replace("-", "") for d in csv["date"]],
        "time": csv["time"].astype(str),
        "start": np.concatenate([[0], np.cumsum(n_samples)])[:-1],
        "n_samples": n_samples,
    })


def session_map(recday: str) -> Tuple[List[int], List[int]]:
    """Concat-row index for each task session, and for each sleep-box session.

    Task sessions are matched on the metadata **Ephys timestamp**, so session `i` is the
    recording that row actually names -- never the i-th wake recording. That matters: the
    wake recordings and the metadata task rows do not always line up one-to-one (a session
    can be recorded but never scored, e.g. ah08's 'error sess'), so counting wake rows
    silently shifts every later session.

    Everything not claimed by a task row is a sleep-box session, numbered in concat order.
    """
    table = concat_table(recday)
    key = {(d, t): i for i, (d, t) in enumerate(zip(table["date"], table["time"]))}

    wake: List[int] = []
    for _, row in rr.task_session_rows(recday).iterrows():
        hit = key.get((str(row["_date"]), str(row["Ephys"])))
        if hit is None:
            raise KeyError(f"{recday}: metadata Ephys {row['_date']} {row['Ephys']} "
                           f"is not in recording_sessions_in_concat.csv")
        wake.append(hit)

    sleep = [i for i in range(len(table)) if i not in set(wake)]
    return wake, sleep


def first_rsync_sample(recday: str, concat_row: int) -> Optional[int]:
    """First sync-pulse sample of one recording, in that recording's own timebase."""
    mouse, _, _ = rr.split_recday(recday)
    row = concat_table(recday).iloc[concat_row]
    date = str(row["date"])
    path = (RSYNC_PATH / mouse /
            f"rsync_timestamps_{date[:4]}-{date[4:6]}-{date[6:]}_{row['time']}.npy")
    if not path.exists():
        return None
    pulses = np.load(path)
    return int(pulses[0]) if len(pulses) else None


def trialtimes(recday: str, session: int) -> np.ndarray:
    """The session's trialtimes array, or an empty array if it has no completed trials."""
    path = TRIALTIMES_PATH / f"trialtimes_{recday}_{session}.npy"
    return np.load(path) if path.exists() else np.zeros((0, 5), dtype=np.int64)


# ---------------------------------------------------------------------------
# Anchoring
# ---------------------------------------------------------------------------

def session_window(recday: str, session: int, concat_row: int,
                   *, sleep: bool) -> Tuple[int, int]:
    """`(origin_sample, n_bins)` for one session, in that recording's own timebase.

    Three cases, all recovered by reproducing the existing arrays byte-for-byte:

    - **sleep box**: origin 0, the whole recording. `n_bins = n_samples // 750`.
    - **task session with trials**: origin is the first A_on, snapped to the pyControl
      25 ms grid (`(first_A_on // 25) * 25`, exactly as `extract_pokes.to_bins` does) and
      carried into ephys samples through the rsync offset. Truncated to `trialtimes.max()`,
      which is why `trialtimes.max() == Neuron_raw.shape[1]` for every scored session.
    - **task session with no completed trials**: no A_on to anchor to, so origin is the
      first sync pulse and the array runs to the end of the recording.

    The rsync offset is a **constant** shift, not a fitted linear map. Fitting the ~1e-5
    clock drift across the session is more accurate in principle, but it is not what
    produced the existing files and reproducing them exactly is the gate that makes this
    script trustworthy. The drift is ~13 ms over a 20 min session, i.e. half a bin.
    """
    row = concat_table(recday).iloc[concat_row]
    n_samples = int(row["n_samples"])

    if sleep:
        return 0, n_samples // SAMPLES_PER_BIN

    sync = first_rsync_sample(recday, concat_row)
    if sync is None:
        raise FileNotFoundError(f"{recday} s{session}: no rsync timestamps")

    times, _ = _parse_pycontrol(rr.pycontrol_file(recday, session))
    a_on = times.get("A_on", np.array([], dtype=np.int64))
    tt = trialtimes(recday, session)

    if len(a_on) == 0 or tt.size == 0:
        return sync, (n_samples - sync) // SAMPLES_PER_BIN

    # Ephys sample of pyControl time 0, then of the binned first A_on.
    pc_zero = sync - int(times["rsync"][0]) * SAMPLES_PER_MS
    origin = pc_zero + (int(a_on[0]) // BIN_MS) * SAMPLES_PER_BIN
    return origin, int(tt.max())


def _parse_pycontrol(path: Path):
    """`extract_pokes.parse_pycontrol`, imported lazily to keep the module import cheap."""
    from extract_pokes import parse_pycontrol
    return parse_pycontrol(path)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def recover_origin(spike_samples: np.ndarray, unit_rows: np.ndarray,
                   stored: np.ndarray) -> Optional[int]:
    """Recover the time origin a stored array was binned at, from the array itself.

    For the true origin, the number of QC-unit spikes falling below
    `origin + (k+1) * 750` is a fixed count plus the array's own cumulative population
    total through bin k. Scoring candidate origins by how badly that identity fails
    picks the right one out exactly, in two passes (bin-resolution, then sample).

    Needed for ah08's 2025-06-18/19 and ah10's 2025-06-16/17, the only recordings with no
    cached rsync pulse train, whose task arrays could otherwise not be re-derived at all.

    **What this does and does not prove.** It reads the origin off the stored array, so it
    cannot by itself tell you the origin was *right*. What it establishes is that the
    array is a correctly-binned segment of this block's QC units in QC order — i.e. the
    right day's spikes, right units, right convention. The origin itself is corroborated
    separately: `session_window`'s formula run backwards from the recovered origin gives an
    implied first-sync sample, and that matches the recording's raw TTL exactly for ah10
    (8/8 sessions) and to within the stream-dependent offset seen on every ProbeA/ProbeD
    recording whose cache does exist for ah08.
    """
    n_bins = stored.shape[1]
    keep = unit_rows >= 0
    samples = np.sort(spike_samples[keep])
    if not len(samples) or not n_bins:
        return None
    cumulative = np.cumsum(np.asarray(stored, dtype=np.int64).sum(axis=0))
    steps = (np.arange(n_bins) + 1) * SAMPLES_PER_BIN

    def error(origin: int) -> int:
        below = np.searchsorted(samples, origin)
        return int(np.abs(np.searchsorted(samples, origin + steps)
                          - (below + cumulative)).sum())

    span = int(samples[-1]) - n_bins * SAMPLES_PER_BIN
    if span < 0:
        return None
    coarse = min(range(0, span + 1, SAMPLES_PER_BIN), key=error)
    fine = min(range(coarse - SAMPLES_PER_BIN, coarse + SAMPLES_PER_BIN + 1), key=error)
    return fine if error(fine) == 0 else None


def bin_session(spike_samples: np.ndarray, unit_rows: np.ndarray,
                n_units: int, origin: int, n_bins: int) -> np.ndarray:
    """Spike counts per (unit, 25 ms bin), for spikes already restricted to one recording.

    `spike_samples` are relative to the start of that recording. Arithmetic is in integer
    samples throughout: computing bin indices in float milliseconds instead moves ~50 of
    5 million bins by one, through nothing but rounding.
    """
    bins = (spike_samples - origin) // SAMPLES_PER_BIN
    keep = (unit_rows >= 0) & (bins >= 0) & (bins < n_bins)
    counts = np.zeros((n_units, n_bins), dtype=np.int64)
    np.add.at(counts, (unit_rows[keep], bins[keep]), 1)
    return counts.astype(NEURON_DTYPE)


def extract_recday(recday: str, verbose: bool = True):
    """`(arrays, unresolved, recovered)` for one recday, keyed by output filename stem.

    ah08 has no cached rsync train for 2025-06-18/19 and ah10 none for 2025-06-16/17.
    For those, an array that already exists has its origin read back off it
    (`recover_origin`) and is still checked; the name goes in `recovered` so the caller can
    label it honestly. `unresolved` holds sessions with neither a cache nor a stored array
    — nothing is ever anchored by guesswork.
    """
    ks = rr.kilosort_dir(recday)
    qc = np.load(ks / "QC_single_units.npy")
    spike_times = np.load(ks / "sorter_output/spike_times.npy").squeeze()
    spike_clusters = np.load(ks / "sorter_output/spike_clusters.npy").squeeze()

    # cluster id -> row in QC order; -1 for clusters that did not survive QC
    lut = np.full(int(spike_clusters.max()) + 2, -1, dtype=np.int64)
    lut[qc] = np.arange(len(qc))

    order = np.argsort(spike_times, kind="stable")
    spike_times, spike_clusters = spike_times[order], spike_clusters[order]

    table = concat_table(recday)
    wake, sleep = session_map(recday)
    if verbose:
        print(f"{recday}: {len(qc)} units, {len(wake)} task + {len(sleep)} sleep sessions")

    out: Dict[str, np.ndarray] = {}
    unresolved: Dict[str, str] = {}
    recovered: set = set()
    for rows, is_sleep in ((wake, False), (sleep, True)):
        for idx, concat_row in enumerate(rows):
            name = (f"Neuron_raw_{recday}_sb_{idx}" if is_sleep
                    else f"Neuron_raw_{recday}_{idx}")
            start = int(table["start"].iloc[concat_row])
            stop = start + int(table["n_samples"].iloc[concat_row])
            lo, hi = np.searchsorted(spike_times, [start, stop])
            samples = spike_times[lo:hi] - start
            unit_rows = lut[spike_clusters[lo:hi]]

            try:
                origin, n_bins = session_window(recday, idx, concat_row, sleep=is_sleep)
            except (KeyError, FileNotFoundError) as exc:
                # No cached rsync train for this recording. If the array already exists,
                # read its origin back off it so the session can still be checked; if it
                # does not, there is nothing to anchor to and we say so rather than
                # inventing one. One unanchorable session must not take the rest of the
                # recday with it either -- sleep arrays need no sync at all.
                stored_path = NEURON_PATH / f"{name}.npy"
                origin = None
                if stored_path.exists():
                    stored = np.load(stored_path)
                    origin = recover_origin(samples, unit_rows, stored)
                if origin is None:
                    unresolved[name] = str(exc)
                    continue
                n_bins = stored.shape[1]
                recovered.add(name)

            out[name] = bin_session(samples, unit_rows, len(qc), origin, n_bins)

    if verbose:
        if recovered:
            print(f"  {len(recovered)} session(s) had no rsync cache; origin recovered "
                  f"from the stored array (see recover_origin)")
        if unresolved:
            print(f"  {len(unresolved)} session(s) could not be anchored at all, e.g. "
                  f"{next(iter(unresolved.values()))}")
    return out, unresolved, recovered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(recdays: List[str], *, write: bool, verbose: bool = True) -> pd.DataFrame:
    """Extract the given recdays; compare against anything already on disk.

    Existing files are never overwritten silently -- a byte difference is reported as a
    failure, and `--write` only ever creates files that are absent.
    """
    rows = []
    for recday in recdays:
        try:
            arrays, unresolved, recovered = extract_recday(recday, verbose=verbose)
        except (KeyError, FileNotFoundError) as exc:
            print(f"  {recday}: SKIPPED ({exc})")
            rows.append({"recday": recday, "status": "skipped", "detail": str(exc)})
            continue

        for name, why in sorted(unresolved.items()):
            rows.append({"recday": recday, "array": name, "status": "unvalidated",
                         "detail": why})

        for name, array in sorted(arrays.items()):
            path = NEURON_PATH / f"{name}.npy"
            if path.exists():
                existing = np.load(path)
                exact = (existing.shape == array.shape
                         and np.array_equal(existing, array.astype(existing.dtype)))
                status = ("exact (origin recovered)" if exact and name in recovered
                          else "exact" if exact else "MISMATCH")
                if not exact and verbose:
                    print(f"  {name}: MISMATCH stored {existing.shape} "
                          f"vs rebuilt {array.shape}")
            elif write:
                np.save(path, array)
                status = "written"
                if verbose:
                    print(f"  {name}: wrote {array.shape}")
            else:
                status = "missing"
                if verbose:
                    print(f"  {name}: absent (use --write to create) {array.shape}")
            rows.append({"recday": recday, "array": name, "status": status,
                         "n_units": array.shape[0], "n_bins": array.shape[1]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recdays", nargs="*", default=None,
                        help="recdays to process (default: every block on disk)")
    parser.add_argument("--write", action="store_true",
                        help="create arrays that do not exist yet; never overwrites")
    parser.add_argument("--validate-only", action="store_true",
                        help="re-derive existing arrays and assert byte-equality")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    recdays = args.recdays or rr.all_recdays()
    report = process(recdays, write=args.write and not args.validate_only,
                     verbose=not args.quiet)

    counts = report["status"].value_counts().to_dict()
    print(f"\n{len(recdays)} recday(s): {counts}")

    bad = report[report["status"] == "MISMATCH"]
    if len(bad):
        print(f"\n{len(bad)} array(s) do NOT reproduce:")
        print(bad.to_string(index=False))
        sys.exit(1)
    print("every existing array reproduced byte-for-byte")


if __name__ == "__main__":
    main()
