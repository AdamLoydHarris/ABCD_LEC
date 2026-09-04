"""
data_dic Build

Joins the per-session arrays on disk into the four pickles every LEC analysis loads:

    data_dic_lec.pkl      {mouse_recday: {session: {Neuron_raw, Task, Trial_times, ...}}}
    norm_neurons_dic.pkl  {mouse_recday: [Neurons_norm per session]}
    session_inds_dic.pkl  {mouse_recday: [session indices with neurons + trialtimes]}
    tasks_dic.pkl         {mouse_recday: {session: Task}}

Lifted out of `LEC_sploratory_analysis_with_glm_and_population.ipynb` (cells 16-21), whose
copy is commented out and gitignored, so that the step that assembles the dataset is
tracked, re-runnable and — the point of moving it — **guarded**.

Everything is joined by filename: `Neuron_raw_{recday}_{s}.npy` against
`trialtimes_{recday}_{s}.npy` against `data/processed/{mouse}/{d1}_{d2}/Locs_raw_{recday}_{s}.npy`.
That was never the problem. The mispairing that produced
`docs/BUG_ly05_recday_mismatch.md` happened upstream, when the neural extraction wrote one
day's spikes under another day's name; this builder faithfully propagated it. So before
anything is written, both registry guards run:

    validate_data_dic               rows == QC units of the block the recday is named after
    validate_tasks_against_pycontrol  Task == that day's own pyControl active_poke

Run `python build_data_dic.py --check` to build in memory and diff against the pickles
already on disk without writing anything.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.stats as st
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recday_registry as rr                                          # noqa: E402

PROCESSED_DATA = rr.REPO_ROOT / "data" / "processed_data"
NEURON_PATH = PROCESSED_DATA / "neuron_raw_mingyutest"
TRIALTIMES_PATH = PROCESSED_DATA / "trialtimes_raw_mingyutest"
TRACKING_PATH = rr.REPO_ROOT / "data" / "processed"

MAX_SESSIONS = 12          # highest plausible number of task sessions in a recday
NUM_BINS = 90              # normalised bins per state
NUM_STATES = 4

#: Maze locations as stored in `Locs_raw`: nodes 1-9, then the 12 edges as 10-21.
#: The edge ORDER defines the integer codes, so it is fixed here and must not be sorted.
MAZE_EDGES = ("1-2", "2-3", "1-4", "2-5", "3-6", "4-5",
              "5-6", "4-7", "5-8", "6-9", "7-8", "8-9")
LOCATION_MAPPING = {f"node_{i}": i for i in range(1, 10)}
LOCATION_MAPPING.update({f"edge_{e}": 10 + i for i, e in enumerate(MAZE_EDGES)})
REVERSE_MAPPING = {v: k for k, v in LOCATION_MAPPING.items()}


def locs_to_int(locs_raw) -> np.ndarray:
    """String location labels -> the integer codes in `LOCATION_MAPPING`; 0 for unknown."""
    return np.array([LOCATION_MAPPING.get(str(loc), 0) for loc in locs_raw], dtype=int)


# ---------------------------------------------------------------------------
# Normalisation (verbatim behaviour from the notebook it replaces)
# ---------------------------------------------------------------------------

def partition(alist, indices):
    """Split a list at `indices`, as an object array so ragged trials survive."""
    return np.asarray([np.asarray(alist[i:j]) for i, j in zip(indices[:-1], indices[1:])],
                      dtype=object)


def normalise(xx, num_bins=NUM_BINS):
    """Resample one trial's firing to `num_bins` by binned mean (time-warp to phase)."""
    length = len(xx)
    if length < num_bins:
        xx = np.repeat(xx, 10) / 10
        length *= 10
    return st.binned_statistic(np.arange(length), xx, "mean", bins=num_bins)[0]


def smooth_circ(xx, sigma=10, axis=0):
    """Gaussian smooth with wrap-around, for quantities defined on the task loop."""
    return gaussian_filter1d(np.hstack((xx, xx, xx)), sigma, axis=axis)[len(xx):len(xx) * 2]


def std_err(data_neuron):
    return smooth_circ(sem(data_neuron, axis=0))


def normalise_session(neuron_raw: np.ndarray, trial_times: np.ndarray) -> Dict[str, np.ndarray]:
    """Trial x normalised-phase matrices for one session, plus its summaries."""
    num_neurons, num_trials = len(neuron_raw), len(trial_times)
    width = NUM_BINS * NUM_STATES
    conc = np.hstack((np.concatenate(trial_times[:, :-1]), trial_times[-1, -1])).astype(int)

    out = {k: np.full((num_neurons, width), np.nan)
           for k in ("Mean_norm", "Smoothed_norm", "Std_err_smooth")}
    norm = np.full((num_neurons, num_trials, width), np.nan)

    for neuron in range(num_neurons):
        split = partition(list(neuron_raw[neuron, :]), list(conc))
        split_norm = np.asarray([normalise(s) for s in split])
        per_trial = split_norm.reshape(len(split_norm) // NUM_STATES,
                                       len(split_norm[0]) * NUM_STATES)
        norm[neuron] = per_trial
        out["Mean_norm"][neuron] = np.mean(per_trial, axis=0)
        out["Smoothed_norm"][neuron] = smooth_circ(out["Mean_norm"][neuron])
        out["Std_err_smooth"][neuron] = std_err(per_trial)

    return {"Neurons_norm": norm, "Neurons_mean": np.mean(norm, axis=1), **out}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def discover_recdays() -> List[str]:
    """Recdays with at least one `Neuron_raw` array on disk.

    Quarantined arrays carry an `.INVALID_*` suffix rather than `.npy`, so a recday whose
    data has been withdrawn simply stops appearing here.
    """
    stems = set()
    for path in NEURON_PATH.glob("Neuron_raw_*.npy"):
        recday = "_".join(path.stem[len("Neuron_raw_"):].split("_")[:-1])
        if not recday.endswith("_sb"):
            stems.add(recday)
    return sorted(stems)


def build(recdays: List[str] = None, verbose: bool = True):
    """Assemble the four dictionaries. Returns them; writes nothing."""
    recdays = recdays or discover_recdays()
    data_dic, norm_neurons_dic, session_inds_dic, tasks_dic = {}, {}, {}, {}

    for recday in recdays:
        mouse, date1, date2 = rr.split_recday(recday)
        tracking_dir = TRACKING_PATH / mouse / f"{date1}_{date2}"
        data_dic[recday], tasks_dic[recday] = {}, {}
        norm_list, session_inds = [], []

        for session in range(MAX_SESSIONS):
            neuron_file = NEURON_PATH / f"Neuron_raw_{recday}_{session}.npy"
            trial_file = TRIALTIMES_PATH / f"trialtimes_{recday}_{session}.npy"
            task_file = TRIALTIMES_PATH / f"Task_data_{recday}_{session}.npy"
            entry = {}

            neuron_raw = np.load(neuron_file) if neuron_file.exists() else None
            if neuron_raw is not None:
                entry["Neuron_raw"] = neuron_raw
            if task_file.exists():
                task = np.load(task_file, allow_pickle=True)
                entry["Task"] = task
                tasks_dic[recday][session] = task

            if neuron_raw is not None and trial_file.exists():
                trial_times = np.load(trial_file)
                entry["Trial_times"] = trial_times
                entry["num_trials"] = len(trial_times)
                entry["num_neurons"] = len(neuron_raw)
                session_inds.append(session)
                if len(trial_times) > 0 and len(neuron_raw) > 0:
                    normalised = normalise_session(neuron_raw, trial_times)
                    entry.update(normalised)
                    norm_list.append(normalised["Neurons_norm"])
            elif neuron_raw is not None:
                entry["Trial_times"] = None
                entry["num_trials"] = 0
                entry["num_neurons"] = len(neuron_raw)

            if not entry:
                continue

            # Tracking. Missing files are normal (a session can lack SLEAP output) but
            # must be visible, not silent: an analysis that indexes a session without
            # Locs_raw will fail far from here.
            entry["location_mapping"] = LOCATION_MAPPING
            entry["reverse_mapping"] = REVERSE_MAPPING
            for key, loader in (("HD_raw", None), ("XY_raw", None), ("Locs_raw", locs_to_int)):
                path = tracking_dir / f"{key}_{recday}_{session}.npy"
                if path.exists():
                    array = np.load(path, allow_pickle=True)
                    entry[key] = loader(array) if loader else array
                elif verbose:
                    print(f"  {recday} s{session}: no {key}")

            # Key order the pickle has always had, so a diff against the previous build
            # compares content rather than ordering.
            order = ["Neuron_raw", "Task", "Trial_times", "num_trials", "num_neurons",
                     "Neurons_norm", "Neurons_mean", "Mean_norm", "Smoothed_norm",
                     "Std_err_smooth", "HD_raw", "XY_raw", "Locs_raw",
                     "location_mapping", "reverse_mapping"]
            data_dic[recday][session] = {k: entry[k] for k in order if k in entry}

        norm_neurons_dic[recday] = norm_list
        session_inds_dic[recday] = session_inds
        if verbose:
            print(f"{recday}: {len(data_dic[recday])} sessions, "
                  f"{len(session_inds)} with neurons + trialtimes")

    return data_dic, norm_neurons_dic, session_inds_dic, tasks_dic


def diff_against_disk(data_dic, verbose: bool = True) -> List[str]:
    """Recdays whose rebuilt content differs from the `data_dic_lec.pkl` on disk.

    Run before overwriting: every recday that was not meant to change should come back
    identical, which is what makes a change to the remaining ones interpretable.
    """
    path = PROCESSED_DATA / "data_dic_lec.pkl"
    if not path.exists():
        return []
    with open(path, "rb") as f:
        old = pickle.load(f)
    old = {str(k): v for k, v in old.items()}

    changed = []
    for recday in sorted(set(old) | set(data_dic)):
        if recday not in old:
            changed.append(f"{recday}: NEW")
            continue
        if recday not in data_dic:
            changed.append(f"{recday}: REMOVED")
            continue
        for session in sorted(set(old[recday]) | set(data_dic[recday])):
            a = old[recday].get(session, {})
            b = data_dic[recday].get(session, {})
            if set(a) != set(b):
                changed.append(f"{recday} s{session}: keys {sorted(set(a) ^ set(b))}")
                continue
            for key in a:
                if key in ("location_mapping", "reverse_mapping"):
                    continue
                if not _same(a[key], b[key]):
                    changed.append(f"{recday} s{session}: {key} differs")
    if verbose:
        # Summarise by recday FIRST, then show examples. A truncated list of individual
        # differences can hide an affected recday entirely, which is the failure mode this
        # whole exercise is about.
        per_recday = {}
        for line in changed:
            per_recday.setdefault(line.split()[0].rstrip(":"), 0)
            per_recday[line.split()[0].rstrip(":")] += 1
        print(f"\ndiff vs disk: {len(changed)} difference(s) "
              f"across {len(per_recday)} recday(s)")
        for recday, count in sorted(per_recday.items()):
            print(f"  {recday}: {count}")
        for line in changed[:20]:
            print(f"    e.g. {line}")
    return changed


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.allclose(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recdays", nargs="*", default=None)
    parser.add_argument("--check", action="store_true",
                        help="build and diff against the pickles on disk; write nothing")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the registry guards (for debugging a known-bad build)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    data_dic, norm_dic, session_dic, tasks_dic = build(args.recdays, verbose=verbose)
    print(f"\nbuilt {len(data_dic)} recdays")

    if not args.no_validate:
        rr.validate_data_dic(data_dic, strict=True, verbose=verbose)
        rr.validate_tasks_against_pycontrol(data_dic, strict=True, verbose=verbose)

    if args.check:
        diff_against_disk(data_dic, verbose=verbose)
        print("\n--check: nothing written")
        return

    for name, obj in (("data_dic_lec", data_dic), ("norm_neurons_dic", norm_dic),
                      ("session_inds_dic", session_dic), ("tasks_dic", tasks_dic)):
        with open(PROCESSED_DATA / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
        print(f"wrote {name}.pkl")


if __name__ == "__main__":
    main()
