"""
Recday <-> sorted-block registry, and the guards that keep them paired correctly.

A "recday" is a `{mouse}_{YYYYMMDD}_{YYYYMMDD}` name; a "block" is the matching
`{YYYY-MM-DD}_{YYYY-MM-DD}_preprocessed` directory under `data/preprocessed/ephys/{mouse}/`.
Every lookup here matches on the **date string**. Nothing in this module pairs the two
lists positionally, because that is exactly how `ly05_20250618_20250619` came to hold one
day's spikes against another day's behaviour (docs/BUG_ly05_recday_mismatch.md): ly05 has
five sorted blocks but only four were extracted, and a `zip`-style pairing shifted every
block after the skipped one into the previous block's name.

The bug was silent for months because the neural data is binned onto the behavioural
timeline, so every shape and length check passes. The two observables that separate the
days are the **number of units** and the **task sequences**, which is what the two
validators here check:

    validate_data_dic(dd)                 # neural side:      rows == QC units of the named block
    validate_tasks_against_pycontrol(dd)  # behavioural side: Task == that day's active_poke

Either alone would have caught it. Run both after building or loading a `data_dic`.

The behavioural guard reads the task from the **pyControl file's own `active_poke` line**,
not from the `Structure` column of the metadata CSV. `Structure` is typed by hand and has
five known errors (four sessions of `ly06_20250616_20250617` and one of
`ly06_20250618_20250619`), where the data is right and the spreadsheet is wrong; see
`metadata_task_discrepancies`. Asserting against it would fail on good data.

Usage::

    import recday_registry as rr
    rr.qc_unit_count('ly05_20250618_20250619')     # 109
    rr.validate_data_dic(data_dic)                 # raises on any mismatch
"""

from __future__ import annotations

import glob
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EPHYS_ROOT = REPO_ROOT / "data" / "preprocessed" / "ephys"
METADATA_DIR = REPO_ROOT / "data" / "metadata"
PYCONTROL_DIR = REPO_ROOT / "data" / "raw" / "pycontrol"

#: Mice with ephys. ah09 has histology but no recordings.
EPHYS_MICE = ("ah08", "ah10", "ly05", "ly06", "ly07")

#: Recdays that must not enter any `data_dic`-joined analysis, with the reason.
#: Empty is the healthy state -- an entry here means known-bad data is on disk.
#:
#: `ly05_20250618_20250619` lived here until it was re-extracted from its own sorting by
#: `preprocessing/extract_neuron_raw.py`; the mismatched arrays are quarantined on disk
#: under an `.INVALID_ly05_recday_mismatch` suffix.
EXCLUDE_RECDAYS: dict[str, str] = {}

#: Recdays whose **cached fits** predate a data fix and must not be reused, even though the
#: recday itself is now good. Distinct from `EXCLUDE_RECDAYS`: the data is fine, the
#: pickles on disk are stale. Dropped at load time by `glm_analysis_v2.load_glm_results`
#: and `run_or_load_glm`, so aggregates computed from an old cache silently lose the stale
#: recday rather than silently keeping a fit of corrupt data.
#:
#: Remove an entry once that recday has actually been refitted.
STALE_CACHE_RECDAYS: dict[str, str] = {
    "ly05_20250618_20250619":
        "PRE-REFIT fits are of the 06-20/23 spikes; refit against the re-extracted "
        "109-unit arrays (docs/BUG_ly05_recday_mismatch.md)",
}

#: Staleness is a property of a PICKLE, not of a recday -- but `STALE_CACHE_RECDAYS` is keyed
#: by recday, so on its own it cannot tell a pre-refit fit from a post-refit one and drops
#: both. The W1 production fits ARE built from the corrected data (verified: the new
#: `ly05_20250618_20250619` fit has 109 neurons, matching `unit_regions`, where the old
#: caches have 91 -- the wrong day).
#:
#: Post-refit sections are identifiable by name: `w1_refit.section_name` stamps the
#: configuration into it as `{regset}_{width}ms_{scheme}`. A section carrying that stamp
#: postdates the fix, so the stale list must not apply to it.
_POST_REFIT_SECTION = re.compile(r'__(full|matched)_\d+ms_(decile|uniform)$')


def is_post_refit_section(section_name) -> bool:
    """True if `section_name` was produced by the W1 production refit.

    Used to stop `STALE_CACHE_RECDAYS` from dropping good recdays out of the new fits while
    still protecting every pre-refit cache that genuinely holds the wrong day's spikes.
    """
    return bool(section_name and _POST_REFIT_SECTION.search(str(section_name)))


# ---------------------------------------------------------------------------
# recday <-> block, by date
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def blocks_for(mouse: str) -> tuple[str, ...]:
    """Sorted recording-block directory names for a mouse."""
    return tuple(sorted(os.path.basename(p)
                        for p in glob.glob(str(EPHYS_ROOT / mouse / "*_preprocessed"))))


def block_to_recday(mouse: str, block: str) -> str:
    """``ah08`` + ``2025-06-13_2025-06-15_preprocessed`` -> ``ah08_20250613_20250615``."""
    dates = re.findall(r"(\d{4})-(\d{2})-(\d{2})", block)
    return f"{mouse}_" + "_".join("".join(d) for d in dates)


def split_recday(recday: str) -> tuple[str, str, str]:
    """``ly05_20250618_20250619`` -> ``('ly05', '20250618', '20250619')``."""
    mouse, date1, date2 = str(recday).split("_")
    return mouse, date1, date2


def recday_to_block(recday: str) -> str:
    """Block directory name whose dates match `recday`.

    Raises `KeyError` if no block matches -- never falls back to a positional guess.
    """
    mouse, _, _ = split_recday(recday)
    for block in blocks_for(mouse):
        if block_to_recday(mouse, block) == str(recday):
            return block
    raise KeyError(f"no sorted block matches {recday} "
                   f"(mouse has {blocks_for(mouse)})")


def kilosort_dir(recday: str) -> Path:
    """`kilosort_output` directory for the block whose dates match `recday`."""
    mouse, _, _ = split_recday(recday)
    return EPHYS_ROOT / mouse / recday_to_block(recday) / "kilosort_output"


def qc_unit_count(recday: str) -> int:
    """Number of post-QC single units in the block whose dates match `recday`.

    This is the row count every `Neuron_raw_{recday}_*.npy` must have.
    """
    return int(len(np.load(kilosort_dir(recday) / "QC_single_units.npy")))


def all_recdays() -> list[str]:
    """Every recday the ephys directory can support, from the block directories."""
    return sorted(block_to_recday(m, b) for m in EPHYS_MICE for b in blocks_for(m))


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def _normalise_date(date_str) -> str:
    """Metadata dates are written 2025-6-13; everything else uses 20250613."""
    parts = str(date_str).split("-")
    if len(parts) != 3:
        return str(date_str)
    return f"{parts[0]}{int(parts[1]):02d}{int(parts[2]):02d}"


@lru_cache(maxsize=None)
def task_session_rows(recday: str) -> pd.DataFrame:
    """Metadata rows for a recday's task sessions, in session-index order.

    Rows for the recday's two dates that carry a behaviour timestamp, sorted by
    (date, Behaviour). Row `i` is session `i` -- the same ordering the extraction uses
    to number sessions, and the same one `extract_pokes.sessions_for_mouse` reproduces.
    """
    mouse, date1, date2 = split_recday(recday)
    meta = pd.read_csv(METADATA_DIR / f"MetaData-{mouse}.csv", dtype=str)
    meta = meta.assign(_date=meta["Date"].map(_normalise_date))
    rows = meta[meta["_date"].isin((date1, date2))
                & meta["Behaviour"].notna()
                & (meta["Behaviour"].astype(str) != "-")]
    return rows.sort_values(["_date", "Behaviour"]).reset_index(drop=True)


def pycontrol_file(recday: str, session: int) -> Path:
    """Path to the pyControl .txt for one session of a recday.

    Resolved from the recday's **own two dates**, so a file from another day can never be
    returned: the filename carries the date, and it is checked against the recday name.
    """
    mouse, date1, date2 = split_recday(recday)
    rows = task_session_rows(recday)
    if session >= len(rows):
        raise KeyError(f"{recday} has {len(rows)} task sessions in the metadata, "
                       f"no session {session}")
    row = rows.iloc[session]
    date = str(row["_date"])
    if date not in (date1, date2):
        raise AssertionError(f"{recday} s{session} resolved to date {date}")
    stamp = str(int(float(row["Behaviour"]))).zfill(6)
    path = PYCONTROL_DIR / f"{mouse}-{date[:4]}-{date[4:6]}-{date[6:]}-{stamp}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def pycontrol_active_poke(path: Path) -> tuple[int, ...] | None:
    """The rewarded port sequence from a pyControl file's own `V ... active_poke` line.

    Machine-written by the task, so this is ground truth for what the animal actually ran
    -- unlike the hand-typed `Structure` column of the metadata CSV.
    """
    for line in Path(path).read_text().split("\n"):
        if line.startswith("V ") and "active_poke" in line:
            import ast
            return tuple(ast.literal_eval(line[2:].split(" active_poke ")[1]))
    return None


def metadata_tasks(recday: str) -> list[tuple[int, ...]]:
    """Task sequences for a recday, in session order, as **typed in the metadata CSV**.

    Advisory only -- see `metadata_task_discrepancies`. Use `pycontrol_active_poke` for
    ground truth.
    """
    tasks = []
    for structure in task_session_rows(recday)["Structure"]:
        parts = str(structure).split("-") if isinstance(structure, str) else []
        # Non-ABCD sessions name their condition instead ('config1', 'openfield', ...).
        tasks.append(tuple(int(x) for x in parts) if parts and all(p.isdigit() for p in parts)
                     else ())
    return tasks


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def _first_session(recday_data: dict):
    """The lowest-numbered session dict that actually holds data."""
    for key in sorted(recday_data, key=lambda k: (not isinstance(k, int), k)):
        value = recday_data[key]
        if isinstance(value, dict) and value:
            return key, value
    return None, None


def validate_data_dic(data_dic, *, strict: bool = True, verbose: bool = True) -> list[tuple]:
    """Assert every recday's `Neuron_raw` rows match the QC units of the block it is named after.

    The one assertion that would have caught the ly05 mismatch at extraction time: the
    neural data is binned onto the behavioural timeline, so unit count is the only
    observable that distinguishes one recording day's spikes from another's.

    Returns the list of `(recday, num_neurons, qc_units)` mismatches; raises when
    `strict` and the list is non-empty.
    """
    bad, checked, skipped = [], 0, []
    for recday in sorted(data_dic):
        recday = str(recday)
        _, session = _first_session(data_dic[recday])
        if session is None or "num_neurons" not in session:
            skipped.append(recday)
            continue
        try:
            qc = qc_unit_count(recday)
        except (KeyError, FileNotFoundError):
            skipped.append(recday)
            continue
        checked += 1
        if int(session["num_neurons"]) != qc:
            bad.append((recday, int(session["num_neurons"]), qc))

    if verbose:
        print(f"validate_data_dic: {checked} recdays checked, "
              f"{len(skipped)} skipped (no sorting on disk), "
              f"{len(bad)} mismatched")
        for recday, n, qc in bad:
            print(f"  MISMATCH {recday}: Neuron_raw has {n} rows, "
                  f"{recday_to_block(recday)} has {qc} QC units")
    if strict and bad:
        raise AssertionError(
            "Neuron_raw row count disagrees with the QC unit count of the block each "
            f"recday is named after -- the recdays are mispaired: {bad}")
    return bad


def validate_tasks_against_pycontrol(data_dic, *, strict: bool = True,
                                     verbose: bool = True) -> list[tuple]:
    """Assert every session's stored `Task` matches the pyControl file for that recday's dates.

    The behavioural-side counterpart to `validate_data_dic`, and independent of it. This is
    the test that identified the mispaired recday's behaviour as genuinely 06-18/19 (8 of 8
    sequences matched those dates, 0 of 8 matched 06-20/23), tightened to compare against
    the task's own `active_poke` rather than the hand-typed metadata.

    Returns `(recday, session, stored_task, active_poke)` for every disagreement.
    """
    bad, checked, unresolved = [], 0, []
    for recday in sorted(data_dic):
        recday = str(recday)
        for session in sorted(k for k in data_dic[recday] if isinstance(k, (int, np.integer))):
            stored = data_dic[recday][session].get("Task")
            if stored is None or np.ndim(stored) == 0:
                continue
            stored = tuple(int(x) for x in np.asarray(stored).ravel())
            try:
                active = pycontrol_active_poke(pycontrol_file(recday, session))
            except (KeyError, FileNotFoundError, AssertionError) as exc:
                unresolved.append((recday, session, str(exc)))
                continue
            checked += 1
            if active is not None and stored != tuple(active):
                bad.append((recday, session, stored, tuple(active)))

    if verbose:
        print(f"validate_tasks_against_pycontrol: {checked} sessions checked, "
              f"{len(unresolved)} unresolved, {len(bad)} mismatched")
        for recday, session, stored, active in bad:
            print(f"  MISMATCH {recday} s{session}: stored {stored}, active_poke {active}")
        for recday, session, why in unresolved:
            print(f"  unresolved {recday} s{session}: {why}")
    if strict and bad:
        raise AssertionError(
            "stored task sequences disagree with the pyControl files for the recday's own "
            f"dates -- behaviour is paired to the wrong day: {bad}")
    return bad


def metadata_task_discrepancies(verbose: bool = True) -> pd.DataFrame:
    """Where the hand-typed `Structure` column disagrees with pyControl's `active_poke`.

    Advisory: these are spreadsheet-entry errors, not data errors. Kept as a report rather
    than an assertion so a typo in the CSV cannot fail a good dataset. Known at the time of
    writing: four sessions of `ly06_20250616_20250617` and one of `ly06_20250618_20250619`.
    """
    rows = []
    for recday in all_recdays():
        try:
            typed = metadata_tasks(recday)
        except FileNotFoundError:
            continue
        for session, expected in enumerate(typed):
            try:
                active = pycontrol_active_poke(pycontrol_file(recday, session))
            except (KeyError, FileNotFoundError, AssertionError):
                continue
            if active is not None and expected and tuple(expected) != tuple(active):
                rows.append({"recday": recday, "session": session,
                             "metadata_Structure": "-".join(map(str, expected)),
                             "pycontrol_active_poke": "-".join(map(str, active))})
    table = pd.DataFrame(rows)
    if verbose:
        if len(table):
            print(f"{len(table)} metadata Structure entries disagree with pyControl "
                  f"(the CSV is wrong, the data is right):")
            print(table.to_string(index=False))
        else:
            print("metadata Structure agrees with pyControl everywhere")
    return table


def apply_exclusions(data_dic, *, verbose: bool = True) -> dict:
    """Drop `EXCLUDE_RECDAYS` from a data_dic (or any recday-keyed dict), with a reason."""
    out = {}
    for recday, value in data_dic.items():
        reason = EXCLUDE_RECDAYS.get(str(recday))
        if reason is None:
            out[recday] = value
        elif verbose:
            print(f"  excluding {recday}: {reason}")
    return out


def report(verbose: bool = True) -> pd.DataFrame:
    """Every block on disk with its recday name and QC unit count, for eyeballing."""
    rows = []
    for mouse in EPHYS_MICE:
        for block in blocks_for(mouse):
            recday = block_to_recday(mouse, block)
            try:
                qc = qc_unit_count(recday)
            except FileNotFoundError:
                qc = -1
            rows.append({"mouse": mouse, "block": block, "recday": recday,
                         "qc_units": qc, "excluded": recday in EXCLUDE_RECDAYS})
    table = pd.DataFrame(rows)
    if verbose:
        print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    report()
