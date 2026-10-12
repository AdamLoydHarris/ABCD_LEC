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

#: Which dataset this process works on. `TSA_DATASET=mec` in the environment switches every
#: path and mouse list below to the cohort-7 mEC tree; the CLI scripts set it from their
#: `--dataset` flag before importing this module. Unset means LEC, and every value is then
#: what it was before the switch existed (docs/handoff/MEC_INTEGRATION.md).
DATASET = os.environ.get("TSA_DATASET", "lec")

#: Per-dataset facts that are not derivable from the tree. `qc_gate` records the
#: `get_single_units` thresholds the sorting ran with -- the amplitude threshold differs by
#: probe (NP2 per-shank CAR needed 20; NP1 kept the original 40) -- so a re-gate can be
#: reproduced without reading the merge script's history.
DATASETS = {
    "lec": dict(
        root=Path(__file__).resolve().parents[1],
        mice=("ah08", "ah10", "ly05", "ly06", "ly07"),   # ah09 has histology but no recordings
        probe="Neuropixels 2.0, 4-shank",
        qc_gate=dict(isi_violations_ratio_lt=0.1, amplitude_cutoff_lt=0.1, firing_rate_gt=0.1,
                     presence_ratio_gt=0.9, amplitude_median_abs_gt=20, sd_ratio_lt=3),
    ),
    "mec": dict(
        root=Path("/ceph/behrens/adam_harris/Taskspace_abstraction_mEC"),
        mice=("bp01", "bp02", "bp04"),
        probe="Neuropixels 1.0",
        qc_gate=dict(isi_violations_ratio_lt=0.1, amplitude_cutoff_lt=0.1, firing_rate_gt=0.1,
                     presence_ratio_gt=0.9, amplitude_median_abs_gt=40, sd_ratio_lt=3),
    ),
}
if DATASET not in DATASETS:
    raise KeyError(f"TSA_DATASET={DATASET!r}; known datasets: {sorted(DATASETS)}")

REPO_ROOT = DATASETS[DATASET]["root"]
EPHYS_ROOT = REPO_ROOT / "data" / "preprocessed" / "ephys"
METADATA_DIR = REPO_ROOT / "data" / "metadata"
PYCONTROL_DIR = REPO_ROOT / "data" / "raw" / "pycontrol"

#: Mice with ephys, for the active dataset.
EPHYS_MICE = DATASETS[DATASET]["mice"]
QC_GATE = DATASETS[DATASET]["qc_gate"]

#: Which build of the data dictionary an analysis loads. v4 (2026-10-02) is the corrected
#: spike clock: v1 pairs each session against a raw ProbeB TTL train whatever the mouse's own
#: probe, which leaves ly05_20250620_20250623 off by 2.6-38.6 s and every other recday off by
#: 0-18 bins (docs/handoff/SPIKE_CLOCK_V4.md section 1.5). v4 is therefore the default, and
#: 'v1' is kept reachable only so a refit can be proved equivalent to what is on disk.
#: TSA_DATA_VERSION overrides for a whole process.
DATA_VERSION = os.environ.get("TSA_DATA_VERSION") or "v4"   # empty env var == unset

#: The v1 pickles carry no tag, so 'v1' maps to the untagged name and every other tag suffixes.
_UNTAGGED_VERSION = "v1"


def version_suffix(version=None) -> str:
    """'' for v1, '_v4' for v4. The one place the output-tree tagging rule lives, so a cache
    directory and the pickle it was built from can never disagree."""
    version = DATA_VERSION if version is None else version
    return "" if version == _UNTAGGED_VERSION else f"_{version}"


def data_dic_path(version=None, dataset=None) -> Path:
    """Path to a data_dic pickle. `version=None` takes DATA_VERSION; 'v1' is the untagged file."""
    version = DATA_VERSION if version is None else version
    dataset = DATASET if dataset is None else dataset
    root = DATASETS[dataset]["root"] if dataset in DATASETS else REPO_ROOT
    tag = "" if version == _UNTAGGED_VERSION else f"_{version}"
    return root / "data" / "processed_data" / f"data_dic_{dataset}{tag}.pkl"


def norm_neurons_path(version=None) -> Path:
    """Path to the matching norm_neurons_dic pickle (same tagging rule)."""
    version = DATA_VERSION if version is None else version
    tag = "" if version == _UNTAGGED_VERSION else f"_{version}"
    return REPO_ROOT / "data" / "processed_data" / f"norm_neurons_dic{tag}.pkl"


def session_inds_path(version=None) -> Path:
    """Path to the matching session_inds_dic pickle (same tagging rule)."""
    version = DATA_VERSION if version is None else version
    tag = "" if version == _UNTAGGED_VERSION else f"_{version}"
    return REPO_ROOT / "data" / "processed_data" / f"session_inds_dic{tag}.pkl"


def recdays_for_version(version=None) -> list[str]:
    """Recdays a data version contains, from its 1 KB session_inds_dic rather than the 3.8 GB
    dict. Built by the same `build_data_dic.py` run, so it cannot drift from it, and it picks
    up July automatically at v5 where `all_recdays()` already lists unsorted blocks."""
    import pickle
    with open(session_inds_path(version), "rb") as fh:
        return sorted(pickle.load(fh))


def add_data_version_arg(parser):
    """`--data-version` on an entry point, defaulting to the registry (currently v4)."""
    parser.add_argument('--data-version', default=None,
                        help=f"data_dic build to fit; default {DATA_VERSION} "
                             f"(TSA_DATA_VERSION). 'v1' is the pre-clock-fix untagged file.")
    return parser


def resolve_data_version(version=None) -> tuple[str, Path]:
    """(version, path), raising if the pickle is absent -- a typo must not silently fall back."""
    version = DATA_VERSION if version is None else version
    path = data_dic_path(version)
    if not path.exists():
        raise FileNotFoundError(
            f"data_dic version {version!r} -> {path} does not exist. "
            f"Available: {sorted(q.name for q in path.parent.glob('data_dic_*.pkl'))}")
    return version, path


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
#: configuration into it as `{regset}_{width}ms_{scheme}`, optionally followed by a V3 arm
#: token (`_cap30s_tfrU10b`, `_cap60s`; see `w1_refit.arm_extra`). A section carrying that
#: stamp postdates the fix, so the stale list must not apply to it. The optional group is
#: what keeps the V3 fits from silently losing `ly05_20250618_20250619` at load time -- the
#: regex used to be anchored right after the scheme.
_POST_REFIT_SECTION = re.compile(r'__(full|matched)_\d+ms_(decile|uniform)(_[A-Za-z0-9_.]+)?$')


def is_post_refit_section(section_name) -> bool:
    """True if `section_name` was produced by the W1 production refit or a V3 arm.

    Used to stop `STALE_CACHE_RECDAYS` from dropping good recdays out of the new fits while
    still protecting every pre-refit cache that genuinely holds the wrong day's spikes.
    """
    return bool(section_name and _POST_REFIT_SECTION.search(str(section_name)))


# Every name the production and V3 fits are written under must match, and the pre-refit
# names must not -- checked at import so a naming change cannot silently re-arm the stale list.
for _name in ('all_regressors__full_250ms_decile', 'all_regressors__matched_250ms_decile',
              'core_progress_time__matched_250ms_decile_cap30s_tfrU10b',
              'core_progress_time__matched_250ms_decile_cap60s_tfrD10b',
              'core_progress_only__matched_250ms_decile_cap30s',
              'core_progress_only__matched_250ms_decile_cap60s'):
    assert is_post_refit_section(_name), _name
for _name in ('all_regressors', 'distance_gp_state_filtered', 'pokes_filtered'):
    assert not is_post_refit_section(_name), _name
del _name


# ---------------------------------------------------------------------------
# recday <-> block, by date
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def blocks_for(mouse: str) -> tuple[str, ...]:
    """Sorted recording-block directory names for a mouse."""
    return tuple(sorted(os.path.basename(p)
                        for p in glob.glob(str(EPHYS_ROOT / mouse / "*_preprocessed"))))


def block_to_recday(mouse: str, block: str) -> str:
    """``ah08`` + ``2025-06-13_2025-06-15_preprocessed`` -> ``ah08_20250613_20250615``.

    A single-day block (``2024-03-21_preprocessed``, mEC only) gives ``bp01_20240321``.
    """
    dates = re.findall(r"(\d{4})-(\d{2})-(\d{2})", block)
    return f"{mouse}_" + "_".join("".join(d) for d in dates)


#: Most dates one sort may concatenate. The July 2025 ABCDE week is sorted as one 6-day block
#: so that the interleaved ABCD and ABCDE sessions share a unit set; 2-day blocks are still
#: the June default. A cap is kept so a malformed name still raises instead of parsing.
MAX_RECDAY_DATES = 6


def recday_dates(recday: str) -> tuple[str, tuple[str, ...]]:
    """``ly05_20250618_20250619`` -> ``('ly05', ('20250618', '20250619'))``.

    One to `MAX_RECDAY_DATES` dates. The mEC tree has both 2-day merged sorts and single-day
    sorts of the same recordings; LEC June is 2-day, LEC July is one 6-day block. Anything
    that builds a path or a filter from a recday's dates should go through this rather than
    `split_recday`.
    """
    mouse, *dates = str(recday).split("_")
    if (not 1 <= len(dates) <= MAX_RECDAY_DATES
            or not all(re.fullmatch(r"\d{8}", d) for d in dates)):
        raise ValueError(f"not a recday name: {recday!r}")
    return mouse, tuple(dates)


def recday_kind(recday: str) -> str:
    """``'merged'`` for a multi-date recday, ``'single'`` for a one-date recday.

    The two kinds hold the same physical units twice, so a pooled analysis must pick one.
    A 6-day July block is `merged` like a 2-day June one: both are multi-day sorts, and no
    single-day sort of the same recordings exists to collide with.
    """
    return "single" if len(recday_dates(recday)[1]) == 1 else "merged"


def split_recday(recday: str) -> tuple[str, str, str]:
    """``ly05_20250618_20250619`` -> ``('ly05', '20250618', '20250619')``.

    Two-date recdays only; raises on any other length so a caller that assumes two dates
    fails loudly instead of building a path from a missing one. Use `recday_dates`.
    """
    mouse, dates = recday_dates(recday)
    if len(dates) != 2:
        raise ValueError(f"{recday} has {len(dates)} date(s), not 2; use recday_dates()")
    return mouse, dates[0], dates[1]


def recday_to_block(recday: str) -> str:
    """Block directory name whose dates match `recday`.

    Raises `KeyError` if no block matches -- never falls back to a positional guess.
    """
    mouse, _ = recday_dates(recday)
    for block in blocks_for(mouse):
        if block_to_recday(mouse, block) == str(recday):
            return block
    raise KeyError(f"no sorted block matches {recday} "
                   f"(mouse has {blocks_for(mouse)})")


def kilosort_dir(recday: str) -> Path:
    """`kilosort_output` directory for the block whose dates match `recday`."""
    mouse, _ = recday_dates(recday)
    return EPHYS_ROOT / mouse / recday_to_block(recday) / "kilosort_output"


def qc_unit_count(recday: str) -> int:
    """Number of post-QC single units in the block whose dates match `recday`.

    This is the row count every `Neuron_raw_{recday}_*.npy` must have.
    """
    return int(len(np.load(kilosort_dir(recday) / "QC_single_units.npy")))


def all_recdays(kind: str | None = None) -> list[str]:
    """Every recday the ephys directory can support, from the block directories.

    `kind` restricts to ``'merged'`` or ``'single'`` recdays; the default (None) keeps
    every block on disk, which for LEC is identical to the pre-switch behaviour.
    """
    recdays = sorted(block_to_recday(m, b) for m in EPHYS_MICE for b in blocks_for(m))
    if kind is None:
        return recdays
    if kind not in ("merged", "single"):
        raise ValueError(f"kind must be 'merged', 'single' or None, not {kind!r}")
    return [r for r in recdays if recday_kind(r) == kind]


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
    mouse, dates = recday_dates(recday)
    meta = pd.read_csv(METADATA_DIR / f"MetaData-{mouse}.csv", dtype=str)
    meta = meta.assign(_date=meta["Date"].map(_normalise_date))
    rows = meta[meta["_date"].isin(dates)
                & meta["Behaviour"].notna()
                & (meta["Behaviour"].astype(str) != "-")]
    return rows.sort_values(["_date", "Behaviour"]).reset_index(drop=True)


#: Sheet `Structure_abstract` spellings -> the session types the pipeline distinguishes.
_SESSION_TYPES = {"abcd": "ABCD", "abcde": "ABCDE", "abcd_ot": "ABCD_ot",
                  "object": "object", "objectvector": "object", "objects": "object",
                  "openfield": "openfield", "open_field": "openfield"}


def session_type(recday: str, session: int) -> str:
    """``'ABCD' | 'ABCDE' | 'object' | 'openfield' | ...`` from the sheet's `Structure_abstract`.

    LEC object sessions are typed `Object` and run the ABCD task file with no A_on; mEC ones are
    `objectvector` on the pyControl-2 open-field task. Both become `'object'`. Unknown spellings
    are returned stripped, so a caller can decide rather than crash.
    """
    raw = str(task_session_rows(recday).iloc[session]["Structure_abstract"]).strip()
    return _SESSION_TYPES.get(raw.lower(), raw)


def pycontrol_file(recday: str, session: int) -> Path:
    """Path to the pyControl file for one session of a recday.

    Resolved from the recday's **own dates**, so a file from another day can never be
    returned: the filename carries the date, and it is checked against the recday name.
    ABCD sessions are pyControl-1 ``.txt``; the mEC open-field sessions are pyControl-2
    ``.tsv`` with the same stem, tried second. LEC has ``.txt`` only, so its resolution is
    unchanged.
    """
    mouse, dates = recday_dates(recday)
    rows = task_session_rows(recday)
    if session >= len(rows):
        raise KeyError(f"{recday} has {len(rows)} task sessions in the metadata, "
                       f"no session {session}")
    row = rows.iloc[session]
    date = str(row["_date"])
    if date not in dates:
        raise AssertionError(f"{recday} s{session} resolved to date {date}")
    stamp = str(int(float(row["Behaviour"]))).zfill(6)
    day = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    stem = PYCONTROL_DIR / f"{mouse}-{day}-{stamp}"
    for suffix in (".txt", ".tsv"):
        if stem.with_suffix(suffix).exists():
            return stem.with_suffix(suffix)
    # The stamp is hand-typed. Five mEC sessions have it 1-20 s off the real filename; LEC
    # has none. Accept the UNIQUE file of that mouse on that same date within 60 s, so the
    # date guarantee above still holds and a typo cannot pull in a neighbouring session.
    near = [p for p in PYCONTROL_DIR.glob(f"{mouse}-{day}-??????.t??")
            if p.suffix in (".txt", ".tsv")
            and abs(_stamp_seconds(p.stem[-6:]) - _stamp_seconds(stamp)) <= STAMP_TOLERANCE_S]
    if len(near) == 1:
        return near[0]
    raise FileNotFoundError(f"{stem}.txt|.tsv ({len(near)} files within {STAMP_TOLERANCE_S}s)")


#: Seconds of hand-typed Behaviour-stamp error tolerated when the exact file is absent.
STAMP_TOLERANCE_S = 60


def match_ephys_stamp(recday: str, session: int, candidates) -> str | None:
    """The concat recording time (``HH-MM-SS``) that session `session`'s sheet row names.

    Exact match first; otherwise the UNIQUE candidate within `STAMP_TOLERANCE_S` on the same
    date (bp02 2024-03-28 is typed ``14-30-34`` for the folder ``14-30-43``). `candidates` is
    an iterable of (date_YYYYMMDD, time_HH-MM-SS) pairs, e.g. from
    `recording_sessions_in_concat.csv`. Returns None when nothing qualifies.
    """
    row = task_session_rows(recday).iloc[session]
    date, stamp = str(row["_date"]), str(row["Ephys"])
    cands = [t for d, t in candidates if str(d) == date]
    if stamp in cands:
        return stamp
    try:
        want = _stamp_seconds(stamp.replace("-", ""))
    except (ValueError, IndexError):
        return None
    near = [t for t in cands if abs(_stamp_seconds(t.replace("-", "")) - want) <= STAMP_TOLERANCE_S]
    return near[0] if len(near) == 1 else None


def _stamp_seconds(hhmmss: str) -> int:
    return int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6])


def metadata_stamp_discrepancies(verbose: bool = True) -> pd.DataFrame:
    """Sessions whose metadata `Behaviour` stamp is not the pyControl filename's stamp.

    Advisory, like `metadata_task_discrepancies`: the file was found by the 60 s rule and
    the data is right; the spreadsheet is off by the seconds shown.
    """
    rows = []
    for recday in all_recdays("merged") or all_recdays():
        table = task_session_rows(recday)
        for session, row in table.iterrows():
            try:
                path = pycontrol_file(recday, session)
            except (FileNotFoundError, KeyError, AssertionError):
                continue
            typed = str(int(float(row["Behaviour"]))).zfill(6)
            if path.stem[-6:] != typed:
                rows.append({"recday": recday, "session": session, "metadata_Behaviour": typed,
                             "pycontrol_stamp": path.stem[-6:],
                             "offset_s": _stamp_seconds(path.stem[-6:]) - _stamp_seconds(typed)})
    table = pd.DataFrame(rows)
    if verbose:
        print(f"{len(table)} Behaviour stamps resolved by the {STAMP_TOLERANCE_S}s rule"
              + (":" if len(table) else ""))
        if len(table):
            print(table.to_string(index=False))
    return table


def pycontrol_active_poke(path: Path) -> tuple[int, ...] | None:
    """The rewarded port sequence from a pyControl file's own `active_poke` variable.

    Machine-written by the task, so this is ground truth for what the animal actually ran
    -- unlike the hand-typed `Structure` column of the metadata CSV. pyControl-1 ``.txt``
    files carry it as a ``V <t> active_poke [..]`` line; pyControl-2 ``.tsv`` files as a
    ``variable`` row whose JSON content has an ``active_poke`` key (the mEC open-field task
    has none, so those return None).
    """
    import ast
    import json
    path = Path(path)
    if path.suffix == ".tsv":
        for line in path.read_text().split("\n"):
            parts = line.split("\t")
            if len(parts) == 4 and parts[1] == "variable" and "active_poke" in parts[3]:
                value = json.loads(parts[3]).get("active_poke")
                return tuple(int(x) for x in value) if value is not None else None
        return None
    for line in path.read_text().split("\n"):
        if line.startswith("V ") and "active_poke" in line:
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
