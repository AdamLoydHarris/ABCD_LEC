#!/usr/bin/env python3
"""Write the session-bookkeeping arrays the OSF deposit omits, into the writable mirror.

None of these families were deposited, and no deposited notebook writes them:

    awake_session_<rd>.npy             All_session_<rd>.npy
    awake_session_behaviour_<rd>.npy   All_session_behaviour_<rd>.npy
    sleep_session_<rd>.npy             Num_trials_<rd>.npy
    Task_num_<rd>.npy
    3_task_days.npy  3_task_all_days.npy  combined_days.npy

WHY NOT FROM THE DEPOSITED DICTS: `REPLICATION_STATUS.md` §3(i) proposes recovering these
from `session_dic` / `Variable_dic`. Measured: direct lookup by combined recday name hits
only 3 of 25 recdays (those dicts are keyed by SINGLE-day names), and mice ab03 and ah07 --
6 of the 25 recdays, the whole 2023 cohort -- are absent from both dicts entirely. The
filesystem plus the MetaData CSVs cover all of them, and every step here is gated.

THE RULE THAT MAKES THIS DETERMINATE: every bookkeeping array is indexed by *Task_data_
row*, and a combined recday is the contiguous concatenation of its two single days.
Verified: len(Task_data_<combined>) == len(Task_data_<d1>) + len(Task_data_<d2>) for all
29 combined recdays, and the prefix split predicts exactly the session indices missing
from El-Gaby's own cell-20 stderr (ah07_27082023_28082023 idx 7, me08_12092021_13092021
idx 2, me10_14122021_15122021 idx 5 and 6).

TIMESTAMPS ARE REAL, NOT PLACEHOLDERS. Figure2/3/5 only ever take len() of the session
arrays, but Figure7 cells 20/24/28 do `np.where(All_sessions==timestamp)[0][0]` to index
`binned_FR_dic_<rd>_<idx>`, so the values must be genuine and mutually consistent. They
come from `MetaData.xlsx - <mouse>.csv` (awake) and `- <mouse>_sleep.csv` (sleep).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
MIRROR = os.path.join(ROOT, 'replication_run', 'data', 'Intermediate_objects')
LOGDIR = os.path.join(ROOT, 'replication_run', 'logs')

# The 11 `3_task` recdays, recovered from the stored outputs of Figure2 cells 46/48.
THREE_TASK = [
    'ah03_18082021', 'ah04_06122021', 'ah04_10122021', 'ah04_26112021', 'ah04_30112021',
    'me10_08122021', 'me10_16122021', 'me11_05122021', 'me11_07122021', 'me11_08122021',
    'me11_30112021',
]

TS_RE = re.compile(r'^\d{2}-\d{2}-\d{2}$')
EXCEL_DATE_RE = re.compile(r'^(\d{1,2})/(\d{1,2})/(\d{4})$')


# ------------------------------------------------------------------ Excel de-mangling
def repair_ephys(val):
    """Undo Excel's coercion of an `HH-MM-SS` timestamp into a date.

    Excel read `17-04-13` as 17 April 2013 and rewrote it `17/04/2013`. The transform is
    deterministic and invertible: D/M/YYYY -> f'{D:02d}-{M:02d}-{YYYY%100:02d}'.
    Validated against the sibling `Session_time` column (e.g. Session_time 170422 pairs
    with a repaired Ephys of 17-04-13).

    Returns (repaired_or_original, was_repaired, is_valid).
    """
    v = (val or '').strip()
    if TS_RE.match(v):
        return v, False, True
    m = EXCEL_DATE_RE.match(v)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f'{d:02d}-{mo:02d}-{y % 100:02d}', True, True
    return v, False, False


def ts_key(date_ddmmyyyy, hhmmss):
    """Sort key: (date, time) -> chronological across the two days of a recday."""
    d = date_ddmmyyyy
    iso = f'{d[4:8]}{d[2:4]}{d[0:2]}'          # ddmmyyyy -> yyyymmdd
    return iso + hhmmss.replace('-', '')


# ------------------------------------------------------------------------ CSV loading
def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


_csv_cache: dict = {}


def mouse_rows(mouse, sleep=False):
    key = (mouse, sleep)
    if key not in _csv_cache:
        name = f'MetaData.xlsx - {mouse}{"_sleep" if sleep else ""}.csv'
        _csv_cache[key] = load_csv(os.path.join(MIRROR, name))
    return _csv_cache[key]


def csv_date(ddmmyyyy):
    """ddmmyyyy -> the CSV's D/M/YYYY form, matching however the sheet wrote it."""
    d, m, y = ddmmyyyy[0:2], ddmmyyyy[2:4], ddmmyyyy[4:8]
    return {f'{d}/{m}/{y}', f'{int(d)}/{int(m)}/{y}'}


def sessions_for(mouse, dates, sleep=False):
    """Ephys timestamps for one mouse on the given dates, in CSV (=chronological) order.

    Returns list of (date_ddmmyyyy, timestamp, structure_string, n_repaired, n_bad).
    """
    rows = mouse_rows(mouse, sleep=sleep)
    if rows is None:
        return None
    out, nrep, nbad = [], 0, 0
    for d in dates:                                    # preserve day order
        wanted = csv_date(d)
        for r in rows:
            if (r.get('Date') or '').strip() not in wanted:
                continue
            ts, rep, ok = repair_ephys(r.get('Ephys'))
            nrep += rep
            if not ok:
                nbad += 1
                continue
            out.append((d, ts, (r.get('Structure') or '').strip()))
    return out, nrep, nbad


# ---------------------------------------------------------------------- file scanning
_listing = None


def listing():
    global _listing
    if _listing is None:
        _listing = set(os.listdir(MIRROR))
    return _listing


def sess_ids(rd, prefix, ext='.npy'):
    """Session indices present on disk for one family.

    Anchored + re.escape so `Neuron_raw_ab03_01092023_0.npy` (a single day) cannot match
    the combined recday `ab03_01092023_02092023`. Reuses the pattern established at
    mFC_data/code/glm_analysis_v3.py:197-201. Indices are NOT contiguous.
    """
    pat = re.compile(rf'^{re.escape(prefix)}{re.escape(rd)}_(\d+){re.escape(ext)}$')
    return sorted(int(m.group(1)) for m in (pat.match(f) for f in listing()) if m)


def task_data(rd):
    p = os.path.join(MIRROR, f'Task_data_{rd}.npy')
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)


def split_recday(rd):
    """`ah04_01122021_02122021` -> ('ah04', ['01122021', '02122021']); single -> 1 date."""
    parts = rd.split('_')
    return parts[0], parts[1:]


# ------------------------------------------------------------------------------- main
def main():
    os.makedirs(LOGDIR, exist_ok=True)
    if not os.path.isdir(MIRROR):
        sys.exit(f'FATAL: mirror not found at {MIRROR} -- run 00_build_mirror.sh first')
    # Refuse to write into the deposit, whatever happens.
    assert os.path.realpath(MIRROR) != os.path.realpath(
        os.path.join(ROOT, 'data', 'Intermediate_objects')), 'MIRROR == DEPOSIT'

    comb_abcd = [str(x) for x in np.load(os.path.join(MIRROR, 'combined_ABCDonly_days.npy'))]
    comb_abcde = [str(x) for x in np.load(os.path.join(MIRROR, 'combined_ABCDE_days.npy'))]
    singles = [str(x) for x in np.load(os.path.join(MIRROR, 'single_ABCDonly_days.npy'))]

    # Recday universe: everything any cohort loop can reach.
    universe = OrderedDict()
    for rd in comb_abcd + comb_abcde + singles + THREE_TASK:
        universe[rd] = None
    for rd in comb_abcd + comb_abcde:                  # single-day constituents
        mouse, dates = split_recday(rd)
        for d in dates:
            universe.setdefault(f'{mouse}_{d}', None)
    universe = list(universe)
    print(f'recday universe: {len(universe)} '
          f'({len(comb_abcd)} ABCDonly + {len(comb_abcde)} ABCDE + {len(singles)} single '
          f'+ constituents/3_task)')

    manifest, problems, notes = {}, [], []
    nt_source = {}
    n_written = 0
    sleep_ok, sleep_bad = [], []

    # ---- Gate A: combined == d1 + d2, for every combined recday
    print('\n=== Gate A: len(Task_data_<combined>) == len(d1) + len(d2) ===')
    for rd in comb_abcd + comb_abcde:
        mouse, dates = split_recday(rd)
        a = task_data(rd)
        halves = [task_data(f'{mouse}_{d}') for d in dates]
        if a is None or any(h is None for h in halves):
            problems.append(f'{rd}: missing Task_data for combined or a constituent')
            continue
        if len(a) != sum(len(h) for h in halves):
            problems.append(f'GATE A FAIL {rd}: {len(a)} != '
                            f'{"+".join(str(len(h)) for h in halves)}')
    print(f'  {"FAIL" if any("GATE A" in p for p in problems) else "PASS"} '
          f'({len(comb_abcd) + len(comb_abcde)} combined recdays checked)')

    # ---- per-recday bookkeeping
    print('\n=== writing per-recday bookkeeping ===')
    hdr = (f'{"recday":24s} {"rows":>4s} {"trialt":>6s} {"awake":>5s} {"sleep":>5s} '
           f'{"All":>4s} {"bFR":>4s} {"struct":>6s}')
    print(hdr)
    for rd in universe:
        T = task_data(rd)
        if T is None:
            problems.append(f'{rd}: no Task_data_ -- skipped, nothing fabricated')
            continue
        rows = len(T)
        mouse, dates = split_recday(rd)

        # Num_trials_: CONTENTS MATTER. non_repeat_ses_maker does
        #   np.logical_and(non_repeat_bool_all, num_trials_day > 0)
        # over range(len(Task_data_)), so this must be one entry per Task_data_ row:
        # the trial count from trialtimes_<rd>_<s>.npy where present, 0 where absent.
        num_trials = np.zeros(rows, dtype=int)
        tt_found = 0
        for s in range(rows):
            p = os.path.join(MIRROR, f'trialtimes_{rd}_{s}.npy')
            if os.path.exists(p):
                num_trials[s] = len(np.load(p, allow_pickle=True))
                tt_found += 1

        # PREFER HIS OWN COUNTS where the deposit has them. `Num_trials_dic2` is a
        # joblib dict nested [recday][session] and it does hold real per-session trial
        # counts -- it covers 14 of our 36 GLM recdays. His values run 0-3 higher than
        # len(trialtimes) (e.g. ah03_18082021 his [32,52,54,37] vs rows [31,49,52,37]),
        # so he counted trials slightly differently from complete trialtimes rows.
        #
        # Only `> 0` is ever consumed (`non_repeat_ses_maker` does
        # np.logical_and(non_repeat_bool_all, num_trials_day > 0); nothing reads the
        # magnitude), so the practical effect is exactly one session: he assigns
        # ah04_05122021_06122021 session 3 zero trials and therefore drops that fold,
        # where a trialtimes-derived count keeps it. Adopting his numbers makes our
        # session selection identical to his everywhere the deposit lets us check.
        #
        # This leaves a MIXED basis -- his values for the recdays he covers, derived
        # counts for the other 22 -- which is recorded per recday in the manifest as
        # num_trials_source.
        src_nt = 'derived:trialtimes'
        his = _his_num_trials(rd, rows)
        if his is not None:
            n_ch = int(np.sum((his > 0) != (num_trials > 0)))
            if n_ch:
                notes.append(f'{rd}: Num_trials_ >0 mask changed for {n_ch} session(s) '
                             f'by adopting Num_trials_dic2 (derived {list(num_trials)} '
                             f'-> his {[int(x) for x in his]})')
            num_trials = his.astype(int)
            src_nt = 'deposit:Num_trials_dic2'
        nt_source[rd] = src_nt

        # awake / sleep / All from the CSVs, with the Excel repair.
        aw = sessions_for(mouse, dates, sleep=False)
        sl = sessions_for(mouse, dates, sleep=True)
        struct_match = ''
        awake_ts = None

        if aw is not None:
            arows, nrep, nbad = aw
            if len(arows) == rows:
                awake_ts = [t for (_, t, _) in arows]
                # Cross-check the alignment: the CSV `Structure` column should reproduce
                # Task_data_ row for row. A match is the strongest evidence that CSV row
                # order IS Task_data row order.
                #
                # A per-row DISAGREEMENT is NOT a disqualifier. Only the Ephys timestamp
                # is taken from the CSV, and the row count and date ordering still line
                # up, so the timestamps stay correctly positioned; the mismatch is a
                # spreadsheet-vs-Task_data content discrepancy in the deposit (e.g.
                # ah04_07122021_08122021 row 3 is labelled as a repeat of row 0 but
                # Task_data repeats row 2; me10_09122021_10122021 rows 0/3 read 3-5-7-9
                # where Task_data has 5-3-7-9 -- the first two rewards transposed).
                # The real disqualifier is a COUNT mismatch, handled in the else branch.
                got = ['-'.join(str(int(v)) for v in row) for row in T]
                exp = [s for (_, _, s) in arows]
                if not any(exp):
                    struct_match = 'n/a'          # e.g. ah03 has no Structure column
                elif got == exp:
                    struct_match = 'OK'
                else:
                    nd = sum(g != e for g, e in zip(got, exp))
                    struct_match = f'DIFF{nd}'
                    notes.append(
                        f'{rd}: CSV Structure disagrees with Task_data in {nd}/{rows} '
                        f'row(s) (count and order still align; timestamps unaffected)')
            else:
                notes.append(f'{rd}: CSV awake rows {len(arows)} != Task_data rows {rows} '
                             f'-- cannot align timestamps, falling back to synthetic')
            if nrep:
                notes.append(f'{rd}: repaired {nrep} Excel-mangled awake Ephys value(s)')
            if nbad:
                notes.append(f'{rd}: {nbad} unparseable awake Ephys value(s)')

        if awake_ts is None:
            awake_ts = _synthetic_awake(rd, mouse, dates, rows)
            notes.append(f'{rd}: awake_session_ is SYNTHETIC (len-only; not valid for Figure7)')

        sleep_ts, all_ts = [], None
        if sl is not None and awake_ts and TS_RE.match(str(awake_ts[0])):
            srows, snrep, snbad = sl
            sleep_ts = [t for (_, t, _) in srows]
            if snrep:
                notes.append(f'{rd}: repaired {snrep} Excel-mangled sleep Ephys value(s)')
            merged = sorted(
                [(ts_key(d, t), t) for (d, t, _) in srows] +
                [(ts_key(d, t), t) for (d, t, _) in aw[0]])
            all_ts = [t for _, t in merged]

        nbfr = len(sess_ids(rd, 'binned_FR_dic_', ext=''))

        # ---- Figure7 gate. Cells 24/28 do np.where(All_sessions==timestamp)[0][0] and
        # use the result to index binned_FR_dic_<rd>_<idx>, so len(All_session_) MUST
        # equal the binned_FR_dic file count or every index past a missing session is
        # silently shifted -- a wrong file read as if it were the right one.
        #
        # When the counts disagree we therefore REFUSE to publish real timestamps and
        # fall back to synthetic ones, which makes Figure7's np.where miss and its
        # existing try/except skip the session ("File not found for session") instead of
        # reading a shifted file. A clean skip beats a plausible wrong answer.
        #
        # The four ABCDonly recdays this affects, and why (each verified by hand):
        #   ab03_29082023_30082023  10 CSV awake rows vs 9 Task_data rows (one aborted
        #                           and restarted session is in the sheet, not the data)
        #   me08_10092021_11092021  8 vs 6
        #   me10_20122021_21122021  10 vs 6; the sheet's Structure carries an '-ot'
        #                           suffix (one-tone variants) and one row has Ephys '-'
        #   me10_14122021_15122021  counts align (8 == 8) but All=20 vs binned_FR=19:
        #                           a session with behaviour but no neural data. THIS is
        #                           the dangerous one -- it would look fine and shift.
        # Reconstructing his session-inclusion rule is not possible from the deposit, and
        # guessing it is exactly how a silent misindex would get introduced.
        if nbfr and all_ts is not None and len(all_ts) != nbfr:
            notes.append(f'{rd}: All_session_ {len(all_ts)} != binned_FR_dic {nbfr} '
                         f'-- refusing real timestamps; Figure7 sleep chain excluded')
            all_ts, sleep_ts = None, []
            awake_ts = _synthetic_awake(rd, mouse, dates, rows)

        if all_ts is None:
            all_ts = list(awake_ts)
        if nbfr:
            (sleep_ok if len(all_ts) == nbfr else sleep_bad).append(rd)

        out = {
            f'Num_trials_{rd}.npy': num_trials,
            f'awake_session_{rd}.npy': np.array(awake_ts, dtype='<U32'),
            f'awake_session_behaviour_{rd}.npy': np.array(awake_ts, dtype='<U32'),
            f'All_session_{rd}.npy': np.array(all_ts, dtype='<U32'),
            f'All_session_behaviour_{rd}.npy': np.array(all_ts, dtype='<U32'),
            f'sleep_session_{rd}.npy': np.array(sleep_ts, dtype='<U32'),
            # Task_num_: only its length is ever consumed (cells compute `sessions`,
            # `num_refses`, `num_comparisons`, `repeat_ses` and never read them again).
            # Integer-label each distinct Task_data_ row in first-occurrence order.
            f'Task_num_{rd}.npy': _task_num(T),
        }
        for fn, arr in out.items():
            np.save(os.path.join(MIRROR, fn), arr)
            manifest[fn] = dict(shape=list(np.shape(arr)), dtype=str(np.asarray(arr).dtype))
            n_written += 1

        # Hard gates
        assert len(num_trials) == rows, f'{rd}: Num_trials_ length != Task_data rows'
        assert len(out[f'awake_session_{rd}.npy']) == rows, f'{rd}: awake_session_ length'
        if src_nt.startswith('derived'):
            assert (num_trials > 0).sum() == tt_found, \
                f'{rd}: Num_trials_>0 != trialtimes found'

        print(f'{rd:24s} {rows:>4d} {tt_found:>6d} {len(awake_ts):>5d} {len(sleep_ts):>5d} '
              f'{len(all_ts):>4d} {nbfr:>4d} {struct_match:>6s}')

    # ---- day lists
    print('\n=== day lists ===')
    three_all = np.load(os.path.join(MIRROR, 'single_ABCDonly_days.npy'))
    np.save(os.path.join(MIRROR, '3_task_all_days.npy'), three_all)
    print(f'  3_task_all_days.npy   = single_ABCDonly_days ({len(three_all)})')

    missing = sorted(set(THREE_TASK) - set(singles))
    assert not missing, f'3_task names not in single_ABCDonly_days: {missing}'
    np.save(os.path.join(MIRROR, '3_task_days.npy'), np.array(THREE_TASK, dtype=three_all.dtype))
    print(f'  3_task_days.npy       = {len(THREE_TASK)} recovered names (all in singles: OK)')

    comb = np.load(os.path.join(MIRROR, 'combined_ABCDonly_days.npy'))
    np.save(os.path.join(MIRROR, 'combined_days.npy'), comb)
    print(f'  combined_days.npy     = combined_ABCDonly_days ({len(comb)})')

    # ---- report
    print(f'\n=== summary ===')
    print(f'  arrays written        : {n_written}')
    print(f'  recdays with Task_data: {len([r for r in universe if task_data(r) is not None])}'
          f'/{len(universe)}')
    print(f'  Figure7 All_session gate: {len(sleep_ok)} pass, {len(sleep_bad)} fail')
    if sleep_bad:
        print(f'    FAILING (sleep chain not valid for these): {sleep_bad}')
    if problems:
        print(f'\n  PROBLEMS ({len(problems)}):')
        for p in problems:
            print(f'    {p}')
    if notes:
        print(f'\n  NOTES ({len(notes)}):')
        for p in notes[:40]:
            print(f'    {p}')
        if len(notes) > 40:
            print(f'    ... and {len(notes) - 40} more (see the JSON report)')

    with open(os.path.join(LOGDIR, 'preflight_manifest.json'), 'w') as f:
        json.dump(dict(manifest=manifest, problems=problems, notes=notes,
                       num_trials_source=nt_source,
                       figure7_all_session_pass=sleep_ok,
                       figure7_all_session_fail=sleep_bad), f, indent=2)
    print(f'\nwrote {LOGDIR}/preflight_manifest.json')
    return 1 if any('GATE A FAIL' in p for p in problems) else 0


_num_trials_dic2 = None


def _his_num_trials(rd, rows):
    """His per-session trial counts from the deposited `Num_trials_dic2`, or None.

    The dict is a joblib dump of defaultdict(rec_dd) nested [recday][session], so it
    needs `rec_dd` importable AND a stub for `mBaseFunctions`, the module it was pickled
    from, which the deposit does not ship. Absent keys auto-vivify to an empty
    defaultdict rather than raising, so an empty/short entry must be rejected explicitly.
    """
    global _num_trials_dic2
    if _num_trials_dic2 is None:
        try:
            from joblib import load as _jl
            import types as _types

            def rec_dd():
                return defaultdict(rec_dd)

            sys.modules.setdefault('mBaseFunctions', _types.ModuleType('mBaseFunctions'))
            sys.modules['mBaseFunctions'].rec_dd = rec_dd
            import __main__ as _m
            _m.rec_dd = rec_dd
            _num_trials_dic2 = _jl(os.path.join(MIRROR, 'Num_trials_dic2'))
        except Exception:
            _num_trials_dic2 = {}
    v = _num_trials_dic2.get(rd) if hasattr(_num_trials_dic2, 'get') else None
    if not isinstance(v, dict) or not v:
        return None
    if not all(i in v for i in range(rows)):
        return None                      # partial coverage: do not half-adopt
    try:
        return np.array([float(v[i]) for i in range(rows)])
    except (TypeError, ValueError):
        return None


def _synthetic_awake(rd, mouse, dates, rows):
    """Length-only session tokens, used when real timestamps cannot be aligned.

    Date-prefixed so the two days of a combined recday stay disjoint -- that is what
    Figure2 cell 20's `awake_sessions[ii] in awake_sessions_day` membership test needs.
    Deliberately NOT of the form HH-MM-SS, so Figure7's np.where misses rather than
    matching the wrong session.
    """
    out = []
    for d in dates:
        if len(dates) > 1:
            half = task_data(f'{mouse}_{d}')       # `or []` is unsafe on an ndarray:
            n = 0 if half is None else len(half)   # truthiness is ambiguous
        else:
            n = rows
        out += [f'{d}-{i:02d}' for i in range(n)]
    if len(out) != rows:
        out = [f'{dates[0]}-{i:02d}' for i in range(rows)]
    return out


def _task_num(T):
    seen, labels = {}, []
    for row in T:
        k = tuple(int(v) for v in row)
        labels.append(seen.setdefault(k, len(seen)))
    return np.array(labels, dtype=int)


if __name__ == '__main__':
    sys.exit(main())
