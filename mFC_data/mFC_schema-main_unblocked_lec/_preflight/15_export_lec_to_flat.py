#!/usr/bin/env python3
"""Export the LEC `data_dic` into the flat per-session `.npy` layout El-Gaby's notebooks read.

This is the LEC counterpart of `10_make_bookkeeping.py`: it builds a SEPARATE
`Input_folder` (`lec_replication_run/data/Intermediate_objects/`) from
`data/processed_data/data_dic_lec.pkl`, so the unblocked notebooks run on LEC data
unchanged. Nothing here touches the PFC mirror or the deposit.

Run with the repo python (conda env `maze_ephys`, numpy 2.x), NOT `mfc_replication`:
the pickle was written by numpy 2 and numpy 1.22 cannot unpickle it. The `.npy` files
written here are format 1.0 and are re-read with the 3.9 kernel at the end as a check.

Conventions, each one measured on the pickle before being adopted (see LEC_REPRODUCTION.md):

  * `Trial_times` are 25 ms BIN indices (max == Neuron_raw.shape[1] in every usable
    session); PFC's deposited `trialtimes_` are ms and every consumer does `//25`.
    -> export `int32(Trial_times * 25)`. Integer dtype is load-bearing: cell 18 slices
       `distances[start:end]` with these values and a float slice raises.
  * `Locs_raw == 0` means untracked (3.8 % of bins); PFC uses NaN. -> `0 -> NaN`, float64.
    A 0 left in place becomes node index -1 after `(Location_raw-1).astype(int)` and
    silently indexes the last spatial module.
  * `XY_raw` is already in cm (-4.2..51.5 measured), PFC is pixels. The notebooks' speed
    cell treats `x < 0` as untracked; on LEC that sentinel would blank 0.84 % of speed bins
    (122 sessions, up to 10.4 % in one). -> add XY_OFFSET_CM to both coordinates. Speed and
    distance are translation-invariant, so nothing else changes.
  * The maze calibration files give pixels_per_cm = mean(C1..R3)/L. LEC is already in cm,
    so both Maze1 and Maze2 files are written with C=R=L (ratio exactly 1). Figure2 cell
    14 loads mazes '1' and '2' only, so LEC mice are mapped to maze 1 (edit UNBLOCK-LEC-01).
  * `Neuron_raw` is uint16 in the pickle and float64 in every PFC file. -> float64, so
    scipy filters (which preserve input dtype) behave identically on both datasets.
  * Neural arrays are written for EVERY recorded session (212), tracking arrays wherever
    tracking exists (188), `trialtimes_` only where completed trials exist (182). This is
    what his deposit looks like (Neuron_raw for recorded sessions regardless of behaviour),
    and `Num_trials_ = 0` is the gate that keeps the 30 trial-less sessions out of every
    analysis: every per-session consumer either iterates `non_repeat_ses` or loads
    `trialtimes_` in the same `try`. It also guarantees `Neuron_raw_<day>_0.npy` exists for
    every single day (Figure2 cells 51/54 load it outside any try).
  * Session index == data_dic key, never renumbered within a combined recday; `Task_data_`
    gets one row per index (real Task where present, zeros otherwise).
  * SINGLE DAYS. Figure2 cell 19 and Figure3 cells 23/29/32/42/51-89 need each combined
    recday split into its two calendar days, as El-Gaby's own two-day PFC recordings were.
    Each session is dated from the pyControl file the registry resolves for it (the name
    carries the date: `ah08-2025-06-13-114642.txt`; every LEC session falls on exactly one
    of the recday's two named dates). Each day becomes a recday `<mouse>_<YYYYMMDD>` with
    its own `Task_data_`/`Num_trials_`/`Task_num_`/session arrays and per-session files
    renumbered 0..n-1 within the day (copies of the combined files). The 50 day names form
    `3_task_all_days.npy`; `3_task_days.npy` stays EMPTY, so every day inherits the combined
    day's tuning through Figure2 cells 44/61/73 -- exactly the treatment his split PFC days
    got. `combined_ABCDE_days.npy` is empty.
    A single day holds its TRIAL-BEARING sessions only (user-approved 2026-09-14). Figure3
    cells 23/29/36 read `Neuron_<day>_0.npy` -- a 360-bin array that cannot exist for a
    trial-less session -- purely for `num_neurons`, so a day whose first session had no
    trials was silently dropped from the coherence analysis (5 of 50 days, 14 of 146
    sessions, in the first build). Measured before adopting: dropping trial-less rows leaves
    `non_repeat_ses` identical on all 50 days, no trial-less row shadows a later
    trial-bearing one on any day, and the only change is `X_all` on those five days (from an
    empty set, because their session 0 was an Object session carrying the all-zeros Task
    placeholder, to a one-member set). Combined recdays are unchanged and keep every row.
    `--single-days-only` rebuilds just the single-day arrays (and `3_task_all_days.npy`),
    touching no combined-name file, so it can run while a combined-only notebook is executing.
  * Session-timestamp arrays are synthetic tokens (length-only consumers in
    Figure2/3/5; deliberately not HH-MM-SS so Figure7's np.where misses, as in the PFC run).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap

import numpy as np

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
MFC = os.path.join(ROOT, 'mFC_data')
MIRROR = os.path.join(MFC, 'lec_replication_run', 'data', 'Intermediate_objects')
LOGDIR = os.path.join(MFC, 'lec_replication_run', 'logs')
PFC_MIRROR = os.path.join(MFC, 'replication_run', 'data', 'Intermediate_objects')
REPO_CODE = os.path.join(ROOT, 'code')
ENVPY = '/nfs/nhome/live/aharris/.conda/envs/mfc_replication/bin/python'

BIN_MS = 25
XY_OFFSET_CM = 10.0
MAZE_UNITS = 37          # measured outer-node extent in cm; only the C/L ratio is consumed
NUM_NODES = 9
NUM_LOCS = 21
RECDAY_DTYPE = '<U22'    # what the PFC cohort lists use; LEC names are exactly 22 chars
SESSION_FAMILIES = ('Neuron_raw_', 'Location_raw_', 'XY_raw_', 'trialtimes_')

# Read-only reference from the handoff / this session's measurement; reported, not asserted.
EXPECTED = dict(recdays=25, sessions=212, usable=182, object_sessions=21,
                zero_trial_sessions=9, neurons=2851, single_days=50)


# ----------------------------------------------------------------------------- helpers
def present(sess, key):
    """A key counts as present only if it is there AND not None (21 Object sessions carry
    `Trial_times = None`)."""
    return key in sess and sess[key] is not None


def task_row(sess):
    """(4,) int array in 1..9, or None. Object sessions have no `Task`."""
    if not present(sess, 'Task'):
        return None
    t = sess['Task']
    if 'defaultdict' in str(type(t)):
        return None
    a = np.asarray(t).ravel()
    if a.size != 4:
        raise AssertionError(f'Task has {a.size} entries, expected 4: {t!r}')
    a = a.astype(int)
    if not np.all((a >= 1) & (a <= NUM_NODES)):
        raise AssertionError(f'Task values outside 1..{NUM_NODES}: {a}')
    return a


def non_repeat_ses_maker(Tasks, num_trials_day):
    """Verbatim transcription of Figure2 cell 6 / Figure5_Regression cell 8."""
    non_repeat_bool_all = []
    for ses_ind in np.arange(len(Tasks)):
        if ses_ind == 0:
            non_repeat_bool = True
        else:
            num_prev_repeats = np.sum([np.array_equal(Tasks[ses_ind], Tasks[:ses_ind][jj])
                                       for jj in range(len(Tasks[:ses_ind]))])
            non_repeat_bool = num_prev_repeats == 0
        non_repeat_bool_all.append(non_repeat_bool)
    non_repeat_bool_all = np.hstack((non_repeat_bool_all))
    num_trials_bool = num_trials_day > 0
    return np.where(np.logical_and(non_repeat_bool_all, num_trials_bool))[0]


def task_num(Task_data):
    """First-occurrence label of each Task row (what 10_make_bookkeeping.py writes)."""
    seen, labels = {}, []
    for row in Task_data:
        k = tuple(int(v) for v in row)
        labels.append(seen.setdefault(k, len(seen)))
    return np.array(labels, dtype=int)


_registry = None


def session_day(rd, s, d1, d2):
    """Which of the recday's two dates a session belongs to, from the pyControl file the
    registry resolves for it. Returns d1 or d2, or None if undatable (Object sessions)."""
    global _registry
    if _registry is None:
        try:
            import recday_registry as _r   # REPO_CODE is on sys.path by then
            _registry = _r
        except Exception as e:  # noqa: BLE001
            _registry = False
            print(f'  WARNING: recday_registry unavailable ({e!r}); day split falls back to order')
    if not _registry:
        return None
    try:
        name = os.path.basename(str(_registry.pycontrol_file(rd, int(s))))
        parts = name.split('-')
        date = int(parts[1] + parts[2] + parts[3])        # YYYY MM DD -> YYYYMMDD
    except Exception:  # noqa: BLE001
        return None
    return d1 if date < int(d2) else d2


def maze_file_text(units):
    """Same layout as the deposited Maze{1,2}_measurements.txt: tab-separated, CRLF,
    no terminator after the last line (cell 14 reads it with lineterminator='\\r')."""
    rows = [('C1', units), ('C2', units), ('C3', units),
            ('R1', units), ('R2', units), ('R3', units),
            ('L', units), ('W', units)]
    return 'Label\tValue\r\n' + '\r\n'.join(f'{k}\t{v}' for k, v in rows)


def verify_maze_calibration(path):
    """Parse exactly as Figure2 cell 14 does and return pixels_per_cm."""
    import pandas as pd
    df = pd.read_csv(path, sep='\t', lineterminator='\r')
    df = df.replace(r'\n', '', regex=True)
    m = df.values
    return float(np.mean(m[:6, 1])) / float(m[6, 1])


def verify_with_kernel_env(files):
    """Re-read a handful of the written files with the 3.9 / numpy 1.22 kernel."""
    if not os.path.exists(ENVPY):
        print(f'  WARNING: {ENVPY} missing; skipped the numpy-1.22 re-read check')
        return None
    code = textwrap.dedent('''
        import sys, json, numpy as np
        out = {}
        for f in sys.argv[1:]:
            a = np.load(f, allow_pickle=False)
            out[f.split('/')[-1]] = [str(a.dtype), list(a.shape),
                                    int(np.isnan(a).sum()) if a.dtype.kind == 'f' else None]
        print(json.dumps({'numpy': np.__version__, 'files': out}))
    ''')
    r = subprocess.run([ENVPY, '-c', code, *files], capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f'numpy-1.22 re-read FAILED:\n{r.stderr}')
    return json.loads(r.stdout.strip().splitlines()[-1])


def save(name, arr):
    np.save(os.path.join(MIRROR, name), arr)


# -------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true',
                    help='allow writing into a folder that already holds Neuron_raw_ files')
    ap.add_argument('--no-kernel-check', action='store_true')
    ap.add_argument('--pickle', default=None, help='override the data_dic path')
    ap.add_argument('--single-days-only', action='store_true',
                    help='rebuild only the single-day recdays and 3_task_all_days.npy; write no '
                         'combined-name file (safe while a combined-only notebook is running)')
    args = ap.parse_args()
    sdo = args.single_days_only

    assert os.path.realpath(MIRROR) != os.path.realpath(PFC_MIRROR), 'LEC MIRROR == PFC MIRROR'
    os.makedirs(MIRROR, exist_ok=True)
    os.makedirs(os.path.join(MIRROR, 'Maze_measurements'), exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    existing = [f for f in os.listdir(MIRROR) if f.startswith('Neuron_raw_')]
    if existing and not args.overwrite:
        print(f'REFUSING: {MIRROR} already holds {len(existing)} Neuron_raw_ files. '
              f'Pass --overwrite to rebuild (a notebook run may be using them).')
        return 2

    # ---- load through the repo's validated loader (two registry validators run)
    sys.path.insert(0, REPO_CODE)
    try:
        from glm_analysis_v3 import load_data_dic
    except Exception as e:  # noqa: BLE001
        print(f'FATAL: cannot import glm_analysis_v3.load_data_dic from {REPO_CODE}: {e!r}')
        return 1
    print(f'python {sys.version.split()[0]}  numpy {np.__version__}')
    data_dic = load_data_dic(args.pickle) if args.pickle else load_data_dic()
    recdays = sorted(data_dic.keys())
    assert all(len(rd) == 22 for rd in recdays), 'recday names are not all 22 chars'

    # ---- remove stale single-day files (any family) so a renumbered rebuild leaves no
    # higher-index leftovers behind. Single-day names are <mouse>_<8 digits> followed by
    # either .npy or _<1-2 digit session>.npy; combined names carry a second 8-digit date,
    # so the anchored pattern cannot match them. Maze_measurements/ and Edge_grid.npy are
    # not files of either shape.
    single_re = re.compile(r'(ah|ly)\d{2}_\d{8}(_\d{1,2})?\.npy$')
    stale = [f for f in os.listdir(MIRROR) if single_re.search(f)
             and not re.search(r'(ah|ly)\d{2}_\d{8}_\d{8}', f)]
    for f in stale:
        os.remove(os.path.join(MIRROR, f))
    print(f'removed {len(stale)} stale single-day files' + ('' if stale else ' (none present)'))
    if sdo:
        print('--single-days-only: combined-name files are NOT rewritten')

    manifest, neural_only, problems, notes = [], [], [], []
    n_written = 0
    neurons_per_recday = {}
    sample_files = []
    nonrepeat_report, day_split_report, single_days = {}, {}, {}
    n_status = {'exported': 0, 'neural_only': 0}

    print(f'\n{"recday":24s} {"rows":>4s} {"full":>4s} {"neur":>4s} {"cells":>5s}  day split   non_repeat_ses')
    for rd in recdays:
        rd_data = data_dic[rd]
        keys = sorted(k for k in rd_data.keys() if isinstance(k, (int, np.integer)))
        assert keys, f'{rd}: no integer session keys'
        rows = int(max(keys)) + 1                      # NOT len(keys): keys may be sparse
        if rows != len(keys):
            notes.append(f'{rd}: sparse session keys {keys} -> {rows} Task_data rows')
        mouse, d1, d2 = rd.split('_')

        # ---- date every session (single-day split); undatable ones inherit the previous day
        day_of, undated = [], []
        for s in range(rows):
            d = session_day(rd, s, d1, d2) if s in keys else None
            if d is None:
                undated.append(s)
                d = day_of[-1] if day_of else d1
            day_of.append(d)
        assert day_of.count(d1) + day_of.count(d2) == rows
        if undated:
            notes.append(f'{rd}: session(s) {undated} not datable from pyControl; assigned to '
                         f'the previous session\'s day')

        Task_data = np.zeros((rows, 4), dtype=np.int32)
        Num_trials = np.zeros(rows, dtype=np.int64)
        written = {}                     # session -> list of per-session family prefixes written
        n_neurons = None
        n_full = 0

        for s in keys:
            sess = rd_data[s]
            nt = int(sess.get('num_trials') or 0)
            t = task_row(sess)
            if t is not None:
                Task_data[s] = t
            neuron = np.asarray(sess['Neuron_raw'], dtype=np.float64)
            assert neuron.ndim == 2, f'{rd} s{s}: Neuron_raw ndim {neuron.ndim}'
            nn, nb = neuron.shape
            if n_neurons is None:
                n_neurons = nn
            assert nn == n_neurons, f'{rd} s{s}: {nn} neurons vs {n_neurons} in the recday'

            has = {k: present(sess, k) for k in ('Locs_raw', 'XY_raw', 'Trial_times', 'Task')}
            tt_empty = has['Trial_times'] and np.asarray(sess['Trial_times']).size == 0
            if t is None:
                reason = 'object_session_no_task'
            elif nt <= 0 or tt_empty:
                reason = 'zero_trials'
            elif not (has['Locs_raw'] and has['XY_raw']):
                reason = 'missing_tracking'
            else:
                reason = None
            if reason is not None and nt > 0 and reason != 'zero_trials':
                problems.append(f'{rd} s{s}: num_trials={nt} but {reason}')   # would lose a fold

            files = {f'Neuron_raw_{rd}_{s}.npy': neuron}
            fams = ['Neuron_raw_']
            rec = dict(recday=rd, session=s, day=day_of[s], n_neurons=nn, n_bins=nb, n_trials=nt,
                       task=Task_data[s].tolist(), status='exported' if reason is None else 'neural_only',
                       reason=reason)

            if has['Locs_raw'] and has['XY_raw']:
                locs_i = np.asarray(sess['Locs_raw'])
                xy = np.asarray(sess['XY_raw'], dtype=np.float64)
                assert len(locs_i) == len(xy), f'{rd} s{s}: len(Locs)={len(locs_i)} != len(XY)={len(xy)}'
                assert xy.ndim == 2 and xy.shape[1] == 2, f'{rd} s{s}: XY shape {xy.shape}'
                assert not np.isnan(xy).any(), f'{rd} s{s}: NaN in XY_raw (offset logic assumes none)'
                uniq = np.unique(locs_i)
                assert np.all((uniq >= 0) & (uniq <= NUM_LOCS)), f'{rd} s{s}: Locs values {uniq}'
                locs = locs_i.astype(np.float64)
                locs[locs_i == 0] = np.nan
                xy_out = xy + XY_OFFSET_CM
                assert xy_out.min() >= 0, f'{rd} s{s}: XY still negative after +{XY_OFFSET_CM} cm'
                files[f'Location_raw_{rd}_{s}.npy'] = locs
                files[f'XY_raw_{rd}_{s}.npy'] = xy_out
                fams += ['Location_raw_', 'XY_raw_']
                rec.update(len_tracking=int(len(locs_i)), tracking_minus_bins=int(len(locs_i) - nb),
                           frac_untracked=round(float((locs_i == 0).mean()), 4),
                           xy_min_raw=float(xy.min()), xy_min_exported=float(xy_out.min()))

            if reason is None:
                tt = np.asarray(sess['Trial_times'], dtype=np.float64)
                assert tt.ndim == 2 and tt.shape[1] == 5, f'{rd} s{s}: Trial_times shape {tt.shape}'
                assert not np.isnan(tt).any(), f'{rd} s{s}: NaN in Trial_times'
                assert np.all(np.mod(tt, 1) == 0), f'{rd} s{s}: Trial_times not integer-valued'
                assert tt.shape[0] == nt, f'{rd} s{s}: len(Trial_times)={tt.shape[0]} != num_trials={nt}'
                assert tt.max() == nb, f'{rd} s{s}: Trial_times.max()={tt.max()} != n_bins={nb}'
                assert len(locs_i) >= tt.max(), f'{rd} s{s}: len(Locs_raw)={len(locs_i)} < Trial_times.max()'
                assert tt.max() * BIN_MS < 2**31, f'{rd} s{s}: trialtimes overflow int32'
                tt_ms = (tt * BIN_MS).astype(np.int32)
                assert np.array_equal(tt_ms // BIN_MS, tt.astype(np.int64)), f'{rd} s{s}: ms round-trip failed'
                files[f'trialtimes_{rd}_{s}.npy'] = tt_ms
                fams.append('trialtimes_')
                rec['trialtimes_ms_max'] = int(tt_ms.max())
                Num_trials[s] = nt
                n_full += 1

            if not sdo:
                for fn, arr in files.items():
                    save(fn, arr)
                    n_written += 1
            if len(sample_files) < 8 and s == keys[0]:
                sample_files += [os.path.join(MIRROR, fn) for fn in files]
            written[s] = fams
            n_status[rec['status']] += 1
            (manifest if reason is None else neural_only).append(rec)

        # ---- combined-recday bookkeeping (all indexed by Task_data_ row)
        tokens = np.array([f'{rd}-{i:02d}' for i in range(rows)], dtype='<U32')
        if not sdo:
            for fn, arr in {
                f'Task_data_{rd}.npy': Task_data,
                f'Num_trials_{rd}.npy': Num_trials,
                f'awake_session_{rd}.npy': tokens,
                f'awake_session_behaviour_{rd}.npy': tokens,
                f'All_session_{rd}.npy': tokens,
                f'All_session_behaviour_{rd}.npy': tokens,
                f'sleep_session_{rd}.npy': np.array([], dtype='<U32'),
                f'Task_num_{rd}.npy': task_num(Task_data),
            }.items():
                save(fn, arr)
                n_written += 1
        assert int((Num_trials > 0).sum()) == n_full, f'{rd}: Num_trials_>0 != fully exported sessions'
        neurons_per_recday[rd] = n_neurons

        # ---- single days: each calendar day becomes its own recday holding its
        # TRIAL-BEARING sessions, renumbered 0..n-1 in chronological order (see docstring)
        split = {}
        for d in (d1, d2):
            all_idx = [s for s in range(rows) if day_of[s] == d]
            idx = [s for s in all_idx if Num_trials[s] > 0]
            dropped = [s for s in all_idx if Num_trials[s] <= 0]
            single = f'{mouse}_{d}'
            split[d] = idx
            if not idx:
                notes.append(f'{single}: no trial-bearing session on this day -- day not written')
                single_days[single] = dict(combined=rd, sessions=[], dropped_trialless=dropped,
                                           n_rows=0, written=False)
                continue
            tok = np.array([f'{single}-{s:02d}' for s in idx], dtype='<U32')
            for fn, arr in {
                f'Task_data_{single}.npy': Task_data[idx],
                f'Num_trials_{single}.npy': Num_trials[idx],
                f'Task_num_{single}.npy': task_num(Task_data[idx]),
                f'awake_session_{single}.npy': tok,
                f'awake_session_behaviour_{single}.npy': tok,
                f'All_session_{single}.npy': tok,
                f'All_session_behaviour_{single}.npy': tok,
                f'sleep_session_{single}.npy': np.array([], dtype='<U32'),
            }.items():
                save(fn, arr)
                n_written += 1
            n_copied = 0
            for j, s in enumerate(idx):
                for fam in written.get(s, []):
                    shutil.copy2(os.path.join(MIRROR, f'{fam}{rd}_{s}.npy'),
                                 os.path.join(MIRROR, f'{fam}{single}_{j}.npy'))
                    n_written += 1
                    n_copied += 1
            for fam in ('Neuron_raw_', 'trialtimes_'):
                assert os.path.exists(os.path.join(MIRROR, f'{fam}{single}_0.npy')), \
                    f'{single}: no {fam}0 -- session 0 must be trial-bearing (cells 23/29/36/51/54 read it)'
            single_days[single] = dict(combined=rd, sessions=idx, dropped_trialless=dropped,
                                       n_rows=len(idx), written=True,
                                       non_repeat_ses=[int(x) for x in non_repeat_ses_maker(Task_data[idx], Num_trials[idx])],
                                       files_copied=n_copied)
        day_split_report[rd] = {d1: split[d1], d2: split[d2], 'undated_inherited': undated,
                                'trialless_dropped_from_days': [s for s in range(rows) if Num_trials[s] <= 0]}

        # ---- what El-Gaby's own selector will do with the combined files (report only)
        nrs = non_repeat_ses_maker(Task_data, Num_trials)
        dropped = []
        for s in keys:
            if Num_trials[s] > 0 and s not in nrs:
                earlier = [j for j in range(s) if np.array_equal(Task_data[j], Task_data[s])]
                live = [j for j in earlier if Num_trials[j] > 0]
                dropped.append(dict(session=s, repeats=earlier, only_zero_trial_predecessors=(len(live) == 0)))
                if not live:
                    notes.append(f'{rd} s{s}: dropped as a repeat of ZERO-TRIAL session(s) {earlier}')
        one_trial = [int(s) for s in nrs if Num_trials[s] == 1]
        nonrepeat_report[rd] = dict(non_repeat_ses=[int(x) for x in nrs], dropped_repeats=dropped,
                                    one_trial_folds=one_trial)
        if one_trial:
            notes.append(f'{rd}: session(s) {one_trial} enter as folds with ONE trial (his gate is >0)')

        print(f'{rd:24s} {rows:>4d} {n_full:>4d} {rows - n_full:>4d} {n_neurons:>5d}  '
              f'{len(split[d1])}+{len(split[d2]):<4d}  {list(map(int, nrs))}')

    # ---- cohort lists
    names = np.array(recdays, dtype=RECDAY_DTYPE)
    singles = np.array(sorted(k for k, v in single_days.items() if v['written']), dtype=RECDAY_DTYPE)
    empty = np.array([], dtype=RECDAY_DTYPE)
    lists = {'3_task_all_days.npy': singles}
    if not sdo:
        lists.update({'combined_ABCDonly_days.npy': names, 'combined_days.npy': names,
                      '3_task_days.npy': empty, 'combined_ABCDE_days.npy': empty})
    for fn, arr in lists.items():
        save(fn, arr)
        n_written += 1

    # ---- maze calibration (ratio 1) + Edge_grid
    if not sdo:
        for maze in ('1', '2'):
            p = os.path.join(MIRROR, 'Maze_measurements', f'Maze{maze}_measurements.txt')
            with open(p, 'w', newline='') as f:
                f.write(maze_file_text(MAZE_UNITS))
            ppc = verify_maze_calibration(p)
            assert ppc == 1.0, f'{p}: pixels_per_cm parsed as {ppc}, expected exactly 1.0'
        eg = os.path.join(PFC_MIRROR, 'Edge_grid.npy')
        if os.path.exists(eg):
            shutil.copy2(eg, os.path.join(MIRROR, 'Edge_grid.npy'))
        else:
            notes.append('Edge_grid.npy not found in the PFC mirror (no notebook loads it)')

    # ---- summary
    by_reason = {}
    for r in neural_only:
        by_reason[r['reason']] = by_reason.get(r['reason'], 0) + 1
    total_neurons = int(sum(neurons_per_recday.values()))
    n_sessions = len(manifest) + len(neural_only)
    n_days_written = sum(1 for v in single_days.values() if v['written'])
    n_day_sessions = sum(v['n_rows'] for v in single_days.values())
    n_day_dropped = sum(len(v['dropped_trialless']) for v in single_days.values())
    print('\n=== export summary ===' + ('  [--single-days-only]' if sdo else ''))
    print(f'  recdays (combined)      : {len(recdays)}   (expected {EXPECTED["recdays"]})')
    print(f'  single days (3_task_all): {n_days_written}   (expected {EXPECTED["single_days"]});  3_task = 0')
    print(f'  single-day sessions     : {n_day_sessions} trial-bearing kept, {n_day_dropped} trial-less dropped '
          f'(combined recdays keep all {n_day_sessions + n_day_dropped} rows)')
    print(f'  sessions in data_dic    : {n_sessions}   (expected {EXPECTED["sessions"]})')
    print(f'  fully exported (trials) : {len(manifest)}   (expected {EXPECTED["usable"]})')
    print(f'  neural-only (Num_trials_=0): {len(neural_only)}   {by_reason}')
    print(f'  neurons (sum recdays)   : {total_neurons}   (expected {EXPECTED["neurons"]})')
    print(f'  arrays written          : {n_written}  -> {MIRROR}')
    print(f'  XY offset applied       : +{XY_OFFSET_CM} cm on both coordinates')
    print(f'  maze calibration        : pixels_per_cm = 1.0 (Maze1 and Maze2 files, {MAZE_UNITS} units)')
    if problems:
        print(f'\n  PROBLEMS ({len(problems)}):')
        for p in problems:
            print(f'    {p}')
    if notes:
        print(f'\n  NOTES ({len(notes)}):')
        for p in notes:
            print(f'    {p}')

    kernel_check = None
    if not args.no_kernel_check:
        print('\n=== re-reading a sample with the mfc_replication kernel (numpy 1.22) ===')
        first_single = sorted(k for k, v in single_days.items() if v['written'])[0]
        kernel_check = verify_with_kernel_env(
            ([] if sdo else sample_files[:4]) +
            [os.path.join(MIRROR, 'combined_ABCDonly_days.npy'),
             os.path.join(MIRROR, '3_task_all_days.npy'),
             os.path.join(MIRROR, f'Task_data_{recdays[0]}.npy'),
             os.path.join(MIRROR, f'Task_data_{first_single}.npy'),
             os.path.join(MIRROR, f'Num_trials_{first_single}.npy'),
             os.path.join(MIRROR, f'Neuron_raw_{first_single}_0.npy'),
             os.path.join(MIRROR, f'trialtimes_{first_single}_0.npy')])
        if kernel_check:
            print(f'  numpy {kernel_check["numpy"]}')
            for fn, (dt, shp, nnan) in kernel_check['files'].items():
                print(f'  {fn:45s} {dt:>8s} {str(shp):>16s}  nan={nnan}')
            want = {'Neuron_raw_': 'float64', 'Location_raw_': 'float64',
                    'XY_raw_': 'float64', 'trialtimes_': 'int32'}
            for fn, (dt, _, _) in kernel_check['files'].items():
                for pre, exp in want.items():
                    if fn.startswith(pre):
                        assert dt == exp, f'{fn}: dtype {dt} != {exp}'

    with open(os.path.join(LOGDIR, 'lec_export_manifest.json'), 'w') as f:
        json.dump(dict(mirror=MIRROR, xy_offset_cm=XY_OFFSET_CM, bin_ms=BIN_MS,
                       maze_units=MAZE_UNITS, python=sys.version.split()[0],
                       numpy=np.__version__, expected=EXPECTED,
                       summary=dict(recdays=len(recdays), single_days=len(single_days),
                                    sessions=n_sessions, exported=len(manifest),
                                    neural_only=by_reason, neurons=total_neurons,
                                    arrays_written=n_written),
                       neurons_per_recday=neurons_per_recday,
                       exported=manifest, neural_only=neural_only,
                       non_repeat_ses=nonrepeat_report, day_split=day_split_report,
                       single_days=single_days,
                       problems=problems, notes=notes, kernel_check=kernel_check),
                  f, indent=2)
    print(f'\nwrote {LOGDIR}/lec_export_manifest.json')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
