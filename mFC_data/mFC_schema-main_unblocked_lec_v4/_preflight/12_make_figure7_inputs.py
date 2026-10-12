#!/usr/bin/env python3
"""Build Figure7's own Input_folder: a symlink overlay of the mirror plus the inputs the deposit lacks.

    python 12_make_figure7_inputs.py [--refresh] [--sleep]      # --sleep: LEC only

Figure7 (sleep / offline reactivation) never ran on either dataset. Cell 11 reads
`MetaData.xlsx - <mouse>.csv` with `usecols=np.arange(0,18)` and then needs a column named
`Include`; cells 14/20 compare the sheet's `Date` with ISO `YYYY-MM-DD` and its `Ephys` with
the `HH-MM-SS` tokens in `All_session_<rd>.npy`. The deposited sheets are 12 / 6 / 8 / 11 / 25
columns, have no `Include`, write `DD/MM/YYYY`, and carry Excel-mangled `Ephys` on ~15 % of rows.
The executed 18-column schema is this lab's own metadata format
(`data/metadata/MetaData-*.csv`), so it is rebuilt here from the deposited columns by name.

Only Figure7 reads the sheets, and only Figure7 reads the *values* of the session arrays
(Figure2/3/5 take len()), so everything goes into a separate overlay directory
`<run>/data/Intermediate_objects_f7/` -- relative symlinks to every mirror file, with the
rebuilt sheets (and, for LEC, the sleep shards and real session arrays) as real files on top.
The mirror stays byte-identical to what the other notebooks ran on.

LEC (`--sleep`): the sleep-box arrays already exist (`Neuron_raw_<rd>_sb_<j>.npy`, from
code/preprocessing/extract_neuron_raw.py, byte-exact boundaries from
recording_sessions_in_concat.csv). This writes his format: `binned_FR_dic_<rd>_<i>` joblib
shards for EVERY recording in chronological concat order (origin 0, whole recording, as his
awake shards), `All_session_` / `awake_session_` / `sleep_session_` with the real `HH-MM-SS`
start times, and the two sheets from `data/metadata/MetaData-<mouse>.csv` (Sleepbox rows inline).
Sleep shards are asserted equal to the existing `_sb_` arrays.

Gates fail loudly; `logs/figure7_inputs_manifest.json` records everything.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys

import numpy as np

REGION = 'lec'                                   # the PFC copy says 'pfc'
ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
RUN = 'replication_run' if REGION == 'pfc' else 'lec_replication_run_v4'
MIRROR = os.path.join(ROOT, RUN, 'data', 'Intermediate_objects')
OVERLAY = os.path.join(ROOT, RUN, 'data', 'Intermediate_objects_f7')
LOGDIR = os.path.join(ROOT, RUN, 'logs')
HERE = os.path.dirname(os.path.abspath(__file__))
LEC_REPO = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
LEC_META = os.path.join(LEC_REPO, 'data', 'metadata')
LEC_MICE = ('ah08', 'ah10', 'ly05', 'ly06', 'ly07')

SCHEMA18 = ['Date', 'Maze', 'Structure_abstract', 'Structure_no', 'Structure', 'Structure_session',
            'Session_time', 'Reward_time', 'Entry', 'Behaviour', 'Tracking', 'Ephys', 'Probe',
            'Include', 'Performance', 'Notes1', 'Notes2', 'Notes3']
SLEEP18 = ['Date', 'Sleep_box', 'Stage', 'Session_time', 'Tracking', 'Ephys', 'Include'] + \
          [f'x{i}' for i in range(7, 18)]
TS_RE = re.compile(r'^\d{2}-\d{2}-\d{2}$')


def _bookkeeping():
    spec = importlib.util.spec_from_file_location('mk', os.path.join(HERE, '10_make_bookkeeping.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


repair_ephys = _bookkeeping().repair_ephys


# ------------------------------------------------------------------------------ sheets
def iso_date(s):
    """'01/09/2023' | '1/9/2023' | '2025-6-13' | '2025-06-13' -> 'YYYY-MM-DD'; blank -> '-'."""
    s = (s or '').strip()
    if not s:
        return '-'
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        return f'{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}'
    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', s)
    if m:
        return f'{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}'
    return s


def clean(v):
    """No commas (his reader is genfromtxt(delimiter=',')), no surrounding whitespace."""
    return (v or '').strip().replace(',', ';')


def read_rows(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        out.append({header[i]: (r[i] if i < len(r) else '') for i in range(len(header)) if header[i]})
    return out


def sheet_rows(rows, schema, include_all=True):
    """Map rows into `schema` by header name. Missing columns -> '-'. Date -> ISO, Ephys ->
    repaired HH-MM-SS, blanks in Date/Ephys -> '-' (remove_empty runs per column in cell 11,
    so a blank would shift one column against the other). Include -> '1'."""
    out, n_rep, n_blank = [], 0, 0
    for r in rows:
        vals = []
        for col in schema:
            if col == 'Include':
                vals.append('1' if include_all else clean(r.get(col, '')) or '1')
                continue
            v = clean(r.get(col, '-' if col not in r else r.get(col)))
            if col == 'Date':
                v = iso_date(v)
            elif col == 'Ephys':
                rep_v, was_rep, ok = repair_ephys(v)
                n_rep += int(was_rep)
                v = rep_v if ok else (v or '-')
            if col in ('Date', 'Ephys') and v == '':
                v = '-'
                n_blank += 1
            if v == '' and col not in schema[:12]:
                v = '-'
            vals.append(v if v != '' else '-')
        out.append(vals)
    return out, n_rep, n_blank


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(r) + '\n')


def parse_like_cell11(path):
    with open(path, 'r') as f:
        arr = np.genfromtxt(f, delimiter=',', dtype=str, usecols=np.arange(0, 18))
    if arr.ndim != 2 or arr.shape[1] != 18:
        raise SystemExit(f'GATE sheet {path}: genfromtxt gives shape {arr.shape}, not 18 columns')
    if 'Include' not in arr[0]:
        raise SystemExit(f'GATE sheet {path}: no Include header')
    return arr


def build_sheets_pfc(manifest):
    names = sorted(f for f in os.listdir(MIRROR) if f.startswith('MetaData.xlsx - ') and f.endswith('.csv'))
    for name in names:
        rows = read_rows(os.path.join(MIRROR, name))
        is_sleep = name.endswith('_sleep.csv')
        schema = SLEEP18 if is_sleep else SCHEMA18
        out, n_rep, n_blank = sheet_rows(rows, schema)
        dst = os.path.join(OVERLAY, name)
        if os.path.islink(dst):
            os.unlink(dst)
        write_csv(dst, schema, out)
        arr = parse_like_cell11(dst)
        manifest['sheets'][name] = dict(rows=len(out), source_columns=len(rows[0]) if rows else 0,
                                        ephys_repaired=n_rep, blanks_to_dash=n_blank,
                                        parsed_shape=list(arr.shape))
    print(f'  {len(names)} sheets rebuilt to 18 columns')


def build_sheets_lec(manifest):
    for mouse in LEC_MICE:
        rows = read_rows(os.path.join(LEC_META, f'MetaData-{mouse}.csv'))
        # his task sheets hold no sleep rows; cell 14 takes the FIRST day-2 row of the task
        # sheet as the first task session, so the inline Sleepbox rows go to the _sleep sheet only
        sb = [r for r in rows if (r.get('Maze') or '').strip().lower() == 'sleepbox']
        task_src = [r for r in rows if (r.get('Maze') or '').strip().lower() != 'sleepbox']
        task, n_rep, n_blank = sheet_rows(task_src, SCHEMA18)
        write_csv(os.path.join(OVERLAY, f'MetaData.xlsx - {mouse}.csv'), SCHEMA18, task)
        sleep_src = [{'Date': r.get('Date'), 'Sleep_box': r.get('Probe'), 'Stage': '-',
                      'Session_time': r.get('Session_time'), 'Tracking': r.get('Tracking'),
                      'Ephys': r.get('Ephys')} for r in sb]
        sleep, _, _ = sheet_rows(sleep_src, SLEEP18)
        sleep.sort(key=lambda v: (v[0], v[5]))
        write_csv(os.path.join(OVERLAY, f'MetaData.xlsx - {mouse}_sleep.csv'), SLEEP18, sleep)
        for nm in (f'MetaData.xlsx - {mouse}.csv', f'MetaData.xlsx - {mouse}_sleep.csv'):
            arr = parse_like_cell11(os.path.join(OVERLAY, nm))
            manifest['sheets'][nm] = dict(rows=int(arr.shape[0] - 1), parsed_shape=list(arr.shape))
        notes = [f"{r.get('Date')} {r.get('Ephys')}: {r.get('Notes1')}" for r in sb if (r.get('Notes1') or '').strip()]
        manifest['sleep_notes'][mouse] = notes
        print(f'  {mouse}: task sheet {len(task)} rows ({n_rep} Ephys repaired), sleep sheet {len(sleep)} rows, '
              f'{len(notes)} annotated sleep rows kept')


# ----------------------------------------------------------------------------- overlay
def build_overlay(refresh=False):
    os.makedirs(OVERLAY, exist_ok=True)
    n_new = 0
    for name in os.listdir(MIRROR):
        if name.startswith('MetaData.xlsx - '):
            continue
        dst = os.path.join(OVERLAY, name)
        if os.path.lexists(dst):
            if refresh and os.path.islink(dst):
                os.unlink(dst)
            else:
                continue
        os.symlink(os.path.relpath(os.path.join(MIRROR, name), OVERLAY), dst)
        n_new += 1
    print(f'  overlay {OVERLAY}: {n_new} new symlinks, {len(os.listdir(OVERLAY))} entries')


# --------------------------------------------------------------------------- LEC sleep
def export_lec_sleep(manifest):
    """Shards + real session arrays for the 25 LEC recdays, from spikes and the concat table."""
    import joblib
    sys.path.insert(0, os.path.join(LEC_REPO, 'code', 'preprocessing'))
    sys.path.insert(0, os.path.join(LEC_REPO, 'code'))
    import extract_neuron_raw as enr
    import recday_registry as rr
    flat = os.path.join(LEC_REPO, 'data', 'processed_data', 'neuron_raw_mingyutest')

    recdays = [str(x) for x in np.load(os.path.join(MIRROR, 'combined_ABCDonly_days.npy'))]
    for rd in recdays:
        table = enr.concat_table(rd)
        wake, sleep = enr.session_map(rd)
        ks = rr.kilosort_dir(rd)
        qc = np.load(ks / 'QC_single_units.npy')
        spike_times = np.load(ks / 'sorter_output/spike_times.npy').squeeze()
        spike_clusters = np.load(ks / 'sorter_output/spike_clusters.npy').squeeze()
        lut = np.full(int(spike_clusters.max()) + 2, -1, dtype=np.int64)
        lut[qc] = np.arange(len(qc))
        order = np.argsort(spike_times, kind='stable')
        spike_times, spike_clusters = spike_times[order], spike_clusters[order]

        times = np.array([str(t) for t in table['time']], dtype='<U32')
        wake_t, sleep_t = times[wake], times[sleep]
        n_task = len(np.load(os.path.join(MIRROR, f'Task_data_{rd}.npy'), allow_pickle=True))
        bins = (table['n_samples'] // enr.SAMPLES_PER_BIN).to_numpy()
        dropped = []
        if len(wake) < n_task:
            # data_dic (and so the flat export) can carry a session the metadata does not name:
            # ah08_20250624_20250625 s10 is the recday's last sleep-box recording, paired as a task
            # session by the original positional pipeline (Num_trials 0, whole-recording length).
            # Accept only if every extra array is exactly the length of a sleep recording; then
            # Figure7's copies of the bookkeeping arrays drop those rows (session indices < len(wake)
            # are unchanged, so every Neuron_<rd>_<s> file they point to still matches).
            for s in range(len(wake), n_task):
                p = os.path.join(MIRROR, f'Neuron_raw_{rd}_{s}.npy')
                n = np.load(p, mmap_mode='r').shape[1] if os.path.exists(p) else None
                hit = [int(i) for i in sleep if bins[i] == n]
                if not hit:
                    raise SystemExit(f'GATE {rd}: flat session {s} ({n} bins) is not a sleep recording; '
                                     f'{len(wake)} task recordings vs {n_task} Task_data rows')
                dropped.append(dict(session=s, bins=int(n), sleep_row=hit[0], time=str(times[hit[0]])))
            for fam in ('Task_data', 'Num_trials', 'Task_num'):
                arr = np.load(os.path.join(MIRROR, f'{fam}_{rd}.npy'), allow_pickle=True)[:len(wake)]
                dst = os.path.join(OVERLAY, f'{fam}_{rd}.npy')
                if os.path.lexists(dst):
                    os.unlink(dst)
                np.save(dst, arr)
        elif len(wake) > n_task:
            raise SystemExit(f'GATE {rd}: {len(wake)} task recordings vs {n_task} Task_data rows')
        if wake != sorted(wake) or not all(TS_RE.match(t) for t in times):
            raise SystemExit(f'GATE {rd}: wake rows not increasing or bad time token')

        info = dict(n_all=len(table), n_awake=len(wake), n_sleep=len(sleep), shards=0,
                    sleep_exact=0, wake_mean_ratio=[], dropped_flat_sessions=dropped)
        for i in range(len(table)):
            start = int(table['start'].iloc[i])
            n_samples = int(table['n_samples'].iloc[i])
            lo, hi = np.searchsorted(spike_times, [start, start + n_samples])
            arr = enr.bin_session(spike_times[lo:hi] - start, lut[spike_clusters[lo:hi]],
                                  len(qc), 0, n_samples // enr.SAMPLES_PER_BIN)
            if i in sleep:
                j = sleep.index(i)
                stored = np.load(os.path.join(flat, f'Neuron_raw_{rd}_sb_{j}.npy'))
                if not np.array_equal(stored, arr):
                    raise SystemExit(f'GATE {rd} sleep {j}: shard differs from Neuron_raw_{rd}_sb_{j}.npy')
                info['sleep_exact'] += 1
            else:
                k = wake.index(i)
                p = os.path.join(MIRROR, f'Neuron_raw_{rd}_{k}.npy')
                if os.path.exists(p):
                    m_flat = float(np.load(p).mean())
                    info['wake_mean_ratio'].append(round(float(arr.mean()) / m_flat, 3) if m_flat > 0 else None)
            dst = os.path.join(OVERLAY, f'binned_FR_dic_{rd}_{i}')
            if os.path.lexists(dst):
                os.unlink(dst)
            joblib.dump(arr.astype(np.float64), dst)
            info['shards'] += 1

        for name, vals in ((f'All_session_{rd}.npy', times), (f'All_session_behaviour_{rd}.npy', times),
                           (f'awake_session_{rd}.npy', wake_t), (f'awake_session_behaviour_{rd}.npy', wake_t),
                           (f'sleep_session_{rd}.npy', sleep_t)):
            dst = os.path.join(OVERLAY, name)
            if os.path.lexists(dst):
                os.unlink(dst)
            np.save(dst, vals)
        info['stale_wake_files'] = sorted(
            f for f in os.listdir(flat)
            if re.match(rf'^Neuron_raw_{re.escape(rd)}_(\d+)\.npy$', f)
            and int(re.match(rf'^Neuron_raw_{re.escape(rd)}_(\d+)\.npy$', f).group(1)) >= len(wake))
        manifest['lec_sleep'][rd] = info
        print(f'  {rd}: {len(table)} recordings = {len(wake)} task + {len(sleep)} sleep, '
              f'{info["shards"]} shards, sleep exact {info["sleep_exact"]}/{len(sleep)}, '
              f'wake mean ratio {np.nanmedian([x for x in info["wake_mean_ratio"] if x]):.2f}'
              + (f', STALE flat files {info["stale_wake_files"]}' if info['stale_wake_files'] else ''))


# -------------------------------------------------------------------------------- gates
def gate_lookups(manifest):
    """Emulate the cell-14 / cell-20 lookups on the rebuilt sheets against the overlay's arrays."""
    recdays = [str(x) for x in np.load(os.path.join(MIRROR, 'combined_ABCDonly_days.npy'))]
    sheets = {}

    def sheet(mouse, sleep=False):
        key = (mouse, sleep)
        if key not in sheets:
            arr = parse_like_cell11(os.path.join(OVERLAY, f'MetaData.xlsx - {mouse}{"_sleep" if sleep else ""}.csv'))
            hdr = list(arr[0])
            sheets[key] = (arr[1:, hdr.index('Date')], arr[1:, hdr.index('Ephys')])
        return sheets[key]

    ok, synthetic, bad = [], [], []
    for rd in recdays:
        mouse, d1, d2 = rd.split('_')
        if REGION == 'pfc':
            date2 = f'{d2[-4:]}-{d2[2:4]}-{d2[:2]}'          # ddmmyyyy -> ISO, his cell-14 formula
        else:
            date2 = f'{d2[:4]}-{d2[4:6]}-{d2[6:]}'            # yyyymmdd (LEC names)
        allts = np.load(os.path.join(OVERLAY, f'All_session_{rd}.npy'))
        awake = np.load(os.path.join(OVERLAY, f'awake_session_{rd}.npy'))
        slp = np.load(os.path.join(OVERLAY, f'sleep_session_{rd}.npy'))
        if not all(TS_RE.match(str(t)) for t in allts):
            synthetic.append(rd)
            continue
        td, te = sheet(mouse)
        sd, se = sheet(mouse, sleep=True)
        problems = []
        first_task = te[td == date2]
        if len(first_task) == 0 or first_task[0] not in awake:
            problems.append(f'cell 14: first day-2 task row {first_task[:1]} not in awake_session_')
        first_sleep = se[sd == date2]
        if len(first_sleep) == 0 or first_sleep[0] not in slp:
            problems.append(f'cell 20: first day-2 sleep row {first_sleep[:1]} not in sleep_session_')
        missing = [t for t in allts if t not in set(te) | set(se)]
        n_bfr = sum(1 for f in os.listdir(OVERLAY) if re.match(rf'^binned_FR_dic_{re.escape(rd)}_\d+$', f))
        if n_bfr != len(allts):
            problems.append(f'{n_bfr} binned_FR_dic shards vs {len(allts)} All_session_ entries')
        if set(awake) | set(slp) != set(allts) or set(awake) & set(slp):
            problems.append('awake/sleep sets do not partition All_session_')
        rec = dict(n_all=int(len(allts)), n_awake=int(len(awake)), n_sleep=int(len(slp)), n_shards=n_bfr,
                   tokens_not_in_sheets=missing, problems=problems)
        manifest['recdays'][rd] = rec
        (bad if problems else ok).append(rd)
    print(f'  lookups: {len(ok)} recdays pass, {len(synthetic)} on synthetic tokens (excluded by design), '
          f'{len(bad)} FAIL')
    for rd in bad:
        print(f'    FAIL {rd}: {manifest["recdays"][rd]["problems"]}')
    manifest['gate'] = dict(pass_=ok, synthetic=synthetic, fail=bad)
    if bad:
        raise SystemExit('GATE: cell-14/20 lookups fail on the recdays above')
    # Cell 14 has no try/except around its awake_session_ lookup, so a recday on synthetic
    # tokens aborts the notebook before the skip logic of cells 24/28 is reached. Figure7's
    # own day list therefore holds only the recdays whose sleep bookkeeping is consistent.
    days_path = os.path.join(OVERLAY, 'combined_ABCDonly_days.npy')
    if synthetic:
        if os.path.lexists(days_path):
            os.unlink(days_path)
        np.save(days_path, np.array(ok, dtype='<U32'))
        print(f'  combined_ABCDonly_days.npy in the overlay: {len(ok)} recdays (dropped {synthetic})')
    manifest['figure7_days'] = ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='re-create existing symlinks')
    ap.add_argument('--sleep', action='store_true', help='LEC: bin the shards and write real session arrays')
    args = ap.parse_args()
    if args.sleep and REGION != 'lec':
        raise SystemExit('--sleep is for the LEC copy')
    os.makedirs(LOGDIR, exist_ok=True)
    manifest = dict(region=REGION, overlay=OVERLAY, sheets={}, recdays={}, lec_sleep={}, sleep_notes={})
    print(f'[{REGION}] overlay')
    build_overlay(refresh=args.refresh)
    print(f'[{REGION}] sheets')
    (build_sheets_pfc if REGION == 'pfc' else build_sheets_lec)(manifest)
    if args.sleep:
        print(f'[{REGION}] sleep shards + session arrays')
        export_lec_sleep(manifest)
    if REGION == 'lec' and not args.sleep:
        print('  (LEC session arrays are the mirror symlinks until --sleep has run; lookups will report them synthetic)')
    print(f'[{REGION}] gates')
    gate_lookups(manifest)
    with open(os.path.join(LOGDIR, 'figure7_inputs_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, default=str)
    print(f'wrote {os.path.join(LOGDIR, "figure7_inputs_manifest.json")}')


if __name__ == '__main__':
    main()
