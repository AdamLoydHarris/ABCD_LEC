#!/usr/bin/env python3
"""Are the tracking gaps SLEAP failures, or frames the QC removed?

    python 89_tracking_gap_provenance.py
    -> logs/tracking_gap_provenance.csv, and the split on stdout

`tracking_clean.py` writes a per-frame, per-bodypart int8 flag beside each SLEAP session in
data/preprocessed/sleap/maze_v2/: tens digit = outcome (0 kept, 1 bridged, 2 left NaN), units
digit = reason (1 raw NaN from SLEAP, 2/3 low score kept, 4 R1 velocity, 5 R2 spike,
6 R3 Hampel, 7 low score AND off trajectory). So the question is answerable per frame rather
than by inference: reason 1 is a SLEAP failure, 4-7 are QC removals.

Why it matters: a SLEAP failure is a data limitation and nothing to do about it short of
re-tracking. A large share of QC removals would say the velocity or Hampel thresholds are too
aggressive on these sessions, and the gaps are partly self-inflicted.

Position is `head_back` (what sleap_to_rois turns into the exported XY, and tracking_clean's own
default in report_row). Frames are the 60 fps video; the export's bins are 25 ms, so counts here
are frames, not bins, and are ~1.5x the bin counts in tracking_gaps.csv.

I/O: reads logs/tracking_gaps.csv, recday_registry for the session timestamps, and the flag
CSVs; writes logs/tracking_gap_provenance.csv.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
LOGDIR = os.path.join(ROOT, 'mFC_data', 'lec_replication_run_v4', 'logs')
SLEAP = os.path.join(ROOT, 'data', 'preprocessed', 'sleap', 'maze_v2')
BODYPART = 'head_back'
REASON = {0: 'ok', 1: 'raw_nan (SLEAP)', 2: 'low_point_kept', 3: 'low_instance_kept',
          4: 'r1_velocity (QC)', 5: 'r2_spike (QC)', 6: 'r3_hampel (QC)',
          7: 'lowscore_off_trajectory (QC)'}
QC_REASONS = (4, 5, 6, 7)
STAMP = re.compile(r'(\d{4}-\d{2}-\d{2})-(\d{6})')


def flag_file(mouse, date, hhmmss):
    """The SLEAP session whose video starts at or just before the pyControl session."""
    best, best_dt = None, None
    for f in sorted(os.listdir(os.path.join(SLEAP, mouse))) if os.path.isdir(os.path.join(SLEAP, mouse)) else []:
        if not f.endswith('track_flags.csv'):
            continue
        m = STAMP.search(f)
        if not m or m.group(1) != date:
            continue
        dt = int(hhmmss) - int(m.group(2))
        if dt < 0:                      # video starts after the session: not this one
            continue
        if best_dt is None or dt < best_dt:
            best, best_dt = os.path.join(SLEAP, mouse, f), dt
    return best, best_dt


def main():
    src = os.path.join(LOGDIR, 'tracking_gaps.csv')
    if not os.path.exists(src):
        print(f'NOT RUN: {src} missing')
        return 1
    sys.path.insert(0, os.path.join(ROOT, 'code'))
    import recday_registry as rr

    sessions = sorted({(r['recday'], int(r['session'])) for r in csv.DictReader(open(src))})
    rows, agg = [], Counter()
    unresolved = []
    for rd, s in sessions:
        mouse = rd.split('_')[0]
        try:
            stem = os.path.basename(str(rr.pycontrol_file(rd, s)))
            m = STAMP.search(stem)
            date, hhmmss = m.group(1), m.group(2)
        except Exception as e:                                  # noqa: BLE001
            unresolved.append(f'{rd}_{s}: no pyControl stamp ({type(e).__name__})')
            continue
        f, dt = flag_file(mouse, date, hhmmss)
        if f is None:
            unresolved.append(f'{rd}_{s}: no flag file for {mouse} {date} <= {hhmmss}')
            continue
        col = None
        with open(f) as fh:
            rdr = csv.reader(fh)
            head = next(rdr)
            if BODYPART not in head:
                unresolved.append(f'{rd}_{s}: {BODYPART} not in {f}')
                continue
            col = head.index(BODYPART)
            flags = np.array([int(line[col]) for line in rdr if line], dtype=int)
        out, rea = flags // 10, flags % 10
        left = rea[out == 2]
        c = Counter(left.tolist())
        agg.update(c)
        n_left, n_bridged = int((out == 2).sum()), int((out == 1).sum())
        raw = c.get(1, 0)
        qc = sum(c.get(k, 0) for k in QC_REASONS)
        rows.append(dict(recday=rd, session=s, video=os.path.basename(f), video_lead_s=dt,
                         n_frames=len(flags), n_left_nan=n_left, n_bridged=n_bridged,
                         left_raw_nan=raw, left_qc=qc,
                         pct_qc_of_left=round(100 * qc / n_left, 1) if n_left else None,
                         **{f'r{k}_{REASON[k].split()[0]}': c.get(k, 0) for k in (1, 4, 5, 6, 7)}))

    if not rows:
        print('NOT RUN: no session resolved to a flag file')
        for u in unresolved:
            print('  ', u)
        return 1

    tot_left = sum(r['n_left_nan'] for r in rows)
    tot_raw = sum(r['left_raw_nan'] for r in rows)
    tot_qc = sum(r['left_qc'] for r in rows)
    tot_bridged = sum(r['n_bridged'] for r in rows)
    print(f'=== provenance of the left-NaN frames, {len(rows)} sessions with gaps ===')
    print(f'  {"session":34} {"frames":>8} {"leftNaN":>8} {"rawNaN":>8} {"QC":>7} {"QC%":>6}  bridged')
    for r in sorted(rows, key=lambda x: -x['n_left_nan']):
        print(f"  {r['recday']+'_'+str(r['session']):34} {r['n_frames']:8d} {r['n_left_nan']:8d} "
              f"{r['left_raw_nan']:8d} {r['left_qc']:7d} {str(r['pct_qc_of_left']):>6}  {r['n_bridged']}")
    print(f'\n  TOTAL left-NaN frames {tot_left}: raw SLEAP {tot_raw} ({100*tot_raw/tot_left:.1f} %), '
          f'QC-removed {tot_qc} ({100*tot_qc/tot_left:.1f} %)')
    print('  by reason: ' + ', '.join(f'{REASON[k]} {v}' for k, v in sorted(agg.items())))
    print(f'  separately, {tot_bridged} frames were BRIDGED (gaps <= 2 s, silently interpolated)')
    if unresolved:
        print(f'  unresolved sessions ({len(unresolved)}):')
        for u in unresolved:
            print('    ', u)

    dst = os.path.join(LOGDIR, 'tracking_gap_provenance.csv')
    with open(dst, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in sorted(rows, key=lambda x: -x['n_left_nan']):
            w.writerow(r)
    print(f'  wrote {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
