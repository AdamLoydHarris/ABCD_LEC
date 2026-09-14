#!/usr/bin/env python3
"""Snapshot (md5) the files Figure5_Regression consumes, and compare two snapshots.

Why: the Figure2 re-run that adds the single-day cohort rewrites everything Figure2 saves.
Figure5_Regression's inputs should be unchanged by construction -- State_95 derives from the
deterministic State_zmax_bool, the prep arrays from the raw files plus the deterministic
Phases_raw2_, and the stochastic Phase_ is loaded but never used -- so the decision was to
KEEP the existing Figure 5 results and VERIFY rather than re-run 8.5 h. This is the verify.

    python 92_snapshot_inputs.py --snapshot before_rerun
    ... Figure2 re-run + bridge ...
    python 92_snapshot_inputs.py --snapshot after_rerun
    python 92_snapshot_inputs.py --compare before_rerun after_rerun    # exit 1 on any diff
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
MIRROR = os.path.join(ROOT, 'lec_replication_run', 'data', 'Intermediate_objects')
LOGDIR = os.path.join(ROOT, 'lec_replication_run', 'logs')

# Everything Figure5_Regression cells 15/21/26/32/38 np.load from Input_folder, restricted to
# the combined recdays it iterates. Phase_ is included deliberately: it is EXPECTED to differ
# (unseeded shuffle) and is reported separately so a diff there is not mistaken for a problem.
FAMILIES_MUST_MATCH = [
    r'^State_95', r'^State_99', r'^State_zmax_', r'^tuning_phase_boolean_max_',
    r'^Phases_raw2_', r'^States_raw_', r'^Times_from_reward_', r'^trialtimes_',
    r'^Neuron_raw_', r'^Location_raw_', r'^XY_raw_', r'^speed_', r'^Location_(?!raw_)',
    r'^Task_data_', r'^Num_trials_', r'^Task_num_', r'^awake_session_behaviour_',
    r'^combined_ABCDonly_days', r'^GLM_anchoring_prep_dic_',
]
FAMILIES_MAY_DIFFER = [r'^Phase_']
COMBINED = re.compile(r'(ah|ly)\d\d_\d{8}_\d{8}')


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def select(names):
    out = {}
    for n in names:
        if not n.endswith('.npy') or not COMBINED.search(n):
            continue
        for pat in FAMILIES_MUST_MATCH:
            if re.match(pat, n):
                out[n] = 'must'
                break
        else:
            for pat in FAMILIES_MAY_DIFFER:
                if re.match(pat, n):
                    out[n] = 'may'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', metavar='TAG')
    ap.add_argument('--compare', nargs=2, metavar=('TAG_A', 'TAG_B'))
    args = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)

    if args.snapshot:
        sel = select(os.listdir(MIRROR))
        snap = {}
        for i, (n, cls) in enumerate(sorted(sel.items())):
            p = os.path.join(MIRROR, n)
            snap[n] = dict(cls=cls, md5=md5(p), size=os.path.getsize(p))
            if i % 500 == 0:
                print(f'  {i}/{len(sel)} hashed', flush=True)
        out = os.path.join(LOGDIR, f'f5_inputs_{args.snapshot}.json')
        json.dump(snap, open(out, 'w'), indent=1)
        must = sum(v['cls'] == 'must' for v in snap.values())
        print(f'snapshot {args.snapshot}: {len(snap)} files ({must} must-match, {len(snap) - must} may-differ) -> {out}')
        return 0

    if args.compare:
        a, b = [json.load(open(os.path.join(LOGDIR, f'f5_inputs_{t}.json'))) for t in args.compare]
        diff_must, diff_may, only_a, only_b = [], [], [], []
        for n in sorted(set(a) | set(b)):
            if n not in a:
                only_b.append(n); continue
            if n not in b:
                only_a.append(n); continue
            if a[n]['md5'] != b[n]['md5']:
                (diff_must if a[n]['cls'] == 'must' else diff_may).append(n)
        same_must = sum(1 for n in a if n in b and a[n]['cls'] == 'must' and a[n]['md5'] == b[n]['md5'])
        print(f'compare {args.compare[0]} -> {args.compare[1]}:')
        print(f'  must-match files identical : {same_must}')
        print(f'  must-match files DIFFERENT : {len(diff_must)}')
        for n in diff_must[:30]:
            print(f'      {n}')
        print(f'  may-differ (Phase_) changed: {len(diff_may)}  (expected: unseeded shuffle)')
        print(f'  only in {args.compare[0]}: {len(only_a)};  only in {args.compare[1]}: {len(only_b)}')
        for n in (only_a + only_b)[:10]:
            print(f'      {n}')
        json.dump(dict(diff_must=diff_must, diff_may=diff_may, only_a=only_a, only_b=only_b,
                       same_must=same_must),
                  open(os.path.join(LOGDIR, f'f5_inputs_compare_{args.compare[0]}_{args.compare[1]}.json'), 'w'), indent=1)
        return 1 if (diff_must or only_a) else 0

    ap.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
