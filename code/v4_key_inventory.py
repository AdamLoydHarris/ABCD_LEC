"""Which keys does each session carry, v1 vs v4? Tests the "structure is unchanged" claim.

Written 2026-10-09 after `w0_gates.py` died on `KeyError: 'Locs_raw'` under v4 having passed
its first two gates under v1. A missing key is either a real regression in the build or a
session that legitimately never had it; this says which, per session, rather than crashing on
the first one.

    python code/v4_key_inventory.py --versions v1 v4 --out <csv>
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', nargs='+', default=['v1', 'v4'])
    ap.add_argument('--out', default='data/_v4_migration_scratch/key_inventory.csv')
    args = ap.parse_args()

    import glm_analysis_v3 as glm
    import recday_registry as rr

    frames = []
    for ver in args.versions:
        version, path = rr.resolve_data_version(ver)
        print(f'--- {version}: {path.name}', flush=True)
        dd = glm.load_data_dic(path=str(path), validate=False, apply_exclusions=False,
                               verbose=True)
        for recday in sorted(dd):
            for s, sd in sorted(dd[recday].items(),
                                key=lambda kv: (str(type(kv[0])), kv[0])):
                if not isinstance(sd, dict):
                    continue
                frames.append({'version': version, 'recday': recday, 'session': s,
                               **{k: True for k in sd}})
        del dd
    df = pd.DataFrame(frames).fillna(False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    keys = [c for c in df.columns if c not in ('version', 'recday', 'session')]
    print(f'\n=== key presence, {len(df)} session rows ===')
    piv = df.groupby('version')[keys].sum().T
    piv['differs'] = piv.nunique(axis=1) > 1
    print(piv.sort_values('differs', ascending=False).to_string())

    if len(args.versions) == 2:
        a, b = args.versions
        A = df[df.version == a].set_index(['recday', 'session'])
        B = df[df.version == b].set_index(['recday', 'session'])
        both = A.index.intersection(B.index)
        print(f'\n=== per-session key losses {a} -> {b} ({len(both)} shared sessions) ===')
        lost = []
        for k in keys:
            m = A.loc[both, k].astype(bool) & ~B.loc[both, k].astype(bool)
            if m.any():
                lost.append((k, int(m.sum()), list(both[m])[:6]))
        gained = []
        for k in keys:
            m = ~A.loc[both, k].astype(bool) & B.loc[both, k].astype(bool)
            if m.any():
                gained.append((k, int(m.sum()), list(both[m])[:6]))
        for name, rows in (('LOST', lost), ('GAINED', gained)):
            print(f'  {name}:' + ('' if rows else ' none'))
            for k, n, ex in rows:
                print(f'    {k}: {n} session(s), e.g. {ex}')
    print(f'\nsaved {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
