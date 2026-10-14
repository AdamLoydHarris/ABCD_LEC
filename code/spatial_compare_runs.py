"""Every number of one spatial run's shards against another run's, per recday (the equivalence control
for engine options whose default must preserve every existing number).

    python code/spatial_compare_runs.py --engine of --dataset mec --old 8a49b0e2d93c --new <tag>
    python code/spatial_compare_runs.py --self-test

Old shards are walked key by key with `spatial_common.compare_tree`; keys and list entries that only the
new shard has are allowed. Exit 0 only when every old recday is present, max abs diff is 0, no NaN
mismatches and nothing is missing. Reads pickles only. code/ only: not mirrored to mFC_data/code/.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SKIP = ('stamp', 'elapsed_s', 'null_timing')       # provenance and wall-clock time, not results


def run_dir(dataset: str, engine: str, tag: str) -> str:
    import spatial_io as sio
    if dataset != 'pfc':
        sio.set_dataset(dataset)
    return os.path.join(sio.output_root(dataset), engine, tag)


def equivalent(out: dict) -> bool:
    return out['max_diff'] == 0 and out['nan_mismatch'] == 0 and not out['missing']


def compare_runs(old_dir: str, new_dir: str, skip=SKIP) -> pd.DataFrame:
    """One row per shard of the old run: hosts, max abs diff, worst path, NaN mismatches, missing paths, n values."""
    import spatial_common as sc
    rows = []
    for f in sorted(glob.glob(os.path.join(old_dir, 'shards', '*.pkl'))):
        rd = os.path.basename(f)[:-4]
        g = os.path.join(new_dir, 'shards', f'{rd}.pkl')
        if not os.path.exists(g):
            rows.append(dict(recday=rd, present=False, equal=False))
            continue
        a, b = pickle.load(open(f, 'rb')), pickle.load(open(g, 'rb'))
        out = sc.compare_tree({k: v for k, v in a.items() if k not in skip}, {k: v for k, v in b.items() if k not in skip})
        rows.append(dict(recday=rd, present=True, old_host=a.get('stamp', {}).get('host', ''),
                         new_host=b.get('stamp', {}).get('host', ''), max_abs_diff=out['max_diff'], worst=out['worst'],
                         nan_mismatch=out['nan_mismatch'], n_missing=len(out['missing']),
                         first_missing=out['missing'][0] if out['missing'] else '', n_values=out['n'],
                         equal=equivalent(out)))
    return pd.DataFrame(rows)


def self_test() -> int:
    """The comparator accepts identical and extended shards and rejects each kind of difference."""
    import spatial_common as sc
    rng = np.random.default_rng(0)
    base = dict(units=pd.DataFrame(dict(u=np.arange(5), si=rng.random(5))),
                sessions={1: dict(real=dict(border=rng.random(5), hd_R=rng.random(5)), null=[rng.random(5) for _ in range(3)],
                                  label='openfield')},
                stamp=dict(host='a'), elapsed_s=1.0)

    def variant(fn):
        b = pickle.loads(pickle.dumps(base))
        fn(b)
        return b

    cases = [('identical', variant(lambda b: None), True),
             ('provenance differs', variant(lambda b: b.update(stamp=dict(host='b'), elapsed_s=9.0)), True),
             ('added key, column, list entry', variant(lambda b: (b.update(moves=[1.0]), b['units'].insert(2, 'new', 0.0),
                                                                   b['sessions'][1]['null'].append(np.zeros(5)))), True),
             ('one value off by 1e-15', variant(lambda b: b['sessions'][1]['real']['border'].__setitem__(2, b['sessions'][1]['real']['border'][2] + 1e-15)), False),
             ('number became NaN', variant(lambda b: b['sessions'][1]['real']['hd_R'].__setitem__(0, np.nan)), False),
             ('key removed', variant(lambda b: b['sessions'][1]['real'].pop('hd_R')), False),
             ('shape changed', variant(lambda b: b['sessions'][1]['real'].__setitem__('border', np.zeros(6))), False),
             ('list shortened', variant(lambda b: b['sessions'][1]['null'].pop()), False),
             ('column changed', variant(lambda b: b.__setitem__('units', b['units'].assign(si=b['units']['si'] * 2))), False),
             ('string changed', variant(lambda b: b['sessions'][1].__setitem__('label', 'object')), False)]
    fails = 0
    for name, b, want in cases:
        got = equivalent(sc.compare_tree({k: v for k, v in base.items() if k not in SKIP}, {k: v for k, v in b.items() if k not in SKIP}))
        ok = got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} comparator {name}: {'accepted' if got else 'rejected'}, expected {'accept' if want else 'reject'}")
    # the run walk: a recday missing from the new run fails, an extra new recday is ignored
    with tempfile.TemporaryDirectory() as tmp:
        old, new = os.path.join(tmp, 'old'), os.path.join(tmp, 'new')
        for d in (old, new):
            os.makedirs(os.path.join(d, 'shards'))
        for rd in ('r1', 'r2'):
            pickle.dump(base, open(os.path.join(old, 'shards', f'{rd}.pkl'), 'wb'))
        for rd in ('r1', 'r3'):
            pickle.dump(base, open(os.path.join(new, 'shards', f'{rd}.pkl'), 'wb'))
        df = compare_runs(old, new)
        ok = list(df['recday']) == ['r1', 'r2'] and list(df['equal']) == [True, False] and not df.set_index('recday').loc['r2', 'present']
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} run walk: r1 equal, r2 missing from the new run flagged, r3 ignored: {df[['recday', 'present', 'equal']].to_dict('records')}")
    print(f'{len(cases) + 1 - fails}/{len(cases) + 1} passed')
    return int(fails > 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--engine', choices=('of', 'maze', 'link'))
    ap.add_argument('--dataset', choices=('mec', 'lec', 'pfc'))
    ap.add_argument('--old')
    ap.add_argument('--new')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not all((args.engine, args.dataset, args.old, args.new)):
        ap.error('pass --engine, --dataset, --old and --new, or --self-test')
    df = compare_runs(run_dir(args.dataset, args.engine, args.old), run_dir(args.dataset, args.engine, args.new))
    if df.empty:
        print(f'no shards under run {args.old}')
        return 1
    pd.set_option('display.width', 250)
    print(df.to_string(index=False))
    n_eq = int(df['equal'].sum())
    print(f'{n_eq}/{len(df)} recdays of {args.old} reproduced exactly by {args.new}')
    return int(n_eq < len(df))


if __name__ == '__main__':
    sys.exit(main())
