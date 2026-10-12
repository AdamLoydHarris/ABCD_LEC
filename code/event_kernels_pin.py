"""Pin: `event_kernels.build_design` with its default `kin='full'` equals a committed version bit
for bit, on synthetic sessions with tracking gaps; `kin='strict'` changes only `kin_fast` (extra
columns), `kin_missing` (split by segment, rows summing to the default indicator) and the A /
reward columns on untracked rows (zeroed).

    python code/event_kernels_pin.py --against 57eac39        # exit 0 = pinned

No data is read. The committed module is extracted with `git show` into a temporary tree shaped
like the repo (it locates its root by the `code/` + `mFC_data/` pair).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import event_kernels as new                                            # noqa: E402


def _committed(rev: str, tmp: str):
    src = subprocess.run(['git', '-C', HERE, 'show', f'{rev}:code/event_kernels.py'],
                         capture_output=True, text=True, check=True).stdout
    os.makedirs(os.path.join(tmp, 'code'))
    os.makedirs(os.path.join(tmp, 'mFC_data'))
    path = os.path.join(tmp, 'code', f'event_kernels_{rev}.py')
    with open(path, 'w') as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(f'event_kernels_{rev}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_session(rng, T, n_units=5):
    xy = np.cumsum(rng.normal(0, 0.3, (T, 2)), axis=0) + 20
    gap = rng.integers(T // 4, T // 2)
    xy[gap:gap + 200] = np.nan                                         # an untracked stretch
    xy[:40] = np.nan                                                   # untracked start
    starts = np.sort(rng.choice(np.arange(400, T - 2000), 12, replace=False))
    tt = np.column_stack([starts + k * 150 for k in range(5)]).astype(np.int64)
    return dict(counts=rng.poisson(0.05, (n_units, T)).astype(np.uint16), xy=xy,
                locs=rng.integers(0, 22, T).astype(np.int16), trial_times=tt,
                first_A_bin=int(tt[0, 0]), pokes=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--against', required=True, help='git revision holding the reference code/event_kernels.py')
    args = ap.parse_args(argv)
    rng = np.random.default_rng(1)
    sessions = [fake_session(rng, T) for T in (24000, 30000, 27000)]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        old = _committed(args.against, tmp)
        for kw in (dict(), dict(bank_init='zero'), dict(events=('first_A', 'A', 'reward'))):
            a, b = old.build_design(sessions, **kw), new.build_design(sessions, **kw)
            same = (a['X'].shape == b['X'].shape and np.array_equal(a['X'], b['X'], equal_nan=True)
                    and a['names'] == b['names'] and a['groups'] == b['groups']
                    and np.array_equal(a['FR'], b['FR']))
            print(f"kin='full' {kw or 'defaults'}: X {a['X'].shape}, identical to {args.against}: {same}")
            ok &= same
        a, c = old.build_design(sessions), new.build_design(sessions, kin='strict')
    extra = [n for n in c['names'] if n not in a['names']]
    miss = a['kin_missing']
    fast_ev = tuple(f'ev_{e}:' for e in new.FAST_EVENTS)
    col = lambda D, n: D['X'][:, D['names'].index(n)]
    kept = all(np.array_equal(col(a, n), col(c, n), equal_nan=True)
               for n in a['names'] if not n.startswith(('kin_missing:',) + fast_ev))
    masked = all(np.array_equal(col(a, n)[~miss], col(c, n)[~miss]) and not col(c, n)[miss].any()
                 for n in a['names'] if n.startswith(fast_ev))
    only_kin = all(n.startswith(('kin_fast:', 'kin_missing:')) for n in extra)
    split = np.array_equal(c['X'][:, c['groups']['kin_missing']].sum(axis=1),
                           a['X'][:, a['groups']['kin_missing']].sum(axis=1))
    print(f"kin='strict': {len(extra)} new columns, all in kin_fast / kin_missing: {only_kin}; "
          f"other default columns unchanged: {kept}; A / reward equal on tracked rows and 0 on "
          f"{int(miss.sum())} untracked: {masked}; missing split sums to the default: {split}")
    ok &= only_kin and kept and masked and split
    print('PIN', 'OK' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
