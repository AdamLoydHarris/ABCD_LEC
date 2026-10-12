"""Compare two GLM shard sets artefact by artefact: the v4 migration's §1.5.1 gate.

Two uses:

  * `--expect-identical` (the refactor gate): a shard refitted with `--data-version v1` must
    reproduce its September original to `max|diff| = 0`. Anything non-zero means the
    `--data-version` plumbing changed a number, not just a path.
  * the default (the asymmetry test): v1 vs v4 on the same recday, reporting how far each
    artefact moved. `ly05_20250620_20250623` should move far more than any other recday.

`permutation_results` is EXCLUDED from the identity claim and reported separately.
`run_glm_analysis` draws its in-sample shifts with an unseeded `np.random.randint`
(glm_analysis_v3.py:2151), so those p-values are not reproducible run-to-run; the seeded CV
path (glm_cv.py:296) is. `elapsed_s` is wall clock and always differs.

    python code/v4_equivalence_check.py --a <dir> --b <dir> --recday <rd> --expect-identical
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from glm_v3_synthetics import max_abs_diff as _max_abs_diff      # noqa: E402


def max_abs_diff(a, b, path='', ignore=()):
    """Recursive max|A - B|, with equal values compared equal BEFORE any subtraction.

    `glm_v3_synthetics.max_abs_diff` goes straight to `|A - B|`, and `-inf - -inf` is NaN,
    which it reports as "NaN in one result but not the other". V3's `bin_occupancy` stores
    its outer bin edges as +/-inf, so an IDENTICAL uniform-tfr shard fails against itself;
    the same block also holds string fields, which cannot be subtracted at all. Both are
    comparer bugs, not differences, and both are safe (a false FAIL, never a false PASS),
    so the shared byte-mirrored module is left alone and this one is used instead."""
    if _identical(a, b, ignore):
        return 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        shared = [k for k in a if k in b and k not in ignore]
        missing = ((set(a) ^ set(b)) - set(ignore))
        if missing:
            raise AssertionError(f'{path}: keys in one result only: {sorted(map(str, missing))}')
        vals = [max_abs_diff(a[k], b[k], f'{path}/{k}', ignore) for k in shared]
        return max(vals) if vals else 0.0
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f'{path}: length {len(a)} vs {len(b)}')
        vals = [max_abs_diff(x, y, f'{path}[{i}]', ignore) for i, (x, y) in enumerate(zip(a, b))]
        return max(vals) if vals else 0.0
    A, B = np.atleast_1d(np.asarray(a)), np.atleast_1d(np.asarray(b))
    if A.shape != B.shape:
        raise AssertionError(f'{path}: shape {A.shape} vs {B.shape}')
    if A.dtype.kind not in 'fciub' or B.dtype.kind not in 'fciub':
        raise AssertionError(f'{path}: non-numeric and unequal ({A.dtype} / {B.dtype})')
    A, B = A.astype(float), B.astype(float)
    same = (A == B) | (np.isnan(A) & np.isnan(B))     # +/-inf and NaN agree with themselves
    d = np.zeros(A.shape, float)
    rest = ~same
    if rest.any():
        d[rest] = np.abs(A[rest] - B[rest])
        if np.isnan(d[rest]).any():
            raise AssertionError(f'{path}: NaN in one result but not the other')
    return float(np.max(d)) if d.size else 0.0


def _identical(a, b, ignore=()):
    """Exact structural equality, treating +/-inf and strings as comparable."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka = [k for k in a if k not in ignore]
        kb = [k for k in b if k not in ignore]
        return sorted(map(str, ka)) == sorted(map(str, kb)) and all(
            _identical(a[k], b[k], ignore) for k in ka)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_identical(x, y, ignore) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
            return bool(np.array_equal(a, b, equal_nan=True))
        return bool(np.array_equal(a, b))
    try:
        return bool(a == b) or (a != a and b != b)          # NaN == NaN
    except Exception:                                        # noqa: BLE001
        return False

ARTEFACTS = ('glm_results', 'cpd_results', 'cv_results', 'permutation_results')
#: not evidence of a difference: wall clock, and the stamp key this migration adds
IGNORE = ('elapsed_s',)
NONDETERMINISTIC = ('permutation_results',)


def _find(d, section_stem, recday, artefact):
    g = sorted(glob.glob(os.path.join(d, f'{section_stem}__{recday}__{artefact}.pkl')))
    if not g:
        g = sorted(glob.glob(os.path.join(d, f'*{recday}__{artefact}.pkl')))
    return g[-1] if g else None


def _load(p):
    with open(p, 'rb') as fh:
        return pickle.load(fh)


def compare_config(ca, cb):
    """Key-by-key, separating new keys from changed values."""
    added = {k: cb[k] for k in cb if k not in ca}
    removed = {k: ca[k] for k in ca if k not in cb}
    changed = {k: (ca[k], cb[k]) for k in ca if k in cb and ca[k] != cb[k]}
    return added, removed, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='reference shard dir (e.g. the September LEC_shards)')
    ap.add_argument('--b', required=True, help='new shard dir')
    ap.add_argument('--recday', required=True)
    ap.add_argument('--section-stem', default='core_progress_time__matched_250ms_decile_cap30s_tfrU10b')
    ap.add_argument('--expect-identical', action='store_true',
                    help='exit 1 unless every deterministic artefact matches at exactly 0')
    args = ap.parse_args()

    print(f'A: {args.a}\nB: {args.b}\nrecday: {args.recday}\n')
    worst, missing, fails = {}, [], []
    for art in ARTEFACTS:
        pa, pb = (_find(d, args.section_stem, args.recday, art) for d in (args.a, args.b))
        if not pa or not pb:
            missing.append(f'{art}: ' + ('A missing' if not pa else '') + ('B missing' if not pb else ''))
            continue
        A, B = _load(pa), _load(pb)
        added, removed, changed = compare_config(A.get('config', {}), B.get('config', {}))
        try:
            d = max_abs_diff(A.get('data'), B.get('data'), path=art, ignore=IGNORE)
        except AssertionError as exc:
            d = float('nan')
            fails.append(f'{art}: {exc}')
        worst[art] = d
        tag = '  (nondeterministic, not a gate)' if art in NONDETERMINISTIC else ''
        print(f'{art:22s} max|diff| = {d:.6g}{tag}')
        if added:
            print(f'{"":22s} config keys added: {added}')
        if removed:
            print(f'{"":22s} config keys REMOVED: {removed}')
        if changed:
            print(f'{"":22s} config CHANGED: {changed}')

    if missing:
        print('\nmissing artefacts:')
        for m in missing:
            print('  ' + m)
    if fails:
        print('\nstructural mismatches:')
        for f in fails:
            print('  ' + f)

    gated = {k: v for k, v in worst.items() if k not in NONDETERMINISTIC}
    print()
    if args.expect_identical:
        bad = {k: v for k, v in gated.items() if not (v == 0)}
        if bad or missing or fails:
            print(f'EQUIVALENCE FAILED: {bad or ""} {missing or ""} {fails or ""}')
            return 1
        print(f'EQUIVALENCE OK: {", ".join(gated)} all max|diff| = 0 '
              f'({len(gated)} artefacts); permutation_results excluded (unseeded shifts)')
        return 0
    print('summary (deterministic artefacts): '
          + ', '.join(f'{k} {v:.4g}' for k, v in gated.items()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
