"""Fit ONE recday's GLM, or merge the per-recday shards back into section pickles.

Designed to be launched per recday on SLURM (see `sbatch_files/glm_lec.sbatch`), mirroring
`run_ph_batch.py`.

Why shards
----------
`run_or_load_glm` writes ONE pickle per section holding every recday. Twenty-five parallel
jobs writing that same path would race and clobber each other, and the loser's recdays would
vanish silently -- the pickle would still load, just with fewer keys. So each job writes its
own `{section}__{recday}__{artifact}.pkl` under `--shard-dir`, and `--merge` combines them
into the section-level dicts the rest of the pipeline expects.

The merge is also where the join contract is enforced: it asserts every shard's per-neuron
keys are contiguous `0..n-1`, because `compute_tuning_arrays` writes row *k* for the *k*-th
sorted key, so a gap would shift every later neuron relative to `Neuron_raw` and break the
join to `unit_regions`.

Usage
-----
    python code/run_glm_batch.py <recday> --section all_regressors [config flags]
    python code/run_glm_batch.py --merge --section all_regressors [config flags]
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEFAULT_SHARDS = os.path.join(REPO, 'data', 'glm_outputs', 'LEC_shards')
DEFAULT_OUT = os.path.join(REPO, 'data', 'glm_outputs', 'LEC')


def _artifacts(compute_cpd, cross_validate):
    """Names in the order `run_glm_analysis` returns them."""
    names = ['glm_results', 'permutation_results']
    if compute_cpd:
        names.append('cpd_results')
    if cross_validate:
        names.append('cv_results')
    return names


def add_config_args(ap):
    ap.add_argument('--section', default='all_regressors')
    ap.add_argument('--width-ms', type=int, default=250)
    ap.add_argument('--scheme', choices=('decile', 'uniform'), default='decile')
    ap.add_argument('--regset', choices=('full', 'matched'), default='full')
    ap.add_argument('--permutations', type=int, default=100)
    ap.add_argument('--cv-perms', type=int, default=100)
    ap.add_argument('--no-cv', action='store_true')
    ap.add_argument('--nulls', nargs='+', default=['freedman_lane'],
                    choices=['shuffle', 'freedman_lane', 'column'],
                    help='permutation nulls to compute; see glm_cv.NULL_METHODS')
    ap.add_argument('--shard-dir', default=DEFAULT_SHARDS)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    return ap


def fit_one(recday, args):
    import glm_analysis_v2 as glm
    import w1_refit as w

    cfg = w.SECTIONS[args.section]
    regs = cfg['regressors']
    if args.regset == 'matched' and regs is not None:
        regs = w.matched_regressors(regs)
    sect = w.section_name(args.section, width_ms=args.width_ms,
                          scheme=args.scheme, regset=args.regset)
    factor = max(1, int(round(args.width_ms / 25)))
    param = w.choose_parameterization(regs)

    print(f'recday={recday}  section={sect}')
    print(f'  {len(regs) if regs else 9} regressors, {args.width_ms} ms bins '
          f'(factor {factor}), {args.scheme} placement, {param}')

    data_dic = glm.load_data_dic(validate=True, apply_exclusions=True, verbose=True)
    if recday not in data_dic:
        raise KeyError(f'{recday} not in data_dic ({len(data_dic)} recdays)')

    # Without this the poke regressors are all-zero: CPD exactly 0.00000 and one lost rank
    # each. Only attach when the design actually uses them.
    if regs and any('poke' in r for r in regs):
        n = glm.attach_pokes(data_dic, verbose=False)
        print(f'  attach_pokes: {n} session tables')

    joint = [j for j in cfg['joint_drop_groups']
             if all(r in (regs or []) for r in j[1])] or None

    t0 = time.time()
    out = glm.run_glm_analysis(
        [recday], data_dic,
        num_permutations=args.permutations,
        regressors_to_include=regs,
        joint_drop_groups=joint,
        filter_correct_paths=cfg.get('filter_correct_paths', False),
        max_transition_seconds=cfg.get('max_transition_seconds'),
        compute_cpd=True,
        parameterization=param,
        downsample_factor=factor,
        downsample_mode='bin',
        continuous_binning=args.scheme,
        cross_validate=not args.no_cv,
        cv_n_perm=0 if args.no_cv else args.cv_perms,
        cv_nulls=tuple(args.nulls),
        cv_center_within_sessions=True,
    )
    el = time.time() - t0

    # Stamp the configuration into every shard. The merge only refuses DUPLICATE recdays;
    # without this it would happily combine a 2-permutation recday with 100-permutation ones
    # (exactly what a hand-run smoke test produces) and nothing downstream would show it.
    cfg_stamp = {'section': args.section, 'width_ms': args.width_ms, 'scheme': args.scheme,
                 'regset': args.regset, 'permutations': args.permutations,
                 'cv_perms': 0 if args.no_cv else args.cv_perms,
                 'nulls': list(args.nulls),
                 'downsample_mode': 'bin',
                 'filter_correct_paths': cfg.get('filter_correct_paths', False),
                 'max_transition_seconds': cfg.get('max_transition_seconds')}
    os.makedirs(args.shard_dir, exist_ok=True)
    names = _artifacts(True, not args.no_cv)
    for name, obj in zip(names, out):
        p = os.path.join(args.shard_dir, f'{sect}__{recday}__{name}.pkl')
        with open(p, 'wb') as fh:
            pickle.dump({'config': cfg_stamp, 'data': obj}, fh)
        print(f'  wrote {os.path.basename(p)}')
    print(f'  done in {el / 60:.1f} min')


def merge(args):
    import w1_refit as w

    sect = w.section_name(args.section, width_ms=args.width_ms,
                          scheme=args.scheme, regset=args.regset)
    names = _artifacts(True, not args.no_cv)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'merging shards for {sect}')
    summary = {}
    for name in names:
        pat = os.path.join(args.shard_dir, f'{sect}__*__{name}.pkl')
        shards = sorted(glob.glob(pat))
        if not shards:
            print(f'  {name}: no shards found ({pat})')
            continue
        merged = {}
        seen_cfg = None
        for p in shards:
            with open(p, 'rb') as fh:
                raw = pickle.load(fh)
            if isinstance(raw, dict) and set(raw) == {'config', 'data'}:
                shard_cfg, d = raw['config'], raw['data']
                if seen_cfg is None:
                    seen_cfg = shard_cfg
                elif shard_cfg != seen_cfg:
                    diff = {k: (seen_cfg.get(k), shard_cfg.get(k))
                            for k in set(seen_cfg) | set(shard_cfg)
                            if seen_cfg.get(k) != shard_cfg.get(k)}
                    raise ValueError(
                        f'{name}: {os.path.basename(p)} was fitted with a different '
                        f'configuration — refusing to merge. Differences (first, this): '
                        f'{diff}')
            else:
                raise ValueError(f'{name}: {os.path.basename(p)} predates config stamping; '
                                 f'delete the shard directory and refit.')
            overlap = set(d) & set(merged)
            if overlap:
                raise ValueError(f'{name}: recday(s) {sorted(overlap)} appear in more than '
                                 f'one shard — refusing to merge silently')
            merged.update(d)
        out = os.path.join(args.out_dir, f'{sect}__{name}.pkl')
        with open(out, 'wb') as fh:
            pickle.dump(merged, fh)
        summary[name] = len(merged)
        print(f'  {name}: {len(shards)} shards -> {len(merged)} recdays -> '
              f'{os.path.basename(out)}')
        if seen_cfg:
            print(f'    config: {seen_cfg}')

    # The join contract. `compute_tuning_arrays` writes row k for the k-th SORTED key, so a
    # gap in the keys shifts every later neuron relative to Neuron_raw and silently breaks
    # the join to unit_regions. Cheaper to catch here than in a figure.
    gpath = os.path.join(args.out_dir, f'{sect}__glm_results.pkl')
    if os.path.exists(gpath):
        try:
            import anatomy_split as asp
            with open(gpath, 'rb') as fh:
                g = pickle.load(fh)
            ok = asp.assert_glm_keys_contiguous(g, asp.load_unit_regions(), strict=False)
            print(f'  key-contiguity check: {"PASS" if ok else "FAIL — see above"}')
        except Exception as exc:                                   # noqa: BLE001
            print(f'  key-contiguity check skipped ({exc})')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recday', nargs='?', default=None)
    ap.add_argument('--merge', action='store_true')
    add_config_args(ap)
    args = ap.parse_args()

    if args.merge:
        merge(args)
    elif args.recday:
        fit_one(args.recday, args)
    else:
        ap.error('give a recday, or --merge')


if __name__ == '__main__':
    main()
