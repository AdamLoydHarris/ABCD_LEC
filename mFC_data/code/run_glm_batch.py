"""Fit ONE PFC recday's GLM, or merge the per-recday shards — mirror of `code/run_glm_batch.py`.

Two differences from the LEC copy, both forced by the dataset:

  * the data is a DIRECTORY, so `build_data_dic_from_pfc` loads only the requested recday
    instead of the LEC side's monolithic 3.8 GB pickle -- these jobs are far lighter;
  * PFC has no `unit_regions`, so the merge cannot run the anatomy key-contiguity check. It
    still refuses to merge overlapping shards, which is the failure that would actually lose
    data.

Usage
-----
    python mFC_data/code/run_glm_batch.py <recday> --section all_regressors [config flags]
    python mFC_data/code/run_glm_batch.py --merge --section all_regressors [config flags]
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MFC = os.path.abspath(os.path.join(HERE, '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEFAULT_SHARDS = os.path.join(MFC, 'glm_outputs', 'PFC_shards')
DEFAULT_OUT = os.path.join(MFC, 'glm_outputs', 'PFC')


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
    ap.add_argument('--regset', choices=('matched',), default='matched',
                    help="PFC can only fit the matched design (no HD, no pokes)")
    # V3 arms only (sections with `tfr_arm` / `cap_arm` in w1_refit.SECTIONS); ignored and
    # stamped as None for the v2 sections, whose config stamps must stay byte-identical.
    ap.add_argument('--leg-cap-s', type=float, default=30.0,
                    help='V3 arms: drop legs longer than this; also the time_from_reward '
                         'range under --tfr-scheme uniform')
    ap.add_argument('--tfr-scheme', choices=('uniform', 'decile'), default='uniform',
                    help='V3 core_progress_time: time_from_reward coding -- equal-width '
                         'bins in seconds over [0, leg cap], or the v2 quantile path')
    ap.add_argument('--tfr-bins', type=int, default=10,
                    help='V3 core_progress_time: number of time_from_reward bins')
    ap.add_argument('--normalise', choices=('session-centre', 'session-z', 'none'),
                    default='session-centre',
                    help='firing normalisation before the CV fit (w1_refit.NORMALISATIONS): '
                         'per-session offset only (production), offset AND gain, or neither. '
                         'A non-default choice appends _zscored / _nonorm to the section name')
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
    import importlib
    import w1_refit as w

    cfg = w.SECTIONS[args.section]
    # The engine is a property of the SECTION: v2 for every production fit on disk (frozen so
    # they stay reproducible), v3 for the reduced arms. See GLM_V3.md.
    engine = cfg.get('engine', 'v2')
    glm = importlib.import_module(f'glm_analysis_{engine}')
    regs = w.matched_regressors(cfg['regressors'])
    arm_kw = dict(leg_cap_s=args.leg_cap_s, tfr_scheme=args.tfr_scheme, tfr_bins=args.tfr_bins)
    extra = w.arm_extra(cfg, normalise=args.normalise, **arm_kw)
    norm_kw = w.norm_kwargs(args.normalise)
    overrides = w.binning_overrides_for(cfg, **arm_kw)
    max_leg = w.leg_cap_for(cfg, args.leg_cap_s)
    sect = w.section_name(args.section, width_ms=args.width_ms,
                          scheme=args.scheme, regset=args.regset, extra=extra)
    factor = max(1, int(round(args.width_ms / 25)))
    n_cols_override = {k: v['n_bins'] for k, v in (overrides or {}).items()} or None
    param = w.choose_parameterization(regs, engine=engine, n_cols_override=n_cols_override)
    v3_kwargs = {'binning_overrides': overrides} if engine != 'v2' else {}

    print(f'recday={recday}  section={sect}  engine={engine}')
    print(f'  {len(regs)} regressors, {args.width_ms} ms bins (factor {factor}), '
          f'{args.scheme} placement, {param}, leg cap {max_leg} s'
          + (f', time_from_reward {args.tfr_scheme} x{args.tfr_bins}' if overrides else '')
          + f', normalise={args.normalise} {norm_kw}')

    # Only this recday -- the PFC layout is per-file, so there is no whole-cohort load to pay.
    data_dic = glm.build_data_dic_from_pfc(w.DATA_FOLDER, [recday], verbose=True)
    if recday not in data_dic:
        raise KeyError(f'{recday} could not be built from {w.DATA_FOLDER}')

    joint = [j for j in cfg['joint_drop_groups']
             if all(r in regs for r in j[1])] or None

    t0 = time.time()
    out = glm.run_glm_analysis(
        [recday], data_dic,
        num_permutations=args.permutations,
        regressors_to_include=regs,
        joint_drop_groups=joint,
        filter_correct_paths=cfg.get('filter_correct_paths', False),
        max_transition_seconds=max_leg,
        compute_cpd=True,
        parameterization=param,
        downsample_factor=factor,
        downsample_mode='bin',
        continuous_binning=args.scheme,
        cross_validate=not args.no_cv,
        cv_n_perm=0 if args.no_cv else args.cv_perms,
        cv_nulls=tuple(args.nulls),
        **norm_kw,
        **v3_kwargs,
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
                 'max_transition_seconds': max_leg}
    if engine != 'v2':
        # Only the V3 arms carry these keys, so the v2 production stamps stay byte-identical
        # and a later v2 shard still merges with the ones on disk.
        tfr = (overrides or {}).get('time_from_reward')
        cfg_stamp.update({'engine': engine, 'leg_cap_s': max_leg,
                          'tfr_scheme': tfr['scheme'] if tfr else None,
                          'tfr_bins': tfr['n_bins'] if tfr else None,
                          'tfr_range_s': list(tfr['range_s']) if tfr and tfr['range_s'] else None})
    # Stamped ONLY when non-default, so every stamp already on disk stays byte-identical and a
    # later production shard still merges with the existing ones.
    if args.normalise != 'session-centre':
        cfg_stamp['normalise'] = args.normalise
    os.makedirs(args.shard_dir, exist_ok=True)
    for name, obj in zip(_artifacts(True, not args.no_cv), out):
        p = os.path.join(args.shard_dir, f'{sect}__{recday}__{name}.pkl')
        with open(p, 'wb') as fh:
            pickle.dump({'config': cfg_stamp, 'data': obj}, fh)
        print(f'  wrote {os.path.basename(p)}')
    print(f'  done in {el / 60:.1f} min')


def merge(args):
    import w1_refit as w

    cfg = w.SECTIONS[args.section]
    extra = w.arm_extra(cfg, leg_cap_s=args.leg_cap_s, tfr_scheme=args.tfr_scheme,
                        tfr_bins=args.tfr_bins, normalise=args.normalise)
    sect = w.section_name(args.section, width_ms=args.width_ms,
                          scheme=args.scheme, regset=args.regset, extra=extra)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'merging shards for {sect}')
    for name in _artifacts(True, not args.no_cv):
        shards = sorted(glob.glob(os.path.join(args.shard_dir, f'{sect}__*__{name}.pkl')))
        if not shards:
            print(f'  {name}: no shards found')
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
        print(f'  {name}: {len(shards)} shards -> {len(merged)} recdays -> '
              f'{os.path.basename(out)}')
        if seen_cfg:
            print(f'    config: {seen_cfg}')
    print('  (no anatomy key-contiguity check: PFC has no unit_regions)')


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
