"""Run one spatial engine on one recday (writes a shard), or merge a run's shards into tables.

    python code/run_spatial.py --engine of --dataset mec --recday bp01_20240402_20240403
    python code/run_spatial.py --engine of --dataset mec --merge --expect-recdays 9 --run-tag <tag>

TSA_DATASET is set from --dataset before anything imports the registry. PFC runs from the
`mFC_data/code/` copy. Shards go to `<dataset root>/data/spatial/<engine>/<run-tag>/shards/`; the
default tag hashes the config and the code, so a changed recipe never reuses a directory, and an
existing shard is never overwritten. `--config` takes a JSON of overrides (one file for every dataset).

This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ENGINES = {'of': 'spatial_tuning', 'maze': 'maze_spatial', 'link': 'maze_of_link'}
# the link engine (mEC only, code/ only) runs the other engines' code too, so it is hashed into its tag
EXTRA_CODE = {'link': ('spatial_tuning.py', 'maze_spatial.py', 'of_lag_scan.py')}
UNHASHED = ('n_jobs',)             # changes speed, not results: shifts are drawn before parallelising
# Haswell / Broadwell nodes: PFC reruns there differed at 1e-15 and flipped BIC field counts for 19 of
# 1229 units (run 1617859bc6fd), so production runs on the modern families only
LEGACY_HOSTS = ('gpu-350-', 'gpu-380-')


def _merge_cfg(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_cfg(out[k], v)
        else:
            if k not in out:
                raise KeyError(f'unknown config key {k!r}')
            out[k] = v
    return out


def setup(args):
    """Import the engine after TSA_DATASET is set; return (engine module, cfg, hashed cfg, code files, out dir)."""
    import spatial_io as sio
    sio.set_dataset(args.dataset)
    import importlib
    import spatial_common as sc
    eng = importlib.import_module(ENGINES[args.engine])
    cfg = copy.deepcopy(eng.DEFAULT_CFG)
    if args.config:
        with open(args.config) as fh:
            cfg = _merge_cfg(cfg, json.load(fh))
    if args.n_jobs:
        cfg['n_jobs'] = args.n_jobs
    elif os.environ.get('SLURM_CPUS_PER_TASK'):
        cfg['n_jobs'] = int(os.environ['SLURM_CPUS_PER_TASK'])
    hashed = {k: v for k, v in cfg.items() if k not in UNHASHED}
    code = [os.path.join(HERE, f) for f in ('spatial_common.py', 'spatial_io.py', ENGINES[args.engine] + '.py')
            + EXTRA_CODE.get(args.engine, ())]
    tag = args.run_tag or sc.run_tag(hashed, code)
    out = os.path.join(sio.output_root(args.dataset), args.engine, tag)
    return eng, cfg, hashed, code, out, tag


def run_recday(args) -> None:
    import spatial_io as sio
    import spatial_common as sc
    eng, cfg, hashed, code, out, tag = setup(args)
    t0 = time.time()
    import socket
    host = socket.gethostname()
    if host.startswith(LEGACY_HOSTS) and not args.allow_legacy_host:
        raise RuntimeError(f'{host} is a legacy node (results differ at 1e-15); resubmit with --exclude or pass --allow-legacy-host')
    code_hash0 = sc.file_hash(code)
    path = os.path.join(out, 'shards', f'{args.recday}.pkl')
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; pass a new --run-tag')
    views = sio.load(args.dataset, [args.recday], args.data_version)[args.recday]
    n_units = next(iter(views.values()))['counts'].shape[0]
    units = sio.unit_table(args.dataset, args.recday, n_units)
    print(f'{args.recday}: {len(views)} sessions, {n_units} units, run tag {tag}, n_jobs {cfg["n_jobs"]}', flush=True)
    if args.engine == 'of':
        # every measured wall; the engine applies each arena block's own limit
        wall_rms = {s: sio.wall_exclusions(args.dataset, args.recday, s, 0.0)
                    for s, v in views.items() if v.get('session_type') in sio.ARENA_TYPES}
        res = eng.run_of_recday(views, args.recday, units, cfg, wall_rms=wall_rms)
    elif args.engine == 'link':
        if args.dataset != 'mec':
            raise ValueError('the link engine is mEC only (PFC has no arena sessions; LEC has no maze-arena pairs in one sort)')
        import pickle
        import recday_registry as rr
        root = sio.output_root(args.dataset)
        of_shard = pickle.load(open(os.path.join(root, 'of', cfg['of_tag'], 'shards', f'{args.recday}.pkl'), 'rb'))
        maze_shard = pickle.load(open(os.path.join(root, 'maze', cfg['maze_tag'], 'shards', f'{args.recday}.pkl'), 'rb'))
        res = eng.run_link_recday(views, args.recday, units, cfg, of_shard, maze_shard, str(rr.DATASETS[args.dataset]['root']))
    else:
        res = eng.run_maze_recday(views, args.recday, units, cfg)
    res['stamp'] = sc.stamp(args.engine, hashed, code, args.dataset, args.data_version)
    # loky workers import the engine from disk when the null phase starts: an edit during the job would mix code
    if res['stamp']['code_hash'] != code_hash0:
        raise RuntimeError('engine code changed while this job ran; nothing written, resubmit')
    res['elapsed_s'] = time.time() - t0
    sc.write_new(path, res)
    print(f'wrote {path} in {res["elapsed_s"]:.0f} s', flush=True)


def merge(args) -> None:
    import spatial_common as sc
    import spatial_io as sio
    eng, cfg, hashed, code, out, tag = setup(args)
    shard_dir = os.path.join(out, 'shards')
    shards = sc.load_shards(shard_dir)
    stamp = sc.check_stamps(shards)
    if args.expect_recdays and len(shards) != args.expect_recdays:
        raise ValueError(f'{len(shards)} shards in {shard_dir}, expected {args.expect_recdays}; refusing a partial merge')
    if args.dataset != 'pfc':
        import recday_registry as rr
        for rd, sh in shards.items():
            if len(sh['units']) != rr.qc_unit_count(rd):
                raise ValueError(f'{rd}: {len(sh["units"])} unit rows vs {rr.qc_unit_count(rd)} QC units')
    tables = eng.merge_tables(shards)
    mdir = os.path.join(out, 'merged')
    os.makedirs(mdir, exist_ok=True)
    for name, df in tables.items():
        p = sc.versioned(os.path.join(mdir, f'{args.engine}_{name}.csv'))
        df.to_csv(p, index=False)
        print(f'wrote {p} ({len(df)} rows)')
    p = sc.versioned(os.path.join(mdir, f'{args.engine}_merged.pkl'))
    sc.write_new(p, dict(stamp=stamp, tables=tables, recdays=sorted(shards)))
    print(f'wrote {p}; stamp cfg {stamp.get("cfg_hash", "")[:10]} code {stamp.get("code_hash", "")[:10]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--engine', choices=sorted(ENGINES), required=True)
    ap.add_argument('--dataset', choices=('mec', 'lec', 'pfc'), required=True)
    ap.add_argument('--recday')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--expect-recdays', type=int, default=0)
    ap.add_argument('--run-tag', default='')
    ap.add_argument('--config', default='')
    ap.add_argument('--data-version', default='v4')
    ap.add_argument('--n-jobs', type=int, default=0)
    ap.add_argument('--print-tag', action='store_true', help='print the default run tag and exit')
    ap.add_argument('--allow-legacy-host', action='store_true', help='run on gpu-350/380 nodes anyway')
    args = ap.parse_args(argv)
    if args.print_tag:
        print(setup(args)[5])
        return
    if args.merge:
        merge(args)
    elif args.recday:
        run_recday(args)
    else:
        ap.error('pass --recday or --merge')


if __name__ == '__main__':
    main()
