#!/usr/bin/env python
"""
Batch driver for persistent (co)homology of one mouse_recday.

Designed to be launched per mouse_recday on SLURM (see sbatch_files/ph_lec.sbatch).
Loads the wake data_dic and/or the sleep-box .npy files, runs the heavy ripser +
shuffle-null + circular-coordinate computations, and writes one pickle per
session/state to --out. The notebook then only loads + plots these pickles.

Usage
-----
  python code/run_ph_batch.py <mouse_recday> <wake|sleep|both> [options]

Examples
--------
  python code/run_ph_batch.py ah08_20250613_20250615 both
  python code/run_ph_batch.py ah08_20250613_20250615 wake --quick      # smoke test

Outputs (in --out, default data/ph_outputs/)
  <mouse_recday>__wake_s<session>.pkl
  <mouse_recday>__sleep_sb<idx>.pkl
"""

import os
import sys
import time
import pickle
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import persistent_homology_analysis as ph     # noqa: E402
import glm_analysis_v2 as glm                  # noqa: E402  (get_sessions_for_glm)


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _save(obj, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {path}", flush=True)
    return path


def build_config(args):
    kw = dict(n_landmarks=args.n_landmarks, n_shuffles=args.n_shuffles,
              n_pca=args.n_pca, bin_ms=args.bin_ms, metric=args.metric,
              random_state=args.seed)
    if args.quick:
        kw.update(n_landmarks=min(args.n_landmarks, 500), n_shuffles=10)
    return ph.PHConfig(**kw)


def run(args):
    cfg = build_config(args)
    out_dir = _resolve(args.out)
    print(f"mouse_recday={args.mouse_recday}  state={args.state}", flush=True)
    print(f"config: {cfg.to_dict()}", flush=True)

    wake_results = {}
    primary_wake = None

    # ---------------------------------------------------------------- wake
    if args.state in ("wake", "both"):
        t0 = time.time()
        dd_path = _resolve(args.data_dic)
        print(f"loading wake data_dic: {dd_path} ...", flush=True)
        with open(dd_path, "rb") as f:
            data_dic = pickle.load(f)
        if args.mouse_recday not in data_dic:
            raise KeyError(f"{args.mouse_recday} not in data_dic "
                           f"(have {list(data_dic)[:5]}...)")
        recday = data_dic[args.mouse_recday]
        sessions, tasks = glm.get_sessions_for_glm(recday)
        if args.quick:
            sessions, tasks = sessions[:1], tasks[:1]
        print(f"loaded in {time.time()-t0:.0f}s; sessions={sessions} tasks={tasks}",
              flush=True)

        for sess, task in zip(sessions, tasks):
            t1 = time.time()
            print(f"[wake] session {sess} (task {task}) ...", flush=True)
            res = ph.analyse_wake_session(recday[sess], cfg,
                                          run_null=not args.no_null, decode=True)
            if res is None:
                print("  skipped (no usable neural data)", flush=True)
                continue
            res["mouse_recday"] = args.mouse_recday
            res["session"] = sess
            wake_results[sess] = res
            _report(res)
            _save(res, out_dir, f"{args.mouse_recday}__wake_s{sess}")
            print(f"  done in {time.time()-t1:.0f}s", flush=True)

        # primary wake model = the unique-task session with the most trials
        # (get_sessions_for_glm already orders by trial count, so sessions[0]).
        for sess in sessions:
            if sess in wake_results and "theta" in wake_results[sess]:
                primary_wake = wake_results[sess]
                break
        # free the big dict before the sleep stage
        del data_dic, recday

    # --------------------------------------------------------------- sleep
    if args.state in ("sleep", "both"):
        data_root = _resolve(args.data_root)
        sb_files = ph.load_sleep_files(args.mouse_recday, data_root)
        if args.quick:
            sb_files = sb_files[:1]
        print(f"[sleep] {len(sb_files)} sleep-box file(s) for "
              f"{args.mouse_recday}", flush=True)
        for sb_idx, sleep_raw in sb_files:
            t1 = time.time()
            print(f"[sleep] sb_{sb_idx}  shape={sleep_raw.shape} ...", flush=True)
            res = ph.analyse_sleep_session(sleep_raw, cfg,
                                           wake_result=primary_wake,
                                           run_null=not args.no_null)
            if res is None:
                print("  skipped (too short)", flush=True)
                continue
            res["mouse_recday"] = args.mouse_recday
            res["sb_idx"] = sb_idx
            if primary_wake is not None:
                res["wake_session"] = primary_wake.get("session")
            _report(res)
            _save(res, out_dir, f"{args.mouse_recday}__sleep_sb{sb_idx}")
            print(f"  done in {time.time()-t1:.0f}s", flush=True)

    print("ALL DONE", flush=True)


def _report(res):
    """Console one-liner with the headline topology numbers."""
    s = res["summary"][2]
    pv = res.get("pvalues", {})
    msg = (f"  H1 top={s[1]['top']:.3f} (ratio {s[1]['gap_ratio']:.1f}, "
           f"p={pv.get(1, float('nan')):.3f}) | "
           f"H2 top={s[2]['top']:.3f} (p={pv.get(2, float('nan')):.3f})")
    if "coord_vs_behaviour" in res:
        tp = res["coord_vs_behaviour"]["task_phase"]
        msg += f" | theta~task_phase R={tp['r']:.3f} p={tp['p']:.3f}"
    if "replay" in res:
        rp = res["replay"]
        msg += (f" | replay coverage={rp['coverage']:.2f} "
                f"continuity p={rp['p_continuity']:.3f} rev={rp['net_revolutions']:.1f}")
    print(msg, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mouse_recday")
    p.add_argument("state", choices=["wake", "sleep", "both"])
    p.add_argument("--data-dic", default="data/processed_data/data_dic_lec.pkl",
                   help="wake data_dic pickle (relative to repo root)")
    p.add_argument("--data-root", default="data",
                   help="data/ dir holding processed_data/neuron_raw_mingyutest")
    p.add_argument("--out", default="data/ph_outputs")
    p.add_argument("--n-landmarks", type=int, default=1200)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--n-pca", type=int, default=6)
    p.add_argument("--bin-ms", type=int, default=100)
    p.add_argument("--metric", default="euclidean", choices=["euclidean", "geodesic"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-null", action="store_true",
                   help="skip the shuffle significance null (much faster)")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke-test run (1 session, few landmarks/shuffles)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
