#!/usr/bin/env python
"""
Batch driver for persistent (co)homology of one PFC mouse_recday (wake + sleep).

PFC sibling of code/run_ph_batch.py. Differences from the LEC driver:
  * wake data_dic is built on the fly with build_data_dic_from_pfc (no pickle);
  * sleep is loaded from joblib `binned_FR_dic_*` files (load_pfc_sleep_files);
  * a min_neurons filter drops the many small PFC recdays;
  * PFC has no head direction (theta-vs-HD is NaN, handled in the core module).

Usage
-----
  python mFC_data/code/run_ph_batch.py <recday> <wake|sleep|both> [options]
  python mFC_data/code/run_ph_batch.py ah04_01122021_02122021 both --quick

Outputs (in --out, default mFC_data/data/ph_outputs/)
  <recday>__wake_s<session>.pkl
  <recday>__sleep_sb<idx>.pkl
"""

import os
import sys
import time
import pickle
import argparse
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))   # mFC_data/code -> repo root
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import persistent_homology_analysis as ph     # noqa: E402
import glm_analysis_v2 as glm                  # noqa: E402  (build_data_dic_from_pfc)


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
              min_neurons=args.min_neurons, random_state=args.seed)
    if args.quick:
        kw.update(n_landmarks=min(args.n_landmarks, 500), n_shuffles=10)
    return ph.PHConfig(**kw)


def run(args):
    cfg = build_config(args)
    out_dir = _resolve(args.out)
    data_folder = _resolve(args.data_folder)
    print(f"PFC recday={args.mouse_recday}  state={args.state}", flush=True)
    print(f"config: {cfg.to_dict()}", flush=True)

    wake_results = {}
    primary_wake = None
    wake_n_neurons = None

    # ---------------------------------------------------------------- wake
    if args.state in ("wake", "both"):
        t0 = time.time()
        print(f"building PFC wake data_dic from {data_folder} ...", flush=True)
        data_dic = glm.build_data_dic_from_pfc(data_folder, [args.mouse_recday],
                                               verbose=True)
        if args.mouse_recday not in data_dic:
            raise KeyError(f"{args.mouse_recday} produced no usable sessions "
                           f"(missing files or empty).")
        recday = data_dic[args.mouse_recday]
        sessions, tasks = glm.get_sessions_for_glm(recday)
        if args.quick:
            sessions, tasks = sessions[:1], tasks[:1]
        print(f"built in {time.time()-t0:.0f}s; sessions={sessions} tasks={tasks}",
              flush=True)

        for sess, task in zip(sessions, tasks):
            t1 = time.time()
            print(f"[wake] session {sess} (task {task}) ...", flush=True)
            res = ph.analyse_wake_session(recday[sess], cfg,
                                          run_null=not args.no_null, decode=True)
            if res is None:
                n = np.asarray(recday[sess].get("Neuron_raw")).shape[0] \
                    if recday[sess].get("Neuron_raw") is not None else 0
                print(f"  skipped (n_neurons={n} < {cfg.min_neurons} or no data)",
                      flush=True)
                continue
            res["mouse_recday"] = args.mouse_recday
            res["session"] = sess
            wake_results[sess] = res
            wake_n_neurons = res["n_neurons"]
            _report(res)
            _save(res, out_dir, f"{args.mouse_recday}__wake_s{sess}")
            print(f"  done in {time.time()-t1:.0f}s", flush=True)

        for sess in sessions:
            if sess in wake_results and "theta" in wake_results[sess]:
                primary_wake = wake_results[sess]
                break
        del data_dic, recday

    # --------------------------------------------------------------- sleep
    if args.state in ("sleep", "both"):
        sb_files = ph.load_pfc_sleep_files(args.mouse_recday, data_folder)
        if args.quick:
            sb_files = sb_files[:1]
        print(f"[sleep] {len(sb_files)} binned_FR_dic file(s) for "
              f"{args.mouse_recday}", flush=True)
        for sb_idx, sleep_raw in sb_files:
            t1 = time.time()
            sleep_raw = np.asarray(sleep_raw)
            print(f"[sleep] sb_{sb_idx}  shape={sleep_raw.shape} ...", flush=True)
            # projection needs the SAME neuron set as wake; skip on mismatch
            if wake_n_neurons is not None and sleep_raw.shape[0] != wake_n_neurons:
                warnings.warn(f"sb_{sb_idx} has {sleep_raw.shape[0]} neurons != "
                              f"wake {wake_n_neurons}; running de novo only")
                wr = None
            else:
                wr = primary_wake
            res = ph.analyse_sleep_session(sleep_raw, cfg, wake_result=wr,
                                           run_null=not args.no_null)
            if res is None:
                print(f"  skipped (n_neurons={sleep_raw.shape[0]} < "
                      f"{cfg.min_neurons} or too short)", flush=True)
                continue
            res["mouse_recday"] = args.mouse_recday
            res["sb_idx"] = sb_idx
            if wr is not None:
                res["wake_session"] = wr.get("session")
            _report(res)
            _save(res, out_dir, f"{args.mouse_recday}__sleep_sb{sb_idx}")
            print(f"  done in {time.time()-t1:.0f}s", flush=True)

    print("ALL DONE", flush=True)


def _report(res):
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
    p.add_argument("--data-folder", default="mFC_data/data",
                   help="mFC_data/data dir (parent of Neuronal_activity, ...)")
    p.add_argument("--out", default="mFC_data/data/ph_outputs")
    p.add_argument("--n-landmarks", type=int, default=1200)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--n-pca", type=int, default=6)
    p.add_argument("--bin-ms", type=int, default=100)
    p.add_argument("--min-neurons", type=int, default=12)
    p.add_argument("--metric", default="euclidean", choices=["euclidean", "geodesic"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-null", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
