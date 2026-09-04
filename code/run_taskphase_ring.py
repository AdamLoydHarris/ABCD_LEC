#!/usr/bin/env python
"""
Task-phase ring sweep (trial-averaged) for LEC — complementary to run_ph_batch.py.

Each session's ring is only 360 points (one per task-phase bin of `Neurons_norm`),
so ripser is instant and no landmarks / subsampling / density trimming are used.
The whole dataset therefore runs locally in minutes — no SLURM needed.

Also (optionally) projects instantaneous wake and sleep onto the ring to decode
task phase over time. The wake decode is the validation of that projection: if
decoded theta does not track true task phase on wake, the sleep decode is not
interpretable and is reported as such rather than trusted.

Usage
-----
  python code/run_taskphase_ring.py                       # all recdays, ring only
  python code/run_taskphase_ring.py --project             # + wake/sleep decode
  python code/run_taskphase_ring.py --recdays ah08_20250613_20250615 --project

Outputs (in --out, default data/ph_outputs/)
  <recday>__ring_s<session>.pkl
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
import glm_analysis_v2 as glm                  # noqa: E402


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _save(obj, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_data_dic(args):
    """LEC: unpickle. Overridden by the PFC sibling."""
    with open(_resolve(args.data_dic), "rb") as f:
        return pickle.load(f)


def wake_speed(session_data):
    XY = session_data.get("XY_raw")
    if XY is None:
        return None
    XY = np.asarray(XY)
    if XY.ndim != 2 or XY.shape[1] != 2:
        return None
    return glm.smooth_and_calculate_scalar_derivatives(XY)[:, 2]


def report(rd, sess, r):
    """top/diam is the primary, theoretically-calibrated criterion (~0.87 = ideal
    circle, ~0 = degenerate). coverage/alignment are necessary but NOT sufficient
    (an open ramp scores ~0.72 coverage)."""
    g = r["geometry"]
    flag = "  <-- RING?" if (g["top_over_diameter"] > 0.5 and r["coverage"] > 0.5) else ""
    print(f"  [{rd} s{sess}] n={r['n_neurons']} trials={r['n_trials']} | "
          f"top/diam={g['top_over_diameter']:.2f}  coverage={r['coverage']:.3f}  "
          f"align={r['alignment']:.3f}  ratio={r['gap_ratio']:.2f}  "
          f"pc12={g['pc12_frac']:.2f}  CV={r.get('cv_alignment', np.nan):.3f}{flag}",
          flush=True)


def run(args, data_dic=None, sleep_loader=None, region="LEC"):
    cfg = ph.PHConfig(n_pca=args.n_pca, n_shuffles=args.n_shuffles,
                      coeff_fields=(2,), maxdim=2, min_trials=args.min_trials,
                      min_neurons=args.min_neurons, random_state=args.seed)
    out_dir = _resolve(args.out)
    if data_dic is None:
        print(f"loading {region} data_dic ...", flush=True)
        t0 = time.time()
        data_dic = load_data_dic(args)
        print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    recdays = args.recdays or list(data_dic.keys())
    n_done = 0
    for rd in recdays:
        if rd not in data_dic:
            print(f"{rd}: not in data_dic, skipping", flush=True)
            continue
        if args.recday_average:
            r = ph.analyse_recday_taskphase_ring(data_dic[rd], cfg)
            if r is None:
                print(f"{rd}: no usable sessions for recday-average", flush=True)
                continue
            r["mouse_recday"], r["region"] = rd, region
            g = r["geometry"]
            flag = "  <-- RING?" if (g["top_over_diameter"] > 0.5
                                     and r["coverage"] > 0.5) else ""
            print(f"  [{rd}] n={r['n_neurons']} sessions={r['n_sessions']} "
                  f"pooled_trials={r['n_trials_total']} | "
                  f"top/diam={g['top_over_diameter']:.2f} coverage={r['coverage']:.3f} "
                  f"align={r['alignment']:.3f} CV={r.get('cv_alignment', np.nan):.3f}{flag}",
                  flush=True)
            _save(r, out_dir, f"{rd}__ringavg")
            n_done += 1
            continue

        sessions, _ = glm.get_sessions_for_glm(data_dic[rd])
        rings = {}
        for sess in sessions:
            sd = data_dic[rd][sess]
            r = ph.analyse_taskphase_ring(sd, cfg, run_null=args.null)
            if r is None:
                continue
            r["mouse_recday"], r["session"], r["region"] = rd, sess, region
            report(rd, sess, r)
            rings[sess] = r
            n_done += 1

        # ---- optional: decode instantaneous wake (+ sleep) with the ring ----
        if args.project and rings:
            primary = rings[max(rings, key=lambda s: rings[s]["n_trials"])]
            for sess, r in rings.items():
                sd = data_dic[rd][sess]
                try:
                    proj = ph.project_onto_taskphase_ring(
                        sd["Neuron_raw"], r["_model"], r["_X"], r["theta"], cfg,
                        speed=wake_speed(sd))
                    # true task phase on the same rebinned timeline, then the
                    # same speed-filter subset the projection kept
                    beh = ph.extract_behaviour(sd, cfg)
                    tp = ph._match_len(np.asarray(beh["task_phase"], float),
                                       proj["n_bins"])[proj["keep_idx"]]
                    r["wake_decode"] = ph.corr_with_shuffle_p(
                        proj["theta_t"], tp, circular_var=True,
                        n_shuffles=200, seed=args.seed)
                    print(f"     wake decode theta~task_phase R="
                          f"{r['wake_decode']['r']:.3f} p={r['wake_decode']['p']:.3f}",
                          flush=True)
                except Exception as e:
                    r["wake_decode_error"] = repr(e)
                    print(f"     wake decode failed: {e!r}", flush=True)

            if sleep_loader is not None:
                for sb_idx, sleep_raw in sleep_loader(rd):
                    try:
                        proj = ph.project_onto_taskphase_ring(
                            sleep_raw, primary["_model"], primary["_X"],
                            primary["theta"], cfg)
                        primary.setdefault("sleep_decode", {})[sb_idx] = dict(
                            replay=ph.replay_metrics(proj["theta_t"], cfg,
                                                     pop_rate=proj["pop_rate"]),
                            theta_t=proj["theta_t"])
                        rp = primary["sleep_decode"][sb_idx]["replay"]
                        print(f"     sleep sb{sb_idx}: coverage={rp['coverage']:.2f} "
                              f"continuity p={rp['p_continuity']:.3f}", flush=True)
                    except Exception as e:
                        print(f"     sleep sb{sb_idx} failed: {e!r}", flush=True)

        for sess, r in rings.items():
            _save(r, out_dir, f"{rd}__ring_s{sess}")
    print(f"ALL DONE — {n_done} session rings written to {out_dir}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recdays", nargs="*", default=None)
    p.add_argument("--data-dic", default="data/processed_data/data_dic_lec.pkl")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="data/ph_outputs")
    p.add_argument("--n-pca", type=int, default=6)
    p.add_argument("--n-shuffles", type=int, default=50)
    p.add_argument("--min-trials", type=int, default=6)
    p.add_argument("--min-neurons", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--null", action="store_true",
                   help="also run the (invalid, reference-only) shift null")
    p.add_argument("--project", action="store_true",
                   help="also decode instantaneous wake (+ sleep) with the ring")
    p.add_argument("--recday-average", dest="recday_average", action="store_true",
                   help="one cross-session pooled ring per recday (session-weighted, "
                        "A-anchored) instead of per-session rings")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    loader = None
    if a.project:
        loader = lambda rd: ph.load_sleep_files(rd, _resolve(a.data_root))  # noqa: E731
    run(a, sleep_loader=loader, region="LEC")
