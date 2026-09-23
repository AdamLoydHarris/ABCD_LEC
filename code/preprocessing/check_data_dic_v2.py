"""
Load-check of a built data_dic through `glm_analysis_v3.load_data_dic` and a per-session
summary of what changed against the previous version (LEC v1 -> v2). Run under sbatch (two
3.8 GB pickles).

Usage:  python check_data_dic_v2.py --dataset lec --version v2 --against data_dic_lec.pkl
        python check_data_dic_v2.py --dataset mec
Writes data/processed_data/data_dic_diff_{dataset}[_{version}].csv (versioned) when --against.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recday_registry as rr                                          # noqa: E402
import glm_analysis_v3 as g                                           # noqa: E402


def pop_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 40) -> tuple[int, float]:
    """Lag (bins, b relative to a) maximising the population-rate correlation."""
    ra, rb = a.sum(axis=0).astype(float), b.sum(axis=0).astype(float)
    n = min(len(ra), len(rb)) - 2 * max_lag          # every lagged slice must fit both arrays
    if n < 100:
        return 0, float("nan")
    best = (0, -2.0)
    for lag in range(-max_lag, max_lag + 1):
        x = ra[max_lag:max_lag + n]
        y = rb[max_lag + lag:max_lag + lag + n]
        c = np.corrcoef(x, y)[0, 1] if x.std() and y.std() else -2
        if c > best[1]:
            best = (lag, c)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=rr.DATASET)
    ap.add_argument("--version", default=None)
    ap.add_argument("--against", default=None)
    ap.add_argument("--recday-kind", default=None, help="merged | single (required for the mEC pickle, which holds both)")
    args = ap.parse_args()
    kw = dict(dataset=args.dataset)
    if args.version:
        kw["version"] = args.version
    if args.recday_kind:
        kw["recday_kind"] = args.recday_kind
    out = g.load_data_dic(**kw)
    d = out[0] if isinstance(out, tuple) else out
    n_sess = sum(len(v) for v in d.values())
    types, keys, with_xy, nan_xy = {}, set(), 0, []
    for rd, ss in d.items():
        for s, e in ss.items():
            types[e.get("session_type", "n/a")] = types.get(e.get("session_type", "n/a"), 0) + 1
            keys |= set(e.keys())
            if "XY_raw" in e:
                with_xy += 1
                nan_xy.append(float(np.isnan(e["XY_raw"]).any(axis=1).mean()))
    print(f"{args.dataset} {args.version or ''}: {len(d)} recdays, {n_sess} sessions, {with_xy} with XY_raw "
          f"(NaN fraction median {np.median(nan_xy):.3f}, max {np.max(nan_xy):.3f}); session types {types}")
    print("keys seen:", sorted(keys))
    if not args.against:
        return
    prev_path = rr.REPO_ROOT / "data" / "processed_data" / args.against
    with open(prev_path, "rb") as f:
        prev = pickle.load(f)
    if isinstance(prev, tuple):
        prev = prev[0]
    rows = []
    for rd in sorted(set(d) | set(prev)):
        for s in sorted(set(d.get(rd, {})) | set(prev.get(rd, {}))):
            a, b = prev.get(rd, {}).get(s), d.get(rd, {}).get(s)
            row = dict(recday=rd, session=s, in_v1=a is not None, in_v2=b is not None)
            if a is not None and b is not None:
                row["type"] = b.get("session_type")
                na, nb = a.get("Neuron_raw"), b.get("Neuron_raw")
                if na is not None and nb is not None:
                    row["neuron_bins_v1"], row["neuron_bins_v2"] = na.shape[1], nb.shape[1]
                    row["neuron_identical"] = bool(na.shape == nb.shape and np.array_equal(na, nb))
                    lag, c = pop_lag(na, nb)
                    row["neuron_lag_bins_v2_minus_v1"], row["neuron_lag_r"] = lag, round(c, 3)
                    row["spikes_v1"], row["spikes_v2"] = int(na.sum()), int(nb.sum())
                xa, xb = a.get("XY_raw"), b.get("XY_raw")
                row["xy_v1"], row["xy_v2"] = xa is not None, xb is not None
                if xa is not None and xb is not None:
                    n = min(len(xa), len(xb))
                    dd = np.linalg.norm(xa[:n] - xb[:n], axis=1)
                    row["xy_len_v1"], row["xy_len_v2"] = len(xa), len(xb)
                    row["xy_shift_cm_median"] = round(float(np.nanmedian(dd)), 2)
                    row["xy_shift_cm_p95"] = round(float(np.nanpercentile(dd, 95)), 2)
                    row["xy_shift_cm_first_min"] = round(float(np.nanmedian(dd[:2400])), 2)
                    row["xy_shift_cm_last_min"] = round(float(np.nanmedian(dd[n - 2400:n])), 2)
                ta, tb = a.get("Trial_times"), b.get("Trial_times")
                row["trialtimes_identical"] = bool(ta is not None and tb is not None and np.array_equal(np.asarray(ta), np.asarray(tb))) if (ta is not None or tb is not None) else None
                row["new_keys"] = ",".join(sorted(set(b) - set(a)))
            rows.append(row)
    t = pd.DataFrame(rows)
    outp = rr.REPO_ROOT / "data" / "processed_data" / f"data_dic_diff_{args.dataset}{('_' + args.version) if args.version else ''}.csv"
    n = 0
    while (p := outp.with_name(outp.stem + (f".v{n}" if n else "") + ".csv")).exists():
        n += 1
    t.to_csv(p, index=False)
    print(f"diff table -> {p}")
    both = t[t.in_v1 & t.in_v2]
    print(f"sessions: v1 {int(t.in_v1.sum())}, v2 {int(t.in_v2.sum())}, only v1 {int((t.in_v1 & ~t.in_v2).sum())}, only v2 {int((~t.in_v1 & t.in_v2).sum())}")
    print("only in v1:", t[t.in_v1 & ~t.in_v2][["recday", "session"]].to_records(index=False).tolist())
    print("only in v2:", t[~t.in_v1 & t.in_v2][["recday", "session"]].to_records(index=False).tolist())
    nn = both[both.neuron_identical.notna()]
    print(f"Neuron_raw identical in {int(nn.neuron_identical.sum())}/{len(nn)} sessions; lag v2-v1 per mouse (median, min, max bins):")
    nn = nn.assign(mouse=nn.recday.str[:4])
    print(nn.groupby("mouse").neuron_lag_bins_v2_minus_v1.agg(["median", "min", "max", "count"]).to_string())
    print(f"sessions gaining XY_raw in v2: {int((both.xy_v2 & ~both.xy_v1).sum())}; losing: {int((both.xy_v1 & ~both.xy_v2).sum())} {both[both.xy_v1 & ~both.xy_v2][['recday','session']].to_records(index=False).tolist()}")
    xx = both[both.xy_v1 & both.xy_v2].assign(mouse=both.recday.str[:4])
    print("tracking shift v2 vs v1 (cm) per mouse: median over sessions of the per-session median, and of the last-minute median:")
    print(xx.groupby("mouse")[["xy_shift_cm_median", "xy_shift_cm_first_min", "xy_shift_cm_last_min", "xy_shift_cm_p95"]].median().round(2).to_string())
    print("trialtimes identical:", int(both.trialtimes_identical.sum()), "of", int(both.trialtimes_identical.notna().sum()))


if __name__ == "__main__":
    main()
