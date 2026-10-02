"""
Impact of a tracking re-derivation: per-session old-vs-new comparison of the copilot arrays.

Stage 5 of `TRACKING_QC.md`. Compares `XY_raw`, `HD_raw`, `Locs_raw` between two processed trees of
one dataset (e.g. mEC `processed_v2` = geometry fix, v1 cleaning vs `processed_v3` = cleaning v2;
LEC `processed_v2` vs `processed_v3`), session by session, on the bins both have:

- position: |new - old| in cm, p50 / p90 over MOVING bins (old smooth-only speed > 5 cm/s) and
  over all bins; fraction of bins moved by more than 1 cm;
- speed (per bin, from a gaussian sigma-3 smoothed track as GLM V3 does): p90 old / new; path
  length old / new (ratio);
- NaN: fraction of `XY_raw` bins NaN old / new, longest NaN run new (s); `HD_raw` NaN fraction;
- `Locs_raw`: agreement (same code) over bins both track, fraction of new untracked (0) bins;
- salvaged flag (a `SALVAGED_FROM_V1_*.json` beside the arrays).

Outputs `data/processed_data/tracking_impact_{dataset}_{old}_vs_{new}.csv` (versioned `.vN`) in
the dataset's tree and prints per-mouse medians. Reads arrays only; never writes into either tree.

    python tracking_impact.py --dataset mec --old v2 --new v3
    python tracking_impact.py --dataset lec --old v2 --new v3
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recday_registry as rr                                          # noqa: E402

BIN_S = 0.025
MOVING_CM_S = 5.0


def tree(tag: str) -> Path:
    return rr.REPO_ROOT / "data" / ("processed" if tag in ("", "processed", "v1") else f"processed_{tag}")


def sessions(root: Path) -> dict[tuple[str, str, int], Path]:
    """{(mouse, recday, session): dir} from every XY_raw_*.npy in a processed tree."""
    out = {}
    for p in root.glob("*/*/XY_raw_*.npy"):
        m = re.match(r"XY_raw_(.+)_(\d+)\.npy$", p.name)
        if not m:
            continue
        out[(p.parts[-3], m.group(1), int(m.group(2)))] = p.parent
    return out


def load(d: Path, key: str, recday: str, s: int):
    p = d / f"{key}_{recday}_{s}.npy"
    return np.load(p, allow_pickle=True) if p.exists() else None


def speed_glm(xy: np.ndarray) -> np.ndarray:
    """Speed per bin the way GLM V3 computes it (sigma 3, np.gradient), NaN-aware by interpolation."""
    x, y = xy[:, 0].astype(float), xy[:, 1].astype(float)
    nan = np.isnan(x) | np.isnan(y)
    if nan.any():
        idx = np.arange(len(x))
        ok = ~nan
        if ok.sum() < 2:
            return np.full(len(x), np.nan)
        x, y = np.interp(idx, idx[ok], x[ok]), np.interp(idx, idx[ok], y[ok])
    vx, vy = np.gradient(gaussian_filter1d(x, 3)), np.gradient(gaussian_filter1d(y, 3))
    v = np.hypot(vx, vy) / BIN_S
    v[nan] = np.nan
    return v


def path_len(xy: np.ndarray) -> float:
    d = np.diff(xy, axis=0)
    return float(np.nansum(np.hypot(d[:, 0], d[:, 1])))


def longest_nan_run(mask: np.ndarray) -> int:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((e - s).max()) if len(s) else 0


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else np.nan


def compare_session(d_old: Path, d_new: Path, recday: str, s: int) -> dict:
    row = dict(recday=recday, session=s)
    xo, xn = load(d_old, "XY_raw", recday, s), load(d_new, "XY_raw", recday, s)
    ho, hn = load(d_old, "HD_raw", recday, s), load(d_new, "HD_raw", recday, s)
    lo, ln = load(d_old, "Locs_raw", recday, s), load(d_new, "Locs_raw", recday, s)
    row["salvaged"] = any(d_new.glob(f"SALVAGED_FROM_V1_{recday}_{s}.json"))
    if xo is None or xn is None:
        row["status"] = "missing_" + ("old" if xo is None else "new")
        return row
    n = min(len(xo), len(xn))
    row.update(n_bins_old=len(xo), n_bins_new=len(xn))
    xo, xn = xo[:n].astype(float), xn[:n].astype(float)
    nan_o, nan_n = np.isnan(xo).any(1), np.isnan(xn).any(1)
    both = ~nan_o & ~nan_n
    d = np.hypot(*(xn - xo).T)
    v_old, v_new = speed_glm(xo), speed_glm(xn)
    moving = np.nan_to_num(v_old) > MOVING_CM_S
    row.update(xy_nan_old=float(nan_o.mean()), xy_nan_new=float(nan_n.mean()),
               xy_nan_new_longest_s=round(longest_nan_run(nan_n) * BIN_S, 2),
               xy_diff_p50_moving=pct(d[both & moving], 50), xy_diff_p90_moving=pct(d[both & moving], 90),
               xy_diff_p50_all=pct(d[both], 50), xy_diff_p90_all=pct(d[both], 90),
               xy_moved_gt1cm_frac=float(np.mean(d[both] > 1.0)) if both.any() else np.nan,
               speed_p90_old=pct(v_old, 90), speed_p90_new=pct(v_new, 90),
               path_old_m=round(path_len(xo) / 100, 1), path_new_m=round(path_len(xn) / 100, 1),
               path_ratio=round(path_len(xn) / path_len(xo), 4) if path_len(xo) > 0 else np.nan,
               moving_frac_old=float(moving.mean()))
    if ho is not None and hn is not None:
        m = min(len(ho), len(hn))
        h_o, h_n = np.asarray(ho[:m], float)[:, 0], np.asarray(hn[:m], float)[:, 0]
        ok = np.isfinite(h_o) & np.isfinite(h_n)
        dh = np.abs((h_n[ok] - h_o[ok] + 180) % 360 - 180)
        row.update(hd_nan_old=float(np.isnan(h_o).mean()), hd_nan_new=float(np.isnan(h_n).mean()),
                   hd_diff_p50=pct(dh, 50), hd_diff_p90=pct(dh, 90))
    if lo is not None and ln is not None:
        m = min(len(lo), len(ln))
        a, b = lo[:m].astype(str), ln[:m].astype(str)
        # untracked bins are 'None' in the ROI files, NaN objects once pandas has read them
        # ('None' is a default NA token) and 0 after locs_to_int; count all three
        untr_a = np.isin(a, ("None", "0", "nan"))
        untr_b = np.isin(b, ("None", "0", "nan"))
        tracked = ~untr_a & ~untr_b
        row.update(locs_agree=float(np.mean(a[tracked] == b[tracked])) if tracked.any() else np.nan,
                   locs_untracked_old=float(untr_a.mean()), locs_untracked_new=float(untr_b.mean()))
    elif ln is not None and lo is None:
        row["locs_agree"] = np.nan
    row["status"] = "ok"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--old", required=True, help="processed tree tag of the reference (e.g. v2)")
    ap.add_argument("--new", required=True, help="processed tree tag of the re-derivation (e.g. v3)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    old, new = sessions(tree(args.old)), sessions(tree(args.new))
    keys = sorted(set(old) | set(new))
    print(f"{args.dataset}: {len(old)} sessions in {tree(args.old).name}, {len(new)} in {tree(args.new).name}, {len(keys)} in the union")
    rows = []
    for mouse, recday, s in keys:
        if (mouse, recday, s) not in old or (mouse, recday, s) not in new:
            rows.append(dict(mouse=mouse, recday=recday, session=s, status="only_" + ("old" if (mouse, recday, s) in old else "new")))
            continue
        r = compare_session(old[(mouse, recday, s)], new[(mouse, recday, s)], recday, s)
        r["mouse"] = mouse
        rows.append(r)
    t = pd.DataFrame(rows)
    ok = t[t.status == "ok"]
    pd.set_option("display.width", 250)
    print(t.status.value_counts().to_string())
    cols = [c for c in ("xy_diff_p50_moving", "xy_diff_p90_moving", "xy_moved_gt1cm_frac", "speed_p90_old", "speed_p90_new",
                        "path_ratio", "xy_nan_old", "xy_nan_new", "xy_nan_new_longest_s", "hd_nan_old", "hd_nan_new",
                        "hd_diff_p90", "locs_agree", "locs_untracked_old", "locs_untracked_new") if c in ok]
    print("\nper mouse, medians over sessions:")
    print(ok.groupby("mouse")[cols].median().round(3).T.to_string())
    if args.write and len(t):
        out = rr.REPO_ROOT / "data" / "processed_data" / f"tracking_impact_{args.dataset}_{args.old}_vs_{args.new}.csv"
        n = 0
        while (p := out.with_name(out.stem + (f".v{n}" if n else "") + out.suffix)).exists():
            n += 1
        t.to_csv(p, index=False)
        print(f"-> {p}")


if __name__ == "__main__":
    main()
