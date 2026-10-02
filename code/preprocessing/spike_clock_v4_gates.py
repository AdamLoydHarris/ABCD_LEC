"""
Gates for the v4 spike clock: v4 may differ from v2 only by the counter origin
(docs/handoff/SPIKE_CLOCK_V4.md).

| # | gate | what a failure would mean |
|---|---|---|
| 1 | every sorted block: binary samples (file size / (n_channels_dat x 2)) == sum(dat_size) / 768 | spike sample minus concat start is not the `.dat` index, so no origin fix can work |
| 2 | every recording: v4 TTL train == v2 train + (sn0 - origin), exactly | the v4 extractor changed more than the origin |
| 3 | every task recording: v4 slope == v2 slope (1e-12), v4 b - v2 b == sn0 - origin (< 1e-3 samples), same n_matched; v4 rsync train == v2 + (sn0 - origin) to 1 sample | the pairing moved, so v4 is not v2 with the origin fixed |
| 4 | every array where origin == sn0, and every `_sb_` array: v4 byte-identical to v2 | the v4 run changed sessions the bug never touched |
| 5 | every restart array: v2's population rate trails v4's by (sn0 - origin) / (25 a) bins, ± 1, with r >= 0.8, searched over every lag from 0 to past the expected one | the spikes moved by something other than the origin |

Reads only. Writes `spike_clock_v4_gates_{dataset}.csv` (versioned); exit 1 on any failure.

    python spike_clock_v4_gates.py --dataset lec
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
import extract_neuron_raw as enr                                      # noqa: E402

PROCESSED = rr.REPO_ROOT / "data" / "processed_data"
V2_RSYNC, V4_RSYNC = PROCESSED / "ephys_rsync", PROCESSED / "ephys_rsync_v4"
V2_NEURAL, V4_NEURAL = PROCESSED / "neuron_raw_v2", PROCESSED / "neuron_raw_v4"


def latest_qc(stem: str) -> pd.DataFrame:
    """Every `<stem>*.csv` the extractor wrote, latest row per recording."""
    cands = sorted(PROCESSED.glob(f"{stem}*.csv"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"no {stem}*.csv under {PROCESSED}")
    table = pd.concat([pd.read_csv(p) for p in cands], ignore_index=True)
    return table.drop_duplicates(subset=["mouse", "recording"], keep="last")


def gate_binary(recday: str) -> dict:
    """1. sorted-binary length against the concat manifest."""
    ks = rr.kilosort_dir(recday)
    row = dict(gate="1_binary", recday=recday)
    params = ks / "sorter_output" / "params.py"
    if not params.exists():
        return {**row, "status": "no sort", "ok": None}
    text = params.read_text()
    nch = int(re.search(r"n_channels_dat\s*=\s*(\d+)", text).group(1))
    dat = re.search(r"dat_path\s*=\s*r?['\"]([^'\"]+)['\"]", text).group(1)
    samples, rem = divmod(os.path.getsize(dat), nch * 2)
    concat = pd.read_csv(ks.parent / "recording_sessions_in_concat.csv")
    want = int((concat["dat_size"] // 768).sum())
    return {**row, "status": "checked", "binary_samples": samples, "concat_samples": want,
            "ok": bool(rem == 0 and samples == want)}


def gate_recording(r: pd.Series) -> dict:
    """2-3. one recording's v4 trains and fit against v2's."""
    m, rec = r["mouse"], r["recording"]
    row = dict(gate="2_3_rsync", mouse=m, recording=rec, kind=r["kind"])
    if pd.isna(r.get("origin")) or pd.isna(r.get("ap_first_sample")):   # e.g. no pyControl file
        return {**row, "status": f"no origin ({r.get('error', '')})", "ok": None}
    shift = int(r["ap_first_sample"]) - int(r["origin"])
    row["shift_samples"] = shift
    t2, t4 = V2_RSYNC / m / f"ttl_rising_{rec}.npy", V4_RSYNC / m / f"ttl_rising_{rec}.npy"
    if not t2.exists():
        return {**row, "status": "no v2", "ok": None}
    a2, a4 = np.load(t2), np.load(t4)
    row["ttl_ok"] = bool(len(a2) == len(a4) and np.array_equal(a4 - a2, np.full(len(a2), shift)))
    ok = row["ttl_ok"]
    if r["kind"] == "task" and (V4_RSYNC / m / f"rsync_fit_{rec}.json").exists():
        if not (V2_RSYNC / m / f"rsync_fit_{rec}.json").exists():
            return {**row, "status": "no v2 fit", "ok": None}
        f2 = json.loads((V2_RSYNC / m / f"rsync_fit_{rec}.json").read_text())
        f4 = json.loads((V4_RSYNC / m / f"rsync_fit_{rec}.json").read_text())
        r2 = np.load(V2_RSYNC / m / f"rsync_timestamps_{rec}.npy")
        r4 = np.load(V4_RSYNC / m / f"rsync_timestamps_{rec}.npy")
        d = (r4 - r2 - shift) if len(r2) == len(r4) else np.array([np.inf])
        row.update(slope_ok=bool(np.isclose(f4["a_samples_per_ms"], f2["a_samples_per_ms"], rtol=1e-12, atol=0)),
                   b_err=float(f4["b_samples"] - f2["b_samples"] - shift),
                   matched_ok=f4["n_matched"] == f2["n_matched"],
                   rsync_max_err=float(np.abs(d).max()) if len(d) else 0.0,
                   rsync_off_by_one=int((d != 0).sum()))
        ok = ok and row["slope_ok"] and abs(row["b_err"]) < 1e-3 and row["matched_ok"] and row["rsync_max_err"] <= 1
    return {**row, "status": "checked", "ok": bool(ok)}


def pop_lag(early: np.ndarray, late: np.ndarray, center: float, pad: int = 200) -> tuple[int, float]:
    """(lag L in bins maximising corr(early_pop[t], late_pop[t + L]), r there).

    Searched over every lag from min(0, center) - pad to max(0, center) + pad. A window around
    `center` alone let v2-against-itself pass 1 of 7 restart arrays on a local maximum.
    Rates are smoothed (sigma 2 bins) so a fractional shift does not halve r, and r is the
    Pearson r over the overlap at that lag: the global normalisation put LEC's 1166-bin shift at
    0.805 although the lag was exact.
    """
    a = gaussian_filter1d(early.sum(axis=0).astype(float), 2)
    b = gaussian_filter1d(late.sum(axis=0).astype(float), 2)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    za, zb = (a - a.mean()) / (a.std() or 1), (b - b.mean()) / (b.std() or 1)
    c = fftconvolve(zb, za[::-1], "full") / n
    lags = np.arange(-(n - 1), n)
    window = (lags >= min(0, center) - pad) & (lags <= max(0, center) + pad)
    lag = int(lags[window][int(np.argmax(c[window]))])
    x, y = a[max(0, -lag):n - max(0, lag)], b[max(0, lag):n - max(0, -lag)]
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() and y.std() else np.nan
    return lag, r


def gate_arrays(recday: str, qc: pd.DataFrame) -> list[dict]:
    """4-5. every array of one recday, v4 against v2."""
    mouse, _ = rr.recday_dates(recday)
    table = enr.concat_table(recday)
    wake, sleep = enr.session_map(recday)
    rows = []
    for names, is_sleep in (([(i, c) for i, c in enumerate(wake)], False), ([(i, c) for i, c in enumerate(sleep)], True)):
        for idx, concat_row in names:
            name = f"Neuron_raw_{recday}_sb_{idx}" if is_sleep else f"Neuron_raw_{recday}_{idx}"
            date, time = table["date"].iloc[concat_row], table["time"].iloc[concat_row]
            tag = f"{date[:4]}-{date[4:6]}-{date[6:]}_{time}"
            row = dict(gate="4_5_arrays", recday=recday, array=name, recording=tag)
            p2, p4 = V2_NEURAL / f"{name}.npy", V4_NEURAL / f"{name}.npy"
            if not (p2.exists() and p4.exists()):
                rows.append({**row, "status": f"missing v2={p2.exists()} v4={p4.exists()}", "ok": None})
                continue
            v2, v4 = np.load(p2), np.load(p4)
            hit = qc[(qc["mouse"] == mouse) & (qc["recording"] == tag)]
            if not is_sleep and len(hit) and hit[["ap_first_sample", "origin"]].isna().any(axis=None):
                rows.append({**row, "status": "no origin", "ok": None})
                continue
            shift = 0 if is_sleep or not len(hit) else int(hit["ap_first_sample"].iloc[0]) - int(hit["origin"].iloc[0])
            row.update(shift_samples=shift, spikes_v2=int(v2.sum()), spikes_v4=int(v4.sum()),
                       bins_v2=v2.shape[1], bins_v4=v4.shape[1])
            if shift == 0:
                row.update(status="identical expected", ok=bool(v2.shape == v4.shape and np.array_equal(v2, v4)))
            else:
                a = float(json.loads((V4_RSYNC / mouse / f"rsync_fit_{tag}.json").read_text())["a_samples_per_ms"])
                want = shift / (25 * a)
                lag, r = pop_lag(v4, v2, want)
                row.update(status="shift expected", lag_bins=lag, want_bins=round(want, 2), lag_r=round(r, 3),
                           ok=bool(abs(lag - want) <= 1 and r >= 0.8))
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--recdays", nargs="*", default=None)
    args = ap.parse_args()
    recdays = args.recdays or rr.all_recdays()
    qc = latest_qc("ephys_rsync_v4_qc")

    rows = [gate_binary(rd) for rd in recdays]
    seen = {(m, r) for m in {rr.recday_dates(rd)[0] for rd in recdays}
            for r in qc.loc[qc["mouse"] == m, "recording"]}
    rows += [gate_recording(r) for _, r in qc.iterrows() if (r["mouse"], r["recording"]) in seen]
    for rd in recdays:
        try:
            rows += gate_arrays(rd, qc)
        except (KeyError, FileNotFoundError) as exc:
            rows.append(dict(gate="4_5_arrays", recday=rd, status=f"skipped ({exc})", ok=None))
    report = pd.DataFrame(rows)

    pd.set_option("display.width", 250)
    for gate, g in report.groupby("gate"):
        seen_ok = g["ok"].dropna()
        print(f"{gate}: {int(seen_ok.astype(bool).sum())}/{len(seen_ok)} pass; status {g['status'].value_counts().to_dict()}")
    arr = report[(report["gate"] == "4_5_arrays") & (report["status"] == "shift expected")]
    if len(arr):
        print(f"restart arrays: lag - expected in bins, min {(arr['lag_bins'] - arr['want_bins']).min():+.2f} "
              f"max {(arr['lag_bins'] - arr['want_bins']).max():+.2f}; shift {arr['shift_samples'].min() / 30000:.2f}"
              f"-{arr['shift_samples'].max() / 30000:.2f} s")
    rec = report[report["gate"] == "2_3_rsync"]
    if "rsync_off_by_one" in rec:
        print(f"rsync trains: {int(rec['rsync_off_by_one'].fillna(0).sum())} samples off by one in total; "
              f"max |b err| {rec['b_err'].abs().max():.2e} samples")
    out = PROCESSED / f"spike_clock_v4_gates_{rr.DATASET}.csv"
    n = 0
    while (p := out.with_name(out.stem + (f".v{n}" if n else "") + ".csv")).exists():
        n += 1
    report.to_csv(p, index=False)
    print(f"report -> {p}")
    bad = report[report["ok"] == False]                                 # noqa: E712
    if len(bad):
        print(f"\n{len(bad)} FAIL:")
        print(bad.to_string(index=False))
        sys.exit(1)
    print("all v4 gates pass")


if __name__ == "__main__":
    main()
