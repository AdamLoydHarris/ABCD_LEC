"""Re-score the QC gate over a sub-interval of a sorted block, without re-running Kilosort.

Written 2026-09-22 to answer "would we get more units if the sort excluded the sleep sessions?"
and reused to score the 6-day July blocks day by day.

The four gate criteria that depend on *which time you score* -- `num_spikes`, `firing_rate`,
`presence_ratio`, `isi_violations_ratio` -- are recomputed from the sorting analyzer's own
spike train over any chosen subset of the concatenated recordings. The two that are waveform
properties (`amplitude_cutoff`, `amplitude_median`) and `sd_ratio` are taken from the stored
`quality_metrics.htsv` unchanged, because they do not depend on the interval.

Definitions are spikeinterface 0.101.2's, reproduced exactly:
  presence_ratio = (60 s bins containing >=1 spike) / (total 60 s bins)
  isi_violations_ratio = n_viol * T / (2 * N^2 * isi_threshold_s),  isi_threshold_s = 0.0015
ISIs must be compared in SECONDS, not samples: `0.0015 * 30000` is 45.000000000000004 in
float, so a sample-space test counts spikes exactly 45 samples apart and the violation count
comes out wrong (this cost one debugging pass).

Every run prints a CONTROL first: scoring the full concat must reproduce the block's stored
`QC_single_units.npy` count exactly. It has held on 12/12 mEC and 5/5 LEC blocks. Do not read
any sub-interval number from a run whose control failed.

    python code/preprocessing/rescore_qc_epochs.py --recdays all --split task_sleep
    TSA_DATASET=mec python code/preprocessing/rescore_qc_epochs.py --recdays all --split task_sleep
    python code/preprocessing/rescore_qc_epochs.py --recdays ah08_20250714_..._20250719 --split per_day

I/O: reads the sorted blocks; writes one CSV of per-block counts. Nothing is overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import recday_registry as rr
import extract_neuron_raw as en

FS = 30000.0
ISI_THRESHOLD_S = 0.0015
PRESENCE_BIN_S = 60.0


def _load_spikes(kd: Path):
    """`(sample_index sorted within unit, unit boundaries, unit_ids)` for one block."""
    uids = np.array(json.load(
        open(kd / "sorting_analyzer/sorting/numpysorting_info.json"))["unit_ids"])
    sp = np.load(kd / "sorting_analyzer/sorting/spikes.npy", mmap_mode="r")
    smp = np.asarray(sp["sample_index"]).astype(np.int64)
    uix = np.asarray(sp["unit_index"])
    order = np.argsort(uix, kind="stable")          # stable => sample order kept within unit
    smp = smp[order]
    bounds = np.searchsorted(uix[order], np.arange(len(uids) + 1))
    return smp, bounds, uids


def score_interval(smp, bounds, uids, seg, starts, n_samples, rows):
    """Recompute the interval-dependent metrics over the concat rows in `rows`.

    Time is compressed so the kept recordings are contiguous; ISIs are never counted across
    a recording boundary, which is the one place this differs from scoring the full concat.
    """
    rows = list(rows)
    total_s = n_samples[rows].sum() / FS
    n_bins = int(np.ceil(total_s / PRESENCE_BIN_S))

    offset, run = {}, 0
    for i in rows:
        offset[i] = run - starts[i]
        run += n_samples[i]
    shift = np.zeros(len(smp), np.int64)
    for i in rows:
        shift[seg == i] = offset[i]
    keep = np.isin(seg, rows)

    out = np.zeros((len(uids), 4))
    for k in range(len(uids)):
        a, b = bounds[k], bounds[k + 1]
        m = keep[a:b]
        t = ((smp[a:b] + shift[a:b])[m]) / FS
        s = seg[a:b][m]
        n = len(t)
        if n > 1:
            d = np.diff(t)
            n_viol = int(((d < ISI_THRESHOLD_S) & (s[1:] == s[:-1])).sum())
            isi = n_viol * total_s / (2.0 * n * n * ISI_THRESHOLD_S)
        else:
            isi = np.nan
        pr = (len(np.unique(np.minimum((t / PRESENCE_BIN_S).astype(int), n_bins - 1))) / n_bins
              if n else 0.0)
        out[k] = [n, isi, pr, n / total_s]
    return pd.DataFrame(out, index=uids,
                        columns=["n_spikes", "isi", "presence_ratio", "firing_rate"]), total_s


def gate(scored, qm, uids):
    """The shipped six-criterion gate, interval metrics from `scored` and the rest stored."""
    g = rr.QC_GATE
    amp = qm["amplitude_median"].abs().reindex(uids)
    return int(((scored["isi"] < g["isi_violations_ratio_lt"])
                & (qm["amplitude_cutoff"].reindex(uids) < g["amplitude_cutoff_lt"])
                & (scored["firing_rate"] > g["firing_rate_gt"])
                & (scored["presence_ratio"] > g["presence_ratio_gt"])
                & (amp > g["amplitude_median_abs_gt"])
                & (qm["sd_ratio"].reindex(uids) < g["sd_ratio_lt"])).sum())


def run_block(recday: str, split: str) -> dict:
    kd = rr.kilosort_dir(recday)
    qm = pd.read_csv(kd / "quality_metrics.htsv", sep="\t", index_col=0)
    smp, bounds, uids = _load_spikes(kd)

    table = en.concat_table(recday)
    starts = table["start"].to_numpy()
    n_samples = table["n_samples"].to_numpy()
    seg = np.searchsorted(starts, smp, side="right") - 1
    all_rows = list(range(len(table)))

    shipped = int(rr.qc_unit_count(recday))
    full, t_full = score_interval(smp, bounds, uids, seg, starts, n_samples, all_rows)
    rec = dict(recday=recday, n_clusters=len(uids), shipped=shipped,
               control_full=gate(full, qm, uids), hours_total=t_full / 3600)
    rec["control_ok"] = rec["control_full"] == shipped

    if split == "task_sleep":
        task, sleep = en.session_map(recday)
        sc, t = score_interval(smp, bounds, uids, seg, starts, n_samples, task)
        rec.update(gate_task_only=gate(sc, qm, uids), hours_task=t / 3600,
                   isi_med_task=float(np.nanmedian(sc["isi"])))
        sc, t = score_interval(smp, bounds, uids, seg, starts, n_samples, sleep)
        rec.update(hours_sleep=t / 3600, isi_med_sleep=float(np.nanmedian(sc["isi"])))
        rec["isi_med_full"] = float(np.nanmedian(full["isi"]))
    elif split == "per_day":
        for date in sorted(set(table["date"])):
            rows = [i for i in all_rows if table["date"].iloc[i] == date]
            sc, t = score_interval(smp, bounds, uids, seg, starts, n_samples, rows)
            rec[f"gate_{date}"] = gate(sc, qm, uids)
            rec[f"pr_pass_{date}"] = int((sc["presence_ratio"] > 0.9).sum())
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recdays", default="all",
                   help="'all', or a comma-separated list of recday names")
    p.add_argument("--kind", default=None, choices=[None, "merged", "single"],
                   help="restrict 'all' to one recday kind")
    p.add_argument("--split", default="task_sleep", choices=["task_sleep", "per_day", "none"])
    p.add_argument("--out", default=None, help="CSV to write; refuses to overwrite")
    args = p.parse_args()

    recdays = (rr.all_recdays(args.kind) if args.recdays == "all"
               else [r.strip() for r in args.recdays.split(",")])
    rows = []
    for recday in recdays:
        rows.append(run_block(recday, args.split))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)

    bad = df[~df["control_ok"]]
    print(f"\nCONTROL: {int(df['control_ok'].sum())}/{len(df)} blocks reproduce their stored "
          f"QC_single_units count exactly")
    if len(bad):
        print("  FAILED on:", list(bad["recday"]))
        sys.exit("control failed; sub-interval numbers are not trustworthy")
    print(f"shipped {df.shipped.sum()}  control {df.control_full.sum()}"
          + (f"  task-only {df.gate_task_only.sum()}" if "gate_task_only" in df else ""))

    if args.out:
        out = Path(args.out)
        if out.exists():
            sys.exit(f"{out} exists; outputs are never overwritten")
        df.to_csv(out, index=False)
        print(f"wrote {out}")
