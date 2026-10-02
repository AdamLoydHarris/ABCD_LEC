"""
Rsync pulse trains from the OpenEphys TTL streams, for `extract_neuron_raw.first_rsync_sample`.

LEC's `ephys_rsync_mingyutest/` caches were written by an untracked script. The mEC tree has
none, so this derives them from the raw TTL events and, unlike the caches, records how.

Per recording in a block's `recording_sessions_in_concat.csv` (using that row's own
`ttl_path` and `sample_numbers_path`, i.e. the mouse's own probe stream):

1. Rising edges of the sync line (`states > 0`), converted from OpenEphys sample numbers to
   `continuous.dat` indices by subtracting the counter origin (`counter_origin`) -- the
   timebase Kilosort spike times use once `extract_neuron_raw` subtracts the recording's
   concat offset. Saved as `ttl_rising_{date}_{time}.npy`.
2. If the recording is a task session (a metadata row names its `Ephys` stamp), the TTL
   train is matched to that session's pyControl `rsync` events and every pyControl rsync
   time is mapped into ephys samples. That array is saved as
   `rsync_timestamps_{date}_{time}.npy`, so element 0 is the ephys sample of pyControl's
   first rsync -- exactly the pairing `session_window` assumes. Unmatched leading pulses
   (ephys started after pyControl) are extrapolated with the fitted slope and flagged.

   The match is a pairwise-difference histogram, not `Rsync_aligner`: the mEC PXI sync line
   carries BOTH paired boxes' rsync trains (bp01+bp02, bp03+bp04), so an ABCD recording has
   ~2x the edges pyControl sent (484 vs 247 on 2024-03-24_10-52-32) and interval matching
   fails, while the open-field recordings (one box) match 360/360. Every pyControl pulse
   is at a fixed offset from its TTL edge, so the mode of all TTL-minus-pyControl
   differences is that offset (peak ~n_pycontrol vs a chance floor of a few), the other
   box's pulses are just background, and a linear fit on the matched pairs gives the
   drift. The leftover TTL edges are reported; on the paired mouse they are its partner's.
3. Sleep-box recordings get the raw relative train under the `rsync_timestamps_` name too;
   `session_window` never reads it for sleep (origin 0), it is there for completeness.

The origin (`--origin`; docs/handoff/SPIKE_CLOCK_V4.md). OpenEphys 0.6.6 writes the first
716-2932 samples of some recordings with a stale counter and then restarts the count near 0.
`sample_numbers[0]` and `sync_messages.txt` report the stale value; the TTL events carry the
restarted count. That is 95 of 758 LEC recordings and every LEC task recording whose first
sample is > 0. On bp01 2024-03-24_10-52-32 the count runs 603,172..604,163, then 480 onwards,
which is why its TTL edges looked ~136k samples early.
- `counter` (default): origin = `sn[j] - j` over the final contiguous run. Right in all 758.
- `first_sample`: origin = `sn[0]`, the v2 rule that wrote `ephys_rsync/`. It puts spikes
  0.94-29.2 s late in the 42 LEC restart task recordings; kept only to reproduce v2.

QC per recording -> `<out>_qc.csv`: n_ttl, n_pycontrol, n_matched, slope, residual, origin.
Nothing is overwritten.

Usage::

    python extract_rsync_from_ttl.py --dataset mec [--recdays ...] [--write]
    python extract_rsync_from_ttl.py --origin first_sample --compare-to ephys_rsync   # v2 control
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402

PROCESSED = rr.REPO_ROOT / "data" / "processed_data"
DEFAULT_OUT = "ephys_rsync_v4"   # v2's `ephys_rsync/` holds the first_sample-origin trains
ORIGIN_MODES = ("counter", "first_sample")
MAX_STALE_SAMPLES = 30_000       # restarts sit at 716-2932 samples in all 95 LEC cases; a jump past
                                 # 1 s would leave two plausible counts for the TTL, so it raises
SCAN_CHUNK = 10_000_000          # sample_numbers are scanned in 80 MB pieces (a recording is ~290 MB)
SAMPLE_RATE_HZ = 30_000
SAMPLES_PER_MS = SAMPLE_RATE_HZ // 1000
COARSE_BIN_MS = 20.0     # histogram bin for the offset search: wider than the ~12 ms drift over a
                         # session, so the true offset lands in one bin (peak ~n_pycontrol vs a
                         # coincidence floor of ~2); at 2 ms the peak smeared to 58 vs 57
COARSE_TOL_MS = 30.0     # first pairing tolerance around the mode's 3-bin neighbourhood centre
FINE_TOL_MS = 2.0        # pairing tolerance after the linear fit; pulses are ms-precise on both clocks
MIN_PEAK_RATIO = 5.0     # mode must beat the second-highest bin by this factor to count as a match


def match_trains(ttl_ms: np.ndarray, pc_ms: np.ndarray) -> dict:
    """Pair this mouse's pyControl rsync pulses with edges of a (possibly shared) TTL train.

    Returns slope/offset of `ttl_ms = slope * pc_ms + offset`, the matched TTL indices, and
    the counts. Raises ValueError when no offset explains more than a handful of pulses.
    """
    if len(ttl_ms) < 5 or len(pc_ms) < 5:
        raise ValueError(f"too few pulses (ttl {len(ttl_ms)}, pycontrol {len(pc_ms)})")
    diffs = (ttl_ms[None, :] - pc_ms[:, None]).ravel()
    edges = np.arange(diffs.min(), diffs.max() + COARSE_BIN_MS, COARSE_BIN_MS)
    hist, _ = np.histogram(diffs, edges)
    # The offset drifts ~1e-5 across a session (12-18 ms), so it can straddle a bin edge and
    # split the mode over two neighbours (measured: 154 + 102 = 256 = n_pycontrol). Judge
    # dominance on the mode's 3-bin neighbourhood against the best bin outside it.
    k = int(hist.argmax())
    lo, hi = max(k - 1, 0), min(k + 2, len(hist))
    neighbourhood = int(hist[lo:hi].sum())
    rest = hist.copy()
    rest[lo:hi] = 0
    second = int(rest.max())
    if neighbourhood < MIN_PEAK_RATIO * max(second, 1):
        raise ValueError(f"no dominant offset (peak neighbourhood {neighbourhood} vs next {second})")
    centres = (edges[lo:hi] + edges[lo + 1:hi + 1]) / 2
    offset = float(np.average(centres, weights=hist[lo:hi]))
    top = (second, neighbourhood)

    def pair(predicted, tol):
        idx = np.clip(np.searchsorted(ttl_ms, predicted), 1, len(ttl_ms) - 1)
        left_closer = np.abs(ttl_ms[idx - 1] - predicted) < np.abs(ttl_ms[idx] - predicted)
        cand = np.where(left_closer, idx - 1, idx)
        return cand, np.abs(ttl_ms[cand] - predicted) < tol

    cand, ok = pair(pc_ms + offset, COARSE_TOL_MS)
    slope, intercept = np.polyfit(pc_ms[ok], ttl_ms[cand[ok]], 1)
    cand, ok = pair(slope * pc_ms + intercept, FINE_TOL_MS)
    slope, intercept = np.polyfit(pc_ms[ok], ttl_ms[cand[ok]], 1)
    resid = ttl_ms[cand[ok]] - (slope * pc_ms[ok] + intercept)
    return dict(slope=float(slope), intercept_ms=float(intercept), matched=ok,
                ttl_index=cand, peak=int(top[1]), second_peak=int(top[0]),
                residual_rms_ms=float(np.sqrt(np.mean(resid ** 2))),
                n_ttl_unmatched=int(len(ttl_ms) - len(np.unique(cand[ok]))))


def pycontrol_rsync_ms(path: Path) -> np.ndarray:
    """pyControl rsync event times in ms, from a v1 `.txt` or v2 `.tsv` file."""
    if path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
        t = df.loc[(df["type"] == "event") & (df["content"] == "rsync"), "time"].to_numpy(float)
        return np.rint(t * 1000).astype(np.int64)
    from extract_pokes import parse_pycontrol
    times, _ = parse_pycontrol(path)
    return np.asarray(times.get("rsync", []), dtype=np.int64)


def counter_origin(sample_numbers_path, chunk: int = SCAN_CHUNK) -> dict:
    """Sample number of `.dat` index 0 on the count the TTL events use.

    Returns {origin, restart_at, stale_first, n, n_jumps}; `restart_at` is the first index of
    the final contiguous run (0 when the count never restarts).
    """
    sn = np.load(sample_numbers_path, mmap_mode="r")
    n = len(sn)
    if n == 0:
        raise ValueError(f"{sample_numbers_path}: empty sample_numbers")
    jumps: list[int] = []
    for lo in range(0, n - 1, chunk):
        seg = np.asarray(sn[lo:lo + chunk + 1])           # one sample of overlap covers the seam
        jumps.extend(int(lo + i) for i in np.flatnonzero(np.diff(seg) != 1))
    start = jumps[-1] + 1 if jumps else 0
    if start > MAX_STALE_SAMPLES:
        raise ValueError(f"{sample_numbers_path}: sample-number jump at index {jumps[-1]}, past the "
                         f"first {MAX_STALE_SAMPLES} samples; the TTL count cannot be placed")
    return dict(origin=int(sn[start]) - start, restart_at=start, stale_first=int(sn[0]),
                n=n, n_jumps=len(jumps))


def ttl_rising_relative(row: pd.Series, origin_mode: str = "counter") -> tuple[np.ndarray, dict]:
    """(rising edges as `.dat` indices, origin and QC fields) for one concat-manifest row."""
    if origin_mode not in ORIGIN_MODES:
        raise ValueError(f"origin_mode {origin_mode!r} not in {ORIGIN_MODES}")
    ttl = Path(row["ttl_path"])
    if ttl.suffix == ".npy":            # the concat manifest stores the sample_numbers file itself
        ttl = ttl.parent
    samples = np.load(ttl / "sample_numbers.npy").astype(np.int64)
    states = np.load(ttl / "states.npy")
    rising = samples[states > 0]
    ap_first = int(np.load(row["sample_numbers_path"], mmap_mode="r")[0])
    info = dict(ap_first_sample=ap_first, first_ttl_raw=int(samples[0]) if len(samples) else -1,
                origin_mode=origin_mode)
    if origin_mode == "first_sample":
        info["origin"] = ap_first
    else:
        c = counter_origin(row["sample_numbers_path"])
        rel = rising - c["origin"]
        info.update(origin=c["origin"], restart_at=c["restart_at"], n_jumps=c["n_jumps"],
                    n_samples=c["n"], ttl_outside_dat=int(((rel < 0) | (rel >= c["n"])).sum()),
                    ttl_in_stale_window=int(((rel >= 0) & (rel < c["restart_at"])).sum()))
    return rising - info["origin"], info


def save_new(path: Path, array: np.ndarray) -> str:
    """Write unless an identical array is already there; never replace a differing one."""
    if path.exists():
        if np.array_equal(np.load(path), array):
            return "skip_identical"
        n = 1
        while (alt := path.with_name(path.stem + f".conflict{n}.npy")).exists():
            n += 1
        np.save(alt, array)
        return f"conflict->{alt.name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return "written"


def process_row(row: pd.Series, pc_ms: np.ndarray | None, origin_mode: str = "counter"):
    """One concat-manifest row -> (QC fields, TTL `.dat` indices, rsync train, fit dict or None).

    `pc_ms` is the session's pyControl rsync train, or None for a sleep / unlisted recording,
    whose rsync train is the raw TTL train. The rsync train is None when matching fails.
    """
    rel, info = ttl_rising_relative(row, origin_mode)
    rec = dict(n_ttl=len(rel), **info, first_ttl_rel=int(rel[0]) if len(rel) else None,
               ttl_before_ap_start=int((rel < 0).sum()))
    if pc_ms is None:
        rec.update(kind="sleep_or_unlisted", n_pycontrol=0, n_matched=0)
        return rec, rel, rel, None
    rec.update(kind="task", n_pycontrol=len(pc_ms))
    try:
        m = match_trains(rel / SAMPLES_PER_MS, pc_ms.astype(float))
    except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
        rec.update(n_matched=0, error=f"{type(exc).__name__}: {exc}")
        return rec, rel, None, None
    fitted_ms = m["slope"] * pc_ms + m["intercept_ms"]
    fitted = np.rint(fitted_ms * SAMPLES_PER_MS).astype(np.int64)
    rec.update(n_matched=int(m["matched"].sum()), slope_ephys_per_pc=m["slope"],
               residual_rms_ms=m["residual_rms_ms"], hist_peak=m["peak"],
               hist_second=m["second_peak"], n_ttl_unmatched=m["n_ttl_unmatched"],
               first_pc_rsync_matched=bool(m["matched"][0]),
               first_pc_rsync_sample=int(fitted[0]))
    # The fitted clock itself, for `extract_neuron_raw --sync fitted`:
    # dat_index = a * pycontrol_ms + b, the index on this recording's own continuous.dat
    fit = dict(a_samples_per_ms=m["slope"] * SAMPLES_PER_MS,
               b_samples=m["intercept_ms"] * SAMPLES_PER_MS,
               n_rsync=int(len(pc_ms)), n_matched=int(m["matched"].sum()),
               residual_rms_ms=m["residual_rms_ms"],
               first_rsync_ms=float(pc_ms[0]), last_rsync_ms=float(pc_ms[-1]))
    if origin_mode != "first_sample":      # first_sample must reproduce v2's json text exactly
        fit.update(origin=info["origin"], origin_mode=origin_mode)
    return rec, rel, fitted, fit


def process_recday(recday: str, write: bool, done: set, *, origin_mode: str = "counter",
                   out_dir: Path = PROCESSED / DEFAULT_OUT,
                   compare_dir: Path | None = None) -> list[dict]:
    """QC rows for every recording of one block; writes into `out_dir` with `write`."""
    mouse, _ = rr.recday_dates(recday)
    concat = pd.read_csv(rr.kilosort_dir(recday).parent / "recording_sessions_in_concat.csv")
    task_rows = rr.task_session_rows(recday)
    cands = [(str(d).replace("-", ""), str(t)) for d, t in zip(concat["date"], concat["time"])]
    stamp_to_session = {}
    for i, r in task_rows.iterrows():
        stamp = rr.match_ephys_stamp(recday, i, cands)     # exact, else unique within 60 s
        if stamp is not None:
            stamp_to_session[(str(r["_date"]), stamp)] = i
    out = []
    for _, row in concat.iterrows():
        date = str(row["date"]).replace("-", "")
        key = (mouse, date, str(row["time"]))
        if key in done:                       # the same recording sits in a merged and a single block
            continue
        done.add(key)
        tag = f"{row['date']}_{row['time']}"
        session = stamp_to_session.get((date, str(row["time"])))
        pc_ms = None
        extra = {}
        if session is not None:
            try:
                pc_path = rr.pycontrol_file(recday, session)
            except (FileNotFoundError, KeyError, AssertionError) as exc:
                # ah10 2025-07-18 has ephys rows but no pyControl file; one such row must not
                # stop the block, and it must not pass for a sleep recording either
                out.append(dict(mouse=mouse, recording=tag, recday_seen=recday, kind="task",
                                session=session, n_pycontrol=0, n_matched=0,
                                error=f"pycontrol: {type(exc).__name__}: {exc}"))
                continue
            pc_ms = pycontrol_rsync_ms(pc_path)
            extra = dict(session=session, pycontrol=pc_path.name)
        fields, rel, fitted, fit = process_row(row, pc_ms, origin_mode)
        if fit is not None:
            fit["pycontrol"] = extra["pycontrol"]
        rec = dict(mouse=mouse, recording=tag, recday_seen=recday, **fields, **extra)
        if compare_dir is not None:
            rec.update(compare_existing(compare_dir / mouse, tag, rel, fitted, fit))
        if write:
            rec["ttl_file"] = save_new(out_dir / mouse / f"ttl_rising_{tag}.npy", rel)
            if fitted is not None:
                rec["rsync_file"] = save_new(out_dir / mouse / f"rsync_timestamps_{tag}.npy", fitted)
            if fit is not None:
                rec["fit_file"] = save_new_json(out_dir / mouse / f"rsync_fit_{tag}.json", fit)
        out.append(rec)
    return out


def compare_existing(folder: Path, tag: str, ttl, rsync, fit) -> dict:
    """Read-only twin of the writes, per file: 'identical', 'DIFFERS', 'absent' (computed, not on
    disk), 'stale' (on disk, not computed now) or '-'.

    'stale' is not a difference in what the code computes: four bp01 2024-03-21 fits in mEC's
    `ephys_rsync/` survive from a first run that mapped bp01's 12:59 pyControl file to the 13:23
    recording, which every later run, and the registry, treat as sleep.
    """
    import json

    def status(on_disk: bool, new_none: bool, same) -> str:
        if not on_disk:
            return "-" if new_none else "absent"
        if new_none:
            return "stale"
        return "identical" if same() else "DIFFERS"

    out = {}
    for key, name, new in (("cmp_ttl", f"ttl_rising_{tag}.npy", ttl),
                           ("cmp_rsync", f"rsync_timestamps_{tag}.npy", rsync)):
        p = folder / name
        out[key] = status(p.exists(), new is None, lambda: np.array_equal(np.load(p), new))
    p = folder / f"rsync_fit_{tag}.json"
    text = None if fit is None else json.dumps(fit, indent=1, sort_keys=True)
    out["cmp_fit"] = status(p.exists(), text is None, lambda: p.read_text() == text)
    return out


def save_new_json(path: Path, obj: dict) -> str:
    """JSON twin of `save_new`: identical content is skipped, differing content goes beside."""
    import json
    text = json.dumps(obj, indent=1, sort_keys=True)
    if path.exists():
        if path.read_text() == text:
            return "skip_identical"
        n = 1
        while (alt := path.with_name(path.stem + f".conflict{n}.json")).exists():
            n += 1
        alt.write_text(text)
        return f"conflict->{alt.name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return "written"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--recdays", nargs="*", default=None)
    ap.add_argument("--origin", choices=ORIGIN_MODES, default="counter",
                    help="TTL sample-number origin: counter (default) or first_sample (v2's rule)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output tree under data/processed_data")
    ap.add_argument("--compare-to", default=None,
                    help="read-only: compare every output with this tree under data/processed_data")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.write and args.compare_to:
        ap.error("--compare-to is read-only; run it without --write")
    out_dir = PROCESSED / args.out
    compare_dir = PROCESSED / args.compare_to if args.compare_to else None

    rows, done = [], set()
    for recday in args.recdays or rr.all_recdays():
        rows.extend(process_recday(recday, args.write, done, origin_mode=args.origin,
                                   out_dir=out_dir, compare_dir=compare_dir))
    qc = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    task = qc[qc["kind"] == "task"]
    print(f"{len(qc)} recordings ({len(task)} task, {len(qc) - len(task)} sleep/unlisted)  "
          f"[origin={args.origin}, out={args.out}]")
    if len(task):
        print("task recordings: n_ttl vs n_pycontrol vs n_matched")
        cols = ["mouse", "recording", "session", "n_ttl", "n_pycontrol", "n_matched", "n_ttl_unmatched",
                "hist_peak", "hist_second", "first_pc_rsync_matched", "slope_ephys_per_pc",
                "residual_rms_ms", "ap_first_sample", "origin", "restart_at", "ttl_before_ap_start"] + (
                    ["error"] if "error" in task else [])
        print(task[[c for c in cols if c in task]].to_string(index=False))
        ok = task[task["n_matched"] > 0]
        if len(ok):
            frac = ok["n_matched"] / ok["n_pycontrol"]
            print(f"\nmatched recordings: {len(ok)}/{len(task)}; pyControl pulses matched "
                  f"min {frac.min():.3f} median {frac.median():.3f}; slope-1 range "
                  f"{(ok['slope_ephys_per_pc'] - 1).min():.2e} .. {(ok['slope_ephys_per_pc'] - 1).max():.2e}; "
                  f"residual rms ms median {ok['residual_rms_ms'].median():.3f} max {ok['residual_rms_ms'].max():.3f}")
            print(f"first pyControl rsync observed in TTL: {int(ok['first_pc_rsync_matched'].sum())}/{len(ok)}")
    if "restart_at" in qc:
        restart = qc["restart_at"] > 0
        print(f"\ncounter origin: {int(restart.sum())}/{len(qc)} recordings restart the count "
              f"(task {int((restart & (qc['kind'] == 'task')).sum())}); first sample > 0 without a "
              f"restart: {int(((qc['ap_first_sample'] > 0) & ~restart).sum())}; TTL edges outside the "
              f".dat: {int(qc['ttl_outside_dat'].sum())}; in a stale window: {int(qc['ttl_in_stale_window'].sum())}")
    failed = False
    if compare_dir is not None:
        for col in ("cmp_ttl", "cmp_rsync", "cmp_fit"):
            print(f"{col} vs {args.compare_to}: {qc[col].value_counts().to_dict()}")
        stale = qc[(qc[["cmp_ttl", "cmp_rsync", "cmp_fit"]] == "stale").any(axis=1)]
        if len(stale):
            print(f"\n{len(stale)} recording(s) have files in {args.compare_to} that this run does not compute "
                  f"(left by an earlier run; not a difference in what the code computes):")
            print(stale[["mouse", "recording", "kind", "cmp_ttl", "cmp_rsync", "cmp_fit"]].to_string(index=False))
        differs = qc[(qc[["cmp_ttl", "cmp_rsync", "cmp_fit"]] == "DIFFERS").any(axis=1)]
        if len(differs):
            failed = True
            print(f"\n{len(differs)} recording(s) DIFFER from {args.compare_to}:")
            print(differs[["mouse", "recording", "cmp_ttl", "cmp_rsync", "cmp_fit"]].to_string(index=False))
    if args.write:
        qc_path = PROCESSED / f"{args.out}_qc.csv"
        n = 0
        while (p := qc_path.with_name(qc_path.stem + (f".v{n}" if n else "") + ".csv")).exists():
            n += 1
        qc.to_csv(p, index=False)
        print(f"QC -> {p}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
