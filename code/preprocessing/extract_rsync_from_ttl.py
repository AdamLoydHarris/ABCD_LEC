"""
Rsync pulse trains from the OpenEphys TTL streams, for `extract_neuron_raw.first_rsync_sample`.

LEC's `ephys_rsync_mingyutest/` caches were written by an untracked script. The mEC tree has
none, so this derives them from the raw TTL events and, unlike the caches, records how.

Per recording in a block's `recording_sessions_in_concat.csv` (using that row's own
`ttl_path` and `sample_numbers_path`, i.e. the mouse's own probe stream):

1. Rising edges of the sync line (`states > 0`), in OpenEphys sample numbers, made relative
   to the AP continuous stream's first sample -- the timebase Kilosort spike times use once
   `extract_neuron_raw` subtracts the recording's concat offset. Saved raw as
   `ttl_rising_{date}_{time}.npy`.
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

Measured on 2024-03-24_10-52-32: the first TTL edges sit ~136k samples BEFORE the AP
stream's first sample (the events stream starts earlier than the continuous one), which is
why matching by intervals rather than "first TTL = first rsync" is not optional here.

QC per recording -> `ephys_rsync_qc.csv`: n_ttl, n_pycontrol, n_matched, slope, residual.
Nothing is overwritten.

Usage::

    python extract_rsync_from_ttl.py --dataset mec [--recdays ...] [--write]
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

RSYNC_PATH = rr.REPO_ROOT / "data" / "processed_data" / "ephys_rsync"
QC_PATH = rr.REPO_ROOT / "data" / "processed_data" / "ephys_rsync_qc.csv"
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


def ttl_rising_relative(row: pd.Series) -> tuple[np.ndarray, int, int]:
    """(rising-edge samples relative to the AP stream's first sample, ap_first, first_raw_ttl)."""
    ttl = Path(row["ttl_path"])
    if ttl.suffix == ".npy":            # the concat manifest stores the sample_numbers file itself
        ttl = ttl.parent
    samples = np.load(ttl / "sample_numbers.npy").astype(np.int64)
    states = np.load(ttl / "states.npy")
    rising = samples[states > 0]
    ap_first = int(np.load(row["sample_numbers_path"], mmap_mode="r")[0])
    return rising - ap_first, ap_first, int(samples[0]) if len(samples) else -1


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


def process_recday(recday: str, write: bool, done: set) -> list[dict]:
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
        rec = dict(mouse=mouse, recording=tag, recday_seen=recday)
        rel, ap_first, first_raw = ttl_rising_relative(row)
        rec.update(n_ttl=len(rel), ap_first_sample=ap_first, first_ttl_raw=first_raw,
                   first_ttl_rel=int(rel[0]) if len(rel) else None,
                   ttl_before_ap_start=int((rel < 0).sum()))

        session = stamp_to_session.get((date, str(row["time"])))
        fit = None
        if session is None:
            rec.update(kind="sleep_or_unlisted", n_pycontrol=0, n_matched=0)
            fitted = rel
        else:
            pc_path = rr.pycontrol_file(recday, session)
            pc_ms = pycontrol_rsync_ms(pc_path)
            rec.update(kind="task", session=session, pycontrol=pc_path.name, n_pycontrol=len(pc_ms))
            try:
                m = match_trains(rel / SAMPLES_PER_MS, pc_ms.astype(float))
                fitted_ms = m["slope"] * pc_ms + m["intercept_ms"]
                fitted = np.rint(fitted_ms * SAMPLES_PER_MS).astype(np.int64)
                rec.update(n_matched=int(m["matched"].sum()), slope_ephys_per_pc=m["slope"],
                           residual_rms_ms=m["residual_rms_ms"], hist_peak=m["peak"],
                           hist_second=m["second_peak"], n_ttl_unmatched=m["n_ttl_unmatched"],
                           first_pc_rsync_matched=bool(m["matched"][0]),
                           first_pc_rsync_sample=int(fitted[0]))
                # The fitted clock itself, for `extract_neuron_raw --sync fitted`:
                # ephys_sample = a * pycontrol_ms + b  (relative to the recording's first AP sample)
                fit = dict(a_samples_per_ms=m["slope"] * SAMPLES_PER_MS,
                           b_samples=m["intercept_ms"] * SAMPLES_PER_MS,
                           n_rsync=int(len(pc_ms)), n_matched=int(m["matched"].sum()),
                           residual_rms_ms=m["residual_rms_ms"], pycontrol=pc_path.name,
                           first_rsync_ms=float(pc_ms[0]), last_rsync_ms=float(pc_ms[-1]))
            except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
                rec.update(n_matched=0, error=f"{type(exc).__name__}: {exc}")
                fitted = None

        if write:
            rec["ttl_file"] = save_new(RSYNC_PATH / mouse / f"ttl_rising_{tag}.npy", rel)
            if fitted is not None:
                rec["rsync_file"] = save_new(RSYNC_PATH / mouse / f"rsync_timestamps_{tag}.npy", fitted)
            if fit is not None:
                rec["fit_file"] = save_new_json(RSYNC_PATH / mouse / f"rsync_fit_{tag}.json", fit)
        out.append(rec)
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
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows, done = [], set()
    for recday in args.recdays or rr.all_recdays():
        rows.extend(process_recday(recday, args.write, done))
    qc = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    task = qc[qc["kind"] == "task"]
    print(f"{len(qc)} recordings ({len(task)} task, {len(qc) - len(task)} sleep/unlisted)")
    if len(task):
        print("task recordings: n_ttl vs n_pycontrol vs n_matched")
        cols = ["mouse", "recording", "session", "n_ttl", "n_pycontrol", "n_matched", "n_ttl_unmatched",
                "hist_peak", "hist_second", "first_pc_rsync_matched", "slope_ephys_per_pc",
                "residual_rms_ms", "ttl_before_ap_start"] + (["error"] if "error" in task else [])
        print(task[[c for c in cols if c in task]].to_string(index=False))
        ok = task[task["n_matched"] > 0]
        if len(ok):
            frac = ok["n_matched"] / ok["n_pycontrol"]
            print(f"\nmatched recordings: {len(ok)}/{len(task)}; pyControl pulses matched "
                  f"min {frac.min():.3f} median {frac.median():.3f}; slope-1 range "
                  f"{(ok['slope_ephys_per_pc'] - 1).min():.2e} .. {(ok['slope_ephys_per_pc'] - 1).max():.2e}; "
                  f"residual rms ms median {ok['residual_rms_ms'].median():.3f} max {ok['residual_rms_ms'].max():.3f}")
            print(f"first pyControl rsync observed in TTL: {int(ok['first_pc_rsync_matched'].sum())}/{len(ok)}")
    if args.write:
        n = 0
        while (p := QC_PATH.with_name(QC_PATH.stem + (f".v{n}" if n else "") + ".csv")).exists():
            n += 1
        qc.to_csv(p, index=False)
        print(f"QC -> {p}")


if __name__ == "__main__":
    main()
