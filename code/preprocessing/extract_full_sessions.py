"""Full LEC task sessions on the v4 spike clock: the pre-A and post-last-A segments that
`Neuron_raw` cuts, pre-A tracking, a recording index and cheap unit stability.

`Neuron_raw` runs from the first A to the last completed trial, so maze entry and the
session's tail are missing (recording start precedes the first A by tens of seconds). This
writes only the missing pieces; nothing that exists is rewritten:

    data/processed_data/neuron_raw_v4_full/Neuron_raw_{pre,post}_<rd>_<s>.npy   (units, bins)
    data/processed_v3_full/<mouse>/<block>/{XY,HD,Locs}_raw_pre_<rd>_<s>.npy     (bins, ...)
    data/processed_data/full_sessions_v4/full_sessions_<rd>[.vN].csv     one row per task session
    data/processed_data/full_sessions_v4/recording_index_<rd>[.vN].csv   one row per recording
    data/processed_data/full_sessions_v4/unit_stability_<rd>[.vN].csv    one row per unit x recording
    data/processed_data/full_sessions_v4/gates_<rd>[.vN].csv             spike gate per session

Full array = pre + `neuron_raw_v4` + post, on the same 25 ms grid; `first_A_bin` is a multiple
of 10 so 250 ms blocks line up with V3's. Tracking pre-segments are sampled the way
`sleap_alignment_copilot._save_matched` samples v3 (raw first A_on + 25 k ms through the fitted
frame clock), at negative k.

Gates, exit 1 on any failure: the core slice of every full spike array equals `neuron_raw_v4`
byte for byte; the k >= 0 part of the re-sampled tracking equals `processed_v3` (NaN equal).

    sbatch sbatch_files/mec_stage.sbatch extract_full_sessions.py --recdays <rd> ... [--write]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                           # noqa: E402
import extract_neuron_raw as enr                                       # noqa: E402
from extract_pokes import parse_pycontrol                              # noqa: E402

if rr.DATASET != "lec":
    sys.exit("extract_full_sessions.py is LEC-only (mEC waits for its re-sorts)")

# extract_neuron_raw picks its rsync tree from argv at import; v4 is the only tree without the
# OpenEphys counter-restart error (SPIKE_CLOCK_V4.md).
enr.RSYNC_PATH = rr.REPO_ROOT / "data" / "processed_data" / "ephys_rsync_v4"

PROC = rr.REPO_ROOT / "data" / "processed_data"
V4_DIR = PROC / "neuron_raw_v4"
OUT_DIR = PROC / "neuron_raw_v4_full"
TRACK_V3 = rr.REPO_ROOT / "data" / "processed_v3"
TRACK_OUT = rr.REPO_ROOT / "data" / "processed_v3_full"
TRIALTIMES = PROC / "trialtimes_raw_mingyutest"
BIN_MS = enr.BIN_MS
BLOCK_BINS = 10
MISCLICK_S = 120.0        # rest recordings under 2 min are OpenEphys misclicks (user, 2026-10-01)
_TASK_FILE = re.compile(r"^Neuron_raw_([a-z]{2}\d{2}_\d{8}_\d{8})_(\d+)\.npy$")


def june_recdays() -> list[str]:
    """Recdays with v4 task arrays on disk (the July blocks have none yet)."""
    return sorted({m.group(1) for f in V4_DIR.iterdir() if (m := _TASK_FILE.match(f.name))})


def _versioned(path: Path) -> Path:
    """`path`, or `path.vN` with the first free N: tables are never overwritten."""
    k = 0
    while (p := path.with_name(path.stem + (f".v{k}" if k else "") + path.suffix)).exists():
        k += 1
    return p


def _first_poke_ms(times: dict) -> float:
    pokes = [v[0] for k, v in times.items() if re.fullmatch(r"poke_\d", k) and len(v)]
    return float(min(pokes)) if pokes else np.nan


def full_window(recday: str, session: int, concat_row: int) -> dict:
    """Bin geometry of one task session's full recording on the v4 clock (pyControl ms)."""
    clock, t0, n_core = enr.session_window_fitted(recday, session, concat_row)
    n_samples = int(enr.concat_table(recday)["n_samples"].iloc[concat_row])
    t_rec0 = (0.0 - clock["b"]) / clock["a"]
    t_end = (n_samples - clock["b"]) / clock["a"]
    first_grid = BIN_MS * int(np.ceil(t_rec0 / BIN_MS))
    span = t0 - first_grid
    if span < 0:
        raise ValueError(f"{recday} s{session}: recording starts {-span} ms after bin 0")
    step = BIN_MS * BLOCK_BINS
    t0_full = t0 - step * (span // step)
    n_full = int(np.floor((t_end - t0_full) / BIN_MS))
    k = (t0 - t0_full) // BIN_MS
    if n_full < k + n_core:
        raise ValueError(f"{recday} s{session}: full window {n_full} < {k} + {n_core}")
    return dict(clock=clock, t0_ms=int(t0), t0_full_ms=int(t0_full), t_rec0_ms=float(t_rec0),
                t_end_ms=float(t_end), first_A_bin=int(k), n_core=int(n_core),
                n_post=int(n_full - k - n_core), n_full=n_full)


# ---------------------------------------------------------------------------
# spikes, recording index, unit stability
# ---------------------------------------------------------------------------

def _load_block(recday: str):
    ks = rr.kilosort_dir(recday)
    qc = np.load(ks / "QC_single_units.npy")
    st = np.load(ks / "sorter_output/spike_times.npy").squeeze()
    sc = np.load(ks / "sorter_output/spike_clusters.npy").squeeze()
    amp = np.load(ks / "sorter_output/amplitudes.npy").squeeze()
    lut = np.full(int(sc.max()) + 2, -1, dtype=np.int64)
    lut[qc] = np.arange(len(qc))
    order = np.argsort(st, kind="stable")
    ops = np.load(ks / "sorter_output/ops.npy", allow_pickle=True).item()
    return qc, st[order], sc[order], amp[order], lut, ops


def _wall(date: str, hhmmss: str) -> datetime:
    return datetime.strptime(f"{date} {hhmmss}", "%Y%m%d %H-%M-%S")


def _stability_rows(recday, kind, index, rows_t, samples, unit_rows, amps, n_units, n_samples):
    """Per unit: spike count, median KS amplitude, first/last-quarter amplitude ratio and the
    within-recording amplitude slope as a fraction of the median per 10 min."""
    out = []
    q = n_samples // 4
    for u in range(n_units):
        m = unit_rows == u
        n = int(m.sum())
        row = dict(recday=recday, kind=kind, index=index, unit=u, n_spikes=n, **rows_t)
        if n >= 20:
            a, t = amps[m], samples[m] / enr.SAMPLE_RATE_HZ
            med = float(np.median(a))
            first, last = a[samples[m] < q], a[samples[m] >= n_samples - q]
            slope = np.polyfit(t, a, 1)[0] if np.ptp(t) > 0 else np.nan
            row.update(amp_median=med,
                       amp_last_over_first=(float(np.median(last) / np.median(first))
                                            if len(first) >= 5 and len(last) >= 5 else np.nan),
                       amp_slope_frac_per_10min=float(slope * 600.0 / med) if med else np.nan)
        out.append(row)
    return out


def spikes_for_recday(recday: str, write: bool):
    qc, st, sc, amp, lut, ops = _load_block(recday)
    n_units = len(qc)
    table = enr.concat_table(recday)
    wake, sleep = enr.session_map(recday)
    # KS4 rigid drift, one value per batch of `batch_size` samples on the concatenated stream.
    dshift = np.asarray(ops["dshift"]) if "dshift" in ops else np.zeros((0, 1))
    dshift = dshift.reshape(len(dshift), -1) if len(dshift) else np.zeros((0, 1))
    nt = int(ops.get("batch_size", ops.get("NT", 60000)))
    session_rows, index_rows, stab_rows, gate = [], [], [], []

    kind_of = {r: ("task", i) for i, r in enumerate(wake)}
    kind_of.update({r: ("rest", j) for j, r in enumerate(sleep)})
    for concat_row in range(len(table)):
        kind, idx = kind_of[concat_row]
        start = int(table["start"].iloc[concat_row])
        n_samples = int(table["n_samples"].iloc[concat_row])
        lo, hi = np.searchsorted(st, [start, start + n_samples])
        samples, unit_rows, amps = st[lo:hi] - start, lut[sc[lo:hi]], amp[lo:hi]
        date, hhmmss = str(table["date"].iloc[concat_row]), str(table["time"].iloc[concat_row])
        w0 = _wall(date, hhmmss)
        b0, b1 = start // nt, (start + n_samples) // nt
        ds = dshift[b0:b1, 0] if len(dshift) else np.zeros(0)
        index_rows.append(dict(recday=recday, concat_row=concat_row, kind=kind, index=idx,
                               date=date, time=hhmmss, wall_start=w0.isoformat(),
                               duration_s=n_samples / enr.SAMPLE_RATE_HZ,
                               dshift_median_um=float(np.median(ds)) if len(ds) else np.nan,
                               dshift_p2p_um=float(np.ptp(ds)) if len(ds) else np.nan))
        stab_rows += _stability_rows(recday, kind, idx, dict(concat_row=concat_row), samples,
                                     unit_rows, amps, n_units, n_samples)
        if kind != "task":
            continue

        s = idx
        name = f"{recday}_{s}"
        stype = rr.session_type(recday, s)
        try:
            win = full_window(recday, s, concat_row)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            gate.append(dict(recday=recday, session=s, spikes="SKIPPED", detail=str(exc)))
            continue
        full = enr.bin_session_fitted(samples, unit_rows, n_units, win["clock"],
                                      win["t0_full_ms"], win["n_full"])
        core = np.load(V4_DIR / f"Neuron_raw_{name}.npy")
        k, n_core = win["first_A_bin"], win["n_core"]
        exact = (core.shape == (n_units, n_core)
                 and np.array_equal(full[:, k:k + n_core], core))
        gate.append(dict(recday=recday, session=s, spikes="exact" if exact else "MISMATCH",
                         detail="" if exact else f"core {core.shape} vs full slice "
                                                  f"{full[:, k:k + n_core].shape}"))
        times, _ = enr._parse_pycontrol(rr.pycontrol_file(recday, s))
        tt = enr.trialtimes(recday, s)
        session_rows.append(dict(
            recday=recday, session=s, session_type=stype, concat_row=concat_row,
            n_trials=int(len(tt)), n_units=n_units, wall_rec_start=w0.isoformat(),
            wall_bin0=(w0 + timedelta(milliseconds=win["t0_full_ms"] - win["t_rec0_ms"])).isoformat(),
            t_rec0_ms=win["t_rec0_ms"], t_end_ms=win["t_end_ms"], t0_ms=win["t0_ms"],
            t0_full_ms=win["t0_full_ms"], first_A_bin=k, n_pre=k, n_core=n_core,
            n_post=win["n_post"], n_full=win["n_full"],
            pycontrol_start_bin=-win["t0_full_ms"] / BIN_MS,
            first_poke_bin=(_first_poke_ms(times) - win["t0_full_ms"]) / BIN_MS,
            first_A_on_ms=float(times["A_on"][0]) if len(times.get("A_on", [])) else np.nan))
        if write and exact:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            for part, arr in (("pre", full[:, :k]), ("post", full[:, k + n_core:])):
                p = OUT_DIR / f"Neuron_raw_{part}_{name}.npy"
                if not p.exists():
                    np.save(p, arr.astype(enr.NEURON_DTYPE))
    return session_rows, index_rows, stab_rows, gate


def add_index_context(index: pd.DataFrame) -> pd.DataFrame:
    """Same-date gaps, the preceding task of each rest block (by session and by task value),
    pre/post stage and the misclick flag."""
    out = []
    for (rd, date), g in index.groupby(["recday", "date"], sort=False):
        g = g.sort_values("wall_start").copy()
        start = pd.to_datetime(g["wall_start"])
        end = start + pd.to_timedelta(g["duration_s"], unit="s")
        g["gap_prev_s"] = (start - end.shift(1)).dt.total_seconds()
        last_task, prev = [], None
        for _, r in g.iterrows():
            last_task.append(prev)
            if r["kind"] == "task":
                prev = int(r["index"])
        g["preceding_task_session"] = last_task
        first_task = start[g["kind"] == "task"].min()
        g["stage"] = np.where(start < first_task, "pre", "post")
        g["misclick"] = (g["kind"] == "rest") & (g["duration_s"] < MISCLICK_S)
        out.append(g)
    idx = pd.concat(out, ignore_index=True)

    def task_of(row):
        s = row["preceding_task_session"]
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return None
        p = TRIALTIMES / f"Task_data_{row['recday']}_{int(s)}.npy"
        return tuple(int(x) for x in np.load(p, allow_pickle=True)) if p.exists() else None

    idx["preceding_task"] = idx.apply(task_of, axis=1)
    return idx


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------

def tracking_for_recday(recday: str, sessions: pd.DataFrame, write: bool) -> list[dict]:
    """Re-sample v3 tracking over each session's full window; gate the k >= 0 part."""
    import sleap_alignment_copilot as cop
    cop.SYNC_METHOD, cop.SESSIONS = "matched", "all"
    cop.set_tracking_clean("v2")
    captured: dict[int, tuple] = {}

    def capture(mouse_id, date1, date2, session_idx, output_dir, sleap_data, pinstate,
                pycontrol_data, first_A_on, used_fallback_mouse, pinstate_name, verbose):
        captured[session_idx] = (sleap_data, pinstate, pycontrol_data, first_A_on,
                                 used_fallback_mouse)
        return True

    saved, cop._save_matched = cop._save_matched, capture
    mouse, (d1, d2) = rr.recday_dates(recday)
    try:
        folder = rr.EPHYS_ROOT / mouse / rr.recday_to_block(recday)
        cop.process_mouse_ephys_folder_by_metadata(mouse, folder, cop.load_metadata(mouse),
                                                   verbose=False)
    finally:
        cop._save_matched = saved

    block = f"{d1}_{d2}"
    rows = []
    for _, srow in sessions.iterrows():
        s = int(srow["session"])
        name = f"{recday}_{s}"
        if s not in captured:
            rows.append(dict(recday=recday, session=s, tracking="NO_SLEAP", detail=""))
            continue
        sleap_data, pinstate, pyc, first_A_on, fallback = captured[s]
        if fallback is not None:
            rows.append(dict(recday=recday, session=s, tracking="FALLBACK_REFUSED", detail=fallback))
            continue
        clock = cop.fit_frame_clock(cop.rising_edges(pinstate), cop.all_rsync_timestamps(pyc))
        n_frames = min(len(v) for v in sleap_data.values())
        k, n_full = int(srow["first_A_bin"]), int(srow["n_full"])
        t_bins = first_A_on + np.arange(-k, n_full - k) * (1000.0 / cop.TARGET_FPS)
        frames = cop.frames_at(clock, t_bins)
        uncertain = cop._frame_clock.uncertain_mask(clock, t_bins)
        outside = (frames < 0) | (frames > n_frames - 1)

        df = sleap_data["coordinates"]
        cols = ["head_back.x", "head_back.y"]
        coords = df[cols].values if all(c in df.columns for c in cols) else df.values[:, 4:6]
        xy = cop.sample_at_frames(coords, frames, "linear")
        xy[uncertain | outside] = np.nan
        if cop.PROPAGATE_TRACK_NAN:
            xy[cop.untracked_bins(coords, frames)] = np.nan
        # Object-box sessions have no ROI file (no maze locations); HD can be absent too.
        if "head_direction" in sleap_data:
            hd = cop.sample_at_frames(sleap_data["head_direction"].values, frames, "nearest").astype(float)
            hd[uncertain | outside] = np.nan
        else:
            hd = None
        if "rois" in sleap_data:
            rdf = sleap_data["rois"]
            rois = rdf["head_back"].values if "head_back" in rdf.columns else rdf.values[:, 2]
            locs = cop.sample_at_frames(rois, frames, "nearest")
            locs = locs.astype(object) if locs.dtype.kind in "OU" else locs
            locs[uncertain | outside] = 0
        else:
            locs = None

        ok, detail = True, []
        for tag, arr in (("XY", xy), ("HD", hd), ("Locs", locs)):
            p = TRACK_V3 / mouse / block / f"{tag}_raw_{name}.npy"
            if arr is None or not p.exists():
                if tag == "XY" or (arr is not None) != p.exists():
                    ok = False
                    detail.append(f"{tag}: source {'absent' if arr is None else 'present'}, "
                                  f"v3 {'present' if p.exists() else 'absent'}")
                continue
            # The video can outlast the ephys recording, so only the overlap is compared.
            ref = np.load(p, allow_pickle=True)
            n = min(len(ref), n_full - k)
            mine, ref = arr[k:k + n], ref[:n]
            if tag == "Locs":
                same = np.array_equal(np.asarray(mine, dtype=str), np.asarray(ref, dtype=str))
            else:
                a, b = np.asarray(mine, float), np.asarray(ref, float)
                same = a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) \
                    and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
            if not same:
                ok = False
                detail.append(f"{tag} differs")
        pre_tracked = np.isfinite(xy[:k]).all(axis=1) if k else np.zeros(0, bool)
        v3_len = len(np.load(TRACK_V3 / mouse / block / f"XY_raw_{name}.npy", mmap_mode="r")) \
            if (TRACK_V3 / mouse / block / f"XY_raw_{name}.npy").exists() else 0
        rows.append(dict(recday=recday, session=s, tracking="exact" if ok else "MISMATCH",
                         detail="; ".join(detail), v3_bins_after_A=v3_len,
                         video_start_bin=float(np.interp(0.0, frames, np.arange(len(frames)))
                                               if frames[0] < 0 < frames[-1] else 0.0),
                         pre_tracked_frac=float(pre_tracked.mean()) if k else np.nan))
        if write and ok and k:
            out = TRACK_OUT / mouse / block
            out.mkdir(parents=True, exist_ok=True)
            for tag, arr in (("XY", xy), ("HD", hd), ("Locs", locs)):
                p = out / f"{tag}_raw_pre_{name}.npy"
                if arr is not None and not p.exists():
                    np.save(p, arr[:k], allow_pickle=True)
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

FULL_TABLES = PROC / "full_sessions_v4"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recdays", nargs="*", default=None, help="default: every recday with v4 arrays")
    ap.add_argument("--write", action="store_true", help="create missing outputs (never overwrites)")
    ap.add_argument("--skip-tracking", action="store_true")
    args = ap.parse_args()
    recdays = args.recdays or june_recdays()

    all_sess, all_gate, all_tgate = [], [], []
    for rd in recdays:
        print(f"{rd}: spikes", flush=True)
        srows, irows, strows, g = spikes_for_recday(rd, args.write)
        tg = []
        if not args.skip_tracking and srows:
            print(f"{rd}: tracking", flush=True)
            tg = tracking_for_recday(rd, pd.DataFrame(srows), args.write)
        sess_df, gate, tgate = pd.DataFrame(srows), pd.DataFrame(g), pd.DataFrame(tg)
        if len(tgate):
            sess_df = sess_df.merge(tgate, on=["recday", "session"], how="left")
        if args.write:
            FULL_TABLES.mkdir(parents=True, exist_ok=True)
            tables = {"full_sessions": sess_df, "recording_index": add_index_context(pd.DataFrame(irows)),
                      "unit_stability": pd.DataFrame(strows), "gates": gate}
            for stem, df in tables.items():
                p = _versioned(FULL_TABLES / f"{stem}_{rd}.csv")
                df.to_csv(p, index=False)
            print(f"{rd}: tables written to {FULL_TABLES.name}/", flush=True)
        all_sess.append(sess_df)
        all_gate.append(gate)
        all_tgate.append(tgate)

    sess_df = pd.concat(all_sess, ignore_index=True)
    gate = pd.concat(all_gate, ignore_index=True)
    tgate = pd.concat([t for t in all_tgate if len(t)], ignore_index=True) if any(len(t) for t in all_tgate) else pd.DataFrame()
    bad_s = gate[gate["spikes"] != "exact"] if len(gate) else gate
    bad_t = tgate[~tgate["tracking"].isin(["exact"])] if len(tgate) else tgate
    print(f"\nspike gate: {int((gate['spikes'] == 'exact').sum()) if len(gate) else 0}/{len(gate)} exact")
    if len(tgate):
        print(f"tracking gate: {int((tgate['tracking'] == 'exact').sum())}/{len(tgate)} exact")
        k = sess_df["n_pre"].astype(float) * BIN_MS / 1000
        print(f"pre-A window: median {k.median():.1f} s, range {k.min():.1f}-{k.max():.1f} s; "
              f"pre-A tracked fraction median {sess_df['pre_tracked_frac'].median():.3f}")
    if len(bad_s) or len(bad_t):
        print("\nFAILURES:")
        if len(bad_s):
            print(bad_s.to_string(index=False))
        if len(bad_t):
            print(bad_t.to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
