"""
Brain surface from the raw AP data: which upper channels sat outside the brain in each recording?

Why (2026-09-19): the sorter's `channel_assignments.json` labelled 0 of 384 channels `out` in all 36
mEC blocks, but it ran spikeinterface's IBL-style `detect_bad_channels` on data the NP1 hardware had
already high-passed at 300 Hz (no LFP stream was saved), so the low-frequency coherence that method
relies on was gone. Read directly, the top of the bank shows the signature of channels in fluid or
dura in most sampled recordings: shared low-frequency noise (neighbour correlation ~1 vs 0.2-0.4 in
tissue) and threshold crossings that are coincident across many channels (common-mode noise) rather
than local (spikes).

Per recording: a window of `SECS` s from the middle of `continuous.dat` (int16, samples x 384),
common-median referenced; per channel the 300-6000 Hz RMS, the -4.5 MAD negative-crossing rate, the
fraction of those crossings that coincide (within 1 sample) on >= COINC_MIN channels of the same
half-probe, the 2-200 Hz RMS (on 3 kHz decimated data), and the correlation with the channel two
rows up (40 um) in the 2-200 Hz and the 300-1000 Hz bands. Channels are averaged into the 192 rows.
A row is FLUID when corr_hf >= 0.8, corr_lf >= 0.5 and (coincident_frac >= 0.5 or xing < 0.5/s);
the surface is the lowest row with a MIN_ROWS solid fluid base above which >= 80 % of rows are
fluid; otherwise the bank is reported inside. Confidence = mean over the fluid rows of the number of
signatures (of three) that fired.

Outputs (mEC tree, never overwritten): data/processed_data/probe_surface/{mouse}/{recording}.csv
(per-row features), data/processed_data/probe_surface_mec[.vN].csv (one row per recording), and
data/processed_data/probe_surface_{mouse}[.vN].png.

Gate (`--gate`): the detector is fed one real window with its top 60 rows replaced by a shared noise
waveform plus small independent noise and must return the planted surface row within 2 rows; the
unmodified window of a recording whose bank is inside must return "inside".

Usage (maze_ephys env; under sbatch_files/mec_stage.sbatch):
  python probe_surface_from_raw.py --dataset mec [--mouse bp01] [--secs 20] [--gate] [--write]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recday_registry as rr                                          # noqa: E402

FS = 30000
N_CH = 384
SECS = 20
ROW_UM = 20.0
TIP_OFFSET_UM = 175.0          # NP1: distance from the shank tip to the first electrode row
#: Common-mode crossings: a spike never spans 12 rows (24 channels), fluid noise spans the whole
#: half-probe. The first run used 6 channels and counted bp04's large mid-bank units (coincident
#: fraction 0.8-0.9 in its dense band) as common-mode, so it needed all three signatures and
#: flagged only 3 of 26 bp04 recordings although the figure shows the same fluid band in all 26.
COINC_MIN = 16                 # 8 rows: above any spike's footprint, below the smallest fluid band seen (11 rows)
MIN_ROWS = 5                   # 100 um: the smallest segment reported as a surface
#: Fluid signatures measured on bp01/bp02/bp04 (2026-09-19): spike-band neighbour correlation
#: 0.95-1.0 in fluid vs <= 0.75 in tissue; low-band 0.8-1.0 vs <= 0.8 (dense LFP bands reach 0.8);
#: common-mode fraction ~1 vs low once COINC_MIN excludes spikes.
CORR_HF_FLUID = 0.8
CORR_LF_FLUID = 0.5
#: Above the fluid band the topmost 2-4 rows of bp04 (y >= 3760 um) lose the neighbour correlation
#: again while keeping common-mode or no crossings -- above the fluid level, in air or dura. They
#: are not tissue, so the fraction of fluid rows above the surface only has to reach 0.6 and no run
#: of >= 3 tissue-like rows (local spiking, not common-mode) may lie above it.
FLUID_FRAC = 0.6
TISSUE_RUN_MAX = 2
OUT = rr.REPO_ROOT / "data" / "processed_data"


def task_recordings(mouse: str) -> pd.DataFrame:
    """Task recordings (rsync-matched) with their continuous.dat path, from the rsync QC and the
    per-mouse ephys metadata."""
    qc = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("ephys_rsync_qc*.csv"))])
    qc = qc[(qc["mouse"] == mouse) & (qc["kind"] == "task")].drop_duplicates("recording", keep="last")
    md = pd.read_csv(rr.REPO_ROOT / "data" / "metadata" / f"{mouse}_ephys_metadata.csv")
    md = md[md["stream_name"].str.contains("-AP")].drop_duplicates("Recording_ID", keep="last")
    df = qc.merge(md[["Recording_ID", "continuous_path", "dat_size"]], left_on="recording", right_on="Recording_ID", how="left")
    df["dur_s"] = df["dat_size"] / (N_CH * 2 * FS)
    return df.sort_values("recording").reset_index(drop=True)


def channel_rows() -> np.ndarray:
    """Row index (0 = tip-most) of every channel from a sorted block's channel_positions.npy."""
    p = sorted((rr.REPO_ROOT / "data" / "preprocessed" / "ephys").glob("*/*_preprocessed/kilosort_output/sorter_output/channel_positions.npy"))[0]
    y = np.load(p)[:, 1]
    return np.rint(y / ROW_UM).astype(int)


def read_window(path: str, secs: float = SECS, where: float = 0.5) -> np.ndarray:
    mm = np.memmap(path, dtype=np.int16, mode="r")
    n = mm.size // N_CH
    s0 = int(n * where) - int(secs * FS) // 2
    s0 = max(0, min(s0, n - int(secs * FS)))
    x = np.array(mm[s0 * N_CH:(s0 + int(secs * FS)) * N_CH]).reshape(-1, N_CH).astype(np.float32)
    x -= np.median(x, axis=1, keepdims=True)
    return x


_SOS = {"ap": butter(3, [300, 6000], btype="band", fs=FS, output="sos"),
        "hf": butter(3, [300, 1000], btype="band", fs=FS, output="sos"),
        "lf": butter(3, [2, 200], btype="band", fs=FS / 10, output="sos")}


def channel_features(x: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    """Per-channel features of one window (see module docstring)."""
    ap = sosfiltfilt(_SOS["ap"], x, axis=0)
    hf = sosfiltfilt(_SOS["hf"], x, axis=0)
    lf = sosfiltfilt(_SOS["lf"], x[::10], axis=0)
    secs = x.shape[0] / FS
    mad = np.median(np.abs(ap), axis=0) / 0.6745
    below = ap < -4.5 * mad[None, :]
    xing = below.sum(axis=0) / secs
    # common-mode crossings: a sample where >= COINC_MIN channels of the same half cross together
    coinc = np.zeros(N_CH)
    for half in (slice(0, N_CH // 2), slice(N_CH // 2, N_CH)):
        n_cross = below[:, half].sum(axis=1)
        common = n_cross >= COINC_MIN
        common = common | np.roll(common, 1) | np.roll(common, -1)
        hits = below[:, half] & common[:, None]
        coinc[half] = hits.sum(axis=0) / np.maximum(below[:, half].sum(axis=0), 1)
    up2 = np.minimum(np.arange(N_CH) + 4, N_CH - 1)                  # two rows up = +4 channels
    def corr(a):
        a = a - a.mean(axis=0); s = a.std(axis=0) + 1e-9
        return (a * a[:, up2]).mean(axis=0) / (s * s[up2])
    return pd.DataFrame(dict(channel=np.arange(N_CH), row=rows, y_um=rows * ROW_UM,
                             rms_ap=ap.std(axis=0), xing=xing, coincident_frac=coinc,
                             rms_lf=lf.std(axis=0), corr_lf=corr(lf), corr_hf=corr(hf)))


def row_profile(feat: pd.DataFrame) -> pd.DataFrame:
    g = feat.groupby("row").agg(y_um=("y_um", "first"), rms_ap=("rms_ap", "median"), xing=("xing", "median"),
                               coincident_frac=("coincident_frac", "median"), rms_lf=("rms_lf", "median"),
                               corr_lf=("corr_lf", "median"), corr_hf=("corr_hf", "median")).reset_index()
    s_corr_lf = g.corr_lf >= CORR_LF_FLUID
    s_corr_hf = g.corr_hf >= CORR_HF_FLUID
    s_common = (g.coincident_frac >= 0.5) | (g.xing < 0.5)
    g["n_signatures"] = s_corr_lf.astype(int) + s_corr_hf.astype(int) + s_common.astype(int)
    g["fluid"] = s_corr_lf & s_corr_hf & s_common
    g["tissue_like"] = ~g.fluid & (g.xing >= 5) & (g.coincident_frac < 0.5)     # local spiking
    return g


def _longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def detect_surface(prof: pd.DataFrame) -> dict:
    """Lowest row r such that rows r..r+MIN_ROWS-1 are all fluid (a solid base: the boundary itself
    must be sharp, otherwise a >= FLUID_FRAC rule alone walks 25 % below a planted surface -- the
    gate caught that), >= FLUID_FRAC of rows r..top are fluid, and no run of more than
    TISSUE_RUN_MAX tissue-like rows lies above r (rows above the fluid level are allowed)."""
    fluid = prof.fluid.to_numpy()
    tissue = prof.tissue_like.to_numpy()
    n = len(fluid)
    best = None
    for r in range(n - MIN_ROWS, -1, -1):
        if fluid[r:r + MIN_ROWS].all() and fluid[r:].mean() >= FLUID_FRAC and _longest_true_run(tissue[r:]) <= TISSUE_RUN_MAX:
            best = r
        elif best is not None and not fluid[r:r + MIN_ROWS].any():
            break
    if best is None:
        return dict(surface_row=np.nan, surface_y_um=np.nan, n_rows_out=0, depth_tip_um=np.nan, confidence=np.nan, inside=True)
    seg = prof.iloc[best:]
    return dict(surface_row=int(best), surface_y_um=float(prof.y_um.iloc[best]), n_rows_out=int(n - best),
                depth_tip_um=float(prof.y_um.iloc[best] + TIP_OFFSET_UM),
                confidence=float(seg.n_signatures.mean() / 3.0), inside=False)


def summarise(prof: pd.DataFrame, det: dict) -> dict:
    top = prof[prof.y_um >= 3400]; mid = prof[(prof.y_um >= 1200) & (prof.y_um < 2800)]
    return dict(**det, top_rms_ap=float(top.rms_ap.median()), mid_rms_ap=float(mid.rms_ap.median()),
                top_xing=float(top.xing.median()), mid_xing=float(mid.xing.median()),
                top_corr_lf=float(top.corr_lf.median()), mid_corr_lf=float(mid.corr_lf.median()),
                top_corr_hf=float(top.corr_hf.median()), mid_corr_hf=float(mid.corr_hf.median()),
                top_coincident=float(top.coincident_frac.median()), mid_coincident=float(mid.coincident_frac.median()),
                n_rows_fluid_total=int(prof.fluid.sum()))


def run_gate(mouse: str, secs: float) -> bool:
    """Plant a fluid segment on a real window; the detector must find it, and must not find one
    in the unmodified window when that window is inside."""
    rows = channel_rows()
    recs = task_recordings(mouse)
    x = read_window(recs.continuous_path.iloc[len(recs) // 2], secs)
    base = detect_surface(row_profile(channel_features(x, rows)))
    print(f"gate base window ({mouse}): {'inside' if base['inside'] else f'surface row {base['surface_row']}'}")
    planted_row = 132                                                   # top 60 rows = 120 channels
    rng = np.random.default_rng(0)
    xs = x.copy()
    top = rows >= planted_row
    shared = rng.normal(0, 1, size=(x.shape[0], 1)).astype(np.float32)
    shared = sosfiltfilt(butter(2, 300, btype="low", fs=FS, output="sos"), shared, axis=0)
    shared = shared / shared.std() * float(np.median(x[:, ~top].std(axis=0)))
    xs[:, top] = 1.5 * shared + rng.normal(0, 0.1 * shared.std(), size=(x.shape[0], top.sum())).astype(np.float32)
    det = detect_surface(row_profile(channel_features(xs, rows)))
    ok_plant = (not det["inside"]) and abs(det["surface_row"] - planted_row) <= 2
    print(f"gate planted surface at row {planted_row}: detected {det['surface_row']} ({det['n_rows_out']} rows out) -> {'PASS' if ok_plant else 'FAIL'}")
    # negative: the bank below the planted surface, treated as a whole probe, must be inside
    neg = row_profile(channel_features(x, rows)); neg = neg[neg.row < planted_row].reset_index(drop=True)
    det_neg = detect_surface(neg)
    print(f"gate unmodified rows 0-{planted_row - 1}: {'inside' if det_neg['inside'] else 'surface at row ' + str(det_neg['surface_row'])} -> {'PASS' if det_neg['inside'] else 'FAIL'}")
    return ok_plant and det_neg["inside"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--mouse", nargs="*", default=None)
    ap.add_argument("--secs", type=float, default=SECS)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    mice = args.mouse or list(rr.EPHYS_MICE)
    if args.gate:
        ok = run_gate(mice[0], args.secs)
        print("GATE", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    rows = channel_rows()
    summary = []
    for mouse in mice:
        recs = task_recordings(mouse)
        print(f"== {mouse}: {len(recs)} task recordings")
        profiles = {}
        for _, r in recs.iterrows():
            if not isinstance(r.continuous_path, str) or not Path(r.continuous_path).exists():
                print(f"  {r.recording}: no continuous.dat"); continue
            feat = channel_features(read_window(r.continuous_path, args.secs), rows)
            prof = row_profile(feat)
            det = detect_surface(prof)
            row = dict(mouse=mouse, recording=r.recording, session=r.get("session"), recday=r.get("recday_seen"),
                       dur_s=round(float(r.dur_s), 1), secs=args.secs, **summarise(prof, det))
            summary.append(row)
            profiles[r.recording] = prof
            print(f"  {r.recording}: {'inside' if det['inside'] else f'surface y={det['surface_y_um']:.0f} um ({det['n_rows_out']} rows out, conf {det['confidence']:.2f})'}"
                  f" | top/mid corr_lf {row['top_corr_lf']:.2f}/{row['mid_corr_lf']:.2f} corr_hf {row['top_corr_hf']:.2f}/{row['mid_corr_hf']:.2f} coincident {row['top_coincident']:.2f}/{row['mid_coincident']:.2f}")
            if args.write:
                d = OUT / "probe_surface" / mouse
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"{r.recording}.csv"
                k = 0
                while (q := p.with_name(p.stem + (f".v{k}" if k else "") + ".csv")).exists():
                    k += 1
                prof.to_csv(q, index=False)
        if args.write and profiles:
            import matplotlib.pyplot as plt
            keys = list(profiles)
            fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
            for ax, col, vmax in zip(axes, ["corr_lf", "corr_hf", "coincident_frac", "xing"], [1, 1, 1, None]):
                m = np.stack([profiles[k][col].to_numpy() for k in keys], axis=1)
                if col == "xing":
                    m = np.log10(m + 0.1)
                im = ax.imshow(m, aspect="auto", origin="lower", cmap="viridis", vmin=0 if vmax else None, vmax=vmax,
                               extent=[-0.5, len(keys) - 0.5, -0.5 * ROW_UM, (len(m) - 0.5) * ROW_UM])
                ax.set_title(col if col != "xing" else "log10 crossings/s", fontsize=9); ax.set_xlabel("recording (chronological)")
                plt.colorbar(im, ax=ax, fraction=0.04)
                for j, k in enumerate(keys):
                    s = [r for r in summary if r["recording"] == k][0]
                    if not s["inside"]:
                        ax.plot([j - 0.5, j + 0.5], [s["surface_y_um"]] * 2, "r-", lw=2)
            axes[0].set_ylabel("y from tip-most row (um)")
            fig.suptitle(f"{mouse}: per-row features along the probe, red = detected brain surface", fontsize=10)
            out = OUT / f"probe_surface_{mouse}.png"
            n = 0
            while (p := out.with_name(out.stem + (f".v{n}" if n else "") + ".png")).exists():
                n += 1
            fig.tight_layout(); fig.savefig(p, dpi=100); plt.close(fig)
            print(f"  figure -> {p}")
    df = pd.DataFrame(summary)
    if args.write and len(df):
        out = OUT / f"probe_surface_{rr.DATASET}.csv"
        n = 0
        while (p := out.with_name(out.stem + (f".v{n}" if n else "") + ".csv")).exists():
            n += 1
        df.to_csv(p, index=False)
        print(f"summary -> {p}")
    if len(df):
        print(df.groupby("mouse").agg(n=("recording", "size"), n_out=("inside", lambda s: int((~s).sum())),
                                      surface_y_median=("surface_y_um", "median"), surface_y_min=("surface_y_um", "min"),
                                      surface_y_max=("surface_y_um", "max")).to_string())


if __name__ == "__main__":
    main()
