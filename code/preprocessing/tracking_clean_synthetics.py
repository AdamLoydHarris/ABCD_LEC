"""
Synthetic controls for tracking cleaning v2 (`tracking_clean.py`) and the tracking QC.

| # | control | what a failure would mean |
|---|---|---|
| 1 | v2 positive: 50 planted 30 cm spikes removed and bridged to < 0.5 cm of truth | a rule threshold is wrong or the bridge is not applied |
| 2 | v2 positive: a 300-frame 40 cm/s run loses 0 frames | the physical thresholds are mobility-dependent after all (the v1 failure) |
| 3 | v2 positive: dropouts of 10 / 100 frames bridged whole, 200 frames left NaN whole, lengths exact | `interpolate_limited` fills part of a long gap (pandas `limit=` behaviour) |
| 4 | conditional gate: low-score frames ON the trajectory kept with flag 2/3; a 20-frame low-score run 6 cm OFF it removed (reason 7 / R1); a hard gate (`lowscore_thr_cm=0`) removes both | the gate is unconditional, or the chord test misses a sustained mislocalisation |
| 5 | rejection: v1 (median + 3 MAD) deletes > 50 % of the planted run; v1 removes every spike on a still-only track | the synthetic does not reproduce the v1 failure, so control 2 proves nothing |
| 6 | smoothing edge: NaN-run lengths identical before and after `smooth_remask`; a plain rolling mean shrinks them | the re-mask is off and gaps grow back by 2 frames a side |
| 7 | flags: every planted frame carries the expected code; `outcome`/`reason` round-trip | the provenance file lies |
| 8 | guards: an h5 without `point_scores` raises KeyError; mismatched score shapes raise ValueError | a silently score-less cleaning |
| 9 | the real door: the planted h5 (in LEC maze-1 pixels) through `sleap_to_rois.process_one` under `--clean v2` writes the four CSVs incl. `*_track_flags.csv`; a spike frame's ROI label equals the label of the true position, a bridged dropout frame carries a label, every frame of the 200-frame gap is `None`; the coordinates equal `clean_v2` applied directly (max diff 0) | the stage script labels ROIs from the raw track after all, or writes the wrong tree |

The synthetic enters the same way the data does: `write_synthetic_h5` writes a real SLEAP-style
analysis h5 (tracks, point_scores, instance_scores, node_names, track_occupancy, labels_path)
which `tracking_qc.py --gate` and the stage scripts read through `load_tracks` / `load_scores`.
Byte-identical mirror not required (preprocessing is LEC-tree only).

    python tracking_clean_synthetics.py --quick
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking_clean as tc                                           # noqa: E402

NODES = ["head_front", "head_mid", "head_back", "ear_L", "ear_R", "body_front", "body_mid", "body_back"]
FPS = 60.0
HB = NODES.index("head_back")

_FAILS: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    if not passed:
        _FAILS.append(name)


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

def make_track(T: int = 12000, seed: int = 0, side: float = 47.0, run_speed: float = 30.0,
               period: int = 1200, duty: float = 0.25, jitter_cm: float = 0.05,
               fast: tuple[int, int, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Random walk in cm: every `period` frames the mouse runs (run_speed) for the first `duty`
    fraction and is still for the rest, like a maze mouse (moving ~20-25 % of frames, which is
    what collapses the v1 MAD threshold). `fast=(start, end, speed)` inserts a straight run at
    that speed, continuous with the walk. Jitter 0.05 cm ~ 0.9 px, the measured SLEAP jitter
    (v1 thresholds of 0.3-1.1 px/frame on still mice). Returns (xy (T,2), moving mask)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    moving = (t % period) < period * duty
    speed = np.where(moving, run_speed, 0.0)
    dang = rng.normal(0, 0.05, T)
    if fast is not None:
        s, e, v = fast
        speed[s:e] = v
        dang[s:e] = 0.0                       # straight
        moving[s:e] = True
    ang = np.cumsum(dang)
    x = np.cumsum(speed * np.cos(ang) / FPS) + side / 2
    y = np.cumsum(speed * np.sin(ang) / FPS) + side / 2
    # fold at the walls (triangle map into [2, side-2]) so the track stays in the arena; the
    # speed is preserved, the heading reverses at a wall
    for arr in (x, y):
        arr[:] = np.abs(((arr - 2) % (2 * (side - 4))) - (side - 4)) + 2
    return np.c_[x, y] + rng.normal(0, jitter_cm, (T, 2)), moving


def body_plan(xy: np.ndarray) -> dict[str, np.ndarray]:
    """Eight nodes from one head_back track: fixed offsets along +x (cm)."""
    return {bp: xy + np.array([[(k - HB) * 0.7, 0.0]]) for k, bp in enumerate(NODES)}


def planted_session(seed: int = 0, plant_low_off: bool = True) -> dict:
    """The standard planted session: truth, raw (with plants), scores, plant indices.

    Frames 4800-5100 are a straight 40 cm/s run (the v1 failure case), continuous with the walk.
    Moving bouts are frames 0-299 of every 1200 (period 1200, duty 0.25). Low-score plants:
    on-trajectory runs in a moving bout (6000-6050) and a still bout (6500-6550), instance-low
    on trajectory (7250-7260); off-trajectory plants (`plant_low_off`): a STATIC 20-frame run
    15 cm off the path while the mouse runs (7200-7220) and a static 20-frame run 6 cm off
    while the mouse is still (6700-6720). `plant_low_off=False` leaves those out so that every
    R1-R3 removal is a planted spike (the QC gate counts them exactly)."""
    rng = np.random.default_rng(seed + 1)
    truth, moving = make_track(seed=seed, fast=(4800, 5100, 40.0))
    T = len(truth)
    nodes = body_plan(truth)
    raw = {bp: v.copy() for bp, v in nodes.items()}
    ps = np.full((len(NODES), T), 0.9)
    isc = np.full(T, 0.9)
    plants = dict(T=T, moving=moving)
    # 50 single-frame spikes 30 cm away on head_back, clear of the other plants (both bouts)
    clear = np.r_[np.arange(10, 900), np.arange(1300, 1900), np.arange(2200, 2900), np.arange(3300, 4700),
                  np.arange(8400, 9500), np.arange(9700, 10700)]
    plants["spikes"] = np.sort(rng.choice(clear[::4], 50, replace=False))     # never adjacent
    raw["head_back"][plants["spikes"], 0] += 30.0
    # dropouts: 10, 100, 200 frames (all nodes)
    plants["dropouts"] = [(1000, 10), (2000, 100), (3000, 200)]
    for s, L in plants["dropouts"]:
        for bp in NODES:
            raw[bp][s:s + L] = np.nan
    plants["run"] = np.arange(4800, 5100)
    plants["low_ok"] = np.arange(6000, 6050)          # moving bout (6000-6299)
    plants["low_ok_still"] = np.arange(6500, 6550)    # still bout
    ps[HB, plants["low_ok"]] = 0.2
    ps[HB, plants["low_ok_still"]] = 0.2
    plants["inst_low"] = np.arange(7250, 7260)        # moving bout (7200-7499)
    isc[plants["inst_low"]] = 0.1
    plants["low_off"] = np.arange(7200, 7220)         # static 15 cm off while the mouse runs
    plants["low_off_still"] = np.arange(6700, 6720)   # static 6 cm off while the mouse is still
    if plant_low_off:
        ps[HB, plants["low_off"]] = 0.2
        raw["head_back"][plants["low_off"]] = truth[7200] + np.array([0.0, 15.0])
        ps[HB, plants["low_off_still"]] = 0.2
        raw["head_back"][plants["low_off_still"]] = truth[6700] + np.array([0.0, 6.0])
    plants["truth"] = truth
    plants["raw"] = raw
    plants["point_scores"] = ps
    plants["instance_scores"] = isc
    return plants


def raw_dataframe(raw: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({f"{bp}.{c}": raw[bp][:, i] for bp in NODES for i, c in enumerate("xy")})


def write_synthetic_h5(path: Path, raw: dict[str, np.ndarray], point_scores: np.ndarray,
                       instance_scores: np.ndarray, px_per_cm: float = 1.0, drop_scores: bool = False) -> Path:
    """A SLEAP-style single-instance analysis h5: tracks (1, 2, n_nodes, T) in pixels, scores,
    node_names, track_occupancy, labels_path. `px_per_cm` scales cm -> px."""
    T = len(instance_scores)
    tracks = np.zeros((1, 2, len(NODES), T))
    for k, bp in enumerate(NODES):
        tracks[0, 0, k] = raw[bp][:, 0] * px_per_cm
        tracks[0, 1, k] = raw[bp][:, 1] * px_per_cm
    with h5py.File(path, "w") as f:
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("node_names", data=np.array([n.encode() for n in NODES], dtype="S10"))
        f.create_dataset("track_occupancy", data=(~np.isnan(tracks[0, 0, HB])).astype(np.uint8)[:, None])
        f.create_dataset("labels_path", data="K:/SLEAP/models/synthetic/labels.synthetic.slp")
        if not drop_scores:
            f.create_dataset("point_scores", data=point_scores[None])
            f.create_dataset("instance_scores", data=instance_scores[None])
            f.create_dataset("tracking_scores", data=instance_scores[None])
    return path


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def quick() -> int:
    p = planted_session()
    raw_df = raw_dataframe(p["raw"])
    scores = dict(point_scores=p["point_scores"], instance_scores=p["instance_scores"])
    ident = lambda a: a                                                   # cm in, cm out
    cm, flags, rep = tc.clean_v2(raw_df, NODES, scores, ident)
    hb = cm[["head_back.x", "head_back.y"]].to_numpy()
    f = flags["head_back"].to_numpy()
    out, why = tc.outcome(f), tc.reason(f)
    truth = p["truth"]

    # 1 spikes
    sp = p["spikes"]
    removed = np.isin(why[sp], [tc.R_VELOCITY, tc.R_SPIKE, tc.R_HAMPEL]).sum()
    err = np.linalg.norm(hb[sp] - truth[sp], axis=1)
    check("1 spikes removed", removed == 50, f"{removed}/50 removed by R1-R3")
    # 1 cm, not 0.5: the folded synthetic walk reverses heading within one frame at a wall, which
    # a straight bridge across that frame misses by ~0.5 cm at 30 cm/s; a real mouse does not
    check("1 spikes bridged close to truth", (out[sp] == tc.BRIDGED).all() and np.nanmax(err) < 1.0,
          f"bridged {(out[sp] == tc.BRIDGED).sum()}/50, max |bridge - truth| {np.nanmax(err):.2f} cm, p50 {np.nanmedian(err):.2f}")
    # 2 run intact
    run = p["run"]
    lost = int((out[run] != tc.KEPT).sum())
    check("2 40 cm/s run loses 0 frames under v2", lost == 0, f"{lost}/300 frames removed")
    # 3 dropouts
    for s, L in p["dropouts"]:
        o = out[s:s + L]
        if L <= tc.V2["max_gap_frames"]:
            check(f"3 dropout {L} f bridged whole", (o == tc.BRIDGED).all() and out[s - 1] == tc.KEPT and out[s + L] == tc.KEPT,
                  f"{(o == tc.BRIDGED).sum()}/{L} bridged")
        else:
            check(f"3 dropout {L} f left NaN whole", (o == tc.LEFT_NAN).all() and np.isnan(hb[s:s + L]).all()
                  and not np.isnan(hb[s - 1]).any() and not np.isnan(hb[s + L]).any(),
                  f"{(o == tc.LEFT_NAN).sum()}/{L} NaN, neighbours finite")
    # 4 conditional gate
    ok, oks, il = p["low_ok"], p["low_ok_still"], p["inst_low"]
    check("4 low-score on a curved running path kept + flag 2", (f[ok] == tc.R_LOW_POINT).all(), f"{(f[ok] == 2).sum()}/{len(ok)}")
    check("4 low-score while still kept + flag 2", (f[oks] == tc.R_LOW_POINT).all(), f"{(f[oks] == 2).sum()}/{len(oks)}")
    check("4 instance-low on trajectory kept + flag 3", (f[il] == tc.R_LOW_INSTANCE).all(), f"{(f[il] == 3).sum()}/{len(il)}")
    for name, off in (("static 15 cm off while running", p["low_off"]), ("static 6 cm off while still", p["low_off_still"])):
        n7, n4 = int((why[off] == tc.R_LOWSCORE_OFF).sum()), int((why[off] == tc.R_VELOCITY).sum())
        check(f"4 low-score run {name} removed", (out[off] != tc.KEPT).all() and n7 > 0,
              f"{n7} by the chord/path test, {n4} by R1, {(out[off] == tc.BRIDGED).sum()} bridged")
    hard = dict(tc.V2, lowscore_thr_cm=0.0, lowscore_chord_frac=0.0, lowscore_chord_frac_long=0.0)
    _, fh, _ = tc.clean_v2(raw_df, NODES, scores, ident, cfg=hard)
    fh = fh["head_back"].to_numpy()
    check("4 hard gate (thr 0) removes on-trajectory low-score frames too",
          (tc.outcome(fh[ok]) != tc.KEPT).all() and (tc.outcome(fh[p['low_off']]) != tc.KEPT).all(),
          f"{(tc.outcome(fh[ok]) != 0).sum()}/{len(ok)} and {(tc.outcome(fh[p['low_off']]) != 0).sum()}/{len(p['low_off'])} removed")
    # 5 rejection: v1 on the same track
    from sleap_to_rois import drop_jumps
    v1 = drop_jumps(raw_df.copy(), NODES, multiplier=3.0)
    v1_lost_run = int(np.isnan(v1["head_back.x"].to_numpy()[run]).sum())
    check("5 v1 deletes > 50 % of the 40 cm/s run", v1_lost_run > 150, f"{v1_lost_run}/300 frames NaN'd by median + 3 MAD")
    still_truth, _ = make_track(T=6000, seed=3, run_speed=0.0)
    still_raw = body_plan(still_truth)
    sp2 = np.arange(100, 5900, 200)
    still_raw["head_back"][sp2, 0] += 30.0
    v1s = drop_jumps(raw_dataframe(still_raw), NODES, multiplier=3.0)
    got = int(np.isnan(v1s["head_back.x"].to_numpy()[sp2]).sum())
    check("5 v1 removes every spike on a still-only track", got == len(sp2), f"{got}/{len(sp2)}")
    # 6 smoothing edge
    xy_gap = truth.copy()
    xy_gap[500:530] = np.nan
    xy_gap[800:801] = np.nan
    sm = tc.smooth_remask(xy_gap, 5)
    plain = pd.DataFrame(xy_gap).rolling(5, center=True, min_periods=1).mean().to_numpy()
    same = tc.describe_gaps(np.isnan(sm).any(1)) == tc.describe_gaps(np.isnan(xy_gap).any(1))
    shrunk = np.isnan(plain).any(1).sum() < np.isnan(xy_gap).any(1).sum()
    check("6 smooth_remask preserves NaN-run lengths", same, f"runs {tc.describe_gaps(np.isnan(sm).any(1))}")
    check("6 plain rolling mean would shrink the gaps (the re-mask is doing work)", shrunk,
          f"{np.isnan(plain).any(1).sum()} vs {np.isnan(xy_gap).any(1).sum()} NaN frames")
    # 7 flags round trip and clean nodes untouched
    check("7 outcome/reason round-trip", np.array_equal(tc.outcome(f) * 10 + tc.reason(f), f), "")
    check("7 unplanted node (body_back) has 0 rule removals", rep["body_back"]["removed_by_rules"] == 0,
          f"{rep['body_back']['removed_by_rules']} removed by rules, {rep['body_back']['raw_nan']} raw NaN (the dropouts)")
    # where did any extra head_back removal come from?
    planted = np.zeros(p["T"], bool)
    planted[sp] = True
    for s, L in p["dropouts"]:
        planted[s:s + L] = True
    for k in ("low_off", "low_off_still"):
        planted[p[k]] = True
    extra = np.flatnonzero((out != tc.KEPT) & ~planted)
    check("7 no unplanted head_back frame removed", len(extra) == 0,
          f"{len(extra)} extra: frames {extra[:8].tolist()} reasons {why[extra][:8].tolist()}")
    # 8 guards
    with tempfile.TemporaryDirectory() as td:
        h5 = write_synthetic_h5(Path(td) / "noscores.analysis.h5", p["raw"], p["point_scores"], p["instance_scores"], drop_scores=True)
        try:
            tc.load_scores(h5)
            check("8 missing point_scores raises", False, "no exception")
        except KeyError as exc:
            check("8 missing point_scores raises", True, str(exc)[:60])
        h5 = write_synthetic_h5(Path(td) / "ok.analysis.h5", p["raw"], p["point_scores"], p["instance_scores"])
        sc = tc.load_scores(h5)
        check("8 scores read back", sc["point_scores"].shape == (8, p["T"]) and sc["instance_scores"].shape == (p["T"],),
              f"{sc['point_scores'].shape}, {sc['instance_scores'].shape}")
    try:
        tc.clean_v2(raw_df, NODES, dict(point_scores=p["point_scores"][:, :100], instance_scores=p["instance_scores"]), ident)
        check("8 mismatched score shape raises", False, "no exception")
    except ValueError as exc:
        check("8 mismatched score shape raises", True, str(exc)[:60])

    # 9 through the stage script's own door, in LEC maze-1 pixels against the real masks
    import sleap_to_rois as s2r
    if s2r.rr.DATASET != "lec":
        check("9 process_one door (needs --dataset lec masks)", False, f"registry is on {s2r.rr.DATASET}")
    else:
        A, b = s2r.affine(s2r.LEC_MAZE1, s2r.PHYSICAL_PTS)
        Ainv = np.linalg.inv(A)
        raw_px = {bp: (Ainv @ (v - b).T).T for bp, v in p["raw"].items()}
        with tempfile.TemporaryDirectory() as td:
            h5 = write_synthetic_h5(Path(td) / "syn_2000-01-01-000000.analysis.h5", raw_px, p["point_scores"], p["instance_scores"])
            keep = (s2r.CLEAN_VERSION, s2r.GEOMETRY_VERSION)
            s2r.CLEAN_VERSION, s2r.GEOMETRY_VERSION = "v2", "v2"
            try:
                out_dir = Path(td) / "maze_v2" / "syn"
                row = s2r.process_one(h5, "syn_2000-01-01-000000", s2r.MASKS_ROOT / "lEC_maze1", s2r.LEC_MAZE1, out_dir, write=True)
            finally:
                s2r.CLEAN_VERSION, s2r.GEOMETRY_VERSION = keep
            files = sorted(q.name.split("analysis_")[1] for q in out_dir.glob("*.csv"))
            check("9 process_one writes 4 files incl. flags", files == ["ROIs.csv", "cleaned_coordinates.csv", "head_direction.csv", "track_flags.csv"], str(files))
            cc = pd.read_csv(out_dir / "syn_2000-01-01-000000.analysis_cleaned_coordinates.csv")
            fl = pd.read_csv(out_dir / "syn_2000-01-01-000000.analysis_track_flags.csv")
            ro = pd.read_csv(out_dir / "syn_2000-01-01-000000.analysis_ROIs.csv").fillna("None")
            # coordinates equal clean_v2 applied directly to the cm track (the affine is exact)
            d = np.nanmax(np.abs(cc[["head_back.x", "head_back.y"]].to_numpy() - hb))
            check("9 coordinates == direct clean_v2 (max |diff|)", d < 1e-9, f"{d:.2e} cm")
            check("9 flags file == direct flags", np.array_equal(fl["head_back"].to_numpy(), f), "")
            # ROI of a spike frame = ROI of the true position; gap frames None; bridged frames labelled
            labels, names, _ = s2r.label_image(s2r.MASKS_ROOT / "lEC_maze1")
            truth_px = (Ainv @ (truth - b).T).T
            truth_lab = s2r.roi_labels(pd.DataFrame({"head_back.x": truth_px[:, 0], "head_back.y": truth_px[:, 1]}), ["head_back"], labels, names)["head_back"].to_numpy()
            raw_lab = s2r.roi_labels(pd.DataFrame({"head_back.x": raw_px["head_back"][:, 0], "head_back.y": raw_px["head_back"][:, 1]}), ["head_back"], labels, names)["head_back"].to_numpy()
            got = ro["head_back"].astype(str).to_numpy()
            agree_truth = float(np.mean(got[sp] == truth_lab[sp]))
            agree_raw = float(np.mean(got[sp] == raw_lab[sp]))
            check("9 spike frames labelled from the cleaned (true) position", agree_truth >= 0.9 and agree_raw < 0.5,
                  f"agree with truth {agree_truth:.2f}, with the raw spike {agree_raw:.2f} ({len(sp)} spikes)")
            g_s, g_L = p["dropouts"][-1]
            check("9 frames of the 200-frame gap are 'None'", (got[g_s:g_s + g_L] == "None").all(), f"{(got[g_s:g_s + g_L] == 'None').sum()}/{g_L}")
            b_s, b_L = p["dropouts"][1]
            check("9 bridged dropout frames carry a label", (got[b_s:b_s + b_L] != "None").mean() > 0.9, f"{(got[b_s:b_s + b_L] != 'None').mean():.2f} labelled")
            # 50 spikes + 110 dropout frames + the two static low-score runs (20 + 20) are bridged
            check("9 QC row carries v2 counts", row.get("clean_version") == "v2" and row.get("v2_bridged") == 200,
                  f"clean_version={row.get('clean_version')} v2_bridged={row.get('v2_bridged')} (expected 200)")

    print(f"\n{len(_FAILS)} failed" + (": " + ", ".join(_FAILS) if _FAILS else ""))
    return 1 if _FAILS else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="run the controls (the only mode)")
    ap.parse_args()
    sys.exit(quick())
