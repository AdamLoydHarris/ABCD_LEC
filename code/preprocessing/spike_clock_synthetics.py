"""
Synthetic controls for the v4 spike clock: the TTL origin in `extract_rsync_from_ttl` and the
behaviour probes in `validate_alignment` (docs/handoff/SPIKE_CLOCK_V4.md).

| # | control | what a failure would mean |
|---|---|---|
| 1 | `counter_origin` on planted sample numbers: 728 stale samples then 8.. (origin -720); 1480 stale then 732.. (-748); a count from 0 (0); a continuous count from 231,685 (231,685); the jump on a chunk seam | the rule misreads one of the three recording classes, or loses a jump at a seam |
| 2 | guards: a jump 40,000 samples in, an empty file and an unknown origin mode all raise | a mid-recording discontinuity silently gets an origin |
| 3 | the real door: an OpenEphys-shaped folder per class (TTL on the restarted count, a partner box's pulses, -1.6e-5 drift, 1-sample jitter) through `process_row(row, pc_ms, 'counter')` and `extract_neuron_raw.bin_session_fitted`: every planted spike lands in its planted bin | the clock is wrong between the TTL file and the binned array |
| 4 | rejection that also accepts: under `first_sample` the restart class's spikes land late by exactly (sn0 - origin) / a ms, and the other two classes give arrays identical to `counter`; `first_sample` equals v2's `rising - sn[0]` | the synthetic does not reproduce v2's failure, or `first_sample` is wrong where it is right |
| 5 | the handoff's fix (raw TTL, origin 0) leaves 720 samples (24 ms) on the restart class and is 7.7 s early on the continuous class | neither simple rule is enough; documents why the counter rule exists |
| 6 | QC columns: under `counter`, `ttl_outside_dat` and `ttl_in_stale_window` are 0 in every class; under `first_sample` the restart class has TTL edges before index 0 | the QC could not flag a wrong origin |
| 7 | speed probe: a planted -0.30 s lead and a +0.35 s lag are recovered to one bin with a distinct peak (ratio >= 1.2); 19.31 s and 1 s spike delays move the lag by that much and fail the ±0.5 s gate; Poisson noise has r < 0.2; units sharing only a slow session-long trend with speed have r > 0.2 but a flat peak (ratio < 1.2), so they are untested | the probe cannot see the error it gates, or tests a session whose cross-correlation has no lag in it (ly05: 12 such sessions failed with lags scattered over ±70 s) |
| 8 | reward probe: a planted +0.25 s response peaks inside [-2, +4] s; a 19.31 s delay moves it outside; noise has peak/median < 2 | as 7 |
| 9 | wall clock: a 1 s `.txt` stamp (BST) against a UTC-ms Software Time gives (-1, 0] s on a right clock and +19.31 s on v2's; a `.tsv` stamp is read as UTC on both sides of the 2024-03-31 DST change (as London time it read -3599.4 s in April); `.txt` local time on both sides | a timezone or resolution error would read as a clock error |
| 10 | `probe_gates` end to end: clean sessions of two mice (references -0.30 s and +0.35 s) pass; sessions delayed 2.0-29.2 s fail speed, the large ones fail reward, wall clock and early start | the per-mouse gates do not separate delayed sessions, or the reference is not per mouse |
| 10b | a mouse with 2 clean and 4 delayed sessions takes the cohort's clean reference, and its delayed sessions still fail; a second rig whose PC clock sits +3.5 s away passes on its own band and a 19.3 s error on that rig still fails | the reference falls back to the suspect sessions (on a 2-recday subset that put ah10's at +4.55 s) |
| 11 | `spike_clock_v4_gates.pop_lag`: an array against itself delayed 772.49 bins gives lag 772-773 with r >= 0.8; an array against itself with 772.49 expected gives lag 0 and fails | gate 5 passes on a local maximum (it did, 1 of 7, with a ±100-bin window) |

The synthetic enters the way the data does: real `.npy` files in an OpenEphys folder layout,
read by `process_row` exactly as `process_recday` calls it. Not mirrored (preprocessing is
LEC-tree only).

    python spike_clock_synthetics.py --quick
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_neuron_raw as enr                                      # noqa: E402
import extract_rsync_from_ttl as ert                                  # noqa: E402
import validate_alignment as va                                       # noqa: E402
import spike_clock_v4_gates as gates                                  # noqa: E402

SPMS = 30                     # samples per ms
SLOPE = 1 - 1.6e-5            # ephys ms per pyControl ms, the measured LEC drift
B_MS = 3_000.0                # pyControl t = 0 sits 3 s into the .dat
N_SAMPLES = 4_500_000         # 150 s; ~110 rsync pulses
T0_MS = 10_000                # first A_on, on the 25 ms grid
N_BINS = 5_000                # 125 s from T0, wide enough to hold a 19.3 s shift

_FAILS: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    if not passed:
        _FAILS.append(name)


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

def rsync_train(seed: int, t_end_ms: float, t_start_ms: int = 5_000) -> np.ndarray:
    """pyControl rsync times in ms, intervals uniform 100-1900 ms."""
    rng = np.random.default_rng(seed)
    t = [t_start_ms]
    while t[-1] < t_end_ms:
        t.append(t[-1] + int(rng.integers(100, 1901)))
    return np.array(t[:-1], dtype=np.int64)


def to_dat(pc_ms, b_ms: float = B_MS) -> np.ndarray:
    """True `.dat` index of pyControl times on the planted clock."""
    return np.rint((np.asarray(pc_ms, float) * SLOPE + b_ms) * SPMS).astype(np.int64)


def sample_numbers(n: int, sn0: int, restart_at: int, new_first: int) -> np.ndarray:
    """OpenEphys continuous sample numbers: `restart_at` stale samples from sn0, then new_first..."""
    sn = np.arange(n, dtype=np.int64) - restart_at + new_first
    sn[:restart_at] = sn0 + np.arange(restart_at)
    return sn


def write_recording(root: Path, sn: np.ndarray, origin: int, pulses_dat: np.ndarray,
                    seed: int) -> pd.Series:
    """OpenEphys-shaped recording under `root`; the concat-manifest row that points at it."""
    rec = root / "Record Node 101" / "experiment1" / "recording1"
    cont = rec / "continuous" / "Neuropix-PXI-100.ProbeB"
    ttl = rec / "events" / "Neuropix-PXI-100.ProbeB" / "TTL"
    cont.mkdir(parents=True)
    ttl.mkdir(parents=True)
    np.save(cont / "sample_numbers.npy", sn)
    jitter = np.random.default_rng(seed).integers(-1, 2, len(pulses_dat))
    rise = pulses_dat + jitter + origin                     # TTL carries the (restarted) count
    edges = np.concatenate([rise, rise + 1500])             # 50 ms pulses
    states = np.concatenate([np.ones(len(rise), np.int16), -np.ones(len(rise), np.int16)])
    order = np.argsort(edges, kind="stable")
    np.save(ttl / "sample_numbers.npy", edges[order])
    np.save(ttl / "states.npy", states[order])
    return pd.Series({"ttl_path": str(ttl / "sample_numbers.npy"),
                      "sample_numbers_path": str(cont / "sample_numbers.npy")})


CLASSES = {   # name: (sn0, restart_at, new_first) -> origin = new_first - restart_at
    "restart": (578_636, 728, 8),
    "zero": (0, 0, 0),
    "continuous": (231_685, 0, 231_685),
}


def planted_spikes(seed: int, n_units: int = 3, per_unit: int = 400):
    """(dat indices, unit rows, planted bins): spikes mid-bin at pyControl T0 .. T0 + 100 s."""
    rng = np.random.default_rng(seed)
    bins = rng.integers(0, 4_000, (n_units, per_unit))
    t_ms = T0_MS + bins * 25 + 12.5
    units = np.repeat(np.arange(n_units), per_unit)
    k = to_dat(t_ms.ravel())
    order = np.argsort(k, kind="stable")
    return k[order], units[order], bins.ravel()[order]


def clock_of(fit: dict) -> dict:
    """`extract_neuron_raw.fitted_clock`'s parse of a fit dict."""
    return dict(a=float(fit["a_samples_per_ms"]), b=float(fit["b_samples"]))


def spike_ms(k: np.ndarray, fit: dict) -> np.ndarray:
    c = clock_of(fit)
    return (k.astype(np.float64) - c["b"]) / c["a"]


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def c1_c2_origin(tmp: Path) -> None:
    cases = [("728 stale then 8..", (578_636, 728, 8), -720),
             ("1480 stale then 732..", (663_132, 1480, 732), -748),
             ("count from 0", (0, 0, 0), 0),
             ("continuous from 231,685", (231_685, 0, 231_685), 231_685)]
    for name, (sn0, ra, nf), want in cases:
        p = tmp / f"sn_{ra}_{nf}.npy"
        np.save(p, sample_numbers(200_000, sn0, ra, nf))
        got = ert.counter_origin(p)
        check(f"1 origin: {name}", got["origin"] == want and got["restart_at"] == ra,
              f"origin {got['origin']} (want {want}), restart_at {got['restart_at']}, jumps {got['n_jumps']}")
    p = tmp / "sn_728_8.npy"
    seam = [ert.counter_origin(p, chunk=c)["origin"] for c in (727, 728, 729, 1000)]
    check("1 origin: jump on a chunk seam", seam == [-720] * 4, f"chunks 727/728/729/1000 -> {seam}")

    late = tmp / "sn_late.npy"
    np.save(late, sample_numbers(200_000, 0, 40_000, 5))
    for name, fn in (("jump 40,000 samples in", lambda: ert.counter_origin(late)),
                     ("empty sample_numbers", lambda: (np.save(tmp / "empty.npy", np.array([], np.int64)),
                                                      ert.counter_origin(tmp / "empty.npy"))),
                     ("unknown origin mode", lambda: ert.ttl_rising_relative(
                         pd.Series({"ttl_path": "x", "sample_numbers_path": "y"}), "raw"))):
        try:
            fn()
            check(f"2 guard: {name}", False, "did not raise")
        except ValueError as exc:
            check(f"2 guard: {name}", True, f"ValueError: {str(exc)[:70]}")


def c3_to_c6_door(tmp: Path) -> None:
    pc = rsync_train(1, t_end_ms=140_000)
    partner = rsync_train(2, t_end_ms=140_000)
    k, units, want_bins = planted_spikes(3)
    want = np.zeros((3, N_BINS), np.int64)
    np.add.at(want, (units, want_bins), 1)
    results = {}
    for cls, (sn0, ra, nf) in CLASSES.items():
        origin = nf - ra
        sn = sample_numbers(N_SAMPLES, sn0, ra, nf)
        pulses = np.sort(np.concatenate([to_dat(pc), to_dat(partner, B_MS + 37.0)]))
        row = write_recording(tmp / cls, sn, origin, pulses, seed=4)
        for mode in ert.ORIGIN_MODES:
            rec, rel, fitted, fit = ert.process_row(row, pc, mode)
            arr = enr.bin_session_fitted(k, units, 3, clock_of(fit), T0_MS, N_BINS)
            results[cls, mode] = (rec, rel, fitted, fit, arr)
        rec, _, _, fit, arr = results[cls, "counter"]
        check(f"3 door counter, {cls}: every planted spike in its bin",
              np.array_equal(arr.astype(np.int64), want),
              f"{int((arr == want).all(axis=0).sum())}/{N_BINS} bins identical; matched "
              f"{rec['n_matched']}/{len(pc)}, residual {rec['residual_rms_ms']:.3f} ms, "
              f"b {fit['b_samples']:.1f} vs planted {B_MS * SPMS:.1f}")
        check(f"6 QC counter, {cls}: no TTL outside the .dat or in a stale window",
              rec["ttl_outside_dat"] == 0 and rec["ttl_in_stale_window"] == 0 and rec["origin"] == origin,
              f"origin {rec['origin']}, outside {rec['ttl_outside_dat']}, stale {rec['ttl_in_stale_window']}")

    # 4. first_sample: v2's formula; the bug in the restart class, identical elsewhere
    for cls, (sn0, ra, nf) in CLASSES.items():
        origin = nf - ra
        rec_c, rel_c, _, fit_c, arr_c = results[cls, "counter"]
        rec_f, rel_f, _, fit_f, arr_f = results[cls, "first_sample"]
        ttl = np.load(Path(rec_path := tmp / cls / "Record Node 101" / "experiment1" / "recording1" /
                           "events" / "Neuropix-PXI-100.ProbeB" / "TTL") / "sample_numbers.npy")
        st = np.load(rec_path / "states.npy")
        check(f"4 first_sample == v2's rising - sn[0], {cls}",
              np.array_equal(rel_f, ttl[st > 0] - sn0), f"sn0 {sn0}")
        if cls == "restart":
            d_ms = spike_ms(k, fit_f) - spike_ms(k, fit_c)
            want_ms = (sn0 - origin) / fit_c["a_samples_per_ms"]
            check("4 first_sample, restart: spikes late by exactly (sn0 - origin) / a",
                  np.allclose(d_ms, want_ms, atol=1e-6)
                  and np.isclose(fit_f["a_samples_per_ms"], fit_c["a_samples_per_ms"], rtol=1e-12, atol=0),
                  f"late by {d_ms.min():.4f}..{d_ms.max():.4f} ms, want {want_ms:.4f} ms "
                  f"({(sn0 - origin) / (SPMS * 1000):.3f} s); b shift "
                  f"{fit_c['b_samples'] - fit_f['b_samples']:.6f} samples")
            check("6 QC first_sample, restart: TTL edges before index 0", rec_f["ttl_before_ap_start"] > 0,
                  f"{rec_f['ttl_before_ap_start']} edges")
        else:
            check(f"4 first_sample, {cls}: identical to counter (accepts where right)",
                  np.array_equal(arr_f, arr_c) and fit_f["b_samples"] == fit_c["b_samples"],
                  f"b {fit_f['b_samples']:.3f} vs {fit_c['b_samples']:.3f}")

    # 5. the handoff's proposed fix: subtract nothing
    for cls, (sn0, ra, nf) in CLASSES.items():
        origin = nf - ra
        ttl_dir = tmp / cls / "Record Node 101" / "experiment1" / "recording1" / "events" / "Neuropix-PXI-100.ProbeB" / "TTL"
        raw = np.load(ttl_dir / "sample_numbers.npy")[np.load(ttl_dir / "states.npy") > 0]
        m = ert.match_trains(raw / SPMS, pc.astype(float))
        fit_raw = dict(a_samples_per_ms=m["slope"] * SPMS, b_samples=m["intercept_ms"] * SPMS)
        err_ms = float(np.median(spike_ms(k, fit_raw) - spike_ms(k, results[cls, "counter"][3])))
        want_ms = -origin / results[cls, "counter"][3]["a_samples_per_ms"]
        check(f"5 raw TTL (handoff fix), {cls}: spikes off by -origin",
              abs(err_ms - want_ms) < 1e-3, f"{err_ms:+.3f} ms (predicted {want_ms:+.3f} ms)")


def make_speed_session(seed: int, lag_bins: int, n_bins: int = 40_000, n_units: int = 30):
    """(neuron_raw, xy_cm): units driven by speed[t + lag_bins] (negative lag = rate leads)."""
    rng = np.random.default_rng(seed)
    speed = np.zeros(n_bins)
    t = 0
    while t < n_bins:                                      # runs of 2-6 s, pauses of 2-10 s
        run, rest = int(rng.integers(80, 240)), int(rng.integers(80, 400))
        speed[t:t + run] = rng.uniform(20, 40)
        t += run + rest
    speed = np.convolve(speed, np.ones(8) / 8, mode="same")
    heading = np.cumsum(rng.normal(0, 0.1, n_bins))
    xy = np.c_[np.cumsum(speed * 0.025 * np.cos(heading)), np.cumsum(speed * 0.025 * np.sin(heading))]
    drive = np.roll(speed, lag_bins)                       # rate at t follows speed at t - lag
    rate = 0.05 + 0.004 * drive[None, :] * rng.uniform(0.5, 1.5, (n_units, 1))
    return rng.poisson(rate).astype(np.uint16), xy


def delay(neural: np.ndarray, bins: int) -> np.ndarray:
    """Spikes `bins` later (the v2 failure), zero-filled at the start."""
    out = np.zeros_like(neural)
    out[:, bins:] = neural[:, :neural.shape[1] - bins]
    return out


def c7_speed() -> None:
    for name, lag_bins, want in (("rate leads by 0.30 s", -12, -0.30), ("rate lags by 0.35 s (ah08)", 14, 0.35)):
        neural, xy = make_speed_session(5, lag_bins)
        lag, r, ratio = va.speed_probe(neural, xy)
        check(f"7 speed: {name}", abs(lag - want) <= 0.025 + 1e-9 and r > va.SPEED_R_MIN
              and ratio >= va.SPEED_PEAK_RATIO_MIN, f"lag {lag:+.3f} s, r {r:.2f}, peak ratio {ratio:.2f}")
    neural, xy = make_speed_session(5, -12)
    for d_s in (19.31, 1.0):
        d = int(round(d_s * 1000 / 25))
        lag, r, ratio = va.speed_probe(delay(neural, d), xy)
        moved = abs(lag - (-0.30 + d * 0.025)) <= 0.05
        check(f"7 speed: {d_s} s delay seen and gated", moved and abs(lag - (-0.30)) > va.SPEED_TOL_S
              and ratio >= va.SPEED_PEAK_RATIO_MIN,
              f"lag {lag:+.3f} s (want {-0.30 + d * 0.025:+.3f}), r {r:.2f}, peak ratio {ratio:.2f}")
    rng = np.random.default_rng(6)
    lag, r, ratio = va.speed_probe(rng.poisson(0.1, (30, 40_000)).astype(np.uint16), xy)
    check("7 speed: Poisson noise untested (r < 0.2)", r < va.SPEED_R_MIN, f"r {r:.3f}, lag {lag:+.2f} s")
    # ly05's failure mode: speed and rate share only a slow trend (both fall over the session),
    # so r clears 0.2 while the cross-correlation has no peak and its argmax wanders
    _, xy0 = make_speed_session(15, 0)
    trend = np.linspace(1.6, 0.4, len(xy0))
    xy = np.cumsum(np.diff(xy0, axis=0, prepend=xy0[:1]) * trend[:, None], axis=0)
    neural = np.random.default_rng(15).poisson(0.05 * trend[None, :] * np.ones((30, 1))).astype(np.uint16)
    lag, r, ratio = va.speed_probe(neural, xy)
    flat = pd.DataFrame([dict(speed_r=r, speed_peak_ratio=ratio)])
    check("7 speed: a shared slow trend has r > 0.2 but no distinct peak, so untested",
          r > va.SPEED_R_MIN and ratio < va.SPEED_PEAK_RATIO_MIN and not bool(va._speed_testable(flat).iloc[0]),
          f"r {r:.2f}, peak ratio {ratio:.2f}, lag {lag:+.2f} s")


def make_reward_session(seed: int, n_bins: int = 40_000, n_units: int = 20, respond: bool = True):
    """(neuron_raw, pokes): rewarded entries every ~20 s, units peak 250 ms after entry."""
    rng = np.random.default_rng(seed)
    entries = np.cumsum(rng.integers(600, 1000, 60))
    entries = entries[entries < n_bins - 1300]
    rate = np.full((n_units, n_bins), 0.05)
    if respond:
        t = np.arange(n_bins)
        bump = np.zeros(n_bins)
        for e in entries:
            lo, hi = e, min(e + 40, n_bins)
            bump[lo:hi] += np.exp(-((t[lo:hi] - (e + 10)) ** 2) / (2 * 4.0 ** 2))
        rate += 0.3 * bump[None, :] * rng.uniform(0.5, 1.5, (n_units, 1))
    pokes = np.c_[entries, entries + 20, np.full(len(entries), 3), np.ones(len(entries)), np.zeros(len(entries))]
    return rng.poisson(rate).astype(np.uint16), pokes.astype(np.int64)


def c8_reward() -> None:
    neural, pokes = make_reward_session(7)
    peak, ptm, n = va.reward_probe(neural, pokes)
    check("8 reward: planted +0.25 s response inside [-1, +4] s",
          va.REWARD_WINDOW_S[0] <= peak <= va.REWARD_WINDOW_S[1] and ptm > va.REWARD_PTM_MIN,
          f"peak {peak:+.2f} s, peak/median {ptm:.1f}, {n} rewards")
    peak, ptm, n = va.reward_probe(delay(neural, 772), pokes)
    check("8 reward: a 19.31 s delay moves the peak outside", not (va.REWARD_WINDOW_S[0] <= peak <= va.REWARD_WINDOW_S[1]),
          f"peak {peak:+.2f} s, peak/median {ptm:.1f}")
    neural, pokes = make_reward_session(8, respond=False)
    peak, ptm, n = va.reward_probe(neural, pokes)
    check("8 reward: noise untested (peak/median < 2)", ptm < va.REWARD_PTM_MIN, f"peak/median {ptm:.2f}")


def c9_wallclock(tmp: Path) -> None:
    sync = tmp / "sync_messages.txt"
    sync.write_text("Software Time (milliseconds since midnight Jan 1st 1970 UTC): 1750433852867\n"
                    "Start Time for Neuropix-PXI (100) - ProbeB @ 30000 Hz: 578636\n")
    txt = tmp / "ah10-2025-06-20-163736.txt"
    txt.write_text("I Experiment name  : x\nI Start date : 2025/06/20 16:37:36\n\nS {}\n")
    true_start = datetime(2025, 6, 20, 16, 37, 36, 400_000, tzinfo=va.LONDON)
    pc_zero = (true_start - va.oe_start(sync)).total_seconds() * 30_000
    d = va.wallclock_discrepancy_s(va.pycontrol_start(txt), va.oe_start(sync), pc_zero)
    check("9 wall clock: right clock, 1 s stamp", -1.0 < d <= 0.0, f"{d:+.3f} s")
    d2 = va.wallclock_discrepancy_s(va.pycontrol_start(txt), va.oe_start(sync), pc_zero - 579_356)
    check("9 wall clock: v2's origin reads +19.31 s", abs(d2 - d - 19.312) < 1e-3, f"{d2:+.3f} s")
    tsv = tmp / "bp02-2024-03-26-180620.tsv"
    tsv.write_text("time\ttype\tsubtype\tcontent\n0.000\tinfo\tstart_time\t2024-03-26T18:06:20.863\n")
    got = va.pycontrol_start(tsv)
    check("9 wall clock: .tsv ms stamp read as UTC (March)", got == datetime(2024, 3, 26, 18, 6, 20, 863_000, tzinfo=timezone.utc),
          f"{got}")
    tsv4 = tmp / "bp01-2024-04-02-120000.tsv"
    tsv4.write_text("time\ttype\tsubtype\tcontent\n0.000\tinfo\tstart_time\t2024-04-02T12:00:00.000\n")
    got = va.pycontrol_start(tsv4)
    check("9 wall clock: .tsv stamp read as UTC after the DST change", got == datetime(2024, 4, 2, 12, 0, tzinfo=timezone.utc),
          f"{got} = {got.astimezone(va.LONDON).strftime('%H:%M %Z')} London")
    for day, offset_h in (("2024/03/30", 0), ("2024/04/01", 1)):
        p = tmp / f"dst_{day.replace('/', '')}.txt"
        p.write_text(f"I Start date : {day} 12:00:00\n")
        utc = datetime.strptime(day + " 12:00:00", "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
        d = (va.pycontrol_start(p) - utc).total_seconds() / 3600
        check(f"9 wall clock: {day} local vs UTC", d == -offset_h, f"local - UTC = {d:+.0f} h")


def c10_gates() -> None:
    rows = []
    rng = np.random.default_rng(9)
    for mouse, ref in (("ah10", -0.30), ("ah08", 0.35)):
        for s in range(8):
            rows.append(dict(mouse=mouse, recday=f"{mouse}_x", session=s, ap_first_sample=0,
                             speed_lag_s=ref + rng.normal(0, 0.03), speed_r=0.5, reward_peak_s=0.25,
                             reward_ptm=3.0, wallclock_s=-0.5 + rng.uniform(-0.5, 0.5), t0_dat_s=2.0))
    for s, d in enumerate((2.0, 4.9, 19.31, 29.16)):
        rows.append(dict(mouse="ah10", recday="ah10_x", session=8 + s, ap_first_sample=1000,
                         speed_lag_s=-0.30 + d, speed_r=0.5, reward_peak_s=0.25 + d, reward_ptm=3.0,
                         wallclock_s=-0.5 + d, t0_dat_s=2.0 - d))
    g = va.probe_gates(pd.DataFrame(rows))
    clean, shifted = g[g["ap_first_sample"] == 0], g[g["ap_first_sample"] > 0]
    check("10 gates: every clean session passes every probe",
          all((clean[c].dropna() == True).all() for c in ("speed_ok", "reward_ok", "wallclock_ok")),  # noqa: E712
          f"refs {g.groupby('mouse')['speed_ref_s'].first().round(2).to_dict()}")
    check("10 gates: every delayed session fails speed", (shifted["speed_ok"] == False).all(),   # noqa: E712
          f"{shifted['speed_ok'].tolist()}")
    want = [True, False, False, False]                     # reward window [-1, +4] s: a 2 s delay hides
    check("10 gates: reward fails for delays > 3.75 s", shifted["reward_ok"].tolist() == want,
          f"{shifted['reward_ok'].tolist()}")
    check("10 gates: wall clock and early start fail for the 4.9-29.2 s delays",
          shifted["wallclock_ok"].tolist() == want and shifted["early_ok"].tolist()[1:] == [False] * 3,
          f"wall {shifted['wallclock_ok'].tolist()}, early {shifted['early_ok'].tolist()}")


def c10b_c11_fallback_and_lag() -> None:
    rows = []
    rng = np.random.default_rng(10)
    for s in range(8):                                     # a well-sampled mouse sets the cohort
        rows.append(dict(mouse="ah10", recday="ah10_x", session=s, ap_first_sample=0, speed_lag_s=-0.30 + rng.normal(0, 0.03),
                         speed_r=0.5, reward_peak_s=0.25, reward_ptm=3.0, wallclock_s=-0.5 + rng.uniform(-0.5, 0.5), t0_dat_s=2.0))
    for s, d in enumerate((0.0, 0.0, 4.9, 6.8, 19.31, 29.16)):  # 2 clean, 4 delayed
        rows.append(dict(mouse="ly07", recday="ly07_x", session=s, ap_first_sample=0 if d == 0 else 1000,
                         speed_lag_s=-0.30 + d, speed_r=0.5, reward_peak_s=0.25 + d, reward_ptm=3.0,
                         wallclock_s=-0.5 + d, t0_dat_s=2.0 - d))
    for s, (off, d) in enumerate(((3.5, 0.0),) * 5 + ((3.5, 19.31),)):   # the open-field rig: own PC clock
        rows.append(dict(mouse="ah10", recday="ah10_y", session=20 + s, ap_first_sample=0 if d == 0 else 1000,
                         pycontrol_suffix=".tsv", speed_lag_s=np.nan, speed_r=0.0, reward_peak_s=np.nan,
                         reward_ptm=0.0, wallclock_s=-0.5 + off + rng.uniform(-0.3, 0.3) + d, t0_dat_s=2.0))
    g = va.probe_gates(pd.DataFrame(rows))
    rig = g[(g["mouse"] == "ah10") & (g["pycontrol_suffix"] == ".tsv")]
    check("10b gates: a second rig's clock offset passes on its own band, a 19.3 s error on it fails",
          rig[rig["ap_first_sample"] == 0]["wallclock_ok"].tolist() == [True] * 5 and rig[rig["ap_first_sample"] > 0]["wallclock_ok"].tolist() == [False],
          f"band [{rig['wallclock_lo_s'].iloc[0]:+.2f}, {rig['wallclock_hi_s'].iloc[0]:+.2f}] s; ok {rig['wallclock_ok'].tolist()}")
    ly = g[g["mouse"] == "ly07"]
    check("10b gates: too few clean sessions -> cohort reference, delayed still fail",
          abs(ly["speed_ref_s"].iloc[0] + 0.30) < 0.05 and (ly[ly["ap_first_sample"] > 0]["speed_ok"] == False).all()  # noqa: E712
          and ly["ref_source"].iloc[0] == "speed cohort, wall cohort",
          f"ly07 reference {ly['speed_ref_s'].iloc[0]:+.2f} s ({ly['ref_source'].iloc[0]}); delayed speed_ok "
          f"{ly[ly['ap_first_sample'] > 0]['speed_ok'].tolist()}")

    neural, _ = make_speed_session(11, -12)
    # rebin the same spikes at a fractional shift, as v4 against v2 is
    rng = np.random.default_rng(12)
    units, bins = np.nonzero(neural)
    counts = neural[units, bins]
    units, bins = np.repeat(units, counts), np.repeat(bins, counts)
    t = bins * 25 + rng.uniform(0, 25, len(bins))
    late = np.zeros_like(neural)
    shifted = np.floor((t + 772.49 * 25) / 25).astype(int)
    keep = shifted < neural.shape[1]
    np.add.at(late, (units[keep], shifted[keep]), 1)
    early = np.zeros_like(neural)
    np.add.at(early, (units, np.floor(t / 25).astype(int)), 1)
    lag, r = gates.pop_lag(early, late, 772.49)
    check("11 gate 5: true 772.49-bin shift found", abs(lag - 772.49) <= 1 and r >= 0.8, f"lag {lag}, r {r:.3f}")
    lag, r = gates.pop_lag(early, early, 772.49)
    check("11 gate 5: array against itself rejected", abs(lag - 772.49) > 1, f"lag {lag}, r {r:.3f}")


def quick() -> int:
    with tempfile.TemporaryDirectory(prefix="spike_clock_synth_") as d:
        tmp = Path(d)
        c1_c2_origin(tmp)
        c3_to_c6_door(tmp)
        c9_wallclock(tmp)
    c7_speed()
    c8_reward()
    c10_gates()
    c10b_c11_fallback_and_lag()
    print(f"\n{len(_FAILS)} failure(s)" + (": " + ", ".join(_FAILS) if _FAILS else ""))
    return 1 if _FAILS else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="run the controls (the only mode)")
    ap.parse_args()
    sys.exit(quick())


if __name__ == "__main__":
    main()
