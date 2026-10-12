"""Synthetic controls for lfp_anatomy_check, through the real loaders and entry points.

    python lfp_anatomy_check_synthetics.py [--quick] [--keep]

Probe-map cases are written as probe-map folders (lfp.signal.npy, lfp.metrics.htsv carrying Peter's
shank-major labels and positions, a real settings.xml, a short raw continuous.dat) and analysed by
analyse_probemap_config. Raw cases are an OpenEphys folder read by RawRecording. Signals are built
at each column's TRUE position (probeinterface on the real settings.xml). Exit 0 only if every
check passes, including those that must NOT fire.

| # | control | what a failure would mean |
|---|---|---|
| 1 | ripples planted on HPF contacts (current truth) -> verdict current | detector or AUC broken |
| 2 | ripples planted at the mirror's HPF contacts -> verdict mirror | the test cannot accept the alternative |
| 3 | no ripples, uniform theta -> inconclusive on every seed | verdict biased |
| 4 | theta + ripples injected through the reference -> raise monopolar, not bipolar | bipolar pairing wrong |
| 5 | MUA-rich channel, no ripples -> not the reference, no guarded events; unguarded it fires | spikes read as ripples |
| 6 | 1/f^chi with and without an 8 Hz bump -> chi and excess recovered; pure 1/f excess ~0 | aperiodic fit biased |
| 7 | Poisson, bursty, slow-modulated units -> theta FPR bounded; m=0.3/0.6 theta units detected | theta index miscalibrated |
| 8 | units locked at a known phase -> recovered within 10 deg; spikes offset 10 s -> locking gone | spike-LFP alignment wrong |
| 9 | guards: missing position, dead channel, few spikes, wrong serial, nested folder, two streams | silent wrong-probe reads |
| 10 | gate A0 accepts the true column map and flags exactly a 2-column swap | probe-map column gate vacuous |
| 11 | ENTm theta planted -> ENTm vs ENTl AUC high under current, low under mirror (mono + bipolar) | theta contrast wiring wrong |
| 12 | raw door: decimate_lfp + analyse_lfp_block recover the planted map; decimation r > 0.99 | raw path broken |
| 13 | lfp.metrics x/y instead of probeinterface -> the planted map is lost (AUC < 0.7) | the re-map is not needed (vacuous) |
| 14 | clock_lag recovers planted 0.2 s and 2 s lags | clock gate blind |
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

import lfp_anatomy_check as lac

PM_SETTINGS = lac.PROBEMAP_ROOT / 'raw_data/ephys/ah10/config_1/2025-05-29_15-12-41/Record Node 104/settings.xml'
RAW_SETTINGS = lac.REPO / 'data/raw/ephys/2025-06-20_16-37-32/Record Node 101/settings.xml'
AH10_SERIAL = '23107807392'
UV = 0.195
CHECKS = []


def check(name, ok, detail=''):
    CHECKS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}  {detail}')


# ---------------------------------------------------------------------------
# Synthetic anatomy: HPF only on shank 1 (y < 300), ENTl5 on shanks 0-2, ENTm5 on shank 3
# ---------------------------------------------------------------------------

def _label(x, y):
    s = int(round(x / 250.0))
    if y > 705:
        return 'VISl'
    if s == 1 and y < 300:
        return 'SUB'
    return 'ENTm5' if s == 3 else 'ENTl5'


def synth_anatomy():
    rows = []
    for x in (0, 32, 250, 282, 500, 532, 750, 782):
        for y in np.arange(0, 3600, 15):
            rows.append({'mouse': 'synth', 'x': float(x), 'y': float(y),
                         'label_current': _label(x, y), 'label_mirror': _label(782 - x, y),
                         'tissue_current': 'grey', 'tissue_mirror': 'grey'})
    t = pd.DataFrame(rows)
    for tag in ('current', 'mirror'):
        lab = t[f'label_{tag}']
        t[f'd_signed_{tag}'] = np.where(lab.str.startswith('ENTm'), 100.0,
                                        np.where(lab.str.startswith('ENTl'), -100.0, np.nan))
    return t


# ---------------------------------------------------------------------------
# Signal model
# ---------------------------------------------------------------------------

def colored(n_ch, n, fs, exponent, rng):
    """Unit-SD noise with power ~ 1/f^exponent."""
    X = np.fft.rfft(rng.standard_normal((n_ch, n)), axis=1)
    f = np.fft.rfftfreq(n, 1 / fs)
    f[0] = f[1]
    y = np.fft.irfft(X * f ** (-exponent / 2), n, axis=1)
    return (y / y.std(axis=1, keepdims=True)).astype(np.float32)


def theta_wave(n, fs, rng, f0=8.0):
    """cos(phase) with a slowly wandering frequency, so phase decorrelates over seconds."""
    walk = np.cumsum(rng.standard_normal(n)) / np.sqrt(fs * 2.0)
    walk = signal.detrend(walk)
    f_inst = f0 + 0.6 * walk / max(np.abs(walk).max(), 1e-9)
    ph = 2 * np.pi * np.cumsum(f_inst) / fs
    return np.cos(ph), np.angle(np.exp(1j * ph))


def make_lfp(xy, fs, T, rng, truth, *, ripples=True, ripple_rate=1.5, theta_entm=False,
             ref_artifact=False, mua_rows=(), dead_rows=(), period_s=10.0):
    """(n_ch, n) µV, theta-state mask, theta phase, ripple peak samples.

    truth: label per channel under the generating mapping (ripples where it is SUB/CA1).
    """
    n = int(T * fs)
    n_ch = len(xy)
    t = np.arange(n) / fs
    theta_state = (np.floor(t / period_s) % 2) == 0
    ramp = signal.windows.hann(int(fs), sym=False)
    gate = np.convolve(theta_state.astype(float), ramp / ramp.sum(), mode='same')
    x = 30.0 * colored(n_ch, n, fs, 1.6, rng) + 15.0 * colored(1, n, fs, 1.6, rng)
    # independent broadband floor, as spiking gives real channels; without it ripple channels are
    # identical above 100 Hz and no column gate could tell them apart
    x += 8.0 * rng.standard_normal((n_ch, n)).astype(np.float32)
    th, phase = theta_wave(n, fs, rng)
    x += 25.0 * gate * th
    if theta_entm:
        for c, (lab, (_, y)) in enumerate(zip(truth, xy)):
            if lab.startswith('ENTm'):
                x[c] += 80.0 * (0.2 + y / 705.0) * gate * th
    peaks = np.zeros(0, int)
    rest_idx = np.flatnonzero(~theta_state)
    burst = None
    if ripples or ref_artifact:
        n_ev = rng.poisson(ripple_rate * (~theta_state).sum() / fs)
        cand = rng.permutation(rest_idx[int(0.2 * fs):-int(0.2 * fs)])
        peaks = []
        for c in cand:
            if len(peaks) >= n_ev:
                break
            if all(abs(c - p) > 0.15 * fs for p in peaks):
                peaks.append(c)
        peaks = np.sort(np.array(peaks, int))
        tt = np.arange(-int(0.05 * fs), int(0.05 * fs) + 1) / fs
        burst = np.zeros(n)
        for pk in peaks:
            f_r = rng.uniform(150, 220)
            burst[pk + (tt * fs).astype(int)] += np.exp(-tt ** 2 / (2 * 0.012 ** 2)) * np.cos(2 * np.pi * f_r * tt)
    if ripples:
        for c, lab in enumerate(truth):
            if lab in lac.RIPPLE_SET:
                x[c] += 60.0 * burst
    if ref_artifact:
        x += 30.0 * gate * th + 40.0 * burst
    for c in mua_rows:
        # bursts of 6 spikes ~2.5 ms apart at 2 Hz: broadband transients that cross a ripple-band
        # threshold, which the MUA guard must reject
        spk = np.zeros(n)
        for b0 in np.flatnonzero(rng.random(n) < 2.0 / fs):
            for k_ in range(6):
                i = b0 + int(round(k_ * 0.0025 * fs + rng.uniform(0, 0.001) * fs))
                if i < n:
                    spk[i] += rng.uniform(0.8, 1.2)
        kern = np.array([0.0, 1.0, -0.6, -0.3])
        x[c] += 120.0 * np.convolve(spk, kern, mode='same')
    for c in dead_rows:
        x[c] = 0.0
    return x.astype(np.float32), theta_state, phase, peaks


# ---------------------------------------------------------------------------
# Probe-map folder (Peter's labelling convention)
# ---------------------------------------------------------------------------

def peter_columns():
    """Labels Peter's code gives the left column: CH(96 s + 2 r) at his (250 s, 15 r)."""
    ch, px, py, ps = [], [], [], []
    for s in range(4):
        for r in range(48):
            ch.append(96 * s + 2 * r)
            px.append(250.0 * s)
            py.append(15.0 * r)
            ps.append(s)
    return np.array(ch), np.array(px), np.array(py), np.array(ps)


def write_oebin(rec_dir, stream, n_ch=384, fs=30000.0):
    rec_dir.mkdir(parents=True, exist_ok=True)
    o = {'GUI version': '0.6.6', 'continuous': [{
        'folder_name': f'Neuropix-PXI-100.{stream}/', 'sample_rate': fs, 'num_channels': n_ch,
        'channels': [{'channel_name': f'CH{i}', 'bit_volts': UV} for i in range(n_ch)]}]}
    (rec_dir / 'structure.oebin').write_text(json.dumps(o))


def write_dat(path, cols, x_lfp, fs_lfp, raw_s, rng, fs=30000.0, n_ch=384):
    """int16 raw: the given raw channels are the LFP upsampled plus white noise; the rest noise."""
    q = int(round(fs / fs_lfp))
    n = int(raw_s * fs) // q * q
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, dtype=np.int16, mode='w+', shape=(n, n_ch))      # headerless, as OpenEphys
    step = int(fs)
    for s in range(0, n, step):
        e = min(s + step, n)
        mm[s:e] = (rng.standard_normal((e - s, n_ch)) * 3.0 / UV).astype(np.int16)
    for k0 in range(0, len(cols), 32):
        up = signal.resample_poly(x_lfp[k0:k0 + 32, :n // q].astype(np.float64), q, 1, axis=1)[:, :n]
        up += rng.standard_normal(up.shape) * 3.0
        for k, c in enumerate(cols[k0:k0 + 32]):
            mm[:, c] = np.clip(up[k] / UV, -32768, 32767).astype(np.int16)
    mm.flush()
    del mm


def write_probemap(root, x_lfp, fs, raw_s, rng, *, swap=None, mouse='synth', config='config_1',
                   settings=PM_SETTINGS, extra_stream=False):
    ch, px, py, ps = peter_columns()
    proc = Path(root) / 'processed_data' / mouse / config
    proc.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(root) / 'raw_data' / 'ephys' / mouse / config / '2025-01-01_00-00-00' / 'Record Node 104'
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(settings, raw_dir / 'settings.xml')
    rec = raw_dir / 'experiment1' / 'recording1'
    write_oebin(rec, 'ProbeA')
    write_dat(rec / 'continuous' / 'Neuropix-PXI-100.ProbeA' / 'continuous.dat', ch, x_lfp, fs, raw_s, rng)
    if extra_stream:
        (rec / 'continuous' / 'Neuropix-PXI-100.ProbeB').mkdir(parents=True, exist_ok=True)
    sig = x_lfp.copy()
    if swap is not None:
        a, b = swap
        sig[[a, b]] = sig[[b, a]]
    np.save(proc / 'lfp.signal.npy', sig.T.astype(np.float16))
    pd.DataFrame({'contact.id': [f's{s}e{2 * (i % 48)}' for i, s in enumerate(ps)],
                  'contact.index': np.arange(len(ch)), 'contact.label': [f'CH{c}' for c in ch],
                  'contact.qc': 'good', 'contact.x': px, 'contact.y': py, 'shank.index': ps,
                  'sampling_rate': int(fs)}).to_csv(proc / 'lfp.metrics.htsv', sep='\t', index=False)


def true_positions(settings=PM_SETTINGS, stream='ProbeA'):
    ch, *_ = peter_columns()
    g = lac.probe_geometry(settings, stream).set_index('ch')
    return np.stack([g.loc[ch, 'x'].values, g.loc[ch, 'y'].values], axis=1)


def probemap_case(root, anat, truth_mapping, seed, T, *, raw_s=14.0, **kw):
    """Write one synthetic probe-map config and analyse it through analyse_probemap_config."""
    rng = np.random.default_rng(seed)
    xy = true_positions()
    lab = anat.labels(xy[:, 0], xy[:, 1], truth_mapping)[0]
    swap = kw.pop('swap', None)
    x, theta_state, phase, peaks = make_lfp(xy, 1500.0, T, rng, lab, **kw)
    case = Path(root) / f'case_{truth_mapping}_{seed}_{len(list(Path(root).glob("case_*")))}'
    write_probemap(case, x, 1500.0, raw_s, rng, swap=swap)
    ch, bip, psd, meta = lac.analyse_probemap_config('synth', 'config_1', lac.LFPConfig(), root=case, gate=True)
    return ch, bip, meta, x, case


def orient(ch, anat, measure='ripple_gain'):
    return lac.orientation_test(ch['x'].values, ch['y'].values, ch[measure].values, anat, lac.LFPConfig(),
                                keep=~ch['is_ref'].values)


# ---------------------------------------------------------------------------
# Raw OpenEphys folder
# ---------------------------------------------------------------------------

def raw_case(root, anat, seed, T, *, theta_entm=True, nested=False):
    rng = np.random.default_rng(seed)
    g = lac.probe_geometry(RAW_SETTINGS, 'ProbeB')
    xy = g[['x', 'y']].values
    lab = anat.labels(xy[:, 0], xy[:, 1], 'current')[0]
    fs_l = 1250.0
    x, theta_state, phase, peaks = make_lfp(xy, fs_l, T, rng, lab, ripple_rate=2.0, theta_entm=theta_entm,
                                            period_s=5.0)
    top = Path(root) / f'raw_{seed}' / '2025-01-01_00-00-00'
    node = top / 'Record Node 101'
    node.mkdir(parents=True, exist_ok=True)
    shutil.copy(RAW_SETTINGS, node / 'settings.xml')
    rec = node / 'experiment1' / 'recording1'
    write_oebin(rec, 'ProbeB')
    dat = rec / 'continuous' / 'Neuropix-PXI-100.ProbeB' / 'continuous.dat'
    write_dat(dat, g['ch'].values, x, fs_l, T, rng)
    if nested:
        (top / '2025-01-01_00-00-01').mkdir()
    return dat, x, theta_state, phase, peaks, g


# ---------------------------------------------------------------------------
# Unit trains
# ---------------------------------------------------------------------------

def inhomogeneous(rate_fn, T, rmax, rng):
    n = rng.poisson(rmax * T)
    t = np.sort(rng.uniform(0, T, n))
    return t[rng.random(n) < rate_fn(t) / rmax]


def unit_case(kind, T, rng, m=0.0):
    fs = 1000.0
    if kind == 'poisson':
        t = np.sort(rng.uniform(0, T, rng.poisson(6 * T)))
    elif kind == 'bursty':
        c = np.sort(rng.uniform(0, T, rng.poisson(2.5 * T)))
        t = np.sort(np.concatenate([c, c + 0.005, c + 0.010]))
    elif kind == 'ahp':
        t, out = 0.0, []
        while t < T:
            t += 0.06 + rng.exponential(1 / 3.0)
            nb = 1 + rng.poisson(1.5)
            out.extend(t + np.concatenate([[0], np.cumsum(rng.uniform(0.003, 0.007, nb - 1))]))
        t = np.array(out)
    elif kind == 'slow':
        t = inhomogeneous(lambda u: 6 * (1 + 0.8 * np.sin(2 * np.pi * 0.3 * u)), T, 12, rng)
    else:
        n = int(T * fs)
        _, ph = theta_wave(n, fs, rng)
        t = inhomogeneous(lambda u: 6 * (1 + m * np.cos(ph[np.minimum((u * fs).astype(int), n - 1)])),
                          T, 6 * (1 + m), rng)
    return t


def run_units(kind, n_units, T, cfg, seed, m=0.0):
    rng = np.random.default_rng(seed)
    nb = int(T / 0.025)
    speed = np.where((np.arange(nb) * 0.025) % 10.0 < 6.0, 10.0, 0.5).astype(np.float32)
    clock = {'a': 30.0, 'b': 0.0}
    out = []
    for u in range(n_units):
        t = unit_case(kind, T, rng, m)
        samples = np.round(t * 30000.0).astype(np.int64)
        rows = np.zeros(len(samples), np.int64)
        trains, run_s = lac.running_spike_trains(samples, rows, 1, speed, clock, 0, 0, cfg)
        out.append(lac.unit_theta_rhythmicity(trains[0][0], trains[0][1], cfg, rng))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(quick, keep):
    cfg = lac.LFPConfig(n_jitter=100 if quick else 200)
    T = 120.0 if quick else 240.0
    n_seeds = 5 if quick else 20
    n_units = 40 if quick else 150
    root = Path(tempfile.mkdtemp(prefix='lfp_anat_synth_'))
    anat = lac.Anatomy(synth_anatomy())
    print(f'synthetic root {root}')
    try:
        print('(1)(10)(13) ripples at current-truth HPF contacts')
        ch, bip, meta, x, case = probemap_case(root, anat, 'current', 0, T)
        o = orient(ch, anat)
        check('verdict current', o['verdict'] == 'current',
              f'AUC current {o["auc_current"]:.3f} mirror {o["auc_mirror"]:.3f} rank {o["rank_current"]}/24')
        dt = lac.discordant_test(ch.x.values, ch.y.values, ch.ripple_gain.values, anat, keep=~ch.is_ref.values, n_perm=500)
        check('discordant AUC favours current (> 0.9, p < 0.05)', dt['auc_discordant'] > 0.9 and dt['p_favours_current'] < 0.05,
              f'{dt["auc_discordant"]:.3f}, p {dt["p_favours_current"]:.3f}, n {dt["n_current_only"]}/{dt["n_mirror_only"]}')
        ref = ch.iloc[meta['ref']]
        check('reference on an HPF contact', anat.labels([ref.x], [ref.y])[0][0] in lac.RIPPLE_SET,
              f'ref x {ref.x:.0f} y {ref.y:.0f}, {meta["n_events"]} events')
        check('gate A0 passes on the true map', meta['gate_a0_pass'],
              f'median r_own {meta["gate_a0"].r_own.median():.3f}, min {meta["gate_a0"].r_own.min():.3f}')
        lfp, fs, cols, paths = lac.load_probemap('synth', 'config_1', case)
        om = lac.orientation_test(cols['metrics_x'].values, cols['metrics_y'].values, ch['ripple_gain'].values,
                                  anat, cfg, keep=~ch['is_ref'].values)
        check('(13) lfp.metrics x/y lose the planted map', om['auc_current'] < 0.7,
              f'AUC current via metrics x/y {om["auc_current"]:.3f}; metrics agree {meta["metrics_xy_agree"]:.2f}')

        print('(10) gate A0 flags a 2-column swap')
        _, _, meta_s, _, _ = probemap_case(root, anat, 'current', 1, 60.0, swap=(5, 100))
        bad = set(meta_s['gate_a0'].loc[~meta_s['gate_a0'].ok, 'col'])
        check('exactly the swapped columns flagged', bad == {5, 100}, f'flagged {sorted(bad)}')

        print('(2) ripples at mirror-truth HPF contacts')
        ch2, _, _, _, _ = probemap_case(root, anat, 'mirror', 2, T)
        o2 = orient(ch2, anat)
        check('verdict mirror', o2['verdict'] == 'mirror',
              f'AUC current {o2["auc_current"]:.3f} mirror {o2["auc_mirror"]:.3f}')
        dt2 = lac.discordant_test(ch2.x.values, ch2.y.values, ch2.ripple_gain.values, anat, keep=~ch2.is_ref.values, n_perm=500)
        check('discordant AUC favours mirror (< 0.1, p < 0.05)', dt2['auc_discordant'] < 0.1 and dt2['p_favours_mirror'] < 0.05,
              f'{dt2["auc_discordant"]:.3f}, p {dt2["p_favours_mirror"]:.3f}')

        print('(3) no ripples: inconclusive on every seed')
        noise_cases = {}
        for measure in ('ripple_gain', 'rest_ripple_excess_db', 'ripple_rate_hz'):
            verdicts, aucs = [], []
            for s in range(n_seeds):
                if s not in noise_cases:
                    noise_cases[s] = probemap_case(root, anat, 'current', 100 + s, T, ripples=False)[0]
                c3 = noise_cases[s]
                o3 = orient(c3, anat, measure)
                verdicts.append(o3['verdict'])
                aucs.append(o3['auc_current'])
            n_false = sum(v != 'inconclusive' for v in verdicts)
            fin = [a for a in aucs if np.isfinite(a)]
            check(f'{measure}: false verdicts 0/{n_seeds}', n_false == 0,
                  f'{n_false} non-inconclusive; AUC current ' +
                  (f'{min(fin):.2f}..{max(fin):.2f}' if fin else 'all NaN (< min_ripples events)'))

        sig = 0
        da = []
        for s_ in range(n_seeds):
            d3 = lac.discordant_test(noise_cases[s_].x.values, noise_cases[s_].y.values,
                                     noise_cases[s_].rest_ripple_excess_db.values, anat, n_perm=500, seed=s_)
            da.append(d3['auc_discordant'])
            sig += int(d3['p_favours_current'] < 0.05 or d3['p_favours_mirror'] < 0.05)
        check(f'discordant test under noise: <= 1/{n_seeds} significant, AUC 0.25-0.75', sig <= 1 and
              min(da) > 0.25 and max(da) < 0.75, f'{sig} significant; AUC {min(da):.2f}..{max(da):.2f}')
        print('(4) reference artifact: monopolar moves, bipolar does not')
        c4a, b4a, _, _, _ = probemap_case(root, anat, 'current', 3, T)
        c4b, b4b, _, _, _ = probemap_case(root, anat, 'current', 3, T, ref_artifact=True)
        dm = np.nanmedian(c4b['th_theta_excess_db'] - c4a['th_theta_excess_db'])
        db = np.nanmedian(b4b['th_theta_excess_db'] - b4a['th_theta_excess_db'])
        check('monopolar theta excess rises > 1 dB', dm > 1.0, f'{dm:+.2f} dB')
        check('bipolar theta excess within 0.5 dB', abs(db) < 0.5, f'{db:+.2f} dB')
        non = np.array([l not in lac.RIPPLE_SET for l in anat.labels(c4a.x, c4a.y)[0]])
        dmr = np.nanmedian((c4b['rest_ripple_excess_db'] - c4a['rest_ripple_excess_db'])[non])
        nonb = np.array([l not in lac.RIPPLE_SET for l in anat.labels(b4a.x, b4a.y)[0]])
        dbr = np.nanmedian((b4b['rest_ripple_excess_db'] - b4a['rest_ripple_excess_db'])[nonb])
        check('ripple artifact: monopolar non-HPF ripple excess rises > 1 dB, bipolar < 0.5 dB',
              dmr > 1.0 and abs(dbr) < 0.5, f'mono {dmr:+.2f} dB, bip {dbr:+.2f} dB')

        print('(5) MUA-rich channel')
        xy = true_positions()
        target = int(np.argmin(np.hypot(xy[:, 0] - 500, xy[:, 1] - 600)))
        c5, _, m5, x5, _ = probemap_case(root, anat, 'current', 4, T, mua_rows=(target,), ripples=False)
        check('MUA channel not the reference', m5['ref'] != target, f'ref {m5["ref"]}, mua {target}')
        check('MUA channel ripple excess < 1 dB', c5['rest_ripple_excess_db'].iloc[target] < 1.0,
              f'{c5["rest_ripple_excess_db"].iloc[target]:.2f} dB')
        starts, w = lac.window_grid(x5.shape[1], 1500.0, cfg.window_s)
        rest_mask = lac.mask_from_windows(m5['rest_win'], m5['window_samples'], x5.shape[1])
        env = lac.band_envelope(x5[target:target + 1], 1500.0, cfg.ripple_band, cfg.ripple_smooth_ms)[0]
        mua = lac.band_envelope(x5[target:target + 1], 1500.0, cfg.mua_band, cfg.ripple_smooth_ms)[0]
        rest_s = rest_mask.sum() / 1500.0
        n_g = len(lac.detect_events(lac.robust_z(env, rest_mask), 1500.0, rest_mask, cfg,
                                    z_guard=lac.robust_z(mua, rest_mask)))
        n_u = len(lac.detect_events(lac.robust_z(env, rest_mask), 1500.0, rest_mask, cfg))
        check('guarded rate < 0.05 Hz, unguarded > 0.2 Hz (guard not vacuous)',
              n_g / rest_s < 0.05 and n_u / rest_s > 0.2, f'guarded {n_g / rest_s:.3f} Hz, unguarded {n_u / rest_s:.3f} Hz')

        print('(6) aperiodic fit')
        rng = np.random.default_rng(6)
        fs = 1500.0
        for chi in (1.0, 2.0):
            base = colored(4, int(200 * fs), fs, chi, rng) * 30
            bump = bandlimited(4, int(200 * fs), fs, (7.0, 9.0), rng) * 30
            st, w = lac.window_grid(base.shape[1], fs, 2.0)
            f, P0 = lac.window_psd(base, fs, st, w)
            _, P1 = lac.window_psd(base + bump, fs, st, w)
            s0 = lac.spectral_summary(f, P0, cfg)
            s1 = lac.spectral_summary(f, P1, cfg)
            check(f'chi {chi}: exponent within 0.15', np.all(np.abs(s0['ap_exponent'] - chi) < 0.15),
                  f'{np.round(s0["ap_exponent"], 2)}')
            check(f'chi {chi}: pure 1/f theta excess |.| < 0.5 dB',
                  np.all(np.abs(s0['theta_excess_db']) < 0.5), f'{np.round(s0["theta_excess_db"], 2)}')
            check(f'chi {chi}: bump excess > 3 dB, peak 7-9 Hz',
                  np.all(s1['theta_excess_db'] > 3) and np.all((s1['theta_peak_hz'] >= 7) & (s1['theta_peak_hz'] <= 9)),
                  f'{np.round(s1["theta_excess_db"], 1)} dB at {s1["theta_peak_hz"]}')

        print('(7) theta rhythmicity from spikes')
        Tu = 1200.0 if quick else 2400.0
        med = {}
        for kind in ('poisson', 'bursty', 'ahp', 'slow'):
            r = run_units(kind, n_units, Tu, cfg, 70 + len(kind))
            fpr = float((r.p_jitter < 0.05).mean())
            lim = 0.05 + 2.5 * np.sqrt(0.05 * 0.95 / n_units)
            med[kind] = r.theta_excess_db.median()
            check(f'{kind}: FPR <= {lim:.3f}', fpr <= lim,
                  f'{fpr:.3f} of {n_units}, median excess {med[kind]:.2f} dB, ratio {r.theta_ratio.median():.2f}, '
                  f'burst_frac {r.burst_frac.median():.2f}, spikes {int(r.n_spikes.median())}')
        # m=0.3 at ~4k spikes is near the floor (23% in the prototype): require 3x alpha
        for m, need in ((0.3, 0.15), (0.6, 0.9)):
            r = run_units('theta', n_units // 2, Tu, cfg, 80, m=m)
            det = float((r.p_jitter < 0.05).mean())
            check(f'theta m={m}: detected >= {need}', det >= need,
                  f'{det:.2f}, median excess {r.theta_excess_db.median():.2f} dB, freq {r.theta_hz.median():.1f} Hz')
            if m == 0.6:
                check('theta m=0.6 median excess > noise median + 6 dB',
                      r.theta_excess_db.median() > med['poisson'] + 6.0,
                      f'{r.theta_excess_db.median():.2f} vs {med["poisson"]:.2f} dB')
        few = lac.unit_theta_rhythmicity(np.sort(np.random.default_rng(0).uniform(0, 100, 100)),
                                         np.zeros(100, int), cfg, np.random.default_rng(0))
        check('(9) < min_spikes -> NaN, no exception', np.isnan(few['theta_excess_db']), f'{few}')

        print('(12)(8) raw door')
        Tr = 60.0 if quick else 120.0
        dat, xr, th_state, phase, peaks, g = raw_case(root, anat, 12, Tr)
        rec = lac.RawRecording(dat, expected_serial=AH10_SERIAL)
        lfp, lfp_start = lac.decimate_lfp(rec, 0, rec.n_samples, 1250)
        r_dec = lac.verify_decimation(rec, lfp, lfp_start, 1250, t0_s=0.0, dur_s=min(20.0, Tr))
        check('decimation r median > 0.99', np.median(r_dec) > 0.99, f'median {np.median(r_dec):.4f}, min {r_dec.min():.4f}')
        err = np.abs(lfp - xr[:, :lfp.shape[1]]).mean() / np.abs(xr).mean()
        check('decimated LFP equals the planted signal (rel err < 0.1)', err < 0.1, f'{err:.3f}')
        pos = pd.DataFrame({'name': rec.names})
        pos['ch'] = pos['name'].str[2:].astype(int)
        pos = pos.merge(rec.geom[['ch', 'x', 'y', 'shank']], on='ch', how='left')
        st, w = lac.window_grid(lfp.shape[1], 1250.0, cfg.window_s)
        win_theta = np.array([s for s in st if th_state[s:s + w].all()])
        win_rest = np.array([s for s in st if not th_state[s:s + w].any()])
        rest_mask = lac.mask_from_windows(win_rest, w, lfp.shape[1])
        chr_, bipr, _, mr = lac.analyse_lfp_block(lfp, 1250.0, pos, cfg, theta_win=win_theta,
                                                  rest_win=win_rest, rest_mask=rest_mask)
        orr = lac.orientation_test(chr_.x.values, chr_.y.values, chr_['ripple_gain'].values, anat, cfg,
                                   keep=~chr_['is_ref'].values)
        check('raw: verdict current', orr['verdict'] == 'current',
              f'AUC current {orr["auc_current"]:.3f} mirror {orr["auc_mirror"]:.3f}, {mr["n_events"]} events')
        tr = lac.ec_theta_test(chr_.x.values, chr_.y.values, chr_['th_theta_excess_db'].values, anat)
        check('(11) raw monopolar ENTm vs ENTl: current > 0.9, mirror < 0.5',
              tr['auc_entm_entl_current'] > 0.9 and tr['auc_entm_entl_mirror'] < 0.5,
              f'{tr["auc_entm_entl_current"]:.3f} / {tr["auc_entm_entl_mirror"]:.3f}')
        tb = lac.ec_theta_test(bipr.x.values, bipr.y.values, bipr['th_theta_excess_db'].values, anat)
        check('(11) raw bipolar ENTm vs ENTl: current > 0.9, mirror < 0.5',
              tb['auc_entm_entl_current'] > 0.9 and tb['auc_entm_entl_mirror'] < 0.5,
              f'{tb["auc_entm_entl_current"]:.3f} / {tb["auc_entm_entl_mirror"]:.3f}')
        # (8) phase locking through unit_lfp_measures, in .dat samples
        rng = np.random.default_rng(8)
        q = 24
        n_l = lfp.shape[1]
        phi0 = np.deg2rad(60.0)
        ph = phase[:n_l]
        rate = 10.0 * (1 + 0.6 * np.cos(ph - phi0)) * th_state[:n_l]
        spk_l = np.flatnonzero(rng.random(n_l) < rate / 1250.0)
        samples = lfp_start + spk_l * q + rng.integers(0, q, len(spk_l))
        names = list(pos['name'])
        peak = [names[int(np.argmin(np.hypot(pos.x - 0, pos.y - 450)))]]
        um = lac.unit_lfp_measures(samples, np.zeros(len(samples), int), 1, peak, lfp, lfp_start, q, 1250.0,
                                   pos, win_theta, w, rest_mask, mr['events'][:, 1], chr_, cfg)
        dphi = np.rad2deg(np.angle(np.exp(1j * (um.phase_ref.iloc[0] - phi0))))
        check('(8) preferred phase within 10 deg', abs(dphi) < 10,
              f'{dphi:+.1f} deg, MRL {um.mrl_ref.iloc[0]:.3f} (expected 0.30)')
        um2 = lac.unit_lfp_measures(samples + int(10 * 30000), np.zeros(len(samples), int), 1, peak, lfp,
                                    lfp_start, q, 1250.0, pos, win_theta, w, rest_mask, mr['events'][:, 1], chr_, cfg)
        check('(8) 10 s offset kills locking (MRL < 0.08)', um2.mrl_ref.iloc[0] < 0.08,
              f'MRL {um2.mrl_ref.iloc[0]:.3f}')

        print('(9) guards')
        lab, tis = anat.labels([9999.0], [0.0])
        check('missing position -> outside, excluded', lab[0] == 'outside' and lac.hpf_class(lab, tis)[0] is None,
              f'{lab[0]}')
        try:
            lac.RawRecording(dat, expected_serial='00000000000')
            check('wrong serial raises', False)
        except ValueError as e:
            check('wrong serial raises', True, str(e)[:60])
        datn, *_ = raw_case(root, anat, 13, 4.0, nested=True)
        try:
            lac.RawRecording(datn)
            check('nested recording folder raises', False)
        except RuntimeError as e:
            check('nested recording folder raises', True, str(e)[-40:])
        rng = np.random.default_rng(9)
        x9 = make_lfp(true_positions(), 1500.0, 60.0, rng, anat.labels(*true_positions().T)[0],
                      dead_rows=(10,))[0]
        # an LFP-only channel (no >300 Hz power): its steep spectrum must not fake a ripple 'excess'
        x9[20] = signal.sosfiltfilt(signal.butter(6, 40.0, fs=1500.0, output='sos'), x9[20]).astype(np.float32)
        two = Path(root) / 'two_streams'
        write_probemap(two, x9, 1500.0, 4.0, rng, extra_stream=True)
        try:
            lac.probemap_paths('synth', 'config_1', two)
            check('two streams in a probe-map folder raises', False)
        except ValueError as e:
            check('two streams in a probe-map folder raises', True, str(e)[:50])
        dead = Path(root) / 'dead'
        write_probemap(dead, x9, 1500.0, 14.0, rng)
        c9, _, _, m9 = lac.analyse_probemap_config('synth', 'config_1', cfg, root=dead, gate=False)
        check('flat and LFP-only channels: flagged, measures NaN, neither the reference',
              bool(c9['dead'].iloc[10]) and bool(c9['dead'].iloc[20]) and np.isnan(c9['ripple_rate_hz'].iloc[10])
              and np.isnan(c9['th_theta_excess_db'].iloc[20]) and m9['ref'] not in (10, 20)
              and int(c9['dead'].sum()) == 2,
              f'ref {m9["ref"]}, dead flags {sorted(np.flatnonzero(c9["dead"].values))}')

        print('(11) probe-map ENTm theta')
        c11, b11, _, _, _ = probemap_case(root, anat, 'current', 11, T, theta_entm=True)
        tm = lac.ec_theta_test(c11.x.values, c11.y.values, c11['th_theta_excess_db'].values, anat)
        tbm = lac.ec_theta_test(b11.x.values, b11.y.values, b11['th_theta_excess_db'].values, anat)
        check('probe-map mono ENTm vs ENTl: current > 0.9, mirror < 0.5',
              tm['auc_entm_entl_current'] > 0.9 and tm['auc_entm_entl_mirror'] < 0.5,
              f'{tm["auc_entm_entl_current"]:.3f} / {tm["auc_entm_entl_mirror"]:.3f}')
        check('probe-map bipolar ENTm vs ENTl: current > 0.9, mirror < 0.5',
              tbm['auc_entm_entl_current'] > 0.9 and tbm['auc_entm_entl_mirror'] < 0.5,
              f'{tbm["auc_entm_entl_current"]:.3f} / {tbm["auc_entm_entl_mirror"]:.3f}')

        print('(14) clock lag')
        rng = np.random.default_rng(14)
        sp = np.convolve(rng.random(3000), np.ones(10) / 10, mode='same')
        for lag_s in (0.2, 2.0):
            k = int(round(lag_s / 0.1))
            pw = np.roll(sp, k) + 0.05 * rng.standard_normal(len(sp))
            got, r, _, _ = lac.clock_lag(pw, sp, 0.1)
            check(f'planted {lag_s} s recovered', abs(got - lag_s) < 0.11, f'{got:+.2f} s, r {r:.2f}')

        print('(16) median periodogram and rank clock lag (post-hoc options)')
        rng = np.random.default_rng(16)
        fs16 = 1500.0
        base = colored(4, int(240 * fs16), fs16, 1.6, rng) * 30 + bandlimited(4, int(240 * fs16), fs16, (7.0, 9.0), rng) * 20
        st, w = lac.window_grid(base.shape[1], fs16, 2.0)
        cm = lac.LFPConfig()
        s_mean = lac.spectral_summary(*lac.window_psd(base, fs16, st, w, 'mean'), cm)
        s_med = lac.spectral_summary(*lac.window_psd(base, fs16, st, w, 'median'), cm)
        d0 = np.abs(s_mean['theta_excess_db'] - s_med['theta_excess_db']).max()
        check('clean data: median and mean PSD give the same theta excess (< 1 dB)', d0 < 1.0, f'max diff {d0:.2f} dB')
        art = base.copy()
        lp60 = signal.butter(4, 60.0, fs=fs16, output='sos')
        for s0 in st[rng.choice(len(st), 6, replace=False)]:          # 6 of 120 windows: a 1 mV broadband burst <60 Hz
            art[:, s0:s0 + w] += 1000.0 * signal.sosfiltfilt(lp60, rng.standard_normal((4, w)), axis=1)
        a_mean = lac.spectral_summary(*lac.window_psd(art, fs16, st, w, 'mean'), cm)
        a_med = lac.spectral_summary(*lac.window_psd(art, fs16, st, w, 'median'), cm)
        dm = np.abs(a_mean['theta_excess_db'] - s_mean['theta_excess_db']).max()
        dd = np.abs(a_med['theta_excess_db'] - s_med['theta_excess_db']).max()
        check('artifact windows: mean PSD moves > 1 dB, median < 0.5 dB (option not vacuous)', dm > 1.0 and dd < 0.5,
              f'mean moves {dm:.2f} dB, median {dd:.2f} dB')
        sp16 = np.convolve(rng.random(3000), np.ones(10) / 10, mode='same')
        pw16 = np.log10(np.roll(sp16, 3) + 0.05) + np.where(rng.random(3000) < 0.02, 5.0, 0.0)   # heavy-tailed bursts
        got, rho, _, _ = lac.clock_lag(pw16, sp16, 0.1, rank=True)
        gp, rp, _, _ = lac.clock_lag(pw16, sp16, 0.1)
        check('rank clock lag recovers a planted 0.3 s lag where Pearson fails (r < 0.3)', abs(got - 0.3) < 0.11 and rp < 0.3,
              f'rank {got:+.2f} s rho {rho:.2f}; Pearson {gp:+.2f} s r {rp:.2f}')

        print('(15) running / still windows on the ephys clock')
        fs_raw, q = 30000.0, 24
        clock = {'a': 30.0004, 'b': 98765.0}
        t0 = 1475
        nb = int(600 / 0.025)
        speed = np.where((np.arange(nb) * 0.025) % 6.0 < 3.0, 12.0, 0.5).astype(np.float32)
        lfp_start = 7 * q
        n_lfp = int((clock['a'] * (t0 + 25 * nb) + clock['b'] - lfp_start) / q)
        w = int(cfg.window_s * 1250)

        def truth_run(st):
            # planted state as a function of raw sample, written from the clock definition
            s = lfp_start + (st + np.arange(w)) * q
            return (((s - clock['b']) / clock['a'] - t0) / 1000.0) % 6.0 < 3.0

        rw, sw, _ = lac.speed_windows(speed, clock, t0, lfp_start, fs_raw, 1250.0, w, cfg, n_lfp)
        fr = [truth_run(s).mean() for s in rw]
        fs_ = [1 - truth_run(s).mean() for s in sw]
        check('run windows lie >= 75% in planted running', len(rw) > 50 and min(fr) >= 0.75,
              f'{len(rw)} windows, worst {min(fr):.2f}')
        check('still windows lie >= 90% in planted still', len(sw) > 50 and min(fs_) >= 0.9,
              f'{len(sw)} windows, worst {min(fs_):.2f}')
        rw_bad, _, _ = lac.speed_windows(speed, clock, t0, lfp_start + int(3.0 * fs_raw), fs_raw, 1250.0, w,
                                         cfg, n_lfp)
        fr_bad = [truth_run(s).mean() for s in rw_bad]
        check('a 3 s lfp_start error puts run windows in the wrong state (check not vacuous)',
              np.median(fr_bad) < 0.3, f'median running fraction {np.median(fr_bad):.2f}')
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
    n_fail = sum(not ok for _, ok in CHECKS)
    print(f'\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed')
    for name, ok in CHECKS:
        if not ok:
            print(f'  FAILED: {name}')
    return 1 if n_fail else 0


def bandlimited(n_ch, n, fs, band, rng):
    x = signal.sosfiltfilt(signal.butter(4, band, btype='bandpass', fs=fs, output='sos'),
                           rng.standard_normal((n_ch, n)), axis=1)
    return (x / x.std(axis=1, keepdims=True)).astype(np.float32)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--keep', action='store_true', help='keep the synthetic folders')
    a = ap.parse_args()
    sys.exit(main(a.quick, a.keep))
