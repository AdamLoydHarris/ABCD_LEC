"""LFP and spike checks of the LEC probe anatomy: ripples test shank orientation, theta tests MEC vs LEC.

    python run_lfp_anatomy_check.py --stage anatomy|probemap|speed|units|raw|merge   (under sbatch)

Why: unit ENTm/ENTl labels depend on the shank-order convention (shank 0 = anterior), which dye
cannot test; under the mirror 0 of 284 ENTm unit-recordings stay ENTm. Ripples mark CA1/SUB/ProS
independently of the MEC-vs-LEC question, so they settle orientation; theta then asks whether
ENTm-labelled sites look like MEC (Deshmukh et al. 2010: relative theta 0.50 MEC vs 0.18 LEC).

Inputs per mouse: a contact anatomy table (stage `anatomy`, histology env) with labels at every
probe (x, y) under the current fit and under the exact mirror (u, normal, offset_x, theta negated).
Probe-map LFP (Peter's, 1500 Hz) is re-mapped through probeinterface: its lfp.metrics x/y follow a
shank-major channel numbering that is wrong for 50-100% of channels (LFP_ANATOMY_CHECK.md).
Raw task/sleep data are the 30 kHz wideband continuous.dat, decimated here to 1250 Hz.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson, rankdata

REPO = Path('/ceph/behrens/adam_harris/Taskspace_abstraction_lEC')
PROBEMAP_ROOT = Path('/ceph/behrens/adam_harris/LEC_probe_map/data')
OUT_DIR = REPO / 'data' / 'processed_data' / 'lfp_anatomy_check'
FIG_DIR = REPO / 'data' / 'figures' / 'lfp_anatomy_check'
MICE = ('ah08', 'ah10', 'ly05', 'ly06', 'ly07')
SERIALS = {'ah10': '23107807392', 'ly07': '23107807752', 'ly05': '23107802752',
           'ah08': '23107807764', 'ly06': '23107807394'}

MIRROR_X_UM = 782.0          # mirror sends contact (x, y) to the position of (782 - x, y)
SHANK_PITCH_UM = 250.0
ROW_PITCH_UM = 15.0
#: labels that generate or follow sharp-wave ripples; DG is excluded from both classes
RIPPLE_SET = ('CA1', 'CA2', 'CA3', 'SUB', 'ProS', 'HPF')
N_CH = 384


@dataclass
class LFPConfig:
    """Defaults argued for in LFP_ANATOMY_CHECK.md."""
    theta_band: tuple = (6.0, 10.0)
    delta_band: tuple = (1.0, 4.0)
    total_band: tuple = (1.0, 50.0)          # Deshmukh's relative-theta denominator
    ap_range: tuple = (2.0, 45.0)
    ap_exclude: tuple = ((4.0, 12.0), (48.0, 52.0))
    ripple_band: tuple = (150.0, 250.0)
    ripple_ap_range: tuple = (80.0, 360.0)   # the probe-map LFP rolls off above ~400 Hz
    ripple_ap_exclude: tuple = ((130.0, 270.0),)
    mua_band: tuple = (300.0, 450.0)
    window_s: float = 2.0
    theta_state_frac: float = 0.30
    rest_state_frac: float = 0.50
    ripple_z_lo: float = 3.0
    ripple_z_hi: float = 5.0
    ripple_min_ms: float = 20.0
    ripple_max_ms: float = 300.0
    ripple_merge_ms: float = 15.0
    ripple_smooth_ms: float = 4.0
    ripple_half_win_ms: float = 25.0
    ripple_mod_half_win_ms: float = 50.0
    min_ripples: int = 20
    bipolar_rows: int = 4                    # 60 um at 15 um pitch
    run_cm_s: float = 5.0                    # repo convention (tracking_qc.MOVING_CM_S)
    still_cm_s: float = 2.0
    run_frac: float = 0.75                   # of a window's 25 ms bins above run_cm_s
    still_frac: float = 0.90                 # of a window's bins below still_cm_s (none above run_cm_s)
    min_epoch_s: float = 1.0
    acg_bin_ms: float = 5.0
    acg_max_ms: float = 500.0
    acg_skip_ms: float = 20.0                # refractory dip and 3-5 spike bursts stay out of the spectrum
    theta_search: tuple = (5.0, 11.0)
    theta_halfwidth_hz: float = 1.0
    acg_spec_max_hz: float = 50.0
    n_jitter: int = 200
    jitter_ms: float = 62.5                  # half a theta cycle: smears 8 Hz to zero
    min_spikes: int = 500
    max_spikes: int = 30000                  # uniform thinning keeps the ACG shape; bounds 40 Hz units
    fs_lfp: int = 1250
    psd_stat: str = 'mean'                   # 'median' = robustness to artifact windows (post-hoc)
    auc_support: float = 0.7
    auc_margin: float = 0.2
    clock_max_lag_s: float = 0.5             # theta can lead running by ~0.3 s; v4 errors were >= 0.94 s
    seed: int = 0


# ---------------------------------------------------------------------------
# Geometry and anatomy lookups
# ---------------------------------------------------------------------------

def shank_of(x):
    return np.round(np.asarray(x, float) / SHANK_PITCH_UM).astype(int)


def probe_geometry(settings_xml, stream_name, expected_serial=None) -> pd.DataFrame:
    """Per hardware channel: (ch, x, y, shank, contact_id) from probeinterface on settings.xml."""
    import warnings
    import probeinterface as pi
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pr = pi.read_openephys(str(settings_xml), stream_name=stream_name)
    serial = str(pr.annotations.get('serial_number'))
    if expected_serial is not None and serial != str(expected_serial):
        raise ValueError(f'{settings_xml} {stream_name}: serial {serial} != expected {expected_serial}')
    dci = np.asarray(pr.device_channel_indices)
    pos = np.asarray(pr.contact_positions, float)
    g = pd.DataFrame({'ch': dci, 'x': pos[:, 0], 'y': pos[:, 1],
                      'shank': np.asarray(pr.shank_ids).astype(int),
                      'contact_id': np.asarray(pr.contact_ids)})
    g = g.sort_values('ch').reset_index(drop=True)
    g.attrs['serial'] = serial
    return g


class Anatomy:
    """Contact labels for one mouse under the current fit and the mirror, keyed by (x, y)."""

    def __init__(self, table: pd.DataFrame):
        self.t = table.reset_index(drop=True)
        keys = list(zip(self.t['x'].astype(float).round(1), self.t['y'].astype(float).round(1)))
        self._maps = {c: dict(zip(keys, self.t[c].values)) for c in self.t.columns}

    def lookup(self, x, y, column, default=np.nan):
        keys = zip(np.round(np.asarray(x, float), 1), np.round(np.asarray(y, float), 1))
        m = self._maps[column]
        return np.array([m.get(k, default) for k in keys], dtype=object)

    def labels(self, x, y, mapping='current'):
        """(label, tissue) arrays for probe positions under 'current', 'mirror' or a shank permutation."""
        x = np.asarray(x, float)
        if mapping in ('current', 'mirror'):
            return (self.lookup(x, y, f'label_{mapping}', 'outside'),
                    self.lookup(x, y, f'tissue_{mapping}', 'outside'))
        perm = np.asarray(mapping)
        s = shank_of(x)
        x2 = x - SHANK_PITCH_UM * s + SHANK_PITCH_UM * perm[s]
        return (self.lookup(x2, y, 'label_current', 'outside'),
                self.lookup(x2, y, 'tissue_current', 'outside'))

    def signed_distance(self, x, y, mapping='current'):
        return self.lookup(x, y, f'd_signed_{mapping}').astype(float)


def hpf_class(labels, tissue):
    """True = ripple structure, False = other grey matter, None = excluded (fibre, DG, ventricle, outside)."""
    out = []
    for lab, tis in zip(labels, tissue):
        if tis != 'grey' or not isinstance(lab, str):
            out.append(None)
        elif lab in RIPPLE_SET:
            out.append(True)
        elif lab.startswith('DG'):
            out.append(None)
        else:
            out.append(False)
    return np.array(out, dtype=object)


def ec_class(labels, tissue):
    """'ENTm', 'ENTl' or None."""
    out = []
    for lab, tis in zip(labels, tissue):
        if tis == 'grey' and isinstance(lab, str) and lab[:4] in ('ENTm', 'ENTl'):
            out.append(lab[:4])
        else:
            out.append(None)
    return np.array(out, dtype=object)


def build_contact_anatomy(mouse: str) -> tuple[pd.DataFrame, dict]:
    """Stage 0 (histology env): every contact's label under the current fit and the exact mirror.

    Returns (table, gates). Columns: mouse, x, y, shank, col, in_bank, label/tissue/id for
    current and mirror, d_lec_mec_um (bank only, unsigned, a property of physical position) and
    d_signed_{current,mirror} (+ inside ENTm, - inside ENTl, NaN elsewhere).
    """
    import sys
    sys.path.insert(0, str(REPO / 'code' / 'histology_refit'))
    import probe_refit as pr
    import registration_qc as rq

    fit = pr.load_fit(mouse)
    plane, params = pr.split_fit(fit)
    plane_m, params_m = dict(plane), dict(params)
    # the exact degeneracy (probe_refit.enforce_shank_order): all four move together
    plane_m['u_axis'] = -np.asarray(plane['u_axis'], float)
    plane_m['normal'] = -np.asarray(plane['normal'], float)
    params_m['offset_x'] = -float(params['offset_x'])
    params_m['theta'] = -float(params['theta'])

    data = pr.load_volumes(mouse, fast=True)
    cur = pr.project_probe(plane, params, data)
    mir = pr.project_probe(plane_m, params_m, data)
    pr._VOLUME_CACHE.clear()

    tree = rq.structure_tree()
    fibre_root = tree['acro2id'].get('fiber tracts', 1009)
    vent_root = tree['acro2id'].get('VS', 73)

    def tissue(ids, inside):
        out = []
        for i, ok in zip(ids, inside):
            if not ok or not np.isfinite(float(i if i is not None else np.nan)):
                out.append('outside')
                continue
            anc = set(tree['id2anc'].get(int(i), []))
            out.append('fibre' if fibre_root in anc else 'ventricle' if vent_root in anc else 'grey')
        return out

    def frame(df, tag):
        ids = pd.to_numeric(df['structure.id'], errors='coerce').values
        return pd.DataFrame({
            'x': df['probe_coords.x'].round(1).values, 'y': df['probe_coords.y'].round(1).values,
            f'label_{tag}': df['structure.acronym'].values, f'id_{tag}': ids,
            f'tissue_{tag}': tissue(ids, df['inside_volume'].values),
            f'i_{tag}': df['downsample_coords.i'].values, f'j_{tag}': df['downsample_coords.j'].values,
            f'k_{tag}': df['downsample_coords.k'].values})

    a, b = frame(cur, 'current'), frame(mir, 'mirror')
    t = a.merge(b, on=['x', 'y'], how='outer', validate='one_to_one')
    t.insert(0, 'mouse', mouse)
    t['shank'] = shank_of(t['x'])
    t['col'] = ((t['x'] % SHANK_PITCH_UM) > 0).astype(int)
    t['in_bank'] = t['y'] <= 705.0

    gates = {}

    def lab(v):
        # pandas 3 keeps NaN through astype(str), so compare missing explicitly (above-brain contacts)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) or v == 'nan' else str(v)

    # the htsv is what unit_regions / channel_regions were built from; current must equal it
    htsv = pd.read_csv(pr.subject_dir(mouse) / 'ProbeA_anatomy.htsv', sep='\t')
    hk = {(x, y): lab(a) for x, y, a in zip(htsv['probe_coords.x'].round(1), htsv['probe_coords.y'].round(1),
                                            htsv['structure.acronym'])}
    same = [(x, y) in hk and hk[(x, y)] == lab(l) for x, y, l in zip(t['x'], t['y'], t['label_current'])]
    gates['current_vs_htsv'] = float(np.mean(same))
    # mirror position of (x, y) == current position of (782 - x, y): the degeneracy itself
    ck = {(x, y): (i, j, k) for x, y, i, j, k in
          zip(t['x'], t['y'], t['i_current'], t['j_current'], t['k_current'])}
    dev, agree = [], []
    lab_c = {(x, y): lab(l) for x, y, l in zip(t['x'], t['y'], t['label_current'])}
    for x, y, i, j, k, lm in zip(t['x'], t['y'], t['i_mirror'], t['j_mirror'], t['k_mirror'],
                                 t['label_mirror']):
        twin = (round(MIRROR_X_UM - x, 1), y)
        if twin in ck:
            dev.append(np.linalg.norm(np.subtract((i, j, k), ck[twin])))
            agree.append(lab_c[twin] == lab(lm))
    gates['mirror_position_dev_vox_max'] = float(np.max(dev)) if dev else np.nan
    gates['mirror_label_vs_twin_lookup'] = float(np.mean(agree)) if agree else np.nan

    dist = rq.contact_boundary_distances(mouse)
    dk = dict(zip(zip(dist['probe_coords.x'].round(1), dist['probe_coords.y'].round(1)),
                  dist['d_lec_mec_um']))
    t['d_lec_mec_um'] = [dk.get((x, y), np.nan) for x, y in zip(t['x'], t['y'])]
    twin_d = [dk.get((round(MIRROR_X_UM - x, 1), y), np.nan) for x, y in zip(t['x'], t['y'])]
    for tag, d in (('current', t['d_lec_mec_um'].values), ('mirror', np.asarray(twin_d))):
        lab = t[f'label_{tag}'].astype(str)
        sign = np.where(lab.str.startswith('ENTm'), 1.0, np.where(lab.str.startswith('ENTl'), -1.0, np.nan))
        t[f'd_signed_{tag}'] = sign * d
    return t, gates


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def window_grid(n, fs, window_s):
    w = int(round(window_s * fs))
    return np.arange(0, n - w + 1, w), w


def window_psd(x, fs, starts, w, stat='mean'):
    """Hann periodogram over windows, mean or median across windows: (freqs, (channels, freqs)).

    'median' resists rare artifact windows: in ly06 s4 a few running windows carried 8x the 1-4 Hz
    power of still ones while log 1-4 Hz power did not track speed (rho -0.07). For a stationary
    signal the median is ln2 x the mean at every frequency, so excess and ratio measures agree.
    """
    taper = signal.windows.hann(w, sym=False).astype(np.float32)
    scale = 1.0 / (fs * float((taper ** 2).sum()))
    if stat == 'median':
        out = np.empty((x.shape[0], w // 2 + 1))
        for c0 in range(0, x.shape[0], 32):
            seg = np.stack([np.asarray(x[c0:c0 + 32, s:s + w], np.float32) for s in starts], axis=0)
            seg = seg - seg.mean(axis=2, keepdims=True)
            p = np.abs(np.fft.rfft(seg * taper, axis=2)) ** 2 * scale
            p[:, :, 1:-1] *= 2
            out[c0:c0 + 32] = np.median(p, axis=0)
        return np.fft.rfftfreq(w, 1.0 / fs), out
    acc = np.zeros((x.shape[0], w // 2 + 1))
    for s in starts:
        seg = np.asarray(x[:, s:s + w], np.float32)
        seg = seg - seg.mean(axis=1, keepdims=True)
        p = np.abs(np.fft.rfft(seg * taper, axis=1)) ** 2 * scale
        p[:, 1:-1] *= 2
        acc += p
    return np.fft.rfftfreq(w, 1.0 / fs), acc / max(len(starts), 1)


def per_window_band_ratio(x1d, fs, starts, w, num, den):
    """Band-power ratio num/den per window of a 1-D signal (state segmentation)."""
    f = np.fft.rfftfreq(w, 1.0 / fs)
    taper = signal.windows.hann(w, sym=False)
    segs = np.stack([x1d[s:s + w] - x1d[s:s + w].mean() for s in starts]) * taper
    p = np.abs(np.fft.rfft(segs, axis=1)) ** 2
    mn = (f >= num[0]) & (f <= num[1])
    md = (f >= den[0]) & (f <= den[1])
    return p[:, mn].mean(axis=1) / np.maximum(p[:, md].mean(axis=1), 1e-30)


def aperiodic_fit(f, P, frange, exclude):
    """Line in log10-log10 over frange minus exclude, one outlier pass -> (offset, exponent) per channel."""
    m = (f >= frange[0]) & (f <= frange[1])
    for lo, hi in exclude:
        m &= ~((f >= lo) & (f <= hi))
    lf = np.log10(f[m])
    lp = np.log10(np.maximum(P[:, m], 1e-30))
    A = np.vstack([np.ones_like(lf), lf]).T
    coef = np.linalg.lstsq(A, lp.T, rcond=None)[0]
    # drop bins far above the line (beta/gamma bumps) and refit once
    resid = lp - (A @ coef).T
    sd = 1.4826 * np.median(np.abs(resid - np.median(resid, axis=1, keepdims=True)), axis=1, keepdims=True)
    ok = resid < 2.5 * np.maximum(sd, 1e-6)
    off, ex = np.empty(len(P)), np.empty(len(P))
    for i in range(len(P)):
        c = np.linalg.lstsq(A[ok[i]], lp[i, ok[i]], rcond=None)[0] if ok[i].sum() > 3 else coef[:, i]
        off[i], ex[i] = c[0], -c[1]
    return off, ex


def band_excess_db(f, P, off, ex, band):
    """Mean dB above the aperiodic line over a band, and the frequency of the maximum."""
    m = (f >= band[0]) & (f <= band[1])
    ap = off[:, None] - ex[:, None] * np.log10(f[m])[None]
    resid = 10.0 * (np.log10(np.maximum(P[:, m], 1e-30)) - ap)
    return resid.mean(axis=1), f[m][resid.argmax(axis=1)]


def band_power(f, P, band):
    m = (f >= band[0]) & (f <= band[1])
    return P[:, m].sum(axis=1) * (f[1] - f[0])


def spectral_summary(f, P, cfg: LFPConfig, prefix=''):
    """Theta and ripple measures from a PSD block -> dict of per-channel arrays."""
    off, ex = aperiodic_fit(f, P, cfg.ap_range, cfg.ap_exclude)
    th, th_pk = band_excess_db(f, P, off, ex, cfg.theta_band)
    roff, rex = aperiodic_fit(f, P, cfg.ripple_ap_range, cfg.ripple_ap_exclude)
    rip, _ = band_excess_db(f, P, roff, rex, cfg.ripple_band)
    tot = band_power(f, P, cfg.total_band)
    return {f'{prefix}theta_excess_db': th, f'{prefix}theta_peak_hz': th_pk,
            f'{prefix}rel_theta': band_power(f, P, cfg.theta_band) / np.maximum(tot, 1e-30),
            f'{prefix}theta_delta': band_power(f, P, cfg.theta_band)
            / np.maximum(band_power(f, P, cfg.delta_band), 1e-30),
            f'{prefix}ap_exponent': ex, f'{prefix}ripple_excess_db': rip,
            f'{prefix}mua_power': band_power(f, P, cfg.mua_band)}


def bipolar_pairs(x, y, rows):
    """Index pairs (i, j): j sits `rows` rows above i on the same shank and column."""
    key = {(round(a, 1), round(b, 1)): k for k, (a, b) in enumerate(zip(x, y))}
    pairs = []
    for i, (a, b) in enumerate(zip(x, y)):
        j = key.get((round(a, 1), round(b + rows * ROW_PITCH_UM, 1)))
        if j is not None:
            pairs.append((i, j))
    return np.array(pairs, int).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Ripples
# ---------------------------------------------------------------------------

def bandpass(x, fs, band, order=3):
    sos = signal.butter(order, band, btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, np.asarray(x, np.float64), axis=-1)


def band_envelope(x, fs, band, smooth_ms, block=32):
    """Smoothed Hilbert envelope per channel, float32, in channel blocks to bound memory."""
    x = np.atleast_2d(x)
    out = np.empty(x.shape, np.float32)
    for c0 in range(0, len(x), block):
        xf = bandpass(x[c0:c0 + block], fs, band)
        env = np.abs(signal.hilbert(xf, axis=-1))
        out[c0:c0 + block] = gaussian_filter1d(env, smooth_ms * fs / 1000.0, axis=-1)
    return out


def robust_z(env, mask):
    v = env[mask] if mask is not None and mask.any() else env
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return (env - med) / max(mad, 1e-12)


def detect_events(z, fs, valid, cfg: LFPConfig, z_guard=None):
    """(n, 3) int [start, peak, end]: z > lo for min..max ms with a peak > hi inside `valid`.

    z_guard: a broadband (MUA) z trace; events whose guard z at the peak reaches the ripple z
    are spike bursts, not oscillations, and are dropped.
    """
    above = np.concatenate([[False], z > cfg.ripple_z_lo, [False]])
    d = np.diff(above.astype(np.int8))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if not len(st):
        return np.zeros((0, 3), int)
    gap = int(cfg.ripple_merge_ms * fs / 1000)
    ms, me = [st[0]], [en[0]]
    for s, e in zip(st[1:], en[1:]):
        if s - me[-1] <= gap:
            me[-1] = e
        else:
            ms.append(s)
            me.append(e)
    ev = []
    lo_n, hi_n = cfg.ripple_min_ms * fs / 1000, cfg.ripple_max_ms * fs / 1000
    for s, e in zip(ms, me):
        if not (lo_n <= e - s <= hi_n):
            continue
        pk = s + int(np.argmax(z[s:e]))
        if z[pk] < cfg.ripple_z_hi or (valid is not None and not valid[pk]):
            continue
        if z_guard is not None and z_guard[pk] >= z[pk]:
            continue
        ev.append((s, pk, e))
    return np.array(ev, int).reshape(-1, 3)


def triggered_gain(env, peaks, half, baseline):
    """Mean envelope within +-half samples of event peaks over each channel's baseline."""
    if len(peaks) == 0:
        return np.full(len(env), np.nan)
    idx = peaks[:, None] + np.arange(-half, half + 1)[None]
    idx = idx[(idx.min(axis=1) >= 0) & (idx.max(axis=1) < env.shape[1])]
    return env[:, idx].mean(axis=(1, 2)) / np.maximum(baseline, 1e-12)


def ripple_maps(x, fs, rest_mask, rip_excess, cfg: LFPConfig, exclude_ref=None, block=32):
    """Reference-channel ripples -> per-channel gain; per-channel own event rate with a MUA guard.

    x: (channels, samples) monopolar µV. rest_mask: (samples,) bool. rip_excess: per-channel
    rest-state ripple-band excess (dB) used to choose the reference, which keeps broadband MUA
    channels from being picked. Envelopes are made in channel blocks (a 42 min session holds
    4.9 GB per 384-channel float32 array). Returns dict with gain, rate_hz, ref index, events.
    """
    cand = np.where(np.isfinite(rip_excess), rip_excess, -np.inf)
    if exclude_ref is not None:
        cand[exclude_ref] = -np.inf
    ref = int(np.argmax(cand))
    env_ref = band_envelope(x[ref:ref + 1], fs, cfg.ripple_band, cfg.ripple_smooth_ms)[0]
    mua_ref = band_envelope(x[ref:ref + 1], fs, cfg.mua_band, cfg.ripple_smooth_ms)[0]
    ev = detect_events(robust_z(env_ref, rest_mask), fs, rest_mask, cfg,
                       z_guard=robust_z(mua_ref, rest_mask))
    half = int(cfg.ripple_half_win_ms * fs / 1000)
    rest_s = rest_mask.sum() / fs
    gain = np.full(len(x), np.nan)
    rate = np.empty(len(x))
    for c0 in range(0, len(x), block):
        env = band_envelope(x[c0:c0 + block], fs, cfg.ripple_band, cfg.ripple_smooth_ms)
        mua = band_envelope(x[c0:c0 + block], fs, cfg.mua_band, cfg.ripple_smooth_ms)
        base = np.array([np.median(e[rest_mask]) for e in env])
        if len(ev) >= cfg.min_ripples:
            gain[c0:c0 + block] = triggered_gain(env, ev[:, 1], half, base)
        for k in range(len(env)):
            n = len(detect_events(robust_z(env[k], rest_mask), fs, rest_mask, cfg,
                                  z_guard=robust_z(mua[k], rest_mask)))
            rate[c0 + k] = n / max(rest_s, 1e-9)
    return {'gain': gain, 'rate_hz': rate, 'ref': ref, 'events': ev, 'n_events': len(ev)}


def probe_median(x, chunk=1_000_000):
    """Median across channels per sample, in time chunks (np.median copies its input)."""
    out = np.empty(x.shape[1], np.float32)
    for s in range(0, x.shape[1], chunk):
        out[s:s + chunk] = np.median(x[:, s:s + chunk], axis=0)
    return out


# ---------------------------------------------------------------------------
# Orientation statistic
# ---------------------------------------------------------------------------

def auc(score, positive):
    """Mann-Whitney AUC of score for positive (True) vs negative (False); None entries excluded."""
    score = np.asarray(score, float)
    pos = np.array([p is True for p in positive])
    neg = np.array([p is False for p in positive])
    ok = np.isfinite(score)
    pos, neg = pos & ok, neg & ok
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score[pos | neg])
    is_pos = pos[pos | neg]
    return float((r[is_pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def verdict(a_cur, a_mir, cfg: LFPConfig):
    """'current' / 'mirror' / 'inconclusive'. A margin, not a ceiling, on the loser: when the mirror
    puts ripple labels on noise-level contacts its AUC sits at 0.45-0.50 by construction (0.494 in
    synthetic check 1), so 'loser <= 0.5' would flip a correct verdict on an unlucky draw."""
    if not (np.isfinite(a_cur) and np.isfinite(a_mir)):
        return 'inconclusive'
    if a_cur >= cfg.auc_support and a_cur - a_mir >= cfg.auc_margin:
        return 'current'
    if a_mir >= cfg.auc_support and a_mir - a_cur >= cfg.auc_margin:
        return 'mirror'
    return 'inconclusive'


def orientation_test(x, y, score, anatomy: Anatomy, cfg: LFPConfig, keep=None):
    """AUC of score for ripple structures vs other grey matter, current vs mirror, plus 24 permutations."""
    import itertools
    keep = np.ones(len(x), bool) if keep is None else np.asarray(keep, bool)
    out = {}
    for mapping in ('current', 'mirror'):
        lab, tis = anatomy.labels(x, y, mapping)
        cls = hpf_class(lab, tis)
        cls[~keep] = None
        out[f'auc_{mapping}'] = auc(score, cls)
        out[f'n_pos_{mapping}'] = int(sum(c is True for c in cls))
        out[f'n_neg_{mapping}'] = int(sum(c is False for c in cls))
    perms = {}
    for p in itertools.permutations(range(4)):
        lab, tis = anatomy.labels(x, y, p)
        cls = hpf_class(lab, tis)
        cls[~keep] = None
        perms[p] = auc(score, cls)
    vals = np.array([v for v in perms.values() if np.isfinite(v)])
    out['perm_aucs'] = perms
    out['rank_current'] = int((vals > out['auc_current']).sum() + 1) if np.isfinite(out['auc_current']) else -1
    out['rank_reversed'] = int((vals > perms[(3, 2, 1, 0)]).sum() + 1) if np.isfinite(perms[(3, 2, 1, 0)]) else -1
    out['verdict'] = verdict(out['auc_current'], out['auc_mirror'], cfg)
    return out


def discordant_test(x, y, score, anatomy: Anatomy, strata=None, keep=None, n_perm=1000, seed=0):
    """Post-hoc orientation test on the contacts whose class differs between the two mappings.

    A = ripple structure under current but other grey matter under the mirror; B = the reverse.
    Contacts labelled the same both ways cannot discriminate and only dilute the per-mapping AUCs
    (ly06 probe map: 471 vs 453 ripple contacts, mostly the same ones). AUC(A > B) > 0.5 favours
    current. Ranks are taken within strata (bank / recording: ripple gain has one reference per
    stratum), combined as sum(U) / sum(nA nB); p from shuffling A/B within strata.
    """
    score = np.asarray(score, float)
    keep = np.ones(len(x), bool) if keep is None else np.asarray(keep, bool)
    strata = np.zeros(len(x), int) if strata is None else np.asarray(strata)
    cc = hpf_class(*anatomy.labels(x, y, 'current'))
    cm = hpf_class(*anatomy.labels(x, y, 'mirror'))
    a = np.array([p is True and q is False for p, q in zip(cc, cm)]) & keep & np.isfinite(score)
    b = np.array([p is False and q is True for p, q in zip(cc, cm)]) & keep & np.isfinite(score)

    def stat(a_, b_):
        num = den = 0.0
        for s in np.unique(strata[a_ | b_]):
            ia, ib = a_ & (strata == s), b_ & (strata == s)
            na, nb = int(ia.sum()), int(ib.sum())
            if na == 0 or nb == 0:
                continue
            r = rankdata(score[ia | ib])
            isa = ia[ia | ib]
            num += r[isa].sum() - na * (na + 1) / 2
            den += na * nb
        return num / den if den else np.nan

    obs = stat(a, b)
    rng = np.random.default_rng(seed)
    null = []
    if np.isfinite(obs):
        idx = np.flatnonzero(a | b)
        for _ in range(n_perm):
            a2, b2 = np.zeros(len(x), bool), np.zeros(len(x), bool)
            for s in np.unique(strata[idx]):
                k = idx[strata[idx] == s]
                lab = np.r_[np.ones(int(a[k].sum()), bool), np.zeros(int(b[k].sum()), bool)]
                lab = rng.permutation(lab)
                a2[k[lab]], b2[k[~lab]] = True, True
            null.append(stat(a2, b2))
    null = np.asarray(null)
    p_cur = float((1 + np.sum(null >= obs)) / (1 + len(null))) if len(null) else np.nan
    p_mir = float((1 + np.sum(null <= obs)) / (1 + len(null))) if len(null) else np.nan
    return {'auc_discordant': obs, 'n_current_only': int(a.sum()), 'n_mirror_only': int(b.sum()),
            'p_favours_current': p_cur, 'p_favours_mirror': p_mir}


def ec_theta_test(x, y, score, anatomy: Anatomy, keep=None):
    """AUC of score for ENTm vs ENTl (all layers, and layer 5 only), under current and mirror."""
    keep = np.ones(len(x), bool) if keep is None else np.asarray(keep, bool)
    out = {}
    for mapping in ('current', 'mirror'):
        lab, tis = anatomy.labels(x, y, mapping)
        ec = ec_class(lab, tis)
        cls = np.array([None if not k else True if e == 'ENTm' else False if e == 'ENTl' else None
                        for e, k in zip(ec, keep)], dtype=object)
        l5 = np.array([None if not k else True if l == 'ENTm5' else False if l == 'ENTl5' else None
                       for l, k in zip(lab, keep)], dtype=object)
        out[f'auc_entm_entl_{mapping}'] = auc(score, cls)
        out[f'auc_l5_{mapping}'] = auc(score, l5)
        out[f'n_entm_{mapping}'] = int(sum(c is True for c in cls))
        out[f'n_entl_{mapping}'] = int(sum(c is False for c in cls))
        out[f'n_entm5_{mapping}'] = int(sum(c is True for c in l5))
        out[f'n_entl5_{mapping}'] = int(sum(c is False for c in l5))
    return out


# ---------------------------------------------------------------------------
# Probe-map LFP (Peter's extraction, re-mapped)
# ---------------------------------------------------------------------------

def probemap_paths(mouse, config, root=PROBEMAP_ROOT):
    proc = Path(root) / 'processed_data' / mouse / config
    raw = sorted(p for p in (Path(root) / 'raw_data' / 'ephys' / mouse / config).glob('20*/Record Node *')
                 if p.is_dir())
    if len(raw) != 1:
        raise FileNotFoundError(f'{mouse} {config}: expected one raw Record Node folder, found {raw}')
    rec = raw[0] / 'experiment1' / 'recording1'
    streams = sorted((rec / 'continuous').glob('Neuropix-PXI-*'))
    if len(streams) != 1:
        raise ValueError(f'{mouse} {config}: {len(streams)} streams; the probe-map loader needs exactly one')
    return {'proc': proc, 'settings': raw[0] / 'settings.xml', 'oebin': rec / 'structure.oebin',
            'dat': streams[0] / 'continuous.dat', 'stream': streams[0].name.split('.')[-1]}


def load_probemap(mouse, config, root=PROBEMAP_ROOT):
    """lfp (n, 192) float16 memmap, fs, and per-column (ch, x, y, shank) from probeinterface.

    The column holds raw channel CH<label> (Peter's select_channels by label); its position is
    whatever settings.xml says for that channel, NOT lfp.metrics x/y.
    """
    p = probemap_paths(mouse, config, root)
    met = pd.read_csv(p['proc'] / 'lfp.metrics.htsv', sep='\t')
    lfp = np.load(p['proc'] / 'lfp.signal.npy', mmap_mode='r')
    if lfp.shape[1] != len(met):
        raise ValueError(f'{mouse} {config}: {lfp.shape[1]} columns vs {len(met)} metrics rows')
    fs = float(met['sampling_rate'].iloc[0])
    geo = probe_geometry(p['settings'], p['stream'], SERIALS.get(mouse)).set_index('ch')
    ch = met['contact.label'].str.replace('CH', '').astype(int).values
    cols = pd.DataFrame({'col': np.arange(len(ch)), 'ch': ch,
                         'x': geo.loc[ch, 'x'].values, 'y': geo.loc[ch, 'y'].values,
                         'shank': geo.loc[ch, 'shank'].values,
                         'metrics_x': met['contact.x'].values, 'metrics_y': met['contact.y'].values})
    return lfp, fs, cols, p


def read_oebin(oebin, stream):
    o = json.loads(Path(oebin).read_text())
    hit = [c for c in o['continuous'] if c['folder_name'].rstrip('/').endswith(stream)]
    if len(hit) != 1:
        raise ValueError(f'{oebin}: stream {stream} matched {len(hit)} entries')
    c = hit[0]
    names = [ch['channel_name'] for ch in c['channels']]
    return float(c['sample_rate']), int(c['num_channels']), float(c['channels'][0]['bit_volts']), names


def verify_probemap_columns(lfp, fs, cols, paths, t0_s=300.0, dur_s=60.0, band=(100.0, 400.0)):
    """Gate A0: recompute each column from the raw .dat; its best match must be its own channel.

    Compared in 100-400 Hz, where neighbouring contacts decorrelate (LFP below ~50 Hz is nearly
    identical across 15 um and would accept a wrong neighbour). Returns per-column DataFrame.
    """
    fs_raw, n_ch, uv, names = read_oebin(paths['oebin'], paths['stream'])
    q = int(round(fs_raw / fs))
    n_raw = os.path.getsize(paths['dat']) // (2 * n_ch)
    mm = np.memmap(paths['dat'], np.int16, 'r', shape=(n_raw, n_ch))
    a = int(t0_s * fs_raw) // q * q
    b = a + int(dur_s * fs_raw) // q * q
    if b > n_raw:
        a, b = 0, min(n_raw, int(dur_s * fs_raw)) // q * q
    name_col = {n: i for i, n in enumerate(names)}
    raw_cols = [name_col[f'CH{c}'] for c in cols['ch']]
    block = np.asarray(mm[a:b, raw_cols], np.float32).T * uv
    mine = signal.resample_poly(bandpass(block, fs_raw, band), 1, q, axis=1)
    theirs = bandpass(np.asarray(lfp[a // q: a // q + mine.shape[1]], np.float32).T, fs, band)
    edge = int(2 * fs)
    n = min(mine.shape[1], theirs.shape[1])

    def zs(M):
        return (M - M.mean(1, keepdims=True)) / np.maximum(M.std(1, keepdims=True), 1e-12)

    # one global lag: a resampler offset of a sample would decorrelate 400 Hz on every column
    lags = range(-5, 6)
    score = [np.mean(np.sum(zs(mine[:, edge:n - edge]) * zs(theirs[:, edge + L:n - edge + L]), axis=1))
             for L in lags]
    lag = list(lags)[int(np.argmax(score))]
    A = zs(mine[:, edge:n - edge])
    B = zs(theirs[:, edge + lag:n - edge + lag])
    R = B @ A.T / A.shape[1]                # R[k, j] = corr(Peter column k, raw channel of column j)
    best = R.argmax(axis=1)
    srt = np.sort(R, axis=1)
    r_own = np.diag(R)
    # flat channels (Peter's qc 'dead') correlate with nothing; exclude them, do not fail the config
    dead = (mine.std(axis=1) < 1e-3) | (theirs.std(axis=1) < 1e-3)
    # a ratio, not an absolute margin: in ProS/SUB the next contact reaches r 0.95 in 100-400 Hz
    # (ly06 config_1: own 0.998, second 0.95 on 39 columns), so a 0.05 margin rejected true matches
    ok = (best == np.arange(len(R))) & (r_own >= 0.98) & ((1 - srt[:, -2]) >= 5 * (1 - r_own))
    return pd.DataFrame({'col': cols['col'], 'ch': cols['ch'], 'r_own': r_own,
                         'best_col': best, 'r_second': srt[:, -2], 'lag_samples': lag,
                         'dead': dead, 'ok': np.where(dead, True, ok)})


def analyse_lfp_block(x, fs, pos, cfg: LFPConfig, theta_win=None, rest_win=None, rest_mask=None):
    """Per-channel theta/ripple measures for one continuous block; x (channels, samples) µV.

    pos: DataFrame with x, y (one row per channel). theta_win/rest_win: window start indices
    (if None, chosen from the probe-median theta/delta ratio). Returns (channel DataFrame, psd dict).
    """
    starts, w = window_grid(x.shape[1], fs, cfg.window_s)
    if theta_win is None or rest_win is None:
        xmed = probe_median(x)
        r = per_window_band_ratio(xmed, fs, starts, w, cfg.theta_band, cfg.delta_band)
        theta_win = starts[r >= np.quantile(r, 1 - cfg.theta_state_frac)]
        rest_win = starts[r <= np.quantile(r, cfg.rest_state_frac)]
    if rest_mask is None:
        rest_mask = np.zeros(x.shape[1], bool)
        for s in rest_win:
            rest_mask[s:s + w] = True
    out = pos[['x', 'y']].reset_index(drop=True).copy()
    psd = {}
    f, Pt = window_psd(x, fs, theta_win, w, cfg.psd_stat)
    _, Pr = window_psd(x, fs, rest_win, w, cfg.psd_stat)
    psd['f'], psd['theta'], psd['rest'] = f, Pt.astype(np.float32), Pr.astype(np.float32)
    # abnormal channels: flat (SD 0; e.g. ly06 config_1 CH290), or LFP with no >300 Hz power at all
    # (ah08 config_3 CH120: 300-450 Hz power 2e-6 vs ~55 on its CA1 neighbours). The latter's steep
    # spectrum fakes a ripple-band 'excess' and won the reference, so both are excluded everywhere.
    sd = np.asarray(x[:, :int(min(x.shape[1], 120 * fs))], np.float32).std(axis=1)
    mua = band_power(f, Pr, cfg.mua_band)
    dead = (sd < 0.05 * np.median(sd)) | (mua < 0.01 * np.median(mua))
    out['dead'] = dead
    for k, v in spectral_summary(f, Pt, cfg, 'th_').items():
        out[k] = np.where(dead, np.nan, v)
    for k, v in spectral_summary(f, Pr, cfg, 'rest_').items():
        out[k] = np.where(dead, np.nan, v)
    pairs = bipolar_pairs(out['x'].values, out['y'].values, cfg.bipolar_rows)
    bip = pd.DataFrame({'x': out['x'].values[pairs[:, 0]],
                        'y': (out['y'].values[pairs[:, 0]] + out['y'].values[pairs[:, 1]]) / 2,
                        'i': pairs[:, 0], 'j': pairs[:, 1]})
    if len(pairs):
        Pbt = np.empty((len(pairs), len(f)))
        Pbr = np.empty((len(pairs), len(f)))
        for p0 in range(0, len(pairs), 64):
            pp = pairs[p0:p0 + 64]
            xb = np.asarray(x[pp[:, 1]], np.float32) - np.asarray(x[pp[:, 0]], np.float32)
            Pbt[p0:p0 + 64] = window_psd(xb, fs, theta_win, w, cfg.psd_stat)[1]
            Pbr[p0:p0 + 64] = window_psd(xb, fs, rest_win, w, cfg.psd_stat)[1]
        bad = dead[pairs[:, 0]] | dead[pairs[:, 1]]
        for k, v in spectral_summary(f, Pbt, cfg, 'th_').items():
            bip[k] = np.where(bad, np.nan, v)
        for k, v in spectral_summary(f, Pbr, cfg, 'rest_').items():
            bip[k] = np.where(bad, np.nan, v)
        psd['bip_theta'] = Pbt.astype(np.float32)
    rip = ripple_maps(x, fs, rest_mask, out['rest_ripple_excess_db'].values, cfg)
    out['ripple_gain'] = np.where(dead, np.nan, rip['gain'])
    out['ripple_rate_hz'] = np.where(dead, np.nan, rip['rate_hz'])
    out['is_ref'] = np.arange(len(out)) == rip['ref']
    meta = {'n_theta_win': len(theta_win), 'n_rest_win': len(rest_win), 'ref': rip['ref'],
            'n_events': rip['n_events'], 'events': rip['events'], 'window_samples': w,
            'theta_win': np.asarray(theta_win), 'rest_win': np.asarray(rest_win)}
    return out, bip, psd, meta


def mask_from_windows(starts, w, n):
    m = np.zeros(n, bool)
    for s in starts:
        m[s:s + w] = True
    return m


def analyse_probemap_config(mouse, config, cfg: LFPConfig, root=PROBEMAP_ROOT, gate=True):
    """One probe-map config -> (channels, bipolar, psd, meta). Gate A0 runs first when gate=True."""
    lfp, fs, cols, paths = load_probemap(mouse, config, root)
    meta = {'mouse': mouse, 'config': config, 'fs': fs}
    if gate:
        g = verify_probemap_columns(lfp, fs, cols, paths)
        meta['gate_a0'] = g
        meta['gate_a0_pass'] = bool(g['ok'].all())
        meta['metrics_xy_agree'] = float(((cols['x'] == cols['metrics_x'])
                                          & (cols['y'] == cols['metrics_y'])).mean())
    x = np.asarray(lfp, np.float32).T
    ch, bip, psd, m = analyse_lfp_block(x, fs, cols, cfg)
    for c in ('ch', 'shank', 'col'):
        ch[c] = cols[c].values
    ch.insert(0, 'config', config)
    ch.insert(0, 'mouse', mouse)
    bip.insert(0, 'config', config)
    bip.insert(0, 'mouse', mouse)
    meta.update(m)
    return ch, bip, psd, meta


# ---------------------------------------------------------------------------
# Raw OpenEphys recordings
# ---------------------------------------------------------------------------

class RawRecording:
    """Memory-mapped NP2.0 wideband stream with its probeinterface geometry."""

    def __init__(self, continuous_dat, expected_serial=None):
        dat = Path(continuous_dat)
        stream = dat.parent.name.split('.')[-1]
        rec_dir = dat.parents[2]
        node = rec_dir.parents[1]
        top = node.parent
        nested = [q for q in top.iterdir() if q.is_dir() and q.name[:2] == '20']
        if nested:
            # bp02 2024-03-19 held a nested copy that the sorter read instead of the real data
            raise RuntimeError(f'{top}: nested recording folder(s) {nested}')
        self.fs, self.n_ch, self.uv, self.names = read_oebin(rec_dir / 'structure.oebin', stream)
        self.n_samples = os.path.getsize(dat) // (2 * self.n_ch)
        self.data = np.memmap(dat, np.int16, 'r', shape=(self.n_samples, self.n_ch))
        self.geom = probe_geometry(node / 'settings.xml', stream, expected_serial)
        self.name_col = {n: i for i, n in enumerate(self.names)}
        self.path = dat

    def channel_table(self):
        g = self.geom.copy()
        g['name'] = [f'CH{c}' for c in g['ch']]
        g['col_in_dat'] = [self.name_col[n] for n in g['name']]
        return g


def decimate_lfp(rec: RawRecording, start, stop, fs_out, chunk_s=60.0, pad_s=2.0, ch_block=96):
    """(n_ch, n_out) float32 µV at fs_out from raw samples [start, stop); output j <-> start + j*q."""
    q = int(round(rec.fs / fs_out))
    if abs(q * fs_out - rec.fs) > 1e-6:
        raise ValueError(f'fs {rec.fs} not an integer multiple of {fs_out}')
    start = int(start) + (-int(start)) % q
    n_out = (int(stop) - start) // q
    out = np.empty((rec.n_ch, n_out), np.float32)
    chunk = int(chunk_s * rec.fs) // q * q
    pad = int(pad_s * rec.fs) // q * q
    for c0 in range(start, start + n_out * q, chunk):
        c1 = min(c0 + chunk, start + n_out * q)
        a = max(c0 - pad, start % q)
        b = min(c1 + pad, rec.n_samples)
        raw = np.asarray(rec.data[a:b])
        i0 = (c0 - a) // q
        n = (c1 - c0) // q
        for k in range(0, rec.n_ch, ch_block):
            # no per-chunk offset removal: a DC step at every chunk edge would leak broadband
            blk = raw[:, k:k + ch_block].astype(np.float32).T * rec.uv
            y = signal.resample_poly(blk, 1, q, axis=1)
            out[k:k + ch_block, (c0 - start) // q:(c0 - start) // q + n] = y[:, i0:i0 + n]
    return out, start


def verify_decimation(rec: RawRecording, lfp, lfp_start, fs_out, t0_s=300.0, dur_s=20.0):
    """Gate: our decimated LFP vs an FFT resample of the same raw window, r in 1-300 Hz per channel.

    20 s keeps the 384-channel FFT under ~4 GB transient.
    """
    q = int(round(rec.fs / fs_out))
    lfp_end = lfp_start + lfp.shape[1] * q      # the decimated span, which may stop before the file
    a = lfp_start + int(t0_s * rec.fs) // q * q
    b = a + int(dur_s * rec.fs) // q * q
    if b > lfp_end:
        a, b = lfp_start, lfp_start + min(lfp_end - lfp_start, int(dur_s * rec.fs)) // q * q
    raw = np.asarray(rec.data[a:b], np.float32).T * rec.uv
    ref = signal.resample(raw - raw.mean(1, keepdims=True), (b - a) // q, axis=1)
    j0 = (a - lfp_start) // q
    mine = lfp[:, j0:j0 + ref.shape[1]]
    edge = int(2 * fs_out)
    A = bandpass(mine, fs_out, (1.0, 300.0))[:, edge:-edge]
    B = bandpass(ref, fs_out, (1.0, 300.0))[:, edge:-edge]
    r = np.array([np.corrcoef(u, v)[0, 1] for u, v in zip(A, B)])
    return r


# ---------------------------------------------------------------------------
# Speed epochs on the ephys clock
# ---------------------------------------------------------------------------

def bin_to_sample(k, clock, t0_ms, bin_ms=25):
    """.dat sample of the start of 25 ms bin k (inverse of extract_neuron_raw.bin_session_fitted)."""
    return clock['a'] * (t0_ms + bin_ms * np.asarray(k, float)) + clock['b']


def sample_to_bin(s, clock, t0_ms, bin_ms=25):
    t_ms = (np.asarray(s, np.float64) - clock['b']) / clock['a']
    return np.floor(t_ms / bin_ms).astype(np.int64) - (t0_ms // bin_ms)


def epochs_from_mask(mask, min_len):
    """[(start, stop)) runs of True at least min_len long."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (en - st) >= min_len
    return np.stack([st[keep], en[keep]], axis=1) if keep.any() else np.zeros((0, 2), int)


def speed_windows(speed, clock, t0_ms, lfp_start, fs_raw, fs_lfp, w, cfg: LFPConfig, n_lfp):
    """LFP window starts that are entirely running / entirely still, and a still-sample mask."""
    q = int(round(fs_raw / fs_lfp))
    starts = np.arange(0, n_lfp - w + 1, w)
    # bin index at each window's first and last LFP sample
    s_first = lfp_start + starts * q
    s_last = lfp_start + (starts + w - 1) * q
    b0 = sample_to_bin(s_first, clock, t0_ms)
    b1 = sample_to_bin(s_last, clock, t0_ms)
    run_win, still_win = [], []
    sp = np.asarray(speed, float)
    for st, k0, k1 in zip(starts, b0, b1):
        if k0 < 0 or k1 >= len(sp):
            continue
        seg = sp[k0:k1 + 1]
        if not np.isfinite(seg).all():
            continue
        # fractions, not all-bins: maze runs last ~1 s (ly06 s4 median bout 1.05 s), so 'every
        # bin > 5 cm/s' kept 0.1 of 8.7 running min; these rules keep 5.7 min running, 3.2 min still
        if (seg > cfg.run_cm_s).mean() >= cfg.run_frac and seg.mean() > cfg.run_cm_s:
            run_win.append(st)
        elif (seg < cfg.still_cm_s).mean() >= cfg.still_frac and seg.max() < cfg.run_cm_s:
            still_win.append(st)
    still_mask = np.zeros(n_lfp, bool)
    for st in still_win:
        still_mask[st:st + w] = True
    return np.array(run_win, int), np.array(still_win, int), still_mask


# ---------------------------------------------------------------------------
# Unit measures
# ---------------------------------------------------------------------------

def acg_counts(t, epoch, bin_s, max_s):
    """One-sided autocorrelogram counts for lags in [0, max_s) over pairs in the same epoch; t sorted."""
    nb = int(round(max_s / bin_s))
    counts = np.zeros(nb, np.int64)
    n = len(t)
    for k in range(1, n):
        d = t[k:] - t[:-k]
        if d.min() >= max_s:
            break
        ok = (d < max_s) & (epoch[k:] == epoch[:-k])
        if ok.any():
            counts += np.bincount((d[ok] / bin_s).astype(np.int64), minlength=nb)[:nb]
    return counts


def theta_index(counts, cfg: LFPConfig):
    """(ratio, excess_db, peak Hz) from a one-sided ACG.

    ratio: Brandon et al. 2011 / Kropff et al. 2015, power within +-1 Hz of the 5-11 Hz peak over
    mean power 0-50 Hz; kept for comparison with the literature. excess_db: the same peak in dB
    above a smooth (quadratic in log f) fit to the ACG spectrum over 1-30 Hz minus 4-12 Hz. Bursts
    and after-hyperpolarisation give smooth spectra and are absorbed: against a jitter null the
    ratio called 35/40 bursty non-rhythmic synthetic units rhythmic, the excess 0/30.
    """
    bin_s = cfg.acg_bin_ms / 1000.0
    c = counts.astype(float)
    skip = int(round(cfg.acg_skip_ms / cfg.acg_bin_ms))
    if c[skip:].sum() <= 0:
        return np.nan, np.nan, np.nan
    c[:skip] = c[skip:].mean()
    c = (c - c.mean()) * signal.windows.hann(len(c), sym=False)
    nfft = 4096
    f = np.fft.rfftfreq(nfft, bin_s)
    P = np.abs(np.fft.rfft(c, nfft)) ** 2
    band = (f >= cfg.theta_search[0]) & (f <= cfg.theta_search[1])
    hw = cfg.theta_halfwidth_hz
    fpk = f[band][np.argmax(P[band])]
    ratio = P[(f >= fpk - hw) & (f <= fpk + hw)].mean() / max(P[(f > 0) & (f <= cfg.acg_spec_max_hz)].mean(), 1e-30)
    lp = np.log10(np.maximum(P, 1e-30))
    m = (f >= 1.0) & (f <= 30.0) & ~((f >= 4.0) & (f <= 12.0))
    lf = np.log10(f[m])
    coef = np.linalg.lstsq(np.vstack([np.ones_like(lf), lf, lf ** 2]).T, lp[m], rcond=None)[0]
    lfb = np.log10(f[band])
    resid = 10.0 * (lp[band] - (coef[0] + coef[1] * lfb + coef[2] * lfb ** 2))
    fx = f[band][np.argmax(resid)]
    w = (f[band] >= fx - hw) & (f[band] <= fx + hw)
    return float(ratio), float(resid[w].mean()), float(fx)


def unit_theta_rhythmicity(t, epoch, cfg: LFPConfig, rng):
    """Theta index of one unit's running spikes with a +-62.5 ms jitter null -> dict."""
    bin_s, max_s = cfg.acg_bin_ms / 1000.0, cfg.acg_max_ms / 1000.0
    n_all = len(t)
    empty = {'n_spikes': n_all, 'theta_excess_db': np.nan, 'theta_ratio': np.nan, 'theta_hz': np.nan,
             'p_jitter': np.nan, 'null_median_db': np.nan, 'burst_frac': np.nan}
    if n_all < cfg.min_spikes:
        return empty
    if n_all > cfg.max_spikes:
        keep = np.sort(rng.choice(n_all, cfg.max_spikes, replace=False))
        t, epoch = t[keep], epoch[keep]
    order = np.argsort(t, kind='stable')
    t, epoch = t[order], epoch[order]
    ratio, ex, fpk = theta_index(acg_counts(t, epoch, bin_s, max_s), cfg)
    null = np.empty(cfg.n_jitter)
    half = cfg.jitter_ms / 1000.0
    for i in range(cfg.n_jitter):
        tj = t + rng.uniform(-half, half, len(t))
        o = np.argsort(tj, kind='stable')
        null[i] = theta_index(acg_counts(tj[o], epoch[o], bin_s, max_s), cfg)[1]
    p = (1 + np.sum(null >= ex)) / (1 + cfg.n_jitter)
    isi = np.diff(t)[np.diff(epoch) == 0]
    return {'n_spikes': n_all, 'theta_excess_db': ex, 'theta_ratio': ratio, 'theta_hz': fpk,
            'p_jitter': float(p), 'null_median_db': float(np.median(null)),
            'burst_frac': float(np.mean(isi < 0.010)) if len(isi) else np.nan}


def running_spike_trains(samples, rows, n_units, speed, clock, t0_ms, session_key, cfg: LFPConfig,
                         fs_raw=30000.0):
    """Per unit (t_s, epoch_id) of spikes inside running epochs, and the running seconds.

    samples are .dat indices of one recording; epochs are runs of 25 ms bins above run_cm_s
    lasting >= min_epoch_s. Times are .dat samples / fs (exact within a recording), offset by
    session_key so trains from different sessions can be concatenated without touching.
    """
    sp = np.asarray(speed, float)
    run = np.isfinite(sp) & (sp > cfg.run_cm_s)
    ep = epochs_from_mask(run, int(round(cfg.min_epoch_s / 0.025)))
    epoch_of_bin = np.full(len(run), -1, np.int64)
    for k, (a, b) in enumerate(ep):
        epoch_of_bin[a:b] = session_key * 100000 + k
    bins = sample_to_bin(samples, clock, t0_ms)
    inb = (bins >= 0) & (bins < len(run)) & (rows >= 0)
    eid = np.full(len(samples), -1, np.int64)
    eid[inb] = epoch_of_bin[bins[inb]]
    keep = eid >= 0
    t_s = (np.asarray(samples, np.float64)[keep] + session_key * 1e9) / fs_raw
    r_k, e_k = rows[keep], eid[keep]
    trains = []
    for r in range(n_units):
        m = r_k == r
        trains.append((t_s[m], e_k[m]))
    return trains, float((ep[:, 1] - ep[:, 0]).sum() * 0.025)


def unit_lfp_measures(samples, rows, n_units, peak_names, lfp, lfp_start, q, fs, pos, theta_win, w,
                      rest_mask, peaks, ch_table, cfg: LFPConfig):
    """Per unit: theta locking (probe-median and local bipolar) and ripple modulation -> DataFrame.

    samples: .dat indices; LFP sample j <-> .dat index lfp_start + j*q. pos: one row per LFP row
    with name, x, y. theta_win: window starts of the theta state (running, or REM-like in sleep).
    peaks: ripple peak LFP indices; rest_mask: the state they were detected in.
    """
    lfp_idx = np.round((np.asarray(samples, np.float64) - lfp_start) / q).astype(np.int64)
    ref_phase = theta_phase(probe_median(lfp), fs, cfg.theta_band)
    theta_mask = mask_from_windows(theta_win, w, lfp.shape[1])
    name_row = {n: i for i, n in enumerate(pos['name'])}
    pairs = bipolar_pairs(pos.x.values, pos.y.values, cfg.bipolar_rows)
    partner = {}
    for i, j in pairs:
        partner.setdefault(i, j)
        partner.setdefault(j, i)
    local_phase = {}
    recs = []
    for r in range(n_units):
        si = lfp_idx[rows == r]
        si = si[(si >= 0) & (si < lfp.shape[1])]
        d = {'order': r}
        th = si[theta_mask[si]]
        d['n_theta_spikes'] = len(th)
        d['mrl_ref'], d['phase_ref'], d['p_ref'] = phase_locking(ref_phase[th])
        i = name_row.get(peak_names[r]) if r < len(peak_names) else None
        d['mrl_local'] = d['phase_local'] = d['p_local'] = np.nan
        if i is not None and i in partner and len(th) >= 10:
            j = partner[i]
            if (i, j) not in local_phase:
                local_phase[(i, j)] = theta_phase(lfp[j] - lfp[i], fs, cfg.theta_band)
            d['mrl_local'], d['phase_local'], d['p_local'] = phase_locking(local_phase[(i, j)][th])
        d['ripple_ratio'], d['ripple_p'] = ripple_modulation(si, peaks, rest_mask, fs,
                                                             cfg.ripple_mod_half_win_ms)
        for col in ('th_theta_excess_db', 'th_rel_theta', 'rest_ripple_excess_db', 'ripple_gain'):
            d[f'peak_{col}'] = ch_table[col].iloc[i] if i is not None else np.nan
        recs.append(d)
    return pd.DataFrame(recs)


def phase_locking(phases):
    """(MRL, preferred phase rad, Rayleigh p) of spike phases."""
    phases = np.asarray(phases, float)
    n = len(phases)
    if n < 10:
        return np.nan, np.nan, np.nan
    z = np.exp(1j * phases).mean()
    r = np.abs(z)
    R = n * r
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))   # Zar's approximation
    return float(r), float(np.angle(z)), float(min(max(p, 0.0), 1.0))


def ripple_modulation(spike_lfp_idx, peaks, rest_mask, fs, half_ms):
    """(ratio, p) of the rate within +-half_ms of ripple peaks vs the rest-state rate outside them."""
    half = int(half_ms * fs / 1000)
    in_ev = np.zeros(len(rest_mask), bool)
    for pk in peaks:
        in_ev[max(pk - half, 0):pk + half + 1] = True
    s = spike_lfp_idx[(spike_lfp_idx >= 0) & (spike_lfp_idx < len(rest_mask))]
    k_in = int(in_ev[s].sum())
    base_mask = rest_mask & ~in_ev
    k_base = int(base_mask[s].sum())
    t_in, t_base = in_ev.sum() / fs, base_mask.sum() / fs
    if t_in <= 0 or t_base <= 0:
        return np.nan, np.nan
    rate_base = k_base / t_base
    mu = rate_base * t_in
    ratio = (k_in / t_in) / rate_base if rate_base > 0 else np.nan
    return ratio, float(poisson.sf(k_in - 1, mu)) if mu > 0 else np.nan


def theta_phase(x1d, fs, band):
    return np.angle(signal.hilbert(bandpass(x1d, fs, band)))


def clock_lag(theta_pow, speed, step_s, max_lag_s=5.0, rank=False):
    """Lag (s) maximising corr(power, speed); positive = power follows speed.

    rank=True correlates ranks: band power is heavy-tailed (rare artifact bursts), which swamped
    Pearson r on ly06 s4 (60-100 Hz: Pearson weak, Spearman +0.44 at 0 s).
    """
    a = np.asarray(theta_pow, float)
    b = np.asarray(speed, float)
    if rank:
        from scipy.stats import rankdata as _rk
        fin = np.isfinite(a) & np.isfinite(b)
        a2, b2 = np.full(len(a), np.nan), np.full(len(b), np.nan)
        a2[fin], b2[fin] = _rk(a[fin]), _rk(b[fin])
        a, b = a2, b2
    ok = np.isfinite(a) & np.isfinite(b)
    a = np.where(ok, a - np.nanmean(a[ok]), 0.0)
    b = np.where(ok, b - np.nanmean(b[ok]), 0.0)
    lags = np.arange(-int(max_lag_s / step_s), int(max_lag_s / step_s) + 1)
    r = []
    for L in lags:
        if L >= 0:
            u, v = a[L:], b[:len(b) - L]
        else:
            u, v = a[:L], b[-L:]
        r.append(np.dot(u, v) / max(np.sqrt(np.dot(u, u) * np.dot(v, v)), 1e-30))
    r = np.array(r)
    return float(lags[np.argmax(r)] * step_s), float(r.max()), lags * step_s, r


# ---------------------------------------------------------------------------
# Unit-level contrasts (mice are the replicates)
# ---------------------------------------------------------------------------

def unit_group(label, tissue):
    if tissue != 'grey' or not isinstance(label, str):
        return 'other'
    if label.startswith('ENTm'):
        return 'ENTm'
    if label.startswith('ENTl'):
        return 'ENTl-deep' if label[4:5] in '56' else 'ENTl-sup'
    if label in RIPPLE_SET:
        return 'HPF'
    return 'other'


def shuffle_contrast(df, value, group_col, a, b, n_shuffle=1000, seed=0):
    """Per mouse: median(value | a) - median(value | b) with labels shuffled within recday."""
    rng = np.random.default_rng(seed)
    rows = []
    for mouse, d in df.groupby('mouse'):
        d = d[d[group_col].isin([a, b]) & np.isfinite(d[value])]
        if (d[group_col] == a).sum() < 3 or (d[group_col] == b).sum() < 3:
            continue
        obs = d.loc[d[group_col] == a, value].median() - d.loc[d[group_col] == b, value].median()
        null = np.empty(n_shuffle)
        g = d[group_col].values.copy()
        rd = d['recday'].values
        v = d[value].values
        for i in range(n_shuffle):
            gs = g.copy()
            for r in np.unique(rd):
                m = rd == r
                gs[m] = rng.permutation(gs[m])
            null[i] = np.median(v[gs == a]) - np.median(v[gs == b])
        rows.append({'mouse': mouse, 'contrast': f'{a} - {b}', 'value': value,
                     'n_a': int((g == a).sum()), 'n_b': int((g == b).sum()), 'diff': float(obs),
                     'p_shuffle': float((1 + np.sum(null >= obs)) / (1 + n_shuffle))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots (notebook only)
# ---------------------------------------------------------------------------

#: anatomy_split.REGION_COLORS, keyed by contact group
REGION_COLORS = {'ENTl-sup': '#45B5AA', 'ENTl-deep': '#0F4C81', 'ENTm': '#6B3FA0',
                 'SUB/ProS': '#BE3455', 'CA1/HPF': '#FF6F61', 'fibre/other': '#B4B2A9',
                 'outside': '#FFFFFF'}
GROUP_ORDER = ['ENTl-sup', 'ENTl-deep', 'ENTm', 'SUB/ProS', 'CA1/HPF', 'fibre/other', 'outside']


def contact_group(label, tissue):
    if tissue == 'outside' or not isinstance(label, str):
        return 'outside'
    if tissue != 'grey':
        return 'fibre/other'
    if label.startswith('ENTm'):
        return 'ENTm'
    if label.startswith('ENTl'):
        return 'ENTl-deep' if label[4:5] in '56' else 'ENTl-sup'
    if label in ('SUB', 'ProS'):
        return 'SUB/ProS'
    if label in ('CA1', 'CA2', 'CA3', 'HPF') or label.startswith('DG'):
        return 'CA1/HPF'
    return 'fibre/other'


def _probe_x(x, col_gap=40.0):
    """Plot abscissa: shanks 0-3 side by side, the two columns of a shank 1 unit apart."""
    x = np.asarray(x, float)
    return shank_of(x) * 3.0 + ((x % SHANK_PITCH_UM) > 0)


def plot_probe_map(ch, anat, mouse, measures, titles=None, y_max=None, out_path=None, vlims=None):
    """Atlas labels (current, mirror) next to physiology maps for one mouse; one dot per contact.

    ch: rows with x, y and the measure columns (probe-map or raw channels). Shank 0 is drawn on
    the left, so under the current convention left = anterior.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    titles = titles or measures
    vlims = vlims or {}
    with mpl.rc_context(_rc()):
        n = 2 + len(measures)
        fig, axes = plt.subplots(1, n, figsize=(1.25 * n + 0.6, 4.2), sharey=True)
        x = _probe_x(ch['x'].values)
        y = ch['y'].values
        for k, mapping in enumerate(('current', 'mirror')):
            lab, tis = anat.labels(ch['x'].values, y, mapping)
            g = [contact_group(a, b) for a, b in zip(lab, tis)]
            axes[k].scatter(x, y, c=[REGION_COLORS[v] for v in g], s=9, marker='s', linewidths=0)
            axes[k].set_title(f'atlas, {mapping}')
        for k, (m, t) in enumerate(zip(measures, titles)):
            ax = axes[2 + k]
            v = ch[m].values.astype(float)
            lo, hi = vlims.get(m, (np.nanpercentile(v, 2), np.nanpercentile(v, 98)))
            sc = ax.scatter(x, y, c=v, s=9, marker='s', linewidths=0, cmap='magma', vmin=lo, vmax=hi)
            if 'is_ref' in ch:
                r = ch['is_ref'].values.astype(bool)
                ax.scatter(x[r], y[r], s=30, facecolors='none', edgecolors='#2C2C2A', linewidths=0.6)
            ax.set_title(t)
            fig.colorbar(sc, ax=ax, fraction=0.08, pad=0.02, shrink=0.6)
        for ax in axes:
            ax.set_xticks([0.5, 3.5, 6.5, 9.5])
            ax.set_xticklabels(['0', '1', '2', '3'])
            ax.set_xlabel('shank')
            ax.set_xlim(-1, 11)
            if y_max:
                ax.set_ylim(-20, y_max)
        axes[0].set_ylabel('distance from tip (µm)')
        handles = [mpl.lines.Line2D([], [], marker='s', ls='', color=REGION_COLORS[g], label=g)
                   for g in GROUP_ORDER[:-1]]
        fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(mouse, x=0.02, ha='left')
        fig.subplots_adjust(bottom=0.17, wspace=0.35)
        if out_path:
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
    return fig


def plot_orientation(orient, measure='ripple_gain', perm=None, out_path=None):
    """AUC under the current mapping and under the mirror, one pair per mouse (or recording)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    d = orient[orient['measure'] == measure]
    key = 'mouse' if 'mouse' in d else 'recording'
    with mpl.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(0.45 * len(d) + 1.2, 2.2))
        for i, (_, r) in enumerate(d.iterrows()):
            ax.plot([i - 0.12, i + 0.12], [r['auc_current'], r['auc_mirror']], color='#B4B2A9', lw=0.8)
            ax.scatter([i - 0.12], [r['auc_current']], color='#0F4C81', s=16, zorder=3,
                       label='current' if i == 0 else None)
            ax.scatter([i + 0.12], [r['auc_mirror']], color='#BE3455', s=16, zorder=3,
                       label='mirror' if i == 0 else None)
        ax.axhline(0.5, color='#2C2C2A', lw=0.6, ls=':')
        ax.axhline(0.7, color='#2C2C2A', lw=0.6, ls='--')
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d[key], rotation=30, ha='right')
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f'AUC, {measure}\nripple structures vs rest')
        ax.legend(frameon=False, loc='lower right')
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
    return fig


def plot_unit_groups(units, value, mapping='current', groups=('ENTl-sup', 'ENTl-deep', 'ENTm', 'HPF'),
                     ylabel=None, out_path=None, log=False):
    """Units as faint dots, per-mouse medians as large dots, by label group."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    col = {'ENTl-sup': '#45B5AA', 'ENTl-deep': '#0F4C81', 'ENTm': '#6B3FA0', 'HPF': '#BE3455',
           'other': '#B4B2A9'}
    mice = sorted(units['mouse'].unique())
    marks = dict(zip(mice, 'osD^v'))
    rng = np.random.default_rng(0)
    with mpl.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(0.75 * len(groups) + 1.4, 2.4))
        for i, g in enumerate(groups):
            d = units[(units[f'group_{mapping}'] == g) & np.isfinite(units[value].astype(float))]
            ax.scatter(i + rng.uniform(-0.25, 0.25, len(d)), d[value], s=3, color=col[g], alpha=0.25,
                       linewidths=0)
            for m, dm in d.groupby('mouse'):
                if len(dm) >= 3:
                    ax.scatter([i + 0.32], [dm[value].median()], marker=marks[m], s=22, color=col[g],
                               edgecolors='#2C2C2A', linewidths=0.5, zorder=3)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=30, ha='right')
        ax.set_ylabel(ylabel or value)
        if log:
            ax.set_yscale('log')
        handles = [mpl.lines.Line2D([], [], marker=marks[m], ls='', color='#B4B2A9', markeredgecolor='#2C2C2A',
                                    label=m) for m in mice]
        ax.legend(handles=handles, frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title(f'{mapping} labels')
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
    return fig


def plot_border_profile(df, value, anat_by_mouse, mapping='current', ylabel=None, out_path=None):
    """value vs signed distance to the ENTl/ENTm border (- ENTl, + ENTm), EC contacts, per mouse."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    with mpl.rc_context(_rc()):
        mice = sorted(df['mouse'].unique())
        fig, axes = plt.subplots(1, len(mice), figsize=(1.7 * len(mice) + 0.4, 2.0), sharey=True, squeeze=False)
        for ax, m in zip(axes[0], mice):
            d = df[df['mouse'] == m]
            dist = anat_by_mouse[m].signed_distance(d['x'].values, d['y'].values, mapping)
            ok = np.isfinite(dist) & np.isfinite(d[value].values.astype(float))
            ax.scatter(dist[ok], d[value].values[ok], s=4, c=np.where(dist[ok] > 0, '#6B3FA0', '#0F4C81'),
                       linewidths=0)
            ax.axvline(0, color='#2C2C2A', lw=0.6, ls=':')
            ax.set_title(m)
            ax.set_xlabel('signed dist. to border (µm)')
        axes[0][0].set_ylabel(ylabel or value)
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
    return fig


_ARIAL = '/ceph/behrens/max_kirkby/goal_sequencing_1/font/Arial.ttf'


def _rc():
    rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8, 'axes.titlesize': 8,
          'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6,
          'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False}
    try:
        from matplotlib import font_manager
        if os.path.exists(_ARIAL):
            font_manager.fontManager.addfont(_ARIAL)
            rc['font.family'] = 'Arial'
    except Exception:                                    # noqa: BLE001 - font is cosmetic
        pass
    return rc
