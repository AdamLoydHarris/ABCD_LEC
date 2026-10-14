"""Shared pieces of the spatial pipelines (open field and maze): speed, movement direction, binning,
the two bridge-safe smoothers, scores used by both engines, circular-shift nulls, and shard I/O.

Maps are indexed [ix, iy] (x first, as `np.histogram2d`); plot with `.T` and `origin='upper'` so the
y-down arena frame reads like the camera image. Rates are Hz, occupancy seconds, distances cm.

This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import socket
import subprocess
import sys
import time

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BIN_S = 0.025                 # every dataset is binned at 25 ms
FS_AP = 30000.0               # Neuropixels AP sampling rate, for template widths


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------

def speed_cm_s(xy: np.ndarray) -> np.ndarray:
    """(T,) running speed in cm/s from (T, 2) cm, via the GLM's NaN-aware smoother (sigma 3 bins)."""
    import glm_analysis_v3 as g3
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return np.full(len(xy), np.nan)
    return g3.smooth_and_calculate_scalar_derivatives(xy, sigma=3)[:, 2] / BIN_S


def movement_direction(xy: np.ndarray, half: int = 5) -> np.ndarray:
    """(T,) movement direction (rad, image frame) from the displacement over +-`half` bins."""
    xy = np.asarray(xy, float)
    out = np.full(len(xy), np.nan)
    if len(xy) > 2 * half:
        d = xy[2 * half:] - xy[:-2 * half]
        out[half:-half] = np.arctan2(d[:, 1], d[:, 0])
    return out


def angular_speed_dps(theta: np.ndarray, sigma_bins: float = 2.0, max_gap_bins: int = 4,
                      max_dps: float = 2000.0) -> tuple:
    """(T,) |angular speed| in deg/s of an angle series (radians, 25 ms grid) and a QC dict.

    Single-bin spikes (> 90 deg from both neighbours that agree within 45 deg) are dropped; the angle is
    unwrapped within tracked segments (gaps longer than max_gap_bins split them, shorter gaps are
    interpolated), Gaussian-smoothed and differentiated; values above max_dps become NaN. Only |w| is
    returned: the maze frames are mirrored relative to the arena and PFC's handedness is unknown.
    """
    from scipy.ndimage import gaussian_filter1d
    th = np.asarray(theta, float).copy()
    T = len(th)
    n_spike = 0
    if T >= 3:
        def ad(a, b):
            return np.abs(np.angle(np.exp(1j * (a - b))))
        prev, mid, nxt = th[:-2], th[1:-1], th[2:]
        with np.errstate(invalid='ignore'):
            spike = (np.isfinite(prev) & np.isfinite(mid) & np.isfinite(nxt) & (ad(prev, nxt) < np.radians(45))
                     & (ad(mid, prev) > np.radians(90)) & (ad(mid, nxt) > np.radians(90)))
        idx = np.nonzero(spike)[0] + 1
        th[idx] = np.nan
        n_spike = len(idx)
    out = np.full(T, np.nan)
    f_idx = np.nonzero(np.isfinite(th))[0]
    if len(f_idx) >= 3:
        cuts = np.nonzero(np.diff(f_idx) > max_gap_bins + 1)[0]
        for seg in np.split(f_idx, cuts + 1):
            if len(seg) < 3:
                continue
            i0, i1 = seg[0], seg[-1]
            x = np.arange(i0, i1 + 1)
            full = np.interp(x, seg, np.unwrap(th[seg]))
            w = np.degrees(np.abs(np.gradient(gaussian_filter1d(full, sigma_bins, mode='nearest')))) / BIN_S
            w[~np.isfinite(th[i0:i1 + 1])] = np.nan
            out[i0:i1 + 1] = w
    n_cap = int(np.nansum(out > max_dps))
    out[out > max_dps] = np.nan
    return out, dict(n_spike=n_spike, n_cap=n_cap, n_finite=int(np.isfinite(out).sum()), T=T)


def hd_guard(xy: np.ndarray, hd_rad: np.ndarray, speed: np.ndarray, min_speed: float = 5.0,
             max_offset_deg: float = 15.0, min_r: float = 0.8) -> dict:
    """HD must track movement direction while running; returns offset (deg), R, n and ok."""
    md = movement_direction(xy)
    sel = (speed > min_speed) & np.isfinite(md) & np.isfinite(hd_rad)
    if sel.sum() < 100:
        return dict(ok=False, offset_deg=np.nan, R=np.nan, n=int(sel.sum()))
    z = np.exp(1j * (hd_rad[sel] - md[sel])).mean()
    off = float(np.degrees(np.angle(z)))
    return dict(ok=bool(abs(off) <= max_offset_deg and abs(z) >= min_r), offset_deg=off,
                R=float(abs(z)), n=int(sel.sum()))


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------

def fixed_edges(lo: float, hi: float, bin_cm: float) -> np.ndarray:
    """Bin edges with a fixed width covering [lo, hi]; n bins = round((hi - lo) / bin_cm)."""
    n = max(1, int(round((hi - lo) / bin_cm)))
    return np.linspace(lo, lo + n * bin_cm, n + 1)


def bin_index(xy: np.ndarray, ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
    """(T,) flat bin index ix * ny + iy, or -1 for NaN / outside the edges."""
    xy = np.asarray(xy, float)
    nx, ny = len(ex) - 1, len(ey) - 1
    ix = np.searchsorted(ex, xy[:, 0], side='right') - 1
    iy = np.searchsorted(ey, xy[:, 1], side='right') - 1
    ok = np.isfinite(xy).all(axis=1) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.where(ok, ix * ny + iy, -1)


def onehot(flat: np.ndarray, mask: np.ndarray, nbins: int) -> sparse.csr_matrix:
    """(T, nbins) sparse indicator of each masked bin's location; rows outside the mask are empty."""
    T = len(flat)
    sel = mask & (flat >= 0)
    rows = np.nonzero(sel)[0]
    return sparse.csr_matrix((np.ones(len(rows)), (rows, flat[sel])), shape=(T, nbins))


def occupancy_map(M: sparse.csr_matrix, shape: tuple) -> np.ndarray:
    """Seconds per bin from the indicator matrix."""
    return np.asarray(M.sum(axis=0)).ravel().reshape(shape) * BIN_S


def spike_maps(counts: np.ndarray, M: sparse.csr_matrix, shape: tuple) -> np.ndarray:
    """(N, *shape) spike counts per bin for every unit at once."""
    S = (M.T @ np.asarray(counts, float).T).T
    return np.asarray(S).reshape((counts.shape[0],) + tuple(shape))


# ---------------------------------------------------------------------------
# smoothers
# ---------------------------------------------------------------------------

def smooth_normconv(arr: np.ndarray, sigma: float, valid: np.ndarray) -> np.ndarray:
    """The user's bridge-safe smoother (local_analysis cell 48): G(arr*valid)/G(valid), NaN where weight 0.

    Zeros outside `valid` enter neither numerator nor denominator, so a 1-bin bridge flanked by
    unvisited bins keeps its value (measured 1.00 of truth at sigma 1-6; zero-fill gave 0.31-0.66).
    """
    if sigma is None or sigma <= 0:
        return np.asarray(arr, float).copy()
    filled = np.where(valid, arr, 0.0).astype(float)
    weight = ndi.gaussian_filter(valid.astype(float), sigma=sigma, mode='constant', cval=0.0)
    smoothed = ndi.gaussian_filter(filled, sigma=sigma, mode='constant', cval=0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = smoothed / weight
    out[weight == 0] = np.nan
    return out


def rate_maps_normconv(spk: np.ndarray, occ: np.ndarray, sigma: float, occ_threshold: float = 0.0) -> np.ndarray:
    """(N, nx, ny) Hz: raw rate per visited bin, smoothed with `smooth_normconv`, as `make_firing_rate_maps`.

    `occ_threshold` (s) is applied after smoothing; with 0 only unvisited bins become NaN.
    """
    visited = occ > 0
    out = np.full(spk.shape, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        raw = np.where(visited[None], spk / np.where(visited, occ, 1.0)[None], 0.0)
    for i in range(spk.shape[0]):
        r = smooth_normconv(raw[i], sigma, visited)
        if occ_threshold > 0:
            r[occ < occ_threshold] = np.nan
        else:
            r[~visited] = np.nan
        out[i] = r
    return out


def rate_maps_separate(spk: np.ndarray, occ: np.ndarray, sigma: float, valid: np.ndarray | None = None,
                       min_smoothed_occ: float = 0.0) -> np.ndarray:
    """(N, nx, ny) Hz: G(spikes*valid)/G(occ*valid), occupancy-weighted; NaN outside `valid` or where
    smoothed occupancy < `min_smoothed_occ` s. `valid` defaults to visited bins."""
    if valid is None:
        valid = occ > 0
    so = ndi.gaussian_filter(np.where(valid, occ, 0.0), sigma=sigma, mode='constant', cval=0.0) if sigma else np.where(valid, occ, 0.0)
    out = np.empty(spk.shape)
    for i in range(spk.shape[0]):
        ss = ndi.gaussian_filter(np.where(valid, spk[i], 0.0), sigma=sigma, mode='constant', cval=0.0) if sigma else np.where(valid, spk[i], 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[i] = ss / so
    bad = ~valid | (so <= 0) | (so < min_smoothed_occ)
    out[:, bad] = np.nan
    return out


# ---------------------------------------------------------------------------
# scores shared by both engines
# ---------------------------------------------------------------------------

def skaggs_info(rate: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """(N,) Skaggs information, bits/spike, over bins with a finite rate and occupancy > 0."""
    rate = rate[None] if rate.ndim == 2 else rate
    rate = rate.reshape(rate.shape[0], -1)
    occ = occ.ravel()
    out = np.full(rate.shape[0], np.nan)
    for i in range(rate.shape[0]):
        m = np.isfinite(rate[i]) & (occ > 0)
        if m.sum() < 2:
            continue
        p = occ[m] / occ[m].sum()
        lam = rate[i, m]
        mu = (p * lam).sum()
        if mu <= 0:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            t = p * (lam / mu) * np.log2(lam / mu)
        out[i] = np.nansum(t)
    return out


def map_corr(a: np.ndarray, b: np.ndarray, min_bins: int = 10) -> np.ndarray:
    """(N,) Pearson r between paired maps over bins finite in both; NaN below `min_bins` or flat."""
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        m = np.isfinite(a[i]) & np.isfinite(b[i])
        if m.sum() < min_bins:
            continue
        x, y = a[i, m], b[i, m]
        if x.std() == 0 or y.std() == 0:
            continue
        out[i] = np.corrcoef(x, y)[0, 1]
    return out


# ---------------------------------------------------------------------------
# circular-shift nulls
# ---------------------------------------------------------------------------

def shift_offsets(T: int, n: int, min_shift_bins: int, rng: np.random.Generator) -> np.ndarray:
    """(n,) circular shifts in [min_shift, T - min_shift]; raises if the session is too short."""
    if T <= 2 * min_shift_bins + 1:
        raise ValueError(f'session of {T} bins too short for a {min_shift_bins}-bin minimum shift')
    return rng.integers(min_shift_bins, T - min_shift_bins + 1, size=n)


def pooled_threshold(null: np.ndarray, eligible: np.ndarray, q: float) -> float:
    """q-th percentile of the (n_shifts, N) null pooled over eligible units."""
    v = null[:, eligible].ravel()
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else np.nan


def unit_threshold(null: np.ndarray, q: float) -> np.ndarray:
    """(N,) each unit's own q-th percentile of the (n_shifts, N) null."""
    with np.errstate(all='ignore'):
        return np.nanpercentile(null, q, axis=0)


# ---------------------------------------------------------------------------
# provenance, shards
# ---------------------------------------------------------------------------

def file_hash(paths) -> str:
    """sha1 over the bytes of `paths`, in order."""
    h = hashlib.sha1()
    for p in paths:
        with open(p, 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    """sha1 of the canonical JSON of a config."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def git_sha(repo_dir: str) -> str:
    try:
        return subprocess.run(['git', '-C', repo_dir, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def stamp(engine: str, cfg: dict, code_files: list, dataset: str, data_version: str) -> dict:
    """Provenance stamp written into every shard; the merge refuses shards whose stamps differ."""
    return dict(engine=engine, dataset=dataset, data_version=data_version, cfg=cfg,
                cfg_hash=config_hash(cfg), code_hash=file_hash(code_files),
                git_sha=git_sha(HERE), host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID', ''),
                time=time.strftime('%Y-%m-%dT%H:%M:%S'))


def compare_tree(a, b, path: str = '', out: dict | None = None) -> dict:
    """Every number of the old object `a` against the new `b`: dicts by key, lists by index (b may be longer),
    arrays elementwise, DataFrames by column. -> max_diff, worst path, nan_mismatch, missing paths, n values."""
    import pandas as pd
    if out is None:
        out = dict(max_diff=0.0, worst='', nan_mismatch=0, missing=[], n=0)
    if isinstance(a, dict):
        for k, v in a.items():
            if k not in b:
                out['missing'].append(f'{path}/{k}')
            else:
                compare_tree(v, b[k], f'{path}/{k}', out)
    elif isinstance(a, (list, tuple)):
        if len(b) < len(a):
            out['missing'].append(f'{path} length {len(a)} > {len(b)}')
        for i, v in enumerate(a[:len(b)]):
            compare_tree(v, b[i], f'{path}[{i}]', out)
    elif isinstance(a, pd.DataFrame):
        for col in a.columns:
            if col not in b:
                out['missing'].append(f'{path}.{col}')
            else:
                compare_tree(a[col].to_numpy(), b[col].to_numpy(), f'{path}.{col}', out)
    else:
        x, y = np.asarray(a), np.asarray(b)
        if x.dtype.kind in 'fiub' and y.dtype.kind in 'fiub':
            if x.shape != y.shape:
                out['missing'].append(f'{path} shape {x.shape} vs {y.shape}')
                return out
            xf, yf = x.astype(float), y.astype(float)
            out['nan_mismatch'] += int((np.isnan(xf) != np.isnan(yf)).sum())
            ok = ~np.isnan(xf) & ~np.isnan(yf)
            if ok.any():
                d = float(np.abs(xf[ok] - yf[ok]).max())
                if d > out['max_diff']:
                    out['max_diff'], out['worst'] = d, path
            out['n'] += x.size
        elif x.shape != y.shape or not np.array_equal(x.astype(str), y.astype(str)):
            out['missing'].append(f'{path} (non-numeric differs)')
    return out


def run_tag(cfg: dict, code_files: list) -> str:
    """Short default tag = hash of config + code, so a changed recipe can never reuse a directory."""
    return (config_hash(cfg)[:6] + file_hash(code_files)[:6])


def write_new(path: str, obj) -> None:
    """Pickle to `path`, refusing to overwrite (user rule: never overwrite an existing file)."""
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; use a new --run-tag rather than overwriting')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.partial'
    if os.path.exists(tmp):
        raise FileExistsError(f'{tmp} exists from an earlier crash; inspect it before rerunning')
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh, protocol=4)
    os.rename(tmp, path)


def versioned(path: str) -> str:
    """First free name among path, path.v1, path.v2, ... (stem.vK.ext)."""
    stem, ext = os.path.splitext(path)
    k = 0
    while os.path.exists(p := stem + (f'.v{k}' if k else '') + ext):
        k += 1
    return p


def load_shards(shard_dir: str) -> dict:
    """{recday: shard} from every `*.pkl` in `shard_dir`; raises on duplicate recdays."""
    out = {}
    for f in sorted(os.listdir(shard_dir)):
        if not f.endswith('.pkl'):
            continue
        with open(os.path.join(shard_dir, f), 'rb') as fh:
            sh = pickle.load(fh)
        rd = sh['recday']
        if rd in out:
            raise ValueError(f'two shards for {rd} in {shard_dir}')
        out[rd] = sh
    return out


def check_stamps(shards: dict) -> dict:
    """The common stamp of a set of shards; raises if any engine, config, code or data version differs."""
    keys = ('engine', 'dataset', 'data_version', 'cfg_hash', 'code_hash')
    ref = None
    for rd, sh in shards.items():
        s = {k: sh['stamp'][k] for k in keys}
        if ref is None:
            ref = s
        elif s != ref:
            raise ValueError(f'{rd}: stamp {s} differs from {ref}')
    return ref or {}
