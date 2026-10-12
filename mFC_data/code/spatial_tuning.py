"""Open-field spatial tuning, stage 1 (classic scores), for the mEC big field and the LEC object box.

Per unit and arena session: Skaggs information and split-half stability, gridness (Sargolini, expanding
annulus), border score (Solstad), head direction with the pxd factorisation (Burgess 2005), speed score
(Kropff 2015), object-proximal firing against virtual objects, object-vector firing within a sort, and
the dihedral rotation between two arena sessions of one sort. Nulls are circular spike shifts; chance
counts come from applying one shift to every unit at once.

Entry point: `run_of_recday(views, recday, units, cfg)` with `views = spatial_io.load(...)[recday]` and
`units = spatial_io.unit_table(...)`; returns the shard dict written by `run_spatial.py`.

This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import spatial_common as sc                                           # noqa: E402

ENGINE = 'of'
ENGINE_VERSION = 1

DEFAULT_CFG = dict(
    bin_cm=2.5, sigma_bins=2.0, min_smoothed_occ_s=0.1, min_speed=2.5,
    n_shifts=1000, min_shift_s=20.0, seed=0, n_jobs=8,
    grid=dict(enabled=True, rmax_frac=0.5, inner_fallback_r=0.1, min_ring_width_bins=2, peak_thr=0.1,
              peak_min_sep_cm=10.0, min_overlap=20),
    border=dict(field_thr=0.3, min_area_cm2=200.0, max_area_frac=0.6, min_valid_wall_frac=0.5,
                max_wall_rms_cm=1.5),
    hd=dict(nbins=36, pxd_pos_cm=7.5, pxd_dir_deg=30.0, pxd_tol=1e-6, pxd_maxit=50,
            guard_offset_deg=15.0, guard_r=0.8, guard_speed=5.0),
    speed=dict(sigma_s=0.25, lo=2.0, hi=50.0),          # Kropff r: descriptive; the class uses speed_f
    pds=dict(speed_edges=(2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 1e9), tol=1e-4, maxit=200),
    obj=dict(margin_cm=4.0, band_cm=20.0, n_sectors=8, n_virtual=200, min_virtual=100, wall_tol_cm=5.0,
             speed_bin=2.5, radius_default_cm=6.0, radius_search_cm=40.0),
    units=dict(narrow_ms=0.4, rate_cut_hz=10.0, min_moving_spikes=100),
    q=dict(pooled=99.0, unit=99.0, stability=95.0),
    rotation=dict(n_perm=1000, n_boot=2000, n_shift_object=200),
)

STATS = ('si', 'stab', 'grid', 'border', 'hd_r', 'pxd_r', 'pxd_split', 'speed_r', 'speed_r1', 'speed_r2',
         'si_p', 'stab_p', 'speed_f', 'speed_f1', 'speed_f2')


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def measure_object_radius(xy: np.ndarray, centre, cfg: dict) -> float:
    """Radius (cm) of the occupancy hole around an object: first 1 cm annulus whose dwell density
    reaches half the 15-35 cm reference. Clicked radii are a 6 cm default, not a measurement."""
    c = np.asarray(centre, float)
    d = np.linalg.norm(xy - c, axis=1)
    d = d[np.isfinite(d) & (d < cfg['obj']['radius_search_cm'])]
    if len(d) < 500:
        return float(cfg['obj']['radius_default_cm'])
    edges = np.arange(0, cfg['obj']['radius_search_cm'] + 1, 1.0)
    h, _ = np.histogram(d, edges)
    dens = h / (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2))
    ref = np.median(dens[(edges[:-1] >= 15) & (edges[:-1] < 35)])
    hit = np.nonzero(dens >= 0.5 * ref)[0]
    if ref <= 0 or not len(hit):
        return float(cfg['obj']['radius_default_cm'])
    return float(np.clip(edges[hit[0]], 2.0, 15.0))


def bin_centres(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2


def wall_distance(x: np.ndarray, y: np.ndarray, side: float) -> np.ndarray:
    return np.minimum.reduce([x, side - x, y, side - y])


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def autocorr(rate: np.ndarray, min_overlap: int = 20) -> np.ndarray:
    """NaN-aware Pearson spatial autocorrelogram, shape (2nx-1, 2ny-1); NaN below `min_overlap` bins."""
    m = np.isfinite(rate)
    z = np.where(m, rate, 0.0)
    mf = m.astype(float)

    def f(a, b):
        return fftconvolve(a, b[::-1, ::-1], mode='full')

    n = np.rint(f(mf, mf))
    sx, sy = f(z, mf), f(mf, z)
    sxx, syy, sxy = f(z * z, mf), f(mf, z * z), f(z, z)
    with np.errstate(invalid='ignore', divide='ignore'):
        den = np.sqrt(np.maximum(n * sxx - sx ** 2, 0)) * np.sqrt(np.maximum(n * syy - sy ** 2, 0))
        ac = (n * sxy - sx * sy) / den
    ac[(n < min_overlap) | ~np.isfinite(ac)] = np.nan
    return ac


class GridGeometry:
    """Ring indices and rotation helpers for autocorrelograms of one map size."""

    def __init__(self, nx: int, ny: int, bin_cm: float, rmax_cm: float):
        self.shape = (2 * nx - 1, 2 * ny - 1)
        cx, cy = nx - 1, ny - 1
        X, Y = np.meshgrid(np.arange(self.shape[0]) - cx, np.arange(self.shape[1]) - cy, indexing='ij')
        self.R = np.hypot(X, Y)
        self.ring = np.floor(self.R).astype(int)
        self.nring = int(self.ring.max()) + 1
        self.bin_cm = bin_cm
        self.rmax = int(min(np.floor(rmax_cm / bin_cm), self.nring - 1))
        self.theta = np.arctan2(Y, X)


def gridness(ac: np.ndarray, geo: GridGeometry, cfg: dict) -> dict:
    """Sargolini gridness min(r60, r120) - max(r30, r90, r150), maximised over the outer radius."""
    g = cfg['grid']
    out = dict(grid=np.nan, inner_cm=np.nan, outer_cm=np.nan, spacing_cm=np.nan, orientation_deg=np.nan,
               n_peaks=0)
    fin = np.isfinite(ac)
    if fin.sum() < 50:
        return out
    a = np.where(fin, ac, 0.0)
    prof = np.bincount(geo.ring[fin], weights=ac[fin], minlength=geo.nring) / np.maximum(
        np.bincount(geo.ring[fin], minlength=geo.nring), 1)
    inner = None
    for r in range(1, min(geo.rmax, geo.nring - 1)):
        if prof[r] < prof[r - 1] and prof[r] <= prof[r + 1]:
            inner = r
            break
    if inner is None:
        below = np.nonzero(prof[1:geo.rmax] < g['inner_fallback_r'])[0]
        inner = int(below[0]) + 1 if len(below) else None
    if inner is None or inner + g['min_ring_width_bins'] > geo.rmax:
        return out
    best, best_outer = -np.inf, np.nan
    corrs = {}
    for ang in (30, 60, 90, 120, 150):
        rot = ndi.rotate(a, ang, reshape=False, order=1, mode='constant', cval=0.0)
        rotm = ndi.rotate(fin.astype(float), ang, reshape=False, order=1, mode='constant', cval=0.0) > 0.999
        v = fin & rotm & (geo.ring >= inner) & (geo.ring <= geo.rmax)
        rg = geo.ring[v] - inner
        nr = geo.rmax - inner + 1
        sums = [np.bincount(rg, weights=w, minlength=nr) for w in
                (np.ones(v.sum()), a[v], rot[v], a[v] ** 2, rot[v] ** 2, a[v] * rot[v])]
        cs = [np.cumsum(s) for s in sums]
        n, sa, sb, saa, sbb, sab = cs
        with np.errstate(invalid='ignore', divide='ignore'):
            corrs[ang] = (n * sab - sa * sb) / np.sqrt((n * saa - sa ** 2) * (n * sbb - sb ** 2))
    gs = np.minimum(corrs[60], corrs[120]) - np.maximum.reduce([corrs[30], corrs[90], corrs[150]])
    gs[:g['min_ring_width_bins']] = np.nan
    if np.all(~np.isfinite(gs)):
        return out
    k = int(np.nanargmax(gs))
    best, best_outer = float(gs[k]), inner + k
    out.update(grid=best, inner_cm=inner * geo.bin_cm, outer_cm=best_outer * geo.bin_cm)
    sep = max(1, int(round(g['peak_min_sep_cm'] / geo.bin_cm)))
    mx = ndi.maximum_filter(np.where(fin, ac, -np.inf), size=2 * sep + 1, mode='constant', cval=-np.inf)
    pk = fin & (ac == mx) & (ac > g['peak_thr']) & (geo.R >= inner) & (geo.R <= geo.rmax)
    if pk.sum() >= 6:
        r = geo.R[pk]
        order = np.argsort(r)[:6]
        out['spacing_cm'] = float(np.median(r[order]) * geo.bin_cm)
        th = np.degrees(geo.theta[pk][order]) % 60
        out['orientation_deg'] = float(np.min(th))
    out['n_peaks'] = int(pk.sum())
    return out


# ---------------------------------------------------------------------------
# border
# ---------------------------------------------------------------------------

def border_score(rate: np.ndarray, side: float, bin_cm: float, excluded: set, cfg: dict) -> dict:
    """Solstad border score (CM - DM)/(CM + DM) with the preferred wall; NaN without a valid field."""
    b = cfg['border']
    out = dict(border=np.nan, cm=np.nan, dm=np.nan, wall='')
    valid = np.isfinite(rate)
    if valid.sum() < 10:
        return out
    pk = np.nanmax(rate)
    if not np.isfinite(pk) or pk <= 0:
        return out
    lab, n = ndi.label(valid & (np.where(valid, rate, 0) > b['field_thr'] * pk))
    area_bin = bin_cm ** 2
    fields = []
    for i in range(1, n + 1):
        f = lab == i
        if f.sum() * area_bin >= b['min_area_cm2'] and f.sum() <= b['max_area_frac'] * valid.sum():
            fields.append(f)
    if not fields:
        return out
    nx, ny = rate.shape
    walls = {}
    for name in ('left', 'right', 'bottom', 'top'):
        if name in excluded:
            continue
        idx = []
        for j in range(ny if name in ('left', 'right') else nx):
            line = valid[:, j] if name in ('left', 'right') else valid[j, :]
            pos = np.nonzero(line)[0]
            if not len(pos):
                idx.append(None)
                continue
            k = pos[0] if name in ('left', 'bottom') else pos[-1]
            idx.append((k, j) if name in ('left', 'right') else (j, k))
        ok = [p for p in idx if p is not None]
        if len(ok) >= b['min_valid_wall_frac'] * len(idx):
            walls[name] = ok
    if not walls:
        return out
    cm_best, wall_best = -1.0, ''
    for f in fields:
        for name, pts in walls.items():
            cov = np.mean([f[p] for p in pts])
            if cov > cm_best:
                cm_best, wall_best = cov, name
    allf = np.any(fields, axis=0)
    c = bin_centres(np.arange(nx + 1) * bin_cm)
    cy = bin_centres(np.arange(ny + 1) * bin_cm)
    X, Y = np.meshgrid(c, cy, indexing='ij')
    dw = wall_distance(X, Y, side)
    w = np.where(allf, np.where(valid, rate, 0.0), 0.0)
    dm = float((w * dw).sum() / w.sum() / (side / 2)) if w.sum() > 0 else np.nan
    out.update(border=(cm_best - dm) / (cm_best + dm), cm=cm_best, dm=dm, wall=wall_best)
    return out


# ---------------------------------------------------------------------------
# head direction
# ---------------------------------------------------------------------------

def vector_length(curve: np.ndarray, centres: np.ndarray) -> tuple:
    """(R, preferred direction rad) of rate curves (N, nbins) over bin centres."""
    z = (curve * np.exp(1j * centres)[None]).sum(1)
    s = curve.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.abs(z) / s, np.angle(z)


def pxd(n: np.ndarray, t: np.ndarray, tol: float, maxit: int) -> tuple:
    """Maximum-likelihood position x direction factorisation for all units at once.

    n (N, P, D) spike counts, t (P, D) dwell s -> p (N, P) Hz, d (N, D) mean 1, converged (N,), iters.
    Iterates p = sum_d n / sum_d t d, d = sum_p n / sum_p t p, renormalising d to mean 1.
    """
    N, P, D = n.shape
    d = np.ones((N, D))
    eps = 1e-12
    ll_old = np.full(N, -np.inf)
    conv = np.zeros(N, bool)
    it = 0
    for it in range(1, maxit + 1):
        p = n.sum(2) / np.maximum((t[None] * d[:, None, :]).sum(2), eps)
        d = n.sum(1) / np.maximum((t[None] * p[:, :, None]).sum(1), eps)
        m = d.mean(1, keepdims=True)
        m[m <= 0] = 1.0
        d = d / m
        p = p * m
        lam = p[:, :, None] * d[:, None, :] * t[None]
        with np.errstate(divide='ignore', invalid='ignore'):
            ll = np.where(lam > 0, n * np.log(np.maximum(lam, eps)) - lam, 0.0).sum((1, 2))
        conv = np.abs(ll - ll_old) <= tol * np.maximum(np.abs(ll), 1.0)
        ll_old = ll
        if conv.all():
            break
    return p, d, conv, it


def factorize3(n: np.ndarray, t: np.ndarray, tol: float, maxit: int) -> tuple:
    """Maximum-likelihood rate = p(position) d(direction) s(speed) for all units at once.

    n (N, P, D, S) spike counts, t (P, D, S) dwell s -> p (N, P) Hz, d (N, D), s (N, S) (mean 1), iters.
    Iterative scaling with matrix products over the marginals; stops when p changes by less than
    `tol` relative (evaluating the full rate tensor each iteration made the gate 5x slower).
    """
    N, P, D, S = n.shape
    nP, nD, nS = n.sum((2, 3)), n.sum((1, 3)), n.sum((1, 2))
    tP = t.reshape(P, D * S)
    tD = t.transpose(1, 0, 2).reshape(D, P * S)
    tS = t.transpose(2, 0, 1).reshape(S, P * D)
    d = np.ones((N, D))
    sv = np.ones((N, S))
    p = np.zeros((N, P))
    eps = 1e-12
    it = 0
    for it in range(1, maxit + 1):
        p_new = nP / np.maximum((d[:, :, None] * sv[:, None, :]).reshape(N, D * S) @ tP.T, eps)
        d = nD / np.maximum((p_new[:, :, None] * sv[:, None, :]).reshape(N, P * S) @ tD.T, eps)
        md = d.mean(1, keepdims=True)
        md[md <= 0] = 1
        d, p_new = d / md, p_new * md
        sv = nS / np.maximum((p_new[:, :, None] * d[:, None, :]).reshape(N, P * D) @ tS.T, eps)
        ms_ = sv.mean(1, keepdims=True)
        ms_[ms_ <= 0] = 1
        sv, p_new = sv / ms_, p_new * ms_
        delta = np.abs(p_new - p).max(1) / np.maximum(p_new.max(1), eps)
        p = p_new
        if np.all(delta < tol):
            break
    return p, d, sv, it


def _factorize3_reference(n, t, tol, maxit):
    """Slow einsum version kept for the equivalence check of `factorize3`."""
    N, P, D, S = n.shape
    d, sv = np.ones((N, D)), np.ones((N, S))
    for _ in range(maxit):
        p = n.sum((2, 3)) / np.maximum(np.einsum('pds,nd,ns->np', t, d, sv), 1e-12)
        d = n.sum((1, 3)) / np.maximum(np.einsum('pds,np,ns->nd', t, p, sv), 1e-12)
        md = d.mean(1, keepdims=True); md[md <= 0] = 1; d, p = d / md, p * md
        sv = n.sum((1, 2)) / np.maximum(np.einsum('pds,np,nd->ns', t, p, d), 1e-12)
        ms_ = sv.mean(1, keepdims=True); ms_[ms_ <= 0] = 1; sv, p = sv / ms_, p * ms_
    return p, d, sv


# ---------------------------------------------------------------------------
# one session: context, statistics, nulls
# ---------------------------------------------------------------------------

def _context(view: dict, cfg: dict, excluded_walls: set) -> dict:
    """Arrays shared by the real statistics and every shift of one arena session."""
    counts = view['counts']
    xy = view['xy']
    T = min(counts.shape[1], len(xy))
    counts = np.asarray(counts[:, :T], np.float32)
    xy = np.asarray(xy[:T], float)
    side = float(view['arena']['side_cm'])
    fin = np.isfinite(xy).all(axis=1)
    clipped = fin & ((xy < 0) | (xy >= side)).any(axis=1)
    xy_c = np.clip(xy, 0, side - 1e-6)
    speed = sc.speed_cm_s(xy)
    moving = fin & (speed > cfg['min_speed'])
    hd = view['hd'][:T] if view['hd'] is not None else None
    guard = dict(ok=False, offset_deg=np.nan, R=np.nan, n=0)
    if hd is not None:
        h = cfg['hd']
        guard = sc.hd_guard(xy, hd, speed, h['guard_speed'], h['guard_offset_deg'], h['guard_r'])
        if not guard['ok']:
            hd = None
    bc = cfg['bin_cm']
    ex = ey = sc.fixed_edges(0.0, side, bc)
    nb = (len(ex) - 1, len(ey) - 1)
    flat = sc.bin_index(xy_c, ex, ey)
    X, Y = np.meshgrid(bin_centres(ex), bin_centres(ey), indexing='ij')
    objs = []
    in_mask_t = np.zeros(T, bool)
    valid_bins = np.ones(nb, bool)
    for o in (view.get('objects') or []):
        c = np.asarray(o['xy_cm'], float)
        if not (np.all(c >= 0) and np.all(c <= side)):
            raise ValueError(f"object {o.get('label', '')} at {c.tolist()} lies outside the {side} cm arena")
        r_obj = measure_object_radius(xy_c[fin], c, cfg)
        r_mask = r_obj + cfg['obj']['margin_cm']
        objs.append(dict(label=o.get('label', ''), xy=c, r_obj=r_obj, r_mask=r_mask))
        in_mask_t |= np.linalg.norm(xy_c - c, axis=1) < r_mask
        valid_bins &= np.hypot(X - c[0], Y - c[1]) >= r_mask
    half = np.arange(T) < T // 2
    use = moving & ~in_mask_t
    Ms = {k: sc.onehot(flat, m, nb[0] * nb[1]) for k, m in
          (('all', moving), ('use', use), ('h1', use & half), ('h2', use & ~half))}
    occ = {k: sc.occupancy_map(M, nb) for k, M in Ms.items()}
    ctx = dict(T=T, counts=counts, xy=xy_c, side=side, speed=speed, moving=moving, use=use, half=half,
               hd=hd, guard=guard, ex=ex, ey=ey, nb=nb, flat=flat, Ms=Ms, occ=occ, objs=objs,
               valid_bins=valid_bins, clipped_frac=float(clipped.sum() / max(fin.sum(), 1)),
               excluded_walls=set(excluded_walls), cfg=cfg,
               geo=sc_grid_geometry(nb, bc, side, cfg))
    # head direction indicators: 10 deg curve bins and the coarse pxd joint bins
    if hd is not None:
        h = cfg['hd']
        hm = use & np.isfinite(hd)
        hd0 = np.where(np.isfinite(hd), hd, 0.0)
        dbin = np.floor((np.mod(hd0, 2 * np.pi)) / (2 * np.pi) * h['nbins']).astype(int)
        dbin = np.where(hm, np.clip(dbin, 0, h['nbins'] - 1), -1)
        ctx['Mhd'] = sc.onehot(dbin, hm, h['nbins'])
        nd = int(round(360 / h['pxd_dir_deg']))
        np_ = max(1, int(round(side / h['pxd_pos_cm'])))
        pe = np.linspace(0, side, np_ + 1)
        pflat = sc.bin_index(xy_c, pe, pe)
        jd = np.floor(np.mod(hd0, 2 * np.pi) / (2 * np.pi) * nd).astype(int)
        jflat = np.where(hm & (pflat >= 0), pflat * nd + np.clip(jd, 0, nd - 1), -1)
        for k, m in (('all', hm), ('h1', hm & half), ('h2', hm & ~half)):
            ctx['Mj_' + k] = sc.onehot(jflat, m & (jflat >= 0), np_ * np_ * nd)
            ctx['tj_' + k] = sc.occupancy_map(ctx['Mj_' + k], (np_ * np_, nd))
        ctx['pxd_shape'] = (np_ * np_, nd)
        ctx['hd_centres'] = (np.arange(h['nbins']) + 0.5) * 2 * np.pi / h['nbins']
        ctx['pxd_centres'] = (np.arange(nd) + 0.5) * 2 * np.pi / nd
        ctx['occ_hd'] = np.asarray(ctx['Mhd'].sum(axis=0)).ravel() * sc.BIN_S
    sp = cfg['speed']
    ctx['speed_sel'] = use & np.isfinite(speed) & (speed >= sp['lo']) & (speed <= sp['hi'])
    ctx['speed_sigma_bins'] = sp['sigma_s'] / sc.BIN_S
    # position x direction x speed joint bins: the spatial class is decided on the position factor,
    # because raw maps of pure HD cells (95 %) and speed cells (80 %) passed as spatial in the gate
    h, pd_ = cfg['hd'], cfg['pds']
    np3 = max(1, int(round(side / h['pxd_pos_cm'])))
    pe3 = np.linspace(0, side, np3 + 1)
    pflat3 = sc.bin_index(xy_c, pe3, pe3)
    se = np.asarray(pd_['speed_edges'], float)
    sbin = np.searchsorted(se, np.nan_to_num(speed, nan=-1.0), side='right') - 1
    nsb = len(se) - 1
    ok3 = use & (pflat3 >= 0) & (sbin >= 0) & (sbin < nsb)
    if hd is not None:
        nd3 = int(round(360 / h['pxd_dir_deg']))
        d3 = np.floor(np.mod(np.where(np.isfinite(hd), hd, 0.0), 2 * np.pi) / (2 * np.pi) * nd3).astype(int)
        ok3 &= np.isfinite(hd)
    else:
        nd3, d3 = 1, np.zeros(T, int)
    flat3 = np.where(ok3, (pflat3 * nd3 + np.clip(d3, 0, nd3 - 1)) * nsb + np.clip(sbin, 0, nsb - 1), -1)
    ctx['pds_shape'] = (np3 * np3, nd3, nsb)
    hi_edges = np.r_[se[1:-1], 50.0]
    ctx['speed_logc'] = np.log(np.sqrt(se[:-1] * hi_edges))       # geometric bin centres, last capped at 50
    for k, m in (('all', ok3), ('h1', ok3 & half), ('h2', ok3 & ~half)):
        ctx['M3_' + k] = sc.onehot(flat3, m & (flat3 >= 0), np3 * np3 * nd3 * nsb)
        ctx['t3_' + k] = sc.occupancy_map(ctx['M3_' + k], ctx['pds_shape'])
    return ctx


def sc_grid_geometry(nb: tuple, bin_cm: float, side: float, cfg: dict) -> GridGeometry:
    return GridGeometry(nb[0], nb[1], bin_cm, cfg['grid']['rmax_frac'] * side)


def _rates(ctx: dict, counts: np.ndarray, key: str) -> np.ndarray:
    spk = sc.spike_maps(counts, ctx['Ms'][key], ctx['nb'])
    return sc.rate_maps_separate(spk, ctx['occ'][key], ctx['cfg']['sigma_bins'], valid=ctx['valid_bins'],
                                 min_smoothed_occ=ctx['cfg']['min_smoothed_occ_s'])


def _corr_rows(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,) Pearson r of each row of A with vector b."""
    A = A - A.mean(1, keepdims=True)
    b = b - b.mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        return (A @ b) / (np.sqrt((A ** 2).sum(1)) * np.sqrt((b ** 2).sum()))


def session_stats(ctx: dict, counts: np.ndarray, rate_s: np.ndarray, full: bool) -> dict:
    """Every per-unit statistic for one (real or shifted) spike array; `full` also keeps maps and details."""
    cfg = ctx['cfg']
    N = counts.shape[0]
    rate = _rates(ctx, counts, 'use')
    r1, r2 = _rates(ctx, counts, 'h1'), _rates(ctx, counts, 'h2')
    out = {k: np.full(N, np.nan) for k in STATS}
    out['si'] = sc.skaggs_info(rate, ctx['occ']['use'])
    out['stab'] = sc.map_corr(r1, r2)
    det = dict(grid=[], border=[])
    nogrid = dict(grid=np.nan, inner_cm=np.nan, outer_cm=np.nan, spacing_cm=np.nan, orientation_deg=np.nan, n_peaks=0)
    for i in range(N):
        gi = (gridness(autocorr(rate[i], cfg['grid']['min_overlap']), ctx['geo'], cfg)
              if cfg['grid']['enabled'] else nogrid)
        bi = border_score(rate[i], ctx['side'], cfg['bin_cm'], ctx['excluded_walls'], cfg)
        out['grid'][i], out['border'][i] = gi['grid'], bi['border']
        if full:
            det['grid'].append(gi)
            det['border'].append(bi)
    if ctx['hd'] is not None:
        h = cfg['hd']
        hc = np.asarray((ctx['Mhd'].T @ counts.T).T) / np.maximum(ctx['occ_hd'], 1e-9)[None]
        out['hd_r'], hd_pref = vector_length(hc, ctx['hd_centres'])
        res = {}
        for k in ('all', 'h1', 'h2'):
            nj = np.asarray((ctx['Mj_' + k].T @ counts.T).T).reshape((N,) + ctx['pxd_shape'])
            res[k] = pxd(nj, ctx['tj_' + k], h['pxd_tol'], h['pxd_maxit'])
        out['pxd_r'], pxd_pref = vector_length(res['all'][1], ctx['pxd_centres'])
        out['pxd_split'] = sc.map_corr(res['h1'][1][:, :, None], res['h2'][1][:, :, None], min_bins=4)
        if full:
            det.update(hd_curve=hc, hd_pref=hd_pref, pxd_d=res['all'][1], pxd_pref=pxd_pref,
                       pxd_converged=res['all'][2], pxd_iters=res['all'][3])
    pf, sf = {}, {}
    for k in ('all', 'h1', 'h2'):
        n3 = np.asarray((ctx['M3_' + k].T @ counts.T).T).reshape((N,) + ctx['pds_shape'])
        res3 = factorize3(n3, ctx['t3_' + k], cfg['pds']['tol'], cfg['pds']['maxit'])
        pf[k], sf[k] = res3[0], res3[2]
        # speed factor vs log speed over bins with dwell: tuning beyond position and direction
        dwell = ctx['t3_' + k].sum((0, 1)) > 0
        if dwell.sum() >= 3:
            out[{'all': 'speed_f', 'h1': 'speed_f1', 'h2': 'speed_f2'}[k]] = _corr_rows(
                sf[k][:, dwell], ctx['speed_logc'][dwell])
    occ_p = ctx['t3_all'].sum((1, 2))
    pmap = np.where(occ_p[None] > 0, pf['all'], np.nan)
    out['si_p'] = sc.skaggs_info(pmap[:, :, None], occ_p[:, None])
    o1, o2 = ctx['t3_h1'].sum((1, 2)), ctx['t3_h2'].sum((1, 2))
    out['stab_p'] = sc.map_corr(np.where(o1[None] > 0, pf['h1'], np.nan), np.where(o2[None] > 0, pf['h2'], np.nan), min_bins=10)
    if full:
        det['p_factor'] = pmap
    sel = ctx['speed_sel']
    for key, m in (('speed_r', sel), ('speed_r1', sel & ctx['half']), ('speed_r2', sel & ~ctx['half'])):
        if m.sum() > 100:
            out[key] = _corr_rows(rate_s[:, m], ctx['speed'][m])
    if full:
        det['rate'] = rate
        det['rate_h1'], det['rate_h2'] = r1, r2
        if cfg['grid']['enabled']:
            det['autocorr'] = np.array([autocorr(rate[i], cfg['grid']['min_overlap']) for i in range(N)])
        out['details'] = det
    return out


def _smoothed_rate(ctx: dict, counts: np.ndarray) -> np.ndarray:
    return gaussian_filter1d(counts / sc.BIN_S, ctx['speed_sigma_bins'], axis=1, mode='wrap')


def _null_chunk(ctx: dict, shifts: np.ndarray) -> dict:
    """Statistics for a chunk of circular shifts (runs in a joblib worker)."""
    rs = _smoothed_rate(ctx, ctx['counts'])
    acc = {k: [] for k in STATS}
    for k in shifts:
        st = session_stats(ctx, np.roll(ctx['counts'], int(k), axis=1), np.roll(rs, int(k), axis=1), full=False)
        for key in STATS:
            acc[key].append(st[key])
    return {k: np.array(v) for k, v in acc.items()}


def session_null(ctx: dict, shifts: np.ndarray, n_jobs: int) -> dict:
    """{stat: (n_shifts, N)} under circular spike shifts."""
    chunks = [c for c in np.array_split(shifts, max(1, min(n_jobs * 4, len(shifts)))) if len(c)]
    if n_jobs > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_null_chunk)(ctx, c) for c in chunks)
    else:
        parts = [_null_chunk(ctx, c) for c in chunks]
    return {k: np.concatenate([p[k] for p in parts]) for k in STATS}


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------

def _speed_matched(idx_band: np.ndarray, idx_rest: np.ndarray, speed: np.ndarray, width: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Subsample `idx_rest` so its speed histogram matches `idx_band` (bins of `width` cm/s)."""
    bb = np.floor(speed[idx_band] / width).astype(int)
    br = np.floor(speed[idx_rest] / width).astype(int)
    keep = []
    for b in np.unique(bb):
        need = int((bb == b).sum() * len(idx_rest) / max(len(idx_band), 1))
        pool = idx_rest[br == b]
        if len(pool):
            keep.append(rng.choice(pool, min(need, len(pool)), replace=False))
    return np.concatenate(keep) if keep else idx_rest[:0]


def proximal_stat(ctx: dict, centre, r_obj: float, rng: np.random.Generator) -> np.ndarray | None:
    """(N,) log rate ratio of the most active angular sector of the band (object surface to +band cm)
    over the rest of the floor outside every object's mask + band, on speed-matched samples.

    A sector, not the whole ring: a field 10 cm off one side of an object is diluted ~25-fold in the
    full annulus (gate A12 detected 20 % of such cells with the ring mean). None when a region is thin.
    """
    cfg = ctx['cfg']['obj']
    xy, base = ctx['xy'], ctx['moving']
    rel = xy - np.asarray(centre, float)
    d = np.linalg.norm(rel, axis=1) - r_obj
    band = base & (d >= 0) & (d < cfg['band_cm'])
    rest = base & (d >= cfg['band_cm'])
    for o2 in ctx['objs']:
        rest &= np.linalg.norm(xy - o2['xy'], axis=1) >= o2['r_mask'] + cfg['band_cm']
    ib, ir = np.nonzero(band)[0], np.nonzero(rest)[0]
    if len(ib) < 400 or len(ir) < 400:
        return None
    ir = _speed_matched(ib, ir, ctx['speed'], cfg['speed_bin'], rng)
    if len(ir) < 400:
        return None
    ns = cfg['n_sectors']
    sec = (np.floor(np.mod(np.arctan2(rel[ib, 1], rel[ib, 0]), 2 * np.pi) / (2 * np.pi) * ns)).astype(int)
    rr = ctx['counts'][:, ir].sum(1) / (len(ir) * sc.BIN_S)
    best = np.full(ctx['counts'].shape[0], -np.inf)
    for k in range(ns):
        ik = ib[sec == k]
        if len(ik) < 100:
            continue
        rb = ctx['counts'][:, ik].sum(1) / (len(ik) * sc.BIN_S)
        best = np.maximum(best, np.log((rb + 0.1) / (rr + 0.1)))
    return np.where(np.isfinite(best), best, np.nan)


def proximal_with_null(ctx: dict, centre, r_obj: float, r_mask: float, rng: np.random.Generator) -> dict:
    """The proximal statistic at `centre` and at >= min_virtual virtual centres whose wall distance is
    within +-wall_tol of the real one and whose band cannot reach the real location."""
    cfg = ctx['cfg']['obj']
    side = ctx['side']
    centre = np.asarray(centre, float)
    real = proximal_stat(ctx, centre, r_obj, rng)
    wd = wall_distance(centre[0], centre[1], side)
    g = np.arange(0.5, side, 1.0)
    GX, GY = np.meshgrid(g, g, indexing='ij')
    # virtual bands must not overlap the real object's band (a nearby virtual band absorbed the planted
    # field in gate A12), nor any other object's
    sep = 2 * (r_obj + cfg['band_cm'])
    cand = ((np.abs(wall_distance(GX, GY, side) - wd) <= cfg['wall_tol_cm'])
            & (np.hypot(GX - centre[0], GY - centre[1]) >= sep))
    for o2 in ctx['objs']:
        cand &= np.hypot(GX - o2['xy'][0], GY - o2['xy'][1]) >= r_obj + cfg['band_cm'] + o2['r_obj'] + cfg['band_cm']
    pts = np.column_stack([GX[cand], GY[cand]])
    out = dict(xy=centre, r_obj=r_obj, r_mask=r_mask, stat=real, null=None, n_virtual=0, reason='')
    if real is None:
        out['reason'] = 'too few samples in the band or the rest'
        return out
    if len(pts) < cfg['min_virtual']:
        out['reason'] = f'{len(pts)} wall-distance-matched virtual positions'
        return out
    pick = pts[rng.choice(len(pts), min(cfg['n_virtual'], len(pts)), replace=False)]
    vals = [v for v in (proximal_stat(ctx, p, r_obj, rng) for p in pick) if v is not None]
    if len(vals) < cfg['min_virtual']:
        out['reason'] = f'{len(vals)} usable virtual positions'
        return out
    null = np.array(vals)
    q = ctx['cfg']['q']['unit']
    out.update(null=null, n_virtual=len(vals), own_p99=np.nanpercentile(null, q, axis=0),
               own_p95=np.nanpercentile(null, 95, axis=0))
    return out


def object_proximal(ctx: dict, rng: np.random.Generator) -> list:
    """One `proximal_with_null` result per object of the session."""
    return [dict(label=o['label'], **proximal_with_null(ctx, o['xy'], o['r_obj'], o['r_mask'], rng))
            for o in ctx['objs']]


# ---------------------------------------------------------------------------
# rotation between two arena sessions of one sort
# ---------------------------------------------------------------------------

DIHEDRAL = {
    'id': lambda m: m, 'rot90': lambda m: np.rot90(m, 1, axes=(-2, -1)), 'rot180': lambda m: np.rot90(m, 2, axes=(-2, -1)),
    'rot270': lambda m: np.rot90(m, 3, axes=(-2, -1)), 'flip_x': lambda m: m[..., ::-1, :],
    'flip_y': lambda m: m[..., :, ::-1], 'transpose': lambda m: np.swapaxes(m, -1, -2),
    'antitranspose': lambda m: np.rot90(np.swapaxes(m, -1, -2), 2, axes=(-2, -1)),
}


def transform_geometry(name: str, n: int = 21) -> dict:
    """How a dihedral map transform moves directions in the y-down arena frame: a rotation angle (rad)
    or a reflection axis; measured by transforming a map with one hot bin east and one south."""
    c = n // 2
    pts = {}
    for lab, (i, j) in (('e', (n - 1, c)), ('s', (c, n - 1))):
        m = np.zeros((n, n))
        m[i, j] = 1
        ti, tj = np.unravel_index(np.argmax(DIHEDRAL[name](m)), (n, n))
        pts[lab] = np.arctan2(tj - c, ti - c)
    e, s = pts['e'], pts['s']
    det = np.sin(s - e)                       # +1 for a proper rotation in this frame
    if det > 0:
        return dict(kind='rotation', angle=float(np.angle(np.exp(1j * e))))
    return dict(kind='reflection', axis=float(np.angle(np.exp(1j * e)) / 2))


def rotation_test(a: dict, b: dict, units_ok: np.ndarray, cfg: dict, rng: np.random.Generator,
                  hd_units: np.ndarray | None = None) -> dict:
    """Best dihedral transform of session-b maps onto session-a maps (median same-unit r over units stable
    in both), its unit-shuffle null, margin, leave-one-out r, and the HD preferred-direction shift."""
    ra, rb = a['details']['rate'], b['details']['rate']
    if ra.shape != rb.shape:
        return dict(ok=False, reason='map shapes differ')
    sel = np.nonzero(units_ok)[0]
    if len(sel) < 4:
        return dict(ok=False, reason=f'{len(sel)} units stable in both sessions')
    R = {k: sc.map_corr(ra[sel], f(rb[sel])) for k, f in DIHEDRAL.items()}
    med = {k: float(np.nanmedian(v)) for k, v in R.items()}
    order = sorted(med, key=lambda k: -med[k] if np.isfinite(med[k]) else np.inf)
    best = order[0]
    nperm = cfg['rotation']['n_perm']
    null = np.empty(nperm)
    for p in range(nperm):
        perm = rng.permutation(sel)
        null[p] = max(np.nanmedian(sc.map_corr(ra[sel], f(rb[perm]))) for f in DIHEDRAL.values())
    loo = np.full(len(sel), np.nan)
    for j in range(len(sel)):
        keep = np.delete(np.arange(len(sel)), j)
        bk = max(DIHEDRAL, key=lambda k: np.nanmedian(R[k][keep]))
        loo[j] = R[bk][j]
    geo = transform_geometry(best)                       # maps session b onto a
    geo_ab = transform_geometry(inverse_transform(best))  # takes session a to b: what HD prefs follow
    out = dict(ok=True, n_units=len(sel), units=sel, median_r=med, best=best, best_r=med[best],
               second=order[1], margin=med[best] - med[order[1]], null_p99=float(np.percentile(null, 99)),
               p=float((1 + (null >= med[best]).sum()) / (1 + nperm)), loo_r=loo, transform=geo)
    da, db = a['details'].get('pxd_pref'), b['details'].get('pxd_pref')
    if da is not None and db is not None:
        # HD units need not be spatially stable: select on eligibility and the HD class in both sessions
        base = units_ok if hd_units is None else hd_units
        hsel = np.nonzero(base & np.isfinite(da) & np.isfinite(db) & a['hd_class'] & b['hd_class'])[0]
        if len(hsel) >= 3:
            if geo_ab['kind'] == 'rotation':
                dd = np.angle(np.exp(1j * (db[hsel] - da[hsel])))
                z = np.exp(1j * dd).mean()
                boot = [np.angle(np.exp(1j * dd[rng.integers(0, len(dd), len(dd))]).mean())
                        for _ in range(cfg['rotation']['n_boot'])]
                lo, hi = np.percentile(np.angle(np.exp(1j * (np.array(boot) - np.angle(z)))), [2.5, 97.5]) + np.angle(z)
                v = np.sqrt(2 * len(dd)) * np.mean(np.cos(dd - geo_ab['angle']))
                out['hd'] = dict(n=len(hsel), mean_shift_deg=float(np.degrees(np.angle(z))), R=float(abs(z)),
                                 ci_deg=(float(np.degrees(lo)), float(np.degrees(hi))),
                                 v_stat=float(v), expected_deg=float(np.degrees(geo_ab['angle'])))
            else:
                s = np.angle(np.exp(1j * (da[hsel] + db[hsel])).mean()) / 2
                out['hd'] = dict(n=len(hsel), reflection_axis_deg=float(np.degrees(s)),
                                 R=float(abs(np.exp(1j * (da[hsel] + db[hsel])).mean())),
                                 expected_axis_deg=float(np.degrees(geo_ab['axis'])))
    return out


# ---------------------------------------------------------------------------
# classes, thresholds, chance
# ---------------------------------------------------------------------------

def classify(real: dict, null: dict, pools: dict, cfg: dict) -> dict:
    """Thresholds per pool and class booleans for the real data and for every shift (chance counts)."""
    q = cfg['q']
    N = len(real['si'])
    thr = {}
    for pool, members in pools.items():
        thr[pool] = {k: sc.pooled_threshold(null[k], members, q['pooled']) for k in ('grid', 'border')}
        thr[pool]['speed'] = sc.pooled_threshold(np.abs(null['speed_r']), members, q['pooled'])
        thr[pool]['speed_f'] = sc.pooled_threshold(np.abs(null['speed_f']), members, q['pooled'])
    own = dict(si=sc.unit_threshold(null['si'], q['unit']), pxd_r=sc.unit_threshold(null['pxd_r'], q['unit']),
               si_p=sc.unit_threshold(null['si_p'], q['unit']), stab_p=sc.unit_threshold(null['stab_p'], q['stability']),
               stab=sc.unit_threshold(null['stab'], q['stability']),
               pxd_split=sc.unit_threshold(null['pxd_split'], q['stability']),
               speed_r1=sc.unit_threshold(np.abs(null['speed_r1']), q['stability']),
               speed_r2=sc.unit_threshold(np.abs(null['speed_r2']), q['stability']),
               speed_f1=sc.unit_threshold(np.abs(null['speed_f1']), q['stability']),
               speed_f2=sc.unit_threshold(np.abs(null['speed_f2']), q['stability']))
    pool_of = np.full(N, '', dtype=object)
    for pool, members in pools.items():
        pool_of[members] = pool

    def classes(st: dict) -> dict:
        g = np.array([thr.get(p, {}).get('grid', np.inf) for p in pool_of])
        b = np.array([thr.get(p, {}).get('border', np.inf) for p in pool_of])
        s = np.array([thr.get(p, {}).get('speed_f', np.inf) for p in pool_of])
        stable = st['stab'] > own['stab']
        c = dict(grid=(st['grid'] > g) & stable, border=(st['border'] > b) & stable)
        c['spatial'] = ((st['si_p'] > own['si_p']) & (st['stab_p'] > own['stab_p'])
                        & ~c['grid'] & ~c['border'])
        c['hd'] = (st['pxd_r'] > own['pxd_r']) & (st['pxd_split'] > own['pxd_split'])
        same = np.sign(st['speed_f1']) == np.sign(st['speed_f2'])
        c['speed'] = ((np.abs(st['speed_f']) > s) & same & (np.abs(st['speed_f1']) > own['speed_f1'])
                      & (np.abs(st['speed_f2']) > own['speed_f2']))
        return c

    real_c = classes(real)
    ns = len(null['si'])
    chance = {k: {p: np.zeros(ns, int) for p in pools} for k in real_c}
    for j in range(ns):
        cj = classes({k: null[k][j] for k in STATS})
        for k, v in cj.items():
            for p, members in pools.items():
                chance[k][p][j] = int(v[members].sum())
    return dict(thresholds=thr, own=own, classes=real_c, chance=chance)


# ---------------------------------------------------------------------------
# recday entry point
# ---------------------------------------------------------------------------

def unit_pools(units, rates_hz: np.ndarray, moving_spikes: np.ndarray, cfg: dict) -> tuple:
    """(pools {'broad', 'narrow'} -> unit rows, eligible mask). Ineligible: < min moving spikes, or the
    lower-spike member of a duplicate pair. Narrow by waveform width; by rate when width is missing."""
    u = cfg['units']
    width = units['width_ms'].to_numpy(float)
    narrow = np.where(np.isfinite(width), width < u['narrow_ms'], rates_hz > u['rate_cut_hz'])
    elig = moving_spikes >= u['min_moving_spikes']
    dup = units['dup_partner'].to_numpy(int)
    for i, j in enumerate(dup):
        if j >= 0 and moving_spikes[j] > moving_spikes[i]:
            elig[i] = False
    pools = dict(broad=np.nonzero(elig & ~narrow)[0], narrow=np.nonzero(elig & narrow)[0])
    return pools, elig, narrow


def run_of_recday(views: dict, recday: str, units, cfg: dict, wall_exclusions=None) -> dict:
    """Score every arena session of one recday; returns the shard dict (no file I/O here)."""
    cfg = copy.deepcopy(cfg)
    rng = np.random.default_rng(cfg['seed'])
    arena_sessions = sorted(s for s, v in views.items() if v.get('session_type') in ('openfield', 'object')
                            and v.get('xy') is not None and v.get('arena') is not None)
    sessions, ctxs = {}, {}
    for s in arena_sessions:
        v = views[s]
        excl = set((wall_exclusions or {}).get(s, {}).keys())
        ctx = _context(v, cfg, excl)
        ctxs[s] = ctx
        rs = _smoothed_rate(ctx, ctx['counts'])
        real = session_stats(ctx, ctx['counts'], rs, full=True)
        shifts = sc.shift_offsets(ctx['T'], cfg['n_shifts'], int(round(cfg['min_shift_s'] / sc.BIN_S)), rng)
        null = session_null(ctx, shifts, cfg['n_jobs'])
        mov_spk = ctx['counts'][:, ctx['use']].sum(1)
        rate_hz = ctx['counts'].sum(1) / (ctx['T'] * sc.BIN_S)
        pools, elig, narrow = unit_pools(units, rate_hz, mov_spk, cfg)
        cl = classify(real, null, pools, cfg)
        objs = object_proximal(ctx, rng) if ctx['objs'] else []
        for o in objs:
            o['proximal'] = (elig & (o['stat'] > o['own_p99'])) if o['null'] is not None else np.zeros(len(elig), bool)
        sessions[s] = dict(
            session=s, session_type=v['session_type'], side_cm=ctx['side'], T=ctx['T'],
            clipped_frac=ctx['clipped_frac'], hd_guard=ctx['guard'], excluded_walls=sorted(excl),
            moving_frac=float(ctx['moving'].mean()), objects=objs, shifts=shifts,
            real={k: real[k] for k in STATS}, details=real['details'], null=null,
            thresholds=cl['thresholds'], own=cl['own'], classes=cl['classes'], chance=cl['chance'],
            pools={k: v_.tolist() for k, v_ in pools.items()}, eligible=elig, narrow=narrow,
            rate_hz=rate_hz, moving_spikes=mov_spk, occ=ctx['occ']['use'], edges=ctx['ex'],
            valid_bins=ctx['valid_bins'], xy=ctx['xy'].astype(np.float32), hd_class=cl['classes']['hd'])
    rotations = {}
    for i, s1 in enumerate(arena_sessions):
        for s2 in arena_sessions[i + 1:]:
            a, b = sessions[s1], sessions[s2]
            ok = ((a['real']['stab'] > a['own']['stab']) & (b['real']['stab'] > b['own']['stab'])
                  & a['eligible'] & b['eligible'])
            rot = rotation_test(a, b, ok, cfg, rng, hd_units=a['eligible'] & b['eligible'])
            rot['object_vector'] = object_vector(ctxs, s1, s2, sessions, rot, rng)
            rotations[(s1, s2)] = rot
    return dict(recday=recday, engine=ENGINE, engine_version=ENGINE_VERSION, sessions=sessions,
                rotations=rotations, units=units)


def inverse_transform(name: str) -> str:
    """The dihedral transform that undoes `name`."""
    t = np.arange(25.0).reshape(5, 5)
    return next(k for k, f in DIHEDRAL.items() if np.array_equal(f(DIHEDRAL[name](t)), t))


def transform_point(name: str, xy, side: float, bin_cm: float):
    """Where a dihedral map transform sends the point `xy` of a square arena of `side` cm."""
    n = int(round(side / bin_cm))
    m = np.zeros((n, n))
    i, j = (np.clip((np.asarray(xy, float) / bin_cm).astype(int), 0, n - 1))
    m[i, j] = 1
    ti, tj = np.unravel_index(np.argmax(DIHEDRAL[name](m)), (n, n))
    return np.array([(ti + 0.5) * bin_cm, (tj + 0.5) * bin_cm])


def object_vector(ctxs: dict, s1: int, s2: int, sessions: dict, rot: dict, rng) -> dict:
    """Within one sort, firing that appears with the object.

    effect = proximal statistic at the object (object session) - the same statistic at the object's
    rotation-aligned location in the paired open-field session ('ghost'); null = the same difference
    with each session's spikes circularly shifted. Needs no virtual objects, so it also works for the
    04-03 bottle at the arena centre, where no wall-distance-matched virtual position exists.
    """
    if not rot.get('ok'):
        return {}
    a, b = sessions[s1], sessions[s2]
    if bool(a['objects']) == bool(b['objects']):
        return {}
    obj_s, of_s = (s2, s1) if b['objects'] else (s1, s2)
    name = rot['best'] if obj_s == s2 else inverse_transform(rot['best'])     # f(session2) ~ session1
    c_obj, c_of = ctxs[obj_s], ctxs[of_s]
    min_shift = int(round(c_obj['cfg']['min_shift_s'] / sc.BIN_S))
    nsh = c_obj['cfg']['rotation']['n_shift_object']
    elig = sessions[obj_s]['eligible'] & sessions[of_s]['eligible']
    res = {}
    for o in c_obj['objs']:
        ghost_xy = transform_point(name, o['xy'], c_of['side'], c_of['cfg']['bin_cm'])
        eo = proximal_stat(c_obj, o['xy'], o['r_obj'], rng)
        eg = proximal_stat(c_of, ghost_xy, o['r_obj'], rng)
        key = o['label'] or 'object'
        if eo is None or eg is None:
            res[key] = dict(reason='too few samples near the object or its ghost', ghost_xy=ghost_xy)
            continue
        effect = eo - eg
        null = []
        so = sc.shift_offsets(c_obj['T'], nsh, min_shift, rng)
        sg = sc.shift_offsets(c_of['T'], nsh, min_shift, rng)
        co, cg = c_obj['counts'], c_of['counts']
        for k1, k2 in zip(so, sg):
            c_obj['counts'] = np.roll(co, int(k1), axis=1)
            c_of['counts'] = np.roll(cg, int(k2), axis=1)
            x = proximal_stat(c_obj, o['xy'], o['r_obj'], rng)
            y = proximal_stat(c_of, ghost_xy, o['r_obj'], rng)
            if x is not None and y is not None:
                null.append(x - y)
        c_obj['counts'], c_of['counts'] = co, cg
        null = np.array(null)
        p99 = np.nanpercentile(null, c_obj['cfg']['q']['unit'], axis=0)
        res[key] = dict(ghost_xy=ghost_xy, transform=name, object_stat=eo, ghost_stat=eg, effect=effect,
                        null_n=len(null), own_p99=p99, object_vector=elig & (effect > p99))
    return res


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def merge_tables(shards: dict) -> dict:
    """{'units', 'sessions', 'rotations'} DataFrames from a run's shards. Counts are per recday and
    per pool, each with its measured chance distribution (one shift applied to all units at once)."""
    import pandas as pd
    urows, srows, rrows = [], [], []
    for rd, sh in sorted(shards.items()):
        units = sh['units'].reset_index(drop=True)
        mouse = rd.split('_')[0]
        for s, ss in sorted(sh['sessions'].items()):
            N = len(ss['rate_hz'])
            det = ss['details']
            for i in range(N):
                pool = 'narrow' if ss['narrow'][i] else 'broad'
                thr = ss['thresholds'].get(pool, {})
                row = dict(recday=rd, mouse=mouse, session=s, session_type=ss['session_type'], unit=i,
                           cluster_id=units.loc[i, 'cluster_id'], region=units.loc[i, 'region'],
                           width_ms=units.loc[i, 'width_ms'], narrow=bool(ss['narrow'][i]),
                           eligible=bool(ss['eligible'][i]), rate_hz=float(ss['rate_hz'][i]),
                           moving_spikes=float(ss['moving_spikes'][i]))
                row.update({k: float(ss['real'][k][i]) for k in STATS})
                row.update(grid_thr=thr.get('grid', np.nan), border_thr=thr.get('border', np.nan),
                           speed_thr=thr.get('speed_f', np.nan), kropff_thr=thr.get('speed', np.nan),
                           **{f'{k}_own_thr': float(v[i]) for k, v in ss['own'].items()})
                row.update({f'is_{k}': bool(v[i]) for k, v in ss['classes'].items()})
                row.update(grid_spacing_cm=det['grid'][i]['spacing_cm'],
                           grid_orientation_deg=det['grid'][i]['orientation_deg'],
                           border_wall=det['border'][i]['wall'])
                if 'pxd_pref' in det:
                    row['pxd_pref_deg'] = float(np.degrees(det['pxd_pref'][i]))
                for o in ss['objects']:
                    lab = o['label'] or 'object'
                    row[f'proximal_stat_{lab}'] = np.nan if o['stat'] is None else float(o['stat'][i])
                    row[f'is_proximal_{lab}'] = bool(o['proximal'][i])
                urows.append(row)
            for pool, members in ss['pools'].items():
                members = np.asarray(members, int)
                for k, v in ss['classes'].items():
                    obs = int(np.asarray(v)[members].sum())
                    ch = np.asarray(ss['chance'][k][pool])
                    srows.append(dict(recday=rd, mouse=mouse, session=s, session_type=ss['session_type'],
                                      pool=pool, cls=k, n_units=len(members), observed=obs,
                                      chance_mean=float(ch.mean()), chance_p95=float(np.percentile(ch, 95)),
                                      p=float((1 + (ch >= obs).sum()) / (1 + len(ch))),
                                      hd_guard_ok=bool(ss['hd_guard']['ok']), clipped_frac=ss['clipped_frac'],
                                      excluded_walls=';'.join(ss['excluded_walls'])))
            for o in ss['objects']:
                srows.append(dict(recday=rd, mouse=mouse, session=s, session_type=ss['session_type'],
                                  pool='eligible', cls=f'proximal_{o["label"] or "object"}',
                                  n_units=int(ss['eligible'].sum()), observed=int(o['proximal'].sum()),
                                  chance_mean=np.nan, chance_p95=np.nan, p=np.nan,
                                  note=o['reason'] or f'{o["n_virtual"]} virtual objects, r_obj {o["r_obj"]:.1f} cm'))
        for (s1, s2), rot in sh['rotations'].items():
            row = dict(recday=rd, mouse=mouse, s1=s1, s2=s2, ok=rot.get('ok', False), reason=rot.get('reason', ''))
            if rot.get('ok'):
                row.update(best=rot['best'], best_r=rot['best_r'], second=rot['second'], margin=rot['margin'],
                           null_p99=rot['null_p99'], p=rot['p'], n_units=rot['n_units'],
                           transform_kind=rot['transform']['kind'])
                hd = rot.get('hd', {})
                row.update({f'hd_{k}': (v if not isinstance(v, tuple) else f'{v[0]:.0f}..{v[1]:.0f}') for k, v in hd.items()})
                for lab, ov in rot.get('object_vector', {}).items():
                    row[f'object_vector_{lab}'] = int(ov['object_vector'].sum()) if 'object_vector' in ov else ov.get('reason', '')
            rrows.append(row)
    return dict(units=pd.DataFrame(urows), sessions=pd.DataFrame(srows), rotations=pd.DataFrame(rrows))
