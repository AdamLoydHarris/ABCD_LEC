"""Spike-to-track lag scan: tests the spike/tracking alignment of arena sessions against the maze
sessions of the same recday (same spike clock, separately synchronised tracking).

    python code/of_lag_scan.py --dataset mec          (sbatch: it loads the data_dic)

Spikes at t are paired with behaviour at t + lag, lag -1..+1 s in 25 ms steps; a positive lag means
the cell fires before the animal reaches the position or heading. Per unit: Skaggs information of the
2.5 cm map and the vector length of the direction factor of the engine's position x direction (pxd)
factorisation. A unit enters a clock only if its lag-0 value beats its own circular-shift p99 and its
curve peaks inside the scanned range. pxd, not the raw HD curve: heading and place are correlated along
walls, which tilted raw vector-length curves towards -1 s (run 3712215), and 92 % of planted pure HD
units carried significant spatial information on the real track (run 3712236), so excluding spatial
units left no HD clock.

A misaligned arena clock moves the arena sessions' peak lags away from the maze sessions' in the same
recday. Whether a lag matters for grids is measured directly: every arena session is rescored by the
open-field engine itself (`spatial_tuning.session_stats`) with the spikes shifted by -100..+300 ms, against
the pooled thresholds and pools of the open-field run given by --of-tag. A synthetic check runs first on the first arena session's real track: planted place and HD
units with known lags must be recovered and untuned units must not enter either clock. Output:
<root>/data/spatial/qc/of_lag_scan_<dataset>.pkl (versioned, never overwritten). mEC and LEC only:
PFC has no arena sessions, so this file is not mirrored to `mFC_data/code/`.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import spatial_common as sc                                           # noqa: E402

LAGS = np.arange(-40, 41)                       # bins of 25 ms: -1..+1 s
N_SHIFT = 100
MIN_SHIFT_S = 20.0
TASK_TYPES = ('ABCD', 'ABCDE')


def _peak(curve: np.ndarray, lags: np.ndarray) -> float:
    """Lag (bins) of the curve maximum with parabolic interpolation; NaN at the range edge."""
    if not np.isfinite(curve).any():
        return np.nan
    i = int(np.nanargmax(curve))
    if i == 0 or i == len(curve) - 1:
        return np.nan
    a, b, c = curve[i - 1:i + 2]
    den = a - 2 * b + c
    return float(lags[i] + 0.5 * (a - c) / den) if np.isfinite(den) and den < 0 else float(lags[i])


class Scanner:
    """Binning of one session's track; info and HD vector length for any spike array at any lag."""

    def __init__(self, xy: np.ndarray, hd, edges: tuple, cfg: dict):
        import spatial_tuning as st
        self.st, self.cfg = st, cfg
        self.T = len(xy)
        speed = sc.speed_cm_s(xy)
        self.moving = np.isfinite(xy).all(axis=1) & (speed > cfg['min_speed'])
        ex, ey = edges
        self.nb = (len(ex) - 1, len(ey) - 1)
        self.flat = sc.bin_index(xy, ex, ey)
        h = cfg['hd']
        self.jflat = None
        if hd is not None:
            nd = int(round(360 / h['pxd_dir_deg']))
            npx = max(1, int(round((ex[-1] - ex[0]) / h['pxd_pos_cm'])))
            npy = max(1, int(round((ey[-1] - ey[0]) / h['pxd_pos_cm'])))
            pflat = sc.bin_index(xy, np.linspace(ex[0], ex[-1], npx + 1), np.linspace(ey[0], ey[-1], npy + 1))
            d = np.floor(np.mod(np.where(np.isfinite(hd), hd, 0.0), 2 * np.pi) / (2 * np.pi) * nd).astype(int)
            self.jflat = np.where(np.isfinite(hd) & (pflat >= 0), pflat * nd + np.clip(d, 0, nd - 1), -1)
            self.pxd_shape = (npx * npy, nd)
            self.centres = (np.arange(nd) + 0.5) * 2 * np.pi / nd

    def at_lag(self, counts: np.ndarray, k: int) -> tuple:
        """(info (N,), R (N,)) with behaviour at t + k bins."""
        T = self.T
        s_sl, b_sl = (slice(0, T - k), slice(k, T)) if k >= 0 else (slice(-k, T), slice(0, T + k))
        cnt = counts[:, s_sl]
        m = self.moving[b_sl] & (self.flat[b_sl] >= 0)
        M = sc.onehot(self.flat[b_sl], m, self.nb[0] * self.nb[1])
        occ = sc.occupancy_map(M, self.nb)
        rate = sc.rate_maps_separate(sc.spike_maps(cnt, M, self.nb), occ, self.cfg['sigma_bins'],
                                     min_smoothed_occ=self.cfg['min_smoothed_occ_s'])
        info = sc.skaggs_info(rate, occ)
        R = np.full(counts.shape[0], np.nan)
        if self.jflat is not None:
            jf = self.jflat[b_sl]
            Mj = sc.onehot(jf, self.moving[b_sl] & (jf >= 0), self.pxd_shape[0] * self.pxd_shape[1])
            nj = np.asarray((Mj.T @ cnt.T).T).reshape((counts.shape[0],) + self.pxd_shape)
            tj = sc.occupancy_map(Mj, self.pxd_shape)
            h = self.cfg['hd']
            d = self.st.pxd(nj, tj, h['pxd_tol'], h['pxd_maxit'])[1]
            R = self.st.vector_length(d, self.centres)[0]
        return info, R


def scan(counts: np.ndarray, sc_: Scanner, rng, lags=LAGS) -> dict:
    """Lag curves, lag-0 shift nulls, per-unit peak lags (ms) and clock membership for one session."""
    info = np.full((counts.shape[0], len(lags)), np.nan)
    R = np.full_like(info, np.nan)
    for j, k in enumerate(lags):
        info[:, j], R[:, j] = sc_.at_lag(counts, int(k))
    ms = int(round(MIN_SHIFT_S / sc.BIN_S))
    null_i, null_r = [], []
    for k in rng.integers(ms, sc_.T - ms, N_SHIFT):
        a, b = sc_.at_lag(np.roll(counts, int(k), axis=1), 0)
        null_i.append(a)
        null_r.append(b)
    p99_i = np.nanpercentile(np.array(null_i), 99, axis=0)
    p99_r = np.nanpercentile(np.array(null_r), 99, axis=0)
    i0 = int(np.nonzero(lags == 0)[0][0])
    sig_i = info[:, i0] > p99_i
    sig_r = R[:, i0] > p99_r
    pk_i = np.array([_peak(c, lags) for c in info]) * sc.BIN_S * 1000
    pk_r = np.array([_peak(c, lags) for c in R]) * sc.BIN_S * 1000
    return dict(info=info, R=R, p99_info=p99_i, p99_R=p99_r, sig_info=sig_i, sig_R=sig_r,
                peak_info_ms=pk_i, peak_hd_ms=pk_r, use_info=sig_i & np.isfinite(pk_i),
                use_hd=sig_r & np.isfinite(pk_r))


def grid_by_lag(view: dict, cfg: dict, walls: set, of_session: dict, lags_ms=(-100, 0, 100, 200, 300)) -> dict:
    """Engine statistics of one arena session with spikes shifted by each lag (np.roll by +k pairs spikes at
    t with behaviour at t + k), scored against the open-field run's lag-0 pooled thresholds."""
    import spatial_tuning as st
    ctx = st._context(view, cfg, walls)
    pb = np.array(of_session['pools']['broad'], int)
    thr = of_session['thresholds']['broad']
    null_g = np.asarray(of_session['null']['grid'])[:, pb] if len(pb) else np.zeros((1, 0))
    p95 = float(np.nanpercentile(null_g, 95)) if null_g.size else np.nan
    out = dict(lags_ms=list(lags_ms), broad=pb, thr_p99=float(thr['grid']), thr_p95=p95, by_lag={})
    for ms in lags_ms:
        k = int(round(ms / 1000 / sc.BIN_S))
        c_k = np.roll(ctx['counts'], k, axis=1)
        r = st.session_stats(ctx, c_k, st._smoothed_rate(ctx, c_k), False)
        g = r['grid'][pb]
        out['by_lag'][ms] = dict(grid=r['grid'], border=r['border'], si=r['si'], stab=r['stab'],
                                 max_grid=float(np.nanmax(g)) if np.isfinite(g).any() else np.nan,
                                 n_above_p99=int(np.nansum(g > thr['grid'])), n_above_p95=int(np.nansum(g > p95)))
    return out


def median_ci(v: np.ndarray, rng, n_boot: int = 2000) -> tuple:
    """(median, 2.5th, 97.5th percentile of the bootstrap median over units); NaNs for fewer than 3."""
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return (float(np.median(v)) if len(v) else np.nan, np.nan, np.nan)
    b = np.median(rng.choice(v, (n_boot, len(v))), axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def session_inputs(view: dict, cfg: dict) -> tuple:
    """(counts, xy, hd or None, edges, hd guard): arena sessions as `spatial_tuning._context`
    truncates them, in the arena square; task sessions over their own track extent."""
    T = min(view['counts'].shape[1], len(view['xy']))
    counts = np.asarray(view['counts'][:, :T], np.float32)
    xy = np.asarray(view['xy'][:T], float)
    hd = view['hd'][:T] if view['hd'] is not None else None
    if view.get('arena') is not None and view.get('session_type') not in TASK_TYPES:
        side = float(view['arena']['side_cm'])
        xy = np.where(np.isfinite(xy), np.clip(xy, 0, side - 1e-6), np.nan)
        ex = ey = sc.fixed_edges(0.0, side, cfg['bin_cm'])
    else:
        lo, hi = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
        ex = sc.fixed_edges(float(lo[0]), float(hi[0]) + 1e-6, cfg['bin_cm'])
        ey = sc.fixed_edges(float(lo[1]), float(hi[1]) + 1e-6, cfg['bin_cm'])
    guard = dict(ok=False)
    if hd is not None:
        h = cfg['hd']
        guard = sc.hd_guard(xy, hd, sc.speed_cm_s(xy), h['guard_speed'], h['guard_offset_deg'], h['guard_r'])
        if not guard['ok']:
            hd = None
    return counts, xy, hd, (ex, ey), guard


def synthetic(view: dict, cfg: dict, rng) -> list:
    """Planted lags on a real track: place units at -250/0/+250 ms, HD units at -100/0/+100 ms."""
    counts, xy, hd, edges, guard = session_inputs(view, cfg)
    T = len(xy)
    fin = np.isfinite(xy).all(axis=1)
    lag_bins = lambda ms: int(round(ms / 1000 / sc.BIN_S))             # noqa: E731

    def shifted(a, k):
        out = np.full(a.shape, np.nan)
        if k >= 0:
            out[:T - k] = a[k:]
        else:
            out[-k:] = a[:T + k]
        return out

    units, planted, kinds = [], [], []
    for ms in (-250, 0, 250):
        for _ in range(12):
            c = xy[fin][rng.integers(fin.sum())]
            d2 = ((shifted(xy, lag_bins(ms)) - c) ** 2).sum(1)
            lam = 0.5 + 15 * np.exp(-np.where(np.isfinite(d2), d2, np.inf) / (2 * 12.0 ** 2))
            units.append(rng.poisson(lam * sc.BIN_S))
            planted.append(ms)
            kinds.append('place')
    if hd is not None:
        for ms in (-100, 0, 100):
            for _ in range(12):
                mu = rng.uniform(0, 2 * np.pi)
                h = shifted(hd, lag_bins(ms))
                # weak tuning, as the real units: vector length ~0.3 at kappa 0.6
                lam = 1.0 + 8 * np.exp(0.6 * (np.cos(np.where(np.isfinite(h), h, mu + np.pi) - mu) - 1))
                units.append(rng.poisson(lam * sc.BIN_S))
                planted.append(ms)
                kinds.append('hd')
    for _ in range(24):
        units.append(rng.poisson(np.full(T, 5.0 * sc.BIN_S)))
        planted.append(np.nan)
        kinds.append('untuned')
    res = scan(np.array(units, np.float32), Scanner(xy, hd, edges, cfg), rng)
    planted, kinds = np.array(planted), np.array(kinds)
    checks = []
    # 25 ms: under half the smallest clock defect found so far (64 ms, LEC v1 neural origins)
    for ms in (-250, 0, 250):
        sel = (kinds == 'place') & (planted == ms) & res['use_info']
        med = np.median(res['peak_info_ms'][sel]) if sel.any() else np.nan
        checks.append((f'place group {ms:+d} ms: median recovered lag within 25 ms', abs(med - ms) <= 25,
                       f'{med:.0f} ms over {sel.sum()} of 12 units in the info clock'))
    if hd is not None:
        for ms in (-100, 0, 100):
            sel = (kinds == 'hd') & (planted == ms) & res['use_hd']
            med = np.median(res['peak_hd_ms'][sel]) if sel.any() else np.nan
            checks.append((f'HD group {ms:+d} ms: median recovered lag within 25 ms', abs(med - ms) <= 25,
                           f'{med:.0f} ms over {sel.sum()} of 12 units in the HD clock'))
        sel = (kinds == 'place') & (planted == 0)
        checks.append(('place units without HD tuning stay out of the pxd HD clock', np.mean(res['use_hd'][sel]) <= 0.2,
                       f'{np.mean(res["use_hd"][sel]):.2f} of {sel.sum()}'))
    sel = kinds == 'untuned'
    k = int((res['use_info'][sel] | res['use_hd'][sel]).sum())
    checks.append(('untuned units enter neither clock', k <= 2, f'{k} of {sel.sum()}'))
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', choices=('mec', 'lec'), required=True)
    ap.add_argument('--recday', nargs='*', default=None)
    ap.add_argument('--data-version', default='v4')
    ap.add_argument('--of-tag', default='8a49b0e2d93c', help='open-field run whose thresholds and pools score the shifted maps')
    args = ap.parse_args(argv)
    import spatial_io as sio
    sio.set_dataset(args.dataset)
    import spatial_tuning as st
    cfg = copy.deepcopy(st.DEFAULT_CFG)
    t0 = time.time()
    views = sio.load(args.dataset, args.recday or sio.default_recdays(args.dataset), args.data_version)
    arena_rds = sorted({rd for rd in views for v in views[rd].values()
                        if v.get('session_type') in sio.ARENA_TYPES and v['xy'] is not None})
    todo = [(rd, s) for rd in arena_rds for s, v in sorted(views[rd].items())
            if v['xy'] is not None and v.get('session_type') in sio.ARENA_TYPES + TASK_TYPES]
    print(f'{len(arena_rds)} recdays with arena sessions, {len(todo)} sessions to scan; loaded in {time.time() - t0:.0f} s', flush=True)
    rng = np.random.default_rng(20261009)
    first = next((rd, s) for rd, s in todo if views[rd][s]['session_type'] in sio.ARENA_TYPES)
    checks = synthetic(views[first[0]][first[1]], cfg, rng)
    n_fail = 0
    for name, ok, msg in checks:
        print(f"{'PASS' if ok else 'FAIL'} synthetic {name}: {msg}", flush=True)
        n_fail += not ok
    out = dict(dataset=args.dataset, data_version=args.data_version, lags_ms=LAGS * sc.BIN_S * 1000,
               n_shift=N_SHIFT, synthetic=checks, sessions={})
    print(f"\n{'recday':24s}{'s':>3s}{'type':>10s}{'N':>4s}{'HD':>4s}{'n info':>7s}{'info peak ms [95% CI]':>25s}"
          f"{'n HD':>6s}{'HD peak ms [95% CI]':>23s}", flush=True)
    fmt = lambda m: f'{m[0]:5.0f} [{m[1]:.0f}, {m[2]:.0f}]' if np.isfinite(m[0]) else '    -'    # noqa: E731
    grid_rows = []
    for rd, s in todo:
        v = views[rd][s]
        counts, xy, hd, edges, guard = session_inputs(v, cfg)
        res = scan(counts, Scanner(xy, hd, edges, cfg), rng)
        mi = median_ci(res['peak_info_ms'][res['use_info']], rng)
        mh = median_ci(res['peak_hd_ms'][res['use_hd']], rng)
        out['sessions'][(rd, s)] = dict(session_type=v['session_type'], guard=guard, median_info=mi,
                                        median_hd=mh, **res)
        if v['session_type'] in sio.ARENA_TYPES:
            of_path = os.path.join(sio.output_root(args.dataset), 'of', args.of_tag, 'shards', f'{rd}.pkl')
            if os.path.exists(of_path):
                import pickle
                of_sess = pickle.load(open(of_path, 'rb'))['sessions'][s]
                walls = sio.wall_exclusions(args.dataset, rd, s, cfg['border']['max_wall_rms_cm'])
                gl = grid_by_lag(v, cfg, walls, of_sess)
                out['sessions'][(rd, s)]['grid_by_lag'] = gl
                grid_rows.append((rd, s, len(gl['broad']), gl['thr_p99'], gl['thr_p95'],
                                  [(ms, x['max_grid'], x['n_above_p95'], x['n_above_p99']) for ms, x in gl['by_lag'].items()]))
        print(f"{rd:24s}{s:>3d}{v['session_type']:>10s}{counts.shape[0]:>4d}{'y' if hd is not None else 'n':>4s}"
              f"{int(res['use_info'].sum()):>7d}{fmt(mi):>25s}{int(res['use_hd'].sum()):>6d}{fmt(mh):>23s}", flush=True)
    print(f'\narena sessions rescored by the engine with shifted spikes (broad units; open-field run {args.of_tag}):')
    print(f"{'recday':24s}{'s':>3s}{'n':>4s}{'p95':>6s}{'p99':>6s}   per lag ms: max gridness / n > p95 / n > p99")
    for rd, s, n, t99, t95, per in grid_rows:
        print(f'{rd:24s}{s:>3d}{n:>4d}{t95:6.2f}{t99:6.2f}   ' + '   '.join(f'{ms:+d}: {mg:.2f}/{a}/{b}' for ms, mg, a, b in per), flush=True)
    path = sc.versioned(os.path.join(sio.output_root(args.dataset), 'qc', f'of_lag_scan_{args.dataset}.pkl'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc.write_new(path, out)
    print(f'\nwrote {path}; synthetic failures {n_fail}; {time.time() - t0:.0f} s', flush=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
