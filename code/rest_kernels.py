"""Rest blocks for the event-kernel analysis (M4 part 1): a spike-based SWS proxy, signal-loss QC,
the rest-entry design, intrinsic timescales and drift segments.

SWS proxy (`sws_proxy`): Kanter's 25 ms rule, with its threshold calibrated on the recday's own
task data (awake), because without staged SWS segments a percentile of the rest block itself
would call quiet-wake lulls DOWN states. pop25 is the summed 25 ms population count; theta is its
1st percentile over the task sessions; DOWN periods are runs of >= 4 bins (100 ms) at or below
theta; `down_frac` is the time in DOWN periods per 10 s window; a window is SWS-like when its
`down_frac` exceeds the 99.9th percentile of the task windows'; SWS needs >= 20 s of SWS-like
windows (Kanter's minimum).

Rest-entry design (`build_rest_design`): the shape-free kernel of time since recording start in
the rest box (`EDGES_S['rest_entry']`) and continuous state terms as nuisance, in
`ek.build_design`'s format so that `ek.stage1` and `ek.stage2_unit` run on it unchanged. The
state terms are deciles of `down_frac`, and each unit's counts are divided by the other half's
relative population rate (`relative_pop_rate`; units split even / odd, so a unit never divides
by itself).
2026-10-15 (user decision): continuous only, no binary SWS flag. The LFP check could not validate
staging (theta/delta < 0.8 in 82-100% of every block's windows; rest delta never reached the
awake threshold), and the planted quiet wake (0.7x, every unit) locked noise to rest entry.
2026-10-16 (user decision): the population rate divides rather than entering as deciles. Deciles
cannot follow a multiplicative shared change: a ramp shared by every unit put 90-94% of the top
decile's windows in the first 5 min and still locked 79 of 92 noise units. The divisor is a
median, because the plain sum moved 13-18% at rest entry in the PFC gate from 6 ramping units
of 60. Rest windows are split into low-DOWN (`down_frac` within the awake range) and high-DOWN
(>= 0.1) for the timescales and drift.

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import population_drift as pdft                                        # noqa: E402

WIN_BINS = 400                                 # 10 s proxy windows
DOWN_MIN_BINS = 4                              # 100 ms
THETA_PCT, SWS_PCT, SWS_MIN_WIN = 1.0, 99.9, 2
HIGH_DOWN = 0.1                                # SWS-like: >= 10% of a window in DOWN states; the task cut alone was 0 on planted data
LOSS_WIN_BINS, LOSS_FRAC, LOSS_MIN_KEEP_S = 2400, 0.05, 300.0
MIN_REST_S = 120.0                             # shorter rests are OpenEphys misclicks (user, 2026-10-01)
SEG_S, SEG_EDGE_S = 60.0, 6.0                  # Kanter: 60 s from runs >= 72 s, 6 s trimmed at each end
TASK_SKIP_S = (120.0, 60.0)                    # Kanter: the first 2 min and last 1 min of a session
TS_WIN_S = 300.0                               # length-matched windows: the ACF fit's bias depends on length (M3)
MURRAY_BIN, MURRAY_LAGS = 2, 20                # 50 ms counts (2 x 25 ms), lags 50-1000 ms
MAX_SEGMENTS = 120                             # G7's rest / task ratio swung 0.70-1.36 on 10-25 segments
SEGS_PER_RUN = 20                              # every non-overlapping 60 s segment a run holds
POP_SPAN_WIN = 3                               # the divisor's counts span 30 s: low-rate units are too sparse per 10 s
POP_MIN_COUNT = 3.0                            # spikes per 10 s window (0.3 Hz) for a unit to enter the median
POP_FLOOR = 0.2                                # never divide by less: a near-silent window is a gap, not a state


# ---------------------------------------------------------------------------
# SWS proxy
# ---------------------------------------------------------------------------

def down_bins(pop25: np.ndarray, theta: float, min_bins: int = DOWN_MIN_BINS) -> np.ndarray:
    """(T,) True where a bin belongs to a run of >= min_bins bins with pop25 <= theta."""
    low = np.r_[False, pop25 <= theta, False]
    d = np.diff(low.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = np.zeros(len(pop25), bool)
    for a, b in zip(starts, ends):
        if b - a >= min_bins:
            out[a:b] = True
    return out


def window_mean(x: np.ndarray, win: int = WIN_BINS) -> np.ndarray:
    """(n_win,) mean of x per window; the trailing partial window is dropped."""
    n = len(x) // win
    return x[:n * win].reshape(n, win).mean(axis=1)


def runs(mask: np.ndarray, min_len: int = 1) -> list:
    """[(start, end)] of runs of True of length >= min_len."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(np.int8))
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def live_windows(pop25: np.ndarray, win: int = WIN_BINS) -> np.ndarray:
    """(n_win,) True where a window's population count is >= 5% of the median window: a dead window
    is a data gap, not a DOWN state (PFC me11 task shards end in one silent 10 s window, which
    alone pushed the SWS cut to 1.0)."""
    w = window_mean(pop25.astype(float), win)
    return w >= LOSS_FRAC * np.median(w) if len(w) else np.zeros(0, bool)


def calibrate(task_counts: list) -> dict:
    """theta (1st percentile of pop25) and the SWS cut (99.9th percentile of window down_frac) from
    the live windows of the recday's task sessions, which are awake."""
    pops, live = [], []
    for c in task_counts:
        p = c.sum(axis=0)
        lv = live_windows(p)
        lv[-3:] = False          # recordings end inside their last 1-2 windows: every high PFC task window but one session's sat there (me11, ah04)
        pops.append(p[:len(lv) * WIN_BINS].reshape(len(lv), WIN_BINS)[lv].ravel())
        live.append((p, lv))
    theta = float(np.percentile(np.concatenate(pops), THETA_PCT))
    task_df = np.concatenate([window_mean(down_bins(p, theta).astype(float))[lv] for p, lv in live])
    return dict(theta=theta, cut=float(np.percentile(task_df, SWS_PCT)), task_down_frac=task_df)


def apply_proxy(counts: np.ndarray, theta: float, cut: float) -> dict:
    """Per 10 s window of one block: down_frac and the SWS flag (SWS-like runs of >= 20 s)."""
    df = window_mean(down_bins(counts.sum(axis=0), theta).astype(float))
    flag = np.zeros(len(df), bool)
    for a, b in runs(df > cut, SWS_MIN_WIN):
        flag[a:b] = True
    return dict(down_frac=df, sws=flag)


def sws_proxy(rest_counts: list, task_counts: list | None = None, cal: dict | None = None) -> dict:
    """The proxy for every rest block of a recday, calibrated on its task sessions, or with `cal`
    given: the gate passes the real recday's, because planted Poisson task data have no lulls and
    calibrate the cut to 0."""
    cal = cal or calibrate(task_counts)
    return {**cal, 'blocks': [apply_proxy(c, cal['theta'], cal['cut']) for c in rest_counts]}


def state_masks(down_frac: np.ndarray, cut: float) -> tuple:
    """(low, high) per 10 s window: `down_frac` within the awake range (<= the task cut), and
    SWS-like (>= HIGH_DOWN and above the cut)."""
    return down_frac <= cut, (down_frac >= HIGH_DOWN) & (down_frac > cut)


def signal_loss_cut(counts: np.ndarray) -> int | None:
    """25 ms bin at which to truncate a block: the start of the first 60 s window whose population
    count falls below 5% of the block's median window; None when there is none."""
    w = window_mean(counts.sum(axis=0).astype(float), LOSS_WIN_BINS)
    if not len(w):
        return None
    bad = np.flatnonzero(w < LOSS_FRAC * np.median(w))
    return int(bad[0] * LOSS_WIN_BINS) if len(bad) else None


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

def rest_blocks(rec: dict) -> tuple:
    """(valid rest blocks, QC table). Valid = >= 2 min and not a misclick, truncated at signal
    loss and dropped if that leaves < 5 min; each carries the day clock of `ek.add_day_clock`."""
    out, qc = [], []
    for r in rec['rest']:
        T = r['counts'].shape[1]
        row = dict(index=r['index'], n_bins=T, duration_s=T * ek.BIN_S, misclick=bool(r['misclick']), cut_bin=None,
                   kept=False, reason='')
        if r['misclick'] or T * ek.BIN_S < MIN_REST_S:
            row['reason'] = 'shorter than 2 min'
        else:
            cut = signal_loss_cut(r['counts'])
            row['cut_bin'] = cut
            if cut is not None and cut * ek.BIN_S < LOSS_MIN_KEEP_S:
                row['reason'] = f'signal lost at {cut * ek.BIN_S:.0f} s'
            else:
                c = r['counts'] if cut is None else r['counts'][:, :cut]
                c = c[:, :max(0, c.shape[1] - 2 * WIN_BINS)]      # drop the final 20 s: the recording ends inside it
                out.append({**r, 'counts': c, 'cut_bin': cut})
                row['kept'] = True
                row['reason'] = '' if cut is None else f'truncated at {cut * ek.BIN_S:.0f} s'
        qc.append(row)
    fake = dict(recday=rec['recday'], dataset=rec['dataset'], sessions=out)
    if all(b.get('wall_bin0') is not None for b in out):
        ek.add_day_clock(fake)
        days = sorted({b['wall_bin0'].date() for b in out})
        for b in out:
            b['calendar_day'] = days.index(b['wall_bin0'].date())
    return out, pd.DataFrame(qc)


def halves(n_units: int) -> list:
    """Even and odd unit indices; unit order is probe depth, so each half spans the probe."""
    return [np.arange(n_units)[0::2], np.arange(n_units)[1::2]]


def relative_pop_rate(blocks: list, idx: np.ndarray) -> tuple:
    """(per block, the relative population rate per 10 s window; units used). Per window, the
    median over the units in `idx` with >= POP_MIN_COUNT spikes per window on average of each
    unit's mean count per window over the centred 30 s, divided by its own mean; scaled to mean 1
    over the recday's windows."""
    W = []
    for b in blocks:
        c = b['counts'][idx].astype(float)
        n = c.shape[1] // WIN_BINS
        W.append(c[:, :n * WIN_BINS].reshape(len(idx), n, WIN_BINS).sum(axis=2))
    m = np.concatenate(W, axis=1).mean(axis=1) if len(idx) else np.zeros(0)
    keep = m >= POP_MIN_COUNT
    k = np.ones(POP_SPAN_WIN)
    out = []
    for w in W:
        if not keep.any() or not w.shape[1]:
            out.append(np.ones(w.shape[1]))
            continue
        x = w[keep]
        span = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, x) / np.convolve(np.ones(x.shape[1]), k, 'same')
        out.append(np.median(span / m[keep, None], axis=0))
    mu = np.mean(np.concatenate(out)) if out else 1.0
    return [o / mu for o in out], int(keep.sum())


def block_features(blk: dict, prox: dict, pop_windows: list | None = None) -> dict:
    """250 ms rows of one rest block in `ek.session_blocks`' format: counts, the rest-entry lag
    one-hots and lag, start times, and per row its 10 s window's `down_frac` and each half's
    relative population rate (`pop_windows`, one per-window array per half)."""
    T = blk['counts'].shape[1]
    n = T // ek.BLOCK
    t = ek.time_since(np.array([0]), T)
    win = np.minimum(np.arange(n) * ek.BLOCK // WIN_BINS, len(prox['down_frac']) - 1)
    out = dict(counts=ek.block_mean(blk['counts'].T, 'sum').T.astype(np.float64),
               ev_rest_entry=ek.lag_onehot_blocks(t, ek.EDGES_S['rest_entry']),
               dt_rest_entry=ek.block_mean(t, 'mean'), t0_s=np.arange(n) * ek.BLOCK * ek.BIN_S,
               down_frac=prox['down_frac'][win])
    if pop_windows is not None:
        out['pop_half'] = np.stack([p[np.minimum(win, len(p) - 1)] for p in pop_windows])
    return out


def build_rest_design(feats: list, units: np.ndarray | None = None, pop: int | None = None,
                      state: bool = True) -> dict:
    """Stacked rest design in `ek.build_design`'s format: groups 'state' (deciles of `down_frac`)
    and 'ev_rest_entry'; `FR` holds `units` (all when None), divided by half `pop`'s relative
    population rate when given. `state=False` leaves the deciles and the division out, the gate's
    control."""
    sids = np.concatenate([np.full(f['counts'].shape[1], i) for i, f in enumerate(feats)])
    cols, groups, names = [], {}, []

    def add(name, M, labels):
        start = sum(c.shape[1] for c in cols)
        cols.append(M)
        groups[name] = list(range(start, start + M.shape[1]))
        names.extend(f'{name}:{lab}' for lab in labels)

    df = np.concatenate([f['down_frac'] for f in feats])
    if state and np.ptp(df) > 0:
        # 0 is its own bin: with ~36% of windows at 0, tied deciles merged 0-0.17 into one bin
        # holding 4-5% of planted SWS windows, and SWS-gain units locked to rest entry through it
        zero = df <= 0
        pos = ek.g3.compute_decile_edges(df[~zero], n_bins=10, outlier_pct=1, scheme='decile') if (~zero).sum() >= 20 else None
        oh = ek.g3.apply_onehot(df, pos) * (~zero)[:, None] if pos is not None else (~zero)[:, None].astype(float)
        M, kept, _ = ek._reference_coded(np.hstack([zero[:, None].astype(float), oh]), sids)
        add('state', M, ['down_zero' if k == 0 else f'down_d{k - 1}' for k in kept])
    M, kept, _ = ek._reference_coded(np.vstack([f['ev_rest_entry'] for f in feats]), sids)
    e = ek.EDGES_S['rest_entry']
    add('ev_rest_entry', M, [f'{e[j]}-{e[j + 1]}s' if j < len(e) - 1 else ('overflow' if j == len(e) - 1 else 'pre')
                             for j in kept])
    FR = np.hstack([f['counts'] for f in feats])
    units = np.arange(FR.shape[0]) if units is None else np.asarray(units)
    FR = FR[units]
    if state and pop is not None:
        FR = FR / np.maximum(np.concatenate([f['pop_half'][pop] for f in feats]), POP_FLOOR)[None, :]
    return dict(X=np.hstack(cols), FR=FR, units=units, session_ids=sids, groups=groups,
                names=names, blocks=feats, n_rows=len(sids))


def rest_reset_test(D: dict, blocks: list) -> dict:
    """M2's reset test on rest: uncentred, held out by block, the rest-entry kernel against a linear
    hours-since-day-start term (`ek._day_basis`). `entry_beats_day` per unit."""
    import glm_cv
    if any('hours_since_day_start' not in b for b in blocks):
        return dict(skipped='no wall clock for every rest block')
    day = np.vstack([ek._day_basis(b, f) for b, f in zip(blocks, D['blocks'])])
    day = day[:, [j for j in range(day.shape[1]) if np.ptp(day[:, j]) > 0]]
    X = np.column_stack([np.ones(D['n_rows']), D['X'], day])
    p = D['X'].shape[1] + 1
    groups = {'ev_rest_entry': [c + 1 for c in D['groups']['ev_rest_entry']], 'day': list(range(p, p + day.shape[1]))}
    res = glm_cv.cv_scores(X, D['FR'], D['session_ids'], groups, n_perm=0, center_within_sessions=False)
    res['entry_beats_day'] = res['rss_reduced']['day'] < res['rss_reduced']['ev_rest_entry']
    return res


# ---------------------------------------------------------------------------
# timescales and drift segments
# ---------------------------------------------------------------------------

def context_windows(blocks: list, proxy: dict, task_sessions: list, win_s: float = TS_WIN_S) -> dict:
    """Non-overlapping windows of `win_s` (25 ms counts, units x bins): within low-DOWN and within
    high-DOWN stretches of rest, and within task sessions after their first 2 min; 'meta' holds
    each rest window's (block, first bin)."""
    w = int(round(win_s / ek.BIN_S))
    per_win = int(round(win_s * 1000 / 25 / WIN_BINS))       # proxy windows per timescale window
    out = dict(rest_low=[], rest_high=[], task=[])
    meta = dict(rest_low=[], rest_high=[])
    for i, (b, p) in enumerate(zip(blocks, proxy['blocks'])):
        low, high = state_masks(p['down_frac'], proxy['cut'])
        for key, mask in (('rest_low', low), ('rest_high', high)):
            for a, e in runs(mask, per_win):
                for k in range((e - a) // per_win):
                    s0 = (a + k * per_win) * WIN_BINS
                    out[key].append(b['counts'][:, s0:s0 + w])
                    meta[key].append((i, s0))
    for s in task_sessions:
        c = s['counts']
        s0 = int(TASK_SKIP_S[0] / ek.BIN_S)
        for k in range((c.shape[1] - s0) // w):
            out['task'].append(c[:, s0 + k * w:s0 + (k + 1) * w])
    out['meta'] = meta
    return out


def murray_tau(windows: list, unit: int) -> dict:
    """Murray et al. 2014's intrinsic timescale for one unit: the autocorrelation of 50 ms counts at
    lags 50-1000 ms averaged over windows, then A (exp(-lag / tau) + B) from the first decreasing
    lag. tau in s; NaN when there is no decay."""
    acs = []
    for c in windows:
        x = c[unit].astype(float)
        x = x[:len(x) // MURRAY_BIN * MURRAY_BIN].reshape(-1, MURRAY_BIN).sum(axis=1)
        x = x - x.mean()
        v = float(np.dot(x, x))
        if v > 0:
            acs.append(np.array([np.dot(x[:-k], x[k:]) / v for k in range(1, MURRAY_LAGS + 1)]))
    out = dict(murray_tau_s=np.nan, murray_ok=False, n_windows=len(acs))
    if len(acs) < 2:
        return out
    ac = np.mean(acs, axis=0)
    lags = np.arange(1, MURRAY_LAGS + 1) * MURRAY_BIN * ek.BIN_S
    k0 = int(np.argmax(ac))
    if MURRAY_LAGS - k0 < 4:
        return out
    try:
        (A, tau, B), _ = curve_fit(lambda t, A, tau, B: A * (np.exp(-t / tau) + B), lags[k0:], ac[k0:],
                                   p0=(max(ac[k0], 1e-3), 0.2, 0.0), bounds=([0, 1e-3, 0], [np.inf, 50.0, 1]),
                                   maxfev=10000)        # B >= 0, as Murray's offset; free, it traded against tau (median 25 s for a 2 s OU)
    except (RuntimeError, ValueError):
        return out
    out.update(murray_tau_s=float(tau), murray_ok=bool(tau < 5.0), murray_ac=ac.astype(np.float32))
    return out


def drift_segments(blocks: list, proxy: dict, task_sessions: list, rng) -> dict:
    """Kanter's 60 s segments (25 ms counts): up to SEGS_PER_RUN non-overlapping ones per low-DOWN
    and per high-DOWN run of >= 72 s, from its trimmed interior; task segments from sessions
    without their first 2 and last 1 min, as many as the low-DOWN rest segments. Each context is
    capped at MAX_SEGMENTS."""
    L = int(round(SEG_S / ek.BIN_S))
    edge = int(round(SEG_EDGE_S / ek.BIN_S))
    need = int(np.ceil((SEG_S + 2 * SEG_EDGE_S) * 1000 / 25 / WIN_BINS))
    out = dict(rest_low=[], rest_high=[], task=[])
    meta = dict(rest_low=[], rest_high=[], task=[])
    for i, (b, p) in enumerate(zip(blocks, proxy['blocks'])):
        low, high = state_masks(p['down_frac'], proxy['cut'])
        for key, mask in (('rest_low', low), ('rest_high', high)):
            for a, e in runs(mask, need):
                span0, span1 = a * WIN_BINS + edge, e * WIN_BINS - edge
                k = min(SEGS_PER_RUN, (span1 - span0) // L)
                sub = (span1 - span0) // max(k, 1)
                for j in range(k):
                    lo = span0 + j * sub
                    s0 = int(rng.integers(lo, lo + sub - L + 1))
                    out[key].append(b['counts'][:, s0:s0 + L])
                    meta[key].append((i, s0))
    for key in ('rest_low', 'rest_high'):
        if len(out[key]) > MAX_SEGMENTS:
            keep = np.sort(rng.choice(len(out[key]), MAX_SEGMENTS, replace=False))
            out[key] = [out[key][i] for i in keep]
            meta[key] = [meta[key][i] for i in keep]
    n_task = max(len(out['rest_low']), 1)
    pool = []
    for s in task_sessions:
        T = s['counts'].shape[1]
        lo, hi = int(TASK_SKIP_S[0] / ek.BIN_S), T - int(TASK_SKIP_S[1] / ek.BIN_S) - L
        if hi > lo:
            pool.append((s, lo, hi))
    for k in range(min(n_task, MAX_SEGMENTS)):
        if not pool:
            break
        s, lo, hi = pool[k % len(pool)]
        s0 = int(rng.integers(lo, hi))
        out['task'].append(s['counts'][:, s0:s0 + L])
        meta['task'].append((k % len(pool), s0))
    out['meta'] = meta
    return out


def segment_drift(seg: np.ndarray, rng, k: int = 10, n_folds: int = 5) -> dict:
    """Kanter's drift measures on one 60 s segment at 1 s bins and 6 s epochs: the distance-vs-lag
    curve, its growth (lags 7-9 minus lag 1), and within-segment kNN epoch accuracy (chance 0.1)."""
    d = pdft.distance_vs_lag(seg, bin_s=1.0, epoch_s=SEG_EDGE_S, max_lag=9)
    X = pdft.soft_normalize(pdft.rebin(seg, 1.0)).T
    y = pdft.epoch_labels(len(X), 1.0, SEG_EDGE_S)
    order = rng.permutation(len(y))
    pred = np.empty_like(y)
    for f in np.array_split(order, n_folds):
        tr = np.setdiff1d(order, f)
        pred[f] = pdft._knn_predict(X[tr], y[tr], X[f], k)
    return dict(dist=d.astype(np.float32), growth=float(np.nanmean(d[7:10]) - d[1]), acc=float(np.mean(pred == y)),
                chance=1.0 / len(np.unique(y)))
