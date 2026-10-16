"""Synthetic controls for `rest_kernels` (M4 part 1): planted units on one real recday's rest
blocks and task sessions, through `run_rest_kernels.fit_recday`.

    python code/run_rest_kernels.py --dataset lec --gate [--quick]

Every planted unit shares a sleep process: each rest block falls into SWS at a random time 3-15
min in, after 2 min of quiet wake (rate x 0.7); in SWS the population alternates DOWN periods
(150-400 ms, rate x 0.05) and UP periods (0.3-1.5 s, rate x 1.3). Task sessions are awake. Base
rates are the recday's real task rates, except OU units at a fixed 5 Hz; the planted population
has max(real units, 120) units. The proxy is applied with the real recday's calibration: planted
Poisson task data have no lulls, so their own cut is 0 and the quiet wake read as SWS
(2026-10-15: proxy onset 5-130 s early in 10 of 11 blocks).

| row | planted | must hold |
|---|---|---|
| G1 | the sleep process at the recday's real unit count and rates, scored with its real calibration (`proxy_power`) | `down_frac` AUC >= 0.95 for SWS windows (LEC; PFC reported); recall and quiet wake called SWS reported |
| G2 | rest-entry ramps, tau 120-600 s, both signs | rest-entry locked >= 0.8 with the state terms; Stage 2 tau within 2x in >= 0.8 of the locked |
| G3 | SWS-gain units, rate x 0.3 in SWS | locked at chance with the state terms (exact binomial vs 5/101, p > 0.01); >= 0.5 without them |
| G4 | noise | locked at chance with the state terms (exact binomial vs 5/101, p > 0.01) |
| G4b | every unit's rest rate x (1 + 0.5 exp(-t / 300 s)), a separate Stage-1-only run | reported: noise locked with / without the state terms, what the population-rate term absorbs |
| G5 | day trends | the day term beats rest entry held out in >= 0.8 |
| G6 | OU log-rate at 5 Hz, 8 units per tau 0.25 / 2 / 30 / 120 s, in rest and task | ratio of the slow ACF tau, low-DOWN rest clear of the planted quiet wake / task, within 0.75-1.33 at 30 and 120 s; Murray tau within 2x at 0.25 s (2 s reported: past its 1 s lags). 2026-10-15: at real rates (LEC median 0.8 Hz) the 0.25 s fits failed in 2 of 4 |
| G7 | OU drift tau 180 s in both / tau 45 s in rest | rest / task drift growth 0.8-1.25 / >= 1.5, on low-DOWN segments clear of the planted quiet-wake dip (all segments reported) |
| G8 | a dropout in one block | `signal_loss_cut` within one 60 s window of it |

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import rest_kernels as rk                                              # noqa: E402

MIN_PLANTED = 120
QUIET_S = 120.0
OU_RATE_HZ = 5.0
SHARED_RAMP = (0.5, 300.0)                     # G4b: amplitude and tau of the ramp every unit shares


def _ou(T, tau_s, rng):
    a = np.exp(-ek.BIN_S / tau_s)
    e = rng.standard_normal(T) * np.sqrt(1 - a ** 2)
    from scipy.signal import lfilter
    x, _ = lfilter([1.0], [1, -a], e, zi=[a * rng.standard_normal()])
    return x


def sleep_process(T: int, rng, onset_s: float):
    """(rate factor (T,), SWS truth per 25 ms bin, quiet-wake truth per bin)."""
    f = np.ones(T)
    sws = np.zeros(T, bool)
    quiet = np.zeros(T, bool)
    on = int(onset_s / ek.BIN_S)
    q0 = max(0, on - int(QUIET_S / ek.BIN_S))
    f[q0:on] = 0.7
    quiet[q0:on] = True
    t = on
    while t < T:
        d = int(rng.uniform(0.15, 0.40) / ek.BIN_S)
        f[t:t + d] = 0.05
        t += d
        u = int(rng.uniform(0.3, 1.5) / ek.BIN_S)
        f[t:t + u] = 1.3
        t += u
    sws[on:] = True
    return f, sws, quiet


def specs(n_units: int, quick: bool) -> list:
    """(kind, params) per planted unit, padded with noise to n_units."""
    r = 1 if quick else 2
    U = []
    for tau in (120, 300, 600):
        for sign in ('down', 'up'):
            U += [('rest_ramp', dict(tau=tau, sign=sign))] * r
    U += [('sws_gain', dict())] * (4 * r)
    for per_hour in (-0.3, 0.3):
        U += [('day_trend', dict(per_hour=per_hour))] * (2 * r)
    for tau in (0.25, 2, 30, 120):
        U += [('ou', dict(tau=tau))] * 8             # on 2 units the 120 s ratio swung 0.68 and 1.47, on 4 it read 1.44
    U += [('drift_equal', dict())] * 20          # 20 units: M2's matched size, so each forms a region_full set
    U += [('drift_fast', dict())] * 20
    return U + [('noise', dict())] * max(0, n_units - len(U))


def _log_rate(kind, p, T, t_s, hours, rng, rest: bool):
    if kind == 'rest_ramp':
        return (np.log(3.0) * np.exp(-t_s / p['tau']) * (1 if p['sign'] == 'down' else -1)) if rest else np.zeros(T)
    if kind == 'day_trend':
        return p['per_hour'] * hours
    if kind == 'ou':
        return 0.5 * _ou(T, p['tau'], rng)
    if kind == 'drift_equal':
        return 0.8 * _ou(T, 180.0, rng)
    if kind == 'drift_fast':
        return 0.8 * _ou(T, 45.0 if rest else 180.0, rng)
    return np.zeros(T)


def make_population(blocks: list, task_sessions: list, rates_hz: np.ndarray, quick: bool = False, seed: int = 0,
                    dropout_block: int | None = None, shared_ramp: bool = False):
    """Planted copies of the rest blocks and task sessions, the truth table, and per block the
    planted SWS / quiet-wake truth and SWS onset. `shared_ramp` scales every unit's rest rate by
    1 + 0.5 exp(-t / 300 s) (G4b)."""
    rng = np.random.default_rng(seed)
    n = max(len(rates_hz), MIN_PLANTED)
    base = np.resize(np.asarray(rates_hz, float), n)
    base = np.clip(base, 0.05, None)
    U = specs(n, quick)[:n]
    base[[i for i, (k, _) in enumerate(U) if k == 'ou']] = OU_RATE_HZ
    truth = pd.DataFrame([dict(unit=i, kind=k, **p) for i, (k, p) in enumerate(U)])
    out_b, sleep = [], []
    for i, b in enumerate(blocks):
        T = b['counts'].shape[1]
        t_s = np.arange(T) * ek.BIN_S
        hours = b.get('hours_since_day_start', 0.0) + t_s / 3600.0
        onset = float(rng.uniform(180.0, min(900.0, max(240.0, T * ek.BIN_S - 120.0))))
        f, sws, quiet = sleep_process(T, rng, onset)
        lr = np.stack([_log_rate(k, p, T, t_s, hours, rng, True) for k, p in U])
        if shared_ramp:
            f = f * (1.0 + SHARED_RAMP[0] * np.exp(-t_s / SHARED_RAMP[1]))
        gain = np.ones((n, T)) * f[None, :]
        for u, (k, _) in enumerate(U):
            if k == 'sws_gain':
                gain[u, sws] *= 0.3
        lam = base[:, None] * ek.BIN_S * np.exp(np.clip(lr, -5, 5)) * gain
        c = rng.poisson(lam).astype(np.uint16)
        if dropout_block == i:
            c[:, int(0.7 * T):] = 0
        out_b.append({**b, 'counts': c})
        sleep.append(dict(onset_s=onset, sws=sws, quiet=quiet, dropout_bin=int(0.7 * T) if dropout_block == i else None))
    out_s = []
    for s in task_sessions:
        T = s['counts'].shape[1]
        t_s = np.arange(T) * ek.BIN_S
        hours = s.get('hours_since_day_start', 0.0) + t_s / 3600.0
        lr = np.stack([_log_rate(k, p, T, t_s, hours, rng, False) for k, p in U])
        out_s.append({**s, 'counts': rng.poisson(base[:, None] * ek.BIN_S * np.exp(np.clip(lr, -5, 5))).astype(np.uint16)})
    return out_b, out_s, truth, sleep


def window_truth(mask: np.ndarray) -> np.ndarray:
    """Per 10 s proxy window: True when the bin mask is True for more than half of it."""
    return rk.window_mean(mask.astype(float)) > 0.5


def proxy_power(rates_hz: np.ndarray, theta: float, cut: float, n_blocks: int = 4, block_s: float = 1200.0,
                seed: int = 0) -> dict:
    """The recday's own proxy (theta and cut from its real task data) on planted sleep at its real
    unit count and rates: AUC of `down_frac` for SWS windows against awake ones, SWS recall, and
    the fraction of quiet-wake windows (rate x 0.7) called SWS. The awake reference is the real task
    data, not Poisson trains, because real awake activity has far more low-count bins (LEC ah10:
    task 1st percentile 4 spikes per 25 ms; Poisson at the same rates: 14)."""
    rng = np.random.default_rng([seed, 11])
    base = np.clip(np.asarray(rates_hz, float), 0.05, None)[:, None] * ek.BIN_S
    T = int(block_s / ek.BIN_S)
    df, sw, tr, qu = [], [], [], []
    for _ in range(n_blocks):
        f, sws, q = sleep_process(T, rng, float(rng.uniform(180, 900)))
        P = rk.apply_proxy(rng.poisson(base * f[None, :]).astype(np.uint16), theta, cut)
        n = len(P['down_frac'])
        df.append(P['down_frac'])
        sw.append(P['sws'])
        tr.append(window_truth(sws)[:n])
        qu.append(window_truth(q)[:n])
    df, sw, tr, qu = (np.concatenate(x) for x in (df, sw, tr, qu))
    from scipy.stats import rankdata
    awake = ~tr
    r = rankdata(df)
    auc = float((r[tr].sum() - tr.sum() * (tr.sum() + 1) / 2) / (tr.sum() * awake.sum())) if tr.any() and awake.any() else np.nan
    return dict(n_units=len(base), auc=auc, quiet_called_sws=float(sw[qu].mean()) if qu.any() else np.nan,
                sws_recall=float(sw[tr].mean()) if tr.any() else np.nan, theta=theta, cut=cut)
