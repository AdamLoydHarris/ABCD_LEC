"""Tsao 2018 and Kanter 2025 single-unit time tables on the event-kernel session format (M3 part 2).

Tsao: per unit, an in-sample stepwise Poisson GLM on 500 ms bins of each session's core (first A
to the end of the last trial). Candidate terms: day_time, entry_time, leg_time, loop_time (linear,
one predictor each) and task, place (categorical, one term each). A term enters if it raises
adjusted R^2 by > 0.01 and leaves if it contributes < 0.005 (`stepwise_poisson`: the rules of
MATLAB's stepwiseglm, which Tsao used). Time-selective terms get f = a e^(bx) fits.

Kanter: per unit and session, on 0.5 s counts from bin 0, a Poisson GLM of elapsed time plus the
other units' mean rate (`kanter_glm`), the autocorrelation decay over 1-90 s (`acf_tau`) and the
log-normalised Fano factor of the 30 s-smoothed rate (`population_drift.log_fano`). The last two
also run on held-out residuals of the Stage 2 full model (`heldout_mu`), which also gives the
entry factors that the ramp test divides out.

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402

HALF = 20                                      # 25 ms bins per 500 ms bin (Tsao and Kanter)
HALF_S = HALF * ek.BIN_S
TSAO_ENTER, TSAO_REMOVE = 0.01, 0.005
LINEAR_TERMS = ('day_time', 'entry_time', 'leg_time', 'loop_time')
CATEGORICAL_TERMS = ('task', 'place')
MIN_PLACE_BINS = 20                            # rarer codes merge into unknown: a silent unit has no finite MLE there
ACF_MAX_LAG_S, ACF_FIT_S, ACF_TAIL_S = 90.0, (1.0, 90.0), (30.0, 90.0)
SLOW_RAMP_FLOORS_S = (60.0, 120.0, 240.0)      # ramp-test floors: at 60 s the slow ramp still absorbed 65% of a 15 s transient's first-minute advantage (LEC gate 3766748)
SURVIVOR_FRAC = 0.5                            # leg / loop averages stop where half the legs or loops have ended


# ---------------------------------------------------------------------------
# Poisson fits
# ---------------------------------------------------------------------------

def irls(y: np.ndarray, X: np.ndarray, iters: int = 100, tol: float = 1e-10):
    """Unpenalised Poisson MLE by Newton-IRLS: (beta, mu, Fisher information X'WX).
    Matches statsmodels' GLM to 1e-15 in adjusted R^2 on real units and is ~14x faster
    (39 vs 530 ms at 19542 x 23, ah10_20250616_20250617, 2026-10-14)."""
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(float(y.mean()), 1e-9))
    dev_old = np.inf
    for _ in range(iters):
        mu = np.exp(np.clip(X @ b, -30, 30))
        H = X.T @ (X * mu[:, None])
        b = b + np.linalg.lstsq(H, X.T @ (y - mu), rcond=None)[0]
        mu = np.exp(np.clip(X @ b, -30, 30))
        dev = ek._dev(y, mu)
        if abs(dev_old - dev) <= tol * (abs(dev) + 1.0):
            break
        dev_old = dev
    return b, mu, X.T @ (X * mu[:, None])


def adjusted_r2(y: np.ndarray, mu: np.ndarray, p: int) -> float:
    """MATLAB's `Rsquared.Adjusted` for a GLM: 1 - (n-1)/(n-p) SSE/SST on the response scale."""
    n = len(y)
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0 or n <= p:
        return np.nan
    return 1.0 - (n - 1) / (n - p) * float(np.sum((y - mu) ** 2)) / sst


def exp_fit(x, f) -> dict:
    """f = a e^(bx) by least squares from a log-linear start (Tsao, Kanter). tau = 1/|b| (s), with
    the sign of b. `ok` is False when the fit fails or tau exceeds 10x the x range (a line)."""
    x, f = np.asarray(x, float), np.asarray(f, float)
    m = np.isfinite(x) & np.isfinite(f)
    x, f = x[m], f[m]
    out = dict(a=np.nan, b=np.nan, tau_s=np.nan, sign=0, ok=False)
    pos = f > 0
    if len(x) < 5 or np.ptp(x) <= 0 or pos.sum() < 3:
        return out
    b0, la0 = np.polyfit(x[pos], np.log(f[pos]), 1)
    x0 = float(x.min())                        # fit a e^(b (x - x0)): x in seconds-to-hours is badly scaled
    try:
        (a1, b), _ = curve_fit(lambda xx, a, bb: a * np.exp(bb * (xx - x0)), x, f,
                               p0=(np.exp(la0 + b0 * x0), b0), maxfev=10000)
    except (RuntimeError, ValueError, FloatingPointError):
        return out
    if not (np.isfinite(a1) and np.isfinite(b)):
        return out
    tau = 1.0 / abs(b) if b != 0 else np.inf
    out.update(a=float(a1 * np.exp(-b * x0)), b=float(b), tau_s=float(tau), sign=int(np.sign(b)),
               ok=bool(np.isfinite(tau) and tau <= 10 * np.ptp(x)))
    return out


# ---------------------------------------------------------------------------
# Tsao: the stepwise table
# ---------------------------------------------------------------------------

def tsao_rows(sessions: list) -> dict:
    """500 ms rows over each session's core: counts (units, n) and the six candidate terms, in s.
    `day_time` is NaN for a session without a wall clock; the term is then dropped."""
    cols = {k: [] for k in (*LINEAR_TERMS, 'place', 'task', 'sid')}
    cnt = []
    for i, s in enumerate(sessions):
        k0, end, T = int(s['first_A_bin']), int(s['trial_times'].max()), s['counts'].shape[1]
        n = (end - k0) // HALF
        mid = k0 + HALF * np.arange(n) + HALF // 2          # centre 25 ms bin of each 500 ms bin
        cnt.append(s['counts'][:, k0:k0 + HALF * n].reshape(-1, n, HALF).sum(axis=2))
        t = mid * ek.BIN_S
        cols['entry_time'].append(t)
        h = s.get('hours_since_day_start')
        cols['day_time'].append(np.full(n, np.nan) if h is None or not np.isfinite(h) else h * 3600.0 + t)
        cols['leg_time'].append(ek.time_since(ek.event_bins(s, 'reward'), T)[mid])
        cols['loop_time'].append(ek.time_since(ek.event_bins(s, 'A'), T)[mid])
        occ = np.zeros((n, 22))
        np.add.at(occ, (np.repeat(np.arange(n), HALF), np.clip(s['locs'][k0:k0 + HALF * n].astype(int), 0, 21)), 1)
        cols['place'].append(occ.argmax(axis=1))            # modal code; ties go to the lower code
        cols['task'].append(np.full(n, str(s['task'])))
        cols['sid'].append(np.full(n, i))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    rare = np.bincount(out['place'], minlength=22) < MIN_PLACE_BINS
    out['place'] = np.where(rare[out['place']], 0, out['place'])
    out['counts'] = np.hstack(cnt).astype(float)
    return out


def term_blocks(rows: dict) -> dict:
    """Design columns per candidate term: z-scored linear terms, reference-coded categorical ones
    (the most occupied level is the reference). Terms without variation are left out."""
    out = {}
    for t in LINEAR_TERMS:
        v = rows[t]
        if np.isfinite(v).all() and np.std(v) > 0:
            out[t] = ((v - v.mean()) / v.std())[:, None]
    for t in CATEGORICAL_TERMS:
        lev, inv = np.unique(rows[t], return_inverse=True)
        if len(lev) < 2:
            continue
        ref = np.bincount(inv).argmax()
        out[t] = np.column_stack([(inv == j).astype(float) for j in range(len(lev)) if j != ref])
    return out


def stepwise_poisson(y: np.ndarray, blocks: dict, enter: float = TSAO_ENTER, remove: float = TSAO_REMOVE,
                     max_steps: int = 100) -> dict:
    """MATLAB stepwiseglm(Criterion 'adjrsquared', PEnter `enter`, PRemove `remove`, Upper 'linear')
    from the constant model. Add the term that raises adjusted R^2 most while it raises it by
    > enter; then drop the term whose removal costs least if it costs < remove, and go back to
    adding. A categorical term enters or leaves whole. Returns the final terms, adjusted R^2 and
    the path of (+/-, term, adjusted R^2)."""
    n = len(y)
    cache = {}

    def score(terms) -> float:
        key = frozenset(terms)
        if key not in cache:
            X = np.column_stack([np.ones(n)] + [blocks[t] for t in sorted(terms)])
            cache[key] = adjusted_r2(y, irls(y, X)[1], X.shape[1])
        return cache[key]

    model, path = [], []
    cur = score(model)
    for _ in range(max_steps):
        cand = [t for t in blocks if t not in model]
        gains = {t: score(model + [t]) - cur for t in cand}
        if gains and max(gains.values()) > enter:
            t = max(gains, key=gains.get)
            model.append(t)
            cur = score(model)
            path.append(('+', t, cur))
            continue
        if not model:
            break
        losses = {t: cur - score([m for m in model if m != t]) for t in model}
        t = min(losses, key=losses.get)
        if losses[t] < remove:
            model.remove(t)
            cur = score(model)
            path.append(('-', t, cur))
            continue
        break
    return dict(terms=tuple(sorted(model)), adj_r2=cur, path=path, n_fits=len(cache))


def survivor_mean(t: np.ndarray, y: np.ndarray, n_starts: int, bin_s: float = HALF_S):
    """Mean of y by bins of t across the legs or loops still running, up to the first bin that fewer
    than SURVIVOR_FRAC of the `n_starts` legs or loops reach. The shortest legs are ~2 s, so Tsao's
    fixed-length trial average would leave nothing. Returns (bin centres in s, mean)."""
    k = np.full(len(t), -1)
    fin = np.isfinite(t)
    k[fin] = np.floor(t[fin] / bin_s).astype(int)
    ok = k >= 0
    if n_starts == 0 or not ok.any():
        return np.zeros(0), np.zeros(0)
    cnt = np.bincount(k[ok])
    s = np.bincount(k[ok], weights=y[ok])
    low = np.flatnonzero(cnt < SURVIVOR_FRAC * n_starts)
    stop = int(low[0]) if len(low) else len(cnt)
    return (np.arange(stop) + 0.5) * bin_s, s[:stop] / cnt[:stop]


def tsao_exp_inputs(rows: dict, y: np.ndarray, term: str):
    """(x in s, rate in Hz) for the exponential fit of one time-selective term, as Tsao: session-time
    cells on the rate over all core bins, trial-time cells on the rate averaged across trials
    (entry: across sessions; leg and loop: across legs or loops still running)."""
    rate = y / HALF_S
    if term == 'day_time':
        return rows['day_time'], rate
    if term == 'entry_time':
        k = np.floor(rows['entry_time'] / HALF_S).astype(int)
        cnt = np.bincount(k)
        s = np.bincount(k, weights=rate)
        keep = np.flatnonzero(cnt >= 2)                     # bins present in >= 2 sessions
        return keep * HALF_S, s[keep] / cnt[keep]
    t = rows[term]
    n_starts = int(np.sum(np.r_[True, np.diff(t) < 0] & np.isfinite(t)))   # a new leg or loop where the clock resets
    return survivor_mean(t, rate, n_starts)


def tsao_unit(rows: dict, blocks: dict, u: int) -> dict:
    """Stepwise fit for one unit, the selected terms as flags, and the exponential fit of each
    selected time term."""
    y = rows['counts'][u]
    out = dict(unit=u, rate_hz=float(y.mean() / HALF_S))
    st = stepwise_poisson(y, blocks) if y.sum() > 0 else dict(terms=(), adj_r2=np.nan, path=[], n_fits=0)
    out.update(st)
    for t in (*LINEAR_TERMS, *CATEGORICAL_TERMS):
        out[f'has_{t}'] = t in st['terms']
    for t in LINEAR_TERMS:
        if t in st['terms']:
            x, f = tsao_exp_inputs(rows, y, t)
            e = exp_fit(x, f)
            out.update({f'{t}_{k}': v for k, v in e.items()})
    return out


# ---------------------------------------------------------------------------
# Kanter: GLM time constants and autocorrelation
# ---------------------------------------------------------------------------

def kanter_glm(y: np.ndarray, t: np.ndarray, pop: np.ndarray) -> dict:
    """Kanter's first method on one session: Poisson GLM of 0.5 s counts on elapsed time and the
    population rate; the Wald p of elapsed time (as glmfit, dispersion 1); if p < 0.05, the
    exponential fit of the rate on time."""
    out = dict(glm_slope=np.nan, glm_p=np.nan, glm_tau_s=np.nan, glm_sign=0, glm_exp_ok=False)
    if y.sum() < 5 or np.std(pop) <= 0:
        return out
    X = np.column_stack([np.ones(len(y)), (t - t.mean()) / t.std(), (pop - pop.mean()) / pop.std()])
    b, mu, H = irls(y, X)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        return out
    p = float(2 * norm.sf(abs(b[1] / se[1])))
    out.update(glm_slope=float(b[1] / t.std()), glm_p=p)
    if p < 0.05:
        e = exp_fit(t, y / HALF_S)
        out.update(glm_tau_s=e['tau_s'], glm_sign=e['sign'], glm_exp_ok=e['ok'])
    return out


def acf_curve(x: np.ndarray, max_lag: int) -> np.ndarray:
    """(max_lag,) autocorrelation at lags 1..max_lag bins, divided by the lag-0 variance."""
    x = np.asarray(x, float) - np.mean(x)
    v = float(np.dot(x, x))
    if v <= 0 or len(x) <= max_lag:
        return np.full(max_lag, np.nan)
    f = np.fft.rfft(x, 2 * len(x))
    r = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return r[1:] / v


def acf_tau(x: np.ndarray, bin_s: float = HALF_S) -> dict:
    """Kanter's second method: positive-lag autocorrelation to 90 s, divided by its maximum, then
    the exponential fit over 1-90 s; tau in s when it decays. `tail` is the mean lag-0-normalised
    autocorrelation over 30-90 s, which K4 compares between raw counts and residuals."""
    L = int(round(ACF_MAX_LAG_S / bin_s))
    c = acf_curve(x, L)
    lags = np.arange(1, L + 1) * bin_s
    out = dict(acf=c.astype(np.float32), acf_tau_s=np.nan, acf_ok=False,
               acf_tail=float(np.nanmean(c[(lags >= ACF_TAIL_S[0]) & (lags <= ACF_TAIL_S[1])])))
    if not np.isfinite(c).all() or np.max(c) <= 0:
        return out
    m = (lags >= ACF_FIT_S[0]) & (lags <= ACF_FIT_S[1])
    e = exp_fit(lags[m], c[m] / np.max(c))
    if e['sign'] < 0:
        out.update(acf_tau_s=e['tau_s'], acf_ok=e['ok'])
    return out


# ---------------------------------------------------------------------------
# held-out Stage 2 full model: residuals and the ramp test's entry factors
# ---------------------------------------------------------------------------

def heldout_mu(D: dict, unit: int, glm_alpha: float = 1e-4, slow_ramp: bool = False,
               floors=SLOW_RAMP_FLOORS_S) -> dict:
    """Per row, the Stage 2 full model's held-out prediction for one unit (the training-fold glum
    fit of `ek.stage2_unit`, the held-out session's intercept refit as `ek._heldout_dev`), its
    summed held-out deviance (= Stage 2's `dev_np`), and the held-out entry factors, each scaled to
    mean 1 per session: the shape-free kernel exp(X_entry b_entry) and, with `slow_ramp`, for each
    floor the ramp exp(-t/tau) with tau >= floor profiled on the training sessions as Stage 2 does."""
    from glum import GeneralizedLinearRegressor
    y_all = D['FR'][unit]
    sids = D['session_ids']
    X = D['X']
    S = (sids[:, None] == np.unique(sids)[None, :]).astype(float)
    ecols = D['groups']['ev_entry']
    n = len(y_all)
    mu, f_entry = np.empty(n), np.empty(n)
    f_ramp = {fl: np.full(n, np.nan) for fl in floors} if slow_ramp else {}
    tau_pick = {fl: [] for fl in floors}
    dev = 0.0
    if slow_ramp:
        dt = np.concatenate([b['dt_entry'] for b in D['blocks']])
        taus = [float(t) for t in ek.shape_grid('entry')[0] if t >= min(floors)]
        R = {t: ek.ramp(dt, t) for t in taus}
    for f in np.unique(sids):
        tr, te = sids != f, sids == f
        cols_s = np.unique(sids[tr])
        m = GeneralizedLinearRegressor(family='poisson', alpha=glm_alpha, fit_intercept=False, max_iter=200)
        m.fit(np.hstack([X[tr], S[tr][:, cols_s]]), y_all[tr])
        bx, bs = m.coef_[:X.shape[1]], m.coef_[X.shape[1]:]
        eta = X[te] @ bx
        c = np.log(max(y_all[te].sum(), 1e-9) / max(np.exp(np.clip(eta, -30, 30)).sum(), 1e-12))
        mu[te] = np.exp(np.clip(eta + c, -30, 30))
        dev += ek._dev(y_all[te], mu[te])
        e = np.exp(np.clip(X[te][:, ecols] @ bx[ecols], -30, 30))
        f_entry[te] = e / e.mean()
        if slow_ramp:
            off_tr = (X[tr] @ bx - X[tr][:, ecols] @ bx[ecols]) + S[tr][:, cols_s] @ bs
            fits = []
            for t in taus:
                b, d = ek._poisson_newton(y_all[tr], np.column_stack([np.ones(tr.sum()), R[t][tr]]), off_tr)
                fits.append((d, t, b))
            for fl in floors:
                _, t_best, b_best = min((z for z in fits if z[1] >= fl), key=lambda z: z[0])
                r = np.exp(np.clip(b_best[1] * R[t_best][te], -30, 30))
                f_ramp[fl][te] = r / r.mean()
                tau_pick[fl].append(t_best)
    return dict(mu=mu, f_entry=f_entry, f_ramp=f_ramp, dev=dev,
                slow_ramp_tau_median={fl: float(np.median(v)) if v else np.nan for fl, v in tau_pick.items()})
