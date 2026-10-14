"""Kanter et al. 2025 population measures on the event-kernel session format.

Separates the two population readouts of time that `event_kernels` keeps apart per neuron:
a locked clock (the same population state at the same time since entry in every session) and
drift (a session-unique slow wander). kNN on held-out sessions sees only the first; kNN
within a session (bins shuffled into folds, as Kanter did) sees both.

Measures, all in the ambient space of simultaneously recorded units after Kanter's soft
normalisation (count / (range + 5)):
  distance_vs_lag     mean cosine distance between 1-min epochs as a function of their lag
  knn_epochs          kNN (k = 10, cosine, distance-weighted) 1-min epoch decoding, within-session
                      5-fold vs held-out session; accuracy and mean absolute error in epochs
  event_acceleration  change in trajectory speed d(t, t+1) - d(t-1, t) around events, against
                      the same statistic at every other bin
  synchrony_onsets    Kanter's agnostic boundaries: population rate above its 90th percentile
                      and its increase above the 90th percentile
  log_fano            (M3 part 2) one unit's minute-scale variability: the Fano factor of its 30 s
                      smoothed rate over that of Poisson trains, log10; raw or against a prediction
  boundary_*, pv_correlation, trial_number_decoding
                      (M3 part 2) Kanter Fig 6: preferred transition around rewards, the PV
                      correlation across events of up- / down-modulated units vs other units,
                      decoding which event a 1 s population vector came from
  coding_axis, axis_offset, axis_stability, multiplexing
                      (M3 part 2) Kanter Fig 4G: PCA (50%) -> LDA coding dimensions and their angle
  fano_removal        (M3 part 2) Kanter Fig 5: drift growth without the top 25% of units by Fano

Every function takes counts at 25 ms (units, T) and works on one session; `subsample` draws
size-matched random unit sets (unit order is depth-ordered on the probes: never take the first N).

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import numpy as np

BIN_S = 0.025


def rebin(counts: np.ndarray, bin_s: float) -> np.ndarray:
    """(units, T) 25 ms counts -> (units, T // k) summed counts in bins of `bin_s`."""
    k = int(round(bin_s / BIN_S))
    n = counts.shape[1] // k
    return counts[:, :n * k].reshape(counts.shape[0], n, k).sum(axis=2).astype(float)


def soft_normalize(binned: np.ndarray) -> np.ndarray:
    """Kanter's soft normalisation per unit: counts / (range + 5)."""
    rng = binned.max(axis=1, keepdims=True) - binned.min(axis=1, keepdims=True)
    return binned / (rng + 5.0)


def cosine_distance(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Pairwise cosine distance between rows (time bins) of A and B; 0 identical, 1 orthogonal."""
    B = A if B is None else B
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = (A @ B.T) / (na * nb.T)
    return 1.0 - np.nan_to_num(S, nan=0.0)


def subsample(n_units: int, n: int, n_draws: int, seed: int = 0) -> list:
    """`n_draws` random unit sets of size `n` (all units once when n >= n_units)."""
    if n >= n_units:
        return [np.arange(n_units)]
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_units, n, replace=False)) for _ in range(n_draws)]


def epoch_labels(n_bins: int, bin_s: float, epoch_s: float) -> np.ndarray:
    return (np.arange(n_bins) * bin_s // epoch_s).astype(int)


def distance_vs_lag(counts: np.ndarray, bin_s: float = 10.0, epoch_s: float = 60.0,
                    max_lag: int = 15) -> np.ndarray:
    """(max_lag + 1,) mean cosine distance between bins of epochs i and i + lag (lag 0 = within
    epoch, distinct bins only); NaN where no epoch pair exists."""
    X = soft_normalize(rebin(counts, bin_s)).T
    lab = epoch_labels(len(X), bin_s, epoch_s)
    D = cosine_distance(X)
    out = np.full(max_lag + 1, np.nan)
    li, lj = np.meshgrid(lab, lab, indexing='ij')
    lagm = lj - li
    off = ~np.eye(len(X), dtype=bool)
    for L in range(max_lag + 1):
        m = (lagm == L) & off
        if m.any():
            out[L] = D[m].mean()
    return out


def _knn_predict(train_X, train_y, test_X, k=10):
    """Distance-weighted kNN on cosine distance (sklearn KNeighborsClassifier with
    metric='cosine', weights='distance', as Kanter)."""
    D = cosine_distance(test_X, train_X)
    idx = np.argsort(D, axis=1)[:, :k]
    d = np.take_along_axis(D, idx, axis=1)
    w = 1.0 / np.maximum(d, 1e-12)
    labs = train_y[idx]
    classes = np.unique(train_y)
    score = np.stack([(w * (labs == c)).sum(axis=1) for c in classes], axis=1)
    return classes[np.argmax(score, axis=1)]


def knn_epochs(sessions_counts: list, bin_s: float = 10.0, epoch_s: float = 60.0,
               max_epochs: int = 18, k: int = 10, n_folds: int = 5, seed: int = 0) -> dict:
    """Epoch decoding within each session (bins shuffled into `n_folds`) and across sessions
    (train on the others, test on the held-out one). Epochs beyond `max_epochs` are dropped so
    every session has the same classes. Returns accuracy and MAE (epochs), each with chance."""
    rng = np.random.default_rng(seed)
    data = []
    for c in sessions_counts:
        X = soft_normalize(rebin(c, bin_s)).T
        y = epoch_labels(len(X), bin_s, epoch_s)
        keep = y < max_epochs
        data.append((X[keep], y[keep]))
    within_acc, within_mae, across_acc, across_mae, chance = [], [], [], [], []
    for i, (X, y) in enumerate(data):
        order = rng.permutation(len(y))
        folds = np.array_split(order, n_folds)
        pred = np.empty_like(y)
        for f in folds:
            tr = np.setdiff1d(order, f)
            pred[f] = _knn_predict(X[tr], y[tr], X[f], k)
        within_acc.append(np.mean(pred == y))
        within_mae.append(np.mean(np.abs(pred - y)))
        others = [j for j in range(len(data)) if j != i]
        if others:
            Xtr = np.vstack([data[j][0] for j in others])
            ytr = np.concatenate([data[j][1] for j in others])
            p2 = _knn_predict(Xtr, ytr, X, k)
            across_acc.append(np.mean(p2 == y))
            across_mae.append(np.mean(np.abs(p2 - y)))
        n_cls = len(np.unique(y))
        chance.append(1.0 / n_cls)
    yy = np.concatenate([d[1] for d in data])
    perm = rng.permutation(yy)
    return dict(within_acc=np.array(within_acc), within_mae=np.array(within_mae),
                across_acc=np.array(across_acc), across_mae=np.array(across_mae),
                chance_acc=np.array(chance), chance_mae=float(np.mean(np.abs(perm - yy))))


def trajectory_speed(counts: np.ndarray, bin_s: float = 1.0) -> np.ndarray:
    """(n - 1,) cosine distance between consecutive population vectors."""
    X = soft_normalize(rebin(counts, bin_s)).T
    na = np.linalg.norm(X, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.sum(X[1:] * X[:-1], axis=1) / (na[1:] * na[:-1])
    return 1.0 - np.nan_to_num(s, nan=0.0)


def event_acceleration(counts: np.ndarray, event_bins_25ms: np.ndarray, bin_s: float = 1.0,
                       window: int = 5) -> dict:
    """Kanter's change in trajectory speed, a(t) = v(t) - v(t - 1) with v the distance between
    bins t and t + 1, averaged around events (lags -window..window bins) and z-scored against
    a(t) at all bins. Returns the event-triggered mean, its z per lag, and n events."""
    v = trajectory_speed(counts, bin_s)
    a = np.r_[np.nan, np.diff(v)]
    k = int(round(bin_s / BIN_S))
    ev = (np.asarray(event_bins_25ms) // k).astype(int)
    ev = ev[(ev - window >= 1) & (ev + window < len(a))]
    if not len(ev):
        return dict(mean=np.full(2 * window + 1, np.nan), z=np.full(2 * window + 1, np.nan), n=0)
    lags = np.arange(-window, window + 1)
    M = np.stack([a[ev + L] for L in lags], axis=1)
    mu, sd = np.nanmean(a), np.nanstd(a)
    mean = np.nanmean(M, axis=0)
    se = sd / np.sqrt(len(ev))
    return dict(mean=mean, z=(mean - mu) / se, n=int(len(ev)), lags=lags)


def synchrony_onsets(counts: np.ndarray, bin_s: float = 1.0, pct: float = 90.0) -> np.ndarray:
    """25 ms bin indices of Kanter's agnostic boundaries: population rate above its `pct`
    percentile and its rise from the previous bin above the `pct` percentile of rises."""
    pop = rebin(counts, bin_s).sum(axis=0)
    rise = np.r_[0.0, np.diff(pop)]
    on = np.flatnonzero((pop > np.percentile(pop, pct)) & (rise > np.percentile(rise, pct)))
    return on * int(round(bin_s / BIN_S))


# ---------------------------------------------------------------------------
# M3 part 2 (2026-10-14): Kanter part 2
# ---------------------------------------------------------------------------

def gauss_smooth(v: np.ndarray, sigma_bins: float) -> np.ndarray:
    """`scipy.ndimage.gaussian_filter1d(v, sigma_bins, axis=-1)` (truncate 4, mode 'reflect') by FFT
    convolution: equal to 2e-15 and 6x faster (56 vs 337 ms on 500 trains x 2413 bins, sigma 60)."""
    from scipy.signal import fftconvolve
    r = int(4.0 * sigma_bins + 0.5)
    x = np.arange(-r, r + 1)
    w = np.exp(-0.5 * (x / sigma_bins) ** 2)
    w /= w.sum()
    v = np.atleast_2d(v)
    if v.shape[-1] <= r:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(v, sigma_bins, axis=-1)
    pad = np.pad(v, [(0, 0)] * (v.ndim - 1) + [(r, r)], mode='symmetric')
    return fftconvolve(pad, w[None], mode='valid', axes=-1)


def log_fano(counts: np.ndarray, mu: np.ndarray | None = None, bin_s: float = 0.5, sigma_s: float = 30.0,
             n_poisson: int = 500, seed: int = 0) -> float:
    """Kanter's minute-scale variability for one unit: var / mean of its rate smoothed with a
    Gaussian of `sigma_s`, over the mean of the same for `n_poisson` Poisson trains of its mean rate
    and length, log10 (0 = Poisson). With `mu`, the expected count per bin (a held-out prediction),
    the rate is counts / mu and the reference trains are Poisson(mu) / mu, so behaviour and locked
    kernels the prediction carries are not counted as variability."""
    y = np.asarray(counts, dtype=float)
    if y.sum() <= 0:
        return np.nan
    if mu is None:
        lam, scale = np.full(len(y), y.mean()), np.full(len(y), 1.0 / bin_s)
    else:
        lam = np.maximum(np.asarray(mu, dtype=float), 1e-12)
        scale = 1.0 / lam

    def fano(v):
        sm = gauss_smooth(v, sigma_s / bin_s)
        return sm.var(axis=-1) / sm.mean(axis=-1)

    ref = fano(np.random.default_rng(seed).poisson(lam, size=(n_poisson, len(y))) * scale)
    return float(np.log10(fano((y * scale)[None])[0] / np.mean(ref)))


BOUNDARY_PRE_S, BOUNDARY_POST_S = 3, 4         # Kanter's figure-eight window, -3 to +4 s in 1 s bins


def reward_legs(sess: dict) -> list:
    """(arrival bin, reward type 0-3 for A-D, previous arrival bin) for every reward reached from a
    previous reward; the session's first A has no leg."""
    tt = sess['trial_times']
    out = []
    for i in range(len(tt)):
        for j in range(1, 5):
            out.append((int(tt[i, j]), j % 4, int(tt[i, j - 1])))
    return out


def boundary_windows(counts: np.ndarray, ev_bins, pre_s: int = BOUNDARY_PRE_S,
                     post_s: int = BOUNDARY_POST_S) -> np.ndarray:
    """(events, units, pre_s + post_s) spike counts in 1 s bins from -pre_s to +post_s s around
    each event (25 ms bins); events whose window leaves the session are dropped."""
    k = int(round(1.0 / BIN_S))
    ev = np.asarray(ev_bins, dtype=np.int64)
    ev = ev[(ev - pre_s * k >= 0) & (ev + post_s * k <= counts.shape[1])]
    nb = pre_s + post_s
    if not len(ev):
        return np.zeros((0, counts.shape[0], nb))
    W = counts[:, ev[:, None] + np.arange(-pre_s * k, post_s * k)[None, :]]
    return W.reshape(counts.shape[0], len(ev), nb, k).sum(axis=3).transpose(1, 0, 2).astype(float)


def window_speed(W: np.ndarray) -> np.ndarray:
    """(bins - 1,) mean over events of the cosine distance between consecutive soft-normalised
    population vectors of the window: its peak is Kanter's shift time."""
    if not len(W):
        return np.full(W.shape[2] - 1, np.nan)
    rng = W.max(axis=(0, 2)) - W.min(axis=(0, 2))
    X = W / (rng[None, :, None] + 5.0)
    a, b = X[:, :, :-1], X[:, :, 1:]
    with np.errstate(invalid='ignore', divide='ignore'):
        c = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return 1.0 - np.nanmean(np.nan_to_num(c, nan=0.0), axis=0)


def boundary_responses(W: np.ndarray) -> dict:
    """Kanter's boundary responses: per unit, the change of the event-averaged rate from each 1 s
    bin to the next (units, bins - 1), the transition with the largest absolute change (`pref`,
    -1 for silent units) and the fraction of active units preferring each transition."""
    d = np.diff(W.mean(axis=0), axis=1)
    active = W.sum(axis=(0, 2)) > 0
    pref = np.where(active, np.argmax(np.abs(d), axis=1), -1)
    frac = np.bincount(pref[active], minlength=d.shape[1]) / max(int(active.sum()), 1)
    return dict(delta=d, pref=pref, frac=frac)


def pv_correlation(W: np.ndarray, units: np.ndarray, shift: int) -> float:
    """Mean pairwise Pearson correlation across events of the per-event rate change into bin
    shift + 1, over `units`: high when the same ensemble responds the same way at every event."""
    if len(units) < 3 or len(W) < 2:
        return np.nan
    C = W[:, units, shift + 1] - W[:, units, shift]
    C = C[C.std(axis=1) > 0]
    if len(C) < 2:
        return np.nan
    R = np.corrcoef(C)
    return float(R[np.triu_indices(len(C), 1)].mean())


def boundary_ensembles(W: np.ndarray, n_other: int = 25, n_draws: int = 50, seed=0) -> list:
    """For every candidate shift transition: the fraction of units preferring it, and the PV
    correlation of units modulated up / down there against `n_draws` random sets of `n_other`
    units preferring other transitions (Kanter Fig 6E)."""
    br = boundary_responses(W)
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(W.shape[2] - 1):
        pos = np.flatnonzero((br['pref'] == s) & (br['delta'][:, s] > 0))
        neg = np.flatnonzero((br['pref'] == s) & (br['delta'][:, s] < 0))
        oth = np.flatnonzero((br['pref'] >= 0) & (br['pref'] != s))
        other = ([pv_correlation(W, np.sort(rng.choice(oth, n_other, replace=False)), s) for _ in range(n_draws)]
                 if len(oth) >= n_other else [])
        rows.append(dict(shift=s, frac_pref=float(br['frac'][s]), n_pos=len(pos), n_neg=len(neg),
                         pv_pos=pv_correlation(W, pos, s), pv_neg=pv_correlation(W, neg, s),
                         pv_other=float(np.nanmean(other)) if len(other) and np.isfinite(other).any() else np.nan))
    return rows


def trial_number_decoding(W: np.ndarray, k: int = 10, n_folds: int = 5, seed=0) -> dict:
    """Kanter's trial-identity decoding: kNN (cosine, distance-weighted) on soft-normalised 1 s
    population vectors of the windows, label = event number, folds of shuffled bins; the mean
    absolute error in events against that with shuffled labels."""
    n_ev, n_u, n_b = W.shape
    if n_ev < 3:
        return dict(error=np.nan, shuffle=np.nan, n_events=n_ev)
    X = soft_normalize(W.transpose(1, 0, 2).reshape(n_u, n_ev * n_b)).T
    y = np.repeat(np.arange(n_ev), n_b)
    rng = np.random.default_rng(seed)

    def cv(lab):
        order = rng.permutation(len(lab))
        pred = np.empty_like(lab)
        for f in np.array_split(order, n_folds):
            tr = np.setdiff1d(order, f)
            pred[f] = _knn_predict(X[tr], lab[tr], X[f], k)
        return pred

    p = cv(y)
    ys = rng.permutation(y)
    return dict(error=float(np.mean(np.abs(p - y))), shuffle=float(np.mean(np.abs(cv(ys) - ys))), n_events=n_ev)


def coding_axis(X: np.ndarray, y: np.ndarray, var: float = 0.5):
    """Kanter's coding dimension: PCA keeping `var` of the variance, LDA on the PC scores, and the
    loadings (units,) of the PC contributing most to LD1 (|LD1 weight| x PC sd); None if LDA
    cannot run (fewer than 2 classes)."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    if len(np.unique(y)) < 2 or len(X) < 3:
        return None
    pca = PCA().fit(X)
    n = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var) + 1)
    Z = pca.transform(X)[:, :n]
    lda = LinearDiscriminantAnalysis().fit(Z, y)
    w = np.abs(lda.scalings_[:, 0]) * Z.std(axis=0)
    return pca.components_[int(np.argmax(w))]


def axis_offset(L1, L2) -> float:
    """Kanter's angular offset, degrees: arccos |L1 . L2| of the unit-norm loadings."""
    if L1 is None or L2 is None:
        return np.nan
    c = abs(float(np.dot(L1, L2))) / (np.linalg.norm(L1) * np.linalg.norm(L2))
    return float(np.degrees(np.arccos(min(1.0, c))))


def axis_stability(X: np.ndarray, y: np.ndarray, groups: np.ndarray, max_leave: int = 30, seed=0):
    """(axis, median angle between it and the axis refitted without one group) over up to
    `max_leave` random groups (epochs or trials): the noise floor of an offset."""
    full = coding_axis(X, y)
    if full is None:
        return None, np.nan
    g = np.unique(groups)
    pick = g if len(g) <= max_leave else np.random.default_rng(seed).choice(g, max_leave, replace=False)
    ang = [axis_offset(full, coding_axis(X[groups != gi], y[groups != gi])) for gi in pick]
    ang = [a for a in ang if np.isfinite(a)]
    return full, (float(np.median(ang)) if ang else np.nan)


def multiplexing(L1, L2, pct: float = 75.0) -> float:
    """Fraction of units above the `pct` percentile of |loading| on both axes (chance 1/16 at 75)."""
    if L1 is None or L2 is None:
        return np.nan
    a, b = np.abs(L1), np.abs(L2)
    return float(np.mean((a > np.percentile(a, pct)) & (b > np.percentile(b, pct))))


def fano_removal(counts: np.ndarray, fano: np.ndarray, frac: float = 0.25, n_draws: int = 50, seed=0,
                 bin_s: float = 10.0, epoch_s: float = 60.0) -> dict:
    """Kanter Fig 5C: drift growth (distance between epochs 5-10 min apart minus 1 min apart) with
    all units, without the top `frac` by Fano, and without random sets of the same size."""
    n = counts.shape[0]
    m = int(round(frac * n))

    def growth(idx):
        c = distance_vs_lag(counts[idx], bin_s, epoch_s)
        with np.errstate(invalid='ignore'):
            return float(np.nanmean(c[5:11]) - c[1]) if np.isfinite(c[5:11]).any() else np.nan

    keep_top = np.sort(np.argsort(-np.nan_to_num(np.asarray(fano, float), nan=-np.inf))[m:])
    rng = np.random.default_rng(seed)
    rand = [growth(np.sort(rng.choice(n, n - m, replace=False))) for _ in range(n_draws)]
    return dict(full=growth(np.arange(n)), top_removed=growth(keep_top),
                random_removed=float(np.nanmean(rand)) if np.isfinite(rand).any() else np.nan, n_removed=m)
