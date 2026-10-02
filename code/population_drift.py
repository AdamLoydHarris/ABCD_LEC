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
