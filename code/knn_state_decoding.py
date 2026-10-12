"""kNN ABCD state decoding, trial by trial, with time in session controlled two ways.

    python run_knn_state_decoding.py --dataset lec|pfc     (under sbatch; loads the data_dic)

Samples come from ccgp_state_pairs.build_task_state_matrices: one session per unique task, the
mean over leg bins 15-75 of Neurons_norm (drops the reward tone and reward-end place fields).
Then trial 0 (search) and trials with a non-finite leg are dropped and each task is z-scored.
There is one sample per (trial, state), so any function of trial time is state-balanced.

Within a task, W2 asks whether nearby trials decode each other better than distant ones
(near / far / rand pools of equal size). Across tasks, X0 and X1 ask whether matching or removing
time rescues an abstract state code. Feature variants:
  raw
  detrend  a cubic spline in TRIAL time, per feature, label-free
  centred  minus the trial's own 4-state mean
kNN weights every dimension equally, so a small shared subspace can be swamped by remapped ones;
a cross-task null here adds little to CCGP (CCGP_STATE_PAIRS.md).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import ccgp_state_pairs as ccgp

BIN_S = 0.025
N_STATES = 4
VARIANTS = ('raw', 'detrend', 'centred', 'project')


@dataclass
class KNNConfig:
    """Defaults argued for in KNN_STATE_DECODING.md."""
    trim_bins: int = 15             # both leg ends, as CCGP; 0 only for the tone synthetic
    drop_first_trial: bool = True   # trial 0 is search, legs ~2x longer
    n_pcs: int = 15                 # matches the LDA's 15-PC control
    k_all: int = 5                  # W0, X0, X1, X4
    k_small: int = 3                # W2 pools hold 2 samples per class; k must stay below that x2
    min_trials_w2: int = 13         # usable trials for near / far / rand
    far_gap: int = 5                # within-task far pool: |trial lag| >= 5
    near_s: float = 90.0            # cross-task near: |dT| <= 90 s, below the ~150 s time MAE
    far_s: float = 300.0
    per_task_pool: int = 2          # trials per training task in the X1 pools
    n_pool_draws: int = 50
    n_null: int = 1000
    n_null_within: int = 200        # the within-task label null is a sanity check only
    spline_knots: int = 3
    min_train_tasks: int = 3
    seed: int = 0


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def _trial_finite(norm, ccfg):
    """(n_trials,) True where every neuron has a finite mean in all four trimmed legs."""
    ok = np.ones(norm.shape[1], bool)
    for s in range(N_STATES):
        lo, hi = ccgp._state_slice(s, ccfg)
        with np.errstate(invalid='ignore'), _quiet():
            ok &= np.isfinite(np.nanmean(norm[:, :, lo:hi], axis=2)).all(axis=0)
    return ok


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def _zscore(X):
    sd = X.std(axis=0)
    return np.where(sd > 0, (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1), 0.0)


def _session_samples(sd, neuron_idx, ccfg, cfg):
    """One session -> (X z-scored, y, trial, t_sec, T, leg_s) after the trial filters."""
    norm = np.asarray(sd['Neurons_norm'], float)[neuron_idx]
    tt = np.asarray(sd['Trial_times'], float)
    n_tr = norm.shape[1]
    if tt.shape[0] < n_tr:
        raise AssertionError(f'Trial_times has {tt.shape[0]} rows for {n_tr} trials')
    # rows that restart (the legacy synthetic) would make every trial look simultaneous
    if n_tr > 1 and np.any(np.diff(tt[:n_tr, 0]) <= 0):
        raise AssertionError('Trial_times rows do not increase across trials')
    X, y, trial = ccgp._norm_to_samples(norm, ccfg)
    valid = _trial_finite(norm, ccfg)
    if cfg.drop_first_trial:
        valid[0] = False
    m = valid[trial]
    X, y, trial = X[m], y[m], trial[m]
    t0 = tt[0, 0]
    a, b = tt[trial, y], tt[trial, y + 1]
    return {'X': _zscore(X), 'y': y, 'trial': trial,
            't_sec': ((a + b) / 2 - t0) * BIN_S,
            'T': ((tt[trial, 0] + tt[trial, N_STATES]) / 2 - t0) * BIN_S,
            'leg_s': (b - a) * BIN_S}


def prepare_recday(data_dic, mouse_recday, cfg=None, neuron_subset=None):
    """Tasks of one recday as dicts of per-sample arrays; None if unusable.

    Each task: session, task, first_exposure, X, y, trial, t_sec, T, leg_s, dest, and `repeat`
    (the same samples from another session of the same layout, or None).
    """
    cfg = cfg or KNNConfig()
    ccfg = replace(ccgp.CCGPConfig(), zscore='none', trim_start_bins=cfg.trim_bins,
                   trim_end_bins=cfg.trim_bins)
    tasks = ccgp.build_task_state_matrices(data_dic, mouse_recday, ccfg, neuron_subset)
    if not tasks:
        return None
    rd = data_dic[mouse_recday]
    keys = sorted(k for k in rd if isinstance(rd[k], dict) and 'Task' in rd[k])
    out = []
    for t in tasks:
        smp = _session_samples(rd[t['session']], t['neuron_idx'], ccfg, cfg)
        if len(np.unique(smp['trial'])) < 3:
            continue
        same = [k for k in keys if np.array_equal(np.asarray(rd[k]['Task']), t['task'])]
        rep = None
        for k in same:
            if k == t['session'] or rd[k].get('Neurons_norm') is None:
                continue
            try:
                r = _session_samples(rd[k], t['neuron_idx'], ccfg, cfg)
            except AssertionError:
                continue
            if len(np.unique(r['trial'])) >= 3:
                rep = {'session': k, **r}
                break
        out.append({'session': t['session'], 'task': t['task'],
                    'first_exposure': same[0] == t['session'],
                    'dest': t['task'][(smp['y'] + 1) % N_STATES], 'repeat': rep,
                    'neuron_idx': t['neuron_idx'], **smp})
    return out if len(out) >= 2 else None


# ---------------------------------------------------------------------------
# Feature variants and PCA
# ---------------------------------------------------------------------------

def _check_trial_level(T, trial):
    """The detrend regressor must be constant within a trial, or it absorbs state order."""
    for k in np.unique(trial):
        v = T[trial == k]
        if np.ptp(v) > 0:
            raise ValueError('detrend regressor varies within a trial; use trial-level time')


def detrend(X, T, trial, fit, knots=3):
    """Remove a cubic spline in trial time from every column; fit on rows in `fit` only."""
    from sklearn.preprocessing import SplineTransformer
    _check_trial_level(T, trial)
    if len(np.unique(T[fit])) < 4:
        return X - X[fit].mean(axis=0)
    st = SplineTransformer(n_knots=knots, degree=3, extrapolation='linear')
    B = st.fit_transform(T[fit, None])
    coef, *_ = np.linalg.lstsq(B, X[fit], rcond=None)
    return X - st.transform(T[:, None]) @ coef


def project_drift(X, T, trial, fit, knots=3):
    """Detrend, then remove the drift subspace itself from every sample.

    Detrending and centring leave the within-trial part of a slow drift, which runs A < B < C < D
    in every trial and task; on a pure-drift synthetic that alone decoded 0.34 across tasks
    (chance 0.25). The subspace is the span of the fitted trial-time trend (label-free).
    """
    from sklearn.preprocessing import SplineTransformer
    _check_trial_level(T, trial)
    if len(np.unique(T[fit])) < 4:
        return X - X[fit].mean(axis=0)
    st = SplineTransformer(n_knots=knots, degree=3, extrapolation='linear')
    B = st.fit_transform(T[fit, None])
    coef, *_ = np.linalg.lstsq(B, X[fit], rcond=None)
    trend = B @ coef
    _, sv, vt = np.linalg.svd(trend - trend.mean(axis=0), full_matrices=False)
    V = vt[:int((sv > 1e-8 * sv.max()).sum())].T if sv.max() > 0 else vt[:0].T
    Xd = X - st.transform(T[:, None]) @ coef
    return Xd - Xd @ V @ V.T


def centre_trials(X, trial, group=None):
    """Subtract each trial's own mean over its 4 states (label-free, state-balanced)."""
    key = trial if group is None else group * 100000 + trial
    out = X.copy()
    for k in np.unique(key):
        m = key == k
        out[m] -= X[m].mean(axis=0)
    return out


def pca_fit(X, fit, n_pcs):
    """PCA on rows in `fit`; returns a transform function (shared by every pool of a fold)."""
    mu = X[fit].mean(axis=0)
    n = int(min(n_pcs, fit.sum() - 1, X.shape[1]))
    _, _, vt = np.linalg.svd(X[fit] - mu, full_matrices=False)
    W = vt[:n].T
    return lambda Z: (Z - mu) @ W


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------

def vote(labels_nb, n_classes=N_STATES):
    """Majority vote over neighbour labels (nearest first) -> (pred, class scores).

    Ties go to the tied class whose member is nearest: permutation-equivariant, so label nulls
    are exact. sklearn breaks ties to class 0 (A), which biases the shortest-leg diagnostic.
    """
    k = labels_nb.shape[-1]
    eq = labels_nb[..., None] == np.arange(n_classes)
    counts = eq.sum(-2)
    first = np.where(eq.any(-2), eq.argmax(-2), k)
    score = counts * (k + 1) - first
    return score.argmax(-1), score


def rank_score(score, y):
    """(4 - rank of the true class) / 3, ties averaged; chance 0.5 under the vote rule."""
    s_true = np.take_along_axis(score, y[..., None], -1)
    gt = (score > s_true).sum(-1)
    eqn = (score == s_true).sum(-1) - 1
    return (3 - gt - 0.5 * eqn) / 3


def _nearest(Dsub, k):
    return np.argsort(Dsub, axis=1, kind='stable')[:, :k]


def _score_pool(D, cols, ytr, yte, k):
    """Accuracy and rank score of the test samples against training columns `cols`."""
    idx = cols[_nearest(D[:, cols], k)]
    pred, sc = vote(ytr[idx])
    return float(np.mean(pred == yte)), float(np.mean(rank_score(sc, yte))), idx, pred


def _locality(idx, t_tr, t_te, cols):
    """1 - mean |dt| of the neighbours / mean |dt| of the whole pool (0 = no time structure)."""
    nb = np.abs(t_tr[idx] - t_te[:, None]).mean()
    pool = np.abs(t_tr[cols][None, :] - t_te[:, None]).mean()
    return float(1 - nb / pool) if pool > 0 else np.nan


# ---------------------------------------------------------------------------
# Within task
# ---------------------------------------------------------------------------

def _variant(X, variant, T, trial, fit, cfg, group=None):
    if variant == 'raw':
        return X
    if variant == 'detrend':
        return detrend(X, T, trial, fit, cfg.spline_knots)
    if variant == 'project':
        return project_drift(X, T, trial, fit, cfg.spline_knots)
    return centre_trials(X, trial, group)


def within_task(task, cfg, rng, ti, variants=VARIANTS):
    """W0, W2 (near / far / far duration-matched / rand), lag curve and the W0 label null."""
    X, y, trial, T, leg = task['X'], task['y'], task['trial'], task['T'], task['leg_s']
    trials = np.unique(trial)
    rows, lag_rows = [], []
    null_correct = {v: np.zeros(cfg.n_null_within) for v in variants}
    n_tested = 0
    for kt in trials:
        te = trial == kt
        fit = ~te
        for v in variants:
            Xv = _variant(X, v, T, trial, fit, cfg)
            Z = pca_fit(Xv, fit, cfg.n_pcs)(Xv)
            D = cdist(Z[te], Z)
            yte = y[te]
            base = dict(task_idx=ti, trial=int(kt), T=float(T[te][0]), variant=v)
            cols = np.flatnonzero(fit)
            acc, rk, idx, pred = _score_pool(D, cols, y, yte, cfg.k_all)
            rows.append({**base, 'analysis': 'W0', 'pool': 'all', 'acc': acc, 'rank': rk,
                         'L': _locality(idx, task['t_sec'], task['t_sec'][te], cols),
                         'short_err': _short_err(pred, yte, leg[te])})
            # label null: permute A-D within each training trial; neighbours are label-free,
            # so a draw is a relabelling of fixed neighbour indices
            perms = rng.permuted(np.tile(np.arange(N_STATES), (cfg.n_null_within,
                                                               trial.max() + 1, 1)), axis=2)
            yp = perms[:, trial, y]                                   # (draws, samples)
            null_correct[v] += (vote(yp[:, idx])[0] == yte).sum(axis=1)
            # W2: equal-size pools, near = the trials either side
            if len(trials) >= cfg.min_trials_w2 and (kt - 1) in trials and (kt + 1) in trials:
                far = [j for j in trials if abs(j - kt) >= cfg.far_gap]
                rand = [j for j in trials if j != kt]
                near_leg = np.log(leg[np.isin(trial, [kt - 1, kt + 1])].mean())
                far_dm = [j for j in far if abs(np.log(leg[trial == j].mean()) - near_leg)
                          <= np.log(1.2)]
                if len(far) >= 2:
                    pools = {'near': [[kt - 1, kt + 1]],
                             'far': [rng.choice(far, 2, replace=False)
                                     for _ in range(cfg.n_pool_draws)],
                             'rand': [rng.choice(rand, 2, replace=False)
                                      for _ in range(cfg.n_pool_draws)]}
                    if len(far_dm) >= 2:
                        pools['far_dm'] = [rng.choice(far_dm, 2, replace=False)
                                           for _ in range(cfg.n_pool_draws)]
                    for name, draws in pools.items():
                        sc = [_score_pool(D, np.flatnonzero(np.isin(trial, p)), y, yte,
                                          cfg.k_small)[:2] for p in draws]
                        rows.append({**base, 'analysis': 'W2', 'pool': name,
                                     'acc': float(np.mean([s[0] for s in sc])),
                                     'rank': float(np.mean([s[1] for s in sc])),
                                     'L': np.nan, 'short_err': np.nan})
            if v in ('raw', 'centred', 'project'):
                for j in trials:
                    if j == kt:
                        continue
                    acc1 = _score_pool(D, np.flatnonzero(trial == j), y, yte, 1)[0]
                    lag_rows.append({'task_idx': ti, 'trial': int(kt), 'ref': int(j),
                                     'lag': int(j - kt), 'variant': v, 'acc': acc1})
        n_tested += 4
    return rows, lag_rows, null_correct, n_tested


def _short_err(pred, yte, leg):
    """Among errors whose true state is not the shortest leg, fraction called the shortest
    leg's state (chance 1/3). A duration-driven decoder would sit above 1/3."""
    s = int(np.argmin(leg))
    m = (pred != yte) & (yte != yte[s])
    return float(np.mean(pred[m] == yte[s])) if m.any() else np.nan


# ---------------------------------------------------------------------------
# Across tasks
# ---------------------------------------------------------------------------

def cross_task(tasks, cfg, rng, variants=VARIANTS):
    """Leave one task out: X0 (all), X1 (near / far / rand by trial time), X3 place flags,
    X4 repeat-session ceiling, and the 4-way role-permutation null per variant."""
    rows, x3, diag = [], [], []
    null_correct = {v: np.zeros(cfg.n_null) for v in variants}
    n_tested = 0
    if len(tasks) < cfg.min_train_tasks + 1:
        return rows, x3, null_correct, n_tested, diag
    X = np.vstack([t['X'] for t in tasks])
    y = np.concatenate([t['y'] for t in tasks])
    trial = np.concatenate([t['trial'] for t in tasks])
    T = np.concatenate([t['T'] for t in tasks])
    tsec = np.concatenate([t['t_sec'] for t in tasks])
    g = np.concatenate([np.full(len(t['y']), i) for i, t in enumerate(tasks)])
    for v in variants:
        if v in ('detrend', 'project'):
            # per task, on its own trials: label-free and state-balanced, as the z-score is
            f = detrend if v == 'detrend' else project_drift
            Xv = np.vstack([f(t['X'], t['T'], t['trial'], np.ones(len(t['y']), bool),
                              cfg.spline_knots) for t in tasks])
        elif v == 'centred':
            Xv = centre_trials(X, trial, g)
        else:
            Xv = X
        for h, th in enumerate(tasks):
            te, fit = g == h, g != h
            Z = pca_fit(Xv, fit, cfg.n_pcs)(Xv)
            cols = np.flatnonzero(fit)
            D_all = cdist(Z[te], Z)
            train_ids = [u for u in range(len(tasks)) if u != h]
            dest_match = np.array([any(tasks[u]['task'][(s + 1) % N_STATES]
                                       == th['task'][(s + 1) % N_STATES] for u in train_ids)
                                   for s in th['y']])
            leg_match = np.array([any(tasks[u]['task'][s] == th['task'][s] and
                                      tasks[u]['task'][(s + 1) % N_STATES]
                                      == th['task'][(s + 1) % N_STATES] for u in train_ids)
                                  for s in th['y']])
            idx_all = cols[_nearest(D_all[:, cols], cfg.k_all)]
            pred_all, sc_all = vote(y[idx_all])
            # role null: an independent A-D relabelling per training task
            perms = rng.permuted(np.tile(np.arange(N_STATES), (cfg.n_null, len(tasks), 1)),
                                 axis=2)
            yp = perms[:, g, y]
            null_correct[v] += (vote(yp[:, idx_all])[0] == th['y']).sum(axis=1)
            n_tested += int(te.sum()) if v == variants[0] else 0
            ent = _entropy(g[idx_all], len(train_ids))
            loc = _locality(idx_all, tsec, tsec[te], cols)
            diag.append({'task_idx': h, 'variant': v, 'L': loc, 'source_entropy': ent})
            te_idx = np.flatnonzero(te)
            for kt in np.unique(th['trial']):
                mk = th['trial'] == kt
                base = dict(task_idx=h, trial=int(kt), T=float(th['T'][mk][0]), variant=v)
                yk = th['y'][mk]
                pk = pred_all[mk]
                rows.append({**base, 'analysis': 'X0', 'pool': 'all',
                             'acc': float(np.mean(pk == yk)),
                             'rank': float(np.mean(rank_score(sc_all[mk], yk))),
                             'short_err': _short_err(pk, yk, th['leg_s'][mk])})
                if v == 'raw':
                    for s_, p_, dm, lm in zip(yk, pk, dest_match[mk], leg_match[mk]):
                        x3.append({'task_idx': h, 'trial': int(kt), 'state': int(s_),
                                   'pred': int(p_), 'correct': bool(p_ == s_),
                                   'dest_match': bool(dm), 'leg_match': bool(lm)})
                    rows += _x1_rows(D_all[mk], th['T'][mk][0], T, trial, g, train_ids, y, yk,
                                     cfg, rng, base)
            if v == 'raw' and th['repeat'] is not None:
                rows += _x4_rows(th, cfg, h)
    return rows, x3, null_correct, n_tested, diag


def _x1_rows(D, Tk, T, trial, g, train_ids, y, yk, cfg, rng, base):
    """Pools of 2 trials per training task: nearest in trial time, far (>= far_s), or random.
    A task enters only if it can supply all three, so every pool spans the same tasks."""
    near, far, rand = [], [], []
    for u in train_ids:
        tu = np.unique(trial[g == u])
        Tu = np.array([T[(g == u) & (trial == j)][0] for j in tu])
        dT = np.abs(Tu - Tk)
        nc, fc = tu[dT <= cfg.near_s], tu[dT >= cfg.far_s]
        if len(nc) < cfg.per_task_pool or len(fc) < cfg.per_task_pool:
            continue
        near.append((u, tu[np.argsort(dT)[:cfg.per_task_pool]]))
        far.append((u, fc))
        rand.append((u, tu))
    if len(near) < cfg.min_train_tasks:
        return []

    def cols_of(sel):
        return np.flatnonzero(np.any([(g == u) & np.isin(trial, js) for u, js in sel], axis=0))

    out = []
    sc = _score_pool(D, cols_of(near), y, yk, cfg.k_all)
    out.append({**base, 'analysis': 'X1', 'pool': 'near', 'acc': sc[0], 'rank': sc[1],
                'short_err': np.nan, 'n_tasks': len(near)})
    for name, cand in (('far', far), ('rand', rand)):
        s = [_score_pool(D, cols_of([(u, rng.choice(js, cfg.per_task_pool, replace=False))
                                     for u, js in cand]), y, yk, cfg.k_all)[:2]
             for _ in range(cfg.n_pool_draws)]
        out.append({**base, 'analysis': 'X1', 'pool': name,
                    'acc': float(np.mean([a for a, _ in s])),
                    'rank': float(np.mean([r for _, r in s])),
                    'short_err': np.nan, 'n_tasks': len(cand)})
    return out


def _x4_rows(th, cfg, h):
    """Same layout, other session: this session's trials decode the repeat's (raw, k_all)."""
    rep = th['repeat']
    X = np.vstack([th['X'], rep['X']])
    fit = np.r_[np.ones(len(th['y']), bool), np.zeros(len(rep['y']), bool)]
    Z = pca_fit(X, fit, cfg.n_pcs)(X)
    D = cdist(Z[~fit], Z[fit])
    ytr = th['y']
    out = []
    for kt in np.unique(rep['trial']):
        mk = rep['trial'] == kt
        acc, rk, _, _ = _score_pool(D[mk], np.arange(len(ytr)), ytr, rep['y'][mk], cfg.k_all)
        out.append({'task_idx': h, 'trial': int(kt), 'T': float(rep['T'][mk][0]),
                    'variant': 'raw', 'analysis': 'X4', 'pool': 'repeat', 'acc': acc,
                    'rank': rk, 'short_err': np.nan})
    return out


def _entropy(src, n):
    """Mean normalised entropy of the neighbours' source tasks (1 = spread evenly)."""
    if n < 2:
        return np.nan
    h = []
    for row in src:
        p = np.bincount(row, minlength=src.max() + 1) / len(row)
        p = p[p > 0]
        h.append(-(p * np.log(p)).sum() / np.log(n))
    return float(np.mean(h))


def pc_time_r2(tasks, n_pcs=5):
    """R^2 of each top PC (raw features, all tasks) on trial time: how much PCA favours time."""
    X = np.vstack([t['X'] for t in tasks])
    T = np.concatenate([t['T'] for t in tasks])
    Z = pca_fit(X, np.ones(len(X), bool), n_pcs)(X)
    return [float(np.corrcoef(Z[:, i], T)[0, 1] ** 2) for i in range(Z.shape[1])]


# ---------------------------------------------------------------------------
# One recday, and aggregation
# ---------------------------------------------------------------------------

def run_recday(data_dic, mouse_recday, cfg=None, neuron_subset=None):
    """Everything for one recday -> dict of DataFrames and null arrays; None if unusable."""
    cfg = cfg or KNNConfig()
    tasks = prepare_recday(data_dic, mouse_recday, cfg, neuron_subset)
    if tasks is None:
        return None
    import zlib
    rng = np.random.default_rng([cfg.seed, zlib.crc32(mouse_recday.encode())])   # hash() is salted
    rows, lags = [], []
    null_w = {v: np.zeros(cfg.n_null_within) for v in VARIANTS}
    n_w = 0
    for ti, t in enumerate(tasks):
        r, lg, nc, n = within_task(t, cfg, rng, ti)
        rows += r
        lags += lg
        for v in VARIANTS:
            null_w[v] += nc[v]
        n_w += n
    xr, x3, null_x, n_x, diag = cross_task(tasks, cfg, rng)
    rows += xr
    meta = pd.DataFrame([{'task_idx': i, 'session': t['session'], 'task': tuple(t['task']),
                          'first_exposure': t['first_exposure'],
                          'n_trials_used': int(len(np.unique(t['trial']))),
                          'has_repeat': t['repeat'] is not None} for i, t in enumerate(tasks)])
    df = pd.DataFrame(rows).merge(meta[['task_idx', 'first_exposure']], on='task_idx')
    for d in (df, meta):
        d.insert(0, 'recday', mouse_recday)
        d.insert(1, 'mouse', mouse_recday.split('_')[0])
    return {'rows': df, 'lags': pd.DataFrame(lags).assign(recday=mouse_recday),
            'x3': pd.DataFrame(x3).assign(recday=mouse_recday),
            'diag': pd.DataFrame(diag).assign(recday=mouse_recday), 'meta': meta,
            'null_within': {v: null_w[v] / max(n_w, 1) for v in VARIANTS},
            'null_cross': {v: null_x[v] / max(n_x, 1) for v in VARIANTS},
            'n_cross_tested': n_x, 'pc_time_r2': pc_time_r2(tasks),
            'n_neurons': int(len(tasks[0]['neuron_idx']))}


def check_null(null, n_samples, chance=0.25, n_se=4.0):
    """The role / label nulls have mean exactly 0.25 under the vote rule; raise if not."""
    null = np.asarray(null)
    if not len(null) or n_samples == 0:
        return
    se = np.sqrt(chance * (1 - chance) / n_samples) / np.sqrt(len(null))
    if abs(null.mean() - chance) > n_se * max(se, null.std() / np.sqrt(len(null))):
        raise AssertionError(f'null mean {null.mean():.4f} is not at chance {chance}')


def recday_means(rows):
    """Trial -> task -> recday means per (analysis, pool, variant); tasks weigh equally."""
    t = rows.groupby(['recday', 'mouse', 'analysis', 'pool', 'variant', 'task_idx'])[
        ['acc', 'rank']].mean()
    return t.groupby(['recday', 'mouse', 'analysis', 'pool', 'variant']).mean().reset_index()


def contrasts(rd_means):
    """Per recday: W2 near-rand / far-rand / far_dm-rand and X1 near-rand / far-rand, X0 - .25."""
    p = rd_means.pivot_table(index=['recday', 'mouse', 'variant'], columns=['analysis', 'pool'],
                             values='acc')
    out = pd.DataFrame(index=p.index)
    for a, pools in (('W2', ('near', 'far', 'far_dm')), ('X1', ('near', 'far'))):
        for pl in pools:
            if (a, pl) in p and (a, 'rand') in p:
                out[f'{a}_{pl}-rand'] = p[(a, pl)] - p[(a, 'rand')]
    if ('X0', 'all') in p:
        out['X0-chance'] = p[('X0', 'all')] - 0.25
    if ('W0', 'all') in p:
        out['W0-chance'] = p[('W0', 'all')] - 0.25
    return out.reset_index()


def mouse_test(con, col, variant):
    """One-sample t on mouse means (recday -> mouse), with every mouse and the sign count."""
    from scipy import stats
    s = con[con.variant == variant].groupby('mouse')[col].mean().dropna()
    if len(s) < 2:
        return {'n_mice': len(s)}
    t = stats.ttest_1samp(s, 0)
    ci = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s))
    return {'n_mice': len(s), 'mean': float(s.mean()), 'ci': (float(ci[0]), float(ci[1])),
            't': float(t.statistic), 'p': float(t.pvalue), 'n_pos': int((s > 0).sum()),
            'per_mouse': s.round(4).to_dict()}


PRIMARIES = [('W2_near-rand', 'raw'), ('W2_near-rand', 'centred'), ('X0-chance', 'raw'),
             ('X1_near-rand', 'raw')]


def primary_table(con):
    """The four pre-registered primaries with Holm-adjusted p (mice as replicates)."""
    res = [{'contrast': c, 'variant': v, **mouse_test(con, c, v)} for c, v in PRIMARIES]
    ps = [r.get('p', np.nan) for r in res]
    order = np.argsort([p if np.isfinite(p) else np.inf for p in ps])
    m, running = sum(np.isfinite(ps)), 0.0
    for rank, i in enumerate(order):
        if not np.isfinite(ps[i]):
            res[i]['p_holm'] = np.nan
            continue
        running = max(running, min(1.0, (m - rank) * ps[i]))
        res[i]['p_holm'] = running
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------
# Plots (notebook only; mice are the points)
# ---------------------------------------------------------------------------

_ARIAL = '/ceph/behrens/max_kirkby/goal_sequencing_1/font/Arial.ttf'
LAG_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 99)]


def _rc():
    import os
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


def lag_curve(lags):
    """Mouse-mean 1-NN accuracy per |lag| bin and variant -> DataFrame (mouse, variant, bin)."""
    d = lags.assign(mouse=lags.recday.str.split('_').str[0], a=lags.lag.abs())
    d['bin'] = np.select([(d.a >= lo) & (d.a <= hi) for lo, hi in LAG_BINS],
                         [f'{lo}' if lo == hi else f'{lo}-{hi if hi < 99 else "+"}'
                          for lo, hi in LAG_BINS], default='')
    g = d.groupby(['recday', 'mouse', 'variant', 'bin', 'task_idx']).acc.mean()
    g = g.groupby(['recday', 'mouse', 'variant', 'bin']).mean()
    return g.groupby(['mouse', 'variant', 'bin']).mean().reset_index()


def plot_knn_summary(payloads, out_path=None):
    """(a) W0 / X0 by variant, (b) W2 pools, (c) lag curve, (d) X1 pools; one colour per region.

    payloads: {'LEC': payload, 'PFC': payload} from run_knn_state_decoding.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    colors = ccgp.REGION_COLORS
    with mpl.rc_context(_rc()):
        fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.4), gridspec_kw={'wspace': 0.45})
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.3, top=0.82)
        for ri, (reg, p) in enumerate(payloads.items()):
            c = colors.get(reg, '#2C2C2A')
            m = p['recday_means'].groupby(['analysis', 'pool', 'variant', 'mouse']).acc.mean()
            off = (ri - 0.5) * 0.18
            # (a) ceiling and cross-task by variant
            for ai, an in enumerate(('W0', 'X0')):
                for vi, v in enumerate(VARIANTS):
                    s = m.get((an, 'all', v))
                    if s is None:
                        continue
                    x = ai * 5 + vi + off
                    ax[0].scatter(np.full(len(s), x), s.values, s=6, color=c, lw=0)
                    ax[0].hlines(s.mean(), x - 0.08, x + 0.08, color=c, lw=1.2)
            # (b) within-task pools, raw and centred
            for vi, v in enumerate(('raw', 'centred')):
                pools = [pl for pl in ('near', 'rand', 'far') if ('W2', pl, v) in m.index]
                if not pools:
                    continue
                y = pd.concat([m[('W2', pl, v)] for pl in pools], axis=1)
                xs = vi * 4 + np.arange(len(pools)) + off
                for _, row in y.iterrows():
                    ax[1].plot(xs, row.values, color=c, lw=0.5, alpha=0.6)
                ax[1].plot(xs, y.mean().values, color=c, lw=1.5, marker='o', ms=3,
                           label=reg if vi == 0 else None)
            # (c) lag curve, mouse means
            lc = lag_curve(p['lags'])
            order = [f'{lo}' if lo == hi else f'{lo}-{hi if hi < 99 else "+"}'
                     for lo, hi in LAG_BINS]
            for v, ls in (('raw', '-'), ('centred', '--')):
                s = lc[lc.variant == v].groupby('bin').acc.mean().reindex(order)
                ax[2].plot(range(len(order)), s.values, color=c, ls=ls, marker='o', ms=2.5,
                           label=f'{reg} {v}')
            # (d) cross-task pools by trial time
            pools = [pl for pl in ('near', 'rand', 'far') if ('X1', pl, 'raw') in m.index]
            if pools:
                y = pd.concat([m[('X1', pl, 'raw')] for pl in pools], axis=1)
                xs = np.arange(len(pools)) + off
                for _, row in y.iterrows():
                    ax[3].plot(xs, row.values, color=c, lw=0.5, alpha=0.6)
                ax[3].plot(xs, y.mean().values, color=c, lw=1.5, marker='o', ms=3)
        ax[0].set_xticks([0, 1, 2, 3, 5, 6, 7, 8],
                         [f'W0 {v[:3]}' for v in VARIANTS] + [f'X0 {v[:3]}' for v in VARIANTS],
                         rotation=90)
        ax[0].set_ylabel('4-way accuracy')
        ax[0].set_title('within (W0) / across (X0)')
        ax[1].set_xticks([0, 1, 2, 4, 5, 6], ['near', 'rand', 'far'] * 2, rotation=90)
        ax[1].set_title('within, 2-trial pools\nraw | centred')
        ax[1].legend(frameon=False)
        ax[2].set_xticks(range(len(order)), order)
        ax[2].set_xlabel('|lag| (trials)')
        ax[2].set_title('1-NN from one trial')
        ax[2].legend(frameon=False)
        ax[3].set_xticks([0, 1, 2], ['near', 'rand', 'far'])
        ax[3].set_title('across, by trial time')
        for a in ax:
            a.axhline(0.25, ls='--', color='#555555', lw=0.8)
    if out_path is not None:
        with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0}):
            fig.savefig(out_path, bbox_inches=None)
    return fig
