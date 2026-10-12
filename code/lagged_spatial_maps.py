"""Lagged spatial maps (El-Gaby Fig. 5e/f, ED 6b/7h) and the place-vs-goal frame test.

Everything reads the arrays El-Gaby's Figure5_Figure6 notebook saved (cell 76) from one
region's run tree; no data_dic, no fit. Driven by `run_lagged_spatial_maps.py --region`.

Map axes: `Spatial_anchoring_Phase_shifted_maps_<rd>.npy` is (neurons, sessions, 12 lags,
21 locations). Lag k = where the mouse was 30*k degrees of task earlier; k=0 is now. Locations
0-8 are nodes 1-9 row-major on the 3x3 grid, 9-20 are the edges in `EDGES` order. Values are
mean spike counts per 25 ms bin in the neuron's preferred goal-progress third.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.stats as st

REPO = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
RUN_DIRS = {'lec': os.path.join(REPO, 'mFC_data', 'lec_replication_run', 'data',
                                'Intermediate_objects'),
            'pfc': os.path.join(REPO, 'mFC_data', 'replication_run', 'data',
                                'Intermediate_objects')}
OUT_DIRS = {'lec': os.path.join(REPO, 'data', 'figures', 'lagged_spatial'),
            'pfc': os.path.join(REPO, 'mFC_data', 'data', 'figures', 'lagged_spatial')}

#: His cell-34 printout, the equivalence target: panel -> (n, t). State-tuned pool.
CELL34 = {'lec': {'all': (1426, 45.50978149152688), 'nonzero': (356, 8.42113801183875),
                  'strict': (179, 5.8257878238940695)},
          'pfc': {'all': (738, 20.208118105653917), 'nonzero': (307, 5.722014341994327),
                  'strict': (164, 3.815675275953653)}}

N_LAGS = 12
N_NODES = 9
BIN_HZ = 40.0                                  # 25 ms bins
CLOSE30 = [0, 11]                              # his close_to_anchor_bins30
CLOSE90 = [0, 1, 2, 11, 10, 9]                 # his close_to_anchor_bins90
#: edges 10-21 as node pairs, and their cells on the 5x5 node+edge grid (his cell 13)
EDGES = [(1, 2), (2, 3), (1, 4), (2, 5), (3, 6), (4, 5), (5, 6), (4, 7), (5, 8), (6, 9),
         (7, 8), (8, 9)]
EDGE_GRID = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4),
             (4, 1), (4, 3)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def recdays(region, io_dir=None):
    """Recdays in his loop order (`combined_ABCDonly_days.npy`); cell 34 pools in this order."""
    io_dir = io_dir or RUN_DIRS[region]
    return [str(r) for r in np.load(os.path.join(io_dir, 'combined_ABCDonly_days.npy'))]


def non_repeat_sessions(tasks, num_trials):
    """His `non_repeat_ses_maker`: first session of each distinct task with trials > 0."""
    tasks = np.asarray(tasks)
    first = [not any(np.array_equal(tasks[i], tasks[j]) for j in range(i))
             for i in range(len(tasks))]
    return np.flatnonzero(np.asarray(first) & (np.asarray(num_trials) > 0))


def load_recday(region, rd, io_dir=None, rates=True):
    """One recday's maps + bookkeeping. maps (N, S, 12, 21); tasks (S, 4) nodes 1-9."""
    io_dir = io_dir or RUN_DIRS[region]
    f = lambda name: os.path.join(io_dir, name)            # noqa: E731
    tasks = np.load(f(f'Task_data_{rd}.npy'), allow_pickle=True).astype(int)
    num_trials = np.load(f(f'Num_trials_{rd}.npy'), allow_pickle=True).astype(int)
    out = {'recday': rd, 'mouse': rd.split('_')[0],
           'maps': np.load(f(f'Spatial_anchoring_Phase_shifted_maps_{rd}.npy')),
           'tasks': tasks, 'num_trials': num_trials,
           'non_repeat': non_repeat_sessions(tasks, num_trials),
           'state': np.load(f(f'State_{rd}.npy')).astype(bool)}
    if rates:
        # mean over the task sessions he used, in Hz; only for rate matching
        tot, nb = 0.0, 0
        for s in out['non_repeat']:
            p = f(f'Neuron_raw_{rd}_{s}.npy')
            if os.path.exists(p):
                a = np.load(p)
                tot = tot + a.sum(axis=1)
                nb += a.shape[1]
        out['rate_hz'] = tot / nb * BIN_HZ if nb else np.full(len(out['state']), np.nan)
    return out


# ---------------------------------------------------------------------------
# His cell 31, per neuron
# ---------------------------------------------------------------------------

def _nanargmax(x):
    """His `_nanargmax`: NaN when every entry is NaN."""
    try:
        return np.nanargmax(x)
    except ValueError:
        return np.nan


def _upper(a):
    return a[np.triu_indices(len(a), k=1)]


def _pdcorr(X):
    """Session x session pairwise-complete Pearson of X (S, L), exactly as his pandas call."""
    return pd.DataFrame.to_numpy(pd.DataFrame(X).T.corr())


def heldout_lagged_corr(M):
    """His cell 31 for one neuron. M (S, 12, 9) nodes-only maps over non-repeat sessions.

    Returns (r_all, r_nonzero, r_strict, max_lags (S,)). The best lag for held-out session s is
    chosen on the other sessions only; s is then scored against each of them at that lag.
    """
    S = M.shape[0]
    others = [np.setdiff1d(np.arange(S), s) for s in range(S)]
    max_lags = [_nanargmax([np.nanmean(_upper(_pdcorr(M[others[s], lag])))
                            for lag in range(M.shape[1])]) for s in range(S)]

    def score(s, exclude):
        L = max_lags[s]
        if np.isnan(L) or L in exclude:
            return np.nan
        return np.nanmean(_pdcorr(M[:, L])[s][others[s]])

    with np.errstate(invalid='ignore'), _quiet():
        r_all = np.nanmean([score(s, ()) for s in range(S)])
        r_nz = np.nanmean([score(s, CLOSE30) for s in range(S)])
        r_st = np.nanmean([score(s, CLOSE90) for s in range(S)])
    # his "removing spatial neurons" override: a modal best lag near 0, seen twice or more
    mode = st.mode(max_lags, keepdims=True)
    if mode[0][0] in CLOSE30 and mode[1][0] > 1:
        r_nz = np.nan
    if mode[0][0] in CLOSE90 and mode[1][0] > 1:
        r_st = np.nan
    return r_all, r_nz, r_st, np.asarray(max_lags, dtype=float)


class _quiet:
    """Silence the all-NaN mean warnings his code also raises."""
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def recday_heldout(rec):
    """Cell 31 over every neuron of a recday -> dict of (N,) arrays + max_lags (N, S)."""
    M = rec['maps'][:, rec['non_repeat'], :, :N_NODES]
    rows = [heldout_lagged_corr(M[n]) for n in range(M.shape[0])]
    return {'r_all': np.array([r[0] for r in rows]),
            'r_nonzero': np.array([r[1] for r in rows]),
            'r_strict': np.array([r[2] for r in rows]),
            'max_lags': np.vstack([r[3] for r in rows]) if rows else np.zeros((0, 0))}


def cell34(per_recday, recday_order, state):
    """His cell 34: pool state-tuned neurons in recday order, drop NaN -> {panel: (n, t)}."""
    out = {}
    for panel in ('all', 'nonzero', 'strict'):
        v = np.hstack([per_recday[rd][f'r_{panel}'] for rd in recday_order])
        v = v[np.hstack([state[rd] for rd in recday_order])]
        v = v[~np.isnan(v)]
        out[panel] = (len(v), float(st.ttest_1samp(v, 0).statistic))
    return out


# ---------------------------------------------------------------------------
# Place frame vs goal frame (item 13, primary)
# ---------------------------------------------------------------------------

def _rows_corr(a, b):
    """Pearson r along the last axis; NaN unless all entries of both are finite."""
    ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        a = a - a.mean(-1, keepdims=True)
        b = b - b.mean(-1, keepdims=True)
        den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
        r = (a * b).sum(-1) / den
    r[~ok | ~(den > 0)] = np.nan
    return r


def frame_pair_corr(M, goals, node_perms):
    """Pairwise 4-vector correlations in both frames, for every node relabelling given.

    M (S, 12, 9) node maps; goals (S, 4) 0-based nodes in A-D order; node_perms (D, S, 9).
    Draw d relabels session s's nodes by node_perms[d, s] (the same at every lag), then:
      goal[d, s, t]  = corr(map_s[g_s], map_t[g_t])                   same letter, wherever
      place[d, s, t] = mean(corr(map_s[g_s], map_t[g_s]), corr(map_s[g_t], map_t[g_t]))
    The identity is the observed statistic. The null is the same statistic on scrambled maps,
    which keeps how pairs share nodes and sessions; a letter-only shuffle did not, and passed
    14% of pure-noise cells at alpha 0.05 (synthetic h).
    Returns (place, goal), each (D, S, S, 12), NaN on the diagonal.
    """
    D, S = node_perms.shape[:2]
    Mp = M[np.arange(S)[None, :, None, None], np.arange(M.shape[1])[None, None, :, None],
           node_perms[:, :, None, :]]                                        # (D, S, 12, 9)
    W = np.moveaxis(Mp[..., goals], 3, 2)             # W[d, s, k] = map_s at g_k: (D, S, S, 12, 4)
    si, ti = np.meshgrid(np.arange(S), np.arange(S), indexing='ij')
    Wss, Wtt = W[:, si, si], W[:, ti, ti]
    goal = _rows_corr(Wss, Wtt)
    with _quiet():
        place = np.nanmean(np.stack([_rows_corr(Wss, W[:, ti, si]),
                                     _rows_corr(W[:, si, ti], Wtt)]), axis=0)
    eye = np.eye(S, dtype=bool)
    place[:, eye] = np.nan
    goal[:, eye] = np.nan
    return place, goal


def _nanargmax_rows(X):
    """nanargmax over the last axis, NaN where a row is all NaN."""
    allnan = np.isnan(X).all(-1)
    out = np.nanargmax(np.where(allnan[..., None], 0.0, X), axis=-1).astype(float)
    out[allnan] = np.nan
    return out


def heldout_frame_score(C):
    """Cell-31 held-out score from pair matrices C (D, S, S, 12) -> (scores (D,), lags (D, S)).

    For held-out s the best lag maximises the mean pair correlation among the other sessions;
    s is then scored against each of them at that lag, and the neuron takes the mean over s.
    """
    D, S = C.shape[:2]
    scores = np.full((D, S), np.nan)
    lags = np.full((D, S), np.nan)
    with _quiet():
        for s in range(S):
            tr = np.setdiff1d(np.arange(S), s)
            if len(tr) < 2:
                continue
            iu = np.triu_indices(len(tr), k=1)
            L = _nanargmax_rows(np.nanmean(C[:, tr[iu[0]], tr[iu[1]], :], axis=1))
            ok = np.isfinite(L)
            v = C[:, s][:, tr, :]                                             # (D, ntr, 12)
            Li = np.where(ok, L, 0).astype(int)
            v = np.take_along_axis(v, np.broadcast_to(Li[:, None, None], (D, len(tr), 1)),
                                   axis=2)[..., 0]
            scores[:, s] = np.where(ok, np.nanmean(v, axis=1), np.nan)
            lags[:, s] = L
        return np.nanmean(scores, axis=1), lags


def frame_test(M, goals, n_null=200, rng=None):
    """Held-out place and goal scores for one neuron, each against a node-scramble null.

    M (S, 12, 9), goals (S, 4) 0-based. The null reruns lag selection, so the max over 12
    lags is inside it. Returns r_place, r_goal, one-sided p_place / p_goal, null means, lags.
    """
    rng = rng or np.random.default_rng(0)
    S = M.shape[0]
    base = np.tile(np.arange(N_NODES), (S, 1))
    perms = np.concatenate([base[None],
                            np.stack([rng.permuted(base, axis=1) for _ in range(n_null)])])
    place, goal = frame_pair_corr(M, goals, perms)
    rp, lp = heldout_frame_score(place)
    rg, lg = heldout_frame_score(goal)

    def pval(obs, null):
        null = null[np.isfinite(null)]
        if not np.isfinite(obs) or not len(null):
            return np.nan
        return (1 + np.sum(null >= obs)) / (1 + len(null))

    with _quiet():
        return {'r_place': rp[0], 'r_goal': rg[0],
                'p_place': pval(rp[0], rp[1:]), 'p_goal': pval(rg[0], rg[1:]),
                'null_place_mean': np.nanmean(rp[1:]), 'null_goal_mean': np.nanmean(rg[1:]),
                'lag_place': lp[0], 'lag_goal': lg[0]}


def recday_frame_test(rec, n_null=200, seed=0, neurons=None):
    """frame_test over a recday's neurons -> DataFrame, one row per neuron tested."""
    rng = np.random.default_rng(seed)
    M = rec['maps'][:, rec['non_repeat'], :, :N_NODES]
    goals = rec['tasks'][rec['non_repeat']] - 1
    neurons = range(M.shape[0]) if neurons is None else neurons
    rows = []
    for n in neurons:
        r = frame_test(M[n], goals, n_null=n_null, rng=rng)
        rows.append({'recday': rec['recday'], 'neuron': int(n),
                     **{k: v for k, v in r.items() if not k.startswith('lag_')},
                     'lag_place_modal': _mode_or_nan(r['lag_place']),
                     'lag_goal_modal': _mode_or_nan(r['lag_goal'])})
    return pd.DataFrame(rows)


def _mode_or_nan(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    v, c = np.unique(x, return_counts=True)
    return float(v[np.argmax(c)])


def goal_overlap(goals):
    """Mean number of shared goal nodes over session pairs (4 = identical sets)."""
    S = len(goals)
    if S < 2:
        return np.nan
    return float(np.mean([len(set(goals[s]) & set(goals[t]))
                          for s in range(S) for t in range(s + 1, S)]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def node_edge_grid(v21):
    """(21,) node+edge values -> (5, 5) grid, NaN at the four (odd, odd) cells."""
    g = np.full((5, 5), np.nan)
    for i in range(N_NODES):
        g[2 * (i // 3), 2 * (i % 3)] = v21[i]
    for j, (r, c) in enumerate(EDGE_GRID):
        g[r, c] = v21[N_NODES + j]
    return g


def plot_lagged_maps(rec, neuron, best_lag=None, title=None, cmap='viridis', out_path=None):
    """Fig. 5e layout on the 9 nodes his statistic scores: rows = tasks, columns = lags.

    The current location is on the right, then successive lags into the past. Each map is
    normalised on its own, with max rate (Hz) in the corner and goal letters a-d on their nodes.
    The top row is the cross-task correlation at each lag; the right-hand column zooms in on
    `best_lag` (default: the neuron's modal held-out best lag).
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    M = rec['maps'][neuron, rec['non_repeat'], :, :N_NODES]           # (S, 12, 9)
    goals = rec['tasks'][rec['non_repeat']]
    S = M.shape[0]
    if best_lag is None:
        best_lag = _mode_or_nan(heldout_lagged_corr(M)[3])
    best_lag = 0 if not np.isfinite(best_lag) else int(best_lag)
    lags = list(range(N_LAGS))[::-1]
    corr = [np.nanmean(_upper(_pdcorr(M[:, L]))) for L in lags]
    with mpl.rc_context(_rc()):
        fig = plt.figure(figsize=(7.2, 0.9 + 0.55 * S))
        gs = fig.add_gridspec(S + 1, N_LAGS + 2, width_ratios=[1] * N_LAGS + [0.4, 2.2],
                              height_ratios=[1.2] + [1] * S, wspace=0.12, hspace=0.18,
                              top=0.84, bottom=0.08)
        a = fig.add_subplot(gs[0, :N_LAGS])
        a.bar(range(N_LAGS), corr, color=[COLORS['lagged'] if L == best_lag else COLORS['bar']
                                          for L in lags], width=0.7)
        a.axhline(0, color=COLORS['ink'], lw=0.5)
        a.set_xlim(-0.5, N_LAGS - 0.5)
        a.set_xticks(range(N_LAGS), [f'{30 * L}' for L in lags], fontsize=6)
        a.xaxis.tick_top()
        a.set_ylabel('cross-task r', fontsize=6)
        _clean(a)
        a.spines['bottom'].set_visible(False)
        for i in range(S):
            for j, L in enumerate(lags + [None]):
                big = L is None
                ax = fig.add_subplot(gs[i + 1, N_LAGS + 1] if big else gs[i + 1, j])
                m = M[i, best_lag if big else L].reshape(3, 3)
                lo, hi = np.nanmin(m), np.nanmax(m)
                ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(big)
                    sp.set_color(COLORS['lagged'])
                if big or L == best_lag:
                    for k, node in enumerate(goals[i]):
                        ax.text((node - 1) % 3, (node - 1) // 3, 'abcd'[k], ha='center',
                                va='center', fontsize=7 if big else 4, color='white')
                if big and np.isfinite(hi):
                    ax.text(2.45, -0.55, f'{hi * BIN_HZ:.1f} Hz', fontsize=5, ha='right',
                            va='bottom')
                if j == 0:
                    ax.set_ylabel(f'task {i + 1}', fontsize=6)
        fig.text(0.44, 0.02, 'lag (degrees of task in the past; 0 = current location)',
                 ha='center', fontsize=6)
        fig.text(0.86, 0.02, f'best lag {30 * best_lag}°', ha='center', fontsize=6,
                 color=COLORS['lagged'])
        fig.suptitle(title or f'{rec["recday"]} neuron {neuron}', fontsize=8, y=0.985)
    _save(fig, out_path)
    return fig


def pick_examples(table, k=3, rate=(0.5, 20.0), min_sessions=5):
    """Top held-out r in the 90° set among plausibly isolated rates, one per recday, plus the
    best zero-lag neuron as a contrast. Picking by r alone chose a 48 Hz and a 0.3 Hz unit."""
    ok = (table.rate_hz.between(*rate) & (table.n_sessions >= min_sessions) & table.state_tuned)
    lagged = (table[ok & table.lagged90].sort_values('r_strict', ascending=False)
              .drop_duplicates('recday').head(k).assign(kind='lagged 90°'))
    zero = (table[ok & (table.best_lag_modal == 0)].nlargest(1, 'r_all')
            .assign(kind='lag 0'))
    return pd.concat([lagged, zero])


#: GridMaze colours: lagged Viva Magenta, other / null Stone, bars Classic Blue, ink Caviar
COLORS = {'lagged': '#BE3455', 'other': '#B4B2A9', 'bar': '#0F4C81', 'ink': '#2C2C2A'}


def per_mouse_mean(t, col, mask=None):
    """Recday mean -> mouse mean of `col` over the rows in `mask` (mice are the replicates)."""
    s = t if mask is None else t[mask]
    return s.groupby(['mouse', 'recday'])[col].mean().groupby('mouse').mean()


def _rc():
    import elasticnet_regression_v5 as v5
    return v5._gridmaze_rc()


def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save(fig, out_path):
    if out_path is None:
        return
    import matplotlib as mpl
    with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig.savefig(out_path, bbox_inches=None)


def plot_heldout_population(table, region, out_path=None):
    """Fig. 5f / ED 7h: per-neuron held-out r of state-tuned neurons, and per-mouse means."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    panels = [('r_all', 'all state-tuned'), ('r_nonzero', 'non-zero lag, 30°'),
              ('r_strict', 'non-zero lag, 90°')]
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.4),
                                 gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.9})
        tuned = table.state_tuned
        mice = sorted(table.mouse.unique())
        for j, (col, name) in enumerate(panels):
            v = table.loc[tuned, col].dropna()
            ax = axes[0, j]
            ax.hist(v, bins=50, color=COLORS['other'])
            ax.axvline(0, ls='--', color=COLORS['ink'], lw=0.8)
            ax.set_title(f'{region.upper()} {name}\nn {len(v)} neurons, '
                         f't {st.ttest_1samp(v, 0).statistic:.1f}')
            ax.set_xlabel('held-out lagged-map r')
            ax.set_ylabel('neurons' if j == 0 else '')
            _clean(ax)
            pm = per_mouse_mean(table, col, tuned).reindex(mice)
            ax = axes[1, j]
            ax.scatter(range(len(mice)), pm.values, color=COLORS['bar'], s=12, zorder=3)
            ax.axhline(0, ls='--', color=COLORS['ink'], lw=0.8)
            ax.set_xticks(range(len(mice)), mice, rotation=90 if len(mice) > 6 else 0)
            ax.set_ylabel('mouse mean r' if j == 0 else '')
            _clean(ax)
    _save(fig, out_path)
    return fig


def plot_best_lag(table, region, out_path=None):
    """Modal best lag of state-tuned neurons; lags his 90° rule excludes are Stone."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    c = table.loc[table.state_tuned, 'best_lag_modal'].value_counts().reindex(range(N_LAGS),
                                                                               fill_value=0)
    with mpl.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(3.0, 1.8))
        ax.bar(c.index, c.values, color=[COLORS['other'] if k in CLOSE90 else COLORS['bar']
                                         for k in c.index])
        ax.set_xticks(range(N_LAGS), [f'{30 * k}' for k in range(N_LAGS)], rotation=90)
        ax.set_xlabel('best lag (degrees of task in the past)')
        ax.set_ylabel('state-tuned neurons')
        ax.set_title(f'{region.upper()}, n {int(c.sum())}')
        _clean(ax)
    _save(fig, out_path)
    return fig


def plot_frame_test(table, region, lag_col='lagged30', out_path=None):
    """Held-out place-frame vs goal-frame r per state-tuned neuron, and per-mouse r_goal - r_place."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    f = table[table.state_tuned & table.r_place.notna() & table.r_goal.notna()]
    lag = f[lag_col].fillna(False).astype(bool)
    with mpl.rc_context(_rc()):
        fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 2.4),
                                   gridspec_kw={'width_ratios': [1.4, 1], 'wspace': 0.5})
        for m, key, s in ((~lag, 'other', 4), (lag, 'lagged', 7)):
            a.scatter(f.r_place[m], f.r_goal[m], s=s, color=COLORS[key], lw=0,
                      label=f'{key} (n {int(m.sum())})', alpha=0.8)
        a.plot([-1, 1], [-1, 1], ls='--', color=COLORS['ink'], lw=0.8)
        a.axhline(0, color=COLORS['ink'], lw=0.5)
        a.axvline(0, color=COLORS['ink'], lw=0.5)
        a.set_xlabel('place frame r (same nodes)')
        a.set_ylabel('goal frame r (same letter)')
        a.set_title(f'{region.upper()} state-tuned, held out')
        a.legend(frameon=False, fontsize=6, loc='upper left')
        _clean(a)
        d = f.assign(delta=f.r_goal - f.r_place)
        pl, po = per_mouse_mean(d, 'delta', lag), per_mouse_mean(d, 'delta', ~lag)
        for m in sorted(set(pl.index) | set(po.index)):
            b.plot([0, 1], [po.get(m, np.nan), pl.get(m, np.nan)], color=COLORS['ink'], lw=0.6)
            b.scatter([0, 1], [po.get(m, np.nan), pl.get(m, np.nan)],
                      color=[COLORS['other'], COLORS['lagged']], s=12, zorder=3)
        b.axhline(0, ls='--', color=COLORS['ink'], lw=0.8)
        b.set_xticks([0, 1], ['other', lag_col])
        b.set_xlim(-0.4, 1.4)
        b.set_ylabel('mouse mean r_goal - r_place')
        _clean(b)
    _save(fig, out_path)
    return fig


def plot_anatomy(table, col, ylabel, out_path=None):
    """LEC: per-mouse effect of `col` per region group (state-tuned), mice as points."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import anatomy_split as asp
    t = table[table.state_tuned & table.group.isin(asp.ANALYSIS_GROUPS)]
    pm = asp.per_mouse_effect(t, col)
    with mpl.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(2.8, 2.0))
        for i, g in enumerate(asp.ANALYSIS_GROUPS):
            s = pm[pm.group == g]
            if not len(s):
                continue
            x = i + np.linspace(-0.12, 0.12, len(s))
            ax.scatter(x, s.effect, color=asp.REGION_COLORS[g], s=12, zorder=3)
            ax.hlines(s.effect.mean(), i - 0.25, i + 0.25, color=COLORS['ink'], lw=1)
            for xi, (_, r) in zip(x, s.iterrows()):
                ax.annotate(r.mouse, (xi, r.effect), fontsize=4, xytext=(2, 0),
                            textcoords='offset points')
        ax.axhline(0, ls='--', color=COLORS['ink'], lw=0.8)
        ax.set_xticks(range(len(asp.ANALYSIS_GROUPS)), asp.ANALYSIS_GROUPS, rotation=45,
                      ha='right')
        ax.set_ylabel(ylabel)
        _clean(ax)
    _save(fig, out_path)
    return fig
