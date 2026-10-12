"""Reward-place anchor bias (item 13, secondary): do lagged neurons anchor at reward ports?

    python anchor_distribution.py --region lec|pfc [--n-null 1000] [--n-perm 2000]

Anchors from two sources, side by side, each scored against the goals of the tasks it was fit
on (fold -> training sessions -> Task_data rows; never by position):
  v5      spec 12 (LEC) / spec 2 (PFC) Poisson fits. Per fold: the argmax (location, lag) of
          `cv_betas_collapsed`; phase = expected_anchor_phase(pref, lag).
  elgaby  Figure5_Figure6 cell 37 `best_node_phase_used`: [phase, location] per test split,
          chosen from that split's training sessions only.
reward_frac = fraction of training tasks in which the anchor node is a goal AND the anchor phase
is 0, i.e. at the port just after reward (ED 5a's "rewarded place"). goal_frac drops the phase.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

import anatomy_split as asp
import elasticnet_regression_v5 as v5
import lagged_spatial_maps as lsm

REPO = lsm.REPO
V5_RUNS = {'lec': os.path.join(REPO, 'data', 'figures',
                               'repro_poisson_v5_past_20260908_192043'),
           'pfc': os.path.join(REPO, 'mFC_data', 'data', 'figures',
                               'poisson_v5_past_20260907_153716')}
N_NODES = lsm.N_NODES


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_v5(run_dir):
    """`{recday: results}` from a V5 export dir, and the config it was run with."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(run_dir, '*_past_arrays.npz'))):
        res = v5.load_regression_outputs(path)
        runs[res['mouse_recday']] = res
    if not runs:
        raise FileNotFoundError(f'no *_past_arrays.npz in {run_dir}')
    stored = dict(next(iter(runs.values()))['config'])
    init = {k: v for k, v in stored.items()
            if k in v5.RegressionConfigV5.__init__.__code__.co_varnames and k != 'self'}
    return runs, v5.RegressionConfigV5(**init)


def v5_unit_table(region=None, run_dir=None, runs=None, config=None):
    """V5 per-unit table (state_tuned, selected_eg, selected_strict_eg, peak_lag, ...)."""
    if runs is None:
        runs, config = load_v5(run_dir or V5_RUNS[region])
    t = v5.build_unit_table(runs, config, regions={}, require_anatomy=False)
    return t.rename(columns={'order': 'neuron'})


def task_files(region, rd, io_dir=None):
    """(Task_data (S, 4) nodes 1-9, Num_trials (S,), num_passes (S, 3, 9)) from the run tree."""
    io_dir = io_dir or lsm.RUN_DIRS[region]
    f = lambda name: os.path.join(io_dir, name)            # noqa: E731
    return (np.load(f(f'Task_data_{rd}.npy'), allow_pickle=True).astype(int),
            np.load(f(f'Num_trials_{rd}.npy'), allow_pickle=True).astype(int),
            np.load(f(f'Spatial_anchoring_num_passes_all_{rd}.npy'), allow_pickle=True)
            .astype(float))


def _score(loc, phase, train, tasks, passes):
    """Goal membership of every node over the training tasks, and the visit weights at phase."""
    member = np.mean([np.isin(np.arange(N_NODES), tasks[s] - 1) for s in train], axis=0)
    w = passes[train, phase].sum(axis=0)
    # a phase never visited in training gives no weights; fall back to uniform, flagged
    w = w / w.sum() if w.sum() > 0 else np.full(N_NODES, 1 / N_NODES)
    g = float(member[loc])
    eg = float((w * member).sum())
    return {'goal_frac': g, 'reward_frac': g if phase == 0 else 0.0,
            'exp_goal_frac': eg, 'exp_reward_frac': eg if phase == 0 else 0.0,
            'member': member, 'w': w}


def _check_used_sessions(rd, used, tasks, num_trials):
    """V5's folds must be distinct tasks with trials; otherwise the task join is wrong."""
    rows = [tuple(tasks[s]) for s in used]
    if len(set(rows)) != len(rows) or (num_trials[used] < 1).any():
        raise AssertionError(f'{rd}: used_sessions {used} are not distinct trial-bearing tasks')


def v5_fold_anchors(runs, config, region, io_dir=None):
    """One row per (recday, neuron, fold) with a positive best beta."""
    rows = []
    for rd, res in runs.items():
        tasks, num_trials, passes = task_files(region, rd, io_dir)
        used = [int(s) for s in res['used_sessions']]
        _check_used_sessions(rd, used, tasks, num_trials)
        B = np.asarray(res['cv_betas_collapsed'], dtype=float)          # (N, F, 9, 12)
        pref = np.asarray(res['pref_phases'])
        for n in range(B.shape[0]):
            for fi in range(B.shape[1]):
                b = B[n, fi]
                if not np.isfinite(b).any() or np.nanmax(b) <= 0 or pref[n, fi] < 0:
                    continue
                loc, lag = np.unravel_index(np.nanargmax(b), b.shape)
                phase = int(v5.expected_anchor_phase(int(pref[n, fi]), int(lag), config))
                train = [s for j, s in enumerate(used) if j != fi]
                rows.append({'source': 'v5', 'recday': rd, 'mouse': rd.split('_')[0],
                             'neuron': n, 'fold': fi, 'test_session': used[fi],
                             'loc': int(loc), 'phase': phase, 'lag': int(lag),
                             'beta': float(b[loc, lag]),
                             **_score(int(loc), phase, train, tasks, passes)})
    return pd.DataFrame(rows)


def _load_or_none(path):
    """np.load, or None for a missing file or an empty pickled rec_dd (PFC me10)."""
    try:
        a = np.load(path, allow_pickle=True)
        return None if a.dtype == object and a.shape == () else a
    except Exception:                                    # noqa: BLE001 - unpicklable rec_dd
        return None


def elgaby_split_anchors(region, io_dir=None):
    """One row per (recday, neuron, split) from his cross-validated anchors, plus unit flags."""
    io_dir = io_dir or lsm.RUN_DIRS[region]
    f = lambda name: os.path.join(io_dir, name)            # noqa: E731
    rows, units = [], []
    for rd in lsm.recdays(region, io_dir):
        bnp = _load_or_none(f(f'Spatial_anchoring_best_node_phase_used_{rd}.npy'))
        anch = _load_or_none(f(f'Spatial_anchoring_Anchored_bool_{rd}.npy'))
        if bnp is None or anch is None:
            print(f'  elgaby: {rd} skipped (no cross-validated anchors saved)')
            continue
        bnp = bnp.astype(float)
        tasks, num_trials, passes = task_files(region, rd, io_dir)
        ses = np.intersect1d(lsm.non_repeat_sessions(tasks, num_trials),
                             np.flatnonzero(num_trials > 2))
        if bnp.shape[1] != len(ses):
            raise AssertionError(f'{rd}: {bnp.shape[1]} splits vs {len(ses)} sessions')
        state = np.load(f(f'State_{rd}.npy')).astype(bool)
        lag = np.load(f(f'Anchor_trial_Anchor_lags_mean_{rd}.npy'), allow_pickle=True)
        lag = lag.astype(float)
        units.append(pd.DataFrame({'recday': rd, 'mouse': rd.split('_')[0],
                                   'neuron': np.arange(len(state)), 'state_tuned': state,
                                   'anchored': anch.astype(bool), 'lag_deg': lag,
                                   'lagged30': (lag > 30) & (lag < 330),
                                   'lagged90': (lag > 90) & (lag < 270)}))
        for k, test in enumerate(ses):
            train = [int(s) for s in ses if s != test]
            for n in range(bnp.shape[2]):
                ph, loc = bnp[0, k, n], bnp[1, k, n]
                if not (np.isfinite(ph) and np.isfinite(loc)):
                    continue
                rows.append({'source': 'elgaby', 'recday': rd, 'mouse': rd.split('_')[0],
                             'neuron': n, 'fold': k, 'test_session': int(test),
                             'loc': int(loc), 'phase': int(ph), 'lag': np.nan, 'beta': np.nan,
                             **_score(int(loc), int(ph), train, tasks, passes)})
    return pd.DataFrame(rows), pd.concat(units, ignore_index=True)


# ---------------------------------------------------------------------------
# Scores and tests
# ---------------------------------------------------------------------------

SCORES = ['goal_frac', 'reward_frac', 'exp_goal_frac', 'exp_reward_frac']


def neuron_scores(folds):
    """Mean over folds per neuron -> DataFrame keyed by (recday, neuron)."""
    g = folds.groupby(['recday', 'mouse', 'neuron'])
    out = g[SCORES].mean()
    out['n_folds'] = g.size()
    out['phase0_frac'] = g['phase'].apply(lambda p: float(np.mean(p == 0)))
    return out.reset_index()


def contrast(neurons, value, lagged_col, pool_col='pool', n_perm=2000, seed=0):
    """Lagged minus other, mouse-level, against a within-recday shuffle of the lagged label."""
    sub = neurons[neurons[pool_col]].copy()
    sub['group'] = np.where(sub[lagged_col], 'lagged', 'other')
    res = asp.within_recday_permutation(sub, value, 'lagged', 'other', n_perm=n_perm, seed=seed)
    per_m = asp.per_mouse_effect(sub, value, groups=['lagged', 'other'])
    res['per_mouse'] = per_m.pivot(index='mouse', columns='group', values='effect').round(4)
    res['n_lagged'] = int((sub.group == 'lagged').sum())
    res['n_other'] = int((sub.group == 'other').sum())
    return res


def occupancy_null(folds, neurons_mask_keys, value='reward_frac', n_null=1000, seed=0):
    """Observed per-mouse mean of `value` against anchors redrawn in proportion to visits.

    Each neuron-fold keeps its phase; only the node is redrawn, from that fold's training
    visit counts at that phase. Aggregation is neuron -> recday -> mouse, as for the observed.
    """
    rng = np.random.default_rng(seed)
    keep = folds.set_index(['recday', 'neuron']).index.isin(neurons_mask_keys)
    f = folds[keep].reset_index(drop=True)
    if not len(f):
        return pd.DataFrame()
    member = np.stack(f['member'].to_numpy())
    cw = np.cumsum(np.stack(f['w'].to_numpy()), axis=1)
    use_phase = (f['phase'].to_numpy() == 0) if value == 'reward_frac' else np.ones(len(f), bool)
    nid = f.groupby(['recday', 'neuron']).ngroup().to_numpy()
    n_neur = nid.max() + 1
    neur_rd = f.groupby(nid)['recday'].first()
    rid = pd.factorize(neur_rd)[0]
    rd_names = pd.factorize(neur_rd)[1]
    mice = pd.Series([r.split('_')[0] for r in rd_names])
    mid, mouse_names = pd.factorize(mice)

    def aggregate(per_fold):
        per_n = np.bincount(nid, per_fold, n_neur) / np.bincount(nid, minlength=n_neur)
        per_r = np.bincount(rid, per_n) / np.bincount(rid)
        return np.bincount(mid, per_r) / np.bincount(mid)

    obs = aggregate(f[value].to_numpy(dtype=float))
    null = np.empty((n_null, len(mouse_names)))
    ar = np.arange(len(f))
    for i in range(n_null):
        x = (rng.random(len(f))[:, None] > cw).sum(axis=1).clip(0, N_NODES - 1)
        null[i] = aggregate(member[ar, x] * use_phase)
    p = (1 + (null >= obs[None]).sum(axis=0)) / (1 + n_null)
    return pd.DataFrame({'mouse': mouse_names, 'observed': obs, 'null_mean': null.mean(axis=0),
                         'null_p95': np.percentile(null, 95, axis=0), 'p': p,
                         'n_recdays': np.bincount(mid)})


def anchor_histogram(folds, keys):
    """(3 phases, 9 nodes) counts of neuron-fold anchors for the neurons in `keys`."""
    keep = folds.set_index(['recday', 'neuron']).index.isin(keys)
    h = np.zeros((3, N_NODES))
    np.add.at(h, (folds.phase[keep].to_numpy(), folds['loc'][keep].to_numpy()), 1)
    return h


def occupancy_histogram(region, recday_list, io_dir=None):
    """(3, 9) visit counts over the non-repeat sessions, each recday normalised to sum 1."""
    tot = np.zeros((3, N_NODES))
    for rd in recday_list:
        tasks, num_trials, passes = task_files(region, rd, io_dir)
        p = passes[lsm.non_repeat_sessions(tasks, num_trials)].sum(axis=0)
        tot += p / p.sum() if p.sum() else 0
    return tot


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

#: True / shuffle pair for observed vs null; Stone for the null band
NULL_COLORS = {'observed': '#C03030', 'null': '#555555', 'band': '#B4B2A9'}


def plot_anchor_histograms(out, src='v5', win='30deg', out_path=None):
    """ED 7d-style anchor densities over phase x node: lagged, other, and visits (viridis)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    hs = [('lagged', out['hist'][(src, win, 'lagged')]), ('other', out['hist'][(src, win, 'other')]),
          ('visits', out['hist']['occupancy'])]
    hs = [(k, h / h.sum() if h.sum() else h) for k, h in hs]
    vmax = max(np.nanmax(h) for _, h in hs)
    with mpl.rc_context(lsm._rc()):
        fig, axes = plt.subplots(1, 3, figsize=(6.0, 1.3))
        for ax, (k, h) in zip(axes, hs):
            im = ax.imshow(h, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
            ax.set_xticks(range(N_NODES), range(1, N_NODES + 1))
            ax.set_yticks(range(3), ['early', 'mid', 'late'] if ax is axes[0] else [])
            ax.set_xlabel('node')
            ax.set_title(f'{out["region"].upper()} {src} {win}: {k}')
        fig.colorbar(im, ax=axes, fraction=0.02, label='fraction')
    lsm._save(fig, out_path)
    return fig


def plot_occupancy_null(out, src='v5', win='30deg', value='reward_frac', out_path=None):
    """Per mouse: observed mean `value` against the visit-weighted null (mean, 95th pct)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    with mpl.rc_context(lsm._rc()):
        fig, axes = plt.subplots(1, 2, figsize=(4.4, 1.9), sharey=True)
        for ax, grp in zip(axes, ('lagged', 'other')):
            t = out['occupancy'].get((src, win, value, grp))
            if t is None or not len(t):
                continue
            x = np.arange(len(t))
            ax.vlines(x, t.null_mean, t.null_p95, color=NULL_COLORS['band'], lw=3)
            ax.scatter(x, t.null_mean, color=NULL_COLORS['null'], s=10, label='visit null')
            ax.scatter(x, t.observed, color=NULL_COLORS['observed'], s=12, label='observed')
            ax.set_xticks(x, t.mouse, rotation=90 if len(t) > 6 else 0)
            ax.set_title(f'{out["region"].upper()} {src} {win} {grp}')
            lsm._clean(ax)
        axes[0].set_ylabel(value)
        axes[0].legend(frameon=False, fontsize=6)
    lsm._save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def analyse(region, n_null=1000, n_perm=2000, io_dir=None, run_dir=None, seed=0):
    """Both sources, both lag windows -> dict of tables; nothing written."""
    out = {'region': region}
    runs, config = load_v5(run_dir or V5_RUNS[region])
    ut = v5_unit_table(runs=runs, config=config)
    vf = v5_fold_anchors(runs, config, region, io_dir)
    vn = neuron_scores(vf).merge(
        ut[['recday', 'neuron', 'state_tuned', 'selected_eg', 'selected_strict_eg', 'peak_lag']],
        on=['recday', 'neuron'], how='left')
    vn['pool'] = vn.state_tuned.fillna(False).astype(bool)
    ef, eu = elgaby_split_anchors(region, io_dir)
    en = neuron_scores(ef).merge(eu, on=['recday', 'mouse', 'neuron'], how='left')
    en['pool'] = (en.state_tuned & en.anchored).fillna(False).astype(bool)
    out['folds'] = {'v5': vf, 'elgaby': ef}
    out['neurons'] = {'v5': vn, 'elgaby': en}
    lag_cols = {'v5': [('30deg', 'selected_eg'), ('90deg', 'selected_strict_eg')],
                'elgaby': [('30deg', 'lagged30'), ('90deg', 'lagged90')]}
    out['contrast'], out['occupancy'], out['hist'] = {}, {}, {}
    for src, nd in (('v5', vn), ('elgaby', en)):
        nd[[c for _, c in lag_cols[src]]] = nd[[c for _, c in lag_cols[src]]].fillna(False)
        for win, col in lag_cols[src]:
            for value in ('reward_frac', 'goal_frac'):
                out['contrast'][(src, win, value)] = contrast(nd, value, col, n_perm=n_perm,
                                                              seed=seed)
            lagged = nd[nd.pool & nd[col].astype(bool)]
            other = nd[nd.pool & ~nd[col].astype(bool)]
            k_l = pd.MultiIndex.from_frame(lagged[['recday', 'neuron']])
            k_o = pd.MultiIndex.from_frame(other[['recday', 'neuron']])
            f = out['folds'][src]
            for value in ('reward_frac', 'goal_frac'):
                out['occupancy'][(src, win, value, 'lagged')] = occupancy_null(
                    f, k_l, value, n_null, seed)
                out['occupancy'][(src, win, value, 'other')] = occupancy_null(
                    f, k_o, value, n_null, seed)
            out['hist'][(src, win, 'lagged')] = anchor_histogram(f, k_l)
            out['hist'][(src, win, 'other')] = anchor_histogram(f, k_o)
    out['hist']['occupancy'] = occupancy_histogram(region, sorted(runs), io_dir)
    out['config'] = {'v5_run': run_dir or V5_RUNS[region], 'n_null': n_null, 'n_perm': n_perm,
                     'seed': seed}
    return out


def summary_lines(out):
    """Plain-text tables for the log and the write-up."""
    lines = [f'== {out["region"].upper()} anchor distribution']
    for src in ('v5', 'elgaby'):
        nd = out['neurons'][src]
        lines.append(f'-- {src}: {int(nd.pool.sum())} pool neurons '
                     f'({"state-tuned" if src == "v5" else "state-tuned and anchored"}) '
                     f'over {nd.recday.nunique()} recdays')
    lines.append('contrast: lagged minus other, per-mouse mean; p = within-recday label shuffle')
    for (src, win, value), r in out['contrast'].items():
        lines.append(f'  {src:6s} {win:5s} {value:11s} n {r["n_lagged"]:4d}/{r["n_other"]:4d}  '
                     f'diff {r["observed"]:+.4f}  p {r["p_perm"]:.4f}')
    lines.append('occupancy null: observed per-mouse mean vs anchors redrawn by visits')
    for (src, win, value, grp), t in out['occupancy'].items():
        if not len(t):
            continue
        lines.append(f'  {src:6s} {win:5s} {value:11s} {grp:7s} '
                     + '  '.join(f'{m} {o:.3f}/{nm:.3f} p{p:.3f}' for m, o, nm, p in
                                 zip(t.mouse, t.observed, t.null_mean, t.p)))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', choices=['lec', 'pfc'], required=True)
    ap.add_argument('--n-null', type=int, default=1000)
    ap.add_argument('--n-perm', type=int, default=2000)
    args = ap.parse_args()
    out = analyse(args.region, args.n_null, args.n_perm)
    od = lsm.OUT_DIRS[args.region]
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, 'anchor_distribution.pkl'), 'wb') as fh:
        pickle.dump(out, fh)
    txt = '\n'.join(summary_lines(out))
    print(txt)
    with open(os.path.join(od, 'anchor_distribution_summary.txt'), 'w') as fh:
        fh.write(txt + '\n')
    print(f'-> {od}')


if __name__ == '__main__':
    main()
