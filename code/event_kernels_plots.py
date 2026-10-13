"""Figures F1-F14 and their tables for the event-kernel M2 report, from the merged pickles.

Inputs are what `run_event_kernels.py --merge` and `run_population_drift.py --merge` write
(`load_latest` finds the highest `.vN`). Every summary is per mouse: recday mean first, then the
mouse mean, as `anatomy_split`. Each `plot_*` returns a Figure and saves PDF + PNG when given
`out_path` (a path without extension); each `table_*` returns the numbers its figure draws.
"Locked" is Stage 1's call: Freedman-Lane p < 0.05 and held-out cpd > 0.

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import glob
import os
import pickle
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE if os.path.isdir(os.path.join(HERE, 'mFC_data')) else os.path.dirname(HERE)
while not (os.path.isdir(os.path.join(REPO, 'code')) and os.path.isdir(os.path.join(REPO, 'mFC_data'))):
    REPO = os.path.dirname(REPO)
STAGE1_DIR = os.path.join(REPO, 'data', 'event_kernels', 'stage1')
DRIFT_DIR = os.path.join(REPO, 'data', 'population_drift')
FIG_DIR = os.path.join(REPO, 'data', 'figures', 'event_kernels', 'M2')
DRIFT_FIG_DIR = os.path.join(REPO, 'data', 'figures', 'population_drift', 'M2')
_ARIAL = '/ceph/behrens/max_kirkby/goal_sequencing_1/font/Arial.ttf'

# GridMaze palettes (gridmaze-colors): Viva Magenta / Classic Blue for the two datasets; regions
# and blocks drawn from the named swatches; Stone for nulls, Caviar for ink.
DATASET_COLORS = {'LEC': '#BE3455', 'PFC': '#0F4C81'}
REGION_COLORS = {'ENTl-sup': '#FFA500', 'ENTl-deep': '#9B1B30', 'ENTm': '#6667AB', 'SUB/ProS': '#009473',
                 'CA1/HPF': '#00A6A6', 'PFC': '#0F4C81'}
REGION_ORDER = ['ENTl-sup', 'ENTl-deep', 'ENTm', 'SUB/ProS', 'CA1/HPF']
BLOCK_COLORS = {'ev_entry': '#6B3FA0', 'ev_first_A': '#6667AB', 'ev_A': '#FFA500', 'ev_reward': '#009473',
                'ev_A+reward': '#00A6A6', 'kin_fast': '#888780', 'kin_slow': '#B4B2A9', 'place': '#2C2C2A'}
NULL, INK = '#B4B2A9', '#2C2C2A'
BLOCKS = ['ev_entry', 'ev_first_A', 'ev_A', 'ev_reward', 'ev_A+reward', 'kin_fast', 'kin_slow', 'place']
EVENT_BLOCKS = ['ev_entry', 'ev_A', 'ev_reward', 'ev_A+reward']
LABEL = {'ev_entry': 'entry', 'ev_first_A': 'first A', 'ev_A': 'reward A', 'ev_reward': 'reward',
         'ev_A+reward': 'A + reward', 'kin_fast': 'speed bank, fast', 'kin_slow': 'speed bank, slow',
         'place': 'place', 'entry': 'entry', 'A': 'reward A', 'reward': 'reward', 'first_A': 'first A',
         'sync': 'synchrony onset'}
SPECIAL_RECDAYS = {'ly05_20250618_20250619': '1 session at 10 trials: skipped',
                   'ah08_20250624_20250625': '2 folds',
                   'ly05_20250620_20250623': 'v1 clock-fault recday'}


# ---------------------------------------------------------------------------
# I/O and style
# ---------------------------------------------------------------------------

def load_latest(stem: str):
    """Highest-version `stem.pkl` / `stem.vN.pkl`."""
    hits = [p for p in glob.glob(stem + '*.pkl') if re.fullmatch(re.escape(stem) + r'(\.v\d+)?\.pkl', p)]
    if not hits:
        raise FileNotFoundError(stem + '.pkl')
    p = max(hits, key=lambda q: int(m.group(1)) if (m := re.search(r'\.v(\d+)\.pkl$', q)) else 0)
    with open(p, 'rb') as fh:
        out = pickle.load(fh)
    out['_path'] = p
    return out


def _rc() -> dict:
    rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8, 'figure.titlesize': 8, 'axes.titlesize': 8,
          'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6,
          'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
          'savefig.bbox': None, 'savefig.pad_inches': 0.0}
    try:
        from matplotlib import font_manager
        if os.path.exists(_ARIAL):
            font_manager.fontManager.addfont(_ARIAL)
            rc['font.family'] = 'Arial'
    except Exception:                                    # noqa: BLE001 - the font is cosmetic
        pass
    return rc


def _save(fig, out_path):
    if not out_path:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with mpl.rc_context(_rc()):
        for ext in ('pdf', 'png'):
            fig.savefig(f'{out_path}.{ext}', bbox_inches=None, dpi=300 if ext == 'png' else None)


def alpha_null(n_perm: int) -> float:
    """Exact null rate of p < 0.05 for p = (1 + #null >= obs) / (1 + n_perm): 5/101 at 100."""
    return float((np.ceil(0.05 * (n_perm + 1)) - 1) / (n_perm + 1))


def _jitter(n, width=0.3):
    return np.linspace(-width / 2, width / 2, n) if n > 1 else np.zeros(n)


def per_mouse(d: pd.DataFrame, value: str) -> pd.Series:
    """Recday mean, then mouse mean."""
    return d.groupby(['mouse', 'recday'])[value].mean().groupby('mouse').mean()


def _rows(M: dict, block: str) -> pd.DataFrame:
    """Unit rows of one block; first A from its own anchor's fit, everything else from `entry`."""
    T = M['table']
    anchor = 'first_A' if block in ('ev_first_A', 'ev_first_A+kin_slow') else 'entry'
    d = T[(T['anchor'] == anchor) & (T['block'] == block)].copy()
    d['locked_f'] = d['locked'].astype(float)
    return d


def _symlog(ax, axis='x', linthresh=1e-4):
    (ax.set_xscale if axis == 'x' else ax.set_yscale)('symlog', linthresh=linthresh)


def table_run_summary(stage1: dict) -> pd.DataFrame:
    """Per dataset and anchor: recdays fitted / skipped, units, sessions, rank, held-out r2."""
    rows = []
    for ds, M in stage1.items():
        for a, s in M['summary'].groupby('anchor'):
            f = s[s['skipped'].isna()]
            rows.append(dict(dataset=ds, anchor=a, recdays=len(s), skipped=int(s['skipped'].notna().sum()),
                             units=int(f['n_units'].sum()), sessions=int(f['n_sessions'].sum()),
                             # V3 shards: no held-out leak through a tested column; V2 (M2): only exact duplicates
                             rank_ok=bool(f['rank_benign'].eq(True).all()),
                             r2_cv_median_of_recdays=float(f['r2_cv_median'].median()),
                             reset_skipped=int(((f['reset'] != 'ok') & (a == 'entry')).sum()),
                             n_perm=M['config']['n_perm'], kin=M['config']['kin'],
                             min_trials=M['config']['min_trials']))
    return pd.DataFrame(rows)


def table_entry_peak_bins(stage1: dict) -> pd.DataFrame:
    """Per mouse, of the entry-locked units: the fraction whose kernel's max |beta| is in the
    overflow bin (> 25 min after recording start, mostly the post-task segment) or in the first
    minute; a slow ramp should put it in neither."""
    rows = []
    for ds, M in stage1.items():
        d = _rows(M, 'ev_entry')
        d = d[d['locked']]
        for rd, g in d.groupby('recday'):
            k = M['shards']['entry'][rd]['kernels']['entry']
            B = np.abs(k['beta_hz'][g['unit'].to_numpy(), :-1].astype(float))
            ok = np.isfinite(B).any(axis=1)
            j = np.full(len(B), -1)
            j[ok] = np.nanargmax(np.where(np.isfinite(B[ok]), B[ok], -np.inf), axis=1)
            first_min = sum(e < 60 for e in k['edges_s'][:-1])
            rows += [dict(dataset=ds, recday=rd, mouse=rd.split('_')[0], unit=int(u), peak_bin=int(b),
                          overflow=b == len(k['labels']) - 2, first_minute=0 <= b < first_min)
                     for u, b in zip(g['unit'], j)]
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t['overflow_f'], t['first_minute_f'] = t['overflow'].astype(float), t['first_minute'].astype(float)
    return pd.concat([per_mouse(t[t['dataset'] == ds], ['overflow_f', 'first_minute_f']).assign(dataset=ds)
                      for ds in t['dataset'].unique()]).reset_index()


# ---------------------------------------------------------------------------
# F1-F3: how many units, how much variance, and against kinematics
# ---------------------------------------------------------------------------

def table_fraction_locked(stage1: dict, blocks=BLOCKS) -> pd.DataFrame:
    """Per dataset (LEC unfiltered, LEC with El-Gaby's 20% rule, PFC), block and mouse."""
    rows = []
    for ds, M in stage1.items():
        variants = [(ds, False)] + ([(f'{ds}, 20% rule', True)] if ds == 'LEC' else [])
        for name, rule in variants:
            for b in blocks:
                d = _rows(M, b)
                if rule:
                    d = d[~d['elgaby_drop'].astype(bool)]
                for mouse, v in per_mouse(d, 'locked_f').items():
                    dm = d[d['mouse'] == mouse]
                    rows.append(dict(dataset=name, block=b, mouse=mouse, fraction=v, n_units=len(dm),
                                     n_recdays=dm['recday'].nunique()))
    return pd.DataFrame(rows)


def plot_fraction_locked(stage1: dict, n_perm: int = 100, out_path=None):
    """F1: fraction of units locked per block; a point per mouse, bar = mean of mice."""
    t = table_fraction_locked(stage1)
    names = list(dict.fromkeys(t['dataset']))
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2), layout='constrained')
        for ax, b in zip(axes.flat, BLOCKS):
            for i, name in enumerate(names):
                d = t[(t['dataset'] == name) & (t['block'] == b)].sort_values('mouse')
                c = DATASET_COLORS[name.split(',')[0]]
                ax.bar(i, d['fraction'].mean(), width=0.65, color=c, alpha=0.25 if 'rule' in name else 0.4, lw=0)
                ax.scatter(i + _jitter(len(d)), d['fraction'], s=9, color=c, zorder=3, lw=0)
            ax.axhline(alpha_null(n_perm), ls='--', lw=0.8, color=NULL)
            ax.set_xticks(range(len(names)), [n.replace(', ', '\n') for n in names])
            ax.set_title(LABEL[b])
            ax.set_ylim(0, 1)
        for ax in axes[:, 0]:
            ax.set_ylabel('fraction of units locked')
        n = t.groupby('dataset').agg(mice=('mouse', 'nunique'))
        fig.suptitle('F1. Units locked per block (p < 0.05 and held-out cpd > 0); point = mouse, '
                     f'dashed = exact null rate {alpha_null(n_perm):.3f}; mice: '
                     + ', '.join(f'{k} {v}' for k, v in n['mice'].items()))
    _save(fig, out_path)
    return fig


def plot_cpd_ecdf(stage1: dict, out_path=None):
    """F2: held-out cpd distribution per block, pooled (thick) and per mouse (thin)."""
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), layout='constrained', sharey=True)
        for ax, b in zip(axes.flat, BLOCKS):
            hi = 1e-3
            for ds, M in stage1.items():
                d = _rows(M, b)
                c = DATASET_COLORS[ds]
                for _, dm in d.groupby('mouse'):
                    v = np.sort(dm['cpd'].dropna().to_numpy())
                    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=0.5, alpha=0.35)
                v = np.sort(d['cpd'].dropna().to_numpy())
                if len(v):
                    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=1.4, label=f'{ds} ({len(v)} units)')
                    hi = max(hi, np.quantile(v, 0.995))
            _symlog(ax)
            ax.set_xlim(-1e-3, hi)
            ax.axvline(0, color=NULL, lw=0.6)
            ax.set_title(LABEL[b])
            ax.legend(frameon=False, loc='lower right')
        for ax in axes[1]:
            ax.set_xlabel('held-out cpd')
        for ax in axes[:, 0]:
            ax.set_ylabel('fraction of units')
        fig.suptitle('F2. Held-out cpd per block: all units pooled (thick), one thin line per mouse')
    _save(fig, out_path)
    return fig


def _wide(M: dict, anchor: str = 'entry', value: str = 'cpd') -> pd.DataFrame:
    T = M['table']
    return T[T['anchor'] == anchor].pivot_table(index=['recday', 'mouse', 'unit'], columns='block',
                                                 values=value, aggfunc='first')


def plot_event_vs_kin(stage1: dict, out_path=None):
    """F3: event cpd against the fast speed bank's cpd, for units locked to the event."""
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), layout='constrained')
        for ax, b in zip(axes, EVENT_BLOCKS):
            txt = []
            for ds, M in stage1.items():
                cpd, lk = _wide(M), _wide(M, value='locked')
                sel = lk[b].fillna(False).astype(bool)
                x, y = cpd.loc[sel, 'kin_fast'], cpd.loc[sel, b]
                ax.scatter(x, y, s=3, color=DATASET_COLORS[ds], alpha=0.6, lw=0, label=ds)
                txt.append(f'{ds}: {sel.sum()} locked, {np.mean(y > x):.2f} event > kin')
            lim = [-1e-3, 0.3]
            ax.plot(lim, lim, color=NULL, lw=0.6)
            _symlog(ax, 'x')
            _symlog(ax, 'y')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_title(LABEL[b] + '\n' + '\n'.join(txt), fontsize=6)
            ax.set_xlabel('cpd, speed bank (fast)')
        axes[0].set_ylabel('cpd, event block')
        axes[0].legend(frameon=False, loc='upper left')
        fig.suptitle('F3. Locked units: event cpd vs fast kinematic cpd (identity line grey)')
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# F4-F5: kernel shapes
# ---------------------------------------------------------------------------

def _tick_label(v):
    return f'{v / 60:g}m' if v >= 60 and v % 60 == 0 else f'{v:g}'


def kernel_rows(M: dict, event: str, sel: pd.DataFrame, anchor: str = 'entry'):
    """(betas, raw rates, labels) for the units in `sel` (recday, unit); the `pre` column dropped."""
    B, R, labels = [], [], None
    for rd, g in sel.groupby('recday'):
        k = M['shards'][anchor][rd]['kernels'][event]
        labels = k['labels'][:-1]
        idx = g['unit'].to_numpy()
        B.append(k['beta_hz'][idx, :-1])
        R.append(k['raw_hz'][idx, :-1])
    if not B:
        return np.zeros((0, 0)), np.zeros((0, 0)), []
    return np.vstack(B).astype(float), np.vstack(R).astype(float), labels


def _peak_order(B):
    """Sign-aware peak: positive-peaked rows first, each group sorted by the lag of max |beta|."""
    with np.errstate(invalid='ignore', divide='ignore'):
        Z = B / np.nanmax(np.abs(B), axis=1, keepdims=True)
    k = np.nanargmax(np.abs(np.nan_to_num(Z)), axis=1)
    sign = np.sign(Z[np.arange(len(Z)), k])
    return np.lexsort((k, sign < 0)), Z


def plot_kernel_heatmaps(M: dict, title: str, events=('entry', 'A', 'reward'), groups=None, out_path=None):
    """F4: locked units' shape-free kernels (beta / max|beta|, RdBu_r) sorted by peak lag, and their
    raw rate per lag bin (rate / max, viridis) in the same row order."""
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2 * len(events), figsize=(7.2, 4.0), layout='constrained')
        for j, e in enumerate(events):
            d = _rows(M, f'ev_{e}')
            if groups is not None:
                d = d[d['group'].isin(groups) & ~d['near_boundary'].astype(bool)]
            d = d[d['locked']]
            B, R, labels = kernel_rows(M, e, d)
            axk, axr = axes[2 * j], axes[2 * j + 1]
            if not len(B):
                axk.set_title(f'{LABEL[e]}: no locked units')
                axk.axis('off'); axr.axis('off')
                continue
            order, Z = _peak_order(B)
            with np.errstate(invalid='ignore', divide='ignore'):
                Rn = R / np.nanmax(R, axis=1, keepdims=True)
            imk = axk.imshow(Z[order], aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            imr = axr.imshow(Rn[order], aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
            edges = [float(x.split('-')[0]) for x in labels[:-1]]
            ticks = list(range(0, len(edges), max(1, len(edges) // 5))) + [len(labels) - 1]
            for ax in (axk, axr):
                ax.set_xticks(ticks, [_tick_label(edges[t]) if t < len(edges) else 'ovf' for t in ticks],
                              rotation=90)
                ax.set_yticks([])
            axk.set_title(f'{LABEL[e]}: kernel\n{len(B)} units')
            axr.set_title(f'{LABEL[e]}: raw rate')
            axk.set_xlabel('lag (s; m = min)')
        axes[0].set_ylabel('locked units, sorted by peak lag')
        if 'imk' in locals():
            fig.colorbar(imk, ax=list(axes), shrink=0.45, location='right', label='beta / max |beta|')
            fig.colorbar(imr, ax=list(axes), shrink=0.45, location='right', label='rate / max')
        fig.suptitle(f'F4. {title}: kernels relative to the reference lag bin, and raw rates')
    _save(fig, out_path)
    return fig


def _centres(edges):
    c = [(edges[j] + edges[j + 1]) / 2 for j in range(len(edges) - 1)]
    return np.array(c + [edges[-1] * 1.3])


def plot_example_units(stage1: dict, n: int = 3, events=('entry', 'A', 'reward'), out_path=None):
    """F5: the n units with the highest cpd per event and dataset: raw rate (ink) and the model's
    kernel added to the raw rate at the reference bin (colour), against lag."""
    cols = [(ds, i) for ds in stage1 for i in range(n)]
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(len(events), len(cols), figsize=(7.2, 1.55 * len(events)), layout='constrained')
        for r, e in enumerate(events):
            for c, (ds, i) in enumerate(cols):
                ax = axes[r, c]
                d = _rows(stage1[ds], f'ev_{e}')
                d = d[d['locked']].sort_values('cpd', ascending=False)
                if i >= len(d):
                    ax.axis('off')
                    continue
                u = d.iloc[i]
                k = stage1[ds]['shards']['entry'][u['recday']]['kernels'][e]
                x = _centres(k['edges_s'])
                raw, beta = k['raw_hz'][int(u['unit']), :-1], k['beta_hz'][int(u['unit']), :-1]
                ax.plot(x, raw, color=INK, lw=0.8, marker='o', ms=1.5)
                ax.plot(x, raw[k['ref']] + beta, color=DATASET_COLORS[ds], lw=1.0)
                ax.set_xscale('symlog', linthresh=max(k['edges_s'][1], 0.5))
                ax.set_title(f"{ds} {u['recday'][:4]} u{int(u['unit'])} {u['group']}\ncpd {u['cpd']:.3f}", fontsize=6)
                if c == 0:
                    ax.set_ylabel(f'{LABEL[e]}\nrate (Hz)')
                if r == len(events) - 1:
                    ax.set_xlabel('lag (s)')
        fig.suptitle('F5. Top units by cpd: raw rate per lag bin (black) and the partial kernel, the block\'s '
                     'effect beyond the other regressors added to the raw rate at the reference bin (colour); '
                     'last point = overflow', wrap=True)
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# F6-F9: reset test, anchors, threshold, latency
# ---------------------------------------------------------------------------

def table_reset(stage1: dict) -> pd.DataFrame:
    """Per dataset and mouse: of the entry-locked units, the fraction where entry beats day."""
    rows = []
    for ds, M in stage1.items():
        if M['reset'].empty:
            continue
        d = _rows(M, 'ev_entry')[['recday', 'mouse', 'unit', 'locked']].merge(
            M['reset'], on=['recday', 'mouse', 'unit'], how='inner')
        d['entry_beats_day_f'] = d['entry_beats_day'].astype(float)
        for which, sub in (('entry-locked', d[d['locked']]), ('all units', d)):
            for mouse, v in per_mouse(sub, 'entry_beats_day_f').items():
                rows.append(dict(dataset=ds, units=which, mouse=mouse, fraction_entry_beats_day=v,
                                 n_units=int((sub['mouse'] == mouse).sum())))
    return pd.DataFrame(rows)


def plot_reset(stage1: dict, out_path=None):
    """F6: the reset test. Left: log10 of held-out RSS day-only / entry-only for entry-locked units
    (> 0: the entry kernel predicts held-out sessions better than a day-long trend). Right: per
    mouse, the fraction of entry-locked units where entry beats day."""
    t = table_reset(stage1)
    with mpl.rc_context(_rc()):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.4), layout='constrained')
        notes = []
        for i, (ds, M) in enumerate(stage1.items()):
            c = DATASET_COLORS[ds]
            if not M['reset'].empty:
                d = _rows(M, 'ev_entry').merge(M['reset'], on=['recday', 'mouse', 'unit'])
                d = d[d['locked']]
                lr = np.clip(np.log10(d['rss_day_only'] / d['rss_entry_only']), -0.02, 0.02)   # tails drawn in the edge bins
                a1.hist(lr, bins=np.linspace(-0.02, 0.02, 41), histtype='step', color=c, lw=1, label=f'{ds} ({len(d)})')
                tm = t[(t['dataset'] == ds) & (t['units'] == 'entry-locked')]
                a2.bar(i, tm['fraction_entry_beats_day'].mean(), color=c, alpha=0.4, width=0.6, lw=0)
                a2.scatter(i + _jitter(len(tm)), tm['fraction_entry_beats_day'], s=9, color=c, lw=0, zorder=3)
            s = M['summary']
            sk = s[(s['anchor'] == 'entry') & s['skipped'].isna() & (s['reset'] != 'ok')]
            if len(sk):
                notes.append(f"{ds}: reset skipped on {len(sk)} recdays (no wall clock)")
        a1.axvline(0, color=NULL, lw=0.6)
        a1.set_xlabel('log10 RSS day-only / entry-only\n(clipped at +/-0.02)')
        a1.set_ylabel('entry-locked units')
        a1.legend(frameon=False)
        a2.axhline(0.5, ls='--', color=NULL, lw=0.6)
        a2.set_xticks(range(len(stage1)), list(stage1))
        a2.set_ylim(0, 1)
        a2.set_ylabel('fraction where entry beats day')
        fig.suptitle('F6. Reset test (uncentred, held out): entry kernel vs a day-long trend'
                     + ('\n' + '; '.join(notes) if notes else ''))
    _save(fig, out_path)
    return fig


def table_anchor_agreement(stage1: dict) -> pd.DataFrame:
    rows = []
    for ds, M in stage1.items():
        a = _rows(M, 'ev_entry')[['recday', 'mouse', 'unit', 'cpd', 'locked']]
        b = _rows(M, 'ev_first_A')[['recday', 'unit', 'cpd', 'locked']]
        d = a.merge(b, on=['recday', 'unit'], suffixes=('_entry', '_firstA'))
        for mouse, dm in d.groupby('mouse'):
            rows.append(dict(dataset=ds, mouse=mouse, n_units=len(dm),
                             both=int((dm.locked_entry & dm.locked_firstA).sum()),
                             entry_only=int((dm.locked_entry & ~dm.locked_firstA).sum()),
                             first_A_only=int((~dm.locked_entry & dm.locked_firstA).sum()),
                             agreement=float(np.mean(dm.locked_entry == dm.locked_firstA))))
    return pd.DataFrame(rows)


def plot_anchor_agreement(stage1: dict, out_path=None):
    """F7: entry vs first-A anchor on the same rows: cpd per unit and agreement of the calls."""
    cats = {'both': '#BE3455', 'entry only': '#FFA500', 'first A only': '#0F4C81', 'neither': NULL}
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(stage1), figsize=(2.6 * len(stage1), 2.5), layout='constrained')
        for ax, (ds, M) in zip(np.atleast_1d(axes), stage1.items()):
            a = _rows(M, 'ev_entry')[['recday', 'unit', 'cpd', 'locked']]
            b = _rows(M, 'ev_first_A')[['recday', 'unit', 'cpd', 'locked']]
            d = a.merge(b, on=['recday', 'unit'], suffixes=('_e', '_f'))
            cat = np.where(d.locked_e & d.locked_f, 'both', np.where(d.locked_e, 'entry only',
                           np.where(d.locked_f, 'first A only', 'neither')))
            for k, c in cats.items():
                m = cat == k
                ax.scatter(d.loc[m, 'cpd_e'], d.loc[m, 'cpd_f'], s=3, color=c, lw=0, label=f'{k} ({m.sum()})',
                           zorder=1 if k == 'neither' else 2)
            lim = [-1e-3, max(0.01, float(np.nanmax(d[['cpd_e', 'cpd_f']].to_numpy())))]
            ax.plot(lim, lim, color=INK, lw=0.5)
            _symlog(ax, 'x'); _symlog(ax, 'y')
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel('cpd, entry kernel')
            ax.set_ylabel('cpd, first-A kernel')
            ax.set_title(f'{ds}: calls agree in {np.mean(d.locked_e == d.locked_f):.2f} of {len(d)} units')
            ax.legend(frameon=False, loc='upper left', markerscale=2)
        fig.suptitle('F7. Anchor swap on identical rows: recording start vs the first A')
    _save(fig, out_path)
    return fig


def table_threshold(M10: dict, M5: dict, blocks=EVENT_BLOCKS) -> pd.DataFrame:
    """Per block: unit-level agreement of the call at 10 vs 5 trials on shared recdays, per mouse."""
    rows = []
    for b in blocks:
        d = _rows(M10, b)[['recday', 'mouse', 'unit', 'cpd', 'locked']].merge(
            _rows(M5, b)[['recday', 'unit', 'cpd', 'locked']], on=['recday', 'unit'], suffixes=('_10', '_5'))
        for mouse, dm in d.groupby('mouse'):
            rows.append(dict(block=b, mouse=mouse, n_recdays=dm['recday'].nunique(), n_units=len(dm),
                             fraction_10=per_mouse(dm.assign(v=dm.locked_10.astype(float)), 'v').iloc[0],
                             fraction_5=per_mouse(dm.assign(v=dm.locked_5.astype(float)), 'v').iloc[0],
                             agreement=float(np.mean(dm.locked_10 == dm.locked_5))))
    return pd.DataFrame(rows)


def plot_threshold_sensitivity(M10: dict, M5: dict, out_path=None):
    """F8: LEC at 10 vs 5 trials per session. Top: cpd per unit (shared recdays). Bottom: per-mouse
    fraction locked at each threshold, ly05 highlighted."""
    t = table_threshold(M10, M5)
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(2, len(EVENT_BLOCKS), figsize=(7.2, 4.0), layout='constrained')
        for j, b in enumerate(EVENT_BLOCKS):
            d = _rows(M10, b)[['recday', 'unit', 'cpd']].merge(_rows(M5, b)[['recday', 'unit', 'cpd']],
                                                               on=['recday', 'unit'], suffixes=('_10', '_5'))
            ax = axes[0, j]
            ax.scatter(d['cpd_10'], d['cpd_5'], s=3, color=BLOCK_COLORS[b], lw=0)
            lim = [-1e-3, max(0.01, float(np.nanmax(d[['cpd_10', 'cpd_5']].to_numpy())))]
            ax.plot(lim, lim, color=INK, lw=0.5)
            _symlog(ax, 'x'); _symlog(ax, 'y')
            ax.set_xlim(lim); ax.set_ylim(lim)
            tb = t[t['block'] == b]
            ax.set_title(f"{LABEL[b]}: same call in {np.average(tb['agreement'], weights=tb['n_units']):.2f}")
            ax.set_xlabel('cpd, >= 10 trials')
            ax = axes[1, j]
            for _, r in tb.iterrows():
                hl = r['mouse'] == 'ly05'
                ax.plot([0, 1], [r['fraction_10'], r['fraction_5']], color=DATASET_COLORS['LEC'] if hl else NULL,
                        lw=1.2 if hl else 0.8, marker='o', ms=3)
                ax.text(1.05, r['fraction_5'], r['mouse'], fontsize=5, va='center')
            ax.set_xticks([0, 1], ['>= 10', '>= 5'])
            ax.set_xlim(-0.2, 1.4)
            ax.set_ylim(0, 1)
            ax.set_xlabel('trials per session')
        axes[0, 0].set_ylabel('cpd, >= 5 trials')
        axes[1, 0].set_ylabel('fraction locked (per mouse)')
        fig.suptitle('F8. LEC threshold sensitivity on the shared recdays (ly05 in colour)')
    _save(fig, out_path)
    return fig


def table_reward_latency(stage1: dict) -> pd.DataFrame:
    """Per reward-locked unit: lag-bin centre of the kernel's maximum (overflow and pre excluded)."""
    rows = []
    for ds, M in stage1.items():
        d = _rows(M, 'ev_reward')
        d = d[d['locked']]
        for rd, g in d.groupby('recday'):
            k = M['shards']['entry'][rd]['kernels']['reward']
            cen = _centres(k['edges_s'])[:-1]
            B = k['beta_hz'][g['unit'].to_numpy(), :len(cen)].astype(float)
            ok = np.isfinite(B).any(axis=1)
            pk = np.full(len(B), np.nan)
            pk[ok] = cen[np.nanargmax(np.where(np.isfinite(B[ok]), B[ok], -np.inf), axis=1)]
            rows += [dict(dataset=ds, recday=rd, mouse=rd.split('_')[0], unit=int(u), peak_s=p)
                     for u, p in zip(g['unit'], pk)]
    return pd.DataFrame(rows)


def plot_reward_latency(stage1: dict, out_path=None):
    """F9: peak lag of the reward kernel per reward-locked unit, one strip per mouse (ah08 flagged:
    its speed reference sat at +0.25 to +0.40 s against -0.30 s elsewhere). Bins are 0.5 s wide up
    to 2 s, so a shift smaller than a bin moves units between bins rather than the median."""
    t = table_reward_latency(stage1)
    mice = [(ds, m) for ds in stage1 for m in sorted(t.loc[t['dataset'] == ds, 'mouse'].unique())]
    rng = np.random.default_rng(0)
    with mpl.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(5.0, 2.6), layout='constrained')
        for i, (ds, m) in enumerate(mice):
            v = t[(t['dataset'] == ds) & (t['mouse'] == m)]['peak_s'].dropna().to_numpy()
            ax.scatter(i + rng.uniform(-0.25, 0.25, len(v)), v * np.exp(rng.uniform(-0.15, 0.15, len(v))),
                       s=3, color=DATASET_COLORS[ds], alpha=0.6, lw=0)
            if len(v):
                ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=INK, lw=1.2)
        ax.set_yscale('log')
        ax.set_xticks(range(len(mice)), [m + (' *' if m == 'ah08' else '') for _, m in mice], rotation=90)
        ax.set_ylabel('reward kernel peak (s after reward)')
        fig.suptitle('F9. Reward-kernel latency per mouse (bar = median; * ah08, speed-reference offset)')
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# F10-F11: regions and per-recday QC
# ---------------------------------------------------------------------------

def _asp():
    sys.path.insert(0, os.path.join(REPO, 'code'))
    import anatomy_split as asp
    return asp


def _within_recday_null(rm, value, a, b, n_perm, seed):
    """`anatomy_split.within_recday_permutation`'s loop, keeping the null: (observed, null, p)."""
    asp = _asp()
    rng = np.random.default_rng(seed)
    labels = rm['group'].to_numpy().copy()
    idx = [np.flatnonzero(rm['recday'].to_numpy() == rd) for rd in rm['recday'].unique()]
    obs = asp._contrast_from_labels(rm, value, labels, a, b, 'mean', 1)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = labels.copy()
        for ix in idx:
            perm[ix] = rng.permutation(perm[ix])
        null[i] = asp._contrast_from_labels(rm, value, perm, a, b, 'mean', 1)
    fin = null[np.isfinite(null)]
    return obs, null, float((1 + np.sum(np.abs(fin) >= abs(obs))) / (1 + len(fin)))


def region_contrast(M_lec: dict, block: str, value: str = 'locked_f', a='ENTl-deep', b='SUB/ProS',
                    n_perm: int = 10000, seed: int = 0) -> dict:
    """Rate-matched, off-boundary contrast a - b at mouse level on its within-recday permutation
    null (`anatomy_split.rate_match`, then its permutation loop kept to draw the null; checked
    against `within_recday_permutation` at 200 permutations, same seed)."""
    asp = _asp()
    d = _rows(M_lec, block)
    d = d[~d['near_boundary'].astype(bool) & d['group'].isin([a, b])].reset_index(drop=True)
    rm = asp.rate_match(d, 'rate_hz', a, b, seed=seed).reset_index(drop=True)
    ref = asp.within_recday_permutation(rm, value, a, b, n_perm=200, seed=seed)
    o200, _, p200 = _within_recday_null(rm, value, a, b, 200, seed)
    if not (np.isclose(o200, ref['observed'], equal_nan=True) and np.isclose(p200, ref['p_perm'])):
        raise AssertionError('replicated permutation differs from anatomy_split.within_recday_permutation')
    obs, null, p = _within_recday_null(rm, value, a, b, n_perm, seed)
    pm = rm.groupby(['mouse', 'recday', 'group'])[value].mean().groupby(['mouse', 'group']).mean().unstack()
    return dict(contrast=f'{a} - {b}', observed=obs, p_perm=p, n_perm=n_perm, null=null, n_units=len(rm),
                mice=sorted(pm.dropna().index), per_mouse=pm)


def table_region_contrasts(M_lec: dict, blocks=EVENT_BLOCKS, n_perm: int = 10000) -> pd.DataFrame:
    """ENTl-deep - SUB/ProS for each block, on the fraction locked and on cpd."""
    rows = []
    for b in blocks:
        for value in ('locked_f', 'cpd'):
            c = region_contrast(M_lec, b, value=value, n_perm=n_perm)
            pm = c['per_mouse'].dropna()
            rows.append(dict(block=b, value='fraction locked' if value == 'locked_f' else 'cpd',
                             observed=c['observed'], p_perm=c['p_perm'], n_mice=len(c['mice']),
                             n_units=c['n_units'],
                             per_mouse=', '.join(f'{m} {r["ENTl-deep"] - r["SUB/ProS"]:+.3f}' for m, r in pm.iterrows())))
    return pd.DataFrame(rows)


def table_regions(M_lec: dict, M_pfc: dict, blocks=EVENT_BLOCKS) -> pd.DataFrame:
    rows = []
    for b in blocks:
        d = _rows(M_lec, b)
        d = d[~d['near_boundary'].astype(bool) & d['group'].isin(REGION_ORDER)]
        for (g, mouse), _ in d.groupby(['group', 'mouse']):
            dm = d[(d['group'] == g) & (d['mouse'] == mouse)]
            rows.append(dict(block=b, group=g, mouse=mouse, fraction=per_mouse(dm, 'locked_f').iloc[0],
                             cpd_mean=per_mouse(dm, 'cpd').iloc[0], n_units=len(dm)))
        p = _rows(M_pfc, b)
        for mouse, v in per_mouse(p, 'locked_f').items():
            rows.append(dict(block=b, group='PFC', mouse=mouse, fraction=v,
                             cpd_mean=per_mouse(p[p['mouse'] == mouse], 'cpd').iloc[0],
                             n_units=int((p['mouse'] == mouse).sum())))
    return pd.DataFrame(rows)


def plot_regions(M_lec: dict, M_pfc: dict, n_perm: int = 10000, out_path=None):
    """F10: top, fraction locked per LEC region (units >= 50 um from a boundary) and PFC, a point
    per mouse; bottom, the rate-matched ENTl-deep vs SUB/ProS difference on its null."""
    t = table_regions(M_lec, M_pfc)
    groups = REGION_ORDER + ['PFC']
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(2, len(EVENT_BLOCKS), figsize=(7.2, 4.4), layout='constrained')
        for j, b in enumerate(EVENT_BLOCKS):
            ax = axes[0, j]
            for i, g in enumerate(groups):
                d = t[(t['block'] == b) & (t['group'] == g)].sort_values('mouse')
                if not len(d):
                    continue
                ax.bar(i, d['fraction'].mean(), color=REGION_COLORS[g], alpha=0.4, width=0.65, lw=0)
                ax.scatter(i + _jitter(len(d)), d['fraction'], s=8, color=REGION_COLORS[g], lw=0, zorder=3)
                ax.text(i, 0.97, str(len(d)), ha='center', fontsize=5)
            ax.set_xticks(range(len(groups)), groups, rotation=90)
            ax.set_ylim(0, 1)
            ax.set_title(LABEL[b])
            c = region_contrast(M_lec, b, n_perm=n_perm)
            ax = axes[1, j]
            ax.hist(c['null'][np.isfinite(c['null'])], bins=40, color=NULL)
            ax.axvline(c['observed'], color=REGION_COLORS['ENTl-deep'], lw=1.2)
            nm = len(c['mice'])
            ax.set_title(f"diff {c['observed']:+.3f}, p {c['p_perm']:.3f}\n"
                         f"{nm} {'mouse' if nm == 1 else 'mice'}, {c['n_units']} units", fontsize=6)
            ax.set_xlabel('ENTl-deep - SUB/ProS')
        axes[0, 0].set_ylabel('fraction locked\n(number = mice)')
        axes[1, 0].set_ylabel('permutations')
        fig.suptitle('F10. Regions: off-boundary units; the contrast is rate-matched, mouse-level, '
                     'labels shuffled within recday')
    _save(fig, out_path)
    return fig


def plot_recday_qc(stage1: dict, out_path=None):
    """F11: per recday, units, analysis sessions, median held-out r2 and the reward-locked fraction."""
    rows = []
    for ds, M in stage1.items():
        s = M['summary'][M['summary']['anchor'] == 'entry'].copy()
        rl = _rows(M, 'ev_reward').groupby('recday')['locked_f'].mean()
        s['reward_locked'] = s['recday'].map(rl)
        s['dataset'] = ds
        rows.append(s)
    s = pd.concat(rows).sort_values(['dataset', 'recday']).reset_index(drop=True)
    x = np.arange(len(s))
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(4, 1, figsize=(7.2, 5.6), layout='constrained', sharex=True)
        for ax, col, lab in zip(axes, ('n_units', 'n_sessions', 'r2_cv_median', 'reward_locked'),
                                ('units', 'analysis sessions', 'median held-out r2', 'reward-locked fraction')):
            ax.bar(x, s[col].fillna(0), color=[DATASET_COLORS[d] for d in s['dataset']], width=0.7, lw=0)
            ax.set_ylabel(lab)
            for i in np.flatnonzero(s['skipped'].notna().to_numpy()):
                ax.text(x[i], 0, 'x', ha='center', va='bottom', fontsize=6, color=INK)
        for i, rd in enumerate(s['recday']):
            if rd in SPECIAL_RECDAYS:
                axes[0].annotate(SPECIAL_RECDAYS[rd], (x[i], s.loc[i, 'n_units']), xytext=(0, 6),
                                 textcoords='offset points', fontsize=5, rotation=90, ha='center')
        axes[-1].set_xticks(x, s['recday'], rotation=90, fontsize=5)
        fig.suptitle('F11. Per recday (x = skipped: fewer than 2 analysis sessions)')
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# F12-F14: population drift (Kanter part 1)
# ---------------------------------------------------------------------------

def _drift_mouse(df: pd.DataFrame, keys: list, value: str) -> pd.DataFrame:
    """Draws and sessions averaged within recday, then recdays within mouse."""
    r = df.groupby(['recday', 'mouse'] + keys)[value].mean().reset_index()
    return r.groupby(['mouse'] + keys)[value].mean().reset_index()


def table_knn(drift: dict) -> pd.DataFrame:
    rows = []
    for ds, P in drift.items():
        k = P['knn'].assign(within_over_chance=lambda x: x.within_acc / x.chance_acc,
                            across_over_chance=lambda x: x.across_acc / x.chance_acc)
        for name, sub in (('all 20', k[k['set'] == 'all_n20']), ('region 20', k[k['set'] == 'region'])):
            if sub.empty:
                continue
            for v in ('within_over_chance', 'across_over_chance', 'within_acc', 'across_acc', 'chance_acc'):
                r = _drift_mouse(sub, ['group'], v)
                r['dataset'], r['set'], r['measure'] = ds, name, v
                rows.append(r.rename(columns={v: 'value'}))
    return pd.concat(rows, ignore_index=True)


def plot_distance_vs_lag(drift: dict, out_path=None):
    """F12: cosine distance between 1-min epochs vs their lag, 20-unit draws: datasets (left) and LEC
    regions (right); thin = mouse, thick = mean of mice."""
    with mpl.rc_context(_rc()):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.5), layout='constrained', sharey=True)
        for ds, P in drift.items():
            d = P['dist'][P['dist']['set'] == 'all_n20']
            if d.empty:
                continue
            m = _drift_mouse(d, ['lag_min'], 'distance')
            for _, g in m.groupby('mouse'):
                a1.plot(g['lag_min'], g['distance'], color=DATASET_COLORS[ds], lw=0.5, alpha=0.5)
            mm = m.groupby('lag_min')['distance'].mean()
            a1.plot(mm.index, mm.values, color=DATASET_COLORS[ds], lw=1.5, label=f"{ds} ({m['mouse'].nunique()} mice)")
        a1.set_xlabel('lag (min)')
        a1.set_ylabel('cosine distance')
        a1.legend(frameon=False)
        a1.set_title('all units, 20-unit draws')
        if 'LEC' in drift:
            d = drift['LEC']['dist'][drift['LEC']['dist']['set'] == 'region']
            m = _drift_mouse(d, ['group', 'lag_min'], 'distance')
            for g in REGION_ORDER:
                mg = m[m['group'] == g]
                if mg.empty:
                    continue
                mm = mg.groupby('lag_min')['distance'].mean()
                a2.plot(mm.index, mm.values, color=REGION_COLORS[g], lw=1.2, label=f"{g} ({mg['mouse'].nunique()})")
        a2.set_xlabel('lag (min)')
        a2.set_title('LEC regions, 20-unit draws')
        a2.legend(frameon=False, title='mice')
        fig.suptitle('F12. Population distance between epochs grows with lag (Kanter)')
    _save(fig, out_path)
    return fig


def plot_knn(drift: dict, out_path=None):
    """F13: 1-min epoch decoding. Top: confusion matrices (rows normalised) within session and on the
    held-out session, 20-unit draws. Bottom: accuracy / chance per mouse."""
    t = table_knn(drift)
    with mpl.rc_context(_rc()):
        fig = plt.figure(figsize=(7.2, 4.6), layout='constrained')
        sf = fig.subfigures(2, 1, height_ratios=[1.1, 1])
        top = sf[0].subplots(1, 2 * len(drift))
        j = 0
        for ds, P in drift.items():
            for which, w in ((0, 'within session'), (1, 'held-out session')):
                C = sum(v[which].astype(float) for k, v in P['confusion'].items() if k[1] == 'all_n20')
                ax = top[j]
                j += 1
                if not isinstance(C, np.ndarray):
                    ax.axis('off')
                    continue
                with np.errstate(invalid='ignore', divide='ignore'):
                    Cn = C / C.sum(axis=1, keepdims=True)
                ax.imshow(Cn, cmap='viridis', vmin=0, vmax=np.nanmax(Cn), origin='lower')
                ax.set_title(f'{ds}, {w}')
                ax.set_xlabel('decoded epoch (min)')
            top[0].set_ylabel('true epoch (min)')
        bot = sf[1].subplots(1, 2, sharey=True)
        for ax, meas in zip(bot, ('within_over_chance', 'across_over_chance')):
            cats = []
            for ds in drift:
                cats.append((ds, 'all 20', 'all', DATASET_COLORS[ds]))
            cats += [('LEC', 'region 20', g, REGION_COLORS[g]) for g in REGION_ORDER]
            for i, (ds, st, g, c) in enumerate(cats):
                d = t[(t['dataset'] == ds) & (t['set'] == st) & (t['group'] == g) & (t['measure'] == meas)]
                if d.empty:
                    continue
                ax.bar(i, d['value'].mean(), color=c, alpha=0.4, width=0.65, lw=0)
                ax.scatter(i + _jitter(len(d)), d['value'], s=8, color=c, lw=0, zorder=3)
            ax.axhline(1, ls='--', color=NULL, lw=0.8)
            ax.set_xticks(range(len(cats)), [f'{ds} all' if g == 'all' else g for ds, _, g, _ in cats], rotation=90)
            ax.set_title('within session' if meas.startswith('within') else 'held-out session (locked clock)')
        bot[0].set_ylabel('accuracy / chance')
        fig.suptitle('F13. Decoding the 1-min epoch: within session (locked + drift) vs held-out session '
                     '(locked only)')
    _save(fig, out_path)
    return fig


def plot_event_acceleration(drift: dict, out_path=None):
    """F14: change in population trajectory speed around events (z against all bins), 20-unit draws;
    thin = mouse, thick = mean of mice. Synchrony onsets are a positive reference, not a test."""
    events = ('first_A', 'A', 'reward', 'sync')
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(events), figsize=(7.2, 2.2), layout='constrained', sharey=True)
        for ax, e in zip(axes, events):
            for ds, P in drift.items():
                a = P['acc'][(P['acc']['set'] == 'all_n20') & (P['acc']['event'] == e)]
                if a.empty:
                    continue
                m = _drift_mouse(a, ['lag_s'], 'z')
                for _, g in m.groupby('mouse'):
                    ax.plot(g['lag_s'], g['z'], color=DATASET_COLORS[ds], lw=0.5, alpha=0.5)
                mm = m.groupby('lag_s')['z'].mean()
                ax.plot(mm.index, mm.values, color=DATASET_COLORS[ds], lw=1.5, label=ds)
            ax.axhline(0, color=NULL, lw=0.6)
            ax.axvline(0, color=NULL, lw=0.6)
            ax.set_title(LABEL[e])
            ax.set_xlabel('lag (s)')
        axes[0].set_ylabel('acceleration z')
        axes[0].legend(frameon=False)
        fig.suptitle('F14. Population trajectory acceleration at events (Kanter boundary shift)')
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# F15-F18: Stage 2 (kernel shapes and time constants; M3 part 1)
# ---------------------------------------------------------------------------

STAGE2_DIR = os.path.join(REPO, 'data', 'event_kernels', 'stage2')
CLASSES = ['ramp_down', 'ramp_up', 'bump', 'other', 'none']
STAGE2_EVENTS = ('entry', 'A', 'reward')
SESSION_S = 1800.0             # M1 envelope: tau >= 30 min is "slower than the session", not a number


def table_stage2_classes(S2: dict) -> pd.DataFrame:
    """Per dataset, event, role (locked / null_sample) and mouse: the fraction of units in each shape
    class (recday mean, then mouse mean)."""
    rows = []
    for ds, M in S2.items():
        T = M['table']
        for (ev, role), d in T.groupby(['event', 'role']):
            for c in CLASSES:
                pm = per_mouse(d.assign(v=(d['shape'] == c).astype(float)), 'v')
                rows += [dict(dataset=ds, event=ev, role=role, shape=c, mouse=m, fraction=v,
                              n_units=int((d['mouse'] == m).sum())) for m, v in pm.items()]
    return pd.DataFrame(rows)


def plot_stage2_classes(S2: dict, out_path=None):
    """F15: shape class of each event's locked units, a point per mouse, bar = mean of mice; the open
    marker is the null sample's mean (non-locked units through the same Stage 2)."""
    t = table_stage2_classes(S2)
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(STAGE2_EVENTS), figsize=(7.2, 2.6), layout='constrained', sharey=True)
        w = 0.8 / max(len(S2), 1)
        for ax, ev in zip(axes, STAGE2_EVENTS):
            for i, ds in enumerate(S2):
                for j, c in enumerate(CLASSES):
                    x = j + (i - (len(S2) - 1) / 2) * w
                    d = t[(t['dataset'] == ds) & (t['event'] == ev) & (t['role'] == 'locked') & (t['shape'] == c)]
                    nl = t[(t['dataset'] == ds) & (t['event'] == ev) & (t['role'] == 'null_sample') & (t['shape'] == c)]
                    if len(d):
                        ax.bar(x, d['fraction'].mean(), width=w * 0.9, color=DATASET_COLORS[ds], alpha=0.4, lw=0)
                        ax.scatter(x + _jitter(len(d), w * 0.6), d['fraction'], s=6, color=DATASET_COLORS[ds], lw=0, zorder=3)
                    if len(nl):
                        ax.scatter(x, nl['fraction'].mean(), s=14, facecolor='none', edgecolor=INK, lw=0.7, zorder=4)
            ax.set_xticks(range(len(CLASSES)), [c.replace('_', ' ') for c in CLASSES], rotation=45)
            ax.set_title(LABEL[ev])
            ax.set_ylim(0, 1)
        axes[0].set_ylabel('fraction of locked units')
        fig.suptitle('F15. Stage 2 shape of each event kernel (a class needs >= 0.5 of the shape-free '
                     'kernel\'s held-out gain); point = mouse, open circle = null sample')
    _save(fig, out_path)
    return fig


def table_tau(S2: dict) -> pd.DataFrame:
    """Ramp-classed locked units: tau (median over folds), at the grid edge, beyond the session."""
    rows = []
    for ds, M in S2.items():
        T = M['table']
        d = T[(T['role'] == 'locked') & T['shape'].isin(['ramp_down', 'ramp_up'])].copy()
        d['dataset'] = ds
        d['beyond_session'] = d['tau_median'] >= SESSION_S
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_tau_spectrum(S2: dict, out_path=None):
    """F16: time constants of ramp-classed units. Top: per dataset (thick = pooled, thin = mouse).
    Bottom: LEC regions, off-boundary units. Shaded: tau >= 30 min, which a 20-min session cannot
    pin (M1: recovered at 0.13-3.43x of truth there)."""
    t = table_tau(S2)
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(2, len(STAGE2_EVENTS), figsize=(7.2, 4.4), layout='constrained')
        for j, ev in enumerate(STAGE2_EVENTS):
            for r, (keys, colours) in enumerate(((list(S2), DATASET_COLORS), (REGION_ORDER, REGION_COLORS))):
                ax = axes[r, j]
                for k in keys:
                    if r == 0:
                        d = t[(t['dataset'] == k) & (t['event'] == ev)]
                    else:
                        d = t[(t['dataset'] == 'LEC') & (t['event'] == ev) & (t['group'] == k) & ~t['near_boundary'].astype(bool)]
                    v = np.sort(d['tau_median'].dropna().to_numpy())
                    if not len(v):
                        continue
                    if r == 0:
                        for _, dm in d.groupby('mouse'):
                            vm = np.sort(dm['tau_median'].dropna().to_numpy())
                            ax.plot(vm, np.arange(1, len(vm) + 1) / len(vm), color=colours[k], lw=0.5, alpha=0.35)
                    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=colours[k], lw=1.4,
                            label=f"{k} {len(v)} ({d['mouse'].nunique()} mice)")
                ax.set_xscale('log')
                ax.axvspan(SESSION_S, ax.get_xlim()[1] if ax.get_xlim()[1] > SESSION_S else SESSION_S * 4, color=NULL, alpha=0.3, lw=0)
                ax.set_title(f'{LABEL[ev]} ramps' + (', LEC regions' if r else ''))
                ax.legend(frameon=False, loc='lower right', fontsize=5)
                ax.set_xlabel('tau (s)')
            axes[0, j].set_ylim(0, 1)
        axes[0, 0].set_ylabel('fraction of ramp units')
        axes[1, 0].set_ylabel('fraction of ramp units')
        fig.suptitle('F16. Time constants of ramp-classed units (ECDF); grey band: tau >= 30 min, '
                     'slower than the session')
    _save(fig, out_path)
    return fig


def plot_bump_peaks(S2: dict, out_path=None):
    """F17: peak lag of bump-classed units per event (gamma bumps, peak grid 2 per octave)."""
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(STAGE2_EVENTS), figsize=(7.2, 2.3), layout='constrained')
        for ax, ev in zip(axes, STAGE2_EVENTS):
            txt = []
            for ds, M in S2.items():
                T = M['table']
                d = T[(T['role'] == 'locked') & (T['event'] == ev) & (T['shape'] == 'bump')]
                v = d['peak_median'].dropna().to_numpy()
                if not len(v):
                    continue
                bins = np.geomspace(max(v.min() * 0.8, 0.1), v.max() * 1.25, 20)
                ax.hist(v, bins=bins, histtype='step', color=DATASET_COLORS[ds], lw=1, label=f'{ds} ({len(v)})')
                if ev == 'entry':
                    txt.append(f'{ds}: {np.mean(v < 60):.2f} peak < 1 min')
            ax.set_xscale('log')
            ax.set_title(LABEL[ev] + ' bumps' + ('\n' + '; '.join(txt) if txt else ''), fontsize=7)
            ax.set_xlabel('peak lag (s)')
            ax.legend(frameon=False, fontsize=5)
        axes[0].set_ylabel('bump units')
        fig.suptitle('F17. Where bump kernels peak')
    _save(fig, out_path)
    return fig


def plot_stage2_examples(S2: dict, S1: dict, ds: str = 'LEC', out_path=None):
    """F18: per event, the locked unit with the largest held-out gain in each of the ramp and bump
    classes: its shape-free kernel (points, Stage 1 betas) and the fitted shape scaled onto it by
    least squares (line), against lag."""
    import event_kernels as ek
    T, M1 = S2[ds]['table'], S1[ds]
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(2, len(STAGE2_EVENTS), figsize=(7.2, 3.8), layout='constrained')
        for j, ev in enumerate(STAGE2_EVENTS):
            for r, cls in enumerate((('ramp_down', 'ramp_up'), ('bump',))):
                ax = axes[r, j]
                d = T[(T['role'] == 'locked') & (T['event'] == ev) & T['shape'].isin(cls)].sort_values('gain_np', ascending=False)
                u = None
                for _, cand in d.iterrows():
                    # A recday whose all-session null space spans the block has NaN betas there
                    # (ly05_20250620_20250623's reward block at >= 5 trials): take the next unit.
                    k = M1['shards']['entry'][cand['recday']]['kernels'][ev]
                    x = _centres(k['edges_s'])[:-1]
                    beta = k['beta_hz'][int(cand['unit']), :len(x)].astype(float)
                    if np.isfinite(beta).sum() >= 5:
                        u = cand
                        break
                if u is None:
                    ax.axis('off')
                    continue
                f = ek.ramp(x, u['tau_median']) if cls[0].startswith('ramp') else ek.bump(x, u['peak_median'], u['k_median'])
                ok = np.isfinite(beta)
                A = np.column_stack([np.ones(ok.sum()), f[ok]])
                coef = np.linalg.lstsq(A, beta[ok], rcond=None)[0]
                ax.plot(x[ok], beta[ok], 'o', ms=2.5, color=INK)
                ax.plot(x, coef[0] + coef[1] * f, color=DATASET_COLORS[ds], lw=1.2)
                ax.set_xscale('log')
                par = f"tau {u['tau_median']:.3g} s" if cls[0].startswith('ramp') else f"peak {u['peak_median']:.3g} s, k {u['k_median']:.0f}"
                ax.set_title(f"{LABEL[ev]}: {u['shape'].replace('_', ' ')}, {par}\n{u['recday'][:4]} u{int(u['unit'])} {u['group']}, "
                             f"ramp {u['frac_ramp']:.2f} / bump {u['frac_bump']:.2f} of the gain", fontsize=6)
                ax.set_xlabel('lag (s)')
        for ax in axes[:, 0]:
            ax.set_ylabel('kernel (Hz)')
        fig.suptitle(f'F18. {ds}: best ramp and bump fits against the shape-free kernel')
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------

def make_all(stage1: dict, stage1_mt5: dict | None, drift: dict, n_perm: int = 100) -> dict:
    """Every figure, saved under FIG_DIR / DRIFT_FIG_DIR; returns {figure id: path stem}."""
    out = {}

    def go(fid, fn, *a, d=FIG_DIR, **kw):
        p = os.path.join(d, fid)
        fig = fn(*a, out_path=p, **kw)
        plt.close(fig)
        out[fid] = p

    go('F1_fraction_locked', plot_fraction_locked, stage1, n_perm=n_perm)
    go('F2_cpd_ecdf', plot_cpd_ecdf, stage1)
    go('F3_event_vs_kin', plot_event_vs_kin, stage1)
    for ds, M in stage1.items():
        go(f'F4_kernels_{ds}', plot_kernel_heatmaps, M, ds)
    if 'LEC' in stage1:
        for g in REGION_ORDER:
            go(f"F4_kernels_LEC_{g.replace('/', '-')}", plot_kernel_heatmaps, stage1['LEC'], f'LEC {g}', groups=[g])
    go('F5_example_units', plot_example_units, stage1)
    go('F6_reset', plot_reset, stage1)
    go('F7_anchor_agreement', plot_anchor_agreement, stage1)
    if stage1_mt5 is not None:
        go('F8_threshold_sensitivity', plot_threshold_sensitivity, stage1['LEC'], stage1_mt5)
    go('F9_reward_latency', plot_reward_latency, stage1)
    if 'LEC' in stage1 and 'PFC' in stage1:
        go('F10_regions', plot_regions, stage1['LEC'], stage1['PFC'])
    go('F11_recday_qc', plot_recday_qc, stage1)
    go('F12_distance_vs_lag', plot_distance_vs_lag, drift, d=DRIFT_FIG_DIR)
    go('F13_knn', plot_knn, drift, d=DRIFT_FIG_DIR)
    go('F14_event_acceleration', plot_event_acceleration, drift, d=DRIFT_FIG_DIR)
    return out
