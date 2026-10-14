"""Figures and tables for the event-kernel reports, from the merged pickles: F1-F14 (M2), F15-F18
(M3 part 1, Stage 2), F19-F21 (entry clock vs leaky reward count), F22-F30 (M3 part 2).

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
# F19-F21: the entry clock vs a leaky reward count (M5 brought forward)
# ---------------------------------------------------------------------------

LEAKY_DIR = os.path.join(REPO, 'data', 'event_kernels', 'leaky')


def latest_leaky_gate(dataset: str) -> pd.DataFrame:
    """Per-unit table of the most recent `--gate-leaky` run for a dataset."""
    runs = sorted(glob.glob(os.path.join(LEAKY_DIR, 'runner_gate', f'{dataset.lower()}_*', 'leaky_gate_units.csv')))
    if not runs:
        raise FileNotFoundError(f'no leaky gate run for {dataset}')
    return pd.read_csv(runs[-1])


def table_leaky(L: dict) -> pd.DataFrame:
    """Per unit: held-out preference and unique gains, in units of the shape-free kernel's gain.
    pref > 0: the leaky reward count predicts held-out sessions better than the entry ramp."""
    rows = []
    for ds, M in L.items():
        T = M['table'].copy()
        g = T['dev_null'] - T['dev_np']
        ok = g > 0
        T['dataset'] = ds
        T['count_better'] = T['dev_count'] < T['dev_ramp']
        T['pref'] = np.where(ok, (T['dev_ramp'] - T['dev_count']) / g, np.nan)
        T['count_unique'] = np.where(ok, (T['dev_ramp'] - T['dev_joint']) / g, np.nan)
        T['ramp_unique'] = np.where(ok, (T['dev_count'] - T['dev_joint']) / g, np.nan)
        T['ramp_like'] = T['shape'].isin(['ramp_up', 'ramp_down']) | ((T['shape'] == 'bump') & (T['frac_ramp'] >= 0.5))
        rows.append(T)
    return pd.concat(rows, ignore_index=True)


def plot_leaky_preference(L: dict, out_path=None):
    """F19: left, the held-out preference (dev_ramp - dev_count) / gain of the shape-free kernel for
    ramp-like entry-locked units and the null sample; right, per mouse, the fraction of
    entry-locked units where the leaky count beats the ramp, with the gate's planted rates."""
    t = table_leaky(L)
    with mpl.rc_context(_rc()):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6), layout='constrained')
        bins = np.linspace(-0.5, 0.5, 41)
        for ds in L:
            d = t[(t['dataset'] == ds) & (t['role'] == 'locked') & t['ramp_like']]['pref'].clip(-0.5, 0.5)
            a1.hist(d.dropna(), bins=bins, histtype='step', color=DATASET_COLORS[ds], lw=1.2, label=f'{ds} ({d.notna().sum()})')
        nl = t[t['role'] == 'null_sample']['pref'].clip(-0.5, 0.5).dropna()
        a1.hist(nl, bins=bins, histtype='step', color=NULL, lw=1, label=f'null ({len(nl)})')
        a1.axvline(0, color=INK, lw=0.6)
        a1.set_xlabel('(dev ramp - dev count) / kernel gain\n> 0: leaky count better (clipped at +/-0.5)')
        a1.set_ylabel('units')
        a1.legend(frameon=False)
        for i, ds in enumerate(L):
            d = t[(t['dataset'] == ds) & (t['role'] == 'locked')].assign(v=lambda x: x['count_better'].astype(float))
            pm = per_mouse(d, 'v')
            a2.bar(i, pm.mean(), color=DATASET_COLORS[ds], alpha=0.4, width=0.6, lw=0)
            a2.scatter(i + _jitter(len(pm)), pm.values, s=9, color=DATASET_COLORS[ds], lw=0, zorder=3)
            try:
                g = latest_leaky_gate(ds)
                lo = g['tau'] <= 600
                for kind, ls in (('leaky_count', '-'), ('entry_ramp', '--')):
                    r = g[lo & (g['kind'] == kind)]['count_better'].mean()
                    a2.plot([i - 0.35, i + 0.35], [r, r], color=INK, ls=ls, lw=0.8)
            except FileNotFoundError:
                pass
        a2.set_xticks(range(len(L)), list(L))
        a2.set_ylim(0, 1)
        a2.set_ylabel('fraction where count beats ramp')
        a2.set_title('point = mouse; lines: gate, planted counts (solid)\nand planted ramps (dashed), tau <= 600 s', fontsize=6)
        fig.suptitle('F19. Entry-locked units: entry clock or leaky reward count?')
    _save(fig, out_path)
    return fig


def plot_leaky_unique(L: dict, out_path=None):
    """F20: left, each family's unique held-out gain beyond the other (from the joint refit), per
    entry-locked unit; right, the leaky count's tau against the ramp's tau where both fit."""
    t = table_leaky(L)
    t = t[t['role'] == 'locked']
    with mpl.rc_context(_rc()):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.8), layout='constrained')
        for ds in L:
            d = t[t['dataset'] == ds]
            a1.scatter(d['ramp_unique'].clip(-0.5, 1), d['count_unique'].clip(-0.5, 1), s=3, color=DATASET_COLORS[ds],
                       alpha=0.6, lw=0, label=ds)
            e = d[(d['frac_ramp'] >= 0.5) & (d['frac_count'] >= 0.5)]
            a2.scatter(e['tau_median'], e['tau_count_median'], s=4, color=DATASET_COLORS[ds], alpha=0.6, lw=0,
                       label=f'{ds} ({len(e)})')
        a1.axhline(0, color=NULL, lw=0.6)
        a1.axvline(0, color=NULL, lw=0.6)
        a1.set_xlabel('ramp: unique gain beyond the count')
        a1.set_ylabel('count: unique gain beyond the ramp')
        a1.legend(frameon=False, markerscale=3)
        lim = [3, 1e4]
        a2.plot(lim, lim, color=INK, lw=0.5)
        a2.set_xscale('log')
        a2.set_yscale('log')
        a2.set_xlabel('ramp tau (s)')
        a2.set_ylabel('leaky count tau (s)')
        a2.legend(frameon=False, markerscale=3)
        fig.suptitle('F20. Unique gains (fractions of the kernel gain, clipped) and matched time constants')
    _save(fig, out_path)
    return fig


def plot_leaky_gate(L: dict, out_path=None):
    """F21: the gate envelope, the fraction of planted units whose own family wins held out, by planted
    tau; and the real recdays' ramp-count correlation, which sets how separable they are."""
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), layout='constrained')
        for ds in L:
            try:
                g = latest_leaky_gate(ds)
            except FileNotFoundError:
                continue
            for kind, ls in (('leaky_count', '-'), ('entry_ramp', '--')):
                d = g[g['kind'] == kind]
                correct = d['count_better'] if kind == 'leaky_count' else ~d['count_better'].astype(bool)
                r = correct.groupby(d['tau']).mean()
                axes[0].plot(r.index, r.values, ls=ls, marker='o', ms=3, color=DATASET_COLORS[ds],
                             label=f"{ds} {kind.replace('_', ' ')}")
        axes[0].axhline(0.8, color=NULL, lw=0.6)
        axes[0].set_xscale('log')
        axes[0].set_ylim(0, 1.05)
        axes[0].set_xlabel('planted tau (s)')
        axes[0].set_ylabel('own family wins held out')
        axes[0].legend(frameon=False, fontsize=5)
        for i, ds in enumerate(L):
            sm = L[ds]['summary']
            for j, col in enumerate([c for c in sm.columns if c.startswith('r_ramp_count_')]):
                v = sm[col].dropna().to_numpy()
                axes[1].scatter(np.full(len(v), i * 3 + j) + _jitter(len(v), 0.4), v, s=6, color=DATASET_COLORS[ds], lw=0)
        axes[1].set_xticks([0, 1, 3, 4], ['LEC 300 s', 'LEC 1200 s', 'PFC 300 s', 'PFC 1200 s'], rotation=30)
        axes[1].set_ylabel('ramp-count correlation per recday')
        fig.suptitle('F21. Can the behaviour tell a clock from a count? Gate envelope and collinearity')
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# F22-F30: M3 part 2 (Tsao's stepwise table, Kanter's single-unit timescales and part 2, the ramp test)
# ---------------------------------------------------------------------------

TS_DIR = os.path.join(REPO, 'data', 'event_kernels', 'timescales')
PART2_DIR = os.path.join(DRIFT_DIR, 'part2')
M3P2_FIG_DIR = os.path.join(REPO, 'data', 'figures', 'event_kernels', 'M3_part2')
M3P2_DRIFT_FIG_DIR = os.path.join(REPO, 'data', 'figures', 'population_drift', 'M3_part2')
TSAO_TERMS = ['day_time', 'entry_time', 'leg_time', 'loop_time', 'task', 'place']
TSAO_REF = {'day_time': (0.182, 0.057), 'entry_time': (0.062, None), 'leg_time': (None, 0.357)}   # (BW12, figure-eight)
LEG_S, LOOP_S = 9.3, 43.6                      # LEC median leg and loop, 2026-10-14
MIN_UNITS_ALL = 20                             # population measures on the 'all' set need M2's matched size; a 1-unit
                                               # recday (PFC me10_20122021_21122021) decodes worse than its shuffle
STABLE_DEG = 30.0                              # as run_population_drift: an axis whose leave-one-out angle exceeds this is not a coding axis
RAMP_INPUT_COLORS = {'raw': '#2C2C2A', 'div_entry': '#6B3FA0', 'div_ramp60': '#FFA500', 'div_ramp120': '#C46A00',
                     'div_ramp240': '#7A4100'}
TERM_LABEL = {'day_time': 'day time\n(Tsao session)', 'entry_time': 'entry time\n(BW12 trial)',
              'leg_time': 'leg time\n(fig-8 trial)', 'loop_time': 'loop time\n(since A)', 'task': 'task\n(wall colour)',
              'place': 'place'}


def _unit_info(M: dict) -> pd.DataFrame:
    """Unit table of every recday of a timescales or part-2 merge: region, boundary, rate, El-Gaby flag."""
    rows = []
    for rd, u in M['units'].items():
        u = u.copy()
        u['recday'], u['mouse'] = rd, rd.split('_')[0]
        rows.append(u)
    return pd.concat(rows, ignore_index=True)


def _stage1_calls(ds: str) -> pd.DataFrame:
    """Per unit: the Stage 1 (>= 5 trials, anchor entry) locked calls for entry, reward A and reward."""
    T = pd.read_csv(load_latest(os.path.join(STAGE1_DIR, f'stage1_{ds.lower()}_kin-strict_mt5_np100'))['_path'][:-4] + '_units.csv')
    T = T[(T['anchor'] == 'entry') & T['block'].isin(['ev_entry', 'ev_A', 'ev_reward'])]
    return T.pivot_table(index=['recday', 'unit'], columns='block', values='locked', aggfunc='first').astype(bool).reset_index()


def table_tsao(TS: dict) -> pd.DataFrame:
    """Per dataset and term: fraction of units selecting it (mean of mice, range), and its overlap
    with Stage 1's held-out calls."""
    rows = []
    for ds, M in TS.items():
        t = M['tsao'].merge(_stage1_calls(ds), on=['recday', 'unit'], how='left')
        for term in TSAO_TERMS:
            t['v'] = t[f'has_{term}'].astype(float)
            pm = per_mouse(t, 'v')
            row = dict(dataset=ds, term=term, units=len(t), mean_of_mice=pm.mean(), lo=pm.min(), hi=pm.max(),
                       pooled=t['v'].mean())
            for blk, tm in (('ev_entry', 'entry_time'), ('ev_reward', 'leg_time'), ('ev_A', 'loop_time')):
                if term == tm and blk in t:
                    s1 = t[blk].fillna(False).astype(bool)
                    sel = t[f'has_{term}'].astype(bool)
                    row.update(stage1_block=blk, p_stage1_given_tsao=s1[sel].mean(), p_tsao_given_stage1=sel[s1].mean(),
                               p_stage1_overall=s1.mean())
            rows.append(row)
        t['time_cell'] = t[['has_day_time', 'has_entry_time']].any(axis=1).astype(float)
        pm = per_mouse(t, 'time_cell')
        rows.append(dict(dataset=ds, term='day or entry time', units=len(t), mean_of_mice=pm.mean(), lo=pm.min(),
                         hi=pm.max(), pooled=t['time_cell'].mean()))
    return pd.DataFrame(rows)


def plot_tsao_table(TS: dict, out_path=None):
    """F22: left, the fraction of units selecting each term (bar = mean of mice, point = mouse), with
    Tsao's LEC fractions marked; right, how Tsao's in-sample time terms overlap Stage 1's held-out
    calls (P(Stage 1 locked | term selected) and P(term selected | Stage 1 locked))."""
    t = table_tsao(TS)
    with mpl.rc_context(_rc()):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8), layout='constrained', width_ratios=[2.2, 1])
        x = np.arange(len(TSAO_TERMS))
        for j, ds in enumerate(TS):
            d = TS[ds]['tsao']
            for i, term in enumerate(TSAO_TERMS):
                pm = per_mouse(d.assign(v=d[f'has_{term}'].astype(float)), 'v')
                xx = i + (j - 0.5) * 0.36
                a1.bar(xx, pm.mean(), width=0.34, color=DATASET_COLORS[ds], alpha=0.4, lw=0, label=ds if i == 0 else None)
                a1.scatter(xx + _jitter(len(pm), 0.2), pm.values, s=7, color=DATASET_COLORS[ds], lw=0, zorder=3)
        for i, term in enumerate(TSAO_TERMS):
            bw, f8 = TSAO_REF.get(term, (None, None))
            if bw is not None:
                a1.plot([i - 0.4, i + 0.4], [bw, bw], color=INK, lw=0.9, ls='-')
            if f8 is not None:
                a1.plot([i - 0.4, i + 0.4], [f8, f8], color=INK, lw=0.9, ls=':')
        a1.set_xticks(x, [TERM_LABEL[t_] for t_ in TSAO_TERMS])
        a1.set_ylabel('fraction of units selecting the term')
        a1.set_ylim(0, 1)
        a1.legend(frameon=False, loc='upper left')
        a1.set_title('Tsao LEC: solid = BW12, dotted = figure-eight (session or mixed / trial or mixed)', fontsize=6)
        c = t.dropna(subset=['p_stage1_given_tsao'])
        for j, ds in enumerate(TS):
            cc = c[c['dataset'] == ds]
            for i, r in enumerate(cc.itertuples()):
                a2.scatter(i + (j - 0.5) * 0.3, r.p_stage1_given_tsao, marker='o', s=14, color=DATASET_COLORS[ds], lw=0)
                a2.scatter(i + (j - 0.5) * 0.3, r.p_tsao_given_stage1, marker='^', s=14, color=DATASET_COLORS[ds], lw=0)
                a2.plot([i + (j - 0.5) * 0.3 - 0.1, i + (j - 0.5) * 0.3 + 0.1], [r.p_stage1_overall] * 2, color=NULL, lw=0.8)
        terms = list(dict.fromkeys(c['term']))
        a2.set_xticks(range(len(terms)), [f"{tm.replace('_', ' ')}\nvs {LABEL[b]}" for tm, b in
                                          dict.fromkeys(zip(c['term'], c['stage1_block']))])
        a2.set_ylim(0, 1)
        a2.set_ylabel('overlap, pooled units')
        a2.set_title('o P(Stage 1 locked | selected), ^ P(selected | locked)\ngrey: Stage 1 base rate', fontsize=6)
        fig.suptitle("F22. Tsao's in-sample stepwise GLM on our units (500 ms bins, adjusted R² +0.01 / -0.005)")
    _save(fig, out_path)
    return fig


def table_tsao_tau(TS: dict, S2: dict) -> pd.DataFrame:
    """Tsao's exponential tau per selected time term (fits flagged ok), and for entry time against
    Stage 2's entry-ramp tau of the same unit."""
    rows = []
    for ds, M in TS.items():
        t = M['tsao']
        for term in ('day_time', 'entry_time', 'leg_time', 'loop_time'):
            if f'{term}_tau_s' not in t:
                continue
            d = t[t[f'has_{term}'].astype(bool) & t[f'{term}_ok'].fillna(False).astype(bool)]
            rows.append(dict(dataset=ds, term=term, n=len(d), median_tau_s=d[f'{term}_tau_s'].median(),
                             q25=d[f'{term}_tau_s'].quantile(0.25), q75=d[f'{term}_tau_s'].quantile(0.75),
                             frac_rising=float((d[f'{term}_sign'] > 0).mean()) if len(d) else np.nan))
    return pd.DataFrame(rows)


def plot_tsao_tau(TS: dict, S2: dict, out_path=None):
    """F23: left, Tsao's exponential time constants per term (fits that resolve, tau <= 10x the fitted
    range); right, entry time: Tsao's tau against Stage 2's ramp tau for the same units."""
    with mpl.rc_context(_rc()):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), layout='constrained')
        terms = ['entry_time', 'leg_time', 'loop_time', 'day_time']
        for j, ds in enumerate(TS):
            t = TS[ds]['tsao']
            for i, term in enumerate(terms):
                if f'{term}_tau_s' not in t:
                    continue
                d = t[t[f'has_{term}'].astype(bool) & t[f'{term}_ok'].fillna(False).astype(bool)][f'{term}_tau_s']
                if len(d):
                    y = np.log10(d.to_numpy())
                    a1.scatter(i + (j - 0.5) * 0.35 + _jitter(len(y), 0.25), y, s=2, color=DATASET_COLORS[ds], alpha=0.5, lw=0)
                    a1.plot([i + (j - 0.5) * 0.35 - 0.12, i + (j - 0.5) * 0.35 + 0.12], [np.median(y)] * 2, color=INK, lw=1.2)
            s2 = S2[ds]['table']
            s2 = s2[(s2['event'] == 'entry') & s2['shape'].isin(['ramp_up', 'ramp_down'])][['recday', 'unit', 'tau_median']]
            m = t.merge(s2, on=['recday', 'unit'])
            if 'entry_time_tau_s' in m:
                m = m[m['has_entry_time'].astype(bool) & m['entry_time_ok'].fillna(False).astype(bool)]
                a2.scatter(m['tau_median'], m['entry_time_tau_s'], s=5, color=DATASET_COLORS[ds], lw=0, alpha=0.7,
                           label=f'{ds} ({len(m)})')
        a1.set_xticks(range(len(terms)), [TERM_LABEL[t_].split('\n')[0] for t_ in terms])
        a1.set_ylabel('log10 tau (s)')
        lim = [10, 1e5]
        a2.plot(lim, lim, color=INK, lw=0.5)
        a2.set_xscale('log')
        a2.set_yscale('log')
        a2.set_xlabel('Stage 2 entry-ramp tau (s), held out')
        a2.set_ylabel("Tsao's entry-time tau (s), in sample")
        a2.legend(frameon=False, markerscale=2)
        fig.suptitle("F23. Tsao's exponential time constants (f = a e^bx; tau = 1/|b|)")
    _save(fig, out_path)
    return fig


def _unit_medians(K: pd.DataFrame, cols) -> pd.DataFrame:
    return K.groupby(['mouse', 'recday', 'unit'])[cols].median().reset_index()


def table_kanter(TS: dict) -> pd.DataFrame:
    """Per dataset: Kanter's GLM elapsed-time fraction (unit-sessions with p < 0.05), its tau, and
    the ACF and Fano measures raw and against the held-out prediction (per-unit medians over
    sessions, then mean of mice)."""
    rows = []
    for ds, M in TS.items():
        K = M['kanter'].copy()
        K['sig'] = (K['glm_p'] < 0.05).astype(float)
        U = _unit_medians(K, ['sig', 'acf_tau_s', 'res_acf_tau_s', 'acf_tail', 'res_acf_tail', 'log_fano', 'res_log_fano'])
        r = dict(dataset=ds, unit_sessions=len(K), units=len(U))
        for c in ('sig', 'acf_tau_s', 'res_acf_tau_s', 'acf_tail', 'res_acf_tail', 'log_fano', 'res_log_fano'):
            pm = U.groupby(['mouse', 'recday'])[c].median().groupby('mouse').mean()
            r[c] = pm.mean()
            r[c + '_range'] = f'{pm.min():.3g}-{pm.max():.3g}'
        g = K[(K['glm_p'] < 0.05) & K['glm_exp_ok'].astype(bool)]
        r['glm_tau_median_s'] = g['glm_tau_s'].median()
        r['glm_frac_rising'] = float((g['glm_sign'] > 0).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def plot_kanter_units(TS: dict, out_path=None):
    """F24: left, the fraction of unit-sessions where elapsed time is significant in Kanter's GLM
    (per mouse) and its tau; right, the autocorrelation tau of each unit, raw against residual
    (medians over sessions)."""
    with mpl.rc_context(_rc()):
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.5), layout='constrained')
        for i, (ds, M) in enumerate(TS.items()):
            K = M['kanter'].assign(sig=lambda x: (x['glm_p'] < 0.05).astype(float))
            pm = per_mouse(K, 'sig')
            a1.bar(i, pm.mean(), color=DATASET_COLORS[ds], alpha=0.4, width=0.6, lw=0)
            a1.scatter(i + _jitter(len(pm)), pm.values, s=9, color=DATASET_COLORS[ds], lw=0, zorder=3)
            g = K[(K['glm_p'] < 0.05) & K['glm_exp_ok'].astype(bool)]['glm_tau_s']
            a2.hist(np.log10(g.dropna()), bins=40, histtype='step', color=DATASET_COLORS[ds], label=f'{ds} ({len(g)})')
            U = _unit_medians(K, ['acf_tau_s', 'res_acf_tau_s'])
            a3.scatter(U['acf_tau_s'], U['res_acf_tau_s'], s=2, color=DATASET_COLORS[ds], alpha=0.5, lw=0, label=ds)
        a1.axhline(0.05, color=NULL, lw=0.6)
        a1.set_xticks(range(len(TS)), list(TS))
        a1.set_ylim(0, 1)
        a1.set_ylabel('unit-sessions with elapsed time p < 0.05')
        a2.set_xlabel('log10 GLM tau (s)')
        a2.set_ylabel('unit-sessions')
        a2.legend(frameon=False)
        lim = [0.5, 1e3]
        a3.plot(lim, lim, color=INK, lw=0.5)
        a3.set_xscale('log')
        a3.set_yscale('log')
        a3.set_xlabel('ACF tau, raw counts (s)')
        a3.set_ylabel('ACF tau, held-out residuals (s)')
        a3.legend(frameon=False, markerscale=3)
        fig.suptitle("F24. Kanter's single-unit time constants: GLM of elapsed time + population rate; ACF decay 1-90 s")
    _save(fig, out_path)
    return fig


def plot_acf_curves(TS: dict, out_path=None):
    """F25: median autocorrelation (lag-0 normalised) over units, raw counts vs held-out Pearson
    residuals, with the leg and loop periods marked; band = interquartile range of unit medians."""
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(TS), figsize=(3.2 * len(TS), 2.5), layout='constrained', squeeze=False)
        for ax, (ds, M) in zip(axes[0], TS.items()):
            raw = np.concatenate([np.nanmedian(v[0], axis=1) for v in M['acf'].values()])
            res = np.concatenate([np.nanmedian(v[1], axis=1) for v in M['acf'].values()])
            lags = (np.arange(raw.shape[1]) + 1) * 0.5
            for arr, col, lab in ((raw, DATASET_COLORS[ds], 'raw counts'), (res, INK, 'held-out residuals')):
                q = np.nanpercentile(arr, [25, 50, 75], axis=0)
                ax.fill_between(lags, q[0], q[2], color=col, alpha=0.2, lw=0)
                ax.plot(lags, q[1], color=col, lw=1, label=lab)
            for p in (LEG_S, LOOP_S, 2 * LOOP_S):
                ax.axvline(p, color=NULL, lw=0.6, ls=':')
            ax.axhline(0, color=NULL, lw=0.5)
            ax.set_xscale('log')
            ax.set_xlabel('lag (s); dotted: median leg, loop, 2 loops')
            ax.set_ylabel('autocorrelation')
            ax.set_title(f'{ds}: {raw.shape[0]} units', fontsize=7)
            ax.legend(frameon=False)
        fig.suptitle('F25. Autocorrelation of 0.5 s counts, raw and held-out Pearson residuals (median unit, IQR)')
    _save(fig, out_path)
    return fig


def plot_fano(TS: dict, P2: dict, out_path=None):
    """F26: left, log10 Fano of the 30 s-smoothed rate over Poisson trains, raw and against the
    held-out prediction (per mouse); right, drift growth after removing the top 25% of units by
    Fano, over that after removing a random 25% (set 'all', per mouse)."""
    with mpl.rc_context(_rc()):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6), layout='constrained')
        for i, (ds, M) in enumerate(TS.items()):
            U = _unit_medians(M['kanter'], ['log_fano', 'res_log_fano'])
            for j, c in enumerate(('log_fano', 'res_log_fano')):
                pm = U.groupby(['mouse', 'recday'])[c].median().groupby('mouse').mean()
                x = i * 2.5 + j
                a1.bar(x, pm.mean(), color=DATASET_COLORS[ds], alpha=0.4 if j == 0 else 0.15, width=0.8, lw=0)
                a1.scatter(x + _jitter(len(pm)), pm.values, s=9, color=DATASET_COLORS[ds], lw=0, zorder=3)
            if ds in P2:
                F = _all_set(P2[ds]['fano_removal']).assign(ratio=lambda x: x['top_removed'] / x['random_removed'])
                pm = per_mouse(F, 'ratio')
                a2.bar(i, pm.mean(), color=DATASET_COLORS[ds], alpha=0.4, width=0.6, lw=0)
                a2.scatter(i + _jitter(len(pm)), pm.values, s=9, color=DATASET_COLORS[ds], lw=0, zorder=3)
        a1.axhline(0, color=INK, lw=0.6)
        a1.set_xticks([0, 1, 2.5, 3.5][:2 * len(TS)], ['raw', 'residual', 'raw', 'residual'][:2 * len(TS)])
        a1.set_ylabel('log10 Fano / Poisson (median unit)')
        a1.set_title(' / '.join(TS), fontsize=7)
        a2.axhline(1, color=INK, lw=0.6)
        a2.set_xticks(range(len(TS)), list(TS))
        a2.set_ylabel('drift growth: top-25% removed / random removed')
        fig.suptitle('F26. Minute-scale variability (Kanter Fig 5): Fano factors and removing the most variable units')
    _save(fig, out_path)
    return fig


def _all_set(tab: pd.DataFrame) -> pd.DataFrame:
    """Rows of the 'all' unit set from recdays with at least MIN_UNITS_ALL units."""
    return tab[(tab['set'] == 'all') & (tab['n_units'] >= MIN_UNITS_ALL)]


def shift_transitions(P2: dict) -> dict:
    """Per dataset: Kanter's shift, the transition where the reward-aligned window speed (set
    'all') peaks, averaged over every recday, session and reward type."""
    out = {}
    for ds, P in P2.items():
        s = _all_set(P['speeds'])
        out[ds] = int(np.nanargmax(np.nanmean(np.stack(s['speed'].to_numpy()), axis=0)))
    return out


TRANSITION_LABEL = ['-3>-2', '-2>-1', '-1>0', '0>1', '1>2', '2>3']      # into the 1 s bin starting at the right edge


def table_boundaries(P2: dict) -> pd.DataFrame:
    """Per dataset at its shift transition: fraction of units preferring it, PV correlation of up /
    down / other units, and loop-number decoding error vs shuffle (set 'all', mean of mice)."""
    sh = shift_transitions(P2)
    rows = []
    for ds, P in P2.items():
        B = _all_set(P['boundary'])
        B = B[B['shift'] == sh[ds]]
        D = _all_set(P['decoding'])
        r = dict(dataset=ds, shift=TRANSITION_LABEL[sh[ds]], chance=1 / 6)
        for c in ('frac_pref', 'pv_pos', 'pv_neg', 'pv_other', 'n_pos', 'n_neg'):
            pm = per_mouse(B, c)
            r[c], r[c + '_range'] = pm.mean(), f'{pm.min():.3g}-{pm.max():.3g}'
        for c in ('error', 'shuffle'):
            r['decode_' + c] = per_mouse(D, c).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def plot_boundaries(P2: dict, out_path=None):
    """F27: left, the fraction of units whose largest change falls on each 1 s transition around
    the reward (mean of mice; chance 1/6); middle, PV correlation across loops of up- and down-
    modulated units at the shift vs other units; right, loop-number decoding error vs shuffle."""
    sh = shift_transitions(P2)
    with mpl.rc_context(_rc()):
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.4, 2.5), layout='constrained')
        for i, (ds, P) in enumerate(P2.items()):
            B = _all_set(P['boundary'])
            fr = B.groupby(['mouse', 'recday', 'shift'])['frac_pref'].mean().groupby(['mouse', 'shift']).mean().unstack()
            a1.plot(range(6), fr.mean().to_numpy(), marker='o', ms=3, color=DATASET_COLORS[ds], label=f'{ds} (shift {TRANSITION_LABEL[sh[ds]]} s)')
            b = B[B['shift'] == sh[ds]]
            for j, c in enumerate(('pv_pos', 'pv_neg', 'pv_other')):
                pm = per_mouse(b, c)
                x = i * 4 + j
                a2.bar(x, pm.mean(), color=DATASET_COLORS[ds], alpha=0.4 if c != 'pv_other' else 0.15, width=0.8, lw=0)
                a2.scatter(x + _jitter(len(pm)), pm.values, s=8, color=DATASET_COLORS[ds], lw=0, zorder=3)
            D = _all_set(P['decoding'])
            e, s = per_mouse(D, 'error'), per_mouse(D, 'shuffle')
            for m in e.index:
                a3.plot([i * 3, i * 3 + 1], [e[m], s[m]], color=DATASET_COLORS[ds], lw=0.6, marker='o', ms=3)
        a1.axhline(1 / 6, color=NULL, lw=0.6)
        a1.set_xticks(range(6), TRANSITION_LABEL, rotation=30)
        a1.set_xlabel('transition (s from reward arrival)')
        a1.set_ylabel('fraction of units preferring')
        a1.legend(frameon=False)
        a2.axhline(0, color=INK, lw=0.5)
        a2.set_xticks([0, 1, 2, 4, 5, 6][:3 * len(P2)], ['up', 'down', 'other'] * len(P2))
        a2.set_ylabel('PV correlation across loops at the shift')
        a3.set_xticks([0, 1, 3, 4][:2 * len(P2)], ['decoded', 'shuffled'] * len(P2))
        a3.set_ylabel('loop-number error (loops)')
        a3.set_title('lines: mice; ' + ' / '.join(P2), fontsize=6)
        fig.suptitle('F27. Reward boundaries (Kanter Fig 6): which transition, which ensemble, and time stamps')
    _save(fig, out_path)
    return fig


AXIS_PAIRS = ['leg_session', 'legA_session', 'legB_session', 'legC_session', 'legD_session', 'loop_session', 'leg_loop']


def table_axes(P2: dict, sets=('all', 'all_n20')) -> pd.DataFrame:
    """Per dataset, unit set and pair: offset and the two stabilities (deg) and multiplexing,
    session mean per recday, then mouse mean."""
    rows = []
    for ds, P in P2.items():
        A = P['axes']
        for st in sets:
            for pair in AXIS_PAIRS:
                a = A[(A['set'] == st) & (A['pair'] == pair)]
                if not len(a):
                    continue
                a = a.assign(stable=lambda x: ((x['stab_a'] <= STABLE_DEG) & (x['stab_b'] <= STABLE_DEG)).astype(float))
                r = dict(dataset=ds, set=st, pair=pair, mice=a['mouse'].nunique())
                for c in ('offset', 'stab_a', 'stab_b', 'multiplex', 'stable'):
                    pm = per_mouse(a, c)
                    r[c], r[c + '_range'] = pm.mean(), f'{pm.min():.3g}-{pm.max():.3g}'
                rows.append(r)
    return pd.DataFrame(rows)


def plot_axes(P2: dict, out_path=None):
    """F28: angle between coding axes (Kanter Fig 4G) per pair, against each axis's leave-one-out
    stability, at 20 matched units (mean of draws per recday, then mouse); right, multiplexing."""
    t = table_axes(P2, sets=('all_n20',))
    with mpl.rc_context(_rc()):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), layout='constrained', width_ratios=[2.4, 1])
        pairs = [p for p in AXIS_PAIRS if p in set(t['pair'])]
        for j, ds in enumerate(P2):
            A = P2[ds]['axes']
            A = A[A['set'] == 'all_n20']
            for i, pair in enumerate(pairs):
                a = A[A['pair'] == pair]
                if not len(a):
                    continue
                x = i + (j - 0.5) * 0.36
                for c, mk in (('offset', 'o'), ('stab_a', '_'), ('stab_b', '_')):
                    pm = per_mouse(a, c)
                    a1.scatter(x + _jitter(len(pm), 0.15), pm.values, s=10 if c == 'offset' else 18, marker=mk,
                               color=DATASET_COLORS[ds] if c == 'offset' else INK, lw=0.8 if mk == '_' else 0)
                pm = per_mouse(a, 'multiplex')
                if pair == 'leg_session':
                    a2.scatter(j + _jitter(len(pm)), pm.values, s=9, color=DATASET_COLORS[ds], lw=0)
        a1.axhline(90, color=NULL, lw=0.6)
        a1.axhline(STABLE_DEG, color=NULL, lw=0.6, ls=':')
        a1.set_xticks(range(len(pairs)), [p.replace('_', ' vs ') for p in pairs], rotation=30)
        a1.set_ylabel('angle (deg): o offset, - stabilities\n(dotted: stability limit 30 deg)')
        a1.set_ylim(0, 95)
        a2.axhline(1 / 16, color=NULL, lw=0.6)
        a2.set_xticks(range(len(P2)), list(P2))
        a2.set_ylabel('multiplexing, leg vs session')
        fig.suptitle('F28. Orthogonal coding axes (Kanter Fig 4G), 20-unit draws')
    _save(fig, out_path)
    return fig


def table_ramp_test(TS: dict, set_name: str = 'all_n20') -> pd.DataFrame:
    """Held-out and within-session accuracy by input: first minute, minutes 2-17, pooled over
    recdays, draws and sessions, bins weighting (as M2's F13)."""
    rows = []
    for ds, M in TS.items():
        R = M['ramp']
        R = R[R['set'] == set_name]
        for inp, r in R.groupby('input'):
            for lab, m in (('minute 1', r['epoch'] == 0), ('minutes 2-17', r['epoch'] > 0)):
                rr = r[m]
                w = rr['n_bins']
                rows.append(dict(dataset=ds, set=set_name, input=inp, epochs=lab, n_bins=int(w.sum()),
                                 within=float(np.average(rr['within'], weights=w)),
                                 across=float(np.average(rr['across'].fillna(0), weights=w * rr['across'].notna())),
                                 chance=float(rr['chance'].mean())))
    return pd.DataFrame(rows)


def plot_ramp_test(TS: dict, out_path=None):
    """F29: accuracy by true epoch, held out (solid) and within session (dashed), for raw counts and
    with the entry-locked units divided by their held-out shape-free entry kernel or slow ramp
    (20-unit draws, pooled); right, the entry-locked units alone vs size-matched non-locked units."""
    cols = RAMP_INPUT_COLORS
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(TS) + 1, figsize=(3.0 * (len(TS) + 1), 2.6), layout='constrained', squeeze=False)
        for ax, (ds, M) in zip(axes[0], TS.items()):
            R = M['ramp']
            R = R[R['set'] == 'all_n20']
            for inp in [i for i in RAMP_INPUT_COLORS if i in set(R['input'])]:
                r = R[R['input'] == inp]
                g = r.groupby('epoch').apply(lambda x: pd.Series(dict(
                    within=np.average(x['within'], weights=x['n_bins']),
                    across=np.average(x['across'].fillna(0), weights=x['n_bins'] * x['across'].notna()))))
                ax.plot(g.index, g['across'], color=cols.get(inp, NULL), lw=1, label=inp)
                ax.plot(g.index, g['within'], color=cols.get(inp, NULL), lw=0.8, ls='--')
            ax.axhline(1 / 18, color=NULL, lw=0.6)
            ax.set_xticks(range(0, 18, 3))
            ax.set_xlabel('true 1-min epoch')
            ax.set_ylabel('accuracy')
            ax.set_title(ds, fontsize=7)
            ax.legend(frameon=False)
        ax = axes[0][-1]
        for j, (ds, M) in enumerate(TS.items()):
            R = M['ramp']
            for k, st in enumerate(('locked_only', 'nonlocked_matched')):
                r = R[(R['set'] == st) & (R['epoch'] == 0)]
                if not len(r):
                    continue
                pm = r.groupby(['mouse', 'recday']).apply(lambda x: np.average(x['across'].fillna(0), weights=x['n_bins'] * x['across'].notna())).groupby('mouse').mean()
                x = j * 2.5 + k
                ax.bar(x, pm.mean(), color=DATASET_COLORS[ds], alpha=0.4 if k == 0 else 0.15, width=0.8, lw=0)
                ax.scatter(x + _jitter(len(pm)), pm.values, s=8, color=DATASET_COLORS[ds], lw=0, zorder=3)
        ax.axhline(1 / 18, color=NULL, lw=0.6)
        ax.set_xticks([0, 1, 2.5, 3.5][:2 * len(TS)], ['locked', 'other'] * len(TS))
        ax.set_ylabel('held-out accuracy, first minute')
        fig.suptitle('F29. The ramp test: does dividing out the entry kernels remove the held-out first-minute decoding?')
    _save(fig, out_path)
    return fig


def latest_gate_csv(root: str, name: str, dataset: str, tag: str = '') -> pd.DataFrame:
    runs = sorted(glob.glob(os.path.join(root, 'runner_gate', f'{dataset.lower()}_{tag}*', name)))
    if not runs:
        raise FileNotFoundError(f'{name} for {dataset}')
    return pd.read_csv(runs[-1])


def table_m3p2_gates(datasets=('LEC', 'PFC')) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for root, name, tag in ((TS_DIR, 'gate.csv', 'full'), (PART2_DIR, 'gate_part2.csv', '')):
            try:
                g = latest_gate_csv(root, name, ds, tag)
            except FileNotFoundError:
                continue
            rows.append(g.assign(dataset=ds))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def region_contrast_df(d: pd.DataFrame, value: str, a='ENTl-deep', b='SUB/ProS', n_perm: int = 10000,
                       seed: int = 0) -> dict:
    """`region_contrast` for any per-unit table (recday, mouse, group, near_boundary, rate_hz and
    `value`): rate-matched, off-boundary, mouse-level difference on the within-recday null."""
    asp = _asp()
    d = d[~d['near_boundary'].astype(bool) & d['group'].isin([a, b])].dropna(subset=[value]).reset_index(drop=True)
    rm = asp.rate_match(d, 'rate_hz', a, b, seed=seed).reset_index(drop=True)
    ref = asp.within_recday_permutation(rm, value, a, b, n_perm=200, seed=seed)
    o200, _, p200 = _within_recday_null(rm, value, a, b, 200, seed)
    if not (np.isclose(o200, ref['observed'], equal_nan=True) and np.isclose(p200, ref['p_perm'])):
        raise AssertionError('replicated permutation differs from anatomy_split.within_recday_permutation')
    obs, null, p = _within_recday_null(rm, value, a, b, n_perm, seed)
    pm = rm.groupby(['mouse', 'recday', 'group'])[value].mean().groupby(['mouse', 'group']).mean().unstack()
    return dict(contrast=f'{a} - {b}', value=value, observed=obs, p_perm=p, n_perm=n_perm, n_units=len(rm),
                mice=sorted(pm.dropna().index), per_mouse=pm)


def unit_measures(M: dict) -> pd.DataFrame:
    """One row per unit of a timescales merge: the unit table, Tsao's flags, and Kanter's per-unit
    medians over sessions (fraction of sessions with a significant elapsed-time term)."""
    U = _unit_info(M)
    t = M['tsao'][['recday', 'unit'] + [c for c in M['tsao'] if c.startswith('has_')]]
    K = M['kanter'].assign(glm_sig=lambda x: (x['glm_p'] < 0.05).astype(float))
    k = K.groupby(['recday', 'unit'])[['glm_sig', 'acf_tau_s', 'res_acf_tau_s', 'log_fano', 'res_log_fano']].median().reset_index()
    out = U.merge(t, on=['recday', 'unit'], how='left').merge(k, on=['recday', 'unit'], how='left')
    for c in [c for c in out if c.startswith('has_')]:
        out[c] = out[c].astype(float)
    out['log_res_acf_tau'] = np.log10(out['res_acf_tau_s'])
    return out


M3P2_CONTRAST_VALUES = ('has_entry_time', 'has_leg_time', 'has_day_time', 'glm_sig', 'log_res_acf_tau', 'res_log_fano')


def table_m3p2_regions(M_lec: dict, n_perm: int = 10000) -> pd.DataFrame:
    """ENTl-deep - SUB/ProS for each single-unit measure, as M2's region contrasts."""
    U = unit_measures(M_lec)
    rows = []
    for v in M3P2_CONTRAST_VALUES:
        r = region_contrast_df(U, v, n_perm=n_perm)
        rows.append(dict(value=v, contrast=r['contrast'], observed=r['observed'], p_perm=r['p_perm'],
                         n_units=r['n_units'], mice=len(r['mice']),
                         per_mouse=' '.join(f"{m}:{x:+.3f}" for m, x in (r['per_mouse']['ENTl-deep'] - r['per_mouse']['SUB/ProS']).dropna().items())))
    return pd.DataFrame(rows)


RAMP_INPUTS = ['div_entry', 'div_ramp60', 'div_ramp120', 'div_ramp240']


def _removed(acc0: dict, chance: float) -> dict:
    """Fraction of the first-minute held-out advantage over chance removed by each input."""
    adv = acc0['raw'] - chance
    return {i: (1.0 - (acc0[i] - chance) / adv) if adv > 0 and i in acc0 else np.nan for i in RAMP_INPUTS}


def table_ramp_envelope(TS: dict, set_name: str = 'all_n20') -> pd.DataFrame:
    """Fraction of the first-minute held-out advantage removed per input: real data (pooled bins,
    and per mouse) against the gate's planted slow ramps and 15 s transients (set 'all')."""
    rows = []
    for ds, M in TS.items():
        R = M['ramp']
        R = R[(R['set'] == set_name) & (R['epoch'] == 0)]
        acc0 = {i: float(np.average(r['across'].fillna(0), weights=r['n_bins'] * r['across'].notna())) for i, r in R.groupby('input')}
        rows.append(dict(source=f'{ds} real', mouse='pooled', **_removed(acc0, 1 / 18)))
        for (mouse,), r in R.groupby(['mouse']):
            a = {i: float(np.average(x['across'].fillna(0), weights=x['n_bins'] * x['across'].notna())) for i, x in r.groupby('input')}
            rows.append(dict(source=f'{ds} real', mouse=mouse, **_removed(a, 1 / 18)))
        for which, lab in (('ramp', 'planted slow ramps'), ('transient', 'planted 15 s transients')):
            try:
                g = latest_gate_csv(TS_DIR, f'{which}_ramp_summary.csv', ds, 'full').set_index('input')
            except FileNotFoundError:
                continue
            rows.append(dict(source=f'{ds} {lab}', mouse='gate', **_removed(g['across_e0'].to_dict(), float(g['chance'].iloc[0]))))
    return pd.DataFrame(rows)


def plot_ramp_envelope(TS: dict, out_path=None):
    """F30: how much of the first-minute held-out advantage each division removes, for the real
    data (pooled line, mouse points) and for the gate's planted slow ramps and 15 s transients:
    the real data's profile over the slow-ramp floors says which of the two it resembles."""
    t = table_ramp_envelope(TS)
    x = np.arange(len(RAMP_INPUTS))
    with mpl.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(TS), figsize=(3.3 * len(TS), 2.7), layout='constrained', squeeze=False)
        for ax, ds in zip(axes[0], TS):
            for src, ls, col in ((f'{ds} planted slow ramps', '--', NULL), (f'{ds} planted 15 s transients', ':', NULL)):
                r = t[t['source'] == src]
                if len(r):
                    ax.plot(x, r[RAMP_INPUTS].to_numpy()[0], ls=ls, color=INK, lw=0.9, marker='s', ms=3,
                            label=src.split(' ', 1)[1])
            real = t[t['source'] == f'{ds} real']
            pooled = real[real['mouse'] == 'pooled']
            ax.plot(x, pooled[RAMP_INPUTS].to_numpy()[0], color=DATASET_COLORS[ds], lw=1.4, marker='o', ms=4, label='real, pooled')
            pm = real[real['mouse'] != 'pooled']
            for _, r in pm.iterrows():
                ax.scatter(x + _jitter(len(x), 0.0), r[RAMP_INPUTS].to_numpy(float), s=6, color=DATASET_COLORS[ds], alpha=0.5, lw=0)
            ax.axhline(0, color=NULL, lw=0.5)
            ax.axhline(1, color=NULL, lw=0.5)
            ax.set_xticks(x, ['shape-free', 'ramp >= 60 s', '>= 120 s', '>= 240 s'], rotation=20)
            ax.set_ylabel('first-minute held-out advantage removed')
            ax.set_title(ds, fontsize=7)
            ax.legend(frameon=False, fontsize=5)
        fig.suptitle('F30. The ramp test against its gate: slow ramps vs a fast transient locked to bin 0')
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
