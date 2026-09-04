"""Plots for the PFC production GLM, on PFC's own terms.

The cross-dataset comparison lives in `PFC_glm_production.ipynb` section 3. This module is
the other half: what does the PFC fit look like as a result in itself?

Every function here reads the cross-validated results (`{recday: {..., 'delta_r2_cv': ...}}`)
produced by `run_glm_batch.py`, returns a `matplotlib.figure.Figure`, and takes an optional
`out_path`.

Two conventions the numbers here depend on:

  * **delta_r2 by default, CPD available.** Every plotting function takes
    `value='delta_r2_cv'` (dRSS/TSS, one denominator shared by all regressors, so bars are
    comparable and roughly additive toward r2_cv) or `value='cpd_cv'` (dRSS/RSS_reduced, each
    group divided by its own reduced-model fit). Pass `'cpd_cv'` for continuity with the
    existing corpus; prefer delta_r2 for comparing regressors, because CPD inflates weak ones
    when a dominant regressor remains in the model -- by up to 12x in simulation, and this
    dataset is exactly that case (place dominates).
  * **Mice, not recdays, are the unit of inference.** A mouse contributes several recdays of
    the same brain, so every pooled statistic averages within mouse first and shows the
    per-mouse values individually.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: GridMaze palette. `true/shuffle` for signal-against-null, `past/future` for the
#: retrospective/prospective split that the LEC fits showed, neutrals for non-data ink.
C_SIGNAL = '#C03030'      # true / signal
C_NULL = '#555555'        # shuffle grey
C_PAST = '#C03030'        # retrospective
C_FUTURE = '#2A6FB5'      # prospective
C_STONE = '#B4B2A9'       # CIs, subdued controls
C_CAVIAR = '#2C2C2A'      # primary trace / structural
C_NEUTRAL = '#888780'     # legacy neutral

#: Regressors grouped by what they are about, for consistent ordering and coloring.
_FAMILY = {
    'place': 'space', 'distance_from_reward': 'space', 'distance_to_reward': 'space',
    'task_state': 'task', 'goal_progress': 'task', 'goal_progress_distance': 'task',
    'progress_since_A': 'task',
    'time_from_reward': 'time', 'time_to_reward': 'time',
    'time_since_A': 'time', 'time_to_A': 'time',
    'speed': 'motor', 'acceleration': 'motor',
    'head_direction': 'motor',
}
_FAMILY_COLOR = {'space': '#0F4C81', 'task': '#BE3455', 'time': '#88B04B',
                 'motor': '#888780', 'other': '#B4B2A9'}



#: Statistics a plot can show. The `_corrected` variants subtract each regressor's own
#: permutation-null centre from its observed value.
#:
#: WHY THIS MATTERS. Held-out delta_r2 and CPD both carry a DOWNWARD bias proportional to the
#: regressor's column count: the full model's extra k parameters always fit training noise and
#: cost ~k*sigma^2 of held-out error, so a regressor explaining nothing scores about
#: -k*sigma^2/denominator rather than 0. Measured on the LEC fit, corr(n_cols, null centre) =
#: -0.838, and correcting for it moves 14 of 16 regressors in the ranking -- `head_direction`
#: (35 columns, the largest penalty in the design) rises from 4th to 2nd, `poke_rewarded`
#: (1 column, almost no penalty) falls from 5th to 10th.
#:
#: So ZERO IS NOT THE REFERENCE for a raw value; the null centre is. A raw delta_r2 of 0.000
#: means "signal exactly cancelling the parameter penalty", not "no signal".
#:
#: The raw variants are kept because they are what the existing corpus reports, and because
#: the residual corr(n_cols, corrected) = +0.517 is not obviously bias -- a 35-column block
#: genuinely has more capacity to capture real structure than a 1-column indicator. The
#: correction removes the parameter penalty, not any advantage of expressiveness.
VALUE_OPTIONS = ('delta_r2_cv', 'cpd_cv', 'delta_r2_corrected', 'cpd_corrected')

_VALUE_LABEL = {
    'delta_r2_cv':        r'$\Delta R^2$ (held-out, unique)',
    'cpd_cv':             'CPD (held-out, unique / reduced-model RSS)',
    'delta_r2_corrected': r'$\Delta R^2$ - null centre (bias-corrected)',
    'cpd_corrected':      'CPD - null centre (bias-corrected)',
}
_VALUE_SHORT = {'delta_r2_cv': 'delta_r2', 'cpd_cv': 'CPD',
                'delta_r2_corrected': 'delta_r2_corr', 'cpd_corrected': 'CPD_corr'}


def resolve_value(r, value, null='freedman_lane'):
    """`{group: array}` for one recday's requested statistic, correcting on the fly.

    Falls back through the available null keys so results fitted before the multi-null
    refactor (which stored only `null_mean`, CPD under the shuffle null) still work.
    """
    if value in ('delta_r2_cv', 'cpd_cv'):
        return r[value]
    if value not in VALUE_OPTIONS:
        raise ValueError(f'value must be one of {VALUE_OPTIONS}, got {value!r}')
    stat = 'delta_r2' if value.startswith('delta_r2') else 'cpd'
    base = r[f'{stat}_cv']
    for key in (f'null_mean_{null}__{stat}', f'null_mean_shuffle__{stat}', 'null_mean'):
        if key in r:
            nm = r[key]
            break
    else:
        raise KeyError(f'no null centre stored for {value!r}; refit with n_perm > 0')
    return {g: np.asarray(base[g], float) - np.asarray(nm[g], float)
            for g in base if g in nm}


def _style():
    """Apply the repo publication style. Idempotent."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import glm_analysis_v2 as glm
    glm.apply_gridmaze_style()


def _save(fig, out_path):
    """Save without tight auto-cropping, so panel sizes stay exactly as laid out."""
    if not out_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig.savefig(out_path, bbox_inches=None, dpi=300)


def regressor_names(cv_results, value='delta_r2_cv'):
    """Regressor groups present in the fit, in family order, excluding joint tests."""
    first = next(iter(cv_results.values()))
    names = [g for g in resolve_value(first, value) if not g.startswith('__')]
    joint = {'time_any', 'distance_any', 'gp_any'}
    names = [n for n in names if n not in joint]
    order = {'space': 0, 'task': 1, 'time': 2, 'motor': 3, 'other': 4}
    return sorted(names, key=lambda n: (order[_FAMILY.get(n, 'other')], n))


def to_long(cv_results, value='delta_r2_cv'):
    """Per-neuron long-format table: recday, mouse, neuron, regressor, value, p."""
    rows = []
    for rd, r in cv_results.items():
        mouse = rd.split('_')[0]
        vals = resolve_value(r, value)
        for g in regressor_names(cv_results, value):
            v = np.asarray(vals[g], dtype=float)
            p = (np.asarray(r['p_cv'][g], dtype=float) if 'p_cv' in r
                 else np.full(len(v), np.nan))
            rows.append(pd.DataFrame({'recday': rd, 'mouse': mouse,
                                      'neuron': np.arange(len(v)),
                                      'regressor': g, 'value': v, 'p': p}))
    return pd.concat(rows, ignore_index=True)


def per_mouse_stat(long, col='value', statistic='median'):
    """Collapse to one number per (mouse, regressor): per RECDAY first, then across recdays.

    The two-step matters and is not interchangeable with a single pooled median. Recdays of
    one mouse are the same probe in the same brain, re-sorted, so they are repeated measures
    rather than replicates -- and they carry very different neuron counts (1 to 117 in PFC).
    Pooling every neuron into one median therefore lets the biggest recday dominate the
    mouse. Taking the recday medians first weights each recday equally.

    Measured on the PFC fit, the two differ by up to 33% (acceleration), 22% (speed), and
    `goal_progress` changes sign. This matches `anatomy_split.per_mouse_effect`, which the
    region plots use; the two must agree or the pooled and regional panels of the same figure
    set are computed differently.
    """
    per_rd = (long.groupby(['recday', 'mouse', 'regressor'])[col]
              .agg(statistic).reset_index())
    return (per_rd.groupby(['mouse', 'regressor'])[col].mean().reset_index())


# ---------------------------------------------------------------------------

def plot_model_fit(cv_results, ax=None, out_path=None):
    """Held-out R^2 per neuron, pooled and per mouse.

    The first thing to look at: does the GLM explain PFC at all? A neuron with r2_cv <= 0
    is one the model fails to predict out of sample, so the fraction above zero is the
    honest headline, not the mean.
    """
    _style()
    r2 = {rd: np.asarray(r['r2_cv'], dtype=float) for rd, r in cv_results.items()}
    pooled = np.concatenate(list(r2.values()))
    per_mouse = {}
    for rd, v in r2.items():
        per_mouse.setdefault(rd.split('_')[0], []).append(np.nanmedian(v))

    fig, axes = (plt.subplots(1, 2, figsize=(6.0, 2.4)) if ax is None
                 else (ax.figure, [ax, None]))
    a = axes[0]
    lo, hi = np.nanpercentile(pooled, [0.5, 99.5])
    a.hist(pooled, bins=np.linspace(lo, hi, 60), color=C_NEUTRAL, edgecolor='none')
    a.axvline(0, color=C_CAVIAR, lw=0.8, ls='--')
    a.set_xlabel('held-out $R^2$ per neuron')
    a.set_ylabel('neurons')
    a.set_title(f'n={len(pooled)} neurons, {np.nanmean(pooled > 0):.0%} above zero',
                loc='left')

    if axes[1] is not None:
        b = axes[1]
        mice = sorted(per_mouse)
        for i, m in enumerate(mice):
            vals = per_mouse[m]
            b.scatter(np.full(len(vals), i), vals, s=14, color=C_SIGNAL,
                      zorder=3, clip_on=False)
            b.scatter([i], [np.mean(vals)], s=45, marker='_', color=C_CAVIAR, zorder=4)
        b.axhline(0, color=C_CAVIAR, lw=0.8, ls='--')
        b.set_xticks(range(len(mice)))
        b.set_xticklabels(mice, rotation=45, ha='right')
        b.set_ylabel('median held-out $R^2$')
        b.set_title('per recday, grouped by mouse', loc='left')
        fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_regressor_ranking(cv_results, value='delta_r2_cv', ax=None,
                           out_path=None, show_mice=True):
    """Delta_r2 per regressor: what PFC encodes, ranked.

    Bars are the mean over mice; each mouse is the mean over its recdays of that recday's
    median neuron (see `per_mouse_stat` -- recdays are repeated measures, not replicates, and
    pooling their neurons would let the biggest recday dominate). Points are the individual
    mice, shown because with a handful of animals the reader should see them rather than a
    summary that hides a single-animal effect.
    """
    _style()
    long = to_long(cv_results, value)
    per_mouse = per_mouse_stat(long, 'value', 'median')
    order = regressor_names(cv_results, value)
    means = per_mouse.groupby('regressor')['value'].mean().reindex(order)

    fig, a = (plt.subplots(figsize=(4.2, 3.2)) if ax is None else (ax.figure, ax))
    y = np.arange(len(order))
    a.barh(y, means.values,
           color=[_FAMILY_COLOR[_FAMILY.get(n, 'other')] for n in order],
           edgecolor='none', height=0.7, zorder=2)
    if show_mice:
        for i, g in enumerate(order):
            v = per_mouse.loc[per_mouse.regressor == g, 'value'].values
            a.scatter(v, np.full(len(v), i), s=8, color=C_CAVIAR, alpha=0.7,
                      zorder=3, clip_on=False)
    a.axvline(0, color=C_CAVIAR, lw=0.8)
    a.set_yticks(y)
    a.set_yticklabels(order)
    a.invert_yaxis()
    a.set_xlabel(_VALUE_LABEL.get(value, value))
    a.set_title(f'{per_mouse.mouse.nunique()} mice — {_VALUE_SHORT.get(value, value)}',
                loc='left')
    handles = [mpl.patches.Patch(color=c, label=k) for k, c in _FAMILY_COLOR.items()
               if k in {_FAMILY.get(n, 'other') for n in order}]
    a.legend(handles=handles, frameon=False, loc='lower right')
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_significant_fraction(cv_results, alpha=0.05, ax=None, out_path=None):
    """Fraction of neurons whose held-out delta_r2 beats its own permutation null.

    The dashed line is `alpha` -- the fraction expected by chance. A bar at the line means
    that regressor is indistinguishable from noise in PFC, which is a result, not a gap.
    """
    _style()
    long = to_long(cv_results)
    if long['p'].isna().all():
        raise ValueError('no p_cv in these results — refit with cv_n_perm > 0')
    order = regressor_names(cv_results)
    per_mouse = per_mouse_stat(long.assign(sig=long['p'] < alpha), 'sig', 'mean')
    means = per_mouse.groupby('regressor')['sig'].mean().reindex(order)

    fig, a = (plt.subplots(figsize=(4.2, 3.2)) if ax is None else (ax.figure, ax))
    y = np.arange(len(order))
    a.barh(y, means.values,
           color=[_FAMILY_COLOR[_FAMILY.get(n, 'other')] for n in order],
           edgecolor='none', height=0.7, zorder=2)
    for i, g in enumerate(order):
        v = per_mouse.loc[per_mouse.regressor == g, 'sig'].values
        a.scatter(v, np.full(len(v), i), s=8, color=C_CAVIAR, alpha=0.7,
                  zorder=3, clip_on=False)
    a.axvline(alpha, color=C_NULL, lw=0.8, ls='--', zorder=1)
    a.text(alpha, len(order) - 0.3, f' chance ({alpha:g})', color=C_NULL,
           va='top', ha='left')
    a.set_yticks(y)
    a.set_yticklabels(order)
    a.invert_yaxis()
    a.set_xlabel(f'fraction of neurons with p < {alpha:g}')
    a.set_title('significant against the within-session shift null (tests CPD)', loc='left')
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_neuron_heatmap(cv_results, sort_by='place', value='delta_r2_cv',
                        max_neurons=400, ax=None, out_path=None):
    """Neurons x regressors delta_r2, to see whether PFC cells are mixed or specialised.

    Diverging `RdBu_r` centered at zero, because delta_r2 is signed: a negative value means
    dropping that regressor IMPROVED held-out prediction, i.e. it was fitting noise for that
    neuron. Symmetric limits keep white at exactly zero.
    """
    _style()
    order = regressor_names(cv_results, value)
    M = np.vstack([np.column_stack([np.asarray(resolve_value(r, value)[g], dtype=float)
                                    for g in order])
                   for r in cv_results.values()])
    if sort_by in order:
        M = M[np.argsort(-M[:, order.index(sort_by)])]
    if len(M) > max_neurons:
        idx = np.linspace(0, len(M) - 1, max_neurons).astype(int)
        M = M[idx]

    v = np.nanpercentile(np.abs(M), 99)
    fig, a = (plt.subplots(figsize=(4.0, 4.4)) if ax is None else (ax.figure, ax))
    im = a.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v,
                  interpolation='none')
    a.set_xticks(range(len(order)))
    a.set_xticklabels(order, rotation=90)
    a.set_ylabel(f'neuron (sorted by {sort_by}, n={len(M)})')
    a.set_title('per-neuron unique variance', loc='left')
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    cb.set_label(_VALUE_LABEL.get(value, value))
    cb.outline.set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_mixed_selectivity(cv_results, alpha=0.05, ax=None, out_path=None):
    """How many regressors does a single PFC neuron significantly encode?

    A population of specialists piles up at 1; a mixed-selectivity population spreads right.
    The grey bars are the same count computed on the permutation null, so the excess over
    chance is visible rather than assumed.
    """
    _style()
    long = to_long(cv_results)
    if long['p'].isna().all():
        raise ValueError('no p_cv in these results — refit with cv_n_perm > 0')
    n_sig = (long.assign(sig=long['p'] < alpha)
             .groupby(['recday', 'neuron'])['sig'].sum().values)
    n_reg = len(regressor_names(cv_results))
    # Chance: each regressor independently significant with probability alpha.
    from scipy.stats import binom
    bins = np.arange(-0.5, n_reg + 1.5)
    exp = binom.pmf(np.arange(n_reg + 1), n_reg, alpha) * len(n_sig)

    fig, a = (plt.subplots(figsize=(4.0, 2.6)) if ax is None else (ax.figure, ax))
    a.hist(n_sig, bins=bins, color=C_SIGNAL, edgecolor='none', label='observed', zorder=2)
    a.step(np.arange(n_reg + 1), exp, where='mid', color=C_NULL, lw=1.0,
           label=f'chance (binomial, p={alpha:g})', zorder=3)
    a.set_xlabel(f'significant regressors per neuron (of {n_reg})')
    a.set_ylabel('neurons')
    a.set_title(f'median {np.median(n_sig):.0f}, '
                f'{np.mean(n_sig == 0):.0%} encode nothing', loc='left')
    a.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def summary_table(cv_results, alpha=0.05, value='delta_r2_cv'):
    """One row per regressor: delta_r2 and significant fraction, pooled over mice."""
    long = to_long(cv_results, value)
    tagged = long.assign(sig=long['p'] < alpha)
    pm = per_mouse_stat(tagged, 'value', 'median').merge(
        per_mouse_stat(tagged, 'sig', 'mean'), on=['mouse', 'regressor'])
    pm = pm.rename(columns={'value': _VALUE_SHORT.get(value, value)})
    vcol = _VALUE_SHORT.get(value, value)
    out = (pm.groupby('regressor')
           .agg(**{vcol: (vcol, 'mean'), f'{vcol}_sd': (vcol, 'std'),
                   'frac_sig': ('sig', 'mean'), 'n_mice': ('mouse', 'nunique')})
           .reindex(regressor_names(cv_results, value)))
    out['family'] = [_FAMILY.get(n, 'other') for n in out.index]
    return out


def plot_cpd_vs_delta_r2(cv_results, ax=None, out_path=None):
    """The two effect sizes against each other, per neuron per regressor.

    They share a numerator (held-out dRSS) and differ only in denominator: delta_r2 divides
    by TSS, the same for every regressor, while CPD divides by that group's own reduced-model
    RSS. So CPD >= delta_r2 always, and the gap grows with how well the reduced model still
    fits -- i.e. CPD inflates a weak regressor precisely when a dominant one remains in the
    model. Place dominates here, so the inflation lands on everything else.

    The dashed line is y = x. Distance above it is the inflation, per point.
    """
    _style()
    a_long = to_long(cv_results, 'delta_r2_cv').rename(columns={'value': 'delta_r2'})
    b_long = to_long(cv_results, 'cpd_cv')[['recday', 'neuron', 'regressor', 'value']]
    m = a_long.merge(b_long.rename(columns={'value': 'cpd'}),
                     on=['recday', 'neuron', 'regressor'])

    fig, a = (plt.subplots(figsize=(4.0, 3.6)) if ax is None else (ax.figure, ax))
    order = regressor_names(cv_results)
    for g in order:
        sub = m[m.regressor == g]
        a.scatter(sub['delta_r2'], sub['cpd'], s=3, alpha=0.25,
                  color=_FAMILY_COLOR[_FAMILY.get(g, 'other')], edgecolor='none',
                  rasterized=True)
    lo = float(np.nanpercentile(m[['delta_r2', 'cpd']].values, 0.5))
    hi = float(np.nanpercentile(m[['delta_r2', 'cpd']].values, 99.5))
    a.plot([lo, hi], [lo, hi], color=C_CAVIAR, lw=0.8, ls='--', zorder=3)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel(r'$\Delta R^2$ (denominator: TSS)')
    a.set_ylabel('CPD (denominator: reduced-model RSS)')
    ratio = (m['cpd'] / m['delta_r2']).replace([np.inf, -np.inf], np.nan)
    a.set_title(f'median CPD/$\\Delta R^2$ = {np.nanmedian(ratio):.2f}x', loc='left')
    handles = [mpl.patches.Patch(color=c, label=k) for k, c in _FAMILY_COLOR.items()
               if k in {_FAMILY.get(n, 'other') for n in order}]
    a.legend(handles=handles, frameon=False, loc='lower right', fontsize=7)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_bias_correction(cv_results, value='delta_r2_cv', n_cols=None, ax=None,
                         out_path=None):
    """Raw vs bias-corrected effect size, with each regressor's column count.

    The parameter penalty is what this shows: a regressor's held-out score is pulled down by
    ~k*sigma^2 for its k design columns, so `head_direction` (35 cols) is penalised most and
    `poke_rewarded` (1 col) barely at all. Arrows run from raw to corrected; long arrows are
    big blocks.

    Pass `n_cols={regressor: n}` to annotate; without it the arrows still show the shift.
    """
    _style()
    corrected = ('delta_r2_corrected' if value.startswith('delta_r2') else 'cpd_corrected')
    raw = per_mouse_stat(to_long(cv_results, value), 'value', 'median')
    cor = per_mouse_stat(to_long(cv_results, corrected), 'value', 'median')
    a_ = raw.groupby('regressor')['value'].mean()
    b_ = cor.groupby('regressor')['value'].mean()
    order = b_.sort_values(ascending=False).index.tolist()

    fig, a = (plt.subplots(figsize=(4.6, 3.6)) if ax is None else (ax.figure, ax))
    y = np.arange(len(order))
    for i, g in enumerate(order):
        col = _FAMILY_COLOR[_FAMILY.get(g, 'other')]
        a.annotate('', xy=(b_[g], i), xytext=(a_[g], i),
                   arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
        a.scatter([a_[g]], [i], s=14, facecolor='none', edgecolor=col, zorder=3)
        a.scatter([b_[g]], [i], s=22, color=col, zorder=4)
    a.axvline(0, color=C_CAVIAR, lw=0.8, ls='--')
    a.set_yticks(y)
    labels = ([f'{g}  ({n_cols[g]})' for g in order] if n_cols else order)
    a.set_yticklabels(labels)
    a.invert_yaxis()
    a.set_xlabel(_VALUE_LABEL.get(value, value) + '   →  bias-corrected')
    a.set_title('open = raw, filled = null-corrected'
                + ('  (n columns in brackets)' if n_cols else ''), loc='left')
    fig.tight_layout()
    _save(fig, out_path)
    return fig
