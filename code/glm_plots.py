"""Plots for the production GLM: the dataset on its own terms, and split by region.

The first half of this module is SHARED WITH `mFC_data/code/pfc_glm_plots.py` and must stay
byte-identical to it -- the two datasets have to be plotted the same way for any comparison
between them to mean anything. The second half is LEC-only: the region splits, which PFC
cannot have because it has no `unit_regions`.

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

import anatomy_split as asp

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
    joint = {'time_any', 'distance_any', 'gp_any', 'progress_or_time'}
    names = [n for n in names if n not in joint]
    order = {'space': 0, 'task': 1, 'time': 2, 'motor': 3, 'other': 4}
    return sorted(names, key=lambda n: (order[_FAMILY.get(n, 'other')], n))


def p_values(r, p_stat='cpd', null='freedman_lane'):
    """`{group: p}` for one recday: the permutation p of `p_stat` under `null`, or None.

    Reads `p_{null}__{p_stat}` ('cpd' | 'delta_r2'); results fitted before the multi-null
    refactor only carry `p_cv` (CPD under the first requested null), which is the fallback --
    and in that case a request for `'delta_r2'` is answered with the CPD p, so say which
    statistic a figure tests by reading the key that was actually used (`p_key`).
    """
    key = f'p_{null}__{p_stat}'
    if key in r:
        return r[key]
    return r.get('p_cv')


def p_key(cv_results, p_stat='cpd', null='freedman_lane'):
    """Which stored key `p_values` will read for these results -- for figure titles."""
    first = next(iter(cv_results.values()))
    key = f'p_{null}__{p_stat}'
    return key if key in first else ('p_cv' if 'p_cv' in first else None)


def to_long(cv_results, value='delta_r2_cv', p_stat='cpd', null='freedman_lane'):
    """Per-neuron long-format table: recday, mouse, neuron, regressor, value, p.

    `value` and `p` are independent choices: `value` is the effect size shown, `p` the
    permutation p-value of `p_stat` under `null` (see `p_values`). Pass `p_stat='delta_r2'`
    when a figure shows delta_r2 significance, so the bar and the test agree.
    """
    rows = []
    for rd, r in cv_results.items():
        mouse = rd.split('_')[0]
        vals = resolve_value(r, value)
        pv = p_values(r, p_stat, null)
        for g in regressor_names(cv_results, value):
            v = np.asarray(vals[g], dtype=float)
            p = (np.asarray(pv[g], dtype=float) if pv is not None and g in pv
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


def plot_significant_fraction(cv_results, alpha=0.05, ax=None, out_path=None,
                              p_stat='cpd', null='freedman_lane'):
    """Fraction of neurons whose held-out `p_stat` beats its own permutation null.

    The dashed line is `alpha` -- the fraction expected by chance. A bar at the line means
    that regressor is indistinguishable from noise, which is a result, not a gap. The null is
    Freedman-Lane by default, i.e. "g adds nothing beyond the OTHER regressors in this
    design" -- so the fraction depends on what else is in the model, and a reduced design
    is expected to score higher than the full one for the same data (GLM_V3.md, caveat 5).
    """
    _style()
    long = to_long(cv_results, p_stat=p_stat, null=null)
    if long['p'].isna().all():
        raise ValueError('no p-values in these results — refit with cv_n_perm > 0')
    key = p_key(cv_results, p_stat, null)
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
    a.set_title(f'significant against the {null} null ({key})', loc='left')
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


def plot_mixed_selectivity(cv_results, alpha=0.05, ax=None, out_path=None,
                           p_stat='cpd', null='freedman_lane'):
    """How many regressors does a single neuron significantly encode?

    A population of specialists piles up at 1; a mixed-selectivity population spreads right.
    The grey bars are the same count computed on the permutation null, so the excess over
    chance is visible rather than assumed.
    """
    _style()
    long = to_long(cv_results, p_stat=p_stat, null=null)
    if long['p'].isna().all():
        raise ValueError('no p-values in these results — refit with cv_n_perm > 0')
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


def summary_table(cv_results, alpha=0.05, value='delta_r2_cv', p_stat='cpd',
                  null='freedman_lane'):
    """One row per regressor: effect size and significant fraction, pooled over mice.

    `frac_sig` tests `p_stat` under `null` (default CPD, Freedman-Lane); pass
    `p_stat='delta_r2'` to test the statistic shown when `value` is a delta_r2 variant.
    """
    long = to_long(cv_results, value, p_stat=p_stat, null=null)
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


# ---------------------------------------------------------------------------
# V3: comparing fits of the same data (GLM_V3.md). Reduced designs, codings, leg caps.
# ---------------------------------------------------------------------------

def _per_mouse_matrix(per_mouse_frames, mice, regressor=None, col='value'):
    """(n_mice, n_fits) matrix of one per-mouse statistic across an ordered list of frames."""
    out = np.full((len(mice), len(per_mouse_frames)), np.nan)
    for j, pm in enumerate(per_mouse_frames):
        sub = pm if regressor is None else pm[pm.regressor == regressor]
        s = sub.set_index('mouse')[col]
        for i, m in enumerate(mice):
            if m in s.index:
                out[i, j] = float(s.loc[m])
    return out


def _r2_per_mouse(cv_results, statistic='median'):
    """Recday `statistic` of r2_cv -> mean over that mouse's recdays; DataFrame(mouse, value)."""
    rows = [{'recday': rd, 'mouse': rd.split('_')[0],
             'value': float(getattr(np, 'nan' + statistic)(np.asarray(r['r2_cv'], float)))}
            for rd, r in cv_results.items()]
    df = pd.DataFrame(rows)
    return df.groupby('mouse')['value'].mean().reset_index()


def plot_paired_fits(fits, regressors=None, value='delta_r2_cv', out_path=None,
                     statistic='median'):
    """Per-mouse effect sizes across two or more fits, each mouse joined across the fits.

    `fits` is an ordered `{label: cv_results}`. One panel per regressor shared by every fit
    (or `regressors`), plus an r2_cv panel. A point is one mouse (recday `statistic` -> mouse
    mean, `per_mouse_stat`), the grey line follows that mouse across the fits, the bar is the
    mean over mice.

    This is the figure for the one-factor comparisons of GLM_V3.md: the same regressors under
    a different regressor SET (full-16 -> core-5: a bigger CPD here is arithmetic, not
    evidence -- shared variance is reassigned to the survivors), a different time_from_reward
    CODING, a different leg CAP, or with/without the absolute-time regressor. Read the shift
    and the r2_cv panel together: a reduced design has bigger CPDs AND a worse fit.
    """
    _style()
    labels = list(fits)
    if regressors is None:
        common = None
        for cv in fits.values():
            names = set(regressor_names(cv, value))
            common = names if common is None else common & names
        regressors = [g for g in regressor_names(next(iter(fits.values())), value)
                      if g in common]
    per = [per_mouse_stat(to_long(cv, value), 'value', statistic) for cv in fits.values()]
    r2 = [_r2_per_mouse(cv, statistic) for cv in fits.values()]
    mice = sorted(set().union(*[set(p.mouse) for p in per]))
    n = len(regressors) + 1
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n + 0.6, 2.9))
    axes = np.atleast_1d(axes)
    x = np.arange(len(labels))
    for k, g in enumerate(regressors):
        a = axes[k]
        vals = _per_mouse_matrix(per, mice, g)
        for i in range(len(mice)):
            a.plot(x, vals[i], color=C_STONE, lw=0.8, zorder=2)
            a.scatter(x, vals[i], s=12, color=_FAMILY_COLOR[_FAMILY.get(g, 'other')],
                      zorder=3, clip_on=False)
        a.scatter(x, np.nanmean(vals, axis=0), s=70, marker='_', color=C_CAVIAR, zorder=4)
        a.axhline(0, color=C_CAVIAR, lw=0.6, ls='--')
        a.set_title(g, loc='left')
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=45, ha='right')
        if k == 0:
            a.set_ylabel(_VALUE_LABEL.get(value, value))
    a = axes[-1]
    vals = _per_mouse_matrix(r2, mice)
    for i in range(len(mice)):
        a.plot(x, vals[i], color=C_STONE, lw=0.8, zorder=2)
        a.scatter(x, vals[i], s=12, color=C_NEUTRAL, zorder=3, clip_on=False)
    a.scatter(x, np.nanmean(vals, axis=0), s=70, marker='_', color=C_CAVIAR, zorder=4)
    a.axhline(0, color=C_CAVIAR, lw=0.6, ls='--')
    a.set_title('$R^2_{cv}$ (model fit)', loc='left')
    a.set_xticks(x)
    a.set_xticklabels(labels, rotation=45, ha='right')
    fig.suptitle(f'{len(mice)} mice — points are mice (recday {statistic} → mouse mean), '
                 f'bars the mean over mice', x=0.01, ha='left', fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_factorial_grid(arms, regressor, value='delta_r2_cv', out_path=None,
                        statistic='median'):
    """The leg-cap x time_from_reward-coding 2x2 for ONE regressor, per mouse.

    `arms` is `{(cap_s, scheme): cv_results}`. One row per cap, the codings side by side on
    the x-axis, a shared y-axis: the coding effect is read along a row, the cap effect down a
    column. Points are mice (recday `statistic` -> mouse mean), joined across codings; bars
    are the mean over mice. Remember the 30 s arms sit on ~80 % of the 60 s rows (LEC), so a
    cap effect is partly a row-set effect (GLM_V3.md caveat 6).
    """
    _style()
    caps = sorted({c for c, _ in arms})
    schemes = sorted({s for _, s in arms})
    per = {k: per_mouse_stat(to_long(cv, value), 'value', statistic) for k, cv in arms.items()}
    mice = sorted(set().union(*[set(p.mouse) for p in per.values()]))
    fig, axes = plt.subplots(len(caps), 1, figsize=(2.6, 1.8 * len(caps) + 0.7),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(schemes))
    col = _FAMILY_COLOR[_FAMILY.get(regressor, 'other')]
    for r_, cap in enumerate(caps):
        a = axes[r_]
        frames = [per[(cap, s)] for s in schemes if (cap, s) in per]
        vals = _per_mouse_matrix(frames, mice, regressor)
        for i in range(len(mice)):
            a.plot(x[:vals.shape[1]], vals[i], color=C_STONE, lw=0.8, zorder=2)
            a.scatter(x[:vals.shape[1]], vals[i], s=12, color=col, zorder=3, clip_on=False)
        a.scatter(x[:vals.shape[1]], np.nanmean(vals, axis=0), s=70, marker='_',
                  color=C_CAVIAR, zorder=4)
        a.axhline(0, color=C_CAVIAR, lw=0.6, ls='--')
        a.set_ylabel(f'leg cap {cap:g} s')
        a.set_xticks(x)
        a.set_xticklabels([f'tfr {s}' for s in schemes])
    axes[0].set_title(f'{regressor} — {_VALUE_SHORT.get(value, value)}, {len(mice)} mice',
                      loc='left')
    fig.tight_layout()
    _save(fig, out_path)
    return fig


#: Axis labels for the binned regressors `plot_beta_profile` can draw. `time_from_reward` is
#: handled specially (its edges come from the stored occupancy, in seconds).
_PROFILE_AXES = {
    'goal_progress': ('goal progress (fraction of leg, lower bin edge)',
                      lambda n: [f'{k / n:.1f}' for k in range(n)]),
    'goal_progress_distance': ('goal progress by distance (fraction, lower bin edge)',
                               lambda n: [f'{k / n:.1f}' for k in range(n)]),
    'progress_since_A': ('progress since A (fraction of loop, lower bin edge)',
                         lambda n: [f'{k / n:.1f}' for k in range(n)]),
}


def plot_beta_profile(glm_results, cv_results, col_idx, regressor, *, region_of=None,
                      groups=None, colors=None, alpha=0.05, null='freedman_lane',
                      bottom='auto', center='reference', p_stat='cpd', out_path=None):
    """Where along a binned regressor do the neurons tuned to it load.

    Top: for every neuron whose `regressor` beats its permutation null (p < `alpha`, `p_stat`
    under `null`), the reference-coded beta vector over the regressor's bins, scaled to unit
    max|beta| and averaged (mean +/- s.e.m.) per label -- a SHAPE average, so a high-rate neuron
    does not dominate it.

    WHAT THE REFERENCE BIN MEANS. Under reference coding bin 0 has no column: its level is absorbed
    into the intercept, and each plotted beta_k is the difference in expected firing between bin k
    and bin 0 with every other regressor held fixed. So bin 0 sits at exactly 0 for every neuron BY
    CONSTRUCTION (its s.e.m. band vanishes), a curve above zero means "fires more in bin k than in
    bin 0", and the absolute offset carries no information -- only the shape across bins does.
    `center='reference'` draws exactly that. `center='mean'` subtracts each neuron's mean beta
    over all bins (bin 0 included) before normalising, so the curve is the deviation from that
    neuron's average across the regressor and the reference bin is no longer privileged; the
    shape is identical, the baseline moves. For `goal_progress` bin 0 is the first tenth of the
    leg -- the reward-consumption period -- so the reference-centred curve reads as "relative to
    consumption".

    Bottom, `bottom=`: `'occupancy'` -- the fraction of design rows per bin averaged over recdays,
    from `cv_results[rd]['bin_occupancy'][regressor]` (stored for the regressors
    `binning_overrides` touched, i.e. `time_from_reward` in the V3 arms), with the out-of-range
    count; `'peaks'` -- the fraction of the same neurons whose PEAK falls in each bin, where the
    peak is the bin of highest firing, bin 0 included (the signed argmax of the beta vector; a
    neuron whose every beta is negative fires most in bin 0 and peaks there -- the same argmax
    under either `center`); `'auto'` -- occupancy when it is stored, peaks otherwise
    (`goal_progress` has ~10 % of rows in every bin by construction, so its occupancy panel would
    say nothing).

    Read with GLM_V3.md caveats 3 and 4 for `time_from_reward`: with the pokes out of the model a
    bin-0 peak is reward consumption, not elapsed time; a late-bin peak sits on the sparsest bins,
    reached by the fewest legs. For `goal_progress` the peaks panel is the population's within-leg
    tuning landscape: where along the leg do the phase-significant neurons fire most.

    `col_idx`: the regressor's design columns, from
    `glm_analysis_v3._resolve_regressor_groups(regressors, parameterization='reference_coded',
    n_cols_override=...)`. `region_of={recday: labels per neuron}` splits by region (LEC); omit it
    to pool under one curve.
    """
    _style()
    if center not in ('reference', 'mean'):
        raise ValueError(f"center must be 'reference' or 'mean', got {center!r}")
    name = regressor
    profiles, peaks, occ, edges_s = {}, {}, [], []
    n_out = n_rows = 0
    scheme = None
    for rd, r in cv_results.items():
        if rd not in glm_results or not glm_results[rd]:
            continue
        pv = p_values(r, p_stat, null)
        p = np.asarray(pv[name], float) if pv is not None and name in pv else None
        bo = (r.get('bin_occupancy') or {}).get(name)
        if bo is not None:
            occ.append(bo['frac_rows'])
            scheme = bo.get('scheme', scheme)
            if bo.get('edges_s') is not None:
                edges_s.append(bo['edges_s'])
            n_out += int(bo.get('n_out_of_range', 0))
            n_rows += int(bo.get('n_rows', 0))
        keys = sorted(glm_results[rd])
        labels = (np.asarray(region_of[rd]) if region_of is not None
                  else np.array(['all'] * len(keys)))
        for k, nrn in enumerate(keys):
            if p is not None and not (p[k] < alpha):
                continue
            params = np.asarray(glm_results[rd][nrn], float)
            beta = np.concatenate([[0.0], params[np.asarray(col_idx)]])
            if center == 'mean':
                beta = beta - np.nanmean(beta)
            m = np.nanmax(np.abs(beta))
            if not np.isfinite(m) or m == 0:
                continue
            profiles.setdefault(str(labels[k]), []).append(beta / m)
            peaks.setdefault(str(labels[k]), []).append(int(np.argmax(beta)))
    n_bins = len(col_idx) + 1
    if name == 'time_from_reward' and edges_s:
        e = np.nanmedian(np.asarray(edges_s, float), axis=0)
        lower = np.concatenate([[0.0], e[1:-1]])
        xlab = [f'{lo:.0f}' for lo in lower]
        xname = 'time from reward (s, lower bin edge' + \
                (', median over recdays)' if scheme == 'decile' else ')')
    elif name in _PROFILE_AXES:
        xname, fn = _PROFILE_AXES[name]
        xlab = fn(n_bins)
    else:
        xlab = [str(i) for i in range(n_bins)]
        xname = f'{name} bin'
    x = np.arange(n_bins)
    order = [g for g in (groups or sorted(profiles)) if g in profiles]
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.6, 3.6), sharex=True,
                               gridspec_kw={'height_ratios': [2.2, 1]})
    for g in order:
        Pm = np.asarray(profiles[g])
        mu = Pm.mean(axis=0)
        se = Pm.std(axis=0, ddof=1) / np.sqrt(len(Pm)) if len(Pm) > 1 else np.zeros(n_bins)
        c = (colors or {}).get(g)
        a.plot(x, mu, color=c, lw=1.2, label=f'{g} (n={len(Pm)})', zorder=3)
        a.fill_between(x, mu - se, mu + se, color=c, alpha=0.2, lw=0, zorder=2)
    a.axhline(0, color=C_CAVIAR, lw=0.6, ls='--')
    if center == 'reference':
        a.set_ylabel(r'$\beta$ / max|$\beta$|   (0 = bin 0, the reference)')
        a.set_title(f'{name}-significant neurons (p<{alpha:g}, {null}); '
                    f'each bin vs bin 0, which is 0 by construction', loc='left')
    else:
        a.set_ylabel(r'($\beta - \bar\beta$) / max|$\beta - \bar\beta$|')
        a.set_title(f'{name}-significant neurons (p<{alpha:g}, {null}); '
                    f'deviation from each neuron\'s mean over bins', loc='left')
    a.legend(frameon=False, fontsize=6)
    mode = bottom if bottom != 'auto' else ('occupancy' if occ else 'peaks')
    if mode == 'occupancy' and occ:
        occ_m = np.asarray(occ, float).mean(axis=0)
        b.bar(x, occ_m, color=C_NEUTRAL, edgecolor='none')
        b.set_ylabel('rows')
        b.set_title(f'occupancy ({scheme}); {n_out} of {n_rows} rows beyond the range '
                    f'coded as bin 0', loc='left', fontsize=7)
    else:
        wdt = 0.8 / max(len(order), 1)
        for i, g in enumerate(order):
            h = np.bincount(np.asarray(peaks[g]), minlength=n_bins)[:n_bins] / max(len(peaks[g]), 1)
            b.bar(x + (i - (len(order) - 1) / 2) * wdt, h, width=wdt,
                  color=(colors or {}).get(g, C_NEUTRAL), edgecolor='none')
        b.axhline(1.0 / n_bins, color=C_NULL, lw=0.6, ls='--')
        b.set_ylabel('peak fraction')
        b.set_title('bin of highest firing per neuron, bin 0 included (dashed: uniform)',
                    loc='left', fontsize=7)
    b.set_xticks(x)
    b.set_xticklabels(xlab)
    b.set_xlabel(xname)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_tfr_beta_profile(glm_results, cv_results, col_idx, *, region_of=None, groups=None,
                          colors=None, alpha=0.05, null='freedman_lane', out_path=None):
    """`plot_beta_profile` for `time_from_reward` with the occupancy panel (GLM_V3.md 9.9)."""
    return plot_beta_profile(glm_results, cv_results, col_idx, 'time_from_reward',
                             region_of=region_of, groups=groups, colors=colors, alpha=alpha,
                             null=null, bottom='occupancy', out_path=out_path)


# ===========================================================================
# LEC only, below this line: the region splits. PFC has no `unit_regions`.
# ===========================================================================

def join_region(cv_results, regressor, unit_regions=None, value='delta_r2_cv', *,
                p_stat=None, null='freedman_lane'):
    """Per-neuron values for one regressor, joined to anatomy.

    The join is positional and asserted -- `anatomy_split.join_regions` refuses a result
    whose length disagrees with `unit_regions`, because row k of one must be row k of the
    other and a mismatch would misalign every neuron after the first.

    With `p_stat` ('cpd' | 'delta_r2') the joined column `v` holds that statistic's
    permutation p-value under `null` instead of the effect size (see `p_values`).
    """
    unit_regions = unit_regions if unit_regions is not None else asp.load_unit_regions()
    per_recday = {}
    for rd, r in cv_results.items():
        if p_stat is not None:
            pv = p_values(r, p_stat, null)
            if pv is not None and regressor in pv:
                per_recday[rd] = np.asarray(pv[regressor], dtype=float)
        else:
            vals = resolve_value(r, value)
            if regressor in vals:
                per_recday[rd] = np.asarray(vals[regressor], dtype=float)
    return asp.join_regions(per_recday, unit_regions, value_name='v', strict=True)


#: Colour for a non-anatomical column appended to a region figure (another dataset).
C_EXTRA = '#5B5B8F'


def plot_regressor_ranking_by_region(cv_results, unit_regions=None, groups=None,
                                     value='delta_r2_cv', out_path=None, extra=None):
    """Effect size per regressor, one bar group per brain region.

    Every region is shown on its own before any contrast is drawn, and each bar is the mean
    over MICE of that mouse's median neuron -- recdays of one mouse share a probe and a
    brain, so they are repeated measures, not replicates.

    `extra={'PFC': cv_pfc}` appends non-anatomical columns computed the same way
    (`per_mouse_stat`) -- licensed only when the other fit carries the IDENTICAL design, which
    the V3 reduced designs do (nothing in them is unavailable to PFC).
    """
    _style()
    groups = groups or asp.ANALYSIS_GROUPS
    order = regressor_names(cv_results, value)
    ur = unit_regions if unit_regions is not None else asp.load_unit_regions()

    mat = pd.DataFrame(index=order, columns=groups, dtype=float)
    for g in order:
        j = join_region(cv_results, g, ur, value)
        pm = asp.per_mouse_effect(j, 'v', statistic='median', groups=groups)
        if len(pm):
            mat.loc[g] = pm.groupby('group')['effect'].mean().reindex(groups).values
    cols = list(groups)
    for lab, cv in (extra or {}).items():
        pm = per_mouse_stat(to_long(cv, value), 'value', 'median')
        mat[lab] = pm.groupby('regressor')['value'].mean().reindex(order).values
        cols.append(lab)

    fig, a = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(order))
    h = 0.8 / len(cols)
    for k, grp in enumerate(cols):
        a.barh(y + k * h - 0.4 + h / 2, mat[grp].values, height=h,
               color=asp.REGION_COLORS.get(grp, C_EXTRA if grp in (extra or {}) else '#B4B2A9'),
               edgecolor='none', label=grp, zorder=2)
    a.axvline(0, color=C_CAVIAR, lw=0.8)
    a.set_yticks(y)
    a.set_yticklabels(order)
    a.invert_yaxis()
    a.set_xlabel(_VALUE_LABEL.get(value, value))
    a.legend(frameon=False, fontsize=7, ncol=1, loc='lower right')
    a.set_title(f'LEC by region — mice averaged within region '
                f'({_VALUE_SHORT.get(value, value)})'
                + (f'; {", ".join(extra)} = other dataset, same design' if extra else ''),
                loc='left')
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_significant_fraction_by_region(cv_results, unit_regions=None, groups=None,
                                        alpha=0.05, p_stat='cpd', null='freedman_lane',
                                        out_path=None, extra=None):
    """Fraction of neurons with p < alpha, per regressor, one bar group per region.

    Same aggregation as `plot_regressor_ranking_by_region` (recday mean -> mouse -> mean over
    mice). Significance is RATE-DEPENDENT and region is confounded with rate (SUB/ProS fires
    ~3x ENTl-deep), so a regional gap here is not interpretable until it survives
    `anatomy_split.rate_match` -- this panel is descriptive, the primary-contrast cell in the
    notebook is where the inference happens.
    """
    _style()
    groups = groups or asp.ANALYSIS_GROUPS
    order = regressor_names(cv_results)
    ur = unit_regions if unit_regions is not None else asp.load_unit_regions()
    key = p_key(cv_results, p_stat, null)

    mat = pd.DataFrame(index=order, columns=groups, dtype=float)
    for g in order:
        j = join_region(cv_results, g, ur, p_stat=p_stat, null=null)
        j = j.assign(sig=(j['v'] < alpha).astype(float))
        pm = asp.per_mouse_effect(j, 'sig', statistic='mean', groups=groups)
        if len(pm):
            mat.loc[g] = pm.groupby('group')['effect'].mean().reindex(groups).values
    cols = list(groups)
    for lab, cv in (extra or {}).items():
        long = to_long(cv, p_stat=p_stat, null=null)
        pm = per_mouse_stat(long.assign(sig=(long['p'] < alpha).astype(float)), 'sig', 'mean')
        mat[lab] = pm.groupby('regressor')['sig'].mean().reindex(order).values
        cols.append(lab)

    fig, a = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(order))
    h = 0.8 / len(cols)
    for k, grp in enumerate(cols):
        a.barh(y + k * h - 0.4 + h / 2, mat[grp].values, height=h,
               color=asp.REGION_COLORS.get(grp, C_EXTRA if grp in (extra or {}) else '#B4B2A9'),
               edgecolor='none', label=grp, zorder=2)
    a.axvline(alpha, color=C_NULL, lw=0.8, ls='--', zorder=1)
    a.set_yticks(y)
    a.set_yticklabels(order)
    a.invert_yaxis()
    a.set_xlabel(f'fraction of neurons with p < {alpha:g}  ({key})')
    a.legend(frameon=False, fontsize=7, ncol=1, loc='lower right')
    a.set_title(f'LEC by region — significant fraction, {null} null; '
                f'descriptive (rate-confounded)', loc='left')
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_region_report(cv_results, regressor, unit_regions=None, groups=None,
                       value='delta_r2_cv', ax=None, out_path=None):
    """One regressor, every region on its own, with the individual mice shown.

    This is the panel that has to come before any contrast: a region with n=1 mouse still
    gets a point, labelled descriptive, so a "clean" contrast between two single-mouse arms
    cannot pass unnoticed.
    """
    _style()
    groups = groups or asp.ANALYSIS_GROUPS
    ur = unit_regions if unit_regions is not None else asp.load_unit_regions()
    j = join_region(cv_results, regressor, ur, value)
    rep = asp.per_region_report(j, 'v', statistic='median', groups=groups)
    pm = asp.per_mouse_effect(j, 'v', statistic='median', groups=groups)

    fig, a = (plt.subplots(figsize=(4.4, 3.0)) if ax is None else (ax.figure, ax))
    for i, grp in enumerate(rep['group']):
        v = pm.loc[pm.group == grp, 'effect'].values
        col = asp.REGION_COLORS.get(grp, '#B4B2A9')
        a.scatter(np.full(len(v), i), v, s=22, color=col, zorder=3, clip_on=False)
        row = rep[rep.group == grp].iloc[0]
        a.scatter([i], [row['effect_pooled_over_mice']], s=70, marker='_',
                  color=C_CAVIAR, zorder=4)
        if np.isfinite(row['ci_lo']):
            a.plot([i, i], [row['ci_lo'], row['ci_hi']], color=C_STONE, lw=1.2, zorder=2)
        a.annotate(f"{int(row['n_mice'])}", (i, a.get_ylim()[0]), ha='center',
                   va='bottom', fontsize=6, color=C_NEUTRAL)
    a.axhline(0, color=C_CAVIAR, lw=0.8, ls='--')
    a.set_xticks(range(len(rep)))
    a.set_xticklabels(rep['group'], rotation=45, ha='right')
    a.set_ylabel(_VALUE_LABEL.get(value, value))
    a.set_title(f'{regressor} ({_VALUE_SHORT.get(value, value)}) — points are mice, '
                f'bar is the mouse-level mean', loc='left')
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_quality_by_region(unit_regions=None, meta_path=None, ax=None, out_path=None):
    """Firing rate per region, published BEFORE any tuning result.

    Region is derived from a unit's max-amplitude channel, and depth drives amplitude,
    isolation and yield -- so a regional difference in any rate-dependent statistic can be a
    recording-quality gradient wearing an anatomical label. On this cohort that is not a
    small effect: SUB/ProS fires ~3x faster than ENTl-deep, and the direction is not even
    consistent across mice (ly06 inverts it).
    """
    _style()
    import w0_gates
    ur = unit_regions if unit_regions is not None else asp.load_unit_regions()
    joined, _ = w0_gates.gate_quality_by_region(
        ur, meta_path or w0_gates.CLUSTER_META)

    groups = [g for g in asp.ANALYSIS_GROUPS if g in set(joined['group'])]
    fig, a = (plt.subplots(figsize=(4.2, 2.8)) if ax is None else (ax.figure, ax))
    data = [joined.loc[joined.group == g, 'mean_rate_hz'].dropna().values for g in groups]
    bp = a.boxplot(data, positions=range(len(groups)), widths=0.6, showfliers=False,
                   patch_artist=True, medianprops=dict(color=C_CAVIAR, lw=1.2))
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(asp.REGION_COLORS.get(g, '#B4B2A9'))
        patch.set_edgecolor('none')
    a.set_xticks(range(len(groups)))
    a.set_xticklabels(groups, rotation=45, ha='right')
    a.set_ylabel('mean firing rate (Hz)')
    a.set_title('unit quality by region — read this before any tuning result', loc='left')
    fig.tight_layout()
    _save(fig, out_path)
    return fig
