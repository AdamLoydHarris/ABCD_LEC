"""Figures and region statistics for W3 -- remapping and coherence, split by region.

Every panel here obeys three rules from `docs/handoff/README.md` §3, and they are the
reason this module exists rather than five notebook cells:

**Mice are the replicates, not recdays and not units.** The five recdays of a mouse are the
same probe in the same brain, re-sorted, so a physical neuron plausibly appears in several.
Pooled histograms are for SHAPE; every number that carries a claim is computed per mouse.

**Every region is reported on its own before any contrast**, including the ones with n=1
mouse -- the mouse<->region confound lets a contrast look clean while both arms are a single
animal. ENTl-sup is 90% ah08 and ENTm is 63% ly07; those captions are not optional.

**Coherence is a TWO-comparison criterion, so its chance level is 1/16.** A pair counts as
coherent only if its relative rotation survives X->Y *and* X->Z. One comparison landing
within 45 deg is a one-in-four coin flip given the h=4 quantisation -- far too easy to
satisfy by accident to support a claim about rigid rotation -- so the criterion is squared.
`dual_rate` is the headline; `per_comparison_rate` (chance 1/4) stays for the single-neuron
figures and as a diagnostic.

**Two reference bands on every angle histogram.** The single-comparison chance level is 1/4,
not uniform, because the h=4-dominated spectrum quantises the argmax onto multiples of 90
(see `w3_remapping`). So a real distribution is only interpretable between:
  - the CEILING, X vs X' -- the same physical task run twice, so anything but 0 is noise; and
  - the FLOOR, the cell-identity shuffle -- the same curves with the correspondence broken.
A result that sits on the floor is a null result and gets reported as one.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import anatomy_split as asplit                       # noqa: E402
import w3_remapping as w3                            # noqa: E402

NEUTRAL = '#2C2C2A'
CEILING_COLOR = '#45B5AA'
FLOOR_COLOR = '#B4B2A9'
REAL_COLOR = '#0F4C81'

#: Chance for the DUAL coherence criterion -- coherent in X->Y *and* X->Z. Given the h=4
#: quantisation a single comparison is a one-in-four coin flip, so requiring two squares it.
DUAL_CHANCE = w3.CHANCE ** 2                              # 1/16


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def angle_cols(frame, prefix):
    """Columns `prefix_0..prefix_k` in numeric order (not lexicographic: _10 after _9)."""
    cols = [c for c in frame.columns
            if c.startswith(prefix + '_') and c[len(prefix) + 1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix) + 1:]))


def stack(frame, prefix):
    """The `(n_rows, n_comparisons)` angle matrix for a prefix; empty if absent."""
    cols = angle_cols(frame, prefix)
    if not cols:
        return np.zeros((len(frame), 0))
    return frame[cols].to_numpy(dtype=float)


def per_comparison_rate(frame, prefix, threshold=w3.GEN_THRESHOLD_DEG):
    """Fraction of (row, comparison) angles within threshold of 0. Chance 1/4.

    The SINGLE-comparison rate. Used for the single-neuron figures and as a diagnostic; the
    headline coherence statistic is `dual_rate` below.
    """
    A = stack(frame, prefix)
    v = A[np.isfinite(A)]
    return float(np.mean(np.abs(v) < threshold)) if len(v) else np.nan


def _dual_pairs_of_comparisons(n_comp):
    return [(a, b) for a in range(n_comp) for b in range(a + 1, n_comp)]


def dual_per_row(A, threshold=w3.GEN_THRESHOLD_DEG):
    """Per row: fraction of comparison-PAIRS in which the row is coherent in BOTH.

    One value per neuron/pair, so it can be averaged, matched and subsampled like any other
    per-row quantity while still carrying the two-comparison definition.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[1] < 2:
        return np.full(A.shape[0] if A.ndim == 2 else 0, np.nan)
    ok, valid = np.abs(A) < threshold, np.isfinite(A)
    num = np.zeros(A.shape[0])
    den = np.zeros(A.shape[0])
    for c1, c2 in _dual_pairs_of_comparisons(A.shape[1]):
        v = valid[:, c1] & valid[:, c2]
        num += np.where(v, ok[:, c1] & ok[:, c2], 0)
        den += v
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def dual_rate(frame, prefix, threshold=w3.GEN_THRESHOLD_DEG):
    """THE headline coherence statistic: coherent across TWO task comparisons at once.

    A pair counts only if its relative rotation is preserved in X->Y *and* X->Z, which is
    El-Gaby's dual criterion. Chance is (1/4)**2 = **1/16**, not 1/4: one comparison staying
    within 45 deg is a one-in-four coin flip given the h=4 quantisation, and requiring two
    independent ones squares it. That is the whole point of the dual metric -- a single
    comparison is far too easy to satisfy by accident to support a claim about rigid
    rotation.

    Averaged over every unordered pair of the recday's comparisons rather than just the
    first two. Recdays here have 3-6 comparisons (so 3-15 comparison-pairs), and using all
    of them is a lower-variance estimate of the same quantity -- the chance level is 1/16
    however many there are. `dual_rate_parity` is the strict El-Gaby version.
    """
    v = dual_per_row(stack(frame, prefix), threshold)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else np.nan


def dual_rate_parity(frame, prefix, threshold=w3.GEN_THRESHOLD_DEG):
    """El-Gaby parity: the first two comparisons only (X vs Y and X vs Z)."""
    A = stack(frame, prefix)
    if A.shape[1] < 2:
        return np.nan
    ok = np.abs(A[:, :2]) < threshold
    v = np.all(np.isfinite(A[:, :2]), axis=1)
    return float(np.mean(np.all(ok[v], axis=1))) if v.sum() else np.nan


# ---------------------------------------------------------------------------
# Aggregation: the recday is the atom, the mouse is the mean of its recdays
# ---------------------------------------------------------------------------

def by_recday(frame, prefixes, statfn=None, min_rows=20):
    """One row per recday, one column per angle prefix. THE aggregation atom.

    Pooling a mouse's rows and taking a single rate is pair-weighted, and `n_pairs ~ n**2`,
    so a recday with twice the neurons carries four times the weight -- the mouse's number
    becomes its best-yield day. Computing the statistic per recday first and averaging
    afterwards gives every recday equal weight, which is what "recday" being the unit means.

    Recdays and mice are BOTH reported downstream. A recday is a legitimate replicate for a
    within-recday quantity like pair coherence; mice are the conservative unit because the
    five recdays of a mouse are the same probe in the same brain, re-sorted.
    """
    statfn = statfn or dual_rate
    rows = []
    for rd, sub in frame.groupby('recday', sort=True):
        if len(sub) < min_rows:
            continue
        row = {'recday': rd, 'mouse': rd.split('_')[0], 'n_rows': len(sub)}
        for name, pref in prefixes.items():
            row[name] = statfn(sub, pref)
        rows.append(row)
    return pd.DataFrame(rows)


def by_mouse(per_recday):
    """Collapse recdays within a mouse -- each mouse contributes ONE number."""
    if not len(per_recday):
        return per_recday
    num = per_recday.select_dtypes(include=[np.number])
    out = per_recday.groupby('mouse')[num.columns.tolist()].mean().reset_index()
    out['n_recdays'] = per_recday.groupby('mouse').size().to_numpy()
    return out


def _stars(p):
    if not np.isfinite(p):
        return ''
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


def test_vs_chance(values, chance):
    """One-sample t-test against the analytic chance level, one-sided (greater)."""
    from scipy import stats
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return {'n': len(v), 'mean': float(v.mean()) if len(v) else np.nan, 'p': np.nan}
    p = float(stats.ttest_1samp(v, chance, alternative='greater')[1])
    return {'n': len(v), 'mean': float(v.mean()), 'p': p, 'stars': _stars(p)}


def test_vs_shuffle(real, shuffle):
    """Paired test against each unit's OWN shuffle -- stronger than the analytic chance level.

    Every recday has its own firing rates, trial counts and task count, so its shuffle floor
    sits in a slightly different place. Pairing each recday against its own null removes that
    variation, where a test against a single analytic constant carries it as noise.
    """
    from scipy import stats
    r, s = np.asarray(real, dtype=float), np.asarray(shuffle, dtype=float)
    ok = np.isfinite(r) & np.isfinite(s)
    r, s = r[ok], s[ok]
    if len(r) < 3:
        return {'n': len(r), 'p': np.nan}
    try:
        p = float(stats.wilcoxon(r, s, alternative='greater')[1])
    except ValueError:
        p = float(stats.ttest_rel(r, s, alternative='greater')[1])
    return {'n': int(len(r)), 'mean_gap': float(np.mean(r - s)),
            'n_positive': int(np.sum(r > s)), 'p': p, 'stars': _stars(p)}


def _levels_panel(ax, per_recday, per_mouse, cols, labels, colors, chance, ylabel,
                  chance_label, shuffle_col=None):
    """The standard two-level summary: recdays (open) and mice (filled), stars vs chance."""
    x = np.arange(len(cols))
    for _, r in per_recday.iterrows():
        ax.plot(x - 0.16, [r[c] for c in cols], '-', color='0.85', lw=0.6, zorder=1)
    ax.plot(np.repeat(x - 0.16, len(per_recday)),
            np.concatenate([per_recday[c].to_numpy() for c in cols]), 'o',
            mfc='none', mec='0.6', ms=2.6, lw=0, zorder=2,
            label=f'recday (n={len(per_recday)})')
    for _, r in per_mouse.iterrows():
        ax.plot(x + 0.16, [r[c] for c in cols], '-o', color='0.35', ms=3.4, lw=0.8, zorder=3)
    ax.plot([], [], '-o', color='0.35', ms=3.4, lw=0.8,
            label=f'mouse (n={len(per_mouse)})')
    ax.bar(x, [per_mouse[c].mean() for c in cols], color=colors, alpha=0.4, width=0.75,
           zorder=0)
    ax.axhline(chance, color='k', ls=':', lw=0.8)
    ax.text(x[-1] + 0.45, chance, chance_label, fontsize=6, va='bottom', ha='right')

    top = np.nanmax(per_recday[cols].to_numpy())
    for k, c in enumerate(cols):
        pr = test_vs_chance(per_recday[c], chance)
        pm = test_vs_chance(per_mouse[c], chance)
        txt = f"{pr.get('stars', '')}/{pm.get('stars', '')}"
        if shuffle_col and c != shuffle_col:
            ps = test_vs_shuffle(per_recday[c], per_recday[shuffle_col])
            txt += f"\n{ps.get('n_positive', 0)}/{ps.get('n', 0)}>null"
        ax.text(k, top * 1.04, txt, ha='center', va='bottom', fontsize=5.5)
    ax.set_ylim(0, top * 1.30)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=5.5, frameon=False, loc='upper left', ncol=2)


# ---------------------------------------------------------------------------
# The shared polar histogram
# ---------------------------------------------------------------------------

def polar_hist(ax, angles, overlays=(), n_bins=36, color=REAL_COLOR, label='data',
               title=None, legend=True, show_window=True, fontsize=7, ticklabels=True):
    """Density polar histogram of signed angles in (-180, 180], with reference bands.

    Densities, not counts, so distributions with very different n stay comparable -- the
    X-vs-X' ceiling has far fewer angles than the real data and the shuffle floor has the
    same n as the real data by construction.

    The X-vs-X' ceiling is deliberately NOT overlaid on a panel whose subject is the real
    distribution: it peaks so hard at 0 that it sets the radial limit and flattens everything
    else into the middle of the plot. It gets its own panel instead.
    """
    import matplotlib.pyplot as plt

    a = np.asarray(angles, dtype=float).ravel()
    a = a[np.isfinite(a)]
    edges = np.linspace(-180, 180, n_bins + 1)
    theta = np.deg2rad((edges[:-1] + edges[1:]) / 2)
    width = 2 * np.pi / n_bins

    if show_window:
        ax.bar(0, 1e6, width=np.deg2rad(2 * w3.GEN_THRESHOLD_DEG), bottom=0,
               color='0.92', edgecolor='none', zorder=0)
    if len(a):
        dens = np.histogram(a, edges)[0] / len(a)
        ax.bar(theta, dens, width=width, color=color, alpha=0.85, zorder=2,
               label=f'{label} (n={len(a)})')
        top = dens.max()
    else:
        top = 1.0

    for name, ov, ocolor, ls in overlays:
        o = np.asarray(ov, dtype=float).ravel()
        o = o[np.isfinite(o)]
        if not len(o):
            continue
        d = np.histogram(o, edges)[0] / len(o)
        ax.plot(np.append(theta, theta[0]), np.append(d, d[0]), color=ocolor, lw=1.2,
                ls=ls, zorder=3, label=f'{name} (n={len(o)})')
        top = max(top, d.max())

    ax.set_ylim(0, top * 1.15)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    # In a row of small panels the 90/-90 labels of neighbours collide; 0 and 180 carry the
    # reading anyway (0 = generalising, 180 = opposite state).
    ax.set_xticklabels(['0°', '90°', '±180°', '−90°'] if ticklabels
                       else ['0°', '', '±180°', ''], fontsize=fontsize)
    ax.set_yticklabels([])
    ax.grid(alpha=0.3, lw=0.5)
    if title:
        ax.set_title(title, fontsize=fontsize + 1, pad=12)
    if legend:
        # Below the dial: a polar axes has no empty corner, so any in-axes or top-left
        # placement either sits on the 0-degree tick or runs off the canvas.
        ax.legend(fontsize=fontsize - 1, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  frameon=False, handlelength=1.6, borderaxespad=0)
    return ax


def _stagger(ax, xs, ys, labels, fontsize=5.5):
    """Point labels that do not sit on top of each other when values nearly coincide."""
    ys = np.asarray(ys, dtype=float)
    keep = np.flatnonzero(np.isfinite(ys))
    for rank, i in enumerate(keep[np.argsort(ys[keep])]):
        dy = 5 if rank % 2 == 0 else -9
        ax.annotate(labels[i], (xs[i], ys[i]), fontsize=fontsize, xytext=(0, dy),
                    textcoords='offset points', ha='center')


#: Every figure is written in both, always: PNG to look at, PDF (Type-42 via
#: `apply_gridmaze_style`) to drop into a figure without rasterising.
SAVE_FORMATS = ('png', 'pdf')


def _save(fig, name, save_dir, formats=SAVE_FORMATS):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(save_dir, f'{name}.{ext}'), bbox_inches=None, dpi=200)
    print(f"  wrote {os.path.join(save_dir, name)}.{{{','.join(formats)}}}")


# ---------------------------------------------------------------------------
# Figure 1 — all neurons
# ---------------------------------------------------------------------------

def fig1_all_neurons(neurons, method='xcorr', save_dir=None, name='fig1_all_neurons'):
    """Remapping angle of every gated neuron against the reference task.

    Peak at 0 would mean the population keeps its task-state preference across tasks. The
    two reference bands decide whether any peak is real: X-vs-X' shows what the estimator
    returns when there IS no remapping, the cell-identity shuffle what it returns when there
    is no correspondence at all.
    """
    import matplotlib.pyplot as plt

    inc = neurons[neurons['included']]
    real = stack(inc, f'angles_{method}')
    shuf = stack(inc, f'angles_shuffle_{method}')
    ceil = stack(inc, f'angles_X_{method}')

    fig = plt.figure(figsize=(11.5, 3.9))
    ax = fig.add_subplot(1, 3, 1, projection='polar')
    polar_hist(ax, real,
               overlays=[('cell-identity shuffle (floor)', shuf, FLOOR_COLOR, '--')],
               label='cross-task', title='Cross-task remapping angle\n(all gated neurons)')
    axc = fig.add_subplot(1, 3, 2, projection='polar')
    polar_hist(axc, ceil, color=CEILING_COLOR, label="X vs X′",
               title="Ceiling: X vs X′\n(the same task, run twice)")

    ax2 = fig.add_subplot(1, 3, 3)
    prefixes = {'shuffle': f'angles_shuffle_{method}', 'real': f'angles_{method}',
                'ceiling': f'angles_X_{method}'}
    per_recday = by_recday(inc, prefixes, statfn=per_comparison_rate)
    per_mouse = by_mouse(per_recday)
    _levels_panel(ax2, per_recday, per_mouse, ['shuffle', 'real', 'ceiling'],
                  ['shuffle\n(floor)', 'cross-task\n(real)', "X vs X\u2032\n(ceiling)"],
                  [FLOOR_COLOR, REAL_COLOR, CEILING_COLOR], w3.CHANCE,
                  f'|angle| < {w3.GEN_THRESHOLD_DEG:.0f}\u00b0 (single comparison)',
                  'chance \u00bc', shuffle_col='shuffle')
    ax2.set_title(f'Recdays (n={len(per_recday)}), mice (n={len(per_mouse)})\n'
                  f'stars: t-test vs chance (recday/mouse)\n'
                  f'"k/n>null": recdays above their own shuffle', fontsize=7)
    fig.subplots_adjust(top=0.70, bottom=0.28, wspace=0.42, left=0.07, right=0.97)
    _save(fig, f'{name}_{method}', save_dir)
    return fig, per_recday, per_mouse


# ---------------------------------------------------------------------------
# Figure 2 — per region
# ---------------------------------------------------------------------------

def fig2_per_region(neurons, method='xcorr', groups=None, floors=None, save_dir=None,
                    name='fig2_per_region', min_units_per_mouse=8):
    """Per region: the cross-task dial, its own X-vs-X\u2032 dial, and both aggregation levels.

    X-vs-X\u2032 gets its OWN row of dials rather than an overlay. It peaks so hard at 0 that
    overlaying it sets the radial limit and flattens the real distribution into the middle of
    the plot -- and it is a result in its own right, not an annotation: it is what this
    estimator returns for these neurons when there is no remapping to find.

    Rate matching and each region's split-half floor are the two things that decide whether
    this figure means anything -- a low-rate region gives a noisier argmax, hence a FLATTER
    dial, which reads as "more remapping" with no biology behind it.
    """
    import matplotlib.pyplot as plt

    groups = groups or asplit.ANALYSIS_GROUPS
    inc = neurons[neurons['included']]
    present = [g for g in groups if (inc['group'] == g).sum() >= 10]

    fig = plt.figure(figsize=(2.3 * len(present), 7.4))
    gs = fig.add_gridspec(3, len(present), height_ratios=[1.25, 1.25, 1.15], hspace=0.85,
                          top=0.86, bottom=0.07, left=0.07, right=0.98)
    for k, g in enumerate(present):
        sub = inc[inc['group'] == g]
        col = asplit.REGION_COLORS.get(g, NEUTRAL)

        ax = fig.add_subplot(gs[0, k], projection='polar')
        polar_hist(ax, stack(sub, f'angles_{method}'),
                   overlays=[('cell-identity shuffle', stack(sub, f'angles_shuffle_{method}'),
                              FLOOR_COLOR, '--')],
                   color=col, label='cross-task', legend=False, fontsize=6,
                   ticklabels=False)
        floor_txt = ''
        if floors is not None and len(floors):
            f = floors.loc[floors['group'] == g, 'angle_deg'].abs()
            if len(f) >= 20:
                floor_txt = (f'\nsplit-half floor {f.median():.0f}\u00b0 '
                             f'({(f < w3.GEN_THRESHOLD_DEG).mean():.0%}<45\u00b0)')
        caveat = asplit.SINGLE_MOUSE_GROUPS.get(g, '')
        ax.set_title(f'{g}\n{len(sub)} units, {sub["mouse"].nunique()} mice'
                     + (f'\n{caveat}' if caveat else '') + floor_txt, fontsize=6.5, pad=16)

        axc = fig.add_subplot(gs[1, k], projection='polar')
        polar_hist(axc, stack(sub, f'angles_X_{method}'), color=CEILING_COLOR,
                   label="X vs X\u2032", legend=False, fontsize=6, ticklabels=False)
        axc.set_title(f"X vs X\u2032 ceiling\n{g}", fontsize=6.5, pad=14)

    ax2 = fig.add_subplot(gs[2, :])
    prefixes = {'shuffle': f'angles_shuffle_{method}', 'real': f'angles_{method}'}
    xs, labels, colors = [], [], []
    for k, g in enumerate(present):
        sub = inc[inc['group'] == g]
        pr = by_recday(sub, prefixes, statfn=per_comparison_rate, min_rows=min_units_per_mouse)
        if not len(pr):
            continue
        pm = by_mouse(pr)
        xs.append((k, pr, pm))
        labels.append(g)
        colors.append(asplit.REGION_COLORS.get(g, NEUTRAL))
        ax2.bar(k, pm['real'].mean(), color=colors[-1], alpha=0.4, width=0.75, zorder=0)
        ax2.plot(np.full(len(pr), k) - 0.17, pr['real'], 'o', mfc='none', mec='0.6', ms=2.6,
                 lw=0, zorder=2, label='recday' if k == 0 else None)
        mx = k + np.linspace(0.08, 0.30, len(pm))
        ax2.plot(mx, pm['real'], 'o', color='0.3', ms=3.6, lw=0, zorder=3,
                 label='mouse' if k == 0 else None)
        _stagger(ax2, mx, pm['real'].to_numpy(), list(pm['mouse']))
        p_rd = test_vs_chance(pr['real'], w3.CHANCE)
        p_ms = test_vs_chance(pm['real'], w3.CHANCE)
        ax2.text(k, 0.44, f"{p_rd.get('stars', '')}/{p_ms.get('stars', '')}\n"
                          f"{len(pr)} rd, {len(pm)} mice", ha='center', va='bottom',
                 fontsize=5.5)
    ax2.axhline(w3.CHANCE, color='k', ls=':', lw=0.8)
    ax2.text(len(present) - 0.55, w3.CHANCE, 'chance \u00bc', fontsize=6, va='bottom',
             ha='right')
    ax2.set_xticks(range(len(present)))
    ax2.set_xticklabels(present, fontsize=6.5)
    ax2.set_ylim(0, 0.56)
    ax2.set_ylabel(f'|angle| < {w3.GEN_THRESHOLD_DEG:.0f}\u00b0')
    ax2.set_title('Top dials: bars = cross-task angles, dashed = cell-identity shuffle. '
                  'Middle dials: X-vs-X\u2032 ceiling.\nBelow: open = recday, filled = '
                  'mouse; stars are t-tests vs chance (recday/mouse).', fontsize=7)
    ax2.legend(fontsize=5.5, frameon=False, loc='upper left', ncol=2)
    _save(fig, f'{name}_{method}', save_dir)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — coherence, all pairs
# ---------------------------------------------------------------------------

def fig3_all_pairs(pairs, save_dir=None, name='fig3_all_pairs'):
    """Relative rotation of every simultaneously-recorded pair of gated neurons.

    Peak at 0 means the two cells rotated by the same amount -- the population moving as a
    rigid body. This is the figure that can be positive while figure 1 is null: cells can
    remap completely and still remap TOGETHER.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.5, 3.9))
    ax = fig.add_subplot(1, 3, 1, projection='polar')
    polar_hist(ax, stack(pairs, 'rel'),
               overlays=[('cell-identity shuffle (floor)', stack(pairs, 'relS'),
                          FLOOR_COLOR, '--'),
                         ('direct (parity)', stack(pairs, 'relD'), '#6B3FA0', ':')],
               label='relative rotation',
               title='Pairwise relative rotation\n(all simultaneous pairs)')
    axc = fig.add_subplot(1, 3, 2, projection='polar')
    polar_hist(axc, stack(pairs, 'relX'), color=CEILING_COLOR, label="X vs X′",
               title="Ceiling: X vs X′\n(the same task, run twice)")

    ax2 = fig.add_subplot(1, 3, 3)
    prefixes = {'shuffle': 'relS', 'real': 'rel', 'direct': 'relD', 'ceiling': 'relX'}
    per_recday = by_recday(pairs, prefixes, statfn=dual_rate)
    per_recday['real_parity'] = [dual_rate_parity(pairs[pairs.recday == rd], 'rel')
                                 for rd in per_recday['recday']]
    per_mouse = by_mouse(per_recday)
    # The ceiling is NOT a bar here: at ~0.50 it sets the y-limit and squashes the
    # shuffle/real/direct comparison that the panel exists to show. It has its own dial.
    _levels_panel(ax2, per_recday, per_mouse, ['shuffle', 'real', 'direct'],
                  ['shuffle', 'real\n(ref-anch.)', 'real\n(direct)'],
                  [FLOOR_COLOR, REAL_COLOR, '#6B3FA0'], DUAL_CHANCE,
                  'DUAL coherence (two comparisons)', 'chance 1/16', shuffle_col='shuffle')
    # The dual ceiling needs TWO repeat comparisons in a recday. Only 9 of 25 recdays have
    # two exact task repeats (8 have none) -- fewer than the nominal "6 unique tasks + 2
    # repeats" design implies. Stated, not quietly averaged over the recdays that lack it.
    n_ceil = int(np.isfinite(per_recday['ceiling']).sum())
    ax2.set_title(f'Recdays (n={len(per_recday)}), mice (n={len(per_mouse)})\n'
                  f'stars: t-test vs chance (recday/mouse)\n'
                  f'X-vs-X\u2032 ceiling {per_recday["ceiling"].mean():.2f} '
                  f'({n_ceil} recdays have 2 repeats)', fontsize=7)
    fig.subplots_adjust(top=0.70, bottom=0.28, wspace=0.42, left=0.07, right=0.97)
    _save(fig, name, save_dir)
    return fig, per_recday, per_mouse


# ---------------------------------------------------------------------------
# Figure 4 — coherence within each region
# ---------------------------------------------------------------------------

def fig4_within_region(pairs, groups=None, save_dir=None, name='fig4_within_region',
                       min_pairs=200, min_pairs_per_recday=30):
    """Within-region pair coherence: the dial, its own X-vs-X\u2032 dial, and both levels.

    Densities use every pair -- a density estimate is unbiased whatever n is, so nothing is
    gained by discarding pairs for the SHAPE. The equal-count matching W3 requires belongs to
    the scalar comparison in figure 5, where unequal n really would change the answer.
    """
    import matplotlib.pyplot as plt

    groups = groups or asplit.ANALYSIS_GROUPS
    within = {g: pairs[(pairs['group_i'] == g) & (pairs['group_j'] == g)] for g in groups}
    present = [g for g in groups if len(within[g]) >= min_pairs]

    fig = plt.figure(figsize=(2.3 * len(present), 7.4))
    gs = fig.add_gridspec(3, len(present), height_ratios=[1.25, 1.25, 1.15], hspace=0.85,
                          top=0.86, bottom=0.07, left=0.07, right=0.98)
    for k, g in enumerate(present):
        sub = within[g]
        col = asplit.REGION_COLORS.get(g, NEUTRAL)
        ax = fig.add_subplot(gs[0, k], projection='polar')
        polar_hist(ax, stack(sub, 'rel'),
                   overlays=[('cell-identity shuffle', stack(sub, 'relS'), FLOOR_COLOR, '--')],
                   color=col, label='within-region', legend=False, fontsize=6,
                   ticklabels=False)
        caveat = asplit.SINGLE_MOUSE_GROUPS.get(g, '')
        ax.set_title(f'{g}\n{len(sub)} pairs, {sub["mouse"].nunique()} mice'
                     + (f'\n{caveat}' if caveat else ''), fontsize=6.5, pad=16)
        axc = fig.add_subplot(gs[1, k], projection='polar')
        polar_hist(axc, stack(sub, 'relX'), color=CEILING_COLOR, label="X vs X\u2032",
                   legend=False, fontsize=6, ticklabels=False)
        axc.set_title(f"X vs X\u2032 ceiling\n{g}", fontsize=6.5, pad=14)

    ax2 = fig.add_subplot(gs[2, :])
    top = 0.0
    for k, g in enumerate(present):
        pr = by_recday(within[g], {'shuffle': 'relS', 'real': 'rel'}, statfn=dual_rate,
                       min_rows=min_pairs_per_recday)
        if not len(pr):
            continue
        pm = by_mouse(pr)
        col = asplit.REGION_COLORS.get(g, NEUTRAL)
        ax2.bar(k, pm['real'].mean(), color=col, alpha=0.4, width=0.75, zorder=0)
        ax2.plot(np.full(len(pr), k) - 0.17, pr['real'], 'o', mfc='none', mec='0.6', ms=2.6,
                 lw=0, zorder=2, label='recday' if k == 0 else None)
        ax2.plot(np.full(len(pr), k) - 0.17, pr['shuffle'], '_', color=FLOOR_COLOR, ms=5,
                 zorder=2, label='recday shuffle' if k == 0 else None)
        mx = k + np.linspace(0.08, 0.30, len(pm))
        ax2.plot(mx, pm['real'], 'o', color='0.3', ms=3.6, lw=0, zorder=3,
                 label='mouse' if k == 0 else None)
        _stagger(ax2, mx, pm['real'].to_numpy(), list(pm['mouse']))
        p_rd = test_vs_chance(pr['real'], DUAL_CHANCE)
        p_ms = test_vs_chance(pm['real'], DUAL_CHANCE)
        p_sh = test_vs_shuffle(pr['real'], pr['shuffle'])
        top = max(top, np.nanmax(pr['real']))
        ax2.text(k, np.nanmax(pr['real']) * 1.04,
                 f"{p_rd.get('stars', '')}/{p_ms.get('stars', '')}\n"
                 f"{p_sh.get('n_positive', 0)}/{p_sh.get('n', 0)}>null", ha='center',
                 va='bottom', fontsize=5.5)
    ax2.axhline(DUAL_CHANCE, color='k', ls=':', lw=0.8)
    ax2.text(len(present) - 0.55, DUAL_CHANCE, 'chance 1/16', fontsize=6, va='bottom',
             ha='right')
    ax2.set_xticks(range(len(present)))
    ax2.set_xticklabels(present, fontsize=6.5)
    ax2.set_ylim(0, top * 1.35)
    ax2.set_ylabel('DUAL coherence (two comparisons)')
    ax2.set_title('Top dials: bars = within-region pair rotations, dashed = cell-identity '
                  'shuffle. Middle dials: X-vs-X\u2032 ceiling.\nBelow: open = recday, '
                  'filled = mouse; stars are t-tests vs chance (recday/mouse).', fontsize=7)
    ax2.legend(fontsize=5.5, frameon=False, loc='upper left', ncol=2)
    _save(fig, name, save_dir)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — same-region or cross-region?
# ---------------------------------------------------------------------------

def _categorise(gi, gj, group_a, group_b):
    """0 = within A, 1 = within B, 2 = cross A-B, -1 = irrelevant."""
    cat = np.full(len(gi), -1)
    cat[(gi == group_a) & (gj == group_a)] = 0
    cat[(gi == group_b) & (gj == group_b)] = 1
    cat[((gi == group_a) & (gj == group_b)) | ((gi == group_b) & (gj == group_a))] = 2
    return cat


def _matched(cat, dist, value, rng, n_dist_bins=4):
    """Coherence per category, matched on pair COUNT and on initial tuning distance.

    Both matchings are load-bearing:

    *Count* -- `n_pairs ~ n**2`, so a region with twice the neurons brings four times the
    pairs and would dominate any pooled number.

    *Initial tuning distance* -- coherence depends on how similar the two cells' task-space
    tuning was to begin with (this is why `coherence_by_tuning_dist` exists in the El-Gaby
    port). Same-region neurons plausibly have more similar tuning than cross-region ones, so
    without this matching the figure returns "same-region pairs are more coherent" for free
    and is measuring tuning similarity rather than anatomy.
    """
    edges = np.linspace(0, 180, n_dist_bins + 1)[1:-1]
    dbin = np.digitize(dist, edges)
    sums = np.zeros(3)
    counts = np.zeros(3)
    for b in range(n_dist_bins):
        idx = [np.flatnonzero((cat == c) & (dbin == b) & np.isfinite(value)) for c in range(3)]
        m = min(len(i) for i in idx)
        if m == 0:
            continue                       # cannot match this distance bin; drop it entirely
        for c in range(3):
            pick = rng.choice(idx[c], m, replace=False)
            sums[c] += np.nansum(value[pick])
            counts[c] += m
    with np.errstate(invalid='ignore'):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan), counts


def _prep_recdays(pairs, value_col=None):
    """Per-recday numpy views: neuron labels, pair endpoints, distances, values.

    The permutation null relabels neurons thousands of times, so it must not go back through
    pandas each draw. Everything it needs is extracted once here.
    """
    prepped = []
    for rd, sub in pairs.groupby('recday', sort=True):
        ids = np.unique(np.concatenate([sub['neuron_i'].to_numpy(),
                                        sub['neuron_j'].to_numpy()]))
        pos = {n: k for k, n in enumerate(ids)}
        lab = np.empty(len(ids), dtype=object)
        for cn, cg in (('neuron_i', 'group_i'), ('neuron_j', 'group_j')):
            for n, g in zip(sub[cn].to_numpy(), sub[cg].to_numpy()):
                lab[pos[n]] = g
        if value_col is None:
            value = dual_per_row(stack(sub, 'rel'))
            null = dual_per_row(stack(sub, 'relS'))
        else:
            value = sub[value_col].to_numpy(dtype=float)
            null = sub['coherent_relS'].to_numpy(dtype=float)
        prepped.append({
            'recday': rd, 'mouse': rd.split('_')[0], 'labels': lab,
            'i': np.array([pos[n] for n in sub['neuron_i'].to_numpy()]),
            'j': np.array([pos[n] for n in sub['neuron_j'].to_numpy()]),
            'dist': sub['tuning_dist'].to_numpy(dtype=float),
            'value': value, 'null': null,
        })
    return prepped


def _gap_from_labels(prepped, labels_by_rd, group_a, group_b, rng, n_dist_bins, n_draws):
    """Mouse-level mean of (within − cross) under an arbitrary neuron labelling."""
    rows = []
    for pr, lab in zip(prepped, labels_by_rd):
        cat = _categorise(lab[pr['i']], lab[pr['j']], group_a, group_b)
        if not all((cat == c).sum() for c in range(3)):
            continue
        draws = np.array([_matched(cat, pr['dist'], pr['value'], rng, n_dist_bins)[0]
                          for _ in range(n_draws)])
        m = draws.mean(axis=0)
        if not np.all(np.isfinite(m)):
            continue
        rows.append((pr['mouse'], (m[0] + m[1]) / 2 - m[2]))
    if not rows:
        return np.nan
    df = pd.DataFrame(rows, columns=['mouse', 'gap'])
    return float(df.groupby('mouse')['gap'].mean().mean())


def matched_coherence(pairs, group_a='ENTl-deep', group_b='SUB/ProS', value_col=None,
                      n_dist_bins=4, seed=0, n_draws=25, prepped=None):
    """Per-recday matched coherence for within-A, within-B and cross pairs.

    `value_col` defaults to the per-comparison indicator (comparable across recdays with
    different task counts) rather than the strict all-comparisons flag, whose chance level is
    `0.25**n_comparisons` and so differs 60-fold between a 3-comparison and a 6-comparison
    recday.
    """
    rng = np.random.default_rng(seed)
    prepped = prepped if prepped is not None else _prep_recdays(pairs, value_col)
    rows = []
    for pr in prepped:
        lab = pr['labels']
        cat = _categorise(lab[pr['i']], lab[pr['j']], group_a, group_b)
        if not all((cat == c).sum() for c in range(3)):
            continue
        draws = np.array([_matched(cat, pr['dist'], pr['value'], rng, n_dist_bins)[0]
                          for _ in range(n_draws)])
        ndraws = np.array([_matched(cat, pr['dist'], pr['null'], rng, n_dist_bins)[0]
                           for _ in range(n_draws)])
        _, counts = _matched(cat, pr['dist'], pr['value'], rng, n_dist_bins)
        m = draws.mean(axis=0)
        if not np.all(np.isfinite(m)):
            continue
        nm = ndraws.mean(axis=0)
        rows.append({'recday': pr['recday'], 'mouse': pr['mouse'],
                     f'within_{group_a}': m[0], f'within_{group_b}': m[1], 'cross': m[2],
                     'within_pooled': (m[0] + m[1]) / 2,
                     'null_within': (nm[0] + nm[1]) / 2, 'null_cross': nm[2],
                     'n_matched_per_cat': int(counts[0])})
    return pd.DataFrame(rows)


def matched_coherence_null(pairs, group_a='ENTl-deep', group_b='SUB/ProS', n_perm=500,
                           seed=0, n_dist_bins=4, n_draws=4, value_col=None):
    """Within-minus-cross gap under a WITHIN-RECDAY shuffle of the region labels.

    Shuffling within recday preserves each recday's region composition, its n, and the
    mouse<->region association -- which is the confound, so it has to survive into the null.
    Shuffling across recdays would let a null sample put SUB/ProS units in ah08, which never
    happens in the data, and would make the null far too easy to beat.

    The labels are permuted over NEURONS, not over pairs. A pair's two labels are not
    independent draws -- each neuron appears in many pairs -- so permuting pair labels
    directly would break the `n_pairs ~ n**2` dependence structure and give a null that is
    far too tight.
    """
    rng = np.random.default_rng(seed)
    prepped = _prep_recdays(pairs, value_col)
    obs = matched_coherence(pairs, group_a, group_b, value_col=value_col,
                            n_dist_bins=n_dist_bins, seed=seed, prepped=prepped)
    if not len(obs):
        return {'n_recdays': 0, 'note': 'no recday has all three categories'}
    observed = float((obs['within_pooled'] - obs['cross'])
                     .groupby(obs['mouse']).mean().mean())

    null = np.full(n_perm, np.nan)
    for p in range(n_perm):
        labels = [rng.permutation(pr['labels']) for pr in prepped]
        null[p] = _gap_from_labels(prepped, labels, group_a, group_b, rng, n_dist_bins,
                                   n_draws)
    finite = null[np.isfinite(null)]
    p_val = (1 + np.sum(np.abs(finite) >= abs(observed))) / (1 + len(finite))
    return {'observed_gap': observed, 'null_mean': float(np.mean(finite)),
            'null_sd': float(np.std(finite)), 'p_perm': float(p_val),
            'n_perm': int(len(finite)), 'n_mice': int(obs['mouse'].nunique()),
            'n_recdays': int(len(obs)), 'per_mouse': obs, 'null_draws': finite}


def fig5_same_vs_cross(pairs, group_a='ENTl-deep', group_b='SUB/ProS', null=None,
                       save_dir=None, name='fig5_same_vs_cross'):
    """Are coherent pairs same-region or cross-region?

    The question only has content in the partial-coherence regime. If figure 3 shows the
    whole population rotating rigidly, everything is coherent with everything and there is
    no anatomy to find -- that is a property of the result, not a failure of the analysis,
    and the caption should say so rather than reporting a null.
    """
    import matplotlib.pyplot as plt

    per_rd = matched_coherence(pairs, group_a, group_b)
    if not len(per_rd):
        print('fig5: no recday has all three pair categories — cannot draw')
        return None, per_rd
    per_mouse = per_rd.groupby('mouse').mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))

    cats = [f'within_{group_a}', f'within_{group_b}', 'cross']
    labels = [f'within\n{group_a}', f'within\n{group_b}', f'cross\n{group_a}×{group_b}']
    colors = [asplit.REGION_COLORS.get(group_a, NEUTRAL),
              asplit.REGION_COLORS.get(group_b, NEUTRAL), '#6B3FA0']
    ax = axes[0]
    # BOTH levels: open circles are recdays (n=19), filled+lines are mice (n=4).
    for _, r in per_rd.iterrows():
        ax.plot(np.arange(3) - 0.13, [r[c] for c in cats], 'o', mfc='none', mec='0.65',
                ms=2.6, lw=0, zorder=2)
    ax.plot([], [], 'o', mfc='none', mec='0.65', ms=2.6, lw=0,
            label=f'recday (n={len(per_rd)})')
    for _, r in per_mouse.iterrows():
        ax.plot(np.arange(3) + 0.13, [r[c] for c in cats], '-o', color='0.35', ms=3.6,
                lw=0.8, zorder=3)
        ax.annotate(r['mouse'], (2 + 0.13, r['cross']), fontsize=6, xytext=(4, 0),
                    textcoords='offset points', va='center')
    ax.plot([], [], '-o', color='0.35', ms=3.6, lw=0.8, label=f'mouse (n={len(per_mouse)})')
    ax.bar(range(3), [per_mouse[c].mean() for c in cats], color=colors, alpha=0.45,
           width=0.6, zorder=1)
    ax.plot(range(3), [per_mouse['null_within'].mean(), per_mouse['null_within'].mean(),
                       per_mouse['null_cross'].mean()], '_', color=FLOOR_COLOR, ms=16,
            zorder=4, label='shuffle floor')
    ax.axhline(DUAL_CHANCE, color='k', ls=':', lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel('matched DUAL coherence (two comparisons)')
    ax.set_title(f'Matched on pair count and initial tuning distance\n'
                 f'{len(per_rd)} recdays, {per_mouse["mouse"].nunique()} mice, '
                 f'{per_rd["n_matched_per_cat"].min()}–{per_rd["n_matched_per_cat"].max()} '
                 f'pairs/category', fontsize=7)
    ax.set_ylim(0, max(0.12, per_mouse[cats].to_numpy().max() * 1.15))
    ax.legend(fontsize=6, frameon=False, loc='lower center')

    ax = axes[1]
    gaps = (per_mouse['within_pooled'] - per_mouse['cross']).to_numpy()
    rd_gaps = (per_rd['within_pooled'] - per_rd['cross']).to_numpy()
    ax.axhline(0, color='k', lw=0.8)
    ax.plot(np.full(len(rd_gaps), -0.18) + np.linspace(-0.05, 0.05, len(rd_gaps)), rd_gaps,
            'o', mfc='none', mec='0.65', ms=2.6, lw=0, zorder=2,
            label=f'recday (n={len(rd_gaps)})')
    if null is not None and 'null_sd' in null:
        ax.axhspan(-1.96 * null['null_sd'], 1.96 * null['null_sd'], color=FLOOR_COLOR,
                   alpha=0.35, label='95% of within-recday\nlabel shuffles')
    ax.plot(np.full(len(gaps), 0.18) + np.linspace(-0.05, 0.05, len(gaps)), gaps, 'o',
            color=REAL_COLOR, ms=5, lw=0, zorder=3, label=f'mouse (n={len(gaps)})')
    for xo, g, m in zip(0.18 + np.linspace(-0.05, 0.05, len(gaps)), gaps, per_mouse['mouse']):
        ax.annotate(m, (xo, g), fontsize=6, xytext=(0, 5), textcoords='offset points',
                    ha='center')
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel('within − cross')
    ttl = 'Same-region advantage, per mouse'
    if null is not None and 'p_perm' in null:
        ttl += f"\ngap {null['observed_gap']:+.4f}, p_perm = {null['p_perm']:.3f}"
    ax.set_title(ttl, fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc='lower right', ncol=2)

    ax = axes[2]
    edges = np.linspace(0, 180, 5)
    centres = (edges[:-1] + edges[1:]) / 2
    sub = pairs[_categorise(pairs['group_i'].to_numpy(), pairs['group_j'].to_numpy(),
                            group_a, group_b) >= 0].copy()
    sub['cat'] = _categorise(sub['group_i'].to_numpy(), sub['group_j'].to_numpy(),
                             group_a, group_b)
    sub['coh'] = dual_per_row(stack(sub, 'rel'))
    sub['dbin'] = np.digitize(sub['tuning_dist'], edges[1:-1])
    for c, lab, col in zip(range(3), labels, colors):
        m = [sub.loc[(sub['cat'] == c) & (sub['dbin'] == b), 'coh'].mean() for b in range(4)]
        ax.plot(centres, m, '-o', color=col, ms=3.5, lw=1.2,
                label=lab.replace('\n', ' '))
    ax.set_xlabel('initial tuning distance in the reference task (°)')
    ax.set_ylabel('DUAL coherence')
    ax.set_title('Why the matching is needed:\ncoherence depends on initial tuning distance',
                 fontsize=7)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    _save(fig, name, save_dir)
    return fig, per_rd


# ---------------------------------------------------------------------------
# Figure 6 — coherence vs initial pairwise tuning distance
# ---------------------------------------------------------------------------

def coherence_by_initial_angle(pairs, n_bins=4):
    """Coherence as a function of how far apart the two cells were to begin with."""
    edges = np.linspace(0, 180, n_bins + 1)
    dbin = np.clip(np.digitize(pairs['tuning_dist'].to_numpy(dtype=float), edges[1:-1]),
                   0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sub = pairs[dbin == b]
        for mouse, s in list(sub.groupby('mouse')) + [('ALL', sub)]:
            rows.append({'bin': b, 'label': f'{edges[b]:.0f}–{edges[b + 1]:.0f}°',
                         'mouse': mouse, 'n_pairs': len(s),
                         'real': dual_rate(s, 'rel'), 'shuffle': dual_rate(s, 'relS'),
                         'ceiling': dual_rate(s, 'relX'),
                         'real_single': per_comparison_rate(s, 'rel'),
                         'shuffle_single': per_comparison_rate(s, 'relS')})
    df = pd.DataFrame(rows)
    df['gap'] = df['real'] - df['shuffle']
    return df


def coherence_by_initial_angle_levels(pairs, n_bins=4, min_pairs_per_recday=30):
    """Per distance bin: one value per RECDAY, and the mouse means of those.

    Both levels are tested against the 1/16 chance line separately, per bin. Pooling pairs
    within a mouse would weight each recday by n_pairs ~ n**2.
    """
    edges = np.linspace(0, 180, n_bins + 1)
    dbin = np.clip(np.digitize(pairs['tuning_dist'].to_numpy(dtype=float), edges[1:-1]),
                   0, n_bins - 1)
    out = []
    for b in range(n_bins):
        sub = pairs[dbin == b]
        pr = by_recday(sub, {'real': 'rel', 'shuffle': 'relS', 'ceiling': 'relX'},
                       statfn=dual_rate, min_rows=min_pairs_per_recday)
        if not len(pr):
            continue
        pr['bin'] = b
        pr['label'] = f'{edges[b]:.0f}-{edges[b + 1]:.0f}deg'
        out.append(pr)
    per_recday = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not len(per_recday):
        return per_recday, pd.DataFrame()
    per_mouse = (per_recday.groupby(['bin', 'label', 'mouse'])[['real', 'shuffle', 'ceiling']]
                 .mean().reset_index())
    return per_recday, per_mouse


def fig6_coherence_by_initial_angle(pairs, save_dir=None,
                                    name='fig6_coherence_by_initial_angle', n_bins=4):
    """Does the coherence survive once similarly-tuned pairs are excluded?

    THE control on figure 3. A pair whose two cells already peak at the same task-space phase
    is close to trivially coherent -- and two cells with the same PLACE field sit exactly
    there, then move together across tasks for a reason that has nothing to do with the
    population rotating in task space. So the coherence claim has to hold for pairs that
    started far apart on the ring, where no shared place field can produce it.

    Every bin is tested against the 1/16 chance line at BOTH levels: recdays and mice.
    """
    import matplotlib.pyplot as plt

    per_recday, per_mouse = coherence_by_initial_angle_levels(pairs, n_bins=n_bins)
    if not len(per_recday):
        print('fig6: no distance bin has enough pairs per recday')
        return None, per_recday
    bins = sorted(per_recday['bin'].unique())
    labels = [per_recday.loc[per_recday['bin'] == b, 'label'].iloc[0].replace('deg', '\u00b0')
              for b in bins]
    x = np.arange(len(bins))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    for lvl, frame, marker, colr, ms in (('recday', per_recday, 'o', '0.62', 2.6),
                                         ('mouse', per_mouse, 'o', REAL_COLOR, 4.2)):
        off = -0.12 if lvl == 'recday' else 0.12
        for i, b in enumerate(bins):
            v = frame.loc[frame['bin'] == b, 'real'].to_numpy()
            ax.plot(np.full(len(v), x[i]) + off, v, marker,
                    mfc='none' if lvl == 'recday' else colr, mec=colr, ms=ms, lw=0,
                    zorder=2, label=f'{lvl} (n={len(v)})' if i == 0 else None)
        m = [frame.loc[frame['bin'] == b, 'real'].mean() for b in bins]
        ax.plot(x + off, m, '-', color=colr, lw=1.4 if lvl == 'mouse' else 0.8, zorder=3)
    sh = [per_recday.loc[per_recday['bin'] == b, 'shuffle'].mean() for b in bins]
    ax.plot(x, sh, '--s', color=FLOOR_COLOR, ms=3.5, label='shuffle floor')
    ax.axhline(DUAL_CHANCE, color='k', ls=':', lw=0.8)
    ax.text(x[-1] + 0.4, DUAL_CHANCE, 'chance 1/16', fontsize=6, va='bottom', ha='right')
    top = per_recday['real'].max()
    for i, b in enumerate(bins):
        pr = test_vs_chance(per_recday.loc[per_recday['bin'] == b, 'real'], DUAL_CHANCE)
        pm = test_vs_chance(per_mouse.loc[per_mouse['bin'] == b, 'real'], DUAL_CHANCE)
        ax.text(x[i], top * 1.04, f"{pr.get('stars', '')}/{pm.get('stars', '')}",
                ha='center', va='bottom', fontsize=6)
    ax.set_ylim(0, top * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_xlabel('initial pairwise tuning distance in the reference task')
    ax.set_ylabel('DUAL coherence (two comparisons)')
    ax.set_title('Coherence declines with initial separation but does not vanish\n'
                 'stars: t-test vs 1/16 (recday/mouse)', fontsize=7.5)
    ax.legend(fontsize=6, frameon=False)

    ax = axes[1]
    per_recday = per_recday.assign(gap=per_recday['real'] - per_recday['shuffle'])
    pm_gap = (per_recday.groupby(['bin', 'mouse'])['gap'].mean().reset_index())
    for mouse, srt in pm_gap.groupby('mouse'):
        srt = srt.sort_values('bin')
        ax.plot(x[:len(srt)], srt['gap'], '-o', color='0.6', ms=3, lw=0.8, zorder=2)
    ends = [(m, g.sort_values('bin')['gap'].iloc[-1]) for m, g in pm_gap.groupby('mouse')]
    _stagger(ax, np.full(len(ends), float(x[-1])) + 0.13,
             np.array([v for _, v in ends]), [m for m, _ in ends], fontsize=6)
    pooled = [per_recday.loc[per_recday['bin'] == b, 'gap'].mean() for b in bins]
    ax.plot(x, pooled, '-o', color=REAL_COLOR, ms=5, lw=1.6, zorder=3,
            label='mean over recdays')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_xlim(-0.3, len(bins) - 0.35)
    ax.set_xlabel('initial pairwise tuning distance')
    ax.set_ylabel('real \u2212 shuffle')
    ax.set_title('Coherence above each recday\u2019s own shuffle floor.\nThe far bins are '
                 'the ones a shared place field cannot explain.', fontsize=7.5)
    ax.legend(fontsize=6, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.32, left=0.08, right=0.97)
    _save(fig, name, save_dir)
    return fig, per_recday


def fig6_per_region(pairs, groups=None, save_dir=None, name='fig6_by_initial_angle_per_region',
                    n_bins=4, min_pairs=200, min_pairs_per_recday=25):
    """Figure 6, split by region: does each region's coherence survive distal pairs?

    Within-region pairs only (both neurons in the same region) -- a cross-region pair belongs
    to no single region, and figure 5 is where those live.

    Read the far bins. A pair whose cells already peak at the same task-space phase is close
    to trivially coherent, and two cells sharing a place field sit exactly there; only the
    distal pairs test rigid rotation in a way a shared place field cannot mimic.
    """
    import matplotlib.pyplot as plt

    groups = groups or asplit.ANALYSIS_GROUPS
    within = {g: pairs[(pairs['group_i'] == g) & (pairs['group_j'] == g)] for g in groups}
    present = [g for g in groups if len(within[g]) >= min_pairs]

    fig = plt.figure(figsize=(2.5 * len(present), 5.6))
    gs = fig.add_gridspec(2, len(present), height_ratios=[1.0, 1.0], hspace=0.62,
                          top=0.83, bottom=0.10, left=0.07, right=0.98)
    x = np.arange(n_bins)
    labels = None
    for k, g in enumerate(present):
        pr, pm = coherence_by_initial_angle_levels(
            within[g], n_bins=n_bins, min_pairs_per_recday=min_pairs_per_recday)
        col = asplit.REGION_COLORS.get(g, NEUTRAL)
        ax = fig.add_subplot(gs[0, k])
        if not len(pr):
            ax.text(0.5, 0.5, f'{g}\ntoo few pairs\nper recday', ha='center', va='center',
                    fontsize=7, transform=ax.transAxes)
            ax.set_xticks([])
            continue
        bins = sorted(pr['bin'].unique())
        labels = [pr.loc[pr['bin'] == b, 'label'].iloc[0].replace('deg', '\u00b0')
                  for b in bins]
        for i, b in enumerate(bins):
            v = pr.loc[pr['bin'] == b, 'real'].to_numpy()
            ax.plot(np.full(len(v), i) - 0.12, v, 'o', mfc='none', mec='0.65', ms=2.4, lw=0,
                    zorder=2)
            mv = pm.loc[pm['bin'] == b, 'real'].to_numpy()
            ax.plot(np.full(len(mv), i) + 0.12, mv, 'o', color=col, ms=3.6, lw=0, zorder=3)
        ax.plot([bins.index(b) for b in bins],
                [pr.loc[pr['bin'] == b, 'real'].mean() for b in bins], '-', color=col, lw=1.5,
                zorder=3)
        ax.plot([bins.index(b) for b in bins],
                [pr.loc[pr['bin'] == b, 'shuffle'].mean() for b in bins], '--s',
                color=FLOOR_COLOR, ms=3, lw=1, zorder=3)
        ax.axhline(DUAL_CHANCE, color='k', ls=':', lw=0.8)
        top = pr['real'].max()
        for i, b in enumerate(bins):
            t_rd = test_vs_chance(pr.loc[pr['bin'] == b, 'real'], DUAL_CHANCE)
            t_ms = test_vs_chance(pm.loc[pm['bin'] == b, 'real'], DUAL_CHANCE)
            ax.text(i, top * 1.03, f"{t_rd.get('stars', '')}\n{t_ms.get('stars', '')}",
                    ha='center', va='bottom', fontsize=5)
        ax.set_ylim(0, top * 1.38)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(labels, fontsize=5.5, rotation=45, ha='right')
        caveat = asplit.SINGLE_MOUSE_GROUPS.get(g, '')
        nm, nr = pm['mouse'].nunique(), pr['recday'].nunique()
        ax.set_title(f'{g}\n{len(within[g])} pairs, {nm} '
                     f'{"mouse" if nm == 1 else "mice"}, {nr} '
                     f'{"recday" if nr == 1 else "recdays"}'
                     + (f'\n{caveat}' if caveat else ''), fontsize=6.5)
        if k == 0:
            ax.set_ylabel('DUAL coherence')

        ax2 = fig.add_subplot(gs[1, k])
        pr = pr.assign(gap=pr['real'] - pr['shuffle'])
        pmg = pr.groupby(['bin', 'mouse'])['gap'].mean().reset_index()
        for mouse, srt in pmg.groupby('mouse'):
            srt = srt.sort_values('bin')
            ax2.plot(range(len(srt)), srt['gap'], '-o', color='0.65', ms=2.6, lw=0.7, zorder=2)
        ax2.plot(range(len(bins)), [pr.loc[pr['bin'] == b, 'gap'].mean() for b in bins],
                 '-o', color=col, ms=4, lw=1.6, zorder=3)
        ax2.axhline(0, color='k', lw=0.8)
        ax2.set_xticks(range(len(bins)))
        ax2.set_xticklabels(labels, fontsize=5.5, rotation=45, ha='right')
        ax2.set_title('real \u2212 own shuffle', fontsize=6.5)
        if k == 0:
            ax2.set_ylabel('real \u2212 shuffle')
    fig.suptitle('Within-region DUAL coherence vs initial pairwise tuning distance. '
                 'Open = recday, filled = mouse;\nstars = t-test vs 1/16 (top: recday, '
                 'bottom: mouse). The FAR bins are the ones a shared place field cannot '
                 'explain.', fontsize=7.5, y=0.97)
    _save(fig, name, save_dir)
    return fig


def direct_vs_own_chance(results, unit_regions, groups=None, n_bins=4, seed=0,
                         n_shuffles=5, min_pairs=25):
    """Per recday x distance bin x region: the DIRECT metric and its own pair-identity chance.

    The direct metric `D_ij(Y) - D_ij(X)` shares `D_ij(X)` with the bin `|D_ij(X)|`, so its
    chance level is bin-dependent, not 1/16 (see `pair_identity_null_by_distance`). The only
    honest way to plot it conditioned on distance is against a null computed the same way in
    the same bin -- which is what this returns.

    Null: keep `D_ij(X)`, but compare it against a RANDOM OTHER pair's angle in each
    comparison task. Destroys every real coherence relation while preserving that task's
    marginal distribution of pair angles, which is exactly what makes chance move.

    `group == 'ALL'` is every included pair; the others are within-region pairs.
    """
    groups = groups or asplit.ANALYSIS_GROUPS
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, 180, n_bins + 1)
    rows = []
    for rd, r in sorted(results.items()):
        idx = r.get('included_idx', np.array([], dtype=int))
        uniq = r.get('unique_sessions', [])
        if len(idx) < 3 or len(uniq) < 3 or rd not in unit_regions:
            continue
        # one pairwise matrix per task over ALL included neurons; region subsets are
        # sub-blocks of it, so the expensive part is done once per recday
        mats = {s: asplit.pairwise_angles(r['tuning'][s][idx]) for s in uniq}
        reg = unit_regions[rd]['group'].to_numpy()[idx]
        X, others = uniq[0], list(uniq[1:])
        subsets = [('ALL', np.arange(len(idx)))]
        subsets += [(g, np.flatnonzero(reg == g)) for g in groups]
        for gname, sub in subsets:
            if len(sub) < 3:
                continue
            iu = np.triu_indices(len(sub), 1)
            blocks = {s: mats[s][np.ix_(sub, sub)][iu] for s in uniq}
            dX = blocks[X]
            real = dual_per_row(np.column_stack(
                [w3._wrap(blocks[s] - dX) for s in others]))
            null = np.nanmean([
                dual_per_row(np.column_stack(
                    [w3._wrap(blocks[s][rng.permutation(len(dX))] - dX) for s in others]))
                for _ in range(n_shuffles)], axis=0)
            b = np.digitize(np.abs(dX), edges[1:-1])
            for k in range(n_bins):
                m = b == k
                if m.sum() < min_pairs:
                    continue
                rows.append({'recday': rd, 'mouse': rd.split('_')[0], 'group': gname,
                             'bin': k, 'label': f'{edges[k]:.0f}-{edges[k + 1]:.0f}\u00b0',
                             'real': float(np.nanmean(real[m])),
                             'chance': float(np.nanmean(null[m])),
                             'n_pairs': int(m.sum())})
    df = pd.DataFrame(rows)
    if len(df):
        df['ratio'] = df['real'] / df['chance']
        df['gap'] = df['real'] - df['chance']
    return df


def fig7_direct_vs_own_chance(results, unit_regions, groups=None, n_bins=4, save_dir=None,
                              name='fig7_direct_vs_own_chance', df=None):
    """The direct metric plotted against its OWN chance, globally and per region.

    This is the figure that reconciles W3 with the earlier prototype. The dotted line is the
    flat 1/16 the prototype tested against; the dashed line is what chance ACTUALLY is in
    each bin. Against the flat line the distal bins look null; against the real one they are
    above chance everywhere.
    """
    import matplotlib.pyplot as plt

    groups = groups or asplit.ANALYSIS_GROUPS
    if df is None:
        df = direct_vs_own_chance(results, unit_regions, groups=groups, n_bins=n_bins)
    if not len(df):
        print('fig7: nothing to plot')
        return None, df
    panels = [g for g in ['ALL'] + list(groups) if (df['group'] == g).any()]

    fig, axes = plt.subplots(1, len(panels), figsize=(2.35 * len(panels), 3.9))
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, panels):
        d = df[df['group'] == g]
        bins = sorted(d['bin'].unique())
        labels = [d.loc[d['bin'] == b, 'label'].iloc[0] for b in bins]
        x = np.arange(len(bins))
        col = REAL_COLOR if g == 'ALL' else asplit.REGION_COLORS.get(g, NEUTRAL)
        per_mouse = d.groupby(['bin', 'mouse'])[['real', 'chance']].mean().reset_index()
        for i, b in enumerate(bins):
            v = d.loc[d['bin'] == b, 'real'].to_numpy()
            ax.plot(np.full(len(v), i) - 0.11, v, 'o', mfc='none', mec='0.7', ms=2.3, lw=0,
                    zorder=2)
            mv = per_mouse.loc[per_mouse['bin'] == b, 'real'].to_numpy()
            ax.plot(np.full(len(mv), i) + 0.11, mv, 'o', color=col, ms=3.4, lw=0, zorder=3)
        ax.plot(x, [d.loc[d['bin'] == b, 'real'].mean() for b in bins], '-', color=col,
                lw=1.7, zorder=3, label='direct metric')
        ax.plot(x, [d.loc[d['bin'] == b, 'chance'].mean() for b in bins], '--s',
                color=FLOOR_COLOR, ms=3.2, lw=1.2, zorder=3, label='its OWN chance')
        ax.axhline(DUAL_CHANCE, color='k', ls=':', lw=0.9,
                   label='flat 1/16 (wrong here)' if g == panels[0] else None)
        top = d['real'].max()
        for i, b in enumerate(bins):
            sub = d[d['bin'] == b]
            pm = per_mouse[per_mouse['bin'] == b]
            t_rd = test_vs_shuffle(sub['real'], sub['chance'])
            t_ms = test_vs_shuffle(pm['real'], pm['chance'])
            ax.text(i, top * 1.03, f"{t_rd.get('stars', '')}\n{t_ms.get('stars', '')}",
                    ha='center', va='bottom', fontsize=5)
        ax.set_ylim(0, top * 1.42)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.5, rotation=45, ha='right')
        n_mice = d['mouse'].nunique()
        caveat = asplit.SINGLE_MOUSE_GROUPS.get(g, '')
        ax.set_title(('all included pairs' if g == 'ALL' else g)
                     + f"\n{d['recday'].nunique()} recdays, {n_mice} "
                       f"{'mouse' if n_mice == 1 else 'mice'}"
                     + (f'\n{caveat}' if caveat else ''), fontsize=6.5)
        if g == panels[0]:
            ax.set_ylabel('DUAL coherence, DIRECT metric')
            ax.legend(fontsize=5.5, frameon=False, loc='upper right')
    fig.suptitle('The direct metric against its OWN chance level, not a flat 1/16. Dotted = '
                 '1/16 (what the earlier prototype tested against);\ndashed = the real, '
                 'bin-dependent chance from a pair-identity shuffle. Open = recday, filled = '
                 'mouse;\nstars = paired test vs own chance (top: recday, bottom: mouse).',
                 fontsize=7, y=0.99)
    fig.subplots_adjust(top=0.70, bottom=0.22, wspace=0.36, left=0.06, right=0.98)
    _save(fig, name, save_dir)
    return fig, df


def pair_identity_null_by_distance(results, n_bins=4, seed=0, max_recdays=None):
    """Why a t-test against 1/16 is INVALID for the direct metric binned by initial distance.

    The direct metric is `D_ij(Y) - D_ij(X)` and the bin is `|D_ij(X)|` -- the pair's own
    angle appears in both. Its chance level is therefore NOT 1/16 but
    `1/16 * (marginal frequency of that angle / 0.25)**2`: a pair at 0 deg is likely to meet
    another pair at 0 deg by accident, because pair angles pile up near 0 (measured marginal
    0.299 in the 0-45 bin against 0.25 for uniform). The ref-anchored metric `r_j - r_i` is
    built from each neuron's OWN rotation and never conditions on `D_ij(X)`, so 1/16 is the
    right yardstick for it -- which is why the figures use it.

    This matters historically: the earlier prototype in
    `LEC_sploratory_analysis_with_glm_and_population.ipynb` (cells 341/347/350) used the
    direct metric against a flat 1/16 and read the distal bins as non-significant. Against
    their OWN null they are 1.3x above chance.

    The null here is a PAIR-IDENTITY shuffle: keep `D_ij(X)`, but compare it against a random
    other pair's `D(Y)` and `D(Z)`. That destroys all real coherence while preserving each
    task's marginal distribution of pair angles, which is exactly the thing that makes the
    chance level bin-dependent.

    Returns a DataFrame with, per bin, the direct metric's real rate and its own chance, the
    ref-anchored rate, and the flat 1/16 for comparison.
    """
    rng = np.random.default_rng(seed)
    edges = list(np.linspace(0, 180, n_bins + 1)[1:-1])
    labels = [f'{np.linspace(0, 180, n_bins + 1)[b]:.0f}-'
              f'{np.linspace(0, 180, n_bins + 1)[b + 1]:.0f}deg' for b in range(n_bins)]
    acc = {k: {l: [] for l in labels} for k in ('direct', 'direct_null', 'ref')}
    items = sorted(results.items())[:max_recdays] if max_recdays else sorted(results.items())
    for _rd, r in items:
        uniq, tun = r['unique_sessions'], r['tuning']
        idx = r.get('included_idx', np.array([], dtype=int))
        if len(idx) < 3 or len(uniq) < 3:
            continue
        iu = np.triu_indices(len(idx), 1)
        X, Y, Z = uniq[0], uniq[1], uniq[2]
        dX = asplit.pairwise_angles(tun[X][idx])[iu]
        dY = asplit.pairwise_angles(tun[Y][idx])[iu]
        dZ = asplit.pairwise_angles(tun[Z][idx])[iu]
        thr = w3.GEN_THRESHOLD_DEG
        real = (np.abs(w3._wrap(dY - dX)) < thr) & (np.abs(w3._wrap(dZ - dX)) < thr)
        p1, p2 = rng.permutation(len(dX)), rng.permutation(len(dX))
        null = (np.abs(w3._wrap(dY[p1] - dX)) < thr) & (np.abs(w3._wrap(dZ[p2] - dX)) < thr)
        rot = {t: w3.angles_between(tun[X][idx], tun[t][idx]) for t in (Y, Z)}
        ref = np.ones(len(dX), dtype=bool)
        for t in (Y, Z):
            ref &= np.abs(w3._wrap(rot[t][iu[1]] - rot[t][iu[0]])) < thr
        b = np.digitize(np.abs(dX), edges)
        for k, l in enumerate(labels):
            m = b == k
            if m.sum() > 5:
                acc['direct'][l].append(real[m].mean())
                acc['direct_null'][l].append(null[m].mean())
                acc['ref'][l].append(ref[m].mean())
    rows = []
    for l in labels:
        if not acc['direct'][l]:
            continue
        d, dn = np.mean(acc['direct'][l]), np.mean(acc['direct_null'][l])
        rows.append({'bin': l.replace('deg', '\u00b0'), 'n_recdays': len(acc['direct'][l]),
                     'direct': d, 'direct_own_chance': dn,
                     'direct_over_own_chance': d / dn if dn else np.nan,
                     'ref_anchored': np.mean(acc['ref'][l]), 'flat_chance': DUAL_CHANCE})
    return pd.DataFrame(rows)


def ccgp_levels(df):
    """CCGP collapsed the standard way: draws -> recday -> mouse."""
    per_recday = (df.groupby(['recday', 'mouse', 'group'])
                  [['acc', 'null_mean', 'ceiling', 'n_units']].mean().reset_index())
    per_mouse = (per_recday.groupby(['mouse', 'group'])
                 [['acc', 'null_mean', 'ceiling', 'n_units']].mean().reset_index())
    return per_recday, per_mouse


def fig8_ccgp_by_region(ccgp_matched=None, ccgp_full=None, groups=None, save_dir=None,
                        name='fig8_ccgp_by_region'):
    """State-pair CCGP per region -- the population form of the Stage B question.

    A decoder trained on some tasks and tested on a held-out one generalises only if state
    identity is abstract across tasks. This can find structure distributed across cells that
    is invisible one neuron at a time, so it is a genuine independent check on figure 1.

    Two panels, answering different questions:

    - **matched n = 20** -- every region subsampled to the same neuron count, so regions can
      be COMPARED. Decoder accuracy scales with n and per-recday region counts differ
      two-fold, so an unmatched comparison would measure yield.
    - **full n** -- each region at its actual count. A WITHIN-region existence claim against
      that region's own null, never a cross-region comparison: full-n ENTl-deep at 38 units
      beating full-n ENTm at 24 would be expected from n alone. Mean n is printed per bar.

    The `ceiling` triangles are within-task cross-validated accuracy -- what the same
    population achieves when it does NOT have to generalise. They are the reason a low `acc`
    reads as "does not generalise" rather than "these cells carry no state information".
    """
    import matplotlib.pyplot as plt

    groups = groups or asplit.ANALYSIS_GROUPS
    panels = [(ccgp_matched, 'matched n = 20 — comparable across regions'),
              (ccgp_full, 'full n — WITHIN-region existence only, never a cross-region test')]
    panels = [(d, t) for d, t in panels if d is not None and len(d)]
    if not panels:
        print('fig8: no CCGP results')
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (df, title) in zip(axes, panels):
        per_recday, per_mouse = ccgp_levels(df)
        present = [g for g in groups if (per_mouse['group'] == g).any()]
        for k, g in enumerate(present):
            rd = per_recday[per_recday['group'] == g]
            pm = per_mouse[per_mouse['group'] == g]
            col = asplit.REGION_COLORS.get(g, NEUTRAL)
            ax.bar(k, pm['acc'].mean(), color=col, alpha=0.4, width=0.7, zorder=0)
            ax.plot(np.full(len(rd), k) - 0.16, rd['acc'], 'o', mfc='none', mec='0.7',
                    ms=2.6, lw=0, zorder=2, label='recday' if k == 0 else None)
            ax.plot(np.full(len(rd), k) - 0.16, rd['null_mean'], '_', color=FLOOR_COLOR,
                    ms=6, zorder=2, label='role-permutation null' if k == 0 else None)
            mx = k + np.linspace(0.04, 0.34, len(pm))
            ax.plot(mx, pm['acc'], 'o', color='0.25', ms=4, lw=0, zorder=3,
                    label='mouse' if k == 0 else None)
            ax.plot(np.full(len(rd), k) - 0.16, rd['ceiling'], '^', color=CEILING_COLOR,
                    ms=3.2, lw=0, zorder=2,
                    label='within-task ceiling' if k == 0 else None)
            _stagger(ax, mx, pm['acc'].to_numpy(), list(pm['mouse']))
            t_rd = test_vs_shuffle(rd['null_mean'], rd['acc'])   # null > acc = fails to generalise
            n_mice = pm['mouse'].nunique()
            ax.text(k, 0.845, f"{len(rd)} rd, {n_mice} "
                              f"{'mouse' if n_mice == 1 else 'mice'}\n"
                              f"n={pm['n_units'].mean():.0f}\n"
                              f"{t_rd.get('n_positive', 0)}/{t_rd.get('n', 0)} below null",
                    ha='center', va='bottom', fontsize=5.2)
        ax.axhline(0.5, color='k', ls=':', lw=0.9, label='chance 0.5' if True else None)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, fontsize=6.5)
        ax.set_ylim(0.30, 0.98)
        ax.set_ylabel('balanced accuracy, held-out task')
        ax.set_title(title, fontsize=7.5)
        ax.legend(fontsize=5.5, frameon=False, ncol=2, loc='lower left')
    fig.suptitle('State-pair CCGP by region. Decoders read state WITHIN a task (triangles) '
                 'but sit at or below their own null across tasks (bars vs dashes)\n'
                 '\u2014 in every region. The population form of the figure-1 null.',
                 fontsize=8, y=0.99)
    fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.22, left=0.06, right=0.98)
    _save(fig, name, save_dir)
    return fig


# ---------------------------------------------------------------------------
# The reliability floor
# ---------------------------------------------------------------------------

def splithalf_floor_by_region(data_dic, neurons, groups=None, gated_only=True, quiet=True):
    """Within-session split-half rotation angle per region — the measurement noise floor.

    First half of a session's trials against the last half, same neuron, same task. Whatever
    spread this returns is what the estimator produces when there IS no remapping, so it
    bounds how much of any regional difference in figure 2 could be measurement noise. It is
    rate-dependent in the same direction as the gate, which is exactly why it has to be read
    alongside the gate rather than instead of it.
    """
    import contextlib
    import io

    import splithalf_ratemap_consistency as sh

    groups = groups or asplit.ANALYSIS_GROUPS
    rows = []
    for rd, sub in neurons.groupby('recday'):
        if rd not in data_dic:
            continue
        sel = sub[sub['included']] if gated_only else sub
        for g in groups:
            idx = sel.loc[sel['group'] == g, 'neuron'].to_numpy()
            if len(idx) < 3:
                continue
            buf = io.StringIO()
            with (contextlib.redirect_stdout(buf) if quiet
                  else contextlib.nullcontext()):
                angles, corrs, _ = sh.compute_splithalf_remapping_angles(
                    data_dic, rd, valid_sessions=None, neuron_subset=idx)
            if not len(angles):
                continue
            deg = w3._wrap(np.degrees(np.asarray(angles, dtype=float)))
            rows.append(pd.DataFrame({'recday': rd, 'mouse': rd.split('_')[0], 'group': g,
                                      'angle_deg': deg,
                                      'best_corr': np.asarray(corrs, dtype=float)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
