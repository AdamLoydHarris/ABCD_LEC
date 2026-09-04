"""Shared machinery for the anatomy split: the region join, and the statistics.

Every anatomy-split notebook imports this. The point is that the join and the inference
design exist exactly once -- four notebooks each rolling their own `groupby('group')` is how
the mouse-vs-recday confound gets quietly dropped from one of them.

The two things this module exists to enforce
--------------------------------------------
**1. The join is positional and must be asserted.** `unit_regions[recday]` has one row per QC
unit in `QC_single_units.npy` order, i.e. row *k* is `Neuron_raw` row *k*. Nothing matches
by name. A cached result computed on a different unit count silently misaligns every neuron
after the first mismatch, which is exactly how `ly05_20250618_20250619` (91 cached rows vs
109 units) would corrupt a whole figure. `join_regions` refuses to join unless the lengths
agree.

**2. Recdays are not independent replicates; mice are.** The 5 recdays of a mouse are the
same probe in the same brain, re-sorted, so a single physical neuron plausibly appears in
several. 2851 is a count of unit-*recordings*. The effective n for any anatomical claim is
the number of MICE, which is at most 5 and for most contrasts is 1-3. `cluster_bootstrap`
resamples mice; `within_recday_permutation` shuffles region labels inside a recday so each
recday's region composition and n are preserved. A null that shuffles across recdays is
degenerate -- it breaks the mouse<->region association that is the whole confound.

See `docs/ANATOMY_SPLIT_PLAN.md` and `code/ANATOMY_SPLIT.md`.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
UNIT_REGIONS_PATH = os.path.join(REPO, 'data', 'processed_data', 'unit_regions.pkl')

#: Coarse region labels, ordered superficial -> deep along a typical trajectory.
GROUP_ORDER = ['ENTl-sup', 'ENTl-deep', 'ENTm', 'SUB/ProS', 'CA1/HPF', 'fibre/other']

#: Groups that carry a claim. `fibre/other` is white matter and unassigned channels.
ANALYSIS_GROUPS = ['ENTl-sup', 'ENTl-deep', 'ENTm', 'SUB/ProS', 'CA1/HPF']

#: The only contrast replicated across >=3 animals (ah10, ly06, ly07). Everything else is
#: secondary (ENTl-deep vs CA1/HPF, 2 mice) or descriptive (1 mouse).
PRIMARY_CONTRAST = ('ENTl-deep', 'SUB/ProS')

#: GridMaze palette for the anatomy. A cool-to-warm ramp ordered the way the probe passes
#: through the tissue (superficial ENTl -> deep ENTl -> ENTm -> SUB/ProS -> CA1/HPF), built
#: from named GridMaze colours so it sits alongside the rest of the figures: Turquoise,
#: Classic Blue, Ultra Violet, Viva Magenta, Living Coral. `fibre/other` takes Stone, the
#: neutral, because it is not a claim-bearing group.
REGION_COLORS = {
    'ENTl-sup':    '#45B5AA',
    'ENTl-deep':   '#0F4C81',
    'ENTm':        '#6B3FA0',
    'SUB/ProS':    '#BE3455',
    'CA1/HPF':     '#FF6F61',
    'fibre/other': '#B4B2A9',
}

#: Regions whose pooled units are dominated by a single animal. Any pooled claim about one
#: of these is a claim about that mouse wearing a region label -- caption it that way.
SINGLE_MOUSE_GROUPS = {'ENTl-sup': 'ah08 (611/682 units)', 'ENTm': 'ly07 (179/284 units)'}


def load_unit_regions(path=UNIT_REGIONS_PATH):
    """`{recday: DataFrame}`, one row per QC unit in `Neuron_raw` row order."""
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def join_regions(result, unit_regions, value_name='value', *,
                 strict=True, expect_recdays=None):
    """Join a per-neuron result onto region labels, positionally, with a hard length check.

    Parameters
    ----------
    result : dict
        `{recday: array}` where `array` has one entry (or one row) per neuron, in
        `Neuron_raw` row order. 1-D gives one column `value_name`; 2-D of shape
        (n_neurons, k) gives `f'{value_name}_{i}'` per column.
    unit_regions : dict
        From `load_unit_regions()`.
    strict : bool, default True
        Raise on any length mismatch. Set False only to survey what is broken -- mismatched
        recdays are then DROPPED, never truncated, because truncation would silently keep
        the first N neurons of a misaligned array.
    expect_recdays : iterable of str, optional
        If given, warn about recdays present here but absent from `result`.

    Returns
    -------
    DataFrame with recday, mouse, neuron, group, acronym, shank, y_um, ap_i + value columns.
    """
    frames, problems = [], []
    for recday, arr in result.items():
        if recday not in unit_regions:
            problems.append((recday, None, None, 'absent from unit_regions'))
            continue
        reg = unit_regions[recday]
        a = np.asarray(arr)
        n = a.shape[0]
        if n != len(reg):
            problems.append((recday, n, len(reg), 'length mismatch'))
            continue
        cols = ['mouse', 'group', 'acronym', 'shank', 'y_um', 'ap_i']
        df = reg[[c for c in cols if c in reg.columns]].copy().reset_index(drop=True)
        df.insert(0, 'recday', recday)
        df.insert(2, 'neuron', np.arange(n))
        if a.ndim == 1:
            df[value_name] = a
        else:
            for j in range(a.shape[1]):
                df[f'{value_name}_{j}'] = a[:, j]
        frames.append(df)

    if problems:
        msg = '\n'.join(
            f"  {rd}: result has {nr} rows, unit_regions has {nu} — {why}"
            for rd, nr, nu, why in problems)
        header = (f"join_regions: {len(problems)} recday(s) could not be joined. The join is "
                  f"POSITIONAL, so a mismatch means the two disagree about which units exist "
                  f"(the ly05 recday-mismatch signature) and joining anyway would misalign "
                  f"every neuron:\n{msg}")
        if strict:
            raise AssertionError(header)
        print('WARNING: ' + header + '\n  (dropped, not truncated)')

    if not frames:
        raise ValueError('join_regions: nothing joined')
    out = pd.concat(frames, ignore_index=True)
    if 'mouse' not in out.columns:
        out['mouse'] = out['recday'].str.split('_').str[0]

    if expect_recdays is not None:
        missing = sorted(set(expect_recdays) - set(result))
        if missing:
            print(f"join_regions: {len(missing)} expected recday(s) absent from the result "
                  f"(a stale cache does not cover them): {missing}")
    return out


def assert_glm_keys_contiguous(glm_results, unit_regions, strict=True):
    """Check that per-neuron GLM dicts are keyed 0..n-1 with no gaps.

    `compute_tuning_arrays` builds its output by enumerating `sorted(GLM_results[rd])` and
    writing row *k* for the *k*-th key -- so the row index is a POSITION in that sorted key
    list, not the neuron id. If a neuron were missing from `GLM_results`, every row after
    the gap would shift by one relative to `Neuron_raw`, and the length gate would not catch
    it whenever the count happened to match. Verified clean for all 24 cached recdays except
    the known-stale `ly05_20250618_20250619` (91 keys vs 109 units), which the length gate
    rejects anyway.
    """
    problems = []
    for rd, per_neuron in glm_results.items():
        keys = sorted(per_neuron.keys())
        if keys != list(range(len(keys))):
            problems.append(f"  {rd}: keys are not contiguous 0..n-1 "
                            f"(first {keys[:3]}, last {keys[-3:]}, n={len(keys)})")
        elif rd in unit_regions and len(keys) != len(unit_regions[rd]):
            problems.append(f"  {rd}: {len(keys)} GLM neurons vs "
                            f"{len(unit_regions[rd])} units in unit_regions")
    if problems:
        msg = ("assert_glm_keys_contiguous: tuned_dict rows would not align with Neuron_raw "
               "rows:\n" + '\n'.join(problems))
        if strict:
            raise AssertionError(msg)
        print('WARNING: ' + msg)
    return not problems


def boundary_margin_filter(joined, margin_um=50.0, unit_regions=None):
    """Flag units within `margin_um` of a region transition along their own shank.

    The local deformation field runs 8.06-12.95 um per voxel, so a structure boundary carries
    at least that much positional uncertainty -- and the empirical ENTl-deep/ENTm separation
    in ly07 wobbles by 15-75 um between recdays (see `w0_gates.gate_depth_ordering`). If a
    contrast depends on the units nearest a boundary, it is not robust to the registration.

    Adds a boolean column `near_boundary`. Filter with `joined[~joined.near_boundary]`.
    """
    df = joined.copy()
    if 'shank' not in df.columns or 'y_um' not in df.columns:
        raise KeyError('boundary_margin_filter needs `shank` and `y_um` columns')
    near = np.zeros(len(df), dtype=bool)
    for (_rd, _sh), sub in df.groupby(['recday', 'shank'], sort=False):
        s = sub.sort_values('y_um')
        y = s['y_um'].to_numpy(dtype=float)
        g = s['group'].to_numpy()
        # A transition sits between two adjacent units with different labels; take its depth
        # as their midpoint. Units within `margin_um` of ANY transition are flagged.
        edges = [(y[i] + y[i + 1]) / 2.0 for i in range(len(y) - 1) if g[i] != g[i + 1]]
        if not edges:
            continue
        d = np.min(np.abs(y[:, None] - np.asarray(edges)[None, :]), axis=1)
        near[s.index.to_numpy()] = d < margin_um
    df['near_boundary'] = near
    return df


# ---------------------------------------------------------------------------
# Feasibility -- run BEFORE an analysis, not after
# ---------------------------------------------------------------------------

def feasibility_table(joined, mask_col=None, min_units=20, groups=None):
    """Units per region x recday, after whatever gate an analysis applies.

    `mask_col` is a boolean column naming the units that survived the analysis's own gate
    (e.g. state-tuned). Pass None for the raw census.

    Per-recday counts are the binding constraint on this dataset, not the pooled 2851:
    ENTl-deep runs 15-80 per recday and SUB/ProS 0-41, so an analysis needing >=20 units per
    region per recday loses contrasts that look comfortable in the pooled table. Publish
    this before running the analysis and drop the contrasts that cannot clear it.
    """
    groups = groups or ANALYSIS_GROUPS
    d = joined[joined[mask_col]] if mask_col else joined
    counts = (d.groupby(['recday', 'group']).size().unstack(fill_value=0)
              .reindex(columns=groups, fill_value=0)
              .reindex(index=sorted(joined['recday'].unique()), fill_value=0))
    counts.insert(0, 'mouse', [r.split('_')[0] for r in counts.index])

    print(f"Feasibility at min_units={min_units} per region per recday"
          + (f" (gate: {mask_col})" if mask_col else " (no gate)"))
    rows = []
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            ga, gb = groups[a], groups[b]
            ok = counts[(counts[ga] >= min_units) & (counts[gb] >= min_units)]
            rows.append({'contrast': f'{ga} vs {gb}',
                         'recdays': len(ok), 'mice': ok['mouse'].nunique(),
                         'which_mice': ','.join(sorted(ok['mouse'].unique())),
                         'usable_as': ('primary' if ok['mouse'].nunique() >= 3
                                       else 'secondary' if ok['mouse'].nunique() == 2
                                       else 'descriptive only')})
    feas = pd.DataFrame(rows).sort_values('mice', ascending=False).reset_index(drop=True)
    return counts, feas


# ---------------------------------------------------------------------------
# Statistics: region first, then contrasts, with mice as the unit
# ---------------------------------------------------------------------------

def _agg(values, statistic):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    if statistic == 'mean':
        return float(np.mean(v))
    if statistic == 'median':
        return float(np.median(v))
    if statistic == 'fraction':
        return float(np.mean(v != 0))
    if callable(statistic):
        return float(statistic(v))
    raise ValueError(f'unknown statistic {statistic!r}')


def per_recday_effect(joined, value_col, statistic='mean', groups=None, min_units=1):
    """One number per (recday, group) -- the atom every other statistic is built from."""
    groups = groups or ANALYSIS_GROUPS
    rows = []
    for (rd, g), sub in joined[joined['group'].isin(groups)].groupby(['recday', 'group']):
        v = sub[value_col].to_numpy(dtype=float)
        if np.isfinite(v).sum() < min_units:
            continue
        rows.append({'recday': rd, 'mouse': rd.split('_')[0], 'group': g,
                     'n_units': int(np.isfinite(v).sum()),
                     'effect': _agg(v, statistic)})
    return pd.DataFrame(rows)


def per_mouse_effect(joined, value_col, statistic='mean', groups=None, min_units=1):
    """Collapse recdays within a mouse, so each mouse contributes ONE number per region.

    This is the level at which contrasts are legitimate. The recdays of a mouse share a
    probe and a brain, so averaging them is pooling repeated measures of the same thing,
    not gaining n.
    """
    per_rd = per_recday_effect(joined, value_col, statistic, groups, min_units)
    if not len(per_rd):
        return per_rd
    return (per_rd.groupby(['mouse', 'group'])
            .agg(effect=('effect', 'mean'), n_recdays=('effect', 'size'),
                 n_units=('n_units', 'sum'))
            .reset_index())


def per_region_report(joined, value_col, statistic='mean', groups=None, min_units=1,
                      n_boot=10000, seed=0):
    """Every region on its own, before any contrast is drawn.

    Returns one row per region: n units / recdays / mice, the per-mouse effects listed
    individually, and a mouse-level bootstrap CI. A region with n=1 mouse still gets a row
    -- labelled descriptive -- it just never grounds a contrast claim. This ordering matters
    because the mouse<->region confound lets a contrast look clean while both of its arms
    are single-mouse.
    """
    groups = groups or ANALYSIS_GROUPS
    per_m = per_mouse_effect(joined, value_col, statistic, groups, min_units)
    rng = np.random.default_rng(seed)
    rows = []
    for g in groups:
        sub = per_m[per_m['group'] == g]
        if not len(sub):
            continue
        eff = sub['effect'].to_numpy(dtype=float)
        mice = sub['mouse'].to_numpy()
        if len(eff) > 1:
            boot = np.array([np.mean(rng.choice(eff, len(eff), replace=True))
                             for _ in range(n_boot)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
        else:
            lo = hi = np.nan
        rows.append({
            'group': g,
            'n_mice': len(eff), 'n_recdays': int(sub['n_recdays'].sum()),
            'n_units': int(sub['n_units'].sum()),
            'effect_pooled_over_mice': float(np.mean(eff)),
            'ci_lo': lo, 'ci_hi': hi,
            'per_mouse': {m: round(float(e), 4) for m, e in zip(mice, eff)},
            'evidence': ('primary-capable' if len(eff) >= 3 else
                         'secondary' if len(eff) == 2 else 'descriptive (1 mouse)'),
            'single_mouse_caveat': SINGLE_MOUSE_GROUPS.get(g, ''),
        })
    return pd.DataFrame(rows)


def cluster_bootstrap(joined, value_col, group_a, group_b, statistic='mean',
                      n_boot=10000, seed=0, min_units=1):
    """Contrast two regions, resampling MICE (not units, not recdays).

    Resamples mice with replacement; each drawn mouse brings its own per-region effect. With
    n=3 mice the CI is wide and should be -- reporting a pooled p over 2851 units instead
    would be wrong by a large factor.
    """
    per_m = per_mouse_effect(joined, value_col, statistic, [group_a, group_b], min_units)
    wide = per_m.pivot(index='mouse', columns='group', values='effect').dropna()
    if len(wide) == 0:
        return {'n_mice': 0, 'note': 'no mouse has both regions'}
    d = (wide[group_a] - wide[group_b]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    return {
        'contrast': f'{group_a} - {group_b}',
        'n_mice': int(len(d)),
        'per_mouse_diff': {m: round(float(x), 4) for m, x in zip(wide.index, d)},
        'mean_diff': float(np.mean(d)), 'ci_lo': float(lo), 'ci_hi': float(hi),
        'p_boot': min(1.0, p),
        'usable_as': ('primary' if len(d) >= 3 else
                      'secondary' if len(d) == 2 else 'descriptive only'),
    }


def within_recday_permutation(joined, value_col, group_a, group_b, statistic='mean',
                              n_perm=10000, seed=0, min_units=1):
    """Null for a region contrast: shuffle `group` WITHIN each recday.

    Preserves each recday's region composition and n, and preserves the mouse<->region
    association -- which is the confound, so it must survive into the null. Shuffling labels
    across recdays would let a "null" sample put SUB/ProS units in ah08, which never happens
    in the data, and would make the null far too easy to beat.
    """
    rng = np.random.default_rng(seed)
    sub = joined[joined['group'].isin([group_a, group_b])].copy()
    obs = _contrast_from_labels(sub, value_col, sub['group'].to_numpy(),
                                group_a, group_b, statistic, min_units)
    idx_by_recday = [np.flatnonzero(sub['recday'].to_numpy() == rd)
                     for rd in sub['recday'].unique()]
    labels = sub['group'].to_numpy().copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = labels.copy()
        for idx in idx_by_recday:
            perm[idx] = rng.permutation(perm[idx])
        null[i] = _contrast_from_labels(sub, value_col, perm, group_a, group_b,
                                        statistic, min_units)
    finite = null[np.isfinite(null)]
    p = (1 + np.sum(np.abs(finite) >= abs(obs))) / (1 + len(finite))
    return {'contrast': f'{group_a} - {group_b}', 'observed': obs,
            'null_mean': float(np.mean(finite)), 'null_sd': float(np.std(finite)),
            'p_perm': float(p), 'n_perm': int(len(finite))}


def _contrast_from_labels(sub, value_col, labels, group_a, group_b, statistic, min_units):
    """Mouse-level mean difference under an arbitrary label vector (used by the null)."""
    tmp = pd.DataFrame({'recday': sub['recday'].to_numpy(),
                        'mouse': sub['mouse'].to_numpy(),
                        'group': labels,
                        'v': sub[value_col].to_numpy(dtype=float)})
    per_rd = []
    for (rd, g), s in tmp.groupby(['recday', 'group']):
        if np.isfinite(s['v']).sum() < min_units:
            continue
        per_rd.append({'mouse': rd.split('_')[0], 'group': g,
                       'effect': _agg(s['v'], statistic)})
    if not per_rd:
        return np.nan
    pm = (pd.DataFrame(per_rd).groupby(['mouse', 'group'])['effect'].mean()
          .unstack().dropna())
    if group_a not in pm.columns or group_b not in pm.columns or not len(pm):
        return np.nan
    return float(np.mean(pm[group_a] - pm[group_b]))


def rate_match(joined, rate_col, group_a, group_b, n_bins=10, seed=0):
    """Subsample two regions to a matched firing-rate distribution, within each recday.

    Region is derived from the unit's max-amplitude channel, and depth drives amplitude,
    isolation and yield -- so region is confounded with rate. On this dataset that is not a
    small effect: median rate is 5.86 Hz in SUB/ProS against 1.91 Hz in ENTl-deep (a 3x
    difference on the PRIMARY contrast) and 1.07 Hz in ENTl-sup. Any statistic whose power
    scales with rate -- the state-tuning t-test, permutation significance, the reliability
    of a rotation estimate -- inherits that difference. If an effect survives rate matching
    it is not a yield artefact; if it does not, say so.

    Matching is stratified on log10 rate, within recday (so it cannot smuggle in a
    between-recday difference), taking min(n_a, n_b) units per bin from each group.
    """
    rng = np.random.default_rng(seed)
    sub = joined[joined['group'].isin([group_a, group_b])]
    keep = []
    for rd, s in sub.groupby('recday'):
        r = s[rate_col].to_numpy(dtype=float)
        ok = np.isfinite(r) & (r > 0)
        s, r = s[ok], r[ok]
        if not len(s):
            continue
        lr = np.log10(r)
        edges = np.linspace(lr.min(), lr.max() + 1e-9, n_bins + 1)
        b = np.clip(np.digitize(lr, edges) - 1, 0, n_bins - 1)
        ga = s['group'].to_numpy() == group_a
        for k in range(n_bins):
            ia = np.flatnonzero((b == k) & ga)
            ib = np.flatnonzero((b == k) & ~ga)
            m = min(len(ia), len(ib))
            if m:
                keep.append(s.index.to_numpy()[rng.choice(ia, m, replace=False)])
                keep.append(s.index.to_numpy()[rng.choice(ib, m, replace=False)])
    if not keep:
        return joined.iloc[:0]
    return joined.loc[np.concatenate(keep)]


# ---------------------------------------------------------------------------
# Remapping angle -- ONE implementation
# ---------------------------------------------------------------------------
# `compute_remapping_angle` is defined four separate times in
# LEC_sploratory_analysis_with_glm_and_population.ipynb (cells 337, 339, 370, 402) and they
# do not agree on the return unit -- three return a shift in bins, one returns radians.
# This is the single implementation. It returns DEGREES, signed, wrapped to (-180, 180],
# with 360 task-space bins so 1 bin = 1 deg and one task state = 90 deg.

def circular_xcorr(curve_a, curve_b):
    """Pearson r between `curve_a` rolled by every shift and `curve_b`, via FFT.

    Returns `(n_bins,)` where element *s* is `corr(roll(curve_a, s), curve_b)`. O(n log n)
    instead of the O(n^2) Python shift loop -- required for the pairwise coherence metric,
    which needs ~3.4M of these per recday (80 neurons x 6 tasks) and will not finish
    otherwise. NaNs are zero-filled after centring, matching the loop implementations.
    """
    a = np.asarray(curve_a, dtype=float)
    b = np.asarray(curve_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f'curve shape mismatch: {a.shape} vs {b.shape}')
    a = np.nan_to_num(a - np.nanmean(a), nan=0.0)
    b = np.nan_to_num(b - np.nanmean(b), nan=0.0)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.full(len(a), np.nan)
    n = len(a)
    # corr(roll(a, s), b) = sum_k a[k-s] b[k] = ifft(conj(fft(a)) * fft(b)) -- the standard
    # circular cross-correlation, normalised by the (shift-invariant) vector norms.
    cc = np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n=n)
    return cc / (na * nb)


def pairwise_angles(curves, method='xcorr'):
    """All-pairs remapping angles for one task's tuning curves, batched.

    Parameters
    ----------
    curves : (n_neurons, n_bins) task-space rate maps for a SINGLE task.
    method : {'xcorr', 'peak'} -- as `remapping_angle`.

    Returns
    -------
    (n_neurons, n_neurons) signed degrees in (-180, 180], NaN on the diagonal and for
    flat/all-NaN curves. Element [i, j] is the angle that rotates curve i onto j.

    Antisymmetric MOD 360, not exactly: a half-turn is its own inverse, so an exact 180 deg
    pair comes back as -180 in both directions. Harmless for everything downstream, which
    either thresholds |angle| (both signs give 180, so neither generalises) or takes
    circular statistics of the angle across tasks -- but do not assert `M == -M.T` on the
    nose.

    Why this exists rather than a loop over `remapping_angle`: at ~80 neurons x 6 tasks x 25
    recdays that is ~3.4M pair-correlations per recday, and one-call-at-a-time costs ~300 us
    each (dominated by per-call overhead, not the 360-point FFT) -- about 7 hours over the
    cohort. Transforming the whole matrix once and batching the inverse transform per row
    brings it to seconds.
    """
    X = np.asarray(curves, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'expected (n_neurons, n_bins), got {X.shape}')
    n, nb = X.shape
    out = np.full((n, n), np.nan)

    if method == 'peak':
        peaks = np.full(n, np.nan)
        ok = ~np.all(np.isnan(X), axis=1)
        peaks[ok] = np.nanargmax(X[ok], axis=1)
        d = (peaks[None, :] - peaks[:, None]) * (360.0 / nb)
        out = (d + 180.0) % 360.0 - 180.0
        bad = ~ok
        out[bad, :] = np.nan
        out[:, bad] = np.nan
        np.fill_diagonal(out, np.nan)
        return out
    if method != 'xcorr':
        raise ValueError(f"method must be 'xcorr' or 'peak', got {method!r}")

    Xc = X - np.nanmean(X, axis=1, keepdims=True)
    Xc = np.nan_to_num(Xc, nan=0.0)
    norms = np.linalg.norm(Xc, axis=1)
    good = norms > 0
    F = np.fft.rfft(Xc, axis=1)                       # (n, nb//2 + 1)
    for i in np.flatnonzero(good):
        j = np.flatnonzero(good)
        j = j[j > i]
        if not len(j):
            continue
        cc = np.fft.irfft(np.conj(F[i])[None, :] * F[j], n=nb, axis=1)
        shifts = np.argmax(cc, axis=1).astype(float)
        deg = (shifts * (360.0 / nb) + 180.0) % 360.0 - 180.0
        out[i, j] = deg
        out[j, i] = -deg
    np.fill_diagonal(out, np.nan)
    return out


def remapping_angle(curve_a, curve_b, method='xcorr', n_states=4):
    """Angle (degrees) by which `curve_a` must rotate to match `curve_b`.

    Parameters
    ----------
    curve_a, curve_b : (n_bins,) task-space rate maps, n_bins = n_states * bins_per_state.
        With the standard 360 bins, 1 bin = 1 degree and one task state = 90 degrees.
    method : {'xcorr', 'peak'}
        ``'xcorr'`` -- the correlation-maximising circular shift, using the whole curve
        shape. A 30-bin shift on a 360-bin map gives 30 degrees.
        ``'peak'``  -- the circular difference of the argmax bins. A peak at bin 10 in one
        map and bin 100 in the other gives 90 degrees. Uses only the mode, so it is robust
        to overall profile but noisy for broad or multi-peaked tuning.

    Returns
    -------
    float : signed degrees in (-180, 180]. A generalising cell is one whose angle is near 0.
        NaN if either curve is flat or all-NaN.

    Both methods are reported, and so is their agreement rate: where they diverge, the cell
    has broad or bimodal tuning, which is itself worth knowing.
    """
    a = np.asarray(curve_a, dtype=float)
    b = np.asarray(curve_b, dtype=float)
    n = len(a)
    if method == 'xcorr':
        cc = circular_xcorr(a, b)
        if not np.any(np.isfinite(cc)):
            return np.nan
        shift = int(np.nanargmax(cc))
    elif method == 'peak':
        if np.all(np.isnan(a)) or np.all(np.isnan(b)):
            return np.nan
        # roll(a, s) puts a[argmax_a] at argmax_a + s, so the aligning shift is the
        # difference of the peak positions -- same sign convention as 'xcorr'.
        shift = int(np.nanargmax(b)) - int(np.nanargmax(a))
    else:
        raise ValueError(f"method must be 'xcorr' or 'peak', got {method!r}")
    deg = shift * (360.0 / n)
    return float((deg + 180.0) % 360.0 - 180.0)


def is_generalising(angles, threshold_deg=45.0):
    """A cell generalises if |angle| < threshold on EVERY task comparison.

    45 degrees matches `RemapConfig.coherence_threshold_deg`, and is half a task state, so a
    cell that has moved to a neighbouring state cannot pass.
    """
    a = np.asarray(angles, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    ok = np.abs(a) < threshold_deg
    return np.where(np.all(np.isfinite(a), axis=1), np.all(ok, axis=1), False)


__all__ = [
    'GROUP_ORDER', 'ANALYSIS_GROUPS', 'PRIMARY_CONTRAST', 'SINGLE_MOUSE_GROUPS',
    'REGION_COLORS',
    'load_unit_regions', 'join_regions', 'assert_glm_keys_contiguous',
    'boundary_margin_filter', 'feasibility_table',
    'per_recday_effect', 'per_mouse_effect', 'per_region_report', 'cluster_bootstrap',
    'within_recday_permutation', 'rate_match',
    'circular_xcorr', 'pairwise_angles', 'remapping_angle', 'is_generalising',
]
