"""Per-recday change in the CV statistics, v1 -> v4: the migration's acceptance test.

`docs/handoff/BAD_RECDAY_ly05_20250620_20250623.md` predicts the shape of the answer in
advance: `ly05_20250620_20250623` (spikes 2.6-38.6 s late under v1) should move far more than
any other recday, and 16 of 25 recdays shifted by under one 25 ms bin so should barely move.

    A large move in ly05_20250620_20250623 is the expected result;
    a large move in any other recday would need explaining.

Reads the CROSS-VALIDATED artefacts only. The in-sample `permutation_results` null is drawn
unseeded (glm_analysis_v3.py:2151), so ~2.5% of its tuning calls flip between two runs of the
SAME data -- that is a noise floor, not a v4 effect, and this script stays clear of it.

    python code/v4_asymmetry_table.py --v1 data/glm_outputs/LEC_shards \
        --v4 data/glm_outputs/LEC_v4_shards --section <stem> [--out <csv>]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BAD_RECDAY = 'ly05_20250620_20250623'


def _shards(d, section, artefact):
    pat = os.path.join(d, f'{section}__*__{artefact}.pkl')
    out = {}
    for p in sorted(glob.glob(pat)):
        m = re.search(rf'{re.escape(section)}__(.+?)__{artefact}\.pkl$', os.path.basename(p))
        if m:
            out[m.group(1)] = p
    return out


def _load(p):
    import pickle
    with open(p, 'rb') as fh:
        return pickle.load(fh)


def compare(cv1, cv4):
    """v1 vs v4 for one recday.

    The headline is the per-neuron Spearman rho of each regressor's CPD, NOT a relative
    change in the median. A relative change is dominated by whichever regressor has the
    smallest CPD: on ah10_20250616_20250617 `acceleration` has a median CPD of 0.00075, so
    an absolute move of 0.0006 reads as "+83%" while the neurons stay rank-correlated at
    rho = 0.91. Rho is scale-free and answers the question actually being asked -- did the
    fit change? -- so it is what the acceptance test is built on.
    """
    row = {'n_neurons': len(np.asarray(cv4['r2_cv']))}
    r1, r4 = np.asarray(cv1['r2_cv'], float), np.asarray(cv4['r2_cv'], float)
    ok = np.isfinite(r1) & np.isfinite(r4)
    row['r2_cv__v1'] = float(np.nanmedian(r1))
    row['r2_cv__v4'] = float(np.nanmedian(r4))
    row['r2_cv__rho'] = float(spearmanr(r1[ok], r4[ok]).statistic) if ok.sum() > 2 else np.nan
    for reg in cv1['cpd_cv']:
        a = np.asarray(cv1['cpd_cv'][reg], float)
        b = np.asarray(cv4['cpd_cv'][reg], float)
        m = np.isfinite(a) & np.isfinite(b)
        row[f'cpd_med__{reg}__v1'] = float(np.nanmedian(a))
        row[f'cpd_absd__{reg}'] = float(np.nanmedian(b) - np.nanmedian(a))
        row[f'cpd_rho__{reg}'] = float(spearmanr(a[m], b[m]).statistic) if m.sum() > 2 else np.nan
    for reg, v in cv1.get('p_freedman_lane__cpd', {}).items():
        row[f'nsig__{reg}__v1'] = int(np.sum(np.asarray(v, float) < 0.05))
    for reg, v in cv4.get('p_freedman_lane__cpd', {}).items():
        row[f'nsig__{reg}__v4'] = int(np.sum(np.asarray(v, float) < 0.05))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1', default='data/glm_outputs/LEC_shards')
    ap.add_argument('--v4', default='data/glm_outputs/LEC_v4_shards')
    ap.add_argument('--section',
                    default='core_progress_time__matched_250ms_decile_cap30s_tfrU10b')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    a, b = (_shards(d, args.section, 'cv_results') for d in (args.v1, args.v4))
    common = sorted(set(a) & set(b))
    print(f'section {args.section}')
    print(f'  v1 shards {len(a)}   v4 shards {len(b)}   comparable {len(common)}')
    only = sorted(set(a) ^ set(b))
    if only:
        print(f'  NOT comparable (one side only): {only}')
    if not common:
        return 1

    rows = []
    for rd in common:
        cv1, cv4 = _load(a[rd])['data'][rd], _load(b[rd])['data'][rd]
        n1, n4 = len(np.asarray(cv1['r2_cv'])), len(np.asarray(cv4['r2_cv']))
        if n1 != n4:
            print(f'  WARNING {rd}: {n1} vs {n4} neurons -- not comparable, skipped')
            continue
        rows.append({'recday': rd, **compare(cv1, cv4)})
    df = pd.DataFrame(rows)

    # The headline is HELD-OUT r2, per neuron: its change, and its rank correlation.
    #
    # Two statistics were tried and rejected first, both for the same reason -- they are
    # dominated by whichever regressor carries no signal:
    #   * relative change in median CPD: `acceleration` has a median CPD of -0.00024, so an
    #     absolute move of 0.0006 reads as "+83%";
    #   * `min` over the regressors' per-neuron CPD rank correlations: `acceleration` supplies
    #     the min in 23 of 25 recdays, and the rank of a quantity that is zero up to noise is
    #     noise. Across the 24 non-pathological recdays that statistic correlates with v1 fit
    #     quality at rho = +0.54 (p = 0.007), i.e. it mostly measures how weak a recday is.
    #
    # r2_cv is the model's held-out explanatory power per neuron: well scaled, signed, and
    # interpretable. A negative median means the fit is worse than predicting the mean, which
    # is exactly what misaligned spikes produce -- so it also gives the test a QUALITATIVE
    # criterion that no noise floor can manufacture.
    rho_cols = [c for c in df.columns if c.startswith('cpd_rho__')]
    df['min_cpd_rho'] = df[rho_cols].min(axis=1)        # kept for diagnosis, not the verdict
    df['med_cpd_rho'] = df[rho_cols].median(axis=1)
    df['d_r2_cv'] = df['r2_cv__v4'] - df['r2_cv__v1']
    df['moved'] = df['d_r2_cv'].abs()

    pd.set_option('display.width', 220)
    show = df.sort_values('moved', ascending=False)
    print('\n=== per-recday movement, ranked by |change in median held-out r2| ===')
    print(show[['recday', 'n_neurons', 'r2_cv__v1', 'r2_cv__v4', 'd_r2_cv',
                'r2_cv__rho', 'med_cpd_rho']].round(5).to_string(index=False))

    print('\n=== the acceptance test ===')
    if BAD_RECDAY not in set(df.recday):
        print(f'  {BAD_RECDAY} is NOT in the comparison -- the test cannot be run')
        verdict = 'INCOMPLETE'
    else:
        bad = float(df.loc[df.recday == BAD_RECDAY, 'moved'].iloc[0])
        others = df.loc[df.recday != BAD_RECDAY, 'moved']
        ratio = bad / max(others.max(), 1e-12)
        neg1 = sorted(df.loc[df.r2_cv__v1 < 0, 'recday'])
        neg4 = sorted(df.loc[df.r2_cv__v4 < 0, 'recday'])
        print(f'  {BAD_RECDAY}: median r2_cv '
              f'{float(df.loc[df.recday == BAD_RECDAY, "r2_cv__v1"].iloc[0]):+.5f} -> '
              f'{float(df.loc[df.recday == BAD_RECDAY, "r2_cv__v4"].iloc[0]):+.5f}'
              f'  (|change| {bad:.5f})')
        nxt = df.loc[others.idxmax(), 'recday']
        print(f'  next largest ({nxt}): {others.max():.5f}   ratio {ratio:.1f}x')
        print(f'  median |change| over the other 24: {others.median():.5f}')
        print(f'  recdays whose fit was WORSE THAN THE MEAN (median r2_cv < 0):')
        print(f'    v1: {neg1 or "none"}')
        print(f'    v4: {neg4 or "none"}')
        quality_repaired = BAD_RECDAY in neg1 and BAD_RECDAY not in neg4
        verdict = ('PASS' if (ratio > 3 and quality_repaired)
                   else 'FAIL — another recday moved comparably')
        print(f'  VERDICT: {verdict}')
        if not quality_repaired:
            print(f'  -> {BAD_RECDAY} was not the uniquely pathological fit, or was not '
                  f'repaired. That is not what the clock diagnostics predict.')
        if ratio <= 3:
            print('  -> chase this. 16 of 25 recdays shifted by under one 25 ms bin, so a '
                  'comparable move elsewhere is most likely a plumbing error, not a result.')

    if args.out:
        df.to_csv(args.out, index=False)
        print(f'\nsaved {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
