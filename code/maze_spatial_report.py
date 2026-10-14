"""Tables from merged maze runs: the PFC thesis-recipe reproduction, and per-mouse summaries of the
decided recipe against its nulls for mEC, LEC and PFC. Reads merged tables and shards only.

    python code/maze_spatial_report.py --tag 1617859bc6fd --equivalent-to 21c552de331e
    python code/maze_spatial_report.py --tag <B2-head tag> --datasets mec lec      # runs without PFC

The field-shuffle null (`place_shuffled`) is the bar for generalisation claims (user's choice,
2026-10-12); the fitted and random-centre nulls are reported beside it.

Thesis targets (Harris 2025, chapter 7; one-sample two-sided t-tests across 25 recording double-days):
RSA Euclidean t 5.45 and node-degree t 3.51 (against 0); corner/cardinal balanced accuracy t 3.36 and
node/bridge t 4.63 (against 0.5); same-type share of location-decoder errors t 4.32 (against 3/7);
plane angles 59-69 deg across goal-progress bins against 8-10 deg within (Figure 7.2 d).
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats

ROOTS = {'mec': '/ceph/behrens/adam_harris/Taskspace_abstraction_mEC/data/spatial/maze',
         'lec': '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/data/spatial/maze',
         'pfc': '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data/data/spatial/maze'}
TARGETS = [('thesis_rsa_E', 0.0, 5.45, 'RSA Euclidean beta'), ('thesis_rsa_D', 0.0, 3.51, 'RSA node-degree beta'),
           ('dec_corner_cardinal_ba', 0.5, 3.36, 'corner/cardinal BA'), ('dec_node_bridge_ba', 0.5, 4.63, 'node/bridge BA'),
           ('dec_same_type_error_share', 3 / 7, 4.32, 'same-type error share')]


def latest(pattern: str) -> str:
    """Highest version of a versioned output (name.csv, name.v1.csv, ...)."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(pattern)
    def ver(f):
        parts = os.path.basename(f).split('.')
        return int(parts[-2][1:]) if len(parts) > 2 and parts[-2].startswith('v') and parts[-2][1:].isdigit() else 0
    return max(files, key=ver)


def load(dataset: str, tag: str) -> dict:
    """Merged tables and the stamp of one run."""
    m = pickle.load(open(latest(os.path.join(ROOTS[dataset], tag, 'merged', 'maze_merged*.pkl')), 'rb'))
    return dict(stamp=m['stamp'], recdays=m['recdays'], **m['tables'])


def ttest(v, mu: float) -> tuple:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan, len(v)
    t, p = stats.ttest_1samp(v, mu)
    return float(t), float(p), len(v)


def thesis_reproduction(pop: pd.DataFrame) -> pd.DataFrame:
    """The five t-test targets across recdays and across mice, PFC thesis recipe."""
    th = pop[pop.recipe == 'thesis']
    rows = []
    for col, mu, target, label in TARGETS:
        t_r, p_r, n_r = ttest(th[col], mu)
        per_mouse = th.groupby('mouse')[col].mean()
        t_m, p_m, n_m = ttest(per_mouse, mu)
        rows.append(dict(statistic=label, chance=round(mu, 3), thesis_t=target, mean=th[col].mean(),
                         t_recdays=t_r, p_recdays=p_r, n_recdays=n_r, t_mice=t_m, p_mice=p_m, n_mice=n_m))
    return pd.DataFrame(rows)


def thesis_planes(dataset: str, tag: str) -> pd.DataFrame:
    """Thesis-recipe plane angles by goal-progress bin distance (0 = within-bin halves), mean and SEM
    across recdays, from the shards."""
    by = {}
    for f in sorted(glob.glob(os.path.join(ROOTS[dataset], tag, 'shards', '*.pkl'))):
        pl = pickle.load(open(f, 'rb'))['real']['thesis']['planes']
        vals = {0: list(pl['within'].values())}
        for (a, b), v in pl['across'].items():
            vals.setdefault(b - a, []).append(v)
        for k, v in vals.items():
            by.setdefault(k, []).append(np.nanmean(v))
    return pd.DataFrame([dict(bin_distance=k, mean_deg=np.mean(v), sem_deg=stats.sem(v), n_recdays=len(v))
                         for k, v in sorted(by.items())])


NULLS = ('place_shuffled', 'place', 'place_random')
#: every null kind in report order; a run reports the kinds it has (turn-matched runs carry three)
NULL_ORDER = ('place_shuffled', 'kinematic', 'speed', 'place', 'place_random', 'boundary')
CLAIMS = [('rsa9_ED_X', 'RSA node-degree beta, E+D, 9 nodes', None), ('dec_node_bridge_ba', 'node/bridge BA', 0.5),
          ('dec_corner_cardinal_ba', 'corner/cardinal BA', 0.5), ('dec_same_type_error_share', 'same-type error share', 3 / 7),
          ('rsa21_contrast', 'E/G diagnostic contrast, 21 locations', None)]
#: claims tested at mouse level, (stat, label, chance, direction); +1 = generalisation is larger
MOUSE_CLAIMS = [(c, l, ch, +1) for c, l, ch in CLAIMS] + [('rsa21_ED_X', 'RSA node/bridge beta, E+NB, 21 locations', None, +1)]
LEGACY_HOSTS = ('gpu-350-', 'gpu-380-')
UNINFORMATIVE = {'mec': ('bp02',)}          # 17 units over 5 recdays, 0-4 per recday


def equivalence(dataset: str, old: str, new: str) -> dict:
    """Max abs difference between two runs over every number the old run has: units, population, and the
    nulls of the old run's kinds. 0 means the new run only added numbers."""
    a, b = load(dataset, old), load(dataset, new)
    out = {}
    for name in ('units', 'population'):
        x, y = a[name], b[name]
        num = x.select_dtypes('number').columns
        out[name] = float(np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)))) if len(x) == len(y) else np.inf
        out[name + '_nan_pattern_equal'] = bool((x[num].isna().to_numpy() == y[num].isna().to_numpy()).all()) if len(x) == len(y) else False
        other = [c for c in x.columns if c not in num]
        out[name + '_other_columns_equal'] = bool(len(x) == len(y) and list(x.columns) == list(y.columns)
                                                  and (x[other].astype(str).to_numpy() == y[other].astype(str).to_numpy()).all())
    key = ['recday', 'null', 'stat']
    m = a['nulls'].merge(b['nulls'], on=key, suffixes=('_a', '_b'), how='left')
    cols = ['n_sims', 'real', 'null_mean', 'null_p95', 'null_p05', 'frac_null_ge_real']
    out['nulls'] = float(np.nanmax([np.nanmax(np.abs(m[c + '_a'] - m[c + '_b'])) for c in cols]))
    out['nulls_rows_missing'] = int(m['n_sims_b'].isna().sum())
    out['new_null_rows'] = len(b['nulls']) - len(a['nulls'])
    return out


def run_nulls(run: dict) -> tuple:
    """The run's null kinds in NULL_ORDER."""
    have = set(run['nulls']['null'])
    return tuple(k for k in NULL_ORDER if k in have)


def mouse_claims(dataset: str, run: dict, nulls=None) -> pd.DataFrame:
    """Per claim and null: mouse-level tests (maze_spatial.mouse_null_tests) and the group statement across
    mice (maze_spatial.group_tests). Needs the run's null_sims table (runs after 2026-10-12)."""
    import maze_spatial as ms
    if 'null_sims' not in run:
        raise KeyError('this run has no null_sims table; use legacy_claim_counts')
    rows = []
    for stat, label, chance, direction in MOUSE_CLAIMS:
        for k in (nulls or run_nulls(run)):
            m = ms.mouse_null_tests(run['population'], run['null_sims'], stat, k, direction, chance,
                                    exclude_mice=UNINFORMATIVE.get(dataset, ()))
            g = ms.group_tests(m)
            rows.append(dict(dataset=dataset, statistic=label, null=k, n_mice=g['n_mice'], hits=g['hits'],
                             p_binomial=g['p_binomial'], p_signflip=g['p_signflip'],
                             mice=','.join(m.loc[m['hit'], 'mouse']), min_n_sims=int(m['n_sims'].min()) if len(m) else 0,
                             range=f"{m['real'].min():.2f} to {m['real'].max():.2f}" if len(m) else ''))
    return pd.DataFrame(rows)


def shard_equivalence(dataset: str, new: str, refs: list, keys=('units', 'single', 'maps2d', 'place_models', 'real', 'nulls')) -> pd.DataFrame:
    """Per recday: every number of the reference shard against the new shard (old simulations are a prefix of
    the new lists). The reference is the first run in `refs` whose shard ran on a modern node: legacy nodes
    differ at 1e-15 (run 1617859bc6fd)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOTS[dataset], new, 'shards', '*.pkl'))):
        b = pickle.load(open(f, 'rb'))
        rd = b['recday']
        ref, a = None, None
        for t in refs:
            fa = os.path.join(ROOTS[dataset], t, 'shards', f'{rd}.pkl')
            if os.path.exists(fa):
                cand = pickle.load(open(fa, 'rb'))
                if not str(cand['stamp'].get('host', '')).startswith(LEGACY_HOSTS):
                    ref, a = t, cand
                    break
        row = dict(recday=rd, new_host=b['stamp'].get('host', ''), reference=ref or 'none on a modern node')
        if a is not None:
            import spatial_common as sc
            out = sc.compare_tree({k: a[k] for k in keys if k in a}, {k: b[k] for k in keys if k in b})
            row.update(max_abs_diff=out['max_diff'], worst=out['worst'], nan_mismatch=out['nan_mismatch'],
                       n_missing=len(out['missing']), first_missing=out['missing'][0] if out['missing'] else '', n_values=out['n'])
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_claim_counts(dataset: str, run: dict) -> pd.DataFrame:
    """Superseded by mouse_claims (2026-10-12): per claim and null, mice beyond the null p95 by the mean over
    recdays of each recday's fraction of simulations >= real, on the generalisation side of chance."""
    rows = []
    for col, label, chance in CLAIMS:
        row = dict(dataset=dataset, statistic=label)
        for k in (k_ for k_ in NULLS if k_ in run_nulls(run)):
            bm = decided_by_mouse(run['population'], run['nulls'], [col], k).set_index('mouse')
            bm = bm.drop(index=[m for m in UNINFORMATIVE.get(dataset, ()) if m in bm.index])
            v = bm[col]
            side = v > (chance if chance is not None else 0)
            hit = (bm[col + '_frac_null_ge'] <= 0.05) & side
            row['n_mice'] = len(bm)
            row['range'] = f'{v.min():.2f} to {v.max():.2f}'
            row[k] = int(hit.sum())
            row[k + '_mice'] = ','.join(bm.index[hit])
        rows.append(row)
    return pd.DataFrame(rows)


def decided_by_mouse(pop: pd.DataFrame, nulls: pd.DataFrame, stats_: list, null_kind: str = 'place') -> pd.DataFrame:
    """Per mouse: mean real value and mean fraction of the null at or above it, decided recipe."""
    real = pop[pop.recipe == 'decided']
    nl = nulls[nulls.null == null_kind]
    rows = []
    for mouse, g in real.groupby('mouse'):
        row = dict(mouse=mouse, n_recdays=len(g))
        for s in stats_:
            row[s] = g[s].mean() if s in g else np.nan
            ns = nl[(nl.mouse == mouse) & (nl.stat == s)]
            row[s + '_frac_null_ge'] = ns.frac_null_ge_real.mean() if len(ns) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--equivalent-to', default='', help='older run whose every number must be unchanged')
    ap.add_argument('--datasets', nargs='+', default=list(ROOTS), choices=list(ROOTS))
    args = ap.parse_args(argv)
    pd.set_option('display.width', 250)
    runs = {d: load(d, args.tag) for d in args.datasets}
    cfgs = {d: r['stamp'].get('cfg_hash') for d, r in runs.items()}
    print('config hash per dataset:', cfgs, '| identical:', len(set(cfgs.values())) == 1)
    if 'pfc' in runs and (runs['pfc']['population'].recipe == 'thesis').any():
        print('\n== PFC thesis recipe against the thesis targets')
        print(thesis_reproduction(runs['pfc']['population']).round(4).to_string(index=False))
        try:
            pl = thesis_planes('pfc', args.tag)
            print('\n== PFC thesis-recipe plane angles by goal-progress bin distance (0 = within-bin halves)')
            print(pl.round(2).to_string(index=False))
        except (KeyError, TypeError):
            print('\n(no thesis planes in this run)')
    key = ['rsa9_ED_E', 'rsa9_ED_X', 'rsa9_GD_G', 'rsa21_cvr2_ED', 'rsa21_cvr2_GD', 'rsa21_contrast',
           'dec_corner_cardinal_ba', 'dec_node_bridge_ba', 'dec_location_acc', 'dec_same_type_error_share',
           'planes_k3_halves_min', 'planes_k3_tasks_min', 'planes_k3_across_min']
    if args.equivalent_to:
        print(f'\n== equivalence with run {args.equivalent_to} (max abs diff over every number of the old run)')
        for d in runs:
            print(d, equivalence(d, args.equivalent_to, args.tag))
            eq = shard_equivalence(d, args.tag, [args.equivalent_to])
            print(f"{d} shards: {len(eq)} recdays, max abs diff {eq['max_abs_diff'].max()}, NaN mismatches "
                  f"{int(eq['nan_mismatch'].sum())}, missing paths {int(eq['n_missing'].sum())}, "
                  f"values {eq['n_values'].sum() / 1e6:.1f} M, references {sorted(set(eq['reference']))}")
    if all('null_sims' in r for r in runs.values()):
        print('\n== mouse-level claims: hits = mice with p <= 0.05 against the mean-over-recdays null, on the generalisation side')
        print(pd.concat([mouse_claims(d, r) for d, r in runs.items()]).to_string(index=False))
    print('\n== legacy counts: mean over recdays of the per-recday null fraction (superseded by the mouse-level claims)')
    print(pd.concat([legacy_claim_counts(d, r) for d, r in runs.items()]).to_string(index=False))
    for d, r in runs.items():
        for kind in run_nulls(r):
            print(f'\n== {d} decided recipe by mouse, against the {kind} null (frac_null_ge: fraction of simulations >= real)')
            print(decided_by_mouse(r['population'], r['nulls'], key, kind).round(3).T.to_string())


if __name__ == '__main__':
    main()
