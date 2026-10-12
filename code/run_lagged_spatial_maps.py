"""Items 9 + 13a: his held-out lagged-map statistic per neuron, gated, then the frame test.

    python run_lagged_spatial_maps.py --region lec|pfc [--n-null 200] [--gate-only]

Rebuilds El-Gaby's Figure5_Figure6 cell 31 from the saved maps and refuses to write anything
unless it reproduces his printed cell-34 n and t on all three panels (n exact, |dt| < 1e-9).
Then adds what he did not compute: per-mouse results, the LEC anatomy split with rate matching,
the best-lag distribution, and the place-vs-goal frame test. Writes to lsm.OUT_DIRS[region].
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as st

import anatomy_split as asp
import anchor_distribution as ad
import lagged_spatial_maps as lsm

TOL_T = 1e-9
PANELS = ('all', 'nonzero', 'strict')


def gate(region, per, order, state, got=None):
    got = lsm.cell34(per, order, state) if got is None else got
    want = lsm.CELL34[region]
    lines, ok = [f'== equivalence gate, {region.upper()}: rebuilt cell 31 vs his cell-34 print'], True
    for p in PANELS:
        dn, dt = got[p][0] - want[p][0], got[p][1] - want[p][1]
        good = dn == 0 and abs(dt) < TOL_T
        ok &= good
        lines.append(f'  {p:8s} his n {want[p][0]:5d} t {want[p][1]:.10f} | ours n {got[p][0]:5d} '
                     f't {got[p][1]:.10f} | dt {dt:+.1e} {"OK" if good else "MISMATCH"}')
    return ok, got, lines


def neuron_table(region, recs, per):
    rows = []
    for rd, rec in recs.items():
        h, S = per[rd], len(rec['non_repeat'])
        modal = [lsm._mode_or_nan(h['max_lags'][n]) for n in range(len(rec['state']))]
        rows.append(pd.DataFrame({
            'recday': rd, 'mouse': rec['mouse'], 'neuron': np.arange(len(rec['state'])),
            'state_tuned': rec['state'], 'r_all': h['r_all'], 'r_nonzero': h['r_nonzero'],
            'r_strict': h['r_strict'], 'best_lag_modal': modal, 'n_sessions': S,
            'rate_hz': rec['rate_hz'],
            'goal_overlap': lsm.goal_overlap(rec['tasks'][rec['non_repeat']])}))
    t = pd.concat(rows, ignore_index=True)
    t['lagged30'] = t.state_tuned & t.r_nonzero.notna()
    t['lagged90'] = t.state_tuned & t.r_strict.notna()
    return t


def add_anatomy(t):
    regions = asp.load_unit_regions()
    j = asp.join_regions({rd: g.sort_values('neuron').neuron.to_numpy()
                          for rd, g in t.groupby('recday')}, regions, value_name='neuron_chk')
    assert (j.neuron == j.neuron_chk).all()
    return t.merge(j[['recday', 'neuron', 'group', 'acronym', 'y_um']], on=['recday', 'neuron'],
                   how='left')


def _per_mouse(t, col, mask=None):
    """Recday mean -> mouse mean of `col` over rows in `mask` (mice are the replicates)."""
    s = t if mask is None else t[mask]
    rd = s.groupby(['mouse', 'recday'])[col].mean()
    return rd.groupby('mouse').mean()


def summarise(t, region, got):
    L = [f'== {region.upper()}: {len(t)} units, {int(t.state_tuned.sum())} state-tuned, '
         f'{t.recday.nunique()} recdays, {t.mouse.nunique()} mice']
    tuned = t.state_tuned
    L.append('-- Fig. 5f held-out lagged-map r, state-tuned, per mouse (recday mean -> mouse mean)')
    L.append(f'  {"mouse":6s} {"n":>5s} {"r_all":>7s} {"r_30":>7s} {"n_30":>5s} {"r_90":>7s} '
             f'{"n_90":>5s} {"f_30":>6s} {"f_90":>6s}')
    pm = pd.DataFrame({
        'n': t[tuned].groupby('mouse').size(),
        'r_all': _per_mouse(t, 'r_all', tuned), 'r_30': _per_mouse(t, 'r_nonzero', tuned),
        'n_30': t[t.lagged30].groupby('mouse').size(), 'r_90': _per_mouse(t, 'r_strict', tuned),
        'n_90': t[t.lagged90].groupby('mouse').size(),
        'f_30': _per_mouse(t.assign(x=t.lagged30.astype(float)), 'x', tuned),
        'f_90': _per_mouse(t.assign(x=t.lagged90.astype(float)), 'x', tuned)}).fillna(0)
    for m, r in pm.iterrows():
        L.append(f'  {m:6s} {int(r.n):5d} {r.r_all:7.3f} {r.r_30:7.3f} {int(r.n_30):5d} '
                 f'{r.r_90:7.3f} {int(r.n_90):5d} {r.f_30:6.3f} {r.f_90:6.3f}')
    for col, name in (('r_all', 'all'), ('r_30', '30 deg'), ('r_90', '90 deg')):
        v = pm[col][pm['n' if col == 'r_all' else f'n_{col[2:]}'] > 0]
        k = int((v > 0).sum())
        p = st.binomtest(k, len(v), 0.5, alternative='greater').pvalue if len(v) else np.nan
        L.append(f'  mice with mean r > 0, {name}: {k}/{len(v)}, one-sided binomial p {p:.4f}')
    L.append(f'  pooled over neurons (his unit): ' + ', '.join(
        f'{p} n {got[p][0]} t {got[p][1]:.2f}' for p in PANELS))

    L.append('-- modal best lag of state-tuned neurons (count per lag, 30 deg steps)')
    c = t[tuned].best_lag_modal.value_counts().reindex(range(12), fill_value=0)
    L.append('  ' + ' '.join(f'{k}:{int(v)}' for k, v in c.items()))

    f = t[tuned & t.r_place.notna()]
    L.append(f'-- frame test, state-tuned with a score ({len(f)}): held-out 4-vector r, '
             f'p vs node-scramble null')
    for name, m in (('all', np.ones(len(f), bool)), ('lagged30 (his)', f.lagged30),
                    ('not lagged30', ~f.lagged30), ('lagged90 (his)', f.lagged90),
                    ('V5 selected_eg', f.selected_eg.fillna(False).astype(bool)),
                    ('V5 not selected_eg', ~f.selected_eg.fillna(False).astype(bool))):
        s = f[np.asarray(m)]
        if not len(s):
            continue
        pl, go = s.p_place < 0.05, s.p_goal < 0.05
        L.append(f'  {name:20s} n {len(s):5d}  r_place {s.r_place.mean():+.3f}  '
                 f'r_goal {s.r_goal.mean():+.3f}  place-only {np.mean(pl & ~go):.3f}  '
                 f'goal-only {np.mean(go & ~pl):.3f}  both {np.mean(pl & go):.3f}  '
                 f'neither {np.mean(~pl & ~go):.3f}')
    fd = f.assign(delta=f.r_goal - f.r_place)
    L.append('  per mouse, r_goal - r_place (recday mean -> mouse mean):')
    for grp, col in (('lagged30', 'lagged30'), ('V5 selected_eg', 'selected_eg')):
        lag = fd[col].fillna(False).astype(bool)
        a, b = _per_mouse(fd, 'delta', lag), _per_mouse(fd, 'delta', ~lag)
        L.append(f'    {grp:15s} lagged: ' + ' '.join(f'{m} {v:+.3f}' for m, v in a.items()))
        L.append(f'    {"":15s} other:  ' + ' '.join(f'{m} {v:+.3f}' for m, v in b.items()))
        sub = fd.assign(group=np.where(lag, 'lagged', 'other'))
        # delta alone can move through either frame, so each frame is contrasted on its own
        for col in ('r_goal', 'r_place', 'delta'):
            r = asp.within_recday_permutation(sub, col, 'lagged', 'other', n_perm=2000)
            L.append(f'    {"":15s} {col:7s} lagged - other {r["observed"]:+.4f}, '
                     f'within-recday label shuffle p {r["p_perm"]:.4f}')
        gm = _per_mouse(fd.assign(x=(fd.p_goal < 0.05).astype(float)), 'x', lag)
        L.append(f'    {"":15s} lagged, fraction p_goal < 0.05 per mouse: '
                 + ' '.join(f'{m} {v:.3f}' for m, v in gm.items()))

    if 'group' in t.columns:
        L.append('-- LEC anatomy (state-tuned); per-mouse effects, mice are the replicates')
        tt = t[tuned].assign(f30=t.lagged30.astype(float),
                             delta=t.r_goal - t.r_place)
        for col in ('r_all', 'f30', 'delta'):
            rep = asp.per_region_report(tt, col)
            L.append(f'  {col}:')
            for _, r in rep.iterrows():
                L.append(f'    {r.group:9s} mice {r.n_mice} units {r.n_units:4d} '
                         f'effect {r.effect_pooled_over_mice:+.3f} '
                         f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] {r.per_mouse}')
            a, b = asp.PRIMARY_CONTRAST
            perm = asp.within_recday_permutation(tt, col, a, b, n_perm=2000)
            L.append(f'    {a} - {b}: {perm["observed"]:+.4f}, p_perm {perm["p_perm"]:.4f}')
            rm = asp.rate_match(tt, 'rate_hz', a, b)
            permr = asp.within_recday_permutation(rm, col, a, b, n_perm=2000)
            L.append(f'    rate-matched ({len(rm)} units): {permr["observed"]:+.4f}, '
                     f'p_perm {permr["p_perm"]:.4f}')
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', choices=['lec', 'pfc'], required=True)
    ap.add_argument('--n-null', type=int, default=200)
    ap.add_argument('--gate-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--summary-only', action='store_true',
                    help='rewrite the summary from the saved pickle; no recompute')
    args = ap.parse_args()
    t0 = time.time()
    od = lsm.OUT_DIRS[args.region]
    if args.summary_only:
        with open(os.path.join(od, 'lagged_spatial_maps.pkl'), 'rb') as fh:
            d = pickle.load(fh)
        _, _, lines = gate(args.region, None, None, None, got=d['gate'])
        summary = lines + summarise(d['table'], args.region, d['gate'])
        with open(os.path.join(od, 'lagged_spatial_maps_summary.txt'), 'w') as fh:
            fh.write('\n'.join(summary) + '\n')
        print('\n'.join(summary))
        return
    order = lsm.recdays(args.region)
    recs, per = {}, {}
    for rd in order:
        recs[rd] = lsm.load_recday(args.region, rd, rates=not args.gate_only)
        per[rd] = lsm.recday_heldout(recs[rd])
    ok, got, lines = gate(args.region, per, order, {rd: recs[rd]['state'] for rd in order})
    print('\n'.join(lines), flush=True)
    if not ok:
        print('EXIT=1 (gate failed; nothing written)')
        sys.exit(1)
    if args.gate_only:
        print(f'EXIT=0 (gate only, {time.time() - t0:.0f} s)')
        return

    t = neuron_table(args.region, recs, per)
    frames = pd.concat([lsm.recday_frame_test(recs[rd], args.n_null, args.seed,
                                              neurons=np.flatnonzero(recs[rd]['state']))
                        for rd in order], ignore_index=True)
    t = t.merge(frames, on=['recday', 'neuron'], how='left')
    ut = ad.v5_unit_table(args.region)
    t = t.merge(ut[['recday', 'neuron', 'state_tuned', 'selected_eg', 'selected_strict_eg',
                    'peak_lag']].rename(columns={'state_tuned': 'v5_state_tuned',
                                                 'peak_lag': 'v5_peak_lag'}),
                on=['recday', 'neuron'], how='left')
    if args.region == 'lec':
        t = add_anatomy(t)
    summary = lines + summarise(t, args.region, got)

    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, 'lagged_spatial_maps.pkl'), 'wb') as fh:
        pickle.dump({'table': t, 'gate': got, 'cell34': lsm.CELL34[args.region],
                     'max_lags': {rd: per[rd]['max_lags'] for rd in order},
                     'non_repeat': {rd: recs[rd]['non_repeat'] for rd in order},
                     'config': vars(args)}, fh)
    t.to_csv(os.path.join(od, 'lagged_spatial_maps_units.csv'), index=False)
    with open(os.path.join(od, 'lagged_spatial_maps_summary.txt'), 'w') as fh:
        fh.write('\n'.join(summary) + '\n')
    print('\n'.join(summary[len(lines):]))
    print(f'-> {od}\nEXIT=0 ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
