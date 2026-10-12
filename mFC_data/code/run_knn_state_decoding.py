"""kNN ABCD state decoding (knn_state_decoding.py) for one dataset, under sbatch.

    python run_knn_state_decoding.py --dataset lec|pfc [--recdays rd1,rd2] [--n-pcs 15]
        [--seed 0] [--out PATH]

Same file in code/ and mFC_data/code/. Loads the data_dic as run_lda_reward_progress does (LEC
pickle; PFC rebuilt with compute_norm=True). Per recday: within-task and cross-task kNN under the
four feature variants, with their nulls; the null-mean guard raises on a broken null.
Output: a pickle {'rows', 'lags', 'x3', 'diag', 'meta', 'nulls', 'recday_means', 'contrasts',
'primaries', 'skipped', 'config', 'elapsed_s'} and CSVs of the recday means and primaries.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import knn_state_decoding as kn                             # noqa: E402
from run_lda_reward_progress import load_lec, load_pfc      # noqa: E402

REPO = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
OUT_DEFAULT = {
    'lec': os.path.join(REPO, 'data', 'glm_outputs', 'LEC_knn_state', 'knn_state.pkl'),
    'pfc': os.path.join(REPO, 'mFC_data', 'glm_outputs', 'PFC_knn_state', 'knn_state.pkl'),
}


def _bools(x3):
    for c in ('correct', 'dest_match', 'leg_match'):
        if c in x3:
            x3[c] = x3[c].astype(bool)
    return x3


def per_recday_p(results, variant='raw'):
    """X0 accuracy of each recday against its own role null (descriptive; mice are the unit)."""
    out = {}
    for rd, r in results.items():
        x = r['rows']
        obs = x[(x.analysis == 'X0') & (x.variant == variant)].groupby('task_idx').acc.mean()
        null = r['null_cross'][variant]
        if not len(obs) or not len(null):
            continue
        acc = float(obs.mean())
        out[rd] = {'acc': acc, 'null_mean': float(null.mean()),
                   'p': float((1 + np.sum(null >= acc)) / (1 + len(null)))}
    return pd.DataFrame(out).T


def summary_lines(p):
    rdm, con, prim = p['recday_means'], p['contrasts'], p['primaries']
    L = [f'== {p["dataset"].upper()}: {rdm.recday.nunique()} recdays, {rdm.mouse.nunique()} mice; '
         f'skipped {len(p["skipped"])}']
    L.append('-- primaries (mouse means, one-sample t, Holm over four)')
    for _, r in prim.iterrows():
        if 'mean' not in r or not np.isfinite(r.get('mean', np.nan)):
            L.append(f'  {r.contrast:14s} {r.variant:8s} n_mice {r.n_mice}')
            continue
        L.append(f'  {r.contrast:14s} {r.variant:8s} mean {r["mean"]:+.4f} '
                 f'[{r.ci[0]:+.4f}, {r.ci[1]:+.4f}] t {r.t:+.2f} p {r.p:.4f} '
                 f'p_holm {r.p_holm:.4f} +{r.n_pos}/{r.n_mice}  {r.per_mouse}')
    L.append('-- accuracy, mouse means (recday -> mouse; chance 0.25)')
    pm = rdm.groupby(['analysis', 'pool', 'variant', 'mouse']).acc.mean().unstack('mouse')
    for idx, row in pm.iterrows():
        L.append(f'  {"/".join(idx):26s} mean {row.mean():.3f}  '
                 + ' '.join(f'{m} {v:.3f}' for m, v in row.items()))
    x3 = _bools(p['x3'])
    if len(x3):
        L.append('-- X3 place split, X0 raw, pooled samples: correct when the leg / destination '
                 'is shared with a training task vs not')
        for col in ('leg_match', 'dest_match'):
            L.append(f'  {col:10s} shared {x3[x3[col]].correct.mean():.3f} '
                     f'(n {int(x3[col].sum())}) vs not {x3[~x3[col]].correct.mean():.3f}')
    d = p['diag']
    if len(d):
        L.append('-- neighbour time-locality L and source-task entropy, cross-task, mean over '
                 'held-out tasks')
        for v, g in d.groupby('variant'):
            L.append(f'  {v:8s} L {g.L.mean():+.3f}  entropy {g.source_entropy.mean():.3f}')
    w = p['rows'][p['rows'].analysis == 'W0'].groupby('variant').L.mean()
    L.append('-- within-task L: ' + ' '.join(f'{v} {x:+.3f}' for v, x in w.items()))
    L.append('-- per-recday X0 raw vs role null (descriptive): '
             f'{int((p["recday_p"].p < 0.05).sum())}/{len(p["recday_p"])} at p < 0.05')
    r2 = np.array([v for v in p['pc_time_r2'].values()])
    if len(r2):
        L.append('-- R^2 of PCs 1-5 on trial time, mean over recdays: '
                 + ' '.join(f'{x:.3f}' for x in np.nanmean(r2, axis=0)))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['lec', 'pfc'], required=True)
    ap.add_argument('--recdays', default=None, help='comma-separated subset (smoke test)')
    ap.add_argument('--n-pcs', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    t0 = time.time()
    only = args.recdays.split(',') if args.recdays else None
    data_dic, recdays = (load_lec if args.dataset == 'lec' else load_pfc)(only)
    print(f'loaded {len(recdays)} recdays in {time.time() - t0:.0f} s', flush=True)
    cfg = kn.KNNConfig(n_pcs=args.n_pcs, seed=args.seed)
    results, skipped = {}, {}
    for rd in recdays:
        t1 = time.time()
        r = kn.run_recday(data_dic, rd, cfg)
        if r is None:
            skipped[rd] = 'unusable: < 2 tasks after the trial filters'
            print(f'  {rd}: skipped', flush=True)
            continue
        for v in kn.VARIANTS:
            kn.check_null(r['null_cross'][v], r['n_cross_tested'])
        results[rd] = r
        print(f'  {rd}: {len(r["meta"])} tasks, {r["n_neurons"]} neurons, '
              f'{time.time() - t1:.0f} s', flush=True)
    rows = pd.concat([r['rows'] for r in results.values()], ignore_index=True)
    rdm = kn.recday_means(rows)
    con = kn.contrasts(rdm)
    payload = {
        'dataset': args.dataset, 'config': asdict(cfg), 'rows': rows,
        'lags': pd.concat([r['lags'] for r in results.values()], ignore_index=True),
        # recdays with < 4 tasks have no cross-task rows; an empty frame in the concat turns the
        # flags to object dtype, where ~ gives -1 / -2 (this crashed LEC job 3667796)
        'x3': _bools(pd.concat([r['x3'] for r in results.values() if len(r['x3'])],
                               ignore_index=True)),
        'diag': pd.concat([r['diag'] for r in results.values() if len(r['diag'])],
                          ignore_index=True),
        'meta': pd.concat([r['meta'] for r in results.values()], ignore_index=True),
        'nulls': {rd: {'within': r['null_within'], 'cross': r['null_cross']}
                  for rd, r in results.items()},
        'pc_time_r2': {rd: r['pc_time_r2'] for rd, r in results.items()},
        'recday_means': rdm, 'contrasts': con, 'primaries': kn.primary_table(con),
        'recday_p': per_recday_p(results), 'skipped': skipped,
    }
    payload['elapsed_s'] = time.time() - t0
    out = args.out or OUT_DEFAULT[args.dataset]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as fh:
        pickle.dump(payload, fh)
    rdm.to_csv(out.replace('.pkl', '_recday_means.csv'), index=False)
    payload['primaries'].to_csv(out.replace('.pkl', '_primaries.csv'), index=False)
    lines = summary_lines(payload)
    with open(out.replace('.pkl', '_summary.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {out} ({payload["elapsed_s"] / 60:.1f} min)\nEXIT=0')


if __name__ == '__main__':
    main()
