"""How many 250 ms design rows are built from partially-tracked raw bins, under v4.

v1 interpolated every tracking gap with no length limit, so Speed was always finite and this
question did not exist. v4 leaves NaN (21 of 211 sessions, up to 9.4%), and two things then
happen in `run_glm_analysis`:

  * a row whose aggregated Speed is NaN is DROPPED from the fit (~:1719-1740);
  * but `_DOWNSAMPLE_AGG['Speed'] = 'nanmean'` returns a FINITE speed when even one of the ten
    raw bins in a block is tracked, while `FR` is 'sum' over all ten. So a block mostly inside
    a gap survives with a full spike count against a one-bin speed estimate.

The second is undocumented and new in v4. This script measures its exposure before anything is
changed: per recday, how many rows entering the fit are built from k tracked raw bins, and what
share of the spikes those rows carry.

    sbatch sbatch_files/glm_lec.sbatch --  (see v4_tracking_gap_check.sbatch)
    python code/v4_tracking_gap_check.py --data-version v4 --out <pkl>
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-version', default=None)
    ap.add_argument('--cap-s', type=float, default=30.0)
    ap.add_argument('--factor', type=int, default=10)
    ap.add_argument('--recdays', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'data', 'processed_data',
                                                  'v4_tracking_gap_check.pkl'))
    args = ap.parse_args()

    import glm_analysis_v3 as glm
    import recday_registry as rr
    version, path = rr.resolve_data_version(args.data_version)
    print(f'data: {os.path.basename(path)}  cap {args.cap_s} s  factor {args.factor}')
    data_dic = glm.load_data_dic(path=str(path), validate=True, apply_exclusions=True,
                                 verbose=True)
    recdays = args.recdays or sorted(data_dic)

    rows = []
    for rd in recdays:
        sessions, _ = glm.get_sessions_for_glm(data_dic[rd])
        for s in sessions:
            sd = data_dic[rd][s]
            prep = glm.prepare_session_data(sd, gp_n_bins=10, task=sd.get('Task'),
                                            filter_correct_paths=False,
                                            max_transition_seconds=args.cap_s, bin_size_ms=25)
            prep = glm.truncate_all_arrays(prep)
            raw_nan = ~np.isfinite(np.asarray(prep['Speed'], dtype=float))
            n_blocks = len(raw_nan) // args.factor
            if n_blocks == 0:
                continue
            # tracked raw bins per 250 ms block, in the same blocking `_aggregate_blocks` uses
            tracked = (~raw_nan[:n_blocks * args.factor]).reshape(n_blocks, args.factor).sum(1)

            d = glm.downsample_session_data(prep, args.factor, mode='bin')
            keep = (d['Locs'] <= 21) & d['valid_transition_mask'].astype(bool)
            keep &= np.isfinite(np.asarray(d['GP_binned'], dtype=float))
            n = min(len(keep), n_blocks)
            keep, tr = keep[:n], tracked[:n]
            # the fit's own rule: finite aggregated speed survives, NaN is dropped
            finite_speed = np.isfinite(np.asarray(d['Speed'], dtype=float)[:n])
            in_fit = keep & finite_speed
            spikes = np.asarray(d['FR'], dtype=float)[:, :n].sum(0)

            rows.append(dict(
                recday=rd, session=int(s), n_blocks=int(n), rows_in_fit=int(in_fit.sum()),
                rows_dropped_nan=int((keep & ~finite_speed).sum()),
                rows_full=int((in_fit & (tr == args.factor)).sum()),
                rows_diluted=int((in_fit & (tr < args.factor)).sum()),
                rows_majority_untracked=int((in_fit & (tr * 2 < args.factor)).sum()),
                rows_single_bin=int((in_fit & (tr == 1)).sum()),
                spikes_in_fit=float(spikes[in_fit].sum()),
                spikes_diluted=float(spikes[in_fit & (tr < args.factor)].sum()),
                spikes_majority_untracked=float(spikes[in_fit & (tr * 2 < args.factor)].sum()),
            ))
        print(f'  {rd}: {len([r for r in rows if r["recday"] == rd])} sessions', flush=True)

    df = pd.DataFrame(rows)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as fh:
        pickle.dump({'data_version': version, 'cap_s': args.cap_s, 'factor': args.factor,
                     'per_session': df}, fh)

    tot = df[['rows_in_fit', 'rows_dropped_nan', 'rows_diluted', 'rows_majority_untracked',
              'rows_single_bin', 'spikes_in_fit', 'spikes_diluted',
              'spikes_majority_untracked']].sum()
    print(f'\n== cohort, {version}, {len(df)} sessions')
    print(f'  rows entering the fit         {tot.rows_in_fit:>10,.0f}')
    print(f'  rows dropped (all-NaN speed)  {tot.rows_dropped_nan:>10,.0f}')
    print(f'  rows diluted (1-9 tracked)    {tot.rows_diluted:>10,.0f}'
          f'  = {100 * tot.rows_diluted / max(tot.rows_in_fit, 1):.3f}% of the fit')
    print(f'  rows majority-untracked (<5)  {tot.rows_majority_untracked:>10,.0f}'
          f'  = {100 * tot.rows_majority_untracked / max(tot.rows_in_fit, 1):.3f}%')
    print(f'  rows from a single tracked bin{tot.rows_single_bin:>10,.0f}')
    print(f'  spikes in majority-untracked  {100 * tot.spikes_majority_untracked / max(tot.spikes_in_fit, 1):.3f}% of fitted spikes')
    worst = df.assign(frac=df.rows_majority_untracked / df.rows_in_fit.clip(lower=1)) \
              .nlargest(8, 'frac')[['recday', 'session', 'rows_in_fit',
                                    'rows_majority_untracked', 'frac']]
    print('\n  worst sessions by majority-untracked share:')
    print(worst.to_string(index=False))
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
