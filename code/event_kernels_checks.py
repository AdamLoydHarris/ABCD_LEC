"""Two LEC data-path checks for `event_kernels`, against the v4 pickle (sbatch, ~25 GB resident).

1. Loader vs pickle: the core window (first A .. last trial) of every session that
   `event_kernels.load_recday` builds from per-session files equals `data_dic_lec_v4.pkl`:
   `Neuron_raw`, `XY_raw` (NaN equal, over the overlap), `Locs_raw` codes, `Trial_times`.
2. Shared primitives vs GLM V3 on the same session: 250 ms block counts, speed, signed
   acceleration and time since reward from `event_kernels.session_blocks` equal
   `glm_analysis_v3.prepare_session_data` -> `downsample_session_data(10, 'bin')` (speed and
   acceleration rescaled from per-bin to per-second and compared 2 blocks inside the core's edges,
   where the smoothing kernels see the same samples; tfr compared where both are defined).

    sbatch --job-name=ek_checks --mem=64G sbatch_files/event_kernels.sbatch code/event_kernels_checks.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import glm_analysis_v3 as g3                                           # noqa: E402


def _nan_equal(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and \
        np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def main() -> int:
    dd = g3.load_data_dic(version='v4', validate=True, apply_exclusions=True, verbose=True)
    rows = []
    recdays = sorted({f.split('full_sessions_')[1].split('.')[0]
                      for f in os.listdir(ek.LEC_FULL) if f.startswith('full_sessions_')})
    for rd in recdays:
        rec = ek.load_recday(rd, 'lec')
        for sess in rec['sessions']:
            s = sess['session']
            meta = dict(recday=rd, session=s, session_type=sess['session_type'], n_trials=sess['n_trials'])
            if s not in dd.get(rd, {}):
                rows.append(dict(**meta, status='not in pickle'))
                continue
            ref = dd[rd][s]
            k = sess['first_A_bin']
            nr = np.asarray(ref['Neuron_raw'])
            core = sess['counts'][:, k:k + nr.shape[1]]
            r = dict(**meta, status='compared', counts=np.array_equal(core, nr))
            xy_ref = ref.get('XY_raw')
            if xy_ref is not None:
                n = min(len(xy_ref), sess['xy'].shape[0] - k)
                r['xy'] = _nan_equal(sess['xy'][k:k + n], np.asarray(xy_ref, float)[:n])
            lr = ref.get('Locs_raw')
            if lr is not None:
                n = min(len(lr), len(sess['locs']) - k)
                r['locs'] = np.array_equal(sess['locs'][k:k + n].astype(int), np.asarray(lr)[:n].astype(int))
            tt = ref.get('Trial_times')
            if tt is not None and np.size(tt):
                r['trial_times'] = np.array_equal(sess['trial_times'] - k, np.asarray(tt).astype(np.int64))
            if sess['session_type'] == 'ABCD' and sess['n_trials'] >= 2 and xy_ref is not None:
                prep = g3.truncate_all_arrays(g3.prepare_session_data(ref))
                down = g3.downsample_session_data(prep, 10, mode='bin')
                b = ek.session_blocks({**sess}, events=('reward',))
                k_blk = k // ek.BLOCK
                n = down['FR'].shape[1]
                r['blk_counts'] = np.array_equal(b['counts'][:, k_blk:k_blk + n], down['FR'])
                # Smoothing (sigma 3 bins) reaches ~2 blocks past the core's edges, where our track
                # continues into the pre-A segment and V3's starts at the first A.
                e = slice(k_blk + 2, k_blk + n - 2)
                r['blk_speed'] = np.allclose(b['speed'][e], down['Speed'][2:n - 2] / ek.BIN_S,
                                             rtol=1e-9, atol=1e-9, equal_nan=True)
                r['blk_acc'] = np.allclose(b['acc'][e], down['Acc'][2:n - 2] / ek.BIN_S ** 2,
                                           rtol=1e-9, atol=1e-6, equal_nan=True)
                ours = b['dt_reward'][k_blk:k_blk + n]
                theirs = down['time_from_reward'] * ek.BIN_S
                both = np.isfinite(ours) & np.isfinite(theirs)
                r['blk_tfr_max_abs_diff_s'] = float(np.max(np.abs(ours[both] - theirs[both]))) if both.any() else np.nan
                r['blk_tfr_rows'] = int(both.sum())
            rows.append(r)
        print(rd, 'checked', flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(ek.REPO, 'data', 'event_kernels', 'checks')
    os.makedirs(out, exist_ok=True)
    k = 0
    while os.path.exists(p := os.path.join(out, 'loader_and_primitives' + (f'.v{k}' if k else '') + '.csv')):
        k += 1
    df.to_csv(p, index=False)
    bool_cols = [c for c in ('counts', 'xy', 'locs', 'trial_times', 'blk_counts', 'blk_speed', 'blk_acc') if c in df]
    for c in bool_cols:
        v = df[c].dropna()
        print(f'{c}: {int(v.sum())}/{len(v)} equal')
    if 'blk_tfr_max_abs_diff_s' in df:
        print(f"time since reward, max |ours - V3| over sessions: {df['blk_tfr_max_abs_diff_s'].max():.4f} s")
    print('->', p)
    bad = df[[c for c in bool_cols]].fillna(True).astype(bool).all(axis=1) == False   # noqa: E712
    # A session absent from the pickle has NaN in every column above, which reads as a pass.
    miss = df[df['status'] == 'not in pickle']
    print(f"{int((df['status'] == 'compared').sum())} sessions compared, {len(miss)} not in the pickle "
          f"({miss.groupby('session_type').size().to_dict()})")
    lost = miss[(miss['session_type'] == 'ABCD') & (miss['n_trials'] >= ek.MIN_TRIALS)]
    if len(lost):
        print('analysis sessions missing from the pickle:', list(zip(lost['recday'], lost['session'])))
    return 1 if bad.any() or len(lost) or not (df['status'] == 'compared').any() else 0


if __name__ == '__main__':
    sys.exit(main())
