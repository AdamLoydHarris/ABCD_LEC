"""PFC (El-Gaby deposit) session index for the event-kernel analysis: where each
trial-anchored `Neuron_raw` sits inside its whole-recording `Awake_Sleep` shard, every
recording's kind and clock time, a per-session pixel -> cm affine, and the behaviour probes.

The deposit is never modified. Outputs, in `mFC_data/data/processed/` (new names only):
    pfc_session_index[.vN].csv     one row per task session: shard, offset, bins, probes
    pfc_recording_index[.vN].csv   one row per shard: kind, day, clock time, duration
    pfc_xy_to_cm[.vN].csv          one row per task session: affine px -> LEC maze frame (cm)

Gate (exit 1): every wake array is a unique exact slice of one shard. XY conversion: one affine
per recday from node centroids pooled over its sessions; a session whose recday fit misses by
> 2.5 cm, or whose own fitted scale (>= 6 nodes) is > 5% off its recday's, gets `xy_ok` False
and is loaded as untracked. El-Gaby's measured px/cm is reported, not gated:
it fits the 2021 cohort (corner-node span 726-747 px) but not ab03 / ah07 (564-594 px, a
different camera), and one me08 recday's XY spans 59 px.

    sbatch sbatch_files/mec_stage.sbatch ../../mFC_data/code/pfc_session_index.py [--write]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code'))
sys.path.insert(0, os.path.join(REPO, 'code', 'preprocessing'))
import validate_alignment as va                                       # noqa: E402

DATA = os.path.join(REPO, 'mFC_data', 'data')
AWAKE = os.path.join(DATA, 'Neuronal_activity', 'Awake')
SHARDS = os.path.join(DATA, 'Neuronal_activity', 'Awake_Sleep')
BOOK = os.path.join(REPO, 'mFC_data', 'replication_run', 'data', 'Intermediate_objects_f7')
OUT = os.path.join(DATA, 'processed')
LEC_V3 = os.path.join(REPO, 'data', 'processed_v3')
BIN_MS = 25
# El-Gaby's own measurements (Intermediate_objects/Maze_measurements, Figure2 `Mice_maze_dic`):
# both mazes 47 cm across; maze 1 948.8 px, maze 2 964.3 px.
MAZE_OF = {'me08': 1, 'ah03': 2, 'me10': 1, 'me11': 2, 'ah04': 1, 'ab03': 1, 'ah07': 2}
PX_PER_CM = {1: 948.8333 / 47.0, 2: 964.3333 / 47.0}
MAX_NODE_RESID_CM = 2.5         # recday fit; ab03 reaches 2.1 cm with nodes 11 cm wide
MAX_SESSION_SCALE_DEV = 0.05    # one session's own fitted scale against its recday's


def _versioned(path: str) -> str:
    stem, ext = os.path.splitext(path)
    k = 0
    while os.path.exists(p := stem + (f'.v{k}' if k else '') + ext):
        k += 1
    return p


def recdays() -> list[str]:
    return [str(r) for r in np.load(os.path.join(DATA, 'MetaData', 'combined_ABCDonly_days.npy'))]


def _dates(rd: str) -> list[datetime]:
    return [datetime.strptime(d, '%d%m%Y') for d in rd.split('_')[1:]]


def shard_files(rd: str) -> dict[int, str]:
    pat = re.compile(rf'^binned_FR_dic_{re.escape(rd)}_(\d+)$')
    return {int(m.group(1)): os.path.join(SHARDS, f) for f in os.listdir(SHARDS)
            if (m := pat.match(f))}


def wake_files(rd: str) -> dict[int, str]:
    pat = re.compile(rf'^Neuron_raw_{re.escape(rd)}_(\d+)\.npy$')
    return {int(m.group(1)): os.path.join(AWAKE, f) for f in os.listdir(AWAKE) if (m := pat.match(f))}


def find_slice(pop: np.ndarray, shard_pop: np.ndarray) -> list[int]:
    """Offsets j where shard_pop[j:j+n] equals pop exactly (sum of squared differences 0)."""
    n = len(pop)
    if len(shard_pop) < n:
        return []
    c2 = np.concatenate([[0.0], np.cumsum(shard_pop.astype(float) ** 2)])
    win = c2[n:] - c2[:-n]
    cross = fftconvolve(shard_pop.astype(float), pop[::-1].astype(float), mode='valid')
    ssd = win - 2 * cross + float(np.sum(pop.astype(float) ** 2))
    cand = np.flatnonzero(ssd < 0.5)
    return [int(j) for j in cand if np.array_equal(shard_pop[j:j + n], pop)]


def lec_node_centroids() -> np.ndarray:
    """(9, 2) median over LEC v3 ABCD sessions of each node's mean tracked position, cm."""
    per = {n: [] for n in range(1, 10)}
    for xyf in glob.glob(os.path.join(LEC_V3, '*', '*', 'XY_raw_*.npy')):
        lf = xyf.replace('XY_raw_', 'Locs_raw_')
        if not os.path.exists(lf) or '2025071' in xyf:
            continue
        xy = np.load(xyf).astype(float)
        locs = np.load(lf, allow_pickle=True).astype(str)
        if xy.ndim != 2 or len(locs) != len(xy):
            continue
        if np.nanmax(xy) > 60:          # object/arena sessions are in a box frame, not the maze
            continue
        for n in range(1, 10):
            m = (locs == f'node_{n}') & np.isfinite(xy).all(axis=1)
            if m.sum() >= 50:
                per[n].append(np.median(xy[m], axis=0))
    return np.array([np.median(per[n], axis=0) for n in range(1, 10)])


def fit_affine(px: np.ndarray, cm: np.ndarray):
    """Least-squares affine px -> cm; (A (2, 3), residual per point in cm)."""
    P = np.column_stack([px, np.ones(len(px))])
    A, *_ = np.linalg.lstsq(P, cm, rcond=None)
    resid = np.linalg.norm(P @ A - cm, axis=1)
    return A.T, resid


def apply_affine(A: np.ndarray, xy_px: np.ndarray) -> np.ndarray:
    return np.column_stack([xy_px, np.ones(len(xy_px))]) @ A.T


def index_recday(rd: str, targets: np.ndarray):
    mouse = rd.split('_')[0]
    shards = shard_files(rd)
    wakes = wake_files(rd)
    shard_pop, shard_len = {}, {}
    for i, f in sorted(shards.items()):
        S = joblib.load(f, mmap_mode='r')
        shard_pop[i] = np.asarray(S, dtype=float).sum(axis=0)
        shard_len[i] = S.shape[1]

    tokens = {k: (np.load(os.path.join(BOOK, f'{k}_session_{rd}.npy'), allow_pickle=True).astype(str)
                  if os.path.exists(os.path.join(BOOK, f'{k}_session_{rd}.npy')) else None)
              for k in ('All', 'awake', 'sleep')}
    rec_rows = []
    if tokens['All'] is not None and len(tokens['All']) == len(shards):
        day, prev, dates = 0, None, _dates(rd)
        for i, tok in enumerate(tokens['All']):
            t = datetime.strptime(tok, '%H-%M-%S').time()
            if prev is not None and t < prev:
                day += 1
            prev = t
            kind = ('task' if tokens['awake'] is not None and tok in tokens['awake'] else
                    'rest' if tokens['sleep'] is not None and tok in tokens['sleep'] else 'unknown')
            start = datetime.combine(dates[min(day, len(dates) - 1)].date(), t)
            rec_rows.append(dict(recday=rd, shard=i, kind=kind, day=day, token=tok,
                                 wall_start=start.isoformat(),
                                 duration_s=shard_len[i] * BIN_MS / 1000.0))
    else:
        for i in sorted(shards):
            rec_rows.append(dict(recday=rd, shard=i, kind='unknown', day=np.nan, token=None,
                                 wall_start=None, duration_s=shard_len[i] * BIN_MS / 1000.0))

    sess_rows, aff_rows, cache = [], [], {}
    for s, f in sorted(wakes.items()):
        X = np.load(f)
        pop = X.sum(axis=0)
        hits = [(i, j) for i in shard_pop for j in find_slice(pop, shard_pop[i])]
        exact = []
        for i, j in hits:
            S = joblib.load(shards[i], mmap_mode='r')
            if np.array_equal(np.asarray(S[:, j:j + X.shape[1]]), X):
                exact.append((i, j))
        row = dict(recday=rd, session=s, n_units=X.shape[0], n_core=X.shape[1],
                   n_matches=len(exact))
        if len(exact) == 1:
            i, j = exact[0]
            row.update(shard=i, first_A_bin=j, n_post=shard_len[i] - j - X.shape[1],
                       token_ok=(tokens['awake'] is not None and s < len(tokens['awake'])
                                 and tokens['All'] is not None and i < len(tokens['All'])
                                 and tokens['All'][i] == tokens['awake'][s]))
        sess_rows.append(row)
        xy_px = np.load(os.path.join(DATA, 'XY position', f'XY_raw_{rd}_{s}.npy')).astype(float)
        locs = np.load(os.path.join(DATA, 'Maze location', f'Location_raw_{rd}_{s}.npy')).astype(float)
        n = min(len(xy_px), len(locs))
        xy_px, locs = xy_px[:n], locs[:n]
        good = np.isfinite(xy_px).all(axis=1) & (xy_px >= 0).all(axis=1)
        cent = {k: np.median(xy_px[(locs == k) & good], axis=0) for k in range(1, 10)
                if ((locs == k) & good).sum() >= 50}
        cache[s] = (X, xy_px, good, cent)

    # One affine per recday from node centroids pooled over its sessions: the camera is fixed
    # within a recday, and single-session fits wander 2-3 cm where a node is rarely visited.
    pooled = {k: np.median([c[3][k] for c in cache.values() if k in c[3]], axis=0)
              for k in range(1, 10) if any(k in c[3] for c in cache.values())}
    A_rd = resid_rd = None
    if len(pooled) >= 4:
        nodes = sorted(pooled)
        A_rd, resid_rd = fit_affine(np.array([pooled[k] for k in nodes]), targets[np.array(nodes) - 1])
    ref = PX_PER_CM.get(MAZE_OF.get(mouse))
    for row in sess_rows:
        s = row['session']
        X, xy_px, good, cent = cache[s]
        aff = dict(recday=rd, session=s, maze=MAZE_OF.get(mouse), n_nodes=len(cent),
                   affine_source='recday')
        if len(cent) >= 4:
            nodes = sorted(cent)
            A_s, r_s = fit_affine(np.array([cent[k] for k in nodes]), targets[np.array(nodes) - 1])
            aff['session_fit_px_per_cm'] = 1.0 / float(np.sqrt(abs(np.linalg.det(A_s[:, :2]))))
        if A_rd is not None:
            A = A_rd
            px_per_cm = 1.0 / float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
            sess_resid = (np.linalg.norm(apply_affine(A, np.array([cent[k] for k in sorted(cent)]))
                                         - targets[np.array(sorted(cent)) - 1], axis=1).max()
                          if cent else np.nan)
            aff.update(a11=A[0, 0], a12=A[0, 1], b1=A[0, 2], a21=A[1, 0], a22=A[1, 1], b2=A[1, 2],
                       recday_resid_max_cm=float(resid_rd.max()), node_resid_max_cm=float(sess_resid),
                       px_per_cm=px_per_cm, px_per_cm_elgaby=ref,
                       scale_err=px_per_cm / ref - 1.0 if ref else np.nan)
            xy_cm = apply_affine(A, xy_px)
            xy_cm[~good] = np.nan
            lag, r, ratio = va.speed_probe(X, xy_cm)
            tt = (np.load(os.path.join(DATA, 'Trial_times', f'trialtimes_{rd}_{s}.npy')).astype(np.int64)
                  // BIN_MS)
            ev = np.unique(tt[tt >= 0])
            pokes = np.column_stack([ev, ev, np.zeros_like(ev), np.ones_like(ev), np.zeros_like(ev)])
            peak, ptm, n_rew = va.reward_probe(X, pokes)
            speed = np.hypot(*np.gradient(xy_cm[good], axis=0).T) * (1000.0 / BIN_MS)
            row.update(speed_lag_s=lag, speed_r=r, speed_peak_ratio=ratio, reward_peak_s=peak,
                       reward_ptm=ptm, n_rewards=n_rew,
                       speed_median_cm_s=float(np.nanmedian(speed)) if len(speed) else np.nan)
        aff_rows.append(aff)
    return sess_rows, rec_rows, aff_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recdays', nargs='*', default=None)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    targets = lec_node_centroids()
    print('LEC node centroids (cm):', np.round(targets, 2).tolist(), flush=True)
    sess, rec, aff = [], [], []
    for rd in args.recdays or recdays():
        print(rd, flush=True)
        s, r, a = index_recday(rd, targets)
        sess += s
        rec += r
        aff += a
    sess, rec, aff = pd.DataFrame(sess), pd.DataFrame(rec), pd.DataFrame(aff)
    bad_slice = sess[sess['n_matches'] != 1]
    aff['mouse'] = aff['recday'].str[:4]
    aff['scale_dev'] = aff['session_fit_px_per_cm'] / aff['px_per_cm'] - 1
    # XY feeds speed and distance only, which depend on scale, not translation: a session whose
    # centroids sit 3.7-4.7 cm off its recday fit (camera moved between the two days) keeps a
    # scale within 3%, so the session test is on scale, where >= 6 nodes make its own fit usable.
    own = aff['n_nodes'] >= 6
    aff['xy_ok'] = ~((aff['recday_resid_max_cm'] > MAX_NODE_RESID_CM) | aff['recday_resid_max_cm'].isna()
                     | (own & (aff['scale_dev'].abs() > MAX_SESSION_SCALE_DEV)))
    bad_aff = aff[~aff['xy_ok']]
    if args.write:
        os.makedirs(OUT, exist_ok=True)
        for stem, df in (('pfc_session_index', sess), ('pfc_recording_index', rec), ('pfc_xy_to_cm', aff)):
            p = _versioned(os.path.join(OUT, f'{stem}.csv'))
            df.to_csv(p, index=False)
            print(f'wrote {os.path.basename(p)}: {len(df)} rows')
    print(f"\nslice gate: {int((sess['n_matches'] == 1).sum())}/{len(sess)} unique exact; "
          f"token cross-check true in {int(sess.get('token_ok', pd.Series(dtype=bool)).fillna(False).sum())}")
    print(f"affine gate: {len(aff) - len(bad_aff)}/{len(aff)} pass; node residual max median "
          f"{aff['node_resid_max_cm'].median():.2f} cm")
    print('fitted px/cm per mouse (median) vs El-Gaby:')
    print(aff.groupby('mouse').agg(px_per_cm=('px_per_cm', 'median'), elgaby=('px_per_cm_elgaby', 'first'),
                                   n=('session', 'size')).round(2).to_string())
    if 'speed_lag_s' in sess:
        t = sess[(sess['speed_r'] > va.SPEED_R_MIN) & (sess['speed_peak_ratio'] >= va.SPEED_PEAK_RATIO_MIN)]
        print(f"speed probe, testable {len(t)}/{len(sess)}: lag median {t['speed_lag_s'].median():+.2f} s "
              f"(IQR {t['speed_lag_s'].quantile(.25):+.2f} to {t['speed_lag_s'].quantile(.75):+.2f})")
        w = sess[sess['reward_ptm'] >= va.REWARD_PTM_MIN]
        print(f"reward probe, testable {len(w)}/{len(sess)}: peak median {w['reward_peak_s'].median():+.2f} s; "
              f"outside [-2, 4] s: {int(((w['reward_peak_s'] < -2) | (w['reward_peak_s'] > 4)).sum())}")
        print(f"running speed median per session: {sess['speed_median_cm_s'].median():.1f} cm/s")
    if len(bad_aff):
        print('\nXY not converted (xy_ok False; the loader treats these sessions as untracked):')
        print(bad_aff[['recday', 'session', 'n_nodes', 'recday_resid_max_cm', 'node_resid_max_cm', 'px_per_cm', 'scale_dev']].to_string(index=False))
    if len(bad_slice):
        print('\nFAILURES:')
        print(bad_slice[['recday', 'session', 'n_matches']].to_string(index=False))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
