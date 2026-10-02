"""Event-locked kernels and time constants per neuron (Tsao 2018 / Kanter 2025 replication).

Per unit: how firing depends on time since maze entry (or the first A), since the last reward A
and since the last reward, beyond place and a kinematic history bank. Stage 1 is leave-one-
session-out CPD with a Freedman-Lane null (`glm_cv.cv_scores`); the reset test compares an
entry kernel with a wall-clock day term without per-session centring; Stage 2 fits parametric
ramps exp(-dt/tau) and gamma bumps by Poisson profile likelihood, scored on held-out sessions.

Both datasets go through one session format, built from per-session files (never the 3.8 GB
pickle): `load_recday(rd, 'lec')` reads `neuron_raw_v4` + `neuron_raw_v4_full` + `processed_v3`
(+ `processed_v3_full`), `load_recday(rd, 'pfc')` reads the El-Gaby deposit + its whole-recording
shards (`pfc_session_index.csv`) with XY in cm (`pfc_xy_to_cm.csv`). Bin 0 of every session is
the start of its recording; 25 ms bins; analysis rows are 250 ms blocks.

This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).
"""

from __future__ import annotations

import glob
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import glm_cv                                                          # noqa: E402
import glm_analysis_v3 as g3                                           # noqa: E402


def _repo_root() -> str:
    p = HERE
    for _ in range(4):
        if os.path.isdir(os.path.join(p, 'code')) and os.path.isdir(os.path.join(p, 'mFC_data')):
            return p
        p = os.path.dirname(p)
    raise RuntimeError('cannot locate the repo root from ' + HERE)


REPO = _repo_root()
LEC_PROC = os.path.join(REPO, 'data', 'processed_data')
LEC_TRACK = os.path.join(REPO, 'data', 'processed_v3')
LEC_TRACK_PRE = os.path.join(REPO, 'data', 'processed_v3_full')
PFC_DATA = os.path.join(REPO, 'mFC_data', 'data')

BIN_S = 0.025
BLOCK = 10                                     # 250 ms rows, as V3
MIN_TRIALS = 10

#: Lag-bin edges (s). The last edge starts the overflow column; rows before an event's first
#: occurrence get a pre-event column. Legs > 60 s hold 22.4% of LEC samples (W1), hence overflow.
EDGES_S = {
    'entry':      [0, 10, 20, 40, 60, 90, 120, 180, 240, 360, 480, 600, 780, 960, 1200, 1500],
    'first_A':    [0, 10, 20, 40, 60, 90, 120, 180, 240, 360, 480, 600, 780, 960, 1200, 1500],
    'A':          [0, 1, 2, 4, 6, 9, 13, 18, 25, 35, 50, 70, 100, 140],
    'reward':     [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 60],
    'poke_exit':  [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 11, 16, 22, 32],
    'rest_entry': [0, 10, 20, 40, 60, 90, 120, 180, 240, 360, 480, 600, 780, 960, 1200, 1800, 2400],
}
#: Kinematic bank: every 2 octaves, because neighbours at 2 per octave correlate 0.985 on white
#: input (2 sqrt(t1 t2) / (t1 + t2)).
BANK_TAUS_S = (0.5, 2.0, 8.0, 32.0, 128.0, 512.0)
FAST_TAU_MAX_S = 8.0


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _latest(pattern: str) -> str | None:
    """Highest-version file among `stem.csv`, `stem.v1.csv`, ... matching `pattern`."""
    hits = glob.glob(pattern)
    if not hits:
        return None
    def ver(p):
        m = re.search(r'\.v(\d+)\.csv$', p)
        return int(m.group(1)) if m else 0
    return max(hits, key=ver)


#: `build_data_dic.MAZE_EDGES`, in its order, so the codes equal the pickle's `Locs_raw`.
_MAZE_EDGES = ('1-2', '2-3', '1-4', '2-5', '3-6', '4-5', '5-6', '4-7', '5-8', '6-9', '7-8', '8-9')
_LOC_CODES = {**{f'node_{i}': i for i in range(1, 10)},
              **{f'edge_{e}': 10 + i for i, e in enumerate(_MAZE_EDGES)}}


def _locs_to_int(locs) -> np.ndarray:
    """v3 string labels -> `build_data_dic.LOCATION_MAPPING` codes; 0 unknown."""
    return np.array([_LOC_CODES.get(str(v), 0) for v in locs], dtype=np.int16)


def _fit_len(a: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    a = np.asarray(a)
    if len(a) >= n:
        return a[:n]
    pad = np.full((n - len(a),) + a.shape[1:], fill, dtype=float if a.dtype.kind == 'f' else a.dtype)
    return np.concatenate([a, pad])


def load_recday(recday: str, dataset: str, with_rest: bool = False) -> dict:
    """`{'recday', 'dataset', 'sessions': [session dict], 'units': DataFrame, 'rest': [...]}`.

    Session dict: counts (units, T) on 25 ms bins from recording start; xy (T, 2) cm, NaN where
    unknown; locs (T,) int 0..21; first_A_bin; trial_times (n, 5) and pokes (n, 5) in the same
    bins; session_type, task, n_trials, wall_bin0, calendar_day, order (chronological).
    """
    if dataset == 'lec':
        rec = _load_lec(recday, with_rest)
    elif dataset == 'pfc':
        rec = _load_pfc(recday, with_rest)
    else:
        raise ValueError(dataset)
    rec['units'] = _unit_table(rec)
    return rec


LEC_FULL = os.path.join(LEC_PROC, 'full_sessions_v4')


def _load_lec(recday: str, with_rest: bool) -> dict:
    table = pd.read_csv(_latest(os.path.join(LEC_FULL, f'full_sessions_{recday}*.csv')))
    table = table.sort_values('wall_rec_start')
    flags_p = os.path.join(LEC_PROC, 'alignment_qc_lec_nv4_tv3_rv4.v2.csv')
    flags = pd.read_csv(flags_p).set_index(['recday', 'session']) if os.path.exists(flags_p) else None
    mouse, d1, d2 = recday.split('_')
    tt_dir = os.path.join(LEC_PROC, 'trialtimes_raw_mingyutest')
    day0 = {}
    sessions = []
    for order, row in enumerate(table.itertuples()):
        s, name = int(row.session), f'{recday}_{int(row.session)}'
        core = np.load(os.path.join(LEC_PROC, 'neuron_raw_v4', f'Neuron_raw_{name}.npy'))
        parts = [os.path.join(LEC_PROC, 'neuron_raw_v4_full', f'Neuron_raw_{p}_{name}.npy')
                 for p in ('pre', 'post')]
        pre = np.load(parts[0]) if os.path.exists(parts[0]) else np.zeros((core.shape[0], 0), core.dtype)
        post = np.load(parts[1]) if os.path.exists(parts[1]) else np.zeros((core.shape[0], 0), core.dtype)
        if pre.shape[1] != int(row.n_pre) or post.shape[1] != int(row.n_post):
            raise ValueError(f'{name}: full-session segments missing or wrong length')
        counts = np.concatenate([pre, core, post], axis=1)
        T, k = counts.shape[1], int(row.first_A_bin)

        blk = os.path.join(LEC_TRACK, mouse, f'{d1}_{d2}')
        blk_pre = os.path.join(LEC_TRACK_PRE, mouse, f'{d1}_{d2}')
        xy, locs = np.full((T, 2), np.nan), np.zeros(T, dtype=np.int16)
        xyf = os.path.join(blk, f'XY_raw_{name}.npy')
        if os.path.exists(xyf):
            xy[k:] = _fit_len(np.load(xyf).astype(float), T - k)
            lf = os.path.join(blk, f'Locs_raw_{name}.npy')
            if os.path.exists(lf):
                locs[k:] = _fit_len(_locs_to_int(np.load(lf, allow_pickle=True)), T - k, fill=0)
            pf = os.path.join(blk_pre, f'XY_raw_pre_{name}.npy')
            if k and os.path.exists(pf):
                xy[:k] = np.load(pf).astype(float)
                lpf = os.path.join(blk_pre, f'Locs_raw_pre_{name}.npy')
                if os.path.exists(lpf):
                    locs[:k] = _locs_to_int(np.load(lpf, allow_pickle=True))

        ttp = os.path.join(tt_dir, f'trialtimes_{name}.npy')
        tt = (np.load(ttp).astype(np.int64) + k) if os.path.exists(ttp) else np.zeros((0, 5), np.int64)
        pp = os.path.join(tt_dir, f'pokes_{name}.npy')
        pokes = np.load(pp).astype(np.int64) if os.path.exists(pp) else None
        if pokes is not None and len(pokes):
            pokes[:, :2] += k
        task_p = os.path.join(tt_dir, f'Task_data_{name}.npy')
        task = tuple(int(x) for x in np.load(task_p, allow_pickle=True)) if os.path.exists(task_p) else None
        wall0 = datetime.fromisoformat(str(row.wall_bin0))
        day0.setdefault(wall0.date(), len(day0))
        fl = {}
        if flags is not None and (recday, s) in flags.index:
            fl = flags.loc[(recday, s), ['speed_ok', 'reward_ok', 'wallclock_ok', 'early_ok']].to_dict()
        sessions.append(dict(recday=recday, session=s, dataset='lec', session_type=row.session_type,
                             task=task, n_trials=int(len(tt)), counts=counts, xy=xy, locs=locs,
                             first_A_bin=k, trial_times=tt, pokes=pokes, wall_bin0=wall0,
                             calendar_day=day0[wall0.date()], order=order, flags=fl))
    rec = dict(recday=recday, dataset='lec', sessions=sessions, rest=[])
    if with_rest:
        rec['rest'] = _load_lec_rest(recday)
    return rec


def _load_lec_rest(recday: str) -> list:
    idx = pd.read_csv(_latest(os.path.join(LEC_FULL, f'recording_index_{recday}*.csv')))
    idx = idx[idx['kind'] == 'rest'].sort_values('wall_start')
    out = []
    for r in idx.itertuples():
        p = os.path.join(LEC_PROC, 'neuron_raw_v4', f'Neuron_raw_{recday}_sb_{int(r.index)}.npy')
        if not os.path.exists(p):
            continue
        out.append(dict(recday=recday, index=int(r.index), counts=np.load(p),
                        wall_bin0=datetime.fromisoformat(str(r.wall_start)), stage=r.stage,
                        misclick=bool(r.misclick), preceding_task=r.preceding_task,
                        preceding_task_session=r.preceding_task_session))
    return out


def _load_pfc(recday: str, with_rest: bool) -> dict:
    import joblib
    proc = os.path.join(PFC_DATA, 'processed')
    sidx = pd.read_csv(_latest(os.path.join(proc, 'pfc_session_index*.csv')))
    sidx = sidx[sidx['recday'] == recday]
    ridx = pd.read_csv(_latest(os.path.join(proc, 'pfc_recording_index*.csv')))
    ridx = ridx[ridx['recday'] == recday].set_index('shard')
    aff = pd.read_csv(_latest(os.path.join(proc, 'pfc_xy_to_cm*.csv')))
    aff = aff[aff['recday'] == recday].set_index('session')
    tasks = np.load(os.path.join(PFC_DATA, 'Tasks', f'Task_data_{recday}.npy'), allow_pickle=True)
    shards = os.path.join(PFC_DATA, 'Neuronal_activity', 'Awake_Sleep')
    sessions, day0 = [], {}
    rows = sidx.dropna(subset=['shard']).copy()
    rows['shard'] = rows['shard'].astype(int)
    rows = rows.sort_values('shard')
    for order, row in enumerate(rows.itertuples()):
        s, i, off = int(row.session), int(row.shard), int(row.first_A_bin)
        S = joblib.load(os.path.join(shards, f'binned_FR_dic_{recday}_{i}'), mmap_mode='r')
        start = off % BLOCK                       # first A on a block boundary, as for LEC
        counts = np.asarray(S[:, start:], dtype=np.float64).astype(np.uint16)
        T, k = counts.shape[1], off - start
        n_core = int(row.n_core)
        xy, locs = np.full((T, 2), np.nan), np.zeros(T, dtype=np.int16)
        xy_px = np.load(os.path.join(PFC_DATA, 'XY position', f'XY_raw_{recday}_{s}.npy')).astype(float)
        if (s in aff.index and np.isfinite(aff.loc[s].get('a11', np.nan))
                and bool(aff.loc[s].get('xy_ok', True))):
            a = aff.loc[s]
            A = np.array([[a.a11, a.a12, a.b1], [a.a21, a.a22, a.b2]])
            good = np.isfinite(xy_px).all(axis=1) & (xy_px >= 0).all(axis=1)
            cm = np.column_stack([xy_px, np.ones(len(xy_px))]) @ A.T
            cm[~good] = np.nan
            xy[k:k + min(len(cm), T - k)] = cm[:T - k]
        lr = np.load(os.path.join(PFC_DATA, 'Maze location', f'Location_raw_{recday}_{s}.npy')).astype(float)
        lr = np.where(np.isfinite(lr), lr, 0).astype(np.int16)
        locs[k:k + min(len(lr), T - k)] = lr[:T - k]
        tt = (np.load(os.path.join(PFC_DATA, 'Trial_times', f'trialtimes_{recday}_{s}.npy')).astype(np.int64)
              // 25 + k)
        wall = ridx.loc[i, 'wall_start'] if i in ridx.index else None
        wall0 = (datetime.fromisoformat(str(wall)) + timedelta(seconds=start * BIN_S)
                 if isinstance(wall, str) else None)
        day = ridx.loc[i, 'day'] if i in ridx.index else np.nan
        sessions.append(dict(recday=recday, session=s, dataset='pfc', session_type='ABCD',
                             task=tuple(int(x) for x in tasks[s]) if s < len(tasks) else None,
                             n_trials=int(len(tt)), counts=counts, xy=xy, locs=locs, first_A_bin=k,
                             trial_times=tt, pokes=None, wall_bin0=wall0,
                             calendar_day=int(day) if np.isfinite(day) else 0, order=order,
                             flags={}, n_core=n_core))
    rest = []
    if with_rest:
        for i, r in ridx.iterrows():
            if r['kind'] != 'rest':
                continue
            S = joblib.load(os.path.join(shards, f'binned_FR_dic_{recday}_{int(i)}'), mmap_mode='r')
            rest.append(dict(recday=recday, index=int(i), counts=np.asarray(S).astype(np.uint16),
                             wall_bin0=datetime.fromisoformat(str(r['wall_start'])) if isinstance(r['wall_start'], str) else None,
                             stage=None, misclick=bool(r['duration_s'] < 120)))
    return dict(recday=recday, dataset='pfc', sessions=sessions, rest=rest)


def elgaby_rule_drop(rates: np.ndarray) -> np.ndarray:
    """El-Gaby 2024 curation rule 2 on (sessions, units) mean rates: True = he would discard."""
    peak = rates.max(axis=0)
    return (rates < 0.2 * peak[None, :]).sum(axis=0) >= 3


def _unit_table(rec: dict) -> pd.DataFrame:
    """Per unit: region (LEC from `anatomy_split`, PFC 'PFC'), depth, task-session rate, the
    El-Gaby rule flag (computed over every task session of the recday)."""
    task = [s for s in rec['sessions'] if s['n_trials'] > 0 or s['session_type'] == 'ABCD']
    if not task:
        return pd.DataFrame()
    rates = np.array([s['counts'].mean(axis=1) / BIN_S for s in task])
    df = pd.DataFrame(dict(recday=rec['recday'], unit=np.arange(rates.shape[1]),
                           rate_hz=rates.mean(axis=0), elgaby_drop=elgaby_rule_drop(rates)))
    if rec['dataset'] == 'lec':
        sys.path.insert(0, os.path.join(REPO, 'code'))
        import anatomy_split as asp
        ur = asp.load_unit_regions()[rec['recday']].reset_index(drop=True)
        if len(ur) != len(df):
            raise ValueError(f"{rec['recday']}: {len(ur)} region rows vs {len(df)} units")
        df['region'] = ur['group'].to_numpy()
        df['acronym'] = ur['acronym'].to_numpy()
        df['y_um'] = ur['y_um'].to_numpy()
        df['shank'] = ur['shank'].to_numpy()
    else:
        df['region'] = 'PFC'
    df['mouse'] = rec['recday'].split('_')[0]
    return df


def analysis_sessions(rec: dict, min_trials: int = MIN_TRIALS) -> list:
    """ABCD sessions with at least `min_trials` completed trials, in chronological order."""
    return [s for s in rec['sessions'] if s['session_type'] == 'ABCD' and s['n_trials'] >= min_trials]


# ---------------------------------------------------------------------------
# 25 ms features -> 250 ms blocks
# ---------------------------------------------------------------------------

def event_bins(sess: dict, kind: str) -> np.ndarray:
    """Bins of one event type in the session's full timeline."""
    tt = sess['trial_times']
    if kind in ('entry', 'rest_entry'):
        return np.array([0])
    if kind == 'first_A':
        return np.array([sess['first_A_bin']])
    if kind == 'A':
        return np.unique(np.r_[tt[:, 0], tt[-1:, -1]]) if len(tt) else np.zeros(0, int)
    if kind == 'reward':
        return np.unique(tt.ravel()) if len(tt) else np.zeros(0, int)
    if kind == 'poke_exit':
        p = sess.get('pokes')
        return np.unique(p[p[:, 3] == 1, 1]) if p is not None and len(p) else np.zeros(0, int)
    raise ValueError(kind)


def time_since(events: np.ndarray, T: int) -> np.ndarray:
    """(T,) seconds since the latest event at or before each bin; NaN before the first."""
    out = np.full(T, np.nan)
    ev = np.sort(np.asarray(events, dtype=np.int64))
    ev = ev[(ev >= 0) & (ev < T)]
    if not len(ev):
        return out
    t = np.arange(T)
    j = np.searchsorted(ev, t, side='right') - 1
    ok = j >= 0
    out[ok] = (t[ok] - ev[j[ok]]) * BIN_S
    return out


def block_mean(a: np.ndarray, how: str = 'mean') -> np.ndarray:
    """(T, ...) -> (T // BLOCK, ...) by block mean, nanmean or sum."""
    n = len(a) // BLOCK
    b = np.asarray(a[:n * BLOCK]).reshape((n, BLOCK) + a.shape[1:])
    if how == 'sum':
        return b.sum(axis=1)
    if how == 'nanmean':
        with np.errstate(invalid='ignore'):
            return np.nanmean(b.astype(float), axis=1)
    return b.astype(float).mean(axis=1)


def lag_onehot_blocks(t: np.ndarray, edges) -> np.ndarray:
    """Fraction of each 250 ms block in each lag bin, then overflow, then pre-event: (n, k+2)."""
    nb = len(edges) - 1
    idx = np.full(len(t), nb + 1)
    fin = np.isfinite(t)
    idx[fin] = np.clip(np.digitize(t[fin], edges) - 1, 0, nb)
    n = len(t) // BLOCK
    out = np.zeros((n, nb + 2))
    rows = np.arange(n * BLOCK) // BLOCK
    np.add.at(out, (rows, idx[:n * BLOCK]), 1.0 / BLOCK)
    return out


def kinematics(xy: np.ndarray):
    """(speed cm/s, |accel| cm/s^2, signed accel cm/s^2) at 25 ms, NaN where untracked."""
    d = g3.smooth_and_calculate_scalar_derivatives(np.asarray(xy, dtype=float), sigma=3)
    speed = d[:, 2] / BIN_S
    acc = d[:, 3] / BIN_S ** 2
    return speed, np.abs(acc), acc


def exp_bank(x: np.ndarray, taus_s=BANK_TAUS_S, init: str = 'steady') -> np.ndarray:
    """Causal leaky traces y = a y[-1] + (1 - a) x at 25 ms for each tau: (T, n_tau).

    NaN input is interpolated (gaps are coded missing downstream). `init='steady'` starts each
    trace at the session mean; `'zero'` starts at 0, which makes the bank itself rise as
    mean (1 - exp(-t / tau)) after entry (the start-up artefact WP1 measures).
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.zeros((len(x), len(taus_s)))
    x = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
    y0 = float(np.mean(x)) if init == 'steady' else 0.0
    out = np.empty((len(x), len(taus_s)))
    for j, tau in enumerate(taus_s):
        a = np.exp(-BIN_S / tau)
        out[:, j], _ = lfilter([1 - a], [1, -a], x, zi=[a * y0])
    return out


def session_blocks(sess: dict, events=('entry', 'A', 'reward'), bank_init: str = 'steady') -> dict:
    """250 ms block features of one session: counts (units, n), the raw blocks per family."""
    T = sess['counts'].shape[1]
    n = T // BLOCK
    speed, absacc, acc = kinematics(sess['xy'])
    bank_s = exp_bank(speed, init=bank_init)
    bank_a = exp_bank(absacc, init=bank_init)
    kin_missing = ~np.isfinite(block_mean(speed, 'nanmean'))
    locs = sess['locs'].astype(int)
    place = np.zeros((n, 22))
    np.add.at(place, (np.arange(n * BLOCK) // BLOCK, np.clip(locs[:n * BLOCK], 0, 21)), 1.0 / BLOCK)
    out = dict(counts=block_mean(sess['counts'].T, 'sum').T.astype(np.float64),
               place=place, speed=block_mean(speed, 'nanmean'), acc=block_mean(acc, 'nanmean'),
               bank=np.column_stack([block_mean(bank_s), block_mean(bank_a)]), kin_missing=kin_missing,
               t0_s=np.arange(n) * BLOCK * BIN_S)
    for e in events:
        out[f'ev_{e}'] = lag_onehot_blocks(time_since(event_bins(sess, e), T), EDGES_S[e])
        out[f'dt_{e}'] = block_mean(time_since(event_bins(sess, e), T), 'mean')
    return out


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------

def _reference_coded(M: np.ndarray, sids: np.ndarray, min_sessions: int = 2):
    """Drop the most-occupied column (the reference) and columns populated in < 2 sessions."""
    occ = M.sum(axis=0)
    keep = np.ones(M.shape[1], bool)
    keep[int(np.argmax(occ))] = False
    for j in range(M.shape[1]):
        if keep[j] and len(np.unique(sids[M[:, j] > 0])) < min_sessions:
            keep[j] = False
    return M[:, keep], np.flatnonzero(keep), int(np.argmax(occ))


def build_design(sessions: list, events=('entry', 'A', 'reward'), *, kin: str = 'full',
                 place: bool = True, day_clock: bool = False, bank_init: str = 'steady',
                 blocks: list | None = None) -> dict:
    """Stacked design over sessions: X (rows, p), FR (units, rows), session_ids, groups.

    Groups: 'place', 'kin_inst' (speed / signed accel deciles), 'kin_fast' / 'kin_slow' (bank tau
    <= / > 8 s), 'kin_missing', one per event ('ev_<e>'), and 'day' when `day_clock`.
    `kin='none'` drops every kinematic column (the control that the bank is not vacuous);
    `place=False` drops the place block (the same control for place).
    """
    blocks = blocks or [session_blocks(s, events, bank_init) for s in sessions]
    sids = np.concatenate([np.full(b['counts'].shape[1], i) for i, b in enumerate(blocks)])
    cols, groups, names = [], {}, []

    def add(name, M, labels):
        start = sum(c.shape[1] for c in cols)
        cols.append(M)
        groups[name] = list(range(start, start + M.shape[1]))
        names.extend(f'{name}:{l}' for l in labels)

    if place:
        P, kept, _ = _reference_coded(np.vstack([b['place'] for b in blocks]), sids)
        add('place', P, [str(k) for k in kept])
    miss = np.concatenate([b['kin_missing'] for b in blocks])
    if kin != 'none':
        for var in ('speed', 'acc'):
            v = np.concatenate([b[var] for b in blocks])
            edges = g3.compute_decile_edges(v[np.isfinite(v)], n_bins=10, outlier_pct=1, scheme='decile')
            M = g3.apply_onehot(v, edges)
            M, kept, _ = _reference_coded(M, sids)
            add(f'kin_inst_{var}', M, [str(k) for k in kept])
        bank = np.vstack([b['bank'] for b in blocks])
        bank = (bank - np.nanmean(bank[~miss], axis=0)) / np.nanstd(bank[~miss], axis=0)
        bank[miss] = 0.0
        taus = list(BANK_TAUS_S) * 2
        fast = [j for j, t in enumerate(taus) if t <= FAST_TAU_MAX_S]
        slow = [j for j, t in enumerate(taus) if t > FAST_TAU_MAX_S]
        lab = [f'speed_tau{t}' for t in BANK_TAUS_S] + [f'absacc_tau{t}' for t in BANK_TAUS_S]
        add('kin_fast', bank[:, fast], [lab[j] for j in fast])
        add('kin_slow', bank[:, slow], [lab[j] for j in slow])
        if miss.any() and len(np.unique(sids[miss])) >= 1 and miss.mean() < 1:
            add('kin_missing', miss[:, None].astype(float), ['missing'])
    for e in events:
        M = np.vstack([b[f'ev_{e}'] for b in blocks])
        M, kept, _ = _reference_coded(M, sids)
        edges = EDGES_S[e]
        lab = [f'{edges[j]}-{edges[j + 1]}s' if j < len(edges) - 1 else ('overflow' if j == len(edges) - 1 else 'pre')
               for j in kept]
        add(f'ev_{e}', M, lab)
    if day_clock:
        D = np.vstack([_day_basis(s, b) for s, b in zip(sessions, blocks)])
        keep = [j for j in range(D.shape[1]) if np.ptp(D[:, j]) > 0]     # a one-day recday has no day-2 columns
        add('day', D[:, keep], [('hours_day1', 'hours_day2', 'day2')[j] for j in keep])
    X = np.hstack(cols)
    FR = np.hstack([b['counts'] for b in blocks])
    return dict(X=X, FR=FR, session_ids=sids, groups=groups, names=names, blocks=blocks,
                n_rows=len(sids), kin_missing=miss)


def _day_basis(sess: dict, b: dict) -> np.ndarray:
    """Hours since the session's calendar day began, one slope per day, plus a day-2 offset:
    the drift-like alternative to an entry kernel. Linear, because hat functions cannot
    extrapolate to a held-out first or last session (synthetic day trend: held-out RSS 7095
    for the day model against 5803 for the entry model)."""
    h = sess['hours_since_day_start'] + b['t0_s'] / 3600.0
    d2 = float(sess['calendar_day'] > 0)
    return np.column_stack([h * (1 - d2), h * d2, np.full(len(h), d2)])


def add_day_clock(rec: dict) -> None:
    """Attach `hours_since_day_start` to every task session (first recording of its calendar day,
    task or rest, from the recording index)."""
    starts = {}
    if rec['dataset'] == 'lec':
        idx = pd.read_csv(_latest(os.path.join(LEC_FULL, f"recording_index_{rec['recday']}*.csv")))
        for d, g in idx.groupby('date'):
            starts[datetime.strptime(str(d), '%Y%m%d').date()] = pd.to_datetime(g['wall_start']).min()
    else:
        ridx = pd.read_csv(_latest(os.path.join(PFC_DATA, 'processed', 'pfc_recording_index*.csv')))
        ridx = ridx[(ridx['recday'] == rec['recday']) & ridx['wall_start'].notna()]
        for _, g in ridx.groupby('day'):
            t = pd.to_datetime(g['wall_start'])
            starts[t.min().date()] = t.min()
    for s in rec['sessions']:
        w = s.get('wall_bin0')
        if w is not None and w.date() in starts:
            s['hours_since_day_start'] = (w - starts[w.date()].to_pydatetime()).total_seconds() / 3600.0


def center_within_sessions(X: np.ndarray, sids: np.ndarray) -> np.ndarray:
    """Per-session column means removed: with FR centred too this equals per-session offsets."""
    Xc = X.astype(float).copy()
    for s in np.unique(sids):
        m = sids == s
        Xc[m] -= Xc[m].mean(axis=0, keepdims=True)
    return Xc


# ---------------------------------------------------------------------------
# Stage 1 and the reset test
# ---------------------------------------------------------------------------

TESTED = ('ev_entry', 'ev_first_A', 'ev_A', 'ev_reward', 'ev_poke_exit', 'kin_fast', 'kin_slow', 'place')


def stage1(sessions: list, events=('entry', 'A', 'reward'), *, n_perm: int = 100, seed: int = 0,
           kin: str = 'full', place: bool = True, bank_init: str = 'steady',
           design: dict | None = None, tested=TESTED) -> dict:
    """Leave-one-session-out CPD per unit with a Freedman-Lane null, both sides centred within
    session. Returns `cv_scores` output plus the design bookkeeping and a per-fold rank check."""
    D = design or build_design(sessions, events, kin=kin, place=place, bank_init=bank_init)
    X = center_within_sessions(D['X'], D['session_ids'])
    groups = {g: D['groups'][g] for g in tested if g in D['groups']}
    joints = []
    ev = [f'ev_{e}' for e in events if f'ev_{e}' in D['groups']]
    if 'ev_A' in ev and 'ev_reward' in ev:
        joints.append(('ev_A+reward', ['ev_A', 'ev_reward']))
    anchor = 'ev_entry' if 'ev_entry' in ev else ('ev_first_A' if 'ev_first_A' in ev else None)
    if anchor and 'kin_slow' in D['groups']:
        groups.setdefault('kin_slow', D['groups']['kin_slow'])
        joints.append((f'{anchor}+kin_slow', [anchor, 'kin_slow']))
    res = glm_cv.cv_scores(X, D['FR'], D['session_ids'], groups, n_perm=n_perm, seed=seed,
                           joint_specs=joints, center_within_sessions=True, nulls=('freedman_lane',))
    ranks = []
    for f in np.unique(D['session_ids']):
        tr = D['session_ids'] != f
        ranks.append(int(np.linalg.matrix_rank(X[tr])))
    res['fold_rank_min'] = min(ranks)
    res['n_cols'] = X.shape[1]
    res['col_names'] = D['names']
    res['groups'] = D['groups']
    res['n_rows'] = D['n_rows']
    return res


def reset_test(rec: dict, sessions: list, events=('entry', 'A', 'reward')) -> dict:
    """Uncentred leave-one-session-out: entry kernel vs wall-clock day term vs both.

    A day-long trend shows the same within-session slope in every session, so under per-session
    centring it fits an entry kernel too; only the session means separate the two. Returns
    `cv_scores` on [nuisance, events, day] with groups ev_entry and day: cpd of each beyond the
    other; rss_reduced['day'] is the entry-only model, rss_reduced['ev_entry'] the day-only one.
    """
    add_day_clock(rec)
    if any('hours_since_day_start' not in s for s in sessions):
        return dict(skipped='no wall clock for every session')
    D = build_design(sessions, events, day_clock=True)
    X = np.column_stack([np.ones(D['n_rows']), D['X']])
    groups = {g: [c + 1 for c in D['groups'][g]] for g in ('ev_entry', 'day') if g in D['groups']}
    res = glm_cv.cv_scores(X, D['FR'], D['session_ids'], groups, n_perm=0,
                           center_within_sessions=False)
    res['entry_beats_day'] = res['rss_reduced']['day'] < res['rss_reduced']['ev_entry']
    return res


# ---------------------------------------------------------------------------
# Stage 2: parametric shapes
# ---------------------------------------------------------------------------

def ramp(dt, tau):
    """exp(-dt / tau); 0 where dt is NaN (before the first event)."""
    return np.where(np.isfinite(dt), np.exp(-np.nan_to_num(dt) / tau), 0.0)


def bump(dt, peak, k):
    """Gamma-shaped bump, height 1 at `peak`; FWHM grows as peak / sqrt(k)."""
    x = np.nan_to_num(dt) / peak
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(x > 0, np.exp(k * (np.log(np.maximum(x, 1e-12)) + 1 - x)), 0.0)
    return np.where(np.isfinite(dt), g, 0.0)


def shape_grid(event: str):
    """Ramp taus and (peak, k) bump pairs over the event's lag range, 2 per octave."""
    e = EDGES_S[event]
    lo, hi = max(e[1], 0.25), e[-1]
    taus = lo / 2 * 2 ** (np.arange(int(np.ceil(2 * np.log2(8 * hi / lo))) + 1) / 2)
    peaks = lo * 2 ** (np.arange(int(np.ceil(2 * np.log2(hi / lo))) + 1) / 2)
    return taus, [(p, k) for p in peaks for k in (2, 4, 8, 16)]


def _poisson_newton(y, Z, offset, iters=30, ridge=1e-6):
    """Few-parameter Poisson MLE with offset: (beta, deviance)."""
    beta = np.zeros(Z.shape[1])
    beta[0] = np.log(max(y.sum(), 1e-9) / max(np.exp(offset).sum(), 1e-12))
    for _ in range(iters):
        eta = np.clip(offset + Z @ beta, -30, 30)
        mu = np.exp(eta)
        g = Z.T @ (y - mu)
        H = Z.T @ (Z * mu[:, None]) + ridge * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = np.exp(np.clip(offset + Z @ beta, -30, 30))
    return beta, _dev(y, mu)


def _dev(y, mu):
    mu = np.maximum(mu, 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.sum(t - (y - mu)))


def _heldout_dev(y, offset):
    """Held-out deviance after refitting only the session's own intercept (its per-session
    offset, which leave-one-session-out cannot know)."""
    c = np.log(max(y.sum(), 1e-9) / max(np.exp(np.clip(offset, -30, 30)).sum(), 1e-12))
    return _dev(y, np.exp(np.clip(offset + c, -30, 30)))


def stage2_unit(D: dict, unit: int, event: str, glm_alpha: float = 1e-4) -> dict:
    """Held-out (leave one session out) Poisson deviance of: no event-e kernel, the shape-free
    kernel, the best ramp and the best bump, with tau / peak chosen on the training sessions.

    The offset is the training-fold full model's linear predictor minus the event block, so
    shared variance stays with the other regressors as in the full model.
    """
    from glum import GeneralizedLinearRegressor
    y_all = D['FR'][unit]
    sids = D['session_ids']
    ecols = D['groups'][f'ev_{event}']
    dt_all = np.concatenate([b[f'dt_{event}'] for b in D['blocks']])
    pre_all = (~np.isfinite(dt_all)).astype(float)
    S = (sids[:, None] == np.unique(sids)[None, :]).astype(float)        # session intercepts
    X = D['X']
    taus, bumps = shape_grid(event)
    out = dict(unit=unit, event=event, dev_null=0.0, dev_np=0.0, dev_ramp=0.0, dev_bump=0.0)
    picks = []
    for f in np.unique(sids):
        tr, te = sids != f, sids == f
        Xtr = np.hstack([X[tr], S[tr][:, np.unique(sids[tr])]])
        m = GeneralizedLinearRegressor(family='poisson', alpha=glm_alpha, fit_intercept=False,
                                       max_iter=200)
        m.fit(Xtr, y_all[tr])
        coef = m.coef_
        beta_x, beta_s = coef[:X.shape[1]], coef[X.shape[1]:]
        off_x = X @ beta_x - X[:, ecols] @ beta_x[ecols]
        off_tr = off_x[tr] + S[tr][:, np.unique(sids[tr])] @ beta_s
        off_te = off_x[te]
        y_tr, y_te = y_all[tr], y_all[te]
        out['dev_null'] += _heldout_dev(y_te, off_te)
        out['dev_np'] += _heldout_dev(y_te, off_te + X[te][:, ecols] @ beta_x[ecols])
        Zpre_tr, Zpre_te = pre_all[tr], pre_all[te]
        use_pre = Zpre_tr.any()
        best = {}
        for kind, grid in (('ramp', taus), ('bump', bumps)):
            fits = []
            for g in grid:
                k_all = ramp(dt_all, g) if kind == 'ramp' else bump(dt_all, *g)
                cols = [np.ones(tr.sum()), k_all[tr]] + ([Zpre_tr] if use_pre else [])
                b, dev = _poisson_newton(y_tr, np.column_stack(cols), off_tr)
                fits.append((dev, g, b))
            dev, g, b = min(fits, key=lambda z: z[0])
            k_all = ramp(dt_all, g) if kind == 'ramp' else bump(dt_all, *g)
            eta_te = off_te + b[1] * k_all[te] + (b[2] * Zpre_te if use_pre else 0.0)
            out[f'dev_{kind}'] += _heldout_dev(y_te, eta_te)
            best[kind] = (g, b[1])
        picks.append(best)
    gain_np = out['dev_null'] - out['dev_np']
    out['gain_np'] = gain_np
    for kind in ('ramp', 'bump'):
        out[f'frac_{kind}'] = (out['dev_null'] - out[f'dev_{kind}']) / gain_np if gain_np > 0 else np.nan
    tau = np.array([p['ramp'][0] for p in picks], float)
    w = np.array([p['ramp'][1] for p in picks], float)
    pk = np.array([p['bump'][0][0] for p in picks], float)
    kk = np.array([p['bump'][0][1] for p in picks], float)
    out.update(tau_median=float(np.median(tau)), tau_log_iqr=float(np.subtract(*np.percentile(np.log(tau), [75, 25]))),
               ramp_weight_median=float(np.median(w)), ramp_sign_consistency=float(np.mean(np.sign(w) == np.sign(np.median(w)))),
               peak_median=float(np.median(pk)), k_median=float(np.median(kk)),
               tau_at_edge=bool(np.median(tau) >= taus[-1] or np.median(tau) <= taus[0]))
    fr, fb = out['frac_ramp'], out['frac_bump']
    if not gain_np > 0:
        cls = 'none'
    elif max(np.nan_to_num(fr), np.nan_to_num(fb)) < 0.5:
        cls = 'other'
    elif np.nan_to_num(fr) >= np.nan_to_num(fb):
        cls = 'ramp_down' if out['ramp_weight_median'] > 0 else 'ramp_up'
    else:
        cls = 'bump'
    out['shape'] = cls
    return out
