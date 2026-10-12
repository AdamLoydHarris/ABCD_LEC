"""Event-kernel Stage 1 and the reset test on real data: one recday per job, a stamped merge (M2).

    python code/run_event_kernels.py --dataset lec --recday ah10_20250616_20250617 --anchor entry
    python code/run_event_kernels.py --dataset lec --merge [--min-trials 10 --n-perm 100]
    python code/run_event_kernels.py --dataset lec --gate          # planted units through this path

Anchor `entry` fits events (entry, A, reward) and the reset test; `first_A` fits (first_A, A,
reward) on the same rows. Shards: `data/event_kernels/stage1/shards/<arm>/<recday>_<anchor>.pkl`
as `{'config': stamp, 'data': ...}`, arm = `<dataset>_kin-<kin>_mt<min_trials>_np<n_perm>`. A
recday with < 2 analysis sessions writes a `skipped` shard. The merge refuses mismatched stamps,
duplicate recdays and a short recday count, and writes a versioned pickle + long CSVs. Betas and
raw lag-bin rates are kept for figures only; the call is Stage 1's (locked = Freedman-Lane
p < 0.05 and held-out cpd > 0, `event_kernels_synthetics.locked`).

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import pickle
import re
import shutil
import socket
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import event_kernels_synthetics as syn                                 # noqa: E402
import glm_cv                                                          # noqa: E402

IS_PFC = 'mFC_data' in HERE
ANCHORS = ('entry', 'first_A')
N_RECDAYS = 25
STAGE1_DIR = os.path.join(ek.REPO, 'data', 'event_kernels', 'stage1')


# ---------------------------------------------------------------------------
# shards: stamp, atomic write, versioned output, stamped load
# ---------------------------------------------------------------------------

def _sha1(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def versioned(path: str) -> str:
    """`stem.ext`, else `stem.v1.ext`, ...: the first name not on disk (nothing is overwritten)."""
    stem, ext = os.path.splitext(path)
    k = 0
    while os.path.exists(p := stem + (f'.v{k}' if k else '') + ext):
        k += 1
    return p


def write_atomic(path: str, obj) -> None:
    """Pickle to a temp file, then rename: a preempted job leaves no half-written shard."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_shards(shard_dir: str, key) -> tuple:
    """(config, [data]) from every shard in `shard_dir`; refuses unstamped shards, a stamp that
    differs from the first one, and two shards with the same `key(data)`."""
    paths = sorted(glob.glob(os.path.join(shard_dir, '*.pkl')))
    if not paths:
        raise FileNotFoundError(f'no shards in {shard_dir}')
    seen, out, keys = None, [], {}
    for p in paths:
        with open(p, 'rb') as fh:
            raw = pickle.load(fh)
        if not (isinstance(raw, dict) and set(raw) == {'config', 'data'}):
            raise ValueError(f'{os.path.basename(p)} is not a stamped shard; refusing to merge')
        if seen is None:
            seen = raw['config']
        elif raw['config'] != seen:
            diff = {k: (seen.get(k), raw['config'].get(k)) for k in set(seen) | set(raw['config'])
                    if seen.get(k) != raw['config'].get(k)}
            raise ValueError(f'{os.path.basename(p)} has a different stamp; refusing to merge. '
                             f'Differences (first, this): {diff}')
        k = key(raw['data'])
        if k in keys:
            raise ValueError(f'{k} appears in {keys[k]} and {os.path.basename(p)}; refusing to merge')
        keys[k] = os.path.basename(p)
        out.append(raw['data'])
    return seen, out


def check_or_skip(path: str, st: dict) -> bool:
    """True when `path` already holds a shard with this stamp (a requeued job that had finished);
    raises when it holds a different one, because outputs are never overwritten."""
    if not os.path.exists(path):
        return False
    with open(path, 'rb') as fh:
        old = pickle.load(fh)
    if isinstance(old, dict) and old.get('config') == st:
        return True
    raise SystemExit(f'{path} exists with a different stamp; move it aside first')


def arm_name(dataset, kin, min_trials, n_perm, synthetic=None) -> str:
    a = f'{dataset}_kin-{kin}_mt{min_trials}_np{n_perm}'
    return f'synthetic-{synthetic}_{a}' if synthetic else a


def input_versions(dataset: str) -> dict:
    """Dataset-wide inputs whose version must agree across shards."""
    if dataset == 'lec':
        return dict(alignment_qc='alignment_qc_lec_nv4_tv3_rv4.v2.csv',
                    unit_regions_sha1=_sha1(os.path.join(ek.LEC_PROC, 'unit_regions.pkl')))
    proc = os.path.join(ek.PFC_DATA, 'processed')
    return {k: os.path.basename(ek._latest(os.path.join(proc, f'{k}*.csv')) or '')
            for k in ('pfc_session_index', 'pfc_recording_index', 'pfc_xy_to_cm')}


def stamp(dataset, kin, min_trials, n_perm, seed, synthetic=None) -> dict:
    """Everything that must agree across the shards of one merge (recday and anchor excluded)."""
    return dict(runner='run_event_kernels', dataset=dataset, kin=kin, min_trials=int(min_trials),
                n_perm=int(n_perm), seed=int(seed), synthetic=synthetic, block=ek.BLOCK,
                bin_s=ek.BIN_S, edges_s={e: list(ek.EDGES_S[e]) for e in ('entry', 'first_A', 'A', 'reward')},
                bank_taus_s=list(ek.BANK_TAUS_S), fast_tau_max_s=ek.FAST_TAU_MAX_S,
                sha1_event_kernels=_sha1(ek.__file__), sha1_glm_cv=_sha1(glm_cv.__file__),
                sha1_glm_analysis_v3=_sha1(ek.g3.__file__), sha1_runner=_sha1(os.path.abspath(__file__)),
                inputs=input_versions(dataset))


# ---------------------------------------------------------------------------
# one recday
# ---------------------------------------------------------------------------

def unit_table(rec: dict) -> pd.DataFrame:
    """`rec['units']` plus the `anatomy_split` columns (`neuron`, `group`) and `near_boundary`
    (within 50 um of a region transition on the unit's shank)."""
    u = rec['units'].copy().reset_index(drop=True)
    u['neuron'] = u['unit']
    u['group'] = u['region']
    if {'shank', 'y_um'} <= set(u.columns):
        sys.path.insert(0, os.path.join(ek.REPO, 'code'))
        import anatomy_split as asp
        u['near_boundary'] = asp.boundary_margin_filter(u, margin_um=50.0)['near_boundary'].to_numpy()
    else:
        u['near_boundary'] = False
    return u


def synthetic_units(truth: pd.DataFrame, recday: str) -> pd.DataFrame:
    """Planted units in the unit-table format; `group` is the planted kind."""
    u = truth.copy()
    u['recday'], u['mouse'] = recday, recday.split('_')[0]
    u['neuron'], u['region'], u['group'] = u['unit'], u['kind'], u['kind']
    u['rate_hz'], u['elgaby_drop'], u['near_boundary'] = np.nan, False, False
    return u


def session_table(sessions: list) -> pd.DataFrame:
    rows = []
    for s in sessions:
        rows.append(dict(session=s['session'], order=s['order'], n_trials=s['n_trials'], task=str(s['task']),
                         n_bins=s['counts'].shape[1], first_A_bin=s['first_A_bin'],
                         calendar_day=s['calendar_day'], wall_bin0=str(s['wall_bin0']),
                         hours_since_day_start=s.get('hours_since_day_start', np.nan),
                         **{f'flag_{k}': v for k, v in (s.get('flags') or {}).items()}))
    return pd.DataFrame(rows)


def lag_labels(event: str) -> list:
    """Every lag column of an event block, in `lag_onehot_blocks` order, labelled as `build_design`."""
    e = ek.EDGES_S[event]
    return [f'{e[j]}-{e[j + 1]}s' for j in range(len(e) - 1)] + ['overflow', 'pre']


def rank_report(X: np.ndarray, names: list) -> tuple:
    """(columns with a loading on the null space of X, whether the deficiency is only all-zero
    and duplicated columns). The smoke run found only those: the `pre` columns of A, reward and
    first A mark the same rows before the first A (and in PFC equal `kin_missing:pre_A`), and a
    fold whose held-out session holds every tracking gap has an all-zero `kin_missing:gap`."""
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    null = vt[s <= s.max() * max(X.shape) * np.finfo(float).eps]
    if not len(null):
        return [], True
    support = np.flatnonzero(np.abs(null).max(axis=0) > 1e-6)
    zero = [j for j in support if not np.any(X[:, j])]
    groups = {}
    for j in support:
        if j not in zero:
            groups.setdefault(X[:, j].tobytes(), []).append(j)
    return [names[j] for j in support], len(zero) + sum(len(g) - 1 for g in groups.values()) == len(null)


def rank_check(D: dict) -> dict:
    """Null-space columns of every training fold and of the all-session design (centred)."""
    sids = D['session_ids']
    Xc = ek.center_within_sessions(D['X'], sids)
    cols, benign = set(), True
    for f in np.unique(sids):
        c, ok = rank_report(Xc[sids != f], D['names'])
        cols |= set(c)
        benign &= ok
    c_all, ok_all = rank_report(Xc, D['names'])
    return dict(fold_null_columns=sorted(cols), fold_deficiency_benign=bool(benign),
                all_null_columns=c_all, all_deficiency_benign=bool(ok_all))


def kernels(D: dict, events, kin: str, unidentified=()) -> dict:
    """Per event block, for figures: betas of one all-session OLS fit on the centred design (Hz;
    the reference bin 0, bins dropped for rarity or in the null space NaN) and the raw mean rate
    per lag bin (Hz)."""
    sids = D['session_ids']
    B = glm_cv._pinv(ek.center_within_sessions(D['X'], sids)) @ glm_cv.center_within_session(D['FR'], sids).T
    B[[D['names'].index(c) for c in unidentified], :] = np.nan
    to_hz = 1.0 / (ek.BLOCK * ek.BIN_S)
    out = {}
    for e in events:
        labels = lag_labels(e)
        M = np.vstack([b[f'ev_{e}'] for b in D['blocks']])
        if kin == 'strict' and e in ek.FAST_EVENTS:
            M[D['kin_missing']] = 0.0          # as build_design, so the reference is the same column
        occ = M.sum(axis=0)
        ref = int(np.argmax(occ))
        beta = np.full((D['FR'].shape[0], len(labels)), np.nan)
        beta[:, ref] = 0.0
        for j, lab in enumerate(labels):
            name = f'ev_{e}:{lab}'
            if name in D['names']:
                beta[:, j] = B[D['names'].index(name)] * to_hz
        with np.errstate(invalid='ignore', divide='ignore'):
            raw = (D['FR'] @ M) / occ[None, :] * to_hz
        out[e] = dict(labels=labels, edges_s=list(ek.EDGES_S[e]), ref=ref, beta_hz=beta.astype(np.float32),
                      raw_hz=raw.astype(np.float32), occupancy_s=occ * ek.BLOCK * ek.BIN_S)
    return out


def compact_reset(res: dict) -> dict:
    """Per unit: entry and day cpd beyond each other, the held-out RSS of each model alone."""
    if 'skipped' in res:
        return dict(skipped=res['skipped'])
    return dict(cpd_entry=res['cpd_cv']['ev_entry'], cpd_day=res['cpd_cv']['day'],
                rss_full=res['rss_full'], rss_entry_only=res['rss_reduced']['day'],
                rss_day_only=res['rss_reduced']['ev_entry'], tss=res['tss'], r2_cv=res['r2_cv'],
                entry_beats_day=res['entry_beats_day'], n_folds=res['n_folds'])


def fit_recday(dataset, recday, anchor, *, kin, min_trials, n_perm, seed, synthetic=None) -> dict:
    """Stage 1 (+ the reset test for anchor `entry`) on one recday; the shard's `data`."""
    t0 = time.time()
    rec = ek.load_recday(recday, dataset)
    ek.add_day_clock(rec)
    sessions = ek.analysis_sessions(rec, min_trials)
    truth = None
    if synthetic:
        sessions, truth = syn.make_synthetic(rec, sessions, syn.unit_specs(dataset, synthetic == 'quick'), seed=seed)
        units = synthetic_units(truth, recday)
    else:
        units = unit_table(rec)
    data = dict(recday=recday, dataset=dataset, anchor=anchor, units=units, truth=truth, skipped=None,
                sessions=session_table(sessions), host=socket.gethostname(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if dataset == 'lec':
        data['full_sessions_table'] = os.path.basename(
            ek._latest(os.path.join(ek.LEC_FULL, f'full_sessions_{recday}*.csv')) or '')
    if len(sessions) < 2:
        data['skipped'] = f'{len(sessions)} analysis session(s) at min_trials {min_trials}: no held-out fit'
        data['elapsed_s'] = time.time() - t0
        return data
    # No tracked position anywhere (PFC me08_12092021_13092021, xy_ok False in all 7 sessions): the
    # speed deciles have no edges, and without the kinematic nuisance speed leaks into event blocks (M1 G5).
    if not any(np.isfinite(ek.kinematics(s['xy'])[0]).any() for s in sessions):
        data['skipped'] = 'no tracked position in any analysis session: no kinematic nuisance possible'
        data['elapsed_s'] = time.time() - t0
        return data
    events = (anchor, 'A', 'reward')
    D = ek.build_design(sessions, events, kin=kin)
    data['stage1'] = ek.stage1(sessions, events, n_perm=n_perm, seed=seed, design=D)
    data['rank'] = rank_check(D)
    data['kernels'] = kernels(D, events, kin, unidentified=data['rank']['all_null_columns'])
    if anchor == 'entry':
        data['reset'] = compact_reset(ek.reset_test(rec, sessions, events, kin=kin))
    data['elapsed_s'] = time.time() - t0
    return data


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def long_table(d: dict) -> pd.DataFrame:
    """One row per unit x tested block (groups and joints) for one fitted shard."""
    res, u = d['stage1'], d['units']
    keep = [c for c in ('recday', 'mouse', 'unit', 'group', 'acronym', 'y_um', 'shank', 'near_boundary',
                        'rate_hz', 'elgaby_drop') if c in u.columns]
    frames = []
    for g in res['cpd_cv']:
        f = u[keep].copy()
        f['anchor'], f['block'] = d['anchor'], g
        f['cpd'] = res['cpd_cv'][g]
        f['delta_r2'] = res['delta_r2_cv'][g]
        f['p'] = res['p_freedman_lane__cpd'][g] if 'p_freedman_lane__cpd' in res else np.nan
        f['locked'] = syn.locked(res, g) if 'p_freedman_lane__cpd' in res else False
        f['r2_cv'] = res['r2_cv']
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def reset_table(d: dict) -> pd.DataFrame | None:
    r = d.get('reset')
    if r is None or 'skipped' in r:
        return None
    f = d['units'][['recday', 'mouse', 'unit', 'group']].copy()
    for k in ('cpd_entry', 'cpd_day', 'rss_full', 'rss_entry_only', 'rss_day_only', 'tss', 'r2_cv',
              'entry_beats_day'):
        f[k] = r[k]
    return f


def merge_dir(shard_dir: str, expect: int = N_RECDAYS) -> dict:
    """Merge every shard of one arm: refusals as `load_shards`, plus `expect` recdays per anchor
    (skipped shards count; 0 disables)."""
    cfg, shards = load_shards(shard_dir, key=lambda d: (d['recday'], d['anchor']))
    by_anchor = {a: {d['recday']: d for d in shards if d['anchor'] == a} for a in ANCHORS}
    for a, m in by_anchor.items():
        if m and expect and len(m) != expect:
            raise ValueError(f'anchor {a}: {len(m)} recdays, expected {expect}; pass --expect-recdays 0 to merge anyway')
    fitted = [d for d in shards if not d['skipped']]
    table = pd.concat([long_table(d) for d in fitted], ignore_index=True) if fitted else pd.DataFrame()
    resets = [t for t in (reset_table(d) for d in fitted) if t is not None]
    summary = pd.DataFrame([dict(
        recday=d['recday'], anchor=d['anchor'], n_units=len(d['units']), n_sessions=len(d['sessions']),
        skipped=d['skipped'], fold_rank_min=d['stage1']['fold_rank_min'] if not d['skipped'] else np.nan,
        n_cols=d['stage1']['n_cols'] if not d['skipped'] else np.nan,
        n_rows=d['stage1']['n_rows'] if not d['skipped'] else np.nan,
        null_columns=' '.join(d['rank']['fold_null_columns']) if 'rank' in d else None,
        rank_benign=d['rank']['fold_deficiency_benign'] if 'rank' in d else None,
        r2_cv_median=float(np.nanmedian(d['stage1']['r2_cv'])) if not d['skipped'] else np.nan,
        reset=(None if d['anchor'] != 'entry' or d['skipped'] else d['reset'].get('skipped', 'ok')),
        elapsed_s=d.get('elapsed_s'), slurm_job_id=d.get('slurm_job_id')) for d in shards])
    return dict(config=cfg, shards=by_anchor, table=table,
                reset=pd.concat(resets, ignore_index=True) if resets else pd.DataFrame(),
                summary=summary.sort_values(['anchor', 'recday']).reset_index(drop=True))


def write_merged(M: dict, arm: str) -> str:
    p = versioned(os.path.join(STAGE1_DIR, f'stage1_{arm}.pkl'))
    with open(p, 'wb') as fh:
        pickle.dump(M, fh, protocol=pickle.HIGHEST_PROTOCOL)
    stem = p[:-len('.pkl')]
    M['table'].to_csv(stem + '_units.csv', index=False)
    M['reset'].to_csv(stem + '_reset.csv', index=False)
    M['summary'].to_csv(stem + '_summary.csv', index=False)
    return p


# ---------------------------------------------------------------------------
# runner gate (planted units through this exact path)
# ---------------------------------------------------------------------------

def _latest_pickle(stem: str) -> str:
    hits = [p for p in glob.glob(stem + '*.pkl') if re.fullmatch(re.escape(stem) + r'(\.v\d+)?\.pkl', p)]
    if not hits:
        raise FileNotFoundError(stem + '.pkl')
    return max(hits, key=lambda p: int(m.group(1)) if (m := re.search(r'\.v(\d+)\.pkl$', p)) else 0)


def gate(dataset: str, kin: str, seed: int = 0) -> pd.DataFrame:
    """The shard + merge path reproduces the M1 quick gate pickle exactly; the merge refuses a
    changed stamp and a duplicate recday; a 1-session recday writes `skipped` and is counted."""
    rd = syn.DEFAULT_RECDAY[dataset]
    ref_path = _latest_pickle(os.path.join(syn.OUT_DIR, f'synthetics_{dataset}_{rd}_quick_kin-{kin}'))
    with open(ref_path, 'rb') as fh:
        ref = pickle.load(fh)
    n_perm = int(ref['n_perm'])
    root = os.path.join(STAGE1_DIR, 'runner_gate', f"{dataset}_{time.strftime('%Y%m%d_%H%M%S')}")
    arm = arm_name(dataset, kin, ek.MIN_TRIALS, n_perm, synthetic='quick')
    st = stamp(dataset, kin, ek.MIN_TRIALS, n_perm, seed, synthetic='quick')
    adir = os.path.join(root, arm)
    rows = []

    def add(check, value, ok):
        rows.append(dict(check=check, value=value, passed=bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {check}: {value}", flush=True)

    print(f'runner gate {dataset} {rd} vs {os.path.basename(ref_path)} (n_perm {n_perm}, kin {kin})', flush=True)
    data = fit_recday(dataset, rd, 'entry', kin=kin, min_trials=ek.MIN_TRIALS, n_perm=n_perm, seed=seed,
                      synthetic='quick')
    write_atomic(os.path.join(adir, f'{rd}_entry.pkl'), {'config': st, 'data': data})
    M = merge_dir(adir, expect=1)

    add('planted units identical to the gate pickle', f"{len(data['truth'])} units",
        data['truth'].equals(ref['truth']))
    s_ref, s_new = ref['stage1'], data['stage1']
    add('same blocks tested', sorted(s_new['cpd_cv']), sorted(s_new['cpd_cv']) == sorted(s_ref['cpd_cv']))
    for key in ('cpd_cv', 'p_freedman_lane__cpd'):
        d = max(float(np.max(np.abs(np.nan_to_num(s_ref[key][g]) - np.nan_to_num(s_new[key][g])))) for g in s_ref[key])
        add(f'shard stage1 {key}: max |runner - gate|', d, d == 0.0)
    T = M['table']
    same = all(np.array_equal(T[T['block'] == g].sort_values('unit')['locked'].to_numpy(), syn.locked(s_ref, g))
               for g in s_ref['cpd_cv'])
    add('merged table locked == gate locked, every block', len(T), same)
    t_ref = syn.summarise(ref['truth'], s_ref, 'bank').set_index('kind').drop(columns=['arm', 'n'])
    t_new = T.pivot_table(index='group', columns='block', values='locked', aggfunc='mean')
    t_new = t_new.reindex(index=t_ref.index, columns=t_ref.columns)
    d = float(np.nanmax(np.abs(t_ref.to_numpy(float) - t_new.to_numpy(float))))
    add('merged fraction locked per (kind, block) == gate table (G1-G4 inputs)', d, d == 0.0)
    if 'skipped' not in ref['reset']:
        r_new = M['reset'].sort_values('unit')['entry_beats_day'].to_numpy()
        add('reset test entry_beats_day == gate', int(r_new.sum()), np.array_equal(r_new, ref['reset']['entry_beats_day']))

    # negative arms: each must fire
    neg = os.path.join(root, 'negative_stamp')
    os.makedirs(neg)
    shutil.copy(os.path.join(adir, f'{rd}_entry.pkl'), neg)
    write_atomic(os.path.join(neg, 'other_entry.pkl'),
                 {'config': {**st, 'n_perm': n_perm + 1}, 'data': {**data, 'recday': 'other'}})
    try:
        merge_dir(neg, expect=0)
        add('merge refuses a shard with one stamp field changed', 'merged', False)
    except ValueError as exc:
        add('merge refuses a shard with one stamp field changed', str(exc)[:60], 'different stamp' in str(exc))
    dup = os.path.join(root, 'negative_duplicate')
    os.makedirs(dup)
    shutil.copy(os.path.join(adir, f'{rd}_entry.pkl'), dup)
    shutil.copy(os.path.join(adir, f'{rd}_entry.pkl'), os.path.join(dup, 'copy_entry.pkl'))
    try:
        merge_dir(dup, expect=0)
        add('merge refuses a duplicate recday', 'merged', False)
    except ValueError as exc:
        add('merge refuses a duplicate recday', str(exc)[:60], 'appears in' in str(exc))
    skp = os.path.join(root, 'skip_path')
    one = fit_recday(dataset, rd, 'entry', kin=kin, min_trials=10 ** 6, n_perm=n_perm, seed=seed)
    write_atomic(os.path.join(skp, f'{rd}_entry.pkl'), {'config': st, 'data': one})
    Ms = merge_dir(skp, expect=1)
    add('a recday without 2 sessions writes `skipped` and the merge counts it', one['skipped'],
        bool(one['skipped']) and len(Ms['summary']) == 1 and Ms['table'].empty)
    try:
        merge_dir(skp, expect=2)
        add('merge refuses a short recday count', 'merged', False)
    except ValueError as exc:
        add('merge refuses a short recday count', str(exc)[:60], 'expected 2' in str(exc))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(root, 'runner_gate.csv'), index=False)
    print(f'-> {root}', flush=True)
    return out


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=('lec', 'pfc'), default='pfc' if IS_PFC else 'lec')
    ap.add_argument('--recday')
    ap.add_argument('--anchor', choices=ANCHORS)
    ap.add_argument('--kin', choices=('full', 'strict'), default='strict', help='M2 decision 2026-10-12: strict')
    ap.add_argument('--min-trials', type=int, default=ek.MIN_TRIALS)
    ap.add_argument('--n-perm', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--synthetic', choices=('quick', 'full'), help='planted units on the recday instead of spikes')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--expect-recdays', type=int, default=N_RECDAYS)
    ap.add_argument('--gate', action='store_true', help='runner gate on planted units (exit 0 = every check passes)')
    args = ap.parse_args(argv)

    if args.gate:
        out = gate(args.dataset, args.kin, args.seed)
        return 0 if out['passed'].all() else 1
    arm = arm_name(args.dataset, args.kin, args.min_trials, args.n_perm, args.synthetic)
    if args.merge:
        M = merge_dir(os.path.join(STAGE1_DIR, 'shards', arm), expect=args.expect_recdays)
        p = write_merged(M, arm)
        s = M['summary']
        print(s.to_string(index=False))
        print(f"{len(s)} shards ({int(s['skipped'].notna().sum())} skipped) -> {p}")
        return 0
    if not (args.recday and args.anchor):
        ap.error('--recday and --anchor are required for a fit')
    path = os.path.join(STAGE1_DIR, 'shards', arm, f'{args.recday}_{args.anchor}.pkl')
    st = stamp(args.dataset, args.kin, args.min_trials, args.n_perm, args.seed, args.synthetic)
    if check_or_skip(path, st):
        print(f'{path} already holds this stamp; nothing to do')
        return 0
    d = fit_recday(args.dataset, args.recday, args.anchor, kin=args.kin, min_trials=args.min_trials,
                   n_perm=args.n_perm, seed=args.seed, synthetic=args.synthetic)
    write_atomic(path, {'config': st, 'data': d})
    if d['skipped']:
        print(f"{args.recday} {args.anchor}: skipped ({d['skipped']}) -> {path}")
        return 0
    res = d['stage1']
    locked = {g: f'{np.mean(syn.locked(res, g)):.2f}' for g in res['cpd_cv']}
    rk = d['rank']
    print(f"{args.recday} {args.anchor}: {len(d['sessions'])} sessions, {len(d['units'])} units, "
          f"{res['n_rows']} rows x {res['n_cols']} cols, min fold rank {res['fold_rank_min']} "
          f"(null space {rk['fold_null_columns'] or 'none'}; only duplicate/zero columns: {rk['fold_deficiency_benign']}), "
          f"median r2_cv {np.nanmedian(res['r2_cv']):.4f}, {d['elapsed_s']:.0f} s")
    print(f'  fraction locked: {locked}')
    if 'reset' in d:
        r = d['reset']
        print('  reset:', r['skipped'] if 'skipped' in r else f"entry beats day in {np.mean(r['entry_beats_day']):.2f}")
    print(f'-> {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
