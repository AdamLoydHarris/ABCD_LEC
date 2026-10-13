"""Event-kernel Stage 1 and the reset test on real data: one recday per job, a stamped merge (M2).

    python code/run_event_kernels.py --dataset lec --recday ah10_20250616_20250617 --anchor entry
    python code/run_event_kernels.py --dataset lec --merge [--min-trials 10 --n-perm 100]
    python code/run_event_kernels.py --dataset lec --gate          # planted units through this path
    python code/run_event_kernels.py --dataset lec --stage2 --event reward --recday <rd> --min-trials 5
    python code/run_event_kernels.py --dataset lec --stage2 --merge --min-trials 5
    python code/run_event_kernels.py --dataset lec --gate-stage2    # Stage 2 on planted units, exact vs M1
    python code/run_event_kernels.py --compare <old merged.pkl> <new merged.pkl>

Anchor `entry` fits events (entry, A, reward) and the reset test; `first_A` fits (first_A, A,
reward) on the same rows. Shards: `data/event_kernels/stage1/shards/<arm>/<recday>_<anchor>.pkl`
as `{'config': stamp, 'data': ...}`, arm = `<dataset>_kin-<kin>_mt<min_trials>_np<n_perm>`. A
recday with < 2 analysis sessions writes a `skipped` shard. The merge refuses mismatched stamps,
duplicate recdays and a short recday count, and writes a versioned pickle + long CSVs. Betas and
raw lag-bin rates are kept for figures only; the call is Stage 1's (locked = Freedman-Lane
p < 0.05 and held-out cpd > 0, `event_kernels_synthetics.locked`).

Since 2026-10-12 (V3): anchor `first_A` fits `build_design(first_A_pre=False)`, because the
first-A pre column is the A pre column plus `kin_missing:pre_A`, and in M2 that sum let an
arbitrary share of the pre-A offset reach held-out rows on 8 LEC recdays. Stage 2
(`ek.stage2_unit`) runs per recday and event on the Stage 1 passers plus a seeded null sample.

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
import zlib

import numpy as np
import pandas as pd
from scipy.stats import binom

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
STAGE2_DIR = os.path.join(ek.REPO, 'data', 'event_kernels', 'stage2')
STAGE2_EVENTS = ('entry', 'A', 'reward')
N_NULL = 10                    # non-locked units per recday and event: the reference for shape fractions


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
                first_A_anchor_pre=False, inputs=input_versions(dataset))


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


def rank_report(Xtr: np.ndarray, Xte: np.ndarray, names: list, tested: set) -> dict:
    """What a training fold's rank deficit does to its held-out session.

    Columns all-zero in training get beta 0 in every model, so they cannot move a cpd and are set
    aside. In what remains, a deficit reaches held-out cpd only through null-space directions with a
    held-out component (a training identity that fails on held-out rows) touching a tested column.
    Duplicates are matched with a tolerance: one-hot fractions sum to 0.9999999999999999 where an
    indicator is 1.0, which the M2 exact-bytes test flagged as non-duplicates.
    """
    nz = np.flatnonzero(np.abs(Xtr).max(axis=0) > 1e-12)
    out = dict(zero_columns=[names[j] for j in np.setdiff1d(np.arange(Xtr.shape[1]), nz)],
               null_columns=[], leak=0.0, leak_columns=[], tested_leak=False, only_duplicates=True)
    A, B = Xtr[:, nz], Xte[:, nz]
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    N = vt[s <= s.max() * max(A.shape) * np.finfo(float).eps]
    if not len(N):
        return out
    sup = np.flatnonzero(np.abs(N).max(axis=0) > 1e-6)
    out['null_columns'] = [names[nz[j]] for j in sup]
    reps = []
    for j in sup:
        if not any(np.allclose(A[:, j], A[:, r], atol=1e-9) for r in reps):
            reps.append(j)
    out['only_duplicates'] = bool(len(sup) - len(reps) == len(N))
    if len(B):
        P = B @ N.T                                        # held-out rows along each null direction
        scale = max(float(np.abs(B).max()), 1e-12)
        out['leak'] = float(np.abs(P).max() / scale)
        if out['leak'] > 1e-9:
            _, sv, wt = np.linalg.svd(P, full_matrices=False)
            L = wt[sv > 1e-9 * scale] @ N                  # the part of the null space that leaks
            cols = np.flatnonzero(np.abs(L).max(axis=0) > 1e-6)
            out['leak_columns'] = [names[nz[j]] for j in cols]
            out['tested_leak'] = bool(any(nz[j] in tested for j in cols))
    return out


def rank_check(D: dict) -> dict:
    """`rank_report` for every training fold against its held-out session, plus the null-space
    columns of the all-session design (betas not identified; `kernels` sets them NaN)."""
    sids = D['session_ids']
    Xc = ek.center_within_sessions(D['X'], sids)
    tested = {c for g in ek.TESTED if g in D['groups'] for c in D['groups'][g]}
    folds = [{**rank_report(Xc[sids != f], Xc[sids == f], D['names'], tested), 'fold': int(f)}
             for f in np.unique(sids)]
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    N = vt[s <= s.max() * max(Xc.shape) * np.finfo(float).eps]
    all_null = [D['names'][j] for j in np.flatnonzero(np.abs(N).max(axis=0) > 1e-6)] if len(N) else []
    union = lambda k: sorted(set().union(*[r[k] for r in folds]))
    return dict(folds=folds, fold_null_columns=union('null_columns'), fold_zero_columns=union('zero_columns'),
                leak_columns=union('leak_columns'), max_leak=max(r['leak'] for r in folds),
                tested_leak=any(r['tested_leak'] for r in folds),
                only_duplicates=all(r['only_duplicates'] for r in folds), all_null_columns=all_null)


def kernels(D: dict, events, kin: str, unidentified=(), first_A_pre: bool = True) -> dict:
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
        if e == 'first_A' and not first_A_pre:
            M[:, -1] = 0.0
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


def fit_recday(dataset, recday, anchor, *, kin, min_trials, n_perm, seed, synthetic=None,
               first_A_pre=None) -> dict:
    """Stage 1 (+ the reset test for anchor `entry`) on one recday; the shard's `data`.
    `first_A_pre` None means the V3 rule: dropped for anchor `first_A` (the gate passes True)."""
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
    fa_pre = (anchor != 'first_A') if first_A_pre is None else bool(first_A_pre)
    data['first_A_pre'] = fa_pre
    D = ek.build_design(sessions, events, kin=kin, first_A_pre=fa_pre)
    data['stage1'] = ek.stage1(sessions, events, n_perm=n_perm, seed=seed, design=D)
    data['rank'] = rank_check(D)
    data['kernels'] = kernels(D, events, kin, unidentified=data['rank']['all_null_columns'], first_A_pre=fa_pre)
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


def _rank_ok(d: dict):
    """V3 shards: no fold's null space reaches held-out rows through a tested column. V2 shards
    (M2): the deficit was only exactly duplicated or all-zero columns."""
    r = d.get('rank')
    if r is None:
        return None
    return (not r['tested_leak']) if 'tested_leak' in r else r.get('fold_deficiency_benign')


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
        rank_benign=_rank_ok(d),
        leak_columns=' '.join(d['rank'].get('leak_columns', [])) if 'rank' in d else None,
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

    # The first-A anchor, never gated in M1, now without its pre column (V3).
    fa = fit_recday(dataset, rd, 'first_A', kin=kin, min_trials=ek.MIN_TRIALS, n_perm=n_perm, seed=seed,
                    synthetic='quick')
    s_fa, truth = fa['stage1'], ref['truth']
    kind_units = lambda kind, tau_max=np.inf: truth[(truth['kind'] == kind) &
                                                    (truth['tau'].fillna(0) <= tau_max)]['unit'].to_numpy()
    a0 = syn.alpha_null(n_perm)
    h = syn.locked(s_fa, 'ev_first_A')[kind_units('noise')]
    p = float(binom.sf(h.sum() - 1, len(h), a0))
    add(f'first-A anchor: noise on ev_first_A (binomial p vs {a0:.3f})', f'{h.mean():.2f} (p={p:.2f})', p >= 0.01)
    v = float(syn.locked(s_fa, 'ev_first_A')[kind_units('entry_ramp', 1800)].mean())
    add('first-A anchor: entry ramps tau <= 30 min on ev_first_A', v, v >= 0.8)
    for kind in ('A_ramp', 'reward_ramp', 'reward_bump'):
        u = kind_units(kind)
        hit = syn.locked(s_fa, 'ev_A')[u] | syn.locked(s_fa, 'ev_reward')[u] | syn.locked(s_fa, 'ev_A+reward')[u]
        add(f'first-A anchor: {kind} on A, reward or joint', float(hit.mean()), hit.mean() >= 0.8)
    fp = fit_recday(dataset, rd, 'first_A', kin=kin, min_trials=ek.MIN_TRIALS, n_perm=n_perm, seed=seed,
                    synthetic='quick', first_A_pre=True)['stage1']
    gone = [n for n in fp['col_names'] if n not in s_fa['col_names']]
    add('first-A design drops exactly ev_first_A:pre', gone, gone == ['ev_first_A:pre'])
    # Every block's reduced model keeps a pre-A indicator either way, except the A+reward joint's: with
    # the first-A pre column gone it carries the pre-A offset, as under anchor entry (LEC gate: 0.019).
    dc = {g: float(np.nanmax(np.abs(fp['cpd_cv'][g] - s_fa['cpd_cv'][g]))) for g in s_fa['cpd_cv']}
    same_p = {g: np.array_equal(fp['p_freedman_lane__cpd'][g], s_fa['p_freedman_lane__cpd'][g]) for g in s_fa['cpd_cv']}
    rest = [g for g in s_fa['cpd_cv'] if g != 'ev_A+reward']
    add('first_A_pre False vs True, every block but the A+reward joint: max |cpd diff| <= 1e-9, p identical',
        {g: f'{dc[g]:.0e}' for g in rest}, all(dc[g] <= 1e-9 and same_p[g] for g in rest))
    flips = int(np.sum(syn.locked(fp, 'ev_A+reward') != syn.locked(s_fa, 'ev_A+reward')))
    rows.append(dict(check='first_A_pre: the A+reward joint now carries the pre-A offset (reported)',
                     value=f"max |cpd diff| {dc['ev_A+reward']:.1e}, calls changed {flips}", passed=True))
    print(f"  [info] A+reward joint under first_A_pre False vs True: max |cpd diff| {dc['ev_A+reward']:.1e}, "
          f"calls changed {flips}", flush=True)
    add('first-A fit: no held-out leak through a tested column', fa['rank']['leak_columns'], not fa['rank']['tested_leak'])

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
# Stage 2: kernel shapes and time constants of the Stage 1 passers
# ---------------------------------------------------------------------------

def stage1_units_csv(dataset, kin, min_trials, n_perm) -> str:
    """The unit table of the latest merged Stage 1 pickle of an arm."""
    return _latest_pickle(os.path.join(STAGE1_DIR, f'stage1_{arm_name(dataset, kin, min_trials, n_perm)}'))[:-4] + '_units.csv'


def stage2_stamp(dataset, kin, min_trials, n_perm, seed, glm_alpha, synthetic=None) -> dict:
    s1 = None if synthetic else stage1_units_csv(dataset, kin, min_trials, n_perm)
    return {**stamp(dataset, kin, min_trials, n_perm, seed, synthetic), 'runner': 'run_event_kernels.stage2',
            'stage1_units': os.path.basename(s1) if s1 else None, 'stage1_sha1': _sha1(s1) if s1 else None,
            'glm_alpha': float(glm_alpha), 'n_null': N_NULL, 'events': list(STAGE2_EVENTS)}


def fit_stage2(dataset, recday, event, *, kin, min_trials, n_perm, seed, glm_alpha=1e-4, synthetic=None,
               units=None) -> dict:
    """`ek.stage2_unit` for one recday and event, on the design Stage 1 fitted (anchor `entry`):
    the units locked to `ev_<event>` plus N_NULL seeded non-locked ones, or `units` [(unit, role)]."""
    t0 = time.time()
    rec = ek.load_recday(recday, dataset)
    ek.add_day_clock(rec)
    sessions = ek.analysis_sessions(rec, min_trials)
    data = dict(recday=recday, dataset=dataset, event=event, skipped=None, host=socket.gethostname(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if synthetic:
        sessions, truth = syn.make_synthetic(rec, sessions, syn.unit_specs(dataset, synthetic == 'quick'), seed=seed)
        info = synthetic_units(truth, recday)
    else:
        T = pd.read_csv(stage1_units_csv(dataset, kin, min_trials, n_perm))
        T = T[(T['recday'] == recday) & (T['anchor'] == 'entry') & (T['block'] == f'ev_{event}')]
        if T.empty:
            data['skipped'] = 'no Stage 1 rows: the recday was skipped in Stage 1'
            data['elapsed_s'] = time.time() - t0
            return data
        info = T.drop(columns=['anchor', 'block'])
        if units is None:
            lk, nl = T[T['locked'].astype(bool)], T[~T['locked'].astype(bool)]
            rng = np.random.default_rng([seed, zlib.crc32(f'{recday}_{event}'.encode())])
            null = nl.iloc[np.sort(rng.choice(len(nl), min(N_NULL, len(nl)), replace=False))] if len(nl) else nl
            units = [(int(u), 'locked') for u in lk['unit']] + [(int(u), 'null_sample') for u in null['unit']]
    D = ek.build_design(sessions, STAGE2_EVENTS, kin=kin)
    if not synthetic:
        p1 = os.path.join(STAGE1_DIR, 'shards', arm_name(dataset, kin, min_trials, n_perm), f'{recday}_entry.pkl')
        with open(p1, 'rb') as fh:
            names1 = pickle.load(fh)['data']['stage1']['col_names']
        if names1 != D['names']:
            raise AssertionError(f'{recday}: the Stage 2 design differs from the Stage 1 design; passers would not match')
    rows = [{**ek.stage2_unit(D, u, event, glm_alpha=glm_alpha), 'role': role} for u, role in units]
    t = pd.DataFrame(rows)
    if len(t):
        cols = [c for c in info.columns if c not in t.columns or c == 'unit']
        t = t.merge(info[cols].drop_duplicates('unit'), on='unit', how='left')
    data['rows'] = t
    data['elapsed_s'] = time.time() - t0
    return data


def merge_stage2_dir(shard_dir: str, expect: int) -> dict:
    """Stage 2 shards (recday x event) into one long table; refusals as `load_shards`."""
    cfg, shards = load_shards(shard_dir, key=lambda d: (d['recday'], d['event']))
    if expect and len(shards) != expect:
        raise ValueError(f'{len(shards)} shards, expected {expect}; pass --expect-recdays 0 to merge anyway')
    rows = [d['rows'].assign(recday=d['recday'], event=d['event']) for d in shards
            if not d['skipped'] and len(d['rows'])]
    summary = pd.DataFrame([dict(recday=d['recday'], event=d['event'], skipped=d['skipped'],
                                 n_locked=int((d['rows']['role'] == 'locked').sum()) if not d['skipped'] and len(d['rows']) else 0,
                                 n_null=int((d['rows']['role'] == 'null_sample').sum()) if not d['skipped'] and len(d['rows']) else 0,
                                 elapsed_s=d.get('elapsed_s'), slurm_job_id=d.get('slurm_job_id')) for d in shards])
    return dict(config=cfg, table=pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
                summary=summary.sort_values(['event', 'recday']).reset_index(drop=True))


def write_merged_stage2(M: dict, arm: str) -> str:
    p = versioned(os.path.join(STAGE2_DIR, f'stage2_{arm}.pkl'))
    with open(p, 'wb') as fh:
        pickle.dump(M, fh, protocol=pickle.HIGHEST_PROTOCOL)
    M['table'].to_csv(p[:-4] + '_units.csv', index=False)
    M['summary'].to_csv(p[:-4] + '_summary.csv', index=False)
    return p


STAGE2_COLS = ('dev_null', 'dev_np', 'dev_ramp', 'dev_bump', 'gain_np', 'frac_ramp', 'frac_bump', 'tau_median',
               'tau_log_iqr', 'ramp_weight_median', 'ramp_sign_consistency', 'peak_median', 'k_median')


def gate_stage2(dataset: str, kin: str, seed: int = 0) -> pd.DataFrame:
    """Planted units through the Stage 2 shard + merge path reproduce the M1 quick pickle's
    `stage2` table exactly (same units, events and design as `event_kernels_synthetics.run`)."""
    rd = syn.DEFAULT_RECDAY[dataset]
    ref_path = _latest_pickle(os.path.join(syn.OUT_DIR, f'synthetics_{dataset}_{rd}_quick_kin-{kin}'))
    with open(ref_path, 'rb') as fh:
        ref = pickle.load(fh)
    n_perm = int(ref['n_perm'])
    truth = ref['truth']
    want = truth[truth['kind'].isin(['entry_ramp', 'A_ramp', 'reward_ramp', 'reward_bump',
                                     'reward_fixed_width', 'reward_superposition'])].groupby('kind').head(3)
    ev_of = {'entry_ramp': 'entry', 'A_ramp': 'A'}
    root = os.path.join(STAGE2_DIR, 'runner_gate', f"{dataset}_{time.strftime('%Y%m%d_%H%M%S')}")
    st = stage2_stamp(dataset, kin, ek.MIN_TRIALS, n_perm, seed, 1e-4, synthetic='quick')
    print(f'stage 2 gate {dataset} {rd} vs {os.path.basename(ref_path)}: {len(want)} planted units', flush=True)
    for event in STAGE2_EVENTS:
        u = [(int(x), 'planted') for x, k in zip(want['unit'], want['kind']) if ev_of.get(k, 'reward') == event]
        d = fit_stage2(dataset, rd, event, kin=kin, min_trials=ek.MIN_TRIALS, n_perm=n_perm, seed=seed,
                       synthetic='quick', units=u)
        write_atomic(os.path.join(root, f'{rd}_{event}.pkl'), {'config': st, 'data': d})
    M = merge_stage2_dir(root, expect=len(STAGE2_EVENTS))
    a = ref['stage2'].set_index(['unit', 'event']).sort_index()
    b = M['table'].set_index(['unit', 'event']).sort_index()
    rows = [dict(check='same unit-event pairs as M1', value=len(b), passed=a.index.equals(b.index))]
    if rows[0]['passed']:
        # glum's deviance sums differ in the last bits across nodes (M1 ran elsewhere): 1.5e-11 absolute
        # on values ~1e4 in the first pass, so the test is relative, with NaN patterns required equal.
        va = {c: a[c].to_numpy(float) for c in STAGE2_COLS}
        vb = {c: b[c].to_numpy(float) for c in STAGE2_COLS}
        nan_same = all(np.array_equal(np.isnan(va[c]), np.isnan(vb[c])) for c in STAGE2_COLS)
        d = max(float(np.nanmax(np.abs(va[c] - vb[c]) / np.maximum(1.0, np.abs(va[c])))) for c in STAGE2_COLS)
        rows.append(dict(check='every numeric Stage 2 output: max relative |runner - M1| <= 1e-9, NaN patterns equal',
                         value=d, passed=d <= 1e-9 and nan_same))
        same = bool((a['shape'] == b['shape']).all())
        rows.append(dict(check='shape class identical', value=f"{int((a['shape'] == b['shape']).sum())}/{len(a)}", passed=same))
    for r in rows:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']}: {r['value']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(root, 'stage2_gate.csv'), index=False)
    print(f'-> {root}', flush=True)
    return out


def compare_merged(old_path: str, new_path: str) -> pd.DataFrame:
    """Per anchor and block: units in both, max |cpd| and |p| differences, locked calls that
    differ, and the recdays whose cpd changed, between two merged Stage 1 pickles."""
    A, B = [pickle.load(open(p, 'rb'))['table'] for p in (old_path, new_path)]
    key = ['recday', 'unit', 'anchor', 'block']
    m = A[key + ['cpd', 'p', 'locked']].merge(B[key + ['cpd', 'p', 'locked']], on=key, how='outer',
                                               suffixes=('_old', '_new'), indicator=True)
    rows = []
    for (a, b), g in m.groupby(['anchor', 'block']):
        both = g[g['_merge'] == 'both']
        dc = (both['cpd_old'] - both['cpd_new']).abs()
        rows.append(dict(anchor=a, block=b, units=len(both), only_old=int((g['_merge'] == 'left_only').sum()),
                         only_new=int((g['_merge'] == 'right_only').sum()), max_abs_cpd=float(dc.max()),
                         max_abs_p=float((both['p_old'] - both['p_new']).abs().max()),
                         calls_differ=int((both['locked_old'] != both['locked_new']).sum()),
                         recdays_cpd_changed=' '.join(sorted(both.loc[dc > 0, 'recday'].unique()))))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
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
    ap.add_argument('--stage2', action='store_true', help='Stage 2 for --recday and --event (or --merge them)')
    ap.add_argument('--event', choices=STAGE2_EVENTS)
    ap.add_argument('--glm-alpha', type=float, default=1e-4)
    ap.add_argument('--gate-stage2', action='store_true', help='Stage 2 on planted units, exact against M1')
    ap.add_argument('--compare', nargs=2, metavar=('OLD', 'NEW'), help='diff two merged Stage 1 pickles')
    args = ap.parse_args(argv)

    if args.compare:
        compare_merged(*args.compare)
        return 0
    if args.gate:
        out = gate(args.dataset, args.kin, args.seed)
        return 0 if out['passed'].all() else 1
    if args.gate_stage2:
        out = gate_stage2(args.dataset, args.kin, args.seed)
        return 0 if out['passed'].all() else 1
    arm = arm_name(args.dataset, args.kin, args.min_trials, args.n_perm, args.synthetic)
    if args.stage2:
        if args.merge:
            M = merge_stage2_dir(os.path.join(STAGE2_DIR, 'shards', arm), expect=args.expect_recdays * len(STAGE2_EVENTS))
            p = write_merged_stage2(M, arm)
            print(M['summary'].groupby('event')[['n_locked', 'n_null']].sum().to_string())
            print(f"{len(M['summary'])} shards ({int(M['summary']['skipped'].notna().sum())} skipped) -> {p}")
            return 0
        if not (args.recday and args.event):
            ap.error('--recday and --event are required for a Stage 2 fit')
        path = os.path.join(STAGE2_DIR, 'shards', arm, f'{args.recday}_{args.event}.pkl')
        st = stage2_stamp(args.dataset, args.kin, args.min_trials, args.n_perm, args.seed, args.glm_alpha, args.synthetic)
        if check_or_skip(path, st):
            print(f'{path} already holds this stamp; nothing to do')
            return 0
        d = fit_stage2(args.dataset, args.recday, args.event, kin=args.kin, min_trials=args.min_trials,
                       n_perm=args.n_perm, seed=args.seed, glm_alpha=args.glm_alpha, synthetic=args.synthetic)
        write_atomic(path, {'config': st, 'data': d})
        if d['skipped']:
            print(f"{args.recday} {args.event}: skipped ({d['skipped']}) -> {path}")
        else:
            t = d['rows']
            print(f"{args.recday} {args.event}: {int((t['role'] == 'locked').sum())} locked + "
                  f"{int((t['role'] == 'null_sample').sum())} null units, {d['elapsed_s']:.0f} s; "
                  f"classes of locked units {t[t['role'] == 'locked']['shape'].value_counts().to_dict()} -> {path}")
        return 0
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
          f"(null space {rk['fold_null_columns'] or 'none'}; held-out leak through a tested column: {rk['tested_leak']}), "
          f"median r2_cv {np.nanmedian(res['r2_cv']):.4f}, {d['elapsed_s']:.0f} s")
    print(f'  fraction locked: {locked}')
    if 'reset' in d:
        r = d['reset']
        print('  reset:', r['skipped'] if 'skipped' in r else f"entry beats day in {np.mean(r['entry_beats_day']):.2f}")
    print(f'-> {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
