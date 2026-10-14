"""Tsao's stepwise table, Kanter's single-unit timescales and the ramp test, one recday per job
(M3 part 2).

    python code/run_unit_timescales.py --dataset lec --recday ah10_20250616_20250617 --min-trials 5
    python code/run_unit_timescales.py --dataset lec --merge --min-trials 5
    python code/run_unit_timescales.py --dataset lec --gate [--quick]   # planted units through this path

Per recday (ABCD sessions >= min_trials, `kin='strict'`, the Stage 2 design):
  tsao    per unit, `unit_timescales.tsao_unit` on 500 ms core rows (in-sample stepwise GLM)
  kanter  per unit and session on 0.5 s counts from bin 0: `kanter_glm`, `acf_tau`, `log_fano`; the
          autocorrelation and Fano again on held-out Pearson residuals / ratios of the Stage 2 full
          model (`heldout_mu`, every unit)
  ramp    M2's held-out epoch kNN (`run_population_drift.knn_predictions`, M2 settings and draws) on
          raw counts, on counts with each Stage 1 entry-locked unit divided by its held-out
          shape-free entry factor or its held-out slow ramp (tau >= 60, 120 or 240 s), and on the
          locked units alone against size-matched non-locked draws
Shards: `data/event_kernels/timescales/shards/<arm>/<recday>.pkl`, stamped as `run_event_kernels`.
`--merge` refuses mismatched stamps and checks two equivalences: `heldout_mu`'s held-out deviance
against Stage 2's `dev_np`, and (LEC) the raw ramp-test accuracies against the M2 drift arm.

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import population_drift as pdft                                        # noqa: E402
import run_event_kernels as rek                                        # noqa: E402
import run_population_drift as rpd                                     # noqa: E402
import unit_timescales as ut                                           # noqa: E402
import unit_timescales_synthetics as uts                               # noqa: E402

IS_PFC = 'mFC_data' in HERE
PARTS = ('tsao', 'kanter', 'ramp')
TS_DIR = os.path.join(ek.REPO, 'data', 'event_kernels', 'timescales')
MIN_LOCKED = 10                # locked-only vs non-locked kNN needs a population, not a handful
N_LOCKED_DRAWS = 20


def arm_name(dataset, kin, min_trials, n_perm, synthetic=None) -> str:
    return rek.arm_name(dataset, kin, min_trials, n_perm, synthetic)


def stamp(dataset, kin, min_trials, n_perm, seed, glm_alpha, synthetic=None) -> dict:
    s1 = None if synthetic else rek.stage1_units_csv(dataset, kin, min_trials, n_perm)
    return dict(runner='run_unit_timescales', dataset=dataset, kin=kin, min_trials=int(min_trials),
                n_perm=int(n_perm), seed=int(seed), glm_alpha=float(glm_alpha), synthetic=synthetic,
                parts=list(PARTS), half_bins=ut.HALF, tsao_enter=ut.TSAO_ENTER, tsao_remove=ut.TSAO_REMOVE,
                min_place_bins=ut.MIN_PLACE_BINS, acf=(ut.ACF_MAX_LAG_S, ut.ACF_FIT_S, ut.ACF_TAIL_S),
                slow_ramp_floors_s=list(ut.SLOW_RAMP_FLOORS_S), survivor_frac=ut.SURVIVOR_FRAC,
                knn=dict(rpd.KNN), min_locked=MIN_LOCKED, n_locked_draws=N_LOCKED_DRAWS,
                stage1_units=os.path.basename(s1) if s1 else None, stage1_sha1=rek._sha1(s1) if s1 else None,
                sha1_unit_timescales=rek._sha1(ut.__file__), sha1_event_kernels=rek._sha1(ek.__file__),
                sha1_population_drift=rek._sha1(pdft.__file__), sha1_run_event_kernels=rek._sha1(rek.__file__),
                sha1_run_population_drift=rek._sha1(rpd.__file__), sha1_synthetics=rek._sha1(uts.__file__),
                sha1_runner=rek._sha1(os.path.abspath(__file__)), inputs=rek.input_versions(dataset))


# ---------------------------------------------------------------------------
# one recday
# ---------------------------------------------------------------------------

def kanter_tables(sessions: list, D: dict, H: dict, seed: int) -> tuple:
    """Per unit and session on 0.5 s counts (two 250 ms rows): Kanter's GLM, ACF and Fano, raw and
    against the held-out prediction. Returns (table, raw ACF, residual ACF), the curves as
    (units, sessions, lags) float32."""
    nU = D['FR'].shape[0]
    L = int(round(ut.ACF_MAX_LAG_S / ut.HALF_S))
    acf_raw = np.full((nU, len(sessions), L), np.nan, np.float32)
    acf_res = np.full_like(acf_raw, np.nan)
    rows = []
    for i, s in enumerate(sessions):
        m = D['session_ids'] == i
        nb = (int(m.sum()) // 2) * 2
        y05 = D['FR'][:, m][:, :nb].reshape(nU, -1, 2).sum(axis=2)
        t05 = (np.arange(y05.shape[1]) + 0.5) * ut.HALF_S
        tot = y05.sum(axis=0)
        for u in range(nU):
            mu05 = H[u]['mu'][m][:nb].reshape(-1, 2).sum(axis=1)
            g = ut.kanter_glm(y05[u], t05, (tot - y05[u]) / max(nU - 1, 1))
            a = ut.acf_tau(y05[u])
            r = ut.acf_tau((y05[u] - mu05) / np.sqrt(np.maximum(mu05, 1e-12)))
            acf_raw[u, i], acf_res[u, i] = a['acf'], r['acf']
            rows.append(dict(unit=u, session=s['session'], order=i, n_bins=len(t05),
                             rate_hz=float(y05[u].mean() / ut.HALF_S), **g,
                             acf_tau_s=a['acf_tau_s'], acf_ok=a['acf_ok'], acf_tail=a['acf_tail'],
                             res_acf_tau_s=r['acf_tau_s'], res_acf_ok=r['acf_ok'], res_acf_tail=r['acf_tail'],
                             log_fano=pdft.log_fano(y05[u], seed=[seed, u, i]),
                             res_log_fano=pdft.log_fano(y05[u], mu=mu05, seed=[seed, u, i, 1])))
    return pd.DataFrame(rows), acf_raw, acf_res


def _compact(preds: list) -> list:
    return [dict(y=p['y'].astype(np.int8), within=p['within'].astype(np.int8),
                 across=None if p['across'] is None else p['across'].astype(np.int8)) for p in preds]


def ramp_test(sessions: list, D: dict, H: dict, units: pd.DataFrame, locked: np.ndarray, seed: int) -> dict:
    """M2's held-out epoch kNN on raw counts and with the locked units divided by their held-out
    entry factors (set 'all' and M2's 20-unit draws, M2's seeds), and on the locked units alone
    against size-matched non-locked draws. Factors are per 250 ms row; the trailing part of a
    session beyond the last full row is left undivided."""
    counts = [s['counts'].astype(float) for s in sessions]

    def divided(factor):
        out = []
        for i, c in enumerate(counts):
            m = D['session_ids'] == i
            c2 = c.copy()
            nb = int(m.sum())
            for u in locked:
                c2[u, :nb * ek.BLOCK] /= np.repeat(factor(u)[m], ek.BLOCK)
            out.append(c2)
        return out

    inputs = {'raw': counts}
    if len(locked):
        inputs['div_entry'] = divided(lambda u: H[u]['f_entry'])
        for fl in ut.SLOW_RAMP_FLOORS_S:
            inputs[f'div_ramp{int(fl)}'] = divided(lambda u, fl=fl: H[u]['f_ramp'][fl])
    res = []
    for name, g, draw, idx in rpd.unit_sets(units, seed):
        if name not in ('all', 'all_n20'):
            continue
        for inp, C in inputs.items():
            preds, chance_mae = rpd.knn_predictions([c[idx] for c in C], seed=seed, **rpd.KNN)
            res.append(dict(set=name, draw=draw, input=inp, n_units=len(idx), chance_mae=chance_mae,
                            preds=_compact(preds)))
    nU = len(units)
    if len(locked) >= MIN_LOCKED and nU - len(locked) >= len(locked):
        res.append(dict(set='locked_only', draw=0, input='raw', n_units=len(locked), chance_mae=np.nan,
                        preds=_compact(rpd.knn_predictions([c[locked] for c in counts], seed=seed, **rpd.KNN)[0])))
        others = np.setdiff1d(np.arange(nU), locked)
        for dr, j in enumerate(pdft.subsample(len(others), len(locked), N_LOCKED_DRAWS, seed=seed + 101)):
            idx = others[j]
            res.append(dict(set='nonlocked_matched', draw=dr, input='raw', n_units=len(idx), chance_mae=np.nan,
                            preds=_compact(rpd.knn_predictions([c[idx] for c in counts], seed=seed, **rpd.KNN)[0])))
    return dict(knn=res, n_locked=int(len(locked)))


def fit_recday(dataset, recday, *, kin='strict', min_trials=5, n_perm=100, seed=0, glm_alpha=1e-4,
               synthetic=None, parts=PARTS, quick=False) -> dict:
    """Every part for one recday; the shard's `data`. `synthetic` names a planted population of
    `unit_timescales_synthetics` (its truth table is kept and its planted units are the locked set)."""
    t0 = time.time()
    rec = ek.load_recday(recday, dataset)
    ek.add_day_clock(rec)
    sessions = ek.analysis_sessions(rec, min_trials)
    data = dict(recday=recday, dataset=dataset, skipped=None, parts=list(parts), host=socket.gethostname(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), synthetic=synthetic)
    if len(sessions) < 2:
        data['skipped'] = f'{len(sessions)} analysis session(s) at min_trials {min_trials}: no held-out fit'
    elif not any(np.isfinite(ek.kinematics(s['xy'])[0]).any() for s in sessions):
        data['skipped'] = 'no tracked position in any analysis session: no kinematic nuisance possible'
    if data['skipped']:
        data['elapsed_s'] = time.time() - t0
        return data
    if synthetic:
        sessions, truth = uts.make_population(rec, sessions, synthetic, seed=seed, quick=quick)
        data['truth'] = truth
        units = rek.synthetic_units(truth, recday)
        locked = np.sort(uts.locked_units(truth, synthetic))
    else:
        T = pd.read_csv(rek.stage1_units_csv(dataset, kin, min_trials, n_perm))
        T = T[(T['recday'] == recday) & (T['anchor'] == 'entry') & (T['block'] == 'ev_entry')]
        if T.empty:
            data['skipped'] = 'no Stage 1 rows: the recday was skipped in Stage 1'
            data['elapsed_s'] = time.time() - t0
            return data
        units = rek.unit_table(rec)
        locked = np.sort(T.loc[T['locked'].astype(bool), 'unit'].to_numpy(int))
    D = ek.build_design(sessions, rek.STAGE2_EVENTS, kin=kin)
    if not synthetic:
        p1 = os.path.join(rek.STAGE1_DIR, 'shards', rek.arm_name(dataset, kin, min_trials, n_perm), f'{recday}_entry.pkl')
        with open(p1, 'rb') as fh:
            if pickle.load(fh)['data']['stage1']['col_names'] != D['names']:
                raise AssertionError(f'{recday}: the design differs from the Stage 1 design')
    nU = D['FR'].shape[0]
    data.update(units=units, sessions=rek.session_table(sessions), locked_entry=locked)
    if 'tsao' in parts:
        rows = ut.tsao_rows(sessions)
        B = ut.term_blocks(rows)
        data['tsao_terms'] = list(B)
        data['tsao'] = pd.DataFrame([ut.tsao_unit(rows, B, u) for u in range(nU)])
        print(f"  tsao: {nU} units, terms {list(B)}, {time.time() - t0:.0f} s", flush=True)
    if 'kanter' in parts or 'ramp' in parts:
        lk = set(locked.tolist()) if 'ramp' in parts else set()
        todo = range(nU) if 'kanter' in parts else sorted(lk)      # the ramp test alone needs only the divided units
        H = {u: ut.heldout_mu(D, u, glm_alpha, slow_ramp=u in lk) for u in todo}
        data['heldout'] = pd.DataFrame(dict(unit=list(H), dev=[h['dev'] for h in H.values()],
                                            **{f'slow_ramp_tau_{int(fl)}': [h['slow_ramp_tau_median'][fl] for h in H.values()]
                                               for fl in ut.SLOW_RAMP_FLOORS_S}))
        print(f"  held-out fits: {len(H)} units, {time.time() - t0:.0f} s", flush=True)
    if 'kanter' in parts:
        data['kanter'], data['acf_raw'], data['acf_res'] = kanter_tables(sessions, D, H, seed)
        print(f"  kanter: {time.time() - t0:.0f} s", flush=True)
    if 'ramp' in parts:
        data['ramp'] = ramp_test(sessions, D, H, units, locked, seed)
        print(f"  ramp test: {len(locked)} divided units, {time.time() - t0:.0f} s", flush=True)
    data['elapsed_s'] = time.time() - t0
    return data


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def ramp_long(d: dict) -> pd.DataFrame:
    """One row per set, draw, input, session and true epoch: within-session and held-out hits."""
    rows = []
    for r in d['ramp']['knn']:
        for k, p in enumerate(r['preds']):
            y = p['y'].astype(int)
            n_ep = len(np.unique(y))
            for e in np.unique(y):
                m = y == e
                rows.append(dict(set=r['set'], draw=r['draw'], input=r['input'], n_units=r['n_units'], order=k,
                                 epoch=int(e), n_bins=int(m.sum()), chance=1.0 / n_ep,
                                 within=float(np.mean(p['within'][m] == e)),
                                 across=np.nan if p['across'] is None else float(np.mean(p['across'][m] == e))))
    return pd.DataFrame(rows)


def merge_dir(shard_dir: str, expect: int) -> dict:
    cfg, shards = rek.load_shards(shard_dir, key=lambda d: d['recday'])
    if expect and len(shards) != expect:
        raise ValueError(f'{len(shards)} shards, expected {expect}; pass --expect-recdays 0 to merge anyway')
    tsao, kan, held, ramp, acf, units = [], [], [], [], {}, {}
    for d in shards:
        if d['skipped']:
            continue
        rd = d['recday']
        base = dict(recday=rd, mouse=rd.split('_')[0])
        units[rd] = d['units']
        if 'tsao' in d:
            tsao.append(d['tsao'].assign(**base))
        if 'kanter' in d:
            kan.append(d['kanter'].assign(**base))
            acf[rd] = (d['acf_raw'], d['acf_res'])
        if 'heldout' in d:
            held.append(d['heldout'].assign(**base))
        if 'ramp' in d:
            ramp.append(ramp_long(d).assign(**base))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    summary = pd.DataFrame([dict(recday=d['recday'], skipped=d['skipped'],
                                 n_units=len(d['units']) if 'units' in d else 0,
                                 n_locked_entry=len(d.get('locked_entry', [])),
                                 elapsed_s=d.get('elapsed_s'), slurm_job_id=d.get('slurm_job_id')) for d in shards])
    return dict(config=cfg, tsao=cat(tsao), kanter=cat(kan), heldout=cat(held), ramp=cat(ramp), acf=acf,
                units=units, summary=summary.sort_values('recday').reset_index(drop=True))


def equivalence_stage2(held: pd.DataFrame, dataset: str, kin: str, min_trials: int, n_perm: int) -> pd.DataFrame:
    """heldout_mu's summed held-out deviance against Stage 2's dev_np, every shared unit and event:
    the same design, units and glum call, so they must agree to rounding."""
    S2 = pickle.load(open(rek._latest_pickle(os.path.join(rek.STAGE2_DIR, f"stage2_{rek.arm_name(dataset, kin, min_trials, n_perm)}")), 'rb'))['table']
    m = held.merge(S2[['recday', 'unit', 'event', 'dev_np']], on=['recday', 'unit'])
    rel = np.abs(m['dev'] - m['dev_np']) / np.maximum(1.0, np.abs(m['dev_np']))
    return pd.DataFrame([dict(check='heldout dev == Stage 2 dev_np', rows=len(m), units=m[['recday', 'unit']].drop_duplicates().shape[0],
                              max_rel_diff=float(rel.max()) if len(m) else np.nan, passed=bool(len(m) and rel.max() <= 1e-9))])


def equivalence_m2(ramp: pd.DataFrame, dataset: str, min_trials: int) -> pd.DataFrame:
    """The raw input's per-session accuracies on sets 'all' and 'all_n20' against the M2 drift arm at
    the same threshold (same counts, draws and seeds): exact."""
    p = os.path.join(rpd.DRIFT_DIR, f'drift_{dataset}_mt{min_trials}.pkl')
    if not os.path.exists(p):
        return pd.DataFrame([dict(check='raw kNN == M2 drift arm', rows=0, max_abs_diff=np.nan, passed=True,
                                  note=f'no M2 drift arm at min_trials {min_trials}')])
    K = pickle.load(open(p, 'rb'))['knn']
    K = K[K['set'].isin(['all', 'all_n20'])]
    r = ramp[(ramp['input'] == 'raw') & ramp['set'].isin(['all', 'all_n20'])].copy()
    r['w'] = r['within'] * r['n_bins']
    r['a'] = r['across'] * r['n_bins']
    g = r.groupby(['recday', 'set', 'draw', 'order']).agg(w=('w', 'sum'), a=('a', 'sum'), n=('n_bins', 'sum')).reset_index()
    g['within_acc'], g['across_acc'] = g['w'] / g['n'], g['a'] / g['n']
    K = K.assign(order=K.groupby(['recday', 'set', 'draw']).cumcount())
    m = g.merge(K, on=['recday', 'set', 'draw', 'order'], suffixes=('', '_m2'))
    d = max(np.nanmax(np.abs(m['within_acc'] - m['within_acc_m2'])), np.nanmax(np.abs(m['across_acc'] - m['across_acc_m2'])))
    return pd.DataFrame([dict(check='raw kNN == M2 drift arm', rows=len(m), max_abs_diff=float(d),
                              passed=bool(len(m) == len(g) and d <= 1e-12), note=os.path.basename(p))])


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def gate(dataset: str, kin: str = 'strict', seed: int = 0, min_trials: int = 5, quick: bool = False) -> pd.DataFrame:
    """Planted populations on the gate recday through `fit_recday`: rows T1-T8 and K1-K4 ('tk'), K5,
    K5b and K5c (the ramp test's populations). Exit 0 when every must-hold row passes."""
    rd = rek.syn.DEFAULT_RECDAY[dataset]
    tag = 'quick' if quick else 'full'
    root = os.path.join(TS_DIR, 'runner_gate', f"{dataset}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    rows = []
    for which in uts.POPULATIONS:
        parts = ('tsao', 'kanter') if which == 'tk' else ('ramp',)
        print(f'gate {dataset} {rd}: population {which}, parts {parts}', flush=True)
        d = fit_recday(dataset, rd, kin=kin, min_trials=min_trials, n_perm=20, seed=seed, synthetic=which,
                       parts=parts, quick=quick)
        rek.write_atomic(os.path.join(root, f'{rd}_{which}.pkl'),
                         {'config': stamp(dataset, kin, min_trials, 20, seed, 1e-4, which), 'data': d})
        rows += uts.tk_rows(d, d['truth']) if which == 'tk' else uts.ramp_rows(which, d)
        if which == 'tk':
            d['tsao'].merge(d['truth'], on='unit').to_csv(os.path.join(root, 'tk_tsao_units.csv'), index=False)
            d['kanter'].merge(d['truth'], on='unit').to_csv(os.path.join(root, 'tk_kanter_unit_sessions.csv'), index=False)
        else:
            uts.ramp_summary(d).to_csv(os.path.join(root, f'{which}_ramp_summary.csv'))
    out = pd.DataFrame(rows)
    for r in out.itertuples():
        tag = ('PASS' if r.passed else 'FAIL') if r.must else 'info'
        print(f'  [{tag}] {r.gate} {r.check}: {r.value}', flush=True)
    out.to_csv(os.path.join(root, 'gate.csv'), index=False)
    print(f'-> {root}', flush=True)
    return out


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=('lec', 'pfc'), default='pfc' if IS_PFC else 'lec')
    ap.add_argument('--recday')
    ap.add_argument('--kin', choices=('full', 'strict'), default='strict')
    ap.add_argument('--min-trials', type=int, default=5)
    ap.add_argument('--n-perm', type=int, default=100, help='the Stage 1 arm whose entry-locked units are used')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--glm-alpha', type=float, default=1e-4)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--expect-recdays', type=int, default=rek.N_RECDAYS)
    ap.add_argument('--gate', action='store_true', help='planted populations through this path (exit 0 = all pass)')
    ap.add_argument('--quick', action='store_true', help='with --gate: smaller populations, a smoke run')
    args = ap.parse_args(argv)

    if args.gate:
        out = gate(args.dataset, args.kin, args.seed, args.min_trials, args.quick)
        return 0 if out.loc[out['must'], 'passed'].all() else 1
    arm = arm_name(args.dataset, args.kin, args.min_trials, args.n_perm)
    if args.merge:
        M = merge_dir(os.path.join(TS_DIR, 'shards', arm), expect=args.expect_recdays)
        M['equivalence'] = pd.concat([equivalence_stage2(M['heldout'], args.dataset, args.kin, args.min_trials, args.n_perm),
                                      equivalence_m2(M['ramp'], args.dataset, args.min_trials)], ignore_index=True)
        p = rek.versioned(os.path.join(TS_DIR, f'timescales_{arm}.pkl'))
        with open(p, 'wb') as fh:
            pickle.dump(M, fh, protocol=pickle.HIGHEST_PROTOCOL)
        stem = p[:-len('.pkl')]
        for k in ('tsao', 'kanter', 'heldout', 'ramp', 'summary', 'equivalence'):
            M[k].drop(columns=[c for c in ('path',) if c in M[k].columns]).to_csv(f'{stem}_{k}.csv', index=False)
        print(M['equivalence'].to_string(index=False))
        print(f"{len(M['summary'])} shards ({int(M['summary']['skipped'].notna().sum())} skipped) -> {p}")
        return 0 if M['equivalence']['passed'].all() else 1
    if not args.recday:
        ap.error('--recday is required for a fit')
    path = os.path.join(TS_DIR, 'shards', arm, f'{args.recday}.pkl')
    st = stamp(args.dataset, args.kin, args.min_trials, args.n_perm, args.seed, args.glm_alpha)
    if rek.check_or_skip(path, st):
        print(f'{path} already holds this stamp; nothing to do')
        return 0
    d = fit_recday(args.dataset, args.recday, kin=args.kin, min_trials=args.min_trials, n_perm=args.n_perm,
                   seed=args.seed, glm_alpha=args.glm_alpha)
    rek.write_atomic(path, {'config': st, 'data': d})
    if d['skipped']:
        print(f"{args.recday}: skipped ({d['skipped']}) -> {path}")
    else:
        t = d['tsao']
        print(f"{args.recday}: {len(t)} units, {len(d['locked_entry'])} entry-locked; Tsao time cells "
              f"(day or entry time) {t[['has_day_time', 'has_entry_time']].any(axis=1).mean():.2f}; "
              f"{d['elapsed_s']:.0f} s -> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
