"""Rest-entry kernels, intrinsic timescales and rest drift, one recday per job (M4 part 1).

    python code/run_rest_kernels.py --dataset lec --recday ah10_20250616_20250617 --min-trials 5
    python code/run_rest_kernels.py --dataset lec --merge --min-trials 5
    python code/run_rest_kernels.py --dataset lec --gate [--quick]       # planted units through this path

Per recday: the valid rest blocks (`rest_kernels.rest_blocks`: >= 2 min, truncated at signal
loss) and the task sessions with >= min_trials trials (the awake reference).
  proxy     `rest_kernels.sws_proxy` with the recday's real task calibration, and `proxy_power`:
            how well `down_frac` ranks planted SWS at the real unit count and rates (reported)
  stage 1   `ek.stage1` on `build_rest_design`, once per unit half (even / odd), each half's
            counts divided by the other half's relative population rate (`relative_pop_rate`):
            leave one rest block out, Freedman-Lane, groups `ev_rest_entry` and `state`
            (`down_frac` deciles); the reset test against a day clock (`rest_reset_test`). Needs
            >= 10 units per half at >= 0.3 Hz
  stage 2   `ek.stage2_unit(D, u, 'rest_entry')` for the rest-entry-locked units and 10 null units
  timescales Kanter's slow ACF and Murray's fast tau per unit, in 5-min windows of low-DOWN rest,
            high-DOWN rest and task (length-matched)
  drift     Kanter's 60 s segments: distance growth and within-segment epoch decoding, low-DOWN
            rest, high-DOWN rest and task, on M2's unit sets
  windows   per 10 s window: `down_frac` and the population rate, for the descriptive figures
2026-10-15 revision (user decisions): continuous state terms only, and a population-rate term;
the gate scores planted data with the real calibration. 2026-10-16: the population rate divides,
as a median. Reasons: `rest_kernels` docstring.
Shards: `data/event_kernels/rest/shards/<arm>/<recday>.pkl`, stamped as `run_event_kernels`.

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import event_kernels_synthetics as syn                                 # noqa: E402
import population_drift as pdft                                        # noqa: E402
import rest_kernels as rk                                              # noqa: E402
import rest_kernels_synthetics as rks                                  # noqa: E402
import run_event_kernels as rek                                        # noqa: E402
import run_population_drift as rpd                                     # noqa: E402
import unit_timescales as ut                                           # noqa: E402

IS_PFC = 'mFC_data' in HERE
REST_DIR = os.path.join(ek.REPO, 'data', 'event_kernels', 'rest')
TESTED = ('ev_rest_entry', 'state')
N_NULL = 10
MIN_BLOCKS = 3
MIN_UNITS_HALF = 10                            # the other half's rate must be a population, not a few units
PARTS = ('stage1', 'stage2', 'timescales', 'drift')
USABLE_AUC, USABLE_RECALL, USABLE_QUIET = 0.95, 0.8, 0.05     # AUC decides `proxy_usable`; recall and quiet wake are reported


def arm_name(dataset, min_trials, n_perm, synthetic=None) -> str:
    a = f'{dataset}_mt{min_trials}_np{n_perm}'
    return f'synthetic-{synthetic}_{a}' if synthetic else a


def stamp(dataset, min_trials, n_perm, seed, glm_alpha, synthetic=None) -> dict:
    return dict(runner='run_rest_kernels', dataset=dataset, min_trials=int(min_trials), n_perm=int(n_perm),
                seed=int(seed), glm_alpha=float(glm_alpha), synthetic=synthetic,
                proxy=dict(win_bins=rk.WIN_BINS, down_min_bins=rk.DOWN_MIN_BINS, theta_pct=rk.THETA_PCT,
                           sws_pct=rk.SWS_PCT, sws_min_win=rk.SWS_MIN_WIN, high_down=rk.HIGH_DOWN),
                state=dict(terms=('down_frac deciles',), divisor='other half relative population rate, median',
                           halves='even/odd', pop_span_win=rk.POP_SPAN_WIN, pop_min_count=rk.POP_MIN_COUNT,
                           pop_floor=rk.POP_FLOOR, min_units_half=MIN_UNITS_HALF), segs_per_run=rk.SEGS_PER_RUN,
                qc=dict(loss_win_bins=rk.LOSS_WIN_BINS, loss_frac=rk.LOSS_FRAC, loss_min_keep_s=rk.LOSS_MIN_KEEP_S,
                        min_rest_s=rk.MIN_REST_S),
                usable=dict(auc=USABLE_AUC, recall=USABLE_RECALL, quiet=USABLE_QUIET), ts_win_s=rk.TS_WIN_S, seg=(rk.SEG_S, rk.SEG_EDGE_S),
                max_segments=rk.MAX_SEGMENTS, edges_rest_entry=list(ek.EDGES_S['rest_entry']), n_null=N_NULL,
                sha1_rest_kernels=rek._sha1(rk.__file__), sha1_synthetics=rek._sha1(rks.__file__),
                sha1_event_kernels=rek._sha1(ek.__file__), sha1_unit_timescales=rek._sha1(ut.__file__),
                sha1_population_drift=rek._sha1(pdft.__file__), sha1_run_population_drift=rek._sha1(rpd.__file__),
                sha1_runner=rek._sha1(os.path.abspath(__file__)), inputs=rek.input_versions(dataset))


# ---------------------------------------------------------------------------
# one recday
# ---------------------------------------------------------------------------

def stage1_table(res: dict, unit_ids: np.ndarray) -> pd.DataFrame:
    """Long table per tested group; `unit_ids` maps the fit's rows (one half) back to unit indices."""
    rows = []
    for g in res['cpd_cv']:
        rows.append(pd.DataFrame(dict(unit=np.asarray(unit_ids), group=g, cpd=res['cpd_cv'][g],
                                      p=res['p_freedman_lane__cpd'][g], locked=syn.locked(res, g))))
    return pd.concat(rows, ignore_index=True)


def timescale_table(W: dict, n_units: int) -> pd.DataFrame:
    """Per unit and context: median slow ACF tau and tail over the context's windows, and Murray's tau."""
    rows = []
    for ctx, wins in W.items():
        c05 = [pdft.rebin(c, ut.HALF_S) for c in wins]
        for u in range(n_units):
            a = [ut.acf_tau(c[u]) for c in c05 if c[u].sum() > 0]
            m = rk.murray_tau(wins, u)
            rows.append(dict(unit=u, context=ctx, n_windows=len(wins),
                             acf_tau_s=float(np.nanmedian([x['acf_tau_s'] for x in a])) if a else np.nan,
                             acf_tail=float(np.nanmedian([x['acf_tail'] for x in a])) if a else np.nan,
                             murray_tau_s=m['murray_tau_s'], murray_ok=m['murray_ok'],
                             rate_hz=float(np.mean([c[u].mean() for c in wins]) / ek.BIN_S) if wins else np.nan))
    return pd.DataFrame(rows)


def drift_table(segs: dict, units: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng([seed, 5])
    rows = []
    for name, g, d, idx in rpd.unit_sets(units, seed):
        for ctx, L in segs.items():
            for k, s in enumerate(L):
                r = rk.segment_drift(s[idx], rng)
                rows.append(dict(set=name, group=g, draw=d, n_units=len(idx), context=ctx, segment=k,
                                 growth=r['growth'], acc=r['acc'], chance=r['chance'], dist=r['dist']))
    return pd.DataFrame(rows)


def window_table(blocks: list, P: dict) -> pd.DataFrame:
    """Per 10 s rest window: block, time since rest entry, `down_frac` and the population rate (Hz
    per unit), for the descriptive figures."""
    rows = []
    for b, p in zip(blocks, P['blocks']):
        pr = rk.window_mean(b['counts'].sum(axis=0).astype(float)) / b['counts'].shape[0] / ek.BIN_S
        n = min(len(pr), len(p['down_frac']))
        rows.append(pd.DataFrame(dict(block=b['index'], stage=b.get('stage'), t_s=np.arange(n) * rk.WIN_BINS * ek.BIN_S,
                                      down_frac=p['down_frac'][:n], pop_rate_hz=pr[:n])))
    return pd.concat(rows, ignore_index=True)


def fit_recday(dataset, recday, *, min_trials=5, n_perm=100, seed=0, glm_alpha=1e-4, synthetic=None,
               quick=False, state_terms=True, shared_ramp=False, parts=PARTS) -> dict:
    """Every part in `parts` for one recday; the shard's `data`. `synthetic` plants
    `rest_kernels_synthetics`'s population on the recday's real blocks and task sessions, scored
    with the real calibration; `state_terms=False` and `shared_ramp` are the gate's controls."""
    t0 = time.time()
    rec = ek.load_recday(recday, dataset, with_rest=True)
    ek.add_day_clock(rec)
    sessions = ek.analysis_sessions(rec, min_trials)
    blocks, qc = rk.rest_blocks(rec)
    data = dict(recday=recday, dataset=dataset, skipped=None, stage1_skipped=None, qc=qc, host=socket.gethostname(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), synthetic=synthetic, state_terms=state_terms,
                shared_ramp=shared_ramp, parts=list(parts))
    if not sessions:
        data['skipped'] = f'no task session at min_trials {min_trials}: no awake reference'
    elif len(blocks) < MIN_BLOCKS:
        data['skipped'] = f'{len(blocks)} valid rest block(s)'
    if data['skipped']:
        data['elapsed_s'] = time.time() - t0
        return data
    rates = np.mean([s['counts'].mean(axis=1) for s in sessions], axis=0) / ek.BIN_S
    cal = rk.calibrate([s['counts'] for s in sessions])          # the real awake data, also under `synthetic`
    data['power'] = rks.proxy_power(rates, cal['theta'], cal['cut'], seed=seed)
    if synthetic:
        blocks, sessions, truth, sleep = rks.make_population(blocks, sessions, rates, quick=quick, seed=seed,
                                                             shared_ramp=shared_ramp)
        data['truth'], data['sleep'] = truth, [dict(onset_s=s['onset_s']) for s in sleep]
        data['sleep_windows'] = [dict(sws=rks.window_truth(s['sws']), quiet=rks.window_truth(s['quiet'])) for s in sleep]
        units = rek.synthetic_units(truth, recday)
    else:
        units = rek.unit_table(rec)
    P = rk.sws_proxy([b['counts'] for b in blocks], cal=cal)
    masks = [rk.state_masks(p['down_frac'], P['cut']) for p in P['blocks']]
    data.update(units=units, proxy_usable=bool(data['power']['auc'] >= USABLE_AUC), theta=P['theta'], sws_cut=P['cut'],
                blocks=pd.DataFrame([dict(index=b['index'], stage=b.get('stage'), n_bins=b['counts'].shape[1],
                                          hours_since_day_start=b.get('hours_since_day_start', np.nan),
                                          low_frac=float(lo.mean()), high_frac=float(hi.mean()),
                                          first_high_s=float(np.argmax(hi) * rk.WIN_BINS * ek.BIN_S) if hi.any() else np.nan,
                                          down_frac_mean=float(p['down_frac'].mean()))
                                     for b, p, (lo, hi) in zip(blocks, P['blocks'], masks)]),
                proxy_windows=[p['down_frac'].astype(np.float32) for p in P['blocks']],
                windows=window_table(blocks, P))
    hv = rk.halves(len(units))
    pops = [rk.relative_pop_rate(blocks, h) for h in hv]
    data['pop_units'] = [n for _, n in pops]
    feats = [rk.block_features(b, p, [pops[0][0][i], pops[1][0][i]]) for i, (b, p) in enumerate(zip(blocks, P['blocks']))]
    if 'stage1' in parts and min(data['pop_units']) < MIN_UNITS_HALF:
        data['stage1_skipped'] = (f"{len(units)} units, {data['pop_units']} per half at >= {rk.POP_MIN_COUNT:g} spikes "
                                  f"per 10 s: fewer than {MIN_UNITS_HALF} to divide by")
        data['stage1'], data['stage2'], data['reset'] = pd.DataFrame(), pd.DataFrame(), dict(skipped=data['stage1_skipped'])
    elif 'stage1' in parts:
        S1, R, Ds, data['design'] = [], [], [], []
        for h in (0, 1):
            D = rk.build_rest_design(feats, units=hv[h], pop=1 - h, state=state_terms)
            res = ek.stage1(None, ('rest_entry',), n_perm=n_perm, seed=seed, design=D, tested=TESTED)
            S1.append(stage1_table(res, hv[h]))
            data['design'].append(dict(half=h, n_units=len(hv[h]), n_rows=D['n_rows'], n_cols=D['X'].shape[1],
                                       names=D['names'], fold_rank_min=res['fold_rank_min']))
            R.append(rk.rest_reset_test(D, blocks))
            Ds.append(D)
        data['stage1'] = pd.concat(S1, ignore_index=True).sort_values(['group', 'unit'], ignore_index=True)
        data['reset'] = (dict(skipped=R[0]['skipped']) if 'skipped' in R[0] else
                         pd.concat([pd.DataFrame(dict(unit=hv[h], entry_beats_day=R[h]['entry_beats_day'])) for h in (0, 1)],
                                   ignore_index=True).sort_values('unit', ignore_index=True))
        print(f"  stage 1: {Ds[0]['n_rows']} rows x {Ds[0]['X'].shape[1]} cols per half, {time.time() - t0:.0f} s", flush=True)
        if 'stage2' in parts:
            s1 = data['stage1']
            lk = s1[(s1['group'] == 'ev_rest_entry') & s1['locked']]['unit'].to_numpy()
            nl = np.setdiff1d(np.arange(len(units)), lk)
            rng = np.random.default_rng([seed, zlib.crc32(f'{recday}_rest'.encode())])
            null = np.sort(rng.choice(nl, min(N_NULL, len(nl)), replace=False)) if len(nl) else nl
            pos = {int(u): (h, i) for h in (0, 1) for i, u in enumerate(hv[h])}
            rows = []
            for u, role in [(int(u), 'locked') for u in lk] + [(int(u), 'null_sample') for u in null]:
                h, i = pos[u]
                rows.append({**ek.stage2_unit(Ds[h], i, 'rest_entry', glm_alpha=glm_alpha), 'unit': u, 'half': h, 'role': role})
            data['stage2'] = pd.DataFrame(rows)
            print(f"  stage 2: {len(rows)} units, {time.time() - t0:.0f} s", flush=True)
    if 'timescales' in parts:
        W = rk.context_windows(blocks, P, sessions)
        meta = W.pop('meta')
        if synthetic:                    # the gate scores rest clear of the planted quiet wake: its 0.7x step is not the OU
            q = data['sleep_windows']
            per = int(round(rk.TS_WIN_S / ek.BIN_S)) // rk.WIN_BINS
            W['rest_low_clear'] = [x for x, (i, s0) in zip(W['rest_low'], meta['rest_low'])
                                   if not q[i]['quiet'][s0 // rk.WIN_BINS:s0 // rk.WIN_BINS + per].any()]
        data['timescales'] = timescale_table(W, len(units))
        data['n_windows'] = {k: len(v) for k, v in W.items()}
    if 'drift' in parts:
        segs = rk.drift_segments(blocks, P, sessions, np.random.default_rng([seed, 3]))
        data['segment_meta'] = segs.pop('meta')
        data['drift'] = drift_table(segs, units, seed)
        data['n_segments'] = {k: len(v) for k, v in segs.items()}
    data['elapsed_s'] = time.time() - t0
    print(f"  done: windows {data.get('n_windows')}, segments {data.get('n_segments')}, {data['elapsed_s']:.0f} s", flush=True)
    return data


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def merge_dir(shard_dir: str, expect: int) -> dict:
    cfg, shards = rek.load_shards(shard_dir, key=lambda d: d['recday'])
    if expect and len(shards) != expect:
        raise ValueError(f'{len(shards)} shards, expected {expect}; pass --expect-recdays 0 to merge anyway')
    out = {k: [] for k in ('stage1', 'stage2', 'reset', 'timescales', 'drift', 'blocks', 'windows', 'qc')}
    summary, units = [], {}
    for d in shards:
        base = dict(recday=d['recday'], mouse=d['recday'].split('_')[0])
        out['qc'].append(d['qc'].assign(**base))
        summary.append(dict(**base, skipped=d['skipped'], stage1_skipped=d.get('stage1_skipped'),
                            proxy_usable=d.get('proxy_usable'),
                            power_auc=(d.get('power') or {}).get('auc'), power_recall=(d.get('power') or {}).get('sws_recall'),
                            power_quiet=(d.get('power') or {}).get('quiet_called_sws'), sws_cut=d.get('sws_cut'),
                            theta=d.get('theta'), n_blocks=len(d['blocks']) if 'blocks' in d else 0,
                            n_units=len(d['units']) if 'units' in d else 0, elapsed_s=d.get('elapsed_s'),
                            slurm_job_id=d.get('slurm_job_id')))
        if d['skipped']:
            continue
        units[d['recday']] = d['units']
        for k in ('stage1', 'stage2', 'timescales', 'drift', 'blocks', 'windows'):
            out[k].append(d[k].assign(**base))
        if isinstance(d['reset'], pd.DataFrame):
            out['reset'].append(d['reset'].assign(**base))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    M = {k: cat(v) for k, v in out.items()}
    M.update(config=cfg, units=units, summary=pd.DataFrame(summary).sort_values('recday').reset_index(drop=True))
    return M


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def gate(dataset: str, seed: int = 0, min_trials: int = 5, quick: bool = False) -> pd.DataFrame:
    """G1-G8 on planted units through `fit_recday` (rows in `rest_kernels_synthetics`)."""
    rd = syn.DEFAULT_RECDAY[dataset]
    root = os.path.join(REST_DIR, 'runner_gate', f"{dataset}_{'quick' if quick else 'full'}_{time.strftime('%Y%m%d_%H%M%S')}")
    n_perm = 20 if quick else 100
    rows = []

    def add(gid, check, value, ok, must=True):
        rows.append(dict(gate=gid, check=check, value=value, passed=bool(ok) if must else True, must=must))
        print(f"  [{('PASS' if ok else 'FAIL') if must else 'info'}] {gid} {check}: {value}", flush=True)

    print(f'rest gate {dataset} {rd}', flush=True)
    kw = dict(min_trials=min_trials, n_perm=n_perm, seed=seed, synthetic='planted', quick=quick)
    d = fit_recday(dataset, rd, **kw)
    d0 = fit_recday(dataset, rd, **kw, state_terms=False, parts=('stage1',))
    ds = fit_recday(dataset, rd, **kw, shared_ramp=True, parts=('stage1',))
    ds0 = fit_recday(dataset, rd, **kw, shared_ramp=True, state_terms=False, parts=('stage1',))
    for tag, x in (('', d), ('_no_state', d0), ('_shared_ramp', ds), ('_shared_ramp_no_state', ds0)):
        rek.write_atomic(os.path.join(root, f'{rd}{tag}.pkl'), {'config': stamp(dataset, min_trials, n_perm, seed, 1e-4, 'planted'),
                                                                 'data': x})
    T = d['truth']
    a = (np.ceil(0.05 * (n_perm + 1)) - 1) / (n_perm + 1)      # exact null rate of p < 0.05: 5/101, 1/21
    pw = d['power']
    add('G1', 'proxy at the real unit count: SWS-window AUC', f"{pw['auc']:.3f}", pw['auc'] >= USABLE_AUC, must=dataset == 'lec')
    add('G1', 'the same: SWS recall / quiet wake called SWS / units (reported)',
        f"{pw['sws_recall']:.3f} / {pw['quiet_called_sws']:.3f} / {pw['n_units']}", True, must=False)
    sw = np.concatenate([w['sws'] for w in d['sleep_windows']])
    qw = np.concatenate([w['quiet'] for w in d['sleep_windows']])
    df = np.concatenate(d['proxy_windows'])
    n = min(len(sw), len(df))
    sw, qw, df = sw[:n], qw[:n], df[:n]
    from scipy.stats import rankdata
    rr = rankdata(df)
    auc_planted = float((rr[sw].sum() - sw.sum() * (sw.sum() + 1) / 2) / (sw.sum() * (~sw).sum()))
    _, high = rk.state_masks(df, d['sws_cut'])
    add('G1', 'planted population, real calibration: SWS-window AUC / high-DOWN recall / quiet wake high-DOWN',
        f'{auc_planted:.3f} / {high[sw].mean():.3f} / {high[qw].mean() if qw.any() else np.nan:.3f}', True, must=False)
    S1 = d['stage1'][d['stage1']['group'] == 'ev_rest_entry'].merge(T, on='unit')
    S0 = d0['stage1'][d0['stage1']['group'] == 'ev_rest_entry'].merge(T, on='unit')
    v = float(S1.loc[S1['kind'] == 'rest_ramp', 'locked'].mean())
    add('G2', 'rest-entry ramps: locked with the state terms', round(v, 3), v >= 0.8)
    s2 = d['stage2'].merge(T, on='unit')
    s2 = s2[(s2['kind'] == 'rest_ramp') & (s2['role'] == 'locked')]
    w = float(np.mean(np.abs(np.log2(s2['tau_median'] / s2['tau'])) <= 1)) if len(s2) else np.nan
    add('G2', 'rest-entry ramps: Stage 2 tau within 2x', round(w, 3), w >= 0.8)
    g = S1[S1['kind'] == 'sws_gain']['locked']
    p_with = binomtest(int(g.sum()), len(g), a, alternative='greater').pvalue
    g0 = float(S0[S0['kind'] == 'sws_gain']['locked'].mean())
    add('G3', 'SWS-gain units: locked with the state terms (binomial p vs 5/101) / without them',
        f'{int(g.sum())} of {len(g)} (p {p_with:.3f}) / {g0:.3f}', p_with > 0.01 and g0 >= 0.5)
    nz = S1[S1['kind'] == 'noise']['locked']
    p_nz = binomtest(int(nz.sum()), len(nz), a, alternative='greater').pvalue
    add('G4', 'noise: locked with the state terms (binomial p vs 5/101)', f'{int(nz.sum())} of {len(nz)} (p {p_nz:.3f})', p_nz > 0.01)
    nz0 = S0[S0['kind'] == 'noise']['locked']
    add('G4', 'noise: locked without the state terms (reported)', f'{int(nz0.sum())} of {len(nz0)}', True, must=False)
    for tag, x in (('with', ds), ('without', ds0)):
        z = x['stage1'][(x['stage1']['group'] == 'ev_rest_entry')].merge(T, on='unit')
        z = z[z['kind'] == 'noise']['locked']
        add('G4b', f'shared rest-entry ramp in every unit: noise locked {tag} the state terms (reported)',
            f'{int(z.sum())} of {len(z)}', True, must=False)
    if isinstance(d['reset'], pd.DataFrame):
        rs = d['reset'].merge(T, on='unit')
        v = float((~rs.loc[rs['kind'] == 'day_trend', 'entry_beats_day'].astype(bool)).mean())
        add('G5', 'day trends: the day term beats rest entry', round(v, 3), v >= 0.8)
    else:
        add('G5', 'day trends: reset test', d['reset']['skipped'], False)
    ts = d['timescales'].merge(T, on='unit')
    for tau in (2, 30, 120):
        x = ts[(ts['kind'] == 'ou') & (ts['tau'] == tau)].pivot_table(index='unit', columns='context', values='acf_tau_s')
        ratio = float((x['rest_low_clear'] / x['task']).median()) if {'rest_low_clear', 'task'} <= set(x.columns) else np.nan
        add('G6', f"OU tau {tau} s: slow ACF tau, low-DOWN rest clear of quiet wake / task ({d['n_windows'].get('rest_low_clear')} windows)",
            round(ratio, 3), 0.75 <= ratio <= 1.33,
            must=tau >= 30)                  # the 1-90 s fit starts at 1 s: a 2 s OU is Murray's measure, not this one
    for tau, must in ((0.25, True), (2, False)):
        m = ts[(ts['kind'] == 'ou') & (ts['tau'] == tau) & (ts['context'] == 'task')]['murray_tau_s']
        wm = float(np.mean(np.abs(np.log2(m / tau)) <= 1)) if len(m) else np.nan
        add('G6', f'OU tau {tau} s: Murray tau within 2x (task)', f'{wm:.3f} (median {m.median():.3f} s)', wm >= 0.8,
            must=must)
    dr = d['drift']
    # segments clear of the planted quiet-wake dip: its x0.7 step inside a segment adds growth that is not drift
    q = d['sleep_windows']
    clean = {k for k, (i, s0) in enumerate(d['segment_meta']['rest_low'])
             if not q[i]['quiet'][s0 // rk.WIN_BINS:(s0 + int(rk.SEG_S / ek.BIN_S)) // rk.WIN_BINS + 1].any()}
    for kind, lo, hi in (('drift_equal', 0.8, 1.25), ('drift_fast', 1.5, np.inf)):
        x = dr[(dr['set'] == 'region_full') & (dr['group'] == kind)]
        task = x[x['context'] == 'task']['growth'].mean()
        allr = x[x['context'] == 'rest_low']['growth'].mean()
        cl = x[(x['context'] == 'rest_low') & x['segment'].isin(clean)]['growth'].mean()
        add('G7', f'{kind}: drift growth, low-DOWN rest clear of quiet wake ({len(clean)} segments) / task',
            round(float(cl / task), 3), lo <= cl / task <= hi)
        add('G7', f'{kind}: the same over every low-DOWN segment (reported)', round(float(allr / task), 3), True, must=False)
    rec = ek.load_recday(rd, dataset, with_rest=True)
    blocks, _ = rk.rest_blocks(rec)
    c = blocks[0]['counts'].copy()
    k = int(0.7 * c.shape[1])
    c[:, k:] = 0
    cut = rk.signal_loss_cut(c)
    add('G8', 'planted dropout at bin %d: cut at' % k, cut, cut is not None and abs(cut - k) <= rk.LOSS_WIN_BINS)
    out = pd.DataFrame(rows)
    os.makedirs(root, exist_ok=True)
    out.to_csv(os.path.join(root, 'gate.csv'), index=False)
    print(f'-> {root}', flush=True)
    return out


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=('lec', 'pfc'), default='pfc' if IS_PFC else 'lec')
    ap.add_argument('--recday')
    ap.add_argument('--min-trials', type=int, default=5)
    ap.add_argument('--n-perm', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--glm-alpha', type=float, default=1e-4)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--expect-recdays', type=int, default=rek.N_RECDAYS)
    ap.add_argument('--gate', action='store_true', help='planted units through this path (exit 0 = all pass)')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args(argv)
    if args.gate:
        out = gate(args.dataset, args.seed, args.min_trials, args.quick)
        return 0 if out.loc[out['must'], 'passed'].all() else 1
    arm = arm_name(args.dataset, args.min_trials, args.n_perm)
    if args.merge:
        M = merge_dir(os.path.join(REST_DIR, 'shards', arm), expect=args.expect_recdays)
        p = rek.versioned(os.path.join(REST_DIR, f'rest_{arm}.pkl'))
        with open(p, 'wb') as fh:
            pickle.dump(M, fh, protocol=pickle.HIGHEST_PROTOCOL)
        for k in ('stage1', 'stage2', 'reset', 'timescales', 'blocks', 'windows', 'qc', 'summary'):
            M[k].to_csv(f"{p[:-len('.pkl')]}_{k}.csv", index=False)
        M['drift'].drop(columns=['dist']).to_csv(f"{p[:-len('.pkl')]}_drift.csv", index=False)
        print(M['summary'].to_string(index=False))
        print(f"{len(M['summary'])} shards ({int(M['summary']['skipped'].notna().sum())} skipped) -> {p}")
        return 0
    if not args.recday:
        ap.error('--recday is required for a fit')
    path = os.path.join(REST_DIR, 'shards', arm, f'{args.recday}.pkl')
    st = stamp(args.dataset, args.min_trials, args.n_perm, args.seed, args.glm_alpha)
    if rek.check_or_skip(path, st):
        print(f'{path} already holds this stamp; nothing to do')
        return 0
    d = fit_recday(args.dataset, args.recday, min_trials=args.min_trials, n_perm=args.n_perm, seed=args.seed,
                   glm_alpha=args.glm_alpha)
    rek.write_atomic(path, {'config': st, 'data': d})
    if d['skipped']:
        print(f"{args.recday}: skipped ({d['skipped']}) -> {path}")
    else:
        s1 = d['stage1']
        lk = 'not fitted: ' + d['stage1_skipped'] if d['stage1_skipped'] else \
            f"{s1[s1['group'] == 'ev_rest_entry']['locked'].mean():.2f}"
        print(f"{args.recday}: {len(d['blocks'])} blocks, {len(d['units'])} units, proxy usable {d['proxy_usable']} "
              f"(AUC {d['power']['auc']:.3f}); rest-entry locked {lk}; {d['elapsed_s']:.0f} s -> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
