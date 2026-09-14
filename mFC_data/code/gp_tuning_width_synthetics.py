"""Synthetic controls for W5 (`gp_tuning_width.py`), through the REAL pipeline.

Repo practice: a synthetic enters at the same door the data does. Planted Poisson cells are
driven by each cached recday's REAL trial times and go through `build_curves` -> `split_half`
-> `peak_and_width` -> `per_session_peaks` unchanged; planted beta populations go through
`plot_beta_heatmap` unchanged. Byte-identical in `code/` and `mFC_data/code/`; detects its tree.

    python code/gp_tuning_width_synthetics.py              # every cached recday, 3 seeds, sigma 2/3/5 (~10 min)
    python code/gp_tuning_width_synthetics.py --quick      # 3 recdays, 1 seed

| # | control | a failure would mean |
|---|---|---|
| 1 | `_rebin_leg` == `scipy.stats.binned_statistic` (mean) after the `raw_to_norm` repeat rule, legs of 3-1200 samples | the curves are not the legacy per-leg curves |
| 2 | `split_half` halves are disjoint and interleaved on a real cache; an overlapping order raises | peak and width would share legs (the noise coupling the design exists to avoid) |
| 3 | width calibration: planted phase cells of width 0.1-0.3 at peaks 0.1-0.9 come back within +-1 bin at EVERY peak (no peak-dependent bias); planted peaks within +-1 bin; the 0.05 field records the smoothing floor | the width estimator is biased by where the field sits |
| 4 | time-cell null: fixed-latency cells' recovered phase peak rises monotonically with t0 and fixed-lead cells' falls with t1; width is broadest mid-leg (the V); per-state peak SD of time cells >= 3x that of phase cells | the null the real cells are read against is not what the plan says |
| 5 | noise cells: most fail to reproduce their peak; among those that do, rho(peak, width) is not significant | the estimator manufactures a relation from noise |
| 6 | planted phase cells: rho(peak, width) ~ 0 (flat); planted time cells: rho(|peak - 0.5|, width) < 0 (the V) | the two readings of section 6 B4 are not separable |
| 7 | heatmap gate: BOTH a planted bump population and a pure-noise population sort to a clean diagonal (that is the point); only the planted one has structure around the peak -- peak-aligned neighbour mean exceeds its far-bin mean by > 0.4 for the bumps and by < 0.15 for the noise; n_rows == neurons with a non-zero profile | the heatmap is read as evidence |
| 8 | `assemble` refuses a beta table whose neuron count disagrees with the curves | a positional misalignment would pass silently |
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gp_tuning_width as G                                  # noqa: E402

BIN = 1.0 / G.N_BINS


def control_1_rebin():
    rng = np.random.default_rng(0)
    worst = 0.0
    for L in (3, 17, 50, 89, 90, 91, 100, 137, 400, 1199, 1200):
        x = rng.poisson(0.3, size=(4, L)).astype(np.float32)
        mine = G._rebin_leg(x.copy(), G.N_BINS)
        for i in range(4):
            seg = x[i].astype(float)
            if len(seg) < G.N_BINS:
                seg = np.repeat(seg, int(np.ceil(G.N_BINS / max(len(seg), 1))))
            ref = binned_statistic(np.arange(len(seg)), seg, statistic='mean', bins=G.N_BINS)[0]
            worst = max(worst, float(np.nanmax(np.abs(ref - mine[i]))))
    return worst < 1e-5, {'max_abs_diff': worst}


def control_2_split(cache):
    A, B, r, h = G.split_half(cache['curves'])
    ok = h['n_odd'] + h['n_even'] == cache['curves'].shape[1] and abs(h['n_odd'] - h['n_even']) <= 1
    raised = False
    try:
        bad = np.arange(cache['curves'].shape[1]); bad[1] = 0      # leg 0 in both halves
        G.split_half(cache['curves'], order=bad)
    except AssertionError:
        raised = True
    return ok and raised, {'n_odd': h['n_odd'], 'n_even': h['n_even'], 'overlap_raises': raised,
                           'median_splithalf_r': float(np.nanmedian(r))}


def run_sims(caches, seeds, sigmas):
    tabs = []
    for rd, c in caches.items():
        for sd in seeds:
            t = G.simulate_recday(c, seed=sd, sigmas=sigmas)
            tabs.append(t)
    return pd.concat(tabs, ignore_index=True)


def control_3_calibration(sims, sigma=G.SIGMA):
    d = sims[(sims['kind'] == 'phase') & (sims['sigma'] == sigma)]
    nc = G.null_curves(d)
    info, ok = {}, True
    for wd, sub in nc.groupby('param2'):
        err = (sub['width_med'] - wd).abs()
        spread = sub['width_med'].max() - sub['width_med'].min()
        rec = {'recovered_width_by_peak': dict(zip(sub['param'].round(1), sub['width_med'].round(3))),
               'max_abs_err_bins': round(float(err.max() * G.N_BINS), 2),
               'peak_dependence_bins': round(float(spread * G.N_BINS), 2)}
        if wd >= 0.1:
            c_ok = bool(err.max() <= BIN + 1e-9) and bool(spread <= BIN + 1e-9)
            ok &= c_ok; rec['pass'] = c_ok
        else:
            rec['note'] = 'smoothing floor, recorded not gated'
        info[f'planted_width_{wd:g}'] = rec
    pk = d.groupby('param')['peak_phase'].median()
    pk_err = (pk - pk.index.to_numpy()).abs() * G.N_BINS
    ok &= bool(pk_err.max() <= 1.0 + 1e-9)
    info['peak_max_abs_err_bins'] = round(float(pk_err.max()), 2)
    info['frac_not_reproduced_phase_cells'] = round(float(d['peak_not_reproduced'].mean()), 4)
    # sensitivity: the recovered width of the 0.1 field at each sigma
    sens = {}
    for sg, sub in sims[(sims['kind'] == 'phase') & np.isclose(sims['param2'], 0.1)].groupby('sigma'):
        sens[int(sg)] = round(float(sub['width_hm'].median()), 3)
    info['width_0.1_recovered_by_sigma'] = sens
    return ok, info


def control_4_time_null(sims, sigma=G.SIGMA):
    d = sims[sims['sigma'] == sigma]
    nc = G.null_curves(d)
    tfr = nc[nc['kind'] == 'tfr'].sort_values('param'); ttr = nc[nc['kind'] == 'ttr'].sort_values('param')
    # Spearman, not strict monotonicity: beyond ~10 s the field is past the end of most legs, so
    # the recovered phase peak SATURATES near 1 (retrospective) or 0 (prospective) and can dip a
    # bin. That saturation is a property of the leg-duration distribution and is recorded, not
    # a failure -- but the trend over the range must be perfect.
    mono_up = float(spearmanr(tfr['param'], tfr['peak_med']).correlation) > 0.95
    mono_dn = float(spearmanr(ttr['param'], ttr['peak_med']).correlation) < -0.95
    sat_tfr = [float(t) for t, p in zip(tfr['param'], tfr['peak_med']) if p > 0.9]
    sat_ttr = [float(t) for t, p in zip(ttr['param'], ttr['peak_med']) if p < 0.1]
    # the V: width broadest in the middle of the leg, narrow at both ends (both anchors pooled)
    tc = pd.concat([tfr, ttr])
    mid = tc[(tc['peak_med'] > 0.35) & (tc['peak_med'] < 0.65)]['width_med']
    ends = tc[(tc['peak_med'] < 0.2) | (tc['peak_med'] > 0.8)]['width_med']
    v_ok = len(mid) > 0 and len(ends) > 0 and float(mid.median()) > float(ends.median())
    # the per-state discriminator
    st = [k for k in d.columns if k.startswith('peak_state')]
    def circ_sd_states(sub):
        P = sub[st].to_numpy(float)
        th = 2 * np.pi * P
        R = np.abs(np.nanmean(np.exp(1j * th), axis=1))
        return np.sqrt(-2 * np.log(np.clip(R, 1e-12, 1))) / (2 * np.pi)
    sd_t = np.nanmedian(circ_sd_states(d[d['kind'] == 'tfr']))
    sd_p = np.nanmedian(circ_sd_states(d[d['kind'] == 'phase']))
    disc_ok = sd_t >= 3 * sd_p
    ok = mono_up and mono_dn and v_ok and disc_ok
    return ok, {'tfr_peak_by_t0': dict(zip(tfr['param'], tfr['peak_med'].round(3))),
                'tfr_width_by_t0': dict(zip(tfr['param'], tfr['width_med'].round(3))),
                'ttr_peak_by_t1': dict(zip(ttr['param'], ttr['peak_med'].round(3))),
                'ttr_width_by_t1': dict(zip(ttr['param'], ttr['width_med'].round(3))),
                'spearman_ok_tfr_up': mono_up, 'spearman_ok_ttr_down': mono_dn,
                'saturated_latencies_s_tfr': sat_tfr, 'saturated_leads_s_ttr': sat_ttr,
                'width_mid_vs_ends': (round(float(mid.median()), 3) if len(mid) else None,
                                      round(float(ends.median()), 3) if len(ends) else None),
                'per_state_peak_circSD_time_vs_phase': (round(float(sd_t), 4), round(float(sd_p), 4)),
                'frac_not_reproduced_time_cells': round(float(d[d['kind'].isin(['tfr', 'ttr'])]['peak_not_reproduced'].mean()), 3)}


def control_5_noise(sims, sigma=G.SIGMA):
    d = sims[(sims['kind'] == 'noise') & (sims['sigma'] == sigma)]
    frac_nr = float(d['peak_not_reproduced'].mean())
    dd = d.dropna(subset=['width_hm'])
    rho, p = spearmanr(dd['peak_phase'], dd['width_hm']) if len(dd) > 5 else (np.nan, np.nan)
    ok = frac_nr > 0.3 and (not np.isfinite(p) or p > 0.01)
    return ok, {'frac_not_reproduced': round(frac_nr, 3), 'n_defined': int(len(dd)),
                'rho_peak_width': round(float(rho), 3), 'p': round(float(p), 4) if np.isfinite(p) else None,
                'median_width_when_defined': round(float(dd['width_hm'].median()), 3) if len(dd) else None,
                'median_splithalf_r': round(float(d['splithalf_r'].median()), 3)}


def control_6_readings(sims, sigma=G.SIGMA):
    d = sims[sims['sigma'] == sigma].dropna(subset=['width_hm'])
    ph = d[d['kind'] == 'phase']
    tc = d[d['kind'].isin(['tfr', 'ttr'])]
    r_ph, p_ph = spearmanr(ph['peak_phase'], ph['width_hm'])
    r_tv, p_tv = spearmanr(np.abs(tc['peak_phase'] - 0.5), tc['width_hm'])
    r_t, p_t = spearmanr(tc['peak_phase'], tc['width_hm'])
    ok = abs(r_ph) < 0.1 and r_tv < -0.3
    return ok, {'phase_cells_rho_peak_width': round(float(r_ph), 3),
                'time_cells_rho_absoff_width': round(float(r_tv), 3), 'time_cells_rho_peak_width': round(float(r_t), 3),
                'n_phase': int(len(ph)), 'n_time': int(len(tc))}


def control_7_heatmap(seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    col_idx = G.gp_col_idx(G.SELECTING_ARM)
    n_bins = len(col_idx) + 1
    ncol = max(col_idx) + 1
    n = 200
    peaks = rng.integers(0, n_bins, n)
    glm_results, cv = {}, {}
    for rd, kind in (('sim_planted', 'bump'), ('sim_noise', 'noise')):
        per = {}
        for i in range(n):
            if kind == 'bump':
                x = np.arange(n_bins)
                d = np.minimum(np.abs(x - peaks[i]), n_bins - np.abs(x - peaks[i]))
                prof = np.exp(-0.5 * (d / 1.0) ** 2) + rng.normal(0, 0.15, n_bins)
            else:
                prof = rng.normal(0, 1, n_bins)
            prof = prof - prof[0]                     # reference coding: bin 0 is 0 by construction
            params = np.zeros(ncol); params[np.asarray(col_idx)] = prof[1:]
            per[i] = params
        glm_results[rd] = per
        cv[rd] = {'p_freedman_lane__cpd': {'goal_progress': np.full(n, 0.001)},
                  'delta_r2_cv': {'goal_progress': np.zeros(n)}}
    region_of = {'sim_planted': np.array(['planted'] * n), 'sim_noise': np.array(['noise'] * n)}
    fig, info = G.plot_beta_heatmap(glm_results, cv, col_idx, 'goal_progress', only_significant=True,
                                    region_of=region_of, groups=['planted', 'noise'],
                                    colors={'planted': '#2C2C2A', 'noise': '#B4B2A9'})
    plt.close(fig)
    M, labels, _, _, _ = G._profile_rows(glm_results, cv, col_idx, 'goal_progress', only_significant=True,
                                      alpha=0.05, null='freedman_lane', p_stat='cpd', center='reference',
                                      region_of=region_of)
    out, ok = {}, info['n_rows'] == 2 * n
    for g in ('planted', 'noise'):
        rows = M[labels == g]
        order = G.sort_by_peak(rows)
        pk = np.argmax(rows[order], axis=1)
        diagonal = bool(np.all(np.diff(pk) >= 0))          # the sort-induced diagonal, present for BOTH
        # peak-aligned mean profile: neighbours of the peak
        al = np.array([np.roll(r, -int(np.argmax(r))) for r in rows])
        neigh = float(np.mean([al[:, 1].mean(), al[:, -1].mean()]))
        far = float(al[:, 3:-2].mean())
        out[g] = {'diagonal_after_sort': diagonal, 'peak_neighbour_mean': round(neigh, 3),
                  'far_bins_mean': round(far, 3), 'neighbour_excess': round(neigh - far, 3),
                  'n': int(len(rows))}
    # BOTH populations must give the diagonal -- that is what makes the sort no evidence. Only
    # the planted one may have structure around the peak.
    ok &= out['planted']['diagonal_after_sort'] and out['noise']['diagonal_after_sort']
    ok &= out['planted']['neighbour_excess'] > 0.4 and out['noise']['neighbour_excess'] < 0.15
    out['n_rows'] = info['n_rows']; out['n_per_block'] = info['n_per_block']
    return ok, out


def control_8_assemble_refuses(cache):
    """A beta table one neuron short must be refused by the positional join."""
    rd = cache['recday']
    n = cache['curves'].shape[0]
    col_idx = G.gp_col_idx(G.SELECTING_ARM); ncol = max(col_idx) + 1
    def fake(nn, with_tfr):
        glm_results = {rd: {i: np.zeros(ncol) for i in range(nn)}}
        cv = {rd: {'p_freedman_lane__cpd': {'goal_progress': np.ones(nn)},
                   'cpd_cv': {'goal_progress': np.zeros(nn)},
                   'delta_r2_cv': {'goal_progress': np.zeros(nn), **({'time_from_reward': np.zeros(nn)} if with_tfr else {})}}}
        return {'glm_results': glm_results, 'cv_results': cv}
    raised = False
    try:
        G.assemble({rd: cache}, fake(n - 1, True), fake(n, False))
    except AssertionError:
        raised = True
    ok_full = len(G.assemble({rd: cache}, fake(n, True), fake(n, False))) == n
    return raised and ok_full, {'short_table_raises': raised, 'matched_table_rows': n if ok_full else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--max-recdays', type=int, default=None)
    args = ap.parse_args(argv)
    seeds = [0] if args.quick else list(range(args.seeds))
    sigmas = (G.SIGMA,) + tuple(G.SIGMA_SENSITIVITY)
    t0 = time.time()
    caches = G.load_all_caches()
    if not caches:
        print(f'no caches under {G.CACHE_DIR}; run gp_tuning_width.py --build first'); return 1
    rds = sorted(caches)
    if args.quick:
        rds = rds[:3]
    if args.max_recdays:
        rds = rds[:args.max_recdays]
    caches = {rd: caches[rd] for rd in rds}
    results = {}

    def record(name, fn, *a, **k):
        t = time.time()
        try:
            ok, info = fn(*a, **k)
        except Exception as exc:                               # noqa: BLE001
            import traceback
            ok, info = False, {'error': f'{type(exc).__name__}: {exc}', 'tb': traceback.format_exc()[-1500:]}
        results[name] = {'pass': bool(ok), 'info': info, 'elapsed_s': round(time.time() - t, 1)}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({results[name]['elapsed_s']} s)")
        for kk, vv in info.items():
            print(f'        {kk}: {vv}')

    print(f'W5 synthetic controls -- {G.DATASET} tree, {len(caches)} recdays, seeds {seeds}, sigmas {sigmas}')
    first = caches[rds[0]]
    record('1  rebin == binned_statistic', control_1_rebin)
    record('2  split halves disjoint + interleaved', control_2_split, first)
    print(f'simulating {len(caches)} recdays x {len(seeds)} seeds ...', flush=True)
    sims = run_sims(caches, seeds, sigmas)
    print(f'  {len(sims)} planted-cell rows ({time.time() - t0:.0f} s)')
    record('3  width / peak calibration (phase cells)', control_3_calibration, sims)
    record('4  time-cell null: the V and the per-state shift', control_4_time_null, sims)
    record('5  noise cells', control_5_noise, sims)
    record('6  readings separable (phase flat, time V)', control_6_readings, sims)
    record('7  heatmap gate: planted diagonal vs noise', control_7_heatmap)
    record('8  assemble refuses a misaligned table', control_8_assemble_refuses, first)

    n_fail = sum(not r['pass'] for r in results.values())
    print('=' * 78)
    print(f"{len(results) - n_fail}/{len(results)} controls passed  ({(time.time() - t0) / 60:.1f} min)")
    null = {sg: G.null_curves(sims[sims['sigma'] == sg]) for sg in sigmas}
    G._ensure_writable_dir(os.path.dirname(G.SYNTH_OUT))
    with open(G.SYNTH_OUT, 'wb') as fh:
        pickle.dump({'dataset': G.DATASET, 'recdays': rds, 'seeds': seeds, 'sigmas': sigmas,
                     'results': results, 'sims': sims, 'null_curves': null,
                     'built': time.strftime('%Y-%m-%d %H:%M:%S')}, fh, protocol=4)
    print(f'saved {G.SYNTH_OUT}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
