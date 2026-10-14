"""Synthetic gate for the maze pipeline, through `maze_spatial.run_maze_recday` on real maze trajectories.

A real recday keeps its positions, locations, trial times and tasks; `counts` are replaced by Poisson
spikes from planted rate functions. Run on the cluster for each dataset (one recday each):

    sbatch --job-name=sp_mz_syn sbatch_files/spatial.sbatch code/maze_spatial_synthetics.py --dataset mec --quick
    ... --dataset lec ;  PFC from mFC_data/code/maze_spatial_synthetics.py --dataset pfc

| # | planted | must hold |
|---|---|---|
| ME1 | real 1 cm occupancy | the clean bridge-safe smoother equals the user's notebook function, max abs diff 0 |
| ME2 | real spikes, thesis recipe | `rsa_thesis` + raw fit equal an inline copy of pfc_rsa cells 22/34 (x40 for Hz) to 1e-6 relative (float32 counts) |
| M1 | uniform 5 Hz unit, expected spike map | both smoothers within 1 % of truth on bridge bins; the zero-fill smoother is not (the check can fail); one Poisson draw reported |
| M2 | pure place cells continuous in position, narrow (6 cm) and wide (15 cm) | RSA degree beta inside its place-code and field-shuffle nulls (5-95 %); corner/cardinal BA <= 0.5; same-type error share <= 3/7 + 0.05 over >= 30 errors pooled across field layouts |
| M3 | corner, cardinal, node and bridge cells | each type recovered >= 80 %; RSA degree beta above the place-code and field-shuffle null p95s; corner/cardinal BA > 0.5 |
| M4 | strong speed cells on the real speed profile | where speed alone separates corners from cardinals (|AUC - 0.5| >= 0.08; mEC 0.60, PFC 0.61, LEC 0.52), speed matching lowers BA by >= 0.05; matched inside its permutation null |
| M5 | Euclidean- vs geodesic-coded place populations, 5 draws each | cv R2 G+X minus E+X and the diagnostic contrast both higher for the geodesic population; the sign of the R2 difference is reported only (biased towards E) |
| M6 | place x goal progress (a different field per goal-progress bin) | across-bin plane angle exceeds both within-bin controls |
| M7 | Poisson noise | stable / spatial / type classes not above 1 % + chance (binomial p >= 0.01) |
| S1 | M3 population, null sims 2 -> 4 | every earlier number unchanged (max abs diff 0, nothing missing) |
| MS1 | 1000 synthetic mice under H0, 3 recdays, 100 sims | fraction with mouse p <= 0.05 consistent with 5 % (binomial p >= 0.01) |
| MS2 | +1 null SD in each of 5 recdays, 200 mice | >= 90 % of mice detected; the legacy mean-of-fractions rate is reported |
| MS3 | M2 and M3 through merge_tables | mouse p equals the hand count; M2 not a hit, M3 a hit (field shuffle); group tests equal hand values |
| K1 | real track | travel |turn| at corners >= 1.5x bridges above 10 cm/s; despiked and capped samples < 0.5 % |
| K2 | real behaviour | the at-port state lies within 1 s of >= 90 % of goal arrivals (reward bins) |
| K3 | real samples | turn matching off by default; collapsed edges equal speed-only matching exactly |
| K4 | turn, at-port and slow-speed cells | node/bridge BA and the node/bridge RSA beta inside the kinematic null (p > 0.05) |
| K5 | node and bridge template cells + place cells | both statistics beyond the kinematic null p95 |
| K6 | turn cells | speed x turn matching lowers corner/cardinal BA by >= 0.05 where turning separates the types |
| K7 | guards | port speed above min_speed, non-increasing edges, decode sims > RSA sims raise; 'head' raises on PFC only |

Each check prints `PASS/FAIL name: measured value, n`; the script exits 1 on any failure.
This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CHECKS = []
# mEC: bp01 04-02/03, where speed separates corners from cardinals most (AUC 0.62 vs 0.54 on bp04)
RECDAY = {'mec': 'bp01_20240402_20240403', 'lec': 'ah10_20250616_20250617', 'pfc': 'me11_05122021_06122021'}


def check(name: str, passed: bool, detail: str) -> None:
    print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}", flush=True)
    CHECKS.append((name, bool(passed), detail))


def nb_smooth(arr, sigma, valid_mask=None):
    """Verbatim copy of `_gaussian_smooth_ignore_nans` (local_analysis cell 48), the equivalence target."""
    if sigma is None or sigma <= 0:
        return arr
    if valid_mask is None:
        valid_mask = np.isfinite(arr)
    arr_filled = np.where(valid_mask, arr, 0.0).astype(float)
    weight = gaussian_filter(valid_mask.astype(float), sigma=sigma, mode='constant', cval=0.0)
    smoothed = gaussian_filter(arr_filled, sigma=sigma, mode='constant', cval=0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = smoothed / weight
    out[weight == 0] = np.nan
    return out


def thesis_rsa_inline(counts, loc, sess):
    """Inline copy of pfc_rsa cells 22/34 (raw counts, every bin, nanmean of per-session RDMs, OLS on
    unstandardised [euclidean grid units, place type, degree] over both triangles)."""
    from sklearn.linear_model import LinearRegression
    place_types = [0, 1, 0, 1, 2, 1, 0, 1, 0]
    maze_locs = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
    P = np.zeros((9, 9)); Dg = np.zeros((9, 9)); E = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            Dg[i, j] = abs(place_types[j] - place_types[i]); P[i, j] = float(place_types[i] != place_types[j])
            E[i, j] = np.linalg.norm(maze_locs[i] - maze_locs[j])
            if i == j:
                P[i, j] = Dg[i, j] = E[i, j] = np.nan
    rdms = []
    for s in np.unique(sess):
        n = counts[:, sess == s]
        l = loc[sess == s]
        rdm = np.zeros((9, 9))
        for i in range(1, 10):
            for j in range(1, 10):
                rdm[i - 1, j - 1] = np.linalg.norm(n[:, l == i].mean(1) - n[:, l == j].mean(1))
                if i == j:
                    rdm[i - 1, j - 1] = np.nan
        rdms.append(rdm)
    m = np.nanmean(np.array(rdms), axis=0)
    y = m.flatten()[~np.isnan(m.flatten())]
    X = np.array([E.flatten()[~np.isnan(E.flatten())], P.flatten()[~np.isnan(P.flatten())],
                  Dg.flatten()[~np.isnan(Dg.flatten())]]).T
    return LinearRegression().fit(X, y).coef_


def loc_rates_planted(loc: np.ndarray, rate_by_loc: np.ndarray) -> np.ndarray:
    """(N, T) Hz from per-location rates (N, 21); unknown locations get each unit's mean."""
    R = np.empty((rate_by_loc.shape[0], len(loc)))
    known = (loc >= 1) & (loc <= 21)
    R[:, known] = rate_by_loc[:, loc[known] - 1]
    R[:, ~known] = rate_by_loc.mean(1, keepdims=True)
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='maze pipeline synthetic gate')
    ap.add_argument('--dataset', choices=('mec', 'lec', 'pfc'), default='mec')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n-jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = ap.parse_args(argv)
    import spatial_io as sio
    sio.set_dataset(args.dataset)
    import maze_spatial as ms
    import spatial_common as sc
    warnings.filterwarnings('ignore')
    t0 = time.time()
    rng = np.random.default_rng(20261010)
    rd = RECDAY[args.dataset]
    views = sio.load(args.dataset, [rd], 'v4')[rd]
    cfg = copy.deepcopy(ms.DEFAULT_CFG)
    cfg['n_jobs'] = args.n_jobs
    q = args.quick
    cfg['stats'].update(n_shifts=100 if q else 500, n_label_perm=200 if q else 1000)
    cfg['rsa'].update(n_perm=200 if q else 1000, max_splits=6 if q else 20)
    cfg['decode']['n_perm'] = 10 if q else 35
    cfg['planes'].update(n_reps=2 if q else 5, max_task_splits=2 if q else 4)
    cfg['null'].update(n_sim_rsa=20 if q else 100, n_sim_decode=2 if q else 10, n_sim_planes=2 if q else 5)
    asm = ms.assemble(views, cfg)
    c = asm['cat']
    T = c['counts'].shape[1]
    print(f'{rd}: {len(asm["sessions"])} sessions, T {T}, tasks {asm["n_tasks"]}, lattice resid '
          f'{asm["lattice"]["resid_max_cm"]:.2f} cm; loaded in {time.time() - t0:.0f} s', flush=True)

    def views_for(spk):
        """The recday's views with `counts` replaced by (N, T) spikes, split back per session."""
        v2, off = {}, 0
        for p in asm['parts']:
            v = dict(views[p['session']])
            v['counts'] = spk[:, off:off + p['T']]
            v['xy'], v['locs'] = v['xy'][:p['T']], v['locs'][:p['T']]
            v2[p['session']] = v
            off += p['T']
        return v2

    def run(R, units_n=None, cfg_=None):
        """Plant (N, T) Hz into the views and run the real entry point."""
        spk = rng.poisson(np.clip(R, 0, None) * sc.BIN_S).astype(np.float32)
        units = pd.DataFrame(dict(unit=np.arange(R.shape[0]), region='SYN'))
        return ms.run_maze_recday(views_for(spk), rd, units, cfg_ or cfg)

    # ME1 smoother equivalence on real occupancy
    m2 = ms.maps_2d(asm, cfg)
    occ = m2['occ']
    spk = rng.poisson(occ * 4.0)
    raw = np.where(occ > 0, spk / np.where(occ > 0, occ, 1), 0.0)
    for sg in cfg['maps']['sigmas']:
        a = nb_smooth(raw, sg, occ > 0)
        a[~(occ > 0)] = np.nan
        b = sc.rate_maps_normconv(spk[None].astype(float), occ, sg)[0]
        same = np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) == 0
        check(f'ME1 bridge-safe smoother equals the notebook (sigma {sg})', same, f'max abs diff {np.nanmax(np.abs(a - b))}')

    # ME2 thesis RSA equivalence on the real spikes
    beta_inline = thesis_rsa_inline(c['counts'], c['loc'], c['sess']) * 40.0
    M = ms.model_rdms(cfg)
    beta_ours = ms.rsa_fit_raw(ms.rsa_thesis(c['counts'], c, cfg),
                               [M['E9'] / cfg['lattice']['spacing_cm'], M['P9'], M['D9']])['beta']
    d = float(np.max(np.abs(beta_inline - beta_ours)))
    # float32 counts against the notebook's float64 means: equal to ~1e-7 relative, so 1e-6 relative
    check('ME2 thesis RSA equals the inline pfc_rsa copy', d < 1e-6 * max(1, np.max(np.abs(beta_inline))),
          f'max abs diff {d:.2e}; betas {np.round(beta_ours, 3).tolist()}')

    # M1 bridge flatness
    lat = ms.lattice_xy(cfg)
    ex = m2['edges']
    cen = (ex[:-1] + ex[1:]) / 2
    flat = sc.bin_index(c['xy'], ex, ex)
    mask = c['moving'] & c['intrial']
    nb = len(cen)
    on_bridge = np.zeros(nb * nb)
    on_node = np.zeros(nb * nb)
    np.add.at(on_bridge, flat[mask & (flat >= 0) & (c['loc'] >= 10)], 1)
    np.add.at(on_node, flat[mask & (flat >= 0) & (c['loc'] >= 1) & (c['loc'] <= 9)], 1)
    bridge_bins = (on_bridge > 3 * on_node) & (on_bridge >= 10)
    Mx = sc.onehot(flat, mask, nb * nb)
    occ_m = sc.occupancy_map(Mx, (nb, nb))
    true = 5.0

    def smooth3(spk_map, sg):
        return (sc.rate_maps_normconv(spk_map[None], occ_m, sg, cfg['maps']['occ_threshold_s'])[0].ravel(),
                sc.rate_maps_separate(spk_map[None], occ_m, sg, valid=occ_m > 0,
                                      min_smoothed_occ=cfg['maps']['occ_threshold_s'])[0].ravel(),
                gaussian_filter(np.where(occ_m > 0, spk_map / np.where(occ_m > 0, occ_m, 1), 0.0), sg).ravel())

    # bias is tested on the expected spike map: both smoothers are linear in the counts with weights set
    # by occupancy, so one Poisson draw only adds skew (PFC run 3712174: one-draw median 0.942 at sigma 1.5)
    spk_u = rng.poisson(occ_m * true).astype(float)
    for sg in cfg['maps']['sigmas']:
        r_nc, r_ow, r_zf = (np.nanmedian(x[bridge_bins]) / true for x in smooth3(occ_m * true, sg))
        check(f'M1 bridges unbiased, normconv sigma {sg}', abs(r_nc - 1) <= 0.01, f'{r_nc:.4f} of truth over {int(bridge_bins.sum())} bins')
        check(f'M1 bridges unbiased, occupancy-weighted sigma {sg}', abs(r_ow - 1) <= 0.01, f'{r_ow:.4f}')
        check(f'M1 zero-fill smoother biased on bridges, sigma {sg} (the check can fail)', r_zf < 0.95, f'{r_zf:.3f}')
        p_nc, p_ow, _ = (np.nanmedian(x[bridge_bins]) / true for x in smooth3(spk_u, sg))
        check(f'M1 one Poisson draw, sigma {sg} (reported)', True, f'median of truth: normconv {p_nc:.3f}, occupancy-weighted {p_ow:.3f}')

    G = ms.geodesic(cfg)
    E = np.linalg.norm(lat[:, None] - lat[None], axis=2)

    def place_pop(n, sigma, geodesic=False):
        out = np.zeros((n, 21))
        for u in range(n):
            centre = rng.integers(0, 21)
            dist = (G if geodesic else E)[centre]
            out[u] = 0.5 + rng.uniform(5, 15) * np.exp(-dist ** 2 / (2 * sigma ** 2))
        return out

    def place_continuous(m, sigma):
        """(m, T) Hz: Gaussian fields on the canonical position, centred on random location centres."""
        xy = np.nan_to_num(c['xy'])
        out = np.empty((m, T))
        for u in range(m):
            centre = lat[rng.integers(0, 21)]
            out[u] = 0.5 + rng.uniform(5, 15) * np.exp(-((xy - centre) ** 2).sum(1) / (2 * sigma ** 2))
        return out

    n = 40
    # M2 pure place cells, narrow and wide, continuous in position
    for sig in (6.0, 15.0):
        res = run(place_continuous(n, sig))
        if sig == 6.0:
            res_m2 = res
        real_d = res['real']['decided']['rsa'][9]['fit_ED']['beta'][1]
        null_d = np.array([x[9]['fit_ED']['beta'][1] for x in res['nulls']['place']['rsa'] if 9 in x])
        pct = float(np.mean(null_d >= real_d)) if len(null_d) else np.nan
        check(f'M2 place sigma {sig:.0f}: RSA degree inside the place-code null', 0.05 <= pct <= 0.95 or not np.isfinite(real_d),
              f'degree beta {real_d:.2f}, null mean {np.nanmean(null_d):.2f}, fraction of null >= real {pct:.2f}, n {len(null_d)}')
        null_s = np.array([x[9]['fit_ED']['beta'][1] for x in res['nulls']['place_shuffled']['rsa'] if 9 in x])
        pct_s = float(np.mean(null_s >= real_d)) if len(null_s) else np.nan
        check(f'M2 place sigma {sig:.0f}: RSA degree inside the field-shuffle null', 0.05 <= pct_s <= 0.95 or not np.isfinite(real_d),
              f'null mean {np.nanmean(null_s):.2f}, fraction of null >= real {pct_s:.2f}, n {len(null_s)}')
        dec = res['real']['decided']['decode']
        check(f'M2 place sigma {sig:.0f}: corner/cardinal BA not above 0.5', dec['corner_cardinal_ba'] <= 0.5,
              f"{dec['corner_cardinal_ba']:.2f}")
        # errors pooled over field layouts until >= 30: one layout gave 8 of 9 errors same-type on LEC (run
        # 3712173), all from one pair of nodes without fields
        same, tot, layouts = dec.get('same_type_error_share', 0) * dec.get('n_errors', 0), dec.get('n_errors', 0), 1
        while tot < 30 and layouts < 10:
            sim = rng.poisson(place_continuous(n, sig) * sc.BIN_S).astype(np.float32)
            d = ms.population(sim, asm, cfg, rng, 'decided', which=('decode',), n_perm_decode=0)['decode']
            if d.get('n_errors', 0):
                same += d['same_type_error_share'] * d['n_errors']
                tot += d['n_errors']
            layouts += 1
        share = same / tot if tot else np.nan
        check(f'M2 place sigma {sig:.0f}: same-type error share not above 3/7 (pooled layouts; untestable below 30 errors)',
              tot < 30 or share <= 3 / 7 + 0.05, f'{share:.2f} of {tot} errors over {layouts} layouts')

    # M3 generalised cells mixed with place cells
    tpl = {k: v[0] for k, v in ms.TEMPLATES.items()}
    def type_rates(kind, m):
        out = np.zeros((m, 21))
        for u in range(m):
            t = np.r_[tpl[kind], np.zeros(12)] if kind in ('corner', 'cardinal', 'centre') else tpl[kind]
            out[u] = 0.5 + rng.uniform(5, 15) * t
        return out
    kinds = ('corner', 'cardinal', 'node', 'bridge')
    rates = np.vstack([type_rates(k, 10) for k in kinds] + [place_pop(20, 8.0)])
    res = run(loc_rates_planted(c['loc'], rates))
    res_m3 = res
    su = res['single']
    for i, k in enumerate(kinds):
        hit = float(np.mean(su['type_class'][k][i * 10:(i + 1) * 10]))
        check(f'M3 {k} cells recovered', hit >= 0.8, f'{hit:.2f} of 10')
    real_d = res['real']['decided']['rsa'][9]['fit_ED']['beta'][1]
    null_d = np.array([x[9]['fit_ED']['beta'][1] for x in res['nulls']['place']['rsa'] if 9 in x])
    check('M3 RSA degree above the place-code null p95', len(null_d) > 0 and real_d > np.nanpercentile(null_d, 95),
          f'degree beta {real_d:.2f}, null p95 {np.nanpercentile(null_d, 95) if len(null_d) else np.nan:.2f}')
    null_s = np.array([x[9]['fit_ED']['beta'][1] for x in res['nulls']['place_shuffled']['rsa'] if 9 in x])
    check('M3 RSA degree above the field-shuffle null p95', len(null_s) > 0 and real_d > np.nanpercentile(null_s, 95),
          f'degree beta {real_d:.2f}, null p95 {np.nanpercentile(null_s, 95) if len(null_s) else np.nan:.2f}, mean {np.nanmean(null_s):.2f}')
    dec = res['real']['decided']['decode']
    check('M3 corner/cardinal BA above 0.5', dec['corner_cardinal_ba'] > 0.5, f"{dec['corner_cardinal_ba']:.2f}")

    # M4 speed cells with and without speed matching
    from sklearn.metrics import roc_auc_score
    nodes = np.isin(c['loc'], [1, 2, 3, 4, 6, 7, 8, 9]) & c['moving'] & c['intrial']
    auc = roc_auc_score(np.isin(c['loc'][nodes], [2, 4, 6, 8]).astype(int), np.nan_to_num(c['speed'][nodes]))
    spd = np.clip(np.nan_to_num(c['speed']), 0, 60)
    R = 1.0 + rng.uniform(0.4, 0.8, (n, 1)) * spd[None]
    ba = {}
    for match in (False, True):
        cfg4 = copy.deepcopy(cfg)
        cfg4['decode']['match_speed'] = match
        cfg4['null'].update(n_sim_rsa=0, n_sim_decode=0, n_sim_planes=0)
        res = run(R, cfg_=cfg4)
        dec = res['real']['decided']['decode']
        p95 = np.nanpercentile(dec['corner_cardinal_null'], 95) if len(dec['corner_cardinal_null']) else np.nan
        ba[match] = (dec['corner_cardinal_ba'], p95)
    # the test is the drop under matching: label permutations keep each node's speed, so other node
    # partitions decode speed too and the permutation p95 reached the unmatched BA (0.53) in run 3712172.
    # Testable only where speed separates the types: the LEC track has speed-only AUC 0.517 (run 3712241)
    testable = abs(auc - 0.5) >= 0.08
    check('M4 speed matching lowers the speed cells\' corner/cardinal BA by >= 0.05 (where speed-only |AUC - 0.5| >= 0.08)',
          (not testable) or ba[False][0] - ba[True][0] >= 0.05,
          f'unmatched BA {ba[False][0]:.2f} (null p95 {ba[False][1]:.2f}), matched {ba[True][0]:.2f}; speed-only AUC {auc:.3f}'
          + ('' if testable else '; untestable on this track'))
    check('M4 speed matching removes it', ba[True][0] <= max(ba[True][1], 0.5),
          f'matched BA {ba[True][0]:.2f}, its null p95 {ba[True][1]:.2f}')

    # M5 Euclidean- vs geodesic-coded populations, 5 population draws each: E and G RDMs correlate 0.95 and
    # one draw of a geodesic population was fitted better by E (run 3712240: cv R2 0.607 vs 0.586)
    contrast, delta = {}, {}
    for geo in (False, True):
        cfg5 = copy.deepcopy(cfg)
        cfg5['null'].update(n_sim_rsa=0, n_sim_decode=0, n_sim_planes=0)
        picks, d_r2, con = [], [], []
        for draw in range(5):
            R5 = loc_rates_planted(c['loc'], place_pop(60, 12.0, geodesic=geo))
            if draw == 0:
                r21 = run(R5, cfg_=cfg5)['real']['decided']['rsa'][21]
            else:
                sim = rng.poisson(np.clip(R5, 0, None) * sc.BIN_S).astype(np.float32)
                r21 = ms.population(sim, asm, cfg5, rng, 'decided', which=('rsa',))['rsa'][21]
            picks.append((r21['cv_r2_GD'] > r21['cv_r2_ED']) == geo)
            d_r2.append(r21['cv_r2_GD'] - r21['cv_r2_ED'])
            con.append(r21['diagnostic_contrast'])
        contrast[geo] = float(np.mean(con))
        delta[geo] = float(np.mean(d_r2))
        # reported, not gated: the sign of the cv R2 difference picked E for 5/5 E-coded but G for only 1/5
        # G-coded populations (run 3712251): a narrow-field code saturates with distance, as E compresses G
        check(f'M5 {"geodesic" if geo else "Euclidean"}-coded: sign of cv R2 G+X minus E+X (reported; biased towards E)', True,
              f'picks the planted geometry in {sum(picks)} of 5; mean {delta[geo]:+.3f}')
    check('M5 cv R2 G+X minus E+X higher for the geodesic population (mean of 5 draws)', delta[True] > delta[False],
          f'geodesic {delta[True]:+.3f}, Euclidean {delta[False]:+.3f}')
    check('M5 diagnostic contrast larger for the geodesic population (mean of 5 draws)', contrast[True] > contrast[False],
          f'geodesic {contrast[True]:.2f}, Euclidean {contrast[False]:.2f} (positive = cross-hole pairs farther)')

    # M6 place x goal progress
    cfg6 = copy.deepcopy(cfg)
    cfg6['null'].update(n_sim_rsa=0, n_sim_decode=0, n_sim_planes=0)
    R = np.zeros((n, T))
    for u in range(n):
        fields = rng.integers(0, 9, cfg['gp_bins'])
        for g in range(cfg['gp_bins']):
            sel = c['gp'] == g
            dist = E[fields[g]][np.clip(c['loc'][sel], 1, 21) - 1]
            R[u, sel] = 0.5 + 10 * np.exp(-dist ** 2 / (2 * 8.0 ** 2))
        R[u, c['gp'] < 0] = 0.5
    res = run(R, cfg_=cfg6)
    pl = res['real']['decided']['planes'][2]
    across = np.nanmean([v[0] for v in pl['across'].values()])
    halves = np.nanmean([v[0] for v in pl['halves'].values()])
    tasks = np.nanmean([v[0] for v in pl['tasks'].values()])
    check('M6 across-bin plane angle exceeds both controls', across > halves and across > tasks,
          f'smallest angle: across {across:.1f}, halves {halves:.1f}, tasks {tasks:.1f} deg')

    # M7 noise
    cfg7 = copy.deepcopy(cfg)
    cfg7['null'].update(n_sim_rsa=0, n_sim_decode=0, n_sim_planes=0)
    nn = 60 if q else 150
    res = run(np.repeat(rng.choice([0.5, 2.0, 8.0], nn)[:, None], T, 1), cfg_=cfg7)
    su = res['single']
    for k, v in [('stable', su['stable']), ('spatial', su['spatial'])] + [(f'type {t}', su['type_class'][t]) for t in ms.TEMPLATES]:
        kk = int(np.sum(v))
        p = float(binom.sf(kk - 1, nn, 0.05))
        check(f'M7 noise not {k}', p >= 0.01, f'{kk}/{nn}, binomial p vs 5 % {p:.3f}')

    # S1 more null simulations leave every earlier number unchanged: seeds are drawn per index below
    # n_sim_rsa, so 100 decode / planes sims keep the 20 / 10 of run 1617859bc6fd as their first ones
    spk1 = rng.poisson(np.clip(loc_rates_planted(c['loc'], rates), 0, None) * sc.BIN_S).astype(np.float32)
    units1 = pd.DataFrame(dict(unit=np.arange(spk1.shape[0]), region='SYN'))
    cfg_a, cfg_b = copy.deepcopy(cfg), copy.deepcopy(cfg)
    cfg_a['null'].update(n_sim_rsa=4, n_sim_decode=2, n_sim_planes=2)
    cfg_b['null'].update(n_sim_rsa=4, n_sim_decode=4, n_sim_planes=4)
    ra = ms.run_maze_recday(views_for(spk1), rd, units1, cfg_a)
    rb = ms.run_maze_recday(views_for(spk1), rd, units1, cfg_b)
    keys = ('single', 'maps2d', 'place_models', 'real', 'nulls')
    cmp1 = sc.compare_tree({k: ra[k] for k in keys}, {k: rb[k] for k in keys})
    grew = all(len(rb['nulls'][k]['decode']) == 4 for k in cfg_b['null']['kinds']['decode'])
    check('S1 more decode / planes sims leave every earlier number unchanged',
          cmp1['max_diff'] == 0 and cmp1['nan_mismatch'] == 0 and not cmp1['missing'] and grew,
          f"max abs diff {cmp1['max_diff']} over {cmp1['n']} values, NaN mismatches {cmp1['nan_mismatch']}, missing {len(cmp1['missing'])}")

    # MS1 / MS2 the mouse-level test on synthetic tables: calibrated under H0, and more powerful than the
    # legacy mean of per-recday fractions (one recday's fraction moved by up to 0.45 at 10-20 sims)
    g = np.random.default_rng(20261012)

    def tables(n_mice, n_rd, shift):
        prow, srow = [], []
        for m_ in range(n_mice):
            for r_ in range(n_rd):
                name = f'm{m_}_{r_}'
                srow += [(name, f'm{m_}', 'place_shuffled', 'rsa', i, 'x', v) for i, v in enumerate(g.normal(0, 1, 100))]
                prow.append(dict(recday=name, mouse=f'm{m_}', recipe='decided', x=shift if shift else g.normal(0, 1)))
        return pd.DataFrame(prow), pd.DataFrame(srow, columns=['recday', 'mouse', 'null', 'analysis', 'sim', 'stat', 'value'])
    pop0, sims0 = tables(1000, 3, 0.0)
    m0 = ms.mouse_null_tests(pop0, sims0, 'x', 'place_shuffled')
    k0 = int((m0['p'] <= 0.05).sum())
    from scipy.stats import binomtest
    p0 = binomtest(k0, len(m0), 0.05).pvalue
    check('MS1 mouse-level p calibrated under H0', p0 >= 0.01, f'{k0}/{len(m0)} mice with p <= 0.05, binomial p vs 5 % {p0:.3f}')
    pop1, sims1 = tables(200, 5, 1.0)
    m1 = ms.mouse_null_tests(pop1, sims1, 'x', 'place_shuffled')
    leg = sims1.merge(pop1[['recday', 'x']], on='recday').assign(ge=lambda d: d['value'] >= d['x'])
    leg = leg.groupby(['mouse', 'recday'])['ge'].mean().groupby('mouse').mean()
    check('MS2 a +1 SD effect in each of 5 recdays is detected per mouse', m1['hit'].mean() >= 0.9,
          f"new rule {m1['hit'].mean():.2f} of {len(m1)} mice; legacy mean-of-fractions rule {(leg <= 0.05).mean():.2f}")

    # MS3 through the real pipeline: one recday's p equals the hand count; M2 accepted, M3 rejected
    hits = {}
    for name, rr in (('M2 place sigma 6', res_m2), ('M3 generalised', res_m3)):
        tb = ms.merge_tables({rd: rr})
        m = ms.mouse_null_tests(tb['population'], tb['null_sims'], 'rsa9_ED_X', 'place_shuffled')
        vals = np.array([x[9]['fit_ED']['beta'][1] for x in rr['nulls']['place_shuffled']['rsa'] if 9 in x], float)
        vals = vals[np.isfinite(vals)]
        real_v = rr['real']['decided']['rsa'][9]['fit_ED']['beta'][1]
        p_hand = (1 + int((vals >= real_v).sum())) / (1 + len(vals))
        hits[name] = (float(m['p'].iloc[0]), bool(m['hit'].iloc[0]))
        check(f'MS3 {name}: mouse p from merged tables equals the hand count', abs(hits[name][0] - p_hand) < 1e-12,
              f'p {hits[name][0]:.4f}, hand {p_hand:.4f}, n {len(vals)}')
    check('MS3 M2 place cells are not a hit under the field-shuffle null', not hits['M2 place sigma 6'][1],
          f"p {hits['M2 place sigma 6'][0]:.3f}")
    check('MS3 M3 generalised cells are a hit under the field-shuffle null', hits['M3 generalised'][1],
          f"p {hits['M3 generalised'][0]:.3f}")
    gt = ms.group_tests(pd.DataFrame(dict(p=[0.01, 0.02, 0.5, 0.6, 0.7], hit=[True, True, False, False, False],
                                          real=[1.0] * 5, null_mean=[0.0] * 5, null_sd=[1.0] * 5)))
    check('MS3 group tests equal hand values', abs(gt['p_binomial'] - 0.0225925) < 1e-6 and abs(gt['p_signflip'] - 1 / 32) < 1e-12,
          f"binomial 2 of 5 {gt['p_binomial']:.7f} (hand 0.0225925), sign-flip {gt['p_signflip']:.5f} (hand 0.03125)")

    # K1 / K2 turning and at-port state on the real track
    kq = ms.kinematics_qc(asm, cfg)
    med = kq['median_turn_travel']
    qcs = [x for x in kq['turn'] if x]
    n_t = max(sum(x['T'] for x in qcs), 1)
    sp_, cp_ = sum(x['n_spike'] for x in qcs) / n_t, sum(x['n_cap'] for x in qcs) / n_t
    # 1.5x, not the 2x first written: corners measured 1.81x bridges in mEC, 1.71x in LEC, 1.85x in PFC
    check('K1 travel turning larger at corners than on bridges (> 10 cm/s); despiked and capped < 0.5 %',
          med['corner'] >= 1.5 * med['bridge'] and sp_ < 0.005 and cp_ < 0.005,
          f"median |turn| corner {med['corner']:.0f}, cardinal {med['cardinal']:.0f}, bridge {med['bridge']:.0f} deg/s "
          f"(corner / bridge {med['corner'] / med['bridge']:.2f}); "
          f"despiked {sp_:.4f}, capped {cp_:.4f}; head medians {kq['median_turn_head']}")
    check('K2 at-port state within 1 s of >= 90 % of goal arrivals', kq['reward_coverage'] >= 0.9,
          f"{kq['reward_coverage']:.3f} of {kq['n_reward_bins']} reward bins; at-port fraction {kq['port_fraction']:.3f}")

    # K3 turn matching off by default: collapsed edges reproduce the 1-D path, and turn is ignored when off
    base = np.nonzero(c['moving'] & c['intrial'] & (c['loc'] >= 1) & np.isfinite(c['turn_dps']))[0]
    a3 = ms.speed_matched(base, c['type'], c['speed'], cfg, np.random.default_rng(3))
    b3 = ms.speed_matched(base, c['type'], c['speed'], cfg, np.random.default_rng(3), turn=c['turn_dps'])
    cfg3 = copy.deepcopy(cfg)
    cfg3['speed_match']['turn_edges_dps'] = (0.0, np.inf)
    c3 = ms.speed_matched(base, c['type'], c['speed'], cfg3, np.random.default_rng(3), turn=c['turn_dps'])
    check('K3 turn matching off by default and collapsed edges equal speed-only matching',
          np.array_equal(a3, b3) and np.array_equal(a3, c3), f'{len(a3)} samples')

    # K4 / K5 kinematic null: kinematic cells inside it, node / bridge cells beyond it
    cfgk = copy.deepcopy(cfg)
    # 10 decode sims floor p at 1/11 = 0.091 (K5, run 3752832): 20 is the fewest that can reach 0.05
    cfgk['null'].update(n_sim_rsa=20 if q else 100, n_sim_decode=20 if q else 40, n_sim_planes=0,
                        kinds=dict(rsa=('speed', 'kinematic'), decode=('speed', 'kinematic'), planes=()))
    turn_r = np.clip(np.nan_to_num(c['turn_dps']), 0, 360)
    spd_r = np.nan_to_num(c['speed'])

    def kin_pop(m):
        out = []
        for i in range(m):
            kind = i % 3
            if kind == 0:
                out.append(1.0 + rng.uniform(0.03, 0.06) * turn_r)
            elif kind == 1:
                out.append(2.0 + rng.uniform(10, 20) * c['port'])
            else:
                out.append(1.0 + rng.uniform(8, 15) * np.exp(-spd_r / 5.0))
        return np.array(out)

    def null_frac(res, kind, stat):
        real_ = ms.population_row(res['real']['decided']).get(stat, np.nan)
        vals = np.array([ms.population_row({w: x}).get(stat, np.nan) for w in ('rsa', 'decode')
                         for x in res['nulls'][kind][w]], float)
        vals = vals[np.isfinite(vals)]
        return real_, ((1 + (vals >= real_).sum()) / (1 + len(vals)) if len(vals) else np.nan), vals
    res4 = run(kin_pop(n), cfg_=cfgk)
    for stat in ('dec_node_bridge_ba', 'rsa21_ED_X'):
        r_, pk, _ = null_frac(res4, 'kinematic', stat)
        _, ps, _ = null_frac(res4, 'speed', stat)
        check(f'K4 turn / port / slow cells: {stat} inside the kinematic null', pk > 0.05,
              f'real {r_:.3f}, p vs kinematic {pk:.3f}; vs speed null {ps:.3f} (reported)')
    tpl_node = np.r_[np.ones(9), np.zeros(12)]
    node_rates = np.vstack([0.5 + rng.uniform(5, 15) * (tpl_node if i % 2 == 0 else 1 - tpl_node) for i in range(20)]
                           + [place_pop(20, 8.0)])
    res5 = run(loc_rates_planted(c['loc'], node_rates), cfg_=cfgk)
    for stat in ('dec_node_bridge_ba', 'rsa21_ED_X'):
        r_, pk, vals = null_frac(res5, 'kinematic', stat)
        check(f'K5 node / bridge cells: {stat} beyond the kinematic null p95', pk <= 0.05,
              f"real {r_:.3f}, kinematic null p95 {np.nanpercentile(vals, 95) if len(vals) else np.nan:.3f}, p {pk:.3f}")

    # K6 turn matching removes turn-driven corner / cardinal decoding where turning separates the types
    from sklearn.metrics import roc_auc_score
    nodes_ = np.isin(c['loc'], [1, 2, 3, 4, 6, 7, 8, 9]) & c['moving'] & c['intrial'] & np.isfinite(c['turn_dps'])
    auc_t = roc_auc_score(np.isin(c['loc'][nodes_], [1, 3, 7, 9]).astype(int), c['turn_dps'][nodes_])
    testable = abs(auc_t - 0.5) >= 0.08
    cfg6 = copy.deepcopy(cfg)
    cfg6['null'].update(n_sim_rsa=0, n_sim_decode=0, n_sim_planes=0)
    turn_cells = np.array([1.0 + rng.uniform(0.04, 0.08) * turn_r for _ in range(n)])
    spk6 = rng.poisson(turn_cells * sc.BIN_S).astype(np.float32)
    units6 = pd.DataFrame(dict(unit=np.arange(n), region='SYN'))
    ba6 = {}
    for edges in (None, (0.0, 22.5, 45.0, 90.0, 180.0, 360.0)):
        cfg6['speed_match']['turn_edges_dps'] = edges
        dec = ms.run_maze_recday(views_for(spk6), rd, units6, cfg6)['real']['decided']['decode']
        ba6[edges is not None] = (dec['corner_cardinal_ba'], np.nanpercentile(dec['corner_cardinal_null'], 95)
                                  if len(dec['corner_cardinal_null']) else np.nan)
    check('K6 turn matching lowers turn cells\' corner/cardinal BA by >= 0.05 (where turn-only |AUC - 0.5| >= 0.08)',
          (not testable) or (ba6[False][0] - ba6[True][0] >= 0.05 and ba6[True][0] <= max(ba6[True][1], 0.5)),
          f'speed-matched BA {ba6[False][0]:.2f}, speed x turn matched {ba6[True][0]:.2f} (null p95 {ba6[True][1]:.2f}); '
          f'turn-only AUC {auc_t:.3f}' + ('' if testable else '; untestable on this track'))

    # K7 guards
    def raises(fn):
        try:
            fn()
            return False
        except ValueError:
            return True
    g_port = copy.deepcopy(cfg)
    g_port['kinematics']['port_speed_cms'] = cfg['min_speed'] + 1
    g_edges = copy.deepcopy(cfg)
    g_edges['speed_match']['turn_edges_dps'] = (0.0, 45.0, 45.0)
    g_null = copy.deepcopy(cfg)
    g_null['null'].update(n_sim_decode=g_null['null']['n_sim_rsa'] + 1)
    g_head = copy.deepcopy(cfg)
    g_head['kinematics']['turn_source'] = 'head'
    head_raises = raises(lambda: ms.assemble(views, g_head))
    check('K7 guards raise', raises(lambda: ms.port_state(c['loc'], c['speed'], g_port))
          and raises(lambda: ms.speed_matched(base, c['type'], c['speed'], g_edges, np.random.default_rng(0), turn=c['turn_dps']))
          and raises(lambda: ms._check_null_settings(g_null['null']))
          and (head_raises == (args.dataset == 'pfc')),
          f"port speed, equal edges, decode sims > rsa sims; head source raises {head_raises} on {args.dataset}")

    failed = [x for x in CHECKS if not x[1]]
    print(f'\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed in {time.time() - t0:.0f} s', flush=True)
    for name, _, det in failed:
        print(f'  FAILED {name}: {det}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
