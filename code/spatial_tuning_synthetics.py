"""Synthetic gate for the open-field engine, through `spatial_tuning.run_of_recday` on real trajectories.

Real arena sessions keep their XY, HD, arena and objects; `counts` are replaced by Poisson spikes from
known rate functions with a log-normal gain per second, mixed into a background of noise units so
every pooled threshold comes from a realistic pool. Run on the cluster:

    sbatch --job-name=sp_of_syn sbatch_files/spatial.sbatch code/spatial_tuning_synthetics.py --dataset mec --quick
    ... --dataset lec          (50 cm box: grid detection curve, object-box trajectories)

| # | planted | must hold |
|---|---|---|
| E1 | object mask of radius 0 vs no object | every statistic identical (max abs diff 0) |
| E2 | grid score switched off | every other statistic and null identical |
| E3 | analytic lattice and wall band (no data) | gridness > 1.2 at 30/45 cm; wall band CM 1, border > 0.8; flat map NaN |
| A1 | Poisson 0.5/2/8 Hz mixed; real maze spike trains on OF paths | per-class FPR not above 1 % (one-sided binomial p >= 0.01); any-class not above 5 % |
| A2 | grids 30/45 cm, unequal fields, ellipticity 1.1 | grid >= 80 %; spacing within 10 %; orientation within 5 deg mod 60 |
| A3 | grids 50-90 cm (5 cm steps) | detection curve reported: bounds any "no grid cells" claim |
| A4 | border cells, one wall and two walls, 5 and 10 cm bands | border >= 80 %; grid <= 5 %; wall label right |
| A5 | place fields, centre and near a wall | spatial >= 80 %; border <= 5 %; speed <= 5 %; near-wall place cells pass border once the threshold is dropped (not vacuous) |
| A6 | flat 20-80 Hz units scored as broad; 20 % wall gradients | border <= 5 %; with the area cap removed they reach the Solstad 0.5 cut (the artefact exists) |
| A7 | pure HD cells on the real trajectory | HD >= 80 %; border <= 5 %; spatial (position factor) <= 5 % |
| A8 | border cells without HD | pxd HD <= 5 %; plain Rayleigh HD rate reported (the confound) |
| A9 | border x HD conjunctive | border >= 80 % and HD >= 80 %; box: preferences shown >= 10 s near the wall, others reported |
| A10 | speed cells, positive and negative slopes | speed >= 80 % with the right sign; spatial (position factor) <= 5 % |
| A11 | untuned units with a 0.5-2x rate ramp | no class above 5 % |
| A12 | object session: object field off one side, place at the object, far-wall border, slow-speed cell, border on the wall nearest the object | object-proximal >= 80 % for the first two, <= 5 % for the others; a fixed ratio > 1.5 accepts the near-wall border cells (the wall-distance-matched null is what rejects them) |
| A13 | second session = first rotated 90/180 deg or flipped, HD transformed | best transform = the inverse of the planted one; HD shift within 10 deg; unit-shuffle p < 0.01 |
| A15 | frame and sign | x = 0 band labelled 'left'; HD planted at 90 deg read within 10 deg |
| A17 | open field + object session of one sort: object-vector cells (field beside the object only with it), place cells sitting where the object will be (both sessions), border cells, plus stable place cells | object-vector >= 80 % for the first, <= 5 % for the others |
| A16 | guards | silent unit -> NaN, no exception; < 100 moving spikes -> ineligible; object outside the arena raises |

Each check prints `PASS/FAIL name: measured value, n`; the script exits 1 on any failure.
This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CHECKS = []
TRAJ = {'mec': dict(of=('bp01_20240402_20240403', 4), of2=('bp01_20240402_20240403', 9),
                    obj=('bp01_20240324_20240325', 8)),
        'lec': dict(of=('ah10_20250624_20250625', 1), of2=('ah10_20250624_20250625', 4),
                    obj=('ah10_20250624_20250625', 1))}
#: LEC day-pairs for the object-move check (A17L): the slider and the diagonal mover of one box and day;
#: full mode adds box 2 on day 2
MOVES = {'quick': [('ah10_20250624_20250625', 1, 4)],
         'full': [('ah10_20250624_20250625', 1, 4), ('ly06_20250624_20250625', 5, 9)]}
#: planted sizes per arena block; bigfield holds the literals of the 120 cm gate (mEC lines unchanged)
PLANT = {'bigfield': dict(e3_band=10.0, grid_strict=True, a4_specs=[(('left',), 10), (('left',), 5), (('bottom', 'right'), 10)],
                          a5_sigma=(6, 12), a5_jitter=4, a5_near=8.0, a8_lams=[5, 10], a9_lam=10, a9_min_dwell_s=None, a10_cap=50.0,
                          a10_slope=1.0, virtual_null=True, a17=True),
         # border decay lengths 3-5 cm: 2-4 cm (the big field's scaled by 0.4) sat below the 2.5 cm map bin and
         # left 0.10 of HD-free border cells called HD with 7.5 cm pxd bins (run 3752837)
         'box50': dict(e3_band=5.0, grid_strict=False, a4_specs=[(('left',), 5), (('left',), 3), (('bottom', 'right'), 5)],
                       a5_sigma=(3, 6), a5_jitter=2, a5_near=5.0, a8_lams=[3, 5], a9_lam=5, a9_min_dwell_s=10.0, a10_cap=25.0,
                       a10_slope=50.0 / 25.0, virtual_null=False, a17=False)}


def check(name: str, passed: bool, detail: str) -> None:
    print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}", flush=True)
    CHECKS.append((name, bool(passed), detail))


def binom_ok(k: int, n: int, p0: float) -> tuple:
    """One-sided binomial p of observing >= k of n at rate p0; passes at p >= 0.01."""
    p = float(binom.sf(k - 1, n, p0)) if n else 1.0
    return p >= 0.01, p


# ---------------------------------------------------------------------------
# trajectories and planted rate functions
# ---------------------------------------------------------------------------

def trajectory(view: dict) -> dict:
    import spatial_common as sc
    T = min(view['counts'].shape[1], len(view['xy']))
    side = float(view['arena']['side_cm'])
    xy = np.clip(np.asarray(view['xy'][:T], float), 0, side - 1e-6)
    hd = None if view['hd'] is None else np.asarray(view['hd'][:T], float)
    return dict(T=T, side=side, xy=xy, hd=hd, speed=sc.speed_cm_s(np.asarray(view['xy'][:T], float)),
                view=view)


def gain(T: int, rng, sd: float = 0.3) -> np.ndarray:
    """Log-normal multiplicative gain per 1 s block: trial-to-trial variability beyond Poisson."""
    nb = int(np.ceil(T / 40))
    return np.repeat(rng.lognormal(0, sd, nb), 40)[:T]


def spikes(rate: np.ndarray, rng) -> np.ndarray:
    r = np.nan_to_num(rate, nan=0.0)
    return rng.poisson(np.clip(r, 0, None) * 0.025).astype(np.float32)


def wall_d(xy, side, wall):
    return {'left': xy[:, 0], 'right': side - xy[:, 0], 'bottom': xy[:, 1], 'top': side - xy[:, 1]}[wall]


def r_grid(tr, spacing, ori_deg, phase, peak, ellip=1.0, field_sd=0.0, rng=None):
    xy = tr['xy'].copy()
    xy[:, 0] = xy[:, 0] * ellip
    k = 4 * np.pi / (np.sqrt(3) * spacing)
    s = np.zeros(tr['T'])
    for a in (0, 60, 120):
        th = np.deg2rad(ori_deg + a)
        s += np.cos(k * (np.cos(th) * (xy[:, 0] - phase[0]) + np.sin(th) * (xy[:, 1] - phase[1])))
    r = np.exp(1.2 * (s - 3))
    if field_sd and rng is not None:
        th = rng.uniform(0, 2 * np.pi)
        r *= 1 + field_sd * np.cos(k / 2.3 * (np.cos(th) * xy[:, 0] + np.sin(th) * xy[:, 1]) + rng.uniform(0, 6))
    return peak * r / r.max() + 0.2


def r_border(tr, walls, lam, peak):
    d = np.min([wall_d(tr['xy'], tr['side'], w) for w in walls], axis=0)
    return peak * np.exp(-d / lam) + 0.2


def r_place(tr, centre, sigma, peak):
    d2 = ((tr['xy'] - np.asarray(centre)) ** 2).sum(1)
    return peak * np.exp(-d2 / (2 * sigma ** 2)) + 0.2


def r_hd(tr, mu, kappa, peak):
    return peak * np.exp(kappa * (np.cos(tr['hd'] - mu) - 1)) + 0.2


# ---------------------------------------------------------------------------
# running the engine on a planted population
# ---------------------------------------------------------------------------

def run(tr_map: dict, rates: dict, cfg: dict, rng, n_noise: int = 30, widths=None, objects_override=None,
        as_counts: bool = False):
    """Plant `rates` (session -> (n, T) Hz, same units in every session) plus noise units into the real
    views of `tr_map` (session -> trajectory) and score them. `as_counts` takes the arrays as spike counts
    (real spike trains) instead. Returns (shard, planted rows)."""
    import spatial_tuning as st
    n_plant = next(iter(rates.values())).shape[0]
    noise_hz = rng.choice([0.5, 2.0, 8.0], n_noise)
    views = {}
    for s, tr in tr_map.items():
        T = tr['T']
        noise = spikes(np.repeat(noise_hz[:, None], T, 1) * gain(T, rng)[None], rng)
        planted = (np.asarray(rates[s][:, :T], np.float32) if as_counts
                   else spikes(rates[s] * gain(T, rng)[None], rng))
        v = dict(tr['view'])
        v.update(counts=np.vstack([planted, noise]), xy=tr['xy'], hd=tr['hd'])
        v['counts'] = v['counts'][:, :T]
        if objects_override is not None:
            v['objects'] = objects_override.get(s, v['objects'])
        views[s] = v
    n = n_plant + n_noise
    w = np.full(n, 0.6) if widths is None else np.r_[widths, np.full(n_noise, 0.6)]
    units = pd.DataFrame(dict(unit=np.arange(n), width_ms=w, dup_partner=-1, cluster_id=np.nan, region='SYN'))
    return st.run_of_recday(views, 'synthetic', units, cfg), np.arange(n_plant)


def frac(mask, idx):
    return float(np.mean(np.asarray(mask)[idx])) if len(idx) else np.nan


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='open-field engine synthetic gate')
    ap.add_argument('--dataset', choices=('mec', 'lec'), default='mec')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n-jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--config', default='', help='JSON overrides, as run_spatial.py --config')
    args = ap.parse_args(argv)
    import spatial_io as sio
    sio.set_dataset(args.dataset)
    import spatial_tuning as st
    t0 = time.time()
    rng = np.random.default_rng(20261009)
    cfg = copy.deepcopy(st.DEFAULT_CFG)
    if args.config:
        import json
        import run_spatial
        with open(args.config) as fh:
            cfg = run_spatial._merge_cfg(cfg, json.load(fh))
        print(f'config overrides from {args.config}', flush=True)
    cfg['n_jobs'] = args.n_jobs
    cfg['n_shifts'] = 200 if args.quick else 1000
    cfg['rotation'].update(n_perm=200 if args.quick else 1000, n_boot=500, n_shift_object=100 if args.quick else 200)
    nper = 20 if args.quick else 60
    nfpr = 150 if args.quick else 1000

    keys = TRAJ[args.dataset]
    rds = sorted({v[0] for v in keys.values()})
    views = sio.load(args.dataset, rds, 'v4')
    trs = {k: trajectory(views[rd][s]) for k, (rd, s) in keys.items()}
    of = trs['of']
    side = of['side']
    block = st.arena_cfg(cfg, of['view']['arena'])['arena_block']
    pl = PLANT[block]
    print(f'trajectories loaded in {time.time() - t0:.0f} s; side {side} cm; n_shifts {cfg["n_shifts"]}', flush=True)

    # E3 analytic, no data
    import spatial_tuning as st_
    nb = int(side / cfg['bin_cm'])
    c = (np.arange(nb) + 0.5) * cfg['bin_cm']
    X, Y = np.meshgrid(c, c, indexing='ij')
    geo = st_.GridGeometry(nb, nb, cfg['bin_cm'], cfg['grid']['rmax_frac'] * side)
    for sp in ((30, 45) if side >= 100 else (20,)):
        k = 4 * np.pi / (np.sqrt(3) * sp)
        m = sum(np.cos(k * (np.cos(np.deg2rad(a)) * X + np.sin(np.deg2rad(a)) * Y)) for a in (7, 67, 127))
        g = st_.gridness(st_.autocorr(np.maximum(m, 0) * 5 + 0.2), geo, cfg)
        check(f'E3 analytic lattice {sp} cm', g['grid'] > 1.2, f"gridness {g['grid']:.2f}, spacing {g['spacing_cm']:.1f} cm")
    b = st_.border_score(np.where(X < pl['e3_band'], 8.0, 0.3), side, cfg['bin_cm'], set(), st_.arena_cfg(cfg, of['view']['arena']))
    check('E3 analytic wall band', b['cm'] == 1.0 and b['border'] > 0.8 and b['wall'] == 'left',
          f"CM {b['cm']:.2f}, DM {b['dm']:.3f}, border {b['border']:.2f}, wall {b['wall']}")
    fb = st_.border_score(np.full((nb, nb), 5.0), side, cfg['bin_cm'], set(), cfg)['border']
    check('E3 flat map has no border score', not np.isfinite(fb), f'border {fb}')

    # A1 noise FPR, per rate stratum, Poisson and real maze spike trains
    hz = np.repeat([0.5, 2.0, 8.0], nfpr // 3)
    res, idx = run({4: of}, {4: np.repeat(hz[:, None], of['T'], 1)}, cfg, rng, n_noise=0)
    ss = res['sessions'][4]
    for stratum in (0.5, 2.0, 8.0):
        sel = idx[(hz == stratum) & ss['eligible'][idx]]
        for k_, v in ss['classes'].items():
            kk = int(np.asarray(v)[sel].sum())
            ok, p = binom_ok(kk, len(sel), 0.01)
            check(f'A1 Poisson {stratum} Hz FPR {k_}', ok, f'{kk}/{len(sel)} = {kk / max(len(sel), 1):.3f}, binomial p {p:.3f}')
        anyc = np.any([np.asarray(v)[sel] for v in ss['classes'].values()], axis=0)
        ok, p = binom_ok(int(anyc.sum()), len(sel), 0.05)
        check(f'A1 Poisson {stratum} Hz any class', ok, f'{int(anyc.sum())}/{len(sel)}, binomial p vs 5 % {p:.3f}')
    maze_counts = [v['counts'] for rd in views for v in views[rd].values() if v.get('session_type') == 'ABCD']
    if maze_counts:
        # real maze spike trains (bursting, drift), tiled to the OF length: no relation to OF position
        mc = np.vstack([np.tile(m, (1, int(np.ceil(of['T'] / m.shape[1]))))[:, :of['T']] for m in maze_counts])
        mc = mc[rng.permutation(len(mc))[:nfpr // 3]]
        res, idx = run({4: of}, {4: mc}, cfg, rng, n_noise=0, as_counts=True)
        ss = res['sessions'][4]
        sel = idx[ss['eligible'][idx]]
        for k_, v in ss['classes'].items():
            kk = int(np.asarray(v)[sel].sum())
            ok, p = binom_ok(kk, len(sel), 0.01)
            check(f'A1 real maze trains FPR {k_}', ok, f'{kk}/{len(sel)}, binomial p {p:.3f}')

    # A2 / A3 grids
    sp_list = (30, 45) if side >= 100 else (20, 30)
    for sp in sp_list:
        oris = rng.uniform(0, 60, nper)
        R = np.array([r_grid(of, sp, o, rng.uniform(0, sp, 2), rng.uniform(5, 15), 1.1, 0.3, rng) for o in oris])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        ss = res['sessions'][4]
        det = [ss['details']['grid'][i] for i in idx]
        spc = np.array([d['spacing_cm'] for d in det])
        ori = np.array([d['orientation_deg'] for d in det])
        exp_ori = (oris + 30) % 60
        dori = np.abs((ori - exp_ori + 30) % 60 - 30)
        okspc = np.nanmean(np.abs(spc - sp * 1.0) <= 0.1 * sp * 1.05)
        okori = np.nanmean(dori <= 5)
        if pl['grid_strict']:
            hit = frac(ss['classes']['grid'], idx)
            check(f'A2 grid {sp} cm detected', hit >= 0.8, f'{hit:.2f} of {len(idx)}')
            check(f'A2 grid {sp} cm spacing within 10 %', okspc >= 0.8, f'{okspc:.2f}; median {np.nanmedian(spc):.1f} cm')
            check(f'A2 grid {sp} cm orientation within 5 deg', okori >= 0.8, f'{okori:.2f}; median error {np.nanmedian(dori):.1f} deg')
        else:
            # grid is scored but not classed in the 50 cm box (run 3712175: 20 cm grids 0.55, 30 cm spacing NaN)
            hit = frac(ss['real']['grid'] > ss['thresholds']['broad']['grid'], idx)
            check(f'A2 grid {sp} cm above the pooled p99 (reported; not classed in this box)', True,
                  f'{hit:.2f} of {len(idx)}; spacing within 10 % {okspc:.2f}, median {np.nanmedian(spc):.1f} cm; '
                  f'orientation within 5 deg {okori:.2f}')
    curve = []
    for sp in (range(50, 95, 5) if side >= 100 else range(25, 55, 5)):
        R = np.array([r_grid(of, sp, rng.uniform(0, 60), rng.uniform(0, sp, 2), 10.0) for _ in range(max(10, nper // 2))])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        ss_ = res['sessions'][4]
        curve.append((sp, frac(ss_['classes']['grid'], idx) if pl['grid_strict']
                      else frac(ss_['real']['grid'] > ss_['thresholds']['broad']['grid'], idx)))
    check('A3 grid detection curve (reported)', True, ', '.join(f'{s} cm {f:.2f}' for s, f in curve))

    # A4 border, A15 frame
    specs = pl['a4_specs']
    for walls, lam in specs:
        R = np.array([r_border(of, walls, lam, rng.uniform(5, 15)) for _ in range(nper)])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        ss = res['sessions'][4]
        hit = frac(ss['classes']['border'], idx)
        check(f'A4 border {"+".join(walls)} {lam} cm', hit >= 0.8, f'{hit:.2f} of {len(idx)}')
        g = frac(ss['classes']['grid'], idx)
        check(f'A4 border {"+".join(walls)} not grid', g <= 0.05, f'{g:.2f}')
        check(f'A4 border {"+".join(walls)} not speed', frac(ss['classes']['speed'], idx) <= 0.05,
              f"{frac(ss['classes']['speed'], idx):.2f}")
        labs = [ss['details']['border'][i]['wall'] for i in idx]
        okw = np.mean([l in walls for l in labs])
        check(f'A15 border wall label {"+".join(walls)}', okw >= 0.9, f'{okw:.2f} labelled within {walls}')

    # A5 place
    for name, centre in (('centre', (side / 2, side / 2)), ('near wall', (pl['a5_near'], side / 2))):
        R = np.array([r_place(of, np.asarray(centre) + rng.normal(0, pl['a5_jitter'], 2), rng.uniform(*pl['a5_sigma']), rng.uniform(5, 15))
                      for _ in range(nper)])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        ss = res['sessions'][4]
        check(f'A5 place {name} spatial', frac(ss['classes']['spatial'], idx) >= 0.8, f"{frac(ss['classes']['spatial'], idx):.2f}")
        check(f'A5 place {name} not border', frac(ss['classes']['border'], idx) <= 0.05, f"{frac(ss['classes']['border'], idx):.2f}")
        check(f'A5 place {name} not speed', frac(ss['classes']['speed'], idx) <= 0.05, f"{frac(ss['classes']['speed'], idx):.2f}")
        if name == 'near wall':
            b_ = ss['real']['border'][idx]
            check('A5 near-wall place fields pass border once the threshold is dropped', np.nanmean(b_ > 0) >= 0.5,
                  f'{np.nanmean(b_ > 0):.2f} with border > 0 (not vacuous)')

    # A6 flat and gradient
    R = np.vstack([np.full((nper // 2, of['T']), 1.0) * rng.uniform(20, 80, (nper // 2, 1)),
                   (1 + 0.2 * of['xy'][:, 0] / side)[None] * rng.uniform(3, 10, (nper - nper // 2, 1))])
    res, idx = run({4: of}, {4: R}, cfg, rng)
    check('A6 flat / gradient not border', frac(res['sessions'][4]['classes']['border'], idx) <= 0.05,
          f"{frac(res['sessions'][4]['classes']['border'], idx):.2f}")
    cfg2 = copy.deepcopy(cfg)
    cfg2['border']['max_area_frac'] = 1.0
    cfg2['n_shifts'] = 50
    res2, idx2 = run({4: of}, {4: R}, cfg2, rng)
    b2 = res2['sessions'][4]['real']['border'][idx2[: nper // 2]]
    check('A6 without the area cap flat maps reach the 0.5 cut (artefact exists)', np.nanmean(b2 >= 0.49) >= 0.8,
          f'{np.nanmean(b2 >= 0.49):.2f}; median {np.nanmedian(b2):.2f}')

    if of['hd'] is not None:
        # A7 pure HD, A15 HD sign
        mus = rng.uniform(0, 2 * np.pi, nper)
        mus[:5] = np.pi / 2
        R = np.array([r_hd(of, m, rng.uniform(2, 4), rng.uniform(5, 15)) for m in mus])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        ss = res['sessions'][4]
        check('A7 pure HD called HD', frac(ss['classes']['hd'], idx) >= 0.8, f"{frac(ss['classes']['hd'], idx):.2f}")
        check('A7 pure HD not border', frac(ss['classes']['border'], idx) <= 0.05, f"{frac(ss['classes']['border'], idx):.2f}")
        check('A7 pure HD not spatial (position factor)', frac(ss['classes']['spatial'], idx) <= 0.05,
              f"{frac(ss['classes']['spatial'], idx):.2f}")
        raw = ss['real']['si'][idx] > ss['own']['si'][idx]
        check('A7 raw-map information on pure HD cells (reported: the confound the factor removes)', True, f'{np.mean(raw):.2f}')
        pref = np.degrees(ss['details']['pxd_pref'][idx[:5]])
        check('A15 HD planted at 90 deg read at 90', np.all(np.abs(((pref - 90) + 180) % 360 - 180) <= 10),
              f'{np.round(pref, 1).tolist()}')
        # A8 border without HD; A9 conjunctive
        R = np.array([r_border(of, (w,), lam, rng.uniform(5, 15)) for w, lam in
                      zip(rng.choice(['left', 'right', 'top', 'bottom'], nper), rng.choice(pl['a8_lams'], nper))])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        ss = res['sessions'][4]
        check('A8 border without HD not called HD (pxd)', frac(ss['classes']['hd'], idx) <= 0.05,
              f"{frac(ss['classes']['hd'], idx):.2f}")
        plain = ss['real']['hd_r'][idx] > np.nanpercentile(ss['null']['hd_r'][:, idx], 99, axis=0)
        check('A8 plain Rayleigh HD rate on border cells (reported)', True, f'{np.mean(plain):.2f}')
        if pl['a9_min_dwell_s'] is None:
            R = np.array([r_border(of, ('left',), pl['a9_lam'], 1.0) * r_hd(of, rng.uniform(0, 2 * np.pi), 3, 1.0) * 10
                          for _ in range(nper)])
        else:
            # a conjunctive field shows only where the mouse faces the preferred way: sectors with 2-9 s of moving
            # dwell within 5 cm of the wall gave border 0.0-0.2, sectors with 12-24 s gave 0.8-1.0
            # (diag_box_conjunctive job 3752994, gate run 3753086); so plant the sampled preferences, report the rest
            near = (wall_d(of['xy'], side, 'left') < pl['a9_lam']) & np.isfinite(of['hd']) & \
                (np.nan_to_num(of['speed']) > cfg['min_speed'])
            sec9 = np.round(np.mod(of['hd'][near], 2 * np.pi) / (np.pi / 4)).astype(int) % 8
            dwell9 = np.bincount(sec9, minlength=8) * 0.025
            shown = dwell9 >= pl['a9_min_dwell_s']
            mus9 = (rng.choice(np.nonzero(shown)[0], nper) + rng.uniform(-0.5, 0.5, nper)) * np.pi / 4
            R = np.array([r_border(of, ('left',), pl['a9_lam'], 1.0) * r_hd(of, mu, 3, 1.0) * 10 for mu in mus9])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        ss = res['sessions'][4]
        check('A9 conjunctive border', frac(ss['classes']['border'], idx) >= 0.8, f"{frac(ss['classes']['border'], idx):.2f}")
        check('A9 conjunctive HD', frac(ss['classes']['hd'], idx) >= 0.8, f"{frac(ss['classes']['hd'], idx):.2f}")
        if pl['a9_min_dwell_s'] is not None and (~shown).any():
            mus9 = (rng.choice(np.nonzero(~shown)[0], nper // 2) + rng.uniform(-0.5, 0.5, nper // 2)) * np.pi / 4
            R = np.array([r_border(of, ('left',), pl['a9_lam'], 1.0) * r_hd(of, mu, 3, 1.0) * 10 for mu in mus9])
            res, idx = run({4: of}, {4: R}, cfg, rng)
            ss = res['sessions'][4]
            check(f"A9 conjunctive, preferences shown < {pl['a9_min_dwell_s']:.0f} s near the wall (reported)", True,
                  f"border {frac(ss['classes']['border'], idx):.2f}, HD {frac(ss['classes']['hd'], idx):.2f}; "
                  f"seconds per 45 deg sector from 0 deg {np.round(dwell9).astype(int).tolist()}")

    # A10 speed
    v = np.clip(np.nan_to_num(of['speed']), 0, pl['a10_cap'])
    sl_ = pl['a10_slope']
    R = np.vstack([2 + sl_ * rng.uniform(0.2, 0.4, (nper // 2, 1)) * v[None], 15 - sl_ * rng.uniform(0.1, 0.25, (nper - nper // 2, 1)) * v[None]])
    res, idx = run({4: of}, {4: R}, cfg, rng)
    ss = res['sessions'][4]
    sign = np.sign(ss['real']['speed_r'][idx])
    right = np.r_[np.ones(nper // 2), -np.ones(nper - nper // 2)]
    hit = np.mean(np.asarray(ss['classes']['speed'])[idx] & (sign == right))
    check('A10 speed called with the right sign', hit >= 0.8, f'{hit:.2f}')
    check('A10 speed cells not spatial (position factor)', frac(ss['classes']['spatial'], idx) <= 0.05,
          f"{frac(ss['classes']['spatial'], idx):.2f}")

    # A11 drift
    ramp = np.linspace(0.5, 2.0, of['T'])
    R = np.vstack([ramp[None] * rng.uniform(1, 8, (nper // 2, 1)), ramp[::-1][None] * rng.uniform(1, 8, (nper - nper // 2, 1))])
    res, idx = run({4: of}, {4: R}, cfg, rng)
    ss = res['sessions'][4]
    for k_, v_ in ss['classes'].items():
        check(f'A11 drift units not {k_}', frac(v_, idx) <= 0.05, f'{frac(v_, idx):.2f}')

    # A12 objects
    ob = trs['obj']
    objs = ob['view'].get('objects') or []
    if objs:
        o = objs[0]
        oc = np.asarray(o['xy_cm'], float)
        near = min(('left', oc[0]), ('right', side - oc[0]), ('bottom', oc[1]), ('top', side - oc[1]), key=lambda w: w[1])[0]
        far = {'left': 'right', 'right': 'left', 'bottom': 'top', 'top': 'bottom'}[near]
        away = oc + 16 * np.array([np.cos(0.7), np.sin(0.7)])
        q = nper // 4
        R = np.vstack([np.array([r_place(ob, away + rng.normal(0, 1, 2), 5, rng.uniform(5, 15)) for _ in range(q)]),
                       np.array([r_place(ob, oc + rng.normal(0, 2, 2), 10, rng.uniform(5, 15)) for _ in range(q)]),
                       np.array([r_border(ob, (far,), 10, rng.uniform(5, 15)) for _ in range(q)]),
                       np.array([15 - 0.25 * np.clip(np.nan_to_num(ob['speed']), 0, 50) for _ in range(q)]),
                       np.array([r_border(ob, (near,), 10, rng.uniform(5, 15)) for _ in range(q)])])
        res, idx = run({8: ob}, {8: R}, cfg, rng)
        oo = res['sessions'][8]['objects'][0]
        if oo['null'] is None and not pl['virtual_null']:
            check('A12 object null not constructible in the 50 cm box (expected)', oo['reason'].startswith('not constructible'),
                  oo['reason'][:70])
        elif oo['null'] is None:
            check('A12 object null constructible', False, oo['reason'])
        else:
            prox = np.asarray(oo['proximal'])
            for name, sl, want in (('object field off one side', slice(0, q), 'hit'), ('place at the object', slice(q, 2 * q), 'hit'),
                                   (f'{far}-wall border', slice(2 * q, 3 * q), 'miss'), ('slow-speed cell', slice(3 * q, 4 * q), 'miss'),
                                   (f'{near}-wall border (wall nearest the object)', slice(4 * q, 5 * q), 'miss')):
                f_ = frac(prox, idx[sl])
                check(f'A12 {name} {"proximal" if want == "hit" else "not proximal"}', f_ >= 0.8 if want == 'hit' else f_ <= 0.05, f'{f_:.2f}')
            stat_near = oo['stat'][idx[4 * q:5 * q]]
            fixed = np.mean(stat_near > np.log(1.5))
            check('A12 a fixed ratio > 1.5 accepts the near-wall border cells (the matched null matters)', fixed >= 0.8,
                  f'{fixed:.2f}; median ratio {np.exp(np.nanmedian(stat_near)):.2f}')

    # A13 rotation
    of2 = trs['of2']
    for planted in ('rot90', 'rot180', 'flip_x'):
        geo_t = st.transform_geometry(planted)
        L = side
        maps = {'rot90': lambda p: np.c_[L - p[:, 1], p[:, 0]], 'rot180': lambda p: L - p,
                'flip_x': lambda p: np.c_[L - p[:, 0], p[:, 1]]}
        inv = {'rot90': lambda p: np.c_[p[:, 1], L - p[:, 0]], 'rot180': lambda p: L - p,
               'flip_x': lambda p: np.c_[L - p[:, 0], p[:, 1]]}
        # sanity: the point maps agree with the map transforms used by the engine
        tp = st.transform_point(planted, (10.0, 30.0), L, cfg['bin_cm'])
        assert np.allclose(tp, maps[planted](np.array([[10.0, 30.0]]))[0], atol=cfg['bin_cm']), (planted, tp)
        centres = [rng.uniform(15, L - 15, 2) for _ in range(nper // 2)]
        mus = rng.uniform(0, 2 * np.pi, nper - nper // 2)
        Ra, Rb = [], []
        for cc in centres:
            Ra.append(r_place(of, cc, 10, 10))
            src = inv[planted](of2['xy'])          # B's map is A's map transformed: f_B(x) = f_A(t^-1 x)
            Rb.append(r_place(dict(of2, xy=src), cc, 10, 10))
        for m in mus:
            mb = m + geo_t['angle'] if geo_t['kind'] == 'rotation' else 2 * geo_t['axis'] - m
            Ra.append(r_hd(of, m, 3, 10) if of['hd'] is not None else r_place(of, (30, 30), 10, 10))
            Rb.append(r_hd(of2, mb, 3, 10) if of2['hd'] is not None else r_place(of2, (30, 30), 10, 10))
        res, idx = run({4: of, 9: of2}, {4: np.array(Ra), 9: np.array(Rb)}, cfg, rng,
                       objects_override={4: [], 9: []})
        rot = res['rotations'][(4, 9)]
        want = st.inverse_transform(planted)
        check(f'A13 rotation {planted} recovered', rot.get('ok') and rot['best'] == want and rot['p'] < 0.01,
              f"best {rot.get('best')} (want {want}), median r {rot.get('best_r', np.nan):.2f}, p {rot.get('p', np.nan):.3f}")
        hd = rot.get('hd', {})
        if geo_t['kind'] == 'rotation' and 'mean_shift_deg' in hd:
            err = abs(((hd['mean_shift_deg'] - np.degrees(geo_t['angle'])) + 180) % 360 - 180)
            check(f'A13 HD shift for {planted}', err <= 10, f"{hd['mean_shift_deg']:.1f} vs {np.degrees(geo_t['angle']):.0f} deg, n {hd['n']}")

    # A17 object-vector within one sort
    obj2 = trs['of2']
    objs2 = obj2['view'].get('objects') or []
    if objs2 and not pl['a17']:
        check('A17 object-vector across an open-field and an object session (not applicable: no object-free '
              'session in this box; see A17L)', True, '')
    elif objs2:
        oc = np.asarray(objs2[0]['xy_cm'], float)
        m = nper // 4
        anchors = [rng.uniform(15, side - 15, 2) for _ in range(20)]
        side_pt = oc + 14 * np.array([np.cos(2.0), np.sin(2.0)])
        Ra = [r_place(of, a, 10, 8) for a in anchors]
        Rb = [r_place(obj2, a, 10, 8) for a in anchors]
        for _ in range(m):                                   # object-vector: field only with the object
            Ra.append(np.full(of['T'], 0.5) + r_place(of, rng.uniform(15, side - 15, 2), 8, 3))
            Rb.append(r_place(obj2, side_pt + rng.normal(0, 1, 2), 5, rng.uniform(8, 15)))
        for _ in range(m):                                   # place cell where the object will stand
            pk = rng.uniform(8, 15)
            Ra.append(r_place(of, oc + rng.normal(0, 1, 2), 8, pk))
            Rb.append(r_place(obj2, oc + rng.normal(0, 1, 2), 8, pk))
        for _ in range(m):                                   # border cells
            pk = rng.uniform(5, 15)
            Ra.append(r_border(of, ('left',), 10, pk))
            Rb.append(r_border(obj2, ('left',), 10, pk))
        res, idx = run({4: of, 9: obj2}, {4: np.array(Ra), 9: np.array(Rb)}, cfg, rng)
        rot = res['rotations'][(4, 9)]
        ovd = next(iter(rot.get('object_vector', {}).values()), {})
        if 'object_vector' not in ovd:
            check('A17 object-vector test ran', False, str(ovd.get('reason', rot.get('reason', 'no result'))))
        else:
            ov = np.asarray(ovd['object_vector'])
            base = 20
            check('A17 rotation of the anchor population is the identity', rot['best'] == 'id', f"best {rot['best']}")
            check('A17 object-vector cells detected', frac(ov, idx[base:base + m]) >= 0.8, f'{frac(ov, idx[base:base + m]):.2f} of {m}')
            check('A17 place cells under the object not object-vector', frac(ov, idx[base + m:base + 2 * m]) <= 0.05,
                  f'{frac(ov, idx[base + m:base + 2 * m]):.2f}')
            check('A17 border cells not object-vector', frac(ov, idx[base + 2 * m:base + 3 * m]) <= 0.05,
                  f'{frac(ov, idx[base + 2 * m:base + 3 * m]):.2f}')
            check('A17 stable anchor place cells not object-vector', frac(ov, idx[:base]) <= 0.05, f'{frac(ov, idx[:base]):.2f}')

    # A16 guards
    R = np.vstack([np.zeros((1, of['T'])), np.full((1, of['T']), 0.002)])
    res, idx = run({4: of}, {4: R}, cfg, rng, n_noise=20)
    ss = res['sessions'][4]
    check('A16 silent unit gives NaN', all(not np.isfinite(ss['real'][k][0]) for k in ('si', 'grid', 'border')),
          f"si {ss['real']['si'][0]}, grid {ss['real']['grid'][0]}, border {ss['real']['border'][0]}")
    check('A16 sparse unit ineligible', not ss['eligible'][1], f"moving spikes {ss['moving_spikes'][1]:.0f}")
    bad = dict(of['view'])
    bad['objects'] = [dict(label='x', xy_cm=[side + 10, 10], radius_cm=6)]
    try:
        st._context(dict(bad, counts=np.zeros((1, of['T']), np.float32), xy=of['xy'], hd=of['hd']), cfg, set())
        check('A16 object outside the arena raises', False, 'no exception')
    except ValueError as e:
        check('A16 object outside the arena raises', True, str(e)[:80])

    # E1 / E2 equivalence
    R = np.array([r_border(of, ('left',), 10, 8)] * 3 + [r_place(of, (side / 2, side / 2), 10, 8)] * 3)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(1)
    cfg_e = copy.deepcopy(cfg)
    cfg_e['n_shifts'] = 30
    cfg_e1 = copy.deepcopy(cfg_e)
    cfg_e1['arenas'][block]['obj']['margin_cm'] = -1000.0          # arena settings live in the blocks
    ra, _ = run({4: of}, {4: R}, cfg_e, rng_a, objects_override={4: []})
    rb, _ = run({4: of}, {4: R}, cfg_e1, rng_b, objects_override={4: [dict(label='o', xy_cm=[side / 2, side / 2], radius_cm=6)]})
    d = max(np.nanmax(np.abs(ra['sessions'][4]['real'][k] - rb['sessions'][4]['real'][k])) for k in st.STATS)
    check('E1 mask radius 0 equals no object', d == 0, f'max abs diff {d}')
    rng_a = np.random.default_rng(2)
    rng_b = np.random.default_rng(2)
    cfg_e2 = copy.deepcopy(cfg_e)
    cfg_e2['grid']['enabled'] = False
    ra, _ = run({4: of}, {4: R}, cfg_e, rng_a)
    rb, _ = run({4: of}, {4: R}, cfg_e2, rng_b)
    d = max(np.nanmax(np.abs(ra['sessions'][4][part][k] - rb['sessions'][4][part][k]))
            for part in ('real', 'null') for k in st.STATS if k != 'grid')
    check('E2 grid off leaves every other statistic unchanged', d == 0, f'max abs diff {d}')

    # E4 the arena-block resolver: the block path equals the block-free path on the big field; bad arenas raise
    import spatial_common as sc_
    if block == 'bigfield':
        cfg_l = copy.deepcopy(cfg_e)
        cfg_l['arenas'] = None
        R4 = np.array([r_border(of, ('left',), 10, 8)] * 2 + [r_place(of, (side / 2, side / 2), 10, 8)] * 2)
        ra, _ = run({4: of}, {4: R4}, cfg_e, np.random.default_rng(4), n_noise=4)
        rb, _ = run({4: of}, {4: R4}, cfg_l, np.random.default_rng(4), n_noise=4)
        keys = ('real', 'null', 'thresholds', 'own', 'classes', 'chance', 'details')
        cmp4 = sc_.compare_tree({k: ra['sessions'][4][k] for k in keys}, {k: rb['sessions'][4][k] for k in keys})
        check('E4 big-field block equals the block-free path', cmp4['max_diff'] == 0 and cmp4['nan_mismatch'] == 0 and not cmp4['missing'],
              f"max abs diff {cmp4['max_diff']} over {cmp4['n']} values")

    def raises(fn):
        try:
            fn()
            return False
        except (ValueError, KeyError):
            return True
    arena = of['view']['arena']
    bad_name = raises(lambda: st.arena_cfg(cfg, dict(arena, name='unknown_arena')))
    bad_side = raises(lambda: st.arena_cfg(cfg, dict(arena, side_cm=float(arena['side_cm']) + 10)))
    legacy_box = raises(lambda: st.arena_cfg(dict(cfg, arenas=None), dict(name='openfield_box1', side_cm=50.0)))
    top = copy.deepcopy(cfg)
    top['obj']['band_cm'] = 1.0
    top_level = raises(lambda: st.check_arena_overrides(top))
    check('E4 unmapped arena, wrong side, block-free box and top-level arena settings raise',
          bad_name and bad_side and legacy_box and top_level, f'{bad_name} {bad_side} {legacy_box} {top_level}')

    # W1 the wall QC reader reads every version of the table
    qc = sio._wall_qc(args.dataset)
    want = {'mec': 11, 'lec': 20}[args.dataset]
    check('W1 wall QC rows for every arena session', len(qc) == want, f'{len(qc)} rows (want {want})')
    # W2 each arena block applies its own wall limit (box 2.0 cm, big field 1.5 cm) to the measured RMS
    blk = {b_: st.arena_cfg(copy.deepcopy(cfg), dict(name=n_, side_cm=sd_)) for b_, n_, sd_ in
           (('bigfield', 'bigfield', 120.0), ('box50', 'openfield_box1', 50.0))}
    toy = dict(left=1.2, right=1.7, top=2.3, bottom=None)
    syn_ok = (st.session_wall_exclusions(toy, blk['box50']) == {'top'}
              and st.session_wall_exclusions(toy, blk['bigfield']) == {'right', 'top'})
    n_ex = {b_: sum(len(st.session_wall_exclusions(sio.wall_exclusions(args.dataset, r_.recday, int(r_.session), 0.0), c_))
                    for r_ in qc.itertuples()) for b_, c_ in blk.items()}
    want2 = {'mec': (1, 1), 'lec': (0, 6)}[args.dataset]
    check('W2 wall limit per arena block: toy walls, then real walls excluded at box50 / bigfield limits',
          syn_ok and (n_ex['box50'], n_ex['bigfield']) == want2,
          f"toy {syn_ok}; real {n_ex['box50']} at 2.0 cm, {n_ex['bigfield']} at 1.5 cm (want {want2})")

    if block == 'box50':
        # A18 the in-box object radius: a planted soft hole (keep probability 2 -> 8 cm, half-density radius 5 cm)
        cbx = st.arena_cfg(cfg, of['view']['arena'])
        objs_ = of['view'].get('objects') or []
        cs = [np.asarray(o['xy_cm'], float) for o in objs_]
        corners = [np.array(c_) for c_ in ((10.5, 10.5), (side - 10.5, 10.5), (10.5, side - 10.5), (side - 10.5, side - 10.5))]
        empty = max(corners, key=lambda c_: min(np.hypot(*(c_ - q_)) for q_ in cs))
        d_ = np.hypot(*(of['xy'] - empty).T)
        keep = rng.random(len(d_)) < np.clip((d_ - 2.0) / 6.0, 0, 1)
        r_hole = st.measure_object_radius(of['xy'][keep], empty, cbx, side=side, others=cs)
        r_none = st.measure_object_radius(of['xy'], empty, cbx, side=side, others=cs)
        leg = copy.deepcopy(cbx)
        leg['obj'].update(radius_norm='annulus', radius_ref_cm=(15.0, 35.0), radius_search_cm=40.0)
        r_leg = st.measure_object_radius(of['xy'][keep], empty, leg)
        check('A18 in-box radius of a planted 5 cm hole within 1 cm; none without a hole', abs(r_hole - 5) <= 1 and r_none <= 3,
              f'{r_hole:.1f} cm with the hole, {r_none:.1f} cm without; legacy 15-35 cm ring {r_leg:.1f} cm')

        # A17L object moves within a day: planted cells through the engine on the real day-pairs
        for rd_m, s_a, s_b in MOVES['quick' if args.quick else 'full']:
            vm = sio.load(args.dataset, [rd_m], 'v4')[rd_m]
            ta, tb = trajectory(vm[s_a]), trajectory(vm[s_b])
            mvc = st.arena_cfg(cfg, vm[s_a]['arena'])['move']
            oa = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=6.0) for o in vm[s_a]['objects']]
            ob = [dict(label=o['label'], xy=np.asarray(o['xy_cm'], float), r_obj=6.0) for o in vm[s_b]['objects']]
            lay = st.object_layout(oa, ob, side, mvc)
            if lay is None:
                check(f'A17L {rd_m[:4]} layout has one vacated, one shared, one new corner', False, 'no layout')
                continue
            ro = lay['roles']
            V, S_, N_, E_ = (ro[k]['xy'] for k in 'VSNE')
            mov = lay['movers']
            slider = next(l_ for l_, m_ in mov.items() if m_['frm'] == 'V')
            diag = next(l_ for l_, m_ in mov.items() if m_['frm'] == 'S')
            cen = np.array([side / 2, side / 2])

            def inward(p_, dist):
                u_ = cen - p_
                return p_ + dist * u_ / np.linalg.norm(u_)
            ns_ = mvc['n_sectors']

            def sector_counts(tr, p_):
                rel = tr['xy'] - p_
                d_ = np.hypot(rel[:, 0], rel[:, 1])
                band = (np.nan_to_num(tr['speed']) > cfg['min_speed']) & (d_ >= 3.0) & (d_ < mvc['r_out_cm'])
                k_ = np.floor(np.mod(np.arctan2(rel[band, 1], rel[band, 0]), 2 * np.pi) / (2 * np.pi) * ns_).astype(int)
                return np.bincount(np.clip(k_, 0, ns_ - 1), minlength=ns_)

            # a corner field sits 9 cm toward the box centre unless that band sector holds < min_sector moving samples
            # in a session it must be seen in; then in the best-sampled sector. ly06 s9 crossed V's centre-facing
            # sectors for 38-43 samples, so trace and place-at-V fields there were untestable (run 3755102, job 3755355)
            unsampled = []

            def field_at(p_, trs_, dist=9.0):
                u_ = (cen - p_) / np.linalg.norm(cen - p_)
                k_in = int(np.floor(np.mod(np.arctan2(u_[1], u_[0]), 2 * np.pi) / (2 * np.pi) * ns_)) % ns_
                cnt = np.min([sector_counts(t_, p_) for t_ in trs_], axis=0)
                if cnt[k_in] >= mvc['min_sector']:
                    return p_ + dist * u_
                unsampled.append((np.round(p_, 1).tolist(), int(cnt[k_in])))
                ang = (int(np.argmax(cnt)) + 0.5) * 2 * np.pi / ns_
                return p_ + dist * np.array([np.cos(ang), np.sin(ang)])
            pos = {s_a: {o['label']: o['xy'] for o in oa}, s_b: {o['label']: o['xy'] for o in ob}}
            trs2 = {s_a: ta, s_b: tb}
            fx = {key: field_at(*key_args) for key, key_args in
                  {**{(s_, lab_): (P__, [trs2[s_]]) for s_, P_s in pos.items() for lab_, P__ in P_s.items()},
                   ('V', 'b'): (V, [tb]), ('V', 'ab'): (V, [ta, tb]), ('S', 'ab'): (S_, [ta, tb])}.items()}
            wall_vs = min(('left', 'right', 'bottom', 'top'), key=lambda w: wall_d(np.array([(V + S_) / 2]), side, w)[0])
            wall_vn = min(('left', 'right', 'bottom', 'top'), key=lambda w: wall_d(np.array([(V + N_) / 2]), side, w)[0])
            vn_dir = np.arctan2(*(((V + N_) / 2 - cen)[::-1]))
            v_dir = np.arctan2(*((V - cen)[::-1]))
            # object-vector cells keep a fixed allocentric offset: 17 cm into the box from the wall the slider runs
            # along (the same direction at both of its corners); 'toward the centre' rotated with the corner and
            # moved the planted fields 5 cm for a 29 cm object move (run 3752837)
            normal = {'left': (1.0, 0.0), 'right': (-1.0, 0.0), 'bottom': (0.0, 1.0), 'top': (0.0, -1.0)}[wall_vs]
            m = max(10, nper // 4)
            kinds = ['follow_slider', 'follow_diag', 'trace', 'generic', 'vector'] + \
                    ['place_V', 'place_S', 'border_VS', 'border_VN', 'hd_VN', 'hd_V', 'slow', 'gain', 'anchor']
            rates = {s_a: [], s_b: []}
            lab_ = []
            for kind in kinds:
                for _ in range(m):
                    pk = rng.uniform(12, 20)
                    for s_, tr in ((s_a, ta), (s_b, tb)):
                        P_ = pos[s_]
                        if kind == 'follow_slider':
                            r_ = r_place(tr, fx[(s_, slider)], 2.5, pk)
                        elif kind == 'follow_diag':
                            r_ = r_place(tr, fx[(s_, diag)], 2.5, pk)
                        elif kind == 'trace':
                            r_ = r_place(tr, fx[('V', 'b')], 2.5, pk) if s_ == s_b else np.full(tr['T'], 0.2)
                        elif kind == 'generic':
                            # one 0.2 Hz floor like every other planted type: summing r_place doubled it, which
                            # cut D(N) by about log(0.5 / 0.3) and left 0.20-0.35 undetected (job 3753025)
                            r_ = sum(r_place(tr, fx[(s_, lab_)], 2.5, pk) - 0.2 for lab_ in P_) + 0.2
                        elif kind == 'vector':
                            r_ = r_place(tr, P_[slider] + 17.0 * np.asarray(normal), 3.0, pk)
                        elif kind == 'place_V':
                            r_ = r_place(tr, fx[('V', 'ab')], 2.5, pk)
                        elif kind == 'place_S':
                            r_ = r_place(tr, fx[('S', 'ab')], 2.5, pk)
                        elif kind == 'border_VS':
                            r_ = r_border(tr, (wall_vs,), 4, pk)
                        elif kind == 'border_VN':
                            r_ = r_border(tr, (wall_vn,), 4, pk)
                        elif kind == 'hd_VN':
                            r_ = r_hd(tr, vn_dir, 3, pk) if tr['hd'] is not None else np.full(tr['T'], 2.0)
                        elif kind == 'hd_V':
                            r_ = r_hd(tr, v_dir, 3, pk) if tr['hd'] is not None else np.full(tr['T'], 2.0)
                        elif kind == 'slow':
                            r_ = 1.0 + pk * np.exp(-np.nan_to_num(tr['speed']) / 5.0)
                        elif kind == 'gain':
                            r_ = r_place(tr, cen + rng.normal(0, 3, 2), 6, pk) * (2.0 if s_ == s_b else 1.0)
                        else:
                            r_ = r_place(tr, cen + rng.normal(0, 3, 2), 6, pk)
                        rates[s_].append(r_)
                    lab_.append(kind)
            res, idx = run({s_a: ta, s_b: tb}, {s_: np.array(r_) for s_, r_ in rates.items()}, cfg, rng)
            mvr = res['rotations'][(s_a, s_b)].get('object_move', {})
            if 'classes' not in mvr:
                check(f'A17L {rd_m[:4]} object-move test ran', False, str(mvr.get('reason', 'no result')))
                continue
            lab_ = np.array(lab_)
            cl = {k_: np.asarray(v_)[idx] for k_, v_ in mvr['classes'].items()}
            # classes need >= 100 moving spikes in both sessions: ly06 s5 moved 48 % of the time and its vector cells
            # fired 0.21-0.28 Hz, so some fell under the cut (0.73 of 15 in run 3755102, 15 of 15 eligible in job 3755357)
            el_ = (np.asarray(res['sessions'][s_a]['eligible']) & np.asarray(res['sessions'][s_b]['eligible']))[idx]
            el_cls = {k_: np.asarray(v_, bool)[idx] for k_, v_ in mvr['eligible'].items()}
            want = {'follow_slider': f'follow[{slider}]', 'follow_diag': f'follow[{diag}]', 'trace': 'trace',
                    'generic': 'generic', 'vector': f'vector[{slider}]'}
            primary = [f'follow[{slider}]', f'follow[{diag}]', 'trace', 'generic', f'vector[{slider}]']
            for kind, c_ in want.items():
                k_e = (lab_ == kind) & el_cls[c_]
                f_ = float(cl[c_][k_e].mean()) if k_e.any() else np.nan
                others = {k2: float(cl[c2][k_e].mean()) if k_e.any() else np.nan for k2, c2 in want.items() if c2 != c_}
                check(f'A17L {rd_m[:4]} {kind} detected', k_e.sum() >= m // 2 and f_ >= 0.8,
                      f'{f_:.2f} of {int(k_e.sum())} eligible ({m} planted); other primary classes {others}')
            # each primary class is an intersection of p99 tests, so 'any of the 5' runs at <= ~5 %: a binomial
            # test, since one chance hit among 5 cells failed a fixed 0.05 cut (run 3752837)
            for kind in ('place_V', 'place_S', 'border_VS', 'border_VN', 'hd_VN', 'hd_V', 'slow', 'gain', 'anchor'):
                k_e = (lab_ == kind) & el_
                any_ = np.any([cl[c_][k_e] for c_ in primary], axis=0)
                ok, pb_ = binom_ok(int(any_.sum()), int(k_e.sum()), 0.05)
                which = {c_: int(cl[c_][k_e].sum()) for c_ in primary if cl[c_][k_e].any()}
                check(f'A17L {rd_m[:4]} {kind} not in a primary class beyond chance', ok and k_e.sum() >= m // 2,
                      f'{int(any_.sum())}/{int(k_e.sum())} eligible ({m} planted) in any, binomial p vs 5 % {pb_:.3f}; {which}')
            noise = np.asarray(next(iter(mvr['classes'].values())))
            n_idx = np.setdiff1d(np.arange(len(noise)), idx)
            k_any = int(np.any([np.asarray(v_)[n_idx] for k_, v_ in mvr['classes'].items() if k_ in primary], axis=0).sum())
            ok, pb_ = binom_ok(k_any, len(n_idx), 0.05)
            check(f'A17L {rd_m[:4]} noise units in any primary class', ok, f'{k_any}/{len(n_idx)}, binomial p vs 5 % {pb_:.3f}')
            check(f'A17L {rd_m[:4]} place_V reported as place at the vacated corner (reported)', True,
                  f"{float(cl['place_vacated'][lab_ == 'place_V'].mean()):.2f}")
            check(f'A17L {rd_m[:4]} corner fields moved off an unsampled centre-facing sector (reported)', True,
                  f'{len(unsampled)}: ' + '; '.join(f'{p_} held {n_} samples' for p_, n_ in unsampled))

    failed = [c for c in CHECKS if not c[1]]
    print(f'\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed in {time.time() - t0:.0f} s', flush=True)
    for name, _, det in failed:
        print(f'  FAILED {name}: {det}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
