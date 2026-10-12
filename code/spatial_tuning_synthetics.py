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
| A9 | border x HD conjunctive | border >= 80 % and HD >= 80 % |
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
    b = st_.border_score(np.where(X < 10, 8.0, 0.3), side, cfg['bin_cm'], set(), cfg)
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
        hit = frac(ss['classes']['grid'], idx)
        check(f'A2 grid {sp} cm detected', hit >= 0.8, f'{hit:.2f} of {len(idx)}')
        okspc = np.nanmean(np.abs(spc - sp * 1.0) <= 0.1 * sp * 1.05)
        check(f'A2 grid {sp} cm spacing within 10 %', okspc >= 0.8, f'{okspc:.2f}; median {np.nanmedian(spc):.1f} cm')
        okori = np.nanmean(dori <= 5)
        check(f'A2 grid {sp} cm orientation within 5 deg', okori >= 0.8, f'{okori:.2f}; median error {np.nanmedian(dori):.1f} deg')
    curve = []
    for sp in (range(50, 95, 5) if side >= 100 else range(25, 55, 5)):
        R = np.array([r_grid(of, sp, rng.uniform(0, 60), rng.uniform(0, sp, 2), 10.0) for _ in range(max(10, nper // 2))])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        curve.append((sp, frac(res['sessions'][4]['classes']['grid'], idx)))
    check('A3 grid detection curve (reported)', True, ', '.join(f'{s} cm {f:.2f}' for s, f in curve))

    # A4 border, A15 frame
    specs = [(('left',), 10), (('left',), 5), (('bottom', 'right'), 10)]
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
    for name, centre in (('centre', (side / 2, side / 2)), ('near wall', (8.0, side / 2))):
        R = np.array([r_place(of, np.asarray(centre) + rng.normal(0, 4, 2), rng.uniform(6, 12), rng.uniform(5, 15))
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
                      zip(rng.choice(['left', 'right', 'top', 'bottom'], nper), rng.choice([5, 10], nper))])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        ss = res['sessions'][4]
        check('A8 border without HD not called HD (pxd)', frac(ss['classes']['hd'], idx) <= 0.05,
              f"{frac(ss['classes']['hd'], idx):.2f}")
        plain = ss['real']['hd_r'][idx] > np.nanpercentile(ss['null']['hd_r'][:, idx], 99, axis=0)
        check('A8 plain Rayleigh HD rate on border cells (reported)', True, f'{np.mean(plain):.2f}')
        R = np.array([r_border(of, ('left',), 10, 1.0) * r_hd(of, rng.uniform(0, 2 * np.pi), 3, 1.0) * 10
                      for _ in range(nper)])
        res, idx = run({4: of}, {4: R}, cfg, rng)
        ss = res['sessions'][4]
        check('A9 conjunctive border', frac(ss['classes']['border'], idx) >= 0.8, f"{frac(ss['classes']['border'], idx):.2f}")
        check('A9 conjunctive HD', frac(ss['classes']['hd'], idx) >= 0.8, f"{frac(ss['classes']['hd'], idx):.2f}")

    # A10 speed
    v = np.clip(np.nan_to_num(of['speed']), 0, 50)
    R = np.vstack([2 + rng.uniform(0.2, 0.4, (nper // 2, 1)) * v[None], 15 - rng.uniform(0.1, 0.25, (nper - nper // 2, 1)) * v[None]])
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
        if oo['null'] is None:
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
    if objs2:
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
    R = np.array([r_border(of, ('left',), 10, 8)] * 3 + [r_place(of, (60, 60), 10, 8)] * 3)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(1)
    cfg_e = copy.deepcopy(cfg)
    cfg_e['n_shifts'] = 30
    cfg_e1 = copy.deepcopy(cfg_e)
    cfg_e1['obj']['margin_cm'] = -1000.0
    ra, _ = run({4: of}, {4: R}, cfg_e, rng_a, objects_override={4: []})
    rb, _ = run({4: of}, {4: R}, cfg_e1, rng_b, objects_override={4: [dict(label='o', xy_cm=[60, 60], radius_cm=6)]})
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

    failed = [c for c in CHECKS if not c[1]]
    print(f'\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed in {time.time() - t0:.0f} s', flush=True)
    for name, _, det in failed:
        print(f'  FAILED {name}: {det}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
