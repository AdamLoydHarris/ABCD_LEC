"""Synthetic gate for the maze-open-field link (`maze_of_link.run_link_recday`), on real mEC recdays.

Planted units ride the real maze and arena tracks of one recday. The planted counts go through the
open-field engine (`spatial_tuning.run_of_recday`), the maze engine (`maze_spatial.run_maze_recday`)
and the link, as the data do.

    sbatch --job-name=sp_link_syn --partition=cpu_backfill sbatch_files/spatial.sbatch code/maze_of_link_synthetics.py --quick

| # | planted | must hold |
|---|---|---|
| L1 | 20 HD cells, arena preference = -maze preference + 70 deg | phi within 10 deg of 70; p <= 0.01; R above the other reflection's |
| L2 | 20 HD cells with independent preferences (one pipeline draw + 4 statistic draws) | p > 0.05 in >= 4 of 5 |
| L3 | the L1 maze preferences with an unreflected arena (+maze + 70) | flagged: the other reflection fits better |
| L4 | L1 HD cells and 15 maze place cells | maze HD class >= 80 % of HD cells, <= 10 % of place cells |
| L5 | one maze session with every HD preference turned 180 deg | its maze pairs flagged near 180, other pairs near 0; closure error <= 15 deg |
| L6 | 12 slow-speed cells (P1) / 15 location-only node cells (P2) | node-speed link p <= 0.05 in P1, > 0.05 in P2 |
| L7 | guards | mismatched unit tables, and an ear axis contradicting the calibrations, raise |
| L8 | bp02 03-19/21 as recorded, real shards | runs to the end; its maze sessions without XY or HD are skipped |

mEC only; this file is not mirrored to mFC_data/code/.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

RECDAYS = ('bp01_20240402_20240403', 'bp04_20240321_20240323')
#: a recday with an untracked maze session (link run f2d17144bfee crashed on it)
UNTRACKED = 'bp02_20240319_20240321'
CHECKS = []


def check(name: str, passed: bool, detail: str) -> None:
    print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}", flush=True)
    CHECKS.append((name, bool(passed), detail))


def vm(theta: np.ndarray, mu: float, kappa: float = 2.0, peak: float = 15.0, base: float = 0.5) -> np.ndarray:
    """Von Mises HD tuning in Hz; NaN heading gives the baseline."""
    th = np.where(np.isfinite(theta), theta, mu + np.pi)
    return base + peak * np.exp(kappa * (np.cos(th - mu) - 1))


def plant(views: dict, rng, kind: str, flip_session=None, phi=np.radians(70.0), s=-1.0) -> tuple:
    """Planted counts for every session of a recday: P1 = coherent HD cells (+ a 180 deg flip in one maze
    session), maze place cells and slow-speed cells; P2 = HD cells with independent arena preferences and
    location-only node cells. Returns (views with counts, labels)."""
    import spatial_common as sc
    n_hd = 20
    mu_m = rng.uniform(0, 2 * np.pi, n_hd)
    mu_a = (s * mu_m + phi) if kind == 'P1' else rng.uniform(0, 2 * np.pi, n_hd)
    maze_keys = [k for k, v in views.items() if v.get('session_type') in ('ABCD', 'ABCDE')]
    first = views[maze_keys[0]]
    fin = np.isfinite(first['xy']).all(1)
    place_c = first['xy'][fin][rng.integers(fin.sum(), size=15)]
    labels = ['hd'] * n_hd + (['place'] * 15 + ['slow'] * 12 if kind == 'P1' else ['node'] * 15)
    out = {}
    for k, v in views.items():
        T = min(v['counts'].shape[1], len(v['xy']))
        xy, hd = np.asarray(v['xy'][:T], float), (np.asarray(v['hd'][:T], float) if v['hd'] is not None else np.full(T, np.nan))
        speed = np.nan_to_num(sc.speed_cm_s(xy))
        maze = k in maze_keys
        rates = []
        for u in range(n_hd):
            mu = (mu_m[u] + (np.pi if k == flip_session else 0.0)) if maze else mu_a[u]
            rates.append(vm(hd, mu))
        if kind == 'P1':
            for u in range(15):
                if maze:
                    d2 = np.nansum((xy - place_c[u]) ** 2, axis=1)
                    rates.append(0.5 + 15 * np.exp(-np.where(np.isfinite(xy).all(1), d2, np.inf) / (2 * 6.0 ** 2)))
                else:
                    rates.append(np.full(T, 2.0))
            for u in range(12):
                rates.append(1.0 + 12.0 * np.exp(-speed / 5.0))
        else:
            for u in range(15):
                if maze:
                    loc = np.asarray(v['locs'][:T])
                    rates.append(np.where((loc >= 1) & (loc <= 9), 8.0, 1.0))
                else:
                    rates.append(np.full(T, 2.0))
        w = dict(v)
        w['counts'] = rng.poisson(np.array(rates) * sc.BIN_S).astype(np.float32)
        w['xy'], w['hd'] = v['xy'][:T], (v['hd'][:T] if v['hd'] is not None else None)
        if v.get('hd_ear') is not None:
            w['hd_ear'] = v['hd_ear'][:T]
        if v.get('locs') is not None:
            w['locs'] = v['locs'][:T]
        out[k] = w
    return out, np.array(labels), mu_m, mu_a


def run_population(views: dict, recday: str, ocfg: dict, mcfg: dict, lcfg: dict, root: str) -> tuple:
    """Planted views through the open-field engine, the maze engine and the link."""
    import maze_of_link as ml
    import maze_spatial as ms
    import spatial_tuning as st
    n = next(iter(views.values()))['counts'].shape[0]
    units = pd.DataFrame(dict(unit=np.arange(n), width_ms=0.6, dup_partner=-1, cluster_id=np.nan, region='SYN'))
    of_shard = st.run_of_recday(views, recday, units, ocfg, wall_exclusions={})
    of_shard['stamp'] = dict(cfg={k: v for k, v in ocfg.items() if k != 'n_jobs'})
    maze_shard = ms.run_maze_recday(views, recday, units, mcfg)
    maze_shard['stamp'] = dict(cfg={k: v for k, v in mcfg.items() if k != 'n_jobs'})
    link = ml.run_link_recday(views, recday, units, lcfg, of_shard, maze_shard, root)
    return link, of_shard, maze_shard, units


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='maze-open-field link synthetic gate')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n-jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = ap.parse_args(argv)
    import spatial_io as sio
    sio.set_dataset('mec')
    import maze_of_link as ml
    import maze_spatial as ms
    import recday_registry as rr
    import spatial_tuning as st
    t0 = time.time()
    root = str(rr.DATASETS['mec']['root'])
    q = args.quick
    ocfg = copy.deepcopy(st.DEFAULT_CFG)
    ocfg.update(n_jobs=args.n_jobs, n_shifts=100 if q else 500)
    ocfg['rotation'].update(n_perm=100, n_boot=200, n_shift_object=50)
    mcfg = copy.deepcopy(ms.DEFAULT_CFG)
    mcfg['n_jobs'] = args.n_jobs
    mcfg['stats'].update(n_shifts=50 if q else 200, n_label_perm=100 if q else 500)
    mcfg['null'].update(n_sim_rsa=0, n_sim_decode=0, n_sim_planes=0)
    lcfg = copy.deepcopy(ml.DEFAULT_CFG)
    lcfg.update(n_shifts=100 if q else 500, n_perm=2000 if q else 10000, n_label_perm=2000 if q else 10000)
    rng = np.random.default_rng(20261012)
    all_views = sio.load('mec', list(RECDAYS) + [UNTRACKED], 'v4')
    print(f'loaded {list(all_views)} in {time.time() - t0:.0f} s', flush=True)

    rd = RECDAYS[0]
    views = all_views[rd]
    maze_keys = sorted(k for k, v in views.items() if v.get('session_type') in ('ABCD', 'ABCDE') and v['hd'] is not None)
    # the longest session: short ones (< 5 trials, dropped by the maze engine) classed too few HD cells to
    # pair with, so a flip there left L5 with 0 pairs (run 3752835)
    flip = max(maze_keys, key=lambda k: views[k]['counts'].shape[1])
    # P1: coherent HD cells with one flipped maze session, maze place cells, slow-speed cells
    v1, lab1, mu_m, mu_a = plant(views, rng, 'P1', flip_session=flip)
    link1, of1, mz1, units = run_population(v1, rd, ocfg, mcfg, lcfg, root)
    hd = lab1 == 'hd'
    pairs = [p for p in link1['pairs'] if p['selection'] == 'hd_both' and p['maze'] != flip and 'phi_deg' in p]
    errs = [abs(ml.wrap_deg(p['phi_deg'] - 70.0)) for p in pairs]
    ps = [p['p'] for p in pairs]
    better = [p['R'] > p['R_other_reflection'] for p in pairs]
    check('L1 coherent HD: rotation recovered within 10 deg in every unflipped maze-arena pair',
          len(pairs) > 0 and max(errs) <= 10, f'{len(pairs)} pairs; max error {max(errs) if errs else np.nan:.1f} deg; '
          f"n units {[p['n'] for p in pairs]}")
    check('L1 coherent HD: unit-shuffle p <= 0.01 and R above the other reflection', len(pairs) > 0 and max(ps) <= 0.01 and all(better),
          f'max p {max(ps) if ps else np.nan:.4f}; R above other reflection in {sum(better)}/{len(better)}')
    mh = [h for h in link1['maze_hd'].values() if 'R' in h]
    frac_hd = float(np.mean([h['hd_class'][hd].mean() for h in mh]))
    frac_pl = float(np.mean([h['hd_class'][lab1 == 'place'].mean() for h in mh]))
    check('L4 maze HD class: HD cells >= 80 %, place cells <= 10 %', frac_hd >= 0.8 and frac_pl <= 0.1,
          f'HD cells {frac_hd:.2f}, place cells {frac_pl:.2f} over {len(mh)} maze sessions')
    fl = [m for m in link1['maze_pairs'] if 'phi_deg' in m]
    with_flip = [abs(ml.wrap_deg(m['phi_deg'] - 180.0)) for m in fl if flip in (m['maze1'], m['maze2'])]
    without = [abs(ml.wrap_deg(m['phi_deg'])) for m in fl if flip not in (m['maze1'], m['maze2'])]
    check(f'L5 flipped maze session (s{flip}): its pairs near 180 deg, the others near 0',
          len(with_flip) > 0 and len(without) > 0 and max(with_flip) <= 20 and max(without) <= 20,
          f'max |phi - 180| {max(with_flip) if with_flip else np.nan:.1f} over {len(with_flip)}; max |phi| {max(without) if without else np.nan:.1f} over {len(without)}')
    cl = [abs(c['error_deg']) for c in link1['closure']]
    check('L5 closure of rotations across sessions', len(cl) > 0 and max(cl) <= 15, f'max error {max(cl) if cl else np.nan:.1f} deg over {len(cl)}')
    t6 = link1['links']['of_speed_by_maze_node']
    check('L6 slow-speed cells: node-speed link detected', t6.get('p', 1.0) <= 0.05,
          f"node cells {t6.get('n_with')}, p {t6.get('p', np.nan):.4f}")
    # L3: the same maze preferences with an unreflected arena must be flagged
    m_pref = np.nanmean(np.array([h['pref'] for h in mh]), axis=0) if mh else np.full(len(lab1), np.nan)
    u = np.nonzero(hd & np.isfinite(m_pref))[0]
    res3 = ml.coherence(np.mod(m_pref[u] + np.radians(70.0), 2 * np.pi), m_pref[u], -1.0, lcfg, rng)
    check('L3 unreflected arena preferences are flagged', res3.get('reflection_flag', False),
          f"R {res3.get('R', np.nan):.2f} vs other reflection {res3.get('R_other_reflection', np.nan):.2f}")

    # P2: independent preferences and location-only node cells
    v2, lab2, _, _ = plant(views, rng, 'P2')
    link2, _, _, _ = run_population(v2, rd, ocfg, mcfg, lcfg, root)
    p2 = [p['p'] for p in link2['pairs'] if p['selection'] == 'hd_both' and 'p' in p]
    extra = []
    hd2 = lab2 == 'hd'
    mh2 = [h for h in link2['maze_hd'].values() if 'R' in h]
    m2 = np.nanmean(np.array([h['pref'] for h in mh2]), axis=0)
    u2 = np.nonzero(hd2 & np.isfinite(m2))[0]
    for _ in range(4):
        extra.append(ml.coherence(rng.uniform(0, 2 * np.pi, len(u2)), m2[u2], -1.0, lcfg, rng)['p'])
    null_ok = [x > 0.05 for x in [float(np.median(p2)) if p2 else np.nan] + extra]
    check('L2 independent preferences: no coherence in >= 4 of 5 draws', sum(null_ok) >= 4,
          f'pipeline median p {np.median(p2) if p2 else np.nan:.3f} over {len(p2)} pairs; statistic draws {np.round(extra, 3).tolist()}')
    t6b = link2['links']['of_speed_by_maze_node']
    check('L6 location-only node cells: node-speed link not detected', t6b.get('p', 1.0) > 0.05,
          f"node cells {t6b.get('n_with')}, p {t6b.get('p', np.nan):.4f}")

    # L7 guards
    try:
        ml.run_link_recday(v1, rd, units.iloc[:-1], lcfg, of1, mz1, root)
        ok = False
    except ValueError:
        ok = True
    check('L7 mismatched unit tables raise', ok, 'ValueError' if ok else 'no error')
    bad = {k: dict(w) for k, w in v1.items()}
    a = next(k for k, w in bad.items() if w.get('session_type') in ('openfield', 'object'))
    bad[a]['hd_ear'] = 2 * bad[a]['hd'] - bad[a]['hd_ear']          # mirror the ear axis about the heading
    try:
        ml.reflection(bad, root)
        ok = False
    except ValueError:
        ok = True
    check('L7 an ear axis contradicting the calibrations raises', ok, 'ValueError' if ok else 'no error')

    # L8 the recorded recday with an untracked maze session, through the link with its real shards
    import pickle
    v8 = all_views[UNTRACKED]
    un = sorted(k for k, w in v8.items() if w.get('session_type') in ('ABCD', 'ABCDE')
                and (w.get('xy') is None or w.get('hd') is None))
    sroot = sio.output_root('mec')
    of8 = pickle.load(open(os.path.join(sroot, 'of', lcfg['of_tag'], 'shards', f'{UNTRACKED}.pkl'), 'rb'))
    mz8 = pickle.load(open(os.path.join(sroot, 'maze', lcfg['maze_tag'], 'shards', f'{UNTRACKED}.pkl'), 'rb'))
    units8 = sio.unit_table('mec', UNTRACKED, next(iter(v8.values()))['counts'].shape[0])
    try:
        out8 = ml.run_link_recday(v8, UNTRACKED, units8, lcfg, of8, mz8, root)
        ok, det = len(un) > 0 and all('R' not in out8['maze_hd'][k] for k in un), ''
    except Exception as e:                                          # noqa: BLE001
        ok, det = False, f'{type(e).__name__}: {e}'
    check('L8 a recday with an untracked maze session runs; those sessions are skipped', ok,
          f'untracked maze sessions {un} {det}')

    failed = [x for x in CHECKS if not x[1]]
    print(f'\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed in {time.time() - t0:.0f} s', flush=True)
    for name, _, det in failed:
        print(f'  FAILED {name}: {det}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
