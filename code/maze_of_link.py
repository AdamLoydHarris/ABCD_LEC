"""Link each mEC unit's open-field tuning to its maze tuning within one sort: head-direction coherence across
the two rigs, and open-field classes against maze metrics.

    python code/run_spatial.py --engine link --dataset mec --recday bp01_20240402_20240403      (sbatch)

The big field is filmed by another camera than the task maze, and the two cm frames are mirror images: the
maze pixel-to-cm affine has det < 0 in all 102 mEC maze sessions, the arena map det > 0 (Jacobian +0.017
cm/px at its centre). A counter-clockwise turn therefore raises the HD angle in one frame and lowers it in
the other. The reflection s is read from the calibrations and checked on the ear axis of every session;
the rotation between rigs is fitted per session pair from the HD cells and tested against a unit-shuffle
null.

Inputs: the recday's open-field shard (run `of_tag`), maze shard (run `maze_tag`) and views (maze HD needs
the time series). mEC only: this file is not mirrored to mFC_data/code/.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import spatial_common as sc                                           # noqa: E402

ENGINE = 'link'
ENGINE_VERSION = 1
TASK_TYPES = ('ABCD', 'ABCDE')
ARENA_TYPES = ('openfield', 'object')

DEFAULT_CFG = dict(
    seed=0, n_jobs=8,
    of_tag='8a49b0e2d93c', maze_tag='b705159d8440',
    n_shifts=500, min_shift_s=20.0, q_unit=99.0, q_stab=95.0,
    min_units=5, n_perm=10000, n_boot=2000, max_exact=8,
    # maze edge index: rate within 1.5 cm of a drop-off against rate >= 4.5 cm from it (bridges are 3-4 cm wide)
    edge=dict(near_cm=1.5, far_cm=4.5, min_samples=200),
    n_label_perm=10000,
)


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def arena_handedness(dataset_root: str) -> float:
    """Sign of the arena pixel-to-cm Jacobian at the floor centre. Lens undistortion is radial and keeps
    orientation (and needs cv2), so the projective part alone sets the sign."""
    from pathlib import Path
    sys.path.insert(0, os.path.join(HERE, 'preprocessing'))
    import arena_calibration as ac
    cal = ac.latest_calibration(Path(dataset_root), 'mec', 'bigfield')
    H = np.asarray(cal.H, float)

    def proj(px):
        h = H @ np.c_[px, np.ones(len(px))].T
        return cal._calib_to_arena((h[:2] / h[2]).T)
    p0 = np.asarray(cal.corners_px, float).mean(0)
    cm = proj(np.array([p0, p0 + [1.0, 0.0], p0 + [0.0, 1.0]]))
    return float(np.sign(np.linalg.det(np.column_stack([cm[1] - cm[0], cm[2] - cm[0]]))))


def maze_handedness(dataset_root: str) -> float:
    """Common sign of every mEC maze session's pixel-to-cm affine (geometry v2 QC table)."""
    import pandas as pd
    q = pd.read_csv(os.path.join(dataset_root, 'data', 'preprocessed', 'sleap', 'sleap_to_rois_qc_mec_v2.csv'))
    det = q['affine_det'].to_numpy(float)
    det = det[np.isfinite(det)]
    if not len(det) or (np.abs(det) < 1e-4).any() or len(np.unique(np.sign(det))) != 1:
        raise ValueError('maze affine determinants are missing, degenerate or of mixed sign')
    return float(np.sign(det[0]))


def ear_sign(view: dict) -> float:
    """Side of the left-to-right ear axis relative to heading in this session's frame: sign(mean sin(ear - hd))."""
    if view.get('hd') is None or view.get('hd_ear') is None:
        return np.nan
    d = np.sin(np.asarray(view['hd_ear'], float) - np.asarray(view['hd'], float))
    return float(np.sign(np.nanmean(d))) if np.isfinite(d).any() else np.nan


def reflection(views: dict, dataset_root: str) -> dict:
    """s = maze handedness x arena handedness, checked on the ear axis of every session; raises on a mismatch."""
    s_maze, s_arena = maze_handedness(dataset_root), arena_handedness(dataset_root)
    s = s_maze * s_arena
    ears = {k: ear_sign(v) for k, v in views.items()}
    em = {ears[k] for k, v in views.items() if v.get('session_type') in TASK_TYPES and np.isfinite(ears[k])}
    ea = {ears[k] for k, v in views.items() if v.get('session_type') in ARENA_TYPES and np.isfinite(ears[k])}
    if len(em) != 1 or len(ea) != 1 or em.pop() * ea.pop() != s:
        raise ValueError(f'ear-axis signs {ears} disagree with the calibration reflection s = {s:+.0f}')
    return dict(s=s, s_maze=s_maze, s_arena=s_arena, ear_signs=ears)


# ---------------------------------------------------------------------------
# maze head direction
# ---------------------------------------------------------------------------

def maze_hd(view: dict, ocfg: dict, cfg: dict, rng) -> dict:
    """pxd head-direction tuning of every unit in one maze session, with the open-field run's settings
    (position x 30 deg bins), split-half r, and each unit's own circular-shift p99 (R) and p95 (split)."""
    import of_lag_scan as ols
    import spatial_tuning as st
    if view.get('xy') is None or view.get('hd') is None:
        # bp02 03-19/21 has a maze session without tracking; link run f2d17144bfee crashed on it
        return dict(guard=dict(ok=False, reason='no XY or HD in this session'))
    counts, xy, hd, edges, guard = ols.session_inputs(view, ocfg)
    if hd is None:
        return dict(guard=guard)
    scn = ols.Scanner(xy, hd, edges, ocfg)
    T, N = counts.shape[1], counts.shape[0]
    half = np.arange(T) < T // 2
    h = ocfg['hd']
    masks = {'all': scn.moving, 'h1': scn.moving & half, 'h2': scn.moving & ~half}
    Ms = {k: sc.onehot(scn.jflat, m & (scn.jflat >= 0), scn.pxd_shape[0] * scn.pxd_shape[1]) for k, m in masks.items()}
    tj = {k: sc.occupancy_map(M, scn.pxd_shape) for k, M in Ms.items()}

    def factors(cnt):
        d = {}
        for k, M in Ms.items():
            nj = np.asarray((M.T @ cnt.T).T).reshape((N,) + scn.pxd_shape)
            d[k] = st.pxd(nj, tj[k], h['pxd_tol'], h['pxd_maxit'])[1]
        R, pref = st.vector_length(d['all'], scn.centres)
        split = sc.map_corr(d['h1'][:, :, None], d['h2'][:, :, None], min_bins=4)
        return R, pref, split, d['all']
    R, pref, split, d = factors(counts)
    nullR, nullS = [], []
    for k in sc.shift_offsets(T, cfg['n_shifts'], int(round(cfg['min_shift_s'] / sc.BIN_S)), rng):
        r_, _, s_, _ = factors(np.roll(counts, int(k), axis=1))
        nullR.append(r_)
        nullS.append(s_)
    thr_R = sc.unit_threshold(np.array(nullR), cfg['q_unit'])
    thr_S = sc.unit_threshold(np.array(nullS), cfg['q_stab'])
    return dict(guard=guard, R=R, pref=pref, split=split, d=d, thr_R=thr_R, thr_S=thr_S,
                hd_class=(R > thr_R) & (split > thr_S))


# ---------------------------------------------------------------------------
# coherence of preferred directions across sessions
# ---------------------------------------------------------------------------

def coherence(theta_b: np.ndarray, theta_a: np.ndarray, s: float, cfg: dict, rng) -> dict:
    """Rotation phi with theta_b = s * theta_a + phi, from units' preferred directions (radians): delta =
    wrap(theta_b - s theta_a); mean angle, R, bootstrap CI, leave-one-out error, R under the other reflection,
    and a unit-shuffle null on R (all permutations when n <= max_exact). Keeps the null R for pooling."""
    n = len(theta_a)
    out = dict(n=n)
    if n < cfg['min_units']:
        return out
    z = np.exp(1j * (theta_b - s * theta_a))
    mz = z.mean()
    phi, R = float(np.angle(mz)), float(abs(mz))
    loo = np.array([abs(np.angle(z[u] * np.exp(-1j * np.angle(np.delete(z, u).mean())))) for u in range(n)])
    boot = z[rng.integers(0, n, (cfg['n_boot'], n))].mean(1)
    dev = np.angle(boot * np.exp(-1j * phi))
    if n <= cfg['max_exact']:
        perms = np.array(list(itertools.permutations(range(n))))
    else:
        perms = np.array([rng.permutation(n) for _ in range(cfg['n_perm'])])
    nullR = np.abs(np.exp(1j * (theta_b[None, :] - s * theta_a[perms])).mean(1))
    R_other = float(abs(np.exp(1j * (theta_b + s * theta_a)).mean()))
    out.update(phi_deg=float(np.degrees(phi)), R=R, ci_deg=tuple(np.degrees(phi + np.percentile(dev, [2.5, 97.5]))),
               loo_median_err_deg=float(np.degrees(np.median(loo))), p=float((1 + (nullR >= R).sum()) / (1 + len(nullR))),
               R_other_reflection=R_other, reflection_flag=bool(R_other >= R), null_R=nullR.astype(np.float32))
    return out


def v_test(phi_deg: float, R: float, n: int, mu_deg: float = 0.0) -> tuple:
    """Rayleigh V-test against a known direction: (u, one-sided p)."""
    from scipy.stats import norm
    u = np.sqrt(2 * n) * R * np.cos(np.radians(phi_deg - mu_deg))
    return float(u), float(norm.sf(u))


def wrap_deg(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------------------
# maze edge index
# ---------------------------------------------------------------------------

def edge_index(views: dict, mcfg: dict, cfg: dict, rng) -> np.ndarray:
    """(N,) log((rate near a drop-off + 0.1) / (rate far from it + 0.1)) on moving in-trial maze samples,
    near and far samples speed-matched (corners and drop-offs are run slower)."""
    import maze_spatial as ms
    asm = ms.assemble(views, mcfg)
    c = asm['cat']
    d = ms.boundary_distance(asm, np.nan_to_num(c['xy']))
    e = cfg['edge']
    base = c['moving'] & c['intrial'] & np.isfinite(c['speed'])
    near, far = base & (d <= e['near_cm']), base & (d >= e['far_cm'])
    idx = np.nonzero(near | far)[0]
    idx = ms.speed_matched(idx, near.astype(int), c['speed'], mcfg, rng)
    a, b = idx[near[idx]], idx[far[idx]]
    if len(a) < e['min_samples'] or len(b) < e['min_samples']:
        return np.full(asm['N'], np.nan)
    rn = c['counts'][:, a].mean(1) / sc.BIN_S
    rf = c['counts'][:, b].mean(1) / sc.BIN_S
    return np.log((rn + 0.1) / (rf + 0.1))


# ---------------------------------------------------------------------------
# recday entry point
# ---------------------------------------------------------------------------

def _label_test(x: np.ndarray, lab: np.ndarray, rng, n: int, direction: int = 1) -> dict:
    """direction x (median of x with the label - median without), against shuffles of the label among units
    (one-sided); keeps the null for pooling across recdays."""
    ok = np.isfinite(x)
    x, lab = x[ok], lab[ok].astype(bool)
    if lab.sum() < 2 or (~lab).sum() < 2:
        return dict(n_with=int(lab.sum()), n_without=int((~lab).sum()), direction=direction)

    def f(l_):
        return direction * float(np.median(x[l_]) - np.median(x[~l_]))
    real = f(lab)
    null = np.array([f(rng.permutation(lab)) for _ in range(n)], np.float32)
    return dict(n_with=int(lab.sum()), n_without=int((~lab).sum()), direction=direction, real=real,
                p=float((1 + (null >= real).sum()) / (1 + n)), null=null)


def _spearman_test(x: np.ndarray, y: np.ndarray, rng, n: int) -> dict:
    from scipy.stats import rankdata
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return dict(n=int(ok.sum()))
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    real = float(np.corrcoef(rx, ry)[0, 1])
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)], np.float32)
    return dict(n=int(ok.sum()), real=real, p=float((1 + (null >= real).sum()) / (1 + n)), null=null)


def run_link_recday(views: dict, recday: str, units, cfg: dict, of_shard: dict, maze_shard: dict,
                    dataset_root: str) -> dict:
    """Every link analysis for one recday; returns the shard dict (no file I/O here)."""
    import maze_spatial as ms
    cfg = copy.deepcopy(cfg)
    rng = np.random.default_rng(cfg['seed'])
    for name, shard in (('open-field', of_shard), ('maze', maze_shard)):
        if len(shard['units']) != len(units):
            raise ValueError(f'{name} shard has {len(shard["units"])} units, the views {len(units)}')
        if 'cluster_id' in units and 'cluster_id' in shard['units'] and \
                not np.array_equal(shard['units']['cluster_id'].to_numpy(float), units['cluster_id'].to_numpy(float), equal_nan=True):
            raise ValueError(f'{name} shard cluster ids differ from the views')
    ocfg = of_shard['stamp']['cfg']
    mcfg = copy.deepcopy(ms.DEFAULT_CFG)
    mcfg.update({k: v for k, v in maze_shard['stamp']['cfg'].items() if k in mcfg})
    refl = reflection(views, dataset_root)
    s = refl['s']

    # maze HD per session, and the open-field HD of every arena session
    mhd = {k: maze_hd(v, ocfg, cfg, rng) for k, v in sorted(views.items()) if v.get('session_type') in TASK_TYPES}
    of = of_shard['sessions']
    ahd = {a: dict(pref=np.asarray(x['details']['pxd_pref'], float), hd_class=np.asarray(x['classes']['hd'], bool),
                   eligible=np.asarray(x['eligible'], bool), ok=bool(x['hd_guard'].get('ok', False)))
           for a, x in of.items()}

    # maze -> arena pairs: units HD-classed in both; sensitivity: open-field HD units only
    pairs = []
    for m, mh in mhd.items():
        if 'R' not in mh:
            continue
        for a, ah in ahd.items():
            if not ah['ok']:
                continue
            for sel_name, sel in (('hd_both', ah['hd_class'] & mh['hd_class'] & ah['eligible']),
                                  ('hd_open_field', ah['hd_class'] & ah['eligible'])):
                u = np.nonzero(sel & np.isfinite(ah['pref']) & np.isfinite(mh['pref']))[0]
                res = coherence(ah['pref'][u], mh['pref'][u], s, cfg, rng)
                pairs.append(dict(maze=m, arena=a, selection=sel_name, units=u, **res))

    # maze -> maze pairs (same frame: s = +1) with a V-test against no rotation
    mpairs = []
    keys = [m for m, mh in mhd.items() if 'R' in mh]
    for m1, m2 in itertools.combinations(keys, 2):
        sel = mhd[m1]['hd_class'] & mhd[m2]['hd_class']
        u = np.nonzero(sel & np.isfinite(mhd[m1]['pref']) & np.isfinite(mhd[m2]['pref']))[0]
        res = coherence(mhd[m2]['pref'][u], mhd[m1]['pref'][u], 1.0, cfg, rng)
        if 'phi_deg' in res:
            res['v_u'], res['v_p'] = v_test(res['phi_deg'], res['R'], res['n'])
            res['flip_flag'] = bool(abs(res['phi_deg']) > 45)
        mpairs.append(dict(maze1=m1, maze2=m2, units=u, **res))

    # closure: the arena-to-arena HD shift of the open-field rotation test, and maze-to-maze shifts
    closure = []
    phi = {(p['maze'], p['arena']): p['phi_deg'] for p in pairs if p['selection'] == 'hd_both' and 'phi_deg' in p}
    for (a1, a2), rot in of_shard.get('rotations', {}).items():
        hdr = rot.get('hd') if isinstance(rot, dict) else None
        if not hdr or 'mean_shift_deg' not in hdr:
            continue
        for m in keys:
            if (m, a1) in phi and (m, a2) in phi:
                closure.append(dict(kind='arena pair', maze=m, a1=a1, a2=a2, expected_deg=hdr['mean_shift_deg'],
                                    observed_deg=wrap_deg(phi[(m, a2)] - phi[(m, a1)]),
                                    error_deg=wrap_deg(phi[(m, a2)] - phi[(m, a1)] - hdr['mean_shift_deg'])))
    for mp in mpairs:
        if 'phi_deg' not in mp:
            continue
        for a in ahd:
            if (mp['maze1'], a) in phi and (mp['maze2'], a) in phi:
                # theta_a = s theta_m + phi(m, a) and theta_m2 = theta_m1 + phi(m1, m2) give
                # phi(m2, a) - phi(m1, a) = -s phi(m1, m2)
                exp_ = wrap_deg(-s * mp['phi_deg'])
                obs = wrap_deg(phi[(mp['maze2'], a)] - phi[(mp['maze1'], a)])
                closure.append(dict(kind='maze pair', maze=f"{mp['maze1']}-{mp['maze2']}", a1=a, a2=a, expected_deg=exp_,
                                    observed_deg=obs, error_deg=wrap_deg(obs - exp_)))

    # class links per unit
    su = maze_shard['single']
    N = len(units)
    of_hd = np.zeros(N, bool)
    of_border = np.zeros(N, bool)
    of_elig = np.zeros(N, bool)
    speed_f = []
    si_p = []
    for a, x in of.items():
        el = np.asarray(x['eligible'], bool)
        of_elig |= el
        of_hd |= np.asarray(x['classes']['hd'], bool) & el
        of_border |= np.asarray(x['classes']['border'], bool) & el
        speed_f.append(np.where(el, np.asarray(x['real']['speed_f'], float), np.nan))
        si_p.append(np.where(el, np.asarray(x['real']['si_p'], float), np.nan))
    speed_f = np.nanmean(np.array(speed_f), axis=0) if speed_f else np.full(N, np.nan)
    si_p = np.nanmean(np.array(si_p), axis=0) if si_p else np.full(N, np.nan)
    maze_R = np.nanmean(np.array([mh['R'] for mh in mhd.values() if 'R' in mh]), axis=0) if keys else np.full(N, np.nan)
    edge = edge_index({k: v for k, v in views.items() if v.get('session_type') in TASK_TYPES}, mcfg, cfg, rng)
    node = np.asarray(su['type_class']['node'], bool)
    bridge = np.asarray(su['type_class']['bridge'], bool)
    nl = cfg['n_label_perm']
    el = of_elig
    links = dict(
        maze_R_by_of_hd=_label_test(np.where(el, maze_R, np.nan), of_hd, rng, nl),
        edge_by_of_border=_label_test(np.where(el, edge, np.nan), of_border, rng, nl),
        # a behavioural account of node cells predicts slow-speed cells: node cells' open-field speed
        # factor below the others' (direction -1), bridge cells' above it
        of_speed_by_maze_node=_label_test(np.where(el, speed_f, np.nan), node, rng, nl, direction=-1),
        of_speed_by_maze_bridge=_label_test(np.where(el, speed_f, np.nan), bridge, rng, nl, direction=+1),
        of_si_vs_maze_stab=_spearman_test(np.where(el, si_p, np.nan), np.asarray(su['stab21'], float), rng, nl),
    )
    unit_rows = dict(of_eligible=el, of_hd=of_hd, of_border=of_border, of_speed_f=speed_f, of_si_p=si_p,
                     maze_R=maze_R, maze_hd_any=np.any([mh['hd_class'] for mh in mhd.values() if 'R' in mh], axis=0)
                     if keys else np.zeros(N, bool), edge_index=edge, maze_node=node, maze_bridge=bridge,
                     maze_stab21=np.asarray(su['stab21'], float))
    return dict(recday=recday, engine=ENGINE, engine_version=ENGINE_VERSION, units=units, reflection=refl,
                maze_hd={k: {kk: vv for kk, vv in v.items() if kk != 'd'} for k, v in mhd.items()},
                pairs=pairs, maze_pairs=mpairs, closure=closure, links=links, unit_rows=unit_rows,
                inputs=dict(of_tag=cfg['of_tag'], maze_tag=cfg['maze_tag']))


# ---------------------------------------------------------------------------
# merge and mouse-level pooling
# ---------------------------------------------------------------------------

def merge_tables(shards: dict) -> dict:
    """DataFrames: hd_pairs, maze_pairs, closure, link_tests, unit_links, maze_hd."""
    import pandas as pd
    P, MP, CL, LT, UR, MH = [], [], [], [], [], []
    for rd, sh in sorted(shards.items()):
        mouse = rd.split('_')[0]
        for p in sh['pairs']:
            P.append(dict(recday=rd, mouse=mouse, **{k: v for k, v in p.items() if k not in ('units', 'null_R')}))
        for p in sh['maze_pairs']:
            MP.append(dict(recday=rd, mouse=mouse, **{k: v for k, v in p.items() if k not in ('units', 'null_R')}))
        for c in sh['closure']:
            CL.append(dict(recday=rd, mouse=mouse, **c))
        for name, t in sh['links'].items():
            LT.append(dict(recday=rd, mouse=mouse, test=name, **{k: v for k, v in t.items() if k != 'null'}))
        ur = sh['unit_rows']
        for u in range(len(sh['units'])):
            UR.append(dict(recday=rd, mouse=mouse, unit=u, **{k: (v[u] if hasattr(v, '__len__') else v) for k, v in ur.items()}))
        for m, h in sh['maze_hd'].items():
            if 'R' not in h:
                MH.append(dict(recday=rd, mouse=mouse, session=m, guard_ok=False))
                continue
            for u in range(len(h['R'])):
                MH.append(dict(recday=rd, mouse=mouse, session=m, guard_ok=True, unit=u, R=h['R'][u],
                               pref_deg=float(np.degrees(h['pref'][u])), split=h['split'][u], hd_class=bool(h['hd_class'][u])))
    return dict(hd_pairs=pd.DataFrame(P), maze_pairs=pd.DataFrame(MP), closure=pd.DataFrame(CL),
                link_tests=pd.DataFrame(LT), unit_links=pd.DataFrame(UR), maze_hd=pd.DataFrame(MH))


def mouse_pool(shards: dict, kind: str, key, exclude_mice=('bp02',)) -> list:
    """Per mouse: mean real statistic over its pairs or recdays against null draw i averaged over the same
    pairs or recdays (nulls are independent between pairs). kind 'pairs' pools coherence R of the hd_both
    maze->arena pairs; kind 'links' pools the link test named `key`."""
    by = {}
    for rd, sh in shards.items():
        mouse = rd.split('_')[0]
        if mouse in exclude_mice:
            continue
        items = ([p for p in sh['pairs'] if p['selection'] == key and 'R' in p] if kind == 'pairs'
                 else [sh['links'][key]] if 'real' in sh['links'].get(key, {}) else [])
        for it in items:
            real = it['R'] if kind == 'pairs' else it['real']
            null = it['null_R'] if kind == 'pairs' else it['null']
            by.setdefault(mouse, []).append((real, np.asarray(null, float)))
    rows = []
    for mouse, lst in sorted(by.items()):
        n = min(len(x[1]) for x in lst)
        x = float(np.mean([r for r, _ in lst]))
        y = np.mean([nl[:n] for _, nl in lst], axis=0)
        rows.append(dict(mouse=mouse, n_items=len(lst), n_null=n, real=x, null_mean=float(y.mean()),
                         p=float((1 + (y >= x).sum()) / (1 + n))))
    return rows
