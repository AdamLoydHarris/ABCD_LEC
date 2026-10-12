"""Spatial structure on the 3x3 maze: one pipeline for mEC, LEC and PFC (user rule 2026-10-09).

Single units: 21-location maps (Hz) per session, cross-task stability, Skaggs information, generalised
types (corner / cardinal / centre / node / bridge templates), 2-D maps in a canonical maze frame with
the bridge-safe smoothers. Population: RSA on 9 nodes and 21 locations against Euclidean, geodesic,
degree and place-type models; SVM decoders (held-out corner vs cardinal, held-out node vs bridge,
held-out task location with error structure); lasso location planes per goal-progress bin. Every
population statistic is also computed on simulated populations fitted to the recday's own maps (the
place-code null), its random-centre and field-shuffle variants, a speed-cell null and a boundary null.

Two recipes run in the same job on every dataset: 'decided' (per-unit z-scoring, speed matching,
cross-validated RDMs, disjoint controls) and 'thesis' (raw rates, a speed threshold, the three-model
RSA, overlapping halves and the largest principal angle, as the thesis code computed them).

Entry point: `run_maze_recday(views, recday, units, cfg)` -> shard dict.
This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).
"""

from __future__ import annotations

import copy
import itertools
import os
import sys
import warnings

import numpy as np
from scipy import optimize
from scipy.linalg import subspace_angles
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import spatial_common as sc                                           # noqa: E402

ENGINE = 'maze'
ENGINE_VERSION = 1

#: Maze edges in `Locs_raw` code order 10..21 (build_data_dic.MAZE_EDGES).
EDGES = ((1, 2), (2, 3), (1, 4), (2, 5), (3, 6), (4, 5), (5, 6), (4, 7), (5, 8), (6, 9), (7, 8), (8, 9))
CORNER, CARDINAL, CENTRE = (1, 3, 7, 9), (2, 4, 6, 8), (5,)
NODE_TYPE = {**{k: 'corner' for k in CORNER}, **{k: 'cardinal' for k in CARDINAL}, 5: 'centre'}
DEGREE = {1: 2, 3: 2, 7: 2, 9: 2, 2: 3, 4: 3, 6: 3, 8: 3, 5: 4}

DEFAULT_CFG = dict(
    # min_trials 5, not 10: at 10 LEC keeps 136/191 sessions and loses ly05 06-18/19 (8/13/3/0/6/8/1/5
    # trials); at 5 it keeps 164/191, PFC 263/270 (measured 2026-10-09)
    seed=0, n_jobs=8, min_trials=5, min_speed=2.5, gp_bins=5,
    lattice=dict(spacing_cm=19.0, origin_cm=5.0, margin_cm=8.0, max_resid_cm=4.0),
    maps=dict(bin_cm=1.0, sigmas=(1.5, 3.0), occ_threshold_s=0.1, min_loc_occ_s=1.0),
    stats=dict(n_shifts=500, min_shift_s=20.0, q_unit=99.0, q_stab=95.0, n_label_perm=1000),
    speed_match=dict(bin_cms=2.5, max_cms=60.0),
    rsa=dict(n_perm=1000, max_splits=20, min_samples=10, min_units=3),
    decode=dict(bin_bins=20, C=1.0, min_frac_same_loc=0.8, n_perm=35, match_speed=True),
    # thesis recipe as pfc_rsa / pfc_decoding computed it; px/bin speed cuts converted at ~20 px/cm
    thesis=dict(C=1.5, speed_cc_cms=6.0, speed_nb_cms=2.0),
    planes=dict(bin_bins=10, n_lags=20, alpha=0.005, n_reps=5, ks=(2, 3), max_task_splits=4),
    # place_shuffled (user's choice 2026-10-12): the bar for generalisation claims. The fitted null kept
    # 3 of a planted corner cell's 4 corners and absorbed it (gate M3, LEC: degree 0.81 vs p95 0.88)
    null=dict(n_sim_rsa=200, n_sim_decode=20, n_sim_planes=10, max_fields=3,
              kinds=dict(rsa=('place', 'place_random', 'place_shuffled', 'speed', 'boundary'),
                         decode=('place', 'place_random', 'place_shuffled', 'speed', 'boundary'),
                         planes=('place', 'place_random', 'place_shuffled'))),
)
#: null kinds of run 21c552de331e, seeded first and in this order so their numbers do not change
V1_NULL_KINDS = ('boundary', 'place', 'place_random', 'speed')


# ---------------------------------------------------------------------------
# geometry of the maze
# ---------------------------------------------------------------------------

def lattice_xy(cfg: dict) -> np.ndarray:
    """(21, 2) canonical cm positions of locations 1..21: nodes on the lattice, bridges at midpoints."""
    L = cfg['lattice']
    p = np.zeros((22, 2))
    for k in range(1, 10):
        p[k] = L['origin_cm'] + L['spacing_cm'] * np.array([(k - 1) % 3, (k - 1) // 3])
    for i, (a, b) in enumerate(EDGES):
        p[10 + i] = (p[a] + p[b]) / 2
    return p[1:]


def geodesic(cfg: dict) -> np.ndarray:
    """(21, 21) shortest-path distance along the maze (bridge midpoints half a step from their nodes)."""
    S = cfg['lattice']['spacing_cm']
    D = np.full((21, 21), np.inf)
    np.fill_diagonal(D, 0)
    for i, (a, b) in enumerate(EDGES):
        m = 10 + i
        for n in (a, b):
            D[m - 1, n - 1] = D[n - 1, m - 1] = S / 2
    for k in range(21):                      # Floyd-Warshall on 21 locations
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def model_rdms(cfg: dict) -> dict:
    """Model RDM matrices: 9-node E, G, D, P and 21-location E, G, NB (node vs bridge)."""
    p = lattice_xy(cfg)
    E = np.linalg.norm(p[:, None] - p[None], axis=2)
    G = geodesic(cfg)
    deg = np.array([DEGREE[k] for k in range(1, 10)], float)
    typ = np.array([0 if k in CORNER else 1 if k in CARDINAL else 2 for k in range(1, 10)])
    nb = np.r_[np.zeros(9), np.ones(12)]
    return dict(E9=E[:9, :9], G9=G[:9, :9], D9=np.abs(deg[:, None] - deg[None]), P9=(typ[:, None] != typ[None]).astype(float),
                E21=E, G21=G, NB21=(nb[:, None] != nb[None]).astype(float))


def diagnostic_pairs() -> dict:
    """Location pairs (0-based) where Euclidean and geodesic distance disagree, with their controls.

    9 nodes: diagonal neighbours (E = 1.41 S, G = 2 S) vs two-step straight pairs (E = G = 2 S).
    21 locations: parallel bridges across a hole (E = S, G = 2 S) vs collinear neighbouring bridges (E = G = S).
    """
    diag = [(0, 4), (1, 3), (1, 5), (2, 4), (3, 7), (4, 6), (4, 8), (5, 7)]
    straight2 = [(0, 2), (3, 5), (6, 8), (0, 6), (1, 7), (2, 8)]
    e = {edge: 9 + i for i, edge in enumerate(EDGES)}
    across = [(e[(1, 2)], e[(4, 5)]), (e[(2, 3)], e[(5, 6)]), (e[(4, 5)], e[(7, 8)]), (e[(5, 6)], e[(8, 9)]),
              (e[(1, 4)], e[(2, 5)]), (e[(2, 5)], e[(3, 6)]), (e[(4, 7)], e[(5, 8)]), (e[(5, 8)], e[(6, 9)])]
    collinear = [(e[(1, 2)], e[(2, 3)]), (e[(4, 5)], e[(5, 6)]), (e[(7, 8)], e[(8, 9)]),
                 (e[(1, 4)], e[(4, 7)]), (e[(2, 5)], e[(5, 8)]), (e[(3, 6)], e[(6, 9)])]
    return dict(diag=diag, straight2=straight2, across=across, collinear=collinear)


def fit_lattice(xy: np.ndarray, locs: np.ndarray, cfg: dict) -> dict:
    """Affine mapping this recday's node centroids (median head position per node label) onto the
    canonical lattice; every dataset's maze frame is mapped to the same cm frame."""
    lat = lattice_xy(cfg)[:9]
    src, dst = [], []
    for k in range(1, 10):
        sel = (locs == k) & np.isfinite(xy).all(axis=1)
        if sel.sum() >= 50:
            src.append(np.median(xy[sel], axis=0))
            dst.append(lat[k - 1])
    if len(src) < 6:
        raise ValueError(f'only {len(src)} node centroids; cannot fit the canonical frame')
    src, dst = np.array(src), np.array(dst)
    Xa = np.column_stack([src, np.ones(len(src))])
    A, *_ = np.linalg.lstsq(Xa, dst, rcond=None)
    resid = np.linalg.norm(Xa @ A - dst, axis=1)
    if resid.max() > cfg['lattice']['max_resid_cm']:
        raise ValueError(f'canonical-frame fit residual {resid.max():.1f} cm > {cfg["lattice"]["max_resid_cm"]}')
    return dict(A=A, resid_max_cm=float(resid.max()), n_nodes=len(src))


def apply_affine(A: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.column_stack([xy, np.ones(len(xy))]) @ A


def loc_type(loc: np.ndarray) -> np.ndarray:
    """0 corner, 1 cardinal, 2 centre, 3 bridge, -1 unknown, per sample."""
    out = np.full(len(loc), -1)
    out[np.isin(loc, CORNER)] = 0
    out[np.isin(loc, CARDINAL)] = 1
    out[loc == 5] = 2
    out[(loc >= 10) & (loc <= 21)] = 3
    return out


# ---------------------------------------------------------------------------
# assembly of one recday
# ---------------------------------------------------------------------------

def assemble(views: dict, cfg: dict) -> dict:
    """Concatenate the recday's task sessions on the 25 ms grid: counts (N, T), canonical xy, location,
    speed, goal-progress bin, state, trial, session, task, within-trial mask."""
    import glm_analysis_v3 as g3
    keep = []
    for s, v in sorted(views.items()):
        if v.get('session_type') not in (None, 'ABCD', 'ABCDE') or v.get('trial_times') is None:
            continue
        if v['locs'] is None or v['xy'] is None or len(v['trial_times']) < cfg['min_trials']:
            continue
        keep.append(s)
    if len(keep) < 2:
        raise ValueError(f'{len(keep)} usable task sessions')
    parts = []
    for s in keep:
        v = views[s]
        T = min(v['counts'].shape[1], len(v['xy']), len(v['locs']))
        tt = v['trial_times'].astype(int)
        state, gp, gpb, _, _ = g3.compute_task_state_arrays(tt, num_bins=cfg['gp_bins'])
        n = min(T, len(gpb))
        intrial = np.zeros(T, bool)
        intrial[int(tt[0, 0]):min(int(tt[-1, -1]), T)] = True
        st_ = np.full(T, -1)
        st_[:n] = state[:n]
        g_ = np.full(T, -1)
        g_[:n] = np.clip(gpb[:n], 0, cfg['gp_bins'] - 1)
        trial = np.full(T, -1)
        for r in range(len(tt)):
            trial[int(tt[r, 0]):min(int(tt[r, -1]), T)] = r
        xy = np.asarray(v['xy'][:T], float)
        parts.append(dict(session=s, T=T, counts=np.asarray(v['counts'][:, :T], np.float32), xy=xy,
                          loc=np.asarray(v['locs'][:T], int), speed=sc.speed_cm_s(xy), gp=g_, state=st_,
                          trial=trial, intrial=intrial & (g_ >= 0), task=v.get('task') or (s,)))
    xy_all = np.vstack([p['xy'] for p in parts])
    loc_all = np.concatenate([p['loc'] for p in parts])
    lat = fit_lattice(xy_all, loc_all, cfg)
    tasks = {}
    for p in parts:
        p['xy_c'] = apply_affine(lat['A'], p['xy'])
        p['task_id'] = tasks.setdefault(tuple(p['task']), len(tasks))
    cat = dict(counts=np.hstack([p['counts'] for p in parts]),
               xy=np.vstack([p['xy_c'] for p in parts]),
               loc=np.concatenate([p['loc'] for p in parts]),
               speed=np.concatenate([p['speed'] for p in parts]),
               gp=np.concatenate([p['gp'] for p in parts]),
               state=np.concatenate([p['state'] for p in parts]),
               trial=np.concatenate([p['trial'] + 1000 * i for i, p in enumerate(parts)]),
               intrial=np.concatenate([p['intrial'] for p in parts]),
               sess=np.concatenate([np.full(p['T'], i) for i, p in enumerate(parts)]),
               task=np.concatenate([np.full(p['T'], p['task_id']) for p in parts]))
    cat['type'] = loc_type(cat['loc'])
    cat['moving'] = np.isfinite(cat['speed']) & (cat['speed'] > cfg['min_speed'])
    return dict(parts=parts, cat=cat, lattice=lat, n_tasks=len(tasks), sessions=keep,
                N=cat['counts'].shape[0])


def speed_matched(idx: np.ndarray, group: np.ndarray, speed: np.ndarray, cfg: dict, rng) -> np.ndarray:
    """Subsample `idx` so every group (location type) has the same speed histogram (2.5 cm/s bins):
    in mEC corners are run 2-3 cm/s slower than cardinals, so speed cells would carry 'corner-ness'."""
    sm = cfg['speed_match']
    b = np.floor(np.clip(speed[idx], 0, sm['max_cms'] - 1e-6) / sm['bin_cms']).astype(int)
    g = group[idx]
    groups = np.unique(g)
    keep = []
    for k in np.unique(b):
        n_min = min(int(((b == k) & (g == gg)).sum()) for gg in groups)
        if n_min == 0:
            continue
        for gg in groups:
            pool = idx[(b == k) & (g == gg)]
            keep.append(rng.choice(pool, n_min, replace=False))
    return np.sort(np.concatenate(keep)) if keep else idx[:0]


# ---------------------------------------------------------------------------
# single units
# ---------------------------------------------------------------------------

def loc_onehot(loc: np.ndarray, mask: np.ndarray):
    """(T, 21) sparse indicator of locations 1..21 over `mask`."""
    return sc.onehot(np.where((loc >= 1) & (loc <= 21), loc - 1, -1), mask, 21)


def loc_maps_M(counts: np.ndarray, M, min_occ_s: float) -> tuple:
    """(N, 21) Hz and (21,) s from a location indicator; NaN where a location has < `min_occ_s`."""
    occ = np.asarray(M.sum(axis=0)).ravel() * sc.BIN_S
    S = np.asarray((M.T @ counts.T).T)
    with np.errstate(invalid='ignore', divide='ignore'):
        R = S / occ[None]
    R[:, occ < min_occ_s] = np.nan
    return R, occ


def loc_maps(counts: np.ndarray, loc: np.ndarray, mask: np.ndarray, min_occ_s: float) -> tuple:
    return loc_maps_M(counts, loc_onehot(loc, mask), min_occ_s)


def masked_pearson(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson r between X and Y (N, L) over entries finite in both; NaN below 5 entries."""
    V = np.isfinite(X) & np.isfinite(Y)
    x, y = np.where(V, X, 0.0), np.where(V, Y, 0.0)
    n = V.sum(1)
    sx, sy = x.sum(1), y.sum(1)
    sxx, syy, sxy = (x * x).sum(1), (y * y).sum(1), (x * y).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    r[n < 5] = np.nan
    return r


def cross_task_stability(task_maps: dict, sl: slice) -> np.ndarray:
    """(N,) mean r over pairs of different tasks of each unit's map restricted to `sl`."""
    keys = sorted(task_maps)
    rs = [masked_pearson(task_maps[a][:, sl], task_maps[b][:, sl]) for a, b in itertools.combinations(keys, 2)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmean(np.array(rs), axis=0) if rs else np.full(next(iter(task_maps.values())).shape[0], np.nan)


def _template_r(M: np.ndarray, T: np.ndarray) -> np.ndarray:
    """r of every unit map (N, L, NaN allowed) with every template row (P, L) -> (N, P)."""
    V = np.isfinite(M).astype(float)
    x = np.where(V > 0, M, 0.0)
    n = V.sum(1)[:, None]
    sx = x.sum(1)[:, None]
    sxx = (x * x).sum(1)[:, None]
    sxy = x @ T.T
    sy = V @ T.T
    syy = V @ (T * T).T
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    r[(n < 5).ravel()] = np.nan
    return r


TEMPLATES = dict(corner=(np.array([k in CORNER for k in range(1, 10)], float), slice(0, 9)),
                 cardinal=(np.array([k in CARDINAL for k in range(1, 10)], float), slice(0, 9)),
                 centre=(np.array([k == 5 for k in range(1, 10)], float), slice(0, 9)),
                 node=(np.r_[np.ones(9), np.zeros(12)], slice(0, 21)),
                 bridge=(np.r_[np.zeros(9), np.ones(12)], slice(0, 21)))


def type_scores(task_maps: dict, rng, n_perm: int) -> dict:
    """Per template: (N,) mean and minimum over tasks of r(map, template), and the unit's p99 under
    template permutations (within nodes for corner/cardinal/centre, across all 21 for node/bridge)."""
    keys = sorted(task_maps)
    out = {}
    for name, (tpl, sl) in TEMPLATES.items():
        perms = []
        while len(perms) < n_perm:
            cand = rng.permutation(tpl)
            if not np.array_equal(cand, tpl):           # the tested template is not part of its own null
                perms.append(cand)
        perms = np.array(perms)
        r_true = np.stack([_template_r(task_maps[k][:, sl], tpl[None])[:, 0] for k in keys])   # tasks, N
        r_null = np.mean([_template_r(task_maps[k][:, sl], perms) for k in keys], axis=0)      # N, P
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[name] = dict(mean=np.nanmean(r_true, 0), min=np.nanmin(r_true, 0),
                             null_p99=np.nanpercentile(r_null, 99, axis=1))
    return out


def _session_roll_index(parts: list, rng, min_shift: int) -> np.ndarray:
    """Row index that circularly shifts behaviour within each session (spikes stay put)."""
    idx, off = [], 0
    for p in parts:
        k = int(rng.integers(min_shift, p['T'] - min_shift))
        idx.append(off + np.roll(np.arange(p['T']), k))
        off += p['T']
    return np.concatenate(idx)


def single_units(asm: dict, cfg: dict, rng) -> dict:
    """Maps, cross-task stability (with a within-session circular-shift null), information and types."""
    c = asm['cat']
    mask = c['moving'] & c['intrial']
    st = cfg['stats']
    mo = cfg['maps']['min_loc_occ_s']
    tasks = [int(t) for t in np.unique(c['task'])]
    Ms = {t: loc_onehot(c['loc'], mask & (c['task'] == t)) for t in tasks}
    Mp = loc_onehot(c['loc'], mask)
    task_maps = {t: loc_maps_M(c['counts'], Ms[t], mo)[0] for t in tasks}
    pooled, occ = loc_maps_M(c['counts'], Mp, mo)
    stab21 = cross_task_stability(task_maps, slice(0, 21))
    stab9 = cross_task_stability(task_maps, slice(0, 9))
    info = sc.skaggs_info(pooled[:, :, None], occ[:, None])
    min_shift = int(round(st['min_shift_s'] / sc.BIN_S))
    null_stab, null_info = [], []
    for _ in range(st['n_shifts']):
        ri = _session_roll_index(asm['parts'], rng, min_shift)
        tm = {t: loc_maps_M(c['counts'], Ms[t][ri], mo)[0] for t in tasks}
        null_stab.append(cross_task_stability(tm, slice(0, 21)))
        pm, _ = loc_maps_M(c['counts'], Mp[ri], mo)
        null_info.append(sc.skaggs_info(pm[:, :, None], occ[:, None]))
    null_stab, null_info = np.array(null_stab), np.array(null_info)
    N = asm['N']
    other = []
    if N > 1 and len(tasks) > 1:
        for _ in range(2000):
            u, w = rng.choice(N, 2, replace=False)
            a, b = rng.choice(tasks, 2, replace=False)
            other.append(masked_pearson(task_maps[a][[u]], task_maps[b][[w]])[0])
    types = type_scores(task_maps, rng, st['n_label_perm'])
    stab_thr = sc.unit_threshold(null_stab, st['q_stab'])
    info_thr = sc.unit_threshold(null_info, st['q_unit'])
    stable = stab21 > stab_thr
    spatial = stable & (info > info_thr)
    type_class = {k: stable & (v['mean'] > v['null_p99']) & (v['min'] > 0) for k, v in types.items()}
    return dict(task_maps=task_maps, pooled=pooled, occ=occ, stab21=stab21, stab9=stab9, info=info,
                stab_thr=stab_thr, info_thr=info_thr, other_unit_r=np.array(other), types=types,
                stable=stable, spatial=spatial, type_class=type_class)


def maps_2d(asm: dict, cfg: dict) -> dict:
    """Canonical-frame 2-D maps (1 cm) per task and pooled, both smoothers at each sigma; footprint
    from occupancy. Cross-task 2-D r per smoother and sigma."""
    c = asm['cat']
    mask = c['moving'] & c['intrial']
    L = cfg['lattice']
    lo = L['origin_cm'] - L['margin_cm']
    hi = L['origin_cm'] + 2 * L['spacing_cm'] + L['margin_cm']
    ex = sc.fixed_edges(lo, hi, cfg['maps']['bin_cm'])
    nb = (len(ex) - 1, len(ex) - 1)
    flat = sc.bin_index(c['xy'], ex, ex)
    out = dict(edges=ex)
    tasks = np.unique(c['task'])

    def maps_for(m):
        M = sc.onehot(flat, m, nb[0] * nb[1])
        occ = sc.occupancy_map(M, nb)
        spk = sc.spike_maps(c['counts'], M, nb)
        res = {}
        for sg in cfg['maps']['sigmas']:
            res[('normconv', sg)] = sc.rate_maps_normconv(spk, occ, sg, cfg['maps']['occ_threshold_s'])
            res[('occweighted', sg)] = sc.rate_maps_separate(spk, occ, sg, valid=occ > 0,
                                                             min_smoothed_occ=cfg['maps']['occ_threshold_s'])
        return res, occ

    pooled, occ = maps_for(mask)
    out['occ'] = occ
    out['pooled'] = {k: v.astype(np.float32) for k, v in pooled.items() if k[1] == cfg['maps']['sigmas'][0]}
    per_task = {int(t): maps_for(mask & (c['task'] == t))[0] for t in tasks}
    out['cross_task_r'] = {}
    for key in pooled:
        rs = []
        for a, b in itertools.combinations(per_task, 2):
            rs.append(sc.map_corr(per_task[a][key], per_task[b][key], min_bins=20))
        out['cross_task_r'][key] = np.nanmean(np.array(rs), axis=0) if rs else np.full(asm['N'], np.nan)
    return out


# ---------------------------------------------------------------------------
# population: RSA
# ---------------------------------------------------------------------------

def _std(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / np.nanstd(v)


def rsa_fit(rdm: np.ndarray, models: list) -> dict:
    """Least-squares betas of the standardised data RDM (upper triangle) on standardised model RDMs."""
    iu = np.triu_indices(rdm.shape[0], 1)
    y = rdm[iu]
    ok = np.isfinite(y)
    if ok.sum() < 2 * (len(models) + 2) or np.nanstd(y[ok]) == 0:
        return dict(beta=np.full(len(models), np.nan), r2=np.nan, n_pairs=int(ok.sum()))
    X = np.column_stack([np.ones(ok.sum())] + [_std(m[iu])[ok] for m in models])
    yz = _std(y[ok])
    beta, *_ = np.linalg.lstsq(X, yz, rcond=None)
    pred = X @ beta
    r2 = 1 - ((yz - pred) ** 2).sum() / ((yz - yz.mean()) ** 2).sum()
    return dict(beta=beta[1:], r2=float(r2), n_pairs=int(ok.sum()))


def task_splits(task_ids: np.ndarray, max_splits: int, rng) -> list:
    """Balanced splits of the tasks into two disjoint halves (all of them up to `max_splits`)."""
    t = np.unique(task_ids)
    h = len(t) // 2
    combos = [c for c in itertools.combinations(t, h) if t[0] in c]
    if len(combos) > max_splits:
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    return [(np.array(c), np.setdiff1d(t, c)) for c in combos]


def rsa_decided(counts: np.ndarray, c: dict, idx_by_n: dict, cfg: dict, rng) -> dict:
    """Cross-validated, per-unit z-scored RDMs per goal-progress bin (disjoint task halves), averaged.

    d(i, j) = mean over units of (zA_i - zA_j)(zB_i - zB_j), each pair over the units with both
    locations sampled in both halves: speed matching inside a goal-progress bin leaves some nodes thin.
    """
    R = cfg['rsa']
    res = {}
    for nloc, idx in idx_by_n.items():
        splits = task_splits(c['task'][idx], R['max_splits'], rng)
        per_gp = []
        for g in range(cfg['gp_bins']):
            ig = idx[c['gp'][idx] == g]
            D = np.zeros((nloc, nloc))
            W = np.zeros((nloc, nloc))
            rall = _loc_rates(counts, c['loc'], ig, nloc, R['min_samples'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mu = np.nanmean(rall, 1, keepdims=True)
                sd = np.nanstd(rall, 1, keepdims=True)
            sd[~(sd > 0)] = np.nan
            for A, B in splits:
                za = (_loc_rates(counts, c['loc'], ig[np.isin(c['task'][ig], A)], nloc, R['min_samples']) - mu) / sd
                zb = (_loc_rates(counts, c['loc'], ig[np.isin(c['task'][ig], B)], nloc, R['min_samples']) - mu) / sd
                fa, fb = np.isfinite(za), np.isfinite(zb)
                for i in range(nloc):
                    for j in range(i + 1, nloc):
                        ok = fa[:, i] & fa[:, j] & fb[:, i] & fb[:, j]
                        if ok.sum() >= R['min_units']:
                            D[i, j] += np.mean((za[ok, i] - za[ok, j]) * (zb[ok, i] - zb[ok, j]))
                            W[i, j] += 1
            Dg = np.where(W > 0, D / np.maximum(W, 1), np.nan)
            per_gp.append(np.triu(Dg, 1) + np.triu(Dg, 1).T + np.diag(np.full(nloc, np.nan)))
        res[nloc] = np.array(per_gp)
    return res


def _loc_rates(counts, loc, idx, nloc, min_samples: int = 10) -> np.ndarray:
    """(N, nloc) mean rate per location over sample indices `idx`; NaN below `min_samples` samples."""
    out = np.full((counts.shape[0], nloc), np.nan)
    for k in range(1, nloc + 1):
        s = idx[loc[idx] == k]
        if len(s) >= min_samples:
            out[:, k - 1] = counts[:, s].mean(1) / sc.BIN_S
    return out


def rsa_thesis(counts: np.ndarray, c: dict, cfg: dict) -> np.ndarray:
    """Thesis recipe (pfc_rsa cells 22/34): every 25 ms bin of each session, no goal-progress split, no
    speed filter; mean raw rate per node; Euclidean RDM per session; nan-mean over sessions."""
    rdms = []
    for s in np.unique(c['sess']):
        ii = np.nonzero(c['sess'] == s)[0]
        r = _loc_rates(counts, c['loc'], ii, 9, 1)
        if np.isfinite(r).all():
            rdms.append(np.linalg.norm(r.T[:, None] - r.T[None], axis=2))
    return np.nanmean(rdms, axis=0) if rdms else np.full((9, 9), np.nan)


def rsa_fit_raw(rdm: np.ndarray, models: list) -> dict:
    """Unstandardised OLS of the RDM on the model RDMs, as the thesis regression."""
    iu = np.triu_indices(rdm.shape[0], 1)
    y = rdm[iu]
    ok = np.isfinite(y)
    if ok.sum() < 2 * (len(models) + 2):
        return dict(beta=np.full(len(models), np.nan), r2=np.nan, n_pairs=int(ok.sum()))
    X = np.column_stack([np.ones(ok.sum())] + [m[iu][ok] for m in models])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    pred = X @ beta
    r2 = 1 - ((y[ok] - pred) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    return dict(beta=beta[1:], r2=float(r2), n_pairs=int(ok.sum()))


def rsa_summary(per_gp: dict, thesis_rdm: np.ndarray, M: dict, cfg: dict, rng) -> dict:
    """Model fits, cross-validated E vs G comparison across goal-progress bins, unique variance,
    diagnostic contrasts, node-label permutation p."""
    out = {}
    dp = diagnostic_pairs()
    for nloc, key_e, key_g, key_x in ((9, 'E9', 'G9', 'D9'), (21, 'E21', 'G21', 'NB21')):
        if nloc not in per_gp:
            continue
        stack = per_gp[nloc]
        mean = np.nanmean(stack, axis=0)
        f_ed = rsa_fit(mean, [M[key_e], M[key_x]])
        f_gd = rsa_fit(mean, [M[key_g], M[key_x]])
        f_egd = rsa_fit(mean, [M[key_e], M[key_g], M[key_x]])
        cv = dict(ED=[], GD=[])
        for g in range(len(stack)):
            train = np.nanmean(np.delete(stack, g, axis=0), axis=0)
            test = stack[g]
            for name, mods in (('ED', [M[key_e], M[key_x]]), ('GD', [M[key_g], M[key_x]])):
                cv[name].append(_cv_r2(train, test, mods))
        perm = []
        for _ in range(cfg['rsa']['n_perm']):
            p = rng.permutation(nloc)
            perm.append(rsa_fit(mean[np.ix_(p, p)], [M[key_e], M[key_x]])['beta'])
        perm = np.array(perm)
        p_ed = (1 + (perm >= f_ed['beta'][None]).sum(0)) / (1 + len(perm))
        if nloc == 9:
            a, b = dp['diag'], dp['straight2']
        else:
            a, b = dp['across'], dp['collinear']
        adj = np.nanmean([mean[i, j] for i in range(nloc) for j in range(nloc) if i < j])
        contrast = (np.nanmean([mean[i, j] for i, j in a]) - np.nanmean([mean[i, j] for i, j in b])) / abs(adj) if adj else np.nan
        out[nloc] = dict(rdm=mean, per_gp=stack, fit_ED=f_ed, fit_GD=f_gd, fit_EGD=f_egd,
                         cv_r2_ED=float(np.nanmean(cv['ED'])), cv_r2_GD=float(np.nanmean(cv['GD'])),
                         unique_E=f_egd['r2'] - f_gd['r2'], unique_G=f_egd['r2'] - f_ed['r2'],
                         perm_p_ED=p_ed, diagnostic_contrast=float(contrast))
    if thesis_rdm is not None and np.isfinite(thesis_rdm).any():
        E_nodes = M['E9'] / cfg['lattice']['spacing_cm']          # the thesis used grid units
        out['thesis'] = dict(rdm=thesis_rdm, fit_EPD=rsa_fit_raw(thesis_rdm, [E_nodes, M['P9'], M['D9']]),
                             fit_ED_std=rsa_fit(thesis_rdm, [M['E9'], M['D9']]))
    return out


def _cv_r2(train: np.ndarray, test: np.ndarray, models: list) -> float:
    iu = np.triu_indices(train.shape[0], 1)
    ok = np.isfinite(train[iu]) & np.isfinite(test[iu])
    if ok.sum() < 5:
        return np.nan
    X = np.column_stack([np.ones(ok.sum())] + [_std(m[iu])[ok] for m in models])
    beta, *_ = np.linalg.lstsq(X, _std(train[iu][ok]), rcond=None)
    yt = _std(test[iu][ok])
    return float(1 - ((yt - X @ beta) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())


# ---------------------------------------------------------------------------
# population: decoders
# ---------------------------------------------------------------------------

def decode_samples(counts: np.ndarray, c: dict, cfg: dict) -> dict:
    """500 ms samples within trials, moving, with one location for >= 80 % of the bin."""
    w = cfg['decode']['bin_bins']
    T = counts.shape[1]
    starts = np.arange(0, T - w + 1, w)
    X, loc, task, spd = [], [], [], []
    for s0 in starts:
        sl = slice(s0, s0 + w)
        if c['sess'][s0] != c['sess'][s0 + w - 1] or not c['intrial'][sl].all():
            continue
        l = c['loc'][sl]
        vals, cnt = np.unique(l[l > 0], return_counts=True)
        if not len(vals) or cnt.max() < cfg['decode']['min_frac_same_loc'] * w:
            continue
        v = np.nanmean(c['speed'][sl])
        if not np.isfinite(v) or v <= cfg['min_speed']:
            continue
        X.append(counts[:, sl].sum(1))
        loc.append(vals[np.argmax(cnt)])
        task.append(c['task'][s0])
        spd.append(v)
    return dict(X=np.array(X, float), loc=np.array(loc), task=np.array(task), speed=np.array(spd))


def _svm(Xtr, ytr, Xte, C):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=C, class_weight='balanced'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def _balanced_acc(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def _pmap(fn, items: list, n_jobs: int) -> list:
    """fn over items, in parallel threads when there is more than one (sklearn's SVC releases the GIL)."""
    if n_jobs <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=min(n_jobs, len(items)), prefer='threads')(delayed(fn)(x) for x in items)


def decoders(ds: dict, cfg: dict, rng, matched: bool, n_perm: int) -> dict:
    """Held-out corner vs cardinal, held-out node vs bridge, held-out-task 9-way with error structure."""
    C = cfg['decode']['C']
    out = {}
    X, loc, task, spd = ds['X'], ds['loc'], ds['task'], ds['speed']
    typ = loc_type(loc)
    # corner vs cardinal: leave one peripheral node out
    per = np.isin(loc, CORNER + CARDINAL)
    idx = np.nonzero(per)[0]
    if matched:
        idx = speed_matched(idx, typ, spd, cfg, rng)

    def cc(label_of_node):
        y = np.array([label_of_node[l] for l in loc[idx]])
        pred = np.empty(len(idx), dtype=y.dtype)
        for k in CORNER + CARDINAL:
            te = loc[idx] == k
            tr = ~te
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                return np.nan
            pred[te] = _svm(X[idx][tr], y[tr], X[idx][te], C)
        return _balanced_acc(y, pred)

    true_lab = {k: int(k in CORNER) for k in CORNER + CARDINAL}
    out['corner_cardinal_ba'] = cc(true_lab)
    assigns = [set(c) for c in itertools.combinations(CORNER + CARDINAL, 4) if 1 in c and set(c) != set(CORNER)]
    pick = rng.choice(len(assigns), min(n_perm, len(assigns)), replace=False)
    out['corner_cardinal_null'] = np.array(_pmap(cc, [{k: int(k in assigns[i]) for k in CORNER + CARDINAL} for i in pick],
                                                 cfg['n_jobs']))
    # node vs bridge: leave one location out
    idx = np.nonzero(loc >= 1)[0]
    if matched:
        idx = speed_matched(idx, (loc >= 10).astype(int), spd, cfg, rng)

    def nb(is_bridge):
        y = is_bridge[loc[idx] - 1]
        pred = np.empty(len(idx), dtype=int)
        for k in range(1, 22):
            te = loc[idx] == k
            if te.sum() == 0:
                continue
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                return np.nan
            pred[te] = _svm(X[idx][tr], y[tr], X[idx][te], C)
        return _balanced_acc(y, pred)

    truth = np.r_[np.zeros(9, int), np.ones(12, int)]
    out['node_bridge_ba'] = nb(truth)
    perms = [rng.permutation(truth) for _ in range(min(n_perm, 50))]          # drawn before dispatch
    out['node_bridge_null'] = np.array(_pmap(nb, perms, cfg['n_jobs']))
    # 9-way location, leave one task out
    idx = np.nonzero((loc >= 1) & (loc <= 9))[0]
    if matched:
        idx = speed_matched(idx, typ, spd, cfg, rng)
    y = loc[idx]
    pred = np.zeros(len(idx), int)
    tasks = np.unique(task[idx])
    if len(tasks) >= 2:
        for t in tasks:
            te = task[idx] == t
            pred[te] = _svm(X[idx][~te], y[~te], X[idx][te], C)
        conf = np.zeros((9, 9))
        for a, b in zip(y, pred):
            conf[a - 1, b - 1] += 1
        out['location_acc'] = float((pred == y).mean())
        out['confusion'] = conf
        err = (pred != y) & (y != 5) & (pred != 5)
        same = np.array([NODE_TYPE[a] == NODE_TYPE[b] for a, b in zip(y[err], pred[err])])
        out['same_type_error_share'] = float(same.mean()) if len(same) else np.nan
        out['n_errors'] = int(err.sum())
    out['n_samples'] = len(X)
    return out


def decoders_thesis(counts: np.ndarray, c: dict, cfg: dict) -> dict:
    """Thesis recipe (pfc_decoding cells 20/36/42): 500 ms mean counts per session, the location of each
    block's first bin, no feature scaling, SVC(rbf, C=1.5, balanced); speed cuts converted to cm/s.
    The notebook's stale speed mask in cell 36 is not reproduced."""
    w = cfg['decode']['bin_bins']
    th = cfg['thesis']
    X, loc, sess, spd = [], [], [], []
    for s in np.unique(c['sess']):
        ii = np.nonzero(c['sess'] == s)[0]
        nblk = len(ii) // w
        blk = ii[:nblk * w].reshape(nblk, w)
        X.append(counts[:, blk].mean(2).T)
        loc.append(c['loc'][blk[:, 0]])
        sess.append(np.full(nblk, s))
        spd.append(np.nanmean(c['speed'][blk], axis=1))
    X, loc, sess, spd = np.vstack(X), np.concatenate(loc), np.concatenate(sess), np.concatenate(spd)
    out = {}
    from sklearn.svm import SVC

    def fit_pred(Xtr, ytr, Xte):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return SVC(kernel='rbf', C=th['C'], class_weight='balanced').fit(Xtr, ytr).predict(Xte)
    m = (loc >= 1) & (loc <= 9) & (spd > th['speed_cc_cms'])
    y, pred = [], []
    for k in CORNER + CARDINAL:
        tr = m & (loc != k) & (loc != 5)
        te = m & (loc == k)
        if te.sum() == 0:
            continue
        pred.append(fit_pred(X[tr], np.isin(loc[tr], CORNER).astype(int), X[te]))
        y.append(np.full(te.sum(), int(k in CORNER)))
    out['corner_cardinal_ba'] = _balanced_acc(np.concatenate(y), np.concatenate(pred)) if y else np.nan
    m = (loc >= 1) & (spd > th['speed_nb_cms'])
    y, pred = [], []
    for k in range(1, 22):
        te = m & (loc == k)
        if te.sum() == 0:
            continue
        tr = m & (loc != k)
        pred.append(fit_pred(X[tr], (loc[tr] >= 10).astype(int), X[te]))
        y.append(np.full(te.sum(), int(k >= 10)))
    out['node_bridge_ba'] = _balanced_acc(np.concatenate(y), np.concatenate(pred)) if y else np.nan
    m = (loc >= 1) & (loc <= 9)
    yy, pp = [], []
    for s in np.unique(sess):
        te = m & (sess == s)
        tr = m & (sess != s)
        if te.sum() == 0 or len(np.unique(loc[tr])) < 2:
            continue
        pp.append(fit_pred(X[tr], loc[tr], X[te]))
        yy.append(loc[te])
    if yy:
        yy, pp = np.concatenate(yy), np.concatenate(pp)
        out['location_acc'] = float((yy == pp).mean())
        err = (pp != yy) & (yy != 5) & (pp != 5)
        same = np.array([NODE_TYPE[a] == NODE_TYPE[b] for a, b in zip(yy[err], pp[err])])
        out['same_type_error_share'] = float(same.mean()) if len(same) else np.nan
        out['n_errors'] = int(err.sum())
    out['n_samples'] = len(X)
    return out


# ---------------------------------------------------------------------------
# population: lasso location planes per goal-progress bin
# ---------------------------------------------------------------------------

def plane_design(counts: np.ndarray, c: dict, cfg: dict) -> dict:
    """250 ms rows on nodes within trials: z-scored rates, current + 19 lagged node one-hots (lags in
    task-space segments: trial, state, goal-progress bin), z-scored speed and acceleration, gp bin, task."""
    P = cfg['planes']
    w = P['bin_bins']
    seg = np.where(c['intrial'], (c['trial'] * 4 + np.maximum(c['state'], 0)) * cfg['gp_bins'] + c['gp'], -1)
    # modal node of every task-space segment; bridges count as no node, as in the thesis code
    seg_node = {}
    for sgm in np.unique(seg[seg >= 0]):
        l = c['loc'][seg == sgm]
        l = l[(l >= 1) & (l <= 9)]
        seg_node[int(sgm)] = int(np.bincount(l).argmax()) if len(l) else 0
    rows = []
    for s0 in range(0, counts.shape[1] - w + 1, w):
        mid = s0 + w // 2
        if not c['intrial'][mid] or not (1 <= c['loc'][mid] <= 9) or c['sess'][s0] != c['sess'][s0 + w - 1]:
            continue
        rows.append(s0)
    rows = np.array(rows)
    Y = np.array([counts[:, r:r + w].sum(1) for r in rows], float)
    Y = (Y - Y.mean(0)) / np.where(Y.std(0) > 0, Y.std(0), 1)
    nl = P['n_lags']
    Xl = np.zeros((len(rows), nl * 9))
    seg_rows = seg[rows + w // 2]
    for i, sg in enumerate(seg_rows):
        for k in range(nl):
            node = c['loc'][rows[i] + w // 2] if k == 0 else seg_node.get(int(sg) - k, 0)
            if 1 <= node <= 9:
                Xl[i, k * 9 + node - 1] = 1
    spd = np.array([np.nanmean(c['speed'][r:r + w]) for r in rows])
    spd = np.nan_to_num((spd - np.nanmean(spd)) / np.nanstd(spd))
    acc = np.gradient(spd)
    X = np.column_stack([Xl, spd, acc])
    return dict(X=X, Y=Y, gp=c['gp'][rows + w // 2], task=c['task'][rows], n_lag_cols=nl * 9)


def _lasso_betas(X, Y, alpha, max_iter: int = 5000) -> np.ndarray:
    """(N, n_cols) lasso coefficients; sklearn fits each target independently in one call."""
    from sklearn.linear_model import Lasso
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.atleast_2d(Lasso(alpha=alpha, max_iter=max_iter).fit(X, Y).coef_)


def _plane(B: np.ndarray, k: int) -> np.ndarray:
    """Top-k PCA plane (N x k) of the 9 current-node points in neuron space, after removing each
    neuron's mean beta (as the thesis code)."""
    B = B - B.mean(1, keepdims=True)
    pts = B[:, :9].T                                  # 9 x N
    pts = pts - pts.mean(0, keepdims=True)
    U, Sv, Vt = np.linalg.svd(pts, full_matrices=False)
    return Vt[:k].T


def _angles(P1, P2) -> tuple:
    a = np.degrees(subspace_angles(P1, P2))
    return float(a.min()), float(a.max())


def planes(design: dict, cfg: dict, rng, recipe: str) -> dict:
    """Plane angles across goal-progress bins, and the within-bin controls (halves and tasks)."""
    P = cfg['planes']
    X, Y, gp, task = design['X'], design['Y'], design['gp'], design['task']
    nbins = cfg['gp_bins']
    out = dict(recipe=recipe)
    if recipe == 'thesis':
        # As the thesis notebook (allnorm_lowalpha cell 14): 5 Bernoulli(0.5) splits; the 10 fits are
        # stacked [A1, B1, A2, B2, ..., A5, B5] and beta1 / beta2 are the means of the first / last
        # five, so beta1 already holds A1 and B1 (all the data); PCA(2); the LARGEST principal angle.
        B1, B2 = {}, {}
        for g in range(nbins):
            ig = np.nonzero(gp == g)[0]
            stack = []
            for _ in range(P['n_reps']):
                m = rng.random(len(ig)) < 0.5
                stack.append(_lasso_betas(X[ig[m]], Y[ig[m]], P['alpha'], max_iter=1000))
                stack.append(_lasso_betas(X[ig[~m]], Y[ig[~m]], P['alpha'], max_iter=1000))
            h = len(stack) // 2
            B1[g], B2[g] = np.mean(stack[:h], 0), np.mean(stack[h:], 0)
        out['within'] = {g: _angles(_plane(B1[g], 2), _plane(B2[g], 2))[1] for g in range(nbins)}
        out['across'] = {(a, b): _angles(_plane(B1[a], 2), _plane(B1[b], 2))[1] for a in range(nbins) for b in range(a + 1, nbins)}
        return out
    fits = {}
    for g in range(nbins):
        ig = np.nonzero(gp == g)[0]
        for r in range(P['n_reps']):
            perm = rng.permutation(ig)
            h = len(perm) // 2
            fits[(g, 'half', r)] = (_lasso_betas(X[perm[:h]], Y[perm[:h]], P['alpha']),
                                    _lasso_betas(X[perm[h:]], Y[perm[h:]], P['alpha']))
        for j, (A, Bt) in enumerate(task_splits(task[ig], P['max_task_splits'], rng)):
            ia, ib = ig[np.isin(task[ig], A)], ig[np.isin(task[ig], Bt)]
            if len(ia) > 20 and len(ib) > 20:
                fits[(g, 'task', j)] = (_lasso_betas(X[ia], Y[ia], P['alpha']), _lasso_betas(X[ib], Y[ib], P['alpha']))
    for k in P['ks']:
        halves, tasks_c, across = {}, {}, {}
        for g in range(nbins):
            hs = [_angles(_plane(a, k), _plane(b, k)) for (gg, kind, _), (a, b) in fits.items() if gg == g and kind == 'half']
            ts = [_angles(_plane(a, k), _plane(b, k)) for (gg, kind, _), (a, b) in fits.items() if gg == g and kind == 'task']
            halves[g] = tuple(np.mean(hs, 0)) if hs else (np.nan, np.nan)
            tasks_c[g] = tuple(np.mean(ts, 0)) if ts else (np.nan, np.nan)
        for a in range(nbins):
            for b in range(a + 1, nbins):
                # same data amount as the halves control: half A of the first repetition of each bin
                across[(a, b)] = _angles(_plane(fits[(a, 'half', 0)][0], k), _plane(fits[(b, 'half', 0)][0], k))
        out[k] = dict(halves=halves, tasks=tasks_c, across=across)
    return out


# ---------------------------------------------------------------------------
# simulated populations for the nulls
# ---------------------------------------------------------------------------

def _gauss_fields(params: np.ndarray, X: np.ndarray, K: int) -> np.ndarray:
    base = np.exp(params[0])
    lam = np.full(len(X), base)
    for k in range(K):
        a, cx, cy, ls = params[1 + 4 * k:5 + 4 * k]
        lam = lam + np.exp(a) * np.exp(-((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2) / (2 * np.exp(ls) ** 2))
    return lam


def _nll_grad(p: np.ndarray, X: np.ndarray, K: int, n: np.ndarray, t: np.ndarray) -> tuple:
    """Poisson negative log-likelihood of K Gaussian fields + baseline, and its analytic gradient."""
    base = np.exp(p[0])
    lam = np.full(len(X), base)
    parts = []
    for k in range(K):
        a, cx, cy, ls = p[1 + 4 * k:5 + 4 * k]
        s2 = np.exp(2 * ls)
        dx, dy = X[:, 0] - cx, X[:, 1] - cy
        g = np.exp(a) * np.exp(-(dx ** 2 + dy ** 2) / (2 * s2))
        lam = lam + g
        parts.append((g, dx, dy, s2))
    lam = np.maximum(lam, 1e-12)
    f = float((t * lam - n * np.log(t * lam + 1e-300)).sum())
    w = t - n / lam                                   # dNLL / dlambda
    grad = [float((w * base).sum())]
    for g, dx, dy, s2 in parts:
        grad += [float((w * g).sum()), float((w * g * dx / s2).sum()), float((w * g * dy / s2).sum()),
                 float((w * g * (dx ** 2 + dy ** 2) / s2).sum())]
    return f, np.array(grad)


def fit_place_models(asm: dict, maps2d: dict, cfg: dict) -> list:
    """Per unit, 1..max_fields isotropic Gaussian fields + baseline fitted by Poisson ML to the 1 cm
    spike and occupancy maps; the field count chosen by BIC."""
    c = asm['cat']
    mask = c['moving'] & c['intrial']
    ex = maps2d['edges']
    nb = len(ex) - 1
    flat = sc.bin_index(c['xy'], ex, ex)
    M = sc.onehot(flat, mask, nb * nb)
    occ = sc.occupancy_map(M, (nb, nb)).ravel()
    spk = sc.spike_maps(c['counts'], M, (nb, nb)).reshape(asm['N'], -1)
    cen = (ex[:-1] + ex[1:]) / 2
    GX, GY = np.meshgrid(cen, cen, indexing='ij')
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    vis = occ > 0
    smooth = maps2d['pooled'][('normconv', cfg['maps']['sigmas'][0])].reshape(asm['N'], -1)
    models = []
    for u in range(asm['N']):
        n, t, P = spk[u, vis], occ[vis], pts[vis]
        best = None
        mean_rate = n.sum() / max(t.sum(), 1e-9)
        if n.sum() < 20:
            models.append(dict(K=0, params=np.array([np.log(max(mean_rate, 1e-3))])))
            continue
        order = np.argsort(-np.nan_to_num(smooth[u, vis]))
        for K in range(0, cfg['null']['max_fields'] + 1):
            p0 = [np.log(max(mean_rate * 0.5, 1e-3))]
            used = []
            for i in order:
                if len(used) == K:
                    break
                if all(np.hypot(*(P[i] - P[j])) > 8 for j in used):
                    used.append(i)
            if len(used) < K:
                break
            for i in used:
                p0 += [np.log(max(np.nan_to_num(smooth[u, vis][i]), 0.5)), P[i, 0], P[i, 1], np.log(6.0)]

            lo_xy, hi_xy = P.min(0) - 5, P.max(0) + 5
            bounds = [(np.log(1e-3), np.log(200.0))]
            for _ in range(K):
                # unbounded fits diverged on PFC units (amplitude -> inf, width -> 0): 'lam value too large'
                bounds += [(np.log(1e-2), np.log(500.0)), (lo_xy[0], hi_xy[0]), (lo_xy[1], hi_xy[1]),
                           (np.log(2.0), np.log(40.0))]
            x0 = np.clip(np.array(p0), [b[0] for b in bounds], [b[1] for b in bounds])
            r = optimize.minimize(_nll_grad, x0, args=(P, K, n, t), jac=True, method='L-BFGS-B', bounds=bounds)
            bic = 2 * r.fun + len(p0) * np.log(max(vis.sum(), 2))     # n-only constant terms cancel across K
            if best is None or bic < best[0]:
                best = (bic, K, r.x)
        models.append(dict(K=best[1], params=best[2]))
    return models


def simulate(asm: dict, kind: str, models: list, rng) -> np.ndarray:
    """(N, T) Poisson counts on the recday's real trajectory under one null family.

    place: the fitted fields. place_random: the same fields at random visited positions. place_shuffled:
    each field's centre drawn from the recday's pooled fitted centres, so the population keeps where its
    fields are and each unit keeps its count, widths and heights, but a unit's fields no longer co-occur
    at locations of one type.
    """
    c = asm['cat']
    T = c['counts'].shape[1]
    N = asm['N']
    out = np.zeros((N, T), np.float32)
    xy = np.nan_to_num(c['xy'])
    if kind in ('place', 'place_random', 'place_shuffled'):
        vis = xy[c['moving'] & c['intrial']]
        pool = np.array([m['params'][2 + 4 * k:4 + 4 * k] for m in models for k in range(m['K'])]).reshape(-1, 2)
        for u, m in enumerate(models):
            p = m['params'].copy()
            if kind == 'place_random':
                for k in range(m['K']):
                    p[2 + 4 * k:4 + 4 * k] = vis[rng.integers(len(vis))]
            elif kind == 'place_shuffled' and len(pool):
                for k in range(m['K']):
                    p[2 + 4 * k:4 + 4 * k] = pool[rng.integers(len(pool))]
            out[u] = rng.poisson(np.clip(_gauss_fields(p, xy, m['K']), 0, 1000.0) * sc.BIN_S)
    elif kind == 'speed':
        spd = np.nan_to_num(c['speed'])
        bins = np.minimum((spd / 5).astype(int), 11)
        for u in range(N):
            f = np.array([c['counts'][u, bins == b].mean() if (bins == b).any() else 0 for b in range(12)])
            out[u] = rng.poisson(f[bins])
    elif kind == 'boundary':
        d = boundary_distance(asm, xy)
        bins = np.minimum((d / 1.5).astype(int), 7)
        for u in range(N):
            f = np.array([c['counts'][u, bins == b].mean() if (bins == b).any() else 0 for b in range(8)])
            out[u] = rng.poisson(f[bins])
    else:
        raise ValueError(kind)
    return out


def boundary_distance(asm: dict, xy: np.ndarray) -> np.ndarray:
    """(T,) cm from each position to the nearest unvisited 1 cm bin: the maze's drop-off edges."""
    c = asm['cat']
    ex = np.arange(-10, 60.5, 1.0)
    nb = len(ex) - 1
    flat = sc.bin_index(xy, ex, ex)
    occ = np.bincount(flat[flat >= 0], minlength=nb * nb).reshape(nb, nb)
    footprint = occ >= 5
    dist = distance_transform_edt(footprint)
    out = np.zeros(len(xy))
    ok = flat >= 0
    out[ok] = dist.ravel()[flat[ok]]
    return out


# ---------------------------------------------------------------------------
# recday entry point
# ---------------------------------------------------------------------------

def population(counts: np.ndarray, asm: dict, cfg: dict, rng, recipe: str, which=('rsa', 'decode', 'planes'),
               n_perm_decode: int | None = None) -> dict:
    """RSA, decoders and planes for one spike array under one recipe."""
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    c = asm['cat']
    out = {}
    M = model_rdms(cfg)
    base = np.nonzero(c['moving'] & c['intrial'] & (c['loc'] >= 1))[0]
    if 'rsa' in which:
        if recipe == 'decided':
            idx9 = speed_matched(base[c['loc'][base] <= 9], c['type'], c['speed'], cfg, rng)
            idx21 = speed_matched(base, c['type'], c['speed'], cfg, rng)
            per_gp = rsa_decided(counts, c, {9: idx9, 21: idx21}, cfg, rng)
            out['rsa'] = rsa_summary(per_gp, None, M, cfg, rng)
        else:
            out['rsa'] = rsa_summary({}, rsa_thesis(counts, c, cfg), M, cfg, rng)
    if 'decode' in which:
        if recipe == 'decided':
            ds = decode_samples(counts, c, cfg)
            npd = cfg['decode']['n_perm'] if n_perm_decode is None else n_perm_decode
            out['decode'] = decoders(ds, cfg, rng, matched=cfg['decode']['match_speed'], n_perm=npd)
        else:
            out['decode'] = decoders_thesis(counts, c, cfg)
    if 'planes' in which:
        out['planes'] = planes(plane_design(counts, c, cfg), cfg, rng, recipe)
    return out


def _null_sim(asm: dict, cfg: dict, models: list, kind: str, which: tuple, seed: int) -> dict:
    """One simulated population under a null family, through the same population pipeline."""
    rng = np.random.default_rng(seed)
    sim = simulate(asm, kind, models, rng)
    return population(sim, asm, cfg, rng, 'decided', which=which, n_perm_decode=0)


def run_maze_recday(views: dict, recday: str, units, cfg: dict) -> dict:
    """Every maze analysis for one recday; returns the shard dict (no file I/O here)."""
    cfg = copy.deepcopy(cfg)
    rng = np.random.default_rng(cfg['seed'])
    asm = assemble(views, cfg)
    su = single_units(asm, cfg, rng)
    m2 = maps_2d(asm, cfg)
    real = {r: population(asm['cat']['counts'], asm, cfg, rng, r) for r in ('decided', 'thesis')}
    models = fit_place_models(asm, m2, cfg)
    nl = cfg['null']
    jobs = []
    allk = set().union(*nl['kinds'].values())
    kinds = [k for k in V1_NULL_KINDS if k in allk] + sorted(allk - set(V1_NULL_KINDS))
    for kind in kinds:
        for i in range(max(nl['n_sim_rsa'], nl['n_sim_decode'], nl['n_sim_planes'])):
            which = tuple(w for w, n in (('rsa', nl['n_sim_rsa']), ('decode', nl['n_sim_decode']),
                                         ('planes', nl['n_sim_planes'])) if i < n and kind in nl['kinds'][w])
            if which:
                jobs.append((kind, which, int(rng.integers(2 ** 31))))
    if cfg['n_jobs'] > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=cfg['n_jobs'], backend='loky')(
            delayed(_null_sim)(asm, cfg, models, kind, which, seed) for kind, which, seed in jobs)
    else:
        results = [_null_sim(asm, cfg, models, kind, which, seed) for kind, which, seed in jobs]
    nulls = {kind: dict(rsa=[], decode=[], planes=[]) for kind in kinds}
    for (kind, which, _), res in zip(jobs, results):
        for w in which:
            nulls[kind][w].append(res[w])
    lat = asm['lattice']
    return dict(recday=recday, engine=ENGINE, engine_version=ENGINE_VERSION, units=units,
                sessions=asm['sessions'], n_tasks=asm['n_tasks'], lattice=dict(A=lat['A'], resid_max_cm=lat['resid_max_cm']),
                single=su, maps2d=m2, real=real, place_models=models, nulls=nulls)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def population_row(pop: dict) -> dict:
    """Flat scalar summary of one population result (one recipe or one simulation)."""
    row = {}
    r = pop.get('rsa', {})
    for n in (9, 21):
        if n in r:
            x = r[n]
            row.update({f'rsa{n}_ED_E': x['fit_ED']['beta'][0], f'rsa{n}_ED_X': x['fit_ED']['beta'][1],
                        f'rsa{n}_GD_G': x['fit_GD']['beta'][0], f'rsa{n}_GD_X': x['fit_GD']['beta'][1],
                        f'rsa{n}_cvr2_ED': x['cv_r2_ED'], f'rsa{n}_cvr2_GD': x['cv_r2_GD'],
                        f'rsa{n}_unique_E': x['unique_E'], f'rsa{n}_unique_G': x['unique_G'],
                        f'rsa{n}_contrast': x['diagnostic_contrast'], f'rsa{n}_perm_p_E': x['perm_p_ED'][0],
                        f'rsa{n}_perm_p_X': x['perm_p_ED'][1]})
    if 'thesis' in r:
        b = r['thesis']['fit_EPD']['beta']
        row.update(thesis_rsa_E=b[0], thesis_rsa_P=b[1], thesis_rsa_D=b[2])
    d = pop.get('decode', {})
    for k in ('corner_cardinal_ba', 'node_bridge_ba', 'location_acc', 'same_type_error_share', 'n_samples'):
        if k in d:
            row[f'dec_{k}'] = d[k]
    for k in ('corner_cardinal_null', 'node_bridge_null'):
        v = d.get(k)
        if v is not None and len(v):
            row[f'dec_{k}_p95'] = float(np.nanpercentile(v, 95))
    pl = pop.get('planes', {})
    if pl.get('recipe') == 'thesis':
        row['planes_thesis_within_max'] = float(np.nanmean(list(pl['within'].values())))
        row['planes_thesis_across_max'] = float(np.nanmean(list(pl['across'].values())))
        row['planes_thesis_across0_max'] = float(np.nanmean([v for (a, b), v in pl['across'].items() if a == 0]))
    else:
        for k in (2, 3):
            if k in pl:
                for part in ('halves', 'tasks', 'across'):
                    v = np.array(list(pl[k][part].values()), float)
                    row[f'planes_k{k}_{part}_min'] = float(np.nanmean(v[:, 0]))
                    row[f'planes_k{k}_{part}_max'] = float(np.nanmean(v[:, 1]))
    return row


def merge_tables(shards: dict) -> dict:
    """{'units', 'population', 'nulls'} DataFrames from a run's maze shards."""
    import pandas as pd
    urows, prows, nrows = [], [], []
    for rd, sh in sorted(shards.items()):
        mouse = rd.split('_')[0]
        su = sh['single']
        units = sh['units'].reset_index(drop=True)
        N = len(su['stab21'])
        m2 = sh['maps2d']['cross_task_r']
        for u in range(N):
            row = dict(recday=rd, mouse=mouse, unit=u, region=units.loc[u, 'region'] if 'region' in units else '',
                       stab21=su['stab21'][u], stab9=su['stab9'][u], stab_thr=su['stab_thr'][u], info=su['info'][u],
                       info_thr=su['info_thr'][u], stable=bool(su['stable'][u]), spatial=bool(su['spatial'][u]),
                       place_fields=sh['place_models'][u]['K'])
            for t, v in su['types'].items():
                row.update({f'type_{t}_mean': v['mean'][u], f'type_{t}_min': v['min'][u], f'type_{t}_p99': v['null_p99'][u],
                            f'is_{t}': bool(su['type_class'][t][u])})
            for (sm, sg), v in m2.items():
                row[f'xtask2d_{sm}_{sg}'] = v[u]
            urows.append(row)
        for recipe, pop in sh['real'].items():
            prows.append(dict(recday=rd, mouse=mouse, recipe=recipe, n_units=N, n_tasks=sh['n_tasks'],
                              lattice_resid_cm=sh['lattice']['resid_max_cm'], **population_row(pop)))
        real = population_row(sh['real']['decided'])
        for kind, sims in sh['nulls'].items():
            rows = [population_row({w: x}) for w, lst in sims.items() for x in lst]
            keys = sorted({k for r_ in rows for k in r_})
            for k in keys:
                vals = np.array([r_[k] for r_ in rows if k in r_], float)
                if not len(vals) or k not in real:
                    continue
                nrows.append(dict(recday=rd, mouse=mouse, null=kind, stat=k, n_sims=len(vals), real=real[k],
                                  null_mean=float(np.nanmean(vals)), null_p95=float(np.nanpercentile(vals, 95)),
                                  null_p05=float(np.nanpercentile(vals, 5)),
                                  frac_null_ge_real=float(np.nanmean(vals >= real[k]))))
    return dict(units=pd.DataFrame(urows), population=pd.DataFrame(prows), nulls=pd.DataFrame(nrows))
