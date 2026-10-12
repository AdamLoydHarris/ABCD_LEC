"""Loader adapters for the spatial pipelines: the ONLY per-dataset code (user rule 2026-10-09: the same
pipeline runs on mEC, LEC and PFC; datasets differ only in how they are read).

`load(dataset, recdays, version)` returns `{recday: {session: view}}` where a view is
    counts (N, T) float32 spikes per 25 ms bin | xy (T, 2) cm, NaN unknown | hd (T,) rad or None |
    locs (T,) int16, 1-21 maze location, 0 unknown | trial_times (n, 5) int bins or None |
    session_type, task, arena, objects.
`unit_table(dataset, recday, n_units)` returns one row per `Neuron_raw` row.

mEC / LEC go through `glm_analysis_v3.load_data_dic` (validated v4 pickles, merged recdays) and must
run from `code/` with TSA_DATASET set before import (`set_dataset`). PFC goes through
`build_data_dic_from_pfc` with XY converted to cm by `pfc_xy_to_cm` and must run from `mFC_data/code/`.

This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).
"""

from __future__ import annotations

import ast
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

IS_PFC_TREE = 'mFC_data' in HERE
DATASETS = ('mec', 'lec', 'pfc')
FS_AP_MS = 30.0                     # Neuropixels samples per ms, for template widths
ARENA_TYPES = ('openfield', 'object')


def _repo_root() -> str:
    p = HERE
    for _ in range(4):
        if os.path.isdir(os.path.join(p, 'code')) and os.path.isdir(os.path.join(p, 'mFC_data')):
            return p
        p = os.path.dirname(p)
    raise RuntimeError('cannot locate the repo root from ' + HERE)


REPO = _repo_root()
PFC_DATA = os.path.join(REPO, 'mFC_data', 'data')


def set_dataset(dataset: str) -> None:
    """Set TSA_DATASET for the registry; must run before `recday_registry` / `glm_analysis_v3` import."""
    if dataset not in DATASETS:
        raise ValueError(f'dataset {dataset!r} not in {DATASETS}')
    if dataset == 'pfc':
        if not IS_PFC_TREE:
            raise RuntimeError('PFC runs from mFC_data/code (its glm_analysis_v3 carries the PFC loader)')
        return
    if IS_PFC_TREE:
        raise RuntimeError(f'{dataset} runs from code/, not mFC_data/code')
    if 'recday_registry' in sys.modules and sys.modules['recday_registry'].DATASET != dataset:
        raise RuntimeError('recday_registry was imported before set_dataset; set TSA_DATASET first')
    os.environ['TSA_DATASET'] = dataset


def output_root(dataset: str) -> str:
    """`<dataset root>/data/spatial`: mEC results land in the mEC tree, PFC in mFC_data/data."""
    if dataset == 'pfc':
        return os.path.join(PFC_DATA, 'spatial')
    import recday_registry as rr
    return os.path.join(str(rr.DATASETS[dataset]['root']), 'data', 'spatial')


def latest_versioned(path: str) -> str:
    """The newest of path, path.v1, path.v2 ... (`versioned()` writes a new suffix per rebuild)."""
    stem, ext = os.path.splitext(path)
    cands = [p for p in glob.glob(stem + '*' + ext) if re.fullmatch(re.escape(stem) + r'(\.v\d+)?' + re.escape(ext), p)]
    if not cands:
        raise FileNotFoundError(path)
    def k(p):
        m = re.search(r'\.v(\d+)' + re.escape(ext) + '$', p)
        return int(m.group(1)) if m else 0
    return max(cands, key=k)


# ---------------------------------------------------------------------------
# recday lists and loading
# ---------------------------------------------------------------------------

def default_recdays(dataset: str, data_version: str = 'v4') -> list:
    """Every merged recday of the dataset (mEC/LEC: the registry's sorted blocks; PFC: El-Gaby's list)."""
    if dataset == 'pfc':
        return [str(x) for x in np.load(os.path.join(PFC_DATA, 'MetaData', 'combined_ABCDonly_days.npy'),
                                        allow_pickle=True)]
    import recday_registry as rr
    return sorted(r for r in rr.all_recdays() if rr.recday_kind(r) == 'merged'
                  and r not in getattr(rr, 'EXCLUDE_RECDAYS', {}))


def load(dataset: str, recdays: list | None, data_version: str = 'v4') -> dict:
    """{recday: {session: view}} for the requested recdays (None = all merged recdays)."""
    import glm_analysis_v3 as g3
    if dataset == 'pfc':
        recdays = recdays or default_recdays('pfc')
        dd = g3.build_data_dic_from_pfc(PFC_DATA, recdays, verbose=False)
        aff = pd.read_csv(latest_versioned(os.path.join(PFC_DATA, 'processed', 'pfc_xy_to_cm.csv')))
        aff = aff.set_index(['recday', 'session'])
    else:
        dd = g3.load_data_dic(dataset=dataset, version=data_version, recday_kind='merged', verbose=False)
        aff = None
        if recdays:
            missing = [r for r in recdays if r not in dd]
            if missing:
                raise KeyError(f'not in the {dataset} {data_version} data_dic: {missing}')
            dd = {r: dd[r] for r in recdays}
    out = {}
    for rd, sessions in dd.items():
        out[rd] = {int(s): _view(dataset, rd, int(s), sd, aff) for s, sd in sessions.items()}
    return out


def _locs_int(locs) -> np.ndarray:
    a = np.asarray(locs, dtype=object) if np.asarray(locs).dtype == object else np.asarray(locs, float)
    v = np.array([np.nan if x is None else float(x) for x in a], float) if a.dtype == object else a
    v = np.where(np.isfinite(v), v, 0)
    if np.any((v < 0) | (v > 21)):
        raise ValueError('Locs_raw outside 0..21')
    return v.astype(np.int16)


def _view(dataset: str, recday: str, s: int, sd: dict, aff) -> dict:
    counts = np.asarray(sd['Neuron_raw'], dtype=np.float32)
    T = counts.shape[1]
    xy = sd.get('XY_raw')
    xy = None if xy is None else np.asarray(xy, float)
    if dataset == 'pfc' and xy is not None:
        xy = _pfc_cm(recday, s, xy, aff)
    hd = sd.get('HD_raw')
    if hd is not None:
        hd = np.asarray(hd, float)
        if hd.ndim != 2 or hd.shape[1] != 2:
            # The GLM once flattened (T, 2) HD; every HD result before the fix was noise.
            raise ValueError(f'{recday} s{s}: HD_raw shape {hd.shape}, expected (T, 2)')
        hd = np.deg2rad(hd[:, 0])              # back2mid; column 1 is the ear axis (+90 deg)
    locs = sd.get('Locs_raw')
    locs = None if locs is None else _locs_int(locs)
    tt = sd.get('Trial_times')
    tt = None if tt is None else np.asarray(tt).astype(np.int64)
    stype = sd.get('session_type', 'ABCD' if dataset == 'pfc' else None)
    task = sd.get('Task')
    task = None if task is None else tuple(int(x) for x in np.asarray(task).ravel()[:4])
    return dict(recday=recday, session=s, dataset=dataset, counts=counts, T_neural=T, xy=xy, hd=hd,
                locs=locs, trial_times=tt, session_type=stype, task=task,
                arena=sd.get('arena'), objects=sd.get('objects'))


def _pfc_cm(recday: str, s: int, xy_px: np.ndarray, aff: pd.DataFrame) -> np.ndarray:
    """PFC XY pixels -> cm with the per-session affine of `pfc_xy_to_cm`; NaN when not `xy_ok`."""
    out = np.full(xy_px.shape, np.nan)
    if (recday, s) not in aff.index:
        return out
    a = aff.loc[(recday, s)]
    if not bool(a.get('xy_ok', True)) or not np.isfinite(a.get('a11', np.nan)):
        return out
    A = np.array([[a.a11, a.a12, a.b1], [a.a21, a.a22, a.b2]], float)
    good = np.isfinite(xy_px).all(axis=1) & (xy_px >= 0).all(axis=1)
    cm = np.column_stack([xy_px, np.ones(len(xy_px))]) @ A.T
    cm[~good] = np.nan
    return cm


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def unit_table(dataset: str, recday: str, n_units: int) -> pd.DataFrame:
    """One row per `Neuron_raw` row: cluster id, waveform width (ms), shank position, sorter metrics,
    duplicate partner, region. PFC has no sorter output, so only the row index and region."""
    df = pd.DataFrame(dict(unit=np.arange(n_units)))
    df['recday'] = recday
    df['mouse'] = recday.split('_')[0]
    if dataset == 'pfc':
        df['region'] = 'PFC'
        for c in ('cluster_id', 'width_ms', 'x_um', 'y_um', 'drift_ptp', 'sync_spike_2'):
            df[c] = np.nan
        df['dup_partner'] = -1
        return df
    import recday_registry as rr
    ko = rr.kilosort_dir(recday)
    ids = np.load(ko / 'QC_single_units.npy').astype(int)
    if len(ids) != n_units:
        raise ValueError(f'{recday}: {len(ids)} QC units vs {n_units} Neuron_raw rows')
    sa = ko / 'sorting_analyzer'
    orig = sa / 'sorting' / 'properties' / 'original_cluster_id.npy'
    if orig.exists():
        o = np.load(orig)
        if not np.array_equal(o, np.arange(len(o))):
            raise ValueError(f'{recday}: original_cluster_id is not the identity; index by it')
    tmpl = np.load(sa / 'extensions' / 'templates' / 'average.npy', mmap_mode='r')
    width = np.full(n_units, np.nan)
    for i, cid in enumerate(ids):
        t = np.asarray(tmpl[cid], float)                 # (samples, channels), indexed by cluster id
        w = t[:, int(np.argmax(np.ptp(t, axis=0)))]
        tr = int(np.argmin(w))
        width[i] = (int(np.argmax(w[tr:]))) / FS_AP_MS
    loc = np.load(sa / 'extensions' / 'unit_locations' / 'unit_locations.npy')
    df['cluster_id'] = ids
    df['width_ms'] = width
    df['x_um'] = loc[ids, 0]
    df['y_um'] = loc[ids, 1]
    # 4 LEC blocks (ah08 06-18/19, 06-20/23; ly05 06-13/15, 06-20/23) have no quality_metrics.htsv;
    # their analyzer metrics.csv reproduces the QC selection exactly (passing set == QC units)
    if (ko / 'quality_metrics.htsv').exists():
        qm, source = pd.read_csv(ko / 'quality_metrics.htsv', sep='\t').set_index('unit_id'), 'htsv'
    else:
        qm, source = pd.read_csv(sa / 'extensions' / 'quality_metrics' / 'metrics.csv', index_col=0), 'metrics.csv'
    qm = qm.loc[ids]
    gate = rr.QC_GATE
    ok = ((qm['isi_violations_ratio'] < gate['isi_violations_ratio_lt'])
          & (qm['amplitude_cutoff'] < gate['amplitude_cutoff_lt'])
          & (qm['firing_rate'] > gate['firing_rate_gt'])
          & (qm['presence_ratio'] > gate['presence_ratio_gt'])
          & (qm['amplitude_median'].abs() > gate['amplitude_median_abs_gt'])
          & (qm['sd_ratio'] < gate['sd_ratio_lt']))
    if not ok.all():
        raise ValueError(f'{recday}: {int((~ok).sum())} joined units fail QC_GATE; the join is wrong')
    df['metrics_source'] = source
    df['drift_ptp'] = qm['drift_ptp'].to_numpy()
    df['sync_spike_2'] = qm['sync_spike_2'].to_numpy()
    df['dup_partner'] = _duplicate_partners(sa, ids)
    if dataset == 'lec':
        import anatomy_split as asp
        ur = asp.load_unit_regions()[recday].reset_index(drop=True)
        if len(ur) != n_units:
            raise ValueError(f'{recday}: {len(ur)} region rows vs {n_units} units')
        df['region'] = ur['group'].to_numpy()
    else:
        df['region'] = 'MEC'                          # unregistered: anatomy deferred
    return df


def _duplicate_partners(sa, ids: np.ndarray, sim_thr: float = 0.9, sync_ratio: float = 5.0) -> np.ndarray:
    """Per unit, the row of a likely duplicate (template similarity >= 0.9 AND a zero-lag CCG peak
    >= 5x the flank mean), else -1. Kilosort splits can turn one border cell into two."""
    out = np.full(len(ids), -1)
    simp = sa / 'extensions' / 'template_similarity' / 'similarity.npy'
    ccgp = sa / 'extensions' / 'correlograms' / 'ccgs.npy'
    binp = sa / 'extensions' / 'correlograms' / 'bins.npy'
    if not simp.exists():
        return out
    sim = np.load(simp, mmap_mode='r')
    ccg = np.load(ccgp, mmap_mode='r') if ccgp.exists() else None
    bins = np.load(binp) if binp.exists() else None
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if sim[ids[a], ids[b]] < sim_thr:
                continue
            dup = True
            if ccg is not None and bins is not None:
                c = np.asarray(ccg[ids[a], ids[b]], float)
                centre = len(c) // 2
                flank = np.r_[c[:max(centre - 5, 1)], c[centre + 6:]]
                dup = flank.mean() > 0 and c[centre - 1:centre + 2].mean() >= sync_ratio * flank.mean()
            if dup:
                out[a], out[b] = b, a
    return out


def wall_exclusions(dataset: str, recday: str, session: int, max_rms_cm: float) -> dict:
    """{wall: rms_cm} of walls whose occupancy edge bends by more than `max_rms_cm` (excluded from CM).

    Wall names follow `arena_calibration.straight_wall_test`: left x=0, right x=side, bottom y=0,
    top y=side (y-down frame, so 'bottom' is the image top)."""
    if dataset == 'pfc':
        return {}
    import recday_registry as rr
    root = str(rr.DATASETS[dataset]['root'])
    path = latest_versioned(os.path.join(root, 'data', 'preprocessed', 'sleap', f'openfield_to_xy_qc_{dataset}_v2.csv'))
    qc = pd.read_csv(path)
    row = qc[(qc['recday'] == recday) & (qc['session'] == session)]
    if row.empty:
        return {}
    rms = ast.literal_eval(str(row['wall_rms_cm'].iloc[-1]))
    return {k: v for k, v in rms.items() if v is not None and v > max_rms_cm}
