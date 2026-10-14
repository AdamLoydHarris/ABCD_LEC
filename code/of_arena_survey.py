"""Survey of the arena sessions that sets the box50 settings of the open-field engine before it runs:
speed distribution and dwell per speed bin, object radius by both methods, wall straightness, HD guard,
samples per object location, and units per region.

    python code/of_arena_survey.py --dataset lec          (sbatch: it loads the data_dic)

Writes <root>/data/spatial/qc/of_arena_survey_<dataset>.csv (versioned) and prints the table.
mEC and LEC only: this file is not mirrored to mFC_data/code/.
"""

from __future__ import annotations

import argparse
import ast
import copy
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CAND_EDGES = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0)


def wall_rms(dataset: str) -> pd.DataFrame:
    """Latest valid wall RMS per (recday, session) over every version of the wall QC table."""
    import recday_registry as rr
    root = str(rr.DATASETS[dataset]['root'])
    base = os.path.join(root, 'data', 'preprocessed', 'sleap', f'openfield_to_xy_qc_{dataset}_v2.csv')
    files = [base] + sorted(glob.glob(base[:-4] + '.v*.csv'), key=lambda f: int(f.rsplit('.v', 1)[1][:-4]))
    q = pd.concat([pd.read_csv(f).assign(_v=i) for i, f in enumerate(files) if os.path.exists(f)], ignore_index=True)
    q = q[q['wall_rms_cm'].astype(str).str.startswith('{')]
    return q.sort_values('_v', kind='stable').groupby(['recday', 'session']).tail(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', choices=('mec', 'lec'), required=True)
    args = ap.parse_args(argv)
    import spatial_common as sc
    import spatial_io as sio
    sio.set_dataset(args.dataset)
    import spatial_tuning as st
    # every recday in the v4 data_dic: the registry also lists the July 6-day LEC sorts, which v4 lacks
    views = sio.load(args.dataset, None, 'v4')
    walls = wall_rms(args.dataset)
    regions = {}
    if args.dataset == 'lec':
        import anatomy_split as asp
        regions = asp.load_unit_regions()
    rows = []
    for rd in sorted(views):
        for s, v in sorted(views[rd].items()):
            if v.get('session_type') not in sio.ARENA_TYPES or v.get('xy') is None or v.get('arena') is None:
                continue
            cfg = st.arena_cfg(copy.deepcopy(st.DEFAULT_CFG), v['arena'])
            side = float(v['arena']['side_cm'])
            T = min(v['counts'].shape[1], len(v['xy']))
            xy = np.asarray(v['xy'][:T], float)
            fin = np.isfinite(xy).all(1)
            xy_c = np.clip(xy, 0, side - 1e-6)
            spd = sc.speed_cm_s(xy)
            mov = fin & (spd > cfg['min_speed'])
            row = dict(recday=rd, session=s, mouse=rd.split('_')[0], arena=v['arena']['name'], block=cfg['arena_block'],
                       side_cm=side, minutes=T * sc.BIN_S / 60, moving_frac=float(mov.mean()))
            for qq in (50, 90, 95, 99):
                row[f'speed_p{qq}'] = float(np.nanpercentile(spd[mov], qq))
            e = np.r_[CAND_EDGES, np.inf]
            h = np.histogram(spd[mov], e)[0] * sc.BIN_S
            for lo, hi_, dwell in zip(e[:-1], e[1:], h):
                row[f'dwell_s_{lo:g}_{hi_:g}'] = float(dwell)
            objs = v.get('objects') or []
            cs = [np.asarray(o['xy_cm'], float) for o in objs]
            for i, (o, c) in enumerate(zip(objs, cs)):
                others = [q for q in cs if not np.array_equal(q, c)]
                legacy = copy.deepcopy(cfg)
                legacy['obj'].update(radius_norm='annulus', radius_ref_cm=(15.0, 35.0), radius_search_cm=40.0)
                r_new = st.measure_object_radius(xy_c[fin], c, cfg, side=side, others=others)
                r_old = st.measure_object_radius(xy_c[fin], c, legacy)
                wall = min(c[0], c[1], side - c[0], side - c[1])
                band = mov & (np.hypot(xy_c[:, 0] - c[0], xy_c[:, 1] - c[1]) >= r_new) & \
                    (np.hypot(xy_c[:, 0] - c[0], xy_c[:, 1] - c[1]) < cfg['move']['r_out_cm'])
                row.update({f'obj{i}_label': o.get('label', ''), f'obj{i}_x': c[0], f'obj{i}_y': c[1],
                            f'obj{i}_r_in_arena': r_new, f'obj{i}_r_legacy': r_old, f'obj{i}_wall_cm': wall,
                            f'obj{i}_mask_to_wall_cm': wall - (r_new + cfg['obj']['margin_cm']),
                            f'obj{i}_band_samples': int(band.sum()), f'obj{i}_r_gt_7': bool(r_new > 7.0)})
            w = walls[(walls['recday'] == rd) & (walls['session'] == s)]
            rms = ast.literal_eval(str(w['wall_rms_cm'].iloc[-1])) if len(w) else {}
            row.update({f'wall_rms_{k}': val for k, val in rms.items()})
            hd = v.get('hd')
            if hd is not None:
                g = sc.hd_guard(xy, np.asarray(hd[:T], float), spd, cfg['hd']['guard_speed'], cfg['hd']['guard_offset_deg'], cfg['hd']['guard_r'])
                row.update(hd_guard_ok=g['ok'], hd_offset_deg=g['offset_deg'], hd_R=g['R'])
            row['n_units'] = v['counts'].shape[0]
            if rd in regions:
                row.update({f'units_{k}': int(n) for k, n in regions[rd]['group'].value_counts().items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    out = sc.versioned(os.path.join(sio.output_root(args.dataset), 'qc', f'of_arena_survey_{args.dataset}.csv'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option('display.width', 250)
    cols = ['recday', 'session', 'block', 'minutes', 'speed_p50', 'speed_p90', 'speed_p99'] + \
           [c for c in df.columns if c.startswith('dwell_s_')]
    print(df[cols].round(1).to_string(index=False))
    ocols = [c for c in df.columns if c.startswith('obj') and any(k in c for k in ('r_in_arena', 'r_legacy', 'mask_to_wall', 'band_samples'))]
    print(df[['recday', 'session'] + ocols].round(1).to_string(index=False))
    wcols = [c for c in df.columns if c.startswith('wall_rms_') or c.startswith('hd_')]
    print(df[['recday', 'session'] + wcols].round(2).to_string(index=False))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
