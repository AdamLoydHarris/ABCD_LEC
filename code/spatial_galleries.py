"""Gallery PDFs of every unit for one run of a spatial engine, one PDF per recday, from the shards.

    python code/spatial_galleries.py --engine maze --tag 21c552de331e --dataset mec lec pfc

Writes <dataset root>/data/spatial/<engine>/<tag>/figures/gallery_<recday>.pdf (versioned, never
overwritten). Reads shards only, no data_dic; run under sbatch because a run holds ~3000 units.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROOTS = {'mec': '/ceph/behrens/adam_harris/Taskspace_abstraction_mEC/data/spatial',
         'lec': '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/data/spatial',
         'pfc': '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data/data/spatial'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--engine', choices=('of', 'maze'), required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--dataset', nargs='+', choices=sorted(ROOTS), required=True)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import spatial_common as sc
    import spatial_plots as sp
    draw = sp.maze_gallery_pdf if args.engine == 'maze' else sp.of_gallery_pdf
    for d in args.dataset:
        run = os.path.join(ROOTS[d], args.engine, args.tag)
        fig_dir = os.path.join(run, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        shards = sorted(glob.glob(os.path.join(run, 'shards', '*.pkl')))
        for f in shards:
            sh = pickle.load(open(f, 'rb'))
            out = sc.versioned(os.path.join(fig_dir, f"gallery_{sh['recday']}.pdf"))
            draw(sh, out)
            print(f'{d} {sh["recday"]}: {out}', flush=True)
        print(f'{d}: {len(shards)} galleries', flush=True)


if __name__ == '__main__':
    main()
