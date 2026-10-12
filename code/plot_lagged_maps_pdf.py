"""Every state-tuned neuron's lagged maps, one page each, in one PDF per region.

    python plot_lagged_maps_pdf.py --region lec|pfc [--limit N]

Per-neuron numbers come from `lagged_spatial_maps.pkl` (run_lagged_spatial_maps.py), the maps
from El-Gaby's saved `Phase_shifted_maps`. Order: lagged 90°, then lagged 30° only, then lag-0 /
other, each by held-out r descending. Page 1 gives the order and the page ranges.
Writes `<out dir>/lagged_maps_state_tuned_<region>.pdf`.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl                      # noqa: E402
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages   # noqa: E402

import lagged_spatial_maps as lsm             # noqa: E402

CLASSES = [('lagged 90°', 'r_strict'), ('lagged 30° only', 'r_nonzero'), ('not lagged', 'r_all')]


def page_order(t):
    """State-tuned rows in page order, with a `kind` column."""
    t = t[t.state_tuned].copy()
    t['kind'] = np.where(t.lagged90, CLASSES[0][0],
                         np.where(t.lagged30, CLASSES[1][0], CLASSES[2][0]))
    parts = [t[t.kind == k].sort_values(col, ascending=False, na_position='last')
             for k, col in CLASSES]
    import pandas as pd
    return pd.concat(parts, ignore_index=True)


def _fmt(x, f='{:.2f}'):
    return '--' if x is None or not np.isfinite(x) else f.format(x)


def title(region, r):
    g = f" · {r.group} ({r.acronym})" if isinstance(r.get('group'), str) else ''
    v5 = r.get('selected_eg')
    v5 = '--' if not isinstance(v5, (bool, np.bool_)) else ('yes' if v5 else 'no')
    lag = r.best_lag_modal
    return (f"{region.upper()} {r.recday} n{int(r.neuron)}{g} · {_fmt(r.rate_hz, '{:.1f}')} Hz · "
            f"{r.kind}\n"
            f"held-out r: all {_fmt(r.r_all)}, 30° {_fmt(r.r_nonzero)}, 90° {_fmt(r.r_strict)} · "
            f"modal best lag {_fmt(30 * lag, '{:.0f}')}° · place r {_fmt(r.r_place)} "
            f"(p {_fmt(r.p_place, '{:.3f}')}) · goal r {_fmt(r.r_goal)} "
            f"(p {_fmt(r.p_goal, '{:.3f}')}) · V5 selected {v5}")


def cover(region, order, n_pages):
    fig = plt.figure(figsize=(7.2, 4.5))
    lines = [f'{region.upper()}: lagged spatial maps of every state-tuned neuron ({len(order)})',
             '',
             'Layout (El-Gaby Fig. 5e, 9 nodes): rows = tasks (first session of each distinct task',
             'with trials), columns = lag into the past with the current location on the right. Each',
             'map is normalised on its own; the zoomed column is the modal held-out best lag, with its',
             'max rate in Hz and goal letters a-d. Bars = mean cross-task r of the node maps per lag.',
             '',
             'Header numbers: held-out r = his cell-31 statistic (reproduces his cell 34 exactly);',
             '30° / 90° are NaN when the neuron is not in that non-zero-lag set. place / goal r = the',
             'held-out 4-vector frame test, one-sided p against a node-scramble null (200 draws).',
             '', 'Pages:']
    p = 2
    for k, _ in CLASSES:
        n = int((order.kind == k).sum())
        if n:
            lines.append(f'  {k:16s} {n:5d} neurons, pages {p}-{p + n - 1}, by held-out r')
            p += n
    lines += ['', f'{n_pages} pages. Source: code/plot_lagged_maps_pdf.py, code/LAGGED_SPATIAL.md']
    fig.text(0.06, 0.94, '\n'.join(lines), va='top', family='monospace', fontsize=7)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', choices=['lec', 'pfc'], required=True)
    ap.add_argument('--limit', type=int, default=None, help='first N pages only (timing)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    od = lsm.OUT_DIRS[args.region]
    with open(os.path.join(od, 'lagged_spatial_maps.pkl'), 'rb') as fh:
        t = pickle.load(fh)['table']
    order = page_order(t)
    if args.limit:
        order = order.head(args.limit)
    recs = {rd: lsm.load_recday(args.region, rd, rates=False) for rd in order.recday.unique()}
    out = args.out or os.path.join(od, f'lagged_maps_state_tuned_{args.region}.pdf')
    mpl.rcParams.update({**lsm._rc(), 'pdf.fonttype': 42, 'ps.fonttype': 42})
    t0 = time.time()
    n_pages = len(order) + 1
    with PdfPages(out) as pdf:
        fig = cover(args.region, order, n_pages)
        pdf.savefig(fig)
        plt.close(fig)
        for i, r in enumerate(order.itertuples(index=False), start=2):
            row = order.iloc[i - 2]
            lag = r.best_lag_modal if np.isfinite(r.best_lag_modal) else None
            fig = lsm.plot_lagged_maps(recs[r.recday], int(r.neuron), best_lag=lag,
                                       title=title(args.region, row))
            fig.text(0.99, 0.005, f'{i}/{n_pages}', ha='right', va='bottom', fontsize=5)
            pdf.savefig(fig)
            plt.close(fig)
            if i % 100 == 0:
                print(f'  {i}/{n_pages} pages, {time.time() - t0:.0f} s', flush=True)
        d = pdf.infodict()
        d['Title'] = f'Lagged spatial maps, {args.region.upper()} state-tuned neurons'
    print(f'{n_pages} pages -> {out} ({os.path.getsize(out) / 1e6:.1f} MB, '
          f'{time.time() - t0:.0f} s)\nEXIT=0')


if __name__ == '__main__':
    main()
