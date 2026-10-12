"""Figures for the spatial pipelines (open field and maze), read from shards and merged tables only.

GridMaze style: Arial 8 pt (fallback font path, the GridMaze package is not installed in maze_ephys),
no top/right spines, viridis for rate maps, RdBu_r for autocorrelograms, true / shuffle = #C03030 /
#555555, datasets mEC / LEC / PFC = #BE3455 / #0F4C81 / #88B04B. Maps are drawn `.T` with
origin='upper' so the y-down arena frame reads like the camera image.

This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TRUE, SHUFFLE, STONE, CAVIAR = '#C03030', '#555555', '#B4B2A9', '#2C2C2A'
DATASET_COLOURS = {'mec': '#BE3455', 'lec': '#0F4C81', 'pfc': '#88B04B'}
CLASS_COLOURS = {'grid': '#6B3FA0', 'border': '#BE3455', 'spatial': '#0F4C81', 'hd': '#009B77',
                 'speed': '#FFA500', 'object': '#9B1B30'}
ARIAL = '/ceph/behrens/max_kirkby/goal_sequencing_1/font/Arial.ttf'


def setup_style():
    """GridMaze rcParams: Arial 8 pt, thin axes, embeddable fonts."""
    import matplotlib as mpl
    from matplotlib import font_manager
    try:
        from GridMaze.analysis.core.font import setup_arial_font
        setup_arial_font()
    except Exception:
        if os.path.exists(ARIAL):
            font_manager.fontManager.addfont(ARIAL)
            mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8, 'figure.titlesize': 8,
                             'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                             'legend.fontsize': 8, 'axes.linewidth': 0.8})
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False


def _save(fig, out_path):
    import matplotlib as mpl
    if out_path:
        with mpl.rc_context({'savefig.bbox': None}):
            fig.savefig(out_path)


# ---------------------------------------------------------------------------
# open field
# ---------------------------------------------------------------------------

def plot_of_unit(axes, sess: dict, u: int):
    """One unit, four panels: rate map with objects, autocorrelogram, HD (raw and pxd), position factor."""
    side = sess['side_cm']
    det = sess['details']
    rate = det['rate'][u]
    ax = axes[0]
    ax.imshow(rate.T, origin='upper', cmap='viridis', extent=(0, side, side, 0), interpolation='nearest')
    for o in sess['objects']:
        ax.add_patch(__import__('matplotlib').patches.Circle(o['xy'], o['r_obj'], fill=False, color='w', lw=0.8))
    r = sess['real']
    cls = [k for k, v in sess['classes'].items() if v[u]]
    if not sess['eligible'][u]:
        cls.append('inelig.')
    if sess['narrow'][u]:
        cls.append('narrow')
    pk = np.nanmax(rate) if np.isfinite(rate).any() else 0.0
    ax.set_title(f"u{u} {pk:.1f} Hz {' '.join(cls)}\nb {r['border'][u]:.2f} g {r['grid'][u]:.2f}", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1]
    if 'autocorr' in det:
        ax.imshow(det['autocorr'][u].T, origin='upper', cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[2]
    if 'hd_curve' in det:
        c = det['hd_curve'][u]
        th = (np.arange(len(c)) + 0.5) * 2 * np.pi / len(c)
        ax.plot(np.r_[th, th[0]], np.r_[c, c[0]] / max(np.nanmax(c), 1e-9), color=SHUFFLE, lw=0.8)
        d = det['pxd_d'][u]
        thd = (np.arange(len(d)) + 0.5) * 2 * np.pi / len(d)
        ax.plot(np.r_[thd, thd[0]], np.r_[d, d[0]] / max(np.nanmax(d), 1e-9), color=TRUE, lw=1.0)
        ax.set_title(f"pxd R {r['pxd_r'][u]:.2f}", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[3]
    if 'p_factor' in det:
        n = int(round(np.sqrt(det['p_factor'].shape[1])))
        ax.imshow(det['p_factor'][u].reshape(n, n).T, origin='upper', cmap='viridis', interpolation='nearest')
        ax.set_title(f"pos. factor SI {r['si_p'][u]:.2f}", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def of_gallery_pdf(shard: dict, out_pdf: str, max_units: int | None = None):
    """Multi-page PDF per arena session: EVERY unit, ordered by border, grid and spatial score; units
    outside the pools (< 100 moving spikes, duplicate) are drawn and flagged 'inelig.'."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.projections import polar  # noqa: F401
    setup_style()
    with PdfPages(out_pdf) as pdf:
        for s, sess in sorted(shard['sessions'].items()):
            r = sess['real']
            order = np.lexsort((-np.nan_to_num(r['si_p'], nan=-9), -np.nan_to_num(r['grid'], nan=-9),
                                -np.nan_to_num(r['border'], nan=-9)))
            order = list(order)[:max_units]
            per_page = 5
            for start in range(0, len(order), per_page):
                fig = plt.figure(figsize=(7.5, 9.5))
                fig.subplots_adjust(hspace=0.75, wspace=0.35, top=0.92, bottom=0.03)
                for row, u in enumerate(order[start:start + per_page]):
                    axs = [fig.add_subplot(per_page, 4, row * 4 + 1), fig.add_subplot(per_page, 4, row * 4 + 2),
                           fig.add_subplot(per_page, 4, row * 4 + 3, projection='polar'),
                           fig.add_subplot(per_page, 4, row * 4 + 4)]
                    plot_of_unit(axs, sess, u)
                fig.suptitle(f"{shard['recday']} s{s} {sess['session_type']}", fontsize=8)
                pdf.savefig(fig)
                plt.close(fig)


def plot_class_counts(sessions_df, ax=None, out_path=None):
    """Observed class counts per session against the chance mean and p95 (one shift for all units)."""
    import matplotlib.pyplot as plt
    setup_style()
    df = sessions_df[(sessions_df['pool'] == 'broad') & sessions_df['cls'].isin(['grid', 'border', 'spatial', 'hd', 'speed'])]
    labels = [f"{r.recday.split('_')[0]} {r.recday.split('_')[1][4:]} s{r.session}" for r in df.drop_duplicates(['recday', 'session']).itertuples()]
    fig, axs = plt.subplots(1, 5, figsize=(7.5, 2.4), sharey=False) if ax is None else (ax.figure, ax)
    for k, cls in enumerate(['grid', 'border', 'spatial', 'hd', 'speed']):
        d = df[df['cls'] == cls]
        x = np.arange(len(d))
        a = axs[k]
        a.bar(x, d['observed'] / d['n_units'], color=CLASS_COLOURS[cls], width=0.7, label='observed')
        # chance counts are skewed (mostly 0), so the mean can exceed p95: draw both as markers
        a.scatter(x, d['chance_mean'] / d['n_units'], marker='_', color=SHUFFLE, s=40, label='chance mean')
        a.scatter(x, d['chance_p95'] / d['n_units'], marker='^', color=SHUFFLE, s=8, label='chance p95')
        a.set_title(cls)
        a.set_xticks(x)
        a.set_xticklabels(labels[:len(d)], rotation=90, fontsize=6)
    axs[0].set_ylabel('fraction of broad units')
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# maze
# ---------------------------------------------------------------------------

def maze_grid_image(map21: np.ndarray) -> np.ndarray:
    """5x5 image of a 21-location map: nodes at even (row, col), bridges between, holes NaN."""
    from lagged_spatial_maps import EDGES, EDGE_GRID
    img = np.full((5, 5), np.nan)
    for k in range(1, 10):
        img[2 * ((k - 1) // 3), 2 * ((k - 1) % 3)] = map21[k - 1]
    for i, (r, c) in enumerate(EDGE_GRID):
        img[r, c] = map21[9 + i]
    return img


def plot_maze_unit_maps(task_maps: dict, u: int, axes):
    """One unit's 21-location map in each task (viridis, shared scale)."""
    vmax = np.nanmax([np.nanmax(m[u]) for m in task_maps.values()])
    for ax, (t, m) in zip(axes, sorted(task_maps.items())):
        ax.imshow(maze_grid_image(m[u]), cmap='viridis', vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'task {t}', fontsize=6)


def plot_population_vs_null(pop_df, nulls_df, stats: list, out_path=None, null_kind: str = 'place'):
    """Per dataset and mouse: each statistic's real value against one null family (mean and p95)."""
    import matplotlib.pyplot as plt
    setup_style()
    fig, axs = plt.subplots(1, len(stats), figsize=(2.2 * len(stats), 2.4))
    for a, st in zip(np.atleast_1d(axs), stats):
        d = nulls_df[(nulls_df['stat'] == st) & (nulls_df['null'] == null_kind)]
        for i, (ds, g) in enumerate(d.groupby('dataset') if 'dataset' in d else [('', d)]):
            gm = g.groupby('mouse')[['real', 'null_mean', 'null_p95']].mean()
            x = np.full(len(gm), i) + np.linspace(-0.2, 0.2, len(gm))
            a.scatter(x, gm['real'], color=DATASET_COLOURS.get(ds, TRUE), s=12, zorder=3, label=f'{ds} real')
            a.errorbar(x, gm['null_mean'], yerr=[np.zeros(len(gm)), gm['null_p95'] - gm['null_mean']], fmt='_',
                       color=SHUFFLE, lw=0.8)
        a.set_title(st, fontsize=7)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def maze_gallery_pdf(shard: dict, out_pdf: str, max_units: int | None = None):
    """Multi-page PDF of EVERY unit on the maze: pooled 2-D map (bridge-safe smoother, canonical frame)
    and the 21-location map in each task, ordered by cross-task stability."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    setup_style()
    su = shard['single']
    m2 = shard['maps2d']
    key = next(k for k in m2['pooled'] if k[0] == 'normconv')
    pooled = m2['pooled'][key]
    ex = m2['edges']
    tasks = sorted(su['task_maps'])
    order = np.argsort(-np.nan_to_num(su['stab21'], nan=-9))[:max_units]
    per_page = 8
    ncol = 1 + len(tasks)
    with PdfPages(out_pdf) as pdf:
        for start in range(0, len(order), per_page):
            fig, axs = plt.subplots(per_page, ncol, figsize=(1.0 * ncol + 1.2, 9.5), squeeze=False)
            for row in range(per_page):
                for a in axs[row]:
                    a.axis('off')
            for row, u in enumerate(order[start:start + per_page]):
                a = axs[row][0]
                a.axis('on')
                a.imshow(pooled[u].T, origin='upper', cmap='viridis', extent=(ex[0], ex[-1], ex[-1], ex[0]),
                         interpolation='nearest')
                flags = [t for t, v in su['type_class'].items() if v[u]]
                a.set_title(f"u{u} r {su['stab21'][u]:.2f}{' ' + ' '.join(flags) if flags else ''}", fontsize=6)
                a.set_xticks([]); a.set_yticks([])
                vmax = np.nanmax([np.nanmax(su['task_maps'][t][u]) for t in tasks])
                for j, t in enumerate(tasks):
                    a = axs[row][1 + j]
                    a.axis('on')
                    a.imshow(maze_grid_image(su['task_maps'][t][u]), cmap='viridis', vmin=0, vmax=vmax,
                             interpolation='nearest')
                    a.set_xticks([]); a.set_yticks([])
                    if row == 0:
                        a.set_title(f'task {t}', fontsize=6)
            fig.suptitle(f"{shard['recday']} maze: pooled 2-D map and 21-location map per task", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
