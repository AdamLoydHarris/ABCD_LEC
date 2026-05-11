"""
Spatial firing rate maps on the task maze.

Locs_raw values: 1–9 = towers (nodes), 10–21 = corridors (edges).
The maze is visualised on a 5×5 pcolormesh grid with non-uniform cell edges:
  towers      → even row, even col  (small square cells)
  h-corridors → even row, odd  col  (wide, flat cells → long thin horizontal strips)
  v-corridors → odd  row, even col  (narrow, tall cells → long thin vertical strips)

White rectangles are overlaid on corridor cells to trim them to _CORR_THICKNESS,
so nodes appear larger than corridors while corridors still have the correct
orientation.

Firing rates are z-scored per neuron across all sessions concatenated so that
cross-session spatial tuning is directly comparable on a shared colour axis.

Main entry point
----------------
plot_spatial_ratemaps(data_dic, mouse_recday, valid_sessions,
                      neuron_subset=None)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

# ─── Maze layout ──────────────────────────────────────────────────────────────

LOC_TO_GRID = {
    # towers
    1: (0,0), 2: (0,2), 3: (0,4),
    4: (2,0), 5: (2,2), 6: (2,4),
    7: (4,0), 8: (4,2), 9: (4,4),
    # horizontal corridors (even row, odd col)
    10: (0,1), 11: (0,3),
    12: (2,1), 13: (2,3),
    14: (4,1), 15: (4,3),
    # vertical corridors (odd row, even col)
    16: (1,0), 17: (1,2), 18: (1,4),
    19: (3,0), 20: (3,2), 21: (3,4),
}

_NODE_SZ        = 3.0   # node cell side length
_CORR_LEN       = 9.0   # corridor cell long dimension
_CORR_THICKNESS = 1.0   # visible corridor strip thickness after masking

_CELL_EDGES = np.array([
    0,
    _NODE_SZ,
    _NODE_SZ  + _CORR_LEN,
    2*_NODE_SZ + _CORR_LEN,
    2*_NODE_SZ + 2*_CORR_LEN,
    3*_NODE_SZ + 2*_CORR_LEN,
], dtype=float)
# cell sizes along each axis: [NODE_SZ, CORR_LEN, NODE_SZ, CORR_LEN, NODE_SZ]
#                            = [3,       9,        3,       9,        3       ]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _add_corridor_masks(ax):
    """
    Overlay white rectangles on corridor cells so they appear as thin strips
    of width _CORR_THICKNESS centred within each cell.
    Also blanks the unused (odd-row, odd-col) corner cells.
    """
    t = _CORR_THICKNESS / 2.0
    for row in range(5):
        for col in range(5):
            x0, x1 = _CELL_EDGES[col], _CELL_EDGES[col + 1]
            y0, y1 = _CELL_EDGES[row], _CELL_EDGES[row + 1]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            kw = dict(facecolor='white', edgecolor='none', zorder=3)
            is_hcorr = (row % 2 == 0) and (col % 2 == 1)
            is_vcorr = (row % 2 == 1) and (col % 2 == 0)
            is_empty = (row % 2 == 1) and (col % 2 == 1)
            if is_hcorr:
                ax.add_patch(Rectangle((x0, y0),   x1-x0, cy-t-y0,    **kw))
                ax.add_patch(Rectangle((x0, cy+t), x1-x0, y1-(cy+t),  **kw))
            elif is_vcorr:
                ax.add_patch(Rectangle((x0,   y0), cx-t-x0,  y1-y0,  **kw))
                ax.add_patch(Rectangle((cx+t, y0), x1-(cx+t), y1-y0, **kw))
            elif is_empty:
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, **kw))


def _compute_spatial_ratemap(locs, rate_matrix):
    """
    Mean firing rate per maze location.

    Parameters
    ----------
    locs : (T,) int array, values 1–21
    rate_matrix : (n_neurons, T) float array

    Returns
    -------
    rm : (n_neurons, 5, 5) float array, NaN for unused grid cells
    """
    n_neurons = rate_matrix.shape[0]
    rm = np.full((n_neurons, 5, 5), np.nan)
    for loc_id, (gr, gc) in LOC_TO_GRID.items():
        mask = (locs == loc_id)
        if mask.any():
            rm[:, gr, gc] = rate_matrix[:, mask].mean(axis=1)
    return rm


# ─── Main entry point ─────────────────────────────────────────────────────────

def plot_spatial_ratemaps(data_dic, mouse_recday, valid_sessions,
                          neuron_subset=None,
                          cmap='RdBu_r',
                          cell_size_in=1.2):
    """
    Spatial rate maps, z-scored across all sessions concatenated.

    Rows = neurons, columns = sessions.  Colour scale is per-neuron and
    symmetric about zero (diverging), shared across sessions so that
    cross-session changes in spatial tuning are visible.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional
    cmap : str
        Diverging colourmap recommended (default 'RdBu_r').
    cell_size_in : float
        Figure size in inches per subplot cell.
    """
    # ── collect raw data ──────────────────────────────────────────────────────
    sess_data = []
    n_neurons = None
    for sidx in valid_sessions:
        sess = data_dic[mouse_recday][sidx]
        locs = np.asarray(sess['Locs_raw'], dtype=int).ravel()
        raw  = np.asarray(sess['Neuron_raw'], dtype=float)
        if neuron_subset is not None:
            raw = raw[np.asarray(neuron_subset)]
        if n_neurons is None:
            n_neurons = raw.shape[0]
        T = min(len(locs), raw.shape[1])
        sess_data.append({'sidx': sidx, 'locs': locs[:T],
                          'raw': raw[:, :T],
                          'task': sess.get('Task', str(sidx))})

    if not sess_data:
        print(f"No sessions for {mouse_recday}.")
        return

    # ── z-score per neuron across all sessions concatenated ───────────────────
    concat  = np.concatenate([sd['raw'] for sd in sess_data], axis=1)
    mu      = np.nanmean(concat, axis=1, keepdims=True)
    sd_     = np.nanstd(concat,  axis=1, keepdims=True)
    sd_safe = np.where(sd_ == 0, 1.0, sd_)
    for sd_item in sess_data:
        sd_item['raw'] = (sd_item['raw'] - mu) / sd_safe

    # ── compute ratemaps ──────────────────────────────────────────────────────
    ratemaps = [_compute_spatial_ratemap(sd['locs'], sd['raw'])
                for sd in sess_data]

    # ── figure ────────────────────────────────────────────────────────────────
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad('0.85')

    n_sessions = len(valid_sessions)
    fig, axes = plt.subplots(n_neurons, n_sessions,
                             figsize=(cell_size_in * n_sessions,
                                      cell_size_in * n_neurons),
                             squeeze=False)

    for ni in range(n_neurons):
        # symmetric colour scale, shared across sessions for this neuron
        all_vals = np.concatenate([rm[ni][~np.isnan(rm[ni])] for rm in ratemaps])
        vlim = np.abs(all_vals).max() if len(all_vals) else 1.0
        norm = Normalize(vmin=-vlim, vmax=vlim)

        for si, (rm, sd) in enumerate(zip(ratemaps, sess_data)):
            ax = axes[ni, si]
            ax.pcolormesh(_CELL_EDGES, _CELL_EDGES, rm[ni],
                          cmap=cmap_obj, norm=norm, rasterized=True)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            _add_corridor_masks(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ni == 0:
                ax.set_title(sd['task'], fontsize=6, pad=2)
            if si == 0:
                label = neuron_subset[ni] if neuron_subset is not None else ni
                ax.set_ylabel(f'n{label}', fontsize=5, rotation=0,
                              labelpad=20, va='center')

    fig.suptitle(f'Spatial rate maps (z-scored) — {mouse_recday}',
                 fontsize=9, fontweight='bold', y=1.002)
    plt.tight_layout()
    plt.show()
