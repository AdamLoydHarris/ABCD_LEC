"""
Mean firing rate heatmap: neurons × sessions.

Z-scores per neuron across all sessions concatenated, then takes the mean
within each session.  A positive value means the neuron fired above its
overall average during that session.

Main entry point
----------------
plot_mean_firing_rates(data_dic, mouse_recday, valid_sessions,
                       neuron_subset=None)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_mean_firing_rates(data_dic, mouse_recday, valid_sessions,
                           neuron_subset=None):
    """
    Neurons × sessions heatmap of mean firing rate.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional
    """
    sess_data = []
    n_neurons = None
    for sidx in valid_sessions:
        sess = data_dic[mouse_recday][sidx]
        raw = np.asarray(sess['Neuron_raw'], dtype=float)
        if neuron_subset is not None:
            raw = raw[np.asarray(neuron_subset)]
        if n_neurons is None:
            n_neurons = raw.shape[0]
        sess_data.append({'sidx': sidx, 'raw': raw,
                          'task': sess.get('Task', str(sidx))})

    # z-score per neuron across all sessions concatenated
    concat  = np.concatenate([sd['raw'] for sd in sess_data], axis=1)
    mu      = np.nanmean(concat, axis=1, keepdims=True)
    sd_     = np.nanstd(concat,  axis=1, keepdims=True)
    sd_safe = np.where(sd_ == 0, 1.0, sd_)

    # mean z-score per session → (n_neurons, n_sessions)
    matrix = np.column_stack([
        ((sd['raw'] - mu) / sd_safe).mean(axis=1)
        for sd in sess_data
    ])

    vlim = np.abs(matrix).max()
    fig, ax = plt.subplots(figsize=(0.5 * len(valid_sessions) + 1,
                                    0.2 * n_neurons + 1))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(sess_data)))
    ax.set_xticklabels([sd['task'] for sd in sess_data],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Neuron', fontsize=9)
    plt.colorbar(im, ax=ax, label='Mean z-scored firing rate', shrink=0.6)
    ax.set_title(f'Mean firing rate — {mouse_recday}',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()
