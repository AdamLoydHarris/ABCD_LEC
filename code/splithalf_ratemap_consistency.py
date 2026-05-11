"""
Split-half consistency of task-space rate maps.

For each session with >= MIN_TRIALS trials:
  1. Concatenate Neurons_norm across trials → (n_neurons, n_trials * 360)
  2. Z-score per neuron across the full concatenated session
  3. Smooth along the bin axis with a Gaussian kernel
  4. Reshape back to (n_neurons, n_trials, 360)
  5. Average over first-half trials and last-half trials separately
  6. Compute per-neuron Pearson r between the two halved rate maps (360 bins each)

Main entry point
----------------
corrs, meta = compute_splithalf_correlations(data_dic, mouse_recday)
plot_splithalf_distribution(corrs, mouse_recday)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

MIN_TRIALS  = 8      # sessions with fewer trials are skipped
SIGMA_BINS  = 10.0    # Gaussian smoothing SD in bins
N_BINS      = 360    # bins per trial
NCOLS_POLAR = 5     # columns in polar rate-map grid


# ─────────────────────────────────────────────────────────────────────────────
def _splithalf_session(nn, sigma=SIGMA_BINS):
    """
    Compute per-neuron split-half Pearson r for one session.

    Parameters
    ----------
    nn : np.ndarray
        Shape (n_neurons, n_trials, N_BINS).
    sigma : float
        Gaussian smoothing SD in bins.

    Returns
    -------
    corrs : np.ndarray
        Shape (n_neurons,).  Pearson r for each neuron.
    """
    n_neurons, n_trials, n_bins = nn.shape

    # 1. Concatenate trials → (n_neurons, n_trials * n_bins)
    concat = nn.reshape(n_neurons, n_trials * n_bins).astype(float)

    # 2. Z-score per neuron
    mu  = np.nanmean(concat, axis=1, keepdims=True)
    sd  = np.nanstd(concat,  axis=1, keepdims=True)
    sd_safe = np.where(sd == 0, 1.0, sd)
    concat_z = (concat - mu) / sd_safe

    # 3. Smooth along bin axis
    concat_sm = gaussian_filter1d(concat_z, sigma=sigma, axis=1)

    # 4. Reshape back to trials
    trials_sm = concat_sm.reshape(n_neurons, n_trials, n_bins)

    # 5. First half vs last half (balanced: same number of trials each)
    half      = n_trials // 2
    first_avg = np.nanmean(trials_sm[:, :half,   :], axis=1)  # (n_neurons, n_bins)
    last_avg  = np.nanmean(trials_sm[:, -half:,  :], axis=1)  # (n_neurons, n_bins)

    # 6. Per-neuron Pearson r
    corrs = np.array([
        pearsonr(first_avg[ni], last_avg[ni])[0]
        for ni in range(n_neurons)
    ])

    return corrs


# ─────────────────────────────────────────────────────────────────────────────
def compute_splithalf_correlations(data_dic, mouse_recday,
                                   valid_sessions=None,
                                   neuron_subset=None,
                                   sigma=SIGMA_BINS,
                                   min_trials=MIN_TRIALS,
                                   key='Neurons_norm'):
    """
    Compute split-half rate-map correlations across all qualifying sessions.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list, optional
        Session keys to use.  Defaults to all keys in data_dic[mouse_recday].
    neuron_subset : array-like of int, optional
        Row indices of Neurons_norm to retain.
    sigma : float
        Gaussian smoothing SD in bins.
    min_trials : int
        Minimum number of trials for a session to be included.
    key : str
        Key to use for the rate-map array (e.g. 'Neurons_norm' or 'Smoothed_norm').

    Returns
    -------
    corrs : np.ndarray
        All per-neuron split-half correlations (pooled across sessions).
    meta : list of dict
        One entry per qualifying session with keys:
          'session', 'n_neurons', 'n_trials', 'corrs'
    """
    sessions = valid_sessions if valid_sessions is not None \
               else list(data_dic[mouse_recday].keys())

    all_corrs = []
    meta      = []

    for sidx in sessions:
        sess = data_dic[mouse_recday][sidx]

        if key not in sess:
            print(f"  Session {sidx}: '{key}' not found — skipping.")
            continue

        nn = np.array(sess[key], dtype=float)

        if nn.ndim != 3:
            print(f"  Session {sidx}: unexpected ndim={nn.ndim} — skipping.")
            continue

        n_neurons, n_trials, n_bins = nn.shape

        if n_trials < min_trials:
            print(f"  Session {sidx}: only {n_trials} trials "
                  f"(< {min_trials}) — skipping.")
            continue

        if n_bins != N_BINS:
            print(f"  Session {sidx}: n_bins={n_bins} (expected {N_BINS}) — "
                  f"proceeding anyway.")

        if neuron_subset is not None:
            nn = nn[np.asarray(neuron_subset)]
            n_neurons = nn.shape[0]

        corrs = _splithalf_session(nn, sigma=sigma)
        all_corrs.append(corrs)
        meta.append({
            'session'  : sidx,
            'n_neurons': n_neurons,
            'n_trials' : n_trials,
            'corrs'    : corrs,
        })

        print(f"  Session {sidx}: {n_trials} trials, {n_neurons} neurons — "
              f"median r = {np.nanmedian(corrs):.3f}")

    if not all_corrs:
        print(f"No qualifying sessions for {mouse_recday}.")
        return np.array([]), meta

    corrs_pooled = np.concatenate(all_corrs)
    print(f"\n{mouse_recday}: {len(meta)} sessions, "
          f"{len(corrs_pooled)} neurons total — "
          f"overall median r = {np.nanmedian(corrs_pooled):.3f}")

    return corrs_pooled, meta


# ─────────────────────────────────────────────────────────────────────────────
def plot_splithalf_distribution(corrs, mouse_recday, meta=None, bins=40):
    """
    Plot distribution of split-half Pearson correlations.

    If meta is provided, also shows per-session medians as a rug/overlay.

    Parameters
    ----------
    corrs : np.ndarray
        Per-neuron correlations (pooled).
    mouse_recday : str
    meta : list of dict, optional
    bins : int
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(corrs, bins=bins, color='steelblue', edgecolor='white',
            linewidth=0.4, alpha=0.85, label='per-neuron r')

    median_r = np.nanmedian(corrs)
    ax.axvline(median_r, color='black', lw=2, linestyle='--',
               label=f'median = {median_r:.3f}')
    ax.axvline(0, color='grey', lw=1, linestyle=':')

    # Per-session medians as tick marks at the top
    if meta:
        sess_medians = [np.nanmedian(m['corrs']) for m in meta]
        ymax = ax.get_ylim()[1]
        ax.plot(sess_medians,
                [ymax * 0.97] * len(sess_medians),
                'v', color='tomato', markersize=7, clip_on=False,
                label='session medians', zorder=5)

    ax.set_xlabel('Split-half Pearson r  (first half vs last half trials)',
                  fontsize=11)
    ax.set_ylabel('Number of neurons', fontsize=11)
    ax.set_title(f'Rate-map split-half consistency — {mouse_recday}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-1, 1)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def _preprocess_ratemaps_split(nn, sigma=SIGMA_BINS):
    """
    Z-score on the full session, circularly smooth each trial, then return
    mean and SEM across trials for the first and last halves separately.

    Smoothing is applied per-trial so the SEM reflects the same smoothed
    quantity as the mean.

    Parameters
    ----------
    nn : np.ndarray
        Shape (n_neurons, n_trials, n_bins).

    Returns
    -------
    first_mean, first_sem, last_mean, last_sem : np.ndarray
        Each shape (n_neurons, n_bins).
    """
    n_neurons, n_trials, n_bins = nn.shape

    # Z-score per neuron across the full concatenated session
    concat  = nn.reshape(n_neurons, n_trials * n_bins).astype(float)
    mu      = np.nanmean(concat, axis=1, keepdims=True)
    sd      = np.nanstd(concat,  axis=1, keepdims=True)
    sd_safe = np.where(sd == 0, 1.0, sd)
    trials_z = ((concat - mu) / sd_safe).reshape(n_neurons, n_trials, n_bins)

    # Smooth each trial individually with circular wrap → (n_neurons, n_trials, n_bins)
    trials_sm = gaussian_filter1d(trials_z, sigma=sigma, axis=2, mode='wrap')

    half = n_trials // 2

    def _mean_sem(t):  # t: (n_neurons, k, n_bins)
        k    = t.shape[1]
        mean = np.nanmean(t, axis=1)
        sem  = np.nanstd(t, axis=1) / np.sqrt(k)
        return mean, sem

    first_mean, first_sem = _mean_sem(trials_sm[:, :half,  :])
    last_mean,  last_sem  = _mean_sem(trials_sm[:, -half:, :])
    return first_mean, first_sem, last_mean, last_sem


def _draw_polar(ax, mean, sem, angles, angles_c, bins_per_state, state_colors, color):
    """Draw one polar rate map with mean ± SEM shading onto ax."""
    mean_c = np.append(mean, mean[0])
    sem_c  = np.append(sem,  sem[0])
    lo_c   = mean_c - sem_c
    hi_c   = mean_c + sem_c

    # Outer ring: state-boundary arcs
    r_arc = hi_c.max() * 1.15 if hi_c.max() > 0 else 1.0
    for si, sc in enumerate(state_colors):
        a0    = angles[si * bins_per_state]
        a1    = angles[min((si + 1) * bins_per_state, len(angles) - 1)]
        arc_a = np.linspace(a0, a1, 30)
        ax.plot(arc_a, [r_arc] * 30, color=sc, lw=3, solid_capstyle='butt')

    # SEM band between lo and hi (clipped to 0 at bottom)
    ax.fill_between(angles_c, np.clip(lo_c, 0, None), np.clip(hi_c, 0, None),
                    color=color, alpha=0.35, linewidth=0)

    # Mean line
    ax.plot(angles_c, np.clip(mean_c, 0, None), color=color, lw=0.9)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(pad=0)
    ax.spines['polar'].set_visible(False)


def plot_polar_ratemaps(data_dic, mouse_recday, session,
                        neuron_subset=None,
                        sigma=SIGMA_BINS,
                        key='Neurons_norm',
                        ncols=NCOLS_POLAR,
                        state_colors=('tab:red', 'tab:blue', 'tab:green', 'tab:orange')):
    """
    Plot a grid of polar rate maps showing both split halves per neuron.

    Each neuron occupies two adjacent polar subplots:
      left  (blue)  = first half of trials
      right (orange) = last half of trials

    Coloured arcs on the outer ring mark state boundaries (A/B/C/D).

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    session : int / key
    neuron_subset : array-like of int, optional
    sigma : float
    key : str
    ncols : int
        Number of neurons per row (figure has 2*ncols polar columns).
    state_colors : tuple of 4 colour strings
    """
    sess = data_dic[mouse_recday][session]
    if key not in sess:
        raise KeyError(f"'{key}' not found in session {session}.")

    nn = np.array(sess[key], dtype=float)
    if nn.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {nn.shape}.")

    if neuron_subset is not None:
        nn = nn[np.asarray(neuron_subset)]

    n_neurons, n_trials, n_bins = nn.shape
    first_mean, first_sem, last_mean, last_sem = _preprocess_ratemaps_split(
        nn, sigma=sigma)

    angles         = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    angles_c       = np.append(angles, angles[0])
    bins_per_state = n_bins // 4

    nrows      = int(np.ceil(n_neurons / ncols))
    total_cols = ncols * 2   # two polar panels per neuron

    fig, axes = plt.subplots(
        nrows, total_cols,
        figsize=(total_cols * 0.9, nrows * 1.4),
        subplot_kw={'projection': 'polar'},
    )
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for ni in range(n_neurons):
        row = ni // ncols
        col = (ni % ncols) * 2   # left of each pair

        ax_first = axes[row, col]
        ax_last  = axes[row, col + 1]

        _draw_polar(ax_first, first_mean[ni], first_sem[ni], angles, angles_c,
                    bins_per_state, state_colors, color='steelblue')
        _draw_polar(ax_last,  last_mean[ni],  last_sem[ni],  angles, angles_c,
                    bins_per_state, state_colors, color='darkorange')

        # Neuron label above the left panel
        ax_first.set_title(str(ni), fontsize=6, pad=1)
        ax_last.set_title('',       fontsize=6, pad=1)

    # Hide unused axes
    used = set()
    for ni in range(n_neurons):
        row = ni // ncols
        col = (ni % ncols) * 2
        used.add((row, col))
        used.add((row, col + 1))
    for r in range(nrows):
        for c in range(total_cols):
            if (r, c) not in used:
                axes[r, c].set_visible(False)

    task = sess.get('Task', '')
    fig.suptitle(
        f'Polar rate maps — {mouse_recday}  session {session}  [{task}]\n'
        f'{n_neurons} neurons · {n_trials} trials · σ={sigma} bins  '
        f'|  blue = first half   orange = last half',
        fontsize=9, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def _best_circular_shift(map1, map2):
    """
    Return the circular shift (bins) of map1 that maximises Pearson r with map2,
    and the corresponding angle in radians.

    Mirrors compute_remapping_angle from the notebook (cell 242).
    """
    n_bins = len(map1)
    corrs  = np.array([
        np.corrcoef(np.roll(map1, s), map2)[0, 1]
        for s in range(n_bins)
    ])
    best_shift = int(np.argmax(corrs))
    best_angle = best_shift * (2 * np.pi / n_bins)
    return best_shift, best_angle, corrs[best_shift]


def compute_splithalf_remapping_angles(data_dic, mouse_recday,
                                       valid_sessions=None,
                                       neuron_subset=None,
                                       sigma=SIGMA_BINS,
                                       min_trials=MIN_TRIALS,
                                       key='Neurons_norm'):
    """
    For each neuron in each qualifying session, find the circular shift of
    the first-half rate map that maximises its correlation with the last-half
    rate map.  Return the corresponding angles (radians) pooled across neurons.

    Parameters
    ----------
    data_dic, mouse_recday, valid_sessions, neuron_subset, sigma,
    min_trials, key : same as compute_splithalf_correlations.

    Returns
    -------
    angles : np.ndarray
        Per-neuron best-shift angles in radians, pooled across sessions.
    best_corrs : np.ndarray
        The maximised Pearson r at each neuron's best shift.
    meta : list of dict
        Per-session dicts with keys 'session', 'n_neurons', 'n_trials',
        'angles', 'best_corrs'.
    """
    sessions = valid_sessions if valid_sessions is not None \
               else list(data_dic[mouse_recday].keys())

    all_angles, all_corrs, meta = [], [], []

    for sidx in sessions:
        sess = data_dic[mouse_recday][sidx]

        if key not in sess:
            continue

        nn = np.array(sess[key], dtype=float)
        if nn.ndim != 3:
            continue

        n_neurons, n_trials, n_bins = nn.shape
        if n_trials < min_trials:
            continue

        if neuron_subset is not None:
            nn = nn[np.asarray(neuron_subset)]
            n_neurons = nn.shape[0]

        first_mean, _, last_mean, _ = _preprocess_ratemaps_split(nn, sigma=sigma)

        sess_angles = []
        sess_corrs  = []
        for ni in range(n_neurons):
            _, angle, best_r = _best_circular_shift(first_mean[ni], last_mean[ni])
            sess_angles.append(angle)
            sess_corrs.append(best_r)

        sess_angles = np.array(sess_angles)
        sess_corrs  = np.array(sess_corrs)
        all_angles.append(sess_angles)
        all_corrs.append(sess_corrs)
        meta.append({
            'session'   : sidx,
            'n_neurons' : n_neurons,
            'n_trials'  : n_trials,
            'angles'    : sess_angles,
            'best_corrs': sess_corrs,
        })
        print(f"  Session {sidx}: {n_trials} trials, {n_neurons} neurons — "
              f"median remapping angle = "
              f"{np.degrees(np.nanmedian(sess_angles)):.1f}°")

    if not all_angles:
        print(f"No qualifying sessions for {mouse_recday}.")
        return np.array([]), np.array([]), meta

    angles     = np.concatenate(all_angles)
    best_corrs = np.concatenate(all_corrs)
    print(f"\n{mouse_recday}: {len(meta)} sessions, {len(angles)} neurons pooled")
    return angles, best_corrs, meta



def plot_remapping_angle_histogram(angles, mouse_recday,
                                   n_bins=36, color='steelblue'):
    """
    Plot a polar histogram of split-half remapping angles.

    Parameters
    ----------
    angles : np.ndarray
        Per-neuron best-shift angles in radians (0 to 2π).
    mouse_recday : str
    n_bins : int
        Number of angular histogram bins (default 36 → 10° each).
    color : str
    """
    bin_edges   = np.linspace(0, 2 * np.pi, n_bins + 1)
    counts, _   = np.histogram(angles, bins=bin_edges)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    width       = 2 * np.pi / n_bins

    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw={'projection': 'polar'})

    ax.bar(bin_centres, counts, width=width * 0.9,
           color=color, edgecolor='white', linewidth=0.4, alpha=0.85)

    # Mark 0° (no shift) and 180° (half-cycle shift)
    ax.axvline(0,     color='black', lw=1.5, linestyle='--', alpha=0.6)
    ax.axvline(np.pi, color='tomato', lw=1.5, linestyle='--', alpha=0.6)

    ymax = counts.max() * 1.25

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=9)
    ax.set_ylim(0, ymax)

    ax.set_title(
        f'Split-half remapping angles\n{mouse_recday}  (n={len(angles)} neurons)',
        fontsize=11, fontweight='bold', pad=20,
    )
    plt.tight_layout()
    plt.show()
