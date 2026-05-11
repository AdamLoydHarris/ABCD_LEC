"""
Circular correlation structure in task-state neural representations.

For each session, a 'reward-point' population vector is extracted for each
(trial × state) by averaging the first n_onset_bins time bins of Neurons_norm
(z-scored across the full session) immediately following each state onset.

Pairwise Pearson correlations are then computed only for trial pairs separated
by at least trial_gap trials (default 5) to control for temporal
autocorrelation.  This yields a (n_states × n_states) mean correlation matrix.

Circular distance hypothesis
----------------------------
States form a ring: A → B → C → D → A.  Circular distance is
min(|s-t|, n_states - |s-t|), so A-D = distance 1 (not 3).

  distance-1 pairs (A-B, B-C, C-D, D-A)   — expected HIGHER correlation
  distance-2 pairs (A-C, B-D)              — expected LOWER  correlation

Main entry point
----------------
results = run_circular_correlation(
    data_dic, mouse_recday, valid_sessions,
    min_trials=10, trial_gap=5, n_onset_bins=10,
)
plot_correlation_matrix(results)
plot_distance_comparison(results)
"""

import numpy as np
from scipy.stats import pearsonr, wilcoxon
from scipy.stats import zscore as sp_zscore
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────

def _zscore_session(norm):
    """Z-score each neuron's time series across the full session."""
    mu  = norm.mean(axis=1, keepdims=True)
    std = norm.std(axis=1, keepdims=True)
    std[std == 0] = 1.0          # avoid divide-by-zero for silent neurons
    return (norm - mu) / std


def _get_state_vectors(sess_data, n_onset_bins=10, bins_per_state=90,
                       neuron_subset=None):
    """
    Extract z-scored population vectors at each (trial, state) onset.

    Neurons_norm is (n_neurons, n_trials, bins_per_trial) where the trial
    is time-normalised: each of the 4 states occupies exactly bins_per_state
    bins (default 90), so bins_per_trial = 4 * 90 = 360.

    State onset positions within each trial are derived from this fixed
    structure (0, 90, 180, 270) rather than from Trial_times, which may
    contain extra columns (e.g. trial-end markers).

    The trials are concatenated along the time axis and z-scored across the
    full session before onset windows are extracted.

    Parameters
    ----------
    sess_data : dict
        Session data with 'Neurons_norm' (n_neurons × n_trials × bins_per_trial).
    n_onset_bins : int
        Bins to average from each state onset (default 10).
    bins_per_state : int
        Normalised bins per state epoch (default 90).
    neuron_subset : array-like of int, optional

    Returns
    -------
    vectors : (n_trials, n_states, n_neurons) float32
    valid   : (n_trials, n_states) bool
    """
    raw = np.asarray(sess_data['Neurons_norm'], dtype=float)

    if raw.ndim != 3:
        raise ValueError(f"Neurons_norm expected shape (n_neurons, n_trials, "
                         f"bins_per_trial), got {raw.shape}")

    n_neurons_raw, n_trials, bins_per_trial = raw.shape
    n_states = bins_per_trial // bins_per_state   # 360 // 90 = 4

    if neuron_subset is not None:
        raw = raw[np.asarray(neuron_subset)]
        n_neurons_raw = raw.shape[0]

    # Concatenate trials → (n_neurons, n_trials * bins_per_trial), then z-score
    raw_cat  = raw.reshape(n_neurons_raw, n_trials * bins_per_trial)
    norm     = _zscore_session(raw_cat)
    n_neurons = norm.shape[0]

    vectors = np.full((n_trials, n_states, n_neurons), np.nan, dtype=np.float32)
    valid   = np.zeros((n_trials, n_states), dtype=bool)

    for t in range(n_trials):
        for s in range(n_states):
            # fixed onset position in the concatenated array
            onset = t * bins_per_trial + s * bins_per_state
            end   = onset + n_onset_bins
            vectors[t, s, :] = norm[:, onset:end].mean(axis=1)
            valid[t, s]      = True

    return vectors, valid


def _row_pearsonr(A, B):
    """
    Vectorised row-wise Pearson r between matching rows of A and B.

    Parameters
    ----------
    A, B : (n, p) float

    Returns
    -------
    r : (n,) float  — nan where a row has zero variance
    """
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)
    num   = (A_c * B_c).sum(axis=1)
    denom = np.sqrt((A_c ** 2).sum(axis=1) * (B_c ** 2).sum(axis=1))
    with np.errstate(invalid='ignore'):
        return np.where(denom > 0, num / denom, np.nan)


def _session_correlation_matrix(vectors, valid, trial_gap=5):
    """
    Compute (n_states × n_states) mean Pearson r for a single session.

    Only trial pairs (i, j) with j - i >= trial_gap are used.

    Parameters
    ----------
    vectors  : (n_trials, n_states, n_neurons)
    valid    : (n_trials, n_states) bool
    trial_gap: int

    Returns
    -------
    corr_mat  : (n_states, n_states) float — mean Pearson r per state pair
    count_mat : (n_states, n_states) int   — number of pairs per cell
    """
    n_trials, n_states, _ = vectors.shape

    # pre-build valid trial-pair indices
    i_idx = np.array([i for i in range(n_trials)
                      for j in range(i + trial_gap, n_trials)])
    j_idx = np.array([j for i in range(n_trials)
                      for j in range(i + trial_gap, n_trials)])

    corr_mat  = np.full((n_states, n_states), np.nan)
    count_mat = np.zeros((n_states, n_states), dtype=int)

    for s in range(n_states):
        for t in range(n_states):
            mask = valid[i_idx, s] & valid[j_idx, t]
            if mask.sum() < 2:
                continue
            A  = vectors[i_idx[mask], s, :]
            B  = vectors[j_idx[mask], t, :]
            rs = _row_pearsonr(A, B)
            corr_mat[s, t]  = np.nanmean(rs)
            count_mat[s, t] = int(np.isfinite(rs).sum())

    return corr_mat, count_mat


# ─────────────────────────────────────────────────────────────────────────────

def _circ_dist(s, t, n):
    """Circular distance between state indices s and t on a ring of size n."""
    d = abs(s - t)
    return min(d, n - d)


def run_circular_correlation(data_dic, mouse_recday, valid_sessions,
                              min_trials=10, trial_gap=5, n_onset_bins=10,
                              bins_per_state=90, neuron_subset=None):
    """
    Compute state-pair correlation matrices and circular-distance summaries.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    min_trials : int
        Minimum trials for a session to be included (default 10).
    trial_gap : int
        Minimum trial separation for a pair (default 5).
    n_onset_bins : int
        Time bins averaged at each state onset (default 10).
    bins_per_state : int
        Normalised bins per state epoch in Neurons_norm (default 90,
        giving 4 states × 90 = 360 bins per trial).
    neuron_subset : array-like of int, optional

    Returns
    -------
    results : dict
        'mean_corr_matrix'  : (n_states, n_states) — grand mean across sessions
        'session_matrices'  : list of (n_states, n_states) per session
        'session_keys'      : list of session keys used
        'dist1_per_session' : (n_sessions,) — mean r for distance-1 pairs
        'dist2_per_session' : (n_sessions,) — mean r for distance-2 pairs
        'diag_per_session'  : (n_sessions,) — mean r for same-state pairs (d=0)
        'state_labels'      : list of state label strings
        'mouse_recday'      : str
    """
    print(f"\n{'='*60}")
    print(f"Circular state correlation: {mouse_recday}")
    print(f"min_trials={min_trials}, trial_gap={trial_gap}, "
          f"n_onset_bins={n_onset_bins}, bins_per_state={bins_per_state}")
    print(f"{'='*60}")

    session_matrices, session_keys = [], []

    for sidx in valid_sessions:
        sess_data          = data_dic[mouse_recday][sidx]
        trial_times        = np.asarray(sess_data['Trial_times'])
        n_trials, n_states = trial_times.shape

        if n_trials < min_trials:
            print(f"  Session {sidx}: {n_trials} trials — skipped (< {min_trials})")
            continue

        vectors, valid = _get_state_vectors(
            sess_data, n_onset_bins=n_onset_bins,
            bins_per_state=bins_per_state, neuron_subset=neuron_subset,
        )
        corr_mat, count_mat = _session_correlation_matrix(
            vectors, valid, trial_gap=trial_gap,
        )
        session_matrices.append(corr_mat)
        session_keys.append(sidx)

        print(f"  Session {sidx}: {n_trials} trials, "
              f"min pairs/cell = {count_mat.min()}")

    if not session_matrices:
        raise RuntimeError("No sessions passed the min_trials filter.")

    stacked   = np.stack(session_matrices)              # (n_sess, n_states, n_states)
    mean_corr = np.nanmean(stacked, axis=0)
    n_states  = mean_corr.shape[0]

    # ── per-session circular-distance summaries ───────────────────────────────
    dist0_per_session, dist1_per_session, dist2_per_session = [], [], []

    for mat in session_matrices:
        d0v, d1v, d2v = [], [], []
        for s in range(n_states):
            for t in range(n_states):
                v = mat[s, t]
                if not np.isfinite(v):
                    continue
                d = _circ_dist(s, t, n_states)
                if   d == 0: d0v.append(v)
                elif d == 1: d1v.append(v)
                elif d == 2: d2v.append(v)
        dist0_per_session.append(np.mean(d0v) if d0v else np.nan)
        dist1_per_session.append(np.mean(d1v) if d1v else np.nan)
        dist2_per_session.append(np.mean(d2v) if d2v else np.nan)

    dist0 = np.array(dist0_per_session)
    dist1 = np.array(dist1_per_session)
    dist2 = np.array(dist2_per_session)

    print(f"\n  {len(session_keys)} sessions included")
    print(f"  Same state  (d=0) mean r = {np.nanmean(dist0):.3f} ± {np.nanstd(dist0):.3f}")
    print(f"  Distance-1  (d=1) mean r = {np.nanmean(dist1):.3f} ± {np.nanstd(dist1):.3f}")
    print(f"  Distance-2  (d=2) mean r = {np.nanmean(dist2):.3f} ± {np.nanstd(dist2):.3f}")

    # Wilcoxon: distance-1 > distance-2
    mask = np.isfinite(dist1) & np.isfinite(dist2)
    if mask.sum() >= 3:
        stat, p = wilcoxon(dist1[mask], dist2[mask])
        print(f"  Wilcoxon d1 vs d2: W = {stat:.1f}, p = {p:.4f}")
    print(f"{'='*60}\n")

    state_letters = list('ABCDEFGH')
    state_labels  = state_letters[:n_states]

    return {
        'mean_corr_matrix':  mean_corr,
        'session_matrices':  session_matrices,
        'session_keys':      session_keys,
        'dist0_per_session': dist0,
        'dist1_per_session': dist1,
        'dist2_per_session': dist2,
        'state_labels':      state_labels,
        'mouse_recday':      mouse_recday,
    }


# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_matrix(results):
    """
    Heatmap of the grand-mean (n_states × n_states) state-pair correlation
    matrix, with circular-distance annotations on the off-diagonal cells.
    """
    mat          = results['mean_corr_matrix']
    state_labels = results['state_labels']
    mouse_recday = results['mouse_recday']
    n_sess       = len(results['session_keys'])
    n_states     = len(state_labels)

    vmax = np.nanmax(np.abs(mat))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean Pearson r')

    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(state_labels)
    ax.set_yticklabels(state_labels)
    ax.set_xlabel(f'Trial n+5 — state')
    ax.set_ylabel(f'Trial n   — state')

    for i in range(n_states):
        for j in range(n_states):
            v = mat[i, j]
            d = _circ_dist(i, j, n_states)
            if np.isfinite(v):
                label = f'{v:.2f}\n(d={d})'
                ax.text(j, i, label, ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > vmax * 0.6 else 'black')

    ax.set_title(
        f'State-pair correlations (mean, n={n_sess} sessions)\n{mouse_recday}',
        fontweight='bold',
    )
    plt.tight_layout()
    plt.show()
    return fig


def plot_distance_comparison(results):
    """
    Bar + individual-point plot comparing same-state, distance-1 and
    distance-2 correlations across sessions.
    """
    d0           = results['dist0_per_session']
    d1           = results['dist1_per_session']
    d2           = results['dist2_per_session']
    title_label  = results.get('mouse_recday', '')

    # Wilcoxon d1 vs d2
    mask = np.isfinite(d1) & np.isfinite(d2)
    if mask.sum() >= 3:
        _, p_wilcox = wilcoxon(d1[mask], d2[mask])
    else:
        p_wilcox = float('nan')

    all_vals  = [d0, d1, d2]
    labels    = ['Same state\n(d=0)', 'Adjacent\n(d=1)', 'Opposite\n(d=2)']
    colors    = ['#4C72B0', '#55A868', '#C44E52']
    positions = [0, 1, 2]

    fig, ax = plt.subplots(figsize=(5, 5))
    rng = np.random.default_rng(0)

    for pos, vals, col in zip(positions, all_vals, colors):
        v = vals[np.isfinite(vals)]
        if len(v) == 0:
            continue
        mean = v.mean()
        sem  = v.std() / np.sqrt(len(v))
        ax.bar(pos, mean, yerr=sem, width=0.5, color=col, alpha=0.7, capsize=5,
               error_kw={'lw': 1.5})
        jx = pos + rng.uniform(-0.08, 0.08, size=len(v))
        ax.scatter(jx, v, color=col, s=35, zorder=3, alpha=0.85,
                   edgecolors='white', linewidths=0.5)

    # paired lines d1 vs d2
    for v1, v2 in zip(d1[mask], d2[mask]):
        ax.plot([1, 2], [v1, v2], color='grey', lw=0.8, alpha=0.4)

    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Pearson r')
    sig_str = (f'p = {p_wilcox:.3f}' if np.isfinite(p_wilcox)
               else 'insufficient sessions')
    ax.set_title(
        f'Circular distance comparison — {title_label}\n'
        f'd1 vs d2, Wilcoxon: {sig_str}  (n={mask.sum()} sessions)',
        fontweight='bold',
    )
    plt.tight_layout()
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def run_population_drift(data_dic, mouse_recday, valid_sessions,
                         max_lag=10, min_trials=20,
                         n_onset_bins=10, bins_per_state=90,
                         neuron_subset=None):
    """
    Measure within-state population drift across trial intervals.

    For each state and each trial lag (1 … max_lag), computes the mean
    Pearson correlation between the state's population vector at trial t
    and trial t+lag.  A drop with increasing lag indicates representational
    drift; a flat profile indicates stability.

    Only sessions with >= min_trials trials are included.

    Parameters
    ----------
    data_dic, mouse_recday, valid_sessions
        As for run_circular_correlation.
    max_lag : int
        Maximum trial interval to test (default 10).
    min_trials : int
        Minimum trials per session (default 20).
    n_onset_bins, bins_per_state, neuron_subset
        As for run_circular_correlation.

    Returns
    -------
    results : dict
        'lag_corr_mean'     : (n_states, max_lag) — mean r across sessions
        'lag_corr_sem'      : (n_states, max_lag) — SEM across sessions
        'lag_corr_sessions' : (n_sessions, n_states, max_lag) — per-session values
        'lags'              : 1 … max_lag
        'state_labels'      : list of state label strings
        'session_keys'      : list of included session keys
        'mouse_recday'      : str
    """
    print(f"\n{'='*60}")
    print(f"Population drift: {mouse_recday}  "
          f"(max_lag={max_lag}, min_trials={min_trials})")
    print(f"{'='*60}")

    lags          = np.arange(1, max_lag + 1)
    session_drift = []   # (n_sessions, n_states, max_lag)
    session_keys  = []
    state_labels  = None

    for sidx in valid_sessions:
        sess_data   = data_dic[mouse_recday][sidx]
        trial_times = np.asarray(sess_data['Trial_times'])
        n_trials    = trial_times.shape[0]

        if n_trials < min_trials:
            print(f"  Session {sidx}: {n_trials} trials — skipped (< {min_trials})")
            continue

        vectors, valid = _get_state_vectors(
            sess_data, n_onset_bins=n_onset_bins,
            bins_per_state=bins_per_state, neuron_subset=neuron_subset,
        )
        n_trials_v, n_states, _ = vectors.shape
        state_labels = list('ABCDEFGH')[:n_states]

        drift = np.full((n_states, max_lag), np.nan)

        for s in range(n_states):
            for li, lag in enumerate(lags):
                # all trial pairs (t, t+lag) where both are valid
                i_idx = np.arange(n_trials_v - lag)
                j_idx = i_idx + lag
                mask  = valid[i_idx, s] & valid[j_idx, s]
                if mask.sum() < 2:
                    continue
                A  = vectors[i_idx[mask], s, :]
                B  = vectors[j_idx[mask], s, :]
                rs = _row_pearsonr(A, B)
                drift[s, li] = np.nanmean(rs)

        session_drift.append(drift)
        session_keys.append(sidx)
        print(f"  Session {sidx}: {n_trials} trials  "
              f"mean r at lag 1 = "
              f"{np.nanmean(drift[:, 0]):.3f}")

    if not session_drift:
        raise RuntimeError("No sessions passed the min_trials filter.")

    arr      = np.stack(session_drift)          # (n_sess, n_states, max_lag)
    lag_mean = np.nanmean(arr, axis=0)          # (n_states, max_lag)
    lag_sem  = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))

    print(f"\n  {len(session_keys)} sessions used")
    print(f"{'='*60}\n")

    return {
        'lag_corr_mean':     lag_mean,
        'lag_corr_sem':      lag_sem,
        'lag_corr_sessions': arr,
        'lags':              lags,
        'state_labels':      state_labels,
        'session_keys':      session_keys,
        'mouse_recday':      mouse_recday,
    }


def plot_population_drift(results):
    """
    Line plot of within-state correlation vs trial lag, one line per state.
    Shaded band = ±SEM across sessions.  Individual session points overlaid.

    Parameters
    ----------
    results : dict
        Output of run_population_drift (or aggregated equivalent with same keys).
    """
    mean_arr     = results['lag_corr_mean']      # (n_states, max_lag)
    sem_arr      = results['lag_corr_sem']
    sess_arr     = results['lag_corr_sessions']  # (n_sess, n_states, max_lag)
    lags         = results['lags']
    state_labels = results['state_labels']
    title        = results.get('mouse_recday', results.get('label', ''))
    n_states     = len(state_labels)
    n_sess       = sess_arr.shape[0]

    colors = plt.cm.tab10(np.linspace(0, 0.6, n_states))

    fig, ax = plt.subplots(figsize=(7, 4))

    for s, (lbl, col) in enumerate(zip(state_labels, colors)):
        mu  = mean_arr[s]
        sem = sem_arr[s]
        # SEM band
        ax.fill_between(lags, mu - sem, mu + sem, alpha=0.2, color=col)
        # mean line
        ax.plot(lags, mu, color=col, lw=2, label=lbl)
        # individual session dots (jittered slightly)
        rng = np.random.default_rng(s)
        for sess_i in range(n_sess):
            vals = sess_arr[sess_i, s, :]
            jx   = lags + rng.uniform(-0.15, 0.15, size=len(lags))
            ax.scatter(jx, vals, color=col, s=8, alpha=0.3, zorder=2)

    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Trial lag')
    ax.set_ylabel('Mean Pearson r (same state)')
    ax.set_xticks(lags)
    ax.legend(title='State', bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=9)
    ax.set_title(f'Population drift — {title}\n'
                 f'(n={n_sess} sessions)',
                 fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig
