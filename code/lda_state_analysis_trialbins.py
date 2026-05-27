"""
LDA state separation using trial-averaged population vectors.

Instead of fine time bins, one feature vector per (session, trial, state)
is built by averaging z-scored firing rates across the raw time window of
each state within each trial.

Normalization: z-score per neuron across the fully concatenated (all sessions)
               raw recording, then extract per-(trial, state) averages.

Main entry points
-----------------
- `run_trialbins_lda_analysis`         : full LDA pipeline + LOGO CV
- `run_lda_eigenvalue_separation`      : Pillai-trace separability with
                                         roll-shuffle null
- `compute_pseudotime_state_correlations` : population-vector correlations
                                         across sessions, matched by trial
                                         number (pseudotime drift control)
- `build_trialbins_gp_dataset`         : like build_trialbins_dataset, but
                                         splits each state into goal-progress
                                         sub-windows (early/middle/late)
- `run_lda_eigenvalue_separation_loto` : T1 — leave-one-task-out Pillai trace
- `run_within_task_pillai`             : T2 — per-session within-task Pillai
- `run_within_session_shuffle_null`    : T3 — Pillai null built from within-
                                         session state-label shuffles

Reuses run_lda_state_separation and run_logo_state_cv from lda_state_separation.py.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lda_state_separation import run_lda_state_separation, run_logo_state_cv
import matplotlib.pyplot as plt

# ── Configuration (keep consistent with notebook cell 130) ────────────────────
SAMPLING_RATE       = 2400        # samples per minute (= 40 Hz)
EVENT_LABELS        = ['A', 'B', 'C', 'D']
EVENT_COLOURS       = {'A': '#e41a1c', 'B': '#377eb8',
                       'C': '#4daf4a', 'D': '#ff7f00'}
PCA_VARIANCE_THRESH = 0.75         # cumulative explained-variance threshold
MIN_STATE_SAMPLES   = 2           # minimum raw samples a state window must span


# ─────────────────────────────────────────────────────────────────────────────
def zscore_concat_sessions(raw_sessions):
    """
    Concatenate raw recordings and z-score per neuron across the joint timeline.

    Parameters
    ----------
    raw_sessions : list of np.ndarray
        Each element shape (n_neurons, T_i).  All must share the same n_neurons.

    Returns
    -------
    concat_z : np.ndarray
        Shape (n_neurons, sum_T).  Z-scored per neuron.
    offsets : list of int
        offsets[i] = sample index in concat_z where session i starts.
    lengths : list of int
        lengths[i] = T_i.
    zero_var_mask : np.ndarray of bool
        Shape (n_neurons,).  True where neuron std == 0 (set to 0.0 in output).
    """
    n_neurons = raw_sessions[0].shape[0]
    for i, rs in enumerate(raw_sessions):
        if rs.shape[0] != n_neurons:
            raise ValueError(
                f"Session {i} has {rs.shape[0]} neurons; expected {n_neurons}."
            )

    concat_raw = np.concatenate(raw_sessions, axis=1)   # (n_neurons, sum_T)

    mu  = concat_raw.mean(axis=1, keepdims=True)
    sig = concat_raw.std(axis=1, keepdims=True)

    zero_var_mask = (sig.squeeze() == 0)
    sig_safe = np.where(sig == 0, 1.0, sig)
    concat_z = (concat_raw - mu) / sig_safe
    concat_z[zero_var_mask, :] = 0.0

    offsets, lengths = [], []
    offset = 0
    for rs in raw_sessions:
        offsets.append(offset)
        lengths.append(rs.shape[1])
        offset += rs.shape[1]

    return concat_z, offsets, lengths, zero_var_mask


# ─────────────────────────────────────────────────────────────────────────────
def extract_state_trial_vectors(concat_z, session_offset, session_length,
                                trial_times, sess_index,
                                min_samples=MIN_STATE_SAMPLES,
                                aggregator='mean'):
    """
    Extract one aggregated population vector per (trial, state) from a z-scored
    concatenated recording.

    Parameters
    ----------
    concat_z : np.ndarray
        Shape (n_neurons, total_T).  Full z-scored concatenated array.
    session_offset : int
        Sample index in concat_z where this session starts.
    session_length : int
        Number of samples in this session.
    trial_times : np.ndarray
        Shape (n_trials, 5).  Session-local sample indices.
        Columns: [A_start, B_start, C_start, D_start, trial_end].
        State X spans [col_X, col_{X+1}).
    sess_index : int
        Integer session index written into sess_id output.
    min_samples : int
        Minimum window length; shorter windows are skipped.
    aggregator : {'mean', 'max'}, default 'mean'
        How to aggregate z-scored firing across the state window:
        'mean' = average firing across the window (sustained drive)
        'max'  = peak firing across the window (punctate response)

    Returns
    -------
    X_sess : np.ndarray
        Shape (n_valid, n_neurons).
    y_state_sess : np.ndarray of str
        Shape (n_valid,).  Values in EVENT_LABELS.
    sess_id_sess : np.ndarray of int
    trial_id_sess : np.ndarray of int
        Trial index within this session.
    skipped : list of tuple
        (trial_idx, state_label, reason) for every skipped window.
    """
    n_neurons = concat_z.shape[0]
    n_trials  = trial_times.shape[0]

    X_list, y_list, sid_list, tid_list = [], [], [], []
    skipped = []

    for trial_idx in range(n_trials):
        for state_i, label in enumerate(EVENT_LABELS):
            t_start_local = int(trial_times[trial_idx, state_i])
            t_end_local   = int(trial_times[trial_idx, state_i + 1])

            # Guard: within session bounds
            if t_start_local < 0 or t_end_local > session_length:
                skipped.append((trial_idx, label,
                    f"out of bounds [{t_start_local}, {t_end_local}) "
                    f"for session length {session_length}"))
                continue

            # Guard: positive duration
            if t_end_local <= t_start_local:
                skipped.append((trial_idx, label,
                    f"zero/negative duration: "
                    f"start={t_start_local}, end={t_end_local}"))
                continue

            # Guard: minimum length
            n_samp = t_end_local - t_start_local
            if n_samp < min_samples:
                skipped.append((trial_idx, label,
                    f"too short ({n_samp} samples < {min_samples})"))
                continue

            t_start_g = session_offset + t_start_local
            t_end_g   = session_offset + t_end_local
            window = concat_z[:, t_start_g:t_end_g]
            if aggregator == 'mean':
                vec = window.mean(axis=1)
            elif aggregator == 'max':
                vec = window.max(axis=1)
            else:
                raise ValueError(f"aggregator must be 'mean' or 'max', got {aggregator!r}")

            # Guard: NaN propagation
            if np.any(np.isnan(vec)):
                skipped.append((trial_idx, label, "NaN in averaged vector"))
                continue

            X_list.append(vec)
            y_list.append(label)
            sid_list.append(sess_index)
            tid_list.append(trial_idx)

    if len(X_list) == 0:
        return (np.empty((0, n_neurons)),
                np.empty(0, dtype='<U1'),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                skipped)

    return (np.vstack(X_list),
            np.array(y_list, dtype='<U1'),
            np.array(sid_list, dtype=int),
            np.array(tid_list, dtype=int),
            skipped)


# ─────────────────────────────────────────────────────────────────────────────
def extract_state_gp_trial_vectors(concat_z, session_offset, session_length,
                                    trial_times, sess_index,
                                    num_gp_bins=3,
                                    min_samples=MIN_STATE_SAMPLES,
                                    aggregator='mean'):
    """
    Like extract_state_trial_vectors but splits each state window into
    `num_gp_bins` equal-length sub-windows (goal-progress bins) and returns
    one aggregated population vector per (trial, state, gp_bin).

    Parameters mirror extract_state_trial_vectors plus `num_gp_bins`.
    `aggregator='mean'|'max'` controls the within-sub-window aggregation.

    Returns
    -------
    X_sess        : (n_valid, n_neurons)
    y_state_sess  : (n_valid,) state labels in EVENT_LABELS
    y_gp_sess     : (n_valid,) ints in [0, num_gp_bins-1]
                    0 = early, num_gp_bins-1 = late within the state
    sess_id_sess  : (n_valid,) ints
    trial_id_sess : (n_valid,) ints (trial index within session)
    skipped       : list of (trial_idx, state_label, gp_bin, reason)
    """
    n_neurons = concat_z.shape[0]
    n_trials = trial_times.shape[0]

    X_list, y_state_list, y_gp_list, sid_list, tid_list = [], [], [], [], []
    skipped = []

    for trial_idx in range(n_trials):
        for state_i, label in enumerate(EVENT_LABELS):
            t_start_local = int(trial_times[trial_idx, state_i])
            t_end_local = int(trial_times[trial_idx, state_i + 1])

            if t_start_local < 0 or t_end_local > session_length:
                skipped.append((trial_idx, label, None,
                    f"out of bounds [{t_start_local}, {t_end_local}) for length {session_length}"))
                continue

            if t_end_local <= t_start_local:
                skipped.append((trial_idx, label, None,
                    f"zero/negative duration: start={t_start_local}, end={t_end_local}"))
                continue

            state_duration = t_end_local - t_start_local

            for gp in range(num_gp_bins):
                sub_start = t_start_local + (state_duration * gp) // num_gp_bins
                sub_end = t_start_local + (state_duration * (gp + 1)) // num_gp_bins

                n_samp = sub_end - sub_start
                if n_samp < min_samples:
                    skipped.append((trial_idx, label, gp,
                        f"gp sub-window too short ({n_samp} < {min_samples})"))
                    continue

                t_start_g = session_offset + sub_start
                t_end_g = session_offset + sub_end
                window = concat_z[:, t_start_g:t_end_g]
                if aggregator == 'mean':
                    vec = window.mean(axis=1)
                elif aggregator == 'max':
                    vec = window.max(axis=1)
                else:
                    raise ValueError(f"aggregator must be 'mean' or 'max', got {aggregator!r}")

                if np.any(np.isnan(vec)):
                    skipped.append((trial_idx, label, gp, "NaN in averaged vector"))
                    continue

                X_list.append(vec)
                y_state_list.append(label)
                y_gp_list.append(gp)
                sid_list.append(sess_index)
                tid_list.append(trial_idx)

    if len(X_list) == 0:
        return (np.empty((0, n_neurons)),
                np.empty(0, dtype='<U1'),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                skipped)

    return (np.vstack(X_list),
            np.array(y_state_list, dtype='<U1'),
            np.array(y_gp_list, dtype=int),
            np.array(sid_list, dtype=int),
            np.array(tid_list, dtype=int),
            skipped)


# ─────────────────────────────────────────────────────────────────────────────
def build_trialbins_dataset(data_dic, mouse_recday, valid_sessions,
                             neuron_subset=None,
                             balance_trials_per_task=False,
                             balance_rng_seed=42,
                             aggregator='mean'):
    """
    Build the trial-averaged z-scored dataset for one mouse_recday.

    Normalization is performed once on the concatenated raw recordings
    (z-score per neuron), then per-(trial, state) averages are extracted.

    Parameters
    ----------
    data_dic : dict
        data_dic[mouse_recday][session_idx] must contain:
          - 'Neuron_raw'  : np.ndarray (n_neurons_total, T_i)
          - 'Trial_times' : np.ndarray (n_trials, 5)
    mouse_recday : str
    valid_sessions : list
        Ordered session keys.  Determines integer sess_id values (0, 1, …).
    neuron_subset : array-like of int, optional
        Row indices of Neuron_raw to retain (e.g. significant LEC neurons).
    balance_trials_per_task : bool, default False
        If True, subsample each session's trials to min(n_trials) across sessions
        before extracting (trial, state) vectors. Z-scoring still uses the full
        raw recording (more samples for normalization stats); only the trial
        extraction is balanced. Prevents tasks with more trials from dominating
        the within-state scatter.
    balance_rng_seed : int, default 42
        Seed for the trial-subsampling RNG.

    Returns
    -------
    X : np.ndarray
        Shape (n_samples, n_neurons).
    y_state : np.ndarray of str
        Shape (n_samples,).  Values in ['A', 'B', 'C', 'D'].
    sess_id : np.ndarray of int
        Shape (n_samples,).
    trial_id : np.ndarray of int
        Shape (n_samples,).  Trial index within its session.
    zero_var_mask : np.ndarray of bool
        Shape (n_neurons,).  True = zero-variance neuron across concat recording.
    n_skipped : int
        Number of (trial, state) windows that were skipped.
    """
    raw_sessions, trial_times_list = [], []

    for sidx in valid_sessions:
        raw = data_dic[mouse_recday][sidx]['Neuron_raw']
        print(f"Task is {data_dic[mouse_recday][sidx]['Task']}")
        if neuron_subset is not None:
            raw = raw[np.asarray(neuron_subset), :]
        raw_sessions.append(raw)
        trial_times_list.append(data_dic[mouse_recday][sidx]['Trial_times'])

    if not raw_sessions:
        raise ValueError(f"No valid sessions found for {mouse_recday}.")

    n_neurons = raw_sessions[0].shape[0]
    print(f"{mouse_recday}: {len(valid_sessions)} sessions, {n_neurons} neurons")

    # Optionally subsample each session's trials down to the min across sessions.
    # Done BEFORE z-scoring's downstream use (z-score still uses full raw recording),
    # but trial extraction below iterates only over the subsampled trial_times.
    if balance_trials_per_task:
        n_trials_per_session = [tt.shape[0] for tt in trial_times_list]
        min_n_trials = min(n_trials_per_session)
        print(f"  Balancing trials per task: keeping {min_n_trials} per session "
              f"(originally {n_trials_per_session})")
        rng_bal = np.random.default_rng(balance_rng_seed)
        trial_times_list = [
            tt[np.sort(rng_bal.choice(tt.shape[0], min_n_trials, replace=False))]
            if tt.shape[0] > min_n_trials else tt
            for tt in trial_times_list
        ]

    # Z-score across all sessions concatenated
    concat_z, offsets, lengths, zero_var_mask = zscore_concat_sessions(raw_sessions)

    n_zero = int(zero_var_mask.sum())
    if n_zero:
        print(f"  WARNING: {n_zero} neuron(s) with zero variance set to 0.")

    X_list, y_list, sid_list, tid_list = [], [], [], []
    total_skipped = 0

    for sess_i, (sidx, offset, length, tt) in enumerate(
            zip(valid_sessions, offsets, lengths, trial_times_list)):

        print(f"  Session {sidx} (index {sess_i}): "
              f"{tt.shape[0]} trials, T={length} samples")

        X_s, y_s, sid_s, tid_s, skipped = extract_state_trial_vectors(
            concat_z, offset, length, tt, sess_i,
            aggregator=aggregator,
        )

        for trial_idx, label, reason in skipped:
            print(f"    SKIP trial={trial_idx} state={label}: {reason}")
        total_skipped += len(skipped)

        if X_s.shape[0] > 0:
            X_list.append(X_s)
            y_list.append(y_s)
            sid_list.append(sid_s)
            tid_list.append(tid_s)

    if not X_list:
        raise RuntimeError(
            f"No valid (trial, state) samples extracted for {mouse_recday}."
        )

    X        = np.vstack(X_list)
    y_state  = np.concatenate(y_list)
    sess_id  = np.concatenate(sid_list)
    print(np.unique(sess_id, return_counts=True))
    trial_id = np.concatenate(tid_list)

    print(f"  Dataset: {X.shape[0]} samples "
          f"({total_skipped} skipped) from "
          f"{X.shape[0] + total_skipped} (trial, state) pairs")

    return X, y_state, sess_id, trial_id, zero_var_mask, total_skipped


# ─────────────────────────────────────────────────────────────────────────────
def build_trialbins_gp_dataset(data_dic, mouse_recday, valid_sessions,
                                neuron_subset=None,
                                num_gp_bins=3,
                                balance_trials_per_task=False,
                                balance_rng_seed=42,
                                aggregator='mean'):
    """
    Like build_trialbins_dataset but each state window is split into
    `num_gp_bins` equal-length goal-progress sub-windows, yielding one
    population vector per (trial, state, gp_bin).

    Parameters
    ----------
    data_dic, mouse_recday, valid_sessions, neuron_subset,
    balance_trials_per_task, balance_rng_seed : see `build_trialbins_dataset`.
    num_gp_bins : int, default 3
        Number of equal-length sub-windows per state window
        (3 ⇒ early / middle / late).

    Returns
    -------
    X             : (n_samples, n_neurons)
    y_state       : (n_samples,) state labels in EVENT_LABELS
    y_gp          : (n_samples,) ints in [0, num_gp_bins-1]
    sess_id       : (n_samples,) ints
    trial_id      : (n_samples,) ints (trial index within its session)
    zero_var_mask : (n_neurons,) bool
    """
    raw_sessions, trial_times_list = [], []

    for sidx in valid_sessions:
        raw = data_dic[mouse_recday][sidx]['Neuron_raw']
        if neuron_subset is not None:
            raw = raw[np.asarray(neuron_subset), :]
        raw_sessions.append(raw)
        trial_times_list.append(data_dic[mouse_recday][sidx]['Trial_times'])

    if not raw_sessions:
        raise ValueError(f"No valid sessions found for {mouse_recday}.")

    if balance_trials_per_task:
        n_trials_per_session = [tt.shape[0] for tt in trial_times_list]
        min_n_trials = min(n_trials_per_session)
        print(f"  Balancing trials per task: keeping {min_n_trials} per session "
              f"(originally {n_trials_per_session})")
        rng_bal = np.random.default_rng(balance_rng_seed)
        trial_times_list = [
            tt[np.sort(rng_bal.choice(tt.shape[0], min_n_trials, replace=False))]
            if tt.shape[0] > min_n_trials else tt
            for tt in trial_times_list
        ]

    concat_z, offsets, lengths, zero_var_mask = zscore_concat_sessions(raw_sessions)

    X_list, y_state_list, y_gp_list, sid_list, tid_list = [], [], [], [], []
    total_skipped = 0

    for sess_i, (sidx, offset, length, tt) in enumerate(
            zip(valid_sessions, offsets, lengths, trial_times_list)):
        X_s, y_s, y_gp_s, sid_s, tid_s, skipped = extract_state_gp_trial_vectors(
            concat_z, offset, length, tt, sess_i, num_gp_bins=num_gp_bins,
            aggregator=aggregator,
        )
        total_skipped += len(skipped)
        if X_s.shape[0] > 0:
            X_list.append(X_s)
            y_state_list.append(y_s)
            y_gp_list.append(y_gp_s)
            sid_list.append(sid_s)
            tid_list.append(tid_s)

    if not X_list:
        raise RuntimeError(
            f"No valid (trial, state, gp) samples extracted for {mouse_recday}."
        )

    X = np.vstack(X_list)
    y_state = np.concatenate(y_state_list)
    y_gp = np.concatenate(y_gp_list)
    sess_id = np.concatenate(sid_list)
    trial_id = np.concatenate(tid_list)

    print(f"  GP dataset: {X.shape[0]} samples "
          f"({total_skipped} skipped); {num_gp_bins} gp bins per state")

    return X, y_state, y_gp, sess_id, trial_id, zero_var_mask


# ─────────────────────────────────────────────────────────────────────────────
def apply_pca_trialbins(X, variance_thresh=PCA_VARIANCE_THRESH):
    """
    Fit PCA on trial-binned feature matrix; retain components up to
    variance_thresh cumulative explained variance.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_neurons).
    variance_thresh : float

    Returns
    -------
    X_pca : np.ndarray
        Shape (n_samples, n_pcs).
    pca : sklearn.decomposition.PCA
        Fitted PCA object.
    n_pcs : int
    """
    n_comp_max = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp_max)
    X_pca_full = pca.fit_transform(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pcs  = int(np.searchsorted(cumvar, variance_thresh)) + 1
    n_pcs  = min(n_pcs, n_comp_max)

    print(f"  PCA: keeping {n_pcs}/{n_comp_max} PCs "
          f"({cumvar[n_pcs - 1]:.1%} cumulative variance)")

    return X_pca_full[:, :n_pcs], pca, n_pcs

def plot_ld_projections(X_state_ld, y_state, EVENT_LABELS, EVENT_COLOURS,
                        mouse_recday, max_lds=5, n_shuffles=1000, seed=42):
    """
    For each LD, plot all trial-bin samples (jittered by state row) with
    real centroids (mean ± SEM, black) and null centroids from circular
    label rolls (grey, mean ± SE across rolls).

    Parameters
    ----------
    X_state_ld : np.ndarray
        LDA-projected data, shape (n_samples, n_lds).
    y_state : np.ndarray of str
        State labels for each sample.
    EVENT_LABELS : list
        Ordered state labels, e.g. ['A', 'B', 'C', 'D'].
    EVENT_COLOURS : dict
        Colour per state.
    mouse_recday : str
        Title identifier.
    max_lds : int
        Maximum number of LDs to plot.
    n_shuffles : int
        Number of circular rolls for the null distribution.
    seed : int
        RNG seed for jitter and roll offsets.
    """
    from matplotlib.lines import Line2D

    n_ld = min(X_state_ld.shape[1], max_lds)
    n_samples = len(y_state)
    rng = np.random.default_rng(seed)

    # ── Precompute null centroids: (n_shuffles, n_ld, n_states) ───────
    roll_offsets = rng.integers(1, n_samples, size=n_shuffles)
    null_centroids = np.zeros((n_shuffles, n_ld, len(EVENT_LABELS)))

    print(f"Computing null centroids with {n_shuffles} circular label rolls...")
    for si, offset in enumerate(roll_offsets):
        y_rolled = np.roll(y_state, offset)
        for li, label in enumerate(EVENT_LABELS):
            mask = y_rolled == label
            if mask.sum() > 0:
                null_centroids[si, :, li] = X_state_ld[mask].mean(axis=0)[:n_ld]

    # Mean and SE of null centroids across rolls
    null_mean = null_centroids.mean(axis=0)  # (n_ld, n_states)
    null_se   = null_centroids.std(axis=0) / np.sqrt(n_shuffles)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_ld, 1, figsize=(14, 2.5 * n_ld))
    if n_ld == 1:
        axes = [axes]

    fig.suptitle(
        f'Trial bins projected onto each LD (coloured by Task State) — {mouse_recday}',
        fontsize=16, fontweight='bold', y=1.02
    )

    centroid_y = {lab: i for i, lab in enumerate(EVENT_LABELS)}
    jitter = rng.uniform(-0.5, 0.5, size=n_samples)

    for ld_i, ax in enumerate(axes):
        proj = X_state_ld[:, ld_i]

        # ── Scatter: one row per state ────────────────────────────────
        for label in EVENT_LABELS:
            mask = y_state == label
            if mask.any():
                ybase = centroid_y[label]
                ax.scatter(proj[mask], ybase + jitter[mask] * 0.3,
                           c=EVENT_COLOURS[label], s=14, alpha=0.4,
                           zorder=2)

        # ── Null centroids (grey, mean ± SE across rolls) ────────────
        for li, label in enumerate(EVENT_LABELS):
            cx = null_mean[ld_i, li]
            se = null_se[ld_i, li]
            cy = centroid_y[label]
            ax.errorbar(cx, cy + 0.15, xerr=se, fmt='none',
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        ecolor='0.5', zorder=8)
            ax.vlines(cx, cy, cy + 0.3,
                      colors='0.5', linewidth=1.5, zorder=8)

        # ── Real centroids (black, mean ± SEM) ───────────────────────
        cx_list = []
        for label in EVENT_LABELS:
            mask = y_state == label
            if mask.sum() > 0:
                cx = proj[mask].mean()
                sem = proj[mask].std() / np.sqrt(mask.sum())
                cy = centroid_y[label]
                cx_list.append((cx, cy))
                ax.errorbar(cx, cy - 0.15, xerr=sem, fmt='none',
                            capsize=5, capthick=2, elinewidth=2,
                            ecolor='black', zorder=10)
                ax.vlines(cx, cy - 0.4, cy + 0.1,
                          colors='black', linewidth=2, zorder=10)

        # Connect real centroids
        if len(cx_list) > 1:
            xs = [c[0] for c in cx_list]
            ys = [c[1] - 0.15 for c in cx_list]
            ax.plot(xs, ys, 'k-', lw=2, zorder=9, alpha=0.7)

        # ── Axes ──────────────────────────────────────────────────────
        ax.set_xlabel(f'LD {ld_i + 1}')
        ax.set_yticks(list(centroid_y.values()))
        ax.set_yticklabels(EVENT_LABELS, fontweight='bold')
        for v in centroid_y.values():
            ax.axhline(v, color='k', lw=0.3, alpha=0.15)

        if ld_i == 0:
            custom_handles = [
                Line2D([0], [0], color='black', lw=2, label='Real (mean ± SEM)'),
                Line2D([0], [0], color='0.5', lw=1.5,
                       label=f'Null — {n_shuffles} rolls (mean ± SE)'),
            ]
            leg = ax.legend(handles=custom_handles, fontsize=11,
                            loc='upper right', frameon=True,
                            framealpha=0.9, edgecolor='black')
            for text in leg.get_texts():
                text.set_fontweight('bold')

    plt.tight_layout()
    plt.show()

# def plot_ld_projections(X_state_ld, y_state, EVENT_LABELS, EVENT_COLOURS,
#                         mouse_recday, max_lds=5, seed=42):
#     """
#     For each LD, plot all trial-bin samples (jittered by state row) with
#     real centroids (mean ± SEM, black) and label-shuffled centroids (grey).

#     Parameters
#     ----------
#     X_state_ld : np.ndarray
#         LDA-projected data, shape (n_samples, n_lds).
#     y_state : np.ndarray of str
#         State labels for each sample.
#     EVENT_LABELS : list
#         Ordered state labels, e.g. ['A', 'B', 'C', 'D'].
#     EVENT_COLOURS : dict
#         Colour per state.
#     mouse_recday : str
#         Title identifier.
#     max_lds : int
#         Maximum number of LDs to plot.
#     seed : int
#         RNG seed for jitter and shuffle.
#     """
#     from matplotlib.lines import Line2D

#     n_ld = min(X_state_ld.shape[1], max_lds)
#     fig, axes = plt.subplots(n_ld, 1, figsize=(14, 2.5 * n_ld))
#     if n_ld == 1:
#         axes = [axes]

#     fig.suptitle(
#         f'Trial bins projected onto each LD (coloured by Task State) — {mouse_recday}',
#         fontsize=16, fontweight='bold', y=1.02
#     )

#     centroid_y = {lab: i for i, lab in enumerate(EVENT_LABELS)}

#     rng = np.random.default_rng(seed)
#     y_shuffled = rng.permutation(y_state)
#     jitter = rng.uniform(-0.5, 0.5, size=len(y_state))

#     for ld_i, ax in enumerate(axes):
#         proj = X_state_ld[:, ld_i]

#         # ── Scatter: one row per state ────────────────────────────────
#         for label in EVENT_LABELS:
#             mask = y_state == label
#             if mask.any():
#                 ybase = centroid_y[label]
#                 ax.scatter(proj[mask], ybase + jitter[mask] * 0.3,
#                            c=EVENT_COLOURS[label], s=14, alpha=0.4,
#                            zorder=2)

#         # ── Shuffled centroids (grey) ─────────────────────────────────
#         for label in EVENT_LABELS:
#             mask_shuf = y_shuffled == label
#             if mask_shuf.sum() > 0:
#                 cx = proj[mask_shuf].mean()
#                 sem = proj[mask_shuf].std() / np.sqrt(mask_shuf.sum())
#                 cy = centroid_y[label]
#                 ax.errorbar(cx, cy + 0.15, xerr=sem, fmt='none',
#                             capsize=4, capthick=1.5, elinewidth=1.5,
#                             ecolor='0.5', zorder=8)
#                 ax.vlines(cx, cy, cy + 0.3,
#                           colors='0.5', linewidth=1.5, zorder=8)

#         # ── Real centroids (black, mean ± SEM) ───────────────────────
#         cx_list = []
#         for label in EVENT_LABELS:
#             mask = y_state == label
#             if mask.sum() > 0:
#                 cx = proj[mask].mean()
#                 sem = proj[mask].std() / np.sqrt(mask.sum())
#                 cy = centroid_y[label]
#                 cx_list.append((cx, cy))
#                 ax.errorbar(cx, cy - 0.15, xerr=sem, fmt='none',
#                             capsize=5, capthick=2, elinewidth=2,
#                             ecolor='black', zorder=10)
#                 ax.vlines(cx, cy - 0.4, cy + 0.1,
#                           colors='black', linewidth=2, zorder=10)

#         # Connect real centroids
#         if len(cx_list) > 1:
#             xs = [c[0] for c in cx_list]
#             ys = [c[1] - 0.15 for c in cx_list]
#             ax.plot(xs, ys, 'k-', lw=2, zorder=9, alpha=0.7)

#         # ── Axes ──────────────────────────────────────────────────────
#         ax.set_xlabel(f'LD {ld_i + 1}')
#         ax.set_yticks(list(centroid_y.values()))
#         ax.set_yticklabels(EVENT_LABELS, fontweight='bold')
#         for v in centroid_y.values():
#             ax.axhline(v, color='k', lw=0.3, alpha=0.15)

#         if ld_i == 0:
#             custom_handles = [
#                 Line2D([0], [0], color='black', lw=2, label='Real'),
#                 Line2D([0], [0], color='0.5', lw=1.5, label='Shuffled'),
#             ]
#             leg = ax.legend(handles=custom_handles, fontsize=12,
#                             loc='upper right', frameon=True,
#                             framealpha=0.9, edgecolor='black')
#             for text in leg.get_texts():
#                 text.set_fontweight('bold')

#     plt.tight_layout()
#     plt.show()

# ─────────────────────────────────────────────────────────────────────────────
def run_trialbins_lda_analysis(data_dic, mouse_recday, valid_sessions,
                                neuron_subset=None,
                                variance_thresh=PCA_VARIANCE_THRESH,
                                n_logo_shuffles=1000,
                                balance_trials_per_task=False,
                                balance_rng_seed=42):
    """
    Full pipeline: build trial-binned dataset → PCA → LDA state separation
    → leave-one-session-out cross-validation.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional
    variance_thresh : float
    n_logo_shuffles : int
    balance_trials_per_task : bool, default False
        Subsample trials per session to min(n_trials) before extraction.
    balance_rng_seed : int, default 42

    Returns
    -------
    results : dict
        'X'            : (n_samples, n_neurons) z-scored trial-state matrix
        'X_pca'        : (n_samples, n_pcs) PCA-reduced
        'X_state_ld'   : (n_samples, n_lds) LDA-projected
        'y_state'      : (n_samples,) state labels
        'sess_id'      : (n_samples,) session indices
        'trial_id'     : (n_samples,) trial indices within session
        'pca'          : fitted PCA object
        'lda_state'    : fitted LDA object
        'zero_var_mask': (n_neurons,) bool array
        'real_acc'     : LOGO CV balanced accuracy
        'null_accs'    : LOGO CV null distribution
        'p_value'      : permutation p-value
    """
    print(f"\n{'='*70}")
    print(f"Trial-bin LDA analysis: {mouse_recday}")
    print(f"Sessions: {valid_sessions}")
    print(f"{'='*70}")

    # 1. Build z-scored trial-averaged dataset
    X, y_state, sess_id, trial_id, zero_var_mask, _ = build_trialbins_dataset(
        data_dic, mouse_recday, valid_sessions,
        neuron_subset=neuron_subset,
        balance_trials_per_task=balance_trials_per_task,
        balance_rng_seed=balance_rng_seed,
    )

    # 2. Verify all four states are present
    present = set(y_state)
    missing = set(EVENT_LABELS) - present
    if missing:
        raise RuntimeError(
            f"States {sorted(missing)} absent from dataset — "
            f"cannot fit 4-class LDA."
        )

    # 3. Guard: LOGO CV requires ≥ 2 sessions
    n_sessions = len(np.unique(sess_id))
    if n_sessions < 2:
        print(f"  WARNING: only {n_sessions} session — "
              f"LOGO CV will be skipped.")

    # 4. PCA
    X_pca, pca, _ = apply_pca_trialbins(X, variance_thresh)

    # 5. LDA state separation + centroid plot
    # run_lda_state_separation filters last_event != ''; since every y_state
    # value is in EVENT_LABELS, all rows are retained.
    lda_state, X_state_ld, y_state_out = run_lda_state_separation(
        X_pca, y_state, EVENT_LABELS, EVENT_COLOURS, mouse_recday
    )

    # 6. Leave-one-session-out CV (skip if < 2 sessions)
    if n_sessions >= 2:
        real_acc, null_accs, p_value = run_logo_state_cv(
            X_pca, y_state_out, sess_id,
            EVENT_LABELS, mouse_recday, n_shuffles=n_logo_shuffles
        )
    else:
        real_acc  = float('nan')
        null_accs = np.array([])
        p_value   = float('nan')

    return {
        'X'            : X,
        'X_pca'        : X_pca,
        'X_state_ld'   : X_state_ld,
        'y_state'      : y_state_out,
        'sess_id'      : sess_id,
        'trial_id'     : trial_id,
        'pca'          : pca,
        'lda_state'    : lda_state,
        'zero_var_mask': zero_var_mask,
        'real_acc'     : real_acc,
        'null_accs'    : null_accs,
        'p_value'      : p_value,
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_lda_eigenvalue_separation(data_dic, mouse_recday, valid_sessions,
                                   neuron_subset=None,
                                   variance_thresh=PCA_VARIANCE_THRESH,
                                   n_shuffles=100,
                                   rng_seed=42,
                                   balance_trials_per_task=False,
                                   balance_rng_seed=42):
    """
    Fit a global LDA on trial-averaged state vectors, extract W⁻¹B eigenvalues
    (Pillai's trace), and compare to a null distribution built from circular
    roll shuffles of the raw continuous recordings.

    Roll shuffles are applied per session to the raw data *before* z-scoring,
    trial parcellation, and PCA — preserving autocorrelation structure while
    destroying state-activity alignment.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional
    variance_thresh : float
    n_shuffles : int, default 100
    rng_seed : int, default 42
        Seed for the roll-shuffle RNG.
    balance_trials_per_task : bool, default False
        If True, subsample trials per session to min(n_trials) across sessions
        before extracting (trial, state) vectors. Applied identically to real
        and shuffled data so null and observed distributions use matched trial
        counts.
    balance_rng_seed : int, default 42
        Seed for the trial-subsampling RNG.

    Returns
    -------
    dict with keys:
        'eigenvalues'  : (n_lds,) real eigenvalues
        'pillai'       : float — Pillai's trace on real data
        'null_pillais' : (n_valid_shuffles,) null distribution
        'null_per_eig' : (n_valid_shuffles, n_lds) per-eigenvalue null
        'p_value'      : permutation p-value
        'lda'          : fitted LDA on real data
        'X_ld'         : (n_samples, n_lds) projected real data
        'y_state'      : (n_samples,) state labels for real data
    """
    rng = np.random.default_rng(rng_seed)

    # ── Step 1: load raw sessions and trial_times once ───────────────────────
    raw_sessions, trial_times_list = [], []
    for sidx in valid_sessions:
        raw = np.array(data_dic[mouse_recday][sidx]['Neuron_raw'], dtype=float)
        tt = np.array(data_dic[mouse_recday][sidx].get('Trial_times'))
        if tt is None or len(tt) == 0:
            print(f"  Session {sidx}: empty/missing Trial_times — skipped")
            continue
        if neuron_subset is not None:
            raw = raw[np.asarray(neuron_subset), :]
        raw_sessions.append(raw)
        trial_times_list.append(tt)

    if not raw_sessions:
        raise ValueError(f"No valid sessions for {mouse_recday}.")

    n_neurons = raw_sessions[0].shape[0]
    print(f"{mouse_recday}: {len(raw_sessions)} sessions, {n_neurons} neurons")

    # ── Step 2: optional trial balancing (applied to BOTH real and shuffled) ─
    if balance_trials_per_task:
        n_trials_per_session = [tt.shape[0] for tt in trial_times_list]
        min_n_trials = min(n_trials_per_session)
        print(f"  Balancing trials per task: keeping {min_n_trials} per session "
              f"(originally {n_trials_per_session})")
        rng_bal = np.random.default_rng(balance_rng_seed)
        trial_times_list = [
            tt[np.sort(rng_bal.choice(tt.shape[0], min_n_trials, replace=False))]
            if tt.shape[0] > min_n_trials else tt
            for tt in trial_times_list
        ]

    # ── Step 3: pipeline helper (real or roll-shuffled raws) ─────────────────
    def _pipeline(raws):
        """Returns (X_pca, y_state, lda, eigvals, pillai, X_ld) or None on failure."""
        concat_z, offsets, lengths, _ = zscore_concat_sessions(raws)

        X_list, y_list = [], []
        for sess_i, (offset, length, tt) in enumerate(
                zip(offsets, lengths, trial_times_list)):
            X_s, y_s, _, _, _ = extract_state_trial_vectors(
                concat_z, offset, length, tt, sess_i
            )
            if X_s.shape[0] > 0:
                X_list.append(X_s)
                y_list.append(y_s)

        if not X_list:
            return None

        X = np.vstack(X_list)
        y_state = np.concatenate(y_list)

        if set(EVENT_LABELS) - set(y_state):
            return None

        n_comp = min(X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp)
        X_pca_full = pca.fit_transform(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_pcs = min(int(np.searchsorted(cumvar, variance_thresh)) + 1, n_comp)
        X_pca = X_pca_full[:, :n_pcs]

        n_features = X_pca.shape[1]
        W = np.zeros((n_features, n_features))
        B = np.zeros((n_features, n_features))
        mu_global = X_pca.mean(axis=0)

        for label in EVENT_LABELS:
            Xc = X_pca[y_state == label]
            if len(Xc) == 0:
                continue
            mu_k = Xc.mean(axis=0)
            Xc_c = Xc - mu_k
            W += Xc_c.T @ Xc_c
            diff = (mu_k - mu_global)[:, None]
            B += len(Xc) * (diff @ diff.T)

        WinvB = np.linalg.pinv(W) @ B
        eigvals = np.linalg.eigvals(WinvB).real
        eigvals = np.sort(eigvals)[::-1][:len(EVENT_LABELS) - 1]
        pillai = float(np.sum(eigvals / (1 + eigvals)))

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_pca, y_state)
        X_ld = lda.transform(X_pca)

        return X_pca, y_state, lda, eigvals, pillai, X_ld

    # ── Step 4: real data ─────────────────────────────────────────────────────
    real_result = _pipeline(raw_sessions)
    if real_result is None:
        raise RuntimeError(f"Pipeline failed on real data for {mouse_recday}.")

    X_pca_real, y_state_real, lda_real, real_eigvals, real_pillai, X_ld_real = real_result
    n_lds = len(real_eigvals)
    print(f"  Real Pillai = {real_pillai:.4f} | eigenvalues: {np.round(real_eigvals, 3)}")

    # ── Step 5: roll-shuffle null ─────────────────────────────────────────────
    null_pillais = np.zeros(n_shuffles)
    null_per_eig = np.zeros((n_shuffles, n_lds))

    for i in range(n_shuffles):
        rolled_sessions = []
        for raw, tt in zip(raw_sessions, trial_times_list):
            session_length = raw.shape[1]
            mean_trial_len = int(tt[:, -1].mean() - tt[:, 0].mean())
            min_roll = max(mean_trial_len, 1)
            max_roll = session_length - min_roll
            if max_roll <= min_roll:
                offset = rng.integers(1, session_length)
            else:
                offset = rng.integers(min_roll, max_roll)
            rolled_sessions.append(np.roll(raw, offset, axis=1))

        null_result = _pipeline(rolled_sessions)
        if null_result is None:
            null_pillais[i] = np.nan
            null_per_eig[i, :] = np.nan
            print(f"  Shuffle {i}: pipeline returned None — skipped")
            continue

        _, _, _, null_eigvals, null_pillai, _ = null_result
        null_pillais[i] = null_pillai
        n_ev = min(len(null_eigvals), n_lds)
        null_per_eig[i, :n_ev] = null_eigvals[:n_ev]

        if (i + 1) % 10 == 0:
            print(f"  Shuffle {i+1}/{n_shuffles} done")

    valid_mask = ~np.isnan(null_pillais)
    null_pillais = null_pillais[valid_mask]
    null_per_eig = null_per_eig[valid_mask]

    p_value = (np.sum(null_pillais >= real_pillai) + 1) / (len(null_pillais) + 1)
    print(f"  p = {p_value:.4f}  (n_valid_shuffles = {valid_mask.sum()})")

    # ── Step 6: plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bal_tag = " (trials balanced)" if balance_trials_per_task else ""
    fig.suptitle(f'LDA state separability — {mouse_recday}{bal_tag}\n'
                 f'(null = circular roll shuffles, n={len(null_pillais)})',
                 fontsize=12)

    ax = axes[0]
    ax.hist(null_pillais, bins=30, color='0.65', edgecolor='white',
            density=True, alpha=0.85, label='Null (roll shuffled)')
    ax.axvline(real_pillai, color='crimson', lw=2.5,
               label=f'Real  ({real_pillai:.3f})')
    ax.text(0.97, 0.95, f'p = {p_value:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, color='crimson', fontweight='bold')
    ax.set_xlabel("Pillai's trace  Σ λᵢ/(1+λᵢ)", fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    x = np.arange(1, n_lds + 1)
    null_95 = np.nanpercentile(null_per_eig, 95, axis=0)
    null_50 = np.nanpercentile(null_per_eig, 50, axis=0)

    ax.bar(x, real_eigvals, color='crimson', alpha=0.8, label='Real')
    ax.plot(x, null_95, 'k--o', lw=1.5, markersize=6, label='Null 95th pct')
    ax.plot(x, null_50, 'k:o', lw=1.0, markersize=5, label='Null median')
    ax.set_xticks(x)
    ax.set_xticklabels([f'LD{j}' for j in x])
    ax.set_xlabel('Discriminant axis', fontsize=11)
    ax.set_ylabel('Eigenvalue λ (between/within ratio)', fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    return {
        'eigenvalues': real_eigvals,
        'pillai': real_pillai,
        'null_pillais': null_pillais,
        'null_per_eig': null_per_eig,
        'p_value': p_value,
        'lda': lda_real,
        'X_ld': X_ld_real,
        'y_state': y_state_real,
    }


# ─────────────────────────────────────────────────────────────────────────────
def compute_pseudotime_state_correlations(X, y_state, sess_id, trial_id):
    """
    Pseudotime-controlled population-vector correlation between sessions.

    For every (session_pair, trial_number shared by both, state_a_in_session1,
    state_b_in_session2) combination, correlate the two population vectors and
    bucket by whether the states match.

    Trial number acts as a pseudotime control: paired trials sit at roughly the
    same point in the recording timeline, so slow drift in firing rates affects
    matched trials similarly and partially out of the comparison.

    Parameters
    ----------
    X        : (n_samples, n_neurons) z-scored population vectors,
               from `build_trialbins_dataset`.
    y_state  : (n_samples,) state labels in EVENT_LABELS.
    sess_id  : (n_samples,) integer session index.
    trial_id : (n_samples,) trial index within session.

    Returns
    -------
    within_state : list of dicts {state, sess1, sess2, trial, corr}
    across_state : list of dicts {state_a, state_b, sess1, sess2, trial, corr}
    """
    sessions = sorted(np.unique(sess_id).tolist())
    within_state = []
    across_state = []

    for i_s1 in range(len(sessions)):
        for i_s2 in range(i_s1 + 1, len(sessions)):
            s1, s2 = sessions[i_s1], sessions[i_s2]

            trials_s1 = set(trial_id[sess_id == s1].tolist())
            trials_s2 = set(trial_id[sess_id == s2].tolist())
            matched = sorted(trials_s1 & trials_s2)

            for t in matched:
                vecs_s1, vecs_s2 = {}, {}
                for state in EVENT_LABELS:
                    m1 = (sess_id == s1) & (trial_id == t) & (y_state == state)
                    m2 = (sess_id == s2) & (trial_id == t) & (y_state == state)
                    if m1.sum() == 1:
                        vecs_s1[state] = X[m1].ravel()
                    if m2.sum() == 1:
                        vecs_s2[state] = X[m2].ravel()

                for sa, v1 in vecs_s1.items():
                    for sb, v2 in vecs_s2.items():
                        if np.std(v1) == 0 or np.std(v2) == 0:
                            continue
                        r = float(np.corrcoef(v1, v2)[0, 1])
                        if not np.isfinite(r):
                            continue
                        rec = {'sess1': int(s1), 'sess2': int(s2),
                               'trial': int(t), 'corr': r}
                        if sa == sb:
                            rec['state'] = sa
                            within_state.append(rec)
                        else:
                            rec['state_a'] = sa
                            rec['state_b'] = sb
                            across_state.append(rec)

    return within_state, across_state


# ─────────────────────────────────────────────────────────────────────────────
def plot_pseudotime_state_correlations(within_state, across_state, mouse_recday=''):
    """
    Plot pseudotime-controlled within-state vs across-state correlation
    distributions for one recording day.

    Returns
    -------
    fig    : matplotlib figure
    t_stat : two-sample t-statistic (within vs across)
    p_val  : two-sided p-value
    """
    from scipy import stats as _stats

    within_r = np.array([d['corr'] for d in within_state])
    across_r = np.array([d['corr'] for d in across_state])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f'Pseudotime-matched state correlation — {mouse_recday}',
                 fontsize=12)

    ax = axes[0]
    bins = np.linspace(-1, 1, 41)
    ax.hist(across_r, bins=bins, color='steelblue', alpha=0.6,
            label=f'Across (n={len(across_r)}, μ={across_r.mean():.3f})')
    ax.hist(within_r, bins=bins, color='crimson', alpha=0.6,
            label=f'Within (n={len(within_r)}, μ={within_r.mean():.3f})')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(across_r.mean(), color='steelblue', linewidth=2)
    ax.axvline(within_r.mean(), color='crimson', linewidth=2)
    ax.set_xlabel('Population vector correlation')
    ax.set_ylabel('Count')
    ax.set_title('Within vs across state')
    ax.legend()

    t_stat, p_val = _stats.ttest_ind(within_r, across_r)

    ax = axes[1]
    ax.axis('off')
    summary = (f"Mean within: {within_r.mean():.4f}\n"
               f"Mean across: {across_r.mean():.4f}\n"
               f"Difference:  {within_r.mean() - across_r.mean():.4f}\n"
               f"\n"
               f"Two-sample t-test:\n"
               f"  t = {t_stat:.3f}\n"
               f"  p = {p_val:.2e}\n"
               f"\n"
               f"n within: {len(within_r)}\n"
               f"n across: {len(across_r)}")
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.show()

    return fig, float(t_stat), float(p_val)


# ═════════════════════════════════════════════════════════════════════════════
# Pillai-trace control analyses (T1: LOTO, T2: within-task, T3: within-session
# label-shuffle null). All three are tests of whether the across-task LDA
# separation is genuine cross-task state tuning vs an artefact of within-task
# similarity.
# ═════════════════════════════════════════════════════════════════════════════

def _compute_pillai_from_pca(X_pca, y_state, event_labels=None):
    """Compute W⁻¹B eigenvalues and Pillai's trace from PCA-projected data.

    Parameters
    ----------
    X_pca : (n_samples, n_features)
    y_state : (n_samples,) state labels
    event_labels : list of state labels to include. Defaults to module-level EVENT_LABELS.

    Returns
    -------
    eigvals : (k-1,) descending eigenvalues of W⁻¹B (top k-1 only)
    pillai  : float, Σ λᵢ/(1+λᵢ)
    """
    if event_labels is None:
        event_labels = EVENT_LABELS

    n_features = X_pca.shape[1]
    W = np.zeros((n_features, n_features))
    B = np.zeros((n_features, n_features))
    mu_global = X_pca.mean(axis=0)

    for label in event_labels:
        Xc = X_pca[y_state == label]
        if len(Xc) == 0:
            continue
        mu_k = Xc.mean(axis=0)
        Xc_c = Xc - mu_k
        W += Xc_c.T @ Xc_c
        diff = (mu_k - mu_global)[:, None]
        B += len(Xc) * (diff @ diff.T)

    WinvB = np.linalg.pinv(W) @ B
    eigvals = np.linalg.eigvals(WinvB).real
    eigvals = np.sort(eigvals)[::-1][:len(event_labels) - 1]
    pillai = float(np.sum(eigvals / (1 + eigvals)))
    return eigvals, pillai


# ─────────────────────────────────────────────────────────────────────────────
def run_within_session_shuffle_null(data_dic, mouse_recday, valid_sessions,
                                     neuron_subset=None,
                                     variance_thresh=PCA_VARIANCE_THRESH,
                                     n_shuffles=100,
                                     rng_seed=42,
                                     balance_trials_per_task=False,
                                     balance_rng_seed=42):
    """
    T3: Pillai trace null built from within-session state-label shuffles.

    Same as `run_lda_eigenvalue_separation` but the null permutes state labels
    WITHIN each session (preserving session identity, trial structure, and the
    actual extracted population vectors). The real Pillai is identical to the
    one returned by `run_lda_eigenvalue_separation`; only the null differs.

    Stricter than the roll-shuffle null because the within-session vector
    structure (clusters, autocorrelation, drift) is preserved in the null —
    only the label assignment is randomized.

    Returns
    -------
    dict with keys:
        'pillai'              : float, real Pillai
        'eigenvalues'         : (k-1,) real eigenvalues
        'null_pillais'        : (n_valid,) null Pillai under within-session shuffle
        'p_value'             : permutation p-value
        'null_type'           : 'within_session_shuffle'
    """
    rng = np.random.default_rng(rng_seed)

    # Build real dataset once (z-scored across sessions)
    X, y_state, sess_id, trial_id, zero_var_mask, _ = build_trialbins_dataset(
        data_dic, mouse_recday, valid_sessions,
        neuron_subset=neuron_subset,
        balance_trials_per_task=balance_trials_per_task,
        balance_rng_seed=balance_rng_seed,
    )

    if set(EVENT_LABELS) - set(y_state):
        raise RuntimeError(f"Some EVENT_LABELS absent from {mouse_recday}.")

    # PCA once on the real data
    n_comp = min(X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca_full = pca.fit_transform(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = min(int(np.searchsorted(cumvar, variance_thresh)) + 1, n_comp)
    X_pca = X_pca_full[:, :n_pcs]

    # Real Pillai
    real_eigvals, real_pillai = _compute_pillai_from_pca(X_pca, y_state)
    print(f"{mouse_recday}: real Pillai = {real_pillai:.4f}  (within-session shuffle null)")

    # Null: shuffle labels within each session, recompute Pillai on same X_pca
    sessions = sorted(np.unique(sess_id).tolist())
    null_pillais = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        y_shuf = y_state.copy()
        for s in sessions:
            idx = np.where(sess_id == s)[0]
            y_shuf[idx] = rng.permutation(y_shuf[idx])
        try:
            _, p_null = _compute_pillai_from_pca(X_pca, y_shuf)
            null_pillais[i] = p_null
        except Exception:
            continue
        if (i + 1) % 20 == 0:
            print(f"  Shuffle {i+1}/{n_shuffles} done")

    valid = ~np.isnan(null_pillais)
    null_pillais = null_pillais[valid]
    p_value = (np.sum(null_pillais >= real_pillai) + 1) / (len(null_pillais) + 1)
    print(f"  p = {p_value:.4f} (n_valid_shuffles = {len(null_pillais)})")

    return {
        'pillai': real_pillai,
        'eigenvalues': real_eigvals,
        'null_pillais': null_pillais,
        'p_value': p_value,
        'null_type': 'within_session_shuffle',
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_lda_eigenvalue_separation_loto(data_dic, mouse_recday, valid_sessions,
                                        neuron_subset=None,
                                        variance_thresh=PCA_VARIANCE_THRESH,
                                        n_shuffles=100,
                                        rng_seed=42,
                                        balance_trials_per_task=False,
                                        balance_rng_seed=42):
    """
    T1: Leave-one-task-out Pillai trace.

    For each held-out session: fit PCA + LDA on the N-1 remaining sessions,
    project the held-out session's data through that fixed PCA+LDA, then
    compute Pillai from the projected held-out data's scatter matrices.
    Tests whether the LDA direction generalizes across tasks.

    Null per fold: shuffle the held-out session's state labels n_shuffles times
    and recompute Pillai on the projected data.

    Returns
    -------
    dict with keys:
        'loto_pillais'      : (n_sessions,) real Pillai per held-out fold
        'null_loto_pillais' : (n_sessions, n_shuffles) null
        'p_values'          : (n_sessions,) per-fold one-sided p
        'mean_loto_pillai'  : float
        'session_order'     : list of held-out session IDs
    """
    rng = np.random.default_rng(rng_seed)

    # Build the full dataset once
    X, y_state, sess_id, trial_id, _, _ = build_trialbins_dataset(
        data_dic, mouse_recday, valid_sessions,
        neuron_subset=neuron_subset,
        balance_trials_per_task=balance_trials_per_task,
        balance_rng_seed=balance_rng_seed,
    )

    sessions = sorted(np.unique(sess_id).tolist())
    if len(sessions) < 2:
        raise RuntimeError(f"{mouse_recday}: LOTO needs >= 2 sessions (got {len(sessions)}).")

    n_lds = len(EVENT_LABELS) - 1
    loto_pillais = np.full(len(sessions), np.nan)
    null_pillais = np.full((len(sessions), n_shuffles), np.nan)
    p_values = np.full(len(sessions), np.nan)

    print(f"{mouse_recday}: LOTO over {len(sessions)} sessions")

    for k, s_test in enumerate(sessions):
        train_mask = sess_id != s_test
        test_mask = sess_id == s_test

        X_train = X[train_mask]
        y_train = y_state[train_mask]
        X_test = X[test_mask]
        y_test = y_state[test_mask]

        if set(EVENT_LABELS) - set(y_test):
            print(f"  Fold {k+1}/{len(sessions)} (held-out s={s_test}): missing state in test, skipped")
            continue

        # PCA fit on train, project both train and test
        n_comp = min(X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_comp)
        X_train_pca_full = pca.fit_transform(X_train)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_pcs = min(int(np.searchsorted(cumvar, variance_thresh)) + 1, n_comp)

        # Project test through train PCA
        X_test_centered = X_test - pca.mean_
        X_test_pca = X_test_centered @ pca.components_[:n_pcs].T

        # Real Pillai on test data
        try:
            _, p_real = _compute_pillai_from_pca(X_test_pca, y_test)
        except Exception as e:
            print(f"  Fold {k+1} (held-out s={s_test}): real Pillai failed ({e})")
            continue
        loto_pillais[k] = p_real

        # Null: shuffle test labels, recompute Pillai
        for i in range(n_shuffles):
            y_shuf = rng.permutation(y_test)
            try:
                _, p_null = _compute_pillai_from_pca(X_test_pca, y_shuf)
                null_pillais[k, i] = p_null
            except Exception:
                continue

        valid = ~np.isnan(null_pillais[k])
        n_valid = valid.sum()
        if n_valid > 0:
            p_values[k] = (np.sum(null_pillais[k, valid] >= p_real) + 1) / (n_valid + 1)
        print(f"  Fold {k+1}/{len(sessions)} (held-out s={s_test}): "
              f"Pillai={p_real:.4f}, p={p_values[k]:.4f}")

    mean_loto = float(np.nanmean(loto_pillais))
    print(f"{mouse_recday}: mean LOTO Pillai = {mean_loto:.4f}")

    return {
        'loto_pillais': loto_pillais,
        'null_loto_pillais': null_pillais,
        'p_values': p_values,
        'mean_loto_pillai': mean_loto,
        'session_order': sessions,
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_within_task_pillai(data_dic, mouse_recday, valid_sessions,
                            neuron_subset=None,
                            variance_thresh=PCA_VARIANCE_THRESH,
                            n_shuffles=100,
                            rng_seed=42):
    """
    T2: Per-session (within-task) Pillai trace.

    For each session separately, extract that session's (trial, state) vectors,
    z-score within session, PCA + LDA, compute Pillai. Returns per-session
    array. Comparison to the across-task Pillai (from run_lda_eigenvalue_separation)
    tells you whether pooling tasks adds signal.

    Null per session: shuffle that session's state labels and recompute Pillai.

    Returns
    -------
    dict with keys:
        'per_session_pillais'      : (n_sessions,) within-task Pillai
        'null_per_session_pillais' : (n_sessions, n_shuffles)
        'p_values'                 : (n_sessions,) one-sided p
        'mean_within_task_pillai'  : float
        'session_order'            : list of session IDs analysed
    """
    rng = np.random.default_rng(rng_seed)

    pillais = np.full(len(valid_sessions), np.nan)
    null_p = np.full((len(valid_sessions), n_shuffles), np.nan)
    p_values = np.full(len(valid_sessions), np.nan)
    sessions_used = []

    print(f"{mouse_recday}: within-task Pillai over {len(valid_sessions)} sessions")

    for k, sidx in enumerate(valid_sessions):
        # Load raw for this session only
        raw = data_dic[mouse_recday][sidx]['Neuron_raw']
        if neuron_subset is not None:
            raw = raw[np.asarray(neuron_subset), :]
        tt = data_dic[mouse_recday][sidx]['Trial_times']

        # Per-session z-score (different from build_trialbins_dataset which
        # z-scores across all sessions concatenated). Within-task analysis
        # should be self-contained per session.
        raw = np.asarray(raw, dtype=float)
        mu = np.nanmean(raw, axis=1, keepdims=True)
        sd = np.nanstd(raw, axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        zsc = (raw - mu) / sd

        # Extract (trial, state) vectors for this single session
        X_s, y_s, _, _, _ = extract_state_trial_vectors(
            zsc, session_offset=0, session_length=zsc.shape[1],
            trial_times=tt, sess_index=0,
        )

        if X_s.shape[0] == 0 or set(EVENT_LABELS) - set(y_s):
            print(f"  Session {sidx}: insufficient state samples, skipped")
            sessions_used.append(sidx)
            continue

        # PCA + Pillai
        n_comp = min(X_s.shape[0] - 1, X_s.shape[1])
        pca = PCA(n_components=n_comp)
        X_pca_full = pca.fit_transform(X_s)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_pcs = min(int(np.searchsorted(cumvar, variance_thresh)) + 1, n_comp)
        X_pca = X_pca_full[:, :n_pcs]

        try:
            _, p_real = _compute_pillai_from_pca(X_pca, y_s)
        except Exception as e:
            print(f"  Session {sidx}: real Pillai failed ({e})")
            sessions_used.append(sidx)
            continue
        pillais[k] = p_real

        for i in range(n_shuffles):
            y_shuf = rng.permutation(y_s)
            try:
                _, p_null = _compute_pillai_from_pca(X_pca, y_shuf)
                null_p[k, i] = p_null
            except Exception:
                continue

        valid = ~np.isnan(null_p[k])
        n_valid = valid.sum()
        if n_valid > 0:
            p_values[k] = (np.sum(null_p[k, valid] >= p_real) + 1) / (n_valid + 1)
        print(f"  Session {sidx}: Pillai={p_real:.4f}, p={p_values[k]:.4f}")
        sessions_used.append(sidx)

    mean_within = float(np.nanmean(pillais))
    print(f"{mouse_recday}: mean within-task Pillai = {mean_within:.4f}")

    return {
        'per_session_pillais': pillais,
        'null_per_session_pillais': null_p,
        'p_values': p_values,
        'mean_within_task_pillai': mean_within,
        'session_order': sessions_used,
    }

