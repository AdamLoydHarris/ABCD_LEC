"""
LDA state separation using trial-averaged population vectors.

Instead of fine time bins, one feature vector per (session, trial, state)
is built by averaging z-scored firing rates across the raw time window of
each state within each trial.

Normalization: z-score per neuron across the fully concatenated (all sessions)
               raw recording, then extract per-(trial, state) averages.

Main entry point
----------------
results = run_trialbins_lda_analysis(
    data_dic, mouse_recday, valid_sessions,
    neuron_subset=sig_lec_neurons_dic.get(mouse_recday),
)

Reuses run_lda_state_separation and run_logo_state_cv from lda_state_separation.py.
"""

import numpy as np
from sklearn.decomposition import PCA
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
                                min_samples=MIN_STATE_SAMPLES):
    """
    Extract one averaged population vector per (trial, state) from a z-scored
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
            vec = concat_z[:, t_start_g:t_end_g].mean(axis=1)  # (n_neurons,)

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
def build_trialbins_dataset(data_dic, mouse_recday, valid_sessions,
                             neuron_subset=None):
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
            concat_z, offset, length, tt, sess_i
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
                                n_logo_shuffles=1000):
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
        data_dic, mouse_recday, valid_sessions, neuron_subset=neuron_subset
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


