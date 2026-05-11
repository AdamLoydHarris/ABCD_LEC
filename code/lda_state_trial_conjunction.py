"""
LDA on the joint (state × trial_number) label, with goal-progress binning.

Each (session, trial, state) combination yields 3 data points (early/middle/late
thirds of the state window). LDA is fit to separate the joint (state, trial)
label, aiming to capture:
  LD1 — trial number (analogue of time in session)
  LD2 — state identity (A / B / C / D)

Only sessions with >= min_trials (default 7) are included.

Two analyses
------------
1. Aggregate all qualifying sessions → visualise LD projections via
   plot_conjunction_ld_scatter (2-D scatter) and plot_conjunction_ld_axes
   (per-axis 1-D strip plots, up to 10 LDs, coloured by state and trial).
2. Decoding of state (4-class) or trial number (n-class) with
   leave-one-session-out cross-validation via run_conjunction_decoding.

Main entry point
----------------
results = run_conjunction_lda_analysis(
    data_dic, mouse_recday, valid_sessions,
    neuron_subset=None, min_trials=7,
)

Reuses
------
  zscore_concat_sessions, apply_pca_trialbins, EVENT_LABELS, EVENT_COLOURS,
    PCA_VARIANCE_THRESH, MIN_STATE_SAMPLES
  from lda_state_analysis_trialbins

  extract_state_goalprogress_vectors, PROGRESS_BINS, N_PROGRESS_BINS
  from lda_state_analysis_goalprogress

  run_logo_state_cv
  from lda_state_separation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

from lda_state_analysis_trialbins import (
    zscore_concat_sessions,
    apply_pca_trialbins,
    EVENT_LABELS,
    EVENT_COLOURS,
    PCA_VARIANCE_THRESH,
    MIN_STATE_SAMPLES,
)
from lda_state_analysis_goalprogress import (
    extract_state_goalprogress_vectors,
    PROGRESS_BINS,
    N_PROGRESS_BINS,
)
from lda_state_separation import run_logo_state_cv

MIN_TRIALS_DEFAULT = 7


# ─────────────────────────────────────────────────────────────────────────────
def filter_sessions_by_trials(data_dic, mouse_recday, valid_sessions,
                               min_trials=MIN_TRIALS_DEFAULT,
                               min_sessions=3):
    """
    Return only the sessions that have at least min_trials trials, provided
    that at least min_sessions sessions pass the threshold.  Raises ValueError
    if fewer than min_sessions sessions qualify.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    min_trials : int
    min_sessions : int
        Minimum number of qualifying sessions required (default 3).

    Returns
    -------
    filtered : list
        Subset of valid_sessions passing the threshold.
    """
    filtered, excluded = [], []
    for sidx in valid_sessions:
        n = data_dic[mouse_recday][sidx]['num_trials']
        if n >= min_trials:
            filtered.append(sidx)
        else:
            excluded.append((sidx, n))

    print(f"  Session filter (min_trials={min_trials}): "
          f"{len(filtered)} kept, {len(excluded)} excluded")
    for sidx, n in excluded:
        print(f"    excluded session {sidx}: {n} trials < {min_trials}")

    if len(filtered) < min_sessions:
        raise ValueError(
            f"{mouse_recday}: only {len(filtered)} session(s) with "
            f">= {min_trials} trials; need at least {min_sessions}."
        )

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
def build_conjunction_dataset(data_dic, mouse_recday, valid_sessions,
                               neuron_subset=None,
                               min_trials=MIN_TRIALS_DEFAULT):
    """
    Build the (trial × state × progress-bin) z-scored dataset, restricted to
    sessions with enough trials.  The conjunction class label is the string
    "{state}_{trial_index:02d}", e.g. "A_00", "B_07".

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional
    min_trials : int

    Returns
    -------
    X : np.ndarray  (n_samples, n_neurons)
    y_state : np.ndarray of str  (n_samples,)   'A'/'B'/'C'/'D'
    y_trial : np.ndarray of int  (n_samples,)   0-based within-session trial index
    y_conjunction : np.ndarray of str  (n_samples,)  e.g. 'A_00'
    progress : np.ndarray of str  (n_samples,)  'early'/'middle'/'late'
    sess_id : np.ndarray of int  (n_samples,)
    trial_id : np.ndarray of int  (n_samples,)
    zero_var_mask : np.ndarray of bool  (n_neurons,)
    n_skipped : int
    """
    filtered_sessions = filter_sessions_by_trials(
        data_dic, mouse_recday, valid_sessions, min_trials=min_trials
    )
    if not filtered_sessions:
        raise ValueError(
            f"No sessions with >= {min_trials} trials for {mouse_recday}."
        )

    raw_sessions, trial_times_list = [], []
    for sidx in filtered_sessions:
        raw = data_dic[mouse_recday][sidx]['Neuron_raw']
        if neuron_subset is not None:
            raw = raw[np.asarray(neuron_subset), :]
        raw_sessions.append(raw)
        trial_times_list.append(data_dic[mouse_recday][sidx]['Trial_times'])

    n_neurons = raw_sessions[0].shape[0]
    print(f"{mouse_recday}: {len(filtered_sessions)} sessions, {n_neurons} neurons")

    concat_z, offsets, lengths, zero_var_mask = zscore_concat_sessions(raw_sessions)

    n_zero = int(zero_var_mask.sum())
    if n_zero:
        print(f"  WARNING: {n_zero} neuron(s) with zero variance set to 0.")

    X_list, y_state_list, y_trial_list, y_conj_list = [], [], [], []
    prog_list, sid_list, tid_list = [], [], []
    total_skipped = 0

    for sess_i, (sidx, offset, length, tt) in enumerate(
            zip(filtered_sessions, offsets, lengths, trial_times_list)):

        tt_trunc = tt[:min_trials, :]
        print(f"  Session {sidx} (index {sess_i}): "
              f"{tt.shape[0]} trials total, using first {tt_trunc.shape[0]}, "
              f"T={length} samples")

        X_s, y_s, prog_s, sid_s, tid_s, skipped = \
            extract_state_goalprogress_vectors(
                concat_z, offset, length, tt_trunc, sess_i
            )

        for trial_idx, label, bin_label, reason in skipped:
            print(f"    SKIP trial={trial_idx} state={label} "
                  f"bin={bin_label}: {reason}")
        total_skipped += len(skipped)

        if X_s.shape[0] == 0:
            continue

        # Build conjunction labels: "{state}_{trial:02d}"
        y_conj_s = np.array(
            [f"{st}_{ti:02d}" for st, ti in zip(y_s, tid_s)],
            dtype=object
        )

        X_list.append(X_s)
        y_state_list.append(y_s)
        y_trial_list.append(tid_s)
        y_conj_list.append(y_conj_s)
        prog_list.append(prog_s)
        sid_list.append(sid_s)
        tid_list.append(tid_s)

    if not X_list:
        raise RuntimeError(
            f"No valid samples extracted for {mouse_recday}."
        )

    X            = np.vstack(X_list)
    y_state      = np.concatenate(y_state_list)
    y_trial      = np.concatenate(y_trial_list)
    y_conjunction = np.concatenate(y_conj_list).astype(str)
    progress     = np.concatenate(prog_list)
    sess_id      = np.concatenate(sid_list)
    trial_id     = np.concatenate(tid_list)

    print(f"  Dataset: {X.shape[0]} samples ({total_skipped} skipped)")
    print(f"  Unique conjunction classes: {len(np.unique(y_conjunction))}")
    print(f"  Samples per session: "
          + str(dict(zip(*np.unique(sess_id, return_counts=True)))))

    return (X, y_state, y_trial, y_conjunction, progress,
            sess_id, trial_id, zero_var_mask, total_skipped)


# ─────────────────────────────────────────────────────────────────────────────
def plot_conjunction_ld_scatter(X_conj_ld, y_state, trial_id, sess_id,
                                 mouse_recday, max_lds=3, seed=42):
    """
    Two-panel scatter of LD1 vs LD2 (and optionally LD1 vs LD3).

    Left panel  — points coloured by state (A/B/C/D).
    Right panel — points coloured by within-session trial number (viridis).

    Class centroids (per conjunction label) are shown as faint '+' markers.

    Parameters
    ----------
    X_conj_ld : np.ndarray  (n_samples, n_lds)
    y_state : np.ndarray of str  (n_samples,)
    trial_id : np.ndarray of int  (n_samples,)
    sess_id : np.ndarray of int  (n_samples,)
    mouse_recday : str
    max_lds : int  (3 → also plot LD1 vs LD3)
    seed : int
    """
    rng = np.random.default_rng(seed)
    n_ld = min(X_conj_ld.shape[1], max_lds)

    # Number of LD pairs to plot: at least (LD1, LD2); add (LD1, LD3) if available
    ld_pairs = [(0, 1)]
    if n_ld >= 3:
        ld_pairs.append((0, 2))

    n_pairs = len(ld_pairs)
    fig, axes = plt.subplots(n_pairs, 2,
                             figsize=(12, 5.5 * n_pairs),
                             squeeze=False)

    fig.suptitle(
        f"Conjunction LDA (state \u00d7 trial) \u2014 {mouse_recday}",
        fontsize=15, fontweight='bold', y=1.01
    )

    # Normalise trial_id for colormap (0→1)
    t_min, t_max = trial_id.min(), trial_id.max()
    t_norm = (trial_id - t_min) / max(t_max - t_min, 1)
    trial_cmap = cm.get_cmap('viridis')

    for row_i, (xi, yi) in enumerate(ld_pairs):
        ax_state = axes[row_i, 0]
        ax_trial = axes[row_i, 1]

        proj_x = X_conj_ld[:, xi]
        proj_y = X_conj_ld[:, yi]

        jitter_x = rng.uniform(-0.02, 0.02, size=len(proj_x)) * (proj_x.max() - proj_x.min())
        jitter_y = rng.uniform(-0.02, 0.02, size=len(proj_y)) * (proj_y.max() - proj_y.min())

        # ── Left panel: colour by state ──────────────────────────────
        for label in EVENT_LABELS:
            mask = y_state == label
            if mask.any():
                ax_state.scatter(proj_x[mask] + jitter_x[mask],
                                 proj_y[mask] + jitter_y[mask],
                                 c=EVENT_COLOURS[label],
                                 s=18, alpha=0.5, label=label, zorder=2)

        # State centroids
        for label in EVENT_LABELS:
            mask = y_state == label
            if mask.sum() > 0:
                cx, cy = proj_x[mask].mean(), proj_y[mask].mean()
                ax_state.scatter(cx, cy, c=EVENT_COLOURS[label],
                                 s=180, marker='*', edgecolors='black',
                                 linewidths=0.8, zorder=5)
                ax_state.text(cx, cy, f' {label}', fontsize=10,
                              fontweight='bold', color=EVENT_COLOURS[label],
                              va='center', zorder=6)

        ax_state.set_xlabel(f'LD {xi + 1}')
        ax_state.set_ylabel(f'LD {yi + 1}')
        ax_state.set_title(f'LD{xi+1} vs LD{yi+1} — coloured by state')
        ax_state.legend(title='State', fontsize=9, markerscale=1.4)

        # ── Right panel: colour by trial number ──────────────────────
        sc = ax_trial.scatter(proj_x + jitter_x,
                              proj_y + jitter_y,
                              c=t_norm, cmap='viridis',
                              s=18, alpha=0.5, zorder=2)
        cbar = fig.colorbar(sc, ax=ax_trial, pad=0.02)
        cbar.set_label('Trial number (within session)', fontsize=9)
        # Re-label colorbar ticks as actual trial numbers
        tick_vals = np.linspace(0, 1, 5)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(
            [f"{int(round(v * (t_max - t_min) + t_min))}" for v in tick_vals]
        )

        # Per-trial centroids as faint crosses (aggregated over states/sessions)
        unique_trials = np.unique(trial_id)
        for t in unique_trials:
            mask = trial_id == t
            cx, cy = proj_x[mask].mean(), proj_y[mask].mean()
            ax_trial.scatter(cx, cy,
                             c=[trial_cmap(t_norm[mask][0])],
                             s=60, marker='+',
                             linewidths=1.5, alpha=0.7, zorder=4)

        ax_trial.set_xlabel(f'LD {xi + 1}')
        ax_trial.set_ylabel(f'LD {yi + 1}')
        ax_trial.set_title(f'LD{xi+1} vs LD{yi+1} — coloured by trial number')

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def plot_conjunction_ld_axes(X_conj_ld, y_state, trial_id,
                              mouse_recday, max_lds=10, n_shuffles=1000,
                              seed=42):
    """
    For each LD axis (up to max_lds), show two side-by-side 1-D strip plots:

      Left  — samples arranged in state rows (A / B / C / D), coloured by
               state.  Real centroids (black ± SEM) and null centroids from
               circular label rolls (grey ± SE) are overlaid, mirroring the
               style of plot_ld_projections in lda_state_analysis_trialbins.

      Right — samples arranged in trial-number rows, coloured by trial number
               via the viridis colormap.  Per-trial centroids (black ±SEM) and
               circular-roll null centroids (grey ±SE) are also shown.

    Parameters
    ----------
    X_conj_ld : np.ndarray  (n_samples, n_lds)
    y_state   : np.ndarray of str  (n_samples,)
    trial_id  : np.ndarray of int  (n_samples,)   0-based within-session index
    mouse_recday : str
    max_lds   : int   maximum number of LD axes to plot (default 10)
    n_shuffles : int  number of circular rolls for null centroids
    seed      : int
    """
    from matplotlib.lines import Line2D

    n_ld      = min(X_conj_ld.shape[1], max_lds)
    n_samples = len(y_state)
    rng       = np.random.default_rng(seed)

    unique_trials = np.sort(np.unique(trial_id))
    n_trials      = len(unique_trials)
    trial_cmap    = cm.get_cmap('viridis')
    t_norm_map    = {t: i / max(n_trials - 1, 1) for i, t in enumerate(unique_trials)}

    state_y  = {lab: i for i, lab in enumerate(EVENT_LABELS)}
    trial_y  = {t: i for i, t in enumerate(unique_trials)}

    jitter = rng.uniform(-0.3, 0.3, size=n_samples)

    # ── Precompute circular-roll null centroids ────────────────────────────
    roll_offsets = rng.integers(1, n_samples, size=n_shuffles)

    # state null: (n_shuffles, n_ld, n_states)
    null_state = np.zeros((n_shuffles, n_ld, len(EVENT_LABELS)))
    # trial null: (n_shuffles, n_ld, n_trials)
    null_trial = np.zeros((n_shuffles, n_ld, n_trials))

    print(f"Computing null centroids with {n_shuffles} circular label rolls…")
    for si, offset in enumerate(roll_offsets):
        y_roll  = np.roll(y_state,   offset)
        ti_roll = np.roll(trial_id,  offset)
        for li in range(n_ld):
            proj = X_conj_ld[:, li]
            for si2, lab in enumerate(EVENT_LABELS):
                m = y_roll == lab
                if m.sum() > 0:
                    null_state[si, li, si2] = proj[m].mean()
            for ti2, t in enumerate(unique_trials):
                m = ti_roll == t
                if m.sum() > 0:
                    null_trial[si, li, ti2] = proj[m].mean()

    null_state_mean = null_state.mean(axis=0)   # (n_ld, n_states)
    null_state_se   = null_state.std(axis=0)  / np.sqrt(n_shuffles)
    null_trial_mean = null_trial.mean(axis=0)   # (n_ld, n_trials)
    null_trial_se   = null_trial.std(axis=0)  / np.sqrt(n_shuffles)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_ld, 2,
                             figsize=(18, 2.8 * n_ld),
                             squeeze=False)

    fig.suptitle(
        f"Conjunction LDA — per-axis projections — {mouse_recday}",
        fontsize=15, fontweight='bold', y=1.01
    )

    for ld_i in range(n_ld):
        proj = X_conj_ld[:, ld_i]

        # ══ Left panel: state rows ════════════════════════════════════
        ax = axes[ld_i, 0]

        for lab in EVENT_LABELS:
            mask = y_state == lab
            if mask.any():
                ybase = state_y[lab]
                ax.scatter(proj[mask], ybase + jitter[mask] * 0.35,
                           c=EVENT_COLOURS[lab], s=14, alpha=0.4, zorder=2)

        # Null centroids (grey)
        for si2, lab in enumerate(EVENT_LABELS):
            cx = null_state_mean[ld_i, si2]
            se = null_state_se[ld_i, si2]
            cy = state_y[lab]
            ax.errorbar(cx, cy + 0.15, xerr=se, fmt='none',
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        ecolor='0.55', zorder=8)
            ax.vlines(cx, cy, cy + 0.3,
                      colors='0.55', linewidth=1.5, zorder=8)

        # Real centroids (black)
        cx_list = []
        for lab in EVENT_LABELS:
            mask = y_state == lab
            if mask.sum() > 0:
                cx  = proj[mask].mean()
                sem = proj[mask].std() / np.sqrt(mask.sum())
                cy  = state_y[lab]
                cx_list.append((cx, cy))
                ax.errorbar(cx, cy - 0.15, xerr=sem, fmt='none',
                            capsize=5, capthick=2, elinewidth=2,
                            ecolor='black', zorder=10)
                ax.vlines(cx, cy - 0.4, cy + 0.1,
                          colors='black', linewidth=2, zorder=10)
        if len(cx_list) > 1:
            xs = [c[0] for c in cx_list]
            ys = [c[1] - 0.15 for c in cx_list]
            ax.plot(xs, ys, 'k-', lw=1.5, zorder=9, alpha=0.6)

        ax.set_xlabel(f'LD {ld_i + 1}')
        ax.set_yticks(list(state_y.values()))
        ax.set_yticklabels(EVENT_LABELS, fontweight='bold')
        for v in state_y.values():
            ax.axhline(v, color='k', lw=0.3, alpha=0.15)
        ax.set_title(f'LD {ld_i + 1} — by state')

        if ld_i == 0:
            ax.legend(handles=[
                Line2D([0], [0], color='black', lw=2,
                       label='Real (mean ± SEM)'),
                Line2D([0], [0], color='0.55', lw=1.5,
                       label=f'Null — {n_shuffles} rolls (mean ± SE)'),
            ], fontsize=9, loc='upper right', frameon=True,
               framealpha=0.9, edgecolor='black')

        # ══ Right panel: trial-number rows ═══════════════════════════
        ax = axes[ld_i, 1]

        for t in unique_trials:
            mask = trial_id == t
            if mask.any():
                ybase = trial_y[t]
                colour = trial_cmap(t_norm_map[t])
                ax.scatter(proj[mask], ybase + jitter[mask] * 0.35,
                           color=colour, s=14, alpha=0.4, zorder=2)

        # Null centroids (grey)
        for ti2, t in enumerate(unique_trials):
            cx = null_trial_mean[ld_i, ti2]
            se = null_trial_se[ld_i, ti2]
            cy = trial_y[t]
            ax.errorbar(cx, cy + 0.15, xerr=se, fmt='none',
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        ecolor='0.55', zorder=8)
            ax.vlines(cx, cy, cy + 0.3,
                      colors='0.55', linewidth=1.5, zorder=8)

        # Real centroids (per-trial colour, black outline)
        for t in unique_trials:
            mask = trial_id == t
            if mask.sum() > 0:
                cx  = proj[mask].mean()
                sem = proj[mask].std() / np.sqrt(mask.sum())
                cy  = trial_y[t]
                colour = trial_cmap(t_norm_map[t])
                ax.errorbar(cx, cy - 0.15, xerr=sem, fmt='none',
                            capsize=5, capthick=2, elinewidth=2,
                            ecolor=colour, zorder=10)
                ax.vlines(cx, cy - 0.4, cy + 0.1,
                          colors=colour, linewidth=2, zorder=10)

        ax.set_xlabel(f'LD {ld_i + 1}')
        ax.set_yticks(list(trial_y.values()))
        ax.set_yticklabels([str(t) for t in unique_trials], fontsize=8)
        for v in trial_y.values():
            ax.axhline(v, color='k', lw=0.3, alpha=0.15)
        ax.set_ylabel('Trial number')
        ax.set_title(f'LD {ld_i + 1} — by trial number')

        # Colorbar for trial axis (only on first row to save space)
        if ld_i == 0:
            sm = cm.ScalarMappable(
                cmap='viridis',
                norm=plt.Normalize(vmin=unique_trials.min(),
                                   vmax=unique_trials.max())
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
            cbar.set_label('Trial number', fontsize=9)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def run_conjunction_lda_analysis(data_dic, mouse_recday, valid_sessions,
                                  neuron_subset=None,
                                  min_trials=MIN_TRIALS_DEFAULT,
                                  variance_thresh=PCA_VARIANCE_THRESH,
                                  n_logo_shuffles=1000):
    """
    Full pipeline: build conjunction dataset → PCA → LDA on joint
    (state × trial) labels → plot LD scatter.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional
    min_trials : int
        Sessions with fewer trials are excluded.
    variance_thresh : float
        PCA cumulative variance threshold.
    n_logo_shuffles : int
        Not used in this function but reserved for run_conjunction_decoding.

    Returns
    -------
    results : dict
        'X'                : (n_samples, n_neurons) z-scored matrix
        'X_pca'            : (n_samples, n_pcs) PCA-reduced
        'X_conj_ld'        : (n_samples, n_lds) LDA-projected (conjunction)
        'y_state'          : (n_samples,) state labels
        'y_trial'          : (n_samples,) trial indices (0-based, within session)
        'y_conjunction'    : (n_samples,) conjunction labels e.g. 'A_03'
        'progress'         : (n_samples,) 'early'/'middle'/'late'
        'sess_id'          : (n_samples,) session indices
        'trial_id'         : (n_samples,) trial indices
        'pca'              : fitted PCA object
        'lda_conj'         : fitted LDA object (conjunction classes)
        'zero_var_mask'    : (n_neurons,) bool
        'filtered_sessions': list of sessions that passed the trial threshold
    """
    print(f"\n{'='*70}")
    print(f"Conjunction LDA analysis (state × trial): {mouse_recday}")
    print(f"Sessions: {valid_sessions}  |  min_trials={min_trials}")
    print(f"{'='*70}")

    # 1. Build dataset
    try:
        (X, y_state, y_trial, y_conjunction, progress,
         sess_id, trial_id, zero_var_mask, _) = build_conjunction_dataset(
            data_dic, mouse_recday, valid_sessions,
            neuron_subset=neuron_subset, min_trials=min_trials
        )
    except ValueError as e:
        print(f"  SKIP {mouse_recday}: {e}")
        return None

    # 2. Check all four states are present
    present = set(y_state)
    missing = set(EVENT_LABELS) - present
    if missing:
        raise RuntimeError(
            f"States {sorted(missing)} absent from dataset — "
            f"cannot fit conjunction LDA."
        )

    # 3. Check enough conjunction classes for LDA (need >= 2)
    n_classes = len(np.unique(y_conjunction))
    print(f"  {n_classes} conjunction classes, {X.shape[0]} samples")
    if n_classes < 2:
        raise RuntimeError(
            f"Only {n_classes} conjunction class — cannot fit LDA."
        )

    # 4. PCA
    X_pca, pca, _ = apply_pca_trialbins(X, variance_thresh)

    # 5. LDA on conjunction labels
    lda_conj = LinearDiscriminantAnalysis()
    lda_conj.fit(X_pca, y_conjunction)
    X_conj_ld = lda_conj.transform(X_pca)

    n_lds = X_conj_ld.shape[1]
    print(f"  LDA: {n_lds} discriminant dimensions "
          f"(explained variance ratios: "
          f"{lda_conj.explained_variance_ratio_[:min(5, n_lds)].round(3)}...)")

    # 6. Visualise
    filtered_sessions = filter_sessions_by_trials(
        data_dic, mouse_recday, valid_sessions, min_trials=min_trials
    )
    plot_conjunction_ld_scatter(
        X_conj_ld, y_state, trial_id, sess_id, mouse_recday
    )
    plot_conjunction_ld_axes(
        X_conj_ld, y_state, trial_id, mouse_recday
    )

    return {
        'X'                : X,
        'X_pca'            : X_pca,
        'X_conj_ld'        : X_conj_ld,
        'y_state'          : y_state,
        'y_trial'          : y_trial,
        'y_conjunction'    : y_conjunction,
        'progress'         : progress,
        'sess_id'          : sess_id,
        'trial_id'         : trial_id,
        'pca'              : pca,
        'lda_conj'         : lda_conj,
        'zero_var_mask'    : zero_var_mask,
        'filtered_sessions': filtered_sessions,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _collect_logo_predictions(X, y, groups):
    """
    Run one LOGO CV pass and return concatenated true and predicted labels.
    Folds where training has < 2 classes or LDA fails are skipped.
    """
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        clf = LinearDiscriminantAnalysis()
        try:
            clf.fit(X[train_idx], y_tr)
            y_pred = clf.predict(X[test_idx])
            y_true_all.extend(y_te)
            y_pred_all.extend(y_pred)
        except Exception:
            continue
    return np.array(y_true_all), np.array(y_pred_all)


def _plot_confusion_matrix(y_true, y_pred, labels, title):
    """
    Plot a row-normalised confusion matrix (recall per class).
    Per-cell text is shown only when there are <= 20 classes.
    """
    from sklearn.metrics import confusion_matrix

    cm_raw  = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_raw, row_sums, where=row_sums > 0,
                         out=np.zeros_like(cm_raw, dtype=float))

    n = len(labels)
    fig_size = max(5, n * 0.35)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='Recall (row-normalised)')

    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                colour = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=max(5, 9 - n // 4), color=colour)

    tick_labels = [str(l) for l in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels,
                       rotation=90 if n > 10 else 45,
                       fontsize=max(5, 9 - n // 6), ha='right')
    ax.set_yticklabels(tick_labels, fontsize=max(5, 9 - n // 6))
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def run_conjunction_decoding(results_dict, decode_target='state',
                              mouse_recday=None, n_shuffles=1000):
    """
    Leave-one-session-out cross-validated decoding of state or trial number
    from PCA features (avoids leakage from the conjunction LDA).

    Parameters
    ----------
    results_dict : dict
        Output of run_conjunction_lda_analysis.
    decode_target : str
        'state'  — 4-class LDA decoding of state (A/B/C/D).
                   Delegates to run_logo_state_cv.
        'trial'  — n-class LDA decoding of within-session trial index.
    mouse_recday : str, optional
        Used in plot titles.  Defaults to results_dict label if None.
    n_shuffles : int
        Number of label permutations for the null distribution.

    Returns
    -------
    real_acc : float
    null_accs : np.ndarray  (n_shuffles,)
    p_value : float
    """
    X_pca   = results_dict['X_pca']
    sess_id = results_dict['sess_id']
    label   = mouse_recday or 'unknown'

    if decode_target == 'state':
        y_state = results_dict['y_state']
        print(f"\n{'='*70}")
        print(f"State decoding (LOGO CV) — {label}")
        print(f"{'='*70}")
        real_acc, null_accs, p_value = run_logo_state_cv(
            X_pca, y_state, sess_id,
            EVENT_LABELS, f"{label} [state decoding]",
            n_shuffles=n_shuffles
        )
        y_true_cm, y_pred_cm = _collect_logo_predictions(X_pca, y_state, sess_id)
        _plot_confusion_matrix(
            y_true_cm, y_pred_cm, EVENT_LABELS,
            title=f'Confusion matrix — state decoding — {label}'
        )

    elif decode_target == 'trial':
        y_trial = results_dict['y_trial']
        unique_trials = np.unique(y_trial)
        n_trial_classes = len(unique_trials)

        print(f"\n{'='*70}")
        print(f"Trial-number decoding (LOGO CV) — {label}")
        print(f"{n_trial_classes} trial classes, "
              f"{len(np.unique(sess_id))} sessions")
        print(f"{'='*70}")

        logo = LeaveOneGroupOut()

        def _cv_acc(X, y, groups):
            accs = []
            for train_idx, test_idx in logo.split(X, y, groups):
                y_train, y_test = y[train_idx], y[test_idx]
                # Skip fold if train or test has only one class
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 1:
                    continue
                clf = LinearDiscriminantAnalysis()
                try:
                    clf.fit(X[train_idx], y_train)
                    y_pred = clf.predict(X[test_idx])
                    accs.append(balanced_accuracy_score(y_test, y_pred))
                except Exception:
                    continue
            return np.mean(accs) if accs else float('nan')

        real_acc = _cv_acc(X_pca, y_trial, sess_id)

        rng = np.random.default_rng(0)
        null_accs = np.array([
            _cv_acc(X_pca, rng.permutation(y_trial), sess_id)
            for _ in range(n_shuffles)
        ])
        null_accs = null_accs[~np.isnan(null_accs)]

        p_value = np.mean(null_accs >= real_acc) if len(null_accs) else float('nan')
        chance  = 1.0 / n_trial_classes

        print(f"Real CV balanced accuracy: {real_acc:.3f}")
        if len(null_accs):
            print(f"Null: {null_accs.mean():.3f} ± {null_accs.std():.3f}")
        print(f"p = {p_value:.4f}  (chance = {chance:.3f})")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(null_accs, bins=30, color='0.7', edgecolor='white',
                label='Shuffled')
        ax.axvline(real_acc, color='red', lw=2.5,
                   label=f'Real ({real_acc:.2%})')
        ax.axvline(chance, color='k', ls='--', lw=1, alpha=0.5,
                   label=f'Chance ({chance:.2%})')
        ax.set_xlabel('LOGO CV Balanced Accuracy')
        ax.set_ylabel('Count')
        ax.set_title(
            f'Trial-number decoding \u2014 {label}\n'
            f'p = {p_value:.4f}  ({n_trial_classes} trial classes)'
        )
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

        y_true_cm, y_pred_cm = _collect_logo_predictions(X_pca, y_trial, sess_id)
        _plot_confusion_matrix(
            y_true_cm, y_pred_cm, unique_trials.tolist(),
            title=f'Confusion matrix — trial decoding — {label}'
        )

    else:
        raise ValueError(
            f"decode_target must be 'state' or 'trial', got '{decode_target}'."
        )

    return real_acc, null_accs, p_value


# ─────────────────────────────────────────────────────────────────────────────
def plot_decoding_summary(decoding_results, alpha=0.05):
    """
    Summarise decoding results across recdays with two pie charts.

    Parameters
    ----------
    decoding_results : dict
        Nested dict of the form:
          { mouse_recday: { 'state': {'real_acc', 'null_accs', 'p_value'},
                            'trial': {'real_acc', 'null_accs', 'p_value'} } }
        Recdays that were skipped (None results) should simply be absent.
    alpha : float
        Significance threshold (default 0.05).
    """
    targets = ['state', 'trial']
    titles  = ['State decoding', 'Trial-number decoding']

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle(
        f'Conjunction LDA decoding — fraction of significant recdays '
        f'(p < {alpha})',
        fontsize=13, fontweight='bold'
    )

    for ax, target, title in zip(axes, targets, titles):
        p_vals = [
            rec[target]['p_value']
            for rec in decoding_results.values()
            if target in rec and rec[target]['p_value'] is not None
                and not np.isnan(rec[target]['p_value'])
        ]
        n_total = len(p_vals)
        n_sig   = sum(p < alpha for p in p_vals)
        n_ns    = n_total - n_sig

        if n_total == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            ax.axis('off')
            continue

        sizes  = [n_sig, n_ns]
        labels = [f'Significant\n(n={n_sig})', f'Not significant\n(n={n_ns})']
        colours = ['#d62728', '#aec7e8']

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colours,
            autopct='%1.0f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.5),
            textprops=dict(fontsize=11),
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight('bold')

        ax.set_title(f'{title}\n(n={n_total} recdays)', fontsize=12,
                     fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print summary table
    print(f"\n{'Recday':<35} {'State p':>9} {'Trial p':>9}")
    print('-' * 55)
    for recday, rec in decoding_results.items():
        sp = rec.get('state', {}).get('p_value', float('nan'))
        tp = rec.get('trial', {}).get('p_value', float('nan'))
        sig_s = '*' if (not np.isnan(sp) and sp < alpha) else ' '
        sig_t = '*' if (not np.isnan(tp) and tp < alpha) else ' '
        print(f"{recday:<35} {sp:>8.4f}{sig_s} {tp:>8.4f}{sig_t}")
