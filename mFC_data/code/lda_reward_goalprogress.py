"""
LDA on the joint (reward_number × goal_progress) label.

Data source: Neurons_norm (already spatially normalised), shape
  (n_neurons, n_trials, 360)  where  360 = 4 states × 90 bins.

Each (trial, state, progress_bin) yields one feature vector: the mean of 30
consecutive bins within that state's 90-bin window:
  early  → bins  0–29  within the state
  middle → bins 30–59
  late   → bins 60–89

State identity is NOT part of the class label — the four states simply
contribute 4 independent samples to each (reward_number, goal_progress) class.

Conjunction class label: "{reward:02d}_{progress}"  e.g. "07_early"

The hypothesis is that:
  LD1 captures reward number  (analogue of time in session)
  LD2 captures goal progress  (early / middle / late within a state)

Session requirements
--------------------
Only sessions with >= min_trials trials are used, and a recday must have at
least min_sessions (default 3) qualifying sessions.  Every session is
truncated to exactly min_trials trials so conjunction classes are balanced.

Two analyses
------------
1. Aggregate qualifying sessions → visualise via
   plot_reward_progress_ld_scatter  (2-D scatter LD1 vs LD2)
   plot_reward_progress_ld_axes     (per-axis 1-D strip plots, up to max_lds)
2. Decoding (reward number or goal progress) with leave-one-session-out CV
   via run_reward_progress_decoding.
   Summary across recdays via plot_decoding_summary.

Main entry point
----------------
results = run_reward_progress_lda_analysis(
    data_dic, mouse_recday, valid_sessions,
    neuron_subset=None, min_trials=10,
)

Reuses
------
  apply_pca_trialbins, EVENT_LABELS, PCA_VARIANCE_THRESH
  from lda_state_analysis_trialbins

  run_logo_state_cv
  from lda_state_separation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score
from matplotlib.lines import Line2D

from lda_state_analysis_trialbins import (
    apply_pca_trialbins,
    EVENT_LABELS,
    PCA_VARIANCE_THRESH,
)
from lda_state_separation import run_logo_state_cv

PROGRESS_BINS      = ['early', 'middle', 'late']
BINS_PER_STATE     = 90
BINS_PER_PROGRESS  = 30   # 90 / 3
N_STATES           = 4
MIN_TRIALS_DEFAULT = 7
MIN_SESSIONS       = 3

PROGRESS_COLOURS = {'early': '#1b7837', 'middle': '#762a83', 'late': '#e08214'}


# ─────────────────────────────────────────────────────────────────────────────
def filter_sessions_by_trials(data_dic, mouse_recday, valid_sessions,
                               min_trials=MIN_TRIALS_DEFAULT,
                               min_sessions=MIN_SESSIONS):
    """
    Return sessions with >= min_trials trials.  Raises ValueError if fewer
    than min_sessions sessions qualify.
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
def extract_reward_progress_vectors(neurons_norm, sess_index,
                                     n_trials_use):
    """
    Extract one averaged population vector per (trial, state, progress_bin)
    from Neurons_norm.

    Parameters
    ----------
    neurons_norm : np.ndarray  (n_neurons, n_trials, 360)
    sess_index : int
    n_trials_use : int
        Only the first n_trials_use trials are extracted.

    Returns
    -------
    X_sess : np.ndarray  (n_valid, n_neurons)
    y_reward_sess : np.ndarray of int  (n_valid,)   0-based reward index (trial_idx * 4 + state_i)
    y_progress_sess : np.ndarray of str  (n_valid,)  'early'/'middle'/'late'
    y_conjunction_sess : np.ndarray of str  (n_valid,)  e.g. '07_early'
    sess_id_sess : np.ndarray of int  (n_valid,)
    trial_id_sess : np.ndarray of int  (n_valid,)
    skipped : int  number of NaN vectors dropped
    """
    n_neurons = neurons_norm.shape[0]
    n_trials  = min(n_trials_use, neurons_norm.shape[1])

    X_list, y_rew, y_prog, y_conj, sid, tid = [], [], [], [], [], []
    skipped = 0

    for trial_idx in range(n_trials):
        for state_i in range(N_STATES):
            reward_num = trial_idx * N_STATES + state_i
            for prog_i, prog_label in enumerate(PROGRESS_BINS):
                b_start = state_i * BINS_PER_STATE + prog_i * BINS_PER_PROGRESS
                b_end   = b_start + BINS_PER_PROGRESS
                vec = neurons_norm[:, trial_idx, b_start:b_end].mean(axis=1)

                if np.any(np.isnan(vec)):
                    skipped += 1
                    continue

                X_list.append(vec)
                y_rew.append(reward_num)
                y_prog.append(prog_label)
                y_conj.append(f"{reward_num:02d}_{prog_label}")
                sid.append(sess_index)
                tid.append(trial_idx)

    if len(X_list) == 0:
        return (np.empty((0, n_neurons)),
                np.empty(0, dtype=int),
                np.empty(0, dtype=object),
                np.empty(0, dtype=object),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                skipped)

    return (np.vstack(X_list),
            np.array(y_rew, dtype=int),
            np.array(y_prog, dtype=object),
            np.array(y_conj, dtype=object),
            np.array(sid, dtype=int),
            np.array(tid, dtype=int),
            skipped)


# ─────────────────────────────────────────────────────────────────────────────
def build_reward_progress_dataset(data_dic, mouse_recday, valid_sessions,
                                   neuron_subset=None,
                                   min_trials=MIN_TRIALS_DEFAULT):
    """
    Build the (reward × progress) z-scored feature matrix across sessions.

    Neurons_norm is used as the data source (already spatially normalised).
    Feature vectors are z-scored per neuron across the concatenated dataset
    for cross-session comparability.

    Returns
    -------
    X : np.ndarray  (n_samples, n_neurons)
    y_reward : np.ndarray of int  (n_samples,)
    y_progress : np.ndarray of str  (n_samples,)
    y_conjunction : np.ndarray of str  (n_samples,)
    sess_id : np.ndarray of int  (n_samples,)
    trial_id : np.ndarray of int  (n_samples,)
    zero_var_mask : np.ndarray of bool  (n_neurons,)
    n_skipped : int
    filtered_sessions : list
    """
    filtered_sessions = filter_sessions_by_trials(
        data_dic, mouse_recday, valid_sessions, min_trials=min_trials
    )

    X_list, y_rew_list, y_prog_list, y_conj_list = [], [], [], []
    sid_list, tid_list = [], []
    total_skipped = 0

    for sess_i, sidx in enumerate(filtered_sessions):
        nn = data_dic[mouse_recday][sidx]['Neurons_norm']   # (n_neurons, n_trials, 360)
        if neuron_subset is not None:
            nn = nn[np.asarray(neuron_subset), :, :]

        n_trials_sess = data_dic[mouse_recday][sidx]['num_trials']
        print(f"  Session {sidx} (index {sess_i}): "
              f"{n_trials_sess} trials total, using first {min_trials}")

        X_s, y_rew_s, y_prog_s, y_conj_s, sid_s, tid_s, sk = \
            extract_reward_progress_vectors(nn, sess_i, min_trials)

        total_skipped += sk
        if X_s.shape[0] == 0:
            continue

        X_list.append(X_s)
        y_rew_list.append(y_rew_s)
        y_prog_list.append(y_prog_s)
        y_conj_list.append(y_conj_s)
        sid_list.append(sid_s)
        tid_list.append(tid_s)

    if not X_list:
        raise RuntimeError(
            f"No valid samples extracted for {mouse_recday}."
        )

    X            = np.vstack(X_list)
    y_reward     = np.concatenate(y_rew_list)
    y_progress   = np.concatenate(y_prog_list).astype(str)
    y_conjunction = np.concatenate(y_conj_list).astype(str)
    sess_id      = np.concatenate(sid_list)
    trial_id     = np.concatenate(tid_list)

    # Z-score per neuron across the full concatenated dataset
    mu  = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    zero_var_mask = (sig.squeeze() == 0)
    sig_safe = np.where(sig == 0, 1.0, sig)
    X = (X - mu) / sig_safe
    X[:, zero_var_mask] = 0.0

    n_zero = int(zero_var_mask.sum())
    if n_zero:
        print(f"  WARNING: {n_zero} neuron(s) with zero variance set to 0.")

    print(f"  Dataset: {X.shape[0]} samples ({total_skipped} skipped), "
          f"{len(np.unique(y_conjunction))} conjunction classes")

    return (X, y_reward, y_progress, y_conjunction,
            sess_id, trial_id, zero_var_mask, total_skipped,
            filtered_sessions)


# ─────────────────────────────────────────────────────────────────────────────
def plot_reward_progress_ld_scatter(X_rp_ld, y_reward, y_progress,
                                     mouse_recday, max_lds=3, seed=42):
    """
    Two-panel scatter of LD1 vs LD2 (and optionally LD1 vs LD3).

    Left  — points coloured by goal-progress bin (early/middle/late).
    Right — points coloured by reward number (viridis).
    """
    rng = np.random.default_rng(seed)
    n_ld = min(X_rp_ld.shape[1], max_lds)

    ld_pairs = [(0, 1)]
    if n_ld >= 3:
        ld_pairs.append((0, 2))

    n_pairs = len(ld_pairs)
    fig, axes = plt.subplots(n_pairs, 2,
                             figsize=(12, 5.5 * n_pairs),
                             squeeze=False)

    fig.suptitle(
        f"Reward \u00d7 goal-progress LDA \u2014 {mouse_recday}",
        fontsize=15, fontweight='bold', y=1.01
    )

    unique_rewards = np.sort(np.unique(y_reward))
    r_min, r_max   = unique_rewards.min(), unique_rewards.max()
    r_norm         = (y_reward - r_min) / max(r_max - r_min, 1)

    for row_i, (xi, yi) in enumerate(ld_pairs):
        ax_prog   = axes[row_i, 0]
        ax_reward = axes[row_i, 1]

        proj_x = X_rp_ld[:, xi]
        proj_y = X_rp_ld[:, yi]
        jx = rng.uniform(-0.02, 0.02, size=len(proj_x)) * (proj_x.max() - proj_x.min())
        jy = rng.uniform(-0.02, 0.02, size=len(proj_y)) * (proj_y.max() - proj_y.min())

        # ── Left: colour by progress ──────────────────────────────────
        for prog in PROGRESS_BINS:
            mask = y_progress == prog
            if mask.any():
                ax_prog.scatter(proj_x[mask] + jx[mask],
                                proj_y[mask] + jy[mask],
                                c=PROGRESS_COLOURS[prog],
                                s=18, alpha=0.5, label=prog, zorder=2)

        for prog in PROGRESS_BINS:
            mask = y_progress == prog
            if mask.sum() > 0:
                cx, cy = proj_x[mask].mean(), proj_y[mask].mean()
                ax_prog.scatter(cx, cy, c=PROGRESS_COLOURS[prog],
                                s=180, marker='*', edgecolors='black',
                                linewidths=0.8, zorder=5)
                ax_prog.text(cx, cy, f'  {prog}', fontsize=9,
                             fontweight='bold', color=PROGRESS_COLOURS[prog],
                             va='center', zorder=6)

        ax_prog.set_xlabel(f'LD {xi + 1}')
        ax_prog.set_ylabel(f'LD {yi + 1}')
        ax_prog.set_title(f'LD{xi+1} vs LD{yi+1} — coloured by goal progress')
        ax_prog.legend(title='Progress', fontsize=9, markerscale=1.4)

        # ── Right: colour by reward number ────────────────────────────
        sc = ax_reward.scatter(proj_x + jx, proj_y + jy,
                                c=r_norm, cmap='viridis',
                                s=18, alpha=0.5, zorder=2)
        cbar = fig.colorbar(sc, ax=ax_reward, pad=0.02)
        cbar.set_label('Reward number (trial)', fontsize=9)
        tick_vals = np.linspace(0, 1, 5)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(
            [f"{int(round(v * (r_max - r_min) + r_min))}" for v in tick_vals]
        )

        reward_cmap = cm.get_cmap('viridis')
        for r in unique_rewards:
            mask = y_reward == r
            cx, cy = proj_x[mask].mean(), proj_y[mask].mean()
            colour = reward_cmap((r - r_min) / max(r_max - r_min, 1))
            ax_reward.scatter(cx, cy, color=colour,
                              s=60, marker='+', linewidths=1.5,
                              alpha=0.7, zorder=4)

        ax_reward.set_xlabel(f'LD {xi + 1}')
        ax_reward.set_ylabel(f'LD {yi + 1}')
        ax_reward.set_title(f'LD{xi+1} vs LD{yi+1} — coloured by reward number')

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def plot_reward_progress_ld_axes(X_rp_ld, y_reward, y_progress,
                                  mouse_recday, max_lds=10,
                                  n_shuffles=1000, seed=42):
    """
    Per-axis 1-D strip plots (up to max_lds).

    Left panel  — progress rows (early / middle / late), coloured by progress.
    Right panel — reward-number rows, coloured by reward (viridis).

    Real centroids (coloured ± SEM) and circular-roll null centroids
    (grey ± SE) are overlaid on both panels.
    """
    n_ld      = min(X_rp_ld.shape[1], max_lds)
    n_samples = len(y_reward)
    rng       = np.random.default_rng(seed)

    unique_rewards = np.sort(np.unique(y_reward))
    n_rewards      = len(unique_rewards)
    reward_cmap    = cm.get_cmap('viridis')
    r_norm_map     = {r: i / max(n_rewards - 1, 1)
                      for i, r in enumerate(unique_rewards)}

    prog_y   = {p: i for i, p in enumerate(PROGRESS_BINS)}
    reward_y = {r: i for i, r in enumerate(unique_rewards)}

    jitter = rng.uniform(-0.3, 0.3, size=n_samples)

    # ── Precompute null centroids ─────────────────────────────────────
    roll_offsets = rng.integers(1, n_samples, size=n_shuffles)

    null_prog   = np.zeros((n_shuffles, n_ld, len(PROGRESS_BINS)))
    null_reward = np.zeros((n_shuffles, n_ld, n_rewards))

    print(f"Computing null centroids ({n_shuffles} circular rolls)…")
    for si, offset in enumerate(roll_offsets):
        yp_roll = np.roll(y_progress, offset)
        yr_roll = np.roll(y_reward,   offset)
        for li in range(n_ld):
            proj = X_rp_ld[:, li]
            for pi, prog in enumerate(PROGRESS_BINS):
                m = yp_roll == prog
                if m.sum() > 0:
                    null_prog[si, li, pi] = proj[m].mean()
            for ri, r in enumerate(unique_rewards):
                m = yr_roll == r
                if m.sum() > 0:
                    null_reward[si, li, ri] = proj[m].mean()

    null_prog_mean   = null_prog.mean(axis=0)
    null_prog_se     = null_prog.std(axis=0)   / np.sqrt(n_shuffles)
    null_reward_mean = null_reward.mean(axis=0)
    null_reward_se   = null_reward.std(axis=0) / np.sqrt(n_shuffles)

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_ld, 2,
                             figsize=(18, 2.8 * n_ld),
                             squeeze=False)
    fig.suptitle(
        f"Reward \u00d7 goal-progress LDA \u2014 per-axis \u2014 {mouse_recday}",
        fontsize=15, fontweight='bold', y=1.01
    )

    for ld_i in range(n_ld):
        proj = X_rp_ld[:, ld_i]

        # ══ Left: progress rows ═══════════════════════════════════════
        ax = axes[ld_i, 0]
        for prog in PROGRESS_BINS:
            mask = y_progress == prog
            if mask.any():
                ax.scatter(proj[mask], prog_y[prog] + jitter[mask] * 0.35,
                           c=PROGRESS_COLOURS[prog], s=14, alpha=0.4, zorder=2)

        for pi, prog in enumerate(PROGRESS_BINS):
            cx = null_prog_mean[ld_i, pi]
            se = null_prog_se[ld_i, pi]
            cy = prog_y[prog]
            ax.errorbar(cx, cy + 0.15, xerr=se, fmt='none',
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        ecolor='0.55', zorder=8)
            ax.vlines(cx, cy, cy + 0.3, colors='0.55', linewidth=1.5, zorder=8)

        for prog in PROGRESS_BINS:
            mask = y_progress == prog
            if mask.sum() > 0:
                cx  = proj[mask].mean()
                sem = proj[mask].std() / np.sqrt(mask.sum())
                cy  = prog_y[prog]
                ax.errorbar(cx, cy - 0.15, xerr=sem, fmt='none',
                            capsize=5, capthick=2, elinewidth=2,
                            ecolor=PROGRESS_COLOURS[prog], zorder=10)
                ax.vlines(cx, cy - 0.4, cy + 0.1,
                          colors=PROGRESS_COLOURS[prog], linewidth=2, zorder=10)

        ax.set_xlabel(f'LD {ld_i + 1}')
        ax.set_yticks(list(prog_y.values()))
        ax.set_yticklabels(PROGRESS_BINS, fontweight='bold')
        for v in prog_y.values():
            ax.axhline(v, color='k', lw=0.3, alpha=0.15)
        ax.set_title(f'LD {ld_i + 1} — by goal progress')

        if ld_i == 0:
            ax.legend(handles=[
                Line2D([0], [0], color='black', lw=2,
                       label='Real (mean ± SEM)'),
                Line2D([0], [0], color='0.55', lw=1.5,
                       label=f'Null — {n_shuffles} rolls (mean ± SE)'),
            ], fontsize=9, loc='upper right', frameon=True,
               framealpha=0.9, edgecolor='black')

        # ══ Right: reward-number rows ══════════════════════════════════
        ax = axes[ld_i, 1]
        for r in unique_rewards:
            mask = y_reward == r
            if mask.any():
                colour = reward_cmap(r_norm_map[r])
                ax.scatter(proj[mask], reward_y[r] + jitter[mask] * 0.35,
                           color=colour, s=14, alpha=0.4, zorder=2)

        for ri, r in enumerate(unique_rewards):
            cx = null_reward_mean[ld_i, ri]
            se = null_reward_se[ld_i, ri]
            cy = reward_y[r]
            ax.errorbar(cx, cy + 0.15, xerr=se, fmt='none',
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        ecolor='0.55', zorder=8)
            ax.vlines(cx, cy, cy + 0.3, colors='0.55', linewidth=1.5, zorder=8)

        for r in unique_rewards:
            mask = y_reward == r
            if mask.sum() > 0:
                cx     = proj[mask].mean()
                sem    = proj[mask].std() / np.sqrt(mask.sum())
                cy     = reward_y[r]
                colour = reward_cmap(r_norm_map[r])
                ax.errorbar(cx, cy - 0.15, xerr=sem, fmt='none',
                            capsize=5, capthick=2, elinewidth=2,
                            ecolor=colour, zorder=10)
                ax.vlines(cx, cy - 0.4, cy + 0.1,
                          colors=colour, linewidth=2, zorder=10)

        ax.set_xlabel(f'LD {ld_i + 1}')
        ax.set_yticks(list(reward_y.values()))
        ax.set_yticklabels([str(r) for r in unique_rewards], fontsize=8)
        for v in reward_y.values():
            ax.axhline(v, color='k', lw=0.3, alpha=0.15)
        ax.set_ylabel('Reward number')
        ax.set_title(f'LD {ld_i + 1} — by reward number')

        if ld_i == 0:
            sm = cm.ScalarMappable(
                cmap='viridis',
                norm=plt.Normalize(vmin=unique_rewards.min(),
                                   vmax=unique_rewards.max())
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
            cbar.set_label('Reward number', fontsize=9)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def run_reward_progress_lda_analysis(data_dic, mouse_recday, valid_sessions,
                                      neuron_subset=None,
                                      min_trials=MIN_TRIALS_DEFAULT,
                                      variance_thresh=PCA_VARIANCE_THRESH):
    """
    Full pipeline: build reward × goal-progress dataset → PCA → LDA →
    visualise scatter and per-axis plots.

    Returns
    -------
    results : dict  or  None if the recday fails the session threshold.
        'X'                : (n_samples, n_neurons)
        'X_pca'            : (n_samples, n_pcs)
        'X_rp_ld'          : (n_samples, n_lds)
        'y_reward'         : (n_samples,) int
        'y_progress'       : (n_samples,) str
        'y_conjunction'    : (n_samples,) str  e.g. '07_early'
        'sess_id'          : (n_samples,) int
        'trial_id'         : (n_samples,) int
        'pca'              : fitted PCA object
        'lda_rp'           : fitted LDA object
        'zero_var_mask'    : (n_neurons,) bool
        'filtered_sessions': list
    """
    print(f"\n{'='*70}")
    print(f"Reward × goal-progress LDA: {mouse_recday}")
    print(f"Sessions: {valid_sessions}  |  min_trials={min_trials}")
    print(f"{'='*70}")

    try:
        (X, y_reward, y_progress, y_conjunction,
         sess_id, trial_id, zero_var_mask, _,
         filtered_sessions) = build_reward_progress_dataset(
            data_dic, mouse_recday, valid_sessions,
            neuron_subset=neuron_subset, min_trials=min_trials
        )
    except ValueError as e:
        print(f"  SKIP {mouse_recday}: {e}")
        return None

    n_classes = len(np.unique(y_conjunction))
    print(f"  {n_classes} conjunction classes, {X.shape[0]} samples")
    if n_classes < 2:
        print(f"  SKIP: only {n_classes} class.")
        return None

    # PCA
    X_pca, pca, _ = apply_pca_trialbins(X, variance_thresh)

    # LDA on conjunction labels
    lda_rp = LinearDiscriminantAnalysis()
    lda_rp.fit(X_pca, y_conjunction)
    X_rp_ld = lda_rp.transform(X_pca)

    n_lds = X_rp_ld.shape[1]
    print(f"  LDA: {n_lds} discriminant dimensions "
          f"(top-5 explained variance: "
          f"{lda_rp.explained_variance_ratio_[:min(5, n_lds)].round(3)})")

    plot_reward_progress_ld_scatter(X_rp_ld, y_reward, y_progress, mouse_recday)
    plot_reward_progress_ld_axes(X_rp_ld, y_reward, y_progress, mouse_recday)

    return {
        'X'                : X,
        'X_pca'            : X_pca,
        'X_rp_ld'          : X_rp_ld,
        'y_reward'         : y_reward,
        'y_progress'       : y_progress,
        'y_conjunction'    : y_conjunction,
        'sess_id'          : sess_id,
        'trial_id'         : trial_id,
        'pca'              : pca,
        'lda_rp'           : lda_rp,
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

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    # Row-normalise: each row sums to 1 (recall per true class)
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
def run_reward_progress_decoding(results_dict, decode_target='progress',
                                  mouse_recday=None, n_shuffles=1000):
    """
    Leave-one-session-out CV decoding from PCA features.

    Parameters
    ----------
    decode_target : str
        'progress' — 3-class decoding of goal-progress (early/middle/late).
        'reward'   — n-class decoding of reward number.
    mouse_recday : str, optional
    n_shuffles : int

    Returns
    -------
    real_acc : float
    null_accs : np.ndarray
    p_value : float
    """
    X_pca   = results_dict['X_pca']
    sess_id = results_dict['sess_id']
    label   = mouse_recday or 'unknown'

    if decode_target == 'progress':
        y      = results_dict['y_progress']
        labels = PROGRESS_BINS
        title  = f'Goal-progress decoding — {label}'
        chance = 1.0 / len(PROGRESS_BINS)
    elif decode_target == 'reward':
        y      = results_dict['y_reward']
        labels = np.unique(y).tolist()
        title  = f'Reward-number decoding — {label}'
        chance = 1.0 / len(labels)
    else:
        raise ValueError(
            f"decode_target must be 'progress' or 'reward', got '{decode_target}'."
        )

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    logo = LeaveOneGroupOut()

    def _cv_acc(X, y_lbl, groups):
        accs = []
        for train_idx, test_idx in logo.split(X, y_lbl, groups):
            y_tr, y_te = y_lbl[train_idx], y_lbl[test_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            clf = LinearDiscriminantAnalysis()
            try:
                clf.fit(X[train_idx], y_tr)
                y_pred = clf.predict(X[test_idx])
                accs.append(balanced_accuracy_score(y_te, y_pred))
            except Exception:
                continue
        return np.mean(accs) if accs else float('nan')

    real_acc = _cv_acc(X_pca, y, sess_id)

    rng = np.random.default_rng(0)
    null_accs = np.array([
        _cv_acc(X_pca, rng.permutation(y), sess_id)
        for _ in range(n_shuffles)
    ])
    null_accs = null_accs[~np.isnan(null_accs)]

    p_value = np.mean(null_accs >= real_acc) if len(null_accs) else float('nan')

    print(f"Real CV balanced accuracy: {real_acc:.3f}")
    if len(null_accs):
        print(f"Null: {null_accs.mean():.3f} ± {null_accs.std():.3f}")
    print(f"p = {p_value:.4f}  (chance = {chance:.3f})")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_accs, bins=30, color='0.7', edgecolor='white', label='Shuffled')
    ax.axvline(real_acc, color='red', lw=2.5, label=f'Real ({real_acc:.2%})')
    ax.axvline(chance, color='k', ls='--', lw=1, alpha=0.5,
               label=f'Chance ({chance:.2%})')
    ax.set_xlabel('LOGO CV Balanced Accuracy')
    ax.set_ylabel('Count')
    ax.set_title(f'{title}\np = {p_value:.4f}  ({len(labels)} classes)')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # Confusion matrix
    y_true_cm, y_pred_cm = _collect_logo_predictions(X_pca, y, sess_id)
    _plot_confusion_matrix(
        y_true_cm, y_pred_cm, labels,
        title=f'Confusion matrix — {title}'
    )

    return real_acc, null_accs, p_value


# ─────────────────────────────────────────────────────────────────────────────
def plot_decoding_summary(decoding_results, alpha=0.05):
    """
    Two pie charts: fraction of recdays with significant decoding for
    goal-progress and reward-number.

    Parameters
    ----------
    decoding_results : dict
        { mouse_recday: { 'progress': {'real_acc', 'null_accs', 'p_value'},
                          'reward':   {'real_acc', 'null_accs', 'p_value'} } }
    alpha : float
    """
    targets = ['progress', 'reward']
    titles  = ['Goal-progress decoding', 'Reward-number decoding']

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle(
        f'Reward \u00d7 goal-progress LDA — fraction significant (p < {alpha})',
        fontsize=13, fontweight='bold'
    )

    for ax, target, title in zip(axes, targets, titles):
        p_vals = [
            rec[target]['p_value']
            for rec in decoding_results.values()
            if target in rec
            and rec[target]['p_value'] is not None
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

        wedges, texts, autotexts = ax.pie(
            [n_sig, n_ns],
            labels=[f'Significant\n(n={n_sig})', f'Not significant\n(n={n_ns})'],
            colors=['#d62728', '#aec7e8'],
            autopct='%1.0f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.5),
            textprops=dict(fontsize=11),
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight('bold')

        ax.set_title(f'{title}\n(n={n_total} recdays)',
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"\n{'Recday':<35} {'Progress p':>11} {'Reward p':>10}")
    print('-' * 58)
    for recday, rec in decoding_results.items():
        pp = rec.get('progress', {}).get('p_value', float('nan'))
        rp = rec.get('reward',   {}).get('p_value', float('nan'))
        sig_p = '*' if (not np.isnan(pp) and pp < alpha) else ' '
        sig_r = '*' if (not np.isnan(rp) and rp < alpha) else ' '
        print(f"{recday:<35} {pp:>10.4f}{sig_p} {rp:>9.4f}{sig_r}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix_from_results(results_dict, decode_target='progress',
                                        mouse_recday=None):
    """
    Plot a LOGO CV confusion matrix from a saved results dict without
    re-running the full analysis or shuffles.

    Parameters
    ----------
    results_dict : dict
        Output of run_reward_progress_lda_analysis (must contain X_pca,
        y_progress / y_reward, sess_id).
    decode_target : str
        'progress' or 'reward'.
    mouse_recday : str, optional
    """
    X_pca   = results_dict['X_pca']
    sess_id = results_dict['sess_id']
    label   = mouse_recday or 'unknown'

    if decode_target == 'progress':
        y      = results_dict['y_progress']
        labels = PROGRESS_BINS
    elif decode_target == 'reward':
        y      = results_dict['y_reward']
        labels = np.unique(y).tolist()
    else:
        raise ValueError(
            f"decode_target must be 'progress' or 'reward', got '{decode_target}'."
        )

    y_true, y_pred = _collect_logo_predictions(X_pca, y, sess_id)
    _plot_confusion_matrix(
        y_true, y_pred, labels,
        title=f'Confusion matrix — {decode_target} decoding — {label}'
    )


# ─────────────────────────────────────────────────────────────────────────────
def plot_aggregate_confusion_matrix(results_by_recday, decode_target='progress'):
    """
    Concatenate LOGO CV predictions across all recdays and plot a single
    aggregate confusion matrix.

    Parameters
    ----------
    results_by_recday : dict
        { mouse_recday: results_dict }  from run_reward_progress_lda_analysis.
    decode_target : str
        'progress' or 'reward'.
    """
    if decode_target == 'progress':
        labels = PROGRESS_BINS
    elif decode_target == 'reward':
        # Infer from first available results dict
        first = next(iter(results_by_recday.values()))
        labels = np.unique(first['y_reward']).tolist()
    else:
        raise ValueError(
            f"decode_target must be 'progress' or 'reward', got '{decode_target}'."
        )

    all_true, all_pred = [], []
    for recday, res in results_by_recday.items():
        X_pca   = res['X_pca']
        sess_id = res['sess_id']
        y = res['y_progress'] if decode_target == 'progress' else res['y_reward']
        y_true, y_pred = _collect_logo_predictions(X_pca, y, sess_id)
        all_true.append(y_true)
        all_pred.append(y_pred)
        print(f"  {recday}: {len(y_true)} predictions collected")

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    n_recdays = len(results_by_recday)
    _plot_confusion_matrix(
        y_true_all, y_pred_all, labels,
        title=f'Aggregate confusion matrix — {decode_target} decoding\n'
              f'(n={n_recdays} recdays)'
    )
