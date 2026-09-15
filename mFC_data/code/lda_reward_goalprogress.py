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
BIN_S = 0.025            # Trial_times are 25 ms bin indices
CONJUNCTIONS = ('trial_progress', 'reward_progress')


def extract_reward_progress_vectors(neurons_norm, sess_index, n_trials_use,
                                    trial_times=None, conjunction='trial_progress'):
    """
    One averaged population vector per (trial, state, progress_bin) from Neurons_norm.

    conjunction : 'trial_progress' -> class label f'{trial:02d}_{prog}', the 4 states of a
                  trial are 4 samples of one class; 'reward_progress' -> the legacy
                  f'{reward:02d}_{prog}' (reward = trial*4 + state), 1 sample per class.
    trial_times : (n_trials, 5) state boundaries in 25 ms bins, or None. Each sample gets its
                  centre time in seconds since the session's first A onset (NaN if None).

    Returns X (n_valid, n_neurons), y_reward, y_progress, y_conjunction, sess_id, trial_id,
    state_id, t_sec, skipped.
    """
    if conjunction not in CONJUNCTIONS:
        raise ValueError(f"conjunction must be one of {CONJUNCTIONS}, got {conjunction!r}")
    n_neurons = neurons_norm.shape[0]
    n_trials  = min(n_trials_use, neurons_norm.shape[1])
    tt = None if trial_times is None else np.asarray(trial_times, dtype=float)

    X_list, y_rew, y_prog, y_conj, sid, tid, stid, tsec = [], [], [], [], [], [], [], []
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

                if tt is not None and trial_idx < tt.shape[0] and state_i + 1 < tt.shape[1]:
                    a, b = tt[trial_idx, state_i], tt[trial_idx, state_i + 1]
                    t = (a + (b - a) * (prog_i + 0.5) / len(PROGRESS_BINS) - tt[0, 0]) * BIN_S
                else:
                    t = np.nan

                key = trial_idx if conjunction == 'trial_progress' else reward_num
                X_list.append(vec)
                y_rew.append(reward_num)
                y_prog.append(prog_label)
                y_conj.append(f"{key:02d}_{prog_label}")
                sid.append(sess_index)
                tid.append(trial_idx)
                stid.append(state_i)
                tsec.append(t)

    if len(X_list) == 0:
        return (np.empty((0, n_neurons)),
                np.empty(0, dtype=int),
                np.empty(0, dtype=object),
                np.empty(0, dtype=object),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                np.empty(0, dtype=float),
                skipped)

    return (np.vstack(X_list),
            np.array(y_rew, dtype=int),
            np.array(y_prog, dtype=object),
            np.array(y_conj, dtype=object),
            np.array(sid, dtype=int),
            np.array(tid, dtype=int),
            np.array(stid, dtype=int),
            np.array(tsec, dtype=float),
            skipped)


# ─────────────────────────────────────────────────────────────────────────────
def build_reward_progress_dataset(data_dic, mouse_recday, valid_sessions,
                                   neuron_subset=None,
                                   min_trials=MIN_TRIALS_DEFAULT,
                                   conjunction='trial_progress'):
    """
    Z-scored (trial x state x progress-bin) feature matrix across sessions, from Neurons_norm.
    Feature vectors are z-scored per neuron across the concatenated dataset.

    Returns X, y_reward, y_progress, y_conjunction, sess_id, trial_id, state_id, t_sec,
    zero_var_mask, n_skipped, filtered_sessions.
    """
    filtered_sessions = filter_sessions_by_trials(
        data_dic, mouse_recday, valid_sessions, min_trials=min_trials
    )

    X_list, y_rew_list, y_prog_list, y_conj_list = [], [], [], []
    sid_list, tid_list, stid_list, tsec_list = [], [], [], []
    total_skipped = 0

    for sess_i, sidx in enumerate(filtered_sessions):
        nn = data_dic[mouse_recday][sidx]['Neurons_norm']   # (n_neurons, n_trials, 360)
        if neuron_subset is not None:
            nn = nn[np.asarray(neuron_subset), :, :]

        n_trials_sess = data_dic[mouse_recday][sidx]['num_trials']
        print(f"  Session {sidx} (index {sess_i}): "
              f"{n_trials_sess} trials total, using first {min_trials}")

        X_s, y_rew_s, y_prog_s, y_conj_s, sid_s, tid_s, stid_s, tsec_s, sk = \
            extract_reward_progress_vectors(
                nn, sess_i, min_trials,
                trial_times=data_dic[mouse_recday][sidx].get('Trial_times'),
                conjunction=conjunction)

        total_skipped += sk
        if X_s.shape[0] == 0:
            continue

        X_list.append(X_s)
        y_rew_list.append(y_rew_s)
        y_prog_list.append(y_prog_s)
        y_conj_list.append(y_conj_s)
        sid_list.append(sid_s)
        tid_list.append(tid_s)
        stid_list.append(stid_s)
        tsec_list.append(tsec_s)

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
    state_id     = np.concatenate(stid_list)
    t_sec        = np.concatenate(tsec_list)

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
          f"{len(np.unique(y_conjunction))} {conjunction} classes")

    return (X, y_reward, y_progress, y_conjunction,
            sess_id, trial_id, state_id, t_sec,
            zero_var_mask, total_skipped, filtered_sessions)


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
    if n_ld < 2:
        print(f"  {mouse_recday}: {n_ld} discriminant dimension -- no LD1 x LD2 scatter.")
        return

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
                                      variance_thresh=PCA_VARIANCE_THRESH,
                                      conjunction='trial_progress',
                                      plot=True, n_pcs=None):
    """
    Dataset -> PCA -> joint LDA on the conjunction label -> LD scatter and per-axis plots.

    Returns None if the recday fails the session / PC gates, else a dict with
    'X' (n_samples, n_neurons), 'X_pca', 'X_rp_ld', 'y_reward', 'y_progress', 'y_conjunction',
    'y_time' (trial index, or reward index for the legacy conjunction), 'sess_id', 'trial_id',
    'state_id', 't_sec' (s since the session's first A onset), 'pca', 'lda_rp', 'zero_var_mask',
    'filtered_sessions', 'conjunction'.
    """
    print(f"\n{'='*70}")
    print(f"{conjunction} LDA: {mouse_recday}")
    print(f"Sessions: {valid_sessions}  |  min_trials={min_trials}")
    print(f"{'='*70}")

    try:
        (X, y_reward, y_progress, y_conjunction,
         sess_id, trial_id, state_id, t_sec,
         zero_var_mask, _, filtered_sessions) = build_reward_progress_dataset(
            data_dic, mouse_recday, valid_sessions,
            neuron_subset=neuron_subset, min_trials=min_trials,
            conjunction=conjunction
        )
    except ValueError as e:
        print(f"  SKIP {mouse_recday}: {e}")
        return None

    n_classes = len(np.unique(y_conjunction))
    print(f"  {n_classes} conjunction classes, {X.shape[0]} samples")
    if n_classes < 2:
        print(f"  SKIP: only {n_classes} class.")
        return None

    # PCA: 75 % variance by default; n_pcs fixes the LDA input dimension across recdays and
    # datasets (LEC recdays reach 75 % at ~29 PCs, PFC at ~16, and the time scores track it)
    X_pca, pca, _ = apply_pca_trialbins(X, variance_thresh)
    if n_pcs is not None:
        k = min(int(n_pcs), pca.n_components_)
        X_pca = pca.transform(X)[:, :k]
        print(f"  PCA: fixed {k} PCs ({np.sum(pca.explained_variance_ratio_[:k]):.1%} variance)"
              + ('' if k == n_pcs else f'  [only {k} available]'))

    # a 2-D LDA needs two PCs; me10_20122021_21122021 (PFC) has one and used to crash here
    if X_pca.shape[1] < 2:
        print(f"  SKIP {mouse_recday}: only {X_pca.shape[1]} PC(s) -- too few for a 2-D LDA.")
        return None

    # LDA on conjunction labels
    lda_rp = LinearDiscriminantAnalysis()
    lda_rp.fit(X_pca, y_conjunction)
    X_rp_ld = lda_rp.transform(X_pca)

    n_lds = X_rp_ld.shape[1]
    print(f"  LDA: {n_lds} discriminant dimensions "
          f"(top-5 explained variance: "
          f"{lda_rp.explained_variance_ratio_[:min(5, n_lds)].round(3)})")

    y_time = trial_id if conjunction == 'trial_progress' else y_reward
    if plot:
        plot_reward_progress_ld_scatter(X_rp_ld, y_time, y_progress, mouse_recday)
        plot_reward_progress_ld_axes(X_rp_ld, y_time, y_progress, mouse_recday)

    return {
        'X'                : X,
        'X_pca'            : X_pca,
        'X_rp_ld'          : X_rp_ld,
        'y_reward'         : y_reward,
        'y_progress'       : y_progress,
        'y_conjunction'    : y_conjunction,
        'y_time'           : y_time,
        'sess_id'          : sess_id,
        'trial_id'         : trial_id,
        'state_id'         : state_id,
        't_sec'            : t_sec,
        'pca'              : pca,
        'lda_rp'           : lda_rp,
        'zero_var_mask'    : zero_var_mask,
        'filtered_sessions': filtered_sessions,
        'conjunction'      : conjunction,
        'n_pcs_mode'       : f'fixed_{n_pcs}' if n_pcs is not None else f'var_{variance_thresh:g}',
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


# ─────────────────────────────────────────────────────────────────────────────
# One cross-validated readout of the joint LDA for both variables (2026-09-14).
def _split_conjunction(labels):
    """'07_early' -> (7, 'early') as arrays."""
    labels = np.asarray(labels).astype(str)
    keys = np.array([int(s.split('_', 1)[0]) for s in labels], dtype=int)
    progs = np.array([s.split('_', 1)[1] for s in labels], dtype=object)
    return keys, progs


def _safe_corr(a, b, kind='pearson'):
    from scipy.stats import pearsonr, spearmanr
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    f = pearsonr if kind == 'pearson' else spearmanr
    return float(f(a[m], b[m])[0])


def run_joint_lda_readout(results_dict, n_shuffles=1000, ridge_alpha=1.0, seed=0, verbose=True):
    """
    Leave-one-session-out readout of the joint conjunction LDA for both variables.

    Per fold: LDA on the training sessions' y_conjunction; on the held-out session `predict`
    gives a (time key, progress) pair per sample and `transform` gives LD scores. Scores:
      key_mae / key_rmse / key_rho / key_acc  -- time key (trial index; reward index for legacy)
      sec_mae / sec_r                          -- seconds, Ridge(alpha) on training LD scores
      prog_acc (balanced) / prog_rho           -- goal progress, early < middle < late
      joint_acc                                -- exact conjunction class (chance 1/n_classes)
      ld1_r_key / ld1_r_sec / ld2_rho          -- |corr| of held-out LD1 with time, LD2 with progress
    Null: the sample order within each session is circularly shifted (labels and t_sec together,
    X fixed) and everything is refit, n_shuffles times. p = fraction of null at least as good
    (<= for *_mae/*_rmse, >= otherwise). Returns the scores, `<score>_null` arrays,
    `<score>_null_mean`, `<score>_p`, and the pooled held-out predictions.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import balanced_accuracy_score

    X = np.asarray(results_dict['X_pca'], dtype=float)
    y_conj = np.asarray(results_dict['y_conjunction']).astype(str)
    sess = np.asarray(results_dict['sess_id'])
    t_sec = np.asarray(results_dict.get('t_sec', np.full(len(sess), np.nan)), dtype=float)
    prog_rank = {p: i for i, p in enumerate(PROGRESS_BINS)}
    n_classes = len(np.unique(y_conj))
    logo = LeaveOneGroupOut()

    def one_pass(labels, t):
        out = {k: [] for k in ('key_t', 'key_p', 'prog_t', 'prog_p', 't_t', 't_p', 'ld1', 'ld2')}
        for tr, te in logo.split(X, labels, sess):
            if len(np.unique(labels[tr])) < 2:
                continue
            clf = LinearDiscriminantAnalysis()
            try:
                clf.fit(X[tr], labels[tr])
            except Exception:
                continue
            k_p, p_p = _split_conjunction(clf.predict(X[te]))
            k_t, p_t = _split_conjunction(labels[te])
            Z_tr, Z_te = clf.transform(X[tr]), clf.transform(X[te])
            ok = np.isfinite(t[tr])
            if ok.sum() >= 10:
                t_p = Ridge(alpha=ridge_alpha).fit(Z_tr[ok], t[tr][ok]).predict(Z_te)
            else:
                t_p = np.full(len(te), np.nan)
            out['key_t'].append(k_t); out['key_p'].append(k_p)
            out['prog_t'].append(p_t); out['prog_p'].append(p_p)
            out['t_t'].append(t[te]); out['t_p'].append(t_p)
            out['ld1'].append(Z_te[:, 0])
            out['ld2'].append(Z_te[:, 1] if Z_te.shape[1] > 1 else np.full(len(te), np.nan))
        return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}

    def score(P):
        if len(P['key_t']) == 0:
            return {k: np.nan for k in ('key_mae', 'key_rmse', 'key_rho', 'key_acc', 'sec_mae', 'sec_r',
                                        'prog_acc', 'prog_rho', 'joint_acc', 'ld1_r_key', 'ld1_r_sec',
                                        'ld2_rho')}
        err = P['key_p'] - P['key_t']
        pr_t = np.array([prog_rank[p] for p in P['prog_t']], dtype=float)
        pr_p = np.array([prog_rank[p] for p in P['prog_p']], dtype=float)
        s = dict(
            key_mae=float(np.mean(np.abs(err))),
            key_rmse=float(np.sqrt(np.mean(err ** 2))),
            key_rho=_safe_corr(P['key_t'], P['key_p'], 'spearman'),
            key_acc=float(np.mean(err == 0)),
            sec_mae=float(np.nanmean(np.abs(P['t_p'] - P['t_t']))) if np.isfinite(P['t_p']).any() else np.nan,
            sec_r=_safe_corr(P['t_t'], P['t_p'], 'pearson'),
            prog_acc=float(balanced_accuracy_score(P['prog_t'].astype(str), P['prog_p'].astype(str))),
            prog_rho=_safe_corr(pr_t, pr_p, 'spearman'),
            joint_acc=float(np.mean((err == 0) & (P['prog_t'] == P['prog_p']))),
            ld1_r_key=abs(_safe_corr(P['ld1'], P['key_t'], 'pearson')),
            ld1_r_sec=abs(_safe_corr(P['ld1'], P['t_t'], 'pearson')),
            ld2_rho=abs(_safe_corr(P['ld2'], pr_t, 'spearman')),
        )
        return s

    real = one_pass(y_conj, t_sec)
    scores = score(real)
    if verbose:
        print(f"  joint LDA readout ({n_classes} classes): key MAE {scores['key_mae']:.2f}, "
              f"sec MAE {scores['sec_mae']:.1f} s, r {scores['sec_r']:.2f}; progress acc "
              f"{scores['prog_acc']:.3f}; joint acc {scores['joint_acc']:.3f}; "
              f"|r| LD1~sec {scores['ld1_r_sec']:.2f}, |rho| LD2~prog {scores['ld2_rho']:.2f}")

    rng = np.random.default_rng(seed)
    nulls = {k: [] for k in scores}
    for _ in range(n_shuffles):
        y_sh, t_sh = y_conj.copy(), t_sec.copy()
        for g in np.unique(sess):
            idx = np.flatnonzero(sess == g)
            if len(idx) < 2:
                continue
            k = int(rng.integers(1, len(idx)))
            y_sh[idx] = np.roll(y_conj[idx], k)
            t_sh[idx] = np.roll(t_sec[idx], k)
        sc = score(one_pass(y_sh, t_sh))
        for key in nulls:
            nulls[key].append(sc[key])

    out = dict(scores)
    for key, vals in nulls.items():
        arr = np.asarray(vals, dtype=float)
        out[f'{key}_null'] = arr
        out[f'{key}_null_mean'] = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
        out[f'{key}_null_sd'] = float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan
        if np.isfinite(scores[key]) and np.isfinite(arr).any():
            better = arr <= scores[key] if key.endswith(('_mae', '_rmse')) else arr >= scores[key]
            out[f'{key}_p'] = float(np.nanmean(better))
        else:
            out[f'{key}_p'] = np.nan
    out.update(y_true_key=real['key_t'], y_pred_key=real['key_p'], y_true_prog=real['prog_t'],
               y_pred_prog=real['prog_p'], t_true=real['t_t'], t_pred=real['t_p'],
               n_classes=n_classes, n_shuffles=n_shuffles, ridge_alpha=ridge_alpha,
               conjunction=results_dict.get('conjunction', 'reward_progress'))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset summary from the pickles written by run_lda_reward_progress.py.
_REPO = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
REWARD_PROGRESS_PICKLES = {
    'lec': f'{_REPO}/data/glm_outputs/LEC_lda/reward_progress.pkl',
    'pfc': f'{_REPO}/mFC_data/glm_outputs/PFC_lda/reward_progress.pkl',
}
DATASET_COLOURS = {'lec': '#BE3455', 'pfc': '#0F4C81'}   # GridMaze Viva Magenta / Classic Blue
GROUP_COLOURS = {'ENTl': '#0F4C81', 'ENTl-deep': '#0F4C81', 'ENTl-sup': '#45B5AA',   # anatomy_split
                 'ENTm': '#6B3FA0', 'SUB': '#BE3455', 'SUBCA1': '#BE3455', 'CA1': '#FF6F61'}
NULL_COLOUR, INK = '#B4B2A9', '#2C2C2A'                  # GridMaze Stone / Caviar


def _colour(label):
    return DATASET_COLOURS.get(label) or GROUP_COLOURS.get(label) or INK


def load_reward_progress_results(dataset, path=None, tag=''):
    """The pickle written by run_lda_reward_progress.py for 'lec' or 'pfc'; `tag` picks a
    variant such as '_pcs15' (reward_progress_pcs15.pkl)."""
    import os
    import pickle
    path = path or REWARD_PROGRESS_PICKLES[dataset].replace('.pkl', f'{tag}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found -- run: sbatch sbatch_files/lda_reward_progress.sbatch {dataset}')
    with open(path, 'rb') as f:
        return pickle.load(f)


READOUT_METRICS = ('key_mae', 'key_rmse', 'key_rho', 'key_acc', 'sec_mae', 'sec_r',
                   'prog_acc', 'prog_rho', 'joint_acc', 'ld1_r_key', 'ld1_r_sec', 'ld2_rho')


def reward_progress_table(payload):
    """One row per recday: n, every readout score with its null mean/sd and p, chance levels.
    Legacy pickles (no 'readout') give the old two-decoder accuracies."""
    import pandas as pd
    rows = []
    if 'readout' in payload:
        for rd, ro in payload['readout'].items():
            res = payload['results_by_recday'][rd]
            row = dict(dataset=payload['dataset'], recday=rd, mouse=rd.split('_')[0],
                       conjunction=ro.get('conjunction'), n_neurons=res['X'].shape[1],
                       n_samples=res['X'].shape[0], n_pcs=res['X_pca'].shape[1],
                       n_lds=res['X_rp_ld'].shape[1], n_classes=ro['n_classes'],
                       n_sessions=len(res['filtered_sessions']))
            for k in READOUT_METRICS:
                row[k] = ro[k]
                row[f'{k}_null_mean'] = ro[f'{k}_null_mean']
                row[f'{k}_null_sd'] = ro[f'{k}_null_sd']
                row[f'{k}_p'] = ro[f'{k}_p']
            row['prog_chance'] = 1.0 / len(PROGRESS_BINS)
            row['joint_chance'] = 1.0 / ro['n_classes']
            # seconds MAE as a fraction of the mean 10-trial span, so datasets with different
            # trial durations are comparable
            t, s = np.asarray(res.get('t_sec', []), dtype=float), np.asarray(res['sess_id'])
            spans = [np.nanmax(t[s == g]) - np.nanmin(t[s == g]) for g in np.unique(s)
                     if len(t) and np.isfinite(t[s == g]).any()]
            row['span_s_mean'] = float(np.mean(spans)) if spans else np.nan
            row['sec_mae_frac'] = ro['sec_mae'] / row['span_s_mean'] if spans else np.nan
            rows.append(row)
        return pd.DataFrame(rows)
    for rd, dec in payload['decoding_results'].items():
        res = payload['results_by_recday'][rd]
        row = dict(dataset=payload['dataset'], recday=rd, mouse=rd.split('_')[0],
                   n_neurons=res['X'].shape[1], n_pcs=res['X_pca'].shape[1],
                   n_lds=res['X_rp_ld'].shape[1], n_sessions=len(res['filtered_sessions']))
        for target in ('progress', 'reward'):
            d = dec[target]
            null = np.asarray(d['null_accs'], dtype=float)
            row[f'{target}_acc'] = d['real_acc']
            row[f'{target}_null_mean'] = float(null.mean()) if len(null) else np.nan
            row[f'{target}_null_sd'] = float(null.std()) if len(null) else np.nan
            row[f'{target}_p'] = d['p_value']
        row['progress_chance'] = 1.0 / len(PROGRESS_BINS)
        row['reward_chance'] = 1.0 / len(np.unique(res['y_reward']))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_draw_readouts(readouts):
    """Combine run_joint_lda_readout outputs from repeated neuron subsamples of one recday:
    scores are averaged over draws, null arrays pooled, p recomputed against the pooled null
    (a single draw's null is wider than the null of a draw-mean, so p is conservative)."""
    out = {}
    for k in READOUT_METRICS:
        vals = np.array([r[k] for r in readouts], dtype=float)
        pooled = np.concatenate([np.asarray(r[f'{k}_null'], dtype=float) for r in readouts])
        real = float(np.nanmean(vals))
        out[k] = real
        out[f'{k}_draws'] = vals
        out[f'{k}_null'] = pooled
        out[f'{k}_null_mean'] = float(np.nanmean(pooled)) if np.isfinite(pooled).any() else np.nan
        out[f'{k}_null_sd'] = float(np.nanstd(pooled)) if np.isfinite(pooled).any() else np.nan
        if np.isfinite(real) and np.isfinite(pooled).any():
            better = pooled <= real if k.endswith(('_mae', '_rmse')) else pooled >= real
            out[f'{k}_p'] = float(np.nanmean(better))
        else:
            out[f'{k}_p'] = np.nan
    last = readouts[-1]
    for k in ('y_true_key', 'y_pred_key', 'y_true_prog', 'y_pred_prog', 't_true', 't_pred'):
        out[k] = last[k]
    out.update(n_classes=last['n_classes'], n_shuffles=last['n_shuffles'] * len(readouts),
               ridge_alpha=last['ridge_alpha'], conjunction=last['conjunction'],
               n_draws=len(readouts))
    return out


def time_behaviour_table(payload):
    """Per session: how trial index maps onto seconds. Pearson / Spearman r(t_sec, trial),
    10-trial span, trial-duration mean, CV and drift (s per trial). The trial-index and
    seconds readouts are comparable across datasets only if these agree."""
    from scipy.stats import linregress, pearsonr, spearmanr
    import pandas as pd
    rows = []
    for rd, res in payload['results_by_recday'].items():
        t = np.asarray(res.get('t_sec', []), dtype=float)
        if not len(t) or not np.isfinite(t).any():
            continue
        tr, st, s = res['trial_id'], res['state_id'], res['sess_id']
        pr = np.asarray(res['y_progress']).astype(str)
        for g in np.unique(s):
            m = s == g
            starts = {}
            for k in np.unique(tr[m]):
                v = t[m & (tr == k) & (st == 0) & (pr == PROGRESS_BINS[0])]
                if len(v):
                    starts[int(k)] = float(v[0])
            ks = sorted(starts)
            dur = np.diff([starts[k] for k in ks])
            rows.append(dict(
                dataset=payload['dataset'], recday=rd, mouse=rd.split('_')[0], session=int(g),
                r_pearson=float(pearsonr(t[m], tr[m])[0]), r_spearman=float(spearmanr(t[m], tr[m])[0]),
                span_s=float(np.nanmax(t[m]) - np.nanmin(t[m])),
                trial_dur_mean=float(dur.mean()) if len(dur) else np.nan,
                trial_dur_cv=float(dur.std() / dur.mean()) if len(dur) else np.nan,
                dur_slope_s_per_trial=float(linregress(ks[:-1], dur).slope) if len(dur) > 2 else np.nan))
    return pd.DataFrame(rows)


def _strip_panel(ax, df, metric, datasets, rng, title, ylabel, chance_col=None, ylim=None,
                 alpha=0.05):
    """Per-recday points (jittered), per-mouse open markers, null band (mean +/- 2 sd), the
    number of recdays with p < alpha above each column."""
    from matplotlib.patches import Rectangle
    for i, ds in enumerate(datasets):
        sub = df[df['dataset'] == ds]
        nm, nsd = sub[f'{metric}_null_mean'].mean(), sub[f'{metric}_null_sd'].mean()
        if np.isfinite(nm) and np.isfinite(nsd):
            ax.add_patch(Rectangle((i - 0.32, nm - 2 * nsd), 0.64, 4 * nsd,
                                   color=NULL_COLOUR, alpha=0.6, lw=0, zorder=1))
        vals = sub[metric].to_numpy(dtype=float)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(sub)), vals, s=8,
                   color=_colour(ds), alpha=0.65, lw=0, zorder=2)
        mm = sub.groupby('mouse')[metric].mean()
        xs = i + np.linspace(-0.12, 0.12, len(mm)) if len(mm) > 1 else np.array([i])
        ax.scatter(xs, mm.values, s=24, facecolor='white', edgecolor=_colour(ds),
                   lw=1.0, zorder=3)
        n_sig = int((sub[f'{metric}_p'] < alpha).sum())
        ax.text(i, 1.02, f'{n_sig}/{len(sub)}', ha='center', va='bottom', fontsize=7,
                color=_colour(ds), transform=ax.get_xaxis_transform())
    if chance_col is not None:
        ax.axhline(df[chance_col].mean(), ls='--', lw=0.8, color=INK, zorder=0)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([d.upper() if d in DATASET_COLOURS else d for d in datasets])
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, None)
    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)


def plot_lec_vs_pfc_decoding(paths=None, out_path=None, alpha=0.05, seed=0, tag=''):
    """Held-out readouts of the joint LDA, LEC against PFC, 2 x 3 panels.

    Row 1: goal-progress accuracy | time MAE in trials | time MAE in seconds.
    Row 2: joint-class accuracy | held-out LD1 |r| vs seconds | held-out LD2 |rho| vs progress.
    Points = recdays, open markers = per-mouse means, grey band = circular-shift null, dashed =
    chance; the number above each column is recdays with p < alpha. A dataset whose pickle is
    missing is skipped. Returns (fig, table).
    """
    import matplotlib as mpl
    import pandas as pd

    # `paths`: {label: pickle path}. Default labels are the two datasets; any other labels (e.g.
    # 'ENTl' / 'SUBCA1' region-restricted runs of one dataset) are plotted as the columns.
    if paths is None:
        paths = {ds: None for ds in ('lec', 'pfc')}
    tables = []
    for label, path in paths.items():
        ds = label if label in DATASET_COLOURS else 'lec'
        try:
            tb = reward_progress_table(load_reward_progress_results(ds, path, tag=tag))
        except FileNotFoundError as exc:
            print(f'  skipping {label}: {exc}')
            continue
        tb['dataset'] = label
        tables.append(tb)
    if not tables:
        raise FileNotFoundError('no reward_progress pickle found for any label')
    df = pd.concat(tables, ignore_index=True)
    if 'prog_acc' not in df:
        raise ValueError('legacy pickle without a joint-LDA readout; re-run run_lda_reward_progress.py')
    datasets = [d for d in paths if d in set(df['dataset'])]

    try:
        from glm_analysis_v2 import apply_gridmaze_style
        apply_gridmaze_style()
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    key_name = 'trials' if (df['conjunction'] == 'trial_progress').all() else 'rewards'
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    panels = [
        ('prog_acc', 'goal progress (3 classes)', 'balanced accuracy', 'prog_chance', (0, 1)),
        ('key_mae', f'time in session: MAE ({key_name})', f'MAE ({key_name})', None, None),
        ('sec_mae', 'time in session: MAE (s)', 'MAE (s)', None, None),
        ('joint_acc', 'joint trial x progress class', 'accuracy', 'joint_chance', None),
        ('ld1_r_sec', 'held-out LD1 vs seconds', '|r|', None, (0, 1)),
        ('ld2_rho', 'held-out LD2 vs progress', '|rho|', None, (0, 1)),
    ]
    for ax, (metric, title, ylabel, chance_col, ylim) in zip(axes.ravel(), panels):
        _strip_panel(ax, df, metric, datasets, rng, title, ylabel, chance_col, ylim, alpha)
    fig.tight_layout()
    if out_path:
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                             'pdf.fonttype': 42, 'ps.fonttype': 42}):
            fig.savefig(out_path, bbox_inches=None)
    return fig, df
