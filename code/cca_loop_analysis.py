"""
Cross-session CCA analysis of the A→B→C→D→A loop structure.

For each pair of sessions within a recday, Canonical Correlation Analysis
(CCA) finds the shared temporal dimensions of the trial-averaged population
trajectory. If the top shared dimensions exhibit consistent loop geometry
(A→B→C→D→A), that is evidence for a stable cyclic neural representation
across sessions.

Data representation
-------------------
Per session: Neurons_norm (n_neurons, n_trials, 360) is averaged across trials
→ (n_neurons, 360) mean trajectory. After transposing to (360, n_neurons) and
PCA-reducing to (360, k) (k ≤ max_pcs=30), each time bin is a point in
k-dimensional space.

CCA finds directions in (360-bin) space that maximise correlation between a
pair of sessions.  Null distribution: circular-roll one session's trajectory
along the time axis (preserves within-session autocorrelation but destroys
cross-session alignment).

Loop geometry in the shared CCA space is quantified with the cyclicity score
from cyclic_structure_analysis (lower = more cyclic).

Main entry point
----------------
results = run_recday_cca(
    data_dic, mouse_recday, valid_sessions,
    n_cca_components=5, n_shuffles=1000,
)

Reuses
------
  compute_cyclicity_score  from cyclic_structure_analysis
  EVENT_LABELS, EVENT_COLOURS  from lda_state_analysis_trialbins
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from lda_state_analysis_trialbins import EVENT_LABELS, EVENT_COLOURS
from cyclic_structure_analysis import compute_cyclicity_score

BINS_PER_STATE  = 90
N_STATES        = 4
TOTAL_BINS      = 360          # 4 × 90
MAX_PCS_DEFAULT = 30
PCA_THRESH      = 0.75
N_CCA_COMPONENTS = 5
N_SHUFFLES       = 1000

# Bin indices marking the start of each state in the 360-bin layout
STATE_STARTS = {lab: i * BINS_PER_STATE for i, lab in enumerate(EVENT_LABELS)}


# ─────────────────────────────────────────────────────────────────────────────
def prepare_session_trajectory(neurons_norm, pca_thresh=PCA_THRESH,
                                max_pcs=MAX_PCS_DEFAULT):
    """
    Build a PCA-reduced mean trajectory from Neurons_norm.

    Parameters
    ----------
    neurons_norm : np.ndarray  (n_neurons, n_trials, 360)
    pca_thresh : float   cumulative variance to retain
    max_pcs : int        hard cap on PCA components (keeps CCA well-determined)

    Returns
    -------
    traj : np.ndarray  (360, k)
    pca  : fitted PCA object
    n_pcs : int
    """
    # Average across trials; replace NaN with 0
    mean_act = np.nanmean(neurons_norm, axis=1)          # (n_neurons, 360)
    mean_act = np.nan_to_num(mean_act, nan=0.0)
    X = mean_act.T                                        # (360, n_neurons)

    n_comp_max = min(max_pcs, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp_max)
    X_pca_full = pca.fit_transform(X)                    # (360, n_comp_max)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pcs  = int(np.searchsorted(cumvar, pca_thresh)) + 1
    n_pcs  = min(n_pcs, n_comp_max)

    return X_pca_full[:, :n_pcs], pca, n_pcs


# ─────────────────────────────────────────────────────────────────────────────
def run_pairwise_cca(traj_i, traj_j, n_components=N_CCA_COMPONENTS):
    """
    CCA between two (360, k) session trajectories.

    Parameters
    ----------
    traj_i, traj_j : np.ndarray  (360, k_i/k_j)
    n_components : int

    Returns
    -------
    corrs  : np.ndarray  (n_comp,)   canonical correlations
    X_c    : np.ndarray  (360, n_comp)  canonical variates for session i
    Y_c    : np.ndarray  (360, n_comp)  canonical variates for session j
    cca    : fitted CCA object
    """
    n_comp = min(n_components, traj_i.shape[1], traj_j.shape[1],
                 traj_i.shape[0] - 1)
    cca = CCA(n_components=n_comp, max_iter=2000, tol=1e-6)
    X_c, Y_c = cca.fit_transform(traj_i, traj_j)

    corrs = np.array([
        np.corrcoef(X_c[:, d], Y_c[:, d])[0, 1]
        for d in range(n_comp)
    ])
    return corrs, X_c, Y_c, cca


# ─────────────────────────────────────────────────────────────────────────────
def compute_cca_null(traj_i, traj_j, n_components=N_CCA_COMPONENTS,
                     n_shuffles=N_SHUFFLES, seed=42):
    """
    Circular-permutation null: roll traj_j along the time axis.

    Rolling by a random offset preserves within-session autocorrelation but
    destroys the cross-session temporal alignment.

    Parameters
    ----------
    traj_i, traj_j : np.ndarray  (360, k)
    n_components : int
    n_shuffles : int
    seed : int

    Returns
    -------
    null_corrs : np.ndarray  (n_shuffles, n_components)
    """
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, TOTAL_BINS, size=n_shuffles)

    n_comp = min(n_components, traj_i.shape[1], traj_j.shape[1],
                 traj_i.shape[0] - 1)
    null_corrs = np.full((n_shuffles, n_comp), np.nan)

    for si, offset in enumerate(offsets):
        traj_j_rolled = np.roll(traj_j, offset, axis=0)
        try:
            corrs, _, _, _ = run_pairwise_cca(traj_i, traj_j_rolled,
                                               n_components=n_comp)
            null_corrs[si] = corrs
        except Exception:
            pass

    return null_corrs


# ─────────────────────────────────────────────────────────────────────────────
def _cyclicity_from_canonical(canonical_variate):
    """
    Compute cyclicity score from a (360, n_comp) canonical variate array.

    Extracts per-state mean vectors from the top 2 canonical dimensions,
    then calls compute_cyclicity_score.

    Returns
    -------
    score : float   (lower = more cyclic)
    details : dict
    """
    n_comp = min(2, canonical_variate.shape[1])
    proj   = canonical_variate[:, :n_comp]   # (360, ≤2)

    state_vectors = {}
    for lab in EVENT_LABELS:
        b0 = STATE_STARTS[lab]
        b1 = b0 + BINS_PER_STATE
        state_vectors[lab] = proj[b0:b1].mean(axis=0)   # (n_comp,)

    score, details = compute_cyclicity_score(state_vectors, return_details=True)
    return score, details


# ─────────────────────────────────────────────────────────────────────────────
def plot_cca_loop(X_c, Y_c, sess_i_label, sess_j_label, mouse_recday,
                  score_i=None, score_j=None):
    """
    Visualise both sessions' canonical trajectories in shared CCA space.

    Left panel  — trajectory colored by task state (A/B/C/D).
    Right panel — trajectory colored by time bin (viridis), showing loop
                  direction.

    Both sessions are shown: session i = solid line, session j = dashed.

    Parameters
    ----------
    X_c, Y_c : np.ndarray  (360, n_comp)  canonical variates
    sess_i_label, sess_j_label : str
    mouse_recday : str
    score_i, score_j : float or None   cyclicity scores to annotate
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"CCA loop — {mouse_recday}  |  sessions {sess_i_label} vs {sess_j_label}",
        fontsize=13, fontweight='bold'
    )

    bin_cmap  = cm.get_cmap('viridis')
    bin_norm  = plt.Normalize(0, TOTAL_BINS - 1)

    for ax_idx, (ax, title) in enumerate(zip(
            axes, ['Coloured by state', 'Coloured by time bin'])):

        for traj, ls, label in [(X_c, '-', sess_i_label),
                                 (Y_c, '--', sess_j_label)]:
            x = traj[:, 0]
            y = traj[:, 1] if traj.shape[1] > 1 else np.zeros(TOTAL_BINS)

            if ax_idx == 0:
                # Color by state
                for state_i, lab in enumerate(EVENT_LABELS):
                    b0 = STATE_STARTS[lab]
                    b1 = b0 + BINS_PER_STATE
                    ax.plot(x[b0:b1], y[b0:b1],
                            color=EVENT_COLOURS[lab], lw=1.8,
                            linestyle=ls, alpha=0.8,
                            label=f'{lab} ({label})' if ls == '-' else None)
                    # State centroid marker
                    if ls == '-':
                        cx, cy = x[b0:b1].mean(), y[b0:b1].mean()
                        ax.scatter(cx, cy, color=EVENT_COLOURS[lab],
                                   s=120, zorder=5, edgecolors='black',
                                   linewidths=0.8)
                        ax.text(cx, cy, f'  {lab}', fontsize=10,
                                fontweight='bold', color=EVENT_COLOURS[lab],
                                va='center')
            else:
                # Color by time bin
                for b in range(TOTAL_BINS - 1):
                    ax.plot(x[b:b+2], y[b:b+2],
                            color=bin_cmap(bin_norm(b)),
                            lw=1.5, linestyle=ls, alpha=0.7)

        ax.set_xlabel('CCA dim 1', fontsize=11)
        ax.set_ylabel('CCA dim 2', fontsize=11)
        ax.set_title(title, fontsize=11)

        if ax_idx == 0:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color='black', lw=2, ls='-',
                       label=f'Session {sess_i_label}'),
                Line2D([0], [0], color='black', lw=2, ls='--',
                       label=f'Session {sess_j_label}'),
            ] + [
                Line2D([0], [0], color=EVENT_COLOURS[lab], lw=3, label=lab)
                for lab in EVENT_LABELS
            ]
            ax.legend(handles=handles, fontsize=8, loc='best')
        else:
            sm = cm.ScalarMappable(cmap='viridis', norm=bin_norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03,
                         label='Time bin (0–359)')

    # Annotate cyclicity scores
    annots = []
    if score_i is not None:
        annots.append(f'Cyclicity  sess {sess_i_label}: {score_i:.2f}')
    if score_j is not None:
        annots.append(f'Cyclicity  sess {sess_j_label}: {score_j:.2f}')
    if annots:
        fig.text(0.5, -0.02, '   |   '.join(annots),
                 ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def run_recday_cca(data_dic, mouse_recday, valid_sessions,
                   n_cca_components=N_CCA_COMPONENTS,
                   n_shuffles=N_SHUFFLES,
                   pca_thresh=PCA_THRESH,
                   max_pcs=MAX_PCS_DEFAULT,
                   neuron_subset=None):
    """
    Run pairwise CCA for all session pairs in a recday.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    n_cca_components : int
    n_shuffles : int
    pca_thresh : float
    max_pcs : int
    neuron_subset : array-like of int, optional

    Returns
    -------
    results : dict or None (if fewer than 2 usable sessions)
        'session_pairs'   : list of (sidx_i, sidx_j) tuples
        'corrs'           : np.ndarray (n_pairs, n_comp)
        'null_corrs'      : np.ndarray (n_pairs, n_shuffles, n_comp)
        'p_values'        : np.ndarray (n_pairs, n_comp)
        'cyclicity_scores': np.ndarray (n_pairs, 2)  [sess_i, sess_j]
        'mouse_recday'    : str
        'n_sessions'      : int
    """
    print(f"\n{'='*70}")
    print(f"CCA loop analysis: {mouse_recday}")
    print(f"{'='*70}")

    # ── Prepare per-session trajectories ─────────────────────────────
    trajectories = {}
    for sidx in valid_sessions:
        sess = data_dic[mouse_recday][sidx]
        if 'Neurons_norm' not in sess or sess['num_trials'] < 2:
            print(f"  Session {sidx}: skipped (no Neurons_norm or < 2 trials)")
            continue
        nn = sess['Neurons_norm']
        if neuron_subset is not None:
            nn = nn[np.asarray(neuron_subset), :, :]
        try:
            traj, pca_obj, n_pcs = prepare_session_trajectory(
                nn, pca_thresh=pca_thresh, max_pcs=max_pcs
            )
            trajectories[sidx] = traj
            print(f"  Session {sidx}: {sess['num_trials']} trials, "
                  f"{n_pcs} PCs retained")
        except Exception as e:
            print(f"  Session {sidx}: skipped ({e})")

    if len(trajectories) < 2:
        print(f"  SKIP {mouse_recday}: fewer than 2 usable sessions.")
        return None

    # ── Pairwise CCA ─────────────────────────────────────────────────
    pairs     = list(combinations(sorted(trajectories.keys()), 2))
    n_pairs   = len(pairs)
    n_comp    = n_cca_components

    all_corrs    = []
    all_nulls    = []
    all_pvals    = []
    all_cyclicity = []

    for pair_idx, (si, sj) in enumerate(pairs):
        print(f"\n  Pair ({si}, {sj})  [{pair_idx+1}/{n_pairs}]")
        ti, tj = trajectories[si], trajectories[sj]

        corrs, X_c, Y_c, _ = run_pairwise_cca(ti, tj, n_components=n_comp)
        n_actual = len(corrs)
        print(f"    Canonical correlations: {corrs.round(3)}")

        print(f"    Computing null ({n_shuffles} circular rolls)…")
        null_corrs = compute_cca_null(ti, tj,
                                       n_components=n_actual,
                                       n_shuffles=n_shuffles)
        # p-value: fraction of null >= real
        p_vals = np.mean(null_corrs >= corrs[np.newaxis, :], axis=0)
        print(f"    p-values: {p_vals.round(4)}")

        # Cyclicity scores in canonical space
        score_i, _ = _cyclicity_from_canonical(X_c)
        score_j, _ = _cyclicity_from_canonical(Y_c)
        print(f"    Cyclicity scores: sess {si}={score_i:.2f}, "
              f"sess {sj}={score_j:.2f}  (lower = more cyclic)")

        # Pad to n_comp if CCA returned fewer components
        if n_actual < n_comp:
            pad = n_comp - n_actual
            corrs      = np.concatenate([corrs,      np.full(pad, np.nan)])
            null_corrs = np.concatenate([null_corrs,
                                         np.full((n_shuffles, pad), np.nan)],
                                        axis=1)
            p_vals     = np.concatenate([p_vals,     np.full(pad, np.nan)])

        all_corrs.append(corrs)
        all_nulls.append(null_corrs)
        all_pvals.append(p_vals)
        all_cyclicity.append([score_i, score_j])

        plot_cca_loop(X_c, Y_c, str(si), str(sj), mouse_recday,
                      score_i=score_i, score_j=score_j)

    results = {
        'session_pairs'    : pairs,
        'corrs'            : np.array(all_corrs),           # (n_pairs, n_comp)
        'null_corrs'       : np.array(all_nulls),           # (n_pairs, n_shuffles, n_comp)
        'p_values'         : np.array(all_pvals),           # (n_pairs, n_comp)
        'cyclicity_scores' : np.array(all_cyclicity),       # (n_pairs, 2)
        'mouse_recday'     : mouse_recday,
        'n_sessions'       : len(trajectories),
    }

    # Brief summary
    print(f"\n  Summary — {mouse_recday}")
    print(f"  {n_pairs} session pairs, {len(trajectories)} sessions")
    for comp_i in range(n_comp):
        real_c = results['corrs'][:, comp_i]
        real_c = real_c[~np.isnan(real_c)]
        if len(real_c):
            p_c = results['p_values'][:, comp_i]
            p_c = p_c[~np.isnan(p_c)]
            n_sig = int((p_c < 0.05).sum())
            print(f"  CCA dim {comp_i+1}: mean r={real_c.mean():.3f}  "
                  f"significant pairs={n_sig}/{len(real_c)}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
def plot_cca_summary(all_results, n_components_show=3, alpha=0.05):
    """
    Aggregate summary across all recdays.

    Figure 1 — Violin plots of canonical correlations vs null, one subplot
               per canonical dimension.
    Figure 2 — Bar chart of fraction of session pairs with p < alpha per
               canonical component.

    Parameters
    ----------
    all_results : dict  { mouse_recday: results_dict }
    n_components_show : int
    alpha : float   significance threshold
    """
    # Collect real and null correlations across all pairs and recdays
    real_by_comp = [[] for _ in range(n_components_show)]
    null_by_comp = [[] for _ in range(n_components_show)]
    pval_by_comp = [[] for _ in range(n_components_show)]

    for res in all_results.values():
        corrs  = res['corrs']           # (n_pairs, n_comp)
        nulls  = res['null_corrs']      # (n_pairs, n_shuffles, n_comp)
        pvals  = res['p_values']        # (n_pairs, n_comp)

        for ci in range(min(n_components_show, corrs.shape[1])):
            r = corrs[:, ci]
            real_by_comp[ci].extend(r[~np.isnan(r)].tolist())

            n_flat = nulls[:, :, ci].ravel()
            null_by_comp[ci].extend(n_flat[~np.isnan(n_flat)].tolist())

            p = pvals[:, ci]
            pval_by_comp[ci].extend(p[~np.isnan(p)].tolist())

    # ── Figure 1: violin plots ────────────────────────────────────────
    fig1, axes = plt.subplots(1, n_components_show,
                              figsize=(4.5 * n_components_show, 5),
                              sharey=True)
    if n_components_show == 1:
        axes = [axes]

    fig1.suptitle('CCA canonical correlations vs circular-roll null\n'
                  '(all session pairs, all recdays)',
                  fontsize=13, fontweight='bold')

    for ci, ax in enumerate(axes):
        real_vals = np.array(real_by_comp[ci])
        null_vals = np.array(null_by_comp[ci])

        parts_null = ax.violinplot([null_vals], positions=[0],
                                   showmedians=True, widths=0.6)
        parts_real = ax.violinplot([real_vals], positions=[1],
                                   showmedians=True, widths=0.6)

        for pc in parts_null['bodies']:
            pc.set_facecolor('0.75')
            pc.set_alpha(0.7)
        for pc in parts_real['bodies']:
            pc.set_facecolor('#2166ac')
            pc.set_alpha(0.8)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Null', 'Real'], fontsize=11)
        ax.set_title(f'CCA dim {ci + 1}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Canonical correlation' if ci == 0 else '')
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)

        # Annotate n
        ax.text(0.98, 0.02, f'n={len(real_vals)} pairs',
                transform=ax.transAxes, fontsize=9,
                ha='right', va='bottom', color='0.4')

    plt.tight_layout()
    plt.show()

    # ── Figure 2: fraction significant ───────────────────────────────
    frac_sig = []
    n_pairs_total = []
    for ci in range(n_components_show):
        p = np.array(pval_by_comp[ci])
        frac_sig.append(np.mean(p < alpha) if len(p) else np.nan)
        n_pairs_total.append(len(p))

    fig2, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(n_components_show)
    bars = ax.bar(x, frac_sig, color='#2166ac', edgecolor='white',
                  linewidth=1.2, alpha=0.85)
    ax.axhline(alpha, color='red', ls='--', lw=1.5,
               label=f'α = {alpha}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'CCA dim {i+1}' for i in range(n_components_show)],
                       fontsize=11)
    ax.set_ylabel(f'Fraction of pairs with p < {alpha}', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title('Fraction of significant session pairs\n(all recdays)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    for bar, n in zip(bars, n_pairs_total):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'n={n}', ha='center', fontsize=9, color='0.3')
    plt.tight_layout()
    plt.show()
