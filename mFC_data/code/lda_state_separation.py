"""
LDA for separating task states (A, B, C, D) in neural population data.
Includes leave-one-session-out cross-validation for state separability testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score


def run_lda_state_separation(X_fine_pca_full, last_event, EVENT_LABELS, EVENT_COLOURS, mouse_recday):
    """
    Run LDA to separate task states A, B, C, D.
    
    Parameters
    ----------
    X_fine_pca_full : np.ndarray
        PCA-reduced data, shape (n_samples, n_pcs)
    last_event : np.ndarray
        Array of event labels ('A', 'B', 'C', 'D', or '')
    EVENT_LABELS : list
        List of event labels, e.g., ['A', 'B', 'C', 'D']
    EVENT_COLOURS : dict
        Dict mapping event labels to colors
    mouse_recday : str
        Mouse and recording day identifier for plot titles
        
    Returns
    -------
    lda_state : LinearDiscriminantAnalysis
        Fitted LDA model
    X_state_ld : np.ndarray
        LDA-transformed data
    y_state : np.ndarray
        State labels for samples with events
    """
    # Only use fine bins that have an assigned event state
    has_event = last_event != ''
    X_state = X_fine_pca_full[has_event]  # Use PCA-reduced data
    y_state = last_event[has_event]

    # Fit LDA to separate states A, B, C, D
    lda_state = LinearDiscriminantAnalysis()
    lda_state.fit(X_state, y_state)

    # Project all data (including those with events)
    X_state_ld = lda_state.transform(X_state)

    print(f"LDA fitted to separate task states A, B, C, D")
    print(f"Number of samples: {len(y_state)}")
    print(f"Number of LDA components: {X_state_ld.shape[1]}")
    print(f"Explained variance ratio: {lda_state.explained_variance_ratio_}")

    # Compute per-state centroids and define adjacency cycle
    centroids = {label: X_state_ld[y_state == label].mean(axis=0) for label in EVENT_LABELS}
    cycle = EVENT_LABELS + [EVENT_LABELS[0]]  # A->B->C->D->A

    n_ld = min(X_state_ld.shape[1], 3)

    def _plot_centroids_2d(ax, xi, yi, xlabel, ylabel, title):
        for label in EVENT_LABELS:
            c = centroids[label]
            ax.scatter(c[xi], c[yi], c=EVENT_COLOURS[label], s=150, zorder=5)
            ax.text(c[xi], c[yi], f'  {label}', fontsize=12, fontweight='bold',
                    color=EVENT_COLOURS[label], va='center')
        for i in range(len(cycle) - 1):
            src, dst = centroids[cycle[i]], centroids[cycle[i + 1]]
            ax.annotate('', xy=(dst[xi], dst[yi]), xytext=(src[xi], src[yi]),
                        arrowprops=dict(arrowstyle='->', color='k', lw=1.5))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    if n_ld >= 2:
        n_panels = 1 if n_ld == 2 else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        _plot_centroids_2d(axes[0], 0, 1, 'LD 1', 'LD 2', 'LDA centroids: LD1 vs LD2')
        if n_ld >= 3:
            _plot_centroids_2d(axes[1], 0, 2, 'LD 1', 'LD 3', 'LDA centroids: LD1 vs LD3')
            _plot_centroids_2d(axes[2], 1, 2, 'LD 2', 'LD 3', 'LDA centroids: LD2 vs LD3')

        fig.suptitle(f'LDA state centroids (A\u2192B\u2192C\u2192D\u2192A) \u2014 {mouse_recday}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    # 3D plot if 3 components
    if n_ld >= 3:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        for label in EVENT_LABELS:
            c = centroids[label]
            ax.scatter(c[0], c[1], c[2], c=EVENT_COLOURS[label], s=150, zorder=5)
            ax.text(c[0], c[1], c[2], f'  {label}', fontsize=12, fontweight='bold',
                    color=EVENT_COLOURS[label])

        for i in range(len(cycle) - 1):
            src, dst = centroids[cycle[i]], centroids[cycle[i + 1]]
            ax.quiver(src[0], src[1], src[2],
                      dst[0] - src[0], dst[1] - src[1], dst[2] - src[2],
                      arrow_length_ratio=0.2, color='k', linewidth=1.5)

        ax.set_xlabel('LD 1')
        ax.set_ylabel('LD 2')
        ax.set_zlabel('LD 3')
        ax.set_title(f'3D LDA centroids \u2014 Task States \u2014 {mouse_recday}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return lda_state, X_state_ld, y_state


def run_logo_state_cv(X_state_ld, y_state, sess_id, EVENT_LABELS, mouse_recday, n_shuffles=1000):
    """
    Leave-one-session-out CV for state separability using LDA.

    Parameters
    ----------
    X_state_ld : np.ndarray
        LDA-projected features for bins with an assigned event state, shape (n_samples, n_lds)
    y_state : np.ndarray
        State labels ('A', 'B', 'C', 'D') for each sample
    sess_id : np.ndarray
        Integer session index for each sample (same length as y_state)
    EVENT_LABELS : list
        Ordered list of event labels, e.g. ['A', 'B', 'C', 'D']
    mouse_recday : str
        Identifier used in plot title
    n_shuffles : int
        Number of label permutations for the null distribution

    Returns
    -------
    real_acc : float
        Mean LOGO balanced accuracy on real labels
    null_accs : np.ndarray
        Balanced accuracies under label permutation
    p_value : float
        Fraction of null >= real
    """
    logo = LeaveOneGroupOut()
    n_sessions = len(np.unique(sess_id))
    print(f"Working with sessions {np.unique(sess_id)} (total {n_sessions})")
    def _cv_acc(X, y, groups):
        accs = []
        for train_idx, test_idx in logo.split(X, y, groups):
            clf = LinearDiscriminantAnalysis()
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            # print("="*60)
            # print(f"Test session: {np.unique(groups[test_idx])}")
            # print(f"Predicted: {np.unique(y_pred)}")
            # print(f"Actual: {np.unique(y[test_idx])}")
            accs.append(balanced_accuracy_score(y[test_idx], y_pred))
        return np.mean(accs)

    real_acc = _cv_acc(X_state_ld, y_state, sess_id)

    rng = np.random.default_rng(0)
    null_accs = np.array([
        _cv_acc(X_state_ld, rng.permutation(y_state), sess_id)
        for _ in range(n_shuffles)
    ])

    p_value = np.mean(null_accs >= real_acc)

    print(f"Leave-one-session-out CV ({n_sessions} sessions)")
    print(f"Real CV balanced accuracy: {real_acc:.3f}")
    print(f"Null: {null_accs.mean():.3f} ± {null_accs.std():.3f}")
    print(f"p = {p_value:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_accs, bins=30, color='0.7', edgecolor='white', label='Shuffled')
    ax.axvline(real_acc, color='red', lw=2.5, label=f'Real ({real_acc:.2%})')
    ax.set_xlabel('Leave-one-session-out CV Balanced Accuracy')
    ax.set_ylabel('Count')
    ax.set_title(f'LDA decoding \u2014 {mouse_recday}\n'
                 f'p = {p_value:.4f}, chance = {1/len(EVENT_LABELS):.0%}')
    ax.axvline(1 / len(EVENT_LABELS), color='k', ls='--', lw=1, alpha=0.5, label='Chance')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    return real_acc, null_accs, p_value


if __name__ == "__main__":
    # Example usage:
    # lda_state, X_state_ld, y_state = run_lda_state_separation(
    #     X_fine_pca_full, last_event, EVENT_LABELS, EVENT_COLOURS, mouse_recday
    # )
    # has_event = last_event != ''
    # real_acc, null_accs, p_value = run_logo_state_cv(
    #     X_state_ld, y_state, sess_id_fine[has_event], EVENT_LABELS, mouse_recday
    # )
    pass
