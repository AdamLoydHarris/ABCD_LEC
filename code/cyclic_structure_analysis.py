"""
Cyclic Structure Analysis for Task Space Representations

Analyzes whether population neural responses show cyclic geometric structure
across task states (A, B, C, D). Tests if:
- Diagonal vectors (A→C, B→D) are orthogonal
- Adjacent vectors (A→B, C→D) are parallel

A perfect cyclic structure would have:
- angle(A→C, B→D) = 90°
- angle(A→B, C→D) = 0° (or 180° if antiparallel)
"""

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA


def apply_pca_to_population(neurons_norm, n_components=10, return_pca=False):
    """
    Apply PCA to population activity.
    
    Parameters
    ----------
    neurons_norm : ndarray
        Neural data (num_neurons, num_trials, 360) or (num_neurons, 360)
    n_components : int
        Number of PCA components to keep
    return_pca : bool
        If True, also return the fitted PCA object
        
    Returns
    -------
    pca_activity : ndarray
        Activity in PCA space (n_components, 360) or (n_components, num_trials, 360)
    pca : PCA object (if return_pca=True)
        Fitted PCA object with explained_variance_ratio_ etc.
    """
    # Handle 3D data
    if neurons_norm.ndim == 3:
        num_neurons, num_trials, num_bins = neurons_norm.shape
        # Reshape to (num_neurons, num_trials * num_bins)
        reshaped = neurons_norm.reshape(num_neurons, -1)
    else:
        reshaped = neurons_norm
        num_bins = neurons_norm.shape[1]
    
    # Handle NaN - replace with column means
    col_means = np.nanmean(reshaped, axis=0)
    nan_mask = np.isnan(reshaped)
    reshaped_filled = reshaped.copy()
    reshaped_filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    
    # Fit PCA (neurons are features, time bins are samples)
    # Transpose so samples are rows: (n_samples, n_features) = (time, neurons)
    pca = PCA(n_components=min(n_components, reshaped_filled.shape[0], reshaped_filled.shape[1]))
    pca_scores = pca.fit_transform(reshaped_filled.T).T  # (n_components, n_samples)
    
    # Reshape back if 3D
    if neurons_norm.ndim == 3:
        pca_activity = pca_scores.reshape(pca.n_components_, num_trials, num_bins)
    else:
        pca_activity = pca_scores
    
    if return_pca:
        return pca_activity, pca
    return pca_activity


def get_state_vectors_pca(neurons_norm, n_components=10, bins_per_state=90):
    """
    Get state vectors in PCA space.
    
    Parameters
    ----------
    neurons_norm : ndarray
        Neural data
    n_components : int
        Number of PCA components
    bins_per_state : int
        Bins per state
        
    Returns
    -------
    state_vectors : dict
        State vectors in PCA space
    pca : PCA object
        Fitted PCA for visualization
    explained_variance : float
        Total explained variance by n_components
    """
    pca_activity, pca = apply_pca_to_population(neurons_norm, n_components, return_pca=True)
    
    # Now get state vectors from PCA space
    state_vectors = get_state_population_vectors(pca_activity, bins_per_state=bins_per_state)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    
    return state_vectors, pca, explained_variance


def analyze_cyclicity_pca(neurons_norm, n_components=10, bins_per_state=90):
    """
    Analyze cyclicity in PCA space.
    
    Parameters
    ----------
    neurons_norm : ndarray
        Neural data
    n_components : int
        Number of PCA components
    bins_per_state : int
        Bins per state
        
    Returns
    -------
    score : float
        Cyclicity score
    details : dict
        Detailed results including PCA info
    """
    state_vectors, pca, explained_var = get_state_vectors_pca(
        neurons_norm, n_components, bins_per_state
    )
    
    score, details = compute_cyclicity_score(state_vectors, return_details=True)
    details['pca_components'] = n_components
    details['explained_variance'] = explained_var
    details['explained_variance_per_pc'] = pca.explained_variance_ratio_
    
    return score, details


def plot_pca_state_geometry(neurons_norm, n_components=3, bins_per_state=90, 
                            session_label='', save_path=None, show_individual_bins=False,
                            show_trial_points=False):
    """
    Visualize state geometry in PCA space (2D and 3D).
    
    Parameters
    ----------
    neurons_norm : ndarray
        Neural data
    n_components : int
        Number of PCA components for analysis (default 3)
    bins_per_state : int
        Bins per state
    session_label : str
        Label for plots
    save_path : str, optional
        Path to save figure
    show_individual_bins : bool
        If True, show each time bin as a point (can be noisy)
    show_trial_points : bool
        If True, show one point per trial per state (shows variance)
        
    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Get PCA representation
    pca_activity, pca = apply_pca_to_population(neurons_norm, n_components=n_components, return_pca=True)
    
    # Average across trials if 3D
    if pca_activity.ndim == 3:
        pca_mean = np.nanmean(pca_activity, axis=1)  # (n_components, 360)
    else:
        pca_mean = pca_activity
    
    # Get state centroids
    state_labels = ['A', 'B', 'C', 'D']
    state_colors = ['red', 'blue', 'green', 'orange']
    state_centroids = {}
    
    for i, label in enumerate(state_labels):
        start_bin = i * bins_per_state
        end_bin = (i + 1) * bins_per_state
        state_centroids[label] = np.nanmean(pca_mean[:, start_bin:end_bin], axis=1)
    
    # Compute cyclicity score
    state_vectors = get_state_population_vectors(pca_activity, bins_per_state=bins_per_state)
    score, details = compute_cyclicity_score(state_vectors, return_details=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Compute trial-level state centroids if needed
    trial_state_centroids = None
    if show_trial_points and pca_activity.ndim == 3:
        # pca_activity shape: (n_components, n_trials, 360)
        n_trials = pca_activity.shape[1]
        trial_state_centroids = {label: [] for label in state_labels}
        for i, label in enumerate(state_labels):
            start_bin = i * bins_per_state
            end_bin = (i + 1) * bins_per_state
            # Mean across bins within state, for each trial
            for t in range(n_trials):
                trial_centroid = np.nanmean(pca_activity[:, t, start_bin:end_bin], axis=1)
                if not np.any(np.isnan(trial_centroid)):
                    trial_state_centroids[label].append(trial_centroid)
    
    # 1. 2D projection (PC1 vs PC2)
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Plot trajectory (thin line connecting mean activity)
    ax1.plot(pca_mean[0, :], pca_mean[1, :], 'k-', alpha=0.2, linewidth=0.5)
    
    # Show individual bins if requested (can be noisy)
    if show_individual_bins:
        for i, (label, color) in enumerate(zip(state_labels, state_colors)):
            start_bin = i * bins_per_state
            end_bin = (i + 1) * bins_per_state
            ax1.scatter(pca_mean[0, start_bin:end_bin], pca_mean[1, start_bin:end_bin], 
                       c=color, alpha=0.2, s=5)
    
    # Show trial-level points if requested
    if show_trial_points and trial_state_centroids is not None:
        for label, color in zip(state_labels, state_colors):
            pts = np.array(trial_state_centroids[label])
            if len(pts) > 0:
                ax1.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.3, s=30, 
                           marker='o', edgecolor='none')
    
    # Plot state centroids (always shown, larger markers)
    for label, color in zip(state_labels, state_colors):
        c = state_centroids[label]
        ax1.scatter(c[0], c[1], c=color, s=300, marker='o', edgecolor='black', linewidth=2,
                   label=f'State {label}', zorder=10)
        ax1.annotate(label, (c[0], c[1]), fontsize=14, fontweight='bold', ha='center', va='center',
                    color='white', zorder=11)
    
    # Draw the quadrilateral
    centroids_2d = np.array([state_centroids[l][:2] for l in state_labels])
    centroids_2d = np.vstack([centroids_2d, centroids_2d[0]])  # Close the loop
    ax1.plot(centroids_2d[:, 0], centroids_2d[:, 1], 'k-', linewidth=2, alpha=0.7)
    
    # Draw diagonals
    ax1.plot([state_centroids['A'][0], state_centroids['C'][0]], 
             [state_centroids['A'][1], state_centroids['C'][1]], 
             'purple', linewidth=2, linestyle='--', label='A→C diagonal')
    ax1.plot([state_centroids['B'][0], state_centroids['D'][0]], 
             [state_centroids['B'][1], state_centroids['D'][1]], 
             'brown', linewidth=2, linestyle='--', label='B→D diagonal')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax1.set_title('2D State Geometry (PC1 vs PC2)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_aspect('equal')
    
    # 2. PC1 vs PC3
    ax2 = fig.add_subplot(2, 3, 2)
    
    if show_individual_bins:
        for i, (label, color) in enumerate(zip(state_labels, state_colors)):
            start_bin = i * bins_per_state
            end_bin = (i + 1) * bins_per_state
            ax2.scatter(pca_mean[0, start_bin:end_bin], pca_mean[2, start_bin:end_bin], 
                       c=color, alpha=0.2, s=5)
    
    if show_trial_points and trial_state_centroids is not None:
        for label, color in zip(state_labels, state_colors):
            pts = np.array(trial_state_centroids[label])
            if len(pts) > 0:
                ax2.scatter(pts[:, 0], pts[:, 2], c=color, alpha=0.3, s=30, 
                           marker='o', edgecolor='none')
    
    for label, color in zip(state_labels, state_colors):
        c = state_centroids[label]
        ax2.scatter(c[0], c[2], c=color, s=300, marker='o', edgecolor='black', linewidth=2, zorder=10)
        ax2.annotate(label, (c[0], c[2]), fontsize=14, fontweight='bold', color='white',
                    ha='center', va='center', zorder=11)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=11)
    ax2.set_title('PC1 vs PC3', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    
    # 3. 3D plot
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    if show_individual_bins:
        for i, (label, color) in enumerate(zip(state_labels, state_colors)):
            start_bin = i * bins_per_state
            end_bin = (i + 1) * bins_per_state
            ax3.scatter(pca_mean[0, start_bin:end_bin], pca_mean[1, start_bin:end_bin],
                       pca_mean[2, start_bin:end_bin], c=color, alpha=0.2, s=5)
    
    if show_trial_points and trial_state_centroids is not None:
        for label, color in zip(state_labels, state_colors):
            pts = np.array(trial_state_centroids[label])
            if len(pts) > 0:
                ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, alpha=0.3, s=30,
                           marker='o', edgecolor='none')
    
    for label, color in zip(state_labels, state_colors):
        c = state_centroids[label]
        ax3.scatter(c[0], c[1], c[2], c=color, s=300, marker='o', edgecolor='black', linewidth=2)
        ax3.text(c[0], c[1], c[2], label, fontsize=12, fontweight='bold')
    
    # Draw quadrilateral in 3D
    for i in range(4):
        s1 = state_labels[i]
        s2 = state_labels[(i+1) % 4]
        c1, c2 = state_centroids[s1], state_centroids[s2]
        ax3.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 'k-', linewidth=2)
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax3.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax3.set_title('3D State Geometry', fontsize=12, fontweight='bold')
    
    # 4. Explained variance
    ax4 = fig.add_subplot(2, 3, 4)
    n_show = min(10, len(pca.explained_variance_ratio_))
    ax4.bar(range(1, n_show+1), pca.explained_variance_ratio_[:n_show] * 100, 
           color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Principal Component', fontsize=11)
    ax4.set_ylabel('Explained Variance (%)', fontsize=11)
    ax4.set_title('PCA Scree Plot', fontsize=12, fontweight='bold')
    cumvar = np.cumsum(pca.explained_variance_ratio_[:n_show]) * 100
    ax4.plot(range(1, n_show+1), cumvar, 'ro-', label='Cumulative')
    ax4.legend()
    
    # 5. Angles summary
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    summary_text = f"""
    Cyclicity Analysis in PCA Space
    ================================
    Session: {session_label}
    
    PCA Components: {n_components}
    Explained Variance: {np.sum(pca.explained_variance_ratio_[:n_components])*100:.1f}%
    
    Diagonal Angle (A→C vs B→D):
      Observed: {details['diagonal_angle']:.1f}°
      Ideal: 90° (orthogonal)
      Deviation: {details['diagonal_deviation']:.1f}°
    
    Parallel Angle (A→B vs C→D):
      Observed: {details['parallel_angle']:.1f}°
      Ideal: 0° or 180° (parallel)
      Deviation: {details['parallel_deviation']:.1f}°
    
    Cyclicity Score: {score:.1f}°
    (0° = perfect cyclic structure)
    """
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 6. PC trajectory over trial
    ax6 = fig.add_subplot(2, 3, 6)
    bins = np.arange(360)
    for i, (pc, color) in enumerate(zip(pca_mean[:3], ['red', 'blue', 'green'])):
        ax6.plot(bins, pc, color=color, linewidth=1.5, label=f'PC{i+1}')
    
    # Mark state boundaries
    for i in range(4):
        ax6.axvline(i * bins_per_state, color='gray', linestyle='--', alpha=0.5)
        ax6.text(i * bins_per_state + bins_per_state//2, ax6.get_ylim()[1], 
                state_labels[i], ha='center', fontsize=12, fontweight='bold')
    
    ax6.set_xlabel('Bin (within trial)', fontsize=11)
    ax6.set_ylabel('PC Score', fontsize=11)
    ax6.set_title('PC Trajectories Across Trial', fontsize=12, fontweight='bold')
    ax6.legend()
    
    plt.suptitle(f'PCA-Based Cyclic Structure Analysis: {session_label}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def analyze_all_sessions_pca(data_dic, mouse_recday, n_components=10, bins_per_state=90, verbose=True):
    """
    Analyze cyclicity in PCA space across all sessions.
    
    Parameters
    ----------
    data_dic : dict
        Data dictionary
    mouse_recday : str
        Recording identifier
    n_components : int
        Number of PCA components
    bins_per_state : int
        Bins per state
    verbose : bool
        Print results
        
    Returns
    -------
    results : dict
        Results for each session
    """
    sessions = list(data_dic[mouse_recday].keys())
    
    results = {
        'sessions': [],
        'scores': [],
        'scores_full': [],  # Full population space for comparison
        'diagonal_angles': [],
        'parallel_angles': [],
        'explained_variances': [],
        'details': []
    }
    
    for session in sessions:
        session_data = data_dic[mouse_recday][session]
        
        if 'Smoothed_norm' not in session_data or session_data['Smoothed_norm'] is None:
            continue
            
        neurons_norm = session_data['Smoothed_norm']
        task = session_data.get('Task', 'Unknown')
        
        # PCA space analysis
        score_pca, details_pca = analyze_cyclicity_pca(neurons_norm, n_components, bins_per_state)
        
        # Full space analysis for comparison
        score_full, _ = analyze_session_cyclicity(neurons_norm, bins_per_state)
        
        results['sessions'].append(session)
        results['scores'].append(score_pca)
        results['scores_full'].append(score_full)
        results['diagonal_angles'].append(details_pca['diagonal_angle'])
        results['parallel_angles'].append(details_pca['parallel_angle'])
        results['explained_variances'].append(details_pca['explained_variance'])
        results['details'].append(details_pca)
        
        if verbose:
            print(f"Session {session} (Task: {task}):")
            print(f"  PCA ({n_components} PCs, {details_pca['explained_variance']*100:.1f}% var):")
            print(f"    Diagonal: {details_pca['diagonal_angle']:.1f}°, Parallel: {details_pca['parallel_angle']:.1f}°")
            print(f"    Cyclicity score: {score_pca:.1f}° (full space: {score_full:.1f}°)")
            print()
    
    # Summary
    if results['scores']:
        results['mean_score_pca'] = np.nanmean(results['scores'])
        results['mean_score_full'] = np.nanmean(results['scores_full'])
        
        if verbose:
            print("=" * 50)
            print(f"Summary ({len(results['scores'])} sessions):")
            print(f"  Mean cyclicity (PCA): {results['mean_score_pca']:.1f}°")
            print(f"  Mean cyclicity (full): {results['mean_score_full']:.1f}°")
    
    return results


def get_state_population_vectors(neurons_norm, config=None, bins_per_state=90):
    """
    Extract mean population vectors for each state.
    
    Parameters
    ----------
    neurons_norm : ndarray
        Neural data of shape (num_neurons, num_trials, 360) or (num_neurons, 360)
    config : object, optional
        Config with bins_per_state attribute
    bins_per_state : int
        Bins per state (default 90 for 4 states in 360 bins)
        
    Returns
    -------
    state_vectors : dict
        Dictionary with keys 'A', 'B', 'C', 'D', each containing 
        mean population vector of shape (num_neurons,)
    """
    if config is not None:
        bins_per_state = config.bins_per_state
    
    # Handle both trial-averaged and single-trial data
    if neurons_norm.ndim == 3:
        # (num_neurons, num_trials, 360) -> average across trials first
        mean_activity = np.nanmean(neurons_norm, axis=1)  # (num_neurons, 360)
    else:
        mean_activity = neurons_norm  # (num_neurons, 360)
    
    num_neurons = mean_activity.shape[0]
    
    # Define state boundaries
    state_labels = ['A', 'B', 'C', 'D']
    state_vectors = {}
    
    for i, label in enumerate(state_labels):
        start_bin = i * bins_per_state
        end_bin = (i + 1) * bins_per_state
        
        # Mean activity across bins in this state
        state_vectors[label] = np.nanmean(mean_activity[:, start_bin:end_bin], axis=1)
    
    return state_vectors


def compute_vector_angle(v1, v2, degrees=True):
    """
    Compute angle between two vectors.
    
    Parameters
    ----------
    v1, v2 : ndarray
        Vectors to compare
    degrees : bool
        Return angle in degrees (default) or radians
        
    Returns
    -------
    angle : float
        Angle between vectors (0 to 180 degrees)
    """
    # Handle NaN values
    valid = ~np.isnan(v1) & ~np.isnan(v2)
    if np.sum(valid) < 2:
        return np.nan
    
    v1_valid = v1[valid]
    v2_valid = v2[valid]
    
    # Normalize
    norm1 = np.linalg.norm(v1_valid)
    norm2 = np.linalg.norm(v2_valid)
    
    if norm1 == 0 or norm2 == 0:
        return np.nan
    
    # Compute cosine similarity
    cos_angle = np.dot(v1_valid, v2_valid) / (norm1 * norm2)
    
    # Clip for numerical stability
    cos_angle = np.clip(cos_angle, -1, 1)
    
    angle = np.arccos(cos_angle)
    
    if degrees:
        angle = np.degrees(angle)
    
    return angle


def compute_cyclicity_score(state_vectors, return_details=False):
    """
    Compute cyclicity score measuring how well population vectors 
    exhibit cyclic geometric structure.
    
    Tests:
    1. Orthogonality: A→C and B→D should be at 90°
    2. Parallelism: A→B and C→D should be at 0° (or 180°)
    
    Parameters
    ----------
    state_vectors : dict
        Dictionary with 'A', 'B', 'C', 'D' keys containing population vectors
    return_details : bool
        If True, return detailed angle information
        
    Returns
    -------
    cyclicity_score : float
        Score where 0 = perfect cyclic structure, higher = worse
        Computed as: |diagonal_angle - 90| + min(|parallel_angle|, |parallel_angle - 180|)
    details : dict (if return_details=True)
        Contains individual angles and deviations
    """
    A = state_vectors['A']
    B = state_vectors['B']
    C = state_vectors['C']
    D = state_vectors['D']
    
    # Compute difference vectors
    AC = C - A  # Diagonal A to C
    BD = D - B  # Diagonal B to D
    AB = B - A  # Adjacent A to B
    CD = D - C  # Adjacent C to D
    
    # Compute angles
    diagonal_angle = compute_vector_angle(AC, BD)  # Should be ~90°
    parallel_angle = compute_vector_angle(AB, CD)  # Should be ~0° or ~180°
    
    if np.isnan(diagonal_angle) or np.isnan(parallel_angle):
        if return_details:
            return np.nan, {'diagonal_angle': diagonal_angle, 'parallel_angle': parallel_angle}
        return np.nan
    
    # Compute deviations from ideal
    diagonal_deviation = np.abs(diagonal_angle - 90)
    
    # For parallel, 0° and 180° are both "parallel" (parallel vs antiparallel)
    parallel_deviation = min(np.abs(parallel_angle), np.abs(parallel_angle - 180))
    
    # Cyclicity score: lower is better
    cyclicity_score = diagonal_deviation + parallel_deviation
    
    # Alternative score: difference between angles (ideally = 90)
    angle_difference = diagonal_angle - parallel_deviation  # Ideally 90
    
    if return_details:
        details = {
            'diagonal_angle': diagonal_angle,       # A→C vs B→D (ideal: 90°)
            'parallel_angle': parallel_angle,       # A→B vs C→D (ideal: 0° or 180°)
            'diagonal_deviation': diagonal_deviation,
            'parallel_deviation': parallel_deviation,
            'angle_difference': angle_difference,    # diagonal - parallel (ideal: 90°)
            'cyclicity_score': cyclicity_score,     # lower = more cyclic
        }
        return cyclicity_score, details
    
    return cyclicity_score


def analyze_session_cyclicity(neurons_norm, bins_per_state=90):
    """
    Analyze cyclicity for a single session.
    
    Parameters
    ----------
    neurons_norm : ndarray
        Neural data (num_neurons, num_trials, 360) or (num_neurons, 360)
    bins_per_state : int
        Bins per state (default 90)
        
    Returns
    -------
    score : float
        Cyclicity score (0 = perfect)
    details : dict
        Detailed angle information
    """
    state_vectors = get_state_population_vectors(neurons_norm, bins_per_state=bins_per_state)
    score, details = compute_cyclicity_score(state_vectors, return_details=True)
    return score, details


def analyze_all_sessions(data_dic, mouse_recday, bins_per_state=90, verbose=True):
    """
    Analyze cyclicity across all sessions for a recording.
    
    Parameters
    ----------
    data_dic : dict
        Data dictionary
    mouse_recday : str
        Recording identifier
    bins_per_state : int
        Bins per state
    verbose : bool
        Print results
        
    Returns
    -------
    results : dict
        Results for each session and summary statistics
    """
    sessions = list(data_dic[mouse_recday].keys())
    
    results = {
        'sessions': [],
        'scores': [],
        'diagonal_angles': [],
        'parallel_angles': [],
        'details': []
    }
    
    for session in sessions:
        session_data = data_dic[mouse_recday][session]
        
        # Get neural data
        if 'Smoothed_norm' not in session_data or session_data['Smoothed_norm'] is None:
            continue
            
        neurons_norm = session_data['Smoothed_norm']
        
        # Get task structure if available
        task = session_data.get('Task', 'Unknown')
        
        # Analyze
        score, details = analyze_session_cyclicity(neurons_norm, bins_per_state)
        
        results['sessions'].append(session)
        results['scores'].append(score)
        results['diagonal_angles'].append(details['diagonal_angle'])
        results['parallel_angles'].append(details['parallel_angle'])
        results['details'].append(details)
        
        if verbose:
            print(f"Session {session} (Task: {task}):")
            print(f"  Diagonal angle (A→C vs B→D): {details['diagonal_angle']:.1f}° (ideal: 90°)")
            print(f"  Parallel angle (A→B vs C→D): {details['parallel_angle']:.1f}° (ideal: 0° or 180°)")
            print(f"  Cyclicity score: {score:.1f}° (0 = perfect)")
            print()
    
    # Summary statistics
    valid_scores = [s for s in results['scores'] if not np.isnan(s)]
    if valid_scores:
        results['mean_score'] = np.mean(valid_scores)
        results['std_score'] = np.std(valid_scores)
        results['mean_diagonal'] = np.nanmean(results['diagonal_angles'])
        results['mean_parallel'] = np.nanmean(results['parallel_angles'])
        
        if verbose:
            print("=" * 50)
            print(f"Summary ({len(valid_scores)} sessions):")
            print(f"  Mean cyclicity score: {results['mean_score']:.1f}° ± {results['std_score']:.1f}°")
            print(f"  Mean diagonal angle: {results['mean_diagonal']:.1f}° (ideal: 90°)")
            print(f"  Mean parallel angle: {results['mean_parallel']:.1f}° (ideal: 0° or 180°)")
    
    return results


def plot_cyclicity_results(results, mouse_recday='', save_path=None):
    """
    Visualize cyclicity analysis results.
    
    Parameters
    ----------
    results : dict
        Results from analyze_all_sessions
    mouse_recday : str
        Recording identifier for title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sessions = results['sessions']
    scores = results['scores']
    diagonal_angles = results['diagonal_angles']
    parallel_angles = results['parallel_angles']
    
    # 1. Cyclicity scores per session
    ax1 = axes[0, 0]
    x = np.arange(len(sessions))
    ax1.bar(x, scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='green', linestyle='--', linewidth=2, label='Perfect (0°)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sessions], rotation=45, ha='right')
    ax1.set_xlabel('Session', fontsize=11)
    ax1.set_ylabel('Cyclicity Score (°)', fontsize=11)
    ax1.set_title('Cyclicity Score per Session\n(lower = more cyclic)', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2. Diagonal angles (should be 90°)
    ax2 = axes[0, 1]
    ax2.bar(x, diagonal_angles, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.axhline(90, color='green', linestyle='--', linewidth=2, label='Ideal (90°)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in sessions], rotation=45, ha='right')
    ax2.set_xlabel('Session', fontsize=11)
    ax2.set_ylabel('Angle (°)', fontsize=11)
    ax2.set_title('Diagonal Angle: A→C vs B→D\n(should be orthogonal = 90°)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 180)
    ax2.legend()
    
    # 3. Parallel angles (should be 0° or 180°)
    ax3 = axes[1, 0]
    ax3.bar(x, parallel_angles, color='darkgreen', alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='blue', linestyle='--', linewidth=2, label='Ideal (0°)')
    ax3.axhline(180, color='blue', linestyle='--', linewidth=2, label='Ideal (180°)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(s) for s in sessions], rotation=45, ha='right')
    ax3.set_xlabel('Session', fontsize=11)
    ax3.set_ylabel('Angle (°)', fontsize=11)
    ax3.set_title('Parallel Angle: A→B vs C→D\n(should be parallel = 0° or 180°)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 180)
    ax3.legend()
    
    # 4. Scatter: diagonal vs parallel
    ax4 = axes[1, 1]
    ax4.scatter(parallel_angles, diagonal_angles, c='purple', s=100, alpha=0.7, edgecolor='black')
    for i, session in enumerate(sessions):
        ax4.annotate(str(session), (parallel_angles[i], diagonal_angles[i]), 
                    fontsize=9, ha='left', va='bottom')
    
    # Mark ideal point (0, 90) or (180, 90)
    ax4.scatter([0, 180], [90, 90], c='green', s=200, marker='*', 
               label='Ideal cyclic structure', zorder=5)
    ax4.axhline(90, color='green', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='blue', linestyle='--', alpha=0.5)
    ax4.axvline(180, color='blue', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Parallel Angle (A→B vs C→D)', fontsize=11)
    ax4.set_ylabel('Diagonal Angle (A→C vs B→D)', fontsize=11)
    ax4.set_title('Geometric Structure', fontsize=12, fontweight='bold')
    ax4.set_xlim(-10, 190)
    ax4.set_ylim(-10, 190)
    ax4.legend()
    ax4.set_aspect('equal')
    
    plt.suptitle(f'Cyclic Structure Analysis: {mouse_recday}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def run_permutation_test(neurons_norm, n_permutations=1000, bins_per_state=90):
    """
    Test if observed cyclicity is significantly better than chance.
    
    Shuffles neuron identities to create null distribution.
    
    Parameters
    ----------
    neurons_norm : ndarray
        Neural data
    n_permutations : int
        Number of permutations
    bins_per_state : int
        Bins per state
        
    Returns
    -------
    p_value : float
        Probability of observing cyclicity by chance
    observed_score : float
        Observed cyclicity score
    null_scores : ndarray
        Null distribution of scores
    """
    # Observed score
    observed_score, _ = analyze_session_cyclicity(neurons_norm, bins_per_state)
    
    if np.isnan(observed_score):
        return np.nan, observed_score, np.array([])
    
    # Handle both 2D and 3D data
    if neurons_norm.ndim == 3:
        mean_activity = np.nanmean(neurons_norm, axis=1)
    else:
        mean_activity = neurons_norm
    
    num_neurons = mean_activity.shape[0]
    
    # Null distribution: shuffle neuron identities
    null_scores = []
    for _ in range(n_permutations):
        # Shuffle neurons
        shuffled_indices = np.random.permutation(num_neurons)
        shuffled_activity = mean_activity[shuffled_indices, :]
        
        # Compute score
        state_vectors = get_state_population_vectors(shuffled_activity, bins_per_state=bins_per_state)
        score = compute_cyclicity_score(state_vectors)
        null_scores.append(score)
    
    null_scores = np.array(null_scores)
    
    # P-value: proportion of null scores as good or better than observed
    # Lower score = better, so we count how many are <= observed
    p_value = np.mean(null_scores <= observed_score)
    
    return p_value, observed_score, null_scores


def plot_permutation_test(observed_score, null_scores, p_value, session_label='', save_path=None):
    """
    Plot the null distribution from permutation test with observed score.
    
    Parameters
    ----------
    observed_score : float
        Observed cyclicity score
    null_scores : ndarray
        Null distribution of scores
    p_value : float
        P-value from permutation test
    session_label : str
        Label for the session
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot null distribution
    ax.hist(null_scores, bins=50, color='gray', alpha=0.7, edgecolor='black',
            label=f'Null distribution (n={len(null_scores)})', density=True)
    
    # Mark observed score
    ax.axvline(observed_score, color='red', linestyle='-', linewidth=3,
               label=f'Observed = {observed_score:.1f}°')
    
    # Mark mean of null
    null_mean = np.mean(null_scores)
    ax.axvline(null_mean, color='blue', linestyle='--', linewidth=2,
               label=f'Null mean = {null_mean:.1f}°')
    
    # Add perfect score line
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.5,
               label='Perfect cyclic (0°)')
    
    ax.set_xlabel('Cyclicity Score (°)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Permutation Test for Cyclic Structure {session_label}\n'
                 f'p = {p_value:.4f} (lower score = more cyclic)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Add text box with stats
    textstr = f'Observed: {observed_score:.1f}°\nNull mean: {null_mean:.1f}°\nNull std: {np.std(null_scores):.1f}°\np-value: {p_value:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def run_permutation_test_all_sessions(data_dic, mouse_recday, n_permutations=1000, 
                                       bins_per_state=90, verbose=True):
    """
    Run permutation tests for all sessions in a recording.
    
    Parameters
    ----------
    data_dic : dict
        Data dictionary
    mouse_recday : str
        Recording identifier
    n_permutations : int
        Number of permutations per session
    bins_per_state : int
        Bins per state
    verbose : bool
        Print progress
        
    Returns
    -------
    perm_results : dict
        Results including p-values and null distributions for each session
    """
    sessions = list(data_dic[mouse_recday].keys())
    
    perm_results = {
        'sessions': [],
        'p_values': [],
        'observed_scores': [],
        'null_scores': [],
        'null_means': [],
        'null_stds': []
    }
    
    for session in sessions:
        session_data = data_dic[mouse_recday][session]
        
        if 'Smoothed_norm' not in session_data or session_data['Smoothed_norm'] is None:
            continue
        
        neurons_norm = session_data['Smoothed_norm']
        task = session_data.get('Task', 'Unknown')
        
        if verbose:
            print(f"Session {session} (Task: {task})... ", end='', flush=True)
        
        p_val, obs_score, null = run_permutation_test(
            neurons_norm, n_permutations=n_permutations, bins_per_state=bins_per_state
        )
        
        perm_results['sessions'].append(session)
        perm_results['p_values'].append(p_val)
        perm_results['observed_scores'].append(obs_score)
        perm_results['null_scores'].append(null)
        perm_results['null_means'].append(np.mean(null) if len(null) > 0 else np.nan)
        perm_results['null_stds'].append(np.std(null) if len(null) > 0 else np.nan)
        
        if verbose:
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
            print(f"observed={obs_score:.1f}°, null={np.mean(null):.1f}±{np.std(null):.1f}°, p={p_val:.4f} {sig}")
    
    return perm_results


def plot_permutation_summary(perm_results, mouse_recday='', save_path=None):
    """
    Plot summary of permutation tests across all sessions.
    
    Parameters
    ----------
    perm_results : dict
        Results from run_permutation_test_all_sessions
    mouse_recday : str
        Recording identifier for title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    sessions = perm_results['sessions']
    p_values = perm_results['p_values']
    observed = perm_results['observed_scores']
    null_means = perm_results['null_means']
    null_stds = perm_results['null_stds']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(sessions))
    
    # 1. Observed vs Null means
    ax1 = axes[0, 0]
    ax1.bar(x - 0.15, observed, 0.3, color='red', alpha=0.7, label='Observed', edgecolor='black')
    ax1.bar(x + 0.15, null_means, 0.3, color='gray', alpha=0.7, label='Null mean', edgecolor='black',
            yerr=null_stds, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sessions], rotation=45, ha='right')
    ax1.set_xlabel('Session', fontsize=11)
    ax1.set_ylabel('Cyclicity Score (°)', fontsize=11)
    ax1.set_title('Observed vs Null Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2. P-values per session
    ax2 = axes[0, 1]
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    ax2.bar(x, p_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05')
    ax2.axhline(0.01, color='orange', linestyle='--', linewidth=2, label='p = 0.01')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in sessions], rotation=45, ha='right')
    ax2.set_xlabel('Session', fontsize=11)
    ax2.set_ylabel('P-value', fontsize=11)
    ax2.set_title('Permutation Test P-values\n(green = significant p < 0.05)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(1, max(p_values) * 1.1))
    ax2.legend()
    
    # 3. Effect size (observed - null_mean) / null_std
    ax3 = axes[1, 0]
    effect_sizes = [(obs - null_m) / null_s if null_s > 0 else 0 
                    for obs, null_m, null_s in zip(observed, null_means, null_stds)]
    colors = ['green' if es < -1.96 else ('yellow' if es < 0 else 'red') for es in effect_sizes]
    ax3.bar(x, effect_sizes, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.axhline(-1.96, color='green', linestyle='--', linewidth=2, label='z = -1.96 (p~0.025)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(s) for s in sessions], rotation=45, ha='right')
    ax3.set_xlabel('Session', fontsize=11)
    ax3.set_ylabel('Z-score (effect size)', fontsize=11)
    ax3.set_title('Effect Size: (Observed - Null) / Null_SD\n(negative = better than chance)', 
                 fontsize=12, fontweight='bold')
    ax3.legend()
    
    # 4. Combined null distributions
    ax4 = axes[1, 1]
    all_null = np.concatenate(perm_results['null_scores'])
    ax4.hist(all_null, bins=50, color='gray', alpha=0.7, edgecolor='black', 
             label=f'Combined null (n={len(all_null)})', density=True)
    
    # Plot each observed score
    for i, (obs, sess) in enumerate(zip(observed, sessions)):
        ax4.axvline(obs, color=plt.cm.tab10(i % 10), linestyle='-', linewidth=2, alpha=0.7,
                   label=f'Session {sess}: {obs:.1f}°')
    
    ax4.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect (0°)')
    ax4.set_xlabel('Cyclicity Score (°)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Combined Null Distribution vs Observed', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    
    # Summary stats
    n_sig = sum(1 for p in p_values if p < 0.05)
    n_total = len(p_values)
    
    plt.suptitle(f'Permutation Test Summary: {mouse_recday}\n'
                 f'{n_sig}/{n_total} sessions significant (p < 0.05)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def analyze_all_recordings(data_dic, bins_per_state=90, verbose=True):
    """
    Analyze cyclicity across all recordings in data_dic.
    
    Parameters
    ----------
    data_dic : dict
        Full data dictionary
    bins_per_state : int
        Bins per state
    verbose : bool
        Print results
        
    Returns
    -------
    all_results : dict
        Results for each mouse_recday
    """
    all_results = {}
    
    for mouse_recday in data_dic.keys():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Recording: {mouse_recday}")
            print('='*60)
        
        results = analyze_all_sessions(data_dic, mouse_recday, bins_per_state, verbose)
        all_results[mouse_recday] = results
    
    return all_results


# Example usage
if __name__ == "__main__":
    print("Cyclic Structure Analysis Module")
    print("=" * 40)
    print("""
Usage in notebook:
    
    from cyclic_structure_analysis import (
        analyze_all_sessions, 
        plot_cyclicity_results,
        run_permutation_test
    )
    
    # Analyze all sessions for one recording
    results = analyze_all_sessions(data_dic, 'ah04_230526')
    
    # Plot results
    plot_cyclicity_results(results, mouse_recday='ah04_230526')
    
    # Permutation test for significance
    neurons_norm = data_dic['ah04_230526'][0]['Smoothed_norm']
    p_val, obs_score, null = run_permutation_test(neurons_norm)
    print(f"p-value: {p_val:.4f}")
    """)
