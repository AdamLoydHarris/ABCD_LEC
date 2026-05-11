"""
Elastic Net Regression Analysis for Task-Space Abstraction

Implementation of the regression model from El-Gaby et al. 2024 (Nature).

Method: For each neuron, compute a regression model that describes state-tuning 
activity as a function of all possible combinations of goal-progress/place and 
all task lags from each possible goal-progress/place. Uses ElasticNet with 
L1:L2 ratio of 1:1 and alpha=0.01.

Regressors: num_locations x num_goal_progress_bins x num_lags
For the original paper: 9 x 3 x 12 = 324 regressors

Data format expected:
- Neurons_norm: shape (num_neurons, num_trials, 360) - normalized firing rates
- Locs_norm: shape (num_trials, 360) - location at each normalized bin
- 360 bins = 90 bins per state × 4 states
- Locations 1-9 are nodes (used for regression), 10-21 are edges (ignored)

Usage:
    This requires loading data_dic with the appropriate normalized data
"""

import numpy as np
from sklearn.linear_model import ElasticNet, PoissonRegressor
from scipy import stats
import warnings
from tqdm import tqdm

# Suppress constant input warnings for correlation
warnings.filterwarnings('ignore', message='An input array is constant')
warnings.filterwarnings('ignore', message='Mean of empty slice')

# ============================================================================
# Configuration
# ============================================================================

class RegressionConfig:
    """Configuration for the regression analysis."""
    
    def __init__(
        self,
        num_locations=9,            # Number of node locations (1-9, excluding edges)
        num_goal_progress_bins=3,   # Number of goal-progress bins per state
        num_task_states=4,          # Number of task states (A, B, C, D)
        num_lags=12,                # Number of lags in task space (states × phases)
        alpha=0.01,                 # Regularization strength
        use_poisson=False,          # Use Poisson regression instead of ElasticNet
        use_positive_only=True,     # Constrain coefficients to be positive
        smoothing_sigma=10,         # Smoothing sigma for neural data
        num_bins_per_state=90,      # Number of normalized bins per state (90)
        bins_per_phase=30,          # Bins per goal-progress phase (90/3 = 30)
    ):
        self.num_locations = num_locations
        self.num_goal_progress_bins = num_goal_progress_bins
        self.num_task_states = num_task_states
        self.num_lags = num_lags
        self.alpha = alpha
        self.use_poisson = use_poisson
        self.use_positive_only = use_positive_only
        self.smoothing_sigma = smoothing_sigma
        self.num_bins_per_state = num_bins_per_state
        self.bins_per_phase = bins_per_phase
        self.total_bins = num_bins_per_state * num_task_states  # 360
        
        # Derived
        self.num_regressors = num_locations * num_goal_progress_bins * num_lags


# ============================================================================
# Helper Functions for Normalized Data (360 bins)
# ============================================================================

def get_goal_progress_from_bin(bin_idx, config):
    """
    Get goal progress phase (0, 1, 2) from a normalized bin index.
    
    With 90 bins per state and 3 phases:
    - bins 0-29: phase 0
    - bins 30-59: phase 1  
    - bins 60-89: phase 2
    
    Parameters
    ----------
    bin_idx : int or ndarray
        Bin index within a state (0-89)
    config : RegressionConfig
        Configuration object
        
    Returns
    -------
    phase : int or ndarray
        Goal progress phase (0, 1, or 2)
    """
    bin_within_state = bin_idx % config.num_bins_per_state
    phase = bin_within_state // config.bins_per_phase
    return np.minimum(phase, config.num_goal_progress_bins - 1)


def get_state_from_bin(bin_idx, config):
    """
    Get task state (0, 1, 2, 3) from a normalized bin index (0-359).
    
    Parameters
    ----------
    bin_idx : int or ndarray
        Bin index (0-359)
    config : RegressionConfig
        Configuration object
        
    Returns
    -------
    state : int or ndarray
        Task state (0, 1, 2, or 3)
    """
    return bin_idx // config.num_bins_per_state


def generate_regressors_from_norm(
    locs_norm,
    config,
    multiple_bumps=True
):
    """
    Generate lagged regressors from normalized location data.
    
    This creates "bumps" that are initiated when the animal visits a particular
    location/goal-progress combination, and then move through task space as
    trial progresses.
    
    Parameters
    ----------
    locs_norm : ndarray
        Shape (num_trials, 360) - location at each normalized bin
    config : RegressionConfig
        Configuration object
    multiple_bumps : bool
        Whether to allow multiple bumps per anchor point
        
    Returns
    -------
    regressors : ndarray
        Shape (num_trials, 360, num_regressors) - regressor matrix
    """
    num_trials, num_bins = locs_norm.shape
    num_locs = config.num_locations
    num_phases = config.num_goal_progress_bins
    num_lags = config.num_lags
    
    # Output: (num_trials, num_bins, num_regressors)
    regressors = np.zeros((num_trials, num_bins, config.num_regressors))
    
    for trial_idx in range(num_trials):
        # Module anchor progress: tracks activity "bumps" for each anchor point
        # Shape: (num_locations, num_phases, num_lags)
        module_anchor_progress = np.zeros((num_locs, num_phases, num_lags))
        
        prev_phase = -1
        prev_location = -1
        
        for bin_idx in range(num_bins):
            loc = locs_norm[trial_idx, bin_idx]
            
            # Skip if NaN or edge location (>9)
            if np.isnan(loc) or loc > num_locs or loc < 1:
                regressors[trial_idx, bin_idx] = module_anchor_progress.flatten()
                continue
            
            # Convert to 0-indexed location (1-9 -> 0-8)
            current_loc = int(loc) - 1
            current_phase = get_goal_progress_from_bin(bin_idx, config)
            
            # Check for phase transition
            phase_changed = (current_phase != prev_phase)
            location_changed = (current_loc != prev_location and 
                               current_loc >= 0 and current_loc < num_locs)
            
            if phase_changed:
                # Move bumps forward in task space for all anchor points
                for loc_idx in range(num_locs):
                    for phase_idx in range(num_phases):
                        # Roll the bump forward
                        module_anchor_progress[loc_idx, phase_idx] = np.roll(
                            module_anchor_progress[loc_idx, phase_idx], 1
                        )
                        
                        # Check if this location/phase matches current position
                        if (current_loc == loc_idx and current_phase == phase_idx):
                            if multiple_bumps or np.sum(module_anchor_progress[loc_idx, phase_idx]) == 0:
                                module_anchor_progress[loc_idx, phase_idx, 0] = 1
                            else:
                                module_anchor_progress[loc_idx, phase_idx, 1] = 0
                        else:
                            module_anchor_progress[loc_idx, phase_idx, 1] = 0
                            
            elif location_changed:
                # Location change without phase change - new anchor initiation
                for loc_idx in range(num_locs):
                    for phase_idx in range(num_phases):
                        if (current_loc == loc_idx and current_phase == phase_idx):
                            if multiple_bumps or np.sum(module_anchor_progress[loc_idx, phase_idx]) == 0:
                                module_anchor_progress[loc_idx, phase_idx, 1] = 1
            
            # Store current regressor state
            regressors[trial_idx, bin_idx] = module_anchor_progress.flatten()
            
            prev_phase = current_phase
            prev_location = current_loc
    
    # Roll back by 1 (compensate for forward lag in loop)
    # Apply along the lag dimension within the flattened array
    regressors_reshaped = regressors.reshape(num_trials, num_bins, num_locs, num_phases, num_lags)
    regressors_reshaped = np.roll(regressors_reshaped, -1, axis=4)
    regressors = regressors_reshaped.reshape(num_trials, num_bins, config.num_regressors)
    
    return regressors


def compute_preferred_phase_from_norm(neurons_norm, config):
    """
    Compute the preferred goal-progress phase for neurons from normalized data.
    
    Parameters
    ----------
    neurons_norm : ndarray
        Shape (num_neurons, num_trials, 360) - normalized neural activity
    config : RegressionConfig
        Configuration object
        
    Returns
    -------
    pref_phases : ndarray
        Shape (num_neurons,) - preferred phase for each neuron
    phase_means : ndarray
        Shape (num_neurons, 3) - mean activity in each phase
    """
    num_neurons = neurons_norm.shape[0]
    num_phases = config.num_goal_progress_bins
    
    pref_phases = np.zeros(num_neurons, dtype=int)
    phase_means = np.zeros((num_neurons, num_phases))
    
    # Create phase mask for all 360 bins
    bin_indices = np.arange(config.total_bins)
    phases_per_bin = get_goal_progress_from_bin(bin_indices, config)
    
    for neuron_idx in range(num_neurons):
        # Average across trials, then compute mean per phase
        neuron_mean = np.nanmean(neurons_norm[neuron_idx], axis=0)  # Shape (360,)
        
        for phase in range(num_phases):
            phase_mask = phases_per_bin == phase
            phase_means[neuron_idx, phase] = np.nanmean(neuron_mean[phase_mask])
        
        pref_phases[neuron_idx] = np.argmax(phase_means[neuron_idx])
    
    return pref_phases, phase_means


def identify_state_tuned_neurons(neurons_norm, config, p_threshold=0.05):
    """
    Identify state-tuned neurons using the El-Gaby et al. 2024 method.
    
    Method:
    1. Take peak firing rate in each state (A, B, C, D) for each trial
    2. Z-score each row (across states) so each trial has mean=0, std=1
    3. Extract z-scores for the preferred state across all trials
    4. t-test this array against 0
    5. p < threshold → state-tuned neuron
    
    Parameters
    ----------
    neurons_norm : ndarray
        Shape (num_neurons, num_trials, 360) - normalized neural activity
    config : RegressionConfig
        Configuration object
    p_threshold : float
        P-value threshold for significance (default 0.05, use 0.01 for stringent)
        
    Returns
    -------
    is_state_tuned : ndarray
        Shape (num_neurons,) - boolean mask of state-tuned neurons
    state_tuning_results : dict
        Dictionary with detailed results:
        - 'pref_states': preferred state for each neuron
        - 'p_values': p-value from t-test
        - 't_stats': t-statistic from t-test
        - 'z_scores_pref': z-scores for preferred state (num_neurons, num_trials)
    """
    num_neurons, num_trials, num_bins = neurons_norm.shape
    num_states = config.num_task_states
    bins_per_state = config.num_bins_per_state
    
    # Storage
    is_state_tuned = np.zeros(num_neurons, dtype=bool)
    pref_states = np.zeros(num_neurons, dtype=int)
    p_values = np.full(num_neurons, np.nan)
    t_stats = np.full(num_neurons, np.nan)
    z_scores_pref_all = np.full((num_neurons, num_trials), np.nan)
    
    for neuron_idx in range(num_neurons):
        neuron_data = neurons_norm[neuron_idx]  # Shape: (num_trials, 360)
        
        # Step 1: Get peak firing rate in each state for each trial
        # Shape: (num_trials, num_states)
        peak_per_state = np.zeros((num_trials, num_states))
        
        for state in range(num_states):
            state_start = state * bins_per_state
            state_end = (state + 1) * bins_per_state
            state_data = neuron_data[:, state_start:state_end]  # (num_trials, 90)
            peak_per_state[:, state] = np.nanmax(state_data, axis=1)
        
        # Step 2: Z-score each row (across the 4 states)
        # Each trial gets mean=0, std=1 across states
        row_means = np.nanmean(peak_per_state, axis=1, keepdims=True)
        row_stds = np.nanstd(peak_per_state, axis=1, keepdims=True)
        
        # Avoid division by zero
        row_stds[row_stds == 0] = np.nan
        z_scored = (peak_per_state - row_means) / row_stds
        
        # Step 3: Find preferred state (highest mean z-score)
        mean_z_per_state = np.nanmean(z_scored, axis=0)
        pref_state = np.nanargmax(mean_z_per_state)
        pref_states[neuron_idx] = pref_state
        
        # Step 4: Extract z-scores for preferred state across all trials
        z_scores_pref = z_scored[:, pref_state]
        z_scores_pref_all[neuron_idx] = z_scores_pref
        
        # Remove NaN trials
        z_valid = z_scores_pref[~np.isnan(z_scores_pref)]
        
        if len(z_valid) < 3:
            continue
        
        # Step 5: t-test against 0 (two-sided)
        t_stat, p_val = stats.ttest_1samp(z_valid, 0)
        t_stats[neuron_idx] = t_stat
        p_values[neuron_idx] = p_val
        
        # Step 6: Check significance
        if p_val < p_threshold:
            is_state_tuned[neuron_idx] = True
    
    results = {
        'pref_states': pref_states,
        'p_values': p_values,
        't_stats': t_stats,
        'z_scores_pref': z_scores_pref_all,
        'p_threshold': p_threshold,
        'num_state_tuned': np.sum(is_state_tuned),
        'fraction_state_tuned': np.mean(is_state_tuned)
    }
    
    return is_state_tuned, results


def identify_state_tuned_neurons_across_sessions(data_dic, mouse_recday, config, 
                                                   valid_sessions=None, p_threshold=0.05,
                                                   require_all_sessions=False):
    """
    Identify state-tuned neurons consistently across sessions.
    
    Parameters
    ----------
    data_dic : dict
        Data dictionary
    mouse_recday : str
        Recording identifier
    config : RegressionConfig
        Configuration object
    valid_sessions : list
        Sessions to analyze
    p_threshold : float
        P-value threshold
    require_all_sessions : bool
        If True, neuron must be state-tuned in ALL sessions
        If False (default), neuron must be state-tuned in ANY session
        
    Returns
    -------
    is_state_tuned : ndarray
        Combined boolean mask
    session_results : dict
        Results per session
    """
    if valid_sessions is None:
        valid_sessions = list(data_dic[mouse_recday].keys())
    
    # Filter to sessions with data
    valid_sessions = [
        s for s in valid_sessions
        if ('Neurons_norm' in data_dic[mouse_recday][s] and 
            data_dic[mouse_recday][s]['Neurons_norm'] is not None)
    ]
    
    if len(valid_sessions) == 0:
        return None, None
    
    # Get number of neurons
    first_session = valid_sessions[0]
    num_neurons = data_dic[mouse_recday][first_session]['Neurons_norm'].shape[0]
    
    # Store results per session
    session_masks = np.zeros((num_neurons, len(valid_sessions)), dtype=bool)
    session_results = {}
    
    for i, session in enumerate(valid_sessions):
        neurons_norm = data_dic[mouse_recday][session]['Neurons_norm']
        is_tuned, results = identify_state_tuned_neurons(neurons_norm, config, p_threshold)
        session_masks[:, i] = is_tuned
        session_results[session] = results
    
    # Combine across sessions
    if require_all_sessions:
        is_state_tuned = np.all(session_masks, axis=1)
    else:
        is_state_tuned = np.any(session_masks, axis=1)
    
    # Summary
    summary = {
        'valid_sessions': valid_sessions,
        'session_masks': session_masks,
        'num_tuned_per_session': np.sum(session_masks, axis=0),
        'num_tuned_combined': np.sum(is_state_tuned),
        'fraction_tuned': np.mean(is_state_tuned),
        'require_all_sessions': require_all_sessions
    }
    
    return is_state_tuned, {'session_results': session_results, 'summary': summary}


# ============================================================================
# Main Regression Functions
# ============================================================================

def fit_elasticnet_regression(
    X,
    y,
    config,
    return_model=False
):
    """
    Fit elastic net (or Poisson) regression.
    
    Parameters
    ----------
    X : ndarray
        Shape (n_samples, n_features) - regressor matrix
    y : ndarray
        Shape (n_samples,) - neural activity
    config : RegressionConfig
        Configuration object
    return_model : bool
        Whether to return the fitted model
        
    Returns
    -------
    coeffs : ndarray
        Shape (n_features,) - regression coefficients
    model : sklearn model (optional)
        Fitted model if return_model=True
    """
    # Remove NaN samples
    valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    if len(y_valid) < 10:
        coeffs = np.full(X.shape[1], np.nan)
        if return_model:
            return coeffs, None
        return coeffs
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if config.use_poisson:
            model = PoissonRegressor(alpha=config.alpha, max_iter=1000)
            model.fit(X_valid, y_valid)
        else:
            # ElasticNet with l1_ratio=0.5 gives 1:1 mix of L1 and L2
            model = ElasticNet(
                alpha=config.alpha,
                l1_ratio=0.5,
                positive=config.use_positive_only,
                max_iter=1000
            )
            model.fit(X_valid, y_valid)
    
    coeffs = model.coef_
    
    if return_model:
        return coeffs, model
    return coeffs


def run_regression_for_session(
    data_dic,
    mouse_recday,
    session,
    config,
    verbose=False
):
    """
    Run regression for all neurons in a single session using normalized data.
    
    Parameters
    ----------
    data_dic : dict
        Data dictionary with structure data_dic[mouse_recday][session][...]
    mouse_recday : str
        Recording day identifier
    session : int
        Session index
    config : RegressionConfig
        Configuration object
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Dictionary with regression results
    """
    session_data = data_dic[mouse_recday][session]
    
    # Extract normalized data
    neurons_norm = session_data.get('Neurons_norm')
    locs_norm = session_data.get('Locs_norm')
    
    if neurons_norm is None or locs_norm is None:
        if verbose:
            print(f"Missing Neurons_norm or Locs_norm for {mouse_recday} session {session}")
        return None
    
    num_neurons, num_trials, num_bins = neurons_norm.shape
    
    if num_bins != config.total_bins:
        if verbose:
            print(f"Warning: Expected {config.total_bins} bins, got {num_bins}")
    
    # Generate regressors from normalized locations
    # Shape: (num_trials, num_bins, num_regressors)
    regressors = generate_regressors_from_norm(locs_norm, config)
    
    # Compute preferred phases for all neurons
    pref_phases, phase_means = compute_preferred_phase_from_norm(neurons_norm, config)
    
    # Create phase mask for filtering
    bin_indices = np.arange(config.total_bins)
    phases_per_bin = get_goal_progress_from_bin(bin_indices, config)
    
    # Fit regression for each neuron
    coeffs_all = np.zeros((num_neurons, config.num_regressors))
    
    iterator = range(num_neurons)
    if verbose:
        iterator = tqdm(iterator, desc=f"Fitting neurons for {mouse_recday} session {session}")
    
    for neuron_idx in iterator:
        pref_phase = pref_phases[neuron_idx]
        
        # Get mask for preferred phase bins
        phase_mask = phases_per_bin == pref_phase
        
        # Flatten across trials, keeping only preferred phase bins
        # neurons_norm shape: (num_neurons, num_trials, 360)
        # regressors shape: (num_trials, 360, num_regressors)
        
        y_all = []
        X_all = []
        
        for trial_idx in range(num_trials):
            y_trial = neurons_norm[neuron_idx, trial_idx, phase_mask]
            X_trial = regressors[trial_idx, phase_mask, :]
            y_all.append(y_trial)
            X_all.append(X_trial)
        
        y = np.concatenate(y_all)
        X = np.vstack(X_all)
        
        # Fit regression
        coeffs = fit_elasticnet_regression(X, y, config)
        coeffs_all[neuron_idx] = coeffs
    
    results = {
        'coeffs': coeffs_all,
        'pref_phases': pref_phases,
        'phase_means': phase_means,
        'regressors': regressors,
        'num_neurons': num_neurons,
        'num_trials': num_trials
    }
    
    return results


def filter_unique_task_sessions(data_dic, mouse_recday, sessions):
    """
    Filter sessions to exclude duplicate task structures.
    
    Keeps only the first occurrence of each unique task structure.
    For example, if sessions have tasks [x, y, z, x, a, b, c, a], 
    returns sessions [0, 1, 2, 4, 5, 6] (excluding sessions 3 and 7).
    
    Parameters
    ----------
    data_dic : dict
        Data dictionary
    mouse_recday : str
        Recording day identifier
    sessions : list
        List of session indices to filter
        
    Returns
    -------
    filtered_sessions : list
        Sessions with unique task structures (first occurrence kept)
    """
    seen_tasks = set()
    filtered_sessions = []
    
    for session in sessions:
        # Get task structure - try 'Task' key, handle various formats
        session_data = data_dic[mouse_recday][session]
        
        if 'Task' in session_data:
            task = session_data['Task']
            # Convert to hashable format if needed
            if isinstance(task, np.ndarray):
                task_key = tuple(task.tolist())
            elif isinstance(task, list):
                task_key = tuple(task)
            else:
                task_key = task
        else:
            # If no Task key, include the session (can't filter)
            filtered_sessions.append(session)
            continue
        
        if task_key not in seen_tasks:
            seen_tasks.add(task_key)
            filtered_sessions.append(session)
    
    return filtered_sessions


def run_cross_validated_regression(
    data_dic,
    mouse_recday,
    config,
    valid_sessions=None,
    state_tuned_mask=None,
    state_tuning_p_threshold=0.05,
    require_state_tuning=True,
    filter_duplicate_tasks=True,
    verbose=True
):
    """
    Run leave-one-out cross-validated regression across sessions.
    
    For each neuron, train on N-1 sessions and test on the left-out session.
    Uses normalized data (Neurons_norm, Locs_norm).
    
    Following El-Gaby et al. 2024, only state-tuned neurons are analyzed.
    
    Parameters
    ----------
    data_dic : dict
        Data dictionary
    mouse_recday : str
        Recording day identifier
    config : RegressionConfig
        Configuration object
    valid_sessions : list
        List of valid session indices (if None, use all sessions)
    state_tuned_mask : ndarray, optional
        Pre-computed boolean mask of state-tuned neurons.
        If None and require_state_tuning=True, will compute automatically.
    state_tuning_p_threshold : float
        P-value threshold for state tuning identification (default 0.05)
    require_state_tuning : bool
        If True (default), only analyze state-tuned neurons.
        If False, analyze all neurons.
    filter_duplicate_tasks : bool
        If True (default), exclude sessions with duplicate task structures.
        Keeps only the first occurrence of each unique task.
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Cross-validation results including correlations
    """
    if valid_sessions is None:
        valid_sessions = list(data_dic[mouse_recday].keys())
    
    # Filter to sessions with required normalized data
    valid_sessions = [
        s for s in valid_sessions
        if ('Neurons_norm' in data_dic[mouse_recday][s] and 
            'Locs_norm' in data_dic[mouse_recday][s] and
            data_dic[mouse_recday][s]['Neurons_norm'] is not None and
            data_dic[mouse_recday][s]['Locs_norm'] is not None)
    ]
    
    # Filter out duplicate task structures (keep first occurrence)
    if filter_duplicate_tasks:
        sessions_before = len(valid_sessions)
        valid_sessions = filter_unique_task_sessions(data_dic, mouse_recday, valid_sessions)
        if verbose and len(valid_sessions) < sessions_before:
            print(f"Filtered {sessions_before - len(valid_sessions)} duplicate task sessions")
    
    if len(valid_sessions) < 2:
        print(f"Not enough valid sessions with normalized data for {mouse_recday}")
        return None
    
    num_sessions = len(valid_sessions)
    
    # Get number of neurons from first valid session
    first_session = valid_sessions[0]
    num_neurons = data_dic[mouse_recday][first_session]['Neurons_norm'].shape[0]
    
    # Identify state-tuned neurons (El-Gaby et al. 2024 method)
    if require_state_tuning:
        if state_tuned_mask is None:
            if verbose:
                print(f"Identifying state-tuned neurons (p < {state_tuning_p_threshold})...")
            state_tuned_mask, state_tuning_results = identify_state_tuned_neurons_across_sessions(
                data_dic, mouse_recday, config, 
                valid_sessions=valid_sessions,
                p_threshold=state_tuning_p_threshold,
                require_all_sessions=False
            )
            if verbose:
                n_tuned = np.sum(state_tuned_mask)
                print(f"  Found {n_tuned}/{num_neurons} state-tuned neurons ({100*n_tuned/num_neurons:.1f}%)")
        else:
            state_tuning_results = None
        
        # Get indices of state-tuned neurons
        neurons_to_analyze = np.where(state_tuned_mask)[0]
        if len(neurons_to_analyze) == 0:
            print("No state-tuned neurons found!")
            return None
    else:
        state_tuned_mask = np.ones(num_neurons, dtype=bool)
        state_tuning_results = None
        neurons_to_analyze = np.arange(num_neurons)
    
    if verbose:
        print(f"Analyzing {len(neurons_to_analyze)}/{num_neurons} neurons, {num_sessions} sessions")
    
    # Pre-compute regressors and preferred phases for all sessions
    session_data_cache = {}
    for session in valid_sessions:
        neurons_norm = data_dic[mouse_recday][session]['Neurons_norm']
        locs_norm = data_dic[mouse_recday][session]['Locs_norm']
        
        # Generate regressors
        regressors = generate_regressors_from_norm(locs_norm, config)
        
        session_data_cache[session] = {
            'neurons_norm': neurons_norm,
            'regressors': regressors,
            'num_trials': neurons_norm.shape[1]
        }
    
    # Create phase mask
    bin_indices = np.arange(config.total_bins)
    phases_per_bin = get_goal_progress_from_bin(bin_indices, config)
    
    # Cross-validation: leave-one-session-out
    cv_coeffs = np.zeros((num_neurons, num_sessions, config.num_regressors))
    cv_correlations = np.zeros((num_neurons, num_sessions))
    cv_correlations_nonzero = np.zeros((num_neurons, num_sessions))
    cv_coeffs[:] = np.nan
    cv_correlations[:] = np.nan
    cv_correlations_nonzero[:] = np.nan
    
    for test_idx, test_session in enumerate(valid_sessions):
        if verbose:
            print(f"CV fold {test_idx + 1}/{num_sessions}: testing on session {test_session}")
        
        train_sessions = [s for s in valid_sessions if s != test_session]
        
        # Compute preferred phases from training data
        train_neurons_stacked = np.concatenate(
            [session_data_cache[s]['neurons_norm'] for s in train_sessions],
            axis=1  # Concatenate along trials
        )
        pref_phases, _ = compute_preferred_phase_from_norm(train_neurons_stacked, config)
        
        # Get test data
        test_cache = session_data_cache[test_session]
        test_neurons = test_cache['neurons_norm']
        test_regressors = test_cache['regressors']
        test_num_trials = test_cache['num_trials']
        
        # Fit and predict for each STATE-TUNED neuron only
        for neuron_idx in neurons_to_analyze:
            pref_phase = pref_phases[neuron_idx]
            phase_mask = phases_per_bin == pref_phase
            
            # Concatenate training data for this neuron
            y_train_all = []
            X_train_all = []
            
            for train_session in train_sessions:
                cache = session_data_cache[train_session]
                neurons = cache['neurons_norm']
                regressors = cache['regressors']
                
                for trial_idx in range(cache['num_trials']):
                    y = neurons[neuron_idx, trial_idx, phase_mask]
                    X = regressors[trial_idx, phase_mask, :]
                    y_train_all.append(y)
                    X_train_all.append(X)
            
            y_train = np.concatenate(y_train_all)
            X_train = np.vstack(X_train_all)
            
            # Fit model
            coeffs = fit_elasticnet_regression(X_train, y_train, config)
            cv_coeffs[neuron_idx, test_idx] = coeffs
            
            if np.any(np.isnan(coeffs)):
                continue
            
            # Predict on test set
            y_test_all = []
            X_test_all = []
            
            for trial_idx in range(test_num_trials):
                y = test_neurons[neuron_idx, trial_idx, phase_mask]
                X = test_regressors[trial_idx, phase_mask, :]
                y_test_all.append(y)
                X_test_all.append(X)
            
            y_test = np.concatenate(y_test_all)
            X_test = np.vstack(X_test_all)
            
            # Predicted activity
            y_pred = X_test @ coeffs
            
            # Correlation between predicted and actual
            valid = ~np.isnan(y_test) & ~np.isnan(y_pred)
            if np.sum(valid) > 3:
                corr, _ = stats.pearsonr(y_test[valid], y_pred[valid])
                cv_correlations[neuron_idx, test_idx] = corr
            
            # Non-zero lag analysis: exclude lag 0 coefficients
            coeffs_nonzero = coeffs.copy()
            # Lag 0 indices: every num_lags-th coefficient starting from 0
            lag0_indices = np.arange(0, config.num_regressors, config.num_lags)
            coeffs_nonzero[lag0_indices] = 0
            
            y_pred_nonzero = X_test @ coeffs_nonzero
            valid = ~np.isnan(y_test) & ~np.isnan(y_pred_nonzero)
            if np.sum(valid) > 3:
                corr, _ = stats.pearsonr(y_test[valid], y_pred_nonzero[valid])
                cv_correlations_nonzero[neuron_idx, test_idx] = corr
    
    # Compute means across sessions
    mean_correlations = np.nanmean(cv_correlations, axis=1)
    mean_correlations_nonzero = np.nanmean(cv_correlations_nonzero, axis=1)
    
    results = {
        'cv_coeffs': cv_coeffs,
        'cv_correlations': cv_correlations,
        'cv_correlations_nonzero': cv_correlations_nonzero,
        'mean_correlations': mean_correlations,
        'mean_correlations_nonzero': mean_correlations_nonzero,
        'valid_sessions': valid_sessions,
        'num_neurons': num_neurons,
        'num_state_tuned': len(neurons_to_analyze),
        'state_tuned_mask': state_tuned_mask,
        'state_tuning_results': state_tuning_results,
        'neurons_analyzed': neurons_to_analyze,
        'num_sessions': num_sessions
    }
    
    return results


def run_cross_validated_regression_tuning_curves(
    data_dic,
    mouse_recday,
    config,
    valid_sessions=None,
    state_tuned_mask=None,
    state_tuning_p_threshold=0.05,
    require_state_tuning=True,
    filter_duplicate_tasks=True,
    verbose=True
):
    """
    Run leave-one-out cross-validated regression with tuning curve correlations.
    
    Instead of bin-by-bin correlation, this version:
    1. Predicts activity at each bin for all trials
    2. Averages predicted activity across trials to get a 360-bin predicted tuning curve
    3. Compares to the actual 360-bin tuning curve from the held-out session
    
    Following El-Gaby et al. 2024, only state-tuned neurons are analyzed.
    
    Parameters
    ----------
    data_dic : dict
        Data dictionary
    mouse_recday : str
        Recording day identifier
    config : RegressionConfig
        Configuration object
    valid_sessions : list
        List of valid session indices (if None, use all sessions)
    state_tuned_mask : ndarray, optional
        Pre-computed boolean mask of state-tuned neurons.
    state_tuning_p_threshold : float
        P-value threshold for state tuning identification (default 0.05)
    require_state_tuning : bool
        If True (default), only analyze state-tuned neurons.
    filter_duplicate_tasks : bool
        If True (default), exclude sessions with duplicate task structures.
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Cross-validation results including tuning curve correlations
    """
    if valid_sessions is None:
        valid_sessions = list(data_dic[mouse_recday].keys())
    
    # Filter to sessions with required normalized data
    valid_sessions = [
        s for s in valid_sessions
        if ('Neurons_norm' in data_dic[mouse_recday][s] and 
            'Locs_norm' in data_dic[mouse_recday][s] and
            data_dic[mouse_recday][s]['Neurons_norm'] is not None and
            data_dic[mouse_recday][s]['Locs_norm'] is not None)
    ]
    
    # Filter out duplicate task structures (keep first occurrence)
    if filter_duplicate_tasks:
        sessions_before = len(valid_sessions)
        valid_sessions = filter_unique_task_sessions(data_dic, mouse_recday, valid_sessions)
        if verbose and len(valid_sessions) < sessions_before:
            print(f"Filtered {sessions_before - len(valid_sessions)} duplicate task sessions")
    
    if len(valid_sessions) < 2:
        print(f"Not enough valid sessions with normalized data for {mouse_recday}")
        return None
    
    num_sessions = len(valid_sessions)
    
    # Get number of neurons from first valid session
    first_session = valid_sessions[0]
    num_neurons = data_dic[mouse_recday][first_session]['Neurons_norm'].shape[0]
    
    # Identify state-tuned neurons (El-Gaby et al. 2024 method)
    if require_state_tuning:
        if state_tuned_mask is None:
            if verbose:
                print(f"Identifying state-tuned neurons (p < {state_tuning_p_threshold})...")
            state_tuned_mask, state_tuning_results = identify_state_tuned_neurons_across_sessions(
                data_dic, mouse_recday, config, 
                valid_sessions=valid_sessions,
                p_threshold=state_tuning_p_threshold,
                require_all_sessions=False
            )
            if verbose:
                n_tuned = np.sum(state_tuned_mask)
                print(f"  Found {n_tuned}/{num_neurons} state-tuned neurons ({100*n_tuned/num_neurons:.1f}%)")
        else:
            state_tuning_results = None
        
        neurons_to_analyze = np.where(state_tuned_mask)[0]
        if len(neurons_to_analyze) == 0:
            print("No state-tuned neurons found!")
            return None
    else:
        state_tuned_mask = np.ones(num_neurons, dtype=bool)
        state_tuning_results = None
        neurons_to_analyze = np.arange(num_neurons)
    
    if verbose:
        print(f"Analyzing {len(neurons_to_analyze)}/{num_neurons} neurons, {num_sessions} sessions")
        print("Computing tuning curve correlations...")
    
    # Pre-compute regressors for all sessions
    session_data_cache = {}
    for session in valid_sessions:
        neurons_norm = data_dic[mouse_recday][session]['Neurons_norm']
        locs_norm = data_dic[mouse_recday][session]['Locs_norm']
        
        # Generate regressors
        regressors = generate_regressors_from_norm(locs_norm, config)
        
        session_data_cache[session] = {
            'neurons_norm': neurons_norm,
            'regressors': regressors,
            'num_trials': neurons_norm.shape[1]
        }
    
    # Create phase mask
    bin_indices = np.arange(config.total_bins)
    phases_per_bin = get_goal_progress_from_bin(bin_indices, config)
    
    # Cross-validation storage
    cv_coeffs = np.zeros((num_neurons, num_sessions, config.num_regressors))
    cv_tuning_corrs = np.zeros((num_neurons, num_sessions))
    cv_tuning_corrs_nonzero = np.zeros((num_neurons, num_sessions))
    cv_actual_tuning = np.zeros((num_neurons, num_sessions, config.total_bins))
    cv_predicted_tuning = np.zeros((num_neurons, num_sessions, config.total_bins))
    cv_predicted_tuning_nonzero = np.zeros((num_neurons, num_sessions, config.total_bins))
    
    cv_coeffs[:] = np.nan
    cv_tuning_corrs[:] = np.nan
    cv_tuning_corrs_nonzero[:] = np.nan
    cv_actual_tuning[:] = np.nan
    cv_predicted_tuning[:] = np.nan
    cv_predicted_tuning_nonzero[:] = np.nan
    
    for test_idx, test_session in enumerate(valid_sessions):
        if verbose:
            print(f"CV fold {test_idx + 1}/{num_sessions}: testing on session {test_session}")
        
        train_sessions = [s for s in valid_sessions if s != test_session]
        
        # Compute preferred phases from training data
        train_neurons_stacked = np.concatenate(
            [session_data_cache[s]['neurons_norm'] for s in train_sessions],
            axis=1
        )
        pref_phases, _ = compute_preferred_phase_from_norm(train_neurons_stacked, config)
        
        # Get test data
        test_cache = session_data_cache[test_session]
        test_neurons = test_cache['neurons_norm']  # (num_neurons, num_trials, 360)
        test_regressors = test_cache['regressors']  # (num_trials, 360, num_regressors)
        test_num_trials = test_cache['num_trials']
        
        # Fit and predict for each neuron (only state-tuned neurons if filtering)
        for neuron_idx in neurons_to_analyze:
            pref_phase = pref_phases[neuron_idx]
            phase_mask = phases_per_bin == pref_phase
            
            # Concatenate training data for this neuron (only pref phase)
            y_train_all = []
            X_train_all = []
            
            for train_session in train_sessions:
                cache = session_data_cache[train_session]
                neurons = cache['neurons_norm']
                regressors = cache['regressors']
                
                for trial_idx in range(cache['num_trials']):
                    y = neurons[neuron_idx, trial_idx, phase_mask]
                    X = regressors[trial_idx, phase_mask, :]
                    y_train_all.append(y)
                    X_train_all.append(X)
            
            y_train = np.concatenate(y_train_all)
            X_train = np.vstack(X_train_all)
            
            # Fit model
            coeffs = fit_elasticnet_regression(X_train, y_train, config)
            cv_coeffs[neuron_idx, test_idx] = coeffs
            
            if np.any(np.isnan(coeffs)):
                continue
            
            # Predict on ALL bins of test set (not just pref phase)
            # to get a full 360-bin predicted tuning curve
            predicted_trials = np.zeros((test_num_trials, config.total_bins))
            predicted_trials_nonzero = np.zeros((test_num_trials, config.total_bins))
            
            # Non-zero lag coefficients
            coeffs_nonzero = coeffs.copy()
            lag0_indices = np.arange(0, config.num_regressors, config.num_lags)
            coeffs_nonzero[lag0_indices] = 0
            
            for trial_idx in range(test_num_trials):
                X_trial = test_regressors[trial_idx]  # (360, num_regressors)
                predicted_trials[trial_idx] = X_trial @ coeffs
                predicted_trials_nonzero[trial_idx] = X_trial @ coeffs_nonzero
            
            # Average across trials to get tuning curves
            predicted_tuning = np.nanmean(predicted_trials, axis=0)  # (360,)
            predicted_tuning_nonzero = np.nanmean(predicted_trials_nonzero, axis=0)
            actual_tuning = np.nanmean(test_neurons[neuron_idx], axis=0)  # (360,)
            
            # Store tuning curves
            cv_actual_tuning[neuron_idx, test_idx] = actual_tuning
            cv_predicted_tuning[neuron_idx, test_idx] = predicted_tuning
            cv_predicted_tuning_nonzero[neuron_idx, test_idx] = predicted_tuning_nonzero
            
            # Correlation between predicted and actual tuning curves
            valid = ~np.isnan(actual_tuning) & ~np.isnan(predicted_tuning)
            if np.sum(valid) > 10:
                corr, _ = stats.pearsonr(actual_tuning[valid], predicted_tuning[valid])
                cv_tuning_corrs[neuron_idx, test_idx] = corr
            
            valid_nz = ~np.isnan(actual_tuning) & ~np.isnan(predicted_tuning_nonzero)
            if np.sum(valid_nz) > 10:
                corr_nz, _ = stats.pearsonr(actual_tuning[valid_nz], predicted_tuning_nonzero[valid_nz])
                cv_tuning_corrs_nonzero[neuron_idx, test_idx] = corr_nz
    
    # Compute means across sessions
    mean_tuning_corrs = np.nanmean(cv_tuning_corrs, axis=1)
    mean_tuning_corrs_nonzero = np.nanmean(cv_tuning_corrs_nonzero, axis=1)
    
    results = {
        'cv_coeffs': cv_coeffs,
        'cv_tuning_correlations': cv_tuning_corrs,
        'cv_tuning_correlations_nonzero': cv_tuning_corrs_nonzero,
        'mean_tuning_correlations': mean_tuning_corrs,
        'mean_tuning_correlations_nonzero': mean_tuning_corrs_nonzero,
        'cv_actual_tuning': cv_actual_tuning,
        'cv_predicted_tuning': cv_predicted_tuning,
        'cv_predicted_tuning_nonzero': cv_predicted_tuning_nonzero,
        'valid_sessions': valid_sessions,
        'num_neurons': num_neurons,
        'num_sessions': num_sessions,
        'num_state_tuned': len(neurons_to_analyze),
        'state_tuned_mask': state_tuned_mask,
        'state_tuning_results': state_tuning_results,
        'neurons_analyzed': neurons_to_analyze
    }
    
    return results


def identify_nonzero_lag_neurons(results, config):
    """
    Identify non-zero lag neurons following El-Gaby et al. 2024.

    Criterion: all three of the highest regression coefficients must be at
    intermediate lags (2–10). Lags 0 and 1 (current/near-current location) and
    lag 11 (one full state back, same location in cyclic task) are excluded.

    Parameters
    ----------
    results : dict
        Results from run_cross_validated_regression or run_cross_validated_regression_tuning_curves
    config : RegressionConfig
        Configuration object

    Returns
    -------
    nonzero_lag_mask : ndarray
        Boolean mask of non-zero lag neurons
    peak_lags : ndarray
        Lag of the single highest coefficient for each neuron
    """
    cv_coeffs = results['cv_coeffs']  # (num_neurons, num_sessions, num_regressors)
    mean_coeffs = np.nanmean(cv_coeffs, axis=1)  # (num_neurons, num_regressors)

    num_neurons = mean_coeffs.shape[0]
    num_anchors = config.num_locations * config.num_goal_progress_bins

    peak_lags = np.full(num_neurons, np.nan)
    nonzero_lag_mask = np.zeros(num_neurons, dtype=bool)

    for neuron_idx in range(num_neurons):
        neuron_coeffs = mean_coeffs[neuron_idx]

        if np.all(np.isnan(neuron_coeffs)) or np.all(neuron_coeffs == 0):
            continue

        # Reshape to (num_anchors, num_lags) and flatten to get all coefficients
        coeff_matrix = neuron_coeffs.reshape(num_anchors, config.num_lags)

        # Top 3 coefficient indices (flattened)
        flat_coeffs = coeff_matrix.flatten()
        top3_flat = np.argsort(flat_coeffs)[-3:]

        # Convert flat indices to (anchor, lag) positions
        top3_lags = top3_flat % config.num_lags

        # Record the single highest coefficient's lag
        peak_lags[neuron_idx] = top3_lags[-1]

        # El-Gaby criterion: all 3 highest coefficients must be at intermediate lags
        # Exclude lags 0, 1 (current/near-current) and lag 11 (one full state back)
        nonzero_lag_mask[neuron_idx] = np.all((top3_lags >= 2) & (top3_lags <= config.num_lags - 2))

    return nonzero_lag_mask, peak_lags


# Alias for backward compatibility
identify_sequence_neurons = identify_nonzero_lag_neurons


def plot_polar_tuning_curves(results, config, data_dic=None, mouse_recday='',
                             num_examples=9, sort_by='correlation',
                             nonzero_lag_only=False,
                             use_smoothed=True, plot_smooth_sigma=10,
                             save_path=None):
    """
    Plot polar plots comparing actual vs predicted 360-bin tuning curves.
    
    Shows all held-out sessions for each selected neuron.
    
    Parameters
    ----------
    results : dict
        Results from run_cross_validated_regression_tuning_curves
    config : RegressionConfig
        Configuration object
    data_dic : dict, optional
        Data dictionary for smoothed arrays. Required if use_smoothed=True.
    mouse_recday : str
        Recording identifier for title
    num_examples : int
        Number of example neurons to show
    sort_by : str
        'correlation' to show top neurons, 'random' for random selection
    nonzero_lag_only : bool
        If True, only show non-zero lag neurons (El-Gaby criterion: top-3 betas all at lag > 0)
    use_smoothed : bool
        If True and data_dic provided, use Smoothed_norm for actual tuning curves
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    mean_corrs = results['mean_tuning_correlations']
    cv_actual = results['cv_actual_tuning']  # (num_neurons, num_sessions, 360)
    cv_pred = results['cv_predicted_tuning']
    num_sessions = results['num_sessions']
    valid_sessions = results['valid_sessions']
    
    # Get smoothed actual tuning curves if requested
    if use_smoothed and data_dic is not None and mouse_recday:
        smoothed_actual = {}
        for sess_idx, session in enumerate(valid_sessions):
            if 'Smoothed_norm' in data_dic[mouse_recday][session]:
                smoothed_actual[sess_idx] = data_dic[mouse_recday][session]['Smoothed_norm']
            else:
                smoothed_actual[sess_idx] = None
    else:
        smoothed_actual = None
    
    # Identify non-zero lag neurons if requested
    if nonzero_lag_only:
        nz_mask, peak_lags = identify_nonzero_lag_neurons(results, config)
        valid_mask = ~np.isnan(mean_corrs) & nz_mask
        n_nz = np.sum(nz_mask & ~np.isnan(mean_corrs))
        print(f"Found {n_nz} non-zero lag neurons (top-3 betas all at lags 2–{config.num_lags - 2})")
    else:
        valid_mask = ~np.isnan(mean_corrs)
        peak_lags = None
    
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print("No valid neurons to plot!")
        return None
    
    # Select neurons
    if sort_by == 'correlation':
        sorted_indices = valid_indices[np.argsort(mean_corrs[valid_indices])[::-1]]
        selected = sorted_indices[:num_examples]
        title_suffix = "(top by tuning correlation)"
    else:
        selected = np.random.choice(valid_indices, size=min(num_examples, len(valid_indices)), replace=False)
        selected = np.sort(selected)
        title_suffix = "(random selection)"
    
    if nonzero_lag_only:
        title_suffix += " - Non-zero lag neurons (top-3 betas at lag > 0)"
    
    # Create polar angles
    theta = np.linspace(0, 2 * np.pi, config.total_bins, endpoint=False)
    theta_closed = np.append(theta, theta[0])
    
    # Layout: num_sessions columns per neuron
    ncols = num_sessions
    nrows = len(selected)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), 
                             subplot_kw=dict(projection='polar'))
    
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Normalize function
    def normalize(arr):
        arr_valid = arr[~np.isnan(arr)]
        if len(arr_valid) == 0 or np.max(arr_valid) == np.min(arr_valid):
            return arr
        return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    
    for row_idx, neuron_idx in enumerate(selected):
        mean_corr = mean_corrs[neuron_idx]
        
        for sess_idx in range(num_sessions):
            ax = axes[row_idx, sess_idx]
            
            # Get actual tuning curve (smoothed if available)
            if smoothed_actual is not None and smoothed_actual.get(sess_idx) is not None:
                actual = smoothed_actual[sess_idx][neuron_idx]
            else:
                actual = cv_actual[neuron_idx, sess_idx]
            
            # Get predicted tuning curve
            predicted = cv_pred[neuron_idx, sess_idx]
            
            # Get per-session correlation (on unsmoothed data)
            valid_bins = ~np.isnan(actual) & ~np.isnan(predicted)
            if np.sum(valid_bins) > 10:
                sess_corr, _ = stats.pearsonr(actual[valid_bins], predicted[valid_bins])
            else:
                sess_corr = np.nan

            # Circular smooth for display only (does NOT affect correlation)
            if plot_smooth_sigma is not None and plot_smooth_sigma > 0:
                from scipy.ndimage import gaussian_filter1d
                actual_plot = gaussian_filter1d(actual, sigma=plot_smooth_sigma, mode='wrap')
                predicted_plot = gaussian_filter1d(predicted, sigma=plot_smooth_sigma, mode='wrap')
            else:
                actual_plot = actual
                predicted_plot = predicted

            # Close loops for plotting
            actual_closed = np.append(actual_plot, actual_plot[0])
            pred_closed = np.append(predicted_plot, predicted_plot[0])
            
            # Normalize
            actual_norm = normalize(actual_closed)
            pred_norm = normalize(pred_closed)
            
            # Plot
            ax.plot(theta_closed, actual_norm, 'b-', linewidth=2, label='Actual')
            ax.plot(theta_closed, pred_norm, 'r-', linewidth=2, label='Predicted')
            ax.fill(theta_closed, actual_norm, alpha=0.2, color='blue')
            ax.fill(theta_closed, pred_norm, alpha=0.2, color='red')
            
            # Add state boundaries
            for i in range(4):
                ax.axvline(i * np.pi / 2, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            
            ax.set_ylim(0, 1.1)
            ax.set_rticks([0.5, 1.0])
            
            # Title
            session_label = valid_sessions[sess_idx]
            if row_idx == 0:
                ax.set_title(f'Session {session_label}\nr = {sess_corr:.3f}', fontsize=10)
            else:
                ax.set_title(f'r = {sess_corr:.3f}', fontsize=10)
            
            # Add neuron info on first column
            if sess_idx == 0:
                peak_lag_str = f", peak lag={int(peak_lags[neuron_idx])}" if peak_lags is not None else ""
                ax.annotate(f'N{neuron_idx}\nmean r={mean_corr:.3f}{peak_lag_str}', 
                           xy=(0.02, 0.98), xycoords='axes fraction',
                           fontsize=9, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Legend on last column
            if sess_idx == num_sessions - 1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle(f'Actual vs Predicted Tuning Curves: {mouse_recday}\n{title_suffix}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

    return fig, selected


def plot_tuning_correlation_distribution(results, config, mouse_recday='', save_path=None):
    """
    Plot distribution of tuning curve correlations (predicted vs actual 360-bin curves).

    Parameters
    ----------
    results : dict
        Results from run_cross_validated_regression_tuning_curves
    config : RegressionConfig, optional
        Required to identify non-zero lag neurons for filtered panel
    mouse_recday : str
        Recording identifier for title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt

    mean_corrs = results['mean_tuning_correlations']
    mean_corrs_nz_all = results['mean_tuning_correlations_nonzero']
    cv_corrs = results['cv_tuning_correlations']
    num_neurons = results['num_neurons']
    num_sessions = results['num_sessions']

    # Filter non-zero lag correlations to only non-zero lag neurons
    nz_mask, _ = identify_nonzero_lag_neurons(results, config)
    mean_corrs_nz = np.where(nz_mask, mean_corrs_nz_all, np.nan)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Overall distribution (all lags)
    ax1 = axes[0, 0]
    corrs_valid = mean_corrs[~np.isnan(mean_corrs)]
    ax1.hist(corrs_valid, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    mean_r = np.mean(corrs_valid)
    ax1.axvline(mean_r, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_r:.3f}')
    ax1.set_xlabel('Tuning curve correlation', fontsize=11)
    ax1.set_ylabel('Number of neurons', fontsize=11)
    ax1.set_title('All-lag tuning curve prediction\n(mean across sessions)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    
    if len(corrs_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(corrs_valid, 0)
        ax1.text(0.95, 0.95, f'n = {len(corrs_valid)}\nt = {t_stat:.2f}\np = {p_val:.2e}', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Non-zero lag distribution
    ax2 = axes[0, 1]
    corrs_nz_valid = mean_corrs_nz[~np.isnan(mean_corrs_nz)]
    ax2.hist(corrs_nz_valid, bins=40, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=2)
    mean_r_nz = np.mean(corrs_nz_valid)
    ax2.axvline(mean_r_nz, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_r_nz:.3f}')
    ax2.set_xlabel('Tuning curve correlation', fontsize=11)
    ax2.set_ylabel('Number of neurons', fontsize=11)
    ax2.set_title('Non-zero lag tuning curve prediction\n(mean across sessions)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    
    if len(corrs_nz_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(corrs_nz_valid, 0)
        ax2.text(0.95, 0.95, f'n = {len(corrs_nz_valid)}\nt = {t_stat:.2f}\np = {p_val:.2e}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. All-lag vs Non-zero lag scatter
    ax3 = axes[0, 2]
    valid_both = ~np.isnan(mean_corrs) & ~np.isnan(mean_corrs_nz)
    if np.sum(valid_both) > 0:
        ax3.scatter(mean_corrs[valid_both], mean_corrs_nz[valid_both], 
                   alpha=0.5, s=20, c='teal')
        ax3.plot([-1, 1], [-1, 1], 'k--', linewidth=1)
        ax3.set_xlabel('All-lag tuning corr', fontsize=11)
        ax3.set_ylabel('Non-zero lag tuning corr', fontsize=11)
        ax3.set_title('All-lag vs Non-zero lag', fontsize=12, fontweight='bold')
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_aspect('equal')
    
    # 4. Correlations by session
    ax4 = axes[1, 0]
    valid_sessions = results['valid_sessions']
    session_means = []
    session_sems = []
    for ses_idx in range(num_sessions):
        ses_corrs = cv_corrs[:, ses_idx]
        ses_valid = ses_corrs[~np.isnan(ses_corrs)]
        session_means.append(np.mean(ses_valid) if len(ses_valid) > 0 else np.nan)
        session_sems.append(np.std(ses_valid) / np.sqrt(len(ses_valid)) if len(ses_valid) > 0 else np.nan)
    
    x_pos = np.arange(num_sessions)
    ax4.bar(x_pos, session_means, yerr=session_sems, color='steelblue', alpha=0.7, capsize=5)
    ax4.axhline(0, color='black', linestyle='--', linewidth=1)
    ax4.axhline(mean_r, color='red', linestyle='-', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Held-out session', fontsize=11)
    ax4.set_ylabel('Mean tuning correlation', fontsize=11)
    ax4.set_title('Tuning correlation by session', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Ses {s}' for s in valid_sessions], rotation=45)
    
    # 5. Histogram of % neurons with positive correlation
    ax5 = axes[1, 1]
    prop_positive = np.sum(corrs_valid > 0) / len(corrs_valid) * 100
    prop_sig_pos = np.sum(corrs_valid > 0.3) / len(corrs_valid) * 100
    prop_sig_neg = np.sum(corrs_valid < -0.3) / len(corrs_valid) * 100
    
    categories = ['r > 0', 'r > 0.3', 'r < -0.3']
    values = [prop_positive, prop_sig_pos, prop_sig_neg]
    colors = ['lightgreen', 'darkgreen', 'salmon']
    ax5.bar(categories, values, color=colors, edgecolor='black')
    ax5.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax5.set_ylabel('% of neurons', fontsize=11)
    ax5.set_title('Proportion with significant correlation', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 100)
    
    # 6. Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
    Tuning Curve Prediction Summary
    ================================
    Recording: {mouse_recday}

    Neurons: {num_neurons}
    Valid neurons: {len(corrs_valid)}
    Sessions: {num_sessions}

    All-lag prediction:
      Mean r = {mean_r:.4f}
      % positive = {prop_positive:.1f}%
      % r > 0.3 = {prop_sig_pos:.1f}%

    Non-zero lag prediction:
      n neurons = {len(corrs_nz_valid)}
      Mean r = {mean_r_nz:.4f}

    Method:
      Predicted 360-bin tuning curve from
      regression coefficients, compared to
      actual tuning curve in held-out session.
    """
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle(f'Tuning Curve Correlation Analysis: {mouse_recday}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def identify_nonzero_lag_neurons_from_coeffs(coeffs, config, num_top=3, lag_threshold=1):
    """
    Identify neurons whose top coefficients are at non-zero lag (from coeffs array).
    
    From the paper: "For non-zero-lag neurons, we only used state-tuned neurons 
    with all of the three highest regression coefficient values at non-zero lag 
    from an anchor."
    
    Parameters
    ----------
    coeffs : ndarray
        Shape (num_neurons, num_regressors) or (num_neurons, num_sessions, num_regressors)
    config : RegressionConfig
        Configuration object
    num_top : int
        Number of top coefficients to check
    lag_threshold : int
        Minimum lag to be considered "non-zero" (1 = 30°, 3 = 90° for 12 lags)
        
    Returns
    -------
    is_nonzero_lag : ndarray
        Shape (num_neurons,) - boolean mask for non-zero lag neurons
    """
    if len(coeffs.shape) == 3:
        # Average across sessions
        coeffs = np.nanmean(coeffs, axis=1)
    
    num_neurons = coeffs.shape[0]
    is_nonzero_lag = np.zeros(num_neurons, dtype=bool)
    
    # Reshape coefficients to identify lag positions
    # Shape: (num_locations, num_phases, num_lags)
    reshaped_size = (config.num_locations, config.num_goal_progress_bins, config.num_lags)
    
    for neuron_idx in range(num_neurons):
        neuron_coeffs = coeffs[neuron_idx]
        
        if np.all(np.isnan(neuron_coeffs)):
            continue
        
        # Get top N coefficient indices
        valid_coeffs = np.where(~np.isnan(neuron_coeffs), neuron_coeffs, -np.inf)
        top_indices = np.argsort(valid_coeffs)[-num_top:][::-1]
        
        # Check if all top coefficients are at non-zero lag
        all_nonzero = True
        for idx in top_indices:
            # Get lag position
            lag_pos = idx % config.num_lags
            if lag_pos < lag_threshold:
                all_nonzero = False
                break
        
        is_nonzero_lag[neuron_idx] = all_nonzero
    
    return is_nonzero_lag


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_coefficient_matrix(coeffs, config, neuron_idx=0, ax=None):
    """
    Plot regression coefficients as a matrix (anchor × lag).
    
    Parameters
    ----------
    coeffs : ndarray
        Coefficient array
    config : RegressionConfig
        Configuration object
    neuron_idx : int
        Neuron index to plot
    ax : matplotlib axis
        Axis to plot on
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    neuron_coeffs = coeffs[neuron_idx] if len(coeffs.shape) > 1 else coeffs
    
    # Reshape to (num_anchors, num_lags)
    num_anchors = config.num_locations * config.num_goal_progress_bins
    coeff_matrix = neuron_coeffs.reshape(num_anchors, config.num_lags)
    
    im = ax.imshow(coeff_matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel('Lag (task space)')
    ax.set_ylabel('Anchor (location × phase)')
    ax.set_title(f'Regression coefficients for neuron {neuron_idx}')
    plt.colorbar(im, ax=ax)
    
    return ax


def plot_example_betas(results, config, neuron_indices=None, num_examples=6, save_path=None):
    """
    Plot example beta (coefficient) matrices for multiple neurons.
    
    Shows the anchor × lag structure of regression coefficients.
    
    Parameters
    ----------
    results : dict
        Results from run_cross_validated_regression
    config : RegressionConfig
        Configuration object
    neuron_indices : list, optional
        Specific neuron indices to plot. If None, selects neurons with highest mean correlations.
    num_examples : int
        Number of example neurons to show (if neuron_indices not specified)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Get coefficients (average across CV folds)
    coeffs = np.nanmean(results['cv_coeffs'], axis=1)  # Shape: (num_neurons, num_regressors)
    # Handle both regular and tuning curve results
    if 'mean_correlations' in results:
        corrs = results['mean_correlations']
    else:
        corrs = results['mean_tuning_correlations']
    
    # Select neurons to plot
    if neuron_indices is None:
        # Select neurons with highest correlations (that have valid coefficients)
        valid_mask = ~np.isnan(corrs) & ~np.all(np.isnan(coeffs), axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            print("No valid neurons to plot!")
            return None
        
        # Sort by correlation and take top ones
        sorted_indices = valid_indices[np.argsort(corrs[valid_indices])[::-1]]
        neuron_indices = sorted_indices[:num_examples]
    
    num_plots = len(neuron_indices)
    ncols = min(3, num_plots)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if num_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    num_anchors = config.num_locations * config.num_goal_progress_bins
    
    for i, neuron_idx in enumerate(neuron_indices):
        ax = axes[i]
        
        neuron_coeffs = coeffs[neuron_idx]
        neuron_corr = corrs[neuron_idx]
        
        # Reshape to anchor × lag
        coeff_matrix = neuron_coeffs.reshape(num_anchors, config.num_lags)
        
        # Plot
        im = ax.imshow(coeff_matrix, aspect='auto', cmap='hot', interpolation='nearest')
        
        # Add location separators
        for loc in range(1, config.num_locations):
            ax.axhline(loc * config.num_goal_progress_bins - 0.5, color='white', 
                      linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Labels
        ax.set_xlabel('Lag (0=current → 11=oldest)', fontsize=10)
        ax.set_ylabel('Anchor', fontsize=10)
        ax.set_title(f'Neuron {neuron_idx}\nr = {neuron_corr:.3f}', fontsize=11)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
        # Y-axis labels for locations
        yticks = [i * config.num_goal_progress_bins + 1 for i in range(config.num_locations)]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'L{i+1}' for i in range(config.num_locations)], fontsize=8)
        
        # X-axis
        ax.set_xticks([0, 3, 6, 9, 11])
        ax.set_xticklabels(['0', '3', '6', '9', '11'], fontsize=8)
    
    # Hide unused axes
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Beta Coefficient Matrices (Anchor × Lag)\n'
                 f'{config.num_locations} locations × {config.num_goal_progress_bins} phases = {num_anchors} anchors, '
                 f'{config.num_lags} lags', 
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print interpretation
    print("\n" + "="*60)
    print("INTERPRETING BETA MATRICES")
    print("="*60)
    print("""
    Each matrix shows coefficients for one neuron:
    
    Y-axis (Anchors): 27 total = 9 locations × 3 goal-progress phases
      - L1, L2, ... L9 = spatial locations (nodes 1-9)
      - Each location has 3 rows for phases 0, 1, 2
      
    X-axis (Lags): 12 lags in task space
      - Lag 0 = neuron fires AT the anchor location/phase
      - Lag 1 = neuron fires 1 phase AFTER visiting the anchor
      - Lag 11 = neuron fires 11 phases after (almost full trial later)
    
    Interpretation:
      - Bright colors at lag 0 → direct place/phase tuning
      - Bright colors at lag > 0 → non-zero lag neuron / prospective coding
      - A neuron with peaks at lag 3-6 fires in the FUTURE relative 
        to when it visited that anchor point
    """)
    
    return fig


def plot_population_beta_summary(results, config, save_path=None):
    """
    Plot population summary of beta coefficients.
    
    Parameters
    ----------
    results : dict
        Results from run_cross_validated_regression
    config : RegressionConfig
        Configuration object
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Get coefficients
    coeffs = np.nanmean(results['cv_coeffs'], axis=1)  # (num_neurons, num_regressors)
    # Handle both regular and tuning curve results
    if 'mean_correlations' in results:
        corrs = results['mean_correlations']
    else:
        corrs = results['mean_tuning_correlations']
    
    # Only use neurons with valid data
    valid_mask = ~np.isnan(corrs) & ~np.all(np.isnan(coeffs), axis=1)
    valid_coeffs = coeffs[valid_mask]
    valid_corrs = corrs[valid_mask]
    
    num_anchors = config.num_locations * config.num_goal_progress_bins
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Mean coefficient matrix across all neurons
    ax1 = axes[0, 0]
    mean_coeffs = np.nanmean(valid_coeffs, axis=0)
    mean_matrix = mean_coeffs.reshape(num_anchors, config.num_lags)
    im1 = ax1.imshow(mean_matrix, aspect='auto', cmap='hot')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Anchor')
    ax1.set_title('Mean coefficients\n(all neurons)')
    plt.colorbar(im1, ax=ax1)
    for loc in range(1, config.num_locations):
        ax1.axhline(loc * config.num_goal_progress_bins - 0.5, color='white', 
                   linestyle='-', linewidth=0.3)
    
    # 2. Mean coefficient matrix for TOP neurons (high correlation)
    ax2 = axes[0, 1]
    top_n = min(50, len(valid_corrs) // 4)
    if top_n > 0:
        top_indices = np.argsort(valid_corrs)[-top_n:]
        top_mean = np.nanmean(valid_coeffs[top_indices], axis=0)
        top_matrix = top_mean.reshape(num_anchors, config.num_lags)
        im2 = ax2.imshow(top_matrix, aspect='auto', cmap='hot')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Anchor')
        ax2.set_title(f'Mean coefficients\n(top {top_n} neurons by r)')
        plt.colorbar(im2, ax=ax2)
        for loc in range(1, config.num_locations):
            ax2.axhline(loc * config.num_goal_progress_bins - 0.5, color='white', 
                       linestyle='-', linewidth=0.3)
    
    # 3. Coefficient magnitude by lag
    ax3 = axes[0, 2]
    lag_means = []
    lag_sems = []
    for lag in range(config.num_lags):
        lag_indices = np.arange(lag, config.num_regressors, config.num_lags)
        lag_coeffs = valid_coeffs[:, lag_indices].flatten()
        lag_coeffs = lag_coeffs[~np.isnan(lag_coeffs)]
        lag_means.append(np.mean(lag_coeffs))
        lag_sems.append(np.std(lag_coeffs) / np.sqrt(len(lag_coeffs)))
    
    ax3.bar(range(config.num_lags), lag_means, yerr=lag_sems, 
            color='steelblue', alpha=0.7, capsize=3)
    ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Lag 0')
    ax3.set_xlabel('Lag in task space')
    ax3.set_ylabel('Mean coefficient')
    ax3.set_title('Coefficient by lag\n(population average)')
    ax3.legend()
    
    # 4. Coefficient magnitude by location
    ax4 = axes[1, 0]
    loc_means = []
    loc_sems = []
    for loc in range(config.num_locations):
        loc_start = loc * config.num_goal_progress_bins * config.num_lags
        loc_end = (loc + 1) * config.num_goal_progress_bins * config.num_lags
        loc_coeffs = valid_coeffs[:, loc_start:loc_end].flatten()
        loc_coeffs = loc_coeffs[~np.isnan(loc_coeffs)]
        loc_means.append(np.mean(loc_coeffs))
        loc_sems.append(np.std(loc_coeffs) / np.sqrt(len(loc_coeffs)))
    
    ax4.bar(range(config.num_locations), loc_means, yerr=loc_sems,
            color='teal', alpha=0.7, capsize=3)
    ax4.set_xlabel('Location')
    ax4.set_ylabel('Mean coefficient')
    ax4.set_title('Coefficient by location\n(population average)')
    ax4.set_xticks(range(config.num_locations))
    ax4.set_xticklabels([f'L{i+1}' for i in range(config.num_locations)])
    
    # 5. Coefficient magnitude by phase
    ax5 = axes[1, 1]
    phase_means = []
    phase_sems = []
    for phase in range(config.num_goal_progress_bins):
        phase_indices = []
        for loc in range(config.num_locations):
            anchor_idx = loc * config.num_goal_progress_bins + phase
            start_idx = anchor_idx * config.num_lags
            phase_indices.extend(range(start_idx, start_idx + config.num_lags))
        phase_coeffs = valid_coeffs[:, phase_indices].flatten()
        phase_coeffs = phase_coeffs[~np.isnan(phase_coeffs)]
        phase_means.append(np.mean(phase_coeffs))
        phase_sems.append(np.std(phase_coeffs) / np.sqrt(len(phase_coeffs)))
    
    ax5.bar(range(config.num_goal_progress_bins), phase_means, yerr=phase_sems,
            color='coral', alpha=0.7, capsize=3)
    ax5.set_xlabel('Goal progress phase')
    ax5.set_ylabel('Mean coefficient')
    ax5.set_title('Coefficient by phase\n(population average)')
    ax5.set_xticks(range(config.num_goal_progress_bins))
    ax5.set_xticklabels(['Early', 'Mid', 'Late'])
    
    # 6. Distribution of peak lags
    ax6 = axes[1, 2]
    peak_lags = []
    for neuron_coeffs in valid_coeffs:
        coeff_matrix = neuron_coeffs.reshape(num_anchors, config.num_lags)
        # Find the lag with maximum coefficient (across all anchors)
        max_idx = np.nanargmax(neuron_coeffs)
        peak_lag = max_idx % config.num_lags
        peak_lags.append(peak_lag)
    
    ax6.hist(peak_lags, bins=np.arange(-0.5, config.num_lags + 0.5, 1),
             color='purple', alpha=0.7, edgecolor='black')
    ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='Lag 0')
    ax6.set_xlabel('Peak lag')
    ax6.set_ylabel('Number of neurons')
    ax6.set_title('Distribution of peak lags')
    ax6.legend()
    
    plt.suptitle('Population Beta Coefficient Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_prediction_correlations(results, mouse_recday='', save_path=None):
    """
    Plot the distribution of correlations between predicted and actual 
    task maps in held-out sessions.
    
    Parameters
    ----------
    results : dict
        Results from run_cross_validated_regression
    mouse_recday : str
        Recording identifier for title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    cv_corrs = results['cv_correlations']  # Shape: (num_neurons, num_sessions)
    cv_corrs_nz = results['cv_correlations_nonzero']
    mean_corrs = results['mean_correlations']
    mean_corrs_nz = results['mean_correlations_nonzero']
    valid_sessions = results['valid_sessions']
    num_neurons = results['num_neurons']
    num_sessions = results['num_sessions']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Overall distribution of mean correlations (all lags)
    ax1 = axes[0, 0]
    corrs_valid = mean_corrs[~np.isnan(mean_corrs)]
    ax1.hist(corrs_valid, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    mean_r = np.mean(corrs_valid)
    ax1.axvline(mean_r, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_r:.3f}')
    ax1.set_xlabel('Correlation (predicted vs actual)', fontsize=11)
    ax1.set_ylabel('Number of neurons', fontsize=11)
    ax1.set_title('All-lag prediction\n(mean across held-out sessions)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # t-test against 0
    if len(corrs_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(corrs_valid, 0)
        ax1.text(0.05, 0.95, f'n = {len(corrs_valid)}\nt = {t_stat:.2f}\np = {p_val:.2e}', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Overall distribution (non-zero lag only)
    ax2 = axes[0, 1]
    corrs_nz_valid = mean_corrs_nz[~np.isnan(mean_corrs_nz)]
    ax2.hist(corrs_nz_valid, bins=40, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=2)
    mean_r_nz = np.mean(corrs_nz_valid)
    ax2.axvline(mean_r_nz, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_r_nz:.3f}')
    ax2.set_xlabel('Correlation (predicted vs actual)', fontsize=11)
    ax2.set_ylabel('Number of neurons', fontsize=11)
    ax2.set_title('Non-zero lag prediction\n(mean across held-out sessions)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    
    if len(corrs_nz_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(corrs_nz_valid, 0)
        ax2.text(0.05, 0.95, f'n = {len(corrs_nz_valid)}\nt = {t_stat:.2f}\np = {p_val:.2e}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Comparison: all lag vs non-zero lag
    ax3 = axes[0, 2]
    valid_both = ~np.isnan(mean_corrs) & ~np.isnan(mean_corrs_nz)
    if np.sum(valid_both) > 0:
        ax3.scatter(mean_corrs[valid_both], mean_corrs_nz[valid_both], 
                   alpha=0.5, s=20, c='teal')
        ax3.plot([-1, 1], [-1, 1], 'k--', linewidth=1)
        ax3.set_xlabel('All-lag correlation', fontsize=11)
        ax3.set_ylabel('Non-zero lag correlation', fontsize=11)
        ax3.set_title('All-lag vs Non-zero lag', fontsize=12, fontweight='bold')
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_aspect('equal')
        
        # Paired t-test
        paired_t, paired_p = stats.ttest_rel(mean_corrs[valid_both], mean_corrs_nz[valid_both])
        ax3.text(0.05, 0.95, f'Paired t = {paired_t:.2f}\np = {paired_p:.2e}', 
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Correlations by held-out session
    ax4 = axes[1, 0]
    session_means = []
    session_sems = []
    for ses_idx in range(num_sessions):
        ses_corrs = cv_corrs[:, ses_idx]
        ses_corrs_valid = ses_corrs[~np.isnan(ses_corrs)]
        session_means.append(np.mean(ses_corrs_valid) if len(ses_corrs_valid) > 0 else np.nan)
        session_sems.append(np.std(ses_corrs_valid) / np.sqrt(len(ses_corrs_valid)) if len(ses_corrs_valid) > 0 else np.nan)
    
    x_pos = np.arange(num_sessions)
    ax4.bar(x_pos, session_means, yerr=session_sems, color='steelblue', alpha=0.7, capsize=5)
    ax4.axhline(0, color='black', linestyle='--', linewidth=1)
    ax4.axhline(mean_r, color='red', linestyle='-', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Held-out session', fontsize=11)
    ax4.set_ylabel('Mean correlation', fontsize=11)
    ax4.set_title('Prediction by held-out session', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Ses {s}' for s in valid_sessions], rotation=45)
    
    # 5. All individual correlations (flattened across sessions)
    ax5 = axes[1, 1]
    all_corrs = cv_corrs.flatten()
    all_corrs_valid = all_corrs[~np.isnan(all_corrs)]
    ax5.hist(all_corrs_valid, bins=50, color='gray', alpha=0.7, edgecolor='black')
    ax5.axvline(0, color='black', linestyle='--', linewidth=2)
    ax5.axvline(np.mean(all_corrs_valid), color='red', linestyle='-', linewidth=2)
    ax5.set_xlabel('Correlation', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title(f'All individual correlations\n({num_neurons} neurons × {num_sessions} sessions)', 
                 fontsize=12, fontweight='bold')
    
    if len(all_corrs_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(all_corrs_valid, 0)
        ax5.text(0.05, 0.95, f'n = {len(all_corrs_valid)}\nmean = {np.mean(all_corrs_valid):.3f}\nt = {t_stat:.2f}\np = {p_val:.2e}', 
                transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate some additional stats
    prop_positive = np.sum(corrs_valid > 0) / len(corrs_valid) * 100 if len(corrs_valid) > 0 else 0
    prop_sig = np.sum(corrs_valid > 0.1) / len(corrs_valid) * 100 if len(corrs_valid) > 0 else 0
    
    summary_text = f"""
    Cross-Validated Prediction Summary
    ===================================
    Recording: {mouse_recday}
    
    Neurons: {num_neurons}
    Valid neurons: {len(corrs_valid)}
    Sessions: {num_sessions}
    
    All-lag prediction:
      Mean r = {mean_r:.4f}
      % positive = {prop_positive:.1f}%
      % r > 0.1 = {prop_sig:.1f}%
    
    Non-zero lag prediction:
      Mean r = {mean_r_nz:.4f}
    
    Interpretation:
      - Positive correlation means the model
        trained on other sessions can predict
        activity in the held-out session
      - Non-zero lag tests if neurons encode
        lagged task-space information
    """
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle(f'Predicted vs Actual Task Maps: {mouse_recday}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_nonzero_lag_neuron_correlations(results, config, min_lag_distance=2, max_lag=10, mouse_recday='', save_path=None):
    """
    Plot correlation distribution for neurons whose peak coefficient is at 
    a non-zero lag (i.e., prospective/non-place neurons).
    
    Parameters
    ----------
    results : dict
        Results from run_cross_validated_regression
    config : RegressionConfig
        Configuration object
    min_lag_distance : int
        Minimum lag from 0 to be considered non-zero lag neuron (default=2)
    max_lag : int
        Maximum lag to include (default=10)
    mouse_recday : str
        Recording identifier for title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib figure
    nonzero_lag_mask : np.array
        Boolean mask of non-zero lag neurons
    """
    import matplotlib.pyplot as plt
    
    min_lag = min_lag_distance
    
    # Handle both regular and tuning curve results
    if 'cv_correlations' in results:
        # Regular results
        cv_corrs = results['cv_correlations']
        cv_corrs_nz = results['cv_correlations_nonzero']
        mean_corrs = results['mean_correlations']
        mean_corrs_nz = results['mean_correlations_nonzero']
    elif 'tuning_correlations' in results:
        # Tuning curve results
        cv_corrs = results['tuning_correlations']
        cv_corrs_nz = results['tuning_correlations_nonzero']
        mean_corrs = results['mean_tuning_correlations']
        mean_corrs_nz = results['mean_tuning_correlations_nonzero']
    else:
        raise KeyError("Results dict must contain either 'cv_correlations' or 'tuning_correlations'")
    
    coeffs = np.nanmean(results['cv_coeffs'], axis=1)  # (num_neurons, num_regressors)
    
    num_neurons = results['num_neurons']
    num_anchors = config.num_locations * config.num_goal_progress_bins
    
    # Find peak lag for each neuron
    peak_lags = np.zeros(num_neurons)
    for i in range(num_neurons):
        if np.all(np.isnan(coeffs[i])):
            peak_lags[i] = np.nan
        else:
            max_idx = np.nanargmax(coeffs[i])
            peak_lags[i] = max_idx % config.num_lags
    
    # Identify non-zero lag neurons: peak lag within range
    nonzero_lag_mask = (peak_lags >= min_lag) & (peak_lags <= max_lag)
    nonzero_lag_mask = nonzero_lag_mask & ~np.isnan(mean_corrs)
    
    # Also get all valid neurons for comparison
    all_valid = ~np.isnan(mean_corrs)
    zero_lag_mask = ((peak_lags < min_lag) | (peak_lags > max_lag)) & ~np.isnan(mean_corrs)
    
    print(f"Total neurons: {num_neurons}")
    print(f"Valid neurons: {np.sum(all_valid)}")
    print(f"Non-zero lag neurons (peak lag {min_lag}-{max_lag}): {np.sum(nonzero_lag_mask)}")
    print(f"Zero/low lag neurons (peak lag <{min_lag} or >{max_lag}): {np.sum(zero_lag_mask)}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Distribution of peak lags (all neurons)
    ax1 = axes[0, 0]
    valid_lags = peak_lags[~np.isnan(peak_lags)]
    ax1.hist(valid_lags, bins=np.arange(-0.5, config.num_lags + 0.5, 1),
             color='gray', alpha=0.7, edgecolor='black', label='All neurons')
    ax1.axvline(min_lag - 0.5, color='red', linestyle='--', linewidth=2)
    ax1.axvline(max_lag + 0.5, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Peak lag', fontsize=11)
    ax1.set_ylabel('Number of neurons', fontsize=11)
    ax1.set_title('Distribution of peak lags', fontsize=12, fontweight='bold')
    
    # Shade the non-zero lag region
    ax1.axvspan(min_lag - 0.5, max_lag + 0.5, alpha=0.2, color='red', 
               label=f'Non-zero lag (lag {min_lag}-{max_lag})')
    ax1.legend()
    
    # 2. Comparison histogram: non-zero lag vs zero lag neurons
    ax2 = axes[0, 1]
    bins = np.linspace(-0.5, 1, 40)
    ax2.hist(mean_corrs[zero_lag_mask], bins=bins, color='steelblue', alpha=0.6, 
             label=f'Zero lag (n={np.sum(zero_lag_mask)})', edgecolor='black')
    ax2.hist(mean_corrs[nonzero_lag_mask], bins=bins, color='darkorange', alpha=0.6,
             label=f'Non-zero lag (n={np.sum(nonzero_lag_mask)})', edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Mean correlation (all lags)', fontsize=11)
    ax2.set_ylabel('Number of neurons', fontsize=11)
    ax2.set_title('Non-zero lag vs Zero lag neurons', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # 3. Non-zero lag neurons: all-lag correlation distribution
    ax3 = axes[0, 2]
    nzl_corrs = mean_corrs[nonzero_lag_mask]
    ax3.hist(nzl_corrs, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='black', linestyle='--', linewidth=2)
    mean_nzl = np.mean(nzl_corrs) if len(nzl_corrs) > 0 else np.nan
    ax3.axvline(mean_nzl, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_nzl:.3f}')
    ax3.set_xlabel('Correlation (all lags)', fontsize=11)
    ax3.set_ylabel('Number of neurons', fontsize=11)
    ax3.set_title(f'Non-zero lag neurons (peak lag {min_lag}-{max_lag})\nAll-lag prediction', 
                 fontsize=12, fontweight='bold')
    ax3.legend()
    
    if len(nzl_corrs) > 1:
        t_stat, p_val = stats.ttest_1samp(nzl_corrs, 0)
        ax3.text(0.05, 0.95, f'n = {len(nzl_corrs)}\nt = {t_stat:.2f}\np = {p_val:.2e}', 
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Non-zero lag neurons: non-zero coefficient correlation distribution
    ax4 = axes[1, 0]
    nzl_corrs_nz = mean_corrs_nz[nonzero_lag_mask]
    nzl_corrs_nz_valid = nzl_corrs_nz[~np.isnan(nzl_corrs_nz)]
    ax4.hist(nzl_corrs_nz_valid, bins=30, color='darkred', alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='black', linestyle='--', linewidth=2)
    mean_nzl_nz = np.mean(nzl_corrs_nz_valid) if len(nzl_corrs_nz_valid) > 0 else np.nan
    ax4.axvline(mean_nzl_nz, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_nzl_nz:.3f}')
    ax4.set_xlabel('Correlation (non-zero lags)', fontsize=11)
    ax4.set_ylabel('Number of neurons', fontsize=11)
    ax4.set_title(f'Non-zero lag neurons (peak lag {min_lag}-{max_lag})\nNon-zero coeff prediction', 
                 fontsize=12, fontweight='bold')
    ax4.legend()
    
    if len(nzl_corrs_nz_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(nzl_corrs_nz_valid, 0)
        ax4.text(0.05, 0.95, f'n = {len(nzl_corrs_nz_valid)}\nt = {t_stat:.2f}\np = {p_val:.2e}', 
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Peak lag vs correlation scatter
    ax5 = axes[1, 1]
    valid_both = ~np.isnan(peak_lags) & ~np.isnan(mean_corrs)
    colors = np.where(nonzero_lag_mask[valid_both], 'darkorange', 'steelblue')
    ax5.scatter(peak_lags[valid_both], mean_corrs[valid_both], c=colors, alpha=0.5, s=20)
    ax5.axvline(min_lag - 0.5, color='red', linestyle='--', linewidth=2)
    ax5.axvline(max_lag + 0.5, color='red', linestyle='--', linewidth=2)
    ax5.axhline(0, color='black', linestyle='--', linewidth=1)
    ax5.set_xlabel('Peak lag', fontsize=11)
    ax5.set_ylabel('Mean correlation', fontsize=11)
    ax5.set_title('Peak lag vs prediction quality', fontsize=12, fontweight='bold')
    
    # 6. Summary comparison
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate stats for both groups
    zl_corrs = mean_corrs[zero_lag_mask]
    
    # Two-sample t-test
    if len(nzl_corrs) > 1 and len(zl_corrs) > 1:
        t_2samp, p_2samp = stats.ttest_ind(nzl_corrs, zl_corrs)
    else:
        t_2samp, p_2samp = np.nan, np.nan
    
    summary_text = f"""
    Non-zero Lag vs Zero Lag Comparison
    ====================================
    Recording: {mouse_recday}
    Threshold: peak lag in [{min_lag}, {max_lag}]
    
    Zero/low lag neurons:
      n = {np.sum(zero_lag_mask)}
      Mean r = {np.mean(zl_corrs):.4f}
      % positive = {np.sum(zl_corrs > 0) / len(zl_corrs) * 100:.1f}%
    
    Non-zero lag neurons:
      n = {np.sum(nonzero_lag_mask)}
      Mean r = {mean_nzl:.4f}
      % positive = {np.sum(nzl_corrs > 0) / len(nzl_corrs) * 100:.1f}%
    
    Two-sample t-test:
      t = {t_2samp:.3f}
      p = {p_2samp:.2e}
    
    Interpretation:
      Non-zero lag neurons encode task-space
      information {min_lag}-{max_lag} lags ahead/behind.
      Higher correlation suggests predictable
      future location representation.
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle(f'Non-zero Lag Neuron Analysis: {mouse_recday}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig, nonzero_lag_mask


# Alias for backward compatibility
plot_sequence_neuron_correlations = plot_nonzero_lag_neuron_correlations


def plot_cross_validation_results(results, ax=None):
    """
    Plot histogram of cross-validation correlations.
    
    Parameters
    ----------
    results : dict
        Results from run_cross_validated_regression
    ax : matplotlib axis
        Axis to plot on
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        axes = [ax, ax.twinx()]
    
    # All neurons
    corrs = results['mean_correlations']
    corrs_valid = corrs[~np.isnan(corrs)]
    
    axes[0].hist(corrs_valid, bins=50, color='gray', alpha=0.7)
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].axvline(np.mean(corrs_valid), color='blue', linestyle='--', label='Mean')
    axes[0].set_xlabel('Correlation (predicted vs actual)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('All neurons')
    
    # t-test
    if len(corrs_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(corrs_valid, 0)
        axes[0].text(0.05, 0.95, f't={t_stat:.2f}, p={p_val:.2e}', 
                     transform=axes[0].transAxes, verticalalignment='top')
    
    # Non-zero lag
    if len(axes) > 1 and isinstance(axes[1], plt.Axes):
        corrs_nz = results['mean_correlations_nonzero']
        corrs_nz_valid = corrs_nz[~np.isnan(corrs_nz)]
        
        axes[1].hist(corrs_nz_valid, bins=50, color='orange', alpha=0.7)
        axes[1].axvline(0, color='black', linestyle='--')
        axes[1].axvline(np.mean(corrs_nz_valid), color='red', linestyle='--', label='Mean')
        axes[1].set_xlabel('Correlation (non-zero lag prediction)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Non-zero lag prediction')
        
        if len(corrs_nz_valid) > 1:
            t_stat, p_val = stats.ttest_1samp(corrs_nz_valid, 0)
        axes[1].text(0.05, 0.95, f't={t_stat:.2f}, p={p_val:.2e}', 
                     transform=axes[1].transAxes, verticalalignment='top')
    
    plt.tight_layout()
    return axes


def plot_regression_results(results, config, mouse_recday='', save_path=None):
    """
    Create a comprehensive summary figure of regression results.
    
    Parameters
    ----------
    results : dict
        Results from run_cross_validated_regression
    config : RegressionConfig
        Configuration object
    mouse_recday : str
        Recording identifier for title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(16, 10))
    
    # Get valid correlations
    corrs = results['mean_correlations']
    corrs_valid = corrs[~np.isnan(corrs)]
    corrs_nz = results['mean_correlations_nonzero']
    corrs_nz_valid = corrs_nz[~np.isnan(corrs_nz)]
    
    # 1. Histogram of all correlations
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(corrs_valid, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    mean_corr = np.mean(corrs_valid) if len(corrs_valid) > 0 else 0
    ax1.axvline(mean_corr, color='red', linestyle='-', linewidth=2, label=f'Mean={mean_corr:.3f}')
    ax1.set_xlabel('Correlation (predicted vs actual)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('All lag prediction', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    if len(corrs_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(corrs_valid, 0)
        ax1.text(0.05, 0.90, f'n={len(corrs_valid)}\nt={t_stat:.2f}\np={p_val:.2e}', 
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Histogram of non-zero lag correlations
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(corrs_nz_valid, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=2)
    mean_nz = np.mean(corrs_nz_valid) if len(corrs_nz_valid) > 0 else 0
    ax2.axvline(mean_nz, color='red', linestyle='-', linewidth=2, label=f'Mean={mean_nz:.3f}')
    ax2.set_xlabel('Correlation (non-zero lag prediction)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Non-zero lag prediction only', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    if len(corrs_nz_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(corrs_nz_valid, 0)
        ax2.text(0.05, 0.90, f'n={len(corrs_nz_valid)}\nt={t_stat:.2f}\np={p_val:.2e}', 
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Scatter: all lag vs non-zero lag
    ax3 = fig.add_subplot(2, 3, 3)
    valid_both = ~np.isnan(corrs) & ~np.isnan(corrs_nz)
    if np.sum(valid_both) > 0:
        ax3.scatter(corrs[valid_both], corrs_nz[valid_both], alpha=0.5, s=20)
        ax3.plot([-1, 1], [-1, 1], 'k--', linewidth=1)
        ax3.set_xlabel('All lag correlation', fontsize=11)
        ax3.set_ylabel('Non-zero lag correlation', fontsize=11)
        ax3.set_title('Lag comparison', fontsize=12, fontweight='bold')
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_aspect('equal')
    
    # 4. Mean coefficient matrix across neurons
    ax4 = fig.add_subplot(2, 3, 4)
    mean_coeffs = np.nanmean(results['cv_coeffs'], axis=(0, 1))  # Average across neurons and CV folds
    num_anchors = config.num_locations * config.num_goal_progress_bins
    coeff_matrix = mean_coeffs.reshape(num_anchors, config.num_lags)
    im = ax4.imshow(coeff_matrix, aspect='auto', cmap='viridis')
    ax4.set_xlabel('Lag in task space', fontsize=11)
    ax4.set_ylabel('Anchor (location × phase)', fontsize=11)
    ax4.set_title('Mean coefficients', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Coefficient')
    
    # Add phase separators
    for i in range(1, config.num_goal_progress_bins):
        ax4.axhline(i * config.num_locations - 0.5, color='white', linestyle='--', linewidth=0.5)
    
    # 5. Coefficient distribution by lag
    ax5 = fig.add_subplot(2, 3, 5)
    coeffs_all = results['cv_coeffs'].reshape(-1, config.num_regressors)
    coeffs_by_lag = []
    for lag in range(config.num_lags):
        lag_indices = np.arange(lag, config.num_regressors, config.num_lags)
        lag_coeffs = coeffs_all[:, lag_indices].flatten()
        lag_coeffs = lag_coeffs[~np.isnan(lag_coeffs)]
        coeffs_by_lag.append(np.mean(lag_coeffs) if len(lag_coeffs) > 0 else 0)
    
    ax5.bar(range(config.num_lags), coeffs_by_lag, color='teal', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Lag (task space phases)', fontsize=11)
    ax5.set_ylabel('Mean coefficient', fontsize=11)
    ax5.set_title('Coefficients by lag', fontsize=12, fontweight='bold')
    ax5.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)  # Mark lag 0
    
    # 6. Summary text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    Elastic Net Regression Summary
    ==============================
    Recording: {mouse_recday}
    
    Configuration:
    - Locations: {config.num_locations}
    - Goal-progress bins: {config.num_goal_progress_bins}
    - Task lags: {config.num_lags}
    - Total regressors: {config.num_regressors}
    - Alpha: {config.alpha}
    
    Results:
    - Total neurons: {results['num_neurons']}
    - Valid neurons (all lag): {len(corrs_valid)}
    - Valid neurons (non-zero lag): {len(corrs_nz_valid)}
    - Sessions used: {results['num_sessions']}
    
    All lag prediction:
    - Mean r: {mean_corr:.4f}
    
    Non-zero lag prediction:
    - Mean r: {mean_nz:.4f}
    """
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle(f'Elastic Net Regression: {mouse_recday}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# Example Usage
# ============================================================================

def visualize_regressors(locs_norm, config, trial_idx=0, time_range=None):
    """
    Visualize the regressor structure for a single trial.
    
    Shows how regressors are organized: 27 anchors (9 locations × 3 phases) × 12 lags.
    
    Parameters
    ----------
    locs_norm : ndarray
        Shape (num_trials, 360) - normalized location data
    config : RegressionConfig
        Configuration object
    trial_idx : int
        Which trial to visualize
    time_range : tuple, optional
        (start_bin, end_bin) to show subset of bins
        
    Returns
    -------
    regressors : ndarray
        The generated regressors for inspection
    """
    import matplotlib.pyplot as plt
    
    # Generate regressors
    regressors = generate_regressors_from_norm(locs_norm, config)
    
    print("=" * 60)
    print("REGRESSOR STRUCTURE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Locations: {config.num_locations}")
    print(f"  - Goal-progress phases: {config.num_goal_progress_bins}")
    print(f"  - Anchors (loc × phase): {config.num_locations} × {config.num_goal_progress_bins} = {config.num_locations * config.num_goal_progress_bins}")
    print(f"  - Lags per anchor: {config.num_lags}")
    print(f"  - Total regressors: {config.num_regressors}")
    
    print(f"\nRegressor array shape: {regressors.shape}")
    print(f"  - Dimension 0: {regressors.shape[0]} trials")
    print(f"  - Dimension 1: {regressors.shape[1]} bins (360 = 90 bins/state × 4 states)")
    print(f"  - Dimension 2: {regressors.shape[2]} regressors")
    
    # Reshape to show anchor × lag structure
    num_anchors = config.num_locations * config.num_goal_progress_bins
    
    print(f"\nRegressor indexing:")
    print(f"  Regressors are organized as: [anchors × lags] flattened")
    print(f"  Anchor = location × phase")
    print(f"  ")
    print(f"  Index mapping:")
    for loc in range(min(3, config.num_locations)):  # Show first 3 locations
        for phase in range(config.num_goal_progress_bins):
            anchor_idx = loc * config.num_goal_progress_bins + phase
            start_idx = anchor_idx * config.num_lags
            end_idx = start_idx + config.num_lags
            print(f"    Location {loc+1}, Phase {phase}: anchor {anchor_idx} → regressor indices [{start_idx}:{end_idx}]")
    print(f"    ...")
    
    # Get trial data
    trial_regs = regressors[trial_idx]  # Shape: (360, num_regressors)
    trial_locs = locs_norm[trial_idx]   # Shape: (360,)
    
    if time_range is None:
        time_range = (0, min(120, trial_regs.shape[0]))  # First 120 bins (first state + some of second)
    
    start_bin, end_bin = time_range
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Full regressor matrix for this trial segment
    ax1 = axes[0, 0]
    reg_subset = trial_regs[start_bin:end_bin, :]
    im1 = ax1.imshow(reg_subset.T, aspect='auto', cmap='Blues', interpolation='nearest')
    ax1.set_xlabel(f'Bin index ({start_bin} to {end_bin})')
    ax1.set_ylabel('Regressor index')
    ax1.set_title(f'All regressors (trial {trial_idx}, bins {start_bin}-{end_bin})')
    plt.colorbar(im1, ax=ax1)
    
    # Add anchor separators
    for i in range(1, num_anchors):
        ax1.axhline(i * config.num_lags - 0.5, color='red', linestyle='-', linewidth=0.3, alpha=0.5)
    
    # 2. Reshaped as anchor × lag for one timepoint
    ax2 = axes[0, 1]
    # Find a timepoint with some activity
    active_bins = np.where(np.sum(trial_regs, axis=1) > 0)[0]
    if len(active_bins) > 0:
        example_bin = active_bins[min(len(active_bins)//2, len(active_bins)-1)]
    else:
        example_bin = 60  # Middle of first state
    
    reg_at_time = trial_regs[example_bin].reshape(num_anchors, config.num_lags)
    im2 = ax2.imshow(reg_at_time, aspect='auto', cmap='Blues', interpolation='nearest')
    ax2.set_xlabel('Lag (0=current, 11=oldest)')
    ax2.set_ylabel('Anchor (location × phase)')
    ax2.set_title(f'Regressors at bin {example_bin}\n(reshaped to anchors × lags)')
    plt.colorbar(im2, ax=ax2)
    
    # Add location separators
    for i in range(1, config.num_locations):
        ax2.axhline(i * config.num_goal_progress_bins - 0.5, color='white', linestyle='--', linewidth=1)
    
    # Label y-axis with location groups
    yticks = [(i * config.num_goal_progress_bins + 1) for i in range(config.num_locations)]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f'Loc {i+1}' for i in range(config.num_locations)])
    
    # 3. Location trace
    ax3 = axes[1, 0]
    bins = np.arange(start_bin, end_bin)
    ax3.plot(bins, trial_locs[start_bin:end_bin], 'b-', linewidth=1)
    ax3.scatter(bins, trial_locs[start_bin:end_bin], c=trial_locs[start_bin:end_bin], 
                cmap='tab10', s=10, vmin=0, vmax=10)
    ax3.set_xlabel(f'Bin index')
    ax3.set_ylabel('Location')
    ax3.set_title('Location trajectory')
    ax3.set_ylim(0, 10)
    
    # Add state boundaries
    for state in range(5):
        ax3.axvline(state * 90, color='gray', linestyle='--', alpha=0.5)
        if state < 4:
            ax3.text(state * 90 + 45, 9.5, f'State {state}', ha='center', fontsize=9)
    
    # 4. Goal progress and phase
    ax4 = axes[1, 1]
    phases = get_goal_progress_from_bin(bins, config)
    ax4.plot(bins, phases, 'g-', linewidth=2)
    ax4.set_xlabel('Bin index')
    ax4.set_ylabel('Goal progress phase')
    ax4.set_title('Goal progress phase (0, 1, 2)')
    ax4.set_yticks([0, 1, 2])
    ax4.set_ylim(-0.5, 2.5)
    
    # Add state boundaries
    for state in range(5):
        ax4.axvline(state * 90, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print some example values
    print(f"\n" + "=" * 60)
    print("EXAMPLE REGRESSOR VALUES")
    print("=" * 60)
    
    print(f"\nAt bin {example_bin}:")
    print(f"  Location: {trial_locs[example_bin]:.0f}")
    print(f"  Goal progress phase: {get_goal_progress_from_bin(example_bin, config)}")
    print(f"  State: {get_state_from_bin(example_bin, config)}")
    
    print(f"\n  Active regressors (value > 0):")
    active_regs = np.where(trial_regs[example_bin] > 0)[0]
    for reg_idx in active_regs[:10]:  # Show up to 10
        anchor_idx = reg_idx // config.num_lags
        lag = reg_idx % config.num_lags
        loc = anchor_idx // config.num_goal_progress_bins + 1
        phase = anchor_idx % config.num_goal_progress_bins
        print(f"    Regressor {reg_idx}: anchor={anchor_idx} (loc={loc}, phase={phase}), lag={lag}, value={trial_regs[example_bin, reg_idx]:.2f}")
    
    if len(active_regs) > 10:
        print(f"    ... and {len(active_regs) - 10} more")
    
    return regressors

def example_usage():
    """
    Example of how to use this module in your notebook.
    
    Copy and paste this into your notebook after loading data_dic.
    """
    example_code = '''
# ============================================================================
# Example: Running Elastic Net Regression on your data
# ============================================================================

# Import the functions (if running as a module)
# from elasticnet_regression import *

# Or copy all the code above into a cell first, or:
exec(open('elasticnet_regression.py').read())

# 1. Create configuration for 9 node locations, 360-bin normalized data
config = RegressionConfig(
    num_locations=9,            # 9 node locations (ignoring edges)
    num_goal_progress_bins=3,   # As in the paper
    num_task_states=4,          # ABCD = 4 states
    num_lags=12,                # 4 states × 3 phases = 12 lags per trial
    alpha=0.01,                 # Regularization (as in paper)
    use_poisson=False,          # Set True for Poisson regression
    use_positive_only=True,     # Constrain positive coefficients
    num_bins_per_state=90,      # 90 bins per state
    bins_per_phase=30,          # 30 bins per goal-progress phase
)

print(f"Total regressors: {config.num_regressors}")  # Should be 9 × 3 × 12 = 324

# 2. Run cross-validated regression for one recording
mouse_recday = list(data_dic.keys())[0]  # or specify: 'ah08_...'
valid_sessions = valid_sessions_dic[mouse_recday]  # Your valid sessions

results = run_cross_validated_regression(
    data_dic,
    mouse_recday,
    config,
    valid_sessions=valid_sessions,
    verbose=True
)

# 3. Analyze results
print(f"Mean correlation: {np.nanmean(results['mean_correlations']):.3f}")
print(f"Mean correlation (non-zero lag): {np.nanmean(results['mean_correlations_nonzero']):.3f}")

# 4. Identify non-zero lag neurons
nonzero_lag_mask, _ = identify_nonzero_lag_neurons(results, config)
print(f"Non-zero lag neurons: {np.sum(nonzero_lag_mask)}/{len(nonzero_lag_mask)}")

# 5. Plot results
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of correlations
corrs = results['mean_correlations']
corrs_valid = corrs[~np.isnan(corrs)]
axes[0].hist(corrs_valid, bins=50, color='gray', alpha=0.7)
axes[0].axvline(0, color='black', linestyle='--')
axes[0].axvline(np.mean(corrs_valid), color='blue', linestyle='--')
axes[0].set_xlabel('Correlation (predicted vs actual)')
axes[0].set_ylabel('Count')
axes[0].set_title('Cross-validated prediction')
t_stat, p_val = stats.ttest_1samp(corrs_valid, 0)
axes[0].text(0.05, 0.95, f't={t_stat:.2f}, p={p_val:.2e}', 
             transform=axes[0].transAxes, verticalalignment='top')

# Plot coefficients for a specific neuron
plot_coefficient_matrix(results['cv_coeffs'][:, 0, :], config, neuron_idx=0, ax=axes[1])
plt.tight_layout()
plt.show()

# 6. Run for all mice
all_results = {}
for mouse_recday in mouse_recdays:
    print(f"Processing {mouse_recday}...")
    try:
        valid_sessions = valid_sessions_dic.get(mouse_recday, None)
        results = run_cross_validated_regression(
            data_dic,
            mouse_recday,
            config,
            valid_sessions=valid_sessions,
            verbose=True
        )
        if results is not None:
            all_results[mouse_recday] = results
    except Exception as e:
        print(f"Error for {mouse_recday}: {e}")

# 7. Aggregate across all recordings
all_correlations = np.concatenate([r['mean_correlations'] for r in all_results.values()])
all_correlations_valid = all_correlations[~np.isnan(all_correlations)]

t_stat, p_val = stats.ttest_1samp(all_correlations_valid, 0)
print(f"All neurons: mean={np.mean(all_correlations_valid):.3f}, t={t_stat:.2f}, p={p_val:.2e}")

# Plot
plt.figure(figsize=(8, 6))
plt.hist(all_correlations_valid, bins=50, color='gray', alpha=0.7)
plt.axvline(0, color='black', linestyle='--')
plt.axvline(np.mean(all_correlations_valid), color='blue', linestyle='--')
plt.xlabel('Correlation')
plt.ylabel('Count')
plt.title('Cross-validated prediction correlation')
plt.show()
'''
    print(example_code)
    return example_code


def run_and_summarise_all_mice(
    data_dic,
    config,
    valid_sessions_dic=None,
    num_examples=6,
    save_dir=None,
    plot_smooth_sigma=10,
):
    """
    Run regression for every mouse_recday and produce per-mouse and cross-mouse summaries.

    Per mouse
    ---------
    - Polar tuning curves (actual vs predicted) for top non-zero lag neurons
    - Beta coefficient matrices for those same neurons
    - Tuning correlation distribution

    Cross-mouse summary
    -------------------
    - Distribution of tuning correlations pooled across all mice
    - Bar chart of mean r per mouse (non-zero lag neurons only)

    Parameters
    ----------
    data_dic : dict
        Full data dictionary keyed by mouse_recday
    config : RegressionConfig
    valid_sessions_dic : dict, optional
        {mouse_recday: [session_list]}.  If None all sessions are used.
    num_examples : int
        Number of example neurons to show per mouse (default 6)
    save_dir : str, optional
        Directory to save figures.  If None, figures are only displayed.

    Returns
    -------
    all_results : dict
        {mouse_recday: results_tc}  – regression results for each mouse
    all_nz_corrs : dict
        {mouse_recday: array of mean tuning correlations for NZ-lag neurons}
    """
    import matplotlib.pyplot as plt
    import os

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    all_results = {}
    all_nz_corrs = {}

    mouse_recdays = list(data_dic.keys())

    for mouse_recday in mouse_recdays:
        print(f"\n{'='*60}")
        print(f"Processing {mouse_recday}")
        print('='*60)

        valid_sessions = None if valid_sessions_dic is None else valid_sessions_dic.get(mouse_recday)

        try:
            results_tc = run_cross_validated_regression_tuning_curves(
                data_dic, mouse_recday, config,
                valid_sessions=valid_sessions,
                verbose=True,
            )
        except Exception as e:
            print(f"  Regression failed: {e}")
            continue

        if results_tc is None:
            print("  Skipping – not enough sessions.")
            continue

        all_results[mouse_recday] = results_tc

        # --- identify non-zero lag neurons ---
        nz_mask, peak_lags = identify_nonzero_lag_neurons(results_tc, config)
        mean_corrs = results_tc['mean_tuning_correlations']
        valid_nz = nz_mask & ~np.isnan(mean_corrs)
        nz_indices = np.where(valid_nz)[0]

        print(f"  Non-zero lag neurons: {len(nz_indices)}/{np.sum(~np.isnan(mean_corrs))}")
        all_nz_corrs[mouse_recday] = mean_corrs[valid_nz]

        if len(nz_indices) == 0:
            print("  No non-zero lag neurons – skipping plots.")
            continue

        # sort by correlation, take top num_examples
        top_nz = nz_indices[np.argsort(mean_corrs[nz_indices])[::-1]][:num_examples]

        save_prefix = os.path.join(save_dir, mouse_recday) if save_dir else None

        # --- 1. polar tuning curves for top NZ-lag neurons ---
        fig_polar, _ = plot_polar_tuning_curves(
            results_tc, config,
            num_examples=num_examples,
            sort_by='correlation',
            mouse_recday=mouse_recday,
            nonzero_lag_only=True,
            plot_smooth_sigma=plot_smooth_sigma,
        )
        if save_prefix:
            fig_polar.savefig(f"{save_prefix}_polar_nz.svg", bbox_inches='tight')
        plt.close(fig_polar)

        # --- 2. beta matrices for those same neurons ---
        fig_betas = plot_example_betas(
            results_tc, config,
            neuron_indices=top_nz,
        )
        if save_prefix and fig_betas is not None:
            fig_betas.savefig(f"{save_prefix}_betas_nz.svg", bbox_inches='tight')
        if fig_betas is not None:
            plt.close(fig_betas)

        # --- 3. tuning correlation distribution ---
        fig_dist = plot_tuning_correlation_distribution(
            results_tc, config, mouse_recday=mouse_recday,
        )
        if save_prefix and fig_dist is not None:
            fig_dist.savefig(f"{save_prefix}_corr_dist.svg", bbox_inches='tight')
        if fig_dist is not None:
            plt.close(fig_dist)

    # ---------------------------------------------------------------
    # Cross-mouse summary
    # ---------------------------------------------------------------
    if len(all_nz_corrs) == 0:
        print("No results to summarise.")
        return all_results, all_nz_corrs

    fig_sum, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: pooled distribution
    ax = axes[0]
    pooled = np.concatenate(list(all_nz_corrs.values()))
    pooled_valid = pooled[~np.isnan(pooled)]
    ax.hist(pooled_valid, bins=40, color='darkorange', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    mean_pool = np.mean(pooled_valid)
    ax.axvline(mean_pool, color='red', linewidth=2, label=f'Mean = {mean_pool:.3f}')
    if len(pooled_valid) > 1:
        t, p = stats.ttest_1samp(pooled_valid, 0)
        ax.text(0.97, 0.97,
                f'n = {len(pooled_valid)}\nt = {t:.2f}\np = {p:.2e}',
                transform=ax.transAxes, fontsize=10,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Tuning curve correlation', fontsize=11)
    ax.set_ylabel('Number of neurons', fontsize=11)
    ax.set_title('Non-zero lag neurons – pooled across mice', fontsize=12, fontweight='bold')
    ax.legend()

    # Panel 2: per-mouse mean r
    ax = axes[1]
    labels = list(all_nz_corrs.keys())
    means = [np.nanmean(v) for v in all_nz_corrs.values()]
    sems  = [np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))) for v in all_nz_corrs.values()]
    ns    = [np.sum(~np.isnan(v)) for v in all_nz_corrs.values()]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sems, color='darkorange', alpha=0.7, capsize=5, edgecolor='black')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Mean tuning correlation', fontsize=11)
    ax.set_title('Per-mouse mean r (non-zero lag neurons)', fontsize=12, fontweight='bold')

    plt.suptitle('Cross-mouse summary – non-zero lag neurons', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_dir:
        fig_sum.savefig(os.path.join(save_dir, 'cross_mouse_summary.svg'), bbox_inches='tight')
    plt.show()

    return all_results, all_nz_corrs


if __name__ == '__main__':
    print("Elastic Net Regression Analysis Module")
    print("=" * 50)
    print("\nExpected data format:")
    print("  - Neurons_norm: shape (num_neurons, num_trials, 360)")
    print("  - Locs_norm: shape (num_trials, 360)")
    print("  - Locations 1-9 = nodes (used), 10-21 = edges (ignored)")
    print("\nExample usage:\n")
    example_usage()
