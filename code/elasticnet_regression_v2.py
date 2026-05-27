"""
Elastic Net Regression Analysis for Task-Space Abstraction (slim sibling).

This is a trimmed copy of `elasticnet_regression.py`. Functions kept here are
exactly those referenced in the analysis notebooks (LEC/PFC sploratory,
Figure5_Regression). Behaviour is unchanged; the original file remains intact.

Method (El-Gaby et al. 2024, Nature):
  For each neuron, regress activity onto every (location × goal-progress × lag)
  combination. ElasticNet with l1_ratio=0.5, alpha=0.01.
  Regressors: num_locations × num_goal_progress_bins × num_lags  (9×3×12 = 324).

Data format expected:
  - Neurons_norm:  (num_neurons, num_trials, 360)
  - Locs_norm:     (num_trials, 360)
  - 360 bins = 90 bins/state × 4 states
  - Locations 1–9 are nodes (used); 10–21 are edges (ignored).
"""

import numpy as np
from sklearn.linear_model import ElasticNet, PoissonRegressor
from scipy import stats
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', message='An input array is constant')
warnings.filterwarnings('ignore', message='Mean of empty slice')


# ============================================================================
# Configuration
# ============================================================================

class RegressionConfig:
    """Configuration for the regression analysis."""

    def __init__(
        self,
        num_locations=9,
        num_goal_progress_bins=3,
        num_task_states=4,
        num_lags=12,
        alpha=0.01,
        use_poisson=False,
        use_positive_only=True,
        smoothing_sigma=10,
        num_bins_per_state=90,
        bins_per_phase=30,
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
        self.num_regressors = num_locations * num_goal_progress_bins * num_lags


# ============================================================================
# Helper Functions for Normalized Data (360 bins)
# ============================================================================

def get_goal_progress_from_bin(bin_idx, config):
    """Goal-progress phase (0,1,2) for a normalized bin index."""
    bin_within_state = bin_idx % config.num_bins_per_state
    phase = bin_within_state // config.bins_per_phase
    return np.minimum(phase, config.num_goal_progress_bins - 1)


def generate_regressors_from_norm(locs_norm, config, multiple_bumps=True):
    """Generate lagged (location × phase × lag) regressors from normalized location data.

    Creates "bumps" initiated when the animal visits a particular location/phase,
    then rolls them forward through task space as the trial progresses.
    Returns array of shape (num_trials, 360, num_regressors).
    """
    num_trials, num_bins = locs_norm.shape
    num_locs = config.num_locations
    num_phases = config.num_goal_progress_bins
    num_lags = config.num_lags

    regressors = np.zeros((num_trials, num_bins, config.num_regressors))

    for trial_idx in range(num_trials):
        module_anchor_progress = np.zeros((num_locs, num_phases, num_lags))

        prev_phase = -1
        prev_location = -1

        for bin_idx in range(num_bins):
            loc = locs_norm[trial_idx, bin_idx]

            if np.isnan(loc) or loc > num_locs or loc < 1:
                regressors[trial_idx, bin_idx] = module_anchor_progress.flatten()
                continue

            current_loc = int(loc) - 1
            current_phase = get_goal_progress_from_bin(bin_idx, config)

            phase_changed = (current_phase != prev_phase)
            location_changed = (current_loc != prev_location and
                                current_loc >= 0 and current_loc < num_locs)

            if phase_changed:
                for loc_idx in range(num_locs):
                    for phase_idx in range(num_phases):
                        module_anchor_progress[loc_idx, phase_idx] = np.roll(
                            module_anchor_progress[loc_idx, phase_idx], 1
                        )

                        if (current_loc == loc_idx and current_phase == phase_idx):
                            if multiple_bumps or np.sum(module_anchor_progress[loc_idx, phase_idx]) == 0:
                                module_anchor_progress[loc_idx, phase_idx, 0] = 1
                            else:
                                module_anchor_progress[loc_idx, phase_idx, 1] = 0
                        else:
                            module_anchor_progress[loc_idx, phase_idx, 1] = 0

            elif location_changed:
                for loc_idx in range(num_locs):
                    for phase_idx in range(num_phases):
                        if (current_loc == loc_idx and current_phase == phase_idx):
                            if multiple_bumps or np.sum(module_anchor_progress[loc_idx, phase_idx]) == 0:
                                module_anchor_progress[loc_idx, phase_idx, 1] = 1

            regressors[trial_idx, bin_idx] = module_anchor_progress.flatten()

            prev_phase = current_phase
            prev_location = current_loc

    # Roll back by 1 (compensate for forward lag in loop)
    regressors_reshaped = regressors.reshape(num_trials, num_bins, num_locs, num_phases, num_lags)
    regressors_reshaped = np.roll(regressors_reshaped, -1, axis=4)
    regressors = regressors_reshaped.reshape(num_trials, num_bins, config.num_regressors)

    return regressors


def compute_preferred_phase_from_norm(neurons_norm, config):
    """Compute each neuron's preferred goal-progress phase (argmax of mean activity per phase)."""
    num_neurons = neurons_norm.shape[0]
    num_phases = config.num_goal_progress_bins

    pref_phases = np.zeros(num_neurons, dtype=int)
    phase_means = np.zeros((num_neurons, num_phases))

    bin_indices = np.arange(config.total_bins)
    phases_per_bin = get_goal_progress_from_bin(bin_indices, config)

    for neuron_idx in range(num_neurons):
        neuron_mean = np.nanmean(neurons_norm[neuron_idx], axis=0)

        for phase in range(num_phases):
            phase_mask = phases_per_bin == phase
            phase_means[neuron_idx, phase] = np.nanmean(neuron_mean[phase_mask])

        pref_phases[neuron_idx] = np.argmax(phase_means[neuron_idx])

    return pref_phases, phase_means


def identify_state_tuned_neurons(neurons_norm, config, p_threshold=0.05):
    """Identify state-tuned neurons via the El-Gaby et al. 2024 z-score-and-t-test method.

    Steps: peak per state per trial -> z-score across states -> mean z across
    trials -> pref state = argmax -> t-test that state's z's against 0.
    """
    num_neurons, num_trials, num_bins = neurons_norm.shape
    num_states = config.num_task_states
    bins_per_state = config.num_bins_per_state

    is_state_tuned = np.zeros(num_neurons, dtype=bool)
    pref_states = np.zeros(num_neurons, dtype=int)
    p_values = np.full(num_neurons, np.nan)
    t_stats = np.full(num_neurons, np.nan)
    z_scores_pref_all = np.full((num_neurons, num_trials), np.nan)

    for neuron_idx in range(num_neurons):
        neuron_data = neurons_norm[neuron_idx]

        peak_per_state = np.zeros((num_trials, num_states))
        for state in range(num_states):
            state_start = state * bins_per_state
            state_end = (state + 1) * bins_per_state
            state_data = neuron_data[:, state_start:state_end]
            peak_per_state[:, state] = np.nanmax(state_data, axis=1)

        row_means = np.nanmean(peak_per_state, axis=1, keepdims=True)
        row_stds = np.nanstd(peak_per_state, axis=1, keepdims=True)
        row_stds[row_stds == 0] = np.nan
        z_scored = (peak_per_state - row_means) / row_stds

        mean_z_per_state = np.nanmean(z_scored, axis=0)
        pref_state = np.nanargmax(mean_z_per_state)
        pref_states[neuron_idx] = pref_state

        z_scores_pref = z_scored[:, pref_state]
        z_scores_pref_all[neuron_idx] = z_scores_pref

        z_valid = z_scores_pref[~np.isnan(z_scores_pref)]
        if len(z_valid) < 3:
            continue

        t_stat, p_val = stats.ttest_1samp(z_valid, 0)
        t_stats[neuron_idx] = t_stat
        p_values[neuron_idx] = p_val

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
    """Combine per-session state-tuning masks (ANY by default, ALL if require_all_sessions=True)."""
    if valid_sessions is None:
        valid_sessions = list(data_dic[mouse_recday].keys())

    valid_sessions = [
        s for s in valid_sessions
        if ('Neurons_norm' in data_dic[mouse_recday][s] and
            data_dic[mouse_recday][s]['Neurons_norm'] is not None)
    ]

    if len(valid_sessions) == 0:
        return None, None

    first_session = valid_sessions[0]
    num_neurons = data_dic[mouse_recday][first_session]['Neurons_norm'].shape[0]

    session_masks = np.zeros((num_neurons, len(valid_sessions)), dtype=bool)
    session_results = {}

    for i, session in enumerate(valid_sessions):
        neurons_norm = data_dic[mouse_recday][session]['Neurons_norm']
        is_tuned, results = identify_state_tuned_neurons(neurons_norm, config, p_threshold)
        session_masks[:, i] = is_tuned
        session_results[session] = results

    if require_all_sessions:
        is_state_tuned = np.all(session_masks, axis=1)
    else:
        is_state_tuned = np.any(session_masks, axis=1)

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

def fit_elasticnet_regression(X, y, config, return_model=False):
    """Fit elastic net (or Poisson) regression, ignoring NaN samples."""
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
            # l1_ratio=0.5 → 1:1 mix of L1 and L2
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


def filter_unique_task_sessions(data_dic, mouse_recday, sessions):
    """Drop later occurrences of duplicate task structures (keep first occurrence)."""
    seen_tasks = set()
    filtered_sessions = []

    for session in sessions:
        session_data = data_dic[mouse_recday][session]

        if 'Task' in session_data:
            task = session_data['Task']
            if isinstance(task, np.ndarray):
                task_key = tuple(task.tolist())
            elif isinstance(task, list):
                task_key = tuple(task)
            else:
                task_key = task
        else:
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
    """Leave-one-out cross-validated regression with bin-by-bin prediction correlation.

    For each neuron, train on N-1 sessions and test on the held-out one. Each
    neuron is fit only on bins of its preferred phase (computed from training
    data). Returns per-neuron, per-fold correlations and coefficients.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    config : RegressionConfig
    valid_sessions : list, optional
        Session indices to use. If None, all sessions are used.
    state_tuned_mask : ndarray, optional
        Pre-computed boolean mask of state-tuned neurons. Computed internally
        if None and require_state_tuning=True.
    state_tuning_p_threshold : float
    require_state_tuning : bool
        If True (default), restrict to state-tuned neurons (El-Gaby 2024).
    filter_duplicate_tasks : bool
        If True (default), keep only first occurrence of each task structure.
    verbose : bool
    """
    if valid_sessions is None:
        valid_sessions = list(data_dic[mouse_recday].keys())

    valid_sessions = [
        s for s in valid_sessions
        if ('Neurons_norm' in data_dic[mouse_recday][s] and
            'Locs_norm' in data_dic[mouse_recday][s] and
            data_dic[mouse_recday][s]['Neurons_norm'] is not None and
            data_dic[mouse_recday][s]['Locs_norm'] is not None)
    ]

    if filter_duplicate_tasks:
        sessions_before = len(valid_sessions)
        valid_sessions = filter_unique_task_sessions(data_dic, mouse_recday, valid_sessions)
        if verbose and len(valid_sessions) < sessions_before:
            print(f"Filtered {sessions_before - len(valid_sessions)} duplicate task sessions")

    if len(valid_sessions) < 2:
        print(f"Not enough valid sessions with normalized data for {mouse_recday}")
        return None

    num_sessions = len(valid_sessions)

    first_session = valid_sessions[0]
    num_neurons = data_dic[mouse_recday][first_session]['Neurons_norm'].shape[0]

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

    # Pre-compute regressors for all sessions
    session_data_cache = {}
    for session in valid_sessions:
        neurons_norm = data_dic[mouse_recday][session]['Neurons_norm']
        locs_norm = data_dic[mouse_recday][session]['Locs_norm']
        regressors = generate_regressors_from_norm(locs_norm, config)

        session_data_cache[session] = {
            'neurons_norm': neurons_norm,
            'regressors': regressors,
            'num_trials': neurons_norm.shape[1]
        }

    bin_indices = np.arange(config.total_bins)
    phases_per_bin = get_goal_progress_from_bin(bin_indices, config)

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

        train_neurons_stacked = np.concatenate(
            [session_data_cache[s]['neurons_norm'] for s in train_sessions],
            axis=1
        )
        pref_phases, _ = compute_preferred_phase_from_norm(train_neurons_stacked, config)

        test_cache = session_data_cache[test_session]
        test_neurons = test_cache['neurons_norm']
        test_regressors = test_cache['regressors']
        test_num_trials = test_cache['num_trials']

        for neuron_idx in neurons_to_analyze:
            pref_phase = pref_phases[neuron_idx]
            phase_mask = phases_per_bin == pref_phase

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

            coeffs = fit_elasticnet_regression(X_train, y_train, config)
            cv_coeffs[neuron_idx, test_idx] = coeffs

            if np.any(np.isnan(coeffs)):
                continue

            y_test_all = []
            X_test_all = []

            for trial_idx in range(test_num_trials):
                y = test_neurons[neuron_idx, trial_idx, phase_mask]
                X = test_regressors[trial_idx, phase_mask, :]
                y_test_all.append(y)
                X_test_all.append(X)

            y_test = np.concatenate(y_test_all)
            X_test = np.vstack(X_test_all)

            y_pred = X_test @ coeffs

            valid = ~np.isnan(y_test) & ~np.isnan(y_pred)
            if np.sum(valid) > 3:
                corr, _ = stats.pearsonr(y_test[valid], y_pred[valid])
                cv_correlations[neuron_idx, test_idx] = corr

            # Non-zero lag analysis: zero out lag-0 coefficients before predicting
            coeffs_nonzero = coeffs.copy()
            lag0_indices = np.arange(0, config.num_regressors, config.num_lags)
            coeffs_nonzero[lag0_indices] = 0

            y_pred_nonzero = X_test @ coeffs_nonzero
            valid = ~np.isnan(y_test) & ~np.isnan(y_pred_nonzero)
            if np.sum(valid) > 3:
                corr, _ = stats.pearsonr(y_test[valid], y_pred_nonzero[valid])
                cv_correlations_nonzero[neuron_idx, test_idx] = corr

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
    """Leave-one-out CV regression with tuning-curve correlations.

    Like `run_cross_validated_regression` but instead of bin-by-bin correlation,
    averages predicted activity across trials to get a 360-bin predicted tuning
    curve and compares it to the actual 360-bin tuning curve in the held-out
    session.

    Returns three parallel correlation metrics:
      - `cv_tuning_correlations`: Pearson r over the full 360-bin tuning curve.
        Captures both state-level tuning AND within-state goal-progress structure.
      - `cv_state_correlations`: Pearson r over 4-state-MEAN vectors (each 90-bin
        state window collapsed to its mean). Isolates state-level tuning by
        removing the within-state goal-progress ramp as a shared confound.
      - `cv_state_correlations_max`: Pearson r over 4-state-MAX vectors (peak
        firing per 90-bin state window). Alternative aggregator that emphasises
        punctate / transient state responses rather than sustained drive.
        Per-neuron p-values on n=4 correlations are noisy; aggregate across
        neurons (mean r + t-test) for inference.

    All three metrics are also computed for the lag-0-zeroed prediction variant
    (`*_nonzero` keys), which tests prediction from non-zero-lag coefficients only.

    Parameters
    ----------
    See `run_cross_validated_regression` (same signature).
    """
    if valid_sessions is None:
        valid_sessions = list(data_dic[mouse_recday].keys())

    valid_sessions = [
        s for s in valid_sessions
        if ('Neurons_norm' in data_dic[mouse_recday][s] and
            'Locs_norm' in data_dic[mouse_recday][s] and
            data_dic[mouse_recday][s]['Neurons_norm'] is not None and
            data_dic[mouse_recday][s]['Locs_norm'] is not None)
    ]

    if filter_duplicate_tasks:
        sessions_before = len(valid_sessions)
        valid_sessions = filter_unique_task_sessions(data_dic, mouse_recday, valid_sessions)
        if verbose and len(valid_sessions) < sessions_before:
            print(f"Filtered {sessions_before - len(valid_sessions)} duplicate task sessions")

    if len(valid_sessions) < 2:
        print(f"Not enough valid sessions with normalized data for {mouse_recday}")
        return None

    num_sessions = len(valid_sessions)

    first_session = valid_sessions[0]
    num_neurons = data_dic[mouse_recday][first_session]['Neurons_norm'].shape[0]

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

    session_data_cache = {}
    for session in valid_sessions:
        neurons_norm = data_dic[mouse_recday][session]['Neurons_norm']
        locs_norm = data_dic[mouse_recday][session]['Locs_norm']
        regressors = generate_regressors_from_norm(locs_norm, config)

        session_data_cache[session] = {
            'neurons_norm': neurons_norm,
            'regressors': regressors,
            'num_trials': neurons_norm.shape[1]
        }

    bin_indices = np.arange(config.total_bins)
    phases_per_bin = get_goal_progress_from_bin(bin_indices, config)

    cv_coeffs = np.zeros((num_neurons, num_sessions, config.num_regressors))
    cv_tuning_corrs = np.zeros((num_neurons, num_sessions))
    cv_tuning_corrs_nonzero = np.zeros((num_neurons, num_sessions))
    cv_actual_tuning = np.zeros((num_neurons, num_sessions, config.total_bins))
    cv_predicted_tuning = np.zeros((num_neurons, num_sessions, config.total_bins))
    cv_predicted_tuning_nonzero = np.zeros((num_neurons, num_sessions, config.total_bins))
    # 4-state-mean tuning correlations: collapse each 90-bin state window to its
    # mean and correlate the resulting 4-vectors. Isolates state-level tuning
    # by removing within-state goal-progress structure as a confound.
    cv_state_corrs = np.zeros((num_neurons, num_sessions))
    cv_state_corrs_nonzero = np.zeros((num_neurons, num_sessions))
    # 4-state-MAX tuning correlations: peak firing per state window (alternative
    # to mean for transient/punctate state responses).
    cv_state_corrs_max = np.zeros((num_neurons, num_sessions))
    cv_state_corrs_nonzero_max = np.zeros((num_neurons, num_sessions))

    cv_coeffs[:] = np.nan
    cv_tuning_corrs[:] = np.nan
    cv_tuning_corrs_nonzero[:] = np.nan
    cv_actual_tuning[:] = np.nan
    cv_predicted_tuning[:] = np.nan
    cv_predicted_tuning_nonzero[:] = np.nan
    cv_state_corrs[:] = np.nan
    cv_state_corrs_nonzero[:] = np.nan
    cv_state_corrs_max[:] = np.nan
    cv_state_corrs_nonzero_max[:] = np.nan

    for test_idx, test_session in enumerate(valid_sessions):
        if verbose:
            print(f"CV fold {test_idx + 1}/{num_sessions}: testing on session {test_session}")

        train_sessions = [s for s in valid_sessions if s != test_session]

        train_neurons_stacked = np.concatenate(
            [session_data_cache[s]['neurons_norm'] for s in train_sessions],
            axis=1
        )
        pref_phases, _ = compute_preferred_phase_from_norm(train_neurons_stacked, config)

        test_cache = session_data_cache[test_session]
        test_neurons = test_cache['neurons_norm']
        test_regressors = test_cache['regressors']
        test_num_trials = test_cache['num_trials']

        for neuron_idx in neurons_to_analyze:
            pref_phase = pref_phases[neuron_idx]
            phase_mask = phases_per_bin == pref_phase

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

            coeffs = fit_elasticnet_regression(X_train, y_train, config)
            cv_coeffs[neuron_idx, test_idx] = coeffs

            if np.any(np.isnan(coeffs)):
                continue

            # Predict on ALL bins of test set to get a full 360-bin tuning curve
            predicted_trials = np.zeros((test_num_trials, config.total_bins))
            predicted_trials_nonzero = np.zeros((test_num_trials, config.total_bins))

            coeffs_nonzero = coeffs.copy()
            lag0_indices = np.arange(0, config.num_regressors, config.num_lags)
            coeffs_nonzero[lag0_indices] = 0

            for trial_idx in range(test_num_trials):
                X_trial = test_regressors[trial_idx]
                predicted_trials[trial_idx] = X_trial @ coeffs
                predicted_trials_nonzero[trial_idx] = X_trial @ coeffs_nonzero

            predicted_tuning = np.nanmean(predicted_trials, axis=0)
            predicted_tuning_nonzero = np.nanmean(predicted_trials_nonzero, axis=0)
            actual_tuning = np.nanmean(test_neurons[neuron_idx], axis=0)

            cv_actual_tuning[neuron_idx, test_idx] = actual_tuning
            cv_predicted_tuning[neuron_idx, test_idx] = predicted_tuning
            cv_predicted_tuning_nonzero[neuron_idx, test_idx] = predicted_tuning_nonzero

            valid = ~np.isnan(actual_tuning) & ~np.isnan(predicted_tuning)
            if np.sum(valid) > 10:
                corr, _ = stats.pearsonr(actual_tuning[valid], predicted_tuning[valid])
                cv_tuning_corrs[neuron_idx, test_idx] = corr

            valid_nz = ~np.isnan(actual_tuning) & ~np.isnan(predicted_tuning_nonzero)
            if np.sum(valid_nz) > 10:
                corr_nz, _ = stats.pearsonr(actual_tuning[valid_nz], predicted_tuning_nonzero[valid_nz])
                cv_tuning_corrs_nonzero[neuron_idx, test_idx] = corr_nz

            # 4-state-mean correlation: collapse each 90-bin state window to its
            # mean and correlate the resulting 4-vectors. Removes within-state
            # goal-progress structure as a shared-confound between predicted and actual.
            n_states = config.num_task_states
            bps = config.num_bins_per_state
            actual_state = np.nanmean(actual_tuning.reshape(n_states, bps), axis=1)
            pred_state = np.nanmean(predicted_tuning.reshape(n_states, bps), axis=1)
            pred_state_nz = np.nanmean(predicted_tuning_nonzero.reshape(n_states, bps), axis=1)

            valid_st = ~np.isnan(actual_state) & ~np.isnan(pred_state)
            if np.sum(valid_st) >= 3:
                corr_st, _ = stats.pearsonr(actual_state[valid_st], pred_state[valid_st])
                cv_state_corrs[neuron_idx, test_idx] = corr_st

            valid_st_nz = ~np.isnan(actual_state) & ~np.isnan(pred_state_nz)
            if np.sum(valid_st_nz) >= 3:
                corr_st_nz, _ = stats.pearsonr(actual_state[valid_st_nz], pred_state_nz[valid_st_nz])
                cv_state_corrs_nonzero[neuron_idx, test_idx] = corr_st_nz

            # 4-state-MAX correlation: peak firing per 90-bin state window.
            # Alternative to mean — emphasises punctate state responses.
            actual_state_max = np.nanmax(actual_tuning.reshape(n_states, bps), axis=1)
            pred_state_max = np.nanmax(predicted_tuning.reshape(n_states, bps), axis=1)
            pred_state_nz_max = np.nanmax(predicted_tuning_nonzero.reshape(n_states, bps), axis=1)

            valid_mx = ~np.isnan(actual_state_max) & ~np.isnan(pred_state_max)
            if (np.sum(valid_mx) >= 3
                    and np.std(actual_state_max[valid_mx]) > 0
                    and np.std(pred_state_max[valid_mx]) > 0):
                corr_mx, _ = stats.pearsonr(actual_state_max[valid_mx], pred_state_max[valid_mx])
                cv_state_corrs_max[neuron_idx, test_idx] = corr_mx

            valid_mx_nz = ~np.isnan(actual_state_max) & ~np.isnan(pred_state_nz_max)
            if (np.sum(valid_mx_nz) >= 3
                    and np.std(actual_state_max[valid_mx_nz]) > 0
                    and np.std(pred_state_nz_max[valid_mx_nz]) > 0):
                corr_mx_nz, _ = stats.pearsonr(
                    actual_state_max[valid_mx_nz], pred_state_nz_max[valid_mx_nz]
                )
                cv_state_corrs_nonzero_max[neuron_idx, test_idx] = corr_mx_nz

    mean_tuning_corrs = np.nanmean(cv_tuning_corrs, axis=1)
    mean_tuning_corrs_nonzero = np.nanmean(cv_tuning_corrs_nonzero, axis=1)
    mean_state_corrs = np.nanmean(cv_state_corrs, axis=1)
    mean_state_corrs_nonzero = np.nanmean(cv_state_corrs_nonzero, axis=1)
    mean_state_corrs_max = np.nanmean(cv_state_corrs_max, axis=1)
    mean_state_corrs_nonzero_max = np.nanmean(cv_state_corrs_nonzero_max, axis=1)

    results = {
        'cv_coeffs': cv_coeffs,
        'cv_tuning_correlations': cv_tuning_corrs,
        'cv_tuning_correlations_nonzero': cv_tuning_corrs_nonzero,
        'mean_tuning_correlations': mean_tuning_corrs,
        'mean_tuning_correlations_nonzero': mean_tuning_corrs_nonzero,
        'cv_state_correlations': cv_state_corrs,
        'cv_state_correlations_nonzero': cv_state_corrs_nonzero,
        'mean_state_correlations': mean_state_corrs,
        'mean_state_correlations_nonzero': mean_state_corrs_nonzero,
        'cv_state_correlations_max': cv_state_corrs_max,
        'cv_state_correlations_nonzero_max': cv_state_corrs_nonzero_max,
        'mean_state_correlations_max': mean_state_corrs_max,
        'mean_state_correlations_nonzero_max': mean_state_corrs_nonzero_max,
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
    """Identify non-zero lag neurons per El-Gaby et al. 2024.

    Criterion: all three of the highest regression coefficients must be at
    intermediate lags (2 to num_lags-2). Lags 0, 1 (current/near-current) and
    the last lag (one full state back, same location in cyclic task) are excluded.
    """
    cv_coeffs = results['cv_coeffs']
    mean_coeffs = np.nanmean(cv_coeffs, axis=1)

    num_neurons = mean_coeffs.shape[0]
    num_anchors = config.num_locations * config.num_goal_progress_bins

    peak_lags = np.full(num_neurons, np.nan)
    nonzero_lag_mask = np.zeros(num_neurons, dtype=bool)

    for neuron_idx in range(num_neurons):
        neuron_coeffs = mean_coeffs[neuron_idx]

        if np.all(np.isnan(neuron_coeffs)) or np.all(neuron_coeffs == 0):
            continue

        coeff_matrix = neuron_coeffs.reshape(num_anchors, config.num_lags)
        flat_coeffs = coeff_matrix.flatten()
        top3_flat = np.argsort(flat_coeffs)[-3:]
        top3_lags = top3_flat % config.num_lags

        peak_lags[neuron_idx] = top3_lags[-1]
        nonzero_lag_mask[neuron_idx] = np.all((top3_lags >= 2) & (top3_lags <= config.num_lags - 2))

    return nonzero_lag_mask, peak_lags


identify_sequence_neurons = identify_nonzero_lag_neurons


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_polar_tuning_curves(results, config, data_dic=None, mouse_recday='',
                             num_examples=9, sort_by='correlation',
                             nonzero_lag_only=False,
                             use_smoothed=True, plot_smooth_sigma=10,
                             save_path=None):
    """Polar plots of actual vs predicted 360-bin tuning curves across held-out sessions."""
    import matplotlib.pyplot as plt

    mean_corrs = results['mean_tuning_correlations']
    cv_actual = results['cv_actual_tuning']
    cv_pred = results['cv_predicted_tuning']
    num_sessions = results['num_sessions']
    valid_sessions = results['valid_sessions']

    if use_smoothed and data_dic is not None and mouse_recday:
        smoothed_actual = {}
        for sess_idx, session in enumerate(valid_sessions):
            if 'Smoothed_norm' in data_dic[mouse_recday][session]:
                smoothed_actual[sess_idx] = data_dic[mouse_recday][session]['Smoothed_norm']
            else:
                smoothed_actual[sess_idx] = None
    else:
        smoothed_actual = None

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

    theta = np.linspace(0, 2 * np.pi, config.total_bins, endpoint=False)
    theta_closed = np.append(theta, theta[0])

    ncols = num_sessions
    nrows = len(selected)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             subplot_kw=dict(projection='polar'))

    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    def normalize(arr):
        arr_valid = arr[~np.isnan(arr)]
        if len(arr_valid) == 0 or np.max(arr_valid) == np.min(arr_valid):
            return arr
        return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

    for row_idx, neuron_idx in enumerate(selected):
        mean_corr = mean_corrs[neuron_idx]

        for sess_idx in range(num_sessions):
            ax = axes[row_idx, sess_idx]

            if smoothed_actual is not None and smoothed_actual.get(sess_idx) is not None:
                actual = smoothed_actual[sess_idx][neuron_idx]
            else:
                actual = cv_actual[neuron_idx, sess_idx]

            predicted = cv_pred[neuron_idx, sess_idx]

            valid_bins = ~np.isnan(actual) & ~np.isnan(predicted)
            if np.sum(valid_bins) > 10:
                sess_corr, _ = stats.pearsonr(actual[valid_bins], predicted[valid_bins])
            else:
                sess_corr = np.nan

            # Smoothing is for display only — does not affect correlation
            if plot_smooth_sigma is not None and plot_smooth_sigma > 0:
                from scipy.ndimage import gaussian_filter1d
                actual_plot = gaussian_filter1d(actual, sigma=plot_smooth_sigma, mode='wrap')
                predicted_plot = gaussian_filter1d(predicted, sigma=plot_smooth_sigma, mode='wrap')
            else:
                actual_plot = actual
                predicted_plot = predicted

            actual_closed = np.append(actual_plot, actual_plot[0])
            pred_closed = np.append(predicted_plot, predicted_plot[0])

            actual_norm = normalize(actual_closed)
            pred_norm = normalize(pred_closed)

            ax.plot(theta_closed, actual_norm, 'b-', linewidth=2, label='Actual')
            ax.plot(theta_closed, pred_norm, 'r-', linewidth=2, label='Predicted')
            ax.fill(theta_closed, actual_norm, alpha=0.2, color='blue')
            ax.fill(theta_closed, pred_norm, alpha=0.2, color='red')

            for i in range(4):
                ax.axvline(i * np.pi / 2, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

            ax.set_ylim(0, 1.1)
            ax.set_rticks([0.5, 1.0])

            session_label = valid_sessions[sess_idx]
            if row_idx == 0:
                ax.set_title(f'Session {session_label}\nr = {sess_corr:.3f}', fontsize=10)
            else:
                ax.set_title(f'r = {sess_corr:.3f}', fontsize=10)

            if sess_idx == 0:
                peak_lag_str = f", peak lag={int(peak_lags[neuron_idx])}" if peak_lags is not None else ""
                ax.annotate(f'N{neuron_idx}\nmean r={mean_corr:.3f}{peak_lag_str}',
                            xy=(0.02, 0.98), xycoords='axes fraction',
                            fontsize=9, ha='left', va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
    """Distribution + summary panels for tuning-curve prediction correlations."""
    import matplotlib.pyplot as plt

    mean_corrs = results['mean_tuning_correlations']
    mean_corrs_nz_all = results['mean_tuning_correlations_nonzero']
    cv_corrs = results['cv_tuning_correlations']
    num_neurons = results['num_neurons']
    num_sessions = results['num_sessions']

    nz_mask, _ = identify_nonzero_lag_neurons(results, config)
    mean_corrs_nz = np.where(nz_mask, mean_corrs_nz_all, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. All-lag distribution
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

    # 3. All-lag vs non-zero lag scatter
    ax3 = axes[0, 2]
    valid_both = ~np.isnan(mean_corrs) & ~np.isnan(mean_corrs_nz)
    if np.sum(valid_both) > 0:
        ax3.scatter(mean_corrs[valid_both], mean_corrs_nz[valid_both], alpha=0.5, s=20, c='teal')
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

    # 5. % neurons with positive correlation
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


def plot_coefficient_matrix(coeffs, config, neuron_idx=0, ax=None):
    """Heatmap of regression coefficients for one neuron (anchor × lag)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    neuron_coeffs = coeffs[neuron_idx] if len(coeffs.shape) > 1 else coeffs

    num_anchors = config.num_locations * config.num_goal_progress_bins
    coeff_matrix = neuron_coeffs.reshape(num_anchors, config.num_lags)

    im = ax.imshow(coeff_matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel('Lag (task space)')
    ax.set_ylabel('Anchor (location × phase)')
    ax.set_title(f'Regression coefficients for neuron {neuron_idx}')
    plt.colorbar(im, ax=ax)

    return ax


def plot_example_betas(results, config, neuron_indices=None, num_examples=6, save_path=None):
    """Grid of beta coefficient matrices (anchor × lag) for selected neurons."""
    import matplotlib.pyplot as plt

    coeffs = np.nanmean(results['cv_coeffs'], axis=1)
    if 'mean_correlations' in results:
        corrs = results['mean_correlations']
    else:
        corrs = results['mean_tuning_correlations']

    if neuron_indices is None:
        valid_mask = ~np.isnan(corrs) & ~np.all(np.isnan(coeffs), axis=1)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            print("No valid neurons to plot!")
            return None

        sorted_indices = valid_indices[np.argsort(corrs[valid_indices])[::-1]]
        neuron_indices = sorted_indices[:num_examples]

    num_plots = len(neuron_indices)
    ncols = min(3, num_plots)
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if num_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    num_anchors = config.num_locations * config.num_goal_progress_bins

    for i, neuron_idx in enumerate(neuron_indices):
        ax = axes[i]

        neuron_coeffs = coeffs[neuron_idx]
        neuron_corr = corrs[neuron_idx]

        coeff_matrix = neuron_coeffs.reshape(num_anchors, config.num_lags)

        im = ax.imshow(coeff_matrix, aspect='auto', cmap='hot', interpolation='nearest')

        for loc in range(1, config.num_locations):
            ax.axhline(loc * config.num_goal_progress_bins - 0.5, color='white',
                       linestyle='-', linewidth=0.5, alpha=0.7)

        ax.set_xlabel('Lag (0=current → 11=oldest)', fontsize=10)
        ax.set_ylabel('Anchor', fontsize=10)
        ax.set_title(f'Neuron {neuron_idx}\nr = {neuron_corr:.3f}', fontsize=11)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        yticks = [i * config.num_goal_progress_bins + 1 for i in range(config.num_locations)]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'L{i+1}' for i in range(config.num_locations)], fontsize=8)

        ax.set_xticks([0, 3, 6, 9, 11])
        ax.set_xticklabels(['0', '3', '6', '9', '11'], fontsize=8)

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

    return fig


def plot_population_beta_summary(results, config, save_path=None):
    """Population-level summary panels of beta coefficients (mean matrices, lag/loc/phase profiles, peak-lag histogram)."""
    import matplotlib.pyplot as plt

    coeffs = np.nanmean(results['cv_coeffs'], axis=1)
    if 'mean_correlations' in results:
        corrs = results['mean_correlations']
    else:
        corrs = results['mean_tuning_correlations']

    valid_mask = ~np.isnan(corrs) & ~np.all(np.isnan(coeffs), axis=1)
    valid_coeffs = coeffs[valid_mask]
    valid_corrs = corrs[valid_mask]

    num_anchors = config.num_locations * config.num_goal_progress_bins

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Mean coefficient matrix (all valid neurons)
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

    # 2. Mean coefficient matrix (top neurons by correlation)
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
    """Distribution + summary panels for bin-by-bin prediction correlations in held-out sessions."""
    import matplotlib.pyplot as plt

    cv_corrs = results['cv_correlations']
    mean_corrs = results['mean_correlations']
    mean_corrs_nz = results['mean_correlations_nonzero']
    valid_sessions = results['valid_sessions']
    num_neurons = results['num_neurons']
    num_sessions = results['num_sessions']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. All-lag distribution
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

    if len(corrs_valid) > 1:
        t_stat, p_val = stats.ttest_1samp(corrs_valid, 0)
        ax1.text(0.05, 0.95, f'n = {len(corrs_valid)}\nt = {t_stat:.2f}\np = {p_val:.2e}',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Non-zero lag distribution
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

    # 3. All-lag vs non-zero lag scatter
    ax3 = axes[0, 2]
    valid_both = ~np.isnan(mean_corrs) & ~np.isnan(mean_corrs_nz)
    if np.sum(valid_both) > 0:
        ax3.scatter(mean_corrs[valid_both], mean_corrs_nz[valid_both], alpha=0.5, s=20, c='teal')
        ax3.plot([-1, 1], [-1, 1], 'k--', linewidth=1)
        ax3.set_xlabel('All-lag correlation', fontsize=11)
        ax3.set_ylabel('Non-zero lag correlation', fontsize=11)
        ax3.set_title('All-lag vs Non-zero lag', fontsize=12, fontweight='bold')
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_aspect('equal')

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

    # 5. All individual correlations
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
        ax5.text(0.05, 0.95,
                 f'n = {len(all_corrs_valid)}\nmean = {np.mean(all_corrs_valid):.3f}\nt = {t_stat:.2f}\np = {p_val:.2e}',
                 transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 6. Summary
    ax6 = axes[1, 2]
    ax6.axis('off')

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


def plot_nonzero_lag_neuron_correlations(results, config, min_lag_distance=2, max_lag=10,
                                         mouse_recday='', save_path=None):
    """Compare prediction correlations for non-zero-lag vs zero-lag neurons (by single-coeff peak lag)."""
    import matplotlib.pyplot as plt

    min_lag = min_lag_distance

    if 'cv_correlations' in results:
        cv_corrs = results['cv_correlations']
        mean_corrs = results['mean_correlations']
        mean_corrs_nz = results['mean_correlations_nonzero']
    elif 'tuning_correlations' in results:
        cv_corrs = results['tuning_correlations']
        mean_corrs = results['mean_tuning_correlations']
        mean_corrs_nz = results['mean_tuning_correlations_nonzero']
    else:
        raise KeyError("Results dict must contain either 'cv_correlations' or 'tuning_correlations'")

    coeffs = np.nanmean(results['cv_coeffs'], axis=1)

    num_neurons = results['num_neurons']

    # Peak lag (single highest coefficient) per neuron
    peak_lags = np.zeros(num_neurons)
    for i in range(num_neurons):
        if np.all(np.isnan(coeffs[i])):
            peak_lags[i] = np.nan
        else:
            max_idx = np.nanargmax(coeffs[i])
            peak_lags[i] = max_idx % config.num_lags

    nonzero_lag_mask = (peak_lags >= min_lag) & (peak_lags <= max_lag)
    nonzero_lag_mask = nonzero_lag_mask & ~np.isnan(mean_corrs)

    all_valid = ~np.isnan(mean_corrs)
    zero_lag_mask = ((peak_lags < min_lag) | (peak_lags > max_lag)) & ~np.isnan(mean_corrs)

    print(f"Total neurons: {num_neurons}")
    print(f"Valid neurons: {np.sum(all_valid)}")
    print(f"Non-zero lag neurons (peak lag {min_lag}-{max_lag}): {np.sum(nonzero_lag_mask)}")
    print(f"Zero/low lag neurons (peak lag <{min_lag} or >{max_lag}): {np.sum(zero_lag_mask)}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Distribution of peak lags
    ax1 = axes[0, 0]
    valid_lags = peak_lags[~np.isnan(peak_lags)]
    ax1.hist(valid_lags, bins=np.arange(-0.5, config.num_lags + 0.5, 1),
             color='gray', alpha=0.7, edgecolor='black', label='All neurons')
    ax1.axvline(min_lag - 0.5, color='red', linestyle='--', linewidth=2)
    ax1.axvline(max_lag + 0.5, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Peak lag', fontsize=11)
    ax1.set_ylabel('Number of neurons', fontsize=11)
    ax1.set_title('Distribution of peak lags', fontsize=12, fontweight='bold')

    ax1.axvspan(min_lag - 0.5, max_lag + 0.5, alpha=0.2, color='red',
                label=f'Non-zero lag (lag {min_lag}-{max_lag})')
    ax1.legend()

    # 2. Non-zero vs zero lag comparison
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

    # 3. Non-zero lag neurons: all-lag correlation
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

    # 4. Non-zero lag neurons: non-zero coefficient correlation
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

    # 5. Peak lag vs correlation
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

    # 6. Summary
    ax6 = axes[1, 2]
    ax6.axis('off')

    zl_corrs = mean_corrs[zero_lag_mask]

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


plot_sequence_neuron_correlations = plot_nonzero_lag_neuron_correlations


# ============================================================================
# Cross-mouse wrapper
# ============================================================================

def run_and_summarise_all_mice(
    data_dic,
    config,
    valid_sessions_dic=None,
    num_examples=6,
    save_dir=None,
    plot_smooth_sigma=10,
):
    """Run regression for every mouse_recday and produce per-mouse and cross-mouse summaries.

    Parameters
    ----------
    data_dic : dict
        Full data dictionary keyed by mouse_recday.
    config : RegressionConfig
    valid_sessions_dic : dict, optional
        {mouse_recday: [session_list]}. If None all sessions are used.
    num_examples : int
        Number of example neurons to show per mouse (default 6).
    save_dir : str, optional
        Directory to save figures. If None, figures are only displayed.

    Returns
    -------
    all_results : dict
        {mouse_recday: results_tc}
    all_nz_corrs : dict
        {mouse_recday: array of mean tuning correlations for NZ-lag neurons}
    """
    import matplotlib.pyplot as plt
    import os

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    all_results = {}
    all_nz_corrs = {}            # 360-bin tuning correlations, NZ-lag neurons
    all_nz_state_corrs = {}      # 4-state-MEAN correlations, NZ-lag neurons
    all_nz_state_corrs_max = {}  # 4-state-MAX  correlations, NZ-lag neurons

    mouse_recdays = list(data_dic.keys())

    for mouse_recday in mouse_recdays:
        print(f"\n{'='*60}")
        print(f"Processing {mouse_recday}")
        print('=' * 60)

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

        nz_mask, peak_lags = identify_nonzero_lag_neurons(results_tc, config)
        mean_corrs = results_tc['mean_tuning_correlations']
        valid_nz = nz_mask & ~np.isnan(mean_corrs)
        nz_indices = np.where(valid_nz)[0]

        print(f"  Non-zero lag neurons: {len(nz_indices)}/{np.sum(~np.isnan(mean_corrs))}")
        all_nz_corrs[mouse_recday] = mean_corrs[valid_nz]

        # Also collect the 4-state-mean correlation for the same NZ-lag neurons.
        mean_state = results_tc.get('mean_state_correlations')
        if mean_state is not None:
            all_nz_state_corrs[mouse_recday] = mean_state[valid_nz]

        # And the 4-state-MAX correlation (peak firing per state window).
        mean_state_max = results_tc.get('mean_state_correlations_max')
        if mean_state_max is not None:
            all_nz_state_corrs_max[mouse_recday] = mean_state_max[valid_nz]

        if len(nz_indices) == 0:
            print("  No non-zero lag neurons – skipping plots.")
            continue

        top_nz = nz_indices[np.argsort(mean_corrs[nz_indices])[::-1]][:num_examples]

        save_prefix = os.path.join(save_dir, mouse_recday) if save_dir else None

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

        fig_betas = plot_example_betas(
            results_tc, config,
            neuron_indices=top_nz,
        )
        if save_prefix and fig_betas is not None:
            fig_betas.savefig(f"{save_prefix}_betas_nz.svg", bbox_inches='tight')
        if fig_betas is not None:
            plt.close(fig_betas)

        fig_dist = plot_tuning_correlation_distribution(
            results_tc, config, mouse_recday=mouse_recday,
        )
        if save_prefix and fig_dist is not None:
            fig_dist.savefig(f"{save_prefix}_corr_dist.svg", bbox_inches='tight')
        if fig_dist is not None:
            plt.close(fig_dist)

    # Cross-mouse summary
    if len(all_nz_corrs) == 0:
        print("No results to summarise.")
        return all_results, all_nz_corrs

    fig_sum, axes = plt.subplots(1, 2, figsize=(14, 5))

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

    ax = axes[1]
    labels = list(all_nz_corrs.keys())
    means = [np.nanmean(v) for v in all_nz_corrs.values()]
    sems = [np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))) for v in all_nz_corrs.values()]
    ns = [np.sum(~np.isnan(v)) for v in all_nz_corrs.values()]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sems, color='darkorange', alpha=0.7, capsize=5, edgecolor='black')
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

    # ------------------------------------------------------------------
    # Parallel cross-mouse summary for the 4-state-mean correlation
    # (goal-progress confound-controlled)
    # ------------------------------------------------------------------
    if all_nz_state_corrs:
        fig_state, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        pooled_st = np.concatenate(list(all_nz_state_corrs.values()))
        pooled_st_valid = pooled_st[~np.isnan(pooled_st)]
        ax.hist(pooled_st_valid, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        mean_pool_st = np.mean(pooled_st_valid)
        ax.axvline(mean_pool_st, color='red', linewidth=2, label=f'Mean = {mean_pool_st:.3f}')
        if len(pooled_st_valid) > 1:
            t_st, p_st = stats.ttest_1samp(pooled_st_valid, 0)
            ax.text(0.97, 0.97,
                    f'n = {len(pooled_st_valid)}\nt = {t_st:.2f}\np = {p_st:.2e}',
                    transform=ax.transAxes, fontsize=10,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('State-mean correlation (4 states)', fontsize=11)
        ax.set_ylabel('Number of neurons', fontsize=11)
        ax.set_title('Non-zero lag neurons – pooled across mice\n(goal-progress confound-controlled)',
                     fontsize=12, fontweight='bold')
        ax.legend()

        ax = axes[1]
        labels_st = list(all_nz_state_corrs.keys())
        means_st = [np.nanmean(v) for v in all_nz_state_corrs.values()]
        sems_st = [np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))) for v in all_nz_state_corrs.values()]
        ns_st = [np.sum(~np.isnan(v)) for v in all_nz_state_corrs.values()]
        x = np.arange(len(labels_st))
        ax.bar(x, means_st, yerr=sems_st, color='steelblue', alpha=0.7, capsize=5, edgecolor='black')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels_st, ns_st)],
                           rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Mean state-mean correlation', fontsize=11)
        ax.set_title('Per-mouse mean r (4-state, NZ-lag neurons)', fontsize=12, fontweight='bold')

        plt.suptitle('Cross-mouse summary – 4-state correlation (goal-progress confound-controlled)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            fig_state.savefig(os.path.join(save_dir, 'cross_mouse_summary_state.svg'),
                              bbox_inches='tight')
        plt.show()

    # ------------------------------------------------------------------
    # Parallel cross-mouse summary for the 4-state-MAX correlation
    # (peak firing per state window — punctate state responses)
    # ------------------------------------------------------------------
    if all_nz_state_corrs_max:
        fig_state_max, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        pooled_mx = np.concatenate(list(all_nz_state_corrs_max.values()))
        pooled_mx_valid = pooled_mx[~np.isnan(pooled_mx)]
        ax.hist(pooled_mx_valid, bins=40, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        mean_pool_mx = np.mean(pooled_mx_valid)
        ax.axvline(mean_pool_mx, color='red', linewidth=2, label=f'Mean = {mean_pool_mx:.3f}')
        if len(pooled_mx_valid) > 1:
            t_mx, p_mx = stats.ttest_1samp(pooled_mx_valid, 0)
            ax.text(0.97, 0.97,
                    f'n = {len(pooled_mx_valid)}\nt = {t_mx:.2f}\np = {p_mx:.2e}',
                    transform=ax.transAxes, fontsize=10,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('State-MAX correlation (4 states)', fontsize=11)
        ax.set_ylabel('Number of neurons', fontsize=11)
        ax.set_title('Non-zero lag neurons – pooled across mice\n(state-MAX: peak firing per state window)',
                     fontsize=12, fontweight='bold')
        ax.legend()

        ax = axes[1]
        labels_mx = list(all_nz_state_corrs_max.keys())
        means_mx = [np.nanmean(v) for v in all_nz_state_corrs_max.values()]
        sems_mx = [np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))) for v in all_nz_state_corrs_max.values()]
        ns_mx = [np.sum(~np.isnan(v)) for v in all_nz_state_corrs_max.values()]
        x = np.arange(len(labels_mx))
        ax.bar(x, means_mx, yerr=sems_mx, color='purple', alpha=0.7, capsize=5, edgecolor='black')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels_mx, ns_mx)],
                           rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Mean state-MAX correlation', fontsize=11)
        ax.set_title('Per-mouse mean r (4-state MAX, NZ-lag neurons)', fontsize=12, fontweight='bold')

        plt.suptitle('Cross-mouse summary – 4-state MAX correlation (peak per state window)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            fig_state_max.savefig(os.path.join(save_dir, 'cross_mouse_summary_state_max.svg'),
                                  bbox_inches='tight')
        plt.show()

    return all_results, all_nz_corrs


if __name__ == '__main__':
    print("Elastic Net Regression Analysis Module (slim)")
    print("=" * 50)
    print("\nExpected data format:")
    print("  - Neurons_norm: shape (num_neurons, num_trials, 360)")
    print("  - Locs_norm: shape (num_trials, 360)")
    print("  - Locations 1-9 = nodes (used), 10-21 = edges (ignored)")
