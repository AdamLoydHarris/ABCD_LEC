# Time Coding Analysis for LEC Neurons
# Analyze coding of elapsed time at three timescales: session, trial, and state

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_1samp, mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')


def calculate_time_variables(data_dic, mouse_recday, session, sampling_rate=40):
    """
    Calculate elapsed time at three timescales for each timepoint.
    
    Parameters:
    -----------
    data_dic : dict
        Dictionary containing neural data
    mouse_recday : str
        Mouse and recording day identifier
    session : int
        Session number
    sampling_rate : int
        Sampling rate in Hz
        
    Returns:
    --------
    time_from_session_start : array
        Time elapsed from start of session (seconds)
    time_from_trial_start : array
        Time elapsed from start of current trial (seconds)
    time_from_state_start : array
        Time elapsed from start of current state (seconds)
    trial_labels : array
        Trial index for each timepoint
    state_labels : array
        State label (0-3) for each timepoint
    """
    
    # Get trial times for this session
    if 'Trial_times' in data_dic[mouse_recday]:
        trial_times = data_dic[mouse_recday]['Trial_times'][session]
    else:
        # Try alternate structure
        trial_times = data_dic[mouse_recday][session]['Trial_times']
    
    num_trials = trial_times.shape[0]
    
    # Each trial has 360 timepoints (4 states × 90 bins each)
    timepoints_per_trial = 360
    state_length = 90  # timepoints per state
    
    # Total timepoints in session
    total_timepoints = num_trials * timepoints_per_trial
    
    # Initialize arrays
    time_from_session_start = np.zeros(total_timepoints)
    time_from_trial_start = np.zeros(total_timepoints)
    time_from_state_start = np.zeros(total_timepoints)
    trial_labels = np.zeros(total_timepoints, dtype=int)
    state_labels = np.zeros(total_timepoints, dtype=int)
    
    # Fill in time variables for each trial
    for trial_idx in range(num_trials):
        start_idx = trial_idx * timepoints_per_trial
        end_idx = start_idx + timepoints_per_trial
        
        # Trial labels
        trial_labels[start_idx:end_idx] = trial_idx
        
        # Time from session start (cumulative)
        session_start_time = trial_idx * timepoints_per_trial / sampling_rate
        time_from_session_start[start_idx:end_idx] = session_start_time + np.arange(timepoints_per_trial) / sampling_rate
        
        # Time from trial start (resets each trial)
        time_from_trial_start[start_idx:end_idx] = np.arange(timepoints_per_trial) / sampling_rate
        
        # Time from state start and state labels (resets at state boundaries)
        for state_idx in range(4):  # 4 states: A, B, C, D
            state_start = start_idx + state_idx * state_length
            state_end = state_start + state_length
            
            state_labels[state_start:state_end] = state_idx
            time_from_state_start[state_start:state_end] = np.arange(state_length) / sampling_rate
    
    return (time_from_session_start, time_from_trial_start, time_from_state_start,
            trial_labels, state_labels, num_trials)


def analyze_single_cell_time_coding(neuron_activity, time_variable, method='pearson'):
    """
    Analyze single neuron's coding of time variable.
    
    Parameters:
    -----------
    neuron_activity : array
        Firing rates for a single neuron across time
    time_variable : array
        Time variable (session, trial, or state elapsed time)
    method : str
        'pearson' or 'spearman' correlation
        
    Returns:
    --------
    correlation : float
        Correlation coefficient
    p_value : float
        P-value
    r_squared : float
        R-squared from linear regression
    slope : float
        Slope from linear regression
    """
    
    # Remove NaN values
    valid_mask = ~np.isnan(neuron_activity)
    neuron_clean = neuron_activity[valid_mask]
    time_clean = time_variable[valid_mask]
    
    if len(neuron_clean) < 10:
        return np.nan, np.nan, np.nan, np.nan
    
    # Correlation
    if method == 'pearson':
        corr, p_val = pearsonr(time_clean, neuron_clean)
    else:
        corr, p_val = spearmanr(time_clean, neuron_clean)
    
    # Linear regression
    X = time_clean.reshape(-1, 1)
    y = neuron_clean
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    slope = reg.coef_[0]
    
    return corr, p_val, r_squared, slope


def analyze_time_coding_all_neurons(data_dic, session_inds_dic, mouse_recday, 
                                    neuron_filter=None, sampling_rate=40):
    """
    Analyze time coding for all neurons in a recording day.
    
    Parameters:
    -----------
    data_dic : dict
        Dictionary containing neural data
    session_inds_dic : dict
        Dictionary mapping mouse_recday to valid session indices
    mouse_recday : str
        Mouse and recording day identifier
    neuron_filter : array, optional
        Indices of neurons to analyze (e.g., LEC neurons only)
    sampling_rate : int
        Sampling rate in Hz
        
    Returns:
    --------
    results_df : pandas DataFrame
        Results for each neuron and time scale
    """
    
    results_list = []
    
    print(f"\nAnalyzing {mouse_recday}")
    
    # Get neural data for this mouse_recday
    if mouse_recday not in norm_neurons_dic:
        print(f"  No neural data found for {mouse_recday}")
        return results_list
    
    # Get valid sessions from session_inds_dic
    if mouse_recday not in session_inds_dic:
        print(f"  No valid sessions for {mouse_recday}")
        return results_list
    
    valid_sessions = session_inds_dic[mouse_recday]
    
    for session_idx, session in enumerate(valid_sessions):
        print(f"  Session {session_idx}")
        
        # Get the neural data for this session (2D: neurons × concatenated_timepoints)
        neuron_raw = norm_neurons_dic[mouse_recday][session_idx]
        
        # Check shape of neural data
        if neuron_raw.ndim != 2:
            print(f"    Warning: Expected 2D neural data, got {neuron_raw.ndim}D. Skipping.")
            continue
        
        num_neurons = neuron_raw.shape[0]
        total_timepoints = neuron_raw.shape[1]
        
        # Calculate time variables
        time_vars = calculate_time_variables(data_dic, mouse_recday, session, sampling_rate)
        time_session, time_trial, time_state, trial_labels, state_labels, num_trials = time_vars
        
        # Verify dimensions match
        if len(time_session) != total_timepoints:
            print(f"    Warning: Dimension mismatch. Time variables length {len(time_session)} != neural data timepoints {total_timepoints}. Skipping.")
            continue
        
        print(f"    Analyzing {num_neurons} neurons, {num_trials} trials, {total_timepoints} total timepoints")
        
        # Apply neuron filter if provided
        if neuron_filter is not None:
            if len(neuron_filter) != num_neurons:
                print(f"    Warning: Neuron filter length {len(neuron_filter)} != num_neurons {num_neurons}. Skipping.")
                continue
            neuron_indices = np.where(neuron_filter)[0] if neuron_filter.dtype == bool else neuron_filter
        else:
            neuron_indices = np.arange(num_neurons)
        
        # For each neuron
        for neuron_idx in range(num_neurons):
            # Skip if not in filter
            if neuron_filter is not None:
                if neuron_filter.dtype == bool:
                    if not neuron_filter[neuron_idx]:
                        continue
                else:
                    if neuron_idx not in neuron_filter:
                        continue
            
            # Get firing rates for this neuron (already flattened across trials)
            neuron_data = neuron_raw[neuron_idx]
            
            # Analyze coding of session time
            corr_sess, p_sess, r2_sess, slope_sess = analyze_single_cell_time_coding(
                neuron_data, time_session, method='pearson'
            )
            
            # Analyze coding of trial time
            corr_trial, p_trial, r2_trial, slope_trial = analyze_single_cell_time_coding(
                neuron_data, time_trial, method='pearson'
            )
            
            # Analyze coding of state time
            corr_state, p_state, r2_state, slope_state = analyze_single_cell_time_coding(
                neuron_data, time_state, method='pearson'
            )
            
            results_list.append({
                'mouse_recday': mouse_recday,
                'session': session,
                'neuron_idx': global_neuron_idx,
                'session_time_corr': corr_sess,
                'session_time_pval': p_sess,
                'session_time_r2': r2_sess,
                'session_time_slope': slope_sess,
                'trial_time_corr': corr_trial,
                'trial_time_pval': p_trial,
                'trial_time_r2': r2_trial,
                'trial_time_slope': slope_trial,
                'state_time_corr': corr_state,
                'state_time_pval': p_state,
                'state_time_r2': r2_state,
                'state_time_slope': slope_state
            })
    
    results_df = pd.DataFrame(results_list)
    return results_df


def population_time_decoding(data_dic, session_inds_dic, mouse_recday,
                             neuron_filter=None, sampling_rate=40, n_folds=5):
    """
    Decode elapsed time from population activity using cross-validated regression.
    
    Parameters:
    -----------
    data_dic : dict
        Dictionary containing neural data
    session_inds_dic : dict
        Dictionary mapping mouse_recday to valid session indices
    mouse_recday : str
        Mouse and recording day identifier
    neuron_filter : array, optional
        Indices of neurons to analyze
    sampling_rate : int
        Sampling rate in Hz
    n_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    decoding_results : dict
        Decoding performance for each time scale
    """
    
    if mouse_recday not in session_inds_dic:
        return None
    
    valid_sessions = session_inds_dic[mouse_recday]
    
    results = {
        'session_time': [],
        'trial_time': [],
        'state_time': []
    }
    
    print(f"\nPopulation decoding for {mouse_recday}")
    
    # Get neural data for this mouse_recday
    if mouse_recday not in norm_neurons_dic:
        print(f"  No neural data found for {mouse_recday}")
        return None
    
    # Get valid sessions from session_inds_dic
    if mouse_recday not in session_inds_dic:
        print(f"  No valid sessions for {mouse_recday}")
        return None
    
    valid_sessions = session_inds_dic[mouse_recday]
    
    for session_idx, session in enumerate(valid_sessions):
        print(f"  Session {session_idx}")
        
        # Get the neural data for this session (2D: neurons × concatenated_timepoints)
        neuron_raw = norm_neurons_dic[mouse_recday][session_idx]
        
        # Check shape
        if neuron_raw.ndim != 2:
            print(f"    Warning: Expected 2D neural data, got {neuron_raw.ndim}D. Skipping.")
            continue
        
        num_neurons = neuron_raw.shape[0]
        total_timepoints = neuron_raw.shape[1]
        
        # Calculate time variables
        time_vars = calculate_time_variables(data_dic, mouse_recday, session, sampling_rate)
        time_session, time_trial, time_state, trial_labels, state_labels, num_trials = time_vars
        
        # Verify dimensions
        if len(time_session) != total_timepoints:
            print(f"    Warning: Dimension mismatch. Time variables ({len(time_session)}) != neural data ({total_timepoints}). Skipping.")
            continue
        
        print(f"    Using {num_neurons} neurons, {num_trials} trials, {total_timepoints} total timepoints")
        
        # Prepare population matrix: (timepoints, neurons)
        # Transpose 2D data: (neurons, timepoints) -> (timepoints, neurons)
        if neuron_filter is not None:
            if len(neuron_filter) != num_neurons:
                print(f"    Warning: Neuron filter length {len(neuron_filter)} != num_neurons {num_neurons}. Skipping.")
                continue
            neuron_indices = np.where(neuron_filter)[0] if neuron_filter.dtype == bool else neuron_filter
            X = neuron_raw[neuron_indices].T
        else:
            X = neuron_raw.T
        
        # Remove NaN rows
        valid_rows = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_rows]
        time_session_clean = time_session[valid_rows]
        time_trial_clean = time_trial[valid_rows]
        time_state_clean = time_state[valid_rows]
        
        if len(X_clean) < 50:
            print(f"    Warning: Too few valid timepoints ({len(X_clean)}). Skipping.")
            continue
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # Cross-validated regression for each time scale
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Session time
        reg_session = LinearRegression()
        scores_session = cross_val_score(reg_session, X_scaled, time_session_clean,
                                        cv=kfold, scoring='r2')
        results['session_time'].append({
            'mouse_recday': mouse_recday,
            'session': session,
            'r2_mean': np.mean(scores_session),
            'r2_std': np.std(scores_session)
        })
        
        # Trial time
        reg_trial = LinearRegression()
        scores_trial = cross_val_score(reg_trial, X_scaled, time_trial_clean,
                                      cv=kfold, scoring='r2')
        results['trial_time'].append({
            'mouse_recday': mouse_recday,
            'session': session,
            'r2_mean': np.mean(scores_trial),
            'r2_std': np.std(scores_trial)
        })
        
        # State time
        reg_state = LinearRegression()
        scores_state = cross_val_score(reg_state, X_scaled, time_state_clean,
                                      cv=kfold, scoring='r2')
        results['state_time'].append({
            'mouse_recday': mouse_recday,
            'session': session,
            'r2_mean': np.mean(scores_state),
            'r2_std': np.std(scores_state)
        })
        
        print(f"      Session time R² = {np.mean(scores_session):.3f}")
        print(f"      Trial time R² = {np.mean(scores_trial):.3f}")
        print(f"      State time R² = {np.mean(scores_state):.3f}")
    
    return results


def plot_single_cell_results(results_df, title_suffix=""):
    """
    Plot single-cell time coding results.
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Results from analyze_time_coding_all_neurons
    title_suffix : str
        Suffix for plot titles
    """
    
    if len(results_df) == 0:
        print("No results to plot")
        return
    
    # Significance threshold
    alpha = 0.05
    
    # Count significant neurons for each time scale
    sig_session = np.sum(results_df['session_time_pval'] < alpha)
    sig_trial = np.sum(results_df['trial_time_pval'] < alpha)
    sig_state = np.sum(results_df['state_time_pval'] < alpha)
    total_neurons = len(results_df)
    
    print(f"\n{'='*80}")
    print(f"SINGLE-CELL TIME CODING SUMMARY - {title_suffix}")
    print(f"{'='*80}")
    print(f"Total neurons: {total_neurons}")
    print(f"Session time: {sig_session} ({100*sig_session/total_neurons:.1f}%) significant")
    print(f"Trial time: {sig_trial} ({100*sig_trial/total_neurons:.1f}%) significant")
    print(f"State time: {sig_state} ({100*sig_state/total_neurons:.1f}%) significant")
    
    # Create comprehensive plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    time_scales = ['session_time', 'trial_time', 'state_time']
    time_labels = ['Session Time', 'Trial Time', 'State Time']
    
    for idx, (time_scale, time_label) in enumerate(zip(time_scales, time_labels)):
        # Correlation distribution
        ax = axes[idx, 0]
        corr_data = results_df[f'{time_scale}_corr'].values
        corr_data = corr_data[~np.isnan(corr_data)]
        
        sns.histplot(corr_data, kde=True, bins=50, ax=ax, color='steelblue', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=np.mean(corr_data), color='black', linestyle='--', 
                  linewidth=2, label=f'Mean={np.mean(corr_data):.3f}')
        ax.set_xlabel('Correlation with Time', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{time_label} - Correlation Distribution', fontsize=13)
        ax.legend()
        sns.despine(ax=ax)
        
        # R-squared distribution
        ax = axes[idx, 1]
        r2_data = results_df[f'{time_scale}_r2'].values
        r2_data = r2_data[~np.isnan(r2_data)]
        
        sns.histplot(r2_data, kde=True, bins=50, ax=ax, color='coral', alpha=0.7)
        ax.axvline(x=np.mean(r2_data), color='black', linestyle='--',
                  linewidth=2, label=f'Mean={np.mean(r2_data):.3f}')
        ax.set_xlabel('R²', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{time_label} - R² Distribution', fontsize=13)
        ax.legend()
        sns.despine(ax=ax)
        
        # Slope distribution
        ax = axes[idx, 2]
        slope_data = results_df[f'{time_scale}_slope'].values
        slope_data = slope_data[~np.isnan(slope_data)]
        
        sns.histplot(slope_data, kde=True, bins=50, ax=ax, color='forestgreen', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=np.mean(slope_data), color='black', linestyle='--',
                  linewidth=2, label=f'Mean={np.mean(slope_data):.3f}')
        ax.set_xlabel('Slope (Hz/sec)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{time_label} - Slope Distribution', fontsize=13)
        ax.legend()
        sns.despine(ax=ax)
    
    plt.suptitle(f'Single-Cell Time Coding Analysis - {title_suffix}', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    print(f"\n{'='*80}")
    print(f"STATISTICAL TESTS - {title_suffix}")
    print(f"{'='*80}")
    
    for time_scale, time_label in zip(time_scales, time_labels):
        corr_data = results_df[f'{time_scale}_corr'].values
        corr_data = corr_data[~np.isnan(corr_data)]
        
        t_stat, p_val = ttest_1samp(corr_data, 0)
        
        print(f"\n{time_label}:")
        print(f"  Mean correlation: {np.mean(corr_data):.4f} ± {np.std(corr_data):.4f}")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_val:.3e}")


def plot_population_decoding_results(decoding_results, title_suffix=""):
    """
    Plot population-level time decoding results.
    
    Parameters:
    -----------
    decoding_results : dict
        Results from population_time_decoding
    title_suffix : str
        Suffix for plot titles
    """
    
    # Convert to DataFrames
    session_df = pd.DataFrame(decoding_results['session_time'])
    trial_df = pd.DataFrame(decoding_results['trial_time'])
    state_df = pd.DataFrame(decoding_results['state_time'])
    
    if len(session_df) == 0:
        print("No decoding results to plot")
        return
    
    print(f"\n{'='*80}")
    print(f"POPULATION TIME DECODING SUMMARY - {title_suffix}")
    print(f"{'='*80}")
    print(f"Session time: R² = {session_df['r2_mean'].mean():.3f} ± {session_df['r2_mean'].std():.3f}")
    print(f"Trial time: R² = {trial_df['r2_mean'].mean():.3f} ± {trial_df['r2_mean'].std():.3f}")
    print(f"State time: R² = {state_df['r2_mean'].mean():.3f} ± {state_df['r2_mean'].std():.3f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot
    ax = axes[0]
    time_scales = ['Session', 'Trial', 'State']
    means = [session_df['r2_mean'].mean(), trial_df['r2_mean'].mean(), state_df['r2_mean'].mean()]
    stds = [session_df['r2_mean'].std(), trial_df['r2_mean'].std(), state_df['r2_mean'].std()]
    
    bars = ax.bar(time_scales, means, yerr=stds, capsize=10, 
                  color=['steelblue', 'coral', 'forestgreen'], alpha=0.7)
    ax.set_ylabel('Cross-Validated R²', fontsize=14)
    ax.set_xlabel('Time Scale', fontsize=14)
    ax.set_title(f'Population Time Decoding - {title_suffix}', fontsize=15)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    sns.despine(ax=ax)
    
    # Violin plot
    ax = axes[1]
    combined_df = pd.DataFrame({
        'R²': np.concatenate([session_df['r2_mean'].values, 
                             trial_df['r2_mean'].values,
                             state_df['r2_mean'].values]),
        'Time Scale': (['Session'] * len(session_df) + 
                      ['Trial'] * len(trial_df) + 
                      ['State'] * len(state_df))
    })
    
    sns.violinplot(data=combined_df, x='Time Scale', y='R²', 
                  palette=['steelblue', 'coral', 'forestgreen'], ax=ax)
    ax.set_ylabel('Cross-Validated R²', fontsize=14)
    ax.set_xlabel('Time Scale', fontsize=14)
    ax.set_title('Distribution Comparison', fontsize=15)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.show()


# Example usage:

# CELL 1: Analyze all neurons - single cell level
print("="*80)
print("SINGLE-CELL TIME CODING - ALL NEURONS")
print("="*80)

all_neurons_results = []
for mouse_recday in data_dic.keys():
    result = analyze_time_coding_all_neurons(
        data_dic, session_inds_dic, mouse_recday,
        neuron_filter=None
    )
    if result is not None:
        all_neurons_results.append(result)

if len(all_neurons_results) > 0:
    all_neurons_df = pd.concat(all_neurons_results, ignore_index=True)
    print(f"\nTotal results: {len(all_neurons_df)}")
    print(all_neurons_df.head())

# CELL 2: Analyze LEC neurons only - single cell level  
print("\n" + "="*80)
print("SINGLE-CELL TIME CODING - LEC NEURONS ONLY")
print("="*80)

# Note: Need to get LEC spatial filter (not state tuning filter)
# This would be something like lec_spatial_neurons_dic that identifies neurons in LEC anatomically
# For now, using sig_lec_neurons_dic as a placeholder - replace with spatial filter

lec_neurons_results = []
for mouse_recday in data_dic.keys():
    if mouse_recday in sig_lec_neurons_dic:  # Replace with LEC spatial filter
        result = analyze_time_coding_all_neurons(
            data_dic, session_inds_dic, mouse_recday,
            neuron_filter=sig_lec_neurons_dic[mouse_recday]  # Replace with spatial filter
        )
        if result is not None:
            lec_neurons_results.append(result)

if len(lec_neurons_results) > 0:
    lec_neurons_df = pd.concat(lec_neurons_results, ignore_index=True)
    print(f"\nTotal results: {len(lec_neurons_df)}")
    print(lec_neurons_df.head())

# CELL 3: Plot single-cell results for all neurons
if len(all_neurons_results) > 0:
    plot_single_cell_results(all_neurons_df, "All Neurons")

# CELL 4: Plot single-cell results for LEC neurons
if len(lec_neurons_results) > 0:
    plot_single_cell_results(lec_neurons_df, "LEC Neurons")

# CELL 5: Population decoding - all neurons
print("\n" + "="*80)
print("POPULATION TIME DECODING - ALL NEURONS")
print("="*80)

all_neurons_decoding = {}
for key in ['session_time', 'trial_time', 'state_time']:
    all_neurons_decoding[key] = []

for mouse_recday in data_dic.keys():
    result = population_time_decoding(
        data_dic, session_inds_dic, mouse_recday,
        neuron_filter=None
    )
    if result is not None:
        for key in ['session_time', 'trial_time', 'state_time']:
            all_neurons_decoding[key].extend(result[key])

# CELL 6: Population decoding - LEC neurons
print("\n" + "="*80)
print("POPULATION TIME DECODING - LEC NEURONS ONLY")
print("="*80)

lec_neurons_decoding = {}
for key in ['session_time', 'trial_time', 'state_time']:
    lec_neurons_decoding[key] = []

for mouse_recday in data_dic.keys():
    if mouse_recday in sig_lec_neurons_dic:  # Replace with spatial filter
        result = population_time_decoding(
            data_dic, session_inds_dic, mouse_recday,
            neuron_filter=sig_lec_neurons_dic[mouse_recday]  # Replace with spatial filter
        )
        if result is not None:
            for key in ['session_time', 'trial_time', 'state_time']:
                lec_neurons_decoding[key].extend(result[key])

# CELL 7: Plot population decoding results - all neurons
if len(all_neurons_decoding['session_time']) > 0:
    plot_population_decoding_results(all_neurons_decoding, "All Neurons")

# CELL 8: Plot population decoding results - LEC neurons
if len(lec_neurons_decoding['session_time']) > 0:
    plot_population_decoding_results(lec_neurons_decoding, "LEC Neurons")

# CELL 9: Compare all neurons vs LEC neurons
print("\n" + "="*80)
print("COMPARISON: ALL NEURONS vs LEC NEURONS")
print("="*80)

if len(all_neurons_df) > 0 and len(lec_neurons_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    time_scales = ['session_time', 'trial_time', 'state_time']
    time_labels = ['Session Time', 'Trial Time', 'State Time']
    
    for idx, (time_scale, time_label) in enumerate(zip(time_scales, time_labels)):
        ax = axes[idx]
        
        all_corrs = all_neurons_df[f'{time_scale}_corr'].dropna().values
        lec_corrs = lec_neurons_df[f'{time_scale}_corr'].dropna().values
        
        # Mann-Whitney U test
        u_stat, p_val = mannwhitneyu(all_corrs, lec_corrs, alternative='two-sided')
        
        # Plot
        combined = pd.DataFrame({
            'Correlation': np.concatenate([all_corrs, lec_corrs]),
            'Group': ['All'] * len(all_corrs) + ['LEC'] * len(lec_corrs)
        })
        
        sns.violinplot(data=combined, x='Group', y='Correlation', 
                      palette=['lightcoral', 'steelblue'], ax=ax)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_xlabel('', fontsize=12)
        ax.set_title(f'{time_label}\n(p={p_val:.3e})', fontsize=13)
        sns.despine(ax=ax)
        
        print(f"\n{time_label}:")
        print(f"  All neurons: {np.mean(all_corrs):.4f} ± {np.std(all_corrs):.4f}")
        print(f"  LEC neurons: {np.mean(lec_corrs):.4f} ± {np.std(lec_corrs):.4f}")
        print(f"  Mann-Whitney U: {u_stat:.2f}, p={p_val:.3e}")
    
    plt.suptitle('Time Coding: All Neurons vs LEC Neurons', fontsize=16)
    plt.tight_layout()
    plt.show()
