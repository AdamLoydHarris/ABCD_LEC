# State Decoding Analysis using SVM-RBF with Leave-One-Out Cross-Validation
# Uses 1-second bins and all neurons

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import binned_statistic
import warnings
warnings.filterwarnings('ignore')


def bin_neural_data_1sec(neuron_raw, trial_times, sampling_rate=40):
    """
    Bin neural data into 1-second bins for each trial and state.
    
    Parameters:
    -----------
    neuron_raw : np.ndarray
        Neural data of shape (num_neurons, time_points)
    trial_times : np.ndarray
        Trial times of shape (num_trials, num_states+1) with time indices for state transitions
    sampling_rate : int
        Sampling rate in Hz (default 40 Hz)
        
    Returns:
    --------
    binned_data : list of np.ndarray
        List of binned data arrays, one per trial-state combination
    state_labels : list of int
        Corresponding state labels (0=A, 1=B, 2=C, 3=D)
    trial_indices : list of int
        Corresponding trial indices
    """
    num_neurons = neuron_raw.shape[0]
    num_states = 4
    bin_size = sampling_rate  # 1 second = sampling_rate points
    
    binned_data = []
    state_labels = []
    trial_indices = []
    
    for trial_idx, trial in enumerate(trial_times):
        for state_idx in range(num_states):
            # Get start and end times for this state
            start_time = int(trial[state_idx])
            end_time = int(trial[state_idx + 1])
            
            # Extract neural data for this state
            state_data = neuron_raw[:, start_time:end_time]
            
            # Calculate number of complete 1-second bins
            num_bins = (end_time - start_time) // bin_size
            
            if num_bins > 0:
                # Bin the data into 1-second bins
                for bin_idx in range(num_bins):
                    bin_start = bin_idx * bin_size
                    bin_end = (bin_idx + 1) * bin_size
                    
                    # Average firing rate across this 1-second bin
                    bin_fr = np.mean(state_data[:, bin_start:bin_end], axis=1)
                    
                    binned_data.append(bin_fr)
                    state_labels.append(state_idx)
                    trial_indices.append(trial_idx)
    
    return binned_data, state_labels, trial_indices


def decode_states_loo_cv(data_dic, session_inds_dic, tasks_dic, mouse_recday, 
                          significant_neurons=None, sampling_rate=40):
    """
    Decode states using SVM-RBF with leave-one-task-out cross-validation.
    
    Parameters:
    -----------
    data_dic : dict
        Dictionary containing neural data
    session_inds_dic : dict
        Dictionary mapping mouse_recday to valid session indices
    tasks_dic : dict
        Dictionary containing task information
    mouse_recday : str
        Mouse and recording day identifier
    significant_neurons : dict, optional
        Dictionary mapping mouse_recday to indices of significant neurons
    sampling_rate : int
        Sampling rate in Hz
        
    Returns:
    --------
    results : dict
        Dictionary containing accuracy scores, predictions, and other metrics
    """
    if mouse_recday not in session_inds_dic:
        print(f"No valid sessions for {mouse_recday}")
        return None
    
    valid_sessions = session_inds_dic[mouse_recday]
    
    # Get non-repeating task sessions
    session_tasks = []
    for session in valid_sessions:
        if mouse_recday in tasks_dic and session in tasks_dic[mouse_recday]:
            task = tasks_dic[mouse_recday][session]
            # Check if this task hasn't been seen yet
            if not any(np.array_equal(task, t) for _, t in session_tasks):
                session_tasks.append((session, task))
    
    if len(session_tasks) < 2:
        print(f"Need at least 2 non-repeating tasks for leave-one-task-out CV (found {len(session_tasks)})")
        return None
    
    print(f"\n{'='*80}")
    print(f"Decoding states for {mouse_recday}")
    print(f"Found {len(session_tasks)} non-repeating task sessions")
    print(f"Performing leave-one-task-out cross-validation")
    print(f"{'='*80}")
    
    # Prepare data for all tasks
    all_task_data = []
    
    for session, task in session_tasks:
        print(f"\nLoading session {session}")
        
        # Get neural data
        if 'Neuron_raw' not in data_dic[mouse_recday][session]:
            print(f"  No neural data found for session {session}")
            continue
        
        neuron_raw = data_dic[mouse_recday][session]['Neuron_raw']
        trial_times = data_dic[mouse_recday][session]['Trial_times']
        
        # Filter for significant neurons if provided
        if significant_neurons is not None and mouse_recday in significant_neurons:
            sig_neuron_inds = significant_neurons[mouse_recday]
            neuron_raw = neuron_raw[sig_neuron_inds]
            num_neurons = len(sig_neuron_inds)
        else:
            num_neurons = neuron_raw.shape[0]
        
        # Bin neural data into 1-second bins
        binned_data, state_labels, trial_indices = bin_neural_data_1sec(
            neuron_raw, trial_times, sampling_rate
        )
        
        if len(binned_data) == 0:
            print(f"  No data after binning")
            continue
        
        # Store data for this task
        all_task_data.append({
            'session': session,
            'X': np.array(binned_data),
            'y': np.array(state_labels),
            'trials': np.array(trial_indices),
            'num_neurons': num_neurons
        })
        
        print(f"  Loaded {len(binned_data)} samples from {len(np.unique(trial_indices))} trials")
    
    if len(all_task_data) < 2:
        print(f"Need at least 2 tasks with data (found {len(all_task_data)})")
        return None
    
    print(f"\n{'='*80}")
    print(f"Performing leave-one-task-out CV with {len(all_task_data)} tasks")
    print(f"Using {all_task_data[0]['num_neurons']} neurons")
    print(f"{'='*80}")
    
    # Leave-one-task-out cross-validation
    task_results = []
    all_predictions = []
    all_true_labels = []
    all_test_sessions = []
    
    for test_idx in range(len(all_task_data)):
        # Separate training and test tasks
        test_task = all_task_data[test_idx]
        train_tasks = [all_task_data[i] for i in range(len(all_task_data)) if i != test_idx]
        
        # Combine training data from all training tasks
        X_train_list = [task['X'] for task in train_tasks]
        y_train_list = [task['y'] for task in train_tasks]
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        X_test = test_task['X']
        y_test = test_task['y']
        
        # Check that all states are present in training data
        if len(np.unique(y_train)) < 4:
            print(f"\nWarning: Not all states present in training data for test task {test_task['session']}")
            continue
        
        print(f"\nFold {test_idx + 1}/{len(all_task_data)}: Testing on session {test_task['session']}")
        print(f"  Training samples: {len(X_train)} from {len(train_tasks)} tasks")
        print(f"  Test samples: {len(X_test)}")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM with RBF kernel
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm.fit(X_train_scaled, y_train)
        
        # Predict on test task
        y_pred = svm.predict(X_test_scaled)
        
        # Calculate accuracy for this fold
        fold_accuracy = accuracy_score(y_test, y_pred)
        
        # Per-state accuracy for this fold
        state_accuracies = []
        for state in range(4):
            state_mask = y_test == state
            if np.sum(state_mask) > 0:
                state_acc = accuracy_score(y_test[state_mask], y_pred[state_mask])
                state_accuracies.append(state_acc)
            else:
                state_accuracies.append(np.nan)
        
        print(f"  Accuracy: {fold_accuracy:.3f}")
        print(f"  State accuracies: A={state_accuracies[0]:.3f}, B={state_accuracies[1]:.3f}, "
              f"C={state_accuracies[2]:.3f}, D={state_accuracies[3]:.3f}")
        
        # Store predictions
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        all_test_sessions.extend([test_task['session']] * len(y_test))
        
        # Store fold results
        task_results.append({
            'test_session': test_task['session'],
            'fold_accuracy': fold_accuracy,
            'state_A_accuracy': state_accuracies[0],
            'state_B_accuracy': state_accuracies[1],
            'state_C_accuracy': state_accuracies[2],
            'state_D_accuracy': state_accuracies[3],
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        })
    
    # Calculate overall metrics across all folds
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    
    # Per-state accuracy overall
    state_accuracies_overall = []
    for state in range(4):
        state_mask = all_true_labels == state
        if np.sum(state_mask) > 0:
            state_acc = accuracy_score(
                all_true_labels[state_mask], 
                all_predictions[state_mask]
            )
            state_accuracies_overall.append(state_acc)
        else:
            state_accuracies_overall.append(np.nan)
    
    # Mean accuracy across folds
    fold_accuracies = [r['fold_accuracy'] for r in task_results]
    mean_fold_accuracy = np.mean(fold_accuracies)
    std_fold_accuracy = np.std(fold_accuracies)
    
    print(f"\n{'='*80}")
    print(f"Overall Results for {mouse_recday}:")
    print(f"  Overall accuracy (all predictions): {overall_accuracy:.3f}")
    print(f"  Mean fold accuracy: {mean_fold_accuracy:.3f} ± {std_fold_accuracy:.3f}")
    print(f"  State A accuracy: {state_accuracies_overall[0]:.3f}")
    print(f"  State B accuracy: {state_accuracies_overall[1]:.3f}")
    print(f"  State C accuracy: {state_accuracies_overall[2]:.3f}")
    print(f"  State D accuracy: {state_accuracies_overall[3]:.3f}")
    print(f"{'='*80}")
    
    # Return single result for this mouse_recday
    return [{
        'mouse_recday': mouse_recday,
        'overall_accuracy': overall_accuracy,
        'mean_fold_accuracy': mean_fold_accuracy,
        'std_fold_accuracy': std_fold_accuracy,
        'state_A_accuracy': state_accuracies_overall[0],
        'state_B_accuracy': state_accuracies_overall[1],
        'state_C_accuracy': state_accuracies_overall[2],
        'state_D_accuracy': state_accuracies_overall[3],
        'confusion_matrix': conf_matrix,
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels,
        'num_neurons': all_task_data[0]['num_neurons'],
        'num_tasks': len(all_task_data),
        'num_samples': len(all_predictions),
        'fold_results': task_results
    }]


def run_decoding_analysis_all_mice(data_dic, session_inds_dic, tasks_dic, 
                                    significant_neurons=None, sampling_rate=40):
    """
    Run state decoding analysis across all mice.
    
    Parameters:
    -----------
    data_dic : dict
        Dictionary containing neural data
    session_inds_dic : dict
        Dictionary mapping mouse_recday to valid session indices
    tasks_dic : dict
        Dictionary containing task information
    significant_neurons : dict, optional
        Dictionary mapping mouse_recday to indices of significant neurons
    sampling_rate : int
        Sampling rate in Hz
        
    Returns:
    --------
    all_results : list
        List of result dictionaries from all mice
    results_df : pd.DataFrame
        DataFrame with all results
    """
    all_results = []
    
    for mouse_recday in data_dic.keys():
        results = decode_states_loo_cv(
            data_dic, session_inds_dic, tasks_dic, mouse_recday,
            significant_neurons=significant_neurons,
            sampling_rate=sampling_rate
        )
        
        if results is not None:
            all_results.extend(results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['confusion_matrix', 'all_predictions', 'all_true_labels']}
        for r in all_results
    ])
    
    return all_results, results_df


def plot_decoding_results(all_results, results_df, title_suffix="All Neurons"):
    """
    Plot state decoding results.
    
    Parameters:
    -----------
    all_results : list
        List of result dictionaries
    results_df : pd.DataFrame
        DataFrame with results
    title_suffix : str
        Suffix for plot titles
    """
    if len(all_results) == 0:
        print("No results to plot")
        return
    
    # 1. Overall accuracy distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall accuracy histogram
    ax = axes[0, 0]
    overall_accs = results_df['overall_accuracy'].values
    chance_level = 0.25  # 4 states
    
    sns.histplot(overall_accs, kde=True, bins=20, ax=ax, color='skyblue', alpha=0.7)
    ax.axvline(x=chance_level, color='red', linestyle='--', linewidth=2, label='Chance (0.25)')
    ax.axvline(x=np.mean(overall_accs), color='black', linestyle='--', linewidth=2, 
              label=f'Mean = {np.mean(overall_accs):.3f}')
    ax.set_xlabel('Overall Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Overall Decoding Accuracy', fontsize=13)
    ax.legend()
    sns.despine(ax=ax)
    
    # Per-state accuracy comparison
    ax = axes[0, 1]
    state_data = results_df[['state_A_accuracy', 'state_B_accuracy', 
                             'state_C_accuracy', 'state_D_accuracy']].values
    state_means = np.nanmean(state_data, axis=0)
    state_stds = np.nanstd(state_data, axis=0)
    
    states = ['A', 'B', 'C', 'D']
    ax.bar(states, state_means, yerr=state_stds, capsize=5, alpha=0.7, color='lightgreen')
    ax.axhline(y=chance_level, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-State Decoding Accuracy', fontsize=13)
    ax.set_ylim([0, 1])
    ax.legend()
    sns.despine(ax=ax)
    
    # Aggregate confusion matrix
    ax = axes[1, 0]
    # Sum all confusion matrices
    total_conf_matrix = np.zeros((4, 4))
    for result in all_results:
        total_conf_matrix += result['confusion_matrix']
    
    # Normalize by row (true labels)
    conf_matrix_norm = total_conf_matrix / total_conf_matrix.sum(axis=1, keepdims=True)
    
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=states, yticklabels=states, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('Predicted State', fontsize=12)
    ax.set_ylabel('True State', fontsize=12)
    ax.set_title('Normalized Confusion Matrix\n(Aggregated Across All Sessions)', fontsize=13)
    
    # Accuracy by mouse
    ax = axes[1, 1]
    results_df['mouse'] = results_df['mouse_recday'].str.split('_').str[0]
    mouse_accs = results_df.groupby('mouse')['overall_accuracy'].agg(['mean', 'std', 'count'])
    
    ax.bar(mouse_accs.index, mouse_accs['mean'], yerr=mouse_accs['std'], 
           capsize=5, alpha=0.7, color='salmon')
    ax.axhline(y=chance_level, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_xlabel('Mouse', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title('Decoding Accuracy by Mouse', fontsize=13)
    ax.set_ylim([0, 1])
    ax.legend()
    sns.despine(ax=ax)
    
    plt.suptitle(f'State Decoding Results - {title_suffix}', fontsize=15, y=1.00)
    plt.tight_layout()
    plt.show()
    
    # 2. Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS - {title_suffix}")
    print(f"{'='*80}")
    print(f"\nOverall Accuracy:")
    print(f"  Mean: {np.mean(overall_accs):.3f} ± {np.std(overall_accs):.3f}")
    print(f"  Median: {np.median(overall_accs):.3f}")
    print(f"  Range: [{np.min(overall_accs):.3f}, {np.max(overall_accs):.3f}]")
    print(f"  Chance level: {chance_level:.3f}")
    
    print(f"\nPer-State Accuracy:")
    for i, state in enumerate(states):
        print(f"  State {state}: {state_means[i]:.3f} ± {state_stds[i]:.3f}")
    
    print(f"\nNumber of sessions analyzed: {len(all_results)}")
    print(f"Total samples across all sessions: {results_df['num_samples'].sum()}")


# Example usage:
# ==============

# CELL 1: Run decoding analysis for all neurons
print("="*80)
print("STATE DECODING ANALYSIS - ALL NEURONS")
print("="*80)

all_results_all, results_df_all = run_decoding_analysis_all_mice(
    data_dic, session_inds_dic, tasks_dic,
    significant_neurons=None,
    sampling_rate=40  # Adjust if your sampling rate is different
)

print(f"\n\nTotal sessions analyzed: {len(all_results_all)}")
if len(all_results_all) > 0:
    print(results_df_all.head())


# CELL 2: Plot results for all neurons
if len(all_results_all) > 0:
    plot_decoding_results(all_results_all, results_df_all, "All Neurons")


# CELL 3 (Optional): Run for significant LEC neurons only
# Uncomment if you want to also decode using only significant neurons
"""
print("\n" + "="*80)
print("STATE DECODING ANALYSIS - SIGNIFICANT LEC NEURONS")
print("="*80)

all_results_lec, results_df_lec = run_decoding_analysis_all_mice(
    data_dic, session_inds_dic, tasks_dic,
    significant_neurons=sig_lec_neurons_dic,
    sampling_rate=40
)

print(f"\n\nTotal sessions analyzed: {len(all_results_lec)}")
if len(all_results_lec) > 0:
    print(results_df_lec.head())
    plot_decoding_results(all_results_lec, results_df_lec, "Significant LEC Neurons")
"""


# CELL 4 (Optional): Compare all neurons vs significant LEC neurons
"""
if len(all_results_all) > 0 and len(all_results_lec) > 0:
    print("\n" + "="*80)
    print("COMPARISON: ALL NEURONS vs SIGNIFICANT LEC NEURONS")
    print("="*80)
    
    from scipy.stats import mannwhitneyu
    
    accs_all = results_df_all['overall_accuracy'].values
    accs_lec = results_df_lec['overall_accuracy'].values
    
    u_stat, p_value = mannwhitneyu(accs_all, accs_lec, alternative='two-sided')
    
    print(f"\nAll Neurons:")
    print(f"  Mean accuracy: {np.mean(accs_all):.3f} ± {np.std(accs_all):.3f}")
    print(f"  N sessions: {len(accs_all)}")
    
    print(f"\nSignificant LEC Neurons:")
    print(f"  Mean accuracy: {np.mean(accs_lec):.3f} ± {np.std(accs_lec):.3f}")
    print(f"  N sessions: {len(accs_lec)}")
    
    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic: {u_stat:.2f}")
    print(f"  p-value: {p_value:.3e}")
    
    # Comparison plot
    fig, ax = plt.subplots(figsize=(8, 6))
    combined_df = pd.DataFrame({
        'Accuracy': np.concatenate([accs_all, accs_lec]),
        'Group': ['All Neurons']*len(accs_all) + ['Sig LEC']*len(accs_lec)
    })
    sns.violinplot(data=combined_df, x='Group', y='Accuracy', palette=['lightcoral', 'skyblue'], ax=ax)
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_ylabel('Decoding Accuracy', fontsize=14)
    ax.set_title(f'Decoding Accuracy Comparison (p = {p_value:.3e})', fontsize=15)
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.show()
"""


