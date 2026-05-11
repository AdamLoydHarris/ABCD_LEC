# Population Vector Similarity Analysis
# For each recday and each pair of tasks, calculate cosine similarity of population vectors across states

from scipy.stats import ttest_1samp, mannwhitneyu, zscore
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_population_vector_correlations(data_dic, session_inds_dic, norm_neurons_dic, 
                                             tasks_dic, significant_neurons=None, 
                                             n_shuffles=1000):
    """
    Calculate population vector cosine similarity between pairs of tasks within each recording day.
    
    Parameters:
    -----------
    data_dic : dict
        Dictionary containing neural data for each mouse_recday and session
    session_inds_dic : dict
        Dictionary mapping mouse_recday to valid session indices
    norm_neurons_dic : dict
        Dictionary containing normalized neuron data
    tasks_dic : dict
        Dictionary containing task information for each session
    significant_neurons : dict, optional
        Dictionary mapping mouse_recday to indices of significant neurons
    n_shuffles : int
        Number of shuffles to generate null distribution (default: 1000)
        
    Returns:
    --------
    results_df : pandas DataFrame
        DataFrame with columns: mouse_recday, mouse, date_pair, task_pair, state, similarity
    null_distribution : numpy array
        Array of null similarity values from shuffled data
    """
    
    results_list = []
    num_bins = 90
    num_states = 4
    
    for mouse_recday in data_dic.keys():
        if mouse_recday not in session_inds_dic:
            continue
            
        # Get valid sessions for this recday
        valid_sessions = session_inds_dic[mouse_recday]
        
        if len(valid_sessions) < 2:
            continue
            
        print(f"\nProcessing {mouse_recday}")
        print(f"  Valid sessions: {valid_sessions}")
        
        # Extract mouse ID and date pair
        parts = mouse_recday.split('_')
        mouse_id = parts[0]
        date_pair_str = '_'.join(parts[1:])
        
        # Get task labels for each session
        session_tasks = []
        for session in valid_sessions:
            if mouse_recday in tasks_dic and session in tasks_dic[mouse_recday]:
                task = tasks_dic[mouse_recday][session]
                session_tasks.append((session, task))
            else:
                print(f"  Warning: No task found for session {session}")
        
        # Create pairs of non-repeating tasks
        task_pairs = []
        for i, (sess1, task1) in enumerate(session_tasks):
            for sess2, task2 in session_tasks[i+1:]:
                if not np.array_equal(task1, task2):  # Only non-repeating tasks
                    task_pairs.append(((sess1, task1), (sess2, task2)))
        
        if len(task_pairs) == 0:
            print(f"  No valid task pairs found")
            continue
            
        print(f"  Task pairs to analyze: {len(task_pairs)}")
        
        # For each task pair, calculate correlations by state
        for (sess1, task1), (sess2, task2) in task_pairs:
            task1_str = str(task1) if isinstance(task1, str) else f"task_{sess1}"
            task2_str = str(task2) if isinstance(task2, str) else f"task_{sess2}"
            print(f"    Comparing {task1_str} (session {sess1}) vs {task2_str} (session {sess2})")
            
            # Get normalized neurons for both sessions
            if mouse_recday not in norm_neurons_dic:
                print(f"      Warning: No normalized neurons found")
                continue
            
            # norm_neurons_dic[mouse_recday] is a list indexed by position in valid_sessions
            # Need to map session numbers to list positions
            try:
                if isinstance(norm_neurons_dic[mouse_recday], dict):
                    # If it's a dict, use session numbers as keys
                    if isinstance(norm_neurons_dic[mouse_recday][sess1], dict):
                        norm_data1 = norm_neurons_dic[mouse_recday][sess1]['Neurons_norm']
                        norm_data2 = norm_neurons_dic[mouse_recday][sess2]['Neurons_norm']
                    else:
                        # Direct array access with session as key
                        norm_data1 = norm_neurons_dic[mouse_recday][sess1]
                        norm_data2 = norm_neurons_dic[mouse_recday][sess2]
                else:
                    # It's a list indexed by position, not session number
                    # Find the position of sess1 and sess2 in the valid_sessions list
                    sess1_idx = valid_sessions.index(sess1)
                    sess2_idx = valid_sessions.index(sess2)
                    norm_data1 = norm_neurons_dic[mouse_recday][sess1_idx]
                    norm_data2 = norm_neurons_dic[mouse_recday][sess2_idx]
            except (IndexError, KeyError, ValueError) as e:
                print(f"      Warning: Could not access normalized data for sessions {sess1}, {sess2}: {e}")
                continue
            
            # Filter for significant neurons if provided
            if significant_neurons is not None and mouse_recday in significant_neurons:
                sig_neuron_inds = significant_neurons[mouse_recday]
                norm_data1 = norm_data1[sig_neuron_inds]
                norm_data2 = norm_data2[sig_neuron_inds]
                print(f"      Using {len(sig_neuron_inds)} significant neurons")
            else:
                print(f"      Using all {norm_data1.shape[0]} neurons")
            
            num_neurons = norm_data1.shape[0]
            
            # Ensure both have the same number of neurons
            if norm_data1.shape[0] != norm_data2.shape[0]:
                print(f"      Warning: Neuron count mismatch, skipping")
                continue
            
            # For each state, calculate population vector cosine similarity
            state_correlations = []
            for state_idx in range(num_states):
                # Extract bins for this state
                state_start = state_idx * num_bins
                state_end = (state_idx + 1) * num_bins
                
                # Get state data for both sessions
                # Shape: (num_neurons, num_trials, num_bins_in_state)
                state_data1 = norm_data1[:, :, state_start:state_end]
                state_data2 = norm_data2[:, :, state_start:state_end]
                
                # Average across trials to get one population vector per task per state
                # Shape: (num_neurons, num_bins_in_state)
                pop_vec_task1 = np.nanmean(state_data1, axis=1)
                pop_vec_task2 = np.nanmean(state_data2, axis=1)
                
                # Flatten to create population vectors
                # Shape: (num_neurons * num_bins,)
                pop_vec1_flat = pop_vec_task1.flatten()
                pop_vec2_flat = pop_vec_task2.flatten()
                
                # Remove NaN values
                valid_mask = ~(np.isnan(pop_vec1_flat) | np.isnan(pop_vec2_flat))
                pop_vec1_clean = pop_vec1_flat[valid_mask]
                pop_vec2_clean = pop_vec2_flat[valid_mask]
                
                if len(pop_vec1_clean) > 1:
                    # Z-score each population vector
                    pop_vec1_z = zscore(pop_vec1_clean, nan_policy='omit')
                    pop_vec2_z = zscore(pop_vec2_clean, nan_policy='omit')
                    
                    # Calculate cosine similarity between the two task-averaged population vectors
                    # Reshape for sklearn: (1, n_features) for each vector
                    similarity = cosine_similarity(
                        pop_vec1_z.reshape(1, -1), 
                        pop_vec2_z.reshape(1, -1)
                    )[0, 0]
                    
                    state_correlations.append(similarity)
                    
                    # Store result
                    state_name = ['A', 'B', 'C', 'D'][state_idx]
                    results_list.append({
                        'mouse_recday': mouse_recday,
                        'mouse': mouse_id,
                        'date_pair': date_pair_str,
                        'session1': sess1,
                        'session2': sess2,
                        'task1': task1_str,
                        'task2': task2_str,
                        'task_pair': f"{task1_str}_vs_{task2_str}",
                        'state': state_name,
                        'similarity': similarity,
                        'num_neurons': num_neurons,
                        'num_trials_task1': state_data1.shape[1],
                        'num_trials_task2': state_data2.shape[1]
                    })
                    
                    print(f"      State {state_name}: similarity={similarity:.3f}")
                else:
                    print(f"      Warning: Not enough valid data for state {state_idx}")
            
            if len(state_correlations) > 0:
                mean_sim = np.mean(state_correlations)
                print(f"      Mean similarity across states: {mean_sim:.3f}")
    
    # Generate null distribution by shuffling neuron identities
    print(f"\n{'='*80}")
    print(f"Generating null distribution with {n_shuffles} shuffles...")
    print(f"{'='*80}")
    
    null_similarities = []
    shuffle_count = 0
    
    for mouse_recday in data_dic.keys():
        if mouse_recday not in session_inds_dic:
            continue
            
        valid_sessions = session_inds_dic[mouse_recday]
        if len(valid_sessions) < 2:
            continue
        
        # Get task pairs for this recday
        session_tasks = []
        for session in valid_sessions:
            if mouse_recday in tasks_dic and session in tasks_dic[mouse_recday]:
                task = tasks_dic[mouse_recday][session]
                if not any(np.array_equal(task, t) for _, t in session_tasks):
                    session_tasks.append((session, task))
        
        if len(session_tasks) < 2:
            continue
        
        # For each task pair
        for (sess1, task1), (sess2, task2) in combinations(session_tasks, 2):
            if np.array_equal(task1, task2):
                continue
            
            # Get normalized neurons
            try:
                if isinstance(norm_neurons_dic[mouse_recday], dict):
                    if isinstance(norm_neurons_dic[mouse_recday][sess1], dict):
                        norm_data1 = norm_neurons_dic[mouse_recday][sess1]['Neurons_norm']
                        norm_data2 = norm_neurons_dic[mouse_recday][sess2]['Neurons_norm']
                    else:
                        norm_data1 = norm_neurons_dic[mouse_recday][sess1]
                        norm_data2 = norm_neurons_dic[mouse_recday][sess2]
                else:
                    sess1_idx = valid_sessions.index(sess1)
                    sess2_idx = valid_sessions.index(sess2)
                    norm_data1 = norm_neurons_dic[mouse_recday][sess1_idx]
                    norm_data2 = norm_neurons_dic[mouse_recday][sess2_idx]
            except (IndexError, KeyError, ValueError):
                continue
            
            # Filter for significant neurons if provided
            if significant_neurons is not None and mouse_recday in significant_neurons:
                sig_neuron_inds = significant_neurons[mouse_recday]
                norm_data1 = norm_data1[sig_neuron_inds]
                norm_data2 = norm_data2[sig_neuron_inds]
            
            # Do a few shuffles for this task pair
            shuffles_per_pair = min(10, n_shuffles // max(1, len(results_list) // 4))
            
            for _ in range(shuffles_per_pair):
                if shuffle_count >= n_shuffles:
                    break
                
                # Randomly shuffle neuron identities in one of the tasks
                neuron_perm = np.random.permutation(norm_data2.shape[0])
                norm_data2_shuffled = norm_data2[neuron_perm]
                
                # Calculate similarity for a random state
                state_idx = np.random.randint(0, num_states)
                state_start = state_idx * num_bins
                state_end = (state_idx + 1) * num_bins
                
                state_data1 = norm_data1[:, :, state_start:state_end]
                state_data2_shuffled = norm_data2_shuffled[:, :, state_start:state_end]
                
                pop_vec1 = np.nanmean(state_data1, axis=1).flatten()
                pop_vec2 = np.nanmean(state_data2_shuffled, axis=1).flatten()
                
                valid_mask = ~(np.isnan(pop_vec1) | np.isnan(pop_vec2))
                pop_vec1_clean = pop_vec1[valid_mask]
                pop_vec2_clean = pop_vec2[valid_mask]
                
                if len(pop_vec1_clean) > 1:
                    pop_vec1_z = zscore(pop_vec1_clean, nan_policy='omit')
                    pop_vec2_z = zscore(pop_vec2_clean, nan_policy='omit')
                    
                    null_sim = cosine_similarity(
                        pop_vec1_z.reshape(1, -1), 
                        pop_vec2_z.reshape(1, -1)
                    )[0, 0]
                    
                    null_similarities.append(null_sim)
                    shuffle_count += 1
            
            if shuffle_count >= n_shuffles:
                break
        
        if shuffle_count >= n_shuffles:
            break
    
    null_distribution = np.array(null_similarities)
    print(f"Generated {len(null_distribution)} null similarity values")
    print(f"Null distribution: mean={np.mean(null_distribution):.3f}, std={np.std(null_distribution):.3f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    return results_df, null_distribution


def calculate_across_state_correlations(data_dic, session_inds_dic, norm_neurons_dic, 
                                        tasks_dic, significant_neurons=None):
    """
    Calculate population vector cosine similarity ACROSS different states (control analysis).
    For each task pair, compare state A from task 1 with states B, C, D from task 2, etc.
    
    Parameters:
    -----------
    data_dic : dict
        Dictionary containing neural data for each mouse_recday and session
    session_inds_dic : dict
        Dictionary mapping mouse_recday to valid session indices
    norm_neurons_dic : dict
        Dictionary containing normalized neuron data
    tasks_dic : dict
        Dictionary containing task information for each session
    significant_neurons : dict, optional
        Dictionary mapping mouse_recday to indices of significant neurons
        
    Returns:
    --------
    results_df : pandas DataFrame
        DataFrame with columns: mouse_recday, mouse, date_pair, task_pair, state1, state2, similarity
    """
    
    results_list = []
    num_bins = 90
    num_states = 4
    state_names = ['A', 'B', 'C', 'D']
    
    for mouse_recday in data_dic.keys():
        if mouse_recday not in session_inds_dic:
            continue
            
        valid_sessions = session_inds_dic[mouse_recday]
        
        if len(valid_sessions) < 2:
            continue
            
        print(f"\nProcessing {mouse_recday}")
        print(f"  Valid sessions: {valid_sessions}")
        
        parts = mouse_recday.split('_')
        mouse_id = parts[0]
        date_pair_str = '_'.join(parts[1:])
        
        # Get task labels for each session
        session_tasks = []
        for session in valid_sessions:
            if mouse_recday in tasks_dic and session in tasks_dic[mouse_recday]:
                task = tasks_dic[mouse_recday][session]
                session_tasks.append((session, task))
        
        # Create pairs of non-repeating tasks
        task_pairs = []
        for i, (sess1, task1) in enumerate(session_tasks):
            for sess2, task2 in session_tasks[i+1:]:
                if not np.array_equal(task1, task2):
                    task_pairs.append(((sess1, task1), (sess2, task2)))
        
        if len(task_pairs) == 0:
            print(f"  No valid task pairs found")
            continue
            
        print(f"  Task pairs to analyze: {len(task_pairs)}")
        
        # For each task pair, calculate across-state similarities
        for (sess1, task1), (sess2, task2) in task_pairs:
            task1_str = str(task1) if isinstance(task1, str) else f"task_{sess1}"
            task2_str = str(task2) if isinstance(task2, str) else f"task_{sess2}"
            print(f"    Comparing {task1_str} (session {sess1}) vs {task2_str} (session {sess2})")
            
            # Get normalized neurons
            try:
                if isinstance(norm_neurons_dic[mouse_recday], dict):
                    if isinstance(norm_neurons_dic[mouse_recday][sess1], dict):
                        norm_data1 = norm_neurons_dic[mouse_recday][sess1]['Neurons_norm']
                        norm_data2 = norm_neurons_dic[mouse_recday][sess2]['Neurons_norm']
                    else:
                        norm_data1 = norm_neurons_dic[mouse_recday][sess1]
                        norm_data2 = norm_neurons_dic[mouse_recday][sess2]
                else:
                    sess1_idx = valid_sessions.index(sess1)
                    sess2_idx = valid_sessions.index(sess2)
                    norm_data1 = norm_neurons_dic[mouse_recday][sess1_idx]
                    norm_data2 = norm_neurons_dic[mouse_recday][sess2_idx]
            except (IndexError, KeyError, ValueError) as e:
                print(f"      Warning: Could not access normalized data: {e}")
                continue
            
            # Filter for significant neurons if provided
            if significant_neurons is not None and mouse_recday in significant_neurons:
                sig_neuron_inds = significant_neurons[mouse_recday]
                norm_data1 = norm_data1[sig_neuron_inds]
                norm_data2 = norm_data2[sig_neuron_inds]
                print(f"      Using {len(sig_neuron_inds)} significant neurons")
            else:
                print(f"      Using all {norm_data1.shape[0]} neurons")
            
            num_neurons = norm_data1.shape[0]
            
            if norm_data1.shape[0] != norm_data2.shape[0]:
                print(f"      Warning: Neuron count mismatch, skipping")
                continue
            
            # For each pair of different states
            for state1_idx in range(num_states):
                for state2_idx in range(num_states):
                    if state1_idx == state2_idx:
                        continue  # Skip same-state comparisons
                    
                    state1_start = state1_idx * num_bins
                    state1_end = (state1_idx + 1) * num_bins
                    state2_start = state2_idx * num_bins
                    state2_end = (state2_idx + 1) * num_bins
                    
                    # Get state data
                    state_data1 = norm_data1[:, :, state1_start:state1_end]
                    state_data2 = norm_data2[:, :, state2_start:state2_end]
                    
                    # Average across trials
                    pop_vec_task1 = np.nanmean(state_data1, axis=1)
                    pop_vec_task2 = np.nanmean(state_data2, axis=1)
                    
                    # Flatten
                    pop_vec1_flat = pop_vec_task1.flatten()
                    pop_vec2_flat = pop_vec_task2.flatten()
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(pop_vec1_flat) | np.isnan(pop_vec2_flat))
                    pop_vec1_clean = pop_vec1_flat[valid_mask]
                    pop_vec2_clean = pop_vec2_flat[valid_mask]
                    
                    if len(pop_vec1_clean) > 1:
                        # Z-score
                        pop_vec1_z = zscore(pop_vec1_clean, nan_policy='omit')
                        pop_vec2_z = zscore(pop_vec2_clean, nan_policy='omit')
                        
                        # Calculate cosine similarity
                        similarity = cosine_similarity(
                            pop_vec1_z.reshape(1, -1), 
                            pop_vec2_z.reshape(1, -1)
                        )[0, 0]
                        
                        # Store result
                        results_list.append({
                            'mouse_recday': mouse_recday,
                            'mouse': mouse_id,
                            'date_pair': date_pair_str,
                            'session1': sess1,
                            'session2': sess2,
                            'task1': task1_str,
                            'task2': task2_str,
                            'task_pair': f"{task1_str}_vs_{task2_str}",
                            'state1': state_names[state1_idx],
                            'state2': state_names[state2_idx],
                            'state_pair': f"{state_names[state1_idx]}_vs_{state_names[state2_idx]}",
                            'similarity': similarity,
                            'num_neurons': num_neurons
                        })
    
    results_df = pd.DataFrame(results_list)
    return results_df


def plot_and_analyze_correlations(results_df, null_distribution=None, title_suffix=""):
    """
    Plot and analyze population vector cosine similarities.
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Results from calculate_population_vector_correlations
    null_distribution : numpy array, optional
        Null distribution from shuffled data
    title_suffix : str
        Suffix to add to plot titles (e.g., "All Neurons" or "Significant LEC Neurons")
    """
    if len(results_df) == 0:
        print(f"No results to plot for {title_suffix}")
        return
    
    # 1. Aggregate by state
    print(f"\n{'='*80}")
    print(f"ANALYSIS BY STATE - {title_suffix}")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    states = ['A', 'B', 'C', 'D']
    state_stats = {}
    
    for idx, state in enumerate(states):
        state_data = results_df[results_df['state'] == state]['similarity'].values
        
        if len(state_data) > 0:
            # Statistical test against null distribution if provided
            if null_distribution is not None and len(null_distribution) > 0:
                null_mean = np.mean(null_distribution)
                null_std = np.std(null_distribution)
                # Test against null mean
                t_stat, p_value = ttest_1samp(state_data, null_mean)
                print(f"\nState {state}:")
                print(f"  N = {len(state_data)} task pairs")
                print(f"  Observed mean = {np.mean(state_data):.3f} ± {np.std(state_data):.3f}")
                print(f"  Null mean = {null_mean:.3f} ± {null_std:.3f}")
                print(f"  t-statistic (vs null) = {t_stat:.3f}")
                print(f"  p-value (vs null) = {p_value:.3e}")
                
                # Calculate effect size (Cohen's d)
                cohens_d = (np.mean(state_data) - null_mean) / null_std
                print(f"  Effect size (Cohen's d) = {cohens_d:.3f}")
            else:
                # Test against zero
                t_stat, p_value = ttest_1samp(state_data, 0)
                print(f"\nState {state}:")
                print(f"  N = {len(state_data)} task pairs")
                print(f"  Mean similarity = {np.mean(state_data):.3f} ± {np.std(state_data):.3f}")
                print(f"  t-statistic = {t_stat:.3f}")
                print(f"  p-value = {p_value:.3e}")
            
            mean_sim = np.mean(state_data)
            std_sim = np.std(state_data)
            
            state_stats[state] = {
                'mean': mean_sim,
                'std': std_sim,
                't_stat': t_stat,
                'p_value': p_value,
                'n': len(state_data)
            }
            
            # Plot histogram
            ax = axes[idx]
            sns.histplot(state_data, kde=True, color="skyblue", bins=30, ax=ax, alpha=0.7)
            
            # Add null distribution if provided
            if null_distribution is not None and len(null_distribution) > 0:
                sns.histplot(null_distribution, kde=True, color="gray", bins=30, 
                           ax=ax, alpha=0.3, label='Null')
                ax.axvline(x=null_mean, color='red', linestyle='--', linewidth=2, 
                          label=f'Null = {null_mean:.3f}')
            else:
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
            
            ax.axvline(x=mean_sim, color='black', linestyle='--', linewidth=2, 
                      label=f'Observed = {mean_sim:.3f}')
            ax.set_xlabel('Cosine Similarity', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'State {state} (p = {p_value:.3e})', fontsize=13)
            ax.legend()
            sns.despine(ax=ax)
    
    plt.suptitle(f'Population Vector Cosine Similarities by State - {title_suffix}', 
                 fontsize=15, y=1.00)
    plt.tight_layout()
    plt.show()
    
    # 2. Aggregate across all states
    print(f"\n{'='*80}")
    print(f"ANALYSIS ACROSS ALL STATES - {title_suffix}")
    print(f"{'='*80}")
    
    all_sims = results_df['similarity'].values
    
    if null_distribution is not None and len(null_distribution) > 0:
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution)
        t_stat, p_value = ttest_1samp(all_sims, null_mean)
        
        print(f"\nAll States Combined:")
        print(f"  N = {len(all_sims)} similarities")
        print(f"  Observed mean = {np.mean(all_sims):.3f} ± {np.std(all_sims):.3f}")
        print(f"  Null mean = {null_mean:.3f} ± {null_std:.3f}")
        print(f"  t-statistic (vs null) = {t_stat:.3f}")
        print(f"  p-value (vs null) = {p_value:.3e}")
        
        cohens_d = (np.mean(all_sims) - null_mean) / null_std
        print(f"  Effect size (Cohen's d) = {cohens_d:.3f}")
    else:
        t_stat, p_value = ttest_1samp(all_sims, 0)
        print(f"\nAll States Combined:")
        print(f"  N = {len(all_sims)} similarities")
        print(f"  Mean similarity = {np.mean(all_sims):.3f} ± {np.std(all_sims):.3f}")
        print(f"  t-statistic = {t_stat:.3f}")
        print(f"  p-value = {p_value:.3e}")
    
    mean_sim = np.mean(all_sims)
    std_sim = np.std(all_sims)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(all_sims, kde=True, color="lightblue", bins=40, alpha=0.7, label='Observed')
    
    if null_distribution is not None and len(null_distribution) > 0:
        sns.histplot(null_distribution, kde=True, color="gray", bins=40, 
                    alpha=0.3, label='Null')
        ax.axvline(x=null_mean, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Null = {null_mean:.3f}')
    else:
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero')
    
    ax.axvline(x=mean_sim, color='black', linestyle='--', linewidth=2.5, 
              label=f'Observed = {mean_sim:.3f}')
    ax.set_xlabel('Population Vector Cosine Similarity', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(f'All States Combined - {title_suffix} (p = {p_value:.3e})', fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    sns.despine()
    plt.tight_layout()
    plt.show()
    
    # 3. Aggregate by mouse
    print(f"\n{'='*80}")
    print(f"ANALYSIS BY MOUSE - {title_suffix}")
    print(f"{'='*80}")
    
    mice = results_df['mouse'].unique()
    mouse_stats = {}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mouse in sorted(mice):
        mouse_data = results_df[results_df['mouse'] == mouse]['similarity'].values
        
        if len(mouse_data) > 0:
            mean_sim = np.mean(mouse_data)
            std_sim = np.std(mouse_data)
            t_stat, p_value = ttest_1samp(mouse_data, 0)
            
            mouse_stats[mouse] = {
                'mean': mean_sim,
                'std': std_sim,
                't_stat': t_stat,
                'p_value': p_value,
                'n': len(mouse_data)
            }
            
            print(f"\nMouse {mouse}:")
            print(f"  N = {len(mouse_data)}")
            print(f"  Mean similarity = {mean_sim:.3f} ± {std_sim:.3f}")
            print(f"  t-statistic = {t_stat:.3f}")
            print(f"  p-value = {p_value:.3e}")
    
    # Violin plot by state
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=results_df, x='state', y='similarity', palette='Set2', ax=ax)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_ylabel('Population Vector Cosine Similarity', fontsize=14)
    ax.set_xlabel('State', fontsize=14)
    ax.set_title(f'Population Vector Cosine Similarities by State - {title_suffix}', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.legend(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.show()
    
    return {
        'state_stats': state_stats,
        'mouse_stats': mouse_stats,
        'overall': {
            'mean': mean_sim,
            'std': std_sim,
            't_stat': t_stat,
            'p_value': p_value,
            'n': len(all_sims)
        }
    }


# Example usage (copy each section into separate cells):

# CELL 1: Run analysis for all neurons
print("=" * 80)
print("ANALYZING ALL NEURONS")
print("=" * 80)
correlation_results_all, null_dist_all = calculate_population_vector_correlations(
    data_dic, session_inds_dic, norm_neurons_dic, tasks_dic, 
    significant_neurons=None
)
print(f"\n\nTotal results: {len(correlation_results_all)}")
if len(correlation_results_all) > 0:
    print(correlation_results_all.head(10))

# CELL 2: Run analysis for significant LEC neurons
print("\n" + "=" * 80)
print("ANALYZING SIGNIFICANT LEC NEURONS ONLY")
print("=" * 80)
correlation_results_lec, null_dist_lec = calculate_population_vector_correlations(
    data_dic, session_inds_dic, norm_neurons_dic, tasks_dic, 
    significant_neurons=sig_lec_neurons_dic
)
print(f"\n\nTotal results: {len(correlation_results_lec)}")
if len(correlation_results_lec) > 0:
    print(correlation_results_lec.head(10))

# CELL 3: Plot and analyze results for all neurons
if len(correlation_results_all) > 0:
    stats_all = plot_and_analyze_correlations(correlation_results_all, null_dist_all, "All Neurons")

# CELL 4: Plot and analyze results for significant LEC neurons
if len(correlation_results_lec) > 0:
    stats_lec = plot_and_analyze_correlations(correlation_results_lec, null_dist_lec, "Significant LEC Neurons")

# CELL 5: Comparison between All Neurons and Significant LEC Neurons
print("\n" + "=" * 80)
print("COMPARISON: ALL NEURONS vs SIGNIFICANT LEC NEURONS")
print("=" * 80)

if len(correlation_results_all) > 0 and len(correlation_results_lec) > 0:
    # Compare overall similarities
    all_sims_all = correlation_results_all['similarity'].values
    all_sims_lec = correlation_results_lec['similarity'].values
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(all_sims_all, all_sims_lec, alternative='two-sided')
    
    print(f"\nAll Neurons:")
    print(f"  N = {len(all_sims_all)}")
    print(f"  Mean = {np.mean(all_sims_all):.3f} ± {np.std(all_sims_all):.3f}")
    
    print(f"\nSignificant LEC Neurons:")
    print(f"  N = {len(all_sims_lec)}")
    print(f"  Mean = {np.mean(all_sims_lec):.3f} ± {np.std(all_sims_lec):.3f}")
    
    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic = {u_stat:.2f}")
    print(f"  p-value = {p_value:.3e}")
    
    # Side-by-side comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram comparison
    ax = axes[0]
    sns.histplot(all_sims_all, kde=True, color="lightcoral", bins=40, 
                 alpha=0.5, label='All Neurons', ax=ax)
    sns.histplot(all_sims_lec, kde=True, color="skyblue", bins=40, 
                 alpha=0.5, label='Significant LEC', ax=ax)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(x=np.mean(all_sims_all), color='darkred', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'Mean All = {np.mean(all_sims_all):.3f}')
    ax.axvline(x=np.mean(all_sims_lec), color='darkblue', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'Mean LEC = {np.mean(all_sims_lec):.3f}')
    ax.set_xlabel('Population Vector Cosine Similarity', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(f'Comparison (Mann-Whitney p = {p_value:.3e})', fontsize=15)
    ax.legend(fontsize=11)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    sns.despine(ax=ax)
    
    # Violin plot comparison
    ax = axes[1]
    combined_df = pd.DataFrame({
        'Similarity': np.concatenate([all_sims_all, all_sims_lec]),
        'Group': ['All Neurons'] * len(all_sims_all) + ['Significant LEC'] * len(all_sims_lec)
    })
    sns.violinplot(data=combined_df, x='Group', y='Similarity', palette=['lightcoral', 'skyblue'], ax=ax)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_ylabel('Population Vector Cosine Similarity', fontsize=14)
    ax.set_xlabel('', fontsize=14)
    ax.set_title('Distribution Comparison', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.legend(fontsize=12)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.show()

# CELL 6: Calculate across-state similarities (control)
print("\n" + "=" * 80)
print("CALCULATING ACROSS-STATE SIMILARITIES (CONTROL)")
print("=" * 80)
across_state_results = calculate_across_state_correlations(
    data_dic, session_inds_dic, norm_neurons_dic, tasks_dic, 
    significant_neurons=None
)
print(f"\nTotal across-state comparisons: {len(across_state_results)}")
if len(across_state_results) > 0:
    print(across_state_results.head(10))

# CELL 7: Compare within-state vs across-state similarities
print("\n" + "=" * 80)
print("COMPARISON: WITHIN-STATE vs ACROSS-STATE SIMILARITIES")
print("=" * 80)

if len(correlation_results_all) > 0 and len(across_state_results) > 0:
    within_state_sims = correlation_results_all['similarity'].values
    across_state_sims = across_state_results['similarity'].values
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(within_state_sims, across_state_sims, alternative='two-sided')
    
    print(f"\nWithin-State (same state across tasks):")
    print(f"  N = {len(within_state_sims)}")
    print(f"  Mean = {np.mean(within_state_sims):.3f} ± {np.std(within_state_sims):.3f}")
    
    print(f"\nAcross-State (different states across tasks):")
    print(f"  N = {len(across_state_sims)}")
    print(f"  Mean = {np.mean(across_state_sims):.3f} ± {np.std(across_state_sims):.3f}")
    
    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic = {u_stat:.2f}")
    print(f"  p-value = {p_value:.3e}")
    
    # Effect size
    cohens_d = (np.mean(within_state_sims) - np.mean(across_state_sims)) / np.sqrt(
        (np.std(within_state_sims)**2 + np.std(across_state_sims)**2) / 2
    )
    print(f"  Effect size (Cohen's d) = {cohens_d:.3f}")
    
    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram comparison
    ax = axes[0]
    sns.histplot(within_state_sims, kde=True, color="steelblue", bins=40, 
                 alpha=0.6, label='Within-State', ax=ax)
    sns.histplot(across_state_sims, kde=True, color="orange", bins=40, 
                 alpha=0.6, label='Across-State', ax=ax)
    ax.axvline(x=np.mean(within_state_sims), color='darkblue', linestyle='--', 
              linewidth=2.5, label=f'Within Mean = {np.mean(within_state_sims):.3f}')
    ax.axvline(x=np.mean(across_state_sims), color='darkorange', linestyle='--', 
              linewidth=2.5, label=f'Across Mean = {np.mean(across_state_sims):.3f}')
    ax.set_xlabel('Cosine Similarity', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(f'Within-State vs Across-State (p = {p_value:.3e})', fontsize=15)
    ax.legend(fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=12)
    sns.despine(ax=ax)
    
    # Violin plot comparison
    ax = axes[1]
    combined_df = pd.DataFrame({
        'Similarity': np.concatenate([within_state_sims, across_state_sims]),
        'Comparison': ['Within-State'] * len(within_state_sims) + ['Across-State'] * len(across_state_sims)
    })
    sns.violinplot(data=combined_df, x='Comparison', y='Similarity', 
                   palette=['steelblue', 'orange'], ax=ax)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_xlabel('', fontsize=14)
    ax.set_title(f'Distribution Comparison (d = {cohens_d:.3f})', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.show()
