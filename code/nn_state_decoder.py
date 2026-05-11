"""
Neural network decoder for task state (A, B, C, D) prediction.
Leave-one-task-out cross validation with unique task structures.

Uses 1 hidden layer (100 units), weak L2 regularization, 10 epochs per fold.
Uses raw firing rates, downsampled from 25ms bins to 500ms bins.

Works with data_dic structure:
    data_dic[mouse_recday][session] = {
        'Neuron_raw': (n_neurons, n_timepoints),
        'Task': array encoding task structure per trial,
        'Trial_times': (n_trials, n_states) in samples,
        ...
    }
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm


class StateDecoder(nn.Module):
    """Simple 1-hidden-layer neural network for state classification."""
    
    def __init__(self, input_dim, hidden_dim=100, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def get_task_structure_from_task_array(Task):
    """
    Convert Task array to string representation of task structure.
    
    Parameters
    ----------
    Task : np.ndarray or similar
        Task structure array (e.g., [0,1,2,3] or similar encoding)
        
    Returns
    -------
    structure : str
        String like 'ABCD', 'ABDC', etc.
    """
    if Task is None:
        return ''
    
    # Convert to string representation
    task_str = str(Task)
    
    # If it's already a string like 'ABCD', return it
    if all(c in 'ABCD' for c in task_str.replace(' ', '').replace('[', '').replace(']', '').replace(',', '')):
        return task_str.replace(' ', '').replace('[', '').replace(']', '').replace(',', '')
    
    # Otherwise return the raw string for comparison
    return task_str


def extract_session_data(data_dic, mouse_recday, session, target_bin_ms=500, sampling_rate=40):
    """
    Extract and prepare data from a single session.
    
    Parameters
    ----------
    data_dic : dict
        Main data dictionary
    mouse_recday : str
        Mouse and recording day key
    session : int or str
        Session key
    target_bin_ms : int
        Target bin size in ms for downsampling
    sampling_rate : int
        Sampling rate in Hz (default 40 for 25ms bins)
        
    Returns
    -------
    X : np.ndarray
        Features (n_samples, n_neurons)
    y : np.ndarray
        Labels (n_samples,)
    task_structure : str
        Task structure string for this session
    """
    session_data = data_dic[mouse_recday][session]
    
    # Get raw neural data
    Neuron_raw = session_data.get('Neuron_raw')
    if Neuron_raw is None:
        return None, None, ''
    
    # Get trial times (in samples, not minutes)
    Trial_times = session_data.get('Trial_times')
    if Trial_times is None or len(Trial_times) == 0:
        return None, None, ''
    
    # Get task structure
    Task = session_data.get('Task')
    task_structure = get_task_structure_from_task_array(Task)
    
    n_neurons, n_timepoints = Neuron_raw.shape
    samples_per_ds_bin = int(sampling_rate * target_bin_ms / 1000)  # 20 for 500ms at 40Hz
    
    EVENT_LABELS = ['A', 'B', 'C', 'D']
    
    X_list = []
    y_list = []
    
    # Trial_times is shape (n_trials, n_states) - contains sample indices for each state onset
    n_trials, n_states = Trial_times.shape
    
    for trial_idx in range(n_trials):
        for state_idx in range(n_states):
            state_start = int(Trial_times[trial_idx, state_idx])
            
            # Determine state end (either next state or end of session)
            if state_idx < n_states - 1:
                state_end = int(Trial_times[trial_idx, state_idx + 1])
            elif trial_idx < n_trials - 1:
                state_end = int(Trial_times[trial_idx + 1, 0])
            else:
                state_end = n_timepoints
            
            # Get state label based on index
            state_label = EVENT_LABELS[state_idx] if state_idx < len(EVENT_LABELS) else ''
            
            if state_label == '' or state_start >= state_end:
                continue
            
            # Extract data for this state and downsample to 500ms bins
            state_data = Neuron_raw[:, state_start:state_end]
            state_length = state_data.shape[1]
            
            # Number of complete 500ms bins
            n_ds_bins = state_length // samples_per_ds_bin
            
            for b in range(n_ds_bins):
                bin_start = b * samples_per_ds_bin
                bin_end = (b + 1) * samples_per_ds_bin
                
                if bin_end <= state_length:
                    bin_data = state_data[:, bin_start:bin_end].mean(axis=1)
                    X_list.append(bin_data)
                    y_list.append(state_label)
    
    if len(X_list) == 0:
        return None, None, task_structure
    
    return np.array(X_list), np.array(y_list), task_structure


def _temporal_features(session_key):
    """
    Return normalised temporal context features from the raw session key.

    Assumes 8 sessions in data_dic (keys 0–7), split into two days of 4,
    with the last session of each day (keys 3 and 7) excluded as repeats,
    leaving 6 valid sessions: [0,1,2, 4,5,6].

    Parameters
    ----------
    session_key : int
        Actual session key in data_dic (0–7).

    Returns
    -------
    feats : np.ndarray, shape (3,)
        [day (0/1),
         session_within_day (0, 0.5, 1.0  for the 3 valid sessions),
         overall_session    (session_key / 7, preserving the day gap)]
    """
    day             = float(session_key // 4)        # 0 or 1
    session_in_day  = float(session_key % 4) / 2.0  # 0→0, 1→0.5, 2→1.0
    overall_session = float(session_key) / 7.0       # preserves gap at day boundary
    return np.array([day, session_in_day, overall_session], dtype=np.float32)


def run_leave_one_task_out_cv(data_dic, mouse_recday, valid_sessions, EVENT_LABELS,
                               hidden_dim=100, weight_decay=1e-4, n_epochs=10,
                               batch_size=64, lr=1e-3, sampling_rate=40,
                               target_bin_ms=500, device='cpu',
                               add_temporal_context=True):
    """
    Run leave-one-task-structure-out cross validation using data_dic.
    
    Parameters
    ----------
    data_dic : dict
        Main data dictionary with structure data_dic[mouse_recday][session]
    mouse_recday : str
        Mouse and recording day key
    valid_sessions : list
        List of valid session keys to use
    EVENT_LABELS : list
        List of event labels ['A', 'B', 'C', 'D']
    add_temporal_context : bool
        If True, append [day, session_within_day, overall_session] (normalised
        to [0,1]) to each sample's feature vector.  Requires exactly 8 sessions.
    hidden_dim : int
        Number of hidden units
    weight_decay : float
        L2 regularization strength
    n_epochs : int
        Number of training epochs per fold
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    sampling_rate : int
        Sampling rate in Hz (default 40 for 25ms bins)
    target_bin_ms : int
        Target bin size in ms after downsampling
    device : str
        'cpu' or 'cuda'
        
    Returns
    -------
    results : dict
        Contains accuracies, predictions, etc.
    """
    print(f"Processing {mouse_recday} with {len(valid_sessions)} valid sessions")
    print(f"Sampling rate: {sampling_rate} Hz, target bin: {target_bin_ms}ms")

    if add_temporal_context and len(valid_sessions) != 6:
        raise ValueError(
            f"add_temporal_context requires exactly 6 valid sessions "
            f"(got {len(valid_sessions)}).  Expected sessions [0,1,2,4,5,6] — "
            f"the repeat at the end of each day (keys 3 and 7) should be excluded. "
            f"Pass add_temporal_context=False to run without temporal inputs."
        )

    # Extract data from all sessions and group by task structure
    session_data = {}  # session -> (X, y, task_structure, temporal_row)
    task_to_sessions = {}  # task_structure -> list of sessions

    for session in valid_sessions:
        X, y, task_structure = extract_session_data(
            data_dic, mouse_recday, session,
            target_bin_ms=target_bin_ms, sampling_rate=sampling_rate
        )

        if X is not None and len(X) > 0:
            if add_temporal_context:
                t_feats = _temporal_features(session)   # uses raw session key
                t_block = np.tile(t_feats, (len(X), 1))
            else:
                t_feats = None
                t_block = None

            session_data[session] = (X, y, task_structure, t_block)

            if task_structure not in task_to_sessions:
                task_to_sessions[task_structure] = []
            task_to_sessions[task_structure].append(session)

            t_str = (f"  day={int(t_feats[0])} "
                     f"sess_in_day={session % 4} "
                     f"overall={session}") if add_temporal_context else ""
            print(f"  Session {session}: {len(X)} samples, task={task_structure}{t_str}")
    
    unique_structures = list(task_to_sessions.keys())
    print(f"\nFound {len(unique_structures)} unique task structures:")
    for struct, sessions in task_to_sessions.items():
        print(f"  '{struct}': {len(sessions)} sessions")
    
    if len(unique_structures) < 2:
        print("Warning: Need at least 2 different task structures for cross-validation")
        return None
    
    # Encode labels
    le = LabelEncoder()
    le.fit(EVENT_LABELS)
    
    # Get number of neurons (should be same across sessions)
    n_neurons   = list(session_data.values())[0][0].shape[1]
    n_temporal  = 3 if add_temporal_context else 0
    input_dim   = n_neurons + n_temporal
    n_classes   = len(EVENT_LABELS)
    print(f"\nModel input dim: {n_neurons} neurons + {n_temporal} temporal = {input_dim}")

    # Fit scaler on ALL sessions concatenated so cross-session firing-rate
    # variation is preserved (not whitened away per fold).
    all_X_concat = np.vstack([sd[0] for sd in session_data.values()])
    scaler = StandardScaler()
    scaler.fit(all_X_concat)
    
    all_preds = []
    all_true = []
    all_probs = []
    fold_accs = []
    fold_structures = []
    fold_train_curves = []   # per-fold list of train balanced acc per epoch
    fold_test_curves  = []   # per-fold list of test  balanced acc per epoch
    
    for fold_idx, held_out_structure in enumerate(unique_structures):
        print(f"\nFold {fold_idx + 1}/{len(unique_structures)}: Holding out '{held_out_structure}'")
        
        # Get train and test sessions
        test_sessions = task_to_sessions[held_out_structure]
        train_sessions = []
        for struct, sessions in task_to_sessions.items():
            if struct != held_out_structure:
                train_sessions.extend(sessions)
        
        if len(train_sessions) == 0 or len(test_sessions) == 0:
            print(f"  Skipping - no train or test sessions")
            continue
        
        # Collect train data
        X_train_list, y_train_list, t_train_list = [], [], []
        for sess in train_sessions:
            X, y, _, t_block = session_data[sess]
            X_train_list.append(X)
            y_train_list.append(y)
            if add_temporal_context:
                t_train_list.append(t_block)
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Collect test data
        X_test_list, y_test_list, t_test_list = [], [], []
        for sess in test_sessions:
            X, y, _, t_block = session_data[sess]
            X_test_list.append(X)
            y_test_list.append(y)
            if add_temporal_context:
                t_test_list.append(t_block)
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)

        print(f"  Train: {len(X_train)} samples from {len(train_sessions)} sessions")
        print(f"  Test: {len(X_test)} samples from {len(test_sessions)} sessions")

        # Encode labels
        y_train_enc = le.transform(y_train)
        y_test_enc  = le.transform(y_test)

        # Apply the global scaler (fit on all sessions) — preserves cross-session variation
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        if add_temporal_context:
            X_train_scaled = np.hstack([X_train_scaled, np.vstack(t_train_list)])
            X_test_scaled  = np.hstack([X_test_scaled,  np.vstack(t_test_list)])

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(device)
        y_train_t = torch.LongTensor(y_train_enc).to(device)
        X_test_t  = torch.FloatTensor(X_test_scaled).to(device)

        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = StateDecoder(input_dim, hidden_dim, n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train — record balanced accuracy on train and test after each epoch
        train_curve, test_curve = [], []
        model.train()
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Evaluate after this epoch
            model.eval()
            with torch.no_grad():
                tr_preds = model(X_train_t).argmax(dim=1).cpu().numpy()
                te_preds = model(X_test_t).argmax(dim=1).cpu().numpy()
            train_curve.append(balanced_accuracy_score(y_train_enc, tr_preds))
            test_curve.append(balanced_accuracy_score(y_test_enc,  te_preds))
            model.train()

        fold_train_curves.append(train_curve)
        fold_test_curves.append(test_curve)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

        # Metrics
        acc = balanced_accuracy_score(y_test_enc, preds)
        fold_accs.append(acc)
        fold_structures.append(held_out_structure)
        
        all_preds.extend(preds)
        all_true.extend(y_test_enc)
        all_probs.extend(probs)
        
        print(f"  Balanced accuracy: {acc:.3f}")
    
    # Overall metrics
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)
    
    overall_acc = balanced_accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)
    
    results = {
        'fold_accuracies': fold_accs,
        'fold_structures': fold_structures,
        'overall_accuracy': overall_acc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_true,
        'probabilities': all_probs,
        'label_encoder': le,
        'unique_structures': unique_structures,
        'task_to_sessions': task_to_sessions,
        'fold_train_curves': fold_train_curves,
        'fold_test_curves':  fold_test_curves,
    }
    
    return results


def plot_learning_curves(results, mouse_recday=''):
    """
    Plot mean ± SEM train and test balanced accuracy over training epochs,
    averaged across all held-out folds.

    Parameters
    ----------
    results : dict
        Output of run_leave_one_task_out_cv.
    mouse_recday : str
    """
    train_curves = np.array(results['fold_train_curves'])  # (n_folds, n_epochs)
    test_curves  = np.array(results['fold_test_curves'])

    n_epochs = train_curves.shape[1]
    epochs   = np.arange(1, n_epochs + 1)

    mean_train = train_curves.mean(axis=0)
    sem_train  = train_curves.std(axis=0) / np.sqrt(len(train_curves))
    mean_test  = test_curves.mean(axis=0)
    sem_test   = test_curves.std(axis=0) / np.sqrt(len(test_curves))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(epochs, mean_train - sem_train, mean_train + sem_train,
                    alpha=0.2, color='steelblue')
    ax.plot(epochs, mean_train, color='steelblue', lw=2, label='Train')
    ax.fill_between(epochs, mean_test - sem_test, mean_test + sem_test,
                    alpha=0.2, color='crimson')
    ax.plot(epochs, mean_test, color='crimson', lw=2, label='Test (held-out)')
    ax.axhline(0.25, color='black', lw=1, ls='--', label='Chance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title(f'Learning curves (mean ± SEM across folds) — {mouse_recday}',
                 fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
    return fig


def plot_results(results, EVENT_LABELS, mouse_recday=''):
    """Plot CV results."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Fold accuracies
    ax = axes[0]
    fold_accs = results['fold_accuracies']
    fold_structs = results['fold_structures']
    x = np.arange(len(fold_accs))
    ax.bar(x, fold_accs, color='steelblue', edgecolor='black')
    ax.axhline(0.25, color='red', ls='--', lw=2, label='Chance (25%)')
    ax.axhline(np.mean(fold_accs), color='green', ls='-', lw=2, 
               label=f'Mean ({np.mean(fold_accs):.1%})')
    ax.set_xticks(x)
    ax.set_xticklabels(fold_structs, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Held-out task structure')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Leave-one-task-out CV')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 2. Confusion matrix
    ax = axes[1]
    cm = results['confusion_matrix']
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(EVENT_LABELS)))
    ax.set_yticks(range(len(EVENT_LABELS)))
    ax.set_xticklabels(EVENT_LABELS)
    ax.set_yticklabels(EVENT_LABELS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (norm)\nOverall acc: {results["overall_accuracy"]:.1%}')
    
    # Add text annotations
    for i in range(len(EVENT_LABELS)):
        for j in range(len(EVENT_LABELS)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center', color=color)
    
    plt.colorbar(im, ax=ax)
    
    # 3. Accuracy distribution
    ax = axes[2]
    ax.hist(fold_accs, bins=min(10, len(fold_accs)), color='steelblue', edgecolor='black')
    ax.axvline(np.mean(fold_accs), color='green', lw=2, label=f'Mean: {np.mean(fold_accs):.1%}')
    ax.axvline(0.25, color='red', ls='--', lw=2, label='Chance')
    ax.set_xlabel('Balanced Accuracy')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of fold accuracies')
    ax.legend()
    
    fig.suptitle(f'NN State Decoder — {mouse_recday}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    if 'fold_train_curves' in results:
        plot_learning_curves(results, mouse_recday=mouse_recday)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE (copy to notebook)
# ─────────────────────────────────────────────────────────────────────────────
"""
# In notebook, after loading data_dic and valid_sessions_dic:

from nn_state_decoder import run_leave_one_task_out_cv, plot_results, plot_learning_curves
import torch

# Run CV for a specific mouse/recording day
# data_dic structure: data_dic[mouse_recday][session] = {'Neuron_raw': ..., 'Task': ..., 'Trial_times': ...}
# valid_sessions_dic: valid_sessions_dic[mouse_recday] = [session1, session2, ...]

results = run_leave_one_task_out_cv(
    data_dic=data_dic,
    mouse_recday=mouse_recday,
    valid_sessions=valid_sessions_dic[mouse_recday],
    EVENT_LABELS=EVENT_LABELS,
    hidden_dim=100,
    weight_decay=1e-4,  # weak L2 regularization
    n_epochs=10,
    batch_size=64,
    lr=1e-3,
    sampling_rate=40,   # 40 Hz = 25ms bins
    target_bin_ms=500,  # Downsample to 500ms bins
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

if results is not None:
    print(f"\\nOverall balanced accuracy: {results['overall_accuracy']:.1%}")
    print(f"Mean fold accuracy: {np.mean(results['fold_accuracies']):.1%} ± {np.std(results['fold_accuracies']):.1%}")
    
    # Plot results
    fig = plot_results(results, EVENT_LABELS, mouse_recday)
"""


if __name__ == "__main__":
    print("NN State Decoder module loaded.")
    print("See docstring at bottom for usage example.")
    print("\nThis module uses RAW firing rates from data_dic[mouse_recday][session]['Neuron_raw']")
    print("Leave-one-task-out CV groups sessions by their Task structure.")
    print("Data is downsampled from 25ms to 500ms bins.")
