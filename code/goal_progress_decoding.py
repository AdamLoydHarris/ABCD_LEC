"""
Goal-progress decoding from Neurons_norm.

For each session Neurons_norm (n_neurons × n_trials × 360) is divided into
12 vectors per trial:  4 states × 3 goal-progress bins (early / middle / late,
30 bins each).

Firing rates are z-scored per neuron across the full concatenation of all
sessions for a given mouse_recday, then split back out before feature
extraction.

A linear classifier is trained to predict the progress
label (early / middle / late) using leave-one-session-out cross-validation.
A null distribution is built by circularly rolling the entire session by a
random amount, preserving autocorrelation structure.

Main entry point
----------------
results = run_goalprogress_decoding(
    data_dic, mouse_recday, valid_sessions,
    neuron_subset=None,
    n_shuffles=1000,
)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

BINS_PER_STATE    = 90
N_STATES          = 4
DEFAULT_N_BINS    = 3   # default number of goal-progress bins


# ─────────────────────────────────────────────────────────────────────────────
def _zscore_concat(nn_list):
    """
    Z-score per neuron across all sessions concatenated.

    Parameters
    ----------
    nn_list : list of np.ndarray, each (n_neurons, n_trials, 360)

    Returns
    -------
    nn_z_list : list of np.ndarray, same shapes, z-scored per neuron
    """
    flat_list = [nn.reshape(nn.shape[0], -1) for nn in nn_list]
    concat    = np.concatenate(flat_list, axis=1)     # (n_neurons, total_T)

    mu      = np.nanmean(concat, axis=1, keepdims=True)
    sd      = np.nanstd(concat,  axis=1, keepdims=True)
    sd_safe = np.where(sd == 0, 1.0, sd)

    nn_z_list = []
    for nn in nn_list:
        flat   = nn.reshape(nn.shape[0], -1)
        flat_z = (flat - mu) / sd_safe
        nn_z_list.append(flat_z.reshape(nn.shape))

    return nn_z_list


# ─────────────────────────────────────────────────────────────────────────────
def _extract_vectors(nn_z, sess_idx, n_progress_bins):
    """
    Extract one averaged vector per (trial, state, progress-bin).

    Parameters
    ----------
    nn_z : np.ndarray, shape (n_neurons, n_trials, 360)
    sess_idx : int
    n_progress_bins : int
        Number of equal bins to divide each state into.

    Returns
    -------
    X          : (n_valid, n_neurons)
    y          : (n_valid,) int, values 0..n_progress_bins-1
    state_ids  : (n_valid,) int, 0–3
    sess_ids   : (n_valid,) int
    trial_ids  : (n_valid,) int
    """
    bins_per_prog = BINS_PER_STATE // n_progress_bins
    n_neurons, n_trials, _ = nn_z.shape
    X_list, y_list, state_list, sid_list, tid_list = [], [], [], [], []

    for trial in range(n_trials):
        for state in range(N_STATES):
            state_offset = state * BINS_PER_STATE
            for prog_i in range(n_progress_bins):
                b_start = state_offset + prog_i * bins_per_prog
                b_end   = b_start + bins_per_prog
                vec = nn_z[:, trial, b_start:b_end].mean(axis=1)   # (n_neurons,)
                if np.any(np.isnan(vec)):
                    continue
                X_list.append(vec)
                y_list.append(prog_i)
                state_list.append(state)
                sid_list.append(sess_idx)
                tid_list.append(trial)

    if not X_list:
        return (np.empty((0, n_neurons)),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int))

    return (np.vstack(X_list),
            np.array(y_list,     dtype=int),
            np.array(state_list, dtype=int),
            np.array(sid_list,   dtype=int),
            np.array(tid_list,   dtype=int))


# ─────────────────────────────────────────────────────────────────────────────
def _roll_session(nn_z, rng):
    """
    Circularly roll the entire session by a random amount.

    The (n_neurons, n_trials, 360) array is flattened to
    (n_neurons, n_trials*360), rolled along the time axis by a random
    shift, then reshaped back.  This preserves autocorrelation across
    the whole session while destroying alignment with goal-progress labels.

    Parameters
    ----------
    nn_z : (n_neurons, n_trials, 360)
    rng  : np.random.Generator

    Returns
    -------
    rolled : (n_neurons, n_trials, 360)
    """
    n_neurons, n_trials, n_bins = nn_z.shape
    total = n_trials * n_bins
    shift = rng.integers(1, total)
    flat  = nn_z.reshape(n_neurons, total)
    return np.roll(flat, shift, axis=1).reshape(n_neurons, n_trials, n_bins)


# ─────────────────────────────────────────────────────────────────────────────
def run_goalprogress_decoding(data_dic, mouse_recday, valid_sessions,
                              neuron_subset=None,
                              n_progress_bins=DEFAULT_N_BINS,
                              n_shuffles=1000):
    """
    Decode goal progress (early / middle / late) from Neurons_norm using
    leave-one-session-out cross-validation.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional
    n_progress_bins : int
        Number of equal bins to divide each 90-bin state into (default 3).
        Must divide 90 evenly.
    n_shuffles : int

    Returns
    -------
    results : dict
        'real_acc'       : float   — LOGO balanced accuracy
        'null_accs'      : (n_shuffles,) float
        'p_value'        : float
        'X'              : (n_samples, n_neurons)
        'y'              : (n_samples,) int, progress bin index
        'sess_id'        : (n_samples,) int
        'state_id'       : (n_samples,) int  0–3
        'n_progress_bins': int
    """
    if BINS_PER_STATE % n_progress_bins != 0:
        raise ValueError(
            f"n_progress_bins={n_progress_bins} must divide "
            f"BINS_PER_STATE={BINS_PER_STATE} evenly."
        )
    print(f"\n{'='*60}")
    print(f"Goal-progress decoding: {mouse_recday}")
    print(f"Sessions: {valid_sessions}")
    print(f"n_progress_bins: {n_progress_bins}  "
          f"({BINS_PER_STATE // n_progress_bins} bins each)")
    print(f"{'='*60}")

    # ── load Neurons_norm ─────────────────────────────────────────────────────
    nn_raw_list = []
    for sidx in valid_sessions:
        sess = data_dic[mouse_recday][sidx]
        nn   = np.asarray(sess['Neurons_norm'], dtype=float)
        if neuron_subset is not None:
            nn = nn[np.asarray(neuron_subset)]
        nn_raw_list.append(nn)
        print(f"  Session {sidx}: {nn.shape[1]} trials, {nn.shape[0]} neurons")

    # ── z-score across full concatenation ────────────────────────────────────
    nn_z_list = _zscore_concat(nn_raw_list)

    # ── extract feature vectors ───────────────────────────────────────────────
    X_all, y_all, state_all, sess_all = [], [], [], []
    for sess_i, nn_z in enumerate(nn_z_list):
        X_s, y_s, st_s, sid_s, _ = _extract_vectors(nn_z, sess_i, n_progress_bins)
        if X_s.shape[0] == 0:
            print(f"  WARNING: no valid vectors for session {valid_sessions[sess_i]}")
            continue
        X_all.append(X_s)
        y_all.append(y_s)
        state_all.append(st_s)
        sess_all.append(sid_s)
        print(f"    sess {valid_sessions[sess_i]}: {X_s.shape[0]} vectors")

    if not X_all:
        raise RuntimeError(f"No valid vectors extracted for {mouse_recday}.")

    X        = np.vstack(X_all)
    y        = np.concatenate(y_all)
    sess_id  = np.concatenate(sess_all)
    state_id = np.concatenate(state_all)

    print(f"\n  Total: {X.shape[0]} vectors, {X.shape[1]} neurons")

    # ── LOGO cross-validation ─────────────────────────────────────────────────
    sessions = np.unique(sess_id)
    if len(sessions) < 2:
        print("WARNING: need ≥2 sessions for LOGO CV — skipping.")
        return None

    def _logo_acc(X_, y_, sid_):  
        preds, trues = [], []
        for held_out in np.unique(sid_):
            tr = sid_ != held_out
            te = sid_ == held_out       
            # clf = LogisticRegression(max_iter=1000, C=1.0)
            clf = SVC(kernel='rbf', C=1.0)
            clf.fit(X_[tr], y_[tr])
            preds.append(clf.predict(X_[te]))
            trues.append(y_[te])
        return balanced_accuracy_score(np.concatenate(trues),
                                       np.concatenate(preds))

    print("\n  Running LOGO cross-validation with SVM...")
    real_acc = _logo_acc(X, y, sess_id)
    print(f"\n  Real LOGO balanced accuracy: {real_acc:.3f}  "
          f"(chance = {1/n_progress_bins:.3f})")

    # ── null: re-extract vectors from circularly rolled data ──────────────────
    rng = np.random.default_rng(42)
    print("\n  Building null distribution by rolling sessions...")
    null_accs = []
    for _ in range(n_shuffles):
        X_null, y_null, sid_null = [], [], []
        for sess_i, nn_z in enumerate(nn_z_list):
            nn_rolled = _roll_session(nn_z, rng)
            X_s, y_s, _, sid_s, _ = _extract_vectors(nn_rolled, sess_i, n_progress_bins)
            if X_s.shape[0] == 0:
                continue
            X_null.append(X_s)
            y_null.append(y_s)
            sid_null.append(sid_s)
        null_accs.append(_logo_acc(np.vstack(X_null),
                                   np.concatenate(y_null),
                                   np.concatenate(sid_null)))
    null_accs = np.array(null_accs)
    p_value = (null_accs >= real_acc).mean()
    print(f"  p = {p_value:.3f}  (n={n_shuffles} shuffles)")
    print(f"{'='*60}\n")

    return {
        'real_acc'    : real_acc,
        'null_accs'   : null_accs,
        'p_value'     : p_value,
        'X'           : X,
        'y'           : y,
        'sess_id'     : sess_id,
        'state_id'       : state_id,
        'mouse_recday'   : mouse_recday,
        'n_progress_bins': n_progress_bins,
    }


def plot_decoding_result(results, bins=40):
    """
    Plot null distribution as a histogram with the observed accuracy marked.

    Parameters
    ----------
    results : dict
        Output of run_goalprogress_decoding.
    bins : int
    """
    real_acc        = results['real_acc']
    null_accs       = results['null_accs']
    p_value         = results['p_value']
    label           = results.get('mouse_recday', '')
    n_progress_bins = results.get('n_progress_bins', DEFAULT_N_BINS)
    chance          = 1 / n_progress_bins

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(null_accs, bins=bins, color='silver', edgecolor='white',
            linewidth=0.4, label='null (session roll)')
    ax.axvline(real_acc, color='crimson', lw=2,
               label=f'observed = {real_acc:.3f}')
    ax.axvline(chance, color='black', lw=1, linestyle='--',
               label=f'chance = {chance:.3f}')

    ax.set_xlabel('Balanced accuracy', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(
        f'Goal-progress decoding — {label}\np = {p_value:.3f}  '
        f'(n={len(null_accs)} shuffles)',
        fontsize=11, fontweight='bold',
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
