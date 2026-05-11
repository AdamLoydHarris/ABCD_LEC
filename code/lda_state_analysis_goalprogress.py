"""
LDA state separation split by goal-progress (early / middle / late).

Each state window is divided into three equal thirds, yielding 12 vectors
per trial (4 states × 3 progress bins) instead of 4.  LDA is run
independently for each progress bin so that the classifier only ever sees
data from the same third of a state transition.

Normalization: identical to lda_state_analysis_trialbins.py — z-score per
neuron across the fully concatenated (all sessions) raw recording, then
extract per-(trial, state, progress-bin) averages.

Main entry point
----------------
results = run_goalprogress_lda_analysis(
    data_dic, mouse_recday, valid_sessions,
    neuron_subset=sig_lec_neurons_dic.get(mouse_recday),
)

Returns a dict keyed by progress-bin label ('early', 'middle', 'late'),
each value being the same result dict as run_trialbins_lda_analysis.

Reuses zscore_concat_sessions and apply_pca_trialbins from
lda_state_analysis_trialbins.py, and run_lda_state_separation /
run_logo_state_cv from lda_state_separation.py.
"""

import numpy as np
from sklearn.decomposition import PCA
from lda_state_separation import run_lda_state_separation, run_logo_state_cv
from lda_state_analysis_trialbins import (
    zscore_concat_sessions,
    apply_pca_trialbins,
    EVENT_LABELS,
    EVENT_COLOURS,
    PCA_VARIANCE_THRESH,
    MIN_STATE_SAMPLES,
)
import matplotlib.pyplot as plt

PROGRESS_BINS   = ['early', 'middle', 'late']   # thirds of each state window
N_PROGRESS_BINS = 3


# ─────────────────────────────────────────────────────────────────────────────
def extract_state_goalprogress_vectors(concat_z, session_offset, session_length,
                                       trial_times, sess_index,
                                       min_samples=MIN_STATE_SAMPLES):
    """
    Extract one averaged population vector per (trial, state, progress-bin).

    Each state window [t_start, t_end) is split into N_PROGRESS_BINS equal
    thirds.  Fractional boundaries are rounded; windows shorter than
    min_samples after splitting are skipped.

    Parameters
    ----------
    concat_z : np.ndarray
        Shape (n_neurons, total_T).  Full z-scored concatenated array.
    session_offset : int
        Sample index in concat_z where this session starts.
    session_length : int
        Number of samples in this session.
    trial_times : np.ndarray
        Shape (n_trials, 5).  Session-local sample indices.
        Columns: [A_start, B_start, C_start, D_start, trial_end].
    sess_index : int
        Integer session index written into sess_id output.
    min_samples : int
        Minimum per-bin window length; shorter bins are skipped.

    Returns
    -------
    X_sess : np.ndarray
        Shape (n_valid, n_neurons).
    y_state_sess : np.ndarray of str
        Shape (n_valid,).  Values in EVENT_LABELS.
    progress_sess : np.ndarray of str
        Shape (n_valid,).  Values in PROGRESS_BINS.
    sess_id_sess : np.ndarray of int
    trial_id_sess : np.ndarray of int
    skipped : list of tuple
        (trial_idx, state_label, bin_label, reason) for every skipped window.
    """
    n_neurons = concat_z.shape[0]
    n_trials  = trial_times.shape[0]

    X_list, y_list, prog_list, sid_list, tid_list = [], [], [], [], []
    skipped = []

    for trial_idx in range(n_trials):
        for state_i, label in enumerate(EVENT_LABELS):
            t_start_local = int(trial_times[trial_idx, state_i])
            t_end_local   = int(trial_times[trial_idx, state_i + 1])

            # Guard: within session bounds
            if t_start_local < 0 or t_end_local > session_length:
                for bin_label in PROGRESS_BINS:
                    skipped.append((trial_idx, label, bin_label,
                        f"out of bounds [{t_start_local}, {t_end_local}) "
                        f"for session length {session_length}"))
                continue

            # Guard: positive duration
            if t_end_local <= t_start_local:
                for bin_label in PROGRESS_BINS:
                    skipped.append((trial_idx, label, bin_label,
                        f"zero/negative duration: "
                        f"start={t_start_local}, end={t_end_local}"))
                continue

            n_samp = t_end_local - t_start_local

            # Compute bin boundaries (rounded integer indices)
            edges = np.round(
                np.linspace(t_start_local, t_end_local, N_PROGRESS_BINS + 1)
            ).astype(int)

            for bin_i, bin_label in enumerate(PROGRESS_BINS):
                b_start = edges[bin_i]
                b_end   = edges[bin_i + 1]
                n_bin   = b_end - b_start

                if n_bin < min_samples:
                    skipped.append((trial_idx, label, bin_label,
                        f"bin too short ({n_bin} samples < {min_samples}); "
                        f"state window={n_samp} samples"))
                    continue

                t_start_g = session_offset + b_start
                t_end_g   = session_offset + b_end
                vec = concat_z[:, t_start_g:t_end_g].mean(axis=1)  # (n_neurons,)

                if np.any(np.isnan(vec)):
                    skipped.append((trial_idx, label, bin_label,
                        "NaN in averaged vector"))
                    continue

                X_list.append(vec)
                y_list.append(label)
                prog_list.append(bin_label)
                sid_list.append(sess_index)
                tid_list.append(trial_idx)

    if len(X_list) == 0:
        return (np.empty((0, n_neurons)),
                np.empty(0, dtype='<U1'),
                np.empty(0, dtype=object),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                skipped)

    return (np.vstack(X_list),
            np.array(y_list, dtype='<U1'),
            np.array(prog_list, dtype=object),
            np.array(sid_list, dtype=int),
            np.array(tid_list, dtype=int),
            skipped)


# ─────────────────────────────────────────────────────────────────────────────
def build_goalprogress_dataset(data_dic, mouse_recday, valid_sessions,
                                neuron_subset=None):
    """
    Build the full (trial, state, progress-bin) z-scored dataset.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional

    Returns
    -------
    X : np.ndarray
        Shape (n_samples, n_neurons).
    y_state : np.ndarray of str
        Shape (n_samples,).
    progress : np.ndarray of str
        Shape (n_samples,).  Values: 'early', 'middle', 'late'.
    sess_id : np.ndarray of int
    trial_id : np.ndarray of int
    zero_var_mask : np.ndarray of bool
    n_skipped : int
    """
    raw_sessions, trial_times_list = [], []

    for sidx in valid_sessions:
        raw = data_dic[mouse_recday][sidx]['Neuron_raw']
        print(f"Task is {data_dic[mouse_recday][sidx]['Task']}")
        if neuron_subset is not None:
            raw = raw[np.asarray(neuron_subset), :]
        raw_sessions.append(raw)
        trial_times_list.append(data_dic[mouse_recday][sidx]['Trial_times'])

    if not raw_sessions:
        raise ValueError(f"No valid sessions found for {mouse_recday}.")

    n_neurons = raw_sessions[0].shape[0]
    print(f"{mouse_recday}: {len(valid_sessions)} sessions, {n_neurons} neurons")

    concat_z, offsets, lengths, zero_var_mask = zscore_concat_sessions(raw_sessions)

    n_zero = int(zero_var_mask.sum())
    if n_zero:
        print(f"  WARNING: {n_zero} neuron(s) with zero variance set to 0.")

    X_list, y_list, prog_list, sid_list, tid_list = [], [], [], [], []
    total_skipped = 0

    for sess_i, (sidx, offset, length, tt) in enumerate(
            zip(valid_sessions, offsets, lengths, trial_times_list)):

        print(f"  Session {sidx} (index {sess_i}): "
              f"{tt.shape[0]} trials, T={length} samples")

        X_s, y_s, prog_s, sid_s, tid_s, skipped = \
            extract_state_goalprogress_vectors(
                concat_z, offset, length, tt, sess_i
            )

        for trial_idx, label, bin_label, reason in skipped:
            print(f"    SKIP trial={trial_idx} state={label} "
                  f"bin={bin_label}: {reason}")
        total_skipped += len(skipped)

        if X_s.shape[0] > 0:
            X_list.append(X_s)
            y_list.append(y_s)
            prog_list.append(prog_s)
            sid_list.append(sid_s)
            tid_list.append(tid_s)

    if not X_list:
        raise RuntimeError(
            f"No valid (trial, state, bin) samples extracted for {mouse_recday}."
        )

    X        = np.vstack(X_list)
    y_state  = np.concatenate(y_list)
    progress = np.concatenate(prog_list)
    sess_id  = np.concatenate(sid_list)
    trial_id = np.concatenate(tid_list)

    print(f"  Dataset: {X.shape[0]} samples "
          f"({total_skipped} skipped) from "
          f"{X.shape[0] + total_skipped} (trial, state, bin) triples")
    for bin_label in PROGRESS_BINS:
        n = (progress == bin_label).sum()
        print(f"    {bin_label}: {n} samples")

    return X, y_state, progress, sess_id, trial_id, zero_var_mask, total_skipped


# ─────────────────────────────────────────────────────────────────────────────
def run_goalprogress_lda_analysis(data_dic, mouse_recday, valid_sessions,
                                   neuron_subset=None,
                                   variance_thresh=PCA_VARIANCE_THRESH,
                                   n_logo_shuffles=1000):
    """
    Full pipeline: build (trial, state, progress-bin) dataset, then for each
    progress bin independently run PCA → LDA state separation →
    leave-one-session-out CV.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional
    variance_thresh : float
    n_logo_shuffles : int

    Returns
    -------
    results : dict
        Keyed by progress-bin label ('early', 'middle', 'late').
        Each value is a dict with keys:
          'X'            : (n_bin_samples, n_neurons)
          'X_pca'        : (n_bin_samples, n_pcs)
          'X_state_ld'   : (n_bin_samples, n_lds)
          'y_state'      : (n_bin_samples,) state labels
          'sess_id'      : (n_bin_samples,)
          'trial_id'     : (n_bin_samples,)
          'pca'          : fitted PCA object
          'lda_state'    : fitted LDA object
          'zero_var_mask': (n_neurons,) bool
          'real_acc'     : LOGO CV balanced accuracy
          'null_accs'    : LOGO CV null distribution
          'p_value'      : permutation p-value
    """
    print(f"\n{'='*70}")
    print(f"Goal-progress LDA analysis: {mouse_recday}")
    print(f"Sessions: {valid_sessions}")
    print(f"{'='*70}")

    # 1. Build full dataset across all bins
    X, y_state, progress, sess_id, trial_id, zero_var_mask, _ = \
        build_goalprogress_dataset(
            data_dic, mouse_recday, valid_sessions,
            neuron_subset=neuron_subset
        )

    results = {}

    for bin_label in PROGRESS_BINS:
        print(f"\n{'-'*60}")
        print(f"  Progress bin: {bin_label.upper()}")
        print(f"{'-'*60}")

        mask = progress == bin_label
        X_bin      = X[mask]
        y_bin      = y_state[mask]
        sess_bin   = sess_id[mask]
        trial_bin  = trial_id[mask]

        # 2. Verify all four states are present in this bin
        present = set(y_bin)
        missing = set(EVENT_LABELS) - present
        if missing:
            print(f"  WARNING: states {sorted(missing)} absent from '{bin_label}' "
                  f"bin — skipping this bin.")
            results[bin_label] = None
            continue

        # 3. Guard: LOGO CV requires ≥ 2 sessions
        n_sessions = len(np.unique(sess_bin))
        if n_sessions < 2:
            print(f"  WARNING: only {n_sessions} session in '{bin_label}' bin — "
                  f"LOGO CV will be skipped.")

        # 4. PCA (fit only on this bin's data)
        X_pca, pca, _ = apply_pca_trialbins(X_bin, variance_thresh)

        # 5. LDA state separation
        lda_state, X_state_ld, y_state_out = run_lda_state_separation(
            X_pca, y_bin, EVENT_LABELS, EVENT_COLOURS,
            f"{mouse_recday} [{bin_label}]"
        )

        # 6. Leave-one-session-out CV
        if n_sessions >= 2:
            real_acc, null_accs, p_value = run_logo_state_cv(
                X_pca, y_state_out, sess_bin,
                EVENT_LABELS, f"{mouse_recday} [{bin_label}]",
                n_shuffles=n_logo_shuffles
            )
        else:
            real_acc  = float('nan')
            null_accs = np.array([])
            p_value   = float('nan')

        results[bin_label] = {
            'X'            : X_bin,
            'X_pca'        : X_pca,
            'X_state_ld'   : X_state_ld,
            'y_state'      : y_state_out,
            'sess_id'      : sess_bin,
            'trial_id'     : trial_bin,
            'pca'          : pca,
            'lda_state'    : lda_state,
            'zero_var_mask': zero_var_mask,
            'real_acc'     : real_acc,
            'null_accs'    : null_accs,
            'p_value'      : p_value,
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary — {mouse_recday}")
    for bin_label in PROGRESS_BINS:
        r = results.get(bin_label)
        if r is None:
            print(f"  {bin_label:8s}: SKIPPED")
        else:
            print(f"  {bin_label:8s}: acc={r['real_acc']:.3f}  "
                  f"p={r['p_value']:.3f}  "
                  f"n={r['X'].shape[0]} samples")
    print(f"{'='*70}\n")

    return results
