"""
Neural network regression decoder for time-in-session.

Predicts elapsed time since the session start using population firing rates
averaged into non-overlapping windows (default 10 s).

label_mode
----------
'seconds'      (default) — labels are bin-centre time in seconds since
               session start.
'normalised'   — labels are [0, 1] within each session (relative position).
'time_since_A'      — labels are elapsed time in seconds since the most
               recent A-state onset (Trial_times[:, 0]).  Bins before the
               first A event are excluded.
'time_since_event'  — labels are elapsed time in seconds since the most
               recent state transition of any kind (all entries in
               Trial_times, all columns).  Bins before the first event
               are excluded.

Leave-one-session-out (LOGO) cross-validation.  The StandardScaler is fit
on all sessions concatenated so that cross-session firing-rate variation
is preserved.

Main entry point
----------------
results = run_time_decoding(
    data_dic, mouse_recday, valid_sessions,
    bin_ms=10_000, n_epochs=50, label_mode='seconds',
)
plot_time_decoding(results)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────

class TimeDecoder(nn.Module):
    """Two-hidden-layer MLP with a single linear output for time regression."""

    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (batch,)


# ─────────────────────────────────────────────────────────────────────────────

def _get_trial_durations(sess_data, sampling_rate):
    """Return trial durations in seconds from Trial_times (end col - start col)."""
    tt = np.asarray(sess_data['Trial_times'])
    return (tt[:, -1] - tt[:, 0]) / sampling_rate


def _bin_session(data_dic, mouse_recday, session,
                 bin_ms=10_000, sampling_rate=40,
                 neuron_subset=None, label_mode='seconds',
                 max_trial_duration=None):
    """
    Average Neuron_raw into non-overlapping bin_ms windows and compute labels.

    Parameters
    ----------
    bin_ms : int
        Bin width in milliseconds (default 10 000 = 10 s).
    sampling_rate : int
        Samples per second (default 40, i.e. 25 ms bins).
    label_mode : str
        'seconds'      — bin-centre elapsed time in seconds since session start.
        'normalised'        — relative position [0, 1] within session.
        'time_since_A'      — elapsed time in seconds since most recent A onset
                              (Trial_times[:, 0]).  Bins before first A excluded.
        'time_since_event'  — elapsed time in seconds since most recent state
                              transition of any kind (all Trial_times columns
                              pooled).  Bins before first event excluded.

    Returns
    -------
    X      : (n_valid, n_neurons) float32
    labels : (n_valid,) float32
    t_sec  : (n_valid,) float32  — bin-centre time in seconds (for plotting)
    """
    sess_data = data_dic[mouse_recday][session]
    raw = np.asarray(sess_data['Neuron_raw'], dtype=float)
    if neuron_subset is not None:
        raw = raw[np.asarray(neuron_subset)]

    samples_per_bin = int(sampling_rate * bin_ms / 1000)
    n_neurons, n_timepoints = raw.shape
    n_bins = n_timepoints // samples_per_bin

    if n_bins < 2:
        return None, None, None

    X = np.stack([
        raw[:, b * samples_per_bin:(b + 1) * samples_per_bin].mean(axis=1)
        for b in range(n_bins)
    ], axis=0).astype(np.float32)           # (n_bins, n_neurons)

    bin_indices = np.arange(n_bins, dtype=np.float32)
    t_sec  = ((bin_indices + 0.5) * bin_ms / 1000.0).astype(np.float32)

    if label_mode == 'seconds':
        labels = t_sec.copy()
        valid  = np.ones(n_bins, dtype=bool)

    elif label_mode == 'normalised':
        labels = (bin_indices / (n_bins - 1)).astype(np.float32)
        valid  = np.ones(n_bins, dtype=bool)

    elif label_mode in ('time_since_A', 'time_since_event'):
        trial_times = np.asarray(sess_data['Trial_times'])
        if label_mode == 'time_since_A':
            onsets = np.sort(trial_times[:, 0].astype(float))
        else:  # time_since_event: pool all state-transition onsets
            onsets = np.sort(trial_times.ravel().astype(float))
        labels      = np.full(n_bins, np.nan, dtype=np.float32)
        for b in range(n_bins):
            bin_centre = (b + 0.5) * samples_per_bin
            preceding  = onsets[onsets <= bin_centre]
            if len(preceding):
                labels[b] = (bin_centre - preceding[-1]) / sampling_rate
        valid = ~np.isnan(labels)

    elif label_mode == 'reward_count':
        # Number of completed trials (rewards) before each bin.
        # Reward time = Trial_times[:, -1] (end of each trial).
        trial_times   = np.asarray(sess_data['Trial_times'])
        reward_times  = np.sort(trial_times[:, -1].astype(float))
        bin_centres   = (np.arange(n_bins) + 0.5) * samples_per_bin
        labels = np.searchsorted(reward_times, bin_centres,
                                 side='right').astype(np.float32)
        valid  = np.ones(n_bins, dtype=bool)

    else:
        raise ValueError(f"Unknown label_mode: {label_mode!r}")

    # ── optional: mask bins belonging to over-long trials ─────────────────────
    if max_trial_duration is not None:
        tt = np.asarray(sess_data['Trial_times'])
        onsets = tt[:, 0].astype(float)          # sample index of each A onset
        dur_sec = (tt[:, -1] - tt[:, 0]) / sampling_rate
        long_trial = dur_sec > max_trial_duration
        bin_centres = (np.arange(n_bins) + 0.5) * samples_per_bin
        trial_idx = np.searchsorted(onsets, bin_centres, side='right') - 1
        in_long = np.zeros(n_bins, dtype=bool)
        ok = (trial_idx >= 0) & (trial_idx < len(dur_sec))
        in_long[ok] = long_trial[trial_idx[ok]]
        valid = valid & ~in_long

    return X[valid], labels[valid], t_sec[valid]


# ─────────────────────────────────────────────────────────────────────────────

def run_time_decoding(data_dic, mouse_recday, valid_sessions,
                      neuron_subset=None,
                      bin_ms=10_000,
                      label_mode='seconds',
                      min_trials=None,
                      max_duration_sd=None,
                      hidden_dim=100,
                      weight_decay=1e-4,
                      n_epochs=50,
                      batch_size=64,
                      lr=1e-3,
                      sampling_rate=40,
                      device='cpu'):
    """
    Leave-one-session-out regression: predict time-in-session.

    Parameters
    ----------
    data_dic : dict
    mouse_recday : str
    valid_sessions : list
    neuron_subset : array-like of int, optional
    bin_ms : int
        Bin width in ms (default 10 000 = 10 s).
    min_trials : int or None
        If set, sessions with fewer trials in Trial_times are skipped
        (default None — no filter).
    max_duration_sd : float or None
        If set, trials whose duration exceeds (mean + max_duration_sd × SD)
        across all sessions are excluded from binning.  Computed globally
        before the session loop (default None — no filter).
    label_mode : str
        'seconds'      — predict true elapsed time (bin centre) in seconds
                         since session start.
        'normalised'        — predict relative position [0, 1] within session.
        'time_since_A'      — elapsed time in seconds since most recent A onset
                              (Trial_times[:, 0]).  Repeats every trial.
        'time_since_event' — elapsed time in seconds since most recent last-state
                              onset (Trial_times[:, -1]).  Repeats every trial.
        'reward_count'     — number of completed trials (rewards) before each bin.
                              Monotonically increasing step function; tests whether
                              neurons track task events rather than wall-clock time.
    hidden_dim : int
    weight_decay : float
    n_epochs : int
    batch_size : int
    lr : float
    sampling_rate : int
    device : str

    Returns
    -------
    results : dict
        'fold_r'           : list of Pearson r per fold
        'fold_r2'          : list of R² per fold
        'fold_true'        : list of (n_bins,) true label arrays
        'fold_pred'        : list of (n_bins,) predicted label arrays
        'fold_t_sec'       : list of (n_bins,) bin-centre times in seconds
        'fold_sessions'    : list of session keys
        'fold_train_curves': list of (n_epochs,) MSE on train per epoch
        'fold_test_curves' : list of (n_epochs,) MSE on test  per epoch
        'mouse_recday'     : str
        'bin_ms'           : int
        'label_mode'       : str
    """
    _VALID_MODES = ('seconds', 'normalised', 'time_since_A', 'time_since_event',
                    'reward_count')
    if label_mode not in _VALID_MODES:
        raise ValueError(f"label_mode must be one of {_VALID_MODES}, got {label_mode!r}")

    print(f"\n{'='*60}")
    print(f"Time-in-session decoding: {mouse_recday}")
    print(f"Sessions: {valid_sessions}  |  bin = {bin_ms/1000:.0f} s  |  labels = {label_mode}")
    print(f"{'='*60}")

    # ── compute global trial-duration threshold ───────────────────────────────
    max_trial_duration = None
    if max_duration_sd is not None:
        all_durs = np.concatenate([
            _get_trial_durations(data_dic[mouse_recday][s], sampling_rate)
            for s in valid_sessions
            if s in data_dic[mouse_recday]
        ])
        max_trial_duration = all_durs.mean() + max_duration_sd * all_durs.std()
        print(f"Trial-duration filter: {max_trial_duration:.1f} s "
              f"(mean={all_durs.mean():.1f}, SD={all_durs.std():.1f}, "
              f"threshold=mean+{max_duration_sd}×SD)")

    # ── load and bin all sessions ─────────────────────────────────────────────
    sess_bins = {}
    for sidx in valid_sessions:
        if min_trials is not None:
            n_trials = np.asarray(
                data_dic[mouse_recday][sidx]['Trial_times']
            ).shape[0]
            if n_trials < min_trials:
                print(f"  Session {sidx}: {n_trials} trials — skipped "
                      f"(< {min_trials})")
                continue

        X, labels, t_sec = _bin_session(
            data_dic, mouse_recday, sidx,
            bin_ms=bin_ms, sampling_rate=sampling_rate,
            neuron_subset=neuron_subset, label_mode=label_mode,
            max_trial_duration=max_trial_duration,
        )
        if X is None or X.shape[0] == 0:
            print(f"  Session {sidx}: too short or no valid bins — skipped")
            continue
        sess_bins[sidx] = (X, labels, t_sec)
        print(f"  Session {sidx}: {X.shape[0]} bins × {X.shape[1]} neurons  "
              f"({t_sec[-1]:.0f} s)")

    if len(sess_bins) < 2:
        raise RuntimeError("Need at least 2 valid sessions for LOGO CV.")

    # ── fit global scaler ─────────────────────────────────────────────────────
    all_X = np.vstack([v[0] for v in sess_bins.values()])
    scaler = StandardScaler()
    scaler.fit(all_X)
    n_neurons = all_X.shape[1]

    print(f"\n  {n_neurons} neurons, {len(sess_bins)} sessions")

    # ── LOGO CV ───────────────────────────────────────────────────────────────
    fold_r, fold_r2 = [], []
    fold_true, fold_pred, fold_t_sec = [], [], []
    fold_sessions = []
    fold_train_curves, fold_test_curves = [], []

    all_sessions = list(sess_bins.keys())

    for test_sess in all_sessions:
        train_sessions = [s for s in all_sessions if s != test_sess]

        # build train
        X_train = np.vstack([sess_bins[s][0] for s in train_sessions])
        t_train = np.concatenate([sess_bins[s][1] for s in train_sessions])

        # build test
        X_test, t_test, t_sec = sess_bins[test_sess]

        X_train_s = scaler.transform(X_train).astype(np.float32)
        X_test_s  = scaler.transform(X_test).astype(np.float32)

        X_train_t = torch.FloatTensor(X_train_s).to(device)
        t_train_t = torch.FloatTensor(t_train).to(device)
        X_test_t  = torch.FloatTensor(X_test_s).to(device)

        loader = DataLoader(
            TensorDataset(X_train_t, t_train_t),
            batch_size=batch_size, shuffle=True
        )

        model     = TimeDecoder(n_neurons, hidden_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)

        train_curve, test_curve = [], []
        model.train()
        for epoch in tqdm(range(n_epochs),
                          desc=f"  Fold (held={test_sess})", leave=False):
            for X_b, t_b in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b), t_b)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                mse_train = criterion(model(X_train_t), t_train_t).item()
                mse_test  = criterion(model(X_test_t),
                                      torch.FloatTensor(t_test).to(device)).item()
            train_curve.append(mse_train)
            test_curve.append(mse_test)
            model.train()

        fold_train_curves.append(train_curve)
        fold_test_curves.append(test_curve)

        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).cpu().numpy()

        r, _ = pearsonr(t_test, pred)
        ss_res = np.sum((t_test - pred) ** 2)
        ss_tot = np.sum((t_test - t_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        fold_r.append(r)
        fold_r2.append(r2)
        fold_true.append(t_test)
        fold_pred.append(pred)
        fold_t_sec.append(t_sec)
        fold_sessions.append(test_sess)

        unit = 's' if label_mode in ('seconds', 'time_since_A', 'time_since_event') else ''
        print(f"  Held-out session {test_sess}:  r = {r:.3f},  R² = {r2:.3f}  "
              f"MAE = {np.mean(np.abs(t_test - pred)):.1f}{unit}")

    print(f"\n  Mean r  = {np.mean(fold_r):.3f} ± {np.std(fold_r):.3f}")
    print(f"  Mean R² = {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")
    print(f"{'='*60}\n")

    return {
        'fold_r':            fold_r,
        'fold_r2':           fold_r2,
        'fold_true':         fold_true,
        'fold_pred':         fold_pred,
        'fold_t_sec':        fold_t_sec,
        'fold_sessions':     fold_sessions,
        'fold_train_curves': fold_train_curves,
        'fold_test_curves':  fold_test_curves,
        'mouse_recday':      mouse_recday,
        'bin_ms':            bin_ms,
        'label_mode':        label_mode,
    }


# ─────────────────────────────────────────────────────────────────────────────

def plot_time_decoding(results):
    """
    Plot predicted vs true time-in-session for each held-out fold.

    Top row: scatter of predicted vs true.
    Bottom row: time-series of true (black) and predicted (red) over seconds.

    Parameters
    ----------
    results : dict
        Output of run_time_decoding.
    """
    fold_true     = results['fold_true']
    fold_pred     = results['fold_pred']
    fold_t_sec    = results['fold_t_sec']
    fold_sessions = results['fold_sessions']
    fold_r        = results['fold_r']
    fold_r2       = results['fold_r2']
    mouse_recday  = results['mouse_recday']
    bin_s         = results['bin_ms'] / 1000
    label_mode    = results.get('label_mode', 'normalised')

    if label_mode == 'seconds':
        axis_label = 'Time (s)'
        ts_ylabel  = 'Elapsed time (s)'
    elif label_mode == 'time_since_A':
        axis_label = 'Time since A (s)'
        ts_ylabel  = 'Time since A (s)'
    elif label_mode == 'time_since_event':
        axis_label = 'Time since event (s)'
        ts_ylabel  = 'Time since event (s)'
    else:
        axis_label = 'Time (norm.)'
        ts_ylabel  = 'Relative time'

    n_folds = len(fold_true)
    fig, axes = plt.subplots(2, n_folds,
                             figsize=(3.5 * n_folds, 7),
                             squeeze=False)

    for fi in range(n_folds):
        t_true = fold_true[fi]
        t_pred = fold_pred[fi]
        t_sec  = fold_t_sec[fi]
        sess   = fold_sessions[fi]
        r      = fold_r[fi]
        r2     = fold_r2[fi]
        mae    = np.mean(np.abs(t_true - t_pred))

        # ── scatter: predicted vs true ────────────────────────────────────────
        ax = axes[0, fi]
        lim_lo = min(t_true.min(), t_pred.min())
        lim_hi = max(t_true.max(), t_pred.max())
        ax.scatter(t_true, t_pred, s=10, alpha=0.5, color='steelblue',
                   rasterized=True)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=1,
                label='identity')
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect('equal')
        mae_str = (f'{mae:.1f} s'
                   if label_mode in ('seconds', 'time_since_A', 'time_since_event')
                   else f'{mae:.3f}')
        ax.set_xlabel(f'True {axis_label}', fontsize=8)
        ax.set_ylabel(f'Predicted {axis_label}', fontsize=8)
        ax.set_title(f'Session {sess}\nr = {r:.3f}, R² = {r2:.3f}, MAE = {mae_str}',
                     fontsize=9, fontweight='bold')

        # ── time series ────────────────────────────────────────────────────────
        ax2 = axes[1, fi]
        ax2.plot(t_sec, t_true, color='black', lw=1.5, label='True')
        ax2.plot(t_sec, t_pred, color='crimson', lw=1.5,
                 alpha=0.8, label='Predicted')
        ax2.set_xlabel(f'Time (s, {bin_s:.0f} s bins)', fontsize=8)
        ax2.set_ylabel(ts_ylabel, fontsize=8)
        ax2.legend(fontsize=7)

    fig.suptitle(
        f'Time-in-session decoding (LOGO, {label_mode}) — {mouse_recday}\n'
        f'Mean r = {np.mean(fold_r):.3f} ± {np.std(fold_r):.3f}  '
        f'R² = {np.mean(fold_r2):.3f}',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()
    return fig


def plot_time_learning_curves(results):
    """
    Plot mean ± SEM MSE on train and test sets over training epochs.

    Parameters
    ----------
    results : dict
        Output of run_time_decoding.
    """
    train_curves = np.array(results['fold_train_curves'])  # (n_folds, n_epochs)
    test_curves  = np.array(results['fold_test_curves'])
    epochs = np.arange(1, train_curves.shape[1] + 1)

    mean_tr = train_curves.mean(axis=0)
    sem_tr  = train_curves.std(axis=0) / np.sqrt(len(train_curves))
    mean_te = test_curves.mean(axis=0)
    sem_te  = test_curves.std(axis=0) / np.sqrt(len(test_curves))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(epochs, mean_tr - sem_tr, mean_tr + sem_tr,
                    alpha=0.2, color='steelblue')
    ax.plot(epochs, mean_tr, color='steelblue', lw=2, label='Train MSE')
    ax.fill_between(epochs, mean_te - sem_te, mean_te + sem_te,
                    alpha=0.2, color='crimson')
    ax.plot(epochs, mean_te, color='crimson', lw=2, label='Test MSE')
    label_mode = results.get('label_mode', 'normalised')
    mse_unit   = 's²' if label_mode in ('seconds', 'time_since_A', 'time_since_event') else '(norm.²)'
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'MSE ({mse_unit})')
    ax.set_title(
        f'Time decoder learning curves — {results["mouse_recday"]}',
        fontweight='bold',
    )
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Goal-progress-preserving null distribution
# ─────────────────────────────────────────────────────────────────────────────

def _get_gp_assignments(sess_data, t_sec, sampling_rate, n_progress_bins):
    """
    Assign each time bin to a (state × goal-progress-bin) group.

    Goal progress is the normalised position within a single state epoch:
    0 at onset, 1 at the next state onset.  Each state is divided into
    n_progress_bins equal bands.

    Parameters
    ----------
    sess_data : dict
        Session dict containing 'Trial_times' (n_trials × n_states), sample
        indices of each state onset.
    t_sec : (n_bins,) float
        Bin-centre times in seconds.
    sampling_rate : int
        Samples per second.
    n_progress_bins : int

    Returns
    -------
    assignments : (n_bins,) int
        state_idx * n_progress_bins + progress_bin_idx, or -1 if the bin
        centre does not fall inside any trial state epoch.
    """
    trial_times = np.asarray(sess_data['Trial_times'])
    n_trials, n_states = trial_times.shape
    bin_samples = (t_sec * sampling_rate).astype(float)

    # pre-build (start, end, state_idx) for every state epoch in the session
    epochs = []
    for t in range(n_trials):
        for s in range(n_states):
            start = float(trial_times[t, s])
            if s < n_states - 1:
                end = float(trial_times[t, s + 1])
            elif t < n_trials - 1:
                end = float(trial_times[t + 1, 0])
            else:
                end = float('inf')
            epochs.append((start, end, s))

    assignments = np.full(len(t_sec), -1, dtype=int)
    for bi, bc in enumerate(bin_samples):
        for start, end, s_idx in epochs:
            if start <= bc < end:
                dur = end - start
                if dur > 0 and np.isfinite(dur):
                    pb = min(int((bc - start) / dur * n_progress_bins),
                             n_progress_bins - 1)
                    assignments[bi] = s_idx * n_progress_bins + pb
                break

    return assignments


def _shuffle_within_gp(X, gp_assignments, rng, cross_state=False,
                       n_progress_bins=None):
    """
    Permute rows of X within goal-progress groups.
    Bins with gp_assignments == -1 are left in place.

    Parameters
    ----------
    cross_state : bool
        False (default) — shuffle within (state × gp_bin): early-A ↔ early-A.
        True            — shuffle within gp_bin only: early-A ↔ early-B/C etc.
                          Requires n_progress_bins.
    n_progress_bins : int
        Required when cross_state=True to recover the gp_bin index from the
        compound assignment code (gp_assignments % n_progress_bins).
    """
    X_shuf = X.copy()
    if cross_state:
        if n_progress_bins is None:
            raise ValueError("n_progress_bins required when cross_state=True")
        # group key is just the gp_bin (ignore which state)
        group_keys = np.where(gp_assignments >= 0,
                              gp_assignments % n_progress_bins, -1)
    else:
        group_keys = gp_assignments

    for gp in np.unique(group_keys[group_keys >= 0]):
        idx = np.where(group_keys == gp)[0]
        if len(idx) > 1:
            X_shuf[idx] = X[rng.permutation(idx)]
    return X_shuf


def run_goalprogress_null(data_dic, mouse_recday, valid_sessions,
                          label_mode='time_since_event',
                          n_progress_bins=3,
                          cross_state=False,
                          n_shuffles=100,
                          min_trials=None,
                          max_duration_sd=None,
                          neuron_subset=None,
                          bin_ms=10_000,
                          hidden_dim=100,
                          weight_decay=1e-4,
                          n_epochs=50,
                          batch_size=64,
                          lr=1e-3,
                          sampling_rate=40,
                          device='cpu',
                          seed=42):
    """
    Null distribution for run_time_decoding using goal-progress-preserving
    shuffles.

    For each shuffle, within every session neural data bins are swapped with
    other bins in the same goal-progress group.  Two shuffle modes:

    cross_state=False (default)
        Shuffle within (state × gp_bin): bins with the same state identity
        AND the same fraction-through-state band.  Preserves state identity
        and goal-progress level.  Tests whether fine temporal ordering within
        a state × gp band is decodable.

    cross_state=True
        Shuffle within gp_bin only (ignoring state): early-A bins can swap
        with early-B, early-C, etc.  Preserves goal-progress level but breaks
        state identity.  Tests whether the decoder uses state-specific
        information beyond just "how far through a state".

    The full LOGO CV is then re-run on the shuffled data and the mean Pearson r
    across folds is recorded.

    Parameters
    ----------
    data_dic, mouse_recday, valid_sessions, label_mode, ...
        Same as run_time_decoding.
    n_progress_bins : int
        Bands per state used for grouping (default 3).
    cross_state : bool
        Shuffle mode — see above (default False).
    n_shuffles : int
        Number of shuffle iterations (default 100).
    min_trials : int or None
        If set, skip sessions with fewer trials than this threshold.
    max_duration_sd : float or None
        If set, trials longer than (mean + max_duration_sd × SD) across all
        sessions are excluded from binning (default None).
    seed : int

    Returns
    -------
    null_r : (n_shuffles,) float
        Mean Pearson r across LOGO folds for each shuffle.
    """
    rng = np.random.default_rng(seed)

    # ── compute global trial-duration threshold ───────────────────────────────
    max_trial_duration = None
    if max_duration_sd is not None:
        all_durs = np.concatenate([
            _get_trial_durations(data_dic[mouse_recday][s], sampling_rate)
            for s in valid_sessions
            if s in data_dic[mouse_recday]
        ])
        max_trial_duration = all_durs.mean() + max_duration_sd * all_durs.std()
        print(f"Trial-duration filter: {max_trial_duration:.1f} s "
              f"(mean={all_durs.mean():.1f}, SD={all_durs.std():.1f}, "
              f"threshold=mean+{max_duration_sd}×SD)")

    # ── load and bin all sessions, plus gp assignments ────────────────────────
    sess_bins = {}   # sidx -> (X, labels, t_sec, gp_assignments)
    for sidx in valid_sessions:
        if min_trials is not None:
            n_trials = np.asarray(
                data_dic[mouse_recday][sidx]['Trial_times']
            ).shape[0]
            if n_trials < min_trials:
                print(f"  Session {sidx}: {n_trials} trials — skipped (< {min_trials})")
                continue
        X, labels, t_sec = _bin_session(
            data_dic, mouse_recday, sidx,
            bin_ms=bin_ms, sampling_rate=sampling_rate,
            neuron_subset=neuron_subset, label_mode=label_mode,
            max_trial_duration=max_trial_duration,
        )
        if X is None or X.shape[0] == 0:
            continue
        gp = _get_gp_assignments(
            data_dic[mouse_recday][sidx], t_sec, sampling_rate, n_progress_bins
        )
        sess_bins[sidx] = (X, labels, t_sec, gp)

    if len(sess_bins) < 2:
        raise RuntimeError("Need at least 2 valid sessions.")

    all_sessions  = list(sess_bins.keys())
    all_X_concat  = np.vstack([v[0] for v in sess_bins.values()])
    scaler        = StandardScaler()
    scaler.fit(all_X_concat)
    n_neurons = all_X_concat.shape[1]

    def _logo_metrics(shuffled_X):
        """LOGO CV on a dict {sess: X_array}; returns (mean_r, mean_mae)."""
        fold_rs, fold_maes = [], []
        for test_sess in all_sessions:
            train_sess = [s for s in all_sessions if s != test_sess]

            X_tr = scaler.transform(
                np.vstack([shuffled_X[s] for s in train_sess])
            ).astype(np.float32)
            t_tr = np.concatenate([sess_bins[s][1] for s in train_sess])
            X_te = scaler.transform(shuffled_X[test_sess]).astype(np.float32)
            t_te = sess_bins[test_sess][1]

            X_tr_t = torch.FloatTensor(X_tr).to(device)
            t_tr_t = torch.FloatTensor(t_tr).to(device)
            X_te_t = torch.FloatTensor(X_te).to(device)

            loader = DataLoader(TensorDataset(X_tr_t, t_tr_t),
                                batch_size=batch_size, shuffle=True)
            model     = TimeDecoder(n_neurons, hidden_dim).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
            model.train()
            for _ in range(n_epochs):
                for X_b, t_b in loader:
                    optimizer.zero_grad()
                    criterion(model(X_b), t_b).backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(X_te_t).cpu().numpy()
            fold_rs.append(pearsonr(t_te, pred)[0])
            fold_maes.append(float(np.mean(np.abs(t_te - pred))))

        return float(np.mean(fold_rs)), float(np.mean(fold_maes))

    mode_str = "cross-state gp" if cross_state else "within-state gp"
    print(f"\nRunning {n_shuffles} goal-progress-preserving shuffles "
          f"({mouse_recday}, n_progress_bins={n_progress_bins}, mode={mode_str})...")
    null_r, null_mae = [], []
    for _ in tqdm(range(n_shuffles), desc=f"  GP null ({mode_str})"):
        shuffled = {
            s: _shuffle_within_gp(sess_bins[s][0], sess_bins[s][3], rng,
                                   cross_state=cross_state,
                                   n_progress_bins=n_progress_bins)
            for s in all_sessions
        }
        r, mae = _logo_metrics(shuffled)
        null_r.append(r)
        null_mae.append(mae)

    return {'null_r': np.array(null_r), 'null_mae': np.array(null_mae)}


def plot_goalprogress_null(real_results, null_results, cross_state=False):
    """
    Two-panel histogram of the goal-progress null distribution.

    Left: Pearson r null vs observed.
    Right: MAE null vs observed.

    Parameters
    ----------
    real_results : dict
        Output of run_time_decoding.
    null_results : dict
        Output of run_goalprogress_null — keys 'null_r' and 'null_mae'.
    cross_state : bool
        Pass True if the null was generated with cross_state=True.

    Returns
    -------
    fig, {'p_r': float, 'p_mae': float}
    """
    null_r   = null_results['null_r']
    null_mae = null_results['null_mae']

    obs_r   = float(np.mean(real_results['fold_r']))
    obs_mae = float(np.mean([np.mean(np.abs(tr - pr))
                             for tr, pr in zip(real_results['fold_true'],
                                               real_results['fold_pred'])]))
    # higher r is better; lower MAE is better
    p_r   = float((null_r   >= obs_r  ).mean())
    p_mae = float((null_mae <= obs_mae).mean())

    label    = real_results.get('mouse_recday', '')
    lmode    = real_results.get('label_mode', '')
    mode_str = 'cross-state gp' if cross_state else 'within-state gp'
    n_shuf   = len(null_r)

    mae_unit = 's' if lmode in ('seconds', 'time_since_A', 'time_since_event') else ''

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── Pearson r ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(null_r, bins=30, color='silver', edgecolor='white', lw=0.5,
            label=f'GP null (n={n_shuf})')
    ax.axvline(obs_r, color='crimson', lw=2,
               label=f'Observed r = {obs_r:.3f}')
    ax.set_xlabel('Mean Pearson r (LOGO CV)')
    ax.set_ylabel('Count')
    ax.set_title(f'Pearson r — p = {p_r:.3f}', fontweight='bold')
    ax.legend()

    # ── MAE ───────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(null_mae, bins=30, color='silver', edgecolor='white', lw=0.5,
             label=f'GP null (n={n_shuf})')
    ax2.axvline(obs_mae, color='crimson', lw=2,
                label=f'Observed MAE = {obs_mae:.2f}{mae_unit}')
    ax2.set_xlabel(f'Mean MAE{" (" + mae_unit + ")" if mae_unit else ""} (LOGO CV)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'MAE — p = {p_mae:.3f}', fontweight='bold')
    ax2.legend()

    fig.suptitle(
        f'Goal-progress null [{mode_str}] — {label}  [{lmode}]',
        fontweight='bold',
    )
    plt.tight_layout()
    plt.show()
    return fig, {'p_r': p_r, 'p_mae': p_mae}


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE (copy to notebook)
# ─────────────────────────────────────────────────────────────────────────────
"""
from nn_time_decoder import (run_time_decoding, plot_time_decoding,
                             plot_time_learning_curves,
                             run_goalprogress_null, plot_goalprogress_null)
import torch

results = run_time_decoding(
    data_dic       = data_dic,
    mouse_recday   = mouse_recday,
    valid_sessions = valid_sessions_dic[mouse_recday],
    bin_ms         = 10_000,     # 10 s bins
    label_mode     = 'seconds',  # predict true elapsed time in seconds
    hidden_dim     = 100,
    n_epochs       = 50,
    batch_size     = 64,
    lr             = 1e-3,
    weight_decay   = 1e-4,
    sampling_rate  = 40,
    device         = 'cuda' if torch.cuda.is_available() else 'cpu',
)

plot_time_decoding(results)
plot_time_learning_curves(results)
"""
