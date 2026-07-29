"""
Anchoring regression — V3 (raw-time reconciliation to El-Gaby et al. 2024, Figure5).

This is a faithful reimplementation of `code/Figure5_Regression.ipynb` on our `data_dic`
(the reference notebook itself cannot be run — its Input_folder is El-Gaby's private
Dropbox/OSF). It differs from `elasticnet_regression_v2.py` on the axes that made the PFC
results diverge:

  * regressors are built in RAW behavioural time (~25 ms bins) from INTEGER node locations
    (edges/>num_locations -> NaN) and the TRUE per-bin goal-progress phase, not from the
    time-warped 360-bin `Locs_norm` (which mean-averages node labels -> fractional/aliased).
  * the per-neuron estimator is a Poisson GLM (alpha=1) by default, with selectable
    ElasticNet (alpha=0.01, positive) and positive LinearRegression branches — exactly the
    reference's `Poisson_regression` / `regularize` switch.
  * leave-one-session-out CV; each neuron is fit only on bins of its preferred goal-progress
    phase.
  * the reported metric is the El-Gaby one: predicted & actual are normalized to 360 bins,
    collapsed to one mean per state over PREFERRED-PHASE bins, and Pearson-correlated (n=4).

Required fields in data_dic[mouse_recday][session]:
    'Neuron_raw'  (n_neurons, n_bins)   raw firing per 25 ms bin
    'Locs_raw'    (n_bins,)             node 1..21 or NaN  (nodes 1..9 used; 10..21 = edges)
    'Trial_times' (n_trials, n_states+1) state-boundary times in 25 ms BIN indices

Regressor layout: (num_locations x num_goal_progress_bins x num_lags) = 9 x 3 x 12 = 324,
flattened C-order so lag is the fastest axis (matches identify_nonzero_lag_neurons).
"""

import numpy as np
from sklearn.linear_model import ElasticNet, PoissonRegressor, LinearRegression
from scipy import stats
from scipy.stats import binned_statistic
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class RegressionConfigV3:
    """Config for the raw-time anchoring regression.

    use_poisson=True  -> PoissonRegressor(alpha=poisson_alpha)        [reference default]
    use_poisson=False & regularize=True
                      -> ElasticNet(alpha=elasticnet_alpha, l1_ratio, positive)
    use_poisson=False & regularize=False
                      -> LinearRegression(positive)
    """

    def __init__(
        self,
        num_locations=9,
        num_goal_progress_bins=3,
        num_task_states=4,
        num_lags=12,
        use_poisson=True,
        regularize=True,
        poisson_alpha=1.0,
        elasticnet_alpha=0.01,
        l1_ratio=0.5,
        positive=True,
        num_bins_per_state=90,
        state_tuning_p_threshold=0.05,
        max_iter=1000,
    ):
        self.num_locations = num_locations
        self.num_goal_progress_bins = num_goal_progress_bins
        self.num_task_states = num_task_states
        self.num_lags = num_lags
        self.use_poisson = use_poisson
        self.regularize = regularize
        self.poisson_alpha = poisson_alpha
        self.elasticnet_alpha = elasticnet_alpha
        self.l1_ratio = l1_ratio
        self.positive = positive
        self.num_bins_per_state = num_bins_per_state
        self.state_tuning_p_threshold = state_tuning_p_threshold
        self.max_iter = max_iter
        self.total_bins = num_bins_per_state * num_task_states           # 360
        self.num_regressors = num_locations * num_goal_progress_bins * num_lags  # 324
        self.bins_per_phase = num_bins_per_state // num_goal_progress_bins       # 30


# ============================================================================
# Raw per-bin task state / goal-progress phase (from Trial_times)
# ============================================================================

def compute_phase_state_raw(trial_times, num_phases=3, num_states=4):
    """Per-bin task state (0..num_states-1) and goal-progress phase (0..num_phases-1) in
    raw time, derived from Trial_times (25 ms bin indices). Port of
    glm_analysis_v2.compute_task_state_arrays with num_bins=num_phases; goal-progress is the
    linear time-fraction through each inter-reward interval, binned into `num_phases` thirds.
    Returns arrays of length max(trial_times)+1.
    """
    trial_times = np.asarray(trial_times, dtype=int)
    max_time = int(np.max(trial_times))
    state_array = np.zeros(max_time + 1, dtype=int)
    phase_array = np.zeros(max_time + 1, dtype=int)

    # Assign state/phase per trial directly from its boundary columns
    # ([s0, s1, ..., s_{num_states}]): state = column index (A,B,C,D), phase = goal-progress
    # (linear time-fraction) within each state interval. Robust to duplicate inter-trial
    # boundaries, unlike a global sort + modulo.
    n_trials, n_cols = trial_times.shape
    n_state_cols = n_cols - 1
    for trial in range(n_trials):
        cols = trial_times[trial]
        for st in range(n_state_cols):
            a, b = int(cols[st]), int(cols[st + 1])
            if b <= a:
                continue
            t_range = np.arange(a, b)
            progress = (t_range - a) / (b - a)
            phase = np.minimum(np.floor(progress * num_phases).astype(int), num_phases - 1)
            state_array[a:b] = st % num_states
            phase_array[a:b] = phase
    return phase_array, state_array


# ============================================================================
# Raw-time lagged regressors (faithful port of Figure5_Regression.ipynb cell 15)
# ============================================================================

def generate_regressors_raw(locs_raw, phases_raw, config, multiple_bumps=True):
    """Build (T, 324) lagged anchoring regressors from raw integer locations and per-bin
    goal-progress phase. Bump state is continuous over the whole (per-session) sequence.

    Faithful to the reference loop: on a phase change the matching (location, phase) anchor is
    seeded at lag 0, all anchors roll +1 along the lag axis, then lag-1 is set to 1 for the
    matching anchor / 0 otherwise; on a within-phase location change the matching anchor is
    seeded at lag 1. A final roll(-1) along the lag axis undoes the +1 book-keeping so a freshly
    visited anchor sits at lag 0.
    """
    num_locs = config.num_locations
    num_phases = config.num_goal_progress_bins
    num_lags = config.num_lags

    locs = np.asarray(locs_raw, dtype=float)
    phases = np.asarray(phases_raw, dtype=int)
    T = len(locs)

    # node index 0..num_locs-1, or -1 for invalid (NaN / edge / out of range)
    nodes = np.full(T, -1, dtype=int)
    valid = (~np.isnan(locs)) & (locs >= 1) & (locs <= num_locs)
    nodes[valid] = locs[valid].astype(int) - 1

    module = np.zeros((num_locs, num_phases, num_lags))
    # float32: bumps are 0/1, and these (T, 324) arrays dominate memory on long raw sessions
    out = np.zeros((T, num_locs, num_phases, num_lags), dtype=np.float32)

    prev_phase = -1
    prev_loc = -1
    for t in range(T):
        loc_t = int(nodes[t])
        phase_t = int(phases[t])
        move_phase = (phase_t != prev_phase)
        move_location = (loc_t != prev_loc and loc_t >= 0)

        if move_phase:
            # 1) seed lag-0 of the matching anchor (before the roll)
            if loc_t >= 0:
                if multiple_bumps or np.sum(module[loc_t, phase_t]) == 0:
                    module[loc_t, phase_t, 0] = 1
            # 2) roll all anchors +1 along the lag axis
            module = np.roll(module, 1, axis=2)
            # 3) adjust lag-1: keep only the matching anchor, zero the rest where lag-1>0
            lag1 = module[:, :, 1]
            active = lag1 > 0
            keep = np.zeros_like(lag1)
            if loc_t >= 0:
                keep[loc_t, phase_t] = 1
            module[:, :, 1] = np.where(active, keep, lag1)
        elif move_location:
            if loc_t >= 0:
                if multiple_bumps or np.sum(module[loc_t, phase_t]) == 0:
                    module[loc_t, phase_t, 1] = 1

        out[t] = module
        prev_phase = phase_t
        prev_loc = loc_t

    out = np.roll(out, -1, axis=3)  # undo the forward lag book-keeping
    return out.reshape(T, config.num_regressors)


# ============================================================================
# Normalisation to 360 bins (for the readout metric and state-tuning)
# ============================================================================

def raw_to_norm(raw_1d, trial_times, config, return_mean=True, statistic='mean'):
    """Normalize a raw per-bin 1-D signal to a per-trial 360-bin grid (90 bins/state).
    Port of the reference `raw_to_norm`/`normalise`. Returns (360,) if return_mean else
    (n_full_trials, 360); None if no complete trial."""
    raw = np.asarray(raw_1d, dtype=float)
    tt = np.asarray(trial_times, dtype=int)
    nbps = config.num_bins_per_state
    nstates = config.num_task_states

    boundaries = np.hstack((np.concatenate(tt[:, :-1]), [tt[-1, -1]])).astype(int)

    rebinned = []
    for i in range(len(boundaries) - 1):
        a, b = int(boundaries[i]), int(boundaries[i + 1])
        if not (b > a and a >= 0 and b <= raw.shape[0]):
            rebinned.append(np.full(nbps, np.nan))
            continue
        seg = raw[a:b]
        if len(seg) < nbps:
            seg = np.repeat(seg, int(np.ceil(nbps / max(len(seg), 1))))
        idx = np.arange(len(seg))
        rb = binned_statistic(idx, seg, statistic=statistic, bins=nbps)[0]
        rebinned.append(rb)

    n_full = (len(rebinned) // nstates) * nstates
    if n_full == 0:
        return None
    arr = np.asarray(rebinned[:n_full]).reshape(n_full // nstates, nbps * nstates)
    if return_mean:
        return np.nanmean(arr, axis=0)
    return arr


def _phase_label_per_norm_bin(config):
    """Phase (0..num_phases-1) of each of the 360 normalized bins (equal thirds per state)."""
    nbps = config.num_bins_per_state
    per_state = np.repeat(np.arange(config.num_goal_progress_bins), nbps // config.num_goal_progress_bins)
    if len(per_state) < nbps:  # pad if not divisible
        per_state = np.append(per_state, np.full(nbps - len(per_state), config.num_goal_progress_bins - 1))
    return np.tile(per_state, config.num_task_states)


# ============================================================================
# State-tuned neuron filter (El-Gaby z-score-and-t-test, computed from raw)
# ============================================================================

def identify_state_tuned_neurons_raw(neuron_raw, trial_times, config, p_threshold=None):
    """Boolean mask of state-tuned neurons. Per neuron: peak firing per state per trial ->
    z-score across states within trial -> mean across trials -> preferred state = argmax ->
    one-sample t-test of that state's per-trial z against 0."""
    if p_threshold is None:
        p_threshold = config.state_tuning_p_threshold
    n_neurons = neuron_raw.shape[0]
    nstates = config.num_task_states
    nbps = config.num_bins_per_state
    is_tuned = np.zeros(n_neurons, dtype=bool)

    for ni in range(n_neurons):
        per_trial = raw_to_norm(neuron_raw[ni], trial_times, config, return_mean=False)
        if per_trial is None or per_trial.shape[0] < 3:
            continue
        n_trials = per_trial.shape[0]
        peak = np.full((n_trials, nstates), np.nan)
        for s in range(nstates):
            peak[:, s] = np.nanmax(per_trial[:, s * nbps:(s + 1) * nbps], axis=1)
        row_mean = np.nanmean(peak, axis=1, keepdims=True)
        row_std = np.nanstd(peak, axis=1, keepdims=True)
        row_std[row_std == 0] = np.nan
        z = (peak - row_mean) / row_std
        pref = int(np.nanargmax(np.nanmean(z, axis=0)))
        zp = z[:, pref]
        zp = zp[~np.isnan(zp)]
        if len(zp) < 3:
            continue
        _, pval = stats.ttest_1samp(zp, 0)
        if pval < p_threshold:
            is_tuned[ni] = True
    return is_tuned


# ============================================================================
# Estimator (reference Poisson / ElasticNet / Linear branches)
# ============================================================================

def fit_regression_v3(X, y, config):
    """Fit the chosen estimator, dropping rows with NaNs. Returns coefficient vector."""
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    Xv, yv = X[valid], y[valid]
    if len(yv) < 10 or np.all(yv == yv[0]):
        return np.full(X.shape[1], np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if config.use_poisson:
            model = PoissonRegressor(alpha=config.poisson_alpha, max_iter=config.max_iter)
        elif config.regularize:
            model = ElasticNet(alpha=config.elasticnet_alpha, l1_ratio=config.l1_ratio,
                               positive=config.positive, max_iter=config.max_iter)
        else:
            model = LinearRegression(positive=config.positive)
        try:
            model.fit(Xv, yv)
        except Exception:
            return np.full(X.shape[1], np.nan)
    return model.coef_


# ============================================================================
# Per-session preparation + cross-validated regression
# ============================================================================

def _prepare_session(session_data, config):
    """Build raw regressors / aligned locations / phases / neuron matrix for one session.
    Returns None if required fields are missing or no complete trial."""
    for key in ('Neuron_raw', 'Locs_raw', 'Trial_times'):
        if key not in session_data or session_data[key] is None:
            return None
    neuron_raw = np.asarray(session_data['Neuron_raw'], dtype=float)   # (n_neurons, n_bins)
    locs_raw = np.asarray(session_data['Locs_raw'], dtype=float)        # (n_bins,)
    tt = np.asarray(session_data['Trial_times'], dtype=int)            # (n_trials, n_states+1)
    if neuron_raw.ndim != 2 or tt.ndim != 2 or tt.shape[0] < 2:
        return None

    phase_per_bin, _ = compute_phase_state_raw(
        tt, num_phases=config.num_goal_progress_bins, num_states=config.num_task_states)

    L = min(len(locs_raw), neuron_raw.shape[1], len(phase_per_bin))
    if L < config.num_bins_per_state:
        return None
    locs = locs_raw[:L].copy()
    locs[locs > config.num_locations] = np.nan          # drop edges (10..21)
    phases = phase_per_bin[:L]
    neuron = neuron_raw[:, :L].T                          # (L, n_neurons)

    regressors = generate_regressors_raw(locs, phases, config)
    return {
        'regressors': regressors,        # (L, 324)
        'locs': locs,                    # (L,)
        'phases': phases,                # (L,)
        'neuron': neuron,                # (L, n_neurons)
        'neuron_raw': neuron_raw,        # (n_neurons, n_bins)  (for state tuning)
        'trial_times': tt,
        'n_neurons': neuron_raw.shape[0],
    }


def run_cross_validated_regression_v3(data_dic, mouse_recday, config, valid_sessions=None,
                                      verbose=False):
    """Leave-one-session-out raw-time anchoring regression for one mouse_recday.

    Returns a results dict with cv_coeffs (n_neurons, n_folds, 324), the El-Gaby 4-state
    preferred-phase correlations `corrs` and `corrs_nonzero` (lag-0 zeroed), and a
    state-tuned mask. Returns None if fewer than 2 usable sessions.
    """
    sessions = list(data_dic[mouse_recday].keys()) if valid_sessions is None else list(valid_sessions)
    preps = {}
    for s in sessions:
        if s == 'valid_sessions' or s not in data_dic[mouse_recday]:
            continue
        p = _prepare_session(data_dic[mouse_recday][s], config)
        if p is not None:
            preps[s] = p
    used = list(preps.keys())
    if len(used) < 2:
        if verbose:
            print(f"  {mouse_recday}: <2 usable sessions, skipping")
        return None

    n_neurons = preps[used[0]]['n_neurons']
    n_folds = len(used)
    n_reg = config.num_regressors
    num_lags = config.num_lags

    cv_coeffs = np.full((n_neurons, n_folds, n_reg), np.nan)
    corrs = np.full((n_neurons, n_folds), np.nan)
    corrs_nonzero = np.full((n_neurons, n_folds), np.nan)
    # 360-bin actual/predicted tuning curves per held-out fold (for rate-map plots)
    cv_actual_tuning = np.full((n_neurons, n_folds, config.total_bins), np.nan, dtype=np.float32)
    cv_predicted_tuning = np.full((n_neurons, n_folds, config.total_bins), np.nan, dtype=np.float32)
    cv_tuning_correlations = np.full((n_neurons, n_folds), np.nan)

    lag0_idx = np.arange(0, n_reg, num_lags)
    phase_norm = _phase_label_per_norm_bin(config)
    nbps = config.num_bins_per_state
    nstates = config.num_task_states

    # state-tuning: a neuron is tuned if it passes in ANY used session
    tuned_any = np.zeros(n_neurons, dtype=bool)
    for s in used:
        tuned_any |= identify_state_tuned_neurons_raw(
            preps[s]['neuron_raw'], preps[s]['trial_times'], config)

    for fold, test_s in enumerate(used):
        train = [s for s in used if s != test_s]
        Xtr = np.vstack([preps[s]['regressors'] for s in train])
        Ltr = np.hstack([preps[s]['locs'] for s in train])
        Ptr = np.hstack([preps[s]['phases'] for s in train])
        Ntr = np.vstack([preps[s]['neuron'] for s in train])      # (sumL, n_neurons)

        keep = ~np.isnan(Ltr)
        Xtr, Ptr, Ntr = Xtr[keep], Ptr[keep], Ntr[keep]

        test = preps[test_s]
        test_reg = test['regressors']
        test_tt = test['trial_times']

        for ni in range(n_neurons):
            y_all = Ntr[:, ni]
            # preferred phase from TRAINING data (mean firing per phase)
            phase_means = [np.nanmean(y_all[Ptr == p]) if np.any(Ptr == p) else -np.inf
                           for p in range(config.num_goal_progress_bins)]
            pref = int(np.argmax(phase_means))
            pm = Ptr == pref
            if np.sum(pm) < 10:
                continue

            coeffs = fit_regression_v3(Xtr[pm], y_all[pm], config)
            cv_coeffs[ni, fold] = coeffs
            if np.all(np.isnan(coeffs)):
                continue

            actual_norm = raw_to_norm(test['neuron'][:, ni], test_tt, config)
            pred_norm = raw_to_norm(test_reg @ coeffs, test_tt, config)
            if actual_norm is None or pred_norm is None:
                continue

            # retain the 360-bin curves for rate-map plots + a full-curve correlation
            cv_actual_tuning[ni, fold] = actual_norm
            cv_predicted_tuning[ni, fold] = pred_norm
            vbins = ~np.isnan(actual_norm) & ~np.isnan(pred_norm)
            if np.sum(vbins) > 10 and np.std(actual_norm[vbins]) > 0 and np.std(pred_norm[vbins]) > 0:
                cv_tuning_correlations[ni, fold] = stats.pearsonr(
                    actual_norm[vbins], pred_norm[vbins])[0]

            corrs[ni, fold] = _state_pref_corr(actual_norm, pred_norm, phase_norm, pref,
                                               nbps, nstates)

            coeffs_nz = coeffs.copy()
            coeffs_nz[lag0_idx] = 0
            pred_norm_nz = raw_to_norm(test_reg @ coeffs_nz, test_tt, config)
            if pred_norm_nz is not None:
                corrs_nonzero[ni, fold] = _state_pref_corr(actual_norm, pred_norm_nz,
                                                           phase_norm, pref, nbps, nstates)
        if verbose:
            print(f"  fold {fold + 1}/{n_folds} (test session {test_s}) done")

    return {
        'mouse_recday': mouse_recday,
        'used_sessions': used,
        'cv_coeffs': cv_coeffs,
        'corrs': corrs,
        'corrs_nonzero': corrs_nonzero,
        'mean_corrs': np.nanmean(corrs, axis=1),
        'mean_corrs_nonzero': np.nanmean(corrs_nonzero, axis=1),
        'state_tuned_mask': tuned_any,
        # full 360-bin tuning curves + v2-compatible keys so plot_polar_tuning_curves /
        # plot_example_betas work directly on v3 results
        'cv_actual_tuning': cv_actual_tuning,
        'cv_predicted_tuning': cv_predicted_tuning,
        'cv_tuning_correlations': cv_tuning_correlations,
        'mean_tuning_correlations': np.nanmean(cv_tuning_correlations, axis=1),
        'valid_sessions': used,
        'num_sessions': n_folds,
        'config': config,
    }


def _state_pref_corr(actual_norm, pred_norm, phase_norm, pref, nbps, nstates):
    """Pearson r over the n=nstates per-state means restricted to preferred-phase bins."""
    a = np.full(nstates, np.nan)
    p = np.full(nstates, np.nan)
    for s in range(nstates):
        sl = slice(s * nbps, (s + 1) * nbps)
        mask = phase_norm[sl] == pref
        a[s] = np.nanmean(actual_norm[sl][mask])
        p[s] = np.nanmean(pred_norm[sl][mask])
    valid = ~np.isnan(a) & ~np.isnan(p)
    if np.sum(valid) >= 3 and np.std(a[valid]) > 0 and np.std(p[valid]) > 0:
        return stats.pearsonr(a[valid], p[valid])[0]
    return np.nan


# ============================================================================
# Non-zero-lag neuron criterion (identical to v2 / El-Gaby)
# ============================================================================

def identify_nonzero_lag_neurons(results, config):
    """Top-3 mean coefficients all at intermediate lags (2 .. num_lags-2)."""
    mean_coeffs = np.nanmean(results['cv_coeffs'], axis=1)
    n_neurons = mean_coeffs.shape[0]
    num_anchors = config.num_locations * config.num_goal_progress_bins
    peak_lags = np.full(n_neurons, np.nan)
    mask = np.zeros(n_neurons, dtype=bool)
    for ni in range(n_neurons):
        c = mean_coeffs[ni]
        if np.all(np.isnan(c)) or np.all(c == 0):
            continue
        top3 = np.argsort(c.reshape(num_anchors, config.num_lags).flatten())[-3:]
        top3_lags = top3 % config.num_lags
        peak_lags[ni] = top3_lags[-1]
        mask[ni] = np.all((top3_lags >= 2) & (top3_lags <= config.num_lags - 2))
    return mask, peak_lags


# ============================================================================
# Plots (ported verbatim from elasticnet_regression_v2 — generic over results/config)
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

        ax.set_xlabel(f'Lag (0=current → {config.num_lags - 1}=oldest)', fontsize=10)
        ax.set_ylabel('Anchor', fontsize=10)
        ax.set_title(f'Neuron {neuron_idx}\nr = {neuron_corr:.3f}', fontsize=11)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        yticks = [i * config.num_goal_progress_bins + 1 for i in range(config.num_locations)]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'L{i+1}' for i in range(config.num_locations)], fontsize=8)

        ax.set_xticks([0, 3, 6, 9, config.num_lags - 1])
        ax.set_xticklabels(['0', '3', '6', '9', str(config.num_lags - 1)], fontsize=8)

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


# ============================================================================
# Cross-mouse summary + plots
# ============================================================================

def run_and_summarise_all_mice_v3(data_dic, config, valid_sessions_dic=None, save_dir=None,
                                  num_examples=6, plot_smooth_sigma=10, verbose=True):
    """Run V3 for every mouse_recday; pool the preferred-phase state correlations for
    state-tuned non-zero-lag neurons and save summary figures."""
    import matplotlib.pyplot as plt
    import os
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    all_results = {}
    pooled = {}            # mouse -> corrs for state-tuned NZ-lag neurons
    for mr in list(data_dic.keys()):
        if verbose:
            print(f"\n{'=' * 60}\nProcessing {mr}\n{'=' * 60}")
        vs = None if valid_sessions_dic is None else valid_sessions_dic.get(mr)
        try:
            res = run_cross_validated_regression_v3(data_dic, mr, config,
                                                    valid_sessions=vs, verbose=verbose)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        if res is None:
            continue
        all_results[mr] = res
        nz_mask, _ = identify_nonzero_lag_neurons(res, config)
        sel = nz_mask & res['state_tuned_mask'] & ~np.isnan(res['mean_corrs'])
        pooled[mr] = res['mean_corrs'][sel]
        if verbose:
            print(f"  state-tuned: {int(res['state_tuned_mask'].sum())}, "
                  f"NZ-lag: {int(nz_mask.sum())}, both (valid r): {int(sel.sum())}")
        if save_dir is not None and sel.sum() > 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(pooled[mr], bins=20, color='darkorange', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='k', ls='--')
            ax.axvline(np.nanmean(pooled[mr]), color='red', lw=2,
                       label=f"mean={np.nanmean(pooled[mr]):.3f}")
            ax.set_xlabel('pref-phase state corr (n=4)')
            ax.set_ylabel('# neurons')
            ax.set_title(f'{mr}\nstate-tuned NZ-lag (v3)')
            ax.legend()
            fig.savefig(os.path.join(save_dir, f'{mr}_v3_corr.svg'), bbox_inches='tight')
            plt.close(fig)

        # Beta matrices + actual-vs-predicted polar rate maps for the top non-zero-lag
        # neurons (ranked by 360-bin tuning correlation), mirroring v2's summary outputs.
        if save_dir is not None:
            mtc = res['mean_tuning_correlations']
            nz_valid = nz_mask & ~np.isnan(mtc)
            top_idx = np.where(nz_valid)[0]
            top_idx = top_idx[np.argsort(mtc[top_idx])[::-1]][:num_examples]
            if len(top_idx) > 0:
                try:
                    figp, _ = plot_polar_tuning_curves(
                        res, config, mouse_recday=mr, num_examples=num_examples,
                        sort_by='correlation', nonzero_lag_only=True,
                        plot_smooth_sigma=plot_smooth_sigma)
                    if figp is not None:
                        figp.savefig(os.path.join(save_dir, f'{mr}_v3_polar_nz.svg'),
                                     bbox_inches='tight')
                        plt.close(figp)
                except Exception as e:
                    print(f"  polar plot failed for {mr}: {e}")
                try:
                    figb = plot_example_betas(res, config, neuron_indices=top_idx)
                    if figb is not None:
                        figb.savefig(os.path.join(save_dir, f'{mr}_v3_betas_nz.svg'),
                                     bbox_inches='tight')
                        plt.close(figb)
                except Exception as e:
                    print(f"  betas plot failed for {mr}: {e}")

    if not pooled:
        print("No results to summarise.")
        return all_results, pooled

    allvals = np.concatenate([v for v in pooled.values() if len(v)])
    allvals = allvals[~np.isnan(allvals)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(allvals, bins=40, color='darkorange', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='k', ls='--')
    mean_all = np.mean(allvals) if len(allvals) else np.nan
    ax.axvline(mean_all, color='red', lw=2, label=f"mean={mean_all:.3f}")
    if len(allvals) > 1:
        t, p = stats.ttest_1samp(allvals, 0)
        ax.text(0.97, 0.97, f"n={len(allvals)}\nt={t:.2f}\np={p:.2e}",
                transform=ax.transAxes, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('pref-phase state corr (n=4)')
    ax.set_ylabel('# neurons')
    ax.set_title('Pooled — state-tuned NZ-lag (v3)')
    ax.legend()

    ax = axes[1]
    labels = list(pooled.keys())
    means = [np.nanmean(v) if len(v) else np.nan for v in pooled.values()]
    ns = [int(np.sum(~np.isnan(v))) for v in pooled.values()]
    x = np.arange(len(labels))
    ax.bar(x, means, color='darkorange', alpha=0.7, edgecolor='black')
    ax.axhline(0, color='k', ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], rotation=30, ha='right',
                       fontsize=7)
    ax.set_ylabel('mean corr')
    ax.set_title('Per-mouse mean (v3)')
    plt.suptitle(f"V3 raw-time {'Poisson' if config.use_poisson else 'ElasticNet'} "
                 f"anchoring — cross-mouse", fontweight='bold')
    plt.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'cross_mouse_v3_summary.svg'), bbox_inches='tight')
    plt.show()
    return all_results, pooled
