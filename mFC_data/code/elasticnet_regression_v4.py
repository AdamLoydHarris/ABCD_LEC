"""
Anchoring regression — V4.

Seeded from `elasticnet_regression_v3.py` (itself a raw-time reimplementation of El-Gaby et
al. 2024 `code/Figure5_Regression.ipynb`). V3 is left untouched so the two can be diffed.

What V4 adds
------------
1. **A future lag direction.** V3/the reference are retrospective only: lag *k* means "this
   (location, phase) anchor was visited *k* phase-steps ago". `lag_direction='future'` builds
   the prospective mirror ("I will be at this anchor in *k* phase-steps") by reversing the
   location/phase sequences, running the *identical* bump loop, and reversing back — so both
   directions carry the same roll/wipe idiosyncrasies. Same trick as
   `elasticnet_regression_v2.generate_regressors_from_norm(lag_direction='future')`.

2. **Per-neuron exports.** `export_regression_outputs` writes one `.npz` per recday holding
   the fitted betas, the 360-bin actual/predicted tuning curves, the n=4 preferred-phase state
   vectors that the headline metric is actually computed from, the per-fold preferred phases,
   and the masks. `plot_neuron_pages` renders every neuron (or just the non-zero-lag ones) as
   a paged PDF.

Structural facts the exports and plots are built around
-------------------------------------------------------
Goal-progress phase advances as a strict 0->1->2 cycle (0 of 39,732 phase transitions across
the dataset deviate from +1 mod 3; segment durations, which range 1-10,285 bins, are
irrelevant). So the anchor phase at lag *k* is *determined* by the current phase:

    ap == (phase - k) % 3          ['past';  (phase + k) % 3 for 'future']

Fits are restricted to preferred-phase rows (both X and y, matching the reference's cell 21),
so only `num_locations * num_lags` = 108 of the 324 columns can ever be non-zero in a fit, and
the fitted betas lie on a single diagonal stripe. Two consequences:

  * the 27x12 beta image is 2/3 structural zeros -- `_collapse_betas` folds it to 9x12;
  * the predicted trace is *exactly* zero at every non-preferred-phase bin, so 240 of the 360
    normalised bins of any "predicted tuning curve" are zero. `cv_tuning_correlations` (the
    full-360 Pearson) is inflated by that; `cv_tuning_correlations_pref` restricts to
    preferred-phase bins.

Preferred phase is recomputed per fold, and ~36% of neurons change it between folds. Averaging
`cv_coeffs` over folds therefore superimposes two different stripes for those neurons, which is
why the non-zero-lag criterion is applied **per fold** (`nz_per_fold=True`, as the reference
does) and why the beta panels average only over folds sharing a preferred phase.

Mirroring
---------
This file is kept **byte-identical** between `code/` (LEC) and `mFC_data/code/` (PFC), the way
`elasticnet_regression_v3.py`, `ccgp_state_pairs.py`, `taskphase_periodicity.py` and
`persistent_homology_analysis.py` already are. Edit one, copy to the other, and check with

    diff code/elasticnet_regression_v4.py mFC_data/code/elasticnet_regression_v4.py

Anything dataset-specific belongs in the notebook or the loader (`build_data_dic_from_pfc` in
`mFC_data/code/glm_analysis_v2.py`), not here. The one asymmetry the module itself carries is
that anatomy is optional: PFC has no `unit_regions`, so `build_unit_table` falls back to
grouping by `mouse`.

Required fields in data_dic[mouse_recday][session]:
    'Neuron_raw'  (n_neurons, n_bins)   raw spike counts per 25 ms bin
    'Locs_raw'    (n_bins,)             node 1..21; 0 = untracked, 10..21 = edges
    'Trial_times' (n_trials, n_states+1) state-boundary times in 25 ms BIN indices

Regressor layout: (num_locations x num_goal_progress_bins x num_lags) = 9 x 3 x 12 = 324,
flattened C-order so lag is the fastest axis.
"""

import copy
import os
import time
import warnings

import numpy as np
from scipy import stats
from scipy.stats import binned_statistic
from sklearn.linear_model import ElasticNet, LinearRegression, PoissonRegressor

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class RegressionConfigV4:
    """Config for the raw-time anchoring regression.

    Estimator branches (unchanged from v3):
        use_poisson=True                      -> PoissonRegressor(alpha=poisson_alpha)
        use_poisson=False & regularize=True   -> ElasticNet(elasticnet_alpha, l1_ratio, positive)
        use_poisson=False & regularize=False  -> LinearRegression(positive)

    New in v4:
        lag_direction        'past' (reference) | 'future' (prospective mirror).
        restrict_to_pref_phase
                             El-Gaby's `use_prefphase`. True (default) fits each neuron only on
                             bins of its preferred goal-progress phase.
        pref_phase_source    'train' (default; no leakage) | 'test' (the reference, whose cell 21
                             reads `tuning_phase_boolean_max[ses_ind_actual]` -- the HELD-OUT
                             session -- for both the fit and the scoring).
        drop_untracked_bins  True drops bins where `Locs_raw == 0`. `build_data_dic.locs_to_int`
                             maps SLEAP nan to integer 0, and v3 only NaNs codes > 9, so those
                             ~4% of bins survived the `~isnan(loc)` filter as training rows.
        alpha_mode           'fixed' (default) uses `elasticnet_alpha` verbatim; 'relative' uses
                             `alpha_frac * alpha_max(neuron)`. ElasticNet branch only -- alpha_max
                             is an L1-path quantity with no meaning for Poisson's L2 penalty.
                             NB fitting on rate rather than counts needs no flag: it is exactly
                             alpha_mode='fixed', elasticnet_alpha=0.00025.
        state_reduce         'mean' (default, matches the reference's `Actual_norm_means`) or
                             'max', for the n=4 per-state reduction. Both are always computed
                             and stored, so this only picks which one `corrs` reports.
        state_tuning_statistic
                             per-state reduction inside the state-tuning filter. 'max' is the
                             reference's, and it is CONFOUNDED BY LEG DURATION: `raw_to_norm`
                             averages more raw bins into each normalised bin when a state
                             interval is longer, which lowers its variance and so its max. On
                             constant-rate Poisson cells the false-positive rate goes 5% ->
                             73-95% -> 100% as the longest:shortest state-duration ratio goes
                             1x -> 2x -> 3x (the 2x figure is rate-dependent: 73% at 2 Hz, 95%
                             at 8 Hz), and the real data's median ratio is 2.26x. 'mean'
                             is duration-invariant and stays calibrated. Default stays 'max'
                             to match the reference -- change it deliberately.
        nonzero_lag_zero_lags
                             lags zeroed when building the `corrs_nonzero` prediction.
        nonzero_lag_min/max  the intermediate-lag window for the non-zero-lag criterion.
        require_positive_top3
                             take the top-3 among *strictly positive* betas. Without it,
                             `np.argsort(c)[-3:]` on a mostly-zero ElasticNet solution picks
                             arbitrary tied zeros: one positive beta at lag 5 gives lags
                             [7,6,5] (passes) under quicksort but [10,11,5] (fails) under
                             mergesort.
        nz_per_fold          apply the criterion to each fold's own betas (one coordinate frame
                             each) and combine by majority vote, as the reference does, rather
                             than to the fold-average.

    Set lag_direction='past', pref_phase_source='train', drop_untracked_bins=False,
    nonzero_lag_zero_lags=(0,), require_positive_top3=False, nz_per_fold=False to reproduce v3.
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
        # --- v4 ---
        lag_direction='past',
        restrict_to_pref_phase=True,
        pref_phase_source='train',
        drop_untracked_bins=True,
        alpha_mode='fixed',
        alpha_frac=0.1,
        state_reduce='mean',
        state_tuning_statistic='max',
        nonzero_lag_zero_lags=(11, 0, 1),
        nonzero_lag_min=2,
        nonzero_lag_max=None,
        require_positive_top3=True,
        nz_per_fold=True,
    ):
        if lag_direction not in ('past', 'future'):
            raise ValueError(f"lag_direction must be 'past' or 'future', got {lag_direction!r}")
        if pref_phase_source not in ('train', 'test'):
            raise ValueError(f"pref_phase_source must be 'train' or 'test', got {pref_phase_source!r}")
        if alpha_mode not in ('fixed', 'relative'):
            raise ValueError(f"alpha_mode must be 'fixed' or 'relative', got {alpha_mode!r}")
        if state_reduce not in ('mean', 'max'):
            raise ValueError(f"state_reduce must be 'mean' or 'max', got {state_reduce!r}")

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

        self.lag_direction = lag_direction
        self.restrict_to_pref_phase = restrict_to_pref_phase
        self.pref_phase_source = pref_phase_source
        self.drop_untracked_bins = drop_untracked_bins
        self.alpha_mode = alpha_mode
        self.alpha_frac = alpha_frac
        self.state_reduce = state_reduce
        if state_tuning_statistic not in ('max', 'mean'):
            raise ValueError("state_tuning_statistic must be 'max' or 'mean', got "
                             f'{state_tuning_statistic!r}')
        self.state_tuning_statistic = state_tuning_statistic
        self.nonzero_lag_zero_lags = tuple(int(k) % num_lags for k in nonzero_lag_zero_lags)
        self.nonzero_lag_min = nonzero_lag_min
        self.nonzero_lag_max = num_lags - 2 if nonzero_lag_max is None else nonzero_lag_max
        self.require_positive_top3 = require_positive_top3
        self.nz_per_fold = nz_per_fold

        self.total_bins = num_bins_per_state * num_task_states           # 360
        self.num_regressors = num_locations * num_goal_progress_bins * num_lags  # 324
        self.bins_per_phase = num_bins_per_state // num_goal_progress_bins       # 30

    def to_dict(self):
        """Plain dict of the settings, for saving alongside the arrays."""
        return {k: v for k, v in vars(self).items()}

    def __repr__(self):
        est = ('Poisson' if self.use_poisson
               else ('ElasticNet' if self.regularize else 'LinearPositive'))
        return (f"RegressionConfigV4({est}, lag_direction={self.lag_direction!r}, "
                f"pref_phase_source={self.pref_phase_source!r}, "
                f"alpha_mode={self.alpha_mode!r}, state_reduce={self.state_reduce!r})")


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
# Raw-time lagged regressors
# ============================================================================

def _bump_loop(locs, phases, config, multiple_bumps=True):
    """The reference bump/roll/wipe loop, run forwards over whatever sequence it is given.

    On a phase change the matching (location, phase) anchor is seeded at lag 0, all anchors
    roll +1 along the lag axis, then lag-1 is set to 1 for the matching anchor / 0 otherwise;
    on a within-phase location change the matching anchor is seeded at lag 1. The final
    roll(-1) undoes the +1 book-keeping so a freshly visited anchor sits at lag 0.
    """
    num_locs = config.num_locations
    num_phases = config.num_goal_progress_bins
    num_lags = config.num_lags
    T = len(locs)

    # node index 0..num_locs-1, or -1 for invalid (NaN / edge / untracked / out of range)
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
    return out


def generate_regressors_raw(locs_raw, phases_raw, config, multiple_bumps=True,
                            lag_direction=None):
    """Build (T, 324) lagged anchoring regressors from raw integer locations and per-bin
    goal-progress phase. Bump state is continuous over the whole (per-session) sequence.

    lag_direction ('past' | 'future', default from config):
      'past'   -- lag-k at bin t marks a (loc, phase) visit k phase-transitions in the PAST
                  (retrospective; the reference's only mode).
      'future' -- lag-k marks a visit k phase-transitions in the FUTURE (prospective).
                  Implemented by reversing locations and phases, running the identical bump
                  loop, then reversing the time axis back, so both directions inherit the same
                  seeding/roll/wipe behaviour.

    Note the two directions do not share lag 0: forwards it accumulates the locations visited
    so far in the current phase-segment, backwards the ones still to come (measured agreement
    0.62). Everything at lag >= 1 refers to complete segments and is directly comparable.
    """
    if lag_direction is None:
        lag_direction = config.lag_direction
    if lag_direction not in ('past', 'future'):
        raise ValueError(f"lag_direction must be 'past' or 'future', got {lag_direction!r}")

    locs = np.asarray(locs_raw, dtype=float)
    phases = np.asarray(phases_raw, dtype=int)
    T = len(locs)

    if lag_direction == 'future':
        out = _bump_loop(locs[::-1].copy(), phases[::-1].copy(), config, multiple_bumps)
        out = out[::-1].copy()
    else:
        out = _bump_loop(locs, phases, config, multiple_bumps)

    return out.reshape(T, config.num_regressors)


def expected_anchor_phase(pref_phase, lag, config):
    """The one anchor phase that can be live at `lag` for a neuron whose preferred phase is
    `pref_phase`. Past: (pref - lag) % 3. Future: (pref + lag) % 3."""
    n = config.num_goal_progress_bins
    if config.lag_direction == 'future':
        return (np.asarray(pref_phase) + np.asarray(lag)) % n
    return (np.asarray(pref_phase) - np.asarray(lag)) % n


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
    """Phase (0..num_phases-1) of each of the 360 normalized bins (equal thirds per state).

    These boundaries coincide exactly with the raw phase boundaries: `raw_to_norm` splits each
    state interval into 90 equal index bins and `compute_phase_state_raw` splits the same
    interval into equal thirds, so norm bin j has phase floor(j/30) exactly.
    """
    nbps = config.num_bins_per_state
    per_state = np.repeat(np.arange(config.num_goal_progress_bins),
                          nbps // config.num_goal_progress_bins)
    if len(per_state) < nbps:  # pad if not divisible
        per_state = np.append(per_state,
                              np.full(nbps - len(per_state), config.num_goal_progress_bins - 1))
    return np.tile(per_state, config.num_task_states)


# ============================================================================
# State-tuned neuron filter (El-Gaby z-score-and-t-test, computed from raw)
# ============================================================================

def identify_state_tuned_neurons_raw(neuron_raw, trial_times, config, p_threshold=None,
                                     return_pref=False):
    """Boolean mask of state-tuned neurons. Per neuron: per-state summary per trial ->
    z-score across states within trial -> mean across trials -> preferred state = argmax ->
    one-sample t-test of that state's per-trial z against 0.

    WARNING, and it is not a small one: with `config.state_tuning_statistic='max'` (the
    reference's choice, and the default here) this test is confounded by leg duration.
    `raw_to_norm` warps each state interval onto 90 bins by averaging, so a longer interval
    puts more raw bins into each normalised bin, lowering its variance and therefore its max.
    Constant-rate Poisson cells then acquire a "preferred state" -- the shortest leg -- and the
    t-test, which selects the argmax and tests it on the same data, passes them. Measured
    false-positive rate on pure noise: 5% at a 1x duration ratio, 73-95% at 2x, ~100% at 3x. The
    median within-session longest:shortest ratio in this dataset is 2.26x (p90 5.4x), and 47%
    of REAL units prefer the shortest leg against a chance of 25%.
    `state_tuning_statistic='mean'` is duration-invariant and stays near nominal.
    """
    if p_threshold is None:
        p_threshold = config.state_tuning_p_threshold
    reducer = np.nanmax if config.state_tuning_statistic == 'max' else np.nanmean
    n_neurons = neuron_raw.shape[0]
    nstates = config.num_task_states
    nbps = config.num_bins_per_state
    is_tuned = np.zeros(n_neurons, dtype=bool)
    pref_states = np.full(n_neurons, -1, dtype=int)

    for ni in range(n_neurons):
        per_trial = raw_to_norm(neuron_raw[ni], trial_times, config, return_mean=False)
        if per_trial is None or per_trial.shape[0] < 3:
            continue
        n_trials = per_trial.shape[0]
        peak = np.full((n_trials, nstates), np.nan)
        for s in range(nstates):
            peak[:, s] = reducer(per_trial[:, s * nbps:(s + 1) * nbps], axis=1)
        row_mean = np.nanmean(peak, axis=1, keepdims=True)
        row_std = np.nanstd(peak, axis=1, keepdims=True)
        row_std[row_std == 0] = np.nan
        z = (peak - row_mean) / row_std
        pref = int(np.nanargmax(np.nanmean(z, axis=0)))
        pref_states[ni] = pref
        zp = z[:, pref]
        zp = zp[~np.isnan(zp)]
        if len(zp) < 3:
            continue
        _, pval = stats.ttest_1samp(zp, 0)
        if pval < p_threshold:
            is_tuned[ni] = True
    return (is_tuned, pref_states) if return_pref else is_tuned


# ============================================================================
# Estimator
# ============================================================================

def elasticnet_alpha_max(X, y, l1_ratio):
    """Smallest ElasticNet alpha that drives every coefficient to zero, for this (X, y).

    Used by alpha_mode='relative' so each neuron sits at the same point on its own
    regularization path. On raw 25 ms counts the median neuron's alpha_max is ~0.007, i.e.
    the paper's fixed alpha=0.01 sits above the whole path for most neurons.
    """
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    denom = len(y) * l1_ratio
    if denom <= 0:
        return np.inf
    return float(np.abs(Xc.T @ yc).max() / denom)


def fit_regression_v4(X, y, config):
    """Fit the chosen estimator, dropping rows with NaNs. Returns the coefficient vector."""
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    Xv, yv = X[valid], y[valid]
    if len(yv) < 10 or np.all(yv == yv[0]):
        return np.full(X.shape[1], np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if config.use_poisson:
            model = PoissonRegressor(alpha=config.poisson_alpha, max_iter=config.max_iter)
        elif config.regularize:
            alpha = config.elasticnet_alpha
            if config.alpha_mode == 'relative':
                alpha = config.alpha_frac * elasticnet_alpha_max(Xv, yv, config.l1_ratio)
                if not np.isfinite(alpha) or alpha <= 0:
                    alpha = config.elasticnet_alpha
            model = ElasticNet(alpha=alpha, l1_ratio=config.l1_ratio,
                               positive=config.positive, max_iter=config.max_iter)
        else:
            model = LinearRegression(positive=config.positive)
        try:
            model.fit(Xv, yv)
        except Exception:
            return np.full(X.shape[1], np.nan)
    return model.coef_


# ============================================================================
# Per-session preparation
# ============================================================================

def _prepare_session(session_data, config):
    """Build raw regressors / aligned locations / phases / neuron matrix for one session.
    Returns None if required fields are missing or no complete trial."""
    for key in ('Neuron_raw', 'Locs_raw', 'Trial_times'):
        if key not in session_data or session_data[key] is None:
            return None
    neuron_raw = np.asarray(session_data['Neuron_raw'], dtype=float)   # (n_neurons, n_bins)
    locs_raw = np.asarray(session_data['Locs_raw'], dtype=float)       # (n_bins,)
    tt = np.asarray(session_data['Trial_times'], dtype=int)            # (n_trials, n_states+1)
    if neuron_raw.ndim != 2 or tt.ndim != 2 or tt.shape[0] < 2:
        return None

    phase_per_bin, state_per_bin = compute_phase_state_raw(
        tt, num_phases=config.num_goal_progress_bins, num_states=config.num_task_states)

    L = min(len(locs_raw), neuron_raw.shape[1], len(phase_per_bin))
    if L < config.num_bins_per_state:
        return None
    locs = locs_raw[:L].copy()
    n_untracked = int(np.sum(locs < 1))
    locs[locs > config.num_locations] = np.nan          # drop edges (10..21)
    if config.drop_untracked_bins:
        # `build_data_dic.locs_to_int` maps SLEAP nan -> 0, which v3 left in the design as a
        # valid row because it only NaN'd codes > num_locations.
        locs[locs < 1] = np.nan
    phases = phase_per_bin[:L]
    neuron = neuron_raw[:, :L].T                          # (L, n_neurons)

    regressors = generate_regressors_raw(locs, phases, config)
    return {
        'regressors': regressors,        # (L, 324)
        'locs': locs,                    # (L,)
        'phases': phases,                # (L,)
        'states': state_per_bin[:L],     # (L,)
        'neuron': neuron,                # (L, n_neurons)
        'neuron_raw': neuron_raw,        # (n_neurons, n_bins)  (for state tuning)
        'trial_times': tt,
        'n_neurons': neuron_raw.shape[0],
        'frac_untracked': n_untracked / max(L, 1),
        'state_durations': np.diff(tt, axis=1).mean(axis=0),   # mean bins per state (A..D)
    }


# ============================================================================
# The n=4 preferred-phase state readout
# ============================================================================

def _state_vectors(curve, phase_norm, pref, nbps, nstates):
    """Reduce a 360-bin curve to n=nstates values over that state's preferred-phase bins,
    both by mean and by max. Returns (mean_vec, max_vec)."""
    m = np.full(nstates, np.nan)
    x = np.full(nstates, np.nan)
    for s in range(nstates):
        sl = slice(s * nbps, (s + 1) * nbps)
        vals = curve[sl][phase_norm[sl] == pref]
        vals = vals[~np.isnan(vals)]
        if vals.size:
            m[s] = vals.mean()
            x[s] = vals.max()
    return m, x


def _corr4(a, p):
    """Pearson r over paired n=4 state vectors, NaN unless both vary over >=3 shared states."""
    valid = ~np.isnan(a) & ~np.isnan(p)
    if np.sum(valid) >= 3 and np.std(a[valid]) > 0 and np.std(p[valid]) > 0:
        return float(stats.pearsonr(a[valid], p[valid])[0])
    return np.nan


def _state_pref_corr(actual_norm, pred_norm, phase_norm, pref, nbps, nstates):
    """The headline metric. Returns (r_mean, r_max, a_mean, p_mean, a_max, p_max).

    Both reductions are computed in the same pass (8 extra floats) so mean-vs-max can be
    switched after the fact without re-running. `mean` is what the reference's
    `Actual_norm_means` uses; `max` over 30 preferred-phase bins of a step-like prediction is
    a far noisier statistic, and with only n=4 points one outlier bin moves r a lot.
    """
    a_mean, a_max = _state_vectors(actual_norm, phase_norm, pref, nbps, nstates)
    p_mean, p_max = _state_vectors(pred_norm, phase_norm, pref, nbps, nstates)
    return _corr4(a_mean, p_mean), _corr4(a_max, p_max), a_mean, p_mean, a_max, p_max


# ============================================================================
# Non-zero-lag criterion
# ============================================================================

def _nz_from_coeffs(c, config):
    """(passes, peak_lag) for ONE coefficient vector, which lives in a single coordinate frame.

    `require_positive_top3` matters because most ElasticNet betas are exactly 0: the plain
    `np.argsort(c)[-3:]` then reports arbitrary tied-zero indices, and the same neuron passes
    or fails depending on the sort algorithm.
    """
    num_lags = config.num_lags
    lo, hi = config.nonzero_lag_min, config.nonzero_lag_max
    if c is None or np.all(np.isnan(c)):
        return False, np.nan
    c = np.nan_to_num(c, nan=0.0)
    if config.require_positive_top3:
        idx = np.flatnonzero(c > 0)
        if idx.size == 0:
            return False, np.nan
        top = idx[np.argsort(c[idx])[-3:]]
    else:
        if np.all(c == 0):
            return False, np.nan
        top = np.argsort(c)[-3:]
    lags = top % num_lags
    return bool(np.all((lags >= lo) & (lags <= hi))), float(lags[-1])


def identify_nonzero_lag_neurons(results, config, return_votes=False):
    """Neurons whose largest betas all sit at intermediate lags.

    With `nz_per_fold=True` (default, and what the reference does) the test is applied to each
    fold's own beta vector and combined by majority vote. That matters because preferred phase
    is refit per fold and ~36% of neurons change it, so `nanmean(cv_coeffs, axis=1)`
    superimposes betas from two different (anchor-phase, lag) frames.
    """
    coeffs = results['cv_coeffs']                       # (n_neurons, n_folds, n_reg)
    n_neurons, n_folds, _ = coeffs.shape
    mask = np.zeros(n_neurons, dtype=bool)
    peak_lags = np.full(n_neurons, np.nan)
    votes = np.zeros((n_neurons, n_folds), dtype=float)
    votes[:] = np.nan

    if not config.nz_per_fold:
        mean_coeffs = np.nanmean(coeffs, axis=1)
        for ni in range(n_neurons):
            ok, lag = _nz_from_coeffs(mean_coeffs[ni], config)
            mask[ni], peak_lags[ni] = ok, lag
        return (mask, peak_lags, votes) if return_votes else (mask, peak_lags)

    for ni in range(n_neurons):
        passes, lags, strengths = [], [], []
        for fi in range(n_folds):
            c = coeffs[ni, fi]
            # A fold that produced no fit, or an all-zero beta vector (61% of them at the
            # fixed default alpha), carries no evidence either way: it ABSTAINS rather than
            # voting No. Counting those as No votes would make the mask a firing-rate filter
            # dressed up as a consistency requirement.
            if np.all(np.isnan(c)) or not np.any(np.nan_to_num(c, nan=0.0) > 0):
                continue
            ok, lag = _nz_from_coeffs(c, config)
            votes[ni, fi] = float(ok)
            passes.append(ok)
            lags.append(lag)
            strengths.append(np.nanmax(np.nan_to_num(c, nan=-np.inf)))
        if not passes:
            continue
        mask[ni] = np.mean(passes) > 0.5
        # peak lag from the fold with the strongest beta -- deterministic, and it comes from a
        # single coordinate frame rather than a cross-fold average.
        finite = [i for i, s in enumerate(strengths) if np.isfinite(s) and not np.isnan(lags[i])]
        if finite:
            peak_lags[ni] = lags[max(finite, key=lambda i: strengths[i])]
    return (mask, peak_lags, votes) if return_votes else (mask, peak_lags)


# ============================================================================
# Cross-validated regression
# ============================================================================

def _pref_phase_from(neuron_mat, phases, num_phases):
    """argmax over goal-progress phase of mean firing, vectorised over neurons."""
    means = np.full((num_phases, neuron_mat.shape[1]), -np.inf)
    for p in range(num_phases):
        m = phases == p
        if np.any(m):
            means[p] = np.nanmean(neuron_mat[m], axis=0)
    return np.argmax(means, axis=0).astype(int), means


def run_cross_validated_regression_v4(data_dic, mouse_recday, config, valid_sessions=None,
                                      verbose=False):
    """Leave-one-session-out raw-time anchoring regression for one mouse_recday.

    Returns a results dict; None if fewer than 2 usable sessions.
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
    num_phases = config.num_goal_progress_bins
    nbps = config.num_bins_per_state
    nstates = config.num_task_states

    cv_coeffs = np.full((n_neurons, n_folds, n_reg), np.nan)
    corrs = np.full((n_neurons, n_folds), np.nan)
    corrs_nonzero = np.full((n_neurons, n_folds), np.nan)
    corrs_max = np.full((n_neurons, n_folds), np.nan)
    corrs_nonzero_max = np.full((n_neurons, n_folds), np.nan)
    pref_phases = np.full((n_neurons, n_folds), -1, dtype=int)
    n_nonzero_betas = np.full((n_neurons, n_folds), np.nan)

    cv_actual_tuning = np.full((n_neurons, n_folds, config.total_bins), np.nan, dtype=np.float32)
    cv_predicted_tuning = np.full((n_neurons, n_folds, config.total_bins), np.nan, dtype=np.float32)
    cv_predicted_nz_tuning = np.full((n_neurons, n_folds, config.total_bins), np.nan, dtype=np.float32)
    cv_tuning_correlations = np.full((n_neurons, n_folds), np.nan)
    cv_tuning_correlations_pref = np.full((n_neurons, n_folds), np.nan)

    state4 = {k: np.full((n_neurons, n_folds, nstates), np.nan, dtype=np.float32)
              for k in ('actual_mean', 'actual_max', 'predicted_mean', 'predicted_max',
                        'predicted_nz_mean', 'predicted_nz_max')}

    # lag columns zeroed for the `corrs_nonzero` prediction (default {11, 0, 1})
    lag_axis = np.arange(n_reg) % num_lags
    zero_cols = np.isin(lag_axis, np.asarray(config.nonzero_lag_zero_lags))

    phase_norm = _phase_label_per_norm_bin(config)

    # state-tuning: a neuron is tuned if it passes in ANY used session. Deliberately includes
    # the held-out fold so the neuron set is identical across folds; `state_tuning_train_only`
    # in run_and_summarise_all_mice_v4 reports whether that choice moves the headline.
    tuned_any = np.zeros(n_neurons, dtype=bool)
    alt_cfg = copy.copy(config)
    alt_cfg.state_tuning_statistic = 'mean' if config.state_tuning_statistic == 'max' else 'max'
    tuned_alt = np.zeros(n_neurons, dtype=bool)
    tuning_pref_state = np.full((n_neurons, len(used)), -1, dtype=int)
    for si, s in enumerate(used):
        m, pr = identify_state_tuned_neurons_raw(
            preps[s]['neuron_raw'], preps[s]['trial_times'], config, return_pref=True)
        tuned_any |= m
        tuning_pref_state[:, si] = pr
        tuned_alt |= identify_state_tuned_neurons_raw(
            preps[s]['neuron_raw'], preps[s]['trial_times'], alt_cfg)

    # How unequal are the legs? The 'max' state-tuning statistic is confounded by this: on
    # constant-rate cells its false-positive rate is 6% at ratio 1x but ~100% at 3x.
    dur_ratios = [float(preps[s]['state_durations'].max()
                        / max(preps[s]['state_durations'].min(), 1.0)) for s in used]
    # Does the "preferred state" just track the shortest leg? (chance = 1/num_states)
    shortest = np.array([int(np.argmin(preps[s]['state_durations'])) for s in used])
    valid_pref = tuning_pref_state >= 0
    frac_pref_shortest = (float(np.mean((tuning_pref_state == shortest[None, :])[valid_pref]))
                          if valid_pref.any() else np.nan)

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

        # Preferred phase. 'train' avoids leakage; 'test' reproduces the reference, whose cell
        # 21 reads the tuning of the HELD-OUT session for both the fit and the scoring.
        if config.pref_phase_source == 'test':
            tkeep = ~np.isnan(test['locs'])
            prefs, _ = _pref_phase_from(test['neuron'][tkeep], test['phases'][tkeep], num_phases)
        else:
            prefs, _ = _pref_phase_from(Ntr, Ptr, num_phases)
        pref_phases[:, fold] = prefs

        # Hoist the per-phase row slice: only `num_phases` distinct subsets exist, but v3
        # rebuilt one per neuron (~50 s/recday).
        if config.restrict_to_pref_phase:
            rows_by_phase = {p: np.flatnonzero(Ptr == p) for p in range(num_phases)}
            X_by_phase = {p: Xtr[r] for p, r in rows_by_phase.items()}
        else:
            # `None` means "all rows": slicing with an arange would copy the whole design
            # matrix once per neuron for no benefit.
            rows_by_phase = {p: None for p in range(num_phases)}
            X_by_phase = {p: Xtr for p in range(num_phases)}

        for ni in range(n_neurons):
            pref = int(prefs[ni])
            rows = rows_by_phase[pref]
            Xfit = X_by_phase[pref]
            if Xfit.shape[0] < 10:
                continue

            yfit = Ntr[:, ni] if rows is None else Ntr[rows, ni]
            coeffs = fit_regression_v4(Xfit, yfit, config)
            cv_coeffs[ni, fold] = coeffs
            if np.all(np.isnan(coeffs)):
                continue
            n_nonzero_betas[ni, fold] = int(np.sum(np.abs(coeffs) > 1e-9))

            actual_norm = raw_to_norm(test['neuron'][:, ni], test_tt, config)
            pred_norm = raw_to_norm(test_reg @ coeffs, test_tt, config)
            if actual_norm is None or pred_norm is None:
                continue

            cv_actual_tuning[ni, fold] = actual_norm
            cv_predicted_tuning[ni, fold] = pred_norm

            vbins = ~np.isnan(actual_norm) & ~np.isnan(pred_norm)
            if np.sum(vbins) > 10 and np.std(actual_norm[vbins]) > 0 and np.std(pred_norm[vbins]) > 0:
                cv_tuning_correlations[ni, fold] = stats.pearsonr(
                    actual_norm[vbins], pred_norm[vbins])[0]
            # The prediction is exactly 0 off the preferred phase, so the full-360 correlation
            # above largely measures phase tuning. Restrict to preferred-phase bins as well.
            pbins = vbins & (phase_norm == pref)
            if (np.sum(pbins) > 10 and np.std(actual_norm[pbins]) > 0
                    and np.std(pred_norm[pbins]) > 0):
                cv_tuning_correlations_pref[ni, fold] = stats.pearsonr(
                    actual_norm[pbins], pred_norm[pbins])[0]

            r_m, r_x, a_m, p_m, a_x, p_x = _state_pref_corr(
                actual_norm, pred_norm, phase_norm, pref, nbps, nstates)
            corrs[ni, fold] = r_m if config.state_reduce == 'mean' else r_x
            corrs_max[ni, fold] = r_x
            state4['actual_mean'][ni, fold] = a_m
            state4['actual_max'][ni, fold] = a_x
            state4['predicted_mean'][ni, fold] = p_m
            state4['predicted_max'][ni, fold] = p_x

            coeffs_nz = coeffs.copy()
            coeffs_nz[zero_cols] = 0
            pred_norm_nz = raw_to_norm(test_reg @ coeffs_nz, test_tt, config)
            if pred_norm_nz is not None:
                cv_predicted_nz_tuning[ni, fold] = pred_norm_nz
                rz_m, rz_x, _, pz_m, _, pz_x = _state_pref_corr(
                    actual_norm, pred_norm_nz, phase_norm, pref, nbps, nstates)
                corrs_nonzero[ni, fold] = rz_m if config.state_reduce == 'mean' else rz_x
                corrs_nonzero_max[ni, fold] = rz_x
                state4['predicted_nz_mean'][ni, fold] = pz_m
                state4['predicted_nz_max'][ni, fold] = pz_x

        if verbose:
            print(f"  fold {fold + 1}/{n_folds} (test session {test_s}) done")

    results = {
        'mouse_recday': mouse_recday,
        'used_sessions': used,
        'lag_direction': config.lag_direction,
        'cv_coeffs': cv_coeffs,
        'corrs': corrs,
        'corrs_nonzero': corrs_nonzero,
        'corrs_max': corrs_max,
        'corrs_nonzero_max': corrs_nonzero_max,
        'mean_corrs': np.nanmean(corrs, axis=1),
        'mean_corrs_nonzero': np.nanmean(corrs_nonzero, axis=1),
        'mean_corrs_max': np.nanmean(corrs_max, axis=1),
        'pref_phases': pref_phases,
        'n_nonzero_betas': n_nonzero_betas,
        'state_tuned_mask': tuned_any,
        'state_tuned_mask_alt': tuned_alt,          # the other state_tuning_statistic
        'tuning_pref_state': tuning_pref_state,
        'state_duration_ratio': float(np.median(dur_ratios)),
        'frac_pref_state_is_shortest': frac_pref_shortest,
        'cv_actual_tuning': cv_actual_tuning,
        'cv_predicted_tuning': cv_predicted_tuning,
        'cv_predicted_nz_tuning': cv_predicted_nz_tuning,
        'cv_tuning_correlations': cv_tuning_correlations,
        'cv_tuning_correlations_pref': cv_tuning_correlations_pref,
        'mean_tuning_correlations': np.nanmean(cv_tuning_correlations, axis=1),
        'mean_tuning_correlations_pref': np.nanmean(cv_tuning_correlations_pref, axis=1),
        'frac_untracked': float(np.mean([preps[s]['frac_untracked'] for s in used])),
        'valid_sessions': used,
        'num_sessions': n_folds,
        'config': config,
    }
    for k, v in state4.items():
        results[f'cv_{k}_state4'] = v

    nz_mask, peak_lags, votes = identify_nonzero_lag_neurons(results, config, return_votes=True)
    results['nonzero_lag_mask'] = nz_mask
    results['peak_lags'] = peak_lags
    results['nz_fold_votes'] = votes                      # NaN = that fold abstained
    results['n_informative_folds'] = np.sum(~np.isnan(votes), axis=1)
    return results


# ============================================================================
# Beta-matrix helpers
# ============================================================================

def _collapse_betas(coeffs, pref, config):
    """Fold a (324,) beta vector to the (num_locations, num_lags) matrix of live cells.

    Only one anchor phase can be live at each lag -- `expected_anchor_phase(pref, lag)` -- so
    the phase axis carries no independent information and the full 27x12 image is 2/3
    structural zeros. Returns None if `pref` is unknown.
    """
    if pref is None or pref < 0:
        return None
    B = np.asarray(coeffs, dtype=float).reshape(
        config.num_locations, config.num_goal_progress_bins, config.num_lags)
    lags = np.arange(config.num_lags)
    ap = expected_anchor_phase(pref, lags, config)
    return B[:, ap, lags]


def _modal_pref(pref_row, weights=None):
    """Most common preferred phase across folds, and the folds that share it.

    `weights` (e.g. the per-fold non-zero beta count) breaks ties toward the frame whose folds
    actually carry evidence. Without it a 2-2 split can pick the frame whose folds all fit
    all-zero, leaving the beta panel empty while the informative folds are excluded.
    """
    valid = pref_row[pref_row >= 0]
    if valid.size == 0:
        return -1, np.zeros(len(pref_row), dtype=bool)
    vals, counts = np.unique(valid, return_counts=True)
    best = counts.max()
    tied = vals[counts == best]
    if len(tied) > 1 and weights is not None:
        w = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0)
        tied = [max(tied, key=lambda v: w[pref_row == v].sum())]
    modal = int(tied[0])
    return modal, pref_row == modal


def betas_in_common_frame(results, config, neuron_idx):
    """(collapsed 9x12 betas, modal pref phase, n folds used, n folds available).

    Averages `cv_coeffs` only over folds that share the modal preferred phase. Averaging over
    all folds would superimpose two different (anchor phase, lag) stripes for the ~36% of
    neurons whose preferred phase changes between folds.
    """
    pref_row = results['pref_phases'][neuron_idx]
    modal, share = _modal_pref(pref_row, weights=results.get('n_nonzero_betas',
                                                             [None])[neuron_idx]
                               if 'n_nonzero_betas' in results else None)
    if modal < 0:
        return None, -1, 0, len(pref_row)
    coeffs = results['cv_coeffs'][neuron_idx][share]
    if coeffs.size == 0 or np.all(np.isnan(coeffs)):
        return None, modal, 0, len(pref_row)
    return _collapse_betas(np.nanmean(coeffs, axis=0), modal, config), modal, int(share.sum()), len(pref_row)


def fold_betas(results, config, neuron_idx):
    """(n_folds, num_locations, num_lags) betas, each fold collapsed in ITS OWN frame.

    Every fold has its own preferred phase, so each fold's matrix is collapsed with that
    fold's `expected_anchor_phase`. Folds with no fit come back all-NaN. This is the
    un-averaged view of the same thing `betas_in_common_frame` summarises.
    """
    prefs = results['pref_phases'][neuron_idx]
    coeffs = results['cv_coeffs'][neuron_idx]
    out = np.full((len(prefs), config.num_locations, config.num_lags), np.nan)
    for fi, pref in enumerate(prefs):
        if pref < 0 or np.all(np.isnan(coeffs[fi])):
            continue
        out[fi] = _collapse_betas(coeffs[fi], int(pref), config)
    return out


def all_fold_betas(results, config):
    """(n_neurons, n_folds, num_locations, num_lags) -- `fold_betas` for every neuron."""
    n = results['cv_coeffs'].shape[0]
    return np.stack([fold_betas(results, config, ni) for ni in range(n)]).astype(np.float32)


def assert_beta_stripe(results, config, tol=1e-9):
    """Every per-fold beta's support must lie on ap == expected_anchor_phase(pref, lag).

    This is the invariant that the fold-averaged beta plots violated. Tolerance rather than
    `!= 0` because ElasticNet leaves dead columns at exactly 0 but Poisson's lbfgs can leave
    float dust. No-op when `restrict_to_pref_phase=False` (all 324 columns are then live).
    """
    if not config.restrict_to_pref_phase:
        return 0, 0
    coeffs = results['cv_coeffs']
    prefs = results['pref_phases']
    nloc, nph, nlag = config.num_locations, config.num_goal_progress_bins, config.num_lags
    lags = np.arange(nlag)
    checked = bad = 0
    for ni in range(coeffs.shape[0]):
        for fi in range(coeffs.shape[1]):
            c = coeffs[ni, fi]
            if np.all(np.isnan(c)) or prefs[ni, fi] < 0:
                continue
            live = np.abs(np.nan_to_num(c, nan=0.0)).reshape(nloc, nph, nlag) > tol
            allowed = np.zeros((nph, nlag), dtype=bool)
            allowed[expected_anchor_phase(prefs[ni, fi], lags, config), lags] = True
            checked += 1
            if np.any(live & ~allowed[None, :, :]):
                bad += 1
    if bad:
        raise AssertionError(
            f"{results['mouse_recday']}: {bad}/{checked} per-fold beta vectors have support off "
            f"the (pref -/+ lag) mod {nph} stripe -- the phase/lag coupling is broken.")
    return checked, bad


# ============================================================================
# Export
# ============================================================================

_EXPORT_ARRAYS = (
    'cv_coeffs', 'cv_actual_tuning', 'cv_predicted_tuning', 'cv_predicted_nz_tuning',
    'cv_actual_mean_state4', 'cv_actual_max_state4',
    'cv_predicted_mean_state4', 'cv_predicted_max_state4',
    'cv_predicted_nz_mean_state4', 'cv_predicted_nz_max_state4',
    'corrs', 'corrs_nonzero', 'corrs_max', 'corrs_nonzero_max',
    'cv_tuning_correlations', 'cv_tuning_correlations_pref',
    'mean_corrs', 'mean_corrs_nonzero', 'mean_corrs_max',
    'mean_tuning_correlations', 'mean_tuning_correlations_pref',
    'pref_phases', 'n_nonzero_betas', 'nz_fold_votes', 'n_informative_folds',
    'state_tuned_mask', 'state_tuned_mask_alt', 'tuning_pref_state',
    'nonzero_lag_mask', 'peak_lags',
)


def export_regression_outputs(results, config, out_dir, check_stripe=True):
    """Write one `{recday}_{direction}_arrays.npz` holding every per-neuron array.

    Betas and tuning curves are stored as float32; everything else keeps its dtype. The
    stripe invariant is asserted before writing.
    """
    os.makedirs(out_dir, exist_ok=True)
    if check_stripe:
        assert_beta_stripe(results, config)

    payload = {}
    for key in _EXPORT_ARRAYS:
        if key not in results:
            continue
        arr = np.asarray(results[key])
        if arr.dtype == np.float64 and key.startswith(('cv_', 'corrs')):
            arr = arr.astype(np.float32)
        payload[key] = arr
    # the per-fold beta matrices, each already collapsed in its own fold's frame
    payload['cv_betas_collapsed'] = all_fold_betas(results, config)
    payload['used_sessions'] = np.asarray(results['used_sessions'])
    payload['num_sessions'] = np.asarray(results['num_sessions'])
    payload['frac_untracked'] = np.asarray(results['frac_untracked'])
    payload['state_duration_ratio'] = np.asarray(results['state_duration_ratio'])
    payload['frac_pref_state_is_shortest'] = np.asarray(
        results['frac_pref_state_is_shortest'])
    payload['mouse_recday'] = np.asarray(results['mouse_recday'])
    payload['lag_direction'] = np.asarray(results['lag_direction'])
    payload['config'] = np.asarray(config.to_dict(), dtype=object)

    path = os.path.join(out_dir, f"{results['mouse_recday']}_{results['lag_direction']}_arrays.npz")
    np.savez_compressed(path, **payload)
    return path


def estimator_name(config):
    """'poisson' | 'elasticnet' | 'linear' -- the branch `fit_regression_v4` will take."""
    if config.use_poisson:
        return 'poisson'
    return 'elasticnet' if config.regularize else 'linear'


def run_dir_name(config, stamp=None, prefix='', version='v4'):
    """Output directory name for a run: `{estimator}_{version}_{direction}_{stamp}`.

    Leading with the estimator because it is the setting that most changes the numbers -- the
    Poisson branch never zeroes a coefficient while ElasticNet at the default alpha zeroes
    ~60% of neurons -- and because the old hard-coded `elasticnet_v4_*` name was written even
    for Poisson runs. Everything else about the run is in `run_config.json` beside the data.
    """
    import datetime
    if stamp is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{prefix}{estimator_name(config)}_{version}_{config.lag_direction}_{stamp}'


def write_run_manifest(out_dir, config, all_results, valid_sessions_dic=None,
                       extra=None, filename='run_config.json'):
    """Write one human-readable `run_config.json` per run directory.

    The config is also embedded in every `.npz`, but that is not enough on its own: it needs
    Python to read, it does not exist at all if a run produced no npz, and it does not record
    the things that are decided *outside* the config -- which recdays ran, which sessions were
    used as folds (`valid_sessions_dic` plus any El-Gaby hand-exclusion), the library versions,
    or when. Those are exactly what you need to know whether two output folders are comparable.

    Never raises: a run is not worth losing to a metadata write.
    """
    import datetime
    import hashlib
    import json
    import platform
    import subprocess

    def _git(*args):
        try:
            return subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__))] +
                                  list(args), capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    try:
        module_path = os.path.abspath(__file__)
        with open(module_path, 'rb') as f:
            module_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        module_path, module_sha = None, None

    used = {}
    if all_results:
        for recday, res in all_results.items():
            used[recday] = {'used_sessions': [int(x) for x in res.get('used_sessions', [])],
                            'n_neurons': int(len(res['nonzero_lag_mask']))}

    manifest = {
        'written': datetime.datetime.now().isoformat(timespec='seconds'),
        'lag_direction': config.lag_direction,
        'estimator': ('Poisson' if config.use_poisson
                      else ('ElasticNet' if config.regularize else 'LinearPositive')),
        'config': config.to_dict(),
        'sessions_used': used,
        'sessions_requested': ({k: [int(x) for x in v] for k, v in valid_sessions_dic.items()}
                               if valid_sessions_dic else None),
        'excluded_sessions': {k: list(v) for k, v in EL_GABY_EXCLUDED_SESSIONS.items()},
        'module': {'path': module_path, 'sha256_16': module_sha,
                   'git_commit': _git('rev-parse', 'HEAD'),
                   'git_dirty': bool(_git('status', '--porcelain'))},
        'environment': {'python': platform.python_version(), 'host': platform.node()},
    }
    try:
        import numpy as _np
        import scipy as _sp
        import sklearn as _sk
        manifest['environment'].update(numpy=_np.__version__, scipy=_sp.__version__,
                                       sklearn=_sk.__version__)
    except Exception:
        pass
    if extra:
        manifest.update(extra)

    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        return path
    except Exception as exc:
        print(f'  could not write {filename}: {exc}')
        return None


def load_regression_outputs(path):
    """Read back an `export_regression_outputs` npz as a plain dict (config as a dict)."""
    with np.load(path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    for k in ('mouse_recday', 'lag_direction'):
        if k in out:
            out[k] = str(out[k])
    for k in ('num_sessions', 'frac_untracked', 'state_duration_ratio',
              'frac_pref_state_is_shortest'):
        if k in out and getattr(out[k], 'ndim', 1) == 0:
            out[k] = out[k].item()
    if 'used_sessions' in out:
        out['used_sessions'] = list(out['used_sessions'])
    if 'config' in out:
        out['config'] = out['config'].item()
    return out


# ============================================================================
# Plots
# ============================================================================

def _lag_xlabel(config):
    if config.lag_direction == 'future':
        return f'Lag (0 = current -> {config.num_lags - 1} = furthest ahead)'
    return f'Lag (0 = current -> {config.num_lags - 1} = oldest)'


def plot_neuron_pages(results, config, out_pdf, neuron_indices=None, per_page=6,
                      sort_by='tuning_corr', title_extra=''):
    """Paged PDF of every requested neuron: betas, 360-bin curves, and the n=4 state readout.

    Three panels per neuron:
      1. betas collapsed to (location x lag), averaged only over folds sharing the modal
         preferred phase, titled with the per-fold preferred phases so a frame change is
         visible rather than silently superimposed;
      2. the 360-bin actual and predicted curves, UNSMOOTHED, with non-preferred-phase thirds
         shaded -- the prediction is exactly zero there by construction;
      3. the n=4 preferred-phase per-state readout that `corrs` is computed from.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_neurons, n_folds = results['pref_phases'].shape
    if neuron_indices is None:
        neuron_indices = np.arange(n_neurons)
    neuron_indices = np.asarray(neuron_indices, dtype=int)
    if sort_by == 'tuning_corr' and neuron_indices.size:
        key = results['mean_tuning_correlations_pref'][neuron_indices]
        neuron_indices = neuron_indices[np.argsort(np.nan_to_num(key, nan=-np.inf))[::-1]]
    if neuron_indices.size == 0:
        return None

    nbps, nstates = config.num_bins_per_state, config.num_task_states
    phase_norm = _phase_label_per_norm_bin(config)
    bins = np.arange(config.total_bins)
    reduce_key = config.state_reduce

    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)) or '.', exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for page_start in range(0, len(neuron_indices), per_page):
            page = neuron_indices[page_start:page_start + per_page]
            fig, axes = plt.subplots(len(page), 3, figsize=(15, 2.7 * len(page)),
                                     gridspec_kw={'width_ratios': [1.0, 1.9, 0.8]})
            axes = np.atleast_2d(axes)

            for row, ni in enumerate(page):
                ax_b, ax_c, ax_s = axes[row]
                pref_row = results['pref_phases'][ni]
                B, modal, n_used, n_tot = betas_in_common_frame(results, config, ni)

                # --- panel 1: betas, (location x lag) in one coordinate frame
                if B is None or np.all(np.isnan(B)):
                    ax_b.text(0.5, 0.5, 'no fit', ha='center', va='center',
                              transform=ax_b.transAxes, fontsize=9)
                    ax_b.set_xticks([]); ax_b.set_yticks([])
                else:
                    bmax = float(np.nanmax(np.abs(B))) if np.isfinite(B).any() else 0.0
                    im = ax_b.imshow(B, aspect='auto', cmap='hot', interpolation='nearest',
                                     vmin=0, vmax=bmax or 1.0)
                    plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
                    ax_b.set_yticks(np.arange(config.num_locations))
                    ax_b.set_yticklabels([f'L{i + 1}' for i in range(config.num_locations)],
                                         fontsize=6)
                    ax_b.set_xticks(np.arange(0, config.num_lags, 2))
                    ax_b.set_xlabel(_lag_xlabel(config), fontsize=7)
                    for lo in (config.nonzero_lag_min, config.nonzero_lag_max):
                        ax_b.axvline(lo + (-0.5 if lo == config.nonzero_lag_min else 0.5),
                                     color='cyan', lw=0.8, ls='--', alpha=0.7)
                flip = '' if n_used == n_tot else f'  [{n_used}/{n_tot} folds]'
                ax_b.set_title(f'N{ni}  pref phase/fold={pref_row.tolist()}{flip}', fontsize=7)

                # --- panel 2: 360-bin curves, unsmoothed
                actual = np.nanmean(results['cv_actual_tuning'][ni], axis=0)
                pred_folds = results['cv_predicted_tuning'][ni]
                pred = np.nanmean(pred_folds, axis=0)
                _shade_offphase(ax_c, phase_norm, modal)
                for s in range(1, nstates):
                    ax_c.axvline(s * nbps, color='0.4', lw=0.6, ls='--')
                # Predicted goes on its own axis: at the default alpha the betas are shrunk so
                # hard that the prediction is 1-2 orders of magnitude below the firing rate,
                # and on a shared axis it is a flat line at the bottom. Pearson r is
                # scale-invariant, so independent scaling changes nothing about the number --
                # but the title says so, because two y-axes are easy to misread.
                ax_p = ax_c.twinx()
                # Per-fold traces for BOTH: each fold holds out a different task, so the
                # fold-averaged actual is a blend of four different tuning curves. See
                # plot_fold_ratemap_pages for the un-averaged view.
                actual_folds = results['cv_actual_tuning'][ni]
                for fi in range(n_folds):
                    if not np.all(np.isnan(actual_folds[fi])):
                        ax_c.plot(bins, actual_folds[fi], color='tab:blue', lw=0.4, alpha=0.22)
                    if not np.all(np.isnan(pred_folds[fi])):
                        ax_p.plot(bins, pred_folds[fi], color='red', lw=0.4, alpha=0.25)
                l1, = ax_c.plot(bins, actual, color='tab:blue', lw=1.4, label='actual')
                l2, = ax_p.plot(bins, pred, color='red', lw=1.4, label='predicted')
                ax_c.set_xlim(0, config.total_bins)
                ax_c.set_xticks(np.arange(nstates) * nbps + nbps / 2)
                ax_c.set_xticklabels(list('ABCD')[:nstates], fontsize=7)
                ax_c.tick_params(labelsize=6)
                ax_c.tick_params(axis='y', labelcolor='tab:blue')
                ax_p.tick_params(axis='y', labelsize=6, labelcolor='red')
                ax_c.set_ylabel('actual rate', fontsize=7, color='tab:blue')
                ax_p.set_ylabel('predicted', fontsize=7, color='red')
                r_full = results['mean_tuning_correlations'][ni]
                r_pref = results['mean_tuning_correlations_pref'][ni]
                ax_c.set_title(f'360-bin r={r_full:.3f}   pref-phase-only r={r_pref:.3f}'
                               '   (y-axes scaled independently; grey = predicted is 0 '
                               'by construction)', fontsize=7)
                if row == 0:
                    ax_c.legend(handles=[l1, l2], fontsize=6, loc='upper right', ncol=2)

                # --- panel 3: the n=4 readout the headline metric uses
                a4 = results[f'cv_actual_{reduce_key}_state4'][ni]
                p4 = results[f'cv_predicted_{reduce_key}_state4'][ni]
                z4 = results[f'cv_predicted_nz_{reduce_key}_state4'][ni]
                x = np.arange(nstates)
                for arr, col, lab in ((a4, 'tab:blue', 'actual'), (p4, 'red', 'pred'),
                                      (z4, 'darkorange', 'pred (no lag %s)'
                                       % ','.join(map(str, config.nonzero_lag_zero_lags)))):
                    m = np.nanmean(arr, axis=0)
                    sd = np.nanstd(arr, axis=0)
                    if np.all(np.isnan(m)):
                        continue
                    ax_s.errorbar(x, m / (np.nanmax(np.abs(m)) or 1),
                                  yerr=sd / (np.nanmax(np.abs(m)) or 1),
                                  color=col, marker='o', ms=3, lw=1.1, capsize=2, label=lab)
                ax_s.set_xticks(x)
                ax_s.set_xticklabels(list('ABCD')[:nstates], fontsize=7)
                ax_s.tick_params(labelsize=6)
                ax_s.set_title(f"r={results['mean_corrs'][ni]:.3f}  "
                               f"nz r={results['mean_corrs_nonzero'][ni]:.3f}\n"
                               f"NZ-lag={bool(results['nonzero_lag_mask'][ni])} "
                               f"peak lag={results['peak_lags'][ni]:.0f} "
                               f"betas={np.nanmean(results['n_nonzero_betas'][ni]):.1f}",
                               fontsize=7)
                if row == 0:
                    ax_s.legend(fontsize=5, loc='upper right')

            fig.suptitle(f"{results['mouse_recday']} - {results['lag_direction']} lags"
                         f"{title_extra}   (normalised n=4 panel; {reduce_key} reduction)",
                         fontsize=9, y=0.999)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    return out_pdf


def plot_fold_beta_pages(results, config, out_pdf, neuron_indices=None, per_page=6,
                         sort_by='tuning_corr', title_extra=''):
    """Paged PDF of the PER-FOLD beta matrices: one row per neuron, one column per fold, plus
    a final column with the fold-average taken in a common frame.

    Each fold panel is collapsed with that fold's own preferred phase, so the columns are
    directly comparable even when the preferred phase changes between folds -- which it does
    for ~36% of neurons, and which is exactly what the fold-average hides. The per-fold panel
    titles carry the held-out session and that fold's preferred phase.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_neurons, n_folds = results['pref_phases'].shape
    if neuron_indices is None:
        neuron_indices = np.arange(n_neurons)
    neuron_indices = np.asarray(neuron_indices, dtype=int)
    if sort_by == 'tuning_corr' and neuron_indices.size:
        key = results['mean_tuning_correlations_pref'][neuron_indices]
        neuron_indices = neuron_indices[np.argsort(np.nan_to_num(key, nan=-np.inf))[::-1]]
    if neuron_indices.size == 0:
        return None

    sessions = list(results['used_sessions'])
    ncols = n_folds + 1
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)) or '.', exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for page_start in range(0, len(neuron_indices), per_page):
            page = neuron_indices[page_start:page_start + per_page]
            fig, axes = plt.subplots(len(page), ncols,
                                     figsize=(2.05 * ncols, 2.0 * len(page)), squeeze=False)
            for row, ni in enumerate(page):
                per_fold = fold_betas(results, config, ni)
                prefs = results['pref_phases'][ni]
                finite = per_fold[np.isfinite(per_fold)]
                vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
                vmax = vmax or 1.0

                for fi in range(n_folds):
                    ax = axes[row][fi]
                    B = per_fold[fi]
                    if np.all(np.isnan(B)):
                        ax.text(0.5, 0.5, 'no fit', ha='center', va='center',
                                transform=ax.transAxes, fontsize=7)
                        ax.set_xticks([]); ax.set_yticks([])
                    else:
                        ax.imshow(B, aspect='auto', cmap='hot', vmin=0, vmax=vmax,
                                  interpolation='nearest')
                        ax.set_xticks(np.arange(0, config.num_lags, 3))
                        ax.set_yticks([] if fi else np.arange(config.num_locations))
                        if not fi:
                            ax.set_yticklabels([f'L{j + 1}' for j in range(config.num_locations)],
                                               fontsize=5)
                        ax.tick_params(labelsize=5)
                    held = sessions[fi] if fi < len(sessions) else fi
                    nb = results['n_nonzero_betas'][ni, fi]
                    ax.set_title(f'held-out {held} | pref {prefs[fi]}\n'
                                 f'{0 if np.isnan(nb) else int(nb)} betas', fontsize=5.5)

                ax = axes[row][-1]
                B, modal, n_used, n_tot = betas_in_common_frame(results, config, ni)
                if B is None:
                    ax.axis('off')
                else:
                    im = ax.imshow(B, aspect='auto', cmap='hot', vmin=0, vmax=vmax,
                                   interpolation='nearest')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=5)
                    ax.set_xticks(np.arange(0, config.num_lags, 3))
                    ax.set_yticks([])
                    ax.tick_params(labelsize=5)
                ax.set_title(f'N{ni} mean [{n_used}/{n_tot} folds, pref {modal}]\n'
                             f"r={results['mean_corrs'][ni]:.3f} "
                             f"NZ={bool(results['nonzero_lag_mask'][ni])}", fontsize=5.5)

            for ax in axes[-1]:
                ax.set_xlabel(_lag_xlabel(config), fontsize=5)
            fig.suptitle(f"{results['mouse_recday']} - {results['lag_direction']} lags - "
                         f"per-fold betas (location x lag){title_extra}", fontsize=9, y=0.999)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    return out_pdf


def _shade_offphase(ax, phase_norm, pref, color='0.88'):
    """Shade the bins where the prediction is exactly zero by construction."""
    if pref is None or pref < 0:
        return
    off = (phase_norm != pref).astype(int)
    for a, b in zip(np.flatnonzero(np.diff(np.r_[0, off]) == 1),
                    np.flatnonzero(np.diff(np.r_[off, 0]) == -1)):
        ax.axvspan(a - 0.5, b + 0.5, color=color, lw=0, zorder=0)


def plot_fold_ratemap_pages(results, config, out_pdf, neuron_indices=None, per_page=4,
                            sort_by='tuning_corr', title_extra=''):
    """Paged PDF of the actual-vs-predicted rate maps for EVERY fold, not the fold-average.

    One row per neuron, one column per fold, plus a final column with the n=4 preferred-phase
    state readout (thin line per fold, thick line for the mean).

    Each fold holds out a *different session*, i.e. a **different task**, so the actual tuning
    curve genuinely differs between columns -- averaging them, as the summary page does, blurs
    together the very task-specific tuning the model is being asked to predict. Each fold also
    has its own preferred phase (~36% of neurons change it between folds), so the shading and
    the n=4 readout are computed per fold too.

    Curves are unsmoothed; the shaded thirds are where the prediction is exactly zero by
    construction. Actual and predicted are on independent y-axes -- at the default alpha the
    prediction is 1-2 orders of magnitude below the firing rate, and Pearson r is
    scale-invariant, so this changes no number.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_neurons, n_folds = results['pref_phases'].shape
    if neuron_indices is None:
        neuron_indices = np.arange(n_neurons)
    neuron_indices = np.asarray(neuron_indices, dtype=int)
    if sort_by == 'tuning_corr' and neuron_indices.size:
        key = results['mean_tuning_correlations_pref'][neuron_indices]
        neuron_indices = neuron_indices[np.argsort(np.nan_to_num(key, nan=-np.inf))[::-1]]
    if neuron_indices.size == 0:
        return None

    nbps, nstates = config.num_bins_per_state, config.num_task_states
    phase_norm = _phase_label_per_norm_bin(config)
    bins = np.arange(config.total_bins)
    sessions = list(results['used_sessions'])
    reduce_key = config.state_reduce
    ncols = n_folds + 1

    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)) or '.', exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for page_start in range(0, len(neuron_indices), per_page):
            page = neuron_indices[page_start:page_start + per_page]
            fig, axes = plt.subplots(len(page), ncols,
                                     figsize=(2.45 * ncols, 2.0 * len(page)), squeeze=False)
            for row, ni in enumerate(page):
                prefs = results['pref_phases'][ni]
                for fi in range(n_folds):
                    ax = axes[row][fi]
                    actual = results['cv_actual_tuning'][ni, fi]
                    pred = results['cv_predicted_tuning'][ni, fi]
                    if np.all(np.isnan(actual)) and np.all(np.isnan(pred)):
                        ax.text(0.5, 0.5, 'no fit', ha='center', va='center',
                                transform=ax.transAxes, fontsize=7)
                        ax.set_xticks([]); ax.set_yticks([])
                    else:
                        _shade_offphase(ax, phase_norm, prefs[fi])
                        for st in range(1, nstates):
                            ax.axvline(st * nbps, color='0.4', lw=0.5, ls='--')
                        ax.plot(bins, actual, color='tab:blue', lw=1.0)
                        axp = ax.twinx()
                        axp.plot(bins, pred, color='red', lw=1.0)
                        axp.tick_params(axis='y', labelsize=4.5, labelcolor='red')
                        ax.set_xlim(0, config.total_bins)
                        ax.set_xticks(np.arange(nstates) * nbps + nbps / 2)
                        ax.set_xticklabels(list('ABCD')[:nstates], fontsize=5)
                        ax.tick_params(labelsize=4.5)
                        ax.tick_params(axis='y', labelcolor='tab:blue')
                    held = sessions[fi] if fi < len(sessions) else fi
                    r4 = results['corrs'][ni, fi]
                    r360 = results['cv_tuning_correlations_pref'][ni, fi]
                    ax.set_title(f'held-out {held} | pref {prefs[fi]}\n'
                                 f'n=4 r={r4:.2f}  pref-bin r={r360:.2f}', fontsize=5.5)

                ax = axes[row][-1]
                a4 = results[f'cv_actual_{reduce_key}_state4'][ni]
                p4 = results[f'cv_predicted_{reduce_key}_state4'][ni]
                x = np.arange(nstates)
                for fi in range(n_folds):
                    for arr, col in ((a4, 'tab:blue'), (p4, 'red')):
                        v = arr[fi]
                        if np.all(np.isnan(v)):
                            continue
                        scale = np.nanmax(np.abs(v)) or 1.0
                        ax.plot(x, v / scale, color=col, lw=0.5, alpha=0.35)
                for arr, col, lab in ((a4, 'tab:blue', 'actual'), (p4, 'red', 'predicted')):
                    m = np.nanmean(arr, axis=0)
                    if np.all(np.isnan(m)):
                        continue
                    ax.plot(x, m / (np.nanmax(np.abs(m)) or 1.0), color=col, marker='o',
                            ms=3, lw=1.4, label=lab)
                ax.set_xticks(x)
                ax.set_xticklabels(list('ABCD')[:nstates], fontsize=5)
                ax.tick_params(labelsize=4.5)
                ax.set_title(f"N{ni}  mean n=4 r={results['mean_corrs'][ni]:.3f}\n"
                             f"NZ-lag={bool(results['nonzero_lag_mask'][ni])} "
                             f"peak lag={results['peak_lags'][ni]:.0f}", fontsize=5.5)
                if row == 0:
                    ax.legend(fontsize=4.5, loc='upper right')

            fig.suptitle(f"{results['mouse_recday']} - {results['lag_direction']} lags - "
                         f"actual vs predicted per fold (each fold = a different held-out task)"
                         f"{title_extra}", fontsize=8, y=0.999)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    return out_pdf


def plot_example_betas(results, config, neuron_indices=None, num_examples=6, save_path=None,
                       show=False):
    """Grid of (location x lag) beta matrices, one coordinate frame each."""
    import matplotlib.pyplot as plt

    corrs = results.get('mean_tuning_correlations_pref', results['mean_tuning_correlations'])
    if neuron_indices is None:
        valid = ~np.isnan(corrs) & ~np.all(np.isnan(results['cv_coeffs']), axis=(1, 2))
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return None
        neuron_indices = idx[np.argsort(corrs[idx])[::-1]][:num_examples]

    num_plots = len(neuron_indices)
    ncols = min(3, num_plots)
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, ni in enumerate(neuron_indices):
        ax = axes[i]
        B, modal, n_used, n_tot = betas_in_common_frame(results, config, ni)
        if B is None:
            ax.axis('off')
            continue
        bmax = float(np.nanmax(np.abs(B))) if np.isfinite(B).any() else 0.0
        im = ax.imshow(B, aspect='auto', cmap='hot', interpolation='nearest',
                       vmin=0, vmax=bmax or 1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
        ax.set_xlabel(_lag_xlabel(config), fontsize=9)
        ax.set_ylabel('Anchor location', fontsize=9)
        ax.set_yticks(np.arange(config.num_locations))
        ax.set_yticklabels([f'L{j + 1}' for j in range(config.num_locations)], fontsize=8)
        ax.set_xticks(np.arange(0, config.num_lags, 2))
        flip = '' if n_used == n_tot else f' [{n_used}/{n_tot} folds]'
        ax.set_title(f'Neuron {ni}  r={corrs[ni]:.3f}\npref phase {modal}{flip}', fontsize=10)

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"Beta matrices ({config.num_locations} locations x {config.num_lags} "
                 f"{config.lag_direction} lags; the anchor-phase axis is determined by "
                 f"pref phase and lag)", fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ============================================================================
# Cross-mouse driver
# ============================================================================

def summarise_recday(results, config):
    """Per-recday diagnostics: the numbers that decide whether a run is interpretable."""
    nz = results['nonzero_lag_mask']
    tuned = results['state_tuned_mask']
    sel = nz & tuned & ~np.isnan(results['mean_corrs'])
    nb = results['n_nonzero_betas']
    fitted = ~np.isnan(nb)
    prefs = results['pref_phases']
    n_distinct = np.array([len(set(row[row >= 0].tolist())) for row in prefs])

    # What the mask would be with the tie-breaking guard flipped. With most betas exactly 0
    # the plain argsort picks arbitrary tied zeros, so the gap between these two numbers is a
    # direct read on how much of the mask is sort-order artefact.
    alt_cfg = copy.copy(config)
    alt_cfg.require_positive_top3 = not config.require_positive_top3
    alt_mask, _ = identify_nonzero_lag_neurons(results, alt_cfg)

    return {
        'mouse_recday': results['mouse_recday'],
        'lag_direction': results['lag_direction'],
        'n_neurons': len(nz),
        'n_folds': results['num_sessions'],
        'n_state_tuned': int(tuned.sum()),
        'n_nonzero_lag': int(nz.sum()),
        'n_selected': int(sel.sum()),
        'mean_r_selected': float(np.nanmean(results['mean_corrs'][sel])) if sel.any() else np.nan,
        'frac_untracked': results['frac_untracked'],
        'frac_allzero_fits': float(np.mean(nb[fitted] == 0)) if fitted.any() else np.nan,
        'median_nonzero_betas': float(np.nanmedian(nb)) if fitted.any() else np.nan,
        'frac_pref_phase_flips': float(np.mean(n_distinct > 1)),
        'n_nonzero_lag_alt_top3': int(alt_mask.sum()),
        'state_duration_ratio': results['state_duration_ratio'],
        'frac_pref_state_is_shortest': results['frac_pref_state_is_shortest'],
        'n_state_tuned_alt_stat': int(results['state_tuned_mask_alt'].sum()),
    }


def _print_recday_summary(s):
    print(f"  neurons={s['n_neurons']} folds={s['n_folds']} | state-tuned={s['n_state_tuned']} "
          f"NZ-lag={s['n_nonzero_lag']} both(valid r)={s['n_selected']} "
          f"mean r={s['mean_r_selected']:.3f}")
    print(f"  diagnostics: all-zero fits={s['frac_allzero_fits']:.0%} "
          f"median non-zero betas={s['median_nonzero_betas']:.0f} | "
          f"pref-phase flips across folds={s['frac_pref_phase_flips']:.0%} | "
          f"untracked bins dropped={s['frac_untracked']:.1%} | "
          f"NZ-lag with the top-3 guard flipped={s['n_nonzero_lag_alt_top3']}")
    print(f"  state-tuning confound: leg longest:shortest = {s['state_duration_ratio']:.2f}x, "
          f"{s['frac_pref_state_is_shortest']:.0%} of units prefer the SHORTEST leg "
          f"(chance 25%), state-tuned under the other statistic={s['n_state_tuned_alt_stat']}")


def run_and_summarise_all_mice_v4(data_dic, config, valid_sessions_dic=None, save_dir=None,
                                  export_dir=None, make_pdfs=True, n_jobs=1, verbose=True,
                                  mouse_recdays=None):
    """Run V4 for every mouse_recday, export per-neuron arrays and paged PDFs, and pool the
    preferred-phase state correlations for state-tuned non-zero-lag neurons.

    Fitting is parallelised over recdays with joblib's THREADING backend: sklearn's coordinate
    descent releases the GIL, and threads avoid shipping the multi-GB `data_dic` to worker
    processes. Plotting runs serially afterwards because matplotlib is not thread-safe.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if export_dir is None:
        export_dir = save_dir
    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)

    recdays = list(data_dic.keys()) if mouse_recdays is None else list(mouse_recdays)
    _t_start = time.time()

    def _one(mr):
        vs = None if valid_sessions_dic is None else valid_sessions_dic.get(mr)
        try:
            return mr, run_cross_validated_regression_v4(
                data_dic, mr, config, valid_sessions=vs, verbose=False)
        except Exception as exc:                       # keep one bad recday from killing the run
            return mr, exc

    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        pairs = Parallel(n_jobs=n_jobs, backend='threading', verbose=10 if verbose else 0)(
            delayed(_one)(mr) for mr in recdays)
    else:
        pairs = []
        for mr in recdays:
            if verbose:
                print(f"\n{'=' * 60}\nProcessing {mr}\n{'=' * 60}")
            pairs.append(_one(mr))

    all_results, pooled, summaries = {}, {}, []
    for mr, res in pairs:
        if isinstance(res, Exception):
            print(f"  {mr} failed: {res}")
            continue
        if res is None:
            continue
        all_results[mr] = res
        sel = res['nonzero_lag_mask'] & res['state_tuned_mask'] & ~np.isnan(res['mean_corrs'])
        pooled[mr] = res['mean_corrs'][sel]

        s = summarise_recday(res, config)
        summaries.append(s)
        if verbose:
            print(f"\n{mr} [{config.lag_direction}]")
            _print_recday_summary(s)

        if export_dir is not None:
            export_regression_outputs(res, config, export_dir)

        if make_pdfs and export_dir is not None:
            tag = f"{mr}_{config.lag_direction}"
            plot_neuron_pages(res, config, os.path.join(export_dir, f'{tag}_all.pdf'),
                              neuron_indices=None, title_extra='  -  all neurons')
            plot_fold_beta_pages(res, config, os.path.join(export_dir, f'{tag}_all_foldbetas.pdf'),
                                 neuron_indices=None, title_extra='  -  all neurons')
            plot_fold_ratemap_pages(res, config,
                                    os.path.join(export_dir, f'{tag}_all_foldratemaps.pdf'),
                                    neuron_indices=None, title_extra='  -  all neurons')
            nz_idx = np.flatnonzero(res['nonzero_lag_mask'])
            if nz_idx.size:
                plot_neuron_pages(res, config, os.path.join(export_dir, f'{tag}_nonzerolag.pdf'),
                                  neuron_indices=nz_idx, title_extra='  -  non-zero-lag neurons')
                plot_fold_beta_pages(res, config,
                                     os.path.join(export_dir, f'{tag}_nonzerolag_foldbetas.pdf'),
                                     neuron_indices=nz_idx,
                                     title_extra='  -  non-zero-lag neurons')
                plot_fold_ratemap_pages(res, config,
                                        os.path.join(export_dir,
                                                     f'{tag}_nonzerolag_foldratemaps.pdf'),
                                        neuron_indices=nz_idx,
                                        title_extra='  -  non-zero-lag neurons')

        if save_dir is not None and sel.sum() > 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(pooled[mr], bins=20, color='darkorange', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='k', ls='--')
            ax.axvline(np.nanmean(pooled[mr]), color='red', lw=2,
                       label=f"mean={np.nanmean(pooled[mr]):.3f}")
            ax.set_xlabel('pref-phase state corr (n=4)')
            ax.set_ylabel('# neurons')
            ax.set_title(f'{mr}\nstate-tuned NZ-lag ({config.lag_direction} lags)')
            ax.legend()
            fig.savefig(os.path.join(save_dir, f'{mr}_v4_{config.lag_direction}_corr.svg'),
                        bbox_inches='tight')
            plt.close(fig)

    manifest_extra = {
        'n_jobs': n_jobs,
        'wall_seconds': round(time.time() - _t_start, 1),
        'recdays_requested': list(recdays),
        'recdays_completed': sorted(all_results),
        'recdays_failed': sorted(set(recdays) - set(all_results)),
    }
    for target in {d for d in (export_dir, save_dir) if d is not None}:
        write_run_manifest(target, config, all_results, valid_sessions_dic, manifest_extra)

    summary_table = _summary_table(summaries)
    # `pooled` can be non-empty yet hold only empty arrays -- every recday ran, but no neuron
    # passed the selection. That is a legitimate outcome (and at the default alpha a likely one
    # on a small recday), so it must not crash the summary.
    nonempty = [v for v in pooled.values() if len(v)]
    if not nonempty:
        print('No selected neurons to summarise '
              f'({len(all_results)} recday(s) ran; check the diagnostics table).')
        return all_results, pooled, summary_table

    allvals = np.concatenate(nonempty)
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
    ax.set_title(f'Pooled - state-tuned NZ-lag ({config.lag_direction})')
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
    ax.set_title('Per-mouse mean')
    est = 'Poisson' if config.use_poisson else ('ElasticNet' if config.regularize else 'Linear')
    fig.suptitle(f"V4 raw-time {est} anchoring - {config.lag_direction} lags", fontweight='bold')
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, f'cross_mouse_v4_{config.lag_direction}_summary.svg'),
                    bbox_inches='tight')
    plt.close(fig)

    if export_dir is not None and summary_table is not None:
        summary_table.to_csv(os.path.join(export_dir,
                                          f'recday_diagnostics_{config.lag_direction}.csv'),
                             index=False)
    return all_results, pooled, summary_table


def _summary_table(summaries):
    if not summaries:
        return None
    try:
        import pandas as pd
    except ImportError:
        return summaries
    return pd.DataFrame([{k: v for k, v in s.items() if not k.startswith('_')}
                         for s in summaries])


# ============================================================================
# Anatomy: joining fitted neurons to their brain region
# ============================================================================

DEFAULT_REGIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed_data', 'unit_regions.pkl')


#: Recday -> session indices El-Gaby excludes by hand. His `Figure5_Regression.ipynb` carries
#: exactly one (verified: it is the only `mouse_recday ==` special case in the notebook).
#: me11 session 3's task [7,4,3,8] shares 3 of 4 goals with session 0's [7,4,3,5] -- his comment
#: is "task in session 3 was almost identical to session 0 (mistake)" -- so it is not a genuinely
#: novel held-out task. Dedup by exact task equality (which is what his `non_repeat_ses_maker`
#: does, and what our valid_sessions_dic reimplements) does NOT catch it, because the two rows
#: differ in one goal. Apply this in the session-selection cell, not inside the fit, so the
#: exclusion stays visible in `used_sessions`. Inert for LEC: that recday does not exist there.
EL_GABY_EXCLUDED_SESSIONS = {'me11_05122021_06122021': (3,)}


def apply_excluded_sessions(valid_sessions_dic, excluded=None, verbose=True):
    """Drop the hand-excluded sessions from a {recday: [session, ...]} mapping."""
    excluded = EL_GABY_EXCLUDED_SESSIONS if excluded is None else excluded
    out = {}
    for recday, sessions in valid_sessions_dic.items():
        drop = set(excluded.get(recday, ()))
        kept = [s for s in sessions if s not in drop]
        if drop and verbose:
            hit = sorted(drop & set(sessions))
            if hit:
                print(f'  {recday}: dropping session(s) {hit} (El-Gaby hand-exclusion) '
                      f'-> {len(kept)} folds')
        out[recday] = kept
    return out


def load_unit_regions(path=None, required=True):
    """`{mouse_recday: DataFrame}` of per-unit anatomy (mouse, acronym, group, y_um, ...).

    Rows are in `Neuron_raw` row order, which is what makes the positional join below valid.

    Returns None instead of raising when the pickle is absent and `required=False` -- the
    PFC/mFC dataset has no anatomy at all, and `build_unit_table` handles that case.
    """
    import pickle
    target = DEFAULT_REGIONS_PATH if path is None else path
    if not os.path.exists(target):
        if required:
            raise FileNotFoundError(f'no unit_regions at {target}')
        return None
    with open(target, 'rb') as f:
        return pickle.load(f)


def mean_firing_rates(data_dic, mouse_recday, used_sessions, bin_seconds=0.025):
    """Mean rate (Hz) per neuron over the sessions the regression actually used.

    Carried into the unit table because at a fixed ElasticNet alpha the fit is partly a
    firing-rate cut (all-zero betas average ~0.7 Hz, surviving fits ~6.4 Hz), so any
    region-by-selection-rate table has to be read against rate.
    """
    tot = None
    n_bins = 0
    for s in used_sessions:
        nr = data_dic[mouse_recday][s].get('Neuron_raw')
        if nr is None:
            continue
        nr = np.asarray(nr, dtype=float)
        tot = nr.sum(axis=1) if tot is None else tot + nr.sum(axis=1)
        n_bins += nr.shape[1]
    if tot is None or n_bins == 0:
        return None
    return tot / (n_bins * bin_seconds)


def build_unit_table(all_results, config, regions=None, data_dic=None, strict=True,
                     require_anatomy=None):
    """One row per recorded unit, optionally joined to `unit_regions.pkl`.

    Columns: `lag_direction`, `nonzero_lag`, `state_tuned`, `state_tuned_alt_stat`, `peak_lag`,
    `mean_corr`, `mean_corr_nonzero`, `tuning_corr`, `tuning_corr_pref`, `n_nonzero_betas`,
    `pref_phase_modal`, `pref_phase_flips`, `n_informative_folds`, `mean_rate_hz`, and
    `selected` (= nonzero_lag & state_tuned & mean_corr.notna(), the criterion the paper's
    figure uses) -- plus, when anatomy is available, everything from the anatomy table
    (mouse, acronym, group, y_um, shank, ...).

    **Anatomy is optional.** The PFC/mFC dataset has none -- no `unit_regions`, no
    `anatomy_split` -- so when it is missing the frame is built from the results alone:
    `recday`, `order` (the `Neuron_raw` row index, which is what the anatomy join is positional
    on anyway), `unit_id`, and `mouse` parsed from the recday prefix. `region_summary` and
    `compare_directions` then group by `mouse` instead of `group`. Set `require_anatomy=True`
    to make a missing join an error instead.

    `strict` keeps the row-count gate that makes the positional join safe: the anatomy table
    must have exactly as many rows as the recday has neurons.
    """
    import pandas as pd
    if regions is None:
        regions = load_unit_regions(required=bool(require_anatomy))
    has_anatomy = regions is not None
    if require_anatomy and not has_anatomy:
        raise FileNotFoundError('build_unit_table(require_anatomy=True) but no unit_regions '
                                'was found or supplied')

    rows = []
    for recday, res in all_results.items():
        n = len(res['nonzero_lag_mask'])
        if has_anatomy:
            if recday not in regions:
                if strict:
                    raise KeyError(f'{recday}: no entry in unit_regions')
                continue
            reg = regions[recday]
            if len(reg) != n:
                msg = (f'{recday}: unit_regions has {len(reg)} rows but the regression has '
                       f'{n} neurons')
                if strict:
                    raise AssertionError(msg)
                print('  skipping -', msg)
                continue
        else:
            # No anatomy: the identity columns the rest of the module needs, nothing invented.
            reg = pd.DataFrame({'recday': recday,
                                'mouse': recday.split('_')[0],
                                'order': np.arange(n),
                                'unit_id': np.arange(n)})

        prefs = res['pref_phases']
        modal = np.array([_modal_pref(row)[0] for row in prefs])
        n_distinct = np.array([len(set(row[row >= 0].tolist())) for row in prefs])

        block = reg.assign(
            lag_direction=res['lag_direction'],
            nonzero_lag=res['nonzero_lag_mask'],
            state_tuned=res['state_tuned_mask'],
            peak_lag=res['peak_lags'],
            mean_corr=res['mean_corrs'],
            mean_corr_nonzero=res['mean_corrs_nonzero'],
            mean_corr_max=res['mean_corrs_max'],
            tuning_corr=res['mean_tuning_correlations'],
            tuning_corr_pref=res['mean_tuning_correlations_pref'],
            n_nonzero_betas=np.nanmean(res['n_nonzero_betas'], axis=1),
            pref_phase_modal=modal,
            pref_phase_flips=n_distinct > 1,
            n_folds=res.get('num_sessions', len(res['used_sessions'])),
            n_informative_folds=res['n_informative_folds'],
            state_tuned_alt_stat=res['state_tuned_mask_alt'],
            nz_fold_vote_frac=np.nanmean(res['nz_fold_votes'], axis=1),
        )
        if data_dic is not None:
            rates = mean_firing_rates(data_dic, recday, res['used_sessions'])
            block = block.assign(mean_rate_hz=rates if rates is not None else np.nan)
        rows.append(block)

    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    table['selected'] = (table.nonzero_lag & table.state_tuned & table.mean_corr.notna())
    table.attrs['has_anatomy'] = has_anatomy
    return table


def _resolve_group_col(table, group_col=None):
    """The column to group by: the caller's, else 'group' (anatomy), else 'mouse'."""
    if group_col is not None:
        if group_col not in table.columns:
            raise KeyError(f'{group_col!r} is not a column of this table. Available: '
                           f'{sorted(table.columns)}')
        return group_col
    for candidate in ('group', 'mouse'):
        if candidate in table.columns:
            return candidate
    raise KeyError(f'no grouping column found. Available: {sorted(table.columns)}')


def region_summary(table, group_col=None, verbose=True):
    """Selection rate and mean r by group, with the two confounds that can fake one.

    `group_col` defaults to `'group'` (anatomy) when present, else `'mouse'` -- the PFC/mFC
    dataset has no anatomy, so there it summarises per animal.

    Returns {'by_region', 'chi2', 'by_region_mouse', 'units_by_region_mouse', 'rate_confound'}.

    `rate_confound` is the reason this is not just a crosstab: at a fixed ElasticNet alpha a
    unit is only fittable if it fires fast enough, so a group difference in selection rate can
    be a group difference in firing rate. It reports mean rate per group and the same
    selection rate within firing-rate quartiles, where a real effect should survive.
    """
    import pandas as pd
    from scipy.stats import chi2_contingency

    group_col = _resolve_group_col(table, group_col)
    # Second axis of the breakdown: mouse normally, recday when the groups ARE mice (a
    # mouse x mouse crosstab is diagonal and says nothing).
    split_col = 'recday' if group_col == 'mouse' else 'mouse'
    if split_col not in table.columns:
        split_col = group_col

    sel = table.selected
    by_region = (table.groupby(group_col)
                 .apply(lambda d: pd.Series({
                     'n_units': len(d),
                     'n_selected': int(d.selected.sum()),
                     'rate': round(d.selected.mean(), 3),
                     'mean_corr': d.loc[d.selected, 'mean_corr'].mean(),
                     'mean_corr_nonzero': d.loc[d.selected, 'mean_corr_nonzero'].mean(),
                     'median_peak_lag': d.loc[d.selected, 'peak_lag'].median(),
                 }), include_groups=False))
    # nunique of the identity columns separately: `include_groups=False` drops the group key,
    # so a groupby-mouse would report n_mice as NaN from inside the apply.
    for col, name in (('mouse', 'n_mice'), ('recday', 'n_recdays')):
        if col in table.columns:
            by_region.insert(2, name, table.groupby(group_col)[col].nunique())
    by_region = by_region.sort_values('rate', ascending=False)

    ct = pd.crosstab(table[group_col], sel)
    chi2, p, dof = (np.nan, np.nan, 0)
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, _ = chi2_contingency(ct)

    by_region_mouse = pd.crosstab(table[group_col], table[split_col],
                                  values=sel, aggfunc='mean').round(3)
    units_by_region_mouse = pd.crosstab(table[group_col], table[split_col])

    rate_confound = None
    if 'mean_rate_hz' in table.columns and table.mean_rate_hz.notna().any():
        t = table[table.mean_rate_hz.notna()].copy()
        try:
            q = pd.qcut(t.mean_rate_hz, 4, labels=False, duplicates='drop')
        except ValueError:                       # too few distinct rates to split
            return {'by_region': by_region, 'chi2': (chi2, p, dof),
                    'by_region_mouse': by_region_mouse,
                    'units_by_region_mouse': units_by_region_mouse, 'rate_confound': None}
        t['rate_quartile'] = [f'Q{int(v) + 1}' for v in q]
        rate_confound = {
            'mean_rate_by_region': t.groupby(group_col, observed=True).mean_rate_hz.mean().round(2),
            'rate_by_region_quartile': pd.crosstab(t[group_col], t.rate_quartile,
                                                   values=t.selected, aggfunc='mean').round(3),
            'units_by_region_quartile': pd.crosstab(t[group_col], t.rate_quartile),
        }

    if verbose:
        print(by_region.to_string())
        if np.isfinite(chi2):
            print(f'\n{group_col} x selected: chi2={chi2:.1f}, dof={dof}, p={p:.2g}')
        else:
            print(f'\n{group_col} x selected: only one level or one outcome, no chi2')
        print(f'\nselection rate by {group_col} x {split_col}:')
        print(by_region_mouse.to_string())
        print('\nunits behind those rates:')
        print(units_by_region_mouse.to_string())
        if rate_confound is not None:
            print('\n--- firing-rate confound ---')
            print(f'mean rate (Hz) by {group_col}:')
            print(rate_confound['mean_rate_by_region'].to_string())
            print(f'\nselection rate by {group_col} within firing-rate quartile '
                  '(a real effect should survive this):')
            print(rate_confound['rate_by_region_quartile'].to_string())
            print('\nunits per cell:')
            print(rate_confound['units_by_region_quartile'].to_string())

    return {'by_region': by_region, 'chi2': (chi2, p, dof),
            'by_region_mouse': by_region_mouse,
            'units_by_region_mouse': units_by_region_mouse,
            'rate_confound': rate_confound}


def compare_directions(table_past, table_future, group_col=None, verbose=True):
    """Merge the past and future unit tables and contrast them per group.

    Joins on the identity keys (recday + Neuron_raw row order), so a unit appears once with a
    `_past` and a `_future` copy of every regression column. `pro_index` is
    (r_future - r_past) / (|r_future| + |r_past|): positive = better explained prospectively.

    `group_col` defaults to `'group'` (anatomy) when present, else `'mouse'`. Needs no anatomy:
    only a grouping column, which is why this works unchanged on the PFC dataset.
    """
    import pandas as pd
    group_col = _resolve_group_col(table_past, group_col)
    keys = list(dict.fromkeys(['recday', 'order', 'unit_id', 'mouse', group_col]))
    keys = [k for k in keys if k in table_past.columns and k in table_future.columns]
    val_cols = ['nonzero_lag', 'state_tuned', 'selected', 'peak_lag', 'mean_corr',
                'mean_corr_nonzero', 'tuning_corr_pref', 'n_nonzero_betas']
    val_cols = [c for c in val_cols if c in table_past.columns]

    merged = table_past[keys + val_cols].merge(
        table_future[keys + val_cols], on=keys, suffixes=('_past', '_future'))
    denom = merged.mean_corr_future.abs() + merged.mean_corr_past.abs()
    merged['pro_index'] = (merged.mean_corr_future - merged.mean_corr_past) / denom.replace(0, np.nan)

    summary = (merged.groupby(group_col)
               .apply(lambda d: pd.Series({
                   'n_units': len(d),
                   'sel_past': int(d.selected_past.sum()),
                   'sel_future': int(d.selected_future.sum()),
                   'sel_both': int((d.selected_past & d.selected_future).sum()),
                   'r_past': d.loc[d.selected_past, 'mean_corr_past'].mean(),
                   'r_future': d.loc[d.selected_future, 'mean_corr_future'].mean(),
                   'pro_index': d.loc[d.selected_past | d.selected_future, 'pro_index'].mean(),
               }), include_groups=False)
               .sort_values('pro_index', ascending=False))
    if verbose:
        print(summary.to_string())
        both = merged[merged.selected_past | merged.selected_future]
        if len(both) > 1:
            t, p = stats.ttest_1samp(both.pro_index.dropna(), 0)
            print(f'\npooled prospective-vs-retrospective index: n={both.pro_index.notna().sum()}, '
                  f'mean={both.pro_index.mean():.3f}, t={t:.2f}, p={p:.2g}')
    return merged, summary
