"""
Anchoring regression — V5.

Seeded from `elasticnet_regression_v4.py`, which is left untouched so the two can be diffed and
so `elasticnet_v5_synthetics.py` control 8b can assert that V5 under the legacy flags
(`state_tuning_statistic='max'`, `state_tuning_nan_policy='omit'`, `pref_phase_method='raw_mean'`)
reproduces V4 to `max|diff| = 0`. V4 was itself seeded from v3 (a raw-time reimplementation of
El-Gaby et al. 2024 `Figure5_Regression.ipynb`).

What V5 changes
---------------
Everything here follows from reading El-Gaby's *deposited code* (`mFC_schema-main/Figure2.ipynb`
cells 31/46/48/56, `Figure5_Regression.ipynb` cells 21/26/38) rather than the paper's text, and
from checking each definition against his published counts (738/489/349 state-tuned neurons).
See `ELGABY_FIGURE5_RECONCILIATION.md`.

1. **His state-tuning test** (`state_tuning_statistic='pref_phase_mean'`, the new default).
   The paper says "peak firing rate in each state and trial"; his code takes the MEAN over the
   neuron's preferred-phase bins per (trial, state), z-scores across states with
   `scipy.stats.zscore` (NaN propagates: `state_tuning_nan_policy='propagate'`), and t-tests
   the preferred state against 0. `'max'` (the paper's text, leg-duration confounded) and
   `'mean'` remain available.

2. **His preferred-phase rule** (`pref_phase_method='elgaby_peak'`, the new default): normalise
   the session to (trials, 360), average over trials and states to a 90-bin curve, take the
   peak bin within each third, argmax. `'raw_mean'` (V4: argmax of the raw time-weighted mean
   rate per phase) remains. The two disagree on ~30% of PFC neuron-sessions.

3. **His fold semantics, post hoc.** A neuron counts on >= 1 passing fold and its value is the
   mean over passing folds only. His cell 38 applies `remove_nan` to each of the three
   correlation arrays INDEPENDENTLY, so "with non-zero beta coefficients" is a per-panel
   requirement: state-tuned AND >= 1 fold with a finite value *for that panel*. (Corrected
   2026-09-08: `'elgaby'` previously also demanded a finite r in every fold, which suppressed
   the non-zero-lag panels ~4x in n and ~2x in t. That rule survives as
   `semantics='elgaby_everyfold'`; see `PANEL_POOL_CORRECTION.md`.) `add_fold_semantics` derives
   `n_passing_folds*`, `mean_corrs_nonzero*_passing` and `n_finite_folds` from the stored
   per-fold arrays, so a V4 export can be re-scored.
   `three_panel_summary(semantics=...)` reports this alongside the V4 majority-vote semantics.

4. **Plots.** Polar actual-vs-predicted rate maps per held-out task (Fig 5g style); for Poisson
   runs the other link's predicted curves are stored and drawn, and the cross-mouse summary has
   one row of panels per link.

5. **Bookkeeping.** `_prepare_session_with_reason` says why a session was dropped
   (`results['sessions_skipped']`, written to the manifest); `require_positive_mean_prediction`
   reproduces his `nanmean(prediction) > 0` gate for the Poisson reproduction; the per-recday
   pooled histogram and `mean_r_selected` now score non-zero-lag neurons with the reduced-beta r.

6. **`y_scaling` — the one option here that is NOT the reference.** A fixed `elasticnet_alpha`
   on raw spike counts is a firing-rate filter: the zeroing threshold scales with sd(y), so at
   alpha=0.01 59% of LEC neurons get an all-zero fit (97% in the slowest rate quartile).
   `y_scaling='zscore_recday'` divides each neuron's fit target by its sd over ALL used sessions
   of the recday at once — one scalar per neuron, so between-session rate differences within a
   recday are preserved — which takes the all-zero fraction to 1%. Such runs are tagged
   `_zscore` in the run-directory name (`run_tag`) and in `run_config.json`. Raw-count runs
   remain the reproduction. Poisson rejects it (needs y >= 0).

What V4 added (unchanged)
-------------------------
1. **A future lag direction.** v3/the reference are retrospective only: lag *k* means "this
   (location, phase) anchor was visited *k* phase-steps ago". `lag_direction='future'` builds
   the prospective mirror ("I will be at this anchor in *k* phase-steps") by reversing the
   location/phase sequences, running the *identical* bump loop, and reversing back — so both
   directions carry the same roll/wipe idiosyncrasies.

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
`elasticnet_regression_v4.py`, `ccgp_state_pairs.py`, `taskphase_periodicity.py` and
`persistent_homology_analysis.py` already are. Edit one, copy to the other, and check with

    diff code/elasticnet_regression_v5.py mFC_data/code/elasticnet_regression_v5.py

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

class RegressionConfigV5:
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
                             per-(trial, state) reduction inside the state-tuning filter.
                             'pref_phase_mean' (default) is what El-Gaby's CODE does (Figure2
                             cell 46): the mean over the neuron's preferred-phase bins, 30 of
                             the 90 normalised bins of that state. 'max' is what the PAPER'S
                             TEXT says ("peak firing rate in each state and trial") and what
                             v2-v4 implemented; it is CONFOUNDED BY LEG DURATION: `raw_to_norm`
                             averages more raw bins into each normalised bin when a state
                             interval is longer, which lowers its variance and so its max. On
                             constant-rate Poisson cells its false-positive rate goes 5% ->
                             73-95% -> 100% as the longest:shortest state-duration ratio goes
                             1x -> 2x -> 3x, and the real data's median ratio is 2.26x. 'mean'
                             (over all 90 bins) is duration-invariant. The other statistic is
                             always computed too (`state_tuned_mask_alt`).
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

    New in v5 (each matches El-Gaby's deposited code; see the module docstring):
        state_tuning_nan_policy
                             'propagate' (default, his `scipy.stats.zscore` + `ttest_1samp`):
                             a trial whose per-state summaries are all equal -- typically all
                             zero -- makes the z-scores NaN and the whole session "not tuned"
                             for that neuron. Removes ~15% of neuron-sessions on PFC and
                             preferentially low-rate units. 'omit' (v4) drops such trials.
        pref_phase_method    'elgaby_peak' (default): normalise the session to (trials, 360),
                             average over trials and states to a 90-bin curve, take the peak
                             bin within each third, argmax (his `tuning_phase_boolean_max`).
                             'raw_mean' (v4): argmax of the raw, time-weighted mean rate per
                             phase over tracked node bins. They disagree on ~30% of PFC
                             neuron-sessions. Used for the fit/scoring phase AND inside the
                             'pref_phase_mean' state test. With pref_phase_source='train' the
                             'elgaby_peak' curve is the trial-weighted mean over the training
                             sessions' curves.
        pref_phase_smooth_sigma
                             None (default) or a sigma in normalised bins for his
                             `smooth_circular` on each trial's 360-bin curve before the peak
                             is taken. Whether his `Neuron_` arrays were smoothed is unknown
                             (the writer cell is not in the repo), so this is a sensitivity
                             knob, not a reference value.
        require_positive_mean_prediction
                             his cell-26 gate `nanmean(prediction) > 0`, applied to the linear
                             predictor before any fold is scored. Identical to the existing
                             std > 0 requirement for ElasticNet(positive=True); for Poisson it
                             drops folds whose linear predictor has a negative mean. Default
                             False; set True for the Poisson reproduction.
        y_scaling            'none' (default, the reproduction: raw integer spike counts) or
                             'zscore_recday'. NOT part of the reference; it exists because a
                             FIXED `elasticnet_alpha` on raw counts is a firing-rate filter.
                             ElasticNet zeroes every coefficient above
                             `alpha_max = max|x_c . y_c| / (n * l1_ratio)`, which scales with
                             sd(y): on one LEC fold the median alpha_max is 0.0074, so at
                             alpha=0.01 59% of neurons get an all-zero fit (97% in the slowest
                             rate quartile) and the all-zero mask correlates +0.90 with firing
                             rate. 'zscore_recday' divides each neuron's fit target by its sd
                             over ALL used sessions of the recday at once (one scalar per
                             neuron, not per session, so genuine between-session rate
                             differences within a recday are preserved rather than normalised
                             away); median alpha_max -> 0.0382, all-zero -> 1%, rate
                             correlation -> +0.64. Note ElasticNet centres y internally, so the
                             mean subtraction is a no-op and only the sd division bites; and
                             since the L1 weight then scales with sd while the L2 weight does
                             not, this is NOT a pure per-neuron alpha rescale at l1_ratio<1
                             (synthetic control 13 pins the exact algebra). Applied to the fit
                             target only: the state test, preferred phases and the n=4
                             state-mean Pearson readout are all invariant to a positive
                             per-neuron affine transform, so those arrays are bit-identical to
                             the raw run (control 12). Poisson needs y >= 0 and so rejects any
                             scaling other than 'none'. Caveat: it also admits very low-spike
                             neurons (a 122-spike unit's alpha_max goes 0.0005 -> 0.0175), so
                             `n_spikes_recday` is exported for post hoc gating.

    Set lag_direction='past', pref_phase_source='train', drop_untracked_bins=False,
    nonzero_lag_zero_lags=(0,), require_positive_top3=False, nz_per_fold=False,
    state_tuning_statistic='max', state_tuning_nan_policy='omit', pref_phase_method='raw_mean'
    to reproduce v3; the last three alone reproduce v4.
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
        pref_phase_source='test',
        drop_untracked_bins=True,
        alpha_mode='fixed',
        alpha_frac=0.1,
        state_reduce='mean',
        state_tuning_statistic='pref_phase_mean',
        state_tuning_min_fraction=1 / 3,
        poisson_link='linear',
        nonzero_lag_zero_lags=(0, 11),
        nonzero_lag_min=1,
        nonzero_lag_max=None,
        nonzero_lag_zero_lags_strict=(0, 1, 2, 9, 10, 11),
        nonzero_lag_min_strict=3,
        nonzero_lag_max_strict=8,
        require_positive_top3=True,
        nz_per_fold=True,
        # --- v5 ---
        state_tuning_nan_policy='propagate',
        pref_phase_method='elgaby_peak',
        pref_phase_smooth_sigma=None,
        require_positive_mean_prediction=False,
        y_scaling='none',
        # --- 2026-09-14: L1 / elastic-net Poisson via glum ---
        poisson_solver='sklearn',
        poisson_l1_ratio=0.0,
        poisson_positive=False,
    ):
        if lag_direction not in ('past', 'future'):
            raise ValueError(f"lag_direction must be 'past' or 'future', got {lag_direction!r}")
        if pref_phase_source not in ('train', 'test'):
            raise ValueError(f"pref_phase_source must be 'train' or 'test', got {pref_phase_source!r}")
        if alpha_mode not in ('fixed', 'relative'):
            raise ValueError(f"alpha_mode must be 'fixed' or 'relative', got {alpha_mode!r}")
        if state_reduce not in ('mean', 'max'):
            raise ValueError(f"state_reduce must be 'mean' or 'max', got {state_reduce!r}")
        if poisson_link not in ('linear', 'log'):
            raise ValueError(f"poisson_link must be 'linear' or 'log', got {poisson_link!r}")
        if not 0 <= state_tuning_min_fraction < 1:
            raise ValueError('state_tuning_min_fraction must be in [0, 1), got '
                             f'{state_tuning_min_fraction!r}')

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
        if state_tuning_statistic not in ('max', 'mean', 'pref_phase_mean'):
            raise ValueError("state_tuning_statistic must be 'max', 'mean' or "
                             f"'pref_phase_mean', got {state_tuning_statistic!r}")
        if state_tuning_nan_policy not in ('propagate', 'omit'):
            raise ValueError("state_tuning_nan_policy must be 'propagate' or 'omit', got "
                             f'{state_tuning_nan_policy!r}')
        if pref_phase_method not in ('raw_mean', 'elgaby_peak'):
            raise ValueError("pref_phase_method must be 'raw_mean' or 'elgaby_peak', got "
                             f'{pref_phase_method!r}')
        if y_scaling not in ('none', 'zscore_recday'):
            raise ValueError("y_scaling must be 'none' or 'zscore_recday', got "
                             f'{y_scaling!r}')
        if y_scaling != 'none' and use_poisson:
            # A z-scored target is negative on ~85% of bins; PoissonRegressor requires y >= 0
            # and would raise inside the fold loop, after hours of preparation.
            raise ValueError(f"y_scaling={y_scaling!r} needs a real-valued target and so cannot "
                             'be combined with use_poisson=True (Poisson requires y >= 0). '
                             'Note use_poisson defaults to True: pass use_poisson=False for an '
                             'ElasticNet run.')
        self.y_scaling = y_scaling
        # sklearn's PoissonRegressor is L2-only and silently ignores `l1_ratio`; a lasso or
        # elastic-net Poisson needs glum (irls-cd), whose objective uses the same convention
        # (1/(2n) deviance + alpha * [l1 ||w||_1 + (1 - l1)/2 ||w||^2]) so `poisson_alpha`
        # means the same thing on both solvers and glum with l1_ratio=0 reproduces sklearn.
        if poisson_solver not in ('sklearn', 'glum'):
            raise ValueError(f"poisson_solver must be 'sklearn' or 'glum', got {poisson_solver!r}")
        if not 0.0 <= float(poisson_l1_ratio) <= 1.0:
            raise ValueError(f'poisson_l1_ratio must be in [0, 1], got {poisson_l1_ratio!r}')
        if poisson_solver == 'sklearn' and (poisson_l1_ratio > 0 or poisson_positive):
            raise ValueError("poisson_l1_ratio > 0 or poisson_positive=True need "
                             "poisson_solver='glum' (sklearn's PoissonRegressor is L2-only and "
                             'unconstrained; it would silently ignore both).')
        self.poisson_solver = poisson_solver
        self.poisson_l1_ratio = float(poisson_l1_ratio)
        self.poisson_positive = bool(poisson_positive)
        self.state_tuning_statistic = state_tuning_statistic
        self.state_tuning_nan_policy = state_tuning_nan_policy
        self.pref_phase_method = pref_phase_method
        self.pref_phase_smooth_sigma = pref_phase_smooth_sigma
        self.require_positive_mean_prediction = bool(require_positive_mean_prediction)
        self.state_tuning_min_fraction = state_tuning_min_fraction
        self.poisson_link = poisson_link
        self.nonzero_lag_zero_lags = tuple(int(k) % num_lags for k in nonzero_lag_zero_lags)
        self.nonzero_lag_min = nonzero_lag_min
        self.nonzero_lag_max = num_lags - 2 if nonzero_lag_max is None else nonzero_lag_max
        self.nonzero_lag_zero_lags_strict = tuple(int(k) % num_lags
                                                  for k in nonzero_lag_zero_lags_strict)
        self.nonzero_lag_min_strict = nonzero_lag_min_strict
        self.nonzero_lag_max_strict = nonzero_lag_max_strict
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
        return (f"RegressionConfigV5({est}, lag_direction={self.lag_direction!r}, "
                f"pref_phase_source={self.pref_phase_source!r}, "
                f"pref_phase_method={self.pref_phase_method!r}, "
                f"state_tuning_statistic={self.state_tuning_statistic!r}, "
                f"alpha_mode={self.alpha_mode!r}, state_reduce={self.state_reduce!r}, "
                f"y_scaling={self.y_scaling!r})")


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
                                     return_pref=False, pref_phases=None, statistic=None,
                                     nan_policy=None):
    """Boolean mask of state-tuned neurons for ONE session.

    Per neuron: per-(trial, state) summary -> z-score across the states within each trial ->
    preferred state -> one-sample t-test of that state's per-trial z against 0 -> p < threshold.

    `statistic` (default `config.state_tuning_statistic`):
      'pref_phase_mean'  El-Gaby's CODE (Figure2.ipynb cell 46): the mean over the neuron's
                         preferred-phase bins of each state (30 of the 90 normalised bins),
                         z-scored across states with `scipy.stats.zscore` semantics, the
                         preferred state = argmax of the trial-mean of those means, t-test on
                         that column. Needs this session's per-neuron preferred phase
                         (`pref_phases`; computed with `session_pref_phases` when not given).
                         Reproduces his `State_95` count on PFC (736 vs the paper's 738).
      'max'              the PAPER'S TEXT ("peak firing rate in each state and trial") and what
                         v2-v4 implemented. Confounded by leg duration: `raw_to_norm` averages
                         more raw bins into each normalised bin when a state interval is
                         longer, lowering its variance and so its max, so constant-rate cells
                         acquire the shortest leg as "preferred state" (false-positive rate 5%
                         at a 1x duration ratio, ~100% at 3x; the real median ratio is 2.26x).
      'mean'             the mean over all 90 bins of the state; duration-invariant.

    `nan_policy` (default `config.state_tuning_nan_policy`): 'propagate' is his -- a trial whose
    four summaries are equal (typically all zero) gives NaN z-scores, `ttest_1samp` returns NaN
    and the session counts as not tuned; 'omit' (v4) drops such trials and needs >= 3 left.
    """
    if p_threshold is None:
        p_threshold = config.state_tuning_p_threshold
    statistic = config.state_tuning_statistic if statistic is None else statistic
    nan_policy = (getattr(config, 'state_tuning_nan_policy', 'omit') if nan_policy is None
                  else nan_policy)
    n_neurons = neuron_raw.shape[0]
    nstates = config.num_task_states
    nbps = config.num_bins_per_state
    is_tuned = np.zeros(n_neurons, dtype=bool)
    pref_states = np.full(n_neurons, -1, dtype=int)

    his = statistic == 'pref_phase_mean'
    if his:
        if pref_phases is None:
            pref_phases = session_pref_phases(neuron_raw, trial_times, config)
        pref_phases = np.asarray(pref_phases, dtype=int)
        phase_norm = _phase_label_per_norm_bin(config)
    reducer = np.nanmax if statistic == 'max' else np.nanmean
    # his code has no minimum trial count (a 1-trial session simply gives a NaN t-test); v4
    # required 3 complete trials and that is kept for the legacy statistics
    min_trials = 2 if his else 3

    for ni in range(n_neurons):
        if his and pref_phases[ni] < 0:
            continue
        per_trial = raw_to_norm(neuron_raw[ni], trial_times, config, return_mean=False)
        if per_trial is None or per_trial.shape[0] < min_trials:
            continue
        n_trials = per_trial.shape[0]
        summary = np.full((n_trials, nstates), np.nan)
        for s in range(nstates):
            block = per_trial[:, s * nbps:(s + 1) * nbps]
            if his:
                block = block[:, phase_norm[s * nbps:(s + 1) * nbps] == pref_phases[ni]]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                summary[:, s] = reducer(block, axis=1)
        if his:
            # preferred state from the trial-mean of the raw means; z-scores with ddof=0 and
            # full NaN propagation, exactly `scipy.stats.zscore(summary, axis=1)`
            col = np.nanmean(summary, axis=0)
            if not np.any(np.isfinite(col)):
                continue
            pref = int(np.nanargmax(col))
            with np.errstate(all='ignore'):
                mu = summary.mean(axis=1, keepdims=True)
                sd = summary.std(axis=1, keepdims=True)
                sd[sd == 0] = np.nan
                z = (summary - mu) / sd
        else:
            row_mean = np.nanmean(summary, axis=1, keepdims=True)
            row_std = np.nanstd(summary, axis=1, keepdims=True)
            row_std[row_std == 0] = np.nan
            z = (summary - row_mean) / row_std
            col = np.nanmean(z, axis=0)
            if not np.any(np.isfinite(col)):
                continue
            pref = int(np.nanargmax(col))
        pref_states[ni] = pref
        zp = z[:, pref]
        if nan_policy == 'omit':
            zp = zp[~np.isnan(zp)]
            if len(zp) < 3:
                continue
        elif np.any(np.isnan(zp)):
            continue                              # his ttest_1samp returns NaN -> not tuned
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


def fit_regression_v5(X, y, config, return_intercept=False):
    """Fit the chosen estimator, dropping rows with NaNs.

    Returns the coefficient vector, or `(coef, intercept)` with `return_intercept=True`.
    The intercept exists only so `poisson_link='log'` can evaluate `exp(X@beta + b)` without
    overflowing; Pearson is scale-invariant, so it changes no reported number.
    """
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    Xv, yv = X[valid], y[valid]
    if len(yv) < 10 or np.all(yv == yv[0]):
        nan = np.full(X.shape[1], np.nan)
        return (nan, np.nan) if return_intercept else nan

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if config.use_poisson and getattr(config, 'poisson_solver', 'sklearn') == 'glum':
            # Lazy import: glum is only needed for the L1 / elastic-net Poisson branch, and
            # the module must keep importing (and the sklearn branch keep running) without it.
            from glum import GeneralizedLinearRegressor
            model = GeneralizedLinearRegressor(
                family='poisson', alpha=config.poisson_alpha,
                l1_ratio=config.poisson_l1_ratio, max_iter=config.max_iter,
                lower_bounds=(np.zeros(Xv.shape[1]) if config.poisson_positive else None))
        elif config.use_poisson:
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
            nan = np.full(X.shape[1], np.nan)
            return (nan, np.nan) if return_intercept else nan
    if return_intercept:
        return model.coef_, float(np.ravel(getattr(model, 'intercept_', 0.0))[0])
    return model.coef_


# ============================================================================
# Per-session preparation
# ============================================================================

def apply_link(eta, intercept, config):
    """Turn the linear predictor into the predicted trace.

    `poisson_link='linear'` reproduces his code, which computes `np.sum(regressors * coeffs)`
    and never applies `exp` or the intercept -- so the Poisson variant's stated purpose,
    robustness to the linearity assumption, is discarded at readout. `'log'` is what the paper
    describes ("linear-nonlinear-Poisson ... logarithmic link function"). The ElasticNet branch
    has an identity link either way, and its missing intercept is harmless because Pearson is
    shift-invariant.
    """
    if not config.use_poisson or config.poisson_link == 'linear':
        return eta
    b = 0.0 if intercept is None or not np.isfinite(intercept) else intercept
    return np.exp(np.clip(eta + b, -700, 700))


def _prepare_session_with_reason(session_data, config):
    """Build raw regressors / aligned locations / phases / neuron matrix for one session.

    Returns `(prep, None)`, or `(None, reason)` when the session cannot be used -- the reason
    is recorded in `results['sessions_skipped']` and the run manifest, because on the LEC pickle
    31 of 212 sessions fail here (21 Object-exploration blocks with no task, 9 ABCD sessions with
    no completed trial, 1 with tracking that could not be synced) and they used to vanish
    silently from the fold set.
    """
    for key in ('Neuron_raw', 'Locs_raw', 'Trial_times'):
        if key not in session_data or session_data[key] is None:
            return None, f'missing {key}'
    neuron_raw = np.asarray(session_data['Neuron_raw'], dtype=float)   # (n_neurons, n_bins)
    locs_raw = np.asarray(session_data['Locs_raw'], dtype=float)       # (n_bins,)
    tt_any = np.asarray(session_data['Trial_times'])
    if neuron_raw.ndim != 2:
        return None, 'Neuron_raw is not 2-D'
    if tt_any.ndim != 2 or tt_any.shape[0] < 2:
        n_tr = 0 if tt_any.size == 0 else (tt_any.shape[0] if tt_any.ndim == 2 else 1)
        return None, ('no completed trials' if n_tr == 0
                      else f'{n_tr} completed trial(s), need >= 2')
    tt = tt_any.astype(int)                                            # (n_trials, n_states+1)

    phase_per_bin, state_per_bin = compute_phase_state_raw(
        tt, num_phases=config.num_goal_progress_bins, num_states=config.num_task_states)

    L = min(len(locs_raw), neuron_raw.shape[1], len(phase_per_bin))
    if L < config.num_bins_per_state:
        return None, f'only {L} aligned bins'
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

    # Both preferred-phase rules, once per session. 'raw_mean' is exactly what v4 computed per
    # fold from the held-out session (same rows: tracked node bins); 'elgaby_peak' is his.
    keep = ~np.isnan(locs)
    pref_raw, _ = _pref_phase_from(neuron[keep], phases[keep], config.num_goal_progress_bins)
    curve, n_norm_trials = elgaby_phase_curve(neuron_raw, tt, config,
                                              smooth_sigma=config.pref_phase_smooth_sigma)
    pref_elgaby = pref_phase_from_curve(curve, config)

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
        'pref_phase_raw': pref_raw,              # (n_neurons,)  v4 rule
        'pref_phase_elgaby': pref_elgaby,        # (n_neurons,)  his rule
        'phase_curve_elgaby': curve,             # (n_neurons, nbps)
        'n_norm_trials': n_norm_trials,
    }, None


def _prepare_session(session_data, config):
    """`_prepare_session_with_reason` without the reason: the prep dict, or None."""
    prep, _ = _prepare_session_with_reason(session_data, config)
    return prep


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


def _smooth_circular(x, sigma):
    """El-Gaby's `smooth_circular`: Gaussian filter on the tripled array, middle third kept."""
    from scipy.ndimage import gaussian_filter1d
    x = np.asarray(x, dtype=float)
    n = len(x)
    return gaussian_filter1d(np.concatenate([x, x, x]), sigma)[n:2 * n]


def elgaby_phase_curve(neuron_raw, trial_times, config, smooth_sigma=None):
    """(n_neurons, nbps) goal-progress curve per neuron for ONE session: the trials normalised to
    (trials, 360) and averaged over trials and states. This is the input to his
    `tuning_z_allphases` (Figure2.ipynb cell 31). Also returns the number of complete trials it
    averaged over (0 if none), which `pref_phase_source='train'` uses as the weight when pooling
    sessions. `smooth_sigma` applies his `smooth_circular` to each trial's 360-bin curve first.
    """
    nstates, nbps = config.num_task_states, config.num_bins_per_state
    n = neuron_raw.shape[0]
    curve = np.full((n, nbps), np.nan)
    n_trials = 0
    for ni in range(n):
        per_trial = raw_to_norm(neuron_raw[ni], trial_times, config, return_mean=False)
        if per_trial is None:
            continue
        n_trials = per_trial.shape[0]
        if smooth_sigma:
            per_trial = np.apply_along_axis(_smooth_circular, 1, per_trial, smooth_sigma)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            curve[ni] = np.nanmean(per_trial.reshape(n_trials, nstates, nbps), axis=(0, 1))
    return curve, n_trials


def pref_phase_from_curve(curve, config):
    """His rule: within each third of the 90-bin curve take the PEAK bin; argmax over thirds.
    Returns -1 where the curve is all-NaN. Accepts one curve or a stack of them."""
    curve = np.atleast_2d(np.asarray(curve, dtype=float))
    labels = _phase_label_per_norm_bin(config)[:config.num_bins_per_state]
    nph = config.num_goal_progress_bins
    peaks = np.full((curve.shape[0], nph), np.nan)
    for p in range(nph):
        sub = curve[:, labels == p]
        ok = np.any(np.isfinite(sub), axis=1)
        if ok.any():
            peaks[ok, p] = np.nanmax(sub[ok], axis=1)
    pref = np.full(curve.shape[0], -1, dtype=int)
    ok = np.any(np.isfinite(peaks), axis=1)
    if ok.any():
        pref[ok] = np.nanargmax(peaks[ok], axis=1)
    return pref


def session_pref_phases(neuron_raw, trial_times, config, locs=None, phases=None, method=None):
    """Per-neuron preferred goal-progress phase for ONE session, by `method` (default
    `config.pref_phase_method`).

    'elgaby_peak'  `pref_phase_from_curve(elgaby_phase_curve(...))` -- his
                   `tuning_phase_boolean_max`.
    'raw_mean'     argmax of the raw time-weighted mean rate per phase over tracked node bins
                   (the v4 rule); needs `locs` (NaN = untracked/edge) and `phases` aligned to
                   `neuron_raw`'s bins.
    """
    method = config.pref_phase_method if method is None else method
    if method == 'elgaby_peak':
        curve, _ = elgaby_phase_curve(neuron_raw, trial_times, config,
                                      smooth_sigma=config.pref_phase_smooth_sigma)
        return pref_phase_from_curve(curve, config)
    if locs is None or phases is None:
        raise ValueError("pref_phase_method='raw_mean' needs `locs` and `phases`")
    L = min(len(locs), len(phases), neuron_raw.shape[1])
    keep = ~np.isnan(np.asarray(locs[:L], dtype=float))
    prefs, _ = _pref_phase_from(np.asarray(neuron_raw[:, :L], dtype=float).T[keep],
                                np.asarray(phases[:L])[keep], config.num_goal_progress_bins)
    return prefs


def run_cross_validated_regression_v5(data_dic, mouse_recday, config, valid_sessions=None,
                                      verbose=False):
    """Leave-one-session-out raw-time anchoring regression for one mouse_recday.

    Returns a results dict; None if fewer than 2 usable sessions.
    """
    sessions = list(data_dic[mouse_recday].keys()) if valid_sessions is None else list(valid_sessions)
    preps, skipped = {}, {}
    for s in sessions:
        if s == 'valid_sessions':
            continue
        if s not in data_dic[mouse_recday]:
            skipped[s] = 'not in data_dic'
            continue
        p, reason = _prepare_session_with_reason(data_dic[mouse_recday][s], config)
        if p is None:
            skipped[s] = reason
            if verbose:
                print(f"  {mouse_recday} session {s}: skipped ({reason})")
            continue
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
    corrs_nonzero_strict = np.full((n_neurons, n_folds), np.nan)
    # Poisson only: the SAME three quantities under the other link, so his linear readout and
    # the paper's log link are both available from one fit instead of two 2-hour runs.
    alt_link = ('log' if config.poisson_link == 'linear' else 'linear') if config.use_poisson else None
    corrs_altlink = np.full((n_neurons, n_folds), np.nan)
    corrs_nonzero_altlink = np.full((n_neurons, n_folds), np.nan)
    corrs_nonzero_strict_altlink = np.full((n_neurons, n_folds), np.nan)
    corrs_max = np.full((n_neurons, n_folds), np.nan)
    corrs_nonzero_max = np.full((n_neurons, n_folds), np.nan)
    cv_intercepts = np.full((n_neurons, n_folds), np.nan)
    pref_phases = np.full((n_neurons, n_folds), -1, dtype=int)
    n_nonzero_betas = np.full((n_neurons, n_folds), np.nan)

    cv_actual_tuning = np.full((n_neurons, n_folds, config.total_bins), np.nan, dtype=np.float32)
    cv_predicted_tuning = np.full((n_neurons, n_folds, config.total_bins), np.nan, dtype=np.float32)
    cv_predicted_nz_tuning = np.full((n_neurons, n_folds, config.total_bins), np.nan, dtype=np.float32)
    cv_predicted_nz_strict_tuning = np.full((n_neurons, n_folds, config.total_bins), np.nan,
                                            dtype=np.float32)
    cv_tuning_correlations = np.full((n_neurons, n_folds), np.nan)
    cv_tuning_correlations_pref = np.full((n_neurons, n_folds), np.nan)
    # Poisson only: the other link's predicted curves too, so both readouts can be drawn and
    # the cross-mouse summary can carry one row of panels per link.
    alt_curves = ({k: np.full((n_neurons, n_folds, config.total_bins), np.nan, dtype=np.float32)
                   for k in ('cv_predicted_tuning_altlink', 'cv_predicted_nz_tuning_altlink',
                             'cv_predicted_nz_strict_tuning_altlink')}
                  if alt_link is not None else {})

    state4 = {k: np.full((n_neurons, n_folds, nstates), np.nan, dtype=np.float32)
              for k in ('actual_mean', 'actual_max', 'predicted_mean', 'predicted_max',
                        'predicted_nz_mean', 'predicted_nz_max',
                        'predicted_nz_strict_mean', 'predicted_nz_strict_max')}

    # lag columns zeroed for the `corrs_nonzero` prediction (default {11, 0, 1})
    lag_axis = np.arange(n_reg) % num_lags
    zero_cols = np.isin(lag_axis, np.asarray(config.nonzero_lag_zero_lags))
    # His third saved array: the 90-degree ("one whole state") exclusion, emitted every run
    # alongside the 30-degree one, mirroring Predicted_Actual_correlation_{,nonzero_,
    # nonzero_strict_}mean.
    zero_cols_strict = np.isin(lag_axis, np.asarray(config.nonzero_lag_zero_lags_strict))

    phase_norm = _phase_label_per_norm_bin(config)

    # Target scaling, ONE scalar per neuron for the whole recday (config.y_scaling).
    # Computed over every 25 ms bin of every used session at once, deliberately NOT per session:
    # a per-session z-score would divide out genuine rate differences between the tasks of a
    # recday, which is exactly the between-session structure the fits should see. Only the fit
    # target is transformed (`yfit` below); `preps[s]['neuron']` is left in raw counts so the
    # state test, the preferred phases and the actual tuning curves are unchanged -- all three
    # are invariant to a positive per-neuron affine transform anyway (synthetic control 12).
    Nall = np.vstack([preps[s]['neuron'] for s in used])          # (all bins, n_neurons)
    n_spikes_recday = Nall.sum(axis=0)
    n_bins_recday = int(Nall.shape[0])
    y_scale_mean = Nall.mean(axis=0)
    y_scale_sd = Nall.std(axis=0)                                 # ddof 0, as sklearn uses
    del Nall
    # A neuron silent across the whole recday has sd 0 and cannot be scaled; its fold is left
    # NaN, which is what a constant target already produced.
    n_sd_zero = int(np.sum(y_scale_sd == 0))
    if config.y_scaling != 'none' and n_sd_zero and verbose:
        print(f"  {mouse_recday}: {n_sd_zero}/{n_neurons} neurons silent across the recday "
              f"(sd = 0), not fit under y_scaling={config.y_scaling!r}")

    # state-tuning: a neuron is tuned if it passes in ANY used session. Deliberately includes
    # the held-out fold so the neuron set is identical across folds; `state_tuning_train_only`
    # in run_and_summarise_all_mice_v5 reports whether that choice moves the headline.
    # Per-SESSION masks, kept rather than OR-ed away. The paper subsets to neurons
    # "state-tuned in more than one-third of the recorded tasks"; V4 originally used the OR
    # (tuned in >= 1 task), which passes 95% of units. Storing the fraction means any
    # threshold can be re-applied later without a re-fit.
    # The other statistic is always computed as well: 'max' (the paper's text) when the run
    # uses his 'pref_phase_mean' or 'mean', and 'mean' when the run uses 'max'.
    alt_cfg = copy.copy(config)
    alt_cfg.state_tuning_statistic = 'mean' if config.state_tuning_statistic == 'max' else 'max'
    # Which preferred-phase rule drives the fit rows, the scoring bins and his state test.
    pref_key = 'pref_phase_elgaby' if config.pref_phase_method == 'elgaby_peak' else 'pref_phase_raw'
    session_pref_raw = np.stack([preps[s]['pref_phase_raw'] for s in used], axis=1)
    session_pref_elgaby = np.stack([preps[s]['pref_phase_elgaby'] for s in used], axis=1)
    tuned_per_session = np.zeros((n_neurons, len(used)), dtype=bool)
    tuned_alt_per_session = np.zeros((n_neurons, len(used)), dtype=bool)
    tuning_pref_state = np.full((n_neurons, len(used)), -1, dtype=int)
    for si, s in enumerate(used):
        m, pr = identify_state_tuned_neurons_raw(
            preps[s]['neuron_raw'], preps[s]['trial_times'], config, return_pref=True,
            pref_phases=preps[s][pref_key])
        tuned_per_session[:, si] = m
        tuning_pref_state[:, si] = pr
        tuned_alt_per_session[:, si] = identify_state_tuned_neurons_raw(
            preps[s]['neuron_raw'], preps[s]['trial_times'], alt_cfg,
            pref_phases=preps[s][pref_key])
    tuned_fraction = tuned_per_session.mean(axis=1)
    tuned_any = tuned_per_session.any(axis=1)
    tuned_mask = tuned_fraction > config.state_tuning_min_fraction
    tuned_alt = (tuned_alt_per_session.mean(axis=1) > config.state_tuning_min_fraction)

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
            prefs = test[pref_key]
        elif config.pref_phase_method == 'elgaby_peak':
            # his curve pooled over the TRAINING sessions (trial-weighted) -- no leakage
            w = np.array([preps[s]['n_norm_trials'] for s in train], dtype=float)
            curves = np.stack([preps[s]['phase_curve_elgaby'] for s in train])  # (n_tr, n, 90)
            finite = np.isfinite(curves)
            num = np.sum(np.where(finite, curves, 0.0) * w[:, None, None], axis=0)
            den = np.sum(finite * w[:, None, None], axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                pooled_curve = np.where(den > 0, num / den, np.nan)
            prefs = pref_phase_from_curve(pooled_curve, config)
        else:
            prefs, _ = _pref_phase_from(Ntr, Ptr, num_phases)
        prefs = np.asarray(prefs, dtype=int)
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
            if pref < 0:                                  # no curve for this neuron/session
                continue
            rows = rows_by_phase[pref]
            Xfit = X_by_phase[pref]
            if Xfit.shape[0] < 10:
                continue

            yfit = Ntr[:, ni] if rows is None else Ntr[rows, ni]
            if config.y_scaling == 'zscore_recday':
                if not y_scale_sd[ni] > 0:
                    continue
                yfit = (yfit - y_scale_mean[ni]) / y_scale_sd[ni]
            coeffs, intercept = fit_regression_v5(Xfit, yfit, config, return_intercept=True)
            cv_coeffs[ni, fold] = coeffs
            cv_intercepts[ni, fold] = intercept
            if np.all(np.isnan(coeffs)):
                continue
            n_nonzero_betas[ni, fold] = int(np.sum(np.abs(coeffs) > 1e-9))

            eta = test_reg @ coeffs                       # the linear predictor, all betas
            if config.require_positive_mean_prediction and not (np.nanmean(eta) > 0):
                # his cell-26 gate: a fold is scored only if nanmean(prediction) > 0. For
                # ElasticNet(positive=True) this is the existing std > 0 requirement; for
                # Poisson it drops folds whose linear predictor has a negative mean.
                continue

            actual_norm = raw_to_norm(test['neuron'][:, ni], test_tt, config)
            pred_norm = raw_to_norm(apply_link(eta, intercept, config), test_tt, config)
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

            if alt_link is not None:
                alt_cfg_link = copy.copy(config)
                alt_cfg_link.poisson_link = alt_link
                pred_alt = raw_to_norm(apply_link(eta, intercept, alt_cfg_link), test_tt, config)
                if pred_alt is not None:
                    alt_curves['cv_predicted_tuning_altlink'][ni, fold] = pred_alt
                    ra_m, ra_x, *_ = _state_pref_corr(actual_norm, pred_alt, phase_norm, pref,
                                                      nbps, nstates)
                    corrs_altlink[ni, fold] = ra_m if config.state_reduce == 'mean' else ra_x

            for cols, tag, store_r in ((zero_cols, 'nz', True),
                                       (zero_cols_strict, 'nz_strict', False)):
                coeffs_nz = coeffs.copy()
                coeffs_nz[cols] = 0
                if alt_link is not None:
                    pa = raw_to_norm(apply_link(test_reg @ coeffs_nz, intercept, alt_cfg_link),
                                     test_tt, config)
                    if pa is not None:
                        alt_curves['cv_predicted_nz_tuning_altlink' if store_r
                                   else 'cv_predicted_nz_strict_tuning_altlink'][ni, fold] = pa
                        rza_m, rza_x, *_ = _state_pref_corr(actual_norm, pa, phase_norm, pref,
                                                            nbps, nstates)
                        rza = rza_m if config.state_reduce == 'mean' else rza_x
                        if store_r:
                            corrs_nonzero_altlink[ni, fold] = rza
                        else:
                            corrs_nonzero_strict_altlink[ni, fold] = rza
                pred_norm_nz = raw_to_norm(
                    apply_link(test_reg @ coeffs_nz, intercept, config), test_tt, config)
                if pred_norm_nz is None:
                    continue
                rz_m, rz_x, _, pz_m, _, pz_x = _state_pref_corr(
                    actual_norm, pred_norm_nz, phase_norm, pref, nbps, nstates)
                rz = rz_m if config.state_reduce == 'mean' else rz_x
                if store_r:
                    cv_predicted_nz_tuning[ni, fold] = pred_norm_nz
                    corrs_nonzero[ni, fold] = rz
                    corrs_nonzero_max[ni, fold] = rz_x
                    state4['predicted_nz_mean'][ni, fold] = pz_m
                    state4['predicted_nz_max'][ni, fold] = pz_x
                else:
                    cv_predicted_nz_strict_tuning[ni, fold] = pred_norm_nz
                    corrs_nonzero_strict[ni, fold] = rz
                    state4['predicted_nz_strict_mean'][ni, fold] = pz_m
                    state4['predicted_nz_strict_max'][ni, fold] = pz_x

        if verbose:
            print(f"  fold {fold + 1}/{n_folds} (test session {test_s}) done")

    results = {
        'mouse_recday': mouse_recday,
        'used_sessions': used,
        'lag_direction': config.lag_direction,
        'cv_coeffs': cv_coeffs,
        'corrs': corrs,
        'corrs_nonzero': corrs_nonzero,
        'corrs_nonzero_strict': corrs_nonzero_strict,
        'corrs_max': corrs_max,
        'corrs_nonzero_max': corrs_nonzero_max,
        'cv_intercepts': cv_intercepts,
        'poisson_link': config.poisson_link,
        'poisson_alt_link': alt_link,
        'corrs_altlink': corrs_altlink,
        'corrs_nonzero_altlink': corrs_nonzero_altlink,
        'corrs_nonzero_strict_altlink': corrs_nonzero_strict_altlink,
        'mean_corrs_altlink': np.nanmean(corrs_altlink, axis=1),
        'mean_corrs_nonzero_altlink': np.nanmean(corrs_nonzero_altlink, axis=1),
        'mean_corrs_nonzero_strict_altlink': np.nanmean(corrs_nonzero_strict_altlink, axis=1),
        'mean_corrs': np.nanmean(corrs, axis=1),
        'mean_corrs_nonzero': np.nanmean(corrs_nonzero, axis=1),
        'mean_corrs_nonzero_strict': np.nanmean(corrs_nonzero_strict, axis=1),
        'mean_corrs_max': np.nanmean(corrs_max, axis=1),
        'pref_phases': pref_phases,
        'n_nonzero_betas': n_nonzero_betas,
        'state_tuned_mask': tuned_mask,             # fraction of tasks > state_tuning_min_fraction
        'state_tuned_mask_any': tuned_any,          # the old OR (tuned in >= 1 task)
        'state_tuned_fraction': tuned_fraction,     # re-threshold post hoc without a re-fit
        'state_tuned_per_session': tuned_per_session,
        'state_tuned_mask_alt': tuned_alt,          # the other state_tuning_statistic
        'tuning_pref_state': tuning_pref_state,
        'state_duration_ratio': float(np.median(dur_ratios)),
        'frac_pref_state_is_shortest': frac_pref_shortest,
        'cv_actual_tuning': cv_actual_tuning,
        'cv_predicted_tuning': cv_predicted_tuning,
        'cv_predicted_nz_tuning': cv_predicted_nz_tuning,
        'cv_predicted_nz_strict_tuning': cv_predicted_nz_strict_tuning,
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
    results.update(alt_curves)
    results['sessions_skipped'] = skipped
    results['pref_phase_method'] = config.pref_phase_method
    # Target scaling, stored for every run (raw runs too) so the rate dependence of a fixed
    # alpha can be examined post hoc without a re-fit.
    results['y_scaling'] = config.y_scaling
    results['y_scale_mean'] = y_scale_mean
    results['y_scale_sd'] = y_scale_sd
    results['n_spikes_recday'] = n_spikes_recday
    results['n_bins_recday'] = n_bins_recday
    results['n_neurons_sd_zero'] = n_sd_zero
    # both preferred-phase rules per session, so their agreement is a standing diagnostic
    results['session_pref_phase_raw'] = session_pref_raw
    results['session_pref_phase_elgaby'] = session_pref_elgaby

    nz_mask, peak_lags, votes = identify_nonzero_lag_neurons(results, config, return_votes=True)
    results['nonzero_lag_mask'] = nz_mask
    results['peak_lags'] = peak_lags
    # the 90-degree ("one whole state") mask, from the same betas
    strict_cfg = copy.copy(config)
    strict_cfg.nonzero_lag_min = config.nonzero_lag_min_strict
    strict_cfg.nonzero_lag_max = config.nonzero_lag_max_strict
    strict_cfg.nonzero_lag_zero_lags = config.nonzero_lag_zero_lags_strict
    m_s, p_s, votes_s = identify_nonzero_lag_neurons(results, strict_cfg, return_votes=True)
    results['nonzero_lag_mask_strict'] = m_s
    results['peak_lags_strict'] = p_s
    results['nz_fold_votes'] = votes                      # NaN = that fold abstained
    results['nz_fold_votes_strict'] = votes_s
    results['n_informative_folds'] = np.sum(~np.isnan(votes), axis=1)
    add_fold_semantics(results, config)
    return results


def _config_from_results(results, **overrides):
    """Rebuild a `RegressionConfigV5` from `results['config']` -- the live object while a run
    is in memory, the plain dict an npz stores -- tolerating keys added since the run."""
    cfg = results.get('config')
    stored = dict(cfg.to_dict()) if hasattr(cfg, 'to_dict') else dict(cfg or {})
    allowed = RegressionConfigV5.__init__.__code__.co_varnames
    init = {k: v for k, v in stored.items() if k in allowed and k != 'self'}
    init.update(overrides)
    return RegressionConfigV5(**init)


def add_fold_semantics(results, config=None):
    """El-Gaby's fold semantics, derived post hoc from the stored per-fold arrays (idempotent).

    His cell 26 has no neuron mask: a fold whose top-3 betas touch the excluded lags gets NaN,
    the neuron's value is the nanmean over the SURVIVING folds, and a neuron counts if >= 1 fold
    survived. His "all state-tuned neurons (with non-zero beta coefficients)" pool is per panel:
    cell 38 calls `remove_nan` separately on each correlation array, so a neuron enters a panel
    if it is state-tuned and >= 1 fold gave a finite value for THAT panel. `n_finite_folds` is
    still exported -- it is a real diagnostic, and it defines the superseded
    `semantics='elgaby_everyfold'` -- but it is not part of the pool. Adds

        n_finite_folds                        finite `corrs` per neuron
        n_passing_folds[_strict]              folds with vote == 1 and a finite reduced-beta r
        mean_corrs_nonzero[_strict]_passing   nanmean over those folds only (his per-neuron value)
        ..._altlink                           the same for the other Poisson link, when stored

    Works on a v4 export as well (`nz_fold_votes_strict` is recomputed from `cv_coeffs` when
    absent), which is how the reconciliation table was pinned to the stored PFC run.
    """
    config = config or _config_from_results(results)
    if 'nz_fold_votes_strict' not in results:
        strict_cfg = copy.copy(config)
        strict_cfg.nonzero_lag_min = config.nonzero_lag_min_strict
        strict_cfg.nonzero_lag_max = config.nonzero_lag_max_strict
        strict_cfg.nonzero_lag_zero_lags = config.nonzero_lag_zero_lags_strict
        _, _, results['nz_fold_votes_strict'] = identify_nonzero_lag_neurons(
            results, strict_cfg, return_votes=True)
    corrs = np.asarray(results['corrs'], dtype=float)
    results['n_finite_folds'] = np.sum(np.isfinite(corrs), axis=1)
    if 'corrs_altlink' in results:
        results['n_finite_folds_altlink'] = np.sum(
            np.isfinite(np.asarray(results['corrs_altlink'], dtype=float)), axis=1)
    for tag, vkey, ckey in (('', 'nz_fold_votes', 'corrs_nonzero'),
                            ('_strict', 'nz_fold_votes_strict', 'corrs_nonzero_strict')):
        votes = np.asarray(results[vkey], dtype=float)
        for suffix in ('', '_altlink'):
            key = ckey + suffix
            if key not in results:
                continue
            c = np.asarray(results[key], dtype=float)
            if suffix and not np.any(np.isfinite(c)):
                continue
            ok = (votes == 1) & np.isfinite(c)
            n_ok = ok.sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                mean_ok = np.where(n_ok > 0,
                                   np.where(ok, c, 0.0).sum(axis=1) / np.maximum(n_ok, 1),
                                   np.nan)
            if not suffix:
                results[f'n_passing_folds{tag}'] = n_ok
            results[f'mean_corrs_nonzero{tag}_passing{suffix}'] = mean_ok
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
    'cv_predicted_nz_strict_tuning',
    'cv_actual_mean_state4', 'cv_actual_max_state4',
    'cv_predicted_mean_state4', 'cv_predicted_max_state4',
    'cv_predicted_nz_mean_state4', 'cv_predicted_nz_max_state4',
    'cv_predicted_nz_strict_mean_state4', 'cv_predicted_nz_strict_max_state4',
    'corrs', 'corrs_nonzero', 'corrs_max', 'corrs_nonzero_max',
    'cv_tuning_correlations', 'cv_tuning_correlations_pref',
    'mean_corrs', 'mean_corrs_nonzero', 'mean_corrs_max',
    'mean_tuning_correlations', 'mean_tuning_correlations_pref',
    'pref_phases', 'n_nonzero_betas', 'nz_fold_votes', 'n_informative_folds',
    'state_tuned_mask', 'state_tuned_mask_any', 'state_tuned_fraction',
    'state_tuned_per_session', 'state_tuned_mask_alt', 'tuning_pref_state',
    'cv_intercepts', 'corrs_nonzero_strict', 'mean_corrs_nonzero_strict',
    'corrs_altlink', 'corrs_nonzero_altlink', 'corrs_nonzero_strict_altlink',
    'mean_corrs_altlink', 'mean_corrs_nonzero_altlink',
    'mean_corrs_nonzero_strict_altlink',
    'nonzero_lag_mask_strict', 'peak_lags_strict',
    'nonzero_lag_mask', 'peak_lags',
    # --- v5 ---
    'nz_fold_votes_strict', 'n_finite_folds', 'n_finite_folds_altlink',
    'n_passing_folds', 'n_passing_folds_strict',
    'mean_corrs_nonzero_passing', 'mean_corrs_nonzero_strict_passing',
    'mean_corrs_nonzero_passing_altlink', 'mean_corrs_nonzero_strict_passing_altlink',
    'cv_predicted_tuning_altlink', 'cv_predicted_nz_tuning_altlink',
    'cv_predicted_nz_strict_tuning_altlink',
    'session_pref_phase_raw', 'session_pref_phase_elgaby',
    # target scaling: stored for raw runs too, as the rate diagnostic for a fixed alpha
    'y_scale_mean', 'y_scale_sd', 'n_spikes_recday',
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
    payload['sessions_skipped'] = np.asarray(
        {str(k): v for k, v in results.get('sessions_skipped', {}).items()}, dtype=object)
    payload['config'] = np.asarray(config.to_dict(), dtype=object)

    path = os.path.join(out_dir, f"{results['mouse_recday']}_{results['lag_direction']}_arrays.npz")
    np.savez_compressed(path, **payload)
    return path


def estimator_name(config):
    """'poisson' | 'elasticnet' | 'linear' -- the branch `fit_regression_v5` will take."""
    if config.use_poisson:
        return 'poisson'
    return 'elasticnet' if config.regularize else 'linear'


def run_tag(config):
    """`'_zscore'` for a recday-z-scored target, `''` for the raw-count reproduction.

    Part of the run-directory name because it changes which neurons survive the fixed alpha (an
    all-zero-fit fraction of 59% vs 1%), so a z-scored run must never be mistaken for the
    reproduction in a folder listing.
    """
    tag = '_zscore' if getattr(config, 'y_scaling', 'none') != 'none' else ''
    # Poisson penalty mixing (2026-09-14): `_l1` for the lasso, `_en<ratio>` for an elastic
    # net, nothing for the L2 reference so existing directory names are unchanged.
    l1 = float(getattr(config, 'poisson_l1_ratio', 0.0)) if getattr(config, 'use_poisson', False) else 0.0
    if l1 >= 1.0:
        tag += '_l1'
    elif l1 > 0.0:
        tag += f'_en{l1:g}'
    return tag


def run_dir_name(config, stamp=None, prefix='', version='v5'):
    """Output directory name for a run: `{estimator}{tag}_{version}_{direction}_{stamp}`.

    Leading with the estimator because it is the setting that most changes the numbers -- the
    Poisson branch never zeroes a coefficient while ElasticNet at the default alpha zeroes
    ~60% of neurons -- and because the old hard-coded `elasticnet_v4_*` name was written even
    for Poisson runs. Everything else about the run is in `run_config.json` beside the data.
    """
    import datetime
    if stamp is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return (f'{prefix}{estimator_name(config)}{run_tag(config)}_{version}_'
            f'{config.lag_direction}_{stamp}')


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
                            'n_neurons': int(len(res['nonzero_lag_mask'])),
                            'n_neurons_sd_zero': int(res.get('n_neurons_sd_zero', 0)),
                            'sessions_skipped': {str(k): v for k, v in
                                                 (res.get('sessions_skipped') or {}).items()}}

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
    for k in ('config', 'sessions_skipped'):
        if k in out and getattr(out[k], 'ndim', 1) == 0:
            out[k] = out[k].item()
    return out


# ============================================================================
# Plots
# ============================================================================

def _lag_xlabel(config):
    if config.lag_direction == 'future':
        return f'Lag (0 = current -> {config.num_lags - 1} = furthest ahead)'
    return f'Lag (0 = current -> {config.num_lags - 1} = oldest)'


#: GridMaze named colours (see .claude/skills/gridmaze-colors). Actual vs predicted is the
#: paper's blue/orange pair, upgraded to Classic Blue / Viva Magenta; the reduced-beta
#: predictions take the warm ramp; the other Poisson link is Saffron; non-preferred thirds are
#: Stone because they are structure, not data.
POLAR_COLORS = {
    'actual': '#0F4C81',               # Classic Blue
    'predicted': '#BE3455',            # Viva Magenta   (all betas)
    'predicted_nz': '#FF6F61',         # Living Coral   (reduced betas, 30-degree set)
    'predicted_nz_strict': '#FFD662',  # Aspen Gold     (reduced betas, 90-degree set)
    'predicted_alt': '#FFA500',        # Saffron        (the other Poisson link)
    'off_phase': '#B4B2A9',            # Stone
    'ink': '#2C2C2A',                  # Caviar
}
_GRIDMAZE_ARIAL = '/ceph/behrens/max_kirkby/goal_sequencing_1/font/Arial.ttf'


def _alt_link_label(config):
    """Name of the Poisson link that is NOT `config.poisson_link` (None for ElasticNet)."""
    if not config.use_poisson:
        return None
    return 'exp(Xb+b)' if config.poisson_link == 'linear' else 'linear Xb'


def _link_label(config):
    """Name of the readout actually plotted as the solid predicted curve, or '' for ElasticNet.

    A Poisson page shows two predicted curves, and "predicted" alone does not say which readout
    the solid one is -- precisely the paper-vs-code divergence the pages exist to show (his code
    correlates the linear predictor `X @ beta`; the paper's LNP implies `exp(X @ beta + b)`).
    """
    if not config.use_poisson:
        return ''
    return (' [linear Xb -- his code]' if config.poisson_link == 'linear'
            else ' [exp(Xb+b) -- paper LNP]')


def _gridmaze_rc():
    """rcParams for the GridMaze figure style (Arial, 8 pt, thin axes). Never raises."""
    rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8, 'figure.titlesize': 8,
          'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
          'legend.fontsize': 8, 'axes.linewidth': 0.8}
    try:
        from GridMaze.analysis.core.font import setup_arial_font
        setup_arial_font()
        rc['font.family'] = 'Arial'
    except Exception:
        try:
            from matplotlib import font_manager
            if os.path.exists(_GRIDMAZE_ARIAL):
                font_manager.fontManager.addfont(_GRIDMAZE_ARIAL)
                rc['font.family'] = 'Arial'
        except Exception:
            pass
    return rc


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
                l2, = ax_p.plot(bins, pred, color='red', lw=1.4, label='predicted (all betas)')
                # The reduced-beta prediction: the reference never scores a non-zero-lag
                # neuron with all its betas -- its `corrs_all_nozero` is always computed from
                # a prediction with the near-anchor lags removed. Show both.
                handles = [l1, l2]
                for key, col, lags, lab in (
                        ('cv_predicted_nz_tuning', 'darkorange',
                         config.nonzero_lag_zero_lags, '30deg'),
                        ('cv_predicted_nz_strict_tuning', 'seagreen',
                         config.nonzero_lag_zero_lags_strict, '90deg')):
                    pnz = np.nanmean(results[key][ni], axis=0)
                    if np.all(np.isnan(pnz)):
                        continue
                    ln, = ax_p.plot(bins, pnz, color=col, lw=1.1, ls='--',
                                    label=f'pred, no lag {",".join(map(str, lags))} ({lab})')
                    handles.append(ln)
                # Poisson: the other link's readout too (his linear Xb vs the paper's exp)
                alt_title = ''
                if 'cv_predicted_tuning_altlink' in results:
                    palt = np.nanmean(results['cv_predicted_tuning_altlink'][ni], axis=0)
                    if not np.all(np.isnan(palt)):
                        ln, = ax_p.plot(bins, palt, color=POLAR_COLORS['predicted_alt'], lw=1.1,
                                        ls=':', label=f'pred, other link {_alt_link_label(config)}')
                        handles.append(ln)
                        alt_title = (f"   other link {_alt_link_label(config)} "
                                     f"r={results['mean_corrs_altlink'][ni]:.3f}")
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
                ax_c.set_title(f'360-bin r={r_full:.3f}   pref-phase-only r={r_pref:.3f}   '
                               f"n=4 r={results['mean_corrs'][ni]:.3f} / "
                               f"{results['mean_corrs_nonzero'][ni]:.3f} (30deg) / "
                               f"{results['mean_corrs_nonzero_strict'][ni]:.3f} (90deg)"
                               f"{alt_title}   (y-axes scaled independently)", fontsize=7)
                if row == 0:
                    ax_c.legend(handles=handles, fontsize=5.5, loc='upper right', ncol=3)

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
                        for key, col in (('cv_predicted_nz_tuning', 'darkorange'),
                                         ('cv_predicted_nz_strict_tuning', 'seagreen')):
                            pnz = results[key][ni, fi]
                            if not np.all(np.isnan(pnz)):
                                axp.plot(bins, pnz, color=col, lw=0.9, ls='--')
                        if 'cv_predicted_tuning_altlink' in results:
                            palt = results['cv_predicted_tuning_altlink'][ni, fi]
                            if not np.all(np.isnan(palt)):
                                axp.plot(bins, palt, color=POLAR_COLORS['predicted_alt'],
                                         lw=0.9, ls=':')
                        axp.tick_params(axis='y', labelsize=4.5, labelcolor='red')
                        ax.set_xlim(0, config.total_bins)
                        ax.set_xticks(np.arange(nstates) * nbps + nbps / 2)
                        ax.set_xticklabels(list('ABCD')[:nstates], fontsize=5)
                        ax.tick_params(labelsize=4.5)
                        ax.tick_params(axis='y', labelcolor='tab:blue')
                    held = sessions[fi] if fi < len(sessions) else fi
                    r4 = results['corrs'][ni, fi]
                    r4z = results['corrs_nonzero'][ni, fi]
                    r4s = results['corrs_nonzero_strict'][ni, fi]
                    alt_txt = ''
                    if 'corrs_altlink' in results and np.isfinite(results['corrs_altlink'][ni, fi]):
                        alt_txt = (f" | {_alt_link_label(config)} "
                                   f"r={results['corrs_altlink'][ni, fi]:.2f}")
                    ax.set_title(f'held-out {held} | pref {prefs[fi]}\n'
                                 f'r={r4:.2f} / {r4z:.2f} (30d) / {r4s:.2f} (90d){alt_txt}',
                                 fontsize=5.5)

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
                z4 = results[f'cv_predicted_nz_{reduce_key}_state4'][ni]
                for arr, col, lab, ls in ((a4, 'tab:blue', 'actual', '-'),
                                          (p4, 'red', 'predicted', '-'),
                                          (z4, 'darkorange', 'pred (no lag)', '--')):
                    m = np.nanmean(arr, axis=0)
                    if np.all(np.isnan(m)):
                        continue
                    ax.plot(x, m / (np.nanmax(np.abs(m)) or 1.0), color=col, marker='o',
                            ms=3, lw=1.4, ls=ls, label=lab)
                ax.set_xticks(x)
                ax.set_xticklabels(list('ABCD')[:nstates], fontsize=5)
                ax.tick_params(labelsize=4.5)
                ax.set_title(f"N{ni}  mean n=4 r={results['mean_corrs'][ni]:.3f} / "
                             f"{results['mean_corrs_nonzero'][ni]:.3f} no-lag\n"
                             f"NZ-lag={bool(results['nonzero_lag_mask'][ni])} "
                             f"peak lag={results['peak_lags'][ni]:.0f}", fontsize=5.5)
                if row == 0:
                    ax.legend(fontsize=4.5, loc='upper right')

            fig.suptitle(f"{results['mouse_recday']} - {results['lag_direction']} lags - "
                         f"actual vs predicted per fold (each fold = a different held-out task)"
                         f"{title_extra}"
                         + (f"\npredicted{_link_label(config)}" if config.use_poisson else ''),
                         fontsize=8, y=0.999)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    return out_pdf


def _polar_curve(ax, curve, color, lw=1.0, ls='-', alpha=1.0, smooth_sigma=10, label=None):
    """Draw a 360-bin task-space curve on a polar axis, closed. Returns (line, smoothed max)."""
    y = np.asarray(curve, dtype=float)
    if np.all(np.isnan(y)):
        return None, 0.0
    y = np.where(np.isnan(y), np.nanmean(y), y)
    if smooth_sigma:
        y = _smooth_circular(y, smooth_sigma)
    theta = np.linspace(0, 2 * np.pi, len(y), endpoint=False)
    line, = ax.plot(np.r_[theta, theta[0]], np.r_[y, y[0]], color=color, lw=lw, ls=ls,
                    alpha=alpha, label=label)
    return line, float(np.max(y))


def _polar_axis_style(ax, nstates, rmax, phase_norm, pref):
    """A at the top, clockwise, state ticks only, non-preferred thirds shaded (Stone).

    The shaded wedges are the bins where the prediction is exactly zero by construction (the
    neuron is fit and scored only in its preferred third of each state). They are drawn with one
    fixed radius and the r-limit is set AFTERWARDS: polar axes autoscale after every `bar`, so
    reading `get_rmax()` inside the loop made each successive wedge 5% taller than the last.
    """
    R = rmax * 1.08 if rmax > 0 else 1.0
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(nstates) * 2 * np.pi / nstates)
    ax.set_xticklabels(list('ABCD')[:nstates])
    ax.set_yticks([])
    ax.grid(True, lw=0.4, color=POLAR_COLORS['off_phase'], alpha=0.6)
    ax.spines['polar'].set_linewidth(0.6)
    if pref is not None and pref >= 0:
        off = (np.asarray(phase_norm) != pref).astype(int)
        n = len(off)
        starts = list(np.flatnonzero(np.diff(np.r_[0, off]) == 1))
        ends = list(np.flatnonzero(np.diff(np.r_[off, 0]) == -1) + 1)
        # a block that wraps through 0 (A) is one wedge, not two half-wedges
        if len(starts) > 1 and starts[0] == 0 and ends[-1] == n:
            ends[0] += n
            starts.pop(-1)
            ends.pop(-1)
        for a, b in zip(starts, ends):
            ax.bar((a + (b - a) / 2) * 2 * np.pi / n, R, width=(b - a) * 2 * np.pi / n,
                   bottom=0, color=POLAR_COLORS['off_phase'], alpha=0.25, lw=0, zorder=0)
    ax.set_ylim(0, R)


def plot_fold_polar_pages(results, config, out_pdf, neuron_indices=None, per_page=3,
                          sort_by='tuning_corr', title_extra='', prediction='all',
                          smooth_sigma=10):
    """Paged PDF of polar actual-vs-predicted rate maps per held-out task (Fig 5g style).

    One neuron = two rows (actual above, predicted below), one column per fold, plus a final
    column with the (location x lag) betas in the modal preferred-phase frame. Angle is task
    space (A at the top, clockwise, 90 degrees per state); curves are smoothed for display with
    his `smooth_circular` (sigma 10 normalised bins) and each actual axis is annotated with its
    unsmoothed peak rate in Hz. Non-preferred thirds are shaded: the prediction is exactly zero
    there by construction, so a predicted curve can only have structure inside the clear wedge.

    `prediction` picks the SOLID predicted curve: 'all' (every beta), 'nz' (the 30-degree
    reduced-beta prediction -- what a non-zero-lag neuron's r is computed from, so the
    `_nonzerolag` pages use it) or 'nz_strict'. The other predictions are drawn dashed and
    faint. A Poisson run also draws the other link's prediction (dotted, Saffron) and puts
    both r values in the title.
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

    pred_keys = {'all': 'cv_predicted_tuning', 'nz': 'cv_predicted_nz_tuning',
                 'nz_strict': 'cv_predicted_nz_strict_tuning'}
    corr_keys = {'all': 'corrs', 'nz': 'corrs_nonzero', 'nz_strict': 'corrs_nonzero_strict'}
    mean_keys = {'all': 'mean_corrs', 'nz': 'mean_corrs_nonzero',
                 'nz_strict': 'mean_corrs_nonzero_strict'}
    if prediction not in pred_keys:
        raise ValueError(f"prediction must be one of {sorted(pred_keys)}, got {prediction!r}")
    labels = {'all': 'predicted (all betas)',
              'nz': f'predicted, no lag {",".join(map(str, config.nonzero_lag_zero_lags))}',
              'nz_strict': ('predicted, no lag '
                            f'{",".join(map(str, config.nonzero_lag_zero_lags_strict))}')}
    colour_of = {'all': 'predicted', 'nz': 'predicted_nz', 'nz_strict': 'predicted_nz_strict'}
    main_key = pred_keys[prediction]
    alt_key = main_key + '_altlink'
    has_alt = alt_key in results and np.any(np.isfinite(results[alt_key]))
    alt_corr_key = corr_keys[prediction] + '_altlink'
    alt_label = _alt_link_label(config)

    nstates = config.num_task_states
    phase_norm = _phase_label_per_norm_bin(config)
    sessions = list(results['used_sessions'])
    ncols = n_folds + 1
    bin_s = 0.025

    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)) or '.', exist_ok=True)
    with matplotlib.rc_context(_gridmaze_rc()), PdfPages(out_pdf) as pdf:
        for page_start in range(0, len(neuron_indices), per_page):
            page = neuron_indices[page_start:page_start + per_page]
            height = 4.6 * len(page) + 1.6
            fig = plt.figure(figsize=(2.2 * ncols, height))
            # fixed margins in inches for the two-line suptitle and the legend, whatever the
            # page size
            gs = fig.add_gridspec(2 * len(page), ncols, hspace=0.6, wspace=0.4,
                                  left=0.04, right=0.98, top=1 - 1.1 / height,
                                  bottom=0.8 / height)
            handles = {}
            for row, ni in enumerate(page):
                prefs = results['pref_phases'][ni]
                for fi in range(n_folds):
                    ax_a = fig.add_subplot(gs[2 * row, fi], projection='polar')
                    ax_p = fig.add_subplot(gs[2 * row + 1, fi], projection='polar')
                    actual = results['cv_actual_tuning'][ni, fi]
                    held = sessions[fi] if fi < len(sessions) else fi
                    if np.all(np.isnan(actual)) and np.all(np.isnan(results[main_key][ni, fi])):
                        for ax in (ax_a, ax_p):
                            ax.text(0.5, 0.5, 'no fit', ha='center', va='center',
                                    transform=ax.transAxes)
                            ax.set_xticks([]); ax.set_yticks([])
                        ax_a.set_title(f'held-out {held}', pad=6)
                        continue
                    ln, amax = _polar_curve(ax_a, actual / bin_s, POLAR_COLORS['actual'], lw=1.1,
                                            smooth_sigma=smooth_sigma, label='actual')
                    _polar_axis_style(ax_a, nstates, amax, phase_norm, prefs[fi])
                    if ln is not None:
                        handles.setdefault('actual', ln)
                        ax_a.text(1.0, 1.0, f'{np.nanmax(actual) / bin_s:.1f} Hz',
                                  transform=ax_a.transAxes, ha='right', va='bottom')
                    ax_a.set_title(f'held-out {held} | pref {prefs[fi]}', pad=6)

                    rmax = 0.0
                    order = [prediction] + [k for k in pred_keys if k != prediction]
                    for name in order:
                        curve = results[pred_keys[name]][ni, fi]
                        solid = name == prediction
                        ln, m = _polar_curve(ax_p, curve, POLAR_COLORS[colour_of[name]],
                                             lw=1.1 if solid else 0.7, ls='-' if solid else '--',
                                             alpha=1.0 if solid else 0.55,
                                             smooth_sigma=smooth_sigma, label=labels[name])
                        if ln is not None:
                            handles.setdefault(labels[name], ln)
                            rmax = max(rmax, m)
                    r_txt = f"r={results[corr_keys[prediction]][ni, fi]:.2f}"
                    if has_alt:
                        ln, m = _polar_curve(ax_p, results[alt_key][ni, fi],
                                             POLAR_COLORS['predicted_alt'], lw=0.9, ls=':',
                                             smooth_sigma=smooth_sigma,
                                             label=f'predicted, other link {alt_label}')
                        if ln is not None:
                            handles.setdefault(f'predicted, other link {alt_label}', ln)
                            rmax = max(rmax, m)
                        r_alt = results[alt_corr_key][ni, fi]
                        if np.isfinite(r_alt):
                            r_txt += f' | {alt_label} r={r_alt:.2f}'
                    _polar_axis_style(ax_p, nstates, rmax, phase_norm, prefs[fi])
                    ax_p.set_title(r_txt, pad=6)

                # betas in the modal frame, spanning both rows
                ax_b = fig.add_subplot(gs[2 * row:2 * row + 2, n_folds])
                B, modal, n_used, n_tot = betas_in_common_frame(results, config, ni)
                if B is None or np.all(np.isnan(B)):
                    ax_b.text(0.5, 0.5, 'no fit', ha='center', va='center', transform=ax_b.transAxes)
                    ax_b.set_xticks([]); ax_b.set_yticks([])
                else:
                    bmax = float(np.nanmax(np.abs(B))) if np.isfinite(B).any() else 0.0
                    ax_b.imshow(B, aspect='auto', cmap='viridis', interpolation='nearest',
                                vmin=0, vmax=bmax or 1.0)
                    ax_b.set_yticks(np.arange(config.num_locations))
                    ax_b.set_yticklabels([f'L{i + 1}' for i in range(config.num_locations)])
                    ax_b.set_xticks(np.arange(0, config.num_lags, 3))
                    ax_b.set_xlabel(_lag_xlabel(config))
                    for side in ('top', 'right'):
                        ax_b.spines[side].set_visible(False)
                mean_r = results[mean_keys[prediction]][ni]
                ax_b.set_title(f'N{ni}  betas [{n_used}/{n_tot} folds, pref {modal}]\n'
                               f'mean {prediction} r={mean_r:.3f}  '
                               f"NZ-lag={bool(results['nonzero_lag_mask'][ni])} "
                               f"peak lag={results['peak_lags'][ni]:.0f}", pad=6)

            if handles:
                fig.legend(handles=list(handles.values()), labels=list(handles.keys()),
                           loc='lower center', ncol=min(len(handles), 5), frameon=False)
            fig.suptitle(f"{results['mouse_recday']} - {results['lag_direction']} lags - polar "
                         f"actual vs predicted per held-out task{title_extra}\n"
                         f"solid predicted = {labels[prediction]}{_link_label(config)}"
                         f"; display smoothing "
                         f"sigma={smooth_sigma} bins; shaded = non-preferred thirds "
                         f"(prediction is exactly 0 there)", y=1 - 0.15 / height)
            with matplotlib.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0}):
                pdf.savefig(fig, bbox_inches=None)
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

def plot_cross_mouse_summary(all_results, config, save_path=None, paper_ref=True,
                             semantics=('elgaby', 'v4')):
    """The paper's three panels, pooled across recdays, one row per (link, semantics), plus the
    per-recday breakdown of that row's 30-degree panel.

    El-Gaby's Fig 5h shows the same correlation under two selections and ED Fig 8a adds a
    third; the number moves a lot between them, so one histogram is not enough. Columns:

        all state-tuned          no lag filter, all-betas correlation   (Fig 5h left)
        non-zero-lag 30 deg      excludes lags {0, 11}                  (Fig 5h right)
        non-zero-lag 90 deg      excludes {0,1,2,9,10,11}               (ED Fig 8a)

    Rows: for each stored link (a Poisson run stores his linear readout AND the paper's
    exp(Xb+b)), the 'elgaby' semantics (per-panel pool of >= 1 finite fold; >= 1 passing fold;
    passing-fold mean -- the reproduction claim) and the 'v4' semantics (majority-vote mask, all
    folds). `'elgaby_everyfold'` is the superseded variant.
    Each panel carries n, mean, t, P and the effect size t/sqrt(n), plus the published (n, t)
    for that estimator (ElasticNet: Fig 5h / ED 8a-b; Poisson: ED 8d).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if isinstance(semantics, str):
        semantics = (semantics,)
    colours = ('#888780', '#C03030', '#2A6FB5')
    excl = (None, ','.join(map(str, config.nonzero_lag_zero_lags)),
            ','.join(map(str, config.nonzero_lag_zero_lags_strict)))

    try:
        table = build_unit_table(all_results, config, require_anatomy=False)
    except Exception as exc:
        print(f'  cross-mouse summary skipped: {exc}')
        return None
    if table.empty:
        return None
    links = _table_links(table)
    est = table.attrs.get('estimator', estimator_name(config))
    paper = PAPER_PANELS.get(est, {}).get(round(float(config.state_tuning_p_threshold), 2),
                                          (None, None, None))
    rows = [(sfx, lab, sem) for sfx, lab in links for sem in semantics]

    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3.6 * len(rows)), squeeze=False)
    for r, (sfx, link_label, sem) in enumerate(rows):
        panels = _panel_columns(sem, sfx)
        for c, ((label, sel_col, val_col), colour, ex, pub) in enumerate(
                zip(panels, colours, excl, paper)):
            ax = axes[r][c]
            title = f'{label}' + (f'  (excl. {ex})' if ex else '')
            v = (table.loc[table[sel_col], val_col].dropna().to_numpy()
                 if sel_col in table.columns and val_col in table.columns else np.array([]))
            if not len(v):
                ax.text(0.5, 0.5, 'no neurons', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=7)
                continue
            ax.hist(v, bins=np.linspace(-1, 1, 41), color=colour, alpha=0.75,
                    edgecolor='black', linewidth=0.4)
            ax.axvline(0, color='k', ls='--', lw=0.8)
            ax.axvline(v.mean(), color='red', lw=1.6)
            t, p = stats.ttest_1samp(v, 0) if len(v) > 1 else (np.nan, np.nan)
            txt = (f"n={len(v)}\nmean={v.mean():+.3f}\n{np.mean(v > 0):.0%} > 0\n"
                   f"t={t:.2f}\np={p:.2g}\nt/sqrt(n)={t / np.sqrt(len(v)):.3f}")
            if paper_ref and pub:
                txt += f"\n\npaper: n={pub[0]}\n t={pub[1]}, {pub[1] / np.sqrt(pub[0]):.3f}"
            ax.text(0.97, 0.97, txt, transform=ax.transAxes, va='top', ha='right', fontsize=6,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            ax.set_xlabel('pref-phase state corr (n=4)', fontsize=7)
            ax.set_title(title, fontsize=7)
            ax.tick_params(labelsize=6)
        axes[r][0].set_ylabel(f'{link_label}\n{sem} semantics\n# neurons', fontsize=7)

        ax = axes[r][3]
        _, sel30, val30 = panels[1]
        by = (table[table[sel30]].groupby('recday')[val30].agg(['mean', 'size'])
              if 'recday' in table and sel30 in table.columns else None)
        if by is not None and len(by):
            x = np.arange(len(by))
            ax.bar(x, by['mean'], color='#C03030', alpha=0.75, edgecolor='black', linewidth=0.4)
            ax.axhline(0, color='k', ls='--', lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{i}\n(n={int(k)})" for i, k in zip(by.index, by['size'])],
                               rotation=90, fontsize=4)
            ax.set_ylabel('mean corr', fontsize=7)
        ax.set_title('Per-recday, non-zero-lag 30deg', fontsize=7)

    leak = '  [pref phase from TEST -- leakage]' if config.pref_phase_source == 'test' else ''
    ysc = ('  [target: recday z-score, NOT the reproduction]'
           if getattr(config, 'y_scaling', 'none') != 'none' else '')
    fig.suptitle(f"V5 {est} anchoring - {config.lag_direction} lags - state tuning "
                 f"'{config.state_tuning_statistic}' p<{config.state_tuning_p_threshold:g} in "
                 f">{config.state_tuning_min_fraction:.2f} of tasks - pref phase "
                 f"'{config.pref_phase_method}'{leak}{ysc}", fontweight='bold', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if len(rows) > 1 else 0.94))
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def regenerate_summary(out_dir, save=True, suffix='', semantics=('elgaby', 'v4')):
    """Redraw the cross-mouse summary for a finished run, from its exports alone.

    Plotting code changes more often than fitting code, and a sweep costs hours. This rebuilds
    the figure (and returns the three-panel table) from the `.npz` files, so a run never has to
    be repeated just to pick up a figure change.

        python -c "import elasticnet_regression_v5 as v5; v5.regenerate_summary('<run dir>')"

    `suffix` is appended to the figure's filename. Pass one (e.g. `'_corrected'`) when the
    scoring rules have changed since the run: the original figure is then left in place beside
    the new one instead of being silently replaced, so a number quoted from the old file can
    still be traced. Repo convention: regenerated outputs go to new files.

    Works on a v4 export too: the semantics columns are derived post hoc by `add_fold_semantics`.
    """
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, '*_arrays.npz'))):
        res = load_regression_outputs(path)
        runs[res['mouse_recday']] = res
    if not runs:
        raise FileNotFoundError(f'no *_arrays.npz in {out_dir}')

    stored = dict(next(iter(runs.values()))['config'])
    init = {k: v for k, v in stored.items()
            if k in RegressionConfigV5.__init__.__code__.co_varnames}
    init.pop('self', None)
    config = RegressionConfigV5(**init)

    # a run may have no anatomy (PFC) or a recday absent from unit_regions; either way the
    # summary only needs the regression columns
    original = globals()['load_unit_regions']
    globals()['load_unit_regions'] = lambda path=None, required=True: (
        None if not required else original(path))
    try:
        out = os.path.join(out_dir,
                           f'cross_mouse_v5_{config.lag_direction}_summary{suffix}.svg')
        fig = plot_cross_mouse_summary(runs, config, save_path=out if save else None,
                                       semantics=semantics)
        table = build_unit_table(runs, config, require_anatomy=False)
    finally:
        globals()['load_unit_regions'] = original
    if fig is not None:
        plt.close(fig)
    print(f'{len(runs)} recdays -> {out if save else "(not saved)"}')
    return three_panel_summary(table, semantics=semantics)


def summarise_recday(results, config):
    """Per-recday diagnostics: the numbers that decide whether a run is interpretable."""
    nz = results['nonzero_lag_mask']
    tuned = results['state_tuned_mask']
    # the paper's right-hand panel: non-zero-lag neurons scored with the REDUCED-beta r
    sel = nz & tuned & np.isfinite(results['mean_corrs_nonzero'])
    nb = results['n_nonzero_betas']
    n_pass = results.get('n_passing_folds', np.zeros(len(nz), dtype=int))
    n_fin = results.get('n_finite_folds', np.sum(np.isfinite(results['corrs']), axis=1))
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
        'mean_r_selected': (float(np.nanmean(results['mean_corrs_nonzero'][sel]))
                            if sel.any() else np.nan),
        # his semantics: any passing fold / finite in every fold
        'n_selected_anyfold': int((tuned & (n_pass > 0)).sum()),
        'n_tuned_finite_all_folds': int((tuned & (n_fin == results['num_sessions'])).sum()),
        'n_sessions_skipped': len(results.get('sessions_skipped') or {}),
        'pref_phase_method': results.get('pref_phase_method', config.pref_phase_method),
        'state_tuning_statistic': config.state_tuning_statistic,
        'frac_untracked': results['frac_untracked'],
        'frac_allzero_fits': float(np.mean(nb[fitted] == 0)) if fitted.any() else np.nan,
        'median_nonzero_betas': float(np.nanmedian(nb)) if fitted.any() else np.nan,
        'frac_pref_phase_flips': float(np.mean(n_distinct > 1)),
        'n_nonzero_lag_alt_top3': int(alt_mask.sum()),
        'state_duration_ratio': results['state_duration_ratio'],
        'frac_pref_state_is_shortest': results['frac_pref_state_is_shortest'],
        'n_state_tuned_alt_stat': int(results['state_tuned_mask_alt'].sum()),
        'n_state_tuned_any': int(results['state_tuned_mask_any'].sum()),
        'state_tuning_min_fraction': config.state_tuning_min_fraction,
        'n_nonzero_lag_strict': int(results['nonzero_lag_mask_strict'].sum()),
        'n_selected_strict': int((results['nonzero_lag_mask_strict'] & tuned
                                  & np.isfinite(results['mean_corrs_nonzero_strict'])).sum()),
        'pref_phase_source': config.pref_phase_source,
        'poisson_link': config.poisson_link,
    }


def _print_recday_summary(s):
    print(f"  neurons={s['n_neurons']} folds={s['n_folds']} | state-tuned={s['n_state_tuned']} "
          f"NZ-lag={s['n_nonzero_lag']} both(valid reduced-beta r)={s['n_selected']} "
          f"mean r={s['mean_r_selected']:.3f} | his semantics: any-fold={s['n_selected_anyfold']} "
          f"finite-all-folds={s['n_tuned_finite_all_folds']}"
          + (f" | sessions skipped={s['n_sessions_skipped']}" if s['n_sessions_skipped'] else ''))
    print(f"  diagnostics: all-zero fits={s['frac_allzero_fits']:.0%} "
          f"median non-zero betas={s['median_nonzero_betas']:.0f} | "
          f"pref-phase flips across folds={s['frac_pref_phase_flips']:.0%} | "
          f"untracked bins dropped={s['frac_untracked']:.1%} | "
          f"NZ-lag with the top-3 guard flipped={s['n_nonzero_lag_alt_top3']}")
    print(f"  state-tuning confound: leg longest:shortest = {s['state_duration_ratio']:.2f}x, "
          f"{s['frac_pref_state_is_shortest']:.0%} of units prefer the SHORTEST leg "
          f"(chance 25%), state-tuned under the other statistic={s['n_state_tuned_alt_stat']}")
    print(f"  state tuning: >{s['state_tuning_min_fraction']:.2f} of tasks -> "
          f"{s['n_state_tuned']} (the old OR over tasks would give {s['n_state_tuned_any']}) | "
          f"strict 90deg: NZ-lag={s['n_nonzero_lag_strict']} selected={s['n_selected_strict']}")
    if s['pref_phase_source'] == 'test':
        print('  *** pref_phase_source=\'test\': the HELD-OUT session chooses which bins the '
              "held-out score averages over -- LEAKAGE, matching the reference. Use 'train' "
              'for an unbiased score. ***')
    if s.get('poisson_link') == 'log':
        print("  poisson_link='log': prediction is exp(X@beta + b), the paper's LNP model "
              "(his code uses the linear predictor instead)")


def run_and_summarise_all_mice_v5(data_dic, config, valid_sessions_dic=None, save_dir=None,
                                  export_dir=None, make_pdfs=True, n_jobs=1, verbose=True,
                                  mouse_recdays=None, manifest_extra=None):
    """Run V5 for every mouse_recday, export per-neuron arrays and paged PDFs, and pool the
    reduced-beta preferred-phase state correlations for state-tuned non-zero-lag neurons.

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
            return mr, run_cross_validated_regression_v5(
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
        # Non-zero-lag neurons are scored with the REDUCED-beta prediction (excluded lags
        # zeroed), as his `Predicted_Actual_correlation_nonzero` is. v4 pooled the all-betas r
        # here, a combination the reference never computes.
        sel = (res['nonzero_lag_mask'] & res['state_tuned_mask']
               & np.isfinite(res['mean_corrs_nonzero']))
        pooled[mr] = res['mean_corrs_nonzero'][sel]

        s = summarise_recday(res, config)
        summaries.append(s)
        if verbose:
            print(f"\n{mr} [{config.lag_direction}]")
            _print_recday_summary(s)
            for sess, why in (res.get('sessions_skipped') or {}).items():
                print(f"  session {sess} skipped: {why}")

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
            plot_fold_polar_pages(res, config,
                                  os.path.join(export_dir, f'{tag}_all_polar.pdf'),
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
                # non-zero-lag pages show the REDUCED-beta prediction, i.e. what their r uses
                plot_fold_polar_pages(res, config,
                                      os.path.join(export_dir, f'{tag}_nonzerolag_polar.pdf'),
                                      neuron_indices=nz_idx, prediction='nz',
                                      title_extra='  -  non-zero-lag neurons')

        if save_dir is not None and sel.sum() > 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(pooled[mr], bins=20, color='darkorange', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='k', ls='--')
            ax.axvline(np.nanmean(pooled[mr]), color='red', lw=2,
                       label=f"mean={np.nanmean(pooled[mr]):.3f}")
            ax.set_xlabel('pref-phase state corr (n=4), reduced-beta prediction')
            ax.set_ylabel('# neurons')
            ax.set_title(f'{mr}\nstate-tuned NZ-lag ({config.lag_direction} lags)')
            ax.legend()
            fig.savefig(os.path.join(save_dir, f'{mr}_v5_{config.lag_direction}_corr.svg'),
                        bbox_inches='tight')
            plt.close(fig)

    # things decided in the NOTEBOOK (e.g. MIN_TRIALS for the session selection) come in via
    # `manifest_extra`, so the manifest records them alongside the config
    extra = dict(manifest_extra or {})
    extra.update({
        'n_jobs': n_jobs,
        'wall_seconds': round(time.time() - _t_start, 1),
        'recdays_requested': list(recdays),
        'recdays_completed': sorted(all_results),
        'recdays_failed': sorted(set(recdays) - set(all_results)),
        'sessions_skipped_total': int(sum(len(r.get('sessions_skipped') or {})
                                          for r in all_results.values())),
    })
    for target in {d for d in (export_dir, save_dir) if d is not None}:
        write_run_manifest(target, config, all_results, valid_sessions_dic, extra)

    summary_table = _summary_table(summaries)
    # `pooled` can be non-empty yet hold only empty arrays -- every recday ran, but no neuron
    # passed the selection. That is a legitimate outcome (and at the default alpha a likely one
    # on a small recday), so it must not crash the summary.
    nonempty = [v for v in pooled.values() if len(v)]
    if not nonempty:
        print('No selected neurons to summarise '
              f'({len(all_results)} recday(s) ran; check the diagnostics table).')
        return all_results, pooled, summary_table

    # The paper's three panels, pooled -- not just the single non-zero-lag histogram the
    # earlier version drew, which showed one of the three and gave no way to see how much the
    # number depends on which selection is used.
    fig = plot_cross_mouse_summary(
        all_results, config,
        save_path=(os.path.join(save_dir, f'cross_mouse_v5_{config.lag_direction}_summary.svg')
                   if save_dir is not None else None))
    if fig is not None:
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
    no_anatomy_recdays = []
    for recday, res in all_results.items():
        n = len(res['nonzero_lag_mask'])
        if has_anatomy and recday in regions:
            reg = regions[recday]
            if len(reg) != n:
                msg = (f'{recday}: unit_regions has {len(reg)} rows but the regression has '
                       f'{n} neurons')
                if strict:
                    raise AssertionError(msg)
                print('  skipping -', msg)
                continue
        elif has_anatomy and require_anatomy:
            raise KeyError(f'{recday}: no entry in unit_regions')
        else:
            # No anatomy (or a recday the anatomy table does not know, e.g. a synthetic one or
            # a PFC run scored from the LEC tree): the identity columns the rest of the module
            # needs, nothing invented.
            if has_anatomy:
                no_anatomy_recdays.append(recday)
            reg = pd.DataFrame({'recday': recday,
                                'mouse': recday.split('_')[0],
                                'order': np.arange(n),
                                'unit_id': np.arange(n)})

        # his fold semantics, derived from the per-fold arrays if this is an older export
        add_fold_semantics(res, config)
        prefs = res['pref_phases']
        modal = np.array([_modal_pref(row)[0] for row in prefs])
        n_distinct = np.array([len(set(row[row >= 0].tolist())) for row in prefs])
        n_folds_val = int(res.get('num_sessions', len(res['used_sessions'])))
        if 'session_pref_phase_raw' in res and 'session_pref_phase_elgaby' in res:
            a, b = res['session_pref_phase_raw'], res['session_pref_phase_elgaby']
            ok = (a >= 0) & (b >= 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                agreement = np.where(ok.sum(1) > 0, ((a == b) & ok).sum(1) / np.maximum(ok.sum(1), 1),
                                     np.nan)
        else:
            agreement = np.full(n, np.nan)

        block = reg.assign(
            lag_direction=res['lag_direction'],
            nonzero_lag=res['nonzero_lag_mask'],
            nonzero_lag_strict=res['nonzero_lag_mask_strict'],
            peak_lag_strict=res['peak_lags_strict'],
            mean_corr_nonzero_strict=res['mean_corrs_nonzero_strict'],
            state_tuned=res['state_tuned_mask'],
            state_tuned_fraction=res['state_tuned_fraction'],
            state_tuned_any=res['state_tuned_mask_any'],
            peak_lag=res['peak_lags'],
            mean_corr=res['mean_corrs'],
            mean_corr_nonzero=res['mean_corrs_nonzero'],
            mean_corr_max=res['mean_corrs_max'],
            tuning_corr=res['mean_tuning_correlations'],
            tuning_corr_pref=res['mean_tuning_correlations_pref'],
            n_nonzero_betas=np.nanmean(res['n_nonzero_betas'], axis=1),
            pref_phase_modal=modal,
            pref_phase_flips=n_distinct > 1,
            pref_phase_agreement=agreement,           # raw-mean vs his peak rule, per session
            n_folds=n_folds_val,
            n_informative_folds=res['n_informative_folds'],
            state_tuned_alt_stat=res['state_tuned_mask_alt'],
            nz_fold_vote_frac=np.nanmean(res['nz_fold_votes'], axis=1),
            # --- his semantics (v5) ---
            n_finite_folds=res['n_finite_folds'],
            finite_all_folds=res['n_finite_folds'] == n_folds_val,
            n_passing_folds=res['n_passing_folds'],
            n_passing_folds_strict=res['n_passing_folds_strict'],
            nonzero_lag_anyfold=res['n_passing_folds'] > 0,
            nonzero_lag_strict_anyfold=res['n_passing_folds_strict'] > 0,
            mean_corr_nonzero_passing=res['mean_corrs_nonzero_passing'],
            mean_corr_nonzero_strict_passing=res['mean_corrs_nonzero_strict_passing'],
            # target scaling: the two quantities a fixed alpha is sensitive to. Present for raw
            # runs too, so "did z-scoring admit low-spike neurons?" is answerable post hoc.
            n_spikes_recday=res.get('n_spikes_recday', np.full(n, np.nan)),
            y_sd_recday=res.get('y_scale_sd', np.full(n, np.nan)),
        )
        if 'corrs_altlink' in res and np.any(np.isfinite(np.asarray(res['corrs_altlink'],
                                                                     dtype=float))):
            block = block.assign(
                mean_corr_altlink=res['mean_corrs_altlink'],
                mean_corr_nonzero_altlink=res['mean_corrs_nonzero_altlink'],
                mean_corr_nonzero_strict_altlink=res['mean_corrs_nonzero_strict_altlink'],
                finite_all_folds_altlink=res['n_finite_folds_altlink'] == n_folds_val,
                mean_corr_nonzero_passing_altlink=res.get(
                    'mean_corrs_nonzero_passing_altlink', np.full(n, np.nan)),
                mean_corr_nonzero_strict_passing_altlink=res.get(
                    'mean_corrs_nonzero_strict_passing_altlink', np.full(n, np.nan)),
            )
        if data_dic is not None:
            rates = mean_firing_rates(data_dic, recday, res['used_sessions'])
            block = block.assign(mean_rate_hz=rates if rates is not None else np.nan)
        rows.append(block)

    if not rows:
        return pd.DataFrame()
    if no_anatomy_recdays:
        print(f'  {len(no_anatomy_recdays)} recday(s) not in unit_regions -> identity columns '
              f'(e.g. {no_anatomy_recdays[0]})')
    table = pd.concat(rows, ignore_index=True)
    # The paper reports three panels; carry all three selections so a crosstab can use any.
    # V4 semantics: per-neuron majority-vote mask, all fitted folds averaged.
    table['selected'] = (table.nonzero_lag & table.state_tuned & table.mean_corr.notna())
    table['selected_strict'] = (table.nonzero_lag_strict & table.state_tuned
                                & table.mean_corr_nonzero_strict.notna())
    table['selected_all'] = (table.state_tuned & table.mean_corr.notna())
    # El-Gaby's semantics (CORRECTED 2026-09-08, see PANEL_POOL_CORRECTION.md). His cell 38
    # applies `remove_nan` to EACH of the three correlation arrays independently, so a panel's
    # pool is "state-tuned AND >= 1 fold produced a finite value FOR THAT PANEL" -- there is no
    # shared across-panel requirement. Valued by the mean over passing folds.
    table['selected_all_eg'] = table.state_tuned & table.mean_corr.notna()
    table['selected_eg'] = (table.state_tuned & table.nonzero_lag_anyfold
                            & table.mean_corr_nonzero_passing.notna())
    table['selected_strict_eg'] = (table.state_tuned & table.nonzero_lag_strict_anyfold
                                   & table.mean_corr_nonzero_strict_passing.notna())
    # The superseded rule, kept so the numbers reported before the correction can still be
    # reproduced and explained. It required a finite r in EVERY fold before a neuron entered any
    # panel -- my inference, adopted because it reproduced the paper's 489/349, which turned out
    # to be a coincidence of ElasticNet sparsity (481 for ElasticNet, but 140 for Poisson).
    table['selected_all_eg_everyfold'] = table.selected_all_eg & table.finite_all_folds
    table['selected_eg_everyfold'] = table.selected_eg & table.finite_all_folds
    table['selected_strict_eg_everyfold'] = table.selected_strict_eg & table.finite_all_folds
    links = {'primary': 'identity', 'alt': None}
    if config.use_poisson:
        lin, log = 'linear Xb (his code)', 'exp(Xb+b) (paper LNP)'
        links = ({'primary': lin, 'alt': log} if config.poisson_link == 'linear'
                 else {'primary': log, 'alt': lin})
    if 'mean_corr_altlink' in table.columns:
        table['selected_altlink'] = (table.nonzero_lag & table.state_tuned
                                     & table.mean_corr_altlink.notna())
        table['selected_strict_altlink'] = (table.nonzero_lag_strict & table.state_tuned
                                            & table.mean_corr_nonzero_strict_altlink.notna())
        table['selected_all_altlink'] = table.state_tuned & table.mean_corr_altlink.notna()
        # his corrected per-panel rule, for the other link
        table['selected_all_eg_altlink'] = table.state_tuned & table.mean_corr_altlink.notna()
        table['selected_eg_altlink'] = (table.state_tuned & table.nonzero_lag_anyfold
                                        & table.mean_corr_nonzero_passing_altlink.notna())
        table['selected_strict_eg_altlink'] = (
            table.state_tuned & table.nonzero_lag_strict_anyfold
            & table.mean_corr_nonzero_strict_passing_altlink.notna())
        table['selected_all_eg_everyfold_altlink'] = (table.selected_all_eg_altlink
                                                      & table.finite_all_folds_altlink)
        table['selected_eg_everyfold_altlink'] = (table.selected_eg_altlink
                                                  & table.finite_all_folds_altlink)
        table['selected_strict_eg_everyfold_altlink'] = (table.selected_strict_eg_altlink
                                                         & table.finite_all_folds_altlink)
    else:
        links['alt'] = None
    table.attrs['has_anatomy'] = has_anatomy
    table.attrs['links'] = links
    table.attrs['estimator'] = estimator_name(config)
    table.attrs['y_scaling'] = getattr(config, 'y_scaling', 'none')
    table.attrs['state_tuning_p_threshold'] = config.state_tuning_p_threshold
    return table


#: Published values (n, t) for the three panels, by estimator and state-tuning threshold.
#: ElasticNet: Fig 5h left/right and ED Fig 8a (p<0.05), ED Fig 8b (p<0.01).
#: Poisson: ED Fig 8d (p<0.05). The paper's Poisson panel is described as the log-link model,
#: while his code scores the linear predictor -- so both Poisson rows are compared to ED 8d.
PAPER_PANELS = {
    'elasticnet': {0.05: ((489, 9.3), (329, 3.9), (224, 2.53)),
                   0.01: ((349, 8.70), (227, 2.83), (154, 1.94))},
    'poisson': {0.05: ((489, 10.7), (346, 4.74), (229, 2.81))},
}

_PANEL_SPECS = {
    # semantics -> [(panel label, selection column stem, value column stem)]
    'v4': (('all state-tuned', 'selected_all', 'mean_corr'),
           ('non-zero-lag 30deg', 'selected', 'mean_corr_nonzero'),
           ('non-zero-lag 90deg (strict)', 'selected_strict', 'mean_corr_nonzero_strict')),
    # His cell 38, corrected 2026-09-08: each panel pools independently on its own finiteness.
    # The label names the pool rule so a figure cannot be misread as the superseded one.
    'elgaby': (('all state-tuned, >=1 finite fold', 'selected_all_eg', 'mean_corr'),
               ('non-zero-lag 30deg, >=1 passing fold', 'selected_eg', 'mean_corr_nonzero_passing'),
               ('non-zero-lag 90deg, >=1 passing fold', 'selected_strict_eg',
                'mean_corr_nonzero_strict_passing')),
    # SUPERSEDED (see PANEL_POOL_CORRECTION.md): additionally required a finite r in every fold.
    'elgaby_everyfold': (
        ('all state-tuned, finite r EVERY fold [superseded]', 'selected_all_eg_everyfold',
         'mean_corr'),
        ('non-zero-lag 30deg, every fold finite [superseded]', 'selected_eg_everyfold',
         'mean_corr_nonzero_passing'),
        ('non-zero-lag 90deg, every fold finite [superseded]', 'selected_strict_eg_everyfold',
         'mean_corr_nonzero_strict_passing')),
}


def _panel_columns(semantics, link_suffix):
    """Column names for one (semantics, link) block of the three panels."""
    out = []
    for label, sel, val in _PANEL_SPECS[semantics]:
        out.append((label, sel + link_suffix, val + link_suffix))
    return out


def _table_links(table):
    """[(suffix, label)] for the links present in a unit table: primary first, alt if stored."""
    links = table.attrs.get('links', {'primary': 'identity', 'alt': None})
    out = [('', links.get('primary', 'identity'))]
    if links.get('alt') and 'mean_corr_altlink' in table.columns:
        out.append(('_altlink', links['alt']))
    return out


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


def three_panel_summary(table, group_col=None, verbose=True, semantics=('elgaby', 'v4'),
                        links='all'):
    """The paper's three panels from one unit table, one block per (link, semantics).

    El-Gaby reports the same correlation under three selections, and the number moves a lot
    between them, so reporting one without the others is misleading:

        all state-tuned        -> Fig 5h left    (no lag filter)
        + non-zero-lag 30 deg  -> Fig 5h right   (excludes lags {0, 11})
        + non-zero-lag 90 deg  -> ED Fig 8a      (excludes {0,1,2,9,10,11})

    `semantics`: 'elgaby' is the reproduction claim -- each panel pools independently
    (state-tuned AND >= 1 fold with a finite value for that panel, his cell-38 `remove_nan`), a
    non-zero-lag neuron needs >= 1 passing fold and is valued by the mean over passing folds;
    'v4' is the per-neuron majority-vote mask with all fitted folds averaged;
    'elgaby_everyfold' is the SUPERSEDED rule that also required a finite r in every fold
    (see `PANEL_POOL_CORRECTION.md`). `links`: 'all' prints one block per stored link (Poisson
    runs store both readouts), or a suffix ('' / '_altlink').

    Each row is `n`, mean r, fraction > 0, t and P, over the correlation appropriate to that
    panel -- the all-betas one for the first, the reduced-beta ones for the other two -- plus
    the published (n, t) for that panel and estimator.
    """
    import pandas as pd
    from scipy import stats as st

    if isinstance(semantics, str):
        semantics = (semantics,)
    link_list = _table_links(table) if links == 'all' else [
        (links, dict(_table_links(table)).get(links, links))]
    est = table.attrs.get('estimator', 'elasticnet')
    thr = float(table.attrs.get('state_tuning_p_threshold', 0.05))
    paper = PAPER_PANELS.get(est, {}).get(round(thr, 2), (None, None, None))

    rows = []
    for suffix, link_label in link_list:
        for sem in semantics:
            for (label, sel_col, val_col), pub in zip(_panel_columns(sem, suffix), paper):
                if sel_col not in table.columns or val_col not in table.columns:
                    continue
                v = table.loc[table[sel_col], val_col].dropna().to_numpy()
                t, p = st.ttest_1samp(v, 0) if len(v) > 1 else (np.nan, np.nan)
                rows.append({'link': link_label, 'semantics': sem, 'panel': label, 'n': len(v),
                             'mean_r': round(float(v.mean()), 4) if len(v) else np.nan,
                             'frac_pos': round(float(np.mean(v > 0)), 3) if len(v) else np.nan,
                             't': round(float(t), 2), 'P': float(p),
                             'effect_size': (round(float(t / np.sqrt(len(v))), 3)
                                             if len(v) > 1 else np.nan),
                             'paper_n': pub[0] if pub else np.nan,
                             'paper_t': pub[1] if pub else np.nan})
    out = pd.DataFrame(rows)
    if verbose:
        ysc = table.attrs.get('y_scaling', 'none')
        if ysc != 'none':
            # The published counts assume raw spike counts at a fixed alpha; a rescaled target
            # changes which betas survive, so the paper_n column is context, not a target.
            print(f'\n[target scaling: {ysc} -- NOT the reproduction; the paper_n/paper_t '
                  'columns below are for orientation only]')
        for (link_label, sem), sub in out.groupby(['link', 'semantics'], sort=False):
            print(f'\n[{link_label} | {sem} semantics]')
            print(sub.drop(columns=['link', 'semantics']).to_string(index=False))
        src = {'elasticnet': 'Fig 5h / ED 8a (p<0.05), ED 8b (p<0.01)',
               'poisson': 'ED 8d (p<0.05)'}.get(est, '')
        print(f'\npaper_n / paper_t: El-Gaby et al. 2024, {est}, PFC, state tuning p<{thr:g} '
              f'-- {src}' if paper[0] else
              f'\n(no published values for estimator={est!r} at p<{thr:g})')
    return out


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
