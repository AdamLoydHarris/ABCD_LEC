"""
GLM analysis with no-intercept, keep-all-bins parameterization.

Sibling of glm_analysis.py. Differences:
  1. No intercept column.
  2. One-hot encodings keep ALL bins (no reference-bin dropping):
     - place: 21 bins (locs 1..21)
     - head direction: 36 bins (10 deg bins)
     - goal progress: 10 bins (configurable via gp_n_bins; e.g. 20 to
       column-match the joint 20-col time_any / distance_any groups)
     - speed, acc, time/distance from/to reward: 10 decile bins each
     Total design matrix: 127 columns (137 when gp_n_bins=20).
  3. Design matrix is rank-deficient by 8 (each one-hot block sums to 1 per row,
     so without an intercept the blocks have 8 independent linear dependencies).
     np.linalg.lstsq returns the minimum-norm solution; individual betas are no
     longer uniquely identified, but predictions and RSS_full / RSS_reduced are
     invariant to the parameterization (the column space is the same). CPD, R²,
     and ΔR² therefore match the conventional reference-coded fit exactly. The
     F-statistic itself differs by a constant df-scaling because the default
     uses raw block sizes for df_num and T − 127 for df_resid (vs block_size−1
     and T − 119 in reference coding); but the permutation-based significance
     classification is essentially identical. Use parameterization='reference_coded'
     in run_glm_analysis to verify empirically.

Original glm_analysis.py is unchanged.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


# ============================================================================
# Task state / kinematic helpers
# ============================================================================

def compute_task_state_arrays(trial_times, num_bins=10):
    max_time = int(np.max(trial_times))
    num_states = trial_times.shape[1] - 1

    state_array = np.zeros(max_time + 1, dtype=int)
    goal_progress_array = np.zeros(max_time + 1, dtype=float)
    goal_progress_binned = np.zeros(max_time + 1, dtype=int)
    time_from_last_reward = np.zeros(max_time + 1, dtype=int)
    time_to_next_reward = np.zeros(max_time + 1, dtype=int)

    trial_times_sorted = np.sort(trial_times.flatten())

    for i in range(len(trial_times_sorted) - 1):
        start_time = int(trial_times_sorted[i])
        end_time = int(trial_times_sorted[i + 1])
        if start_time == end_time:
            continue
        state_array[start_time:end_time] = i % num_states
        time_range = np.arange(start_time, end_time)
        progress = (time_range - start_time) / (end_time - start_time)
        goal_progress_array[start_time:end_time] = progress
        goal_progress_binned[start_time:end_time] = np.floor(progress * num_bins).astype(int)
        time_from_last_reward[start_time:end_time] = time_range - start_time
        time_to_next_reward[start_time:end_time] = end_time - time_range

    last_time = int(trial_times_sorted[-1])
    state_array[last_time:] = (len(trial_times_sorted) - 1) % num_states
    goal_progress_array[last_time:] = 1.0
    goal_progress_binned[last_time:] = num_bins - 1
    time_from_last_reward[last_time:] = np.arange(0, max_time - last_time + 1)
    time_to_next_reward[last_time:] = 0

    return state_array, goal_progress_array, goal_progress_binned, time_from_last_reward, time_to_next_reward


def compute_distance_to_rewards(trial_times, speed_array):
    reward_times = np.unique(np.sort(trial_times.flatten()))
    reward_times = reward_times[reward_times >= 0]

    max_time = int(reward_times[-1])
    T = len(speed_array)
    cumulative_distance = np.zeros(T + 1)
    for i in range(1, T + 1):
        cumulative_distance[i] = cumulative_distance[i - 1] + speed_array[i - 1]

    n_timepoints = min(T, max_time + 1)
    distance_from_last_reward = np.zeros(n_timepoints, dtype=float)
    distance_to_next_reward = np.zeros(n_timepoints, dtype=float)

    for t in range(n_timepoints):
        idx_last = np.searchsorted(reward_times, t, side='right') - 1
        last_reward_time = 0 if idx_last < 0 else int(reward_times[idx_last])
        t_clipped = min(t, T)
        last_clipped = min(last_reward_time, T)
        distance_from_last_reward[t] = cumulative_distance[t_clipped] - cumulative_distance[last_clipped]

        idx_next = np.searchsorted(reward_times, t, side='left')
        next_reward_time = t if idx_next >= len(reward_times) else int(reward_times[idx_next])
        next_clipped = min(next_reward_time, T)
        distance_to_next_reward[t] = cumulative_distance[next_clipped] - cumulative_distance[t_clipped]

    return distance_from_last_reward, distance_to_next_reward


def smooth_and_calculate_scalar_derivatives(data_matrix, sigma=3, dt=1.0):
    x_smoothed = gaussian_filter1d(data_matrix[:, 0], sigma=sigma)
    y_smoothed = gaussian_filter1d(data_matrix[:, 1], sigma=sigma)
    vx = np.gradient(x_smoothed, dt)
    vy = np.gradient(y_smoothed, dt)
    speed = np.sqrt(vx**2 + vy**2)
    acceleration = np.gradient(speed, dt)
    return np.column_stack((x_smoothed, y_smoothed, speed, acceleration))


# ============================================================================
# Session selection and data preparation
# ============================================================================

def get_sessions_for_glm(recday_data):
    valid_sessions = {k: v for k, v in recday_data.items()
                      if 'num_trials' in v and v['num_trials'] > 2
                      and 'Neuron_raw' in v and v['Neuron_raw'] is not None
                      and 'Locs_raw' in v}
    sorted_sessions = sorted(valid_sessions.items(), key=lambda x: x[1]['num_trials'], reverse=True)
    unique_tasks = {}
    for session, data in sorted_sessions:
        task = str(data.get('Task', 'unknown'))
        if task not in unique_tasks:
            unique_tasks[task] = session
    return list(unique_tasks.values()), list(unique_tasks.keys())


def prepare_session_data(session_data, gp_n_bins=10):
    FR = session_data['Neuron_raw']
    Locs = session_data['Locs_raw']

    if 'XY_raw' in session_data and session_data['XY_raw'] is not None:
        XY = session_data['XY_raw']
        if XY.ndim == 1:
            XY = XY.reshape(-1, 2) if len(XY) % 2 == 0 else np.column_stack([XY, XY])
        processed_xy = smooth_and_calculate_scalar_derivatives(XY)
        Speed = processed_xy[:, 2]
        Acc = processed_xy[:, 3]
    else:
        Speed = np.zeros(len(Locs))
        Acc = np.zeros(len(Locs))

    if 'HD_raw' in session_data and session_data['HD_raw'] is not None:
        HD = session_data['HD_raw'].flatten()
    else:
        HD = np.full(len(Locs), np.nan)

    if 'Trial_times' in session_data and session_data['Trial_times'] is not None:
        Trial_times = session_data['Trial_times']
        Trial_times_bins = Trial_times.astype(int)
        State, _, GP_binned, time_from, time_to = compute_task_state_arrays(
            Trial_times_bins, num_bins=gp_n_bins
        )
        if len(Speed) > 0:
            dist_from, dist_to = compute_distance_to_rewards(Trial_times_bins, Speed)
        else:
            dist_from = np.zeros(len(State))
            dist_to = np.zeros(len(State))
    else:
        max_len = FR.shape[1]
        State = np.zeros(max_len, dtype=int)
        GP_binned = np.zeros(max_len, dtype=int)
        time_from = np.zeros(max_len)
        time_to = np.zeros(max_len)
        dist_from = np.zeros(max_len)
        dist_to = np.zeros(max_len)

    # Distance-based goal progress: fraction of the inter-reward path travelled.
    # NaN at samples where the animal didn't move at all in the interval
    # (total path = 0); zero-mask them downstream.
    total_path = dist_from + dist_to
    with np.errstate(divide='ignore', invalid='ignore'):
        gp_distance_cont = np.where(total_path > 0, dist_from / total_path, np.nan)

    return {
        'FR': FR,
        'Locs': Locs,
        'HD': HD,
        'Speed': Speed,
        'Acc': Acc,
        'State': State,
        'GP_binned': GP_binned,
        'GP_dist_continuous': gp_distance_cont,
        'time_from_reward': time_from,
        'time_to_reward': time_to,
        'dist_from_reward': dist_from,
        'dist_to_reward': dist_to,
    }


def truncate_all_arrays(data_dict):
    lengths = []
    for key, arr in data_dict.items():
        if isinstance(arr, np.ndarray):
            lengths.append(arr.shape[1] if arr.ndim == 2 else len(arr))
    max_index = min(lengths) if lengths else 0
    truncated = {}
    for key, arr in data_dict.items():
        if isinstance(arr, np.ndarray):
            truncated[key] = arr[:, :max_index] if arr.ndim == 2 else arr[:max_index]
        else:
            truncated[key] = arr
    return truncated


def downsample_session_data(data_dict, factor):
    """Downsample all arrays by taking every `factor`th sample along the time axis."""
    if factor <= 1:
        return data_dict
    result = {}
    for key, arr in data_dict.items():
        if isinstance(arr, np.ndarray):
            result[key] = arr[:, ::factor] if arr.ndim == 2 else arr[::factor]
        else:
            result[key] = arr
    return result


# ============================================================================
# One-hot encoding helpers
# ============================================================================

def compute_decile_edges(values, n_bins=10, outlier_pct=1):
    """Compute bin edges from training data, excluding outliers."""
    values = np.asarray(values).flatten()
    values = values[np.isfinite(values)]
    if len(values) < n_bins * 2:
        return None
    lo, hi = np.percentile(values, [outlier_pct, 100 - outlier_pct])
    clipped = values[(values >= lo) & (values <= hi)]
    edges = np.percentile(clipped, np.linspace(0, 100, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def apply_onehot(values, edges):
    """Bin values using precomputed edges, one-hot encode keeping ALL bins (no reference drop)."""
    values = np.asarray(values, dtype=float)
    n_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    onehot = (bin_idx[:, None] == np.arange(0, n_bins)).astype(float)
    onehot[~np.isfinite(values)] = 0
    return onehot


# ============================================================================
# Raised-cosine basis (Pillow et al. 2008) — smooth alternative to one-hot bins
# ============================================================================

# Per-variable spacing for the raised-cosine basis. 'log' concentrates
# resolution near the low end of the range (i.e. near the reward event for
# reward-relative variables); 'linear' spreads resolution evenly.
_RAISED_COSINE_SPACING = {
    'speed':                  'linear',
    'acceleration':           'linear',
    'goal_progress':          'linear',
    'goal_progress_distance': 'linear',
    'time_from_reward':       'log',
    'time_to_reward':         'log',
    'distance_from_reward':   'log',
    'distance_to_reward':     'log',
}


def make_raised_cosine_basis(values, n_basis=10, spacing='linear',
                             value_range=None, log_offset=None):
    """Raised-cosine basis for a continuous variable (Pillow et al. 2008).

    Returns an (n_samples, n_basis) matrix. Each column is a smooth cosine
    "bump"; adjacent bumps cross at half-height and sum to ≈ 1 in the interior,
    so the basis is a smooth analogue of one-hot binning.

    Parameters
    ----------
    values : array-like
        Raw continuous values (not pre-binned).
    n_basis : int, default 10
        Number of basis functions (columns). 10 matches the decile one-hot
        column count, keeping the design matrix column-structure identical.
    spacing : {'linear', 'log'}
        'linear' places centers evenly across the range.
        'log' places centers evenly in log-warped space, concentrating
        resolution near the low end of the range.
    value_range : (lo, hi) or None
        Range to cover. If None, uses 1st–99th percentile of finite values
        (matches `compute_decile_edges` outlier clipping).
    log_offset : float or None
        Offset added before the log warp so it's finite at the low edge (and
        at 0). Defaults to a small fraction of the range.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros((len(values), n_basis))

    if value_range is None:
        lo, hi = np.percentile(values[finite], [1, 99])
    else:
        lo, hi = value_range
    if hi <= lo:
        hi = lo + 1e-6

    if spacing == 'log':
        if log_offset is None:
            log_offset = max(1e-6, (hi - lo) / n_basis)
        warp = lambda x: np.log(np.clip(x, lo, hi) - lo + log_offset)
    elif spacing == 'linear':
        warp = lambda x: np.clip(x, lo, hi)
    else:
        raise ValueError(f"spacing must be 'linear' or 'log', got {spacing!r}")

    centers = np.linspace(warp(lo), warp(hi), n_basis)
    # Bump width = 2 × center spacing → adjacent bumps cross at half-height.
    db = centers[1] - centers[0] if n_basis > 1 else 1.0
    width = 2.0 * db if db > 0 else 1.0

    w_vals = warp(values)
    basis = np.zeros((len(values), n_basis))
    for j, cj in enumerate(centers):
        d = (w_vals - cj) * np.pi / width
        basis[:, j] = 0.5 * (1.0 + np.cos(np.clip(d, -np.pi, np.pi)))
    basis[~finite] = 0.0
    return basis


# ============================================================================
# Regressor group index map (no intercept, all bins kept)
# place(21) + HD(36) + GP(10) + speed(10) + acc(10)
# + time_from(10) + time_to(10) + dist_from(10) + dist_to(10) = 127 total
# ============================================================================

regressor_groups = {
    'place':                list(range(0,   21)),
    'head_direction':       list(range(21,  57)),
    'goal_progress':        list(range(57,  67)),
    'speed':                list(range(67,  77)),
    'acceleration':         list(range(77,  87)),
    'time_from_reward':     list(range(87,  97)),
    'time_to_reward':       list(range(97,  107)),
    'distance_from_reward': list(range(107, 117)),
    'distance_to_reward':   list(range(117, 127)),
}

analysis_regressor_names = [
    'place',
    'head_direction',
    'goal_progress',
    'speed',
    'acceleration',
    'time_from_reward',
    'time_to_reward',
    'distance_from_reward',
    'distance_to_reward',
]


# Optional regressors. NOT in `analysis_regressor_names`, so
# `regressors_to_include=None` continues to give the 9-regressor default
# (backwards-compatible). To include one, list it explicitly in
# `regressors_to_include`. `goal_progress_distance` is the path-length analogue
# of time-based `goal_progress`: progress through an inter-reward interval
# measured by integrated path (`dist_from / (dist_from + dist_to)`) rather than
# elapsed time. Shares `gp_n_bins` with time-GP → automatically column-matched.
_OPTIONAL_REGRESSORS = ['goal_progress_distance']
_ALL_REGRESSORS = analysis_regressor_names + _OPTIONAL_REGRESSORS

# Canonical column ordering when a fit includes a mix of default + optional
# regressors. Distance-GP sits adjacent to time-GP for semantic clarity.
_CANONICAL_ORDER = [
    'place',
    'head_direction',
    'goal_progress',
    'goal_progress_distance',
    'speed',
    'acceleration',
    'time_from_reward',
    'time_to_reward',
    'distance_from_reward',
    'distance_to_reward',
]


# User-facing aliases → canonical names. Lets users type "time_since_reward"
# instead of "time_from_reward" (clearer phrasing, same underlying variable).
_REGRESSOR_NAME_ALIASES = {
    'time_since_reward':     'time_from_reward',
    'distance_since_reward': 'distance_from_reward',
}


# Pretty display labels for plots / printouts. Falls back to the canonical name
# if a key is missing.
_REGRESSOR_DISPLAY_NAMES = {
    'place':                  'place',
    'head_direction':         'head direction',
    'goal_progress':          'goal progress',
    'goal_progress_distance': 'goal progress (distance)',
    'speed':                  'speed',
    'acceleration':           'acceleration',
    'time_from_reward':       'time since reward',
    'time_to_reward':         'time to reward',
    'distance_from_reward':   'distance since reward',
    'distance_to_reward':     'distance to reward',
}


def _display(name):
    """Return a pretty label for a regressor name (canonical or alias)."""
    canonical = _REGRESSOR_NAME_ALIASES.get(name, name)
    return _REGRESSOR_DISPLAY_NAMES.get(canonical, canonical)


def _resolve_regressor_groups(regressors_to_include, gp_n_bins=10,
                              parameterization='all_bins'):
    """Resolve (regressor_groups, analysis_regressor_names) for a subset.

    If `regressors_to_include` is None and `gp_n_bins == 10` and
    `parameterization == 'all_bins'`, returns the module-level defaults
    (all 9 regressors, 127-col layout). Otherwise returns a NEW regressor_groups
    dict whose indices reflect a design matrix built from only the included
    regressors, in canonical order (the order they appear in
    `analysis_regressor_names`).

    Accepts both canonical names and aliases defined in `_REGRESSOR_NAME_ALIASES`
    (e.g. 'time_since_reward' is treated as 'time_from_reward').

    `gp_n_bins` sets the number of goal_progress one-hot columns (default 10).
    Pass the same value used in `run_glm_analysis` so the index map matches the
    fitted design matrix (e.g. gp_n_bins=20 to column-match GP to the joint
    20-col time_any / distance_any groups).

    `parameterization` selects the design-matrix encoding:
      - 'all_bins'        : no intercept, keep all bins per categorical block
                            (current default; rank-deficient by 8).
      - 'reference_coded' : intercept column at index 0 + drop the first bin of
                            every categorical block. Each per-regressor index
                            list shrinks by 1; the cursor starts at 1 to leave
                            room for the intercept. Total cols at gp_n_bins=10:
                            1 + 20 + 35 + 9 + 6·9 = 119 (full rank).
    """
    # Per-regressor column counts. Defaults come from the 9-regressor
    # `regressor_groups`; optional regressors (e.g. `goal_progress_distance`)
    # are not in that map, so add them explicitly.
    n_cols_per = {name: len(regressor_groups[name]) for name in analysis_regressor_names}
    n_cols_per['goal_progress'] = gp_n_bins
    n_cols_per['goal_progress_distance'] = gp_n_bins  # shares the gp_n_bins knob
    if parameterization == 'reference_coded':
        n_cols_per = {k: v - 1 for k, v in n_cols_per.items()}
    elif parameterization != 'all_bins':
        raise ValueError(f"parameterization must be 'all_bins' or "
                         f"'reference_coded', got {parameterization!r}")

    if regressors_to_include is None:
        if gp_n_bins == 10 and parameterization == 'all_bins':
            return regressor_groups, list(analysis_regressor_names)
        ordered = list(analysis_regressor_names)
    else:
        # Map aliases → canonical names; validate against the full set
        # (default + optional regressors like goal_progress_distance).
        resolved = [_REGRESSOR_NAME_ALIASES.get(r, r) for r in regressors_to_include]

        invalid = [r for r in resolved if r not in _ALL_REGRESSORS]
        if invalid:
            raise ValueError(f"Unknown regressor names: {invalid}. "
                             f"Valid names: {_ALL_REGRESSORS} "
                             f"(aliases: {list(_REGRESSOR_NAME_ALIASES.keys())})")

        ordered = [r for r in _CANONICAL_ORDER if r in resolved]

    local = {}
    cursor = 1 if parameterization == 'reference_coded' else 0
    for name in ordered:
        n = n_cols_per[name]
        local[name] = list(range(cursor, cursor + n))
        cursor += n
    return local, ordered


# ============================================================================
# GLM analysis — no CV, fit on all available data per recording day
# ============================================================================

def run_glm_analysis(mouse_recdays, data_dic,
                     num_permutations=100, downsample_factor=10,
                     regressors_to_include=None,
                     compute_cpd=False,
                     joint_drop_groups=None,
                     continuous_basis='onehot',
                     gp_n_bins=10,
                     parameterization='all_bins'):
    """Fit per-neuron OLS GLM with permutation F-tests.

    Parameters
    ----------
    mouse_recdays, data_dic : see module docstring.
    num_permutations : int, default 100
        Number of circular firing-rate shifts in the permutation null.
    downsample_factor : int, default 10
    regressors_to_include : list of str or None
        Subset of `analysis_regressor_names` to include in the design matrix.
        Accepts canonical names AND aliases (e.g. 'time_since_reward'). None
        (default) keeps all 9 regressors.
    compute_cpd : bool, default False
        If True, also compute the Coefficient of Partial Determination (CPD)
        per neuron per regressor group: CPD = (RSS_reduced − RSS_full) / RSS_reduced.
        Return signature changes from a 2-tuple to a 3-tuple in this case.
    joint_drop_groups : list of (group_name, [regressor_names]) or None
        Each entry defines a joint test: fit a reduced model with ALL listed
        regressors dropped together, store F-stat (and CPD if compute_cpd=True)
        under `group_name` alongside the per-regressor results.
        Example: `[('time_any', ['time_from_reward', 'time_to_reward'])]`
        gives a joint test of "any time information" using a single F-stat
        on a 20-column reduction.
        Aliases are resolved in the listed regressor names.
    continuous_basis : {'onehot', 'raised_cosine'}, default 'onehot'
        Encoding for the continuous variables (speed, acceleration, GP, and the
        4 reward-relative variables). 'onehot' = decile bins (current behaviour).
        'raised_cosine' = 10 smooth raised-cosine bumps per variable
        (log-spaced for reward-relative variables, linear for speed/accel/GP).
        Place and head_direction are always one-hot. Because n_basis=10 matches
        the decile count, the design matrix is identical in shape (127 cols) and
        all downstream stats/plots are unchanged.
    gp_n_bins : int, default 10
        Number of goal_progress bins (one-hot columns, or raised-cosine bumps).
        The default 10 gives the standard 127-col design matrix. Setting 20
        column-matches goal_progress to the joint 20-col `time_any` /
        `distance_any` groups (design matrix → 137 cols), removing the
        column-count advantage from the headline GP-vs-time/distance comparison.
        Pass the SAME value to `compute_tuning_arrays`.
    parameterization : {'all_bins', 'reference_coded'}, default 'all_bins'
        Design-matrix encoding for categorical blocks:
          - 'all_bins'        : no intercept, keep all bins per block
                                (rank-deficient by 8; current default).
          - 'reference_coded' : prepend an intercept column and drop the FIRST
                                bin of every block (textbook coding; full rank).
        F-stats differ between modes because df accounting differs (block_size
        vs block_size−1 dropped per reduced model; T − 127 vs T − 119 residual
        df at gp_n_bins=10). RSS_full, RSS_reduced, CPD, R², ΔR², and the
        permutation-based significance flags are essentially identical across
        modes — this is intended as an empirical sanity check on the default.
        Pass the SAME value to `compute_tuning_arrays`.

    Returns
    -------
    GLM_results : dict of {mouse_recday: {neuron_idx: params_array}}
    Permutation_results : dict of {mouse_recday: {neuron_idx: (F_real_dict, F_perm_dict)}}
    CPD_results (only if compute_cpd=True) : dict of {mouse_recday: {neuron_idx: cpd_dict}}
        Per-regressor CPDs plus reserved keys:
          '__r2_full__'  : full-model R² (scalar)
          '__delta_r2__' : {regressor: ΔRSS/TSS} unique R² per regressor
    """
    local_groups, local_names = _resolve_regressor_groups(
        regressors_to_include, gp_n_bins=gp_n_bins,
        parameterization=parameterization,
    )

    # Resolve aliases in joint_drop_groups and validate
    joint_specs = []  # list of (group_name, [canonical_regressor_names])
    if joint_drop_groups is not None:
        for group_name, regs in joint_drop_groups:
            resolved_regs = [_REGRESSOR_NAME_ALIASES.get(r, r) for r in regs]
            for r in resolved_regs:
                if r not in local_names:
                    raise ValueError(
                        f"Joint group {group_name!r} references {r!r} which is not "
                        f"in the included regressors. Include it via "
                        f"regressors_to_include or remove from joint_drop_groups."
                    )
            joint_specs.append((group_name, resolved_regs))

    GLM_results = {}
    Permutation_results = {}
    CPD_results = {}

    for mouse_recday in tqdm(mouse_recdays, desc="Processing recording days"):
        print(f"\n{mouse_recday}")

        GLM_results[mouse_recday] = {}
        Permutation_results[mouse_recday] = {}
        CPD_results[mouse_recday] = {}

        sessions_for_glm, _ = get_sessions_for_glm(data_dic[mouse_recday])

        if len(sessions_for_glm) < 2:
            print(f"  Skipping — not enough valid sessions ({len(sessions_for_glm)})")
            continue

        first_session = sessions_for_glm[0]
        num_neurons = data_dic[mouse_recday][first_session]['Neuron_raw'].shape[0]

        prepared_sessions = {}
        for session in sessions_for_glm:
            prep_data = prepare_session_data(data_dic[mouse_recday][session],
                                             gp_n_bins=gp_n_bins)
            prep_data = truncate_all_arrays(prep_data)
            prepared_sessions[session] = downsample_session_data(prep_data, downsample_factor)

        # ----------------------------------------------------------------
        # Pool all behavioral data across sessions (valid locations only)
        # ----------------------------------------------------------------
        all_speed, all_acc   = [], []
        all_tf, all_tt       = [], []
        all_df, all_dt       = [], []
        all_locs, all_hd, all_gp = [], [], []
        all_gpd              = []  # distance-based goal progress (continuous 0–1, NaN where no motion)
        session_filters = []  # (session, node_filter) for reconstructing FR

        for session in sessions_for_glm:
            prep = prepared_sessions[session]
            nf   = prep['Locs'] <= 21
            all_speed.append(prep['Speed'][nf])
            all_acc.append(prep['Acc'][nf])
            all_tf.append(prep['time_from_reward'][nf])
            all_tt.append(prep['time_to_reward'][nf])
            all_df.append(prep['dist_from_reward'][nf])
            all_dt.append(prep['dist_to_reward'][nf])
            all_locs.append(prep['Locs'][nf])
            all_hd.append(prep['HD'][nf])
            all_gp.append(prep['GP_binned'][nf])
            all_gpd.append(prep['GP_dist_continuous'][nf])
            session_filters.append((session, nf))

        speed_all = np.concatenate(all_speed)
        acc_all   = np.concatenate(all_acc)
        tf_all    = np.concatenate(all_tf)
        tt_all    = np.concatenate(all_tt)
        df_all    = np.concatenate(all_df)
        dt_all    = np.concatenate(all_dt)
        locs_all  = np.concatenate(all_locs).astype(int)
        hd_all    = np.concatenate(all_hd)
        gp_all    = np.concatenate(all_gp)
        gpd_all   = np.concatenate(all_gpd)

        # ----------------------------------------------------------------
        # Decile edges from all available data
        # ----------------------------------------------------------------
        speed_edges = compute_decile_edges(speed_all)
        acc_edges   = compute_decile_edges(acc_all)
        tf_edges    = compute_decile_edges(tf_all)
        tt_edges    = compute_decile_edges(tt_all)
        df_edges    = compute_decile_edges(df_all)
        dt_edges    = compute_decile_edges(dt_all)

        if any(e is None for e in [speed_edges, acc_edges, tf_edges,
                                    tt_edges, df_edges, dt_edges]):
            print("  Skipping — insufficient data for decile edges")
            continue

        # ----------------------------------------------------------------
        # Build design matrix (shared across all neurons)
        # No intercept; all bins kept per variable. Rank-deficient by 8.
        # ----------------------------------------------------------------
        # Place: nodes 1–21 → 21 cols
        place_onehot = (locs_all[:, None] == np.arange(1, 22)).astype(float)

        # HD: 36 fixed 10° bins → 36 cols
        hd_bin_idx = np.clip(np.floor((hd_all % 360) / 10).astype(int), 0, 35)
        HD_onehot  = (hd_bin_idx[:, None] == np.arange(0, 36)).astype(float)
        HD_onehot[~np.isfinite(hd_all)] = 0

        if continuous_basis == 'onehot':
            # GP: gp_n_bins equal-width bins → gp_n_bins cols
            GP_enc = (gp_all[:, None] == np.arange(0, gp_n_bins)).astype(float)
            # Distance-based GP: same gp_n_bins, NaN samples zeroed out
            gpd_finite = np.isfinite(gpd_all)
            gpd_binned = np.zeros(len(gpd_all), dtype=int)
            gpd_binned[gpd_finite] = np.clip(
                np.floor(gpd_all[gpd_finite] * gp_n_bins).astype(int),
                0, gp_n_bins - 1,
            )
            GPd_enc = (gpd_binned[:, None] == np.arange(0, gp_n_bins)).astype(float)
            GPd_enc[~gpd_finite] = 0
            # Continuous vars: decile one-hot bins → 10 cols each
            Sp_enc = apply_onehot(speed_all, speed_edges)
            Ac_enc = apply_onehot(acc_all,   acc_edges)
            TF_enc = apply_onehot(tf_all,    tf_edges)
            TT_enc = apply_onehot(tt_all,    tt_edges)
            DF_enc = apply_onehot(df_all,    df_edges)
            DT_enc = apply_onehot(dt_all,    dt_edges)
        elif continuous_basis == 'raised_cosine':
            # Smooth raised-cosine basis (10 bumps each → same col count).
            # Per-variable spacing from _RAISED_COSINE_SPACING (log for
            # reward-relative variables, linear for speed/accel/GP).
            def _rc(vals, name):
                return make_raised_cosine_basis(
                    vals, n_basis=10, spacing=_RAISED_COSINE_SPACING[name]
                )
            # GP gets gp_n_bins bumps so its column count matches the one-hot path
            GP_enc = make_raised_cosine_basis(
                gp_all.astype(float), n_basis=gp_n_bins,
                spacing=_RAISED_COSINE_SPACING['goal_progress']
            )
            # Distance-GP raised-cosine: known [0,1] range; NaN handled by basis
            GPd_enc = make_raised_cosine_basis(
                gpd_all, n_basis=gp_n_bins,
                spacing=_RAISED_COSINE_SPACING['goal_progress_distance'],
                value_range=(0.0, 1.0),
            )
            Sp_enc = _rc(speed_all,             'speed')
            Ac_enc = _rc(acc_all,               'acceleration')
            TF_enc = _rc(tf_all,                'time_from_reward')
            TT_enc = _rc(tt_all,                'time_to_reward')
            DF_enc = _rc(df_all,                'distance_from_reward')
            DT_enc = _rc(dt_all,                'distance_to_reward')
        else:
            raise ValueError(
                f"continuous_basis must be 'onehot' or 'raised_cosine', "
                f"got {continuous_basis!r}"
            )

        onehots = {
            'place':                  place_onehot,
            'head_direction':         HD_onehot,
            'goal_progress':          GP_enc,
            'goal_progress_distance': GPd_enc,
            'speed':                  Sp_enc,
            'acceleration':           Ac_enc,
            'time_from_reward':       TF_enc,
            'time_to_reward':         TT_enc,
            'distance_from_reward':   DF_enc,
            'distance_to_reward':     DT_enc,
        }
        if parameterization == 'reference_coded':
            # Drop the first column of each block (the reference bin) and
            # prepend a single intercept column.
            blocks = [onehots[name][:, 1:] for name in local_names]
            X = np.column_stack([np.ones((blocks[0].shape[0], 1))] + blocks)
        else:
            X = np.column_stack([onehots[name] for name in local_names])
        X = np.nan_to_num(X)
        print(f"  Design matrix: {X.shape[0]} rows × {X.shape[1]} cols "
              f"(regressors: {local_names}, parameterization={parameterization})")

        if np.any(np.isnan(X)) or np.any(np.isinf(X)) or X.shape[0] < X.shape[1]:
            print(f"  Skipping — degenerate design matrix (shape {X.shape})")
            continue

        # Concatenate FR across sessions matching the same node filter
        FR_all = np.concatenate(
            [prepared_sessions[s]['FR'][:, nf] for s, nf in session_filters], axis=1
        )  # [n_neurons × T_total]

        # ----------------------------------------------------------------
        # Precompute reduced design matrices once (shared across neurons)
        # ----------------------------------------------------------------
        X_reduced_dict = {
            reg_name: np.delete(X, indices, axis=1)
            for reg_name, indices in local_groups.items()
        }

        # Joint reduced-model design matrices (drop multiple regressor groups together)
        X_reduced_joint = {}
        joint_n_dropped = {}
        for group_name, regs in joint_specs:
            joint_indices = sorted(set().union(*[local_groups[r] for r in regs]))
            X_reduced_joint[group_name] = np.delete(X, joint_indices, axis=1)
            joint_n_dropped[group_name] = len(joint_indices)

        T, n_params = X.shape
        df_resid = T - n_params
        shifts = np.random.randint(0, T, size=num_permutations)

        # ----------------------------------------------------------------
        # Per-neuron: OLS fit + permutation F-test (vectorised)
        # ----------------------------------------------------------------
        for neuron in tqdm(range(num_neurons), desc="  Neurons", leave=False):
            frs = FR_all[neuron]
            try:
                # --- Full model ---
                params, _, _, _ = np.linalg.lstsq(X, frs, rcond=None)
                resid_full      = frs - X @ params
                rss_full        = resid_full @ resid_full

                # Total sum of squares & full-model R² (for CPD normalization)
                tss = float(np.sum((frs - frs.mean()) ** 2))
                r2_full = (1.0 - rss_full / tss) if tss > 0 else 0.0

                # --- Permuted full models (all at once) ---
                perm_frs        = np.stack([np.roll(frs, s) for s in shifts]).T  # [T × n_perms]
                beta_perms      = np.linalg.lstsq(X, perm_frs, rcond=None)[0]    # [n_params × n_perms]
                resid_full_p    = perm_frs - X @ beta_perms                       # [T × n_perms]
                rss_full_p      = np.einsum('ij,ij->j', resid_full_p, resid_full_p)  # [n_perms]

                # --- F-stat per regressor group ---
                F_real = {}
                F_perm = {}
                CPD_real = {}
                delta_r2_real = {}   # unique R² = ΔRSS / TSS (for normalized CPD)
                for reg_name, X_r in X_reduced_dict.items():
                    df_num = len(local_groups[reg_name])

                    # Real reduced model
                    params_r, _, _, _ = np.linalg.lstsq(X_r, frs, rcond=None)
                    resid_r   = frs - X_r @ params_r
                    rss_r     = resid_r @ resid_r
                    F_real[reg_name] = ((rss_r - rss_full) / df_num) / (rss_full / df_resid)
                    if compute_cpd:
                        CPD_real[reg_name] = ((rss_r - rss_full) / rss_r) if rss_r > 0 else 0.0
                        delta_r2_real[reg_name] = ((rss_r - rss_full) / tss) if tss > 0 else 0.0

                    # Permuted reduced models (vectorised)
                    beta_perms_r = np.linalg.lstsq(X_r, perm_frs, rcond=None)[0]  # [n_r × n_perms]
                    resid_r_p    = perm_frs - X_r @ beta_perms_r                   # [T × n_perms]
                    rss_r_p      = np.einsum('ij,ij->j', resid_r_p, resid_r_p)    # [n_perms]
                    F_perm[reg_name] = ((rss_r_p - rss_full_p) / df_num) / (rss_full_p / df_resid)

                # --- Joint reduced-model F-stats (and CPDs) ---
                for group_name, X_rj in X_reduced_joint.items():
                    df_num_j = joint_n_dropped[group_name]

                    params_rj, _, _, _ = np.linalg.lstsq(X_rj, frs, rcond=None)
                    resid_rj  = frs - X_rj @ params_rj
                    rss_rj    = resid_rj @ resid_rj
                    F_real[group_name] = ((rss_rj - rss_full) / df_num_j) / (rss_full / df_resid)
                    if compute_cpd:
                        CPD_real[group_name] = ((rss_rj - rss_full) / rss_rj) if rss_rj > 0 else 0.0
                        delta_r2_real[group_name] = ((rss_rj - rss_full) / tss) if tss > 0 else 0.0

                    beta_perms_rj = np.linalg.lstsq(X_rj, perm_frs, rcond=None)[0]
                    resid_rj_p    = perm_frs - X_rj @ beta_perms_rj
                    rss_rj_p      = np.einsum('ij,ij->j', resid_rj_p, resid_rj_p)
                    F_perm[group_name] = ((rss_rj_p - rss_full_p) / df_num_j) / (rss_full_p / df_resid)

                GLM_results[mouse_recday][neuron]         = params
                Permutation_results[mouse_recday][neuron] = (F_real, F_perm)
                if compute_cpd:
                    # Reserved (__-prefixed) keys carry model-fit context alongside
                    # the per-regressor CPDs. Consumers iterating regressor names
                    # must skip keys starting with '__'.
                    CPD_real['__r2_full__'] = r2_full
                    CPD_real['__delta_r2__'] = delta_r2_real
                    CPD_results[mouse_recday][neuron] = CPD_real

            except Exception:
                continue

    if compute_cpd:
        return GLM_results, Permutation_results, CPD_results
    return GLM_results, Permutation_results


# ============================================================================
# Significance analysis
# ============================================================================

# Regressors where betas are ordered by bin — slope captures ramp direction
_ORDERED_REGRESSORS = {
    'goal_progress', 'goal_progress_distance',
    'speed', 'acceleration',
    'time_from_reward', 'time_to_reward',
    'distance_from_reward', 'distance_to_reward',
}


def _beta_direction(params, reg_name, groups=None):
    """
    Derive a sign (+1/-1) from the beta profile for a regressor group.
    Ordered variables: slope of a linear fit across all bin betas (all bins are
    present in this parameterization — no reference bin is dropped).
    Unordered variables (place, HD): sign of the mean beta.

    `groups` defaults to module-level `regressor_groups`; pass a subset-specific
    mapping (from `_resolve_regressor_groups`) when fitting a regressor subset.
    """
    if groups is None:
        groups = regressor_groups
    betas = params[groups[reg_name]]
    if reg_name in _ORDERED_REGRESSORS:
        slope = np.polyfit(np.arange(len(betas)), betas, 1)[0]
        return int(np.sign(slope)) or 1
    else:
        return int(np.sign(np.mean(betas))) or 1


def compute_tuning_arrays(GLM_results, Permutation_results, regressors_to_include=None,
                          gp_n_bins=10, parameterization='all_bins'):
    """
    Returns tuned_dict: {mouse_recday: tuning_array [n_neurons × n_regressors]}
    Values: +1 (positive / ramp-up tuning), -1 (negative / ramp-down), 0 (not significant).
    Significance: permutation F-test at p < 0.05 (one-sided on F).
    Direction: slope of beta profile for ordered regressors; sign of mean beta for unordered.

    Pass the same `regressors_to_include`, `gp_n_bins`, and `parameterization`
    used in `run_glm_analysis` so that `_beta_direction` indexes the regressor
    betas with the correct (subset-, bin-count-, and coding-specific) index map.
    """
    local_groups, local_names = _resolve_regressor_groups(
        regressors_to_include, gp_n_bins=gp_n_bins,
        parameterization=parameterization,
    )
    tuned_dict = {}

    for mouse_recday in GLM_results:
        if len(GLM_results[mouse_recday]) == 0:
            continue

        neuron_ids   = sorted(GLM_results[mouse_recday].keys())
        n_regressors = len(local_names)
        tuning_array = np.zeros((len(neuron_ids), n_regressors), dtype=int)

        for neuron_idx, neuron in enumerate(neuron_ids):
            params = GLM_results[mouse_recday].get(neuron)
            perm   = Permutation_results[mouse_recday].get(neuron)
            if params is None or perm is None:
                continue
            F_real, F_perm = perm

            for reg_idx, reg_name in enumerate(local_names):
                f_stat = F_real.get(reg_name)
                f_null = F_perm.get(reg_name)
                if f_stat is None or f_null is None:
                    continue

                # F-test: one-sided at p < 0.05
                if f_stat > np.percentile(f_null, 95):
                    tuning_array[neuron_idx, reg_idx] = _beta_direction(
                        params, reg_name, groups=local_groups
                    )

        tuned_dict[mouse_recday] = tuning_array
        print(f"{mouse_recday}: tuning array shape = {tuning_array.shape}")

    return tuned_dict


def group_tuning_by_mouse(tuned_dict):
    mouse_tuning = defaultdict(list)
    for mouse_recday, tuning_array in tuned_dict.items():
        mouse_tuning[mouse_recday[:4]].append(tuning_array)
    return {
        mouse_id: np.concatenate(arrays, axis=0)
        for mouse_id, arrays in sorted(mouse_tuning.items())
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_tuning_piecharts(mouse_tuning_concat, regressors_to_include=None):
    _, local_names = _resolve_regressor_groups(regressors_to_include)
    n_mice = len(mouse_tuning_concat)
    fig, axes = plt.subplots(
        len(local_names), n_mice,
        figsize=(4 * n_mice, 4 * len(local_names))
    )
    if n_mice == 1:
        axes = axes.reshape(-1, 1)
    if len(local_names) == 1:
        axes = axes.reshape(1, -1)

    for col, (mouse_id, all_tuning) in enumerate(sorted(mouse_tuning_concat.items())):
        n_total = all_tuning.shape[0]
        for row, reg_name in enumerate(local_names):
            ax = axes[row, col]
            reg_idx = local_names.index(reg_name)
            col_data = all_tuning[:, reg_idx]
            pos = np.sum(col_data == 1)
            neg = np.sum(col_data == -1)
            not_tuned = np.sum(col_data == 0)

            sizes  = [pos, neg, not_tuned]
            labels = ['Tuned (+)', 'Tuned (-)', 'Not Tuned']
            colors = ['#66b3ff', '#ff9999', '#d3d3d3']
            nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
            if nonzero:
                sizes, labels, colors = zip(*nonzero)
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            if row == 0:
                ax.set_title(f"Mouse {mouse_id}\n(n={n_total})", fontsize=11)
            if col == 0:
                ax.set_ylabel(reg_name, fontsize=9)

    plt.suptitle("LEC: Fraction of neurons tuned per regressor", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_gp_overlap(mouse_tuning_concat, regressors_to_include=None):
    _, local_names = _resolve_regressor_groups(regressors_to_include)

    # Required regressors for this overlap plot
    required = ['goal_progress', 'place', 'time_from_reward',
                'distance_from_reward', 'speed', 'acceleration']
    missing = [r for r in required if r not in local_names]
    if missing:
        print(f"plot_gp_overlap: skipping — required regressors missing from subset: {missing}")
        return

    gp_idx        = local_names.index('goal_progress')
    place_idx     = local_names.index('place')
    time_from_idx = local_names.index('time_from_reward')
    dist_from_idx = local_names.index('distance_from_reward')
    speed_idx     = local_names.index('speed')
    acc_idx       = local_names.index('acceleration')

    n_mice = len(mouse_tuning_concat)
    fig, axes = plt.subplots(6, n_mice, figsize=(4 * n_mice, 24),
                             gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1, 1, 1]})
    if n_mice == 1:
        axes = axes.reshape(6, 1)

    bar_colors = ['#d3d3d3', '#66b3ff', '#ff9999', '#7a4f9e']
    bar_rows = [
        (['GP only', 'GP + time', 'GP + dist', 'GP + both'], time_from_idx, dist_from_idx),
        (['GP only', 'GP + speed', 'GP + acc',  'GP + both'], speed_idx,     acc_idx),
    ]

    for col, (mouse_id, all_tuning) in enumerate(sorted(mouse_tuning_concat.items())):
        n_total    = all_tuning.shape[0]
        gp_mask    = all_tuning[:, gp_idx]    != 0
        place_mask = all_tuning[:, place_idx] != 0
        speed_mask = all_tuning[:, speed_idx] != 0
        n_gp       = np.sum(gp_mask)

        # Row 0: GP pie
        ax = axes[0, col]
        pos = np.sum(all_tuning[:, gp_idx] == 1)
        neg = np.sum(all_tuning[:, gp_idx] == -1)
        sizes  = [pos, neg, n_total - n_gp]
        labels = ['Tuned (+)', 'Tuned (-)', 'Not Tuned']
        colors = ['#66b3ff', '#ff9999', '#d3d3d3']
        nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if nonzero:
            sizes, labels, colors = zip(*nonzero)
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title(f"Mouse {mouse_id}\n(n={n_total} neurons)", fontsize=11)

        # Rows 1 & 2: 4-way bar charts among GP cells
        for row, (categories, idx_a, idx_b) in enumerate(bar_rows, start=1):
            ax = axes[row, col]
            if n_gp == 0:
                ax.axis('off')
                continue
            gp_tuned = all_tuning[gp_mask]
            a_tuned  = gp_tuned[:, idx_a] != 0
            b_tuned  = gp_tuned[:, idx_b] != 0
            counts = np.array([
                np.sum(~a_tuned & ~b_tuned),
                np.sum( a_tuned & ~b_tuned),
                np.sum(~a_tuned &  b_tuned),
                np.sum( a_tuned &  b_tuned),
            ])
            pcts = 100 * counts / n_gp
            bars = ax.bar(categories, pcts, color=bar_colors, edgecolor='white', linewidth=0.8)
            for bar, pct, cnt in zip(bars, pcts, counts):
                if pct > 3:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)
            if row == 1:
                ax.set_title(f"GP cells (n={n_gp})", fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of GP-tuned neurons" if col == 0 else "")
            ax.tick_params(axis='x', rotation=30)

        # Rows 3–5: single-variable overlap pies
        for row, (denom_mask, denom_label, cross_mask, slice_label, yes_color) in enumerate([
            (gp_mask,    'GP cells',    place_mask, 'Also place tuned', '#ff9999'),
            (speed_mask, 'Speed cells', gp_mask,    'Also GP tuned',    '#66b3ff'),
            (place_mask, 'Place cells', gp_mask,    'Also GP tuned',    '#66b3ff'),
        ], start=3):
            ax = axes[row, col]
            n_denom = np.sum(denom_mask)
            if n_denom == 0:
                ax.set_title(f"(no {denom_label.lower()})")
                ax.axis('off')
                continue
            yes = np.sum(denom_mask & cross_mask)
            no  = n_denom - yes
            sizes  = [yes, no]
            labels = [slice_label, 'Not tuned']
            colors = [yes_color, '#d3d3d3']
            nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
            if nonzero:
                sizes, labels, colors = zip(*nonzero)
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.set_title(f"{denom_label} (n={n_denom})", fontsize=10)

    row_label_texts = [
        'Goal progress\ntuning', 'GP × time / dist', 'GP × speed / acc',
        'GP cells\n→ place overlap', 'Speed cells\n→ GP overlap', 'Place cells\n→ GP overlap',
    ]
    for row, text in enumerate(row_label_texts):
        axes[row, 0].annotate(text, xy=(0, 0.5), xytext=(-0.35, 0.5),
                              xycoords='axes fraction', textcoords='axes fraction',
                              fontsize=11, fontweight='bold', va='center', rotation=90)

    plt.suptitle("Goal progress neurons: overlap with place, temporal, and kinematic regressors",
                 fontsize=13)
    plt.tight_layout()
    plt.show()


def print_tuning_summary(mouse_tuning_concat, regressors_to_include=None):
    _, local_names = _resolve_regressor_groups(regressors_to_include)
    print("\nTuning Summary:")
    for mouse_id, all_tuning in sorted(mouse_tuning_concat.items()):
        total = all_tuning.shape[0]
        print(f"\nMouse {mouse_id} (n={total}):")
        for reg_idx, reg_name in enumerate(local_names):
            col = all_tuning[:, reg_idx]
            pos = np.sum(col == 1)
            neg = np.sum(col == -1)
            print(f"  {reg_name}: {pos+neg}/{total} ({100*(pos+neg)/total:.1f}%)  "
                  f"[{pos}+ / {neg}-]")


# ============================================================================
# Extended analysis: simplified pie charts, time-vs-progress overlap, CPD
# ============================================================================

def plot_tuning_piecharts_binary(mouse_tuning_concat, regressors_to_include=None):
    """Two-slice pie charts (Tuned vs Not Tuned) per regressor.

    Like `plot_tuning_piecharts` but collapses +1/-1 into a single 'tuned' slice.
    Uses pretty display names from `_REGRESSOR_DISPLAY_NAMES`.
    """
    _, local_names = _resolve_regressor_groups(regressors_to_include)
    n_mice = len(mouse_tuning_concat)
    fig, axes = plt.subplots(
        len(local_names), n_mice,
        figsize=(4 * n_mice, 4 * len(local_names))
    )
    if n_mice == 1:
        axes = axes.reshape(-1, 1)
    if len(local_names) == 1:
        axes = axes.reshape(1, -1)

    for col, (mouse_id, all_tuning) in enumerate(sorted(mouse_tuning_concat.items())):
        n_total = all_tuning.shape[0]
        for row, reg_name in enumerate(local_names):
            ax = axes[row, col]
            reg_idx = local_names.index(reg_name)
            col_data = all_tuning[:, reg_idx]
            tuned = int(np.sum(col_data != 0))
            not_tuned = int(np.sum(col_data == 0))

            sizes = [tuned, not_tuned]
            labels = ['Tuned', 'Not Tuned']
            colors = ['#4c9be8', '#d3d3d3']
            nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
            if nonzero:
                sizes, labels, colors = zip(*nonzero)
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            if row == 0:
                ax.set_title(f"Mouse {mouse_id}\n(n={n_total})", fontsize=11)
            if col == 0:
                ax.set_ylabel(_display(reg_name), fontsize=10)

    plt.suptitle("LEC: Fraction of neurons significantly tuned per regressor", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_time_vs_progress_overlap(mouse_tuning_concat,
                                   regressors_to_include=None,
                                   use_joint_time=False):
    """Stacked-bar comparison of goal-progress vs time tuning, per mouse + aggregate.

    For each mouse, splits neurons into 4 categories:
      - GP only
      - Time only (tuned to time_from_reward OR time_to_reward — OR rule)
      - Both
      - Neither

    If `use_joint_time=True`, expects a column in `mouse_tuning_concat` named
    `time_any` (from `joint_drop_groups`) and uses that instead of the OR rule.
    """
    _, local_names = _resolve_regressor_groups(regressors_to_include)

    # Locate columns
    if 'goal_progress' not in local_names:
        print("plot_time_vs_progress_overlap: skipping — 'goal_progress' missing")
        return
    gp_col = local_names.index('goal_progress')

    if use_joint_time:
        if 'time_any' not in local_names:
            print("plot_time_vs_progress_overlap: 'time_any' not in tuning array; "
                  "fit with joint_drop_groups=[('time_any', [...])] and recompute "
                  "tuning_arrays with regressors_to_include including 'time_any'.")
            return
        time_col_indices = [local_names.index('time_any')]
    else:
        time_names_present = [r for r in ('time_from_reward', 'time_to_reward')
                              if r in local_names]
        if not time_names_present:
            print("plot_time_vs_progress_overlap: skipping — no time regressors present")
            return
        time_col_indices = [local_names.index(r) for r in time_names_present]

    n_mice = len(mouse_tuning_concat)
    mouse_ids = sorted(mouse_tuning_concat.keys())

    # Aggregate counts across mice for the summary panel
    agg_counts = np.zeros(4, dtype=int)  # [gp_only, time_only, both, neither]
    per_mouse = {}

    for mouse_id in mouse_ids:
        all_tuning = mouse_tuning_concat[mouse_id]
        gp_tuned = (all_tuning[:, gp_col] != 0)
        time_tuned = np.any(all_tuning[:, time_col_indices] != 0, axis=1)

        gp_only  = int(np.sum(gp_tuned & ~time_tuned))
        time_only = int(np.sum(~gp_tuned & time_tuned))
        both      = int(np.sum(gp_tuned & time_tuned))
        neither   = int(np.sum(~gp_tuned & ~time_tuned))

        per_mouse[mouse_id] = np.array([gp_only, time_only, both, neither])
        agg_counts += per_mouse[mouse_id]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.5 * n_mice + 4), 5),
                              gridspec_kw={'width_ratios': [n_mice, 1]})

    cat_labels = ['GP only', 'Time only', 'Both', 'Neither']
    cat_colors = ['#d62728', '#1f77b4', '#7a4f9e', '#d3d3d3']

    ax = axes[0]
    bar_bottom = np.zeros(n_mice)
    totals = np.array([per_mouse[m].sum() for m in mouse_ids])
    for k, (lbl, c) in enumerate(zip(cat_labels, cat_colors)):
        counts_k = np.array([per_mouse[m][k] for m in mouse_ids])
        pcts = 100 * counts_k / np.where(totals > 0, totals, 1)
        ax.bar(np.arange(n_mice), pcts, bottom=bar_bottom,
               color=c, edgecolor='white', linewidth=0.5, label=lbl)
        bar_bottom += pcts
    ax.set_xticks(np.arange(n_mice))
    ax.set_xticklabels([f"{m}\n(n={t})" for m, t in zip(mouse_ids, totals)],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('% of neurons')
    ax.set_ylim(0, 105)
    ax.set_title(
        f"GP vs time tuning per mouse "
        f"({'joint time' if use_joint_time else 'OR-rule across time vars'})"
    )
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)

    # Aggregate pie
    ax = axes[1]
    nonzero = [(c, l, col) for c, l, col in zip(agg_counts, cat_labels, cat_colors) if c > 0]
    if nonzero:
        sizes_, labels_, colors_ = zip(*nonzero)
        ax.pie(sizes_, labels=labels_, autopct='%1.1f%%', startangle=90, colors=colors_)
    total_agg = int(agg_counts.sum())
    ax.set_title(f'All mice pooled\n(n={total_agg})')

    plt.tight_layout()
    plt.show()


def _cpd_value(cpd, reg, normalize):
    """Per-regressor CPD value, optionally normalized to full-model R².

    normalize='none'    → standard CPD = ΔRSS / RSS_reduced
    normalize='r2_full' → ΔR²_reg / R²_full = unique variance as a fraction of
                          the model's total explainable variance.
    Returns np.nan if the regressor or required reserved keys are absent.
    """
    if normalize == 'none':
        return cpd.get(reg, np.nan)
    elif normalize == 'r2_full':
        dr2 = cpd.get('__delta_r2__', {})
        r2f = cpd.get('__r2_full__', np.nan)
        if reg not in dr2 or not np.isfinite(r2f) or r2f <= 0:
            return np.nan
        return dr2[reg] / r2f
    else:
        raise ValueError(f"normalize must be 'none' or 'r2_full', got {normalize!r}")


def plot_cpd_time_vs_progress(CPD_results, group_by_mouse=True, normalize='none'):
    """Compare per-neuron CPD for goal_progress vs time (any).

    CPD_time is taken from `time_any` if present (joint reduced model), else
    falls back to `max(CPD_time_from_reward, CPD_time_to_reward)`.

    Parameters
    ----------
    CPD_results : dict {mouse_recday: {neuron_idx: cpd_dict}}
        Output of `run_glm_analysis(..., compute_cpd=True)`. Reserved keys
        '__r2_full__' / '__delta_r2__' carry the model-fit context.
    group_by_mouse : bool, default True
        Color scatter points by mouse identity.
    normalize : {'none', 'r2_full'}, default 'none'
        'none'    → standard CPD (ΔRSS / RSS_reduced).
        'r2_full' → CPD normalized to full-model R² (fraction of the model's
                    explainable variance uniquely attributable to the regressor).
                    Requires `compute_cpd=True` was used in the fit.

    Returns
    -------
    cpd_gp, cpd_time : 1d arrays of per-neuron values (pooled across recdays)
    """
    cpd_gp = []
    cpd_time = []
    mouse_tag = []

    for mr, neuron_dict in CPD_results.items():
        mouse = mr.split('_')[0]
        for nidx, cpd in neuron_dict.items():
            g = _cpd_value(cpd, 'goal_progress', normalize)
            if 'time_any' in cpd:
                t = _cpd_value(cpd, 'time_any', normalize)
            else:
                t_f = _cpd_value(cpd, 'time_from_reward', normalize)
                t_t = _cpd_value(cpd, 'time_to_reward', normalize)
                t = np.nanmax([t_f, t_t]) if not (np.isnan(t_f) and np.isnan(t_t)) else np.nan
            cpd_gp.append(g)
            cpd_time.append(t)
            mouse_tag.append(mouse)

    cpd_gp = np.array(cpd_gp, dtype=float)
    cpd_time = np.array(cpd_time, dtype=float)
    mouse_tag = np.array(mouse_tag)

    valid = ~np.isnan(cpd_gp) & ~np.isnan(cpd_time)
    cpd_gp = cpd_gp[valid]
    cpd_time = cpd_time[valid]
    mouse_tag = mouse_tag[valid]

    metric_label = 'CPD' if normalize == 'none' else 'CPD / R²_full'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: scatter
    ax = axes[0]
    mice = sorted(np.unique(mouse_tag).tolist()) if group_by_mouse else [None]
    cmap = plt.get_cmap('tab10')
    for i, m in enumerate(mice):
        mask = (mouse_tag == m) if m is not None else slice(None)
        ax.scatter(cpd_gp[mask], cpd_time[mask], alpha=0.6, s=20,
                   color=cmap(i % 10), label=m)
    lim = max(0.001, max(np.nanmax(cpd_gp), np.nanmax(cpd_time)) * 1.05)
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f'{metric_label} goal progress')
    ax.set_ylabel(f'{metric_label} time (any)')
    ax.set_title(f'Per-neuron {metric_label}: goal progress vs time')
    if group_by_mouse:
        ax.legend(loc='upper right', fontsize=8, title='Mouse')

    # Panel 2: Δ histogram
    ax = axes[1]
    diffs = cpd_time - cpd_gp
    ax.hist(diffs, bins=40, color='darkorange', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    md = float(np.nanmean(diffs))
    ax.axvline(md, color='red', linewidth=2, label=f'Mean Δ = {md:+.4f}')
    ax.set_xlabel(f'{metric_label}_time − {metric_label}_progress')
    ax.set_ylabel('Number of neurons')
    ax.set_title('Δ (time − progress)')
    ax.legend()

    # Panel 3: mean metric per regressor across all neurons
    ax = axes[2]
    all_regressors_cpd = {}
    for mr, neuron_dict in CPD_results.items():
        for cpd in neuron_dict.values():
            for reg in cpd:
                if reg.startswith('__'):   # skip reserved keys
                    continue
                v = _cpd_value(cpd, reg, normalize)
                all_regressors_cpd.setdefault(reg, []).append(v)
    means = {r: float(np.nanmean(vs)) for r, vs in all_regressors_cpd.items()}
    sems = {r: float(np.nanstd(vs) / np.sqrt(max(1, np.sum(~np.isnan(vs)))))
            for r, vs in all_regressors_cpd.items()}
    regs_sorted = sorted(means.keys(), key=lambda r: -means[r])
    xs = np.arange(len(regs_sorted))
    ax.bar(xs,
           [means[r] for r in regs_sorted],
           yerr=[sems[r] for r in regs_sorted],
           color='steelblue', alpha=0.8, edgecolor='black', capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([_display(r) for r in regs_sorted], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'Mean {metric_label} across neurons')
    ax.set_title(f'Population-mean {metric_label} per regressor')

    plt.suptitle(
        f"{'Raw' if normalize == 'none' else 'R²-normalized'} CPD: time vs goal progress",
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return cpd_gp, cpd_time


def plot_full_model_r2(CPD_results):
    """Histogram of per-neuron full-model R² (from `compute_cpd=True` fits).

    Anchors the magnitude of CPD values: shows how much firing-rate variance the
    full GLM explains at all. Per-mouse panels + pooled.

    Returns
    -------
    r2_by_mouse : dict {mouse: np.array of per-neuron R²}
    """
    from collections import defaultdict
    r2_by_mouse = defaultdict(list)
    for mr, neuron_dict in CPD_results.items():
        mouse = mr.split('_')[0]
        for cpd in neuron_dict.values():
            r2 = cpd.get('__r2_full__', np.nan)
            if np.isfinite(r2):
                r2_by_mouse[mouse].append(r2)
    r2_by_mouse = {m: np.array(v) for m, v in sorted(r2_by_mouse.items())}

    pooled = np.concatenate(list(r2_by_mouse.values())) if r2_by_mouse else np.array([])

    n_mice = len(r2_by_mouse)
    fig, axes = plt.subplots(1, n_mice + 1, figsize=(4 * (n_mice + 1), 4), squeeze=False)
    axes = axes[0]

    for i, (mouse, r2) in enumerate(r2_by_mouse.items()):
        ax = axes[i]
        ax.hist(r2, bins=30, color='seagreen', alpha=0.7, edgecolor='black')
        ax.axvline(np.median(r2), color='red', linewidth=2,
                   label=f'median = {np.median(r2):.3f}')
        ax.set_xlabel('Full-model R²')
        if i == 0:
            ax.set_ylabel('Number of neurons')
        ax.set_title(f'Mouse {mouse}\n(n={len(r2)})')
        ax.legend(fontsize=8)

    # Pooled
    ax = axes[-1]
    ax.hist(pooled, bins=40, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.axvline(np.median(pooled), color='red', linewidth=2,
               label=f'median = {np.median(pooled):.3f}')
    ax.set_xlabel('Full-model R²')
    ax.set_title(f'All mice pooled\n(n={len(pooled)})')
    ax.legend(fontsize=8)

    plt.suptitle('Full-model R² per neuron (how much variance the GLM explains)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return r2_by_mouse


# ============================================================================
# Decile-edge sanity check (pre-fit diagnostic)
# ============================================================================

# Maps canonical regressor names → per-session array keys in prepare_session_data
_DECILE_PREP_KEYS = {
    'time_from_reward':     'time_from_reward',
    'time_to_reward':       'time_to_reward',
    'distance_from_reward': 'dist_from_reward',
    'distance_to_reward':   'dist_to_reward',
    'speed':                'Speed',
    'acceleration':         'Acc',
}


def plot_decile_distributions(mouse_recdays, data_dic,
                              downsample_factor=10,
                              regressors=None,
                              n_bins=10, outlier_pct=1,
                              max_recdays=None):
    """Histogram the pooled regressor values that feed `compute_decile_edges`,
    with the decile edges and outlier-clip percentiles overlaid.

    Pools the SAME values that `run_glm_analysis` uses internally (after the
    Locs<=21 node filter and after downsampling), per regressor across all
    included recdays. Lets you sanity-check whether each variable's bin
    boundaries land where you'd expect (e.g. that time_from_reward isn't so
    spiky at 0 that the first few deciles collapse).

    Parameters
    ----------
    mouse_recdays, data_dic : as in `run_glm_analysis`.
    downsample_factor : int
        Match what you pass to `run_glm_analysis`.
    regressors : list of str or None
        Subset of the 6 deciled regressors to inspect. Accepts aliases
        (`time_since_reward` etc.). None → all 6.
    n_bins, outlier_pct : as in `compute_decile_edges`.
    max_recdays : int or None
        Cap on recdays to pool (None = all). Use 3–5 for a fast pass.

    Returns
    -------
    pooled : dict {canonical_name: np.ndarray of finite pooled values}
    """
    if regressors is None:
        regressors = ['time_from_reward', 'time_to_reward',
                      'distance_from_reward', 'distance_to_reward',
                      'speed', 'acceleration']
    # Resolve aliases and validate
    canonical = [_REGRESSOR_NAME_ALIASES.get(r, r) for r in regressors]
    invalid = [r for r in canonical if r not in _DECILE_PREP_KEYS]
    if invalid:
        raise ValueError(f"plot_decile_distributions only supports deciled regressors. "
                         f"Unknown / non-deciled: {invalid}. "
                         f"Valid: {list(_DECILE_PREP_KEYS.keys())}")

    recdays_used = mouse_recdays[:max_recdays] if max_recdays else list(mouse_recdays)
    pooled = {r: [] for r in canonical}

    for mr in recdays_used:
        sessions, _ = get_sessions_for_glm(data_dic[mr])
        for s in sessions:
            prep = prepare_session_data(data_dic[mr][s])
            prep = truncate_all_arrays(prep)
            prep = downsample_session_data(prep, downsample_factor)
            nf = prep['Locs'] <= 21
            for r in canonical:
                vals = prep[_DECILE_PREP_KEYS[r]][nf]
                pooled[r].append(vals)
    pooled = {r: np.concatenate(v) if v else np.array([])
              for r, v in pooled.items()}

    # Filter to finite values for plotting / edge computation
    pooled_finite = {r: v[np.isfinite(v)] for r, v in pooled.items()}

    n = len(canonical)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, r in zip(axes, canonical):
        vals = pooled_finite[r]
        if len(vals) < 2 * n_bins:
            ax.set_title(f'{_display(r)} (insufficient data n={len(vals)})')
            ax.axis('off')
            continue

        lo, hi = np.percentile(vals, [outlier_pct, 100 - outlier_pct])
        edges = compute_decile_edges(vals, n_bins=n_bins, outlier_pct=outlier_pct)
        # Inner edges only (compute_decile_edges replaces the outer two with +-inf)
        inner = edges[1:-1]

        # Plot histogram restricted to ~[lo, hi] for readability + a small margin
        margin = 0.02 * (hi - lo if hi > lo else 1.0)
        plot_lo, plot_hi = lo - margin, hi + margin
        ax.hist(vals[(vals >= plot_lo) & (vals <= plot_hi)],
                bins=80, color='steelblue', alpha=0.7, edgecolor='black',
                linewidth=0.3)

        for e in inner:
            ax.axvline(e, color='red', lw=1.0, alpha=0.7)
        ax.axvline(lo, color='gray', ls='--', lw=1.0,
                   label=f'{outlier_pct}th / {100 - outlier_pct}th pct')
        ax.axvline(hi, color='gray', ls='--', lw=1.0)

        clip_frac = float(np.mean((vals < lo) | (vals > hi)))
        ax.set_title(f'{_display(r)}\nn={len(vals)}  clipped={100 * clip_frac:.1f}%')
        ax.set_xlabel(_display(r))
        ax.set_ylabel('count')
        ax.set_xlim(plot_lo, plot_hi)
        ax.legend(fontsize=8, loc='upper right')

    for ax in axes[n:]:
        ax.axis('off')

    plt.suptitle(f'Decile sanity check: pooled values + bin edges '
                 f'(downsample={downsample_factor}, recdays={len(recdays_used)})',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return pooled_finite


# ============================================================================
# Save / load + light publication style
# ============================================================================
#
# Lightweight section-level wrappers for caching fits and exporting figures.
# Convention: every artifact lives under
#     {save_dir}/{section_name}__{key}.pkl    (result dicts)
#     {save_dir}/{section_name}__{name}.pdf   (figures)
# Persistent dir, section-prefixed filenames; re-running a fit overwrites the
# pickle, and `run_or_load_glm` short-circuits the fit when all expected
# pickles already exist.
#
# Figure styling follows the .claude/skills/gridmaze-plotter SKILL: Arial 8pt,
# top/right spines hidden, Type-42 fonts for editable PDFs. Colors of the
# existing plotting functions are left alone (per user pick).


def apply_gridmaze_style():
    """Idempotent rcParams setter applying the gridmaze-plotter publication
    style: Arial 8pt + hidden top/right spines + Type-42 PDF fonts. Call once
    at notebook top (before any plotting) so every subsequent figure picks it up.

    Does NOT change color settings — the existing plot functions' steelblue /
    crimson / darkorange / etc. references stay as-is.
    """
    import matplotlib as mpl

    rc = {
        # Fonts — Arial primary with a portable fallback for headless boxes.
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'DejaVu Sans'],
        'font.size':          8,
        'figure.titlesize':   8,
        'axes.titlesize':     8,
        'axes.labelsize':     8,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'legend.fontsize':    8,

        # Axes furniture
        'axes.linewidth':     0.8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,

        # Editable PDFs (embed Type-42; required for Illustrator round-trips).
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    }
    mpl.rcParams.update(rc)


# ----- low-level pickle / fs helpers -------------------------------------

def _ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)


def _save_pickle(obj, path):
    import os, pickle
    _ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def _artifact_path(save_dir, section_name, key, ext):
    import os
    return os.path.join(save_dir, f'{section_name}__{key}.{ext}')


# ----- section-level save -----------------------------------------------

# Result keys that `save_section` / `load_glm_results` understand for fitted
# artifacts. Extra entries in `results` are saved verbatim under their key.
_GLM_RESULT_KEYS = ('glm_results', 'permutation_results', 'cpd_results',
                    'tuned_dict', 'mouse_tuning_concat')


def save_section(section_name, save_dir, *, results=None,
                 fig_names=None, close_after=False):
    """Save one section's outputs in one call.

    Parameters
    ----------
    section_name : str
        Filename prefix and cache key (e.g. 'baseline', 'extended_cpd').
    save_dir : str
        Output directory. Created if missing.
    results : dict {str: obj}, optional
        Each value is pickled to `{save_dir}/{section_name}__{key}.pkl`.
        Use the canonical keys when applicable so `run_or_load_glm` /
        `load_glm_results` can pick them up:
        {_GLM_RESULT_KEYS}.
    fig_names : list of str, optional
        Names for currently-open figures in `plt.get_fignums()` order. Saved
        as `{save_dir}/{section_name}__{name}.pdf`. Missing/extra names
        fall back to `fig1`, `fig2`, ... `None` → use auto-names for all.
    close_after : bool, default False
        Close saved figures after saving (memory hygiene for long notebooks;
        breaks inline display so leave False in Jupyter).

    Returns
    -------
    written : list of str
        Absolute paths of every file written.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    _ensure_dir(save_dir)
    written = []

    # ---- pickle each result dict --------------------------------------
    if results:
        for key, obj in results.items():
            path = _artifact_path(save_dir, section_name, key, 'pkl')
            _save_pickle(obj, path)
            written.append(path)

    # ---- save every currently-open figure as PDF ----------------------
    fignums = plt.get_fignums()
    if fig_names is None:
        fig_names = []
    save_rc = {
        'savefig.bbox':       None,
        'savefig.pad_inches': 0.0,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    }
    with mpl.rc_context(save_rc):
        for i, num in enumerate(fignums):
            name = fig_names[i] if i < len(fig_names) and fig_names[i] else f'fig{i+1}'
            path = _artifact_path(save_dir, section_name, name, 'pdf')
            fig = plt.figure(num)
            fig.savefig(path, bbox_inches=None)
            written.append(path)
            if close_after:
                plt.close(fig)

    print(f'save_section({section_name!r}): wrote {len(written)} file(s) to {save_dir}')
    return written


# ----- run-or-load (skip-if-exists fit wrapper) -------------------------

def run_or_load_glm(mouse_recdays, data_dic, save_dir, section_name,
                    *, force_refit=False, **kwargs):
    """Skip-if-exists wrapper around `run_glm_analysis`.

    If `{save_dir}/{section_name}__glm_results.pkl` (plus permutation_results
    and, when `compute_cpd=True`, cpd_results) already exist and
    `force_refit=False`, loads them and returns the same tuple shape
    `run_glm_analysis(**kwargs)` would. Otherwise fits, pickles, returns.

    Use the SAME `section_name` for both `run_or_load_glm` and `save_section`
    so the cache key is consistent.
    """
    import os
    compute_cpd = bool(kwargs.get('compute_cpd', False))

    needed = ['glm_results', 'permutation_results']
    if compute_cpd:
        needed.append('cpd_results')

    paths = {k: _artifact_path(save_dir, section_name, k, 'pkl') for k in needed}

    if not force_refit and all(os.path.exists(p) for p in paths.values()):
        print(f'run_or_load_glm({section_name!r}): loading cached results from {save_dir}')
        loaded = tuple(_load_pickle(paths[k]) for k in needed)
        return loaded

    print(f'run_or_load_glm({section_name!r}): no cache (or force_refit) — fitting')
    result = run_glm_analysis(mouse_recdays, data_dic, **kwargs)

    _ensure_dir(save_dir)
    for k in needed:
        idx = needed.index(k)
        _save_pickle(result[idx], paths[k])
    print(f'run_or_load_glm({section_name!r}): saved {len(needed)} pickle(s) to {save_dir}')
    return result


# ----- manual loader ----------------------------------------------------

def load_glm_results(save_dir, section_name):
    """Return whichever section pickles exist as a dict.

    Looks for the canonical `_GLM_RESULT_KEYS` plus any other `*.pkl` files
    whose name starts with `{section_name}__`. Returns
    `{key: unpickled_object}`; missing keys are simply omitted.
    """
    import os, glob

    out = {}
    prefix = f'{section_name}__'

    # canonical keys first (stable order)
    for k in _GLM_RESULT_KEYS:
        p = _artifact_path(save_dir, section_name, k, 'pkl')
        if os.path.exists(p):
            out[k] = _load_pickle(p)

    # plus any other section pickles
    for p in sorted(glob.glob(os.path.join(save_dir, f'{prefix}*.pkl'))):
        key = os.path.basename(p)[len(prefix):-len('.pkl')]
        if key in out:
            continue
        out[key] = _load_pickle(p)

    if not out:
        print(f'load_glm_results({section_name!r}): no pickles found in {save_dir}')
    return out
