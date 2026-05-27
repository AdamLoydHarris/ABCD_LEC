"""
GLM analysis with no-intercept, keep-all-bins parameterization.

Sibling of glm_analysis.py. Differences:
  1. No intercept column.
  2. One-hot encodings keep ALL bins (no reference-bin dropping):
     - place: 21 bins (locs 1..21)
     - head direction: 36 bins (10 deg bins)
     - goal progress: 10 bins
     - speed, acc, time/distance from/to reward: 10 decile bins each
     Total design matrix: 127 columns.
  3. Design matrix is rank-deficient by 8 (each one-hot block sums to 1 per row,
     so without an intercept the blocks have 8 independent linear dependencies).
     np.linalg.lstsq returns the minimum-norm solution; individual betas are no
     longer uniquely identified, but predictions and RSS_full / RSS_reduced for
     the nested F-test ARE unique (column space is invariant). So F-tests give
     mathematically identical results to v1; only beta interpretation differs.

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


def prepare_session_data(session_data):
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
            Trial_times_bins, num_bins=10
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

    return {
        'FR': FR,
        'Locs': Locs,
        'HD': HD,
        'Speed': Speed,
        'Acc': Acc,
        'State': State,
        'GP_binned': GP_binned,
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


def _resolve_regressor_groups(regressors_to_include):
    """Resolve (regressor_groups, analysis_regressor_names) for a subset.

    If `regressors_to_include` is None, returns the module-level defaults
    (all 9 regressors). Otherwise returns a NEW regressor_groups dict whose
    indices reflect a design matrix built from only the included regressors,
    in canonical order (the order they appear in `analysis_regressor_names`).
    """
    if regressors_to_include is None:
        return regressor_groups, list(analysis_regressor_names)

    invalid = [r for r in regressors_to_include if r not in analysis_regressor_names]
    if invalid:
        raise ValueError(f"Unknown regressor names: {invalid}. "
                         f"Valid names: {analysis_regressor_names}")

    ordered = [r for r in analysis_regressor_names if r in regressors_to_include]
    n_cols_per = {name: len(regressor_groups[name]) for name in analysis_regressor_names}

    local = {}
    cursor = 0
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
                     regressors_to_include=None):
    """Fit per-neuron OLS GLM with permutation F-tests.

    Parameters
    ----------
    mouse_recdays, data_dic : see module docstring.
    num_permutations : int, default 100
        Number of circular firing-rate shifts in the permutation null.
    downsample_factor : int, default 10
    regressors_to_include : list of str or None
        Subset of `analysis_regressor_names` to include in the design matrix.
        None (default) keeps all 9. Pass a list to exclude variables — e.g.
        `['place', 'head_direction', 'goal_progress', 'speed', 'acceleration',
        'time_from_reward', 'distance_from_reward']` to drop the two
        forward-looking variables.
    """
    local_groups, local_names = _resolve_regressor_groups(regressors_to_include)

    GLM_results = {}
    Permutation_results = {}

    for mouse_recday in tqdm(mouse_recdays, desc="Processing recording days"):
        print(f"\n{mouse_recday}")

        GLM_results[mouse_recday] = {}
        Permutation_results[mouse_recday] = {}

        sessions_for_glm, _ = get_sessions_for_glm(data_dic[mouse_recday])

        if len(sessions_for_glm) < 2:
            print(f"  Skipping — not enough valid sessions ({len(sessions_for_glm)})")
            continue

        first_session = sessions_for_glm[0]
        num_neurons = data_dic[mouse_recday][first_session]['Neuron_raw'].shape[0]

        prepared_sessions = {}
        for session in sessions_for_glm:
            prep_data = prepare_session_data(data_dic[mouse_recday][session])
            prep_data = truncate_all_arrays(prep_data)
            prepared_sessions[session] = downsample_session_data(prep_data, downsample_factor)

        # ----------------------------------------------------------------
        # Pool all behavioral data across sessions (valid locations only)
        # ----------------------------------------------------------------
        all_speed, all_acc   = [], []
        all_tf, all_tt       = [], []
        all_df, all_dt       = [], []
        all_locs, all_hd, all_gp = [], [], []
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

        # GP: 10 equal-width bins → 10 cols
        GP_onehot = (gp_all[:, None] == np.arange(0, 10)).astype(float)

        # Continuous vars: decile bins → 10 cols each
        Sp_oh = apply_onehot(speed_all, speed_edges)
        Ac_oh = apply_onehot(acc_all,   acc_edges)
        TF_oh = apply_onehot(tf_all,    tf_edges)
        TT_oh = apply_onehot(tt_all,    tt_edges)
        DF_oh = apply_onehot(df_all,    df_edges)
        DT_oh = apply_onehot(dt_all,    dt_edges)

        onehots = {
            'place':                place_onehot,
            'head_direction':       HD_onehot,
            'goal_progress':        GP_onehot,
            'speed':                Sp_oh,
            'acceleration':         Ac_oh,
            'time_from_reward':     TF_oh,
            'time_to_reward':       TT_oh,
            'distance_from_reward': DF_oh,
            'distance_to_reward':   DT_oh,
        }
        X = np.column_stack([onehots[name] for name in local_names])
        X = np.nan_to_num(X)
        print(f"  Design matrix: {X.shape[0]} rows × {X.shape[1]} cols "
              f"(regressors: {local_names})")

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

                # --- Permuted full models (all at once) ---
                perm_frs        = np.stack([np.roll(frs, s) for s in shifts]).T  # [T × n_perms]
                beta_perms      = np.linalg.lstsq(X, perm_frs, rcond=None)[0]    # [n_params × n_perms]
                resid_full_p    = perm_frs - X @ beta_perms                       # [T × n_perms]
                rss_full_p      = np.einsum('ij,ij->j', resid_full_p, resid_full_p)  # [n_perms]

                # --- F-stat per regressor group ---
                F_real = {}
                F_perm = {}
                for reg_name, X_r in X_reduced_dict.items():
                    df_num = len(local_groups[reg_name])

                    # Real reduced model
                    params_r, _, _, _ = np.linalg.lstsq(X_r, frs, rcond=None)
                    resid_r   = frs - X_r @ params_r
                    rss_r     = resid_r @ resid_r
                    F_real[reg_name] = ((rss_r - rss_full) / df_num) / (rss_full / df_resid)

                    # Permuted reduced models (vectorised)
                    beta_perms_r = np.linalg.lstsq(X_r, perm_frs, rcond=None)[0]  # [n_r × n_perms]
                    resid_r_p    = perm_frs - X_r @ beta_perms_r                   # [T × n_perms]
                    rss_r_p      = np.einsum('ij,ij->j', resid_r_p, resid_r_p)    # [n_perms]
                    F_perm[reg_name] = ((rss_r_p - rss_full_p) / df_num) / (rss_full_p / df_resid)

                GLM_results[mouse_recday][neuron]       = params
                Permutation_results[mouse_recday][neuron] = (F_real, F_perm)

            except Exception:
                continue

    return GLM_results, Permutation_results


# ============================================================================
# Significance analysis
# ============================================================================

# Regressors where betas are ordered by bin — slope captures ramp direction
_ORDERED_REGRESSORS = {
    'goal_progress', 'speed', 'acceleration',
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


def compute_tuning_arrays(GLM_results, Permutation_results, regressors_to_include=None):
    """
    Returns tuned_dict: {mouse_recday: tuning_array [n_neurons × n_regressors]}
    Values: +1 (positive / ramp-up tuning), -1 (negative / ramp-down), 0 (not significant).
    Significance: permutation F-test at p < 0.05 (one-sided on F).
    Direction: slope of beta profile for ordered regressors; sign of mean beta for unordered.

    Pass the same `regressors_to_include` list used in `run_glm_analysis` so that
    `_beta_direction` uses the correct (subset-specific) index map.
    """
    local_groups, local_names = _resolve_regressor_groups(regressors_to_include)
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
