"""
Cross-task remapping + coherent rotation (El-Gaby et al. 2024, Figure3) — reimplemented on our
data_dic. The reference notebook (code/Figure3.ipynb) can't be run (private Input_folder), so
this ports its method faithfully.

Two analyses, both on per-neuron, per-task 360-bin goal-progress tuning curves
(90 bins/state x 4 states, trial-averaged, smoothed sigma=10):

  REMAPPING (single cells): for each state-tuned neuron, circularly cross-correlate its tuning
  curve between a reference task and each other task; the rotation offset (deg; one state = 90)
  that best re-aligns them. Distribution near 0 => cell keeps its task-state tuning (generalises);
  spread => remapping.

  COHERENT ROTATION (cell pairs): even if cells remap, test whether each pair's *relative*
  rotation (circular_angle of the two cells' rotations vs the reference task) stays small
  (<45 deg) across tasks => the population rotates as a rigid body. Proportion coherent vs a
  ~1/num_states chance level, plus a breakdown by the pair's initial tuning distance.

  X-vs-X' control: the same computations between repeated identical-task sessions (rotation ~0,
  coherence high expected).

Cells are index-matched across the sessions of a recday (our v2/v3 CV pipeline relies on this).
Required fields per session: 'Neuron_raw' (n_neurons, n_bins), 'Trial_times' (n_trials, n_states+1)
in 25-ms bin indices, 'Task'.
"""

import math
import numpy as np
from scipy import stats
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Config
# ============================================================================

class RemapConfig:
    def __init__(self, num_task_states=4, num_goal_progress_bins=3, num_bins_per_state=90,
                 smoothing_sigma=10, rotation_step_single=2, rotation_step_pair=10,
                 coherence_threshold_deg=45, state_tuning_p_threshold=0.05,
                 min_tasks_tuned_frac=0.5, min_neurons=10, max_comparisons=2,
                 n_clustering_perms=100, max_clusters=12,
                 tsne_perplexity=5, cluster_distance_threshold='auto',
                 cluster_threshold_percentile=40):
        self.num_task_states = num_task_states
        self.num_goal_progress_bins = num_goal_progress_bins
        self.num_bins_per_state = num_bins_per_state
        self.total_bins = num_bins_per_state * num_task_states            # 360
        self.bins_per_phase = num_bins_per_state // num_goal_progress_bins
        self.smoothing_sigma = smoothing_sigma
        self.rotation_step_single = rotation_step_single                  # deg/bins
        self.rotation_step_pair = rotation_step_pair
        self.coherence_threshold_deg = coherence_threshold_deg
        self.coherence_thr = 1 - math.cos(math.radians(coherence_threshold_deg))
        self.state_tuning_p_threshold = state_tuning_p_threshold
        self.min_tasks_tuned_frac = min_tasks_tuned_frac
        self.min_neurons = min_neurons
        # number of non-reference tasks to compare against (El-Gaby uses 2 => 3-task days);
        # None = use all available unique tasks (much stricter dual-coherence metric).
        self.max_comparisons = max_comparisons
        # Fig 3d clustering: agglomerative clustering on the incoherence matrix (precomputed
        # distances); cluster count chosen by max silhouette over 2..max_clusters.
        self.n_clustering_perms = n_clustering_perms
        self.max_clusters = max_clusters
        # (legacy, unused) El-Gaby's t-SNE-embedding clustering params — kept for reference;
        # superseded by direct incoherence-matrix clustering (see cluster_and_silhouette).
        self.tsne_perplexity = tsne_perplexity
        self.cluster_distance_threshold = cluster_distance_threshold
        self.cluster_threshold_percentile = cluster_threshold_percentile


# ============================================================================
# Small helpers (ported from Figure3.ipynb cell 9)
# ============================================================================

def smooth_circular(x, sigma=10):
    return gaussian_filter1d(np.hstack((x, x, x)), sigma, axis=0)[len(x):int(len(x) * 2)]


def circular_angle(x, y):
    """Signed circular difference in degrees, in (-180, 180]."""
    return (x - y + 180) % 360 - 180


def positive_angle(xx):
    """Map an array of angles to [0, 360)."""
    xx = np.asarray(xx, dtype=float)
    out = np.where(xx < 0, xx + 360, xx).astype(int)
    out[out == 360] = 0
    return out


def matrix_triangle(a, direction='upper'):
    if direction == 'upper':
        idx = np.triu_indices(len(a), k=1)
    else:
        idx = np.tril_indices(len(a), k=-1)
    return a[idx]


def max_bin_safe(xx):
    """argmax of a 1-D array; NaN if the max is tied or the array is all-NaN."""
    if np.all(np.isnan(xx)):
        return np.nan
    xx_max = np.nanmax(xx)
    max_bins = np.where(xx == xx_max)[0]
    return max_bins[0] if len(max_bins) == 1 else np.nan


def two_proportions_test(success_a, size_a, success_b, size_b):
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    prop_pooled = (success_a + success_b) / (size_a + size_b)
    var = prop_pooled * (1 - prop_pooled) * (1 / size_a + 1 / size_b)
    if var <= 0:
        return np.nan, np.nan
    z = np.abs(prop_b - prop_a) / np.sqrt(var)
    p = (1 - stats.norm(0, 1).cdf(z)) * 2
    return z, p


def _stars(p):
    if not np.isfinite(p):
        return ''
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


# ============================================================================
# 360-bin tuning + state-tuning (ported from elasticnet_regression_v3.py)
# ============================================================================

def raw_to_norm(raw_1d, trial_times, config, return_mean=True, statistic='mean'):
    """Normalize a raw per-bin 1-D signal to a per-trial 360-bin grid (90 bins/state)."""
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
        rebinned.append(binned_statistic(idx, seg, statistic=statistic, bins=nbps)[0])
    n_full = (len(rebinned) // nstates) * nstates
    if n_full == 0:
        return None
    arr = np.asarray(rebinned[:n_full]).reshape(n_full // nstates, nbps * nstates)
    return np.nanmean(arr, axis=0) if return_mean else arr


def identify_state_tuned_neurons(neuron_raw, trial_times, config):
    """Boolean mask: peak firing per state per trial -> z across states -> t-test preferred."""
    p_thr = config.state_tuning_p_threshold
    n_neurons = neuron_raw.shape[0]
    nstates, nbps = config.num_task_states, config.num_bins_per_state
    tuned = np.zeros(n_neurons, dtype=bool)
    for ni in range(n_neurons):
        per_trial = raw_to_norm(neuron_raw[ni], trial_times, config, return_mean=False)
        if per_trial is None or per_trial.shape[0] < 3:
            continue
        peak = np.full((per_trial.shape[0], nstates), np.nan)
        for s in range(nstates):
            peak[:, s] = np.nanmax(per_trial[:, s * nbps:(s + 1) * nbps], axis=1)
        rm = np.nanmean(peak, axis=1, keepdims=True)
        rs = np.nanstd(peak, axis=1, keepdims=True)
        rs[rs == 0] = np.nan
        z = (peak - rm) / rs
        pref = int(np.nanargmax(np.nanmean(z, axis=0)))
        zp = z[:, pref][~np.isnan(z[:, pref])]
        if len(zp) >= 3 and stats.ttest_1samp(zp, 0)[1] < p_thr:
            tuned[ni] = True
    return tuned


# ============================================================================
# Per-session tuning curves
# ============================================================================

def build_session_tuning(session_data, config):
    """Return (tuning (n_neurons, 360) smoothed, state_tuned_mask (n_neurons,)) or None."""
    for k in ('Neuron_raw', 'Trial_times'):
        if k not in session_data or session_data[k] is None:
            return None
    neuron_raw = np.asarray(session_data['Neuron_raw'], dtype=float)
    tt = np.asarray(session_data['Trial_times'], dtype=int)
    if neuron_raw.ndim != 2 or tt.ndim != 2 or tt.shape[0] < 2:
        return None
    n_neurons = neuron_raw.shape[0]
    tuning = np.full((n_neurons, config.total_bins), np.nan)
    for ni in range(n_neurons):
        curve = raw_to_norm(neuron_raw[ni], tt, config, return_mean=True)
        if curve is not None and not np.all(np.isnan(curve)):
            tuning[ni] = smooth_circular(np.nan_to_num(curve, nan=0.0),
                                         sigma=config.smoothing_sigma)
    mask = identify_state_tuned_neurons(neuron_raw, tt, config)
    return tuning, mask


# ============================================================================
# Rotation (circular cross-correlation)
# ============================================================================

def best_rotation(curve_x, curve_y, step):
    """Rotation (deg) that best aligns curve_y to curve_x: roll y by each offset, Pearson,
    argmax. Returns NaN if undefined."""
    if (np.all(np.isnan(curve_x)) or np.all(np.isnan(curve_y))
            or np.nanstd(curve_x) == 0 or np.nanstd(curve_y) == 0):
        return np.nan
    offsets = np.arange(0, 360, step)
    corrs = np.full(len(offsets), np.nan)
    for i, off in enumerate(offsets):
        yr = np.roll(curve_y, int(off))
        v = ~np.isnan(curve_x) & ~np.isnan(yr)
        if np.sum(v) > 10 and np.std(curve_x[v]) > 0 and np.std(yr[v]) > 0:
            corrs[i] = stats.pearsonr(curve_x[v], yr[v])[0]
    mb = max_bin_safe(corrs)
    return np.nan if (mb is np.nan or (isinstance(mb, float) and np.isnan(mb))) else mb * step


def rotations_vs_reference(tunings, included_idx, ref_si, other_sis, step):
    """angles[i, c] = rotation of included neuron i in other task c vs reference task ref_si.
    tunings: list/array of (n_neurons, 360) per session."""
    angles = np.full((len(included_idx), len(other_sis)), np.nan)
    for c, sj in enumerate(other_sis):
        for ii, ni in enumerate(included_idx):
            angles[ii, c] = best_rotation(tunings[ref_si][ni], tunings[sj][ni], step)
    return angles


def pairwise_tuning_angles_ref(tuning_ref, included_idx, step):
    """Pairwise rotation aligning neuron Y to X within the reference task -> (n_inc, n_inc)."""
    n = len(included_idx)
    mat = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            if a == b:
                mat[a, b] = 0.0
            else:
                mat[a, b] = best_rotation(tuning_ref[included_idx[a]],
                                          tuning_ref[included_idx[b]], step)
    return mat


# ============================================================================
# Per-recday analysis
# ============================================================================

def included_from_masks(masks, config):
    """Cells state-tuned in >= min_tasks_tuned_frac of the given task masks."""
    frac = np.mean(np.vstack(masks), axis=0)
    return np.where(frac >= config.min_tasks_tuned_frac)[0]


def _relative_pairs(angles_per_comparison):
    """Given angles (n_neurons, n_comparisons) of single-neuron rotations vs ref, return the
    upper-triangle pairwise relative rotations -> (n_pairs, n_comparisons) in [0, 360)."""
    n_comp = angles_per_comparison.shape[1]
    out = []
    for c in range(n_comp):
        a = angles_per_comparison[:, c]
        mat = np.vstack([positive_angle([circular_angle(a[x], a[y]) for x in range(len(a))])
                         for y in range(len(a))]).astype(float)
        out.append(matrix_triangle(mat, 'upper'))
    return np.asarray(out).T  # (n_pairs, n_comparisons)


def analyse_recday(data_dic, mouse_recday, valid_sessions, config, verbose=False):
    """Compute cross-task remapping + coherent rotation (and X-vs-X' control) for one recday.
    Returns a dict, or None if not enough usable tasks/neurons."""
    sess = data_dic[mouse_recday]
    # build tunings for unique tasks
    built = {}
    n_neurons = None
    for s in valid_sessions:
        if s not in sess:
            continue
        bt = build_session_tuning(sess[s], config)
        if bt is None:
            continue
        if n_neurons is None:
            n_neurons = bt[0].shape[0]
        elif bt[0].shape[0] != n_neurons:
            if verbose:
                print(f"  {mouse_recday}: neuron-count mismatch across sessions, skipping")
            return None
        built[s] = bt
    used = list(built.keys())
    if len(used) < 2:
        return None

    tunings = {s: built[s][0] for s in used}
    masks = [built[s][1] for s in used]
    included = included_from_masks(masks, config)
    if len(included) == 0:
        return None

    ref = used[0]
    others = used[1:]
    if config.max_comparisons is not None:
        others = others[:config.max_comparisons]   # El-Gaby parity: ref + first N other tasks
    step_s = config.rotation_step_single

    # --- remapping: single-neuron rotations vs ref ---
    angles_vs_ref = rotations_vs_reference(tunings, included, ref, others, step_s)  # (n_inc, n_others)

    out = {
        'mouse_recday': mouse_recday,
        'used_sessions': used,
        'n_included': len(included),
        'single_neuron_angles': angles_vs_ref,          # (n_inc, n_comparisons)
    }

    # --- coherent rotation: pairwise relative rotations ---
    if len(included) >= config.min_neurons and len(others) >= 2:
        rel_pairs = _relative_pairs(angles_vs_ref)        # (n_pairs, n_comparisons)
        out['relative_pairs'] = rel_pairs
        cos_per = 1 - np.cos(np.deg2rad(rel_pairs))
        cos_max = np.nanmax(cos_per, axis=1)
        valid = ~np.isnan(cos_max)
        out['coherent_prop'] = (np.sum(cos_max[valid] < config.coherence_thr) / np.sum(valid)
                                if np.sum(valid) else np.nan)
        # single-comparison coherent proportion (every pair x comparison; chance 1/num_states)
        sc_valid = ~np.isnan(cos_per)
        out['single_comp_coherent_prop'] = (np.sum(cos_per[sc_valid] < config.coherence_thr)
                                            / np.sum(sc_valid) if np.sum(sc_valid) else np.nan)
        # tuning-distance breakdown (initial pairwise tuning angle in ref task)
        tune_mat = pairwise_tuning_angles_ref(tunings[ref], included, config.rotation_step_pair)
        tune_dist = 1 - np.cos(np.deg2rad(matrix_triangle(tune_mat, 'upper')))
        bins = np.linspace(0, 360, 5)[:-1] / 2  # [0,45,90,135] tuning bins
        tb = np.full(len(bins), np.nan)
        tb_counts = np.zeros((len(bins), 2))     # [coherent, total] per bin (for pooled stats)
        for k, t in enumerate(bins):
            lo = 1 - math.cos(math.radians(t))
            hi = 1 - math.cos(math.radians(t + 45))
            sel = (tune_dist >= lo) & (tune_dist < hi) & valid
            n_sel = int(np.sum(sel))
            n_coh = int(np.sum(cos_max[sel] < config.coherence_thr))
            tb_counts[k] = [n_coh, n_sel]
            if n_sel:
                tb[k] = n_coh / n_sel
        out['coherence_by_tuning_dist'] = tb
        out['coherence_by_tuning_counts'] = tb_counts
        out['n_comparisons'] = rel_pairs.shape[1]

        # goal-progress-preserving shuffle null for the dual coherent proportion (overall + per bin)
        ov_null, pb_null = coherence_null(angles_vs_ref, tune_dist, config)
        out['coherent_prop_null_mean'] = float(np.nanmean(ov_null))
        out['coherence_by_tuning_null'] = np.nanmean(pb_null, axis=0)

        # --- clustering into coherent modules (Fig 3d,e) ---
        valid_neur = ~np.any(np.isnan(angles_vs_ref), axis=1)
        if np.sum(valid_neur) >= max(config.min_neurons, config.tsne_perplexity + 2):
            a_valid = angles_vs_ref[valid_neur]
            M = incoherence_matrix(a_valid, config)
            labels, n_clusters, sil_real = cluster_and_silhouette(M, config)
            sil_null = permuted_silhouette(a_valid, config)
            out['n_clusters'] = n_clusters
            out['silhouette_real'] = sil_real
            out['silhouette_null_mean'] = np.nanmean(sil_null)
            out['cluster_labels'] = labels
            out['cluster_ref_tunings'] = tunings[ref][included[valid_neur]]  # (n_valid, 360) for Fig 3e

    # --- X vs X' control: ref task vs a repeated identical-task session ---
    ref_task = sess[ref].get('Task')
    rep = None
    for s in sess:
        if s == 'valid_sessions' or s in used or s not in sess or not isinstance(sess[s], dict):
            continue
        if 'Task' in sess[s] and ref_task is not None and np.array_equal(sess[s]['Task'], ref_task):
            bt = build_session_tuning(sess[s], config)
            if bt is not None and bt[0].shape[0] == n_neurons:
                rep = s
                tunings[s] = bt[0]
                break
    if rep is not None:
        angles_X = rotations_vs_reference(tunings, included, ref, [rep], step_s)  # (n_inc, 1)
        out['single_neuron_angles_X'] = angles_X
        if len(included) >= config.min_neurons:
            out['relative_pairs_X'] = _relative_pairs(angles_X)  # (n_pairs, 1)
    if verbose:
        cp = out.get('coherent_prop', np.nan)
        print(f"  {mouse_recday}: {len(used)} tasks, {len(included)} included neurons, "
              f"coherent_prop={cp if isinstance(cp, float) else float('nan'):.3f}"
              + (", X-vs-X' present" if rep is not None else ""))
    return out


# ============================================================================
# Module clustering (Fig 3d,e): incoherence matrix -> t-SNE -> agglomerative -> silhouette
# ============================================================================

def _shuffle_state_offsets(angles_vs_ref, rng, num_states):
    """Goal-progress-preserving state shuffle: keep each rotation's within-state component
    (mod 90) but randomise the state offset (a random multiple of 90 deg) per neuron/comparison.
    Preserves single-neuron goal-progress tuning while destroying cross-task coherence."""
    n, ncomp = angles_vs_ref.shape
    resid = angles_vs_ref % 90
    state_off = rng.integers(0, num_states, size=(n, ncomp)) * 90
    return (state_off + resid) % 360


def incoherence_matrix(angles_vs_ref, config):
    """(n, n) incoherence distance matrix: M[k,j] = max over comparisons of
    (1 - cos(relative single-neuron rotation between k and j)). Symmetric, zero diagonal.
    `angles_vs_ref` (n, n_comparisons) must be NaN-free."""
    n = angles_vs_ref.shape[0]
    M = np.zeros((n, n))
    for c in range(angles_vs_ref.shape[1]):
        a = angles_vs_ref[:, c]
        rel = circular_angle(a[:, None], a[None, :])      # (n, n), (-180, 180]
        M = np.maximum(M, 1 - np.cos(np.deg2rad(rel)))
    np.fill_diagonal(M, 0.0)
    return M


def _agglomerative_precomputed(n_clusters, M):
    """AgglomerativeClustering on a precomputed distance matrix (sklearn metric/affinity compat)."""
    from sklearn.cluster import AgglomerativeClustering
    try:
        return AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed',
                                       linkage='average').fit_predict(M)
    except TypeError:  # older sklearn
        return AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                       linkage='average').fit_predict(M)


def cluster_and_silhouette(M, config):
    """Agglomerative clustering directly on the incoherence (precomputed-distance) matrix; the
    number of clusters is chosen to maximise the silhouette score over 2..max_clusters.

    NOTE — deviation from El-Gaby: the paper embeds M with t-SNE (perplexity=5) then clusters with
    a fixed distance_threshold=300. That threshold is specific to their t-SNE version's coordinate
    scale and does not transfer; worse, t-SNE manufactures apparent clusters even from the
    permutation null, collapsing the real-vs-null silhouette contrast. Clustering the incoherence
    matrix directly (the matrix the method says reflects coherence relationships) is scale-free,
    deterministic, and preserves that contrast. Returns (labels, n_clusters, silhouette)."""
    from sklearn.metrics import silhouette_score
    n = M.shape[0]
    if n < 4:
        return None, np.nan, np.nan
    kmax = min(config.max_clusters, n - 1)
    best_sil, best_labels, best_k = -np.inf, None, np.nan
    for k in range(2, kmax + 1):
        labels = _agglomerative_precomputed(k, M)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(M, labels, metric='precomputed')
        if sil > best_sil:
            best_sil, best_labels, best_k = sil, labels, k
    if best_labels is None:
        return None, np.nan, np.nan
    return best_labels, best_k, best_sil


def permuted_silhouette(angles_vs_ref, config, n_perms=None, seed=0):
    """Null silhouettes: per comparison/neuron keep the within-state (goal-progress) component
    of the rotation but randomise the state offset (multiple of 90 deg), rebuild the incoherence
    matrix and re-cluster. Preserves state tuning in task X and goal-progress tuning in all tasks
    while remapping state preference randomly across tasks (El-Gaby permutation)."""
    if n_perms is None:
        n_perms = config.n_clustering_perms
    rng = np.random.default_rng(seed)
    sils = np.full(n_perms, np.nan)
    for i in range(n_perms):
        a_null = _shuffle_state_offsets(angles_vs_ref, rng, config.num_task_states)
        _, _, sils[i] = cluster_and_silhouette(incoherence_matrix(a_null, config), config)
    return sils


def coherence_null(angles_vs_ref, tune_dist, config, n_perms=None, seed=1):
    """Goal-progress-preserving shuffle null for the DUAL coherent proportion.
    Returns (overall_null (n_perms,), per_bin_null (n_perms, n_tuning_bins)). `tune_dist` is the
    real (fixed) pairwise tuning distance in the reference task, in upper-triangle pair order
    (same order as _relative_pairs)."""
    if n_perms is None:
        n_perms = config.n_clustering_perms
    rng = np.random.default_rng(seed)
    bins = np.linspace(0, 360, 5)[:-1] / 2          # [0, 45, 90, 135]
    overall = np.full(n_perms, np.nan)
    per_bin = np.full((n_perms, len(bins)), np.nan)
    for i in range(n_perms):
        a_null = _shuffle_state_offsets(angles_vs_ref, rng, config.num_task_states)
        rel = _relative_pairs(a_null)                # (n_pairs, n_comparisons)
        cos_max = np.nanmax(1 - np.cos(np.deg2rad(rel)), axis=1)
        valid = ~np.isnan(cos_max)
        if np.sum(valid):
            overall[i] = np.sum(cos_max[valid] < config.coherence_thr) / np.sum(valid)
        for k, t in enumerate(bins):
            lo = 1 - math.cos(math.radians(t))
            hi = 1 - math.cos(math.radians(t + 45))
            sel = (tune_dist >= lo) & (tune_dist < hi) & valid
            if np.sum(sel):
                per_bin[i, k] = np.sum(cos_max[sel] < config.coherence_thr) / np.sum(sel)
    return overall, per_bin


# ============================================================================
# Plotting
# ============================================================================

def _polar_hist(counts, title, save_path=None):
    import matplotlib.pyplot as plt
    n = len(counts)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(theta, counts, width=width, color='black', alpha=0.8, align='edge')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels(['0', '90', '180', '270'])
    ax.set_title(title, fontsize=11)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def _generalising(angles_flat, config, n_bins=36):
    """Histogram (n_bins) + proportion of angles within the bins covering 1/num_states of the
    circle around 0 (matching El-Gaby's first-5 + last-4 of 36 bins for 4 states), tested vs the
    1/num_states chance level. The window is sized to exactly 1/num_states so uniform ~ chance."""
    angles_flat = angles_flat[~np.isnan(angles_flat)]
    hist = np.histogram(angles_flat, np.linspace(0, 360, n_bins + 1))[0]
    gen_bins = max(1, n_bins // config.num_task_states)   # 9 bins => 1/num_states of the circle
    lo = (gen_bins + 1) // 2                               # bins 0..lo-1   (e.g. 0..4)
    hi = gen_bins // 2                                     # last hi bins   (e.g. 32..35)
    gen_num = np.sum(hist[:lo]) + (np.sum(hist[-hi:]) if hi > 0 else 0)
    total = np.sum(hist)
    prop = gen_num / total if total else np.nan
    z, p = two_proportions_test(gen_num, total, total * (1 / config.num_task_states), total) \
        if total else (np.nan, np.nan)
    return hist, prop, z, p


# ============================================================================
# Cross-mouse driver
# ============================================================================

def run_remapping_rotation_all_mice(data_dic, valid_sessions_dic, config=None,
                                     save_dir=None, verbose=True):
    """Run the remapping + coherent-rotation analysis for every recday and pool across mice."""
    import matplotlib.pyplot as plt
    import os
    if config is None:
        config = RemapConfig()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    results = {}
    single_angles, single_angles_X = [], []
    rel_pairs, rel_pairs_X = [], []
    coherent_props, coh_by_tuning, coh_by_tuning_counts = [], [], []
    single_comp_props, coherent_props_null, coh_by_tuning_null = [], [], []
    n_comp_seen = []
    sil_real_all, sil_null_all = [], []

    for mr in list(data_dic.keys()):
        vs = valid_sessions_dic.get(mr) if valid_sessions_dic else None
        if vs is None or len(vs) < 2:
            continue
        if verbose:
            print(f"Processing {mr} ...")
        try:
            res = analyse_recday(data_dic, mr, vs, config, verbose=verbose)
        except Exception as e:
            print(f"  {mr} failed: {e}")
            continue
        if res is None:
            continue
        results[mr] = res
        single_angles.append(res['single_neuron_angles'])
        if 'single_neuron_angles_X' in res:
            single_angles_X.append(res['single_neuron_angles_X'])
        if 'relative_pairs' in res:
            rel_pairs.append(res['relative_pairs'])
            coherent_props.append(res['coherent_prop'])
            coh_by_tuning.append(res['coherence_by_tuning_dist'])
            coh_by_tuning_counts.append(res['coherence_by_tuning_counts'])
            n_comp_seen.append(res['n_comparisons'])
            single_comp_props.append(res.get('single_comp_coherent_prop', np.nan))
            coherent_props_null.append(res.get('coherent_prop_null_mean', np.nan))
            coh_by_tuning_null.append(res.get('coherence_by_tuning_null',
                                              np.full(len(res['coherence_by_tuning_dist']), np.nan)))
        if 'silhouette_real' in res:
            sil_real_all.append(res['silhouette_real'])
            sil_null_all.append(res['silhouette_null_mean'])
        if 'relative_pairs_X' in res:
            rel_pairs_X.append(res['relative_pairs_X'])

    if not single_angles:
        print("No usable recdays.")
        return results, {}

    # pool
    single_flat = np.concatenate([a.ravel() for a in single_angles])
    rel_flat = np.concatenate([a.ravel() for a in rel_pairs]) if rel_pairs else np.array([])
    summary = {}

    # --- single-neuron remapping ---
    hist, prop, z, p = _generalising(single_flat, config)
    summary['remapping_generalising_prop'] = prop
    summary['remapping_p_vs_chance'] = p
    if verbose:
        print(f"\nSINGLE-NEURON remapping: generalising prop = {prop:.3f} "
              f"(chance {1 / config.num_task_states:.3f}, p={p:.2e}, n={int(np.sum(hist))})")
    if save_dir is not None:
        _polar_hist(hist, f'Single-neuron rotation\n(gen={prop:.2f}, p={p:.1e})',
                    os.path.join(save_dir, 'single_neuron_rotation_polar.svg'))
        plt.close('all')

    # --- coherent rotation ---
    if rel_flat.size:
        n_comp = int(np.median(n_comp_seen)) if n_comp_seen else (config.max_comparisons or 2)
        dual_chance = (1 / config.num_task_states) ** n_comp     # 1/16 for 2 comparisons
        single_chance = 1 / config.num_task_states               # 1/4

        cprops = np.array(coherent_props, dtype=float)           # per-recday DUAL proportion
        cprops_null = np.array(coherent_props_null, dtype=float)  # per-recday shuffle-null mean
        scprops = np.array(single_comp_props, dtype=float)       # per-recday single-comparison

        def _ttest_greater(vals, popmean):
            v = vals[~np.isnan(vals)]
            return float(stats.ttest_1samp(v, popmean, alternative='greater')[1]) if len(v) >= 2 else np.nan

        # PRIMARY (conservative): recording day = independent unit
        dual_p_perrecday = _ttest_greater(cprops, dual_chance)         # vs 1/16, matches sploratory
        single_p_perrecday = _ttest_greater(scprops, single_chance)    # vs 1/4
        okn = ~np.isnan(cprops) & ~np.isnan(cprops_null)
        if np.sum(okn) >= 2:
            try:
                dual_p_vs_shuffle = float(stats.wilcoxon(cprops[okn], cprops_null[okn],
                                                         alternative='greater')[1])
            except Exception:
                dual_p_vs_shuffle = float(stats.ttest_rel(cprops[okn], cprops_null[okn],
                                                          alternative='greater')[1])
        else:
            dual_p_vs_shuffle = np.nan

        # SECONDARY (pooled, anti-conservative — pseudoreplication across pairs)
        chist, cprop_pooled, _, cp_pooled_single = _generalising(rel_flat, config)  # single-comp, vs 1/4
        tot = np.nansum(np.stack(coh_by_tuning_counts), axis=0).sum(axis=0)          # [coherent, total] dual
        cp_pooled_dual = (two_proportions_test(tot[0], tot[1], tot[1] * dual_chance, tot[1])[1]
                          if tot[1] > 0 else np.nan)

        summary.update({
            'coherence_dual_chance': dual_chance,
            'coherence_dual_mean_recday_prop': float(np.nanmean(cprops)),
            'coherence_dual_shuffle_null_mean': float(np.nanmean(cprops_null)),
            'coherence_dual_p_perrecday_ttest': dual_p_perrecday,   # PRIMARY
            'coherence_dual_p_vs_shuffle': dual_p_vs_shuffle,       # PRIMARY
            'coherence_dual_p_pooled': cp_pooled_dual,             # anti-conservative
            'coherence_single_mean_recday_prop': float(np.nanmean(scprops)),
            'coherence_single_p_perrecday_ttest': single_p_perrecday,
            'coherence_single_pooled_prop': cprop_pooled,
            'coherence_single_p_pooled': cp_pooled_single,         # anti-conservative
        })
        if verbose:
            print(f"COHERENT rotation (DUAL, vs chance {dual_chance:.4f}): "
                  f"mean per-recday prop = {np.nanmean(cprops):.3f}, shuffle-null = "
                  f"{np.nanmean(cprops_null):.3f}\n"
                  f"  PRIMARY per-recday t-test p={dual_p_perrecday:.3f}; "
                  f"real-vs-shuffle p={dual_p_vs_shuffle:.3f}  "
                  f"[pooled-pairs p={cp_pooled_dual:.1e} = anti-conservative]")
        if save_dir is not None:
            _polar_hist(chist, f'Pairwise relative rotation\n(single-comp pooled={cprop_pooled:.2f})',
                        os.path.join(save_dir, 'pairwise_relative_rotation_polar.svg'))
            # --- Fig 3h: coherence by initial (task-1) tuning distance ---
            per_recday = np.vstack(coh_by_tuning)              # (n_recdays, n_bins)
            per_recday_null = np.vstack(coh_by_tuning_null)    # (n_recdays, n_bins) shuffle-null
            cbt = np.nanmean(per_recday, axis=0)
            sem = np.nanstd(per_recday, axis=0) / np.sqrt(np.sum(~np.isnan(per_recday), axis=0))
            null_cbt = np.nanmean(per_recday_null, axis=0)
            labels = ['0-45', '45-90', '90-135', '135-180'][:len(cbt)]
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.bar(np.arange(len(cbt)), cbt, yerr=sem, color='lightgray', edgecolor='black',
                   capsize=3)
            ax.axhline(dual_chance, color='black', ls='--')
            ax.text(len(cbt) - 0.5, dual_chance, 'analytic 1/16', va='bottom', ha='right', fontsize=7)
            # shuffle-null mean per bin (empirical chance) as gray markers
            ax.plot(np.arange(len(cbt)), null_cbt, 'x', color='red', markersize=7,
                    label='shuffle null')
            for k in range(len(cbt)):
                vals = per_recday[:, k]
                pk = _ttest_greater(vals, dual_chance)          # PRIMARY: per-recday t-test
                top = (cbt[k] + sem[k]) if np.isfinite(sem[k]) else cbt[k]
                ax.text(k, top + 0.004, _stars(pk), ha='center', va='bottom', fontsize=10)
            ax.set_xticks(np.arange(len(cbt)))
            ax.set_xticklabels(labels)
            ax.set_xlabel('Tuning difference (degrees)')
            ax.set_ylabel('Proportion coherent pairs')
            ax.set_title(f'Coherence by tuning distance\n(stars: per-recday t-test vs {dual_chance:.3f})')
            ax.legend(fontsize=7)
            fig.savefig(os.path.join(save_dir, 'coherence_by_tuning_distance.svg'),
                        bbox_inches='tight')
            plt.close('all')

    # --- X vs X' controls ---
    if single_angles_X:
        sxf = np.concatenate([a.ravel() for a in single_angles_X])
        hX, pX, _, ppX = _generalising(sxf, config)
        summary['remapping_X_generalising_prop'] = pX
        if verbose:
            print(f"X-vs-X' single-neuron generalising prop = {pX:.3f} (p={ppX:.2e})")
        if save_dir is not None:
            _polar_hist(hX, f"Single-neuron X vs X'\n(gen={pX:.2f})",
                        os.path.join(save_dir, 'single_neuron_rotation_XvsX_polar.svg'))
            plt.close('all')
    if rel_pairs_X:
        rxf = np.concatenate([a.ravel() for a in rel_pairs_X])
        hXr, pXr, _, ppXr = _generalising(rxf, config)
        summary['coherence_X_pair_prop'] = pXr
        if verbose:
            print(f"X-vs-X' pairwise coherent prop = {pXr:.3f} (p={ppXr:.2e})")
        if save_dir is not None:
            _polar_hist(hXr, f"Pairwise relative X vs X'\n(coh={pXr:.2f})",
                        os.path.join(save_dir, 'pairwise_relative_rotation_XvsX_polar.svg'))
            plt.close('all')

    # --- clustering: silhouette real vs permuted (Fig 3d) + example clusters (Fig 3e) ---
    if sil_real_all:
        real = np.array(sil_real_all, dtype=float)
        null = np.array(sil_null_all, dtype=float)
        ok = ~np.isnan(real) & ~np.isnan(null)
        summary['silhouette_real_mean'] = float(np.nanmean(real))
        summary['silhouette_null_mean'] = float(np.nanmean(null))
        summary['n_recdays_clustered'] = int(np.sum(ok))
        if np.sum(ok) >= 2:
            try:
                w, pw = stats.wilcoxon(real[ok], null[ok])
            except Exception:
                w, pw = stats.ttest_rel(real[ok], null[ok])
            summary['silhouette_real_vs_null_p'] = float(pw)
        else:
            pw = np.nan
        if verbose:
            print(f"CLUSTERING: silhouette real={np.nanmean(real):.3f} vs "
                  f"permuted={np.nanmean(null):.3f} (n={int(np.sum(ok))} recdays, p={pw:.2e})")
        if save_dir is not None and np.sum(ok) >= 1:
            fig, ax = plt.subplots(figsize=(4, 4))
            for r, nl in zip(real[ok], null[ok]):
                ax.plot([0, 1], [nl, r], color='gray', alpha=0.5, marker='o')
            ax.bar([0, 1], [np.nanmean(null[ok]), np.nanmean(real[ok])], color=['lightgray', 'teal'],
                   alpha=0.5, width=0.6, zorder=0)
            ax.set_xticks([0, 1]); ax.set_xticklabels(['permuted', 'real'])
            ax.set_ylabel('silhouette score')
            ax.set_title(f'Module clustering (Fig 3d)\nreal vs permuted (p={pw:.1e})')
            fig.savefig(os.path.join(save_dir, 'silhouette_clustering.svg'), bbox_inches='tight')
            plt.close('all')
            # Fig 3e: example recday (most clustered neurons) — ref-task tunings sorted by cluster
            ex = max((mr for mr in results if results[mr].get('cluster_labels') is not None
                      and np.isfinite(results[mr].get('silhouette_real', np.nan))),
                     key=lambda m: len(results[m]['cluster_labels']), default=None)
            if ex is not None:
                lab = results[ex]['cluster_labels']
                tun = results[ex]['cluster_ref_tunings']
                order = np.argsort(lab)
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(tun[order], aspect='auto', cmap='viridis',
                               interpolation='nearest')
                # cluster boundaries
                bnds = np.where(np.diff(np.sort(lab)) != 0)[0] + 0.5
                for b in bnds:
                    ax.axhline(b, color='white', lw=1)
                ax.set_xlabel('task-space bin (90/state x 4 states)')
                ax.set_ylabel('neuron (sorted by cluster)')
                ax.set_title(f'Example clusters: {ex}\n({len(set(lab))} clusters, '
                             f'sil={results[ex]["silhouette_real"]:.2f})')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.savefig(os.path.join(save_dir, 'cluster_example.svg'), bbox_inches='tight')
                plt.close('all')

    summary['n_recdays'] = len(results)
    summary['n_recdays_coherence'] = len(coherent_props)
    return results, summary
