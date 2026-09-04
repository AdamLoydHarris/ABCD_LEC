"""
Posani-style selectivity structure and population geometry for the ABCD maze.

Ports the battery from Posani, Wang, Muscinelli, Paninski & Fusi (Nature 2026),
"Rarely categorical, highly separable representations along the cortical
hierarchy", to LEC / PFC. See SELECTIVITY_GEOMETRY.md for the method write-up;
section references below (§1, §2, ...) point at that document.

Four things are NOT direct ports, and every one of them is a place a naive
translation manufactures a result:

  1. There is no alpha. Posani's selectivity is a sum of time-varying regression
     coefficients over TRIAL TIME. glm_analysis_v2 regresses over pooled samples
     with one-hot decile blocks, so there is no time axis to sum. We collapse
     each coefficient block to one signed scalar (§2, `build_alpha_matrix`).

  2. Betas here carry firing rate. run_glm_analysis fits raw Neuron_raw counts;
     Posani z-scores y per neuron per time step. Without dividing alpha by the
     neuron's own SD, the alpha cloud clusters by firing rate rather than by
     tuning -- a guaranteed false-categorical result.

  3. Neuron inclusion inverts. His dR2 >= 0.015 is measured against a PSTH null,
     i.e. keep cells that vary with the variables BEYOND the trial-averaged time
     course. Here the phase-averaged loop response IS the signal. We use a
     full-model R2 floor plus his firing-rate band, and audit both (§6).

  4. Two regions, not 43. Every "vs. position in the hierarchy" panel of the
     paper collapses to a LEC-vs-PFC contrast. Do not report a gradient.

THE GATE (§5): `run_synthetic_controls()` must pass before any number computed
here on real data means anything. The critical case is 'uneven' -- an elongated
but UNCLUSTERED cloud must read as non-categorical. If it does not, the
covariance matching in the null is broken and the whole module is measuring
anisotropy and calling it clustering. This is the same miscalibration as an
isotropic / shuffle null in the persistent-homology pipeline.

Two correctness tests come free from the paper's own maths and are asserted in
the gate: the participation ratio of the conditions space equals that of the
neural space (row rank = column rank), and the PR of k Gaussian clusters matches
PR = M*k*delta / (1 + M + k*delta).
"""

import warnings
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.svm import LinearSVC


# ============================================================================
# Config
# ============================================================================

@dataclass
class SelectivityConfig:
    """Defaults are the ones argued for in SELECTIVITY_GEOMETRY.md; change them
    knowingly, and re-run `run_synthetic_controls()` if you touch the clustering
    or null knobs."""

    # --- alpha construction (§2) ---
    alpha_mode: str = 'signed_rms'      # 'signed_rms' | 'abs_rms' | 'cpd'
    regressors: tuple = ()              # must match the fitted GLM section
    gp_n_bins: int = 10
    parameterization: str = 'all_bins'
    normalize_by_neuron_sd: bool = True

    # --- neuron inclusion (§2, §6) ---
    min_r2_full: float = 0.02
    min_rate_hz: float = 0.5
    max_rate_hz: float = 50.0

    # --- clustering (§3) ---
    k_min: int = 3
    k_max: int = 20
    # ONE knob, used for both the data and the null on purpose. Giving the data
    # more restarts than the null biases the null's silhouette down and the
    # z-score up. Do not split this into two.
    n_kmeans_init: int = 10
    n_null: int = 100
    min_neurons: int = 50
    purity_max: float = 0.90
    purity_level: str = 'recday'        # 'recday' | 'mouse'
    null_kind: str = 'gaussian_cov'     # 'gaussian_cov' | 'gaussian_iso' | 'shuffle_cols'
    # Silhouette z above which a region is called categorical. Default is
    # two-sided p < 0.05 Bonferroni-corrected over the 2 regions we test
    # (p < 0.025 -> z = 2.24). Raise it if you test more regions. Kept here
    # rather than hardcoded so the headline and the §6 audits cannot drift
    # apart and disagree about the same z.
    z_significant: float = 2.24

    # --- geometry / decoding (§4) ---
    n_sub_neurons: int = 60             # N-matching floor for cross-region claims
    n_sub_reps: int = 100
    # Exclude recdays with fewer than n_sub_neurons cells rather than running
    # them at whatever they have. Off, a "matched" batch silently mixes recdays
    # at different N -- and because M_IC and PR both grow with N, that is exactly
    # the confound the matching exists to remove. Measured on PFC at N=40: 10 of
    # 23 recdays ran at 12-39 neurons.
    require_full_n: bool = True
    mic_threshold: float = 0.666
    n_dichotomies: int = 200
    min_trials_per_condition: int = 5
    n_folds: int = 5
    svc_C: float = 1.0
    svc_max_iter: int = 5000

    random_state: int = 0


# Palette (gridmaze-colors; matches ccgp_state_pairs.py / time_vs_progress_dissociation.py).
REGION_COLORS = {'LEC': '#6667AB', 'PFC': '#FFA500'}
TRUE_COLOR = '#C03030'
NULL_GREY = '#555555'
NEUTRAL = '#2C2C2A'


# ============================================================================
# §1  Participation ratio
# ============================================================================

def _pr_from_eigs(lam, tol=1e-12):
    """PR = (sum lam)^2 / sum lam^2 over the non-negligible eigenvalues."""
    lam = np.asarray(lam, dtype=float)
    lam = lam[lam > tol * max(lam.max(initial=0.0), 1e-300)]
    if lam.size == 0:
        return np.nan
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def participation_ratio(X, center=True):
    """Effective dimensionality of a point cloud.

    Parameters
    ----------
    X : (n_points, n_dims)
    center : bool
        Subtract the mean over points first. Posani's derivation of the
        clustered-PR formula (his eq. 16) assumes zero mean and does NOT
        re-centre, so the analytic check in the gate passes center=False.

    PR is invariant to a global rescaling of the eigenvalues, which is why the
    row-space and column-space PRs agree for a doubly-centred matrix even though
    their covariances differ by a factor of (n-1)/(d-1).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'X must be 2-D, got shape {X.shape}')
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    # Eigenvalues of XX^T and X^TX agree on their non-zero part, so take the
    # cheaper Gram matrix.
    G = X @ X.T if X.shape[0] <= X.shape[1] else X.T @ X
    return _pr_from_eigs(np.linalg.eigvalsh(G))


def _double_center(X):
    """Remove row means, column means and add back the grand mean."""
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0, keepdims=True) - X.mean(axis=1, keepdims=True)
            + X.mean())


def check_pr_identity(X, rtol=1e-6):
    """§5 correctness test: PR of the conditions space == PR of the neural space.

    Posani's argument (Methods, "The conditions space and the neural space have
    the same dimensionality"): for X = USV^T the spectra of X^T X and X X^T
    coincide, so the PR computed over rows equals the PR computed over columns.
    Doubly centring makes both covariances well defined; PR's scale invariance
    handles the differing normalisation.

    Returns (pr_rows, pr_cols, ok).
    """
    Xc = _double_center(X)
    pr_rows = _pr_from_eigs(np.linalg.eigvalsh(np.cov(Xc, rowvar=True)))
    pr_cols = _pr_from_eigs(np.linalg.eigvalsh(np.cov(Xc, rowvar=False)))
    ok = bool(np.isclose(pr_rows, pr_cols, rtol=rtol))
    return pr_rows, pr_cols, ok


def pr_from_covariance_trace(X):
    """Posani eq. 16 evaluated directly: PR = Tr(C)^2 / Tr(C^2).

    `X` is (n_features, n_observations) -- his orientation, neurons x conditions
    -- and C = (1/M) X X^T is the uncentred neuron-by-neuron sample covariance.

    This is the definition `participation_ratio` computes via the cheaper Gram
    matrix; the gate asserts the two agree, which is an exact test of that
    shortcut rather than a test of any approximation.
    """
    X = np.asarray(X, dtype=float)
    M = X.shape[1]
    C = (X @ X.T) / M
    return float(np.trace(C) ** 2 / np.trace(C @ C))


def pr_gaussian_clusters_theory(M, k, sigma):
    """Posani eq. 25 / ED Fig. 9b: PR of k Gaussian clusters over M conditions.

        delta = (1 + sigma^2)^2          (intra-cluster diversity)
        PR    = M * k * delta / (1 + M + k * delta)

    Limits: PR -> k as M -> inf, PR -> M as k -> inf.

    NOTE -- this is a large-N *and* effectively large-M approximation, not an
    identity. Eq. 24 drops <C_ii^2> against (N-1)<C_ij^2>, but the surviving
    terms carry 1/M and 1/(kM) factors, so the error grows as M shrinks. Measured
    against the exact eq. 16 on simulated data (N = 4000, k = 4):

        M = 2000, sigma = 0   ->  0.1 %
        M = 16,   k = 2000    ->  0.1 %
        M = 64,   sigma = 1   ->  1.8 %
        M = 16,   sigma = 1   ->   18 %
        M = 16,   sigma = 0.5 ->   17 %

    So do not use it as ground truth at our condition counts (M = 16-48). The
    gate checks it only where it is valid, and checks the exact identities
    everywhere else.
    """
    delta = (1.0 + sigma ** 2) ** 2
    return float(M * k * delta / (1.0 + M + k * delta))


# ============================================================================
# §2  The selectivity vector alpha
# ============================================================================

def _unpack_scale(entry):
    """neuron_scales values are {'sd','mean'}; tolerate a bare float for
    artifacts written before the dict form."""
    if isinstance(entry, dict):
        return float(entry.get('sd', np.nan)), float(entry.get('mean', np.nan))
    return float(entry), np.nan


def build_alpha_matrix(glm_results, neuron_scales=None, cpd_results=None,
                       config=None, bin_size_ms=25):
    """Collapse each fitted coefficient block to one signed scalar per variable.

    This is the stand-in for Posani's alpha_n^v = sum_t beta_n^v(t). His sum runs
    over trial time, which this GLM does not have: it regresses pooled samples on
    one-hot decile blocks. What is available per regressor is a block of bin
    coefficients, and the two facts that matter are its magnitude and its
    direction:

        alpha[n, v] = sign_v * ||beta_n[cols_v]||_2 / sqrt(|cols_v|) / sd_n

      * RMS, not the raw L2 norm. Blocks have different widths (place 21,
        goal_progress 10, task_state 4); an unnormalised norm would make place
        the largest axis for every neuron purely by column count.
      * sign_v from glm._beta_direction -- slope across bins for the ordered
        regressors, sign of the mean for the unordered ones.
      * /sd_n because this GLM fits raw spike counts. Posani z-scores y per
        neuron; without the equivalent here, alpha magnitude tracks firing rate
        and the cloud clusters by rate rather than by tuning.

    Parameters
    ----------
    glm_results : {recday: {neuron: params}}   from a fitted section
    neuron_scales : {recday: {neuron: {'sd','mean'}}}, optional
        Required when config.normalize_by_neuron_sd (the default). Produced by
        run_glm_analysis(..., return_scales=True); backfill onto an existing
        section cheaply with scales_only=True.
    cpd_results : {recday: {neuron: cpd_dict}}, optional
        Supplies '__r2_full__' for the inclusion filter and the §6 audit.

    Returns
    -------
    A : (n_neurons, n_vars) float
    meta : DataFrame with recday, mouse, neuron, r2_full, sd, mean_rate_hz, keep
        Every fitted neuron is returned; `keep` marks those passing inclusion, so
        the audit can sweep the threshold without refitting. Use
        `A[meta['keep'].values]` for the default selection.
    """
    import pandas as pd
    import glm_analysis_v2 as glm

    config = config or SelectivityConfig()
    if not config.regressors:
        raise ValueError('config.regressors must list the regressors of the '
                         'fitted section, in any order (canonical order is '
                         'applied internally).')

    groups, names = glm._resolve_regressor_groups(
        list(config.regressors), gp_n_bins=config.gp_n_bins,
        parameterization=config.parameterization)
    n_cols_expected = 1 + max(max(v) for v in groups.values())

    rows, alphas = [], []
    for recday in sorted(glm_results):
        for neuron in sorted(glm_results[recday]):
            params = np.asarray(glm_results[recday][neuron], dtype=float)
            if params.size != n_cols_expected:
                raise ValueError(
                    f'{recday} neuron {neuron}: params has {params.size} columns '
                    f'but config.regressors implies {n_cols_expected}. The '
                    f'regressor list / gp_n_bins / parameterization must match '
                    f'the ones the section was fitted with.')

            a = np.empty(len(names))
            for i, reg in enumerate(names):
                cols = groups[reg]
                rms = float(np.linalg.norm(params[cols]) / np.sqrt(len(cols)))
                if config.alpha_mode == 'signed_rms':
                    a[i] = glm._beta_direction(params, reg, groups=groups) * rms
                elif config.alpha_mode == 'abs_rms':
                    a[i] = rms
                else:
                    raise ValueError(f'alpha_mode {config.alpha_mode!r} not '
                                     f'supported by build_alpha_matrix '
                                     f"(use 'signed_rms' or 'abs_rms'; 'cpd' "
                                     f'is read by build_alpha_from_cpd)')

            sd, mean = (np.nan, np.nan)
            if neuron_scales is not None:
                sd, mean = _unpack_scale(neuron_scales.get(recday, {}).get(neuron, np.nan))
            if config.normalize_by_neuron_sd:
                if not np.isfinite(sd) or sd <= 0:
                    continue
                a = a / sd

            r2 = np.nan
            if cpd_results is not None:
                r2 = float(cpd_results.get(recday, {}).get(neuron, {})
                           .get('__r2_full__', np.nan))

            alphas.append(a)
            rows.append({'recday': recday, 'mouse': recday[:4], 'neuron': int(neuron),
                         'r2_full': r2, 'sd': sd,
                         'mean_rate_hz': mean * (1000.0 / bin_size_ms)})

    if not alphas:
        raise ValueError('no neurons survived alpha construction (are '
                         'neuron_scales present for this section?)')

    A = np.vstack(alphas)
    meta = pd.DataFrame(rows)

    keep = np.ones(len(meta), dtype=bool)
    if np.isfinite(meta['r2_full']).any():
        keep &= (meta['r2_full'] >= config.min_r2_full).values
    if np.isfinite(meta['mean_rate_hz']).any():
        keep &= ((meta['mean_rate_hz'] >= config.min_rate_hz) &
                 (meta['mean_rate_hz'] <= config.max_rate_hz)).values
    meta['keep'] = keep
    meta.attrs['alpha_names'] = list(names)
    return A, meta


def build_alpha_from_cpd(cpd_results, config=None):
    """Alpha as the cached per-regressor CPD vector -- a cross-check only.

    Zero fitting required, but CPD is a non-negative variance share, so the cloud
    lives in the positive orthant and a multivariate Gaussian null fits it badly.
    A Gaussian null on a one-sided distribution can read as categorical on its
    own. Never use this as the primary selectivity space; use it to ask whether a
    verdict from `build_alpha_matrix` survives a completely different reduction.

    Returns (A, meta) in the same shape as `build_alpha_matrix`.
    """
    import pandas as pd

    config = config or SelectivityConfig()
    names = [r for r in config.regressors]
    rows, alphas = [], []
    for recday in sorted(cpd_results):
        for neuron in sorted(cpd_results[recday]):
            cpd = cpd_results[recday][neuron]
            if any(n not in cpd for n in names):
                continue
            alphas.append(np.array([float(cpd[n]) for n in names]))
            rows.append({'recday': recday, 'mouse': recday[:4], 'neuron': int(neuron),
                         'r2_full': float(cpd.get('__r2_full__', np.nan)),
                         'sd': np.nan, 'mean_rate_hz': np.nan})
    A = np.vstack(alphas)
    meta = pd.DataFrame(rows)
    meta['keep'] = (meta['r2_full'] >= config.min_r2_full).values
    meta.attrs['alpha_names'] = list(names)
    return A, meta


def check_alpha_against_tuning(A, meta, tuned_dict, config):
    """Consistency test against already-validated output.

    `compute_tuning_arrays` derives its +1/-1 from the same `_beta_direction`
    call, so wherever it reports a significant tuning the sign of alpha must
    agree. A mismatch means the regressor list, gp_n_bins or parameterization
    passed here does not describe the fitted section.

    Returns (n_agree, n_compared, fraction).
    """
    import glm_analysis_v2 as glm
    _, names = glm._resolve_regressor_groups(
        list(config.regressors), gp_n_bins=config.gp_n_bins,
        parameterization=config.parameterization)

    n_ok = n_tot = 0
    for recday, sub in meta.groupby('recday'):
        if recday not in tuned_dict:
            continue
        T = tuned_dict[recday]
        for pos, (_, r) in enumerate(sub.iterrows()):
            n = int(r['neuron'])
            if n >= T.shape[0]:
                continue
            row = A[sub.index[pos]]
            for i in range(len(names)):
                if T[n, i] != 0:
                    n_tot += 1
                    n_ok += int(np.sign(row[i]) == np.sign(T[n, i]))
    return n_ok, n_tot, (n_ok / n_tot if n_tot else np.nan)


# ============================================================================
# §3  Clustering: silhouette sweep, purity guard, matched nulls
# ============================================================================

def _kmeans_labels(A, k, seed, n_init):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return km.fit_predict(A)


def silhouette_sweep(A, config, seed=None):
    """k-means over k = k_min..k_max, keep the k maximising the mean silhouette.

    Returns {'k', 'ss', 'labels', 'sil'} or None if A is too small to cluster.
    `sil` are the per-neuron silhouette values for the winning k.

    Threads are pinned to 1 for the duration. These problems are tiny (a few
    hundred points in <=14 dimensions) and sklearn's OpenMP parallelism costs
    far more in contention than it saves: measured 27 s -> 0.74 s for one sweep
    on an 8-core node. The nulls are parallelised across processes instead.
    """
    from threadpoolctl import threadpool_limits

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    seed = config.random_state if seed is None else seed
    k_hi = min(config.k_max, n - 1)
    if n < 3 or k_hi < config.k_min:
        return None

    best = None
    with threadpool_limits(limits=1):
        for k in range(config.k_min, k_hi + 1):
            labels = _kmeans_labels(A, k, seed, config.n_kmeans_init)
            if np.unique(labels).size < 2:
                continue
            sil = silhouette_samples(A, labels)
            ss = float(sil.mean())
            if best is None or ss > best['ss']:
                best = {'k': int(k), 'ss': ss, 'labels': labels, 'sil': sil}
    return best


def _purity_offender(labels, sil, group_ids, purity_max):
    """Posani step 4: find a cluster whose silhouette mass is >purity_max from a
    single group. Returns (cluster, group) or None.

    He phrases it as the "total silhouette score summed over all neurons" being
    dominated by one session. Negative silhouettes make a share ill-defined, so
    contributions are clipped at zero; a cluster with no positive mass is
    skipped rather than flagged.
    """
    group_ids = np.asarray(group_ids)
    for c in np.unique(labels):
        m = labels == c
        w = np.clip(sil[m], 0.0, None)
        total = w.sum()
        if total <= 0:
            continue
        gs = group_ids[m]
        for g in np.unique(gs):
            if w[gs == g].sum() / total > purity_max:
                return int(c), g
    return None


def _sample_null(A, kind, rng):
    """Null draws matched to A, sampled with the SAME n as the data (§3).

    'gaussian_cov'  -- mean + full covariance matched. THE correct null: it is
                       unimodal by construction, so anything it cannot reproduce
                       is genuine multimodality rather than anisotropy.
    'gaussian_iso'  -- isotropic with matched TOTAL variance. Deliberately wrong;
                       used by `audit_null_kind` to show the verdict flip.
    'shuffle_cols'  -- independently permute each column. Destroys the covariance
                       between variables. Also deliberately wrong.
    """
    n, d = A.shape
    if kind == 'gaussian_cov':
        cov = np.cov(A, rowvar=False)
        cov = np.atleast_2d(cov)
        # method='svd' tolerates the near-singular covariances that show up when
        # two regressors are strongly collinear.
        return rng.multivariate_normal(A.mean(axis=0), cov, size=n, method='svd')
    if kind == 'gaussian_iso':
        sd = np.sqrt(A.var(axis=0).mean())
        return A.mean(axis=0)[None, :] + rng.normal(0.0, sd, size=(n, d))
    if kind == 'shuffle_cols':
        B = A.copy()
        for j in range(d):
            rng.shuffle(B[:, j])
        return B
    raise ValueError(f'unknown null_kind {kind!r}')


def cluster_quality(A, group_ids=None, config=None, null_kind=None, seed=None,
                    n_jobs=-1):
    """Is this cloud of selectivity profiles categorical? (Posani Fig. 3a.)

    Pipeline, faithful to his Methods steps 1-6:
      1. require >= config.min_neurons
      2. k-means, k = k_min..k_max, n_init restarts
      3. keep the k maximising mean silhouette  -> ss_data
      4. drop any cluster whose silhouette mass is >purity_max from one group,
         then restart from 1 (guards against a cluster that is really one recday)
      5. draw n_null matched nulls and push each through the IDENTICAL sweep
      6. z = (ss_data - mean(ss_null)) / std(ss_null)

    `group_ids` is the reproducibility unit -- recday here, session in the paper.
    Pass None to skip step 4.

    Returns a dict with 'z', 'ss_data', 'ss_null', 'k', 'labels', 'sil',
    'n_used', 'n_dropped', 'keep' (index into the input rows), 'null_kind'.
    """
    config = config or SelectivityConfig()
    null_kind = null_kind or config.null_kind
    seed = config.random_state if seed is None else seed
    rng = np.random.default_rng(seed)

    A = np.asarray(A, dtype=float)
    keep = np.arange(A.shape[0])
    group_ids = None if group_ids is None else np.asarray(group_ids)

    # --- steps 1-4: sweep, then drop group-dominated clusters and repeat -----
    best = None
    for _ in range(20):                       # bounded; each pass drops >=1 cluster
        if keep.size < config.min_neurons:
            return {'z': np.nan, 'ss_data': np.nan, 'ss_null': np.array([]),
                    'k': np.nan, 'labels': None, 'sil': None,
                    'n_used': int(keep.size), 'n_dropped': int(A.shape[0] - keep.size),
                    'keep': keep, 'null_kind': null_kind,
                    'reason': f'fewer than min_neurons={config.min_neurons}'}
        best = silhouette_sweep(A[keep], config, seed=seed)
        if best is None:
            return {'z': np.nan, 'ss_data': np.nan, 'ss_null': np.array([]),
                    'k': np.nan, 'labels': None, 'sil': None,
                    'n_used': int(keep.size), 'n_dropped': int(A.shape[0] - keep.size),
                    'keep': keep, 'null_kind': null_kind, 'reason': 'sweep failed'}
        if group_ids is None:
            break
        off = _purity_offender(best['labels'], best['sil'], group_ids[keep],
                               config.purity_max)
        if off is None:
            break
        c, g = off
        drop = (best['labels'] == c) & (group_ids[keep] == g)
        keep = keep[~drop]

    # --- steps 5-6: matched null through the identical sweep ----------------
    # Same config object, so the null gets exactly the same k range and the same
    # number of k-means restarts as the data. Giving the data more restarts
    # would bias the null's silhouette down and the z-score up.
    A_used = A[keep]
    draws = [(_sample_null(A_used, null_kind, rng), int(rng.integers(1 << 31)))
             for _ in range(config.n_null)]

    def _one(args):
        B, s = args
        nb = silhouette_sweep(B, config, seed=s)
        return np.nan if nb is None else nb['ss']

    if n_jobs == 1:
        ss_null = [_one(d) for d in draws]
    else:
        from joblib import Parallel, delayed
        ss_null = Parallel(n_jobs=n_jobs)(delayed(_one)(d) for d in draws)
    ss_null = np.asarray(ss_null, dtype=float)
    ss_null = ss_null[np.isfinite(ss_null)]

    sd = ss_null.std(ddof=1) if ss_null.size > 1 else np.nan
    z = (best['ss'] - ss_null.mean()) / sd if ss_null.size > 1 and sd > 0 else np.nan

    return {'z': float(z), 'ss_data': best['ss'], 'ss_null': ss_null,
            'k': best['k'], 'labels': best['labels'], 'sil': best['sil'],
            'n_used': int(keep.size), 'n_dropped': int(A.shape[0] - keep.size),
            'keep': keep, 'null_kind': null_kind, 'reason': ''}


def alpha_diversity(A, config=None, seed=None):
    """PR of the (n_neurons x n_vars) alpha matrix (Posani Fig. 4).

    Low  = structured selectivity, either uneven (one variable dominates) or
           categorical (neurons cluster).
    High = isotropic, unstructured; the ceiling is n_vars.

    Neuron counts are soft-equalised by subsampling to config.n_sub_neurons and
    averaging the PR over config.n_sub_reps random subsets, as he does at
    N0 = 120. Populations smaller than the floor are used whole.
    """
    config = config or SelectivityConfig()
    rng = np.random.default_rng(config.random_state if seed is None else seed)
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n <= config.n_sub_neurons:
        return participation_ratio(A)
    vals = [participation_ratio(A[rng.choice(n, config.n_sub_neurons, replace=False)])
            for _ in range(config.n_sub_reps)]
    return float(np.mean(vals))


# ============================================================================
# §6  Null-calibration and inclusion-threshold audit
# ============================================================================

def audit_null_kind(A, group_ids=None, config=None,
                    kinds=('gaussian_cov', 'gaussian_iso', 'shuffle_cols'),
                    n_jobs=-1):
    """Same data, three nulls. Does the categorical verdict depend on the null?

    'gaussian_cov' is the correct one (Posani's): unimodal AND covariance
    matched, so a high silhouette can only come from multimodality.
    'gaussian_iso' matches total variance but not its shape, so any elongated
    cloud beats it. 'shuffle_cols' keeps the marginals but destroys the
    correlations between variables, so it fails the same way whenever the
    regressors are correlated -- which place and goal-progress certainly are.

    The synthetic 'uneven' case shows this cleanly in `run_synthetic_controls`;
    this function shows what it does to the real answer.

    Returns a DataFrame with one row per null kind.
    """
    import pandas as pd
    config = config or SelectivityConfig()
    rows = []
    for kind in kinds:
        r = cluster_quality(A, group_ids, config, null_kind=kind, n_jobs=n_jobs)
        rows.append({'null_kind': kind, 'z': r['z'], 'ss_data': r['ss_data'],
                     'ss_null_mean': float(np.mean(r['ss_null'])) if len(r['ss_null']) else np.nan,
                     'k': r['k'], 'n_used': r['n_used'],
                     'categorical': bool(np.isfinite(r['z']) and r['z'] > config.z_significant)})
    return pd.DataFrame(rows)


def audit_inclusion_threshold(A, meta, config=None,
                              grid=(0.0, 0.005, 0.01, 0.02, 0.05, 0.10),
                              apply_rate_band=True, n_jobs=-1):
    """Sweep the full-model R2 floor and watch the verdict move.

    Posani's warning, in his Methods: keep every neuron and the cloud acquires a
    spike of near-zero-selectivity 'junk' cells that no unimodal null can
    reproduce, so the region reads categorical for a reason that has nothing to
    do with functional structure. Cut too hard and the centre of the
    distribution is depleted, which does the same thing from the other side. The
    'junk' synthetic in `run_synthetic_controls` reproduces the first failure
    (z = 19.9 with 50% zero-selectivity cells).

    Returns a DataFrame with one row per threshold.
    """
    import pandas as pd
    config = config or SelectivityConfig()
    rate = meta['mean_rate_hz'].values
    band = np.ones(len(meta), dtype=bool)
    if apply_rate_band and np.isfinite(rate).any():
        band = (rate >= config.min_rate_hz) & (rate <= config.max_rate_hz)

    rows = []
    for thr in grid:
        keep = band & (meta['r2_full'].values >= thr)
        if keep.sum() < config.min_neurons:
            rows.append({'min_r2_full': thr, 'n': int(keep.sum()), 'z': np.nan,
                         'k': np.nan, 'alpha_diversity': np.nan, 'categorical': False})
            continue
        gid = meta['recday'].values[keep] if config.purity_level == 'recday' \
            else meta['mouse'].values[keep]
        r = cluster_quality(A[keep], gid, config, n_jobs=n_jobs)
        rows.append({'min_r2_full': thr, 'n': int(keep.sum()), 'z': r['z'],
                     'k': r['k'], 'alpha_diversity': alpha_diversity(A[keep], config),
                     'categorical': bool(np.isfinite(r['z']) and r['z'] > config.z_significant)})
    return pd.DataFrame(rows)


def variable_contribution(A, meta, config, group_ids=None, n_jobs=-1):
    """Posani ED Fig. 4b: drop one variable at a time, see what the verdict does.

    Read `delta_z`, not `delta_ss`. The raw silhouette score depends on the
    dimensionality of the space it is computed in, and dropping a variable
    changes that dimensionality -- which is the whole reason the matched null
    exists. Comparing `ss` at 9 variables against `ss` at 8 mixes the effect of
    the variable with the effect of the dimension count; only the z-score, which
    is referenced to a null of the same dimensionality, is comparable across
    rows. `delta_ss` is kept because it is the quantity the paper plots.

    Positive `delta_z` means the verdict gets STRONGER without that variable,
    i.e. the variable was masking clustering rather than creating it.
    """
    import pandas as pd

    names = list(meta.attrs.get('alpha_names', []))
    base = cluster_quality(A, group_ids, config, n_jobs=n_jobs)
    rows = [{'dropped': '(none)', 'ss': base['ss_data'], 'delta_ss': 0.0,
             'z': base['z'], 'delta_z': 0.0, 'k': base['k']}]
    for i, nm in enumerate(names):
        sub = np.delete(A, i, axis=1)
        r = cluster_quality(sub, group_ids, config, n_jobs=n_jobs)
        rows.append({'dropped': nm, 'ss': r['ss_data'],
                     'delta_ss': base['ss_data'] - r['ss_data'],
                     'z': r['z'], 'delta_z': r['z'] - base['z'], 'k': r['k']})
    return pd.DataFrame(rows)


# ============================================================================
# §4  Conditions space: cross-validated decoding, M_IC, separability
# ============================================================================

def _interval_folds(group_ids, n_folds, rng):
    """Assign whole groups (trials/legs) to folds.

    Same construction as glm_cv_cpd._interval_folds: samples inside a group are
    strongly autocorrelated, so a plain KFold leaks and inflates every accuracy
    downstream.
    """
    groups = np.unique(group_ids)
    groups = groups.copy()
    rng.shuffle(groups)
    fold_of = {g: i % n_folds for i, g in enumerate(groups)}
    return np.array([fold_of[g] for g in group_ids])


def _balanced_subset(y, rng):
    """Indices giving equal counts per class."""
    classes, counts = np.unique(y, return_counts=True)
    n = counts.min()
    if n == 0:
        return np.array([], dtype=int)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=False)
                          for c in classes])
    idx.sort()
    return idx


def _make_svc(config):
    try:
        return LinearSVC(C=config.svc_C, max_iter=config.svc_max_iter, dual='auto')
    except TypeError:                          # older sklearn
        return LinearSVC(C=config.svc_C, max_iter=config.svc_max_iter)


def cv_decode(X, y, group_ids, config, rng):
    """Group-aware, class-balanced cross-validated linear decoding accuracy.

    Balancing happens inside each fold so that chance is 0.5 regardless of how
    unevenly the conditions are populated.

    Features are standardised using TRAIN-fold statistics only (no leakage).
    This is not cosmetic: liblinear silently fails to converge on features with
    large magnitudes, and because convergence warnings are suppressed here the
    result is a plausible-looking accuracy near chance on data that is in fact
    trivially separable. `build_task_state_matrices` already z-scores per task,
    but this function must be safe for any caller.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    folds = _interval_folds(np.asarray(group_ids), config.n_folds, rng)

    accs = []
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        if not te.any() or not tr.any():
            continue
        if np.unique(y[tr]).size < 2 or np.unique(y[te]).size < 2:
            continue
        itr = _balanced_subset(y[tr], rng)
        ite = _balanced_subset(y[te], rng)
        if itr.size < 2 or ite.size < 2:
            continue
        Xtr, ytr = X[tr][itr], y[tr][itr]
        Xte, yte = X[te][ite], y[te][ite]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        clf = _make_svc(config)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf.fit(Xtr, ytr)
        accs.append(float(clf.score(Xte, yte)))
    return float(np.mean(accs)) if accs else np.nan


def _bron_kerbosch(adj):
    """All maximal cliques of an undirected graph, with pivoting.

    `adj` maps node -> set of neighbours. networkx is not in the maze_ephys
    environment and a 25-line routine does not justify adding it.
    """
    cliques = []

    def expand(R, P, X):
        if not P and not X:
            cliques.append(set(R))
            return
        pivot = max(P | X, key=lambda u: len(adj[u] & P))
        for v in list(P - adj[pivot]):
            expand(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    expand(set(), set(adj), set())
    return cliques


def condition_decoding_matrix(F, cond, group_ids, config, rng, pair_cache=None):
    """M x M one-vs-one cross-validated decoding matrix over the labels in `cond`."""
    labels = np.unique(cond)
    M = labels.size
    C = np.full((M, M), np.nan)
    for a in range(M):
        for b in range(a + 1, M):
            key = (labels[a], labels[b])
            if pair_cache is not None and key in pair_cache:
                acc = pair_cache[key]
            else:
                m = (cond == labels[a]) | (cond == labels[b])
                acc = cv_decode(F[m], cond[m], group_ids[m], config, rng)
                if pair_cache is not None:
                    pair_cache[key] = acc
            C[a, b] = C[b, a] = acc
    np.fill_diagonal(C, 1.0)
    return C, labels


def independent_conditions(F, cond, group_ids, config=None, seed=None):
    """Posani ED Fig. 3: merge conditions the population cannot tell apart.

    Iterate: build the one-vs-one decoding matrix, threshold at
    config.mic_threshold, find all cliques of mutually NON-decodable conditions,
    merge the largest into a single label, repeat until the dependency matrix is
    diagonal. M_IC is the size of the surviving label set.

    Returns {'m_ic', 'labels', 'merge_history', 'ccdm', 'cond_merged'} where
    `ccdm` is the FIRST (unmerged) decoding matrix, which is the one worth
    plotting.
    """
    config = config or SelectivityConfig()
    rng = np.random.default_rng(config.random_state if seed is None else seed)

    F = np.asarray(F, dtype=float)
    # Labels are cast to str up front: merging introduces synthetic string
    # labels, and a mixed int/str object array cannot be sorted by np.unique.
    cond = np.asarray(cond).astype(str).astype(object)
    group_ids = np.asarray(group_ids)

    # Pair accuracies are keyed by the merged label pair, so only pairs touching
    # a freshly merged group need recomputing.
    pair_cache = {}
    ccdm0 = None
    history = []

    for _ in range(64):
        C, labels = condition_decoding_matrix(F, cond, group_ids, config, rng,
                                              pair_cache=pair_cache)
        if ccdm0 is None:
            ccdm0 = (C.copy(), labels.copy())
        if labels.size <= 1:
            break

        # Non-decodable edges. NaN (a pair we could not evaluate) is treated as
        # decodable so that missing data never merges conditions.
        D = np.where(np.isnan(C), 1.0, C) < config.mic_threshold
        np.fill_diagonal(D, False)
        if not D.any():
            break

        adj = {i: set(np.flatnonzero(D[i])) for i in range(labels.size)}
        cliques = [c for c in _bron_kerbosch(adj) if len(c) > 1]
        if not cliques:
            break
        biggest = max(cliques, key=len)

        merged_labels = [labels[i] for i in sorted(biggest)]
        new_label = f'm{len(history)}:' + '+'.join(str(l) for l in merged_labels)
        history.append({'merged': merged_labels, 'into': new_label})

        for l in merged_labels:
            cond[cond == l] = new_label
        # Any cached pair involving a merged label is now stale.
        pair_cache = {k: v for k, v in pair_cache.items()
                      if k[0] not in merged_labels and k[1] not in merged_labels}

    labels_final = np.unique(cond)
    return {'m_ic': int(labels_final.size), 'labels': labels_final,
            'merge_history': history, 'ccdm': ccdm0, 'cond_merged': cond}


def _balanced_dichotomies(labels, n_max, rng):
    """Balanced splits of `labels` into two halves.

    Enumerates exhaustively when there are few enough (with M = 4 there are only
    3, so a sampler would return duplicates); otherwise samples without
    replacement. Odd M splits floor/ceil.
    """
    M = len(labels)
    half = M // 2
    total = 0
    if M <= 20:
        from math import comb
        total = comb(M, half) // (2 if M % 2 == 0 else 1)

    if total and total <= n_max:
        seen, out = set(), []
        for c in combinations(range(M), half):
            key = frozenset(c)
            comp = frozenset(set(range(M)) - key)
            if key in seen or comp in seen:
                continue
            seen.add(key)
            out.append(np.array([1 if i in key else 0 for i in range(M)]))
        return out

    out, seen = [], set()
    for _ in range(n_max * 20):
        if len(out) >= n_max:
            break
        c = frozenset(rng.choice(M, half, replace=False).tolist())
        comp = frozenset(set(range(M)) - c)
        if c in seen or comp in seen:
            continue
        seen.add(c)
        out.append(np.array([1 if i in c else 0 for i in range(M)]))
    return out


def separability(F, cond, group_ids, config=None, seed=None, n_dichotomies=None):
    """Fraction of balanced dichotomies a linear readout can separate (Fig. 6).

    A dichotomy counts as separable when its cross-validated accuracy exceeds the
    99th percentile of a null in which condition labels are shuffled across
    population vectors. Average decodability (AD) is the mean accuracy across
    dichotomies -- Bernardi's shattering dimensionality.

    Returns {'sep', 'ad', 'phi', 'phi_null', 'thresh', 'n_dichotomies', 'M'}.
    """
    config = config or SelectivityConfig()
    rng = np.random.default_rng(config.random_state if seed is None else seed)
    n_dich = n_dichotomies or config.n_dichotomies

    F = np.asarray(F, dtype=float)
    cond = np.asarray(cond)
    group_ids = np.asarray(group_ids)

    labels = np.unique(cond)
    M = labels.size
    if M < 4:
        return {'sep': np.nan, 'ad': np.nan, 'phi': np.array([]),
                'phi_null': np.array([]), 'thresh': np.nan,
                'n_dichotomies': 0, 'M': int(M),
                'reason': 'need at least 4 conditions for a balanced dichotomy'}

    dichs = _balanced_dichotomies(labels, n_dich, rng)
    lab_to_i = {l: i for i, l in enumerate(labels)}
    idx = np.array([lab_to_i[c] for c in cond])

    phi = []
    for d in dichs:
        y = d[idx]
        phi.append(cv_decode(F, y, group_ids, config, rng))
    phi = np.asarray(phi, dtype=float)

    # Null: shuffle the condition labels across population vectors, then decode
    # the same kind of random dichotomy. Shuffling within group would leave the
    # trial structure informative, so the permutation is global.
    phi_null = []
    for _ in range(len(dichs)):
        perm = rng.permutation(len(idx))
        y = dichs[rng.integers(len(dichs))][idx[perm]]
        phi_null.append(cv_decode(F, y, group_ids, config, rng))
    phi_null = np.asarray(phi_null, dtype=float)

    ok = np.isfinite(phi)
    okn = np.isfinite(phi_null)
    thresh = np.percentile(phi_null[okn], 99) if okn.sum() else np.nan
    sep = float(np.mean(phi[ok] > thresh)) if ok.sum() and np.isfinite(thresh) else np.nan
    ad = float(np.mean(phi[ok])) if ok.sum() else np.nan

    return {'sep': sep, 'ad': ad, 'phi': phi, 'phi_null': phi_null,
            'thresh': float(thresh), 'n_dichotomies': len(dichs), 'M': int(M),
            'reason': ''}


def representation_dimensionality(F, cond, config=None, seed=None):
    """PR of the condition centroids in neural space (Posani Fig. 5).

    Centroids rather than single trials, so that trial-to-trial noise does not
    dominate the spectrum. Each neuron's centroid vector is z-scored across
    conditions first, and neuron counts are soft-equalised by subsampling.
    """
    config = config or SelectivityConfig()
    rng = np.random.default_rng(config.random_state if seed is None else seed)

    F = np.asarray(F, dtype=float)
    cond = np.asarray(cond)
    labels = np.unique(cond)
    cents = np.stack([F[cond == l].mean(axis=0) for l in labels])   # (M, n_neurons)

    sd = cents.std(axis=0)
    good = sd > 0
    cents = (cents[:, good] - cents[:, good].mean(axis=0)) / sd[good]

    n = cents.shape[1]
    if n <= config.n_sub_neurons:
        return participation_ratio(cents)
    vals = [participation_ratio(cents[:, rng.choice(n, config.n_sub_neurons, replace=False)])
            for _ in range(config.n_sub_reps)]
    return float(np.mean(vals))


# ============================================================================
# §4b  Condition specs -- what counts as an experimental condition here
# ============================================================================

@dataclass
class ConditionSpec:
    """Which factors define a condition.

    Posani crosses 4 binary variables (block, side, contrast, whisking) into
    M = 16. The maze has no equivalent set of trial-constant binaries, and its
    obvious factors are conjoined by design: within a task, state s always ends
    at tower Task[s+1], so state and place are not independent. What breaks that
    conjunction is the multi-task structure -- the same tracked neurons see 6
    reward configurations, so the state-to-tower mapping rotates. That is why
    'task' is a legitimate condition axis here and is the substrate for the
    CCGP bridge (§7).
    """
    name: str
    factors: tuple                  # from ('state', 'progress', 'speed', 'task', 'tower')
    n_progress_bins: int = 1
    n_speed_bins: int = 1


CONDITION_SPECS = {
    # 4 x 2 x 2 = 16. Closest structural analogue of the IBL grid, and the
    # best-populated: every loop contributes samples to several cells.
    'state_prog_speed': ConditionSpec('state_prog_speed', ('state', 'progress', 'speed'),
                                      n_progress_bins=2, n_speed_bins=2),
    # 4 x 6 = 24. Same population vectors CCGP is computed on.
    'state_task':       ConditionSpec('state_task', ('state', 'task')),
    # 4 x 2 x 6 = 48. Richest, but many cells fall under the 5-trial floor.
    'state_prog_task':  ConditionSpec('state_prog_task', ('state', 'progress', 'task'),
                                      n_progress_bins=2),
    # Puts place explicitly on an axis, to test whether it drives any structure.
    'state_tower':      ConditionSpec('state_tower', ('state', 'tower')),
}


def _phase_windows(state, spec, ccgp_config):
    """Phase-bin slices for one state, split into n_progress_bins sub-windows.

    Trims are inherited from CCGPConfig (15 bins at each end by default). Those
    trims are what take the place and tone synthetics to chance in
    CCGP_STATE_PAIRS.md -- both confounds live in the reward windows at the ends
    of every leg -- so they are not negotiable here either.
    """
    nbps = ccgp_config.num_bins_per_state
    lo = state * nbps + ccgp_config.trim_start_bins
    hi = (state + 1) * nbps - ccgp_config.trim_end_bins
    if hi <= lo:
        raise ValueError('trims leave no bins')
    edges = np.linspace(lo, hi, spec.n_progress_bins + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(spec.n_progress_bins)]


def _warped_speed(session_data, ccgp_config):
    """Per-trial 360-bin speed, warped the same way the neural data is.

    Speed is not in Neurons_norm and PFC has no XY_norm, so it is derived in raw
    time with the GLM's own smoother and then pushed through `raw_to_norm`, the
    same warp used to build the neural tensor. Returns None when XY is missing,
    which makes any spec with a 'speed' factor skip that recday rather than
    silently fall back to something else.
    """
    import glm_analysis_v2 as glm
    from remapping_rotation_analysis import raw_to_norm

    xy = session_data.get('XY_raw')
    tt = session_data.get('Trial_times')
    if xy is None or tt is None:
        return None
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 10:
        return None
    speed = glm.smooth_and_calculate_scalar_derivatives(xy)[:, 2]
    return raw_to_norm(speed, tt, ccgp_config, return_mean=False)


def build_condition_table(data_dic, mouse_recday, spec, config=None, ccgp_config=None):
    """Population vectors labelled by condition, for one recday.

    Mirrors ccgp_state_pairs.build_task_state_matrices -- dedup to one session
    per unique task (sessions [0,3] and [4,7] repeat), require identical neuron
    columns across tasks, drop neurons with no variance in any task, z-score per
    task -- and adds sub-state progress binning and the speed axis.

    With spec.n_progress_bins == 1 and factors == ('state',) the returned `F` is
    identical to that function's output; `check_matches_ccgp` asserts it.

    Returns {'F', 'cond', 'trial_id', 'state', 'task', 'n_neurons', 'n_tasks'}
    or None if the recday is unusable.
    """
    import glm_analysis_v2 as glm
    from ccgp_state_pairs import CCGPConfig

    config = config or SelectivityConfig()
    ccgp_config = ccgp_config or CCGPConfig()
    need_speed = 'speed' in spec.factors

    recday_data = data_dic.get(mouse_recday)
    if not recday_data:
        return None
    sessions, _ = glm.get_sessions_for_glm(recday_data)

    per_task = []
    for sess in sessions:
        sd = recday_data[sess]
        task = np.asarray(sd.get('Task'))
        norm = sd.get('Neurons_norm')
        if norm is None or task is None or task.shape != (ccgp_config.num_task_states,):
            continue
        norm = np.asarray(norm, dtype=float)
        if norm.ndim != 3 or norm.shape[1] < ccgp_config.min_trials:
            continue
        if norm.shape[2] != ccgp_config.num_bins_per_state * ccgp_config.num_task_states:
            continue

        spd = _warped_speed(sd, ccgp_config) if need_speed else None
        if need_speed and spd is None:
            return None
        n_trials = norm.shape[1] if spd is None else min(norm.shape[1], spd.shape[0])

        X, lab, tid = [], [], []
        for t in range(n_trials):
            for s in range(ccgp_config.num_task_states):
                for p, (a, b) in enumerate(_phase_windows(s, spec, ccgp_config)):
                    v = np.nanmean(norm[:, t, a:b], axis=1)
                    X.append(v)
                    key = {'state': s, 'progress': p, 'task': int(sess),
                           'tower': int(task[(s + 1) % ccgp_config.num_task_states]),
                           'speed': np.nanmean(spd[t, a:b]) if spd is not None else np.nan}
                    lab.append(key)
                    tid.append(f'{sess}_{t}')
        X = np.asarray(X, dtype=float)
        X[~np.isfinite(X)] = 0.0
        per_task.append({'session': sess, 'X': X, 'lab': lab, 'tid': tid})

    if len(per_task) < ccgp_config.min_train_tasks + 1:
        return None
    if len({t['X'].shape[1] for t in per_task}) != 1:
        return None

    keep = np.ones(per_task[0]['X'].shape[1], dtype=bool)
    for t in per_task:
        keep &= np.nanstd(t['X'], axis=0) > 0
    if keep.sum() < ccgp_config.min_neurons:
        return None

    for t in per_task:
        Xk = t['X'][:, keep]
        if ccgp_config.zscore == 'per_task':
            Xk = (Xk - Xk.mean(axis=0)) / Xk.std(axis=0)
        t['X'] = Xk

    F = np.vstack([t['X'] for t in per_task])
    labs = [l for t in per_task for l in t['lab']]
    tid = np.array([i for t in per_task for i in t['tid']])

    # Speed is binned by a within-recday median split, so the factor means the
    # same thing across tasks despite drift in absolute running speed.
    if need_speed:
        sp = np.array([l['speed'] for l in labs], dtype=float)
        if not np.isfinite(sp).any():
            return None
        edges = np.nanquantile(sp, np.linspace(0, 1, spec.n_speed_bins + 1)[1:-1])
        sbin = np.digitize(sp, edges)
        for l, b in zip(labs, sbin):
            l['speed'] = int(b)

    cond = np.array(['|'.join(f'{f}{labs[i][f]}' for f in spec.factors)
                     for i in range(len(labs))])
    return {'F': F, 'cond': cond, 'trial_id': tid,
            'state': np.array([l['state'] for l in labs]),
            'task': np.array([l['task'] for l in labs]),
            'n_neurons': int(keep.sum()), 'n_tasks': len(per_task)}


def check_matches_ccgp(data_dic, mouse_recday, config=None, ccgp_config=None):
    """Assert our sampler reproduces ccgp_state_pairs.build_task_state_matrices.

    The CCGP module's synthetics gate its sampler; reproducing it exactly is how
    this module inherits that guarantee, and it is what makes the separability /
    CCGP comparison in §7 a comparison on identical population vectors rather
    than on two things that merely sound alike.
    """
    import ccgp_state_pairs as ccgp
    from ccgp_state_pairs import CCGPConfig

    ccgp_config = ccgp_config or CCGPConfig()
    spec = ConditionSpec('state_only', ('state',))
    ours = build_condition_table(data_dic, mouse_recday, spec, config, ccgp_config)
    theirs = ccgp.build_task_state_matrices(data_dic, mouse_recday, ccgp_config)
    if ours is None or not theirs:
        return None
    F_theirs = np.vstack([t['X'] for t in theirs])
    same_shape = ours['F'].shape == F_theirs.shape
    return {'shape_ours': ours['F'].shape, 'shape_theirs': F_theirs.shape,
            'shapes_match': same_shape,
            'max_abs_diff': float(np.nanmax(np.abs(ours['F'] - F_theirs))) if same_shape else np.nan}


def subsample_neurons(F, n, rng):
    """Fixed-N column subsample, for cross-region comparability.

    M_IC, representation dimensionality and separability all grow with the
    number of neurons. LEC recdays carry 66-186 tracked cells and PFC 1-117, so
    an unmatched LEC-vs-PFC difference in any of them is a statement about
    recording yield, not about cortex.
    """
    if F.shape[1] <= n:
        return F
    return F[:, rng.choice(F.shape[1], n, replace=False)]


# ============================================================================
# §7  Per-recday geometry, N-matching, and the CCGP bridge
# ============================================================================

def run_geometry_recday(data_dic, mouse_recday, spec, config=None, ccgp_config=None,
                        seed=None, table=None):
    """M_IC, representation dimensionality and separability for one recday.

    Reports separability over ALL conditions and over the MERGED independent
    conditions. Posani's headline is that the second is >=0.95 almost
    everywhere while the first tracks response diversity; keeping both is what
    makes that comparison checkable here.

    Neurons are subsampled to config.n_sub_neurons first, so numbers are
    comparable across recdays and regions.
    """
    config = config or SelectivityConfig()
    rng = np.random.default_rng(config.random_state if seed is None else seed)

    if table is None:
        table = build_condition_table(data_dic, mouse_recday, spec, config, ccgp_config)
    if table is None:
        return None

    F_full, cond, tid = table['F'], table['cond'], table['trial_id']

    # Drop conditions with too few distinct trials to cross-validate.
    ok = np.ones(len(cond), dtype=bool)
    for c in np.unique(cond):
        m = cond == c
        if np.unique(tid[m]).size < config.min_trials_per_condition:
            ok &= ~m
    if np.unique(cond[ok]).size < 4:
        return None
    F_full, cond, tid = F_full[ok], cond[ok], tid[ok]

    n_avail = F_full.shape[1]
    if config.require_full_n and n_avail < config.n_sub_neurons:
        return None
    F = subsample_neurons(F_full, config.n_sub_neurons, rng)

    mic = independent_conditions(F, cond, tid, config, seed=config.random_state)
    sep_all = separability(F, cond, tid, config, seed=config.random_state)
    sep_ic = separability(F, mic['cond_merged'], tid, config, seed=config.random_state)

    return {'recday': mouse_recday, 'mouse': mouse_recday[:4], 'spec': spec.name,
            'n_neurons_avail': int(n_avail), 'n_neurons_used': int(F.shape[1]),
            'n_tasks': table['n_tasks'], 'M': int(np.unique(cond).size),
            'm_ic': mic['m_ic'],
            'pr': representation_dimensionality(F, cond, config),
            'pr_ic': representation_dimensionality(F, mic['cond_merged'], config),
            'sep_all': sep_all['sep'], 'ad_all': sep_all['ad'],
            'sep_ic': sep_ic['sep'], 'ad_ic': sep_ic['ad']}


def run_geometry_batch(data_dic, spec, mouse_recdays, config=None, ccgp_config=None,
                       region=None, n_jobs=1, verbose=True):
    """`run_geometry_recday` over many recdays -> tidy DataFrame."""
    import pandas as pd
    config = config or SelectivityConfig()

    def _one(rd):
        try:
            return run_geometry_recday(data_dic, rd, spec, config, ccgp_config)
        except Exception as e:                                  # noqa: BLE001
            if verbose:
                print(f'  {rd}: {type(e).__name__}: {e}')
            return None

    if n_jobs == 1:
        out = [_one(rd) for rd in mouse_recdays]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs)(delayed(_one)(rd) for rd in mouse_recdays)

    rows = [r for r in out if r is not None]
    if verbose:
        print(f'{region or ""} {spec.name}: {len(rows)}/{len(mouse_recdays)} recdays usable')
    df = pd.DataFrame(rows)
    if region is not None and len(df):
        df.insert(0, 'region', region)
    return df


def n_sensitivity(data_dic, spec, mouse_recdays, config=None, ccgp_config=None,
                  n_grid=(40, 60, 80, 120), region=None, verbose=True):
    """How many recdays survive, and how the geometry moves, as the matched N varies.

    Every cross-region claim has to carry this curve. M_IC, PR and separability
    all increase with population size, and the two regions differ systematically
    in yield (LEC 66-186 tracked cells per recday, PFC 1-117), so a difference
    quoted at a single N is not interpretable on its own.
    """
    import pandas as pd
    from dataclasses import replace as _replace
    config = config or SelectivityConfig()

    # Build each recday's table once and reuse it across the N grid.
    tables = {}
    for rd in mouse_recdays:
        try:
            tables[rd] = build_condition_table(data_dic, rd, spec, config, ccgp_config)
        except Exception:                                        # noqa: BLE001
            tables[rd] = None

    frames = []
    for n in n_grid:
        cfg_n = _replace(config, n_sub_neurons=n)
        rows = []
        for rd, tab in tables.items():
            if tab is None or tab['n_neurons'] < n:
                continue                       # below the floor: excluded, not padded
            r = run_geometry_recday(data_dic, rd, spec, cfg_n, ccgp_config, table=tab)
            if r is not None:
                rows.append(r)
        if verbose:
            print(f'  N={n}: {len(rows)} recdays', flush=True)
        if rows:
            df = pd.DataFrame(rows)
            df.insert(0, 'n_matched', n)
            if region is not None:
                df.insert(0, 'region', region)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_ccgp_separability_join(data_dic, mouse_recdays, config=None, ccgp_config=None,
                               region=None, verbose=True):
    """Separability and CCGP on the SAME population vectors, per recday.

    Posani's Discussion states that maximal separability does not imply an
    absence of structure, and that a representation can be abstract in the
    Bernardi (2020) sense and still shatter every dichotomy -- but he never
    measures both. The 'state_task' spec is built from exactly the vectors
    ccgp_state_pairs decodes, so the two numbers are commensurable rather than
    merely similar-sounding.

    Returns one row per recday with sep_all, sep_ic, pr, m_ic and mean CCGP.

    The default `ccgp_config` here lowers `n_shuffles` to 20. The bridge needs
    the CCGP point estimate, not its role-permutation null (which
    `ccgp_state_pairs` already tests properly), and 200 shuffles costs 361 s per
    recday against 23 s at 10. Pass a full CCGPConfig if you also want the null.
    """
    import pandas as pd
    import ccgp_state_pairs as ccgp
    from ccgp_state_pairs import CCGPConfig

    config = config or SelectivityConfig()
    ccgp_config = ccgp_config or replace(CCGPConfig(), n_shuffles=20)
    spec = CONDITION_SPECS['state_task']

    rows = []
    for rd in mouse_recdays:
        try:
            geo = run_geometry_recday(data_dic, rd, spec, config, ccgp_config)
            if geo is None:
                continue
            res = ccgp.run_ccgp_recday(data_dic, rd, ccgp_config)
            cc = res.get('ccgp') if isinstance(res, dict) else None
            if cc is None or not len(cc):
                continue
            # The decoding accuracy column is 'acc'. Name it explicitly rather
            # than falling back to a positional guess: the frame's last column
            # is 'n_trials_test', so a silent fallback averages trial counts
            # and calls them CCGP.
            if 'acc' not in cc.columns:
                raise KeyError(f"run_ccgp_recday returned no 'acc' column; got "
                               f"{list(cc.columns)}")
            geo['ccgp_mean'] = float(np.nanmean(cc['acc'].values))
            # Compare against the module's own ROLE-PERMUTATION null, not against
            # a literal 0.5. Cross-task decoding has no clean analytic chance
            # level -- the training and test tasks differ in composition -- which
            # is exactly why ccgp_state_pairs ships a null per (test task, pair).
            # Testing acc against 0.5 gives a "significantly below chance"
            # artefact when the null itself sits off 0.5.
            geo['ccgp_null'] = (float(np.nanmean(cc['null_mean'].values))
                                if 'null_mean' in cc.columns else np.nan)
            geo['ccgp_delta'] = geo['ccgp_mean'] - geo['ccgp_null']
            geo['ccgp_ceiling'] = (float(np.nanmean(cc['ceiling'].values))
                                   if 'ceiling' in cc.columns else np.nan)
            # Reported so the reader can see it: separability is computed on a
            # config.n_sub_neurons subsample, while run_ccgp_recday uses the full
            # population. Same vectors, different neuron counts.
            geo['ccgp_n_neurons'] = (float(np.nanmean(cc['n_neurons'].values))
                                     if 'n_neurons' in cc.columns else np.nan)
            geo['ccgp_n_pairs'] = int(len(cc))
            rows.append(geo)
            if verbose:
                print(f'  {rd}: sep_all={geo["sep_all"]:.3f} ccgp={geo["ccgp_mean"]:.3f}',
                      flush=True)
        except Exception as e:                                   # noqa: BLE001
            if verbose:
                print(f'  {rd}: {type(e).__name__}: {e}')
    df = pd.DataFrame(rows)
    if region is not None and len(df):
        df.insert(0, 'region', region)
    return df


# ============================================================================
# §5  Synthetics -- THE GATE
# ============================================================================

def make_synthetic_alpha(kind, n_neurons=400, n_vars=8, seed=0, n_groups=8,
                         k=4, separation=4.0, cluster_sd=0.6, decay=0.55,
                         junk_frac=0.5, junk_sd=0.02, rotate=False):
    """Synthetic selectivity clouds that flow through `cluster_quality` unmodified.

    kind:
      'isotropic'   -- Gaussian, identity covariance.        z ~ 0
      'uneven'      -- anisotropic, geometrically decaying variances, NO
                       clusters. z ~ 0.  THIS IS THE GATE: an elongated cloud
                       must not read as categorical.
      'categorical' -- k tight clusters.                     z >> 0
      'junk'        -- 'uneven' plus a spike of near-zero-selectivity cells,
                       which is the artefact Posani warns manufactures a
                       categorical verdict when the inclusion threshold is off.

    `rotate` applies a random orthogonal rotation. A covariance-matched null is
    rotation-invariant, so 'uneven' must give the same z either way -- a cheap
    check that the null really is matching the full covariance and not just the
    per-axis variances.

    Returns (A, group_ids).
    """
    rng = np.random.default_rng(seed)
    group_ids = np.array([f'rec{i % n_groups}' for i in range(n_neurons)])

    if kind == 'isotropic':
        A = rng.normal(size=(n_neurons, n_vars))

    elif kind == 'uneven':
        sd = (1.0 - decay) ** np.arange(n_vars)
        A = rng.normal(size=(n_neurons, n_vars)) * sd[None, :]

    elif kind == 'categorical':
        cents = rng.normal(size=(k, n_vars)) * separation
        which = np.repeat(np.arange(k), int(np.ceil(n_neurons / k)))[:n_neurons]
        A = cents[which] + rng.normal(size=(n_neurons, n_vars)) * cluster_sd

    elif kind == 'junk':
        sd = (1.0 - decay) ** np.arange(n_vars)
        A = rng.normal(size=(n_neurons, n_vars)) * sd[None, :]
        n_junk = int(round(junk_frac * n_neurons))
        A[:n_junk] = rng.normal(size=(n_junk, n_vars)) * junk_sd

    else:
        raise ValueError(f'unknown synthetic alpha kind {kind!r}')

    if rotate:
        Q, _ = np.linalg.qr(rng.normal(size=(n_vars, n_vars)))
        A = A @ Q

    return A, group_ids


def make_synthetic_conditions(kind, n_neurons=120, M=16, n_trials=20, noise=0.35,
                              latent=None, adjacent_gap=None, seed=0):
    """Synthetic population geometries that flow through the real decoders.

    kind:
      'highdim'   -- centroids isotropic in R^n. PR high, separability ~ 1.
      'lowdim'    -- centroids confined to an L-dim subspace (default M//8).
      'collinear' -- centroids evenly spaced along ONE line. Every adjacent pair
                     is decodable, so M_IC is maximal, but the middle-vs-edges
                     dichotomies are not linearly separable, so separability
                     collapses to ~1/3. This is Posani Fig. 6a (middle) and it is
                     the case that proves dimensionality and separability are not
                     the same quantity.

    Returns (F, cond, trial_id) with F of shape (M * n_trials, n_neurons).
    """
    rng = np.random.default_rng(seed)

    if kind in ('highdim', 'lowdim'):
        if kind == 'highdim':
            cents = rng.normal(size=(M, n_neurons))
        else:
            L = latent or max(2, M // 8)
            basis = rng.normal(size=(L, n_neurons))
            cents = rng.normal(size=(M, L)) @ basis
        # ONE global scale, not a per-centroid normalisation: rescaling each
        # centroid to a common norm would distort the relative geometry. Mean
        # norm sqrt(n) puts typical pairwise distances at ~sqrt(2n).
        cents = cents / np.linalg.norm(cents, axis=1).mean() * np.sqrt(n_neurons)

    elif kind == 'collinear':
        # The gap has to be set explicitly, not inherited from a norm
        # constraint. This synthetic exists to show that MAXIMAL M_IC coexists
        # with LOW separability, so every adjacent pair must be comfortably
        # decodable; otherwise adjacent conditions merge, M_IC collapses, and
        # the control silently stops testing what it claims to.
        # Default: match the ~sqrt(2n) pairwise distance of 'highdim', so the
        # two kinds differ in geometry rather than in raw discriminability.
        u = rng.normal(size=(1, n_neurons))
        u = u / np.linalg.norm(u)
        gap = np.sqrt(2.0 * n_neurons) if adjacent_gap is None else adjacent_gap
        cents = (np.arange(M) - (M - 1) / 2.0)[:, None] * gap * u

    else:
        raise ValueError(f'unknown synthetic condition kind {kind!r}')

    F, cond, tid = [], [], []
    for m in range(M):
        F.append(cents[m][None, :] + rng.normal(0, noise, size=(n_trials, n_neurons)))
        cond.extend([m] * n_trials)
        tid.extend(range(n_trials))
    return np.vstack(F), np.asarray(cond), np.asarray(tid)


def make_gaussian_clusters(n_neurons, M, k, sigma, seed=0):
    """Posani's clustered data model (his eq. 15), for the analytic PR check.

        x^mu = z^mu + eta^mu,   z ~ N(0, B),  eta ~ N(0, sigma^2 I)

    B is the clustered covariance: B_ij = 1 when neurons i and j share a cluster,
    0 otherwise. Realised by giving every neuron in a cluster the same draw.

    Returns (n_neurons, M) -- neurons x conditions, his orientation.
    """
    rng = np.random.default_rng(seed)
    which = np.arange(n_neurons) % k
    z = rng.normal(size=(k, M))[which]
    eta = rng.normal(0.0, sigma, size=(n_neurons, M))
    return z + eta


def _plots_import():
    import matplotlib.pyplot as plt
    import glm_analysis_v2 as glm
    glm.apply_gridmaze_style()
    return plt


def _finish(fig, out_path):
    if out_path:
        import matplotlib as mpl
        with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                             'pdf.fonttype': 42, 'ps.fonttype': 42}):
            fig.savefig(out_path, bbox_inches=None, dpi=300)
    return fig


def plot_silhouette_null(result, region='', ax=None, out_path=None):
    """Posani Fig. 3b (right): data silhouette against its matched null."""
    plt = _plots_import()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(2.6, 1.9))
    ax.hist(result['ss_null'], bins=25, color=NULL_GREY, alpha=0.75, label='null')
    ax.axvline(result['ss_data'], color=TRUE_COLOR, lw=1.5, label='data')
    ax.set_xlabel('silhouette score')
    ax.set_ylabel('null draws')
    ax.set_title(f"{region}  z = {result['z']:.2f}, k = {result['k']}")
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_selectivity_matrix(A, result, names, region='', null_A=None,
                            out_path=None, vmax=None):
    """Posani Fig. 3b (left + middle): neurons x variables, sorted by cluster.

    Always plot the null panel beside the data. Sorted selectivity matrices look
    convincingly blocked whenever a few variables are encoded more strongly than
    the rest -- his VISp (categorical) and ACAd (not) look equally structured by
    eye, and only the null distinguishes them. A data panel on its own is not
    evidence.
    """
    plt = _plots_import()
    order = np.lexsort((-result['sil'], result['labels']))
    panels = [('data', A[result['keep']][order])]
    if null_A is not None:
        panels.append(('null', null_A))

    v = vmax if vmax is not None else np.nanpercentile(np.abs(panels[0][1]), 99)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 3.0),
                             squeeze=False)
    for ax, (title, M) in zip(axes[0], panels):
        ax.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v,
                  interpolation='nearest')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90)
        ax.set_title(title)
        ax.set_ylabel('neuron' if title == 'data' else '')
    if null_A is None:
        axes[0][0].set_title(f'{region} (data only — add null_A)')
    fig.suptitle(region)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_null_audit(df, ax=None, out_path=None, z_significant=2.24):
    """§6: the categorical verdict under three nulls, one region per group."""
    plt = _plots_import()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(3.4, 2.1))
    kinds = list(dict.fromkeys(df['null_kind']))
    regions = list(dict.fromkeys(df['region'])) if 'region' in df else ['']
    w = 0.8 / max(len(regions), 1)
    for i, reg in enumerate(regions):
        sub = df[df['region'] == reg] if 'region' in df else df
        x = np.arange(len(kinds)) + i * w - 0.4 + w / 2
        ax.bar(x, [sub[sub['null_kind'] == k]['z'].iloc[0] for k in kinds], width=w,
               color=REGION_COLORS.get(reg, NEUTRAL), label=reg)
    ax.axhline(z_significant, color=NULL_GREY, ls='--', lw=0.8)
    ax.text(0.02, z_significant + 0.15, 'p < 0.05, Bonferroni', fontsize=6, color=NULL_GREY,
            transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, rotation=20, ha='right')
    ax.set_ylabel('silhouette z')
    ax.set_title('same data, three nulls')
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_inclusion_audit(df, ax=None, out_path=None, z_significant=2.24):
    """§6: silhouette z as the full-model R2 floor moves."""
    plt = _plots_import()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(3.2, 2.1))
    regions = list(dict.fromkeys(df['region'])) if 'region' in df else ['']
    for reg in regions:
        sub = df[df['region'] == reg] if 'region' in df else df
        ax.plot(sub['min_r2_full'], sub['z'], 'o-', ms=3, lw=1,
                color=REGION_COLORS.get(reg, NEUTRAL), label=reg)
        for _, r in sub.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r['min_r2_full'], r['z']), fontsize=5,
                        xytext=(0, 4), textcoords='offset points', ha='center',
                        color=REGION_COLORS.get(reg, NEUTRAL))
    ax.axhline(z_significant, color=NULL_GREY, ls='--', lw=0.8)
    ax.set_xlabel('min full-model $R^2$')
    ax.set_ylabel('silhouette z')
    ax.set_title('inclusion threshold vs verdict')
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_variable_contribution(df, region='', ax=None, out_path=None):
    """Posani ED Fig. 4b: drop in silhouette when each variable is removed."""
    plt = _plots_import()
    col = 'delta_z' if 'delta_z' in df.columns else 'delta_ss'
    sub = df[df['dropped'] != '(none)'].sort_values(col)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(2.8, 2.4))
    ax.barh(sub['dropped'], sub[col], color=REGION_COLORS.get(region, NEUTRAL))
    ax.axvline(0, color=NEUTRAL, lw=0.8)
    ax.set_xlabel('change in silhouette z when removed')
    ax.set_title(region)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_separability_vs_ccgp(df, ax=None, out_path=None, sep_col='sep_all'):
    """§7: separability against CCGP, both on the same population vectors."""
    plt = _plots_import()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(2.9, 2.5))
    for reg, sub in df.groupby('region'):
        ax.scatter(sub[sep_col], sub['ccgp_mean'], s=16,
                   color=REGION_COLORS.get(reg, NEUTRAL), label=reg,
                   edgecolor='none', alpha=0.85)
    ax.axhline(0.5, color=NULL_GREY, ls='--', lw=0.8)
    ax.set_xlabel('separability')
    ax.set_ylabel('CCGP (mean over state pairs)')
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_n_sensitivity(df, metric='m_ic', ax=None, out_path=None):
    """§7: a geometry metric against the matched neuron count.

    The point of the figure is that no single N is privileged: read the region
    difference off the whole curve, and off the recday counts printed under it.
    """
    plt = _plots_import()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(3.2, 2.2))
    for reg, sub in df.groupby('region'):
        g = sub.groupby('n_matched')[metric].agg(['mean', 'sem', 'size'])
        ax.errorbar(g.index, g['mean'], yerr=g['sem'], marker='o', ms=3, lw=1,
                    capsize=2, color=REGION_COLORS.get(reg, NEUTRAL), label=reg)
        for n, r in g.iterrows():
            ax.annotate(f"{int(r['size'])}", (n, r['mean']), fontsize=5,
                        xytext=(0, -9), textcoords='offset points', ha='center',
                        color=REGION_COLORS.get(reg, NEUTRAL))
    ax.set_xlabel('matched neurons per recday (labels: n recdays)')
    ax.set_ylabel(metric)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_geometry_summary(df, metrics=('m_ic', 'pr_ic', 'sep_all', 'sep_ic'),
                          out_path=None):
    """Region comparison across the geometry metrics, one panel each.

    Two regions is a contrast, not a gradient: there is no hierarchy axis to
    regress against, so this deliberately does not draw a fitted line.
    """
    plt = _plots_import()
    fig, axes = plt.subplots(1, len(metrics), figsize=(1.6 * len(metrics), 2.2))
    axes = np.atleast_1d(axes)
    regions = list(dict.fromkeys(df['region']))
    for ax, m in zip(axes, metrics):
        for i, reg in enumerate(regions):
            v = df[df['region'] == reg][m].dropna().values
            ax.scatter(np.full(len(v), i) + np.random.uniform(-.09, .09, len(v)), v,
                       s=9, color=REGION_COLORS.get(reg, NEUTRAL), alpha=0.7,
                       edgecolor='none')
            if len(v):
                ax.hlines(np.mean(v), i - 0.25, i + 0.25, color=NEUTRAL, lw=1.2)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions)
        ax.set_title(m)
    fig.tight_layout()
    return _finish(fig, out_path)


def run_synthetic_controls(config=None, seed=0, verbose=True):
    """THE GATE (§5). Every synthetic through the real pipeline, unmodified.

    Gate conditions, in order of how badly a failure invalidates the module:

      1. 'uneven' must read NON-categorical (|z| below the significance line).
         If it does not, the null is not matching the covariance and the module
         is measuring anisotropy.
      2. 'categorical' must read categorical, or the test has no power.
      3. 'collinear' must give high M_IC but LOW separability, or separability
         is just re-measuring dimensionality.
      4. 'highdim' must give separability ~ 1.
      5. The PR row/column identity and the analytic clustered-PR formula must
         hold, or the PR implementation is wrong.

    Returns a pandas DataFrame; prints it with a pass/fail marker per row.
    """
    import pandas as pd

    config = config or SelectivityConfig()
    # min_neurons is a real-data guard; the synthetics are sized above it anyway.
    cfg = replace(config, random_state=seed)
    rows = []

    # --- 1. selectivity-space clustering --------------------------------
    expect_z = {'isotropic': 'ns', 'uneven': 'ns', 'categorical': 'sig', 'junk': 'sig'}
    for kind in ['isotropic', 'uneven', 'categorical', 'junk']:
        A, gids = make_synthetic_alpha(kind, seed=seed)
        res = cluster_quality(A, gids, cfg)
        div = alpha_diversity(A, cfg)
        sig = np.isfinite(res['z']) and res['z'] > 3.0
        ok = (sig if expect_z[kind] == 'sig' else not sig)
        rows.append({'family': 'alpha', 'kind': kind, 'metric': 'silhouette_z',
                     'value': round(float(res['z']), 2),
                     'aux': f"k={res['k']}, a-div={div:.2f}",
                     'expect': expect_z[kind], 'pass': ok})

    # 'uneven' under a random rotation must give the same verdict: a
    # covariance-matched null is rotation-invariant.
    A_rot, gids = make_synthetic_alpha('uneven', seed=seed, rotate=True)
    res_rot = cluster_quality(A_rot, gids, cfg)
    rows.append({'family': 'alpha', 'kind': 'uneven(rotated)', 'metric': 'silhouette_z',
                 'value': round(float(res_rot['z']), 2), 'aux': 'null must be rotation-invariant',
                 'expect': 'ns',
                 'pass': bool(np.isfinite(res_rot['z']) and res_rot['z'] <= 3.0)})

    # --- 2. geometry / separability -------------------------------------
    geo_cfg = replace(cfg, n_sub_neurons=120, n_sub_reps=20, n_dichotomies=60)
    expect_sep = {'highdim': 'high', 'lowdim': 'mid', 'collinear': 'low'}
    for kind in ['highdim', 'lowdim', 'collinear']:
        F, cond, tid = make_synthetic_conditions(kind, seed=seed)
        pr = representation_dimensionality(F, cond, geo_cfg)
        sepres = separability(F, cond, tid, geo_cfg)
        s = sepres['sep']
        ok = {'high': s > 0.9, 'mid': 0.3 < s <= 0.99, 'low': s < 0.6}[expect_sep[kind]]
        rows.append({'family': 'geometry', 'kind': kind, 'metric': 'separability',
                     'value': round(float(s), 3),
                     'aux': f"PR={pr:.2f}, AD={sepres['ad']:.3f}",
                     'expect': expect_sep[kind], 'pass': bool(ok)})

    # collinear must ALSO show that a high M_IC does not buy separability
    F, cond, tid = make_synthetic_conditions('collinear', seed=seed)
    mic = independent_conditions(F, cond, tid, geo_cfg)
    rows.append({'family': 'geometry', 'kind': 'collinear', 'metric': 'M_IC',
                 'value': mic['m_ic'], 'aux': 'high M_IC + low separability',
                 'expect': '>=12', 'pass': bool(mic['m_ic'] >= 12)})

    # --- 3. correctness tests from the paper's own maths ----------------
    # These check OUR implementation against exact identities. Posani's closed
    # form (eq. 25) is deliberately not used as ground truth: it is a large-M
    # approximation and is ~18% off at the condition counts we actually use.
    rng = np.random.default_rng(seed)

    # (a) conditions space and neural space have the same dimensionality
    X = rng.normal(size=(60, 25))
    pr_r, pr_c, ok = check_pr_identity(X)
    rows.append({'family': 'maths', 'kind': 'PR rows == PR cols', 'metric': 'pr',
                 'value': round(pr_r, 4), 'aux': f'cols={pr_c:.4f}',
                 'expect': 'exact', 'pass': bool(ok)})

    # (b) the Gram shortcut in participation_ratio == eq. 16 evaluated directly
    Xc = make_gaussian_clusters(1500, M=16, k=4, sigma=1.0, seed=seed)
    pr_gram = participation_ratio(Xc.T, center=False)
    pr_eq16 = pr_from_covariance_trace(Xc)
    rows.append({'family': 'maths', 'kind': 'PR == eq.16 Tr(C)^2/Tr(C^2)',
                 'metric': 'pr', 'value': round(pr_gram, 4),
                 'aux': f'eq16={pr_eq16:.4f}', 'expect': 'exact',
                 'pass': bool(np.isclose(pr_gram, pr_eq16, rtol=1e-8))})

    # (c) the two asymptotic limits of eq. 26, which ARE exact
    pr_bigM = participation_ratio(
        make_gaussian_clusters(1500, M=2000, k=4, sigma=0.0, seed=seed).T, center=False)
    rows.append({'family': 'maths', 'kind': 'PR -> k as M -> inf', 'metric': 'pr',
                 'value': round(pr_bigM, 3), 'aux': 'k=4', 'expect': '~4',
                 'pass': bool(abs(pr_bigM - 4) < 0.15)})
    pr_bigk = participation_ratio(
        make_gaussian_clusters(2000, M=16, k=2000, sigma=0.0, seed=seed).T, center=False)
    rows.append({'family': 'maths', 'kind': 'PR -> M as k -> inf', 'metric': 'pr',
                 'value': round(pr_bigk, 3), 'aux': 'M=16', 'expect': '~16',
                 'pass': bool(abs(pr_bigk - 16) < 0.5)})

    # (d) eq. 25 only where it is valid (large M)
    pr_meas = participation_ratio(
        make_gaussian_clusters(4000, M=64, k=4, sigma=1.0, seed=seed).T, center=False)
    pr_th = pr_gaussian_clusters_theory(M=64, k=4, sigma=1.0)
    rows.append({'family': 'maths', 'kind': 'eq.25 at M=64', 'metric': 'pr',
                 'value': round(pr_meas, 3), 'aux': f'theory={pr_th:.3f}',
                 'expect': '<5%', 'pass': bool(abs(pr_meas - pr_th) / pr_th < 0.05)})

    df = pd.DataFrame(rows)
    if verbose:
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            print(df.to_string(index=False))
        n_fail = int((~df['pass']).sum())
        print()
        print('GATE PASSES' if n_fail == 0 else f'GATE FAILS: {n_fail} row(s)')
    return df
