"""
Persistent (co)homology of population manifold structure — wake & sleep.

Tests whether population activity lies on a low-dimensional topological manifold
(primarily a cyclic ring S^1 reflecting the cyclic A->B->C->D->A task), and
whether that manifold is preserved during sleep.

Pipeline (Gardner 2022 / Chaudhuri 2019 style)
----------------------------------------------
  1. point cloud  : each (rebinned, Gaussian-smoothed) time bin is one point in
                    neural state space; z-score per neuron -> PCA denoise.
                    Same recipe for wake and sleep so the two are comparable.
  2. subsample    : density + greedy max-min landmarks (ripser cannot take 1e4-1e5
                    points directly).
  3. homology     : ripser Vietoris-Rips, H0-H2, coefficient fields Z/2 and Z/3
                    (Z/3 reveals torsion -> Klein bottle vs torus).
  4. significance : per-neuron circular time-shift shuffle null on the longest
                    H1 / H2 bar.
  5. cohomology   : DREiMac circular-coordinate lift of the most-persistent H1
                    cocycle -> a per-timepoint angle theta, related to behaviour
                    (task phase, head direction, position) by circular correlation.
  6. sleep        : (a) de novo PH on sleep; (b) project sleep onto the wake PCA
                    space and interpolate the wake circular coordinate -> theta(t)
                    timeseries, scored for structured cyclic traversal (replay).

Data format expected
--------------------
  Wake : a session dict as stored in data_dic[mouse_recday][session] with at
         least 'Neuron_raw' (n_neurons, T) uint spike counts at 25 ms bins, and
         (for behaviour) 'Trial_times', 'XY_raw', 'HD_raw', 'Locs_raw', 'Task'.
  Sleep: a raw (n_neurons, T) spike-count array loaded from
         neuron_raw_mingyutest/Neuron_raw_<mouse_recday>_sb_<idx>.npy.

Region-agnostic: works unchanged for LEC (code/) and PFC (mFC_data/code/) —
only the data paths differ. Heavy computation is meant to be driven by
run_ph_batch.py on SLURM; this module just provides the primitives.

Companion of cyclic_structure_analysis.py, which tests cyclic *geometry*
(state-vector angles) rather than *topology*.
"""

import os
import glob
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

BIN_SIZE_MS = 25  # native acquisition bin (40 Hz)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PHConfig:
    """Parameters for the persistent (co)homology pipeline.

    Attributes
    ----------
    bin_ms : int
        Target bin width; raw 25 ms bins are summed into bins of this width.
    smooth_sigma_bins : float
        Gaussian smoothing sigma (in target bins) applied per neuron. The task
        ring is a *slow* (seconds-scale) signal, so smoothing past the fast
        spiking noise is what lets it emerge in instantaneous population vectors.
    zscore : bool
        z-score each neuron before PCA.
    n_pca : int
        PCA dimensionality the cloud is denoised to before computing distances.
    n_landmarks : int
        Number of max-min landmark points fed to ripser / DREiMac.
    density_drop_frac : float
        Fraction of lowest-density points discarded before max-min subsampling
        (removes outliers; 0 disables).
    metric : str
        'euclidean' (default) or 'geodesic' (k-NN graph geodesic distance).
    geodesic_k : int
        k for the geodesic k-NN graph when metric == 'geodesic'.
    maxdim : int
        Top homology dimension (2 -> distinguish ring vs torus vs sphere).
    coeff_fields : tuple[int]
        Prime coefficient fields for ripser (2 and 3 catch Z/2 torsion).
    thresh_pctl : Optional[float]
        Cap the Vietoris-Rips filtration radius at this percentile (0-1) of the
        landmark pairwise distances. The persistent bars are born/die well below
        the diameter, so this massively speeds up H2 with no loss. None -> inf.
    null_maxdim : int
        Homology dimension for the shuffle null (1 = ring-significance only, the
        fast default; 2 also nulls H2 at much greater cost).
    null_n_landmarks : int
        Landmarks used inside each shuffle (fewer than the observed run, since
        the null only needs the top-bar length).
    speed_filter_wake : bool
        Drop low-speed (immobility) bins from wake clouds.
    speed_thresh : Optional[float]
        Speed threshold; None -> the `speed_thresh_pctl`-th percentile of speed.
    speed_thresh_pctl : float
        Percentile used when speed_thresh is None.
    n_shuffles : int
        Per-neuron circular-shift shuffles for the significance null.
    coord_prime : int
        Prime field DREiMac uses for the circular-coordinate cohomology.
    sleep_burst_pctl : float
        Population-rate percentile above which a sleep bin counts as a burst.
    interp_k : int
        Neighbours used to interpolate the wake circular coordinate onto sleep.
    min_neurons : int
        Skip a recording with fewer than this many neurons (PH on a handful of
        neurons is not meaningful). Never triggers on LEC's fixed 151; relevant
        for PFC, whose per-recday counts range 1-117.
    n_bins_per_state, n_states : int
        Task-phase grid of `Neurons_norm` (90 x 4 = 360 bins per trial).
    min_trials : int
        Skip the task-phase ring analysis below this many trials (the
        trial-average would be too noisy / split-half CV underpowered).
    phase_smooth_bins : float
        Circular Gaussian smoothing (sigma, in phase bins) of each neuron's
        360-bin tuning curve. **Default 0 (off) — deliberately.** Smoothing looks
        attractive (with ~10-30 trials the raw per-bin average is noisy) but it
        inflates the NOISE FLOOR just as much as the signal: smoothed noise is a
        smooth random walk, i.e. a closed smooth curve, which manufactures a high
        `top_over_diameter`. Measured calibration (top/diam):
            sigma:          0      4      8     16
            true S1 ring   0.83   0.85   0.85   0.85   <- flat: a real ring needs no smoothing
            pure noise     0.08   0.29   0.47   0.66   <- noise climbs to ring-like values
            real LEC       0.20   0.46   0.50   0.55   <- tracks the NOISE curve, not the ring
        sigma=0 is therefore the most discriminating setting. If you raise it,
        you MUST recalibrate the noise control at the same sigma before believing
        any number.
    random_state : int
        RNG seed.
    """
    bin_ms: int = 100
    smooth_sigma_bins: float = 4.0
    zscore: bool = True
    n_pca: int = 6
    n_landmarks: int = 1200
    density_drop_frac: float = 0.1
    metric: str = "euclidean"
    geodesic_k: int = 12
    maxdim: int = 2
    coeff_fields: tuple = (2, 3)
    thresh_pctl: Optional[float] = 0.9
    null_maxdim: int = 1
    null_n_landmarks: int = 800
    speed_filter_wake: bool = True
    speed_thresh: Optional[float] = None
    speed_thresh_pctl: float = 20.0
    n_shuffles: int = 100
    coord_prime: int = 47
    coord_perc: float = 0.9
    sleep_burst_pctl: float = 75.0
    interp_k: int = 15
    min_neurons: int = 12
    n_bins_per_state: int = 90
    n_states: int = 4
    min_trials: int = 6
    phase_smooth_bins: float = 0.0
    random_state: int = 0

    @property
    def rebin_factor(self) -> int:
        f = int(round(self.bin_ms / BIN_SIZE_MS))
        return max(1, f)

    @property
    def bin_seconds(self) -> float:
        return self.rebin_factor * BIN_SIZE_MS / 1000.0

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Point-cloud construction
# ============================================================================

def rebin_sum(arr, factor, axis=-1):
    """Sum every `factor` consecutive samples along `axis` (truncating remainder)."""
    arr = np.asarray(arr)
    if factor <= 1:
        return arr
    arr = np.moveaxis(arr, axis, -1)
    n = (arr.shape[-1] // factor) * factor
    arr = arr[..., :n]
    new_shape = arr.shape[:-1] + (n // factor, factor)
    out = arr.reshape(new_shape).sum(axis=-1)
    return np.moveaxis(out, -1, axis)


def rebin_mean(arr, factor):
    """Mean every `factor` consecutive samples of a 1-D array."""
    arr = np.asarray(arr, dtype=float)
    if factor <= 1:
        return arr
    n = (len(arr) // factor) * factor
    return arr[:n].reshape(-1, factor).mean(axis=1)


def smooth_population(rates, sigma):
    """Gaussian-smooth each neuron (row) of a (n_neurons, n_bins) matrix in time."""
    if sigma <= 0:
        return rates
    return gaussian_filter1d(rates.astype(float), sigma=sigma, axis=1, mode="nearest")


def build_point_cloud(neuron_raw, config, speed=None, fit_model=None):
    """Turn a (n_neurons, T) raw spike matrix into a denoised point cloud.

    Parameters
    ----------
    neuron_raw : ndarray (n_neurons, T)
        Raw spike counts at 25 ms bins.
    config : PHConfig
    speed : ndarray (T,), optional
        Per-25ms-bin speed; used only when config.speed_filter_wake and
        fit_model is None (wake). Bins below threshold are dropped.
    fit_model : dict, optional
        A model returned by a previous build (keys 'scaler_mean', 'scaler_scale',
        'pca'). If given, the same z-score + PCA transform is *applied* (not
        refit) — used to project sleep into the wake space. Speed filtering is
        skipped in this mode.

    Returns
    -------
    X : ndarray (n_kept_bins, n_pca)
        Denoised cloud; one row per kept time bin.
    keep_idx : ndarray (n_kept_bins,)
        Indices into the rebinned timeline that survived filtering.
    model : dict
        Fitted/used transform: 'scaler_mean', 'scaler_scale', 'pca',
        'n_bins' (rebinned length before filtering), 'pop_rate' (per-bin total).
    """
    from sklearn.decomposition import PCA

    factor = config.rebin_factor
    counts = rebin_sum(neuron_raw, factor, axis=1)          # (n_neurons, n_bins)
    rates = smooth_population(counts, config.smooth_sigma_bins)
    n_bins = rates.shape[1]
    pop_rate = counts.sum(axis=0).astype(float)             # per-bin total spikes

    keep = np.ones(n_bins, dtype=bool)
    if fit_model is None and config.speed_filter_wake and speed is not None:
        spd = rebin_mean(speed, factor)
        spd = _match_len(spd, n_bins)
        thr = (config.speed_thresh if config.speed_thresh is not None
               else np.nanpercentile(spd, config.speed_thresh_pctl))
        keep = spd > thr

    X_full = rates.T                                         # (n_bins, n_neurons)

    if fit_model is None:
        mean = X_full.mean(axis=0)
        scale = X_full.std(axis=0)
        scale[scale == 0] = 1.0
        if not config.zscore:
            mean = np.zeros_like(mean)
            scale = np.ones_like(scale)
        Xz = (X_full - mean) / scale
        n_comp = min(config.n_pca, Xz.shape[1], max(1, keep.sum() - 1))
        pca = PCA(n_components=n_comp, random_state=config.random_state)
        pca.fit(Xz[keep])
        model = dict(scaler_mean=mean, scaler_scale=scale, pca=pca,
                     n_bins=n_bins, pop_rate=pop_rate)
    else:
        mean = fit_model["scaler_mean"]
        scale = fit_model["scaler_scale"]
        pca = fit_model["pca"]
        Xz = (X_full - mean) / scale
        model = dict(scaler_mean=mean, scaler_scale=scale, pca=pca,
                     n_bins=n_bins, pop_rate=pop_rate)

    X = pca.transform(Xz)                                    # (n_bins, n_pca)
    keep_idx = np.where(keep)[0]
    return X[keep_idx], keep_idx, model


def _match_len(arr, n):
    """Pad (edge) or truncate a 1-D array to length n."""
    arr = np.asarray(arr)
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    pad = np.full(n - len(arr), arr[-1] if len(arr) else 0.0)
    return np.concatenate([arr, pad])


# ============================================================================
# Subsampling
# ============================================================================

def maxmin_subsample(X, n_landmarks, seed=0, return_dist=False):
    """Greedy farthest-point (max-min) subsampling for uniform coverage.

    Returns indices of `n_landmarks` points iteratively chosen to maximise the
    minimum distance to the already-selected set.
    """
    n = X.shape[0]
    n_landmarks = int(min(n_landmarks, n))
    rng = np.random.default_rng(seed)
    idx = np.empty(n_landmarks, dtype=int)
    idx[0] = rng.integers(n)
    min_d = np.linalg.norm(X - X[idx[0]], axis=1)
    for i in range(1, n_landmarks):
        idx[i] = int(np.argmax(min_d))
        d = np.linalg.norm(X - X[idx[i]], axis=1)
        min_d = np.minimum(min_d, d)
    if return_dist:
        return idx, min_d
    return idx


def subsample_landmarks(X, config):
    """Density-trim outliers, then max-min subsample to config.n_landmarks.

    Returns landmark indices into X.
    """
    n = X.shape[0]
    if n <= config.n_landmarks:
        return np.arange(n)
    keep = np.arange(n)
    if config.density_drop_frac > 0:
        from sklearn.neighbors import NearestNeighbors
        k = min(config.interp_k, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        d, _ = nn.kneighbors(X)
        density = 1.0 / (d[:, 1:].mean(axis=1) + 1e-12)     # higher = denser
        thr = np.quantile(density, config.density_drop_frac)
        keep = np.where(density >= thr)[0]
    sub = maxmin_subsample(X[keep], config.n_landmarks, seed=config.random_state)
    return keep[sub]


def geodesic_distance_matrix(X, k):
    """Symmetric geodesic distance matrix from a k-NN graph (shortest paths)."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    g = kneighbors_graph(X, n_neighbors=k, mode="distance")
    g = g.maximum(g.T)
    D = shortest_path(g, method="D", directed=False)
    if not np.all(np.isfinite(D)):           # disconnected -> fill with max*1.5
        finite_max = np.nanmax(D[np.isfinite(D)]) if np.any(np.isfinite(D)) else 1.0
        D[~np.isfinite(D)] = finite_max * 1.5
    return D


# ============================================================================
# Persistent homology (ripser)
# ============================================================================

def _pdist_sample(X, seed, max_pts=600):
    """Condensed pairwise distances of up to max_pts rows (for thresh estimation)."""
    from scipy.spatial.distance import pdist
    if X.shape[0] > max_pts:
        idx = np.random.default_rng(seed).choice(X.shape[0], max_pts, replace=False)
        X = X[idx]
    return pdist(X)


def _filtration_thresh(dists, pctl):
    """Filtration cap = pctl-quantile of `dists`; None/empty -> inf (uncapped)."""
    if pctl is None or len(dists) == 0:
        return np.inf
    return float(np.quantile(dists, pctl))


def _bars(dgm):
    """Sorted (descending) finite lifetimes of a persistence diagram."""
    if dgm is None or len(dgm) == 0:
        return np.array([])
    life = dgm[:, 1] - dgm[:, 0]
    life = life[np.isfinite(life)]
    return np.sort(life)[::-1]


def _dim_summary(dgm):
    """Top-bar, second-bar, and gap ratio (top / second) for one dimension."""
    bars = _bars(dgm)
    top = float(bars[0]) if bars.size >= 1 else 0.0
    second = float(bars[1]) if bars.size >= 2 else 0.0
    ratio = float(top / second) if second > 0 else (np.inf if top > 0 else 0.0)
    return dict(top=top, second=second, gap_ratio=ratio, n_bars=int(bars.size),
                lifetimes=bars)


def compute_persistence(X, config, landmark_idx=None, do_cocycles=True):
    """Run ripser over the (sub-sampled) cloud for every coefficient field.

    Parameters
    ----------
    X : ndarray (n_points, n_dims)
    config : PHConfig
    landmark_idx : ndarray, optional
        Pre-computed landmark indices into X; if None they are computed here.
    do_cocycles : bool
        Request cocycle representatives (needed only for the coords path).

    Returns
    -------
    dict with keys:
        'landmark_idx', 'dgms' {coeff: [H0, H1, H2]}, 'cocycles' {coeff: ...},
        'summary' {coeff: {dim: dim_summary}}, 'metric'.
    """
    from ripser import ripser

    if landmark_idx is None:
        landmark_idx = subsample_landmarks(X, config)
    L = X[landmark_idx]

    if config.metric == "geodesic":
        D = geodesic_distance_matrix(L, min(config.geodesic_k, len(L) - 1))
        ripser_in, dm = D, True
        thresh = _filtration_thresh(D[np.triu_indices_from(D, 1)], config.thresh_pctl)
    else:
        ripser_in, dm = L, False
        thresh = _filtration_thresh(_pdist_sample(L, config.random_state),
                                    config.thresh_pctl)

    out = dict(landmark_idx=landmark_idx, dgms={}, cocycles={}, summary={},
               metric=config.metric, n_landmarks=len(L), thresh=thresh)
    for p in config.coeff_fields:
        res = ripser(ripser_in, maxdim=config.maxdim, coeff=p, thresh=thresh,
                     distance_matrix=dm, do_cocycles=do_cocycles)
        out["dgms"][p] = res["dgms"]
        out["cocycles"][p] = res.get("cocycles")
        out["summary"][p] = {d: _dim_summary(res["dgms"][d])
                             for d in range(len(res["dgms"]))}
    return out


# ============================================================================
# Significance — per-neuron circular-shift shuffle null
# ============================================================================

def circular_shift_shuffle(neuron_raw, rng):
    """Independently circular-shift each neuron's spike train by a random offset.

    Destroys cross-neuron correlations (kills any population manifold) while
    preserving every single-neuron autocorrelation / rate.
    """
    out = np.empty_like(neuron_raw)
    T = neuron_raw.shape[1]
    shifts = rng.integers(1, T, size=neuron_raw.shape[0])
    for i in range(neuron_raw.shape[0]):
        out[i] = np.roll(neuron_raw[i], shifts[i])
    return out


def null_persistence(neuron_raw, config, speed=None, dims=(1, 2)):
    """Build a null distribution of longest-bar lengths under the shuffle.

    Returns
    -------
    dict: 'null_top' {dim: ndarray(n_shuffles)} and the config echoed back.
    """
    rng = np.random.default_rng(config.random_state + 1)
    # the null only needs the longest bar up to null_maxdim, computed cheaply
    dims = tuple(d for d in dims if d <= config.null_maxdim)
    null_top = {d: np.zeros(config.n_shuffles) for d in dims}
    null_cfg = PHConfig(**{**config.to_dict(), "coeff_fields": (2,),
                          "maxdim": config.null_maxdim,
                          "n_landmarks": config.null_n_landmarks})
    for s in range(config.n_shuffles):
        sh = circular_shift_shuffle(neuron_raw, rng)
        X, _, _ = build_point_cloud(sh, null_cfg, speed=speed)
        ph = compute_persistence(X, null_cfg, do_cocycles=False)
        summ = ph["summary"][2]
        for d in dims:
            null_top[d][s] = summ.get(d, {}).get("top", 0.0)
    return dict(null_top=null_top, n_shuffles=config.n_shuffles,
                null_maxdim=config.null_maxdim)


def empirical_pvalue(observed_top, null_top):
    """One-sided p that a null top-bar is >= the observed top-bar."""
    null_top = np.asarray(null_top)
    return float((np.sum(null_top >= observed_top) + 1) / (len(null_top) + 1))


# ============================================================================
# Persistent cohomology — circular coordinates (DREiMac)
# ============================================================================

def circular_coordinates(X, config, cocycle_idx=0):
    """Decode a per-point circular coordinate from the most-persistent H1 class.

    Uses DREiMac's persistent-cohomology lift (partition of unity over its own
    max-min landmarks), so a coordinate is returned for *every* row of X.

    Returns
    -------
    dict: 'theta' (n_points,) in [0, 2*pi), 'persistence' diagram, 'model'
          (the fitted CircularCoords object, for diagnostics).
    """
    from dreimac import CircularCoords

    n_land = int(min(config.n_landmarks, X.shape[0]))
    cc = CircularCoords(X, n_landmarks=n_land, prime=config.coord_prime,
                        maxdim=1)
    theta = _call_get_coordinates(cc, cocycle_idx, perc=config.coord_perc)
    theta = np.mod(np.asarray(theta, dtype=float), 2 * np.pi)
    dgm = getattr(cc, "dgms_", [None, None])
    return dict(theta=theta, dgm1=dgm[1] if len(dgm) > 1 else None, model=cc)


def toroidal_coordinates(X, config, cocycle_idxs=(0, 1)):
    """Decode two circular coordinates (toroidal lift) for a torus hypothesis."""
    from dreimac import ToroidalCoords

    n_land = int(min(config.n_landmarks, X.shape[0]))
    tc = ToroidalCoords(X, n_landmarks=n_land, prime=config.coord_prime, maxdim=1)
    try:
        coords = tc.get_coordinates(cocycle_idxs=list(cocycle_idxs))
    except TypeError:
        coords = tc.get_coordinates(cocycle_idx=list(cocycle_idxs))
    coords = np.mod(np.asarray(coords, dtype=float), 2 * np.pi)
    return dict(coords=coords, model=tc)


def _call_get_coordinates(cc, cocycle_idx, perc=0.9):
    """Call DREiMac get_coordinates robustly across versions / weak cocycles.

    A weak / short cohomology class (expected whenever there is no real loop)
    makes DREiMac raise with standard_range=True; we then retry with
    standard_range=False so a (meaningless, correctly non-significant) coordinate
    is still produced rather than aborting the whole session.
    """
    attempts = (
        dict(perc=perc, cocycle_idx=cocycle_idx),
        dict(perc=perc, cocycle_idx=cocycle_idx, standard_range=False),
        dict(perc=perc, cocycle_idx=cocycle_idx, standard_range=False,
             check_cocycle_condition=False),
        dict(cocycle_idx=cocycle_idx, standard_range=False),
        dict(cocycle_idx=[cocycle_idx]),
        dict(),
    )
    last = None
    for kwargs in attempts:
        try:
            return cc.get_coordinates(**kwargs)
        except Exception as e:                # broad: want best-effort coordinate
            last = e
            continue
    raise last


# ============================================================================
# Circular statistics — relate theta to behaviour
# ============================================================================

def circular_linear_corr(theta, x):
    """Mardia circular-linear correlation between angles theta and linear x."""
    theta = np.asarray(theta, float)
    x = np.asarray(x, float)
    m = np.isfinite(theta) & np.isfinite(x)
    if m.sum() < 3:
        return np.nan
    c, s = np.cos(theta[m]), np.sin(theta[m])
    rxc = np.corrcoef(x[m], c)[0, 1]
    rxs = np.corrcoef(x[m], s)[0, 1]
    rcs = np.corrcoef(c, s)[0, 1]
    denom = 1 - rcs**2
    if denom <= 0:
        return np.nan
    return float(np.sqrt(max(0.0, (rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / denom)))


def circular_circular_corr(a, b):
    """Jammalamadaka-Sarma circular-circular correlation between angles a, b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    abar = np.angle(np.mean(np.exp(1j * a)))
    bbar = np.angle(np.mean(np.exp(1j * b)))
    sa, sb = np.sin(a - abar), np.sin(b - bbar)
    num = np.sum(sa * sb)
    den = np.sqrt(np.sum(sa**2) * np.sum(sb**2))
    return float(num / den) if den > 0 else np.nan


def circular_alignment(theta, phi):
    """Resultant length of the residual under the best rotation+reflection map.

    R = max_{s in +/-1} | mean exp( i (theta - s*phi) ) | in [0, 1]. This is the
    natural effect size for "does the topological coordinate `theta` track the
    circular behavioural variable `phi`": R = 1 is a perfect circular relation
    (any offset / direction), R ~ 1/sqrt(N) is chance. Far more stable than the
    Jammalamadaka-Sarma coefficient when phi is near-uniform on the circle.
    """
    theta = np.asarray(theta, float)
    phi = np.asarray(phi, float)
    m = np.isfinite(theta) & np.isfinite(phi)
    if m.sum() < 3:
        return np.nan
    theta, phi = theta[m], phi[m]
    return float(max(abs(np.mean(np.exp(1j * (theta - s * phi)))) for s in (1, -1)))


def corr_with_shuffle_p(theta, var, circular_var=False, n_shuffles=1000, seed=0):
    """Relate theta to `var` with a circular-shift shuffle p-value.

    circular_var=True : statistic = `circular_alignment` resultant length R
                        (reflection-aware), for circular `var` (task phase, HD).
    circular_var=False: statistic = Mardia circular-linear correlation, for
                        linear `var` (goal progress).
    """
    fn = circular_alignment if circular_var else circular_linear_corr
    obs = fn(theta, var)
    if not np.isfinite(obs):
        return dict(r=obs, p=np.nan, null=np.array([]), circular=circular_var)
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffles)
    n = len(theta)
    for i in range(n_shuffles):
        null[i] = fn(np.roll(theta, rng.integers(1, n)), var)
    p = float((np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_shuffles + 1))
    return dict(r=float(obs), p=p, null=null, circular=circular_var)


# ============================================================================
# Sleep — project onto wake manifold & score replay
# ============================================================================

def interpolate_circular_coord(query_pts, ref_pts, ref_theta, k=15):
    """k-NN circular interpolation of a coordinate field onto new points.

    For each query point, take its k nearest reference points and compute the
    inverse-distance-weighted circular mean of their angles. This realises the
    partition-of-unity transfer of the wake circular coordinate onto sleep.
    """
    from sklearn.neighbors import NearestNeighbors
    k = int(min(k, len(ref_pts)))
    nn = NearestNeighbors(n_neighbors=k).fit(ref_pts)
    d, idx = nn.kneighbors(query_pts)
    w = 1.0 / (d + 1e-9)
    z = np.sum(w * np.exp(1j * ref_theta[idx]), axis=1)
    return np.mod(np.angle(z), 2 * np.pi)


def project_onto_wake_manifold(sleep_raw, wake_model, wake_X, wake_theta, config):
    """Map sleep activity into the wake PCA space and decode wake's coordinate.

    Parameters
    ----------
    sleep_raw : ndarray (n_neurons, T_sleep)
    wake_model : dict
        The model returned by build_point_cloud on wake (scaler + pca).
    wake_X : ndarray (n_wake_points, n_pca)
        Wake cloud the coordinate was computed on.
    wake_theta : ndarray (n_wake_points,)
        Wake circular coordinate per wake point.
    config : PHConfig

    Returns
    -------
    dict: 'theta_sleep' (n_sleep_bins,), 'pop_rate' (n_sleep_bins,),
          'X_sleep' (n_sleep_bins, n_pca).
    """
    X_sleep, keep_idx, model = build_point_cloud(sleep_raw, config,
                                                 fit_model=wake_model)
    theta_sleep = interpolate_circular_coord(X_sleep, wake_X, wake_theta,
                                             k=config.interp_k)
    return dict(theta_sleep=theta_sleep, X_sleep=X_sleep,
                pop_rate=model["pop_rate"][keep_idx], keep_idx=keep_idx)


def replay_metrics(theta_t, config, pop_rate=None):
    """Score a decoded sleep angle timeseries for structured cyclic traversal.

    Two complementary, non-degenerate quantities (a *stuck* coordinate must fail
    both):

      coverage      = 1 - |mean exp(i*theta)|  in [0, 1].  How much of the ring
                      the coordinate actually visits (0 = pinned to one phase,
                      ~1 = spread around the circle). Replay needs coverage > 0.
      p_continuity  = shuffle-test that consecutive steps are *smaller* than
                      under a random time order, i.e. the trajectory moves
                      smoothly along the ring rather than jumping. Significant
                      (small p) AND coverage non-trivial => structured traversal.

    Also reports net_revolutions and angular speed. (The old "coherence" metric
    was dropped: it was trivially 1.0 for a near-constant coordinate.)
    """
    theta_t = np.asarray(theta_t, float)
    if theta_t.size < 10:
        return dict(coverage=np.nan, angular_speed=np.nan, step_rad=np.nan,
                    continuity_ratio=np.nan, p_continuity=np.nan,
                    net_revolutions=np.nan)
    dphase = np.angle(np.exp(1j * np.diff(theta_t)))         # wrapped steps
    step = float(np.mean(np.abs(dphase)))
    coverage = float(1.0 - np.abs(np.mean(np.exp(1j * theta_t))))
    net_rev = float(np.abs(np.sum(dphase)) / (2 * np.pi))

    rng = np.random.default_rng(config.random_state + 2)
    n = max(config.n_shuffles, 200)
    null_step = np.empty(n)
    for i in range(n):
        d = np.angle(np.exp(1j * np.diff(theta_t[rng.permutation(theta_t.size)])))
        null_step[i] = np.mean(np.abs(d))
    p_cont = float((np.sum(null_step <= step) + 1) / (n + 1))
    ratio = float(np.mean(null_step) / step) if step > 0 else np.inf

    out = dict(coverage=coverage, angular_speed=step / config.bin_seconds,
               step_rad=step, continuity_ratio=ratio, p_continuity=p_cont,
               net_revolutions=net_rev)
    if pop_rate is not None:
        burst = pop_rate > np.percentile(pop_rate, config.sleep_burst_pctl)
        bm = burst[1:] & burst[:-1]
        if bm.sum() > 10:
            out["burst_coverage"] = float(
                1.0 - np.abs(np.mean(np.exp(1j * theta_t[1:][bm]))))
    return out


# ============================================================================
# Behaviour extraction (wake) — circular task phase & friends
# ============================================================================

def extract_behaviour(session_data, config):
    """Per-(rebinned)-bin behavioural variables aligned to a wake point cloud.

    Reuses glm_analysis_v2.prepare_session_data for the heavy lifting, then
    rebins to config.bin_ms and derives a *circular* task phase (the natural
    coordinate to correlate the neural ring against).

    Returns a dict of length-n_bins arrays: 'task_phase' (rad), 'goal_progress',
    'state', 'hd' (rad), 'speed', plus 'Task'.
    """
    import glm_analysis_v2 as glm
    prep = glm.prepare_session_data(session_data, gp_n_bins=10)
    factor = config.rebin_factor

    state = prep["State"]
    n_states = int(state.max()) + 1 if len(state) else 4
    # continuous goal progress within state (distance-based, falls back to time)
    gp = prep["GP_dist_continuous"].copy()
    if np.all(~np.isfinite(gp)):
        gp = prep["GP_binned"].astype(float) / 10.0
    gp = np.nan_to_num(gp, nan=0.0)
    # circular phase around the full A->B->C->D->A cycle in [0, 2*pi)
    task_phase = 2 * np.pi * ((state + np.clip(gp, 0, 1)) % n_states) / n_states

    hd = np.deg2rad(np.asarray(prep["HD"], float))
    speed = np.asarray(prep["Speed"], float)

    def rb_lin(a):
        return rebin_mean(a, factor)

    def rb_circ(a):
        a = np.asarray(a, float)
        if factor <= 1:
            return a
        n = (len(a) // factor) * factor
        z = np.exp(1j * a[:n]).reshape(-1, factor).mean(axis=1)
        return np.mod(np.angle(z), 2 * np.pi)

    out = dict(
        task_phase=rb_circ(task_phase),
        goal_progress=rb_lin(gp),
        state=rebin_mean(state.astype(float), factor),
        hd=rb_circ(hd),
        speed=rb_lin(speed),
        Task=str(session_data.get("Task", "unknown")),
    )
    return out


# ============================================================================
# Per-session drivers
# ============================================================================

def analyse_wake_session(session_data, config, run_null=True, decode=True):
    """Full wake pipeline for one session.

    Returns a results dict (diagrams, significance, circular coordinate and its
    relation to behaviour) plus the fitted model + cloud needed to later project
    sleep. Returns None if the session lacks usable neural data.
    """
    if session_data.get("Neuron_raw") is None:
        return None
    neuron_raw = np.asarray(session_data["Neuron_raw"])
    if neuron_raw.ndim != 2 or neuron_raw.shape[1] < 50:
        return None
    if neuron_raw.shape[0] < config.min_neurons:
        return None

    speed = None
    if config.speed_filter_wake and session_data.get("XY_raw") is not None:
        import glm_analysis_v2 as glm
        XY = np.asarray(session_data["XY_raw"])
        if XY.ndim == 2 and XY.shape[1] == 2:
            speed = glm.smooth_and_calculate_scalar_derivatives(XY)[:, 2]

    X, keep_idx, model = build_point_cloud(neuron_raw, config, speed=speed)
    ph = compute_persistence(X, config)

    res = dict(state="wake", n_neurons=neuron_raw.shape[0], n_points=X.shape[0],
               config=config.to_dict(), persistence=_strip_cocycles(ph),
               summary=ph["summary"], Task=str(session_data.get("Task", "unknown")))

    if run_null:
        nullp = null_persistence(neuron_raw, config, speed=speed, dims=(1, 2))
        res["null"] = nullp
        for d in nullp["null_top"]:
            obs = ph["summary"][2].get(d, {}).get("top", 0.0)
            res.setdefault("pvalues", {})[d] = empirical_pvalue(
                obs, nullp["null_top"][d])

    if decode:
        try:
            cc = circular_coordinates(X, config)
            theta = cc["theta"]
            res["theta"] = theta
            res["keep_idx"] = keep_idx
            beh = extract_behaviour(session_data, config)
            beh = {k: (_match_len(v, model["n_bins"])[keep_idx]
                       if isinstance(v, np.ndarray) else v)
                   for k, v in beh.items()}
            res["behaviour"] = beh
            res["coord_vs_behaviour"] = dict(
                task_phase=corr_with_shuffle_p(theta, beh["task_phase"],
                                               circular_var=True,
                                               seed=config.random_state),
                goal_progress=corr_with_shuffle_p(theta, beh["goal_progress"],
                                                  seed=config.random_state),
                head_direction=corr_with_shuffle_p(theta, beh["hd"],
                                                   circular_var=True,
                                                   seed=config.random_state),
            )
            res["_model"] = model
            res["_X"] = X
        except Exception as e:                    # decoding is best-effort
            warnings.warn(f"circular coordinate decoding failed: {e!r}")
            res["decode_error"] = repr(e)
    return res


def analyse_sleep_session(sleep_raw, config, wake_result=None, run_null=True):
    """Full sleep pipeline for one sleep-box recording.

    De novo PH on sleep, plus (if wake_result with a stored model is given)
    projection onto the wake manifold and replay scoring.
    """
    sleep_raw = np.asarray(sleep_raw)
    if sleep_raw.ndim != 2 or sleep_raw.shape[1] < 50:
        return None
    if sleep_raw.shape[0] < config.min_neurons:
        return None

    X, keep_idx, model = build_point_cloud(sleep_raw, config)
    ph = compute_persistence(X, config)
    res = dict(state="sleep", n_neurons=sleep_raw.shape[0], n_points=X.shape[0],
               config=config.to_dict(), persistence=_strip_cocycles(ph),
               summary=ph["summary"])

    if run_null:
        nullp = null_persistence(sleep_raw, config, dims=(1, 2))
        res["null"] = nullp
        for d in nullp["null_top"]:
            obs = ph["summary"][2].get(d, {}).get("top", 0.0)
            res.setdefault("pvalues", {})[d] = empirical_pvalue(
                obs, nullp["null_top"][d])

    if wake_result is not None and "_model" in wake_result:
        try:
            proj = project_onto_wake_manifold(
                sleep_raw, wake_result["_model"], wake_result["_X"],
                wake_result["theta"], config)
            res["projection"] = dict(theta_sleep=proj["theta_sleep"],
                                     keep_idx=proj["keep_idx"])
            res["replay"] = replay_metrics(proj["theta_sleep"], config,
                                           pop_rate=proj["pop_rate"])
        except Exception as e:
            warnings.warn(f"sleep projection failed: {e!r}")
            res["projection_error"] = repr(e)
    return res


def _strip_cocycles(ph):
    """Drop bulky cocycle arrays before pickling; keep diagrams + summary."""
    return dict(landmark_idx=ph["landmark_idx"], dgms=ph["dgms"],
                summary=ph["summary"], metric=ph["metric"],
                n_landmarks=ph["n_landmarks"])


# ============================================================================
# Task-phase ring (trial-averaged) + template projection
# ============================================================================
#
# Complementary to the instantaneous-population-vector analysis above. Each point
# is a *task-phase bin* of the trial-averaged response (`Neurons_norm`, 360 bins =
# n_states x n_bins_per_state), so averaging removes the noise and sparse
# excursions that make instantaneous clouds produce outlier-driven H1 bars. Only
# 360 points => ripser is instant and NO landmarks / subsampling / density
# trimming are involved, which is what makes this immune to that artifact.
#
# IMPORTANT (why H1=1 alone proves nothing): the task cycles A->B->C->D->A and bin
# 359 wraps to bin 0, so the averaged trajectory is a CLOSED CURVE BY
# CONSTRUCTION, and any closed curve trivially has H1=1.
#
# PRIMARY CRITERION: `top_over_diameter` = (H1 top bar) / (cloud diameter).
# It is theoretically calibrated: an ideal circle of radius r has its Rips H1 bar
# die at ~sqrt(3)*r with diameter 2r => ~0.87, while a degenerate out-and-back
# path or a blob gives ~0. Synthetic controls (see module tests):
#     true S1 ring      top/diam = 0.82   coverage 0.88   alignment 0.99
#     ramp (open curve) top/diam = 0.13   coverage 0.72   alignment 0.86
#     noise             top/diam = 0.07   coverage 0.07   alignment 0.03
# Because it has a theoretical expectation it needs NO shuffle null — which
# matters, since every shuffle null in this project turned out to be misleading.
#
# TWO TRAPS THIS SECTION EXISTS TO AVOID (both found by the controls above):
#   1. `coverage` alone does NOT prove a ring: the *ramp* (an open arc with no
#      hole) still scores 0.72, because DREiMac will happily wrap a circular
#      coordinate around an open curve. Coverage is necessary, not sufficient.
#   2. The per-neuron phase-shift null is INVALID here (see
#      `taskphase_shift_null`): shifting each neuron's tuning bump only reassigns
#      which neuron prefers which phase — the population still tiles phase, so the
#      ring SURVIVES the shuffle (null coverage 0.87 vs real 0.88, p=0.56). It is
#      kept only for reference and is NOT used by default.
#
# Evidence for a genuine task ring therefore = top_over_diameter near ~0.5-0.87
# AND one dominant H1 bar AND high coverage AND theta winding with task phase
# (alignment), reproduced across a split-half of trials.


def _cloud_from_mean(M, config):
    """(n_neurons, n_bins) trial-averaged matrix -> (n_bins, n_pca) cloud + model.

    z-scores each neuron across task-phase bins, then PCA. Shared by
    `build_taskphase_cloud` and the shift null so they are strictly comparable.
    """
    from sklearn.decomposition import PCA
    M = np.nan_to_num(np.asarray(M, dtype=float), nan=0.0)
    if config.phase_smooth_bins > 0:
        # circular ('wrap') smoothing along task phase — the cycle is periodic
        M = gaussian_filter1d(M, sigma=config.phase_smooth_bins, axis=1,
                              mode="wrap")
    mean = M.mean(axis=1, keepdims=True)
    scale = M.std(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    Mz = (M - mean) / scale if config.zscore else M
    X = Mz.T                                        # (n_bins, n_neurons)
    n_comp = int(min(config.n_pca, X.shape[1], max(1, X.shape[0] - 1)))
    pca = PCA(n_components=n_comp, random_state=config.random_state).fit(X)
    model = dict(pca=pca, scaler_mean=mean.ravel(), scaler_scale=scale.ravel(),
                 n_bins=X.shape[0])
    return pca.transform(X), model


def build_taskphase_cloud(session_data, config, trials=None, neurons_norm=None):
    """Trial-averaged task-phase cloud: (n_bins=360, n_pca), one point per phase bin.

    Parameters
    ----------
    session_data : dict
        Session dict carrying 'Neurons_norm' (n_neurons, n_trials, n_bins).
    config : PHConfig
    trials : array-like of int, optional
        Trial subset to average (for split-half CV). None -> all trials.
    neurons_norm : ndarray, optional
        Supply the array directly (bypasses session_data); accepts a
        (n_neurons, n_bins) already-averaged matrix too.

    Returns
    -------
    (X, model) or (None, None) if unusable.
    """
    nn = neurons_norm if neurons_norm is not None else session_data.get("Neurons_norm")
    if nn is None:
        return None, None
    nn = np.asarray(nn, dtype=float)
    if nn.ndim == 3:
        if trials is not None:
            trials = np.asarray(trials, dtype=int)
            if trials.size == 0:
                return None, None
            nn = nn[:, trials, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN trial slices
            M = np.nanmean(nn, axis=1)
    elif nn.ndim == 2:
        M = nn
    else:
        return None, None
    if M.shape[0] < config.min_neurons or M.shape[1] < 8:
        return None, None
    return _cloud_from_mean(M, config)


def _ring_geometry(X, top_bar):
    """Non-degeneracy of a loop: is it a round ring or a flat out-and-back path?

    For an ideal circle the Rips H1 bar dies at ~sqrt(3)*r while the diameter is
    2r, so `top_over_diameter` ~ 0.87. A degenerate / collinear path gives ~0.
    `pc12_frac` is the share of variance in the first two PCs (a planar loop needs
    two).
    """
    from scipy.spatial.distance import pdist
    d = pdist(X)
    diameter = float(d.max()) if d.size else np.nan
    var = X.var(axis=0)
    pc12 = float(var[:2].sum() / var.sum()) if var.sum() > 0 else np.nan
    return dict(diameter=diameter,
                top_over_diameter=float(top_bar / diameter) if diameter else np.nan,
                pc12_frac=pc12)


def _ring_cfg(config):
    """maxdim=1, single field: the averaged trajectory is a 1-D curve, so H2 is
    meaningless here — and maxdim=2 on 360 points costs ~21 s vs ~1 s."""
    return PHConfig(**{**config.to_dict(), "maxdim": 1,
                       "coeff_fields": (config.coeff_fields[0],)})


def _ring_stats(X, config):
    """H1 top bar / gap ratio, theta, coverage and theta-vs-task-phase alignment."""
    config = _ring_cfg(config)
    ph = compute_persistence(X, config, landmark_idx=np.arange(X.shape[0]),
                             do_cocycles=False)
    coeff = config.coeff_fields[0]
    s = ph["summary"][coeff][1]
    theta = circular_coordinates(X, config)["theta"]
    coverage = float(1.0 - np.abs(np.mean(np.exp(1j * theta))))
    phase = 2 * np.pi * np.arange(X.shape[0]) / X.shape[0]
    align = circular_alignment(theta, phase)
    return dict(top=s["top"], gap_ratio=s["gap_ratio"], theta=theta,
                coverage=coverage, alignment=align, dgms=ph["dgms"][coeff],
                summary=ph["summary"][coeff])


def taskphase_shift_null(M, config, n_shuffles=None):
    """DEPRECATED / INVALID as a ring test — kept for reference only.

    Circularly shifts each neuron's 360-bin tuning curve independently. The
    intent was to destroy cross-neuron phase alignment, but it does NOT: shifting
    a tuning bump only changes which phase that neuron prefers, so the population
    still tiles task phase and the ring survives intact. Verified on a synthetic
    S1: real coverage 0.883 vs null coverage 0.866 (p=0.56) — i.e. it cannot
    detect the very structure it is supposed to null out.

    Use `_ring_geometry`'s `top_over_diameter` (theoretically calibrated) instead.
    """
    rng = np.random.default_rng(config.random_state + 3)
    n = int(n_shuffles if n_shuffles is not None else config.n_shuffles)
    tops = np.zeros(n); covs = np.zeros(n); aligns = np.zeros(n)
    M = np.nan_to_num(np.asarray(M, dtype=float), nan=0.0)
    for i in range(n):
        Ms = np.stack([np.roll(row, rng.integers(1, row.size)) for row in M])
        Xs, _ = _cloud_from_mean(Ms, config)
        st = _ring_stats(Xs, config)
        tops[i], covs[i], aligns[i] = st["top"], st["coverage"], st["alignment"]
    return dict(null_top=tops, null_coverage=covs, null_alignment=aligns,
                n_shuffles=n)


def analyse_taskphase_ring(session_data, config, run_null=False, n_shuffles=None,
                           null_if_coverage=0.25):
    """Task-phase ring test for one session (see section header for the logic).

    Judge the result on `geometry['top_over_diameter']` (~0.87 = ideal circle,
    ~0 = degenerate/open) TOGETHER WITH coverage + alignment + cv_alignment.
    Coverage alone is not sufficient (an open ramp scores ~0.72).

    `run_null` defaults to False because the per-neuron shift null is invalid
    here — it preserves the ring (see `taskphase_shift_null`). It is retained
    only for reference/diagnostics.

    Returns
    -------
    dict or None
        Keys: 'top', 'gap_ratio', 'coverage', 'alignment' (theta vs task phase),
        'geometry' (diameter, top_over_diameter, pc12_frac),
        'cv_alignment' (split-half agreement), 'theta', '_X', '_model'.
        None if the session lacks Neurons_norm / enough trials / enough neurons.
    """
    nn = session_data.get("Neurons_norm")
    if nn is None:
        return None
    nn = np.asarray(nn, dtype=float)
    if nn.ndim != 3:
        return None
    n_neurons, n_trials, n_bins = nn.shape
    if n_trials < config.min_trials or n_neurons < config.min_neurons:
        return None

    X, model = build_taskphase_cloud(session_data, config)
    if X is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        M = np.nanmean(nn, axis=1)

    st = _ring_stats(X, config)
    res = dict(state="ring", n_neurons=n_neurons, n_trials=int(n_trials),
               n_bins=int(n_bins), config=config.to_dict(),
               Task=str(session_data.get("Task", "unknown")),
               top=st["top"], gap_ratio=st["gap_ratio"], coverage=st["coverage"],
               alignment=st["alignment"], theta=st["theta"],
               dgms=st["dgms"], summary=st["summary"],
               geometry=_ring_geometry(X, st["top"]),
               _X=X, _model=model)

    # split-half CV across trials: do the two halves give the same coordinate?
    idx = np.arange(n_trials)
    Xa, _ = build_taskphase_cloud(session_data, config, trials=idx[0::2])
    Xb, _ = build_taskphase_cloud(session_data, config, trials=idx[1::2])
    if Xa is not None and Xb is not None:
        ta = circular_coordinates(Xa, config)["theta"]
        tb = circular_coordinates(Xb, config)["theta"]
        res["cv_alignment"] = circular_alignment(ta, tb)
        res["cv_coverage"] = (float(1 - abs(np.mean(np.exp(1j * ta)))),
                              float(1 - abs(np.mean(np.exp(1j * tb)))))

    gated = (null_if_coverage is not None
             and np.isfinite(st["coverage"]) and st["coverage"] < null_if_coverage)
    if run_null and not gated:
        nullp = taskphase_shift_null(M, config, n_shuffles=n_shuffles)
        res["null"] = nullp
        res["pvalues"] = dict(
            top=empirical_pvalue(st["top"], nullp["null_top"]),
            coverage=empirical_pvalue(st["coverage"], nullp["null_coverage"]),
            alignment=empirical_pvalue(st["alignment"], nullp["null_alignment"]),
        )
    elif run_null:
        res["null_skipped"] = (f"coverage {st['coverage']:.3f} < "
                               f"{null_if_coverage} — no ring to test")
    return res


def build_recday_taskphase_matrix(recday_dict, config, sessions=None):
    """Pool a recday's task-phase tuning across ALL its sessions.

    Each session of a recday is a DIFFERENT reward configuration, but the same
    neurons (by row) are tracked throughout. Averaging each neuron's 360-bin
    task-phase response across sessions therefore cancels place/session-specific
    tuning and keeps only what is consistent in abstract A->B->C->D phase — the
    representation an abstract task ring would live in.

    Session-weighted (per the design decision): each session's trial-mean is
    computed first, then those per-session means are averaged, so every reward
    config contributes equally regardless of its trial count. A-anchored: bin 0
    is 'just after reward A' in every session, so the raw phase bins align.

    Returns
    -------
    dict or None: 'M' (n_neurons, n_bins) session-averaged tuning,
        'session_means' (list of per-session (n_neurons, n_bins)),
        'sessions', 'n_sessions', 'n_trials_total', 'n_neurons'.
    """
    keys = list(recday_dict.keys()) if sessions is None else sessions
    means, used, n_trials = [], [], 0
    n_neurons = None
    for s in keys:
        sd = recday_dict.get(s, {})
        nn = sd.get("Neurons_norm")
        if nn is None:
            continue
        nn = np.asarray(nn, dtype=float)
        if nn.ndim != 3 or nn.shape[1] < 1:
            continue
        if n_neurons is None:
            n_neurons = nn.shape[0]
        elif nn.shape[0] != n_neurons:      # neuron sets must match to pool
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means.append(np.nanmean(nn, axis=1))   # (n_neurons, n_bins)
        used.append(s)
        n_trials += nn.shape[1]
    if len(means) < 1 or n_neurons is None or n_neurons < config.min_neurons:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        M = np.nanmean(np.stack(means, axis=0), axis=0)   # session-weighted mean
    return dict(M=M, session_means=means, sessions=used, n_sessions=len(used),
                n_trials_total=int(n_trials), n_neurons=int(n_neurons))


def fold_to_goalprogress(M, n_states=4):
    """Average the `n_states` legs of a (n_neurons, n_states*per) task-phase matrix
    into a single (n_neurons, per) GOAL-PROGRESS matrix (period-folded).

    The task cycles A->B->C->D and goal progress (0->1 within each leg) repeats
    every leg, so folding the 4 legs onto each other isolates the within-leg
    (goal-progress / distance-to-reward) code and doubles its SNR.
    """
    M = np.asarray(M, dtype=float)
    per = M.shape[1] // n_states
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack([M[:, i * per:(i + 1) * per]
                                    for i in range(n_states)], axis=0), axis=0)


def leg_similarity(M, n_states=4):
    """Mean pairwise correlation between the `n_states` legs' flattened tuning.

    Diagnostic for period-N (goal-progress) vs full-task-phase (identity) coding:
    ~1  => the legs are identical  => goal-progress / period-N dominated, A/B/C/D
           IDENTITY nearly absent (the trajectory 'repeats every per bins');
    ~0 / negative => tuning is unique per task-phase bin (identity / abstract ring).
    Calibrated on synthetics (controls_placecells): speed 1.00, place*speed 0.94,
    session-inconsistent place 0.27, abstract-360 ring -0.33; real LEC ~0.99.
    """
    M = np.asarray(M, dtype=float)
    per = M.shape[1] // n_states
    legs = [M[:, i * per:(i + 1) * per].ravel() for i in range(n_states)]
    cc = [np.corrcoef(legs[i], legs[j])[0, 1]
          for i in range(n_states) for j in range(i + 1, n_states)]
    return float(np.nanmean(cc)) if cc else np.nan


def analyse_recday_taskphase_ring(recday_dict, config, sessions=None):
    """Recday-averaged (cross-session) task-phase ring test.

    The best-powered / most-abstract version of `analyse_taskphase_ring`: one
    pooled (n_neurons, 360) matrix per recday. Judge on `top_over_diameter`
    (~0.87 ideal circle, ~0 degenerate) together with coverage + alignment;
    split-half CV is across SESSIONS (odd/even), meaningful only if coverage>0.
    """
    pooled = build_recday_taskphase_matrix(recday_dict, config, sessions)
    if pooled is None:
        return None
    X, model = _cloud_from_mean(pooled["M"], config)
    st = _ring_stats(X, config)
    res = dict(state="ringavg", n_neurons=pooled["n_neurons"],
               n_sessions=pooled["n_sessions"],
               n_trials_total=pooled["n_trials_total"],
               n_bins=int(pooled["M"].shape[1]), config=config.to_dict(),
               top=st["top"], gap_ratio=st["gap_ratio"], coverage=st["coverage"],
               alignment=st["alignment"], theta=st["theta"],
               dgms=st["dgms"], summary=st["summary"],
               geometry=_ring_geometry(X, st["top"]), _X=X, _model=model)

    # goal-progress (period-fold) view: is the code organised by within-leg
    # progress (repeats every 90 bins) rather than A/B/C/D identity?
    M = pooled["M"]
    res["leg_similarity"] = leg_similarity(M, config.n_states)
    M90 = fold_to_goalprogress(M, config.n_states)
    X90, _ = _cloud_from_mean(M90, config)
    st90 = _ring_stats(X90, config)
    res["fold"] = dict(top_over_diameter=_ring_geometry(X90, st90["top"])["top_over_diameter"],
                       coverage=st90["coverage"], top=st90["top"],
                       gap_ratio=st90["gap_ratio"], n_bins=int(M90.shape[1]))
    res["_X90"] = X90

    # split-half CV across sessions (odd vs even), each half session-averaged
    sm = pooled["session_means"]
    if len(sm) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Ma = np.nanmean(np.stack(sm[0::2], axis=0), axis=0)
            Mb = np.nanmean(np.stack(sm[1::2], axis=0), axis=0)
        Xa, _ = _cloud_from_mean(Ma, config)
        Xb, _ = _cloud_from_mean(Mb, config)
        ta = circular_coordinates(Xa, config)["theta"]
        tb = circular_coordinates(Xb, config)["theta"]
        res["cv_alignment"] = circular_alignment(ta, tb)
        res["cv_coverage"] = (float(1 - abs(np.mean(np.exp(1j * ta)))),
                              float(1 - abs(np.mean(np.exp(1j * tb)))))
    return res


def project_onto_taskphase_ring(neuron_raw, ring_model, ring_X, ring_theta,
                                config, speed=None):
    """Decode the ring's circular coordinate for instantaneous activity.

    The ring is built from time-warped, trial-averaged *rates* while `neuron_raw`
    is rebinned smoothed *counts*, so each is z-scored per neuron **by its own
    statistics** before applying the ring's PCA — that is the bridge between the
    two representations. `decoded theta vs true task phase on wake` is the
    validation of this bridge; if that fails the sleep decode is meaningless.

    Returns
    -------
    dict: 'theta_t' (n_kept_bins,), 'keep_idx', 'pop_rate', 'X'.
    """
    factor = config.rebin_factor
    counts = rebin_sum(np.asarray(neuron_raw), factor, axis=1)
    rates = smooth_population(counts, config.smooth_sigma_bins)
    n_bins = rates.shape[1]
    pop_rate = counts.sum(axis=0).astype(float)

    keep = np.ones(n_bins, dtype=bool)
    if config.speed_filter_wake and speed is not None:
        spd = _match_len(rebin_mean(speed, factor), n_bins)
        thr = (config.speed_thresh if config.speed_thresh is not None
               else np.nanpercentile(spd, config.speed_thresh_pctl))
        keep = spd > thr

    mean = rates.mean(axis=1, keepdims=True)
    scale = rates.std(axis=1, keepdims=True); scale[scale == 0] = 1.0
    Xz = ((rates - mean) / scale).T if config.zscore else rates.T
    X = ring_model["pca"].transform(Xz)
    keep_idx = np.where(keep)[0]
    X = X[keep_idx]
    theta_t = interpolate_circular_coord(X, ring_X, np.asarray(ring_theta),
                                         k=config.interp_k)
    return dict(theta_t=theta_t, keep_idx=keep_idx, pop_rate=pop_rate[keep_idx],
                X=X, n_bins=n_bins)


# ============================================================================
# Sleep-file IO
# ============================================================================

def sleep_dir(data_root):
    return os.path.join(data_root, "processed_data", "neuron_raw_mingyutest")


def load_sleep_files(mouse_recday, data_root, idx=None):
    """Load sleep-box arrays for a session key.

    Parameters
    ----------
    mouse_recday : str
        e.g. 'ah08_20250613_20250615' (the data_dic key).
    data_root : str
        Path to the repo's data/ directory.
    idx : int or None
        Specific sleep-box index, or None for all (sorted by index).

    Returns
    -------
    list of (idx:int, array:(n_neurons, T)).
    """
    d = sleep_dir(data_root)
    pat = os.path.join(d, f"Neuron_raw_{mouse_recday}_sb_*.npy")
    out = []
    for f in glob.glob(pat):
        base = os.path.basename(f)
        try:
            sb = int(base.rsplit("_sb_", 1)[1].split(".npy")[0])
        except (IndexError, ValueError):
            continue
        if idx is not None and sb != idx:
            continue
        out.append((sb, np.load(f, allow_pickle=True)))
    return sorted(out, key=lambda t: t[0])


def load_pfc_sleep_files(recday, data_folder, idx=None):
    """Load PFC sleep-box arrays for a recday (sibling of load_sleep_files).

    PFC sleep recordings live in mFC_data/data/Neuronal_activity/Awake_Sleep/ as
    joblib-serialised `binned_FR_dic_<recday>_<idx>` files. Despite the name each
    is a raw (n_neurons, T) spike-count matrix with the *same neurons (by row) as
    the wake recday*, so it drops straight into analyse_sleep_session and the
    wake-template projection.

    Parameters
    ----------
    recday : str
        PFC recday key, e.g. 'ab03_01092023_02092023'.
    data_folder : str
        Path to mFC_data/data/.
    idx : int or None
        Specific sleep index, or None for all (sorted by index).

    Returns
    -------
    list of (idx:int, array:(n_neurons, T)).
    """
    import joblib
    d = os.path.join(data_folder, "Neuronal_activity", "Awake_Sleep")
    pat = os.path.join(d, f"binned_FR_dic_{recday}_*")
    out = []
    for f in glob.glob(pat):
        tail = os.path.basename(f).rsplit("_", 1)[-1]
        try:
            sb = int(tail)
        except ValueError:
            continue
        if idx is not None and sb != idx:
            continue
        arr = joblib.load(f)
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
            arr = arr.item()
        out.append((sb, np.asarray(arr)))
    return sorted(out, key=lambda t: t[0])


# ============================================================================
# Results IO + summary (notebook side)
# ============================================================================

def load_ph_outputs(out_dir, mouse_recday=None, state=None):
    """Load every result pickle in `out_dir` into a dict keyed by file stem.

    Filters by mouse_recday substring and/or state ('wake'/'sleep') if given.
    """
    import pickle
    res = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "*.pkl"))):
        stem = os.path.basename(f)[:-4]
        if mouse_recday and mouse_recday not in stem:
            continue
        if state and f"__{state}" not in stem:
            continue
        with open(f, "rb") as fh:
            res[stem] = pickle.load(fh)
    return res


def summarise(results):
    """One row per loaded result: the headline topology + decoding numbers.

    Returns a pandas DataFrame (sorted by mouse_recday, state).
    """
    import pandas as pd
    rows = []
    for stem, r in results.items():
        # task-phase ring results live in the same output dir but have a
        # dimension-keyed summary (maxdim=1) rather than the coeff-keyed one used
        # by wake/sleep; they get their own table in the ring section.
        if r.get("state") == "ring" or 2 not in r.get("summary", {}):
            continue
        s = r["summary"][2]
        pv = r.get("pvalues", {})
        row = dict(stem=stem, mouse_recday=r.get("mouse_recday", ""),
                   state=r.get("state", ""), n_neurons=r.get("n_neurons"),
                   n_points=r.get("n_points"),
                   H1_top=s[1]["top"], H1_ratio=s[1]["gap_ratio"],
                   H1_p=pv.get(1, np.nan),
                   H2_top=s[2]["top"], H2_ratio=s[2]["gap_ratio"],
                   H2_p=pv.get(2, np.nan))
        cvb = r.get("coord_vs_behaviour", {})
        for k in ("task_phase", "goal_progress", "head_direction"):
            if k in cvb:
                row[f"{k}_R"] = cvb[k]["r"]
                row[f"{k}_p"] = cvb[k]["p"]
        if "replay" in r:
            row["replay_coverage"] = r["replay"]["coverage"]
            row["replay_p_continuity"] = r["replay"]["p_continuity"]
            row["replay_continuity_ratio"] = r["replay"]["continuity_ratio"]
            row["net_revolutions"] = r["replay"]["net_revolutions"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mouse_recday", "state"]).reset_index(drop=True)


# ============================================================================
# Plotting (publication style via glm_analysis_v2.apply_gridmaze_style)
# ============================================================================

_DIM_COLORS = {0: "0.5", 1: "crimson", 2: "steelblue"}


def plot_persistence(res, ax=None, coeff=2):
    """Persistence diagram (H0/H1/H2) with the shuffle-null H1 threshold marked."""
    import matplotlib.pyplot as plt
    from persim import plot_diagrams
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    dgms = res["persistence"]["dgms"][coeff]
    plot_diagrams(dgms, ax=ax, show=False,
                  labels=[f"H{d}" for d in range(len(dgms))], size=8)
    null = res.get("null", {}).get("null_top", {})
    if 1 in null:
        thr = np.percentile(null[1], 95)
        ax.axhline(ax.get_ylim()[0] + thr, ls="--", lw=0.8, color="crimson",
                   alpha=0.6, label="H1 null 95%")
    ax.set_title(f"{res.get('mouse_recday','')} {res.get('state','')}", fontsize=8)
    return ax


def plot_barcode(res, dim=1, ax=None, coeff=2, max_bars=40):
    """Horizontal barcode for one homology dimension (longest bars on top)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 2.2))
    dgm = res["persistence"]["dgms"][coeff][dim]
    if len(dgm) == 0:
        ax.set_title(f"H{dim}: no bars"); return ax
    life = dgm[:, 1] - dgm[:, 0]
    order = np.argsort(life)[::-1][:max_bars]
    for i, k in enumerate(order):
        ax.plot([dgm[k, 0], dgm[k, 1]], [i, i], lw=1.2, color=_DIM_COLORS.get(dim))
    ax.set_yticks([]); ax.set_xlabel("filtration radius")
    ax.set_title(f"H{dim} barcode  (ratio {res['summary'][coeff][dim]['gap_ratio']:.1f})")
    return ax


def plot_significance(res, dim=1, ax=None):
    """Observed top bar vs the shuffle-null distribution of top bars."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 2.2))
    null = res.get("null", {}).get("null_top", {})
    if dim not in null:
        ax.set_title(f"H{dim}: no null"); return ax
    obs = res["summary"][2][dim]["top"]
    ax.hist(null[dim], bins=20, color="0.7", edgecolor="0.4")
    ax.axvline(obs, color="crimson", lw=2,
               label=f"obs (p={res.get('pvalues',{}).get(dim,np.nan):.3f})")
    ax.set_xlabel(f"H{dim} longest bar"); ax.set_ylabel("# shuffles")
    ax.legend(fontsize=7)
    return ax


def plot_coord_vs_behaviour(res, var="task_phase", ax=None, circular=True):
    """Scatter the decoded circular coordinate against a behavioural variable."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    theta = res.get("theta")
    beh = res.get("behaviour", {}).get(var)
    if theta is None or beh is None:
        ax.set_title(f"{var}: n/a"); return ax
    ax.scatter(beh, theta, s=2, alpha=0.3, color="steelblue", edgecolors="none")
    stat = res.get("coord_vs_behaviour", {}).get(
        {"task_phase": "task_phase", "goal_progress": "goal_progress",
         "hd": "head_direction"}.get(var, var), {})
    ax.set_xlabel(var); ax.set_ylabel(r"neural $\theta$")
    ttl = var
    if stat:
        key = "R" if stat.get("circular") else "r"
        ttl += f"  {key}={stat['r']:.2f}, p={stat['p']:.3f}"
    ax.set_title(ttl)
    return ax


def plot_manifold(res, color_by="theta", ax=None, dims=(0, 1)):
    """2-D PCA scatter of the cloud coloured by decoded angle or task phase."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3))
    X = res.get("_X")
    if X is None:
        ax.set_title("no stored cloud"); return ax
    if color_by == "theta":
        c, cmap = res.get("theta"), "hsv"
    else:
        c = res.get("behaviour", {}).get(color_by)
        cmap = "hsv" if color_by in ("task_phase", "hd") else "viridis"
    sc = ax.scatter(X[:, dims[0]], X[:, dims[1]], c=c, cmap=cmap, s=3,
                    alpha=0.5, edgecolors="none")
    plt.colorbar(sc, ax=ax, fraction=0.046, label=color_by)
    ax.set_xlabel(f"PC{dims[0]+1}"); ax.set_ylabel(f"PC{dims[1]+1}")
    ax.set_aspect("equal", "datalim")
    ax.set_title(f"manifold ({color_by})")
    return ax


def plot_pc_grid(res, color_by="goal_progress", n_pc=4, pairs=None, s=4):
    """Pairwise scatter grid of the stored PCA cloud, coloured by a variable.

    Lets you view the manifold "from every angle" among the stored PCs (the
    cloud is `config.n_pca`-D, so PC3/PC4... are the higher axes). A real ring
    shows the colour winding around an empty centre in *some* projection; a
    gradient across a filled blob does not.

    Parameters
    ----------
    res : dict
        A wake result (must carry the '_X' cloud).
    color_by : str
        'theta' or any key in res['behaviour'] (goal_progress, task_phase, ...).
    n_pc : int
        Use all C(n_pc, 2) pairs of the first n_pc PCs (ignored if `pairs` given).
    pairs : list[(i,j)], optional
        Explicit 0-based PC index pairs to plot.
    """
    import matplotlib.pyplot as plt
    from itertools import combinations
    X = res.get("_X")
    if X is None:
        raise ValueError("result has no stored cloud ('_X')")
    if color_by == "theta":
        c, cmap = res.get("theta"), "hsv"
    else:
        c = res.get("behaviour", {}).get(color_by)
        cmap = "hsv" if color_by in ("task_phase", "hd") else "viridis"
    ev = res.get("_model", {}).get("pca")
    ev = ev.explained_variance_ratio_ if ev is not None else None
    pairs = pairs or list(combinations(range(min(n_pc, X.shape[1])), 2))
    ncol = min(3, len(pairs)); nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.8 * nrow),
                             squeeze=False)
    sc = None
    for ax, (i, j) in zip(axes.ravel(), pairs):
        sc = ax.scatter(X[:, i], X[:, j], c=c, cmap=cmap, s=s, alpha=0.5,
                        edgecolors="none")
        lx = f"PC{i+1}" + (f" ({ev[i]*100:.0f}%)" if ev is not None else "")
        ly = f"PC{j+1}" + (f" ({ev[j]*100:.0f}%)" if ev is not None else "")
        ax.set_xlabel(lx); ax.set_ylabel(ly); ax.set_aspect("equal", "datalim")
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    if sc is not None:
        fig.colorbar(sc, ax=axes, fraction=0.025, label=color_by)
    fig.suptitle(f"{res.get('mouse_recday','')} s{res.get('session','')} "
                 f"PC grid (colour={color_by})", fontsize=9)
    return fig


def plot_taskphase_ring(res, ax=None, dims=(0, 1), show_states=True):
    """The trial-averaged task-phase trajectory, drawn as an ORDERED curve.

    Colour = task-phase bin (0->360, hsv), grey line connects consecutive bins
    (and closes the cycle), black markers = A/B/C/D state centroids. A genuine
    ring is a smooth loop enclosing a hole; real LEC/PFC data is a noisy scatter
    whose consecutive bins jump around. Title reports the calibrated criterion
    `top/diam` (ideal circle ~0.87, degenerate ~0).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 3.2))
    X = res.get("_X")
    if X is None:
        ax.set_title("no stored ring cloud"); return ax
    nb = X.shape[0]
    i, j = dims
    ax.plot(np.r_[X[:, i], X[0, i]], np.r_[X[:, j], X[0, j]], "-",
            color="0.75", lw=0.7, zorder=1)
    ax.scatter(X[:, i], X[:, j], c=np.arange(nb), cmap="hsv", s=10, zorder=2)
    if show_states and nb >= 4:
        per = nb // 4
        for k, lab in enumerate("ABCD"):
            c = X[k * per:(k + 1) * per, [i, j]].mean(axis=0)
            ax.scatter(*c, s=150, c="k", zorder=3)
            ax.annotate(lab, c, color="w", ha="center", va="center",
                        fontweight="bold", zorder=4)
    g = res.get("geometry", {})
    ax.set_xlabel(f"PC{i+1}"); ax.set_ylabel(f"PC{j+1}")
    ax.set_aspect("equal", "datalim")
    ax.set_title(f"{res.get('mouse_recday','')} s{res.get('session','')} "
                 f"({res.get('n_trials','?')} trials)\n"
                 f"top/diam={g.get('top_over_diameter', np.nan):.2f} "
                 f"coverage={res.get('coverage', np.nan):.3f}  "
                 f"[ring 0.83 | noise 0.08]", fontsize=7)
    return ax


def _ringavg_h1_dgm(res, which="full"):
    """H1 diagram for a ring / ringavg result. 'full' = stored 360-bin diagram;
    'fold' = recomputed from the stored folded goal-progress cloud `_X90`."""
    if which == "fold":
        X90 = res.get("_X90")
        if X90 is None:
            raise ValueError("result lacks _X90 — re-run the recday-average sweep")
        cfg = _ring_cfg(PHConfig(**res.get("config", {})))
        ph = compute_persistence(X90, cfg, landmark_idx=np.arange(X90.shape[0]),
                                 do_cocycles=False)
        return ph["dgms"][cfg.coeff_fields[0]][1]
    return res["dgms"][1]                       # stored full-360 H1


def plot_h1_barcode(res=None, which="full", ax=None, max_bars=30, ref=False,
                    dgm=None, label=None):
    """H1 barcode (longest bar on top) for a ring/ringavg result.

    which : 'full' (360-bin task phase) or 'fold' (folded goal-progress cloud).
    dgm   : pass a diagram directly (overrides res/which) — used for the reference.
    ref   : draw green (the synthetic true-ring reference panel).
    Same style as `plot_barcode`; the top/2nd ratio is in the title — a genuine
    ring shows ONE long bar (ratio >> 1), no ring shows many similar bars.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 2.2))
    if dgm is None:
        dgm = _ringavg_h1_dgm(res, which)
    if label is None:
        label = (res or {}).get("mouse_recday", "")
    if dgm is None or len(dgm) == 0:
        ax.set_title(f"{label}\nH1: no bars", fontsize=6); ax.set_yticks([]); return ax
    life = dgm[:, 1] - dgm[:, 0]
    order = np.argsort(life)[::-1][:max_bars]
    srt = np.sort(life)[::-1]
    ratio = srt[0] / srt[1] if len(srt) > 1 and srt[1] > 0 else np.inf
    colour = "seagreen" if ref else _DIM_COLORS[1]
    for i, k in enumerate(order):
        ax.plot([dgm[k, 0], dgm[k, 1]], [i, i], lw=1.4, color=colour)
    ax.set_yticks([]); ax.invert_yaxis()
    ax.set_title(f"{label}\nratio {ratio:.1f}", fontsize=6)
    return ax


def _clean_ring_h1(n_pts=90, seed=0):
    """H1 diagram of a clean noisy circle — the 'what a ring looks like' reference."""
    rng = np.random.default_rng(seed)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
    X = np.zeros((n_pts, 6)); X[:, 0] = np.cos(ang); X[:, 1] = np.sin(ang)
    X += rng.normal(0, 0.05, X.shape)
    cfg = _ring_cfg(PHConfig(coeff_fields=(2,), n_pca=6))
    ph = compute_persistence(X, cfg, landmark_idx=np.arange(n_pts), do_cocycles=False)
    return ph["dgms"][cfg.coeff_fields[0]][1]


def plot_ringavg_barcode_grid(results, which="fold", ncols=6, max_bars=30):
    """Small-multiples grid of H1 barcodes, one per recday-averaged result.

    Sorted by folded top/diam (most ring-like first). The first panel is a clean
    true-ring reference (one long dominant bar) for scale — real recdays should
    instead show many similar-length bars.
    """
    import matplotlib.pyplot as plt
    rings = [r for r in results.values() if r.get("state") == "ringavg"]
    rings.sort(key=lambda r: -r.get("fold", {}).get("top_over_diameter", 0))
    ref_dgm = _clean_ring_h1(90 if which == "fold" else 360)

    n = len(rings) + 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 1.8 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    plot_h1_barcode(ax=axes[0], dgm=ref_dgm, ref=True, max_bars=max_bars,
                    label="TRUE ring (ref)")
    for ax, r in zip(axes[1:], rings):
        plot_h1_barcode(r, which=which, ax=ax, max_bars=max_bars)
    for ax in axes[n:]:
        ax.axis("off")
    lbl = "goal-progress (folded)" if which == "fold" else "task phase (360)"
    fig.suptitle(f"Recday-averaged H1 barcodes — {lbl}  "
                 f"(true ring = one long bar; real = many similar bars)", fontsize=9)
    fig.tight_layout()
    return fig


def _clean_ring_cloud(n_pts=360, seed=0):
    """A clean noisy circle in 2-D — the 'what a loop looks like' reference."""
    rng = np.random.default_rng(seed)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
    X = np.zeros((n_pts, 2))
    X[:, 0], X[:, 1] = np.cos(ang), np.sin(ang)
    return X + rng.normal(0, 0.03, X.shape)


def _loop_panel(ax, X, label, td, show_states=False):
    """One PC1-PC2 loop panel (colour = bin index) for the loop grid."""
    ax.plot(np.r_[X[:, 0], X[0, 0]], np.r_[X[:, 1], X[0, 1]], "-",
            color="0.8", lw=0.5, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], c=np.arange(X.shape[0]), cmap="hsv", s=5, zorder=2)
    if show_states and X.shape[0] >= 4:
        per = X.shape[0] // 4
        for k, lab in enumerate("ABCD"):
            c = X[k * per:(k + 1) * per, :2].mean(0)
            ax.scatter(*c, s=45, c="k", zorder=3)
            ax.annotate(lab, c, color="w", ha="center", va="center",
                        fontsize=5, fontweight="bold", zorder=4)
    ax.set_aspect("equal", "datalim"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{label}\ntop/diam={td:.2f}", fontsize=6)


def plot_ringavg_loop_grid(results, which="fold", ncols=6):
    """Small-multiples grid of PCA loops, one per recday-averaged ring — the loop
    counterpart of `plot_ringavg_barcode_grid` (same layout / sort / reference).

    which='full' = 360-bin task-phase loop (with A/B/C/D centroids);
    'fold' = folded 90-bin goal-progress loop. Colour = bin index (hsv). Sorted by
    folded top/diam; first panel is a clean true-ring reference for scale.
    """
    import matplotlib.pyplot as plt
    rings = [r for r in results.values() if r.get("state") == "ringavg"]
    rings.sort(key=lambda r: -r.get("fold", {}).get("top_over_diameter", 0))

    ref_X = _clean_ring_cloud(90 if which == "fold" else 360)
    rcfg = _ring_cfg(PHConfig(coeff_fields=(2,), n_pca=6))
    rph = compute_persistence(ref_X, rcfg, landmark_idx=np.arange(len(ref_X)),
                              do_cocycles=False)
    ref_td = _ring_geometry(ref_X, rph["summary"][rcfg.coeff_fields[0]][1]["top"])["top_over_diameter"]

    n = len(rings) + 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.0 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    _loop_panel(axes[0], ref_X, "TRUE ring (ref)", ref_td)
    key = "geometry" if which == "full" else "fold"
    for ax, r in zip(axes[1:], rings):
        X = r["_X"] if which == "full" else r["_X90"]
        _loop_panel(ax, X, r.get("mouse_recday", ""), r[key]["top_over_diameter"],
                    show_states=(which == "full"))
    for ax in axes[n:]:
        ax.axis("off")
    lbl = "goal-progress (folded 90)" if which == "fold" else "task phase (360)"
    fig.suptitle(f"Recday-averaged PCA loops — {lbl}  "
                 f"(colour = bin index; true ring = clean circle)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_goalprogress(res, axes=None):
    """3-panel view of a recday-averaged result's goal-progress structure.

    Left  : full trajectory coloured by TASK PHASE (0..n_bins) — A/B/C/D identity.
    Middle: full trajectory coloured by GOAL PROGRESS (period n_bins/n_states) —
            if the legs overlap into one clean gradient, the code 'repeats every
            per bins' (goal-progress dominated).
    Right : the folded goal-progress cloud (a genuine ring closes; the real data
            is an open horseshoe ARC). Title reports folded top/diam + leg_similarity.
    Requires '_X' and '_X90' (produced by analyse_recday_taskphase_ring).
    """
    import matplotlib.pyplot as plt
    X, X90 = res.get("_X"), res.get("_X90")
    if X is None or X90 is None:
        raise ValueError("result lacks _X/_X90 — re-run the recday-average sweep")
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    nb, per = X.shape[0], X90.shape[0]
    gp = np.tile(np.arange(per), nb // per)
    axes[0].scatter(X[:, 0], X[:, 1], c=np.arange(nb), cmap="hsv", s=12)
    axes[0].set_title(f"colour = task phase (0..{nb})", fontsize=8)
    axes[1].scatter(X[:, 0], X[:, 1], c=gp, cmap="hsv", s=12)
    axes[1].set_title(f"colour = goal progress (period {per})", fontsize=8)
    axes[2].plot(np.r_[X90[:, 0], X90[0, 0]], np.r_[X90[:, 1], X90[0, 1]],
                 "-", color="0.7", lw=0.8)
    axes[2].scatter(X90[:, 0], X90[:, 1], c=np.arange(per), cmap="hsv", s=18)
    f = res.get("fold", {})
    axes[2].set_title(f"folded to goal progress\ntop/diam={f.get('top_over_diameter', np.nan):.2f} "
                      f"legSim={res.get('leg_similarity', np.nan):.2f}  [ring 0.83]", fontsize=8)
    for a in axes:
        a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_aspect("equal", "datalim")
    axes[0].figure.suptitle(f"{res.get('mouse_recday','')} — goal-progress structure", fontsize=9)
    return axes[0].figure


def plot_ring_3d(res, color_by="goalprogress", which="full", ax=None,
                 elev=22, azim=-60, s=14):
    """3-D (PC1-2-3) scatter of a ring/ringavg cloud, coloured by task phase or
    goal progress. `which='full'` uses `_X` (360 pts); 'fold' uses `_X90` (90).
    Static (renders under nbconvert); for live rotation use `%matplotlib widget`.
    """
    import matplotlib.pyplot as plt
    X = res.get("_X" if which == "full" else "_X90")
    if X is None or X.shape[1] < 3:
        raise ValueError("result lacks a 3-D cloud (_X/_X90 with >=3 PCs)")
    n = X.shape[0]
    if color_by == "goalprogress" and which == "full":
        per = res.get("_X90").shape[0] if res.get("_X90") is not None else n // 4
        c = np.tile(np.arange(per), n // per)
    else:
        c = np.arange(n)
    if ax is None:
        ax = plt.figure(figsize=(4.2, 4)).add_subplot(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c, cmap="hsv", s=s, depthshade=False)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.view_init(elev=elev, azim=azim)
    g = res.get("geometry", {}); f = res.get("fold", {})
    ax.set_title(f"{res.get('mouse_recday','')} ({which}, {color_by})\n"
                 f"top/diam={g.get('top_over_diameter', np.nan):.2f} "
                 f"fold={f.get('top_over_diameter', np.nan):.2f} "
                 f"legSim={res.get('leg_similarity', np.nan):.2f}", fontsize=7)
    return ax


def plot_sleep_replay(res, config_bin_s=0.1, ax=None, t_window=None):
    """Decoded sleep angle theta(t) over time (replay traversal view)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 2.2))
    proj = res.get("projection", {})
    theta = proj.get("theta_sleep")
    if theta is None:
        ax.set_title("no sleep projection"); return ax
    t = np.arange(len(theta)) * config_bin_s
    if t_window:
        m = (t >= t_window[0]) & (t < t_window[1])
        t, theta = t[m], theta[m]
    ax.scatter(t, theta, s=2, c=theta, cmap="hsv", alpha=0.6, edgecolors="none")
    ax.set_xlabel("sleep time (s)"); ax.set_ylabel(r"decoded $\theta$")
    rp = res.get("replay", {})
    ax.set_title(f"sleep replay  coverage={rp.get('coverage',np.nan):.2f} "
                 f"continuity p={rp.get('p_continuity',np.nan):.3f} "
                 f"(rev={rp.get('net_revolutions',np.nan):.1f})")
    return ax


if __name__ == "__main__":
    print(__doc__)
    print("Default config:", PHConfig())
