"""Manual correction and QC of brainreg_probe probe fits.

The automated pipeline in ``brainreg_probe`` fits a plane to thresholded DiI
signal (PCA) and then places the probe within that plane by optimising five
parameters (depth, shrinkage, width scaling, in-plane rotation ``theta``,
lateral offset).  It mis-placed several tracks in this dataset.  Diagnosis,
measured from the saved ``ProbeA_fit_params.json`` (see ``PROBE_REFIT.md``):

    mouse  LATERAL   AP   theta   depth
    ah08     15.0   12.8    0.1    4225   plausible
    ah10      1.8   25.4  -24.3    2205   FAILS  <- tilt in the wrong plane
    ly05     17.9    1.5  +14.1    2775   FAILS
    ly06      6.0    3.2   -0.5    3971   good
    ly07      6.8   10.6    1.0    3769   FAILS  <- tilt in the wrong plane

The insertion was a nominal **10 degree lateral approach**, so a correct fit has
its tilt in the coronal/lateral plane (~10 deg), ~0 deg in AP, and ``theta`` ~ 0
for a straight insertion.  ah10 has its tilt 25 deg in *AP* and then a -24 deg
in-plane rotation that nearly cancels it -- the signature of a mis-fitted plane
plus a compensating optimiser.  A plane tilted off the true track also truncates
the projected extent, which is why its depth is half the cohort's.

Only the tip-most 705 um of the probe is recorded (384 channels, 96 per shank
over 4 shanks), so a trajectory or depth error puts *every* unit in the wrong
structure.

Nothing here re-runs brainreg: steps 6-12 of the tracing pipeline are a pure
function of (plane, 5 params, volumes) and re-run in seconds.

Three tiers of correction, all sharing :func:`refit_probe`:

1. ``override``               - keep the plane, override placement parameters.
2. ``trajectory_constrained`` - re-fit with the plane's tilt bounded to the
   surgical prior and ``theta`` bounded to +-10 deg.  The dye still drives the
   fit; the prior only supplies bounds.
3. ``manual_track``           - the plane comes from a hand-annotated entry and
   tip.  **Authoritative**: never overridden, re-optimised against the DiI mask,
   or failed by the harness.  See :func:`qc_probe_fit` for who judges what.

Must run in the ``histology`` conda env (needs skimage / tifffile /
probeinterface / brainglobe_heatmap).
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import contextlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths and the brainreg_probe import.
#
# ``probeinterface_tracing`` runs ``pd.read_csv('./brainreg_probe/...')`` at
# import time, relative to the working directory, and its
# PREPROCESSED_BRAINREG_PATH is the relative '../data/preprocessed_data/brainreg'.
# Both assume ``code/`` is the cwd, so we chdir there for the import and derive
# all of our own paths absolutely from REPO_ROOT.
# --------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
BRAINREG_DIR = REPO_ROOT / "data" / "preprocessed_data" / "brainreg"
FIGURE_DIR = REPO_ROOT / "data" / "figures" / "probe_refit"
ATLAS_NAME = "allen_mouse_10um"


@contextlib.contextmanager
def _cwd(path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

with _cwd(CODE_DIR):
    from brainreg_probe import probeinterface_tracing as pit
    from brainreg_probe import plot_util_func as puf


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

VOXEL_SIZE_UM = 10.0

#: Recorded bank: 384 channels over 4 shanks, y in [0, 705] um from the tip.
RECORDED_BANK_MAX_UM = 705.0

#: Nominal surgical trajectory: 10 degree lateral approach, no AP tilt.
SURGICAL_PRIOR = {"lateral_deg": 10.0, "ap_deg": 0.0, "tol_deg": 7.0}

PROBE_MANUFACTURER = "imec"
PROBE_NAME = "NP2020"
TARGET_REGION = "ENTl"
CONTACT_FACE_AXIS = "rl"
INSERTION_AXIS = "si"

PARAM_ORDER = (
    "probe_depth",
    "brain_shrinkage_pct",
    "probe_width_scaling",
    "theta",
    "offset_x",
)

#: Wider than the upstream defaults.  ``theta`` in particular was +-pi/2, which
#: is what let ah10's -24 deg rotation through silently; a straight insertion
#: has no reason to rotate the probe within its own plane.
DEFAULT_BOUNDS = {
    "probe_depth": (500.0, 8000.0),
    "brain_shrinkage_pct": (0.0, 8.0),
    "probe_width_scaling": (0.75, 1.05),
    "theta": (-np.pi / 18, np.pi / 18),  # +-10 degrees
    "offset_x": (-1000.0, 1000.0),
}

#: QC flag thresholds.  Calibrated by run_synthetic_controls(), not hand-tuned
#: until they pass -- see PROBE_REFIT.md.
#:
#: ``resid_um`` applies to **signal2contact** (mean distance from each dye voxel
#: to the nearest contact), never to contact2signal.  The two answer different
#: questions and only one is diagnostic: contact2signal asks "is every contact
#: near dye?", which is high whenever dye is sparse or has faded even though the
#: fit is perfect (ah08: 376 um with an otherwise excellent fit), and *low* for a
#: truncated fit that sits wholly inside a longer dye cloud (ah10: 51 um).
#: signal2contact asks "is every bit of dye explained by the probe?", which
#: correctly separates them (ah08 29 um, ah10 177 um).
QC_THRESHOLDS = {
    "theta_deg": 10.0,
    "trajectory_dev_deg": 12.0,
    "signal_coverage": 0.70,
    "resid_um": 150.0,
    # Angle between the fitted trajectory and the dye's own principal axis.
    # Separates the two real cases with a wide margin: ah10's automated fit was
    # ~21 deg off its dye (mis-fitted), ly07 is 1.7 deg off its dye despite a
    # 9 deg AP tilt (faithful to an AP-tilted track).
    #
    # NECESSARY, NOT SUFFICIENT -- it measures angle only.  A probe rotated about
    # its surface anchor keeps a perfect 0.0 deg score while its tip swings
    # depth*sin(angle) away from the track (ah10: 574 um at 9.7 deg, comparable
    # to the whole 750 um shank span).  Always read it beside
    # `resid_signal2contact_um`, which is what actually catches misplacement.
    "fit_vs_dye_axis_deg": 10.0,
}

_VOLUME_CACHE: dict = {}
_SIGNAL_CACHE: dict = {}


# --------------------------------------------------------------------------
# Trajectory geometry
#
# Orientation is 'asr': i = anterior->posterior, j = superior->inferior,
# k = right->left.  v_axis points from the probe tip up towards the surface.
#
# We parameterise it by two angles from vertical:
#   lateral_deg -- tilt in the coronal plane (k vs j), positive towards -k
#   ap_deg      -- tilt in the sagittal plane (i vs j), positive towards -i
# The signs are chosen so that the cohort's fits come out positive; verified
# against ah08, whose stored v_axis [-0.214, -0.944, -0.253] round-trips to
# lateral 15.0 / ap 12.8 and back to within 1e-3.
# --------------------------------------------------------------------------


def trajectory_angles(v_axis) -> dict:
    """Decompose a ``v_axis`` into lateral / AP tilt from vertical, in degrees.

    Exact inverse of :func:`v_axis_from_angles`: the pair is a composition of
    two rotations of the vertical, so it is norm-preserving and round-trips to
    machine precision (a naive independent-angle form does not, and drifts by
    ~0.4 deg when both tilts are large).
    """
    v = np.asarray(v_axis, dtype=float)
    v = v / np.linalg.norm(v)
    up = np.array([0.0, -1.0, 0.0])
    total = np.degrees(np.arccos(np.clip(v @ up, -1.0, 1.0)))
    ap = np.degrees(np.arctan2(-v[0], np.hypot(v[1], v[2])))
    lateral = np.degrees(np.arctan2(-v[2], -v[1]))
    return {"lateral_deg": lateral, "ap_deg": ap, "total_tilt_deg": total}


def v_axis_from_angles(lateral_deg: float, ap_deg: float) -> np.ndarray:
    """Vertical rotated by ``ap_deg`` in the sagittal plane, then ``lateral_deg``
    in the coronal plane.  Exact inverse of :func:`trajectory_angles`."""
    lat = np.radians(lateral_deg)
    ap = np.radians(ap_deg)
    return np.array([
        -np.sin(ap),
        -np.cos(ap) * np.cos(lat),
        -np.cos(ap) * np.sin(lat),
    ])


def dye_principal_axis(signal_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Principal axis of a dye cloud, oriented superior, plus its explained variance.

    This is the **data-driven** reference for a fit's trajectory, and it is what
    the prior cannot be: the dye is ground truth, the 10 deg is nominal.  Cf.
    `fit_vs_dye_axis_deg` in :func:`qc_probe_fit`.
    """
    from sklearn.decomposition import PCA

    pts = signal_df[["i", "j", "k"]].values.astype(float)
    if len(pts) < 10:
        return np.array([np.nan] * 3), float("nan")
    pca = PCA(3).fit(pts - pts.mean(0))
    pc1 = pca.components_[0]
    pc1 = pc1 if pc1[1] < 0 else -pc1        # point superior, like v_axis
    return pc1, float(pca.explained_variance_ratio_[0])


#: True centre-to-centre shank pitch of an NP2020, in um.  A fixed property of
#: the silicon, not something a fit is entitled to change.
TRUE_SHANK_PITCH_UM = 250.0


def _comb_loglik(u: np.ndarray, centre: float, pitch: float, sigma: float,
                 weights: np.ndarray) -> float:
    """Log-likelihood of a 4-component Gaussian comb with shared spacing."""
    centres = centre + (np.arange(4) - 1.5) * pitch
    z = (u[:, None] - centres[None, :]) / sigma
    comp = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi))
    return float(np.log(np.maximum(comp @ weights, 1e-300)).sum())


def dye_shank_pitch(signal_df: pd.DataFrame, plane: dict, *,
                    pitch_range: tuple[float, float] = (140.0, 360.0),
                    n_boot: int = 60, seed: int = 0) -> dict:
    """Measure the shank pitch the DYE actually shows, independent of the fit.

    Fits a **4-component Gaussian comb with shared spacing** to the dye's
    in-plane ``u`` coordinate by maximum likelihood -- free parameters are the
    array centre, the pitch, a shared streak width, and the four weights (the
    shanks are unevenly labelled, so the weights must be free).

    Two simpler estimators were tried first and **both failed their gate**, which
    is why this one exists:

    * *1-D KMeans on u* scattered +-30% within a single mouse -- the streaks
      overlap and are unevenly populated, so cluster assignment is unstable.
    * *A periodogram* (argmax of ``|sum exp(2*pi*i*u/p)|``) was **systematically
      biased high on every synthetic case tested** (+2.8 to +17.0 um, never low):
      four repeats is too short a train, so the peak is broad and its argmax
      drifts toward longer pitches as jitter rises.

    ``sigma_um`` (fitted streak width) and the bootstrap CI width are the
    confidence measures; a wide CI means the dye does not resolve four streaks
    and the number should not be leaned on.
    """
    from scipy.optimize import minimize

    pts = signal_df[["i", "j", "k"]].values.astype(float)
    u = (pts - np.asarray(plane["surface_coord"], float)) @ np.asarray(
        plane["u_axis"], float) * VOXEL_SIZE_UM
    nan = {"pitch_um": float("nan"), "sigma_um": float("nan"),
           "ci_lo": float("nan"), "ci_hi": float("nan"), "n": int(len(u))}
    if len(u) < 40:
        return nan

    def _fit(sample):
        best = (None, -np.inf)
        for p0 in np.linspace(pitch_range[0] + 20, pitch_range[1] - 20, 7):
            def nll(x):
                c, logp, logs, *lw = x
                p, s = np.exp(logp), np.exp(logs)
                if not (pitch_range[0] <= p <= pitch_range[1]) or s <= 1:
                    return 1e12
                w = np.exp(np.r_[0.0, lw])
                return -_comb_loglik(sample, c, p, s, w / w.sum())
            x0 = [float(np.median(sample)), np.log(p0),
                  np.log(max(np.std(sample) / 3, 10.0)), 0.0, 0.0, 0.0]
            try:
                r = minimize(nll, x0, method="Nelder-Mead",
                             options={"maxiter": 4000, "xatol": 1e-3,
                                      "fatol": 1e-3})
            except Exception:
                continue
            if -r.fun > best[1]:
                best = (r.x, -r.fun)
        return best[0]

    x = _fit(u)
    if x is None:
        return nan
    pitch, sigma = float(np.exp(x[1])), float(np.exp(x[2]))

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        xb = _fit(rng.choice(u, size=len(u), replace=True))
        if xb is not None:
            boot.append(float(np.exp(xb[1])))
    boot = np.array(boot) if boot else np.array([pitch])
    return {"pitch_um": pitch, "sigma_um": sigma,
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "n": int(len(u))}


def fitted_shank_pitch(params: dict) -> float:
    """Shank pitch the fit implies, in um.

    ``transform_2d_probe`` scales in-plane x by ``width_scaling * (1-shrink/100)``
    and y by only ``(1-shrink/100)``, so the probe can be squeezed across its
    width independently of its length -- an anisotropic distortion of a rigid
    object.  This is what that squeeze does to the 250 um pitch.
    """
    return (TRUE_SHANK_PITCH_UM * float(params["probe_width_scaling"])
            * (1.0 - float(params["brain_shrinkage_pct"]) / 100.0))


def trajectory_deviation(v_axis, prior: dict | None = None) -> float:
    """Angle in degrees between ``v_axis`` and the prior's nominal trajectory."""
    prior = prior or SURGICAL_PRIOR
    target = v_axis_from_angles(prior["lateral_deg"], prior["ap_deg"])
    v = np.asarray(v_axis, float)
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(v @ target, -1.0, 1.0))))


def build_plane(v_axis, surface_coord, *, centroid=None, reference_u=None,
                contact_face_axis: str = CONTACT_FACE_AXIS,
                enforce_contact_face: bool = True) -> dict:
    """Assemble a plane dict from a trajectory and a surface point.

    ``reference_u`` (e.g. the u_axis of an existing fit) is re-orthogonalised
    against the new ``v_axis`` where supplied, which preserves the sign
    convention established by ``pit.fit_plane_to_signal``.  Otherwise u is
    derived from ``contact_face_axis``.
    """
    v = np.asarray(v_axis, float)
    v = v / np.linalg.norm(v)
    normal_hint = np.asarray(pit.AXIS2ATLAS_VECTOR[contact_face_axis], float)

    if reference_u is not None:
        u = np.asarray(reference_u, float)
        u = u - (u @ v) * v          # project out any component along v
        if np.linalg.norm(u) < 1e-8:
            u = None
        else:
            u = u / np.linalg.norm(u)
    else:
        u = None

    if u is None:
        u = np.cross(v, normal_hint)
        if np.linalg.norm(u) < 1e-8:
            raise ValueError("contact_face_axis is parallel to v_axis")
        u = u / np.linalg.norm(u)

    # Normal sign is set INDEPENDENTLY of u, matching upstream.
    #
    # In `pit.fit_plane_to_signal`, u_axis and normal are separate PCA
    # components flipped by separate rules, so both conventions -- u aligned to
    # the atlas axis (which fixes shank 0 = anterior) and normal aligned to
    # contact_face_axis -- can hold at once.  An earlier version here computed
    # `normal = +cross(u, v)` and flipped *both* together, which (a) mirrored
    # the probe whenever the convention was enforced, and (b) made the two
    # conventions mutually exclusive: measured across the cohort the auto fits
    # sit at normal = -cross(u, v) and satisfy both, while every plane built the
    # old way satisfied only one.  Hence the leading minus, and the flip below
    # touching `normal` alone.
    normal = -np.cross(u, v)
    normal = normal / np.linalg.norm(normal)
    if enforce_contact_face and normal @ normal_hint < 0:
        normal = -normal

    plane = {
        "centroid": np.asarray(centroid if centroid is not None else surface_coord, float),
        "v_axis": v,
        "u_axis": u,
        "normal": normal,
        "surface_coord": np.asarray(surface_coord, float),
    }
    return plane


def shank_order_axis(contact_face_axis: str = CONTACT_FACE_AXIS,
                     insertion_axis: str = INSERTION_AXIS) -> np.ndarray:
    """The atlas axis ``u_axis`` must align with, per brainreg_probe's convention.

    Reproduces the sign rule in ``pit.fit_plane_to_signal``.  With probeinterface
    placing shank 0 at x = 0-32 and shank 3 at x = 750-782, and
    ``position = surface + (y/10)*v_axis + (x/10)*u_axis``, aligning ``u_axis``
    here fixes **which physical end carries shank 0**.  For this experiment
    (``insertion_axis='si'``, ``contact_face_axis='rl'`` -- the same pair as
    ProbeB/HC in the upstream README) it resolves to ``'ap'`` = +i, so
    **shank 0 is anterior-most**.

    This is a *convention*, not a measurement: it inherits `contact_face_axis`,
    which upstream documents as a guess.  It therefore pins the cohort's
    *relative* consistency; absolute A/P identity needs the surgical record.
    """
    A = pit.AXIS2ATLAS_VECTOR
    candidates = [k for k in A if k[0] not in (insertion_axis + contact_face_axis)]
    one_misaligned = sum(A[insertion_axis] + A[contact_face_axis]) == 0
    return np.asarray(A[candidates[1] if one_misaligned else candidates[0]], float)


def enforce_shank_order(plane: dict, params: dict, *,
                        contact_face_axis: str = CONTACT_FACE_AXIS,
                        insertion_axis: str = INSERTION_AXIS) -> tuple[dict, dict, bool]:
    """Make a fit obey the shank-order convention **without moving any contact**.

    If ``u_axis`` points the wrong way the probe is mirrored, which relabels the
    shanks.  The mirror is only position-preserving as a *triple* --
    ``u_axis -> -u_axis``, ``offset_x -> -offset_x``, ``theta -> -theta``
    (verified to 0.0 um and 0.000% cost across the cohort) -- so all three must
    move together.  Flipping ``u_axis`` alone would displace every contact by up
    to the 750 um shank span.

    Returns ``(plane, params, flipped)``.
    """
    target = shank_order_axis(contact_face_axis, insertion_axis)
    u = np.asarray(plane["u_axis"], float)
    if u @ target >= 0:
        return dict(plane), dict(params), False
    plane = dict(plane)
    params = dict(params)
    plane["u_axis"] = -u
    plane["normal"] = -np.asarray(plane["normal"], float)
    params["offset_x"] = -float(params["offset_x"])
    params["theta"] = -float(params["theta"])
    return plane, params, True


def plane_from_track(entry, tip, *, contact_face_axis: str = CONTACT_FACE_AXIS) -> tuple[dict, float]:
    """Build a plane and depth from a hand-annotated entry point and tip.

    ``entry`` and ``tip`` are (i, j, k) voxel coordinates in sample space.
    Returns ``(plane, probe_depth_um)``.
    """
    entry = np.asarray(entry, float)
    tip = np.asarray(tip, float)
    delta = entry - tip
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        raise ValueError("entry and tip are identical -- cannot define a track")
    v = delta / length
    plane = build_plane(v, entry, centroid=(entry + tip) / 2.0,
                        contact_face_axis=contact_face_axis)
    return plane, length * VOXEL_SIZE_UM


def validate_track(entry, tip, volume_shape) -> None:
    """Mechanical validity only -- typos, not anatomy judgements.

    Raises on: degenerate entry==tip, or coordinates outside the volume.
    Deliberately does *not* judge trajectory, depth or region.
    """
    entry = np.asarray(entry, float)
    tip = np.asarray(tip, float)
    if entry.shape != (3,) or tip.shape != (3,):
        raise ValueError("entry and tip must each be 3 coordinates (i, j, k)")
    if np.linalg.norm(entry - tip) < 1e-6:
        raise ValueError("entry and tip are identical -- cannot define a track")
    for name, pt in (("entry", entry), ("tip", tip)):
        if np.any(pt < 0) or np.any(pt >= np.asarray(volume_shape)):
            raise ValueError(
                f"{name} {pt.tolist()} is outside the volume {tuple(volume_shape)}"
            )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def clear_volume_cache(subject: str | None = None) -> None:
    """Drop cached volumes (~3.5 GB each) -- signal point clouds are kept."""
    if subject is None:
        _VOLUME_CACHE.clear()
    else:
        for key in [k for k in _VOLUME_CACHE if k[0] == subject]:
            del _VOLUME_CACHE[key]


def subject_dir(subject: str) -> Path:
    return BRAINREG_DIR / subject


def list_subjects() -> list[str]:
    return sorted(p.name for p in BRAINREG_DIR.iterdir()
                  if p.is_dir() and (p / ATLAS_NAME).is_dir())


def load_volumes(subject: str, *, fast: bool = True) -> dict:
    """Load the brainreg volumes for a subject, cached across calls.

    ``fast=True`` skips the three deformation fields (~7 GB), which are only
    needed to express coordinates in Allen space for plotting.  Structure
    labels come from ``registered_atlas`` in sample space and are unaffected.
    """
    key = (subject, bool(fast))
    full_key = (subject, False)
    if full_key in _VOLUME_CACHE:
        return _VOLUME_CACHE[full_key]
    if key in _VOLUME_CACHE:
        return _VOLUME_CACHE[key]

    import tifffile

    atlas_path = subject_dir(subject) / ATLAS_NAME
    data = {
        "signal_data": tifffile.imread(atlas_path / "downsampled_2.tiff"),
        "atlas_registration_data": tifffile.imread(atlas_path / "registered_atlas.tiff"),
        "boundaries": tifffile.imread(atlas_path / "boundaries.tiff"),
    }
    if not fast:
        for i in range(3):
            data[f"deformation_field_{i}"] = tifffile.imread(
                atlas_path / f"deformation_field_{i}.tiff"
            )
    _VOLUME_CACHE[key] = data
    return data


def load_signal_df(subject: str, *, gamma: float = 1.5, eps: float = 25.0,
                   min_samples: int = 20, crop: dict | None = None,
                   near: np.ndarray | None = None) -> pd.DataFrame:
    """Threshold and cluster the DiI signal, returning the track cluster.

    ``crop`` optionally restricts the point cloud before clustering, e.g.
    ``{'i': (400, 900)}``, so an artefact cannot drag the plane fit.  ``near``
    selects the cluster closest to a reference point instead of the largest.

    Deviates deliberately from ``pit.cluster_signal``: that function auto-tunes
    ``eps`` to yield exactly ``n_clusters`` and then takes label 0, which for a
    single sparse track can return a small noise blob far from the probe (it
    picked a 167-point cluster 3.8 mm off the track for ly05, and mislabelled
    ly06's perfectly good track).  Here ``eps`` is fixed and the **largest**
    cluster wins, which is both faster and far more stable.  Use
    :func:`signal_diagnostics` to choose per-subject parameters.
    """
    key = (subject, gamma, eps, min_samples,
           json.dumps(crop, sort_keys=True) if crop else None,
           None if near is None else tuple(np.asarray(near, float).round(2)))
    if key in _SIGNAL_CACHE:
        return _SIGNAL_CACHE[key].copy()

    from sklearn.cluster import DBSCAN

    data = load_volumes(subject)
    mask = pit.threshold_signal_gamma(data["signal_data"], gamma=gamma)
    signal_df = pit.make_signal_df(data["signal_data"], mask)
    if crop:
        for axis, (lo, hi) in crop.items():
            signal_df = signal_df[(signal_df[axis] >= lo) & (signal_df[axis] <= hi)]
        signal_df = signal_df.reset_index(drop=True)
    if not len(signal_df):
        raise ValueError(f"{subject}: no signal voxels survived thresholding")

    coords = signal_df[["i", "j", "k"]].values
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    signal_df["cluster"] = labels
    valid = [c for c in np.unique(labels) if c >= 0]
    if not valid:
        raise ValueError(f"{subject}: DBSCAN found no clusters (try larger eps)")

    if near is not None:
        near = np.asarray(near, float)
        pick = min(valid, key=lambda c: np.linalg.norm(coords[labels == c].mean(0) - near))
    else:
        pick = max(valid, key=lambda c: int((labels == c).sum()))

    signal_df = signal_df[signal_df["cluster"] == pick].reset_index(drop=True)
    _SIGNAL_CACHE[key] = signal_df
    return signal_df.copy()


def signal_diagnostics(subject: str, *, gammas=(1.5, 2.0, 2.5),
                       eps: float = 25.0, min_samples: int = 20) -> pd.DataFrame:
    """Survey DiI quality so per-subject extraction parameters can be chosen.

    ``extent_um`` is how far the dye runs along the current fitted trajectory --
    directly comparable to ``probe_depth``, and the cleanest evidence of a
    truncated fit (ah10: 3654 um of dye against a 2205 um fit).  ``dist_um`` is
    how far the selected cluster sits from the current fit's centroid; a large
    value means the dye that was found is not the track.
    """
    fit = load_fit(subject)
    v = np.asarray(fit["v_axis"], float)
    centroid = np.asarray(fit["centroid"], float)
    rows = []
    for gamma in gammas:
        try:
            sig = load_signal_df(subject, gamma=gamma, eps=eps, min_samples=min_samples)
            pts = sig[["i", "j", "k"]].values
            rows.append({
                "subject": subject, "gamma": gamma, "n_points": len(pts),
                "extent_um": float(np.ptp(pts @ v) * VOXEL_SIZE_UM),
                "dist_to_centroid_um": float(np.linalg.norm(pts.mean(0) - centroid) * VOXEL_SIZE_UM),
                "fitted_depth_um": float(fit["probe_depth"]),
            })
        except ValueError as exc:
            rows.append({"subject": subject, "gamma": gamma, "n_points": 0,
                         "extent_um": np.nan, "dist_to_centroid_um": np.nan,
                         "fitted_depth_um": float(fit["probe_depth"]),
                         "error": str(exc)})
    return pd.DataFrame(rows)


def load_fit(subject: str, *, which: str = "current") -> dict:
    """Load saved fit parameters.  ``which`` is 'current' or 'auto'."""
    suffix = "_auto" if which == "auto" else ""
    path = subject_dir(subject) / f"ProbeA_fit_params{suffix}.json"
    if which == "auto" and not path.exists():
        path = subject_dir(subject) / "ProbeA_fit_params.json"
    with open(path) as f:
        params = json.load(f)
    return params


def split_fit(fit: dict) -> tuple[dict, dict]:
    """Split a saved fit dict into (plane, placement params)."""
    plane = {k: np.asarray(fit[k], float)
             for k in ("centroid", "v_axis", "u_axis", "normal", "surface_coord")
             if k in fit}
    params = {k: float(fit[k]) for k in PARAM_ORDER if k in fit}
    return plane, params


# --------------------------------------------------------------------------
# The re-projection core
# --------------------------------------------------------------------------


_SURFACE_CACHE: dict = {}
_ATLAS_MAPS: dict = {}


def _atlas_maps() -> dict:
    """``structure id -> (acronym, name, id)`` for the whole Allen table.

    A **dict, not a dense array indexed by id**.  Allen ids run up to
    614,454,277, so indexing an array by id would allocate three
    614-million-entry object arrays -- 14.7 GB -- and get the process
    OOM-killed.  (That is exactly what the first version of this did; the
    benchmark missed it because the allocation sat outside the timer.)  The
    table has 1327 rows and a probe touches ~26 distinct ids, so a dict plus
    ``np.unique`` is both tiny and fast.
    """
    if _ATLAS_MAPS:
        return _ATLAS_MAPS
    info = pit.ALLEN_ATLAS_INFO_DF
    for i, a, n in zip(info["id"].values, info["acronym"].values,
                       info["name"].values):
        _ATLAS_MAPS[int(i)] = (a, n, i)
    return _ATLAS_MAPS


def fast_structure_labels(points: np.ndarray, data: dict) -> dict:
    """Vectorised drop-in for ``pit.get_structure_labels``.

    Upstream loops in Python running a DataFrame ``.query()`` per contact:
    ~4400 ms for 2016 contacts.  Here the volume is indexed with numpy and only
    the handful of *distinct* ids present are mapped, then expanded back -- so
    the cost scales with unique structures (~26), not contacts.  Output is
    bit-identical, including the ``except``-branch defaults for ids absent from
    the table (acronym NaN, name "outside brain", id NaN).

    Callers must clip ``points`` into the volume first; see `project_probe`,
    which also marks out-of-volume contacts explicitly.
    """
    atlas = data["atlas_registration_data"]
    ic = np.asarray(points).astype(int)
    vol = atlas[ic[:, 0], ic[:, 1], ic[:, 2]]
    maps = _atlas_maps()
    uniq, inv = np.unique(vol, return_inverse=True)
    u_acro = np.empty(len(uniq), dtype=object)
    u_name = np.empty(len(uniq), dtype=object)
    u_id = np.empty(len(uniq), dtype=object)
    for j, v in enumerate(uniq):
        a, n, i = maps.get(int(v), (np.nan, "outside brain", np.nan))
        u_acro[j], u_name[j], u_id[j] = a, n, i
    return {"name": list(u_name[inv]),
            "acronym": list(u_acro[inv]),
            "id": list(u_id[inv])}


def project_probe(plane: dict, params: dict, data: dict, *,
                  with_allen: bool = False) -> pd.DataFrame:
    """Place the probe and label every contact -- steps 6, 7, 11, 12 upstream.

    This is the single path every tier goes through.
    """
    probe_df = pit.get_probe_contacts_df(PROBE_MANUFACTURER, PROBE_NAME)
    values = [params[k] for k in PARAM_ORDER]
    transformed = pit.transform_2d_probe(probe_df, values)
    downsampled = pit.project_2d_points_to_plane(transformed, plane)

    # Select the surviving contacts by the transformed frame's own index rather
    # than re-applying a depth mask.  ``transform_2d_probe`` filters with ``<=``
    # while upstream's caller re-filters with ``<``; when a depth lands exactly
    # on a contact row (round numbers from a hand annotation do this readily)
    # the two disagree and the coordinate array no longer matches the frame --
    # "Length of values (1608) does not match length of index (1600)".  Indexing
    # off the transform makes divergence impossible.
    out = probe_df.loc[transformed.index].copy()
    coords = downsampled.values
    assert len(out) == len(coords), (
        f"contact/coordinate mismatch: {len(out)} vs {len(coords)}")
    for idx, coord in enumerate("ijk"):
        out[f"downsample_coords.{coord}"] = coords[:, idx]

    # Guard the atlas lookup.  ``pit.get_structure_labels`` indexes the volume
    # *outside* its try/except, so a fit that places contacts beyond the volume
    # raises IndexError instead of reporting (this is what a bad fit does --
    # every ly05 re-fit hit it).  Negative indices are worse: numpy wraps them
    # silently and would mislabel a contact with anatomy from the far side of
    # the brain.  So clip for the lookup and mark those contacts explicitly.
    shape = np.asarray(data["atlas_registration_data"].shape)
    inside = np.all((coords >= 0) & (coords < shape - 1), axis=1)
    safe = np.clip(coords, 0, shape - 1)

    anatomy = fast_structure_labels(safe, data)
    for label, values_ in anatomy.items():
        out[f"structure.{label}"] = values_
    if not inside.all():
        out.loc[~inside, "structure.name"] = "outside volume"
        out.loc[~inside, "structure.acronym"] = np.nan
        out.loc[~inside, "structure.id"] = np.nan
    out["inside_volume"] = inside

    if with_allen:
        atlas_coords = pit.sample_coords_to_allen_space(downsampled.values, data)
        for idx, coord in enumerate("ijk"):
            out[f"allen_atlas_coords.{coord}"] = atlas_coords[:, idx]

    out["probe_name"] = "ProbeA"
    return out.reset_index(drop=True)


def _cost(signal_df: pd.DataFrame, plane: dict, params: dict,
          objective: str = "both") -> float:
    """Fit cost.  ``objective`` selects which half of the upstream cost to use.

    ``'both'`` reproduces `pit.compute_cost`:
        mean(contact -> nearest dye) + weighted mean(dye -> nearest contact)

    **The first term biases the fit shallow.** It penalises every contact that
    extends past the visible dye, and dye is deposited on the way in, so it
    generally fades before the tip -- the optimiser is then rewarded for
    stopping short.  ``'signal2contact'`` keeps only the second term, which asks
    "is every bit of dye explained by the probe" and is indifferent to a probe
    that continues beyond the dye.  Use it when depth is the quantity of
    interest; see `depth_objective_comparison`.
    """
    probe_df = pit.get_probe_contacts_df(PROBE_MANUFACTURER, PROBE_NAME)
    if objective == "both":
        return pit.compute_cost(signal_df, probe_df, plane,
                                [params[k] for k in PARAM_ORDER])

    from scipy.spatial import KDTree
    transformed = pit.transform_2d_probe(probe_df, [params[k] for k in PARAM_ORDER])
    coords = pit.project_2d_points_to_plane(transformed, plane).values
    sig = signal_df[["i", "j", "k"]].values
    s2c, _ = KDTree(coords).query(sig, k=1)
    if objective == "signal2contact":
        return float(np.sum(s2c * signal_df["norm_value"].values) / len(signal_df))
    raise ValueError(f"unknown objective {objective!r}")


def refit_probe(subject: str, *,
                plane: dict | None = None,
                params: dict | None = None,
                signal_df: pd.DataFrame | None = None,
                fixed_params: dict | None = None,
                bounds: dict | None = None,
                trajectory_prior: dict | None = None,
                optimize: bool = True,
                fresh_plane: bool = False,
                reset_depth: bool = False,
                objective: str = "both",
                fast: bool = True,
                data: dict | None = None) -> dict:
    """Re-place the probe for one subject.

    The tiers differ only in what they pass in:

    * ``override``               - ``params=...``, ``optimize=False``.
    * ``trajectory_constrained`` - ``trajectory_prior=...``, ``optimize=True``;
      the plane's tilt joins the optimisation, bounded to the prior.
    * ``manual_track``           - ``plane=`` from :func:`plane_from_track`,
      typically with ``probe_depth`` fixed to the annotated length.
    """
    data = data if data is not None else load_volumes(subject, fast=fast)
    saved = load_fit(subject)
    saved_plane, saved_params = split_fit(saved)

    plane = dict(saved_plane) if plane is None else dict(plane)
    params = dict(saved_params) if params is None else {**saved_params, **params}
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    fixed_params = dict(fixed_params or {})

    if (fresh_plane or reset_depth) and signal_df is None:
        signal_df = load_signal_df(subject)
    if fresh_plane:
        plane = fit_plane_from_signal(signal_df, data, subject=subject)
    if reset_depth:
        params["probe_depth"] = float(np.clip(
            depth_from_signal(signal_df, plane),
            *bounds["probe_depth"]))
    params.update(fixed_params)

    if optimize:
        if signal_df is None:
            signal_df = load_signal_df(subject)
        params, plane = _optimize(signal_df, plane, params, bounds,
                                  fixed_params, trajectory_prior, objective)

    probe_df = project_probe(plane, params, data, with_allen=not fast)
    qc = qc_probe_fit(subject, plane, params, probe_df,
                      signal_df=signal_df, trajectory_prior=trajectory_prior,
                      bounds=bounds)
    return {"probe_df": probe_df, "plane": plane, "params": params,
            "qc": qc, "signal_df": signal_df}


def _optimize(signal_df, plane, params, bounds, fixed_params, trajectory_prior,
              objective: str = "both"):
    """Optimise placement, optionally including the plane's tilt.

    When ``trajectory_prior`` is given, ``lateral_deg`` and ``ap_deg`` become
    free parameters bounded to prior +- tol.  The objective is unchanged --
    distance between contacts and dye -- so the dye still drives the fit and the
    prior only stops the optimiser wandering into anatomically impossible
    trajectories (ah10's 25 deg AP tilt).
    """
    from scipy.optimize import minimize

    probe_df = pit.get_probe_contacts_df(PROBE_MANUFACTURER, PROBE_NAME)
    free = [k for k in PARAM_ORDER if k not in fixed_params]

    ref_u = plane.get("u_axis")
    surface = plane["surface_coord"]
    centroid = plane.get("centroid")

    if trajectory_prior is not None:
        prior = {**SURGICAL_PRIOR, **trajectory_prior}
        tol = prior["tol_deg"]
        angle_names = ["lateral_deg", "ap_deg"]
        start_angles = trajectory_angles(plane["v_axis"])
        x0_angles = [
            float(np.clip(start_angles["lateral_deg"],
                          prior["lateral_deg"] - tol, prior["lateral_deg"] + tol)),
            float(np.clip(start_angles["ap_deg"],
                          prior["ap_deg"] - tol, prior["ap_deg"] + tol)),
        ]
        angle_bounds = [(prior["lateral_deg"] - tol, prior["lateral_deg"] + tol),
                        (prior["ap_deg"] - tol, prior["ap_deg"] + tol)]
    else:
        angle_names, x0_angles, angle_bounds = [], [], []

    def unpack(x):
        vals = dict(params)
        for name, value in zip(free, x[:len(free)]):
            vals[name] = float(value)
        if angle_names:
            lat, ap = x[len(free):]
            this_plane = build_plane(v_axis_from_angles(lat, ap), surface,
                                     centroid=centroid, reference_u=ref_u)
        else:
            this_plane = plane
        return vals, this_plane

    def cost_fn(x):
        vals, this_plane = unpack(x)
        return _cost(signal_df, this_plane, vals, objective)

    x0 = [params[k] for k in free] + x0_angles
    bnds = [bounds[k] for k in free] + angle_bounds
    res = minimize(cost_fn, x0=x0, bounds=bnds,
                   options={"maxiter": 1000, "maxls": 1000})
    return unpack(res.x)


# --------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------


def qc_probe_fit(subject: str, plane: dict, params: dict,
                 probe_df: pd.DataFrame, *,
                 signal_df: pd.DataFrame | None = None,
                 trajectory_prior: dict | None = None,
                 bounds: dict | None = None) -> dict:
    """Descriptive statistics for a fit, plus advisory flags.

    **Who these judge.** They are built to catch an *algorithm* fitting the
    wrong thing and carry that authority only over ``auto`` / ``override`` /
    ``trajectory_constrained`` fits.  Against a ``manual_track`` annotation they
    are descriptive only: ``resid_*`` and ``signal_coverage`` are measured
    against the Otsu-thresholded DiI mask, and a poor threshold is exactly why
    one annotates by hand -- scoring an annotation against that mask scores it
    against the thing it was meant to replace.  ``flags`` is therefore reported
    for information; :func:`grade_fit` decides whether it means anything.
    """
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    angles = trajectory_angles(plane["v_axis"])
    qc = dict(angles)
    qc["subject"] = subject
    qc["theta_deg"] = float(np.degrees(params["theta"]))
    qc["probe_depth"] = float(params["probe_depth"])
    qc["trajectory_dev_deg"] = trajectory_deviation(plane["v_axis"], trajectory_prior)

    at_bounds = []
    for name in PARAM_ORDER:
        lo, hi = bounds[name]
        span = hi - lo
        if span <= 0:
            continue
        if abs(params[name] - lo) < 0.01 * span or abs(params[name] - hi) < 0.01 * span:
            at_bounds.append(name)
    qc["params_at_bounds"] = at_bounds

    # Where the recorded bank -- the only part that produces data -- ends up.
    bank = probe_df[probe_df["probe_coords.y"] <= RECORDED_BANK_MAX_UM]
    acro = bank["structure.acronym"].dropna()
    qc["n_bank_contacts"] = int(len(bank))
    qc["bank_structures"] = acro.value_counts().head(8).to_dict()
    qc["target_fraction"] = float(
        acro.str.startswith(TARGET_REGION).mean()) if len(acro) else float("nan")
    if "inside_volume" in probe_df.columns:
        qc["frac_outside_volume"] = float(1.0 - probe_df["inside_volume"].mean())
    else:
        qc["frac_outside_volume"] = 0.0
    tip = probe_df.nsmallest(16, "probe_coords.y")["structure.acronym"].dropna()
    qc["tip_structure"] = tip.mode().iat[0] if len(tip) else None

    if signal_df is not None and len(signal_df):
        from scipy.spatial import KDTree
        coords = probe_df[[f"downsample_coords.{c}" for c in "ijk"]].values
        sig = signal_df[["i", "j", "k"]].values
        c2s, _ = KDTree(sig).query(coords, k=1)
        s2c, _ = KDTree(coords).query(sig, k=1)
        qc["resid_contact2signal_um"] = float(np.mean(c2s) * VOXEL_SIZE_UM)
        qc["resid_signal2contact_um"] = float(np.mean(s2c) * VOXEL_SIZE_UM)
        # How much of the dye's extent along the track the probe actually spans.
        v = plane["v_axis"]
        sig_proj = sig @ v
        probe_proj = coords @ v
        sig_span = np.percentile(sig_proj, 99.5) - np.percentile(sig_proj, 0.5)
        overlap = (min(probe_proj.max(), sig_proj.max())
                   - max(probe_proj.min(), sig_proj.min()))
        qc["signal_coverage"] = float(np.clip(overlap / sig_span, 0, 1)) if sig_span > 0 else float("nan")
        # How the fitted length compares with the dye's own extent.  A fit
        # shorter than its dye is under-reading depth; a fit much longer is
        # extrapolating past the evidence.  Both are reportable, neither is
        # automatically wrong -- dye fades before the tip.
        #
        # Use a ROBUST extent (1-99 pct), not the full range.  Diffuse dye
        # leaves a sparse wisp past the real track end: for ah10 the full range
        # is 3774 um but the deepest 200 um bin holds 23 points against ~1900 in
        # the bulk, and anchoring depth to that tail overshot the probe by ~350
        # um -- pushing the recorded bank out of deep ENTl and into layer 2.
        # `dye_extent_full_um` is kept so the tail is still visible.
        qc["dye_extent_um"] = float(
            (np.percentile(sig_proj, 99) - np.percentile(sig_proj, 1)) * VOXEL_SIZE_UM)
        qc["dye_extent_full_um"] = float(np.ptp(sig_proj) * VOXEL_SIZE_UM)
        qc["dye_tail_ratio"] = (qc["dye_extent_full_um"] / qc["dye_extent_um"]
                                if qc["dye_extent_um"] > 0 else float("nan"))
        qc["depth_over_dye"] = (float(params["probe_depth"]) / qc["dye_extent_um"]
                                if qc["dye_extent_um"] > 0 else float("nan"))
        # Does the fitted trajectory agree with the dye's OWN principal axis?
        # This is the honest version of "is the plane wrong": it compares the fit
        # against the data rather than against the nominal surgical angle.  It
        # separates the two real cases cleanly -- ah10's automated fit sat ~19
        # deg off its own dye (genuinely mis-fitted), while ly07 sits 1.7 deg off
        # its dye despite a 9 deg AP tilt, i.e. faithfully tracking an AP-tilted
        # physical track.  Judging ly07 against the prior would have condemned it
        # for being right.
        pc1, expl = dye_principal_axis(signal_df)
        qc["dye_pc1_expl_var"] = expl
        if np.isfinite(expl):
            qc["fit_vs_dye_axis_deg"] = float(np.degrees(np.arccos(
                np.clip(abs(np.asarray(plane["v_axis"], float) @ pc1), -1.0, 1.0))))
            ang = trajectory_angles(pc1)
            qc["dye_lateral_deg"] = ang["lateral_deg"]
            qc["dye_ap_deg"] = ang["ap_deg"]
        else:
            qc["fit_vs_dye_axis_deg"] = float("nan")
            qc["dye_lateral_deg"] = float("nan")
            qc["dye_ap_deg"] = float("nan")
    else:
        qc["resid_contact2signal_um"] = float("nan")
        qc["resid_signal2contact_um"] = float("nan")
        qc["signal_coverage"] = float("nan")
        qc["dye_extent_um"] = float("nan")
        qc["dye_extent_full_um"] = float("nan")
        qc["dye_tail_ratio"] = float("nan")
        qc["depth_over_dye"] = float("nan")
        qc["dye_pc1_expl_var"] = float("nan")
        qc["fit_vs_dye_axis_deg"] = float("nan")
        qc["dye_lateral_deg"] = float("nan")
        qc["dye_ap_deg"] = float("nan")

    # Hard flags: the fit disagrees with the DYE, or is geometrically
    # implausible.  These decide the grade.
    flags = []
    if abs(qc["theta_deg"]) > QC_THRESHOLDS["theta_deg"]:
        flags.append("large_theta(plane_likely_wrong)")
    # The fit disagrees with the dye's OWN axis -- the honest "plane is wrong"
    # test, since it compares the fit against the data rather than the nominal
    # surgical angle.  Only meaningful when the cloud is elongated enough for a
    # principal axis to mean anything.
    fvd = qc.get("fit_vs_dye_axis_deg", float("nan"))
    expl = qc.get("dye_pc1_expl_var", float("nan"))
    if np.isfinite(fvd) and np.isfinite(expl) and expl >= 0.5:
        if fvd > QC_THRESHOLDS["fit_vs_dye_axis_deg"]:
            flags.append(f"fit_disagrees_with_dye_axis({fvd:.0f}°)")
    if at_bounds:
        flags.append("params_at_bounds:" + ",".join(at_bounds))
    if np.isfinite(qc["signal_coverage"]) and qc["signal_coverage"] < QC_THRESHOLDS["signal_coverage"]:
        flags.append("low_signal_coverage")
    if np.isfinite(qc["resid_signal2contact_um"]) and \
            qc["resid_signal2contact_um"] > QC_THRESHOLDS["resid_um"]:
        flags.append("high_residual")
    if qc.get("frac_outside_volume", 0.0) > 0.02:
        flags.append(f"outside_volume:{qc['frac_outside_volume']:.0%}")

    # Advisory: the trajectory differs from the *nominal* surgical angle.  This
    # does not decide the grade, because the dye is the ground truth and the 10
    # deg figure is nominal -- stereotax misalignment genuinely varies.  A fit
    # that follows the dye tightly has earned its trajectory whatever the
    # nominal was (ah08 sits 13.3 deg off the prior with the best dye agreement
    # in the cohort, 29 um).  It remains diagnostic *in combination*: paired
    # with a large theta or a wrong-plane tilt it is the ah10 signature.
    advisories = []
    if qc["trajectory_dev_deg"] > QC_THRESHOLDS["trajectory_dev_deg"]:
        advisories.append(
            f"trajectory_off_prior({qc['trajectory_dev_deg']:.0f}°)")
    # AP tilt beyond the prior is ADVISORY, for the same reason
    # `trajectory_off_prior` is: the dye is ground truth and the 0-degree AP is
    # nominal.  ly07 tilts 10.6 deg in AP but its dye tilts 9.1 deg the same way
    # -- the fit is right and the prior is simply not met.  As a hard flag this
    # would condemn a fit for faithfully tracking its own dye.
    _prior = {**SURGICAL_PRIOR, **(trajectory_prior or {})}
    if abs(qc["ap_deg"] - _prior["ap_deg"]) > _prior["tol_deg"]:
        advisories.append(f"ap_tilt_beyond_prior({qc['ap_deg']:.0f}°)")
    if np.isfinite(expl) and expl < 0.5:
        advisories.append(f"dye_axis_undefined(expl_var {expl:.2f})")
    # Depth vs the dye's own extent.  Advisory, because depth is frequently
    # *not identifiable* from dye at all: when dye fades before the tip the
    # information is simply absent, and measured on synthetics both objectives
    # then under-read by roughly the missing fraction (up to -2.3 mm at 40%
    # dye).  So a short fit is a prompt to look, and only an annotation or the
    # surgical record can settle it.
    dod = qc.get("depth_over_dye", float("nan"))
    if np.isfinite(dod):
        if dod < 0.9:
            advisories.append(f"fit_stops_short_of_dye({dod:.2f}x)")
        elif dod > 1.3:
            advisories.append(f"depth_extrapolated_past_dye({dod:.2f}x)")
    qc["flags"] = flags
    qc["advisories"] = advisories
    return qc


#: NP2020 shank x-positions: each shank contributes two columns 32 um apart.
SHANK_X_PITCH = 250.0

#: Whether the shank -> channel mapping has been pinned by external knowledge.
#:
#: **The fit cannot determine it.** Mirroring the probe -- u_axis -> -u_axis,
#: offset_x -> -offset_x, theta -> -theta -- is an *exact* degeneracy: measured
#: across ah08/ah10/ly06/ly07 it reproduces the contact positions to 0.0 um and
#: the cost to 0.000%, because the four shanks are geometrically identical and
#: their centred x-set equals its own mirror.  So no dye-based method can ever
#: say which physical end carries channels 0-95; only the surgical record
#: (which way the contact face pointed) can.
#:
#: What this does *not* affect: the structure at a given physical position, which
#: is invariant under the mirror (verified -- identical per-position anatomy for
#: ly07 either way).  Per-shank anatomy reported **by AP position** is therefore
#: already correct and orientation-independent; only the shank *number* attached
#: to it is pending.  Set to True and record `shank_order` once known.
SHANK_ORDER_VERIFIED = False


def shank_id(x) -> np.ndarray:
    """Map probe x-coordinates to shank index 0-3."""
    return np.round(np.asarray(x, float) / SHANK_X_PITCH).astype(int)


def shank_breakdown(result: dict, *, bank_only: bool = True,
                    top: int = 4) -> pd.DataFrame:
    """Structure composition **per shank**, with each shank's AP position.

    A pooled ``tip_structure`` or bank census hides the case where the shanks
    straddle a boundary -- the four shanks span 750 um, which at this location
    is easily the width of a region border.  Reporting per shank is the only
    honest summary when they disagree.
    """
    df = result["probe_df"]
    if bank_only:
        df = df[df["probe_coords.y"] <= RECORDED_BANK_MAX_UM]
    df = df.assign(shank=shank_id(df["probe_coords.x"]))

    rows = []
    for shank, g in df.groupby("shank"):
        counts = g["structure.acronym"].fillna("none").value_counts()
        deepest = g.nsmallest(8, "probe_coords.y")["structure.acronym"].dropna()
        rows.append({
            "shank": int(shank),
            "ap_i": float(g["downsample_coords.i"].mean()),   # lower i = anterior
            "n": int(len(g)),
            "tip": deepest.mode().iat[0] if len(deepest) else None,
            "ENTl_frac": float(g["structure.acronym"].fillna("")
                               .str.startswith(TARGET_REGION).mean()),
            **{f"top{i+1}": f"{a} {c}" for i, (a, c) in
               enumerate(counts.head(top).items())},
        })
    # Order by physical AP position, which is orientation-independent and
    # therefore trustworthy; the shank *number* is not, until
    # SHANK_ORDER_VERIFIED -- see that constant.
    out = pd.DataFrame(rows).sort_values("ap_i").reset_index(drop=True)
    labels = ["most anterior"] + ["…"] * max(len(out) - 2, 0) + ["most posterior"]
    out.insert(1, "ap_rank", labels[:len(out)] if len(out) > 1 else [""])
    out.insert(2, "shank_label",
               out["shank"].map(lambda s: f"{s}" if SHANK_ORDER_VERIFIED
                                else f"{s}?"))
    return out


#: Fit methods produced by a human placing the probe.  These are graded
#: ``annotated`` and are never failed by the harness.
MANUAL_FIT_METHODS = frozenset({"manual_track", "manual_3d"})

#: Every fit method the harness knows how to grade.
KNOWN_FIT_METHODS = frozenset(
    {"auto", "override", "trajectory_constrained"}) | MANUAL_FIT_METHODS


def grade_fit(qc: dict, fit_method: str) -> str:
    """Turn QC into a verdict, respecting who produced the fit.

    A hand placement is always ``annotated`` -- the harness reports its metrics
    but does not sit in judgement over a human annotation.  Reliability for
    those is declared by the annotator via the ``confidence`` field.

    Membership is checked against :data:`MANUAL_FIT_METHODS` rather than a
    single hard-coded name.  The first version tested ``== "manual_track"``
    only, so when `probe_tool` introduced ``manual_3d`` the very first hand
    placement fell through to the automated branch and was graded ``review``
    on `params_at_bounds` -- flagging a human for choosing slider values, which
    is precisely what this function exists to prevent.  An unknown method is
    reported rather than silently graded.
    """
    if fit_method in MANUAL_FIT_METHODS:
        return "annotated"
    if fit_method not in KNOWN_FIT_METHODS:
        return f"ungraded(unknown method {fit_method!r})"
    return "review" if qc.get("flags") else "ok"


# --------------------------------------------------------------------------
# Saving, with provenance
# --------------------------------------------------------------------------


def preserve_auto(subject: str) -> None:
    """Copy the original automated output aside, once and non-destructively."""
    d = subject_dir(subject)
    for stem, ext in (("ProbeA_anatomy", ".htsv"), ("ProbeA_fit_params", ".json")):
        src = d / f"{stem}{ext}"
        dst = d / f"{stem}_auto{ext}"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def save_fit(subject: str, result: dict, *, fit_method: str,
             confidence: str = "unset", note: str = "",
             manual_inputs: dict | None = None) -> dict:
    """Write the corrected anatomy table and fit params with provenance.

    ``confidence`` is set by the annotator, never inferred here.
    """
    if confidence not in {"confident", "uncertain", "unset"}:
        raise ValueError("confidence must be 'confident', 'uncertain' or 'unset'")
    if fit_method not in KNOWN_FIT_METHODS:
        raise ValueError(
            f"unknown fit_method {fit_method!r}; add it to KNOWN_FIT_METHODS "
            f"(and to MANUAL_FIT_METHODS if a human placed it) so it is graded "
            f"correctly. Known: {sorted(KNOWN_FIT_METHODS)}")
    preserve_auto(subject)
    d = subject_dir(subject)

    probe_df = result["probe_df"].copy()
    ordered = [c for c in probe_df.columns if c.startswith("probe_coords")] \
        + [c for c in probe_df.columns if c.startswith("downsample_coords")] \
        + [c for c in probe_df.columns if c.startswith("allen_atlas_coords")] \
        + [c for c in probe_df.columns if c.startswith("structure")] + ["probe_name"]
    probe_df = probe_df[[c for c in ordered if c in probe_df.columns]]
    probe_df.to_csv(d / "ProbeA_anatomy.htsv", sep="\t", index=False)

    payload = {k: float(v) for k, v in result["params"].items()}
    payload["probe_name"] = "ProbeA"
    for key, value in result["plane"].items():
        payload[key] = np.asarray(value, float).tolist()
    payload["fit_method"] = fit_method
    payload["confidence"] = confidence
    payload["note"] = note
    payload["shank_order_verified"] = bool(SHANK_ORDER_VERIFIED)
    payload["corrected_date"] = datetime.now().isoformat(timespec="seconds")
    payload["manual_inputs"] = manual_inputs or {}
    payload["qc"] = _jsonable(result["qc"])
    payload["grade"] = grade_fit(result["qc"], fit_method)
    with open(d / "ProbeA_fit_params.json", "w") as f:
        json.dump(payload, f, indent=4)
    return payload


def mark_provisional(subject: str, note: str, *,
                     confidence: str = "uncertain") -> dict:
    """Annotate a saved fit as provisional without re-running it.

    Used when a fit is good on one axis but explicitly unsettled on another --
    e.g. ah10, whose trajectory is corrected but whose depth is still under
    review -- so the file on disk does not read as finished.
    """
    path = subject_dir(subject) / "ProbeA_fit_params.json"
    with open(path) as f:
        payload = json.load(f)
    payload["confidence"] = confidence
    payload["grade"] = "provisional"
    payload["note"] = (payload.get("note", "") + " || PROVISIONAL: " + note).strip()
    payload["provisional_since"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
    return payload


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# --------------------------------------------------------------------------
# Tier 3: manual track annotation
#
# Authoritative.  The harness reports metrics beside an annotation but never
# overrides it, re-optimises it against the DiI mask, or fails it.  Only
# mechanical input errors raise (see validate_track).
# --------------------------------------------------------------------------


def apply_manual_track(subject: str, entry, tip, *,
                       optimize_placement: bool = True,
                       signal_df: pd.DataFrame | None = None,
                       contact_face_axis: str = CONTACT_FACE_AXIS,
                       params: dict | None = None,
                       fast: bool = True,
                       data: dict | None = None) -> dict:
    """Place the probe from a hand-annotated entry point and tip.

    ``entry`` and ``tip`` are (i, j, k) voxel coordinates in sample space, read
    off :func:`slice_grid` or :func:`annotate_track`.  The trajectory and depth
    they imply are taken as given; with ``optimize_placement`` only ``theta``
    and ``offset_x`` are tuned, and only *within* that trajectory, so the
    annotation is never moved.
    """
    data = data if data is not None else load_volumes(subject, fast=fast)
    validate_track(entry, tip, data["signal_data"].shape)
    plane, depth = plane_from_track(entry, tip, contact_face_axis=contact_face_axis)

    saved = load_fit(subject)
    _, saved_params = split_fit(saved)
    params = {**saved_params, **(params or {})}
    params["probe_depth"] = depth

    if optimize_placement:
        if signal_df is None:
            try:
                signal_df = load_signal_df(subject)
            except ValueError:
                signal_df = None
        if signal_df is not None and len(signal_df):
            fixed = {k: params[k] for k in ("probe_depth", "brain_shrinkage_pct",
                                            "probe_width_scaling")}
            params, plane = _optimize(signal_df, plane, params, DEFAULT_BOUNDS,
                                      fixed, None)

    probe_df = project_probe(plane, params, data, with_allen=not fast)
    qc = qc_probe_fit(subject, plane, params, probe_df, signal_df=signal_df)
    qc["annotated_depth_um"] = depth
    return {"probe_df": probe_df, "plane": plane, "params": params, "qc": qc,
            "signal_df": signal_df,
            "manual_inputs": {"entry": list(map(float, entry)),
                              "tip": list(map(float, tip)),
                              "contact_face_axis": contact_face_axis,
                              "annotated_depth_um": depth}}


def annotation_report(subject: str, result: dict,
                      prior: dict | None = None) -> dict:
    """Advisory comparison of an annotation against the dye and the prior.

    Every field is **information for the annotator**, not a verdict.  A
    disagreement here is a prompt to look again, never grounds for the harness
    to reject or re-fit a hand annotation.
    """
    qc = result["qc"]
    prior = {**SURGICAL_PRIOR, **(prior or {})}
    notes = []
    dev = qc["trajectory_dev_deg"]
    notes.append(
        f"trajectory {qc['lateral_deg']:.1f}° lateral / {qc['ap_deg']:.1f}° AP "
        f"— {dev:.1f}° from the nominal {prior['lateral_deg']:.0f}° lateral approach"
        + (" (within stereotax tolerance)" if dev <= prior["tol_deg"]
           else " (larger than nominal; stereotax misalignment or worth a second look)")
    )
    if result.get("signal_df") is not None and len(result["signal_df"]):
        sig = result["signal_df"][["i", "j", "k"]].values
        v = np.asarray(result["plane"]["v_axis"], float)
        dye_extent = float(np.ptp(sig @ v) * VOXEL_SIZE_UM)
        depth = qc["probe_depth"]
        notes.append(
            f"annotated depth {depth:.0f} µm vs {dye_extent:.0f} µm of thresholded dye "
            f"along the track (ratio {depth / dye_extent:.2f}) — dye can fade before "
            "the track ends, so a ratio above 1 is not itself a problem"
        )
        notes.append(f"mean dye-to-contact distance {qc['resid_signal2contact_um']:.0f} µm")
    notes.append(
        f"recorded bank: tip in {qc['tip_structure']}, "
        f"{qc['target_fraction'] * 100:.0f}% of contacts in {TARGET_REGION}"
    )
    return {"subject": subject, "notes": notes, "qc": qc}


# --------------------------------------------------------------------------
# Plotting
#
# The gap this fills: upstream draws the DiI signal (plot_sample_data_sections)
# and the probe-on-atlas cartoon (plot_atlas_data_sections) in *separate*
# panels, so "the fit does not follow the dye" is structurally invisible.  Here
# the contacts are drawn on top of the signal.
# --------------------------------------------------------------------------


def slice_grid(subject: str, *, axis: str = "k", slices=None, n: int = 9,
               data: dict | None = None, signal_df: pd.DataFrame | None = None,
               show_mask: bool = True, overlay: dict | None = None,
               save: bool | str = False):
    """Static grid of slices through the raw DiI volume, for reading off coords.

    Shown on the **raw** signal, with the thresholded mask only as an optional
    overlay (``show_mask``), so faint dye that Otsu discarded stays visible --
    that faint dye is usually the reason to annotate by hand at all.  Axis
    ticks are voxel indices, so coordinates can be read straight off and passed
    to :func:`apply_manual_track`.

    ``axis='k'`` gives sagittal slices, ``'i'`` coronal, ``'j'`` horizontal.
    """
    import matplotlib.pyplot as plt

    data = data if data is not None else load_volumes(subject)
    signal = data["signal_data"]
    dim = {"i": 0, "j": 1, "k": 2}[axis]
    if slices is None:
        if signal_df is not None and len(signal_df):
            lo, hi = np.percentile(signal_df[axis].values, [2, 98])
        else:
            centre = np.asarray(load_fit(subject)["centroid"], float)[dim]
            lo, hi = centre - 60, centre + 60
        slices = np.linspace(lo, hi, n).round().astype(int)
    slices = [int(s) for s in np.atleast_1d(slices)]

    ncol = min(3, len(slices))
    nrow = int(np.ceil(len(slices) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    for ax, s in zip(axes.ravel(), slices):
        if axis == "k":
            img, xi, yi = signal[:, :, s].T, 0, 1
        elif axis == "i":
            img, xi, yi = signal[s, :, :], 2, 1
        else:
            img, xi, yi = signal[:, s, :].T, 0, 2
        ax.imshow(puf.adjust_contrast(img), cmap="gray", origin="upper")
        if show_mask and signal_df is not None and len(signal_df):
            near = signal_df[np.abs(signal_df[axis] - s) <= 2]
            if len(near):
                ax.scatter(near.iloc[:, xi], near.iloc[:, yi], s=2,
                           c="#00CFFF", alpha=0.4, lw=0)
        if overlay is not None:
            cc = overlay["probe_df"][[f"downsample_coords.{c}" for c in "ijk"]].values
            keep = np.abs(cc[:, dim] - s) <= 3
            ax.scatter(cc[keep, xi], cc[keep, yi], s=3, c="#FF2D2D", alpha=0.7, lw=0)
        ax.set_title(f"{axis}={s}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(slices):]:
        ax.axis("off")
    fig.suptitle(f"{subject} — raw DiI, {axis} slices (ticks are voxel indices)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURE_DIR / (save if isinstance(save, str) else f"{subject}_slices_{axis}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"saved {path}")
    return fig


def annotate_track(subject: str, *, data: dict | None = None,
                   signal_df: pd.DataFrame | None = None):
    """Interactive slice browser for picking the entry point and tip.

    Requires ipywidgets in a notebook.  Click-free by design: scrub the slider,
    read the voxel coordinates off the axes, then call
    :func:`apply_manual_track`.  :func:`slice_grid` is the static fallback.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    data = data if data is not None else load_volumes(subject)
    signal = data["signal_data"]
    if signal_df is None:
        try:
            signal_df = load_signal_df(subject)
        except ValueError:
            signal_df = None

    axis_w = widgets.Dropdown(options=[("sagittal (k)", "k"), ("coronal (i)", "i"),
                                       ("horizontal (j)", "j")], value="k",
                              description="view")
    idx_w = widgets.IntSlider(min=0, max=signal.shape[2] - 1,
                              value=signal.shape[2] // 2, description="slice",
                              continuous_update=False, layout=widgets.Layout(width="70%"))
    mask_w = widgets.Checkbox(value=True, description="show threshold mask")
    out = widgets.Output()

    def _redraw(*_):
        axis = axis_w.value
        dim = {"i": 0, "j": 1, "k": 2}[axis]
        idx_w.max = signal.shape[dim] - 1
        s = min(idx_w.value, idx_w.max)
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(7, 5.5))
            if axis == "k":
                img, xi, yi = signal[:, :, s].T, 0, 1
            elif axis == "i":
                img, xi, yi = signal[s, :, :], 2, 1
            else:
                img, xi, yi = signal[:, s, :].T, 0, 2
            ax.imshow(puf.adjust_contrast(img), cmap="gray", origin="upper")
            if mask_w.value and signal_df is not None and len(signal_df):
                near = signal_df[np.abs(signal_df[axis] - s) <= 2]
                if len(near):
                    ax.scatter(near.iloc[:, xi], near.iloc[:, yi], s=3,
                               c="#00CFFF", alpha=0.45, lw=0)
            ax.set_title(f"{subject} — {axis}={s}   (read coords off the axes)",
                         fontsize=10)
            plt.show()

    for w in (axis_w, idx_w, mask_w):
        w.observe(_redraw, names="value")
    _redraw()
    display(widgets.VBox([widgets.HBox([axis_w, mask_w]), idx_w, out]))
    return {"axis": axis_w, "index": idx_w}


def plot_fit_qc(subject: str, result: dict, *, data: dict | None = None,
                signal_df: pd.DataFrame | None = None,
                compare: dict | None = None, title: str | None = None,
                save: bool | str = False):
    """Contacts overlaid on the dye, plus the recorded bank's structure profile.

    ``compare`` optionally overlays a second fit (e.g. the automated one) in a
    contrasting colour so a correction can be read at a glance.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = data if data is not None else load_volumes(subject)
    signal_df = signal_df if signal_df is not None else result.get("signal_df")
    plane = result["plane"]
    coords = result["probe_df"][[f"downsample_coords.{c}" for c in "ijk"]].values
    centre = np.asarray(plane["centroid"], float)
    signal = data["signal_data"]

    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.1, 1.25], wspace=0.25)
    ax_sag, ax_cor, ax_bank = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                               fig.add_subplot(gs[2]))

    # --- sagittal (fix k) and coronal (fix i) slices through the centroid ---
    sag = puf.adjust_contrast(signal[:, :, int(centre[2])].T)
    cor = puf.adjust_contrast(signal[int(centre[0]), :, :])
    ax_sag.imshow(sag, cmap="gray", origin="upper")
    ax_cor.imshow(cor, cmap="gray", origin="upper")

    if compare is not None:
        cc = compare["probe_df"][[f"downsample_coords.{c}" for c in "ijk"]].values
        ax_sag.scatter(cc[:, 0], cc[:, 1], s=2, c="#FFA500", alpha=0.5, lw=0)
        ax_cor.scatter(cc[:, 2], cc[:, 1], s=2, c="#FFA500", alpha=0.5, lw=0)

    ax_sag.scatter(coords[:, 0], coords[:, 1], s=2, c="#FF2D2D", alpha=0.55, lw=0)
    ax_cor.scatter(coords[:, 2], coords[:, 1], s=2, c="#FF2D2D", alpha=0.55, lw=0)

    # mark the recorded bank -- the only part that yields data
    bank_mask = result["probe_df"]["probe_coords.y"].values <= RECORDED_BANK_MAX_UM
    ax_sag.scatter(coords[bank_mask, 0], coords[bank_mask, 1], s=5,
                   c="#FFE800", alpha=0.9, lw=0)
    ax_cor.scatter(coords[bank_mask, 2], coords[bank_mask, 1], s=5,
                   c="#FFE800", alpha=0.9, lw=0)

    # Dye goes on TOP of the contacts: when the fit is good the contacts sit on
    # the dye, and drawing dye underneath would hide exactly the agreement the
    # figure exists to show (ah08's 458 points vanished beneath 2256 contacts).
    if signal_df is not None and len(signal_df):
        pts = signal_df[["i", "j", "k"]].values
        ax_sag.scatter(pts[:, 0], pts[:, 1], s=1.5, c="#00CFFF", alpha=0.45, lw=0)
        ax_cor.scatter(pts[:, 2], pts[:, 1], s=1.5, c="#00CFFF", alpha=0.45, lw=0)

    ax_sag.set_title("Sagittal (dye + fit)", fontsize=10)
    ax_cor.set_title("Coronal (dye + fit)", fontsize=10)
    for ax in (ax_sag, ax_cor):
        ax.axis("off")

    handles = [Line2D([], [], marker="o", ls="", color="#00CFFF", label="DiI signal"),
               Line2D([], [], marker="o", ls="", color="#FF2D2D", label="fit contacts"),
               Line2D([], [], marker="o", ls="", color="#FFE800", label="recorded bank")]
    if compare is not None:
        handles.append(Line2D([], [], marker="o", ls="", color="#FFA500",
                              label="comparison fit"))
    ax_sag.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.6)

    # --- structures along the recorded bank ---
    bank = result["probe_df"][result["probe_df"]["probe_coords.y"] <= RECORDED_BANK_MAX_UM]
    counts = bank["structure.acronym"].fillna("none").value_counts().head(10)[::-1]
    ax_bank.barh(range(len(counts)), counts.values, color="#6667AB")
    ax_bank.set_yticks(range(len(counts)))
    ax_bank.set_yticklabels(counts.index, fontsize=8)
    ax_bank.set_xlabel("contacts in recorded bank", fontsize=8)
    ax_bank.set_title("Where the data comes from", fontsize=10)
    ax_bank.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax_bank.spines[side].set_visible(False)

    qc = result["qc"]
    head = title or (
        f"{subject} | lat {qc['lateral_deg']:.1f}°  ap {qc['ap_deg']:.1f}°  "
        f"θ {qc['theta_deg']:.1f}°  depth {qc['probe_depth']:.0f}µm  "
        f"resid {qc['resid_signal2contact_um']:.0f}µm  cov {qc['signal_coverage']:.2f}"
    )
    fig.suptitle(head, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURE_DIR / (save if isinstance(save, str) else f"{subject}_fit_qc.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved {path}")
    return fig


# --------------------------------------------------------------------------
# Synthetic controls
#
# Repo doctrine: build synthetics that flow through the *real* pipeline and gate
# the analysis on them, stating the gate in both directions.  Here the ground
# truth is a known (plane, params); dye is synthesised along that known track,
# the fit is then perturbed and re-fitted, and we check both that QC flags the
# perturbation and that the re-fit recovers the truth.  Thresholds in
# QC_THRESHOLDS are justified by this, not by tuning until the real data passes.
# --------------------------------------------------------------------------


def make_synthetic_signal(plane: dict, params: dict, *, n_points: int = 4000,
                          jitter_um: float = 30.0, dye_fraction: float = 1.0,
                          dye_from: str = "tip",
                          artefact: dict | None = None,
                          seed: int = 0) -> pd.DataFrame:
    """Synthesise a DiI cloud along a known probe placement.

    ``dye_fraction`` < 1 keeps only part of the track.  ``dye_from`` says which
    end survives: ``'tip'`` keeps the deep end (dye visible only near the tip),
    ``'surface'`` keeps the shallow end -- the realistic case, since dye is
    wiped onto the shank on the way in and commonly fades before the tip.  The
    two truncations bias depth in *opposite* directions, so which one is modelled
    matters; see `depth_objective_comparison`.  ``artefact`` adds a spurious blob
    ``{'centre': (i,j,k), 'n': 500, 'sigma_um': 60}`` to check that the largest
    cluster rule is not derailed by it.
    """
    rng = np.random.default_rng(seed)
    probe_df = pit.get_probe_contacts_df(PROBE_MANUFACTURER, PROBE_NAME)
    transformed = pit.transform_2d_probe(probe_df, [params[k] for k in PARAM_ORDER])
    coords = pit.project_2d_points_to_plane(transformed, plane).values

    if dye_fraction < 1.0:
        depth = probe_df.loc[transformed.index, "probe_coords.y"].values
        if dye_from == "tip":            # y=0 is the tip: keep small y
            keep = depth <= np.quantile(depth, dye_fraction)
        elif dye_from == "surface":      # keep large y -- dye faded before the tip
            keep = depth >= np.quantile(depth, 1.0 - dye_fraction)
        else:
            raise ValueError(f"dye_from must be 'tip' or 'surface', got {dye_from!r}")
        coords = coords[keep]

    idx = rng.integers(0, len(coords), size=n_points)
    pts = coords[idx] + rng.normal(0, jitter_um / VOXEL_SIZE_UM, size=(n_points, 3))

    if artefact:
        centre = np.asarray(artefact["centre"], float)
        n_art = int(artefact.get("n", 500))
        sigma = float(artefact.get("sigma_um", 60.0)) / VOXEL_SIZE_UM
        pts = np.vstack([pts, rng.normal(centre, sigma, size=(n_art, 3))])

    df = pd.DataFrame(np.round(pts), columns=["i", "j", "k"]).astype(int)
    df["value"] = 1.0
    df["norm_value"] = 1.0
    return df


def _perturb(plane: dict, params: dict, kind: str) -> tuple[dict, dict]:
    """Reproduce the observed failure modes as known perturbations of a truth."""
    angles = trajectory_angles(plane["v_axis"])
    params = dict(params)
    if kind == "truth":
        return dict(plane), params
    if kind == "ap_tilt":            # ah10's plane error
        v = v_axis_from_angles(angles["lateral_deg"], angles["ap_deg"] + 25.0)
    elif kind == "theta":            # a spurious in-plane rotation
        v = plane["v_axis"]
        params["theta"] = params["theta"] + np.radians(24.0)
    elif kind == "depth":            # ah10's truncation
        v = plane["v_axis"]
        params["probe_depth"] = params["probe_depth"] - 2000.0
    elif kind == "offset":
        v = plane["v_axis"]
        params["offset_x"] = params["offset_x"] + 400.0
    elif kind == "ah10_combo":       # plane error + compensating rotation
        v = v_axis_from_angles(angles["lateral_deg"], angles["ap_deg"] + 25.0)
        params["theta"] = params["theta"] - np.radians(24.0)
        params["probe_depth"] = params["probe_depth"] - 2000.0
    else:
        raise ValueError(f"unknown perturbation {kind!r}")
    new_plane = build_plane(v, plane["surface_coord"], centroid=plane.get("centroid"),
                            reference_u=plane.get("u_axis"))
    return new_plane, params


SYNTHETIC_KINDS = ("truth", "ap_tilt", "theta", "depth", "offset", "ah10_combo")


def run_synthetic_controls(subject: str = "ah08", *, seed: int = 0,
                           n_points: int = 4000, verbose: bool = True) -> pd.DataFrame:
    """Gate the QC harness and the trajectory-constrained re-fit.

    Applies **only** to automated / semi-automated fits.  It makes no claim over
    ``manual_track`` annotations, which the harness does not judge.

    The gate, stated in both directions:

    * every perturbation must be flagged (positive control), and
    * ``truth`` must **not** be flagged (negative control) -- otherwise the gate
      is vacuous and merely flags everything.
    * a trajectory-constrained re-fit from ``ah10_combo`` must recover the truth
      (depth +-150 um, theta +-3 deg, lateral/AP +-3 deg).
    """
    fit = load_fit(subject, which="auto")
    plane0, params0 = split_fit(fit)
    # Give the truth a clean, prior-consistent trajectory so recovery is
    # well-posed and independent of whether the real fit was right.
    plane0 = build_plane(v_axis_from_angles(SURGICAL_PRIOR["lateral_deg"], 0.0),
                         plane0["surface_coord"], centroid=plane0.get("centroid"),
                         reference_u=plane0.get("u_axis"))
    params0 = {**params0, "theta": 0.0, "probe_depth": 4000.0,
               "brain_shrinkage_pct": 4.0, "probe_width_scaling": 0.9,
               "offset_x": 0.0}
    truth_angles = trajectory_angles(plane0["v_axis"])
    data = load_volumes(subject)
    signal = make_synthetic_signal(plane0, params0, n_points=n_points, seed=seed)

    rows = []
    for kind in SYNTHETIC_KINDS:
        plane, params = _perturb(plane0, params0, kind)
        probe_df = project_probe(plane, params, data)
        qc = qc_probe_fit(subject, plane, params, probe_df, signal_df=signal)
        rows.append({
            "kind": kind, "flagged": bool(qc["flags"]),
            "theta_deg": qc["theta_deg"], "ap_deg": qc["ap_deg"],
            "lateral_deg": qc["lateral_deg"], "depth": qc["probe_depth"],
            "signal_coverage": qc["signal_coverage"],
            "resid_signal2contact_um": qc["resid_signal2contact_um"],
            "flags": ",".join(qc["flags"]) or "-",
            "expect_flagged": kind != "truth",
        })
    table = pd.DataFrame(rows)
    table["direction_ok"] = table["flagged"] == table["expect_flagged"]

    # Recovery: can a trajectory-constrained re-fit undo the ah10 failure mode?
    bad_plane, bad_params = _perturb(plane0, params0, "ah10_combo")
    rec_params, rec_plane = _optimize(
        signal, bad_plane, bad_params, DEFAULT_BOUNDS, {},
        {**SURGICAL_PRIOR, "lateral_deg": truth_angles["lateral_deg"],
         "ap_deg": truth_angles["ap_deg"]},
    )
    rec_ang = trajectory_angles(rec_plane["v_axis"])
    recovery = {
        "depth_err_um": abs(rec_params["probe_depth"] - params0["probe_depth"]),
        "theta_err_deg": abs(np.degrees(rec_params["theta"] - params0["theta"])),
        "lateral_err_deg": abs(rec_ang["lateral_deg"] - truth_angles["lateral_deg"]),
        "ap_err_deg": abs(rec_ang["ap_deg"] - truth_angles["ap_deg"]),
    }
    recovery["ok"] = bool(
        recovery["depth_err_um"] <= 150
        and recovery["theta_err_deg"] <= 3
        and recovery["lateral_err_deg"] <= 3
        and recovery["ap_err_deg"] <= 3
    )

    # The signature the ah10 diagnosis leans on must actually show up.
    combo = table.set_index("kind").loc["ah10_combo"]
    cancellation = abs(combo["ap_deg"] + combo["theta_deg"]
                       - truth_angles["ap_deg"]) < 5.0

    passed = bool(table["direction_ok"].all() and recovery["ok"] and cancellation)
    table.attrs["recovery"] = recovery
    table.attrs["cancellation_reproduced"] = bool(cancellation)
    table.attrs["passed"] = passed

    if verbose:
        print(table[["kind", "flagged", "expect_flagged", "direction_ok",
                     "theta_deg", "ap_deg", "depth", "signal_coverage",
                     "resid_signal2contact_um", "flags"]].to_string(index=False))
        print(f"\nrecovery from ah10_combo: {recovery}")
        print(f"AP-tilt/theta cancellation reproduced: {cancellation}")
        print(f"\nSYNTHETIC GATE: {'PASS' if passed else 'FAIL'}")
    return table


def depth_objective_comparison(subject: str = "ah08", *,
                               dye_fractions=(1.0, 0.8, 0.6, 0.4),
                               dye_from=("tip", "surface"),
                               true_depth: float = 4000.0,
                               n_points: int = 4000, seed: int = 0,
                               verbose: bool = True) -> pd.DataFrame:
    """Measure the shallow bias of the two-term cost, against a known truth.

    Dye is synthesised along a probe of known depth, then progressively faded
    from the tip (``dye_fraction`` keeps only the deepest fraction of the
    track).  Depth is then re-fitted from a deliberately shallow start under
    each objective.  If the argument in :func:`_cost` is right, the two-term
    cost should under-recover depth as dye fades while ``signal2contact`` should
    not -- that turns "the cost is biased" from an argument into a measurement.
    """
    fit = load_fit(subject, which="auto")
    plane0, params0 = split_fit(fit)
    plane0 = build_plane(v_axis_from_angles(SURGICAL_PRIOR["lateral_deg"], 0.0),
                         plane0["surface_coord"], centroid=plane0.get("centroid"),
                         reference_u=plane0.get("u_axis"))
    params0 = {**params0, "theta": 0.0, "probe_depth": true_depth,
               "brain_shrinkage_pct": 4.0, "probe_width_scaling": 0.9,
               "offset_x": 0.0}

    rows = []
    for where in np.atleast_1d(dye_from):
        for frac in dye_fractions:
            signal = make_synthetic_signal(plane0, params0, n_points=n_points,
                                           dye_fraction=frac, dye_from=str(where),
                                           seed=seed)
            for objective in ("both", "signal2contact"):
                start = {**params0, "probe_depth": true_depth * 0.6}
                fitted, _ = _optimize(signal, plane0, start, DEFAULT_BOUNDS,
                                      {k: start[k] for k in PARAM_ORDER
                                       if k != "probe_depth"},
                                      None, objective)
                rows.append({"dye_from": str(where), "dye_fraction": frac,
                             "objective": objective, "true_depth": true_depth,
                             "fitted_depth": fitted["probe_depth"],
                             "error_um": fitted["probe_depth"] - true_depth})
    table = pd.DataFrame(rows)
    piv = table.pivot(index=["dye_from", "dye_fraction"], columns="objective",
                      values="error_um")
    table.attrs["pivot"] = piv
    if verbose:
        print("Depth error (um) vs a known true depth.\n"
              "  dye_from='surface' is the realistic case: dye wiped on during\n"
              "  insertion, fading before the tip.  negative = fit stops short.\n")
        print(piv.round(0).to_string())
    return table


def depth_sweep(subject: str, *, depths=None, plane: dict | None = None,
                params: dict | None = None, data: dict | None = None,
                signal_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-shank structure at the recorded bank as a function of probe depth.

    Replaces a single verdict with the curve: at what depth does each shank
    enter the target region, and is that depth compatible with the dye?
    """
    data = data if data is not None else load_volumes(subject)
    fit = load_fit(subject)
    saved_plane, saved_params = split_fit(fit)
    plane = plane or saved_plane
    params = dict(params or saved_params)
    if depths is None:
        depths = np.arange(2400, 5001, 200)

    if signal_df is None:
        try:
            signal_df = load_signal_df(subject)
        except ValueError:
            signal_df = None
    dye_extent = (float(np.ptp(signal_df[["i", "j", "k"]].values
                               @ np.asarray(plane["v_axis"], float)) * VOXEL_SIZE_UM)
                  if signal_df is not None and len(signal_df) else np.nan)

    rows = []
    for d in depths:
        p = {**params, "probe_depth": float(d)}
        probe_df = project_probe(plane, p, data)
        bank = probe_df[probe_df["probe_coords.y"] <= RECORDED_BANK_MAX_UM]
        bank = bank.assign(shank=shank_id(bank["probe_coords.x"]))
        row = {"depth_um": float(d), "dye_extent_um": dye_extent,
               "depth_over_dye": float(d) / dye_extent if dye_extent == dye_extent else np.nan}
        for shank, g in bank.groupby("shank"):
            acro = g["structure.acronym"].dropna()
            row[f"s{int(shank)}_ap"] = float(g["downsample_coords.i"].mean())
            row[f"s{int(shank)}_top"] = acro.mode().iat[0] if len(acro) else "none"
            row[f"s{int(shank)}_ENTl"] = float(
                acro.str.startswith(TARGET_REGION).mean()) if len(acro) else 0.0
        pooled = bank["structure.acronym"].dropna()
        row["pooled_top"] = pooled.mode().iat[0] if len(pooled) else "none"
        row["pooled_ENTl"] = float(
            pooled.str.startswith(TARGET_REGION).mean()) if len(pooled) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def qc_table(subjects: list[str] | None = None, *, which: str = "current",
             signal: bool = True, keep_volumes: bool = False,
             verbose: bool = True) -> pd.DataFrame:
    """QC every subject's stored fit into one table."""
    subjects = subjects or list_subjects()
    rows = []
    for subject in subjects:
        if verbose:
            print(f"  QC {subject} ...", flush=True)
        fit = load_fit(subject, which=which)
        plane, params = split_fit(fit)
        data = load_volumes(subject)
        probe_df = project_probe(plane, params, data)
        sig = load_signal_df(subject) if signal else None
        qc = qc_probe_fit(subject, plane, params, probe_df, signal_df=sig)
        qc["fit_method"] = fit.get("fit_method", "auto")
        qc["confidence"] = fit.get("confidence", "unset")
        qc["grade"] = grade_fit(qc, qc["fit_method"])
        qc["n_signal_voxels"] = int(len(sig)) if sig is not None else -1
        rows.append(qc)
        if not keep_volumes:
            clear_volume_cache(subject)
    # resid_signal2contact is the flagged metric -- see QC_THRESHOLDS on why
    # contact2signal is not diagnostic.
    cols = ["subject", "fit_method", "grade", "confidence", "lateral_deg", "ap_deg",
            "dye_lateral_deg", "dye_ap_deg", "fit_vs_dye_axis_deg",
            "dye_pc1_expl_var", "theta_deg", "trajectory_dev_deg", "probe_depth",
            "dye_extent_um", "depth_over_dye", "signal_coverage",
            "resid_signal2contact_um", "n_signal_voxels", "target_fraction",
            "tip_structure", "params_at_bounds", "flags", "advisories"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]
