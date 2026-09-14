"""Registration QC: does the brainreg -> Allen registration support ENTl layer labels?

Everything in ``probe_refit`` judges the *probe fit* against the DiI.  Nothing before
this module validated the *registration* itself beyond ``orientation_check.png``
(axis order).  Adam's conclusions rest on ENTl superficial (1/2/3) vs deep (5/6a/6b),
boundaries ~100-200 um apart at the ventro-lateral surface, so the checks here are
local to the track as well as global.  Full rationale and the measured numbers are
in ``REGISTRATION_QC.md``; the plan that produced this module is
``~/.claude/plans/merry-sleeping-hartmanis.md``.

Conventions (verified from ``niftyreg/affine.log`` and the nii headers)
-----------------------------------------------------------------------
* Sample volumes are ``asr``: i = anterior->posterior, j = superior->inferior,
  k = right->left, 10 um voxels, shape ~(884, ~700, ~1005).  A coronal page is
  ``vol[i]`` -> (j, k); ``imshow(origin='upper')`` puts superior at the top and the
  RIGHT hemisphere on the image-left.
* niftyreg ran with reference = sample, floating = atlas.  ``affine_matrix.txt``
  maps **sample mm -> atlas mm**; rows are (AP, DV, LR).  All nii have
  ``affine = diag(0.01)`` and zero origin, so atlas voxel = mm / 0.01, no flips.
* ``deformation_field_{0,1,2}.tiff`` live on the sample grid and hold the atlas
  coordinate in **mm** (affine included).  Values are negative outside the brain:
  mask before statistics, clip before indexing the annotation.
* ``registered_hemispheres``: 1 = left = high k = the implant side.  The defaults
  inside ``brainreg/core/utils/volume.py`` are reversed -- never use them.
* ``boundaries.tiff`` marks *every* inter-label edge (both sides), not the brain
  outline.  Brain mask = ``registered_atlas > 0``.
* ``niftyreg/brain_filtered.nii`` is the atlas reference after a high-pass
  (``img / (gaussian(img, 5) + 1)``): fine for edges, useless for intensities.  The
  unfiltered 25 um atlas in ``~/.brainglobe/allen_mouse_25um_v1.2`` is used for
  anything that reads intensity.
* ``downsampled_standard.tiff`` was produced with a *separately estimated inverse*
  B-spline, so atlas-space checks do not map 1:1 onto sample-space label error.
* The freeform control grid is **400 um** (``control_point_file.nii``).  The warp
  cannot represent structure finer than that; layer boundaries are the atlas's
  proportional layering carried by a smooth warp between the two edges the
  registration can see -- pia and white matter.  Check E measures exactly those.
* Fibre tracts are DARKER than grey matter in this autofluorescence (~220 vs
  ~1000-1130 in ENTl), so the grey/white edge is a bright->dark step going deeper.
* Allen structure ids reach 614,454,277: dict lookups and ``np.isin`` against small
  id arrays only, never a dense array indexed by id (``probe_refit._atlas_maps``).

Memory: never hold more than one full sample volume; everything else is page ranges
(``tifffile.imread(path, key=range(lo, hi))``, ``nib.load(...).dataobj[lo:hi]``).
Interactive sessions are a 64 GB cgroup.

Runs in the ``histology`` conda env, from ``code/histology_refit/``.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.linalg import polar
from scipy.spatial.transform import Rotation
from scipy.stats import theilslopes

import probe_refit as pr
import probe_tool as pt
import region_assignment as ra
from brainreg_probe import probeinterface_tracing as pit
from brainreg_probe import plot_util_func as puf

# --------------------------------------------------------------------------
# constants and paths
# --------------------------------------------------------------------------

FIGURE_DIR = pr.REPO_ROOT / "data" / "figures" / "registration_qc"
BRAINREG_DIR = pr.BRAINREG_DIR
ATLAS_NAME = pr.ATLAS_NAME
ATLAS_25_DIR = Path.home() / ".brainglobe" / "allen_mouse_25um_v1.2"

VOX_MM = 0.01
VOX_UM = 10.0
ATLAS_SHAPE = (1320, 800, 1140)
ATLAS_25_SCALE = 2.5                      # 25 um index * 2.5 = 10 um index
FIBRE_TRACTS_ID = 1009                    # Allen "fiber tracts"
ROOT_ID = 997

#: AP margin (pages) around the recorded bank for every slab read.
SLAB_MARGIN_VOX = 30

#: Per-page stride for the whole-stack profiles (200 um).
PROFILE_STRIDE = 20

#: Thresholds.  Justified by run_synthetic_controls(), not tuned on the real data.
QC_THRESHOLDS = {
    # Per-axis affine scale (sample mm -> atlas mm).  The cohort's DV scale is 1.08-1.15
    # in every mouse (samples 8-15 % shorter dorsoventrally than the CCF -- preparation
    # and/or atlas bias, systematic, not a per-brain failure; ah10 sits at 1.1500), so
    # the band is set to catch a *per-brain* outlier, not the shared shrinkage.
    "affine_scale": (0.85, 1.20),
    "affine_rotation_deg": 15.0,          # total rotation the affine had to absorb
    "residual_angle_deg": 1.0,            # residual yaw / pitch after registration
    "lr_asym": 0.15,                      # |(L-R)/(L+R)| on LEC/HPF region volumes
    "detj_range": (0.5, 2.0),             # local volume ratio atlas/sample
    "detj_frac_outside": 0.01,
    "laminar_offset_um": 100.0,           # |median atlas - visible edge| at the track
    "fragile_frac_100um": 0.50,           # advisory: ENTl contacts within 100 um of sup/deep
    "lecmec_frac_100um": 0.25,            # advisory: bank contacts within 100 um of the ENTl/ENTm border
    "self_consistency_extra": 0.02,       # disagreement above the boundary floor
}

_PAGE_CACHE: dict = {}
_SLAB_CACHE: dict = {}
_TREE: dict = {}
_ATLAS25: dict = {}


# --------------------------------------------------------------------------
# IO layer
# --------------------------------------------------------------------------


def subject_paths(subject: str) -> dict:
    """Every registration artefact for a mouse, as Paths."""
    d = pr.subject_dir(subject) / ATLAS_NAME
    nr = d / "niftyreg"
    return {
        "dir": d, "niftyreg": nr,
        "green": d / "downsampled.tiff",
        "dye": d / "downsampled_2.tiff",
        "atlas": d / "registered_atlas.tiff",
        "hemispheres": d / "registered_hemispheres.tiff",
        "boundaries": d / "boundaries.tiff",
        "standard": d / "downsampled_standard.tiff",
        "def": [d / f"deformation_field_{c}.tiff" for c in range(3)],
        "volumes": d / "volumes.csv",
        "affine": nr / "affine_matrix.txt",
        "affine_inv": nr / "invert_affine_matrix.txt",
        "annotations": nr / "annotations.nii",
        "brain_filtered": nr / "brain_filtered.nii",
        "affine_atlas": nr / "affine_registered_atlas_brain.nii",
        "control_points": nr / "control_point_file.nii",
        "global_cache": pr.subject_dir(subject) / "registration_qc_global.npz",
        "landmarks": pr.subject_dir(subject) / "registration_landmarks.json",
    }


def n_pages(path) -> int:
    import tifffile
    with tifffile.TiffFile(path) as t:
        return len(t.pages)


def volume_shape(path) -> tuple:
    import tifffile
    with tifffile.TiffFile(path) as t:
        return tuple(t.series[0].shape)


def read_pages(path, lo: int, hi: int, step: int = 1) -> np.ndarray:
    """Pages ``lo:hi:step`` of a tiff stack as one array (n, J, K)."""
    import tifffile
    n = n_pages(path)
    lo, hi = max(int(lo), 0), min(int(hi), n)
    keys = list(range(lo, hi, step))
    if len(keys) == 1:
        return tifffile.imread(path, key=keys[0])[None]
    return tifffile.imread(path, key=keys)


def read_page(path, i: int) -> np.ndarray:
    import tifffile
    return tifffile.imread(path, key=int(i))


def nii_proxy(path):
    """Memory-mapped nii; slice ``proxy[lo:hi]`` reads only those bytes."""
    import nibabel as nib
    return nib.load(str(path)).dataobj


def load_affine(subject: str) -> tuple[np.ndarray, np.ndarray]:
    """(A, A_inv): 4x4 in mm, sample -> atlas.  Rows are (AP, DV, LR)."""
    p = subject_paths(subject)
    A = np.loadtxt(p["affine"])
    if p["affine_inv"].exists():
        A_inv = np.loadtxt(p["affine_inv"])
    else:
        A_inv = np.linalg.inv(A)
    return A, A_inv


def def_slab(subject: str, lo: int, hi: int) -> np.ndarray:
    """Deformation field pages ``lo:hi`` as (3, n, J, K) float32 in atlas mm.

    One 160-page slab is ~1.3 GB, so the cache holds a single slab per subject.
    """
    key = (subject, int(lo), int(hi))
    if key in _SLAB_CACHE:
        return _SLAB_CACHE[key]
    p = subject_paths(subject)
    arr = np.stack([read_pages(f, lo, hi) for f in p["def"]]).astype(np.float32)
    for k in [k for k in _SLAB_CACHE if k[0] == subject]:
        del _SLAB_CACHE[k]
    _SLAB_CACHE[key] = arr
    return arr


def clear_caches(subject: str | None = None) -> None:
    for cache in (_SLAB_CACHE, _PAGE_CACHE):
        for k in [k for k in cache if subject is None or k[0] == subject]:
            del cache[k]


def sample_vox_to_atlas_mm(subject: str, pts: np.ndarray) -> np.ndarray:
    """Atlas coordinates (mm) of sample voxels, reading one deformation page each.

    Nearest-voxel lookup (points are rounded), grouped by page so a contact set
    spanning 90 pages costs 90 x 3 page reads rather than a 7 GB load.
    """
    p = subject_paths(subject)
    pts = np.asarray(pts, float)
    idx = np.round(pts).astype(int)
    shape = volume_shape(p["atlas"])
    idx = np.clip(idx, 0, np.asarray(shape) - 1)
    out = np.full((len(pts), 3), np.nan, dtype=float)
    for page in np.unique(idx[:, 0]):
        sel = idx[:, 0] == page
        for c in range(3):
            pg = read_page(p["def"][c], page)
            out[sel, c] = pg[idx[sel, 1], idx[sel, 2]]
    return out


# --------------------------------------------------------------------------
# structure tree and label groups
# --------------------------------------------------------------------------


def structure_tree() -> dict:
    """id -> acronym / name / parent, plus ancestry, from the pipeline's own table.

    ``allen_brain_atlas_info.htsv`` (1327 rows) has ``parent_structure_id`` but no
    path column, so ancestry is built by walking parents.  Dicts throughout.
    """
    if _TREE:
        return _TREE
    info = pit.ALLEN_ATLAS_INFO_DF
    ids = info["id"].astype(int).values
    acro = info["acronym"].astype(str).values
    name = info["name"].astype(str).values
    parent = pd.to_numeric(info["parent_structure_id"], errors="coerce").values
    id2acro = dict(zip(ids, acro))
    id2name = dict(zip(ids, name))
    id2parent = {int(i): (int(p) if np.isfinite(p) else None) for i, p in zip(ids, parent)}
    acro2id = {a: int(i) for i, a in zip(ids, acro)}
    name2acro = dict(zip(name, acro))

    def ancestors(i):
        out, seen = [], set()
        while i is not None and i not in seen:
            seen.add(i)
            out.append(i)
            i = id2parent.get(i)
        return out

    id2anc = {int(i): ancestors(int(i)) for i in ids}
    _TREE.update({"id2acro": id2acro, "id2name": id2name, "id2parent": id2parent,
                  "acro2id": acro2id, "name2acro": name2acro, "id2anc": id2anc})
    return _TREE


def layer_id_sets() -> dict[str, np.ndarray]:
    """Small id arrays per group, for ``np.isin`` on slices.

    ENTl superficial = layers 1/2/3 (incl. 2a/2b/2-3), deep = 5/6a/6b, following
    ``region_assignment.group_of``.  ``fibre`` = every descendant of "fiber tracts"
    (1009).  The ontology table also lists ENTl4 / ENTl4/5 / ENTl5/6 as nodes;
    CCFv3 has no voxels for them, which :func:`contacts` asserts on the labelled
    contacts, so ``group_of``'s superficial default for ENTl4 can never matter.
    """
    tree = structure_tree()
    groups: dict[str, list[int]] = {"ENTl_sup": [], "ENTl_deep": [], "ENTl": [],
                                    "ENTm": [], "SUB_ProS": [], "CA_HPF": [], "VIS": [],
                                    "fibre": [], "root": [ROOT_ID]}
    for i, a in tree["id2acro"].items():
        if a.startswith("ENTl"):
            groups["ENTl"].append(i)
            groups["ENTl_deep" if ra.group_of(a) == "ENTl-deep" else "ENTl_sup"].append(i)
        elif a.startswith("ENTm"):
            groups["ENTm"].append(i)
        elif a in ("SUB", "ProS"):
            groups["SUB_ProS"].append(i)
        elif a in ("CA1", "CA2", "CA3", "DG", "HPF") or a.startswith("DG-") or a.startswith("CA1") or a.startswith("CA3"):
            groups["CA_HPF"].append(i)
        elif a.startswith("VIS"):
            groups["VIS"].append(i)
        if FIBRE_TRACTS_ID in tree["id2anc"][i] and i != FIBRE_TRACTS_ID:
            groups["fibre"].append(i)
    groups["fibre"].append(FIBRE_TRACTS_ID)
    return {g: np.asarray(sorted(set(v)), dtype=np.int64) for g, v in groups.items()}


#: Colours for the overlay contours (superficial vs deep is the point).
GROUP_COLOURS = {
    "ENTl_sup": "#7fb8e0", "ENTl_deep": "#1f4e9a", "ENTm": "#ff7f0e",
    "SUB_ProS": "#9467bd", "CA_HPF": "#2ca02c", "fibre": "#f0e442", "VIS": "#8c8c8c",
}


def atlas25() -> dict:
    """The unfiltered 25 um Allen atlas (reference + annotation), loaded once."""
    if _ATLAS25:
        return _ATLAS25
    import tifffile
    ref = tifffile.imread(ATLAS_25_DIR / "reference.tiff")
    ann = tifffile.imread(ATLAS_25_DIR / "annotation.tiff")
    _ATLAS25.update({"reference": ref, "annotation": ann})
    return _ATLAS25


def atlas_reference_page(i_atlas: int) -> np.ndarray:
    """Unfiltered atlas reference at 10 um atlas page ``i_atlas`` (zoomed from 25 um)."""
    ref = atlas25()["reference"]
    i25 = int(np.clip(round(i_atlas / ATLAS_25_SCALE), 0, ref.shape[0] - 1))
    page = ref[i25].astype(np.float32)
    return ndimage.zoom(page, ATLAS_25_SCALE, order=1)[: ATLAS_SHAPE[1], : ATLAS_SHAPE[2]]


def atlas_region_volumes(cache: bool = True) -> pd.DataFrame:
    """Per-structure atlas volume (mm3) and AP extent (10 um index) from the 25 um annotation.

    Cached to ``brainreg/atlas_region_volumes.csv``.  25 um discretisation is within
    ~2 % of the 10 um volumes for the regions we compare; the 10 um annotation is
    4.8 GB and would buy nothing here.
    """
    path = BRAINREG_DIR / "atlas_region_volumes.csv"
    if cache and path.exists():
        return pd.read_csv(path)
    ann = atlas25()["annotation"]
    ids, counts = np.unique(ann, return_counts=True)
    tree = structure_tree()
    rows = []
    # AP extent per id: first/last page containing the id
    present = {int(i): [np.inf, -np.inf] for i in ids if i != 0}
    for i25 in range(ann.shape[0]):
        u = np.unique(ann[i25])
        for v in u:
            if v == 0:
                continue
            e = present[int(v)]
            e[0] = min(e[0], i25)
            e[1] = max(e[1], i25)
    for i, c in zip(ids, counts):
        if i == 0:
            continue
        rows.append({"id": int(i), "acronym": tree["id2acro"].get(int(i), "?"),
                     "name": tree["id2name"].get(int(i), "?"),
                     "atlas_volume_mm3": float(c) * 0.025 ** 3,
                     "ap_min_10um": present[int(i)][0] * ATLAS_25_SCALE,
                     "ap_max_10um": (present[int(i)][1] + 1) * ATLAS_25_SCALE})
    df = pd.DataFrame(rows)
    if cache:
        df.to_csv(path, index=False)
    return df


def atlas_brain_extent_10um() -> tuple[float, float]:
    """(first, last) 10 um atlas AP index containing any labelled voxel."""
    ann = atlas25()["annotation"]
    nz = np.where((ann > 0).any(axis=(1, 2)))[0]
    return float(nz.min() * ATLAS_25_SCALE), float((nz.max() + 1) * ATLAS_25_SCALE)


# --------------------------------------------------------------------------
# tissue masks and whole-stack page statistics
# --------------------------------------------------------------------------


def tissue_threshold(subject: str, n_sample: int = 12) -> float:
    """Tissue-vs-agarose threshold: Otsu on log intensity over sampled pages.

    Otsu on *linear* intensity lands inside the tissue range and is confounded by
    L/R illumination (ah09 read a physically impossible 0.7 L/R asymmetry with it);
    on log intensity it separates agarose (~40-50) from any tissue (~90-130).
    """
    from skimage.filters import threshold_otsu
    p = subject_paths(subject)
    n = n_pages(p["green"])
    pages = np.linspace(n * 0.1, n * 0.9, n_sample).astype(int)
    sample = np.concatenate([read_page(p["green"], i)[::3, ::3].ravel() for i in pages])
    return float(np.expm1(threshold_otsu(np.log1p(sample.astype(np.float32)))))


def tissue_mask(page: np.ndarray, thr: float, *, largest: bool = True) -> np.ndarray:
    """Binary tissue mask of one coronal page: threshold, open, fill, largest component."""
    m = page > thr
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    if largest and m.any():
        lab, n = ndimage.label(m)
        if n > 1:
            m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return m


def page_stats(subject: str, *, rebuild: bool = False) -> pd.DataFrame:
    """Per-page tissue area, centroid, mean intensity for every coronal page.

    One pass over ``downsampled.tiff`` (1.2 GB, ~30 s); cached in the global npz.
    Drives the truncation report and the section-continuity check.
    """
    import tifffile
    p = subject_paths(subject)
    cached = _load_global(subject)
    if not rebuild and cached is not None and "page_area" in cached:
        return pd.DataFrame({k[5:]: cached[k] for k in cached if k.startswith("page_")})
    thr = tissue_threshold(subject)
    rows = []
    with tifffile.TiffFile(p["green"]) as t:
        for i, pg in enumerate(t.pages):
            img = pg.asarray()
            # Largest component: six brains shared one block, so fragments of the
            # neighbours enter the field of view at the edges and would throw the
            # centroid by > 1 mm on some pages (ah09, ly07).
            m = tissue_mask(img, thr, largest=True)
            area = int(m.sum())
            if area:
                jj, kk = np.nonzero(m)
                rows.append((i, area, jj.mean(), kk.mean(), float(img[m].mean())))
            else:
                rows.append((i, 0, np.nan, np.nan, np.nan))
    df = pd.DataFrame(rows, columns=["i", "area", "cj", "ck", "mean_int"])
    df["frac"] = df["area"] / (m.shape[0] * m.shape[1])
    _update_global(subject, {f"page_{c}": df[c].values for c in df.columns}, thr=thr)
    return df


def tissue_pages(subject: str, rel: float = 0.10) -> tuple[int, int]:
    """First and last page whose tissue area exceeds ``rel`` x the peak."""
    df = page_stats(subject)
    peak = df["area"].max()
    ok = np.where(df["area"].values > rel * peak)[0]
    return int(ok.min()), int(ok.max())


def section_continuity(subject: str, jump_vox: float = 3.0) -> dict:
    """Page-to-page jumps of the tissue centroid: tissue-shape discontinuities.

    Would catch block slippage or a lost/duplicated section (a jump with no area
    change, everywhere at once).  In practice it fires on loose tissue: ah08 has a
    cortical flap on the right (non-probe) hemisphere displaced laterally over
    sample AP 6.5-7.4 mm, which moves the centroid by up to 11 voxels while the
    area changes smoothly -- descriptive, and a pointer to where the atlas cannot
    follow the tissue.  Odd/even pages (the two optical planes per section) step
    identically (0.49 vs 0.39 vox on ah08), so there is no optical-plane offset.
    """
    df = page_stats(subject)
    lo, hi = tissue_pages(subject, rel=0.3)
    d = df.iloc[lo:hi + 1]
    dj = np.abs(np.diff(d["cj"].values))
    dk = np.abs(np.diff(d["ck"].values))
    da = np.abs(np.diff(d["area"].values)) / np.maximum(d["area"].values[:-1], 1)
    jumps = np.where((dj > jump_vox) | (dk > jump_vox))[0] + lo
    return {"n_jumps": int(len(jumps)),
            "jump_ap_range_mm": [float(jumps.min() * VOX_MM), float(jumps.max() * VOX_MM)] if len(jumps) else None,
            "jump_pages": jumps.tolist()[:20],
            "median_step_vox": float(np.median(np.hypot(dj, dk))),
            "max_step_vox": float(np.max(np.hypot(dj, dk))),
            "max_area_step_frac": float(np.max(da))}


# --------------------------------------------------------------------------
# geometry: the fit, the contacts, the slab
# --------------------------------------------------------------------------


def fit_key(subject: str) -> str:
    """Short hash of the current fit (same recipe as probe_figures._fit_key)."""
    plane, params = pr.split_fit(pr.load_fit(subject))
    blob = json.dumps({"plane": {k: np.asarray(v).tolist() for k, v in plane.items()},
                       "params": {k: float(v) for k, v in params.items()}},
                      sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def contact_coords(subject: str) -> pd.DataFrame:
    """Every contact of the current fit with sample voxel coordinates -- no labels.

    Steps 6-7 of the tracing pipeline exactly as ``probe_refit.project_probe`` does
    them, minus the atlas lookup (which needs the 2.5 GB volume).  Labels are added
    from a slab by :func:`contacts`; equality with ``project_probe`` is asserted by
    :func:`verify_contacts`.
    """
    fit = pr.load_fit(subject)
    plane, params = pr.split_fit(fit)
    probe_df = pit.get_probe_contacts_df(pr.PROBE_MANUFACTURER, pr.PROBE_NAME)
    transformed = pit.transform_2d_probe(probe_df, [params[k] for k in pr.PARAM_ORDER])
    coords = pit.project_2d_points_to_plane(transformed, plane).values
    out = probe_df.loc[transformed.index].copy()
    for idx, c in enumerate("ijk"):
        out[f"downsample_coords.{c}"] = coords[:, idx]
    out["bank"] = out["probe_coords.y"].values <= pr.RECORDED_BANK_MAX_UM
    out["shank"] = pr.shank_id(out["probe_coords.x"].values)
    return out.reset_index(drop=True)


def track_geometry(subject: str) -> dict:
    """Entry, tip, bank bounding box and the AP slab used by every local check."""
    fit = pr.load_fit(subject)
    plane, params = pr.split_fit(fit)
    entry, tip = pt.fit_to_track(plane, params)
    df = contact_coords(subject)
    b = df[df["bank"]]
    coords = b[[f"downsample_coords.{c}" for c in "ijk"]].values
    shape = volume_shape(subject_paths(subject)["atlas"])
    lo = int(max(np.floor(coords[:, 0].min()) - SLAB_MARGIN_VOX, 0))
    hi = int(min(np.ceil(coords[:, 0].max()) + SLAB_MARGIN_VOX + 1, shape[0]))
    return {"entry": entry, "tip": tip, "shape": shape,
            "bank_min": coords.min(0), "bank_max": coords.max(0),
            "bank_centre": coords.mean(0), "slab": (lo, hi),
            "fit_method": fit.get("fit_method", "auto")}


def atlas_slab(subject: str, lo: int, hi: int) -> np.ndarray:
    key = (subject, "atlas", int(lo), int(hi))
    if key not in _PAGE_CACHE:
        for k in [k for k in _PAGE_CACHE if k[0] == subject and k[1] == "atlas"]:
            del _PAGE_CACHE[k]
        _PAGE_CACHE[key] = read_pages(subject_paths(subject)["atlas"], lo, hi)
    return _PAGE_CACHE[key]


def green_slab(subject: str, lo: int, hi: int) -> np.ndarray:
    key = (subject, "green", int(lo), int(hi))
    if key not in _PAGE_CACHE:
        for k in [k for k in _PAGE_CACHE if k[0] == subject and k[1] == "green"]:
            del _PAGE_CACHE[k]
        _PAGE_CACHE[key] = read_pages(subject_paths(subject)["green"], lo, hi)
    return _PAGE_CACHE[key]


def contacts(subject: str) -> pd.DataFrame:
    """Contacts with structure labels read from the registered-atlas slab.

    Same nearest-voxel-by-truncation lookup as ``fast_structure_labels``; contacts
    outside the volume are clipped and marked, as ``project_probe`` does.
    """
    df = contact_coords(subject)
    geo = track_geometry(subject)
    shape = np.asarray(geo["shape"])
    coords = df[[f"downsample_coords.{c}" for c in "ijk"]].values
    inside = np.all((coords >= 0) & (coords < shape - 1), axis=1)
    safe = np.clip(coords, 0, shape - 1).astype(int)
    lo = int(max(safe[:, 0].min(), 0))
    hi = int(min(safe[:, 0].max() + 1, shape[0]))
    slab = read_pages(subject_paths(subject)["atlas"], lo, hi)
    ids = slab[safe[:, 0] - lo, safe[:, 1], safe[:, 2]].astype(np.int64)
    maps = pr._atlas_maps()
    acro = [maps.get(int(v), (np.nan, "outside brain", np.nan))[0] for v in ids]
    name = [maps.get(int(v), (np.nan, "outside brain", np.nan))[1] for v in ids]
    df["structure.id"] = ids
    df["structure.acronym"] = acro
    df["structure.name"] = name
    df.loc[~inside, ["structure.acronym"]] = np.nan
    df.loc[~inside, ["structure.name"]] = "outside volume"
    df["inside_volume"] = inside
    df["group"] = [ra.group_of(a) for a in df["structure.acronym"]]
    odd = [a for a in df["structure.acronym"] if isinstance(a, str) and a.startswith("ENTl4")]
    assert not odd, f"ENTl4-family labels on contacts ({set(odd)}) -- revisit group_of's layer split"
    return df


def verify_contacts(subject: str) -> dict:
    """Assert the slab-based labelling equals ``probe_refit.project_probe`` exactly.

    Loads the full volumes once (4.3 GB transient), so run it in the gate, not in
    every check.
    """
    mine = contacts(subject)
    fit = pr.load_fit(subject)
    plane, params = pr.split_fit(fit)
    data = pr.load_volumes(subject, fast=True)
    ref = pr.project_probe(plane, params, data)
    pr.clear_volume_cache(subject)
    same_n = len(mine) == len(ref)
    coord_err = float(np.abs(mine[[f"downsample_coords.{c}" for c in "ijk"]].values
                             - ref[[f"downsample_coords.{c}" for c in "ijk"]].values).max()) if same_n else np.inf
    a1 = mine["structure.acronym"].astype(str).values
    a2 = ref["structure.acronym"].astype(str).values
    agree = float(np.mean(a1 == a2)) if same_n else 0.0
    out = {"n_contacts": len(mine), "same_n": same_n, "max_coord_err": coord_err,
           "label_agreement": agree, "ok": bool(same_n and coord_err < 1e-9 and agree == 1.0)}
    return out


# --------------------------------------------------------------------------
# Check B -- global sanity
# --------------------------------------------------------------------------


def decompose_affine(A: np.ndarray) -> dict:
    """Polar decomposition ``A3 = R P``: rotation, per-axis scale, shear, det.

    Euler angles are about the sample axes: roll about AP (i), yaw about DV (j),
    pitch about LR (k).  A mount *yaw* makes coronal sections oblique in L/R; a
    *pitch* is L/R-symmetric.  The matrix maps sample mm -> atlas mm, so a scale
    > 1 means the sample is *smaller* than the atlas along that axis.
    """
    A3 = np.asarray(A, float)[:3, :3]
    R, P = polar(A3)
    if np.linalg.det(R) < 0:                      # improper: fold the sign into P
        R, P = -R, -P
    eul = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    total = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return {"scale_ap": float(P[0, 0]), "scale_dv": float(P[1, 1]), "scale_lr": float(P[2, 2]),
            "shear_ap_dv": float(P[0, 1]), "shear_ap_lr": float(P[0, 2]), "shear_dv_lr": float(P[1, 2]),
            "roll_deg": float(eul[0]), "yaw_deg": float(eul[1]), "pitch_deg": float(eul[2]),
            "rotation_total_deg": total, "det": float(np.linalg.det(A3)),
            "translation_mm": np.asarray(A, float)[:3, 3].tolist()}


def verify_affine_convention(subject: str, *, n_sample: int = 20000, seed: int = 0) -> dict:
    """Empirically confirm that ``A`` maps sample mm -> atlas mm.

    Field consistency: over brain voxels of the track slab, the deformation field
    (which includes the affine) must sit within the non-linear residual of ``A x``
    -- sub-mm -- and several mm from ``A_inv x``.  The correct convention wins by
    an order of magnitude.
    """
    A, A_inv = load_affine(subject)
    geo = track_geometry(subject)
    lo, hi = geo["slab"]
    atl = atlas_slab(subject, lo, hi)
    dfield = def_slab(subject, lo, hi)
    rng = np.random.default_rng(seed)
    ijk = np.argwhere(atl > 0)
    sel = ijk[rng.choice(len(ijk), size=min(n_sample, len(ijk)), replace=False)]
    x_mm = np.c_[(sel[:, 0] + lo) * VOX_MM, sel[:, 1] * VOX_MM, sel[:, 2] * VOX_MM, np.ones(len(sel))]
    d = np.stack([dfield[c][sel[:, 0], sel[:, 1], sel[:, 2]] for c in range(3)], 1)
    res_A = np.linalg.norm(d - (x_mm @ A.T)[:, :3], axis=1)
    res_inv = np.linalg.norm(d - (x_mm @ A_inv.T)[:, :3], axis=1)
    # per-axis local step of the field vs the affine diagonal
    steps = [float(np.nanmedian(np.gradient(dfield[c], VOX_MM, axis=c)[atl > 0])) for c in range(3)]
    return {"median_resid_mm_A": float(np.median(res_A)),
            "median_resid_mm_Ainv": float(np.median(res_inv)),
            "field_step_per_axis": steps,
            "affine_diag": [float(A[c, c]) for c in range(3)],
            "convention_ok": bool(np.median(res_A) < 1.0 and np.median(res_inv) > 3 * np.median(res_A))}


def truncation_report(subject: str) -> dict:
    """How much brain is missing, and whether the affine squashed or overhung.

    Atlas AP covered = median of ``def_field_0`` over brain voxels on the first and
    last tissue pages.  Overhang = atlas brain extent beyond those.  With
    ``A_ii`` ~ 1 the truncated sample maps onto a *part* of the atlas (overhang,
    the good outcome); squashing 8.8 mm of tissue onto the whole 13.2 mm atlas
    would need ``A_ii`` ~ 1.4.
    """
    p = subject_paths(subject)
    A, _ = load_affine(subject)
    first, last = tissue_pages(subject)
    df = page_stats(subject)
    covered = []
    for page in (first, last):
        d0 = read_page(p["def"][0], page)
        atl = read_page(p["atlas"], page) > 0
        covered.append(float(np.median(d0[atl])) / VOX_MM if atl.any() else np.nan)
    ext = atlas_brain_extent_10um()
    n = len(df)
    return {"first_tissue_page": first, "last_tissue_page": last,
            "tissue_ap_mm": (last - first + 1) * VOX_MM,
            "stack_ap_mm": n * VOX_MM,
            "tissue_frac_first_page_rel": float(df["area"].iloc[first] / df["area"].max()),
            "tissue_frac_last_page_rel": float(df["area"].iloc[last] / df["area"].max()),
            "atlas_ap_covered_10um": covered,
            "atlas_brain_extent_10um": list(ext),
            "missing_anterior_mm": max((covered[0] - ext[0]) * VOX_MM, 0.0) if np.isfinite(covered[0]) else np.nan,
            "missing_posterior_mm": max((ext[1] - covered[1]) * VOX_MM, 0.0) if np.isfinite(covered[1]) else np.nan,
            "affine_ap_scale": float(A[0, 0]),
            "squash_ratio_if_fit_whole": float((ext[1] - ext[0]) * VOX_MM / ((last - first + 1) * VOX_MM)),
            "mode": "overhang" if abs(A[0, 0] - 1) < 0.15 else "squash?"}


def lr_asymmetry(subject: str) -> pd.DataFrame:
    """(L-R)/(L+R) of region volumes from ``volumes.csv``, per structure and grouped.

    ``left_volume_mm3`` is hemisphere value 1 = high k = the implant side (checked
    against the atlas API and the hemisphere volume; brainreg's ``main.py`` passes
    the values explicitly).
    """
    p = subject_paths(subject)
    vol = pd.read_csv(p["volumes"])
    tree = structure_tree()
    vol["acronym"] = vol["structure_name"].map(tree["name2acro"]).fillna("?")
    vol["asym"] = (vol["left_volume_mm3"] - vol["right_volume_mm3"]) / vol["total_volume_mm3"].replace(0, np.nan)
    groups = {"ENTl": lambda a: a.startswith("ENTl"), "ENTm": lambda a: a.startswith("ENTm"),
              "SUB": lambda a: a == "SUB", "ProS": lambda a: a == "ProS", "CA1": lambda a: a == "CA1",
              "CA3": lambda a: a == "CA3", "DG": lambda a: a.startswith("DG"),
              "VIS": lambda a: a.startswith("VIS"), "TH": lambda a: a == "TH"}
    rows = []
    for g, fn in groups.items():
        sel = vol[vol["acronym"].map(fn)]
        L, R = sel["left_volume_mm3"].sum(), sel["right_volume_mm3"].sum()
        if L + R > 0:
            rows.append({"subject": subject, "group": g, "left_mm3": L, "right_mm3": R,
                         "asym": (L - R) / (L + R), "flag": abs((L - R) / (L + R)) > QC_THRESHOLDS["lr_asym"]})
    return pd.DataFrame(rows)


def hemisphere_check(subject: str) -> dict:
    """Assert hemisphere value 1 sits at high k and holds the recorded bank."""
    p = subject_paths(subject)
    geo = track_geometry(subject)
    page = int(round(geo["bank_centre"][0]))
    h = read_page(p["hemispheres"], page)
    a = read_page(p["atlas"], page) > 0
    k1 = np.nonzero(a & (h == 1))[1].mean() if (a & (h == 1)).any() else np.nan
    k2 = np.nonzero(a & (h == 2))[1].mean() if (a & (h == 2)).any() else np.nan
    df = contacts(subject)
    b = df[df["bank"]]
    c = np.clip(b[[f"downsample_coords.{x}" for x in "ijk"]].values.astype(int), 0, np.asarray(geo["shape"]) - 1)
    codes = []
    for pg in np.unique(c[:, 0]):
        hp = read_page(p["hemispheres"], pg)
        s = c[:, 0] == pg
        codes.extend(hp[c[s, 1], c[s, 2]].tolist())
    codes = np.asarray(codes)
    return {"mean_k_value1": float(k1), "mean_k_value2": float(k2), "width_k": int(h.shape[1]),
            "value1_is_high_k": bool(k1 > k2),
            "bank_frac_in_value1": float(np.mean(codes == 1)),
            "ok": bool(k1 > k2 and np.mean(codes == 1) > 0.95)}


def region_volume_ratios(subject: str) -> pd.DataFrame:
    """Sample / atlas volume for regions wholly inside the covered AP range.

    Compared against the global ``1/det(A)``; the ENTl ratio deviating from CA1's
    by more than ~15 % would mean a local squeeze at the target.
    """
    p = subject_paths(subject)
    A, _ = load_affine(subject)
    vol = pd.read_csv(p["volumes"])
    tree = structure_tree()
    vol["acronym"] = vol["structure_name"].map(tree["name2acro"]).fillna("?")
    atlas = atlas_region_volumes().set_index("acronym")
    trunc = truncation_report(subject)
    lo, hi = trunc["atlas_ap_covered_10um"]
    margin = 20
    groups = {"ENTl": lambda a: a.startswith("ENTl"), "ENTm": lambda a: a.startswith("ENTm"),
              "CA1": lambda a: a == "CA1", "SUB": lambda a: a == "SUB", "ProS": lambda a: a == "ProS",
              "DG": lambda a: a.startswith("DG-") or a == "DG", "TH": lambda a: a == "TH",
              "VIS": lambda a: a.startswith("VIS"), "SC": lambda a: a.startswith("SC"),
              "CB": lambda a: a in ("VERM", "HEM", "CBN") or a.startswith("CUL") or a.startswith("SIM")}
    rows = []
    for g, fn in groups.items():
        acros = [a for a in vol["acronym"] if isinstance(a, str) and fn(a) and a in atlas.index]
        if not acros:
            continue
        sample_v = vol.set_index("acronym").loc[acros, "total_volume_mm3"].sum()
        atlas_v = atlas.loc[acros, "atlas_volume_mm3"].sum()
        ap_min = atlas.loc[acros, "ap_min_10um"].min()
        ap_max = atlas.loc[acros, "ap_max_10um"].max()
        inside = bool(ap_min >= lo + margin and ap_max <= hi - margin)
        rows.append({"subject": subject, "group": g, "sample_mm3": sample_v, "atlas_mm3": atlas_v,
                     "ratio": sample_v / atlas_v if atlas_v else np.nan, "fully_inside": inside,
                     "expected_1_over_detA": 1.0 / float(np.linalg.det(A[:3, :3]))})
    return pd.DataFrame(rows)


def lr_profile(subject: str, *, stride: int = PROFILE_STRIDE) -> pd.DataFrame:
    """Per-page L/R and outline residuals: the yaw-vs-damage-vs-error discriminator.

    Columns per page (AP index ``i``):
      img_asym  -- (L-R)/(L+R) tissue area from the tissue-vs-agarose mask, split
                   by the registered hemisphere map (only the midline is borrowed)
      lab_asym  -- the same for the warped atlas footprint
      lr_res_um -- atlas outline midline minus tissue outline midline (dorsal rows)
      dv_res_um -- atlas top minus tissue top at the midline column
      int_ratio -- mean intensity L/R inside the mask (the illumination confound)

    Yaw appears as a *linear* trend shared by img and lab; damage as a *localised*
    band shared by both; registration error as lab != img.
    """
    p = subject_paths(subject)
    thr = float(_load_global(subject)["thr"]) if _load_global(subject) is not None and "thr" in _load_global(subject) else tissue_threshold(subject)
    n = n_pages(p["green"])
    pages = list(range(stride // 2, n - stride // 2, stride))
    g = read_pages(p["green"], pages[0], pages[-1] + 1, stride).astype(np.float32)
    a = read_pages(p["atlas"], pages[0], pages[-1] + 1, stride) > 0
    h = read_pages(p["hemispheres"], pages[0], pages[-1] + 1, stride)
    rows = []
    for q, pg in enumerate(pages):
        M = tissue_mask(g[q], thr)
        Aq = ndimage.binary_fill_holes(a[q])
        if M.sum() < 5000 or Aq.sum() < 5000:
            continue
        L = int((M & (h[q] == 1)).sum()); R = int((M & (h[q] == 2)).sum())
        aL = int((Aq & (h[q] == 1)).sum()); aR = int((Aq & (h[q] == 2)).sum())
        rows_t = np.where(M.sum(axis=1) > 150)[0]
        rows_t = rows_t[: max(len(rows_t) // 2, 1)]
        res = []
        for j in rows_t:
            kt = np.where(M[j])[0]; ka = np.where(Aq[j])[0]
            if len(ka):
                res.append(0.5 * (ka.min() + ka.max()) - 0.5 * (kt.min() + kt.max()))
        kmid = int(np.median([0.5 * (np.where(M[j])[0].min() + np.where(M[j])[0].max()) for j in rows_t]))
        jt = np.where(M[:, kmid])[0]; ja = np.where(Aq[:, kmid])[0]
        dv = (ja.min() - jt.min()) * VOX_UM if len(jt) and len(ja) else np.nan
        mL = float(g[q][M & (h[q] == 1)].mean()) if L else np.nan
        mR = float(g[q][M & (h[q] == 2)].mean()) if R else np.nan
        # illumination sensitivity: the same asymmetry at 1.5x the threshold.  Geometry is
        # threshold-invariant; an illumination artefact is not.
        M2 = tissue_mask(g[q], thr * 1.5)
        L2 = int((M2 & (h[q] == 1)).sum()); R2 = int((M2 & (h[q] == 2)).sum())
        rows.append({"i": pg, "ap_mm": pg * VOX_MM, "img_L": L, "img_R": R,
                     "img_asym": (L - R) / max(L + R, 1), "lab_asym": (aL - aR) / max(aL + aR, 1),
                     "img_asym_hi_thr": (L2 - R2) / max(L2 + R2, 1),
                     "lr_res_um": float(np.median(res)) * VOX_UM if res else np.nan,
                     "dv_res_um": dv, "int_ratio": mL / mR if (mR and np.isfinite(mR)) else np.nan})
    return pd.DataFrame(rows)


def residual_angles(profile: pd.DataFrame, *, central: float = 0.70) -> dict:
    """Residual yaw / pitch from robust (Theil-Sen) slopes of the outline residuals vs AP.

    Fitted over the central ``central`` fraction of tissue pages so the truncation
    faces (atlas overhang) do not enter; Theil-Sen because a single torn page can
    throw an OLS slope by more than a degree.
    """
    d = profile.dropna(subset=["lr_res_um"]).reset_index(drop=True)
    n = len(d)
    if n < 6:
        return {"residual_yaw_deg": np.nan, "residual_pitch_deg": np.nan}
    cut = int(n * (1 - central) / 2)
    d = d.iloc[cut: n - cut]
    x = d["ap_mm"].values * 1000.0
    ts_lr = theilslopes(d["lr_res_um"].values, x)
    dv = d.dropna(subset=["dv_res_um"])
    ts_dv = theilslopes(dv["dv_res_um"].values, dv["ap_mm"].values * 1000.0) if len(dv) >= 6 else (np.nan,) * 4
    return {"residual_yaw_deg": float(np.degrees(np.arctan(ts_lr[0]))),
            "residual_pitch_deg": float(np.degrees(np.arctan(ts_dv[0]))) if np.isfinite(ts_dv[0]) else np.nan,
            "lr_res_median_um": float(np.median(d["lr_res_um"])),
            "lr_res_iqr_um": float(np.subtract(*np.percentile(d["lr_res_um"], [75, 25]))),
            "dv_res_median_um": float(np.nanmedian(d["dv_res_um"])),
            "img_lab_asym_corr": float(np.corrcoef(d["img_asym"], d["lab_asym"])[0, 1]),
            "img_lab_asym_mad": float(np.median(np.abs(d["img_asym"] - d["lab_asym"]))),
            # threshold sensitivity of the image asymmetry (illumination gate): geometry is
            # invariant to the threshold, an illumination artefact is not.  A plain correlation
            # with the L/R intensity ratio is NOT a valid gate -- both trend with AP under a yaw
            # (ah08: r = 0.95 while the label footprint agrees with the mask at r = 0.89).
            "img_asym_thr_sensitivity": float(np.median(np.abs(d["img_asym"] - d["img_asym_hi_thr"]))) if "img_asym_hi_thr" in d else np.nan,
            "img_asym_vs_intensity_corr": float(pd.Series(d["img_asym"]).corr(pd.Series(d["int_ratio"])))}


def tile_artefact(subject: str, *, n_pages_avg: int = 21) -> dict:
    """Periodic column/row intensity steps from StitchIt tile borders in the registration channel.

    Averages a few coronal pages around the bank, takes the column-mean (k) and
    row-mean (j) profiles of the in-brain intensity, and reports the strongest
    periodic component of their first difference (tile pitch ~ 1.0-1.3 mm at
    10 um = 100-130 voxels).  Descriptive; a large value flags false edges that
    block-matching may lock onto.
    """
    p = subject_paths(subject)
    geo = track_geometry(subject)
    c = int(round(geo["bank_centre"][0]))
    g = read_pages(p["green"], c - n_pages_avg // 2, c + n_pages_avg // 2 + 1).astype(np.float32).mean(0)
    a = read_pages(p["atlas"], c - n_pages_avg // 2, c + n_pages_avg // 2 + 1) > 0
    m = a.mean(0) > 0.5
    out = {}
    for name, axis in (("k", 0), ("j", 1)):
        prof = np.where(m, g, np.nan)
        prof = np.nanmean(prof, axis=axis)
        prof = pd.Series(prof).interpolate(limit_direction="both").values
        prof = prof / np.nanmedian(prof)
        d = np.diff(prof)
        d = d - np.nanmean(d)
        f = np.fft.rfft(np.nan_to_num(d)); freqs = np.fft.rfftfreq(len(d))
        band = (freqs > 1 / 200) & (freqs < 1 / 60)      # 60-200 voxel periods
        if band.any():
            pk = np.argmax(np.abs(f[band]))
            out[f"period_{name}_vox"] = float(1 / freqs[band][pk])
            out[f"step_power_{name}"] = float(np.abs(f[band][pk]) / (np.abs(f).mean() + 1e-12))
        out[f"step_p99_{name}"] = float(np.nanpercentile(np.abs(d), 99))
    return out


# --------------------------------------------------------------------------
# Check A -- overlays
# --------------------------------------------------------------------------


def _contours(ax, label_page: np.ndarray, groups: dict, *, lw: float = 0.9, alpha: float = 0.9,
              transpose: bool = False):
    for g, colour in GROUP_COLOURS.items():
        ids = groups.get(g)
        if ids is None or not len(ids):
            continue
        mask = np.isin(label_page, ids)
        if transpose:
            mask = mask.T
        if mask.any():
            ax.contour(mask, levels=[0.5], colors=[colour], linewidths=lw, alpha=alpha)


def coronal_levels(subject: str, n: int = 8) -> list[int]:
    first, last = tissue_pages(subject)
    geo = track_geometry(subject)
    c = int(round(geo["bank_centre"][0]))
    inner = np.linspace(first + 5, last - 5, n - 1).round().astype(int).tolist()[1:-1]
    levels = sorted(set([first + 5, c, last - 5] + inner))
    return [int(np.clip(l, 0, geo["shape"][0] - 1)) for l in levels][:n]


def plot_sample_overlays(subject: str, *, levels: list[int] | None = None, save: bool = True):
    """Whole-brain sample-space overlays: green autofluorescence + every atlas edge + group contours.

    Coronal pages at ``levels`` (first/last tissue page included so the atlas
    overhang at the truncation faces is *seen*), plus sagittal and horizontal views
    built from every 4th page (40 um AP sampling is plenty for an overview).
    """
    import matplotlib.pyplot as plt
    p = subject_paths(subject)
    groups = layer_id_sets()
    levels = levels or coronal_levels(subject)
    geo = track_geometry(subject)
    ncol = 4
    nrow = int(np.ceil(len(levels) / ncol)) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow))
    axes = axes.ravel()
    for ax, lv in zip(axes, levels):
        g = read_page(p["green"], lv)
        a = read_page(p["atlas"], lv)
        b = read_page(p["boundaries"], lv)
        d0 = read_page(p["def"][0], lv)
        ax.imshow(puf.adjust_contrast(g), cmap="gray", origin="upper")
        edge = np.ma.masked_where(b == 0, b)
        ax.imshow(edge, cmap="autumn", alpha=0.35, origin="upper", interpolation="nearest")
        _contours(ax, a, groups)
        ap_atlas = np.median(d0[a > 0]) / VOX_MM if (a > 0).any() else np.nan
        ax.set_title(f"coronal i={lv}  (atlas AP ≈ {ap_atlas * VOX_MM:.2f} mm)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(levels): (nrow - 1) * ncol]:
        ax.axis("off")
    # sagittal / horizontal from every 4th page
    step = 4
    g4 = read_pages(p["green"], 0, geo["shape"][0], step)
    a4 = read_pages(p["atlas"], 0, geo["shape"][0], step)
    kc = int(round(geo["bank_centre"][2])); jc = int(round(geo["bank_centre"][1]))
    kmid = geo["shape"][2] // 2
    views = [("sagittal through bank k=%d" % kc, g4[:, :, kc].T, a4[:, :, kc].T),
             ("sagittal midline k=%d" % kmid, g4[:, :, kmid].T, a4[:, :, kmid].T),
             ("horizontal through bank j=%d" % jc, g4[:, jc, :].T, a4[:, jc, :].T),
             ("horizontal j=%d" % (geo["shape"][1] // 3), g4[:, geo["shape"][1] // 3, :].T, a4[:, geo["shape"][1] // 3, :].T)]
    for ax, (title, img, lab) in zip(axes[(nrow - 1) * ncol:], views):
        ax.imshow(puf.adjust_contrast(img), cmap="gray", origin="upper", aspect=1 / step)
        _contours(ax, lab, groups)
        ax.set_title(title + f"  (every {step}th page)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], color=c, label=g) for g, c in GROUP_COLOURS.items()]
                   + [Line2D([], [], color="#ff7f00", alpha=0.5, label="all label edges")],
                   loc="lower left", fontsize=6, framealpha=0.6)
    fig.suptitle(f"{subject} — registered atlas over autofluorescence (sample space, asr; image-left = RIGHT hemisphere)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / f"{subject}_overlay_whole.png", dpi=130, bbox_inches="tight")
        print(f"saved {FIGURE_DIR / f'{subject}_overlay_whole.png'}")
    return fig


def plot_lec_zoom(subject: str, *, half_um: float = 1500.0, slab_um: float = 400.0, save: bool = True):
    """The figure Adam reads: LEC zoom through the recorded bank, three planes.

    Green autofluorescence, DiI slab MIP in red, atlas contours from the *centre
    slice* (never the MIP) with ENTl superficial vs deep in two blues, fibre tracts
    in yellow; contacts red, recorded bank yellow.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    p = subject_paths(subject)
    groups = layer_id_sets()
    geo = track_geometry(subject)
    df = contacts(subject)
    coords = df[[f"downsample_coords.{c}" for c in "ijk"]].values
    bank = df["bank"].values
    centre = np.round(geo["bank_centre"]).astype(int)
    half = int(half_um / VOX_UM); slab = int(slab_um / VOX_UM)
    lo, hi = max(centre[0] - half, 0), min(centre[0] + half + 1, geo["shape"][0])
    g = green_slab(subject, lo, hi); a = atlas_slab(subject, lo, hi)
    dye = read_pages(p["dye"], lo, hi)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    specs = [("coronal", 0), ("sagittal", 2), ("horizontal", 1)]
    for ax, (name, axis) in zip(axes, specs):
        c = centre[axis]
        if axis == 0:
            ci = c - lo
            img = g[ci]; lab = a[ci]
            mip = dye[max(ci - slab // 2, 0): ci + slab // 2 + 1].max(0)
            xi, yi = 2, 1
            xoff, yoff = 0, 0
        elif axis == 2:
            img = g[:, :, c].T; lab = a[:, :, c].T
            mip = dye[:, :, max(c - slab // 2, 0): c + slab // 2 + 1].max(2).T
            xi, yi = 0, 1
            xoff, yoff = lo, 0
        else:
            img = g[:, c, :].T; lab = a[:, c, :].T
            mip = dye[:, max(c - slab // 2, 0): c + slab // 2 + 1, :].max(1).T
            xi, yi = 0, 2
            xoff, yoff = lo, 0
        ax.imshow(puf.adjust_contrast(img), cmap="gray", origin="upper")
        red = puf.adjust_contrast(mip, 50, 99.8)
        ax.imshow(np.ma.masked_where(red < 0.35, red), cmap="Reds", alpha=0.55, origin="upper", vmin=0, vmax=1)
        _contours(ax, lab, groups, lw=1.1)
        ax.scatter(coords[:, xi] - xoff, coords[:, yi] - yoff, s=3, c="#FF2D2D", alpha=0.5, lw=0)
        ax.scatter(coords[bank, xi] - xoff, coords[bank, yi] - yoff, s=7, c="#FFE800", alpha=0.95, lw=0)
        cx, cy = centre[xi] - xoff, centre[yi] - yoff
        H, W = img.shape
        x0, x1 = max(cx - half, 0), min(cx + half, W - 1)
        y0, y1 = max(cy - half, 0), min(cy + half, H - 1)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}  (DiI {slab_um:.0f} µm MIP in red; contours from centre slice)", fontsize=9)
    axes[0].legend(handles=[Line2D([], [], color=c, label=g) for g, c in GROUP_COLOURS.items()]
                   + [Line2D([], [], marker="o", ls="", color="#FFE800", label="recorded bank"),
                      Line2D([], [], marker="o", ls="", color="#FF2D2D", label="contacts")],
                   loc="lower left", fontsize=7, framealpha=0.6)
    fig.suptitle(f"{subject} — LEC zoom, ±{half_um / 1000:.1f} mm around the recorded bank "
                 f"(fit: {geo['fit_method']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / f"{subject}_overlay_lec.png", dpi=140, bbox_inches="tight")
        print(f"saved {FIGURE_DIR / f'{subject}_overlay_lec.png'}")
    return fig


def plot_atlas_space_checker(subject: str, *, n_levels: int = 4, tiles: int = 8, save: bool = True):
    """Checkerboard of the sample warped into atlas space vs the unfiltered atlas reference.

    Caveat printed on the figure: ``downsampled_standard`` was made with a
    separately estimated *inverse* warp, so this tests that warp, not the forward
    one that placed the labels.
    """
    import matplotlib.pyplot as plt
    p = subject_paths(subject)
    trunc = truncation_report(subject)
    lo_a, hi_a = trunc["atlas_ap_covered_10um"]
    geo = track_geometry(subject)
    d0 = read_page(p["def"][0], int(round(geo["bank_centre"][0])))
    a = read_page(p["atlas"], int(round(geo["bank_centre"][0]))) > 0
    bank_atlas = float(np.median(d0[a])) / VOX_MM
    levels = sorted(set(np.linspace(lo_a + 60, hi_a - 60, n_levels - 1).round().astype(int).tolist() + [int(round(bank_atlas))]))
    fig, axes = plt.subplots(2, len(levels), figsize=(4.8 * len(levels), 8))
    for col, lv in enumerate(levels):
        std = puf.adjust_contrast(read_page(p["standard"], lv))
        ref = puf.adjust_contrast(atlas_reference_page(lv))
        H, W = std.shape
        ref = ref[:H, :W]
        yy, xx = np.mgrid[:H, :W]
        board = ((yy // (H // tiles)) + (xx // (W // tiles))) % 2 == 0
        axes[0, col].imshow(np.where(board, std, ref), cmap="gray", origin="upper")
        axes[0, col].set_title(f"atlas i={lv} checkerboard (sample-warped ⇄ atlas)", fontsize=9)
        stripes = (yy // max(H // (tiles * 3), 1)) % 2 == 0
        axes[1, col].imshow(np.where(stripes, std, ref), cmap="gray", origin="upper")
        axes[1, col].set_title("horizontal stripes", fontsize=9)
        for ax in axes[:, col]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{subject} — sample warped into atlas space vs unfiltered atlas reference "
                 f"(tests the separately estimated INVERSE warp)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / f"{subject}_checker_atlas.png", dpi=120, bbox_inches="tight")
        print(f"saved {FIGURE_DIR / f'{subject}_checker_atlas.png'}")
    return fig


def plot_lr_profile(subject: str, profile: pd.DataFrame | None = None, save: bool = True):
    """Per-page image vs label L/R asymmetry and the outline residuals along AP."""
    import matplotlib.pyplot as plt
    prof = profile if profile is not None else lr_profile(subject)
    first, last = tissue_pages(subject)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(prof["ap_mm"], prof["img_asym"], "o-", ms=3, label="image tissue mask")
    axes[0].plot(prof["ap_mm"], prof["lab_asym"], "s-", ms=3, label="warped atlas labels")
    axes[0].axhline(0, color="k", lw=0.5); axes[0].set_ylabel("(L−R)/(L+R)"); axes[0].legend(fontsize=8)
    axes[1].plot(prof["ap_mm"], prof["lr_res_um"], "o-", ms=3, color="C2")
    axes[1].axhline(0, color="k", lw=0.5); axes[1].set_ylabel("atlas − tissue\nmidline (µm)")
    axes[1].set_ylim(-300, 300)
    axes[2].plot(prof["ap_mm"], prof["dv_res_um"], "o-", ms=3, color="C3")
    axes[2].axhline(0, color="k", lw=0.5); axes[2].set_ylabel("atlas − tissue\ntop (µm)"); axes[2].set_xlabel("sample AP (mm)")
    axes[2].set_ylim(-300, 300)
    for ax in axes:
        ax.axvspan(0, first * VOX_MM, color="grey", alpha=0.15)
        ax.axvspan(last * VOX_MM, prof["ap_mm"].max() + 0.1, color="grey", alpha=0.15)
    ra_ = residual_angles(prof)
    fig.suptitle(f"{subject} — per-page L/R and outline residuals; residual yaw {ra_['residual_yaw_deg']:+.2f}°, "
                 f"pitch {ra_['residual_pitch_deg']:+.2f}° (grey = beyond tissue)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / f"{subject}_lr_profile.png", dpi=130, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Check C -- deformation-field plausibility
# --------------------------------------------------------------------------


def _det3(J):
    """Determinant of a field of 3x3 matrices given as J[c][a] arrays (cofactor form)."""
    a, b, c = J[0]
    d, e, f = J[1]
    g, h, i = J[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def jacobian_det(dfield: np.ndarray, *, spacing_mm: float = VOX_MM) -> np.ndarray:
    """det J of a (3, n, J, K) deformation field in mm on a 10 um grid (dimensionless).

    ``J[c, a] = d D_c / d x_a``.  det J = local atlas volume / sample volume, so
    ~det(A) on average (>1: the sample is compressed relative to the atlas), and
    < 0 means folding.  Computed with the explicit cofactor formula on nine arrays
    -- never ``np.linalg.det`` on an (n, J, K, 3, 3) stack.
    """
    J = [[np.gradient(dfield[c], spacing_mm, axis=a).astype(np.float32) for a in range(3)] for c in range(3)]
    return _det3(J)


def jacobian_check(subject: str, *, near_um: float = 1000.0) -> dict:
    """det J statistics within ``near_um`` of the recorded bank, plus centre maps."""
    geo = track_geometry(subject)
    A, _ = load_affine(subject)
    lo, hi = geo["slab"]
    lo2, hi2 = max(lo - 1, 0), min(hi + 1, geo["shape"][0])
    dfield = def_slab(subject, lo2, hi2)
    off = lo - lo2                                   # 1 when a guard page was prepended
    detj = jacobian_det(dfield)[off: off + (hi - lo)]
    atl = atlas_slab(subject, lo, hi) > 0
    n = min(detj.shape[0], atl.shape[0])
    detj = detj[:n]; atl = atl[:n]
    # near-bank mask: box around the bank bbox
    r = int(near_um / VOX_UM)
    bmin = np.floor(geo["bank_min"]).astype(int) - r; bmax = np.ceil(geo["bank_max"]).astype(int) + r
    box = np.zeros_like(atl)
    box[max(bmin[0] - lo, 0): bmax[0] - lo + 1, max(bmin[1], 0): bmax[1] + 1, max(bmin[2], 0): bmax[2] + 1] = True
    m = atl & box
    vals = detj[m]
    detA = float(np.linalg.det(A[:3, :3]))
    grad = np.hypot(*np.gradient(detj, axis=(1, 2)))
    stats = {"detj_median": float(np.median(vals)), "detj_p05": float(np.percentile(vals, 5)),
             "detj_p95": float(np.percentile(vals, 95)), "detj_min": float(vals.min()), "detj_max": float(vals.max()),
             "detj_frac_outside": float(np.mean((vals < QC_THRESHOLDS["detj_range"][0]) | (vals > QC_THRESHOLDS["detj_range"][1]))),
             "detj_n_negative": int((vals < 0).sum()), "det_affine": detA,
             "nonlinear_median": float(np.median(vals) / detA),
             "detj_grad_p99": float(np.percentile(grad[m], 99)), "n_voxels": int(m.sum())}
    c = np.round(geo["bank_centre"]).astype(int)
    stats["maps"] = {"coronal": detj[c[0] - lo], "sagittal": detj[:, :, c[2]].T, "horizontal": detj[:, c[1], :].T,
                     "slab_lo": lo}
    return stats


def plot_jacobian(subject: str, stats: dict | None = None, save: bool = True):
    import matplotlib.pyplot as plt
    st = stats or jacobian_check(subject)
    geo = track_geometry(subject)
    df = contacts(subject); coords = df[[f"downsample_coords.{c}" for c in "ijk"]].values; bank = df["bank"].values
    groups = layer_id_sets()
    lo, hi = geo["slab"]; a = atlas_slab(subject, lo, hi)
    c = np.round(geo["bank_centre"]).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    half = 150
    for ax, name, (xi, yi, xoff) in zip(axes, ("coronal", "sagittal", "horizontal"), ((2, 1, 0), (0, 1, lo), (0, 2, lo))):
        img = st["maps"][name]
        lab = {"coronal": a[c[0] - lo], "sagittal": a[:, :, c[2]].T, "horizontal": a[:, c[1], :].T}[name]
        im = ax.imshow(np.ma.masked_where(lab == 0, img), cmap="RdBu_r", vmin=0.5, vmax=1.5, origin="upper")
        _contours(ax, lab, groups, lw=0.8)
        ax.scatter(coords[bank, xi] - xoff, coords[bank, yi], s=5, c="k", lw=0)
        cx, cy = c[xi] - xoff, c[yi]
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"det J — {name}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.6, label="det J (atlas vol / sample vol)")
    fig.suptitle(f"{subject} — Jacobian determinant near the bank: median {st['detj_median']:.2f} "
                 f"(affine {st['det_affine']:.2f}), outside [0.5,2]: {100 * st['detj_frac_outside']:.2f}%, folds: {st['detj_n_negative']}",
                 fontsize=11)
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / f"{subject}_jacobian.png", dpi=130, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Check D -- label fragility
# --------------------------------------------------------------------------


def interface_voxels(label_slab: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    """(N, 3) voxels of ``a`` that touch ``b`` (6-connectivity), plus ``b`` touching ``a``."""
    ma = np.isin(label_slab, ids_a); mb = np.isin(label_slab, ids_b)
    st = ndimage.generate_binary_structure(3, 1)
    face = (ma & ndimage.binary_dilation(mb, st)) | (mb & ndimage.binary_dilation(ma, st))
    return np.argwhere(face)


def contact_boundary_distances(subject: str) -> pd.DataFrame:
    """Per recorded-bank contact: distance to the ENTl sup/deep interface and to any label edge (um)."""
    p = subject_paths(subject)
    geo = track_geometry(subject)
    groups = layer_id_sets()
    df = contacts(subject)
    b = df[df["bank"]].copy()
    coords = b[[f"downsample_coords.{c}" for c in "ijk"]].values
    lo, hi = geo["slab"]
    a = atlas_slab(subject, lo, hi)
    # restrict to a box around the bank (+/- 100 vox) for the KD trees
    r = 100
    bmin = np.maximum(np.floor(coords.min(0)).astype(int) - r, [lo, 0, 0]); bmax = np.ceil(coords.max(0)).astype(int) + r
    sub = a[bmin[0] - lo: bmax[0] - lo + 1, bmin[1]: bmax[1] + 1, bmin[2]: bmax[2] + 1]
    off = np.array([bmin[0], bmin[1], bmin[2]])
    iface = interface_voxels(sub, groups["ENTl_sup"], groups["ENTl_deep"]) + off
    # LEC/MEC border: the ENTl <-> ENTm interface.  It runs along the cortical sheet,
    # so unlike the sup/deep boundary it has no visible intensity edge to re-read
    # against; the distance to it is therefore the whole measurement -- a contact
    # within d um would change region if the border were d um off.
    iface_lm = interface_voxels(sub, groups["ENTl"], groups["ENTm"]) + off
    bnd = read_pages(p["boundaries"], bmin[0], bmax[0] + 1)[:, bmin[1]: bmax[1] + 1, bmin[2]: bmax[2] + 1]
    edges = np.argwhere(bnd == 1) + off
    b["d_sup_deep_um"] = cKDTree(iface).query(coords)[0] * VOX_UM if len(iface) else np.nan
    b["d_lec_mec_um"] = cKDTree(iface_lm).query(coords)[0] * VOX_UM if len(iface_lm) else np.inf
    b["d_any_boundary_um"] = cKDTree(edges).query(coords)[0] * VOX_UM if len(edges) else np.nan
    b["n_interface_voxels"] = len(iface)
    b["n_lecmec_interface_voxels"] = len(iface_lm)
    return b.reset_index(drop=True)


def lec_mec_by_shank(dist: pd.DataFrame) -> pd.DataFrame:
    """Per shank: how many contacts are ENTl / ENTm and how many sit within 100 / 200 um of the border."""
    acro = dist["structure.acronym"].astype(str)
    rows = []
    for sh, s in dist.groupby("shank"):
        a = s["structure.acronym"].astype(str)
        rows.append({"shank": int(sh), "n": len(s),
                     "ENTl": int(a.str.startswith("ENTl").sum()), "ENTm": int(a.str.startswith("ENTm").sum()),
                     "median_d_lecmec_um": float(np.median(s["d_lec_mec_um"])),
                     "within_100um": int((s["d_lec_mec_um"] <= 100).sum()),
                     "within_200um": int((s["d_lec_mec_um"] <= 200).sum())})
    return pd.DataFrame(rows)


def fragility_summary(dist: pd.DataFrame) -> dict:
    ent = dist[dist["group"].isin(["ENTl-sup", "ENTl-deep"])]
    out = {"n_bank": int(len(dist)), "n_bank_ENTl": int(len(ent))}
    for r in (50, 100, 150):
        out[f"frac_ENTl_within_{r}um_of_supdeep"] = float(np.mean(ent["d_sup_deep_um"] <= r)) if len(ent) else np.nan
        out[f"frac_bank_within_{r}um_of_any_edge"] = float(np.mean(dist["d_any_boundary_um"] <= r))
    out["fragile_advisory"] = bool(out.get("frac_ENTl_within_100um_of_supdeep", 0) > QC_THRESHOLDS["fragile_frac_100um"])
    # LEC/MEC border fragility: counts of the whole recorded bank near the ENTl/ENTm
    # interface, split by which side they currently sit on.
    acro = dist["structure.acronym"].astype(str)
    is_l, is_m = acro.str.startswith("ENTl"), acro.str.startswith("ENTm")
    d = dist["d_lec_mec_um"] if "d_lec_mec_um" in dist else pd.Series(np.inf, index=dist.index)
    out["n_bank_ENTm"] = int(is_m.sum())
    for r in (50, 100, 150, 200):
        sel = d <= r
        out[f"n_within_{r}um_of_lecmec"] = int(sel.sum())
        out[f"frac_bank_within_{r}um_of_lecmec"] = float(sel.mean())
    sel150 = d <= 150
    out["n_within_150um_lecmec_from_ENTl"] = int((sel150 & is_l).sum())
    out["n_within_150um_lecmec_from_ENTm"] = int((sel150 & is_m).sum())
    out["lecmec_advisory"] = bool(out["frac_bank_within_100um_of_lecmec"] > QC_THRESHOLDS["lecmec_frac_100um"])
    return out


def label_self_consistency(subject: str, *, perturb: str | None = None) -> dict:
    """Registered label at the sample voxel vs the native annotation at its atlas coordinate.

    Validates the stored Allen coordinates (deformation-field lookup), not the
    registration: both are the same warp.  ``perturb`` = 'permute' swaps the atlas
    axes, 'shift' adds +100 um along DV -- the positive controls.
    """
    df = contacts(subject)
    b = df[df["bank"]]
    coords = b[[f"downsample_coords.{c}" for c in "ijk"]].values
    mm = sample_vox_to_atlas_mm(subject, coords)
    if perturb == "permute":
        mm = mm[:, [2, 1, 0]]
    elif perturb == "shift":
        mm = mm + np.array([0, 0.1, 0])
    idx = np.round(mm / VOX_MM).astype(int)
    idx = np.clip(idx, 0, np.asarray(ATLAS_SHAPE) - 1)
    ann = nii_proxy(subject_paths(subject)["annotations"])
    native = np.zeros(len(idx), dtype=np.int64)
    for pg in np.unique(idx[:, 0]):
        sel = idx[:, 0] == pg
        page = np.asarray(ann[pg])
        native[sel] = page[idx[sel, 1], idx[sel, 2]]
    reg = b["structure.id"].values.astype(np.int64)
    dist = contact_boundary_distances(subject) if perturb is None else None
    floor = float(np.mean(dist["d_any_boundary_um"] <= VOX_UM)) if dist is not None else np.nan
    return {"disagreement": float(np.mean(native != reg)), "boundary_floor": floor,
            "n": int(len(reg)), "perturb": perturb or "none",
            "ok": bool(np.mean(native != reg) <= floor + QC_THRESHOLDS["self_consistency_extra"]) if perturb is None else None}


# --------------------------------------------------------------------------
# Check G -- along-track coherence
# --------------------------------------------------------------------------

_LAMINAR = {"ENTl1": 1, "ENTl2": 2, "ENTl2a": 2, "ENTl2b": 2, "ENTl2/3": 2.5, "ENTl3": 3,
            "ENTl5": 5, "ENTl6a": 6, "ENTl6b": 6.5}


def laminar_ordinal(acronym) -> float:
    return _LAMINAR.get(acronym, np.nan) if isinstance(acronym, str) else np.nan


def laminar_coherence(subject: str) -> pd.DataFrame:
    """Per shank: label changes, short flicker runs and ordinal reversals tip -> surface."""
    df = contacts(subject)
    b = df[df["bank"]].copy()
    rows = []
    for sh, s in b.groupby("shank"):
        s = s.sort_values("probe_coords.y")
        # collapse the two contact columns per row by taking the deeper x? keep both, order by y then x
        labs = s["structure.acronym"].astype(str).values
        runs = [len(list(g)) for _, g in __import__("itertools").groupby(labs)]
        ords = pd.Series([laminar_ordinal(a) for a in labs]).dropna().values
        rev = int(np.sum(np.diff(ords) < 0)) if len(ords) > 1 else 0
        rows.append({"subject": subject, "shank": int(sh), "n": len(s), "n_label_changes": len(runs) - 1,
                     "n_flicker_runs": int(np.sum(np.asarray(runs) < 3)) if len(runs) > 1 else 0,
                     "frac_in_flicker": float(np.sum([r for r in runs if r < 3]) / len(s)) if len(runs) > 1 else 0.0,
                     "n_ordinal_reversals": rev, "tip_label": labs[0], "top_label": labs[-1],
                     "sequence": "→".join(k for k, _ in __import__("itertools").groupby(labs))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Check E -- laminar edges along the surface normal
# --------------------------------------------------------------------------

S_RANGE_UM = (-300.0, 1500.0)
S_STEP_UM = 5.0


def surface_normals(mask: np.ndarray, pts: np.ndarray, *, sigma: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Nearest surface voxel of ``mask`` for each point and the INWARD unit normal there.

    Inward = gradient of the smoothed mask (mask is 1 inside), so ``p0 + s*n`` with
    ``s > 0`` walks into the tissue and ``s < 0`` out into the agarose.
    """
    st = ndimage.generate_binary_structure(3, 1)
    edge = mask & ~ndimage.binary_erosion(mask, st)
    ev = np.argwhere(edge)
    tree = cKDTree(ev)
    _, ix = tree.query(pts)
    p0 = ev[ix].astype(float)
    sm = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    grads = np.gradient(sm)
    n = np.stack([g[tuple(p0.astype(int).T)] for g in grads], 1)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-9)
    return p0, n


def line_profiles(vol: np.ndarray, p0: np.ndarray, n: np.ndarray, *, vox_um: float = VOX_UM,
                  s_range=S_RANGE_UM, step=S_STEP_UM, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``vol`` along ``p0 + (s/vox_um) n`` for every point; returns (s_um, values[n_pts, n_s])."""
    s = np.arange(s_range[0], s_range[1] + step / 2, step)
    out = np.empty((len(p0), len(s)), dtype=np.float32 if order else vol.dtype)
    for q in range(len(p0)):
        pts = p0[q][:, None] + (s[None, :] / vox_um) * n[q][:, None]
        out[q] = ndimage.map_coordinates(vol, pts, order=order, mode="nearest")
    return s, out


def detect_edges(s: np.ndarray, prof: np.ndarray, *, smooth_um: float = 20.0, step: float = S_STEP_UM,
                 min_rel_prominence: float = 0.15) -> pd.DataFrame:
    """Sign-agnostic step edges of one intensity profile: peaks of |d/ds| of the smoothed trace.

    Prominence is relative to the profile's 5-95 percentile range; sign +1 =
    dark->bright going deeper, -1 = bright->dark.
    """
    from scipy.signal import find_peaks
    prof = np.asarray(prof, float)
    sig = smooth_um / step
    sm = ndimage.gaussian_filter1d(prof, sig)
    d = np.gradient(sm, step)
    rng = np.subtract(*np.percentile(sm, [95, 5])) + 1e-9
    scale = rng / 100.0                              # a full-range step over 100 um
    pk, props = find_peaks(np.abs(d), prominence=min_rel_prominence * scale)
    # Noise floor.  On a flat trace the 5-95 range IS the noise, so a relative
    # prominence alone passes every wiggle (38 false edges on a flat synthetic).
    # Require the step amplitude -- mean 20-80 um after minus mean 20-80 um before
    # -- to exceed 5 x the noise of the smoothed trace, estimated robustly from the
    # first difference of the raw profile.
    sigma_raw = 1.4826 * np.median(np.abs(np.diff(prof))) / np.sqrt(2) + 1e-9
    sigma_sm = sigma_raw / np.sqrt(2 * np.sqrt(np.pi) * sig)
    w0, w1 = int(20 / step), int(80 / step)
    keep, amps = [], []
    for q in pk:
        a, b = sm[max(q - w1, 0): max(q - w0, 1)], sm[q + w0: q + w1 + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        amp = abs(b.mean() - a.mean())
        if amp >= max(min_rel_prominence * rng, 5 * sigma_sm):
            keep.append(q); amps.append(amp)
    keep = np.asarray(keep, dtype=int)
    return pd.DataFrame({"s_um": s[keep], "sign": np.sign(d[keep]).astype(int),
                         "strength": np.abs(d[keep]) / scale, "amplitude": amps})


#: Half-width of the search window for the visible grey/white dip around the atlas's
#: ENTl exit.  Registration errors up to this are measured without bias; larger ones
#: return NaN.  Three times the tolerance that matters (100 um).
WM_SEARCH_UM = 300.0


def _dip_edge(s: np.ndarray, sm: np.ndarray, prof: np.ndarray, s_lo: float, s_hi: float,
              *, min_rel_depth: float = 0.08, smooth_sig: float = 20.0 / S_STEP_UM) -> float:
    """Steepest descent into the deepest (most prominent) significant dip in ``[s_lo, s_hi]``.

    Dip significance: prominence >= max(5 x smoothed-trace noise, ``min_rel_depth`` x
    the grey plateau level).  Returns NaN when there is no dip (e.g. ENTl abutting
    grey matter, ah09), which is the right answer -- no grey/white edge to compare.
    """
    from scipy.signal import find_peaks
    sigma_raw = 1.4826 * np.median(np.abs(np.diff(prof))) / np.sqrt(2) + 1e-9
    sigma_sm = sigma_raw / np.sqrt(2 * np.sqrt(np.pi) * smooth_sig)
    win = (s >= s_lo) & (s <= s_hi)
    if win.sum() < 10:
        return np.nan
    plateau = float(np.median(sm[(s > s_lo - 300) & (s < s_lo)])) if ((s > s_lo - 300) & (s < s_lo)).any() else float(np.median(sm[win]))
    prom = max(5 * sigma_sm, min_rel_depth * plateau)
    idx = np.where(win)[0]
    mins, props = find_peaks(-sm[idx], prominence=prom)
    if not len(mins):
        return np.nan
    best = int(np.argmax(props["prominences"]))
    q_min = idx[mins[best]]
    left = int(props["left_bases"][best]) + idx[0]
    left = max(left, q_min - int(250 / S_STEP_UM))
    d = np.gradient(sm, S_STEP_UM)
    seg = np.arange(left, q_min + 1)
    if len(seg) < 2:
        return float(s[q_min])
    return float(s[seg[np.argmin(d[seg])]])


def atlas_edges_along_line(s: np.ndarray, labels: np.ndarray, groups: dict) -> dict:
    """Atlas pia (first non-root label), first sup->deep transition, ENTl -> anything else."""
    in_brain = labels > 0
    sup = np.isin(labels, groups["ENTl_sup"]); deep = np.isin(labels, groups["ENTl_deep"])
    ent = sup | deep
    out = {"pia": np.nan, "sup_deep": np.nan, "l12": np.nan, "ent_exit": np.nan, "exit_label": None,
           "ent_span_um": float(ent.sum() * (s[1] - s[0]))}
    if in_brain.any():
        out["pia"] = float(s[np.argmax(in_brain)])
    l1 = np.isin(labels, groups["ENTl_sup"][np.isin(groups["ENTl_sup"], [structure_tree()["acro2id"].get("ENTl1", -1)])])
    if l1.any():
        after_l1 = np.where(~l1[np.argmax(l1):] & ent[np.argmax(l1):])[0]
        if len(after_l1):
            out["l12"] = float(s[np.argmax(l1) + after_l1[0]])
    if ent.any():
        first_ent = np.argmax(ent)
        after = np.where(~ent[first_ent:])[0]
        if len(after):
            ix = first_ent + after[0]
            out["ent_exit"] = float(s[ix]); out["exit_label"] = int(labels[ix])
        if sup.any() and deep.any():
            first_sup = np.argmax(sup)
            d_after = np.where(deep[first_sup:])[0]
            if len(d_after):
                out["sup_deep"] = float(s[first_sup + d_after[0]])
    return out


def laminar_lines(green: np.ndarray, labels: np.ndarray, pts: np.ndarray, groups: dict, *,
                  vox_um: float = VOX_UM, thr: float | None = None, label_shift_um: float = 0.0,
                  brain_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Core of check E on arrays: one row per point with visible and atlas edges along the normal.

    ``label_shift_um`` reads the labels at ``s + shift`` -- the synthetic gate's
    planted perturbation (a positive shift moves every atlas boundary *deeper*).
    Visible pia = first crossing of ``thr`` (tissue threshold); visible WM = the
    strongest bright->dark step at least 350 um below the pia (fibre tracts are dark
    here); visible L1/L2 = strongest dark->bright step 50-250 um below the pia.
    """
    mask = brain_mask if brain_mask is not None else (labels > 0)
    p0, n = surface_normals(mask, pts)
    s, g = line_profiles(green, p0, n, vox_um=vox_um, order=1)
    s_lab, lab = line_profiles(labels, p0, n, vox_um=vox_um, order=0)
    if label_shift_um:
        shift = int(round(label_shift_um / S_STEP_UM))
        lab = np.roll(lab, shift, axis=1)          # positive shift => boundaries appear deeper
        if shift > 0:
            lab[:, :shift] = 0
        else:
            lab[:, shift:] = lab[:, [shift - 1]]
    if thr is None:
        thr = float(np.percentile(g, 20) * 0.5 + np.percentile(g, 80) * 0.5) * 0.35
    rows = []
    for q in range(len(pts)):
        prof = g[q]; ae = atlas_edges_along_line(s, lab[q], groups)
        sm = ndimage.gaussian_filter1d(prof.astype(float), 20.0 / S_STEP_UM)
        # Visible pia = HALF-MAXIMUM crossing between the agarose level and the
        # tissue plateau.  A low tissue threshold crosses at the *foot* of the
        # blurred edge, ~1.5 blur-sigma outside the true edge: it read +50 um on the
        # sample, on the atlas AND on a label-derived synthetic (the truth negative),
        # which is how the bias was caught.  Half-max is unbiased for a symmetric
        # blurred step.
        above = np.where(sm > thr)[0]
        pia_vis = np.nan
        if len(above):
            q0 = above[0]
            agarose = float(np.median(sm[: max(q0 - int(100 / S_STEP_UM), 1)])) if q0 > 2 else float(sm[0])
            plateau = float(np.median(sm[q0: q0 + int(300 / S_STEP_UM)]))
            half = 0.5 * (agarose + plateau)
            cross = np.where(sm[: q0 + int(300 / S_STEP_UM)] > half)[0]
            if len(cross):
                c0 = cross[0]
                if c0 > 0 and sm[c0] != sm[c0 - 1]:          # linear sub-sample interpolation
                    frac = (half - sm[c0 - 1]) / (sm[c0] - sm[c0 - 1])
                    pia_vis = float(s[c0 - 1] + frac * S_STEP_UM)
                else:
                    pia_vis = float(s[c0])
        ed = detect_edges(s, prof)
        wm_vis = np.nan; l12_vis = np.nan
        if np.isfinite(pia_vis):
            # Grey/white edge = steepest descent into the DEEPEST dark dip within
            # +-WM_SEARCH_UM of the atlas's ENTl exit.  The fibre band under ENTl
            # (angular bundle / external capsule) is a ~100-200 um dark band, i.e. a
            # local minimum, not a step to a new plateau.  Two blind rules both failed:
            # the *strongest* bright->dark step picked deeper hippocampal edges on
            # ah10/ly05/ly07 (1018-1205 um vs the real dip at 680-780 um; fictitious 0.6
            # thickness ratios), and the *first* significant dip picked the genuine
            # ENTl L5->L6 darkening on ah08 (~480 um; L6 is ~13 % darker than L5 here).
            # The window is an honest prior: errors up to WM_SEARCH_UM are measured
            # without bias, a larger one returns NaN (no dip) and shows as a collapse
            # of n_with_wm.  The +-100 um planted-shift gate (E2) stays inside it.
            if np.isfinite(ae["ent_exit"]):
                wm_vis = _dip_edge(s, sm, prof, ae["ent_exit"] - WM_SEARCH_UM, ae["ent_exit"] + WM_SEARCH_UM)
            else:
                wm_vis = np.nan
            cand2 = ed[(ed["sign"] > 0) & (ed["s_um"] > pia_vis + 50) & (ed["s_um"] < pia_vis + 250)]
            if len(cand2):
                l12_vis = float(cand2.sort_values("strength").iloc[-1]["s_um"])
        # where along the line does this contact sit?  its projection onto the normal
        s_contact = float(np.dot(pts[q] - p0[q], n[q]) * vox_um)
        rows.append({"p0_i": p0[q][0], "p0_j": p0[q][1], "p0_k": p0[q][2],
                     "n_i": n[q][0], "n_j": n[q][1], "n_k": n[q][2],
                     "s_contact_um": s_contact,
                     "pia_vis": pia_vis, "wm_vis": wm_vis, "l12_vis": l12_vis,
                     "pia_atlas": ae["pia"], "sup_deep_atlas": ae["sup_deep"], "l12_atlas": ae["l12"],
                     "wm_atlas": ae["ent_exit"], "exit_label": ae["exit_label"], "ent_span_um": ae["ent_span_um"]})
    df = pd.DataFrame(rows)
    # The grey/white comparison is only meaningful where the atlas's ENTl exit IS a
    # grey/white boundary.  Where ENTl abuts piriform/TR or hippocampal grey (ah09:
    # 60 % of lines) the "atlas WM" is a grey-grey boundary and the visible dark step
    # is some other structure -- not a registration error.
    fibre = set(groups["fibre"].tolist())
    df["exit_is_fibre"] = df["exit_label"].apply(lambda v: bool(pd.notna(v)) and (int(v) in fibre))
    df["pia_offset_um"] = df["pia_atlas"] - df["pia_vis"]
    df["wm_offset_um"] = df["wm_atlas"] - df["wm_vis"]
    df["l12_offset_um"] = df["l12_atlas"] - df["l12_vis"]
    df["thick_atlas_um"] = df["wm_atlas"] - df["pia_atlas"]
    df["thick_vis_um"] = df["wm_vis"] - df["pia_vis"]
    df["thickness_ratio"] = df["thick_atlas_um"] / df["thick_vis_um"]
    # sup/deep flip when the contact's fractional depth is re-read against the visible edges
    fb = (df["sup_deep_atlas"] - df["pia_atlas"]) / df["thick_atlas_um"]
    f_atlas = (df["s_contact_um"] - df["pia_atlas"]) / df["thick_atlas_um"]
    f_vis = (df["s_contact_um"] - df["pia_vis"]) / df["thick_vis_um"]
    df["deep_by_atlas"] = f_atlas >= fb
    df["deep_by_visible"] = f_vis >= fb
    df["flips"] = (df["deep_by_atlas"] != df["deep_by_visible"]) & np.isfinite(f_vis) & np.isfinite(fb)
    return df


def page_gain_correct(g: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Divide every coronal page by its in-brain median intensity (x the slab median).

    The two optical planes per 40 um section differ in brightness (depth attenuation),
    so consecutive pages alternate in gain -- severely in ah09 and ly07 (the vertical
    stripes in every sagittal panel).  Any line with an AP component then sees a
    20 um-period oscillation that mimics laminar dips.  A per-page gain is a pure
    acquisition effect and is removed here before profiles are sampled; it changes
    no geometry.
    """
    mask = labels > 0
    out = np.empty_like(g)
    meds = np.array([np.median(g[i][mask[i]]) if mask[i].any() else np.nan for i in range(g.shape[0])])
    ref = float(np.nanmedian(meds))
    for i in range(g.shape[0]):
        out[i] = g[i] * (ref / meds[i]) if np.isfinite(meds[i]) and meds[i] > 0 else g[i]
    return out


def laminar_landmark_check(subject: str, *, label_shift_um: float = 0.0, green_override: np.ndarray | None = None) -> pd.DataFrame:
    """Check E on the real mouse: every ENTl bank contact gets a line along the local surface normal."""
    geo = track_geometry(subject)
    groups = layer_id_sets()
    df = contacts(subject)
    b = df[df["bank"] & df["group"].isin(["ENTl-sup", "ENTl-deep"])]
    if not len(b):
        return pd.DataFrame()
    lo, hi = geo["slab"]
    # widen the slab so 1.5 mm normals stay inside: +/- 160 pages
    lo2, hi2 = max(lo - 130, 0), min(hi + 130, geo["shape"][0])
    a = atlas_slab(subject, lo2, hi2)
    g = green_override if green_override is not None else page_gain_correct(green_slab(subject, lo2, hi2).astype(np.float32), a)
    pts = b[[f"downsample_coords.{c}" for c in "ijk"]].values.copy()
    pts[:, 0] -= lo2
    thr = tissue_threshold(subject)
    out = laminar_lines(g, a, pts, groups, thr=thr, label_shift_um=label_shift_um)
    out["p0_i"] += lo2
    out.insert(0, "subject", subject)
    out["contact_y_um"] = b["probe_coords.y"].values
    out["shank"] = b["shank"].values
    out["group"] = b["group"].values
    return out


def _flips(df: pd.DataFrame, pia_vis: pd.Series, wm_vis: pd.Series) -> pd.Series:
    """Sup/deep label flips when the contact's fractional depth is re-read against given visible edges."""
    thick_atlas = df["wm_atlas"] - df["pia_atlas"]
    fb = (df["sup_deep_atlas"] - df["pia_atlas"]) / thick_atlas
    f_atlas = (df["s_contact_um"] - df["pia_atlas"]) / thick_atlas
    f_vis = (df["s_contact_um"] - pia_vis) / (wm_vis - pia_vis)
    return ((f_atlas >= fb) != (f_vis >= fb)) & np.isfinite(f_vis) & np.isfinite(fb)


def laminar_summary(lines: pd.DataFrame, baseline: dict | None = None) -> dict:
    """Medians/IQRs of the edge offsets; with ``baseline`` (from :func:`atlas_modality_control`)
    also the **baseline-corrected** offsets and flip fraction.

    The Allen annotation's brain surface sits ~50 um inside the template's visible
    edge (measured on the atlas itself: +50 um pia offset, 0 um at the grey/white
    edge), so a sample showing the same +50 um is registered *perfectly*, not 50 um
    off.  Corrected = raw minus the atlas's own offset; the flip fraction is
    recomputed with the visible edges shifted by the baseline.  Flags use the
    corrected values when a baseline is given.
    """
    if lines is None or not len(lines):
        return {"n_lines": 0}
    ok = lines.dropna(subset=["pia_vis", "pia_atlas"])
    wm = lines.dropna(subset=["wm_vis", "wm_atlas"])
    if "exit_is_fibre" in wm:
        wm = wm[wm["exit_is_fibre"].astype(bool)]
    out = {"n_lines": int(len(lines)), "n_with_pia": int(len(ok)), "n_with_wm": int(len(wm)),
           "frac_exit_fibre": float(lines["exit_is_fibre"].astype(bool).mean()) if "exit_is_fibre" in lines else np.nan,
           "pia_offset_median_um": float(ok["pia_offset_um"].median()) if len(ok) else np.nan,
           "pia_offset_iqr_um": float(np.subtract(*np.percentile(ok["pia_offset_um"], [75, 25]))) if len(ok) else np.nan,
           "wm_offset_median_um": float(wm["wm_offset_um"].median()) if len(wm) else np.nan,
           "wm_offset_iqr_um": float(np.subtract(*np.percentile(wm["wm_offset_um"], [75, 25]))) if len(wm) else np.nan,
           "thickness_ratio_median": float(wm["thickness_ratio"].median()) if len(wm) else np.nan,
           "thick_atlas_median_um": float(wm["thick_atlas_um"].median()) if len(wm) else np.nan,
           "thick_vis_median_um": float(wm["thick_vis_um"].median()) if len(wm) else np.nan,
           "flip_frac": float(wm["flips"].mean()) if len(wm) else np.nan,
           "l12_detected_frac": float(np.isfinite(lines["l12_vis"]).mean()),
           "l12_offset_median_um": float(lines["l12_offset_um"].median()) if np.isfinite(lines["l12_offset_um"]).any() else np.nan,
           "exit_labels": {str(structure_tree()["id2acro"].get(int(k), k)): int(v)
                           for k, v in lines["exit_label"].dropna().astype(int).value_counts().head(4).items()}}
    pia_o, wm_o = out["pia_offset_median_um"], out["wm_offset_median_um"]
    if baseline is not None and len(wm):
        bp, bw = float(baseline.get("pia", 0.0)), float(baseline.get("wm", 0.0))
        out["pia_offset_corr_um"] = pia_o - bp
        out["wm_offset_corr_um"] = wm_o - bw
        out["thickness_ratio_corr"] = float(((wm["wm_atlas"] - wm["pia_atlas"]) /
                                             ((wm["wm_vis"] + bw) - (wm["pia_vis"] + bp))).median())
        out["flip_frac_corr"] = float(_flips(wm, wm["pia_vis"] + bp, wm["wm_vis"] + bw).mean())
        pia_o, wm_o = out["pia_offset_corr_um"], out["wm_offset_corr_um"]
    # Flag on the grey/white offset only when it is a *consistent* edge: >= 20 lines
    # exit into fibre AND the lines agree to within the tolerance itself (IQR of the
    # offset <= 100 um).  A wide IQR means the detector is finding different features
    # on different lines (no dark fibre band in the profile, damaged tissue) and the
    # median is then not a registration measurement; it is reported but not flagged.
    wm_valid = len(wm) >= 20 and np.isfinite(wm_o) and out["wm_offset_iqr_um"] <= 100.0
    out["wm_offset_valid"] = bool(wm_valid)
    out["laminar_flag"] = bool((wm_valid and abs(wm_o) > QC_THRESHOLDS["laminar_offset_um"]) or
                               (np.isfinite(pia_o) and abs(pia_o) > QC_THRESHOLDS["laminar_offset_um"]))
    return out


def plot_laminar(subject: str, lines: pd.DataFrame, *, save: bool = True):
    """Mean intensity profile along the normal with the visible and atlas edges marked."""
    import matplotlib.pyplot as plt
    if not len(lines):
        return None
    geo = track_geometry(subject)
    groups = layer_id_sets()
    lo, hi = geo["slab"]
    lo2, hi2 = max(lo - 130, 0), min(hi + 130, geo["shape"][0])
    a = atlas_slab(subject, lo2, hi2)
    g = page_gain_correct(green_slab(subject, lo2, hi2).astype(np.float32), a)   # same profile the numbers use
    p0 = lines[["p0_i", "p0_j", "p0_k"]].values.copy(); p0[:, 0] -= lo2
    n = lines[["n_i", "n_j", "n_k"]].values
    s, prof = line_profiles(g, p0, n, order=1)
    _, lab = line_profiles(a, p0, n, order=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    med = np.nanmedian(prof, 0); q1, q3 = np.nanpercentile(prof, [25, 75], axis=0)
    ax.fill_between(s, q1, q3, color="C0", alpha=0.2); ax.plot(s, med, color="C0", label="green intensity (median, IQR)")
    for col, key, lab_ in (("k", "pia_vis", "visible pia"), ("C3", "wm_vis", "visible grey/white"), ("C2", "l12_vis", "visible L1/L2")):
        v = lines[key].median()
        if np.isfinite(v):
            ax.axvline(v, color=col, ls="-", label=f"{lab_} {v:.0f}")
    for col, key, lab_ in (("k", "pia_atlas", "atlas pia"), ("C3", "wm_atlas", "atlas ENTl exit"), ("C1", "sup_deep_atlas", "atlas sup/deep")):
        v = lines[key].median()
        if np.isfinite(v):
            ax.axvline(v, color=col, ls="--", label=f"{lab_} {v:.0f}")
    ax.set_xlabel("distance along inward normal from atlas surface (µm)"); ax.set_ylabel("intensity")
    ax.legend(fontsize=7); ax.set_title(f"{subject} — {len(lines)} ENTl bank contacts", fontsize=10)
    # label occupancy along s
    ax2 = axes[1]
    for gname, colour in (("ENTl_sup", GROUP_COLOURS["ENTl_sup"]), ("ENTl_deep", GROUP_COLOURS["ENTl_deep"]), ("fibre", GROUP_COLOURS["fibre"])):
        occ = np.isin(lab, groups[gname]).mean(0)
        ax2.plot(s, occ, color=colour, label=gname)
    ax2.plot(s, (lab > 0).mean(0), color="k", lw=0.8, label="in brain")
    ax2.set_xlabel("s (µm)"); ax2.set_ylabel("fraction of lines"); ax2.legend(fontsize=7); ax2.set_title("atlas label occupancy", fontsize=10)
    base = modality_baseline()
    summ = laminar_summary(lines, baseline=base)
    fig.suptitle(f"raw: pia {summ['pia_offset_median_um']:+.0f} µm · WM {summ['wm_offset_median_um']:+.0f} µm · thickness ratio "
                 f"{summ['thickness_ratio_median']:.2f} · flip {summ['flip_frac']:.2f}     |     "
                 f"baseline-corrected (atlas own offsets pia {base['pia']:+.0f}, WM {base['wm']:+.0f}): "
                 f"pia {summ['pia_offset_corr_um']:+.0f} µm · WM {summ['wm_offset_corr_um']:+.0f} µm · "
                 f"ratio {summ['thickness_ratio_corr']:.2f} · flip {summ['flip_frac_corr']:.2f}", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / f"{subject}_laminar.png", dpi=130, bbox_inches="tight")
    return fig


_MODALITY: dict = {}


def atlas_modality_control(*, n_points: int = 300, seed: int = 0) -> dict:
    """The laminar detector on the unfiltered atlas reference vs the atlas annotation at ENTl.

    Two jobs.  (1) Does the detector work on this modality at all -- do the
    visible edges exist and sit consistently relative to the annotation?  (2) What
    is the annotation's own offset from the visible edges?  Measured 2026-09-11:
    pia **+50 um** (the annotated surface sits two 25 um voxels inside the template's
    visible edge), grey/white **0 um**, thickness ratio 0.94.  That is the
    **baseline** every sample offset is corrected by (:func:`laminar_summary`); a
    mouse reproducing +50 / 0 is registered perfectly.  ``ok`` is about (1): the
    grey/white edge found on >= 90 % of lines with tight IQRs.  Cached per process.
    """
    if _MODALITY:
        return _MODALITY
    at = atlas25()
    ref = at["reference"].astype(np.float32); ann = at["annotation"].astype(np.int64)
    groups = layer_id_sets()
    rng = np.random.default_rng(seed)
    # points inside ENTl superficial in the LEFT hemisphere (k > half), one 25um voxel deep
    sup = np.isin(ann, groups["ENTl_sup"]); sup[:, :, : ann.shape[2] // 2] = False
    ijk = np.argwhere(sup)
    pts = ijk[rng.choice(len(ijk), size=min(n_points, len(ijk)), replace=False)].astype(float)
    # the same vox_um logic with 25 um voxels; lower the tissue threshold to agarose ~ 0
    thr = float(np.percentile(ref[ann > 0], 5)) * 0.5
    lines = laminar_lines(ref, ann, pts, groups, vox_um=25.0, thr=thr, brain_mask=ann > 0)
    summ = laminar_summary(lines)
    summ["baseline"] = {"pia": summ["pia_offset_median_um"], "wm": summ["wm_offset_median_um"],
                        "thickness_ratio": summ["thickness_ratio_median"]}
    n_fibre_exit = max(summ["n_lines"] * summ.get("frac_exit_fibre", 1.0), 1)
    summ["ok"] = bool(summ["n_with_wm"] >= 0.8 * n_fibre_exit
                      and summ["pia_offset_iqr_um"] <= 60 and summ["wm_offset_iqr_um"] <= 80
                      and abs(summ["wm_offset_median_um"]) <= 60)
    _MODALITY.update(summ)
    return summ


def modality_baseline() -> dict:
    return atlas_modality_control()["baseline"]


# --------------------------------------------------------------------------
# synthetic gates
# --------------------------------------------------------------------------


def _gate_distances() -> dict:
    """D: two half-spaces with a known planar interface; planted contacts at known distances."""
    slab = np.zeros((40, 60, 60), dtype=np.int64)
    groups = layer_id_sets()
    sup_id, deep_id = int(groups["ENTl_sup"][0]), int(groups["ENTl_deep"][0])
    slab[:, :30, :] = sup_id; slab[:, 30:, :] = deep_id
    iface = interface_voxels(slab, groups["ENTl_sup"], groups["ENTl_deep"])
    truth = np.array([0.5, 1, 3, 5, 10, 20, 30])            # voxels from the interface plane (j = 29.5)
    pts = np.array([[20, 29.5 - t, 30] for t in truth])
    d = cKDTree(iface).query(pts)[0]
    # nearest interface voxel sits at j=29 (sup side) or 30 (deep side): distance = t - 0.5
    err = np.abs(d - np.maximum(truth - 0.5, 0.0))
    return {"max_err_vox": float(err.max()), "ok": bool(err.max() <= 0.5)}


def _gate_jacobian() -> dict:
    """C: a pure-affine field must give det J = det A everywhere; a planted fold must go negative."""
    A = np.array([[0.97, -0.09, -0.14], [0.11, 1.12, 0.13], [0.11, -0.11, 1.03]])
    n, J, K = 12, 40, 40
    ii, jj, kk = np.meshgrid(np.arange(n) * VOX_MM, np.arange(J) * VOX_MM, np.arange(K) * VOX_MM, indexing="ij")
    X = np.stack([ii, jj, kk])
    D = np.einsum("ca,aijk->cijk", A, X).astype(np.float32)
    detj = jacobian_det(D)[1:-1, 1:-1, 1:-1]
    err = float(np.abs(detj - np.linalg.det(A)).max())
    # planted fold: reverse the j-mapping in a block
    D2 = D.copy()
    D2[1, :, 15:25, :] = D2[1, :, 24:14:-1, :]
    detj2 = jacobian_det(D2)
    return {"affine_max_err": err, "fold_detected": bool((detj2 < 0).any()), "ok": bool(err < 1e-3 and (detj2 < 0).any())}


def _gate_edge_detector(seed: int = 0) -> dict:
    """E1: synthetic step profiles at known positions, blurred + noisy, must be recovered to 15 um."""
    rng = np.random.default_rng(seed)
    s = np.arange(S_RANGE_UM[0], S_RANGE_UM[1] + S_STEP_UM / 2, S_STEP_UM)
    errs = []
    for pia, wm, sigma in ((0, 900, 20), (30, 1100, 40), (-40, 700, 30)):
        prof = np.where(s < pia, 50, 1000.0)
        prof = np.where(s > wm, 250, prof)
        prof = ndimage.gaussian_filter1d(prof, sigma / S_STEP_UM) + rng.normal(0, 40, len(s))
        ed = detect_edges(s, prof)
        pos = ed[ed["sign"] > 0].sort_values("strength")
        neg = ed[ed["sign"] < 0].sort_values("strength")
        errs.append(abs(pos.iloc[-1]["s_um"] - pia) if len(pos) else np.inf)
        errs.append(abs(neg.iloc[-1]["s_um"] - wm) if len(neg) else np.inf)
    flat = rng.normal(1000, 40, len(s))
    n_false = len(detect_edges(s, flat))
    return {"max_err_um": float(np.max(errs)), "false_edges_on_flat": int(n_false),
            "ok": bool(np.max(errs) <= 15 and n_false == 0)}


def _gate_label_shift(subject: str = "ah08") -> dict:
    """E2: reading labels shifted by +-100 um along the normal must move the offsets by the same amount."""
    base = laminar_landmark_check(subject)
    if not len(base):
        return {"ok": False, "reason": "no ENTl bank contacts"}
    out = {}
    for d in (+100.0, -100.0):
        sh = laminar_landmark_check(subject, label_shift_um=d)
        out[f"pia_shift_{int(d):+d}"] = float((sh["pia_atlas"] - base["pia_atlas"]).median())
        out[f"wm_shift_{int(d):+d}"] = float((sh["wm_atlas"] - base["wm_atlas"]).median())
    zero = laminar_landmark_check(subject, label_shift_um=0.0)
    out["zero_reproduces"] = bool(np.allclose(zero["pia_atlas"].fillna(-1), base["pia_atlas"].fillna(-1)))
    out["ok"] = bool(abs(out["pia_shift_+100"] - 100) <= 20 and abs(out["pia_shift_-100"] + 100) <= 20
                     and abs(out["wm_shift_+100"] - 100) <= 20 and abs(out["wm_shift_-100"] + 100) <= 20
                     and out["zero_reproduces"])
    return out


def _gate_truth_negative(subject: str = "ah08", seed: int = 0) -> dict:
    """E3: a synthetic green volume built FROM the registered labels must yield ~0 offsets and no flag."""
    geo = track_geometry(subject)
    groups = layer_id_sets()
    lo, hi = geo["slab"]
    lo2, hi2 = max(lo - 130, 0), min(hi + 130, geo["shape"][0])
    a = atlas_slab(subject, lo2, hi2)
    rng = np.random.default_rng(seed)
    g = np.full(a.shape, 1000.0, dtype=np.float32)
    g[a == 0] = 45.0
    g[np.isin(a, groups["fibre"])] = 220.0
    g[np.isin(a, groups["ENTl_sup"][:1])] = 1000.0
    g = ndimage.gaussian_filter(g, 2.0) + rng.normal(0, 30, g.shape).astype(np.float32)
    # No baseline here: the synthetic green has no pial rim by construction, so its
    # visible pia IS the annotated surface and the raw offsets must be ~0.
    lines = laminar_landmark_check(subject, green_override=g)
    summ = laminar_summary(lines)
    summ["ok"] = bool(abs(summ["pia_offset_median_um"]) < 20 and abs(summ["wm_offset_median_um"]) < 40 and not summ["laminar_flag"])
    return summ


def run_synthetic_controls(subject: str = "ah08", *, verbose: bool = True) -> pd.DataFrame:
    """Gate every estimator in both directions before any real number is reported.

    Rows: D distances, C Jacobian, E1 detector, E2 planted label shift, E3 truth
    negative, E4 atlas modality control, self-consistency positive controls, and
    the slab-labelling equivalence with ``probe_refit.project_probe``.
    """
    rows = []
    rows.append({"gate": "D_distances", **_gate_distances()})
    rows.append({"gate": "C_jacobian", **_gate_jacobian()})
    rows.append({"gate": "E1_edge_detector", **_gate_edge_detector()})
    rows.append({"gate": "E4_atlas_modality", **{k: v for k, v in atlas_modality_control().items() if k in ("pia_offset_median_um", "pia_offset_iqr_um", "wm_offset_median_um", "wm_offset_iqr_um", "thickness_ratio_median", "n_with_wm", "ok")}})
    rows.append({"gate": "contacts_equal_project_probe", **verify_contacts(subject)})
    sc0 = label_self_consistency(subject)
    scp = label_self_consistency(subject, perturb="permute")
    scs = label_self_consistency(subject, perturb="shift")
    rows.append({"gate": "self_consistency_negative", "disagreement": sc0["disagreement"], "floor": sc0["boundary_floor"], "ok": sc0["ok"]})
    rows.append({"gate": "self_consistency_permute", "disagreement": scp["disagreement"], "ok": scp["disagreement"] > 0.30})
    rows.append({"gate": "self_consistency_shift100", "disagreement": scs["disagreement"], "ok": scs["disagreement"] > 0.30})
    rows.append({"gate": "E2_label_shift", **_gate_label_shift(subject)})
    rows.append({"gate": "E3_truth_negative", **{k: v for k, v in _gate_truth_negative(subject).items() if k in ("pia_offset_median_um", "wm_offset_median_um", "thickness_ratio_median", "laminar_flag", "ok")}})
    table = pd.DataFrame(rows)
    table.attrs["passed"] = bool(table["ok"].astype(bool).all())
    if verbose:
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(table.to_string(index=False))
        print(f"\nSYNTHETIC GATE: {'PASS' if table.attrs['passed'] else 'FAIL'}")
    return table


# --------------------------------------------------------------------------
# caches, table, verdicts
# --------------------------------------------------------------------------


def _load_global(subject: str):
    p = subject_paths(subject)["global_cache"]
    if p.exists():
        return dict(np.load(p, allow_pickle=True))
    return None


def _update_global(subject: str, arrays: dict, **scalars) -> None:
    p = subject_paths(subject)["global_cache"]
    cur = _load_global(subject) or {}
    cur.update(arrays)
    for k, v in scalars.items():
        cur[k] = np.asarray(v)
    np.savez(p, **cur)


def global_checks(subject: str, *, verbose: bool = True) -> dict:
    """Phase 0/B numbers for one mouse (no figures)."""
    A, _ = load_affine(subject)
    dec = decompose_affine(A)
    conv = verify_affine_convention(subject)
    trunc = truncation_report(subject)
    asym = lr_asymmetry(subject)
    hemi = hemisphere_check(subject)
    prof = lr_profile(subject)
    resid = residual_angles(prof)
    cont = section_continuity(subject)
    tile = tile_artefact(subject)
    ratios = region_volume_ratios(subject)
    lec = asym[asym["group"].isin(["ENTl", "ENTm", "SUB", "ProS", "CA1", "DG"])]
    out = {"subject": subject, **{f"affine_{k}": v for k, v in dec.items() if k != "translation_mm"},
           "convention_ok": conv["convention_ok"], "field_resid_mm": conv["median_resid_mm_A"],
           **{k: trunc[k] for k in ("tissue_ap_mm", "missing_anterior_mm", "missing_posterior_mm", "mode",
                                    "tissue_frac_first_page_rel", "tissue_frac_last_page_rel")},
           "lr_asym_max_lecHPF": float(lec["asym"].abs().max()), "lr_asym_worst_group": lec.loc[lec["asym"].abs().idxmax(), "group"],
           "lr_asym_flags": ",".join(lec.loc[lec["flag"], "group"]) or "-",
           "hemisphere_ok": hemi["ok"], **resid, **{f"section_{k}": v for k, v in cont.items() if k != "jump_pages"},
           **{f"tile_{k}": v for k, v in tile.items()},
           "ratio_ENTl": float(ratios.set_index("group")["ratio"].get("ENTl", np.nan)),
           "ratio_CA1": float(ratios.set_index("group")["ratio"].get("CA1", np.nan)),
           "ratio_expected": float(ratios["expected_1_over_detA"].iloc[0]) if len(ratios) else np.nan}
    flags = []
    if not (QC_THRESHOLDS["affine_scale"][0] <= min(dec["scale_ap"], dec["scale_dv"], dec["scale_lr"]) and
            max(dec["scale_ap"], dec["scale_dv"], dec["scale_lr"]) <= QC_THRESHOLDS["affine_scale"][1]):
        flags.append("affine_scale")
    if dec["rotation_total_deg"] > QC_THRESHOLDS["affine_rotation_deg"]:
        flags.append("affine_rotation")
    advisories = []
    if abs(resid.get("residual_yaw_deg", 0)) > QC_THRESHOLDS["residual_angle_deg"]:
        flags.append("residual_yaw")
    # Residual pitch is ADVISORY: the DV landmark is the tissue top at the midline
    # column, where dura and the sagittal sinus sit in the mask, so it is noisier
    # than the LR envelope midline (which is symmetric by construction).  DV
    # placement where it matters -- at the track -- is what check E measures.
    if abs(resid.get("residual_pitch_deg", 0)) > QC_THRESHOLDS["residual_angle_deg"]:
        advisories.append(f"residual_pitch({resid['residual_pitch_deg']:+.1f}°)")
    # The page-centroid statistics (section_*) are reported but carry no advisory:
    # every mouse shows 50-180 pages with > 3 vox steps and maximum steps of 0.5-2 mm,
    # dominated by the largest component switching identity near the block faces and
    # by loose cortical flaps (ah08, AP 6.5-7.4 mm) -- not by block slippage.  A real
    # slippage test would be page-to-page image cross-correlation; not built.
    if lec["flag"].any():
        flags.append("lr_asym:" + ",".join(lec.loc[lec["flag"], "group"]))
    if not hemi["ok"]:
        flags.append("hemisphere")
    if not conv["convention_ok"]:
        flags.append("affine_convention")
    if trunc["mode"] != "overhang":
        flags.append("truncation_squash")
    out["global_flags"] = ";".join(flags) or "-"
    out["global_advisories"] = ";".join(advisories) or "-"
    out["_tables"] = {"asymmetry": asym, "profile": prof, "ratios": ratios, "truncation": trunc, "continuity": cont}
    if verbose:
        print(f"{subject}: scale AP/DV/LR {dec['scale_ap']:.3f}/{dec['scale_dv']:.3f}/{dec['scale_lr']:.3f}, "
              f"yaw {dec['yaw_deg']:+.1f}° pitch {dec['pitch_deg']:+.1f}° roll {dec['roll_deg']:+.1f}°; "
              f"residual yaw {resid['residual_yaw_deg']:+.2f}° pitch {resid['residual_pitch_deg']:+.2f}°; "
              f"missing ant {trunc['missing_anterior_mm']:.1f} mm ({trunc['mode']}); "
              f"max |asym| LEC/HPF {out['lr_asym_max_lecHPF']:.2f} ({out['lr_asym_worst_group']}); flags {out['global_flags']}")
    return out


def local_checks(subject: str, *, verbose: bool = True) -> dict:
    """Phase 1: D, G, E, C numbers for one mouse (no figures)."""
    dist = contact_boundary_distances(subject)
    frag = fragility_summary(dist)
    sc = label_self_consistency(subject)
    coh = laminar_coherence(subject)
    lines = laminar_landmark_check(subject)
    lam = laminar_summary(lines, baseline=modality_baseline())
    jac = jacobian_check(subject)
    out = {"subject": subject, **frag, "self_consistency_disagreement": sc["disagreement"], "self_consistency_ok": sc["ok"],
           "coherence_flicker_frac": float(coh["frac_in_flicker"].mean()) if len(coh) else np.nan,
           "coherence_reversals": int(coh["n_ordinal_reversals"].sum()) if len(coh) else 0,
           **{f"laminar_{k}": v for k, v in lam.items()},
           **{k: v for k, v in jac.items() if k != "maps"}}
    flags = []
    if lam.get("laminar_flag"):
        flags.append("laminar_offset")
    if jac["detj_frac_outside"] > QC_THRESHOLDS["detj_frac_outside"] or jac["detj_n_negative"] > 0:
        flags.append("jacobian")
    if not sc["ok"]:
        flags.append("self_consistency")
    adv = []
    if frag.get("fragile_advisory"):
        adv.append(f"fragile_supdeep({frag['frac_ENTl_within_100um_of_supdeep']:.2f})")
    if lam.get("n_lines", 0) and not lam.get("wm_offset_valid", True):
        adv.append(f"wm_offset_unreliable(IQR {lam.get('wm_offset_iqr_um', np.nan):.0f} µm, n {lam.get('n_with_wm')})")
    if frag.get("lecmec_advisory"):
        adv.append(f"lecmec_border({frag['frac_bank_within_100um_of_lecmec']:.2f} of bank within 100 µm)")
    out["local_flags"] = ";".join(flags) or "-"
    out["local_advisories"] = ";".join(adv) or "-"
    out["_tables"] = {"distances": dist, "coherence": coh, "lines": lines, "jacobian": jac,
                      "lecmec_shanks": lec_mec_by_shank(dist)}
    if verbose:
        print(f"{subject}: ENTl bank {frag['n_bank_ENTl']}/{frag['n_bank']}, within 100 µm of sup/deep "
              f"{frag.get('frac_ENTl_within_100um_of_supdeep', np.nan):.2f}; laminar pia {lam.get('pia_offset_median_um', np.nan):+.0f} "
              f"WM {lam.get('wm_offset_median_um', np.nan):+.0f} µm, thickness ratio {lam.get('thickness_ratio_median', np.nan):.2f}, "
              f"flip {lam.get('flip_frac', np.nan):.2f}; detJ median {jac['detj_median']:.2f} (affine {jac['det_affine']:.2f}), "
              f"folds {jac['detj_n_negative']}; flags {out['local_flags']} adv {out['local_advisories']}")
    return out


def qc_table(subjects: list[str] | None = None, *, save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """One row per mouse with every Phase 0/1 metric; written to ``brainreg/registration_qc.csv``."""
    subjects = subjects or pr.list_subjects()
    rows = []
    for s in subjects:
        g = global_checks(s, verbose=verbose)
        l = local_checks(s, verbose=verbose)
        row = {k: v for k, v in g.items() if not k.startswith("_")}
        row.update({k: v for k, v in l.items() if not k.startswith("_") and k != "subject"})
        rows.append(row)
        clear_caches(s)
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(BRAINREG_DIR / "registration_qc.csv", index=False)
    return df


def record_verdict(subject: str, verdict: str, note: str = "", *, by: str = "adam") -> dict:
    """Append a human verdict ('ok' / 'suspect' / 'fail') for one mouse."""
    path = BRAINREG_DIR / "registration_qc_verdicts.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur[subject] = {"verdict": verdict, "note": note, "by": by,
                    "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "fit_key": fit_key(subject)}
    path.write_text(json.dumps(cur, indent=2))
    return cur[subject]


def make_all_figures(subject: str) -> None:
    import matplotlib.pyplot as plt
    for fn in (plot_sample_overlays, plot_lec_zoom, plot_atlas_space_checker, plot_lr_profile):
        fig = fn(subject)
        plt.close(fig)
    lines = laminar_landmark_check(subject)
    if len(lines):
        plt.close(plot_laminar(subject, lines))
    plt.close(plot_jacobian(subject))
