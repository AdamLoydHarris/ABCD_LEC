"""Publication figures for probe anatomy: where the silicon sat, and what it sat in.

Ported from https://github.com/AdamLoydHarris/probe_tracing (the PFC/El-Gaby
2024 figures) onto this project's corrected LEC histology.  Three figures:

1. ``plot_channel_regions``  -- one column-block per mouse, every recording site
   coloured by its brain region, plus a matching unit-depth histogram.
   (old repo: ``plot_probes_with_anatomical_locations`` -> ``channel_regions.pdf``)
2. ``plot_probe_tracts``     -- brainrender scene, brain + region meshes + one
   line per shank, coloured by mouse.
   (old repo: ``plot_experiment_probe_tracts`` -> ``probes_in_brain.svg``)
3. ``plot_mouse_legend``     -- the mouse-colour swatch key.
   (old repo: the ``__main__`` block -> ``probe_legend.pdf``)

Run in the ``histology`` conda env.  The 3D figure needs a display; on a
headless node run the notebook or script under ``xvfb-run -a``.

**Anatomy comes from re-projecting the current fit, never from the stored
``ProbeA_anatomy.htsv``.**  That file is a cached artefact and can lag the fit:
ah10's went stale when the shank-order mirror was corrected (identical region
*composition*, but only 20% per-contact agreement -- the mirror permutes which
channel sits where without changing the multiset).  ``_projection`` re-runs
``pr.project_probe`` so the figures always match ``ProbeA_fit_params.json``,
which is what ``channel_regions.csv`` and ``unit_regions.pkl`` also do.

**Probe geometry.**  The recorded bank is exactly 8 x-positions x 48
y-positions = 384 contacts (4 shanks x 2 columns x 48 rows, 15 um y-step).  The
grid is built by *pivoting on measured (x, y)*, never by reshaping a flat array:
a reshape silently scrambles shanks, and shanks sit in different structures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib.patches import Patch

import probe_refit as pr
import region_assignment as ra

FIGURE_DIR = pr.REPO_ROOT / "data" / "figures" / "probe_anatomy"

#: Vector first -- these are figure panels, not previews.  PNG is written too,
#: purely so they can be eyeballed quickly without a PDF viewer.
VECTOR_FORMATS = ("pdf", "svg")
RASTER_FORMATS = ("png",)


def _save(fig, stem: str, *, dpi: int = 200, raster: bool = True) -> list[Path]:
    """Write a matplotlib figure as vector (+ optional PNG preview)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in VECTOR_FORMATS + (RASTER_FORMATS if raster else ()):
        p = FIGURE_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        paths.append(p)
        print(f"saved {p}")
    return paths

#: Mice with ephys.  ah09 has histology but was never recorded.
MICE = ra.EPHYS_MICE

#: Shanks are 250 um apart with two contact columns 32 um apart within each.
SHANK_PITCH_UM = 250.0
N_SHANKS = 4
Y_STEP_UM = 15.0


# --------------------------------------------------------------------------
# palettes -- shared by all three figures
# --------------------------------------------------------------------------

def region_palette() -> dict[str, tuple[float, float, float]]:
    """Acronym -> RGB, graded within family so cortical layers read as a ramp.

    Follows the old repo's ``_get_custom_region_colors``: the target family gets
    a strong sequential ramp, everything off-target is muted, fibre tracts grey.
    """
    ENTl = sns.color_palette("RdPu", 6)[1:]          # 5 layers, superficial -> deep
    ENTm = sns.color_palette("Blues", 6)[1:]         # 5 layers
    subic = sns.color_palette("Purples", 5)[2:4]     # ProS, SUB
    hipp = sns.color_palette("Greens", 5)[2:4]       # CA1, HPF
    fibre = sns.color_palette("Greys", 6)[1:3]       # alv, ec

    return {
        "ENTl1": ENTl[0], "ENTl2": ENTl[1], "ENTl3": ENTl[2],
        "ENTl5": ENTl[3], "ENTl6a": ENTl[4],
        "ENTm1": ENTm[0], "ENTm2": ENTm[1], "ENTm3": ENTm[2],
        "ENTm5": ENTm[3], "ENTm6": ENTm[4],
        "ProS": subic[0], "SUB": subic[1],
        "CA1": hipp[0], "HPF": hipp[1],
        "alv": fibre[0], "ec": fibre[1],
    }


def mouse_palette(mice=MICE) -> dict[str, tuple]:
    """Mouse -> RGB.  Single source of truth for figures 2 and 3."""
    return dict(zip(mice, sns.color_palette("tab10", len(mice))))


#: colour for a contact whose structure is unlabelled / outside the brain,
#: matching the old repo's ``np.nan: (0, 0, 0)``.
OUTSIDE_COLOUR = (0.0, 0.0, 0.0)


def check_palette_complete(mice=MICE, palette=None) -> dict:
    """Raise if any recorded acronym has no colour.

    Ported from the old repo's assert, which is why those figures never
    silently dropped a region.
    """
    palette = palette or region_palette()
    seen = set()
    for m in mice:
        seen |= set(bank_table(m)["acronym"].dropna().unique())
    missing = sorted(a for a in seen if a not in palette)
    assert not missing, f"Missing region colour for {missing}"
    unused = sorted(a for a in palette if a not in seen)
    return {"n_regions": len(seen), "missing": missing, "unused": unused}


# --------------------------------------------------------------------------
# geometry -- the 48 x 8 recorded bank
# --------------------------------------------------------------------------

_PROJ_CACHE: dict[tuple[str, bool], pd.DataFrame] = {}


def _projection(mouse: str, *, with_allen: bool = False) -> pd.DataFrame:
    """Re-project the probe from the *current* fit.  Cached per process.

    ``with_allen`` additionally maps contacts into Allen space, which needs the
    three deformation fields (~7 GB) that ``load_volumes(fast=True)`` skips.
    Only the 3D figure needs them; the region strip does not.
    """
    if (mouse, True) in _PROJ_CACHE:            # the full frame serves both
        return _PROJ_CACHE[(mouse, True)]
    key = (mouse, with_allen)
    if key not in _PROJ_CACHE:
        fit = pr.load_fit(mouse)
        plane, params = pr.split_fit(fit)
        data = pr.load_volumes(mouse, fast=not with_allen)
        _PROJ_CACHE[key] = pr.project_probe(plane, params, data,
                                            with_allen=with_allen)
        pr.clear_volume_cache(mouse)
    return _PROJ_CACHE[key]


def _fit_key(mouse: str) -> str:
    """Short hash of the fit that a cached projection belongs to.

    Keying the cache on the fit (not just the mouse) is what stops the ah10
    class of bug: correct the fit, the key changes, the cache is rebuilt.
    """
    import hashlib
    import json
    plane, params = pr.split_fit(pr.load_fit(mouse))
    blob = json.dumps({"plane": {k: np.asarray(v).tolist() for k, v in plane.items()},
                       "params": {k: float(v) for k, v in params.items()}},
                      sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def allen_coords(mouse: str, *, rebuild: bool = False) -> pd.DataFrame:
    """Per-contact Allen-space coordinates (µm), cached to disk.

    Computing these needs the three deformation fields (~7 GB), which takes
    minutes per mouse.  The result is small, so cache it beside the fit and key
    the cache on ``_fit_key`` so a re-fit invalidates it automatically.
    """
    cache = pr.BRAINREG_DIR / mouse / f"probe_allen_coords_{_fit_key(mouse)}.csv"
    if cache.exists() and not rebuild:
        return pd.read_csv(cache)
    df = _projection(mouse, with_allen=True)
    cols = ["probe_coords.x", "probe_coords.y"] + [f"allen_atlas_coords.{c}" for c in "ijk"]
    out = df[cols].copy()
    for stale in (pr.BRAINREG_DIR / mouse).glob("probe_allen_coords_*.csv"):
        stale.unlink()                       # only one fit's cache is ever valid
    out.to_csv(cache, index=False)
    print(f"  cached Allen coords -> {cache.name}")
    return out


def _recorded_contacts(mouse: str) -> set[tuple[float, float]]:
    """(x, y) of contacts recorded in at least one block.

    The dead set is constant across a mouse's blocks, so this is well defined.
    """
    c = pd.read_csv(ra.OUT_DIR / "channel_regions.csv")
    c = c[c["mouse"] == mouse]
    return {(round(float(x), 1), round(float(y), 1))
            for x, y in zip(c["x_um"], c["y_um"])}


def bank_table(mouse: str) -> pd.DataFrame:
    """The 384 recorded-bank contacts with region, shank and AP position."""
    df = _projection(mouse)
    b = df[df["probe_coords.y"] <= pr.RECORDED_BANK_MAX_UM].copy()
    rec = _recorded_contacts(mouse)
    out = pd.DataFrame({
        "x_um": b["probe_coords.x"].round(1).values,
        "y_um": b["probe_coords.y"].round(1).values,
        "acronym": b["structure.acronym"].values,
        "name": b["structure.name"].values,
        "ap_i": b["downsample_coords.i"].values,
        "shank": pr.shank_id(b["probe_coords.x"].values),
    })
    out["recorded"] = [(x, y) in rec for x, y in zip(out["x_um"], out["y_um"])]
    return out


def shank_ap_order(mouse: str) -> list[int]:
    """Shank ids ordered anterior -> posterior by their mean AP voxel."""
    b = bank_table(mouse)
    return list(b.groupby("shank")["ap_i"].mean().sort_values().index)


def site_grid(mouse: str, palette=None):
    """Build the (48, 8) region-colour grid by pivoting on measured (x, y).

    Returns ``(rgb, recorded, y_vals, order)``.  Column ``2*r + c`` is the
    ``c``-th contact column of the ``r``-th shank in anterior->posterior order.

    A hard gate: every one of the 384 contacts must land in exactly one cell.
    """
    palette = palette or region_palette()
    b = bank_table(mouse)
    order = shank_ap_order(mouse)
    y_vals = np.sort(b["y_um"].unique())
    y_ix = {y: i for i, y in enumerate(y_vals)}

    n_y = len(y_vals)
    rgb = np.full((n_y, 2 * N_SHANKS, 3), np.nan)
    recorded = np.zeros((n_y, 2 * N_SHANKS), dtype=bool)
    filled = np.zeros((n_y, 2 * N_SHANKS), dtype=int)

    for r, sh in enumerate(order):
        s = b[b["shank"] == sh]
        cols = np.sort(s["x_um"].unique())
        assert len(cols) == 2, f"{mouse} shank {sh}: {len(cols)} contact columns"
        for c, x in enumerate(cols):
            for _, row in s[s["x_um"] == x].iterrows():
                i, j = y_ix[row["y_um"]], 2 * r + c
                acro = row["acronym"]
                rgb[i, j] = palette.get(acro, OUTSIDE_COLOUR) \
                    if isinstance(acro, str) and acro else OUTSIDE_COLOUR
                recorded[i, j] = row["recorded"]
                filled[i, j] += 1

    assert filled.sum() == len(b) == 384, f"{mouse}: {filled.sum()} placed of {len(b)}"
    assert (filled == 1).all(), f"{mouse}: {(filled != 1).sum()} cells not filled exactly once"
    return rgb, recorded, y_vals, order


def _with_spacers(rgb, recorded):
    """Insert white spacer columns between shanks so the 4 are visually separate.

    Returns ``(img, recorded, is_contact)``.  ``is_contact`` is essential: without
    it the spacer columns look exactly like unrecorded contacts, and every gap
    gets marked dead.
    """
    n_y = rgb.shape[0]
    blocks, rec_blocks, real_blocks = [], [], []
    gap = np.ones((n_y, 1, 3))
    gap_flag = np.zeros((n_y, 1), dtype=bool)
    for r in range(N_SHANKS):
        blocks.append(rgb[:, 2 * r:2 * r + 2])
        rec_blocks.append(recorded[:, 2 * r:2 * r + 2])
        real_blocks.append(np.ones((n_y, 2), dtype=bool))
        if r < N_SHANKS - 1:
            blocks.append(gap)
            rec_blocks.append(gap_flag)
            real_blocks.append(gap_flag)
    return (np.concatenate(blocks, axis=1),
            np.concatenate(rec_blocks, axis=1),
            np.concatenate(real_blocks, axis=1))


#: x centre of shank ``r``'s 2-column block once spacers are inserted.
def _shank_centre(r: int) -> float:
    return 3 * r + 1.0


# --------------------------------------------------------------------------
# figure 1 -- channel regions (+ unit depth)
# --------------------------------------------------------------------------

def unit_depth_profile(mouse: str, y_vals) -> pd.DataFrame:
    """Unit-recordings per (depth bin x acronym) for a mouse, pooled over blocks."""
    u = pd.read_csv(ra.OUT_DIR / "unit_regions.csv")
    u = u[u["mouse"] == mouse]
    tab = (u.groupby(["y_um", "acronym"]).size().unstack(fill_value=0)
           .reindex(y_vals, fill_value=0))
    return tab.fillna(0).astype(int)


def plot_channel_regions(mice=MICE, *, units_panel: bool = True,
                         save: bool | str = False, dpi: int = 200):
    """Per-mouse recording sites coloured by brain region, tip at the bottom.

    With ``units_panel``, each mouse also gets a depth-matched histogram of its
    QC single units, stacked by region -- where the *cells* were, as against
    where the silicon was.  The two differ, because yield varies by region.
    """
    palette = region_palette()
    check_palette_complete(mice, palette)

    ncol = 2 if units_panel else 1
    fig, axes = plt.subplots(
        1, len(mice) * ncol,
        figsize=(1.9 * len(mice) * (1.6 if units_panel else 1.0), 7.2),
        gridspec_kw={"width_ratios": [1.0, 1.15] * len(mice) if units_panel else None},
    )
    axes = np.atleast_1d(axes)
    names: dict[str, str] = {}

    for k, mouse in enumerate(mice):
        rgb, rec, y_vals, order = site_grid(mouse, palette)
        img, rec_img, is_contact = _with_spacers(rgb, rec)
        b = bank_table(mouse)
        names.update({a: n for a, n in zip(b["acronym"], b["name"])
                      if isinstance(a, str) and a})

        ax = axes[k * ncol]
        ax.imshow(img, aspect="auto", origin="lower",
                  extent=(0, img.shape[1], y_vals[0] - Y_STEP_UM / 2,
                          y_vals[-1] + Y_STEP_UM / 2), interpolation="nearest")
        # contacts never recorded in any block: mark, don't let them read as anatomy
        dead_y, dead_x = np.where(is_contact & ~rec_img)
        if len(dead_y):
            ax.scatter(dead_x + 0.5, y_vals[dead_y], s=14, marker="x",
                       c="k", lw=0.9, zorder=3)
        ax.set_xticks([_shank_centre(r) for r in range(N_SHANKS)])
        ax.set_xticklabels([f"{s}?" for s in order], fontsize=6)
        ax.set_title(mouse, rotation=45, fontsize=9, ha="left")
        ax.set_xlim(0, img.shape[1])
        if k == 0:
            ax.set_ylabel("depth from tip (µm)", fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            ax.set_yticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)

        if units_panel:
            axu = axes[k * ncol + 1]
            tab = unit_depth_profile(mouse, y_vals)
            left = np.zeros(len(y_vals))
            for acro in tab.columns:
                vals = tab[acro].values
                axu.barh(y_vals, vals, height=Y_STEP_UM, left=left,
                         color=palette.get(acro, OUTSIDE_COLOUR),
                         edgecolor="none")
                left += vals
            axu.set_ylim(y_vals[0] - Y_STEP_UM / 2, y_vals[-1] + Y_STEP_UM / 2)
            axu.set_yticks([])
            axu.tick_params(labelsize=6)
            axu.set_xlabel("units", fontsize=7)
            for side in ("top", "right"):
                axu.spines[side].set_visible(False)

    handles = [Patch(facecolor=palette[a], label=f"{a} — {names.get(a, a)}")
               for a in palette if a in names]
    # tight_layout first, then hang the legend off the right-hand edge in figure
    # coordinates.  At 0.92 it sat *inside* the canvas and covered ly07; >1.0
    # puts it outside, and bbox_inches='tight' grows the saved canvas to fit.
    fig.tight_layout()
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
               bbox_transform=fig.transFigure, ncol=1, fontsize=7.5,
               frameon=False)
    fig.suptitle("Recording sites by brain region"
                 + ("   (right column of each pair: QC single units per depth)"
                    if units_panel else ""),
                 fontsize=11, y=1.03)
    fig.text(0.5, -0.03,
             "columns within each mouse are shanks in anterior→posterior order; "
             "shank numbers carry '?' because the array's physical orientation is "
             "not yet confirmed (SHANK_ORDER_VERIFIED = False).  ✗ = contact never "
             "recorded.  Units are unit-recordings pooled over blocks, not unique neurons.",
             ha="center", fontsize=6.5, style="italic")

    if save:
        _save(fig, save if isinstance(save, str) else "channel_regions", dpi=dpi)
    return fig


# --------------------------------------------------------------------------
# figure 2 -- probe tracts in the brain (brainrender)
# --------------------------------------------------------------------------

#: Region meshes to show, and their colours.  Keyed to figure 1's families.
TRACT_REGIONS = {"ENTl": "#c51b8a", "ENTm": "#3182bd", "SUB": "#756bb1",
                 "ProS": "#9e9ac8", "CA1": "#31a354"}

#: Centre of the allen_mouse volume in um (shape 528x320x456 @ 25 um), used as
#: the pivot when deriving rotated cameras.
BRAIN_CENTRE_UM = (6600.0, 4000.0, 5700.0)


def rotate_camera(camera: dict, degrees: float,
                  focal=BRAIN_CENTRE_UM) -> dict:
    """Spin a camera about the dorsoventral axis by ``degrees``.

    Axis 1 is DV in ``asr``, so the rotation happens in the (AP, LR) plane and
    the camera keeps its elevation -- which is what "turn the brain round"
    means for these views.
    """
    cam = {k: tuple(v) if isinstance(v, (list, tuple)) else v
           for k, v in camera.items()}
    px, py, pz = cam["pos"]
    fx, fy, fz = focal
    th = np.deg2rad(degrees)
    dx, dz = px - fx, pz - fz
    cam["pos"] = (fx + dx * np.cos(th) - dz * np.sin(th), py,
                  fz + dx * np.sin(th) + dz * np.cos(th))
    cam.setdefault("focal_point", tuple(focal))
    return cam


def camera(view: str) -> str | dict:
    """Resolve a view name to something ``Scene.render`` accepts.

    Passes brainrender's own presets straight through (``sagittal`` looks from
    +k onto the **left** lateral surface, the side the probes are in;
    ``sagittal2`` is the opposite hemisphere).  Names of the form
    ``<preset>_rot<deg>`` are derived by rotating the preset.
    """
    from brainrender.camera import cameras as _presets
    if view in _presets:
        return view
    if "_rot" in view:
        base, deg = view.rsplit("_rot", 1)
        return rotate_camera(_presets[base], float(deg))
    raise KeyError(f"unknown view {view!r}; presets are {sorted(_presets)}")


#: Left lateral (probe side), right lateral (through the other hemisphere),
#: dorsal, and two three-quarter views 90 degrees apart.
DEFAULT_VIEWS = ("sagittal", "sagittal2", "top",
                 "three_quarters", "three_quarters_rot90")


def track_coords(mouse: str):
    """Per-shank Allen-space tracks in um: ``{shank: (full, bank)}``.

    ``allen_atlas_coords`` are already in um in the atlas's ``asr`` frame, so
    they go straight into brainrender -- no rescaling, and none of the origin
    flips the HERBS pipeline needed.
    """
    df = allen_coords(mouse)
    cols = [f"allen_atlas_coords.{c}" for c in "ijk"]
    df = df.dropna(subset=cols)
    out = {}
    for sh in shank_ap_order(mouse):
        s = df[pr.shank_id(df["probe_coords.x"].values) == sh].sort_values("probe_coords.y")
        if not len(s):
            continue
        bank = s[s["probe_coords.y"] <= pr.RECORDED_BANK_MAX_UM]
        out[sh] = (s[cols].values.astype(float), bank[cols].values.astype(float))
    return out


def site_coords(mouse: str, *, recorded_only: bool = True):
    """Allen-space (µm) positions of the **recording sites** for one mouse.

    The recorded bank only -- not the whole modelled shank -- and by default
    only the channels actually recorded, so dead contacts are not drawn as if
    data came from them.
    """
    df = allen_coords(mouse)
    cols = [f"allen_atlas_coords.{c}" for c in "ijk"]
    b = df[df["probe_coords.y"] <= pr.RECORDED_BANK_MAX_UM].dropna(subset=cols)
    if recorded_only:
        rec = _recorded_contacts(mouse)
        keep = [(round(float(x), 1), round(float(y), 1)) in rec
                for x, y in zip(b["probe_coords.x"], b["probe_coords.y"])]
        b = b[keep]
    return b[cols].values.astype(float)


def plot_probe_tracts(mice=MICE, *, regions=tuple(TRACT_REGIONS),
                      root: bool = True, region_alpha: float = 0.12,
                      views=DEFAULT_VIEWS, save: bool | str = True,
                      interactive: bool = False, vector: bool = False,
                      scale: int = 3, zoom: float = 1.35,
                      background: str = "white", linewidth: int = 10,
                      sites_only: bool = False, site_radius: float = 50.0,
                      recorded_only: bool = True, root_alpha: float | None = None):
    """Brainrender scene: brain + region meshes + one line per shank per mouse.

    Each mouse gets one colour (shared with ``plot_mouse_legend``).  The full
    modelled shank is drawn thin and the recorded bank thick, so trajectory and
    recording extent are both visible.

    ``sites_only=True`` instead draws just the **recording sites** as points --
    no insertion track at all -- which is the honest picture of where data
    actually came from.  With ``recorded_only`` the dead channels are dropped
    too, so nothing is shown that never produced a sample.

    Saved as **PNG at ``scale``x** by default.  ``vector=True`` additionally
    writes PDF/SVG through vedo's GL2PS exporter, but that tessellates every
    mesh triangle in the brain surface and is very slow -- the raster output is
    the practical one for a scene with meshes in it.  (Figures 1 and 3 are
    matplotlib and are always vector.)

    Headless: run under ``xvfb-run -a``; ``OFFSCREEN`` is set automatically when
    no display is present.
    """
    import os
    import brainrender
    from brainrender import Scene
    from brainrender.actors import Line, Points
    import vedo

    vedo.settings.default_backend = "vtk"
    brainrender.settings.SHADER_STYLE = "plastic"
    brainrender.settings.SHOW_AXES = False
    if not os.environ.get("DISPLAY"):
        brainrender.settings.OFFSCREEN = True
    if not interactive:
        brainrender.settings.OFFSCREEN = True
    # the sites sit *inside* the brain, so with the default root alpha the
    # mouse colours are washed out by the surface in front of them
    if root_alpha is None:
        root_alpha = 0.08 if sites_only else 0.2
    brainrender.settings.ROOT_ALPHA = root_alpha        # read when root is built

    colours = mouse_palette(mice)
    # the probes are in the left hemisphere (contact k > midline, verified)
    scene = Scene(atlas_name="allen_mouse_25um", check_latest=False,
                  root=root, title=None)
    scene.plotter.axes = False
    for reg in regions:
        try:
            scene.add_brain_region(reg, alpha=region_alpha, hemisphere="left",
                                   color=TRACT_REGIONS.get(reg), silhouette=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"  region {reg}: not added ({type(exc).__name__}: {exc})")

    for mouse in mice:
        col = colours[mouse]
        if sites_only:
            pts = site_coords(mouse, recorded_only=recorded_only)
            # Points wants a colour *string*: an (r, g, b) tuple is read as a
            # per-point colour list and fails the length check
            actor = Points(pts, colors=to_hex(col), radius=site_radius, alpha=1)
            # unlit, or the shader darkens the spheres towards black and the
            # mouse colours stop being distinguishable inside the brain
            actor.lighting("off")
            scene.add(actor)
            print(f"  {mouse}: {len(pts)} recording sites added")
        else:
            tracks = track_coords(mouse)
            for sh, (full, bank) in tracks.items():
                scene.add(Line(full.tolist(), color=col, linewidth=3, alpha=0.5))
                if len(bank) > 1:
                    scene.add(Line(bank.tolist(), color=col, linewidth=linewidth))
            print(f"  {mouse}: {len(tracks)} shanks added")

    out = []
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        stem = save if isinstance(save, str) else (
            "probe_sites_in_brain" if sites_only else "probes_in_brain")
        for view in views:
            # resetcam=True is essential: vedo's zoom is *cumulative*, so
            # rendering several views on one scene otherwise compounds it
            # (the 5th view came out at 1.1**5).  Refitting first makes every
            # view start from the same baseline.
            scene.render(interactive=False, camera=camera(view), zoom=zoom,
                         resetcam=True)
            # must come *after* render: plotter.show() resets the background,
            # so setting it at construction time has no effect on the output
            scene.plotter.background(background)
            scene.plotter.renderer.SetBackground(*vedo.get_color(background))
            for ext in RASTER_FORMATS + (VECTOR_FORMATS if vector else ()):
                p = FIGURE_DIR / f"{stem}_{view}.{ext}"
                try:
                    # vedo routes .svg/.pdf/.eps through vtkGL2PSExporter --
                    # true vector, but it tessellates the whole brain mesh.
                    scene.plotter.screenshot(str(p), scale=scale if ext == "png" else 1)
                    out.append(p)
                    print(f"saved {p}")
                except Exception as exc:               # noqa: BLE001
                    print(f"  {ext} export failed for {view} "
                          f"({type(exc).__name__}: {exc})")
    elif interactive:
        scene.render(interactive=True)
    return scene, out


# --------------------------------------------------------------------------
# figure 3 -- mouse colour legend
# --------------------------------------------------------------------------

def plot_mouse_legend(mice=MICE, *, save: bool | str = False, spacing: float = 0.5):
    """The standalone mouse-colour swatch key (old repo's ``probe_legend.pdf``)."""
    colours = mouse_palette(mice)
    fig, ax = plt.subplots(figsize=(2, len(colours) * 0.6))
    ax.axis("off")
    for i, (name, colour) in enumerate(colours.items()):
        y = i * (1 + spacing)
        ax.add_patch(plt.Rectangle((0, y), 1, 1, color=colour))
        ax.text(1.5, y + 0.5, name, va="center", fontsize=12)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(colours) * (1 + spacing))
    ax.invert_yaxis()
    if save:
        _save(fig, save if isinstance(save, str) else "probe_legend")
    return fig


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def verify(mice=MICE, verbose: bool = True) -> pd.DataFrame:
    """Gate the figures against the census tables before believing them."""
    chans = pd.read_csv(ra.OUT_DIR / "channel_regions.csv")
    units = pd.read_csv(ra.OUT_DIR / "unit_regions.csv")
    summary = pd.read_csv(pr.BRAINREG_DIR / "final_fit_summary.csv").set_index("mouse")
    palette = region_palette()
    rows, ok = [], True

    for mouse in mice:
        rgb, rec, y_vals, order = site_grid(mouse, palette)      # asserts geometry
        b = bank_table(mouse)

        # grid composition must equal the census over the same contacts
        from_grid = b[b["recorded"]]["acronym"].value_counts()
        per_block = (chans[chans["mouse"] == mouse]
                     .drop_duplicates(["x_um", "y_um"])["acronym"].value_counts())
        same = from_grid.reindex(per_block.index, fill_value=0).equals(per_block)

        # units panel must total the census
        tab = unit_depth_profile(mouse, y_vals)
        n_units = int(tab.values.sum())
        n_census = int((units["mouse"] == mouse).sum())

        # the QC table's tip structure must be one of the structures the four
        # shank tips actually sit in.  Comparing against a single contact would
        # be wrong: at y=0 the shanks are in *different* structures.
        tips = set(b[b["y_um"] == b["y_um"].min()]["acronym"].dropna())
        qc_tip = summary.loc[mouse, "tip"] if mouse in summary.index else None
        tip_ok = qc_tip in tips if qc_tip else True

        row = {"mouse": mouse, "n_contacts": len(b),
               "n_recorded": int(b["recorded"].sum()),
               "shank_order_ant_post": "".join(str(s) for s in order),
               "census_match": bool(same),
               "units_grid": n_units, "units_census": n_census,
               "units_match": n_units == n_census,
               "tip_structures": "/".join(sorted(tips)),
               "qc_tip": qc_tip, "tip_match": bool(tip_ok)}
        ok &= (row["census_match"] and row["units_match"]
               and len(b) == 384 and tip_ok)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["passed"] = bool(ok)
    if verbose:
        print(df.to_string(index=False))
        print("\nVERIFY:", "PASS" if ok else "FAIL")
    return df
