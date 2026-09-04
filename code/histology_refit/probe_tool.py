"""Interactive shank placement: put the probe where your eye says it goes.

Sagittal and coronal DiI panels with live controls over the shanks, initialised
at a subject's current best fit.  Built because the judgement was never the
bottleneck -- Adam's read of the histology beat the optimiser on ah10's depth and
ly07's angle every time -- the *interface* was, when each attempt meant editing a
batch script and waiting a minute for a PNG.

Usage, in a notebook running the ``histology`` env::

    import probe_tool as pt
    tool = pt.place('ly07')      # sliders + panels, initialised at the saved fit
    ...                          # move things until it looks right
    tool.save(confidence='confident', note='placed by hand off the DiI')

Design notes worth knowing before changing anything:

* **Entry + tip, not angles.**  Two points give ``v_axis`` and ``depth`` directly
  through :func:`probe_refit.plane_from_track`, which sidesteps the pivot
  problem: rotating about the surface anchor swung ah10's tip 574 um for a
  9.7 deg change, so "what does the angle slider rotate about" has no innocent
  answer.  Setting both ends has only one interpretation.
* **``offset_x`` is pinned to 0 and not exposed.**  It translates the probe along
  ``u_axis``, which is exactly what moving the entry point does; exposing both
  would make the parameterisation redundant.
* **Panels are slab max-intensity projections.**  The four shanks span ~750 um,
  so no single slice contains them all, and squashing every contact onto one
  slice (as the older figures do) quietly hides shanks.
* **Atlas contours come from the centre slice, never the MIP** -- a
  max-projection of label ids is meaningless.
* **A placement made here is authoritative.**  It saves as ``manual_3d`` and
  grades ``annotated``: the harness reports metrics beside it but never
  re-optimises it against the dye or fails it.  Only mechanical validity
  (:func:`probe_refit.validate_track`) blocks a save.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import probe_refit as pr
from brainreg_probe import probeinterface_tracing as pit
from brainreg_probe import plot_util_func as puf

#: Region groups outlined on the panels, and their colours.
CONTOUR_GROUPS = {
    "ENTl": "#00E676",
    "ENTm": "#FF9100",
    "SUB": "#E040FB",
}

DEFAULT_SLAB_UM = 400.0

_GROUP_IDS: dict = {}
_MIP_CACHE: dict = {}


# --------------------------------------------------------------------------
# atlas region masks
# --------------------------------------------------------------------------


def _group_ids() -> dict:
    """Allen structure ids belonging to each contour group, by acronym prefix.

    Deliberately a small id list per group rather than a relabelled copy of the
    atlas volume.  Building a whole group volume needs a lookup indexed by
    structure id, and Allen ids reach 614,454,277 -- that route allocates
    gigabytes and gets OOM-killed (it did).  Masking a single slice with
    ``np.isin`` against ~20 ids is ~10 ms and needs no extra memory.
    """
    global _GROUP_IDS
    if _GROUP_IDS:
        return _GROUP_IDS
    info = pit.ALLEN_ATLAS_INFO_DF
    acro = info["acronym"].astype(str)
    for group in CONTOUR_GROUPS:
        _GROUP_IDS[group] = info.loc[acro.str.startswith(group), "id"].values.astype(np.int64)
    return _GROUP_IDS


def _slab_mip(data: dict, axis: int, centre: int, slab_vox: int) -> np.ndarray:
    """Max-intensity projection of the DiI over a slab, cached.

    Cached on (axis, centre, thickness) so that tweaking theta or depth -- which
    barely moves the probe centroid -- reuses the projection.  Measured ~83 ms
    for both panels at a 400 um slab, so this cache is what keeps the tool
    responsive.
    """
    key = (id(data), axis, int(centre), int(slab_vox))
    if key in _MIP_CACHE:
        return _MIP_CACHE[key]
    sig = data["signal_data"]
    lo = max(int(centre) - slab_vox // 2, 0)
    hi = min(int(centre) + slab_vox // 2 + 1, sig.shape[axis])
    if axis == 2:
        img = sig[:, :, lo:hi].max(axis=2).T
    elif axis == 0:
        img = sig[lo:hi, :, :].max(axis=0)
    else:
        img = sig[:, lo:hi, :].max(axis=1).T
    img = puf.adjust_contrast(img)
    if len(_MIP_CACHE) > 24:
        _MIP_CACHE.clear()
    _MIP_CACHE[key] = img
    return img


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def fit_to_track(plane: dict, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recover (entry, tip) voxel coordinates from a stored plane + params.

    Inverse of :func:`probe_refit.plane_from_track`: the entry is the plane's
    ``surface_coord`` and the tip lies ``probe_depth`` along ``-v_axis``.

    **``offset_x`` is absorbed into the entry point, not discarded.**  The tool
    pins ``offset_x`` to 0 because it is redundant with moving the entry, but
    every stored fit carries a non-zero one (ly05's is -413 um, over half the
    750 um shank span), so simply dropping it would open the tool on a probe
    shifted sideways from the fit it claims to be showing.  ``transform_2d_probe``
    adds ``offset_x`` in um to the in-plane x before dividing by the voxel size,
    so the equivalent shift is ``offset_x / VOXEL_SIZE_UM`` along ``u_axis``.
    """
    entry = np.asarray(plane["surface_coord"], float)
    v = np.asarray(plane["v_axis"], float)
    u = np.asarray(plane["u_axis"], float)
    entry = entry + u * (float(params.get("offset_x", 0.0)) / pr.VOXEL_SIZE_UM)
    tip = entry - v * (float(params["probe_depth"]) / pr.VOXEL_SIZE_UM)
    return entry, tip


def track_to_fit(entry, tip, *, theta: float = 0.0,
                 width_scaling: float = 0.9, shrinkage_pct: float = 5.0,
                 contact_face_axis: str = pr.CONTACT_FACE_AXIS,
                 reference_u=None) -> tuple[dict, dict]:
    """Build (plane, params) from two points plus the parameters they cannot fix.

    ``offset_x`` is always 0 -- see the module docstring.
    """
    plane, depth = pr.plane_from_track(entry, tip, contact_face_axis=contact_face_axis)
    if reference_u is not None:
        # Trust the stored u_axis: re-imposing the contact-face convention here
        # would mirror the probe relative to the fit we are initialising from.
        plane = pr.build_plane(plane["v_axis"], entry, centroid=plane["centroid"],
                               reference_u=reference_u,
                               contact_face_axis=contact_face_axis,
                               enforce_contact_face=False)
    params = {"probe_depth": depth, "brain_shrinkage_pct": shrinkage_pct,
              "probe_width_scaling": width_scaling, "theta": theta,
              "offset_x": 0.0}
    return plane, params


def bank_summary(probe_df: pd.DataFrame) -> dict:
    """Composition of the recorded bank -- the numbers you steer by."""
    bank = probe_df[probe_df["probe_coords.y"] <= pr.RECORDED_BANK_MAX_UM]
    a = bank["structure.acronym"].fillna("none")
    if not len(a):
        return {"n": 0}
    tip = bank.nsmallest(24, "probe_coords.y")["structure.acronym"].dropna()
    return {
        "n": int(len(bank)),
        "deep_ENTl": float(a.isin(["ENTl5", "ENTl6a"]).mean()),
        "sup_ENTl": float(a.isin(["ENTl1", "ENTl2", "ENTl3"]).mean()),
        "any_ENTl": float(a.str.startswith("ENTl").mean()),
        "SUB_ProS": float(a.isin(["SUB", "ProS"]).mean()),
        "ENTm": float(a.str.startswith("ENTm").mean()),
        "other": float((~a.str.startswith(("ENTl", "ENTm", "SUB", "ProS"))).mean()),
        "tip": tip.mode().iat[0] if len(tip) else "none",
    }


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------


class ProbePlacer:
    """Slider-driven placement of the shanks against the DiI and the atlas."""

    def __init__(self, subject: str, *, slab_um: float = DEFAULT_SLAB_UM,
                 data: dict | None = None, signal_df: pd.DataFrame | None = None):
        self.subject = subject
        self.data = data if data is not None else pr.load_volumes(subject)
        try:
            self.signal_df = (signal_df if signal_df is not None
                              else pr.load_signal_df(subject))
        except ValueError:
            self.signal_df = None
        self.fit = pr.load_fit(subject)
        self.plane0, self.params0 = pr.split_fit(self.fit)
        self.entry0, self.tip0 = fit_to_track(self.plane0, self.params0)
        self.reference_u = np.asarray(self.plane0["u_axis"], float)
        self.slab_um = slab_um
        self.group_ids = _group_ids()
        self._last = None

    # -- current state ----------------------------------------------------
    def current(self, entry, tip, theta_deg, width, shrink):
        plane, params = track_to_fit(entry, tip, theta=np.radians(theta_deg),
                                     width_scaling=width, shrinkage_pct=shrink,
                                     reference_u=self.reference_u)
        probe_df = pr.project_probe(plane, params, self.data)
        qc = pr.qc_probe_fit(self.subject, plane, params, probe_df,
                             signal_df=self.signal_df)
        self._last = {"plane": plane, "params": params, "probe_df": probe_df,
                      "qc": qc, "signal_df": self.signal_df,
                      "entry": np.asarray(entry, float),
                      "tip": np.asarray(tip, float)}
        return self._last

    # -- drawing ----------------------------------------------------------
    def draw(self, state, axes):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        coords = state["probe_df"][[f"downsample_coords.{c}" for c in "ijk"]].values
        bank = state["probe_df"]["probe_coords.y"].values <= pr.RECORDED_BANK_MAX_UM
        centre = np.round(coords.mean(0)).astype(int)
        slab_vox = max(int(self.slab_um / pr.VOXEL_SIZE_UM), 1)
        dye = (self.signal_df[["i", "j", "k"]].values
               if self.signal_df is not None else None)

        for ax, (axis, xi, yi, label) in zip(
                axes, [(2, 0, 1, "sagittal"), (0, 2, 1, "coronal")]):
            ax.clear()
            c = int(np.clip(centre[axis], 0, self.data["signal_data"].shape[axis] - 1))
            ax.imshow(_slab_mip(self.data, axis, c, slab_vox), cmap="gray",
                      origin="upper")
            # atlas contours from the CENTRE slice, never the MIP -- a
            # max-projection of label ids is meaningless
            atlas = self.data["atlas_registration_data"]
            sl_ids = atlas[:, :, c].T if axis == 2 else atlas[c, :, :]
            for group, colour in CONTOUR_GROUPS.items():
                mask = np.isin(sl_ids, self.group_ids[group])
                if mask.any():
                    ax.contour(mask, levels=[0.5], colors=[colour],
                               linewidths=1.0, alpha=0.85)
            ax.scatter(coords[:, xi], coords[:, yi], s=3, c="#FF2D2D", alpha=0.55, lw=0)
            ax.scatter(coords[bank, xi], coords[bank, yi], s=8, c="#FFE800",
                       alpha=0.95, lw=0)
            if dye is not None:
                ax.scatter(dye[:, xi], dye[:, yi], s=2, c="#00CFFF", alpha=0.5, lw=0)
            for pt, mk in ((state["entry"], "o"), (state["tip"], "x")):
                ax.scatter([pt[xi]], [pt[yi]], s=70, marker=mk, c="#FFFFFF",
                           lw=1.6, zorder=5)
            lo = coords[:, [xi, yi]].min(0) - 70
            hi = coords[:, [xi, yi]].max(0) + 70
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(hi[1], lo[1])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{label}  (slab {self.slab_um:.0f} µm MIP)", fontsize=10)

        handles = [Line2D([], [], marker="o", ls="", color="#00CFFF", label="DiI"),
                   Line2D([], [], marker="o", ls="", color="#FF2D2D", label="contacts"),
                   Line2D([], [], marker="o", ls="", color="#FFE800", label="recorded bank"),
                   Line2D([], [], marker="o", ls="", color="#FFFFFF", label="entry / tip")]
        handles += [Line2D([], [], color=c, label=g) for g, c in CONTOUR_GROUPS.items()]
        axes[0].legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.6)

    def readout(self, state) -> str:
        qc, s = state["qc"], bank_summary(state["probe_df"])
        ang = pr.trajectory_angles(state["plane"]["v_axis"])
        sb = pr.shank_breakdown(state)
        lines = [
            f"trajectory   lateral {ang['lateral_deg']:6.1f}°   AP {ang['ap_deg']:6.1f}°"
            f"   θ {qc['theta_deg']:5.1f}°   depth {qc['probe_depth']:6.0f} µm"
            f"   ({qc['depth_over_dye']:.2f}× dye)",
            f"dye fit      resid {qc['resid_signal2contact_um']:5.0f} µm"
            f"   fit-vs-dye {qc['fit_vs_dye_axis_deg']:4.1f}°"
            f"   coverage {qc['signal_coverage']:.2f}",
            f"bank (n={s.get('n', 0)})  deepENTl {s.get('deep_ENTl', 0):.2f}"
            f"  supENTl {s.get('sup_ENTl', 0):.2f}  SUB/ProS {s.get('SUB_ProS', 0):.2f}"
            f"  ENTm {s.get('ENTm', 0):.2f}  other {s.get('other', 0):.2f}"
            f"  tip {s.get('tip', '?')}",
            "per shank (anterior → posterior)   "
            + "   ".join(f"{r.shank_label}:{r.ENTl_frac:.2f}" for r in sb.itertuples()),
        ]
        if qc["flags"]:
            lines.append("FLAGS: " + ", ".join(qc["flags"]))
        if qc["advisories"]:
            lines.append("advisory: " + ", ".join(qc["advisories"]))
        if not pr.SHANK_ORDER_VERIFIED:
            lines.append("shank numbers carry '?' — channel↔position mapping "
                         "is an exact degeneracy, pending the surgery notes")
        return "\n".join(lines)

    # -- saving -----------------------------------------------------------
    def save(self, *, confidence: str = "unset", note: str = "") -> dict:
        """Write the current placement.  Authoritative: graded ``annotated``."""
        if self._last is None:
            raise RuntimeError("nothing placed yet — move a slider first")
        st = self._last
        pr.validate_track(st["entry"], st["tip"], self.data["signal_data"].shape)
        full = pr.load_volumes(self.subject, fast=False)
        probe_df = pr.project_probe(st["plane"], st["params"], full, with_allen=True)
        result = {"probe_df": probe_df, "plane": st["plane"], "params": st["params"],
                  "qc": pr.qc_probe_fit(self.subject, st["plane"], st["params"],
                                        probe_df, signal_df=self.signal_df)}
        return pr.save_fit(self.subject, result, fit_method="manual_3d",
                           confidence=confidence, note=note,
                           manual_inputs={"entry": st["entry"].tolist(),
                                          "tip": st["tip"].tolist(),
                                          "theta_deg": float(np.degrees(
                                              st["params"]["theta"])),
                                          "slab_um": self.slab_um,
                                          "placed_with": "probe_tool.ProbePlacer"})


def place(subject: str, *, slab_um: float = DEFAULT_SLAB_UM,
          step: int = 2, data: dict | None = None):
    """Launch the placement UI for one subject.  Returns the ProbePlacer.

    Sliders update on release (``continuous_update=False``) -- a redraw is
    ~150-200 ms worst case and the slab cache removes most of that.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    tool = ProbePlacer(subject, slab_um=slab_um, data=data)
    shape = tool.data["signal_data"].shape

    def _xyz(name, vec, colour):
        return [widgets.IntSlider(value=int(round(vec[d])), min=0, max=shape[d] - 1,
                                  step=step, description=f"{name} {'ijk'[d]}",
                                  continuous_update=False,
                                  style={"description_width": "60px"},
                                  layout=widgets.Layout(width="330px"))
                for d in range(3)]

    e_s = _xyz("entry", tool.entry0, "#FFFFFF")
    t_s = _xyz("tip", tool.tip0, "#FFFFFF")
    th = widgets.FloatSlider(value=float(np.degrees(tool.params0["theta"])),
                             min=-15, max=15, step=0.25, description="theta°",
                             continuous_update=False,
                             style={"description_width": "60px"},
                             layout=widgets.Layout(width="330px"))
    wd = widgets.FloatSlider(value=float(tool.params0["probe_width_scaling"]),
                             min=0.75, max=1.05, step=0.01, description="width",
                             continuous_update=False,
                             style={"description_width": "60px"},
                             layout=widgets.Layout(width="330px"))
    sh = widgets.FloatSlider(value=float(tool.params0["brain_shrinkage_pct"]),
                             min=0.0, max=8.0, step=0.5, description="shrink%",
                             continuous_update=False,
                             style={"description_width": "60px"},
                             layout=widgets.Layout(width="330px"))
    slab = widgets.FloatSlider(value=slab_um, min=100, max=1000, step=100,
                               description="slab µm", continuous_update=False,
                               style={"description_width": "60px"},
                               layout=widgets.Layout(width="330px"))
    reset = widgets.Button(description="reset to saved fit", icon="undo")
    conf = widgets.Dropdown(options=["unset", "confident", "uncertain"],
                            value="unset", description="confidence",
                            style={"description_width": "80px"},
                            layout=widgets.Layout(width="260px"))
    note = widgets.Text(description="note", placeholder="why this placement",
                        style={"description_width": "50px"},
                        layout=widgets.Layout(width="560px"))
    save_btn = widgets.Button(description="SAVE placement", button_style="success",
                              icon="check")
    out = widgets.Output()
    msg = widgets.HTML()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plt.close(fig)

    def _redraw(*_):
        tool.slab_um = slab.value
        entry = np.array([s.value for s in e_s], float)
        tip = np.array([s.value for s in t_s], float)
        with out:
            out.clear_output(wait=True)
            try:
                st = tool.current(entry, tip, th.value, wd.value, sh.value)
            except ValueError as exc:
                print(f"invalid placement: {exc}")
                return
            tool.draw(st, axes)
            fig.suptitle(f"{tool.subject} — hand placement", fontsize=12)
            fig.tight_layout()
            display(fig)
            print(tool.readout(st))

    def _reset(_):
        for s, v in zip(e_s, tool.entry0):
            s.value = int(round(v))
        for s, v in zip(t_s, tool.tip0):
            s.value = int(round(v))
        th.value = float(np.degrees(tool.params0["theta"]))
        wd.value = float(tool.params0["probe_width_scaling"])
        sh.value = float(tool.params0["brain_shrinkage_pct"])
        _redraw()

    def _save(_):
        try:
            payload = tool.save(confidence=conf.value, note=note.value)
            msg.value = (f"<b style='color:#2e7d32'>saved</b> {tool.subject} — "
                         f"{payload['fit_method']}, grade {payload['grade']}, "
                         f"confidence {payload['confidence']}")
        except Exception as exc:
            msg.value = f"<b style='color:#b00020'>not saved:</b> {exc}"

    for w in e_s + t_s + [th, wd, sh, slab]:
        w.observe(_redraw, names="value")
    reset.on_click(_reset)
    save_btn.on_click(_save)

    display(widgets.VBox([
        widgets.HBox([widgets.VBox(e_s), widgets.VBox(t_s),
                      widgets.VBox([th, wd, sh, slab])]),
        widgets.HBox([reset, conf, save_btn]), note, msg, out]))
    _redraw()
    return tool
