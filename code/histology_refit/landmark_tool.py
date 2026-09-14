"""Landmark target-registration error (check F of the registration QC).

Click ~10 anatomical landmarks on a mouse's autofluorescence (sample space) and
once on the unfiltered Allen reference (atlas space); map the sample clicks through
the deformation field and report the distance to the atlas clicks.  That distance --
the target registration error, TRE -- is the one registration number a reader can
cite.  Landmarks near the LEC are flagged so the error *at the target* can be
reported separately from the whole-brain median.

Usage, in a notebook running the ``histology`` env with ``%matplotlib widget``
(ipympl 0.10 is installed)::

    import landmark_tool as lt
    lt.LandmarkPicker('atlas').show()      # once: click each landmark on the atlas
    lt.LandmarkPicker('ah08').show()       # per mouse
    lt.tre_table('ah08')                   # after both are saved

Clicking places the crosshair; the x/y sliders nudge it; ``record`` stores it under
the selected landmark (repeated records of the same landmark are kept as repeats,
which is how :func:`click_repeatability` measures the click noise floor).  Without
the ipympl backend the sliders alone still work.

Coordinates are stored as sample voxels (10 um, asr) for a mouse and as 10 um atlas
voxels for the atlas (clicked on the 25 um reference and scaled x2.5).  Sagittal and
horizontal views of a mouse come from every 2nd page (20 um AP sampling), loaded on
first use (~600 MB).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import registration_qc as rq
from brainreg_probe import plot_util_func as puf

#: Landmarks visible in green autofluorescence and in the Allen reference.  ``near_lec``
#: ones sit within ~2 mm of the recorded bank and carry the error that matters.
LANDMARKS = [
    {"name": "rhinal_fissure", "near_lec": True,
     "desc": "Rhinal fissure (lateral surface notch) in the LEFT hemisphere at the AP level of the recorded bank"},
    {"name": "dg_crest", "near_lec": True,
     "desc": "Crest of the dentate gyrus granule-cell V, LEFT, at the bank level (ventral hippocampus)"},
    {"name": "hipp_fissure_lateral", "near_lec": True,
     "desc": "Lateral end of the hippocampal fissure, LEFT, at the bank level"},
    {"name": "lv_temporal_horn", "near_lec": True,
     "desc": "Ventral tip of the lateral ventricle temporal horn, LEFT, at the ENT level"},
    {"name": "posterior_commissure", "near_lec": False,
     "desc": "Posterior commissure at the midline, in its densest coronal section"},
    {"name": "anterior_commissure", "near_lec": False,
     "desc": "Anterior commissure decussation at the midline (may be near the anterior cut face)"},
    {"name": "habenular_commissure", "near_lec": False,
     "desc": "Habenular commissure / dorsal tip of the third ventricle"},
    {"name": "aqueduct_dorsal", "near_lec": False,
     "desc": "Cerebral aqueduct, dorsal wall, at the level of the posterior commissure"},
    {"name": "interpeduncular", "near_lec": False,
     "desc": "Ventral midline of the interpeduncular fossa / mammillary body"},
    {"name": "ic_cb_notch", "near_lec": False,
     "desc": "Dorsal midline notch between inferior colliculus and cerebellum"},
]
LANDMARK_NAMES = [l["name"] for l in LANDMARKS]
NEAR_LEC = {l["name"] for l in LANDMARKS if l["near_lec"]}

ATLAS_PATH = rq.BRAINREG_DIR / "atlas_landmarks.json"


def landmark_path(source: str) -> Path:
    return ATLAS_PATH if source == "atlas" else rq.subject_paths(source)["landmarks"]


def load_landmarks(source: str) -> dict:
    p = landmark_path(source)
    return json.loads(p.read_text()) if p.exists() else {}


def save_landmarks(source: str, marks: dict) -> Path:
    p = landmark_path(source)
    p.write_text(json.dumps(marks, indent=2))
    return p


def _record(marks: dict, name: str, ijk, *, view: str, slice_: int) -> dict:
    entry = marks.setdefault(name, {"repeats": []})
    entry["repeats"].append({"ijk": [float(v) for v in ijk], "view": view, "slice": int(slice_),
                             "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")})
    entry["ijk"] = np.mean([r["ijk"] for r in entry["repeats"]], axis=0).tolist()
    return entry


# --------------------------------------------------------------------------
# volume access for the picker
# --------------------------------------------------------------------------


class _Volume:
    """Coronal pages on demand; sagittal/horizontal from a strided copy loaded on first use."""

    def __init__(self, source: str):
        self.source = source
        if source == "atlas":
            self.vol = rq.atlas25()["reference"]
            self.scale = rq.ATLAS_25_SCALE          # stored coords = index * scale (10 um)
            self.stride = 1
            self.shape = self.vol.shape
        else:
            self.paths = rq.subject_paths(source)
            self.shape = rq.volume_shape(self.paths["green"])
            self.scale = 1.0
            self.stride = 2
            self.vol = None

    def _strided(self):
        if self.vol is None:
            self.vol = rq.read_pages(self.paths["green"], 0, self.shape[0], self.stride)
        return self.vol

    def coronal(self, i: int) -> np.ndarray:
        if self.source == "atlas":
            return self.vol[int(np.clip(i, 0, self.shape[0] - 1))]
        return rq.read_page(self.paths["green"], int(np.clip(i, 0, self.shape[0] - 1)))

    def sagittal(self, k: int) -> np.ndarray:            # rows j, cols i (strided)
        v = self._strided()
        return v[:, :, int(np.clip(k, 0, self.shape[2] - 1))].T

    def horizontal(self, j: int) -> np.ndarray:          # rows k, cols i (strided)
        v = self._strided()
        return v[:, int(np.clip(j, 0, self.shape[1] - 1)), :].T


# --------------------------------------------------------------------------
# the picker
# --------------------------------------------------------------------------


class LandmarkPicker:
    """Click-to-place landmark picker with a slice slider and x/y nudge sliders.

    ``view`` semantics (image x, y -> voxel):
      coronal    : img = vol[i]       -> k = x, j = y
      sagittal   : img = vol[:,:,k].T -> i = x * stride, j = y
      horizontal : img = vol[:,j,:].T -> i = x * stride, k = y
    Stored coordinates are in 10 um voxels of the *source* space.
    """

    def __init__(self, source: str, *, contrast=(1, 99.5)):
        self.source = source
        self.vol = _Volume(source)
        self.marks = load_landmarks(source)
        self.contrast = contrast
        self.view = "coronal"
        self.click = None                 # (x, y) in image pixels of the current view
        self._fig = None
        self._ax = None
        self._widgets = None

    # -- geometry -----------------------------------------------------------
    def _slice_max(self):
        return {"coronal": self.vol.shape[0], "sagittal": self.vol.shape[2], "horizontal": self.vol.shape[1]}[self.view] - 1

    def _image(self, s: int) -> np.ndarray:
        return {"coronal": self.vol.coronal, "sagittal": self.vol.sagittal, "horizontal": self.vol.horizontal}[self.view](s)

    def _xy_to_ijk(self, x: float, y: float, s: int) -> np.ndarray:
        st = self.vol.stride if self.view != "coronal" else 1
        if self.view == "coronal":
            ijk = (s, y, x)
        elif self.view == "sagittal":
            ijk = (x * st, y, s)
        else:
            ijk = (x * st, s, y)
        return np.asarray(ijk, float) * self.vol.scale

    def _ijk_to_xy(self, ijk) -> tuple[float, float, float]:
        ijk = np.asarray(ijk, float) / self.vol.scale
        st = self.vol.stride if self.view != "coronal" else 1
        if self.view == "coronal":
            return ijk[2], ijk[1], ijk[0]
        if self.view == "sagittal":
            return ijk[0] / st, ijk[1], ijk[2]
        return ijk[0] / st, ijk[2], ijk[1]

    def default_slice(self) -> int:
        if self.source == "atlas":
            return {"coronal": 950, "sagittal": 320, "horizontal": 480}[self.view] // (1 if self.view != "coronal" else 1)
        geo = rq.track_geometry(self.source)
        c = geo["bank_centre"]
        return int({"coronal": c[0], "sagittal": c[2], "horizontal": c[1]}[self.view])

    # -- drawing ------------------------------------------------------------
    def draw(self, s: int, name: str, show_marks: bool = True):
        import matplotlib.pyplot as plt
        ax = self._ax
        ax.clear()
        img = self._image(s)
        ax.imshow(puf.adjust_contrast(img, *self.contrast), cmap="gray", origin="upper",
                  aspect=(self.vol.stride if self.view != "coronal" else 1) if False else "auto")
        if show_marks:
            for nm, m in self.marks.items():
                if "ijk" not in m:
                    continue
                x, y, sl = self._ijk_to_xy(m["ijk"])
                if abs(sl - s) <= 3:
                    ax.plot(x, y, "o", ms=7, mfc="none", mec="#00E676" if nm == name else "#FFE800", mew=1.4)
                    ax.annotate(nm, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6,
                                color="#00E676" if nm == name else "#FFE800")
        if self.click is not None:
            x, y = self.click
            ax.axhline(y, color="#FF2D2D", lw=0.6, alpha=0.8); ax.axvline(x, color="#FF2D2D", lw=0.6, alpha=0.8)
        st = self.vol.stride if self.view != "coronal" else 1
        ax.set_title(f"{self.source} — {self.view} slice {s}"
                     + (f"  (x axis = every {st}nd page)" if st > 1 else "") + f"   landmark: {name}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        self._fig.canvas.draw_idle()

    def show(self):
        import ipywidgets as widgets
        import matplotlib
        import matplotlib.pyplot as plt
        from IPython.display import display

        interactive = "ipympl" in matplotlib.get_backend().lower() or "widget" in matplotlib.get_backend().lower()
        self._fig, self._ax = plt.subplots(figsize=(9, 7))
        view_w = widgets.Dropdown(options=["coronal", "sagittal", "horizontal"], value="coronal", description="view")
        slice_w = widgets.IntSlider(value=self.default_slice(), min=0, max=self._slice_max(), description="slice",
                                    continuous_update=False, layout=widgets.Layout(width="60%"))
        name_w = widgets.Dropdown(options=LANDMARK_NAMES, value=LANDMARK_NAMES[0], description="landmark",
                                  layout=widgets.Layout(width="40%"))
        desc_w = widgets.HTML()
        x_w = widgets.FloatSlider(value=0, min=0, max=1, step=0.5, description="x", continuous_update=False,
                                  layout=widgets.Layout(width="45%"))
        y_w = widgets.FloatSlider(value=0, min=0, max=1, step=0.5, description="y", continuous_update=False,
                                  layout=widgets.Layout(width="45%"))
        rec = widgets.Button(description="record", button_style="success", icon="check")
        dele = widgets.Button(description="delete landmark", icon="trash")
        save = widgets.Button(description="SAVE", button_style="primary", icon="save")
        bank = widgets.Button(description="jump to bank / default", icon="crosshairs")
        msg = widgets.HTML()
        table = widgets.Output()

        def _set_ranges():
            img_shape = self._image(slice_w.value).shape
            x_w.max, y_w.max = img_shape[1] - 1, img_shape[0] - 1

        def _redraw(*_):
            self.draw(slice_w.value, name_w.value)
            self._refresh_table(table)

        def _view(change):
            self.view = change["new"]
            self.click = None
            slice_w.max = self._slice_max()
            slice_w.value = self.default_slice()
            _set_ranges(); _redraw()

        def _name(change):
            d = next(l for l in LANDMARKS if l["name"] == change["new"])
            desc_w.value = f"<i>{d['desc']}</i>" + ("  <b>[near LEC]</b>" if d["near_lec"] else "")
            _redraw()

        def _nudge(*_):
            self.click = (x_w.value, y_w.value)
            self.draw(slice_w.value, name_w.value)

        def _on_click(event):
            if event.inaxes != self._ax or event.button != 1:
                return
            if self._fig.canvas.toolbar is not None and getattr(self._fig.canvas.toolbar, "mode", ""):
                return                                    # pan/zoom active: ignore
            self.click = (event.xdata, event.ydata)
            x_w.value, y_w.value = float(event.xdata), float(event.ydata)
            self.draw(slice_w.value, name_w.value)

        def _record_click(_):
            if self.click is None:
                msg.value = "<b style='color:#b00020'>click (or set x/y) first</b>"
                return
            ijk = self._xy_to_ijk(self.click[0], self.click[1], slice_w.value)
            e = _record(self.marks, name_w.value, ijk, view=self.view, slice_=slice_w.value)
            msg.value = (f"recorded <b>{name_w.value}</b> at ijk {np.round(ijk, 1).tolist()} "
                         f"(repeat #{len(e['repeats'])}; not yet saved)")
            _redraw()

        def _delete(_):
            self.marks.pop(name_w.value, None)
            msg.value = f"deleted {name_w.value} (not yet saved)"
            _redraw()

        def _save(_):
            p = save_landmarks(self.source, self.marks)
            msg.value = f"<b style='color:#2e7d32'>saved</b> {len(self.marks)} landmarks → {p}"

        def _bank(_):
            slice_w.value = self.default_slice()

        view_w.observe(_view, names="value"); slice_w.observe(_redraw, names="value")
        name_w.observe(_name, names="value")
        x_w.observe(_nudge, names="value"); y_w.observe(_nudge, names="value")
        rec.on_click(_record_click); dele.on_click(_delete); save.on_click(_save); bank.on_click(_bank)
        if interactive:
            self._fig.canvas.mpl_connect("button_press_event", _on_click)
        else:
            msg.value = ("<span style='color:#b26a00'>non-interactive backend: run <code>%matplotlib widget</code> "
                         "for click-to-place; the x/y sliders still work</span>")
        _set_ranges()
        _name({"new": name_w.value})
        display(widgets.VBox([widgets.HBox([view_w, slice_w, bank]), widgets.HBox([name_w, desc_w]),
                              widgets.HBox([x_w, y_w]), widgets.HBox([rec, dele, save]), msg, table]))
        if not interactive:
            display(self._fig)
        self._widgets = dict(view=view_w, slice=slice_w, name=name_w, x=x_w, y=y_w)
        _redraw()
        return self

    def _refresh_table(self, out):
        from IPython.display import display
        with out:
            out.clear_output(wait=True)
            rows = [{"landmark": n, "ijk": np.round(m["ijk"], 1).tolist(), "repeats": len(m.get("repeats", [])),
                     "near_lec": n in NEAR_LEC} for n, m in self.marks.items() if "ijk" in m]
            if rows:
                display(pd.DataFrame(rows))
            else:
                print("no landmarks recorded yet")


# --------------------------------------------------------------------------
# TRE
# --------------------------------------------------------------------------


def tre_table(subject: str) -> pd.DataFrame:
    """Per-landmark TRE (um): sample click mapped through the deformation field vs the atlas click."""
    mouse = load_landmarks(subject)
    atlas = load_landmarks("atlas")
    names = [n for n in LANDMARK_NAMES if n in mouse and n in atlas and "ijk" in mouse[n] and "ijk" in atlas[n]]
    if not names:
        return pd.DataFrame()
    pts = np.array([mouse[n]["ijk"] for n in names], float)
    mapped_um = rq.sample_vox_to_atlas_mm(subject, pts) * 1000.0
    atlas_um = np.array([atlas[n]["ijk"] for n in names], float) * rq.VOX_UM
    d = mapped_um - atlas_um
    df = pd.DataFrame({"subject": subject, "landmark": names, "near_lec": [n in NEAR_LEC for n in names],
                       "sample_ijk": [np.round(p, 1).tolist() for p in pts],
                       "mapped_atlas_um": [np.round(m, 0).tolist() for m in mapped_um],
                       "atlas_um": [np.round(a, 0).tolist() for a in atlas_um],
                       "d_ap_um": d[:, 0], "d_dv_um": d[:, 1], "d_lr_um": d[:, 2],
                       "tre_um": np.linalg.norm(d, axis=1),
                       "n_repeats_sample": [len(mouse[n].get("repeats", [])) for n in names]})
    return df


def tre_summary(df: pd.DataFrame) -> dict:
    if df is None or not len(df):
        return {"n_landmarks": 0}
    near = df[df["near_lec"]]; far = df[~df["near_lec"]]
    return {"n_landmarks": int(len(df)),
            "tre_median_um": float(df["tre_um"].median()),
            "tre_median_near_lec_um": float(near["tre_um"].median()) if len(near) else np.nan,
            "tre_median_elsewhere_um": float(far["tre_um"].median()) if len(far) else np.nan,
            "tre_max_um": float(df["tre_um"].max()),
            "bias_ap_um": float(df["d_ap_um"].median()), "bias_dv_um": float(df["d_dv_um"].median()),
            "bias_lr_um": float(df["d_lr_um"].median())}


def click_repeatability(source: str) -> pd.DataFrame:
    """Spread of repeated clicks per landmark (um): the noise floor below which TRE is not interpretable."""
    marks = load_landmarks(source)
    scale = rq.VOX_UM
    rows = []
    for n, m in marks.items():
        reps = np.array([r["ijk"] for r in m.get("repeats", [])], float)
        if len(reps) >= 2:
            c = reps.mean(0)
            rows.append({"source": source, "landmark": n, "n": len(reps),
                         "rms_um": float(np.sqrt(np.mean(np.sum((reps - c) ** 2, axis=1))) * scale),
                         "max_dev_um": float(np.max(np.linalg.norm(reps - c, axis=1)) * scale)})
    return pd.DataFrame(rows)


def plot_tre(subjects, *, save: bool = True):
    import matplotlib.pyplot as plt
    frames = [tre_table(s) for s in subjects]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    if not len(df):
        print("no landmarks yet")
        return None
    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = [n for n in LANDMARK_NAMES if n in set(df["landmark"])]
    for q, s in enumerate(subjects):
        d = df[df["subject"] == s]
        x = [order.index(n) + (q - len(subjects) / 2) * 0.08 for n in d["landmark"]]
        ax.plot(x, d["tre_um"], "o", ms=5, label=s)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    for q, n in enumerate(order):
        if n in NEAR_LEC:
            ax.axvspan(q - 0.4, q + 0.4, color="#7fb8e0", alpha=0.15)
    ax.set_ylabel("target registration error (µm)"); ax.legend(fontsize=8, ncol=3)
    ax.set_title("Landmark TRE per mouse (blue bands = landmarks near the LEC)", fontsize=10)
    fig.tight_layout()
    if save:
        rq.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(rq.FIGURE_DIR / "ALL_mice_tre.png", dpi=130, bbox_inches="tight")
    return fig
