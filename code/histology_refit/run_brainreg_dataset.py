"""
Dataset-aware brainreg submitter (2026-09-19): the mEC cohort-7 brains with the LEC recipe.

`code/brainreg_probe/run_brainreg.py` registered the six LEC brains and stays byte-identical. This
script reuses its recipe reader and contrast helper, and runs the SAME brainreg command and
parameters (module brainglobe/2024-03-01, allen_mouse_10um, -v 20 4.033 4.033, orientation psl,
brain_geometry full, --debug) on another tree, with two changes that are about the cluster, not the
registration: the CPU partition (niftyreg has no GPU path; the LEC jobs asked for a GPU for nothing)
and the memory (the LEC jobs peaked at 121 GB MaxRSS on a 64 GB request per `sacct`; these stacks
have 42 % more voxels).

Steps
  --orientation-check   per brain: 6 coronal pages of the 25 um green stack with the (0,0) marker and
                        the tissue-area profile along axis 0 (posterior -> anterior under `psl`), plus
                        a damage survey (left/right area per page, total and ventral 40 %) for every
                        brain of the dataset on one figure. Nothing is submitted until these exist.
  --submit              write code/Jobs/brainreg/slurm/brainreg_<mouse>.sh and sbatch it; refuses if
                        the output directory exists (nothing is overwritten) or the orientation PNG is
                        missing. --dry-run writes the script only.

Usage (histology env, any cwd):
  python run_brainreg_dataset.py --dataset mec --orientation-check
  python run_brainreg_dataset.py --dataset mec --submit [--dry-run] [--subjects bp01 bp02]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "brainreg_probe"))
# run_brainreg.py (the LEC submitter) creates Jobs/ dirs and lists LEC subjects at import time
# relative to the cwd, so it is not imported; its two constants and the recipe reader are copied.
from brainreg_probe import plot_util_func as puf                       # noqa: E402

LEC_ROOT = CODE.parent


class rb:                                    # the LEC submitter's values, verbatim
    ATLAS_NAME = "allen_mouse_10um"
    REGISTRATION_CHANNEL = 3

    @staticmethod
    def get_voxel_sizes(recipe_path) -> dict:
        import yaml
        with open(str(recipe_path)) as stream:
            return yaml.safe_load(stream)["VoxelSize"]
MEC_ROOT = Path("/ceph/behrens/adam_harris/Taskspace_abstraction_mEC")
JOBS = CODE / "Jobs" / "brainreg"

#: One entry per dataset. `subjects` maps the histology folder to the mouse name used everywhere
#: downstream (brainreg outputs are stored under the MOUSE name; subject_map.json records the pair).
#: mEC identities: user, 2026-09-19 -- the brainsaw run is AK30_ML__BP1_BP2__BP4, so BP3 is bp04.
DATASETS = {
    "mec": dict(hist_root=MEC_ROOT / "data" / "raw" / "histology",
                subjects={"BP1": "bp01", "BP2": "bp02", "BP3": "bp04"},
                out_root=MEC_ROOT / "data" / "preprocessed_data" / "brainreg",
                orientation="psl", implant_hemisphere="left"),
    "lec": dict(hist_root=LEC_ROOT / "data" / "raw_data" / "histology",
                subjects={m: m for m in ("ah08", "ah09", "ah10", "ly05", "ly06", "ly07")},
                out_root=LEC_ROOT / "data" / "preprocessed_data" / "brainreg",
                orientation="psl", implant_hemisphere="left"),
}
ATLAS = rb.ATLAS_NAME                      # allen_mouse_10um
REG_CH, EXTRA_CH = rb.REGISTRATION_CHANNEL, (2, 4)      # green registers; red + blue carried along
MODULE = "brainglobe/2024-03-01"           # the module that produced the LEC registrations
SLURM = dict(partition="cpu", cpus=16, mem="220G", time="08:00:00")
UM_PER_PX_25 = 25.0


def ds_paths(ds: str, folder: str) -> dict:
    cfg = DATASETS[ds]
    mouse = cfg["subjects"][folder]
    hist = cfg["hist_root"] / folder
    return dict(mouse=mouse, hist=hist, stitched=hist / "stitchedImages_100",
                recipe=sorted(hist.glob("recipe*"))[0],
                ds25=sorted((hist / "downsampled_stacks" / "025_micron").glob(f"*ch0{REG_CH}*.tif"))[0],
                out=cfg["out_root"] / mouse, out_atlas=cfg["out_root"] / mouse / ATLAS)


# --------------------------------------------------------------------------------------
# orientation check + damage survey
# --------------------------------------------------------------------------------------

def tissue_areas(stack: np.ndarray, rel: float = 0.15) -> dict:
    """Per-page tissue area (mm^2) at 25 um: total, left, right, and ventral-40 % left/right.
    Threshold = `rel` x the stack's 99.5th percentile (per subject, so a 7x dimmer brain gets its own).
    Under `psl`, page columns run left -> right and rows superior -> inferior."""
    thr = rel * np.percentile(stack, 99.5)
    px_mm2 = (UM_PER_PX_25 / 1000.0) ** 2
    n, h, w = stack.shape
    out = {k: np.zeros(n) for k in ("total", "left", "right", "left_ventral", "right_ventral")}
    for i in range(n):
        m = stack[i] > thr
        if not m.any():
            continue
        rows = np.where(m.any(axis=1))[0]
        v0 = rows[0] + int(0.6 * (rows[-1] - rows[0]))          # ventral 40 % of the tissue's DV extent
        out["total"][i] = m.sum() * px_mm2
        out["left"][i] = m[:, : w // 2].sum() * px_mm2
        out["right"][i] = m[:, w // 2:].sum() * px_mm2
        out["left_ventral"][i] = m[v0:, : w // 2].sum() * px_mm2
        out["right_ventral"][i] = m[v0:, w // 2:].sum() * px_mm2
    return out


def orientation_check(ds: str, subjects: list[str] | None = None) -> None:
    import matplotlib.pyplot as plt
    import tifffile
    cfg = DATASETS[ds]
    folders = subjects or list(cfg["subjects"])
    areas = {}
    for folder in folders:
        p = ds_paths(ds, folder)
        stack = tifffile.imread(p["ds25"])
        a = tissue_areas(stack)
        areas[p["mouse"]] = a
        n = stack.shape[0]
        pages = [int(v) for v in np.linspace(0, n - 1, 6)]
        fig, axes = plt.subplots(2, 6, figsize=(20, 7))
        for ax, i in zip(axes[0], pages):
            ax.imshow(puf.adjust_contrast(stack[i]), cmap="gray")
            ax.scatter(0, 0, color="red", s=80, zorder=3)
            ax.set_title(f"{p['mouse']} ({folder}) page {i}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        ax = axes[1, 0]
        ax.plot(a["total"], "k", label="total"); ax.plot(a["left"], "C0", label="left (low col)"); ax.plot(a["right"], "C3", label="right")
        ax.set_xlabel("page (axis 0; posterior -> anterior under psl)"); ax.set_ylabel("tissue mm$^2$"); ax.legend(fontsize=7)
        ax.set_title("area profile: cerebellum first, olfactory tail last", fontsize=9)
        ax = axes[1, 1]
        asym = (a["left"] - a["right"]) / np.maximum(a["left"] + a["right"], 1e-9)
        asym_v = (a["left_ventral"] - a["right_ventral"]) / np.maximum(a["left_ventral"] + a["right_ventral"], 1e-9)
        ax.plot(asym, "k", label="(L-R)/(L+R) all"); ax.plot(asym_v, "C1", label="ventral 40 %"); ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylim(-0.6, 0.6); ax.set_xlabel("page"); ax.legend(fontsize=7); ax.set_title("L/R asymmetry (negative = left smaller)", fontsize=9)
        for ax in axes[1, 2:]:
            ax.axis("off")
        txt = (f"{p['mouse']} <- {folder}\nstack {stack.shape} at 25 um\nrecipe {p['recipe'].name}\n"
               f"tissue volume {a['total'].sum() * 0.025:.1f} mm^3 (page area x 25 um)\n"
               f"page 0 area {a['total'][0]:.1f} mm^2; peak {a['total'].max():.1f} at page {int(a['total'].argmax())}\n"
               f"last page with > 1 mm^2: {int(np.where(a['total'] > 1)[0][-1])}\n"
               f"ventral asymmetry min {asym_v.min():+.2f} at page {int(np.nanargmin(asym_v))}, max {asym_v.max():+.2f} at page {int(np.nanargmax(asym_v))}\n"
               f"orientation assumed: {cfg['orientation']}; implant hemisphere: {cfg['implant_hemisphere']}")
        axes[1, 2].text(0, 1, txt, va="top", family="monospace", fontsize=8, transform=axes[1, 2].transAxes)
        p["out"].mkdir(parents=True, exist_ok=True)
        out = p["out"] / "orientation_check.png"
        if out.exists():
            out = out.with_name(f"orientation_check.{datetime.now():%Y%m%d_%H%M%S}.png")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print(f"{p['mouse']}: {txt.splitlines()[3]}; {txt.splitlines()[6]} -> {out}")
    # damage survey across the dataset
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for mouse, a in areas.items():
        axes[0].plot(a["total"], label=mouse)
        axes[1].plot((a["left"] - a["right"]) / np.maximum(a["left"] + a["right"], 1e-9), label=mouse)
        axes[2].plot((a["left_ventral"] - a["right_ventral"]) / np.maximum(a["left_ventral"] + a["right_ventral"], 1e-9), label=mouse)
    axes[0].set_title("tissue area per page (mm^2)"); axes[1].set_title("(L-R)/(L+R), whole page"); axes[2].set_title("(L-R)/(L+R), ventral 40 %")
    for ax in axes[1:]:
        ax.axhline(0, color="grey", lw=0.5); ax.set_ylim(-0.6, 0.6)
    for ax in axes:
        ax.set_xlabel("page (25 um stack; posterior -> anterior)"); ax.legend(fontsize=8)
    out = cfg["out_root"] / "damage_survey.png"
    if out.exists():
        out = out.with_name(f"damage_survey.{datetime.now():%Y%m%d_%H%M%S}.png")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"damage survey -> {out}")


# --------------------------------------------------------------------------------------
# submission
# --------------------------------------------------------------------------------------

def slurm_script(ds: str, folder: str) -> Path:
    cfg = DATASETS[ds]
    p = ds_paths(ds, folder)
    vox = rb.get_voxel_sizes(p["recipe"])
    extra = " ".join(str(p["stitched"] / str(c)) for c in EXTRA_CH if (p["stitched"] / str(c)).is_dir())
    for d in ("slurm", "out", "err"):
        (JOBS / d).mkdir(parents=True, exist_ok=True)
    script = f"""#!/bin/bash
#SBATCH --job-name=brainreg_{p['mouse']}
#SBATCH --output={JOBS}/out/brainreg_{p['mouse']}.out
#SBATCH --error={JOBS}/err/brainreg_{p['mouse']}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={SLURM['cpus']}
#SBATCH -p {SLURM['partition']}
#SBATCH --mem={SLURM['mem']}
#SBATCH --time={SLURM['time']}

# {ds}: {folder} -> {p['mouse']}; same brainreg parameters as the LEC runs (run_brainreg.py), CPU
# partition because niftyreg has no GPU path, 220 GB because the LEC jobs peaked at 121 GB MaxRSS.
echo "node $SLURMD_NODENAME, start $(date)"
source /etc/profile.d/modules.sh
module load {MODULE}
echo "brainreg: $(which brainreg); config dir $BRAINGLOBE_CONFIG_DIR"
NFREE=$(( $(nproc --all) - {SLURM['cpus']} )); [ "$NFREE" -lt 0 ] && NFREE=0
echo "n_free_cpus=$NFREE"

brainreg {p['stitched'] / str(REG_CH)} {p['out_atlas']} --additional {extra} \\
  -v {vox['Z']} {vox['Y']} {vox['X']} --orientation {cfg['orientation']} --atlas {ATLAS} \\
  --brain_geometry full --n-free-cpus $NFREE --debug
rc=$?
echo "All done! $(date) EXIT=$rc"
exit $rc
"""
    path = JOBS / "slurm" / f"brainreg_{p['mouse']}.sh"
    if path.exists() and path.read_text() != script:
        path = path.with_name(f"brainreg_{p['mouse']}.{datetime.now():%Y%m%d_%H%M%S}.sh")
    path.write_text(script)
    return path


def submit(ds: str, subjects: list[str] | None = None, dry_run: bool = False) -> None:
    cfg = DATASETS[ds]
    folders = subjects or list(cfg["subjects"])
    cfg["out_root"].mkdir(parents=True, exist_ok=True)
    smap = cfg["out_root"] / "subject_map.json"
    existing = json.loads(smap.read_text()) if smap.exists() else {}
    for folder in folders:
        p = ds_paths(ds, folder)
        if p["out_atlas"].exists():
            print(f"{p['mouse']}: {p['out_atlas']} exists -- not resubmitting (nothing is overwritten)")
            continue
        if not any(p["out"].glob("orientation_check*.png")):
            print(f"{p['mouse']}: no orientation_check.png yet -- run --orientation-check first")
            continue
        existing[p["mouse"]] = dict(folder=folder, hist=str(p["hist"]), recipe=p["recipe"].name,
                                    implant_hemisphere=cfg["implant_hemisphere"], orientation=cfg["orientation"],
                                    written=datetime.now().isoformat(timespec="seconds"))
        script = slurm_script(ds, folder)
        print(f"{p['mouse']}: script {script}")
        if not dry_run:
            print(subprocess.run(["sbatch", str(script)], capture_output=True, text=True).stdout.strip())
    smap.write_text(json.dumps(existing, indent=1))
    print(f"subject map -> {smap}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="mec", choices=sorted(DATASETS))
    ap.add_argument("--subjects", nargs="*", default=None, help="histology folder names, e.g. BP1 BP2")
    ap.add_argument("--orientation-check", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.orientation_check:
        orientation_check(args.dataset, args.subjects)
    if args.submit:
        submit(args.dataset, args.subjects, dry_run=args.dry_run)
    if not (args.orientation_check or args.submit):
        ap.print_help()


if __name__ == "__main__":
    main()
