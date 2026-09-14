---
name: new-engine-version
description: "How to change a fitting engine in this repo (glm_analysis_v*.py, elasticnet_regression_v*.py, gp_tuning_width.py): copy to a new version, freeze the old one, prove equivalence under legacy flags, mirror to mFC_data/code, document the hunks. Use whenever a request would change what a fit computes -- a new regressor, a new solver, a new selection rule -- rather than how results are plotted."
---

# New engine version

Engines are never edited in place. A change that alters any number a fit produces goes into a new
versioned copy, and the old version stays runnable so the change can be measured.

## 1. Decide whether this is an engine change
- Changes what is fitted, selected, scored or exported -> new version (this skill).
- Changes only plotting, tables or notebook text -> edit `*_plots.py` / the notebook; no new version.
- A new *flag with the old behaviour as default* inside the current version is acceptable only when
  an equivalence control pins the default (see step 4). Prefer it for small solver hooks; use a
  full copy for anything that touches the design matrix or the selection rule.

## 2. Create the copy
- `cp code/<engine>_vN.py code/<engine>_vN+1.py`; bump the `*_VERSION` constant that is stamped
  into results (`GLM_VERSION`, `run_dir_name(... version=)`).
- Copy the synthetics module too (`<engine>_vN_synthetics.py` -> `vN+1`), and the notebook
  (`LEC_<engine>_vN+1.ipynb`, PFC twin) with `RUN_MODE = 'load' | 'slurm' | 'local'`.
- Add the new module and its synthetics to `BYTE_PAIRS` (or `DEF_PAIRS` if it carries the PFC
  loader block) in `code/check_mirror_parity.py`.

## 3. Make the change with the old behaviour reachable
- Every moved default gets a flag whose legacy value reproduces vN exactly. List them in the
  module docstring ("set X=..., Y=... to reproduce vN").
- Config objects must serialise every new field (`to_dict`) so `run_config.json` records them.
- Output directory / section names must change when the numbers change (`run_tag`,
  `section_name`): two configurations sharing a name silently overwrite each other.

## 4. Controls, run through the real pipeline
- Control 1: `vN+1(legacy flags)` == `vN` on a synthetic recday, `max|diff| == 0` over every
  exported array. Pin EVERY flag whose default moved, or the control tests the new defaults.
- One positive control per new feature (a planted cell is recovered) and one negative (the
  thing it must not detect is rejected -- and is accepted when the gate is widened, so the
  rejection is not vacuous).
- Guards: invalid combinations raise at construction, not hours into a fold loop.
- `python code/<engine>_vN+1_synthetics.py --quick` must pass on BOTH trees before any sbatch.

## 5. Mirror, then launch
- `cp` every changed file to `mFC_data/code/`, `python code/check_mirror_parity.py` -> exit 0.
- Launch through the queue (see the `slurm-fit-launch` skill). Never from the interactive shell.

## 6. Document
- `code/<ENGINE>_VN+1.md`: what changed (enumerate the hunks), what did NOT change, the controls
  and their outcomes, runtime per fit, the commands. Mirror pointer in `mFC_data/code/`.
- Add a dated line to `docs/handoff/README.md` under "State of play".
