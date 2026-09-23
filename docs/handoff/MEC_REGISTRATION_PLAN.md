# Plan: register the bp (mEC cohort 7) brains to the Allen CCF — registration only

## Context

The mEC integration is through (data_dic_mec.pkl built 2026-09-18). Anatomy was deferred. The user
now wants to **attempt the histology registration of the bp mice**, expecting bp01 to be poor
("funky as hell"). Scope fixed with the user (2026-09-19): **registration only; the probe track
and per-unit region labels come later.**

**What is on disk (measured 2026-09-19):**

| | BP1 | BP2 | BP3 |
|---|---|---|---|
| mouse | bp01 | bp02 | **bp04** (user; the brainsaw run is `AK30_ML__BP1_BP2__BP4`; the earlier "BP3 = bp03" note is wrong and must be corrected in the memory + docs) |
| acquisition | BakingTray/StitchIt, 313 sections × 2 optical planes = 626 planes at 20 µm, 4.033 µm in-plane, channels 2 (red), 3 (green = autofluorescence), 4 (blue); `stitchedImages_100/{2,3,4}` + `downsampled_stacks/{025,050}_micron` | same | same |
| tissue | **267 mm³, ~30 % missing, grossly asymmetric** (L/R area ratio −0.39…+0.22; e.g. plane 75: left 7.6 vs right 17.3 mm²) | 384 mm³, symmetric | 383 mm³, symmetric; OB tip clipped |
| caudal pole | trimmed before section 1 in all three (plane 0 already 19–29 mm² of cerebellum) | | |
| logs | `acqLog`, `StitchIt_Log`, `scanSettings.mat` are tray-level copies, byte-identical across the three folders (no per-brain record) | | |

Probes: all three NP1.0 (`PRB_1_4_0480_1`), single shank, bank 0 (y 0–3820 µm), **left hemisphere in
all three (user)**, no DiI. The sorter labelled zero channels outside the brain in all 36 blocks, but
the raw AP data say otherwise (R0 below: the top 200–600 µm show fluid-noise signatures in most
sampled recordings). Insertion coordinates are **not to hand** (user). All of this is for the later
track step.

**The LEC pipeline to reuse** (repo `Taskspace_abstraction_lEC`, all verified on disk):
- `code/brainreg_probe/run_brainreg.py`: reads `recipe*.yml` `VoxelSize` (20, 4.033, 4.033), writes and
  `sbatch`s `code/Jobs/brainreg/slurm/brainreg_<mouse>.sh` running
  `module load brainglobe/2024-03-01; brainreg <hist>/stitchedImages_100/3 <out>/allen_mouse_10um
  --additional <hist>/stitchedImages_100/2 -v 20.0 4.033 4.033 --orientation psl --atlas allen_mouse_10um --debug`.
  It hardcodes LEC-relative paths, iterates every subdir of `data/raw_data/histology`, and requests
  `-p gpu --gres gpu:1` with 8 CPUs, 64 GB (niftyreg is CPU-only; ~1 h 15 per brain).
- Parameters that produced the six LEC registrations (`brainreg.json`): niftyreg, affine 6/5 steps,
  freeform 6/4, bending energy 0.95, grid spacing −10 (400 µm control points), histogram 128/128,
  `brain_geometry full`, orientation `psl`, sample space `asr` at 10 µm.
- `code/histology_refit/registration_qc.py` (checks A–G; `global_checks` = affine sanity, convention,
  truncation, L/R asymmetry, hemisphere label, L/R profile + residual angles, section continuity, tile
  artefact, region-volume ratios — **none need a probe fit**; `local_checks` D/E/G/C all read
  `ProbeA_fit_params.json` and are out of scope) with `run_registration_qc.py` and the synthetic gate
  `run_synthetic_controls('ah08')` (PASS 10/10 on 2026-09-11). Paths come from
  `probe_refit.BRAINREG_DIR = REPO_ROOT/data/preprocessed_data/brainreg` and `pr.list_subjects()`;
  figures to `REPO_ROOT/data/figures/registration_qc`. Conventions documented in `REGISTRATION_QC.md`
  (hemisphere label 1 = left = implant side; brainreg's own default is reversed).
- Environments: `histology` conda env (brainreg 1.0.13 with bundled niftyreg, bg-atlasapi 3.0.1,
  tifffile, skimage, ipympl; **no napari**) resolves `~/.brainglobe` which holds only the 25 µm atlas —
  export `BRAINGLOBE_CONFIG_DIR=/ceph/apps/application_data/brainglobe` (has `allen_mouse_10um_v1.2`
  and the v3 `brainglobe-atlasapi/` layout) for anything in that env that touches the 10 µm atlas.
  The module `brainglobe/2024-03-01` sets it itself. `registration_qc.ATLAS_25_DIR` points at the
  local 25 µm copy, which exists.

**Standing rules:** nothing deleted or overwritten (copies, versioned names); LEC defaults and the six
LEC registrations untouched; heavy compute under sbatch; every new check gated by a synthetic or an
equivalence measurement; write-up with numbers and their semantics.

## Approach

### R0. Brain surface from the raw ephys — `code/preprocessing/probe_surface_from_raw.py` (new)
User question (2026-09-19): can the raw data say whether the upper channels were out of the brain?
Yes, directly, and better than the sorter's label: for every task recording (AP stream only; no LFP
stream was recorded) read 20 s from the middle of `continuous.dat`, common-median reference, and per
channel compute the 300–6000 Hz RMS, the −4.5 MAD threshold-crossing rate, the 2–200 Hz RMS and the
correlation with the channel two rows up. Channels in saline or air above the pia show a collapse of
the spike-band RMS and crossing rate together with a jump in low-band neighbour correlation
(conductive medium); the pia is the depth where the profile steps. Output per recording: the
384-channel profile, the estimated surface row (or "bank fully inside"), and the summary table
`data/processed_data/probe_surface_mec.csv`; one figure per mouse (profiles stacked by session).
This is the per-session depth the track step needs and a check on the sorter's
`channel_assignments.json`, which used spikeinterface's `detect_bad_channels(outside="top",
threshold −0.15, n_neighbors 37)` on high-passed data and labelled 0 of 384 channels `out` in all 36
blocks.

**First look (2026-09-19, 20 s windows, 3 recordings per mouse, read-only):** the sorter's label is
wrong. bp01: in 2 of 3 recordings the rows above y ≈ 3200 µm have spike-band RMS 38 vs 105 in the
active mid-bank yet 50–320 threshold crossings/s, low-band neighbour correlation 0.97–0.98 (0.2–0.4 in
tissue) and doubled low-band RMS — correlated fluid noise, not spikes; rows 2400–3200 are quiet
(0.2–25 crossings/s) and bp01 has no QC unit above 2400 µm. bp02 and bp04: the same signature in the
top 200 µm bin in 2 of 3 recordings each (correlation 0.5–0.93, low-band RMS up to 36 vs 17), with
bp02's spiking confined to the tip 800 µm and bp04's to 800–2400 µm. Likely cause of the sorter's
miss: the NP1 AP stream is hardware high-passed at 300 Hz and no LFP stream was saved, so the
coherence the IBL detector relies on had been removed before it ran. The detector here therefore
uses the spike-band 300–1000 Hz neighbour coherence and the crossing-rate/RMS mismatch as well as the
residual low band, and reports the surface row per recording with its confidence (how many of the
three signatures agree and over how many rows). Gate: a synthetic profile with the top 60 rows
replaced by shared scaled noise must yield the planted surface row within 2 rows, and a profile with
all rows in tissue must yield "inside" (no false surface); the three-recording sample above must be
reproduced by the script. Depth for the track step then becomes `surface_y + 175 µm tip offset` per
session (bp01 ≈ 3.4 mm in those two recordings), replacing the "≥ 4.0 mm" bound.
Runs in `maze_ephys` under `mec_stage.sbatch` (~1 min per recording, 107 task recordings).

### R1. Dataset-aware brainreg submitter — `code/histology_refit/run_brainreg_dataset.py` (new)
A thin submitter beside the LEC one (which stays byte-identical), reusing
`run_brainreg.get_voxel_sizes` and `plot_util_func.adjust_contrast`:
- `DATASETS = {'mec': dict(hist_root=MEC/data/raw/histology, subjects={'BP1':'bp01','BP2':'bp02','BP3':'bp04'},
  out_root=MEC/data/preprocessed_data/brainreg, orientation='psl'), 'lec': <the existing layout, for parity runs only>}`.
- `--orientation-check` first: 4 coronal pages of the 25 µm ch3 stack per brain with the (0,0)
  marker, written to `<out_root>/<mouse>/orientation_check.png`, plus the area-along-axis-0 profile
  (posterior → anterior signature already seen: plane 0 ≈ 19–29 mm², OB tail at the high end) so
  `psl` is a measured statement, not an assumption. Submission refuses until the PNG exists.
- `--orientation-check` also writes `damage_survey.png` per brain: 12 coronal pages of the 25 µm
  green stack (every 40 pages) and the per-page left/right tissue area (total and ventral 40 %) for
  BP1 against BP2/BP3, so the missing piece of BP1 is localised (side, AP band) before any compute.
  First numbers: BP1's ventral L/R asymmetry reaches −0.16 at pages 100–120 (2.5–3 mm from the
  posterior end) and +0.28 near page 60, against ≤ 0.09 in BP2/BP3; BP1 is also ~7× dimmer (green
  tissue median 649 vs 4503 / 2712), which NMI registration tolerates but every threshold must be
  per subject (they are).
- `--submit` writes `code/Jobs/brainreg/slurm/brainreg_<mouse>.sh` with the **same brainreg command
  and parameters as LEC** (module `brainglobe/2024-03-01` = brainreg 1.0.6, `allen_mouse_10um`,
  `-v 20.0 4.033 4.033`, `--orientation psl`, `--brain_geometry full`, `--debug`), input
  `stitchedImages_100/3`, `--additional stitchedImages_100/2 stitchedImages_100/4` (red and blue
  autofluorescence as `downsampled_2.tiff` / `downsampled_4.tiff`: extra views for the later lesion
  search, downsampling cost only), output `<out_root>/<mouse>/allen_mouse_10um/` with a
  `subject_map.json` (folder BP1/BP2/BP3 ↔ mouse) beside it. **Memory: the LEC jobs peaked at
  121 GB MaxRSS (`sacct`) on a 64 GB request; the bp stacks have 42 % more voxels (626 planes,
  ~1252 × 740 × 1030 at 10 µm), so `-p cpu --cpus-per-task 16 --mem 220G --time 08:00:00`, on the
  240 GB enc1 or 512 GB enc3 nodes.** brainreg counts the machine's CPUs, so the script passes
  `--n-free-cpus $(( $(nproc --all) - SLURM_CPUS_PER_TASK ))`. Refuses if the output dir exists
  (never overwrite). Three jobs in parallel, expect 1.5–2.5 h each (LEC: 51–92 min for 442 planes).
- BP1: `brain_geometry full` like the others (the loss is a localised piece, not a hemisphere;
  `hemisphere_*` would mask the implant side away; no pre-masking or mirror-filling of the raw
  stack). Optional comparison run into a versioned dir `allen_mouse_10um_be099/` with
  `--bending-energy-weight 0.99` (stiffer warp around the void), judged by the same checks. The
  freeform warp extrapolates into the void, so **labels inside missing tissue are never to be read
  from `registered_atlas.tiff`** (recorded in the write-up for the track step). Usability is judged
  by R3, not assumed.

### R2. QC modules made dataset-aware (LEC behaviour unchanged when nothing is set)
- `probe_refit.py`: `BRAINREG_DIR = Path(os.environ.get('TSA_BRAINREG_DIR', REPO_ROOT/'data'/'preprocessed_data'/'brainreg'))`.
- `registration_qc.py`: `FIGURE_DIR = BRAINREG_DIR.parents[1]/'figures'/'registration_qc'` (identical
  for LEC); region groups for `lr_asymmetry` / `region_volume_ratios` / overlays taken from a
  `QC_REGION_GROUPS` dict keyed by dataset (`lec`: the current ENTl/ENTm/SUB/ProS/CA1/DG; `mec`: add
  PaS, PrS, POST, VISp/VISpm/VISam, RSP, and keep ENTm/ENTl/SUB/ProS/CA1/DG); `truncation_report`
  reports both ends (the bp brains are trimmed caudally, the LEC ones anteriorly).
- New `region_tissue_coverage(subject)`: per atlas region of interest and hemisphere, the fraction of
  its registered voxels that fall on tissue (the module's `tissue_mask` on the green pages) →
  `<BRAINREG_DIR>/region_tissue_coverage.csv`. This is the direct answer to "did bp01's left mEC
  survive" and is cheap (page-wise, 10 µm).
- New `void_report(subject)`: from `lr_profile`, the pages and side where the registered-label
  footprint exceeds the tissue footprint (label asymmetry vs image asymmetry) → the void's AP extent
  in mm and hemisphere, written into the QC row. For BP2/BP3 it must be empty.
- New `global_jacobian(subject)`: det J summary over the whole brain (page-strided), since
  `jacobian_check` is scoped to the track slab.
- `plot_mec_zoom(subject)`: the posterior analogue of `plot_lec_zoom` (coronal levels through ENTm /
  PaS / SUB / DG with atlas contours on the green), both hemispheres side by side.
- `run_registration_qc.py`: `--dataset mec` (sets `TSA_BRAINREG_DIR` before importing), `--global-only`
  (skips `local_checks`, the laminar/Jacobian-slab figures and anything reading `ProbeA_*`), output
  `registration_qc[.vN].csv` (versioned; never overwrite). The synthetic gate stays as is and is run
  once in the env (`--gate`) before the mEC QC, as the equivalence control for the module changes:
  it must still read PASS 10/10 and the LEC `registration_qc.csv` numbers must be reproduced for one
  LEC mouse (ah08) to the last digit by `--dataset lec --global-only ah08` (written to a `.vN` copy).
- A small `sbatch_files/histology_stage.sbatch` (cpu, 32 GB, 8 cpus, `conda activate histology`,
  exports `BRAINGLOBE_CONFIG_DIR`), the `histology`-env twin of `mec_stage.sbatch`.

### R3. Assessment and write-up
For each brain: `orientation_check.png`, brainreg exit 0 and the 14 outputs with shape ≈ 2.5 × the 25 µm
stack (BP1 ≈ 1252 × 735 × 1032 voxels), `volumes.csv`, global checks, `region_tissue_coverage.csv`,
figures (8-level coronal overlays incl. the entorhinal levels, mEC zoom, L/R profile, whole-brain
Jacobian). Human check A: the user looks at the overlays; verdict recorded with `record_verdict`.
Write-up `code/histology_refit/MEC_REGISTRATION.md`: what ran, the per-brain table, the BP1 verdict
(which regions on the implant side have tissue under them), what was not established (no track, no
labels), and the exact commands. Pointer in `docs/handoff/MEC_INTEGRATION.md` (new §9) and the
`mec-cohort7-facts` memory corrected (BP3 = bp04; left hemisphere; brains are trimmed caudally).

### Later (not in this plan, noted for the track step)
No DiI, so the track will have to come from the lesion in the autofluorescence (ch3/ch2 slab MIPs in an
adapted `probe_tool.ProbePlacer`, entry + tip clicks) and/or a stereotaxic prior once the surgery
notes turn up; NP1 single-shank geometry (`RECORDED_BANK_MAX_UM` 3820, tip offset, no shank pitch),
channel ids `AP1..AP384`, and the hard depth bound ≥ 4.0 mm are the facts that step will need.

## Files

| file | change |
|---|---|
| `code/histology_refit/run_brainreg_dataset.py` (new) | R1 submitter with `DATASETS`, `--orientation-check`, `--submit`, `--dataset mec` |
| `code/histology_refit/probe_refit.py` | `BRAINREG_DIR` env override only |
| `code/histology_refit/registration_qc.py` | `FIGURE_DIR` from `BRAINREG_DIR`; `QC_REGION_GROUPS`; two-ended `truncation_report`; `region_tissue_coverage`; `void_report`; `global_jacobian`; `plot_mec_zoom` |
| `code/histology_refit/run_registration_qc.py` | `--dataset`, `--global-only`, versioned CSV |
| `sbatch_files/histology_stage.sbatch` (new) | histology-env runner |
| `code/histology_refit/MEC_REGISTRATION.md` (new), `docs/handoff/MEC_INTEGRATION.md` §9 | write-up and pointer |
| memory `mec-cohort7-facts.md` | BP3 = bp04, left hemisphere, caudal trimming |

## Commands, in order
1. `histology` env: `python run_brainreg_dataset.py --dataset mec --orientation-check` → 3 orientation PNGs + 3 damage surveys; eyeball `psl` and BP1's void.
2. `python run_brainreg_dataset.py --dataset mec --submit` → 3 SLURM jobs (cpu, 16 CPUs, 220 GB; 1.5–2.5 h each, parallel).
3. `sbatch sbatch_files/histology_stage.sbatch histology_refit/run_registration_qc.py --gate` and
   `... --dataset lec --global-only ah08` (equivalence controls, before touching mEC numbers).
4. `sbatch sbatch_files/histology_stage.sbatch histology_refit/run_registration_qc.py --dataset mec --global-only` → CSVs + figures.
5. Write-up, docs pointer, memory fix.

## Verification
- Orientation: the area profile along axis 0 rises from ~20–30 mm² (cerebellum) to a ~40–50 mm² peak
  and falls to a 3–10 mm² olfactory tail; the (0,0) marker sits at superior-left of the coronal page as
  in the LEC `orientation_check.png`s.
- brainreg: exit 0, `brainreg.json` parameters identical to LEC except paths; `registered_atlas.tiff`
  shape matches the sample grid; `volumes.csv` whole-brain volume for BP2/BP3 within ±10 % of each
  other and of the LEC cohort, BP1 ~30 % lower with the deficit on the **left** (implant side) in
  `lr_asymmetry` and `region_tissue_coverage`.
- Global checks (BP2, BP3 expected to pass; BP1 flagged, which is the finding): affine per-axis scale
  in [0.85, 1.20], total rotation < 15°, residual yaw/pitch < 1°, |L/R asymmetry| < 0.15 on the mEC
  region set, det J in [0.5, 2] on ≥ 99 % of brain voxels, `hemisphere_ok` (left = high k = label 1),
  section continuity without jumps > 3 vox.
- Equivalence controls: gate PASS 10/10; `--dataset lec --global-only ah08` reproduces the stored
  ah08 row of `registration_qc.csv` exactly.
- Human: overlays at the entorhinal levels judged by the user (check A) for the three brains.
