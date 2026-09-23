# mEC cohort-7 brains: Allen registration (registration only)

**Started 2026-09-19.** Scope fixed with Adam: register BP1/BP2/BP3 to `allen_mouse_10um` with the
LEC recipe and judge each registration; the probe track and per-unit labels come later. Plan:
`docs/handoff/MEC_REGISTRATION_PLAN.md`. Dated paragraphs below; nothing is rewritten.

## 1. What is on disk (2026-09-19)

| | BP1 | BP2 | BP3 |
|---|---|---|---|
| mouse | bp01 | bp02 | **bp04** (Adam; brainsaw run `AK30_ML__BP1_BP2__BP4`) |
| acquisition | 313 sections x 2 optical planes = 626 planes at 20 um, 4.033 um in-plane, channels 2 red / 3 green / 4 blue | same | same |
| tissue volume, 25 um green stack, 15 % of p99.5 | 244.5 mm^3 | 374.1 | 374.7 |
| peak coronal area | 38.0 mm^2 (page 192) | 47.7 | 48.5 |
| caudal pole | trimmed before section 1 in all three (page 0 already 12-29 mm^2 of cerebellum) | | |
| implant | left hemisphere, NP1.0, no DiI (Adam) | left | left |

`orientation_check.png` per brain (`data/preprocessed_data/brainreg/<mouse>/`, mEC tree) and
`damage_survey.png` (`.../brainreg/`): axis 0 runs posterior -> anterior (cerebellum first, 3-10 mm^2
olfactory tail last), rows superior -> inferior, so the orientation string is `psl` as for the LEC
rig. **BP1's loss is not confined to mEC**: the whole-page L/R asymmetry (L-R)/(L+R) is -0.3 to -0.4
from page 100 to the anterior end, i.e. the LEFT hemisphere -- the implant side -- is about a
third smaller than the right along the whole forebrain; the ventral 40 % shows the same. BP2 and
BP3 sit within +-0.1 except at the olfactory tail (BP2's bulb is partly missing on one side, irrelevant
here). BP1 is also ~7x dimmer in every channel (green tissue median 649 vs 4503 / 2712).

## 2. Brain surface from the raw ephys (step R0, `code/preprocessing/probe_surface_from_raw.py`)

Question from Adam: can the raw data say whether upper channels were out of the brain? The sorter's
`channel_assignments.json` said no channel was (`detect_bad_channels(outside="top", -0.15, 37)` on
the high-passed recording, 0 of 384 `out` in all 36 blocks). Read directly from `continuous.dat`
(20 s at mid-recording, common-median referenced), the top of the bank shows correlated
low-frequency noise (neighbour correlation 0.9-1.0 at 40 um vs 0.2-0.6 in tissue), the same in the
300-1000 Hz band, and threshold crossings that are common-mode across the half-probe rather than
local -- channels in fluid or dura above the pia. Detector: a row is fluid when both correlations
are >= 0.6 and its crossings are common-mode (or absent); the surface is the lowest row with a
5-row solid fluid base above which >= 80 % of rows are fluid. Gate (planted 60 fluid rows on a
real bp02 window): recovered row 132 exactly; the unmodified bank read inside. **PASS.**
Results per recording: `data/processed_data/probe_surface_mec[.vN].csv` (one file per mouse job),
per-row profiles in `probe_surface/<mouse>/`, figure `probe_surface_<mouse>.png`.

**2b. Scan of every task recording (20 s per recording).** A first pass (jobs 3645515-7) counted
crossings shared by >= 6 channels as common-mode, which bp04's large mid-bank units also are
(coincident fraction 0.8-0.9 in its dense band), so it needed all three signatures at the 0.6
correlation level and flagged only 3 of 26 bp04 recordings although `probe_surface_bp04.png`
shows the same fluid band in all 26. Second pass (jobs 3646552-4, 2026-09-20 11:20): common-mode =
shared by >= 16 channels (8 rows, above any spike footprint), fluid = spike-band correlation >= 0.8
and low-band >= 0.5 and common-mode (or no) crossings; the surface needs a 5-row solid fluid base,
>= 60 % fluid rows above it and no run of > 2 tissue-like rows above it (bp04's topmost 2-4 rows
lose the correlation again while staying common-mode -- above the fluid level, in air or dura).
Gate PASS on bp01, bp02 and bp04 windows (planted row 132 recovered exactly, unmodified bp02 inside).

| mouse | recordings | with a surface | surface y from the tip-most row (median / range) | rows out of 192 | tip depth below the surface (y + 175 um) |
|---|---|---|---|---|---|
| bp01 | 37 | **34** | 3340 um (3260-3500) | 25 (500 um) | **~3.5 mm** |
| bp02 | 45 | 0 | - | 0 | >= 4.0 mm (whole bank inside) |
| bp04 | 26 | **26** | 3540 um (3540-3560) | 15 (300 um) | **~3.7 mm** |

bp01's and bp04's boundaries sit at the same row in every recording over three weeks (no drift).
Above them the spike-band neighbour correlation is 0.96 / 0.71 (bp01 / bp04) against 0.55 / 0.67 in
the mid-bank, and 98 % / 87 % of crossings are common-mode against 6 % / 36 %. The three bp01
recordings without a surface (03-23 14:38, 03-25 13:36, 04-03 14:39) carry the same top-row
signature (spike-band correlation 0.54-0.69) just under the 0.8 rule. bp02's top rows look like
tissue in every recording (correlations 0.2-0.3, no common-mode crossings). For the track step this
replaces the "whole bank inside" reading of `channel_assignments.json`. Per-recording rows:
`probe_surface_mec.v3-5.csv`; the first pass is `.csv`, `.v1`, `.v2`.

## 3. Registration (step R1, `code/histology_refit/run_brainreg_dataset.py`)

Same brainreg command and parameters as the LEC runs (`module brainglobe/2024-03-01`,
`allen_mouse_10um`, `-v 20 4.033 4.033 --orientation psl --brain_geometry full --debug`), registration
channel `stitchedImages_100/3`, red and blue carried as `--additional`, outputs under
`Taskspace_abstraction_mEC/data/preprocessed_data/brainreg/{bp01,bp02,bp04}/allen_mouse_10um/`
(`subject_map.json` records folder <-> mouse). Cluster: `-p cpu`, 16 CPUs, 220 GB (the LEC jobs
peaked at 121 GB MaxRSS on 64 GB requests; these stacks have 42 % more voxels), jobs 3645512
(bp01), 3645513 (bp02), 3645514 (bp04), submitted 2026-09-19 evening. LEC submitter untouched.

**3b. Runs (2026-09-20 01:50-05:40, enc3 nodes): all three completed, exit 0, no errors.**

| mouse | wall time | MaxRSS | sample volume at 10 um (pages x rows x cols) |
|---|---|---|---|
| bp01 | 1 h 27 | 129.5 GB | 1252 x 708 x 995 |
| bp02 | 1 h 41 | 109.4 GB | 1252 x 689 x 1032 |
| bp04 | 1 h 32 | 118.2 GB | 1252 x 696 x 988 |

`brainreg.json` differs from the LEC ah08 run only in the paths and `n_free_cpus` (8 vs 4); every
registration parameter is identical. Outputs: the 14 standard files plus `downsampled_4.tiff`
(blue). A 64 GB request, the LEC default, would have been killed on every brain.

## 4. QC (step R2, `registration_qc.py`, `run_registration_qc.py --dataset mec --global-only`)

Dataset switch: `TSA_BRAINREG_DIR` / `TSA_HIST_DATASET` (set by the driver); with nothing set every
LEC path, group and number is as before. Without a probe fit the checks that need one (D, E, G,
slab Jacobian, LEC zoom) are skipped and the others run on a pseudo-bank at the ENTm centroid on
the implant side. New for this dataset: `region_tissue_coverage` (fraction of each registered
region's voxels that lie on tissue, per hemisphere), `void_report` (pages where the label footprint
exceeds the tissue footprint), `global_jacobian` (det J over the whole brain), `plot_mec_zoom`, and
the para/pre/postsubiculum groups in the asymmetry and volume tables. Equivalence controls: the
synthetic gate (job 3645524) and `--dataset lec --global-only ah08` against the stored ah08 row
(job 3645525).

**4b. Equivalence controls (2026-09-19 22:20): both pass.** Gate `SYNTHETIC GATE: PASS` (10/10 as
on 2026-09-11). The ah08 global-only row reproduces the stored `registration_qc.csv` row on all
**49 common columns with 0 differences**; the run adds only the new columns (`cov_*`, `gdetj_*`),
written to `registration_qc_global.csv` beside the untouched original.

**4c. mEC results (`run_registration_qc.py --dataset mec --global-only`, job 3646551, 2026-09-20
11:06; table `registration_qc_global.v1.csv` in the mEC brainreg folder, figures in
`Taskspace_abstraction_mEC/data/figures/registration_qc/`).** LEC thresholds throughout.

| | bp01 | bp02 | bp04 | gate |
|---|---|---|---|---|
| affine scale AP / DV / LR (sample mm -> atlas mm) | 1.062 / 1.132 / 1.122 | 0.971 / 1.146 / 1.063 | 0.996 / 1.137 / 1.089 | [0.85, 1.20] |
| affine rotation total | 6.5 deg | 7.6 | 7.7 | < 15 |
| residual yaw / pitch after registration | -0.12 / +0.72 deg | -0.22 / +0.88 | 0.00 / +0.23 | yaw < 1 |
| affine convention, hemisphere label (1 = left = high k) | ok, ok | ok, ok | ok, ok | |
| truncation | overhang; 0.3 mm anterior, 1.1 mm posterior missing | overhang; 1.2 / 0.9 | overhang; caudal tissue beyond the atlas, 1.4 mm posterior | overhang |
| L/R asymmetry, worst judged group | **0.25 (POST); flags ENTl, CA1, DG, POST** | 0.12 (PAR); none | 0.16 (POST); flag POST | < 0.15 |
| image-vs-label asymmetry agreement (r) | 0.94 | 0.98 | 0.98 | |
| volume ratio sample/atlas: ENTm, CA1 (expected 1/det A) | 0.72, 0.72 (0.74) | 0.78, 0.86 (0.85) | 0.78, 0.81 (0.81) | |
| det J whole brain: median, 5-95 %, outside [0.5, 2] | 1.37, 0.70-2.30, **12.5 %** | 1.19, 0.89-1.93, 4.2 % | 1.25, 0.87-2.06, 5.6 % | <= 1 % is the LEC slab gate |
| **folded voxels (det J < 0)** | **172,101** | 0 | 0 | 0 |
| region tissue coverage (labels on tissue), all groups, both sides | 0.997-1.000 | 0.999-1.000 | 0.991-1.000 | |

Reading. **bp02 is a clean registration**: no flag, whole-brain det J in range on 96 % of voxels
and no fold, residual angles within a quarter degree of the LEC brains, and the medial entorhinal
zoom (`bp02_mec_zoom.png`) shows the ENTm, para/pre/postsubicular, subicular and hippocampal
contours on the corresponding tissue. **bp04 is nearly as clean**: one asymmetry flag just over the
line (POST 0.16), no fold, 5.6 % of det J outside range; its zoom (`bp04_mec_zoom.png`) matches the
tissue, and on the implant side at coronal i = 1062 a dark vertical streak descends through the
para/postsubicular labels towards ENTm -- a candidate for the probe lesion when the track step
comes. **bp01 registered, but the left hemisphere is not to be believed**: the warp had to
squeeze the atlas into a hemisphere a third smaller, and did so with 172k folded voxels and 12.5 %
of the brain outside the plausible det J range (0 and 4-6 % in the other two), the volume ratios
28 % below the atlas, and asymmetry flags on ENTl, CA1, DG and POST. The coverage metric is
uninformative here and is recorded as such: a registration maps the atlas onto whatever tissue
exists, so labels never fall on the void; the damage shows as distortion (folds, det J, asymmetry),
not as labels on nothing. `bp01_mec_zoom.png` shows the left medial-entorhinal contours drawn over
a smooth, structureless mass with the hippocampal fibre contours over a gap, and a detached piece
of tissue above it. Where the folds sit (job 3646569, `registration_qc_global.v2.csv`,
`gdetj_neg_*`): all 172k folded voxels lie within sample AP 7.7-8.8 mm (56 pages, the
hippocampal / posterior-thalamic level), and 158k of them carry the hemisphere label 2 = right
against 14k on the left -- the warp that squeezes the atlas into the missing left tissue turns
inside out across the midline, so the damage distorts the labelled right hemisphere at that level
as well. bp01's right-hemisphere labels are therefore only plausible outside AP 7.7-8.8 mm.

## 5. Verdicts (2026-09-20)
- bp02 (BP2): registration usable for a track fit on either side. Human check A still to be
  recorded (`record_verdict`).
- bp04 (BP3): usable; note the POST asymmetry and the candidate lesion streak at i ~ 1062.
- bp01 (BP1): right hemisphere plausibly usable; left (implant) hemisphere labels are not
  anatomically meaningful where the tissue is gone -- any bp01 track will have to rest on the
  stereotaxic prior and the ephys depth (3.5 mm to the tip), with the registration used at most to
  locate the remaining surface. A stiffer-warp comparison run (`--bending-energy-weight 0.99`) was
  not made: it cannot create the missing tissue.

## Not established
No probe track, no channel or unit labels, no region assignment. Insertion coordinates are not on
disk. The LEC `probe_refit` machinery fits the DiI cloud and gives nothing without dye. Check F
(landmark TRE) not clicked for any bp brain.

## Commands
```
histology env, code/histology_refit:
  python run_brainreg_dataset.py --dataset mec --orientation-check
  python run_brainreg_dataset.py --dataset mec --submit                     # 3 sbatch, 220 GB, ~1.5 h each
  sbatch ../../sbatch_files/histology_stage.sbatch run_registration_qc.py --gate
  sbatch ../../sbatch_files/histology_stage.sbatch run_registration_qc.py --dataset lec --global-only ah08
  sbatch ../../sbatch_files/histology_stage.sbatch run_registration_qc.py --dataset mec --global-only
maze_ephys env:
  sbatch sbatch_files/mec_stage.sbatch probe_surface_from_raw.py --dataset mec --gate
  sbatch sbatch_files/mec_stage.sbatch probe_surface_from_raw.py --dataset mec --mouse bp01 --write   (and bp02, bp04)
```
