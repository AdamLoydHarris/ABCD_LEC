# Spatial structure on the maze: one pipeline for mEC, LEC and PFC

Plan: `~/.claude/plans/wiggly-finding-moth.md`, approved 2026-10-09. User rule: the same pipeline and
config run on all three datasets. Per-dataset code lives only in `spatial_io.py`.

Code:
- `maze_spatial.py`: the engine.
- `maze_spatial_synthetics.py`: the gate.
- `run_spatial.py --engine maze`: the CLI.
- `spatial_plots.py`: figures.

All are byte-identical in `code/` and `mFC_data/code/`.

## 2026-10-09 — what the pipeline computes

**Assembly.** Task sessions with at least 5 trials, on the 25 ms grid. At 10 trials, LEC keeps 136 of 191 sessions and loses the ly05 06-18/19 recday; at 5 it keeps 164 of 191, and PFC keeps 263 of 270.
- Goal-progress bins: 5, from `compute_task_state_arrays`.
- Within-trial mask: from the first trial's A to the last trial's end.
- Speed: from `smooth_and_calculate_scalar_derivatives`, in cm/s. PFC pixels are converted with
  `pfc_xy_to_cm.v2`.
- Canonical frame: per recday, an affine maps the 9 node centroids onto a fixed lattice. Nodes sit at
  5 / 24 / 43 cm, the 19 cm spacing measured in every dataset (18.8-19.3). Fit residuals are reported.

**Single units.**
- 21-location maps in Hz per task.
- Cross-task stability: the mean r over pairs of different tasks.
  - Null: behaviour circularly shifted within each session.
  - Other-unit baseline.
- Skaggs information.
- Generalised types (corner, cardinal, centre, node, bridge): template r in every task, against a
  label-permutation null.
- 2-D maps at 1 cm in the canonical frame:
  - the user's normalised-convolution smoother, reimplemented and equal to the notebook to max abs
    diff 0;
  - an occupancy-weighted variant;
  - each at sigma 1.5 and 3.

**Population, decided recipe.**
- Samples are speed-matched across location types (corner, cardinal, centre, bridge).
- RSA:
  - Per-unit z-scored, cross-validated RDMs across disjoint task halves, per goal-progress bin.
  - Model fits: E+D and G+D, plus E+NB and G+NB on 21 locations, compared by R² cross-validated across
    goal-progress bins.
  - Unique variance, and a diagnostic contrast on the pairs where E and G disagree.
  - Null: node-label permutation.
- SVM decoders:
  - RBF, C 1, balanced classes, z-scored 500 ms features.
  - Held-out corner vs cardinal, held-out node vs bridge, and held-out-task location with error
    structure.
  - Null: label permutations.
- Lasso location planes:
  - 20 lags, alpha 0.005, 250 ms bins.
  - PCA with k = 2 and k = 3.
  - Smallest and largest principal angles.
  - Disjoint-halves and disjoint-task within-bin controls.

**Thesis recipe.** An option run on every dataset, as the notebooks computed it:
- RSA: per-session raw-rate RDMs over every bin, an unstandardised E/P/D fit.
- Decoders: 500 ms mean counts, the first bin's location, C 1.5, no scaling, speed cuts converted to
  6 / 2 cm/s.
- Planes: overlapping halves [A1, B1, ...], PCA(2), the largest angle.
- Not reproduced: the stale speed mask of `pfc_decoding` cell 36.

**Nulls, simulated per recday.**
- Place code: 1-3 Gaussian fields fitted by Poisson ML to each unit's 2-D map.
- The same fields with random centres.
- Field shuffle, added 2026-10-12. Each field's centre is drawn from the recday's pooled fitted centres.
  Each unit keeps its field count, widths and heights, and the population keeps where its fields are.
  What breaks is the co-occurrence of one unit's fields at locations of one type.
- Speed tuning only.
- Distance-to-edge tuning only.

**2026-10-12 — the bar for generalisation claims is the field-shuffle null** (user's choice).
- **Why:**
  - The fitted null refits a 4-corner cell as up to 3 corner fields, so it partly reproduces
    generalisation. Planted generalised cells cleared it by only 0.04-0.22 in degree beta, and one
    LEC draw failed, 0.81 against 0.88.
  - The random-centre null removes only the field-geometry artefact.
- **How it was added:** as a new null kind inside engine v1, under the repo rule of 2026-09-14 for
  new options. The v1 null kinds are seeded first and in their old order, so every number of run
  `21c552de331e` is unchanged; the equivalence check is under Results.
- **Gate:**
  - M2: pure place cells must sit inside the field-shuffle null.
  - M3: planted generalised cells must clear its p95.

Every population statistic is recomputed on the simulations: RSA 200, decoders 20, planes 10 per
family.

## Why the place-code null is needed (2026-10-09 measurements)

| measurement | value |
|---|---|
| pure Gaussian place cells through the thesis RSA recipe | degree beta 0.37-0.41, t 4-16 over 25 simulated recdays |
| thesis PFC degree effect | t 3.51 |
| held-out corner vs cardinal BA for the same simulated place cells | 0.09-0.32, below chance |
| model RDM correlation: Euclidean vs geodesic | 0.95 |
| model RDM correlation: place type vs degree | 0.88 |

## Exclusions (same rule for every dataset)

A recday enters only if its canonical frame can be fitted, which needs node centroids from cm
tracking. Speed matching, 2-D maps and the place-code null all need it.

- 2026-10-09: PFC `me08_12092021_13092021` is excluded from both recipes. All 7 sessions fail
  `pfc_xy_to_cm.v2` (`xy_ok` False, px/cm 1.13), so it has 0 node centroids (job 3712045).
- PFC is therefore merged over 24 recdays. The thesis targets used 25.

## Synthetic gate

### 2026-10-09 — results and the four checks the gate changed

Trajectories: mEC bp01 04-02/03, LEC ah10 06-16/17, PFC me11 05-12/06-12. Values are from quick
runs 3712240-2, with M5 from 3712251. mEC also has a full run, 3712172.

| check | mEC | LEC | PFC |
|---|---|---|---|
| ME1 smoother equals the notebook | max abs diff 0 | 0 | 0 |
| ME2 thesis RSA equals the inline pfc_rsa copy | 5.5e-07 | 2.4e-07 | 2.5e-07 |
| M1 bridge bias, expected map, normconv / occupancy-weighted | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| M1 zero-fill smoother on bridges, sigma 1.5 / 3, must fail | 0.636 / 0.438 | 0.575 / 0.392 | 0.840 / 0.605 |
| M2 place cells, degree beta inside the fitted null, sigma 6 / 15 | 0.15 / 0.75 of null >= real | 0.25 / 0.70 | 0.70 / 0.70 |
| M2 place cells, corner/cardinal BA, sigma 6 / 15 | 0.16 / 0.00 | 0.16 / 0.01 | 0.08 / 0.00 |
| M2 pooled same-type error share, sigma 6 | 0.28 of 32 errors | 0.30 of 33 | 0.03 of 30 |
| M3 generalised cells recovered | 10/10 per type | 10/10 | 10/10 |
| M3 degree beta against the fitted-null p95 | 0.64 vs 0.49 | 0.68 vs 0.64 | 0.77 vs 0.55 |
| M3 degree beta against the field-shuffle p95, runs 3749896-8 | 0.64 vs -0.00 | 0.68 vs 0.06 | 0.77 vs 0.01 |
| M2 place cells inside the field-shuffle null, sigma 6 / 15, fraction of null >= real | 0.20 / 0.40 | 0.20 / 0.50 | 0.35 / 0.40 |
| M3 corner/cardinal BA | 0.73 | 0.76 | 0.89 |
| M4 speed-only AUC; unmatched → matched BA | 0.596; 0.53 → 0.44 | 0.517; untestable | 0.614; 0.56 → 0.44 |
| M5 cv R2 G minus E, E-coded / G-coded population | -0.045 / -0.011 | rerun pending | rerun pending |
| M5 diagnostic contrast, E-coded / G-coded | 0.05 / 0.44 | | |
| M6 smallest plane angle, across vs halves / tasks | 76.0 vs 6.3 / 7.4 | | 72.0 vs 4.9 / 7.1 |
| M7 noise units called stable | 2/60 | | |

What the gate changed, each with the measurement that forced it:
- **M1**: bias is now tested on the expected spike map. One Poisson draw gave a median of 0.942 on
  the PFC track at sigma 1.5, run 3712174. That was skew of noisy low-occupancy bins, not bias: both
  smoothers are linear in the counts with weights set by occupancy, and give 1.0000 on the expected
  map. One-draw medians are still reported: 0.942-0.982.
- **M2**: same-type errors are now pooled over field layouts until there are at least 30. One layout
  on LEC gave 8 of 9 errors same-type, run 3712173, all from one pair of nodes without fields.
- **M4**:
  - The test is now the drop in accuracy under speed matching. Label permutations keep each node's
    speed, so the permutation p95 reached the unmatched BA, 0.53, in run 3712172.
  - It only applies where speed separates corners from cardinals. On the LEC track it does not,
    speed-only AUC 0.517.
- **M5**:
  - The sign of the cv R2 difference is biased towards Euclidean. It picked E for 5/5 E-coded
    populations but G for only 1/5 G-coded ones, run 3712251; PFC run 3712242 also missed G.
  - A narrow-field code saturates with distance the way E compresses G.
  - The gate now requires the R2 difference and the diagnostic contrast to be higher for the
    G-coded population. The sign is reported only. Real-data E-vs-G claims rest on the contrast
    against the nulls.

Open:
- M3's margin over the fitted null is thin: 0.04-0.22, and one LEC draw failed, 0.81 vs 0.88,
  run 3712173. See the null choice under Not established.
- Rerun of all three gates with the M4/M5 changes: jobs 3712378-80, 37/37 on every trajectory.
- 2026-10-12, with the field-shuffle checks: runs 3749896-8, 40/40 on every trajectory.

## Results

### 2026-10-09 — run `21c552de331e`, all three datasets

**The run.**
- One config hash, `21c552ad25`, and one code hash, `de331e7873`, in all three merged tables.
- Recdays: PFC 24, mEC 12, LEC 25. Units: PFC 1229, mEC 251, LEC 2851.
- Every job exited 0.
- Merged tables: `<root>/data/spatial/maze/21c552de331e/merged/`.
- Report: `python code/maze_spatial_report.py --tag 21c552de331e`.
- mEC bp02 has 17 units over 5 recdays, 0-4 per recday. It is run but not interpreted, so mEC has 2
  informative mice.

**PFC thesis reproduction**, thesis recipe:
- Statistics are one-sample two-sided t-tests across recdays, as in the thesis, and across mice.
- The thesis used 25 recdays; me08 12-09/13-09 is excluded here (see Exclusions).

| statistic | chance | thesis t, 25 recdays | ours t, 24 recdays | p | ours t, 7 mice | p |
|---|---|---|---|---|---|---|
| RSA Euclidean beta | 0 | 5.45 | 5.11 | < 0.0001 | 3.06 | 0.022 |
| RSA node-degree beta | 0 | 3.51 | 3.75 | 0.0011 | 2.16 | 0.074 |
| corner/cardinal balanced accuracy | 0.5 | 3.36 | 3.07 | 0.0054 | 1.32 | 0.23 |
| node/bridge balanced accuracy | 0.5 | 4.63 | 5.35 | < 0.0001 | 2.64 | 0.039 |
| same-type share of location-decoder errors | 3/7 | 4.32 | 4.56 | 0.0001 | 2.24 | 0.066 |

Plane angles, thesis recipe, by goal-progress bin distance, mean ± SEM over 24 recdays:

| distance | 0, within-bin halves | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| angle, deg | 8.6 ± 0.7 | 51.8 ± 3.6 | 63.0 ± 4.2 | 65.3 ± 4.1 | 64.6 ± 4.7 |

The thesis reported 8-10 deg within and 59-69 deg across. All six targets reproduce. With mice as
the replicates, only the Euclidean RSA and node/bridge decoding are below p 0.05.

**Decided recipe.**
- Each count is the number of mice whose mean statistic is beyond the null p95 and on the
  generalisation side of chance. "Beyond the null p95" means at most 5 % of simulations reach the
  real value, averaged over the mouse's recdays.
- "Fitted" is the place-code null with 1-3 fields per unit at its own map peaks. "Random" is the
  same fields at random centres.
- mEC counts are out of 2 mice, LEC out of 5, PFC out of 7.

| statistic | mEC range | mEC fitted / random | LEC range | LEC fitted / random | PFC range | PFC fitted / random |
|---|---|---|---|---|---|---|
| RSA node-degree beta, E+D, 9 nodes | 0.07-0.16 | 0 / 0 | 0.34-0.64 | 4 / 2 | 0.14-0.74 | 0 / 3 |
| node/bridge balanced accuracy | 0.45-0.48 | 0 / 0 | 0.45-0.59 | 4 / 4 | 0.53-0.81 | 6 / 5 |
| corner/cardinal balanced accuracy | 0.40-0.50 | 1 / 0 | 0.20-0.31 | 0 / 0 | 0.41-0.74 | 0 / 4 |
| same-type error share | 0.43-0.47 | 0 / 0 | 0.24-0.32 | 0 / 0 | 0.41-0.65 | 2 / 0 |
| E/G diagnostic contrast, 21 locations | -0.31-0.03 | 0 / 0 | 0.13-0.54 | 0 / 2 | -1.16-0.70 | 0 / 3 |

- **Mice beyond each null:**
  - LEC degree, fitted null: ah10, ly05, ly06, ly07.
  - PFC degree, random null: ah04, ah07, me11.
  - Node/bridge: LEC ah08, ly05, ly06, ly07; PFC fitted ab03, ah03, ah04, ah07, me08, me11.
  - PFC corner/cardinal, random null: ah04, ah07, me08, me11.
- **Below chance can still beat a null.** Place codes decode held-out locations *below* chance
  because a held-out location's neighbours are of the other type. So LEC corner/cardinal accuracy
  of 0.20-0.31 is a place-code signature, and the null, not 0.5, is the reference.
- **Plane angles**, k = 3, decided recipe:
  - Across goal-progress bins the angle exceeds the disjoint random-halves control in every mouse:
    mEC 2/2, LEC 5/5, PFC 7/7, on the largest angle.
  - It does not exceed the disjoint-task control: largest angle mEC 0/2, LEC 0/5, PFC 2/7;
    smallest angle 0/2, 0/5, 1/7.
  - Subspaces of one goal-progress bin estimated from different tasks differ as much as subspaces
    of different bins.
- **Single units:**

| dataset | units | stable across tasks | corner | cardinal | centre | node | bridge |
|---|---|---|---|---|---|---|---|
| mEC bp01 / bp04 | 180 / 54 | 180 / 53 | 12 / 1 | 0 / 4 | 11 / 5 | 29 / 4 | 2 / 0 |
| LEC, 5 mice | 2851 | 2541 | 127 | 3 | 203 | 79 | 85 |
| PFC, 7 mice | 1229 | 1094 | 80 | 23 | 129 | 258 | 90 |

  Centre, node and bridge templates are not speed-matched at the single-unit level. A "centre" cell
  is a single-location cell, and node/bridge cells can reflect slowing at nodes.

### 2026-10-12 — run `1617859bc6fd`, with the field-shuffle null as the bar

**The run.**
- One config hash, `16178599bf`, and one code hash, `9bc6fd6ab4`, in all three tables.
- 61/61 jobs exited 0. Gates 3749896-8 passed 40/40 on every trajectory.
- Report: `python code/maze_spatial_report.py --tag 1617859bc6fd --equivalent-to 21c552de331e`.

**Equivalence with run `21c552de331e`**, compared over every number the old run has:
- **mEC and LEC:** max abs diff 0 in units, population and nulls; NaN patterns equal. New null
  rows only: 468 and 975. Seeding the v1 null kinds first left every v1 number unchanged.
- **PFC:** 6 of 24 recdays are bitwise identical. The other 18 differ:
  - Quantities computed before any null differ at 1e-15 to 1e-17: Skaggs information, cross-task
    2-D map r, type-score thresholds. The code change cannot reach them.
  - All 18 ran on `gpu-350` / `gpu-380` nodes, Intel Haswell and Broadwell. Every mEC and LEC job
    ran on `sr675` / `xd670`.
  - The noise flips the BIC field count between 2 and 3 for 19 of 1229 units. That changes the
    fitted and random-centre null populations.
  - For one recday, "fraction of null >= real" moved by up to 0.20 under the fitted null and 0.45
    under the random-centre null. The planes null mean of ah04 07-12 moved from 45.7 to 27.2 deg,
    with 10 simulations. Speed and boundary nulls did not move on any claim statistic.
  - Per-mouse counts under the fitted and random-centre nulls are identical between the two runs
    for every statistic.
- Consequences:
  - The pipeline is not bitwise reproducible across CPU families. Equivalence checks must run on one
    node family.
  - At 10-20 simulations, per-recday null fractions carry large Monte Carlo variability, so claims
    rest on per-mouse aggregates.

**Mice beyond the null p95 and on the generalisation side of chance, decided recipe.** Mice listed
are those beyond the field-shuffle null, the bar.

| statistic | mEC, 2 mice: shuffle / fitted / random | LEC, 5: shuffle / fitted / random | PFC, 7: shuffle / fitted / random |
|---|---|---|---|
| RSA node-degree beta | 0 / 0 / 0 | 3 / 4 / 2: ah10, ly05, ly06 | 2 / 0 / 3: ah07, me11 |
| node/bridge balanced accuracy | 0 / 0 / 0 | 4 / 4 / 4: ah08, ly05, ly06, ly07 | 7 / 6 / 5: all |
| corner/cardinal balanced accuracy | 1 / 1 / 0: bp04, BA 0.503, at chance | 0 / 0 / 0 | 3 / 0 / 4: ah04, ah07, me11 |
| same-type error share | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 2 / 0 |
| E/G diagnostic contrast, 21 locations | 0 / 0 / 0 | 2 / 0 / 2: ah08, ah10 | 3 / 0 / 3: ab03, ah04, ah07 |

Against the bar:
- **mEC:** no generalisation.
- **LEC:** node degree in 3/5 mice and node/bridge in 4/5. Corner/cardinal accuracy stays below
  chance in every mouse, 0.20-0.31, as a place code predicts.
- **PFC:** node/bridge in 7/7 mice and corner/cardinal in 3/7. Node degree reaches only 2/7. The
  same-type error share of the thesis does not clear the bar in any mouse.
- The geodesic-side contrast clears it in LEC 2/5 and PFC 3/7.

## Not established, maze

- Whether node/bridge decoding beyond both nulls is place type or behaviour at nodes, such as
  pauses, turns and decisions. Speed is matched; other behaviour is not.
- Euclidean against geodesic coding. The two model RDMs correlate at 0.95, and the gate's E vs G
  check needed 5 population draws to be reliable (see Synthetic gate).
- Power of the fitted place-code null: on the LEC trajectory, planted generalised cells did not clear
  it (gate M3, degree beta 0.81 against a null p95 of 0.88). Resolved on 2026-10-12: the field-shuffle
  null is the bar, and planted generalised cells clear it by 0.62-0.76 on every trajectory.
- Bitwise reproducibility across CPU families: PFC recdays rerun on Haswell/Broadwell nodes differ at
  1e-15, enough to flip BIC field counts and redraw the place-code nulls (run 1617859bc6fd).

## Not established

- Generalisation claims that fail the place-code null.
- Anatomy of the mEC units.
- LEC units dorsal to LEC are included unless split by `anatomy_split`.

## Commands

    sbatch --job-name=sp_mz_syn --partition=cpu_backfill sbatch_files/spatial.sbatch code/maze_spatial_synthetics.py --dataset mec --quick
    bash sbatch_files/submit_spatial.sh maze pfc <recdays>     # runs from mFC_data/code
    bash sbatch_files/submit_spatial.sh maze mec <recdays>
    bash sbatch_files/submit_spatial.sh maze lec <recdays>
    python code/run_spatial.py --engine maze --dataset mec --merge --expect-recdays 12 --run-tag 21c552de331e   # lec 25; pfc 24 from mFC_data/code
    python code/maze_spatial_report.py --tag 21c552de331e
    sbatch --job-name=sp_maze_gallery --partition=cpu_backfill --cpus-per-task=2 sbatch_files/spatial.sbatch code/spatial_galleries.py --engine maze --tag 21c552de331e --dataset mec lec pfc
