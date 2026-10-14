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

## 2026-10-12 — round 2, stage A: the final-numbers settings

Plan: `~/.claude/plans/wiggly-finding-moth.md`, round 2, approved 2026-10-12.

- **More null simulations.** Decoders and planes now run 100 simulations per null kind, up from 20 and
  10. RSA stays at 200.
  - Why: with 10 simulations, the planes null mean of PFC ah04 07-12 moved from 45.7 to 27.2 deg
    between equivalent runs. One recday's null fraction moved by up to 0.45.
  - Seeds are drawn per null kind for every index below the RSA count. So the old 20 and 10
    simulations are the first 20 and 10 of the new 100, and every earlier number is unchanged. Gate
    S1 shows this at max abs diff 0.
  - `_check_null_settings` refuses settings that would re-seed later kinds.
- **Claims are per mouse.**
  - For each mouse, the statistic is averaged over its recdays. Its null is simulation i averaged over
    the same recdays; simulations are independent between recdays, so this is a draw from the null of
    the mean.
  - One-sided p = (1 + #null on the far side) / (1 + n).
  - A mouse is a hit when p <= 0.05 and its mean lies on the generalisation side of chance.
  - Across mice: hits out of M, with the binomial tail at 0.05. An exact sign-flip test of the mean z
    is reported when M >= 5.
  - Functions: `maze_spatial.mouse_null_tests` and `group_tests`. Table: `maze_spatial_report.mouse_claims`.
  - The earlier mean of per-recday null fractions is kept as `legacy_claim_counts`.
  - Gate checks MS1 and MS2 cover calibration and power. 1000 null mice give 61 hits, binomial p 0.11.
    A +1 SD effect in each of 5 recdays is detected in 98 % of mice by the new rule and 0 % by the
    legacy rule.
- **Modern nodes only.**
  - Haswell and Broadwell nodes (`gpu-350`, `gpu-380`) differ at 1e-15, which flipped BIC field counts
    in run 1617859bc6fd.
  - `submit_spatial.sh` excludes them (`SPATIAL_EXCLUDE`), and `run_spatial.py` refuses them unless
    `--allow-legacy-host` is passed.
  - `run_spatial.py` also refuses to write a shard if the engine code changed during the job; loky
    workers import the engine from disk.
- **Provenance.** Shards record `slurm_job_id` and `null_timing` (seconds per analysis). Merges add a
  long `null_sims` table: recday, mouse, null, analysis, sim, stat, value.

## 2026-10-12 — round 2, stage B: is node/bridge coding behaviour at nodes?

Nodes are where mice slow, turn and poke ports, and every node has a port. Speed was already matched.
Stage B adds turning and the at-port state, as options inside engine v1. The defaults leave every earlier
number unchanged, and the new null kind is seeded after the existing ones.

**Turning**, `spatial_common.angular_speed_dps`:
1. The input angle is the travel direction from XY, NaN at or below 2.5 cm/s. This is identical in all
   three datasets; PFC has no HD.
2. Single-bin spikes are dropped: a sample more than 90 deg from both neighbours, where the neighbours
   agree within 45 deg.
3. The angle is unwrapped within tracked segments. A gap over 4 bins (100 ms) splits a segment; shorter
   gaps are interpolated.
4. Gaussian smoothing, sigma 2 bins (50 ms), then the gradient. Values above 2000 deg/s become NaN.
5. Only |w| is kept: the maze frames are mirrored relative to the arena, and PFC's handedness is unknown.

Head turning, from HD in sessions passing the HD guard, is the mEC/LEC sensitivity run.

**At-port state**, `maze_spatial.port_state`: on a node, Locs 1-9, below 2.5 cm/s for at least 0.3 s.
Stops at the same node merge across gaps of up to 0.25 s. It is derived the same way in all three
datasets.

**Kinematic null**, kind `kinematic`, in the default RSA and decode kinds:
- Each unit's mean count is computed per cell of 12 speed bins (5 cm/s) x 6 |turn| bins (0, 22.5, 45,
  90, 180, 360 deg/s, plus undefined) x 2 port states.
- A cell with under 5 s of dwell backs off to (speed, port), then speed, then the unit's mean.
- Spikes are Poisson-simulated on the real track, as for the other nulls.

**Turn matching**, option `speed_match.turn_edges_dps`, off by default:
- Each location type gets the same speed x |turn| histogram. Samples without a turn rate are dropped.
- Configs: `data/spatial/maze/config_turn_matched.json` (B2, travel turning) and
  `config_turn_matched_head.json` (B2-head, mEC and LEC).
- Both run RSA and decoders only, with kinds speed, place_shuffled and kinematic.

**Gate K checks.** Final runs 3755059-61 passed 57/57 on each dataset's gate trajectory.

| check | mEC | LEC | PFC | requirement |
|---|---|---|---|---|
| K1 median travel |turn| above 10 cm/s, corner / cardinal / bridge, deg/s | 53 / 29 / 29 | 61 / 50 / 35 | 69 / 46 / 37 | corner >= 1.5x bridge |
| K1 corner / bridge ratio | 1.81 | 1.71 | 1.85 | |
| K1 despiked / capped samples | 0 / 0 | 0 / 0 | 0.0031 / 0.0026 | < 0.005 each |
| K2 goal arrivals with an at-port episode within 1 s | 0.908 of 412 | 0.986 of 776 | 0.984 of 1072 | >= 0.9 |
| K3 collapsed turn edges equal speed-only matching | exact, 18476 samples | exact, 65900 | exact, 38872 | exact |
| K4 turn, port and slow cells, p against the kinematic null: decode BA / RSA beta | 0.381 / 0.619 | 0.333 / 0.762 | 0.810 / 0.238 | > 0.05 |
| K4 the same cells against the speed null (reported): decode / RSA | 0.048 / 0.238 | 0.048 / 0.048 | 0.048 / 0.048 | shows what the kinematic null adds |
| K5 node and bridge template cells, p against the kinematic null: decode / RSA | 0.048 / 0.048 | 0.048 / 0.048 | 0.048 / 0.048 | <= 0.05 |
| K5 real vs kinematic null p95: decode BA / RSA beta | 0.949 vs 0.527 / 0.897 vs 0.188 | 0.959 vs 0.584 / 0.896 vs 0.341 | 0.962 vs 0.536 / 0.885 vs 0.254 | |
| K6 turn cells, corner/cardinal BA: speed-matched -> speed x turn matched | 0.59 -> 0.43 | untestable: turn-only AUC 0.531 | 0.57 -> 0.49 | drop >= 0.05 where the turn-only AUC is >= 0.58 or <= 0.42 |
| K7 guards | raise | raise | raise; 'head' raises | raise |

- **K1.** It first asked for 2x, a number written before any data, and failed mEC and PFC (runs 3752832, 3752834). It now asks for 1.5x.
  - Head turning is high everywhere in mEC: corner 103 deg/s, bridge 66. Head scanning on bridges is why travel turning is the primary signal.
- **K5.** Its decode term had 10 simulations, so p could not go below 1/11 = 0.091. It now runs 20 in the quick gate and 40 in the full; the p values above are the floor of 20.
- **K4.** Against the speed null, the turn, port and slow cells beat every simulation in 5 of 6 tests. The speed null would pass them; the kinematic null holds them in.
- **K6, LEC.** On the LEC gate trajectory, turning barely separates corners from cardinals, so turn matching has nothing to remove there.

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

### 2026-10-12 — round 2, stage A: run `b705159d8440`, the final maze numbers

**The run.**
- 61/61 jobs exited 0, all on modern nodes (`gpu-sr670`, `gpu-sr675`, `gpu-xd670`).
- One config hash, `b70515e621`, and one code hash, `9d84400a3a`, in all three tables.
- Gates 3751698-700 passed 48/48 on the mEC, LEC and PFC trajectories.
- Runtime at 32 CPUs: most recdays 36-45 min; the largest (ah04, ah08, ah10) over 1 h.
- Report: `python code/maze_spatial_report.py --tag b705159d8440`.

**Equivalence.** `maze_spatial_report.shard_equivalence`, against each recday's modern-node reference
run:

| dataset | reference | recdays | values compared | max abs diff | NaN mismatches | missing |
|---|---|---|---|---|---|---|
| mEC | 1617859bc6fd | 12 | 39.8 M | 0 | 0 | 0 |
| LEC | 1617859bc6fd | 25 | 96.9 M | 0 | 0 | 0 |
| PFC | 1617859bc6fd (ab03 x3) | 3 | 10.8 M | 0 | 0 | 0 |
| PFC | 21c552de331e | 21 | 59.9 M | 0 | 0 | 0 |

- Every number of the earlier runs is unchanged; the old simulations are the first 20 / 10 of the new 100.
- Once off the Haswell/Broadwell nodes, PFC reproduces its earlier modern-node run bit for bit. This
  confirms that the 1617859bc6fd differences were hardware.

**Mouse-level claims.**
- Each count is mice with p <= 0.05 on the generalisation side of chance. The per-mouse null is the
  mean over the mouse's recdays of simulation i.
- p_binomial is the probability of at least that many hits if each mouse had a 5 % false-positive
  rate.
- Field-shuffle null, the bar:

| statistic | mEC, 2 mice | LEC, 5 mice | PFC, 7 mice | legacy count, same null: mEC / LEC / PFC |
|---|---|---|---|---|
| RSA node-degree beta, E+D, 9 nodes | 0 | 5, p_binomial 3e-7 | 5: ab03, ah04, ah07, me10, me11; p 6e-6 | 0 / 3 / 2 |
| node/bridge balanced accuracy | 0 (BA 0.45-0.48) | 4: ah08, ly05, ly06, ly07; p 3e-5 | 7; p 8e-10 | 0 / 4 / 7 |
| RSA node/bridge beta, E+NB, 21 locations | 2: bp01, bp04; p 0.0025 | 5; p 3e-7 | 7; p 8e-10 | not in the legacy table |
| corner/cardinal balanced accuracy | 1: bp04, BA 0.50, at chance | 0 (BA 0.20-0.31, below chance) | 4: ah04, ah07, me08, me11; p 2e-4 | 1 / 0 / 3 |
| same-type share of location-decoder errors | 0 | 0 (share 0.24-0.32) | 4: ah04, ah07, me08, me11; p 2e-4 | 0 / 0 / 1 |
| E/G diagnostic contrast, 21 locations | 0 | 5; p 3e-7 | 3: ab03, ah04, ah07; p 0.004 | 0 / 2 / 3 |

- **More power.** The mouse-level test finds more than the legacy mean of per-recday fractions because
  averaging over recdays shrinks the null of the mean. Gate MS2 measured this: 98 % against 0 % for a
  +1 SD per-recday effect. Its false-positive rate stays at 5 % (MS1).
- **Fitted and random-centre nulls**, which sit beside the bar in the report:
  - PFC node degree: 2/7 and 4/7.
  - LEC node degree: 4/5 and 4/5.
  - Node/bridge is unchanged across all three nulls.
- **Not yet established.** Node/bridge coding stands against every place-code null but not yet against
  behaviour at nodes. Nodes are where mice slow, turn and poke. Stage B adds the kinematic null and turn
  matching.

### 2026-10-12, later — round 2, stage B: node/bridge coding against behaviour at nodes

Runs, 159 jobs, all exit 0, all on modern nodes (`gpu-sr670`, `gpu-sr675`, `gpu-xd670`), 1 min to 2 h:

| run | tag | config hash | recdays |
|---|---|---|---|
| B1, defaults with the kinematic null | `53934257354e` | `53934241d3` | mEC 12, LEC 25, PFC 24 |
| B2, speed x travel-turn matched | `05fd7f57354e` | `05fd7f412e` | mEC 12, LEC 25, PFC 24 |
| B2-head, speed x head-turn matched | `09482657354e` | `0948262c95` | mEC 12, LEC 25 |

All three share code hash `57354ee2f3`. Reports: `maze_spatial_report.py --tag <tag>`; B2-head with
`--datasets mec lec`.

**B1 changes no earlier number.** Against stage A (`b705159d8440`), shard by shard:

| dataset | recdays | values compared | max abs diff | NaN mismatches | missing |
|---|---|---|---|---|---|
| mEC | 12 | 40.5 M | 0 | 0 | 0 |
| LEC | 25 | 98.4 M | 0 | 0 | 0 |
| PFC | 24 | 85.4 M | 0 | 0 | 0 |

The only additions are the kinematic null's rows: 324 in mEC, 675 in LEC, 648 in PFC.

**Mouse-level claims.** Each cell is the number of mice with p <= 0.05 on the generalisation side, against
the mean-over-recdays null. The kinematic null simulates each unit from its own speed x |turn| x at-port
rates.

| statistic | dataset | field shuffle | kinematic, B1 | kinematic, B2 travel-turn matched | kinematic, B2-head |
|---|---|---|---|---|---|
| node/bridge decoding BA | PFC | 7/7 | 7/7 | 7/7 | |
| | LEC | 4/5 | 3/5: ah08, ly05, ly07 | 4/5: ah08, ah10, ly05, ly07 | 4/5: ah08, ly05, ly06, ly07 |
| | mEC | 0/2 | 0/2 | 0/2 | 0/2 |
| RSA node/bridge beta, E+NB, 21 locations | PFC | 7/7 | 5/7: ah03, ah04, me08, me10, me11 | 5/7 | |
| | LEC | 5/5 | 2/5: ah08, ly07 | 2/5 | 2/5 |
| | mEC | 2/2 | 0/2 | 0/2 | 0/2 |
| RSA node-degree beta, E+D, 9 nodes | PFC | 5/7 | 6/7 | 6/7 | |
| | LEC | 5/5 | 5/5 | 5/5 | 4/5 |
| corner/cardinal BA | PFC | 4/7 | 4/7 | 4/7 | |
| | LEC | 0/5, BA 0.20-0.31 | 0/5 | 0/5 | 0/5 |
| same-type share of decoder errors | PFC | 4/7 | 4/7 | 3/7 | |
| E/G diagnostic contrast | PFC | 3/7 | 3/7 | 2/7 | |
| | LEC | 5/5 | 4/5 | 4/5 | 3/5 |

- Field-shuffle counts are from B1. B2's field-shuffle counts differ from them by at most one mouse,
  e.g. LEC node/bridge BA 5/5.
- The speed null alone gives counts between the field shuffle and the kinematic null. For example,
  LEC RSA node/bridge is 2/5 against speed and 2/5 against the kinematic null.

What this shows:
- **PFC.** Node/bridge decoding is beyond the kinematic null in 7/7 mice, with and without turn matching.
  The RSA node/bridge beta holds in 5/7, node degree in 6/7, and corner/cardinal decoding in 4/7. Speed,
  turning and pausing at ports do not explain PFC's node/bridge or node-type coding.
- **LEC.** Node/bridge decoding holds in 3-4 of 5 mice; the RSA node/bridge beta falls from 5/5 to 2/5.
  Part of LEC's node/bridge structure is behaviour at nodes. The speed-matched decoder keeps most of it.
  Node degree holds in 4-5/5.
- **mEC.** Its one stage A claim, RSA node/bridge 2/2, falls to 0/2. With 2 informative mice, mEC maze
  structure beyond behaviour is not shown.

**The at-port state over all production recdays** (2026-10-13, from the B1 shards). The gate's K2 used one
recday per dataset; across recdays, the share of goal arrivals with an at-port episode within 1 s is:

| dataset | recdays | range | median | below 0.9 |
|---|---|---|---|---|
| LEC | 25 | 0.964-0.998 | 0.988 | 0 |
| mEC | 12 | 0.882-0.975 | 0.902 | 6 |
| PFC | 24 | 0.777-0.984 | 0.913 | 11 |

On some PFC and mEC recdays the mice reach the goal without a 0.3 s stop below 2.5 cm/s, so the port term
misses up to 22 % of reward visits. A kinematic null that misses reward stops is weaker, which favours hits.
The PFC node/bridge result would need a reward-visit term to rule this out.

**Figures** (2026-10-13), `data/spatial/figures/round2/`, PDF + PNG, from `code/spatial_round2_figures.py`:

| file | shows |
|---|---|
| `B1_maze_claims_{mec,lec,pfc}` | per mouse, the real mean of each of the six claims against its field-shuffle, kinematic and speed nulls; * marks mouse p <= 0.05 |
| `B2_maze_scorecard` | mice beyond each null, per claim and dataset, for B1, B2 and B2-head |
| `B3_maze_turning_and_ports` | median travel and head turning by location type per recday, and the at-port coverage above |

- The stage A figure is in `maze_spatial.executed.v1.ipynb`: real value against the field-shuffle null per mouse.
- Per-unit maze galleries are in `<dataset root>/data/spatial/maze/21c552de331e/figures/`. Later runs leave
  every unit's maps and classes unchanged, so they stand for runs A and B.

## Not established, maze

- Whether node/bridge decoding is place type or behaviour at nodes beyond speed, turning and port
  stops: decisions and reward expectation are not modelled. Stage B rules out the three measured
  behaviours for PFC, and in part for LEC.
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

2026-10-12, round 2. Every job excludes the Haswell/Broadwell nodes, which `submit_spatial.sh` does by default.

    # gates, one per dataset (PFC from mFC_data/code)
    sbatch --job-name=sp_mz_syn_mec --partition=cpu_backfill --exclude='gpu-350-[01-05],gpu-380-[10-18]' --cpus-per-task=32 --mem=96G sbatch_files/spatial.sbatch code/maze_spatial_synthetics.py --dataset mec --quick
    # B1 (defaults, with the kinematic null), B2 (speed x travel-turn matched), B2-head (mEC, LEC)
    SBATCH_EXTRA="--cpus-per-task=32 --mem=96G --time=1-00:00" bash sbatch_files/submit_spatial.sh maze mec <recdays>
    SPATIAL_CONFIG=$PWD/data/spatial/maze/config_turn_matched.json SBATCH_EXTRA=... bash sbatch_files/submit_spatial.sh maze mec <recdays>
    SPATIAL_CONFIG=$PWD/data/spatial/maze/config_turn_matched_head.json SBATCH_EXTRA=... bash sbatch_files/submit_spatial.sh maze mec <recdays>
    python code/maze_spatial_report.py --tag 53934257354e --equivalent-to b705159d8440
    python code/maze_spatial_report.py --tag 05fd7f57354e
    python code/maze_spatial_report.py --tag 09482657354e --datasets mec lec
