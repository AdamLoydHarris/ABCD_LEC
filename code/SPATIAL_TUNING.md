# Open-field spatial tuning (mEC big field, LEC object box)

Plan: `~/.claude/plans/wiggly-finding-moth.md`, approved 2026-10-09.
Code:
- `spatial_tuning.py`: the engine.
- `spatial_common.py`, `spatial_io.py`: shared code.
- `spatial_tuning_synthetics.py`: the gate.
- `run_spatial.py`: the CLI.
- `sbatch_files/spatial.sbatch` and `sbatch_files/submit_spatial.sh`: cluster launchers.
- `spatial_plots.py`: figures.

Every module is byte-identical in `code/` and `mFC_data/code/`.

## 2026-10-09 — what the engine computes

**Sessions.** Arena sessions of v4 merged recdays: `session_type` 'openfield' or 'object'.
- Spikes and tracking are truncated to the shorter array; both start at rsync[0].
- Positions outside the arena are clipped, and the clipped fraction is reported. head_back reaches
  -6 to 126 cm in the 120 cm field.
- Moving means speed above 2.5 cm/s, from `smooth_and_calculate_scalar_derivatives`.
- HD is `HD_raw[:, 0]` (back2mid). It is used only when it follows the movement direction within
  ±15 deg with R ≥ 0.8 at speeds above 5 cm/s.

**Units.**
- The unit table is keyed by QC cluster id. The analyzer arrays hold every cluster, 361 rows against
  40 QC units in bp01 04-02/03.
- Narrow units have a trough-to-peak width below 0.4 ms; they get their own pools.
- Units with fewer than 100 moving spikes, and the lower-spike member of a likely duplicate pair, are
  scored but kept out of pools and counts.

**Maps.** 2.5 cm bins. Spikes and occupancy are smoothed separately (Gaussian, sigma 2 bins), and a
rate is defined where smoothed occupancy is at least 0.1 s. Object sessions also have a masked map:
samples within the measured object radius + 4 cm are removed and those bins are NaN.

**Scores.**

| score | definition |
|---|---|
| grid | Sargolini gridness on a NaN-aware Pearson autocorrelogram. Inner radius at the first radial minimum; outer radius expanded to half the arena side, maximum taken. Spacing and orientation from 6 peaks. |
| border | Solstad (CM - DM)/(CM + DM). A field is above 0.3 of peak, between 200 cm² and 60 % of the floor. Wall bins are the outermost valid bins. Walls bent by more than 1.5 cm RMS are excluded. |
| HD | 10 deg curve, no smoothing. Classification uses the direction factor of the pxd factorisation (7.5 cm x 30 deg) and its split-half r. |
| spatial | Skaggs information and split-half r of the position factor of a position x direction x speed factorisation. Raw maps called 95 % of pure HD cells and 80 % of speed cells spatial in the gate. |
| speed | r of the speed factor (8 bins, 2.5 cm/s to above 40) of the position x direction x speed factorisation with log speed, two-tailed. Both halves must exceed their own p95 with the same sign. Raw Kropff r (250 ms smoothing, 2-50 cm/s) is reported but not used: it called 100 % of two-wall border cells and 45 % of near-wall place cells speed cells in gate run 3709988. |
| object-proximal | Log rate ratio of the most active of 8 angular sectors of the band (object surface to +20 cm) over the rest of the floor, on speed-matched samples. Compared with at least 100 virtual objects at the same wall distance (±5 cm) whose bands cannot overlap the real one. |
| object-vector | Within one sort: the object-proximal statistic in the object session minus the same statistic at the rotation-aligned location in the open-field session. Null: both sessions circularly shifted. |
| rotation | Best of 8 dihedral transforms of the second session's maps, by median same-unit r over units stable in both sessions. Null: unit-identity shuffle. HD preferred-direction shift with a bootstrap CI and a V-test. |

**Nulls and classes.**
- Null: 1000 circular spike shifts of at least 20 s.
- Pooled p99 over the session's eligible units of the same pool, for gridness, border and |speed r|.
- Each unit's own p99 for information, pxd R and the object statistic; its own p95 for split-half
  stability.
- Grid and border classes also need map stability.
- Chance counts come from applying one shift to every unit at once. Each count is reported against
  that distribution, never against n x 0.01.

## 2026-10-09 — synthetic gate

Quick gate on mEC trajectories, bp01 04-02/03 and 03-24/25:
- 200 shifts, 20 planted units per type, 150 noise units across 3 rate strata.
- Final run 3712023: 92/92 passed, EXIT=0.
- Full gate (1000 shifts, 1000 noise units per stratum), run 3711774: 87/87 passed, EXIT=0. It
  predates check A17, which passed in the quick run.

| check | measured |
|---|---|
| noise FPR, Poisson 0.5 / 2 / 8 Hz, per class | 0-3 of 50 per class and stratum, every binomial p >= 0.014 |
| noise FPR, real maze spike trains on OF paths | 0 of 50 for every class |
| grid 30 / 45 cm detected | 20/20 and 20/20; spacing within 10 % for all; orientation error median 2.2 / 2.4 deg |
| grid detection by spacing | 100 % from 50 to 80 cm, 90 % at 85, 60 % at 90 |
| border, one wall (5 and 10 cm bands) and two walls | 20/20 each; wall label right 100 %; never called grid or speed |
| place fields, centre and near a wall | spatial 100 %; border 0 %; speed 0 % |
| flat 20-80 Hz units scored as broad | border 0 %; without the 60 % area cap 100 % reach the Solstad 0.5 cut, so the artefact is real and the cap removes it |
| pure HD cells on the real trajectory | HD 100 %; border 0 %; spatial 0 % on the position factor, against 100 % on raw maps |
| border cells without HD | pxd HD 0 %; a plain Rayleigh test called 100 % of them HD |
| border x HD conjunctive | border 95 %, HD 100 % |
| speed cells, both slopes | 100 % with the right sign; spatial 0 % |
| drifting untuned units, 0.5-2x | 0 % in every class |
| object session (bottle 22 cm from the top wall) | object field off one side 100 %; place field at the object 100 %; far-wall border, slow-speed cell and nearest-wall border 0 %, though a fixed ratio > 1.5 accepts 100 % of the nearest-wall border cells (median ratio 2.74) |
| rotation rot90 / rot180 / flip_x planted | best transform = the inverse every time, p 0.005; HD shift 89.3 vs 90 and 179.8 vs 180 deg |
| object-vector within a sort (A17) | object-vector cells 5/5; place cells under the object 0/5; border cells 0/5; stable anchor cells 0/20 |
| guards and equivalence | silent unit NaN; sparse unit ineligible; object outside the arena raises; mask 0 and grid-off leave every other statistic identical (max abs diff 0) |

What the gate changed, in order:
- **Spatial class:** moved to the position factor, because raw maps passed 95 % of pure HD cells
  and 80 % of speed cells.
- **Speed class:** moved to the speed factor, because Kropff r passed 100 % of two-wall border cells
  and 45 % of near-wall place cells.
- **Object statistic:** now the most active of 8 sectors, against virtual bands that cannot overlap
  the real one. The ring mean detected 20 % of planted object fields.
- **Speed of the factorisation:** now computed with matrix products, 300x faster, equal to the
  einsum reference to 1e-5.

## Results

### 2026-10-09 — mEC, merged recdays, v4

Status: the full gate passed on 2026-10-09, 87/87, run 3711774.

Run tag `8a49b0e2d93c`: 9 recdays, merged at `Taskspace_abstraction_mEC/data/spatial/of/8a49b0e2d93c/`.
Galleries of every unit are in `figures/gallery_*.v1.pdf`. Counts below are broad eligible units, and
chance is the distribution of counts when one shift is applied to all units, over 1000 shifts.

| session | broad units | grid | border | border and HD | HD (pxd) | spatial (pos. factor) | speed |
|---|---|---|---|---|---|---|---|
| bp01 03-21 open field | 34 | 0 | 2 | 2 | 23 | 31 | 5 |
| bp01 03-25 object | 35 | 0 | 1 | 1 | 21 | 34 | 4 |
| bp01 04-02 open field | 28 | 0 | 7 | 5 | 20 | 20 | 3 |
| bp01 04-03 object | 28 | 0 | 0 | 0 | 13 | 27 | 2 |
| bp04 03-21 open field | 12 | 0 | 2 | 1 | 5 | 8 | 1 |
| bp04 03-25 object | 12 | 0 | 0 | 0 | 8 | 7 | 0 |

- **Chance.** p95 is 0 for grid, border and speed, and 0-1 for spatial and HD. Every non-zero
  count above has p <= 0.011.
- **Grid.**
  - Of 149 broad unit-sessions, 1 exceeds even the pooled p95: bp04 03-25 u11, gridness 0.79,
    spacing 55 cm, 0.08 bits/spike. Chance predicts 7.5, so grid scores sit below shuffled spike
    trains. That unit misses the pre-declared p99 of 0.89.
  - With detection at 100 % up to 80 cm spacing, there is no evidence for grid cells below about
    85 cm spacing in these units.
- **Border.**
  - 11 cells across the three open-field sessions: bp01 03-21 2 on the x = 120 wall; bp01 04-02 7,
    3 on y = 0 and 4 on x = 0; bp04 03-21 2.
  - 8 of 11 are also HD-tuned after pxd.
  - In the object sessions they almost vanish (0-1), consistent with the quick look's finding that
    they fire beside the bottle.
- **HD.** HD tuning survives pxd in 13-23 of 28-35 broad bp01 units per session (median pxd R
  0.18-0.24) and 5-8 of 12 in bp04. Together with few border cells and no grid cells, this suggests
  the probe sits in an HD-rich region: deep MEC layers, or para/presubiculum. Anatomy is not
  established.
- **Narrow units**, own pool: in bp01, 6-9 of 12-14 are HD-tuned and 4-6 speed-tuned per session.
- **bp02:** 0-2 broad units per session; not interpreted.

**Rotation**, bp01 04-02 open field to 04-03 object, one sort:
- Best transform rot180, median same-unit r 0.28. The unit-shuffle null p99 is 0.11, p 0.001, with
  a margin of 0.21 over flip_x; n = 38 units stable in both sessions.
- HD preferred directions shift by 156 deg, 95 % CI 131-180, R 0.75, n 14, V-test against 180 deg
  3.6.
- Maze HD in the same units does not rotate across the two days (quick look: +1 deg, R 0.81). The
  rotation is specific to the visually symmetric big field.

**Objects:**
- Object-proximal: bp01 03-25 16 of 48 eligible units; bp04 03-25 2 of 16.
- The 04-03 bottle sits at the arena centre, where no wall-distance-matched virtual object exists,
  so it is covered by the within-sort test: 23 of 40 units fire more beside the bottle than at its
  rotation-aligned location in the 04-02 open field, against their own shift p99.
- Gate A17 detected 5/5 planted object-vector cells with 0 false positives, at quick-mode n.

### 2026-10-09, later — could tracking explain the low grid scores?

Prompted by bp01 03-21/23 s4 u7: page 5, row 4 of `gallery_bp01_20240321_20240323.v1.pdf`, which
looks like a noisy grid.

**Track quality**, every arena session (11), from the shards' tracks and the arena QC:

| measurement | value |
|---|---|
| walls after the k1/k2 lens model and corner homography, RMS | 0.2-0.9 cm; bp01 03-21 s4 0.51-0.86 cm |
| per-peg calibration error | <= 0.27 cm |
| track missing | 0-0.62 % of 25 ms bins |
| 25 ms steps over 5 cm | 0 in every session |
| 99th-percentile step | 1.4-1.8 cm per 25 ms |
| frame-clock pulses anchored | all, 337-377 per session |
| coverage, full session / each half | 99-100 % / 77-99 % of 2.5 cm bins; bp01 03-21 second half covers 59 % of the central 60 x 60 cm |

**Spike-to-track timing.** `of_lag_scan.py`, job 3712247, output `qc/of_lag_scan_mec.v3.pkl`:
- Method: spikes are paired with behaviour 1 s earlier to 1 s later. A unit counts only if it beats
  its own shift p99 at lag 0.
- Planted lags are recovered. Place-cell groups at -250 / 0 / +250 ms come back at -251 / -2 /
  +260 ms. HD-cell groups at -100 / 0 / +100 ms come back at -124 / -26 / +93 ms through the pxd
  direction factor, a bias of about -25 ms. The 0 ms HD group misses its 25 ms tolerance by 1 ms.
- Untuned units entering either clock: 1 of 24.

| recday | arena session | spatial-info peak, ms | n | HD peak (pxd), ms | n | maze sessions' spatial-info peak, median of session medians |
|---|---|---|---|---|---|---|
| bp01 03-21/23 | open field | +169 | 44 | -53 | 34 | +59 |
| bp01 03-24/25 | object | +92 | 46 | +70 | 27 | +41 |
| bp01 04-02/03 | open field | +165 | 39 | -20 | 25 | +22 |
| bp01 04-02/03 | object | +101 | 39 | -29 | 20 | +22 |
| bp04 03-21/23 | open field | +366 | 13 | -51 | 6 | +124 |
| bp04 03-24/25 | object | +562 | 12 | -25 | 14 | +38 |

- The HD clock, the sharp one, sits within about 50 ms of zero in every bp01 and bp04 arena session,
  after the -25 ms estimator bias. A misaligned arena clock would move it as far as the spatial peak.
- The spatial-information peak sits later in the arena than on the maze. In bp04 its CIs span
  hundreds of ms, for example 44-501 ms. It reflects how firing leads position, not the clock.
- Maze HD peaks, -122 to -409 ms, are not a clock: in corridors heading is tied to the route, and
  route coding pulls them negative.

**Grids with shifted spikes.**
- Method: every arena session was rescored by the engine with spikes shifted by -100, 0, +100,
  +200 and +300 ms, against the run's lag-0 thresholds.
- No broad unit passes the pooled p99 at any lag. At most 1 per session passes the p95.
- In bp01 03-21 s4 the best gridness at any lag is 0.09.
- u7 scores -0.47 / -0.44 / -0.45 / -0.47 / -0.53 across the five lags.

**u7 itself.**
- The autocorrelogram has no peak between 10 and 74 cm. The nearest peaks are at 75, 82, 88 and
  89 cm, outside the 60 cm cap, so its -0.44 comes from the trough between fields.
- The three strongest fields are 80, 90 and 109 cm apart, with angles of 46-79 deg.
- With a 90 cm cap, 6 peaks are found at about 88 cm, but gridness is still -0.29.
- If it is a grid, its spacing is at least 75 cm. The 120 cm field holds about 1.5 periods of that,
  so hexagonal symmetry cannot be tested.

**Annulus cap sensitivity**, the 75 cm run the plan promised plus 90 cm:
- Runs `1cca8860c63f` (75 cm) and `1958df60c63f` (90 cm), with configs in
  `Taskspace_abstraction_mEC/data/spatial/of/config_grid_rmax0{75,90}.json`.
- Every non-grid statistic, real and null, equals run `8a49b0e2d93c` to a max abs diff of 0. So
  the `spatial_io` edit since that run changed nothing in mEC.

| cap | gate: planted grids found at 85 / 90 cm spacing | gate result | real grid calls, broad units |
|---|---|---|---|
| 60 cm | 0.90 / 0.60 | 87/87 full | 0 |
| 75 cm | 1.00 / 0.90 | 92/92 quick | 0 |
| 90 cm | 1.00 / 1.00 | 92/92 quick | 1: bp01 03-21 s4 u21, gridness 1.01 |

- u21 has a single diagonal band of firing. Its elongated central peak pushes the inner ring to
  70 cm, and a 70-90 cm ring at the edge of the autocorrelogram scores high. It is a false positive
  by inspection, and it is not a grid.
- Conclusion: tracking does not explain the absence of grid cells. With the 75 cm cap, which finds
  planted grids up to 85 cm spacing every time, there are still none. Grids with a spacing above
  about 85-90 cm cannot be tested in this arena.

## Not established

- Anatomy: registration is deferred, and bp01's implant side is folded.
- Grid spacings beyond the gate's detection curve, about 90 cm at the 75 cm cap. A 120 cm field
  cannot test hexagonal symmetry at those scales.
- Unit identity across bp01's three sorts. Counts are per recday and never pooled across them.
- The rotation and object-vector results rest on one same-sort pair, bp01 04-02/04-03, and were first
  seen in the 2026-10-08 quick look.

## Commands

    sbatch --job-name=sp_of_syn --partition=cpu_backfill sbatch_files/spatial.sbatch code/spatial_tuning_synthetics.py --dataset mec --quick
    bash sbatch_files/submit_spatial.sh of mec <recday> ...      # prints the merge command
    python code/run_spatial.py --engine of --dataset mec --merge --expect-recdays 9 --run-tag <tag>
    # annulus-cap sensitivity: the config is hashed into the tag
    SPATIAL_CONFIG=<mEC root>/data/spatial/of/config_grid_rmax075.json bash sbatch_files/submit_spatial.sh of mec <recdays>
    sbatch --job-name=sp_of_syn --partition=cpu_backfill sbatch_files/spatial.sbatch code/spatial_tuning_synthetics.py --dataset mec --quick --config <json>
    # spike-to-track lag scan and rescoring with shifted spikes
    sbatch --job-name=of_lagscan --partition=cpu_backfill --cpus-per-task=8 sbatch_files/spatial.sbatch code/of_lag_scan.py --dataset mec
