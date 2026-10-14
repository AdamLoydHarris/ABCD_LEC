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

## 2026-10-12 — round 2, stage C: one mEC unit's maze tuning against its open-field tuning

Module `code/maze_of_link.py`, mEC only and not mirrored. Run with
`run_spatial.py --engine link --dataset mec --recday <recday>`. Inputs are the open-field shard of run
`8a49b0e2d93c`, the maze shard of run `b705159d8440`, and the recday's views; maze HD needs the time series.

**The two rigs are mirror images.**
- The big field is filmed by a different camera from the task maze.
- The maze pixel-to-cm affine has det < 0 in all 102 mEC maze sessions. The arena map has det > 0, with
  a Jacobian of +0.017 cm/px at the floor centre.
- So a counter-clockwise turn raises the HD angle in one frame and lowers it in the other.
- The reflection is s = sign(maze det) x sign(arena det) = -1.
- Every session's ear axis must agree. The left-to-right ear axis sits at -85 to -88 deg from heading on
  the maze and +85 to +88 deg in the big field. The module raises on any mismatch.

**Maze HD.**
- Per maze session, pxd tuning with the open-field run's own settings: 7.5 cm x 30 deg bins.
- Each unit gets R, a preferred direction and a split-half r. Each is tested against the unit's own
  circular-shift p99 (R) and p95 (split-half r).

**Coherence per maze-arena session pair.**
- Units are HD-classed in both sessions; a sensitivity uses open-field HD units only.
- For each unit, delta = theta_arena - s theta_maze. The rotation phi is the mean of delta.
- Reported per pair: R, a bootstrap CI, and each unit's leave-one-out error.
- The null permutes maze preferences among units: 10 000 permutations, or all of them when n <= 8.
- Maze-maze pairs use s = +1 and a V-test against no rotation; a |phi| > 45 deg is flagged.
- Closure checks:
  - phi(m, a2) - phi(m, a1) must equal the open-field rotation's HD shift;
  - phi(m2, a) - phi(m1, a) must equal -s phi(m1, m2).
- Per mouse: the mean over pairs, against null draw i averaged over the same pairs. bp02 is excluded.

**Class links.** Each is tested by shuffling labels among a recday's units, then pooled per mouse:

| link | statistic | direction |
|---|---|---|
| maze HD R in open-field HD cells | median difference | higher |
| maze edge index in open-field border cells | median difference | higher |
| open-field speed factor of maze node cells | median difference | lower, the behavioural prediction |
| open-field speed factor of maze bridge cells | median difference | higher |
| open-field spatial information against maze stab21 | Spearman r | positive |

- The edge index is log((rate within 1.5 cm of a drop-off + 0.1) / (rate at >= 4.5 cm + 0.1)). It is
  computed on moving in-trial samples, with near and far samples speed-matched.

**Gate** `maze_of_link_synthetics.py --quick`. It runs on bp01 04-02/03 (maze 1) and bp04 03-21/23
(maze 2). Run 3752863 passed 11/11:

| check | planted | measured |
|---|---|---|
| L1 | coherent HD at s = -1, phi = 70 deg | max rotation error 1.5 deg over 8 pairs; max unit-shuffle p 0.0005; R above the other reflection in 8/8 |
| L2 | independent preferences | median p 0.717 over 10 pairs |
| L3 | arena preferences left unreflected | R 0.06, against 1.00 for the other reflection, which is flagged |
| L4 | maze HD cells; maze place cells | called HD 0.83; 0.00 |
| L5 | one maze session flipped 180 deg | its pairs within 1.2 deg of 180 and the others within 1.2 deg of 0; closure error 0.0 deg over 25 |
| L6 | slow-speed cells; location-only node cells | node-speed link p 0.0005; p 0.61 |
| L7 | mismatched unit tables; an ear axis against the calibrations | both raise |

- L5 first flipped a short session that had no HD pairs. It now flips the longest maze session.

### 2026-10-12, later — link run `f2d171c7b3d6`, 9 mEC recdays

- The first run, `f2d17144bfee`, crashed on maze sessions without tracking. One such session is in bp01
  03-21/23 and one in bp02 03-19/21. It is superseded and left with 4 of 9 shards.
- `maze_hd` now skips a session without XY or HD. New gate check L8 runs bp02 03-19/21 as recorded,
  through the link with its real shards. Gate run 3755119 passed 12/12, and L1-L7 were unchanged.
- One config hash `f2d171ed83`, code hash `c7b3d6f529`. Shards ran in 19 s to 2.2 min.
- bp02 has 0-1 HD units per pair and is not interpreted.

**Reflection.** s = -1 in all 9 recdays. Every tracked session's ear axis agrees: the sign is -1 on the
maze and +1 in the arena.

**HD coherence across the rigs.** Per mouse: the mean R over maze->arena pairs, against unit-shuffle null
draw i averaged over the same pairs.

| mouse | pairs | units per pair | R, HD in both | null mean | p | R, open-field HD units only | p |
|---|---|---|---|---|---|---|---|
| bp01 | 21 | 8-21 | 0.80 | 0.25 | 0.0001 | 0.60 | 0.0001 |
| bp04 | 8 | 5-9 | 0.67 | 0.37 | 0.017 | 0.56 | 0.017 |

**The rotation phi**, arena = -maze + phi, per pair with p <= 0.05:

| recday | arena session | phi | note |
|---|---|---|---|
| bp01 03-21/23 | s4 | 63 to 82 deg, 6 maze sessions | maze s3 at -117 deg: its HD is turned 180 deg against the other maze sessions |
| bp01 03-24/25 | s8 | 36 to 43 deg, 6 sessions | |
| bp04 03-24/25 | s8 | 34 to 47 deg, 5 sessions | two more at 12-14 deg, p 0.053-0.058 |
| bp01 04-02/03 | s4 / s9 | -33 to -54 deg / 120 to 132 deg | the open-field rotation between s4 and s9 |
| bp04 03-21/23 | s4 | not testable | 2-5 HD units per pair |

- **Same day, same rotation.** On 03-24/25, bp01 and bp04 give the same phi, 36-43 and 34-47 deg. A
  rotation shared by two mice on one day points to the rigs or the room cues rather than to each mouse's
  map. It changes from day to day.
- **Within-maze flips.** Of 69 testable maze-maze pairs in bp01 and bp04, 6 are flagged at |phi| > 45 deg.
  All 6 pair bp01 03-21/23 maze s3 with another session, at 169-180 deg, R 0.76-0.91. This is the known
  bp01 s3 flip.
- **Closure.**
  - Maze-pair closure errors: median 4.8-13.1 deg per recday, max 36 deg.
  - bp01 04-02/03 arena pair: the open-field rotation test gave an HD shift of 156 deg. Through the maze,
    161-174 deg, errors 5-30 deg over 4 maze sessions.

**Class links.** Per mouse, against label shuffles within recday:

| link | bp01 | bp04 |
|---|---|---|
| maze HD R higher in open-field HD cells | +0.003, p 0.47; 27-34 HD of 40-48 units | +0.35, p 0.008 |
| edge index higher in open-field border cells | no border cell among the linked units | +0.65, p 0.29; 2 border cells |
| open-field speed factor lower in maze node cells | 0.23 lower, p 0.17; 6 node cells per recday | 0.61, p 0.29; 2 node cells |
| open-field speed factor higher in maze bridge cells | 0-1 bridge cells | 0 bridge cells |
| open-field spatial information against maze stab21 | r -0.02, p 0.58 | r -0.25, p 0.91 |

- The behavioural account of node cells predicts slow-speed cells. It is not supported in either mouse,
  but there are only 6 and 2 node cells.
- Commands:

      bash sbatch_files/submit_spatial.sh link mec <9 recdays>
      python code/run_spatial.py --engine link --dataset mec --merge --expect-recdays 9 --run-tag f2d171c7b3d6

## 2026-10-12 — round 2: one engine for the 120 cm field and the 50 cm LEC box

Plan: `~/.claude/plans/wiggly-finding-moth.md`, round 2. The user chose per-arena blocks in one config file.

**Config.**
- `code/spatial_of_arenas.json` is mirrored into `mFC_data/code/`. It holds one block per arena kind:
  - `bigfield`: today's values;
  - `box50`: the LEC box values.
- Each session picks its block by arena name (`bigfield`, `openfield_box1`, `openfield_box2`); the
  arena's side must match the block's.
- The engine fails at import unless the bigfield block equals today's settings.
- An arena setting set at the top level, where the blocks would shadow it, is refused.

| setting | bigfield | box50 | why box50 differs |
|---|---|---|---|
| border minimum field | 200 cm2 | 50 cm2 | 200 cm2 is 8 % of the box floor |
| wall RMS limit | 1.5 cm | 2.0 cm | the box method floor is 0.5-1.6 cm (MEC_INTEGRATION 7.9) |
| grid | classified | scored and reported only | 20 cm grids were found at 0.55 and 30 cm spacing was NaN in the box (gate run 3712175) |
| object band | 20 cm | 7 cm | 20 cm is 40 % of the box side |
| object mask margin | 4 cm | 2.5 cm | keeps wall bins valid beside objects 9.3-12.7 cm from the walls |
| object radius | full-annulus density, 15-35 cm reference | in-box annulus area, 10-20 cm reference | the 15-35 cm ring is cut by two walls in the box |
| virtual-object null | on | off, reported as not constructible | no wall-distance-matched position exists |
| speed-factor bins | 2.5 ... 40 cm/s, top 50 | 2.5 ... 20 cm/s, top 25, >= 20 s dwell per bin | survey, below |
| HD guard R | 0.8 | 0.65 | survey, below |
| object-move test | off | on | the box has no object-free session |

**LEC survey.** `of_arena_survey.py`, job 3751829, output `qc/of_arena_survey_lec.csv`. 20 object
sessions, 15 min each.
- **Moving speed:** p50 5.5-8.6 cm/s, p99 20.1-32.7 cm/s. Above 20 cm/s each session has only 5-49 s
  of dwell, so the top bin opens at 20 cm/s and needs >= 20 s.
- **Object radius** from the occupancy hole: 2-4 cm in all 40 object-sessions, by both methods. Mice
  approach the objects rather than avoid them, so the radius is effectively the 2 cm floor. All are
  below the user's recollection of objects under 14 cm across.
  - Object masks stop 3.8-8.2 cm short of the walls.
  - Each object band, r_obj to 12 cm, holds 2,566-9,157 moving samples.
- **Walls:** RMS 0.48-1.86 cm. No wall is excluded at 2.0 cm. At 1.5 cm six walls in 5 sessions would
  be: ah08 s9, ah10 s9 (two walls), ly07 s4, s5 and s9.
- **HD guard:**
  - At R >= 0.8 it passed in only 10 of 20 sessions: R 0.70-0.84, with offsets -8 to +1 deg.
  - The small offsets show the HD column is the head axis. The lower R reflects mice turning their
    heads while exploring objects in a small box.
  - So the box keeps the +-15 deg offset test and uses R >= 0.65, which all 20 sessions pass.
- **Wall-QC reader fix:**
  - The reader had picked the newest table version, a 4-row ah10 add-on, so 16 of 20 LEC sessions had
    no wall exclusions.
  - `spatial_io._wall_qc` now reads every version: LEC gets 20/20 sessions; mEC is unchanged, 11/11
    rows with identical exclusions.

**Object-move test** (`object_layout`, `object_move` in `spatial_tuning.py`). Within each day-pair,
s1->s4 and s5->s9, the corners are classed:
- V: vacated;
- S: held in both sessions by different objects;
- N: newly occupied;
- E: empty in both.

All 10 LEC day-pairs have one of each. One object slides V->S; its partner moves diagonally S->N.

How each unit is scored:
- **Statistic.** At each corner, take the most active of 8 sectors of the band, from the object edge
  to 12 cm. Its rate is compared with the rest of the floor, reweighted to that sector's mix of speed
  (2.5 cm/s bins) and head direction (45 deg bins).
- **Null.** Spikes of both sessions are circularly shifted, the session-a shift paired with the
  session-b shift. Each unit is scored against its own one-sided p99s.

| class | rule |
|---|---|
| object-following, per object | firing drops at the object's old corner and rises at its new one |
| object-trace | firing rises at V after the object leaves, and is significant there afterwards |
| generic object | firing drops at V and rises at N |
| object-vector, per object | the a map moved by the object's displacement matches the b map better than the unmoved map, with the field beyond the band in both sessions |
| reported | place at V, remapping V->E, E changes, where units that left V arrive |

Two changes from the plan:
- **Reweighting instead of subsampling.** The plan drew a matched subsample of the rest for each
  sector. Reweighting gives the same expectation, is deterministic and is about 1000x faster.
  Consequence: the planned check that the new path equals the old subsampled statistic exactly does
  not apply.
- **Object-vector fields beyond the band.** The plan put planted object-vector fields 8-10 cm from
  the object. That lies inside the 12 cm band, where they would count as object-following. They sit
  16-18 cm out instead, beyond the band.

**Local test.** A synthetic 50 cm box with the LEC layout, 100 shifts and 40 units:
- every planted type found 6/6;
- noise 0/16 in every class;
- one false cross-object call, an object-vector cell of one object also scored for the other (1/6).

The full gate runs after the round-2 maze jobs.

### 2026-10-12, later — the LEC gate in the 50 cm box: what failed and what changed

Quick gate, `spatial_tuning_synthetics.py --dataset lec --quick`, on ah10 06-24/25. A17L plants 10 cells
per kind.

| run | passed | failures |
|---|---|---|
| 3752837 | 92/97 | A8: 0.10 of border cells without HD called HD; A9 0.65; A17L: generic 0.60 of 5, object-vector 0.00 of 5, central anchors in a primary class up to 0.20 |
| 3752932 | 95/97 | A4: two-wall border 0.00, because the pooled threshold was NaN; A9 0.55 |
| 3752973 | 95/97 | A9 0.55; A17L generic 0.50 of 10 |
| 3753086 | 97/98 | A17L generic 0.70 of 10 |
| 3755100 | 98/98 | none |

Engine changes, each with the measurement that forced it:
- **Box position x HD bins 5 cm, from 7.5.** Together with the planting fix below, A8 went from 0.10 to 0.00.
- **Box smoothing 1 bin (2.5 cm), from 2 bins.** 2.5 cm is 5 % of the box side; the big field's 5 cm is
  4 % of its side.
  - Diagnostic `diag_box_border.py`, jobs 3752961 and 3752965: one- and two-wall border cells with
    5 cm decay, plus near-wall and central place cells, 10 of each.
  - At 2 bins, almost no shifted map held a valid border field. The pooled threshold swung between 0.58
    and NaN, so two-wall cells were never called.

| smoothing | planted border score, median | border cells called | shifted maps with a border field | pooled p99 | place cells called border |
|---|---|---|---|---|---|
| 2 bins | 0.71-0.72 | one wall 1.00, two walls 0.00 | 0.000-0.054 | NaN-0.685 | 0.00 |
| 1.5 bins | 0.74 | 1.00 | 0.001-0.184 | 0.60-0.67 | 0.00 |
| 1 bin | 0.76-0.77 | 1.00 | 0.096-0.323 | 0.62-0.67 | 0.00 |

  - 1 bin, not 1.5: at 1.5 bins the two-wall threshold rested on 0.1 % of shifted maps.
- **Object-move classes need firing at the corner.**
  - The follow, generic and remap-to-E classes also need the unit to fire significantly at the old
    corner in a and at the new corner in b.
  - Why: a central place field sits in the rest of the floor. Occupancy changes moved its band/rest
    ratio at corners where it never fires, and 2 of 10 central anchors were called object-following
    (run 3752932). With the requirement: 0 of 10, and 0 of 20 in diagnostic job 3752998.
- **Four-term classes test each term at p95** (`move.q_conj`); trace and object-vector stay at p99.
  - A follow, generic or remap class is four one-sided terms: fires at the old corner in a, drop there,
    rise at the new corner, fires there in b.
  - The old-corner half and the new-corner half are nearly independent under the null, so p95 per
    term runs the class near 0.25 %. Trace is two correlated terms, so it keeps p99.
  - The plan had every term at p99. At that level generic object cells were found in 0.37-0.45. Their
    other fields raise the rest-of-floor rate, and their extra bursts widen the shifted null at the new
    corner.
  - Diagnostic 5, `diag_move_fire5.py`, job 3755058: 20 planted cells per type, ah10 s1->s4.

| variant | follow slider | follow diagonal | trace | generic | 13 negative types | noise in any class |
|---|---|---|---|---|---|---|
| every term p99, regulariser 0.1 Hz | 1.00 | 0.90 | 1.00 | 0.45 | 0 | 0/30 |
| every term p95, 0.1 Hz | 1.00 | 1.00 | 1.00 | 0.95 | 0 | 2/30, both trace |
| every term p99, 0.5 Hz | 1.00 | 1.00 | 1.00 | 0.70 | 0 | 0/30 |
| every term p95, 0.5 Hz | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 2/30, both trace |

  - The regulariser stays at 0.1 Hz.
  - A sector-matched change, the best sector of b minus a at one corner, was worse: follow slider 0.60,
    diagonal 0.30, generic 0.20 (job 3753986).

Gate planting fixes, none of which changes the engine:
- **Object-vector fields keep a fixed allocentric offset**, 17 cm into the box from the slider's wall.
  The old offset pointed toward the box centre, so it rotated with the corner and moved the planted field
  5 cm for a 29 cm move.
- **Border decay 3-5 cm.** At 2-4 cm, the big field's decays scaled by 0.4, the fields sat below the
  2.5 cm bin.
- **A17L planting.**
  - max(10, nper / 4) cells per kind, with 12-20 Hz peaks.
  - Rejections are binomial tests against 5 %, because each primary class is an intersection of tests.
- **Generic cells have one 0.2 Hz floor**, like every other planted type.
  - Summing two place fields had doubled the floor.
  - The fix made the test harder, not easier: detection at p99 fell from 0.65 to 0.37 (jobs 3752998,
    3753163). The p95 decision above was taken on the fixed plant.
- **A9 plants HD preferences the mouse shows at the wall.** Diagnostic `diag_box_conjunctive.py`, job
  3752994, and gate run 3753086:

| HD preference | moving seconds within 5 cm of the wall | conjunctive cells called border, decay 5 cm |
|---|---|---|
| 0, 45, 315 deg (facing away) | 2, 9, 3 | 0.0, 0.2, 0.0 |
| 90 to 270 deg | 12-24 | 0.8-1.0 |

  - All 40 cells were stable and called HD.
  - A conjunctive field shows only where the mouse faces the preferred way. In the box, a mouse
    running along a wall rarely faces away from it.
  - Preferences with under 10 s at the wall are planted separately and reported: border 0.10.

What this means for the data: a real conjunctive border x HD cell whose preferred direction points away
from its wall cannot be called border in this box.

**Full gate, second day-pair.** The full gate adds ly06 s5->s9 and plants 15 cells per kind. Run 3755102
passed 112/114; ah10 passed every A17L check. The two ly06 failures came from the mouse's behaviour, not the
classes:
- **Trace 0.00 of 15; place at V 0.00.** In s9 the mouse crossed the band sectors between the vacated
  corner and the box centre for 38 and 43 moving samples. The engine needs 100 per sector, and ah10 had
  105-1198. A field planted there cannot be scored, in the gate or in the data (diagnostic
  `diag_move_ly06.py`, job 3755355).
- **Object-vector 0.73 of 15.** ly06 moved 48 % of the time in s5; ah10 moved 65-72 %. The planted
  vector cells fired 0.21-0.28 Hz in s5, which is about 100 moving spikes, the eligibility cut.
  Re-planted with the full population and 1000 shifts, all 15 were eligible and all were found: statistic
  0.77-1.03 against p99 0.49-0.60 (job 3755357).
- **Gate changes, no engine change:**
  - A corner field sits 9 cm toward the box centre unless that sector holds under 100 moving samples
    in a session where the field must be seen; then it goes in the best-sampled sector. A reported
    line lists each moved field.
  - Detection and rejection rates are counted over planted cells eligible in both sessions, with the
    eligible count printed.
  - Quick gate run 3755671 passed 99/99. ah10 needed no move, and every value equals run 3755100.
- **Full gate run 3755672: 115/116.**
  - With the fields moved, ly06 place at V was found 1.00, and vector cells in 11 of 11 eligible.
  - Planted trace cells were 0 of 15 eligible. A trace cell fires at 0.2 Hz before the object leaves.
    At ly06 s5's 48 % moving, that is about 89 moving spikes, under the 100-spike cut, and every class
    required eligibility in both sessions.
- **Engine change: trace needs eligibility in session b only.** An object-trace cell may be silent until
  the object leaves; requiring spikes before excludes exactly those cells. The class itself still
  requires firing at V in b. The other classes require firing in both sessions, so they keep
  both-session eligibility. `object_move` returns the per-class eligibility, and the gate counts each
  class over it.
- **Gates after the change:**
  - LEC quick, run 3755717: 99/99, identical to run 3755671.
  - mEC quick, run 3755719: 95/95, identical to run 3752974.
  - LEC full, run 3755718: 116/116. On ly06, trace was found in 0.80 of 15 eligible cells and every
    other class in 1.00.
- **Runs superseded by the change:** `a884c261cfd9` and the 2 completed shards of the sensitivity runs
  started before it. The link run `f2d171c7b3d6` used `spatial_tuning.py` from before the change. The
  change touches only `object_move`, which the link does not call.

**Wall limit per arena, a wiring fix.** The wall-RMS sensitivity run, 1.5 cm in the box, matched the main
run in every count. The reason: `run_spatial.py` filtered walls at the top-level limit, 1.5 cm, before the
engine saw them. That shadowed the box50 block's 2.0 cm, so the main LEC run had excluded 6 walls of
1.5-1.9 cm (ah08 s9; ah10 s9, two walls; ly07 s4, s5 and s9).
- `run_spatial.py` now passes every measured wall. `run_of_recday(..., wall_rms=...)` excludes walls above
  each session's own block limit, via `session_wall_exclusions`.
- Gate W2:
  - toy walls give {top} at the box limit and {right, top} at the big-field limit;
  - real walls excluded at 2.0 / 1.5 cm: LEC 0 / 6, mEC 1 / 1 (bp02 s4 right).
- Quick gates 3755942 and 3755943 passed 100/100 and 96/96. Every line equals the previous run, with W2
  added.
- `a884c25a6eb2` and its sensitivity runs are superseded by `a884c28b4e13`. The mEC big field uses
  1.5 cm both ways, so its exclusions are unchanged.

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

### 2026-10-12, later — run `a884c28b4e13`: mEC unchanged, LEC object box

One config hash, `a884c2f434`, and one code hash, `8b4e13eb9f`, for both datasets.

It replaces two earlier runs:
- `a884c261cfd9`, before the trace-eligibility change;
- `a884c25a6eb2`, before the wall-limit fix.

What changed between the three runs:
- Every object-move count is identical; trace's eligible pool grew from 967 to 986 unit-pairs.
- With the 6 LEC walls of 1.5-1.9 cm no longer excluded, ah08 s9 has 8 border cells instead of 9. No
  other session class changed.

**mEC is unchanged.**
- `spatial_compare_runs.py` against `8a49b0e2d93c`: 9/9 recdays reproduced exactly, for all three runs.
- Max abs diff 0, 0 NaN mismatches and nothing missing over 8.75 M values.
- The per-arena config leaves every big-field number where it was.

**LEC, 5 mice, 1 recday each (06-24/25), 4 object sessions of 15 min.**
- Session classes are counted over broad eligible units, against each session's one-shift-for-all chance
  distribution; p <= 0.004 unless stated.

| mouse | units per session | border | HD | spatial | speed |
|---|---|---|---|---|---|
| ah08 | 140-151 | 5-8 | 4-14 | 6-20 | 1-8; s9 p 0.048 |
| ah10 | 112-119 | 4-6 | 15-27 | 18-25 | 6-17 |
| ly05 | 48-54 | 1-4; p 0.001-0.021 | 9-12 | 5-8 | 4-6 |
| ly06 | 44-48 | 0-4; s1 p 0.008, s4 0 | 10-13 | 4-11 | 5-8 |
| ly07 | 65-68 | 0-1; 3 of 4 sessions 0, s5 p 0.022 | 22-28 | 4-10 | 0-8; s9 0 |

- HD cells are common where the probe reached ENTm: 0.54 of ly06 and 0.40 of ly07 ENTm unit-sessions.
  Anatomy is descriptive only.

**Object moves**, within each day: s1->s4 and s5->s9, 10 day-pairs.
- Counts are over units eligible in both sessions.
- Per mouse, both day-pairs are pooled, and the chance distributions are summed draw by draw.
- A hit is p <= 0.05; across mice, hits out of 5 with a binomial tail at 0.05.

| class | units, all mice | hits / 5 | binomial p | mice |
|---|---|---|---|---|
| generic object, V -> N | 21 of 967 unit-pairs; chance mean 0.22 in total | 5 | 3e-7 | all |
| object-following, slider | 2 | 2 | 0.023 | ah10, ly06, one unit each |
| object-following, diagonal mover | 1; chance mean 0.24 | 0 | 1 | |
| object-trace | 1 of 986; chance mean 4.8 | 0 | 1 | |
| object-vector, either object | 10; chance mean 15.0 | 0 | 1 | ly05 slider p 0.070 |
| place at V, reported | 5 | 4 | 3e-5 | ah08, ah10, ly06, ly07 |
| remapping V -> E, reported | 5 | 1 | 0.23 | ah10 |
| E change, reported | 16 | 1 | 0.23 | ah10 |

**The matched control for generic is remapping V -> E.** It is the same four-term test, with the empty
corner E as the destination instead of the newly occupied N:

| mouse | generic, V -> N | remapping, V -> E |
|---|---|---|
| ah08 | 4 | 0 |
| ah10 | 8 | 5 |
| ly05 | 1 | 0 |
| ly06 | 5 | 0 |
| ly07 | 3 | 0 |

- In 4 mice, generic units exist and the matched remapping count is 0.
- ah10 remaps beyond chance in general: E change 8 units, p 0.017. So its generic count is the weakest
  evidence.

What this shows:
- A small population of LEC units, about 2 % of unit-pairs, fires at whichever corners hold an object and
  follows the occupancy when one object moves. It is above the shift null in 5/5 mice and above the
  matched remapping control in 4/5.
- Object-following, object-trace and object-vector cells are not established. The gate found planted ones
  at 0.87-1.00 on both trajectories.

Not established:
- **Identity and novelty.** The box has no object-free session. Object identity, novelty and time of
  day change with every move, so generic cells cannot be told apart from cells tuned to a change of
  layout.
- **ly06 s9 coverage.** The vacated corner's centre-facing sectors were visited for about 1 s, so fields
  there cannot be scored on that day-pair.
- **Cross-day pairs.** The engine also scores s1->s9 and s4->s5, which have four different objects. They
  are not reported here.

**Sensitivity runs**, the same code with one box50 setting changed. Each cell is units summed over the 20
sessions, then sessions above chance at p <= 0.05:

| class | main: 2.0 cm wall limit, 1-bin smoothing, pxd 5 cm | smoothing 1.5 bins (`c5f13b8b4e13`) | wall limit 1.5 cm (`7a55448b4e13`) | pxd 7.5 cm (`0c43d68b4e13`) |
|---|---|---|---|---|
| border | 64; 16 | 45; 14 | 65; 16 | 64; 16 |
| HD | 304; 20 | 304; 20 | 304; 20 | 328; 20 |
| spatial | 222; 20 | 235; 20 | 222; 20 | 248; 20 |
| speed | 126; 19 | 126; 19 | 126; 19 | 138; 20 |

- Border counts depend on smoothing: at 1.5 bins the pooled threshold rests on few valid shifted maps,
  as the diagnostic showed.
- The coarser 7.5 cm position x HD bins let 8 % more units pass as HD and 12 % more as spatial. That is
  the leak the 5 cm bins were chosen to stop (gate A8).
- Every object-move count and hit is unchanged under all three, except object-vector on the slider: 4 units
  instead of 5 at 1.5 bins, still 0 of 5 mice.

Notebook: `LEC_spatial_tuning.ipynb`.

### 2026-10-13 — figures

Round-2 figures are in `data/spatial/figures/round2/` in the LEC tree, PDF + PNG. They come from
`code/spatial_round2_figures.py`, run under sbatch in about 2 min.

| file | shows |
|---|---|
| `C1_link_hd_offsets` | per recday and arena session, each unit's offset arena - s x maze; one ring per maze session; lines = fitted rotation |
| `C2_link_mouse_tests` | per mouse, HD coherence against unit shuffles and the four class links against label shuffles |
| `C3_link_rotation_by_day` | the rotation per maze session by day for both mice; maze-maze rotations, which show the bp01 03-21/23 s3 flip; closure errors |
| `D1_{lec,mec}_session_classes` | per session, the class fraction against chance p95 |
| `D2_lec_object_layouts` | V, S, N and E with the two moves for the 10 day-pairs |
| `D3_lec_move_classes` | per class and mouse, units against the pooled chance distribution; generic against its matched V -> E control |
| `D4_lec_move_units` | before and after rate maps of every unit in a move class, 3 pages; the PNG is page 1, the 21 generic units |
| `D5_lec_sensitivity` | session and move class counts across the three box sensitivity runs |
| `V1_gate_decisions` | the gate measurements behind box smoothing, the A9 plant and the four-term test level |

Other figures:
- **Per-unit galleries**, one PDF per recday:
  - mEC, `<mEC root>/data/spatial/of/8a49b0e2d93c/figures/`. The final run reproduces it exactly. The
    `.v1.pdf` copies are the latest.
  - LEC, `data/spatial/of/a884c28b4e13/figures/`.
- **Notebooks with saved figures:**
  - `MEC_spatial_tuning.executed.v1.ipynb`: class counts and an example gallery page;
  - `LEC_spatial_tuning.ipynb`: class counts.
- The `.executed.ipynb` copies made first have no figures. `spatial.sbatch` sets `MPLBACKEND=Agg`, which
  the kernel inherited; `run_notebook.py` now clears it.

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

2026-10-12, round 2:

    python code/of_arena_survey.py --dataset lec                                  # sbatch: loads the data_dic
    sbatch ... sbatch_files/spatial.sbatch code/spatial_tuning_synthetics.py --dataset lec [--quick]
    bash sbatch_files/submit_spatial.sh of mec <9 recdays>; bash sbatch_files/submit_spatial.sh of lec <5 recdays>
    python code/spatial_compare_runs.py --engine of --dataset mec --old 8a49b0e2d93c --new <tag>
    SPATIAL_CONFIG=$PWD/data/spatial/of/config_box_sigma15.json bash sbatch_files/submit_spatial.sh of lec <5 recdays>   # also wallrms15, pxd75
    sbatch ... sbatch_files/spatial.sbatch code/maze_of_link_synthetics.py --quick
    bash sbatch_files/submit_spatial.sh link mec <9 recdays>
    # notebooks, load mode: LEC_spatial_tuning.ipynb, MEC_maze_of_link.ipynb
