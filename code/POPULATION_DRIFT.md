# Population drift in LEC vs PFC (Kanter 2025), on the event-kernel session format

Kanter et al. 2025 population measures: how far the population state moves between 1-min
epochs, whether an epoch can be decoded within its own session (locked clock plus drift) or only
from other sessions (locked clock alone), and whether the population trajectory jumps at task
events. Plan, decisions and code map: `docs/handoff/EVENT_KERNELS_HANDOFF.md`. Per-neuron kernels:
`code/EVENT_KERNELS.md`.

## 2026-10-16 — M4 part 1: drift in rest against task (Kanter Fig 2)

Rest-entry kernels and timescales are in `EVENT_KERNELS.md`, same date.

**Outcome.** Kanter found drift in REM equal to drift in foraging. Here, low-DOWN rest drifts more
than task within 60 s segments, but its epochs decode worse.

| 20-unit draws, mean over segments, draws and recdays, per mouse | LEC, 5 mice | PFC, 6 mice |
|---|---|---|
| drift growth, low-DOWN rest | 0.014-0.026 | 0.029-0.072 |
| drift growth, task | -0.003 to 0.017 | 0.005-0.056 |
| rest above task | 5 of 5 mice | 5 of 6 mice |
| epoch decoding over chance, low-DOWN rest | 1.41-1.56 | 1.94-3.32 |
| epoch decoding over chance, task | 1.49-1.85 | 1.90-2.98 |
| rest below task | 5 of 5 mice | 3 of 6 mice |

- High-DOWN rest has segments in one LEC mouse, ly07: growth 0.062, decoding 1.03. In PFC it has
  them in 5 mice: growth -0.004 to 0.013, decoding 1.36-1.68.
- The gate's planted equal drift reads 0.935 (LEC) and 0.909 (PFC) on 54 and 89 segments, and 4x
  faster drift reads 2.20 and 2.29.

### What was done

- `rest_kernels.drift_segments`: 60 s segments at 1 s bins and 6 s epochs, every non-overlapping one
  in each low-DOWN or high-DOWN run of at least 72 s. Task segments come from sessions without their
  first 2 and last 1 min, matched in number; each context is capped at 120.
- `rest_kernels.segment_drift`: distance growth, lags 7-9 minus lag 1, and kNN decoding of the 6 s
  epoch within the segment. M2's unit sets; the tables read the 20-unit draws.

### What was NOT established

- **That rest drift is faster in the representational sense.** Low-DOWN rest mixes quiet wake,
  drowsiness and REM, and state changes move population vectors. Kanter's soft normalisation keeps
  rate changes.
- **That task is the right reference.** A 60 s task segment holds about one loop, median 43.6 s, so
  its distance-by-lag curve carries the task's period. Foraging had none.
- **REM.** It cannot be separated without LFP or tracking.
- **LEC vs PFC.** No contrast was tested.

### Commands

As `EVENT_KERNELS.md`, 2026-10-16: drift runs inside `run_rest_kernels.py`; F35 is in
`code/event_kernels_M4_rest.ipynb`.

## 2026-10-14 — M3 part 2: Kanter part 2 (boundaries, coding axes, Fano removal), and the ramp test

The ramp test itself is in `EVENT_KERNELS.md`, same date. It answers the caution below. M2's
held-out first-minute decoding is mostly the slow entry ramps' early slope: dividing out held-out
ramps with tau >= 240 s removes 55% of the advantage in LEC, against 62% for the whole shape-free
kernel. The rest sits in units Stage 1 did not call entry-locked.

**Outcome.**

- **Rewards are boundaries in Kanter's sense, but each is stamped by the same ensemble every loop,
  not a new one.**
  - Windows run from -3 to +4 s around each arrival at a reward, per reward type, in 1 s bins.
  - The population shift falls on the change into the 0-1 s bin, in both datasets.
  - Units prefer that transition at 0.29 in LEC (0.24-0.35 over 5 mice) and 0.37 in PFC
    (0.28-0.42, 7 mice), against chance 0.167.
  - Units modulated there correlate across loops more than other units do. Kanter found the
    opposite.

    | | up-modulated | down-modulated | other units, 25-unit sets |
    |---|---|---|---|
    | LEC PV correlation across loops | 0.33 (0.24-0.44) | 0.42 (0.37-0.51) | 0.20 (0.16-0.24) |
    | PFC | 0.53 (0.33-0.83) | 0.50 (0.38-0.72) | 0.18 (0.00-0.32) |

  - Down-modulated units outnumber up-modulated ones per session: 19.5 vs 12.8 in LEC, 13.1 vs
    6.4 in PFC. Most reward-locked units dip after the reward (M2, M3).
- **Loop number decodes from the boundary windows, but drift alone does that.**
  - Errors are 4.2 loops in LEC and 5.7 in PFC, against shuffles of 6.5 and 8.2.
  - Planted drift without boundaries decodes it to within 0.2-0.3 loops (gate P8). So this is not
    evidence of event time stamps.
- **The within-leg axis and the session axis are stable and near-orthogonal, as Kanter found, but
  not fully orthogonal.**

  | | leg vs session offset | stability, leg / session | stable sessions |
  |---|---|---|---|
  | LEC, 20-unit draws (5 mice) | 71.7° (70.6-72.7) | 7.0° / 10.2° | 0.90 |
  | LEC, all units | 78.4° (73.1-81.4) | 9.0° / 14.3° | 0.81 |
  | PFC, 20-unit draws (6 mice) | 69.8° (67.2-71.7) | 7.0° / 10.3° | 0.90 |
  | PFC, all units (7 mice) | 67.6° (43.0-81.7) | 7.5° / 8.9° | 0.88 |

  - A session counts as stable when both leave-one-out angles are <= 30°.
  - Noise axes read 81-85°, with stabilities of 50-68° (gate P15). So the real axes are genuine
    coding axes that share some direction.
  - The loop axis behaves like the leg axis: loop vs session is 71.4° in LEC and 71.3° in PFC at
    20 units.
  - Per reward type, Kanter's place-bound analog, the leg axis is less stable: 20-25° in both.
  - Multiplexing, units in the top quarter of both axes, is 0.083-0.11 against chance 0.0625.
- **Part of the drift is carried by the most variable units.**
  - Drift growth after removing the top 25% of units by Fano, over that after removing a random
    25%: LEC 0.51-0.62 per mouse, PFC 0.55-0.70.
  - Planted drift in 25% of units gives 0.00, and drift in every unit 0.77-0.80 (gates P12/P13).
    The real values lie between.

### What was done

- **`population_drift.py` (+ mirror), new functions only:**
  - `reward_legs`, `boundary_windows`, `window_speed`, `boundary_responses`, `pv_correlation`,
    `boundary_ensembles`, `trial_number_decoding`;
  - `coding_axis` (PCA to 50% of the variance, then LDA; the loadings of the PC contributing most
    to LD1, as |weight| x PC sd), `axis_offset`, `axis_stability` (up to 30 random leave-one-out
    groups), `multiplexing`;
  - `log_fano`, `fano_removal`.
- **`population_drift_synthetics.py` (+ mirror):** `planted_part2` with populations
  `fixed_boundary`, `leg_drift`, `shared_axis` and `subset_drift`. `planted` is unchanged.
- **`run_population_drift.py` (+ mirror):** `--part2` per recday and `--part2 --merge`, with M2's
  unit sets. Boundaries, decoding and Fano removal run on `all` and `region_full`; axes also on the
  20-unit draws. `--gate-part2` is new.
- **Data:** sessions with >= 5 trials; every unit and recday. Tables and figures read the `all`
  set only where a recday has >= 20 units. PFC me10_20122021_21122021 has one unit, and its loop
  decoding did worse than its shuffle.
- **Runs:** 25 + 25 jobs, every one `EXIT=0`.

### Gates (final runs 3766906 LEC and 3766908 PFC, `EXIT=0`)

| row | planted | LEC | PFC |
|---|---|---|---|
| P5 | fixed ensemble: shift preferred / PV correlation of up units | 0.394 / 0.863 | 0.351 / 0.962 |
| P5 | window speed peaks at the burst's onset or offset | 3 | 2 |
| P6 | a new 20% at each reward: PV correlation of up units | -0.012 | -0.019 |
| P7 | noise: shift preferred, chance 0.167 | 0.168 | 0.167 |
| P8 | drift: loop-number error / shuffle (reported) | 0.17 / 8.17 | 0.27 / 10.11 |
| P9 | noise: loop-number error / shuffle | 8.18 / 8.13 | 9.86 / 9.97 |
| P10 | leg-locked plus independent drift: offset / stabilities | 76.1 / 2.8 / 13.2 | 70.5 / 3.8 / 14.2 |
| P11 | one shared axis: offset | 19.3 | 19.3 |
| P12 | drift in 25% of units: top-25% / random removal | 0.00 | -0.01 |
| P13 | drift in every unit: same ratio | 0.80 | 0.77 |
| P14 | noise: median log Fano | -0.009 | -0.019 |
| P15 | noise fails P10's criterion: offset / stabilities | 84.6 / 68.3 / 68.0 | 81.2 / 50.4 / 63.2 |

**Four rows were revised.** One before any run, three after the first full runs (3766749,
3766752). The reasons are in the synthetics module's docstring.

0. **P5, before any run:** with 20% of units bursting, the expected preference for the shift is
   0.2 + 0.8/6 = 2x chance, so the planned 3x could not pass. Now 50% of units are planted, and
   2x is required.
1. **P5:** a 1 s burst has equal onset and offset jumps, so the speed peak may fall on either.
2. **P10/P15:** noise met "offset above both stabilities". In 60 dimensions random axes are
   near-orthogonal and unstable, so the criterion now requires both stabilities <= 30°. That is
   twice Kanter's ~15°.
3. **P13:** ranking by realised Fano picks each session's largest drift excursions even when every
   unit drifts. The test is the contrast with P12.

The M2 runner gate (`--gate`, 3766750/53) was rerun after these edits:

- every kNN, chance and acceleration value equals the M1 reference exactly;
- distance curves differ by 1.4e-17 to 5.6e-17;
- the code M2 uses is byte-identical, since the edits only append functions. So this is
  node-dependent BLAS rounding, which M2 saw at 6.9e-18.

### What was NOT established

- **Why reward ensembles repeat here but not in Kanter's data.**
  - Each ABCD reward type sits at one place, and Stage 1 found reward kernels locked across
    sessions.
  - Kanter's single figure-eight reward differs in timing and in how often it is visited.
  - Place and kinematics were not regressed out of the boundary windows.
- **LEC vs PFC.** No contrast was tested. PFC's per-mouse ranges are wide.
- **Time stamps of individual events** beyond drift. The decoding measure cannot separate them
  (P8).
- **Regions** for the population measures. The region sets are in the merge but not analysed.

### Commands

```
python code/check_mirror_parity.py                                          # must exit 0
for ds in lec pfc; do t=code; [ $ds = pfc ] && t=mFC_data/code
  sbatch --time=0-02:00 --job-name=pd2_gate_$ds sbatch_files/event_kernels.sbatch $t/run_population_drift.py --gate-part2 --dataset $ds --min-trials 5
  sbatch --time=0-01:30 --job-name=pd_rgate_$ds sbatch_files/event_kernels.sbatch $t/run_population_drift.py --gate --dataset $ds
done
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner drift2 --min-trials 5 --after <gate job>
bash sbatch_files/submit_event_kernels.sh --dataset pfc --runner drift2 --min-trials 5 --after <gate job>
python code/run_population_drift.py --dataset lec --part2 --merge --min-trials 5
python mFC_data/code/run_population_drift.py --dataset pfc --part2 --merge --min-trials 5
```

Outputs: shards in `data/population_drift/part2/shards/<ds>_mt5/`; merged
`data/population_drift/part2/part2_<ds>_mt5.pkl` with CSVs; figures F26-F28 in
`data/figures/population_drift/M3_part2/`.

## 2026-10-12, night — a caution on the M2 reading below

M2 read the first-minute concentration of held-out decoding as "mostly an entry transient".

- Stage 2 (`EVENT_KERNELS.md`, M3 part 1) finds the entry-locked units are mostly ramps with tau
  around 5 min: a median 320 s, the middle half 160-910 s.
- A bank of such ramps changes fastest at the start, so the first 1-min epoch is the most distinct
  one across sessions even without any brief transient.
- So the measurement stands (held-out accuracy 0.25-0.26 in minute 1 against 0.07 later), but
  "transient" is not established. Whether held-out decoding is just the ramps' early slope needs a
  test, e.g. decoding on ramp-classed units alone, or on units after regressing out the fitted
  entry ramps.

## 2026-10-12 — M2: Kanter part 1 (WP3), both datasets

**Outcome.**

- Within a session, the 1-min epoch is decodable throughout. Across sessions, it is decodable
  almost only in the first minute after recording start.
- At a matched 20 units:
  - LEC decodes the epoch at 2.9x chance within session and 1.5x chance on a held-out session
    (5 mice);
  - PFC at 2.3x and 1.5x (6 mice).
- Pooled over recdays, held-out accuracy in the first minute is 0.25 (LEC) and 0.26 (PFC). In
  minutes 2-17 it is 0.07 in both, against a chance of 0.056.
- So the locked clock these populations share across sessions is mostly an entry transient. Over
  the rest of the session the change is session-unique drift.
- Population distance grows with lag in both datasets, by about 27% from lag 1 to lag 10 min.
- Rewards accelerate the population trajectory in both datasets, more sharply in PFC. The first
  A does not.

### What was done

| file | role |
|---|---|
| `code/run_population_drift.py` (+ mirror) | one job per recday: unit sets, every measure per set and session, shards stamped as `run_event_kernels`, `--merge` to long tables, `--gate` |
| `code/population_drift.py`, `code/population_drift_synthetics.py` | the M1 measures and P1-P4 gates, unchanged |
| `code/event_kernels_plots.py` (+ mirror) | F12-F14 and `table_knn` |
| `sbatch_files/submit_event_kernels.sh --runner drift` | submission, recdays from files |

- **Sessions:** ABCD sessions with >= 10 trials, the Stage 1 rule; LEC also >= 5. Counts run from
  recording start.
- **Unit sets per recday.** Unit order is depth order on the probes, so every subset is a random
  draw.
  - all units;
  - 20 random units, 20 draws;
  - per LEC region, the units at least 50 um from a region boundary: all of them, and 20 random,
    20 draws, when the region has >= 20 such units. `fibre/other` is left out.
- **Measures (Kanter's, as gated in M1).**
  - Soft normalisation: count / (range + 5).
  - Distance: mean cosine distance between 10 s bins of epochs i and i + lag.
  - kNN: k 10, cosine, distance-weighted; 1-min epochs, the first 18 kept. Within session uses
    5 folds of shuffled bins; held-out session trains on the other sessions.
  - Acceleration: a(t) = v(t) - v(t - 1), with v(t) the cosine distance between 1 s bins t and
    t + 1. Its event-triggered mean is z-scored against all bins, at the first A, each reward A,
    every reward and Kanter's synchrony onsets.
  - Confusion matrices come from a copy of `knn_epochs` that keeps its predictions. It makes the
    same random calls, so the predictions are identical (gate below).

### Runner gate (planted P1-P4 populations through the runner, jobs 3750342 / 3750344)

- The gate recday's planted populations are generated from one random stream in the M1 gate's
  order, so the runner reproduces the M1 full gate pickle.
- Every check passes in both datasets (28/28):
  - kNN within, held-out and chance accuracy per session, chance MAE, the distance curve and the
    reward acceleration z, for each of the four populations: max |runner - M1| = 0;
  - P1-P4 read from the merged tables: noise at chance, locked clock 16-18x chance held-out,
    drift 13-14x within and 0.8-0.9x held-out, boundary z 8.9-9.5 at lag 0 or 1;
  - the merge refuses a shard with one stamp field changed.
- **The first pass failed on 6.9e-18.** Two distance curves (LEC) and one (PFC) differed from
  M1 in the last bit. The gate jobs had 2 BLAS threads and the M1 reference 8. At 8 threads
  (jobs 3750338/39) every value is exact. The gate now documents that it needs 8 CPUs; the
  exact checks were not relaxed.
- At 20-unit draws the P verdicts also hold, reported only: locked 15.5x chance held-out, drift
  12.2x within, boundary z 6.1 (PFC recday).

### Runs

| arm | jobs | recdays | units | notes |
|---|---|---|---|---|
| LEC, >= 10 trials | 25, `EXIT=0` | 25 | 2851 | 20-unit draws on all 25 recdays |
| PFC, >= 10 trials | 25, `EXIT=0` | 25 | 1252 | 20-unit draws on 20 recdays. Five have fewer than 20 units: both of ah03's (16, 15), me10_17122021_19122021 (6), me10_20122021_21122021 (1) and me11_12122021_13122021 (12). ah03 drops out of every matched comparison, which is why PFC has 6 mice there. |
| LEC, >= 5 trials | 25, `EXIT=0` | 25 | 2851 | sensitivity arm |

### Measured

Epoch decoding, 20-unit draws. Per recday the value is the mean over sessions and draws; per
mouse the mean over recdays; the table gives the mean of mice and the range over mice.

| set | mice | within session / chance | held-out session / chance |
|---|---|---|---|
| LEC, all regions | 5 | 2.87 (1.95-3.28) | 1.49 (1.41-1.59) |
| PFC | 6 | 2.33 (1.81-2.98) | 1.45 (1.15-1.63) |
| LEC ENTl-deep | 4 | 2.43 (1.99-2.88) | 1.72 (1.59-1.97) |
| LEC SUB/ProS | 3 | 3.20 (2.15-3.85) | 1.38 (1.27-1.54) |
| LEC CA1/HPF | 2 | 1.66 (1.00-2.33) | 1.32 (1.26-1.37) |
| LEC ENTl-sup (ah08 only) | 1 | 3.68 | 1.50 |
| LEC ENTm (ly07 only) | 1 | 3.04 | 1.59 |

Chance is 1/18 = 0.056 (0.056-0.059 where a session has fewer than 18 epochs).

Accuracy by true epoch, from the confusion matrices pooled over recdays and draws (F13):

| | first minute | second minute | minutes 2-17, mean |
|---|---|---|---|
| LEC within session | 0.312 | 0.201 | 0.143 |
| LEC held-out session | 0.247 | 0.122 | 0.071 |
| PFC within session | 0.275 | 0.206 | 0.109 |
| PFC held-out session | 0.263 | 0.123 | 0.070 |

Distance between epochs (F12; mouse means, 20-unit draws):

| | lag 1 min | lag 10 min | lag 10 - lag 1, per mouse |
|---|---|---|---|
| LEC | 0.066 | 0.084 | +0.010 to +0.034 (5 mice) |
| PFC | 0.129 | 0.164 | +0.016 to +0.066 (6 mice) |

Absolute distances are not comparable between datasets, because cosine distance depends on rates
and sparsity. Both rise by about 27% from lag 1 to lag 10.

Trajectory acceleration z at events (F14; mice averaged; 20-unit draws). The LEC events come from
25 recdays and the PFC events from 20, summed over sessions, one draw.

| event | dataset | lag -1 s | lag 0 | lag +1 s | events |
|---|---|---|---|---|---|
| synchrony onset (reference) | LEC / PFC | +1.27 / +1.71 | -1.43 / -1.44 | +0.22 / +0.14 | 6869 / 7571 |
| every reward | LEC / PFC | +1.08 / +2.73 | +1.96 / -1.28 | -1.29 / -3.57 | 12408 / 15632 |
| reward A | LEC / PFC | +0.44 / +1.26 | +0.85 / -0.72 | -0.61 / -1.69 | 3201 / 3998 |
| first A | LEC / PFC | +0.16 / +0.26 | +0.19 / -0.19 | -0.17 / -0.31 | 132 / 120 |

- **Reading the lags.** v(t) spans bins t and t + 1, so a change at the event bin shows as +z at
  lag -1 and -z at lag 0. The synchrony reference has exactly that shape.
- **PFC rewards** follow the same shape and then a larger deceleration at +1 s, a transient.
- **LEC rewards** peak one bin later, at lag 0: the jump comes about 1 s after the reward.
- **First A:** no shift in either dataset, with |z| < 0.35, but only about 125 events.

Threshold sensitivity (LEC, >= 5 vs >= 10 trials, 20-unit draws): within 3.05x vs 2.87x chance,
held-out 1.54x vs 1.49x.

All units, not size-matched, for context. Held-out accuracy per mouse is 0.089-0.129 in LEC (71-158
units per recday on average) and 0.059-0.114 in PFC (14-96).

### What was NOT established

- **Whether LEC and PFC differ.** Every matched-size difference above is smaller than the
  spread over mice. No contrast was tested, and with 5 vs 6 mice it would have little power.
- **Regions:**
  - ENTl-deep's held-out decoding (1.72x, 4 mice) is above SUB/ProS's (1.38x, 3 mice), but the
    contrast was not tested;
  - ENTl-sup and ENTm are single mice.
- **What the first-minute decoding is.**
  - It could be a response to being placed in the maze, a recording-start transient, or the
    pre-A behaviour; the mouse is already in the maze at recording start.
  - The per-neuron entry kernels (`EVENT_KERNELS.md`, 2026-10-12) say which units carry it.
- **Rate vs pattern.** The population rate was not regressed out. Kanter's soft normalisation
  keeps rate changes, so a slow rate drift contributes to the within-session decoding.
- **Rest.** Rest blocks and REM drift are M4.
- **The acceleration z** is a descriptive mean of per-session z scores. Its size depends on the
  number of events per session, so it is not a test.

### Commands

```
python code/check_mirror_parity.py                                           # must exit 0
# runner gate (8 CPUs, as the M1 reference)
sbatch --cpus-per-task=8 --mem=16G --time=0-00:30 --job-name=pd_rgate_lec sbatch_files/event_kernels.sbatch code/run_population_drift.py --gate --dataset lec
sbatch --cpus-per-task=8 --mem=16G --time=0-00:30 --job-name=pd_rgate_pfc sbatch_files/event_kernels.sbatch mFC_data/code/run_population_drift.py --gate --dataset pfc
# one job per recday, then merge
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner drift
bash sbatch_files/submit_event_kernels.sh --dataset pfc --runner drift
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner drift --min-trials 5
python code/run_population_drift.py --dataset lec --merge
python mFC_data/code/run_population_drift.py --dataset pfc --merge
python code/run_population_drift.py --dataset lec --merge --min-trials 5
# figures F12-F14 and the tables: code/event_kernels_M2_comparison.ipynb (RUN_MODE = 'load')
```

Outputs:

- shards: `data/population_drift/shards/<dataset>_mt<k>/`;
- merged: `data/population_drift/drift_<dataset>_mt<k>.pkl` with `_knn`, `_dist`, `_acc` and
  `_summary` CSVs;
- figures: `data/figures/population_drift/M2/`.
