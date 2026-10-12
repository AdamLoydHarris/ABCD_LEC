# Population drift in LEC vs PFC (Kanter 2025), on the event-kernel session format

Kanter et al. 2025 population measures: how far the population state moves between 1-min
epochs, whether an epoch can be decoded within its own session (locked clock plus drift) or only
from other sessions (locked clock alone), and whether the population trajectory jumps at task
events. Plan, decisions and code map: `docs/handoff/EVENT_KERNELS_HANDOFF.md`. Per-neuron kernels:
`code/EVENT_KERNELS.md`.

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
