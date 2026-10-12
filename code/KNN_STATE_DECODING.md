# kNN ABCD state decoding, trial by trial (TODO item 37)

## 2026-09-29 — first pass, both regions

**Outcome.** Controlling time in session does not reveal an abstract ABCD state code in either
region. A kNN decoder trained on other tasks is at chance, and matching or removing time does not
change that. This is the fourth independent negative on cross-task state decoding, after CCGP,
the 4-way leave-one-task-out decode and the SVM.

Within a task, states decode well: 0.62 in LEC, 0.63 in PFC, chance 0.25. That code is largely
preserved in another session of the same layout (X4). The pre-registered test of whether nearby
trials decode each other better (W2) is null. A descriptive lag curve does show 1-NN accuracy
falling slightly between trials 2 and 9+ apart, in every mouse of both regions, and centring
does not remove it. That is a candidate slowly drifting code, not an established one.

**Scope.** kNN weights every dimension equally, so a small shared subspace can be swamped by
remapped ones. A cross-task kNN null therefore adds little to CCGP's linear result; its value
here is the time controls.

### What was done

| file | role |
|---|---|
| `knn_state_decoding.py` | samples, feature variants, kNN, the analyses, nulls, aggregation, summary figure |
| `run_knn_state_decoding.py --dataset lec\|pfc` | runner; loads the data_dic as the LDA runner does |
| `knn_state_decoding_synthetics.py [--quick]` | 44 checks through the real builder |
| `knn_state_decoding.ipynb` | `RUN_MODE='load'`; tables and `knn_state_summary.svg` |
| `sbatch_files/knn_state_decoding.sbatch <lec\|pfc>` | 4 h / 8 CPU / 48 G; runs the parity check first |
| `ccgp_state_pairs.py` | the generator gained `trial_times='session'` and new cell kinds (below) |

All four Python files are mirrored into `mFC_data/code/` and listed in `check_mirror_parity.py`
`BYTE_PAIRS`.

**Samples.** From `ccgp_state_pairs.build_task_state_matrices`:
- one session per unique task;
- one sample per (trial, state), the mean over leg bins 15-75 of `Neurons_norm`, which drops the
  reward tone and the reward-end place fields;
- trial 0 (search) and trials with a non-finite leg dropped;
- z-scored per task.

Per sample it records `t_sec` (leg midpoint minus the first A), trial time T (trial midpoint),
leg duration, the towers, and whether the session is a repeat.

**Decoder.**
- kNN on 15 PCs, with the PCA fit once per fold on every non-test sample and shared by all pools.
- Euclidean distance.
- Own vote rule: a tie goes to the tied class with the nearest member. It is
  label-permutation-equivariant, which sklearn's is not (sklearn breaks ties to class A).

**Feature variants.** Every trial has one sample per state, so any function of trial time is
state-balanced, and all four variants are state-safe.

| variant | what it does |
|---|---|
| raw | per-task z-scored features |
| detrend | cubic B-spline in trial time (3 knots), per feature, fit on training trials |
| centred | minus the trial's own 4-state mean |
| project | detrend, then remove the span of the fitted trend from every sample |

`project` was added after synthetic (d); see Synthetic controls.

**Analyses.**

| id | what it compares | pool, k |
|---|---|---|
| W0 | ceiling | all other trials of the task, k 5 |
| W2 | near = trials k±1, far = 2 trials with \|lag\| ≥ 5, far_dm = far matched on leg duration within ±20%, rand = 2 random trials | k 3, 50 draws; tasks with ≥ 13 usable trials |
| lag curve | 1-NN from a single reference trial, by \|lag\| bin | 1 trial, k 1 |
| X0 | leave one task out | all trials of the other tasks, k 5 |
| X1 | near = 2 trials per training task within 90 s of the test trial; far = 2 per task ≥ 300 s away; rand = 2 per task | k 5, 50 draws |
| X3 | X0 raw, split by whether the test state's whole leg (both towers) or its destination is shared with the same state in a training task | — |
| X4 | the task's trials decoding the same layout in its repeat session | k 5 |

**Nulls.**
- Cross-task role permutation: an independent A-D relabelling per training task, 1000 draws. A
  guard raises if its mean is more than 4 SE from 0.25.
- Within-task: A-D permuted within each training trial, 200 draws, as a sanity check.

**Inference.**
- Mice are the replicates: sample → trial → task → recday → mouse.
- Primaries: one-sample t on mouse means, Holm over four contrasts. They were fixed before the
  data were seen: W2 near − rand raw, W2 near − rand centred, X0 raw − 0.25, X1 near − rand.

**Inclusion.**
- LEC: 24 recdays, 5 mice, 124 tasks (27 with a repeat session), median 20 usable trials per
  task. `ah08_20250624_20250625` was rejected by the CCGP builder, as it was in the CCGP analysis.
- PFC: 23 recdays, 7 mice, 136 tasks (37 with a repeat), median 27 trials. Two `me10` recdays
  were unusable.

### What was measured

**Primaries.** Mouse means with 95% t-CI, the one-sample t p, the Holm-adjusted p, and the number
of mice positive.

| contrast | LEC (5 mice) | PFC (7 mice) |
|---|---|---|
| W2 near − rand, raw | +0.003 [−0.008, +0.013], p 0.52, Holm 1.0, 4/5 | +0.003 [−0.020, +0.026], p 0.78, Holm 1.0, 4/7 |
| W2 near − rand, centred | −0.013 [−0.034, +0.008], p 0.15, Holm 0.61, 1/5 | −0.004 [−0.030, +0.021], p 0.68, Holm 1.0, 3/7 |
| X0 raw − 0.25 | −0.004 [−0.023, +0.015], p 0.60, Holm 1.0, 2/5 | −0.002 [−0.014, +0.009], p 0.63, Holm 1.0, 3/7 |
| X1 near − rand, raw | −0.014 [−0.062, +0.033], p 0.44, Holm 1.0, 3/5 | +0.005 [−0.000, +0.011], p 0.052, Holm 0.21, 6/7 |

None survives.

**Accuracy.** 4-way, chance 0.25, the mean over mouse means. The per-mouse values are in
`*_knn_state_summary.txt`.

| quantity | LEC | PFC |
|---|---|---|
| W0 within task, raw / centred / project | 0.623 / 0.666 / 0.635 | 0.630 / 0.646 / 0.628 |
| X0 across tasks, raw / detrend / centred / project | 0.246 / 0.249 / 0.259 / 0.259 | 0.248 / 0.247 / 0.244 / 0.246 |
| X1 near / rand / far | 0.236 / 0.251 / 0.237 | 0.255 / 0.249 / 0.249 |
| X4, same layout, other session | 0.537 | 0.482 |
| W2 near / rand / far / far_dm, raw | 0.473 / 0.470 / 0.465 / 0.468 | 0.454 / 0.451 / 0.446 / 0.458 |

W0 raw ranges over mice from 0.52 (ly05) to 0.80 (ah10) in LEC, and from 0.35 (me10) to 0.73
(ab03) in PFC. Per recday, X0 raw beats its role null at p < 0.05 in 2 of 22 LEC and 1 of 23 PFC recdays,
the nominal rate.

**Time removal does not rescue a cross-task code.** Relative to raw, X0 changes by at most
+0.013 in LEC (centred, project) and by −0.004 to −0.001 in PFC. On a pure shared-drift synthetic, centring
adds +0.05 with no code at all. The per-session-drift synthetic, where a real code is masked,
recovers by +0.35.

**The drift is real but harmless to within-task decoding.** Neighbour time-locality L measures
how much closer in time the k neighbours are than the pool average (0 = no time structure):

| | within raw | within centred | across raw | across centred |
|---|---|---|---|---|
| LEC | +0.349 | +0.023 | +0.246 | −0.007 |
| PFC | +0.239 | +0.080 | +0.127 | +0.019 |

So raw nearest neighbours are drawn from nearby times, and centring removes that. Yet W0
accuracy barely moves between raw and centred (+0.04 LEC). That is the signature of additive
drift orthogonal to the state code.

**Lag curve (descriptive, not a primary).** 1-NN accuracy from a single reference trial, mean over
mice:

| \|lag\| | 1 | 2 | 3 | 4-5 | 6-8 | 9+ | 9+ minus 2 |
|---|---|---|---|---|---|---|---|
| LEC raw | 0.427 | 0.439 | 0.445 | 0.429 | 0.416 | 0.419 | −0.020, 5/5 mice lower |
| LEC centred | 0.459 | 0.474 | 0.478 | 0.464 | 0.450 | 0.449 | −0.026, 5/5 |
| PFC raw | 0.414 | 0.434 | 0.427 | 0.422 | 0.411 | 0.386 | −0.048, 7/7 |
| PFC centred | 0.436 | 0.464 | 0.451 | 0.447 | 0.440 | 0.413 | −0.051, 7/7 |

Two things follow:
- Accuracy dips at lag 1. That fits the D(k−1)→A(k) boundary, where the two samples are only
  about 3 s apart.
- Accuracy then falls slowly with distance, and centring does not remove the fall.

The dip explains why W2, whose near pool is lag ±1, is null. The fall is what a slowly drifting
state code would produce, most clearly in PFC.

**Place (X3, pooled samples).** Cross-task accuracy is 0.351 when the test leg's two towers are
shared with the same state in a training task (n 639), against 0.242 otherwise. PFC gives 0.289
(n 741) against 0.250. What transfers is place, as in CCGP.

**Across sessions of the same layout.** X4 reaches 86% of the within-session ceiling in LEC
(0.537 / 0.623) and 77% in PFC. The state code is specific to the layout and mostly survives a
session gap.

### Synthetic controls

`knn_state_decoding_synthetics.py`: 44/44 pass, `--quick` in both trees (4.7 min) and full with
10 seeds. The recdays come from `make_synthetic_recday(..., trial_times='session')`, which has
cumulative real-time legs with trial 0 twice as slow and legs shrinking to 0.72×. New kinds:
`corridor`, `drift_axis`, `drift_session`, `drift_code`, `clock`, and `n_repeat` sessions.

The legacy generator path is untouched:
- max|diff| = 0 over 1008 arrays against HEAD, with its sha256 prefixes pinned in the synthetics;
- `run_synthetic_controls()` reproduces every row of `CCGP_STATE_PAIRS.md` §7.

| case | result |
|---|---|
| noise | every accuracy within 0.25 ± 0.05; the role null rejects at 7.5% raw / 10% detrend / 2.5% centred / 5% project at α 0.05 over 40 seeds (binomially consistent with 5%) |
| abstract code | W0 = X0 = 1.00 under all four variants; W2 near − rand 0.000; X4 1.00 |
| corridor (route place code) | W0 0.87, X0 0.18; leg-matched 0.49 vs 0.12 |
| drift_axis, shared, no state | L 0.81 raw → 0.02 centred |
| abstract + drift_session | X0 raw 0.58 → detrend +0.35, project +0.36 (rescue); W2 near − rand +0.31 raw → +0.01 centred |
| drift_code | W2 near − rand +0.10 raw and centred; X1 near − rand +0.05 |
| time-since-A clock | X0 0.92, every error an adjacent state (loop phase, not state) |
| tone at A | A recall 1.00 at trim 0, 0.235 at trim 15 |

**The synthetics caught a design error.** `drift_axis` has a shared drift and no state code. On
it, centring raised X0 by +0.060 and detrending by +0.042 over three seeds. Removing the
between-trial drift leaves its within-trial part, which runs A < B < C < D in every trial and
task. So the planned time controls can manufacture cross-task "state" decoding from drift alone.
`project` also removes the drift direction, and changes X0 by +0.003 on the same seeds. X0 raw
is unaffected (0.257).

**Reading rule:** a time-removed X0 above raw is only a rescue if it clearly exceeds the drift
artefact. On real data it does not.

### Not established

- **Whether the long-lag fall is a drifting code or composition.** Long-lag pairs over-sample
  the edge trials of a session: trial 1 and the last trials, where legs are shortest. The
  pairwise model with trial fixed effects that would separate the two
  (acc_kj ~ a_k + b_j + g(lag) + duration) was not fitted. A W2 with near = lag 2 was not run.
  It would be a post-hoc test.
- **Anything regional within LEC.** There is no anatomy split, and the `--region-group` subsample
  was not run.
- **Whether a linear decoder under the same time controls would differ.** The plan's conditional
  linear arm was not triggered, because X0 was not lifted.
- **First-exposure vs repeat sessions.** Of 124 LEC tasks, 113 used the first exposure; W3,
  position slopes split by exposure, was not computed.
- **The TODO's literal "nearest-neighbour regression".** A kNN regression of within-trial phase
  could reuse these neighbour sets. It was not run.

### Reproduce (maze_ephys)

    python code/knn_state_decoding_synthetics.py --quick                   # 44/44
    python code/check_mirror_parity.py                                     # exit 0
    sbatch --job-name=knn_state_lec sbatch_files/knn_state_decoding.sbatch lec   # 3668100, 8 min, EXIT=0
    sbatch --job-name=knn_state_pfc sbatch_files/knn_state_decoding.sbatch pfc   # 3667797, 19 min, EXIT=0
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=900 code/knn_state_decoding.ipynb   # 0 error outputs

Outputs: `data/glm_outputs/LEC_knn_state/knn_state{.pkl,_summary.txt,_primaries.csv,
_recday_means.csv}` and the PFC twins in `mFC_data/glm_outputs/PFC_knn_state/`. The first LEC
job, 3667796, crashed while printing the X3 line: empty frames turned the flags to object dtype.
It was fixed in the runner and rerun as 3668100, which reproduced the crashed job's pickle
exactly.
