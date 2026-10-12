# Lagged spatial maps (item 9) and the event-vector test (item 13)

## 2026-09-29 — first pass, both regions

**Outcome.** El-Gaby's held-out lagged-map statistic (Fig. 5f / ED 7h) was rebuilt per neuron
from his saved maps and reproduces his printed numbers exactly in both regions. All 5 LEC mice
show it. LEC lagged neurons are **not** event-vector cells:
- in the goal frame they are at chance;
- their anchors are not at reward ports more often than visits predict;
- their anchors are not at ports more often than those of zero-lag neurons.

What little cross-task consistency they have is in the place frame. The same holds in PFC.

### What was done

Everything reads El-Gaby's Figure5_Figure6 outputs (cell 76) from the run trees
`mFC_data/{lec_,}replication_run/data/Intermediate_objects/`. There is no refit and no data_dic.

| file | role |
|---|---|
| `lagged_spatial_maps.py` | loader; his cell 31 per neuron; the place/goal frame test; plots |
| `run_lagged_spatial_maps.py --region` | equivalence gate, then the per-neuron table and summary |
| `anchor_distribution.py --region` | anchors from V5 and from his cell 37; reward-place scores; nulls |
| `lagged_spatial_maps_synthetics.py`, `anchor_distribution_synthetics.py` | controls through the same doors |
| `lagged_spatial_maps.ipynb` | `RUN_MODE='load'`; figures only |

Outputs go to `data/figures/lagged_spatial/` (LEC) and `mFC_data/data/figures/lagged_spatial/`
(PFC). Each holds `lagged_spatial_maps.pkl`, `lagged_spatial_maps_units.csv`, both
`*_summary.txt` files, `anchor_distribution.pkl` and the SVGs. The code is one copy in `code/`
with `--region`, following `poisson_sweep_summary.py`, and is not mirrored.

Two handoff claims were wrong and are corrected in `docs/handoff/TODO_HANDOFF.md`:
- `Spatial_anchoring_Cross_val_corr_*` is the Fig. 5c single-anchor value (cell 47), not
  Fig. 5f. The Fig. 5f value (cell 31) was never saved.
- `Anchor_lag_threshold_*` is passed by 99-100% of neurons, because it compares a mean over lags
  with a max. It is not used here.

### Equivalence gate

`run_lagged_spatial_maps.py` rebuilds cell 31 from `Spatial_anchoring_Phase_shifted_maps_*`. It
uses his own sessions (`non_repeat_ses_maker`), pandas pairwise-complete correlations, his
exclusion sets and his modal-lag override. It then pools state-tuned neurons (`State_`) in his
recday order, as cell 34 does, and refuses to write anything unless n matches exactly and
|dt| < 1e-9.

| panel | LEC his / ours | PFC his / ours |
|---|---|---|
| all state-tuned | n 1426, t 45.5097814915 / same, dt 0.0 | n 738, t 20.2081181057 / same, dt 0.0 |
| non-zero lag, 30° | n 356, t 8.4211380118 / same, dt 0.0 | n 307, t 5.7220143420 / same, dt 0.0 |
| non-zero lag, 90° | n 179, t 5.8257878239 / same, dt 0.0 | n 164, t 3.8156752760 / same, dt 0.0 |

These t values treat neurons as replicates. His other printed Fig. 5 statistics, beside the paper
and his deposit, are now in both REPRODUCTION files (§"2026-09-29 — Figure5_Figure6 printed
statistics").

### Item 9 — what was measured

**Held-out r.** For each neuron, the mean over held-out tasks of the correlation between the
held-out task's 9-node map and each other task's map. The lag is chosen on the other tasks only.
The tables below average over the recdays of each mouse, then give one value per mouse.

LEC, state-tuned neurons:

| mouse | n state-tuned | r, all | r, 30° set (n) | r, 90° set (n) | fraction lagged 30° | fraction lagged 90° |
|---|---|---|---|---|---|---|
| ah08 | 290 | 0.337 | 0.082 (81) | 0.100 (46) | 0.276 | 0.155 |
| ah10 | 544 | 0.409 | 0.093 (122) | 0.064 (59) | 0.222 | 0.108 |
| ly05 | 149 | 0.352 | 0.023 (34) | 0.052 (19) | 0.294 | 0.163 |
| ly06 | 201 | 0.325 | 0.004 (42) | 0.035 (23) | 0.225 | 0.124 |
| ly07 | 242 | 0.346 | 0.115 (77) | 0.089 (32) | 0.309 | 0.130 |

5 of 5 mice have a positive mean in every panel; the one-sided binomial p is 0.031, the floor for
5 mice. In ly06 the 30° value is +0.004, which is essentially zero.

PFC, 7 mice. The all-tuned panel is positive in 7 of 7 (p 0.008). The **30° panel is positive in
only 4 of 7** (p 0.50): ah03 −0.031, me08 −0.014 and me10 −0.079, with 10, 16 and 3 neurons. The
90° panel is positive in 6 of 7 (p 0.063). The pooled t 5.72 on the 30° panel is therefore
carried by the neuron count, not by consistency across mice.

**Best lag.** The modal held-out best lag is 0, the current location, for 1042 of 1426 LEC
state-tuned neurons (73.1%) and 401 of 738 in PFC (54.3%). The remainder spread fairly evenly
over lags 1-11: 22-79 neurons per lag in LEC. `best_lag.svg`.

**LEC anatomy.** State-tuned neurons only; per-mouse effects with a mouse bootstrap CI; the
contrast is the primary ENTl-deep vs SUB/ProS, 4 mice, against a within-recday label shuffle.

| measure | ENTl-deep (5 mice, 538 units) | SUB/ProS (4 mice, 352) | contrast, p | rate-matched (398 units), p |
|---|---|---|---|---|
| held-out r, all | 0.372 [0.356, 0.388] | 0.393 [0.334, 0.434] | −0.027, p 0.26 | −0.009, p 0.83 |
| fraction lagged 30° | 0.245 [0.208, 0.283] | 0.228 [0.201, 0.255] | +0.036, p 0.24 | −0.061, p 0.34 |

No regional difference in either measure. The other groups (ENTl-sup, ENTm, CA1/HPF) are in the
summary. ENTl-sup and ENTm are dominated by a single mouse each (`anatomy_split.SINGLE_MOUSE_GROUPS`).

**Maps.** `maps_<recday>_n<k>.svg`, redrawn 2026-09-29 in the Fig. 5e layout on the 9 nodes the
statistic scores. Each map is normalised on its own, goal letters are drawn at the best lag, a
zoomed best-lag column sits on the right, and the cross-task r per lag is shown above.

The examples are the top three held-out r in the 90° set, one per recday, among units firing
0.5-20 Hz with at least 5 tasks, plus the best lag-0 neuron.

| region | recday | neuron | kind | rate | held-out r | lag |
|---|---|---|---|---|---|---|
| LEC | ly07_20250624_20250625 | 8 | lagged, ENTm | 13.2 Hz | 0.50 | 90° |
| LEC | ah08_20250616_20250617 | 50 | lagged, ENTl-sup | 0.6 Hz | 0.46 | 240° |
| LEC | ah08_20250624_20250625 | 37 | lagged, ENTl-sup | 1.6 Hz | 0.45 | 90° |
| LEC | ah08_20250618_20250619 | 56 | lag 0, ENTl-sup | 0.7 Hz | 0.98 | 0° |
| PFC | ah04_01122021_02122021 | 1 | lagged | 0.7 Hz | 0.49 | 180° |
| PFC | ab03_01092023_02092023 | 45 | lagged | 0.8 Hz | 0.44 | 240° |
| PFC | ah07_27082023_28082023 | 17 | lagged | 0.6 Hz | 0.43 | 90° |
| PFC | ab03_01092023_02092023 | 29 | lag 0 | 11.3 Hz | 0.93 | 0° |

The first draft picked by r alone and drew 5x5 node+edge grids. It chose a 48 Hz unit and a
0.3 Hz unit, and the edges, which his code drops as systematically biased, dominated the colour
scale. The lag-0 examples are clean place cells, the same node at every task. The lagged ones
are visibly noisier, as their held-out r of 0.43-0.50 implies.

### Item 13a — place frame vs goal frame (primary)

**What it tests.** Does a neuron's lagged map agree across tasks by place, or by goal letter? At
lag L and for tasks s and t, both frames correlate 4-vectors read from the same maps:
- place = corr(map_s at s's goals, map_t at s's goals), averaged with the same computed at t's
  goals;
- goal = corr(map_s at s's goals, map_t at t's goals), where the entries are aligned by letter
  A-D.

The score is held out as in cell 31, with the best lag chosen separately for each frame. The null
is the same statistic after an independent permutation of the 9 nodes in each task. Each task
pair shares a mean of 1.82 goal nodes in LEC (range 1.50-2.53 over recdays) and 1.72 in PFC.

| pool (state-tuned) | n | r_place | r_goal | place-only p<.05 | goal-only p<.05 | both | neither |
|---|---|---|---|---|---|---|---|
| LEC, lagged 30° (his) | 356 | +0.105 | −0.011 | 0.132 | 0.020 | 0.003 | 0.846 |
| LEC, not lagged | 1070 | +0.484 | −0.002 | 0.664 | 0.009 | 0.015 | 0.311 |
| LEC, lagged 90° (his) | 179 | +0.092 | −0.007 | 0.145 | 0.017 | 0.006 | 0.832 |
| LEC, V5 selected_eg | 524 | +0.264 | +0.007 | 0.361 | 0.021 | 0.015 | 0.603 |
| PFC, lagged 30° (his) | 307 | +0.071 | +0.004 | 0.124 | 0.026 | 0.000 | 0.850 |
| PFC, not lagged | 431 | +0.300 | −0.006 | 0.436 | 0.016 | 0.014 | 0.534 |
| PFC, V5 selected_eg | 283 | +0.136 | −0.022 | 0.194 | 0.021 | 0.000 | 0.784 |

The goal-only fraction is 0.9-2.6% in every pool. On pure noise the same test gives 3.0% (300
synthetic cells), so **no pool has an excess of reward-following cells**.

Contrast, lagged minus other, per-mouse mean, with the lagged label shuffled within recday (2000
permutations):

| region | lag set | r_goal | r_place |
|---|---|---|---|
| LEC | his 30° | −0.011, p 0.61 | −0.374, p 0.0005 |
| LEC | V5 | +0.022, p 0.17 | −0.167, p 0.0005 |
| PFC | his 30° | +0.061, p 0.13 | −0.251, p 0.0005 |
| PFC | V5 | +0.003, p 0.93 | −0.117, p 0.019 |

r_goal − r_place is higher for lagged neurons (LEC +0.36, p 0.0005). **That comes entirely from
their lower r_place**, not from a higher r_goal; do not quote it as evidence of goal coding. Per
mouse, the fraction of LEC lagged-30° neurons with goal-frame p < 0.05 is 0.000-0.051.

**Reading.** Most lagged neurons in both regions pass neither frame (85%). The 4-node test has
less power than the 9-node Fig. 5f statistic, so "neither" is weak evidence, not an absence of
structure. But nothing points at reward: r_goal is about 0 in every pool and every mouse.

### Item 13b — reward-place anchor bias (secondary)

**Anchors.**
- V5 (spec 12 LEC `repro_poisson_v5_past_20260908_192043`, spec 2 PFC
  `poisson_v5_past_20260907_153716`): per fold, the argmax of `cv_betas_collapsed` over
  (location, lag), with phase from `expected_anchor_phase`. The training tasks come from
  `used_sessions`.
- His cell-37 anchors: `best_node_phase_used`, cross-validated [phase, location] per split, pooled
  over state-tuned and anchored neurons. Lag class from `Anchor_trial_Anchor_lags_mean`.

The V5 pools reproduce the corrected panel sizes: LEC 558 / 165, PFC 298 / 73.

**Measure.** `reward_frac` is the fraction of training tasks in which the anchor node is a goal
and the anchor phase is early goal progress, i.e. at the port just after reward. `goal_frac` is
the same without the phase condition. Both are averaged over folds per neuron, then recday, then
mouse.

Contrast, lagged minus other, reward_frac, with the lagged label shuffled within recday:

| | V5 30° | V5 90° | his 30° | his 90° |
|---|---|---|---|---|
| LEC | −0.024, p 0.024 | −0.016, p 0.35 | +0.008, p 0.63 | +0.001, p 0.97 |
| PFC | +0.011, p 0.60 | +0.045, p 0.15 | +0.048, p 0.17 | +0.054, p 0.066 |

Against the visit null (anchor node redrawn in proportion to his `num_passes_all` at the anchor's
phase over the training sessions; 1000 draws), LEC lagged neurons:
- V5 anchors sit within 0.012 of the null in every mouse, above in 3 and below in 2 (e.g. ah10
  0.182 vs 0.171, ly06 0.182 vs 0.193).
- His anchors sit **below** the null in all 5 mice (0.134-0.247 against 0.192-0.290).

PFC, lagged 30°:
- V5 anchors are within 0.014 of the null in all 7 mice. 3 of 7 are above at p ≤ 0.05, but by at
  most 0.013 (ab03, ah04, ah07).
- His anchors are below the null in all 7. **Lagged neurons do not anchor at reward ports more than non-lagged neurons, or
more than visits predict, in either region or anchor source.** This agrees with the paper's ED 7d
("anchored to all possible goal-progress/place combinations"), whose code was not deposited.
Densities over phase × node: `anchor_hist_{v5,elgaby}.svg`.

### Synthetic controls

`lagged_spatial_maps_synthetics.py`: 16/16 pass, both `--quick` and full.
- A map consistent at lag 5 recovers lag 5 in 100% of folds, with r 0.85.
- A lag-0 map is excluded from both non-zero sets.
- Independent maps are rejected in 1 of 20 populations (full mode; 1 of 10 in `--quick`), mean
  r −0.003.
- Place-anchored cells: place frame rejects in 98%, goal frame in 0%.
- Reward-following cells: goal frame rejects in 100%, place frame in 6%.
- Noise: place 3.3%, goal 3.0% at α 0.05, so the frame test is slightly conservative.

**The synthetic caught a design error.** The first place-frame null permuted only the four goal
letters. That changed how node pairs recur across the two place terms, which lowered the null's
variance, and it passed 14.4% of pure-noise cells at α 0.05. The null is now the same statistic
on node-scrambled maps, which is exact on exchangeable noise.

`anchor_distribution_synthetics.py`: 6/6 pass, both `--quick` and full (30 cells per population;
full-mode figures in brackets below). The recdays are fitted through V5
with the spec-12 config; the tasks are fixed so that node 3 is a goal in every task and node 5 in
none.
- Planted phase-0 anchors at node 3 are recovered in 100% of folds, reward_frac 1.0.
- Anchors at node 5 are recovered in 100% of folds, goal_frac 0.
- A population drawn by visits is accepted: 0.111 vs null 0.131, p 0.84 [0.172 vs 0.157, p 0.15].
- A 60% reward-anchored population is rejected: 0.653 vs 0.350, p 0.002 [0.650 vs 0.369, p < 0.001].

The recdays are built from `make_behaviour` and `anchored_rate` inside this file, so the mirrored
`elasticnet_v5_synthetics.py` is unchanged; the plan's `tasks=None` edit was not needed.

### Not established

- **Why most "state-tuned" LEC neurons are best at lag 0 and place-consistent.** This fits state
  tuning that is inherited from place, since state preferences remap across tasks, but that was not
  tested here. Item 11 asks it directly.
- **Whether the lagged neurons that pass neither frame have structure the 4-node test cannot
  see.** The 9-node Fig. 5f statistic is more powerful; a 9-node goal frame is not defined.
- **Anything per mouse in PFC's 30° panel.** It is positive in only 4 of 7 mice.
- **Any regional contrast beyond "no difference detected".** ENTl-deep vs SUB/ProS rests on 4
  mice.
- **The two definitions of "lagged" agree only in part.** In LEC, his cell-31 set (356) and V5's
  (524) share 205 neurons. Both were run; neither changes the reading.
- **The `spatial_ratemaps.LOC_TO_GRID` edge bug.** It places edges 12-21 on the wrong axes and is
  imported by `rave_visualiser.py`. It is flagged, not fixed (user, 2026-09-29). The new code uses
  his `Edge_grid_coord2` instead.

### 2026-09-29 — PDFs of every state-tuned neuron

There is one page per state-tuned neuron in the Fig. 5e layout above, plus a cover page. Pages
are ordered lagged 90°, then lagged 30° only, then the rest, each by held-out r. The page header
carries the numbers from the per-neuron table: rate, the three held-out r values, modal best lag,
the place and goal frame r with their p, and the V5 selection.

| region | file | pages | size | job |
|---|---|---|---|---|
| LEC | `data/figures/lagged_spatial/lagged_maps_state_tuned_lec.pdf` | 1427 | 55 MB | 3667763, 40 min, EXIT=0 |
| PFC | `mFC_data/data/figures/lagged_spatial/lagged_maps_state_tuned_pfc.pdf` | 739 | 29 MB | 3667764, 19 min, EXIT=0 |

Rebuild with `sbatch sbatch_files/lagged_maps_pdf.sbatch <lec|pfc>`, which runs
`code/plot_lagged_maps_pdf.py`. It measures 1.3 s and 40 KB per page.

### Reproduce (maze_ephys, from `code/`)

    python lagged_spatial_maps_synthetics.py --quick         # 16/16, ~20 s
    python anchor_distribution_synthetics.py --quick         # 6/6, ~6.5 min (V5 Poisson fits)
    python run_lagged_spatial_maps.py --region lec           # gate, then ~8 min
    python run_lagged_spatial_maps.py --region pfc           # ~4 min
    python anchor_distribution.py --region lec               # ~5 min
    python anchor_distribution.py --region pfc
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=900 lagged_spatial_maps.ipynb   # 0 error outputs

`run_lagged_spatial_maps.py --gate-only` checks equivalence alone in about 2 min.
`--summary-only` rewrites the summary from the saved pickle. None of these loads a data_dic or
fits a real recday, so they run interactively.
