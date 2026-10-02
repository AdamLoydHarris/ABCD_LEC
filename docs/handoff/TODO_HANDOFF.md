# TODO list — state and handoff

**Written 2026-09-29.** State of every item in `TODO.txt` (repo root; it moved there from
`mFC_data/`), with where the results are and what to do next. Every number was re-measured on disk
on this date. Numbers carry their semantics; mice, not recdays, are the replicates.

Read first: `.claude/CLAUDE.md` (working rules), `docs/handoff/README.md` (anatomy split, the
confounds), and for anything El-Gaby, `mFC_data/mFC_schema-main_unblocked_lec/RUN.md`.

---

## 1. Live cluster jobs

| job | what | state on 2026-09-29 08:40 | where it lands |
|---|---|---|---|
| 3666091-94 | ABCDE July kilosort resume, ah08 / ah10 / ly05 / ly06 | running ~16 h on GPU, 5-day walltime | `data/preprocessed/ephys/<mouse>/2025-07-14_..._2025-07-19_preprocessed/kilosort_output/` |
| 3666095 | same, ly07 | pending | same |
| 3652267 | LEC Poisson sweep spec 20, lasso α=0.001 | pending, 3-day walltime | `data/figures/alpha0.001_repro_poisson_l1_v5_past_*` |
| 3652268 | LEC Poisson sweep spec 21, elastic net 0.5 at α=0.003 | pending, 3-day walltime | `data/figures/alpha0.003_repro_poisson_en0.5_v5_past_*` |

Check with `squeue -u $USER`. Kilosort logs: `code/preprocessing/slurm_outputs/resumeks-<jobid>.out`.
Sweep logs: `code/slurm_v5/logs/v5_lec_poi_past_s<N>_<jobid>.out`. When 20 and 21 land, run
`python code/poisson_sweep_summary.py --region lec` (maze_ephys python).

---

## 2. Item by item

| # | item | status |
|---|---|---|
| 3 | El-Gaby Poisson, prospective vs retrospective, both regions | **done** |
| 5 | Regularise the Poisson regression, L1:L2 sweep | PFC **done**; LEC 2 of 4 points |
| 7 | Sleep / offline structure, PFC rerun + LEC | both **ran**; LEC result not yet written up |
| 9 | Lagged spatial rate maps | matrices **saved**; plotting and held-out stats not done |
| 11 | Lagged spatial neurons: time or task space | not started |
| 13 | Lagged spatial neurons: event-vector / anchor distribution | not started |
| 15 | Integrated-experience regressor (pokes) in the GLM | not started |
| 17 | Time-in-session vs goal-progress LDA on PFC | **done**, extended to both regions |
| 19 | Skills + CLAUDE.md, style interview | **done** |
| 21 | Tidy codebase | partly done |
| 23 | Integrate the mEC data | in progress, another session |
| 25 | mEC histology | not started |
| 27, 29 | RNN models | not started |
| 31 | Message Anna-Lisa | done |
| 33 | Matt Nour article | yours |
| 35 | Nature Neuro review | done |
| 37 | State decoding with nearest-neighbour regression, trial by trial | not started, new item |
| — | ABCDE July kilosort (not on the list) | in progress |

### 3 — El-Gaby prospective lags — done

His `Figure5_Regression` run as a declared variant (`VARIANT-PRO-01..10`, both
`_preflight/20_apply_edits.py`): cell 15's bump model on the time-reversed session. Sign-checked:
his prospective regressors equal `elasticnet_regression_v5.generate_regressors_raw(...,'future')`
to max|diff| = 0 on all 6 sessions of one recday per region.

Poisson α=1, State_95 pool, his cell-26 semantics:

| panel | LEC retro | LEC prospective | PFC retro | PFC prospective |
|---|---|---|---|---|
| all state-tuned, n (t) | 1235 (35.4) | 1154 (38.3) | 581 (17.1) | 537 (15.1) |
| non-zero lag, 30° excluded | 541 (9.9) | 367 (5.5) | 296 (6.8) | 253 (3.4) |
| non-zero lag, 90° excluded | 153 (2.15) | 98 (1.36) | 79 (2.67) | 50 (−0.26) |

Retrospective carries the non-zero-lag effect in both regions. V5 spec 13 agrees (LEC 30° panel 558
past vs 359 future). Write-ups: `mFC_data/mFC_schema-main_unblocked_lec/LEC_REPRODUCTION.md` and
`mFC_data/mFC_schema-main_unblocked/REPRODUCTION.md`, both dated 2026-09-15. Stats:
`mFC_data/{lec_,}replication_run/logs/figure5_stats.{json,csv}`.

### 5 — Poisson regularisation sweep — PFC done, LEC half done

sklearn's `PoissonRegressor` is L2-only, so L1 goes through **glum** (installed in maze_ephys).
`RegressionConfigV5(poisson_solver='glum', poisson_l1_ratio=..., poisson_positive=...)`; synthetic
controls 16-18 in `elasticnet_v5_synthetics.py`. Queue specs: PFC 14-17 on spec 2's config, LEC
18-21 on spec 12's, same grid.

PFC past, his semantics, linear readout:

| penalty | all state-tuned n (t) | 30° n (t) | 90° n (t) | all-zero fits | median betas of 324 |
|---|---|---|---|---|---|
| L2 α=1, spec 2 reference | 140 (12.09) | 86 (3.34) | 19 (1.24) | 0.1 % | 108 |
| L1 α=0.01 | 357 (11.94) | 158 (2.44) | 39 (0.85) | 61.0 % | 0 |
| L1 α=0.003 | 550 (17.15) | 231 (4.15) | 48 (−0.96) | 23.8 % | 11 |
| L1 α=0.001 | 563 (15.58) | 206 (5.36) | 42 (0.78) | 6.0 % | 37 |
| EN α=0.003, l1=0.5 | 562 (16.56) | 217 (5.15) | 45 (1.83) | 11.0 % | 28 |
| paper ED 8d | 489 (10.7) | 346 (4.74) | 229 (2.81) | | |

LEC past so far:

| penalty | all state-tuned n (t) | 30° n (t) | 90° n (t) | all-zero fits | median betas |
|---|---|---|---|---|---|
| L2 α=1, spec 12 reference | 1256 (35.12) | 558 (9.64) | 165 (2.48) | 0.0 % | 108 |
| L1 α=0.01 | 711 (22.38) | 247 (4.61) | 94 (1.62) | 68.5 % | 0 |
| L1 α=0.003 | 1222 (33.42) | 418 (5.82) | 111 (1.60) | 31.7 % | 5 |

PFC reading: milder L1 brackets the paper on the two panels that ever reproduced, so the α=1
reference was over-shrunk; α=0.01 is uninterpretable because most fits are empty; the 90° panel fails
at every point. **LEC runs the opposite way** — L1 shrinks n below the L2 reference instead of
growing it, and the LEC L2 90° panel already passes (t 2.48). Do not interpret LEC until 20 and 21
land. Tables and figures: `{mFC_data/,}data/figures/poisson_sweep/poisson_sweep_summary.{csv,svg}`.
Write-up: `code/ELASTICNET_V5.md` §"2026-09-14 — L1 / elastic-net Poisson".

**Next:** when 20/21 land, run the summary, write the LEC reading into `ELASTICNET_V5.md`.

### 7 — Sleep / offline structure — both regions ran

El-Gaby's Figure7 was blocked on both datasets. Unblocked with
`_preflight/12_make_figure7_inputs.py` (both copies), which builds a Figure7-only symlink overlay
`<run>/data/Intermediate_objects_f7/` holding 18-column MetaData sheets (the executed schema is this
lab's own format), ISO dates, repaired Excel-mangled `Ephys` tokens, and for LEC the
`binned_FR_dic_<rd>_<i>` sleep shards and real `HH-MM-SS` session arrays. LEC sleep arrays already
existed (`data/processed_data/neuron_raw_mingyutest/Neuron_raw_<rd>_sb_<j>.npy`); no labelling was
needed. Edits F7-01..04, LEC-05, LEC-06, LEC-08.

Headline, cell 89: one correlation per sleep session between forward distance and 250 ms sleep
co-activity, for co-anchored pairs; close = 0-180°, far = 180-360°.

| | LEC | PFC, all 21 recdays | PFC, his 12-recday state | his stored notebook | paper Fig. 7e |
|---|---|---|---|---|---|
| sleep sessions | 100 | 37 | 30 | 36 | 59 |
| close vs far, paired t (p) | **−7.81 (6e-12)** | −2.81 (0.008) | −2.60 (0.015) | −5.16 (1e-5) | −5.55 (7e-7) |
| close arm r < 0, t | **−5.17** | −1.15 | −0.70 | −3.59 | −5.16 |
| far arm r > 0, t | **+6.98** | +3.70 | +4.29 | +3.00 | +4.16 |

- **LEC reproduces the V, strongly.** Forward-distance curve (cell 86): 0.082 at 0-45°, minimum 0.019
  at 180-225°, 0.079 at 315-360°. Only 1 of 25 recdays dropped for having fewer than 10 pairs.
- **PFC does not.** The far arm reproduces; the close arm does not, and the curve is a sawtooth in
  both runs. The sleep shards are his; the pair sets and forward angles come from our runs of his
  Figure3 and Figure5_Figure6, since he deposited none of those intermediates. That is where the
  difference enters. Full diagnosis: PFC `REPRODUCTION.md` §Figure7.
- Not established: anything regional. LEC is the unsplit probe mixture, 5 mice; LEC-08 dropped
  stage `post6` because it had no co-anchored pairs in any recday.

*Correction, 2026-09-29 (later).* Two bullets above overstate. The curve is high at both ends but
not a clean V: it dips to 0.032 at 45-90° and recovers to 0.055 at 90-135°. The dropped recday,
ly05_20250620_20250623, had **zero** co-anchored pairs, not fewer than 10; the 10-pair rule drops
recday x stage units instead, 44 of 144. The 100 "sleep sessions" are those recday x stage units.
Item (1) below is done: `LEC_REPRODUCTION.md` §"Results, 2026-09-18".

**Next:** (1) write the LEC result into `LEC_REPRODUCTION.md` §Figure7, which still says pending;
(2) your refinements from the TODO line — restrict to deep ENTl, and define anchors from the
regression rather than his single-anchor-and-lag method. Executed notebook:
`mFC_data/lec_replication_run/executed/Figure7.ipynb`. Figures:
`mFC_data/lec_replication_run/Output_folder/ephys/`.

### 9 — Lagged spatial rate maps — matrices saved

LEC Figure5_Figure6 completed 2026-09-18 via a checkpoint split (see §4). Cell 76 saved everything:

| file, per recday, in `mFC_data/lec_replication_run/data/Intermediate_objects/` | content |
|---|---|
| `Spatial_anchoring_Phase_shifted_maps_<rd>.npy` | neurons × sessions × 12 lags × 21 locations, e.g. (151, 8, 12, 21) |
| `Spatial_anchoring_Phase_shifted_node_matrices_<rd>.npy`, `..._node_edge_matrices_` | the plottable 3×3 / 5×5 forms, his cell 25 |
| `Spatial_anchoring_Cross_val_corr_<rd>.npy` | per-neuron held-out spatial correlation, his cell 31 |
| `Anchor_lag_<rd>.npy`, `Anchor_lag_threshold_<rd>.npy` | lagged-map correlation per neuron and its shuffle threshold |

PFC equivalents are in `mFC_data/replication_run/data/Intermediate_objects/`.
**Next:** plot the maps and extract the held-out prediction statistics; nothing needs re-running.

*Correction, 2026-09-29 (later).* `Spatial_anchoring_Cross_val_corr` is **not** the held-out
lagged-map correlation. It is cell 47's Fig. 5c single-anchor statistic: a 4-vector correlation
at the cross-validated anchor. The Fig. 5f held-out lagged-map statistic is computed in cell 31
and never saved, so it is re-derived from `Phase_shifted_maps` in `code/lagged_spatial_maps.py`.
Both executed notebooks printed its population result in cell 34. `Anchor_lag_threshold` is
passed by 99-100% of neurons, because it compares a mean over lags against a max; do not use it.

### 11 and 13 — lagged spatial neurons: time vs task, anchor distribution — not started

*2026-09-29 (later): item 9 and item 13 are done; write-up `code/LAGGED_SPATIAL.md`.*
- **Item 9.** His cell 31 is rebuilt per neuron and reproduces cell 34 exactly (dt 0.0 on all six
  panels). All 5 LEC mice are positive, but only 4 of 7 PFC mice on the 30° panel.
- **Item 13.** LEC lagged neurons are not event-vector cells. In the goal frame they are at
  chance: goal-only 2.0% against a noise rate of 3.0%. Their anchors are not at reward ports
  beyond visits, or beyond zero-lag neurons, in either anchor source.
- **Item 11 is still open.** Its inputs, `glm_outputs/LEC_time_vs_progress/{ood_temporal,alpha}.pkl`,
  are dated 2026-06-11/12. That is before the 1 Sep HD fix and the 6 Sep ah10 s5 salvage, so check
  what produced them before joining.

Both are joins on files that exist; no refits.
- **11:** merge the V5 unit table (`elasticnet_regression_v5.build_unit_table`, key `(recday, order)`)
  with `glm_outputs/LEC_time_vs_progress/ood_temporal.pkl` and `alpha.pkl`, then compare `delta_ood`
  and `alpha_star` for non-zero-lag state-tuned neurons against the rest, per region, mice as
  replicates. PFC twin: `mFC_data/glm_outputs/PFC_time_vs_progress/`.
- **13:** V5 npz files store `peak_lags` and `cv_betas_collapsed` (neurons × folds × 9 locations × 12
  lags). Anchor location is the argmax over locations at the peak lag; anchor phase is analytic via
  `expected_anchor_phase`. Test whether anchors concentrate on reward ports against a within-recday
  shuffle.

### 15 — integrated-experience regressor — not started

No regressor registry in `glm_analysis_v3.py`; a new regressor is about eight coordinated edits.
Poke tables are ready: `data/processed_data/trialtimes_raw_mingyutest/pokes_*.npy`, columns
entry_bin / exit_bin / port / rewarded / state. A cumulative count must be computed per session,
before the node filter, and declared an ordered regressor. **Warning:** `glm_analysis_v3.py` and
`glm_v3_synthetics.py` have uncommitted changes in both trees from another session — read those
before editing.

### 17 — LDA — done and extended

Redesigned on your suggestion: one joint LDA on the trial × goal-progress conjunction (30 classes,
the four states of a trial as four samples per class), read out on held-out sessions against a
within-session circular-shift null. Write-up: `code/LDA_REWARD_GOALPROGRESS.md`.

| held-out, seconds r | LEC | PFC |
|---|---|---|
| all units, 75 % PCA | 0.54 | 0.26 |
| 15 PCs for every recday | 0.52 | 0.23 |
| 50 units, 10 draws | 0.39 | 0.21 |

Goal-progress balanced accuracy runs the other way, LEC 0.57 against PFC 0.65, chance 0.333.
Within recdays, 25 ENTl units against 25 SUB/ProS + CA1 units, 12 paired recdays in 3 mice:
seconds r 0.29 vs 0.12, ENTl better in 10 of 12. **Not rate-matched yet** — SUB/ProS fires about 3×
ENTl-deep; `anatomy_split.rate_match` is the next check.

### 19 — skills and CLAUDE.md — done

`.claude/CLAUDE.md` plus three skills: `new-engine-version`, `slurm-fit-launch`, `synthetic-gate`.
Your interview answers are recorded there and in memory.

### 21 — tidy — partly done

Done: 116 `.pyc` untracked, outputs and El-Gaby run trees ignored, duplicate copies removed.
Standing rule: `code/` stays flat until the analyses are finalised for submission.

### 23 and 25 — mEC — in progress elsewhere

Another session owns this. Read `docs/handoff/MEC_INTEGRATION.md` (integration) and
`docs/handoff/MEC_REGISTRATION_PLAN.md` (histology) before touching it.

### 27 and 29 — RNNs — not started

Anna-Lisa's repo, uploaded at `goal_planning_GPS-main/`, already trains ABCD RNNs with and without a
goal input (`src/goal_planning/models/vanilla_rnn.py`, `two_region_rnn.py`), which overlaps item 29.

### 37 — nearest-neighbour state decoding — not started

*2026-09-29 (later): done; write-up `code/KNN_STATE_DECODING.md`.*
- **Across tasks.** A kNN decoder trained on other tasks is at chance: X0 raw is 0.246 in LEC and
  0.248 in PFC (chance 0.25), p 0.60 and 0.63 over mouse means. Matching or removing time does
  not change that. This is the fourth negative, after CCGP, the 4-way decode and the SVM.
- **Within a task**, states decode at 0.62 (LEC) and 0.63 (PFC). The pre-registered
  near − rand contrast is null.
- **Candidate.** A descriptive lag curve falls in every mouse from lag 2 to lag 9+ (LEC −0.02,
  PFC −0.05), and centring does not remove it. That could be a slowly drifting code;
  trial-composition controls are still needed.
- **Synthetic finding.** Centring and detrending manufacture cross-task "state" decoding from
  pure drift (+0.05), so the `project` variant was added.

New item. The LDA runner's per-sample `t_sec` (seconds since the session's first A onset) is the
natural covariate to regress out trial by trial.

### ABCDE July kilosort — in progress (not on the list)

Five six-day blocks, `2025-07-14` to `2025-07-19`, 53-55 recordings each. The first sort failed on
all five at `kilosort/swarmsplitter.py:96`, where `np.NaN` no longer exists in numpy 2. maze_ephys has
numpy 2.0.2, which kilosort 4.0.18, spikeinterface 0.101.2 and neo 0.13.3 all forbid. Fixed
elsewhere: env `maze_ephys_np126` pins numpy 1.26.4, and `code/resume_kilosort_from_binary.py` resumes
from the existing destriped binary. Details: `MEC_INTEGRATION.md` §8.6. **Next:** when the jobs land,
check `QC_single_units.npy` counts against the June blocks, then run `extract_neuron_raw.py`.

---

## 3. Gotchas that cost time this fortnight

- **`0 failed` under `--allow-errors` is not evidence.** PFC Figure7 reported it with 18 error
  outputs in 23 cells. Count error outputs in the executed notebook.
- **The Figure7 overlay is a snapshot.** Files an upstream notebook writes later are invisible to it;
  Figure7 died on the missing `Anchor_trial_*` files. Re-run `12_make_figure7_inputs.py` immediately
  before submitting Figure7. It is idempotent.
- **LEC Figure5_Figure6 cell 17 takes 29.9 h and saves nothing.** The runner now checkpoints the
  kernel after cell 17: run `--only 0-17`, then `--skip 9,17`. A later failure costs an hour.
- **Size walltimes to the job.** A 24 h request for a 2 h job waited 20 h on the degraded partition.
  LEC fits run about 2.1× PFC.
- **El-Gaby cell 38 names figures from `addition` alone**, so a variant overwrites the base run's
  SVGs. Use `line_subs` in the patcher for statements that continue with a backslash.
- **`ah08_20250624_20250625` session 10 in `data_dic_lec.pkl` is a sleep-box recording**, paired as a
  task session by the old positional pipeline. Zero trials, so harmless where it sits; drop it when
  the pickle is next rebuilt.
- **maze_ephys numpy 2.0.2 breaks the sorting stack.** Sort in `maze_ephys_np126`.

---

## 4. Code and commits

All work from this session is committed, last in `6035467`. The 25 uncommitted files are from
other sessions: tracking QC, mEC integration, the kilosort resume, and `glm_analysis_v3.py` in both
trees. Leave them to their owners.

New this fortnight, worth knowing:

| file | purpose |
|---|---|
| `mFC_data/mFC_schema-main_unblocked*/_preflight/12_make_figure7_inputs.py` | Figure7 overlay, sheets, LEC sleep shards, gates |
| `mFC_data/mFC_schema-main_unblocked_lec/_preflight/40_run_notebook.py` | cell-17 checkpoint gates for Figure5_Figure6 |
| `code/run_lda_reward_progress.py`, `sbatch_files/lda_reward_progress.sbatch` | LDA runner; flags `--n-pcs`, `--subsample-neurons`, `--region-group` |
| `code/poisson_sweep_summary.py` | sweep table and figure, `--region pfc|lec` |
| `code/check_mirror_parity.py` | now also guards V5, its synthetics and the LDA module |
