# Time in session x goal progress: one joint LDA, read out on held-out sessions

`lda_reward_goalprogress.py` · `run_lda_reward_progress.py` · `LEC_lda_analyses.ipynb` /
`PFC_lda_analyses.ipynb` (cell 30 onwards) · pickles `data/glm_outputs/LEC_lda/reward_progress.pkl`,
`mFC_data/glm_outputs/PFC_lda/reward_progress.pkl` · figure `data/figures/lda_reward_progress/`.
Append-only log; the module is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`).

## Samples

One population vector per (trial, state, progress bin): `Neurons_norm[:, trial, bins].mean` over
30 of the 360 normalised bins (3 progress bins x 4 states x 90 bins). First `min_trials = 10`
trials of every session that has at least 10; a recday needs at least 3 such sessions
(`filter_sessions_by_trials`) and at least 2 PCs. Vectors are z-scored per neuron across the
recday's sessions, then PCA to 75 % variance (`apply_pca_trialbins`). Each sample carries
`trial_id`, `state_id`, `y_progress`, `y_reward` (= trial*4 + state) and `t_sec`, its centre time in
seconds since the session's first A onset (`Trial_times` are 25 ms bin indices).

## The joint LDA

`conjunction='trial_progress'` (default since 2026-09-14): class label `f'{trial:02d}_{prog}'`,
10 x 3 = 30 classes, the four states of a trial are four samples of one class per session.
The legacy `conjunction='reward_progress'` (`f'{reward:02d}_{prog}'`, up to 120 classes, one
sample per class per session) is kept and reproduces the pre-2026-09-14 outputs exactly
(checked on ab03_01092023_02092023: identical `y_conjunction`, `X_pca`, `n_pcs`, `n_lds`).
The all-sessions fit gives the LD1 / LD2 scatter (colour = trial index / progress).

## Readouts, all leave-one-session-out (`run_joint_lda_readout`)

Per fold the LDA is fit on the training sessions; on the held-out session `predict` gives a
(trial, progress) pair per sample and `transform` gives LD scores.

| score | what | chance / reference |
|---|---|---|
| `key_mae`, `key_rmse`, `key_rho`, `key_acc` | predicted vs true trial index (0..9) | null MAE ~ 3.3 |
| `sec_mae`, `sec_r` | `Ridge(alpha=1)` from training LD scores to `t_sec`, applied held-out | null |
| `prog_acc` (balanced), `prog_rho` | predicted vs true progress bin, early < middle < late | 1/3 |
| `joint_acc` | exact conjunction class | 1/30 |
| `ld1_r_key`, `ld1_r_sec`, `ld2_rho` | \|corr\| of held-out LD1 with time, LD2 with progress | null |

Null: the sample order within each session is circularly shifted (labels and `t_sec` together,
features fixed) and everything is refit, 1000 times. This keeps the within-session
autocorrelation that a label permutation would destroy. `p` = fraction of null at least as good
(<= for MAE / RMSE, >= otherwise).

Superseded: the two separate decoding LDAs (`run_reward_progress_decoding`: 3-class progress,
~40-class reward number scored by exact-class accuracy) are kept behind `--legacy-decoding`.
Their 2026-09-14 numbers (LOGO balanced accuracy, label-permutation null): goal progress LEC 0.61
(21/22 recdays p<0.05, 5 mice) vs PFC 0.66 (24/24, 7 mice), chance 0.333; reward number LEC 0.044
(14/22) vs PFC 0.037 (12/24), chance 0.025 -- exact-class scoring, so a near miss counted as a
miss, which is why the readout above replaces it. Pickles archived as
`reward_progress_legacy_rewardxprogress_20260914.{pkl,csv}` next to the current ones.

## Commands

```
sbatch --job-name=lda_rp_pfc --time=0-06:00 sbatch_files/lda_reward_progress.sbatch pfc
sbatch --job-name=lda_rp_lec --time=0-12:00 sbatch_files/lda_reward_progress.sbatch lec
# smoke test / legacy: ... pfc --recdays ab03_01092023_02092023 --conjunction reward_progress --n-shuffles 5 --out /tmp/x.pkl
```
Notebooks run cell 30 with `RUN_MODE = 'load'`; the last cell draws `plot_lec_vs_pfc_decoding`.

## Synthetic check (2026-09-14)

6 sessions x 10 trials x 4 states x 3 bins on 20 PCs, time planted on PC0 (0.6 per trial),
progress on PC1 (1.5 per bin), 100 shuffles: trial MAE 1.39 (null 3.29 +/- 0.22, p < 0.01),
seconds r 0.86 (null -0.03), progress accuracy 0.65 (null 0.34, p = 0.01), joint accuracy 0.13
(null 0.033). The same features with labels rolled within session: trial MAE 3.22 (p = 0.29),
progress accuracy 0.28 (p = 0.85), seconds r 0.00 (p = 0.38).

## Results — 2026-09-14 (trial_progress, min_trials 10, 1000 circular-shift refits, LOGO by session)

Per-recday scores averaged over recdays; "sig" = recdays with p < 0.05. LEC 22 recdays / 5 mice
(skipped ah08_20250624, ly05_20250616, ly05_20250618: < 3 sessions with >= 10 trials), PFC 24
recdays / 7 mice (skipped me10_20122021: 1 PC). Median neurons 90 (LEC) vs 50 (PFC); median PCs
29 vs 16; median sessions 5 vs 6.

| held-out score | LEC | LEC null | sig | PFC | PFC null | sig |
|---|---|---|---|---|---|---|
| goal progress, balanced acc (chance 0.333) | 0.573 | 0.334 | 19/22 | 0.647 | 0.332 | 24/24 |
| time: trial MAE (trials; null ~ 3.3) | 2.36 | 3.30 | 19/22 | 2.71 | 3.30 | 21/24 |
| time: seconds MAE (s) | 154 | 217 | 18/22 | 150 | 169 | 14/24 |
| time: seconds Pearson r | 0.54 | -0.07 | 21/22 | 0.26 | -0.10 | 22/24 |
| joint class acc (chance 0.033) | 0.100 | 0.033 | 21/22 | 0.093 | 0.033 | 24/24 |
| held-out LD1 vs seconds, \|r\| | 0.44 | 0.12 | 14/22 | 0.09 | 0.07 | 2/24 |
| held-out LD2 vs progress, \|rho\| | 0.32 | 0.06 | 12/22 | 0.12 | 0.08 | 5/24 |

Per-mouse (mice are the replicates): LEC seconds r ah08 0.59, ah10 0.71, ly05 0.46, ly06 0.36,
ly07 0.53; PFC ab03 0.66, ah07 0.52, ah04 0.29, me08 0.17, me11 0.12, me10 0.02, ah03 -0.01.
LEC progress accuracy: ly05 is the outlier at 0.43 (others 0.55-0.64).

Reading:
- Both variables are read out of the same held-out LD space in both regions. Goal progress is
  the stronger signal in PFC; time in session is the stronger signal in LEC (seconds r 0.54 vs
  0.26; trial MAE 2.36 vs 2.71 against the same null of 3.3).
- The axis-level picture "LD1 = time, LD2 = progress" from the all-sessions scatter holds out of
  sample in LEC (LD1 |r| 0.44, 14/22; LD2 |rho| 0.32, 12/22) and **not** in PFC (0.09, 2/24;
  0.12, 5/24), where the variables are decodable from the full space but not from the first two
  axes individually.
- Not established: a regional difference. LEC recdays have ~2x the neurons and PCs of PFC
  recdays, so the time readout and the axis alignment need a neuron-count-matched (or
  PC-count-matched) control before "LEC carries more time information" can be claimed. This is
  a 5-vs-7-mouse comparison.

Legacy two-decoder numbers for the same days are in the "Superseded" paragraph above.
