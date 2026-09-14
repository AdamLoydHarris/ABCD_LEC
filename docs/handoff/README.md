# Anatomy split — master plan and handoff

**Written 2026-09-02.** Self-contained: a fresh session should be able to pick up any work
package from these documents without prior conversation.

- `W0_foundation.md` — gates, HD fix, shared module. **Complete.**
- `W1_production_glm.md` — the cross-validated GLM. **Complete; a refit is running.**
- `W2_headline_rescores.md` — re-score cached headlines by region. **Not started.**
- `W3_remapping.md` — generalising state cells + coherent remapping. **Next.**
- `W4_lagged_place.md` — lagged spatial cells. **Not started.**
- [`code/GLM_V3.md`](../../code/GLM_V3.md) — **GLM V3**: the reduced core model (`place`,
  `goal_progress`, `speed`, `acceleration`, `time_from_reward`) as a 2×2 over tfr coding × leg cap plus
  a goal-progress-only arm, both datasets, LEC by region. Engine `glm_analysis_v3.py` (v2 frozen).
  **Complete 2026-09-07** — results in §9 there and `ANATOMY_SPLIT.md` §W1.8.
- [`code/GLM_V3_NORMALISATION.md`](../../code/GLM_V3_NORMALISATION.md) — firing-normalisation ladder
  sanity check (no per-session normalisation / per-session offset = production / offset and gain), two
  primary arms, both datasets. **Complete 2026-09-08.** An across-session z-score is a proven no-op;
  `r2_cv` varies 4–5× across the ladder while the bias-corrected CPDs vary ≤13% and no V3 conclusion
  changes. Kept out of the `glm_v3` output and figure folders by request.
- `W5_gp_tuning_width.md` — sorted β heatmaps per V3 arm, and goal-progress **peak vs tuning width**
  read against the time-cell null that time-normalisation implies. **Complete 2026-09-08** — results
  in [`code/GP_TUNING_WIDTH.md`](../../code/GP_TUNING_WIDTH.md) §4, PFC pointer
  `mFC_data/code/GP_TUNING_WIDTH.md`, per-region figure guides in `docs/figures/gp_tuning_width*/`.

Supersedes `docs/ANATOMY_SPLIT_PLAN.md` (2026-09-01), which is still worth reading for the
histology background but has several factual errors corrected below.

---

## 1. The problem

The "LEC" recordings are not LEC. Hand-corrected Allen-atlas registration of every probe shows
the recorded bank spans **ENTl (superficial and deep), ENTm, SUB/ProS and CA1/HPF**, in
proportions that differ wildly between the five mice. Every headline in this repo — place
tuning, the goal-progress arc, the selectivity clusters, task-phase periodicity, the reward-time
cells — is an average over that mixture, and no analysis has ever read the anatomy.

Per-unit region labels now exist and are aligned to `Neuron_raw` rows. The job is to re-score
results by region, decide which headlines are regional, and be honest about which are one
mouse wearing a region label.

## 2. The data

| | |
|---|---|
| Recdays | **25** (5 mice × 5 days), `ah08 ah10 ly05 ly06 ly07` |
| Unit-recordings | 2851 |
| Sessions per recday | 8 (6 unique tasks + 2 repeats — the repeats are the X-vs-X' control) |
| `Neuron_raw` | integer spike counts per 25 ms bin, uint16, 85–94% zeros |
| Median leg duration | **9.38 s** (IQR 6.15–15.15, p99 115 s, max 771 s) |

**Region composition (units, per mouse):**

| mouse | ENTl-sup | ENTl-deep | ENTm | SUB/ProS | CA1/HPF | total |
|---|---|---|---|---|---|---|
| ah08 | **611** | 181 | 0 | 0 | 0 | 792 |
| ah10 | 9 | **378** | 22 | 176 | 113 | 772 |
| ly05 | 49 | 182 | 34 | 70 | 136 | 507 |
| ly06 | 13 | 133 | 49 | 137 | 0 | 356 |
| ly07 | 0 | 90 | **179** | 144 | 0 | 424 |

**Per-recday counts are the binding constraint, not the pooled totals:** ENTl-deep 15–80,
SUB/ProS 0–41, ENTm 0–45, CA1/HPF 0–30, ENTl-sup 1–141.

## 3. The inference design — non-negotiable

**Recdays are not independent replicates; mice are.** The five recdays of a mouse are the same
probe in the same brain, re-sorted, so a physical neuron plausibly appears in several. 2851 is a
count of unit-*recordings*. Effective n is the **number of mice** — at most 5, usually 1–3.

| contrast | recdays | mice | status |
|---|---|---|---|
| **ENTl-deep vs SUB/ProS** | 11 | **3** (ah10, ly06, ly07) | **primary** |
| ENTl-deep vs CA1/HPF | 7 | 2 (ah10, ly05) | secondary, report with and without ly05 |
| everything else | ≤5 | 1 | **descriptive only** |

Rules that apply to every analysis:

- **Every region reported on its own before any contrast.** A region with n=1 mouse still gets
  a panel, labelled descriptive — the mouse↔region confound lets a contrast look clean while
  both arms are single-mouse.
- **ENTl-sup is 90% ah08; ENTm is 63% ly07.** Any pooled claim about either is a single-mouse
  claim. Say so in the caption.
- Effect per recday → combine with **mice as the resampling unit** → **show per-mouse points**.
- **Permutation nulls shuffle `group` within recday.** Across-recday shuffling is degenerate —
  it breaks the mouse↔region association that is the whole confound.
- **Robustness on the primary contrast:** rate-matched subsampling and a 50 µm boundary margin.
  A result that moves under either is **unstable, not a finding**.

All of this is implemented in `code/anatomy_split.py`. Use it; do not re-roll `groupby('group')`.

## 4. The confound that threatens everything

**Region is confounded with firing rate, and the direction is not consistent across mice.**

| region | n | median rate (Hz) |
|---|---|---|
| ENTl-sup | 675 | **1.07** |
| ENTl-deep | 933 | 1.91 |
| ENTm | 276 | 2.28 |
| SUB/ProS | 505 | **5.86** |
| CA1/HPF | 213 | 1.93 |

SUB/ProS fires 3× ENTl-deep pooled — but **ly06 has the ratio inverted** (8.43 vs 5.61). So it
cannot be described as a fixed regional property or reasoned away; it has to be handled per
mouse, which is what `anatomy_split.rate_match` does (stratified on log rate, within recday).

Every statistic whose power scales with rate — the state-tuning t-test, permutation
significance, the reliability of a rotation estimate — inherits this.

## 5. Corrections to `docs/ANATOMY_SPLIT_PLAN.md`

That document is the starting point but is wrong in these places:

1. **`tuned_dict` is ternary `{−1, 0, +1}`, not bool.** ±1 is significant with direction, 0 is
   not significant. **A mean cancels +1 against −1**; the tuned fraction is `mean(x != 0)`.
2. **The binary tuned fraction is unusable for regional comparison.** 80.0% of all units are
   "place-tuned" with a range of only 70.3–85.6% across regions — saturated. The graded CPD
   discriminates cleanly. The plan's positive control *fails* on the fraction and *passes* on
   CPD; use CPD.
3. **`cells_LEC.pkl` (task-phase) fails its own "verify before use" check** — 2556 rows over 22
   recdays, missing `ah08_20250624_20250625` entirely (186 units, from the ENTl-sup mouse), and
   no `neuron` column. Re-run it; don't re-score it.
4. **Rank deficiency is data-dependent**, not the fixed 8 the docstring quotes. Blocks whose
   invalid rows are zeroed (HD, `goal_progress_distance`) stop summing to 1 and stop being
   collinear. `glm_cv.check_rank` measures it per fit.

## 6. Repo gotchas that have already cost time

- **`attach_pokes` must be called explicitly.** Without it `poke_rewarded`/`poke_unrewarded` are
  all-zero columns: CPD exactly 0.00000 and one lost rank each. This silently turned a
  "16-regressor" fit into 14 regressors plus two dead columns.
- **Neuron index is depth-ordered** (corr of row index with `y_um` = +0.976 to +0.988). So
  `arange(n)` is a *superficial-biased* subset — in ah10 it caught zero of 29 SUB/ProS and zero
  of 21 CA1/HPF units. **Any neuron subsample must be random.**
- **`compute_tuning_arrays` indexes rows by position in `sorted(GLM_results[rd])`,** not by
  neuron id. A missing neuron shifts every later row relative to `Neuron_raw`. Guard with
  `anatomy_split.assert_glm_keys_contiguous`.
- **`STALE_CACHE_RECDAYS` is keyed by recday but staleness is a property of a pickle.** 28
  pre-refit pickles genuinely hold the wrong day's spikes for `ly05_20250618_20250619` (91 units
  vs the correct 109); the new fits do not. `recday_registry.is_post_refit_section` scopes the
  flag by section name so the new fits keep all 25 recdays.
- **`run_or_load_glm` keys only on section name.** Two configurations sharing a name silently
  overwrite each other. `w1_refit.section_name` stamps `{regset}_{width}ms_{scheme}` into it.
- **The two `glm_analysis_v2.py` copies must never diverge** (`code/` and `mFC_data/code/`).
  Repo convention is duplication, not import.
- **Verify a background job is actually running before believing it.** `pgrep -f script.py`
  matches the shell wrapper too; a "running" job at 0% CPU and 3.7 MB RSS is an orphan. Check
  for a real python process.

## 7. Conventions

- **Plot style**: `glm_analysis_v2.apply_gridmaze_style()` (Arial 8 pt, no top/right spines,
  Type-42). Save with `bbox_inches=None`. Region colours are `anatomy_split.REGION_COLORS`.
- **Synthetic controls on every new piece of code, run through the real pipeline.** This is repo
  practice and has caught six plausible-looking design errors so far.
- **Cluster**: `sbatch_files/glm_{lec,pfc}.sbatch` + `submit_glm_{lec,pfc}.sh`, one job per
  recday writing per-recday shards, then an explicit `--merge`. The shard/merge split exists
  because 25 jobs writing one pickle would race and the loser's recdays would vanish silently.
  Shards carry a config stamp and the merge refuses to combine mismatched settings.

## 8. State of play, 2026-09-07

**Production fits with the Freedman–Lane null landed 2026-09-06**, both datasets, 25/25 recdays:
`data/glm_outputs/LEC/all_regressors__full_250ms_decile__*.pkl` and
`mFC_data/glm_outputs/PFC/all_regressors__matched_250ms_decile__*.pkl` (`nulls=['freedman_lane']`,
`n_perm=100`; keys `p_freedman_lane__{cpd,delta_r2}`, `null_mean_freedman_lane__*`). On the real data
the CPD-based `frac_sig` moved by ≤ 0.016 per regressor relative to the shuffle null — the synthetic
100 % false-positive case was harsher than the data.

**GLM V3 (`code/GLM_V3.md`) landed 2026-09-07**: 12/12 synthetic controls on both trees (v3 ≡ v2 to
`max|diff| = 0`), 300 SLURM jobs in 81 min with 0 failures, all twelve arms merged at 25/25 recdays,
both notebooks executed (`code/LEC_glm_v3.ipynb`, `mFC_data/code/PFC_glm_v3.ipynb`, `RUN_MODE='load'`).
Results in `GLM_V3.md` §9 and `ANATOMY_SPLIT.md` §W1.8. Headlines: the reduced model's CPDs grow by
reassignment of shared variance while it generalises marginally *better* than the 16-regressor fit; the
gp-vs-tfr split flips with the tfr coding; PFC has 4–5× LEC's within-leg structure; ENTl-deep >
SUB/ProS in `time_from_reward` is a candidate (+0.001 CPD, 4 mice, rate matching survived in 2 of 4
arms), goal progress shows no regional difference.

One data note: the production LEC fit of `ah10_20250618_20250619` (2 Sep) predates the salvage of its
session 5 (6 Sep) and has 5 folds where the V3 arms have 6; same-rows comparisons exclude it.

**Two derived pickles are unsafe to index (found 2026-09-08):** `data/processed_data/norm_neurons_dic.pkl`
(1 Sep) has a per-recday value that is a *list over sessions that have trials*, not indexed by session
number — in 9 of 25 recdays entry *i* is not session *i*. `gp_curves.pkl` (3 Sep) lacks five sessions.
Rebuild from `data_dic_lec.pkl`; details in `W5_gp_tuning_width.md` §4.

**Correction to that claim, measured 2026-09-08 (W5 §3.2):** both pickles were also called *stale* on the
grounds that they predate the 6 Sep salvage of `ah10_20250618_20250619` session 5. They are not. Comparing
the current `data_dic` with `data_dic_lec.pkl.PRE_salvage_…` across all 25 recdays and every session, the
salvage **added `Locs_raw`, `XY_raw` and `HD_raw` to that one session and changed nothing else** —
`Neuron_raw` and `Trial_times` are identical everywhere, and a rebuild of the per-leg curves reproduces
`norm_neurons_dic` to 3 × 10⁻⁷ on that very session. What the salvage changes is the **session set**:
`get_sessions_for_glm` needs `Locs_raw`, so session 5 is a sixth fold now and was invisible before, which
is why that recday's production fit has five folds where the V3 arms have six.

**W5 landed 2026-09-08** (`code/GP_TUNING_WIDTH.md`): 8/8 synthetic controls on both trees, both notebooks
executed, per-region figure guides under `docs/figures/gp_tuning_width*/`. Built as **standalone files** at
the user's instruction — `gp_tuning_width.py` + `gp_tuning_width_synthetics.py` (byte-identical copies with
their own `assert_mirror`, deliberately **not** added to `check_mirror_parity.BYTE_PAIRS`) and two new
notebooks; `glm_plots.py`, `pfc_glm_plots.py`, `check_mirror_parity.py` and the V3 notebooks are untouched.
Headline: the goal-progress population traces the width-vs-peak **V that time coding predicts** (ρ(|peak −
0.5|, width) ≈ −0.52 in both datasets against −0.48 for planted time cells and +0.01 for planted phase
cells), with a **retrospective** asymmetry. The pre-registered discriminator did not fire — stratifying by
the GLM's gp-vs-tfr split does not abolish it — so this is a strong description, not a settled reading.

**Next:** unassigned. W2 (headline rescores) and W4 (lagged place) are still not started.

**New guard:** `code/check_mirror_parity.py` runs at the top of both `submit_glm_*.sh` and refuses to
submit when `code/` and `mFC_data/code/` have drifted (this has cost two 25-job runs).

**Committed 2026-09-14** as checkpoint `46c93db` (285 files: GLM V3, V5, El-Gaby tooling, W5, LDA
runner; `.pyc` untracked, outputs and the El-Gaby run trees ignored).

**2026-09-14, launched (results land overnight):**
- El-Gaby's own `Figure5_Regression` with **prospective** lags, both regions -- a declared VARIANT
  (`20_apply_edits.py --variant prospective`: cell 15's bump model on the time-reversed session,
  `_prospective` suffix on every file). His retrospective regressors equal V5 `past` to
  `max|diff| = 0` on all 6 sessions of ah08_20250613 (sign-check harness); the prospective arrays are
  checked against V5 `future` as soon as cell 15 writes them. Jobs 3594655 (LEC), 3594658 (PFC).
- V5 spec 13 (LEC Poisson future, repro config + gate; the last empty cell of the 2x2x2), job 3594633.
- **L1 / elastic-net Poisson** via glum (`poisson_solver='glum'`, `poisson_l1_ratio`,
  `poisson_positive` in `RegressionConfigV5`; sklearn's `PoissonRegressor` is L2-only). Controls
  16-18 added (glum `l1_ratio=0` == sklearn to 0.2 % of max|beta|; lasso recovers the planted cell
  and is monotone in alpha; guards). Sweep specs 14-17 (PFC past, spec 2's config: lasso alpha
  0.01 / 0.003 / 0.001, elastic net 0.5 @ 0.003), jobs 3595313-3595316. Under an L1 penalty a
  fixed alpha is a firing-rate filter -- read the all-zero-fit fraction per alpha before the counts.
- Reward x goal-progress LDA on both datasets through `run_lda_reward_progress.py` (jobs 3594885
  PFC, 3594886 LEC); pickles at `{data,mFC_data}/glm_outputs/{LEC,PFC}_lda/reward_progress.pkl`.
  The module now skips 1-PC recdays instead of crashing. **Landed the same day** (LOGO-CV balanced
  accuracy, min_trials=10, 1000-shuffle null, per-recday; `data/figures/lda_reward_progress/`):
  goal progress LEC 0.61 (21/22 recdays p<0.05, 5 mice) vs PFC 0.66 (24/24, 7 mice), chance 0.333;
  reward number LEC 0.044 (14/22) vs PFC 0.037 (12/24), chance 0.025. LEC skips: ah08_20250624,
  ly05_20250616, ly05_20250618 have < 3 sessions with >= 10 trials; PFC skips me10_20122021 (1 PC).
  ly05 is the outlier mouse on progress (0.43); mice are the replicates, so this is a 5-vs-7-mouse
  comparison with no regional difference worth claiming yet.
  **Superseded the same evening** by one joint LDA on the trial x progress conjunction (30 classes,
  the 4 states of a trial as 4 samples per class), read out on held-out sessions with a
  within-session circular-shift null -- `code/LDA_REWARD_GOALPROGRESS.md`. Time in session as a
  distance: trial MAE LEC 2.36 vs PFC 2.71 (null 3.3; 19/22 and 21/24 recdays p<0.05), seconds
  r 0.54 vs 0.26; progress accuracy 0.57 vs 0.65 (chance 0.333). The in-sample "LD1 = time,
  LD2 = progress" picture survives cross-validation in LEC (14/22, 12/22 recdays) but not in
  PFC (2/24, 5/24). LEC recdays have ~2x PFC's neurons and PCs; no regional claim without a
  count-matched control.
  **Matched control (15 PCs into every LDA) and behavioural covariate, same evening:** the
  full-space readouts hold (LEC seconds r 0.52 vs PFC 0.23, trial MAE 2.45 vs 2.72; progress
  0.58 vs 0.65), the axis-level LD1 = time / LD2 = progress picture collapses in LEC (14 -> 7 and
  12 -> 5 recdays) and was absent in PFC, and trial number maps onto seconds identically in the
  two datasets (r 0.98 both; CV 0.40 vs 0.38; drift -4 s/trial both; LEC trials ~15 % longer).
  Remaining confound: neuron count (90 vs 50) -- a subsampled-LEC run is the next control.
