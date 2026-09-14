# Anchoring regression V5 — PFC / mFC dataset

`elasticnet_regression_v5.py` · `PFC_elasticnet_regression_v5.ipynb` · `elasticnet_v5_synthetics.py`

Mirror of the LEC write-up at [`../../code/ELASTICNET_V5.md`](../../code/ELASTICNET_V5.md).
The module is kept **byte-identical** to the LEC copy — `diff` it — so everything in that
document about the method applies here unchanged. This file records what is different about
*this dataset* and what the V5 reproduction run gave. The V4 notes
([`ELASTICNET_V4.md`](ELASTICNET_V4.md)) still hold for what V5 did not change (the α problem,
the structural facts, the untracked-bin no-op, the recdays to watch).

```bash
python elasticnet_v5_synthetics.py          # 57 controls — run before trusting a result
diff ../../code/elasticnet_regression_v5.py elasticnet_regression_v5.py   # must be empty
```

## This is El-Gaby's own dataset

`data/MetaData/combined_ABCDonly_days.npy` is the file `Figure5_Regression.ipynb` loads, and the
25 recdays are `me08 / me10 / me11 / ah03 / ah04 / ah07 / ab03`. Running V5 here re-runs the
published Figure 5 analysis on the published data with the definitions read off his deposited
code, so every number below is a statement about the paper.

## The reproduction run (2026-09-06)

`elasticnet_v5_past_20260906_112347`: ElasticNet α = 0.01, `MIN_TRIALS = 1` (his
`num_trials > 0`), `pref_phase_source='test'` (his cell 21), his state-tuning statistic with NaN
propagation, his preferred-phase rule, past lags, 25 recdays, 1,252 units, 2.5 h at `n_jobs=6`.

| panel (his semantics) | n | mean r | t | paper (Fig 5h / ED 8a) |
|---|---|---|---|---|
| all state-tuned, finite r in every fold | **447** | +0.208 | **14.5** | 489, t = 9.3 |
| non-zero-lag 30°, ≥ 1 passing fold | **246** | +0.066 | **2.48** | 329, t = 3.9 |
| non-zero-lag 90°, ≥ 1 passing fold | **79** | +0.088 | **1.64** | 224, t = 2.53 |

V4 semantics on the same fits: 610 / 186 / 16 (t = 14.0 / 3.32 / −0.40). State-tuned by his
test: **729** (his 738). Sessions skipped by the fit: 1 (`ah04_07122021_08122021` session 2, one
completed trial).

How this compares with the post hoc re-scoring of the v4 fit under his tuning definitions
(481 / 287 / 92, t = 14.0 / 2.69 / 1.77): the re-fit with **his preferred-phase rule** moved every
panel slightly *away* from the paper, so that rule is not the missing ingredient. It also makes
the fitted row set unstable: **74%** of neurons change preferred phase between folds under his
peak rule, against 36% under the raw-mean rule.

**The smoothing sensitivity** (`smooth10_elasticnet_v5_past_20260906_173444`, identical config
plus `pref_phase_smooth_sigma=10`, his `smooth_circular` default): **463 / 276 / 88**, t = 14.0 /
2.63 / **−0.13**. Smoothing moves the preferred-phase rule back toward the mean rule (agreement
0.66 → 0.82) and the counts back toward the v4-fit numbers, but the 90° panel's correlation
collapses to zero.

| variant (his semantics) | all | 30° | 90° |
|---|---|---|---|
| v4 fit (raw-mean phase), his tuning post hoc | 481, t 14.0 | 287, t 2.69 | 92, t 1.77 |
| V5 re-fit, his peak rule unsmoothed | 447, t 14.5 | 246, t 2.48 | 79, t 1.64 |
| V5 re-fit, his peak rule smoothed σ=10 | 463, t 14.0 | 276, t 2.63 | 88, t −0.13 |
| **paper** | **489, t 9.3** | **329, t 3.9** | **224, t 2.53** |

**Where the reproduction stands.** The denominators reproduce (736 vs 738 state-tuned, 481 vs
489 finite-in-every-fold); the all-neuron right-shift is *stronger* than his in every variant
(t 14 vs 9.3); the 30° panel is smaller in n (246–287 vs 329) and weaker in t (2.5–2.7 vs 3.9);
the **90° panel (79–92 neurons vs 224, t ≤ 1.8 vs 2.53) is not reproduced by any definition we
can read off the deposited code**. Candidates left, none testable from the repository: his
noisy folds (the me11 misalignment in his cell 26 and the 1–4-trial sessions pass a random
top-3 test at 12.5%, above our 6.4% per-fold rate) and an executed strict set different from the
deposited `[0,1,2,11,10,9]`. The 90° claim (ED Fig 8a) should be treated as unreproduced.

## What differs from LEC

| | PFC | LEC |
|---|---|---|
| units / recdays | **1252 / 25** | 2851 / 25 |
| trials per session (median) | **27** | 18 |
| mean firing rate | **6.25 Hz** | 2.94 Hz |
| his state test, p<0.05 in >1/3 of tasks | **736** (59%) | 1,433 (50%) |
| neuron-sessions removed by NaN propagation alone | small | **14.6%** |
| preferred-phase agreement, his rule vs raw mean | **0.70** | 0.63 |
| `alpha=0.01` all-zero fits | **57%** | 60% |
| leg longest:shortest, median | **1.82×** | 2.26× |
| sessions with `num_trials` 1–4 | **7** (4 recdays) | 18 |
| anatomy | **none** | `unit_regions.pkl`, 6 groups |

## Session selection matches the reference

1. **`MIN_TRIALS = 1`** — his `num_trials > 0`. (`_prepare_session` still needs two completed
   trials to define a fold, which drops exactly one PFC session.)
2. **Dedup by exact task equality** — his `non_repeat_ses_maker`.
3. **`EL_GABY_EXCLUDED_SESSIONS`** — `me11_05122021_06122021` session 3 (`[7,4,3,8]` vs session
   0's `[7,4,3,5]`, his "almost identical to session 0 (mistake)"). Note his own cell 26 applies
   it inconsistently: cells 15/21 fit 7 folds with session 3 in training, cell 26 forces 6 and
   drops 3 from the session list, so positions 3–5 pair regressors/betas from sessions 3, 4, 5
   with phases/trial-times from sessions 4, 5, 7 and session 7 is never scored. We apply his
   intent cleanly; me11's 46 neurons (3.7%) are therefore an irreducible difference.

## Poisson runs (his executed default, α = 1, with his `nanmean(prediction) > 0` gate)

`poisson_v5_past_20260907_153716`, `poisson_v5_future_20260908_104201` (MIN_TRIALS=1,
test-session preferred phase, both links stored from one fit).

| semantics | direction | all state-tuned | 30° | 90° |
|---|---|---|---|---|
| his | past | 140, r +0.273, **t 12.1** | 86, r +0.155, **t 3.34** | 19, r +0.150, t 1.24 |
| his | future | 114, r +0.247, t 8.92 | 68, r +0.012, t 0.20 | 8, r +0.102, t 0.43 |
| v4 | past | 573, r +0.257, t 15.8 | 162, r +0.108, t 3.70 | 8, r −0.014, t −0.08 |
| v4 | future | 537, r +0.254, t 13.8 | 109, r +0.110, t 2.70 | 7, r +0.334, t 1.94 |
| paper (ED 8d) | — | 489, t 10.70 | 346, t 4.74 | 229, t 2.81 |

Three things to take from this:

1. **The two links are numerically the same.** `exp(Xβ+b)` reproduces the linear-predictor
   readout to 3–4 decimal places in every panel (140, +0.2733/t 12.09 vs +0.2734/t 12.08),
   because α = 1 shrinks max|β| to ~0.007 and `exp` is affine over that range. The code/paper
   divergence is real but has no numerical consequence here.
2. **His gate, not the link, is what moves n.** `nanmean(prediction) > 0` drops 51% of Poisson
   folds, so "finite r in every fold" leaves only 140/114 units against 573/537 under V4
   semantics. ED 8d's 489 is near the V4 denominator, so that panel cannot have applied the gate
   and the every-fold rule together.
3. **Poisson reproduces the 30° panel better than ElasticNet** (t 3.34 vs 2.48; paper 4.74),
   while the **90° panel fails under both** (t 1.24 here, 1.64 for ElasticNet, vs 2.81) — ED 8a's
   failure is not an artefact of the estimator choice.

## The z-scored-target variant

`alpha=0.01` on raw counts zeroes every coefficient for **57%** of PFC fits (33% of units have no
fold with any non-zero beta at all), and it does so as a function of firing rate. Runs with
`y_scaling='zscore_recday'` (jobs 6/7 of
[`../../code/slurm_v5/submit_v5_run.sh`](../../code/slurm_v5/submit_v5_run.sh), written to
`*_zscore_v5_*` directories) divide each neuron's fit target by its sd over the whole recday and
are the comparison for that. They use the same reproduction criterion as the raw PFC runs
(`MIN_TRIALS=1`, test-session preferred phase) so the two are comparable neuron for neuron, and
everything upstream of the fit is bit-identical between them. They are **not** the reproduction:
the published counts assume raw counts at a fixed alpha. The mechanics, the exact penalty algebra
and the low-spike caveat are in the `y_scaling` section of
[`../../code/ELASTICNET_V5.md`](../../code/ELASTICNET_V5.md).

### Result: the paper's ElasticNet numbers are *bracketed* by the two target scalings

`elasticnet_zscore_v5_past_20260908_122326` vs `elasticnet_v5_past_20260906_112347`, identical in
every other setting (his tuning test, his preferred-phase rule, `MIN_TRIALS=1`, test-session
preferred phase), scored under his fold semantics:

| panel | raw counts | recday z-score | paper |
|---|---|---|---|
| **past** pool (state-tuned, finite r every fold) | 447, r +0.208, t 14.5, d 0.685 | **725**, r +0.216, t 19.7, d 0.730 | **489**, t 9.30 |
| **past** non-zero-lag 30° | 246, r +0.066, t 2.48, d 0.158 | **420**, r +0.090, **t 4.19**, d 0.205 | **329**, t 3.90 |
| **past** non-zero-lag 90° | 79, r +0.088, t 1.64, d 0.185 | 126, r +0.071, t 1.55, d 0.138 | 224, t 2.53 |
| **future** pool | 463, r +0.238, t 16.1, d 0.750 | 725, r +0.231, t 19.6, d 0.726 | — |
| **future** non-zero-lag 30° | 258, r +0.046, t 1.76, d 0.109 | 436, r +0.064, **t 3.01**, d 0.144 | — |
| **future** non-zero-lag 90° | 90, r +0.068, t 1.22, d 0.129 | 106, r +0.028, t 0.53, d 0.051 | — |

The future direction replicates the pattern independently: the 30° effect size rises by 32%
(0.109 → 0.144) against 30% for past, and the 90° effect size falls again (0.129 → 0.051). Past
stays stronger than future in both scalings (t 4.19 vs 3.01 z-scored), as retrospective anchoring
should be. The pool's effect size is essentially unchanged in both directions (0.685 → 0.730,
0.750 → 0.726), which is the expected null: that panel includes lag 0 and is dominated by
place/phase coding that the penalty was never suppressing.

1. **His denominator and his 30° effect both fall between our two scalings**: 447 < **489** < 725,
   and t 2.48 < **3.90** < 4.19. The residual 30° disagreement is therefore consistent with his
   executed shrinkage being *milder than α = 0.01 on raw counts but not absent* — a calibration
   difference rather than a structural one, and the first quantity in this reproduction that
   brackets a published value from both sides.
2. **The 30° gain is not merely a larger n.** Cohen's d rises 0.158 → 0.205 (+30%) alongside
   n 246 → 420, so removing the rate filter strengthens the *per-unit* effect — consistent with
   the filter having discarded genuinely anchored units for firing too slowly.
3. **The 90° panel is neither rescued nor bracketed** (t 1.64 / 1.55 vs 2.53 on 224 units); its
   effect size *falls*. With the Poisson result above (t 1.24), ED 8a now fails under both
   estimators, both semantics and both target scalings — it is not a shrinkage artefact of ours.

With a z-scored target the "(with non-zero beta coefficients)" filter removes almost nobody
(725 of 729 state-tuned units survive), the regime the paper's caption language implies — but it
overshoots 489, which is why point 1 is a bracket and not a match.

## Recdays to watch

Unchanged: `me10_20122021_21122021` has **1 neuron**, `me10_17122021_19122021` 6, the two ah03
recdays 15–16. The cross-dataset comparison
([`../../code/elasticnet_v5_compare.py`](../../code/elasticnet_v5_compare.py)) gates on
`min_neurons=2`, and its `mean_r` is now the reduced-beta r of the selected units.

## Runtime

1,252 units, ~6 folds each. ElasticNet **2.5 h per direction** at `n_jobs=6` on a loaded box
(20 min when idle); Poisson ≈ 7× that.
