# Reconciling our anchoring regression with El-Gaby et al. 2024, Figure 5

`elgaby_figure5.py` · `elgaby_ladder.py` · `elasticnet_regression_v5.py`

This document records the reconciliation as it developed. **Sections 1–6 below were written
against V4 and the paper's text; the update in §0 (2026-09-06) supersedes them where they
conflict**, because the deposited GitHub code was read line by line afterwards and several
"settled" items turned out to be wrong. The V5 module implements the corrected definitions; see
[`ELASTICNET_V5.md`](ELASTICNET_V5.md).

---

## 0. Update 2026-09-06 — what the deposited code says

The full repository is at `mFC_data/mFC_schema-main/`. `Figure2.ipynb` contains the cells that
*build* every precomputed file the regression notebook loads (`Phases_raw2`,
`tuning_phase_boolean_max`, `State_zmax` → `State_95`/`State_99`). The Ceph
`mFC_data/data/Intermediate_objects/` folder holds none of the anchoring intermediates.

### 0.1 The "leading suspect" (§3.3, §3.8) is refuted — and the deposited notebook is not the executed code

* `Figure2.ipynb` cell 18 writes `Phases_raw2` with `num_phases=5` (`num_phases2=3` is defined
  and never used). If that were what ran, the lag axis would advance five steps per state.
* But `Figure5_Regression.ipynb` cell 21 applies one `concatenate_complex2` to `np.vstack`ed
  `(n_trials, 4)` object arrays and boolean-indexes with a per-bin mask — a synthetic
  round-trip raises `IndexError`. The deposited cell cannot run on the deposited arrays.
* The Fig 5g raster (extracted from the PDF with `pdfimages`, 27 × 12 grid) has all 7 non-zero
  betas exactly on the 3-bin stripe `ap == (0 − lag) mod 3`; 5 of 7 would be structurally dead
  under 5-bin phases.

**His executed pipeline used 3-bin phases with a strict cycle, as ours does.** Retire the suspect.

### 0.2 The "25% vs 67%" pass-rate gap (§3.3, §6 of the handoff) was a semantics mismatch

His rule counts a neuron if ≥ 1 fold passes the top-3 test (§3.2); the 25% was our
majority-vote rate. Under his rule on our betas the 30° rate is 66%.

### 0.3 The state-tuning test does NOT match his code (§2 was wrong)

§2 says the test matches "line for line". It matches the **paper's text** ("peak firing rate in
each state and trial"). **His code (Figure2 cell 46)** takes, per (trial, state), the **mean over
the neuron's preferred-phase bins** (30 of the 90 unsmoothed normalised bins), z-scores across
the 4 states with `scipy.stats.zscore` (**NaN propagates**: a trial with four equal means makes
that session NaN → not tuned), takes the preferred state as argmax of the trial-mean raw means,
and t-tests that state against 0. `State_95` = p<0.05 in > 1/3 of non-repeat sessions
(`State_zmax_bool`, cells 48/56); `State_99` = p<0.01. The ED 8b "99th percentile of permuted
distribution" wording is loose; the code is the t-test.

Consequences: (a) the leg-duration confound (`ELASTICNET_V4.md` §1) belongs to the paper's text
statistic, **not** to his code — V5 control 9 measures his statistic's FPR at 3× legs at 0.09
against 0.65–1.00 for `'max'`; "the reference has the confound too" (§6 below) is wrong;
(b) NaN propagation preferentially drops low-rate units (PFC: tuned 7.5 Hz vs untuned 3.8 Hz).

Measured on PFC (all 25 recdays): his test gives **736** state-tuned (paper Fig 5b/c/f: 738/737),
**492** at p<0.01; our `'max'` gives 774 but only 554 of the same neurons.

### 0.4 The preferred phase is not our rule either

His `tuning_phase_boolean_max` (Figure2 cell 31): normalise the session to (trials, 360), average
over states and trials to a 90-bin curve, take the **peak bin within each third**, argmax. Ours
was the argmax of the raw time-weighted mean rate per phase. Agreement: **0.70** on PFC, **0.63**
on LEC neuron-sessions. Under his rule **74%** of PFC neurons change preferred phase between
folds (36% under ours). Whether his `Neuron_` arrays were smoothed first is unknown — the
writer cell is not in the repo, but his `raw_to_norm` helper smooths by default
(`smooth_circular`, σ=10). Measured on 8 PFC recdays (4,246 neuron-sessions): smoothing the
curve with σ=10 raises the peak rule's agreement with the raw-mean rule from **0.66 to 0.82**
(σ=5: 0.76), i.e. the smoothed peak sits between the two rules. `pref_phase_smooth_sigma=10` is
the one remaining cheap sensitivity for the residual (run `smooth10_elasticnet_v5_past_*`).

### 0.5 "With non-zero beta coefficients" means finite in EVERY fold

Cell 38's comment says "at least half of the sessions"; the paper's n is matched only by
requiring a finite correlation in all folds: his-tuned & finite-every-fold = **481** (paper 489),
and **359** at p<0.01 (ED 8b: 349).

### 0.6 Smaller items

* Sessions: he keeps any non-repeat session with `num_trials > 0`; we dropped `< 5` (7 PFC
  sessions in 4 recdays). V5 notebooks expose `MIN_TRIALS`.
* **`me11_05122021_06122021` is misaligned in his cell 26**: cells 15/21 fit 7 folds (session 3
  in training), cell 26 forces 6 and removes 3 from the session list, so positions 3–5 pair
  regressors/betas from sessions 3, 4, 5 with phases/trial-times/pref-phase from sessions 4, 5, 7,
  and session 7 is never scored. 46 neurons. We apply his intent cleanly; irreducible.
* His `normalise` divides legs shorter than 90 bins by 10 (`np.repeat(xx,10)/10`); 1.1% of PFC
  legs. Negligible.
* Two V4 diagnostics (the per-recday pooled histogram, `mean_r_selected`) scored non-zero-lag
  neurons with the all-betas r; V5 uses the reduced-beta r.

### 0.7 What the numbers now are

Post hoc re-scoring of the stored v4 PFC fit (`elasticnet_v4_past_20260904_222909`) under his
tuning statistic and his semantics, no re-fit — then the V5 re-fit with all of his definitions
(`elasticnet_v5_past_20260906_112347`, `MIN_TRIALS=1`, test-session preferred phase):

| panel, his semantics | v4 fit, his tuning (post hoc) | **V5 re-fit** | V5 re-fit, σ=10 | paper |
|---|---|---|---|---|
| all state-tuned, finite r every fold | 481, t = 14.0 | **447, t = 14.5** | 463, t = 14.0 | 489, t = 9.3 |
| non-zero-lag 30°, ≥ 1 passing fold | 287, t = 2.69 | **246, t = 2.48** | 276, t = 2.63 | 329, t = 3.9 |
| non-zero-lag 90°, ≥ 1 passing fold | 92, t = 1.77 | **79, t = 1.64** | 88, t = −0.13 | 224, t = 2.53 |
| p<0.01 pool: all / 30° / 90° | 359 / 205 / 62 | — | — | 349 / 227 / 154 (ED 8b) |

The denominators reproduce; the all-neuron right-shift is stronger than his in every variant;
the 30° panel is smaller and weaker; the **90° panel is not reproduced by any definition that
can be read off the deposited code** (79–92 neurons against 224, t ≤ 1.8 against 2.53). His
preferred-phase rule moved every panel slightly *away* from the paper; smoothing it with his
σ=10 moves the counts back toward the mean-rule numbers (the two rules then agree on 82% of
neuron-sessions) but leaves the 90° correlation at zero. What is left, none of it testable from
the repository: his noisy folds (the me11 misalignment and the 1–4-trial sessions pass a random
top-3 test at 12.5% against our 6.4% per-fold rate) and an executed strict set different from the
deposited `[0,1,2,11,10,9]`. The tie-breaking guard is not it (92 → 89): the pool's fits are
dense (median 9 non-zero betas per fold, 89% of folds have ≥ 3). **Treat ED Fig 8a (the 90°
panel) as unreproduced.**

---

## Outcome

**The premise does not survive.** Our distribution is not less right-shifted than the paper's —
what was being compared was our `{0,1,11}` right-hand panel against his `{0,11}` right-hand
panel. On matched quantities we equal or exceed him. The paper PDF settled every open question
except two, both of which need files that are not in this repository.

The mFC dataset **is El-Gaby's own published data**
(`mFC_data/data/MetaData/combined_ABCDonly_days.npy` is the file `Figure5_Regression.ipynb`
loads), so this is a reproduction attempt and a gap would be a failure to reproduce.

### The comparison, on matched quantities

Effect size `t/√n` is what "right-shifted" means once n differs. PFC, past lags, 25 recdays.

| panel | source | n | t | P | t/√n |
|---|---|---|---|---|---|
| **left** (no lag filter) | paper Fig 5h | 489 | 9.3 | 5.3e-19 | 0.421 |
| left | ours, OR tuning | 824 | 14.1 | 2e-40 | **0.490** |
| left | ours, **his >1/3 rule** | **590** | 12.0 | 7.2e-30 | **0.494** |
| **right** {0,11} | paper Fig 5h | 329 | 3.9 | 1.08e-4 | 0.215 |
| right {0,11} | ours, per-neuron mask | 287 | 4.6 | 5.3e-06 | **0.274** |
| right {0,11} | ours, El-Gaby per-fold | 510 | 3.0 | 0.0027 | 0.133 |
| right {0,1,11} | ours, **the old v4 default** | 155 | 2.8 | 0.0062 | 0.223 |

The last row is the number that prompted the investigation (the `3.38e-02` on the old summary
SVG is the same selection scored on the all-betas prediction: n=155, t=2.14).

### The full reproduction target, from the captions

| variant | state tuning | none | {0,11} (30°) | {0,1,2,9,10,11} (90°) |
|---|---|---|---|---|
| **ElasticNet** (Fig 5h, ED 8a) | p<0.05 | n=489 t=9.3 | n=329 t=3.9 | n=224 t=2.53 |
| **ElasticNet, strict tuning** (ED 8b) | **p<0.01** | n=349 t=8.70 | n=227 t=2.83 | n=154 t=1.94 |
| **Poisson** (ED 8d) | p<0.05 | n=489 t=10.7 | n=346 t=4.74 | n=229 t=2.81 |

Every axis is a V4 config: the estimator, `state_tuning_p_threshold` (0.05 / 0.01), and the
lag-exclusion set — which is why V4 now emits **all three lag levels on every run**.

---

## 1. What the analysis is

Per neuron, per held-out session (leave-one-*task*-out):

1. **Regressors.** Every (location, goal-progress phase) pair is an "anchor". Anchor
   *(loc, ap)* at lag *k* is on when the animal visited `loc` during a phase-`ap` segment
   *k* phase-steps ago. 9 locations × 3 phases × 12 lags = **324 columns**, in raw 25 ms bins.
2. **Preferred-phase restriction.** Each neuron is fit only on bins of its preferred
   goal-progress phase — **both X and y** are subset.
3. **Fit.** One estimator per neuron per fold, on the concatenated training sessions.
4. **Readout.** Apply the betas to the held-out session's regressors, normalise both actual and
   predicted onto 90 bins × 4 states, reduce each state to the **mean of its 30 preferred-phase
   bins**, and Pearson-correlate the two 4-vectors.
5. **Selection.** State-tuned × "non-zero-lag" (the largest betas sit at intermediate lags,
   i.e. the neuron is not simply a place cell).

### Structural facts that follow, verified on his data

* Goal-progress phase advances as a strict 0→1→2 cycle — **0 of 59,904** transitions deviate
  from +1 mod 3. So the anchor phase at lag *k* is *determined*: `ap == (pref − k) % 3`.
* Therefore only **108 of the 324 columns** can be non-zero in any fit, and the fitted betas lie
  on a diagonal stripe.
* Therefore the prediction is **exactly zero at every non-preferred-phase bin**, so 240 of the
  360 normalised bins of any "predicted tuning curve" are zero by construction.

---

## 2. What is identical

Checked line-for-line against `Figure5_Regression.ipynb`.

| | |
|---|---|
| the n=4 readout | `np.nanmean` over **exactly 30** preferred-phase bins per state — `nanmean` in all 8 places that build the 4-vector; `nanmax` appears **0 times in the entire notebook** |
| trial averaging | `use_mean=True`: the 360-bin curve is trial-averaged first, giving n=4 points |
| smoothing | none — `raw_to_norm(..., smoothing=False)` |
| the prediction | `np.sum(regressors * coeffs, axis=1)` — no intercept, and **no `exp`** even in the Poisson branch |
| normalisation grid | 90 bins/state × 4 states, `binned_statistic` mean, phase thirds coinciding exactly with the raw phase boundaries |
| session selection | dedup by exact task equality — this *is* his `non_repeat_ses_maker`, both use `np.array_equal` on the reward sequence |
| the hand-exclusion | `me11_05122021_06122021` session 3 (task `[7,4,3,8]` shares 3 of 4 goals with session 0's `[7,4,3,5]`), the **only** `mouse_recday ==` special case in his notebook. With it applied me11 has **6 folds**, matching his `num_non_repeat_ses_found = 6` |
| **the state-tuning test** | **WRONG — see §0.3.** This row compared our test with the paper's *text*; his *code* uses the mean over the preferred-phase bins, not the peak, with NaN propagation. Only the P<0.05 / P<0.01 thresholds and the "> 1/3 of tasks" rule were right |
| the regressor bump loop | validated against an independent segment-based reference: agreement **1.0000** at lags 1–11 in both directions (lag 0 differs by construction — it accumulates causally) |

---

## 3. What differs, and by how much

Ordered by measured effect on the count.

### 3.1 The lag set — the dominant factor for the count

His `close_to_anchor_bins_30 = [0, 11]`. V4's default window excludes **{0, 1, 11}**.

Lag 11 matters because with `num_lags = 12 = 4 states × 3 phases` the lag ring is exactly one
ABCD loop, so "11 steps back" lands one step *before* the current position, one loop earlier —
a near-current column wearing a large-lag label. He excludes it for that reason; V4 additionally
excludes lag 1.

**Dropping lag 1 nearly doubles the count: 155 → 287**, and raises the effect size from
0.223 to 0.274 — past the paper's 0.215. The paper states the rule in words: *"lag from anchor
 of **30° or more**"*, and 30° is one goal-progress bin, so the excluded set is {0, 11}.

### 3.2 Per-fold NaN vs a per-neuron mask

He has no neuron mask. Cell 26 sets *that fold's* correlation to `np.nan` when a top-3 beta
lands in the excluded set, then `nanmean`s the survivors — so a neuron **counts if ≥ 1 fold
passes**. V4 builds a per-neuron boolean mask by majority vote across folds.

**287 → 510.** Together with 3.1 these bracket the reported 329 — but note the mean r
*falls* (+0.087 → +0.054), because admitting a neuron on one surviving fold admits noise.

His exclusion also applies *only* to the reduced-beta correlation, never to the all-betas one:

| his array | prediction built from | top-3 exclusion? |
|---|---|---|
| `Predicted_Actual_correlation_mean` | all betas | **no** |
| `Predicted_Actual_correlation_nonzero_mean` | lags {0,11} → NaN | **yes** |
| `Predicted_Actual_correlation_nonzero_strict_mean` | lags {0,1,2,9,10,11} → NaN | **yes** |

**`n = 329` is the middle one.** "A non-zero-lag neuron scored with all its betas" is a
combination he never computes — but it is what V4's headline `mean_corrs` over `selected` is.

### 3.3 Estimator, and where the betas actually sit

His *executed* default is **Poisson** (`Poisson_regression=True`, α=1, L2 only). The
ElasticNet α=0.01 branch is the one his comment marks "0.01 used in paper".

On his own data at α=0.01, **410 / 1252 units (33%)** have zero folds that produced any
non-zero beta — they can **never** be non-zero-lag under any criterion. Poisson never zeroes a
coefficient, so his pool has no such ceiling.

Median per-neuron `alpha_max` on PFC is **0.00871**, i.e. the paper's stated α sits above the
median neuron's entire regularization path.

Shrinkage **dilutes but does not explain** the pass-rate gap, and a direct test rules it out
as the cause. Re-fitting three PFC recdays across α:

```
alpha setting               med betas   NZ30 rate   NZ90 rate
fixed 0.01  (paper/code)          1.6       27.8%        2.5%
fixed 0.001                      23.6       20.4%        0.0%
fixed 0.00025 (== rate)          29.5       22.2%        0.6%
relative 0.1*alpha_max           22.2       26.5%        0.0%
PAPER                                       67.3%       45.8%
```

Lowering α makes the betas **15× denser and the pass rate goes down**, and our rates sit far
below even random placement (58% / 12.5% for three randomly placed betas). The cause
is that our betas are concentrated at lag 0, not that there are too few of them.

**But a fixed α is a firing-rate filter, and that is a defect of the reference, not of the
reproduction.** `alpha_max ∝ sd(y)`, so which neurons get any beta at all is a function of rate:
on one fold of `ah04_01122021_02122021` 97% of the slowest rate quartile is zeroed against 59%
overall, and on the LEC recday `ly06_20250613_20250615` (all 6 folds) 80% of the slowest quartile
against 33% overall, with corr(all-zero, rate) = **−0.53**. V5's `y_scaling='zscore_recday'`
(per-neuron sd over the whole recday, `*_zscore_v5_*` run directories) takes those all-zero
fractions to 1% / 0.5% and the rate correlation to −0.14. Those runs are **ours, not the reproduction** — the published counts assume
raw counts at α = 0.01, and rescaling also admits very low-spike units (a 122-spike neuron's
`alpha_max` goes 0.0005 → 0.0175). Reported separately, with `n_spikes_recday` alongside. See
the `y_scaling` section of [`ELASTICNET_V5.md`](ELASTICNET_V5.md).

#### The Poisson link — paper vs code

| | paper | his code | V4 |
|---|---|---|---|
| link | "linear–nonlinear–Poisson … **logarithmic** link function" | log link **at fit** | same |
| **prediction** | implied `exp(Xβ + b)` | **`np.sum(X*β)` — no `exp`, no intercept** | `poisson_link` selects |
| α / `max_iter` | 1 / unstated | 1 / sklearn default **100** | 1 / **1000** |
| sign constraint | unstated | none — `PoissonRegressor` has no `positive` | none |

There are **zero** occurrences of `np.exp`, `.predict(` or `.intercept_` anywhere in his
notebook. So the Poisson variant's stated purpose — robustness to the linearity assumption — is
**discarded at readout**: the correlation is computed against the linear predictor, not the
modelled rate. For ElasticNet the missing intercept is harmless (Pearson is shift-invariant);
for Poisson the missing `exp` is not. And since `PoissonRegressor` has no `positive` parameter
while the ElasticNet call sets `positive=True`, his two "robustness" variants differ in the
**sign constraint as well as the link**.

V5 stores both readouts from one fit: `poisson_link='linear'` (his code, the default) and the
`exp(Xβ + b)` alternative (the paper), as `corrs*` and `corrs*_altlink`.

**Measured verdict (2026-09-07/08): at his α = 1 the link choice is cosmetic after all.** An
early synthetic check suggested otherwise; the full PFC runs settle it. L2 at α = 1 shrinks the
betas to a median max|β| of 0.007, over which `exp` is affine to first order, so the two
readouts give the same Pearson r — median |Δr| = 0.001 per neuron, and on the complete 25-recday
runs every panel agrees to 3–4 decimal places:

| PFC Poisson, his semantics | linear Xβ | exp(Xβ+b) | paper (ED 8d) |
|---|---|---|---|
| past, all state-tuned | 140, r +0.2733, t 12.09 | 140, r +0.2734, t 12.08 | 489, t 10.70 |
| past, 30° | 86, r +0.1546, t 3.34 | 86, r +0.1548, t 3.35 | 346, t 4.74 |
| past, 90° | 19, r +0.1503, t 1.24 | 19, r +0.1513, t 1.25 | 229, t 2.81 |
| future, all state-tuned | 114, r +0.2467, t 8.92 | 114, r +0.2469, t 8.92 | — |
| future, 30° | 68, r +0.0119, t 0.20 | 68, r +0.0116, t 0.19 | — |

So the missing `exp` is a real code/paper divergence with no numerical consequence at α = 1. It
would matter at a smaller α, where the betas are large enough for the nonlinearity to bite.

**What does matter is his companion gate.** `nanmean(prediction) > 0` (cell 26) drops **51% of
Poisson folds**, because `PoissonRegressor` has no `positive` constraint and half the linear
predictors come out negative on average. Combined with the every-fold rule, that collapses the
denominator: **140 (past) / 114 (future)** against **573 / 537** under V4 semantics. The paper's
489 is close to the *V4* denominator and nowhere near the every-fold one, so **ED 8d cannot have
been produced by applying the gate and the every-fold rule together** — one of the two is not in
the executed path.

Two side findings from those runs: Poisson reproduces the **30° panel better than ElasticNet**
does (t 3.34 on 86 units vs 2.48 on 246, against the paper's 4.74), and the **90° panel stays
unreproduced under Poisson too** (19 units, t 1.24 vs 229, t 2.81) — the same failure as ED 8a,
now shown for both estimators.

#### An inconsistency to document, not guess

His `Neuron_raw` is **integer spike counts** (values 0–7 per 25 ms bin, mean 0.096) — verified
directly. At that scale α=0.01 zeroes 33% of our units outright. Yet he reports **n = 489 under
both ElasticNet (Fig 5h) and Poisson (ED 8d)**. Poisson never zeroes a coefficient, so 489 is
his state-tuned count; the ElasticNet panel reporting the same 489 means his
"(with non-zero beta coefficients)" filter removed **nobody**. Either the Fig 5h caption reuses
the Poisson n, or his pipeline scaled `y` somewhere not visible in the notebook. We cannot tell
from here.

#### Where the betas actually sit — THE LIVE OPEN QUESTION

Largest beta per fitted (neuron, fold) pair, PFC past, 4,433 pairs:

```
lag  0  37.8%   <- excluded (30deg)        uniform would be 8.3%
lag  1  11.3%   <- excluded (90deg)
lag  2   8.1%   <- excluded (90deg)
lag  3-8  ~4.5% each (flat floor)
lag  9   3.9%   <- excluded (90deg)
lag 10   4.8%   <- excluded (90deg)
lag 11   6.5%   <- excluded (30deg)

in {0,11}:            44.3%  (uniform 16.7%)
in {0,1,2,9,10,11}:   72.4%  (uniform 50.0%)
```

**Most PFC units in our fits are best explained by the animal's current position**, and the
non-zero-lag criterion rejects them — correctly. Pass rates follow directly: 25% for 30°
(his 67%) and 3.9% for 90° (his 46%).

His pass rates require a peak-lag distribution ours does not show. This is **upstream of every
criterion choice** in this document and is the thing to resolve next. Leading suspect: the
`Phases_raw2_*` phase segmentation (§3.8) — if his differs from our linear thirds, the lag axis
means something different and the whole distribution shifts. Not testable without the OSF
intermediates.

### 3.4 Preferred phase from the held-out session

His cell 21 reads `tuning_phase_boolean_max[ses_ind_actual]` — the **held-out** session — and
uses it for both the fit and the scoring: the held-out session chooses which bins the held-out
score is averaged over. That is leakage.

**The paper never says which.** It states only that analyses were done "in the preferred
goal-progress bin of each neuron"; train versus test is not specified anywhere. V4's default is
now `pref_phase_source='test'` for comparability with the published numbers, with `'train'` one
flag away — and the per-recday summary prints a leakage warning whenever `'test'` is active.

### 3.5 Twelve lags or twenty-four — RESOLVED: twelve

Cells 15/21/26 are set to `limited=True` (12 lags) while cells 32/38 — which load results and
`savefig` the histograms — are set to `limited=False`, reading the 24-lag `_beyond` files. That
looked like the published figure being a 24-lag run.

**The paper settles it as 12.** Methods: *"A total of 9 × 3 × 12 … regressors … 12 lags in task
space from the anchor (4 states × 3 goal-progress bins)"*, and *"We trained the model on five
(training) tasks"*. So `limited=False` in cells 32/38 is leftover state. Hypothesis retired.

(The paper writes "9 × 3 × 12 (**312**)"; the arithmetic is 324.)

### 3.6 The state-tuning subsetting rule — the denominator

The *test* matches ours exactly (§2). The **subsetting** does not:

> "we restricted analyses to neurons **state-tuned in more than one-third of the recorded
> tasks**. This subsetting is used throughout the manuscript where state-tuned cells are
> investigated."

V4 originally OR-ed across sessions — tuned in ≥1 task — which passes 95% of units. Measured on
the PFC past sweep:

```
rule                                  n state-tuned    & non-zero betas & valid r
>= 1 task  (the old OR)                      1191                            824
> 1/3 of tasks  (HIS RULE)                    774                            590
>= 1/2 of tasks                               757                            576
all tasks                                      34                             30
                                     PAPER:    489
```

His rule closes most of the gap and is now the V4 default (`state_tuning_min_fraction=1/3`).
It also helps on the leg-duration confound: it requires consistency across tasks rather than one
lucky session, and it drops the pure-noise state-tuned rate from ~60% to 43%.

### 3.7 Smaller differences, measured and negligible

| | effect |
|---|---|
| `require_positive_top3` (V4 adds it; guards argsort tie-breaking among exact zeros) | 156 vs 157 neurons |
| per-fold majority vote vs fold-average mask | 166 vs 157 |
| state-tuning statistic (`max` reference vs duration-invariant `mean`) | 95% vs 97% pass; counts differ by ≤ 2 |
| `drop_untracked_bins` (V4 addition) | **no-op on PFC** — his `Location_raw` uses real NaN, exact-0 fraction 0. It exists for an LEC-only bug |

### 3.8 Untestable here — RESOLVED in §0

Written before the GitHub repository was downloaded. The cells that build `Phases_raw2_*`,
`tuning_phase_boolean_max_*` and `State_95*` are in `Figure2.ipynb`; §0.1 shows the deposited
phase array is 5-bin but the executed pipeline was 3-bin (Fig 5g raster), §0.3–0.4 give the
tuning definitions. The intermediates themselves are on neither OSF nor Ceph.

---

## 4. The reconciliation table

PFC, past lags, the `nonzero` (30°) quantity the paper reports. Re-scored from the **existing**
betas — no re-fit — by `elgaby_figure5.py`. Validated: the re-derived test regressors reproduce
V4's stored `corrs_nonzero` to **5.3e-07** across all 25 recdays (float32 storage precision).

| lag set | semantics | n | mean r | frac > 0 | p |
|---|---|---|---|---|---|
| **{0,11}** (reference) | per-neuron mask | **287** | **+0.087** | 63% | 5e-06 |
| **{0,11}** (reference) | **El-Gaby per-fold** | **510** | +0.054 | 58% | 0.003 |
| {0,1,11} (V4 default) | per-neuron mask | **155** | +0.072 | 61% | 0.006 |
| {0,1,11} | El-Gaby per-fold | 390 | +0.018 | 53% | 0.41 |
| {0,1,2,9,10,11} (strict) | per-neuron mask | 33 | +0.034 | 58% | 0.56 |
| {0,1,2,9,10,11} (strict) | El-Gaby per-fold | 167 | +0.014 | 49% | 0.70 |
| — | — | **329** | *right-shifted* | | |

All six rows are the same quantity — his `nonzero` correlation, i.e. the prediction rebuilt with
that lag set removed — differing only in which neurons are admitted. All 25 recdays, 1252 units.

**The count is explained by the criterion.** 271 and 486 bracket 329, and the lag set alone
accounts for most of the gap — your hypothesis was right.

**The right-shift is not.** The best any criterion reaches is **mean r = +0.087, 63% > 0**
({0,11}, per-neuron mask) — highly significant at n=287 but nothing like a markedly
right-shifted distribution. And matching his semantics more closely makes it *worse*
(+0.054, 58%), because admitting a neuron on the strength of a single surviving fold admits a
lot of noise. So the remaining gap is in the **fit**, not the selection.

### Why a single fold's r is nearly uninformative

Each r is a Pearson correlation over **4 points**, whose null sampling distribution has
SD = 1/√3 = **0.577** — |r| exceeds 0.5 in 50% of pure-noise draws and 0.9 in 10%. Measured
fold-to-fold spread within a selected neuron is SD 0.55 (LEC) / 0.56 (PFC), i.e.
indistinguishable from sampling noise, and 96% of selected PFC neurons have folds that straddle
zero. Averaging unbiased noisy estimates is legitimate, but it means his ≥1-fold rule admits
neurons on the basis of a statistic that is almost pure noise.

---

## 5. What was and was not closed

**Closed.**

| question | answer |
|---|---|
| lag set | **{0,11}**, stated in the paper as "30° or more". 155 → 287, effect size 0.223 → 0.274 |
| the 24-lag `_beyond` variant | not the published one; the paper says 12 lags explicitly |
| state-tuning *test* | ours already matches his methods line for line |
| state-tuning *subsetting* | ">1/3 of tasks", not the OR. 1191 → 774, and 824 → 590 against his 489 |
| is our distribution less right-shifted? | **No.** Effect size 0.490–0.494 vs his 0.421 (left panel), 0.274 vs 0.215 (right) |
| per-fold vs per-neuron semantics | he counts a neuron on ≥1 surviving fold; that inflates n to 510 and *lowers* the effect size to 0.133 |

**Not closed at the time — now closed by §0:**

* the residual **590 vs 489** state-tuned count → his statistic is the preferred-phase mean with
  NaN propagation (736 vs 738 state-tuned), and "with non-zero beta coefficients" means finite
  in every fold (481 vs 489).
* the **phase definition** → 3-bin in the executed pipeline (Fig 5g stripe); the deposited 5-bin
  cell was never what ran.
* the n=489-in-both-panels puzzle → 489 is the finite-in-every-fold pool; the Poisson caption
  reusing it remains unexplained but is a caption question, not a pipeline one.

**Still open:** the 90° panel (79–92 vs 224), see §0.7.

`elgaby_ladder.py` remains available for the one-factor-at-a-time sweep (Poisson →
`poisson_link='log'` → `pref_phase_source='train'`) now that the criterion questions are
settled; it is no longer needed to answer the original question.

## 6. What this means for the LEC result

The original worry — that our effect was weaker than the published one — **does not hold**. On
matched quantities we reproduce it and slightly exceed it, on his own data.

What the investigation did establish is that the *reported number is highly sensitive to the
selection*, in ways that have nothing to do with the biology:

```
same betas, same data, PFC past lags:
  {0,11}   per-neuron mask     n=287  effect size 0.274
  {0,11}   El-Gaby per-fold    n=510  effect size 0.133
  {0,1,11} per-neuron mask     n=155  effect size 0.223
  90-deg   per-neuron mask     n=33   effect size 0.15
```

So report the criterion with the number, always. V4 now emits all three lag levels every run so
the sensitivity is visible rather than a choice made once and forgotten.

Two standing caveats from [`ELASTICNET_V4.md`](ELASTICNET_V4.md) were said to survive: the
**leg-duration confound** and the **α=0.01 firing-rate cut**. **Correction (§0.3):** the
duration confound belongs to the paper's *text* statistic, which V4 implemented; his code uses
the preferred-phase mean, which is not confounded that way, and V5 adopts it. The α cut does
remain, and his NaN-propagating test adds a second firing-rate filter of its own (14.6% of LEC
neuron-sessions).
