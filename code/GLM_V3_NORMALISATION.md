# GLM V3 — the firing-normalisation ladder (sanity check)

`w1_refit.NORMALISATIONS` · `run_glm_batch.py --normalise` · fits in
`data/glm_outputs/LEC_normalisation/` and `mFC_data/glm_outputs/PFC_normalisation/` · figures in
`data/figures/glm_v3_normalisation/` and `mFC_data/data/figures/glm_v3_normalisation/`

**Deliberately separate from `GLM_V3.md` and from the `glm_v3` output/figure folders**, at the user's
instruction: nothing here overwrites or mixes into the V3 results. The middle rung of the ladder *is*
the production V3 fit and is read from its existing location rather than refitted.

Asked for: "a version of GLM v3 where we z-score all neurons before fitting, z-scored across all
sessions within a recday … just for a sanity check". Clarified by the user to mean **variation across
sessions within a recday**, and "comparing within-session vs across-session z-scoring might be
informative".

## 1. Across-session z-scoring is not a model choice

A per-neuron z-score computed over the whole recday is **exactly a no-op** for every statistic V3
reports. CPD, Δr² and `r2_cv` are ratios of sums of squares of the *same* y, and the design carries an
intercept, so any per-neuron affine map `y → (y − μ)/σ` cancels: every RSS and TSS scales by 1/σ², and
the intercept absorbs −μ/σ.

Measured on `ah10_20250616_20250617` at the production V3 config (uniform/30), z-scoring `Neuron_raw`
per neuron with mean and SD pooled over the recday's GLM sessions, against the unmodified fit:

| statistic | max\|diff\| |
|---|---|
| `cpd_cv` | 8×10⁻¹⁵ |
| `delta_r2_cv` | 9×10⁻¹⁵ |
| `r2_cv` | 9×10⁻¹⁴ |
| `p_freedman_lane__cpd` | **exactly 0** |
| `null_mean_freedman_lane__cpd` | 3×10⁻¹⁵ |
| in-sample `cpd_results` | 1×10⁻¹⁵ |
| in-sample permutation F | 3×10⁻¹⁰ |
| `tss`, `rss_full` | 8×10⁵ — the expected σ² rescaling, cancels in every ratio |
| `glm_results` betas | rescaled by exactly 1/σ (neuron 0: ratio 2.17376 = 1/SD for all 56 non-intercept columns; the intercept absorbs the mean, so its ratio is −0.93) |

The β-profile and heatmap figures unit-normalise each neuron, so even they are unchanged. **Fitting the
12 V3 arms with this normalisation would have written bit-identical pickles.**

It also cannot reach the concern it was aimed at. A scale-free ratio cannot register a per-neuron or
per-recday scale difference, and z-scoring rescales signal and noise identically, so it leaves
signal-to-noise untouched — which is why it is no help with the ENTl-sup rate confound either. What
actually drives recday-to-recday variation in these statistics is trial count (measured ρ = +0.745 with
fit quality, against −0.136 for neuron count) and the rate-dependent Poisson noise floor.

## 2. What does distinguish models: per-session offset and per-session gain

The only normalisation choices that change anything are whether each **session** gets its own offset
and/or its own gain. That is a three-rung ladder, and the production setting is the middle rung.

| rung | per-session offset | per-session gain | flags | section token |
|---|---|---|---|---|
| `none` (**≡ across-session z-scored**) | no | no | both False | `_nonorm` |
| `session-centre` (**production**) | **yes** | no | `cv_center_within_sessions=True` | *(none)* |
| `session-z` | yes | **yes** | `cv_zscore_within_sessions=True` | `_zscored` |

`session-z` is not a neutral normalisation: it makes the CV test tuning **shape up to a per-session
scale** rather than shape and gain together. Its motivation is that RSS is pooled across
leave-one-session-out folds, so without it a high-variance session dominates the pooled CPD.

**Only `cv_results` is affected.** The in-sample loop fits raw `FR_all`, so `glm_results`,
`cpd_results` and `permutation_results` — and therefore every β profile, every sorted heatmap and the
whole W5 β route — are identical across all three rungs.

## 3. What the ladder does to `r2_cv`, and a mechanism I got wrong

On the smoke-test recday (`ah10_20250616_20250617`, 154 neurons, medians over neurons):

| rung | TSS | RSS_full | RSS_full/TSS | `r2_cv` |
|---|---|---|---|---|
| `none` | 29225.4 | 26625.5 | 0.9671 | **+0.0329** |
| `session-centre` (production) | **29225.4** | 26139.5 | 0.9423 | +0.0577 |
| `session-z` | 27444.0 | 25798.9 | 0.9401 | +0.0599 |

**I first wrote that the free per-session offset shrinks TSS by removing between-session variance, and
that this is where the r² difference comes from. That is wrong, and the table refutes it: TSS is
identical between `none` and `session-centre` to the digit.** The reason is that `cv_scores` computes
`tss += sum((Y_te - Y_te.mean(axis=0))**2)` — the held-out fold's **own** mean, in every rung — and
under leave-one-session-out the fold *is* a session, so per-session centring is already baked into the
denominator and cannot change it. The entire r² difference comes from `RSS_full`: a model trained on
un-centred training sessions carries their average level, which is mismatched to the held-out
session's level, and the centring removes that mismatch.

That also means my "43 % of r² comes from information leaked out of the held-out fold" was the wrong
framing. Because TSS always uses the held-out session's own mean, **`r2_cv` was never an
absolute-prediction statistic in any rung.** What the rungs differ on is whether the *model's
predictions* are also freed from the between-session level:

- `none` penalises the model for a between-session level error while the denominator does not credit
  it — so it is a mixed statistic, and its lower r² is that penalty, not a more honest number;
- `session-centre` removes the penalty, making numerator and denominator ask the same question;
- `session-z` additionally frees the per-session gain, which buys very little (0.0577 → 0.0599 here).

So the correct statement is narrower than the one I first wrote: production `r2_cv` is
variance-about-the-session-mean explained by a model that is not asked to get the session's level
right, and the ladder measures what requiring the level would cost.

**And the smoke recday understated that cost by a lot.** On the full 25 recdays with the mouse chain
(§5.3), `r2_cv` is 0.0097 without per-session centring against 0.0417 with it in LEC — a **4.3×
difference, i.e. 77 % of production `r2_cv`** — and 0.0104 vs 0.0497 in PFC (**79 %**). The fraction of
neurons with `r2_cv` > 0 falls from 82 % to 58 % (LEC) and 87 % to 61 % (PFC). `ah10_20250616_20250617`,
where the gap was 0.033 vs 0.058, is simply one of the better recdays; quoting 43 % from it was
premature and is superseded by the numbers in §5.3.

**On the CPDs, the honest position is "small in absolute terms, not necessarily small in relative
terms", and the full run must settle it.** Per-neuron CPD medians moved by ≤ 1×10⁻⁴ across the ladder,
but the per-regressor ΔRSS medians moved by up to 42 % (`none` → `session-centre`: place ×1.02,
`goal_progress` ×1.06, `speed` ×0.83, `acceleration` ×1.42, `time_from_reward` ×1.28). Absolutely
those are changes of ~10⁻⁴ in Δr², which is negligible against place at 0.026 — but the
`goal_progress`-versus-`time_from_reward` comparison that GLM V3 turns on is precisely a comparison of
two numbers of that size. §5 therefore reports **relative as well as absolute** changes, and checks
whether the gp-vs-tfr ordering and the regional contrast survive, rather than declaring stability from
the medians.

## 4. Scope and mechanics

Two new rungs × the two primary arms × both datasets = 8 fits, 200 SLURM jobs, submitted 2026-09-08:

- `core_progress_time`, cap 30 s, tfr **uniform** — the arm W5 selects goal-progress cells on;
- `core_progress_only`, cap 30 s — the "within-leg structure of any kind" arm.

Everything else is the production V3 configuration (250 ms binned, mixed reference coding, LOSO CV,
Freedman–Lane at `n_perm=100`, 30 s leg cap). Reproduce with, e.g.:

```bash
bash sbatch_files/submit_glm_lec.sh --section core_progress_time --regset matched --leg-cap-s 30 \
  --tfr-scheme uniform --normalise session-z --width-ms 250 --scheme decile \
  --permutations 100 --cv-perms 100 \
  --shard-dir data/glm_outputs/LEC_normalisation_shards --out-dir data/glm_outputs/LEC_normalisation
python code/run_glm_batch.py --merge <same flags>
```

Implementation notes:

- `--normalise` is stamped into the shard config **only when non-default**, so every stamp already on
  disk stays byte-identical and a later production shard still merges with the existing ones.
- The section token must stay `[A-Za-z0-9_.]`. The originally requested `_z-scored` **fails**
  `recday_registry._POST_REFIT_SECTION`, which silently re-arms `STALE_CACHE_RECDAYS` and drops
  `ly05_20250618_20250619` from every load with only a printed warning (verified). Hence `_zscored`.
- **Equivalence control.** `--normalise session-centre` reproduces the fit already on disk:
  `max|diff|` 9×10⁻¹⁶ on `cpd_cv`, 7×10⁻¹⁶ on `delta_r2_cv`, 3×10⁻¹⁶ on `r2_cv`, **0** on `tss`. The
  refactor did not move production.

## 5. Results

25/25 recdays per fit, 0 failures, key-contiguity PASS. Mouse chain throughout (per neuron → recday
median → mouse mean → mean over mice). Figures: `*_norm_ladder_*.pdf` in the two
`glm_v3_normalisation` directories.

### 5.1 `session-centre` (production) vs `session-z` — the comparison that was asked for

**Every conclusion holds.** The direction of change is consistent — `session-z` lifts `r2_cv` and most
CPDs by a few per cent — and nothing reorders, changes sign, or crosses a significance boundary.

| | LEC gp+tfr uniform/30 | LEC gp-only/30 | PFC gp+tfr uniform/30 | PFC gp-only/30 |
|---|---|---|---|---|
| `r2_cv` | 0.0417 → 0.0426 | 0.0408 → 0.0418 | 0.0497 → 0.0509 | 0.0477 → 0.0492 |
| neurons with `r2_cv` > 0 | 82.4 → 84.1 % | 83.6 → 84.6 % | 87.3 → 88.8 % | 87.9 → 89.3 % |
| CPD place | 0.01528 → 0.01583 | 0.01560 → 0.01611 | 0.00733 → 0.00765 | 0.00719 → 0.00747 |
| CPD `goal_progress` | 0.00047 → 0.00055 | 0.00171 → 0.00179 | 0.00412 → 0.00418 | 0.00872 → 0.00906 |
| CPD `time_from_reward` | −0.00039 → −0.00035 | — | 0.00059 → 0.00049 | — |
| corrected CPD `goal_progress` | 0.00171 → **0.00175** | 0.00294 → 0.00297 | 0.00509 → 0.00511 | 0.00971 → 0.01003 |
| corrected CPD `time_from_reward` | 0.00123 → **0.00123** | — | 0.00192 → 0.00184 | — |
| frac_sig `goal_progress` | 0.542 → 0.543 | 0.654 → 0.668 | 0.747 → 0.757 | 0.828 → 0.839 |
| frac_sig `time_from_reward` | 0.455 → 0.457 | — | 0.511 → 0.515 | — |
| **gp − tfr CPD** | **+0.00086 → +0.00091** | — | **+0.00353 → +0.00369** | — |

Four things worth stating explicitly:

1. **Per-neuron ranks are preserved**: pooled Spearman of CPD against production is **0.975–0.994** for
   every regressor in every arm and both datasets. So it is not only the aggregates that survive — the
   same neurons come out on top.
2. **The bias-corrected CPDs are more stable than the raw ones**, because the Freedman–Lane null centre
   shifts with the observed value. `time_from_reward` in LEC is identical to five decimals
   (0.00123 → 0.00123) where its raw CPD moved 10 %. Since `GLM_V3.md` §9 headlines the corrected
   variant, the headline numbers are the stable ones.
3. **The §3 worry was justified in magnitude but changes nothing.** Raw CPDs of the small regressors do
   move by up to ~17 % relative (LEC `goal_progress` 0.00047 → 0.00055), which is why I refused to
   declare stability from the medians. But both survivors move the same way, so the gp-vs-tfr contrast
   — the comparison V3 turns on — is unchanged in sign and grows by 6 % (LEC) and 5 % (PFC).
4. **PFC's headline survives**: in the gp-only arm `goal_progress` remains PFC's largest regressor
   (0.00906) and still exceeds place (0.00747), significant in 84 % of neurons.

### 5.2 LEC by region, and the primary contrast

Per-region CPDs agree to 3–4 decimal places and the ordering is identical in every regressor — place
SUB/ProS > CA1/HPF > ENTl-deep > ENTm > ENTl-sup; `goal_progress` in the gp-only arm ENTl-deep (0.0035
→ 0.0037) highest, ENTl-sup (0.0007 → 0.0008) lowest.

ENTl-deep − SUB/ProS (raw CPD, mouse bootstrap), production → `session-z`:

| regressor | production | `session-z` |
|---|---|---|
| `time_from_reward` | +0.00092 [+0.0007, +0.0012] | +0.00075 [+0.0006, +0.0010] |
| place | −0.01115 [−0.0220, −0.0003] | −0.01128 [−0.0220, −0.0005] |
| `goal_progress` | +0.00059 [−0.0008, +0.0021] | +0.00047 [−0.0009, +0.0019] |
| speed | −0.00058 [−0.0040, +0.0022] | −0.00064 [−0.0041, +0.0022] |

So the two regional statements V3 makes are both robust: the ENTl-deep > SUB/ProS
`time_from_reward` difference keeps its sign and a CI clear of zero (it weakens by 18 %), and
`goal_progress` has no regional difference under either normalisation. `place` being *lower* in
ENTl-deep than SUB/ProS also survives with a CI clear of zero.

### 5.3 The full ladder on the gp+tfr uniform/30 arm

25/25 recdays on every rung, both datasets. This is the arm W5 selects goal-progress cells on.

| | LEC `none` | LEC production | LEC `session-z` | PFC `none` | PFC production | PFC `session-z` |
|---|---|---|---|---|---|---|
| `r2_cv` | **0.0097** | 0.0417 | 0.0426 | **0.0104** | 0.0497 | 0.0509 |
| neurons `r2_cv` > 0 | **57.7 %** | 82.4 % | 84.1 % | **60.9 %** | 87.3 % | 88.8 % |
| CPD place | 0.01354 | 0.01528 | 0.01583 | 0.00730 | 0.00733 | 0.00765 |
| CPD `goal_progress` | 0.00037 | 0.00047 | 0.00055 | 0.00363 | 0.00412 | 0.00418 |
| CPD `time_from_reward` | −0.00029 | −0.00039 | −0.00035 | 0.00031 | 0.00059 | 0.00049 |
| **corrected** CPD place | 0.01865 | 0.01850 | 0.01892 | 0.01135 | 0.00964 | 0.00997 |
| **corrected** CPD `goal_progress` | 0.00166 | 0.00171 | 0.00175 | 0.00498 | 0.00509 | 0.00511 |
| **corrected** CPD `time_from_reward` | 0.00139 | 0.00123 | 0.00123 | 0.00199 | 0.00192 | 0.00184 |
| frac_sig `goal_progress` | 0.506 | 0.542 | 0.543 | 0.696 | 0.747 | 0.757 |
| **gp − tfr CPD** | +0.00066 | +0.00086 | +0.00091 | +0.00332 | +0.00353 | +0.00369 |

**The headline result of the whole exercise.** `r2_cv` varies by a factor of 4–5 across the ladder,
while the **bias-corrected CPDs — the statistic `GLM_V3.md` §9 headlines — vary by at most 13 %** and
never change sign or order. `goal_progress` corrected CPD spans 0.00166–0.00175 in LEC and
0.00498–0.00511 in PFC; the gp-vs-tfr contrast keeps its sign and magnitude on every rung. So the V3
conclusions do not rest on the normalisation choice, whereas any statement about `r2_cv` does.

Two honest qualifications:

1. **`none` genuinely reorders neurons.** Pooled per-neuron Spearman of CPD against production is
   0.78–0.96 under `none` (LEC `time_from_reward` 0.775, PFC place 0.837) against 0.975–0.994 under
   `session-z`. So while the aggregates survive, "which neurons are the tuned ones" is not fully
   preserved when the model must also predict the session level.
2. **One regional claim is not robust.** ENTl-deep − SUB/ProS on `place` is −0.0094 [−0.0213, +0.0026]
   under `none`, so its CI includes zero, where production and `session-z` both exclude it
   (−0.0112, −0.0113). By contrast the `time_from_reward` regional difference holds on **all three**
   rungs — +0.00112 [+0.0008, +0.0015], +0.00092 [+0.0007, +0.0012], +0.00075 [+0.0006, +0.0010] — and
   `goal_progress` has no regional difference on any rung. Per-region CPD orderings are identical
   throughout.

### 5.4 The full ladder on the gp-only/30 arm

25/25 recdays on every rung, both datasets. Same picture, and it is the arm where `goal_progress`
carries the most variance, so it is the strongest test of the headline.

| | LEC `none` | LEC production | LEC `session-z` | PFC `none` | PFC production | PFC `session-z` |
|---|---|---|---|---|---|---|
| `r2_cv` | **0.0088** | 0.0408 | 0.0418 | **0.0091** | 0.0477 | 0.0492 |
| neurons `r2_cv` > 0 | 57.7 % | 83.6 % | 84.6 % | 59.8 % | 87.9 % | 89.3 % |
| CPD `goal_progress` | 0.00152 | 0.00171 | 0.00179 | 0.00801 | 0.00872 | 0.00906 |
| **corrected** CPD `goal_progress` | 0.00277 | 0.00294 | 0.00297 | 0.00907 | 0.00971 | 0.01003 |
| **corrected** CPD place | 0.01900 | 0.01887 | 0.01928 | 0.01125 | 0.00949 | 0.00975 |
| frac_sig `goal_progress` | 0.634 | 0.654 | 0.668 | 0.803 | 0.828 | 0.839 |

`goal_progress` corrected CPD spans 0.00277–0.00297 in LEC (7 %) and 0.00907–0.01003 in PFC (11 %)
across a ladder over which `r2_cv` varies 4.6–5.4×. **PFC's headline — `goal_progress` larger than
place in this arm — holds on every rung**: 0.0080 vs 0.0069 (`none`), 0.0087 vs 0.0072 (production),
0.0091 vs 0.0075 (`session-z`), significant in 80–84 % of neurons throughout.

Per-neuron rank correlation with production is again the place where `none` differs most: 0.84–0.98
under `none` (PFC place 0.842) against 0.986–0.994 under `session-z`.

Regionally, `goal_progress` in LEC keeps its ordering on every rung — ENTl-deep highest
(0.0033 / 0.0035 / 0.0037), ENTl-sup lowest (0.0006 / 0.0007 / 0.0008) — and no primary-contrast CI
excludes zero on any rung in this arm, including `place`, which is consistent with §5.3's finding that
the `place` regional difference is the fragile one.

### 5.5 Runtime

200 jobs, 0 failures, elapsed min/median/max **0.2 / 2.4 / 12.2 min**. The `none` rung is the cheapest
(no normalisation pass) and PFC the cheapest dataset. Wall clock was ~1 h 50 m rather than the ~50 m
the same job count took on 2026-09-07, because the `cpu` partition had only 14 of 32 nodes usable and
the LEC jobs' 64 GB request restricts them to the larger-memory nodes.

## 6. Verdict

**The sanity check passes.** Across a normalisation ladder on which `r2_cv` varies by a factor of
4–5, every GLM V3 conclusion is unchanged: the bias-corrected CPDs move by ≤ 13 %, nothing changes
sign or order, the goal-progress-versus-absolute-time contrast keeps its sign and magnitude in both
datasets and both arms, PFC's `goal_progress` > place headline holds on every rung, and the
significant fractions move by ≤ 0.04. Nothing here recommends changing the production setting.

Three things the exercise produced that were not asked for and are worth keeping:

1. **`r2_cv` is almost entirely a per-session-offset statistic.** Without per-session centring it is
   0.009–0.010 rather than 0.041–0.050, and the fraction of neurons above zero falls from ~85 % to
   ~58 %. Any sentence of the form "the model explains x % of held-out variance" needs that context.
2. **CPD is the robust statistic and `r2_cv` is not** — the W1 slogan, now quantified on the reduced
   design over a much wider range of normalisations than W1 tested.
3. **The `place` ENTl-deep vs SUB/ProS difference is the fragile regional claim** (CI includes zero
   under `none`), while the `time_from_reward` one holds on all three rungs. That is a distinction
   `GLM_V3.md` §9.7 could not make.

## 7. What this does and does not license

- It does **not** license re-reading any V3 number. `GLM_V3.md` §9 stands as fitted.
- Across-session z-scoring is settled as a no-op; do not fit it again.
- `r2_cv` is variance about the held-out session's own mean, in every rung — `cv_scores` centres TSS per
  fold regardless of the normalisation (§3). It is not an absolute-prediction statistic and never was;
  the rungs differ only in whether the model's predictions are also freed from the session level.
- A CPD that is stable across the ladder is robust to session-level offset and gain heterogeneity. That
  is a narrower claim than "robust to session heterogeneity": it says nothing about sessions differing
  in *tuning*, which is remapping and is W3's subject.
- `session-z` is not a better model, only a different question. Nothing here recommends changing the
  production setting.
- **No rung on this ladder measures absolute-level prediction across tasks**, and it is worth being
  explicit that the question is unavailable rather than answered. `cv_scores` defines TSS about the
  held-out fold's own mean, so "could the model have predicted this session's firing level at all?"
  would need a different denominator (TSS about the *training* mean) — a change to the CV engine, not a
  normalisation flag. Each rung is internally consistent: `session-centre` and `session-z` compute
  numerator and denominator on the same transformed y, and `none` is the one that is *not* consistent,
  since it charges the model for a level error in RSS that TSS gives it no credit for. If the
  cross-task stability of absolute rate is ever the question, add the alternative TSS rather than
  reading it off `none`.
