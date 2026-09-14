# W1 — The production GLM

**Status: complete. A refit with the corrected null is running (50 SLURM jobs, launched
2026-09-02).** Read `README.md` first.

---

## 1. What changed from the old pipeline, and why

Five things, each measured rather than assumed. All are in `code/w1_encoding_diagnostics.py`,
reproducible with `python code/w1_encoding_diagnostics.py --only <name>`.

### 1.1 Binned aggregation, not strided subsampling — the biggest single effect

`downsample_session_data` did `arr[:, ::factor]` — strided subsampling. At `factor=10` it kept
every 10th 25 ms bin and **discarded 90% of the spikes**, without widening the bins. Aggregating
instead (sum counts, average continuous behaviour, mode for categoricals, circular mean for HD):

| | stride | bin |
|---|---|---|
| counts/sample | 0.18 | 1.81 |
| zero bins | 85.5% | 54.8% |
| **held-out R²** | **+0.0009** | **+0.0411** |

**47×.** Same sample count and spacing, so autocorrelation is unchanged — the extra signal is
free. Five of ten regressors flip negative→positive. Controlled by `downsample_mode='bin'`.

### 1.2 Cross-validation — leave-one-session-out

The old path fitted `lstsq(X, frs)` over all samples and computed RSS, R², CPD and the nested F
on the same samples. On identical pure-noise data: in-sample CPD **+0.002**, held-out
**−0.003**. Sessions are the fold unit because they are different *tasks*, so LOSO tests
across-task generalisation.

A **nested F is not available** under CV — held-out ΔRSS can be negative and there is no F
distribution for it. Significance comes from a permutation null instead (§5).

### 1.3 Head direction was noise in every previous fit

`HD_raw` is `(T, 2)` = `[back2mid_deg, earL2earR_deg]`. The code did `HD_raw.flatten()`,
interleaving the columns into a `2T` vector; `truncate_all_arrays` then cut to `T`. So the GLM
received **half the session, interleaved with a ~90°-rotated copy of itself**, misaligned from
sample 2 on. All 36 HD columns in every LEC fit before this were noise.

Verified before fixing (187 sessions): `earL2earR − back2mid` has median |difference| 84.8° and
98.3% of samples within 25° of ±90°, so column 0 is `back2mid_deg` as documented. Fixed in
`_extract_head_direction`, mirrored to `mFC_data`.

### 1.4 Mixed reference coding — the full set *and* interpretable betas

`reference_coded` used to **raise** on any single-column regressor (the pokes), forcing
poke-containing fits onto rank-deficient `all_bins`. That was over-conservative. Reference-code
the *multi-column* blocks and pass *single-column* blocks through untouched:

| design | cols | rank | deficiency |
|---|---|---|---|
| all_bins + pokes | 97 | 89 | 8 |
| **mixed reference coding** | **89** | **89** | **0** |

The pokes never caused the deficiency — they don't sum to 1 — and never needed a reference bin.
Guard: drop zero-variance columns (a recday with no unrewarded pokes yields an all-zero one).

### 1.5 The transition filter — what nearly went wrong

`filter_correct_paths=True` keeps only **4–16% of transitions**. Combined with a `'all'`
aggregation on `valid_transition_mask` (requiring 10 consecutive valid raw bins), it produced
design matrices of **376–3471 rows against 123–160 columns** and held-out R² of −0.09 to −2.13.
Every job "succeeded".

Fixed three ways: `filter_correct_paths=False`; `max_transition_seconds=60`; mask aggregation
`'all'` → `'majority'`. And a **samples-per-parameter floor** (`min_samples_per_param=10`) now
refuses such a design outright — the old `X.shape[0] < X.shape[1]` check only caught designs
that could not be fitted *at all*.

The 60 s leg cap costs **2.6% of legs but 22.4% of samples** — the legs it removes are the
longest. Median leg is 9.38 s; p99 is 115 s and the max is 771 s, which is a disengaged animal
rather than a leg. **It is a behavioural selection, not a neutral one** — it removes the slowest
legs, so if a population is more active when the animal is disengaged this is not innocent.

## 2. Production configuration

**Linear/Gaussian, LOSO CV, binned aggregation, 250 ms, decile binning, 60 s leg cap.**

| dataset | design | section name |
|---|---|---|
| LEC | full-16 | `all_regressors__full_250ms_decile` |
| PFC | **matched-13** | `all_regressors__matched_250ms_decile` |

**PFC cannot fit the full set**: it has `HD_raw=None` and no poke tables, so 16 − 3 = **13**.
Since CPD is measured relative to the full model, **only the matched-13 pair licenses any
LEC-vs-PFC claim** — a design differing by 38 columns makes the two non-comparable. The PFC
notebook asserts both arms carry the same regressor groups and refuses to compare otherwise.

Poisson was dropped from production: on dense data it adds little (bin/lin R² +0.0170 vs
bin/pois D² +0.0065) and costs 2.3× more. It remains in `glm_cv.cv_scores_poisson`. Ridge does
not rescue anything — it pushes negatives toward zero, never into signal.

## 3. Two effect sizes, and the bias in both

- `cpd_cv = ΔRSS / RSS_reduced` — continuity with the existing corpus.
- `delta_r2_cv = ΔRSS / TSS` — **primary**: one denominator shared by every regressor, so groups
  are comparable and roughly additive toward `r2_cv`.

CPD overstates a weak regressor by up to **12×** when a dominant one stays in the model — though
on *this* data the two agree to within ~8%, because `r2_full` is small enough that
`RSS_reduced ≈ TSS`. Do not over-warn about it as I did.

**Both carry a downward bias ∝ the regressor's column count.** The full model's k extra
parameters fit training noise and cost ≈ k·σ² of held-out error, so a regressor explaining
nothing scores about −k·σ²/denominator, **not zero**. Measured `corr(n_cols, null centre) =
−0.838`, all 16 nulls below zero.

**Zero is not the reference — the null centre is.** The permutation null measures that penalty
per neuron (under permutation the regressor explains nothing by construction), so
`corrected = observed − null_mean` is unbiased. Correcting **moves 14 of 16 regressors**:
`head_direction` (35 columns, largest penalty) rises 4th → 2nd; `poke_rewarded` (1 column) falls
5th → 10th.

It removes the *parameter penalty*, not the **capacity advantage** — `corr(n_cols, corrected) =
+0.517` remains, and a 35-column block genuinely can capture more real structure.

`glm_plots.VALUE_OPTIONS = ('delta_r2_cv', 'cpd_cv', 'delta_r2_corrected', 'cpd_corrected')`.

## 4. Results (previous fit — effect sizes are stable, significance is not)

**LEC, bias-corrected Δr², 25 recdays / 5 mice:**

| | Δr² corrected |
|---|---|
| place | 0.01533 |
| **head_direction** | **0.00389** |
| speed | 0.00231 |
| poke_unrewarded | 0.00110 |
| time_from_reward | 0.00101 |
| … | |
| task_state | **−0.00015** (last) |

**PFC:** place 0.01095, speed 0.00569, time_from_reward 0.00252, acceleration 0.00203;
`task_state` and `progress_since_A` negative.

Two things worth noticing: **`poke_unrewarded` beats `poke_rewarded`** in LEC (poking a *wrong*
tower carries more unique variance than a correct one), and **`task_state` is last in both
datasets** — relevant to W3's prior.

**ENTl-sup is the region this GLM cannot explain** (median `r2_cv` −0.0042, vs SUB/ProS +0.0692,
ENTl-deep +0.0548). ah08 looks like a bad mouse only because it is 77% ENTl-sup — a composition
effect, not misalignment. Verified within ah08 (same sessions, same alignment): ENTl-deep +0.0447
vs ENTl-sup −0.0043, p = 8×10⁻¹⁹. Trial count also predicts fit (ρ = +0.745); neuron count does
not (−0.136).

Whether ENTl-sup *doesn't encode these variables* or is *too sparse to detect* is unresolved —
it fires at 1.07 Hz. The rate-matched comparison is the test.

## 5. The permutation null — the part that was wrong

**Do not quote `frac_sig` from fits that predate the running refit.**

The old null circularly shifted the **neuron's firing**. That destroys *every* regressor's
signal, so it tests the global hypothesis "nothing explains this neuron", not "does *g* add
anything beyond the others" — which is what a unique-variance measure asks. Under the shuffle,
residual variance σ² is inflated, so the k·σ² parameter penalty is inflated with it, the null
sits further below zero than the observed value, and the observed beats it almost always.

Measured on synthetic data where *g* is exactly null and the others carry real signal (correct
answer: 0.05):

| null | frac p<0.05, CPD | frac p<0.05, Δr² |
|---|---|---|
| shuffle the firing | 0.133 | **1.000** |
| **Freedman–Lane** | 0.067 | **0.050** |
| permute *g*'s columns | 0.067 | 0.067 |

All three keep full power (1.000 with real signal) — a specificity failure, not sensitivity.

**Freedman–Lane**, now the production null: fit the reduced model, circularly shift its
residuals within session, `y* = ŷ_red + e*`, run the whole CV on `y*`. The other regressors'
structure survives, so σ² matches the real fit and the parameter penalty matches too. Only *g*'s
relationship is destroyed.

A dead end worth recording so nobody repeats it: the null centre scaling with block size
(ρ ≈ −1.0) is **not** diagnostic — all three nulls do it, because it is the generic
cross-validated parameter penalty. It was the *calibration* test, not the centring test, that
identified the problem.

**Costs** (real LEC recday, 160 cols, 151 neurons, `n_perm=100`): shuffle ~72 min, Freedman–Lane
~246 min. FL is slower because it needs a different `y*` per regressor. An unexploited
optimisation: the rolled residuals do not depend on the fold, but the loop recomputes them
inside it — hoisting the model loop outermost would cut ~6× at ~1.6 GB peak.

## 6. Files

**Built this workstream:** `code/glm_cv.py`, `code/w1_refit.py`, `code/run_glm_batch.py`,
`code/w1_encoding_diagnostics.py`, `code/w1_cv_diagnostic.py`, `code/w1_smoke_test.py`,
`code/glm_plots.py`, `code/LEC_glm_production.ipynb`; mFC mirrors of all of these;
`sbatch_files/glm_{lec,pfc}.sbatch` + `submit_glm_{lec,pfc}.sh`.

**Modified:** both `glm_analysis_v2.py` copies (HD fix, `downsample_mode`, `continuous_binning`,
mixed reference coding, zero-variance guard, samples-per-parameter floor, CV hooks, `cv_nulls`),
`recday_registry.py` (`is_post_refit_section`), `time_vs_progress_dissociation.py`
(`downsample_mode` passthrough).

## 7. When the refit lands

```bash
python code/run_glm_batch.py --merge --section all_regressors --width-ms 250 \
    --scheme decile --regset full --nulls freedman_lane --cv-perms 100
python mFC_data/code/run_glm_batch.py --merge --section all_regressors --width-ms 250 \
    --scheme decile --regset matched --nulls freedman_lane --cv-perms 100
```

Then re-read both notebooks. New keys: `p_freedman_lane__delta_r2`,
`p_freedman_lane__cpd`, `null_mean_freedman_lane__*`, `null_p95_freedman_lane__*`. Legacy
`p_cv` / `null_mean` / `null_p95` still point at the first requested null for back-compat.

**Outstanding:** the `{uniform, decile} × {250, 500 ms}` configuration comparison was never run —
250 ms / decile was chosen as a default, not measured. `w1_encoding_diagnostics.py --only
binning` does the scheme arm; the width arm needs adding. Cheap now that a recday is minutes.
