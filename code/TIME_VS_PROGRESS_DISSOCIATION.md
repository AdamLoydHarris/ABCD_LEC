# Dissociating elapsed-time from goal-progress coding (LEC & PFC)

Companion documentation for `time_vs_progress_dissociation.py` (identical copy in
`code/` and `mFC_data/code/`) and the notebooks `LEC_time_vs_progress.ipynb` /
`PFC_time_vs_progress.ipynb`.

---

## 1. The question

For a temporally-tuned neuron in an ABCD inter-reward leg, does its firing track
**absolute elapsed time since the last reward** (a fixed-latency "time cell"), or
**goal progress** (how far through the current sub-goal)? "Goal progress" is
tested in two senses:

| contrast | frame A | frame B |
|----------|---------|---------|
| `temporal` | elapsed time `t` | **temporal phase** `t / D` (fraction of the leg in *time*) |
| `spatial`  | elapsed time `t` | **spatial progress** (fraction of the *distance* to the next goal) |

`D` = leg duration. These frames are **perfectly confounded if every leg has the
same duration**, so the whole method rides on interval-duration variability.

## 2. Why the existing GLM can't settle it

`glm_analysis_v2.py` already fits both frames as separate blocks
(`time_from_reward`/`time_to_reward` vs `goal_progress`) and partitions variance
with CPD / nested-F. But CPD is an **in-sample** split: when elapsed time and
phase are collinear within a leg, CPD divides the shared variance arbitrarily and
cannot say which frame the cell actually lives in. The decisive test is
**out-of-distribution generalization**, which this module adds.

## 3. The leverage in the data (measured, read-only)

Leg durations span a wide range, which is exactly what separates a fixed-latency
field from a phase field (a 3 s field sits at phase 0.7 in a 4 s leg but 0.13 in a
23 s leg):

| region | p10 | median | p90 | p90/p10 |
|--------|-----|--------|-----|---------|
| LEC | 4.3 s | 9.4 s | 27 s | 6.1× |
| PFC | 3.7 s | 7.3 s | 17 s | 4.7× |

Both regions are well-powered for the dissociation.

## 4. The method

### 4.1 Out-of-distribution (cross-condition) test — the headline

For each cell and each frame, fit on **short** legs and predict held-out **long**
legs, and the reverse; score by held-out pseudo-R². The per-cell statistic is

```
delta_ood = R2_abs - R2_phase     ( >0 → elapsed time ;  <0 → progress )
```

with a **leg-level bootstrap** CI. A cell is *classified* only if its CI excludes
0 **and** the winning frame explains positive held-out variance.

Crucial design choice: each frame's basis range is taken from the **training**
legs only, so the elapsed-time frame genuinely *extrapolates* onto held-out long
legs (a true time cell keeps its field at a fixed latency; a phase cell trained on
short legs systematically mispredicts late firing on long legs). Both
`time_from_reward` and `time_to_reward` bases are included so fixed-latency-after-
reward and fixed-latency-before-reward fields are both representable — a pre-reward
field late in a long leg sits at a small time-to-reward, which is *in* the short-leg
training range.

### 4.2 Continuous warp α — the population object (`temporal` contrast)

```
c_alpha = t / D**alpha       ( alpha = 0 → elapsed time ; alpha = 1 → temporal phase )
```

A raised-cosine basis is evaluated on `c_alpha`; α* is chosen by cross-validated
held-out fit (grid −0.5…1.5). This **basis-expansion** warp (rather than the
parametric log-Gaussian field some pipelines use) handles ramp-to-reward and
multi-peak cells. The per-region histogram of α* + a Hartigan **dip test** says
whether cells form a time/phase dichotomy or a continuum. (No single-parameter
warp exists for time↔space, so the `spatial` contrast reports the signed Δ
continuum instead of α.)

### 4.3 Shared nuisance, event kernels, likelihoods, nulls

- **Nuisance** (subtracted from *both* frames so neither wins by proxy): place,
  head direction (auto-dropped for PFC), speed, acceleration, peri-reward **event
  kernels** (short cosine bumps in the first/last ~1 s, to absorb reward-locked
  transients), plus distance bases for the `temporal` contrast.
- **Likelihoods**: Gaussian least-squares on spike counts (fast, vectorised across
  all neurons of a recday) is the workhorse; **Poisson** (ridge-IRLS, log link) is
  a per-cell cross-check. Counts are integer per 25-ms bin in both regions, so both
  are valid.
- **Nulls**: leg-level bootstrap (per-cell CIs); interval-wise circular-shift
  (`temporal_tuning_pvalue`, gates "is there any temporal/phase structure");
  Hartigan dip (α bimodality); Wilcoxon on the pooled Δ (`population_delta_test`).

## 5. How to run

Open either notebook and run top-to-bottom (results cache under
`../glm_outputs/{LEC,PFC}_time_vs_progress`). Key entry points:

```python
tables  = tp.build_design_tables(mouse_recdays, data_dic,
                                 downsample_factor=10, filter_correct_paths=False,
                                 max_transition_seconds=60)
ood_t   = tp.classify(tp.run_population_ood(tables, "temporal", region="LEC"))
ood_s   = tp.classify(tp.run_population_ood(tables, "spatial",  region="LEC"))
print(tp.population_delta_test(ood_t))          # <- region-level answer
alpha   = tp.run_population_alpha(tables, region="LEC")
```

**Read the population Δ distribution as the headline**, not the per-cell labels
(see §6.1). Two knobs: `filter_correct_paths` (False = all legs, more bootstrap
power; True = shortest-path legs only, cleaner) and the α grid (the slow step,
~70 s/recday, cached).

## 6. What validation discovered

The pipeline was validated two ways: a **synthetic positive control** (a known
fixed-latency time cell and a fixed-phase cell injected into real covariate
structure) and a **real-data smoke test** (3 PFC recdays end-to-end).

### 6.1 Per-cell classification is conservative — use the population

On a single recday, with ~30 legs per duration-half, almost every real cell reads
**"ambiguous"** (e.g. 191/197 on the PFC smoke test): the leg-level bootstrap CI
on a small per-cell effect rarely excludes 0. This is *by design* — the cell-level
test is strict. The region-level answer comes from the **sign/median of the pooled
Δ distribution** (`population_delta_test`), which aggregates the weak per-cell
signal. Treat the stacked-bar classification as secondary colour.

### 6.2 Three bugs the validation caught and fixed

1. **Classification gate.** A pure-noise cell got a tight bootstrap CI that
   *excluded* 0 (Δ≈−0.006) and was mislabeled "phase" — because the bootstrap
   resamples the *same* legs for both frames, so a near-zero difference can be
   stably negative. Fix: a cell is classified only if the winning frame also has
   positive held-out R² (`r2_floor`).

2. **Invalid α confidence intervals.** The first implementation took the α CI from
   an in-sample profile likelihood, which treats all ~25 000 within-leg 250-ms
   bins as independent. They are strongly autocorrelated, so the test was wildly
   over-powered: it returned **zero-width CIs** and "rejected pure time"
   (p≈0.000) for a *true* time cell. Replaced with a **leg-level bootstrap** (the
   real unit of independence), reusing the cached CV errors so it stays cheap.

3. **α zero-point bias from basis resolution.** With 8 cosine bumps a planted time
   cell estimated α*=0.2 instead of 0.0 (the warp basis was slightly more
   expressive at α>0). Raising to 12 bumps moved it to exactly 0.0 while the phase
   cell stayed at 1.0. The default `n_phase` for the warp is now 12.

4. **Gaussian-vs-Poisson agreement metric.** Naive sign agreement on the smoke
   test was only **65%** — but that pools the many near-zero "ambiguous" cells
   whose sign is essentially random. Among cells with a non-trivial effect
   (`|Δ|>0.02`) the two likelihoods agree **100%** (confirmed on a controlled
   mixture). The plot now reports both numbers; judge the cross-check on the
   effect cells.

### 6.3 Positive-control recovery (final, after fixes)

| planted cell | Δ_ood (Gaussian) | Δ_ood (Poisson) | α* | classified |
|--------------|------------------|-----------------|----|-----------|
| fixed-latency **time** (peak t=3 s) | **+1.06** | +0.96 | ≈0 (support_time=1.0) | time |
| fixed-**phase** (φ*=0.6) | **−0.20** | −0.59 | 1.0 (support_phase=1.0) | temporal_phase |
| noise | −0.006 | −0.008 | wide CI (−0.5,1.5) | ambiguous |

Gaussian/Poisson sign agreement = 100%; α shows a clean time/phase dichotomy; the
dip test flags a bimodal mixture (dip=0.49, p=0.002); the circular-shift gate
passes the time cell (p=0.03) and rejects noise (p=0.19).

### 6.4 Preliminary real-data signal (smoke test only — not a result)

Across 3 PFC recdays the pipeline ran clean and **leaned toward progress, not
elapsed time**: 6 phase-classified cells / 0 time, and tuned cells (held-out
R²>0.01) had **median α*=1.0**. This is a sanity-check hint, not a finding — run
the full set, both regions, before interpreting.

## 7. Caveats

- **Power scales with leg count.** Per-recday designs with a strict path filter
  leave few legs; relax `filter_correct_paths` and lean on the pooled Δ.
- **α applies to `temporal` only** (there is no single-parameter time↔space warp);
  the `spatial` contrast is the OOD Δ continuum.
- **α is the slow step** (~70 s/recday). Coarsen the grid (`alpha_grid` step 0.2)
  for a faster pass; the population dichotomy survives it.
- The warp assumes the cell's coordinate is monotone in `c_alpha`; genuinely
  multi-field cells are handled by the basis expansion but their α is harder to
  interpret — cross-check against the OOD Δ and the peak-latency-vs-duration figure.
