# W5 — Goal-progress tuning: sorted β heatmaps, and peak vs width against the time-cell null

`gp_tuning_width.py` · `gp_tuning_width_synthetics.py` · `LEC_gp_tuning_width.ipynb` ·
(PFC mirror: `../mFC_data/code/` — the two module copies are byte-identical; `GP_TUNING_WIDTH.md` there is
a short pointer to this file). Plan: `docs/handoff/W5_gp_tuning_width.md`. Builds on the V3 fits
(`GLM_V3.md`).

**Status: COMPLETE 2026-09-08.** 8/8 synthetic controls on both trees, **four** notebooks executed with
no errors (each dataset ungated and A–B-gated, §4.8), 25/25 recdays in each. Nothing is committed (repo
standing instruction).

```bash
python code/gp_tuning_width.py --check-mirror        # code/ and mFC_data/code/ copies byte-identical
python code/gp_tuning_width.py --build               # per-recday leg-curve caches (LEC: one 3.8 GB load, 2 min)
python mFC_data/code/gp_tuning_width.py --build      # PFC caches (per recday, 9.5 min)
python code/gp_tuning_width.py --verify              # LEC rebuild vs norm_neurons_dic / gp_curves, and the salvage
python code/gp_tuning_width_synthetics.py            # 8 controls: calibration, the time-cell V, the heatmap gate
# then, from the notebook's own directory, with MPLBACKEND=Agg:
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 LEC_gp_tuning_width.ipynb     # ~13 min (PFC ~9 min)
```

## 0. Deviations from the plan, and why

1. **Standalone files only (user instruction, 2026-09-08).** The plan put `plot_beta_heatmap` into the
   shared block of `code/glm_plots.py` (regenerating `pfc_glm_plots.py`), added a §8.2 cell to both V3
   notebooks and a pair to `check_mirror_parity.BYTE_PAIRS`. None of those files is touched. The heatmap
   lives in `gp_tuning_width.py`, the heatmap cells live in the new W5 notebooks, and the module's own
   `assert_mirror()` (called by both notebooks) replaces the parity registration. Adding
   `('code/gp_tuning_width.py', 'mFC_data/code/gp_tuning_width.py')` and the synthetics pair to
   `BYTE_PAIRS` is a two-line follow-up if the submit scripts should guard them too.
2. **Cross-session peak stability** is on the **circular 90-bin leg axis** and the flag is the **maximum
   pairwise circular distance between per-session peaks > 30 bins** (a third of the leg), not the plan's
   "circular SD > 45°". Decided with the user: the circular SD saturates — two peaks 30 bins apart give an
   SD of only ~17 bins (R = 0.5), and SD > 30 bins needs R < 0.11, i.e. near-uniform peaks — so it cannot
   express "moves by more than a third of a leg". The circular SD in bins is still computed and reported.
3. **Width fallback.** The odd legs contribute only the peak **bin**. The width is the contiguous circular
   run of even-leg bins with `(B − min B)/(max B − min B) ≥ 0.5` that contains that bin. If the even curve,
   on its own scale, is below 0.5 at the odd-leg peak, the peak did not reproduce: width **NaN**, flag
   `peak_not_reproduced`, counted per region, excluded from every ρ. (The alternatives — width 0, or a
   threshold at half the even curve's height at the odd-leg peak — were put to the user and declined.)
4. **Planted phase-field widths extended to 0.4 and 0.5** (the plan asked for 0.05–0.3). The real cells'
   median half-max width is ≈ 0.49, and a calibration that stops at 0.3 says nothing about the regime the
   data occupy — which is exactly where an edge effect would show if the estimator had one (§2.1).
5. **Figures the plan did not ask for**, all requested in session: a full-resolution 90-bin curve heatmap
   at σ = 0 and σ = 3 with a cross-validated version; a per-bin figure of how many cells peak in a
   goal-progress bin against how wide those cells are (§4.4). Every peak-vs-width scatter is also produced
   with the simulated overlay removed (`*_nooverlay.pdf`), and the count-vs-width figure with the simulated
   crosses removed (`*_nosim.pdf`), so the data can be looked at alone. The scatters carry the **data's own
   median-per-bin line**, so the reader compares two curves rather than a curve against a cloud.
6. **A reliability control** (§4.3.1) that the plan did not specify, because it is the first objection any
   reader will raise: an untuned cell's peak lands anywhere, so unreliable cells would populate mid-leg and
   could manufacture a V.
7. **A per-region figure guide** (`docs/figures/gp_tuning_width/<region>.md`, one file per region plus an
   index) generated from the executed notebook by `write_region_reports`, so the description of each figure
   and that region's numbers cannot drift from the figures themselves.
8. **A second complete run under an A–B peak-disagreement gate** (§4.8), requested in session, in its own
   directories. It is a robustness variant and the ungated run stays primary, because the gate selects on
   a width-correlated variable. The planted populations are gated identically at every threshold, which
   turns out to matter: in LEC the gate strengthens the null as much as the data. The β heatmaps are
   gated through a `keep_keys` argument added for this purpose; the gate sweep is variant-only.

## 1. Why this exists

Two requests. **(a)** Per V3 arm, a neurons × goal-progress heatmap of the β profiles sorted by peak — the
population picture the per-region β-profile means (`GLM_V3.md` §9.10) average away. **(b)** Whether where a
goal-progress cell peaks in the leg predicts how wide its tuning is.

The second question has a null that must be simulated before any correlation is read (plan §2):
`goal_progress` is time normalised by leg duration D, and D varies (LEC median 9.2 s, p90 25.6 s). A cell
locked to a fixed time t₀ after reward peaks at phase ≈ t₀/median(D) and has a phase width that grows with
t₀ × spread(1/D); one locked to a fixed time before the next reward mirrors that from the leg's end. So
**time cells produce a V-shaped width-vs-peak relation in phase space** (narrow at both ends, broad in the
middle) and **phase cells produce a flat one**. A peak–width relation is therefore the expectation under
time coding, not a finding. Every real relation is drawn beside the curve that planted time cells produce on
this dataset's *own* legs, through the *same* code, and the statistics are compared with the planted
populations rather than with zero (§4.2).

Two statistics, and they answer different questions. **ρ(|peak − 0.5|, width) is the V itself** — negative
for any time-locked population, whatever the anchor, and zero for phase cells. **ρ(peak, width) is the
anchor asymmetry** — positive for a retrospective population (fields at a fixed latency *after* reward),
negative for a prospective one, and about zero for an equal mixture, so it says which end of the leg the
fields are tied to.

## 2. Definitions (implemented in `gp_tuning_width.py`)

| object | definition |
|---|---|
| goal-progress cell | Freedman–Lane p < 0.05 on CPD for `goal_progress` in `core_progress_time__matched_250ms_decile_cap30s_tfrU10b` (phase beyond seconds, 30 s cap). Robustness column: significance in `core_progress_only__matched_250ms_decile_cap30s`. Every cell carries `sel_split = Δr²_gp − Δr²_tfr` from the selecting arm. |
| per-leg curve | every leg `[tt[r,s], tt[r,s+1])` of every trial in the GLM's session set (`get_sessions_for_glm`), rebinned to 90 equal-width phase bins by the `raw_to_norm` rule (`binned_statistic` mean over sample index; legs shorter than 90 samples upsampled by repetition first); Hz. Legs longer than **30 s dropped whole**; the state label is the trial-times column. `leg_phase_curves`. |
| halves | legs in temporal order (sessions ascending, legs ascending) → odd positions = half A, even = half B; means smoothed circularly, σ = 3 bins (sensitivity σ = 2, 5). Asserted disjoint and interleaved. `split_half`. |
| peak | circular argmax of A, phase = bin centre. Also the circular mean direction of A − min A. |
| width (primary) | half-max run on B around A's peak bin (§0.3), fraction of the leg, **circular** (0.9 → 0.1 is one field). A non-circular variant is computed once (`width_hm_linear`). |
| width (threshold-free) | circular SD of B − min B as a density, radians / 2π. `width_csd`. |
| flags | `peak_not_reproduced`; `multimodal` (above-half-max set of B is more than one circular run); `ramp` (peak within a tenth of either end and width > 0.6); `flat`. |
| split-half r | Pearson r between the smoothed A and B curves — reliability; reported per region because the rate confound shows here first. |
| cross-session stability | per-session peak (all legs of that session, σ = 3) → `max_pair_dist_bins`; `remaps = max_pair_dist_bins > 30`; `circ_sd_bins` reported. `per_session_peaks`. |
| β route | from the 10-bin reference-coded β (β₀ = 0): peak = signed argmax (bin 0 included), width = circular half-max run around it in tenths; from the selecting arm (`sel_`) and gp-only/30 (`gpo_`). `beta_table`. |
| simulation null | on each recday's real `Trial_times`: time-from-reward cells (15 Hz for 1 s from t₀ ∈ {0.5, 1, 2, 3, 4, 6, 8, 10, 12} s), time-to-reward cells (the same window ending t₁ before the next reward), phase cells (15 Hz over width w ∈ {0.05, 0.1, 0.2, 0.3} centred at p ∈ {0.1 … 0.9}), 20 noise cells at 2 Hz; base 1 Hz; Poisson at 25 ms; through `build_curves` → `curve_route` unchanged. `simulate_population`, `simulate_recday`, `null_curves`. |
| aggregation | per neuron → per (recday, region) Spearman ρ (≥ 8 cells) → mouse mean → mean over mice, mice shown; contrasts by mouse-resampling bootstrap (`rho_report`, `rho_contrast`); per-neuron quantities (width, split-half r) through `anatomy_split.cluster_bootstrap` with `rate_match` and the 50 µm margin. |

## 2.0 The width rule, worked through (figure: `*_width_explainer.pdf`)

Asked in session, 2026-09-08. **The threshold is B's own half-max, not half of B's height at A's
peak.**

Write B for the even-leg curve of a neuron and A for the odd-leg curve. The rule is:

1. take A's argmax bin, `k` — that is the only thing A contributes;
2. normalise B to its own range, `bn = (B − min B) / (max B − min B)`;
3. the field is the contiguous **circular** run of bins where `bn ≥ 0.5`;
4. the width is the length of the run **containing `k`**, as a fraction of the 90 bins.

So in the example asked about — B at A's peak bin is at 0.8 of B's min-to-max range — the cutoff is
still **0.5 of that range**, not 0.4. Half of B's value at A's peak is never used.

**And yes: the run is then generally asymmetric about A's peak.** That is intended. The run is B's
own field, and A's peak is used only to choose *which* field to measure when B has more than one
bump, so that noise in B's argmax cannot make the estimator jump between them. In the worked
examples the odd-leg peak sits anywhere from the centre of the run to near its edge (one panel has
A's peak at 0.62 inside a run spanning 0.18–0.97). If B is *below* its own half-max at A's peak
there is no run to take: width NaN, `peak_not_reproduced`, counted and excluded (§0.3).

### 2.0.1 What A's peak is actually for, and a stricter gate that was tested

A's peak has **two** jobs, and reliability is only a by-product of the second:

1. **It is the reported peak phase** — the x-axis of every scatter and one half of both correlations.
   It has to come from the held-out legs, because a peak and a width taken from the *same* noisy
   curve are coupled by noise (a noise spike is both a peak and narrow), which is the trap the plan
   named and the reason for the split in the first place.
2. **It selects which run of B to measure** when B is multimodal.

The consistency check falls out of (2): if B is below its own half-max at A's peak, the two halves
disagree about where the cell fires and there is nothing to measure. Those 78 LEC cells (5.1 %) have
a median split-half r of **0.474** against **0.917** for the kept cells, so the flag is picking up
what it should.

**A stricter gate — also drop cells whose two peaks are far apart — has its own full run** (§4.8). It
is a robustness variant, not the primary, for a reason worth stating plainly: the A-to-B offset is
**correlated with width** (cells above 30 bins have a median width of 0.68 against 0.48), so the gate
preferentially removes wide cells, and wide cells are disproportionately mid-leg — exactly the
population that creates the V. Conditioning on a variable related to the outcome can bias the
relation either way, so the object to read is the whole sweep, not one cut, and the planted
populations are gated identically at every threshold.

The offset itself is small for most cells: median 5 bins of 90, p90 21 bins, maximum 45.

One further property the figure makes obvious: because B is min-max normalised before thresholding,
the width measures **shape, not modulation depth**. A 1 Hz ripple on a 12 Hz baseline scores exactly
like a 15 Hz field on a 1 Hz baseline. Depth is carried separately in `range_hz_B` and reliability in
`splithalf_r`; the width axis should never be read as "how strongly tuned".

## 2.1 Is the half-max width affected by edge artefacts?

Asked in session, 2026-09-08. **No — the estimator has no edge, and that is measured, not asserted.**

The half-max run is taken **on the circle**: the leg's end runs into the reward and on into the next leg,
so a field sitting on the boundary is not truncated and no peak position has less room than any other.
Three measurements back that up.

1. **Calibration across the whole width range, at every peak.** Planted phase fields go through the real
   pipeline at widths 0.05–0.5 and peaks 0.1–0.9 (control 3). The recovered width is **identical at every
   planted peak** for widths 0.1, 0.2 and 0.4 (peak-dependence 0.0 bins) and varies by **one bin of ninety**
   for 0.3 and 0.5. A peak-dependence of zero *is* the absence of an edge effect, measured in the regime the
   real cells occupy (median width ≈ 0.49). The 0.05 field returns 0.089: that is the σ = 3 smoothing floor,
   not an edge, and it is recorded rather than gated.
2. **What a non-circular definition would have cost.** `width_hm_linear` is the same run with the wrap
   forbidden. The notebook tabulates the circular-minus-linear deficit per peak decile; it is non-zero only
   for cells whose field straddles the reward, which is precisely the truncation the circular axis avoids.
3. **A threshold-free width.** `width_csd` (circular SD of the baseline-subtracted curve treated as a
   density) uses no half-max crossing at all. Both correlations are recomputed with it, so the reading does
   not rest on the threshold.

Two things that do live at the leg boundary and are **not** estimator artefacts, but must be read as what
they are: the animal is at the port and stationary around phase 0 and 1, so firing there is behaviourally
special (this is caveat 3 of `GLM_V3.md` seen in the curves); and the smoothing wraps within a leg, which is
legitimate only because consecutive legs are pooled — the end of one leg and the start of the next are the
same moment, and both are in the average.

The one asymmetry that *is* real and is not an edge: a curve with no structure has about half its bins above
half-max, so a flat cell that happens to reproduce its peak scores a width near 0.5. That is why the
`peak_not_reproduced` flag, the split-half r and the noise-cell control (median width 0.18 when defined,
52 % not reproduced at all) are reported beside every width.

## 3. Verification (plan §9) — all eight, both trees

`python code/gp_tuning_width_synthetics.py` — **8/8 pass on both trees** (LEC 2.2 min, PFC 2.3 min;
25 recdays × 3 seeds × 3 smoothing widths = 16,650 planted-cell rows per tree).

| # | plan's check | outcome |
|---|---|---|
| 1 | rebuild reproduces `norm_neurons_dic` and `gp_curves.pkl` | **PASS, exactly.** Under the legacy short-leg rule `max|diff| = 3.2 × 10⁻⁷` (float32 round-off) on all 16 sessions of both ah10 recdays, and `1.3 × 10⁻⁷` against `gp_curves.pkl` at σ = 10. Under the rate-preserving rule the only differences are the 3 legs shorter than 90 samples, as they must be. The plan's third clause ("must **differ** on `ah10_20250618_20250619` s5") is **wrong and was replaced** — see §3.2. |
| 2 | width calibration, no peak-dependent bias | **PASS**, and extended to the regime the data occupy (§2.1): planted widths 0.1–0.5 recovered exactly at every planted peak, peak-dependence 0.0 bins (0.2, 0.4) or 1.0 bin of 90 (0.3, 0.5). Planted peaks within 0.5 bins. The 0.05 field returns 0.089 — the σ = 3 smoothing floor. |
| 3 | time-cell null: the V and the per-state shift | **PASS.** Recovered peak rises monotonically with latency (Spearman 1.0) and falls with lead; width is 0.52 mid-leg against 0.27 at the ends. Per-state peak circular SD 0.074 (time cells) against 0.020 (phase cells). Peaks saturate near the leg end beyond ~10 s in PFC (shorter legs); recorded, not gated. |
| 4 | split halves disjoint and interleaved | **PASS**, asserted inside `split_half` and re-tested with a deliberately overlapping order. |
| 5 | heatmap rows = β-profile n; diagonal gate | **PASS.** Row counts equal the `plot_beta_profile` legend exactly for all five regions of the selecting arm (282 / 543 / 151 / 322 / 129). A planted bump population and a pure-noise population **both** give a clean diagonal after sorting; only the bumps have neighbour structure (peak-aligned neighbour excess 0.63 against −0.01). |
| 6 | β-route vs curve-route agreement | **Reported, not gated.** LEC: circular correlation of peaks 0.59, Spearman of widths 0.45, 63 % of peaks within one tenth of the leg. PFC: 0.21 / 0.66 / 72 %. The disagreement is a result (§4.5). |
| 7 | regional panels, primary contrast robustness | **Done** (§4.4). |
| 8 | mirror OK; PFC figure dir writable; nothing committed | **Yes.** `assert_mirror` runs at the top of both notebooks; `_ensure_writable_dir` re-opens `mFC_data/data/**`; nothing is committed. |

### 3.2 Why the plan's currency test was replaced

The plan asserted that `norm_neurons_dic.pkl` holds pre-salvage curves for `ah10_20250618_20250619`
session 5 and that the rebuild must therefore differ there. Comparing the current `data_dic` against
`data_dic_lec.pkl.PRE_salvage_ah10_20250618_20250619_s5` across **all 25 recdays and every session**,
the salvage added `Locs_raw`, `XY_raw` and `HD_raw` to that one session and changed **nothing else**.
`Neuron_raw` and `Trial_times` — all these curves depend on — are identical everywhere, so that pickle's
per-leg curves were never stale and the test as written could not pass.

What the salvage *does* change is the **session set**: `get_sessions_for_glm` requires `Locs_raw`, so
session 5 was invisible before it. `verify_session_set` is the replacement test and passes —
sessions `[1, 2, 3, 4, 5, 6]` now against `[1, 2, 3, 4, 6]` pre-salvage. (This is also the reason the
V3 arms have six folds for that recday where the production fit has five, `GLM_V3.md` §9.4.)

The plan's other warning about that pickle stands and is why nothing here reads it except as a
cross-check: its per-recday value is a **list over sessions that have trials**, not indexed by session
number.

## 4. Results

**Headline: in both datasets the goal-progress population traces the width-vs-peak V that time coding
predicts, and the asymmetry of that V matches a *retrospective* (time-from-reward) population.** The
statistics are read against planted populations that went through the identical code on the same legs,
never against zero.

### 4.1 The population

| | LEC | PFC |
|---|---|---|
| unit-recordings / mice | 2851 / 5 | 1252 / 7 |
| legs (≤ 30 s) | 10,451 | 16,545 |
| legs dropped at the cap | 897 | 331 |
| legs shorter than 90 samples | 104 | 96 |
| gp-significant, uniform/30 | **53.4 %** | **77.8 %** |
| gp-significant, gp-only/30 | 65.0 % | 86.0 % |
| peak reproduced in the held-out half | 94.9 % | 95.9 % |
| median split-half r | 0.910 | 0.925 |
| median half-max width (fraction of leg) | 0.489 | 0.489 |
| multimodal / ramp | 21.3 % / 3.2 % | 19.1 % / 4.2 % |
| peak moves > 30 bins across sessions | 48.1 % | 43.0 % |

The gp-significant fractions reproduce `GLM_V3.md` §9.3 (54 % LEC, 75 % PFC) exactly, which is the
join working.

### 4.2 The V, and what it is read against

ρ over neurons, pooled (the per-mouse version is §4.3):

| population | ρ(peak, width) | ρ(\|peak − 0.5\|, width) | median width |
|---|---|---|---|
| planted phase cells | +0.003 | **+0.011** | 0.250 |
| planted noise cells | −0.024 | **+0.047** | 0.178 |
| planted time cells, both anchors | +0.021 | **−0.483** | 0.367 |
| planted retrospective only | **+0.445** | −0.457 | 0.367 |
| planted prospective only | **−0.406** | −0.508 | 0.356 |
| **real LEC, gp-significant** | **+0.427** | **−0.538** | 0.489 |
| **real PFC, gp-significant** | **+0.234** | **−0.554** | 0.489 |

Two readings, and they are different statistics:

1. **ρ(\|peak − 0.5\|, width) is the V.** Real −0.54 (LEC) and −0.55 (PFC) against −0.48 for planted
   time cells and **+0.01 for planted phase cells**. The real populations sit on the time-cell value.
2. **ρ(peak, width) is the anchor asymmetry.** An equal mixture of retrospective and prospective time
   cells cancels to ≈ 0; retrospective alone gives +0.45, prospective alone −0.41. The real value is
   **+0.43 (LEC)** and **+0.23–0.30 (PFC)**, i.e. the population is asymmetric in the retrospective
   direction — widths grow with phase through most of the leg, as fields anchored to the *last* reward
   do.

Real cells are **broader than any planted population** (0.49 against 0.25–0.38). Some of that is the
mixture (21 % are multimodal) and some is cross-session averaging (§4.3).

### 4.3 Per mouse, per region — the inference panel

Mean over mice of the per-(recday, region) Spearman, mouse-resampling CI, mice shown.

**LEC, ρ(\|peak − 0.5\|, width) — the V**

| region | mice | ρ | 95 % CI | per mouse |
|---|---|---|---|---|
| ENTl-sup | 1 | −0.633 | — | ah08 −0.63 (descriptive) |
| **ENTl-deep** | **5** | **−0.523** | **[−0.599, −0.441]** | −0.66, −0.55, −0.55, … |
| ENTm | 2 | −0.295 | [−0.530, −0.060] | ly06 −0.06, ly07 −0.53 |
| **SUB/ProS** | **4** | **−0.534** | **[−0.562, −0.507]** | −0.50, −0.57, −0.53, … |
| CA1/HPF | 2 | −0.569 | [−0.691, −0.448] | ah10 −0.69, ly05 −0.45 |
| **PFC** | **7** | **−0.521** | **[−0.632, −0.423]** | −0.49, −0.51, −0.49, … |

**LEC, ρ(peak, width) — the anchor asymmetry**: ENTl-sup +0.49 (1 mouse), ENTl-deep +0.35 [0.18, 0.52],
ENTm +0.54, SUB/ProS +0.51 [0.41, 0.64], CA1/HPF +0.49; PFC +0.30 [0.15, 0.48]. Every region positive,
every CI that exists excluding zero. **No region is the exception**, and ENTm is the only one whose V is
much weaker — on two mice that disagree with each other (−0.06 vs −0.53).

**Robustness, all of it in the notebook:**

- **Phase-stable cells only** (peak moves ≤ 30 bins across sessions; 769 of 1445 in LEC): the V is
  unchanged or stronger in every region (ENTl-deep −0.523 → −0.529, ENTl-sup −0.633 → −0.783,
  ENTm −0.295 → −0.731; SUB/ProS −0.534 → −0.500). PFC −0.521 → −0.549. So it is not an artefact of
  averaging a remapping cell's legs across sessions.
- **A single session per recday** (the richest; no cross-session averaging at all): the V weakens but
  survives everywhere — LEC −0.33 to −0.47, PFC −0.478 — and the anchor asymmetry drops (LEC +0.17 to
  +0.40, PFC +0.14). Median width falls to 0.39, so cross-session pooling does broaden the curves.
- **Smoothing** σ = 2 / 3 / 5: ρ(V) −0.52 / −0.52 / −0.51 in ENTl-deep; median width 0.44 / 0.49 / 0.51.
- **Threshold-free width** (circular SD, no half-max crossing): same sign, weaker (−0.23 to −0.35).
  **Non-circular width**: essentially unchanged (−0.56 to −0.68), so the wrap is not producing it.
- **Stratified by the gp-vs-tfr split**: the V does **not** vanish in the cells that lean phase — in
  ENTl-deep it is *stronger* there (−0.640 leans-phase against −0.427 leans-time). This is the one place
  the pre-registered reading fails to discriminate, and §5 says so.
- **It is not reliability** (§4.3.1).

#### 4.3.1 The first objection: is the V just unreliable cells?

An untuned cell's peak lands anywhere, so unreliable cells are spread uniformly over phase while real
fields pile up at the leg edges; if unreliable cells were also scored broad, that alone would draw a V.
Measured, in both datasets, it does not:

| | LEC | PFC |
|---|---|---|
| ρ(split-half r, width) | **+0.146** | **+0.080** |
| ρ(split-half r, \|peak − 0.5\|) | +0.094 | +0.119 |
| median width, least → most reliable quartile | 0.411 → 0.500 → 0.511 → 0.511 | 0.433 → 0.500 → 0.544 → 0.489 |
| fraction peaking mid-leg, least → most reliable | 0.356 → 0.227 | 0.415 → 0.184 |
| ρ(\|peak − 0.5\|, width), reliable half only | −0.34 to −0.84 by region (all still negative) | **−0.531** (from −0.521) |
| planted noise cells, same pipeline | median split-half r **0.13**, ρ(V) **+0.05** | 0.15, −0.05 |

The correlation between reliability and width is **positive and small**: if anything the *more* reliable
cells are slightly *wider*, which is the opposite of the objection. Unreliable cells are indeed more often
mid-leg (the mid-leg fraction falls from 0.36 to 0.23 across reliability quartiles), but restricting to the
reliable half leaves the V intact in every region — it strengthens in ENTl-sup (−0.63 → −0.84) and ENTm
(−0.30 → −0.65) and weakens in CA1/HPF (−0.57 → −0.34). And the real population is nothing like the
planted noise population, whose median split-half r is 0.13 against 0.92.

**The primary contrast (ENTl-deep − SUB/ProS, 4 mice)** shows no regional difference worth reporting:
width −0.012 [−0.113, +0.054], split-half r −0.031, |peak − 0.5| −0.031 [−0.065, +0.003], all CIs
spanning zero and all permutation p > 0.09; ρ(peak, width) −0.085 [−0.156, +0.021]. The one interval
excluding zero is `peak_not_reproduced` (−0.030 [−0.056, −0.004]), which is a reliability difference,
not a tuning one, and it does not survive rate matching.

### 4.4 Where the cells peak, and how wide they are there

Peaks pile up at the **edges** of the leg and are sparse in the middle, and the width follows the V:

| LEC bin (phase) | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| cells peaking | 180 | **259** | 204 | 102 | 137 | 96 | 106 | 82 | 112 | 167 |
| median width | **0.18** | 0.37 | 0.44 | 0.56 | 0.61 | **0.71** | 0.68 | 0.67 | 0.62 | 0.42 |

Across the ten bins, LEC ρ(count, width) = **−0.82** (p = 0.004): the crowded bins hold the narrow
cells. PFC's peaks are more evenly U-shaped (14.9 % in bin 0, 16.9 % in bin 9) and its count–width
correlation is **+0.01** — the same V with a different peak distribution. This figure is therefore a
joint description of the two, not independent evidence; its n is ten bins.

### 4.5 The β route

Peak and width read off the ten-bin β profile agree with the curves only moderately: LEC circular
correlation of peaks 0.59, Spearman of widths 0.45, 63 % of peaks within a tenth of the leg; PFC 0.21 /
0.66 / 72 %. The β route gives the **same** two ρ's (LEC per region +0.24 to +0.57 and −0.25 to −0.69;
PFC +0.27 and −0.66), so the conclusion does not depend on which route is used — but the peaks
themselves move, which is what "goal progress given place, speed, acceleration and elapsed seconds" costs
relative to the raw rate.

### 4.6 Cross-session stability and the per-state discriminator

Half the gp-significant cells move their peak by more than a third of a leg between sessions (LEC 48 %,
PFC 43 %); by region, CA1/HPF is the most stable (34 %) and ENTm/ENTl-sup the least (56 %/55 %). The
per-state peak circular SD is 0.09–0.12 of a leg in the real cells against 0.07 for planted time cells
and 0.02 for planted phase cells — so on that discriminator too the real cells sit at or beyond the
time-cell value, but the per-state median leg durations differ by only ~2–3 s here, which is a weak lever.

### 4.7 Where the figures are

| | |
|---|---|
| LEC figures (61 PDFs) | `data/figures/gp_tuning_width/` |
| PFC figures (37 PDFs) | `mFC_data/data/figures/gp_tuning_width/` |
| per-region figure guide | `docs/figures/gp_tuning_width/{ENTl-sup,ENTl-deep,ENTm,SUB-ProS,CA1-HPF}.md` + `_index.md` |
| PFC figure guide | `docs/figures/gp_tuning_width_pfc/PFC.md` |
| per-leg curve caches | `data/processed_data/gp_leg_curves/`, `mFC_data/data/processed_data/gp_leg_curves/` |
| synthetic-control results and the null curves | `*/processed_data/gp_tuning_width_synthetics.pkl` |

`docs/` is gitignored in this repo, so the region guides live outside version control; regenerate them by
re-running §8 of either notebook. Every scatter has a `_nooverlay` twin with no simulation drawn, and the
count-vs-width figure a `_nosim` twin, for looking at the data alone.

### 4.8 The A–B peak-disagreement gate — a full parallel run

Requested 2026-09-08. The gate keeps only cells whose odd-leg and even-leg peaks lie within *t*
**bins of the 90-bin leg axis** of each other — the headline cut is **30 bins, a third of a leg**.
Note the unit: 30 bins is 120° if the leg is drawn as a full circle, and is *not* the plan's original
"45°", which was on the four-state loop where a leg spans 90°. (The cross-session `remaps` flag
happens to use 30 bins too; different comparison, same number.)

It runs as a **complete second pass** over everything downstream of cell selection — every figure,
the β heatmaps, the per-region ρ panels, the primary contrast, the β route, the per-state peaks and
its own per-region guides — written to its own directories, with the ungated run untouched.
Notebooks `*_gp_tuning_width_abgate.ipynb`.

**How the gate reaches the β heatmaps.** `plot_beta_heatmap` selects rows from the arm's own
significance and knows nothing about the curve route, so a first pass left all 40 LEC / 20 PFC β
heatmaps pixel-identical to the primary run — two thirds of the variant directory was a copy wearing
a variant filename (found 2026-09-08 by rasterising and comparing every pair). It now takes
`keep_keys`, a set of `(recday, neuron)` addresses; `None` in the primary run, so that output is
unchanged by construction. The row-count assertion adapts: exact equality with the `plot_beta_profile`
legend when ungated, and every block a subset of it when gated (the profile lives in `glm_plots.py`,
which this workstream does not modify, so it cannot itself be gated).

**One figure in the variant is necessarily identical** and is documented rather than fixed:
`*_width_calibration.pdf` plots only planted *phase* cells, and the gate excludes none of them at
any threshold, so it cannot change. Everything else in `*_abgate/` differs from its primary twin.
The sweep figure is **variant-only**; an earlier pass wrote it into the primary directories too,
which was wrong and has been removed.

**The sweep, with the planted populations gated identically.** Fraction of cells kept in brackets.

| gate (bins kept) | LEC real | LEC sim time | LEC sim phase | PFC real | PFC sim time | PFC sim phase |
|---|---|---|---|---|---|---|
| none | **−0.538** (100 %) | −0.483 | +0.011 | **−0.554** (100 %) | −0.482 | −0.005 |
| ≤ 30 | **−0.563** (96 %) | −0.491 | +0.011 | **−0.576** (96 %) | −0.482 | −0.005 |
| ≤ 20 | **−0.574** (89 %) | −0.516 | +0.011 | **−0.587** (90 %) | −0.479 | −0.003 |
| ≤ 15 | **−0.585** (81 %) | −0.522 | +0.008 | **−0.599** (85 %) | −0.490 | −0.004 |
| ≤ 10 | **−0.589** (71 %) | −0.542 | +0.016 | **−0.629** (72 %) | (flat) | −0.014 |

(ρ(|peak − 0.5|, width); the anchor asymmetry is stable throughout, LEC +0.427 → +0.451, PFC +0.234 →
+0.254, and the planted phase cells stay at zero at every threshold, which is the estimator behaving.)

**Two different things happen in the two datasets, and the difference is the point of gating the null.**
In **PFC** the planted time cells' V is **flat** under the gate (−0.482 at every threshold) while the
real V strengthens from −0.554 to −0.629, so the gap between data and null *widens* from 0.07 to 0.15.
In **LEC** the gate strengthens the **null too** (−0.483 → −0.542), and the real-minus-null gap is
essentially unchanged (0.055 → 0.047). So the LEC strengthening is largely a property of the gate
rather than of the cells, and only the PFC strengthening survives its own control. Had the null not
been gated, both would have looked like the data getting cleaner.

At the 30-bin cut, per region (mean over mice, ungated → gated): ENTl-sup −0.633 → −0.723 (1 mouse),
ENTl-deep −0.523 → **−0.528** [−0.602, −0.460], 5 mice, ENTm −0.295 → −0.594, SUB/ProS −0.534 →
**−0.586** [−0.639, −0.547], 4 mice, CA1/HPF −0.569 → −0.549. **One caveat specific to the gated run:**
ENTm falls from two mice to one (ly06 drops below the 8-cell floor in every recday), so it becomes a
single-mouse claim there and its large apparent improvement is a change of population, not of estimate.

**What the gated run does not change.** The conclusion. Both datasets still sit on the time-cell V and
away from the phase-cell zero, with the same retrospective asymmetry, on 96 % of the cells.

## 5. What this does not license

- **Not "these are time cells".** The V is what a time-locked population produces, and the real
  populations match it — but the pre-registered discriminator failed in one direction: stratifying by
  the GLM's own gp-vs-tfr split does **not** abolish the relation, and in ENTl-deep the V is strongest in
  the cells that lean *phase*. Either the split is too noisy a per-neuron label to stratify on (its CPDs
  are within ±0.001 of zero, `GLM_V3.md` §9.3), or the population is not simply time-locked. Both remain
  open; this analysis does not settle it.
- **Not a regional claim.** No contrast survives; the primary contrast's intervals all span zero.
  ENTl-sup is one mouse (ah08) and ENTm two that disagree.
- **Not independent of the estimator's breadth.** Real widths (0.49) exceed every planted population's,
  so the real cells are broader than a clean 1 s field would be — the V is a *relative* statement across
  peak positions, not a claim that any particular cell is narrow.
- **The count-vs-width correlation is not evidence.** It is the V multiplied by the peak distribution;
  its sign differs between LEC (−0.82) and PFC (+0.01) purely because the peak distributions differ.
- A 30 s-cap result is a result about legs shorter than 30 s (`GLM_V3.md`), and the selecting arm's
  significance is a Freedman–Lane statement about *that* design.
- Nothing here re-fits the GLM: the β route reads the existing V3 coefficients.
