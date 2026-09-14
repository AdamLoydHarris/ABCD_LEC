# GLM V3 — PFC / mFC dataset

`glm_analysis_v3.py` · `PFC_glm_v3.ipynb` · `glm_v3_synthetics.py` · `w1_refit.py` · `run_glm_batch.py` ·
`pfc_glm_plots.py`

Mirror of the LEC write-up at [`../../code/GLM_V3.md`](../../code/GLM_V3.md), which holds the method,
the design, the caveats and the cross-dataset results. This file records what is different about *this
dataset* and what the PFC run gave.

```bash
python code/check_mirror_parity.py                      # every shared definition identical; must print OK
python mFC_data/code/glm_v3_synthetics.py               # the same 11 controls on a PFC recday
diff code/glm_v3_synthetics.py mFC_data/code/glm_v3_synthetics.py   # must be empty
```

## What is different here

- **The module.** `mFC_data/code/glm_analysis_v3.py` is seeded from the PFC copy of v2 and carries
  exactly the V3 hunks of the LEC copy; the one PFC-specific addition inherited from v2 is
  `build_data_dic_from_pfc` (per-recday `.npy` loader; trial times in ms → 25 ms bins). Two pre-existing
  differences between the frozen v2 copies — a folded cast in `compute_transition_filter_mask` and
  `run_or_load_glm` never collecting `cv_results` — are fixed in this V3 copy, so
  `check_mirror_parity.py` holds V3 to identical code and allowlists only v2.
- **No head direction, no pokes.** Irrelevant here: the reduced designs never contain them, so for the
  first time the PFC and LEC fits are the *same* design (`--regset matched` on both, identical section
  names) and a regional LEC panel can carry a PFC column (`glm_plots.plot_regressor_ranking_by_region(...,
  extra={'PFC': cv_pfc})`).
- **Shorter legs.** Median 7.5 s (p90 17.2, p99 49.8) against LEC's 9.2 / 25.6 / 98.4. A 30 s cap keeps
  90.6 % of the 60 s-cap rows (LEC 80.2 %); under the 60 s cap 2.3 % of rows have tfr > 30 s (LEC
  5.1 %); with uniform 3 s bins the 0–3 s bin holds 32.6 % of in-range rows and the 27–30 s bin 0.8 %
  (a 38× gradient; LEC 15×); at cap 30 the top bin holds 0.1 % of rows. So under uniform coding the same
  bin holds a different share of rows in the two datasets — the caveat on any LEC-vs-PFC panel.
- **Cheaper jobs.** Per-recday loading, no 3.8 GB pickle; 32 GB is ample.
- **No anatomy.** `merge` cannot run the key-contiguity check against `unit_regions`; it still refuses
  overlapping shards and mismatched config stamps.

## Synthetic controls on the PFC recday

Recday `ab03_01092023_02092023` (27,548 rows at cap 60, 26,377 at cap 30; 730 of 732 legs ≤ 60 s,
722 ≤ 30 s), 2026-09-07, `n_perm = 100`, 60 noise cells. Controls 2–11 **all pass** with the same
picture as LEC (the table in `../../code/GLM_V3.md` §7 gives both datasets side by side): the
latency cell loads on tfr with its β peak in the 6–9 s bin (Δr²_tfr 0.582 vs Δr²_gp 0.0001); the
phase cell loads on gp (0.608 vs −0.0001); the consumption cell drives every tfr β for bins 1–9
negative relative to bin 0 (−0.69…−0.85 under uniform coding); the place cell loads on place alone
(0.525); the 60 noise cells give a pooled false-positive rate of 0.043 (uniform) / 0.033 (decile)
under the Freedman–Lane null; the fit's occupancy matches the trial-times measurement to 0.005 with
`legs_reaching` exact and zero out-of-range rows; the 6-bin override gives a 53-column full-rank
design; and in the gp-only arm the latency cell is gp-significant with Δr²_gp 0.0436, falling to
0.0001 once tfr enters — the reading rule of caveat 9. Uniform occupancy at cap 30 (rows per bin):
31.9, 27.7, 19.2, 10.4, 5.2, 2.7, 1.6, 0.8, 0.3, **0.1** %. Controls 1 and 1b **pass with
`max|diff| = 0.0`**: V3 under its defaults reproduces the PFC v2 copy exactly on the production
matched-13 design (every artefact — betas, in-sample F and permutation F, CPD, every cross-validated key),
and V3's decile arm reproduces v2 on the five-regressor design exactly. **12/12.** (The first run of
these two controls reported an 11-unit "difference" that turned out to be `cv_results['elapsed_s']`, the
wall-clock time, which the comparer now ignores.)

## Runtime

150 PFC jobs (6 arms × 25 recdays), 0.3 / **2.3** / 9.6 min (min / median / max), 0 failures, inside
the 81 min wall clock of the 300-job launch of 2026-09-07 (LEC jobs: median 4.2 min).

## Results (25 recdays, 1252 neurons; mouse chain = recday median → mouse mean → mean over mice)

**Arms.** Rows per recday (median): 22,378 at cap 30, 24,554 at cap 60. Full rank in every recday for
the decile and gp-only arms; uniform/30 is rank-deficient in 2 recdays and **uniform/60 in 16 of 25** —
PFC legs are short (median 7.5 s), so 6 s bins above 48 s are empty in most recdays. `r2_cv`: production
matched-13 **0.0411**; decile/30 0.0517, uniform/30 0.0497, gp-only/30 0.0477, decile/60 0.0508,
uniform/60 0.0473, gp-only/60 0.0464. The reduced designs generalise better than the 13-regressor fit
in **80 %** of neurons (same rows, TSS identical in all 25 recdays).

**Occupancy** (% rows per bin, mean over recdays): decile arms 9.8–10.8 % everywhere, with decile
edges moving by 1.0–10.3 s (cap 30) and 1.2–16.2 s (cap 60) between recdays; uniform/30: 33.4, 27.8,
17.0, 9.6, 5.4, 3.2, 1.9, 1.1, 0.5, **0.14**; uniform/60: 57.3, 25.2, 8.9, 4.0, 2.1, 1.2, 0.7, 0.4,
0.16, **0.03**. Rows beyond the fixed range coded as bin 0: 14 of 545,508 (cap 30), 3 of 601,583 (cap 60).

**Raw CPD per arm** (corrected CPD in brackets; frac_sig = Freedman–Lane on CPD)

| regressor | decile/30 | uniform/30 | gp-only/30 | decile/60 | uniform/60 | gp-only/60 | production full-13 |
|---|---|---|---|---|---|---|---|
| place | 0.0074 (0.0097) .68 | 0.0073 (0.0096) .67 | 0.0072 (0.0095) .67 | 0.0080 (0.0101) .69 | 0.0078 (0.0098) .72 | 0.0078 (0.0098) .68 | 0.0091 (0.0114) .70 |
| goal_progress | **0.0020** (0.0029) .65 | **0.0041** (0.0051) .75 | **0.0087** (0.0097) **.83** | 0.0016 (0.0024) .65 | **0.0065** (0.0073) .79 | **0.0078** (0.0087) .80 | −0.0000 (0.0008) .47 |
| speed | 0.0066 (0.0073) .70 | 0.0067 (0.0074) .70 | 0.0069 (0.0076) .70 | 0.0065 (0.0072) .71 | 0.0066 (0.0072) .71 | 0.0068 (0.0075) .71 | 0.0057 (0.0063) .72 |
| acceleration | 0.0022 (0.0028) .65 | 0.0022 (0.0028) .65 | 0.0023 (0.0028) .65 | 0.0023 (0.0028) .68 | 0.0024 (0.0028) .68 | 0.0024 (0.0029) .69 | 0.0018 (0.0023) .63 |
| time_from_reward | **0.0020** (0.0031) .64 | 0.0006 (0.0019) .51 | — | **0.0029** (0.0040) .66 | −0.0006 (0.0008) .34 | — | 0.0016 (0.0025) .62 |
| progress_or_time | **0.0116** | 0.0101 | — | **0.0111** | 0.0077 | — | — |

**What PFC says.** Within-leg structure is PFC's strongest task signal: with `goal_progress` as the only
within-leg regressor it out-ranks place (0.0087 vs 0.0072) and is significant in 83 % of neurons; the
joint block is 0.011–0.012, four to five times LEC's. Under decile coding both survivors stay positive
and equal (0.0020 / 0.0020 at cap 30); under uniform coding the shared budget moves to `goal_progress`
(0.0041 / 0.0006) — the same coding effect as LEC, with the same cause (the sparse late uniform bins).
Full-13 → core-5 (same rows): corrected goal_progress 0.0008 → 0.0024 (×3.2), time_from_reward 0.0025
→ 0.0040 (×1.6), speed ×1.1, acceleration ×1.2, **place 0.0114 → 0.0101 (×0.9)** — place is the one
survivor that loses unique variance when the within-leg family leaves. gp decomposition: 0.0087 →
0.0020 (tfr decile) → 0.0041 (tfr uniform) at cap 30; 0.0078 → 0.0016 → 0.0065 at cap 60; per neuron
gp-only ≥ gp+tfr uniform in 73 % / 63 %.

**tfr β profile** (tfr-significant neurons): 13 % first-bin dominant (consumption) in the decile arms
and uniform/30; 22–26 % peak in the last decile. Same two populations as LEC (`GLM_V3.md` §9.9).

**Goal-progress β profile** (`pfc_gp_beta_profile_*.pdf`, and `*_meancentred.pdf` with each neuron's
mean over bins subtracted; reference bin 0 = the first tenth of the leg, the consumption period, and
0 for every neuron by construction — see `GLM_V3.md` §9.10 for how to read it): in the gp-only arm PFC
**ramps smoothly from 0 to a mid-leg peak** (mean β 0.37 at 0.4–0.7; **43 %** of the 1077
gp-significant neurons fire most there, 14 % during consumption, 21 % early, 22 % late) and declines
towards the next reward — a shape no LEC region has (LEC regions plateau or peak early; the highest
mid-leg share in LEC is ENTl-deep's 35 %). With `time_from_reward` in the model the early-leg component
leaves goal progress and the profile flattens (mean β 0.12 / 0.16 / −0.01 by leg third; peaks 18 / 28 /
31 / 24 %) — the same shift as LEC. Peaks are the bin of highest firing per neuron, bin 0 included.

**Beside the LEC regions** (`PFC_glm_v3.ipynb` §9): PFC's gp-only goal-progress CPD (0.0087) exceeds
every LEC region's (ENTl-deep 0.0035 is the highest); its place CPD (0.0072) is below every LEC region
including ENTl-sup (0.0098); its decile-coded time-from-reward CPD (0.0020) is above every LEC region's
(ENTl-deep 0.0015).

**Not licensed here:** the uniform/60 tfr block (16/25 recdays with empty bins); any phase-vs-time
split read off a single coding; treating this whole-structure PFC column as comparable in kind to an
LEC sub-region.
