# W5 — Goal-progress tuning: sorted β heatmaps, and peak vs width

**Status: EXECUTED 2026-09-08. Results and method: [`code/GP_TUNING_WIDTH.md`](../../code/GP_TUNING_WIDTH.md)**
(PFC pointer `mFC_data/code/GP_TUNING_WIDTH.md`; per-region figure guides
`docs/figures/gp_tuning_width*/`). The rest of this document is the plan as approved, kept as the record
of intent — read the write-up for what was actually done.

## Outcome in five lines

- **The V is there, in both datasets.** ρ(|peak − 0.5|, width) = **−0.54 (LEC, per-region −0.30 to −0.63)**
  and **−0.52 (PFC, 7 mice, CI [−0.63, −0.42])**, against **−0.48** for planted time cells and **+0.01** for
  planted phase cells through the identical code on the same legs. Planted noise gives +0.05.
- **The asymmetry says retrospective.** ρ(peak, width) = +0.43 (LEC) / +0.30 (PFC); a retrospective-only
  planted population gives +0.45/+0.29, a prospective-only one −0.41/−0.29, an equal mixture ≈ 0.
- **The pre-registered discriminator did not fire.** Stratifying by the GLM's gp-vs-tfr split does *not*
  abolish the relation — in ENTl-deep the V is strongest in the cells that lean *phase*. So this is not
  read as settled time coding; §5 of the write-up says why.
- **No regional claim survives.** The primary contrast's intervals all span zero.
- **8/8 synthetic controls** on both trees; the plan's verification 1 was replaced (see below).

Four departures from the plan below, all recorded in §0 of the write-up:

1. **Standalone files** (user instruction): the heatmap lives in the new `gp_tuning_width.py`, not in the
   shared block of `glm_plots.py`; there is no §8.2 in the V3 notebooks and no new `BYTE_PAIRS` entry.
2. **Cross-session stability** is the max pairwise circular peak distance > 30 bins on the 90-bin leg
   axis, not "circular SD > 45°" — the SD saturates and cannot express a third-of-a-leg move.
3. **Verification 1's third clause is wrong.** The salvage did not change `Neuron_raw` or `Trial_times`
   anywhere, so the rebuild *matches* `norm_neurons_dic` on `ah10_20250618_20250619` s5. The currency test
   is the **session set** instead, and it passes.
4. **Planted widths extended to 0.5** and three figures added at the user's request (unsmoothed
   full-resolution curve heatmaps, peak-count vs width, and overlay-free scatters).
5. **A second complete run under an A–B peak-disagreement gate**, in its own directories
   (`*_abgate`), as a robustness variant with the ungated run primary. The planted populations are
   gated identically, which matters: in PFC the null is flat under the gate and the data-minus-null
   gap widens, in LEC the gate moves the null as much as the data.

---

## 0. Where to start (10 minutes)

```bash
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC
module load miniconda; eval "$(conda shell.bash hook)"; conda activate maze_ephys
python code/check_mirror_parity.py                    # must print OK before touching anything shared
ls data/glm_outputs/LEC/core_progress_*__cv_results.pkl mFC_data/glm_outputs/PFC/core_progress_*__cv_results.pkl   # 6 + 6 arms
```

Read, in this order: `code/GLM_V3.md` §1, §3, §6 (caveats), §9.10 (the goal-progress β profile and
how to read the reference bin); `docs/handoff/README.md` §3 (inference rules) and §6 (gotchas). Then
this document.

**Nothing is committed** in this repo by standing instruction; do not commit.

## 1. What was asked

Two requests, one pipeline.

**(a)** Per run (= per V3 arm), a **neurons × goal-progress heatmap of the β profiles, sorted by
peak** — the population picture that the per-region β-profile figures (`data/figures/glm_v3/
lec_gp_beta_profile_*.pdf`) average away.

**(b)** Is there a **relationship between where a goal-progress cell peaks in the leg and how wide
its tuning is?** From the betas directly, or from actual 90-bin tuning curves.

## 2. The confound that has to be the centre of the design

`goal_progress` is **time normalised by leg duration**, and leg duration varies (LEC median 9.2 s,
p90 25.6 s; PFC 7.5 / 17.2). A cell locked to a fixed *time* after reward, t₀, sits at phase t₀/D on a
leg of duration D. Its phase peak is ≈ t₀/median(D) and its phase **width grows with t₀ × spread(1/D)**.
A cell locked to a fixed time *before* the next reward mirrors this from the leg's end. Therefore:

- a population of **time cells** produces a **V-shaped width-vs-peak relation in phase space** — narrow
  at both ends of the leg, broadest in the middle — and a Weber-law timing population steepens the V;
- a population of **phase cells** produces **no** dependence of width on peak;
- so a peak–width correlation is the *null expectation under time coding*, not evidence of anything,
  and the analysis is a **test between time coding and phase coding** that must be read against the
  curve time coding predicts for this dataset's actual leg durations. That curve can be simulated
  exactly, through the real pipeline, as `glm_v3_synthetics.py` already does for the GLM.

A second trap: a peak and a width estimated from the **same** noisy curve are coupled by noise (a noise
spike is both a peak and narrow). Peak and width are estimated on **disjoint, interleaved halves of the
legs**.

Two facts that make this tractable: GLM V3 gives every neuron a **gp-vs-tfr split** (Δr²_gp − Δr²_tfr
in the gp+tfr arms), so the relation can be stratified by "leans phase" vs "leans time"; and goal-
progress **occupancy is equal by construction** (measured 9.5–10.4 % per bin), so widths are not
distorted by sparse bins the way uniform time-from-reward bins are (`GLM_V3.md` §9.10).

## 3. Decisions (the user's, 2026-09-08)

| question | decision |
|---|---|
| Which cells are "goal-progress cells" | **Significant `goal_progress` in the V3 gp+tfr uniform/30 arm** — `core_progress_time__matched_250ms_decile_cap30s_tfrU10b`, Freedman–Lane p < 0.05 on CPD (`cv_results[rd]['p_freedman_lane__cpd']['goal_progress']`): phase *beyond* absolute time coded in seconds, 30 s cap. 54 % of LEC neurons, 75 % of PFC. Annotate every cell with its gp-vs-tfr split in that arm and with membership in the gp-only/30 significant set (65 % / 83 %) as a robustness column. |
| Primary width estimator | **90-bin per-leg phase curves**, split-half (peak from odd legs, width from even legs), circular phase axis, σ = 3 bins smoothing (sensitivity at σ = 2 and 5). |
| β route | **Existing 10-bin betas only** — from the uniform/30 arm (the selecting arm) and, as the "any within-leg structure" comparison, from gp-only/30. No 30-bin refit. |
| Heatmap scope | **Both versions per arm**: gp-significant neurons only, and all neurons; each as one pooled panel and, for LEC, as region-stacked blocks with a colour strip. |
| Simulation null | Always: fixed-latency and fixed-lead time cells and phase cells on each recday's **real** legs, through the same code. |
| Units / stats | Neuron within mouse; per-mouse Spearman ρ(peak, width) and ρ(\|peak − 0.5\|, width) (the V); mean over mice with mice shown; LEC per region under README §3; PFC pooled; simulated curves overlaid. |
| Leg cap | 30 s, matching the selecting arm; legs longer than the cap excluded whole (argument). |

## 4. Facts established (verified 2026-09-08 — do not re-derive)

- **Neither derived pickle is a safe data source; rebuild curves from `data_dic_lec.pkl`.**
  - `data/processed_data/norm_neurons_dic.pkl` (1 Sep 14:37) **predates the 6 Sep salvage** of
    `ah10_20250618_20250619` session 5, so that session's per-trial curves are the pre-salvage
    alignment (identical shapes, undetectable by a shape check). Its per-recday value is a **list over
    sessions that have trials, not indexed by session number**: in 9 of 25 recdays (the `…0624_0625`
    recdays with sessions {0,2,3,6,7,8} and the recdays whose session 3 has zero trial rows) entry *i*
    is not session *i* (`ah08_20250620_20250623` entry 3 is session 4). Elsewhere its shapes agree with
    `data_dic` (`(n_neurons, n_trials, 360)`, 4 states × 90 bins).
  - `data/processed_data/gp_curves.pkl` (3 Sep; `{recday: {session: (n, 90)}}`, trial- and
    state-averaged, σ = 10 circular smoothing from `Smoothed_norm`) also predates the salvage, is too
    smooth for a width, has no trial level, and is missing five sessions with trials
    (`ah08_20250616_20250617` s7, `ly05_20250620_20250623` s6/s7, `ly07_20250624_20250625` s8, every
    empty session 3).
  - Use both **only as cross-checks**: the rebuild must reproduce `norm_neurons_dic` on a recday it is
    current for (`ah10_20250616_20250617`) and must **differ** on `ah10_20250618_20250619` s5.
- `data_dic_lec.pkl` (3.8 GB; `glm_analysis_v3.load_data_dic(validate=True, apply_exclusions=True)`,
  ~1 min when the OS cache is warm, ~4 min cold) holds per session `Neuron_raw` (n_neurons, T) uint16
  counts at 25 ms, `Trial_times` (n_trials, 5) in **bin** indices, `Locs_raw`, `Task`, `Smoothed_norm`.
  PFC (`mFC_data/code/glm_analysis_v3.build_data_dic_from_pfc(w.DATA_FOLDER, [recday])`, per-recday,
  cheap) has `Neuron_raw`, `Trial_times` (bins), `Locs_raw`, `XY_raw`, `Task`, `num_trials` — no
  normalised arrays. So **one code path must build per-leg 90-bin curves from raw for both datasets**.
- Per-trial normalisation reference: `remapping_rotation_analysis.raw_to_norm(raw_1d, trial_times,
  config, return_mean=False, statistic='mean')` → (n_trials, 360): each leg `[tt[r,s], tt[r,s+1])`
  rebinned to 90 bins with `scipy.stats.binned_statistic`; legs shorter than 90 samples are upsampled by
  repetition first. `config` is `RemappingConfig` in that module (`num_task_states=4`,
  `num_bins_per_state=90`, `smoothing_sigma`). `smooth_circular(x, sigma)` is there too.
- Split-half reference: `splithalf_ratemap_consistency._splithalf_session(nn, sigma)` — z-score per
  neuron, Gaussian smooth along bins, **first vs last half** of trials, Pearson r. We want **interleaved**
  halves (odd/even legs) so session drift (leg duration drifts, ρ ≈ −0.3 with session time) does not
  become a difference between halves.
- Sessions to use: `glm_analysis_v3.get_sessions_for_glm(data_dic[rd])` (the GLM's set: ABCD sessions
  deduplicated by task); the 21 Object sessions and 9 zero-trial ABCD sessions are already excluded
  there (`README.md`; `lec-unusable-sessions`).
- V3 pickles and keys: `glm.load_glm_results(rb.DEFAULT_OUT, section, apply_exclusions=True)` →
  `{'glm_results', 'permutation_results', 'cpd_results', 'cv_results'}`. β profile for a neuron:
  `np.concatenate([[0.0], glm_results[rd][neuron][col_idx]])` with
  `col_idx = glm._resolve_regressor_groups(regs, parameterization='reference_coded')[0]['goal_progress']`
  (regs = `w.SECTIONS[section]['regressors']`; 9 columns, bin 0 is the reference and is **0 by
  construction** — see `GLM_V3.md` §9.10 for the reading). Significance:
  `cv_results[rd]['p_freedman_lane__cpd'][g]`; effect sizes `cpd_cv`, `delta_r2_cv`,
  `null_mean_freedman_lane__*`. Neuron *k* of `sorted(glm_results[rd])` is `Neuron_raw` row *k*
  (`anatomy_split.assert_glm_keys_contiguous`).
- Section names (both datasets, `--regset matched`): `core_progress_time__matched_250ms_decile_cap{30,60}s_tfr{U,D}10b`,
  `core_progress_only__matched_250ms_decile_cap{30,60}s`. Helpers: `w1_refit.section_name(base,
  width_ms, scheme, regset, extra)`, `w1_refit.arm_extra(cfg, leg_cap_s=, tfr_scheme=, tfr_bins=)`.
- Regions: `anatomy_split.load_unit_regions()` → `{recday: DataFrame}` with `group`, `mouse`, `y_um`,
  `shank`, one row per `Neuron_raw` row; `region_of = {rd: ur[rd]['group'].to_numpy()}`.
  `anatomy_split.per_mouse_effect(joined, col, statistic, groups)`, `per_region_report`,
  `cluster_bootstrap`, `within_recday_permutation`, `rate_match`, `boundary_margin_filter`,
  `ANALYSIS_GROUPS`, `REGION_COLORS`, `PRIMARY_CONTRAST = ('ENTl-deep', 'SUB/ProS')`.
  `glm_plots.join_region(cv_results, regressor, unit_regions, value, p_stat=None)` joins one regressor's
  values (or p-values) to anatomy.
- Existing plotting: `glm_plots.plot_beta_profile(glm_results, cv_results, col_idx, regressor, *,
  region_of, groups, colors, alpha, null, bottom='auto'|'occupancy'|'peaks', center='reference'|'mean',
  p_stat, out_path)` — unit-norm β per neuron (divided by its own max\|β\|), mean ± s.e.m. per region,
  lower panel occupancy or **peak fraction** (peak = bin of highest β, **bin 0 included**, signed
  argmax — never the argmax of \|β\|, which cannot select bin 0). `plot_tfr_beta_profile` is a wrapper.
  The shared block of `code/glm_plots.py` (from `#: GridMaze palette.` to the `LEC only` marker) is
  regenerated verbatim into `mFC_data/code/pfc_glm_plots.py`; `check_mirror_parity.py` asserts it.
- Synthetic-cell generator to extend: `glm_v3_synthetics.synthetic_recday(data, rd, n_noise, base_hz,
  amp_hz, noise_hz, seed)` — Poisson cells (latency 6–9 s, phase 0.4–0.6, consumption 0–1.5 s, place,
  noise) driven by a recday's real `Trial_times`/`Locs_raw`, via
  `glm.compute_task_state_arrays(tt, num_bins=10)` → `(state, goal_progress_continuous,
  gp_binned, time_from_reward_bins, time_to_reward_bins)`.
- Goal-progress occupancy: equal by construction (cap excludes whole legs); measured 9.9–10.1 % per bin
  at the raw rate, 9.5–10.4 % per 250 ms row; the small tilt is the block-mode tie-break toward the
  lower bin index, and in PFC the tracking filter dropping 1.5–2.3 % of rows in the last three bins.
- Repo quirk: **`mFC_data/data/**` is set read-only after the fact** (every directory under it ends up
  `r-x`, apparently a periodic protection of the published dataset). Writing PFC figures to
  `mFC_data/data/figures/glm_v3/` needs `chmod u+w` on the directory first; `mFC_data/glm_outputs/`
  is unaffected.

## 5. Part A — sorted β heatmaps per run

New `plot_beta_heatmap` in the shared block of `code/glm_plots.py` (then regenerate
`pfc_glm_plots.py`; see §8 for the regeneration snippet):

```python
plot_beta_heatmap(glm_results, cv_results, col_idx, regressor, *, only_significant=True, alpha=0.05,
                  null='freedman_lane', center='reference', sort='peak', region_of=None, groups=None,
                  colors=None, out_path=None)
```

- rows = neurons (gp-significant by default; `only_significant=False` for all), columns = the
  regressor's bins with bin 0 included (0 under `center='reference'`; `center='mean'` subtracts each
  neuron's mean over bins first); each row divided by its own max\|β\| so the colour is shape, not rate;
  diverging colormap (`RdBu_r`) centred on 0, symmetric limits.
- `sort='peak'`: by the bin of highest β (signed argmax, bin 0 included); ties broken by the circular
  centre of mass of the positive part, so rows within a peak bin order by skew.
- `region_of` given → one block per region in `groups` order, each block sorted independently, a colour
  strip at the left in `REGION_COLORS`, block boundaries drawn, n per block in the y-label; without it
  one pooled panel. Style: `_style()`, `_save()`, 8 pt.
- Row count must equal the n in the corresponding `plot_beta_profile` legend (assert in the notebook).
- Produce for all six arms in both datasets, `goal_progress` (and `time_from_reward` for the four
  gp+tfr arms), significant-only and all-neuron, pooled and (LEC) region-stacked →
  `data/figures/glm_v3/lec_gp_beta_heatmap_{tag}[_all][_regions].pdf`, `mFC_data/data/figures/glm_v3/
  pfc_gp_beta_heatmap_{tag}[_all].pdf` (`tag` = section name after `__`). Add a §8.2 cell to both V3
  notebooks (they are generated by a builder script — see §8).
- Gate (repo practice, `synthetic-controls-catch-design-errors`): a planted phase-cell population must
  give a clean diagonal after sorting and a pure-noise population must not look tuned beyond the
  sort-induced diagonal — the same warning `gp_region_heatmaps` carries: the sorting panel is never
  evidence on its own.

## 6. Part B — peak vs width

### B1. Per-leg phase curves — new `code/gp_tuning_width.py`, mirrored byte-identically to `mFC_data/code/`

Tree detection as in `glm_v3_synthetics.py` (`IS_PFC = hasattr(glm, 'build_data_dic_from_pfc')`);
add the pair to `check_mirror_parity.BYTE_PAIRS`.

- `leg_phase_curves(neuron_raw, trial_times, *, n_bins=90, cap_s=30.0, bin_ms=25)` →
  `curves (n_neurons, n_legs, 90)`, `leg_duration_s (n_legs,)`, `leg_state (n_legs,)`. Every leg
  `[tt[r,s], tt[r,s+1])` of every trial rebinned to 90 equal-width phase bins (vectorised: one
  `np.add.reduceat` per leg over all neurons; legs shorter than 90 samples upsampled by repetition as
  `raw_to_norm` does; mean count per bin → Hz optional). Legs longer than the cap dropped **whole**.
  Input is the **current `data_dic`**, keyed by session number.
- `split_half(curves, legs_meta, sigma=3)` → `(curve_odd, curve_even) (n_neurons, 90)`: means over odd
  and even legs (interleaved across the pooled sessions, in temporal order), each circularly smoothed
  (`smooth_circular`, σ bins), plus split-half Pearson r per neuron (reliability; report its
  distribution per region — the rate confound shows here first). Halves asserted disjoint.
- `peak_and_width(curve_A, curve_B)` per neuron:
  - **peak** from half A: circular argmax → phase in [0, 1); also the circular mean direction of the
    baseline-subtracted curve (noise-robust alternative, reported alongside).
  - **width** from half B around half A's peak: half-max width = fraction of the 90 bins in the
    contiguous circular run around the peak where `(B − min B)/(max B − min B) ≥ 0.5`; and the
    threshold-free alternative, circular SD of the baseline-subtracted curve treated as a density.
    Flag (do not silently include) **multimodal** cells (above-half-max set not contiguous) and **ramp**
    cells (monotone across the leg: peak at an edge with width > 0.6).
  - Phase axis is **circular** (leg end → reward → next leg start; a consumption field spans 0.9 → 0.1).
    State this in the doc; compute a non-circular variant once to show what it changes.
- Sessions: `get_sessions_for_glm`; legs pooled over sessions **after** checking per-session peak
  stability (circular SD of per-session peaks; cells that remap phase across tasks get a wide pooled
  curve — report the fraction with cross-session peak SD > 45° as its own number; W3 found state
  identity remaps across tasks while progress may be abstract).

### B2. The β route (covariate-adjusted, secondary)

From the **uniform/30 arm** (the selecting arm): 10-bin β with β₀ = 0, unit-normalised; peak = signed
argmax (bin 0 included); width = contiguous circular run around the peak with
`(β − min)/(max − min) ≥ 0.5`, in tenths of the leg. The same from gp-only/30, labelled "any within-leg
structure", so the effect of holding absolute time fixed is visible per neuron. Agreement with B1 per
neuron (circular correlation of peaks; Spearman of widths) is a result, not a check: where they disagree,
the covariates (place, speed, time-from-reward) are doing the work.

### B3. The simulation null — `gp_tuning_width.simulate_population(data, rd, kind, ...)`

Built on `synthetic_recday`'s pattern, on each recday's **real** legs (`Trial_times`), Poisson at 25 ms:

- **time-from-reward cells**: 15 Hz in a 1 s window at latency t₀ ∈ {0.5, 1, 2, 3, 4, 6, 8, 10, 12 s}
  after each reward; **time-to-reward cells**: the same window at lead t₁ before the next reward;
  **phase cells**: 15 Hz over a phase window of width w ∈ {0.05, 0.1, 0.2, 0.3} centred at
  p ∈ {0.1, …, 0.9}; **noise** at 2 Hz. Base rate 1 Hz. Several seeds.
- Every simulated cell goes through B1 unchanged → expected (peak, width) per type. The **time-cell
  curve** (width vs peak, both anchors) is the null the real cells are read against; the phase-cell grid
  is the **calibration of the width estimator** (planted width 0.1 must come back ≈ 0.1 at every peak;
  smoothing sensitivity read here).
- Also planted and recorded: recovered peak of a time cell vs the state's median leg duration (time
  cells shift with D, phase cells do not) — the discriminator for B5.

### B4. Statistics and figures (`code/LEC_gp_tuning_width.ipynb`, `mFC_data/code/PFC_gp_tuning_width.ipynb`)

- Scatter of width vs peak, per region (LEC) and pooled (PFC), uniform/30 gp-significant cells,
  coloured by Δr²_gp − Δr²_tfr in the same arm, with the simulated time-cell V and the phase-cell flat
  line overlaid; marginal histograms of peak and width. Repeated for the gp-only/30 set (the gate's
  effect).
- Per mouse: Spearman ρ(peak, width) and ρ(\|peak − 0.5\|, width); mean over mice with the mice shown;
  per region for LEC (README §3: regions first, mice as the unit, ENTl-sup = ah08 and ENTm = ly07 are
  single-mouse claims, primary contrast ENTl-deep vs SUB/ProS with `rate_match` and the 50 µm margin);
  the same statistics on the simulated populations.
- Readings, decided in advance: a relation that matches the time-cell V and vanishes in the
  "leans phase" stratum → time coding; a flat relation with a real width distribution → phase coding;
  anything else is reported as such. **Do not read a positive ρ as a finding without the simulated V
  beside it.**
- β-route panels beside the curve-route panels; agreement statistics; split-half reliability per
  region.

### B5. Optional extension (only if asked)

Per-state curves: the four legs have different tower distances, hence different typical durations. Peak
phase vs the state's median leg duration per cell: time cells shift, phase cells do not — a second
discriminator that needs no simulation.

## 7. Files

| action | path |
|---|---|
| modify | `code/glm_plots.py` (shared block: `plot_beta_heatmap`) → regenerate `mFC_data/code/pfc_glm_plots.py` |
| create | `code/gp_tuning_width.py` = `mFC_data/code/gp_tuning_width.py` (byte-identical; add to `check_mirror_parity.BYTE_PAIRS`) |
| create | `code/gp_tuning_width_synthetics.py` (or a `--synthetic` mode): B3 populations + width-estimator calibration; mirrored |
| create | `code/LEC_gp_tuning_width.ipynb`, `mFC_data/code/PFC_gp_tuning_width.ipynb` (generate from one builder script so they stay in step, as the V3 notebooks were) |
| create | `code/GP_TUNING_WIDTH.md` (+ short PFC mirror `mFC_data/code/GP_TUNING_WIDTH.md`), register of `GLM_V3.md`: the confound argument, definitions, calibration, results, what it does not license |
| modify | both V3 notebooks (§8.2 heatmaps — via their builder), `code/GLM_V3.md` (§9.11 pointer), `docs/handoff/README.md` (§8 state of play), this file (status) |

## 8. Conventions and mechanics a fresh session needs

- **Mirror discipline.** `code/` and `mFC_data/code/` are copies, not imports. Every shared module is
  edited in `code/` and copied; `python code/check_mirror_parity.py` must print OK (it is run by the
  submit scripts). To regenerate the PFC plot module after editing `glm_plots.py`: keep the PFC
  docstring header, then imports (`from __future__`, `os`, `matplotlib as mpl`, `pyplot as plt`,
  `numpy as np`, `pandas as pd` — **no** `anatomy_split`), then the LEC block from the line starting
  `#: GridMaze palette.` to the line before the `# ====` rule above `# LEC only, below this line`.
- **Notebook generation.** The V3 notebooks were produced by a builder script (scratch of the previous
  session; the pattern is in `code/LEC_glm_v3.ipynb` itself). Execute headlessly with
  `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1
  --ExecutePreprocessor.kernel_name=python3 <nb>` from the notebook's directory with `MPLBACKEND=Agg`.
  Check `execution_count` and error outputs on every code cell afterwards; the exit code of a piped
  command lies.
- **Synthetics through the real pipeline** before believing a figure (`synthetic-controls-catch-design-
  errors`): here that is the width-estimator calibration and the time-cell V.
- **Documentation register.** `GLM_V3.md` is the model: why it exists, what changes (signatures),
  structural facts with measured numbers, how to read the figures, caveats, synthetic controls with
  outcomes, runtime, results, what it does not license.
- **Aggregation chain** for any pooled number: per neuron → recday median → mouse mean → mean over
  mice, mice shown (`glm_plots.per_mouse_stat`, `anatomy_split.per_mouse_effect`).
- **Where things run.** Curve building for LEC needs the 3.8 GB `data_dic` once (do it in one process
  and cache the per-leg curves per recday under `data/processed_data/gp_leg_curves/` or one pickle;
  ~1–2 GB for 2851 neurons × ~600 legs × 90 float32 — or store per-neuron split-half means only). PFC
  is per-recday and cheap. Nothing here needs SLURM.

## 9. Verification (blocking)

1. `leg_phase_curves` reproduces `norm_neurons_dic` on `ah10_20250616_20250617` (same legs,
   `max|diff|` ≈ 0 up to the cap), reproduces `gp_curves.pkl` when trial-averaged at σ = 10, and
   **differs** from both on `ah10_20250618_20250619` s5 — the proof the rebuild is on current data.
2. Width calibration: planted phase cells of width 0.05–0.3 at peaks 0.1–0.9 come back within ±1 bin at
   every peak (no peak-dependent bias); planted peaks within ±1 bin.
3. Time-cell null: planted fixed-latency cells produce the predicted V (width rising with t₀ on the
   retrospective arm, falling on the prospective arm); their recovered peak shifts with the state's leg
   duration, phase cells' does not.
4. Split halves disjoint and interleaved (asserted).
5. Heatmap row counts equal the β-profile n per region; planted phase population → clean diagonal,
   noise population → no structure beyond the sort.
6. β-route vs curve-route peak agreement reported (circular correlation), disagreement kept as a result.
7. Every regional panel: n mice, per-mouse points, single-mouse caveats; primary contrast with
   `rate_match` and the boundary margin.
8. `check_mirror_parity.py` OK; PFC figure directory unlocked before writing; nothing committed.

## 10. Background results that bear on the reading (from `GLM_V3.md` §9)

- In LEC the raw CPDs of `goal_progress` and `time_from_reward` are within ±0.001 of zero in every
  gp+tfr arm, and **which one wins depends on the tfr coding**: decile → tfr +0.0006 / gp −0.0004;
  uniform → gp +0.0005 / tfr −0.0004 (cap 30). The joint `progress_or_time` block (0.0026 LEC, 0.0116
  PFC) is the coding-independent number. In the gp-only arm gp is 0.0017 (LEC, 65 % significant) and
  **0.0087 in PFC (83 %, larger than place)**.
- β profiles (`GLM_V3.md` §9.10): bin 0 = first tenth of the leg = consumption. gp-only: every LEC
  region above the consumption bin for the whole leg, CA1/HPF highest, ENTl-sup lowest; 9–21 % of
  gp-significant neurons fire most *during* consumption. With tfr in the model the early-leg component
  leaves gp; the entorhinal residual falls through the leg. PFC ramps to a mid-leg peak (43 % of its
  neurons peak at 0.4–0.7, against ≤ 35 % in any LEC region).
- Time-from-reward β profiles: 11–21 % of tfr-significant neurons are first-bin (consumption) cells,
  18–26 % peak in the sparsest last bin.
- Regional: place SUB/ProS > CA1 > ENTl-deep > ENTm > ENTl-sup in every arm; ENTl-deep > SUB/ProS in
  tfr (+0.001, 4 mice) is a candidate; no regional difference in gp anywhere.
