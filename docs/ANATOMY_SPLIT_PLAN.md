# Split every LEC headline result by actual brain region

**Status:** ready to start · **Prerequisites:** all met (see §1) · **Written:** 2026-09-01
**Supersedes:** the 2026-08-25 plan *"The anatomy hunt"*. Two of that plan's factual claims
are now wrong — see §2 before doing anything.

---

## 0. The one-paragraph brief

The "LEC" dataset is not LEC. Allen-atlas registration of every probe (now hand-corrected
and verified) shows the recorded bank spans lateral entorhinal cortex, medial entorhinal
cortex, subiculum/prosubiculum and CA1, in proportions that differ wildly between mice.
Every pooled headline in this repo — place tuning, the goal-progress arc, the selectivity
clusters, task-phase periodicity, the reward-time cells — is an average over that mixture,
and no analysis code has ever read the anatomy. The per-unit region labels now exist and
are aligned to `Neuron_raw` rows. The job is to re-score the cached per-neuron results by
region, decide which headlines are regional and which survive the split, and separately fix
a broken head-direction regressor that has made every HD result in the repo noise.

---

## 1. What is already done — do not rebuild this

| Thing | Where | State |
|---|---|---|
| Probe fits, hand-corrected and sign-verified | `data/preprocessed_data/brainreg/<mouse>/ProbeA_*.{htsv,json}` | done, `final_fit_summary.csv` |
| Region-assignment module | [region_assignment.py](../code/histology_refit/region_assignment.py) | done |
| Per-unit region arrays | `data/processed_data/unit_regions.pkl` | **25 recdays**, aligned to `Neuron_raw` rows |
| Per-channel + per-unit tables | `data/processed_data/{channel_regions,unit_regions}.csv` | 9505 channels / 2851 unit-recordings |
| Census notebook + figures | [LEC_region_census.ipynb](../code/histology_refit/LEC_region_census.ipynb), `data/figures/region_census/` | done |
| Methods record | [PROBE_REFIT.md](../code/histology_refit/PROBE_REFIT.md) | done, §1–9 |

`unit_regions.pkl` is `{recday: DataFrame}` with one row per QC unit **in
`QC_single_units.npy` order**, i.e. row *k* is `Neuron_raw` row *k*. Columns:

```
mouse  block  order  unit_id  max_channel  channel_row  shank  y_um  ap_i  acronym  group  recday
```

`group` is the coarse label (`ENTl-sup`, `ENTl-deep`, `ENTm`, `SUB/ProS`, `CA1/HPF`,
`fibre/other`); `acronym` is the fine Allen label; `y_um` is depth along the shank (0 = tip);
`ap_i` is atlas AP voxel.

**The join is positional and needs no matching logic:**

```python
import pickle
regions = pickle.load(open('data/processed_data/unit_regions.pkl', 'rb'))
tuned   = pickle.load(open('data/glm_outputs/LEC/distance_gp__tuned_dict.pkl', 'rb'))
g = regions[recday]['group'].values          # (n_neurons,)
t = tuned[recday]                            # (n_neurons, 9) bool
assert len(g) == len(t)                      # ← make this a hard assert everywhere
```

### The corrected census (use these numbers, not the old plan's)

Unit-recordings per region, per mouse (`unit_regions.csv`, 25 recdays):

| mouse | ENTl-sup | ENTl-deep | ENTm | SUB/ProS | CA1/HPF | fibre/other | total |
|---|---|---|---|---|---|---|---|
| ah08 | **611** | 181 | 0 | 0 | 0 | 0 | 792 |
| ah10 | 9 | **378** | 22 | 176 | 113 | 74 | 772 |
| ly05 | 49 | **182** | 34 | 70 | 136 | 36 | 507 |
| ly06 | 13 | **133** | 49 | 137 | 0 | 24 | 356 |
| ly07 | 0 | 90 | **179** | 144 | 0 | 11 | 424 |
| **total** | 682 | **964** | 284 | 527 | 249 | 145 | **2851** |

Channels follow the same pattern (ENTl-deep 4390, ENTl-sup 1570, SUB/ProS 1495, ENTm 1240,
CA1/HPF 535, fibre/other 275, of 9505).

---

## 2. What changed since the 2026-08-25 plan — read this or you will carry a false prior

**(a) ah10 is not a hippocampal mouse. It is the *most* LEC mouse in the cohort.**

The old plan's census came from the automated `brainreg_probe` fits, and those fits were
wrong. ah10's plane was tilted 25.4° in AP with a compensating −24.3° in-plane rotation —
the signature of a mis-fitted PCA plane — which truncated its depth to 2205 µm and put the
contacts in CA1/ProS/SUB. After hand correction (lateral 7.9°, AP −2.1°, depth 3273 µm,
tip in ENTl5) ah10 has the **highest deep-ENTl occupancy in the cohort (0.60)** and 378
ENTl-deep units.

So the old plan's stated prediction — *"ah10 = the most place-tuned LEC mouse"* — is
**inverted**, and any positive control built on it will fail for the wrong reason. Discard
the old per-mouse census entirely.

**(b) The cached results are stale, in two directions.**

The `ly05_20250618_20250619` recday paired 06-18/19 behaviour with 06-20/23 spikes
([BUG_ly05_recday_mismatch.md](BUG_ly05_recday_mismatch.md)). That has since been fixed:
both ly05 days were re-extracted, the 18 bad arrays are quarantined as
`*.INVALID_ly05_recday_mismatch`, and **the cohort is now 25 recdays, not 24**
(`session_inds_dic.pkl` and `tasks_dic.pkl` confirm; `data_dic_lec.pkl` rebuilt 2026-09-01).

The cached GLM / selectivity / task-phase pickles predate all of this. They hold **24**
recdays, one of which (`ly05_20250618_20250619`) is the corrupted pairing, and none of which
is the new `ly05_20250620_20250623`. Handling in §4.

`unit_regions.pkl` is built per **sorted block**, so it was never affected and already
covers all 25.

**(c) The join method in the old plan was wrong, and the correct one is already built.**

The old plan proposed matching `unit_locations.npy` (x, y) to the nearest anatomy site. The
superseded `find_LEC_units` did something worse — treated the channel *name* (`CH123`) as a
row index. Dropped channels are **interior** (ah08 is missing CH98, 120, 153, 278, 286, 324,
348), so every channel after the first gap is shifted: that put **47.6% of units on the
wrong shank**. `region_assignment.py` uses the `recording_attributes.json` name→row map and
validates against an independent templates-derived peak channel at **92.5% mean agreement**
(vs 19.5% for the naive arithmetic). Just use `unit_regions.pkl`.

---

## 3. The inference design — this is the part that decides whether any of it is publishable

The old plan said within-recday contrasts are "controlled for everything behavioral". True,
and necessary, but it is not sufficient, for two reasons that the corrected census makes
plain.

### 3.1 Recdays are not independent replicates; mice are

The 5 recdays of a mouse are the **same probe in the same brain**, re-sorted. The same
physical neuron plausibly appears in several blocks. So 2851 is a count of
unit-*recordings*, and the effective n for any anatomical claim is the **number of mice**,
which is at most 5 and for most contrasts is 1–3.

Counting recdays where both groups have ≥20 units:

| contrast | recdays | mice | usable as |
|---|---|---|---|
| **ENTl-deep vs SUB/ProS** | 11 | **3** (ah10, ly06, ly07) | **primary** — the only contrast replicated across ≥3 animals |
| **ENTl-deep vs CA1/HPF** | 9 | 2 (ah10, ly05) | secondary |
| ENTl-sup vs ENTl-deep | 5 | 1 (ah08) | descriptive only — but it is the clean *layer* contrast |
| ENTm vs SUB/ProS | 5 | 1 (ly07) | descriptive only |
| SUB/ProS vs CA1/HPF | 4 | 1 (ah10) | descriptive only |

**Consequence to state in every figure caption:** ENTl-sup's 682 units are 90% ah08
(611/682). Any pooled ENTl-sup claim is an ah08 claim wearing a region label. Same logic for
ENTm (63% ly07).

**Recommended statistics:** compute the effect per recday, then combine with **mouse as the
resampling unit** (cluster bootstrap or a mixed model with recday nested in mouse and region
as a fixed effect). Report the per-mouse effects individually alongside the pooled one — with
n=3 mice, the reader should see all three points. Do not report a pooled p-value over 2851
units as if they were independent; it will be wrong by a large factor.

**Permutation null:** shuffle `group` labels **within recday**, which preserves each
recday's region composition and n. A null that shuffles across recdays is degenerate — it
would break the mouse↔region association that is the whole confound.

### 3.2 Region is confounded with depth on the probe, and therefore with unit quality

Region labels are *derived from* the unit's max-amplitude channel position. Depth along a
shank also drives spike amplitude, isolation quality and yield. So a "regional difference"
can be a recording-quality gradient wearing an anatomical label. This is the most likely way
this analysis produces a false headline.

Three mitigations, in increasing strength — do at least the first two:

1. **Report quality per group first, before any tuning result.** `mean_rate_hz`, `sd` and
   `r2_full` are already in `clustering__meta.pkl` (2742 rows, keyed `recday` × `neuron`).
   If groups differ in mean rate, say so up front.
2. **Rate-matched subsampling.** Re-run the primary contrast on rate-matched subsets and
   report both. If the effect survives matching, it is not a yield artefact.
3. **Within-shank comparison** where a single shank spans two regions — the cleanest
   version, since it removes even the shank-to-shank differences. Check availability from
   `unit_regions[recday].groupby(['shank','group']).size()`.

### 3.3 Registration precision, honestly stated

The local deformation field runs **8.06–12.95 µm per voxel** depending on subject and axis,
so structure boundaries carry that much positional uncertainty. Run a **boundary-margin
sensitivity check**: repeat the primary contrast excluding units whose `y_um` places them
within ~50 µm of a region transition along their shank. If the result depends on the
boundary units, it is not robust.

**ly05 is the weakest anatomy in the cohort** — its DiI was unusable (167 points, 97 µm
span), so its fit is a hand placement carrying `fit_disagrees_with_dye_axis(48°)`,
`low_signal_coverage` and `high_residual` flags. ly05 is one of the two mice supplying the
ENTl-deep vs CA1/HPF contrast, so report that contrast **with and without ly05**.

### 3.4 Not an issue, so don't spend time on it

Shank *numbering* is an exact degeneracy (mirroring `u_axis`, `offset_x`, `theta` reproduces
positions to 0.0 µm), and `SHANK_ORDER_VERIFIED` is still `False`. But anatomy *at a physical
position* is invariant and verified, so region labels are unaffected. Only claims of the form
"shank 0 specifically" need the caveat.

---

## 4. Stage 1 — align to the 25-recday cohort and gate the join

Before any science. Everything here is cheap.

1. **Re-run the length gate** for all 25 recdays: `len(unit_regions[recday])` must equal
   `data_dic[recday][0]['Neuron_raw'].shape[0]`. This is the check that originally caught
   the ly05 bug; it lives in the last cells of `LEC_region_census.ipynb` and should now pass
   on 25/25 against the rebuilt `data_dic_lec.pkl`. **Hard assert.**
2. **Decide the cached-result cohort.** The caches hold 24 recdays keyed by name. Drop
   `ly05_20250618_20250619` from every cached re-score (its 91 units are the wrong day) and
   note that `ly05_20250620_20250623` is absent. The other 23 are valid. That leaves ly05
   contributing 3 of 5 recdays — enough to keep it in the ENTl-deep vs CA1/HPF contrast, but
   say so explicitly in the figure.
   - `glm_analysis_v2.load_data_dic(...)` and `load_glm_results(..., apply_exclusions=True)`
     already exist and route through [recday_registry.py](../code/recday_registry.py). Use
     them rather than raw `pickle.load`. `EXCLUDE_RECDAYS` is currently empty — set it for
     this analysis rather than filtering ad hoc in the notebook.
3. **Positive control, rewritten.** The old plan's control (place tuning peaks in ah10) is
   dead — see §2a. The replacement, which is still a genuine literature prior and now points
   the other way: **place/spatial tuning should be higher in CA1/HPF and SUB/ProS than in
   ENTl-sup**, and this is testable *within* ah10 and ly05, which record ENTl-deep, SUB/ProS
   and CA1/HPF simultaneously. Run it on the `distance_gp_filtered` section before trusting
   anything else. **If it fails, stop and debug the mapping — do not interpret.**
   - Note: only `distance_gp`, `pokes_filtered` and `since_A_filtered` have a cached
     `__tuned_dict.pkl`. For every other section, derive it with
     `compute_tuning_arrays(glm_results, permutation_results, ...)`
     ([glm_analysis_v2.py:1677](../code/glm_analysis_v2.py#L1677)) from the cached
     `__glm_results.pkl` + `__permutation_results.pkl`, which do exist. No refit needed.
4. **Second, independent control:** unit depth `y_um` must order the regions consistently
   with the insertion trajectory in every mouse (deepest structures at the tip). This is a
   pure-geometry check and is cheap.

---

## 5. Stage 2 — re-score the cached headlines by region

No refits. Join `unit_regions` onto each cached per-neuron result and report per group, with
the design from §3: per-mouse effects shown individually, pooled combined over mice, and a
within-recday permutation null.

Verified cache locations and structures (checked 2026-09-01):

| # | Result | File | Structure |
|---|---|---|---|
| 1 | GLM tuning fractions | `data/glm_outputs/LEC/{section}__tuned_dict.pkl` (only 3 sections; else derive — see §4.3) | `{recday: (n_neurons, 9) bool}` |
| 2 | CPDs per variable | `data/glm_outputs/LEC/{section}__cpd_results.pkl` | `{recday: {neuron_idx: {var: float}}}` |
| 3 | Selectivity clusters | `data/glm_outputs/LEC_selectivity_geometry/clustering__{cluster_result,meta}.pkl` | meta = DataFrame(recday, mouse, neuron, r2_full, sd, mean_rate_hz, keep), 2742 rows |
| 4 | Task-phase periodicity | `data/figures/taskphase_periodicity/cells_LEC.pkl` | per-cell h4 power / periodic flags — **verify structure before use** |
| 5 | Poke duration split | `data/glm_outputs/LEC/pokes_filtered__duration_split.pkl` | consumption-vs-timing offsets |
| 6 | Loop-anchored | `data/glm_outputs/LEC/since_A_filtered__{tuned_dict,cpd_results}.pkl` | as #1/#2 |

Sections available: `distance_gp`, `distance_gp_filtered`, `distance_gp_state_filtered`,
`baseline_filtered`, `extended_cpd`, `extended_cpd_filtered`, `pokes_filtered`,
`since_A_filtered`. CPD variables per neuron: `place`, `goal_progress`,
`goal_progress_distance`, `speed`, `acceleration`, `time_from_reward`, `time_to_reward`,
`distance_from_reward`, `distance_to_reward`, `time_any`, `gp_any`, `__r2_full__`.

The questions, in priority order:

1. **Is the place code LEC's?** Place tuning and place CPD by group. Prior says no — it
   should concentrate in CA1/HPF and SUB/ProS. Primary evidence is within-ah10 and
   within-ly05, where all three groups are recorded at once.
2. **Are the selectivity clusters brain regions in disguise?** Contingency table of cluster
   × group, with a within-recday permutation null on the region labels. Both outcomes are
   reportable: a strong association means the clustering has been recovering anatomy; a null
   result means the cell types are genuinely within-region and is a stronger claim than
   currently made.
3. **Where does task-phase periodicity live?** h4 power / periodic-cell fraction by group.
   The ABCD progress code is the lab's headline; whether it is ENTl-proper or shared with
   SUB matters for the story.
4. **Where are the reward-time cells?** Poke duration-split offsets by group.
5. **Goal-progress and loop-anchored (`since_A`) tuning by group** — same treatment.

For each: report the effect per mouse, the pooled estimate with mice as the resampling unit,
the rate-matched version, and the boundary-margin sensitivity version. A result that moves
under any of the three is reported as unstable, not as a finding.

---

## 6. Stage 3 — fix the head-direction regressor, then refit once

**The bug, confirmed present today.** `HD_raw` is stored as `(T, 2)` —
`[back2mid_deg, earL2earR_deg]`, defined in
[sleap_preprocess_lEC.ipynb](../code/preprocessing/sleap_preprocess_lEC.ipynb) as the angle
head_back→head_mid and the angle ear_L→ear_R, which are ~90° apart.
[glm_analysis_v2.py:697](../code/glm_analysis_v2.py#L697) does:

```python
HD = session_data['HD_raw'].flatten()
```

That interleaves the two columns into a `2T`-length vector.
[`truncate_all_arrays`](../code/glm_analysis_v2.py#L799) then cuts every array to the
shortest (`max_index = min(lengths)` = `T`), so the GLM receives `HD[:T]` — which covers
only the **first half of the session**, interleaved with a 90°-rotated copy of itself, and
misaligned with every other regressor from the second sample onward. **All 36 HD design columns in every LEC GLM ever fit are noise,
and the reported ~4% HD tuning is the false-positive floor.** True HD tuning is unknown.

**Fix:** use `HD_raw[:, 0]` (`back2mid_deg`) with a shape guard and a 1-D fallback. Mirror
into `mFC_data/code/glm_analysis_v2.py:875` (inert there — PFC has `HD_raw=None` — but the
two copies must not diverge; that is the repo convention).

**Validate the fix before refitting:** `earL2earR_deg − back2mid_deg` should cluster near
±90°. If it does not, the column identity is not what the preprocessing docstring says, and
the choice of column must be settled first.

**Then refit** one new cached section (`hd_fixed`, same settings as `distance_gp_filtered`,
100 permutations) over all 25 recdays via `run_or_load_glm`. Smoke-test 3 recdays first.
Non-HD CPDs must come back near-identical to `distance_gp_filtered` — only the HD columns
changed — and that comparison is itself the gate that the refit did what it claims.

**Report HD tuning by region.** Prediction: ENTm / SUB/ProS ≫ ENTl, since that is where HD
cells are expected. This does double duty — it is a candidate finding ("this dataset has HD
cells nobody could see") *and* a third independent validation of the anatomy map, because it
predicts a specific region from a completely separate data stream.

Note ly07 is the ENTm-rich mouse (179 ENTm units) and provides the ENTm vs SUB/ProS contrast
in a single animal, so an ENTm HD result will be an ly07 result — state it that way.

---

## 7. Deliverable

- **Notebook** `code/LEC_anatomy_split.ipynb` — census recap, Stage 1 gates with visible
  pass/fail, the per-region re-scores, clusters × anatomy, HD-fixed results.
- **Doc** `code/ANATOMY_SPLIT.md` in the repo's methods-doc style (see
  [PROBE_REFIT.md](../code/histology_refit/PROBE_REFIT.md) for the register): what the
  census is, which headlines moved, which are single-mouse, the three confounds of §3 and
  what was done about each.
- **Figures** under `data/figures/anatomy_split/`, using `apply_gridmaze_style`.
- Update auto-memory at the end: the corrected census, the HD fix, and whichever headlines
  turned out to be regional.

---

## 8. Verification checklist

1. `len(unit_regions[recday]) == Neuron_raw.shape[0]` for **25/25** recdays — hard assert.
2. Positive control (§4.3): spatial tuning higher in CA1/HPF and SUB/ProS than ENTl-sup,
   *within* ah10 and ly05. Failure blocks everything downstream.
3. Depth-ordering control (§4.4) consistent in all 5 mice.
4. Every region contrast reported with n mice stated, per-mouse effects shown, and mice as
   the resampling unit.
5. Primary contrast survives (a) rate matching and (b) the 50 µm boundary margin; the
   ENTl-deep vs CA1/HPF contrast additionally reported with and without ly05.
6. Permutation nulls shuffle `group` **within recday**.
7. HD: `earL2earR − back2mid ≈ ±90°`; fixed HD vector length == tracking length; `hd_fixed`
   non-HD CPDs ≈ `distance_gp_filtered`.
8. No claim rests on `ly05_20250618_20250619` from the stale caches.

## 9. Files

- **Create:** `code/LEC_anatomy_split.ipynb`, `code/ANATOMY_SPLIT.md`,
  `data/figures/anatomy_split/`.
- **Modify:** `code/glm_analysis_v2.py` + `mFC_data/code/glm_analysis_v2.py` (HD fix only,
  mirrored).
- **Reuse, do not rewrite:** `code/histology_refit/region_assignment.py`,
  `data/processed_data/unit_regions.pkl`, `code/recday_registry.py`,
  `glm_analysis_v2.{run_or_load_glm, load_glm_results, load_data_dic, compute_tuning_arrays}`.
- **Do not touch:** `code/histology_refit/probe_refit.py`, `probe_tool.py`, or any stored
  probe fit. The histology is settled and hand-signed-off.

---

## 10. Other veins found but not planned here

Available if wanted, in rough order of promise: ~50k unrewarded pokes never analysed
(poking the wrong tower as a probe of hypothetical-state coding); session-time axis
reset-vs-drift across the 6–9 sessions per recday (the PI's active workfront); 269 sleep
boxes computed and unreported; a full ABCDE five-goal recording week still unsorted; 21
object-exploration sessions with spikes and no analysis.
