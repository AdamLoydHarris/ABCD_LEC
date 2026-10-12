# v4 migration — refitting the LEC neural analyses on the corrected spike clock

Index for the v1 → v4 migration. What was refit, what was not and why, the per-recday change
table, and the acceptance-test outcome. The plan this follows is reproduced in §0.

The v1 numbers are not deleted anywhere. They are the comparison that shows the migration was
sound, and every write-up gains a new dated section beside them rather than a rewrite.

---

## 0. What v4 changes, and how much

`data/processed_data/data_dic_lec.pkl` (v1) is the build every result in this repo was computed
from. `data_dic_lec_v4.pkl` (2026-10-02) corrects the OpenEphys counter-restart fault documented
in `docs/handoff/SPIKE_CLOCK_V4.md`. Nothing was ever fitted on v2 or v3, so this is a one-step
v1 → v4 migration.

All 211 LEC task arrays differ v1 → v4; all 279 sleep arrays are byte-identical. The shift
distribution is concentrated:

| shift | sessions |
|---|---|
| < 1 bin (25 ms) | 70 of 195 testable |
| 1–18 bins | 125 |
| seconds | **8 — all `ly05_20250620_20250623`** (2.6–38.6 s, 91 neurons) |

16 of 25 recdays have a median shift under one 25 ms bin. So the expected result is a small,
mostly sub-bin perturbation everywhere plus one repaired recday, and that asymmetry is the
acceptance test (§5).

Structure is unchanged: 25 recdays, 2851 neurons, per-recday unit counts identical, and
`unit_regions.pkl` still aligns positionally on all 25 recdays. No anatomy rework. Sessions go
212 → 211 (one stray v1 array); 22 sessions gain `XY_raw`.

**Out of scope, because unaffected.** All PFC fits — PFC is built in memory from El-Gaby's
deposited per-session files, with no TTL, no sample counter and no pickle lineage, so it has
exactly one build. `code/EVENT_KERNELS.md` and `code/LFP_ANATOMY_CHECK.md` are already v4-native.
The El-Gaby LEC v4 re-export is **already running** under `mFC_data/lec_replication_run_v4/`
(created 2026-10-08) — coordinate with it, do not start a second.

v5 is additive and distant: same v4 clock, same v3 tracking, adds only July ABCDE, and its spec
requires the June part to be byte-identical to v4. Work fitted on v4 survives v5 untouched.

---

## 1. Stage 0 — plumbing and gates (2026-10-09)

No fits. The goal is that **no entry point can reach a data_dic by accident**, and that a v4
result can never land in a v1 output tree.

### 1.1 One resolver

`glm_analysis_v2.load_data_dic` already accepts `path=`, so the frozen engines needed **no edit**.
`code/recday_registry.py` gained the version block:

```python
DATA_VERSION = os.environ.get("TSA_DATA_VERSION") or "v4"   # empty env var == unset
version_suffix(version=None)      # '' for v1, '_v4' for v4 — the ONE tagging rule
data_dic_path(version=None, dataset=None)
norm_neurons_path(version=None)
session_inds_path(version=None)
recdays_for_version(version=None) # from the 1 KB session_inds_dic, not the 3.8 GB dict
resolve_data_version(version=None)-> (version, path); raises if the pickle is absent
add_data_version_arg(parser)      # the shared argparse flag
```

`'v1'` maps to the untagged `data_dic_lec.pkl`; every other tag suffixes. A typo raises and lists
what exists, rather than silently falling back to v1 — which is how a migration quietly fails.

The **empty string** case is explicit: `submit_ph_lec.sh` forwards `--export=ALL,TSA_DATA_VERSION=`,
which sets the variable to `""`, and `os.environ.get(k, "v4")` returns `""` for that. `or "v4"` is
what makes unset and empty behave the same.

`mFC_data/code/recday_registry.py` gained `version_suffix` only, and it **raises** on any non-None
version. PFC has one build; asking for a version there is a mistake, not a different tree. The two
copies deliberately differ, which is why `gp_tuning_width.py` was added to the byte-parity list.

### 1.2 Entry points

Every one resolves a path, prints the resolved basename, and derives its output tree from the same
version. The engines' internal v1 defaults are now unreachable from any entry point.

| file | flag | output tree follows the version |
|---|---|---|
| `code/run_glm_batch.py` | `--data-version`, `--expect-recdays` | `LEC_v4_shards/` → `LEC_v4/` |
| `code/w0_gates.py` | `--data-version` | records `data_version` in the pickle |
| `code/w3_remapping.py` | `--data-version` | `w3_remapping_v4.pkl` |
| `code/gp_tuning_width.py` | `--data-version` | `gp_leg_curves_v4/` |
| `code/gp_region_heatmaps.py` | `data_version=` | `gp_curves_v4.pkl` |
| `code/run_ph_batch.py` | `--data-version` | `data/ph_outputs_v4` |
| `code/run_taskphase_ring.py` | `--data-version` | `data/ph_outputs_v4` |
| `code/run_lda_reward_progress.py` | `--data-version` (LEC only) | `LEC_lda_v4/` |
| `code/slurm_v5/run_queue_v5.py` | `TSA_DATA_VERSION` | (V5 run dirs, Stage 5) |
| `code/glm_v3_synthetics.py` | `--data-version` | n/a |
| `code/w1_refit.py` | `--data-version` | the legacy v2-sections CLI |
| `code/w1_{smoke_test,cv_diagnostic,encoding_diagnostics}.py` | `--data-version` | n/a |
| `code/rave_visualiser.py` | `DEFAULT_PKL` from the registry | viewer only |

Verified routing, all three forms agreeing:

```
--data-version   ver  data_dic             shards         out     w3 cache             gp curves         PH out
None             v4   data_dic_lec_v4.pkl  LEC_v4_shards  LEC_v4  w3_remapping_v4.pkl  gp_leg_curves_v4  ph_outputs_v4
v1               v1   data_dic_lec.pkl     LEC_shards     LEC     w3_remapping.pkl     gp_leg_curves     ph_outputs
v4               v4   data_dic_lec_v4.pkl  LEC_v4_shards  LEC_v4  w3_remapping_v4.pkl  gp_leg_curves_v4  ph_outputs_v4
```

Audited afterwards: no `load_data_dic()` reaches a default path from any script in `code/` or
`mFC_data/code/` outside the engines themselves and `preprocessing/` (which builds the pickles and
must name them). The notebooks still load directly and are retargeted in Stages 3–4.

**`glm_analysis_v3.load_data_dic` already had a `version=` kwarg** (`:329`), default `None` →
the untagged v1 file; `code/event_kernels_checks.py:36` uses it. The entry points pass `path=`
instead, because that works identically on the frozen v2 engine and states the file outright.
Both reach the same place; neither engine default was edited.

**A latent bug found and fixed on the way.** `gp_tuning_width.cache_path / save_cache / load_cache
/ load_all_caches` all bound `cache_dir=CACHE_DIR` as a **default argument**, evaluated at import.
Rebinding the module global from `main()` would not have reached any of them, so `--data-version v1`
would have read v1 data and written it into the v4 cache. All four now resolve at call time, and
`build_all` computes its own `cache_dir` from the version it loaded rather than trusting the global.

### 1.3 Shell wrappers

`submit_glm_lec.sh` and `submit_ph_lec.sh` each unpickled the **whole 3.8 GB v1 dict on the login
node** just to list recdays, and hard-coded v1. Both now read `recday_registry.recdays_for_version`
(1 KB) and pass the version through — `submit_glm_lec.sh` as `--data-version`, `submit_ph_lec.sh`
as `--export=ALL,TSA_DATA_VERSION=`.

### 1.4 Three harness defects fixed in the same change

- **The config stamp recorded nothing about the data.** A v1 and a v4 shard produced byte-identical
  stamps and merged silently. `cfg_stamp` now carries `data_version`. Tested: a v1-stamped and a
  v4-stamped shard in one directory now raise, and the message names `data_version`.
- **`merge` asserted no recday count** — 18 shards merged and printed "18 recdays". `--expect-recdays`
  (default 25) now raises.
- **No exit trailer on the sbatch scripts**: a failure was visible only as a *missing* "All done!"
  line, which is why 87 of 1432 logs are unreadable. The `set +e` / `rc=$?` / `EXIT=$rc` / `exit $rc`
  pattern was added to `glm_lec.sbatch`, `glm_pfc.sbatch`, `ph_lec.sbatch`, `ph_pfc.sbatch`,
  `knn_state_decoding.sbatch`, `lda_reward_progress.sbatch`, `lagged_maps_pdf.sbatch` and the new
  `v4_tracking_gap_check.sbatch`. The ~30 kilosort merge/resume scripts were left alone: different
  workstream, several queued.

`check_mirror_parity.py` gained `gp_tuning_width.py` to its byte-identity list and passes.

### 1.5 Gates

| gate | state |
|---|---|
| mirror parity exit 0 | **PASS** 2026-10-09 |
| import smoke test, both trees, 19 modules | **PASS** |
| mixed-version merge refusal | **PASS** (names `data_version`) |
| §1.5.1 equivalence: `--data-version v1` reproduces its September shard | **PASS** 2026-10-09 |
| v1-vs-v4 on one recday (first asymmetry data point) | running |
| `w0_gates.py` on v4 | **PASS** 2026-10-09 (after an Object-session fix, below) |
| `w0_gates.py` on v1, for comparison | **PASS** — identical depth orderings, 4/5 mice |
| `glm_v3_synthetics.py --quick` on v4, LEC | **11 PASS / 1 FAIL** — control 1b, see §1.6 |
| `glm_v3_synthetics.py --quick`, PFC | **PASS** 12/12 |
| v1 vs v4 key inventory | **no key lost**, six gained, see §1.6 |

### 1.5.1 Equivalence — PASS, 2026-10-09

`ah10_20250616_20250617`, `core_progress_time` matched / 250 ms / decile / cap 30 s / tfr uniform
10b, refitted with `--data-version v1` into a scratch tree and compared to the September shard:

| artefact | max\|diff\| | |
|---|---|---|
| `glm_results` | **0** | gate |
| `cpd_results` | **0** | gate |
| `cv_results` | **0** | gate |
| `permutation_results` | 57.9 | excluded, see below |

The only config change is the added `data_version: 'v1'` key. The refactor moved no number.

**The reproducibility defect this exposed, and its size.** `run_glm_analysis` draws its in-sample
permutation shifts with an unseeded `np.random.randint` (`code/glm_analysis_v3.py:2151`), so the
null in `permutation_results` is a fresh draw every run; the cross-validated path is seeded
(`code/glm_cv.py:296`). Measured consequences on this recday, 154 neurons × 6 regressors, same
data, two runs:

- the **observed** statistic in `permutation_results[n][0]` is identical in both runs (it is
  deterministic); only the 100-sample null moves, by up to 57.9 against a null median of 1.49;
- the seeded CV p-values `p_freedman_lane__cpd` / `__delta_r2` / `p_cv` in `cv_results` are
  **identical**, 679 significant cells at α = 0.05 in both runs. These are what
  `glm_plots.p_values` reads, so **every GLM V3 figure is reproducible**;
- the in-sample tuning call `F_obs > percentile(F_null, 95)` (`glm_analysis_v3.py:2324`) is not:
  **23 of 924 calls flip between the two runs, 2.5%** — place 0.6%, goal_progress 4.5%,
  speed 0.6%, acceleration 2.6%, time_from_reward 3.9%, progress_or_time 2.6%.

That path is live: `compute_tuning_arrays` consumes `permutation_results`, and so do
`code/plot_gp_any_tuning_pfc_vs_lec.py` and the `LEC_glm_production`,
`LEC_selectivity_geometry` and `LEC_glm_tuning_variables` notebooks.

**Why this matters for the migration:** ~2.5% is the **noise floor for any v1-vs-v4 comparison of
tuning counts.** A tuning-count difference of that order is the unseeded null, not v4. The
CV-based CPD and p-value comparisons have no such floor and are the ones to lean on in §5.3.

Not fixed here. Seeding it changes what a fit computes, so it belongs in a versioned engine bump
with its own equivalence control — not inside a migration whose whole purpose is to hold the code
still while the data changes.

**A comparer bug, recorded not fixed.** `glm_v3_synthetics.max_abs_diff` goes straight to
`|A − B|`, and `-inf − -inf` is NaN, which it reports as "NaN in one result but not the other".
V3's `bin_occupancy` stores its outer bin edges as ±inf, so an *identical* uniform-tfr shard fails
against itself; the same block holds string fields, which cannot be subtracted at all.
`code/v4_equivalence_check.py` wraps it with an exact-equality pre-check. The underlying bug is a
safe one — a false FAIL, never a false PASS — and the module is byte-mirrored with 12 controls
depending on it, so it was left alone rather than changed mid-migration.

---

### 1.6 Two things v4 changes that the plan did not anticipate

#### Object sessions gain tracking, and `w0_gates` assumed they could not have it

`w0_gates.py` died under v4 on `KeyError: 'Locs_raw'` after passing its first two gates.
`code/v4_key_inventory.py` (new) compares every session's keys across the two builds:

| | v1 | v4 |
|---|---|---|
| sessions | 212 | 211 |
| with `HD_raw` | 188 | **211** |
| with `XY_raw` | 188 | **211** |
| with `Locs_raw` | 188 | **191** |
| with `session_type` | 0 | 211 |
| with `arena` / `objects` | 0 | 20 |

**No key is lost on any shared session.** Six are gained. The crash is the gap between
`HD_raw` (211) and `Locs_raw` (191): 20 sessions now have tracking but no maze nodes. They are
exactly the **Object sessions** — sessions 1, 4, 5 and 9 of every `*_20250624_20250625` recday,
all five mice, each carrying `arena` and `objects` and no `Task`. They are open-arena, so there
is no `Locs_raw` to compare an HD length against; in v1 they simply had no tracking at all, so
the gate's `HD_raw implies Locs_raw` assumption held by accident.

Fixed in `gate_hd_lengths` by skipping them and **saying so** rather than silently: the gate now
reports `191/191 sessions (20 Object sessions skipped: tracked under v4 but open-arena, so no
Locs_raw)`. w0 then passes on v4: length gate 25/25, HD columns 211/211, depth ordering 4/5 mice
(unchanged from v1), unit quality by region produced.

`elasticnet_regression_v5` is not exposed — it returns `None, 'missing Locs_raw'` and skips the
session with a reason. No other module makes the assumption.

Also fixed: `w0_gates` wrote to the untagged `w0_gates.pkl` regardless of version, so the first
v4 run replaced the v1 gate results. The output is now `w0_gates{suffix}.pkl`, matching every
other entry point, and the v1 file has been regenerated.

#### Control 1b fails on v4 because v2 and v3 are NOT equivalent once NaN exists

`1b v3 decile arm == v2 (core-5, cap 60)` passes on PFC and fails on LEC/v4 at
`max_abs_diff = 3121`. The cause is visible in the row counts: **v2 fits 29614 rows, v3 fits
29537** — exactly the 77 untracked bins of session 3 that v3 drops and v2 does not.

| | `glm_analysis_v2` | `glm_analysis_v3` |
|---|---|---|
| Speed aggregation | `'mean'` (`:871`) | `'nanmean'` (`:971`) |
| NaN-speed rows | **kept** | dropped, and reported (`:1736-1740`) |

v1 had no NaN anywhere, so the two engines agreed exactly and the control passed. v4 has NaN, so
they diverge. **This is the control working, not a regression** — but its premise is "v3 under
legacy flags reproduces v2", and that premise only ever held on NaN-free data. It should be
rescoped to say so.

**What v2 actually does with a NaN speed, which is the part that matters.** It does not crash and
it does not skip. `apply_onehot` ends with `onehot[~np.isfinite(values)] = 0` (`v2:1032`), so a
NaN value becomes an **all-zero row across every column of that regressor** — and under
`reference_coded`, an all-zero row is indistinguishable from one sitting in the reference bin. No
NaN reaches `X`, so v2's own degenerate-design guard (`:1761`) never fires and nothing is printed.
The row is silently **miscoded as reference**, for every tracking-derived regressor at once
(speed, acceleration, head direction, the distance regressors).

Cohort-wide exposure, from the Stage 1 measurement:

| | rows | share |
|---|---|---|
| v3 engine **drops** | 611 | 0.144% |
| v2 engine **zero-codes** | 646 | 0.152% |
| sessions affected | 8 of 141 | worst 5.23% of one session |

v2 catches 35 more rows than v3 because `'mean'` is NaN when *any* of the ten raw bins is, while
`'nanmean'` needs all ten.

**This matters because five of the seven sections run on v2** — `all_regressors` (the 16-regressor
production model), `distance_gp_state_filtered`, `distance_gp_filtered`, `pokes_filtered`,
`since_A_filtered` — which is 125 of the 175 Stage 2 jobs. Only `core_progress_time` and
`core_progress_only` use v3.

##### Decision, user, 2026-10-09: run the v2 sections on v4 unchanged

The 646 rows stay miscoded. The reason is the migration's own logic: its purpose is to isolate
what the **clock** did, and that needs the code held still while the data changes. Pre-filtering
the NaN rows before v2 sees them would change the row set relative to the September v1 fits, so
every v1 → v4 difference would then mix the clock with a selection-rule change and the asymmetry
test (§5.3) would lose its meaning. Refitting those sections with v3 would break comparability
with every v1 number on them, and the section's engine is part of what that number means.

So: 0.152% of rows carry a reference-bin coding they have not earned, in 8 of 141 sessions, and
that is **recorded here as a known property of the v4 v2-engine fits**, not silently accepted.
Fixing it properly is an engine change — `'nanmean'` plus a row drop, as v3 already has — and
belongs in a future versioned bump with its own control, not in a migration.

##### Decision, user, 2026-10-09: rescope controls 1 and 1b to NaN-free data

`glm_v3_synthetics.nan_free(data, rd)` (new, byte-mirrored) linearly interpolates NaN out of
`XY_raw` before the two equivalence controls fit — which is exactly what v1's builder did. The
controls then test what they were written to test, that **v3 did not change the maths**, instead
of failing on a difference that is deliberate. v4's NaN handling is measured separately above, so
nothing is hidden by this: the control stops reporting a known, quantified design difference as a
surprise. No-op on PFC, which has no NaN.

---

## 2. Stage 1 — the NaN-block measurement (decision gate, blocking)

v4 leaves genuine NaN where v1 interpolated every tracking gap with no length limit. Untracked
rows are **dropped** from the fit (`run_glm_analysis` ~`:1719-1740`), costing ~7.8 min across 13
recdays, worst session 9.4%.

The open issue, found during planning and in no document: under the production
`downsample_mode='bin'`, `_DOWNSAMPLE_AGG['Speed'] = 'nanmean'` returns a **finite** speed if even
one of the ten raw 25 ms bins in a 250 ms block is tracked, while `FR` is `'sum'` over all ten. A
block mostly inside a gap therefore enters the fit with a full spike count against a one-bin speed
estimate. This did not exist in v1, where Speed was always finite.

Measured before anything is changed, by `code/v4_tracking_gap_check.py` (jobs 3708596 v4, 3708597
v1), which counts every row entering the fit by how many of its ten raw bins were tracked, and the
share of fitted spikes those rows carry.

### Result, 2026-10-09 — measured, and the answer is to change nothing

`code/v4_tracking_gap_check.py` run on both builds, 25 recdays, 141 GLM sessions, cap 30 s,
factor 10. The v1 arm is the control and behaves exactly as it must: **zero** NaN anywhere,
because v1 interpolated every gap.

| | v1 | v4 | share of the v4 fit |
|---|---|---|---|
| rows entering the fit | 423,760 | 423,149 | — |
| rows dropped (all-NaN speed) | 0 | 611 | 0.144% |
| rows diluted (1–9 of 10 raw bins tracked) | 0 | **35** | 0.008% |
| rows majority-untracked (< 5 tracked) | 0 | **11** | 0.003% |
| rows from a single tracked bin | 0 | **2** | 0.0005% |
| spikes carried by majority-untracked rows | 0 | — | 0.003% |

Worst single session is `ah10_20250613_20250615` s3 at 3 majority-untracked rows in 3437 (0.09%).

**Decision: no change, no flag.** The `nanmean` Speed against `sum` FR asymmetry is real — a
250 ms row can enter the fit with ten raw bins of spikes against a one-bin speed estimate — but it
does so 11 times in 423,149 rows, and those rows carry 0.003% of the fitted spikes. A
majority-tracked rule would move no number that any conclusion rests on, and adding an option
costs a flag, a synthetic control and a branch in the engine for nothing. The measurement is the
deliverable; it is recorded here so the question does not get re-opened from first principles.

The two-way fit the plan called for (as-is vs majority-tracked on the worst recday) was **not
run**: at 11 affected rows cohort-wide there is nothing for it to resolve.

Dropping 611 rows (0.144%) is the behaviour v4 intends — an untracked row has no speed and cannot
enter a fit that conditions on speed. It touches **8 of 141 sessions**:

| recday | session | rows in fit | dropped | % | diluted |
|---|---|---|---|---|---|
| ah10_20250613_20250615 | 3 | 3437 | 180 | 4.98 | 9 |
| ah10_20250613_20250615 | 7 | 4468 | 233 | 4.96 | 13 |
| ah10_20250616_20250617 | 3 | 4722 | 77 | 1.60 | 4 |
| ly05_20250616_20250617 | 1 | 2893 | 34 | 1.16 | 2 |
| ah08_20250616_20250617 | 6 | 2888 | 29 | 0.99 | 2 |
| ah10_20250613_20250615 | 1 | 4773 | 36 | 0.75 | 2 |
| ah10_20250613_20250615 | 6 | 4373 | 13 | 0.30 | 1 |
| ly07_20250624_20250625 | 7 | 3394 | 9 | 0.26 | 2 |

**The 9.4% figure does not apply to any fit.** `TRACKING_GAPS_PROVENANCE.md:33` records the worst
session as `ah08_20250616_20250617` s3, 4472 bins = 9.4% — that is *raw 25 ms bins as a fraction of
the whole session*, and **s3 is not in the GLM's session set** for that recday (which is
0, 1, 4, 5, 6). Both numbers are correct and measure different things; the GLM-relevant worst case
is `ah10_20250613_20250615` s3 at 4.98% of fitted rows. Quote the right one.

All 141 sessions reconcile exactly: `rows_in_fit(v1) − rows_in_fit(v4) == rows_dropped_nan(v4)` in
every one. Row membership depends on behaviour (`Locs`, `valid_transition_mask`, `GP_binned`), not
on the spike clock, so this is the expected result and it confirms the plumbing changed no
selection rule.

### A free look at the asymmetry test, before any fit

Total fitted spikes fall 78,903,003 → 78,762,694 (−0.178%). Splitting by cause: the 8 sessions that
lost rows account for −133,107, leaving **−7,202 that no row drop explains** — spikes the corrected
clock moved into or out of the fitted window. Restricting to the 133 sessions that lost **no** rows
isolates the clock, and the result is the acceptance test showing up before a single fit has run:

| recday | sessions | fitted spikes (v1) | change | % |
|---|---|---|---|---|
| **ly05_20250620_20250623** | 4 | 1,286,732 | **−5,987** | **−0.465** |
| ly05_20250624_20250625 | 4 | 903,316 | −68 | −0.008 |
| ah08_20250618_20250619 | 6 | 2,263,935 | −152 | −0.007 |
| … 21 further recdays … | | | | −0.006 … +0.003 |
| ah08_20250624_20250625 | 4 | 877,963 | +63 | +0.007 |

`ly05_20250620_20250623` moves **60× more than the next-largest recday**, and every other recday
moves by ≤ 0.008%. That is exactly the predicted shape: one repaired recday, sub-bin noise
everywhere else. 131 of 133 sessions changed at all, median |change| 0.0031%.

This is not a substitute for the CPD-level asymmetry test in §5.3 — a spike count entering the fit
is not a fit result — but it is an independent confirmation, from the data rather than from the
clock diagnostics, that v4 moves one recday and leaves the other 24 alone.

---

## 3. Stage 2 — the GLM refits

375 LEC jobs, nothing for PFC. Not yet submitted; blocked on §1.5 and §2.

| fits | sections | jobs | shards → out |
|---|---|---|---|
| production + V3 2×2 + gp-only | `all_regressors`, `core_progress_time` × {cap30,cap60} × {tfrU,tfrD}, `core_progress_only` × {cap30,cap60} | 175 | `LEC_v4_shards/` → `LEC_v4/` |
| normalisation ladder | `core_progress_time` cap30 tfrU and `core_progress_only` cap30, × {`session-z`, `none`} | 100 | `LEC_normalisation_v4_shards/` → `LEC_normalisation_v4/` |
| legacy v2 sections (feed W2) | `distance_gp_state_filtered`, `distance_gp_filtered`, `pokes_filtered`, `since_A_filtered` | 100 | `LEC_v4_shards/` → `LEC_v4/` |

### The four legacy sections have NO comparable v1 baseline (found 2026-10-09)

They are worth fitting, but **not as a migration**: there is nothing valid to diff them against.
Their v1 outputs in `data/glm_outputs/LEC/` fail on four counts at once.

| | v1 legacy sections | v1 `all_regressors` / V3 arms |
|---|---|---|
| per-recday shards | **none** — merged pickles only | 25 each |
| naming | old `{section}__{artefact}.pkl`, no width/scheme/regset | `{section}__full_250ms_decile__...` |
| config stamp | **none** — predates stamping, so the fit parameters cannot be verified | stamped (250 ms, decile, full, cv_perms 100) |
| `cv_results` | **absent** — fitted without cross-validation | present |
| recdays | **24**, not 25 | 25 |
| dated | 2026-06-13 .. 2026-08-10 | 2026-09-02 |

The last row is decisive. All four sections include `head_direction`, and all four pickles
**predate the 2026-09-01 HD flatten fix**, under which `HD_raw` was `(T, 2)` but got flattened —
so every HD result before that date is noise. These v1 fits were already superseded before this
migration began.

So the acceptance test (§4.5) cannot be run on them, and that is not a gap in the migration: the
v4 fits are simply the **first correct version** of these four sections, with cross-validation, a
config stamp, 25 recdays and a working HD regressor. They should be reported as new results, not
as a v1 → v4 change.

`all_regressors` is **not** in this position — its v1 shards are dated 2026-09-02, stamped, 25 of
them, with `cv_results`. It is fully comparable.

Measured per-recday cost from the September logs: `all_regressors` median 18.2 min (max 40.4), V3
arms median 3.6 min, normalisation arms 1.7–4.2 min. Roughly 30 h CPU.

**Where to submit (changed 2026-10-09).** `cpu` holds only 260 CPUs and this account's fairshare
is 0.006, so a job lands at priority 60 with ~98 ahead of it — congestion there is the normal
state, not an exception. The 2026-10-02 pre-approval named `gpu_lowp`, but `sbatch` now warns:

> WARNING: In the future, CPU-only jobs will be blocked in public GPU partitions.
> Please use 'cpu_backfill'.

`cpu_backfill` has 2632 CPUs (1905 idle, 31 pending) against `gpu_lowp`'s 1504 (1020 idle), and
is what the admins ask for, so **the 375 Stage 2 jobs should go to `cpu_backfill`**. Both overflow
partitions are `PreemptMode=REQUEUE` and a preempted job restarts from scratch; the per-recday
shard pattern already makes that cheap (median 3.6 min per V3 arm). Plain `cpu` is
`PreemptMode=OFF`, so a job that must not restart belongs there.

---

## 4. Stages 3–5 — the rest of the neural surface

**Stage 3, GLM-derived.** Rebuild `gp_curves_v4.pkl` and `gp_leg_curves_v4/` (one data_dic load),
then re-derive W5 peak-vs-width; re-run both GLM V3 notebooks and the normalisation comparison
against the `_v4` trees; regenerate the LEC-vs-PFC panels (new LEC, unchanged PFC).

**Stage 4, other neural analyses**, each to a `_v4` tree: W3 remapping, CCGP state pairs, kNN state
decoding, LDA reward/progress, selectivity geometry, task-phase periodicity, time manifold,
time-vs-progress. Task-phase periodicity carries a **known pre-existing defect** — `cells_LEC.pkl`
covers 22 of 25 recdays and has no `neuron` column (`docs/handoff/README.md` §5.3) — to be fixed on
the way through rather than reproduced.

**Stage 5, the expensive tails — sized 2026-10-09.** This is the only part of the migration whose
cost was not already measured, and it dominates everything else: **~420 CPU-h against Stage 2's 30.**

*Persistent homology.* From 49 completed job logs: 4.4 h per recday job, 215 CPU-h for a full
25-recday `--state both` sweep. Split by arm, from 989 logged per-session fits:

| arm | fits logged | median | total |
|---|---|---|---|
| wake | 273 | 777 s | 59 CPU-h |
| sleep | 716 | 787 s | 156 CPU-h |

**Sleep is 73% of the PH compute, and v4 does not touch its input.** The sleep arm does not read
the data_dic at all — `persistent_homology_analysis.load_sleep_files` reads
`data/processed_data/neuron_raw_mingyutest/Neuron_raw_<recday>_sb_*.npy` (259 files, dated
2025-09-09), and `SPIKE_CLOCK_V4.md:152` states the `_sb_` rest arrays are **not affected**,
because they "use origin 0 and no pyControl clock". So `--data-version` only reaches the wake half.

That does *not* make the 259 sleep pickles reusable as they stand. `analyse_sleep_session`
(`:846`) splits in two: the de novo sleep PH is wake-independent, but the block at `:872-876`
projects sleep onto the **wake** manifold (`project_onto_wake_manifold`) for the replay score, and
the wake manifold does move under v4. `primary_wake` is only populated when the job runs
`--state both`, so `--state sleep` alone silently drops the replay statistics.

Three options, to decide at Stage 5 rather than now:
1. full `--state both` sweep — 215 CPU-h, simplest, nothing to justify;
2. `--state wake` only (59 CPU-h) and keep the v1 sleep pickles, accepting that their replay
   coverage is scored against the old wake manifold — only defensible if no conclusion rests on it;
3. a `--reuse-sleep-ph` path that caches the de novo barcodes and recomputes only the projection.
   Cheapest, but it is a code change and needs its own control, so it is not free either.

*Elasticnet V5 LEC.* 14 LEC run directories (and 12 PFC, not in scope). From 18 logs carrying a
timing stamp: median 7.1 h per job, mean 14.7, **max 48.4 h** — close enough to a walltime ceiling
that the long arms need their `--time` checked before submission. A full LEC re-run is ~206 CPU-h.

Coordinate V5 with the El-Gaby v4 re-export already in flight under
`mFC_data/lec_replication_run_v4/`.

---

## 4.4 Stage 2 — COMPLETE, 2026-10-09

**375/375 jobs COMPLETED, 15/15 sections merged at 25 recdays each, zero failures.** Submitted to
`cpu_backfill` (ids 3709019–3709423). Preemption occurred and was a non-event: every preempted
job was requeued automatically and completed on retry. Each merge reports a consistent
`data_version: 'v4'` stamp and key-contiguity PASS.

Cohort median `r2_cv` for the production model rises 0.05605 -> 0.05837; 20 of 25 recdays improve.

### One recday still fits worse than the mean, and it is not the clock

`all_regressors` has 3 recdays with a negative median `r2_cv` under v1 and **1 under v4**, which
is why the "only one negative fit" criterion holds in 8 of 9 session-centred arms rather than 9.

| recday | r2_cv v1 | r2_cv v4 | change |
|---|---|---|---|
| `ah08_20250624_20250625` | −0.01835 | **−0.01358** | +0.0048 |
| `ah08_20250613_20250615` | −0.00407 | +0.00067 | +0.0047 |
| `ly05_20250620_20250623` | −0.05386 | +0.03959 | +0.0935 |

`ah08_20250624_20250625` was already negative under v1 and v4 **improves** it, so this is not a
v4 regression. It is model capacity: that recday has **5609 design rows against the cohort median
of 16690** (4 GLM sessions, not 6 — it is one of the `_20250624_20250625` recdays that spend
sessions on the open-arena Object task). Against the 16-regressor design's ~160 columns that is
~35 samples per parameter, and LOSO CV leaves each training fold three quarters of it. Its firing
is normal (157 spikes per row against a cohort median of 161), so this is quantity, not quality —
the same failure mode `glm_analysis_v2:1766` already warns about. The reduced V3 arms, with 5
regressors, have no negative fit on that recday at all.

## 4.5 Acceptance test — PASS across all 11 comparable arms, 2026-10-09

Every arm with a valid v1 baseline — 11 of the 15 sections; the four legacy v2 sections are
excluded for the reasons in §3. Compared against the September v1 shards, which §1.5.1 proved a
fresh v1 refit reproduces exactly. `code/v4_asymmetry_summary.py` runs all of them at once.

| | across all 11 arms |
|---|---|
| arms where `ly05_20250620_20250623` is the **most-moved** recday | **11/11** |
| separation from the next-largest recday | 2.31x – 4.77x |
| separation from the **median** recday | **10x – 20x** |
| its median `r2_cv` negative under v1, positive under v4 | **11/11** |
| session-centred arms where it is the only negative fit under v1 | 8/9 (see §4.4) |

The strongest separation is in `all_regressors`, the production model, at **4.09x**.

The two `nonorm` arms have 7 negative fits under v1 rather than 1: stripping the per-session
offset weakens every fit, since it carries ~77% of `r2_cv` (`GLM_V3_NORMALISATION.md`). That is a
property of the arm, not of the clock, and the bad recday is still rank 1 there (3.7x, 3.9x).

**A threshold this document previously carried and should not have.** An earlier version of the
summary demanded a 3x ratio and exactly one negative fit, and reported 6/11 arms passing. Neither
condition is pre-registered. The 3x failed three arms that separate perfectly well at 2.3–2.9x,
and "exactly one negative fit" cannot transfer to an arm whose baseline fit quality differs by
construction. The pre-registered claim is about **rank** — "a large move here is the expected
result; a large move in any other recday would need explaining" — and that is 11/11.

### The single-section result this generalises

| | |
|---|---|
| `ly05_20250620_20250623`, median held-out r² | **−0.01945 → +0.04951** |
| its \|change\| | 0.06896, **3.4x the next largest** (`ly06_20250624_20250625`, 0.02008) |
| median \|change\| over the other 24 | 0.00473 |
| recdays with median `r2_cv` **< 0** | v1: **only `ly05_20250620_20250623`**; v4: **none** |
| its per-neuron `r2_cv` rank correlation | **−0.60**, against >= 0.938 for every other recday |

The recday predicted in advance to be repaired was the only pathological fit in the cohort — the
only one whose model was worse than predicting the mean — and it is the only one repaired. The
other 24 move slightly and mostly upward (median +0.005), which is what a corrected clock should
do; four move very slightly down, all \|change\| <= 0.0031.

**A mouse-level structure that does NOT hold.** It is tempting to expect the restart-affected mice
(ah08 / ly05 / ly06) to move and ah10 / ly07 not to. They do not separate that way:
`ah10_20250613_20250615` has the third-largest change (+0.017), and ly07's median change (+0.0037)
exceeds ly05's (+0.0019). Not asserted.

### A third line of evidence, touching no GLM fit

`gp_curves_v4.pkl` and `gp_leg_curves_v4/` were rebuilt from the v4 dict (jobs 3709554/3709555,
~20 s each: 25 recday caches, 2851 neurons — the full cohort — and 182 sessions of 90-bin curves).

**`Smoothed_norm` is genuinely recomputed under v4**: all 182 shared sessions differ from their v1
values. Worth checking, because it is a *stored* array — had the v4 build carried it over, W5 and
`gp_region_heatmaps` would have silently stayed on the old clock while looking migrated.

Comparing the curves v1 → v4, per recday, independently of every GLM:

| | per-recday curve rho | max shift / curve SD |
|---|---|---|
| **ly05_20250620_20250623** | **0.9809** | **0.957** |
| worst other (`ly05_20250618_20250619`) | 0.9985 | 0.211 |
| median of the other 24 | 0.9999 | 0.041 |

The bad recday's curves move by nearly a full standard deviation of their own spread; the median
recday moves by 4% of one. Same recday, same direction, from a quantity no fit touches.

A softer pattern is visible here that the CPD statistics were too noisy to show: **the four other
ly05 recdays rank next** (rho 0.9985–0.9996 against >= 0.9997 for every other mouse). ly05 is the
mouse carrying the counter-restart recordings, so it is consistent — but it is four recdays in one
mouse, and it is recorded as an observation, not a claim.

### Two statistics were tried and rejected first

The automated verdict was **FAIL** before the statistic was right, and both rejected statistics
failed the same way — dominated by a regressor carrying no signal.

| statistic | why it fails | separation of the bad recday |
|---|---|---|
| relative change in median CPD | `acceleration`'s median CPD is −0.00024, so an absolute move of 0.0006 reads as "+83%" | — |
| `min` per-neuron CPD rank correlation | `acceleration` supplies the min in **23 of 25** recdays; the rank of a quantity that is zero up to noise is noise. Across the 24 non-pathological recdays it correlates with v1 fit quality at rho = +0.54 (p = 0.007) | gap +0.248 |
| median per-neuron CPD rank correlation | same confound, weaker (rho = +0.46) | gap +0.777 |
| **per-neuron `r2_cv` rank correlation** | the one used | gap **+1.542** |

Held-out r² is the right quantity: well scaled, signed, interpretable. Critically it gives the
test a **qualitative** criterion — a negative median r² means the model is worse than predicting
the mean, which is what misaligned spikes produce, and no noise floor can manufacture a sign
change. The diagnosis and both rejections are recorded in `code/v4_asymmetry_table.py` so they are
not re-derived.

### The pre-registered test — ANSWERED 2026-10-12

The LEC v4 El-Gaby chain completed: all 13 jobs exit 0, `Figure5_Figure6_part1` taking 26.5 h of
its 4-day walltime. It was left on the non-preemptible `cpu` partition rather than moved to
`cpu_backfill`; at 26.5 h a requeue would have been expensive, so that was the right call.

**State-tuned fraction** (`State_95`, the same definition the 19% came from), per recday:

| | v1 | v4 | change |
|---|---|---|---|
| **`ly05_20250620_20250623`** | **18.7%** | **36.3%** | **+17.6 pp** |
| next largest rise (`ly06_20250620_20250623`) | 41.8% | 50.7% | +9.0 pp |
| median over the other 24 | — | — | **−0.3 pp** |
| cohort | 50.0% | 50.9% | +0.9 pp |

The predicted recday has the largest rise, at twice the next, while the median recday does not
move and the cohort is unchanged. **But the repair is partial**: 36.3% against a cohort of 50.9%.
The prediction said "toward 50%" and it travelled about half the distance. That recday is still
in the bottom third, so the clock was not the only thing wrong with it.

**Anchored fraction.** A per-recday anchored fraction from
`Anchor_trial_Phase_state_place_anchored` gives `ly05_20250620_20250623` **36.0% -> 55.8%**
(+19.8 pp), again the largest rise, 3.4x the next (+5.8 pp), against a median change of +0.5 pp;
its rank rises from 1/25 (the cohort minimum) to 2/25. Cohort median 73.5% -> 71.8%.

**Caveat on that second number.** This is NOT the statistic the prediction quoted. The
BAD_RECDAY document gives the v1 anchored fraction as 14.3% with a cohort median of 58% and range
14.3–70.7%; the array above gives 36.0% and a median of 73.5%, over 330–915 entries per recday
rather than the 66–186 neurons. It is a related but different quantity, and which array produced
the 14.3% is **not yet identified**. The direction and the asymmetry agree under both, but the
document's exact anchoring statistic has not been reproduced.

Cohort anchoring from `anchoring_score.md`, for the record: `coherent_proportion` 0.4615 -> 0.4718.

`BAD_RECDAY_ly05_20250620_20250623.md:111` predicts "the state-tuned fraction to rise from 19%
toward 50% and the anchored fraction from 14.3% toward the 58% median". Those are **El-Gaby
replication** quantities — state tuning by his pref-phase mean test, and the Figure 5 / anchoring
statistics — produced by `mFC_data/mFC_schema-main_unblocked_lec/_preflight/`, not by our GLM.
`all_regressors` does carry a `task_state` regressor, but the fraction of neurons with a
significant `task_state` CPD is a *different definition*; it is a useful independent analogue and
will be reported as one, not as the pre-registered number.

**The chain that answers it is stuck.** The LEC v4 re-export under `mFC_data/lec_replication_run_v4/`
was created 2026-10-08 14:12 and has produced only tracking-gap logs since. All 14 jobs are
`PENDING` on the `cpu` partition:

| job | name | walltime | state |
|---|---|---|---|
| 3706265 | `unb_Figure2` | 24 h | PENDING, **Priority** (chain head) |
| 3706266 | `v4_bridge` | 30 min | Dependency |
| 3706267–70 | `Figure3_fast`, `Figure5_Regression` ×3 | 24 h each | Dependency |
| 3706293 | `Figure5_Figure6_part1` | **4 days** | Dependency |
| 3706294–97 | resume, `f7inputs`, `Figure7`, `v4_checks` | 24 h / 6 h / 24 h / 4 h | Dependency |

It is queued behind ~100 higher-priority jobs on a 260-CPU partition with this account's
fairshare at 0.006. Our Stage 2 jobs are **not** competing with it — they are on `cpu_backfill`.

**Moving it to `cpu_backfill` is not obviously right and was not done.** That partition is
`PreemptMode=REQUEUE`, and a requeued job restarts from scratch. That is fine for 3–18 min
per-recday GLM shards and bad for a 24 h job, let alone the 4-day one: it could be preempted
repeatedly and never finish. The options — wait, move and accept the restart risk, or split the
long notebooks into resumable pieces — are a scheduling judgement for the user.

---

## 5. Acceptance test (written in advance)

From `docs/handoff/BAD_RECDAY_ly05_20250620_20250623.md`:

> expect the state-tuned fraction to rise from 19 % toward 50 % … **A large move here is the
> expected result; a large move in any other recday would need explaining.**

1. **Equivalence** (§1.5.1): `--data-version v1` reproduces an existing shard exactly, on the
   deterministic artefacts.
2. **Gates**: w0 on v4 passes; synthetics pass on v4; mirror parity exit 0; merges report **25**
   recdays with a consistent `data_version` stamp and key-contiguity PASS.
3. **The asymmetry test.** Per recday, the change in each regressor's CPD and in `r2_cv`, v1 → v4.
   `ly05_20250620_20250623` should move far more than any other recday. **Any other recday moving
   comparably is a finding to chase, not a result to publish** — 16 of 25 shifted by under one bin.
4. **The repair test.** `ly05_20250620_20250623`'s state-tuned fraction rises from 19 % toward the
   cohort's ~50 %, and its fit quality stops being the worst of the 25.
5. **Conclusion-level diff.** For each write-up, whether its headline survives: the GLM V3
   gp-vs-tfr split, PFC's `goal_progress` > place, the ENTl-deep > SUB/ProS `time_from_reward`
   candidate, the W5 peak-vs-width V, W3's two results, the CCGP/kNN cross-task nulls.

Nothing is committed (standing instruction).

---

## 6. Reproducing Stage 0

A cluster job's output must land on **ceph**, not in a session scratchpad under `/tmp` — that is
the compute node's own disk (`/dev/sda2`), so the file is written on the node and is invisible
afterwards while the job still exits 0 and still prints "saved <path>". `data/` is gitignored, so
`data/_v4_migration_scratch/` is the right place. This cost one rerun of all four jobs.

The two equivalence fits must also go to **separate** shard directories: a shard is named
`{section}__{recday}__{artefact}.pkl` with no version in it, so a v4 job writes over a v1 job's
shard for the same recday.

```bash
S=data/_v4_migration_scratch
python code/check_mirror_parity.py                      # must exit 0 before any submission

# §1.5.1: refit one recday on v1 into a scratch tree, compare to the September shard
G="--section core_progress_time --regset matched --leg-cap-s 30 --tfr-scheme uniform \
   --width-ms 250 --scheme decile --permutations 100 --cv-perms 100"
sbatch --partition=cpu_backfill --job-name=equiv_v1 sbatch_files/glm_lec.sbatch \
  ah10_20250616_20250617 $G --data-version v1 \
  --shard-dir $S/shards_v1 --out-dir $S/out_v1
python code/v4_equivalence_check.py --a data/glm_outputs/LEC_shards \
  --b $S/shards_v1 --recday ah10_20250616_20250617 --expect-identical

# the same recday on v4, for the first asymmetry data point
sbatch --partition=cpu_backfill --job-name=equiv_v4 sbatch_files/glm_lec.sbatch \
  ah10_20250616_20250617 $G --data-version v4 \
  --shard-dir $S/shards_v4 --out-dir $S/out_v4

# Stage 1, the tracking-gap measurement (both builds; v1 is the zero-NaN control)
sbatch --partition=cpu_backfill sbatch_files/v4_tracking_gap_check.sbatch \
  --data-version v4 --out $S/gapcheck_v4.pkl
sbatch --partition=cpu_backfill sbatch_files/v4_tracking_gap_check.sbatch \
  --data-version v1 --out $S/gapcheck_v1.pkl

# the remaining Stage 0 gates
sbatch --partition=cpu_backfill sbatch_files/v4_gates.sbatch code/w0_gates.py --data-version v4
sbatch --partition=cpu_backfill sbatch_files/v4_gates.sbatch code/glm_v3_synthetics.py \
  --quick --data-version v4
sbatch --partition=cpu_backfill sbatch_files/v4_gates.sbatch \
  mFC_data/code/glm_v3_synthetics.py --quick
sbatch --partition=cpu_backfill sbatch_files/v4_gates.sbatch code/v4_key_inventory.py \
  --versions v1 v4 --out $S/key_inventory.csv

# the acceptance test, once Stage 2 has merged
python code/v4_asymmetry_table.py --v1 data/glm_outputs/LEC_shards \
  --v4 data/glm_outputs/LEC_v4_shards --out $S/asymmetry.csv
```

New files this stage: `code/v4_tracking_gap_check.py`, `code/v4_equivalence_check.py`,
`code/v4_key_inventory.py`, `code/v4_asymmetry_table.py`, `sbatch_files/v4_gates.sbatch`,
`sbatch_files/v4_tracking_gap_check.sbatch`.
