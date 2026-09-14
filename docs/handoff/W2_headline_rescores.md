# W2 — Re-score the cached headlines by region

**Status: not started.** Depends on W1's production fit. Read `README.md` first.

This is what the original `docs/ANATOMY_SPLIT_PLAN.md` was written to deliver: take each
existing headline, join it to anatomy, and report it per region under the inference design in
`README.md` §3.

## The questions, in priority order

1. **Is the place code LEC's?** Place tuning and place Δr² by group. Primary evidence is
   *within* ah10 and ly05, the two mice that record ENTl-deep, SUB/ProS and CA1/HPF
   simultaneously. Prior (and the W0 positive control) says place concentrates in CA1/HPF and
   SUB/ProS.
2. **Are the selectivity clusters brain regions in disguise?** Contingency of
   `clustering__cluster_result` × `group`, with a within-recday permutation on the labels.
   **Both outcomes are reportable**: a strong association means the clustering has been
   recovering anatomy; a null is a *stronger* version of the current claim.
3. **Where does task-phase periodicity live?** h4 power / periodic fraction by group.
4. **Where are the reward-time cells?** Poke duration-split offsets by group.
5. **Goal-progress and loop-anchored (`since_A`) tuning by group.**
6. **HD tuning by group** — meaningful for the first time, since every previous HD result was
   noise (see `W1_production_glm.md` §1.3). Prediction: ENTm and SUB/ProS ≫ ENTl. This doubles
   as a third independent validation of the anatomy map, because it predicts a specific region
   from a completely separate data stream. **ly07 is the ENTm mouse — an ENTm HD result is an
   ly07 result; say it that way.**

## Blockers and gotchas specific to W2

**Task-phase periodicity must be re-run, not re-scored.** `data/figures/taskphase_periodicity/
cells_LEC.pkl` has 2556 rows over **22** recdays, is missing `ah08_20250624_20250625` entirely
(186 units, from the mouse carrying 90% of ENTl-sup), and has **no `neuron` column** — so the
join would be positional-only and silently wrong. Re-run `run_taskphase_ring.py` /
`taskphase_periodicity.py` over all 25 recdays and **add an explicit `neuron` column**.

**Old caches hold the wrong day's spikes.** 28 pre-refit pickles have 91 units for
`ly05_20250618_20250619` where the correct count is 109. `STALE_CACHE_RECDAYS` still guards
them; `recday_registry.is_post_refit_section` exempts the new production fits. Use
`load_glm_results(..., apply_exclusions=True)` and let it do the work.

**`tuned_dict` is ternary.** `mean(x != 0)` for the tuned fraction — and remember the binary
fraction is saturated (80% place-tuned, 70–86% across regions) and should not carry a regional
comparison. Use Δr², bias-corrected.

## The goal-progress heatmaps, split by region

The first visual anyone will want: **neurons × 90 goal-progress bins, smoothed, z-scored, across
4 example tasks, pooled across mice, one row of panels per region.**

Reference implementation is **cell 257** of
`code/LEC_sploratory_analysis_with_glm_and_population.ipynb`:
`(n,360)` → `reshape(n,4,90).mean(axis=1)` → `gaussian_filter1d(sigma=3)` → per-neuron z-score →
sort by peak in task 1 → 4 `imshow` panels.

Four changes from that reference, each of which matters:

1. **Build the curves with `remapping_rotation_analysis.build_session_tuning`** rather than
   reading `Smoothed_norm`. Same `(n,360)` object, guaranteed present for all 25 recdays, and it
   makes the heatmap and W3's rotation analysis provably the same curves.
2. **`sessions[:4]` is not 4 tasks.** Sessions [0,3] and [4,7] repeat the same task, so it gives
   3 unique tasks plus a repeat. Use 4 **unique** tasks — or deliberately make panel 4 the
   **repeat of task 1**, labelled as the X-vs-X' control, which puts the noise ceiling next to
   the remapping in the same figure.
3. **Do both sorts.** Panel 1 self-sorted (the reference version) *and* a split-half version
   where `sort_idx` comes from half of task 1's trials and the other half is displayed. Sorting
   on task 1 and displaying task 1 is circular — that panel is a clean diagonal even for pure
   noise. The consistency across panels 2–4 is the point either way. Same `sort_idx` across
   panels within each version.
4. **One row per region, pooled across mice**, with composition in the caption.

**Sanity check:** a shuffled-firing version of the figure must show **no** diagonal in any
panel. If panel 1 still looks structured, the sort is circular.

## Deliverables

- `code/LEC_anatomy_split.ipynb` — W0 gates (already written) plus the re-scores and heatmaps.
- Figures under `data/figures/anatomy_split/`.
- A section appended to `code/ANATOMY_SPLIT.md`.
