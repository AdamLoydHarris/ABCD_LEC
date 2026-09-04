# Probe re-fit and QC (`histology_refit`)

Manual correction of the `brainreg_probe` probe fits, with a QC harness that is
calibrated by synthetic controls and that knows the difference between judging an
algorithm and judging a person.

Run everything in the **`histology` conda env** (`~/.conda/envs/histology`); the
default `maze_ephys_si104` env lacks `skimage` and cannot import the tracing
module.

```python
import sys; sys.path.insert(0, 'code/histology_refit')
import probe_refit as pr
pr.run_synthetic_controls()      # gate first
pr.qc_table()                    # where every subject stands
```

## 1. Why this exists

Only the tip-most **705 µm** of the probe is recorded (384 channels, 96 per shank
over 4 shanks; `channel_positions.npy` has y ∈ [0, 705]). A depth or trajectory
error therefore does not degrade the anatomy gracefully — it relocates *every*
unit to the wrong structure.

The insertion was a nominal **10° lateral approach**. In `asr` orientation
(i = anterior→posterior, j = superior→inferior, k = right→left) a correct fit has
its tilt in the coronal/lateral plane at ~10°, ~0° in AP, and an in-plane
rotation `theta` ≈ 0 — a straight insertion has no reason to rotate the probe
within its own plane. Measured from the stored `ProbeA_fit_params.json`:

| mouse | LATERAL° | AP° | theta° | AP+theta | depth µm | width_scale | verdict |
|---|---|---|---|---|---|---|---|
| *expected* | *~10* | *~0* | *~0* | — | — | — | |
| ah08 | 15.0 | 12.3 | 0.1 | 12.4 | 4225 | 0.893 | plausible |
| ah09 | 5.9 | 9.0 | −3.3 | 5.7 | 4262 | **0.800**◄ | not in ephys cohort |
| ah10 | **−1.8** | **25.4** | **−24.3** | **1.2** | **2205** | **0.800**◄ | **fails** |
| ly05 | 17.9 | −1.4 | **+14.1** | 12.7 | 2775 | **0.800**◄ | **fails** |
| ly06 | 6.0 | 3.2 | −0.5 | 2.7 | 3971 | 0.851 | good |
| ly07 | 6.8 | 10.6 | 1.0 | 11.6 | 3769 | **0.800**◄ | ambiguous |

**The ah10 signature.** Its plane is tilted 25.4° in **AP** — the wrong plane
entirely for a lateral approach, with essentially no lateral tilt — and the
optimiser then rotated the probe back in-plane by −24.3°, the two cancelling to
1.2°. That near-perfect cancellation is what a mis-fitted plane plus a
compensating optimiser produces. A plane tilted off the true track also truncates
the projected extent, which is why its depth came out at half the cohort's.

Three independent lines of evidence agree:

1. **The angles**, above.
2. **The dye extent.** ah10's DiI runs **3654 µm** along the trajectory against a
   **2205 µm** fit — a ratio of 1.66, where every healthy subject sits at ~1.0.
3. **The dye's own principal axis.** PCA on the correctly-extracted cloud gives
   lateral 8.0° / AP 6.8°, only **7.1° from the surgical prior** — nothing like
   the stored plane. Upstream's clustering had selected the wrong signal subset
   and run its PCA on that.

Contributing factors, both now fixed here: `probe_width_scaling` was pinned at its
0.8 lower bound in 4 of 6 subjects and `brain_shrinkage_pct` at its 5.0 upper
bound in 5 of 6 (the optimiser straining against a mis-specified plane), and
upstream's `theta` bound of ±π/2 was permissive enough to let a −24° rotation pass
silently. It is ±10° here.

This is **not** a registration failure — `orientation_check.png` is correctly
`asr`-oriented and all brainreg outputs are intact — so **brainreg is never
re-run**. Steps 6–12 of the tracing pipeline are a pure function of
(plane, 5 params, volumes) and re-run in seconds.

## 2. Faithfulness of the re-projection

`project_probe` reproduces the stored `ProbeA_anatomy.htsv` **exactly** from the
stored parameters: 2256/2256 contacts for ah08, coordinates agreeing to 1e-13,
and **100.000% structure-label agreement**. Every difference reported below is
therefore attributable to a deliberate change, not to pipeline drift.

## 3. Two fixes to the machinery

**Signal extraction.** `pit.cluster_signal` auto-tunes DBSCAN's `eps` to yield a
target cluster count and then takes label 0. For a single sparse track that can
return a small noise blob: it selected a 167-point cluster **3.8 mm off the
track** for ly05, and mislabelled ly06's perfectly good track. `load_signal_df`
instead fixes `eps` and takes the **largest** cluster, which is faster and far
more stable. Use `signal_diagnostics()` to choose per-subject parameters.

**Which residual.** The two directions answer different questions and only one is
diagnostic:

| | ah08 (sparse dye, good fit) | ah10 (truncated fit) |
|---|---|---|
| `contact2signal` "is every contact near dye?" | **376 µm** (false alarm) | **51 µm** (misses it) |
| `signal2contact` "is every bit of dye explained?" | **29 µm** ✓ | **177 µm** ✓ |

`contact2signal` is high whenever dye is sparse or has faded even though the fit
is perfect, and *low* for a truncated fit sitting wholly inside a longer dye
cloud. QC flags on **`signal2contact`** only.

### Upstream issues found (worth reporting to `charlesdgburns/brainreg_probe`)

These are in the cloned repo, not in our code; each is worked around here.

1. **`get_structure_labels` indexes the atlas volume outside its `try` block.**
   A fit placing contacts beyond the volume raises `IndexError` instead of
   labelling them (every ly05 re-fit hit this), and *negative* indices are worse:
   numpy wraps them silently, mislabelling a contact with anatomy from the far
   side of the brain. `project_probe` clips for the lookup, marks those contacts
   `outside volume`, and reports `frac_outside_volume`.
2. **A `<` / `<=` mismatch on the depth filter.** `transform_2d_probe` keeps
   contacts with `probe_coords.y <= probe_depth`, but the caller in
   `get_probe_registration_df` re-filters the frame with `<`. When a depth lands
   exactly on a contact row the two disagree and the coordinate array no longer
   matches the frame it is assigned into — 1608 coordinates into a 1600-row
   frame. Optimised depths rarely land exactly on a row, but hand-annotated ones
   readily do. `project_probe` selects by the transformed frame's own index, so
   the two cannot diverge, and asserts the lengths match.
3. **`cluster_signal` can return a noise blob.** It auto-tunes `eps` for a target
   cluster count and then takes label 0; for a single sparse track that selected
   a 167-point cluster 3.8 mm off the track (ly05) and mislabelled ly06's good
   track. See "Signal extraction" above.
4. **The `theta` bound of ±π/2** is wide enough to hide a mis-fitted plane behind
   a compensating rotation, which is exactly what happened to ah10.
5. **`EXAMPLE_INPUT_DICT` is applied to every subject** (`run_probeinterface_tracking`
   has a `# can make subject-level changes here if required` comment but no
   mechanism), so per-subject gamma / eps / contact-face-axis cannot be set.

## 4. The synthetic gate

Ground truth is a known (plane, params); dye is synthesised along that known
track; the fit is then perturbed to reproduce each observed failure mode and
re-fitted. Thresholds in `QC_THRESHOLDS` are justified by this, **not** by tuning
until the real data passes. `run_synthetic_controls()`, ~2 min:

| perturbation | flagged? | expected | recovered |
|---|---|---|---|
| `truth` (negative control) | **no** | no | — |
| `ap_tilt` (+25° AP) | yes | yes | — |
| `theta` (+24°) | yes | yes | — |
| `depth` (−2000 µm) | yes | yes | — |
| `offset` (+400 µm) | yes | yes | — |
| `ah10_combo` (all three) | yes | yes | depth ±66 µm, θ ±0.32°, lat ±0.09°, AP ±0.31° |

The gate is stated in both directions: the perturbations must be flagged **and**
`truth` must not, otherwise the gate is vacuous and merely flags everything. It
also asserts that the **AP-tilt/theta cancellation signature** reproduces, since
that is the diagnostic the ah10 case leans on. **Status: PASS.**

The gate calibrates the harness's judgement of *automated* fits only. It makes no
claim over hand annotations.

## 4b. Hard flags vs advisories

`flags` decide the grade; `advisories` do not.

**Hard flags** mean the fit disagrees with the **dye**, or is geometrically
implausible: `large_theta` (a straight insertion has no reason to rotate the
probe within its own plane, so this is the "plane is wrong" alarm),
`tilt_in_wrong_plane` (AP tilt exceeding lateral, wrong for a lateral approach),
`low_signal_coverage`, `high_residual`, `params_at_bounds`, `outside_volume`.

**`trajectory_off_prior` is advisory only.** The dye is ground truth and the 10°
is nominal — stereotax misalignment genuinely varies, as the surgeon notes. A fit
that follows the dye tightly has earned its trajectory whatever the nominal was:
ah08 sits 13.3° off the prior while having the best dye agreement in the cohort
(29 µm), and failing it on that basis would be a false positive. The deviation
stays diagnostic *in combination* — paired with a large `theta` or a wrong-plane
tilt it is precisely the ah10 signature — but it never condemns a fit on its own.

This is a deliberate structural choice rather than a loosened threshold: the
alternative fix, raising `trajectory_dev_deg` until ah08 passed, would have been
tuning the gate until the data agreed with it.

## 5. Who judges what

QC metrics are built to catch an **algorithm** fitting the wrong thing and carry
that authority only over `auto` / `override` / `trajectory_constrained` fits.

Against a `manual_track` annotation they are **descriptive, not a verdict**:
`resid_*` and `signal_coverage` are measured against the Otsu-thresholded DiI
mask, and a poor threshold is precisely why one annotates by hand — scoring an
annotation against that mask scores it against the thing it was meant to replace.
The 10° prior is nominal and known to deviate with stereotax misalignment, so it
can discipline an optimiser but cannot outrank the annotator.

Accordingly:

- `grade_fit` returns `annotated` for any `manual_track` fit; it is never
  `review`.
- `annotation_report` states disagreements as information — "trajectory is 18°
  lateral vs 10° nominal", "annotated depth is 1.2× the visible dye" — for the
  annotator to accept or revise.
- The only hard blocks are **mechanical input errors** (`validate_track`):
  entry ≡ tip, or coordinates outside the volume. Typos, not anatomy verdicts.
- **Reliability is declared by the annotator** via `confidence`
  (`confident` / `uncertain`), saved with the fit and carried downstream.
  `manual_track` + `confident` is the highest grade in the scheme.

## 6. Strategy comparison (all four, every subject)

`resid` is `signal2contact` in µm; **bold** is the strategy adopted.

| subject | dye pts | baseline | constrained | fresh_plane | fresh+depth | adopted |
|---|---|---|---|---|---|---|
| ah08 | 458 | **29.1** | 50.0 | 54.9 | 520.3 | **baseline** — already correct |
| ah09 | 68663 | **101.1** | 102.5 | 102.0 | 117.6 | **baseline** (not in ephys cohort) |
| ah10 | 16350 | 177.2 | 194.8 | 220.8 | **92.4** | **fresh+depth** — corrected |
| ly05 | 167 | 3837.4 | *crash* | *crash* | *crash* | **none work → Tier 3** |
| ly06 | 25887 | **51.6** | 51.1 | 51.1 | — | **baseline** — already correct |
| ly07 | 950 | 75.1 | 59.8 | 66.1 | 95.6 | **ambiguous → annotator decides** |

**Blanket-applying the strongest correction would have made things worse.**
`fresh+depth` is right for ah10 and actively harmful for ah08, whose dye is sparse
(458 points spanning only 58% of the probe): re-deriving plane and depth from that
partial cloud truncates a fit that was already correct, sending the tip from ENTl2
to CA1 and the residual from 29 µm to 520 µm. Each subject is treated on its own
evidence.

## 7. Per-subject outcome

**ah10 — corrected.** `fresh_plane + reset_depth + trajectory_prior`:

| | before | after |
|---|---|---|
| trajectory | −1.8° lat / **25.4° AP** | **9.6° lat / −2.8° AP** |
| theta | **−24.3°** | −9.3° |
| depth | **2205 µm** | 3060 µm |
| resid (s2c) | 177 µm | **92 µm** |
| signal coverage | 0.674 | **0.857** |
| tip | CA1 | **ENTl6a** |
| ENTl fraction of bank | 0.042 | **0.427** |
| QC flags | 4 | **none** |

The trajectory now matches the surgery, the residual halves, and the recorded
bank moves from CA1/ProS/SUB into ENTl6a — all from a fit driven by the dye, with
the prior supplying only bounds.

**ah08, ah09, ly06 — untouched.** Their automated fits are the best available on
every metric. ah08's baseline `contact2signal` of 376 µm is a dye-sparsity
artefact, not a fit error (its `signal2contact` is 29 µm), and its 13.3°
trajectory deviation is an advisory, not a failure — see §4b. ah08's recorded
bank is pure ENTl (ENTl5 138 / ENTl3 133 / ENTl2 107 / ENTl1 6), spanning
superficial and deep layers.

**ly05 — requires manual annotation.** Its DiI is unusable: 167 points spanning
97 µm, whose centroid is **3772 µm from the track**. Every automated strategy
either fails or places contacts outside the volume. Use `slice_grid()` /
`annotate_track()` and `apply_manual_track()`.

**ly07 — genuinely ambiguous, for the annotator to settle.** All four strategies
give a comparable residual (60–96 µm) and a good coverage, so the dye does not
discriminate between them — yet the tip label swings between **ENTl6a, ENTm5 and
SUB** depending on strategy, and the recorded bank is dominated by SUB/ProS/ENTm
under every one. ly07 sits at the ENTl/ENTm/subiculum junction, where the
assignment is not resolvable at this method's precision. This is a case for a
hand annotation plus an explicit `confidence` flag, not for picking whichever
strategy yields the most ENTl.

> Throughout: the fit objective is agreement with the **dye**, disciplined by the
> surgical trajectory. `target_fraction` is a review flag and **never** an
> optimisation target — otherwise one simply drags every probe into ENTl because
> that is where one wishes it were. A probe that genuinely missed ENTl is a
> result, not a bug.

## 7b. Shank labelling is an exact degeneracy — and depth is often unidentifiable

Two results from following up Adam's reading of the histology ("the most anterior
shanks are firmly in deep LEC"). Both change how per-shank anatomy must be
reported.

### The mirror degeneracy

Mirroring the probe — `u_axis → −u_axis`, `offset_x → −offset_x`, `theta → −theta`
— reproduces the contact positions to **0.0 µm** and the cost to **0.000%**, on
every subject tested (ah08, ah10, ly06, ly07). The four shanks are geometrically
identical and their centred x-set equals its own mirror, so the two solutions are
literally the same point cloud.

Consequences:

- **No dye-based method can determine which physical end carries channels 0–95.**
  A plan to infer it by clustering the dye into four streaks and comparing their
  AP order is provably impossible; it was dropped rather than built. Only the
  surgical record — which way the contact face pointed — can settle it, which is
  the `contact_face_axis` parameter upstream guesses.
- **What the mirror does *not* affect is the structure at a given physical
  position** (verified: identical per-position anatomy for ly07 either way). So
  per-shank anatomy reported **by AP position** is already correct and
  orientation-independent; only the shank *number* attached to it is pending.
- `SHANK_ORDER_VERIFIED` is `False`; `shank_breakdown` therefore prints shank
  numbers with a `?`, and `save_fit` records `shank_order_verified` in the
  provenance. Set it once the surgery notes are checked.
- It also means the earlier ly07 verdict could never have been rescued by a
  labelling fix — that discrepancy is about **depth**.

### Depth is frequently not identifiable from dye

Measured on synthetics against a known 4000 µm truth, varying how far the dye
reaches (`depth_objective_comparison`, error in µm; negative = stops short):

| dye fades from | 40% dye | 60% | 80% | 100% |
|---|---|---|---|---|
| **surface** (realistic) `both` | **−2349** | −1564 | −814 | −1 |
| **surface** `signal2contact` | **−1600** | −1600 | −759 | +20 |
| tip `both` | +920 | +395 | +110 | −1 |
| tip `signal2contact` | +110 | +38 | −6 | +20 |

Dye is wiped onto the shank on the way in and commonly fades before the tip, so
`dye_from='surface'` is the realistic case — and there **depth is under-read by up
to 2.3 mm, with `signal2contact` barely helping** (−1600 vs −2349). This is an
*information limit*, not a cost-function bug: if the dye is not present below some
point, no dye-based objective can know the probe went deeper. (Where dye survives
only near the tip, `signal2contact` genuinely does help, +920 → +110 — but that is
the less realistic configuration.)

Practical consequence: `depth_over_dye` is reported for every fit, with advisories
`fit_stops_short_of_dye` (< 0.9×) and `depth_extrapolated_past_dye` (> 1.3×). Both
are advisory, never hard flags — a fit longer than its dye is expected, and only an
annotation or the surgical record can settle depth when the dye is partial.

### ah10's depth took three passes — and the middle one is the instructive failure

| pass | depth | verdict |
|---|---|---|
| automated | 3060 µm | **too shallow** — 0.81× the dye; most shanks in CA1/ProS/SUB |
| anchored to full dye range | 3774 µm | **overshot** — bank slid into ENTl2/3, tip at the pia |
| **adopted** | **3400 µm** | peak deep-LEC occupancy, tip still in ENTl5 |

Re-optimising depth under `signal2contact` did **not** move it off 3060, because
`probe_depth` also *translates* the probe (`transform_2d_probe` offsets by
`−depth/2`), so length and position are coupled and 3060 is a genuine optimum of
both objectives.

**The overshoot was my error, and it generalises.** I set depth to the dye's
*full range* (3774 µm). But diffuse dye leaves a sparse wisp past the real track
end: ah10's deepest 200 µm holds **23 points** against ~1900 in the bulk, and the
robust 1–99 percentile extent is **3223 µm**, not 3774.

| percentile cut | extent |
|---|---|
| full range | 3774 µm |
| 0.1–99.9 | 3577 µm |
| 1–99 | **3223 µm** |

`dye_extent_um` and `depth_from_signal` now both use the **robust 1–99 pct span**;
`dye_extent_full_um` and `dye_tail_ratio` are kept so the tail stays visible.

Reading depth off layer occupancy is far more discriminating than any single
scalar — deep-LEC occupancy is a clean peak, whereas "fraction in ENTl" saturates
at 1.0 across a 400 µm range and cannot see the overshoot at all:

| depth | ×dye | tip | deep ENTl5/6a | superficial 1/2/3 | non-ENTl |
|---|---|---|---|---|---|
| 3200 | 0.85 | ENTl5 | 0.56 | 0.00 | 0.44 |
| **3400** | **0.90** | **ENTl5** | **0.72** | 0.05 | 0.23 |
| 3500 | 0.93 | ENTl3 | 0.72 | 0.14 | 0.14 |
| 3600 | 0.95 | ENTl3 | 0.63 | 0.30 | 0.07 |
| 3774 | 1.00 | ENTl2 | 0.46 | 0.53 | 0.00 |

**3400 µm** was adopted (Adam, 2026-08-28). The layer progression along the track
(ENTl6a → 5 → 3 → 2, then out of the brain past ~4800 µm) is anatomically coherent
and is a useful sanity check on the whole pipeline.

## 7c. Judge a trajectory against the DYE, not against the prior

Prompted by Adam reading ly07's sagittal panel: *"ly07 fit seems to be tilted on the
AP axis."* True — **AP 10.6° vs lateral 6.8°**, where a nominal 10°-lateral insertion
should be ~10°/~0°. But the decisive question is not whether the fit matches the
nominal angle; it is whether the fit matches its own dye:

| | lateral | AP | fit vs its own dye |
|---|---|---|---|
| ly06 | 6.0 | 3.2 | **0.0°** |
| ah09 | 5.9 | 9.0 | 0.3° |
| **ly07** | 6.8 | **10.6** | **1.7°** |
| ah08 | 15.0 | 12.3 | 7.7° |
| ah10 (corrected) | 9.6 | −2.8 | 9.7° |
| ah10 (original auto) | 1.8 | 25.4 | **20.9°** |
| ly05 | 17.9 | −1.4 | **64.1°** |

**ly07's dye is itself AP-tilted** (its own PC1 is at lateral 6.2° / AP 9.1°), and the
fit tracks it to 1.7°. The tilt is real and in the histology; the fitter is doing its
job. Forcing ly07's AP to the prior would move it *away* from its dye. Whether ~9° of
AP tilt reflects stereotax misalignment or local registration distortion is a question
for the surgeon, not the optimiser.

This produced two changes:

- **New hard flag `fit_disagrees_with_dye_axis`** (`fit_vs_dye_axis_deg` > 10°,
  guarded on `dye_pc1_expl_var ≥ 0.5` so it is only applied where a principal axis
  means anything). Retrospectively it catches ah10's original fit at **20.9°** and
  ly05 at 64°, while leaving ly07 clean — it separates the two real cases with a wide
  margin, and the synthetic gate confirms it fires on a planted 25° AP tilt.
- **`ap_tilt_beyond_prior` downgraded to an advisory.** It was briefly added as a hard
  flag (because `tilt_in_wrong_plane`, `|ap| > |lateral| + 5`, let ly07's 10.6° slip
  under at 11.8°). That was wrong for exactly the reason §4b already gives: the dye is
  ground truth and the 10° is nominal. As a hard flag it condemned ly07 for being right.

> **Do not over-trust PC1.** It is a *screen*, not the target. The cloud's
> maximum-variance direction is perturbed by the 750 µm shank width and by uneven dye
> density, so a good fit can sit several degrees off it. Tested directly on ah10:
> re-aligning its trajectory exactly to PC1 (fit-vs-dye 9.7° → 0.0°) **worsened**
> everything that matters — residual 97.7 → 127.3 µm, deep-ENTl occupancy 0.72 → 0.40,
> non-ENTl 0.23 → 0.48, tip ENTl5 → ENTl3. ah10 was therefore left on its
> prior-constrained trajectory. Treat a flag here as a prompt to look, not as an
> instruction to rotate the fit onto PC1.

### `fit_vs_dye_axis_deg` measures ANGLE ONLY — never read it alone

The ah10 dye-aligned test is the clean demonstration, and it is worth keeping in mind
because the metric is new and easy to over-trust. That candidate scores a **perfect
0.0°** — its axis *is* the dye axis by construction — yet the probe visibly misses the
dye entirely (`data/figures/probe_refit/REJECTED_ah10_dyealigned.png`; note Adam's
reaction on sight: *"this ah10 dyealigned version is craaazy"*).

The reason is geometric. Both planes share the same `surface_coord`, so rotating the
trajectory about that anchor swings the far end of the probe:

```
tip displacement  =  depth x sin(angle)  =  3400 um x sin(9.7 deg)  =  572 um
                                            (measured: 574 um)
```

— comparable to the entire 750 µm shank span. **Perfect axis agreement, badly wrong
placement.** A trajectory has two degrees of freedom the metric cannot see: where the
probe sits along and across the track.

What caught it was the **residual** (`resid_signal2contact_um`, 97.7 → 127.3 µm), which
is sensitive to position. So the QC suite works as a whole, but the rule is: a clean
`fit_vs_dye_axis_deg` is necessary, not sufficient. Always read it beside the residual
and the signal overlay figure.

Rejected candidates are retained with a `REJECTED_` filename prefix so the record
survives without being mistaken for current state.

## 7d. The interactive placement tool (`probe_tool.py`)

After hand-tuning ah10's depth over three rounds and ly07's angle over four — each attempt
meaning an edited batch script, a minute's wait and a squint at a PNG — the bottleneck was
clearly the interface, not the judgement. `probe_tool.place(subject)` opens sagittal and
coronal DiI panels with sliders over the shanks, initialised at that subject's saved fit.

**Controls are entry + tip, not angles.** Two points fix trajectory and depth outright,
which sidesteps the pivot problem: rotating about the surface anchor swings the tip by
`depth × sin(angle)` (574 µm for ah10's 9.7°), so an angle slider has no innocent pivot.
`theta`, width scaling and shrinkage get their own sliders — two points cannot fix them —
and `offset_x` is pinned to 0 because it is redundant with moving the entry point.

**Panels are slab max-intensity projections** (default 400 µm) so all four shanks' dye is
visible at once; no single slice contains them all. **Atlas contours (ENTl/ENTm/SUB) are
taken from the centre slice, never the MIP** — a max-projection of label ids is
meaningless. Live readout gives trajectory, dye residual, fit-vs-dye, bank composition and
per-shank ENTl fraction.

**A placement is authoritative**: saved as `fit_method='manual_3d'`, graded `annotated`.
The harness reports its metrics but never re-optimises it against the dye or fails it; only
`validate_track` (entry ≡ tip, coords outside the volume) can block a save.

Redraw is ~280 ms with the slab cache warm, ~400 ms after a move, ~680 ms on first draw;
sliders update on release.

### Two bugs this shook out, both worth remembering

**1. `offset_x` must be absorbed, not dropped.** Every stored fit carries a non-zero one
(ly05's is −413 µm, over half the 750 µm shank span). Pinning it to 0 without compensating
opens the tool on a probe shifted sideways from the fit it claims to show.
`fit_to_track` now shifts the entry by `offset_x / VOXEL_SIZE_UM` along `u_axis`.

**2. `build_plane` was mirroring 5 of 6 subjects.** It enforces the `contact_face_axis`
sign convention, flipping `u_axis` when a plane does not match it. That is right when
deriving a plane from scratch, but wrong when re-deriving one from a stored fit whose
`u_axis` came from PCA and need not match the (admittedly guessed) contact-face direction —
it displaced contacts by ~the shank span. Only ah10 was exempt, because its plane had been
built by `build_plane` in the first place. Callers with a trusted `reference_u` now pass
`enforce_contact_face=False`. **Verified: all six subjects round-trip
saved fit → (entry, tip) → fit at 0.000000 µm.**

## 7e. `fast_structure_labels` — what made the tool possible

`pit.get_structure_labels` loops in Python with a DataFrame `.query()` per contact:
**~4400 ms** for 2016 contacts, i.e. a 4.4 s redraw. `fast_structure_labels` indexes the
volume with numpy and maps only the distinct ids present (~26 for a probe), then expands
back, so cost scales with unique structures rather than contacts.

| subject | contacts | before | after | speed-up | identical? |
|---|---|---|---|---|---|
| ah08 | 2256 | 4755 ms | 1.05 ms | 4535× | acro/name/id all 2256/2256 |
| ah09 | 2280 | 4723 ms | 0.61 ms | 7760× | all 2280/2280 |
| ah10 | 1816 | 3850 ms | 0.51 ms | 7610× | all 1816/1816 |
| ly05 | 1480 | 3250 ms | 0.35 ms | 9186× | all 1480/1480 |
| ly06 | 2120 | 4564 ms | 0.46 ms | 9975× | all 2120/2120 |
| ly07 | 2016 | 4028 ms | 0.45 ms | 9014× | all 2016/2016 |

It speeds up every batch path too, and the stored-table regression still reproduces at
100.000% on all six.

> **The trap, recorded because the benchmark hid it.** The first version indexed a *dense
> array by structure id*. Allen ids reach **614,454,277**, so it allocated three
> 614-million-entry object arrays — **14.7 GB** — and the process was OOM-killed with no
> traceback. The original micro-benchmark had built that lookup *outside* the timer, so it
> reported a genuine 0.21 ms for the indexing while hiding a catastrophic setup cost. If
> you are timing a "fast path", time its setup too. The shipped version uses a dict over
> the 1327-row table plus `np.unique`, which is both correct and small.

## 7f. Probe/brain geometry audit — is any of this in compatible units?

Prompted by Adam asking whether probe and brain geometry had been verified compatible,
specifically for **recording site geometry**. It had been assumed. Checking it:

### Verified sound

- **The probe model is right.** Every recorded channel matches `probeinterface` NP2020
  geometry **exactly** — 377–383 of 377–383, across all 10 sorted blocks and 5 mice.
- **The recorded bank really is y ≤ 705 µm**, on 100% of channels in every block, with 48
  distinct y positions at a 15 µm step. `RECORDED_BANK_MAX_UM` is correct.

### The census counts contacts, not recording sites

Recorded channels are **377–383, not 384**, and **92–96 per shank, not a uniform 96**
(dead/dropped channels). Everything filters the *geometric* contact set by `y ≤ 705`, so
the censuses describe probe geometry rather than actual recording sites. Small, but it is
exactly the distinction that was asked about.

### The fit distorts a rigid object, anisotropically

`transform_2d_probe` scales x by `width_scaling × (1 − shrink)` but y by only
`(1 − shrink)`, so the shank pitch can be squeezed independently of the probe's length.
Measuring the pitch the **dye itself** shows (`dye_shank_pitch`, gated below):

| subject | n dye | dye pitch (CI) | fitted pitch | fit − dye | width |
|---|---|---|---|---|---|
| ah08 | 458 | 216 [215,218] | 212 | −4 | 0.893 |
| **ah09** | 68663 | **242 [241,242]** | **190** | **−52** | **0.800** ◄ bound |
| ah10 | 16350 | 259 [257,317] wide | 262 | +3 | 1.050 |
| ly05 | 167 | 140 [140,360] **useless** | 250 | — | 1.000 |
| ly06 | 25887 | 202 [201,202] | 202 | 0 | 0.851 |
| ly07 | 950 | 262 [260,264] | 248 | −15 | 1.010 |

The apparent pitch genuinely varies 202–262 µm against a true 250, and most fits track
their own dye. **ah09 is the clear failure**: its dye pins the pitch at 242 µm on 68k
points with a ±1 µm CI, while the fit sits at 190 µm because `width_scaling` is pinned at
its bound. ly05's dye carries no pitch information at all (CI spans the whole search
range).

### Registration is locally anisotropic, and that was never stated

Stepping 20 voxels in sample space and reading the deformation field gives **8.06–12.95 µm
per voxel** by subject and axis — the atlas is locally stretched up to ~30% to match each
brain. Structure boundaries carry that positional uncertainty. The raw histology is
strongly anisotropic to begin with (20 µm sections × 4.033 µm in-plane), a plausible
source of the apparent width compression above.

### The pitch estimator needed three attempts — gate your estimators

`dye_shank_pitch` fits a 4-component Gaussian comb with shared spacing by maximum
likelihood. Two simpler estimators were written first and **both failed a
recover-a-known-pitch gate**:

| estimator | verdict |
|---|---|
| 1-D KMeans on `u` | scattered ±30% *within* a mouse — streaks overlap and are unevenly populated |
| periodogram argmax | **biased high on every case** (+2.8 to +17.0 µm, never low): four repeats is too short a train, so the peak is broad and drifts |
| **comb GMM** | **max error 3.3 µm** across pitches 190/250/300 at three noise levels — PASS |

The first real-data numbers came from the failed estimators and were wrong (they made ah10
look 22% compressed beyond its dye; the gated estimator says +3 µm). Gate the measurement
before you report the measurement.

## 7g. Shank number → A-P position: the convention IS in brainreg_probe

Adam's instinct that this already existed was right, and "unknowable from the dye" was too
strong. `probeinterface_tracing.fit_plane_to_signal` pins the `u_axis` sign
(lines ~321–327). For this experiment's `insertion_axis='si'`, `contact_face_axis='rl'`:

- candidate u-axes `['ap','pa']`; `only_one_axis_misaligned` is False (sum = 2) → picks
  `'ap'` = `[1,0,0]`, so **`u_axis` is forced along +i (posterior)**
- probeinterface puts shank 0 at x = 0–32 and shank 3 at x = 750–782, and
  `position = surface + (y/10)·v_axis + (x/10)·u_axis`, so larger x → more posterior

**⇒ shank 0 is ANTERIOR-most, shank 3 POSTERIOR-most.**

This config matches **ProbeB (HC)** in the upstream README figure exactly (`si` + `rl`),
which is the geometry Adam identifies ours with. Consistent too: `'rl'` = +k = leftward,
and the probes sit at centroid k ≈ 842–884 in a 1003-wide volume (midline ≈ 501), i.e. the
**left hemisphere** — so contacts face laterally, as expected for a dorsally-approached
lateral entorhinal target.

> **This is a convention, not an independent measurement.** It gives a definite answer, but
> only as good as `contact_face_axis='rl'`, which upstream's README flags as a guess
> ("not clearly documented"). If the contacts truly faced the other way, *all six flip
> together*. So it pins **relative** consistency across the cohort — which is what caught
> the discrepancies below — while absolute A/P identity still rests on the surgery notes.

### Two sign inconsistencies it caught, both self-inflicted

1. **ah10's `u_axis` is flipped** (u·i = −0.978; the other five are +0.94 to +0.99). ah10 is
   the one subject whose plane was rebuilt through `build_plane` during the
   trajectory-constrained refit — the code path where the sign flip was later found — and
   the hand placement inherited it as `reference_u`. **This one matters**: it mirrors
   ah10's per-shank table relative to the cohort. Pooled numbers are unaffected (the mirror
   is position-preserving).
2. **ah08 / ly05 / ly07 normals violate the convention** (normal·k < 0 where `'rl'` demands
   > 0), a consequence of `enforce_contact_face=False`. **No contact moves** —
   `project_2d_points_to_plane` uses only `surface_coord`, `v_axis`, `u_axis`; the normal is
   stored but never used for placement. Bookkeeping, but a stored plane that contradicts its
   own stated convention will mislead someone later.

## 7h. `theta` rotates the shanks off the reported trajectory

`theta` rotates (x, y) *within the fitting plane*, so it tilts the shank array away from
`v_axis`. Consequences, none of which were obvious:

- **The reported trajectory is the plane's axis, not the shank direction.** For ah10 at
  θ = −9.28°: `v_axis` reads lateral 7.91° / AP −2.10°, while the **actual shank direction**
  is lateral 6.00° / AP +6.98°. They differ by exactly θ. For θ ≈ 0 fits they coincide.
- **`fit_vs_dye_axis_deg` measures the wrong vector.** It compares `v_axis` to the dye, but
  the dye was laid down by the *shanks*:

| subject | theta | v_axis vs dye | **shank vs dye** |
|---|---|---|---|
| ah08 | 0.12° | 7.70° | 7.58° |
| ah09 | −3.30° | 0.34° | 3.63° |
| **ah10** | **−9.28°** | **8.87°** | **2.00°** |
| ly05 | 15.00° | 59.23° | 46.22° |

  ah10's shanks are **2.0° from its dye, not 8.9°** — it looked close to tripping the 10°
  threshold when it tracks its dye well. The metric should use the shank direction.
- **It breaks the tool's entry/tip semantics.** With θ ≠ 0 the tip marker is not where the
  shank tips land. θ should be pinned to 0 in `probe_tool` — entry+tip already define the
  shank direction — and the slider replaced by a **roll** control (rotation about the
  trajectory), which is the degree of freedom actually missing: `u_axis` is currently
  inherited silently from the stored plane with no way to adjust it.

## 8. Provenance

`save_fit` preserves the automated output once, non-destructively
(`ProbeA_anatomy_auto.htsv`, `ProbeA_fit_params_auto.json`), then writes the
corrected files with `fit_method`, `confidence`, `note`, `corrected_date`,
`manual_inputs` (entry/tip voxels, prior and tolerance, pinned params), the full
`qc` block, and `grade`. Downstream code can always tell how a row was produced
and how far to trust it.

## 9. Downstream

The recorded-bank census changes materially for ah10 (CA1/ProS/SUB → ENTl6a), so
any analysis splitting units by structure must be built on the corrected tables.
Note also that the "LEC" recording spans ENTl, ENTm, SUB/ProS, CA1 and visual
cortex across subjects — the anatomy is a multi-region mixture regardless of these
corrections.
