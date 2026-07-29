# Task-phase 4-periodicity in the population (LEC & PFC)

Companion documentation for `taskphase_periodicity.py` (identical copy in `code/`
and `mFC_data/code/`) and the notebooks `LEC_taskphase_periodicity.ipynb` /
`PFC_taskphase_periodicity.ipynb`.

---

## 1. The question

Binary state decoders do **not** generalise across tasks — state *identity* remaps
(see `CCGP_STATE_PAIRS.md`: LEC CCGP 0.426 vs null 0.500, ceiling 0.830). Given
that, is there **4-fold periodic structure** in the population activity around the
A→B→C→D→A loop — some signal that recurs at the rhythm of the four states — and
crucially, **not assumed to align with the tone at A** (phase-free)?

"4-fold periodic" means: as task phase runs once around the loop (0 → 2π over
A→B→C→D→A), the population has a component that completes **4 cycles** — one per
state. In Fourier terms over the 360-bin loop, that is power at **harmonic 4**
(and its multiples 8, 12). Contrast:

| harmonic | period | meaning |
|---|---|---|
| h=1 | one loop | the 4 states arranged around **one ring** (the classic task ring) |
| h=2 | half loop | A/C vs B/D **alternation** |
| **h=4** | **one state** | **structure that recurs each leg** — a sub-goal / progress / reward rhythm |

Phase-free throughout: we test **magnitude** (which is rotation-invariant), and
where a phase is reported it is **estimated**, never assumed to sit at reward /
state A.

---

## 2. What the data says (measured, read-only, 121 LEC sessions)

The pilot that motivated this. Trial-averaged `Neurons_norm` → `(n_neurons, 360)`
per unique task, per-neuron z-scored across phase, `rfft` along the phase axis.

### 2.1 The structure is genuinely 4-fold, not a ring
Mean per-neuron fraction of non-DC power by harmonic:

| h | 1 | 2 | 3 | **4** | 5 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|---|
| power frac | .057 | .057 | .040 | **.098** | .022 | .019 | .031 | .016 |

h=4 is the **single largest harmonic in 84% of sessions**; its multiples h=8, h=12
are elevated above their neighbours too. The "4 states on one ring" harmonic (h=1)
is weak — tied with h=2/h=3. So the structure is once-per-**state**, not a single
loop.

### 2.2 It is strongly elevated over the within-spectrum baseline
The honest existence test (no shuffle): h=4 vs the **non-multiple-of-4** harmonics
(h5,6,7,9,10,11…), which is what h=4 "should" equal if there were no special 4-fold
structure.

```
h=4 fraction 0.098   vs   non-4k baseline 0.017   →   ratio 5.9   (median 5.4)
```

### 2.3 …but roughly half of it lives at the reward boundaries
Trimming T bins off each end of every 90-bin leg (removing the reward windows) and
recomputing the ratio:

| trim (each end) | 0 | 11% | 22% | 33% |
|---|---|---|---|---|
| h4 / non-4k ratio | **5.9** | 4.3 | **2.7** | 1.6 |

So h=4 is **not** purely a reward transient (a real mid-leg component survives to
2.7× baseline at 22% trim) but it is **not** purely mid-leg either (a third of the
leg's ends carry a large share). Both a boundary component and a genuine mid-leg
periodic component are present. The per-neuron h=4 **phase** peaks mid-leg (bins
~20–30 and ~65 of 90; concentration R≈0.25), **not** at reward onset — consistent
with "doesn't align with A".

### 2.4 The four legs are barely correlated (~0.10)
The within-leg population *trajectory* in state A does not resemble B/C/D
(mean per-neuron leg-to-leg correlation ≈ 0.10). So this is **not** a clean "same
sub-goal cycle repeated four times". This is the key tension: strong h=4 power
(§2.2) but weak leg-to-leg repetition (here). Resolving what that means is
Analysis 2/3.

### 2.5 The warp bakes in *some* h=4
`Neurons_norm` is time-warped to exactly 90 bins/state with the four rewards pinned
at bins 0/90/180/270. Any feature locked to a reward is therefore perfectly
periodic **by construction**. So raw h=4 over-states the effect. The headline claim
must be **"h=4 beyond the warping/reward floor"**, not raw h=4. The floor is
measured, not argued (§5, the state-selective synthetic).

---

## 3. What is new vs what is reused

**No FFT / spectrum / harmonic code exists anywhere in the repo** — the spectral
machinery is new. Reused:

- `glm_analysis_v2.get_sessions_for_glm` — dedupe to one session per unique task
  (mandatory; sessions [0,3] and [4,7] repeat a task — see `CCGP_STATE_PAIRS.md`).
- `glm_analysis_v2.compute_task_state_arrays` — continuous task phase θ(t) for the
  raw-time control (C2).
- `cca_loop_analysis.compute_cca_null` — the circular-roll null pattern.
- `ccgp_state_pairs.build_task_state_matrices` + `LinearSVC` + role-permutation null
  — the cross-task test (Analysis 4).
- `persistent_homology_analysis` circular statistics — phase comparison.
- `apply_gridmaze_style`, gridmaze-plotter/colours conventions.

---

## 4. The analyses

### 4.1 Headline — the phase-free harmonic spectrum
Per-neuron `rfft` power at harmonics 1..16; population = mean per-neuron power
fraction per harmonic. **Existence test = the within-spectrum comparison** of §2.2
(h4/h8/h12 vs the non-4k baseline) — assumption-light, no shuffle. h1–3 shown for
context (ring / alternation).

### 4.2 The nulls — each answers a *different* question (stated explicitly)
This project has twice been misled by a null that answered a different question
than the one claimed (see the PH history in §7). So:

| question | correct null | why |
|---|---|---|
| is h=4 special vs other harmonics? | **within-spectrum** (h4 vs non-4k) + **trial-bootstrap** noise floor | assumption-light; bootstrap gives the finite-trial floor |
| do neurons share a common h=4 **phase**? | per-neuron **circular-roll** | randomises cross-neuron phase, preserves each neuron's power |
| is h=4 beyond the warp floor? | the **state-selective synthetic** through the same warp (§5) | measures what the warp alone produces |

> **Do not** use a circular-roll or FFT-phase-randomisation null for the
> *existence* of per-neuron h=4 power: both **preserve each neuron's power
> spectrum exactly**, so real data can never beat them and any "significance" is
> vacuous. This is the identical trap that made the PH per-neuron circular-shift
> null report rings that weren't there. Circular-roll is used **only** for the
> cross-neuron phase-coherence question, where it is correct.

### 4.3 Phase & state-invariance
- **Phase:** per-neuron h=4 phase (`np.angle` of the complex h4 coefficient) →
  distribution around the 90-bin leg (resultant R, mean, at-reward vs mid-leg).
- **State-invariance:** (a) leg-to-leg trajectory correlation (§2.4); (b) whether
  each neuron's within-leg phase is **consistent across the four legs** (per-leg h1
  phase agreement). Low ⇒ the h=4 is not a single state-invariant waveform.

### 4.4 Confound controls
- **C1 boundary** — trim `trim_reward_bins` off each leg end; does mid-leg h=4
  survive? (§2.3: partly, to ~2.7× at 22% trim.)
- **C2 warp** — an independent **raw-time** spectrum: least-squares fit
  `[cos(kθ), sin(kθ)]` to `Neuron_raw` with θ(t) the continuous task phase from
  `compute_task_state_arrays`, no resampling to fixed bins. h=4 present here ⇒ not a
  warping artifact.
- **C3 ramp vs cycle** — is the sub-goal structure a smooth **closed cycle** or a
  goal-progress **ramp** (open arc)? A ramp/sawtooth has power decaying h4>h8>h12; a
  pure cycle has an isolated h4. Plus within-leg PCA: does the mean leg trajectory
  form a loop or an arc? This is the honest reframe of "control for goal progress":
  the state-invariant component **is** the h=4 — the question is its **geometry**,
  not whether to subtract it.

### 4.5 Cross-task generalisation — the sequel to CCGP
State *identity* remaps across tasks. Does the within-leg periodic structure
**generalise**? Reuse the CCGP pipeline but decode **progress-bin, pooled over the
four states**: per-(trial, state, progress-bin) vectors, train a progress decoder on
N−1 tasks, test on the held-out task, role-permutation null. If **progress-CCGP >
null while state-CCGP ≈ null** (already shown) ⇒ *"which-goal remaps, but
progress-through-goal is abstract."* Phase-free = the within-leg peak location is not
assumed; the reward is the natural per-leg anchor (this is not an A-alignment
assumption). Caveat: within a leg, progress ≈ elapsed time, so this is "temporal
within-leg structure generalises"; the time/progress split lives in
`time_vs_progress_dissociation.py` and is not re-litigated here.

### 4.6 Per-neuron periodic cells (grid-cell style)
Per neuron: is h=4 significant vs the trial-bootstrap floor **and** vs its own non-4k
harmonics? Fraction of "4-periodic" cells; their phase rose; LEC vs PFC.

---

## 5. Validation — synthetics first (mandatory)

`make_synthetic_periodicity(kind, ...)` emits a `data_dic`-shaped object through the
**real** pipeline. The gate is that the module **discriminates** these — each maps to
a different reading of the real data, so if the pipeline can't tell them apart it
can't interpret the real result:

| synthetic | expected signature |
|---|---|
| pure h4 sinusoid, phases tiled | clean h4 peak, **no** h8/h12; C3 → CYCLE |
| goal-progress ramp (linear each leg) | h4>h8>h12 decay; C3 → RAMP/arc |
| reward transient at φ=0 only | h4,8,12; phase **at boundary**; **dies under C1 trim** |
| state-selective bumps, no within-leg structure | h4 **not** elevated vs non-4k; leg-corr ≈ 0 — **this is the warp floor** |
| pure noise | flat spectrum |

Real h=4 must exceed the **state-selective** synthetic (the warp floor), and its
trim/geometry behaviour picks it out among cycle / ramp / boundary. See
`synthetic-controls-catch-design-errors` — this practice has caught five plausible,
wrong design decisions across this project so far.

---

## 6. How to read the result

| pattern | reading |
|---|---|
| h4 ≫ non-4k, survives trim, present in raw-time (C2), C3 → cycle | **genuine 4-fold sub-goal cycle** |
| h4 ≫ non-4k but dies under trim, phase at φ≈0 | **reward-boundary transient**, not a sub-goal cycle |
| h4>h8>h12 decay, C3 → arc | **goal-progress ramp** (open), not a closed cycle |
| h4 present, leg-corr ≈ 0, per-leg phases inconsistent | 4-fold power **without** a state-invariant waveform — likely boundary + warp |
| h4 no more than the state-selective synthetic | **warp floor only** — no real periodicity |
| progress-CCGP > null while state-CCGP ≈ null | **progress abstract, identity not** — the key dissociation |

---

## 7. Why this much care about nulls

The persistent-homology work in this repo was misled for weeks by a per-neuron
circular-shift null that *looked* strongly significant but answered the wrong
question: shifting each neuron independently decorrelates the population, so real
low-dimensional data beat it almost always, and "significance" meant "the population
is low-dimensional", not "there is a ring". §4.2 applies that lesson directly:
circular-roll / phase-randomisation preserve per-neuron power and so cannot test
per-neuron h=4 — they are used only for phase coherence, where they are correct.

## 8. Ring structure, spatial confounds, and grid-like generalisation (v2)

Three follow-ups, all in the same module.

### 8.1 Is there anything ring-structured? (`ring_analysis`)
A "whole-trial ring" = the **1st harmonic** (one cycle per ABCD loop). Two readouts: **h1 vs its
neighbours** (h2,h3 — a real ring has h1 ≫ h2,h3) and the **winding number** of the population
trajectory in its top-2 PC plane (1 = one loop, 4 = four-fold). Synthetic `ring` → h1/nbr 15, winding
1; `cycle` → h1/nbr 1, winding 4. **Real LEC: h1/nbr ≈ 1.0, no clean winding-1 ⇒ no ring** — the
structure is 4-fold, matching the weak h1 spectrum and the persistent-homology no-ring result.

### 8.2 How much of the cross-task progress result is spatial?
The h4 spectrum is largely the state-invariant *progress* signal by construction, and within a task
progress is confounded with location (the GLM/CPD separates those). The cross-task **generalisation**
is the confound-resistant claim, controlled three ways:

- **`run_progress_place_split`** — pairwise cross-task progress, split by reward-tower overlap. Real
  LEC: progress generalises **equally at zero tower overlap** (≈0.68) as with shared towers (≈0.66) ⇒
  shared *reward-tower* place is not the driver.
- **`run_cross_task_variables`** — leave-one-task-out decoding of **time-progress**, **distance-progress**,
  **location** (per-bin node), and **state identity**, each vs a role-permutation null, on the raw-time
  `prepare_session_data` outputs. Real LEC (preview): time-progress ≈ 0.70, distance-progress ≈ 0.56,
  **location ≈ 0.48** (chance 0.065 — location *does* partly generalise, because maze corridors are
  shared across tasks), state ≈ chance. **Honest ceiling:** progress is not cleanly separable from
  shared-*spatial* structure; it generalises more strongly than location and far more than identity,
  but a shared-place component is real.
- Time vs distance progress are collinear within a leg — the definitive split is in
  `time_vs_progress_dissociation.py`; here they are reported side by side, not adjudicated.

### 8.3 How would a grid code for task space generalise? (`grid_generalization`)
A grid cell keeps **coherent phase relationships** across environments: the lattice re-anchors (a
global phase shift) but cell-to-cell offsets are preserved — unlike place/state, which remap
independently. Test on the per-cell h4 phase across task pairs: **coherence** =
`|Σ w exp(i(φ_c^B − φ_c^A))| / Σ w`. High coherence with a non-zero global shift = grid-like
re-anchoring; coherence at the cell-shuffle null = remapping. Synthetics: `grid_coherent` → coherence
1.0 vs null 0.14, shift ≠ 0; `grid_remap` → 0.11 ≈ null. **Real LEC (preview): coherence 0.71–0.81 vs
null 0.13, consistent global shift, low no-shift consistency ⇒ coherent re-anchoring = grid-like.** The
population **cross-task progress** decoding is the population-level version of the same statement.
Caveat: the coherence statistic needs phase *diversity* across cells (clustered phases give real ≈
null, uninformative rather than misleading); real LEC has diverse mid-leg phases, so it is in the
informative regime.

## 9. Known limitations
- **The warp bakes in some h=4** (§2.5). The headline is "beyond the warp floor",
  measured via the state-selective synthetic — never raw h=4.
- **Nulls that preserve the power spectrum can't test per-neuron h=4** — used only for
  phase coherence (§4.2).
- **Cross-task progress generalisation may ride on elapsed time** (§4.5) — scoped as
  temporal within-leg structure, not a time/progress claim.
- Fourier magnitude is a *linear* summary; a nonlinearly-phase-locked code could carry
  4-fold structure the spectrum under-reads. The per-neuron and geometry analyses
  partly cover this.
