---
name: synthetic-gate
description: "Writing synthetic controls for a new or changed analysis in this repo: planted cells that enter through the real data path (a data_dic-shaped dict), positive AND negative checks, equivalence to the previous version, and construction-time guards. Use before trusting any new regressor, selection rule, decoder, null or solver -- this practice has caught six plausible-looking errors."
---

# Synthetic gate

Every analysis in this repo is gated on synthetics run through the real pipeline before its
numbers are believed. The point is feeding the whole analysis a
dataset whose answer is known by construction.

## The door rule
- Build a `data_dic`-shaped dict (`Neuron_raw`, `Locs_raw`, `Trial_times`, `Task`, `num_trials`
  per session) and hand it to the same entry point the real data uses
  (`run_cross_validated_regression_v5`, `run_glm_analysis`, ...). Never pass a ready-made design
  matrix to an inner function -- that skips exactly the code that fails.
- Reuse the existing generators: `elasticnet_v5_synthetics.make_recday` / `make_behaviour`,
  `glm_v3_synthetics` control helpers. Add new cell types to them rather than new generators.

## What every suite must contain
1. **Equivalence**: the new version with legacy flags reproduces the previous one to
   `max|diff| == 0` over every exported array (ignore only stamps like `engine`, `elapsed_s`).
   Pin every flag whose default moved.
2. **Positive**: a planted cell with a known property is recovered (right lag, right regressor,
   right class) with the real selection rule.
3. **Negative, stated in both directions**: the confound cell is rejected -- AND is accepted once
   the gate is widened. A rejection that never accepts anything is vacuous.
4. **Noise**: pure Poisson noise scores at chance on the headline statistic; report the
   false-positive rate of any binary criterion and bound it from a measurement, not a guess.
5. **Guards**: invalid configurations raise at construction; a silent neuron yields NaN, not an
   exception hours into the run.
6. **Sign / direction** when a construction can be mirrored (past/future, prospective/
   retrospective): a cell built to fire AFTER leaving a location must read as past in every
   implementation. A flipped sign is invisible in every downstream figure.

## Print, do not just assert
`check(name, passed, detail)` lines with the measured value, so the log says *how* close a pass
was. The suite exits 1 on any failure and lists them.

## What this has caught (examples to reuse as controls)
- Head-direction regressor flattened from (T, 2) -- every HD result before the fix was noise.
- `compute_task_state_arrays` rotated state labels per trial (72.9 % of legs wrong); the
  collinearity check read GOOD *because* of it.
- Peak-per-state tuning test selects the shortest leg (~100 % FPR at a 3x duration ratio).
- Shuffle null for persistent homology over-reports rings; needs a covariance-matched null.
- `attach_pokes` not called -> all-zero poke columns, CPD exactly 0, one lost rank each.
- A positional per-session list (`norm_neurons_dic`) indexed as if by session number
  (9/25 recdays misaligned).

## Where it goes
`code/<analysis>_synthetics.py` with a header table `| # | control | what a failure would mean |`,
a `--quick` flag, mirrored byte-identically to `mFC_data/code/`, and the outcomes recorded in the
analysis's `*.md`.
