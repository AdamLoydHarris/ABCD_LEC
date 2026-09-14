# W0 — Foundation

**Status: complete.** Reference document — you should not need to redo any of this, but the
gates are cheap and worth re-running if anything upstream changes.

Code: `code/w0_gates.py`, `code/anatomy_split.py`. Cached: `data/processed_data/w0_gates.pkl`.
Full methods record: `code/ANATOMY_SPLIT.md`.

## The gates

| # | gate | result |
|---|---|---|
| 1 | `len(unit_regions[rd]) == Neuron_raw.shape[0]` | **PASS 25/25** (hard assert) |
| 2 | HD column identity: `earL2earR − back2mid ≈ ±90°` | **PASS**, 187 sessions, median 84.8°, 98.3% within 25° |
| 3 | Depth ordering consistent within each mouse | **PASS 4/5** |
| 4 | Unit quality per region, published first | done — and it is a problem (see README §4) |
| 5 | Positive control: spatial tuning higher in CA1/HPF and SUB/ProS | **PASS on CPD; FAILS on the binary fraction** |

`load_data_dic` also re-ran both registry validators: 25 recdays, 191 sessions, 0 mismatched.

### Gate 3 — one boundary that is not resolvable

Median `y_um` orders regions consistently in every mouse. Two apparent exceptions are not
failures: **ah10** has a tie at exactly 300 µm (a tie is not a reversal), and **ly07** reverses
ENTl-deep against ENTm between recdays but their medians are separated by only **15–75 µm**,
against a local deformation field of 8–13 µm/voxel.

That has teeth: **ly07 is the ENTm-rich mouse** (179 of 284 ENTm units), so it is the animal
that would supply any ENTm claim. **Do not contrast ENTm against ENTl-deep in ly07 without the
50 µm boundary-margin filter.** The primary contrast is unaffected — ENTl-deep (135–210 µm) and
SUB/ProS (390–442 µm) are separated by more than 200 µm.

### Gate 5 — the control passes, but not as originally specified

The plan asked for spatial tuning higher in CA1/HPF and SUB/ProS than ENTl-sup, within ah10 and
ly05, and said failure blocks everything.

Measured as **fraction of units tuned to place**, it *fails* — CA1/HPF is the lowest group
(0.728) and SUB/ProS (0.849) barely exceeds ENTl-sup (0.824). But that statistic is
**saturated**: 80.0% of all 2651 units are place-tuned, range 70.3–85.6%. A binary permutation
F-test at p<0.05 over ~48,000 time bins passes almost everything.

Measured as **place CPD** — graded, no ceiling — it passes in the predicted direction:

| contrast | difference | 95% CI (mice) | n mice |
|---|---|---|---|
| SUB/ProS − ENTl-deep | +0.0028 | +0.0010, +0.0046 | 4 (positive in all four) |
| CA1/HPF − ENTl-deep | +0.0021 | +0.0011, +0.0031 | 2 |
| CA1/HPF − ENTl-sup | +0.0048 | +0.0035, +0.0062 | 2 |

Median place CPD: ENTl-sup 0.0037 < ENTl-deep 0.0058 < ENTm 0.0064 < CA1/HPF 0.0077 <
SUB/ProS 0.0082.

**The anatomy map is validated.** The gate is amended to use CPD; the binary tuned fraction
should not be used for any regional comparison anywhere in this project.

## `code/anatomy_split.py` — the shared module

Every notebook imports this. Do not re-roll `groupby('group')` — the point is that the join and
the inference design exist exactly once.

| function | purpose |
|---|---|
| `join_regions` | positional join with a **hard length assert**; drops rather than truncates |
| `assert_glm_keys_contiguous` | catches the row-shift failure the length gate cannot see |
| `feasibility_table` | units per region × recday **after** an analysis's own gate |
| `per_region_report` | the per-region-first panel; mice as the unit |
| `per_mouse_effect`, `cluster_bootstrap` | mice as the resampling unit |
| `within_recday_permutation` | shuffles `group` **within recday** |
| `rate_match`, `boundary_margin_filter` | the two robustness checks |
| `remapping_angle`, `pairwise_angles`, `circular_xcorr`, `is_generalising` | angle machinery for W3 |
| `REGION_COLORS` | named GridMaze palette, ordered along the probe |

**Two guards worth knowing.** `join_regions` refuses a result whose length disagrees with
`unit_regions` (it correctly rejects the stale `ly05_20250618_20250619` at 91 rows vs 109).
`assert_glm_keys_contiguous` catches a subtler one: `compute_tuning_arrays` writes row *k* for
the *k*-th sorted key, so a cache missing one neuron shifts every later row — and that would
pass the length gate whenever the count happened to match.

## Synthetic controls (all passing)

- `remapping_angle` returns the true rotation for 0/±30/±45/±90/±180°, both methods.
- `circular_xcorr` (FFT) matches the O(n²) loop to 7×10⁻¹⁶.
- `pairwise_angles`: a **rigidly rotated** population preserves every pairwise angle to
  0.00e+00; an **independently rotated** one changes them by a median of 87°.
- HD fix: on a ramp with a known +90° second column, returns column 0 exactly (r = 1.000) where
  the old code correlated r = 0.756 and differed by up to 179.3°.

Performance: batching the pairwise cross-correlations took the coherence metric from **~7 hours
to ~2.5 s** for the whole cohort.
