# Registration QC (`registration_qc.py`, `landmark_tool.py`, `LEC_registration_QC.ipynb`)

Does the brainreg → Allen CCF registration support the ENTl **superficial (1/2/3) vs deep
(5/6a/6b)** labels that the analyses rest on?  Everything in `probe_refit` judges the *probe fit*
against the DiI; nothing before this validated the *registration* beyond `orientation_check.png`
(axis order).  Adam had never seen atlas boundaries overlaid on his own autofluorescence.

Run everything in the **`histology` conda env** from `code/histology_refit/`
(import takes ~2 min through the `probe_refit` → brainrender chain; keep one kernel).

```python
import registration_qc as rq, landmark_tool as lt
gate = rq.run_synthetic_controls('ah08'); assert gate.attrs['passed']
rq.qc_table()                      # one row per mouse -> brainreg/registration_qc.csv
```

**Status 2026-09-11: complete for checks A–E and G on all six mice (gate 10/10). Check F (landmark
TRE) is built and waits for Adam's clicks.**  Headline: in ah08, ah10 and ly06 the registered pia
and grey/white edges sit within 10–20 µm (60 µm for ly06's grey/white) of the visible ones at the
recorded bank and 1–6 % of ENTl contacts would flip superficial/deep; ly05 agrees at the bank but
the warp is straining around damaged tissue; ly07's deep ENTl boundary is ~175 µm too deep at the
ENTl/SUB junction.  Nowhere is the registration the dominant uncertainty for the layer split —
probe depth is.

## 1. What the registration can and cannot do — the framing

The freeform B-spline has control points every **400 µm** (`control_point_file.nii`, 26×21×29 at
0.4 mm).  A warp that smooth cannot represent laminar structure.  **Layer labels are therefore the
atlas's proportional layering carried by a smooth warp between the two edges the registration can
actually see: the pia and the grey/white boundary.**  The decisive local test is whether those two
edges land where the tissue shows them (check E), and how many recorded contacts sit within that
uncertainty of the superficial/deep boundary (check D).  Global checks (A, B, C) are necessary but
not sufficient.

A perfect registration cannot rescue a superficial/deep split if the probe *depth* is unconstrained.
ah08 — the "pure ENTl, superficial + deep" mouse — has its depth extrapolated 1.76× past its dye
(`probe_fit_qc.csv`).  Registration error and depth error add; this document reports only the first.

## 2. Conventions (all verified from the files on disk)

| item | fact |
|---|---|
| sample space | `asr`: i ant→post, j sup→inf, k right→left, 10 µm; ah08 (884, 696, 1005). Coronal page `vol[i]` → (j, k); `imshow(origin='upper')` puts superior at top and the **right** hemisphere on the image-left |
| affine | niftyreg ref = sample, flo = atlas ⇒ `affine_matrix.txt` maps **sample mm → atlas mm**, rows (AP, DV, LR). All nii have `affine = diag(0.01)`, zero origin: atlas voxel = mm / 0.01 |
| deformation field | sample grid, atlas mm, forward, affine included; negative outside the brain |
| hemispheres | value **1 = left = high k = implant side** (`brainglobe_atlasapi/core.py:30`; brainreg `main.py` passes it explicitly). The *defaults* in `brainreg/core/utils/volume.py` are reversed — never use them |
| `boundaries.tiff` | every inter-label edge, both sides, int8 {0,1}; not the outline. Brain mask = `registered_atlas > 0` |
| atlas copies | `niftyreg/annotations.nii` = native 10 µm annotation, identical across mice (md5 `a67a123d…` for ah08 and ly06); `niftyreg/brain_filtered.nii` is **high-pass filtered** (`img/(gaussian(img,5)+1)`) — edges only, never intensities. Unfiltered 25 µm atlas: `~/.brainglobe/allen_mouse_25um_v1.2/` |
| `downsampled_standard.tiff` | made with a *separately estimated inverse* B-spline, nearest-neighbour; atlas-space checks test that warp, not the forward one that placed the labels |
| autofluorescence contrast | fibre tracts are **darker** than grey (ah08 page 670: fibre 217, alv/ec near deep ENTl 556, ENTl grey 980–1130, agarose 43). Laminar gradient ENTl1 998 < ENTl2 1047 < ENTl3 1130 ≈ ENTl5 1129 > ENTl6 979 |
| structure ids | reach 614,454,277 — dicts and `np.isin` on small id arrays only |

## 3. Acquisition facts that bear on the registration

**All six brains are anteriorly truncated.**  They were cut in one BakingTray block and the
acquisition was aborted by hand after section 221 of 289 planned.  37–61 % of peak tissue area is
still present on the last plane (ly07 15 %).  ah08, ah10 and ly06 also start mid-cerebellum.
brainreg ran with `brain_geometry full`.  The affine AP scale is 0.97–1.00 for every mouse, so the
atlas **overhangs** the cut face rather than being squashed to fit (a squash would need ≈ 1.4).

**Mount angle.**  Polar decomposition of the affines (yaw about DV, pitch about LR, roll about AP):

| mouse | yaw | pitch | roll | total | scale AP | DV | LR |
|---|---|---|---|---|---|---|---|
| ah08 | −6.7° | +6.0° | −6.9° | 11.1° | 0.983 | 1.134 | 1.049 |
| ah09 | +2.1° | +0.5° | +0.9° | 2.3° | 0.968 | 1.090 | 1.057 |
| ah10 | −0.2° | +1.8° | +2.1° | 2.8° | 1.003 | 1.150 | 1.048 |
| ly05 | +1.5° | **+11.7°** | −1.0° | 11.8° | 0.999 | 1.103 | 1.043 |
| ly06 | −3.9° | −4.2° | +2.9° | 6.4° | 0.977 | 1.140 | 1.044 |
| ly07 | +1.1° | +3.2° | +0.7° | 3.5° | 0.987 | 1.132 | 1.045 |

The DV scale of 1.08–1.15 is systematic across the cohort: the samples are 8–15 % shorter
dorsoventrally than the CCF (preparation and/or atlas bias, not a per-brain failure).

**Yaw is real, visible per section, and absorbed.**  A yaw makes coronal sections oblique in L/R:
ah08's pages run from 21 % more right-hemisphere tissue posteriorly to 32 % more left anteriorly,
and the warped labels track the image ratio page-for-page.  Integrated region volumes cancel it —
ah08 and ly06 have the largest yaw and no L/R asymmetry in LEC/HPF volumes.  After registration
the atlas outline midline sits on the tissue outline midline to a **median 0–5 µm** per page in
every mouse (residual yaw within ±0.2° for four mice; ah08/ly07 read −1.7°/+1.2° only because single
torn pages fell inside an OLS fit window — the module uses Theil–Sen).  Section obliquity costs no
resolution: 20 µm z × sin 7° ≈ 2 µm.

**ly05 is damaged, not mis-registered.**  Its left CA1/SUB/ProS are 12–24 % smaller than right and
left V1 35–45 % smaller.  Per page, both the atlas-label footprint and an image tissue mask show the
deficit in the **same band, sample AP 4.1–5.9 mm** (≈ bregma −3.8…−5.6: posterior cortex /
hippocampus / entorhinal level; image up to −0.31, labels −0.14) and ≈ 0 elsewhere; a yaw would give
a linear trend, not a band.  `ProbeA_sections.png` shows torn tissue at the dorsal-left cortex (the
implant column) and a detached fragment at the right ventro-lateral surface, plus vertical
tile-intensity bands from stitching.  The registration follows the missing tissue.  Whether the
compressed band reaches the recorded bank is what the LEC-zoom overlay and check E settle (§5).

**Two more acquisition artefacts the checks had to be made robust to.**  (i) Page-to-page
in-brain brightness varies by 7–10 % (up to 24 % in ah09) — not as a clean odd/even alternation of
the two optical planes, but with a period of a few pages — and the sagittal panels of ah09 and ly07
show strong per-page stripes that are largely a texture/contrast difference between planes.  Check
E divides each page by its in-brain median before sampling (`page_gain_correct`); the grey/white
detections were unchanged by it, so the ly07 result below is not a stripe artefact.  (ii) Six brains
shared one block, so fragments of the neighbours enter the field of view at the edges; the
page-centroid statistics use the largest connected component only and carry no flag (they are
dominated by component switching at the block faces and by loose flaps, not slippage).

## 4. The checks

| check | question | function | flag |
|---|---|---|---|
| **A** | do atlas boundaries sit on the tissue? | `plot_sample_overlays`, `plot_lec_zoom`, `plot_atlas_space_checker` | human verdict (`record_verdict`) |
| **B** | affine sane; overhang not squash; L/R asymmetry; residual yaw/pitch; section continuity; tile artefact; region volume ratios | `global_checks` | scale ∉ [0.85, 1.15]; rotation > 15°; residual angle > 1°; |asym| > 0.15 on LEC/HPF |
| **C** | deformation field plausible near the track | `jacobian_check` | det J outside [0.5, 2] on > 1 % of voxels, or any fold |
| **D** | contacts within 50/100/150 µm of the sup/deep boundary **and of the ENTl/ENTm (LEC/MEC) border**; stored Allen coords consistent | `contact_boundary_distances`, `lec_mec_by_shank`, `label_self_consistency` | advisory if > 50 % of ENTl contacts within 100 µm of sup/deep, or > 25 % of the bank within 100 µm of the LEC/MEC border |
| **E** | visible pia / grey-white edges vs atlas edges along the local surface normal; **sup/deep flip fraction** | `laminar_landmark_check` | |median offset| > 100 µm |
| **G** | along-track label coherence | `laminar_coherence` | descriptive |
| **F** | landmark target-registration error | `landmark_tool` | reported, with the click noise floor |

### How E works
For every recorded-bank contact labelled ENTl: nearest brain-surface voxel and the inward normal
(gradient of a σ = 5 vox smoothed brain mask); a 5 µm-step line from 300 µm outside to 1500 µm
inside; green intensity (linear interpolation) and labels (nearest) along it.

- **Visible pia** = the **half-maximum** crossing between the agarose level and the tissue plateau
  (unbiased for a blurred step; the first threshold crossing sits at the *foot* of the edge and read
  a spurious +50 µm everywhere — §5).
- **Visible grey/white** = steepest descent into the **deepest significant dip within ±300 µm of the
  atlas's ENTl exit**.  The fibre band under ENTl is a 100–200 µm dark *dip*, not a step to a new
  plateau.  Two blind rules failed on real profiles: the *strongest* bright→dark step picked deeper
  hippocampal edges on ah10/ly05/ly07 (fictitious 0.6 thickness ratios), and the *first* significant
  dip picked the genuine ENTl L5→L6 darkening on ah08 (L6 is ~13 % darker than L5 here).  The ±300 µm
  window is an honest prior: errors up to 300 µm are measured without bias, larger ones return no
  dip and show as a collapse of `n_with_wm`.  Only lines whose atlas exit *is* a fibre tract count
  (`frac_exit_fibre`); where ENTl abuts piriform or hippocampal grey (ah09: 60 % of lines) there is
  no grey/white edge to compare.
- **Atlas edges**: pia = first non-root label; grey/white = where the labels leave ENTl; sup/deep =
  first ENTl3→ENTl5 transition; L1/L2 = first ENTl1→ENTl2.
- `offset = atlas − visible` (positive: atlas boundary deeper).  Reported raw and
  **baseline-corrected** by the atlas's own offsets from the modality control (pia +24 µm, WM +5 µm).
  `thickness_ratio = atlas / visible`.  **Flip fraction** = contacts whose fractional depth, re-read
  against the (baseline-shifted) visible edges, lands on the other side of the atlas's sup/deep
  fraction — the direct answer to "can I trust superficial vs deep".

### The tissue mask
Otsu on **log** intensity, opening, hole-fill, largest component.  Otsu on linear intensity lands
inside the tissue range and is confounded by L/R illumination — it read a physically impossible
−0.7 asymmetry for ah09 — so the mask separates agarose (~45) from any tissue (~90–130) instead,
and `lr_profile` reports the L/R intensity ratio beside the area ratio as a check.

## 5. Synthetic gate

`run_synthetic_controls('ah08')`, stated in both directions:

| gate | plant | must |
|---|---|---|
| D distances | two half-spaces, contacts at 0.5–30 vox | recover to ≤ 0.5 vox |
| C Jacobian | pure-affine field; a folded block | det J = det A to 1e-3; fold → negative det |
| E1 detector | blurred noisy steps at known positions; a flat trace | steps within 15 µm; no false edge |
| E2 planted shift | labels read at ±100 µm along the normal on real ah08 | offsets move by ∓100 ± 20 µm; 0 reproduces baseline |
| E3 truth negative | synthetic green built *from* the registered labels | |offsets| < 20/40 µm, **not flagged** |
| E4 atlas modality | detector on the unfiltered atlas vs its own annotation | |offsets| ≤ 40 µm |
| self-consistency | Allen axes permuted / +100 µm DV | disagreement > 30 %; unperturbed ≤ floor + 2 % |
| equivalence | slab labelling vs `probe_refit.project_probe` | identical coordinates and labels |

### What the gate caught on its first run (2026-09-11)

Eight of ten gates passed first time — including exact equivalence with `project_probe`
(2256 contacts, 0.0 µm, 100 % labels) and the planted ±100 µm shifts recovered as exactly ±100 µm.
The two failures were both **detector calibration**, and both would have gone into the results as
registration findings had the gate not existed:

1. **E1 — 38 false edges on a flat noisy trace.**  Edge prominence was scaled to the profile's
   5–95 % range, which on a flat trace *is* the noise, so every wiggle passed.  Fixed by requiring
   the step amplitude (mean 20–80 µm after minus before) to exceed 5× the smoothed-trace noise,
   estimated robustly from the first difference of the raw profile.
2. **E3 — the truth negative showed the same +50 µm pia offset as the real mouse and the atlas.**
   The visible pia had been defined as the first crossing of the low tissue-vs-agarose threshold,
   which sits at the *foot* of a blurred edge, ~1.5 blur-σ outside the true edge.  Sample, atlas and
   a label-derived synthetic (which has no pial rim by construction) all read +50 µm — the signature
   of an estimator bias, not anatomy.  Fixed by defining the visible pia as the **half-maximum**
   crossing between the agarose level and the tissue plateau, which is unbiased for a symmetric
   blurred step.  The atlas modality control still measures whatever residual offset the annotation
   itself carries, and every sample offset is reported raw and baseline-corrected.

The E3 truth negative is evaluated *raw* (no baseline): a synthetic built from the labels has the
annotated surface as its visible edge, so its offsets must be ≈ 0 by construction.

### Gate result after the fixes (2026-09-11) — **PASS, 10/10**

| gate | measured |
|---|---|
| D distances | max error 0.0 vox |
| C Jacobian | affine det error 3e-6; planted fold detected |
| E1 detector | steps recovered to 0 µm; **0** false edges on a flat trace |
| E2 planted shift | +100 → +100.0 / +100.0 µm (pia / WM); −100 → −100.0 / −100.0; zero reproduces baseline |
| E3 truth negative | pia +6 µm, WM +15 µm, thickness ratio 1.01, not flagged |
| E4 atlas modality | 290/300 lines with a grey/white edge; **annotation's own offsets: pia +24 µm (IQR 15), WM +5 µm (IQR 30)**, thickness ratio 0.97 — this is the baseline subtracted from every mouse |
| self-consistency | unperturbed 4.7 % disagreement vs 15.9 % boundary floor; axes permuted 100 %; +100 µm DV 33 % |
| equivalence | 2256 contacts, coordinates 0.0 µm, labels 100 % identical to `project_probe` |

## 6. Results per mouse

All numbers from `registration_qc.csv` (run 2026-09-11; figures in `data/figures/registration_qc/`).

### 6a. Global (checks B, C, D, G)

| | ah08 | ah09 | ah10 | ly05 | ly06 | ly07 |
|---|---|---|---|---|---|---|
| fit | manual_3d | auto | manual_3d | manual_3d | auto | manual_3d |
| affine scale AP / DV / LR | 0.98 / 1.13 / 1.05 | 0.97 / 1.09 / 1.06 | 1.00 / 1.15 / 1.05 | 1.00 / 1.10 / 1.04 | 0.98 / 1.14 / 1.04 | 0.99 / 1.13 / 1.05 |
| mount yaw / pitch / roll | −6.7 / +6.0 / −6.9° | +2.1 / +0.5 / +0.9° | −0.2 / +1.8 / +2.1° | +1.5 / **+11.7** / −1.0° | −3.9 / −4.2 / +2.9° | +1.1 / +3.2 / +0.7° |
| **residual yaw** (Theil–Sen) | +0.13° | −0.17° | 0.00° | −0.06° | 0.00° | −0.20° |
| residual pitch (advisory) | +0.21° | +1.02° | +0.72° | 0.00° | +0.67° | +0.48° |
| atlas − tissue midline, median (IQR) | +5 (11) µm | −3 (11) | 0 (5) | 0 (11) | −5 (5) | 0 (11) |
| image vs label L/R asymmetry, r | 0.89 | 0.80 | 0.70 | 0.72 | 0.99 | 0.85 |
| mask threshold sensitivity of L/R asym | 0.002 | 0.001 | 0.001 | 0.004 | 0.001 | 0.001 |
| missing anterior / posterior (mm) | 3.3 / 1.4 | 3.3 / 1.7 | 2.8 / 1.8 | **4.3** / 0.5 | 2.9 / 1.9 | 2.9 / 1.9 |
| tissue still present on last plane (rel. peak) | 0.57 | 0.33 | 0.65 | 0.25 | 0.65 | 0.54 |
| max L/R volume asym, LEC/HPF groups | 0.05 (ProS) | 0.06 (ENTm) | 0.10 (SUB) | **0.24 (ProS)** | 0.04 (ProS) | 0.05 (ENTm) |
| volume ratio sample/atlas: ENTl · CA1 · expected 1/det A | 0.83 · 0.91 · 0.86 | 0.93 · 0.88 · 0.90 | 0.81 · 0.78 · 0.83 | 0.84 · 0.77 · 0.87 | 0.84 · 0.91 · 0.86 | 0.84 · 0.89 · 0.86 |
| tile-band power (period, vox) | 3.6 (100) | 5.1 (145) | 5.6 (92) | 3.4 (102) | 4.1 (143) | 4.5 (100) |
| det J near bank: median · 5–95 % · min · max | 1.14 · 0.92–1.51 · 0.81 · 1.90 | 1.15 · 0.87–1.35 · 0.62 · 2.27 | 1.25 · 1.00–1.67 · 0.76 · 2.12 | 1.15 · **0.64–1.91 · 0.13** · 2.34 | 1.19 · 0.96–1.49 · 0.81 · 2.38 | 1.08 · 0.63–1.78 · 0.55 · 2.30 |
| det J outside [0.5, 2] · folds | 0.0 % · 0 | 0.2 % · 0 | 0.2 % · 0 | **5.6 %** · 0 | 0.0 % · 0 | 1.1 % · 0 |
| bank contacts in ENTl | 384/384 | 213/384 | 233/384 | 259/384 | 222/384 | 107/384 |
| ENTl contacts within 50 / 100 / 150 µm of sup/deep | 0.28 / 0.53 / 0.70 | 0.28 / 0.55 / — | 0.09 / 0.23 / — | 0.22 / 0.41 / — | 0.16 / 0.34 / — | 0.13 / 0.27 / — |
| stored Allen coords vs native annotation, disagreement | 4.7 % | 2.6 % | 7.8 % | 8.3 % | 6.0 % | 5.7 % |
| along-track flicker fraction · ordinal reversals | 0.03 · 5 | 0.06 · 2 | 0.12 · 5 | 0.11 · 10 | 0.10 · 2 | 0.16 · 4 |

Reading it:

- **Affine and mount.** Every affine is plausible; the DV compression (1.09–1.15) is shared by all six. Residual yaw after registration is ≤ 0.2° in every mouse and the atlas outline midline sits on the tissue midline to within one voxel (median 0–5 µm). The mount angle is fully absorbed. Residual pitch reads up to 1° (ah09) but the DV landmark is the tissue top at the midline column, where dura and sinus sit in the mask; it is advisory. The mask asymmetry is threshold-invariant (≤ 0.004), so the per-page L/R profiles are geometry, not illumination.
- **Truncation.** 2.8–4.3 mm of anterior brain is missing in every mouse (ly05 most), and 0.5–1.9 mm posteriorly. The affine AP scale is 0.97–1.00 in all, so the atlas overhangs; the overlays at the cut faces (`*_overlay_whole.png`, first and last coronal panels) show label edges continuing into agarose, as expected, and nothing pulled.
- **Local scale at the target.** The ENTl sample/atlas volume ratio is within 0.07 of CA1's in every mouse — no differential squeeze of the target region.
- **Deformation field.** No folds anywhere. Five mice have ≤ 1.1 % of near-bank voxels outside [0.5, 2]; **ly05 has 5.6 % with det J down to 0.13** — the warp is compressing the atlas by up to 7× locally to follow missing tissue (see below).
- **Fragility.** Between 23 % and 55 % of ENTl bank contacts lie within 100 µm of the superficial/deep boundary. This is geometry — the shanks cross the boundary obliquely — and it is why the flip fraction in check E, not the offset alone, is the number to read.
- **Tile bands.** Every registration channel carries stitching bands at a 0.9–1.45 mm period (strongest in ah09, ah10); the sagittal panels show them. brainreg's block-matching survived them (the outline residuals above), but they are the likeliest cause of the per-page noise in the L/R profiles.

### 6b. At the track (check E) — the layer question

Baseline-corrected by the atlas's own offsets (pia +24 µm, grey/white +5 µm). Offsets are
atlas − visible (positive: atlas boundary deeper); IQR across lines in brackets.

| | ah08 | ah09 | ah10 | ly05 | ly06 | ly07 |
|---|---|---|---|---|---|---|
| ENTl bank contacts (lines) | 384 | 213 | 233 | 259 | 222 | 107 |
| lines whose ENTl exit is a fibre tract | 0.91 | 0.65 | 0.54 | 0.86 | 0.66 | 0.57 |
| grey/white edges found (n) | 349 | 138 | 125 | 223 | 147 | 61 |
| **pia offset**, raw → corrected (IQR) | +19 → **−5** (16) | +28 → +4 (21) | +14 → **−10** (21) | +17 → **−7** (16) | +31 → **+7** (35) | +28 → +4 (15) |
| **grey/white offset**, raw → corrected (IQR) | +15 → **+10** (15) | +160 → +155 (120) *unreliable* | +25 → **+20** (90) | +15 → **+10** (55) | +65 → **+60** (100) | +180 → **+175** (50) |
| cortical thickness, atlas / visible (ratio) | 670 / 675 (**1.02**) | 828 / 681 (1.26) | 685 / 663 (**1.05**) | 790 / 773 (**1.03**) | 710 / 692 (**1.06**) | 610 / 458 (**1.39**) |
| L1/L2 offset (fraction of lines with a visible step) | −5 (0.36) | −105 (0.35) | +20 (0.25) | −25 (0.28) | −25 (0.64) | −28 (0.10) |
| **sup/deep flip fraction**, raw → corrected | 0.05 → **0.02** | 0.19 → 0.16 | 0.07 → **0.06** | 0.08 → **0.08** | 0.02 → **0.01** | 0.07 → **0.07** |
| what ENTl exits into (top labels) | ec 229 · alv 120 · HPF 35 | ec 138 · TR 37 · PIR 14 | ec 74 · HPF 59 · alv 51 · ENTm5 27 | alv 196 · ec 27 · ENTm1 21 | ec 83 · HPF 68 · alv 64 | alv 55 · SUB 23 · ENTm2 15 |
| flags / advisories | fragile 0.53 | wm unreliable; fragile 0.55 | – | jacobian | – | **laminar_offset**; jacobian |

Reading it:

- **ah08, ah10, ly05, ly06 — the registered pia and grey/white edges sit on the visible ones.**
  Pia within 10 µm (one voxel) in all four; grey/white within 10–20 µm in ah08/ah10/ly05 and +60 µm
  in ly06 (at the ENTl/ENTm border, IQR 100); cortical thickness ratios 1.02–1.06.  Where the
  autofluorescence shows a layer-1/2 step it sits within 5–25 µm of the atlas's.  Re-reading every
  contact's depth against the visible edges flips the superficial/deep label of **1–8 %** of ENTl
  bank contacts.  For a warp with 400 µm knots this is as good as the method can report.
- **ly07 — the atlas's deep ENTl boundary is ~175 µm too deep** at its 107 ENTl contacts (61 usable
  lines, IQR 50, i.e. a consistent edge; thickness ratio 1.39).  ly07 sits at the ENTl/ENTm/SUB
  junction where ENTl's deep layers wrap round the subiculum; its ENTl exits are alveus / SUB / ENTm.
  Few of its contacts are near the sup/deep boundary (flip 7 %), but every deep-ENTl-vs-SUB/ENTm
  assignment there carries ~200 µm of uncertainty.  This is the same mouse whose tip label swung
  between ENTl6a / ENTm5 / SUB across fit strategies (`PROBE_REFIT.md` §7).
- **ah09 — no grey/white edge to compare.**  Its ENTl abuts TR / piriform on 35 % of lines and the
  profile has no dark fibre band (IQR 120 µm); the pia agrees to +4 µm.  Not in the ephys cohort.
- **ly05 — locally fine, surroundings compromised.**  At the bank itself both edges agree (−7 / +10 µm,
  ratio 1.03, flips 8 %), but the deformation field within 1 mm has 5.6 % of voxels outside
  [0.5, 2] with det J down to 0.13, the L/R volume asymmetry reaches 0.24 (ProS), and the overlays
  show ENTl/ENTm contours drawn over torn, displaced dorsal-left cortex in the band AP 4.1–5.9 mm
  that contains the bank (AP 4.06–5.08 mm).  The warp is following missing tissue.
- **Fragility is geometric, not registration.**  Even with perfect edges, 23–55 % of ENTl bank
  contacts lie within 100 µm of the sup/deep boundary because the shanks cross it obliquely.  The
  registration adds ≤ 20 µm at the edges in four mice; the *depth* of the bank along the shank
  (ah08: extrapolated 1.76× past its dye) is the larger uncertainty for the superficial/deep split.

### 6c. The LEC/MEC border (check D, ENTl ↔ ENTm)

The ENTl/ENTm border runs *along* the cortical sheet, so unlike the superficial/deep boundary it
has no intensity edge to re-read against and check E cannot measure it.  What can be counted is
how many recorded contacts sit within a given distance of the registered border — every one of
them would change region if the border were off by that much.  The edge errors measured in §6b
(5–20 µm in ah08/ah10/ly05, 60 µm at ly06's grey/white, 175 µm at ly07's deep boundary) say which
column to read.  Whole recorded bank (384 contacts), from `registration_qc.csv`; per shank from
`registration_lecmec_shanks.csv`.

| | ah08 | ah09 | ah10 | ly05 | ly06 | ly07 |
|---|---|---|---|---|---|---|
| ENTl / ENTm in bank | 384 / 0 | 213 / 0 | 233 / 5 | 259 / 31 | 222 / 96 | 107 / 117 |
| within 50 µm of the border | 0 | 0 | 52 | 72 | 71 | 70 |
| within 100 µm | 0 | 9 | 89 | 117 | 107 | 154 |
| within 150 µm | 5 | 34 | 96 | 170 | 192 | 200 |
| within 200 µm | 12 | 47 | 96 | 220 | 230 | 240 |
| at 150 µm: currently ENTl → would be MEC / currently ENTm → would be LEC | 5 / 0 | 10 / 0 | 91 / 5 | 139 / 31 | 96 / 96 | 102 / 81 |
| shanks carrying the exposure (contacts within 100 µm, of 96) | – | – | shank 3: 89 | shank 3: 79 · shank 2: 28 | shank 2: 96 · shank 3: 11 | shank 2: 88 · shank 3: 41 |
| advisory (> 25 % of bank within 100 µm) | – | – | – | **0.30** | **0.28** | **0.40** |

Shank ids are the module's (shank 3 posterior-most under the current convention); the channel↔shank
identity is still the unverified surgery-notes item.

Reading it:

- **ah08 and ah09** are nowhere near the border; ah08 is pure LEC at any plausible error.
- **ah10**'s exposure is one shank: shank 3 has 89/96 contacts within 100 µm of the border and
  would go to MEC wholesale if the border were 100 µm too lateral.  Its measured edge errors are
  10–20 µm, so the LEC label there probably stands, but the shank is a border shank.
- **ly06** straddles the border by construction (this was already visible in the per-shank census,
  0.54/0.77/1.00/0.00 ENTl by shank): shank 2 is entirely within 100 µm and shank 3, labelled
  entirely ENTm, is within 200 µm.  Its grey/white edge error was 60 µm.
- **ly07** is the fragile case: 154 contacts within 100 µm and 200 within 150 µm, and its measured
  deep-ENTl boundary error is 175 µm — exactly the scale that relabels half its bank.
- **ly05** has 117 contacts within 100 µm on shanks 2–3, in the damaged band.

For any LEC-versus-MEC comparison, ly06 shank 2, ly07 shanks 2 and 3, and ah10 shank 3 should be
treated as **border contacts**, not assigned to either region.

## 7. Verdicts

Proposed from the numbers and the overlays; Adam records the authoritative verdict with
`rq.record_verdict(mouse, verdict, note)` after looking at `<mouse>_overlay_lec.png`.

| mouse | proposed | why |
|---|---|---|
| **ah08** | **ok** | edges within 10 µm, thickness ratio 1.02, 2 % flips; loose cortical flap on the *right* (non-probe) hemisphere at AP 6.5–7.4 mm only |
| **ah10** | **ok** | edges within 20 µm, ratio 1.05, 6 % flips — the atlas layers its depth (3400 µm) was anchored to are where the tissue shows them |
| **ly06** | **ok**, border shank | pia +7, grey/white +60 (IQR 100 at the ENTl/ENTm border), ratio 1.06, 1 % flips; shank 2 (96) and shank 3 (ENTm) sit within 100–200 µm of the LEC/MEC border |
| **ly05** | **suspect** | edges agree at the bank but det J 0.13–2.3 nearby, L/R asymmetry 0.24, torn dorsal-left cortex in the band holding the bank; 117 contacts within 100 µm of the LEC/MEC border on shanks 2–3 |
| **ly07** | **suspect** | deep ENTl boundary +175 µm too deep (consistent, IQR 50), ratio 1.39; sup/deep split barely affected (7 %) but 154–200 contacts sit within 100–150 µm of the LEC/MEC border, the same scale as the measured error |
| ah09 | ok (global) / n.a. (laminar) | not recorded; registration globally sound, no fibre band under its ENTl to test against |

**Answer to the question that started this** — *does the registration support superficial vs deep
ENTl labels?*  In ah08, ah10 and ly06, yes, to within one or two voxels at both edges the warp can
see.  In ly05, at the bank, yes — but the tissue around it is damaged and the warp is straining.
In ly07 the deep boundary is off by ~175 µm.  Nowhere is the registration the dominant uncertainty
for the superficial/deep split: probe depth is.  None of the hand placements needs redoing; they
were made against the dye, and re-labelling is automatic if brainreg is ever re-run (`ah10` is the
one whose depth leaned on the atlas layers, and those are now verified).

### Not done / next

- **Check F (landmark TRE)** is built (`landmark_tool.py`, `ipympl 0.10.0` installed) but needs
  Adam's clicks: `%matplotlib widget`, `lt.LandmarkPicker('atlas').show()` once, then per mouse.
- A section-slippage test by page-to-page image cross-correlation (the centroid statistic is not
  one).
- If ly07's deep boundary matters for a claim, a local re-registration of the ENTl/SUB junction
  (cropped, finer control grid) or a hand-drawn ENTl/SUB boundary on ly07's sections.

## 8. Provenance and outputs

```
data/figures/registration_qc/<mouse>_{overlay_whole,overlay_lec,checker_atlas,lr_profile,laminar,jacobian}.png
data/preprocessed_data/brainreg/<mouse>/registration_qc_global.npz      # page stats, tissue threshold
data/preprocessed_data/brainreg/<mouse>/registration_landmarks.json     # check F
data/preprocessed_data/brainreg/atlas_landmarks.json                    # check F, clicked once
data/preprocessed_data/brainreg/atlas_region_volumes.csv                # atlas volumes + AP extents (25 µm)
data/preprocessed_data/brainreg/registration_qc.csv                     # one row per mouse
data/preprocessed_data/brainreg/registration_lecmec_shanks.csv          # per shank: ENTl/ENTm counts, distance to the border
data/preprocessed_data/brainreg/registration_qc_verdicts.json           # human verdicts, dated, fit-keyed
```

brainreg was **not** re-run and no stored fit was modified.  `ipympl 0.10.0` was installed into the
`histology` env for click-to-place landmarks (approved 2026-09-11).
