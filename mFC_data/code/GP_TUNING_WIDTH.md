# W5 (PFC mirror) — goal-progress tuning: sorted β heatmaps, and peak vs width

The method, the confound argument, the definitions, the synthetic controls and the full results for
both datasets are in [`../../code/GP_TUNING_WIDTH.md`](../../code/GP_TUNING_WIDTH.md). This file is the
PFC-side pointer, as `GLM_V3.md` is.

```bash
python mFC_data/code/gp_tuning_width.py --build            # per-recday leg-curve caches (9.5 min)
python mFC_data/code/gp_tuning_width_synthetics.py         # 8 controls through the real pipeline
python mFC_data/code/gp_tuning_width.py --check-mirror     # the two module copies are byte-identical
jupyter nbconvert --to notebook --execute --inplace PFC_gp_tuning_width.ipynb
```

`gp_tuning_width.py` is byte-identical to `code/gp_tuning_width.py` and detects its tree
(`IS_PFC`), like `glm_v3_synthetics.py`. It was deliberately **not** added to
`check_mirror_parity.BYTE_PAIRS` (the standalone instruction of 2026-09-08); the module's own
`assert_mirror()` runs at the top of both notebooks instead.

## PFC in one table

| | |
|---|---|
| unit-recordings / mice | 1252 / 7 |
| legs (≤ 30 s) / dropped at the cap | 16,545 / 331 |
| gp-significant (uniform/30) | 77.8 % (86.0 % in gp-only/30) |
| peak reproduced in the held-out half | 95.9 % |
| median split-half r / median half-max width | 0.925 / 0.489 |
| multimodal / ramp / peak moves > 30 bins | 19.1 % / 4.2 % / 43.0 % |
| **ρ(\|peak − 0.5\|, width)** — the V | **−0.521 [−0.632, −0.423]**, 7 mice |
| **ρ(peak, width)** — anchor asymmetry | **+0.303 [+0.153, +0.477]**, 7 mice |
| planted time cells / phase cells, same code | −0.482 / −0.005 (the V) |
| planted retrospective / prospective only | +0.288 / −0.285 (the asymmetry) |
| ρ(count, width) across the ten bins | +0.01 (peaks are evenly U-shaped in PFC) |
| synthetic controls | 8/8 pass |

PFC gives the same V as LEC on more mice, and the same retrospective asymmetry, weaker. It survives
restriction to phase-stable cells (−0.549) and to a single session per recday (−0.478).

## The A–B peak-disagreement gate (robustness variant)

`PFC_gp_tuning_width_abgate.ipynb`, figures in `mFC_data/data/figures/gp_tuning_width_abgate/`, region
guide in `docs/figures/gp_tuning_width_pfc_abgate/`. Keeps cells whose odd-leg and even-leg peaks are
within **30 bins of the 90-bin leg axis** (a third of a leg — bins, not degrees); the planted
populations and the β heatmaps are gated identically. The one figure that cannot change is
`pfc_width_calibration.pdf`, which plots only planted phase cells and the gate excludes none of those.
The gate sweep is written only here, not to the primary directory.

**PFC is where gating the null earns its keep.** The planted time cells' V does not move under the gate
(−0.482 at every threshold) while the real V strengthens, so the gap between data and null widens:

| gate | real | sim time cells | cells kept |
|---|---|---|---|
| none | −0.554 | −0.482 | 100 % |
| ≤ 30 | −0.576 | −0.482 | 96 % |
| ≤ 20 | −0.587 | −0.479 | 90 % |
| ≤ 15 | −0.599 | −0.490 | 85 % |
| ≤ 10 | −0.629 | flat | 72 % |

Per mouse at the 30-bin cut: **−0.573 [−0.686, −0.465]**, 7 mice, against −0.521 ungated. LEC behaves
differently — there the gate moves the null too — see `../../code/GP_TUNING_WIDTH.md` §4.8.

Figures: `mFC_data/data/figures/gp_tuning_width/`. Per-figure guide:
`docs/figures/gp_tuning_width_pfc/PFC.md`. Caches:
`mFC_data/data/processed_data/gp_leg_curves/`. `mFC_data/data/**` is periodically set read-only, so
the module re-opens a figure directory before writing to it.
