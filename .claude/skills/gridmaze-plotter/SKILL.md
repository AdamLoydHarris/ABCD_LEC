---
name: gridmaze-plotter
description: "GridMaze plotting specialist. Use when creating or modifying analysis plots in the GridMaze repository so figures match the repo publication style: Arial font via `setup_arial_font`, 8 pt defaults, clean axes, deterministic colors from the GridMaze Colors skill, and appropriate continuous colormaps. Use for new Matplotlib plotting functions, plot refactors, and saved analysis figures that need repo-consistent styling."
---

# GridMaze Plotter

Use this skill to make GridMaze plots match the repository style without adding unnecessary plotting machinery.

## Apply the repository font and rcParams

- Import `setup_arial_font` from `GridMaze.analysis.core.font`.
- Call `setup_arial_font()` once before plotting.
- Prefer the repo helper over custom rcParams.
- If the helper cannot be imported, add Arial from `/ceph/behrens/max_kirkby/goal_sequencing_1/font/Arial.ttf` and set:
  - `font.family = "Arial"`
  - `pdf.fonttype = 42`
  - `ps.fonttype = 42`
  - `font.size = 8`
  - `figure.titlesize = 8`
  - `axes.titlesize = 8`
  - `axes.labelsize = 8`
  - `xtick.labelsize = 8`
  - `ytick.labelsize = 8`
  - `legend.fontsize = 8`
  - `axes.linewidth = 0.8`

## Style axes cleanly

- Hide the top spine.
- Hide the right spine.
- Keep line widths aligned with the repository defaults.
- Use gray or black only for non-data ink such as reference lines and axis furniture.
- Avoid seaborn global styling unless the user explicitly asks for it.

## Use GridMaze color palettes

- Also use the `gridmaze-colors` skill when choosing plot colors.
- Prefer named GridMaze palettes for common contrasts such as true/shuffle, past/future, maze 1/maze 2/rooms, habit/vector/structure, navigation/RC/ITI, and random/optimal/behaviour/neurons.
- Use exact hex values from `gridmaze-colors` when a named palette applies.
- For categorical groups not covered by a named palette, define a clearly named GridMaze palette before plotting.
- Build deterministic mappings.
- Use sorted categories instead of order-of-appearance only when stability across reordering matters more than preserving input order.
- Do not introduce ad hoc hex colors for categorical data unless the user explicitly requests them.

## Write plot code in a reusable shape

- Create one public plotting function for the requested figure.
- Name it clearly, for example `plot_<thing>(...)`.
- Return a `matplotlib.figure.Figure`.
- Accept `ax=None` when a single-axis version is practical.
- Accept `out_path=None` when saving is requested.
- Keep plotting code out of top-level module scope.
- Prefer simple, skimmable plotting code over abstraction.

## Save figures without tight auto-cropping

- When saving from plotting code, wrap the save call in `mpl.rc_context`.
- Set:
  - `savefig.bbox = None`
  - `savefig.pad_inches = 0.0`
  - `pdf.fonttype = 42`
  - `ps.fonttype = 42`
- Call `fig.savefig(..., bbox_inches=None)`.

## Handle continuous fields separately

- Use `RdBu_r` for z-scored firing rates, signed residuals, and zero-centered diverging heatmaps.
- Use symmetric `vmin` and `vmax` when centering `RdBu_r` at zero.
- Use `viridis` for excess steps, scalar fields, and non-diverging continuous quantities.
- Still use named GridMaze palettes for categorical overlays, groups, or legend entries on the same figure.

## Keep implementations minimal

- Match surrounding module style.
- Use early returns.
- Avoid clever helper layers unless they remove real repetition.
- Prefer concise labels and publication-ready defaults over decorative styling.
