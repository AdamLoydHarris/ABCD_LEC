---
name: gridmaze-colors
description: "GridMaze figure color palette guidance. Use when creating or modifying GridMaze plots, choosing colors for behavioural/ephys conditions, categorical comparisons, ordered ramps, controls, confidence intervals, structural maze elements, or diverging/continuous heatmaps."
---

# GridMaze Colors

Use this skill to choose publication-ready colors for GridMaze figures. Prefer these named palettes over ad hoc hex colors and keep mappings deterministic within each plot.

## General Rules

- Choose the smallest named palette that matches the scientific contrast.
- Use pair palettes for binary comparisons, categorical sets for unordered labels, and ordered ramps only when the plotted variable has an order.
- Keep neutral colors for nulls, controls, confidence intervals, structure, outlines, and non-data ink.
- Use the exact hex values below when a named palette applies.
- For categories not covered below, define a clearly named GridMaze palette before plotting.
- Avoid introducing one-off hex colors unless a user explicitly requests them.

## Two-Condition Pairs

Use these for two conditions with a clear semantic contrast.

- Viva Magenta / Classic Blue: `#BE3455`, `#0F4C81`
  - Direct paper red/blue upgrade. Good for thin red/blue traces.
- Living Coral / Lush Meadow: `#FF6F61`, `#009473`
  - Warm/cool with a grounded green.
- Very Peri / Saffron: `#6667AB`, `#FFA500`
  - Strong luminance contrast; robust in greyscale.
- Opto / control: `#1F77B4`, `#2C2C2A`
  - Manipulation against baseline.
- True / permuted or shuffle: `#C03030`, `#555555`
  - Real signal against null.
- Past / future: `#C03030`, `#2A6FB5`
  - Warm retrospective, cool prospective.

## Three-Way Palettes

- Ultra Violet / Viva Magenta / Peach Fuzz: `#6B3FA0`, `#BE3455`, `#FFBE98`
  - Warm-shift ramp for maze-1 / maze-2 / rooms style ordering.
- Aurora Red / Living Coral / Aspen Gold: `#9B1B30`, `#FF6F61`, `#FFD662`
  - All-warm ordered contrast when ordering matters more than discriminability.
- Classic Blue / Biscay Bay / Greenery: `#0F4C81`, `#00A6A6`, `#88B04B`
  - Cool-shift ramp for habit / vector / structure style components.
- Mazarine Blue / Emerald / Lime Punch: `#34558B`, `#009B77`, `#C0D725`
  - Punchier cool-shift with a high-salience third condition.
- Magenta / Blue / Greenery: `#BE3455`, `#0F4C81`, `#88B04B`
  - Maximum hue separation for genuinely categorical three-way conditions.
- Maze 1 / maze 2 / rooms: `#7000A0`, `#C04070`, `#F09040`
  - Purple to magenta to orange ramp for ordered non-numeric stages.
- Habit / vector / structure: `#305080`, `#209080`, `#60B060`
  - Navy to teal to green for behavioural or variance decompositions.
- Navigation / RC / ITI: `#C03030`, `#888780`, `#2A6FB5`
  - Active / transition / rest split.
- Pulse durations 1 / 10 / 30 s: `#C03030`, `#2A6FB5`, `#60B060`
  - High-contrast categorical triple.

## Four-Way Palettes

- Stone / Turquoise / Classic Blue / Viva Magenta: `#B4B2A9`, `#45B5AA`, `#0F4C81`, `#BE3455`
  - Ordered null-to-best ramp for random / optimal / behaviour / neurons style plots.
- Random / optimal / behaviour / neurons: `#888780`, `#00CED1`, `#2A6FB5`, `#C03030`
  - Existing grey to cyan to blue to red model-comparison ramp.

## Anchor Neutrals

- Cloud Dancer: `#F0EEE9`
  - Paper-warm off-white backgrounds when a non-white background is useful.
- Stone: `#B4B2A9`
  - Shuffle/null, subdued controls, CIs.
- Caviar: `#2C2C2A`
  - Primary trace black and high-contrast structural elements.
- Legacy neutral grey: `#888780`
  - Use when matching existing navigation / model comparison figures.
- Shuffle grey: `#555555`
  - Use with true / shuffle comparisons.

## Continuous Colormaps

- Use `RdBu_r` for z-scored firing rates, population rasters, signed residuals, and other zero-centered diverging values. Center white at zero with symmetric `vmin` and `vmax`.
- Use `viridis` for excess steps, scalar fields on maze plots, occupancy-normalized scalar values, and non-diverging continuous quantities.

## Implementation Pattern

Prefer local named dictionaries in plotting modules when the mapping is figure-specific:

```python
colors = {
    "random": "#888780",
    "optimal": "#00CED1",
    "behaviour": "#2A6FB5",
    "neurons": "#C03030",
}
```

If a palette becomes shared across multiple modules, add a small repo helper rather than duplicating a long dictionary.
