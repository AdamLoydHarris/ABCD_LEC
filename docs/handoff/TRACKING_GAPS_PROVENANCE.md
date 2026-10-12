# Where the LEC tracking gaps come from

**Written 2026-10-08.** Every number below was measured on this machine, not inferred.

**One line.** The NaN stretches in the v3/v4 LEC tracking are **90 % frame-clock uncertainty**,
8 % SLEAP never tracking the frame and 2 % QC rules removing it. The cleaning thresholds are
exonerated; the limit is camera-to-pyControl synchronisation.

---

## 1. Why this was asked

The El-Gaby LEC port is migrating from `data_dic_lec.pkl` (v1) to `data_dic_lec_v4.pkl`. v1
carries **no NaN at all** in `XY_raw`; v4 carries NaN in 21 of 211 sessions, up to 9.4 % of a
session. That looked like a regression. It is not: `tracking_clean.py` line 28 records that
**v1 interpolated every gap with no length limit**, so v1's cleanliness was straight-line
fabrication across stretches up to 30 s, unmarked. v4 declines to invent those positions.

The export then has to decide what to write for them, which is what made the provenance worth
establishing.

## 2. What the gaps are

From `mFC_data/lec_replication_run_v4/logs/tracking_gaps.csv` (written by
`15_export_lec_to_flat.py`, one row per gap):

| | |
|---|---|
| gaps | 57, over 21 sessions |
| total | 17 270 bins of 25 ms = 7.2 min |
| shortest / median / longest | 60 bins (1.5 s) / 316 (7.9 s) / 1213 (30.3 s) |
| overlapping a trial | 52 of 57 |
| worst session | `ah08_20250616_20250617` s3, 4472 bins = 9.4 % of the session |
| next | `ah10_20250613_20250615` s7 (4.5 %), `ly07_20250616_20250617` s3 (4.7 %) |

**There is no short-gap population.** The shortest surviving gap is 1.5 s because
`tracking_clean` already bridges interior runs ≤ 120 frames (2 s) by straight line; 5160 frames
were bridged that way and are in v3/v4 as ordinary positions. Everything that survives is a gap
the cleaner judged too long to bridge honestly.

## 3. The three sources, and how they are separable

`tracking_clean.py` writes a per-frame, per-bodypart int8 flag beside each SLEAP session in
`data/preprocessed/sleap/maze_v2/*.analysis_track_flags.csv` — tens digit = outcome (0 kept,
1 bridged, 2 left NaN), units digit = reason (1 raw NaN from SLEAP, 2/3 low score kept,
4 R1 velocity, 5 R2 spike, 6 R3 Hampel, 7 low score AND off trajectory). 472 files, all five
mice. Position is `head_back`.

That accounts for two sources. The third is **not** in the flags, because it happens later: the
frame clock (`code/preprocessing/frame_clock.py`) marks a stretch uncertain when the alignment
step exceeds `UNCERTAIN_STEP_FRAMES = 6` (0.1 s), or when a gap longer than
`GAP_UNCERTAIN_S = 30` carries a step jump. The aligner then refuses to place the animal at
all. The count is per session in `data/processed_v3/sync_qc_lec.csv`, column `n_bins_uncertain`.

### The split

| source | bins | share |
|---|---|---|
| frame-clock uncertainty (alignment) | 15 612 | **90 %** |
| SLEAP never tracked the frame (`raw_nan`) | ~1 436 | 8 % |
| QC rules removed the frame | ~219 | 2 % |

Of the 2482 frames the cleaner left as NaN, **2154 (86.8 %) were already NaN from SLEAP** and
328 (13.2 %) were QC removals — and all 328 came from **one** rule, `lowscore_off_trajectory`
(reason 7). The velocity (R1), spike (R2) and Hampel (R3) rules removed **nothing** that ended
up as a gap.

**Sessions partition almost perfectly by source.** 14 of the 21 are purely clock-uncertain with
zero cleaner-removed frames; 6 are purely cleaner-side with zero uncertain bins; only
`ly05_20250613_20250615` s3 mixes the two (359 uncertain, 237 cleaner).

### What this means

* **No case for loosening the QC thresholds.** They account for ~2 % of the loss, from a single
  rule. Relaxing R1/R2/R3 would recover nothing.
* **The limit is synchronisation, not vision.** The 90 % are stretches where the frame clock
  cannot say which video frame corresponds to which time. Refusing to place the animal is the
  right behaviour — the alternative is confidently wrong positions, which is what v1 did.
* If anyone wants these bins back, the work is in the rsync/frame-clock matching, not in SLEAP
  or in `tracking_clean`.

## 4. Are the gaps in engaged or disengaged periods?

Neither — they are unremarkable. Rewards are the `trialtimes_` state boundaries; engagement is
the reward rate in a ±60 s context window around each gap, against 500 matched random windows
from the same session's trial span.

| 52 gaps inside the trial span | value |
|---|---|
| context reward rate | 4.00 / min |
| session reward rate | 3.71 / min |
| paired Wilcoxon | p = 0.36 |
| median null p | 0.435 |
| bottom 5 % (disengaged) / top 5 % (engaged) | 3 / 1 |
| expected in each tail by chance | 2.6 |

So tracking loss carries no information about what the animal was doing — consistent with a
camera or synchronisation failure rather than anything behavioural.

**A trap recorded because it bit.** The first version of this test scored each gap's own reward
count with `mean(null <= x)`. In a low-rate session most windows contain zero rewards, so a
zero-reward gap tied with all of them and scored p ≈ 0.95, printing as "engaged"; that produced
13 spurious engaged calls. Fixed two ways: a mid-p rule for ties
(`mean(null < x) + 0.5 * mean(null == x)`, gated to return 0.50 on an all-ties null) and a
context window, since a 9 s window at 4 rewards/min expects 0.6 rewards and is mostly Poisson
noise.

## 5. What the export does with them

`15_export_lec_to_flat.py` (v4 copy) resolves gaps with `--interp-max-bins`, **default 0**:
every gap is written as `XY_UNTRACKED = -1.0`. This is deliberate and faithful. El-Gaby's speed
cell (`Figure2.ipynb` c16 L29) does `whl_XX[whl[:,0]<0] = np.nan` and *then* Gaussian-filters,
so a negative x is his own untracked signal and NaN spreading around it is his intended
behaviour. Writing NaN instead would **not** work: `NaN < 0` is False, so it would slip past his
screen into the filter and poison the whole smoothed trace rather than blanking a few bins.

Interpolation remains available behind the flag, but the measurement in §2 says there is nothing
short to interpolate: the smallest gap is 1.5 s and the median 7.9 s.

## 6. Not established

* The per-gap frame-level join. The provenance here is per session: the flag counts are 60 fps
  frames and the gap counts are 25 ms bins, converted by 2/3, so the 8 % and 2 % figures are
  approximate to a few percent. A per-gap join needs the frame→bin map from the aligner.
* Why these particular 21 sessions have uncertain clock stretches and the other 190 do not.
* Whether the 5160 bridged frames (≤ 2 s, straight-line) are accurate; they are unmarked in the
  pickle and nothing downstream distinguishes them from measured positions.
* Any downstream effect. No analysis has been compared with and without these bins.

## 7. Reproduce

```bash
E=/nfs/nhome/live/aharris/.conda/envs/mfc_replication/bin/python
R=/nfs/nhome/live/aharris/.conda/envs/maze_ephys/bin/python3
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data
P=mFC_schema-main_unblocked_lec_v4/_preflight

$R $P/15_export_lec_to_flat.py --overwrite     # -> logs/tracking_gaps.csv
$E $P/91_tracking_gap_context.py               # -> logs/tracking_gaps_context.csv  (engagement)
$R $P/89_tracking_gap_provenance.py            # -> logs/tracking_gap_provenance.csv (source split)
```

Outputs, all under `mFC_data/lec_replication_run_v4/logs/`:

| file | one row per | key columns |
|---|---|---|
| `tracking_gaps.csv` | gap | `start_bin`, `n_bins`, `seconds`, `inside_a_trial`, `resolution` |
| `tracking_gaps_context.csv` | gap | the above plus `context_rate_per_min`, `session_rate_per_min`, `null_p`, `verdict` |
| `tracking_gap_provenance.csv` | session | `n_left_nan`, `left_raw_nan`, `left_qc`, `pct_qc_of_left`, `n_bridged`, per-reason counts |

The third source is joined from `data/processed_v3/sync_qc_lec.csv`, column `n_bins_uncertain`.
