# BUG: `ly05_20250618_20250619` paired neural data from one day with behaviour from another

**Status:** resolved 2026-09-01 · **Severity:** high (silent, corrupted one recday's results) ·
**Scope:** 1 of 24 recdays · **Found:** 2026-09-01, by the unit-count gate while building
the anatomical region census (`code/histology_refit/region_assignment.py`).

**Outcome:** both affected recording days were re-extracted from their own sortings. The
cohort went from **24 recdays to 25** — the fix recovered data rather than discarding it.

---

## 1. Summary

For the recday `ly05_20250618_20250619`, `data_dic_lec.pkl` held

- **behaviour** (trial times, task sequences) genuinely recorded on **2025-06-18/19**, and
- **spikes** (`Neuron_raw`) from the sorting of **2025-06-20/23**.

The two are from different recording days. Every analysis that used this recday therefore
correlated one day's spikes against another day's behaviour.

It was **not** detectable from array shapes — the neural data is binned onto the
behavioural timeline, so lengths agree exactly by construction. Only the *neuron count*
revealed it.

---

## 2. Evidence

### 2.1 The behaviour is from 06-18/19

The stored task sequences under this recday match the 06-18/19 metadata **8 of 8**, and the
06-20/23 metadata **0 of 8**:

| session | stored `Task_data` | in 06-18/19 metadata | in 06-20/23 metadata |
|---|---|---|---|
| 0 | 5-9-7-6 | ✓ | ✗ |
| 1 | 6-4-8-5 | ✓ | ✗ |
| 2 | 5-2-9-1 | ✓ | ✗ |
| 3 | 5-9-7-6 | ✓ | ✗ |
| 4 | 7-4-5-1 | ✓ | ✗ |
| 5 | 6-7-2-5 | ✓ | ✗ |
| 6 | 1-3-2-6 | ✓ | ✗ |
| 7 | 7-4-5-1 | ✓ | ✗ |

(06-20/23 sequences were 1-2-5-9, 7-4-6-5, 7-4-6-5, 7-9-8-4 / 5-6-9-3, 1-7-6-4, 9-8-6-7,
9-8-6-7 — none appear.)

### 2.2 The spikes are from 06-20/23

`Neuron_raw_ly05_20250618_20250619_0.npy` contained **91 neurons**. Against the QC'd unit
counts of ly05's five sorted blocks:

| sorted block | QC single units | extracted recday | `Neuron_raw` neurons |
|---|---|---|---|
| 2025-06-13_2025-06-15 | 117 | `ly05_20250613_20250615` | 117 ✓ |
| 2025-06-16_2025-06-17 | 116 | `ly05_20250616_20250617` | 116 ✓ |
| **2025-06-18_2025-06-19** | **109** | *(never extracted)* | — |
| **2025-06-20_2025-06-23** | **91** | → `ly05_20250618_20250619` | **91** ✗ |
| 2025-06-24_2025-06-25 | 74 | `ly05_20250624_20250625` | 74 ✓ |

Three of four files matched their labelled block exactly; the fourth matched a block it is
not named after. The 06-18/19 sorting (109 units) never entered `data_dic`.

**Confirmed independently during the fix.** Re-extracting the 06-20/23 block produced sleep
arrays of shape (91, 72021), (91, 49333), (91, 49143), (91, 49218), (91, 48692),
(91, 84020), (91, 53063), (91, 49604), (91, 48625), (91, 48551) — *identical* to the
`ly05_20250618_20250619_sb_*` arrays that were on disk. Those files were the 06-20/23 block,
byte for byte.

### 2.3 It was confined to ly05

Scanning every recday's `Neuron_raw` neuron count against the QC count of the block it is
*named after*, `ly05_20250618_20250619` was the only mismatch in the cohort (ah08, ah10,
ly06, ly07: 5 blocks and 5 extracted recdays each, no mismatch).

---

## 3. Mechanism

ly05 had **five** sorted blocks but only **four** extracted recdays. The observed pairing was

```
recday[0] <- block[0]    13-15  (117)  correct
recday[1] <- block[1]    16-17  (116)  correct
recday[2] <- block[3]    20-23  ( 91)  WRONG  (named 18-19)
recday[3] <- block[4]    24-25  ( 74)  correct
                block[2] 18-19  (109)  never used
```

i.e. **block[2] was skipped while the recday name list kept its name**, so subsequent blocks
shifted up by one into earlier names. This is the signature of positional pairing
(`zip`/index) between two lists of different length, rather than matching on the date.

The writer of `data/processed_data/neuron_raw_mingyutest/` was not in this repository's
tracked code, so the exact line could not be inspected; the mechanism above is inferred from
the pairing pattern and is consistent with every count. It has been replaced by
`code/preprocessing/extract_neuron_raw.py`, which matches on dates only.

The `data_dic` builder was **not** at fault: it joins `Neuron_raw_{recday}_{s}.npy` to
`trialtimes_{recday}_{s}.npy` strictly by name and derives its recday list from the
`Neuron_raw` filenames, so it faithfully propagated whatever the extractor wrote.

---

## 4. Why it went unnoticed

`trialtimes.max()` equals `Neuron_raw.shape[1]` **exactly** for every scored session — the
neural data is binned onto the behavioural timeline by construction. Any shape, length or
alignment check therefore passes; `truncate_all_arrays` has nothing to complain about. The
only observable that separates the two days is the **number of units**, which nothing
downstream was checking.

One analysis did notice and said nothing useful: cell 20 of `LEC_sploratory_analysis.ipynb`
prints *"LEC units mask length … does NOT match … Skipping this mouse_recday"* and drops the
recday. It printed; nothing asserted; nobody read it.

---

## 5. Impact (before the fix)

- **Direct:** every analysis using `ly05_20250618_20250619` related 06-20/23 spikes to
  06-18/19 trial structure. That recday contributed noise, not signal — diluting group
  effects rather than creating spurious ones, since the pairing is arbitrary.
- **Extent:** 1 of 24 recdays (~4%), 91 of ~2740 neurons. Present in every cached GLM
  output, persistent-homology output, and any figure computed per recday.
- **Not affected:** the anatomical region arrays (`unit_regions.pkl`) are built per *sorted
  block*, so they used the correct 06-18/19 sorting. That is precisely why the gate fired:
  they legitimately disagreed with `data_dic` for this recday.

---

## 6. What was done

1. **Quarantined** the 18 mismatched `Neuron_raw_ly05_20250618_20250619*.npy` files and the
   20 stale `ph_outputs/ly05_20250618_20250619__*.pkl`, renaming them with an
   `.INVALID_ly05_recday_mismatch` suffix. Nothing can read them by accident, and the
   06-20/23 spikes in them stayed recoverable.
2. **Wrote a tracked, date-matched extractor**, `code/preprocessing/extract_neuron_raw.py`,
   gated on reproducing existing output byte-for-byte (section 8).
3. **Re-extracted both days** under their own names: `ly05_20250618_20250619` (109 units)
   and `ly05_20250620_20250623` (91 units), which had never been extracted at all.
4. **Moved the `data_dic` build into tracked code**, `code/preprocessing/build_data_dic.py`,
   which runs both guards before writing.
5. **Added the guards** in `code/recday_registry.py`, called by
   `glm_analysis_v2.load_data_dic` which every LEC notebook now uses instead of a bare
   `pickle.load`.

### Corrections to the original analysis in this document

- **§5 "Secondary" (removed).** The claim that `extract_pokes.py` clips poke bins to
  `Neuron_raw.shape[1]` was wrong. `get_end_bin` clips to `trialtimes.max()`
  (`extract_pokes.py:338-346`), which comes from the correct day. The poke files were never
  affected.
- **§3 "Which day was meant to be dropped" (removed).** That section inferred from
  `Performance = NaN` that 06-20/23 was the droppable day. The Performance column was simply
  never filled in for those dates. 06-20/23's trialtimes hold 14, 1, 18, 0, 28, 5, 0, 0
  trials — *more* than 06-18/19's 8, 13, 3, 0, 6, 8, 1, 5 — and its behavioural checks are
  excellent (section 8). Neither day should have been dropped, and both are now in.

---

## 7. The extraction convention

Recovered by reproducing the existing arrays exactly; worth recording because nothing else
documents it. All arithmetic is in **integer samples** — computing bin indices in float
milliseconds moves ~50 of 5 million bins by one, through rounding alone.

- Concat offsets: `cumsum(dat_size // 768)` over the rows of
  `recording_sessions_in_concat.csv`, in file order.
- Bin grid is **recording-anchored**: `bin = (sample - origin) // 750` (750 samples = 25 ms
  at 30 kHz). Block-anchored was tested and ruled out — it fails on every recording after
  the first.
- Rows follow `QC_single_units.npy` order; dtype `uint16`.
- Origin and length, three cases:
  - **sleep box**: `origin = 0`, `n_bins = n_samples // 750` (the whole recording).
  - **task session with trials**: `origin = first_rsync_sample - pc_rsync0_ms * 30
    + (first_A_on_ms // 25) * 750`, i.e. the first A_on snapped to the pyControl 25 ms grid
    exactly as `extract_pokes.to_bins` does, carried into ephys samples by the rsync offset.
    `n_bins = trialtimes.max()`.
  - **task session with no completed trials**: no A_on to anchor to, so
    `origin = first_rsync_sample` and `n_bins = (n_samples - first_rsync_sample) // 750`.
- The rsync offset is a **constant** shift, not a fitted linear map. Fitting the clock drift
  (measured slope 0.999989, ~13 ms over a 20 min session) is more accurate in principle but
  is not what produced the existing files.
- Session index → recording is resolved by the **metadata `Ephys` timestamp**, never by
  counting wake recordings: the wake recordings and the scored task rows do not always line
  up one-to-one (ah08 has a recorded-but-unscored "error sess"), and counting silently
  shifts every later session.

---

## 8. Verification

`python code/preprocessing/extract_neuron_raw.py --validate-only`

**Every array in the cohort reproduces byte-for-byte** — 474 re-derived from the cached
rsync trains, plus 16 whose origin had to be recovered first (below), across all 25 blocks.

### The two recdays with no rsync cache

`ephys_rsync_mingyutest/` holds no pulse trains for ah08 on 2025-06-18/19 or ah10 on
2025-06-16/17, so the first-`A_on` anchor for those 16 task arrays cannot be computed the
normal way. They are **not** left unchecked. `recover_origin` reads the origin back off
each stored array — for the true origin, the number of QC-unit spikes below bin edge *k*
equals a fixed offset plus the array's own cumulative population total through bin *k*,
which pins the origin to the exact sample — and all 16 then reproduce byte-for-byte.

That alone would be partly circular, so the recovered origins were checked against the
**raw TTL**, which does exist. Running `session_window`'s formula backwards gives an
implied first-sync sample:

| recday | stream | implied first-sync vs raw TTL |
|---|---|---|
| `ah10_20250616_20250617` | ProbeB | **exactly 0, in all 8 sessions** |
| `ah08_20250618_20250619` | ProbeD | 2 934–6 512 samples (~0.1–0.2 s) below raw TTL |

The ah10 result is an independent confirmation. The ah08 offset is not a defect: it is the
same stream-dependent discrepancy measured on ly05's ProbeA recordings, where the cached
rsync files *do* exist and the arrays reproduce byte-exactly (offsets 3154, 6195, 9232,
6382 samples). The cached rsync trains are not simply `TTL - continuous[0]` — the offset is
zero on ProbeB recordings and non-zero on ProbeA/ProbeD, and one cache holds more pulses
than its stream's TTL file — so they were derived from a source other than the `ttl_path`
named in `recording_sessions_in_concat.csv`. Whatever that source was, the same convention
is baked into every recday's arrays consistently.

Unrelated to this bug; recorded so it is not rediscovered.

`python code/preprocessing/validate_alignment.py` — all 25 recdays, 212 sessions pass:

| check | what it ties together | aligned | misaligned (measured) |
|---|---|---|---|
| `loc_at_goal` | `Locs_raw` ↔ `Trial_times` ↔ `Task_data` | 0.944–1.000 (90% exactly 1.000) | mean 0.096, max 0.250 |
| `poke_loc` | `pokes_*.npy` ↔ `Locs_raw` | 0.17–1.00, dominated by tracking quality | — |
| `task_ok` | `Task_data` ↔ that day's pyControl `active_poke` | 212/212 | — |
| `bins_ok` | `Neuron_raw` ↔ `trialtimes` ↔ tracking length | all scored sessions | — |

The misaligned column comes from deliberately pairing each session's tracking with a
neighbouring session's trialtimes. The two regimes do not overlap, which is why the
threshold sits at 0.60 — it separates misalignment from tracking noise, and is not a
tracking-quality standard.

Both recovered recdays score `loc_at_goal` = **1.000** and `poke_loc` at the top of ly05's
range (0.980 and 0.992 mean, vs 0.980–0.996 for ly05's other recdays).

**Found along the way, unrelated:** five `Structure` entries in `MetaData-ly06.csv` disagree
with the pyControl files' own `active_poke` (four sessions of `ly06_20250616_20250617`, one
of `ly06_20250618_20250619`). The data is right and the spreadsheet is wrong, which is why
the behavioural guard asserts against pyControl and only *reports* metadata disagreement —
`recday_registry.metadata_task_discrepancies()`.

---

## 9. The guards

Two independent assertions, either of which would have caught this at extraction time. Both
run inside `glm_analysis_v2.load_data_dic` and `preprocessing/build_data_dic.py`:

```python
import recday_registry as rr
rr.validate_data_dic(dd)                 # neural:     rows == QC units of the named block
rr.validate_tasks_against_pycontrol(dd)  # behaviour:  Task == that day's own active_poke
```

The first is the direct check: unit count is the only observable that distinguishes one
day's spikes from another's once the data is on the behavioural timeline. The second is the
generalisation of §2.1 to the whole cohort.

## 10. Reproducing the original diagnosis

```python
import numpy as np
n = np.load('data/processed_data/neuron_raw_mingyutest/'
            'Neuron_raw_ly05_20250618_20250619_0.npy', mmap_mode='r').shape[0]
q18 = len(np.load('data/preprocessed/ephys/ly05/'
                  '2025-06-18_2025-06-19_preprocessed/kilosort_output/QC_single_units.npy'))
q20 = len(np.load('data/preprocessed/ephys/ly05/'
                  '2025-06-20_2025-06-23_preprocessed/kilosort_output/QC_single_units.npy'))
print(n, q18, q20)      # was 91 109 91  (named 18-19, contained 20-23)
                        # now 109 109 91
```
