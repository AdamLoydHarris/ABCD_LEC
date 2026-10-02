# Spike clock v4: the OpenEphys counter restart, the fix, and how to process new ephys

**Written 2026-10-01.** This replaces the root cause and the proposed fix in
`SPIKE_CLOCK_V2_MISALIGNMENT.md`. That document's symptom measurements stand.

## Outcome
- **The bug.** v2's fitted spike clock puts spikes 0.94-29.2 s late (median 4.9 s) in 42 of 211
  LEC task sessions. It is exact in the other 169.
- **The cause.** OpenEphys 0.6.6 sometimes writes the first ~25 ms of a recording with a stale
  sample counter, then restarts the count near 0. The TTL events carry the restarted count.
  `extract_rsync_from_ttl.py` subtracted the stale first value, which shifted every pyControl
  event early and every spike late.
- **Neither simple rule is right.** "Subtract the first value" (v2) is wrong on every restart
  recording. "Subtract nothing" (the earlier handoff's fix) leaves 24-50 ms there and is 7.7 s
  wrong on a recording whose count never restarts. The origin must come from the counter's
  final contiguous run.
- **v4** is v2 with only the origin fixed: `ephys_rsync_v4/`, `neuron_raw_v4/`,
  `data_dic_lec_v4.pkl` (25 recdays), and the same in the mEC tree. `data_dic_mec_v4.pkl` holds
  34 of 36 recdays; the next bullet explains the two left out.
  - v4 equals v2 plus the origin shift exactly, on both datasets.
  - Every session v4 changed passes every behaviour probe it can be tested on, with two mEC
    exceptions, both explained in section 2.4:
    - bp02 03-24 s0 fails only in the merged sort; its single-day twin passes.
    - bp02 03-25 10:34 has a speed lag 0.55 s off, while its reward peak sits at 0.00 s
      (peak/median 8.7).
  - The validator now gates on speed, reward and wall-clock probes, not just pulse pairing.
  - Results: section 2.4.
- **A second, separate fault, found by the new gates: two mEC sorts hold the wrong data.**
  `data/raw/ephys/2024-03-19_11-58-14/` contains a nested copy of the 10-44-12 recording, and the
  sorter read the copy.
  - In bp02_20240319 and bp02_20240319_20240321, 10-44-12's data appears twice and the real
    11-58-14 data is missing.
  - Every later recording is 14.42 s late (432,672 samples). That is v1, v2 and v4 alike; it is
    not a clock bug.
  - Gate 1 (binary length against the manifest) fails on exactly those 2 of 36 mEC blocks. A
    scan finds no other nested recording in either tree.
  - `data_dic_mec_v4.pkl` is built from the other 34 blocks. The repair is open: see section 4.
- **mEC has no unaffected version.** `data_dic_mec.pkl`, `_v2` and `_v3` all carry the bugged
  trains.
- **v1 (`data_dic_lec.pkl`, used by every existing result) is not fixed here.**
  ly05_20250620_20250623 is off by 2.6-38.6 s in all 8 sessions. Restart sessions in ah10 and
  ly07 are 24-26 ms late. User decision 2026-10-01: document, do not exclude; revisit when
  analyses move to v4.
  - Cause, established 2026-10-02: each of that recday's 8 legacy rsync caches is the TTL train
    of the 06-18/19 recording in the same session slot. It is the documented ly05 position-based
    pairing bug, never fixed for the caches. See section 1.5.

## 1. Problem

### 1.1 What OpenEphys writes
ah10 2025-06-20_16-37-32, ProbeB stream:

| item | value |
|---|---|
| continuous samples (`.dat` length) | 36,520,742 |
| `sample_numbers` at `.dat` indices 0-727 | 578,636 .. 579,363 |
| `sample_numbers` at index 728 onward | 8, 9, ... 36,520,021, contiguous to the end |
| `sync_messages.txt` "Start Time" for ProbeB | 578,636, the stale value |
| first two TTL rising edges | 314,674 and 316,188, both below the first value |
| OpenEphys synced-timestamp offset, continuous and TTL | 10.4891 s for both |

The TTL count is the restarted one, so the `.dat` index of a TTL edge is `TTL - origin`. Here
`origin = sn[j] - j = 8 - 728 = -720`.

### 1.2 Three kinds of recording
All 758 LEC probe-recordings, June and July, surveyed 2026-10-01:

| kind | count | origin |
|---|---|---|
| count starts at 0, no restart | 658 | 0 |
| count restarts 716-2932 samples in | 95 (53 sleep, 42 task) | -716 .. -764 in 94; -1488 in ly06 2025-06-24_13-19-07 |
| first sample > 0 with a continuous count | 5, all one sleep session (2025-06-19_11-45-22) | the first sample, 231,655-238,051 |

- Every June task recording whose first sample is above 0 has the restart: 42 of 42.
- All 268 July recordings start at sample 0.
- Each probe stream restarts on its own schedule. ly05 and ly06 in the same 06-24_13-19-07
  session restart at index 2200 and 1496 respectively. The origin is therefore computed per
  stream, never per session.

### 1.3 Why each simple rule fails

| rule | restart recordings (95) | continuous recording (5) |
|---|---|---|
| subtract the first value (v2) | spikes late by (sn0 - origin) / 30 kHz: 0.94-29.2 s in task sessions | right |
| subtract nothing (earlier handoff) | spikes late by -origin / 30 kHz: 24-50 ms | spikes 7.7-7.9 s early |
| `origin = sn[j] - j` over the final contiguous run (v4) | right | right |

### 1.4 Evidence that the TTL count is the restarted one
- 14 task recordings have TTL edges numbered below the first value. A continuous count cannot
  produce that.
- OpenEphys' own synced timestamps map TTL and continuous sample numbers with the same offset,
  10.4891 s on the example, agreeing to 0.1 ms.
- On the v4 origin, every TTL edge lands inside `[0, n)`: 723 of 723 recordings with edges. On
  v2's origin, 27 recordings have edges before the recording began.
- Behaviour agrees at the probe's resolution. On the speed probe, v2's lag minus
  (sn0 - origin)/30 kHz recovers v1's lag to within 0.05 s in every session with r > 0.2:

  | session | r | v1 | v2 | v2 shifted |
  |---|---|---|---|---|
  | ah10 s3 06-20_16-37-32 | 0.53 | -0.28 s | +19.03 s | -0.29 s |
  | ah10 s6 06-23_16-20-49 | 0.63 | -0.35 s | +28.78 s | -0.38 s |
  | ly07 s2 06-24_14-48-56 | 0.22 | -0.30 s | +10.12 s | -0.31 s |
  | ah08 s9 06-25_15-00-21 | 0.28 | +0.40 s | +20.90 s | +0.37 s |

- The spike side was checked too. Each sorted spikeinterface binary is every recording's
  `continuous.dat` back to back, stale samples included:
  - LEC ah10 06-20/23: 381 ch x 704,160,309 samples;
  - mEC bp01 03-24/25: 384 ch x 791,763,044 samples;
  - both equal sum(dat_size)/768 exactly. Gate 1 of `spike_clock_v4_gates.py` now checks every
    block.
- mEC has the same restart. bp01 2024-03-24_10-52-32 counts 603,172..604,163 for 992 samples,
  then restarts at 480, so the origin is -512. That is why the old docstring said its TTL
  edges sat "~136k samples before the AP stream".

### 1.5 Clock error by version, LEC task sessions
Positive means the version's spikes are later than the corrected clock's.

| mouse | v1, sessions without restart | v1, restart sessions | v2, restart sessions | n without / with restart |
|---|---|---|---|---|
| ah08 | -121 .. +38 ms | +54 .. +59 ms | +3.61 .. +20.53 s, median 10.86 | 28 / 6 |
| ah10 | -0.5 .. +0.5 ms | +24 .. +26 ms | +2.56 .. +29.16 s, median 4.49 | 24 / 11 |
| ly05 | median +204 ms, up to +38.6 s | -10.0 .. +10.3 s | +1.01 .. +22.13 s, median 4.76 | 32 / 10 |
| ly06 | +61 .. +171 ms | +86 .. +114 ms | +0.94 .. +21.93 s, median 5.04 | 31 / 11 |
| ly07 | -0.4 .. +0.7 ms | +25 .. +26 ms | +2.93 .. +13.86 s, median 7.09 | 38 / 4 |

- v2 is exact in every session without a restart. In the 42 restart sessions it is late by
  0.94-29.16 s, median 4.91 s.
- **Where v1's errors come from** (established 2026-10-02 by matching every legacy cache against
  every raw TTL stream):
  - All 195 legacy caches are raw **ProbeB** TTL trains, whatever the mouse's own probe, and the
    first pulse is taken as pyControl's first rsync. That gives the 30-460 ms terms: ah08 ProbeD,
    ly05 ProbeA, ly06 ProbeC with the partner's first pulse, ly07 ProbeC under 1 ms.
  - The 8 ly05_20250620_20250623 caches are the ProbeB trains of the 06-18/19 recordings in the
    same session slots, every pulse at offset 0. That is the position-based pairing bug of
    `docs/BUG_ly05_recday_mismatch.md`, never fixed for the caches (addendum §11 there).
  - v2's per-mouse pairing removed all of these.
- 16 task recordings have no legacy cache: ah08 06-18/19 and ah10 06-16/17.
- v1 errors on ly05_20250620_20250623, s0-s7: +5.33, +2.55, +10.37, +10.26, -9.98, +38.58,
  +8.20, +5.71 s.
- Two other ly05 sessions are off in v1: 06-13_14-07-42 by +0.31 s and 06-19_16-57-48 by
  +0.46 s.

**Affected:**
- LEC: the task arrays in `neuron_raw_v2/`, and `data_dic_lec_v2.pkl` / `_v3.pkl` with their
  `norm_neurons_dic_*`. No analysis has been run on them (TRACKING_QC.md:517).
- mEC: every spike array and pickle. Its `neuron_raw_mingyutest` and `neuron_raw_v2` were both
  built from the v2 trains.

**Not affected:** trial times, pokes, tracking (v2 and v3), and the `_sb_` rest arrays, which
use origin 0 and no pyControl clock.

## 2. Solution

### 2.1 The origin rule
`extract_rsync_from_ttl.counter_origin(sample_numbers_path)` does the following:
- scans the continuous `sample_numbers.npy` in 10 M-sample chunks for `diff != 1`;
- takes the final contiguous run `j..n-1`, and returns `origin = sn[j] - j`;
- raises if that run starts more than 30,000 samples (1 s) in, because the TTL count could then
  be either one.

TTL edges become `.dat` indices as `TTL - origin`. Pairing (`match_trains`) is unchanged; it is
shift-invariant. So v4 = v2 + (sn0 - origin) exactly, and v4 = v2 wherever origin = sn0.

### 2.2 Code changes

| file | change |
|---|---|
| `code/preprocessing/extract_rsync_from_ttl.py` | `counter_origin`; `--origin {counter,first_sample}`, default `counter`; `--out`, default `ephys_rsync_v4`, with QC in `<out>_qc.csv`; `--compare-to` for read-only equivalence; `process_row` split out of `process_recday`; a missing pyControl file is a per-recording error; new QC columns `origin`, `restart_at`, `n_jumps`, `ttl_outside_dat`, `ttl_in_stale_window` |
| `code/preprocessing/extract_neuron_raw.py` | `--rsync <tree>`, read before import like `--out`; default unchanged |
| `code/preprocessing/validate_alignment.py` | speed, reward, wall-clock and early-start probes as hard gates; per-mouse speed references, with testability r > 0.2 and peak ratio >= 1.2; reward window [-2, +4] s; wall-clock bands per mouse and pyControl format; `.tsv` stamps read as UTC; `--rsync <tree>`; the ±2 s speed lag is replaced |
| `code/preprocessing/spike_clock_synthetics.py` | new, 47 checks, `--quick` |
| `code/preprocessing/spike_clock_v4_gates.py` | new: v4 against v2, and binary length against the manifest |

The default changed (user decision 2026-10-01): a run with no flags now writes the corrected
trains into a new tree. v2 is still reproducible with `--origin first_sample`.

### 2.3 Synthetic controls (`spike_clock_synthetics.py --quick`: 47 pass, 0 fail)

| control | measured |
|---|---|
| origin on planted counts | -720, -748, 0, 231,685, as planted; a jump on a chunk seam is still found |
| guards | a jump 40,000 samples in, an empty file and an unknown mode all raise |
| real door, `counter` | 5000/5000 bins identical to the planted spikes in all three classes; 131/131 pulses matched with a partner box's train interleaved |
| `first_sample`, restart class | spikes late by 19,312.18 ms, exactly (sn0 - origin)/a; 23 TTL edges before index 0 |
| `first_sample`, other classes | identical to `counter` |
| subtract nothing | +24.000 ms on the restart class, -7,722.96 ms on the continuous class |
| speed probe | -0.300 s and +0.350 s recovered, peak ratio 3.6; delays of 19.31 s and 1 s seen and gated; Poisson noise r = 0.06; units sharing only a slow trend with speed give r 0.31 and peak ratio 1.04, so untested |
| reward probe | +0.25 s recovered, peak/median 15.8; a 19.31 s delay moves the peak to +19.5 s; noise 1.45 |
| wall clock | right clock -0.40 s; v2's origin +18.91 s; `.tsv` stamps read as UTC in March and April; `.txt` local time on both sides of the 2024-03-31 DST change |
| per-mouse gates | references -0.31 s and +0.34 s; delayed sessions fail as designed |
| reference fallback, second rig | a mouse with 2 clean sessions takes the cohort's -0.31 s, and its 4 delayed sessions still fail; a rig whose PC clock sits +3.5 s away passes on its own band, and a 19.3 s error on it fails |
| gate 5 lag search | a true 772.49-bin shift is found at 772, with overlap r 0.997; v2 against itself is rejected at lag 0 |

Two flaws in my first draft were caught by in-memory smoke tests on real data before any job ran:
- **Reference fallback.** With fewer than 3 clean sessions the speed reference fell back to every
  session, delayed ones included. On a 2-recday subset that put ah10's reference at +4.55 s and
  ah08's at +7.22 s. The fallback now takes the cohort's clean sessions only.
- **Gate 5 lag window.** Searching ±100 bins around the expected lag let v2-against-itself pass
  1 of 7 restart arrays on a local maximum. The search now covers every lag from 0 to past the
  expected one, and requires r >= 0.8.

Both fixes have synthetic controls, 10b and 11.

### 2.4 Results of the v4 run (2026-10-01/02)
All jobs ran through `sbatch_files/mec_stage.sbatch`. Logs are
`sbatch_files/slurm_outputs/mec-<job name>-<id>.out`, and each ends in `EXIT=`.

**LEC, the 25 June recdays**

| step | job | result |
|---|---|---|
| equivalence: `--origin first_sample --compare-to ephys_rsync` | 3672248 | 490/490 TTL trains, 490/490 rsync trains and 211/211 fits byte-identical to v2 |
| v4 trains | 3672249 | 95/490 recordings restart, 42 of them task; origins -1488 .. -716, median -742; 5 continuous recordings with a first sample > 0 (sleep, 06-19_11-45-22); 0 TTL edges outside the `.dat`; task recordings with pulses before sample 0: 0, against 14 in v2; pairing unchanged, 211/211, residual median 0.045 ms |
| v4 arrays (`neuron_raw_v4/`, 5.1 GB) | 3672250 | 490 written; lag v4 minus v1 (bins, min / median / max): ah10 -1 / 0 / 0, ly07 -1 / 0 / 0, ly06 -6 / -2 / -2, ah08 -2 / +2 / +7, ly05 -39 / -8 / +37 |
| gates | 3672251; 3673397 re-run with overlap r | binary length 25/25 blocks; trains and fits 490/490, 0 samples off by one, max b error 4.5e-9 samples; 448/448 arrays without a shift byte-identical; 42/42 restart arrays at the expected lag within ±0.49 bins, r 0.981-0.994 |
| pickle `data_dic_lec_v4.pkl` (3.82 GB) | 3672252 | registry guards pass on 25 recdays and 191 sessions; differs from v3 only in the 10 recdays with restart sessions |
| check against v3 | 3672256 | `Neuron_raw` identical in 169/211; tracking shift 0.0 cm; trial times identical 191/191 |

**Behaviour probes, LEC**, failed / tested, under the final testability rule:

The 211 sessions split three ways: A is the 38 restart sessions outside ly05_20250620_20250623,
B is the 8 sessions of ly05_20250620_20250623 (4 of them restart), and C is the other 165.

| sessions | probe | v1 | v2 | v4 |
|---|---|---|---|---|
| A, restart (38) | speed | 0 / 23 | 23 / 23 | 0 / 23 |
| A | reward | 0 / 24 | 15 / 24 | 0 / 24 |
| A | wall clock | 0 / 38 | 22 / 38 | 0 / 38 |
| A | early start | 0 / 0 | 8 / 8 | 0 / 0 |
| B, ly05_20250620_20250623 (8) | reward | 3 / 3 | 2 / 3 | 0 / 3 |
| B | wall clock | 7 / 8 | 4 / 8 | 0 / 8 |
| B | early start | 0 / 0 | 1 / 1 | 0 / 0 |
| C, the rest (165) | speed | 3 / 56 | 3 / 56 | 3 / 56 |
| C | reward | 0 / 123 | 1 / 122 | 1 / 122 |
| C | wall clock | 0 / 149 | 0 / 165 | 0 / 165 |

v1 has no legacy cache for 16 recordings, ah08 06-18/19 and ah10 06-16/17, so its wall clock is
tested on fewer sessions.

The v1 and v2 rows are for reading the table. v1 passes the restart sessions because it is only 24-26 ms late there, which is inside every probe's tolerance.

**v4 passes every probe on every session it changed.** Four failures remain, all in sessions v4 did
not change, which are byte-identical in v2. Jobs: 3673421-3673423, then 3673462-3673464 under
the final probe code (`.tsv` UTC, per-format wall-clock bands, reward window [-2, +4] s), with
identical LEC numbers:
- **3 speed lags 0.6-0.9 s from the mouse's reference:** ah08_20250620_20250623 s0,
  ly06_20250616_20250617 s1 and ly06_20250624_20250625 s4.
  - The same 3 sessions fail the same way in v1. v1 uses a different, legacy anchoring, so a
    shared clock error would need two independent clocks to agree; the failures belong to the
    sessions, not to a clock.
  - The reward response peaks at 0.00 s in the two sessions that have rewards.
- **1 reward peak at the window edge:** -28.75 s, in ly05_20250616_20250617 s4. It has 10 rewards
  and a peak/median of 2.06, just over the testability floor of 2.

The validator therefore still exits 1 on v4 LEC. The four are recorded here, not waived in code.

**Testability, revised after the first v4 run** (jobs 3672253, 3672254, 3672255, 2026-10-02). A
speed lag now needs r > 0.2 *and* a peak that beats every value more than 2 s away by 20%
(`SPEED_PEAK_RATIO_MIN = 1.2`). The first run, on r alone, failed 12 of ly05's 15 testable
sessions. Their lags scattered from -70 s to +40 s, and 11 of the 12 are sessions v4 left
unchanged. Their peak ratios were 1.00-1.14, while 95% of all other sessions are at 1.24 or above.
The rule treats every clock version identically, and v2's restart sessions still fail 23/23 under
it. Synthetic control 7 now includes a session whose units share only a slow trend with speed:
r 0.31, peak ratio 1.04, lag +3.9 s, untested.

**mEC, all 36 recdays** (12 merged, 24 single-day sorts of the same recordings)

| step | job | result |
|---|---|---|
| equivalence | 3672593 | 225/225 TTL trains, 225/225 rsync trains, 108/108 fits byte-identical. 4 bp01 2024-03-21 fits on disk are `stale`: left by v2's first run, which mapped bp01's 12:59 pyControl file to a sleep recording. The first attempt, 3672257, counted those 4 as differences. |
| v4 trains | 3672594 | 115/225 recordings restart, 58 of them task; origins -768 .. -512; no continuous first sample > 0; 0 TTL edges outside the `.dat`; pairing unchanged, 108/108. v2 was late in those 58 by 1.19-67.32 s, median 3.42 (bp01 18 sessions, bp02 24, bp04 16). |
| v4 arrays (994 MB) | 3672595 | 450 written |
| gates, all 36 blocks | 3672596 | trains 225/225; arrays 450/450, 116 restart arrays at the expected lag, r >= 0.984. **Gate 1 fails on 2 of 36 blocks:** bp02_20240319 and bp02_20240319_20240321 binaries are 432,672 samples (14.42 s) longer than their manifests. |
| gates, 34 blocks | 3673456 | all pass |
| pickle `data_dic_mec_v4.pkl` (1.12 GB) | 3673457 | built from the 34 sound blocks; registry guards pass on 34 recdays and 182 sessions |
| check against v3 | 3673458, 3673459 | merged 11 recdays / 99 sessions, single-day 23 / 104; `Neuron_raw` identical in 43 and 48, and the 112 that differ are the restart arrays the gates verified; trial times identical 89/89 and 93/93; tracking unchanged |

**The bp02 2024-03-19 corruption.**
- `data/raw/ephys/2024-03-19_11-58-14/` holds a nested folder, `2024-03-19_10-44-12/...`.
- The nested ProbeB-AP `continuous.dat` has 36,860,268 samples, the outer one 36,427,596. The
  difference is the binary's excess to the sample.
- So in both bp02 03-19 sorts, the 11-58-14 slot holds 10-44-12's data and the real 11-58-14
  data is absent. Every later recording, s2 onward, is 14.42 s late.
- Behaviour agrees. bp02 03-19 sessions s2-s8 sit at a speed lag of +14.28 to +14.45 s and a
  reward peak of +14.0 to +14.75 s, against bp02's -0.15 s reference. Their wall clock is
  normal, because the rsync side is fine.
- This is not a clock bug, and v1, v2 and v4 all carry it.
- A scan of both raw trees finds no other nested recording.

**Behaviour probes, mEC**, failed / tested, as session rows. Merged and single-day sorts list the
same recordings twice.

| rows | probe | v2 | v4 |
|---|---|---|---|
| restart rows in sound blocks (110) | speed | 83 / 83 | 2 / 83 |
| | reward | 29 / 68 | 1 / 73 |
| | wall clock | 52 / 110 | 0 / 110 |
| | early start | 4 / 4 | 0 / 0 |
| corrupted bp02 03-19 rows (13) | speed / reward | 6 / 8, 8 / 12 | 6 / 8, 8 / 12 |
| other rows (91) | speed / reward / wall clock | 4 / 70, 2 / 67, 0 / 91 | 4 / 70, 2 / 67, 0 / 91 |

The 9 v4 failures outside the corrupted blocks are about 7 recordings. Each has a reason it is not
a clock error. In every case, gate 1 and a normal wall clock together bound any clock error below
about a second:
- **Low-distinctness speed peaks** (ratio 1.26-1.32) in sessions with 0 rewards: bp02 03-24 s1,
  bp02 04-03 s3 in both sorts, and bp04 03-27 s3.
- **One speed lag 0.55 s off** (bp02 03-25 s1, in both sorts), in a session with 131 rewards.
- **A merged-sort population that barely tracks behaviour** (speed r 0.02): bp02 03-24 s0. The
  same recording in the single-day sort passes at +0.10 s speed and -0.25 s reward.
- **Reward PSTHs aliased by regular rewards:** bp02_20240321 s3 peaks at -13.25 s. The same
  recording in the corrupted merged sort peaks at +14.00 s, which puts the true response at
  -0.42 s.

**Probe calibration, fixed after the first mEC run** (2026-10-02). The rules are dataset-neutral
and pinned by synthetic controls 9 and 10b.
- **`.tsv` stamps are UTC.** Read as London time, all 8 April `.tsv` sessions sat at
  -3599.3 to -3599.4 s, and the March ones did not.
- **Wall-clock bands are per pyControl format.** The open-field `.tsv` rig's PC clock runs about
  3.5 s from the maze `.txt` PC: +1.4 to +1.9 s against -1.7 s in March.
- **The reward window opens to -2 s.** Eight mEC sessions whose speed probe passes peak at -1.25
  to -1.5 s, before the rewarded entry.

**bp02 2024-03-19 repair, 2026-10-02.**
- **"Missing" means missing from the sort, not from the raw data.** The real 11-58-14 recording is
  intact in the outer folder: ProbeB-AP has 36,427,596 samples.
- **Why the sorter read the copy.** `merge_preprocess_kilosort_spikeinterface_mEC.py` passes the
  recording folder to `se.read_openephys`. neo's `explore_folder` walks the whole tree. The nested
  `Record Node 101/experiment1/recording1` has the same keys as the outer one, is walked second,
  and replaces it silently. The manifest is copied from the metadata CSV, so it kept the outer
  path and size.
- **Done:**

  | step | result |
  |---|---|
  | full `cmp` of the nested tree against `2024-03-19_10-44-12` (job 3674377) | 34 files, 62 GB, 0 differ: a pure duplicate |
  | guard, before the move | flags 11-58-14: loaded 36,860,268 samples against 36,427,596 from `dat_size`, and a nested folder |
  | move | the copy now sits at `M/data/raw/duplicates/2024-03-19_11-58-14__nested_copy_of_2024-03-19_10-44-12/`, kept with a README; the nesting scan prints nothing in either tree |
  | guard, after the move | 9/9 bp02 03-19 recordings load exactly their `dat_size` |
  | old blocks | renamed `<block>.corrupt_nested_copy_2026-10-02`, outside the registry glob; about 815 GB kept. Their `params.py` still holds an absolute `dat_path`, which now names the new block's binary |
  | stale arrays | the 27 `Neuron_raw_bp02_20240319*` arrays in each of `M/data/processed_data/neuron_raw_{mingyutest,v2,v4}` renamed `*.npy.INVALID_nested_copy_2026-10-02` |

- **Sorter guard,** in both `L/code/merge_preprocess_kilosort_spikeinterface.py` and
  `M/code/merge_preprocess_kilosort_spikeinterface_mEC.py`: `check_loaded_recording`, called right
  after each `read_openephys`. It raises on a nested recording folder, or when the loaded sample
  count differs from `dat_size // 768`. It only raises; outputs are unchanged.
- **Re-sorts submitted** with the original script and thresholds (`bp02 <dates> 9 8`):
  - `maze_ephys_np126`, `--gres gpu:l40s:1`, 32 CPUs, 256G;
  - `M/sbatch_files/fixed/bp02_singledays/merge_bp02_2024-03-19_resort_2026-10.sbatch` (job 3674427);
  - `M/sbatch_files/fixed/merge_bp02_2024-03-19_to_2024-03-21_resort_2026-10.sbatch` (job 3674428);
  - chained after them: v4 arrays (3674429), then gates (3674430).
  - Results: filled in below when the jobs land.

## 3. Processing new ephys (July ABCDE and later)
Run in this order. Each step names its output and the check that must pass before the next.
Heavy steps go through `sbatch sbatch_files/mec_stage.sbatch <script> ...`, which runs from
`code/preprocessing/` in `maze_ephys` and prints `EXIT=` at the end. Read logs to that line.

1. **Sort**, in env `maze_ephys_np126` (numpy 1.26.4). numpy 2 breaks KS4 4.0.18 and
   spikeinterface 0.101.2; see MEC_INTEGRATION §8.6.
   - Before sorting, `find data/raw/ephys -mindepth 2 -maxdepth 2 -type d -name "20*"` must print
     nothing. A recording folder nested inside another one gets read in place of the outer
     recording. That happened at mEC 2024-03-19_11-58-14, which cost two sorts, and in the July
     2025-07-19_14-51-59 drag-drop.
   - Command: `sbatch sbatch_files/merge_<mouse>_<dates>.sbatch` for a new block, copied from
     `merge_ah10_2025-07-14_to_2025-07-19.sbatch`; or `resumeks_<mouse>_<dates>.sbatch` to resume
     from an existing destriped binary with `code/resume_kilosort_from_binary.py`.
   - Before any re-run, check that the output block path is absent. The sorter appends `_2`
     instead of overwriting, and `recday_registry` ignores `*_preprocessed_2`.
   - Pin the GPU type: `--gres gpu:a100:1` or `gpu:l40s:1`. A 6-day block needs about 20.4 GB
     of VRAM in final clustering, and the 19.6 GB A4500 died there.
   - Time: 5 days of walltime; a 6-day block takes 45-60 h on an A100 or L40S.
   - After any environment change, run a smoke test first:
     `resume_kilosort_from_binary.py --smoke-minutes 20`.
   - Check: Kilosort's progress is in the `.err`. Then compare `QC_single_units.npy` counts
     (`rr.qc_unit_count`) with the June blocks.
2. **Spike timebase.** `spike_clock_v4_gates.py --dataset lec --recdays <new>`: gate 1 must
   pass, binary samples == sum(dat_size)/768. Gates 2-5 compare against v2 and read `no v2` for
   new blocks, which is expected.
3. **Rsync trains.** `extract_rsync_from_ttl.py --dataset lec --recdays <new> --write`, which
   writes `ephys_rsync_v4/` and a new `ephys_rsync_v4_qc*.csv`.
   - The printed summary must show matched fraction about 1.0, residual 0.08 ms or less, and
     `TTL edges outside the .dat: 0`.
   - A `ValueError: sample-number jump ... past the first 30000 samples` is a new OpenEphys
     pattern: stop and inspect.
   - July: every recording starts at sample 0, so the origin is 0 throughout.
   - ah10 2025-07-18 has ephys but no pyControl file; it is reported as a task row with an
     error and no fit.
4. **Trial times and pokes.** `python extract_pokes.py --dataset lec --write-trialtimes --validate`
   creates `trialtimes_` and `Task_data_` for new sessions, then `pokes_`.
   - On one ABCDE session, confirm that trialtimes has 6 columns, that Task_data has 5 goals,
     and that `--validate` exits 0.
   - The state-id map is read per file. ABCDE_v1 and METraining_Task_v1 number the same states
     differently.
5. **Tracking** on the frame clock into `processed_v3/`.
   - ah10's July recday was done 2026-09-28 with
     `sleap_alignment_copilot.py --animals ah10 --ephys_folder 2025-07-14_2025-07-15 --tracking-clean v2 --out-tag v3`
     (TRACKING_QC.md). Do the same for the other mice.
   - Check `sync_qc_lec.v*.csv`: every pulse anchored, 0 uncertain bins.
6. **Spike arrays.**
   `extract_neuron_raw.py --dataset lec --recdays <new> --sync fitted --rsync ephys_rsync_v4 --out neuron_raw_v4 --write`.
   - Both `--sync fitted` and `--rsync ephys_rsync_v4` are required. The defaults are the v1
     and v2 clocks.
7. **Code that must change before a July pickle**, each with its own equivalence control: the
   June part of the new build must be byte-identical to `data_dic_lec_v4.pkl`.
   - `build_data_dic.py:65` `MAX_SESSIONS = 12`: the 6-day block has 24 task sessions.
   - `build_data_dic.py:67` `NUM_STATES = 4`: must come per session from `session_type()`,
     since ABCDE has 5 states.
   - `validate_alignment.py` `check_recdays` loops `range(12)` and must cover 24.
8. **Pickle.**
   `build_data_dic.py --dataset lec --neural v4 --tracking v3 --tag v5 --diff-against data_dic_lec_v4.pkl`.
   The tag `v3` is taken by the tracking-v3 pickle. MEC_INTEGRATION §8.5's plan to call the
   July pickle `v3` predates that.
9. **Gates.**
   - `validate_alignment.py --dataset lec --neural v4 --tracking v3 --rsync ephys_rsync_v4` must
     exit 0. Every task session must pass every probe it can be tested on.
   - Then `check_data_dic_v2.py --dataset lec --version v5 --against data_dic_lec_v4.pkl`.

**Rules that keep this from recurring:**
- Never take the `.dat` origin from `sample_numbers[0]` or `sync_messages.txt`. Use
  `counter_origin`.
- Pairing QC is not alignment. v2 passed every pairing and shape check while 42 sessions were
  seconds late. A new spike clock is trusted only after the behaviour probes pass.
- Lag searches must be wider than the error they look for. The v2 comparisons searched ±1 s and
  ±2 s and called the sessions at the limit "noise".

## 4. Not established
- Why OpenEphys restarts the count: operator sequence or plugin behaviour. Seen in June LEC and
  in mEC; not in July.
- The 24-50 ms restart term is below the behaviour probes' resolution, which is about 0.3 s. It
  rests on the sample-number logic and OpenEphys' own synced timestamps.
- ah08's speed reference is +0.25 to +0.40 s, against ah10's -0.30 s. Whether that is
  physiology or a residual clock offset is open; the reward probe is the absolute check.
- No v1-based result has been re-run (user decision).
- **The bp02 2024-03-19 repair: re-sort chosen** (user, 2026-10-02). It is in progress; see the
  dated paragraph at the end of section 2.4.
- Why the reward and speed probes fail in about 7 mEC and 4 LEC sessions that v4 did not
  change. The candidates are probe noise, behaviour and weak populations; none is a measured
  clock error.

## 5. Reproduce
Job ids are in section 2.4.

```
python code/preprocessing/spike_clock_synthetics.py --quick                       # 47 checks
S=sbatch_files/mec_stage.sbatch; JUNE="<the 25 June LEC recdays>"
sbatch $S extract_rsync_from_ttl.py --dataset lec --recdays $JUNE --origin first_sample --compare-to ephys_rsync
sbatch $S extract_rsync_from_ttl.py --dataset lec --recdays $JUNE --write
sbatch $S extract_neuron_raw.py --dataset lec --recdays $JUNE --sync fitted --rsync ephys_rsync_v4 --out neuron_raw_v4 --write
sbatch $S spike_clock_v4_gates.py --dataset lec --recdays $JUNE
sbatch $S build_data_dic.py --dataset lec --neural v4 --tracking v3 --tag v4 --diff-against data_dic_lec_v3.pkl
sbatch $S validate_alignment.py --dataset lec --recdays $JUNE --neural v4 --tracking v3 --rsync ephys_rsync_v4
sbatch $S validate_alignment.py --dataset lec --recdays $JUNE --neural v2 --tracking v3 --rsync ephys_rsync
sbatch $S validate_alignment.py --dataset lec --recdays $JUNE --neural mingyutest --tracking v3 --rsync ephys_rsync_mingyutest
sbatch $S check_data_dic_v2.py --dataset lec --version v4 --against data_dic_lec_v3.pkl
# mEC: the same with --dataset mec and no v1 run; gates and build on the 34 sound blocks
MEC34="<all 36 mEC recdays except bp02_20240319 and bp02_20240319_20240321>"
sbatch $S spike_clock_v4_gates.py --dataset mec --recdays $MEC34
sbatch $S build_data_dic.py --dataset mec --recdays $MEC34 --neural v4 --tracking v3 --tag v4 --diff-against data_dic_mec_v3.pkl
sbatch $S check_data_dic_v2.py --dataset mec --version v4 --recday-kind merged --against data_dic_mec_v3.pkl   # and --recday-kind single
```
