# Spike-clock misalignment in `neuron_raw_v2` (LEC) and the mEC v2 pipeline — handoff

**Written 2026-10-01.** Found while planning the event-kernel (Tsao / Kanter) analysis. Nothing
was changed on disk; every number below came from read-only checks on small per-session files.

**Correction, 2026-10-01 (later the same day): see `SPIKE_CLOCK_V4.md`.** The symptom and the
guilty line stand. The mechanism and the proposed fix do not.
- **Mechanism.** The TTL numbers are not "already on the `.dat` timebase". OpenEphys 0.6.6
  writes the first 716-2932 samples of these recordings with a stale counter, then restarts the
  count near 0, and the TTL events carry the restarted count. ah10 2025-06-20_16-37-32: sample
  numbers 578,636..579,363, then 8 from `.dat` index 728.
- **The fix proposed below is wrong.**
  - "Drop only the subtraction" leaves 24-50 ms in all 42 restart task sessions.
  - It would also shift the one LEC recording with a genuinely continuous count by 7.7-7.9 s:
    2025-06-19_11-45-22, a sleep session.
  - v4 uses `origin = sn[j] - j` over the counter's final contiguous run.
- **The gate "v4 equals v2 shifted by exactly ap_first / 30 kHz" becomes (ap_first - origin).**
- **mEC.** `data_dic_mec.pkl` is affected too, not only `_v2` and `_v3`. The mEC legacy arrays
  were built from the same trains.
- **The speed-probe reference is per mouse.** ah08 sits at +0.25 to +0.40 s on a correct clock.

## Outcome
- **LEC v2 is shifted.** The v2 fitted spike clock moves spikes 2-29 s late relative to behaviour
  in 42 of 211 LEC task recordings. Both `data_dic_lec_v2.pkl` and `data_dic_lec_v3.pkl` use it.
  Any behaviour-locked analysis on those sessions would be wrong. Nothing has been run on v2 or
  v3 yet.
- **The shift has one size.** In every shifted recording it equals the AP stream's first sample
  number divided by 30 kHz. Recordings whose AP stream starts at sample 0 are unaffected.
- **Likely bug:** `extract_rsync_from_ttl.ttl_rising_relative` subtracts that first sample number
  from the TTL sample numbers.
- **mEC is probably hit too.** It uses the same code, and 57 of 107 mEC task recordings have a
  non-zero AP start (median 3.6 s, max 67.3 s). So `data_dic_mec_v2.pkl` and
  `data_dic_mec_v3.pkl` are probably shifted.
- **v1 is aligned except one recday.** On `ly05_20250620_20250623` all 8 sessions are off by
  about 3-39 s, because that recday's legacy rsync cache is wrong. Existing results use v1, so
  they include this.

## Probes used
- **Speed probe.** Cross-correlate the population rate with running speed.
  - Population rate: units summed, 25 ms bins, Gaussian sigma = 10 bins.
  - Speed: from `data/processed_v3/<mouse>/<block>/XY_raw_<rd>_<s>.npy`, smoothed sigma = 3 bins,
    then the gradient.
  - v3 tracking reaches the pyControl clock through the frame clock, independent of the ephys
    rsync fit.
  - Reference: in 33 sessions where v1 equals v2 and r > 0.3, population leads speed by 0.30 s
    (IQR 0.25-0.33).
- **Reward probe.** Per-unit PSTH around rewarded poke entries, from
  `trialtimes_raw_mingyutest/pokes_<rd>_<s>.npy` (pyControl clock).
  - 250 ms blocks over +-30 s; each unit z-scored, then mean |z| across units at each lag.
  - Aligned sessions peak at +0.25 s: ah10 v1 peaks within [-1, +4] s in 38 of 38 sessions.
- **Wall-clock check.** The pyControl file's start stamp minus the OpenEphys folder time, minus
  the fit's own offset `b_samples` / 30000.
  - Normal recordings: per-mouse median -0.3 to -0.7 s, IQR about -2 to +1 s.
  - The stamps have 1 s resolution.
- **Tracking check.** The tracked node equals the poked port at rewarded entries in a median 100%
  of rewards per session. The best realignment shift is within 1 s in 173 of 173 sessions.
  Tracking is fine, so the probes test the spikes.

## Results

| speed probe, sessions with r > 0.2 (95 of 211) | v1 | v2 |
|---|---|---|
| lag more than 1 s from -0.30 s | 12, all ly05 | 36, all 5 mice |
| ah08 / ah10 / ly06 / ly07 | 0 / 0 / 0 / 0 | 2 / 11 / 7 / 3 |

- **v1 against v2 directly:**
  - They hold the same spikes, shifted in time: per-unit count correlation 0.77-0.87 at the best
    lag.
  - 44 of 211 task sessions differ by more than 1 s.
- **ly05:** the speed probe fails there because the cross-correlation is weak and multi-peaked.
  The reward probe shows ly05 v1 aligned except on `ly05_20250620_20250623`.

The anchor is the ephys sample of pyControl rsync[0]:

| recording | ap_first_sample | ap_first / 30 kHz | v2 - legacy anchor | v2 speed-probe shift | wall-clock discrepancy |
|---|---|---|---|---|---|
| ah10 2025-06-20_16-37-32 | 578,636 | 19.29 s | -19.29 s | +19.03 s | +18.80 s |
| ah10 2025-06-23_16-20-49 | 873,964 | 29.13 s | -29.13 s | +28.78 s | +26.96 s |
| ah08 2025-06-25_15-00-21 | 615,232 | 20.51 s | -20.47 s | +20.90 s | +20.46 s |
| ly06 2025-06-25_09-14-15 | 389,696 | 12.99 s | -12.93 s | +13.05 s | +13.18 s |
| ly07 2025-06-24_17-39-49 | 415,114 | 13.84 s | -13.84 s | +14.08 s | +10.00 s |
| ah10 2025-06-20_11-49-08 | 91,796 | 3.06 s | -3.06 s | +2.68 s | +2.20 s |

In `data/processed_data/ephys_rsync_qc.csv`, `ap_first_sample` has median 0 in normal task
recordings and median 139,638 in shifted ones. `ttl_before_ap_start` is above 0 only in shifted
ones.

## Root cause
High confidence, but not yet checked against the OpenEphys metadata.
- **The line:** `code/preprocessing/extract_rsync_from_ttl.py:128-137` returns
  `rising - ap_first`, where `ap_first` is element 0 of the AP stream's `sample_numbers.npy`.
- **Why it's wrong:** the legacy caches in `ephys_rsync_mingyutest/` used the raw TTL sample
  numbers with no subtraction, and they are aligned in these recordings. So the TTL numbers
  already sit on the `.dat` sample timebase. Subtracting `ap_first` maps pyControl events too
  early, and spikes land late by `ap_first` / 30 kHz.
- **The docstring's mEC note is probably the same artefact.** It says the first TTL edges sit
  ~136k samples before the AP stream's first sample on 2024-03-24_10-52-32.
- **Pulse pairing is correct:**
  - In shifted recordings the histogram peak holds 172-253 pulses against a second-highest bin of
    6-9, and 245 of 245 pulses match at 0.055 ms RMS.
  - It is not a partner-box mix-up: ah10 was the only task mouse in the 06-20 16:37 and
    06-23 16:20 recordings.
- **To verify:** for one shifted and one normal recording, read:
  - `sync_messages.txt` and `structure.oebin`;
  - the TTL `sample_numbers.npy` and the AP `sample_numbers.npy`;
  - the concat offsets in `extract_neuron_raw`, which derives them from `dat_size`.

  Then settle which origin each one uses.

## ly05_20250620_20250623: v1 is wrong
- `MEC_INTEGRATION.md` §7 already notes that its legacy trains match neither ly05's nor ly06's
  pyControl, while the raw TTL matches ly05's own at 246 of 251 pulses, 0.04 ms.
- The predicted v1 error is the v2 anchor plus the AP start sample, minus the legacy anchor:

  | session | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |
  |---|---|---|---|---|---|---|---|---|
  | predicted v1 error (s) | +5.31 | +2.55 | +10.37 | +10.24 | -10.00 | +38.58 | +8.20 | +5.68 |

- The reward probe on v1 measured +5.5 s (s0), +10.5 s (s2) and -9.5 s (s4). s5 falls outside
  the +-30 s window.

## Affected
- **LEC v2 files:** the task arrays in `data/processed_data/neuron_raw_v2/`. The `_sb_` rest
  arrays there are byte-identical legacy copies with origin 0 and are not affected.
- **LEC pickles:** `data_dic_lec_v2.pkl` and `data_dic_lec_v3.pkl`.
- **LEC rsync outputs:** `ephys_rsync/<mouse>/rsync_timestamps_*.npy` and `rsync_fit_*.json`
  wherever `ap_first` > 0.
- **mEC tree:** the same files, plus `data_dic_mec_v2.pkl` and `data_dic_mec_v3.pkl`.
- **v1:** `ly05_20250620_20250623` only, on top of the known 50-200 ms offsets in ah08, ly06 and
  ly05.
- **Earlier measurements:** any number taken on the fitted clock, such as entry lags and
  "sessions that start before ephys sample 0".
- **Not affected:** trial times, pokes, v2/v3 tracking and rest arrays. PFC uses El-Gaby's own
  alignment and has not been probed.

## Fix requested
1. Confirm the origin of the TTL sample numbers, as above.
2. Write corrected trains and fits, plus `neuron_raw_v4/`, as copies.
   - Keep v2's histogram pairing: it is correct, and it fixes v1's partner-pulse and
     probe-offset errors.
   - Drop only the wrong subtraction.
3. Build `data_dic_lec_v4.pkl` from v4 spikes plus v3 tracking, and the mEC equivalent.
   Overwrite nothing.
4. Add the speed, reward and wall-clock probes to `validate_alignment.py`. At present it checks
   only pulse matching.
5. Finish before the July ABCDE blocks are extracted.

## Gates
- Where `ap_first` = 0: v4 equals v2 byte for byte.
- Where `ap_first` > 0: v4 equals v2 shifted by exactly `ap_first` / 30 kHz, to the bin.
- Every task session must pass every probe it can be tested on:
  - speed-probe lag within +-0.5 s of -0.30 s where r > 0.2;
  - reward-probe peak within [-1, +4] s where the peak-to-median ratio exceeds 2;
  - wall-clock discrepancy within each mouse's IQR +-3 s.
- No bins before ephys sample 0 unless the wall clock confirms a late start.
- All 8 sessions of `ly05_20250620_20250623` pass the reward probe in v4.
- mEC passes the same gates.

## Repo rules
Copies only, never overwrite. Write a dated `.md`. Run `check_mirror_parity.py` if shared modules
change. Heavy work goes through sbatch.

## Reproduction
Run from the repo root with the maze_ephys python
(`/nfs/nhome/live/aharris/.conda/envs/maze_ephys/bin/python`). It reads only small per-session
files.

```python
import numpy as np, os, re, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
D = 'data/processed_data'
def z(x): x = np.asarray(x, float); s = x.std(); return (x - x.mean()) / (s or 1)
def xlag(a, b, m):  # lag in s maximising corr(a[t], b[t+L]); + means b is later
    a, b = z(a), z(b); c = fftconvolve(b, a[::-1], 'full') / len(a); mid = len(a) - 1
    seg = c[mid - m:mid + m + 1]; return (int(seg.argmax()) - m) * 0.025, seg.max()
def speed(xy):
    xy = np.asarray(xy, float).copy()
    for k in range(2):
        ok = np.isfinite(xy[:, k]); xy[:, k] = np.interp(np.arange(len(xy)), np.flatnonzero(ok), xy[ok, k])
    return np.hypot(*np.gradient(gaussian_filter1d(xy, 3, axis=0), axis=0).T)
pat = re.compile(r'^Neuron_raw_([a-z]+\d+_\d{8}_\d{8})_(\d+)\.npy$')   # excludes _sb_ rest arrays
for fn in sorted(os.listdir(f'{D}/neuron_raw_v2')):
    m = pat.match(fn)
    if not m: continue
    rd, s = m.group(1), int(m.group(2)); mouse, blk = rd.split('_', 1)
    xyf = f'data/processed_v3/{mouse}/{blk}/XY_raw_{rd}_{s}.npy'
    if not os.path.exists(xyf): continue
    sp = speed(np.load(xyf)); out = []
    for root in ('neuron_raw_mingyutest', 'neuron_raw_v2'):
        pop = gaussian_filter1d(np.load(f'{D}/{root}/{fn}').sum(0).astype(float), 10)
        n = min(len(pop), len(sp)); lead = int(np.argmax(pop > 0))
        out += list(xlag(sp[lead + 40:n], pop[lead + 40:n], 1600))
    print(rd, s, 'v1 lag %+.2f r %.2f | v2 lag %+.2f r %.2f' % tuple(out))
q = pd.read_csv(f'{D}/ephys_rsync_qc.csv'); q = q[q.kind == 'task']
for _, r in q.iterrows():
    leg = f'{D}/ephys_rsync_mingyutest/{r.mouse}/rsync_timestamps_{r.recording}.npy'
    new = f'{D}/ephys_rsync/{r.mouse}/rsync_timestamps_{r.recording}.npy'
    if os.path.exists(leg) and os.path.exists(new):
        d = (np.load(new)[0] - np.load(leg)[0]) / 30000
        if abs(d) > 1: print(r.mouse, r.recording, 'ap_first/30k %.2f  v2-legacy %+.2f' % (r.ap_first_sample / 30000, d))
```

The reward probe and the tracking check were run the same way: per-unit PSTHs around rewarded
poke entries (`pokes_*.npy` column 3 = rewarded, column 0 = entry bin), and the tracked node
(`processed_v3/.../Locs_raw_*.npy`, strings `node_k`) compared with the poked port (column 2).
