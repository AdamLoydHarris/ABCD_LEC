# LFP anatomy check: ripples test shank orientation, theta tests MEC vs LEC

Code: `lfp_anatomy_check.py`, `lfp_anatomy_check_synthetics.py`, `run_lfp_anatomy_check.py`,
`lfp_anatomy_check.ipynb`, `sbatch_files/lfp_anatomy_check.sbatch`. LEC-only: El-Gaby's PFC data has
no raw continuous data, so nothing is mirrored to `mFC_data/code/`.
Outputs: `data/processed_data/lfp_anatomy_check/`, figures `data/figures/lfp_anatomy_check/`.

## 2026-10-02 — why, and the design (written before any real result was read)

**Question.** Do LEC-dataset units that the atlas tags ENTm, especially near the ENTl/ENTm border,
carry an MEC physiology? And is the shank order relative to the atlas right?

**Why the two are one question.** Shank 0 = anterior is a convention (`probe_refit.CONTACT_FACE_AXIS
= 'rl'`, `SHANK_ORDER_VERIFIED = False`). The probe mirror (u, normal, offset_x and theta negated)
reproduces every contact position to 0.0 µm, so dye cannot test it. The structure at a physical
position does not change under the mirror, but each unit's label does, because it comes from its
channel's probe (x, y).

| mouse | ENTm unit-recordings now | if mirrored | ENTm both ways |
|---|---|---|---|
| ah10 | 22 | 6 | 0 |
| ly05 | 34 | 84 | 0 |
| ly06 | 49 | 176 | 0 |
| ly07 | 179 | 85 | 0 |

So every ENTm/ENTl split rests on the convention. This contradicts `ANATOMY_SPLIT_PLAN.md` §3.4,
which says labels are orientation-independent.

**Adam's surgical prior (2026-10-02):** the contact face "should be lateral" (the current
convention). All 5 LEC implants were left hemisphere with the same holder and mounting. One verdict
therefore covers every mouse, and each mouse is an independent test of it.

**Logic.** Ripples mark CA1/SUB/ProS and do not depend on the MEC-vs-LEC question, so they settle
orientation. Theta then asks whether ENTm-labelled sites are MEC-like. Theta alone would be
circular, because CA1/SUB are theta-rich too.
- Rat expectation (Deshmukh et al. 2010, J Neurophysiol 104:994): relative theta (6-10 Hz over
  1-50 Hz) was 0.50 in MEC vs 0.18 in LEC on a circular track. Median unit theta modulation was
  0.85 vs 0.70.

Current-convention bank-0 anatomy, per shank:

| mouse | shank 0 | shank 1 | shank 2 | shank 3 |
|---|---|---|---|---|
| ly06 | ProS + ENTl-deep | ENTl-deep + SUB | ENTl | all ENTm (2/3/5) |
| ly07 | ProS | SUB + ENTl-deep | ENTl + ENTm6 + SUB | all ENTm (2/3/5) |
| ah10 | CA1 + ENTl-deep | ProS + ENTl-deep | ENTl + SUB | ENTl + 1-column ENTm3 sliver |

- ly05: shank 3 mixes ENTm5 and ENTl5; shanks 0-2 carry CA1/SUB/ENTl-deep.
- ah08: pure ENTl.

The hippocampus sits at one end of the shank row and ENTm at the other, which makes the orientation
testable in 4 of 5 mice.

| measure | current mapping correct | mirror correct |
|---|---|---|
| ripple hotspot | on HPF-labelled contacts (shanks 0-1) | on shanks 2-3 |
| ripple modulation of ENTm-labelled units | weak (EC) | strong (they would sit in ProS/SUB) |
| 1/f-corrected theta, layer-matched ENTm5 vs ENTl5 | ENTm > ENTl if MEC-like | unrelated to labels |

### Data

**Probe map (stage `probemap`).** Peter Doohan's 2025-05-29 to 06-01 recordings, before the task.
- Banks 0-4 cover y 0-3585 µm on all 4 shanks; ly05 has 6 banks (to 4305 µm), ly07 has 4 (to 2865 µm).
- 900 s per bank. Each folder holds one stream with the mouse's own probe serial, external reference.
- `lfp.signal.npy`: float16 µV, 1500 Hz, 1-450 Hz Butterworth, no re-referencing. 192 columns, one
  per raw channel `CH<label>`.

**Peter's `lfp.metrics.htsv` positions are wrong for 50-100% of channels**, so they are not used.
- His code assigns channel names CH0-383 in shank-major contact order. The NP2.0 hardware
  interleaves shanks in 48-channel blocks.
- Measured on all 26 configs, against probeinterface on each recording's own settings.xml:
  - bank 0 and bank 4: 50% of positions agree;
  - banks 1-3 and 5: 0% agree;
  - every disagreement is a shank swap with a 360 µm depth shift.
- The task sort's geometry is the probeinterface one, and it is physically right. In
  ly07_20250620_20250623, 98.5% of each unit's 8 next-strongest template channels lie within 60 µm
  of its peak channel, 100% on the same shank, 0% at 330-390 µm.
- So each column's position is taken from probeinterface for its raw channel. Gate A0 recomputes
  every column from the raw `.dat` and checks that the best match is its own channel.

**Raw pilot (stage `raw`).** ly06_20250618_20250619 and ly07_20250620_20250623, the June blocks
with the most current-ENTm units (12 and 45).
- Per mouse: the ABCD task session with the most running time, plus the longest sleep-box recording
  in the same block (ly06 sb0 at 42.3 min, ly07 sb5 at 35.0 min). Same sort, so the same units.
- Raw is int16 wideband at 30 kHz, 0.195 µV/bit, external reference (979/981 probe entries; the 2
  tip-referenced ones are ah08 and are not used).
- The stream comes from each block's `recording_sessions_in_concat.csv`, never from the probe
  letter, and the serial is asserted. LFP is decimated here to 1250 Hz with a polyphase FIR.

**Spikes (stages `units`, `raw`).** `sorter_output/spike_times.npy` through the `QC_single_units`
lookup, exactly as `extract_neuron_raw` does.
- Per-recording `.dat` index = spike sample − `concat_table.start`.
- Task bins use the v4 fitted clock.
- Every session's re-binned spikes must equal `neuron_raw_v4` (max|diff| = 0).

**Behaviour.** Speed from `XY_raw` of `data_dic_lec_v4.pkl` (cm) via `tracking_impact.speed_glm`
(cm/s). Running = all bins > 5 cm/s (repo convention); still = all bins < 2 cm/s.

### Measures
- **LFP, per channel and state** (2 s Hann windows, 0.5 Hz):
  - theta excess: mean dB above a log-log aperiodic fit over 2-45 Hz (excluding 4-12 and 48-52 Hz,
    one outlier pass), averaged over 6-10 Hz; plus the peak frequency;
  - relative theta 6-10 / 1-50 Hz (Deshmukh's normalisation);
  - ripple-band excess: mean dB above a fit over 80-360 Hz excluding 130-270 Hz, averaged over
    150-250 Hz.
  - Monopolar, plus a 60 µm same-column bipolar derivation, which is reference-free.
- **States.** Task: running vs still. Sleep and probe map (no tracking): top 30% / bottom 50% of 2 s
  windows by probe-median theta/delta.
- **Ripples.** Reference channel = the largest rest-state ripple-band excess, which a broadband MUA
  channel cannot win.
  - Events: 150-250 Hz envelope z > 3 for 20-300 ms, peak z > 5, during rest.
  - MUA guard: an event is dropped when the 300-450 Hz z at its peak reaches the ripple z.
  - Ripple gain per channel = its envelope within ±25 ms of reference peaks over its own rest
    median. The reference channel itself is excluded from AUCs, since it is selection-biased.
- **Units:**
  - theta excess of the running-epoch ACG: 0-500 ms at 5 ms; first 20 ms set to the mean; dB above
    a quadratic-in-log-f fit over 1-30 Hz minus 4-12 Hz, at the 5-11 Hz peak ±1 Hz;
  - p from 200 uniform ±62.5 ms jitters;
  - the Brandon/Kropff ratio, reported for the literature but not used for significance;
  - burst fraction (ISI < 10 ms);
  - raw pilot only: theta phase locking (MRL, Rayleigh) to probe-median theta and to the local
    bipolar at the peak channel; ripple modulation (±50 ms of reference ripples vs rest baseline,
    Poisson p); LFP measures at the peak channel.

### Decision rules (fixed here before running)
- **Orientation, per mouse.**
  - "current" if the ripple-gain AUC (ripple structures CA1/CA2/CA3/SUB/ProS/HPF vs other grey
    matter; fibre, DG, ventricle and outside excluded) is ≥ 0.7 under the current labels and ≤ 0.5
    under the mirror labels. (Amended before any real data, see the synthetic outcomes below:
    winner ≥ 0.7 and winner − loser ≥ 0.2.)
  - "mirror" if the reverse; otherwise "inconclusive".
  - Stage A (probe map) and Stage C (raw pilot, ly06/ly07) are read separately.
  - Ripple-band excess and own-event rate are secondary and must not contradict.
  - The rank of the current mapping among all 24 shank permutations is reported.
- **A hotspot on contacts the atlas calls EC under both mappings** flags a registration or depth
  error, not an orientation one.
- **MEC-like, within mouse:** ENTm > ENTl, and layer-matched ENTm5 > ENTl5, on bipolar theta excess
  (channels) and on unit ACG theta excess. Units are tested against a within-recday label-shuffle
  null. ly06 and ly07 are the only clean tests: ah10's ENTm is a 1-column sliver, ly05's
  registration is suspect, and ah08 has none.

### Synthetic controls (`python lfp_anatomy_check_synthetics.py --quick`)
The table is in the script header. All cases enter through the real loaders: probe-map folders
written in Peter's labelling convention with a real settings.xml, and an OpenEphys folder for the
raw path. Outcomes are below.

### What this cannot establish
- Region identity of single units; only population tendencies.
- Population inference on MEC-vs-LEC theta, with n = 2 informative mice.
- The border position finer than a shank. On ly06/ly07 it falls in the 250 µm gap between shanks 2
  and 3, and the atlas border has no image edge.
- That the probe did not move between the probe map (late May) and the task (June).

## 2026-10-02 — synthetic outcomes

First quick run: 37/41.
- **The ratio theta index failed on bursty units:** 35/40 non-rhythmic bursty units were called
  theta-rhythmic against the jitter null. Jitter destroys burst and refractory structure that the
  ratio reads as theta power.
- Replaced by the ACG theta excess above a smooth spectral fit. Prototype, 30 units each at about
  4.3k running spikes:

  | unit | FPR or detection |
  |---|---|
  | Poisson | 0.00 |
  | bursty | 0.00 |
  | burst + 60 ms AHP | 0.03 |
  | slow-modulated | 0.00 |
  | theta m=0.3 | 0.23 |
  | theta m=0.6 | 0.93 |

- The other three failures were synthetic-length problems: too few ripple events in 30 s of raw,
  too few theta windows for the bipolar contrast, and too few spikes for m=0.3. Fixed by longer
  cases and ≥ 150 ms event spacing.

**Decision-rule amendment (2026-10-02, from synthetic check 1, before any real data was read).**
The orientation verdict was "winner ≥ 0.7 and loser ≤ 0.5". It is now "winner ≥ 0.7 and
winner − loser ≥ 0.2".
- Reason: when the mirror puts ripple labels on noise-level contacts, its AUC sits at 0.45-0.50 by
  construction. Check 1 gave 1.000 vs 0.494, so a correct verdict was one unlucky draw from
  "inconclusive".
- Under noise (check 3) every measure stays inconclusive either way, with AUCs 0.43-0.53.
- The raw-door synthetic hit the same edge in the rerun: 1.000 vs 0.546, which the old rule called
  inconclusive.

## 2026-10-02 — smoke tests on real data, and what they changed

Each stage was first run at small scale in the interactive shell, into scratch, before queueing:
one mouse, one config, one recday, or the first 3 min of one recording. That caught five problems
the synthetics could not:

1. **pandas 3 (histology env) keeps NaN through `astype(str)`.** The anatomy gate compared NaN with
   NaN for above-brain contacts and read ah10 0.991 and ly05 0.862. All mismatches were
   missing-vs-missing; the comparison now treats two missing labels as equal. After the fix all 5
   mice are exact: current labels = htsv 100%, mirror position error 0.0 voxels, mirror labels =
   twin lookup 100%, channel_regions.csv 1901/1901.
2. **Gate A0's absolute margin was too strict.** ly06 config_1: 39 columns matched their own raw
   channel (r 0.998, best match = own) but failed a 0.05 margin, because in ProS/SUB the next contact
   reaches r 0.95 at 100-400 Hz. The rule is now best = own, r_own ≥ 0.98 and
   (1 − r_second) ≥ 5 (1 − r_own). One flat channel (CH290, Peter's qc "dead") is flagged dead and
   excluded instead of failing the config.
3. **The 2 s "every bin > 5 cm/s" running rule kept almost nothing.** Maze runs last about 1 s:
   median bout 1.05 s in ly06 s4, 0.95 s in ly07 s4, 0.43 s in ah08 s0. The rule kept 0.1 of
   8.7 running min.
   - Now: running = ≥ 75% of a window's bins > 5 cm/s with mean > 5; still = ≥ 90% < 2 cm/s
     with max < 5.
   - That keeps 5.7 / 6.3 min running and 3.2 / 2.1 min still in the ly06 / ly07 pilot sessions.
   - Synthetic check 15 now covers `speed_windows`, including a 3 s offset negative control.
4. **The decimation gate's window could start past a shortened LFP.** It is now clamped to the
   decimated span.
5. **Process deviations, because the cpu queue was saturated** (about 50 jobs from another user;
   the first estimated start was the next morning).
   - The anatomy stage, 0.5 min, ran interactively through the runner.
   - The speed stage was rewritten to read the tracking v3 `XY_raw` arrays and the
     `neuron_raw_v4` headers that `data_dic_lec_v4` was built from, instead of loading the 3.8 GB
     pickle. It took 0.7 min: 191 ABCD sessions over 25 recdays, 20 object sessions skipped, every
     clock n_bins equal to Neuron_raw.
   - Neither stage loads a data_dic or fits a recday. Probe-map, units and raw run under sbatch.

Smoke-test values, single config or recday, not results:
- ly06 config_1 ripple-band excess AUC: current 0.943, mirror 0.613.
- ly06 s4, first 3 min, raw: current 0.957, mirror 0.469.
- ly06_20250618_20250619 units: spike round trip max|diff| 0 on all 8 sessions.

Quick synthetics after all fixes: run 3 passed 45/45. **Run 4 passed 48/48** (adds check 15), with
these values:

| check | measured |
|---|---|
| (1) ripples planted at current-truth HPF contacts | AUC current 1.000, mirror 0.494, verdict current, rank 1/24 |
| (2) ripples planted at mirror-truth contacts | AUC current 0.461, mirror 1.000, verdict mirror |
| (3) no ripples, 5 seeds | 0 false verdicts on every measure; finite AUCs 0.43-0.53 |
| (4) reference artifact | monopolar theta +4.0 dB, bipolar +0.00 dB; non-HPF ripple excess mono +5.0 dB, bipolar −0.00 dB |
| (5) MUA channel | not the reference; ripple excess −1.3 dB; guarded events 0.000 Hz vs unguarded 0.950 Hz |
| (6) aperiodic exponent χ=1 and χ=2 | 0.95-1.07 and 2.00-2.03; pure 1/f theta excess within ±0.33 dB; 8 Hz bump +9.3 to +32.8 dB |
| (7) theta FPR at p<0.05, 40 units each | Poisson 0.025, bursty 0.000, burst+AHP 0.075, slow-modulated 0.100 (bound 0.136) |
| (7) theta detection | m=0.3 0.20, m=0.6 1.00 (about 4.3k running spikes per unit) |
| (8) phase locking | phase error < 10°; a 10 s spike offset gives MRL 0.05 |
| (10) gate A0 | flags exactly the 2 swapped columns |
| (11) ENTm theta contrast, monopolar and bipolar | AUC 0.996-1.000 under current, 0.31-0.36 under mirror |
| (12) raw door | decimation r 1.0000; verdict current (1.000 vs 0.546 under the amended rule) |
| (13) using lfp.metrics x/y | loses the planted map (AUC 0.450) |
| (14) clock lag | planted 0.2 s and 2.0 s recovered exactly |
| (15) speed windows | run windows ≥ 0.88 inside planted running; still windows 1.00; a 3 s offset gives 0.00 |

**Sixth problem, found in the first real probe-map job (ah08, old code): an LFP-only channel won the
ripple reference.**
- Banks 2-5 detected 0 ripples, although their CA1 channels detect 0.78-0.92 ripples/s at rest.
- The cause: ah08 config_3's reference was CH120. It has a normal LFP but 300-450 Hz power of
  2×10⁻⁶ against about 55 on its CA1 neighbours.
- Its steeply falling spectrum turns into a spurious ripple-band "excess". Its near-zero MAD then
  made z explode (max 2958), so no event fit the duration limit.
- The MUA guard was not the cause: it removes 8-17% of events on CA1 channels.
- Fix: a channel is abnormal if its SD is below 5% of the probe median or its 300-450 Hz power is
  below 1% of the median. Abnormal channels are excluded from every measure and from the reference.
  Synthetic check 9 now includes an LFP-only channel.
- ah08 config_3 after the fix: 5 abnormal channels, reference on CA1 CH250 (y 1515), 393 ripples.
- The ah08/ah10/ly05 probe-map jobs that started on the old code were cancelled and resubmitted.
- Quick synthetics run 5, after the fix: **48/48**. Both channels are flagged (flat #10, LFP-only
  #20), neither is the reference, and every other check is unchanged.

**Post-hoc test, added after the ly06 probe map, flagged as such everywhere it is reported.**
- The pooled per-mapping AUCs were inconclusive on ly06: ripple gain 0.570 vs 0.472, ripple-band
  excess 0.795 vs 0.679.
- Diagnosis: the hippocampus spans all 4 shanks in the higher banks, so 471 contacts are ripple
  structures under the current mapping and 453 under the mirror, mostly the same ones. A contact
  labelled the same both ways cannot discriminate and pulls both AUCs together. Ripple gain also
  has one reference per bank, so pooling banks mixes scales.
- `discordant_test` uses only contacts whose class differs between the mappings:
  A = ripple structure only under current, B = only under the mirror. Ranks are taken within bank,
  and p comes from shuffling A/B within bank. AUC(A > B) > 0.5 favours current.
- The merge stage also reports per-bank AUCs.
- Synthetic coverage (quick run 6, **51/51**):

  | synthetic case | discordant AUC | p |
  |---|---|---|
  | current truth | 1.000 | 0.002 |
  | mirror truth | 0.000 | 0.002 |
  | noise, 5 seeds | 0.37-0.62 | 0/5 significant |

- The pre-registered verdicts stand as computed. The post-hoc numbers are reported beside them.

## 2026-10-02 — results: unit theta rhythmicity (stage `units`, job 3674938)

**Gates.**
- Spike round trip against `neuron_raw_v4`: max|diff| 0 on all 191 ABCD sessions of all 25 recdays.
- Unit labels equal `unit_regions` 100% (2851/2851 resolved).

**Measure.** ACG theta excess in dB during running (≥ 1 s epochs, > 5 cm/s), all task sessions of a
recday pooled. "Rhythmic" means jitter p < 0.05. n = unit-recordings pooled over the 5 blocks: the
same neuron recurs, so these are not independent cells.

Current labels; each cell is median dB, fraction rhythmic, n:

| mouse | ENTl-sup | ENTl-deep | ENTm | ripple structures (CA1/SUB/ProS) |
|---|---|---|---|---|
| ah08 | 3.7, 0.21, 472 | 4.8, 0.31, 152 | — | — |
| ah10 | 5.9, 0.33, 9 | 8.1, 0.51, 331 | 2.9, 0.23, 22 | 16.2, 0.79, 236 |
| ly05 | 2.8, 0.13, 40 | 3.7, 0.30, 127 | 6.9, 0.42, 33 | 11.2, 0.67, 173 |
| ly06 | 3.4, 0.44, 9 | 9.5, 0.56, 112 | 1.6, 0.21, 48 | 9.4, 0.60, 115 |
| ly07 | — | 4.1, 0.33, 87 | 3.1, 0.25, 156 | 10.2, 0.57, 123 |

Within-recday label-shuffle contrasts on theta excess, current labels:

| contrast | ah10 | ly05 | ly06 | ly07 |
|---|---|---|---|---|
| ripple structures − ENTl-deep | +8.1 dB, p 0.001 | +7.6 dB, p 0.001 | −0.1 dB, p 0.55 | +6.1 dB, p 0.003 |
| ENTm − ENTl-deep | −5.2 dB, p 0.99 | +3.2 dB, p 0.046 | −7.9 dB, p 1.00 | −1.0 dB, p 0.74 |
| ENTm5 − ENTl5, layer-matched | — | +3.9 dB, p 0.017 (33 v 90) | −3.2 dB, p 1.00 (7 v 22) | +3.6 dB, p 0.018 (29 v 36) |

**Reading.**
- ENTm-labelled units are not more theta-rhythmic than deep ENTl as a group.
- Layer-matched, 2 of 3 mice show ENTm5 > ENTl5 by about 3.7 dB; ly06 rests on 7 ENTm5 units.
- Hippocampal units are clearly the most rhythmic.
- Under the mirror labels, every "ENTm − ENTl-deep" difference turns positive (ly07 +6.1 dB,
  p 0.001). That is expected and not evidence for the mirror: mirror "ENTm" sits where the current
  map puts ProS/SUB, the most theta-rhythmic tissue. This is why theta cannot settle orientation.
- ly06's shank-3 units (all ENTm under current) have the weakest theta of any group, 1.6 dB.
  - Under the mirror they would be ProS and deep ENTl, which in ly06 measure about 9.5 dB.
  - So their weak theta fits the current map better than the mirror. It does not look like a
    strongly theta-rhythmic MEC either.

## 2026-10-02 — results: probe-map orientation (stage `probemap`, jobs 3675018/19/20, 3674936/37)

**Gates.**
- Gate A0 passes on all 26 configs: median r_own 0.997-0.999, every non-dead column matching its
  own raw channel.
- Peter's metrics x/y agree on 50% of columns (banks 0 and 4) or 0% (banks 1-3, 5).
- Ripples were detected in every bank that reaches hippocampus.

**Pre-registered** (pooled over banks; ripple structures vs other grey matter; winner ≥ 0.7 and
≥ 0.2 above the loser):

| mouse | ripple gain (primary), current / mirror | rank /24 | ripple-band excess | own event rate | verdict |
|---|---|---|---|---|---|
| ah08 | 0.704 / 0.677 | 11 | 0.772 / 0.690 | 0.890 / 0.858 | inconclusive |
| ah10 | 0.228 / 0.165 | 4 | 0.587 / 0.479 | 0.786 / 0.794 | inconclusive |
| ly05 | **0.788 / 0.398** | **1** (mirror 24) | **0.906 / 0.558** | 0.653 / 0.560 | **current** |
| ly06 | 0.570 / 0.472 | 6 | 0.795 / 0.679 | 0.687 / 0.640 | inconclusive |
| ly07 | **0.906 / 0.565** | 4 | 0.548 / 0.663 | **0.856 / 0.453** | **current** |

No mouse favours the mirror.
- ah08's hippocampal crossing spans all 4 shanks, so there was little to discriminate (by design).
- ah10's pooled gain is below 0.5 under both mappings: one reference per bank makes cross-bank
  ranks meaningless there.
- ly06 is diluted (471 vs 453 ripple contacts, mostly shared).

**Post-hoc discordant test** (contacts whose class differs between mappings, ranked within bank;
AUC > 0.5 favours current; permutation p, floor 0.001):

| mouse | n current-only / mirror-only | ripple gain | ripple-band excess | own event rate |
|---|---|---|---|---|
| ah08 | 23 / 19 | 1.000 (p 0.001) | 1.000 (p 0.001) | 1.000 (p 0.001) |
| ah10 | 79 / 69 | 0.772 (p 0.001) | 0.880 (p 0.001) | 0.315 (favours mirror, p 0.001) |
| ly05 | 263 / 225 | 0.956 (p 0.001) | 0.888 (p 0.001) | 0.733 (p 0.001) |
| ly06 | 111 / 90 | 0.724 (p 0.001) | 0.894 (p 0.001) | 0.639 (p 0.002) |
| ly07 | 130 / 105 | 0.758 (p 0.001) | 0.467 (p 0.79) | 0.789 (p 0.001) |

**Per bank, ripple gain:**
- "current" in 13 banks, "mirror" in 2: ah08 config_4 (0.151 vs 0.991) and ly07 config_4 (0.505 vs
  0.728). Both are the same height, y 2160-2865 µm.
- The rest are inconclusive or NaN (all ripple-structure or none).

**Reading.**
- The primary measure favours the current shank order in 5/5 mice on the post-hoc discordant test,
  and in 2/5 under the pre-registered pooled rule. No mouse favours the mirror under the
  pre-registered rule.
- The own event rate is the least specific measure. It contradicts on ah10, and it also contradicted
  on the task data, where it is sensitive to local spiking. It should not be used as a ripple marker
  here.
- The two "mirror" banks sit higher on the track, about 2.2-2.9 mm from the tip. The orientation
  evidence comes from banks 0-2, near the EC-hippocampal junction.
- The probe-map ripple measures also match the atlas's hippocampal depth under either mapping
  (AUC 0.70-0.89 in ah08). That supports the registered depth independently of orientation.

## 2026-10-02 — verdict

**1. Shank order: the current convention (shank 0 = anterior, contact face lateral) is supported.
No mouse supports the mirror on a measure that passed its diagnostics.**

| mouse | pre-registered verdict | strongest evidence |
|---|---|---|
| ly06 | **current** (raw sleep) | raw sleep ripple gain AUC 0.997 vs 0.547, rank 1/24; ripple-band excess 1.000 vs 0.546; probe-map discordant gain 0.724 (p 0.001) |
| ly07 | **current** (probe map) | probe-map gain 0.906 vs 0.565, discordant 0.758 (p 0.001); raw ripple-band excess 0.788 (task) and 0.815 (sleep) vs about 0.43 |
| ly05 | **current** (probe map) | probe-map gain 0.788 vs 0.398, rank 1/24 (mirror 24/24); excess 0.906 vs 0.558; discordant 0.956 (p 0.001) |
| ah10 | inconclusive | pooled gain unusable (cross-bank references); discordant gain 0.772 and excess 0.880 favour current (p 0.001) |
| ah08 | inconclusive | hippocampus spans all 4 shanks; the 42 discordant contacts all favour current (AUC 1.000) |

- The current map wins on the primary ripple measure in 3 mice under the pre-registered rule and in
  5 mice on the post-hoc discordant test.
- Adam's surgical prior (contact face lateral, all left hemisphere, same setup) agrees.
- Two banks prefer the mirror on ripple gain, both at y 2.2-2.9 mm (ah08 and ly07 config_4). One
  recording does too (ly07 sb5), but its 34 events are broadband artifacts. The evidence sits in
  banks 0-2, near the EC-hippocampal junction.
- This is enough to set `SHANK_ORDER_VERIFIED = True` with `shank_order = 'convention (shank 0
  anterior), confirmed by ripple maps (LFP_ANATOMY_CHECK.md)'`. That is left for Adam to do after
  reading this.

**2. Do ENTm-labelled sites look like MEC? Partly, and only locally.**

| evidence | result |
|---|---|
| Local LFP theta (bipolar, reference-free), raw pilot | ENTm > ENTl in ly06 and ly07, task running and sleep REM: AUC 0.69-0.82; layer-matched ENTm5 > ENTl5 0.80-0.87. Disappears under the mirror in ly06 (0.36-0.48), not in ly07. |
| Monopolar LFP theta | lower or mixed on ENTm: volume-conducted hippocampal theta dominates |
| Probe-map local theta (one column, 15 min, unknown state) | no consistent ENTm effect: bipolar 0.49-0.56 |
| Unit ACG theta excess, all 2851 unit-recordings | ENTm not above ENTl-deep (lower in ah10 and ly06); layer-matched ENTm5 > ENTl5 by about 3.7 dB in ly05 and ly07 (p 0.02), not in ly06 (7 units) |
| Unit theta locking (ly06 task) | ENTm MRL 0.084 vs ENTl-deep 0.112 |
| Unit ripple modulation (ly06 sleep) | ENTm the least ripple-locked group (58% vs 71-88%) |

The ENTm-labelled tissue is not the strongly theta-rhythmic MEC of rat superficial layers. It is
ventral-lateral MEC within about 100-150 µm of the LEC border (ly06/ly07 shank 3, median
112-119 µm). It carries more local theta than the neighbouring ENTl, but its units are not more
rhythmic as a group. Theta is weak in this task, because maze runs last about 1 s and mice lick at
7 Hz at the ports. Mouse MEC units are less theta-rhythmic than rat ones, and dorsal-to-ventral MEC
loses rhythmicity. Each of these would blunt the rat contrast of 0.50 vs 0.18.

**What is NOT established.**
- Region identity of any single unit.
- MEC-vs-LEC at population level: 2 informative mice for the LFP theta contrast (ly06, ly07). The
  unit contrast's 3 mice disagree.
- The border position finer than a shank: on ly06/ly07 it falls in the 250 µm gap between
  shanks 2 and 3.
- That the probe did not move between the probe map (May 29 - Jun 1) and the task (June).
- Artifact-robust theta (median spectrum): the re-analysis is queued.
  - Jobs 3675415-17 and 3675463-64 (`--from-cache --psd-stat median`).
  - Merge 3675536, then notebook 3675537.
  - Results land in `raw_*_medpsd.pkl` and `summary.txt`.

## 2026-10-02 — results: raw pilot, bank 0, both columns (stage `raw`)

Jobs 3674939-42; task sessions re-analysed from cache (3675414/16) under the new clock gate.

**Gates.**
- Geometry: probeinterface equals the sort's `channel_positions` (max 0 µm).
- Spike round trip against `neuron_raw_v4`: max|diff| 0, for task (fitted clock) and sleep (750
  sample bins).
- Decimation r median 1.0000.
- Clock gate on the task recordings (60-100 Hz, Spearman): ly06 s4 lag 0.0 s, ρ 0.44.

**Orientation, pre-registered, ripple structures vs other grey matter:**

| recording | ripples | ripple gain (primary), current / mirror | rank /24 | ripple-band excess | own event rate | verdict |
|---|---|---|---|---|---|---|
| ly06 sb0, sleep 42 min | 308 | **0.997 / 0.547** | **1** (mirror 23) | **1.000 / 0.546** | 0.678 / 0.575 | **current** |
| ly06 s4, task | 7 (< 20) | n/a | – | **0.999 / 0.453** | 0.208 / 0.728 | inconclusive (no gain; secondaries split) |
| ly07 s4, task | 4 (< 20) | n/a | – | **0.788 / 0.448** | 0.586 / 0.431 | inconclusive (no gain) |
| ly07 sb5, sleep 35 min | 34 | 0.020 / **0.775** | 24 | **0.815 / 0.405** | 0.021 / 0.643 | inconclusive (excess contradicts gain) |

**The ly07 sb5 "ripples" are artifacts, not ripples.** Measured on the 34 events against ly06 sb0's
308:

| measure | ly06 sb0 | ly07 sb5 |
|---|---|---|
| ripple-band rise, current-only vs mirror-only ripple contacts | 22.5 vs 7.4 µV (localised) | 24.1 vs 27.4 µV (uniform) |
| 300-450 Hz rise, same split | 8.8 vs 3.4 µV | 19.5 vs 23.8 µV (broadband) |
| event rate | 0.24/s of rest | 0.03/s of rest |

- A uniform burst divided by each channel's baseline gives higher gain to the quieter mirror-only
  contacts (baseline 10.4 vs 13.1 µV). That inverts the gain map.
- Units are barely ripple-modulated in sb5: 2-16% at p < 0.01, against 58-88% in ly06 sb0.
- The mouse was probably awake in the sleep box; ly07's probe-map bank 0 had 206 ripples.
- The event-free ripple-band excess still favours current in both ly07 recordings.

**Theta, ENTm vs ENTl channels** (AUC; layer-matched ENTm5 vs ENTl5 in brackets; theta state =
running in task, REM-like top 30% theta/delta in sleep):

| recording | monopolar, current | bipolar, current | bipolar, mirror |
|---|---|---|---|
| ly06 s4 task | 0.65 (1.00) | **0.81 (0.87)** | 0.48 (0.40) |
| ly06 sb0 sleep | 0.33 (0.00) | **0.82 (0.87)** | 0.36 (0.33) |
| ly07 s4 task | 0.18 (0.03) | **0.69 (0.80)** | 0.56 (0.66) |
| ly07 sb5 sleep | 0.61 (0.37) | **0.69 (0.82)** | 0.77 (0.96) |

- Local (bipolar, reference-free) theta is higher on the ENTm-labelled channels than on ENTl, layer
  for layer, in both mice and both states.
- Monopolar theta is mostly lower or mixed on ENTm. That fits volume-conducted hippocampal theta,
  largest on the ENTl shanks next to SUB/ProS.
- In ly06 the bipolar contrast disappears under the mirror (0.36-0.48). In ly07 it does not (0.56-
  0.77), so ly07's theta map does not discriminate orientation.

**Units, raw pilot.**
- Theta locking to probe-median theta in ly06 s4, median MRL: ENTm 0.084 (n 12), ENTl-deep 0.112
  (n 24), ripple structures 0.143 (n 31).
- Ripple modulation in ly06 sb0, fraction at p < 0.01: ENTm 0.58, ripple structures 0.71, ENTl-deep
  0.88. Deep EC receives hippocampal output.
- So ENTm units are the least ripple-locked group, as the current map predicts for EC. But most
  units are ripple-locked, so this discriminates weakly.

## 2026-10-02 — the task LFP is artifact- and licking-contaminated in the theta band (ly06 s4)

The first full raw recording, ly06 s4 (job 3674939), failed its clock gate: probe-median theta power
against speed gave r 0.05 at −3 s. The LFP shares the `.dat` with the spikes at an exact sample
mapping, the spike round trip is exact on this clock, and the bin-to-sample mapping passed synthetic
check 15. So the alignment was not in doubt; the gate's signal was. Measured on the cached 1250 Hz
LFP, 2 s windows, 0.5 s bins:

| quantity | value |
|---|---|
| running / still power, median over channels | 1-4 Hz 8.26 · 6-10 Hz 0.84 · 30-50 Hz 1.16 · 300-450 Hz 0.95 |
| Spearman, log band power vs speed, at 0 lag | 1-4 Hz −0.07 · 6-10 Hz −0.14 · 30-50 Hz +0.16 · 60-100 Hz +0.44 |
| best lag for 60-100 Hz | 0.0 s |

- **Theta-band power does not rise with running in this task; it is higher when the mouse is
  slow.** Mice lick at about 7 Hz at the reward ports, and theta is weak at this site.
- **The 8× "running" 1-4 Hz excess is a few huge broadband artifact bursts.** They dominate a mean
  spectrum while log power does not track speed.
- **The clock gate now uses log 60-100 Hz probe-median power with Spearman correlation** (|lag| ≤
  0.5 s, ρ > 0.1). The theta-based numbers are still reported (`theta_speed_*`). This change is
  post-hoc.
- **Median periodogram option (`--psd-stat median`).** Synthetic check 16: with broadband artifact
  bursts in 6 of 120 windows, the mean-spectrum theta excess moves 14.9 dB and the median moves
  0.11 dB. On clean data the two agree within 0.17 dB.
  - The raw pilot is therefore re-analysed from its LFP cache (`--from-cache`) under both
    statistics. The median version is the artifact-robust one; the mean stays the pre-registered
    one.
  - Rank-correlation clock lag, also check 16: recovers a planted 0.3 s lag under heavy-tailed
    bursts where Pearson (r 0.08) fails.
  - Quick synthetics run 7, with check 16: **54/54**.
- ly06 s4, mean spectrum, before re-analysis:
  - ENTm vs ENTl theta excess during running: monopolar AUC 0.65 (layer 5: 1.00); bipolar 0.81
    (layer 5: 0.87). Mirror: 0.05 / 0.48.
  - Ripple-band excess AUC: current 0.999 vs mirror 0.453.
  - Own event rate: 0.206 vs 0.730, contradicting.
  - Only 7 ripples in 200 s still, so no gain verdict.
  - Unit theta locking to probe-median theta, median MRL: ENTm 0.084 (n 12), ENTl-deep 0.112
    (n 24), ripple structures 0.143 (n 31).

(further results follow in dated sections)

## Reproduce
```
python code/lfp_anatomy_check_synthetics.py --quick          # 54/54 at run 7 (2026-10-02)
S=sbatch_files/lfp_anatomy_check.sbatch
# light table builds (ran interactively on 2026-10-02 because the queue was saturated; either works)
sbatch --job-name=lfp_anat_anatomy --mem=24G $S --stage anatomy         # histology env
sbatch --job-name=lfp_anat_speed   --mem=16G $S --stage speed
for m in ah08 ah10 ly05 ly06 ly07; do
  sbatch --job-name=lfp_anat_pm_$m --cpus-per-task=2 --mem=20G --time=0-01:30 $S --stage probemap --mice $m
done
sbatch --job-name=lfp_anat_units --cpus-per-task=4 --mem=24G --time=0-01:00 $S --stage units --workers 4
for rd in ly06_20250618_20250619 ly07_20250620_20250623; do for s in auto-task auto-sleep; do
  sbatch --job-name=lfp_anat_raw --cpus-per-task=2 --mem=48G --time=0-02:00 $S --stage raw --recday $rd --session $s
done; done
# re-analysis from the 1250 Hz cache (no raw re-read): new clock gate, and the median spectrum
for x in "ly06_20250618_20250619 s4" "ly07_20250620_20250623 s4" "ly06_20250618_20250619 sb0" "ly07_20250620_20250623 sb5"; do
  set -- $x
  sbatch --job-name=lfp_anat_rawc --mem=40G $S --stage raw --recday $1 --session $2 --from-cache --psd-stat median
done
sbatch --job-name=lfp_anat_merge --mem=16G $S --stage merge                # combines per-mouse files + post-hoc tests
jupyter nbconvert --to notebook --execute --inplace code/lfp_anatomy_check.ipynb   # figures -> data/figures/lfp_anatomy_check/
```
Outputs (data/processed_data/lfp_anatomy_check/):
- `contact_anatomy.csv`: labels under both mappings for every contact.
- `probemap_<mouse>.pkl` and the combined `probemap.pkl`.
- `units_theta_all.pkl`.
- `raw_<recday>_<session>[_medpsd].pkl`.
- `lfp_cache/`: float16 1250 Hz LFP for the 4 pilot recordings, **6.5 GB** (1.2 / 2.4 / 1.3 /
  2.0 GB). Delete it once the median re-analysis has landed.
- `summary.txt` and the `*_gates.json` files.
