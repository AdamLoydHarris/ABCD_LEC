# Event-locked kernels in LEC vs PFC (Tsao 2018, Kanter 2025)

Per-neuron kernels of time since maze entry, first A, reward A and every reward, with their time
constants, in LEC and PFC, read as the state of an SSM trace bank. Plan, user decisions, process
rules and code map: `docs/handoff/EVENT_KERNELS_HANDOFF.md`. Population drift (Kanter) results
are in `code/POPULATION_DRIFT.md`.

## 2026-10-16 — M4 part 1: rest-entry kernels and intrinsic timescales in rest

User decisions, 2026-10-15 and 2026-10-16: continuous state terms only, with no binary SWS flag; a
population-rate term, applied by division; then the PFC end-of-recording check. Plan:
`~/.claude/plans/please-read-this-handoff-crystalline-pinwheel.md`. Rest drift is in
`POPULATION_DRIFT.md`, same date.

**Outcome.**

- **After rest entry, the LEC population rate falls in every mouse.** The rate in minutes 15-20 over
  minutes 0-2, median block per mouse, is 0.79-0.84 (5 mice). In PFC it is 0.74-1.31 (7 mice). The
  design divides this shared change out.
- **LEC mice rarely reach DOWN-rich states in rest.** High-DOWN windows, at least 10% of a 10 s
  window in DOWN states, are 0.4-3.9% of windows per mouse; in PFC 9-38%. The LFP check agrees
  that LEC rests hold little SWS (2026-10-15).
- **Relative to the population, many units still lock to rest entry,** held out by rest block,
  beyond the state terms. Mean of mice, range over mice:

  | | LEC, 5 mice | PFC, 5 mice with a fit |
  |---|---|---|
  | units locked to rest entry | 0.289 (0.137-0.349) | 0.132 (0.036-0.323) |
  | units locked to the state terms | 0.647 (0.463-0.734) | 0.687 (0.548-0.811) |
  | rest entry beats a linear day trend, of rest-entry-locked units | 0.60 (0.34-0.73), 845 units | 0.74, 138 units |

  - Stage 2: 70% of LEC rest-entry-locked units are ramps, up 0.37 and down 0.33. Their median tau
    is 1280 s, IQR 453-10240 s; 47% are at 30 min or slower and 29% sit at the grid edge. PFC: 59%
    ramps, median tau 905 s.
  - Units locked to the state terms lock to rest entry more often: 0.33 against 0.24 in LEC, pooled
    units. Unit-specific state dependence is a candidate source, and the gate does not plant it.
- **The maze-entry clock does not restart at rest entry, unit by unit.**
  - Maze-entry-locked units are not more often rest-entry locked. Per mouse, LEC 0.29-0.35 against
    0.30-0.36 for the other units, with ly07 0.13 against 0.14; PFC the same.
  - Of 107 LEC units that ramp after both entries, 0.59 share the sign. Rest taus run a median
    1280 s against 320 s after maze entry, and the two are unrelated.
- **Intrinsic timescales.**

  | | LEC | PFC |
  |---|---|---|
  | slow ACF tau, low-DOWN rest over task: median ratio; Spearman across units | 1.26; 0.14 (2851 units) | 1.14; 0.26 (1128) |
  | Murray tau, low-DOWN rest over task: median ratio; Spearman | 0.66; 0.47 (2263) | 0.91; 0.50 (1056) |
  | slow ACF tau in task against the maze-entry ramp tau: Spearman | 0.05 (567) | 0.04 (138) |
  | Murray tau in task against the reward ramp tau: Spearman | -0.01 (588) | 0.17 (303) |

  - The fast intrinsic timescale is a unit property: it correlates across rest and task, and in
    LEC it is shorter in rest.
  - The slow timescale barely correlates across contexts and runs longer in rest. Rest windows
    carry the shared decline above.
  - Neither timescale tracks the event kernels' time constants. The SSM reading predicted they would.

Predictions in the plan:

| prediction | held? |
|---|---|
| many units change rate over the first minutes of rest, mostly through sleep onset | partly: the LEC population rate falls 16-21% over 20 min, but LEC rests are mostly free of DOWN states, so it is not sleep onset |
| maze-entry ramp units ramp after rest entry too, with similar tau and sign | no: no enrichment, sign 0.59, rest taus 4x longer and unrelated |
| the slow ACF tau tracks the entry-ramp tau; the fast tau tracks the reward kernel's | no: Spearman 0.04-0.05 and -0.01-0.17 |
| non-SWS rest drifts at the task's rate | no: see `POPULATION_DRIFT.md` |

### What was done

| file | role |
|---|---|
| `code/rest_kernels.py` (+ mirror, new) | the proxy, calibrated on the recday's real task sessions; signal-loss QC; `relative_pop_rate`, the median divisor; `build_rest_design`, with `down_frac` deciles and a 0 bin, plus the division; the reset test; low- and high-DOWN windows; Murray's tau; drift segments |
| `code/rest_kernels_synthetics.py` (+ mirror, new) | the planted sleep process and units, `proxy_power`, the shared-ramp option |
| `code/run_rest_kernels.py` (+ mirror, new) | one job per recday, two half fits; `--merge`; `--gate` with the no-state and shared-ramp controls |
| `code/rest_lfp_check.py` (LEC only, new) | the LFP check; the Spearman readout |
| `code/event_kernels_plots.py` (+ mirror) | F31-F37 and their tables |
| `code/event_kernels_M4_rest.ipynb` | load mode. Figures in `data/figures/event_kernels/M4/` |
| `sbatch_files/submit_event_kernels.sh`, `code/check_mirror_parity.py` | runner `rest`; the three pairs |

- **Rows:** 250 ms rows of every valid rest block: at least 2 min, not a misclick, truncated at
  signal loss and dropped below 5 min, with the final 20 s left out. LEC 267 blocks; PFC 239 shards,
  none in the 4 tokenless recdays.
- **Design:** leave one rest block out, centred within block, Freedman-Lane with 100 permutations.
  The groups are `ev_rest_entry` (lags 0-2400 s) and `state`. Each unit half is divided by the
  other half's relative population rate.
- **Runs:** 25 + 25 jobs, every one `EXIT=0`. Not fitted for rest entry, because fewer than 10 units
  per half reach 0.3 Hz: PFC ah03 x2, me08_06092021_09092021, me08_12092021_13092021,
  me10_17122021_19122021, me11_12122021_13122021. So ah03 and me08 have no rest-entry fit.

### Gates (round 5: quick 3771856/57, full 3771858/59, every one `EXIT=0`)

| row | planted | LEC full | PFC full |
|---|---|---|---|
| G1 | the proxy at the real unit count: SWS-window AUC; planted high-DOWN recall / quiet wake high-DOWN | 1.000; 1.000 / 0.008 | 1.000 (reported); 1.000 / 0.000 |
| G2 | rest-entry ramps locked; Stage 2 tau within 2x | 0.833; 0.8 | 1.0; 1.0 |
| G3 | units at 0.3x in SWS, locked with / without the state terms | 0 of 8 / 8 of 8 | 0 of 8 / 8 of 8 |
| G4 | noise locked, with / without the state terms | 0 of 54 / 24 of 54 | 1 of 20 / 15 of 20 |
| G4b | a ramp shared by every unit: noise locked with / without (reported) | 2 of 54 / 49 of 54 | 0 of 20 / 20 of 20 |
| G5 | day trends: the day term beats rest entry | 1.0 | 1.0 |
| G6 | slow ACF tau ratio at 30 / 120 s; Murray tau within 2x at 0.25 s | 0.941 / 0.924; 1.0 | 0.986 / 1.124; 1.0 |
| G7 | drift ratio, equal / 4x faster in rest | 0.935 / 2.204 | 0.909 / 2.293 |
| G8 | the signal-loss cut on a planted dropout | within 60 s | within 60 s |

**Four rounds failed first.** Each fix had a measured cause, recorded in the module docstrings and
handoff §7:

1. The gate calibrated the proxy on planted Poisson task data, which have no lulls. The cut was 0, so
   the planted quiet wake read as SWS. Planted data now use the real calibration.
2. Deciles of the population rate cannot follow a multiplicative shared change: 79 of 92 noise units
   locked under a ramp shared by every unit. The rate now divides.
3. A summed divisor moved 13-18% at rest entry in the PFC gate, from 6 planted ramp units. The
   divisor is now a median.
4. Tied deciles put `down_frac` 0-0.17 in one bin, with 4-5% of planted SWS windows. 0 is now its
   own bin.
5. G6 and G7 rested on 2 units and 10-25 segments; they now have 8 units and up to 120 segments. G6
   scores rest windows clear of the planted quiet wake, as G7 does.

### What was NOT established

- **What rest-entry locking is.** It is relative to the population. A unit whose state dependence
  differs from the population's locks on average, because state changes come at block-specific
  times that average into a time course. State-locked units lock more often (0.33 against 0.24).
  The gate plants only shared and SWS-specific state changes. 40% of LEC calls are matched by a
  day trend.
- **The rest-maze comparison is not like for like.** Rest kernels are relative to the population;
  maze kernels are not. A clock shared by every unit after rest entry would be absorbed, and the
  LEC population decline is such a shared change.
- **Sleep staging.** No LFP-validated SWS, and no REM: quiet wake, drowsiness and REM are all in
  low-DOWN rest.
- **The low-DOWN class is loose where a recday's cut is high.** Cuts run 0.013-0.68 in LEC. ah08,
  at 0.13-0.26, and ly06_20250613, at 0.68 with proxy AUC 0.755, admit DOWN-rich windows as low-DOWN.
- **PFC's DOWN fraction** is high from the first minute in some blocks. In 20-60 units, silences
  are not necessarily sleep.
- **LEC vs PFC:** no contrast was tested.

### Commands

```
python code/check_mirror_parity.py                                          # must exit 0
for ds in lec pfc; do t=code; [ $ds = pfc ] && t=mFC_data/code
  sbatch --time=0-02:00 --job-name=rk_gate_q_$ds sbatch_files/event_kernels.sbatch $t/run_rest_kernels.py --gate --dataset $ds --quick
  sbatch --time=0-03:00 --job-name=rk_gate_f_$ds --dependency=afterok:<quick> sbatch_files/event_kernels.sbatch $t/run_rest_kernels.py --gate --dataset $ds
done
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner rest --min-trials 5 --after <full gate>
bash sbatch_files/submit_event_kernels.sh --dataset pfc --runner rest --min-trials 5 --after <full gate>
sbatch --time=0-01:00 --mem=48G sbatch_files/event_kernels.sbatch code/run_rest_kernels.py --dataset lec --merge --min-trials 5 --n-perm 100
sbatch --time=0-01:00 --mem=48G sbatch_files/event_kernels.sbatch mFC_data/code/run_rest_kernels.py --dataset pfc --merge --min-trials 5 --n-perm 100
python code/rest_lfp_check.py --pick                                        # then one sbatch job per block, then --summarise
sbatch --job-name=ek_nb_m4 --time=0-01:30 --mem=48G --cpus-per-task=4 sbatch_files/event_kernels.sbatch -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 code/event_kernels_M4_rest.ipynb
```

Outputs: shards in `data/event_kernels/rest/shards/<ds>_mt5_np100/`; merged
`data/event_kernels/rest/rest_<ds>_mt5_np100.pkl` with CSVs; gate runs in
`data/event_kernels/rest/runner_gate/`; the LFP check in `data/event_kernels/rest/lfp_check/`.

## 2026-10-15 — PFC entry kernels peak at the end of the recording

Found while captioning F4 for the results page (https://claude.ai/artifact/557jgXBgUcq5jMarZvJBrp).
M2's `table_entry_peak_bins` counts only the first minute and the overflow bin, so it missed this.

**Measured.** Entry-locked units of the merged Stage 1 tables, anchor `entry`; the bin holding each
unit's largest |beta| over the 15 lag bins and the overflow:

| arm | entry-locked units | peak in 1200-1500 s | negative | per mouse, pooled units |
|---|---|---|---|---|
| PFC, >= 10 trials | 424 | 202 (0.48) | 196 | 0.20-0.79 |
| PFC, >= 5 trials | 414 | 184 (0.44) | 178 | 0.17-0.64 |
| LEC, >= 10 trials | 790 | 16 (0.02) | | 0.01-0.03 |
| LEC, >= 5 trials, V3 | 972 | 16 (0.016) | | 0.00-0.02 |

- The reference bin is 960-1200 s in every recday of both datasets. The overflow, beyond 1500 s,
  has no estimate in any PFC unit.
- Rows of the 1200-1500 s bin that fall after the last trial: PFC 88% over 194 sessions, where the
  task ends a median 1153 s after bin 0 and the recording 1219 s; LEC 59% over 191 ABCD sessions
  (1194 s and 1233 s).
- Stage 2 classes of the 184 PFC units: other 119, bump 27, ramp up 20, none 11, ramp down 7. So
  they are 119 of PFC's 130 `other` entry units. LEC's 16: other 8, ramp up 5, none 2, bump 1.
- In PFC the remaining 230 entry-locked units are ramps up 73, ramps down 58, bumps 56, none 32,
  other 11.

**Not established.**

- How many PFC entry calls survive with the post-task segment, or the 1200-1500 s bin, set aside.
  Its share of each unit's held-out cpd was not separated.
- What the drop is: handling at the end of the session, or a feature of the shard's last seconds.
- What it changes in the PFC entry fraction, the reset test, the leaky test and the ramp test.

**Commands.**

```
cd code && python - <<'EOF'
import pickle, numpy as np, event_kernels_plots as P
for ds, f in [('PFC', 'stage1_pfc_kin-strict_mt5_np100.pkl'), ('LEC', 'stage1_lec_kin-strict_mt5_np100.v1.pkl')]:
    M = pickle.load(open('../data/event_kernels/stage1/' + f, 'rb'))
    d = P._rows(M, 'ev_entry'); d = d[d['locked']]
    B, _, _ = P.kernel_rows(M, 'entry', d)
    j = np.nanargmax(np.where(np.isfinite(B), np.abs(B), -np.inf), axis=1)
    s = np.sign(B[np.arange(len(B)), j])
    print(ds, len(j), int((j == 14).sum()), int(((j == 14) & (s < 0)).sum()))
EOF
# post-task share of 1200-1500 s: mFC_data/data/processed/pfc_session_index.v2.csv (first_A_bin, n_core,
# n_post) and data/processed_data/full_sessions_v4/full_sessions_*.csv (n_pre, n_core, n_post), 25 ms bins
```

## 2026-10-14 — M3 part 2: Tsao's stepwise table, Kanter's single-unit timescales, the ramp test

User decisions, 2026-10-13/14:

- commit the leaky step (9b88256, pushed);
- run M3 part 2 with the ramp test;
- Tsao's trial time is time since the last reward, with time since A as an extra candidate;
- the which-clock tests stay outstanding.

The plan and rationale are in `docs/handoff/EVENT_KERNELS_WRITEUP.md`, Part 2. Kanter's population
measures are in `POPULATION_DRIFT.md`, same date.

**Outcome.**

- **Tsao's recipe on our units** (in sample, 500 ms bins, adjusted R² +0.01 to enter, -0.005 to
  leave), as fraction of units, mean of mice with the range over mice:

  | term | LEC, 5 mice, 2851 units | PFC, 7 mice, 1229 units | Tsao, LEC |
  |---|---|---|---|
  | day time (his session time) | 0.175 (0.109-0.270) | 0.190 (0.071-0.458) | BW12 0.182, figure-eight 0.057 |
  | entry time (his BW12 trial time) | 0.166 (0.132-0.235) | 0.174 (0.086-0.320) | BW12 0.062 |
  | leg time (his figure-eight trial time) | 0.062 (0.006-0.143) | 0.142 (0.076-0.243) | figure-eight 0.357 |
  | loop time, since A | 0.014 (0-0.033) | 0.029 (0-0.053) | |
  | task (his wall colour) | 0.629 | 0.699 | |
  | place | 0.880 | 0.846 | |
  | day or entry time | 0.292 (0.237-0.393) | 0.327 (0.221-0.523) | |

- **It agrees with held-out Stage 1 only in part.**
  - In LEC, 70% of the units it gives entry time are Stage 1 entry-locked, against a base rate of
    34%. But it finds only 33% of the entry-locked units. In PFC: 60% and 22%.
  - Reward coding goes to place. Leg time is selected in 9% of LEC reward-locked units and 16% of
    PFC's, while place is selected in 85-88% of all units. Tsao noted that lap time and position
    covary.
  - In sample, it calls 33% (LEC) and 58% (PFC) of planted OU-drift units time cells (gate T7).
- **Its time constants are not comparable with ours.** For entry-time units, f = a e^(bx) gives
  tau a median 2125 s in LEC (IQR 1225-4202 s, 456 units) and 1951 s in PFC (153), against
  Stage 2's 320 s. The single-term exponential has no baseline, so a ramp on a baseline rate reads
  as much slower. It recovers pure exponentials in rate within 1.25x (gate T8).
- **Kanter's GLM test of elapsed time is not specific.**
  - It is significant in 76% of unit-sessions: LEC mean of mice 0.759 (0.730-0.781), PFC 0.759
    (0.594-0.849).
  - In the gate it also fires for 80-85% of OU-drift units, 61-81% of reward- and A-locked units,
    17-35% of place cells, and 3% of noise.
  - Its Wald p ignores rate variance the model does not carry, so any unmodelled slow change
    passes.
- **Minute-scale variability is large in both regions.** log10 of the Fano factor over Poisson,
  median unit, mean of mice:
  - raw: LEC 1.14 (0.96-1.23), about 14x Poisson; PFC 1.35 (1.11-1.61), about 23x;
  - against the held-out prediction: LEC 0.96 (0.79-1.12); PFC 1.24 (1.03-1.50).
  - The PFC excess is not established: log Fano rises with firing rate (LEC rate terciles 0.93,
    1.26, 1.18), and PFC's median rate is higher, 2.5 vs 1.9 Hz. The comparison is not
    rate-matched.
- **Once the held-out model removes locked structure, entry-locked units are not more slowly
  variable than other units.** Median autocorrelation tail over 30-90 s lags, lag-0 normalised:

  | units | LEC raw / residual | PFC raw / residual |
  |---|---|---|
  | Stage 1 entry-locked | 0.0149 / 0.0091 | 0.0081 / 0.0065 |
  | the rest | 0.0082 / 0.0104 | 0.0056 / 0.0068 |

  - The rise in the rest is the expected inflation: the held-out model carries the average of the
    training sessions' drifts (gate K4, 1.04-1.36x).
  - Autocorrelation tau, raw: LEC 34.5 s (28.5-40.8 over mice), PFC 24.7 s (5.0-40.9). Against
    residuals: 41.1 and 25.5 s. All are biased low by the session length: in 20-min sessions the
    fit returns 0.75x of tau at 30 s and 0.45x at 120 s (simulation below).
- **The ramp test: the held-out first-minute decoding is mostly the slow ramps' early slope.**
  - Held-out accuracy of the first 1-min epoch, 20-unit draws as M2. Raw reproduces M2: 0.248
    against M2's 0.247, and the per-session check is exact on 3444 rows.

    | input | LEC minute 1 | LEC minutes 2-17 | PFC minute 1 | PFC minutes 2-17 |
    |---|---|---|---|---|
    | raw counts | 0.248 | 0.077 | 0.260 | 0.074 |
    | entry-locked units / held-out shape-free entry kernel | 0.128 | 0.062 | 0.159 | 0.062 |
    | / held-out slow ramp, tau >= 60 s | 0.136 | 0.065 | 0.178 | 0.065 |
    | / slow ramp, tau >= 120 s | 0.137 | 0.066 | 0.183 | 0.065 |
    | / slow ramp, tau >= 240 s | 0.141 | 0.066 | 0.185 | 0.066 |

    Chance is 0.056-0.059.
  - Fraction of the first-minute advantage over chance removed, against the gate's planted
    populations (F30):

    | | shape-free | ramp >= 60 s | >= 120 s | >= 240 s |
    |---|---|---|---|---|
    | LEC real | 0.62 | 0.58 | 0.58 | 0.55 |
    | LEC planted slow ramps | 0.97 | 1.04 | 1.04 | 1.02 |
    | LEC planted 15 s transients | 0.98 | 0.66 | 0.55 | 0.33 |
    | PFC real | 0.49 | 0.40 | 0.38 | 0.37 |
    | PFC planted 15 s transients | 0.60 | 0.40 | 0.35 | 0.15 |

    - The real removal stays flat as the floor rises, as planted slow ramps do. A fast transient
      falls by half to two thirds.
    - So a response to the white noise at task start, a median 4.6 s after bin 0, is not needed
      to explain the first-minute decoding.
    - In PFC, 0.09-0.12 of the advantage needs the shape-free kernel. That fits its 27% `other`
      entry shapes, but is not resolved.
  - **40% (LEC) and 51% (PFC) of the advantage survives every division.** It sits in units Stage 1
    did not call entry-locked:
    - size-matched non-locked units decode minute 1 held out at 0.164 (LEC, 23 recdays) and
      0.223 (PFC, 11), against chance 0.056;
    - the locked units alone decode it at 0.472 and 0.363.
  - Within-session accuracy barely moves (LEC minutes 2-17: 0.159 raw, 0.151-0.155 divided). The
    division leaves drift alone.
- **Regions (LEC)** — ENTl-deep - SUB/ProS, rate-matched, labels shuffled within recday, 10000
  permutations, 490 units, 4 mice. Three measures differ in all four mice:

  | measure | difference | p | per mouse |
  |---|---|---|---|
  | day time selected | -0.126 | 0.006 | ah10 -0.09, ly05 -0.14, ly06 -0.04, ly07 -0.23 |
  | log10 residual ACF tau | -0.18 | 0.0002 | -0.25, -0.19, -0.08, -0.19 |
  | residual log Fano | -0.20 | 0.0001 | -0.10, -0.20, -0.21, -0.31 |
  | entry time selected | +0.049 | 0.27 | +0.11, +0.29, -0.20, -0.01 |
  | leg time selected | +0.040 | 0.18 | |
  | GLM elapsed time, fraction of sessions | -0.011 | 0.74 | |

  SUB/ProS carries more slow, session-unique variability than ENTl-deep, and its units select
  Tsao's day term more often. The first three survive Holm over these six tests.
- **El-Gaby's rule** moves no LEC number by more than 0.015: day time 0.175 -> 0.170, log Fano
  1.105 -> 1.120.

Predictions in `EVENT_KERNELS_WRITEUP.md` §2.5:

| prediction | held? |
|---|---|
| entry time selected more often than day time | no: 0.166 vs 0.175 (LEC), 0.174 vs 0.190 (PFC). The in-sample day term also takes session-level drift (T7), which the held-out reset test excluded |
| leg time more often than loop time | yes: 0.062 vs 0.014, 0.142 vs 0.029 |
| the recipe catches the strongest locked units and adds drifting ones | yes: precision 0.70, recall 0.33 (LEC entry) |
| leg time catches the slow recovery, not the ~1 s bumps | yes: leg-time tau 15 s (LEC) and 10.6 s (PFC); 9-16% of reward-locked units |
| exponential tau of the order of Stage 2's 320 s | no: 2125 s, the no-baseline fit |
| residual tail falls in entry-locked units, not in others | yes |
| ramp test: advantage falls toward 0.07 under the shape-free division | partly: 0.248 -> 0.128; the slow ramps alone do nearly as much |

### What was done

| file | role |
|---|---|
| `code/unit_timescales.py` (+ mirror, new) | `tsao_rows`, `term_blocks`, `stepwise_poisson` (the rules of MATLAB's `stepwiseglm`, Criterion 'adjrsquared'; Newton IRLS equal to statsmodels' GLM to 1e-15 in adjusted R², 14x faster), `exp_fit`, `tsao_unit`, `kanter_glm`, `acf_tau`, `heldout_mu` (the Stage 2 full model's held-out prediction, shape-free entry factor and slow-ramp factors at floors 60, 120, 240 s) |
| `code/unit_timescales_synthetics.py` (+ mirror, new) | gate populations and rows T1-T8, K1-K5c; task offsets and pure exponentials planted locally, `event_kernels_synthetics.py` untouched |
| `code/run_unit_timescales.py` (+ mirror, new) | one job per recday; `--merge` with two equivalence checks; `--gate` |
| `code/population_drift.py` (+ mirror) | `log_fano` and `gauss_smooth` (FFT, equal to `gaussian_filter1d` to 2e-15, 6x faster); the part-2 functions (`POPULATION_DRIFT.md`) |
| `code/event_kernels_plots.py` (+ mirror) | F22-F30 and their tables, `region_contrast_df`, `unit_measures` |
| `code/check_mirror_parity.py` | the three new pairs |
| `sbatch_files/submit_event_kernels.sh` | runners `timescales` and `drift2` |
| `code/event_kernels_M3_part2.ipynb` | load mode. Figures in `data/figures/{event_kernels,population_drift}/M3_part2/` |

- **Rows.**
  - Tsao: 500 ms bins over each session's core, first A to the end of the last trial.
  - Kanter: 0.5 s bins over full sessions from bin 0. The residual versions use Pearson residuals
    (y - mu) / sqrt(mu) against `heldout_mu`.
- **Units:** all, in ABCD sessions with >= 5 trials. The entry-locked set is Stage 1's (V3, >= 5
  trials).
- **Runs:** 25 + 25 jobs, every one `EXIT=0`. PFC me08_12092021_13092021 was skipped: no tracking.
- **Equivalences at the merge:**
  - `heldout_mu`'s held-out deviance equals Stage 2's `dev_np`: LEC 3686 rows (2356 units), max
    relative difference 2.7e-14; PFC 2060 rows (1126 units), 1.2e-14.
  - The raw ramp-test kNN equals the M2 drift arm at >= 5 trials: LEC 3444 session rows, max
    difference 0.0. PFC has no M2 arm at this threshold.

### Gates (planted on the gate recdays' real behaviour, through the runner)

Final runs: 3766905 (LEC) and 3766907 (PFC), `EXIT=0`. The CSVs are in
`data/event_kernels/timescales/runner_gate/<ds>_full_20261014_1522*/`.

| row | planted | LEC | PFC |
|---|---|---|---|
| T1 | entry ramps, tau 300 s: entry time selected | 1.00 | 1.00 |
| T1 | tau 60 s / 120 s (reported) | 0.00 / 0.50 | 0.50 / 1.00 |
| T2 | day trends: day time | 1.00 | 1.00 |
| T3 | reward ramps: leg time or place / leg time alone | 1.00 / 1.00 | 1.00 / 1.00 |
| T3b | A ramps: loop time | 1.00 | 1.00 |
| T4 | place cells: place / any time term | 1.00 / 0.00 | 1.00 / 0.00 |
| T5 | task offsets: task | 1.00 | 1.00 |
| T6 | noise: any term | 0.00 | 0.00 |
| T7 | OU drift called a time cell (reported) | 0.33 | 0.58 |
| T8 | pure exponentials: tau within 1.25x, Tsao / Kanter | 1.00 / 1.00 | 1.00 / 1.00 |
| K1 | entry ramps: GLM p < 0.05, unit-sessions | 0.95 | 0.95 |
| K2 | noise: GLM p < 0.05 / median log Fano | 0.025 / -0.010 | 0.030 / -0.044 |
| K3 | OU drift, ACF tau within 2x at 30 s (median) | 1.00 (26 s) | 1.00 (20 s) |
| K3 | at 120 s / 300 s (reported, median) | 0.75 (72 s) / 0.50 (144 s) | 0.50 (66 s) / 0.00 (77 s) |
| K4 | residual / raw ACF tail: ramps / drift | 0.00 / 1.36 | 0.00 / 1.04 |
| K5 | slow ramps, held out: raw / shape-free / ramp 60 / ramp 120 | 0.277 / 0.054 / 0.054 / 0.054 | 0.308 / 0.060 / 0.070 / 0.070 |
| K5b | 15 s transient, first-minute advantage: raw / shape-free / ramp 240 | 0.382 / 0.007 / 0.257 | 0.370 / 0.148 / 0.315 |
| K5c | drift, held out raw / divided; within-session change | 0.069 / 0.053; 0.044 | 0.051 / 0.017; 0.062 |

**Five rows were revised before the final runs.** Each was specified before the recipe was
measured. None hid a code error. Reasons are in the gate module's docstring.

1. **T1, after the quick gate (LEC 3766694).** One linear term cannot reach adjusted R² +0.01 for a
   ramp mostly over in 2-3 min of a 20-min session. It found 60 s ramps in 0/2. The row now holds
   at 300 s, the real median, and the floor is reported.
2. **K3, before any gate.** Kanter's 1-90 s fit is biased low in finite sessions. Simulated on OU
   log-rate Poisson trains, 2 Hz, 40 trains:

   | session | tau 30 s | 120 s | 300 s |
   |---|---|---|---|
   | 10 min (Kanter) | 0.69x | 0.31x | 0.22x |
   | 20 min (ours) | 0.75x | 0.45x | 0.32x |
   | 60 min | 0.92x | 0.78x | 0.52x |
   | 240 min | 0.95x | 0.86x | 0.83x |

   The row now holds at 30 s only.
3. **K4, after the quick gate.** Drift's ratio was 1.30 against a 1.25 limit. The held-out model
   carries the average of the training sessions' independent drifts, about 1/(n-1) of their
   variance. The window is now 0.75-1.5.
4. **K5b, after the quick gate.** The shape-free kernel's first lag bin is 10 s, so a 15 s
   transient is removed only at that resolution. The row now asks for less than half left.
5. **K5b again, after the full gate (LEC 3766748).** A slow ramp with a 60 s floor still removed 65%
   of a transient's first-minute advantage, because it weights minute 1 most. Input 3 now runs at
   floors 60, 120 and 240 s, and the real data are read against that envelope (F30).

### Measured

The outcome above carries the numbers. Tables: `table_tsao`, `table_tsao_tau`, `table_kanter`,
`table_ramp_test`, `table_ramp_envelope` and `table_m3p2_regions` in `event_kernels_plots.py`, run
in the notebook.

### What was NOT established

- **That Tsao-style time cells are clocks.** The recipe is in sample. Stage 1's held-out test is
  the evidence for a locked clock.
- **Time constants comparable with the papers'.** Tsao's and Kanter's exponentials have no
  baseline (2125-2584 s here), and ACF taus are biased low by session length. Stage 2's held-out
  ramp taus remain ours.
- **LEC vs PFC.** No contrast was tested at mouse level. The Fano difference is not rate-matched.
- **What carries the 40-51% of first-minute decoding left after division.** It sits in units
  Stage 1 did not call entry-locked. Sub-threshold entry kernels are the obvious candidate; not
  tested.
- **The regional differences** rest on 4 mice. Probe depth and sorting quality differ between
  ENTl-deep and SUB/ProS and were not examined.
- **Which clock.** Placement is not recorded, and task start vs recording start was not tested.
  See the handoff, §8.

### Commands

```
python code/check_mirror_parity.py                                          # must exit 0
for ds in lec pfc; do t=code; [ $ds = pfc ] && t=mFC_data/code
  sbatch --time=0-03:00 --job-name=ut_gate_$ds sbatch_files/event_kernels.sbatch $t/run_unit_timescales.py --gate --dataset $ds
done
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner timescales --min-trials 5 --after <gate job>
bash sbatch_files/submit_event_kernels.sh --dataset pfc --runner timescales --min-trials 5 --after <gate job>
python code/run_unit_timescales.py --dataset lec --merge --min-trials 5
python mFC_data/code/run_unit_timescales.py --dataset pfc --merge --min-trials 5
sbatch --job-name=ek_nb_m3p2 --time=0-02:00 --mem=48G --cpus-per-task=4 sbatch_files/event_kernels.sbatch -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 code/event_kernels_M3_part2.ipynb
```

Outputs:

- shards: `data/event_kernels/timescales/shards/<ds>_kin-strict_mt5_np100/`; superseded first
  runs in `shards_superseded/pre_gate_fix_2026-10-14/`;
- merged: `data/event_kernels/timescales/timescales_<ds>_kin-strict_mt5_np100.pkl` with
  `_tsao`, `_kanter`, `_heldout`, `_ramp`, `_summary` and `_equivalence` CSVs.

## 2026-10-13 — M5's test brought forward: entry clock vs leaky reward count

User decision, 2026-10-13: commit M3 part 1 (90c10d0, pushed), then run this test before M3 part 2.

**Outcome.**

- **In LEC, the slow entry kernel is a clock, not a leaky count of rewards.** Over the 972
  entry-locked units:
  - the entry ramp exp(-t/tau) predicts held-out sessions better than the best leaky reward count
    in 79% (per mouse 71-88%, 5 mice);
  - the ramp adds unique held-out gain beyond the count in 76%, a median 0.30 of the shape-free
    kernel's gain;
  - the count adds none beyond the ramp: positive in 43%, median -0.003. That is the chance
    split for a quantity with no information.
- **PFC leans the same way, less decisively** (414 units):
  - the ramp wins in 60% (per mouse 38-76%, 7 mice);
  - its unique gain is positive in 51%, median 0.05;
  - the count's unique gain sits at chance: 46%, median 0.002.
- **The test separates the two on real behaviour, at strong planted effects.** The gate scores
  1.0 for both kinds at tau <= 600 s in both datasets, although ramp and count correlate at
  |r| 0.88-0.99 there.
- **Stage 1 could not separate them:** it calls 100% of planted leaky counts entry-locked. Only
  this comparison can.

### What was done

| file | change |
|---|---|
| `event_kernels.py` (+ mirror) | `leaky_trace` and `leaky_bank`: the leaky count (1 - a) sum of a^(t - t_k), scaled so 4 rewards/min reads 1. `stage2_unit(extra=, joint=)` adds families profiled like ramp and bump, plus a joint refit. Defaults unchanged. |
| `event_kernels_synthetics.py` (+ mirror) | kind `leaky_count` and `leaky_specs()`. `unit_specs` is untouched. |
| `run_event_kernels.py` (+ mirror) | `--leaky` per recday, `--gate-leaky`, `--leaky --merge`, which also checks equivalence with the Stage 2 merge. Per-recday ramp-count collinearity. |
| `event_kernels_plots.py` (+ mirror) | F19-F21, `table_leaky` |
| `event_kernels_leaky_count.ipynb` | load mode. Figures in `data/figures/event_kernels/M5_leaky/`. |

- **Units:** the `ev_entry` passers of the V3 Stage 1 table at >= 5 trials (LEC 972, PFC 414),
  plus the same 10 seeded null units per recday as Stage 2 (LEC 250, PFC 219).
- **Fit:** `stage2_unit(D, u, 'entry', extra={'count': (taus, leaky_bank)}, joint=('ramp', 'count'))`.
  Both families use the entry tau grid, 5-7241 s, 2 per octave, chosen on training sessions.
  Scoring is held-out Poisson deviance with only the session intercept refitted.
- **Runs:** 25 + 25 jobs, `EXIT=0`; PFC me08_12092021_13092021 skipped (no Stage 1 rows).
- **Real-data equivalence with Stage 2:** `dev_null`, `dev_np`, `dev_ramp` and `dev_bump` agree to
  4e-15 (LEC, 1222 units) and 3e-14 relative (PFC, 633). tau, peak and shape are identical.

### Gates (jobs 3764521-26, all `EXIT=0`)

- `--gate`, the Stage 1 M1 reproduction, still 19/19: the synthetics change left the M1 units
  alone.
- `--gate-stage2`: within 3.5e-14 (LEC) and 1.5e-13 (PFC) relative of M1. The `stage2_unit`
  default is unchanged.
- `--gate-leaky`, planted on the gate recday's real behaviour at >= 5 trials:

  | row | planted | LEC | PFC |
  |---|---|---|---|
  | L1 | leaky counts, tau <= 600 s, called counts | 1.00 | 1.00 |
  | L2 | entry ramps, tau <= 600 s, called ramps | 1.00 | 1.00 |
  | L1/L2 | each rule on the other kind | 0.00 / 0.00 | 0.00 / 0.00 |
  | L3 | tau 1200 s: counts / ramps called right | 1.00 / 0.75 | 1.00 / 1.00 |
  | L4 | noise: count beats ramp | 0.55 | 0.55 |
  | L5 | Stage 1 `ev_entry` locked: counts / ramps / noise | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.05 |
  | L6 | winning family's tau within 2x: counts / ramps | 0.81 / 1.00 | 0.00 / 1.00 |
  | info | ramp-count correlation, tau 300 / 1200 s | -0.97 / -0.99 | -0.88 / -0.94 |

  PFC's planted counts win as counts but their tau is not recovered. The family call does not
  depend on it.

### Measured (>= 5 trials; the preference is in units of the shape-free kernel's held-out gain)

| units | n | count beats ramp | median (dev ramp - dev count) / gain | ramp unique gain > 0 | count unique gain > 0 |
|---|---|---|---|---|---|
| LEC, entry-locked | 972 | 0.208 | -0.31 | 0.76 (median 0.30) | 0.43 (median -0.003) |
| LEC, null sample | 250 | 0.492 | -0.73 | 0.20 | 0.12 |
| PFC, entry-locked | 414 | 0.401 | -0.05 | 0.51 (median 0.05) | 0.46 (median 0.002) |
| PFC, null sample | 219 | 0.594 | +0.04 | 0.18 | 0.18 |

- **Per mouse,** the fraction of entry-locked units where the count wins (recday mean, then mouse
  mean): LEC 0.12-0.29 (5 mice, mean 0.20); PFC 0.24-0.62 (7 mice, mean 0.42). ah04, me08, me10
  and me11 sit near 0.5.
- **Selection.** Conditioning on "ramp-like" (Stage 2's ramp recovers >= 0.5 of the gain) biases
  toward the ramp; null ramp-like units also prefer it (LEC 0.14). So the table uses all
  entry-locked units.
- **Neutral selection,** where either family recovers >= 0.5 of the gain:
  - LEC: 846 units, the count wins in 0.17, ramp unique positive 0.82, count unique 0.46;
  - PFC: 244 units, 0.31, 0.67 and 0.49.
- **Matched time constants** (F20), where both families fit: the count needs a longer tau than
  the ramp, often the grid maximum (7241 s), effectively a cumulative count.
- **Collinearity on real behaviour,** ramp vs count per recday:

  | tau | LEC: median r (range) | PFC: median r (range) |
  |---|---|---|
  | 300 s | -0.81 (-0.97 to -0.66) | -0.88 (-0.95 to -0.65) |
  | 1200 s | -0.89 (-0.99 to -0.83) | -0.94 (-0.99 to -0.87) |

  Real behaviour is no more collinear than the gate recday's, where the test separated perfectly.

### What was NOT established

- **Power at real effect sizes.** Planted units modulate the rate about 3x; real entry kernels
  are far weaker (median locked cpd about 0.005). Per-unit calls are therefore noisy (noise
  units split 0.55). The robust evidence is the asymmetry of unique gains, not any single call.
- **Which clock.** The ramp is time since recording start. The mouse is already in the maze then
  (M1), so placement, handling or another monotonic session variable is not excluded. The reset
  test (M2) excludes a day-long trend for 72-82% of entry-locked units. Satiety, if it tracks
  rewards, is what the cumulative count tests, and it loses.
- **Trial counts and other event counts.** A leaky count of trials (A) is nearly collinear with
  the reward count; it was not tested separately. Neither were pokes (LEC) or distance run.
- **PFC.** About half the units carry no unique information from either family. The clock
  preference there is weaker and varies by mouse; it is not established per mouse.
- **No LEC vs PFC contrast was tested.**

### Commands

```
python code/check_mirror_parity.py                                          # must exit 0
for ds in lec pfc; do t=code; [ $ds = pfc ] && t=mFC_data/code
  for mode in gate gate-stage2 gate-leaky; do
    sbatch --time=0-01:30 --job-name=ek_${mode}_$ds sbatch_files/event_kernels.sbatch $t/run_event_kernels.py --$mode --dataset $ds --min-trials 5
  done; done
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner leaky --min-trials 5 --after <gate job ids>
bash sbatch_files/submit_event_kernels.sh --dataset pfc --runner leaky --min-trials 5 --after <gate job ids>
python code/run_event_kernels.py --dataset lec --leaky --merge --min-trials 5
python mFC_data/code/run_event_kernels.py --dataset pfc --leaky --merge --min-trials 5
sbatch --job-name=ek_nb_leaky --time=0-01:00 --mem=32G --cpus-per-task=2 sbatch_files/event_kernels.sbatch -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 code/event_kernels_leaky_count.ipynb
```

## 2026-10-12, night — M3 part 1: Stage 2 shapes and time constants (>= 5 trials, both datasets)

**Outcome.**

- **Entry kernels are mostly minutes-scale ramps, not first-minute transients.** This corrects the
  M2 reading below.
  - Of the entry-locked units, mean of mice: ramps 0.58 in LEC (up 0.37, down 0.21) and 0.42 in
    PFC (0.25, 0.17).
  - Broad bumps (k = 2) peak at about 15 min: 0.32 in LEC and 0.19 in PFC.
  - A ramp also fits 82% of LEC entry bumps (81% in PFC), recovering >= 0.5 of the gain; the
    median is 0.96 against the bump's 1.13. Over a 20-min session they are ramp-like.
  - So 85% of entry-locked LEC units are ramps or ramp-like bumps (pooled units). In PFC it is
    54%, where 27% are `other`.
  - The ramp time constant is a median 320 s in both datasets. The middle half is 190-910 s
    (LEC, 567 units) and 160-640 s (PFC, 158 units). 17% are slower than the session.
  - Two thirds of LEC entry ramps rise over the session (59% in PFC).
- **Reward kernels are brief and alike in both datasets.**
  - Bumps peaking a median 1 s after the reward: 0.50 of locked units (LEC) and 0.37 (PFC).
  - Ramps with tau a median 2 s (LEC) and 1.4 s (PFC), about 80% rising: the post-reward dip
    recovering.
- **Reward-A calls are the weakest.** For 31% (LEC) and 44% (PFC) of reward-A passers, the
  Poisson shape-free kernel gains nothing on held-out sessions.
- **No regional difference** in entry or reward time constants: LEC entry-ramp tau medians are
  273-593 s across the five regions, reward 1.6-2 s.

### Correction to the M2 entry (dated, not rewritten)

The M2 entry reads 62-78% of entry-locked LEC units peaking "within a minute of recording start"
as "a first-minute transient, not a slow ramp".

- The measurement stands. The reading does not.
- The shape-free kernel is relative to a late reference bin. Against that bin, a 5-min ramp is
  most different in the first minute, so its largest |beta| sits there whether it rises or falls.
- Stage 2 shows the shapes are ramps with tau of minutes.
- The same caution applies to `POPULATION_DRIFT.md`'s reading of first-minute decoding (noted
  there).

### What was done

- **Stage 1 at >= 5 trials on the V3 runner** (previous entry):
  - LEC: 50 jobs, `EXIT=0`, 25 recdays, 2851 units, 164 sessions;
  - PFC: 50 jobs, `EXIT=0`, 24 recdays, 1229 units, 180 sessions; me08_12092021_13092021 has no
    tracking.
  - No shard has a held-out leak through a tested column.
- **Fraction locked at >= 5 trials**, mean of mice with the range over mice. The 5-trial arm lifts
  ly05 and ah08, so the LEC range narrows from 10 trials.

  | block | LEC (5) | PFC (7) |
  |---|---|---|
  | entry | 0.342 (0.259-0.426) | 0.322 (0.160-0.533) |
  | every reward | 0.587 (0.504-0.665) | 0.609 (0.326-0.800) |
  | reward A | 0.109 (0.099-0.132) | 0.174 (0.098-0.304) |
  | A + reward | 0.436 (0.339-0.541) | 0.502 (0.214-0.742) |

- **Stage 2** (`run_event_kernels.py --stage2`, `ek.stage2_unit`, gated as M1 G10 and by
  `--gate-stage2`):
  - one job per recday and event: 150 jobs plus 2 smoke jobs, all `EXIT=0`;
  - about 10 s per unit;
  - the units locked to `ev_entry`, `ev_A` and `ev_reward` in the anchor-`entry` fit, plus 10
    seeded non-locked units per recday and event as the null sample.

  | dataset | entry: locked + null | reward A: locked + null | every reward: locked + null |
  |---|---|---|---|
  | LEC | 972 + 250 | 308 + 250 | 1656 + 250 |
  | PFC | 414 + 219 | 194 + 225 | 805 + 203 |

- **Notebook:** `code/event_kernels_M3_stage2.ipynb`. Figures F15-F18 are in
  `data/figures/event_kernels/M3/`.

### Measured

Shape class of locked units (F15), mean of mice:

| event | dataset | ramp up | ramp down | bump | other | none | null sample: none |
|---|---|---|---|---|---|---|---|
| entry | LEC (5 mice) | 0.373 | 0.206 | 0.321 | 0.032 | 0.068 | 0.728 |
| entry | PFC (7) | 0.250 | 0.170 | 0.187 | 0.273 | 0.120 | 0.678 |
| reward A | LEC | 0.151 | 0.140 | 0.246 | 0.156 | 0.307 | 0.864 |
| reward A | PFC | 0.051 | 0.089 | 0.271 | 0.149 | 0.440 | 0.881 |
| every reward | LEC | 0.323 | 0.083 | 0.496 | 0.064 | 0.034 | 0.504 |
| every reward | PFC | 0.325 | 0.168 | 0.372 | 0.118 | 0.016 | 0.455 |

Fraction of units whose shape-free kernel gains held-out deviance:

| event | LEC, locked / null | PFC, locked / null |
|---|---|---|
| entry | 0.94 / 0.27 | 0.90 / 0.34 |
| reward A | 0.67 / 0.14 | 0.61 / 0.11 |
| every reward | 0.97 / 0.50 | 0.98 / 0.55 |

- Reward structure is widespread: half of the non-locked units gain from a reward kernel.
- The class is therefore a description of the locked units, not a test; the test is Stage 1.

Ramp time constants (F16), for ramp-classed locked units:

| event | dataset | n | median tau | middle half | slower than the session | at the grid edge |
|---|---|---|---|---|---|---|
| entry | LEC | 567 | 320 s | 193-905 s | 0.17 | 0.11 |
| entry | PFC | 158 | 320 s | 160-640 s | 0.17 | 0.10 |
| reward A | LEC | 88 | 23 s | 2.7-190 s | 0 | 0.18 |
| reward A | PFC | 43 | 5.7 s | 2.8-14 s | 0 | 0.09 |
| every reward | LEC | 676 | 2.0 s | 1.2-2.8 s | 0 | 0.04 |
| every reward | PFC | 359 | 1.4 s | 0.7-2.0 s | 0 | 0.07 |

- **Per-mouse median entry tau:** LEC 273-453 s; PFC 97-866 s, where ah03 and me10 have few
  units.
- **Regions** (LEC, off-boundary), entry-ramp median tau:

  | region | median tau | mice | units |
  |---|---|---|---|
  | ENTl-deep | 320 s | 5 | 167 |
  | SUB/ProS | 320 s | 4 | 109 |
  | ENTm | 320 s | 3 | 54 |
  | ENTl-sup | 273 s | 3 | 74 |
  | CA1/HPF | 593 s | 2 | 38 |

  Reward: 1.6-2 s in every region. Reward-A ramps in ENTl-sup are slower (237 s, 16 units, 3 mice),
  a single small cell.

Bumps (F17), median peak lag:

| event | LEC | PFC |
|---|---|---|
| entry | 905 s, k 2 (315 units); 2.5% peak within 1 min | 905 s (83 units); 3.6% within 1 min |
| every reward | 1.0 s (812 units); 99.8% within 1 min | 1.0 s (353 units) |
| reward A | 11 s (76 units) | 5.7 s (58 units) |

F18 shows the best ramp and bump fit per event against the Stage 1 kernel. One recday's reward
betas are not identified at >= 5 trials: ly05_20250620_20250623, whose all-session null space runs
through the reward block. That relation holds on held-out rows, so cpd is unaffected. F18 skips
units without at least 5 finite betas.

### What was NOT established

- **Time constants beyond the session.** 17% of entry ramps sit at tau >= 30 min, which a 20-min
  session cannot pin (M1: recovered at 0.13-3.43x of truth). The rest are inside the M1 envelope
  for the selection used here, but they are selection-conditioned.
- **Entry clock vs a leaky reward count.** Unchanged since M2: the M5 test. Rising entry ramps
  with tau of minutes are exactly what a leaky count of rewards would look like.
- **Broad entry bumps (peak about 15 min, k = 2).** A ramp fits 82% of them nearly as well. The
  rest may turn over near the session's end, or reflect the post-task segment. Not separated.
- **PFC's 27% `other` entry kernels:** neither shape recovers half the gain. Not characterised.
- **LEC vs PFC.** No dataset contrast was tested. The class fractions and the tau spectra overlap
  at mouse level.
- **Kernels at short lags.** They are evaluated at the block-mean lag, an approximation for
  tau <= 0.5 s; this applies to reward ramps.
- **M3 part 2** (Tsao's stepwise table, Kanter part 2) waits for the user.

### Commands

```
python code/check_mirror_parity.py                                          # must exit 0
python code/event_kernels_pin.py --against HEAD                             # engine default pinned
sbatch --time=0-01:30 --job-name=ek_rgate_lec sbatch_files/event_kernels.sbatch code/run_event_kernels.py --gate --dataset lec
sbatch --time=0-01:30 --job-name=ek_s2gate_lec sbatch_files/event_kernels.sbatch code/run_event_kernels.py --gate-stage2 --dataset lec
# (the same two for PFC from mFC_data/code/)
bash sbatch_files/submit_event_kernels.sh --dataset lec --min-trials 5
bash sbatch_files/submit_event_kernels.sh --dataset pfc --min-trials 5
python code/run_event_kernels.py --dataset lec --min-trials 5 --merge
python mFC_data/code/run_event_kernels.py --dataset pfc --min-trials 5 --merge
python code/run_event_kernels.py --compare data/event_kernels/stage1/stage1_lec_kin-strict_mt5_np100.pkl data/event_kernels/stage1/stage1_lec_kin-strict_mt5_np100.v1.pkl
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner stage2 --min-trials 5
bash sbatch_files/submit_event_kernels.sh --dataset pfc --runner stage2 --min-trials 5
python code/run_event_kernels.py --dataset lec --stage2 --merge --min-trials 5
python mFC_data/code/run_event_kernels.py --dataset pfc --stage2 --merge --min-trials 5
sbatch --job-name=ek_nb_m3 --time=0-01:00 --mem=32G --cpus-per-task=2 sbatch_files/event_kernels.sbatch -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 code/event_kernels_M3_stage2.ipynb
```

## 2026-10-12, evening — runner V3: the rank criterion and the first-A `pre` column

User decisions after M2:

- commit and push M2 (d1b81b2);
- revise the runner before Stage 2;
- run Stage 2 at >= 5 trials in both datasets;
- stop after Stage 2.

**Outcome.**

- The first-A artefact from M2 is gone.
- The new rank check finds no held-out leak through a tested column in any 5-trial shard, under
  either anchor, in LEC or PFC.
- Entry-anchor results are unchanged on real data: cpd agrees to 3e-15, and p and every locked
  call are identical.

### What changed

| file | change |
|---|---|
| `event_kernels.py` (+ mirror) | `build_design(first_A_pre=True)`. `False` drops the first-A block's pre-event column, which equals the A block's pre column, plus `kin_missing:pre_A` under `strict`. The default is unchanged. |
| `event_kernels_pin.py` | Also pins the `strict` default under both anchors, and checks that `first_A_pre=False` removes exactly `ev_first_A:pre`. Passes against HEAD. 57eac39 predates `strict`, so only its `full` rows apply. |
| `run_event_kernels.py` (+ mirror) | Anchor `first_A` fits with `first_A_pre=False` (stamped `first_A_anchor_pre`). `rank_check` sets aside columns all-zero in training, which take beta 0 in every model. It then flags only null-space directions with a held-out component that touch a tested column (`tested_leak`), and matches duplicates with a 1e-9 tolerance. New `--stage2`, `--gate-stage2` and `--compare` modes. |
| `submit_event_kernels.sh` | `--runner stage2` |

### Gates (jobs 3755246-49)

- **Stage 1, 19/19 in each dataset.**
  - The M1 exactness rows still hold: cpd and p max |diff| = 0.
  - New first-A rows, the anchor never gated in M1. On planted units, noise is locked to
    `ev_first_A` in 0.00 (binomial p 1.00), entry ramps with tau <= 30 min in 1.0, and
    A / reward / bump units in 1.0 through A, reward or the joint.
  - `first_A_pre` False vs True: every block's cpd agrees to <= 2e-15 and p is identical, except
    the A + reward joint (LEC: up to 0.019, 8 of 80 planted calls change; PFC: 3e-15).
- **The first pass failed this row.** It assumed one column space. The A + reward joint's reduced
  model kept the pre-A offset through `ev_first_A:pre`; without that column, the joint carries the
  pre-A offset, as it always has under anchor `entry`. The check became per block, with the joint
  reported.
- **Stage 2 gate** (`--gate-stage2`): the quick planted units through the Stage 2 shard and merge
  path reproduce the M1 pickle's `stage2` table.
  - Shape classes are identical, 17/17.
  - Max relative difference: 1.4e-13 (LEC) and 1.5e-13 (PFC).
  - The first pass demanded an exact 0 and failed on 1.5e-11 absolute, on deviance sums of
    about 1e4: M1 ran on another node. The test is now relative at 1e-9 and requires equal NaN
    patterns.
- **The rank criterion was tested both ways on real designs.**
  - ah10_20250618_20250619 first-A with the `pre` column: `tested_leak`, leak 0.11 through
    `ev_first_A:pre`, `ev_A:pre` and `ev_reward:pre`.
  - Without it: none, 9e-16.
  - PFC me11 first-A with it: no leak (6e-16). These were near-duplicates the M2 flag called
    non-duplicates.
  - LEC >= 5 ly05_20250620_20250623 entry, with a relation through the whole reward block: no
    leak (4e-16).

### Real-data equivalence: LEC >= 5, V3 against the V2 merge

Command: `run_event_kernels.py --compare stage1_lec_kin-strict_mt5_np100.pkl ...v1.pkl`.

| fits | max \|cpd diff\| | locked calls changed (of 2851 units) | where |
|---|---|---|---|
| anchor `entry`, every block | 2.9e-15 (p identical) | 0 | floating point across nodes |
| first-A, blocks other than the A + reward joint | <= 1.2e-3 | 0-4 per block | only the 8 recdays with single-session untracked pre-A rows |
| first-A, A + reward joint | 0.156 | 351: 237 gained, 114 lost; 1260 -> 1383 locked | the joint now carries the pre-A offset |

- The old flag also listed ly05_20250618_20250619 and ly05_20250620_20250623. Neither changes:
  they were false alarms.
- No figure or table reads the first-A anchor's A + reward joint. The anchor comparison (F7) uses
  `ev_first_A`, and Stage 2 uses anchor `entry`.
- The V2 merge is kept as `stage1_lec_kin-strict_mt5_np100.pkl`, its shards under
  `shards_superseded/runner_v2_2026-10-12/`. V3 is `.v1`.

## 2026-10-12 — M2: Stage 1 and the reset test on real data

**Outcome.**

- **Event-locked units are common in both datasets.** Every event block, in every mouse, exceeds
  the exact null rate of p < 0.05 at n_perm 100 (5/101 = 0.050). Units locked (Freedman-Lane
  p < 0.05 and held-out cpd > 0), mean of mice:

  | block | LEC, 5 mice | PFC, 7 mice |
  |---|---|---|
  | every reward | 0.51 | 0.61 |
  | entry (recording start) | 0.29 | 0.34 |
  | reward A | 0.12 | 0.16 |

- **The entry kernel is mostly a first-minute transient, not a slow ramp.** Of the entry-locked
  units, 62-78% per LEC mouse have their largest kernel weight within a minute of recording
  start; none peak in the bin beyond 25 min. PFC is mixed: 18-75% per mouse.
  - This is the per-neuron side of the drift result (`POPULATION_DRIFT.md`): held-out sessions
    decode only the first minute.
  - Tsao's slow session ramps are not the main entry-locked pattern here.
- **Entry calls are not day-long trends.** On held-out sessions, the entry kernel predicts
  better than a linear trend in hours since the day began in 82% of entry-locked LEC units (mean
  of 5 mice, 0.67-0.95) and 72% in PFC (7 mice, 0.42-0.94).
- **Kinematics.** In LEC, the event block beats the fast speed bank in 68% of entry- and
  reward-locked units. In PFC the two are level: 55-59%, with similar median cpd. A PFC event
  call is weaker evidence for an event clock.
- **Regions.** No regional difference is established. ENTl-deep is more reward-locked than
  SUB/ProS: +0.13 in fraction, rate-matched, 4 mice, p 0.023, two of the four mice near zero. It
  is a candidate.
- **The 5-trial arm** changes the weakest mice, not the picture. ly05's reward-locked fraction
  rises from 0.27 to 0.51, so per-mouse differences at 10 trials partly reflect how many sessions
  a recday has.

### What was done

| file | role |
|---|---|
| `code/run_event_kernels.py` (+ mirror) | one job per recday and anchor. Stage 1 (`ek.stage1`), the reset test on anchor `entry` (`ek.reset_test`), all-session betas and raw lag-bin rates for figures, a rank report. Stamped shards; `--merge` to long tables; `--gate`. |
| `code/run_population_drift.py` (+ mirror) | the Kanter measures (`POPULATION_DRIFT.md`) |
| `code/event_kernels_plots.py` (+ mirror) | figures F1-F14 and the tables quoted below |
| `sbatch_files/submit_event_kernels.sh` | submission. Recdays come from files, never a pickle. Runs the parity check. `--after` gates the jobs on others. Partition `cpu_backfill`. |
| `sbatch_files/event_kernels.sbatch` | partition `gpu_lowp` -> `cpu_backfill` (the 2026-10-09 site warning) |
| `code/event_kernels_M2_comparison.ipynb`, `code/LEC_event_kernels.ipynb`, `mFC_data/code/PFC_event_kernels.ipynb` | `RUN_MODE = 'load'`, executed through sbatch |

- **Design:** `kin='strict'` (user decision, 2026-10-12); ABCD sessions with >= 10 trials;
  leave one session out; Freedman-Lane, n_perm 100, seed 0.
- **Two anchors on identical rows:**
  - `entry` (recording start): events entry, A, reward, plus the reset test;
  - `first_A`: events first A, A, reward.
- **Arms:** LEC >= 10, PFC >= 10, and LEC >= 5 as a sensitivity arm.
- **Engine:** no change; `event_kernels.py` is byte-identical to 2c7f99c.

### Runner gates (planted units through the runner)

- **Stage 1.** The quick planted units on the gate recday go through `fit_recday`, the shard,
  `merge_dir` and the long table. Every check passes, 11/11 in each dataset:
  - per unit and block, cpd and p equal the stage-1 values in the M1 pickle
    `synthetics_<ds>_<rd>_quick_kin-strict.pkl`: max |runner - gate| = 0;
  - the merged locked table, the fraction locked per kind and block (G1-G4's inputs) and the
    reset calls are identical;
  - the merge refuses a changed stamp field, a duplicate recday and a short recday count;
  - a recday without 2 sessions writes `skipped`.
- **Gate runs.** Three passes:
  - jobs 3750326/28;
  - 3750341/43, after the rank report and runner hash were added;
  - 3751709/10, after the untracked-recday skip below.
- **Drift gate:** `POPULATION_DRIFT.md`.

### Runs

| arm | jobs | fitted recdays | units | sessions | skipped |
|---|---|---|---|---|---|
| LEC, >= 10 trials | 50, `EXIT=0` | 24 | 2742 | 135 | ly05_20250618_20250619: 1 session |
| PFC, >= 10 trials | 50, `EXIT=0` | 24 | 1229 | 171 | me08_12092021_13092021: no tracked position (below) |
| LEC, >= 5 trials | 50 + 1 resubmitted, `EXIT=0` | 25 | 2851 | 164 | none |

- Stage 1 per recday and anchor: 1-20 min at 8 CPUs, median 4.8 min (LEC).
- Median held-out r2 per recday, median of recdays: 0.051 (LEC) and 0.040 (PFC).

Three incidents, each resolved inside M2:

1. **Two jobs refused on a transient mirror drift.**
   - What happened: `cpu_backfill` preempts and requeues jobs, and a requeued job re-runs the
     parity check. Two requeued jobs started while `event_kernels_plots.py` had been edited in
     `code/` and not yet copied to its mirror.
   - Fix: resubmitted (LEC 5-trial ah10_20250613_20250615 entry). The PFC one was superseded by
     incident 2.
   - Rule: while jobs can requeue, write both copies of a registered file in one step.
2. **PFC me08_12092021_13092021 crashed in `build_design`.**
   - Cause: its XY is unusable (`xy_ok` False in all 7 sessions), so no speed sample is finite and
     the decile edges are `None`. M1 never ran this recday through the design.
   - Fix: the runner now skips a recday with no tracked position in any analysis session. A
     rate-only fit would lack the kinematic control that kept speed out of the event blocks in
     M1 (G5).
   - The rule changed the runner, whose sha1 is in every stamp. So the whole PFC arm was rerun
     on the revised runner after the last LEC job ended (jobs 3751711-60). The first PFC shards
     are in `shards_superseded/runner_pre_untracked_skip_2026-10-12/`.
   - The LEC arms ran on the runner before this rule. It cannot fire on LEC, where every
     analysis session is tracked.
3. **A smoke check in the plan was mis-specified.**
   - The plan asked for `fold_rank_min == n_cols`. The M1 design already ran at 147 of 149
     columns on the gate recday, and the smoke run showed why: the `pre` columns of A, reward
     and first A mark the same rows, before the first A.
   - The runner now reports each fold's null space and sets betas of unidentified columns to
     NaN. The analysis of what that null space does is below.

### Rank: what the null space does to held-out predictions

A rank deficit changes a held-out cpd only if the held-out rows have a component along the
training fold's null space; otherwise the min-norm choice never reaches a prediction. Tested
directly with SVD on 5 designs, and characterised for every shard by its null-space columns:

| fits | null space | held-out component |
|---|---|---|
| every `entry` fit, LEC and PFC, both thresholds, except the last row | duplicated `pre` columns; an all-zero `kin_missing:gap` or `:post` when the only session with such rows is held out | 0 for the duplicates (5e-16 on PFC); the zero nuisance columns sit in both models, so they do not enter cpd |
| PFC `first_A` (24 recdays) | `ev_first_A:pre` = `kin_missing:pre_A`: PFC has no tracking before the first A | 5e-16 on 0 rows (me11_05122021_06122021, ah04_05122021_06122021) |
| LEC `first_A`, 8 recdays at >= 10 (9 at >= 5) | `ev_first_A:pre` = `ev_A:pre` + `kin_missing:pre_A` (under `strict` the A and reward `pre` columns are zeroed where untracked) | 0, except in the fold whose held-out session holds the recday's only untracked pre-A rows |
| LEC >= 5, ly05_20250620_20250623 | one relation through the whole reward block, `ev_A:pre` and `kin_missing:post` | 4e-16 on 0 rows |

- **The one case that reaches held-out rows:** the LEC first-A fits.
  - On each of the 8 recdays exactly one session holds 27-62 untracked blocks before the first
    A, 7-16 s and under 0.2% of the recday's blocks.
  - In the fold that holds it out, those rows get an arbitrary share of the pre-A offset in the
    full model and none in the reduced one.
  - Freedman-Lane permutes residuals on the same design, so the p values stay calibrated, but
    those rows can move a first-A cpd.
  - The entry-anchor results carry no such effect.
- **The runner's flag has false alarms.** It marks the PFC first-A and some LEC shards as
  non-duplicate (`rank_benign` False) because it compares exact bytes. One-hot fractions sum
  to 0.9999999999999999 where an indicator is 1.0. The direct tests above override the flag.
  The fix is a tolerance in `rank_report`, at the next runner revision: not mid-run, because the
  runner hash is in the stamp.

### Measured (>= 10 trials unless stated; per mouse = recday mean, then mouse mean)

Fraction of units locked (F1). Mean of mice, range over mice in brackets:

| block | LEC (5 mice) | LEC, El-Gaby's rule | PFC (7 mice) |
|---|---|---|---|
| entry | 0.288 (0.167-0.426) | 0.298 | 0.344 (0.115-0.613) |
| first A | 0.271 (0.151-0.429) | 0.282 | 0.244 (0.021-0.472) |
| reward A | 0.117 (0.100-0.152) | 0.119 | 0.158 (0.084-0.304) |
| every reward | 0.508 (0.271-0.653) | 0.519 | 0.610 (0.313-0.815) |
| A + reward joint | 0.381 (0.217-0.495) | 0.393 | 0.554 (0.450-0.755) |
| speed bank, fast | 0.434 (0.184-0.631) | 0.452 | 0.618 (0.379-0.844) |
| speed bank, slow | 0.102 (0.069-0.120) | 0.107 | 0.120 (0.068-0.185) |
| place | 0.795 (0.579-0.924) | 0.808 | 0.735 (0.481-0.888) |

- **El-Gaby's rule** (curation rule 2, applied by `_unit_table` over each recday's ABCD sessions)
  flags 149 of 2851 LEC units (5.2%), 138 of the 2742 fitted. Removing them moves every
  fraction by <= 0.02.
- **This differs from M1.** M1 quoted 245 of 2851 (8.6%) from a count that is not in the code,
  probably over every task recording, object sessions included. Not reconciled.

Held-out cpd (F2):

- The median over all units is <= 0.002 for every event block: most units' event block does not
  help the held-out session.
- Locked units, median cpd:

  | dataset | entry | reward | reward A | A + reward joint |
  |---|---|---|---|---|
  | LEC | 0.0048 | 0.0039 | 0.0014 | 0.0055 |
  | PFC | 0.0058 | 0.0053 | 0.0017 | 0.0070 |

- Place is larger: 0.028 (LEC) and 0.014 (PFC).

Kinematics beside each event call (F3), for units locked to the block:

| block | dataset | locked units | also locked to the fast bank | event cpd > fast-bank cpd | median cpd, event / fast bank |
|---|---|---|---|---|---|
| entry | LEC / PFC | 790 / 424 | 0.62 / 0.75 | 0.68 / 0.55 | 0.0048 / 0.0018; 0.0058 / 0.0060 |
| reward | LEC / PFC | 1380 / 808 | 0.61 / 0.73 | 0.68 / 0.58 | 0.0039 / 0.0016; 0.0053 / 0.0058 |
| reward A | LEC / PFC | 316 / 176 | 0.54 / 0.64 | 0.54 / 0.44 | 0.0014 / 0.0015; 0.0017 / 0.0039 |
| A + reward | LEC / PFC | 1010 / 663 | 0.65 / 0.76 | 0.68 / 0.59 | 0.0055 / 0.0028; 0.0070 / 0.0065 |

Where entry kernels peak, as the fraction of entry-locked units whose largest |beta| is in each
range:

| range | LEC mice | PFC mice |
|---|---|---|
| first minute after recording start | 0.62-0.78 | 0.18-0.75 (ah04 0.35, me08 0.18, me11 0.29) |
| overflow, > 25 min | 0 | 0 |

The F4 heatmaps show units sorted by peak lag, but the sort uses the same data as the display.
Sorting noise by its peak also draws a diagonal, so the spread of peak times is not yet
evidence of tiling.

Reset test (F6), the fraction where the entry kernel beats a linear hours-since-day-start term
on held-out sessions:

| units | LEC (5 mice) | PFC (7 mice) |
|---|---|---|
| entry-locked | 0.823 (0.669-0.946) | 0.718 (0.417-0.938) |
| all units | 0.748 (0.577-0.856) | 0.609 (0.411-0.870) |

- PFC ah03 and me10 contribute 5 entry-locked units each.
- Skipped, with no wall clock (synthetic session tokens): ab03_29082023_30082023,
  me08_10092021_11092021, me10_14122021_15122021, me10_20122021_21122021.

Anchor swap (F7), recording start vs the first A on the same rows:

- Locked calls agree in 0.88-0.93 of units per LEC mouse.
- In PFC, 0.82-0.94 per mouse, except me08 at 0.65 (55 units).

Threshold sensitivity (F8, LEC, 24 shared recdays), unit-level agreement of the call at 10 vs
5 trials:

| block | agreement |
|---|---|
| reward A | 0.92 |
| A + reward | 0.89 |
| entry | 0.88 |
| reward | 0.86 |

- ah10 is identical: the same sessions pass both thresholds.
- The mice with low-trial sessions rise:

  | mouse | reward, 10 -> 5 trials | entry, 10 -> 5 trials |
  |---|---|---|
  | ly05 | 0.271 -> 0.509 | 0.180 -> 0.266 |
  | ah08 | 0.414 -> 0.542 | 0.167 -> 0.259 |

Reward-kernel latency (F9), the lag of the kernel's maximum in reward-locked units:

- Per-mouse medians are 3.5-19 s (LEC) and 0.75-13.5 s (PFC). Most reward-locked units are
  suppressed right after the reward (F4), so the maximum is the recovery.
- The fraction peaking in the first 0.5 s is 0.04-0.14 per LEC mouse and 0.12-0.44 per PFC
  mouse.
- **ah08** has the highest first-bin fraction among LEC mice, 0.143. Its kernels are not
  later than the other mice's, so the speed-reference offset (+0.25 to +0.40 s against -0.30 s)
  does not show here, at 0.5 s resolution.

Regions (F10; off-boundary units, >= 50 um from a region transition):

- Fraction locked to reward, mean of mice: ENTl-deep 0.595 (5 mice), SUB/ProS 0.567 (4),
  CA1/HPF 0.415 (2), ENTm 0.400 (3), ENTl-sup 0.310 (3); PFC 0.610 (7).
- The ENTl-sup and ENTm cells with >= 20 units are single mice (ah08, ly07).
- ENTl-deep - SUB/ProS: rate-matched within recday, mouse-level difference, labels shuffled
  within recday, 5000 permutations, 484 units, 4 mice.

  | block | fraction locked: difference, p | cpd: difference, p |
  |---|---|---|
  | entry | +0.055, 0.33 | +0.020, 0.0048 |
  | reward | +0.128, 0.023 | +0.003, 0.29 |
  | reward A | -0.008, 0.84 | +0.000, 0.89 |
  | A + reward | +0.075, 0.14 | +0.003, 0.23 |

  - The entry-cpd difference survives Holm over these 8 tests, but ly05 contributes +0.074 and
    the other three mice +0.001 to +0.004.
  - The reward-fraction difference is ah10 +0.04, ly05 -0.04, ly06 +0.29, ly07 +0.22.
  - Both are candidates resting on one or two mice.

Per recday (F11):

- ly05_20250620_20250623, v1's clock-fault recday, is reward-locked at 0.473 under v4: 6th of 24
  LEC recdays from the bottom, median 0.542. It is not the outlier v1 made it.
- The bottom five are the low-session recdays: ah08_20250624_20250625 (2 sessions, 0.113) and
  ly05_20250616_20250617, ly05_20250624_20250625, ly05_20250613_20250615 and
  ah08_20250613_20250615 (3 sessions each, 0.16-0.33).
- Median held-out r2 is negative on ah08_20250624_20250625 (-0.18), ly05_20250613_20250615
  (-0.05) and ly05_20250616_20250617 (-0.05). With 2-3 sessions, the ~150-column model overfits
  its 1-2 training sessions.

### What was NOT established

- **Whether an entry call is an entry clock or a leaky reward count.** WP1 showed superposition
  traces flagged by `ev_entry` (5/6 LEC, 6/6 PFC). M5's reset-vs-superposition test is still
  needed. The first-minute peak argues for a transient, not a count, but was not tested
  against one.
- **What the first-minute transient is.** It could be a response to placement, a recording-start
  artefact, or pre-A behaviour. The mouse is already in the maze at recording start, so entry is
  left-censored. PFC has no pre-A kinematics, so it carries no first-minute kinematic claims.
- **The spread of entry peak times as a sequence.** It needs a cross-validated sort (sort on half
  the sessions, show the other half).
- **LEC vs PFC.** No dataset contrast was tested. PFC's event calls compete with kinematics more
  than LEC's do (the F3 table), so a higher PFC fraction is not a stronger event clock.
- **Low-session recdays.** Recdays with 2-3 sessions sit outside the M1 recovery envelope, which
  was built at 8 sessions. Per-mouse fractions for ly05 and ah08 are power-limited at 10 trials
  (F8).
- **First-A cpd in the 8 LEC recdays** with one session of untracked pre-A rows (the rank
  section). The p values are calibrated; the cpd can move.
- **Kernel shapes and time constants** (Stage 2, M3). Tsao's stepwise table (M3). Rest (M4).
- **The El-Gaby count.** M1's 245 is not reconciled with the code's 149.

### Commands

```
python code/check_mirror_parity.py                                       # must exit 0
# runner gates (planted units through the runner; exit 0 = every check passes)
sbatch --time=0-01:00 --job-name=ek_rgate_lec sbatch_files/event_kernels.sbatch code/run_event_kernels.py --gate --dataset lec
sbatch --time=0-01:00 --job-name=ek_rgate_pfc sbatch_files/event_kernels.sbatch mFC_data/code/run_event_kernels.py --gate --dataset pfc
# Stage 1 + reset test, one job per recday and anchor (add --after <gate job ids> to gate them)
bash sbatch_files/submit_event_kernels.sh --dataset lec
bash sbatch_files/submit_event_kernels.sh --dataset pfc
bash sbatch_files/submit_event_kernels.sh --dataset lec --min-trials 5
# merge after every log reads EXIT=0
python code/run_event_kernels.py --dataset lec --merge
python mFC_data/code/run_event_kernels.py --dataset pfc --merge
python code/run_event_kernels.py --dataset lec --merge --min-trials 5
# notebooks (RUN_MODE = 'load'): two timeouts, SLURM walltime and the per-cell limit
sbatch --job-name=ek_nb_cmp --time=0-03:00 --mem=48G --cpus-per-task=4 sbatch_files/event_kernels.sbatch -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 code/event_kernels_M2_comparison.ipynb
sbatch --job-name=ek_nb_lec --time=0-02:00 --mem=32G --cpus-per-task=4 sbatch_files/event_kernels.sbatch -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 code/LEC_event_kernels.ipynb
sbatch --job-name=ek_nb_pfc --time=0-01:00 --mem=32G --cpus-per-task=2 sbatch_files/event_kernels.sbatch -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 mFC_data/code/PFC_event_kernels.ipynb
```

Outputs:

- shards: `data/event_kernels/stage1/shards/<dataset>_kin-strict_mt<k>_np100/`;
- merged: `data/event_kernels/stage1/stage1_<arm>.pkl`, with `_units`, `_reset` and `_summary`
  CSVs;
- figures: `data/figures/event_kernels/M2/` (F1-F11) and `data/figures/population_drift/M2/`
  (F12-F14).

## 2026-10-02 — M1: data preparation (WP0) and the synthetic gate (WP1)

**Outcome.** The data path is exact in both datasets. With the conservative kinematic design
(`kin='strict'`), the pipeline recovers what is planted on real behaviour and does not flag what
is not. The default design does not meet that bar in PFC. No real-data kernel has been fitted
yet; that is M2.

- LEC: every task session was re-extracted from recording start on the v4 clock. All 211 spike
  segments agree byte for byte with `neuron_raw_v4` on the core window. 210/211 pre-A tracks
  agree with `processed_v3` on the overlap.
- PFC: each of the 194 deposit sessions sits in exactly one place in El-Gaby's whole-recording
  shards. XY is in cm for 187/194.
- The loader equals `data_dic_lec_v4.pkl` in every session. Its 250 ms blocks equal GLM V3's;
  time since reward agrees to 1.1e-13 s.
- Synthetic gates, full runs (n_perm 100, one real recday per dataset). The default design
  leaks fast speed tuning into the A and reward kernels in PFC: 5/8 planted units, mostly
  through the untracked segments. The new `kin='strict'` design removes the leak (0/8) and leaves
  recovery of planted event units unchanged: 1.0 for every kind except LEC's superposition
  traces, 0.83 under both designs. Under `strict`, every gate passes in both datasets.
  `strict` is the recommended M2 design; the decision is the user's.

### What was done

| file | role |
|---|---|
| `code/preprocessing/extract_full_sessions.py` | LEC WP0: pre-A and post-last-trial spike segments on the v4 clock, pre-A tracking, recording index, unit stability |
| `mFC_data/code/pfc_session_index.py` | PFC WP0: shard index, recording index, pixel-to-cm affine, behaviour probes |
| `code/event_kernels.py` (+ mirror) | loaders, 25 ms features, 250 ms design, Stage 1, reset test, Stage 2 |
| `code/event_kernels_synthetics.py` (+ mirror) | gates G1-G11 (table in its docstring) |
| `code/population_drift.py`, `code/population_drift_synthetics.py` (+ mirrors) | Kanter measures; gates P1-P4 |
| `code/event_kernels_checks.py` | loader vs the v4 pickle; shared primitives vs GLM V3 |
| `code/event_kernels_pin.py` | the default design is bit-identical to a committed version; what `kin='strict'` changes |
| `code/event_kernels_leak_diag.py` | the A+reward joint's held-out gain for planted speed units, by row type |
| `sbatch_files/event_kernels.sbatch` | runs any of these on `gpu_lowp` after the mirror parity check |

Design choices and the measurement behind each are in the handoff, section 5.

### WP0, LEC: 25 recdays

| quantity | value |
|---|---|
| task sessions | 211: 191 ABCD, 20 object |
| unit-recordings | 2851; 66-186 per recday, median 108 |
| spike gate: full-session core == `neuron_raw_v4` | 211/211 byte-identical |
| tracking gate: pre-A resample == `processed_v3` on the overlap | 210/211 equal |
| the tracking exception | ah10_20250618_20250619 s5. Its pinstate is empty (0 rising edges), so the copilot has no matched clock. The v3 arrays come from a poke-based salvage. Its 7.5 s before the first A are untracked and coded missing. |
| recording start to first A | median 21.0 s, IQR 12.0-41.6, range 1.8-472.0 (211 sessions) |
| last trial to recording end | median 35.0 s, range 0-1095.7 (211) |
| pre-A tracked fraction | median 1.000, min 0.221; 8/210 sessions below 0.9 |
| first tracked frame | the first video frame in 209/210 sessions. Video starts a median 0.0 s after recording start (max 15.3 s). The exception, ly05_20250618_20250619 s0, is continuously tracked only from 23.1 s. |
| first poke | median 9.6 s after recording start (191 ABCD sessions); first poke to first A median 12.0 s |
| analysis sessions (ABCD, >= 10 trials) | 136 over 25 recdays; 1-8 per recday, median 6 |
| El-Gaby's 20% rule, diagnostic only | would drop 245/2851 unit-recordings (8.6%): ah08 8.7%, ah10 7.8%, ly05 11.4%, ly06 9.8%, ly07 5.4% |

The mouse is in view from the first frame, so it is already in the maze when recording starts.
Recording start is the earliest anchor available, not the true entry.

### WP0, PFC: 25 recdays from the deposit

| quantity | value |
|---|---|
| recdays, sessions, units | 25, 194, 1252 |
| shard search | 194/194 sessions have exactly one exact match |
| wall clock | 167/194 sessions. Four recdays carry synthetic tokens and have no wall clock: ab03_29082023_30082023, me08_10092021_11092021, me10_14122021_15122021, me10_20122021_21122021. The reset test cannot run on them. |
| shard start to first A | median 18.0 s, IQR 8.3-34.4, range 0.2-500.7 (194) |
| XY to cm | `xy_ok` 187/194. Dropped: me08_12092021_13092021, 7 sessions (affine 1.13 px/cm, residual 27.4 cm; its XY is not a maze track) |
| fitted scale | 2021 cohort 19.57-20.10 px/cm (18 recdays), 2023 cohort 15.83-16.13 px/cm (6). El-Gaby's measured value fits only the first. |
| node residual, kept recdays | worst node per recday: median 0.78 cm, max 1.78 cm |
| reward probe | population PSTH peak a median 0.00 s from reward (194 sessions) |
| speed probe | rate-speed lag median -0.47 s, IQR -0.82 to +0.33 (187 sessions); LEC's reference is -0.30 s |
| running speed | median 10.3 cm/s (187) |

The deposit's pixel files are untouched. The transform is applied at load from
`mFC_data/data/processed/pfc_xy_to_cm.v2.csv`.

### Data-path checks (`event_kernels_checks.py`, job 3675174)

| check | sessions equal |
|---|---|
| loader core counts == `Neuron_raw` in `data_dic_lec_v4.pkl` | 211/211 |
| XY, NaN-equal | 211/211 |
| Locs codes | 191/191 (object sessions have none) |
| `Trial_times` | 182/182 (sessions with trials) |
| 250 ms block counts == V3 `downsample_session_data(10, 'bin')` | 177/177 (ABCD sessions with >= 2 trials) |
| block speed and signed acceleration, 2 blocks inside the core's edges | 177/177 each |
| time since reward, max abs difference | 1.1e-13 s over 794,897 blocks |

No session the loader builds is missing from the pickle. The script fails if an analysis session
is missing, because a missing session would otherwise read as a pass.

### WP1, synthetic gate

Planted Poisson units on one real recday per dataset: LEC ah10_20250616_20250617, PFC
me11_05122021_06122021, 8 analysis sessions each. The spikes are replaced; timings, tracking,
place, pokes and the wall clock are real and enter through the same loader dicts. Quick runs use
n_perm 20 and 80 units. Full runs use n_perm 100 and 160 (LEC) or 158 (PFC) units. A unit counts
as locked to a block when the Freedman-Lane p < 0.05 and the held-out cpd > 0.

**Quick passes.** Three were needed. Each fix is in the first, uncommitted version of the
modules, so there is no version bump.

| pass | failed | cause | fix |
|---|---|---|---|
| 1 | G5: speed units locked in 0.50 (LEC) and 0.25 (PFC) | half the hits had cpd <= 0 at the p floor (1/21) | the locked rule needs cpd > 0; 4x more speed units |
| 1 | G7, LEC: day beat entry in 0.67 of trend units | hat functions of hours cannot extrapolate to a held-out first or last session (held-out RSS 7095 for the day model vs 5803 for entry) | day term linear per calendar day plus a day-2 offset, not reference-coded |
| 2 | G9, PFC: 1 of 2 place units flagged by `ev_A` | chance: cpd +0.0001 at the p floor, like the noise hits; Fisher p vs noise 0.24. Fixed cut-offs fail a pure-null process 16-29% of the time at these n. | chance rows became exact one-sided tests at 0.01; 8 place units; a no-place arm as G9's positive control |
| 3 | none | | |

**Full runs, linear bank** (`kin='full'`, jobs 3675170/71, n_perm 100). Values are the fraction of
planted units flagged; n is per dataset.

| gate | n | LEC | PFC |
|---|---|---|---|
| G1 noise, any event block or the A+reward joint | 60 | 0.07 | 0.10 |
| G2 entry ramps, tau <= 30 min, on `ev_entry` | 16 | 1.0 | 1.0 |
| G3 A ramps | 6 | 1.0 | 1.0 |
| G4 reward ramps, bumps, fixed-width bumps, progress and distance clocks | 6, 6, 4, 2, 2 | 1.0 each | 1.0 each |
| G4 reward superposition traces | 6 | 0.83 | 1.0 |
| G5 speed units tau 2 s, any event or the joint | 8 | 0.38 (Fisher p 0.031 vs noise) | **0.62 (p 0.002): fails** |
| G5 speed units tau 128 s | 8 | 0.00 | 0.00 |
| G5 speed units without kinematics | 16 | 0.62 | 0.50 |
| G6 OU drift on `ev_entry` | 8 | 0.00 | 0.00 |
| G7 day beats entry in trend units / entry beats day in ramps | 6 / 16 | 1.0 / 1.0 | 1.0 / 1.0 |
| G8 speed units on `ev_entry`, zero-init vs steady bank | 16 | 0.19 vs 0.06 | 0.06 vs 0.00 |
| G9 place cells, any event: with / without the place block | 16 | 0.06 / 0.50 | 0.12 / 0.69 |
| G10 Stage 2: ramp class and sign, tau within 2x | 32 | 0.91, 0.91 | 0.91, 0.94 |
| G10 Stage 2: bump class | 6 | 1.0 | 1.0 |
| G11 p < 0.05 with shifts >= 0 vs >= 2 min: entry ramps / OU / noise | 20 / 8 / 60 | 0.90 vs 0.90, 0 vs 0, 0.05 vs 0.05 | 0.90 vs 0.90, 0 vs 0, 0.07 vs 0.05 |

**The G5 failure, and why the gate first passed it.** The run passed every row as first written.
Two faults in the gate hid a leak.

- G3/G4 count a unit as A- or reward-locked if `ev_A`, `ev_reward` **or** the A+reward joint
  passes. G5 and G9 did not count the joint.
- G5 pooled fast and slow speed units, which halves its sensitivity. Fast speed history tracks
  reward timing; slow speed history does not.

Re-scored with one hit rule and per tau, PFC's tau 2 s speed units are locked in 5/8. All five
sit at the p floor (1/101), through the joint, with joint cpd +0.0004 to +0.0015. That is 5-10%
of what their `kin_fast` block explains (cpd 0.011-0.017). LEC shows 3/8 through `ev_A` or
`ev_reward` alone, with cpd near +0.0002. The re-scored tables are
`data/event_kernels/synthetics/gates_rescored_<ds>_<rd>.csv`.

**Where the leak comes from.** `event_kernels_leak_diag.py` rebuilds the full PFC synthetic and
splits the joint's held-out SSE gain by row type. Gains are summed over leave-one-session-out
folds for the 8 tau 2 s units; positive means the joint helps.

| rows (PFC, n blocks) | linear bank, one missing indicator | `strict` |
|---|---|---|
| tracked (34,385) | -1.7 to +15.4 | -40 to -14 |
| untracked before the first A (290) | -7.9 to +15.5 | -0.8 to +0.3 |
| untracked after the last trial (1,941) | -22.7 to +21.9 | -2.3 to +4.1 |

- PFC has no tracking outside the core. One `kin_missing` indicator cannot tell the segment
  before the first A from the one after the last trial. The A and reward blocks can: their
  pre-event and overflow columns mark those segments. So any speed-driven rate difference
  between the segments became an "event" gain.
- The same coefficients then spill into tracked rows through shared lag bins. With the segments
  absorbed, the tracked-row gain turns negative for every unit, which is pure overfitting.
- Curvature is the candidate second path. The bank enters linearly while the planted rate is
  exp(0.5 z). LEC's 3/8 must come from tracked rows, since this recday has no untracked pre-A
  rows. The full `strict` run tests it; the quick runs cannot, because the quick linear-bank run
  already had 0/4.

**The fix: `build_design(kin='strict')`.** It is a new option; the default `'full'` is unchanged.

| change | why |
|---|---|
| decile one-hots of each fast-bank trace (speed and \|accel\|, tau <= 8 s) inside `kin_fast`: 54 columns | tracked-row curvature |
| `kin_missing` split into before-first-A, after-last-trial and gap indicators | the segments' means go to the nuisance |
| A, reward and poke-exit columns zeroed on untracked rows | no fast-event claim rests on rows without kinematics |

The entry clock is not masked, because its first seconds are the pre-A rows. The plan already
rules out first-minute kinematic claims in PFC.

`python code/event_kernels_pin.py --against 57eac39` checks both sides:

- the default design is bit-identical to the committed version in three configurations;
- `strict` differs only in those three ways, and its split indicator sums to the default one.

**Full runs, `strict`** (jobs 3675492/93, n_perm 100, same planted units): every row passes in
both datasets.

| gate | n | LEC | PFC |
|---|---|---|---|
| G1 noise, any event block or the A+reward joint | 60 | 0.10 | 0.07 |
| G2 entry ramps, tau <= 30 min, on `ev_entry` | 16 | 1.0 | 1.0 |
| G3 A ramps | 6 | 1.0 | 1.0 |
| G4 reward ramps, bumps, fixed-width bumps, progress and distance clocks | 6, 6, 4, 2, 2 | 1.0 each | 1.0 each |
| G4 reward superposition traces | 6 | 0.83 | 1.0 |
| G5 speed units tau 2 s, any event or the joint | 8 | 0.25 (Fisher p 0.24) | 0.00 |
| G5 speed units tau 128 s | 8 | 0.00 | 0.00 |
| G5 speed units without kinematics | 16 | 0.62 | 0.50 |
| G6 OU drift on `ev_entry` | 8 | 0.00 | 0.00 |
| G7 day beats entry in trend units / entry beats day in ramps | 6 / 16 | 1.0 / 1.0 | 1.0 / 1.0 |
| G8 speed units on `ev_entry`, zero-init vs steady bank | 16 | 0.19 vs 0.06 | 0.19 vs 0.00 |
| G9 place cells, any event: with / without the place block | 16 | 0.06 / 0.50 | 0.19 (Fisher p 0.16) / 0.69 |
| G10 Stage 2: ramp class and sign, tau within 2x | 32 | 0.88, 0.91 | 0.94, 0.97 |
| G10 Stage 2: bump class | 6 | 1.0 | 1.0 |
| G11 p < 0.05 with shifts >= 0 vs >= 2 min: entry ramps / OU / noise | 20 / 8 / 60 | 0.90 vs 0.90, 0 vs 0, 0.07 vs 0.05 | 0.90 vs 0.90, 0 vs 0, 0.05 vs 0.03 |

- **What remains is noise-sized.** Every remaining hit has a cpd no larger than the noise
  units' (up to +0.0004 for noise):
  - LEC's two tau 2 s speed units: +0.0001 to +0.0007;
  - PFC's three place cells: <= +0.0001;
  - two PFC OU drift units (+0.0004 each through `ev_reward`; one also +0.0018 through the
    joint, as under `full`). 2/8 against 4/60 noise is chance.
- **G8 now shows the start-up artefact the steady-state bank prevents.** With a zero-initialised
  bank, 3/16 speed units read as entry-locked in both datasets. With the steady-state bank it is
  1/16 (LEC) and 0/16 (PFC).
- **Cost.**
  - The design grows from 94 to 149 (LEC) and 147 (PFC) columns. The minimum fold rank is 147
    of 149 (LEC) and 147 of 147 (PFC).
  - Planted-unit recovery is unchanged.
  - On real PFC data, masking removes the A / reward kernels from the untracked segments: in
    me11_05122021_06122021 that is 2,247 of 36,632 blocks (6%).

**Recovery envelope** (full runs, linear bank).

- **Entry ramps.**
  - Detected at every tau from 30 s to 30 min: 16/16 in both datasets.
  - At tau 2 h, the falling ramp is detected 2/2 but the rising one 0/2. The rising version runs
    at a third of the base rate, so it carries fewer spikes.
  - Stage 2 tau is within 0.94-1.33x of truth up to 8 min.
  - At 30 min the median is 0.71x (LEC) and 1.72x (PFC), range 0.50-2.43x. At 2 h it is 0.43x
    and 1.01x, range 0.21-1.01x. A 20 min session cannot pin a slower tau, so tau >= 30 min
    should be read as "slower than the session", not as a number.
- **A and reward ramps.** tau within 1.0-1.41x of truth, all classed ramp-down.
- **Bump peaks.** Planted 1, 3 and 8 s are recovered as 1.00, 2.83 (the grid point nearest 3)
  and 8.00 s. Bumps are classed as bumps in 10/10, fixed-width ones included.
- **Shape errors.** Entry ramps are right in 17/20; 2 are classed bump and 1 none.
- **Superposition reads as an entry ramp.** A leaky reward count is also flagged by `ev_entry`:
  - LEC: 2/2 at tau 30 s and 120 s, 1/2 at 8 s;
  - PFC: 2/2 at all three taus.

  It rises over the session as rewards accumulate. An entry-ramp call can therefore be a leaky
  reward count. M5's reset-vs-superposition test is what separates them.
- **Minimum shift.** A 2 min minimum Freedman-Lane shift changes no verdict (G11), so no
  `min_shift` option goes into `glm_cv`.
- **Under `strict`** (same planted units):
  - entry ramps are classed right in 16/20 (LEC) and 18/20 (PFC);
  - tau up to 8 min stays within 0.71-1.41x of truth for entry, A and reward ramps;
  - tau 30 min and 2 h stay unresolved (0.13-3.43x);
  - superposition traces are still flagged by `ev_entry`, in 5/6 (LEC) and 6/6 (PFC).

**Population drift** (full, jobs 3675172/73; P-gates in the docstring of
`population_drift_synthetics.py`): every row passes in both datasets.

| gate | LEC | PFC |
|---|---|---|
| P1 noise kNN accuracy within / across sessions (limit: chance + 0.05) | 0.047 / 0.067 | 0.037 / 0.053 |
| P1 noise distance change from lag 1 to lag 10 | +0.0001 | -0.0004 |
| P2 locked clock, across-session accuracy / chance | 17.6 | 16.1 |
| P3 OU drift, within / across, each over chance | 13.7 / 0.92 | 12.7 / 0.80 |
| P3 OU drift, distance change from lag 1 to lag 10 | +0.21 | +0.21 |
| P4 boundary ensembles, z at lag 0 or 1 / largest abs z of the other populations | 8.9 / 0.95 | 9.5 / 0.71 |

### Deviations from the approved plan

1. **Identity check.** The planned CPD identity with V3 became the primitives identity above.
   V3's design differs on purpose (mode place coding, NaN-row drop, transition mask, core-only
   rows). A CPD identity would have tested a re-implementation of V3, not this pipeline.
2. **PFC XY gate.** El-Gaby's px/cm fits only the 2021 cohort. The gate became one pooled
   affine per recday plus a 5% session scale check where >= 6 nodes were visited.
3. **Missing kinematics are coded, not dropped.** This keeps the PFC pre-A rows and the entry
   kernel's first seconds.
4. **Stage 2 picks tau** by a training-fold profile, not an inner leave-one-session-out; an 8 x 7
   inner loop was too slow. Evaluation stays on held-out sessions.
5. **Gate statistics.** The locked rule and the exact chance tests (quick passes above).
6. **Day term.** Linear per day, not hat functions.
7. **Kinematic design `kin='strict'`** (decile fast bank, per-segment missing indicators, fast
   events masked where untracked). It was added after the full gate exposed the PFC leak. It is
   an option; the approved default is unchanged.
8. **Deferred to M4.** The SWS proxy per rest block was not computed in WP0. The rest synthetics
   (rest-entry ramps confounded with SWS onset, OU units with a known intrinsic timescale) were
   not built. Both are only used in M4 and will be built and gated with it.

### What was NOT established

- Any real-data result: kernels, time constants, reset test, drift.
- The envelope above comes from one recday per dataset with 8 analysis sessions. Recdays with
  fewer sessions are outside it. ly05_20250618_20250619 has 1 analysis session, so no held-out
  fit is possible there. ah08_20250624_20250625 has 2.
- True maze entry. The mouse is in the maze before recording starts, so entry-kernel lags are
  left-censored by an unknown amount.
- ah08's speed reference (+0.25 to +0.40 s against -0.30 s in the other mice): physiology or a
  residual offset is still open. Reward-kernel latencies will be reported per mouse.
- PFC's day-scale ramps will be a lower bound. El-Gaby's 20% rule pre-filtered the deposit and
  cannot be undone.
- `strict` on real data. Its decile bank corrects curvature only to decile resolution. A neuron
  with stronger nonlinear speed-history tuning than the planted exp(0.5 z) could still leak; the
  per-unit kinematic cpd should be reported next to each event call.
- Whether a slow "entry" ramp is an entry clock or a leaky reward count. Superposition traces
  with tau >= 30 s are flagged by `ev_entry` under both designs; M5's test is still needed.

### Commands

```
python code/check_mirror_parity.py                       # must exit 0
# WP0, LEC (per mouse; tables versioned, never overwritten)
sbatch --partition=gpu_lowp --job-name=fullsess_<mouse> --time=0-02:00 --mem=32G \
    sbatch_files/mec_stage.sbatch extract_full_sessions.py --recdays <rd ...> --write
# WP0, PFC
sbatch --partition=gpu_lowp --job-name=pfcidx --time=0-02:00 --mem=48G \
    sbatch_files/mec_stage.sbatch ../../mFC_data/code/pfc_session_index.py --write
# data-path checks (loads the v4 pickle)
sbatch --job-name=ek_checks --mem=64G sbatch_files/event_kernels.sbatch code/event_kernels_checks.py
# WP1, quick then full
sbatch --job-name=ek_syn_lec_q3 sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec --quick
sbatch --job-name=ek_syn_pfc_q3 sbatch_files/event_kernels.sbatch mFC_data/code/event_kernels_synthetics.py --dataset pfc --quick
sbatch --job-name=ek_syn_lec sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec
sbatch --job-name=ek_syn_pfc sbatch_files/event_kernels.sbatch mFC_data/code/event_kernels_synthetics.py --dataset pfc
sbatch --job-name=pd_syn_lec sbatch_files/event_kernels.sbatch code/population_drift_synthetics.py --dataset lec
sbatch --job-name=pd_syn_pfc sbatch_files/event_kernels.sbatch mFC_data/code/population_drift_synthetics.py --dataset pfc
# the conservative kinematic design: pin the default first, then gate it
python code/event_kernels_pin.py --against 57eac39            # exit 0 = default bit-identical
sbatch --job-name=ek_syn_lec_st sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec --kin strict
sbatch --job-name=ek_syn_pfc_st sbatch_files/event_kernels.sbatch mFC_data/code/event_kernels_synthetics.py --dataset pfc --kin strict
# where a leak comes from: the joint's held-out gain by row type, both designs
sbatch --job-name=ek_diag_pfc --time=0-01:00 sbatch_files/event_kernels.sbatch code/event_kernels_leak_diag.py pfc me11_05122021_06122021
```

Logs: `sbatch_files/slurm_outputs/event_kernels-<name>-<id>.out`, read to the `EXIT=` line.
Outputs: `data/event_kernels/synthetics/` and `data/event_kernels/checks/`, versioned `.vN`.
