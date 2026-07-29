# Binary state-pair decoders generalising to held-out tasks (CCGP) — LEC & PFC

Companion documentation for `ccgp_state_pairs.py` (identical copy in `code/` and
`mFC_data/code/`) and the notebooks `LEC_ccgp_state_pairs.ipynb` /
`PFC_ccgp_state_pairs.ipynb`.

---

## 1. The question

> *"Did you try binary decoders between pairs of ABCD. Eg A vs B generalising to
> new task. Stefano says they are much more likely to work."*

Train a binary classifier to separate two task states (e.g. A vs B) using some
tasks, then test it on a **held-out task**. This is **CCGP** (cross-condition
generalisation performance), from Bernardi, Salzman & Fusi (2020).

The four rewards sit at different towers in every task, so a decoder that
transfers to a new task cannot be riding on place — it has to have found a
representation of state that is *task-invariant*, i.e. **abstract**.

### 1.1 Why binary should be more sensitive than what we already ran

This is Stefano's argument, and it is worth stating precisely because it predicts
*where* the effect should show up.

`state_decoding_analysis.decode_states_loo_cv` already does leave-one-task-out —
but 4-way. A 4-way decoder that generalises requires **all four** states to be
simultaneously separable *and* aligned across tasks. Each binary decoder requires
**one dichotomy** to be aligned. Binary is therefore a strictly weaker condition:
any geometry that supports 4-way generalisation supports all 6 binary decoders,
but not conversely. If the population codes only *part* of the task structure
abstractly — say, a phase-like axis that separates near-in-the-loop from
far-in-the-loop states but does not resolve all four identities — the 4-way
decoder fails while some binary decoders succeed.

So the 6 pairs are not 6 noisy replicates of one number. **Which** pairs work is
the result. See §6.

---

## 2. What the data actually says (measured, read-only)

Every design decision below follows from one of these. Each was verified against
the pickles, not assumed.

### 2.1 State *i* is the leg *from* goal *i* to goal *i+1* — so every leg has a reward at both ends

`Trial_times` is `(n_trials, 5)`. Checking `Locs_raw` at each boundary against
`Task`, over 3180 boundaries across 6 recdays:

| `Trial_times` column | animal is at | hits |
|---|---|---|
| col 0 | `Task[0]` = goal A | 636/636 |
| col 1 | `Task[1]` = goal B | 636/636 |
| col 2 | `Task[2]` = goal C | 633/636 |
| col 3 | `Task[3]` = goal D | 632/636 |
| col 4 | `Task[0]` = goal A (next trial) | 636/636 |

So `Trial_times[:, i]` is the time the animal is **at** `Task[i]`, and state *i*
spans `[Trial_times[:,i], Trial_times[:,i+1])` = the journey **from** goal *i*
**to** goal *i+1*. **State A begins at reward A.** This is load-bearing for §2.2.

### 2.2 A is anchored by an auditory tone — which is also a confound, on state A

A is the only reward paired with a tone. That breaks the equivalence across
rewards and is what licenses "A" meaning the same thing in every task. Whether
the mouse *uses* it is an open question — §5.3 tests it rather than assuming it.

But combined with §2.1, the tone is present at reward A in **every task**, so a
purely auditory response transfers across tasks perfectly and would masquerade as
an abstract state code. This is the single most likely source of a false positive.

**The tone comes on *immediately upon collection* of reward A, so it occupies only
the first bit of state A** (post-collection). The **end of state D** — the approach
to A, *before* collection — is **tone-free**. So the tone contaminates **state A
only**, in its early bins. A tone-driven decoder can discriminate a pair only if
the pair **involves A**:

| pairs | tone | |
|---|---|---|
| AB, AC, AD | one state (A) carries it | **tone-decodable** |
| BC, BD, CD | neither state carries it | tone-immune |

Measured on the corrected one-sided tone synthetic (untrimmed): AB = AC = AD ≈
**1.000**, BC/BD/CD ≈ 0.5; trimmed, all ≈ 0.5.

> **Correction (history).** An earlier draft modelled the tone with a *wrapping*
> Gaussian centred at the D→A boundary, which bled into the end of D and led to the
> grouping "{A, D}" (tone-decodable {AB, AC, BD, CD}, immune {AD, BC}). That was an
> artifact of the synthetic, not biology: the experimenter confirms the tone is
> post-collection, so it is early-**A** only. Tone-decodable = the **A-pairs**. The
> headline is unaffected — the mid-leg trim removes the early-A window regardless.

### 2.3 Nothing else distinguishes the states

Leg durations, excluding the first trial of each session (n = 3068 legs):

| state | median | mean |
|---|---|---|
| A | 9.80 s | 14.47 s |
| B | 8.93 s | 14.42 s |
| C | 9.11 s | 14.32 s |
| D | 8.79 s | 12.88 s |

Near-identical. Trials cycle continuously (trial *n* ends where *n+1* starts;
first trials are ~2× slower for all states — that is search). So the tone is the
*only* asymmetry in the loop.

### 2.4 Six unique tasks per recday, and sessions repeat

23/24 LEC recdays have exactly 6 unique tasks from 8 sessions; one has 7. The
repeat structure is regular: **sessions [0,3] are the same task, and [4,7] are the
same task**. Deduplication is mandatory — folding over raw `session_inds_dic`
would put a "held-out" task in the training set. We use
`glm_analysis_v2.get_sessions_for_glm` (one session per unique `Task`,
most-trials-wins).

### 2.5 Reward towers overlap heavily across tasks

Over the 366 unique task pairs:

| | fraction |
|---|---|
| task pairs sharing *zero* towers | **4.6%** |
| task pairs placing some state at the *identical* tower in both | **36.3%** |

A place cell for that tower would let an A-vs-B decoder "generalise" trivially.
Place control is **not optional**. Note the naive fix — only compare tasks with no
shared towers — discards 95% of the data.

### 2.6 Place gets in through the *destination* tower, and mostly hurts rather than helps

Two measured facts about the place synthetic, both of which contradict the
intuition this design started from.

**(a) The confound runs through the leg's destination, not its source.** Training
on one task and testing on another, split by whether the decoded legs' towers
matched positionally:

| | CCGP |
|---|---|
| **destination** tower `Task[s+1]` matched | **0.921** |
| destination not matched | 0.404 |
| source tower `Task[s]` matched | 0.641 |
| source not matched | 0.476 |

The original rule checked only the *source* — it was both misnamed (`dest`) and
controlling the wrong tower. It is retained as `source` only to document the error.

**(b) A pure place code transfers *below* chance** (0.407 overall), because place
remaps: the same tower sits at a different state in each task, so towers get
systematically mis-assigned. Place is therefore mainly a source of **false
negatives** — it *masks* a real abstract code — and only fakes a positive on the
minority of folds where towers happen to match. Tightening the positional rules
makes this *worse*, not better (`endpoints` → 0.260), because they strip out the
positive-transfer folds and leave the anti-transfer ones.

Cost of each rule, filtering *per decoded pair*, training tasks surviving out of 5:

| rule | mean kept | ≥2 left | 0 left |
|---|---|---|---|
| `source` — the state's own tower must differ | 4.01 | 98.6% | 0.1% |
| `leg_exact` — the (source, dest) tuple must differ | 4.93 | 100% | 0% |
| `endpoints` — source AND dest must each differ | **3.46** | 94.4% | 0.5% |
| `leg_anyshare` — legs share no tower at all | 1.80 | 59.7% | 12.8% |

### 2.7 Trimming both ends of the leg is the primary control — and it is free

Both confounds live in the **reward windows at the leg boundaries**: the tone
(§2.2) and the place bumps (§2.6) are all at rewards, and every leg has a reward at
each end. So trim both ends and both confounds go away at once.

Measured on the place synthetic, trimming **symmetrically** (`trim_start_bins ==
trim_end_bins`; this is what `run_synthetic_controls` reports):

| trim | place, `none` | place, `endpoints` | place, `leg_anyshare` |
|---|---|---|---|
| 0 | 0.440 | 0.309 | 0.225 |
| **15** | **0.484** | **0.502** | **0.518** |

And on the corrected (one-sided) tone synthetic, per pair — the tone is in early
state A, so it inflates the **A-pairs** untrimmed and dies once the leg start is
trimmed:

| trim | AB | AC | AD | BC | BD | CD |
|---|---|---|---|---|---|---|
| 0 | **1.00** | **1.00** | **1.00** | 0.5 | 0.5 | 0.5 |
| **15** | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |

At trim = 15 the place code sits at chance under **every** rule including `none`,
and all six tone pairs collapse to chance. **Trimming, not the task filter, is
what removes the confounds** — and it costs no training data, whereas
`leg_anyshare` costs 64% of it.

**Trim symmetrically — for place, not the tone.** The *tone* is only at state A's
start, so `trim_start` alone removes it. But **place** fields sit at *both* reward
ends of every leg (goal *s* at the start, goal *s+1* at the end), so symmetric
trimming is needed to remove them. Keeping `trim_start == trim_end` therefore
controls both confounds at once.

`place_control='endpoints'` is kept as cheap belt-and-braces for **mid-leg
(corridor) place cells**, which trimming does not touch and the synthetic does not
model. `leg_anyshare` remains the robustness check.

### 2.8 `Neurons_norm` is the right input

`(n_neurons, n_trials, 360)` = 90 phase bins × 4 states, time-warped, in raw
spike-count units (mean 0.075, matching `Neuron_raw` — it is **not** z-scored;
"norm" means time-normalised). Trial-resolved *and* phase-warped: the phase bins
are what make the tone-trimming of §5.4 possible.

### 2.9 Sample sizes are workable but small

LEC: 66–186 neurons per recday (median 99); median 18 trials/session; 146/177
sessions have ≥8 trials. That is ~18 samples per state per task, and ~90 per class
after pooling 5 training tasks. Per-recday CCGP will be noisy; inference lives at
the group level (§4.3).

---

## 3. The data path

`CCGPConfig` (see the module for defaults). Per recday:

1. Dedupe to one session per unique task (§2.4).
2. Per task, take `Neurons_norm`. State *s* occupies bins `[s*90, (s+1)*90)`. Drop
   the first `trim_start_bins` (the reward/tone window — §2.1) and the last
   `trim_end_bins` (approach to the next reward); average the rest → **one
   population vector per (trial, state)**.
3. **Z-score per neuron within task.** This makes tasks commensurable despite
   firing-rate drift across sessions. It uses only unlabelled statistics (per-neuron
   mean/std over that task's samples), so it is not label leakage — it is the
   standard CCGP convention of normalising each context separately. Without it,
   cross-session drift alone can dominate the decoder.
4. Drop zero-variance neurons.

Neurons are index-matched across sessions *within* a recday, which is the only
reason a cross-task decoder is possible at all. They are **not** matched across
recdays — so each recday yields its own CCGP, and the group statistic is across
recdays.

---

## 4. The method

### 4.1 The headline

For each held-out task × each of the 6 pairs {AB, AC, AD, BC, BD, CD}:

- training tasks = all other unique tasks, minus those failing the place control
  (`endpoints`, §2.6);
- fit `LinearSVC(C=1.0)` (the Fusi convention) on the pooled training tasks;
- score `balanced_accuracy_score` on the held-out task;
- skip folds with <2 surviving training tasks.

### 4.2 The null — and why it is not 0.5

Chance is 0.5 by construction, but **testing against 0.5 with a binomial is
wrong**: samples within a task are correlated and n is small, so the thing that
needs calibrating is the *variance*, not the centre.

**The role-permutation null** (`null_pair_accuracy`). Each training task
independently gets a **random ordered pair of states** to play the roles of (i,j);
refit; test on the **true** (i,j) in the held-out task. Repeat `n_shuffles` times.

Every training task still contributes two real, well-separated states, so
within-task geometry, sample size and correlation structure are preserved
*exactly*. The only thing destroyed is the **cross-task correspondence** — which
is precisely what "abstract" claims. If the state code is aligned across tasks the
real decoder beats this null; if the population merely has four separable states
that are arbitrarily arranged in each task, it does not.

#### 4.2.1 The null we deliberately do *not* use

The tempting cheap alternative is to score the already-trained (i,j) decoder
against **mismatched ordered test pairs** (k,l) ≠ (i,j) — 12 ordered pairs, 1
true, 11 null, no refitting required. **It is degenerate, and we checked rather
than assumed.**

Balanced accuracy obeys `acc(k,l) == 1 - acc(l,k)`. The 12 ordered pairs are 6
unordered pairs × 2 orientations, each contributing `acc + (1-acc) = 1`, so

```
sum over all 12 ordered pairs == 6      exactly, for any decoder, on any data
mean of the 11 "null" pairs   == (6 - acc_true) / 11
```

The null mean is a **deterministic function of the true accuracy**. The test
`acc_true > null_mean` reduces algebraically to `12·acc_true > 6`, i.e.
`acc_true > 0.5`. It calibrates nothing — it is the binomial-against-0.5 test in
disguise, wearing the costume of a permutation test. (Verified numerically: sum =
6.000000, `acc_true` = 1.0000 → null mean 0.4545 = (6−1)/11 exactly.)

This is exactly why the 6×6 transfer matrix (§5.3) is **descriptive only** and is
never used for inference.

> **Why this much care about the null.** The persistent-homology work in this repo
> was misled by a null that was mis-calibrated in a way that looked strongly
> significant: the per-neuron circular shift decorrelates the population, so real
> (low-dimensional) data beat it almost always — 89/112 PFC and 113/125 LEC
> sessions hit the p-floor. That "significance" meant *"the population is
> low-dimensional"*, not *"there is a ring"*. The lesson: a null must hold
> everything constant except the one thing being claimed — and must be checked
> against that standard rather than assumed to meet it. §4.2.1 is that lesson
> applied; the degenerate null was in the first draft of this design and was caught
> by working out its algebra, not by running it.

### 4.3 Group-level inference

One value per recday → Wilcoxon signed-rank against the per-recday null mean.
Recdays within a mouse are **not** independent (5 LEC mice / 24 recdays; 7 PFC
mice / 25), so a mouse-level summary is reported alongside, grouping by
`mouse_recday[:4]` (the `group_tuning_by_mouse` convention).

---

## 5. The controls, and what each one rules out

Each analysis exists to kill one specific alternative explanation.

### 5.1 Ceiling — within-task decoding

Leave-one-trial-out *within* the test task. Rules out nothing by itself, but it is
what makes a null result interpretable: **high within-task + chance CCGP = there
is a state code, it just isn't abstract.** Without this, "CCGP ≈ chance" is
ambiguous between that and "no state code at all".

### 5.2 Place — matched vs unmatched

Report CCGP split by whether the decoded legs shared a tower with the test task.
This is a *direct measurement* of the place contribution rather than an assumption
that the control worked — and given §2.6(b), the expected sign is **negative**:
place drags cross-task transfer below chance. Robustness: rerun with
`place_control='leg_anyshare'` (§2.6).

### 5.3 Anchoring — the 6×6 transfer matrix

Train on pair (i,j), test on pair (k,l) in the held-out task, **canonical (i<j)
orientation only** (the flipped orientation is just `1 - acc`, so averaging both
would force every cell to 0.5). **Descriptive only — not a null**, for the reason
in §4.2.1.

- **diagonal** = the CCGP that was asked for;
- **off-diagonal** = whether the code is *rotation-symmetric*. If the population
  carries a task-phase ring with a consistent offset, a decoder trained on one
  adjacent pair transfers to other adjacent pairs at a predictable rotation. If A
  is not functionally anchored (i.e. the mouse ignores the tone), the matrix is
  rotation-symmetric and the diagonal is not special.

Also split by cyclic lag: **adjacent** (AB, BC, CD, AD — D→A is adjacent) vs
**opposite** (AC, BD). A phase/ramp code predicts lag-dependence; a discrete
state-identity code predicts uniform accuracy across all 6.

### 5.4 The tone — A-pairs vs non-A-pairs

The one most likely to produce a false positive (§2.2). The tone comes on upon
collection of reward A, so it lives in the **early bins of state A** only (the end
of D is tone-free). So it is decodable exactly for pairs that **involve A**:

- **tone-decodable**: AB, AC, AD;
- **tone-immune**: BC, BD, CD.

**If CCGP is high only on the A-pairs {AB, AC, AD} while {BC, BD, CD} sit at null,
it is the tone, not abstract state.** The two groups are reported separately,
always — the 6 pairs are never pooled into one number.

Primary mitigation is the trim (§2.7): trimming the **start** of each leg removes
the early-A tone window. The `trim_start_bins` sweep (and the leg-window sweep,
§5.6) is the honest check — a tone-evoked response dies as the reward window is
trimmed, an abstract state code survives.

### 5.5 The direct answer to Stefano

Run the existing `decode_states_loo_cv` (4-way) on the same folds and plot binary
CCGP against it. This is the literal test of the claim in the email.

### 5.6 Leg-window sweep — where along the leg does state info live?

The headline uses the **mid-leg** window (bins 15–75) precisely to remove the tone
(early state A) and reward-tower place. `run_ccgp_windows` / `plot_ccgp_windows`
re-run the decoders over several windows of the 90-bin state:

| window | bins | note |
|---|---|---|
| `post_reward` | 0–15 | immediate post-reward; the **tone window** in state A |
| `early` | 0–30 | early goal progress |
| `mid` | 15–75 | the headline (reward windows trimmed) |
| `late` | 60–90 | approach to the next reward |
| `full` | 0–90 | entire leg |

These deliberately **relax** the tone control (place is still filtered by
`place_control` in every window). CCGP is split into **A-pairs vs non-A-pairs**.
Because the tone is *task-invariant* (same tone at A in every task) it
**generalises** — so the prediction is that the `post_reward`/`early` windows lift
the **A-pairs above null while non-A pairs stay at chance**, fading toward mid/late.
That pattern is the tone, and it *validates* the mid-leg headline. A **uniform**
lift across all six pairs that survived would instead be a genuine early-progress
state code (the surprising result). The per-window within-task **ceiling** shows
where state information sits within a task, independent of generalisation.

---

## 6. How to read the result

The pattern across the 6 pairs is the result, not their mean.

| pattern | reading |
|---|---|
| all 6 pairs > null; survives the trim sweep; shared-tower ≈ no-shared-tower | **abstract state code.** The claim. |
| A-pairs {AB, AC, AD} > null but {BC, BD, CD} ≈ null; dies with trimming | **tone response**, not state. |
| only shared-tower folds > null | **place coding** leaking through; the control is doing the work. |
| adjacent ≠ opposite; 6×6 rotation-symmetric | **task-phase / ramp code**, not discrete state identity. A is not functionally anchored (the mouse ignores the tone). |
| all ≈ null, but within-task decoding high | state is coded, **not abstractly**. |
| all ≈ null, within-task at chance | no state code recoverable at this n. Check §2.9 before concluding. |
| all *below* null | consistent with a place/remapping-dominated population (§2.6b), not evidence of anything abstract. |

---

## 7. Validation

Synthetic populations run through the **real pipeline** unmodified
(`make_synthetic_recday`). Each validates one specific claim:

| synthetic | expected | validates |
|---|---|---|
| abstract task-phase tuning, identical across tasks | CCGP ≈ 1 on all 6 pairs, even under `leg_anyshare` | the pipeline can detect the effect, and the controls don't destroy a real one |
| place-tuned only (tuning follows towers) | off chance at `trim_end_bins=0`; ≈ 0.5 at the default | the place control actually works |
| tone response at reward A | untrimmed: {AB,AC,BD,CD} ≫ {AD,BC}; trimmed: all ≈ 0.5 | the tone diagnostic works |
| pure noise | ≈ 0.5, and the role-permutation null covers it | no false positives |

**The gate:** the place synthetic must be **off** chance at trim = 0 (otherwise
there is no place signal to control and the test is vacuous) and **at** chance
under the config actually used. If not, the controls are broken and no result on
real data means anything.

Measured output of `run_synthetic_controls()` at the current defaults (≈3 min):

(`tone`/`immune` = A-pairs {AB,AC,AD} / {BC,BD,CD}):

```
  place     endpoints     trim=0   ccgp=0.309 null=0.504 tone=0.267 immune=0.350
  place     endpoints     trim=15  ccgp=0.502 null=0.494 tone=0.486 immune=0.517   <- gate passes
  place     leg_anyshare  trim=0   ccgp=0.225 null=0.475 tone=0.275 immune=0.139
  place     leg_anyshare  trim=15  ccgp=0.518 null=0.496 tone=0.491 immune=0.563
  abstract  none          trim=15  ccgp=1.000 null=0.496 tone=1.000 immune=1.000   <- detects effect
  abstract  leg_anyshare  trim=15  ccgp=1.000 null=0.527 tone=1.000 immune=1.000   <- controls don't kill it
  tone      none          trim=0   ccgp=0.765 null=0.504 tone=1.000 immune=0.531   <- A-pairs ride the tone
  tone      none          trim=15  ccgp=0.488 null=0.502 tone=0.481 immune=0.495   <- trimmed away
  noise     none          trim=15  ccgp=0.478 null=0.500 tone=0.460 immune=0.495   <- no false positive
```

**This is not ceremony — it is why the defaults are what they are.** These
synthetics caught two design errors in this analysis that the real data could never
have revealed:

1. The place confound runs through the leg's **destination** tower (0.921), so the
   original source-tower rule controlled the wrong thing entirely.
2. The tone timing: the tone is post-collection, in **early state A only** — so the
   tone-decodable pairs are the **A-pairs {AB, AC, AD}**. (A synthetic that let the
   tone bump *wrap* into the end of state D once suggested "{A, D}"; the
   experimenter's clarification and a one-sided bump set it straight. The lesson
   cuts both ways: a synthetic can also *mis*lead if its generative model is wrong,
   so the biology has the final word.)

Both mattered, and either could have produced a confidently mislabelled diagnostic.
(In the persistent-homology work the same practice caught three errors, including a
null that was invalid in the direction of *more* significance.)

---

## 8. Runtime

Runs in-notebook; no SLURM, unlike the PH pipeline. `run_ccgp_batch(..., n_jobs=4)` does the
24 LEC recdays in ~10–25 min.

The **null dominates the cost**, for a reason worth knowing: a real decoder fits in
**~2 ms** because the states are separable, while a role-permuted one takes **~47
ms** because scrambled labels are not, so LinearSVC runs to `max_iter`.

**Do not lower `max_iter` to speed this up.** It would under-converge the *null
only* — weakening the comparison decoder while leaving the real one untouched, and
biasing the test anti-conservatively. Parallelise instead: `n_jobs` changes nothing
statistically. (`clf='logistic'` is a legitimate ~4× faster alternative that always
converges; `linear_svc` is the default only because it is the Fusi convention.)

## 9. Known limitations

- **Small n** (§2.9). Per-recday estimates are noisy; the inference is at group
  level, and the mouse-level grouping matters.
- **A null result needs the ceiling** (§5.1) to be interpretable.
- **The tone could carry the whole effect** (§5.4). If only the A-pairs
  {AB, AC, AD} transfer and trimming kills them, the honest conclusion is "tone
  response, not abstract state".
- CCGP measures whether a *linear* readout transfers. A nonlinearly-mixed state
  code that a downstream area could still read out would score at chance here.
  That is the standard, deliberate meaning of "abstract" in this framework, but it
  is a choice, not a fact about the brain.
