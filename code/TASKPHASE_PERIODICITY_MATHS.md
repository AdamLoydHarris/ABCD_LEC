# Task-phase periodicity — mathematical methods

Formal companion to `taskphase_periodicity.py`. For rationale, pilot facts, and the
decision tables, see `TASKPHASE_PERIODICITY.md`; this document gives the mathematics
of every quantity the module computes, matched to the code.

---

## 0. Notation and data representation

A recording day contains several **tasks** (unique reward configurations). Each task
is deduplicated to one session via `get_sessions_for_glm`. Neurons are index-matched
across the tasks of a day, which is what makes cross-task comparison meaningful.

Within a task, the trial-warped population tensor is

$$
F \in \mathbb{R}^{N \times T \times B}, \qquad B = S \cdot P = 4 \times 90 = 360,
$$

where $N$ neurons, $T$ trials, and $B$ **task-phase bins**: the ABCD loop is warped so
each of $S=4$ states occupies exactly $P=90$ bins. Bin $b$ belongs to state
$s(b)=\lfloor b/P\rfloor$ and within-state progress $b \bmod P$. Rewards sit at bins
$0,90,180,270$ by construction (this is important — see §4.2).

The **task phase** is the angle

$$
\theta_b = \frac{2\pi b}{B} \in [0, 2\pi),
$$

running once around the whole A→B→C→D→A loop as $b:0\to B$.

**Trial average** (over the trial axis, ignoring NaNs):

$$
M_{n,b} = \frac{1}{T}\sum_{t} F_{n,t,b}.
$$

**Per-neuron z-scoring across phase** (this is what enters the FFT):

$$
\tilde M_{n,b} = \frac{M_{n,b} - \mu_n}{\sigma_n}, \qquad
\mu_n = \frac1B\sum_b M_{n,b}, \quad
\sigma_n = \sqrt{\tfrac1B\sum_b (M_{n,b}-\mu_n)^2 }.
$$

Neurons with $\sigma_n=0$ in any task are dropped everywhere.

---

## 1. Harmonic spectrum — the headline

We ask at which **spatial frequency around the loop** the population is structured. A
component completing $h$ cycles over the loop is the $h$-th harmonic; $h=4$ is "once
per state" (the 4-fold hypothesis), $h=1$ is "one ring over the whole loop", $h=2$ is
A/C-vs-B/D alternation.

### 1.1 Discrete Fourier transform

For each neuron, the real DFT of its z-scored tuning curve:

$$
\hat M_{n,h} = \sum_{b=0}^{B-1} \tilde M_{n,b}\, e^{-i 2\pi h b / B},
\qquad h = 0,1,\dots,\lfloor B/2\rfloor .
$$

Power at harmonic $h$: $\;P_{n,h} = |\hat M_{n,h}|^2$.

### 1.2 Power fraction (the unit we report)

Because $\tilde M$ is z-scored, the DC term $P_{n,0}=0$. We normalise each neuron by
its total non-DC power so neurons are comparable regardless of firing scale:

$$
p_{n,h} = \frac{P_{n,h}}{\displaystyle\sum_{h'=1}^{\lfloor B/2\rfloor} P_{n,h'}},
\qquad h = 1,\dots,H \ (H=16).
$$

By Parseval, $\sum_{h\ge1} p_{n,h}=1$, so $p_{n,h}$ is the fraction of a neuron's
across-phase variance carried at frequency $h$. The population spectrum is
$\bar p_h = \langle p_{n,h}\rangle_n$.

This uses **magnitude only**, hence is **phase-free** (rotation-invariant): a rigid
rotation $\tilde M_{n,b}\to\tilde M_{n,(b-\delta)\bmod B}$ multiplies $\hat M_{n,h}$ by
$e^{-i2\pi h\delta/B}$ and leaves $P_{n,h}$ unchanged. This is why nothing here assumes
the structure aligns to reward / state A.

### 1.3 The existence test — within-spectrum baseline

We do **not** test $\bar p_4$ against a shuffle. Under the null "no special 4-fold
structure", $\bar p_4$ should be no larger than the harmonics that are **not multiples
of 4** — its own local baseline. Define the baseline harmonic set

$$
\mathcal{B} = \{\,1\le h\le H : h \bmod 4 \ne 0\,\}
= \{1,2,3,5,6,7,9,10,11,13,14,15\},
$$

and the **existence statistic**

$$
\boxed{\;R_4 = \frac{\bar p_4}{\dfrac{1}{|\mathcal B|}\sum_{h\in\mathcal B}\bar p_h}\;}
$$

$R_4\gg1$ means 4-fold structure beyond the smooth-trajectory falloff. (Measured LEC:
$R_4\approx4.3$; PFC $R_4\approx5$.) It is assumption-light and needs no null. `power_fractions`
returns $p_{n,h}$; `harmonic_ratio` returns $R_4$.

---

## 2. Nulls — each answers a different question

The recurring failure mode (see the persistent-homology history) is using a null that
tests something other than the claim. Three questions, three nulls.

### 2.1 Trial-bootstrap floor (for "is $R_4$ above finite-trial noise")
Resample trials with replacement, $t^{*}_1,\dots,t^{*}_T$, rebuild
$M^{*}_{n,b}=\frac1T\sum_j F_{n,t^{*}_j,b}$, recompute $R_4^{*}$. The distribution of
$R_4^{*}$ over resamples is the noise floor given $T$ trials.

### 2.2 Circular-roll — for **phase coherence only**, and why not for power
Roll each neuron independently by a random $\delta_n\in\{1,\dots,B-1\}$:

$$
\tilde M^{\mathrm{roll}}_{n,b} = \tilde M_{n,(b-\delta_n)\bmod B}.
$$

The **shift theorem** gives $\hat M^{\mathrm{roll}}_{n,h}=e^{-i2\pi h\delta_n/B}\hat M_{n,h}$,
so $|\hat M^{\mathrm{roll}}_{n,h}| = |\hat M_{n,h}|$: **the power spectrum is exactly
preserved.** Therefore circular-roll (and equivalently FFT phase-randomisation) is
**useless as a null for per-neuron power** — real data can never beat it. It changes
only the *relative* phases across neurons, so it is the correct null for the following
coherence question and nothing else. (This is the exact trap that made the PH
per-neuron circular-shift null "significant" for the wrong reason.)

### 2.3 Population phase coherence
Do neurons share a common target-harmonic phase? With complex coefficients
$\hat M_{n,4}=A_n e^{i\phi_n}$, the amplitude-weighted resultant length

$$
\mathcal R = \frac{\bigl|\sum_n A_n e^{i\phi_n}\bigr|}{\sum_n A_n} \in [0,1]
$$

is compared to its circular-roll null. $\mathcal R$ near $1$ = the population's 4-fold
component has a coherent phase; near the null = each neuron periodic but with unrelated
phase. (LEC: $\mathcal R\approx0.27$ vs null $0.10$, $p\approx0.005$.) `phase_coherence`.

---

## 3. Phase and state-invariance

### 3.1 Per-neuron phase
From $\hat M_{n,4}=A_n e^{i\phi_n}$, the peak location **within a leg** (period of the
4th harmonic is exactly $P=90$ bins) as a fraction of the leg is

$$
\varphi_n = \frac{\phi_n \bmod 2\pi}{2\pi} \in [0,1),
$$

with $\varphi_n=0$ at reward onset, $\tfrac12$ at mid-leg. We report the
amplitude-weighted circular concentration $R=|\sum_n w_n e^{i2\pi\varphi_n}|$,
$w_n=A_n/\sum A$, and the fraction of amplitude within the reward window
$\{\varphi<0.15\}\cup\{\varphi>0.85\}$. (Measured LEC: mean $\varphi\approx0.49$
— mid-leg — with only $\approx0.11$ of amplitude at the reward.)

### 3.2 State-invariance
Reshape $\tilde M_n$ into its four legs $L^{(s)}_{n}\in\mathbb R^{P}$, $s=0,\dots,3$.

**Leg-to-leg correlation** — mean pairwise Pearson correlation of the four leg
trajectories, averaged over neurons:

$$
c = \Big\langle \binom{4}{2}^{-1}\!\!\sum_{s<s'} \mathrm{corr}\big(L^{(s)}_n, L^{(s')}_n\big)\Big\rangle_n .
$$

$c\to1$ means the four legs trace the same within-leg trajectory (a clean "sub-goal
cycle ×4"); $c\to0$ means they are unrelated. (Measured LEC: $c\approx0.10$ — the legs
do **not** simply repeat.)

**Per-leg phase consistency** — for each leg take its own fundamental ($h=1$ over the
$P$ bins) $\hat L^{(s)}_{n,1}$, and measure whether a neuron peaks at the same
within-leg phase in all four legs via the resultant across legs:

$$
\rho_n = \frac{\bigl|\sum_{s} \hat L^{(s)}_{n,1}\bigr|}{\sum_{s}\bigl|\hat L^{(s)}_{n,1}\bigr|}.
$$

High $\rho$ = a state-invariant within-leg waveform. (Measured LEC $\approx0.70$: the
*phase* is fairly consistent across legs even though the full waveform, $c$, is not.)

---

## 4. Confound controls

The 4th harmonic is, by construction, the state-invariant / goal-progress component.
The controls decompose *what* produces it.

### 4.1 C1 — reward-boundary trimming
Both the tone/reward transient and place bumps live in the reward windows at leg
boundaries. Drop $\tau$ bins off each end of every leg (leg length $P\to P-2\tau$,
loop length $4(P-2\tau)$) and recompute $R_4$. A boundary transient collapses; a
genuine mid-leg cycle survives. (Measured LEC: $R_4$ falls $4.3\to2.0\to1.2$ across
$\tau=0,20,30$ — a large boundary component and a smaller surviving mid-leg one.)

### 4.2 C2 — raw-time spectrum (no warping)
The warp pins rewards to fixed bins, so *some* $h=4$ is built in. Independent check in
**raw time**: for continuous phase
$\theta(\tau)=2\pi\,(s(\tau)+g(\tau))/S$ (state $s$, within-leg progress $g\in[0,1]$
from `compute_task_state_arrays`), fit each neuron's raw rate $Y_{\tau,n}$ by ordinary
least squares on a harmonic design:

$$
D_\tau = \big[\,1,\ \cos\theta_\tau,\ \sin\theta_\tau,\ \dots,\ \cos H\theta_\tau,\ \sin H\theta_\tau\,\big],
\qquad \hat B = \arg\min_B \lVert DB - Y\rVert_2^2 = (D^\top D)^{-1} D^\top Y.
$$

With $\hat B$ rows $(a_{h,n},b_{h,n})$ per harmonic, power is $P_{h,n}=a_{h,n}^2+b_{h,n}^2$,
and $R_4$ is formed as in §1.3. Presence here shows $h=4$ is not a warping artifact.
(Measured LEC raw-time $R_4\approx8.6$ — if anything stronger unwarped.)

### 4.3 C3 — ramp versus cycle
Is the state-invariant part a smooth **closed cycle** or a goal-progress **ramp** (open
arc)? Two measures.

**Harmonic decay.** A pure sinusoid puts all power at $h=4$; a sawtooth ramp (period
$P$) has Fourier amplitudes $\propto 1/m$ at $h=4m$, i.e. power ratio

$$
\text{decay} = \frac{\bar p_{8}}{\bar p_{4}} \approx \Big(\tfrac{1}{2}\Big)^2 = 0.25 \ \text{(sawtooth)}, \qquad \approx 0\ \text{(sinusoid)}.
$$

(Measured LEC $\approx0.37$ — ramp-like.)

**Openness.** Project the mean-leg trajectory $\bar L_{n}=\frac1S\sum_s L^{(s)}_n$
(a curve in $\mathbb R^{N}$ over $P$ bins) onto its top 2 PCs, $x_1,\dots,x_P\in\mathbb R^2$,
and compare endpoint gap to path length:

$$
\text{openness} = \frac{\lVert x_1 - x_P\rVert}{\sum_{b=2}^{P}\lVert x_b - x_{b-1}\rVert}
\quad\in[0,1],
$$

$\approx0$ closed loop, $\approx1$ open arc.

---

## 5. Ring structure (whole-loop $h=1$)

"Is there anything ring-structured" = is the **fundamental** ($h=1$, one cycle per
loop) special. Two readouts.

**Neighbour ratio.** Unlike $h=4$, the fundamental has no lower harmonics to baseline
against, so compare to its immediate neighbours:

$$
R_1 = \frac{\bar p_1}{\tfrac12(\bar p_2 + \bar p_3)}.
$$

A genuine ring has $R_1\gg1$; a smooth non-ring trajectory has $\bar p_1\approx\bar p_2\approx\bar p_3$
so $R_1\approx1$.

**Winding number.** Project the full trajectory $\tilde M_{\cdot,b}$ onto its top 2 PCs,
centre it, and count net revolutions about the centroid:

$$
w = \frac{1}{2\pi}\Big| \sum_{b=1}^{B-1}\big(\angle x_{b+1} - \angle x_b\big)\Big|,
\qquad \angle x_b = \operatorname{atan2}(x_{b,2}, x_{b,1}),
$$

(with angle differences unwrapped). $w\approx1$ = one ring; $w\approx4$ = four-fold;
$w\approx0$ = no net winding. (Measured both regions: $R_1\approx1.27$, no clean
$w=1$ ⇒ **no ring** — the structure is 4-fold. Synthetic pure-ring control gives
$R_1\approx15$, $w=1$.)

---

## 6. Cross-task generalisation of within-leg progress

State *identity* remaps across tasks (the CCGP result). Does within-leg **progress**
generalise?

### 6.1 Samples
Per (trial, state $s$, progress-bin $\beta\in\{0,\dots,P_g-1\}$, $P_g=3$), average the
per-task z-scored activity over that sub-window to a population vector
$x\in\mathbb R^N$, labelled by $\beta$ (pooled over the four states).

### 6.2 Decoder and metric
A linear SVM in the primal (`LinearSVC`, `dual=False` — exact for $n_{\text{samples}}>N$
and ~130× faster on the non-separable null fits, at the same tolerance). Scored by
**balanced accuracy**

$$
\mathrm{bAcc} = \frac{1}{K}\sum_{k=1}^{K}\frac{\text{TP}_k}{\text{TP}_k+\text{FN}_k},
$$

whose expectation under any label-blind rule is $1/K$, so chance $=1/P_g$ regardless of
class imbalance.

### 6.3 Leave-one-task-out and the role-permutation null
For each held-out task, train on the other tasks' pooled samples, test on the held-out
task. The null must destroy the **cross-task** progress correspondence while preserving
within-task geometry and sample sizes: independently permute the progress labels
*within each training task*, refit, test on the true held-out labels. With $R$ draws,

$$
p = \frac{1 + \#\{\,\mathrm{acc}^{\text{null}}_r \ge \mathrm{acc}\,\}}{1 + R}.
$$

(Measured LEC full run: $\mathrm{acc}=0.656$ vs null $0.333$, chance $0.333$, Wilcoxon
$p=4.8\times10^{-7}$ over recdays.) `run_cross_task_progress`.

---

## 7. Is it spatial? — the three controls

### 7.1 Place-matched split (`run_progress_place_split`)
Pairwise: train progress on one task, test on another, for every ordered task pair,
tagged by **reward-tower overlap**

$$
o_{ab} = \frac{|\,\mathrm{Task}_a \cap \mathrm{Task}_b\,|}{S}.
$$

If cross-task progress accuracy is flat and above chance at $o_{ab}=0$ (disjoint reward
towers), reward-tower place is excluded as the driver. (Measured: LEC $0.61$ at $o=0$ vs
$0.60$ shared; PFC $0.74$ vs $0.65$.)

### 7.2 Cross-task variable decoding (`run_cross_task_variables`)
Using raw-time `prepare_session_data` outputs, one sample per (leg, time-progress-bin),
decode four labels leave-one-task-out with the §6.3 null:

| label | definition |
|---|---|
| time-progress | $\lfloor\text{GP\_binned}\rfloor$, i.e. $\lfloor P_g\, t/D\rfloor$ (elapsed-time fraction of the leg) |
| distance-progress | $\lfloor P_g\, g_{\text{dist}}\rfloor$, $\ g_{\text{dist}}=\dfrac{d_{\text{from}}}{d_{\text{from}}+d_{\text{to}}}$ (path fraction) |
| location | modal maze node in the window (allocentric — the negative control) |
| state | which state $s$ (the known-remapping baseline) |

Location is the honest control: because tasks are different reward configurations on the
*same* maze, allocentric location should be near chance cross-task **iff** the shared
corridors carry no cross-task-generalising signal. Report accuracy above each label's
own null, $\text{above}=\mathrm{acc}-\text{null}$. Time and distance progress are
collinear within a leg (their definitive dissociation is
`time_vs_progress_dissociation.py`); the separable contrast is progress vs **location**.

**Measured (mean above-null; raw acc / chance):**

| | time | distance | location | state |
|---|---|---|---|---|
| LEC | 0.33 | 0.19 | **0.31** (0.40 / 0.06) | 0.00 |
| PFC | 0.40 | 0.22 | **0.14** (0.24 / 0.06) | 0.00 |

State identity is at chance in both (CCGP remapping). The regional split is the result:
in **LEC** location generalises about as much as progress (the maze corridors are
shared) — progress is *not* separable from shared-spatial coding; in **PFC** it is
markedly progress-specific.

---

## 8. Grid-cell generalisation

A grid cell keeps **coherent phase relationships** across environments: the lattice
re-anchors (a global phase shift) but cell-to-cell offsets are preserved — unlike
place/state, which remap independently. Test this on the periodic phase.

Let $\phi_c^{(a)}=\angle\hat M^{(a)}_{c,4}$ and $A_c^{(a)}=|\hat M^{(a)}_{c,4}|$ be cell
$c$'s 4th-harmonic phase and amplitude in task $a$. For a task pair $(a,b)$, weight by
the geometric-mean amplitude $w_c=\sqrt{A_c^{(a)}A_c^{(b)}}$ (keep the top 50%), and form
the **coherence**

$$
\boxed{\;\mathcal C_{ab} = \frac{\bigl|\sum_c w_c\, e^{\,i(\phi_c^{(b)}-\phi_c^{(a)})}\bigr|}{\sum_c w_c}\;}
\;\in[0,1],
\qquad
\Delta_{ab} = \arg\!\Big(\textstyle\sum_c w_c e^{i(\phi_c^{(b)}-\phi_c^{(a)})}\Big).
$$

$\mathcal C_{ab}$ is the resultant length of the **phase differences**. If all cells are
shifted by one global rotation $\Delta$ (grid re-anchoring), the differences concentrate
at $\Delta$ and $\mathcal C_{ab}\to1$; if phases move independently (remapping),
$\mathcal C_{ab}\to$ the chance level. The recovered global shift is $\Delta_{ab}$.

Two comparisons distinguish the mechanisms:

- **Null** — permute cell identity in task $b$ before differencing; $\mathcal C\gg$ null
  ⇒ genuine coherent structure.
- **No-shift consistency** — the resultant *without* allowing a shift,
  $\big|\bar z^{(b)}\,\overline{\bar z^{(a)}}\big|$ with
  $\bar z^{(a)}=\frac{\sum_c w_c e^{i\phi_c^{(a)}}}{\sum_c w_c}$. High coherence but
  **low** no-shift consistency = phases preserved only *after* a global rotation = the
  grid-like re-anchoring signature (as opposed to phases fixed outright).

**Measured:** LEC $\mathcal C=0.82$ vs null $0.15$ ($p=2\times10^{-7}$); PFC $0.87$ vs
$0.26$ ($p=4\times10^{-6}$); both with $\Delta\approx0.25$ rad and low no-shift
consistency ($\approx0.11$) — coherent re-anchoring in both regions. **Caveat:** the
statistic requires phase *diversity* across cells; if all cells share one phase, real
$\approx$ null $\approx1$ (uninformative, not misleading). Real data has diverse mid-leg
phases, so it is in the informative regime. The population-level counterpart of this
single-cell test is the cross-task progress decoding of §6.

---

## 9. Validation — synthetic controls

Every metric is gated on synthetics pushed through the *real* pipeline
(`make_synthetic_periodicity` / `run_synthetic_controls`). Each population is
$F_{n,t,b}=\text{base}_{n}(\theta_b)+\varepsilon$, differing only in $\text{base}$:

| kind | $\text{base}_n(\theta)$ | diagnostic signature |
|---|---|---|
| `cycle` | $\cos(4\theta-\varphi_n)$, $\varphi_n$ tiled | $R_4\!\gg\!1$, decay $\approx0$, $w=4$, $\mathcal C\!\gg$ null, shift $0$ |
| `ramp` | sawtooth in each leg | decay $\approx0.25$, openness $\approx0.7$ (arc) |
| `boundary` | bump at each reward onset | $R_4$ high but dies under C1 trim; $\varphi\approx0$ |
| `state` | one bump in one state | $R_4\approx1.5$ (warp floor); leg-corr $\approx0$ |
| `noise` | $0$ | flat spectrum |
| `ring` | $\cos(\theta-\varphi_n)$ | $R_1\approx15$, $w=1$, $R_4\approx0$ |
| `grid_coherent` | $\cos(4\theta-\varphi_n-\Delta_a)$ | $\mathcal C\!\gg$ null, shift $\ne0$ |
| `grid_remap` | $\cos(4\theta-\varphi_n^{(a)})$, indep. per task | $\mathcal C\approx$ null |

The gate is that these are all discriminated (they are, in ~50 s). The `state` synthetic
also measures the **warp floor**: the real $R_4$ must exceed it.

---

## 10. Summary of estimators

| quantity | symbol / function | one-line definition |
|---|---|---|
| power fraction | $p_{n,h}$ / `power_fractions` | $\lvert\hat M_{n,h}\rvert^2 / \sum_{h'\ge1}\lvert\hat M_{n,h'}\rvert^2$ |
| 4-fold ratio | $R_4$ / `harmonic_ratio` | $\bar p_4 / \operatorname{mean}_{h\bmod4\ne0}\bar p_h$ |
| ring ratio | $R_1$ / `ring_analysis` | $\bar p_1 / \tfrac12(\bar p_2+\bar p_3)$ |
| winding | $w$ / `ring_analysis` | net revolutions of top-2-PC trajectory |
| phase coherence | $\mathcal R$ / `phase_coherence` | amplitude-weighted resultant of $\phi_n$ vs circular-roll null |
| leg correlation | $c$ / `leg_similarity` | mean pairwise corr of the 4 legs |
| decay | / `ramp_vs_cycle` | $\bar p_8/\bar p_4$ (0 sinusoid, 0.25 sawtooth) |
| openness | / `ramp_vs_cycle` | endpoint gap / path length of mean-leg PCA curve |
| progress CCGP | / `run_cross_task_progress` | LOTO balanced acc vs role-permutation null |
| grid coherence | $\mathcal C_{ab}$ / `grid_generalization` | resultant of cross-task phase differences vs cell-shuffle null |
