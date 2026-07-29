"""
Task-phase 4-periodicity in the population (LEC & PFC) — region-agnostic.

Full method + rationale: code/TASKPHASE_PERIODICITY.md. The short version:

Binary state decoders don't generalise across tasks — state IDENTITY remaps (see ccgp_state_pairs).
Given that, is there 4-fold periodic structure in the population around the A->B->C->D->A loop — a
signal that recurs at the rhythm of the 4 states — WITHOUT assuming it aligns to the tone at A?

"4-fold" = as task phase runs once round the loop (0..2pi over ABCD), a component completing 4 cycles,
one per state = power at HARMONIC 4 (and multiples 8, 12) of the 360-bin trajectory. Contrast h=1
(the 4 states on one ring) and h=2 (A/C vs B/D alternation). Phase-free throughout: we test MAGNITUDE
(rotation-invariant); any reported phase is ESTIMATED, never assumed at reward/A.

Pilot facts (121 LEC sessions; verified, drive the design):
  - h=4 dominates: ~0.098 of per-neuron non-DC power, the top harmonic in 84% of sessions; h1 (ring)
    is weak (~0.057). Ratio h4 / non-4k-harmonic baseline = 5.9.
  - But ~half of it is at the reward BOUNDARIES: trimming each leg end drops the ratio 5.9 -> 2.7 (22%
    trim) -> 1.6 (33%). A reduced mid-leg component survives; the phase peaks mid-leg, not at reward.
  - The 4 legs are barely correlated (~0.10) — NOT a clean "same sub-goal cycle x4".
  - Neurons_norm is warped to 90 bins/state with rewards pinned at 0/90/180/270, so SOME h=4 is baked
    in. The headline must be "h=4 beyond the warp floor", measured via the state-selective synthetic.

Nulls answer DIFFERENT questions and must not be swapped (the PH-history lesson):
  - existence / "is h4 special": within-spectrum (h4 vs non-4k harmonics) + trial-bootstrap floor.
  - cross-neuron phase COHERENCE: per-neuron circular-roll (this PRESERVES each neuron's power
    spectrum, so it is useless for per-neuron h4 power and is used ONLY for phase coherence).
  - beyond the warp floor: the state-selective synthetic through the same warp.

Required per session: 'Neurons_norm' (n_neurons, n_trials, 360) (+ 'Neuron_raw' & 'Trial_times' for
the raw-time control), plus 'Task'. Neurons must be index-matched across a recday's sessions.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

import glm_analysis_v2 as glm

warnings.filterwarnings('ignore')


# ============================================================================
# Config / constants
# ============================================================================

STATE_LABELS = ['A', 'B', 'C', 'D']
REGION_COLORS = {'LEC': '#6667AB', 'PFC': '#FFA500'}
NULL_GREY = '#555555'
NEUTRAL = '#2C2C2A'
H4_COLOR = '#C03030'
BASELINE_COLOR = '#B4B2A9'


@dataclass
class PeriodicityConfig:
    """Defaults argued for in TASKPHASE_PERIODICITY.md; change knowingly."""
    n_bins_per_state: int = 90
    n_states: int = 4
    max_harmonic: int = 16
    target_harmonic: int = 4          # the harmonic of interest (4 = once per state)
    zscore: bool = True               # per-neuron z across phase before the FFT
    trim_reward_bins: int = 0         # C1: drop this many bins off EACH end of each leg
    # inclusion
    min_trials: int = 8
    min_neurons: int = 12
    # cross-task progress test
    n_progress_bins: int = 3
    clf_C: float = 1.0
    max_iter: int = 5000
    # nulls
    n_boot: int = 200                 # trial-bootstrap draws
    n_roll: int = 200                 # circular-roll draws (phase coherence)
    n_perm: int = 200                 # role-permutation draws (cross-task)
    min_train_tasks: int = 2
    random_state: int = 0

    @property
    def n_bins(self):
        return self.n_bins_per_state * self.n_states


# ============================================================================
# Curve builder
# ============================================================================

def _zscore_rows(M):
    """z-score each row (neuron) across the phase axis; drop zero-variance rows via a mask."""
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True)
    ok = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
    sd[sd == 0] = 1
    Z = (M - mu) / sd
    Z[~np.isfinite(Z)] = 0.0
    return Z, ok


def _trim_legs(M, config):
    """Drop trim_reward_bins off each end of every leg. M is (..., n_bins) -> (..., n_states*w)."""
    T = config.trim_reward_bins
    if T <= 0:
        return M
    nbps, nst = config.n_bins_per_state, config.n_states
    legs = M.reshape(M.shape[:-1] + (nst, nbps))[..., T:nbps - T]
    return legs.reshape(M.shape[:-1] + (nst * (nbps - 2 * T),))


def build_taskphase_curves(data_dic, mouse_recday, config):
    """Per unique task: the trial-averaged z-scored 360-curve and the raw per-trial tensor.

    Returns list of dicts {session, task (4,), M (n_neurons, 360) z-scored trial-avg,
    Z (n_neurons, n_trials, 360) raw per-trial, session_data (for the raw-time control)}.
    Neuron columns are the SAME across the returned tasks (index-matched within a recday), with
    any neuron that is zero-variance in ANY task dropped everywhere. Empty list if unusable.
    """
    recday_data = data_dic.get(mouse_recday)
    if not recday_data:
        return []
    sessions, _ = glm.get_sessions_for_glm(recday_data)
    nbps, nst = config.n_bins_per_state, config.n_states

    tasks = []
    for sess in sessions:
        sd = recday_data[sess]
        task = np.asarray(sd.get('Task'))
        norm = sd.get('Neurons_norm')
        if task.shape != (nst,) or norm is None:
            continue
        Z = np.asarray(norm, dtype=float)
        if Z.ndim != 3 or Z.shape[1] < config.min_trials or Z.shape[2] != nbps * nst:
            continue
        M = np.nanmean(Z, axis=1)                      # (n_neurons, 360) trial-averaged
        tasks.append({'session': sess, 'task': task.astype(int), 'M': M, 'Z': Z,
                      'session_data': sd})

    if len(tasks) < config.min_train_tasks + 1:
        return []
    n_cols = {t['M'].shape[0] for t in tasks}
    if len(n_cols) != 1:
        return []

    keep = np.ones(tasks[0]['M'].shape[0], dtype=bool)
    for t in tasks:
        keep &= np.nanstd(t['M'], axis=1) > 0
    if keep.sum() < config.min_neurons:
        return []

    for t in tasks:
        t['M'] = t['M'][keep]
        t['Z'] = t['Z'][keep]
        t['keep'] = keep
    return tasks


# ============================================================================
# Harmonic spectrum
# ============================================================================

def power_fractions(M, config, zscore=None):
    """(n_neurons, n_bins) curve -> (n_neurons, max_harmonic) fraction of non-DC power per harmonic.

    Column h-1 is harmonic h (h cycles per loop). Optionally z-scored and reward-trimmed first.
    """
    if zscore is None:
        zscore = config.zscore
    M = _trim_legs(M, config)
    if zscore:
        M, _ = _zscore_rows(M)
    P = np.abs(np.fft.rfft(M, axis=1)) ** 2            # (n_neurons, n//2+1); index = harmonic
    hmax = min(config.max_harmonic, P.shape[1] - 1)
    tot = P[:, 1:].sum(axis=1)
    tot[tot == 0] = 1
    return P[:, 1:hmax + 1] / tot[:, None]             # (n_neurons, hmax)


def _baseline_mask(hmax, period):
    """Harmonics that are NOT multiples of `period` — the within-spectrum baseline."""
    return np.array([h for h in range(1, hmax + 1) if h % period != 0])


def harmonic_ratio(frac, config):
    """h=target power vs the mean of the non-multiple-of-target harmonics (the existence stat).

    frac is (n_neurons, hmax). Returns (ratio, target_frac, baseline_frac), averaged over neurons.
    """
    hmax = frac.shape[1]
    k = config.target_harmonic
    base_h = _baseline_mask(hmax, k)
    tgt = frac[:, k - 1].mean()
    base = frac[:, base_h - 1].mean()
    return (tgt / base if base > 0 else np.nan), tgt, base


def run_spectrum_recday(data_dic, mouse_recday, config, tasks=None):
    """Population harmonic spectrum for one recday, averaged over its unique tasks.

    Returns {'spectrum': DataFrame(one row per harmonic), 'summary': dict}. The spectrum is the
    per-neuron power fraction averaged over neurons and tasks; the summary carries the h=target
    ratio and phase-coherence.
    """
    if tasks is None:
        tasks = build_taskphase_curves(data_dic, mouse_recday, config)
    if not tasks:
        return None

    per_task = [power_fractions(t['M'], config) for t in tasks]
    hmax = per_task[0].shape[1]
    frac = np.concatenate(per_task, axis=0)            # (neurons*tasks, hmax)
    ratio, tgt, base = harmonic_ratio(frac, config)

    spec = pd.DataFrame({
        'mouse_recday': mouse_recday, 'mouse': mouse_recday[:4],
        'harmonic': np.arange(1, hmax + 1),
        'power_frac': frac.mean(axis=0), 'sem': frac.std(axis=0) / np.sqrt(len(frac)),
        'is_target_mult': (np.arange(1, hmax + 1) % config.target_harmonic == 0),
    })
    coh = phase_coherence(tasks, config)
    summary = {
        'mouse_recday': mouse_recday, 'mouse': mouse_recday[:4],
        'n_tasks': len(tasks), 'n_neurons': tasks[0]['M'].shape[0],
        'h_target_ratio': ratio, 'h_target_frac': tgt, 'baseline_frac': base,
        'h1_frac': frac[:, 0].mean(),
        'frac_target_top': float(np.mean(frac.argmax(axis=1) == config.target_harmonic - 1)),
        **coh,
    }
    return {'spectrum': spec, 'summary': summary, 'tasks': tasks}


def run_spectrum_batch(data_dic, config, mouse_recdays=None, verbose=True):
    """Every recday. Returns {'spectrum': long DataFrame, 'summary': DataFrame}."""
    if mouse_recdays is None:
        mouse_recdays = list(data_dic.keys())
    specs, summaries, skipped = [], [], []
    for mr in mouse_recdays:
        try:
            res = run_spectrum_recday(data_dic, mr, config)
        except Exception as e:                                   # noqa: BLE001
            skipped.append((mr, repr(e)))
            continue
        if res is None:
            skipped.append((mr, 'no usable tasks'))
            continue
        specs.append(res['spectrum'])
        summaries.append(res['summary'])
        if verbose:
            s = res['summary']
            print(f'  {mr}: {s["n_tasks"]} tasks, {s["n_neurons"]} neurons, '
                  f'h{config.target_harmonic}/baseline {s["h_target_ratio"]:.2f}, '
                  f'coherence R {s["phase_coherence_R"]:.2f} (null {s["phase_coherence_null"]:.2f})')
    if verbose and skipped:
        print(f'\nskipped {len(skipped)}:')
        for mr, why in skipped:
            print(f'  {mr}: {why}')
    return {'spectrum': pd.concat(specs, ignore_index=True) if specs else pd.DataFrame(),
            'summary': pd.DataFrame(summaries)}


# ============================================================================
# Nulls
# ============================================================================

def trial_bootstrap_ratio(tasks, config, n_boot=None):
    """Finite-trial noise floor for the h=target ratio: resample trials with replacement, rebuild
    the trial-averaged curve, recompute the ratio. Returns an array of bootstrap ratios."""
    n_boot = n_boot or config.n_boot
    rng = np.random.default_rng(config.random_state)
    out = []
    for _ in range(n_boot):
        fracs = []
        for t in tasks:
            Z = t['Z']
            idx = rng.integers(0, Z.shape[1], Z.shape[1])
            M = np.nanmean(Z[:, idx], axis=1)
            fracs.append(power_fractions(M, config))
        out.append(harmonic_ratio(np.concatenate(fracs), config)[0])
    return np.asarray(out, dtype=float)


def phase_coherence(tasks, config):
    """Do neurons share a common h=target PHASE? Resultant length of the per-neuron target-harmonic
    phases (weighted by amplitude), vs a per-neuron circular-roll null.

    Circular-roll preserves each neuron's power spectrum EXACTLY, so it cannot (and must not) be used
    to test power — only this cross-neuron phase-alignment question, where it is the correct null.
    """
    rng = np.random.default_rng(config.random_state)
    k = config.target_harmonic

    def _R(M):
        Mz, ok = _zscore_rows(_trim_legs(M, config)) if config.zscore else (_trim_legs(M, config),
                                                                            np.ones(M.shape[0], bool))
        F = np.fft.rfft(Mz, axis=1)[:, k]              # complex target coeff per neuron
        F = F[ok]
        if not len(F):
            return np.nan
        return np.abs(F.sum()) / np.abs(F).sum()       # amplitude-weighted resultant length

    def _roll_rows(M, shifts):
        # per-row circular shift, vectorised (np.roll can't do per-row shifts)
        n, nb = M.shape
        idx = (np.arange(nb)[None, :] - shifts[:, None]) % nb
        return M[np.arange(n)[:, None], idx]

    real = np.nanmean([_R(t['M']) for t in tasks])
    null = []
    for _ in range(config.n_roll):
        rs = []
        for t in tasks:
            shifts = rng.integers(1, config.n_bins, t['M'].shape[0])
            rs.append(_R(_roll_rows(t['M'], shifts)))
        null.append(np.nanmean(rs))
    null = np.asarray(null)
    return {'phase_coherence_R': real, 'phase_coherence_null': null.mean(),
            'phase_coherence_p': (1 + (null >= real).sum()) / (1 + len(null))}


# ============================================================================
# Phase & state-invariance
# ============================================================================

def _target_phase(M, config):
    """Per-neuron phase (0..1 of a leg) and amplitude of the target harmonic."""
    Mz = _zscore_rows(_trim_legs(M, config))[0] if config.zscore else _trim_legs(M, config)
    F = np.fft.rfft(Mz, axis=1)[:, config.target_harmonic]
    return (np.angle(F) % (2 * np.pi)) / (2 * np.pi), np.abs(F)


def phase_analysis(tasks, config):
    """Per-neuron target-harmonic phase distribution + state-invariance.

    Returns a dict: phase concentration R (0=spread, 1=locked), mean phase in [0,1] of a leg
    (0=reward onset), fraction of amplitude within the first 15% of the leg (reward window),
    the leg-to-leg trajectory correlation, and per-leg phase consistency.
    """
    phases, amps = [], []
    for t in tasks:
        ph, am = _target_phase(t['M'], config)
        phases.append(ph)
        amps.append(am)
    ph = np.concatenate(phases)
    am = np.concatenate(amps)
    w = am / am.sum() if am.sum() > 0 else np.ones_like(am) / len(am)
    z = np.sum(w * np.exp(1j * ph * 2 * np.pi))
    return {
        'phase_R': float(np.abs(z)),
        'phase_mean_frac': float((np.angle(z) % (2 * np.pi)) / (2 * np.pi)),
        'frac_amp_at_reward': float(w[(ph < 0.15) | (ph > 0.85)].sum()),
        'leg_corr': leg_similarity(tasks, config),
        'per_leg_phase_consistency': per_leg_phase_consistency(tasks, config),
    }


def leg_similarity(tasks, config):
    """Mean per-neuron correlation between the 4 within-leg trajectories (state-invariance).

    ~1 => the four legs repeat the same trajectory; ~0 => the legs are unrelated.
    """
    nbps, nst = config.n_bins_per_state, config.n_states
    vals = []
    for t in tasks:
        legs = t['M'].reshape(t['M'].shape[0], nst, nbps)
        for n in range(legs.shape[0]):
            L = legs[n]
            if np.all(L.std(axis=1) > 0):
                C = np.corrcoef(L)
                vals.append(C[np.triu_indices(nst, 1)].mean())
    return float(np.nanmean(vals)) if vals else np.nan


def per_leg_phase_consistency(tasks, config):
    """Does each neuron peak at the SAME within-leg phase in all 4 legs?

    Per neuron, the h=1 phase of each individual 90-bin leg; resultant length across the 4 legs,
    averaged over neurons. High => a state-invariant within-leg waveform.
    """
    nbps, nst = config.n_bins_per_state, config.n_states
    vals = []
    for t in tasks:
        legs = t['M'].reshape(t['M'].shape[0], nst, nbps)
        F = np.fft.rfft(legs, axis=2)[:, :, 1]         # h=1 of each leg -> (n_neurons, n_states)
        R = np.abs(F.sum(axis=1)) / (np.abs(F).sum(axis=1) + 1e-12)
        vals.extend(R.tolist())
    return float(np.nanmean(vals)) if vals else np.nan


# ============================================================================
# Confound controls
# ============================================================================

def spectrum_vs_trim(data_dic, mouse_recday, config, trims=(0, 10, 20, 30), tasks=None):
    """C1 — h=target ratio as reward windows are trimmed off each leg end.

    A reward-boundary transient dies with trimming; a genuine mid-leg cycle survives.
    """
    from dataclasses import replace
    if tasks is None:
        tasks = build_taskphase_curves(data_dic, mouse_recday, config)
    if not tasks:
        return pd.DataFrame()
    rows = []
    for T in trims:
        cfg = replace(config, trim_reward_bins=T)
        frac = np.concatenate([power_fractions(t['M'], cfg) for t in tasks])
        ratio, tgt, base = harmonic_ratio(frac, cfg)
        rows.append({'mouse_recday': mouse_recday, 'mouse': mouse_recday[:4],
                     'trim': T, 'trim_pct': round(T / config.n_bins_per_state * 100),
                     'ratio': ratio, 'target_frac': tgt})
    return pd.DataFrame(rows)


def raw_time_spectrum(session_data, config):
    """C2 — the harmonic spectrum in RAW TIME (no warping), via a cos/sin GLM on continuous phase.

    theta(t) = 2*pi*(state + goal_progress)/n_states from compute_task_state_arrays, fit to
    Neuron_raw. Confirms h=target isn't a warping artifact. Returns (ratio, target_frac, baseline).
    """
    raw = session_data.get('Neuron_raw')
    tt = session_data.get('Trial_times')
    if raw is None or tt is None:
        return (np.nan, np.nan, np.nan)
    state, gp, *_ = glm.compute_task_state_arrays(np.asarray(tt, dtype=float).astype(int),
                                                  num_bins=config.n_bins_per_state)
    raw = np.asarray(raw, dtype=float)
    T = min(raw.shape[1], len(state))
    state, gp = state[:T], gp[:T]
    Y = raw[:, :T].T                                   # (T, n_neurons)
    theta = 2 * np.pi * (state + np.clip(gp, 0, 1)) / config.n_states
    hmax = config.max_harmonic
    cols = [np.ones(T)]
    for k in range(1, hmax + 1):
        cols += [np.cos(k * theta), np.sin(k * theta)]
    D = np.vstack(cols).T                              # (T, 1 + 2*hmax)
    B, *_ = np.linalg.lstsq(D, Y, rcond=None)          # (1+2hmax, n_neurons)
    a = B[1::2]                                          # cos coeffs (hmax, n_neurons)
    b = B[2::2]                                          # sin coeffs
    P = a ** 2 + b ** 2                                  # (hmax, n_neurons) power per harmonic
    tot = P.sum(axis=0)
    tot[tot == 0] = 1
    frac = (P / tot).T                                  # (n_neurons, hmax)
    return harmonic_ratio(frac, config)


def ramp_vs_cycle(tasks, config):
    """C3 — is the sub-goal structure a smooth CYCLE or a goal-progress RAMP (open arc)?

    Two descriptive metrics:
      decay  = power(h=2*target) / power(h=target). A pure sinusoid ~0; a sawtooth ramp ~0.25.
      openness = |start-end| / path-length of the mean-leg trajectory in its top-2 PCs.
                 ~0 closed loop, ~1 open arc. Also returns leg_snr = shared-structure amplitude.
    """
    nbps, nst = config.n_bins_per_state, config.n_states
    k = config.target_harmonic
    decays, opens, snrs = [], [], []
    for t in tasks:
        frac = power_fractions(t['M'], config)
        if 2 * k <= frac.shape[1]:
            d = frac[:, 2 * k - 1].mean() / (frac[:, k - 1].mean() + 1e-12)
            decays.append(d)
        # mean-leg trajectory (state-invariant part)
        Mz = _zscore_rows(t['M'])[0]
        legs = Mz.reshape(Mz.shape[0], nst, nbps)
        mean_leg = legs.mean(axis=1)                    # (n_neurons, 90)
        snrs.append(np.linalg.norm(mean_leg) / (np.linalg.norm(Mz) / np.sqrt(nst) + 1e-12))
        traj = PCA(n_components=2).fit_transform(mean_leg.T)   # (90, 2)
        path = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        gap = np.linalg.norm(traj[0] - traj[-1])
        opens.append(gap / (path + 1e-12))
    return {'decay_ratio': float(np.mean(decays)), 'openness': float(np.mean(opens)),
            'leg_snr': float(np.mean(snrs))}


# ============================================================================
# Ring / whole-loop (h1) readout — "is there anything ring-structured?"
# ============================================================================

def _neighbour_ratio(frac, h, exclude_period):
    """Power at harmonic h vs the mean of nearby harmonics that are NOT multiples of exclude_period.

    For the ring test (h=1, exclude_period=1 is meaningless) we instead compare h=1 to its immediate
    neighbours h2,h3 — a clean ring has h1 >> h2,h3; a smooth non-ring trajectory has h1~h2~h3.
    """
    hmax = frac.shape[1]
    if h == 1:
        nb = [x for x in (2, 3) if x <= hmax]
    else:
        nb = [x for x in (h - 1, h + 1) if 1 <= x <= hmax and x % exclude_period != 0]
    if not nb:
        return np.nan
    base = frac[:, [x - 1 for x in nb]].mean()
    return frac[:, h - 1].mean() / base if base > 0 else np.nan


def _winding_number(M):
    """How many times the population trajectory circles its centroid in the top-2 PC plane.

    ~1 => a single loop (ring). ~4 => a 4-fold trajectory (once per state). ~0 => no net winding.
    """
    Mz = _zscore_rows(M)[0]
    traj = PCA(n_components=2).fit_transform(Mz.T)     # (n_bins, 2)
    traj = traj - traj.mean(0)
    ang = np.unwrap(np.arctan2(traj[:, 1], traj[:, 0]))
    return float(abs(ang[-1] - ang[0]) / (2 * np.pi))


def ring_analysis(tasks, config):
    """Is there whole-loop (h1) ring structure, as opposed to 4-fold (h4)?

    Returns h1 vs its neighbours (h2,h3), h4 vs its neighbours for contrast, and the trajectory
    winding number (1 = one ring, 4 = four-fold). A genuine task ring => h1_ratio >> 1 and
    winding ~1; the pilot expectation (and the PH no-ring result) => h1_ratio ~1, winding ~4.
    """
    h1r, h4r, winds = [], [], []
    for t in tasks:
        frac = power_fractions(t['M'], config)
        h1r.append(_neighbour_ratio(frac, 1, config.target_harmonic))
        h4r.append(_neighbour_ratio(frac, config.target_harmonic, config.target_harmonic))
        winds.append(_winding_number(t['M']))
    return {'h1_ratio': float(np.nanmean(h1r)), 'h4_ratio': float(np.nanmean(h4r)),
            'winding': float(np.nanmean(winds))}


# ============================================================================
# Cross-task generalisation of within-leg progress
# ============================================================================

def _progress_samples(task, config):
    """Per (trial, state, progress-bin): a z-scored population vector. Labels = progress bin.

    z-scored per task across (trials, phase). Pools over the 4 states.
    """
    Z = np.asarray(task['Z'], dtype=float)             # (n_neurons, n_trials, 360)
    n, ntr, _ = Z.shape
    nbps, nst, nb = config.n_bins_per_state, config.n_states, config.n_progress_bins
    T = config.trim_reward_bins
    flat = Z.reshape(n, -1)
    mu = flat.mean(1, keepdims=True)
    sd = flat.std(1, keepdims=True)
    sd[sd == 0] = 1
    Zz = ((flat - mu) / sd).reshape(n, ntr, nbps * nst)
    w = (nbps - 2 * T) // nb                            # bins per progress sub-window
    X, y, tid = [], [], []
    for tr in range(ntr):
        for s in range(nst):
            base = s * nbps + T
            for b in range(nb):
                seg = Zz[:, tr, base + b * w: base + (b + 1) * w]
                v = np.nanmean(seg, axis=1)
                v[~np.isfinite(v)] = 0.0
                X.append(v)
                y.append(b)
                tid.append(tr)
    return np.asarray(X), np.asarray(y), np.asarray(tid)


def _clf(config):
    # dual=False (primal) — correct when n_samples > n_features, which holds here (~hundreds of
    # samples, ~tens-to-low-hundreds of neurons). ~130x faster than dual=True on the non-separable
    # NULL fits, which otherwise dominate the runtime. This is a solver choice at the same tolerance,
    # not a convergence shortcut, so it does not bias the null (unlike lowering max_iter).
    return LinearSVC(C=config.clf_C, dual=False, max_iter=config.max_iter,
                     random_state=config.random_state)


def run_cross_task_progress(data_dic, mouse_recday, config, tasks=None):
    """Does within-leg PROGRESS generalise across tasks (unlike state identity)?

    Leave-one-task-out: train a progress-bin decoder on the other tasks, test on the held-out task.
    Null = shuffle progress labels within each training task (role-permutation). Chance = 1/n_bins.
    Returns a DataFrame, one row per held-out task.
    """
    if tasks is None:
        tasks = build_taskphase_curves(data_dic, mouse_recday, config)
    if not tasks:
        return pd.DataFrame()
    samp = [_progress_samples(t, config) for t in tasks]
    rng = np.random.default_rng(config.random_state)
    rows = []
    for te in range(len(tasks)):
        Xte, yte, _ = samp[te]
        tr_idx = [i for i in range(len(tasks)) if i != te]
        if len(tr_idx) < config.min_train_tasks:
            continue
        Xtr = np.vstack([samp[i][0] for i in tr_idx])
        ytr = np.concatenate([samp[i][1] for i in tr_idx])
        clf = _clf(config).fit(Xtr, ytr)
        acc = balanced_accuracy_score(yte, clf.predict(Xte))
        null = []
        for _ in range(config.n_perm):
            yp = np.concatenate([rng.permutation(samp[i][1]) for i in tr_idx])
            c = _clf(config).fit(Xtr, yp)
            null.append(balanced_accuracy_score(yte, c.predict(Xte)))
        null = np.asarray(null)
        rows.append({'mouse_recday': mouse_recday, 'mouse': mouse_recday[:4],
                     'test_task': str(tasks[te]['task'].tolist()),
                     'progress_acc': acc, 'null_mean': null.mean(),
                     'p_perm': (1 + (null >= acc).sum()) / (1 + len(null)),
                     'chance': 1.0 / config.n_progress_bins,
                     'n_train_tasks': len(tr_idx), 'n_neurons': Xte.shape[1]})
    return pd.DataFrame(rows)


def _tower_overlap(task_a, task_b):
    """Fraction of the 4 reward towers shared between two tasks (0 = disjoint, 1 = identical set)."""
    A, B = set(int(x) for x in task_a), set(int(x) for x in task_b)
    return len(A & B) / len(A)


def run_progress_place_split(data_dic, mouse_recday, config, tasks=None):
    """Control #1 — is cross-task progress generalisation driven by shared PLACE?

    Pairwise: train a progress decoder on ONE task, test on ANOTHER, for every ordered task pair.
    Tag each pair with the reward-tower overlap of the two tasks. If progress still decodes when the
    two tasks share NO towers (overlap 0), shared place is excluded as the driver. The slope of
    accuracy vs overlap is the spatial contribution; the intercept at overlap 0 is the place-free
    generalisation.

    Returns a DataFrame, one row per ordered (train_task, test_task) pair.
    """
    if tasks is None:
        tasks = build_taskphase_curves(data_dic, mouse_recday, config)
    if len(tasks) < 2:
        return pd.DataFrame()
    samp = [_progress_samples(t, config) for t in tasks]
    rows = []
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            if i == j:
                continue
            Xtr, ytr, _ = samp[i]
            Xte, yte, _ = samp[j]
            clf = _clf(config).fit(Xtr, ytr)
            rows.append({
                'mouse_recday': mouse_recday, 'mouse': mouse_recday[:4],
                'train_task': str(tasks[i]['task'].tolist()),
                'test_task': str(tasks[j]['task'].tolist()),
                'progress_acc': balanced_accuracy_score(yte, clf.predict(Xte)),
                'tower_overlap': _tower_overlap(tasks[i]['task'], tasks[j]['task']),
                'chance': 1.0 / config.n_progress_bins,
                'n_neurons': Xte.shape[1],
            })
    return pd.DataFrame(rows)


# ============================================================================
# Grid-cell generalisation — coherent re-anchoring of the periodic phase across tasks
# ============================================================================

def grid_generalization(tasks, config, min_amp_pct=50):
    """Does the population's periodic (h=target) code generalise like a GRID CODE across tasks?

    A grid cell keeps coherent phase relationships across environments: the lattice may re-anchor
    (a global phase shift) but cell-to-cell phase offsets are preserved. Test, per task pair, on the
    per-cell target-harmonic phase phi_c:
      coherence R = |sum_c w_c exp(i(phi_c^B - phi_c^A))| / sum_c w_c
    High R = the phase differences are concentrated => a single global rotation aligns the two tasks
    (grid-like re-anchoring). Low R = phases move independently (place-like remapping). The angle of
    the resultant is the implied global shift (0 => phases simply preserved). Null = permute cell
    identity between the two tasks.

    Returns mean coherence R (real) vs shuffled null, the mean absolute global shift, and the
    single-cell absolute phase consistency (fraction of the coherence that needs no shift).
    """
    rng = np.random.default_rng(config.random_state)
    phases, amps = [], []
    for t in tasks:
        ph, am = _target_phase(t['M'], config)
        phases.append(ph * 2 * np.pi)                  # to radians
        amps.append(am)

    Rs, shuf, shifts, aligned = [], [], [], []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            w = np.sqrt(amps[i] * amps[j])
            keep = w >= np.percentile(w, min_amp_pct)
            if keep.sum() < config.min_neurons:
                continue
            d = phases[j][keep] - phases[i][keep]
            wk = w[keep]
            z = np.sum(wk * np.exp(1j * d)) / wk.sum()
            Rs.append(np.abs(z))
            shifts.append(np.abs((np.angle(z) + np.pi) % (2 * np.pi) - np.pi))
            # aligned = resultant WITHOUT allowing a shift (phases preserved outright)
            aligned.append(np.abs(np.sum(wk * np.exp(1j * phases[j][keep])) /
                                  wk.sum() * np.conj(np.sum(wk * np.exp(1j * phases[i][keep])) /
                                                     wk.sum())))
            perm = rng.permutation(keep.sum())
            zp = np.sum(wk * np.exp(1j * (phases[j][keep][perm] - phases[i][keep]))) / wk.sum()
            shuf.append(np.abs(zp))
    if not Rs:
        return {'grid_coherence': np.nan, 'grid_coherence_null': np.nan,
                'grid_shift': np.nan, 'phase_consistency': np.nan}
    return {'grid_coherence': float(np.mean(Rs)), 'grid_coherence_null': float(np.mean(shuf)),
            'grid_shift': float(np.mean(shifts)), 'phase_consistency': float(np.mean(aligned))}


# ============================================================================
# Cross-task variable decoding (#2 location negative control + #3 time/dist/place)
# ============================================================================

def _mode(a):
    a = a[np.isfinite(a)].astype(int)
    if not len(a):
        return -1
    v, c = np.unique(a, return_counts=True)
    return int(v[c.argmax()])


def _variable_samples(session_data, config):
    """Per (leg, time-progress-bin): a z-scored population vector plus four labels, in RAW time.

    Uses prepare_session_data directly (no warping): GP_binned = time-progress bin, GP_dist_continuous
    = distance progress, Locs = per-bin node, State = which leg. Labels:
      time  = time-progress bin (0..n-1)
      dist  = distance-progress bin (0..n-1)
      loc   = modal node in the window (the negative control — allocentric location)
      state = which state/leg (0..3; the CCGP remapping baseline)
    """
    nb = config.n_progress_bins
    prep = glm.truncate_all_arrays(glm.prepare_session_data(
        session_data, gp_n_bins=nb, task=np.asarray(session_data['Task'])))
    FR = np.asarray(prep['FR'], dtype=float)
    locs = np.asarray(prep['Locs'], dtype=float)
    state = np.asarray(prep['State'], dtype=int)
    tbin = np.asarray(prep['GP_binned'], dtype=int)
    dcont = np.asarray(prep['GP_dist_continuous'], dtype=float)
    dbin = np.clip((dcont * nb).astype(int), 0, nb - 1)
    on = (locs <= 21) & np.isfinite(dcont)              # on-task bins with a defined distance

    # z-score FR per neuron across the session
    mu = FR.mean(1, keepdims=True); sd = FR.std(1, keepdims=True); sd[sd == 0] = 1
    Z = (FR - mu) / sd

    # segment into legs: a new leg starts whenever State changes
    leg = np.cumsum(np.concatenate(([0], np.abs(np.diff(state)) > 0)))
    X, y_time, y_dist, y_loc, y_state = [], [], [], [], []
    for L in np.unique(leg):
        for b in range(nb):
            m = on & (leg == L) & (tbin == b)
            if m.sum() < 3:
                continue
            X.append(Z[:, m].mean(1))
            y_time.append(b)
            y_dist.append(_mode(dbin[m].astype(float)))
            y_loc.append(_mode(locs[m]))
            y_state.append(int(np.bincount(state[m]).argmax()))
    if not X:
        return None
    return (np.asarray(X), {'time': np.asarray(y_time), 'dist': np.asarray(y_dist),
                            'location': np.asarray(y_loc), 'state': np.asarray(y_state)})


def run_cross_task_variables(data_dic, mouse_recday, config, tasks=None):
    """Which within-leg variable generalises across tasks? Decode time-progress, distance-progress,
    allocentric location, and state identity, each leave-one-task-out with a role-permutation null.

    Location is the negative control (should be ~chance cross-task — towers differ per task); state is
    the known-remapping baseline. If progress (time/dist) > null while location ~ chance, the
    generalisation is progress, not place. Returns a DataFrame, one row per (variable, test task).
    """
    recday_data = data_dic.get(mouse_recday)
    if not recday_data:
        return pd.DataFrame()
    sessions, _ = glm.get_sessions_for_glm(recday_data)
    samp = []
    for s in sessions:
        try:
            r = _variable_samples(recday_data[s], config)
        except Exception:                                        # noqa: BLE001
            r = None
        if r is not None:
            samp.append(r)
    if len(samp) < config.min_train_tasks + 1:
        return pd.DataFrame()

    rng = np.random.default_rng(config.random_state)
    rows = []
    for var in ['time', 'dist', 'location', 'state']:
        for te in range(len(samp)):
            Xte, yte = samp[te][0], samp[te][1][var]
            tr_idx = [i for i in range(len(samp)) if i != te]
            Xtr = np.vstack([samp[i][0] for i in tr_idx])
            ytr = np.concatenate([samp[i][1][var] for i in tr_idx])
            # a label only counts on the test task if it also appears in training
            shared = np.isin(yte, np.unique(ytr))
            if len(np.unique(ytr)) < 2 or shared.sum() < 3 or len(np.unique(yte[shared])) < 2:
                continue
            clf = _clf(config).fit(Xtr, ytr)
            acc = balanced_accuracy_score(yte[shared], clf.predict(Xte[shared]))
            null = []
            for _ in range(config.n_perm):
                yp = np.concatenate([rng.permutation(samp[i][1][var]) for i in tr_idx])
                c = _clf(config).fit(Xtr, yp)
                null.append(balanced_accuracy_score(yte[shared], c.predict(Xte[shared])))
            null = np.asarray(null)
            rows.append({'mouse_recday': mouse_recday, 'mouse': mouse_recday[:4],
                         'variable': var, 'test_task': te, 'acc': acc, 'null_mean': null.mean(),
                         'p_perm': (1 + (null >= acc).sum()) / (1 + len(null)),
                         'n_classes': len(np.unique(ytr)), 'chance': 1.0 / len(np.unique(ytr))})
    return pd.DataFrame(rows)


# ============================================================================
# Per-neuron periodic cells
# ============================================================================

def periodic_cells(tasks, config, pct=95):
    """Per neuron: is its h=target power an outlier in its OWN spectrum (> pct-th percentile of its
    non-multiple-of-target harmonics)? Returns a per-neuron DataFrame with h4 fraction, phase, and
    the is_periodic flag. Fractions/phases are averaged over the recday's tasks."""
    frac = np.mean([power_fractions(t['M'], config) for t in tasks], axis=0)   # (n_neurons, hmax)
    hmax = frac.shape[1]
    k = config.target_harmonic
    base_h = _baseline_mask(hmax, k) - 1
    thr = np.percentile(frac[:, base_h], pct, axis=1)
    tgt = frac[:, k - 1]
    ph = np.mean([_target_phase(t['M'], config)[0] for t in tasks], axis=0)
    return pd.DataFrame({
        'mouse_recday': tasks[0].get('mouse_recday', ''),
        'target_frac': tgt, 'baseline_pct': thr,
        'is_periodic': tgt > thr, 'phase_frac': ph,
    })


# ============================================================================
# Synthetic controls (the gate) — §5
# ============================================================================

def _phase_grid(n_bins):
    return 2 * np.pi * np.arange(n_bins) / n_bins       # 0..2pi over the whole loop


def make_synthetic_periodicity(kind, n_neurons=80, n_trials=18, n_tasks=5, noise=1.0,
                               config=None, seed=0, gain=3.0, bump_sd=6.0):
    """A data_dic-shaped synthetic recday that flows through the real pipeline unmodified.

    kind:
      'cycle'    — each neuron cos(target*theta + phi_n), phases tiled  -> clean h_target, no
                   multiples; C3 CYCLE; progress generalises across tasks.
      'ramp'     — linear goal-progress ramp, same each leg             -> h_target>2h decay; C3 RAMP.
      'boundary' — a bump only at each reward onset (leg start)         -> h_target + multiples; phase
                   at 0; dies under C1 trim.
      'state'    — one bump in ONE state per neuron, no within-leg structure -> h_target NOT elevated;
                   leg-corr ~0. THIS IS THE WARP FLOOR.
      'noise'    — flat spectrum.
      'ring'     — each neuron cos(1*theta - phi_n): a single WHOLE-LOOP ring -> h1 dominant, winding 1.
      'grid_coherent' — h_target tuning with per-neuron phase FIXED across tasks plus a per-TASK global
                   rotation -> grid_coherence high, non-zero shift (re-anchoring); progress generalises.
      'grid_remap' — h_target tuning with per-neuron phase INDEPENDENTLY random per task -> grid
                   coherence at null; progress does NOT generalise.
    """
    config = config or PeriodicityConfig()
    rng = np.random.default_rng(seed)
    nbps, nst = config.n_bins_per_state, config.n_states
    nb = nbps * nst
    k = config.target_harmonic
    theta = _phase_grid(nb)
    within = np.linspace(0, 1, nbps, endpoint=False)    # progress within a leg

    def _bump(centre, width, n):
        x = np.arange(n)
        dd = (x - centre + n / 2) % n - n / 2
        return np.exp(-0.5 * (dd / width) ** 2)

    pref = rng.uniform(0, 2 * np.pi, n_neurons)          # per-neuron preferred phase (fixed)
    pref_state = rng.integers(0, nst, n_neurons)
    sign = rng.choice([-1.0, 1.0], n_neurons)
    task_offset = rng.uniform(0, 2 * np.pi, n_tasks)     # per-task global re-anchoring shift

    sessions = {}
    for si in range(n_tasks):
        base = np.zeros((n_neurons, nb))
        if kind == 'cycle':
            for n in range(n_neurons):
                base[n] = gain * np.cos(k * theta - pref[n])
        elif kind == 'ring':
            for n in range(n_neurons):
                base[n] = gain * np.cos(theta - pref[n])
        elif kind == 'grid_coherent':
            for n in range(n_neurons):
                base[n] = gain * np.cos(k * theta - pref[n] - task_offset[si])
        elif kind == 'grid_remap':
            pt = rng.uniform(0, 2 * np.pi, n_neurons)     # independent phases THIS task
            for n in range(n_neurons):
                base[n] = gain * np.cos(k * theta - pt[n])
        elif kind == 'ramp':
            leg = np.tile(within, nst)                    # sawtooth, resets each leg
            for n in range(n_neurons):
                base[n] = gain * sign[n] * leg
        elif kind == 'boundary':
            onset = sum(_bump(s * nbps, bump_sd, nb) for s in range(nst))
            for n in range(n_neurons):
                base[n] = gain * onset
        elif kind == 'state':
            for n in range(n_neurons):
                s = pref_state[n]
                base[n] = gain * _bump(s * nbps + nbps / 2, bump_sd, nb)
        elif kind != 'noise':
            raise ValueError(f'unknown kind: {kind}')

        act = np.clip(base[:, None, :] + rng.normal(0, noise, (n_neurons, n_trials, nb)), 0, None)
        tt = np.cumsum(np.full((n_trials, nst + 1), 40 * 10), axis=1).astype(float)
        sessions[si] = {
            'Neurons_norm': act,
            'Task': rng.choice(np.arange(1, 10), size=nst, replace=False),
            'num_trials': n_trials, 'num_neurons': n_neurons,
            'Neuron_raw': np.zeros((n_neurons, int(tt.max()) + 1)),
            'Locs_raw': np.zeros(int(tt.max()) + 1),
            'Trial_times': tt,
        }
    return {f'synth_{kind}': sessions}


def run_synthetic_controls(config=None, seed=0, verbose=True):
    """Run every synthetic through the real pipeline. THE GATE: the metrics must DISCRIMINATE the
    kinds (see §5 of the doc). Returns a tidy DataFrame."""
    from dataclasses import replace
    base = replace(config or PeriodicityConfig(), n_boot=0, n_roll=40, n_perm=40)
    out = []
    for kind in ['cycle', 'ramp', 'boundary', 'state', 'noise', 'ring',
                 'grid_coherent', 'grid_remap']:
        dd = make_synthetic_periodicity(kind, config=base, seed=seed)
        mr = f'synth_{kind}'
        tasks = build_taskphase_curves(dd, mr, base)
        spec = run_spectrum_recday(dd, mr, base, tasks=tasks)['summary']
        geo = ramp_vs_cycle(tasks, base)
        ph = phase_analysis(tasks, base)
        ring = ring_analysis(tasks, base)
        grid = grid_generalization(tasks, base)
        xt = run_cross_task_progress(dd, mr, base, tasks=tasks)
        row = {'kind': kind, 'h4_ratio': spec['h_target_ratio'],
               'h1_ratio': ring['h1_ratio'], 'winding': ring['winding'],
               'decay': geo['decay_ratio'], 'openness': geo['openness'],
               'leg_corr': ph['leg_corr'], 'phase_at_reward': ph['frac_amp_at_reward'],
               'grid_coh': grid['grid_coherence'], 'grid_null': grid['grid_coherence_null'],
               'grid_shift': grid['grid_shift'],
               'progress_acc': xt['progress_acc'].mean() if len(xt) else np.nan,
               'progress_null': xt['null_mean'].mean() if len(xt) else np.nan}
        out.append(row)
        if verbose:
            print(f"  {kind:13s} h4/base={row['h4_ratio']:5.2f} h1/nbr={row['h1_ratio']:5.2f} "
                  f"wind={row['winding']:.2f} decay={row['decay']:.2f} open={row['openness']:.2f} "
                  f"gridcoh={row['grid_coh']:.2f}/{row['grid_null']:.2f} "
                  f"shift={row['grid_shift']:.2f} prog={row['progress_acc']:.2f}/{row['progress_null']:.2f}")
    return pd.DataFrame(out)


def spec_trim_ratio(summary, data_dic, mouse_recday, config, trim=20):
    """Helper: the h=target ratio at a single trim level (for the synthetic table)."""
    from dataclasses import replace
    df = spectrum_vs_trim(data_dic, mouse_recday, config, trims=(trim,))
    return float(df['ratio'].iloc[0]) if len(df) else np.nan


# ============================================================================
# Plots
# ============================================================================

def _finish(fig, out_path):
    if out_path:
        import matplotlib as mpl
        with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0}):
            fig.savefig(out_path, bbox_inches=None, dpi=300)
    return fig


def plot_spectrum(spectrum, region='', ax=None, out_path=None, config=None):
    """Fig 1 — population harmonic spectrum, target harmonic + multiples highlighted."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    k = (config or PeriodicityConfig()).target_harmonic
    g = spectrum.groupby('harmonic')['power_frac'].agg(['mean', 'sem']).reset_index()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(4.0, 2.4))
    mult = g['harmonic'] % k == 0
    ax.bar(g.loc[~mult, 'harmonic'], g.loc[~mult, 'mean'], yerr=g.loc[~mult, 'sem'],
           width=0.8, color=BASELINE_COLOR, label='other harmonics')
    ax.bar(g.loc[mult, 'harmonic'], g.loc[mult, 'mean'], yerr=g.loc[mult, 'sem'],
           width=0.8, color=REGION_COLORS.get(region, H4_COLOR),
           label=f'multiples of {k}')
    ax.set_xlabel('harmonic (cycles per ABCD loop)')
    ax.set_ylabel('power fraction')
    ax.set_title(f'{region} task-phase harmonic spectrum'.strip())
    ax.set_xticks(g['harmonic'][::2])
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_trim(trim_df, region='', ax=None, out_path=None):
    """Fig 2 (C1) — h=target ratio vs reward-window trimming, per recday + mean."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(3.0, 2.4))
    ax.axhline(1.0, color=NULL_GREY, lw=0.8, ls=':', zorder=0)
    for mr, g in trim_df.groupby('mouse_recday'):
        ax.plot(g['trim_pct'], g['ratio'], '-', lw=0.5, color=BASELINE_COLOR, alpha=0.5, zorder=1)
    g = trim_df.groupby('trim_pct')['ratio'].agg(['mean', 'sem']).reset_index()
    ax.errorbar(g['trim_pct'], g['mean'], yerr=g['sem'], fmt='o-', ms=4, lw=1.5,
                color=REGION_COLORS.get(region, H4_COLOR), zorder=3)
    ax.set_xlabel('% of each leg trimmed (each end)')
    ax.set_ylabel('h4 / baseline ratio')
    ax.set_title(f'{region} reward-boundary control'.strip())
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_phase_rose(cells, region='', ax=None, out_path=None):
    """Fig 3 — distribution of per-neuron target-harmonic phase (0=reward onset, 0.5=mid-leg)."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    per = cells[cells['is_periodic']] if 'is_periodic' in cells else cells
    ph = per['phase_frac'].to_numpy() * 2 * np.pi
    fig, ax = (ax.figure, ax) if ax is not None else \
        plt.subplots(figsize=(2.6, 2.6), subplot_kw={'projection': 'polar'})
    h, edges = np.histogram(ph, bins=18, range=(0, 2 * np.pi))
    ax.bar(edges[:-1], h, width=np.diff(edges), align='edge',
           color=REGION_COLORS.get(region, H4_COLOR), alpha=0.8)
    ax.set_theta_zero_location('E')
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['reward', '¼', 'mid', '¾'])
    ax.set_yticks([])
    ax.set_title(f'{region} 4-periodic cell phase'.strip(), fontsize=8)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_progress_ccgp(prog_df, region='', ax=None, out_path=None):
    """Fig 4 — cross-task progress decoding vs null and chance (the CCGP sequel)."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    rec = prog_df.groupby('mouse_recday').agg(acc=('progress_acc', 'mean'),
                                              null=('null_mean', 'mean'),
                                              chance=('chance', 'first')).reset_index()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(2.6, 2.6))
    rng = np.random.default_rng(0)
    ax.axhline(rec['chance'].iloc[0], color=NULL_GREY, lw=0.8, ls=':', label='chance')
    x = rng.uniform(-0.09, 0.09, len(rec))
    ax.scatter(x, rec['acc'], s=10, color=REGION_COLORS.get(region, H4_COLOR),
               alpha=0.6, edgecolors='none', label='recday')
    ax.errorbar([0], [rec['acc'].mean()], yerr=[rec['acc'].sem()], fmt='o', ms=6, lw=1.5,
                capsize=3, color=REGION_COLORS.get(region, H4_COLOR))
    ax.scatter([0.25] * len(rec), rec['null'], s=8, marker='_', color=NULL_GREY, label='null')
    ax.set_xlim(-0.4, 0.5)
    ax.set_xticks([])
    ax.set_ylabel('progress decoding (balanced acc)')
    ax.set_title(f'{region} cross-task progress'.strip())
    ax.legend(frameon=False, fontsize=7, loc='best')
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_place_split(split, region='', ax=None, out_path=None):
    """Control #1 — cross-task progress accuracy vs reward-tower overlap. Flat + above chance at
    overlap 0 => shared reward-tower place is NOT the driver."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(3.0, 2.4))
    ax.axhline(split['chance'].iloc[0], color=NULL_GREY, lw=0.8, ls=':', zorder=0, label='chance')
    b = pd.cut(split['tower_overlap'], [-0.01, 0.001, 0.26, 0.51, 1.01],
               labels=['0', '¼', '½', '¾+'])
    g = split.groupby(b, observed=False)['progress_acc'].agg(['mean', 'sem', 'count'])
    x = np.arange(len(g))
    ax.errorbar(x, g['mean'], yerr=g['sem'], fmt='o-', ms=4, lw=1.2, capsize=2,
                color=REGION_COLORS.get(region, H4_COLOR))
    for xi, (m, n) in enumerate(zip(g['mean'], g['count'])):
        if np.isfinite(m):
            ax.annotate(f'{int(n)}', (xi, m), textcoords='offset points', xytext=(0, 7),
                        fontsize=6, color=NULL_GREY, ha='center')
    ax.set_xticks(x); ax.set_xticklabels(g.index.astype(str))
    ax.set_xlabel('reward-tower overlap (train vs test task)')
    ax.set_ylabel('cross-task progress acc')
    ax.set_title(f'{region} place-matched split'.strip())
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_cross_task_variables(var_df, region='', ax=None, out_path=None):
    """#2/#3 — cross-task generalisation by decoded variable, as accuracy ABOVE its own null.
    time/dist progress travel; location travels weakly (shared corridors); state does not."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    order = ['time', 'dist', 'location', 'state']
    labels = ['time\nprogress', 'distance\nprogress', 'location\n(node)', 'state\nidentity']
    rec = var_df.groupby(['mouse_recday', 'variable']).agg(
        acc=('acc', 'mean'), null=('null_mean', 'mean')).reset_index()
    rec['above'] = rec['acc'] - rec['null']
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(3.2, 2.5))
    ax.axhline(0, color=NULL_GREY, lw=0.8, ls=':', zorder=0)
    rng = np.random.default_rng(0)
    for i, v in enumerate(order):
        s = rec[rec['variable'] == v]['above']
        ax.scatter(i + rng.uniform(-0.1, 0.1, len(s)), s, s=8, alpha=0.5,
                   color=REGION_COLORS.get(region, H4_COLOR), edgecolors='none')
        ax.errorbar([i], [s.mean()], yerr=[s.sem()], fmt='o', ms=5, lw=1.4, capsize=3,
                    color=NEUTRAL, zorder=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('cross-task acc − null')
    ax.set_title(f'{region} what generalises across tasks'.strip())
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_grid_generalization(grid_df, region='', ax=None, out_path=None):
    """Grid-cell test — per-recday phase coherence across tasks vs the cell-shuffle null.
    coherence >> null (with a non-zero global shift) = coherent re-anchoring = grid-like."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(2.6, 2.5))
    rng = np.random.default_rng(0)
    for i, (col, lab) in enumerate([('grid_coherence', 'real'), ('grid_coherence_null', 'null')]):
        s = grid_df[col].dropna()
        c = REGION_COLORS.get(region, H4_COLOR) if i == 0 else NULL_GREY
        ax.scatter(i + rng.uniform(-0.09, 0.09, len(s)), s, s=9, alpha=0.55, color=c,
                   edgecolors='none')
        ax.errorbar([i], [s.mean()], yerr=[s.sem()], fmt='o', ms=5, lw=1.4, capsize=3, color=NEUTRAL)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['coherence', 'shuffle\nnull'])
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, 1)
    ax.set_ylabel('cross-task phase coherence')
    ax.set_title(f'{region} grid-like re-anchoring'.strip())
    fig.tight_layout()
    return _finish(fig, out_path)
