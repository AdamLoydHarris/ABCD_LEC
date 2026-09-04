"""
Binary state-pair decoders generalising to held-out tasks (CCGP) — region-agnostic.

Full method + rationale: code/CCGP_STATE_PAIRS.md. The short version:

Train a binary classifier to separate two task states (e.g. A vs B) on some tasks, test it on a
HELD-OUT task. The four rewards sit at different towers in every task, so a decoder that transfers
cannot ride on place — it must have found a task-invariant ("abstract") state code. This is CCGP
(Bernardi, Salzman & Fusi 2020). Binary is a strictly weaker condition than the 4-way
leave-one-task-out decoder we already run (state_decoding_analysis.decode_states_loo_cv), hence
more sensitive: WHICH of the 6 pairs transfer is the result, not their mean.

Facts about this dataset that drive the design (all verified — see the doc; several were caught by
the synthetic controls after being got WRONG on first pass):

  1. Trial_times[:, i] is the time the animal is AT Task[i], so state i is the leg FROM goal i TO
     goal i+1. Every leg therefore has a reward at BOTH ends.
  2. A is the only reward paired with a tone, and it comes on IMMEDIATELY UPON COLLECTION of reward A
     — so it occupies only the FIRST BIT of state A (post-collection). The END of state D (the
     approach to A, BEFORE collection) is tone-free. So the tone contaminates state A ONLY; an
     auditory response there transfers across tasks trivially (same tone every task) and would
     masquerade as an abstract state code. Tone-decodable pairs = the A-pairs {AB, AC, AD}; immune =
     {BC, BD, CD}. (An earlier draft used {A, D}, an artifact of a synthetic that let the tone bump
     WRAP into the end of D — corrected to a one-sided bump; the mid-leg trim removes it either way.)
  3. Place fakes CCGP via the leg's DESTINATION tower Task[s+1], not its source: a decoder trained
     with one task and tested on a task sharing the destination positionally scores 0.921, versus
     0.404 when it does not. (Source-tower matching only gets 0.641.)
  4. A pure place code transfers BELOW chance on average (0.407), because place remaps and towers
     get systematically mis-assigned across tasks. Place is therefore mainly a source of false
     NEGATIVES, and only fakes a positive on the subset of folds where towers happen to match.

THE PRIMARY CONTROL IS TRIMMING BOTH ENDS OF EACH LEG (trim_start_bins + trim_end_bins). Both
confounds live in the reward windows at the leg boundaries, and trimming them removes both at once:
on the synthetics, trim_end_bins=15 takes the place code to 0.491 under EVERY place rule (including
'none') and takes all 6 tone pairs to 0.48-0.51. `place_control` is then cheap belt-and-braces for
mid-leg (corridor) place cells, which trimming does not touch and the synthetic does not model.

The null is the ROLE-PERMUTATION null (`null_pair_accuracy`): each training task independently gets
a random ordered pair of states to play the roles of (i,j); refit; test on the TRUE (i,j) in the
held-out task. Every training task still contributes two real, well-separated states, so within-task
geometry and n are preserved exactly — only the cross-task correspondence is destroyed, which is
precisely the quantity being claimed.

Do NOT use the tempting cheap alternative of scoring the trained decoder against mismatched ordered
test pairs. It is degenerate: balanced accuracy obeys acc(k,l) == 1 - acc(l,k), so the 12 ordered
pairs sum to exactly 6 for any decoder and any data, making the null mean a deterministic function
of the true accuracy ((6 - acc_true)/11). "true > null" then reduces algebraically to "acc_true >
0.5" and the null calibrates nothing. That constraint is why the 6x6 transfer matrix below is
DESCRIPTIVE (rotation/anchoring structure) and is never used for inference.

Required per session: 'Neurons_norm' (n_neurons, n_trials, 360) or 'Neuron_raw' + 'Trial_times'
(source='raw'), plus 'Task' (4 reward tower ids). Cells must be index-matched across the sessions
of a recday — they are within a recday, which is the only reason cross-task decoding is possible.
"""

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import glm_analysis_v2 as glm
from remapping_rotation_analysis import raw_to_norm

warnings.filterwarnings('ignore')


# ============================================================================
# Config
# ============================================================================

STATE_LABELS = ['A', 'B', 'C', 'D']
# Unordered state pairs, canonical orientation (i < j).
PAIRS = list(itertools.combinations(range(4), 2))          # AB AC AD BC BD CD
PAIR_NAMES = [f'{STATE_LABELS[i]}{STATE_LABELS[j]}' for i, j in PAIRS]
# All ordered pairs: 12 of them. One matches the trained pair, 11 form the null.
ORDERED_PAIRS = [(i, j) for i in range(4) for j in range(4) if i != j]

# The tone comes on IMMEDIATELY UPON COLLECTION of reward A, so it occupies only the FIRST BIT of
# state A (post-collection). The END of state D — the approach to reward A, BEFORE collection — is
# UNAFFECTED (per the experimenter). So only state A carries the tone, in its early bins. A tone-
# driven decoder therefore separates {A} from the rest: it can discriminate a pair only if the pair
# INVOLVES A. Tone-decodable = the A-pairs {AB, AC, AD}; immune = {BC, BD, CD}. (An earlier version
# used {A, D}; that was an artifact of the synthetic modelling the tone as a bump WRAPPING around the
# D->A boundary into the end of D, not biology. See the one-sided fix in make_synthetic_recday.)
TONE_STATES = {'A'}
TONE_PAIRS = [p for p in PAIR_NAMES if len(set(p) & TONE_STATES) == 1]        # AB AC AD
TONE_IMMUNE_PAIRS = [p for p in PAIR_NAMES if len(set(p) & TONE_STATES) != 1]  # BC BD CD

# Cyclic lag: D->A is adjacent, so AD is lag-1, not lag-3.
def _cyclic_lag(i, j, n_states=4):
    d = abs(i - j) % n_states
    return min(d, n_states - d)

PAIR_LAG = {name: _cyclic_lag(i, j) for name, (i, j) in zip(PAIR_NAMES, PAIRS)}   # AB/BC/CD/AD=1, AC/BD=2

# Palette (gridmaze-colors; matches time_vs_progress_dissociation.py).
REGION_COLORS = {'LEC': '#6667AB', 'PFC': '#FFA500'}
NULL_GREY = '#555555'
NEUTRAL = '#2C2C2A'
CEILING_COLOR = '#B4B2A9'
TRUE_COLOR = '#C03030'


@dataclass
class CCGPConfig:
    """Defaults are the ones argued for in CCGP_STATE_PAIRS.md; change them knowingly."""
    # --- data path ---
    source: str = 'norm'              # 'norm' (Neurons_norm) | 'raw' (rebuild via raw_to_norm)
    num_bins_per_state: int = 90      # named to match remapping_rotation_analysis.raw_to_norm
    num_task_states: int = 4
    # Every leg has a reward at BOTH ends, and both confounds (tone, place) live in those windows.
    # Trimming both is the PRIMARY control — it is what takes the place and tone synthetics to
    # chance. Do not lower these without re-running run_synthetic_controls.
    trim_start_bins: int = 15
    trim_end_bins: int = 15
    aggregator: str = 'mean'          # 'mean' | 'max' over the surviving phase bins
    zscore: str = 'per_task'          # 'per_task' | 'none'
    # --- inclusion ---
    min_trials: int = 8
    min_neurons: int = 12
    min_train_tasks: int = 2
    # --- decoder ---
    clf: str = 'linear_svc'           # 'linear_svc' (Fusi convention) | 'logistic'
    C: float = 1.0
    max_iter: int = 5000
    # --- controls ---
    # Belt-and-braces only: the trims above are what remove the place confound. See place_ok.
    place_control: str = 'endpoints'  # see PLACE_RULES
    n_shuffles: int = 200             # role-permutation null draws per (test task, pair)
    random_state: int = 0


# ============================================================================
# Data path
# ============================================================================

def _state_slice(s, config):
    """Phase-bin indices for state s, after trimming. See fact 1: state s STARTS at reward s."""
    nbps = config.num_bins_per_state
    lo = s * nbps + config.trim_start_bins
    hi = (s + 1) * nbps - config.trim_end_bins
    if hi <= lo:
        raise ValueError(
            f'trim_start_bins={config.trim_start_bins} + trim_end_bins={config.trim_end_bins} '
            f'leaves no bins of the {nbps}-bin state')
    return lo, hi


def _norm_to_samples(norm, config):
    """(n_neurons, n_trials, 360) -> X (n_trials*4, n_neurons), y_state, trial_id.

    One population vector per (trial, state): average over the state's surviving phase bins.
    """
    agg = np.nanmean if config.aggregator == 'mean' else np.nanmax
    n_neurons, n_trials, _ = norm.shape
    X, y, tid = [], [], []
    for t in range(n_trials):
        for s in range(config.num_task_states):
            lo, hi = _state_slice(s, config)
            v = agg(norm[:, t, lo:hi], axis=1)
            X.append(v)
            y.append(s)
            tid.append(t)
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = 0.0
    return X, np.asarray(y), np.asarray(tid)


def build_task_state_matrices(data_dic, mouse_recday, config, neuron_subset=None):
    """Per unique task: one z-scored population vector per (trial, state).

    Returns list of dicts {session, task (4,), X (n_trials*4, n_neurons), y_state, trial_id,
    neuron_idx}, all sharing the same neuron columns. Empty list if the recday is unusable.

    Deduplication to one session per unique task is mandatory (sessions [0,3] and [4,7] repeat the
    same task) — a "held-out" task would otherwise be present in training.

    Parameters
    ----------
    neuron_subset : array-like of int, optional
        Rows of `Neuron_raw` to keep, applied BEFORE the zero-variance filter. This is how a
        per-region or count-matched decoder is built. Note that the neuron index is
        depth-ordered (corr of row index with y_um = +0.976 to +0.988), so `arange(n)` is a
        superficial-biased subset and any count-matching subsample must be RANDOM.

    `neuron_idx` is the join key. The function drops neurons with no variance in any task, so
    the surviving columns are not `neuron_subset` and are not `arange(n)`; without this,
    per-region columns cannot be checked against `unit_regions`, whose join is positional.
    """
    recday_data = data_dic.get(mouse_recday)
    if not recday_data:
        return []
    sessions, _ = glm.get_sessions_for_glm(recday_data)

    tasks = []
    for sess in sessions:
        sd = recday_data[sess]
        task = np.asarray(sd.get('Task'))
        if task is None or task.shape != (config.num_task_states,):
            continue

        if config.source == 'norm':
            norm = sd.get('Neurons_norm')
            if norm is None:
                continue
            norm = np.asarray(norm, dtype=float)
        else:
            raw, tt = sd.get('Neuron_raw'), sd.get('Trial_times')
            if raw is None or tt is None:
                continue
            per_trial = [raw_to_norm(raw[n], tt, config, return_mean=False) for n in range(raw.shape[0])]
            if any(p is None for p in per_trial):
                continue
            norm = np.stack(per_trial, axis=0)             # (n_neurons, n_trials, 360)

        if norm.ndim != 3 or norm.shape[1] < config.min_trials:
            continue
        if norm.shape[2] != config.num_bins_per_state * config.num_task_states:
            continue

        if neuron_subset is not None:
            sub = np.asarray(neuron_subset, dtype=int)
            if sub.size == 0 or sub.max(initial=-1) >= norm.shape[0]:
                continue
            norm = norm[sub]

        X, y, tid = _norm_to_samples(norm, config)
        tasks.append({'session': sess, 'task': task.astype(int), 'X': X, 'y_state': y, 'trial_id': tid})

    if len(tasks) < config.min_train_tasks + 1:
        return []

    # Neuron columns must match across tasks for a cross-task decoder to mean anything.
    n_cols = {t['X'].shape[1] for t in tasks}
    if len(n_cols) != 1:
        return []

    # A neuron with no variance in ANY task can't be z-scored there, so drop it everywhere.
    keep = np.ones(tasks[0]['X'].shape[1], dtype=bool)
    for t in tasks:
        keep &= np.nanstd(t['X'], axis=0) > 0
    if keep.sum() < config.min_neurons:
        return []

    # Neuron_raw rows of the surviving columns, in column order — the positional join key.
    base = (np.arange(tasks[0]['X'].shape[1]) if neuron_subset is None
            else np.asarray(neuron_subset, dtype=int))
    surviving = base[keep]

    for t in tasks:
        t['neuron_idx'] = surviving
        Xk = t['X'][:, keep]
        if config.zscore == 'per_task':
            # Label-free (per-neuron mean/std over this task's samples), so not leakage. This is
            # what makes tasks commensurable despite firing drift across sessions.
            Xk = (Xk - Xk.mean(axis=0)) / Xk.std(axis=0)
        t['X'] = Xk
    return tasks


# ============================================================================
# Place control
# ============================================================================

def _legs(task):
    """State s occupies the leg from Task[s] to Task[s+1] (fact 1)."""
    n = len(task)
    return [(int(task[s]), int(task[(s + 1) % n])) for s in range(n)]


PLACE_RULES = ['none', 'source', 'endpoints', 'leg_exact', 'leg_anyshare']


def place_ok(task_train, task_test, pair, rule):
    """Can this training task be used to decode `pair` against this test task without place leaking?

    The decoded state s occupies the leg Task[s] -> Task[s+1], so the animal visits BOTH towers
    during it and either can carry a place signal. Rules, weakest to strongest:

    'source'       — the state's own (source) tower must differ. INSUFFICIENT: the destination
                     Task[s+1] is what actually drives the confound (0.921 vs 0.641 for the source
                     on the place synthetic), and this rule ignores it.
    'leg_exact'    — the (source, destination) tuple must differ. Weakest of all: one endpoint
                     differing is enough to pass, so it is nearly a no-op (4.93/5 tasks kept, and
                     it leaves the place synthetic exactly where 'none' does).
    'endpoints'    — source AND destination must each differ, positionally. Costs 3.46/5.
    'leg_anyshare' — the two legs must share no tower at all. The only rule that also breaks the
                     cross case (train's source == test's destination), where one place cell is
                     active during leg s in both tasks at different phases. Costs 1.80/5 and loses
                     all training data in 12.8% of folds.

    NOTE: with the default trims, the place synthetic is at chance under EVERY rule including
    'none' — trimming the leg ends, not this filter, is what removes the confound. `place_control`
    is retained for mid-leg (corridor) place cells, which trimming does not touch.
    """
    if rule == 'none':
        return True
    if rule not in PLACE_RULES:
        raise ValueError(f'unknown place_control rule: {rule!r} (expected one of {PLACE_RULES})')
    Ltr, Lte = _legs(task_train), _legs(task_test)
    n = len(task_train)
    for s in pair:
        if rule == 'source':
            if int(task_train[s]) == int(task_test[s]):
                return False
        elif rule == 'endpoints':
            if (int(task_train[s]) == int(task_test[s])
                    or int(task_train[(s + 1) % n]) == int(task_test[(s + 1) % n])):
                return False
        elif rule == 'leg_exact':
            if Ltr[s] == Lte[s]:
                return False
        elif rule == 'leg_anyshare':
            if set(Ltr[s]) & set(Lte[s]):
                return False
    return True


def place_matched(task_train, task_test, pair):
    """Did the decoded legs share ANY tower between train and test? (the §5.2 contrast)

    Deliberately the most permissive definition — any shared tower is an opportunity for place to
    leak, so this flags every fold where that was possible, not just positional matches.
    """
    Ltr, Lte = _legs(task_train), _legs(task_test)
    return any(bool(set(Ltr[s]) & set(Lte[s])) for s in pair)


# ============================================================================
# Core
# ============================================================================

def _make_clf(config):
    if config.clf == 'linear_svc':
        return LinearSVC(C=config.C, dual=True, max_iter=config.max_iter,
                         random_state=config.random_state)
    if config.clf == 'logistic':
        return LogisticRegression(C=config.C, max_iter=config.max_iter,
                                  random_state=config.random_state)
    raise ValueError(f'unknown clf: {config.clf}')


def _fit_roles(tasks_train, roles, config):
    """Fit a binary decoder where training task t contributes states roles[t] = (a, b) as (class0, class1).

    roles == [pair] * len(tasks_train) gives the real decoder. Independent random roles per task
    give the null: each task still contributes two real, well-separated states, so within-task
    geometry survives and only the cross-task correspondence is destroyed.
    """
    Xs, ys = [], []
    for t, (a, b) in zip(tasks_train, roles):
        m = np.isin(t['y_state'], [a, b])
        Xs.append(t['X'][m])
        ys.append((t['y_state'][m] == b).astype(int))
    X, y = np.vstack(Xs), np.concatenate(ys)
    if len(np.unique(y)) < 2:
        return None
    clf = _make_clf(config)
    clf.fit(X, y)
    return clf


def _fit_pair(tasks_train, pair, config):
    """The real decoder: every training task contributes its true `pair`."""
    return _fit_roles(tasks_train, [pair] * len(tasks_train), config)


def null_pair_accuracy(tasks_train, task_test, pair, config, rng):
    """Role-permutation null (see module docstring): the accuracy the decoder gets when the
    cross-task state correspondence is scrambled but within-task geometry is untouched.

    Returns an array of `config.n_shuffles` balanced accuracies on the TRUE (i, j) test labels.
    """
    accs = []
    for _ in range(config.n_shuffles):
        roles = [tuple(rng.choice(config.num_task_states, size=2, replace=False))
                 for _ in tasks_train]
        clf = _fit_roles(tasks_train, roles, config)
        if clf is None:
            continue
        accs.append(_score_ordered(clf, task_test, pair[0], pair[1]))
    return np.asarray(accs, dtype=float)


def _score_ordered(clf, task_test, k, l):
    """Apply a trained decoder to ordered test pair (k, l): k plays class0, l plays class1.

    Balanced accuracy, so class imbalance can't inflate it. Note acc(k,l) == 1 - acc(l,k).
    """
    m = np.isin(task_test['y_state'], [k, l])
    if m.sum() == 0:
        return np.nan
    y = (task_test['y_state'][m] == l).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return balanced_accuracy_score(y, clf.predict(task_test['X'][m]))


def _within_task_cv(task, pair, config):
    """Leave-one-trial-out within a single task — the ceiling (§5.1).

    Distinguishes "no state code" from "state code that isn't abstract"; without it a chance-level
    CCGP is uninterpretable.
    """
    i, j = pair
    m = np.isin(task['y_state'], [i, j])
    X, y, tid = task['X'][m], (task['y_state'][m] == j).astype(int), task['trial_id'][m]
    trials = np.unique(tid)
    if len(trials) < 3:
        return np.nan
    yt, yp = [], []
    for t in trials:
        te = tid == t
        if len(np.unique(y[~te])) < 2 or te.sum() == 0:
            continue
        clf = _make_clf(config)
        clf.fit(X[~te], y[~te])
        yp.append(clf.predict(X[te]))
        yt.append(y[te])
    if not yt:
        return np.nan
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    if len(np.unique(yt)) < 2:
        return np.nan
    return balanced_accuracy_score(yt, yp)


def run_ccgp_recday(data_dic, mouse_recday, config, tasks=None,
                    with_ceiling=True, with_null=True, with_transfer=True):
    """CCGP for one recday: every (held-out task) x (state pair).

    Returns {'ccgp': df, 'transfer': df}.

    `ccgp`     — one row per (test_task, pair): the CCGP `acc`, the role-permutation null summary,
                 the within-task `ceiling`, and the place-matching flag.
    `transfer` — one row per (test_task, train_pair, test_pair_ordered): DESCRIPTIVE only (rotation
                 / anchoring structure, §5.3). Not a null — see the module docstring.
    """
    if tasks is None:
        tasks = build_task_state_matrices(data_dic, mouse_recday, config)
    if not tasks:
        return {'ccgp': pd.DataFrame(), 'transfer': pd.DataFrame()}

    rng = np.random.default_rng(config.random_state)
    rows, trows = [], []
    for te_idx, task_test in enumerate(tasks):
        others = [t for k, t in enumerate(tasks) if k != te_idx]
        for pair, pname in zip(PAIRS, PAIR_NAMES):
            train = [t for t in others
                     if place_ok(t['task'], task_test['task'], pair, config.place_control)]
            if len(train) < config.min_train_tasks:
                continue
            clf = _fit_pair(train, pair, config)
            if clf is None:
                continue

            acc = _score_ordered(clf, task_test, pair[0], pair[1])
            null = (null_pair_accuracy(train, task_test, pair, config, rng)
                    if with_null else np.array([]))
            null = null[np.isfinite(null)]

            rows.append({
                'mouse_recday': mouse_recday,
                'mouse': mouse_recday[:4],
                'test_session': task_test['session'],
                'test_task': str(task_test['task'].tolist()),
                'pair': pname,
                'acc': acc,
                'null_mean': null.mean() if null.size else np.nan,
                'null_std': null.std() if null.size else np.nan,
                # +1 smoothing: a permutation p can never be 0.
                'p_perm': ((1 + (null >= acc).sum()) / (1 + null.size)) if null.size else np.nan,
                'ceiling': _within_task_cv(task_test, pair, config) if with_ceiling else np.nan,
                # The §5.2 place contrast. The boolean is coarse — under place_control='endpoints'
                # ~87% of folds have SOME training task sharing a tower, so it barely splits the
                # data. The fraction is the usable version (mean 0.48, full 0-1 range).
                'place_matched': any(place_matched(t['task'], task_test['task'], pair) for t in train),
                'frac_train_matched': float(np.mean(
                    [place_matched(t['task'], task_test['task'], pair) for t in train])),
                'lag': PAIR_LAG[pname],
                'tone_available': pname in TONE_PAIRS,
                'n_train_tasks': len(train),
                'n_neurons': task_test['X'].shape[1],
                'n_trials_test': len(np.unique(task_test['trial_id'])),
            })

            if with_transfer:
                for (k, l) in ORDERED_PAIRS:
                    trows.append({
                        'mouse_recday': mouse_recday,
                        'mouse': mouse_recday[:4],
                        'test_task': str(task_test['task'].tolist()),
                        'train_pair': pname,
                        'test_pair_ordered': f'{STATE_LABELS[k]}{STATE_LABELS[l]}',
                        'test_pair': ''.join(sorted(STATE_LABELS[k] + STATE_LABELS[l])),
                        # Only the canonical (i<j) orientation is comparable to the diagonal;
                        # the flipped orientation is just 1 - acc.
                        'is_canonical': k < l,
                        'is_true_pair': (k, l) == pair,
                        'acc': _score_ordered(clf, task_test, k, l),
                    })
    return {'ccgp': pd.DataFrame(rows), 'transfer': pd.DataFrame(trows)}


def run_ccgp_batch(data_dic, config, mouse_recdays=None, verbose=True, n_jobs=1, **kw):
    """Run every recday. Returns {'ccgp': df, 'transfer': df} concatenated across recdays.

    The null dominates the runtime: a real decoder fits in ~2 ms because the states are separable,
    but a role-permuted one runs to `max_iter` (~47 ms) because scrambled labels are not. Do NOT
    "optimise" that away by lowering max_iter — it would under-converge the null ONLY, weakening the
    comparison decoder and biasing the test anti-conservatively. Parallelise instead (n_jobs), which
    changes nothing statistically.
    """
    if mouse_recdays is None:
        mouse_recdays = list(data_dic.keys())

    def _one(mr):
        try:
            res = run_ccgp_recday(data_dic, mr, config, **kw)
        except Exception as e:                                   # noqa: BLE001
            return mr, None, repr(e)
        if res['ccgp'].empty:
            return mr, None, 'no usable tasks'
        return mr, res, None

    if n_jobs == 1:
        out = [_one(mr) for mr in mouse_recdays]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one)(mr) for mr in mouse_recdays)

    ccgp, transfer, skipped = [], [], []
    for mr, res, err in out:
        if res is None:
            skipped.append((mr, err))
            continue
        ccgp.append(res['ccgp'])
        transfer.append(res['transfer'])
        if verbose:
            d = res['ccgp']
            print(f'  {mr}: {d["test_task"].nunique()} tasks, {d["n_neurons"].iloc[0]} neurons, '
                  f'CCGP {d["acc"].mean():.3f} (null {d["null_mean"].mean():.3f}, '
                  f'ceiling {d["ceiling"].mean():.3f})')
    if verbose and skipped:
        print(f'\nskipped {len(skipped)}:')
        for mr, why in skipped:
            print(f'  {mr}: {why}')
    return {'ccgp': pd.concat(ccgp, ignore_index=True) if ccgp else pd.DataFrame(),
            'transfer': pd.concat(transfer, ignore_index=True) if transfer else pd.DataFrame()}


# ============================================================================
# Leg-window sweep — where along the leg does (cross-task) state info live?
# ============================================================================

# name -> (trim_start_bins, trim_end_bins) within the 90-bin state. The headline CCGP uses 'mid'.
# The others deliberately RELAX the confound control: 'full'/'early'/'post_reward' re-include the
# reward windows, where the tone (early state A) and reward-tower place live. They are DIAGNOSTIC —
# they show where state information sits and whether any cross-task transfer is a reward/tone artifact.
LEG_WINDOWS = {
    'post_reward': (0, 75),   # bins 0-15   (immediate post-reward; the tone window in state A)
    'early':       (0, 60),   # bins 0-30   (early goal progress)
    'mid':         (15, 15),  # bins 15-75  (the headline: reward windows trimmed)
    'late':        (60, 0),   # bins 60-90  (approach to the next reward)
    'full':        (0, 0),    # bins 0-90   (entire leg)
}
# Order along the leg for plotting (post_reward -> late), with 'full' shown separately as a reference.
_WINDOW_ORDER = ['post_reward', 'early', 'mid', 'late', 'full']


def run_ccgp_windows(data_dic, config, mouse_recdays=None, windows=None, n_shuffles=50,
                     n_jobs=4, verbose=True):
    """Run the CCGP decoders over several leg WINDOWS (see LEG_WINDOWS).

    Each window is just a different (trim_start_bins, trim_end_bins); the data path is unchanged.
    `place_control` stays as configured (reward-tower place is controlled in every window). Returns a
    tidy `ccgp` DataFrame with an added `window` (and `bins`) column; transfer/6x6 are skipped.
    Cheaper `n_shuffles` (default 50) since this is a descriptive sweep, not the headline inference.
    """
    from dataclasses import replace
    windows = windows or LEG_WINDOWS
    out = []
    for name in [w for w in _WINDOW_ORDER if w in windows] + \
                [w for w in windows if w not in _WINDOW_ORDER]:
        ts, te = windows[name]
        cfg = replace(config, trim_start_bins=ts, trim_end_bins=te, n_shuffles=n_shuffles)
        if verbose:
            print(f'window {name}: bins {ts}-{config.num_bins_per_state - te}')
        d = run_ccgp_batch(data_dic, cfg, mouse_recdays=mouse_recdays, verbose=False,
                           n_jobs=n_jobs, with_transfer=False, with_ceiling=True)['ccgp']
        if d.empty:
            continue
        d = d.copy()
        d['window'] = name
        d['bins'] = f'{ts}-{config.num_bins_per_state - te}'
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ============================================================================
# Summaries / statistics
# ============================================================================

def summarise_pairs(ccgp):
    """Per state-pair, pooled over recdays: CCGP vs its role-permutation null, plus the ceiling.

    The pattern ACROSS pairs is the result (§6 of the doc) — never report their mean alone.
    """
    out = ccgp.groupby('pair').agg(
        mean=('acc', 'mean'), sem=('acc', 'sem'), count=('acc', 'count'),
        null=('null_mean', 'mean'), null_std=('null_std', 'mean'),
        ceiling=('ceiling', 'mean')).reset_index()
    out['lag'] = out['pair'].map(PAIR_LAG)
    out['tone_available'] = out['pair'].isin(TONE_PAIRS)
    return out.set_index('pair').reindex(PAIR_NAMES).reset_index()


def summarise_recdays(ccgp):
    """One row per recday — the unit of group-level inference (§4.3)."""
    out = ccgp.groupby(['mouse_recday', 'mouse']).agg(
        ccgp=('acc', 'mean'), null=('null_mean', 'mean'), ceiling=('ceiling', 'mean'),
        n_neurons=('n_neurons', 'first')).reset_index()
    # The tone diagnostic (§5.4): A-pairs and non-A-pairs are reported separately, never pooled.
    for label, pairs in [('ccgp_tone', TONE_PAIRS), ('ccgp_immune', TONE_IMMUNE_PAIRS)]:
        sub = ccgp[ccgp['pair'].isin(pairs)].groupby('mouse_recday')['acc'].mean().rename(label)
        out = out.join(sub, on='mouse_recday')
    return out


def test_ccgp(ccgp):
    """Group-level test: per-recday CCGP vs its own role-permutation null (§4.3).

    Wilcoxon signed-rank across recdays, plus a mouse-level test since recdays within a mouse are
    not independent. The A / non-A split is reported separately because an A-only effect is a tone
    response, not an abstract state code.
    """
    rec = summarise_recdays(ccgp).dropna(subset=['ccgp', 'null'])
    res = {'n_recdays': len(rec), 'n_mice': rec['mouse'].nunique(),
           'ccgp_mean': rec['ccgp'].mean(), 'null_mean': rec['null'].mean(),
           'ceiling_mean': rec['ceiling'].mean()}
    if len(rec) >= 5:
        res['p_recday'] = stats.wilcoxon(rec['ccgp'], rec['null'])[1]
    mouse = rec.groupby('mouse')[['ccgp', 'null']].mean()
    res['n_mice_used'] = len(mouse)
    res['ccgp_mouse_mean'] = mouse['ccgp'].mean()
    if len(mouse) >= 5:
        res['p_mouse'] = stats.wilcoxon(mouse['ccgp'], mouse['null'])[1]
    for label in ['ccgp_tone', 'ccgp_immune']:
        sub = rec.dropna(subset=[label])
        res[f'{label}_mean'] = sub[label].mean()
        if len(sub) >= 5:
            res[f'p_{label}'] = stats.wilcoxon(sub[label], sub['null'])[1]
    return res


def place_control_cost(tasks_by_recday, rule):
    """How many training tasks survive `rule`, per (test task x pair)? The data cost of a rule."""
    kept = []
    for tasks in tasks_by_recday:
        for te, t_test in enumerate(tasks):
            others = [t for k, t in enumerate(tasks) if k != te]
            for pair in PAIRS:
                kept.append(sum(place_ok(t['task'], t_test['task'], pair, rule) for t in others))
    kept = np.asarray(kept)
    return {'rule': rule, 'mean_kept': kept.mean(), 'median_kept': np.median(kept),
            'frac_ge2': (kept >= 2).mean(), 'frac_zero': (kept == 0).mean(), 'n_folds': len(kept)}


# ============================================================================
# Synthetic controls (§7) — the gate
# ============================================================================

def _bump(centre, width, n_bins, circular=True):
    """Gaussian bump over phase bins, wrapping around the A->B->C->D->A cycle."""
    x = np.arange(n_bins)
    d = x - centre
    if circular:
        d = (d + n_bins / 2) % n_bins - n_bins / 2
    return np.exp(-0.5 * (d / width) ** 2)


def make_synthetic_recday(kind, n_neurons=80, n_trials=18, n_tasks=6, noise=1.0,
                          config=None, seed=0, tone_width=6.0, tower_width=6.0,
                          phase_width=25.0, gain=3.0):
    """A data_dic-shaped synthetic recday that flows through the real pipeline unmodified.

    kind:
      'abstract' — task-phase tuning, identical across tasks        -> CCGP ~ 1 on all 6 pairs
      'place'    — tuning follows reward towers, no state code      -> must be ~0.5 WITH the place
                   control and >0.5 without it. THIS IS THE GATE.
      'tone'     — response only to the tone at reward A            -> high on A-pairs, ~0.5 on
                   BC/BD/CD, and dies once trim_start_bins > tone_width
      'noise'    — nothing                                          -> ~0.5

    Tasks are random 4-subsets of towers 1..9, matching the real overlap structure (§2.5).
    """
    config = config or CCGPConfig()
    rng = np.random.default_rng(seed)
    nbps, nst = config.num_bins_per_state, config.num_task_states
    n_bins = nbps * nst

    tasks = [rng.choice(np.arange(1, 10), size=nst, replace=False) for _ in range(n_tasks)]

    # Per-neuron preferences, fixed across tasks (that is the point: the *neuron* is the same cell).
    pref_phase = rng.uniform(0, n_bins, n_neurons)
    pref_tower = rng.integers(1, 10, n_neurons)

    sessions = {}
    for si, task in enumerate(tasks):
        base = np.zeros((n_neurons, n_bins))
        if kind == 'abstract':
            for n in range(n_neurons):
                base[n] = gain * _bump(pref_phase[n], phase_width, n_bins)
        elif kind == 'place':
            # A place cell fires whenever the animal is AT its tower. State s runs Task[s] ->
            # Task[s+1], so the animal is at Task[s] at the leg's START and Task[s+1] at its END.
            for n in range(n_neurons):
                for s in range(nst):
                    if task[s] == pref_tower[n]:
                        base[n] += gain * _bump(s * nbps, tower_width, n_bins)
                    if task[(s + 1) % nst] == pref_tower[n]:
                        base[n] += gain * _bump((s + 1) * nbps % n_bins, tower_width, n_bins)
        elif kind == 'tone':
            # The tone comes on immediately UPON COLLECTION of reward A, so it occupies only the
            # first bins of state A (post-collection) — in every task. circular=False makes the bump
            # ONE-SIDED (rising from bin 0 into A, NOT wrapping back into the end of state D, which is
            # the pre-collection approach and is tone-free).
            half = n_neurons // 2
            for n in range(half):
                base[n] = gain * _bump(0.0, tone_width, n_bins, circular=False)
        elif kind != 'noise':
            raise ValueError(f'unknown synthetic kind: {kind}')

        act = np.clip(base[:, None, :] + rng.normal(0, noise, (n_neurons, n_trials, n_bins)), 0, None)
        tt = np.cumsum(np.full((n_trials, nst + 1), 40 * 10), axis=1)     # plausible Trial_times
        sessions[si] = {
            'Neurons_norm': act,
            'Task': np.asarray(task),
            'num_trials': n_trials,
            'num_neurons': n_neurons,
            # get_sessions_for_glm requires these to consider a session usable.
            'Neuron_raw': np.zeros((n_neurons, int(tt.max()) + 1)),
            'Locs_raw': np.zeros(int(tt.max()) + 1),
            'Trial_times': tt.astype(float),
        }
    return {f'synth_{kind}': sessions}


# (kind, place rules to sweep, symmetric trim values to sweep). Each synthetic only varies the axis
# it tests, so the gate stays cheap enough to actually run before every analysis.
#
# Trims are swept SYMMETRICALLY (start == end). This is justified by PLACE, not the tone: rewards sit
# at BOTH ends of every leg (goal s at the start, goal s+1 at the end), so place fields live in both
# reward windows and symmetric trimming is needed to remove them. The tone specifically is only at
# state A's START (post-collection; the end of D is tone-free), so start-trimming alone removes the
# tone — but symmetric trim is kept for place.
_SYNTH_GRID = [
    # THE GATE. At trim=0 place must be OFF chance (else the synthetic has no place signal and the
    # test is vacuous); at the default trim it must be AT chance.
    ('place',    ['none', 'endpoints', 'leg_anyshare'], [0, None]),
    # A real effect must survive the trims and the strictest place control.
    ('abstract', ['none', 'leg_anyshare'],              [None]),
    # Tone is at state A's START only (post-collection) => tone-decodable = A-pairs {AB, AC, AD},
    # immune = {BC, BD, CD}. Untrimmed, `ccgp_tone` >> `ccgp_immune`; trimmed, both at chance.
    ('tone',     ['none'],                              [0, None]),
    ('noise',    ['none'],                              [None]),
]


def run_synthetic_controls(config=None, seed=0, n_shuffles=20, verbose=True):
    """Run every synthetic through the real pipeline. Returns a tidy DataFrame.

    THE GATE: 'place' must be off chance at trim_end=0 and at chance under the config you intend to
    use. If it is not, the controls are broken and no result on real data means anything.

    This is not ceremony. On first pass these synthetics caught two design errors that the real data
    could never have revealed: (a) the place confound runs through the leg's DESTINATION tower, so
    the original source-tower rule controlled the wrong thing; (b) the tone contaminates state D as
    well as state A, so the original "A-pairs vs non-A-pairs" split was wrong.
    """
    from dataclasses import replace
    base = config or CCGPConfig()
    out = []
    for kind, rules, trims in _SYNTH_GRID:
        for rule in rules:
            for trim in trims:
                trim = base.trim_start_bins if trim is None else trim
                cfg = replace(base, place_control=rule, n_shuffles=n_shuffles,
                              trim_start_bins=trim, trim_end_bins=trim, random_state=seed)
                dd = make_synthetic_recday(kind, config=cfg, seed=seed)
                d = run_ccgp_recday(dd, f'synth_{kind}', cfg, with_transfer=False)['ccgp']
                row = {'kind': kind, 'rule': rule, 'trim': trim}
                if d.empty:
                    row.update(ccgp=np.nan, null=np.nan, ccgp_tone=np.nan, ccgp_immune=np.nan,
                               ceiling=np.nan, n_train_tasks=0, n_folds=0)
                else:
                    row.update(ccgp=d['acc'].mean(), null=d['null_mean'].mean(),
                               ccgp_tone=d.loc[d['tone_available'], 'acc'].mean(),
                               ccgp_immune=d.loc[~d['tone_available'], 'acc'].mean(),
                               ceiling=d['ceiling'].mean(),
                               n_train_tasks=d['n_train_tasks'].mean(), n_folds=len(d))
                out.append(row)
                if verbose:
                    print(f'  {kind:9s} {rule:13s} trim={trim:<3d} '
                          f'ccgp={row["ccgp"]:.3f} null={row["null"]:.3f} '
                          f'tone={row["ccgp_tone"]:.3f} immune={row["ccgp_immune"]:.3f} '
                          f'(n_train={row["n_train_tasks"]:.1f})')
    return pd.DataFrame(out)


# ============================================================================
# Descriptive transfer matrix
# ============================================================================

def transfer_matrix(transfer):
    """6x6 train_pair x test_pair mean accuracy (§5.3). DESCRIPTIVE — not a null.

    Canonical (i<j) orientation only: the flipped orientation is just 1 - acc, and averaging both
    would force every cell to 0.5. Diagonal = CCGP. Off-diagonal shows whether the code is
    rotation-symmetric (i.e. whether A is functionally anchored by the tone).
    """
    t = transfer[transfer['is_canonical']]
    m = t.pivot_table(index='train_pair', columns='test_pair', values='acc', aggfunc='mean')
    return m.reindex(index=PAIR_NAMES, columns=PAIR_NAMES)


# ============================================================================
# Plots
# ============================================================================

def _finish(fig, out_path):
    if out_path:
        import matplotlib as mpl
        with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0}):
            fig.savefig(out_path, bbox_inches=None, dpi=300)
    return fig


def plot_pair_ccgp(ccgp, region='', ax=None, out_path=None):
    """Fig 1 — CCGP per state pair, against its role-permutation null and the within-task ceiling.

    The pattern across pairs is the result: A-pairs (AB/AC/AD) can ride on the tone, BC/BD/CD
    cannot, so they are drawn with a visual break between them.
    """
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    s = summarise_pairs(ccgp)
    order = TONE_PAIRS + TONE_IMMUNE_PAIRS
    s = s.set_index('pair').reindex(order).reset_index()
    x = np.arange(len(order)) + np.where(np.isin(order, TONE_IMMUNE_PAIRS), 0.6, 0.0)

    n_tone = len(TONE_PAIRS)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(4.2, 2.5))
    # Ceiling as background bars: the reference the CCGP points are read against.
    ax.bar(x, s['ceiling'], width=0.72, color=CEILING_COLOR, zorder=1,
           label='within-task (ceiling)')
    # Null as a band (±1 SD of the permutation draws), not whiskers — whiskers on every pair
    # visually swamp the CCGP points, which are the result.
    lo = (s['null'] - s['null_std']).to_numpy(float)
    hi = (s['null'] + s['null_std']).to_numpy(float)
    ax.fill_between(x, lo, hi, color=NULL_GREY, alpha=0.18, lw=0, zorder=2,
                    label='permutation null ±1 SD')
    ax.plot(x, s['null'], '-', color=NULL_GREY, lw=1.0, zorder=3)
    ax.errorbar(x, s['mean'], yerr=s['sem'], fmt='o', ms=4.5, lw=1.2, capsize=2,
                color=REGION_COLORS.get(region, TRUE_COLOR), zorder=4, label='CCGP')
    # Separate the two tone groups: they are different claims, not a continuum.
    ax.axvline((x[n_tone - 1] + x[n_tone]) / 2, color='0.85', lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel('balanced accuracy')
    ax.set_xlabel('state pair')
    ax.set_title(f'{region} binary state-pair CCGP'.strip(), pad=14)
    ax.text(np.mean(x[:n_tone]), 1.005, 'tone-decodable', ha='center', fontsize=7, color=NULL_GREY)
    ax.text(np.mean(x[n_tone:]), 1.005, 'tone-immune', ha='center', fontsize=7, color=NULL_GREY)
    ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, fontsize=6.5, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_ccgp_windows(ccgp_windows, region='', ax=None, out_path=None):
    """Leg-window sweep — CCGP by leg window, split A-pairs (tone) vs non-A-pairs (immune), with the
    role-permutation null band and the within-task ceiling.

    Read: if the early / post-reward window lifts the A-pairs above null while non-A pairs stay at
    chance, and this fades toward mid/late, that is the tone (task-invariant, so it generalises) —
    validating why the headline uses the mid-leg window. Uniform lift across all pairs that survives
    would instead be a genuine early-progress state code.
    """
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    order = [w for w in _WINDOW_ORDER if w in set(ccgp_windows['window'])]
    x = np.arange(len(order))
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(4.0, 2.6))
    ax.axhline(0.5, color=NULL_GREY, lw=0.8, ls=':', zorder=0)

    def _by_window(mask, agg='mean'):
        sub = ccgp_windows[mask]
        g = sub.groupby('window')['acc'].agg(['mean', 'sem'])
        return g.reindex(order)

    # within-task ceiling (band) and role-permutation null (band), pooled over pairs
    ceil = ccgp_windows.groupby('window')['ceiling'].mean().reindex(order)
    ax.plot(x, ceil.values, '-', color=CEILING_COLOR, lw=4, alpha=0.6, zorder=1,
            label='within-task ceiling')
    nl = ccgp_windows.groupby('window')['null_mean'].agg(['mean', 'std']).reindex(order)
    ax.fill_between(x, nl['mean'] - nl['std'], nl['mean'] + nl['std'], color=NULL_GREY,
                    alpha=0.15, lw=0, zorder=1, label='null ±1 SD')

    for lab, mask, colour, mk in [
            ('A-pairs (tone)', ccgp_windows['pair'].isin(TONE_PAIRS), TRUE_COLOR, 'o'),
            ('BC/BD/CD (immune)', ccgp_windows['pair'].isin(TONE_IMMUNE_PAIRS), '#2A6FB5', 's')]:
        g = _by_window(mask)
        ax.errorbar(x, g['mean'], yerr=g['sem'], fmt=mk + '-', ms=4, lw=1.3, capsize=2,
                    color=colour, zorder=3, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w}\n{ccgp_windows[ccgp_windows.window==w]["bins"].iloc[0]}' for w in order],
                       fontsize=6.5)
    ax.set_ylabel('CCGP (balanced accuracy)')
    ax.set_xlabel('leg window (bins of the 90-bin state)')
    ax.set_title(f'{region} CCGP by leg window'.strip())
    ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, fontsize=6.5, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_by_mouse(ccgp, region='', out_path=None, sharey=True):
    """Fig 1b — the 6-pair CCGP broken out per mouse, one panel each.

    Recdays within a mouse are not independent (§4.3), so the per-mouse view is the honest one:
    it shows whether the effect is a consistent property of every animal or is carried by one.
    Points are recday means; the band is the role-permutation null.
    """
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    mice = sorted(ccgp['mouse'].unique())
    order = TONE_PAIRS + TONE_IMMUNE_PAIRS
    n_tone = len(TONE_PAIRS)
    x = np.arange(len(order)) + np.where(np.isin(order, TONE_IMMUNE_PAIRS), 0.6, 0.0)

    fig, axes = plt.subplots(1, len(mice), figsize=(1.55 * len(mice) + 0.7, 2.4),
                             sharey=sharey, squeeze=False)
    axes = axes[0]
    colour = REGION_COLORS.get(region, TRUE_COLOR)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, mice):
        d = ccgp[ccgp['mouse'] == m]
        s = summarise_pairs(d).set_index('pair').reindex(order).reset_index()
        ax.bar(x, s['ceiling'], width=0.72, color=CEILING_COLOR, zorder=1)
        lo = (s['null'] - s['null_std']).to_numpy(float)
        hi = (s['null'] + s['null_std']).to_numpy(float)
        ax.fill_between(x, lo, hi, color=NULL_GREY, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, s['null'], '-', color=NULL_GREY, lw=0.9, zorder=3)
        # Individual recdays behind the mean, so a single outlying day is visible.
        per_rec = d.groupby(['mouse_recday', 'pair'])['acc'].mean().reset_index()
        xi = {p: v for p, v in zip(order, x)}
        ax.scatter(per_rec['pair'].map(xi) + rng.uniform(-0.13, 0.13, len(per_rec)),
                   per_rec['acc'], s=4, color=colour, alpha=0.35, edgecolors='none', zorder=4)
        ax.errorbar(x, s['mean'], yerr=s['sem'], fmt='o', ms=3.5, lw=1.0, capsize=1.5,
                    color=colour, zorder=5)
        ax.axvline((x[n_tone - 1] + x[n_tone]) / 2, color='0.85', lw=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=6, rotation=90)
        ax.set_title(f'{m}\n{d["mouse_recday"].nunique()} recdays', fontsize=7)
        ax.set_ylim(0.2, 1.0)
    axes[0].set_ylabel('balanced accuracy')
    fig.suptitle(f'{region} CCGP per mouse  (bar = within-task ceiling, band = null)'.strip(),
                 fontsize=8)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_mouse_summary(ccgp_by_region, out_path=None):
    """Fig 5b — one point per mouse: CCGP vs its null, with the within-task ceiling above.

    `ccgp_by_region` = {'LEC': df} or {'LEC': df, 'PFC': df}. The compact version of plot_by_mouse:
    shows at a glance whether every animal sits on the same side of its null.
    """
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    n_mice = sum(d['mouse'].nunique() for d in ccgp_by_region.values())
    fig, ax = plt.subplots(figsize=(0.62 * n_mice + 1.6, 2.6))
    pos, ticks, labels, region_spans = 0.0, [], [], []
    for region, df in ccgp_by_region.items():
        rec = summarise_recdays(df)
        colour = REGION_COLORS.get(region, NEUTRAL)
        start = pos
        for m, g in rec.groupby('mouse'):
            ax.scatter([pos] * len(g), g['ceiling'], s=22, marker='_', color=CEILING_COLOR, zorder=2)
            ax.scatter([pos] * len(g), g['null'], s=22, marker='_', color=NULL_GREY, zorder=2)
            ax.scatter([pos] * len(g), g['ccgp'], s=9, color=colour, alpha=0.45,
                       edgecolors='none', zorder=3)
            ax.errorbar([pos], [g['ccgp'].mean()], yerr=[g['ccgp'].sem()], fmt='o', ms=4,
                        capsize=2, lw=1.2, color=colour, zorder=4)
            ticks.append(pos)                       # keep ticks with the data, not range(n)
            labels.append(f'{m}\n({len(g)})')
            pos += 1
        region_spans.append((region, start, pos - 1, colour))
        pos += 0.8
    ax.axhline(0.5, color=NULL_GREY, lw=0.8, ls=':', zorder=0)
    for region, a, b, colour in region_spans:
        ax.text((a + b) / 2, 1.0, region, ha='center', fontsize=8, color=colour,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_xlim(-0.7, pos - 0.4)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel('balanced accuracy')
    ax.set_xlabel('mouse (n recdays)')
    ax.set_title('per-mouse CCGP   (dot = recday; lower — = null, upper — = within-task ceiling)',
                 fontsize=7.5, pad=14)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_transfer_matrix(transfer, region='', ax=None, out_path=None):
    """Fig 2 — 6x6 train-pair x test-pair transfer. Descriptive: is the code A-anchored or
    rotation-symmetric? Diagonal = CCGP."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    m = transfer_matrix(transfer)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(3.2, 2.7))
    im = ax.imshow(m.values, cmap='RdBu_r', vmin=0.2, vmax=0.8, aspect='equal')
    ax.set_xticks(range(len(PAIR_NAMES)), PAIR_NAMES)
    ax.set_yticks(range(len(PAIR_NAMES)), PAIR_NAMES)
    ax.set_xlabel('tested pair')
    ax.set_ylabel('trained pair')
    ax.set_title(f'{region} transfer (descriptive)'.strip())
    fig.colorbar(im, ax=ax, label='balanced accuracy', fraction=0.046)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_tone_diagnostic(sweep, region='', ax=None, out_path=None):
    """Fig 3 — the tone control. `sweep` = DataFrame with columns trim, ccgp_tone, ccgp_immune, null.

    A tone-evoked response dies as the start of state A is trimmed; an abstract state code
    survives. If only the A-pairs are ever above null, the effect is the tone.
    """
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(3.2, 2.4))
    ax.axhline(0.5, color=NULL_GREY, lw=0.8, ls=':', zorder=0)
    ax.plot(sweep['trim'], sweep['ccgp_tone'], 'o-', ms=4, color=TRUE_COLOR,
            label='A-pairs AB/AC/AD (tone)')
    ax.plot(sweep['trim'], sweep['ccgp_immune'], 's-', ms=4, color='#2A6FB5',
            label='BC/BD/CD (tone-immune)')
    if 'null' in sweep:
        ax.plot(sweep['trim'], sweep['null'], '--', lw=1.0, color=NULL_GREY, label='null')
    ax.set_xlabel('phase bins trimmed from start of each state')
    ax.set_ylabel('balanced accuracy')
    ax.set_title(f'{region} tone control'.strip())
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_place_contrast(ccgp, region='', ax=None, out_path=None):
    """Fig 4 — CCGP vs how much place was available to the decoder.

    Binned on `frac_train_matched` (the fraction of training tasks whose decoded legs shared a
    tower with the test task) rather than the boolean, which barely splits the data: under
    place_control='endpoints' ~87% of folds have SOME matched training task.

    A direct measurement of the place contribution rather than an assumption that the control
    worked. Expected sign is NEGATIVE: on the synthetic a pure place code transfers below chance
    (place remaps, so towers get mis-assigned across tasks), so place mostly MASKS a real effect
    rather than faking one.
    """
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    d = ccgp.dropna(subset=['acc', 'frac_train_matched'])
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(3.0, 2.4))
    ax.axhline(0.5, color=NULL_GREY, lw=0.8, ls=':', zorder=0)
    edges = np.array([-0.01, 0.001, 0.34, 0.67, 1.01])
    labels = ['0', '0-⅓', '⅓-⅔', '⅔-1']
    b = pd.cut(d['frac_train_matched'], edges, labels=labels)
    g = d.groupby(b, observed=False)['acc'].agg(['mean', 'sem', 'count'])
    x = np.arange(len(labels))
    ax.errorbar(x, g['mean'], yerr=g['sem'], fmt='o-', ms=4, lw=1.0, capsize=2,
                color=REGION_COLORS.get(region, TRUE_COLOR))
    for xi, (m, n) in enumerate(zip(g['mean'], g['count'])):
        if np.isfinite(m):
            ax.annotate(f'{int(n)}', (xi, m), textcoords='offset points', xytext=(0, 7),
                        fontsize=6, color=NULL_GREY, ha='center')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('fraction of training tasks sharing a tower')
    ax.set_ylabel('CCGP (balanced accuracy)')
    ax.set_title(f'{region} place contribution'.strip())
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_region_comparison(ccgp_by_region, ax=None, out_path=None):
    """Fig 5 — LEC vs PFC. `ccgp_by_region` = {'LEC': df, 'PFC': df}. Points are recdays."""
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(2.8, 2.4))
    ax.axhline(0.5, color=NULL_GREY, lw=0.8, ls=':', zorder=0)
    rng = np.random.default_rng(0)
    for i, (region, df) in enumerate(ccgp_by_region.items()):
        rec = summarise_recdays(df)
        c = REGION_COLORS.get(region, NEUTRAL)
        ax.scatter(i + rng.uniform(-0.09, 0.09, len(rec)), rec['ccgp'], s=9, alpha=0.6,
                   color=c, edgecolors='none', zorder=2)
        ax.errorbar([i], [rec['ccgp'].mean()], yerr=[rec['ccgp'].sem()], fmt='_', ms=18,
                    lw=1.6, capsize=3, color=c, zorder=3)
        ax.scatter([i], [rec['null'].mean()], marker='_', s=180, color=NULL_GREY, zorder=3)
    ax.set_xticks(range(len(ccgp_by_region)), list(ccgp_by_region))
    ax.set_xlim(-0.5, len(ccgp_by_region) - 0.5)
    ax.set_ylabel('CCGP (per recday)')
    ax.set_title('LEC vs PFC')
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_binary_vs_fourway(ccgp, fourway, region='', ax=None, out_path=None):
    """Fig 6 — the literal test of Stefano's claim.

    `fourway` = DataFrame with columns mouse_recday, acc (4-way leave-one-task-out accuracy, chance
    0.25). Plotted against 0.5-chance binary CCGP, so each is shown relative to its own chance.
    """
    import matplotlib.pyplot as plt
    glm.apply_gridmaze_style()
    rec = summarise_recdays(ccgp).merge(
        fourway.groupby('mouse_recday')['acc'].mean().rename('fourway'),
        on='mouse_recday', how='inner')
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(2.8, 2.6))
    ax.axhline(0, color=NULL_GREY, lw=0.8, ls=':', zorder=0)
    ax.axvline(0, color=NULL_GREY, lw=0.8, ls=':', zorder=0)
    ax.scatter(rec['fourway'] - 0.25, rec['ccgp'] - 0.5, s=12,
               color=REGION_COLORS.get(region, NEUTRAL), edgecolors='none')
    ax.set_xlabel('4-way accuracy - chance (0.25)')
    ax.set_ylabel('binary CCGP - chance (0.5)')
    ax.set_title(f'{region} binary vs 4-way'.strip())
    fig.tight_layout()
    return _finish(fig, out_path)
