"""Leave-one-session-out cross-validation for the per-neuron GLM.

Why this exists
---------------
`glm_analysis_v2.run_glm_analysis` fits `lstsq(X, frs)` over ALL samples and computes RSS,
R^2, CPD and the nested F on the same samples it fit. There is no held-out fold anywhere in
that path. Significance is a permutation test (circularly shifted firing, refit in-sample),
so in-sample optimism largely cancels for the *significance call* -- but two things do not:

  * CPD MAGNITUDES are in-sample and inflated; a richer or better-placed basis mechanically
    lowers RSS and wins (this is `glm_cv_cpd.py`'s stated reason for existing), and
  * a circular-shift null is permissive for slowly-varying regressors. A rolled trace keeps
    its autocorrelation, and the animal occupies each maze node for long runs, so place
    alignment survives shifting more than it should.

That second point is measured, not hypothetical: 80.0% of all 2651 LEC units come back
"place-tuned", with a range across brain regions of only 70.3-85.6%. The binary measure is
saturated and cannot discriminate regions at all (see `code/ANATOMY_SPLIT.md`, W0.1 gate 5).

Sessions are the right fold unit: they are different tasks, so leave-one-session-out tests
across-task generalisation -- the scientific question -- and it matches the convention
already used by `elasticnet_regression_v3.run_cross_validated_regression_v3`.

What it computes
----------------
Both models are FIT on the training sessions and SCORED on the held-out session; RSS is
pooled across folds before forming a ratio (more stable than averaging per-fold CPDs):

    cpd_cv[g]  = (RSS_reduced_heldout - RSS_full_heldout) / RSS_reduced_heldout
    r2_cv      = 1 - RSS_full_heldout / TSS_heldout          (may be negative; kept)

A held-out CPD can be negative -- dropping a group can IMPROVE held-out fit if the group was
only fitting noise. That is signal, not an error, and is not clipped.

Speed
-----
The design matrix is shared across neurons, so the pseudo-inverse of each (fold, model) is
formed once and reused for every neuron and every permutation. That turns each fit into a
matmul and makes the cross-validated version cheaper per fit than the existing in-sample
loop, which re-decomposes the same `X` inside `lstsq` for every neuron and every group.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'session_ids_from_filters', 'check_rank', 'roll_within_sessions',
    'center_within_session', 'within_session_folds',
    'cv_scores', 'cv_scores_poisson',
]


# ---------------------------------------------------------------------------
# Fold bookkeeping
# ---------------------------------------------------------------------------

def session_ids_from_filters(session_filters):
    """Per-sample session id, matching the order `FR_all` is concatenated in.

    `session_filters` is `run_glm_analysis`'s list of `(session, node_filter)`; the pooled
    arrays are concatenations of `arr[nf]` in exactly this order, so the id vector is just
    each session's label repeated `nf.sum()` times.
    """
    ids = [np.full(int(np.sum(nf)), i, dtype=int)
           for i, (_s, nf) in enumerate(session_filters)]
    return np.concatenate(ids) if ids else np.zeros(0, dtype=int)


def check_rank(X, name='design matrix', strict=False):
    """Report (and optionally enforce) full column rank.

    `parameterization='all_bins'` is rank-deficient by 8 -- every one-hot block sums to 1 per
    row, so the blocks are mutually collinear through the implicit intercept. `lstsq` returns
    the minimum-norm solution, which leaves R^2, RSS and CPD correct but makes individual
    betas meaningless. `'reference_coded'` drops one bin per block and prepends an intercept,
    giving full rank. CPD only needs RSS, so both are usable; anything reading betas (e.g.
    `_beta_direction`, which sets the SIGN in `tuned_dict`) needs the full-rank coding.
    """
    r = int(np.linalg.matrix_rank(X))
    p = int(X.shape[1])
    info = {'rank': r, 'n_cols': p, 'deficiency': p - r, 'full_rank': r == p}
    if r != p:
        msg = (f"{name}: rank {r} < {p} columns (deficient by {p - r}). "
               f"RSS/R2/CPD remain valid; individual betas are min-norm and not "
               f"interpretable.")
        if strict:
            raise AssertionError(msg)
        info['warning'] = msg
    return info


def roll_within_sessions(fr, session_ids, shifts):
    """Circularly shift each neuron's trace WITHIN each session.

    Shifting the pooled trace across session boundaries would wrap one task's firing onto
    another task's regressors, which is a different (and less conservative) null than the
    one intended. Shifting inside a session preserves that session's autocorrelation and
    destroys only its alignment to the regressors.

    `fr` is (n_neurons, T) or (T,); `shifts` is one shift per session.
    """
    fr = np.atleast_2d(fr)
    out = np.empty_like(fr)
    for k, s in enumerate(np.unique(session_ids)):
        m = session_ids == s
        out[:, m] = np.roll(fr[:, m], int(shifts[k % len(shifts)]), axis=1)
    return out


# ---------------------------------------------------------------------------
# Linear (OLS) cross-validated CPD
# ---------------------------------------------------------------------------

def _pinv(A, rcond=1e-10, ridge=0.0):
    """Moore-Penrose pseudo-inverse; tolerant of the rank-deficient `all_bins` coding.

    With `ridge > 0` returns the ridge solver `(A'A + lambda I)^-1 A'` instead. Unregularised
    OLS on 122 columns is the default because the sample-to-column ratio is ~240:1, but these
    models explain very little variance (in-sample R2 ~0.011 against a chance level of
    p/T ~ 0.004), and in that regime shrinkage can change which regressors survive
    cross-validation. Offered so that can be tested rather than assumed.
    """
    if ridge and ridge > 0:
        G = A.T @ A
        G.flat[::G.shape[0] + 1] += ridge
        return np.linalg.solve(G, A.T)
    return np.linalg.pinv(A, rcond=rcond)


def center_within_session(FR, session_ids):
    """Mean-centre each neuron's trace inside each session.

    Without this, leave-one-session-out is partly a test of whether a neuron's ABSOLUTE
    firing rate is stable across sessions, not whether its tuning is. The design carries a
    single global intercept, so a neuron whose mean rate drifts between sessions gets every
    held-out prediction offset by that drift -- while `r2_cv`'s TSS is computed about the
    held-out session's OWN mean. The model is then scored against a baseline it was never
    allowed to match, and can score below it even with perfect tuning.

    Centring grants a free per-session offset, which is the standard nuisance treatment and
    makes the CV a test of TUNING SHAPE. It uses the held-out session's own mean, which is
    legitimate for a nuisance parameter (and is the same mean `r2_cv` already baselines
    against) but should be stated: the reported CV does not test rate stability.
    """
    FR = np.atleast_2d(np.asarray(FR, dtype=float))
    out = FR.copy()
    for s in np.unique(session_ids):
        m = session_ids == s
        out[:, m] -= FR[:, m].mean(axis=1, keepdims=True)
    return out


def within_session_folds(session_ids, n_folds=6, block_len=None):
    """Fold labels that cut WITHIN each session instead of holding a whole session out.

    Leave-one-session-out and within-session CV answer different questions, and on this
    dataset the difference is the science rather than a technicality. Sessions are DIFFERENT
    TASKS, and cells remap in task space between tasks, so a LOSO model trained on tasks 1-5
    genuinely cannot predict task 6 for any cell that remaps. LOSO therefore measures "does
    this neuron encode V the SAME WAY across tasks"; within-session CV measures "does this
    neuron encode V at all". The GAP between them is remapping.

    Blocks are CONTIGUOUS runs of samples, not random draws: neighbouring bins are strongly
    autocorrelated (the animal occupies a node for many bins), so a random split would put
    near-duplicate samples in train and test and inflate held-out scores. Every fold draws
    from every session, so a fold is never a whole task.
    """
    session_ids = np.asarray(session_ids)
    out = np.empty(len(session_ids), dtype=int)
    for s in np.unique(session_ids):
        idx = np.flatnonzero(session_ids == s)
        n = len(idx)
        L = block_len if block_len else max(1, n // n_folds)
        blk = np.minimum(np.arange(n) // L, n_folds - 1)
        out[idx] = blk
    return out


#: The three nulls, and the hypothesis each actually tests. They are NOT interchangeable.
NULL_METHODS = ('shuffle', 'freedman_lane', 'column')

_NULL_H0 = {
    'shuffle':       'nothing explains this neuron (GLOBAL H0)',
    'freedman_lane': 'g adds nothing beyond the other regressors (SPECIFIC H0)',
    'column':        'g adds nothing beyond the other regressors (SPECIFIC H0, via the design)',
}


def freedman_lane_y(y, X_red, session_ids, shifts):
    """`y* = yhat_reduced + circshift_within_session(residuals)` -- the Freedman-Lane null.

    Tests "does g add anything BEYOND the other regressors", which is what a CPD is defined
    to measure. The reduced model's fit is preserved in `y*`, so under the null the full
    model still fits real structure and pays the SAME parameter penalty as the observed fit.

    That penalty is the whole problem with shuffling the firing instead. Shuffling destroys
    every regressor's signal, so the full model fits ~160 parameters to pure noise and
    generalises worse than the reduced model -- driving CPD_null negative, by more for
    bigger blocks. Measured on the LEC fit: corr(n_cols, null_mean) = -0.841, all 16 nulls
    centred below zero, and `head_direction` (35 columns) called significant in 62% of
    neurons on an observed CPD of ~0.0000.

    Circular shifting rather than free permutation preserves the residual autocorrelation.
    """
    y = np.atleast_2d(y)
    B = _pinv(X_red) @ y.T                       # (p_red, n_neurons)
    yhat = (X_red @ B).T                         # (n_neurons, T)
    resid = y - yhat
    return yhat + roll_within_sessions(resid, session_ids, shifts)


def _fwl_full_rss(X_red_tr, X_red_te, G_tr, G_te, Y_tr, Y_te, P_red):
    """Held-out RSS of the FULL model [X_red, G], via Frisch-Waugh-Lovell.

    Used by the `column` null, where G is a permuted copy of g's design block and therefore
    changes every permutation. Recomputing `pinv` of the whole (T x ~160) design each time
    costs ~71 min/recday; residualising G against the FIXED reduced design and inverting only
    the (T x k) result costs ~8.7 min (measured 356 ms -> 43 ms, k <= 35).

    FWL: with X = [X_red, G], the joint OLS coefficients satisfy
        b_g = pinv(G - X_red P_red G) @ (y - X_red P_red y)
        b_r = P_red @ (y - G b_g)
    so the full-model prediction is reconstructable from the fixed reduced pseudo-inverse
    plus one small inversion. Gated against the naive computation in the tests.
    """
    Gt = G_tr - X_red_tr @ (P_red @ G_tr)        # residualised block, (T_tr, k)
    Yt = Y_tr - X_red_tr @ (P_red @ Y_tr)        # residualised targets
    Bg = _pinv(Gt) @ Yt                          # (k, n_neurons)
    Br = P_red @ (Y_tr - G_tr @ Bg)              # (p_red, n_neurons)
    R = Y_te - (X_red_te @ Br + G_te @ Bg)
    return np.einsum('ij,ij->j', R, R)


def cv_scores(X, FR, session_ids, groups, *, n_perm=0, seed=0, joint_specs=(),
              chunk_perms=10, verbose=False, center_within_sessions=False,
              zscore_within_sessions=False, fold_ids=None, ridge=0.0,
              nulls=('freedman_lane',)):
    """Leave-one-session-out CPD and R^2 for every neuron, plus an optional CV null.

    Parameters
    ----------
    X : (T, p) design matrix, samples in the same order as `FR` columns.
    FR : (n_neurons, T) firing rates.
    session_ids : (T,) fold labels from `session_ids_from_filters`.
    groups : {name: [column indices]} regressor groups to drop for the reduced models.
    n_perm : int, default 0
        Permutations for the cross-validated null. 0 skips it -- a held-out CPD already has
        no in-sample optimism, so the null is for calibration against autocorrelation, not
        for removing overfitting.
    joint_specs : sequence of (name, [group names]) tested as one block.

    Returns
    -------
    dict with `cpd_cv` {group: (n_neurons,)}, `r2_cv` (n_neurons,), `rss_full`, `rss_reduced`,
    `n_folds`, and -- when `n_perm > 0` -- `p_cv` {group: (n_neurons,)} and `null_mean`.
    """
    X = np.asarray(X, dtype=float)
    FR = np.asarray(FR, dtype=float)
    session_ids = np.asarray(session_ids)
    n_neurons, T = FR.shape
    if X.shape[0] != T:
        raise ValueError(f'X has {X.shape[0]} rows but FR has {T} samples')

    # `fold_ids` decouples the fold structure from the session labels: pass
    # `within_session_folds(...)` to cut inside sessions instead of holding one out.
    # Normalisation always stays keyed to SESSION, whatever the folds are.
    fold_vec = session_ids if fold_ids is None else np.asarray(fold_ids)
    folds = np.unique(fold_vec)
    n_folds = len(folds)
    if n_folds < 2:
        raise ValueError(f'need >= 2 folds to cross-validate, got {n_folds}')

    if zscore_within_sessions:
        # Scaling y is a no-op for a single fold -- R^2 and CPD are ratios of sums of
        # squares of the same y -- but NOT across folds: RSS is pooled, so without this a
        # high-variance session dominates the pooled CPD. Z-scoring weights sessions equally.
        FR = center_within_session(FR, session_ids)
        for s in np.unique(session_ids):
            m = session_ids == s
            sd = FR[:, m].std(axis=1, keepdims=True)
            FR[:, m] = FR[:, m] / np.where(sd > 0, sd, 1.0)
    elif center_within_sessions:
        FR = center_within_session(FR, session_ids)

    # Reduced designs: drop each group's columns, and each joint block's columns.
    models = {'__full__': X}
    for g, idx in groups.items():
        models[g] = np.delete(X, idx, axis=1)
    for name, regs in joint_specs:
        idx = sorted(set().union(*[set(groups[r]) for r in regs]))
        models[name] = np.delete(X, idx, axis=1)

    rss = {m: np.zeros(n_neurons) for m in models}
    tss = np.zeros(n_neurons)
    rng = np.random.default_rng(seed)
    nulls = tuple(nulls) if n_perm > 0 else ()
    bad = [k for k in nulls if k not in NULL_METHODS]
    if bad:
        raise ValueError(f'unknown null(s) {bad}; choose from {NULL_METHODS}')

    # rss under each null: {null: {model: (n_perm, n_neurons)}}
    rss_perm = {k: {m: np.zeros((n_perm, n_neurons)) for m in models} for k in nulls}

    # One shift set per permutation, drawn once so every fold and every model sees the SAME
    # permuted data -- otherwise the null would average over shift noise as well.
    perm_shifts = (rng.integers(0, T, size=(n_perm, len(np.unique(session_ids))))
                   if n_perm > 0 else None)

    # `shuffle` shifts the FIRING, so its permuted data is the same for every model and can
    # be built once. `freedman_lane` and `column` are per-model (they depend on which group
    # is dropped), and materialising those up front would cost ~50 GB at 20 models -- so they
    # are generated inside the loop instead. FR_perm alone is already ~2.6 GB.
    FR_perm = (np.stack([roll_within_sessions(FR, session_ids, perm_shifts[i])
                         for i in range(n_perm)])
               if 'shuffle' in nulls else None)

    # Freedman-Lane needs yhat_reduced and its residuals per group. Compute them ONCE on the
    # full data rather than inside the fold loop: y* is a resampling of the whole dataset, and
    # a per-call pinv would cost one decomposition per (permutation, model, fold). Memory is
    # 2 arrays of (n_neurons, T) per model -- ~1 GB at 20 models, against ~50 GB if every
    # permuted y* were materialised.
    fl_base = {}
    if 'freedman_lane' in nulls:
        for m, Xm in models.items():
            if m == '__full__':
                continue
            yhat = (Xm @ (_pinv(Xm, ridge=ridge) @ FR.T)).T
            fl_base[m] = (yhat, FR - yhat)

    # Reduced-model column indices, needed to rebuild [X_red, G] for the `column` null.
    group_cols = dict(groups)
    for name, regs in joint_specs:
        group_cols[name] = sorted(set().union(*[set(groups[r]) for r in regs]))

    # For a null, `reduced_rss[k][g]` and `full_rss[k][g]` are the two RSS values whose
    # difference forms the statistic. They are tracked SEPARATELY per null because the three
    # nulls perturb different things:
    #   shuffle       - y is shifted; both models refit on the same y*, shared across groups
    #   freedman_lane - y* depends on WHICH group is dropped, so both models refit per group
    #   column        - only the FULL design changes; the reduced model is untouched, so its
    #                   observed RSS is the correct denominator
    red_perm = {k: {m: np.zeros((n_perm, n_neurons)) for m in models if m != '__full__'}
                for k in nulls}
    full_perm = {k: {m: np.zeros((n_perm, n_neurons)) for m in models if m != '__full__'}
                 for k in nulls}

    for f in folds:
        te = fold_vec == f
        tr = ~te
        if te.sum() < 2 or tr.sum() <= X.shape[1]:
            continue
        Y_te = FR[:, te].T
        Y_tr = FR[:, tr].T
        tss += np.sum((Y_te - Y_te.mean(axis=0)) ** 2, axis=0)

        # One pseudo-inverse per (fold, model), reused for every neuron and permutation.
        P = {m: _pinv(Xm[tr], ridge=ridge) for m, Xm in models.items()}
        for m, Xm in models.items():
            R = Y_te - Xm[te] @ (P[m] @ Y_tr)
            rss[m] += np.einsum('ij,ij->j', R, R)

        def _rss(Xm, Pm, Ytr, Yte):
            """Held-out RSS for a fixed design and (chunked) targets."""
            return np.einsum('ij,ij->j', Yte - Xm[te] @ (Pm @ Ytr),
                             Yte - Xm[te] @ (Pm @ Ytr))

        def _chunk_rss(Xm, Pm, blk, c):
            Ytr = blk[:, :, tr].transpose(2, 0, 1).reshape(tr.sum(), c * n_neurons)
            Yte = blk[:, :, te].transpose(2, 0, 1).reshape(te.sum(), c * n_neurons)
            Rp = Yte - Xm[te] @ (Pm @ Ytr)
            return np.einsum('ij,ij->j', Rp, Rp).reshape(c, n_neurons)

        for kind in nulls:
            if kind == 'shuffle':
                # y* is the same for every group, so score each model once per chunk.
                for lo in range(0, n_perm, chunk_perms):
                    hi = min(lo + chunk_perms, n_perm); c = hi - lo
                    blk = FR_perm[lo:hi]
                    fr = _chunk_rss(X, P['__full__'], blk, c)
                    for m in red_perm[kind]:
                        red_perm[kind][m][lo:hi] += _chunk_rss(models[m], P[m], blk, c)
                        full_perm[kind][m][lo:hi] += fr

            elif kind == 'freedman_lane':
                # y* = yhat_reduced + shifted residuals, so it differs PER GROUP. Both the
                # full and the reduced model must be refit on that same y*.
                for m in red_perm[kind]:
                    for lo in range(0, n_perm, chunk_perms):
                        hi = min(lo + chunk_perms, n_perm); c = hi - lo
                        yhat, resid = fl_base[m]
                        blk = np.stack([
                            yhat + roll_within_sessions(resid, session_ids, perm_shifts[i])
                            for i in range(lo, hi)])
                        red_perm[kind][m][lo:hi] += _chunk_rss(models[m], P[m], blk, c)
                        full_perm[kind][m][lo:hi] += _chunk_rss(X, P['__full__'], blk, c)

            elif kind == 'column':
                # Only g's block moves. The reduced model is untouched, so its OBSERVED
                # held-out RSS is the right denominator -- recorded once per fold below.
                for m in red_perm[kind]:
                    G = X[:, group_cols[m]]
                    Rm = Y_te - models[m][te] @ (P[m] @ Y_tr)
                    red_perm[kind][m] += np.einsum('ij,ij->j', Rm, Rm)[None, :]
                    for i in range(n_perm):
                        Gp = roll_within_sessions(G.T, session_ids, perm_shifts[i]).T
                        full_perm[kind][m][i] += _fwl_full_rss(
                            models[m][tr], models[m][te], Gp[tr], Gp[te],
                            Y_tr, Y_te, P[m])

        if verbose:
            print(f'    fold {f}: done')

    with np.errstate(divide='ignore', invalid='ignore'):
        num = {m: rss[m] - rss['__full__'] for m in models if m != '__full__'}
        denom = np.where(tss > 0, tss, np.nan)
        cpd_cv = {m: num[m] / rss[m] for m in num}
        # delta_r2_cv is the SAME numerator over a COMMON denominator. CPD divides by each
        # group's own reduced-model RSS, so a group dropped from a worse-fitting reduced
        # model gets a smaller denominator and an inflated score -- groups are not on one
        # scale. Dividing by held-out TSS puts every group in the same units, making them
        # comparable and roughly additive toward r2_cv. Measured: CPD overstates a weak
        # regressor by up to 12x when a dominant one stays in the model.
        delta_r2_cv = {m: num[m] / denom for m in num}
        r2_cv = 1.0 - rss['__full__'] / denom

    out = {'cpd_cv': cpd_cv, 'delta_r2_cv': delta_r2_cv, 'r2_cv': r2_cv,
           'rss_full': rss['__full__'], 'tss': tss,
           'rss_reduced': {m: rss[m] for m in models if m != '__full__'},
           'n_folds': int(n_folds), 'n_perm': int(n_perm), 'nulls': list(nulls),
           'null_hypotheses': {k: _NULL_H0[k] for k in nulls},
           'fold_scheme': 'session' if fold_ids is None else 'within_session'}

    # p-values for BOTH statistics under EVERY null. Same permutations, two ratios, so the
    # second costs nothing -- and it removes the mismatch where the figures showed delta_r2
    # while the only stored p tested CPD.
    #
    # One-sided, upper tail, with the add-one estimator: p = (1 + #{null >= obs}) / (1 + n).
    # A strongly NEGATIVE observed value therefore gets p ~ 1 and is never "significant in
    # the other direction" -- deliberate, since the question is whether g explains variance,
    # but worth stating because a large negative CPD is itself informative (the regressor
    # actively hurt held-out prediction) and this test cannot report it.
    for kind in nulls:
        with np.errstate(divide='ignore', invalid='ignore'):
            n_null = {m: red_perm[kind][m] - full_perm[kind][m] for m in red_perm[kind]}
            cpd_null = {m: n_null[m] / red_perm[kind][m] for m in n_null}
            dr2_null = {m: n_null[m] / denom[None, :] for m in n_null}
        for stat, obs, nul in (('cpd', cpd_cv, cpd_null),
                               ('delta_r2', delta_r2_cv, dr2_null)):
            out[f'p_{kind}__{stat}'] = {
                m: (1 + np.sum(nul[m] >= obs[m][None, :], axis=0)) / (1 + n_perm)
                for m in nul}
            out[f'null_mean_{kind}__{stat}'] = {m: np.nanmean(nul[m], axis=0) for m in nul}
            out[f'null_p95_{kind}__{stat}'] = {m: np.nanpercentile(nul[m], 95, axis=0)
                                               for m in nul}

    # Back-compat: `p_cv` / `null_mean` / `null_p95` keep pointing at the CPD statistic under
    # the FIRST requested null, so existing readers keep working. New code should name the
    # null and statistic explicitly.
    if nulls:
        k0 = nulls[0]
        out['p_cv'] = out[f'p_{k0}__cpd']
        out['null_mean'] = out[f'null_mean_{k0}__cpd']
        out['null_p95'] = out[f'null_p95_{k0}__cpd']
    return out


# ---------------------------------------------------------------------------
# Poisson cross-validated CPD
# ---------------------------------------------------------------------------

def cv_scores_poisson(X, FR, session_ids, groups, *, alpha=1.0, max_iter=100,
                      joint_specs=(), neuron_subset=None, verbose=False, fold_ids=None):
    """Cross-validated Poisson CPD -- the robustness check on the linear result.

    Deviance replaces RSS, so `cpd_cv` is the fraction of held-out *deviance* the group
    explains. `Neuron_raw` is integer spike counts per 25 ms bin (uint16, 89% zeros, mean
    0.131, max 8), so the rounding below is a no-op guard rather than a transformation, and
    Poisson is the correctly specified likelihood here while OLS is not.

    `fold_ids` mirrors `cv_scores`: pass `within_session_folds(...)` to cut inside sessions
    rather than holding a whole session (task) out. Normalisation stays keyed to session.

    There is no shared-pseudo-inverse trick here: a Poisson GLM is fit by IRLS per neuron per
    model per fold, so this costs `n_neurons x (1 + n_groups) x n_folds` iterative fits and
    is orders of magnitude slower than the linear path. Permutations are therefore not
    offered -- report the linear permutation test, and use this only to confirm the linear
    conclusions survive the link function.
    """
    from sklearn.linear_model import PoissonRegressor

    X = np.asarray(X, dtype=float)
    FR = np.asarray(FR, dtype=float)
    n_neurons, T = FR.shape
    idx = np.arange(n_neurons) if neuron_subset is None else np.asarray(neuron_subset)

    models = {'__full__': X}
    for g, gi in groups.items():
        models[g] = np.delete(X, gi, axis=1)
    for name, regs in joint_specs:
        gi = sorted(set().union(*[set(groups[r]) for r in regs]))
        models[name] = np.delete(X, gi, axis=1)

    fold_vec = session_ids if fold_ids is None else np.asarray(fold_ids)
    folds = np.unique(fold_vec)
    dev = {m: np.zeros(len(idx)) for m in models}
    dev_null = np.zeros(len(idx))

    def _deviance(y, mu):
        mu = np.clip(mu, 1e-10, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2.0 * np.sum(t - (y - mu))

    for f in folds:
        te = fold_vec == f
        tr = ~te
        if te.sum() < 2 or tr.sum() <= X.shape[1]:
            continue
        for k, ni in enumerate(idx):
            y_tr = np.clip(np.rint(FR[ni, tr]), 0, None)
            y_te = np.clip(np.rint(FR[ni, te]), 0, None)
            dev_null[k] += _deviance(y_te, np.full(te.sum(), max(y_tr.mean(), 1e-10)))
            for m, Xm in models.items():
                mdl = PoissonRegressor(alpha=alpha, max_iter=max_iter)
                try:
                    mdl.fit(Xm[tr], y_tr)
                    mu = mdl.predict(Xm[te])
                except Exception:
                    mu = np.full(te.sum(), max(y_tr.mean(), 1e-10))
                dev[m][k] += _deviance(y_te, mu)
        if verbose:
            print(f'    poisson fold {f}: done')

    with np.errstate(divide='ignore', invalid='ignore'):
        cpd_cv = {m: (dev[m] - dev['__full__']) / dev[m]
                  for m in models if m != '__full__'}
        d2_cv = 1.0 - dev['__full__'] / np.where(dev_null > 0, dev_null, np.nan)
    return {'cpd_cv': cpd_cv, 'd2_cv': d2_cv, 'deviance_full': dev['__full__'],
            'neuron_index': idx, 'n_folds': int(len(folds)),
            'fold_scheme': 'session' if fold_ids is None else 'within_session'}
