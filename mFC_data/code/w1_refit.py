"""PFC section definitions for the production GLM — the mirror of `code/w1_refit.py`.

Differs from the LEC copy in exactly two ways, both forced by the data rather than chosen:

  * **PFC cannot supply head_direction or the pokes.** `build_data_dic_from_pfc` sets
    `HD_raw=None` (no head tracking) and there are no poke event tables for this dataset. So
    the only design PFC can fit is the MATCHED-13 set, and that is also the only design that
    licenses a LEC-vs-PFC comparison: CPD is measured relative to the full model, so a design
    differing by 38 columns makes the two datasets' CPDs non-comparable.

  * **The data is a directory, not a monolithic pickle.** `build_data_dic_from_pfc` loads
    per-recday `.npy` files, so a per-recday job loads only its own recday (cheap) rather
    than the 3.8 GB `data_dic_lec.pkl` the LEC jobs each pay for.

Everything else -- binned aggregation, mixed reference coding, leave-one-session-out CV,
delta_r2 alongside CPD, config-encoded section names -- is identical, and deliberately so:
the two arms must be the same analysis for the comparison to mean anything.
"""

from __future__ import annotations

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MFC = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(MFC, '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DATA_FOLDER = os.path.join(MFC, 'data')
META = os.path.join(DATA_FOLDER, 'MetaData')
SAVE_DIR = os.path.join(MFC, 'glm_outputs', 'PFC')
BACKUP_SUFFIX = '.PRE_HD_FIX'

#: Drop legs longer than this. Median leg is 9.4 s but the distribution is heavy-tailed --
#: p99 is 115 s and the maximum is 771 s, which is a disengaged animal rather than a leg.
#:
#: The cost is asymmetric and worth stating, because the legs a cutoff removes are the
#: longest ones and therefore carry the most time bins:
#:
#:     cutoff   legs dropped   SAMPLES dropped
#:       20 s        16.2%           50.1%
#:       30 s         8.4%           37.9%
#:       40 s         5.2%           30.7%
#:       60 s         2.6%           22.4%   <- chosen
#:
#: At 60 s that leaves ~22,700 rows against 160 columns (142:1). For contrast,
#: `filter_correct_paths=True` was leaving ~1,000 rows (6:1) and producing held-out R2 of
#: -0.09 to -2.13 -- which is what `min_samples_per_param` now refuses to fit.
#:
#: It is also a behavioural selection, not a neutral one: it removes the slowest legs. If a
#: population is more active when the animal is disengaged, this is not innocent -- say so
#: wherever it matters.
MAX_LEG_SECONDS = 60.0

_TIME_ANY = ('time_any', ['time_from_reward', 'time_to_reward'])
_DIST_ANY = ('distance_any', ['distance_from_reward', 'distance_to_reward'])
_GP_ANY = ('gp_any', ['goal_progress'])

#: Regressors PFC cannot supply. Kept as a module constant so the LEC side can import the
#: same list and build a design that matches exactly.
PFC_UNAVAILABLE = ('head_direction', 'poke_rewarded', 'poke_unrewarded')

#: The matched-13 design: everything in the LEC `all_regressors` set that PFC can also
#: supply. This is the ONLY design used for cross-dataset claims.
MATCHED_REGRESSORS = [
    'place', 'task_state', 'goal_progress', 'goal_progress_distance',
    'speed', 'acceleration', 'time_from_reward', 'time_to_reward',
    'time_since_A', 'time_to_A', 'progress_since_A',
    'distance_from_reward', 'distance_to_reward',
]

SECTIONS = {
    'all_regressors': dict(
        regressors=list(MATCHED_REGRESSORS),
        joint_drop_groups=[_TIME_ANY, _DIST_ANY, _GP_ANY],
        filter_correct_paths=False,   # keeps only 4-16% of transitions; see MAX_LEG_SECONDS
        max_transition_seconds=MAX_LEG_SECONDS,
    ),
    'distance_gp_state_filtered': dict(
        regressors=['place', 'task_state', 'goal_progress', 'goal_progress_distance',
                    'speed', 'acceleration', 'time_from_reward', 'time_to_reward',
                    'distance_from_reward', 'distance_to_reward'],
        joint_drop_groups=[_TIME_ANY, _DIST_ANY, _GP_ANY],
        filter_correct_paths=False,   # keeps only 4-16% of transitions; see MAX_LEG_SECONDS
        max_transition_seconds=MAX_LEG_SECONDS,
    ),
    'since_A_filtered': dict(
        regressors=['place', 'goal_progress', 'speed', 'acceleration',
                    'time_from_reward', 'time_to_reward',
                    'time_since_A', 'time_to_A', 'progress_since_A',
                    'distance_from_reward', 'distance_to_reward'],
        joint_drop_groups=[_TIME_ANY, _DIST_ANY, _GP_ANY],
        filter_correct_paths=False,   # keeps only 4-16% of transitions; see MAX_LEG_SECONDS
        max_transition_seconds=MAX_LEG_SECONDS,
    ),
    # ---- V3 reduced designs (engine `glm_analysis_v3`; see GLM_V3.md) -----------------------
    # Identical to the LEC copy: the reduced designs need nothing PFC lacks, so this is the
    # first design fitted identically in both datasets without a matched/full asymmetry. The
    # leg cap is a FACTOR, set per run from `--leg-cap-s`.
    'core_progress_time': dict(
        engine='v3',
        regressors=['place', 'goal_progress', 'speed', 'acceleration', 'time_from_reward'],
        joint_drop_groups=[('progress_or_time', ['goal_progress', 'time_from_reward'])],
        filter_correct_paths=False,
        max_transition_seconds=None,   # <- --leg-cap-s
        tfr_arm=True,                  # --tfr-scheme / --tfr-bins build `binning_overrides`
    ),
    'core_progress_only': dict(
        # The sanity arm: goal_progress is the ONLY within-leg regressor, so its CPD here is
        # "within-leg structure of any kind", not phase-locking (GLM_V3.md caveat 9).
        engine='v3',
        regressors=['place', 'goal_progress', 'speed', 'acceleration'],
        joint_drop_groups=[],
        filter_correct_paths=False,
        max_transition_seconds=None,   # <- --leg-cap-s
        cap_arm=True,
    ),
}

ARTIFACTS = ('glm_results', 'permutation_results', 'cpd_results', 'cv_results')


def matched_regressors(regressors):
    """Subset both datasets can supply. Identity on PFC sections; kept for API parity."""
    return [r for r in regressors if r not in PFC_UNAVAILABLE]


def section_name(base, *, width_ms, scheme, regset, extra=None):
    """Cache key encoding the configuration -- identical scheme to the LEC copy.

    `run_or_load_glm` keys purely on the section name, so two configurations sharing a name
    silently overwrite each other's pickles. The V3 arms append the `extra` token from
    `arm_extra` (leg cap and time_from_reward coding).
    """
    name = f'{base}__{regset}_{int(width_ms)}ms_{scheme}'
    return f'{name}_{extra}' if extra else name


#: How a neuron's firing is normalised before the cross-validated fit. THE ONLY THING THAT
#: DISTINGUISHES THESE IS WHETHER EACH SESSION GETS ITS OWN OFFSET AND/OR ITS OWN GAIN.
#:
#: A per-neuron z-score computed ACROSS the recday's sessions is not on this list because it is
#: not a model choice at all: one affine map per neuron is absorbed by the intercept, so every
#: RSS and TSS scales by 1/sd^2 and every ratio -- CPD, delta_r2, r2_cv -- is unchanged.
#: Measured on `ah10_20250616_20250617` (2026-09-08): max|diff| 8e-15 on cpd_cv, 9e-15 on
#: delta_r2_cv, EXACTLY 0 on the Freedman-Lane p-values, against 1.6e-2 for `session-z`. Only
#: the unnormalised `tss`/`rss_full` move (by the sd^2 factor) and the betas rescale by 1/sd
#: with the intercept absorbing the mean. So `none` below IS the across-session-z-scored model.
#:
#:   'session-centre' : per-session offset, no per-session gain. THE PRODUCTION SETTING.
#:   'session-z'      : per-session offset AND gain. Weights sessions equally in the pooled
#:                      leave-one-session-out RSS, so a high-variance session cannot dominate;
#:                      but the CV then tests tuning SHAPE UP TO A PER-SESSION SCALE, which is a
#:                      different question, not a neutral normalisation.
#:   'none'           : neither. Identical to an across-session z-score (see above).
#:
#: Only `cv_results` is affected: the in-sample loop fits raw `FR_all`, so `glm_results`,
#: `cpd_results` and `permutation_results` -- and hence every beta profile and heatmap -- are
#: identical across all three.
NORMALISATIONS = ('session-centre', 'session-z', 'none')
_NORM_TOKEN = {'session-centre': '', 'session-z': '_zscored', 'none': '_nonorm'}


def norm_kwargs(normalise='session-centre'):
    """`run_glm_analysis` cross-validation flags for one rung of `NORMALISATIONS`."""
    if normalise not in NORMALISATIONS:
        raise ValueError(f'normalise must be one of {NORMALISATIONS}, got {normalise!r}')
    return {'cv_center_within_sessions': normalise in ('session-centre', 'session-z'),
            'cv_zscore_within_sessions': normalise == 'session-z'}


def arm_extra(cfg, *, leg_cap_s=None, tfr_scheme=None, tfr_bins=None,
              normalise='session-centre'):
    """Cache-name token for a V3 arm, or None for a v2 section.

    `core_progress_time` -> `cap30s_tfrU10b` (uniform, 10 bins over [0, 30] s) or
    `cap30s_tfrD10b` (decile); `core_progress_only` -> `cap30s`. A non-default `normalise`
    appends `_zscored` / `_nonorm`, so the production names are unchanged and cannot be
    overwritten. NOTE: the token must stay `[A-Za-z0-9_.]` -- a hyphen fails
    `recday_registry._POST_REFIT_SECTION`, which silently re-arms `STALE_CACHE_RECDAYS` and
    drops `ly05_20250618_20250619` from every load (verified 2026-09-08).
    """
    if normalise not in NORMALISATIONS:
        raise ValueError(f'normalise must be one of {NORMALISATIONS}, got {normalise!r}')
    tok = _NORM_TOKEN[normalise]
    if cfg.get('tfr_arm'):
        code = {'uniform': 'U', 'decile': 'D'}[tfr_scheme]
        return f'cap{float(leg_cap_s):g}s_tfr{code}{int(tfr_bins)}b{tok}'
    if cfg.get('cap_arm'):
        return f'cap{float(leg_cap_s):g}s{tok}'
    return tok.lstrip('_') or None


def binning_overrides_for(cfg, *, leg_cap_s=None, tfr_scheme=None, tfr_bins=None):
    """`binning_overrides` for `glm_analysis_v3.run_glm_analysis`, or None.

    Uniform: 10 (or `tfr_bins`) equal-width bins over [0, leg cap] SECONDS -- the same edges
    in every recday and both datasets, no overflow bin (the range is the cap). Decile: the v2
    quantile path with a custom bin count. Only `core_progress_time` carries an override.
    """
    if not cfg.get('tfr_arm'):
        return None
    if tfr_scheme not in ('uniform', 'decile'):
        raise ValueError(f"tfr_scheme must be 'uniform' or 'decile', got {tfr_scheme!r}")
    return {'time_from_reward': {
        'scheme': tfr_scheme,
        'range_s': (0.0, float(leg_cap_s)) if tfr_scheme == 'uniform' else None,
        'n_bins': int(tfr_bins),
    }}


def leg_cap_for(cfg, leg_cap_s=None):
    """The leg cap a run actually uses: the CLI value for the V3 arms, the table for v2."""
    if cfg.get('tfr_arm') or cfg.get('cap_arm'):
        if leg_cap_s is None:
            raise ValueError('this section takes its leg cap from --leg-cap-s')
        return float(leg_cap_s)
    return cfg.get('max_transition_seconds')


def choose_parameterization(regressors, engine='v2', n_cols_override=None):
    """`reference_coded` (full rank) where possible, `all_bins` where it is not.

    Reference coding is MIXED: it drops a reference bin from every multi-column block and
    passes single-column indicators through untouched. PFC has no single-column regressors
    (no pokes), so this always returns `reference_coded` here -- but the logic is mirrored so
    the two arms cannot diverge if PFC ever gains one.
    """
    import importlib
    glm = importlib.import_module(f'glm_analysis_{engine}')
    kw = {'n_cols_override': n_cols_override} if n_cols_override else {}
    try:
        glm._resolve_regressor_groups(regressors, gp_n_bins=10,
                                      parameterization='reference_coded', **kw)
        return 'reference_coded'
    except ValueError:
        return 'all_bins'


def pfc_recdays():
    """The canonical 25 double-day ABCD recdays."""
    import numpy as np
    p = os.path.join(META, 'combined_ABCDonly_days.npy')
    return [str(r) for r in np.load(p)]


def backup_existing(section, dry_run=False):
    """Move a section's existing pickles aside. Never overwrite a backup that exists."""
    moved = []
    for art in ARTIFACTS:
        p = os.path.join(SAVE_DIR, f'{section}__{art}.pkl')
        if not os.path.exists(p):
            continue
        dst = p + BACKUP_SUFFIX
        if os.path.exists(dst):
            print(f'    backup already exists, leaving it: {os.path.basename(dst)}')
            continue
        print(f'    {os.path.basename(p)} -> {os.path.basename(dst)}')
        if not dry_run:
            shutil.move(p, dst)
        moved.append(dst)
    return moved
