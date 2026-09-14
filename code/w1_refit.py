"""W1: refit every GLM section on all 25 recdays, cross-validated, with HD fixed.

One clean set of fits that supersedes every cached pickle. The caches this replaces are
stale in three independent ways:

  * the head-direction regressor was interleaved garbage (see ANATOMY_SPLIT.md W0.2), so all
    36 HD columns in every previous fit are noise;
  * `ly05_20250618_20250619` was fitted against the wrong day's spikes (91 units vs 109), and
    `ly05_20250620_20250623` was never fitted at all -- the caches hold 24 recdays, not 25;
  * CPD and significance were computed in-sample, with no held-out fold anywhere.

Old pickles are moved aside to `*.PRE_HD_FIX` rather than deleted, matching the repo's
existing convention (`.INVALID_state_bug`, `.PRE_ly05_recday_fix`).

Usage
-----
    python code/w1_refit.py --sections distance_gp_state_filtered      # one section
    python code/w1_refit.py --all                                      # everything
    python code/w1_refit.py --all --dry-run                            # plan only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SAVE_DIR = os.path.join(REPO, 'data', 'glm_outputs', 'LEC')
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

#: Section definitions. `regressors` is passed as `regressors_to_include`; None keeps the
#: default 9. Order matters -- the first is the smoke test and the backbone of W2/W3.
SECTIONS = {
    'distance_gp_state_filtered': dict(
        regressors=['place', 'task_state', 'head_direction', 'goal_progress',
                    'goal_progress_distance', 'speed', 'acceleration',
                    'time_from_reward', 'time_to_reward',
                    'distance_from_reward', 'distance_to_reward'],
        joint_drop_groups=[_TIME_ANY, _DIST_ANY, _GP_ANY],
        filter_correct_paths=False,   # keeps only 4-16% of transitions; see MAX_LEG_SECONDS
        max_transition_seconds=MAX_LEG_SECONDS,
    ),
    'all_regressors': dict(
        # Every variable in one design, so CPDs are mutually comparable. Heavily collinear
        # by construction (GP ~ time within a leg; poke_rewarded ~ early time_from_reward);
        # report alongside the targeted sections, never instead of them.
        regressors=['place', 'task_state', 'poke_rewarded', 'poke_unrewarded',
                    'head_direction', 'goal_progress', 'goal_progress_distance',
                    'speed', 'acceleration',
                    'time_from_reward', 'time_to_reward',
                    'time_since_A', 'time_to_A', 'progress_since_A',
                    'distance_from_reward', 'distance_to_reward'],
        joint_drop_groups=[_TIME_ANY, _DIST_ANY, _GP_ANY],
        filter_correct_paths=False,   # keeps only 4-16% of transitions; see MAX_LEG_SECONDS
        max_transition_seconds=MAX_LEG_SECONDS,
    ),
    'distance_gp_filtered': dict(
        regressors=None,
        joint_drop_groups=[_TIME_ANY, _DIST_ANY, _GP_ANY],
        filter_correct_paths=False,   # keeps only 4-16% of transitions; see MAX_LEG_SECONDS
        max_transition_seconds=MAX_LEG_SECONDS,
    ),
    'pokes_filtered': dict(
        regressors=['place', 'poke_rewarded', 'poke_unrewarded', 'head_direction',
                    'goal_progress', 'speed', 'acceleration',
                    'time_from_reward', 'time_to_reward',
                    'distance_from_reward', 'distance_to_reward'],
        joint_drop_groups=[_TIME_ANY, _DIST_ANY, _GP_ANY],
        filter_correct_paths=False,   # keeps only 4-16% of transitions; see MAX_LEG_SECONDS
        max_transition_seconds=MAX_LEG_SECONDS,
    ),
    'since_A_filtered': dict(
        regressors=['place', 'head_direction', 'goal_progress', 'speed', 'acceleration',
                    'time_from_reward', 'time_to_reward',
                    'time_since_A', 'time_to_A', 'progress_since_A',
                    'distance_from_reward', 'distance_to_reward'],
        joint_drop_groups=[_TIME_ANY, _DIST_ANY, _GP_ANY],
        filter_correct_paths=False,   # keeps only 4-16% of transitions; see MAX_LEG_SECONDS
        max_transition_seconds=MAX_LEG_SECONDS,
    ),
    # ---- V3 reduced designs (engine `glm_analysis_v3`; see GLM_V3.md) -----------------------
    # The leg cap is a FACTOR here, set per run from `--leg-cap-s`, so `max_transition_seconds`
    # is None in the table and filled by the driver. `run_glm_batch.py` is the only entry point
    # for these sections; `main()` below refuses them.
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

#: Regressors PFC cannot supply: it has `HD_raw=None` and no poke tables. CPD is measured
#: relative to the full model, so a design differing by these blocks makes LEC and PFC CPDs
#: non-comparable -- only the matched design licenses a cross-dataset claim.
PFC_UNAVAILABLE = ('head_direction', 'poke_rewarded', 'poke_unrewarded')


def matched_regressors(regressors):
    """The subset of `regressors` that both datasets can supply (the matched-13 design)."""
    return [r for r in regressors if r not in PFC_UNAVAILABLE]


def section_name(base, *, width_ms, scheme, regset, extra=None):
    """Cache key encoding the configuration.

    `run_or_load_glm` keys purely on the section name, so two configurations sharing a name
    silently overwrite each other's pickles. Every axis that changes the fit has to appear
    here: regressor set, bin width, bin-placement scheme -- and, for the V3 arms, the
    `extra` token from `arm_extra` (leg cap and time_from_reward coding).
    `recday_registry.is_post_refit_section` must keep matching whatever this returns.
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

    `reference_coded` now uses MIXED coding: it drops a reference bin from every
    multi-column block and passes single-column indicators (the pokes) through untouched,
    giving a full-rank design that still contains the pokes. It used to raise on any
    single-column regressor, which forced every poke-containing fit onto the rank-deficient
    `all_bins` and made "the full regressor set" and "interpretable betas" look mutually
    exclusive. They are not -- the pokes do not sum to 1, so they never caused the
    deficiency and never needed a reference bin.

    This therefore returns `reference_coded` for every current section. The fallback is kept
    for designs that genuinely cannot be reference-coded, in which case betas are
    minimum-norm and `tuned_dict` SIGNS are uninterpretable while RSS/R^2/CPD are unaffected.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sections', nargs='*', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--permutations', type=int, default=100)
    ap.add_argument('--cv-perms', type=int, default=100,
                    help='permutations for the cross-validated null')
    ap.add_argument('--no-cv', action='store_true')
    ap.add_argument('--recdays', type=int, default=None, help='limit, for testing')
    ap.add_argument('--width-ms', type=int, default=250,
                    help='aggregation window in ms (downsample_factor = width/25)')
    ap.add_argument('--scheme', choices=('decile', 'uniform'), default='decile',
                    help='bin placement for the continuous regressors')
    ap.add_argument('--regset', choices=('full', 'matched'), default='full',
                    help="'matched' drops HD and pokes so the design equals PFC's")
    ap.add_argument('--no-pokes-attach', action='store_true',
                    help='skip attach_pokes (leaves poke regressors all-zero)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    sections = list(SECTIONS) if args.all else (args.sections or [])
    if not sections:
        ap.error('pass --sections NAME... or --all')
    unknown = [s for s in sections if s not in SECTIONS]
    if unknown:
        ap.error(f'unknown section(s): {unknown}')
    v3 = [s for s in sections if SECTIONS[s].get('engine', 'v2') != 'v2']
    if v3:
        ap.error(f'{v3} are V3 arms whose leg cap and time_from_reward coding are run-time '
                 f'options; fit them with run_glm_batch.py (see GLM_V3.md), not this path.')

    import glm_analysis_v2 as glm

    print('=' * 78)
    print('W1 REFIT — 25 recdays, HD fixed, cross-validated')
    print('=' * 78)

    data_dic = glm.load_data_dic(validate=True, apply_exclusions=True, verbose=True)

    # Without this the poke regressors are all-zero columns: their CPD comes back exactly
    # 0.00000 and they cost the design one rank each. That is how a "16-regressor" fit
    # silently becomes a 13-regressor one with two dead columns.
    needs_pokes = (args.regset == 'full'
                   and any('poke' in r for c in SECTIONS.values()
                           for r in (c['regressors'] or [])))
    if needs_pokes and not args.no_pokes_attach:
        n = glm.attach_pokes(data_dic, verbose=True)
        print(f'attach_pokes: {n} session tables attached')

    recdays = sorted(data_dic)
    if args.recdays:
        recdays = recdays[:args.recdays]
    print(f'\n{len(recdays)} recdays, {len(sections)} section(s): {sections}')
    print(f'permutations={args.permutations}  cv_perms={args.cv_perms}  '
          f'cross_validate={not args.no_cv}\n')

    timings = {}
    for i, section in enumerate(sections, 1):
        cfg = SECTIONS[section]
        print('-' * 78)
        print(f'[{i}/{len(sections)}] {section}')
        print('-' * 78)
        print(f'  section name: {sect}')
        print(f'  regressors: {len(regs) if regs else 9}  width {args.width_ms} ms '
              f'(factor {factor})  scheme {args.scheme}')
        print('  backing up existing pickles:')
        backup_existing(sect, dry_run=args.dry_run)
        if args.dry_run:
            print('  (dry run — not fitting)')
            continue

        regs = cfg['regressors']
        if args.regset == 'matched' and regs is not None:
            regs = matched_regressors(regs)
        sect = section_name(section, width_ms=args.width_ms, scheme=args.scheme,
                            regset=args.regset)
        factor = max(1, int(round(args.width_ms / 25)))
        param = choose_parameterization(regs)
        if param != 'reference_coded':
            print(f'  parameterization={param} (rank-deficient by 8) — this section has a '
                  f'single-column regressor, so reference coding is impossible. CPD and all '
                  f'CV quantities are unaffected; tuned_dict SIGNS are not interpretable.')
        else:
            print(f'  parameterization={param} (full rank)')

        t0 = time.time()
        glm.run_or_load_glm(
            recdays, data_dic, SAVE_DIR, sect,
            force_refit=True,
            num_permutations=args.permutations,
            regressors_to_include=regs,
            joint_drop_groups=[j for j in cfg['joint_drop_groups']
                               if all(r in (regs or []) for r in j[1])] or None,
            filter_correct_paths=cfg.get('filter_correct_paths', False),
            max_transition_seconds=cfg.get('max_transition_seconds'),
            compute_cpd=True,
            parameterization=param,
            downsample_factor=factor,
            downsample_mode='bin',
            continuous_binning=args.scheme,
            cross_validate=not args.no_cv,
            cv_n_perm=0 if args.no_cv else args.cv_perms,
        )
        el = time.time() - t0
        timings[section] = el
        print(f'\n  {section}: {el / 60:.1f} min '
              f'({el / max(1, len(recdays)):.1f} s/recday)')

    if timings:
        print('\n' + '=' * 78)
        print(f'DONE — total {sum(timings.values()) / 3600:.2f} h')
        for s, t in timings.items():
            print(f'  {s:32s} {t / 60:7.1f} min')
        print('\nNext: the refit gate — non-HD in-sample CPDs must come back near-identical')
        print('to the *.PRE_HD_FIX caches on the 23 valid recdays. Only HD changed.')


if __name__ == '__main__':
    main()
