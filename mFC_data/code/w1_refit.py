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
}

ARTIFACTS = ('glm_results', 'permutation_results', 'cpd_results', 'cv_results')


def matched_regressors(regressors):
    """Subset both datasets can supply. Identity on PFC sections; kept for API parity."""
    return [r for r in regressors if r not in PFC_UNAVAILABLE]


def section_name(base, *, width_ms, scheme, regset):
    """Cache key encoding the configuration -- identical scheme to the LEC copy.

    `run_or_load_glm` keys purely on the section name, so two configurations sharing a name
    silently overwrite each other's pickles.
    """
    return f'{base}__{regset}_{int(width_ms)}ms_{scheme}'


def choose_parameterization(regressors):
    """`reference_coded` (full rank) where possible, `all_bins` where it is not.

    Reference coding is MIXED: it drops a reference bin from every multi-column block and
    passes single-column indicators through untouched. PFC has no single-column regressors
    (no pokes), so this always returns `reference_coded` here -- but the logic is mirrored so
    the two arms cannot diverge if PFC ever gains one.
    """
    import glm_analysis_v2 as glm
    try:
        glm._resolve_regressor_groups(regressors, gp_n_bins=10,
                                      parameterization='reference_coded')
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
