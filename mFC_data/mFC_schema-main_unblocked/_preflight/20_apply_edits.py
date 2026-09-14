#!/usr/bin/env python3
"""Build `mFC_schema-main_unblocked/` from the deposit by applying the declared edits.

Design rules, all of which the audit (90_audit_diff.py) re-checks:

  * Every changed or added source line carries a trailing `# UNBLOCK-<ID>` token.
  * Every edit is preceded by exactly one `# UNBLOCK-<ID> (<CLASS>): ...` comment
    recording what the deposit had and why the change is unavoidable.
  * Each op carries a `guard` substring asserted against the deposited line, so if the
    deposit ever differs from what was analysed the script fails loudly instead of
    silently patching the wrong line.
  * Ops within a cell are applied bottom-up so earlier line indices stay valid.
  * G-01: outputs are cleared and execution_count nulled, so the audit only ever
    compares source. The deposit's stored outputs are archived first -- they are the
    numeric reference for verification.

Run:  python 20_apply_edits.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
DEPOSIT = os.path.join(ROOT, 'mFC_schema-main')
UNBLOCKED = os.path.join(ROOT, 'mFC_schema-main_unblocked')
LOGDIR = os.path.join(ROOT, 'replication_run', 'logs')

IN_MIRROR = os.path.join(ROOT, 'replication_run', 'data', 'Intermediate_objects') + '/'
OUT_EPHYS = os.path.join(ROOT, 'replication_run', 'Output_folder', 'ephys') + '/'
OUT_BEHAV = os.path.join(ROOT, 'replication_run', 'Output_folder', 'behaviour') + '/'

NOTEBOOKS = [
    'Basic_analysis.ipynb',
    'Behavioural Analysis (Figure 1).ipynb',
    'Figure2.ipynb',
    'Figure2_UMAP.ipynb',
    'Figure3.ipynb',
    'Figure5_Figure6.ipynb',
    'Figure5_Regression.ipynb',
    'Figure7.ipynb',
]
VERBATIM = ['LICENSE', 'README.md']


# --------------------------------------------------------------------------- ops
def rep(eid, cls, cell, line, guard, new, reason):
    """Replace one line. `new` is a list of replacement lines (markers added here)."""
    return dict(op='replace', id=eid, cls=cls, cell=cell, line=line, guard=guard,
                new=new, reason=reason)


def ins(eid, cls, cell, line, guard, new, reason):
    """Insert `new` lines AFTER `line` (guard asserted against `line`)."""
    return dict(op='insert_after', id=eid, cls=cls, cell=cell, line=line, guard=guard,
                new=new, reason=reason)


def ins_before(eid, cls, cell, line, guard, new, reason):
    return dict(op='insert_before', id=eid, cls=cls, cell=cell, line=line, guard=guard,
                new=new, reason=reason)


def celltype(eid, cls, cell, to, guard, reason):
    return dict(op='cell_type', id=eid, cls=cls, cell=cell, to=to, guard=guard,
                reason=reason)


def newcell(eid, cls, before, source, reason):
    return dict(op='new_cell', id=eid, cls=cls, before=before, source=source,
                reason=reason)


def paths(eid, cell, li, lo, out, nb):
    """The standard Input_folder/Output_folder repoint. Two ops, same ID."""
    return [
        rep(eid, 'PATH', cell, li, 'Input_folder',
            [f"Input_folder = '{IN_MIRROR}'   # UNBLOCK-{eid}"],
            f'repoint at the writable mirror so the deposit is never written to ({nb})'),
        rep(eid, 'PATH', cell, lo, 'Output_folder',
            [f"Output_folder = '{out}'   # UNBLOCK-{eid}"],
            'repoint + add the TRAILING SLASH: every save is string concatenation '
            "(Output_folder+'x.svg'), so without it figures land as a sibling file"),
    ]


# ------------------------------------------------------------------- the 360-bin cell
F2_06_SOURCE = [
    "### UNBLOCK-F2-06 (NEWCELL): write the 360-bin Neuron_/Location_ arrays.",
    "### Six notebooks read Neuron_<rd>_<s>.npy and Location_<rd>_<s>.npy -- Figure2 c31,",
    "### Figure3 c20/23/29/32/36/42/59/80, Figure5_Figure6 c17/22/47/50/89,",
    "### Figure5_Regression c15, Figure7 c52/59, Figure2_UMAP c9 -- and NO deposited",
    "### notebook writes them. Several of those reads sit outside any try and hard-crash;",
    "### others sit inside a try and silently yield empty dicts, which is worse.",
    "### Recipe follows Basic_analysis.ipynb cell 21. It deliberately calls THIS notebook's",
    "### own raw_to_norm (cell 6 defines it three times, last one wins) so the",
    "### normalisation is identical to what cell 46 uses, by construction.",
    "for day_type in ['3_task_all','combined_ABCDonly','combined_ABCDE']:                    # UNBLOCK-F2-06",
    "    recording_days_=np.load(Input_folder+day_type+'_days.npy')                           # UNBLOCK-F2-06",
    "    for mouse_recday in recording_days_:                                                 # UNBLOCK-F2-06",
    "        print(mouse_recday)                                                              # UNBLOCK-F2-06",
    "        Tasks=np.load(Input_folder+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)   # UNBLOCK-F2-06",
    "        for ses_ind in np.arange(len(Tasks)):                                            # UNBLOCK-F2-06",
    "            try:                                                                         # UNBLOCK-F2-06",
    "                Neuron_raw=np.load(Input_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')      # UNBLOCK-F2-06",
    "                Location_raw=np.load(Input_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')  # UNBLOCK-F2-06",
    "                Trial_times=np.load(Input_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')     # UNBLOCK-F2-06",
    "            except Exception as e:                                                       # UNBLOCK-F2-06",
    "                print('Files not found for session '+str(ses_ind))                        # UNBLOCK-F2-06",
    "                continue                                                                 # UNBLOCK-F2-06",
    "            Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25      # UNBLOCK-F2-06",
    "            Neurons_norm=np.asarray([raw_to_norm(Neuron_raw[neuron],Trial_times_conc,\\",
    "                                                 return_mean=False)\\",
    "                                     for neuron in np.arange(len(Neuron_raw))])          # UNBLOCK-F2-06",
    "            Location_norm=raw_to_norm(Location_raw,Trial_times_conc,return_mean=False)   # UNBLOCK-F2-06",
    "            np.save(Input_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy',Neurons_norm)            # UNBLOCK-F2-06",
    "            np.save(Input_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy',Location_norm)         # UNBLOCK-F2-06",
]

# ------------------------------------------------------------------------- the edits
EDITS: dict[str, list] = {

    'Basic_analysis.ipynb': [
        rep('BA-01', 'PATH', 1, 0, 'Data_folder',
            [f"Data_folder='{IN_MIRROR}'   # UNBLOCK-BA-01"],
            'repoint at the writable mirror. Documentation-only notebook; otherwise '
            'runs as deposited (its partition() already omits the outer np.asarray)'),
    ],

    'Behavioural Analysis (Figure 1).ipynb': [
        *paths('BH-01', 3, 2, 3, OUT_BEHAV, 'already runs end to end; repoint only'),
    ],

    'Figure2.ipynb': [
        *paths('F2-01', 1, 2, 3, OUT_EPHYS, 'Figure2'),

        rep('F2-02', 'COVERAGE', 16, 4, "for day_type in ['combined_ABCDonly']:",
            ["for day_type in ['combined_ABCDonly','3_task_all']:   # UNBLOCK-F2-02"],
            "cell 16 builds speed_dic over combined recdays ONLY, but cell 18 iterates "
            "['combined_ABCDonly','3_task_all']. Without '3_task_all' here, speed_dic "
            "misses all 55 single-day recdays, so cell 18's distances=speed_dic[rd][ses] "
            "autovivifies an empty defaultdict, distances[start:end] raises "
            "TypeError: unhashable type: 'slice', cell 18's except swallows it, and ALL "
            "SIX phase/state/time dicts stay silently empty for 3_task_all -- cascading "
            "into cells 23/25/31/40/46/48/50/53. His stored cell-18 output iterates 80 "
            "recdays (25 combined_ABCDonly + 55 3_task_all) with only 4 'not made', "
            "proving his speed_dic covered the single days. With this edit our run "
            "reports 'speed_dic covers 80 recdays, 412 sessions' and cell 18 populates "
            "80 recdays, matching him"),

        # ---- F2-03: the one scientific edit. Applied bottom-up.
        rep('F2-03', 'SCIENCE', 18, 85,
            'Phases_raw_dic2[mouse_recday][ses_ind]=phases_all',
            ["                Phases_raw_dic2[mouse_recday][ses_ind]=phases_all2   # UNBLOCK-F2-03"],
            'store the genuine 3-bin array. The deposit assigned the SAME list object '
            '(phases_all) to both dicts, making Phases_raw2_ and Phases_raw_ '
            'byte-identical 5-bin arrays and leaving num_phases2=3 dead'),
        ins('F2-03', 'SCIENCE', 18, 80, 'phases_all.append(phases_trial)',
            ["                    phases_all2.append(phases_trial2)   # UNBLOCK-F2-03"],
            'accumulate the 3-bin array per trial'),
        ins('F2-03', 'SCIENCE', 18, 74, 'phases_trial.append(phase_trial_state)',
            ["                        phases_trial2.append(phase_trial_state2)   # UNBLOCK-F2-03"],
            'accumulate the 3-bin array per state'),
        # residual block: evaluate ONCE, apply to BOTH, decrement ONCE.
        ins('F2-03', 'SCIENCE', 18, 63,
            'phase_trial_state=np.hstack((phase_trial_state,int(num_phases-1)))',
            ["                            phase_trial_state2=np.hstack((phase_trial_state2,\\",
             "                                                          int(num_phases2-1)))   # UNBLOCK-F2-03"],
            'apply the SAME residual-rounding extra bin to the 3-bin array. This must sit '
            'inside the existing else and residual_cum must be decremented only once, or '
            'the two arrays drift in length and stop aligning bin-for-bin with Location_raw'),
        ins('F2-03', 'SCIENCE', 18, 57, '(num_bins//num_phases)*num_phases)))',
            ["",
             "                        phase_trial_state2=np.hstack((np.repeat(np.arange(num_phases2),\\",
             "                                                                num_bins//num_phases2),\\",
             "                                                     np.repeat(int(num_phases2-1),num_bins-\\",
             "                                                               (num_bins//num_phases2)*num_phases2)))   # UNBLOCK-F2-03"],
            'build the 3-bin twin of the 5-bin phase array, using the num_phases2=3 the '
            'author already defined on line 5 and never referenced. Figure5_Regression '
            'needs 3 bins (num_task_phases=3, 9x3x12=324 regressors) while Figure2 '
            "own GLM (c23/c25) and phase-map c53 use the 5-bin Phases_raw_"),
        ins('F2-03', 'SCIENCE', 18, 44, 'times_from_start_trial=[]',
            ["                    phases_trial2=[]   # UNBLOCK-F2-03"],
            'per-trial accumulator for the 3-bin array'),
        ins('F2-03', 'SCIENCE', 18, 38, 'times_from_start_all=[]',
            ["                phases_all2=[]   # UNBLOCK-F2-03"],
            'per-session accumulator for the 3-bin array'),

        celltype('F2-04', 'SKIP', 20, 'raw', 'Neuron_raw_',
                 "make Run-All safe. Cell 20 splits combined recdays into single days and "
                 "np.save()s Neuron_raw_/Location_raw_/XY_raw_/trialtimes_ under single-day "
                 "names. Its outputs already exist on disk, it reads Distances_from_reward_ "
                 "(written only by cell 79) so it is a no-op on a fresh folder, and it does "
                 "not merely overwrite -- it RENUMBERS (deposited me08_12092021 has session "
                 "indices [0,1,3]; this would write contiguous 0,1,2). Source is untouched; "
                 "only cell_type changes, so the code remains readable in place"),

        rep('F2-05', 'ORDER', 23, 87, "distances=np.load(Input_folder+'Distances_from_reward_'",
            ["                    distances=Distances_from_reward_dic[mouse_recday][ses_ind_training]   # UNBLOCK-F2-05",
             "                    # (deposit loaded this from disk; see UNBLOCK-F2-05)"],
            'break the cell-23 <-> cell-79 circularity. Distances_from_reward_<rd>_<s>.npy '
            'is written ONLY by cell 79, the last cell, and this load sits at the OUTER '
            'per-recday try so a miss aborts the whole recday with "betas not calculated". '
            'Cell 25 lines 80/133 are the author\'s own verbatim in-memory form of exactly '
            'this value, so the substitution is provably equivalent'),
        rep('F2-05', 'ORDER', 23, 88,
            "mouse_recday+'_'+str(ses_ind_training)+'.npy',allow_pickle=True)",
            [], 'second line of the folded np.load call removed with it'),
        # The SAME load appears twice in cell 23 -- once for the training sessions
        # (line 87) and once for the held-out test session (line 141). Both must be
        # replaced: missing the second one leaves the FileNotFoundError in place, and
        # because it sits at the outer per-recday try the cell prints "betas not
        # calculated" for every recday and leaves GLM_dic2['mean_neuron_betas'] empty,
        # which then silently wastes cell 25 and kills cell 56 on
        # `thr_neuron_beta[:,0]` with "TypeError: unhashable type: 'slice'".
        rep('F2-05', 'ORDER', 23, 141,
            "distances_test=np.load(Input_folder+'Distances_from_reward_'",
            ["                distances_test=Distances_from_reward_dic[mouse_recday][ses_ind_test]   # UNBLOCK-F2-05"],
            'same substitution for the TEST session load; cell 25 line 133 is the '
            "author's own in-memory form of exactly this value"),
        rep('F2-05', 'ORDER', 23, 142,
            "mouse_recday+'_'+str(ses_ind_test)+'.npy',allow_pickle=True)",
            [], 'second line of that folded np.load call removed with it'),

        newcell('F2-06', 'NEWCELL', 31, F2_06_SOURCE,
                'produce the 360-bin Neuron_/Location_ arrays that six notebooks read and '
                'no deposited notebook writes'),

        rep('F2-07', 'NAMEDEF', 46, 2, '#Tuned_dic=rec_dd()',
            ["Tuned_dic=rec_dd()   # UNBLOCK-F2-07"],
            'uncomment. Tuned_dic is used in cells 46/48/56/60/63/72/79. Cells 56, 63 and '
            '79 have no try/except so it is a fatal NameError there; cells 46/48/60/72 '
            'swallow it and SILENTLY DISCARD ALL THEIR WORK (cell 46 prints "Not found" '
            'per recday). Without this nothing is persisted at all'),

        ins_before('F2-08', 'NAMEDEF', 56, 10, 'if use_both==True:',
                   ["use_both=True   # UNBLOCK-F2-08",
                    "# UNBLOCK-F2-08: uniquely determined -- line below is",
                    "# `if use_both==True: use_permuted=False` and use_permuted is read",
                    "# unguarded further down, so use_both=False would need a SECOND edit to",
                    "# define use_permuted. use_both is otherwise assigned only in cell 61,",
                    "# which runs LATER (his exec counts 84 then 94), and cell 56's own",
                    "# `try:` is commented out, so this is fatal on a clean kernel."],
                   'define use_both before its first use in cell 56'),
    ],

    'Figure2_UMAP.ipynb': [
        *paths('UM-01', 1, 2, 3, OUT_EPHYS, 'Figure2_UMAP -- was a Windows path'),
    ],

    'Figure3.ipynb': [
        *paths('F3-01', 2, 2, 3, OUT_EPHYS, 'Figure3 -- was a Windows path'),

        rep('F3-03', 'ORDER', 36, 47,
            "ephys_ = np.load(Input_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')",
            ["            try:   # UNBLOCK-F3-03",
             "                ephys_ = np.load(Input_folder+'Neuron_'+mouse_recday+'_'+\\",
             "                                 str(ses_ind)+'.npy')   # UNBLOCK-F3-03",
             "            except FileNotFoundError:   # UNBLOCK-F3-03",
             "                print('Not analyzed')   # UNBLOCK-F3-03",
             "                continue   # UNBLOCK-F3-03"],
            "guard the Neuron_ load, which is the only unguarded one in this notebook. "
            "Cell 36 iterates every session in awake_session_behaviour_ and gates only on "
            "num_trials_day[ses_ind]==0, but a session can have trialtimes (hence "
            "num_trials>0) and no Neuron_raw -- me10_14122021_15122021 session 5 is exactly "
            "that, so no 360-bin Neuron_ array can exist for it and the load dies with "
            "FileNotFoundError, aborting the notebook. Cell 29 wraps the IDENTICAL load in "
            "try/except and continues, and cell 36 already uses a continue-on-empty idiom "
            "two lines below, so this matches the notebook's own convention. Without it "
            "Figure3 cannot reach cell 45, which is the sole producer of the "
            "Xneuron_correlations_*_Angles_* files Figure7 needs"),

        rep('F3-02', 'ORDER', 54, 91, 'goal_progress_stability=np.nanmean',
            ["# UNBLOCK-F3-02: line disabled. It used mean_hist_fine, which is first bound",
             "# at line ~124 OF THIS SAME CELL, so on a clean kernel it raises NameError",
             "# BEFORE the np.save('sigma_goalprogress.npy') four lines below -- and cell 80",
             "# loads that file with no try. It only ran for the author because his kernel",
             "# already held the name from a previous execution (exec_count 119).",
             "# goal_progress_stability is never read again anywhere in the repo.",
             "#goal_progress_stability=np.nanmean(np.reshape(np.roll(mean_hist_fine,45),(4,90)),axis=0)   # UNBLOCK-F3-02"],
            'unblock the sigma_goalprogress.npy write that cell 80 depends on'),
    ],

    'Figure5_Figure6.ipynb': [
        *paths('F6-01', 2, 2, 3, OUT_EPHYS, 'Figure5_Figure6 -- was a Windows path'),
    ],

    'Figure5_Regression.ipynb': [
        *paths('F5-01', 1, 4, 5, OUT_EPHYS, 'Figure5_Regression'),

        rep('F5-02', 'FLAG', 32, 6, 'limited=False',
            ["limited=True ##if true restricts lags to single trial, if false extends lags beyond this (see below)   # UNBLOCK-F5-02"],
            'cells 15/21/26 write the 12-lag no-suffix files but cells 32/38 read the '
            '24-lag _beyond files, so as deposited the plotted histogram comes from a run '
            'the earlier cells never produce. 12 lags is what the Methods specify'),
        rep('F5-03', 'FLAG', 38, 6, 'limited=False',
            ["limited=True ##if true restricts lags to single trial, if false extends lags beyond this (see below)   # UNBLOCK-F5-03"],
            'same as F5-02, for the plotting cell'),
    ],

    'Figure7.ipynb': [
        *paths('F7-01', 1, 2, 3, OUT_EPHYS, 'Figure7 -- was a Windows path'),

        ins_before('F7-02', 'NAMEDEF', 76, 26,
                   'for phase_place_diff in np.arange(num_phase_place_diffs_):',
                   ["    num_phase_place_diffs_=5   # UNBLOCK-F7-02",
                    "    # UNBLOCK-F7-02: num_phase_place_diffs_ is undefined repo-wide, at module",
                    "    # level, ABOVE this cell's own try: -- a fatal NameError. Cell 76 is the",
                    "    # sole producer of cross_corr_dic, consumed by cells 82 and 87 and thence",
                    "    # 86 and 89. Cell 76 is a stale copy of cell 61, which iterates groups_;",
                    "    # the value 5 is recovered from HIS OWN stored cell-76 output, which",
                    "    # prints phase_place_diff = 0,1,2,3,4."],
                   'define the undefined loop bound in cell 76'),
    ],
}



# --------------------------------------------------------------------------- variants
# A VARIANT is a comparison run, not part of the minimal copy. It is written to its own
# filename and is NOT part of the audited 8-notebook set, so the "minimally edited copy"
# claim is unaffected. Variant lines carry `# VARIANT-<ID>` instead of `# UNBLOCK-<ID>`.
#
# elasticnet: flip Poisson_regression True -> False in every cell that branches on it.
# In cell 21 that selects `ElasticNet(alpha=0.01, positive=True)` (regularize=True is
# already set, and alpha becomes 0.01 -- the value the cell's own comment calls "used in
# paper", and the one the V5 reimplementation used). It also switches the output prefix
# from `Poisson_*` to bare, so the ElasticNet results sit alongside the Poisson ones
# rather than overwriting them.
VARIANTS = {
    'elasticnet': dict(
        src='Figure5_Regression.ipynb',
        dst='Figure5_Regression_elasticnet.ipynb',
        skip_cells=[15],          # prep arrays are Poisson-independent; reuse them
        ops=[
            ('EN-01', 21, 27, 'Poisson_regression=True'),
            ('EN-02', 26, 30, 'Poisson_regression=True'),
            ('EN-03', 32,  0, 'Poisson_regression=True'),
            ('EN-04', 38,  0, 'Poisson_regression=True'),
        ],
    ),
}

# prospective (2026-09-14): cell 15 builds retrospective lags only. Run the same bump model on
# the time-reversed session and un-reverse its output (= v5 generate_regressors_raw 'future').
# Neuron/Location arrays stay in neural time. Outputs take the '_prospective' suffix via the
# notebook's own addition/addition2 slot; 96_extract_figure5_stats.py reads them by that suffix.
VARIANTS['prospective'] = dict(
    src='Figure5_Regression.ipynb',
    dst='Figure5_Regression_prospective.ipynb',
    skip_cells=[],
    ops=[],
    subs=[
        ('PRO-01', 15, "    addition=''", "    addition='_prospective'"),
        ('PRO-02', 15, "nodes=(Location_raw_eq-1).astype(int)",
         "nodes=(Location_raw_eq-1).astype(int)[::-1].copy()"),
        ('PRO-03', 15, "        states=states_conc",
         "        states=np.asarray(states_conc)[::-1].copy()"),
        ('PRO-04', 15, "        phases=phases_conc",
         "        phases=np.asarray(phases_conc)[::-1].copy()"),
        ('PRO-05', 15, "regressors=np.roll(module_anchor_progress_all2,-1,axis=3)",
         "regressors=np.roll(module_anchor_progress_all2[::-1],-1,axis=3)"),
        ('PRO-06', 21, "    addition=''", "    addition='_prospective'"),
        ('PRO-07', 26, "    addition2=''", "    addition2='_prospective'"),
        ('PRO-08', 32, "    addition2=''", "    addition2='_prospective'"),
        ('PRO-09', 38, "    addition2=''", "    addition2='_prospective'"),
    ],
)


def write_variant(name):
    """Build a comparison notebook from the already-unblocked copy."""
    v = VARIANTS[name]
    src = os.path.join(UNBLOCKED, v['src'])
    dst = os.path.join(UNBLOCKED, v['dst'])
    nb = json.load(open(src))
    applied = []

    for ci, extra in [v['append_to_cell']] if v.get('append_to_cell') else []:
        cell = nb['cells'][ci]
        lines = ''.join(cell['source']).split('\n')
        lines += extra
        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
        applied.append('F3F-00')

    for eid, ci, old, new in v.get('subs', []):
        cell = nb['cells'][ci]
        src_txt = ''.join(cell['source'])
        if old not in src_txt:
            raise SystemExit(f'FATAL variant {name}: c{ci} substring not found: {old!r}')
        marker = f'   # VARIANT-{eid}'
        src_txt = src_txt.replace(old, new + marker)
        cell['source'] = [l + '\n' for l in src_txt.split('\n')[:-1]] + [src_txt.split('\n')[-1]]
        applied.append(eid)

    for eid, ci, line, guard in v['ops']:
        cell = nb['cells'][ci]
        lines = ''.join(cell['source']).split('\n')
        if guard not in lines[line]:
            raise SystemExit(f'FATAL variant {name}: {v["src"]} c{ci} L{line} '
                             f'guard {guard!r} not found; actual {lines[line]!r}')
        indent = _indent_of(lines[line])
        lines[line:line + 1] = [
            f'{indent}# VARIANT-{eid}: comparison run -- ElasticNet(alpha=0.01, positive=True)',
            f'{indent}#   instead of PoissonRegressor(alpha=1). Also switches the output',
            f'{indent}#   filename prefix from Poisson_* to bare, so both runs coexist.',
            f'{indent}Poisson_regression=False   # VARIANT-{eid}',
        ]
        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
        applied.append(eid)
    for c in nb['cells']:
        if c.get('cell_type') == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    with open(dst, 'w') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')
    print(f'WROTE {v["dst"]}  variant ops: {", ".join(applied)}')
    print(f'  run with: 40_run_notebook.py "{v["dst"]}" --skip '
          f'{",".join(str(x) for x in v["skip_cells"])}')


VARIANTS['figure3_fast'] = dict(
    src='Figure3.ipynb',
    dst='Figure3_fast.ipynb',
    skip_cells=[],
    append_to_cell=(9, ['', "# VARIANT-F3F-00: fast Pearson r. scipy 1.10's st.pearsonr spends ~1.08 ms per call", '#   computing a confidence interval and p-value that Figure3 discards -- both of its', '#   call sites take [0]. This centred dot product agrees with st.pearsonr(a,b)[0] to', '#   4.2e-17 over 200 random pairs and is ~35x faster, taking cells 29+36 from ~15.5 h', '#   to ~26 min. Edge cases match scipy: NaN input -> NaN; constant input (zero', '#   variance) -> NaN, where scipy emits ConstantInputWarning and returns NaN.', 'def _fast_pearson_r(a, b):                                        # VARIANT-F3F-00', '    a = np.asarray(a, dtype=float)                                # VARIANT-F3F-00', '    b = np.asarray(b, dtype=float)                                # VARIANT-F3F-00', '    a = a - a.mean()                                              # VARIANT-F3F-00', '    b = b - b.mean()                                              # VARIANT-F3F-00', '    d = np.sqrt((a * a).sum() * (b * b).sum())                    # VARIANT-F3F-00', '    if not d > 0:                                                 # VARIANT-F3F-00', '        return np.nan                                             # VARIANT-F3F-00', '    return float(np.clip((a * b).sum() / d, -1.0, 1.0))           # VARIANT-F3F-00', '']),
    subs=[
        ('F3F-01', 29, 'st.pearsonr(smoothed_mean_neuronX,smoothed_mean_neuronY_rotated)[0]',
         '_fast_pearson_r(smoothed_mean_neuronX,smoothed_mean_neuronY_rotated)'),
        ('F3F-02', 36, 'st.pearsonr(smoothed_mean_neuronX,smoothed_mean_neuronY_rotated)[0]',
         '_fast_pearson_r(smoothed_mean_neuronX,smoothed_mean_neuronY_rotated)'),
    ],
    ops=[],
)

# ---------------------------------------------------------------------- application
def _indent_of(s):
    return s[:len(s) - len(s.lstrip())]


def marker_comment(eid, cls, reason, indent=''):
    """The one mandatory preceding comment line(s) for an edit."""
    width = max(40, 96 - len(indent))
    words, lines, cur = reason.split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    out = [f'{indent}# UNBLOCK-{eid} ({cls}): {lines[0]}']
    out += [f'{indent}#   {l}' for l in lines[1:]]
    return out


def comment_indent(guard_line, new_lines):
    """Indent the marker comment to match the CODE IT INTRODUCES, not the guard line.

    Guards sometimes land on a folded continuation line (e.g. Figure2 cell 18 line 57,
    which is indented to column 67 inside a multi-line np.hstack). Inheriting that
    indent leaves the comment dangling far to the right. Prefer the indent of the first
    non-blank inserted line; fall back to the guard line for pure deletions.
    """
    for l in new_lines:
        if l.strip():
            return _indent_of(l)
    return _indent_of(guard_line)


def apply_notebook(name, ops, check_only=False):
    src = os.path.join(DEPOSIT, name)
    dst = os.path.join(UNBLOCKED, name)
    nb = json.load(open(src))

    # --- G-01: archive the deposit's outputs, then clear them.
    arch = os.path.join(LOGDIR, 'deposit_outputs')
    os.makedirs(arch, exist_ok=True)
    if not check_only:
        stored = {i: c.get('outputs', []) for i, c in enumerate(nb['cells'])
                  if c.get('outputs')}
        with open(os.path.join(arch, name.replace('.ipynb', '.outputs.json')), 'w') as f:
            json.dump({str(k): v for k, v in stored.items()}, f)

    # --- group ops by cell, apply bottom-up within each cell
    by_cell: dict[int, list] = {}
    newcells, ctypes = [], []
    for o in ops:
        if o['op'] == 'new_cell':
            newcells.append(o)
        elif o['op'] == 'cell_type':
            ctypes.append(o)
        else:
            by_cell.setdefault(o['cell'], []).append(o)

    applied = []
    for ci, cops in by_cell.items():
        cell = nb['cells'][ci]
        lines = ''.join(cell['source']).split('\n')

        def sort_key(o):
            return (o['line'], 0 if o['op'] == 'replace' else 1)

        for o in sorted(cops, key=sort_key, reverse=True):
            n = o['line']
            if n >= len(lines):
                raise SystemExit(f'FATAL {name} c{ci}: line {n} beyond cell '
                                 f'({len(lines)} lines) for {o["id"]}')
            if o['guard'] not in lines[n]:
                raise SystemExit(
                    f'FATAL {name} c{ci} L{n} guard failed for {o["id"]}\n'
                    f'  expected substring: {o["guard"]!r}\n'
                    f'  actual line       : {lines[n]!r}')
            indent = comment_indent(lines[n], o['new'])
            cmt = marker_comment(o['id'], o['cls'], o['reason'], indent)
            if o['op'] == 'replace':
                lines[n:n + 1] = cmt + o['new']
            elif o['op'] == 'insert_after':
                lines[n + 1:n + 1] = cmt + o['new']
            elif o['op'] == 'insert_before':
                lines[n:n] = cmt + o['new']
            applied.append(o['id'])

        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]

    for o in ctypes:
        cell = nb['cells'][o['cell']]
        joined = ''.join(cell['source'])
        if o['guard'] not in joined:
            raise SystemExit(f'FATAL {name} c{o["cell"]}: guard {o["guard"]!r} '
                             f'not in cell for {o["id"]}')
        cell['cell_type'] = o['to']
        # A raw cell must not carry execution_count / outputs.
        cell.pop('execution_count', None)
        cell.pop('outputs', None)
        cell.setdefault('metadata', {})['unblock'] = (
            f'UNBLOCK-{o["id"]} ({o["cls"]}): {o["reason"]}')
        applied.append(o['id'])

    for o in sorted(newcells, key=lambda x: x['before'], reverse=True):
        lines = o['source']
        nb['cells'].insert(o['before'], {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {'unblock': f'UNBLOCK-{o["id"]} ({o["cls"]}): {o["reason"]}'},
            'outputs': [],
            'source': [l + '\n' for l in lines[:-1]] + [lines[-1]],
        })
        applied.append(o['id'])

    # --- G-01 continued: clear outputs / exec counts on every cell
    for c in nb['cells']:
        if c.get('cell_type') == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    if not check_only:
        with open(dst, 'w') as f:
            json.dump(nb, f, indent=1)
            f.write('\n')
    return sorted(set(applied))


NOT_EDITED = """
## Deliberately NOT edited, with the consequence recorded

Distinguishing a deliberate omission from an oversight is the point of this section.

| Deposit behaviour | Why it is left alone |
|---|---|
| **Figure2 cell 61 `use_both=False`** | Leaving it is both more minimal and *better*. `use_both=True` at cell 56 then `False` at cell 61 is exactly the kernel state his execution counts (84 then 94) imply, so it preserves cell 61's stored output as a numeric checkpoint. Its only cost is that cell 63 pickles two empty-defaultdict files, `Goal_progress_strict_` and `Place_strict_`, and **nothing reads either**. The four files that matter (`Place_`, `Goal_progress_`, `State_`, `State_strict_`) are real regardless, because `State_strict_` comes from `Tuned_dic['State_zmax_bool_strict']` either way. |
| **Figure2 cell 79 line 8 `['3_task','combined']`** | Cell 18 populated `combined_ABCDonly` and `3_task_all`, so this pair differs. Rather than edit it, `10_make_bookkeeping.py` creates `combined_days.npy`, and the unedited cell runs. Any `(rd, ses)` cell 18 skipped autovivifies an empty defaultdict that `np.save` pickles under a legitimate filename -- inert, because `Num_trials_ = 0` stops `non_repeat_ses_maker` selecting those sessions, but the empties should be counted and logged. |
| **Figure2 cell 79 overwrites cell 63's `Place_`/`State_`** | With a *different* definition (permutation boolean vs t-test boolean), purely because it runs later. That is the author's ordering; record which definition survives rather than change it. |
| **Figure5_Regression cell 21** | Runs correctly as deposited -- see the note on `REPLICATION_STATUS.md` §5(a) below. Four known scientific quirks are left intact: (a) it reads `tuning_phase_boolean_max[ses_ind_actual]`, the **held-out** session, for both fit and scoring, which is leakage; (b) cell 26 hardcodes `num_non_repeat_ses_found = 6` for `me11_05122021_06122021` while cells 15/21 build 7 folds, so folds 3-5 pair regressors from sessions 3,4,5 with phases/trial-times from 4,5,7 and session 7 is never scored (46 neurons); (c) it fits `PoissonRegressor` (log link, fitted intercept) but cell 26 reads out `np.sum(regressors*coeffs)` -- linear, no `exp`, no intercept, so the non-linear-link robustness check is discarded at readout; (d) `found_ses` and `num_non_repeat_ses_found` are derived from different criteria, so folds can silently pair with the wrong session's phases. |
| **Figure5_Figure6 cell 57's `confition` typo** | A genuine typo, but in a third `elif` that is unreachable under the deposited `condition='non-zero-strict'`, which matches the second branch. Editing it would change nothing. |
| **Figure5_Figure6 cell 89's `curve_fit` / `func_decay`** | Neither is imported or defined in that notebook (they live in Figure3's helper cell) -- but both uses sit inside `if use_kernel==True:` and `use_kernel=False` is set in the same cell and never reassigned. Dead code. Same for `_previous_chocies_coefficients_ABCD.npy`. |
| **Figure5_Figure6 cell 9** | Not edited; **skip it at run time**. It is a standalone exporter with zero consumers inside its own notebook, and the deposited `Xneuron_correlations` joblib cannot substitute for Figure3 cell 45 anyway: it has measures `Max_bins`/`Correlations`/`angle_units` with **no `Angles`**, and covers 3 of 25 recdays. Its `os.mkdir` is already wrapped in `try/except FileExistsError`, so it is a no-op rather than a crash. |
| **Figure3 cell 80 line 42's `ses_ind`** | Undefined in that cell; it leaks from cell 59's loop. Not a `NameError` if cell 59 ran, but it silently loads whichever session index happened to be left bound. Flagged, not fixed -- fixing it would require deciding what he meant. |
| **Figure7 cell 28's `except` branch** | Prints `All_session_ind`, which leaks from cell 24; if no sleep session ever matched, the handler itself raises `NameError`. Reachable only on an error path. |
| **Figure2 cell 20** | Source untouched -- only `cell_type` changed (`F2-04`), so the code stays readable in place while Run-All skips it. |

## Corrections to `REPLICATION_STATUS.md` established while building this copy

1. **§5(a) is wrong.** `Figure5_Regression` cell 21 runs as deposited and needs no fix. The doc reads `np.hstack((np.vstack([...])))` as leaving one entry per (trial, state); but those are *grouping* parens, not a tuple, so `np.hstack` on the `(n_trials, 4)` object array unpacks rows into a `(4*n_trials,)` object array of per-bin 1-D arrays and the single `concatenate_complex2` then descends to per-bin scalars. Reproduced: 1773/1773 bins, correct trial-major/state-minor order, works even on numpy 2.0.2 because the final list is scalars. **Adding a second concatenation would break it.**
2. **§2 is wrong on the paths.** Each notebook has exactly one `Input_folder`/`Output_folder` pair, not two. Four carried Windows paths (`Figure2_UMAP`, `Figure3`, `Figure5_Figure6`, `Figure7`); `Figure2`, `Figure5_Regression` and `Behavioural Analysis` were already local but lacked the trailing slash on `Output_folder`.
3. **§3(i) overstates recoverability.** Recovering session bookkeeping from `session_dic`/`Variable_dic` resolves only 3 of 25 recdays (they are keyed by single-day names), and mice `ab03`/`ah07` -- 6 of the 25 recdays -- are absent from both dicts entirely. The MetaData CSVs cover all of them.
4. **`pingouin` is not imported by `Figure2`** at all, so it never blocked the notebook that does the heavy regeneration.
5. **A blocker the doc missed:** `Figure2` cell 16 builds `speed_dic` over `['combined_ABCDonly']` only, while cell 18 iterates `['combined_ABCDonly','3_task_all']` -- see `F2-02`.
6. **Three orphan families the doc listed as "written by Figure2":** `Neuron_<rd>_<s>.npy`, `Location_<rd>_<s>.npy` and `State_95<rd>.npy` are written by **no** deposited notebook. See `F2-06` and `30_bridge_state_aliases.py`.
"""


def write_ledger():
    """Emit EDITS.md from the same EDITS table the patcher uses, so it cannot drift."""
    rows = []
    for nb in NOTEBOOKS:
        for o in EDITS.get(nb, []):
            where = (f"cell {o['before']} (new cell before)" if o['op'] == 'new_cell'
                     else f"cell {o['cell']}"
                     + (f" L{o['line']}" if 'line' in o else ''))
            rows.append((o['id'], nb, where, o['cls'], o['op'], o['reason']))

    seen, merged = {}, []
    for eid, nb, where, cls, op, reason in rows:
        key = (eid, nb)
        if key in seen:
            seen[key][2].append(where)
            seen[key][5].append(reason)
        else:
            seen[key] = [eid, nb, [where], cls, op, [reason]]
            merged.append(seen[key])

    lines = [
        '# `mFC_schema-main_unblocked` -- the edit ledger',
        '',
        'Every edit applied to El-Gaby\'s deposited notebooks, and why each is strictly',
        'required to make the code execute. The deposit itself is untouched and lives at',
        '`../mFC_schema-main/`.',
        '',
        'This file is **generated** by `_preflight/20_apply_edits.py --ledger` from the same',
        'edit table the patcher applies, so it cannot drift out of sync with the notebooks.',
        '`_preflight/90_audit_diff.py` asserts a bijection between the edit IDs appearing in',
        'the diff and the IDs listed here.',
        '',
        '## Conventions',
        '',
        '* Every changed or added source line carries a trailing `# UNBLOCK-<ID>` token.',
        '* Every edit is preceded by one `# UNBLOCK-<ID> (<CLASS>): ...` comment in the notebook.',
        '* Classes: `PATH`, `NAMEDEF` (undefined name), `ORDER` (dependency/definition order),',
        '  `COVERAGE` (cohort coverage), `SCIENCE` (changes a computed quantity), `FLAG`,',
        '  `SKIP`, `NEWCELL`.',
        '* `G-01` (global, all 8 notebooks): `outputs` cleared and `execution_count` set to',
        '  `null`, so the audit only ever compares source. The deposit\'s stored outputs are',
        '  archived first, to `replication_run/logs/deposit_outputs/` -- they are the numeric',
        '  reference for verification.',
        '',
        f'## The {len({r[0] for r in rows})} edits',
        '',
        '| ID | Notebook | Where | Class | Why strictly necessary |',
        '|---|---|---|---|---|',
    ]
    for eid, nb, wheres, cls, op, reasons in merged:
        reason = ' **/** '.join(dict.fromkeys(reasons))
        reason = reason.replace('|', '\\|')
        lines.append(f'| `UNBLOCK-{eid}` | {nb} | {", ".join(wheres)} | `{cls}` | {reason} |')

    lines += ['', '**Exactly one edit changes a scientific quantity: `F2-03`.**', '',
              NOT_EDITED.strip(), '']
    path = os.path.join(UNBLOCKED, 'EDITS.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {path} ({len({r[0] for r in rows})} edit IDs)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='validate all guards without writing anything')
    ap.add_argument('--ledger', action='store_true',
                    help='regenerate EDITS.md only')
    ap.add_argument('--variant', default='',
                    help='build a comparison notebook (e.g. elasticnet); not audited')
    args = ap.parse_args()

    if args.ledger:
        write_ledger()
        return 0

    if args.variant:
        write_variant(args.variant)
        return 0

    os.makedirs(UNBLOCKED, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)

    total = {}
    for name in NOTEBOOKS:
        ops = EDITS.get(name, [])
        ids = apply_notebook(name, ops, check_only=args.check)
        total[name] = ids
        print(f'{"CHECK" if args.check else "WROTE"} {name:40s} edits: {", ".join(ids) or "(none)"}')

    if not args.check:
        for v in VERBATIM:
            s = os.path.join(DEPOSIT, v)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(UNBLOCKED, v))
                print(f'COPIED verbatim {v}')

    allids = sorted({i for ids in total.values() for i in ids})
    print(f'\n{len(allids)} distinct edit IDs: {", ".join(allids)}')
    with open(os.path.join(LOGDIR, 'applied_edits.json'), 'w') as f:
        json.dump(total, f, indent=2)

    if not args.check:
        write_ledger()


if __name__ == '__main__':
    sys.exit(main())
