#!/usr/bin/env python3
"""Score the run against El-Gaby's OWN stored cell outputs.

His stored outputs are the gold standard here, because the anchoring intermediates were
never deposited: agreement with them is evidence that our regenerated inputs match his,
in a way agreement with the printed figures cannot be.

The deposit's outputs are archived by `20_apply_edits.py` (edit G-01) into
`replication_run/logs/deposit_outputs/`; the executed notebooks land in
`replication_run/executed/`. This compares the two.

Usage:  python 95_checkpoints.py [Figure2.ipynb ...]
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
LOGDIR = os.path.join(ROOT, 'replication_run', 'logs')
ARCHIVE = os.path.join(LOGDIR, 'deposit_outputs')
EXECUTED = os.path.join(ROOT, 'replication_run', 'executed')

RECDAY_RE = re.compile(r'^[a-z]{2}\d{2}_\d{8}(_\d{8})?$')

# CHECKPOINTS below are keyed by DEPOSIT cell index, because that is how the archived
# outputs are keyed. The executed notebooks use the UNBLOCKED numbering, which differs
# wherever a cell was inserted: UNBLOCK-F2-06 adds a cell at index 31 of Figure2, so
# every deposit cell >= 31 sits one later in the executed copy. Comparing arch[str(c)]
# with ours[c] directly would line up cell 46's deposit output against cell 46 of our
# run -- an empty spacer.
INSERTIONS = {'Figure2.ipynb': [31]}


def to_executed_index(notebook, deposit_cell):
    return deposit_cell + sum(1 for at in INSERTIONS.get(notebook, [])
                              if at <= deposit_cell)

# cell -> (label, what to count/expect). `expect` is informational: printed alongside.
CHECKPOINTS = {
    'Figure2.ipynb': {
        18: ('recdays iterated / "not made"', '80 recdays, <= 4 not made'),
        23: ('recdays / "Files not found for session"', '36 recdays, 5x session 5'),
        25: ('recdays / errors', '36 recdays, 0 errors'),
        31: ('recdays', '36'),
        40: ('recdays', '36'),
        46: ('recdays', '11'),
        48: ('recdays', '36'),
        53: ('recdays', '84'),
        56: ('recdays', '25'),
        61: ('tuning counts', 'thr95 n=2182 phase=1825 phase_state=1162 state=1287; '
                              'thr99 phase=1701 state=860'),
        65: ('proportions', '(printed proportions)'),
    },
    'Figure5_Regression.ipynb': {
        32: ('recdays found / neurons', '25 / 25 / 1252'),
        38: ('n neurons + t-test per panel', 'his _beyond run: 482 / 278 / 69 -- NOT our '
                                            'target after UNBLOCK-F5-02/03 (limited=True)'),
    },
    'Figure7.ipynb': {
        76: ('phase_place_diff values', '0,1,2,3,4'),
    },
}


def text_of(outputs):
    out = []
    for o in outputs or []:
        t = o.get('output_type')
        if t == 'stream':
            out.append(''.join(o.get('text', '')))
        elif t == 'error':
            out.append('ERROR: ' + (o.get('ename', '')) + ': ' + str(o.get('evalue', ''))[:200])
        elif t in ('execute_result', 'display_data'):
            d = o.get('data', {})
            if 'text/plain' in d:
                out.append(''.join(d['text/plain']))
    return '\n'.join(out)


# Noise that differs run-to-run and must not count as a divergence:
#  * object memory addresses, e.g. "<AxesImage at 0x78337fc3f950>"
#  * numpy >= 2.2 adds "shape=(N,)" to a truncated array repr; numpy 1.22 does not, so
#    the archived outputs (some of which came from a recent local run) differ cosmetically
#  * matplotlib figure-size banners
_NOISE = [
    (re.compile(r'0x[0-9a-fA-F]+'), '0xADDR'),
    (re.compile(r',\s*shape=\(\d+,?\)'), ''),
    (re.compile(r'<Figure size [\d.]+x[\d.]+ with \d+ Axes>'), '<Figure>'),
]


def denoise(txt):
    for rx, rep in _NOISE:
        txt = rx.sub(rep, txt)
    return txt


def summarise(txt):
    txt = denoise(txt)
    lines = [l.strip() for l in txt.splitlines()]
    recdays = {l for l in lines if RECDAY_RE.match(l)}
    return dict(
        recdays=len(recdays),
        not_made=txt.count('not made'),
        not_found=len(re.findall(r'Files not found for session', txt)),
        not_found_s5=len(re.findall(r'Files not found for session 5\b', txt)),
        not_analysed=txt.count('not analysed'),
        errors=txt.count('ERROR:'),
        betas_not=txt.count('betas not calculated'),
        notfound_generic=txt.count('Not found'),
        ints=re.findall(r'\b\d{3,5}\b', txt)[:12],
    )


def main():
    names = sys.argv[1:] or list(CHECKPOINTS)
    rows = []
    for name in names:
        arch_p = os.path.join(ARCHIVE, name.replace('.ipynb', '.outputs.json'))
        exec_p = os.path.join(EXECUTED, name)
        if not os.path.exists(arch_p):
            print(f'-- {name}: no archived deposit outputs, skipping')
            continue
        arch = json.load(open(arch_p))
        ours = {}
        if os.path.exists(exec_p):
            nb = json.load(open(exec_p))
            ours = {i: c.get('outputs', []) for i, c in enumerate(nb['cells'])}
        else:
            print(f'-- {name}: not executed yet ({exec_p})')

        print(f'\n{"=" * 100}\n=== {name}\n{"=" * 100}')
        for cell, (label, expect) in sorted(CHECKPOINTS.get(name, {}).items()):
            ec = to_executed_index(name, cell)
            his = summarise(text_of(arch.get(str(cell))))
            mine = summarise(text_of(ours.get(ec))) if ours else None
            shift = '' if ec == cell else f' (executed cell {ec})'
            print(f'\n-- deposit cell {cell}{shift}: {label}')
            print(f'   expected (his): {expect}')
            keys = ['recdays', 'not_made', 'not_found', 'not_found_s5', 'not_analysed',
                    'errors', 'betas_not', 'notfound_generic']
            hs = ' '.join(f'{k}={his[k]}' for k in keys if his[k])
            print(f'   deposit : {hs or "(nothing counted)"}')
            if mine is not None:
                ms = ' '.join(f'{k}={mine[k]}' for k in keys if mine[k])
                same = his['recdays'] == mine['recdays']
                print(f'   ours    : {ms or "(nothing counted)"}'
                      f'   [recdays {"MATCH" if same else "DIFFER"}]')
                rows.append(dict(notebook=name, deposit_cell=cell,
                                 executed_cell=ec, label=label,
                                 deposit={k: his[k] for k in keys},
                                 ours={k: mine[k] for k in keys},
                                 recdays_match=same))
            if his['ints']:
                print(f'   deposit numbers: {his["ints"]}')
            if mine is not None and mine['ints']:
                print(f'   our numbers    : {mine["ints"]}')

    with open(os.path.join(LOGDIR, 'checkpoints.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nwrote {LOGDIR}/checkpoints.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
