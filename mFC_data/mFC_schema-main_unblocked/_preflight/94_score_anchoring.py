#!/usr/bin/env python3
"""Score our Figure5_Figure6 anchoring against El-Gaby's own stored cell outputs.

    python 98_score_anchoring.py [--json-only]
    -> logs/anchoring_score.json, logs/anchoring_score.md

Why this exists: Figure7's pair axis comes from the anchoring, El-Gaby deposited none of the
anchoring arrays, and REPRODUCTION.md states the reproduction "was never checked against his
stored outputs at the level of anchored fractions or lags". It can be -- he ran every cell of
Figure5_Figure6 (execution_count on all 112) and the archive keeps what they printed. Cell
indices align 1:1 between the deposit and both executed copies; the F2-06 +1 shift is Figure2
only.

The PFC copy scores ours against his. The LEC copy has no deposit yardstick, so it scores LEC
against our PFC run and says so in the report -- a cross-region comparison, not a reproduction
check.

I/O: reads logs/deposit_outputs/Figure5_Figure6.outputs.json (or the PFC executed notebook,
for REGION='lec') and executed/Figure5_Figure6.ipynb; writes the two report files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

REGION = 'pfc'                                   # the LEC copy says 'lec'
ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
RUN = 'replication_run' if REGION == 'pfc' else 'lec_replication_run'
LOGDIR = os.path.join(ROOT, RUN, 'logs')
EXECUTED = os.path.join(ROOT, RUN, 'executed', 'Figure5_Figure6.ipynb')
# REGION='lec' has no stored LEC outputs, so the reference is our PFC run.
REFERENCE = (os.path.join(LOGDIR, 'deposit_outputs', 'Figure5_Figure6.outputs.json')
             if REGION == 'pfc'
             else os.path.join(ROOT, 'replication_run', 'executed', 'Figure5_Figure6.ipynb'))
REF_NAME = "El-Gaby's stored outputs" if REGION == 'pfc' else 'our PFC run'

# A relative gap above this is called DIFFERS. 2 % is the resolution that matters here: the
# measured PFC coherent-proportion gap is 11 % and the cell-34 t gap 11-46 %, while the
# quantities that genuinely match (cell 61, cell 69 n) agree to under 0.5 %.
TOL = 0.02

# Rows whose value does not scale with the neuron count. Only these are meaningful when the
# reference is the other region (REGION='lec'), where n differs ~2x by construction.
SCALE_FREE = ('coherent_proportion', '.t', 'hist.cosine', '.statistic', 'z')

RECDAY = re.compile(r'^[a-z]{2}\d{2}_\d{8}')
TTEST = re.compile(r'TtestResult\(statistic=([-\d.e+]+), pvalue=([-\d.e+]+), df=(\d+)\)')
KSTEST = re.compile(r'KstestResult\(statistic=([-\d.e+]+), pvalue=([-\d.e+]+)')
TWOPROP = re.compile(r'\(([-\d.e+]+), ([-\d.e+]+)\)')


def stream(cell):
    return ''.join((o.get('text', '') if isinstance(o.get('text', ''), str) else ''.join(o.get('text', '')))
                   for o in cell.get('outputs', []) if o.get('output_type') == 'stream')


def body(text):
    """Printed lines that are not recday banners, warnings or echoed source."""
    out = []
    for l in text.split('\n'):
        s = l.strip()
        if not s or RECDAY.match(s) or 'Warning' in s or s.startswith(('C:\\', '/nfs', '/ceph')):
            continue
        out.append(s)
    return out


def _floats(lines):
    """Every bare float printed on its own line, in order."""
    return [float(s) for s in lines if re.fullmatch(r'-?\d+(\.\d+)?(e[-+]?\d+)?', s)]


def p_cell34(lines):
    """Three (subset, n, t) triples: all / non-zero-lag 30 / non-zero-lag 90."""
    out, names = {}, []
    for i, s in enumerate(lines):
        if s.startswith('mean_bestlag_corr'):
            names.append(s)
        m = TTEST.search(s)
        if m and names:
            k = names[-1]
            n = next((int(x) for x in reversed(lines[:i]) if re.fullmatch(r'\d+', x)), None)
            out[f'{k}.n'], out[f'{k}.t'] = n, float(m.group(1))
    return out


def p_hist(lines):
    """The 24-bin anchor-angle histogram, printed as a wrapped numpy array."""
    txt = ' '.join(l for l in lines if re.match(r'^[\[\s\d.-]', l))
    v = [float(x) for x in re.findall(r'-?\d+\.\d+(?:e[-+]?\d+)?', txt)]
    return {'hist': v} if len(v) >= 8 else {}


def p_coh(lines):
    out = {}
    for s in lines:
        if s.startswith('Total coherent proportion:'):
            out['coherent_proportion'] = float(s.split(':')[1])
        m = TWOPROP.fullmatch(s)
        if m:
            out['z'] = float(m.group(1))
        if re.fullmatch(r'[\d.]+ \d+', s):
            a, b = s.split()
            out['coherent_n'], out['total_n'] = float(a), float(b)
    return out


def p_cell69(lines):
    """'All neurons' then 'Non spatial neurons', each with n and a t-test."""
    out, label = {}, None
    for i, s in enumerate(lines):
        if s in ('All neurons', 'Non spatial neurons'):
            label = s.replace(' ', '_')
        m = TTEST.search(s)
        if m and label:
            n = next((int(x) for x in reversed(lines[:i]) if re.fullmatch(r'\d+', x)), None)
            out[f'{label}.n'], out[f'{label}.t'] = n, float(m.group(1))
    return out


def p_cell72(lines):
    out, ns = {}, [int(x) for x in lines if re.fullmatch(r'\d+', x)]
    ks = [float(m.group(1)) for m in (KSTEST.search(l) for l in lines) if m]
    for j, (n, k) in enumerate(zip(ns, ks)):
        out[f'ks{j}.n'], out[f'ks{j}.statistic'] = n, k
    return out


CHECKS = [
    (34, 'cross-validated best-lag correlation, state-tuned (3 pools)', p_cell34),
    (40, 'anchor-angle histogram, all neurons', p_hist),
    (42, 'coherent proportion, all neurons', p_coh),
    (59, 'anchor-angle histogram, non-zero-lag', p_hist),
    (61, 'coherent proportion, non-zero-lag', p_coh),
    (62, 'two-proportion test, non-zero-lag pool', p_coh),
    (69, 'cross-validated correlation by spatial subset', p_cell69),
    (72, 'lag distribution vs uniform (KS)', p_cell72),
]


def cosine(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    return sum(x * y for x, y in zip(a, b)) / (da * db) if da and db else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-only', action='store_true')
    args = ap.parse_args()

    for p in (REFERENCE, EXECUTED):
        if not os.path.exists(p):
            print(f'MISSING: {p}')
            return 2

    ours = json.load(open(EXECUTED))['cells']
    if REFERENCE.endswith('.ipynb'):
        ref_cells = json.load(open(REFERENCE))['cells']
        ref = {str(i): c.get('outputs', []) for i, c in enumerate(ref_cells)}
    else:
        ref = json.load(open(REFERENCE))

    rows, n_cmp, n_diff, n_scale, missing = [], 0, 0, 0, []
    for idx, label, parse in CHECKS:
        r_out = ref.get(str(idx))
        if r_out is None or idx >= len(ours):
            missing.append(idx)
            continue
        r = parse(body(stream({'outputs': r_out})))
        o = parse(body(stream(ours[idx])))
        if not r or not o:
            missing.append(idx)
            continue
        for k in sorted(set(r) | set(o)):
            rv, ov = r.get(k), o.get(k)
            if k == 'hist' and rv and ov:
                c = cosine(rv, ov)
                rows.append(dict(cell=idx, label=label, key='hist.cosine', ref=1.0, ours=c,
                                 rel=None if c is None else abs(1 - c),
                                 verdict='MATCH' if c is not None and c > 1 - TOL else 'DIFFERS'))
                rows.append(dict(cell=idx, label=label, key='hist.sum', ref=sum(rv), ours=sum(ov),
                                 rel=abs(sum(ov) - sum(rv)) / abs(sum(rv)) if sum(rv) else None,
                                 verdict=('SCALE' if REGION != 'pfc' else
                                          ('MATCH' if sum(rv) and abs(sum(ov) - sum(rv)) / abs(sum(rv)) <= TOL else 'DIFFERS'))))
                n_cmp += 2
                n_diff += sum(1 for x in rows[-2:] if x['verdict'] == 'DIFFERS')
                n_scale += sum(1 for x in rows[-2:] if x['verdict'] == 'SCALE')
                continue
            if rv is None or ov is None:
                rows.append(dict(cell=idx, label=label, key=k, ref=rv, ours=ov, rel=None,
                                 verdict='ONE SIDE ONLY'))
                continue
            rel = abs(ov - rv) / abs(rv) if rv else (0.0 if ov == 0 else None)
            v = 'MATCH' if rel is not None and rel <= TOL else 'DIFFERS'
            if REGION != 'pfc' and not any(t in k for t in SCALE_FREE):
                v = 'SCALE'            # n-dependent; not comparable across regions
            rows.append(dict(cell=idx, label=label, key=k, ref=rv, ours=ov, rel=rel, verdict=v))
            n_cmp += 1
            n_diff += v == 'DIFFERS'
            n_scale += v == 'SCALE'

    n_eff = n_cmp - n_scale
    tag = ('NOT RUN' if n_cmp == 0 else 'ALL MATCH' if n_diff == 0
           else f'{n_diff} of {n_eff} comparable DIFFER' + (f' ({n_scale} scale-dependent, not compared)' if n_scale else ''))
    print(f'=== Figure5_Figure6 anchoring: {RUN} vs {REF_NAME} ===')
    print(f'{"cell":>5} {"quantity":44} {"reference":>14} {"ours":>14} {"rel":>8}  verdict')
    for r in rows:
        f = lambda x: ('' if x is None else (f'{x:14.4f}' if isinstance(x, float) else f'{x:>14}'))
        rel = '' if r['rel'] is None else f"{r['rel']*100:7.2f}%"
        print(f"{r['cell']:>5} {r['key'][:44]:44} {f(r['ref'])} {f(r['ours'])} {rel:>8}  {r['verdict']}")
    print(f'\n{tag}; {n_cmp} comparisons' + (f'; cells with no parsable output: {missing}' if missing else ''))

    os.makedirs(LOGDIR, exist_ok=True)
    json.dump(dict(region=REGION, reference=REFERENCE, reference_name=REF_NAME,
                   executed=EXECUTED, tolerance=TOL, n_comparisons=n_cmp,
                   n_differ=n_diff, n_scale_dependent=n_scale, verdict=tag, missing_cells=missing, rows=rows),
              open(os.path.join(LOGDIR, 'anchoring_score.json'), 'w'), indent=2)
    if not args.json_only:
        with open(os.path.join(LOGDIR, 'anchoring_score.md'), 'w') as f:
            f.write(f'# Figure5_Figure6 anchoring: {RUN} vs {REF_NAME}\n\n')
            f.write(f'{tag}; {n_cmp} comparisons; tolerance {TOL:.0%}\n\n')
            f.write('| cell | quantity | reference | ours | rel | verdict |\n|---|---|---|---|---|---|\n')
            for r in rows:
                g = lambda x: '' if x is None else (f'{x:.4f}' if isinstance(x, float) else str(x))
                rel = '' if r['rel'] is None else f"{r['rel']*100:.2f}%"
                f.write(f"| {r['cell']} | `{r['key']}` | {g(r['ref'])} | {g(r['ours'])} | {rel} | {r['verdict']} |\n")
    print(f"wrote {LOGDIR}/anchoring_score.json" + ('' if args.json_only else ' and .md'))
    return 1 if n_cmp == 0 else 0


if __name__ == '__main__':
    sys.exit(main())
