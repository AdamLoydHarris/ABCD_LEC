"""Refuse to launch when `code/` and `mFC_data/code/` have drifted.

Repo convention is DUPLICATION, not import: the PFC tree carries its own copy of every shared
module so the two datasets are analysed by literally the same code. That only holds if the
copies are kept identical, and twice a 25-job production launch has died on a lag between
them -- once on a missing `import time`, once with `glm_cv.py` 150 lines behind. Both would
have been caught by running this first, which is why `submit_glm_lec.sh` and
`submit_glm_pfc.sh` now do.

    python code/check_mirror_parity.py            # exit 0 = in sync, 1 = drift (diffs printed)
    python code/check_mirror_parity.py --list     # also print the names compared

Three kinds of check:

  * BYTE identity for modules that must be identical end to end (`glm_cv.py`,
    `glm_v3_synthetics.py`).
  * SHARED-DEFINITION identity for `glm_analysis_v2.py` / `glm_analysis_v3.py`: every
    top-level function, class and constant present in BOTH copies must have the same CODE --
    compared as docstring-stripped syntax trees, so comments and docstrings may differ but no
    statement may; a name present in only one copy is drift unless it is in `PFC_ONLY` (the
    PFC data loader block, which the LEC copy legitimately lacks). The two v2 copies carry two
    pre-existing code differences (`KNOWN_V2_DRIFT`); v2 is frozen so the production fits
    stay reproducible, and both are fixed in the v3 copies, which must be clean.
  * The shared PLOT block of `glm_plots.py` / `pfc_glm_plots.py` (from the palette to the
    `LEC only` marker), and `recday_registry.is_post_refit_section` + its regex.
"""

from __future__ import annotations

import ast
import difflib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

BYTE_PAIRS = [
    ('code/glm_cv.py', 'mFC_data/code/glm_cv.py'),
    ('code/glm_v3_synthetics.py', 'mFC_data/code/glm_v3_synthetics.py'),
    # Added 2026-09-14 after the PFC copy was found 15 lines behind (regenerate_summary's
    # `semantics=`/`suffix=` args). The V5 anchoring runs on both datasets import their
    # own tree's copy, so a drift here makes LEC and PFC numbers non-comparable silently.
    ('code/elasticnet_regression_v5.py', 'mFC_data/code/elasticnet_regression_v5.py'),
    ('code/elasticnet_v5_synthetics.py', 'mFC_data/code/elasticnet_v5_synthetics.py'),
]

DEF_PAIRS = [
    ('code/glm_analysis_v2.py', 'mFC_data/code/glm_analysis_v2.py'),
    ('code/glm_analysis_v3.py', 'mFC_data/code/glm_analysis_v3.py'),
]

#: Top-level names that exist only in the mFC copies of glm_analysis_v*: the PFC loader block.
PFC_ONLY = {
    '_PFC_BIN_MS', 'build_data_dic_from_pfc', '_raw_to_norm', '_pfc_session_files',
    '_load_pfc_recday', 'os', 're',
}
#: Top-level names that exist only in the LEC copies (none expected).
LEC_ONLY: set[str] = set()

#: Pre-existing CODE drift between the two frozen v2 copies, found 2026-09-07 when this check
#: was first run. `compute_transition_filter_mask`: the mFC copy folds two statements into
#: one (`seg_locs[finite].astype(int)`), same result. `run_or_load_glm`: the mFC copy never
#: adds `cv_results` to the artefacts it loads/saves when `cross_validate=True` -- a real
#: mirror lag, inert for the production fits (which go through `run_glm_batch.py`). v2 is
#: frozen, so both are allowlisted here and FIXED in the v3 copies, which have no allowlist.
KNOWN_V2_DRIFT = {'compute_transition_filter_mask', 'run_or_load_glm'}

PLOT_PAIR = ('code/glm_plots.py', 'mFC_data/code/pfc_glm_plots.py')
PLOT_START = '#: GridMaze palette.'
PLOT_END = '# LEC only, below this line'

REGISTRY_PAIR = ('code/recday_registry.py', 'mFC_data/code/recday_registry.py')
REGISTRY_NAMES = ('_POST_REFIT_SECTION', 'is_post_refit_section')


def _read(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as fh:
        return fh.read()


def _top_level_defs(src):
    """{name: source} for every top-level def / class / simple assignment."""
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    out[t.id] = ast.get_source_segment(src, node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out[alias.asname or alias.name.split('.')[0]] = ast.get_source_segment(src, node)
    return out


def _norm(s):
    return '\n'.join(line.rstrip() for line in (s or '').splitlines()).strip()


def _strip_docstrings(node):
    """Remove docstring statements from every function/class body, in place."""
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = child.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], 'value', None), ast.Constant) and isinstance(
                    body[0].value.value, str):
                child.body = body[1:] or [ast.Pass()]
    return node


def _code_signature(src):
    """Docstring-free AST dump of a top-level definition: comments and docstrings ignored,
    every statement compared."""
    import textwrap
    tree = ast.parse(textwrap.dedent(src))
    return ast.dump(_strip_docstrings(tree), include_attributes=False)


def _diff(a, b, name):
    return ''.join(difflib.unified_diff(
        _norm(a).splitlines(True), _norm(b).splitlines(True),
        fromfile=f'LEC:{name}', tofile=f'mFC:{name}', n=1))


def check_bytes(pair, problems):
    a, b = pair
    if not (os.path.exists(os.path.join(REPO, a)) and os.path.exists(os.path.join(REPO, b))):
        problems.append(f'{a} / {b}: one copy is missing')
        return
    if _read(a) != _read(b):
        problems.append(f'{a} != {b} (byte identity required)\n'
                        + _diff(_read(a), _read(b), os.path.basename(a))[:3000])


def check_defs(pair, problems, list_names=False, allow=frozenset()):
    a, b = pair
    da, db = _top_level_defs(_read(a)), _top_level_defs(_read(b))
    only_a = set(da) - set(db) - LEC_ONLY
    only_b = set(db) - set(da) - PFC_ONLY
    if only_a:
        problems.append(f'{a}: defined only in the LEC copy: {sorted(only_a)}')
    if only_b:
        problems.append(f'{b}: defined only in the mFC copy: {sorted(only_b)}')
    shared = sorted(set(da) & set(db))
    n_doc_only = 0
    for name in shared:
        same_code = _code_signature(da[name]) == _code_signature(db[name])
        if same_code:
            n_doc_only += _norm(da[name]) != _norm(db[name])
            continue
        if name in allow:
            continue
        problems.append(f'{os.path.basename(a)}: `{name}` CODE differs between the copies\n'
                        + _diff(da[name], db[name], name)[:3000])
    if list_names:
        print(f'  {os.path.basename(a)}: {len(shared)} shared names, '
              f'{n_doc_only} differ only in comments/docstrings'
              + (f', {len(allow)} allowlisted' if allow else ''))


def _plot_block(src, is_lec):
    lines = src.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith(PLOT_START))
    except StopIteration:
        return None
    end = len(lines)
    if is_lec:
        for i in range(start, len(lines)):
            if PLOT_END in lines[i]:
                end = i - 1  # the '# ====' rule above the marker
                break
    return _norm('\n'.join(lines[start:end]))


def check_plots(problems):
    a, b = PLOT_PAIR
    pa, pb = _plot_block(_read(a), True), _plot_block(_read(b), False)
    if pa is None or pb is None:
        problems.append(f'{a} / {b}: plot-block markers not found')
        return
    if pa != pb:
        problems.append(f'shared plot block differs between {a} and {b}\n'
                        + _diff(pa, pb, 'plot block')[:4000])


def check_registry(problems):
    a, b = REGISTRY_PAIR
    da, db = _top_level_defs(_read(a)), _top_level_defs(_read(b))
    for name in REGISTRY_NAMES:
        if name not in da or name not in db:
            problems.append(f'`{name}` missing from {a if name not in da else b}')
        elif _norm(da[name]) != _norm(db[name]):
            problems.append(f'recday_registry `{name}` differs\n' + _diff(da[name], db[name], name))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    list_names = '--list' in argv
    problems = []
    for pair in BYTE_PAIRS:
        check_bytes(pair, problems)
    for pair in DEF_PAIRS:
        allow = KNOWN_V2_DRIFT if pair[0].endswith('_v2.py') else frozenset()
        check_defs(pair, problems, list_names, allow=allow)
    check_plots(problems)
    check_registry(problems)
    if problems:
        print('MIRROR PARITY: FAILED')
        for p in problems:
            print('-' * 78)
            print(p)
        return 1
    byte_names = ', '.join(os.path.basename(a).replace('.py', '') for a, _ in BYTE_PAIRS)
    def_names = '/'.join(os.path.basename(a).replace('glm_analysis_', '').replace('.py', '')
                         for a, _ in DEF_PAIRS)
    print(f'MIRROR PARITY: OK  ({byte_names} byte-identical; glm_analysis_{def_names} '
          'shared definitions have identical code; plot block identical; registry test '
          'identical)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
