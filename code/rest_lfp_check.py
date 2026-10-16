"""Check the spike-based SWS proxy against LFP in LEC rest blocks (M4 part 1; LEC only).

    python code/rest_lfp_check.py --pick                      # the blocks: two per mouse
    sbatch --job-name=rest_lfp_<rd>_sb<j> sbatch_files/event_kernels.sbatch code/rest_lfp_check.py --recday <rd> --block <j>
    python code/rest_lfp_check.py --summarise                 # per block and mouse, with the pre-set rule

Per block: the raw 30 kHz continuous.dat (`run_lfp_anatomy_check.pick_session`'s manifest row);
every 8th channel plus every hippocampal / SUB contact (contact anatomy, current mapping) decimated
to 1250 Hz with `lfp_anatomy_check`'s filter; the median over hippocampal contacts (over all
decimated channels when fewer than 2, as Kanter used CA1 because LEC lacks theta); per 10 s window,
aligned with the spike proxy's (rest bin 0 = sample 0), Welch delta (1-4 Hz) and theta (5-10 Hz)
power. LFP-SWS = theta/delta < 0.8 (Kanter) with delta above the 99th percentile of the same
channels' 10 s windows in the recday's nearest task session (awake), sustained >= 20 s.
2026-10-15: the first version thresholded delta at the block's own median, which labels half of
every block SWS whether or not the mouse slept (38-50% in all 10 blocks, AUC ~0.5 in the pre-task
rests); the threshold is now absolute, calibrated on awake data as the spike proxy is.
Pre-set rule (plan, 2026-10-14): the proxy stages SWS when its AUC for LFP-SWS windows is >= 0.8
in at least 4 of 5 mice; otherwise `down_frac` is only a continuous nuisance.
2026-10-15, afternoon: the rule cannot be applied, because LFP-SWS stays at 0.00-0.02 of windows
in all 10 blocks. theta/delta < 0.8 holds in 82-100% of every block's windows, so the ratio
selects nothing here, and rest delta sits at a median 0-20% of the awake threshold, which task
movement and licking inflate. User decision: continuous state terms only. The readout is now
the Spearman of `down_frac` with log delta per block, by block kind; the binary columns stay in
the CSVs as the superseded rule.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import rest_kernels as rk                                              # noqa: E402

OUT = os.path.join(ek.REPO, 'data', 'event_kernels', 'rest', 'lfp_check')
FS_OUT = 1250.0
DELTA, THETA = (1.0, 4.0), (5.0, 10.0)
RATIO_SWS = 0.8
AUC_RULE, MICE_RULE = 0.8, 4
DELTA_PCT = 99.0
TASK_MAX_S = 600.0                             # the first 10 min of the task session: the awake reference


def pick_blocks() -> pd.DataFrame:
    """Two blocks per mouse from its first recday: the longest pre-task rest and the between-task
    rest of median length; blocks the signal-loss QC truncates are skipped."""
    rows = []
    for p in sorted(glob.glob(os.path.join(ek.LEC_FULL, 'recording_index_*.csv'))):
        rd = os.path.basename(p)[len('recording_index_'):-4].split('.v')[0]
        rows.append((rd.split('_')[0], rd, p))
    out = []
    for mouse in sorted({m for m, _, _ in rows}):
        for _, rd, p in sorted(r for r in rows if r[0] == mouse):
            idx = pd.read_csv(p)
            idx = idx[(idx['kind'] == 'rest') & ~idx['misclick'].astype(bool)]
            rec = ek.load_recday(rd, 'lec', with_rest=True)
            blocks, qc = rk.rest_blocks(rec)
            ok = set(qc.loc[qc['kept'] & (qc['reason'] == ''), 'index'])
            pre = idx[(idx['stage'] == 'pre') & idx['index'].isin(ok)].sort_values('duration_s')
            post = idx[(idx['stage'] == 'post') & idx['index'].isin(ok)].sort_values('duration_s')
            if len(pre) and len(post):
                out.append(dict(mouse=mouse, recday=rd, block=int(pre['index'].iloc[-1]), kind='pre'))
                out.append(dict(mouse=mouse, recday=rd, block=int(post['index'].iloc[len(post) // 2]), kind='post'))
                break
    return pd.DataFrame(out)


def decimate_cols(rec, cols, fs_out=FS_OUT, chunk_s=60.0, pad_s=2.0, max_s=None):
    """(len(cols), n_out) µV at fs_out from sample 0, as `lfp_anatomy_check.decimate_lfp` but for a
    subset of columns (the read is the same; the filtering is 8x smaller)."""
    q = int(round(rec.fs / fs_out))
    n_out = rec.n_samples // q if max_s is None else min(rec.n_samples, int(max_s * rec.fs)) // q
    out = np.empty((len(cols), n_out), np.float32)
    chunk, pad = int(chunk_s * rec.fs) // q * q, int(pad_s * rec.fs) // q * q
    for c0 in range(0, n_out * q, chunk):
        c1 = min(c0 + chunk, n_out * q)
        a, b = max(c0 - pad, 0), min(c1 + pad, rec.n_samples)
        raw = np.asarray(rec.data[a:b])[:, cols].astype(np.float32).T * rec.uv
        y = signal.resample_poly(raw, 1, q, axis=1)
        i0 = (c0 - a) // q
        out[:, c0 // q:c1 // q] = y[:, i0:i0 + (c1 - c0) // q]
    return out


def band_powers(x: np.ndarray, fs: float, n_win: int) -> tuple:
    """Per 10 s window of a 1-D signal: Welch delta and theta power."""
    w = int(round(10.0 * fs))
    d, t = np.full(n_win, np.nan), np.full(n_win, np.nan)
    for k in range(min(n_win, len(x) // w)):
        f, P = signal.welch(x[k * w:(k + 1) * w], fs=fs, nperseg=int(2.5 * fs))
        d[k] = P[(f >= DELTA[0]) & (f < DELTA[1])].mean()
        t[k] = P[(f >= THETA[0]) & (f < THETA[1])].mean()
    return d, t


def run_block(recday: str, block: int) -> pd.DataFrame:
    import lfp_anatomy_check as lac
    import recday_registry as rr
    import run_lfp_anatomy_check as rlac
    mouse = recday.split('_')[0]
    manifest = pd.read_csv(rr.kilosort_dir(recday).parent / 'recording_sessions_in_concat.csv')
    _, _, _, concat_row = rlac.pick_session(recday, f'sb{block}', lac.OUT_DIR)
    rec = lac.RawRecording(manifest.iloc[concat_row]['continuous_path'], expected_serial=lac.SERIALS[mouse])
    ct = rec.channel_table()
    lab, tis = rlac.anatomy_by_mouse(lac.OUT_DIR)[mouse].labels(ct['x'].values, ct['y'].values, 'current')
    hpf = np.array([v is True for v in lac.hpf_class(lab, tis)])
    pick = sorted(set(range(0, len(ct), 8)) | set(np.flatnonzero(hpf)))
    lfp = decimate_cols(rec, ct['col_in_dat'].values[pick])
    hp = [i for i, c in enumerate(pick) if hpf[c]]
    refsig = (lambda x: np.median(x[hp], axis=0)) if len(hp) >= 2 else (lambda x: np.median(x, axis=0))
    ref = refsig(lfp)
    # the spike proxy of this block, calibrated on the recday's task sessions as in the runner
    sp = ek.load_recday(recday, 'lec', with_rest=True)
    ek.add_day_clock(sp)
    blocks, _ = rk.rest_blocks(sp)
    b = next(x for x in blocks if x['index'] == block)
    S = ek.analysis_sessions(sp, 5)
    P = rk.sws_proxy([b['counts']], [s['counts'] for s in S])
    df, sws = P['blocks'][0]['down_frac'], P['blocks'][0]['sws']
    # the awake reference: the same channels in the task session nearest in time
    near = min(S, key=lambda s: abs((s['wall_bin0'] - b['wall_bin0']).total_seconds()))
    _, _, _, row_t = rlac.pick_session(recday, f"s{near['session']}", lac.OUT_DIR)
    rec_t = lac.RawRecording(manifest.iloc[row_t]['continuous_path'], expected_serial=lac.SERIALS[mouse])
    lfp_t = decimate_cols(rec_t, rec_t.channel_table()['col_in_dat'].values[pick], max_s=TASK_MAX_S)
    d_awake, _ = band_powers(refsig(lfp_t), FS_OUT, int(TASK_MAX_S // 10))
    d_thr = float(np.nanpercentile(d_awake, DELTA_PCT))
    d, t = band_powers(ref, FS_OUT, len(df))
    dall, tall = band_powers(np.median(lfp, axis=0), FS_OUT, len(df))
    ratio = t / d
    cand = (ratio < RATIO_SWS) & (d > d_thr)
    lfp_sws = np.zeros(len(df), bool)
    for a, e in rk.runs(cand, rk.SWS_MIN_WIN):
        lfp_sws[a:e] = True
    out = pd.DataFrame(dict(recday=recday, block=block, window=np.arange(len(df)), t_s=np.arange(len(df)) * 10.0,
                            down_frac=df, sws=sws, delta=d, theta=t, ratio=ratio, lfp_sws=lfp_sws,
                            delta_all=dall, theta_all=tall, n_hpf_channels=len(hp), n_channels=len(pick),
                            theta_thr=P['theta'], sws_cut=P['cut'], delta_thr_awake=d_thr, awake_session=near['session']))
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, f'{recday}_sb{block}.csv'), index=False)
    return out


def _auc(score, pos):
    pos = np.asarray(pos, bool)
    if not pos.any() or pos.all():
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def _stage(recday: str, block: int) -> str:
    """'pre' (before the day's first task) or 'post', from the recday's recording index."""
    p = sorted(glob.glob(os.path.join(ek.LEC_FULL, f'recording_index_{recday}*.csv')))[-1]
    idx = pd.read_csv(p)
    return str(idx.loc[idx['index'] == block, 'stage'].iloc[0])


def summarise() -> pd.DataFrame:
    """Per block: the Spearman of `down_frac` with log delta (the readout), and the superseded
    binary-rule columns; per mouse and block kind in `attrs['readout']`."""
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, '*_sb*.csv'))):
        t = pd.read_csv(p).dropna(subset=['delta'])
        rd, b = t['recday'].iloc[0], int(t['block'].iloc[0])
        ok = t['down_frac'].nunique() > 1 and len(t) > 3
        rho = spearmanr(t['down_frac'], np.log(t['delta'])).correlation if ok else np.nan
        rows.append(dict(recday=rd, mouse=rd.split('_')[0], block=b, kind=_stage(rd, b), windows=len(t),
                         spearman_log_delta=float(rho), down_frac_nonzero=float((t['down_frac'] > 0).mean()),
                         hpf_channels=int(t['n_hpf_channels'].iloc[0]),
                         superseded_lfp_sws_frac=float(t['lfp_sws'].mean()), superseded_proxy_sws_frac=float(t['sws'].mean()),
                         superseded_auc=_auc(t['down_frac'], t['lfp_sws'])))
    S = pd.DataFrame(rows)
    if len(S):
        S.attrs['readout'] = S.pivot_table(index='mouse', columns='kind', values='spearman_log_delta').round(2).to_dict()
    return S


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recday')
    ap.add_argument('--block', type=int)
    ap.add_argument('--pick', action='store_true')
    ap.add_argument('--summarise', action='store_true')
    args = ap.parse_args(argv)
    if args.pick:
        print(pick_blocks().to_string(index=False))
        return 0
    if args.summarise:
        S = summarise()
        S.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
        print(S.round(3).to_string(index=False))
        print('Spearman of down_frac with log delta, per mouse and block kind:', S.attrs.get('readout'))
        return 0
    t = run_block(args.recday, args.block)
    print(f"{args.recday} sb{args.block}: {len(t)} windows, LFP-SWS {t['lfp_sws'].mean():.2f}, proxy SWS {t['sws'].mean():.2f}, "
          f"AUC {_auc(t['down_frac'], t['lfp_sws']):.3f}, hippocampal channels {int(t['n_hpf_channels'].iloc[0])}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
