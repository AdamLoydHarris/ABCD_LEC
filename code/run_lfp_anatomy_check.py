"""LFP anatomy check (lfp_anatomy_check.py), one stage per call, under sbatch.

    python run_lfp_anatomy_check.py --stage anatomy  [--mice ah08,ly07]          (histology env)
    python run_lfp_anatomy_check.py --stage probemap [--mice ...] [--configs config_1,...]
    python run_lfp_anatomy_check.py --stage speed                                (loads data_dic v4)
    python run_lfp_anatomy_check.py --stage units    [--recdays ...] [--workers 8]
    python run_lfp_anatomy_check.py --stage raw --recday R --session s3|sb0|auto-task|auto-sleep
    python run_lfp_anatomy_check.py --stage merge

Order: anatomy -> probemap; speed -> units; speed -> raw (x4) -> merge.
Outputs in data/processed_data/lfp_anatomy_check/ (--out-dir overrides). Each stage writes its own
gates json and ends with EXIT=; a downstream stage refuses an upstream product whose gates failed.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / 'preprocessing'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import lfp_anatomy_check as lac                                    # noqa: E402

V4_NEURON = lac.REPO / 'data' / 'processed_data' / 'neuron_raw_v4'
V4_RSYNC = lac.REPO / 'data' / 'processed_data' / 'ephys_rsync_v4'
V4_QC = lac.REPO / 'data' / 'processed_data' / 'ephys_rsync_v4_qc.csv'


def _enr():
    """extract_neuron_raw on the v4 clock (it reads --rsync from sys.argv at import, so set it here)."""
    import extract_neuron_raw as enr
    enr.RSYNC_PATH = V4_RSYNC
    return enr


def _save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)


def _load(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def _require(gates_path):
    g = json.loads(Path(gates_path).read_text())
    if not g.get('pass', False):
        raise RuntimeError(f'{gates_path}: upstream gates failed; refusing to continue')
    return g


def anatomy_by_mouse(out):
    t = pd.read_csv(out / 'contact_anatomy.csv')
    return {m: lac.Anatomy(d) for m, d in t.groupby('mouse')}


# ---------------------------------------------------------------------------
# Stage: anatomy
# ---------------------------------------------------------------------------

def stage_anatomy(args, out):
    tables, gates = [], {}
    for mouse in args.mice:
        t0 = time.time()
        t, g = lac.build_contact_anatomy(mouse)
        tables.append(t)
        gates[mouse] = g
        print(f'{mouse}: {len(t)} contacts, {time.time() - t0:.0f} s, gates {g}')
    tab = pd.concat(tables, ignore_index=True)
    # bank-0 current labels must equal channel_regions.csv, which unit_regions was built from
    cr = pd.read_csv(lac.REPO / 'data' / 'processed_data' / 'channel_regions.csv')
    cr = cr.drop_duplicates(['mouse', 'x_um', 'y_um'])
    j = cr.merge(tab, left_on=['mouse', 'x_um', 'y_um'], right_on=['mouse', 'x', 'y'], how='left')
    agree = (j['acronym'].astype(str) == j['label_current'].astype(str))
    gates['channel_regions_agree'] = float(agree.mean())
    gates['channel_regions_n'] = int(len(j))
    ok = [gates[m]['current_vs_htsv'] == 1.0 and gates[m]['mirror_position_dev_vox_max'] < 0.5
          and gates[m]['mirror_label_vs_twin_lookup'] >= 0.99 for m in args.mice]
    gates['pass'] = bool(all(ok) and gates['channel_regions_agree'] == 1.0)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / 'contact_anatomy.csv', index=False)
    (out / 'anatomy_gates.json').write_text(json.dumps(gates, indent=1, default=float))
    print(f'channel_regions agreement {gates["channel_regions_agree"]:.4f} over {len(j)} channels')
    return 0 if gates['pass'] else 1


# ---------------------------------------------------------------------------
# Stage: probemap
# ---------------------------------------------------------------------------

def stage_probemap(args, out, cfg):
    _require(out / 'anatomy_gates.json')
    anat = anatomy_by_mouse(out)
    chans, bips, psds, metas, gate_rows = [], [], {}, {}, []
    for mouse in args.mice:
        configs = args.configs or sorted(p.name for p in
                                         (lac.PROBEMAP_ROOT / 'processed_data' / mouse).glob('config_*'))
        for config in configs:
            t0 = time.time()
            ch, bip, psd, meta = lac.analyse_probemap_config(mouse, config, cfg)
            g = meta.pop('gate_a0')
            gate_rows.append(g.assign(mouse=mouse, config=config))
            ch['gate_ok'] = meta['gate_a0_pass']
            chans.append(ch)
            bips.append(bip)
            psds[(mouse, config)] = psd
            metas[(mouse, config)] = meta
            print(f'{mouse} {config}: gate A0 {"PASS" if meta["gate_a0_pass"] else "FAIL"} '
                  f'(median r_own {g.r_own.median():.3f}, worst {g.r_own.min():.3f}, '
                  f'{int(g.ok.sum())}/{len(g)} columns, lag {int(g.lag_samples.iloc[0])}); '
                  f'metrics x/y agree {meta["metrics_xy_agree"]:.2f}; ripples {meta["n_events"]} '
                  f'on ch{int(ch.ch.iloc[meta["ref"]])} (x {ch.x.iloc[meta["ref"]]:.0f}, '
                  f'y {ch.y.iloc[meta["ref"]]:.0f}); {time.time() - t0:.0f} s')
    chans = pd.concat(chans, ignore_index=True)
    bips = pd.concat(bips, ignore_index=True)
    gate_df = pd.concat(gate_rows, ignore_index=True)
    for mapping in ('current', 'mirror'):
        chans[f'label_{mapping}'] = None
        chans[f'tissue_{mapping}'] = None
        for mouse, d in chans.groupby('mouse'):
            lab, tis = anat[mouse].labels(d.x.values, d.y.values, mapping)
            chans.loc[d.index, f'label_{mapping}'] = lab
            chans.loc[d.index, f'tissue_{mapping}'] = tis
    orient, theta = [], []
    for mouse, d in chans.groupby('mouse'):
        keep = d['gate_ok'].values & ~d['is_ref'].values
        for measure in ('ripple_gain', 'rest_ripple_excess_db', 'ripple_rate_hz'):
            o = lac.orientation_test(d.x.values, d.y.values, d[measure].values, anat[mouse], cfg, keep)
            o.pop('perm_aucs')
            orient.append({'mouse': mouse, 'measure': measure, **o})
        bank = d['gate_ok'].values & (d.y.values <= 705)
        for measure in ('th_theta_excess_db', 'th_rel_theta'):
            theta.append({'mouse': mouse, 'derivation': 'monopolar', 'measure': measure,
                          **lac.ec_theta_test(d.x.values, d.y.values, d[measure].values, anat[mouse], keep=bank)})
        b = bips[bips.mouse == mouse]
        okc = chans.loc[chans.mouse == mouse].groupby('config')['gate_ok'].first()
        bb = b['config'].map(okc).fillna(False).values.astype(bool) & (b.y.values <= 705)
        theta.append({'mouse': mouse, 'derivation': 'bipolar', 'measure': 'th_theta_excess_db',
                      **lac.ec_theta_test(b.x.values, b.y.values, b['th_theta_excess_db'].values,
                                          anat[mouse], keep=bb)})
    orient = pd.DataFrame(orient)
    theta = pd.DataFrame(theta)
    gates = {'a0_all_pass': bool(gate_df['ok'].all()),
             'a0_configs_failed': sorted({f'{m} {c}' for m, c, ok in
                                          zip(gate_df.mouse, gate_df.config, gate_df.ok) if not ok}),
             'a0_median_r_own': float(gate_df.r_own.median())}
    gates['pass'] = gates['a0_all_pass']
    _save({'channels': chans, 'bipolar': bips, 'psd': psds, 'meta': metas, 'gate_a0': gate_df,
           'orientation': orient, 'theta': theta, 'config': asdict(cfg)}, out / f'probemap_{args.tag}.pkl')
    chans.to_csv(out / f'probemap_{args.tag}_channels.csv', index=False)
    (out / f'probemap_{args.tag}_gates.json').write_text(json.dumps(gates, indent=1))
    print(orient[['mouse', 'measure', 'auc_current', 'auc_mirror', 'rank_current', 'rank_reversed',
                  'n_pos_current', 'n_pos_mirror', 'verdict']].to_string(index=False))
    print(theta.filter(regex='mouse|derivation|measure|auc_').to_string(index=False))
    return 0 if gates['pass'] else 1


# ---------------------------------------------------------------------------
# Stage: speed
# ---------------------------------------------------------------------------

def stage_speed(args, out):
    """Speed per 25 ms bin on the v4 clock.

    Reads the tracking v3 arrays that data_dic_lec_v4 was built from (SPIKE_CLOCK_V4.md:
    build_data_dic --neural v4 --tracking v3 --tag v4) and the bin count from each neuron_raw_v4
    header, so the 3.8 GB pickle is never loaded for 200 small arrays.
    """
    import recday_registry as rr
    from tracking_impact import speed_glm
    enr = _enr()
    ur = _load(lac.REPO / 'data' / 'processed_data' / 'unit_regions.pkl')
    speed, rows, bad, skipped = {}, [], [], []
    for recday in (args.recdays or sorted(ur)):
        mouse, dates = rr.recday_dates(recday)
        tdir = lac.REPO / 'data' / 'processed_v3' / mouse / '_'.join(dates)
        wake, _ = enr.session_map(recday)
        speed[recday] = {}
        for s in range(len(wake)):
            kind = rr.session_type(recday, s)
            if kind in ('object', 'openfield'):
                skipped.append(f'{recday} s{s}: {kind}')
                continue                       # not the maze task; no theta/ripple use here
            npath = V4_NEURON / f'Neuron_raw_{recday}_{s}.npy'
            xpath = tdir / f'XY_raw_{recday}_{s}.npy'
            if not npath.exists() or not xpath.exists():
                bad.append(f'{recday} s{s}: missing {"neuron" if not npath.exists() else "XY"} array')
                continue
            n_bins = int(np.load(npath, mmap_mode='r').shape[1])
            try:
                clock, t0, nb = enr.session_window_fitted(recday, s, wake[s])
            except Exception as exc:           # noqa: BLE001 - recorded in the gates, not hidden
                bad.append(f'{recday} s{s}: clock {type(exc).__name__}: {exc}')
                continue
            if nb != n_bins:
                bad.append(f'{recday} s{s}: clock n_bins {nb} != Neuron_raw {n_bins}')
            xy = np.asarray(np.load(xpath, allow_pickle=True), float)[:n_bins]
            v = speed_glm(xy).astype(np.float32)
            if len(v) < n_bins:
                v = np.concatenate([v, np.full(n_bins - len(v), np.nan, np.float32)])
            speed[recday][s] = {'speed': v, 't0_ms': int(t0), 'a': clock['a'], 'b': clock['b'],
                                'concat_row': int(wake[s]), 'n_bins': n_bins}
            rows.append({'recday': recday, 'session': s, 'session_type': kind, 'n_bins': n_bins,
                         'run_s': float(np.sum(v > 5.0) * 0.025), 'still_s': float(np.sum(v < 2.0) * 0.025),
                         'tracked_frac': float(np.isfinite(v).mean())})
    _save(speed, out / 'speed_v4.pkl')
    pd.DataFrame(rows).to_csv(out / 'session_running.csv', index=False)
    gates = {'n_sessions': len(rows), 'n_recdays': len(speed), 'problems': bad, 'skipped': skipped,
             'pass': not bad}
    (out / 'speed_gates.json').write_text(json.dumps(gates, indent=1))
    print(f'{len(rows)} sessions over {len(speed)} recdays; {len(skipped)} object/open-field skipped; '
          f'{len(bad)} problems')
    for b in bad:
        print('  ', b)
    return 0 if gates['pass'] else 1


# ---------------------------------------------------------------------------
# Stage: units (spike-only theta rhythmicity, every unit-recording)
# ---------------------------------------------------------------------------

def _session_spikes(recday, concat_row, cache):
    """(.dat sample, QC row) for one recording of a sorted block; cache holds the block's arrays."""
    import recday_registry as rr
    enr = _enr()
    if cache.get('recday') != recday:
        ks = rr.kilosort_dir(recday)
        qc = np.load(ks / 'QC_single_units.npy')
        st = np.load(ks / 'sorter_output/spike_times.npy').squeeze().astype(np.int64)
        sc = np.load(ks / 'sorter_output/spike_clusters.npy').squeeze()
        lut = np.full(int(sc.max()) + 2, -1, dtype=np.int64)
        lut[qc] = np.arange(len(qc))
        cache.clear()
        cache.update(recday=recday, st=st, rows=lut[sc], n_units=len(qc),
                     table=enr.concat_table(recday))
    row = cache['table'].iloc[concat_row]
    s0, n = int(row['start']), int(row['n_samples'])
    m = (cache['st'] >= s0) & (cache['st'] < s0 + n)
    return cache['st'][m] - s0, cache['rows'][m], cache['n_units']


def _theta_job(args):
    t, epoch, cfgd, seed = args
    return lac.unit_theta_rhythmicity(t, epoch, lac.LFPConfig(**cfgd), np.random.default_rng(seed))


def unit_positions(recday, ur):
    """unit_regions rows with x from the block's channel_positions.npy."""
    import recday_registry as rr
    pos = np.load(rr.kilosort_dir(recday) / 'sorter_output' / 'channel_positions.npy')
    u = ur.copy()
    ok = u['channel_row'].notna()
    u['x'] = np.nan
    u.loc[ok, 'x'] = pos[u.loc[ok, 'channel_row'].astype(int), 0]
    return u


def label_units(u, anat):
    for mapping in ('current', 'mirror'):
        lab, tis = anat.labels(u['x'].values, u['y_um'].values, mapping)
        u[f'label_{mapping}'] = lab
        u[f'tissue_{mapping}'] = tis
        u[f'group_{mapping}'] = [lac.unit_group(a, b) for a, b in zip(lab, tis)]
        u[f'd_signed_{mapping}'] = anat.signed_distance(u['x'].values, u['y_um'].values, mapping)
    return u


def stage_units(args, out, cfg):
    from concurrent.futures import ProcessPoolExecutor
    enr = _enr()
    _require(out / 'anatomy_gates.json')
    _require(out / 'speed_gates.json')
    anat = anatomy_by_mouse(out)
    speed = _load(out / 'speed_v4.pkl')
    ur_all = _load(lac.REPO / 'data' / 'processed_data' / 'unit_regions.pkl')
    recdays = args.recdays or sorted(r for r in speed if r.split('_')[0] in args.mice)
    rt_rows, units, jobs, cache = [], [], [], {}
    for recday in recdays:
        mouse = recday.split('_')[0]
        u = label_units(unit_positions(recday, ur_all[recday]), anat[mouse])
        u['recday'] = recday
        resolved = u['x'].notna()
        u['label_agree'] = np.where(resolved, u['acronym'].astype(str) == u['label_current'].astype(str), np.nan)
        t_all = [[] for _ in range(len(u))]
        e_all = [[] for _ in range(len(u))]
        run_s = 0.0
        for s, sp in sorted(speed[recday].items()):
            samples, rows, n_units = _session_spikes(recday, sp['concat_row'], cache)
            if n_units != len(u):
                raise ValueError(f'{recday}: {n_units} QC units vs {len(u)} unit_regions rows')
            clock = {'a': sp['a'], 'b': sp['b']}
            # round trip: these spikes must rebuild the v4 array exactly
            counts = enr.bin_session_fitted(samples, rows, n_units, clock, sp['t0_ms'], sp['n_bins'])
            stored = np.load(V4_NEURON / f'Neuron_raw_{recday}_{s}.npy')
            diff = int(np.abs(counts.astype(np.int64) - stored.astype(np.int64)).max()) \
                if counts.shape == stored.shape else -1
            rt_rows.append({'recday': recday, 'session': s, 'max_abs_diff': diff})
            trains, rs = lac.running_spike_trains(samples, rows, n_units, sp['speed'], clock,
                                                  sp['t0_ms'], s, cfg)
            run_s += rs
            for r, (t, e) in enumerate(trains):
                t_all[r].append(t)
                e_all[r].append(e)
        u['run_s'] = run_s
        for r in range(len(u)):
            t = np.concatenate(t_all[r]) if t_all[r] else np.zeros(0)
            e = np.concatenate(e_all[r]) if e_all[r] else np.zeros(0, np.int64)
            jobs.append((t, e, asdict(cfg), cfg.seed + len(jobs)))
        units.append(u)
        print(f'{recday}: {len(u)} units ({int(resolved.sum())} resolved), run {run_s / 60:.1f} min, '
              f'round-trip max|diff| {max(r["max_abs_diff"] for r in rt_rows if r["recday"] == recday)}, '
              f'labels agree {np.nanmean(u["label_agree"].astype(float)):.3f}')
    units = pd.concat(units, ignore_index=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_theta_job, jobs, chunksize=4))
    print(f'theta index for {len(res)} unit-recordings in {(time.time() - t0) / 60:.1f} min')
    res = pd.DataFrame(res)
    units = pd.concat([units.reset_index(drop=True), res], axis=1)
    units['rate_run_hz'] = units['n_spikes'] / units['run_s'].replace(0, np.nan)
    units['theta_rhythmic'] = np.where(units['p_jitter'].notna(), units['p_jitter'] < 0.05, np.nan)
    rt = pd.DataFrame(rt_rows)
    contrasts = []
    for mapping in ('current', 'mirror'):
        d = units.assign(grp=units[f'group_{mapping}'],
                         l5=np.where(units[f'label_{mapping}'].astype(str) == 'ENTm5', 'ENTm5',
                                     np.where(units[f'label_{mapping}'].astype(str) == 'ENTl5', 'ENTl5', 'x')))
        for a, b, col in (('ENTm', 'ENTl-deep', 'grp'), ('ENTm', 'ENTl-sup', 'grp'),
                          ('ENTm5', 'ENTl5', 'l5'), ('HPF', 'ENTl-deep', 'grp')):
            for value in ('theta_excess_db', 'theta_ratio', 'burst_frac'):
                c = lac.shuffle_contrast(d, value, col, a, b, n_shuffle=args.n_shuffle, seed=cfg.seed)
                contrasts.append(c.assign(mapping=mapping))
    contrasts = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()
    frac = (units.groupby(['mouse', 'group_current'])
            .agg(n=('theta_excess_db', lambda v: int(np.isfinite(v).sum())),
                 frac_rhythmic=('theta_rhythmic', 'mean'), median_excess_db=('theta_excess_db', 'median'),
                 median_ratio=('theta_ratio', 'median'), median_burst_frac=('burst_frac', 'median'))
            .reset_index())
    gates = {'round_trip_max_abs_diff': int(rt['max_abs_diff'].max()),
             'round_trip_bad_sessions': rt.loc[rt.max_abs_diff != 0, ['recday', 'session']].values.tolist(),
             'unit_label_agree': float(np.nanmean(units['label_agree'].astype(float)))}
    gates['pass'] = gates['round_trip_max_abs_diff'] == 0 and gates['unit_label_agree'] == 1.0
    _save({'units': units, 'round_trip': rt, 'contrasts': contrasts, 'fractions': frac,
           'config': asdict(cfg)}, out / f'units_theta_{args.tag}.pkl')
    units.to_csv(out / f'units_theta_{args.tag}.csv', index=False)
    (out / f'units_{args.tag}_gates.json').write_text(json.dumps(gates, indent=1, default=str))
    print(frac.to_string(index=False))
    print(contrasts.to_string(index=False))
    print(f'gates: {gates}')
    return 0 if gates['pass'] else 1


# ---------------------------------------------------------------------------
# Stage: raw (one recording)
# ---------------------------------------------------------------------------

def pick_session(recday, session, out):
    enr = _enr()
    wake, sleep = enr.session_map(recday)
    table = enr.concat_table(recday)
    if session == 'auto-task':
        run = pd.read_csv(out / 'session_running.csv')
        run = run[(run.recday == recday) & ~run.session_type.isin(['object', 'openfield'])]
        r = run.sort_values('run_s', ascending=False).iloc[0]
        session = f's{int(r.session)}'
    elif session == 'auto-sleep':
        j = int(np.argmax(table['n_samples'].values[sleep]))
        session = f'sb{j}'
    if session.startswith('sb'):
        j = int(session[2:])
        return session, 'sleep', j, sleep[j]
    s = int(session[1:])
    return session, 'task', s, wake[s]


def stage_raw(args, out, cfg):
    import recday_registry as rr
    enr = _enr()
    _require(out / 'anatomy_gates.json')
    recday = args.recday
    mouse = recday.split('_')[0]
    session, kind, idx, concat_row = pick_session(recday, args.session, out)
    ks = rr.kilosort_dir(recday)
    csv = pd.read_csv(ks.parent / 'recording_sessions_in_concat.csv')
    row = csv.iloc[concat_row]
    if str(row['probe_serial_number']) != lac.SERIALS[mouse]:
        raise ValueError(f'{recday} {session}: manifest serial {row["probe_serial_number"]} is not {mouse}')
    rec = lac.RawRecording(row['continuous_path'], expected_serial=lac.SERIALS[mouse])
    folder = Path(row['continuous_path']).parents[5].name
    qc = pd.read_csv(V4_QC)
    hit = qc[(qc.mouse == mouse) & (qc.recording == folder)]
    restart = int(hit['restart_at'].iloc[0]) if len(hit) else 0
    print(f'{recday} {session} ({kind}): {folder}, {rec.n_samples / rec.fs / 60:.1f} min, '
          f'restart_at {restart}, serial {rec.geom.attrs["serial"]}')
    gates = {'recday': recday, 'session': session, 'kind': kind, 'folder': folder}

    # geometry: probeinterface per raw channel vs the sort's channel_positions, by channel name
    ct = rec.channel_table().set_index('name')
    pos_sort = np.load(ks / 'sorter_output' / 'channel_positions.npy')
    ids = json.loads((ks / 'sorting_analyzer/recording_info/recording_attributes.json').read_text())['channel_ids']
    gx = ct.loc[ids, ['x', 'y']].values
    gates['geometry_vs_sort_max_um'] = float(np.abs(gx - pos_sort).max())

    t0 = time.time()
    q = int(round(rec.fs / cfg.fs_lfp))
    base = f'raw_{recday}_{session}'
    cache = out / 'lfp_cache' / f'{base}_lfp1250_float16.npy'
    if args.from_cache:
        # re-analysis without re-reading 30-60 GB: the cache is this stage's own float16 output
        lfp = np.asarray(np.load(cache), np.float32)
        lfp_start = restart + (-restart) % q
        print(f'loaded cached LFP {lfp.shape} in {(time.time() - t0) / 60:.1f} min')
    else:
        stop = rec.n_samples if not args.max_minutes else min(rec.n_samples, restart + int(args.max_minutes * 60 * rec.fs))
        lfp, lfp_start = lac.decimate_lfp(rec, restart, stop, cfg.fs_lfp)
        print(f'decimated {lfp.shape} in {(time.time() - t0) / 60:.1f} min')
    r_dec = lac.verify_decimation(rec, lfp, lfp_start, cfg.fs_lfp)
    gates['decimation_r_median'] = float(np.median(r_dec))
    gates['decimation_r_min'] = float(np.min(r_dec))
    pos = pd.DataFrame({'name': rec.names})
    pos['ch'] = pos['name'].str[2:].astype(int)
    pos = pos.merge(rec.geom[['ch', 'x', 'y', 'shank']], on='ch', how='left')
    fs = float(cfg.fs_lfp)
    starts, w = lac.window_grid(lfp.shape[1], fs, cfg.window_s)

    sp = None
    if kind == 'task':
        sp = _load(out / 'speed_v4.pkl')[recday][idx]
        clock = {'a': sp['a'], 'b': sp['b']}
        run_win, still_win, still_mask = lac.speed_windows(sp['speed'], clock, sp['t0_ms'], lfp_start,
                                                           rec.fs, fs, w, cfg, lfp.shape[1])
        print(f'running windows {len(run_win)}, still windows {len(still_win)}')
        ch, bip, psd, meta = lac.analyse_lfp_block(lfp, fs, pos, cfg, theta_win=run_win,
                                                   rest_win=still_win, rest_mask=still_mask)
        rest_mask = still_mask
        # clock gate: probe-median theta power against speed, per 25 ms bin then 0.5 s
        xmed = lac.probe_median(lfp)
        nb = sp['n_bins']
        s_bin = lac.bin_to_sample(np.arange(nb + 1), clock, sp['t0_ms'])
        j_bin = np.clip(((s_bin - lfp_start) / q).astype(np.int64), 0, lfp.shape[1])
        k = 20
        m = nb // k
        spd = np.nanmean(sp['speed'][:m * k].reshape(m, k), axis=1)

        def binned_log_power(band):
            env = np.abs(signal.hilbert(lac.bandpass(xmed, fs, band))) ** 2
            csum = np.concatenate([[0], np.cumsum(env)])
            pw = (csum[j_bin[1:]] - csum[j_bin[:-1]]) / np.maximum(j_bin[1:] - j_bin[:-1], 1)
            pw[(j_bin[1:] - j_bin[:-1]) <= 0] = np.nan
            return np.log10(np.nanmean(pw[:m * k].reshape(m, k), axis=1) + 1e-9)
        # gate (post-hoc 2026-10-02): theta-band power does not track speed in this task (ly06 s4
        # Spearman -0.14: ~7 Hz licking at ports, weak theta); 60-100 Hz power does (+0.44 at 0 s)
        lag, rmax, lags, rcurve = lac.clock_lag(binned_log_power((60.0, 100.0)), spd, 0.5, rank=True)
        gates['clock_lag_s'], gates['clock_r'] = lag, rmax
        meta['clock_curve'] = (lags, rcurve)
        tl, tr, _, _ = lac.clock_lag(binned_log_power(cfg.theta_band), spd, 0.5, rank=True)
        gates['theta_speed_lag_s'], gates['theta_speed_rho'] = tl, tr
        th_rc = lac.clock_lag(binned_log_power(cfg.theta_band), spd, 0.5, rank=True)
        gates['theta_speed_rho_at_0'] = float(th_rc[3][np.argmin(np.abs(th_rc[2]))])
    else:
        ch, bip, psd, meta = lac.analyse_lfp_block(lfp, fs, pos, cfg)
        rest_mask = lac.mask_from_windows(meta['rest_win'], w, lfp.shape[1])
        run_win = meta['theta_win']
    for c in ('name', 'ch', 'shank'):
        ch[c] = pos[c].values
    anat = anatomy_by_mouse(out)[mouse]
    for mapping in ('current', 'mirror'):
        lab, tis = anat.labels(ch.x.values, ch.y.values, mapping)
        ch[f'label_{mapping}'], ch[f'tissue_{mapping}'] = lab, tis

    # units: round trip, then phase locking (task) or ripple modulation (both)
    samples, rows, n_units = _session_spikes(recday, concat_row, {})
    if kind == 'task':
        counts = enr.bin_session_fitted(samples, rows, n_units, clock, sp['t0_ms'], sp['n_bins'])
        stored = np.load(V4_NEURON / f'Neuron_raw_{recday}_{idx}.npy')
    else:
        counts = enr.bin_session(samples, rows, n_units, 0, rec.n_samples // 750)
        stored = np.load(V4_NEURON / f'Neuron_raw_{recday}_sb_{idx}.npy')
    gates['round_trip_max_abs_diff'] = int(np.abs(counts.astype(np.int64) - stored.astype(np.int64)).max()) \
        if counts.shape == stored.shape else -1
    ur = _load(lac.REPO / 'data' / 'processed_data' / 'unit_regions.pkl')[recday]
    u = label_units(unit_positions(recday, ur), anat)
    peaks = meta['events'][:, 1] if len(meta['events']) else np.zeros(0, int)
    um = lac.unit_lfp_measures(samples, rows, n_units, list(u['max_channel']), lfp, lfp_start, q, fs,
                               pos, run_win, w, rest_mask, peaks, ch, cfg)
    u = pd.concat([u.reset_index(drop=True), um.drop(columns='order')], axis=1)
    u['recday'], u['session'] = recday, session

    orient = []
    keep = ~ch['is_ref'].values
    for measure in ('ripple_gain', 'rest_ripple_excess_db', 'ripple_rate_hz'):
        o = lac.orientation_test(ch.x.values, ch.y.values, ch[measure].values, anat, cfg, keep)
        o.pop('perm_aucs')
        orient.append({'measure': measure, **o})
    theta = [{'derivation': 'monopolar', 'measure': m,
              **lac.ec_theta_test(ch.x.values, ch.y.values, ch[m].values, anat)}
             for m in ('th_theta_excess_db', 'th_rel_theta')]
    theta.append({'derivation': 'bipolar', 'measure': 'th_theta_excess_db',
                  **lac.ec_theta_test(bip.x.values, bip.y.values, bip['th_theta_excess_db'].values, anat)})
    gates['pass'] = bool(gates['round_trip_max_abs_diff'] == 0 and gates['geometry_vs_sort_max_um'] == 0
                         and gates['decimation_r_median'] > 0.99
                         and (kind != 'task' or (abs(gates['clock_lag_s']) <= cfg.clock_max_lag_s
                                                 and gates['clock_r'] > 0.1)))
    meta['restart_at'] = restart
    name = base + (f'_first{args.max_minutes:g}min' if args.max_minutes else '') + \
        ('_medpsd' if cfg.psd_stat == 'median' else '')
    _save({'channels': ch, 'bipolar': bip, 'psd': psd, 'meta': meta, 'units': u,
           'orientation': pd.DataFrame(orient), 'theta': pd.DataFrame(theta), 'gates': gates,
           'config': asdict(cfg)}, out / f'{name}.pkl')
    if not args.no_cache and not args.from_cache:
        (out / 'lfp_cache').mkdir(exist_ok=True)
        np.save(out / 'lfp_cache' / f'{name}_lfp1250_float16.npy', lfp.astype(np.float16))
    (out / f'{name}_gates.json').write_text(json.dumps(gates, indent=1, default=float))
    print(pd.DataFrame(orient)[['measure', 'auc_current', 'auc_mirror', 'rank_current', 'rank_reversed',
                                'verdict']].to_string(index=False))
    print(pd.DataFrame(theta).filter(regex='derivation|measure|auc_|n_entm_|n_entl_').to_string(index=False))
    print(f'ripples {meta["n_events"]}; gates {gates}')
    return 0 if gates['pass'] else 1


# ---------------------------------------------------------------------------
# Stage: merge
# ---------------------------------------------------------------------------

def combine_tagged(out):
    """probemap_<tag>.pkl -> probemap.pkl and units_theta_<tag>.pkl -> units_theta.pkl."""
    pm = [p for p in sorted(out.glob('probemap_*.pkl'))]
    if pm:
        parts = [_load(p) for p in pm]
        comb = {k: pd.concat([q[k] for q in parts], ignore_index=True)
                for k in ('channels', 'bipolar', 'gate_a0', 'orientation', 'theta')}
        comb['psd'] = {k: v for q in parts for k, v in q['psd'].items()}
        comb['meta'] = {k: v for q in parts for k, v in q['meta'].items()}
        comb['config'] = parts[0]['config']
        comb['parts'] = [p.name for p in pm]
        _save(comb, out / 'probemap.pkl')
        comb['channels'].to_csv(out / 'probemap_channels.csv', index=False)
        comb['orientation'].to_csv(out / 'probemap_orientation.csv', index=False)
        comb['theta'].to_csv(out / 'probemap_theta.csv', index=False)
    ut = [p for p in sorted(out.glob('units_theta_*.pkl'))]
    if ut:
        parts = [_load(p) for p in ut]
        comb = {k: pd.concat([q[k] for q in parts], ignore_index=True)
                for k in ('units', 'round_trip', 'contrasts', 'fractions')}
        comb['config'] = parts[0]['config']
        comb['parts'] = [p.name for p in ut]
        _save(comb, out / 'units_theta.pkl')
        comb['units'].to_csv(out / 'units_theta.csv', index=False)


RIPPLE_MEASURES = ('ripple_gain', 'rest_ripple_excess_db', 'ripple_rate_hz')


def posthoc_orientation(out, cfg):
    """POST-HOC (added 2026-10-02 after the ly06 probe map): per-bank AUCs and the discordant test.

    Pooled over banks, contacts labelled the same under both mappings dilute the per-mapping AUCs
    and ripple gain mixes one reference per bank; these are reported beside, never instead of, the
    pre-registered verdicts.
    """
    anat = anatomy_by_mouse(out)
    res = {}
    if (out / 'probemap.pkl').exists():
        pm = _load(out / 'probemap.pkl')
        ch = pm['channels']
        keep = ch['gate_ok'].values.astype(bool) & ~ch['is_ref'].values.astype(bool) & ~ch['dead'].values.astype(bool)
        disc, bycfg = [], []
        for mouse, d in ch.groupby('mouse'):
            pos = ch.index.get_indexer(d.index)
            for m in RIPPLE_MEASURES:
                disc.append({'mouse': mouse, 'measure': m, **lac.discordant_test(
                    d.x.values, d.y.values, d[m].values, anat[mouse], strata=d.config.values, keep=keep[pos])})
            for config, dc in d.groupby('config'):
                pc = ch.index.get_indexer(dc.index)
                for m in RIPPLE_MEASURES:
                    o = lac.orientation_test(dc.x.values, dc.y.values, dc[m].values, anat[mouse], cfg, keep[pc])
                    o.pop('perm_aucs')
                    bycfg.append({'mouse': mouse, 'config': config, 'measure': m, **o})
        pm['discordant'] = pd.DataFrame(disc)
        pm['orientation_by_config'] = pd.DataFrame(bycfg)
        _save(pm, out / 'probemap.pkl')
        pm['discordant'].to_csv(out / 'probemap_discordant.csv', index=False)
        pm['orientation_by_config'].to_csv(out / 'probemap_orientation_by_config.csv', index=False)
        res['probemap_discordant'] = pm['discordant']
        res['probemap_orientation_by_config'] = pm['orientation_by_config']
    rows = []
    for p in sorted(out.glob('raw_*.pkl')):
        r = _load(p)
        c = r['channels']
        mouse = p.stem.split('_')[1]
        keep = ~c['is_ref'].values.astype(bool) & ~c['dead'].values.astype(bool)
        for m in RIPPLE_MEASURES:
            rows.append({'recording': p.stem[4:], 'measure': m, **lac.discordant_test(
                c.x.values, c.y.values, c[m].values, anat[mouse], keep=keep)})
    if rows:
        res['raw_discordant'] = pd.DataFrame(rows)
        res['raw_discordant'].to_csv(out / 'raw_discordant.csv', index=False)
    return res


def stage_merge(args, out):
    combine_tagged(out)
    lines = [f'LFP anatomy check summary ({time.strftime("%Y-%m-%d %H:%M")})', '']
    summary = posthoc_orientation(out, lac.LFPConfig())
    for k in ('probemap_discordant', 'raw_discordant'):
        if k in summary:
            lines += [f'POST-HOC {k} (A = ripple structure only under current; AUC > 0.5 favours current):',
                      summary[k].to_string(index=False), '']
    if (out / 'probemap.pkl').exists():
        pm = _load(out / 'probemap.pkl')
        summary['probemap_orientation'] = pm['orientation']
        summary['probemap_theta'] = pm['theta']
        lines += ['Probe map orientation (AUC ripple structures vs other grey matter):',
                  pm['orientation'][['mouse', 'measure', 'auc_current', 'auc_mirror', 'rank_current',
                                     'verdict']].to_string(index=False), '',
                  'Probe map theta (AUC ENTm vs ENTl, bank 0):',
                  pm['theta'].filter(regex='mouse|derivation|measure|auc_').to_string(index=False), '']
    raws = sorted(p for p in out.glob('raw_*.pkl'))
    if raws:
        ro, rt, ru = [], [], []
        for p in raws:
            r = _load(p)
            tag = p.stem[4:]
            ro.append(r['orientation'].assign(recording=tag))
            rt.append(r['theta'].assign(recording=tag))
            ru.append(r['units'])
            lines.append(f'{tag}: gates {r["gates"]}')
        summary['raw_orientation'] = pd.concat(ro, ignore_index=True)
        summary['raw_theta'] = pd.concat(rt, ignore_index=True)
        summary['raw_units'] = pd.concat(ru, ignore_index=True)
        lines += ['', 'Raw pilot orientation:',
                  summary['raw_orientation'][['recording', 'measure', 'auc_current', 'auc_mirror',
                                              'verdict']].to_string(index=False), '',
                  'Raw pilot theta (AUC ENTm vs ENTl):',
                  summary['raw_theta'].filter(regex='recording|derivation|measure|auc_').to_string(index=False), '']
        uu = summary['raw_units']
        for mapping in ('current', 'mirror'):
            g = (uu.groupby(['recday', 'session', f'group_{mapping}'])
                 .agg(n=('order', 'size'), ripple_ratio=('ripple_ratio', 'median'),
                      frac_ripple_mod=('ripple_p', lambda v: float(np.mean(np.asarray(v) < 0.01))),
                      mrl_ref=('mrl_ref', 'median'))
                 .reset_index())
            summary[f'raw_unit_groups_{mapping}'] = g
            lines += [f'Raw pilot units by {mapping}-label group:', g.to_string(index=False), '']
    if (out / 'units_theta.pkl').exists():
        ut = _load(out / 'units_theta.pkl')
        summary['unit_fractions'] = ut['fractions']
        summary['unit_contrasts'] = ut['contrasts']
        lines += ['Unit theta rhythmicity by current-label group:', ut['fractions'].to_string(index=False), '',
                  'Within-recday shuffle contrasts (theta index):', ut['contrasts'].to_string(index=False)]
    _save(summary, out / 'summary.pkl')
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['anatomy', 'probemap', 'speed', 'units', 'raw', 'merge'])
    ap.add_argument('--mice', default=','.join(lac.MICE))
    ap.add_argument('--configs', default='')
    ap.add_argument('--recdays', default='')
    ap.add_argument('--recday', default='')
    ap.add_argument('--session', default='auto-task')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    ap.add_argument('--n-shuffle', type=int, default=1000)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--max-minutes', type=float, default=0.0, help='raw: smoke-test on the first N minutes')
    ap.add_argument('--from-cache', action='store_true', help='raw: re-analyse the cached 1250 Hz LFP')
    ap.add_argument('--psd-stat', default='mean', choices=['mean', 'median'])
    ap.add_argument('--out-dir', default=str(lac.OUT_DIR))
    ap.add_argument('--tag', default='', help='output suffix for probemap/units; default: the mice, or all')
    args = ap.parse_args()
    args.mice = [m for m in args.mice.split(',') if m]
    args.tag = args.tag or ('all' if set(args.mice) == set(lac.MICE) else '_'.join(args.mice))
    args.configs = [c for c in args.configs.split(',') if c]
    args.recdays = [r for r in args.recdays.split(',') if r]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = lac.LFPConfig(psd_stat=args.psd_stat)
    t0 = time.time()
    rc = {'anatomy': lambda: stage_anatomy(args, out),
          'probemap': lambda: stage_probemap(args, out, cfg),
          'speed': lambda: stage_speed(args, out),
          'units': lambda: stage_units(args, out, cfg),
          'raw': lambda: stage_raw(args, out, cfg),
          'merge': lambda: stage_merge(args, out)}[args.stage]()
    print(f'stage {args.stage} done in {(time.time() - t0) / 60:.1f} min\nEXIT={rc}')
    sys.exit(rc)


if __name__ == '__main__':
    main()
