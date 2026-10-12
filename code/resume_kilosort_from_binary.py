"""Re-run Kilosort onward on a block whose destriped `binary/` already exists.

Written 2026-09-22. The five 6-day July sorts each wrote their 1.6 TB destriped binary and
then died in `run_sorter` on the numpy-2 probe-file bug (see the merge script's
`set_printoptions(legacy="1.25")` note). Re-running the merge script from scratch would redo
~1.5 h of destriping per block and write another 8.5 TB, and it would in fact crash first,
because the binary is saved to a fixed `{dates}_preprocessed/binary` path that already exists.

This resumes from that binary and runs exactly the tail of
`merge_preprocess_kilosort_spikeinterface.py` (:540 onward): run_sorter -> remove_excess_spikes
-> remove_empty_units -> get_quality_metrics -> get_single_units. Both functions are imported
from the merge script rather than reimplemented, so the QC gate cannot drift from it. The
figures (noise levels, drift, probe map) already exist from the failed run and are not redrawn.

    python code/resume_kilosort_from_binary.py ly06 2025-07-14_..._2025-07-19 9 8

I/O: reads `{block}/binary/`, writes `{block}/kilosort_output/`. Refuses if the binary is
missing or its sample count does not match `recording_sessions_in_concat.csv`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# NumPy 2 removed aliases kilosort 4.0.18 still uses, and changed the repr probeinterface
# writes into probe.prb. Must be imported before the sorter runs. See code/numpy2_compat.py.
import numpy2_compat  # noqa: F401,E402

# A transient ceph stall made tqdm's stderr write raise EAGAIN and killed ly07 (3666095);
# stdio now waits instead of raising. See code/robust_stdio.py.
import robust_stdio  # noqa: F401,E402

import spikeinterface as si_top
from spikeinterface import core as si
from spikeinterface import curation as sc
from spikeinterface import sorters as ss

REPO = Path("/ceph/behrens/adam_harris/Taskspace_abstraction_lEC")
EPHYS = REPO / "data" / "preprocessed" / "ephys"
BYTES_PER_SAMPLE_PER_CH = 2


def _merge_module():
    """The merge script, imported for its QC functions so the gate cannot drift."""
    path = REPO / "code" / "merge_preprocess_kilosort_spikeinterface.py"
    spec = importlib.util.spec_from_file_location("merge_ks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resume(mouse: str, block: str, ths: tuple[int, int],
           smoke_minutes: float = 0.0, qc_only: bool = False) -> int:
    """Run Kilosort onward on an existing binary. `smoke_minutes` > 0 sorts only that
    much of it, into a throwaway folder, to prove the path end to end in ~30 min.
    `qc_only` reads a finished sort from kilosort_output/ instead of re-running Kilosort."""
    block_dir = EPHYS / mouse / f"{block}_preprocessed"
    binary_dir = block_dir / "binary"
    ks_dir = block_dir / "kilosort_output"
    if not binary_dir.exists():
        sys.exit(f"no binary at {binary_dir}")

    rec = si.load_extractor(binary_dir)
    print(f"loaded binary: {rec}")

    # The binary must cover the whole concat, or the sort would silently be of a truncation.
    csv = pd.read_csv(block_dir / "recording_sessions_in_concat.csv")
    n_ch_raw = 384
    expected = int((csv["dat_size"] // (BYTES_PER_SAMPLE_PER_CH * n_ch_raw)).sum())
    got = int(rec.get_num_samples())
    print(f"samples: binary {got:,}  concat table {expected:,}  "
          f"({len(csv)} recordings, {got / rec.sampling_frequency / 3600:.2f} h)")
    if got != expected:
        sys.exit(f"binary is {got} samples, concat table says {expected}; refusing")

    if smoke_minutes:
        # Same code path, a slice of the data, and never near the real output folder.
        n = int(smoke_minutes * 60 * rec.sampling_frequency)
        rec = rec.frame_slice(start_frame=0, end_frame=min(n, got))
        ks_dir = block_dir / f"kilosort_output_SMOKE_{smoke_minutes:g}min"
        print(f"SMOKE TEST: {smoke_minutes:g} min slice -> {ks_dir}")

    ks_dir.mkdir(parents=True, exist_ok=True)
    merge = _merge_module()

    sorter_params = ss.get_default_sorter_params("kilosort4")
    sorter_params["Th_universal"] = ths[0]
    sorter_params["Th_learned"] = ths[1]
    sorter_params["do_CAR"] = False        # IBL destriping was done in the preprocessing
    print(f"Running kilosort with parameters: {sorter_params}")

    if qc_only:
        # ly05 (3666093) finished Kilosort in 7 d, then hit the walltime 5.9 h into quality
        # metrics; the sort is on disk, so read it back rather than spend another week.
        if not (ks_dir / "sorter_output" / "spike_times.npy").exists():
            sys.exit(f"--qc-only but no finished sort in {ks_dir}/sorter_output; refusing")
        sorter = ss.read_sorter_folder(str(ks_dir))
        print(f"QC ONLY: read finished sort from {ks_dir}")
    else:
        sorter = ss.run_sorter("kilosort4", recording=rec, folder=str(ks_dir),
                               verbose=True, remove_existing_folder=True, **sorter_params)

    sorter = sc.remove_excess_spikes(sorter, rec)
    sorter = sorter.remove_empty_units()
    print(f"Sorting completed. Found {len(sorter.get_unit_ids())} units including mua.")

    qdf = merge.get_quality_metrics(sorter, rec, str(ks_dir), save_cluster_reports=True)
    qdf.to_csv(ks_dir / "quality_metrics.htsv", sep="\t", index=False)
    single_units = merge.get_single_units(qdf)
    np.save(ks_dir / "QC_single_units.npy", single_units)
    print(f"Found {len(single_units)} single units, passing quality control")
    return len(single_units)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mouse")
    p.add_argument("block", help="dates joined by _, without the _preprocessed suffix")
    p.add_argument("th_universal", type=int)
    p.add_argument("th_learned", type=int)
    p.add_argument("--smoke-minutes", type=float, default=0.0,
                   help="sort only this many minutes, into kilosort_output_SMOKE_*; "
                        "proves the path without spending a day on it")
    p.add_argument("--qc-only", action="store_true",
                   help="skip Kilosort; read the finished sort in kilosort_output/ and run "
                        "only the quality metrics and gate (reuses a saved sorting_analyzer)")
    a = p.parse_args()
    print(f"numpy2_compat restored: {numpy2_compat.RESTORED}")
    resume(a.mouse, a.block, (a.th_universal, a.th_learned), a.smoke_minutes, a.qc_only)
