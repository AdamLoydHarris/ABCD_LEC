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

# Same fix the merge script carries; harmless if the import below sets it again.
np.set_printoptions(legacy="1.25")

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


def resume(mouse: str, block: str, ths: tuple[int, int]) -> int:
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

    ks_dir.mkdir(parents=True, exist_ok=True)
    merge = _merge_module()

    sorter_params = ss.get_default_sorter_params("kilosort4")
    sorter_params["Th_universal"] = ths[0]
    sorter_params["Th_learned"] = ths[1]
    sorter_params["do_CAR"] = False        # IBL destriping was done in the preprocessing
    print(f"Running kilosort with parameters: {sorter_params}")

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
    if len(sys.argv) < 5:
        sys.exit("usage: resume_kilosort_from_binary.py MOUSE BLOCK_DATES TH_UNIVERSAL TH_LEARNED")
    resume(sys.argv[1], sys.argv[2], (int(sys.argv[3]), int(sys.argv[4])))
