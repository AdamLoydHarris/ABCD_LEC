"""
Generate all processed data structures
"""

# %% Imports
import json
import numpy as np

from ProbeMap.preprocessing import get_data_directory as dd
from ProbeMap.preprocessing import get_ephys_data as ed
from ProbeMap.preprocessing import get_lfp_data as ld
from ProbeMap.preprocessing import get_session_info as gsi
from ProbeMap.preprocessing.get_probe_json import save_session_probe

# %% Global Variables

from ProbeMap.paths import PROCESSED_DATA_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)



# %% Main function


def populate_processed_data(
    data_streams=["session_info", "spike"],  # "lfp",  "probe"
    overwrite=False,
):
    """
    Top level function for populating standardised processed data.

    Args:
    - session_data_streams: list of raw(+ preprocessed) data streams to be processed at the session level.
        Options are ["pycontrol", "session_info", "video", "ephys"]
    """
    data_stream2func = {
        "session_info": _populate_session_info,
        "spike": _populate_spike_data,
        "lfp": _populate_lfp_data,
        "probe": _save_probe_info,
    }
    sessions_data_directory = dd.get_sessions_data_directory()
    # process session level data
    for data_stream, func in data_stream2func.items():
        if not data_stream in data_streams:
            continue
        else:
            print(f"Processing {data_stream} data")
            for session_dir in sessions_data_directory.itertuples():
                processed_data_path = PROCESSED_DATA_PATH / session_dir.subject_ID / session_dir.config_name
                processed_data_path.mkdir(parents=True, exist_ok=True)
                print(f"Processing {session_dir.subject_ID} {session_dir.config_name} {data_stream}")
                try:
                    func(session_dir, processed_data_path, overwrite)
                except FileNotFoundError as e:
                    print(f"Missing data to {session_dir.subject_ID} {session_dir.config_name} {data_stream}: {e}")
                    continue

    return print("Finished populating processed data")


# %% Subfunctions
def _save_probe_info(session_dir, processed_data_path, overwrite):
    """ """
    if not overwrite and (processed_data_path / "probe.json").exists():
        return
    else:
        # saves insdie fn
        save_session_probe(session_dir, processed_data_path)
    return


def _populate_session_info(session_dir, processed_data_path, overwrite):
    """Generates and saves session_info.json to the processed_data folder for a given session."""
    if not overwrite and (processed_data_path / "session_info.json").exists():
        return
    else:
        session_info = gsi.get_session_info(session_dir)
        # save
        with open(processed_data_path / "session_info.json", "w") as outfile:
            outfile.write(json.dumps(session_info, indent=4))
    return


def _populate_spike_data(session_dir, processed_data_path, overwrite, max_duration_delta=5):
    """
    Saves spikes.times.npy, spikes.clusters.npy and cluster.metrics.htsv for a given session to the corresponding processed_data_folder.
    If overwrite is True, the function will overwrite existing files.

    Note rest sessions are not processed here because they have not yet been preprocessed with Kilosort (as of 2024-09-24)
    """
    if not overwrite and (processed_data_path / "spikes.times.npy").exists():
        pass
    else:
        spike_pytimes = ed.get_spike_times(session_dir)
        np.save(processed_data_path / "spikes.times.npy", spike_pytimes)
    if not overwrite and (processed_data_path / "spikes.clusters.npy").exists():
        pass
    else:
        spike_clusters = ed.get_spike_clusters(session_dir)
        np.save(processed_data_path / "spikes.clusters.npy", spike_clusters)
    if not overwrite and (processed_data_path / "clusters.metrics.htsv").exists():
        pass
    else:
        cluster_metrics_df = ed.get_cluster_metrics(session_dir)
        cluster_metrics_df.columns = get_flattered_multiindex_columns(cluster_metrics_df)
        cluster_metrics_df.to_csv(processed_data_path / "clusters.metrics.htsv", sep="\t", index=False)
    return


def _populate_lfp_data(session_dir, processed_data_folder, overwrite):
    """
    Saves lfp.signal.npy, lfp.time.npy and lfp.metrics.htsv for a given session to the corresponding processed_data_folder.
    If overwrite is True, the function will overwrite existing files.

    Note rest sessions are not processed here because they have not yet been preprocessed with Kilosort (as of 2024-09-24)
    """
    # save lfp signal
    if not overwrite and (processed_data_folder / "lfp.signal.npy").exists():
        pass
    else:
        lfp_signal = ld.get_LFP_signal(session_dir)
        np.save(processed_data_folder / "lfp.signal.npy", lfp_signal)
    # save lfp times
    if not overwrite and (processed_data_folder / "lfp.times.npy").exists():
        pass
    else:
        lfp_times = ld.get_LFP_times(session_dir)
        np.save(processed_data_folder / "lfp.times.npy", lfp_times)
    # save lfp metrics
    if not overwrite and (processed_data_folder / "lfp.metrics.htsv").exists():
        pass
    else:
        lfp_metrics = ld.get_LFP_metrics(session_dir)
        lfp_metrics.columns = get_flattered_multiindex_columns(lfp_metrics)
        lfp_metrics.to_csv(processed_data_folder / "lfp.metrics.htsv", sep="\t", index=False)
    return


# %% Misc


def get_flattered_multiindex_columns(df):
    """Returns a list of flat column names (str) where columns that were previously multiindex become level0_name.level1_name
    and single index columns stay level0_name"""
    return [f"{x[0]}.{x[1]}" if x[1] != "" else x[0] for x in df.columns.to_flat_index()]
