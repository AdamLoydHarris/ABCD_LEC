""" """

# %% Imports
import json
import numpy as np
import pandas as pd
from datetime import datetime

from probeinterface import read_probeinterface


# %% Global Variables
from ProbeMap.paths import PROCESSED_DATA_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

with open(EXPERIMENT_INFO_PATH / "subject_ID2config_names.json", "r") as f:
    SUBJECT_ID2CONFIG_NAMES = json.load(f)

PROCESSED_DATA_STRUCTURE2FILENAME = {
    "spike_times": "spikes.times.npy",
    "spike_clusters": "spikes.clusters.npy",
    "cluster_metrics": "clusters.metrics.htsv",
    "lfp_times": "lfp.times.npy",
    "lfp_signal": "lfp.signal.npy",
    "lfp_metrics": "lfp.metrics.htsv",
}
# %% Get sessions


def get_sessions(subject_IDs, config_names="all", with_data="all", must_have_data=False):
    """ """
    with_data = list(PROCESSED_DATA_STRUCTURE2FILENAME.keys()) if with_data == "all" else with_data
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    requested_sessions = []
    for subject in subject_IDs:
        config_names = SUBJECT_ID2CONFIG_NAMES[subject] if config_names == "all" else config_names
        for config_name in config_names:
            requested_sessions.append(Session(subject, config_name, with_data))
    # ensure all data is available
    if must_have_data:
        requested_sessions = [s for s in requested_sessions if all(data in s.has_data for data in with_data)]
    # outpute session objects sensibly
    if len(requested_sessions) == 0:
        raise FileNotFoundError("No sessions found with the specified criteria.")
    elif len(requested_sessions) == 1:
        return requested_sessions[0]
    else:
        return requested_sessions


# %% Session class


class Session:
    """ """

    def __init__(self, subject, config_name, with_data):
        """ """
        self.has_data = []
        self.processed_data_path = PROCESSED_DATA_PATH / subject / config_name
        # Load session info
        session_info = load(self.processed_data_path / "session_info.json")
        self.session_info = session_info
        self.name = get_session_name(session_info)
        self.date = datetime.fromisoformat(session_info["datetime"])
        for attr_name in [k for k in session_info.keys() if k != "date"]:
            setattr(self, attr_name, session_info[attr_name])
        # load data
        for attr_name in with_data:
            file_name = PROCESSED_DATA_STRUCTURE2FILENAME[attr_name]
            file_path = self.processed_data_path / file_name
            data = load(file_path)
            if data is None:
                continue
            else:
                setattr(self, attr_name, data)
                self.has_data.append(attr_name)

    def get_probe(self):
        """
        Returns probeinterface object representation of the probe used to collect
        data in the session (Neuropixel 2.0; 4 shank)
        """
        probe_group = read_probeinterface(self.processed_data_path / "probe.json")
        return probe_group.probes[0]

    def __repr__(self):
        total_width = 50  # You can adjust this value if needed for wider text
        return (
            f"\n-ProbeMap Session{'-' * (total_width-4)}\n"
            f"  Subject ID   : {self.subject_ID}\n"
            f"  Config       : {self.config_name}\n"
            f"{'-' * (total_width+13)}\n"
        )


# %% Misc


def get_session_name(session_info):
    return f"{session_info['subject_ID']}.{session_info['config_name']}"


# %% load data function


def load(filepath):
    """Loads processed/analysis data from disk into memory. Making adjustments to the saved data as necessary."""
    # check filepath exists
    if not filepath.exists():
        print(f"File {filepath} does not exist")
        return None
    data_set = filepath.parts[-4]  # processed or analysis data
    if data_set == PROCESSED_DATA_PATH.name:
        data = _processed_data(filepath)
    else:
        raise ValueError(f"Data set {data_set} not recognised, must  in folder processed_data or analysis_data")
    if data is None:
        raise FileNotFoundError(f"Data structure {filepath.name} not loaded")
    else:
        return data


def _processed_data(filepath):
    """Loads processed data from disk into memory. Making adjustments to the saved data as necessary."""
    processed_data_structure = filepath.name
    if processed_data_structure == "session_info.json":
        with open(filepath, "r") as infile:
            data = json.load(infile)
    elif processed_data_structure == "spikes.times.npy":
        data = np.load(filepath)
    elif processed_data_structure == "spikes.clusters.npy":
        data = np.load(filepath)
    elif processed_data_structure == "clusters.metrics.htsv":
        df = pd.read_csv(filepath, sep="\t")
        data = _unflatten_df_columns(df)
    elif processed_data_structure == "lfp.times.npy":
        data = np.load(filepath)
    elif processed_data_structure == "lfp.signal.npy":
        data = np.load(filepath)
        data = data.astype(np.float32)
    elif processed_data_structure == "lfp.metrics.htsv":
        data = pd.read_csv(filepath, sep="\t")
        data = _unflatten_df_columns(data)
    if data is None:
        raise FileNotFoundError(f"Processed data structure {processed_data_structure} not loaded")
    else:
        return data


def _unflatten_df_columns(df):
    """
    Unflattens columns when loading a multi-index columns from disk, see
    populate_processed_data.get_flattered_multiindex_columns to see how the columns were flattened
    and saved.
    """
    df.columns = pd.MultiIndex.from_tuples([tuple(col.split(".")) if "." in col else (col, "") for col in df.columns])
    return df
