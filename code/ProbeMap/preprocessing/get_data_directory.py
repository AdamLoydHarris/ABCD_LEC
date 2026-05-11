"""
This library houses a top-level function "get_raw_data_directory" (and supporting functions), that generates a pandas
DataFrame with a row for each session (of all session-types, eg. maze and openfield) and columns for all the paths
to all raw data files associated with that session (eg. pycontrol, video, SLEAP, ephys, etc.). Note that in GridMaze
experiments we consider data processed through standardised piplines (eg. SLEAP, Kilosort, etc.) to be "raw_data".
Currently, only sessions with all data easily available are selected for further preprocessing. In the future, sessions
with issues such as missing ephys data (record buffer error), restarted pycontrol files etc. can be included but this will
take custom code to deal with edge cases.

This DataFrame is later used in the populate_processed_data.py script where each row of the sessions_data_directory (pandas
Series) is used to access the raw data paths needed for each preprocessing step.
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from datetime import datetime

from spikeinterface import extractors as se


# %% Global variables

from ProbeMap.paths import EPHYS_PATH, SPIKESORTING_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

with open(EXPERIMENT_INFO_PATH / "subject_ID2config_names.json", "r") as f:
    SUBJECT_ID2CONFIG_NAMES = json.load(f)

SUBJECT2ALT_STREAM_ID = SUBJECT2STREAM_ID = {
    "ah10": "0",
    "ly07": "1",
    "ly05": "2",
    "ah08": "3",
    "ly06": "4",
} 
# %% new


def _init_data_directory():
    """ """
    info = []
    for subject, config_names in SUBJECT_ID2CONFIG_NAMES.items():
        for config_name in config_names:
            info.append(
                {
                    "subject_ID": subject,
                    "config_name": config_name,
                }
            )
    return pd.DataFrame(info)


def get_sessions_data_directory():
    """ """
    # init from session_info df
    data_directory = _init_data_directory()
    # add ephys
    ephys_df = add_ephys_paths(data_directory)
    data_directory = pd.concat([data_directory, ephys_df], axis=1)
    # add spikesorting paths
    matched_spikesorting_paths, spikesorting_completed = add_spikesorting_paths(data_directory)
    data_directory["spikesorting_path"] = matched_spikesorting_paths
    data_directory["spikesorting_completed"] = spikesorting_completed
    # done :)
    data_directory["ephys_stream_ID"] = data_directory.apply(get_stream_ID, axis=1)
    return data_directory

def get_stream_ID(row):
    """ """
    if row.config_name in ["dense", "checkerboard"]:
        stream_id = SUBJECT2ALT_STREAM_ID[row.subject_ID]
    else:
        stream_id = "0"
    return stream_id


# %% Ephys Data


def add_ephys_paths(data_directory):
    """ """
    ephys_paths_df = _get_ephys_paths_df()
    matched_ephys_paths_df = pd.DataFrame(
        index=data_directory.index,
        columns=[
            "ephys_path",
            "ephys_datetime",
            "sample_numbers_path",
            "events_sync_path",
            "ephys_duration",
            "ephys_corrupt",
        ],
    )
    for i, row in data_directory.iterrows():

        filtered_sessions = ephys_paths_df[
            np.logical_and.reduce(
                [
                    (ephys_paths_df.subject_ID == row.subject_ID),
                    (ephys_paths_df.config_name == row.config_name),
                ]
            )
        ]
        if filtered_sessions.empty:
            raise ValueError(f"No sessions found for {row.subject_ID}, {row.date}, {row.session_type}")
        elif len(filtered_sessions) > 1:
            for i in filtered_sessions:
                print(i)
            raise ValueError(f"Multiple sessions found for {row.subject_ID}, {row.date}, {row.session_type}")
        else:
            matched_session = filtered_sessions.iloc[0]
            ephys_corrupt = True is matched_session.multiblock_recording or matched_session.data_missing == True
            matched_ephys_paths_df.loc[i, "ephys_datetime"] = matched_session.datetime
            matched_ephys_paths_df.loc[i, "ephys_path"] = str(matched_session.ephys_path)
            matched_ephys_paths_df.loc[i, "spikeinterface_readable"] = matched_session.spikeinterface_readable
            matched_ephys_paths_df.loc[i, "ephys_duration"] = matched_session.duration
            matched_ephys_paths_df.loc[i, "ephys_corrupt"] = ephys_corrupt
    return matched_ephys_paths_df


def _get_ephys_paths_df(force_save=True):
    """
    Returns paths to LFP data in an organsied DataFrame.
    With columns: ephys_path, subject_ID, datetime, LFP_stream_name, AP_stream_name, multiblock_recording, spikeinterface_readable, data_missing

    Note this function is cumbersome to run, checking files one by one etc. Function will first check is cached data is available and load it,
    if its not available it will run the function and save the data to disk (if save is True).
    """
    ephys_paths_cache_path = EXPERIMENT_INFO_PATH / "ephys_paths_df_cache.htsv"
    if ephys_paths_cache_path.exists() and not force_save:
        print("Loading cached ephys_paths_df")
        ephys_paths_df = pd.read_csv(ephys_paths_cache_path, sep="\t")
        ephys_paths_df["datetime"] = ephys_paths_df.datetime.apply(datetime.fromisoformat)
    else:
        # generate ephys paths df
        print("Generating ephys_paths_df")
        ephys_paths = [p for p in EPHYS_PATH.glob("*/*/*") if p.is_dir()]
        ephys_paths_info = []
        for ephys_path in ephys_paths:
            subject, config_type, datetime_string = ephys_path.parts[-3:]
            dt = datetime.strptime(datetime_string, "%Y-%m-%d_%H-%M-%S")
            path_info = {
                "subject_ID": subject,
                "config_name": config_type,
                "datetime": dt,
                "ephys_path": ephys_path,
                "multiblock_recording": None,
                "spikeinterface_readable": None,
                "data_missing": None,
                "duration": None,
            }
            # work through internal ephys folder to check find LFP and AP streams and look for errors
            record_nodes = [f for f in ephys_path.iterdir() if f.is_dir()]
            if len(record_nodes) != 1:
                raise ValueError(f"More than one record node found in ephys folder: {ephys_path}")
            record_node_path = record_nodes[0]
            experiment_paths = [f for f in record_node_path.iterdir() if f.is_dir()]
            if len(experiment_paths) != 1:
                print(f"Multiblock Recording: {record_node_path}")
                path_info["multiblock_recording"] = True
                ephys_paths_info.append(path_info)
                continue
            else:
                path_info["multiblock_recording"] = False
                experiment_path = experiment_paths[0]
            recording_paths = [f for f in experiment_path.iterdir() if f.is_dir()]
            if len(recording_paths) != 1:
                print(f"Multiblock Recording: {record_node_path}")
                path_info["multiblock_recording"] = True
                ephys_paths_info.append(path_info)
                continue
            else:
                path_info["multiblock_recording"] = False
                recording_path = recording_paths[0]
            try:  # validate that open-ephys data can be read with spike interface
                raw_rec = se.read_openephys(ephys_path, stream_id="0")
                path_info["spikeinterface_readable"] = True
            except FileNotFoundError:
                print(f"Error reading open-ephys data in folder: {ephys_path}")
                path_info["spikeinterface_readable"] = False
            except ValueError:
                print(f"Error reading open-ephys data in folder: {ephys_path}")
                path_info["spikeinterface_readable"] = False
            except IndexError:  # timestamps file empty (eg, aborted recording)
                print(f"Error reading open-ephys data in folder: {ephys_path}")
                path_info["spikeinterface_readable"] = False
            ephys_paths_info.append(path_info)
        ephys_paths_df = pd.DataFrame(ephys_paths_info)
        # save
        ephys_paths_df.to_csv(ephys_paths_cache_path, sep="\t", index=False)
    return ephys_paths_df


# %% Spike sorting


def add_spikesorting_paths(data_directory):
    """
    Add spikesorting paths and if spikesorting has run to completion
    by directly matching to ephys datetimes and subject_IDs
    """
    spikesorting_paths_df = _get_spikesorting_paths_df()
    matched_spikesorting_paths = []
    spikesorting_completed = []
    for _, session in data_directory.iterrows():
        session_ID = f"{session.subject_ID}-{session.config_name}"
        if not isinstance(session.ephys_path, str):  # no valid ephys recording for session
            matched_spikesorting_paths.append(np.nan)
            spikesorting_completed.append(np.nan)
        else:
            matched_ss = spikesorting_paths_df[
                (spikesorting_paths_df.datetime == session.ephys_datetime)
                & (spikesorting_paths_df.subject == session.subject_ID)
            ]
            if matched_ss.empty:
                print(f"Missing spikesorting data for {session_ID}")
                matched_spikesorting_paths.append(np.nan)
                spikesorting_completed.append(np.nan)
            elif len(matched_ss) > 1:
                print(f"Multiple spikesorting folders fround for {session_ID}")
            else:
                matched_spikesorting_paths.append(matched_ss.spikesorting_path.values[0])
                spikesorting_completed.append(matched_ss.spikesorting_completed.values[0])
    return matched_spikesorting_paths, spikesorting_completed


def _get_spikesorting_paths_df():
    """
    Returns paths to spike sorted ephys data in an orghanised DataFrame
    """
    spikesorting_paths = [p for p in SPIKESORTING_PATH.glob("*/*/*") if p.is_dir() and p.parts[-3] != "probe_params"]
    paths_info = []
    for ss_path in spikesorting_paths:
        subject, config_name, dt_string = ss_path.parts[-3:]
        dt = datetime.fromisoformat(dt_string)
        paths_info.append(
            {
                "subject": subject,
                "config_name": config_name,
                "datetime": dt,
                "spikesorting_path": ss_path,
                "spikesorting_completed": (ss_path / "DONE.txt").exists(),
            }
        )
    return pd.DataFrame(paths_info)
