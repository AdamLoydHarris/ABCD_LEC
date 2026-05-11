"""
Preprocessing library for extracting and processing LFP data. Converts raw ephys data from raw_data/ephys to 3 GridMaze processed
data structures: 1) lfp.signal.npy (float16, uV), 2) lfp.time (float64, s from start of session), 3) lfp.metrics.htsv (tabular).

@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from pathlib import Path
from spikeinterface import extractors as se
from spikeinterface import preprocessing as sp
from scipy.interpolate import interp1d
import probeinterface as pi

# %% Global variables
from spikeinterface import core as si

si.set_global_job_kwargs(n_jobs=80, chunk_duration="1s", progress_bar=True)

# %% Functions


def get_LFP_signal(
    session_dir,
    bandpass_max=450,
    downsample_frequency=1500,
):
    """
    Function to extract and process LFP data from raw open-ephys data.
    Steps: 1) Load raw LFP data with spikeinterface
           2) Downsample channels to one per row per shank
           3) Bandpass filter LFP data (also de-means data)
           4) Downsample LFP data
           5) Convert LFP data to float16, units = uV
    """
    # load data and configre probe with spike interface
    raw_rec, _ = _load_recording(session_dir)
    # easiest way to get chanel labels to keep is to load LFP merics
    metrics_df = get_LFP_metrics(session_dir)
    downchanneled_recording = raw_rec.select_channels(metrics_df.contact.label.values.astype(str))
    # bandpass filter and downsample
    bp_recording_LFP = sp.bandpass_filter(recording=downchanneled_recording, freq_min=1, freq_max=bandpass_max)
    downsampled_LFP = sp.resample(recording=bp_recording_LFP, resample_rate=downsample_frequency)
    lfp_np32 = downsampled_LFP.get_traces(return_scaled=True)  # units = uV
    lfp_np16 = lfp_np32.astype(np.float16)  # minimal loss of precision while decreasing file size
    return lfp_np16


def get_LFP_times(session_dir, downsample_frequency=1500):
    """
    Returns LFP times (associated with LFP signals) in seconds from the start of the pycontrol session.
    """
    raw_rec = se.read_openephys(session_dir.ephys_path, stream_id="0")
    original_sample_rate = int(raw_rec.get_sampling_frequency())  # Hz
    lfp_times = raw_rec.get_times()  # seconds
    n_downsampled = int(len(lfp_times) * downsample_frequency / original_sample_rate)
    # downsample with linear interpolation
    interp_func = interp1d(np.arange(len(lfp_times)), lfp_times, kind="linear")
    lfp_times = interp_func(np.linspace(0, len(lfp_times) - 1, n_downsampled))
    return lfp_times


def get_LFP_metrics(session_dir, downsample_frequency=1500):
    """
    Generates a dataframe of LFP metrics.
    """
    _, probe = _load_recording(session_dir)
    probe_df = probe.to_dataframe()
    channel_assignments = _load_channel_assignments(session_dir)
    lfp_metrics_df = pd.DataFrame()
    lfp_metrics_df["contact", "id"] = probe_df.contact_ids
    lfp_metrics_df["contact", "index"] = probe_df.index.values.astype(int)
    lfp_metrics_df["contact", "label"] = channel_assignments.keys()
    lfp_metrics_df["contact", "qc"] = channel_assignments.values()
    lfp_metrics_df["contact", "x"] = probe_df.x
    lfp_metrics_df["contact", "y"] = probe_df.y
    lfp_metrics_df["shank", "index"] = probe_df.shank_ids.values.astype(int)
    lfp_metrics_df["sampling_rate", ""] = downsample_frequency
    lfp_metrics_df.columns = pd.MultiIndex.from_tuples(lfp_metrics_df.columns)
    # filter by channels to keep
    keep_channel_ids = get_lfp_channels_to_keep(probe_df)
    lfp_metrics_df = lfp_metrics_df[lfp_metrics_df["contact", "id"].isin(keep_channel_ids)]
    return lfp_metrics_df


# %% supporting functions


def _load_channel_assignments(session_dir):
    """
    Correct instances where channel assignments start from 65 instead of 1.
    Consistant with _load_recording function
    """
    # load channel assignmnets from spikesorting (see code/SpikeSorting/spikesort_sessions.py)
    with open(Path(session_dir.spikesorting_path) / "channel_assignments.json", "r") as file:
        channel_assigments = json.load(file)
    return channel_assigments


def _load_recording(session_dir):
    """
    Note checks for channel IDs starting from 1, if not corrects to start from 1.
    (overwrites spikeinterface probe and recording object)
    """
    raw_rec = se.read_openephys(session_dir.ephys_path, stream_id="0")
    Probe = raw_rec.get_probe()
    return raw_rec, Probe


def get_lfp_channels_to_keep(probe_df):
    """
    Takes one column of contacts for each shank of the NP2.0 probe, returning the channel IDs to keep
    under this criteria
    """
    keep_channel_ids = []
    shank_ids = probe_df.shank_ids.unique()
    for shank in shank_ids:
        shank_df = probe_df[probe_df["shank_ids"] == shank]
        x_pos_to_keep = shank_df.x.min()
        keep_channel_ids.extend(shank_df[shank_df.x == x_pos_to_keep].contact_ids)
    return keep_channel_ids
