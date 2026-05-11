"""Script for converting ephys/Kilosort spike times from ephys refernece to pycontorl reference"""

# %% Imports
import numpy as np
import pandas as pd
from pathlib import Path

from spikeinterface import extractors as se

# %% Global Variables
AP_SAMPLING_RATE = 30_000

# %% Functions


def get_spike_times(session_dir):
    """
    Converts spike times from ephys reference to pycontrol reference.
    Note:
    - extemerly import here: we check if reference ephys samples number start at zero if they dont
    we subtract the sample number such that they do, not doing this causes an offset between ephys
    and pycontrol times. This is a strange feature of open ephys recording software for neuropixels where
    sample numbers reference from start of looking at trances (press play) not recording traces (press record)
    """
    internal_ks_path = Path(session_dir.spikesorting_path) / "kilosort4/sorter_output"
    spike_times = np.load(internal_ks_path / "spike_times.npy")
    # just convert from units = min
    spike_times = spike_times / AP_SAMPLING_RATE
    return spike_times


def get_spike_clusters(session_dir):
    """Retrieves spike_clusters output from kilosort"""
    kilosort_path = Path(session_dir.spikesorting_path) / "kilosort4/sorter_output"
    spike_clusters = np.load(kilosort_path / "spike_clusters.npy")
    return spike_clusters


def get_cluster_metrics(
    session_dir,
    keep_metrics=[
        "unit_id",
        "presence_ratio",
        "firing_rate",
        "isi_violations_ratio",
        "amplitude_cutoff",
        "amplitude_median",
        "sd_ratio",
    ],
):
    """ """
    quality_metrics_path = Path(session_dir.spikesorting_path) / "quality_metrics.htsv"
    quality_metrics_df = pd.read_csv(quality_metrics_path, sep="\t")
    # get single_unit, multi_unit and noise_unit labels
    single_units = _is_single_unit(quality_metrics_df)
    noise_units = _is_noise_unit(quality_metrics_df)
    multi_units = np.logical_and(  # define mutli unit activity as sorted clusters that are not noise or single units
        ~single_units, ~noise_units
    )
    # process quality metrics
    quality_metrics_df.rename(columns={"unit_id": "cluster_ID"}, inplace=True)
    quality_metrics_df.set_index("cluster_ID", inplace=True)
    quality_metrics_df.drop(columns=np.setdiff1d(quality_metrics_df.columns, keep_metrics), inplace=True)
    quality_metrics_df.columns = pd.MultiIndex.from_product([["quality_metrics"], quality_metrics_df.columns])
    quality_metrics_df[("single_unit", "")] = single_units
    quality_metrics_df[("multi_unit", "")] = multi_units
    quality_metrics_df[("noise_unit", "")] = noise_units
    # process unit position
    primary_channels = _get_primary_channel(session_dir)
    quality_metrics_df[("contact", "ind")] = primary_channels
    # adapt to channel number name and shank
    if session_dir.config_name == "dense":
        ephys_path = Path("/ceph/behrens/adam_harris/LEC_probe_map/data/raw_data/multi_animal/2025-06-05_11-09-04")
    elif session_dir.config_name == "checkerboard":
        ephys_path = Path("/ceph/behrens/adam_harris/LEC_probe_map/data/raw_data/multi_animal/2025-06-05_10-34-17")
    else:
        ephys_path = session_dir.ephys_path
    raw_rec = se.read_openephys(ephys_path, stream_id=session_dir.ephys_stream_ID)
    Probe = raw_rec.get_probe()
    probe_df = Probe.to_dataframe()
    cluster_probe_locations = probe_df.loc[quality_metrics_df.contact.ind]
    quality_metrics_df[("contact", "id")] = cluster_probe_locations.contact_ids.values
    quality_metrics_df[("contact", "x")] = cluster_probe_locations.x.values
    quality_metrics_df[("contact", "y")] = cluster_probe_locations.y.values
    quality_metrics_df[("shank", "index")] = cluster_probe_locations.shank_ids.values.astype(int)
    return quality_metrics_df


# %% supporting functions


def _is_single_unit(
    quality_metric_df,
    isi_violations_ratio_thres=0.1,
    amplitude_cutoff_thres=0.1,
    firing_rate_thres=0.1,
    presence_ratio_thres=0.8,
    amplitude_median_thres=20,
    sd_ratio_thres=3,
):
    """
    Returns boolian list of whether cluster passes single unit QC
    """
    query = f"amplitude_cutoff < {amplitude_cutoff_thres} and firing_rate > {firing_rate_thres} and presence_ratio > {presence_ratio_thres} and amplitude_median > {amplitude_median_thres} and sd_ratio < {sd_ratio_thres} and isi_violations_ratio < {isi_violations_ratio_thres}"
    qc_pass_df = quality_metric_df.query(query)
    return quality_metric_df.unit_id.isin(qc_pass_df.unit_id.values).to_numpy()


def _is_noise_unit(quality_metric_df, amplitude_median_thres=20, firing_rate_thres=0.099):
    """
    Defines kilosort clusters as noise if they have low waveform template amplitude
    and average firing rate
    """
    noise_clusters_df = quality_metric_df.query(
        f"amplitude_median < {amplitude_median_thres} or firing_rate < {firing_rate_thres}"
    )
    return quality_metric_df.unit_id.isin(noise_clusters_df.unit_id.values).to_numpy()


def _get_primary_channel(session_dir):
    """
    Returns the primary channel (channel where the waveform aplitude is maximum)
    index and ID for each cluster
    """
    kilosort_path = Path(session_dir.spikesorting_path) / "kilosort4/sorter_output"
    templates = np.load(kilosort_path / "templates.npy")
    channel_map = np.load(kilosort_path / "channel_map.npy")
    channel_map += 1  # index from 1 to be consistant with probe_anatomy_df
    primary_channel = (templates**2).sum(axis=1).argmax(axis=-1)
    primary_channel = channel_map[primary_channel]
    return primary_channel
