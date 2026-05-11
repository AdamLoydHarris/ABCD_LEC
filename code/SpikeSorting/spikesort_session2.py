"""
Integrated code to

1) pre-process ephys data with IBL pipeline,
2) Run KS4 and
3) compute quality metrics for ephys data

This is at 'session' level so we aim to support different probe types as well as multiple probes per subject/session.

The following code is organised as follows:

# Manual setup / inputs
# Global variables
# Top level function (which gets called by run_ephys_preprocessing.py)
# _1 Pre-processing (IBL-style) functions
# _2 Kilosort via spikeinterface
# _3 Spikesorting quality control via spikeinterface
##   Filepath management functions

@peterdoohan and @charlesdgburns
"""

# %% Imports
import shutil
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt

# for preprocessing and kilosort
from spikeinterface import core as si
from spikeinterface import extractors as se
from spikeinterface import preprocessing as sp
from spikeinterface import widgets as sw
from spikeinterface import sorters as ss
from spikeinterface import curation as sc
from spikeinterface import postprocessing as pp
from spikeinterface import qualitymetrics as qm
from spikeinterface import exporters as sx
from spikeinterface import qualitymetrics as sq

# get probe data for cambridge neurotech probe
from probeinterface.plotting import plot_probe
from probeinterface import Probe, get_probe

# for saving unit match inputs:
from UnitMatchPy import extract_raw_data as erd

# %% Global Variables
SAMPLING_FREQUENCY = 30000
EPHYS_PATH = Path("../data/raw_data/ephys")
SPIKESORTING_PATH = Path("../data/preprocessed_data/spikesorting")

si.set_global_job_kwargs(n_jobs=80, chunk_duration="1s", progress_bar=True)

SUBJECT_IDS = ["ah10", "ly07", "ly05", "ah08", "ly06"]
SUBJECT2STREAM_ID = {
    "ah10": "0",
    "ly07": "1",
    "ly05": "2",
    "ah08": "3",
    "ly06": "4",
}  # for reading multi-animal ephys recording folders

# %%


def test():
    ephys_paths_df = get_ephys_paths_df()
    for i, row in ephys_paths_df.iterrows():
        if row.spikesorting_completed:
            continue
        preprocess_ephys_session(
            row.subject_ID,
            row.config_name,
            row.stream_id,
            row.datetime.isoformat(),
            row.ephys_path,
        )

    return


# %% Top Level Function
def preprocess_ephys_session(
    subject_ID,
    config_name,
    stream_ID,
    datetime,
    ephys_path,  # Required inputs
    IBL_preprocessing=True,
    kilosort_Ths=[9, 8],  # Options for preprocessing and kilosort parameters
    save_QC_plots=True,
    cache_preprocessed_data=True,
    remove_cached_data=True,  # QC and caching options
    spikesort_path=SPIKESORTING_PATH,
):
    """Top level function to process all the ephys data related to a session.
    Here we perform the following steps:
    0. separate probes if required
    1. preprocess data with IBL standards (per probe shank) and cache data for spikeinterface
    2. run kilosort (on each probe) and save outputs
    3. save inputs for UnitMatch (across sessions)
    4. get quality metrics
    5. delete temporarily cached data

    NB: kilosort_Ths = [Th_universal, Th_learned] for optimising over different parameters."""

    # 0.  check recording probe properties and get output filepaths (handles separate probes if required):
    print("Starting ephys preprocessing...")
    raw_rec, preprocessed_path = get_probe_recordings(
        subject_ID, config_name, stream_ID, datetime, ephys_path, spikesort_path=spikesort_path
    )
    temp_path = preprocessed_path / "temp_preprocessed"
    print(f"Checking {preprocessed_path} for DONE.txt")
    if not (preprocessed_path / "DONE.txt").exists():
        if (
            not temp_path.exists()
        ):  # If a temporary cached file already exists, we can go straight to loading (see else:)
            print("Not DONE... preprocessing data")
            preprocessed_path.mkdir(parents=True, exist_ok=True)
            # 1. perform preprocessing, here with toggles for IBL-style
            preprocessed_rec = pre_kilosort_processing(
                subject_ID,
                config_name,
                raw_rec,
                preprocessed_path=preprocessed_path,
                IBL_preprocessing=IBL_preprocessing,
                plot=save_QC_plots,  # toggle (True or False)
            )

            if cache_preprocessed_data:  # this should really always be true.
                print("Caching preprocessed data")
                if not temp_path.exists():
                    temp_path.mkdir()
                    preprocessed_rec = preprocessed_rec.save(
                        folder=preprocessed_path / "temp_preprocessed",
                        format="binary",
                        overwrite=True,
                    )
        else:  # if temp_path.exists() exists
            print("loading cached preprocessed data")
            preprocessed_rec = si.load_extractor(preprocessed_path / "temp_preprocessed")
        print("running spikesorting")
        sorter = run_kilosort4(preprocessed_rec, preprocessed_path, kilosort_Ths, IBL_preprocessing)
        print("Computing quality metrics...")
        quality_metrics_df = get_quality_metrics(sorter, preprocessed_rec, preprocessed_path, save_cluster_reports=True)
        quality_metrics_df.to_csv(preprocessed_path / "quality_metrics.htsv", sep="\t", index=False)
        single_units = get_single_units(quality_metrics_df)
        print(f"Found {len(single_units)} single units, passing quality control")
        # Remove temp files
        if remove_cached_data:
            for temp_folder in ["temp_preprocessed", "sorting_analyser"]:
                temp_path = preprocessed_path / temp_folder
                if temp_path.exists():
                    shutil.rmtree(temp_path)
        open(preprocessed_path / "DONE.txt", "w").close()  # stores an empty .txt file to check for completion.
    print(f"completed ephys preprocessing for {subject_ID} {datetime}")
    return print(f"Done for all probes")


# %% _1. IBL preprocessing functions
def pre_kilosort_processing(subject_ID, config_name, raw_rec, preprocessed_path, IBL_preprocessing=True, plot=True):

    # We always want to do phase shift if neuropixel recording
    if raw_rec.get_property("inter_shift_sample") != None:
        temp_rec = sp.phase_shift(raw_rec)
    else:
        temp_rec = raw_rec

    config_params_path = SPIKESORTING_PATH / "probe_params" / subject_ID / config_name

    if IBL_preprocessing == True:
        # high pass filter
        temp_rec = sp.highpass_filter(temp_rec)
        # remove outside brain channels and interpolate bad channels
        channel_assignments = get_channel_assignments(
            raw_rec, preprocessed_path, params_path=config_params_path
        )  # NB: the input is raw recording here
        ##From here on we preprocess data per shank
        n_shanks = len(np.unique(raw_rec.get_property("group")))
        split_recordings = temp_rec.split_by(property="group")
        preprocessed_recs = []  # for later merging of recordings
        for each_shank in range(n_shanks):
            shank_rec = split_recordings[each_shank]
            shank_channels = shank_rec.get_channel_ids()
            outside_brain_channels = [k for k, v in channel_assignments.items() if k in shank_channels and v == "out"]
            if len(outside_brain_channels) > 0:
                shank_rec = shank_rec.remove_channels(outside_brain_channels)
            else:
                outside_brain_channels = None
                # do nothing
            bad_channels = [
                k
                for k, v in channel_assignments.items()
                if k in shank_channels
                and v
                in [
                    "noise",
                    "dead",
                ]
            ]

            if n_shanks == 1:  # IBL pre-processing for neuropixel 1:
                ## interpolate bad channels
                if len(bad_channels) > 0:
                    shank_rec = sp.interpolate_bad_channels(shank_rec, bad_channel_ids=bad_channels)
                ## and do spatial filtering to destripe (denoising)
                n_channels_on_shank = len(shank_rec.get_property("group"))
                n_channel_pad = min(60, n_channels_on_shank)
                shank_rec = sp.highpass_spatial_filter(shank_rec, n_channel_pad=n_channel_pad)
            elif n_shanks > 1:  # alternative simpler pre-processing for multi-shank probes:
                ## remove bad channels
                if len(bad_channels) > 0:
                    shank_rec = shank_rec.remove_channels(bad_channels)
                ## and do common average referencing to destripe (denoise)
                shank_rec = sp.common_reference(shank_rec, operator="median", reference="global")
            preprocessed_recs.append(shank_rec)

        preprocessed_rec = si.aggregate_channels(preprocessed_recs)
        if plot:
            print("saving preprocessed traces for visual inspection")
            fig = _plot_preprocessed_trace_qc(split_recordings, preprocessed_recs)
            fig.savefig(preprocessed_path / f"preprocessed_trace_qc.png")
    elif IBL_preprocessing == False:
        preprocessed_rec = temp_rec

    return preprocessed_rec


def _plot_preprocessed_trace_qc(split_recordings, preprocessed_recs):
    """Saves figure with raw and preprocessed traces for quality control.
    This is plotted per shank"""
    n_shanks = len(preprocessed_recs)
    fig, axs = plt.subplots(2, n_shanks, figsize=(n_shanks * 10, 40))
    for each_shank in range(n_shanks):

        # sort out axes indexing
        if n_shanks == 1:
            top_axis = axs[0]
            bottom_axis = axs[1]
        else:
            top_axis = axs[0, each_shank]
            bottom_axis = axs[1, each_shank]

        top_axis.set(title=f"Raw shank {each_shank+1}", xlabel="Time(s)")
        sw.plot_traces(
            split_recordings[each_shank],
            backend="matplotlib",
            clim=(-100, 100),
            ax=top_axis,
            return_scaled=True,
            time_range=[500, 505],
            order_channel_by_depth=True,
            show_channel_ids=True,
        )

        bottom_axis.set(title=f"preprocessed shank {each_shank+1}", xlabel="Time(s)")
        sw.plot_traces(
            preprocessed_recs[each_shank],
            backend="matplotlib",
            clim=(-100, 100),
            ax=bottom_axis,
            return_scaled=True,
            time_range=[500, 505],
            order_channel_by_depth=True,
            show_channel_ids=True,
        )
    return fig


# $$ _2. Kilosort via spikeinterface
def run_kilosort4(preprocessed_rec, preprocessed_path, kilosort_Ths=[9, 8], IBL_preprocessing=True):
    """Runs kilosort4 after preprocessing using spike-interface.
    We allow changes to Th_universal and Th_learned for optimisation, leaving all other parameters default.
    Note that we also toggle IBL_preprocessing here, as this will stop kilosort preprocessing/"""
    kilosort_output_path = preprocessed_path / "kilosort4"
    # load best Th parameters if kilosort parameters have been optimised. Otherwise default is given above.
    if (SPIKESORTING_PATH / "kilosort_optim" / "best_params.json").exists():
        with open(SPIKESORTING_PATH / "kilosort_optim" / "best_params.json", "r") as f:
            kilosort_Ths = json.load(f)
    if not (
        preprocessed_path / "kilosort4"
    ).exists():  # if the ks folder exists, assume sorting completed with no bugs.
        print("running Kilosort4")
        kilosort_output_path.mkdir(parents=True)

        # Set up parameters for kilosort
        sorter_params = ss.get_default_sorter_params("kilosort4")
        # For optional changes to kilosort parameters
        sorter_params["Th_universal"] = kilosort_Ths[0]
        sorter_params["Th_learned"] = kilosort_Ths[1]
        if IBL_preprocessing == True:
            sorter_params["do_CAR"] = False  # we perform IBL destriping instead using spikeinterface
        n_shanks = len(np.unique(preprocessed_rec.get_property("group")))
        if n_shanks == 1:
            sorter_params["nblocks"] = (
                5  # Default is 1 (rigid), 5 is recommended for single shank neuropixel. Shouldn't have a big influence regardless.
            )

        sorter = ss.run_sorter(
            "kilosort4",
            recording=preprocessed_rec,
            folder=kilosort_output_path,
            verbose=True,
            remove_existing_folder=True,
            **sorter_params,
        )
        sorter = sc.remove_excess_spikes(sorter, preprocessed_rec)
        sorter = sorter.remove_empty_units()
    else:  # if ks already run load sorter
        print("loading Kilosort4 output")
        sorter = ss.read_sorter_folder(
            kilosort_output_path,
            register_recording=preprocessed_rec,
        )
    return sorter


# %% _3. Spike sorting quality control via spikeinterface
def get_quality_metrics(sorter, preprocessed_rec, preprocessed_path, save_cluster_reports):
    sorter = sc.remove_excess_spikes(sorter, preprocessed_rec)
    analyzer_output_path = preprocessed_path / "sorting_analyzer"
    if analyzer_output_path.exists():
        print("Loading Analyzer from Disk")
        analyzer = si.load_sorting_analyzer(analyzer_output_path)
    else:
        analyzer = si.create_sorting_analyzer(
            sorter,
            preprocessed_rec,
            sparse=True,
            format="binary_folder",
            folder=analyzer_output_path,
            overwrite=True,
        )
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
        analyzer.compute("waveforms", ms_before=1.5, ms_after=2.0)
        print("calculating templates")
        analyzer.compute("templates", operators=["average", "median", "std"])
        print("calculating noise levels")
        analyzer.compute("noise_levels")
        print("calculating correlograms")
        analyzer.compute("correlograms")
        print("calculating unit locations")
        analyzer.compute("unit_locations")
        print("calculating spike amplitudes")
        analyzer.compute("spike_amplitudes")
        print("calculating spike locations")
        analyzer.compute("spike_locations")
        print("calculating template similarity")
        analyzer.compute("template_similarity")
    print("computing quality metrics df")
    metrics_list = sq.get_quality_metric_list()
    metrics_df = sq.compute_quality_metrics(analyzer, metric_names=metrics_list)
    # process metrics df
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={"index": "unit_id"}, inplace=True)
    metrics_df["amplitude_median"] = metrics_df["amplitude_median"].abs()
    # save cluster reports
    if save_cluster_reports:
        print("exporting cluster reports")
        try:
            sx.export_report(
                analyzer,
                output_folder=preprocessed_path / "cluster_reports",
                remove_if_exists=True,
            )
        except ValueError:
            print("Failed generating cluster report")
    return metrics_df


def get_single_units(
    quality_metric_df,
    isi_violations_ratio_thres=0.1,
    amplitude_cutoff_thres=0.1,
    firing_rate_thres=0.1,
    presence_ratio_thres=0.9,
    amplitude_median_thres=20,  # For Neuropixel 2.0 - per shank CAR requires lowering this threshold.
    sd_ratio_thres=3,
):
    """
    Filter sortered clusters by quality metrics to find single units (clusters that pass QC metrics)

    Metrics:
    """
    isi_violations_mask = np.logical_or.reduce(
        [
            quality_metric_df.isi_violations_ratio < isi_violations_ratio_thres,
        ]
    )
    qc_pass_df = quality_metric_df[isi_violations_mask]
    remaining_query = f"amplitude_cutoff < {amplitude_cutoff_thres} and firing_rate > {firing_rate_thres} and presence_ratio > {presence_ratio_thres} and amplitude_median > {amplitude_median_thres} and sd_ratio < {sd_ratio_thres}"
    qc_pass_df = qc_pass_df.query(remaining_query)
    return qc_pass_df.unit_id.values


# %% _0. Handling probe and recording information


def save_channel_assignment_params(
    outside_thresh=0.75,
    n_neighbours=11,
):
    """INPUT: subject_ID = 'subject_01'
           outside_thresh = -0.75 is default, but sometimes 0.35 is useful
           n_neighours = 11 is default, but up to 37 has been used
           probe_label = 'ProbeA' # specify ONLY if multiple probes for a given subject.

    OUTPUT: saves parameters to assign bad channels across sessions.

    NOTES: For a given subject, attempts to assign outside brain channels to first and last recording."""

    ephys_paths_df = get_ephys_paths_df()
    for i, row in ephys_paths_df.iterrows():
        base_path = SPIKESORTING_PATH / "probe_params" / row.subject_ID
        assignment_path = base_path / row.config_name
        assignment_path.mkdir(parents=True, exist_ok=True)
        raw_rec, preprocessed_path = get_probe_recordings(
            row.subject_ID,
            row.config_name,
            row.stream_id,
            row.datetime.isoformat(),
            row.ephys_path,
        )
        preprocessed_path.mkdir(parents=True, exist_ok=True)  # we must make the directory for this step.
        get_channel_assignments(
            raw_rec,
            preprocessed_path,
            assignment_path,
            outside_thresh,
            n_neighbours,
            save_params=True,
        )


def get_channel_assignments(
    raw_rec,
    preprocessed_path,  # Required inputs
    params_path=None,
    outside_thres=-0.75,
    n_neighbours=11,
    save_params=False,  # for saving parameters
    plot_outside_brain_channels=True,  # option to avoid plotting
):
    """
    INPUT: raw_rec and preprocessed_path, similar to output of get_probe_recordings()
    OUTPUT: returns a dictionary with {'channel_name':'assignment'} for each channel in the recording.

    Notes:
    Finds channels outside the brain ('out'), noisey channels ('noise'), and good channels ('good') from raw ephys
    recording and saves this information to disk as a .json file.

    Relies primarily on spike interface .detect_bad_channels() function, so see their docs for more detail.
    Here implemented per shank.

    This function will be run for each session, using parameters for a each subject (each probe).
    See notebooks/preprocessing_ephys.ipynb for more detail.
    """

    # 1) if we're saving params, generate new stuff with input parameters.
    # 2) if we're not saving params, generate new stuff with loaded parameters
    if save_params:
        params = {"outside_threshold": outside_thres, "n_neighbours": n_neighbours}
        with open((params_path / "channel_assign_params.json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4))
    else:
        try:
            with open((params_path / "channel_assign_params.json"), "r") as j:
                params = json.loads(j.read())
        except:
            raise print(
                f"Failed loading from {params_path} \n Must verify parameters for bad channel assignment. Please see preprocessing_ephys.ipynb"
            )
    print("Assigning bad channels...")
    # Once we've got our parameters, we will generate new channel_assignments and save them

    # First we need to highpass the data
    highpass_rec = sp.highpass_filter(raw_rec)

    # then we assign bad channels per shank:
    n_shanks = len(np.unique(highpass_rec.get_property("group")))
    if plot_outside_brain_channels:
        fig, axs = plt.subplots(ncols=n_shanks * 2, figsize=(10 * n_shanks, 20))
    split_recordings = highpass_rec.split_by(property="group")
    all_channel_ids = []
    all_channel_labels = []
    for each_shank in range(n_shanks):
        shank_rec = split_recordings[each_shank]
        shank_channel_ids = shank_rec.get_channel_ids()
        # run the main spikeinterface function:
        _, channel_labels = sp.detect_bad_channels(
            shank_rec,
            outside_channels_location="top",
            outside_channel_threshold=params["outside_threshold"],
            noisy_channel_threshold=1,  # default value but specified
            dead_channel_threshold=-0.5,  # default value but specified
            n_neighbors=params[
                "n_neighbours"
            ],  # above we choose 37 as twice the size (due to nyquist) of a missing chunk in mEC_5.
            seed=0,
        )
        # count outside, noisey and dead channels:
        outside_brain_channels = (
            shank_channel_ids[channel_labels == "out"] if (channel_labels == "out").sum() > 0 else None
        )
        noisey_channels = (
            shank_channel_ids[channel_labels == "noise"] if (channel_labels == "noise").sum() > 0 else None
        )
        dead_channels = shank_channel_ids[channel_labels == "dead"] if (channel_labels == "dead").sum() > 0 else None
        # Print lines to report how many dead channels were found
        for label, channels in zip(
            ["outside_brain", "noisey", "dead"], [outside_brain_channels, noisey_channels, dead_channels]
        ):
            if channels is None:
                print(f"No {label} channels found on shank {each_shank+1}")
            else:
                print(f"{len(channels)} {label} channel(s) found on shank {each_shank+1}")
        all_channel_ids.extend(shank_channel_ids)
        all_channel_labels.extend(channel_labels)
        # Plotting for QC!
        if plot_outside_brain_channels:
            mask = np.tile(channel_labels != "good", (10, 1))
            # plot a mask of bad channels
            axs[each_shank * 2].imshow(mask.T, aspect="auto", origin="lower")
            axs[each_shank * 2].set(title=f"Bad channels mask shank {each_shank+1}")
            # plot highpassed data trace
            sw.plot_traces(
                shank_rec,
                channel_ids=shank_channel_ids,
                backend="matplotlib",
                clim=(-100, 100),
                ax=axs[each_shank * 2 + 1],
                return_scaled=True,
                time_range=[100, 200],  # arbitrary time window ~8 min into rec
                order_channel_by_depth=True,
                show_channel_ids=True,
            )
            axs[each_shank * 2 + 1].set(title=f"Highpass shank {each_shank}", xlabel="Time")
    if plot_outside_brain_channels:  # after looping through each shank, we save figure
        fig.tight_layout()
        fig.savefig(preprocessed_path / "outside_brain_channels.png")
    # save out channel assignments dict
    channel_id2assignment = {all_channel_ids[i]: all_channel_labels[i] for i in range(len(all_channel_ids))}
    with open(preprocessed_path / "channel_assignments.json", "w") as outfile:
        outfile.write(json.dumps(channel_id2assignment, indent=4))
    return channel_id2assignment


def get_probe_recordings(subject_ID, config_name, stream_ID, datetime, ephys_path, spikesort_path=SPIKESORTING_PATH):
    """ """
    spikesorting_path = Path(spikesort_path) / subject_ID / config_name / datetime
    raw_rec = se.read_openephys(ephys_path, stream_id=stream_ID)
    return raw_rec, spikesorting_path


def save_probe_config_params():
    """
    hacked together for Adam checkbaord vs dense
    """
    ephys_paths_df = get_ephys_paths_df()
    for i, row in ephys_paths_df.iterrows():
        base_path = SPIKESORTING_PATH / "probe_params" / row.subject_ID
        print(f"processing {row.config_name} for {row.subject_ID}")
        probe_params_path = base_path / row.config_name
        probe_params_path.mkdir(parents=True, exist_ok=True)
        raw_rec = se.read_openephys(row.ephys_path, stream_id=row.stream_id)
        probe = raw_rec.get_probe()
        probe_df = probe.to_dataframe()
        probe_df["device_channel_indices"] = probe.device_channel_indices
        probe_params_path.mkdir(parents=True, exist_ok=True)
        probe_df.to_csv(probe_params_path / "probe_layout.tsv", index=False)
        # also save out visual check
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_probe(probe, ax=ax)  # probeinterface function
        fig.savefig(probe_params_path / "probe_plot.png")

    return print(f"Saved out probe data to \n {probe_params_path}")


# %% Filepath management - this is quite fundamental to all top level functions.


def get_ephys_paths_df():
    """
    Note hardcoded input
    """
    init_info = []
    for subject, stream_id in SUBJECT2STREAM_ID.items():
        for ephys_path, config_name in zip(
            [
                "/ceph/behrens/adam_harris/LEC_probe_map/data/raw_data/multi_animal/2025-06-05_10-34-17",
                "/ceph/behrens/adam_harris/LEC_probe_map/data/raw_data/multi_animal/2025-06-05_11-09-04",
            ],
            ["checkerboard", "dense"],
        ):
            ephys_path = Path(ephys_path)
            datetime_string = ephys_path.parts[-1]
            dt = datetime.strptime(datetime_string, "%Y-%m-%d_%H-%M-%S")
            spike_sorting_completed = (SPIKESORTING_PATH / subject / config_name / dt.isoformat() / "DONE.txt").exists()
            init_info.append(
                {
                    "subject_ID": subject,
                    "ephys_path": ephys_path,
                    "config_name": config_name,
                    "stream_id": stream_id,
                    "datetime": dt,
                    "spikesorting_completed": spike_sorting_completed,
                }
            )
    return pd.DataFrame(init_info)
