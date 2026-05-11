""" """

# %% Imports
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from mne.time_frequency import psd_array_multitaper
from ProbeMap.analysis.load_data import get_sessions

# %% Global Variables
from ProbeMap.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)


FS = 1500  # lfp sampling frequency

#%% plot dense and checkerboard

def test():
    for subject in SUBJECT_IDS:
        fig, axes = plt.subplots(1, 2, figsize=(8, 20))
        for ax, config in zip(axes,["dense", "checkerboard"]):
            ax.set_axis_off()
            try:
                session = get_sessions(subject_IDs=[subject], config_names=[config], with_data=["cluster_metrics"], must_have_data=True)
            except FileNotFoundError:
                print(f"No data for {subject} {config}")
                continue
            cluster_metrics = session.cluster_metrics
            masks = [cluster_metrics.single_unit, cluster_metrics.multi_unit]
            clusters_df = cluster_metrics[np.logical_or.reduce(masks)]
            # draw shanks
            for s in [(0, 0), (250, 0), (500, 0), (750, 0)]:
                rect = patches.Rectangle(s, (50), (4000), linewidth=1.5, facecolor="grey", alpha=0.3)
                ax.add_patch(rect)
            # plot unit locations (with some jitter)
            for unit_type, c, alpha in zip(["single_unit", "multi_unit"], ["darkblue", "darkred"], [1, 0.25]):
                df = clusters_df[clusters_df[unit_type]]
                N_units = len(df)
                x = df.contact.x.values + np.random.uniform(-10, 10, size=len(df))
                y = df.contact.y.values + np.random.uniform(-5, 5, size=len(df))
                ax.scatter(x, y, s=5, label=unit_type.replace("_", " ").title(), color=c, alpha=alpha)
                ax.text(
                    0.0,
                    0.98 - 0.01 * (1 if unit_type == "single_unit" else 2),
                    f"{unit_type.replace('_', ' ').title()}: {N_units}",
                    transform=ax.transAxes,
                    fontsize=12,
                    color=c,
                )
            ax.set_title(f"{subject.upper()} Probe Map", fontsize=16)





# %% Functions


def plot_subject_unitmap(
    subject="ah10",
    ax=None,
):
    """ """
    # combine data across session configs
    sessions = get_sessions(
        subject_IDs=[subject], config_names="config_1", with_data=["cluster_metrics"], must_have_data=True
    )
    cluster_dfs, probe_dfs = [], []
    for session in sessions:
        cluster_metrics = session.cluster_metrics
        masks = [cluster_metrics.single_unit, cluster_metrics.multi_unit]
        df = cluster_metrics[np.logical_or.reduce(masks)]
        cluster_dfs.append(df)
        probe_df = session.get_probe().to_dataframe()
        probe_dfs.append(probe_df)
    clusters_df = pd.concat(cluster_dfs, ignore_index=True)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 20))
    ax.set_axis_off()
    # draw shanks
    for s in [(0, 0), (250, 0), (500, 0), (750, 0)]:
        rect = patches.Rectangle(s, (50), (4000), linewidth=1.5, facecolor="grey", alpha=0.3)
        ax.add_patch(rect)
    # plot unit locations (with some jitter)
    for unit_type, c, alpha in zip(["single_unit", "multi_unit"], ["darkblue", "darkred"], [1, 0.25]):
        df = clusters_df[clusters_df[unit_type]]
        N_units = len(df)
        x = df.contact.x.values + np.random.uniform(-10, 10, size=len(df))
        y = df.contact.y.values + np.random.uniform(-5, 5, size=len(df))
        ax.scatter(x, y, s=5, label=unit_type.replace("_", " ").title(), color=c, alpha=alpha)
        ax.text(
            0.0,
            0.98 - 0.01 * (1 if unit_type == "single_unit" else 2),
            f"{unit_type.replace('_', ' ').title()}: {N_units}",
            transform=ax.transAxes,
            fontsize=12,
            color=c,
        )
    ax.set_title(f"{subject.upper()} Probe Map", fontsize=16)


# %%


def plot_probe_PSD(
    subject="ah10",
    axes=None,
    theta_band=(7, 11),
    ripple_band=(100, 350),
):
    """ """
    sessions = get_sessions(
        subject_IDs=[subject],
        config_names=["config_1", "config_2", "config_3", "config_4", "config_5"],
        with_data=["lfp_signal", "lfp_metrics"],
        must_have_data=True,
    )
    psd_dfs = []
    for session in sessions:
        print(session.name)
        psd_df = get_probe_PSD_df(session)
        psd_dfs.append(psd_df)
    psd_df = pd.concat(psd_dfs, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 20))
    _plot_probe_lfp(psd_df, theta_band=theta_band, ripple_band=ripple_band, axes=axes)
    axes[0].set_title(f"{subject.upper()} Probe Map - Theta Band", fontsize=16)
    axes[1].set_title(f"{subject.upper()} Probe Map - Ripple Band", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{subject}_probe_psd.png", dpi=300)
    plt.show()
    return


def _plot_probe_lfp(
    psd_df,
    theta_band=(7, 11),
    ripple_band=(100, 350),
    axes=None,
):
    """ """
    fig, axes = plt.subplots(1, 2, figsize=(8, 20))
    fig.subplots_adjust(wspace=0.5)
    for ax, freq_range, _cmap, label in zip(
        axes.flatten(), [theta_band, ripple_band], ["viridis", "plasma"], ["theta", "ripple"]
    ):
        # draw shanks
        ax.set_axis_off()
        for s in [(-20, 0), (230, 0), (480, 0), (730, 0)]:
            rect = patches.Rectangle(s, (40), (4000), linewidth=1.5, facecolor="grey", alpha=0.3)
            ax.add_patch(rect)
        # get av band PSD
        cols = psd_df.PSD.columns.astype(float)
        select_psd = psd_df.PSD[cols[(freq_range[0] < cols) & (cols < freq_range[1])]]
        channel_powers = select_psd.mean(axis=1).values  # n_channels
        cmap = plt.get_cmap(_cmap)
        norm = plt.Normalize(vmin=channel_powers.min(), vmax=channel_powers.max())
        channel_colors = cmap(norm(channel_powers))
        # plot contacts
        contact_x, contact_y = psd_df.contact.x.values, psd_df.contact.y.values
        ax.scatter(
            contact_x,
            contact_y,
            s=2,
            c=channel_colors,
            marker="s",
            alpha=0.5,
        )
        # add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
        cbar.set_label(f"Average {label} Power", fontsize=12)
    return


def get_probe_PSD_df(
    session,
    fmin=1,
    fmax=350,
    n_freqs=100,
):
    """ """
    lfp_signal = session.lfp_signal.T  # n_channels x n_samples
    psd, freqs = psd_array_multitaper(
        lfp_signal, sfreq=FS, fmin=1, fmax=250, normalization="full", verbose=False, n_jobs=20
    )
    # downsample PSD
    bin_edges = np.geomspace(fmin, fmax, n_freqs + 1)
    ds_freqs = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_idx = np.digitize(freqs, bin_edges) - 1
    bin_idx[bin_idx < 0] = 0
    bin_idx[bin_idx >= n_freqs] = n_freqs - 1
    n_channels, _ = psd.shape
    ds_psd = np.zeros((n_channels, n_freqs))
    for i in range(n_freqs):
        in_this_bin = np.where(bin_idx == i)[0]  # array of j's
        if in_this_bin.size > 0:
            ds_psd[:, i] = psd[:, in_this_bin].mean(axis=1)
        else:
            ds_psd[:, i] = np.nan
    # return as DataFrame
    lfp_metrics = session.lfp_metrics
    psd_df = pd.DataFrame(columns=pd.MultiIndex.from_product([["PSD"], ds_freqs]), data=ds_psd)
    df = pd.concat(
        [lfp_metrics, psd_df],
        axis=1,
    )
    return df


def _av_psd_multitaper(
    segments,
    fs=FS,
    fmin=1,
    fmax=250,
    n_freqs=100,
    normalization="full",
):
    """
    Compute the average PSD over many segments using MNE's multitaper method.
    Note: Because the segments have variable lengths (and thus may yield different frequency bins),
    each segment's PSD is interpolated onto a common frequency grid before averaging.
    Returns:
    - common_freqs: 1D numpy array with the common frequency grid.
    - avg_psd: 1D numpy array with the averaged PSD across segments.
    """
    common_freqs = np.geomspace(fmin, fmax, n_freqs)
    psd_list = []
    for seg in segments:
        psd, freqs = psd_array_multitaper(
            seg, sfreq=fs, fmin=fmin, fmax=fmax, normalization=normalization, verbose=False
        )
        # Interpolate this segment's PSD onto the common frequency grid.
        interp_psd = np.interp(common_freqs, freqs, psd)
        psd_list.append(interp_psd)
    psd_array = np.array(psd_list)
    avg_psd = np.mean(psd_array, axis=0)
    return common_freqs, avg_psd
