"""
For Yiying - preprocessing code to get from raw neuron activity to normalised firing rate matrices in trial progress bins.
assumes you have Trial_times and Neuron_raw arrays for a session, and that Trial_times is an array of shape (num_trials, num_states)
 with the time indices for each state transition in the trial, and Neuron_raw is an array of shape (num_neurons, total_time_points) with the raw firing rates of each neuron at each time point.

 
Important!!!
Your data should be truncates so that the first timebin of the ephys data is aligend to the first A
so trial 1 state a is 0 in trial_times. First column of the neural array should be the firing rate at time 0, which is the start of state A in trial 1. 
This way, when we partition the neural data using the trial times, we will get the correct segments of neural activity corresponding to each state in each trial.
"""



import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, sem, entropy



def partition(alist, indices):
    """
    Partitions a list into sublists based on a list of indices.
    Returns a numpy array of the sublists with dtype=object to handle varying lengths.
    """
    return np.asarray([np.asarray(alist[i:j]) for i, j in zip(indices[:-1], indices[1:])], dtype=object)
def unroll_listoflists(l):
    flat_list = [item for sublist in l for item in sublist]
    return(flat_list)

def smooth_circ(xx, sigma=10, axis=0):
    x_smoothed = gaussian_filter1d(np.hstack((xx, xx, xx)), sigma, axis=axis)[
        len(xx) : int(len(xx) * 2)
    ]
    return x_smoothed
def std_err(data_neuron):
    return smooth_circ(sem(data_neuron, axis=0))



num_trials=len(Trial_times)
num_neurons=len(Neuron_raw)

num_bins=90
num_states=4
num_phases=3

Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1])).astype(int)
Neurons_norm=np.zeros((num_neurons,num_trials,num_bins*num_states))
Neurons_norm[:]=np.nan
for neuron in np.arange(num_neurons):
    Neuron_raw_neuron=Neuron_raw[neuron,:]
    Neuron_raw_neuron_split=partition(list(Neuron_raw_neuron), list(Trial_times_conc))
    Neuron_raw_neuron_split_norm=np.asarray([normalise(Neuron_raw_neuron_split[ii])\
                                            for ii in np.arange(len(Neuron_raw_neuron_split))])
    Neuron_norm=(Neuron_raw_neuron_split_norm.reshape(len(Neuron_raw_neuron_split_norm)//4,\
                                                    len(Neuron_raw_neuron_split_norm[0])*4))
    
    Neurons_norm[neuron]=Neuron_norm
    
Mean_norm = np.mean(Neuron_norm, axis=0)

Smoothed_norm = smooth_circ(Mean_norm)

Std_err_smooth= std_err(Neuron_norm)

############################################
# -------- POLAR PLOTTING CODE -------- #
############################################

# Total bins across full trial
total_bins = num_bins * num_states

# Circular angles for plotting
angles = np.linspace(0, 2 * np.pi, total_bins, endpoint=False)

for neuron_idx in range(num_neurons):


    neuron_data = Smoothed_norm[neuron_idx]

    error = Std_err_smooth[neuron_idx]


    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")

    ax.plot(angles, neuron_data, linewidth=2)
    ax.fill_between(angles, neuron_data - error, neuron_data + error, alpha=0.3)

    # Optional: mark state boundaries
    state_boundaries = np.arange(0, total_bins, num_bins)
    for sb in state_boundaries:
        ax.plot(
            [angles[sb], angles[sb]],
            [0, np.nanmax(neuron_data)],
            linestyle="--",
            linewidth=1,
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Neuron {neuron_idx} Polar Firing Rate", fontsize=14)

    plt.tight_layout()
    plt.show()