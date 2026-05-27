"""
Library for submitting ephys preprocessing jobs to the HPC. For main spike-sorting library see spikesort_session.py
"""

# %% Imports
import os
from pathlib import Path
from . import spikesort_session as sps

JOBS_FOLDER = Path("SpikeSorting/jobs/")
if not JOBS_FOLDER.exists():
    for sub_folder in ["slurm", "out", "err"]:
        if not (JOBS_FOLDER / sub_folder).exists():
            (JOBS_FOLDER / sub_folder).mkdir(parents=True)


# %% Functions
# %% ephys preprocessing
def run_ephys_preprocessing():
    unsorted_ephys_paths_df = sps.get_ephys_paths_df().query(
        "spike_interface_readable==True and spike_sorting_completed ==False"
    )
    assert (
        not unsorted_ephys_paths_df.empty
    ), "No valid ephys sessions found. Check data paths & get_ephys_paths_df() function."
    # find unspike-sorted sessions
    # unsorted_ephys_paths_df = ephys_paths_df[~ephys_paths_df.spike_sorting_completed]
    if unsorted_ephys_paths_df.empty:
        return print("Ephys preprocessing already complete or data not found.")
    # double-check job folders exit
    for subfolder in ["slurm", "out", "err"]:
        if not (JOBS_FOLDER / subfolder).exists():
            (JOBS_FOLDER / subfolder).mkdir(parents=True)
    for ephys_info in unsorted_ephys_paths_df.itertuples():
        print(f"Submitting {ephys_info.subject_ID} {ephys_info.datetime} to HPC")
        script_path = get_ephys_preprocessing_SLURM_script(ephys_info)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
    return print("All ephys preprocessing jobs submitted to HPC. Check progress with 'squeue -u <username>'")


def submit_test_job(ephys_info):
    script_path = get_ephys_preprocessing_SLURM_script(ephys_info)
    os.system(f"chmod +x {script_path}")
    os.system(f"sbatch {script_path}")
    return print(f"Test job submitted for {ephys_info.subject_ID} {ephys_info.datetime.isoformat()}")


def get_ephys_preprocessing_SLURM_script(
    ephys_info,
    spike_sorter="Kilosort4",
    RAM="64GB",
    time_limit="48:00:00",
    IBL_preprocessing=True,
    kilosort_Ths=[9, 8],
    spikesort_path=sps.SPIKESORTING_PATH,  # parameters relevant for optimising kilosort.
    jobs_folder=JOBS_FOLDER,
    python_path=".",
    import_path="SpikeSorting",
):  # options to edit jobs folder and python path when running tests.
    """Saves a script which submits a SLURM job to perform spikesorting."""
    session_ID = f"{ephys_info.subject_ID}_{ephys_info.datetime.isoformat()}_{spike_sorter}"
    script = f"""#!/bin/bash
#SBATCH --job-name=ephys_preprocessing_{session_ID}
#SBATCH --output={str(jobs_folder)}/out/ephys_pre_{session_ID}_{spike_sorter}_{kilosort_Ths[0]}_{kilosort_Ths[1]}_IBL_{IBL_preprocessing}.out
#SBATCH --error={str(jobs_folder)}/err/ephys_pre_{session_ID}_{spike_sorter}_{kilosort_Ths[0]}_{kilosort_Ths[1]}_IBL_{IBL_preprocessing}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}

source /etc/profile.d/modules.sh
module load miniconda
module load cuda/11.8
conda deactivate
conda deactivate
conda deactivate
conda activate maze_ephys

PYTHONPATH='{python_path}' python -c \"import {import_path}.spikesort_session as sps; sps.preprocess_ephys_session('{ephys_info.subject_ID}', '{ephys_info.config_name}', '{ephys_info.datetime.isoformat()}', '{ephys_info.ephys_path}', {IBL_preprocessing}, {kilosort_Ths}, spikesort_path='{spikesort_path}')\"
"""
    script_path = f"{jobs_folder}/slurm/ephys_pre_{session_ID}_{spike_sorter}_{kilosort_Ths[0]}_{kilosort_Ths[1]}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
