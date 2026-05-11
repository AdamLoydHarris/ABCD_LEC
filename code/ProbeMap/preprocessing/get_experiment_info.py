"""Creates .json file with experiment info"""

# %% Imports
import json
from pathlib import Path

# %% Set experiment info path
from ProbeMap.paths import EXPERIMENT_INFO_PATH

EXPERIMENT_INFO_PATH.mkdir(parents=True, exist_ok=True)

# %% Define experiment info

SUBJECT_IDS = ["ah08", "ah10", "ly05", "ly06", "ly07"]

SUBJECT_ID2CONFIG_NAMES = {
    "ah10": ["config_1", "config_2", "config_3", "config_4", "config_5", "checkerboard", "dense"],
    "ah08": ["config_1", "config_2", "config_3", "config_4", "config_5", "checkerboard", "dense"],
    "ly05": ["config_1", "config_2", "config_3", "config_4", "config_5", "config_6", "checkerboard", "dense"],
    "ly06": ["config_1", "config_2", "config_3", "config_4", "config_5", "checkerboard", "dense"],
    "ly07": ["config_1", "config_2", "config_3", "config_4", "checkerboard", "dense"],
}


# %% Main Function


def save_exp_info():
    """
    Saves out experiment info described above (see get_experiment_info.py)
    """
    # process .json data structures
    filename2json_structure = {
        "subject_IDs": SUBJECT_IDS,
        "subject_ID2config_names": SUBJECT_ID2CONFIG_NAMES,
    }
    for filename, data_structure in filename2json_structure.items():
        with open(EXPERIMENT_INFO_PATH / (filename + ".json"), "w") as outfile:
            outfile.write(json.dumps(data_structure, indent=4))
    return


#
