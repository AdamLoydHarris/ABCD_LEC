"""
Paths
"""

# %% imports
from pathlib import Path

# %% Paths
RAW_DATA_PATH = Path("../data/raw_data")
PREPROCESSED_DATA_PATH = Path("../data/preprocessed_data")
PROCESSED_DATA_PATH = Path("../data/processed_data")
EXPERIMENT_INFO_PATH = Path("../data/experiment_info")

EPHYS_PATH = RAW_DATA_PATH / "ephys"
SPIKESORTING_PATH = PREPROCESSED_DATA_PATH / "spikesorting"
