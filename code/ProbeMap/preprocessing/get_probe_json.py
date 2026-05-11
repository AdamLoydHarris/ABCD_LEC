"""
New preprocessing file to save out probe data as json files that is readable
by probeinterface.
"""

# %% Imports
from spikeinterface import extractors as se
from probeinterface import write_probeinterface

# %% import json


def save_session_probe(session_dir, processed_data_path):
    raw_data = se.read_openephys(session_dir.ephys_path)
    Probe = raw_data.get_probe()
    savepath = processed_data_path / "probe.json"
    write_probeinterface(savepath, Probe)
    return
