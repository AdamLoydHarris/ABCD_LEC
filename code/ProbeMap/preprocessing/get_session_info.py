""" """

# %% Imports
import os
import json

# %% New Main functions


def save_session_info(session_dir, output_path):
    """Maybe move this to populate processed data?"""
    session_info = get_session_info(session_dir)
    with open((os.path.join(output_path, "session_info.json")), "w") as outfile:
        outfile.write(json.dumps(session_info, indent=4))
    return


def get_session_info(session_dir):
    return {
        "subject_ID": session_dir.subject_ID,
        "config_name": session_dir.config_name,
        "datetime": session_dir.ephys_datetime.isoformat(),
    }
