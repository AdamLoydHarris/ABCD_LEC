# Taskspace Abstraction in Lateral Entorhinal Cortex (LEC)

Research project investigating how the lateral entorhinal cortex encodes abstract task-space structure during navigation and goal-directed behavior.

## Code Overview (`code/`)
- **Preprocessing**: spike sorting, probe mapping, ephys alignment (`preprocessing/`, `SpikeSorting/`, `merge_preprocess_kilosort_spikeinterface.py`)
- **Population analyses**: LDA state separation, CCA loop analysis, population vector analysis, spatial ratemaps
- **Decoding**: state decoding, time decoding, goal-progress GLM
- **Notebooks**: exploratory analyses for LEC and PFC, figure generation

## Data
Raw data, dense outputs, and sbatch job files are excluded from version control (see `.gitignore`).
