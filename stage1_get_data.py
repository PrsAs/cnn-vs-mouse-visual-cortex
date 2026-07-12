"""
Stage 1: Pull Allen Brain Observatory data.

Run this on your own machine (Python 3.10 or 3.11 — NOT 3.12, allensdk's
pinned numpy version won't build on 3.12).

conda create -n visualmodel python=3.10
conda activate visualmodel
pip install allensdk torch torchvision matplotlib

First run will download ~1-2 GB into ./allen_cache the first time you
touch a new session, so make sure you're on decent wifi.
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np

# All data lands here — reuse across runs, it won't re-download.
boc = BrainObservatoryCache(manifest_file="allen_cache/manifest.json")

# Step 1: find an experiment session that used the "natural_scenes" stimulus
# in visual cortex area VISp (primary visual cortex) — good starting point,
# it's the most V1-like, well-studied area.
experiments = boc.get_ophys_experiments(
    stimuli=["natural_scenes"],
    targeted_structures=["VISp"],
)
print(f"Found {len(experiments)} candidate sessions.")
print("First few:")
for exp in experiments[:5]:
    print(f"  session_id={exp['id']}  cre_line={exp.get('cre_line')}  "
          f"depth={exp.get('imaging_depth')}")

# Step 2: pick the first session and pull its actual data
session_id = experiments[0]["id"]
print(f"\nPulling full dataset for session {session_id} (this downloads the file)...")
dataset = boc.get_ophys_experiment_data(session_id)

# Step 3: get the natural scene stimulus table (which image was shown, when)
stim_table = dataset.get_stimulus_table("natural_scenes")
print(f"\nStimulus table shape: {stim_table.shape}")
print(stim_table.head())

# Step 4: get the neural responses (dF/F traces — calcium imaging signal proxy for activity)
timestamps, dff_traces = dataset.get_dff_traces()
print(f"\ndF/F traces shape: {dff_traces.shape}  (n_neurons x n_timepoints)")

# Step 5: get the actual natural scene images shown (so you can feed the SAME
# images into your CNN later)
scenes = dataset.get_stimulus_template("natural_scenes")
print(f"\nStimulus images array shape: {scenes.shape}  (n_images x height x width)")

# Save everything you need for Stage 2 so you don't need to re-download
np.savez(
    "stage1_output.npz",
    dff_traces=dff_traces,
    stim_table=stim_table.to_records(index=False),
    scenes=scenes,
)
print("\nSaved to stage1_output.npz — ready for Stage 2 (model activations).")
