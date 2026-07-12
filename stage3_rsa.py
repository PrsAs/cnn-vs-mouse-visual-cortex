"""
Stage 3: Representational Similarity Analysis (RSA).

Run in the same environment:
    conda activate visualmodel
    python stage3_rsa.py

What this does:
  1. Averages the mouse's neural response to each image (across repeats)
  2. Builds a "similarity table" (RDM) for the brain: for every pair of
     images, how differently did the neurons respond?
  3. Builds the same kind of table for each ResNet-18 layer
  4. Compares the brain's table to each layer's table -- the correlation
     between them tells you how "brain-like" that layer is
  5. Plots similarity vs. depth, and saves the RDMs for Stage 4
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

stage1 = np.load("stage1_output.npz", allow_pickle=True)
stage2 = np.load("stage2_output.npz", allow_pickle=True)

dff_traces = stage1["dff_traces"]        # (n_neurons, n_timepoints)
stim_table = stage1["stim_table"]        # has columns: frame, start, end, frame (image id)

# ---- Step 1: average neural response per image, across repeats ----
# Natural scenes are shown many times each; we average the response window
# after each presentation to get one "typical response" per image.
image_ids = np.unique(stim_table["frame"])
image_ids = image_ids[image_ids >= 0]  # -1 usually marks blank/gray screen, drop it

n_neurons = dff_traces.shape[0]
response_window = 15  # frames after stimulus onset to average over (~0.5s at 30Hz)

neural_responses = np.zeros((len(image_ids), n_neurons))

for idx, img_id in enumerate(image_ids):
    trials = stim_table[stim_table["frame"] == img_id]
    trial_responses = []
    for trial in trials:
        start = trial["start"]
        end = min(start + response_window, dff_traces.shape[1])
        trial_responses.append(dff_traces[:, start:end].mean(axis=1))
    neural_responses[idx] = np.mean(trial_responses, axis=0)

print(f"Averaged neural responses: {neural_responses.shape} (n_images x n_neurons)")

# ---- Step 2: build the neural RDM ----
# pdist gives pairwise distance between every image's neural response vector.
# correlation distance = 1 - correlation, so 0 = identical pattern, 2 = opposite.
neural_rdm = squareform(pdist(neural_responses, metric="correlation"))
print(f"Neural RDM shape: {neural_rdm.shape}")

# ---- Step 3: build a model RDM for each layer, and compare to the brain ----
layer_names = ["layer1", "layer2", "layer3", "layer4"]
similarities = []

# IMPORTANT: model activations in stage2_output.npz are indexed by the
# ORIGINAL image order (0..n_scenes-1). We need to select only the
# image_ids that actually appeared in this session's stimulus table,
# in the same order as neural_responses.
image_ids_int = image_ids.astype(int)

for layer in layer_names:
    layer_acts_all = stage2[layer]              # (n_all_images, n_channels)
    layer_acts = layer_acts_all[image_ids_int]   # keep only images used, matched order

    model_rdm = squareform(pdist(layer_acts, metric="correlation"))

    # Compare the two RDMs: take the upper triangle (avoid duplicating pairs
    # and the diagonal), then correlate rank order between them (Spearman).
    triu_idx = np.triu_indices_from(neural_rdm, k=1)
    corr, pval = spearmanr(neural_rdm[triu_idx], model_rdm[triu_idx])
    similarities.append(corr)
    print(f"{layer}: brain-model RSA correlation = {corr:.3f}  (p={pval:.4f})")

    np.save(f"rdm_{layer}.npy", model_rdm)

np.save("rdm_neural.npy", neural_rdm)

# ---- Step 4: plot similarity vs. network depth ----
plt.figure(figsize=(7, 5))
plt.plot(layer_names, similarities, marker="o", linewidth=2)
plt.xlabel("Network layer (early -> late)")
plt.ylabel("RSA correlation with real neural data")
plt.title("Does ResNet-18 look more or less brain-like as it gets deeper?")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("rsa_by_layer.png", dpi=150)
plt.close()
print("\nSaved rsa_by_layer.png -- this is your key result plot.")

print("\nNext: Stage 4 will look at WHICH image pairs the model and brain "
      "disagree on most -- that's where the interpretability story lives.")
