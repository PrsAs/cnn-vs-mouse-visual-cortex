"""
Noise ceiling check (run before/alongside Stage 4).

Run in the same environment:
    conda activate visualmodel
    python noise_ceiling.py

Why this matters: calcium imaging is noisy. Even a "perfect" model of the
brain could never reach a correlation of 1.0 against noisy real data. This
script estimates the actual ceiling -- the best any model could possibly
score -- by checking how well the brain's response to a stimulus predicts
ITS OWN response to the same stimulus, using independent halves of the data.

Your model's RSA scores from Stage 3 should be judged relative to THIS
number, not relative to 1.0.
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

stage1 = np.load("stage1_output.npz", allow_pickle=True)
dff_traces = stage1["dff_traces"]
stim_table = stage1["stim_table"]

image_ids = np.unique(stim_table["frame"])
image_ids = image_ids[image_ids >= 0]

response_window = 15
n_neurons = dff_traces.shape[0]

# Split repeats into two independent halves per image, average each half
# separately, then build an RDM from each half and compare them.
half1_responses = np.zeros((len(image_ids), n_neurons))
half2_responses = np.zeros((len(image_ids), n_neurons))

for idx, img_id in enumerate(image_ids):
    trials = stim_table[stim_table["frame"] == img_id]
    trial_responses = []
    for trial in trials:
        start = trial["start"]
        end = min(start + response_window, dff_traces.shape[1])
        trial_responses.append(dff_traces[:, start:end].mean(axis=1))

    trial_responses = np.array(trial_responses)
    n_trials = trial_responses.shape[0]
    if n_trials < 2:
        # not enough repeats to split -- just duplicate (weakens the estimate
        # for this image, but keeps the script running)
        half1_responses[idx] = trial_responses[0]
        half2_responses[idx] = trial_responses[0]
        continue

    midpoint = n_trials // 2
    half1_responses[idx] = trial_responses[:midpoint].mean(axis=0)
    half2_responses[idx] = trial_responses[midpoint:].mean(axis=0)

rdm_half1 = squareform(pdist(half1_responses, metric="correlation"))
rdm_half2 = squareform(pdist(half2_responses, metric="correlation"))

triu_idx = np.triu_indices_from(rdm_half1, k=1)
noise_ceiling, pval = spearmanr(rdm_half1[triu_idx], rdm_half2[triu_idx])

print(f"Noise ceiling (brain vs. itself, split-half): {noise_ceiling:.3f}  (p={pval:.4g})")
print("\nCompare this to your Stage 3 model-vs-brain correlations:")
print("  layer1: 0.034   layer2: 0.102   layer3: 0.098   layer4: 0.128")
print(f"\nIf your model scores are approaching {noise_ceiling:.3f}, the model is "
      f"capturing nearly all of the reliable signal available -- a genuinely good result.")
print("If your model scores are far below it, there's real room for the model "
      "to improve, and the gap is meaningful, not just noise.")
