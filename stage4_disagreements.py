"""
Stage 4: Find where the model and the brain disagree.

Run in the same environment:
    conda activate visualmodel
    python stage4_disagreements.py

What this does:
  1. Loads the neural RDM and the layer4 model RDM from Stage 3
  2. For every pair of images, computes the "disagreement" -- cases where
     the model says two images are very similar but the brain says they're
     very different, or vice versa
  3. Shows you the top disagreeing pairs, side by side, so you can actually
     LOOK at what kind of visual content trips up the alignment
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

stage1 = np.load("stage1_output.npz", allow_pickle=True)
scenes = stage1["scenes"]
stim_table = stage1["stim_table"]

neural_rdm = np.load("rdm_neural.npy")
model_rdm = np.load("rdm_layer4.npy")   # your best-performing layer from Stage 3

image_ids = np.unique(stim_table["frame"])
image_ids = image_ids[image_ids >= 0].astype(int)

# ---- Step 1: put both RDMs on the same scale ----
# Correlation distances from the two RDMs aren't directly comparable in
# absolute terms (different units of "distance"), so z-score each one first.
# This makes "1 standard deviation more similar than average" comparable
# across the two systems.
triu_idx = np.triu_indices_from(neural_rdm, k=1)

neural_vals_z = zscore(neural_rdm[triu_idx])
model_vals_z = zscore(model_rdm[triu_idx])

# ---- Step 2: compute disagreement for every pair ----
# Large positive difference: model says "similar" (low z) while brain says
# "different" (high z), or the reverse -- either direction is a disagreement.
disagreement = model_vals_z - neural_vals_z

# Map the flat upper-triangle index back to (row, col) image pairs
rows, cols = triu_idx

# ---- Step 3: find the biggest disagreements in both directions ----
n_show = 5
biggest_gap_idx = np.argsort(np.abs(disagreement))[::-1][:n_show]

print(f"Top {n_show} image pairs where model and brain disagree most:\n")
for rank, i in enumerate(biggest_gap_idx):
    img_a, img_b = rows[i], cols[i]
    direction = "model says SIMILAR, brain says DIFFERENT" if disagreement[i] < 0 \
        else "model says DIFFERENT, brain says SIMILAR"
    print(f"{rank+1}. Images {image_ids[img_a]} & {image_ids[img_b]}  "
          f"({direction}, gap={disagreement[i]:.2f} SD)")

# ---- Step 4: plot these pairs side by side so you can actually look ----
fig, axes = plt.subplots(n_show, 2, figsize=(6, 3 * n_show))
for rank, i in enumerate(biggest_gap_idx):
    img_a_idx, img_b_idx = rows[i], cols[i]
    real_img_a = image_ids[img_a_idx]
    real_img_b = image_ids[img_b_idx]

    axes[rank, 0].imshow(scenes[real_img_a], cmap="gray")
    axes[rank, 0].set_title(f"Image {real_img_a}")
    axes[rank, 0].axis("off")

    axes[rank, 1].imshow(scenes[real_img_b], cmap="gray")
    axes[rank, 1].set_title(f"Image {real_img_b}")
    axes[rank, 1].axis("off")

fig.suptitle("Top disagreements: model vs. brain similarity judgments", y=1.0)
plt.tight_layout()
plt.savefig("stage4_disagreements.png", dpi=150)
plt.close()

print("\nSaved stage4_disagreements.png -- look at these pairs and ask yourself:")
print("  - Do they share obvious visual features (texture, color, layout)?")
print("  - Does one image have an object/animal and the other doesn't?")
print("  - Is there a pattern across all 5 pairs, or are they idiosyncratic?")
print("\nThis qualitative pattern -- not just the correlation number -- is your")
print("actual interpretability finding for the write-up.")
