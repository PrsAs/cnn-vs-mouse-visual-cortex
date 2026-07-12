"""
Look at more disagreement pairs (ranks 6-15) to check whether the
same-category pattern from the top 5 holds up, or was a fluke.

Run in the same environment:
    conda activate visualmodel
    python stage4b_more_pairs.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

stage1 = np.load("stage1_output.npz", allow_pickle=True)
scenes = stage1["scenes"]
stim_table = stage1["stim_table"]

neural_rdm = np.load("rdm_neural.npy")
model_rdm = np.load("rdm_layer4.npy")

image_ids = np.unique(stim_table["frame"])
image_ids = image_ids[image_ids >= 0].astype(int)

triu_idx = np.triu_indices_from(neural_rdm, k=1)
neural_vals_z = zscore(neural_rdm[triu_idx])
model_vals_z = zscore(model_rdm[triu_idx])
disagreement = model_vals_z - neural_vals_z

rows, cols = triu_idx

# ---- Change this to look at any range, e.g. (5, 15) for ranks 6-15 ----
RANK_START, RANK_END = 5, 15

sorted_idx = np.argsort(np.abs(disagreement))[::-1]
batch_idx = sorted_idx[RANK_START:RANK_END]

print(f"Disagreement pairs ranked #{RANK_START+1} to #{RANK_END}:\n")
for rank, i in enumerate(batch_idx):
    img_a, img_b = rows[i], cols[i]
    direction = "model=SIMILAR, brain=DIFFERENT" if disagreement[i] < 0 \
        else "model=DIFFERENT, brain=SIMILAR"
    print(f"{RANK_START+rank+1}. Images {image_ids[img_a]} & {image_ids[img_b]}  "
          f"({direction}, gap={disagreement[i]:.2f} SD)")

n_show = len(batch_idx)
fig, axes = plt.subplots(n_show, 2, figsize=(6, 3 * n_show))
for rank, i in enumerate(batch_idx):
    img_a_idx, img_b_idx = rows[i], cols[i]
    real_img_a = image_ids[img_a_idx]
    real_img_b = image_ids[img_b_idx]

    axes[rank, 0].imshow(scenes[real_img_a], cmap="gray")
    axes[rank, 0].set_title(f"Image {real_img_a}")
    axes[rank, 0].axis("off")

    axes[rank, 1].imshow(scenes[real_img_b], cmap="gray")
    axes[rank, 1].set_title(f"Image {real_img_b}")
    axes[rank, 1].axis("off")

fig.suptitle(f"Disagreement pairs, ranks {RANK_START+1}-{RANK_END}", y=1.0)
plt.tight_layout()
plt.savefig("stage4b_more_pairs.png", dpi=150)
plt.close()
print("\nSaved stage4b_more_pairs.png")
