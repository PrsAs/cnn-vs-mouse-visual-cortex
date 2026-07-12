"""
Visualize what you have so far, before diving into Stage 3.

Run in the same environment:
    conda activate visualmodel
    python visualize_progress.py

This produces 3 figures saved as PNGs:
  1. sample_images.png   - a few of the actual images the mouse saw
  2. neural_activity.png - heatmap of real neuron responses across images
  3. model_activity.png  - heatmap of ResNet-18 layer activations across images

The point: SEE that you have two "activity fingerprints" for the same set
of images (one biological, one artificial) before you start comparing them
mathematically in Stage 3.
"""

import numpy as np
import matplotlib.pyplot as plt

stage1 = np.load("stage1_output.npz", allow_pickle=True)
stage2 = np.load("stage2_output.npz", allow_pickle=True)

scenes = stage1["scenes"]
dff_traces = stage1["dff_traces"]

# ---- Figure 1: a handful of actual stimulus images ----
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(scenes[i], cmap="gray")
    ax.set_title(f"Image {i}")
    ax.axis("off")
fig.suptitle("Sample natural scene images shown to the mouse")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
plt.close()
print("Saved sample_images.png")

# ---- Figure 2: neural activity heatmap ----
# dff_traces is (n_neurons x n_timepoints). For a quick look, just show
# a chunk of raw activity — a proper per-image-averaged version comes in Stage 3.
n_neurons_to_show = min(50, dff_traces.shape[0])
n_timepoints_to_show = min(1000, dff_traces.shape[1])

plt.figure(figsize=(12, 5))
plt.imshow(
    dff_traces[:n_neurons_to_show, :n_timepoints_to_show],
    aspect="auto", cmap="viridis"
)
plt.colorbar(label="dF/F (activity level)")
plt.xlabel("Time (frames)")
plt.ylabel("Neuron")
plt.title(f"Real neural activity — first {n_neurons_to_show} neurons, "
          f"first {n_timepoints_to_show} frames")
plt.tight_layout()
plt.savefig("neural_activity.png", dpi=150)
plt.close()
print("Saved neural_activity.png")

# ---- Figure 3: model activation heatmap, one layer as an example ----
layer_name = "layer2"  # a middle layer — interesting middle ground
layer_acts = stage2[layer_name]  # (n_images x n_channels)

plt.figure(figsize=(12, 5))
plt.imshow(layer_acts.T, aspect="auto", cmap="magma")
plt.colorbar(label="Activation strength")
plt.xlabel("Image")
plt.ylabel(f"Channel ({layer_name})")
plt.title(f"ResNet-18 {layer_name} activations across images")
plt.tight_layout()
plt.savefig("model_activity.png", dpi=150)
plt.close()
print("Saved model_activity.png")

print("\nOpen the 3 PNG files to see: real images -> real neural response "
      "pattern -> model's internal response pattern for the same images.")
