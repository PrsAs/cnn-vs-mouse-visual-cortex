"""
Untrained network control -- does training even matter?

Run in the same environment:
    conda activate visualmodel
    python stage3d_random_baseline.py

The logic: convolutional architectures produce edge-detector-like filters
even with completely RANDOM weights, just from the math of convolution +
nonlinearity (this is a well-known phenomenon, sometimes called "random
CNN features"). This script checks how much of your brain-alignment result
is really coming from ImageNet training, versus just architecture.

If random-weight ResNet-18 scores close to trained ResNet-18 -> training
added little; the architecture alone explains most of the alignment.
If random-weight ResNet-18 scores much worse -> training specifically
pushed the model toward brain-like representations, which is the more
interesting and expected result.
"""

import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

stage1 = np.load("stage1_output.npz", allow_pickle=True)
scenes = stage1["scenes"]
dff_traces = stage1["dff_traces"]
stim_table = stage1["stim_table"]

# ---- Rebuild neural RDM (same as before) ----
image_ids = np.unique(stim_table["frame"])
image_ids = image_ids[image_ids >= 0]
image_ids_int = image_ids.astype(int)
n_neurons = dff_traces.shape[0]
response_window = 15

neural_responses = np.zeros((len(image_ids), n_neurons))
for idx, img_id in enumerate(image_ids):
    trials = stim_table[stim_table["frame"] == img_id]
    trial_responses = []
    for trial in trials:
        start = trial["start"]
        end = min(start + response_window, dff_traces.shape[1])
        trial_responses.append(dff_traces[:, start:end].mean(axis=1))
    neural_responses[idx] = np.mean(trial_responses, axis=0)

neural_rdm = squareform(pdist(neural_responses, metric="correlation"))
triu_idx = np.triu_indices_from(neural_rdm, k=1)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_tensors = torch.stack([
    preprocess(scenes[i].astype(np.uint8)) for i in range(scenes.shape[0])
])

return_nodes = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

def get_rsa_scores(model):
    model.eval()
    extractor = create_feature_extractor(model, return_nodes=return_nodes)
    all_acts = {name: [] for name in return_nodes.values()}

    with torch.no_grad():
        batch_size = 16
        for start in range(0, image_tensors.shape[0], batch_size):
            batch = image_tensors[start:start + batch_size]
            out = extractor(batch)
            for name in return_nodes.values():
                pooled = torch.mean(out[name], dim=[2, 3])
                all_acts[name].append(pooled.cpu().numpy())

    scores = {}
    for name in return_nodes.values():
        layer_acts_all = np.concatenate(all_acts[name], axis=0)
        layer_acts = layer_acts_all[image_ids_int]
        model_rdm = squareform(pdist(layer_acts, metric="correlation"))
        corr, _ = spearmanr(neural_rdm[triu_idx], model_rdm[triu_idx])
        scores[name] = corr
    return scores

print("--- Trained ResNet-18 (ImageNet weights) ---")
trained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
trained_scores = get_rsa_scores(trained_model)
for name, score in trained_scores.items():
    print(f"  {name}: {score:.3f}")

print("\n--- Untrained ResNet-18 (random weights, same architecture) ---")
random_model = models.resnet18(weights=None)  # random init, no training at all
random_scores = get_rsa_scores(random_model)
for name, score in random_scores.items():
    print(f"  {name}: {score:.3f}")

# ---- Plot both side by side ----
layer_names = list(return_nodes.values())
plt.figure(figsize=(8, 6))
plt.plot(layer_names, list(trained_scores.values()), marker="o",
         linewidth=2, label="Trained (ImageNet)", color="tab:blue")
plt.plot(layer_names, list(random_scores.values()), marker="o",
         linewidth=2, label="Untrained (random weights)", color="tab:red")
plt.axhline(0.228, color="gray", linestyle="--", label="noise ceiling")
plt.xlabel("Layer")
plt.ylabel("RSA correlation with real neural data")
plt.title("Does ImageNet training actually help, or is it just the architecture?")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("stage3d_random_baseline.png", dpi=150)
plt.close()

print("\nSaved stage3d_random_baseline.png")
gap = trained_scores["layer4"] - random_scores["layer4"]
print(f"\nTrained vs. untrained gap at layer4: {gap:.3f}")
print("A large gap = training matters a lot. A small gap = mostly architecture.")
