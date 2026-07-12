"""
Compare multiple model architectures against the same neural data.

Run in the same environment:
    conda activate visualmodel
    python stage3b_multi_model.py

What this does:
  1. Runs the SAME images through several different pretrained architectures
     (ResNet-18, ResNet-50, VGG16, AlexNet)
  2. For each one, extracts activations at ~4 relative depths (25%, 50%,
     75%, 100% of the way through the network) so architectures of very
     different lengths are still comparable
  3. Computes RSA correlation against the real neural data for every
     model x depth combination
  4. Plots all models on one chart so you can see whether "gets more
     brain-like with depth" is a ResNet-18 quirk or a general pattern

This takes longer to run than Stage 2/3 since it's now 4 models instead
of 1 -- expect several minutes depending on your machine.
"""

import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from scipy.stats import spearmanr, zscore
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

stage1 = np.load("stage1_output.npz", allow_pickle=True)
scenes = stage1["scenes"]
dff_traces = stage1["dff_traces"]
stim_table = stage1["stim_table"]

# ---- Rebuild neural RDM (same as Stage 3) ----
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

# ---- Image preprocessing (same for all models -- all expect ImageNet-style input) ----
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

# ---- Define which models to compare, and 4 evenly-spaced layers for each ----
# These layer names come from inspecting get_graph_node_names(model) for each
# architecture -- picked to sit at roughly 25/50/75/100% depth.
MODEL_CONFIGS = {
    "resnet18": {
        "builder": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "layers": {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4"},
    },
    "resnet50": {
        "builder": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "layers": {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4"},
    },
    "vgg16": {
        "builder": lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT),
        "layers": {"features.9": "block1", "features.16": "block2",
                   "features.23": "block3", "features.30": "block4"},
    },
    "alexnet": {
        "builder": lambda: models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
        "layers": {"features.2": "stage1", "features.5": "stage2",
                   "features.7": "stage3", "features.12": "stage4"},
    },
}

results = {}  # results[model_name][layer_name] = correlation

for model_name, config in MODEL_CONFIGS.items():
    print(f"\n--- {model_name} ---")
    model = config["builder"]()
    model.eval()

    return_nodes = config["layers"]              # {model_node_name: output_key}
    output_names = list(return_nodes.values())    # what out[...] will actually be keyed by
    extractor = create_feature_extractor(model, return_nodes=return_nodes)

    all_acts = {name: [] for name in output_names}

    with torch.no_grad():
        batch_size = 16
        for start in range(0, image_tensors.shape[0], batch_size):
            batch = image_tensors[start:start + batch_size]
            out = extractor(batch)
            for name in output_names:
                feat = out[name]
                if feat.dim() == 4:  # conv output: (batch, channels, h, w)
                    pooled = torch.mean(feat, dim=[2, 3])
                else:  # already flat
                    pooled = feat
                all_acts[name].append(pooled.cpu().numpy())

    results[model_name] = {}
    for name in output_names:
        layer_acts_all = np.concatenate(all_acts[name], axis=0)
        layer_acts = layer_acts_all[image_ids_int]
        model_rdm = squareform(pdist(layer_acts, metric="correlation"))
        corr, _ = spearmanr(neural_rdm[triu_idx], model_rdm[triu_idx])
        results[model_name][name] = corr
        print(f"  {name}: RSA correlation = {corr:.3f}")

# ---- Plot all models together, x-axis = relative depth (0-100%) ----
plt.figure(figsize=(8, 6))
for model_name, layer_scores in results.items():
    depths = np.linspace(25, 100, len(layer_scores))  # relative depth %
    scores = list(layer_scores.values())
    plt.plot(depths, scores, marker="o", label=model_name, linewidth=2)

plt.xlabel("Relative network depth (%)")
plt.ylabel("RSA correlation with real neural data")
plt.title("Brain alignment across architectures and depth")
plt.axhline(0.228, color="gray", linestyle="--", label="noise ceiling")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("stage3b_multi_model.png", dpi=150)
plt.close()

print("\nSaved stage3b_multi_model.png")
print("\nLook for: does EVERY model get more brain-like with depth (a general")
print("pattern), or is that specific to ResNet-18 (an architecture quirk)?")
