"""
Stage 2: Extract model activations for the same images the mouse saw.

Run this in the same conda environment as Stage 1:
    conda activate visualmodel
    pip install torch torchvision   (if not already installed)
    python stage2_extract_activations.py

What this does:
  1. Loads the natural scene images you saved in Stage 1 (stage1_output.npz)
  2. Feeds each one through a pretrained ResNet-18
  3. Records the internal activity at 4 depths of the network (early -> late)
  4. Saves one activation vector per image per layer, so Stage 3 can compare
     these to the mouse's actual neural responses to the same images.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---- Step 1: load the images you pulled from the Allen dataset ----
data = np.load("stage1_output.npz", allow_pickle=True)
scenes = data["scenes"]  # shape: (n_images, height, width), grayscale
print(f"Loaded {scenes.shape[0]} images, each {scenes.shape[1]}x{scenes.shape[2]}")

# ---- Step 2: load a pretrained ResNet-18 ----
# Handles both old and new torchvision APIs
try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
except AttributeError:
    model = models.resnet18(pretrained=True)

model.eval()  # inference mode — we're not training anything

# ---- Step 3: set up hooks to "listen in" on 4 depths of the network ----
# early (layer1) -> mid (layer2, layer3) -> late (layer4)
activations = {}

def make_hook(name):
    def hook(module, input, output):
        # average-pool each feature map down to a manageable vector
        # (otherwise layer4 alone would be huge per image)
        pooled = torch.mean(output, dim=[2, 3])  # (batch, channels)
        activations[name] = pooled.detach().cpu().numpy()
    return hook

layers_to_probe = {
    "layer1": model.layer1,
    "layer2": model.layer2,
    "layer3": model.layer3,
    "layer4": model.layer4,
}
for name, layer in layers_to_probe.items():
    layer.register_forward_hook(make_hook(name))

# ---- Step 4: preprocess images the way ResNet-18 expects ----
# ImageNet-pretrained models expect: 3-channel RGB, 224x224, normalized
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Allen images are grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- Step 5: run every image through the model, one at a time ----
all_layer_activations = {name: [] for name in layers_to_probe}

with torch.no_grad():
    for i in range(scenes.shape[0]):
        img = scenes[i].astype(np.uint8)
        img_tensor = preprocess(img).unsqueeze(0)  # add batch dimension
        _ = model(img_tensor)  # triggers the hooks above

        for name in layers_to_probe:
            all_layer_activations[name].append(activations[name][0])

        if (i + 1) % 20 == 0 or i == scenes.shape[0] - 1:
            print(f"  processed {i + 1}/{scenes.shape[0]} images")

# ---- Step 6: stack into arrays and save ----
final_activations = {
    name: np.stack(vectors) for name, vectors in all_layer_activations.items()
}

for name, arr in final_activations.items():
    print(f"{name}: {arr.shape}  (n_images x n_channels)")

np.savez("stage2_output.npz", **final_activations)
print("\nSaved to stage2_output.npz — ready for Stage 3 (comparing to real neural data).")
