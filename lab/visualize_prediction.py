import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from pycocotools import mask as coco_mask
from torchvision.utils import save_image
import torchvision.transforms as T
import os
from collections import defaultdict
from scipy.ndimage import binary_dilation
import sys

OBJ_IDS = {
    "icbin": [1, 2],
    "ycbv": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "tudl": [1, 2, 3],
    "lmo": [1, 5, 6, 8, 9, 10, 11, 12],
    "tless": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    "itodd": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    "hb": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
}


def generate_distinguishable_colors(num_classes):
    cmap = plt.cm.get_cmap('hsv', num_classes)
    return [cmap(i)[:3] for i in range(num_classes)]

# Function to calculate IoU
def calculate_iou(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return intersection / union if union > 0 else 0


def find_and_convert_images(directory, prefix):
    # Transform to convert images to PyTorch tensors
    transform = T.Compose([
        T.ToTensor(),  # Convert image to tensor
    ])

    # List to store tensors
    image_tensors = []

    # Walk through the directory
    for filename in os.listdir(directory):
        # Check if the file starts with the specific prefix
        if filename.startswith(prefix) and filename.lower().endswith(
            ("png", "jpg", "jpeg", "bmp", "tif", "tiff")
        ):
            # Construct full file path
            file_path = os.path.join(directory, filename)

            # Open the image
            with Image.open(file_path) as img:
                # Apply transformations and convert to tensor
                tensor = transform(img)
                image_tensors.append(tensor)

    return torch.stack(image_tensors)

if len(sys.argv) < 2:
    print("Usage: python script.py <dataset_name>")
    sys.exit(1)

dataset_name = sys.argv[1]

if dataset_name not in OBJ_IDS:
    print(f"Error: Unsupported dataset_name '{dataset_name}'")
    sys.exit(1)


detection_path = "log/self_grounding_CW/"
data_path = "/export/datasets/public/3d/BOP"
dataset_path = os.path.join(data_path, dataset_name)
test_dir = "test"

save_dir = f"log/vis/{dataset_name}"
os.makedirs(save_dir, exist_ok=True)

if dataset_name in ["hb", "tless"]:
    test_dir = "test_primesense"

rgb_dir = "rgb"
extention = ".png"
if dataset_name == "itodd":
    rgb_dir = "gray"
    extention = ".tif"
    
num_classes = len(OBJ_IDS[dataset_name])

# Colors for each class (up to 20 classes for demonstration)
colors = generate_distinguishable_colors(num_classes)
# Load COCO format predictions
with open(os.path.join(detection_path, f"result_{dataset_name}.json"), "r") as f:
    predictions = json.load(f)

# Group predictions by scene_id and image_id
grouped_predictions = defaultdict(list)
for pred in predictions:
    grouped_predictions[(pred["scene_id"], pred["image_id"])].append(pred)

# Iterate over grouped predictions
for (scene_id, image_id), preds in grouped_predictions.items():
    scene_path = os.path.join(dataset_path, test_dir, str(scene_id).zfill(6))

    scene_image = np.array(
        Image.open(os.path.join(scene_path, rgb_dir, str(image_id).zfill(6) + extention))
    )

    # Example ground truth mask (PyTorch tensor of shape (B, 1, h, w))
    gt_masks = find_and_convert_images(
        os.path.join(scene_path, "mask_visib"), str(image_id).zfill(6)
    )
    B, _, H, W = gt_masks.shape

    # Create a blank canvas for visualization
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Process each prediction in the group
    for pred in preds:
        category_id = pred["category_id"]
        segmentation = pred["segmentation"]

        # Convert RLE to mask using pycocotools
        pred_mask = coco_mask.decode(segmentation).astype(bool)

        # Resize predicted mask to ground truth size if necessary
        pred_mask_resized = np.array(
            Image.fromarray(pred_mask).resize((W, H), Image.NEAREST), dtype=bool
        )

        # Extract edges using binary dilation
        edges = binary_dilation(pred_mask_resized) & ~pred_mask_resized

        # Compare with ground truth
        for b in range(B):
            gt_mask = gt_masks[b, 0].numpy().astype(bool)
            iou = calculate_iou(pred_mask_resized, gt_mask)

            if iou > 0.5:
                # Assign the same color to masks of the same class
                color = (np.array(colors[category_id % len(colors)]) * 255).astype(np.uint8)
                canvas[pred_mask_resized] = color

                # Highlight edges with higher intensity
                edge_color = np.clip(color * 1.5, 0, 255).astype(np.uint8)
                canvas[edges] = edge_color

    # Overlay the canvas on the original scene image
    scene_mask_coef = ((canvas > 0) * 0.4)
    scene_mask_coef[scene_mask_coef == 0] = 1
    if dataset_name == "itodd":
        scene_image = scene_image[:, :, np.newaxis].repeat(3, axis=2)
    superimposed_image = (scene_mask_coef * scene_image + 0.6 * canvas).astype(np.uint8)

    # Save the result
    save_image(
        torch.tensor(superimposed_image / 255).permute(2, 0, 1).float(),
        os.path.join(
            save_dir, f"{str(scene_id).zfill(6)}_{str(image_id).zfill(6)}.jpg"
        ),
    )
    print(f"Saved {str(scene_id).zfill(6)}_{str(image_id).zfill(6)}.jpg")
