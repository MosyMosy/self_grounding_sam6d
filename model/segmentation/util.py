import torch
import torchvision
import torch.nn.functional as F
# from scipy.ndimage import label
import numpy as np


# This code is copied from SegmentAnything library at https://github.com/facebookresearch/segment-anything
def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / unions


def mask_iou(mask1, mask2):
    assert mask1.shape == mask2.shape, "Input masks must have the same spatial shape."
    assert (
        mask1.dtype == torch.uint8 and mask2.dtype == torch.uint8
    ), "Input masks must be binary."
    """Calculate the Intersection over Union (IoU) between two binary masks."""
    intersection = (mask1 & mask2).float().sum(dim=(1, 2))
    union = (mask1 | mask2).float().sum(dim=(1, 2))
    return intersection / union


def mask_miou_batch(mask1, mask2):
    """
    Calculate mean IoU between two sets of binary segmentation masks.

    Args:
        mask1 (torch.Tensor): Binary segmentation mask of shape (B1, h, w).
        mask2 (torch.Tensor): Binary segmentation mask of shape (B2, h, w).

    Returns:
        torch.Tensor: mIoU of shape (B1, B2).
    """
    assert (
        mask1.shape[-2:] == mask2.shape[-2:]
    ), "Input masks must have the same spatial shape."
    assert (mask1.dtype == torch.uint8 or mask1.dtype == torch.bool) and (
        mask2.dtype == torch.uint8 or mask2.dtype == torch.bool
    ), "Input masks must be binary."

    # Expand dimensions to enable broadcasting
    mask1 = mask1.unsqueeze(1)  # Shape: (B1, 1, h, w)
    mask2 = mask2.unsqueeze(0)  # Shape: (1, B2, h, w)

    # Calculate intersection
    intersection = (mask1 * mask2).sum(dim=(-1, -2))  # Shape: (B1, B2)

    # Calculate union as: area(mask1) + area(mask2) - intersection
    area1 = mask1.sum(dim=(-1, -2))  # Shape: (B1, 1)
    area2 = mask2.sum(dim=(-1, -2))  # Shape: (1, B2)
    union = area1 + area2 - intersection  # Shape: (B1, B2)

    # Compute IoU
    iou = intersection / (union + 1e-6)  # Adding epsilon to avoid division by zero

    return iou


def mask_nms(masks, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) on segmentation masks in parallel.

    Args:
    - masks: A tensor of shape (N, H, W) containing the binary masks.
    - scores: A tensor of shape (N,) containing the scores for each mask.
    - iou_threshold: IoU threshold for deciding when to suppress masks.

    Returns:
    - keep: Indices of masks to keep after applying NMS.
    """
    # Sort masks by scores in descending order
    idxs = scores.argsort(descending=True)

    # Compute pairwise IoU for all masks
    ious = mask_miou_batch(masks[idxs], masks[idxs])  # Shape: (N, N)

    # Initialize a tensor to mark kept masks
    keep = torch.ones(len(idxs), dtype=torch.bool, device=scores.device)
    # Iterate through each mask and suppress others based on IoU threshold
    for i in range(len(idxs)):
        if keep[i]:  # If the mask is not suppressed yet
            # Suppress masks with IoU greater than the threshold
            keep[i + 1 :] &= ious[i, i + 1 :] < iou_threshold

    # Return the indices of the kept masks
    return idxs[keep]


class BboxNMS:
    def __init__(self, nms_thr: float = 0.5):
        self.nms_thr = nms_thr

    def __call__(self, masks: torch.Tensor, bboxes: torch.Tensor, scores: torch.Tensor):
        if len(scores) == 0:
            return bboxes.new_zeros(0)

        keep_idxs = torchvision.ops.nms(bboxes.float(), scores, self.nms_thr)
        return keep_idxs


# This code is adapted from https://github.com/dbolya/yolact
class MatrixNMS:
    def __init__(
        self, nms_thr: float = 0.5, sigma: float = 2.0, method: str = "gaussian"
    ):
        self.nms_thr = nms_thr
        self.sigma = sigma
        self.method = method

    def __call__(self, masks: torch.Tensor, bboxes: torch.Tensor, scores: torch.Tensor):
        if len(scores) == 0:
            return masks.new_zeros(0)

        masks = masks.view(len(scores), -1).float()
        intersection = torch.mm(masks, masks.t())
        area = masks.sum(dim=1).view(-1, 1)
        union = area + area.t() - intersection
        iou = intersection / union

        iou = iou.triu(diagonal=1)
        max_iou, _ = iou.max(dim=0)
        if self.method == "gaussian":
            decay = torch.exp(-self.sigma * (max_iou**2))
        elif self.method == "linear":
            decay = 1 - max_iou
        scores = scores * decay

        keep_idxs = (scores > self.nms_thr).nonzero().flatten()
        return keep_idxs


#  this code is adapted from https://github.com/open-mmlab/mmdetection/
class MaskNMS:
    def __init__(self, nms_thr: float = 0.5):
        self.nms_thr = nms_thr

    def __call__(self, masks: torch.Tensor, bboxes: torch.Tensor, scores: torch.Tensor):
        if len(scores) == 0:
            return masks.new_zeros(0)

        # Calculate IoU matrix
        iou = MaskNMS.mask_overlaps(
            masks.unsqueeze(0).float(), masks.unsqueeze(0).float(), mode="iou"
        )
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        keep_idxs = (iou_max[0] <= self.nms_thr).nonzero().flatten()
        return keep_idxs

    @staticmethod
    def mask_overlaps(masks1, masks2, mode="iou", is_aligned=False, eps=1e-6):
        """
        Calculate the mask IoU between two sets of masks.

        Args:
            masks1 (Tensor): shape (B, m, H, W), where B is batch size,
                            m is the number of masks, and H, W are height and width of masks.
            masks2 (Tensor): shape (B, n, H, W), where B is batch size,
                            n is the number of masks, and H, W are height and width of masks.
            mode (str): 'iou' (Intersection over Union) or 'iof' (Intersection over Foreground).
                        Default is 'iou'.
            is_aligned (bool, optional): If True, then m and n must be equal.
                                        Default False.
            eps (float, optional): A value added to the denominator for numerical stability. Default 1e-6.

        Returns:
            Tensor: shape (m, n) if `is_aligned` is False else shape (m,).
        """

        assert mode in ["iou", "iof"], f"Unsupported mode {mode}"
        assert (
            masks1.dim() == 4 and masks2.dim() == 4
        ), "Masks should be in the format (B, m/n, H, W)"

        batch_shape = masks1.shape[:-3]  # For batch dimensions

        rows = masks1.size(-3)
        cols = masks2.size(-3)

        if is_aligned:
            assert (
                rows == cols
            ), "For aligned mode, masks1 and masks2 should have the same number of masks"

        if rows * cols == 0:
            if is_aligned:
                return masks1.new(batch_shape + (rows,))
            else:
                return masks1.new(batch_shape + (rows, cols))

        # Flatten masks (B, m/n, H*W)
        masks1_flat = masks1.view(masks1.size(0), masks1.size(1), -1)
        masks2_flat = masks2.view(masks2.size(0), masks2.size(1), -1)

        if is_aligned:
            # Intersection: logical AND between masks (B, m, H*W)
            intersection = (masks1_flat * masks2_flat).sum(dim=-1)

            # Area of each mask
            area1 = masks1_flat.sum(dim=-1)
            area2 = masks2_flat.sum(dim=-1)

            if mode == "iou":
                union = area1 + area2 - intersection
            else:
                union = area1

            ious = intersection / (union + eps)
            return ious

        else:
            # Intersection: logical AND between all pairs of masks (B, m, n, H*W)
            intersection = torch.einsum("bmh,bnh->bmn", masks1_flat, masks2_flat)

            # Area of each mask
            area1 = masks1_flat.sum(dim=-1).unsqueeze(2)
            area2 = masks2_flat.sum(dim=-1).unsqueeze(1)

            if mode == "iou":
                union = area1 + area2 - intersection
            else:
                union = area1

            ious = intersection / (union + eps)
            return ious


class ConfluenceNMS:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def __call__(
        self, masks: torch.Tensor, seg_bbox: torch.Tensor, scores: torch.Tensor
    ):
        if len(scores) == 0:
            return masks.new_zeros(0)

        # Sort masks by score (highest first)
        indices = torch.argsort(scores, descending=True)
        masks = masks[indices]
        scores = scores[indices]

        # Compute the mIoU between all pairs of masks
        ious = mask_miou_batch(masks, masks)  # Shape: (N, N)

        # Find pairs of masks with IoU > iou_threshold
        to_merge = ious > self.iou_threshold  # Shape: (N, N)

        keep_idxs = []
        processed = torch.zeros(masks.shape[0], dtype=torch.bool)

        # Start processing masks in parallel
        for i in range(masks.shape[0]):
            if processed[i]:
                continue

            overlapping_indices = torch.where(to_merge[i])[0]
            if len(overlapping_indices) > 0:
                # Merge all overlapping masks (take the union)
                # Take the highest score of the overlapping masks
                keep_idxs.append(
                    indices[i]
                )  # Keep the index of the mask that will be retained

                # Mark these indices as processed
                processed[overlapping_indices] = True
            else:
                keep_idxs.append(
                    indices[i]
                )  # Keep the index of the current mask if no overlap

        return torch.tensor(keep_idxs)


def clean_small_articles(
    masks: torch.Tensor, kernel_size: int = 14, threshold: float = 0.3
):
    """
    Removes small regions from a batch of binary segmentation masks using convolutional filtering.

    Args:
        masks (torch.Tensor): Binary masks of shape (B, H, W) with values 0 or 1.
        kernel_size (int): Size of the convolution kernel (must be odd).
        threshold (int): Minimum number of neighboring 'true' (1) pixels required to keep a pixel as true.

    Returns:
        torch.Tensor: Cleaned masks with small regions removed, shape (B, H, W).
    """
    assert (
        kernel_size % 2 == 1
    ), "Kernel size must be odd to maintain spatial dimensions."

    # Create a kernel with all ones (a square kernel)
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size), dtype=torch.float32, device=masks.device
    )

    # Add a channel dimension to the masks (batch size remains the same)
    masks = masks.unsqueeze(1).float()  # Shape (B, 1, H, W)

    # Perform the convolution to count neighboring true values for each mask in the batch
    # Use 'same' padding to maintain the same spatial dimensions
    convolved = F.conv2d(masks, kernel, padding=kernel_size // 2)

    # Apply the threshold: keep pixels with enough neighboring true values
    cleaned_masks = (
        (convolved >= (threshold * kernel_size**2)).squeeze(1).int()
    )  # Shape (B, H, W)

    filtered_masks = masks.squeeze(1).int() & cleaned_masks

    return filtered_masks


def seg_mask_max_size_fileter(seg_masks, depth, cam_scale, cam_K, max_feasible_size):
    seg_masks_bboxes = get_bounding_boxes_batch(seg_masks.squeeze(1))
    # seg_masks_sizes = calculate_bbox_size_from_xyz_map(test_image_xyz, seg_masks_bboxes)
    seg_masks_sizes = bbox_size_in_world_coordinates_with_center_depth(
        seg_masks_bboxes, depth.squeeze(1), cam_scale, cam_K
    )
    seg_mask_feasible = (seg_masks_sizes[:, 0] <= max_feasible_size) & (
        seg_masks_sizes[:, 1] <= max_feasible_size
    )

    return seg_masks_sizes, seg_mask_feasible


def calculate_bbox_size_from_xyz_map(xyz_map, bboxes):
    """
    Calculates the size of each bounding box in terms of x, y, z coordinates.

    Parameters:
    - xyz_map: A tensor of shape (1, 3, h, w), representing the XYZ coordinates of each pixel
               The first dimension corresponds to the x, y, and z channels.
    - bboxes: A tensor of shape (n, 4), where each bbox is (x_min, y_min, x_max, y_max)

    Returns:
    - bbox_sizes: A tensor of shape (n, 3), where each row is the size (dx, dy, dz) of the bounding box
    """
    n = bboxes.shape[0]
    bbox_sizes = torch.zeros(
        (n, 3), dtype=torch.float32, device=xyz_map.device
    )  # To store sizes (dx, dy, dz)

    for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes):
        # Extract the region inside the bounding box from the XYZ map
        xyz_region = xyz_map[0, :, y_min:y_max, x_min:x_max]

        # Calculate the size of the bounding box in terms of x, y, z
        x_size = xyz_region[0].max() - xyz_region[0].min()  # X range (dx)
        y_size = xyz_region[1].max() - xyz_region[1].min()  # Y range (dy)
        z_size = xyz_region[2].max() - xyz_region[2].min()  # Z range (dz)

        bbox_sizes[i] = torch.tensor([x_size, y_size, z_size])

    return bbox_sizes


# def split_connected_regions(seg_masks, scores):
#     """
#     Splits each segmentation mask into multiple masks containing only one connected region and adjusts the scores.

#     Parameters:
#     - seg_masks: A tensor of shape (n, h, w) where each element is a binary segmentation mask.
#     - scores: A tensor of shape (n,) where each element is the score for the corresponding segmentation mask.

#     Returns:
#     - all_masks: A tensor of shape (m, h, w), where m is the total number of connected regions across all masks.
#     - all_scores: A tensor of shape (m,), where each score corresponds to a new connected region.
#     """
#     n, h, w = seg_masks.shape
#     all_masks = []
#     all_scores = []

#     for i in range(n):
#         mask_np = seg_masks[i].cpu().numpy()  # Convert to NumPy for processing

#         # Ensure mask is binary (0 or 1)
#         mask_np = mask_np.astype(np.uint8)

#         # Label connected components
#         labeled_mask, num_features = label(mask_np)

#         if num_features == 1:
#             # If only one connected component, retain the original mask and score
#             all_masks.append(seg_masks[i])
#             all_scores.append(scores[i])
#         else:
#             # Get total area of the original mask (sum of 1s)
#             original_area = mask_np.sum()

#             # Create a separate mask for each connected component
#             for label_idx in range(1, num_features + 1):
#                 new_mask = (labeled_mask == label_idx).astype(np.uint8)
#                 new_mask_tensor = torch.tensor(
#                     new_mask, dtype=torch.uint8, device=seg_masks.device
#                 )

#                 # Calculate the area of the new mask
#                 new_area = new_mask_tensor.sum().item()

#                 # Adjust the score based on the area ratio
#                 new_score = scores[i] * (new_area / original_area)

#                 all_masks.append(new_mask_tensor)
#                 all_scores.append(new_score)

#     return torch.stack(all_masks), torch.tensor(
#         all_scores, dtype=torch.float32, device=seg_masks.device
#     )


def get_bounding_box(mask):
    """
    Get the bounding box coordinates of a single segmentation mask.

    Parameters:
    mask (torch.Tensor): A binary segmentation mask of shape (H, W)

    Returns:
    bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max)
    """
    # Find the indices of non-zero elements
    non_zero_indices = torch.nonzero(mask)

    if non_zero_indices.numel() == 0:
        return torch.tensor(
            [0, 0, 2, 2], device=mask.device
        )  # If the mask is empty, return a zero-size bounding box

    # Get the min and max coordinates
    y_min, x_min = torch.min(non_zero_indices, dim=0)[0]
    y_max, x_max = torch.max(non_zero_indices, dim=0)[0]

    return torch.stack([x_min, y_min, x_max, y_max])


def get_bounding_boxes_batch(masks):
    """
    Get bounding box coordinates for a batch of segmentation masks.

    Parameters:
    masks (torch.Tensor): A batch of binary segmentation masks of shape (B, H, W)

    Returns:
    bboxes: tensor of bounding box coordinates for each mask
    """
    bboxes = []
    for mask in masks:
        bbox = get_bounding_box(mask)
        bboxes.append(bbox)

    return torch.stack(bboxes)


def labeled_image_to_masks(image):
    unique_values, inverse_indices = torch.unique(image, return_inverse=True)

    # Get the number of unique values
    v = unique_values.size(0)

    # Reshape the inverse indices to match the original tensor's shape
    inverse_indices = inverse_indices.view(image.shape)

    # Create a one-hot encoded tensor of shape (v, W, H)
    one_hot_tensor = torch.nn.functional.one_hot(inverse_indices, num_classes=v).to(
        image.device
    )

    # Permute the dimensions to get the desired shape (v, W, H)
    output_tensor = one_hot_tensor.permute(2, 0, 1)
    return output_tensor





def build_point_grid(n_per_side: int) -> torch.Tensor:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1] using PyTorch."""
    # Compute offset and evenly spaced points along one side
    offset = 1 / (2 * n_per_side)
    points_one_side = torch.linspace(offset, 1 - offset, n_per_side)

    # Create grid using meshgrid and stack the coordinates
    points_x, points_y = torch.meshgrid(points_one_side, points_one_side, indexing="ij")

    # Stack the points and reshape into (n_per_side^2, 2)
    points = torch.stack([points_x, points_y], dim=-1).reshape(-1, 2)

    return points


def bbox_size_in_world_coordinates_with_center_depth(bboxes, depth_map, scale, K):
    """
    Calculate bounding box sizes in world coordinates using the center depth of each bounding box.

    Parameters:
    - bboxes: A tensor of shape (n, 4) where each row is (x_min, y_min, x_max, y_max) in image coordinates (pixels).
    - depth_map: A tensor of shape (B, H, W) representing the depth for each pixel.
    - scale: A tensor of shape (B,) representing the scale factor for the depth map.
    - K: A tensor of shape (B, 3, 3) representing the camera intrinsic matrix.

    Returns:
    - bbox_sizes_world: A tensor of shape (n, 3) representing the width, height, and depth of each bounding box in world coordinates.
    """
    B, H, W = depth_map.shape
    n = bboxes.shape[0]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    bbox_sizes_world = []

    for i in range(n):
        x_min, y_min, x_max, y_max = bboxes[i]

        # Calculate the center of the bbox (handle odd and even dimensions)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Retrieve the center depth value from the depth map
        Z_center = (
            depth_map[0, center_y, center_x] * scale.item() / 1000
        )  # Convert mm to meters if needed

        # Convert bbox corners to world coordinates using the center depth
        X_min = (x_min - cx) * Z_center / fx
        Y_min = (y_min - cy) * Z_center / fy
        X_max = (x_max - cx) * Z_center / fx
        Y_max = (y_max - cy) * Z_center / fy

        # Compute width, height, and depth in world coordinates
        width_world = X_max - X_min
        height_world = Y_max - Y_min
        depth_world = Z_center  # Constant depth for the bbox

        bbox_sizes_world.append(
            [width_world.item(), height_world.item(), depth_world.item()]
        )

    return torch.tensor(bbox_sizes_world)


def confluence_algorithm_batch_parallel(masks, scores, iou_threshold=0.5):
    """
    Apply the confluence algorithm in parallel to a batch of segmentation masks using batch mIoU.

    Parameters:
        masks (torch.Tensor): Binary segmentation masks of shape (N, H, W)
        scores (torch.Tensor): Confidence scores for each mask, shape (N,)
        iou_threshold (float): IoU threshold to merge masks.

    Returns:
        torch.Tensor: New set of masks after merging.
        torch.Tensor: Updated scores after merging.
    """
    # Sort masks by score (highest first)
    indices = torch.argsort(scores, descending=True)
    masks = masks[indices]
    scores = scores[indices]

    # Compute the mIoU between all pairs of masks
    ious = mask_miou_batch(masks, masks)  # Shape: (N, N)

    # Find pairs of masks with IoU > iou_threshold
    to_merge = ious > iou_threshold  # Shape: (N, N)

    # Create a list to hold the merged mask for each group of masks
    merged_masks = []
    merged_scores = []

    # Start processing masks in parallel
    for i in range(masks.shape[0]):
        if to_merge[i].sum() > 0:  # Check if mask i overlaps with any others
            overlapping_indices = torch.where(to_merge[i])[0]

            # Merge all overlapping masks (take the union)
            merged_mask = torch.any(
                masks[overlapping_indices], dim=0
            )  # Union of all overlapping masks

            # Take the highest score of the overlapping masks
            merged_score = scores[overlapping_indices].max()

            # Append to the list of merged masks
            merged_masks.append(merged_mask)
            merged_scores.append(merged_score)

            # Set the overlapping masks as processed
            to_merge[:, overlapping_indices] = (
                False  # Disable further merging for these masks
            )

    merged_masks = torch.stack(merged_masks, dim=0)
    merged_scores = torch.tensor(merged_scores)

    return merged_masks, merged_scores


def remove_very_small_detections(masks, boxes, min_box_size=0.05, min_mask_size=3e-4):
    img_area = masks.shape[1] * masks.shape[2]
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / img_area
    mask_areas = masks.sum(dim=(1, 2)) / img_area
    keep_idxs = torch.logical_and(
        box_areas > min_box_size**2, mask_areas > min_mask_size
    )
    return keep_idxs
