from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.ops import nms
import torch
from model.segmentation.util import (
    calculate_stability_score,
    get_bounding_boxes_batch as get_bounding_boxes_batch_,
)
from model.segmentation.prompt_helper import (
    generate_grid_points as generate_grid_points_,
    sim_2_point_prompts as sim_2_point_prompts_,
    generate_patch_grid_points as generate_patch_grid_points_,
)
from torchvision.ops import batched_nms as batched_nms_


class Base_Segmentation:
    def __init__(self) -> None:
        self.input_size = 1024
        self.output_channels = 256
        self.patch_size = 16

        self.name = None

    def scale_image_prompt_to_dim(self, image, point_prompt=None, bbox_prompt=None):
        H_old, W_old = image.shape[-2:]
        scale = self.input_size * 1.0 / max(W_old, H_old)
        H_new, W_new = H_old * scale, W_old * scale
        W_new = int(W_new + 0.5)
        H_new = int(H_new + 0.5)
        W_new = min(max(1, W_new), self.input_size)
        H_new = min(max(1, H_new), self.input_size)
        image_sized = torch.nn.functional.interpolate(
            image, size=(H_new, W_new), mode="bilinear", align_corners=False
        )
        if point_prompt is not None:
            # point_prompt = point_prompt * scale
            point_prompt[..., 0] = point_prompt[..., 0] * W_new
            point_prompt[..., 1] = point_prompt[..., 1] * H_new
            point_prompt.clamp_(0, self.input_size)
        if bbox_prompt is not None:
            bbox_prompt[..., 0] = bbox_prompt[..., 0] * (W_new / W_old)
            bbox_prompt[..., 2] = bbox_prompt[..., 2] * (W_new / W_old)
            bbox_prompt[..., 1] = bbox_prompt[..., 1] * (H_new / H_old)
            bbox_prompt[..., 3] = bbox_prompt[..., 3] * (H_new / H_old)
        return image_sized, point_prompt, bbox_prompt, scale

    def encode_image(self, processed_image: torch.Tensor, original_image_size: tuple):
        """
        Encodes the given image using a specific algorithm.
        Parameters:
        - processed_image (torch.Tensor): The input image to be encoded. A tensor of shape (B, N, H,W).
        - original_image_size (tuple): The original size of the image before processing. A tuple of (H, W).
        Returns:
        - embeddings
        """

        pass

    def reset_image(self):
        pass

    def decode_mask_point(self, prompts: torch.Tensor, labels: torch.Tensor):
        """
        Decodes the mask for the given input prompts and labels.

        Args:
            prompts: tensor of shape (B, N, 2) where B is the batch size, N is the number of prompts
            labels: tensor of shape (B, N) where B is the batch size, N is the number of prompts

        Returns:
            return: seg_masks: tensor of shape (B, N, H, W) where B is the batch size, N is the number of prompts, H and W are the height and width of the image
        """
        pass

    def decode_mask_box(self, box_prompts: torch.Tensor):
        """
        Decodes the mask for the given input prompts and labels.

        Args:
            box_prompts: tensor of shape (B, N, 4) where B is the batch size, N is the number of prompts

        Returns:
            return: seg_masks: tensor of shape (B, N, H, W) where B is the batch size, N is the number of prompts, H and W are the height and width of the image
        """
        pass

    def segment_by_prompt_(
        self, prompt, score_threshould, stability_thresh, prompt_mode="point"
    ):
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(1)
            labels = torch.ones((len(prompt), 1), device=prompt.device)
        else:
            labels = torch.ones((prompt.shape[:2]), device=prompt.device)
        if prompt_mode == "point":
            seg_mask, score = self.decode_mask_point(
                prompt,
                labels,
            )
        elif prompt_mode == "box":
            seg_mask, score = self.decode_mask_box(
                prompt,
            )
        else:
            raise ValueError("prompt_mode should be either 'point' or 'box'")
        score = score.flatten()
        seg_mask = seg_mask.flatten(0, 1)

        if score_threshould is not None:
            seg_mask = seg_mask[score > score_threshould]
            score = score[score > score_threshould]

        if stability_thresh is not None:
            stability_score = calculate_stability_score(seg_mask, 0, 1.0)
            seg_mask = seg_mask[stability_score >= stability_thresh]
            score = score[stability_score >= stability_thresh]
            stability_score = stability_score[stability_score >= stability_thresh]
            # score = (score + stability_score) / 2
        return seg_mask, score

    def segment_by_prompt(
        self,
        prompt,
        batch_size=64,
        score_threshould=None,
        stability_thresh=None,
        prompt_mode="point",
    ):
        if len(prompt) > batch_size:
            foreground_prompt_ds = TensorDataset(prompt)
            foreground_prompt_dl = DataLoader(
                foreground_prompt_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
            seg_masks = []
            seg_scores = []
            for prompt_batch in foreground_prompt_dl:
                prompt_ = prompt_batch[0]

                seg_mask, seg_score = self.segment_by_prompt_(
                    prompt_, score_threshould, stability_thresh, prompt_mode=prompt_mode
                )

                seg_masks.append(seg_mask)
                seg_scores.append(seg_score)

            seg_masks = torch.cat(seg_masks, dim=0)
            seg_scores = torch.cat(seg_scores, dim=0)
        else:
            seg_masks, seg_scores = self.segment_by_prompt_(
                prompt, score_threshould, stability_thresh, prompt_mode=prompt_mode
            )

        sorted_idx = torch.argsort(seg_scores, descending=True)
        seg_masks = seg_masks[sorted_idx]
        seg_scores = seg_scores[sorted_idx]
        return seg_masks, seg_scores

    def segment_by_box_prompt(
        self,
        prompt_boxes,
        batch_size=64,
    ):
        prompt_boxes = prompt_boxes
        if len(prompt_boxes) > batch_size:
            foreground_prompt_ds = TensorDataset(prompt_boxes)
            foreground_prompt_dl = DataLoader(
                foreground_prompt_ds, batch_size=batch_size, shuffle=False
            )
            seg_masks = []
            seg_scores = []
            for prompt_batch in foreground_prompt_dl:
                if prompt_batch[0].dim() == 2:
                    seg_mask, score = self.decode_mask_box(
                        prompt_batch[0].unsqueeze(1),
                    )
                else:
                    seg_mask, score = self.decode_mask_box(
                        prompt_batch[0],
                    )
                seg_scores.append(score.flatten())
                seg_masks.append(seg_mask.flatten(0, 1))
            seg_masks = torch.cat(seg_masks, dim=0)
            seg_scores = torch.cat(seg_scores, dim=0)
        else:
            if prompt_boxes.dim() == 2:
                seg_masks, seg_scores = self.decode_mask_box(
                    prompt_boxes.unsqueeze(1),
                )
            else:
                seg_masks, seg_scores = self.decode_mask_box(
                    prompt_boxes,
                )
            seg_masks = seg_masks.flatten(0, 1)
            seg_scores = seg_scores.flatten()
        return seg_masks, seg_scores

    # def post_process_masks(
    #     self,
    #     masks,
    #     scores,
    #     depth,
    #     cam_scale,
    #     cam_K,
    #     nms_function_1,
    #     nms_function_2,
    #     min_area: int = 100,
    #     org_size=(532, 714),
    #     max_size_ratio=0.65,
    #     mask_threshold=0,
    #     article_min_area=29,
    #     max_length=1000,
    #     to_binary=True,
    # ):
    #     seg_bbox = get_bounding_boxes_batch(masks > mask_threshold)  # binarize it
    #     seg_h, seg_w = (
    #         seg_bbox[:, 3] - seg_bbox[:, 1],
    #         seg_bbox[:, 2] - seg_bbox[:, 0],
    #     )
    #     seg_max_dim = torch.max(seg_h, seg_w)
    #     selected_area_idx = (
    #         (seg_max_dim < max(org_size[0], org_size[1]) * max_size_ratio)
    #         .nonzero()
    #         .flatten()
    #     )
    #     masks = masks[selected_area_idx]
    #     scores = scores[selected_area_idx]
    #     seg_bbox = seg_bbox[selected_area_idx]

    #     selected_min_area_idx = (
    #         ((masks > mask_threshold).sum((1, 2)) > min_area).nonzero().flatten()
    #     )
    #     if len(selected_min_area_idx) > 0:
    #         masks = masks[selected_min_area_idx]
    #         scores = scores[selected_min_area_idx]
    #         seg_bbox = seg_bbox[selected_min_area_idx]
    #         selected_area_idx = selected_area_idx[selected_min_area_idx]

    #     # selected_miou_idx = (scores > miou_threshold).nonzero().flatten()
    #     # if len(selected_miou_idx) > 0:
    #     #     masks = masks[selected_miou_idx]
    #     #     scores = scores[selected_miou_idx]
    #     #     seg_bbox = seg_bbox[selected_miou_idx]
    #     #     selected_area_idx = selected_area_idx[selected_miou_idx]

    #     selected_nms_idx = nms_function_1(masks > mask_threshold, seg_bbox, scores)
    #     if len(selected_nms_idx) > 0:
    #         masks = masks[selected_nms_idx]
    #         scores = scores[selected_nms_idx]
    #         selected_area_idx = selected_area_idx[selected_nms_idx]

    #     selected_nms_idx = nms_function_2(masks > mask_threshold, seg_bbox, 1 / scores)
    #     if len(selected_nms_idx) > 0:
    #         masks = masks[selected_nms_idx]
    #         scores = scores[selected_nms_idx]
    #         selected_area_idx = selected_area_idx[selected_nms_idx]

    #     if to_binary:
    #         masks = masks > mask_threshold

    #     masks = clean_small_articles(masks, kernel_size=article_min_area)

    #     _, seg_mask_feasible = seg_mask_max_size_fileter(
    #         masks, depth, cam_scale, cam_K, max_length
    #     )
    #     masks = masks[seg_mask_feasible]
    #     scores = scores[seg_mask_feasible]

    #     masks, scores = split_connected_regions(masks, scores=scores)

    #     selected_min_area_idx = (masks.sum((1, 2)) > min_area).nonzero().flatten()
    #     masks = masks[selected_min_area_idx]
    #     scores = scores[selected_min_area_idx]

    #     selected_nms_idx = nms_function_2(masks, None, scores)
    #     if len(selected_nms_idx) > 0:
    #         masks = masks[selected_nms_idx]
    #         scores = scores[selected_nms_idx]

    #     return masks, scores, selected_area_idx

    def generate_grid_points(self, point_size, target_size, device, with_offset=True):
        return generate_grid_points_(point_size, target_size, device, with_offset)

    def get_bounding_boxes_batch(self, masks):
        return get_bounding_boxes_batch_(masks)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):
        return batched_nms_(boxes, scores, idxs, iou_threshold)

    def sim_2_point_prompts(
        self, scene_obj_sim, grid_prompt_locations, spatial_size, threshold=0.5
    ):
        return sim_2_point_prompts_(
            scene_obj_sim, grid_prompt_locations, spatial_size, threshold=threshold
        )

    def generate_patch_grid_points(self, size, patch_size, device, corners=False):
        return generate_patch_grid_points_(size, patch_size, device, corners=False)