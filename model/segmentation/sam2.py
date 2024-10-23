import warnings

import torch
from torchvision import transforms
from segmentation.base_segmentation import Base_Segmentation

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import Tuple, Optional

import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor


class SAM2_Seg(Base_Segmentation):
    checkpoint_dic = {
        "hiera_l": "checkpoints/SAM2_Wraper/sam2_hiera_large.pt",
        "hiera_b": "checkpoints/SAM2_Wraper/sam2_hiera_base_plus.pt",
        "hiera_s": "checkpoints/SAM2_Wraper/sam2_hiera_small.pt",
        "hiera_t": "checkpoints/SAM2_Wraper/sam2_hiera_tiny.pt",
    }
    config_dic = {
        "hiera_l": "sam2_hiera_l.yaml",
        "hiera_b": "sam2_hiera_b+.yaml",
        "hiera_s": "sam2_hiera_s.yaml",
        "hiera_t": "sam2_hiera_t.yaml",
    }

    def __init__(self, model_type="hiera_l", device="cpu") -> None:
        super().__init__(device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.predictor = SAM2ImagePredictor(
                build_sam2(
                    SAM2_Seg.config_dic[model_type],
                    SAM2_Seg.checkpoint_dic[model_type],
                )
            )
        # self.predictor.eval().to(device)

        self.predictor._transforms.to_tensor = lambda x: x
        self.predictor._transforms.transforms = Resize(
            (
                self.predictor._transforms.resolution,
                self.predictor._transforms.resolution,
            )
        )

    def scale_image_prompt_to_dim(
        self, image, point_prompt=None, bbox_prompt=None, max_size=1024
    ):
        return image, point_prompt, bbox_prompt, 1

    def encode_image(self, processed_image: torch.Tensor, original_image_size: tuple):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.set_torch_image(processed_image, original_image_size)
        return self.predictor.get_image_embedding()

    def decode_mask_point(self, point_prompts: torch.Tensor, labels: torch.Tensor):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = self.predict_torch(
                point_coords=point_prompts, point_labels=labels, return_logits=True
            )
        return masks, scores

    def decode_mask_box(self, box_prompts: torch.Tensor):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = self.predict_torch(
                box=box_prompts,
                point_coords=None,
                point_labels=None,
                return_logits=True,
            )
        return masks, scores

    @torch.no_grad()
    def set_torch_image(
        self,
        processed_image: torch.Tensor,
        original_image_size: Tuple[int, int],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
        transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide or other processing.
        original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        self.predictor.reset_predictor()

        # Ensure the transformed image has the correct shape
        assert (
            len(processed_image.shape) == 4 and processed_image.shape[1] == 3
        ), f"transformed_image must be of size 1x3xHxW, got {processed_image.shape}"

        # Store the original and input image sizes
        self.predictor._orig_hw = [
            original_image_size
        ]  # Original size in (H, W) format

        processed_image = self.predictor._transforms(processed_image)

        self.predictor.input_size = tuple(
            processed_image.shape[-2:]
        )  # Input size (H, W)

        # Forward pass through the model to get the backbone output
        backbone_out = self.predictor.model.forward_image(processed_image)

        # Prepare the backbone features
        _, vision_feats, _, _ = self.predictor.model._prepare_backbone_features(
            backbone_out
        )

        # Optionally add no_mem_embed during feature extraction
        if self.predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.predictor.model.no_mem_embed

        # Process features: Permute dimensions and reshape according to feature sizes
        feats = [
            feat.permute(1, 2, 0).view(feat.shape[1], -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self.predictor._bb_feat_sizes[::-1]
            )
        ][::-1]

        # Store the computed features
        self.predictor._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }

        # Indicate that the image has been set
        self.predictor._is_image_set = True

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        box: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
        point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
        point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
        boxes (torch.Tensor or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
        mask_input (torch.Tensor): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
        multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
        return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
        (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
        (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
        (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.predictor._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self.predictor._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        return masks, iou_predictions, low_res_masks
