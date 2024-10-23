import warnings

import torch
from segment_anything import SamPredictor, sam_model_registry

from model.segmentation.base_segmentation import Base_Segmentation


class SAM_Seg(Base_Segmentation):
    checkpoint_dic = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }

    def __init__(self, model_type="vit_h", checkpoint_dir="./checkpoints/segment-anything/") -> None:
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sam = sam_model_registry[model_type](
                checkpoint=checkpoint_dir + SAM_Seg.checkpoint_dic[model_type],
            )
        self.predictor = SamPredictor(self.sam)
        self.sam.image_encoder.img_size = self.input_size

    def encode_image(self, processed_image: torch.Tensor, original_image_size: tuple):
        embeddings = []
        processed_image *= 255  # Convert to 0-255 range
        for i in range(len(processed_image)):
            with torch.no_grad():
                self.predictor.set_torch_image(
                    processed_image[i].unsqueeze(0),
                    original_image_size=original_image_size,
                )
            embedding = self.predictor.get_image_embedding()
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def decode_mask_point(self, point_prompts: torch.Tensor, labels: torch.Tensor):
        masks, scores, _ = self.predictor.predict_torch(
            point_coords=point_prompts,
            point_labels=labels,
            return_logits=True,
            multimask_output=True,
        )
        return masks, scores

    def decode_mask_box(self, box_prompts: torch.Tensor):
        masks, scores, _ = self.predictor.predict_torch(
            boxes=box_prompts,
            point_coords=None,
            point_labels=None,
            return_logits=True,
            multimask_output=True,
        )
        return masks, scores

    def reset_image(self):
        self.predictor.reset_image()
