import torch
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
import os
import os.path as osp
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from utils.inout import save_json, load_json, save_json_bop23, save_configs
from model.utils import BatchedData, Detections, convert_npz_to_json
from hydra.utils import instantiate
import time
import glob
from functools import partial
import multiprocessing
import trimesh
from model.loss import MaskedPatch_MatrixSimilarity
from utils.trimesh_utils import depth_image_to_pointcloud_translate_torch
from utils.poses.pose_utils import get_obj_poses_from_template_level
from utils.bbox_utils import xyxy_to_xywh, compute_iou
import torch.nn.functional as F
from PIL import Image
import random
from model.segmentation.prompt_helper import get_grid_coordinates

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pytorch3d.ops import sample_farthest_points


def shift_and_pad_nonzero_tokens(x):
    """
    x: (O, T, N, d) tensor where some tokens may be all-zero (masked)
    Returns:
      - padded_tokens: (O, max_L, d) where only non-zero tokens are kept and padded
      - lengths: (O,) tensor of the number of non-zero tokens per object
    """
    O, M, d = x.shape
    

    # Find non-zero (non-masked) tokens per object
    nonzero_mask = x.abs().sum(dim=-1) > 0  # (O, M)
    lengths = nonzero_mask.sum(dim=1)  # (O,)

    max_len = lengths.max().item()
    padded_tokens = torch.zeros((O, max_len, d), dtype=x.dtype, device=x.device)

    for o in range(O):
        valid_tokens = x[o][nonzero_mask[o]]  # shape (L_o, d)
        padded_tokens[o, : valid_tokens.shape[0]] = valid_tokens

    return padded_tokens, lengths


def select_prototypes_cascade(x, k, stages=1, tau=0.05):
    """
    Cascade prototype selection using token coverage and separation.
    - x: (O, T, N, d): input token features
    - k: final number of prototypes to return
    - stages: number of selection refinement stages
    - tau: temperature for cosine similarity
    Returns:
    - P: (O, k, d) selected prototypes
    """
    O, T, N, d = x.shape
    tokens = x.reshape(O, T * N, d)  # (O, M, d)

    # Initial number of candidates: k × 2^(stages - 1)
    k_stage = k * (2 ** (stages - 1))

    tokens, lengths = shift_and_pad_nonzero_tokens(tokens)  # (O, M, d), (O,)
    P = sample_farthest_points(tokens, lengths=lengths, K=k_stage)[0]  # (O, k_stage, d)

    for s in range(stages - 1):
        k_curr = P.shape[1]
        k_next = k_curr // 2

        # 1. Coverage score: similarity to own tokens
        sim_to_own = torch.einsum("okd, omd -> okm", P, tokens)  # (O, k, M)
        mask = (
            torch.arange(tokens.shape[1], device=tokens.device)[None, :]
            < lengths[:, None]
        )  # (O, M)
        mask = mask.unsqueeze(1).float()  # (O, 1, M)
        masked_sim = sim_to_own * mask
        coverage_score = masked_sim.sum(dim=-1) / (mask.sum(dim=-1) + 1e-6)  # (O, k)

        # 2. Separation score: distance to tokens of all objects
        P_norm = F.normalize(P, dim=-1)
        tokens_all = torch.cat(
            [tokens[o, : lengths[o]] for o in range(O)], dim=0
        )  # (total_valid_tokens, d)
        dist_to_all = torch.cdist(P_norm, tokens_all)  # (O, k_curr, OM)
        separation_score = dist_to_all.mean(dim=-1)  # (O, k_curr)

        # 3. Combined score: prefer high coverage + high separation
        # Optional: normalize each score across candidates
        coverage_score = (coverage_score - coverage_score.mean(dim=1, keepdim=True)) / (
            coverage_score.std(dim=1, keepdim=True) + 1e-6
        )
        separation_score = (
            separation_score - separation_score.mean(dim=1, keepdim=True)
        ) / (separation_score.std(dim=1, keepdim=True) + 1e-6)
        final_score = coverage_score + separation_score  # (O, k_curr)

        # 4. Keep top-k_next by score
        top_idx = final_score.topk(k_next, dim=1).indices  # (O, k_next)
        P = torch.gather(P, 1, top_idx.unsqueeze(-1).expand(-1, -1, d))

    return P  # shape: (O, k, d)


class Instance_Segmentation_Model(pl.LightningModule):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        visible_thred,
        pointcloud_sample_num,
        weight_scores=False,
        num_templates=42,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model

        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.log_interval = log_interval
        self.log_dir = log_dir
        self.final_results = []

        self.visible_thred = visible_thred
        self.pointcloud_sample_num = pointcloud_sample_num

        self.weight_scores = weight_scores

        self.num_templates = num_templates

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )

        logging.info(f"Init CNOS done!")

    def set_reference_objects(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data = {
            "descriptors": BatchedData(None),
            "appe_descriptors": BatchedData(None),
        }
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")
        appe_descriptors_path = osp.join(
            self.ref_dataset.template_dir, "descriptors_appe.pth"
        )

        # Loading main descriptors
        if self.onboarding_config.rendering_type == "pbr":
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(self.device)
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken"
                )
                self.ref_data["descriptors"].append(ref_feats)

            self.ref_data["descriptors"].stack()  # N_objects x descriptor_size
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["descriptors"], descriptors_path)

        # Loading appearance descriptors
        if self.onboarding_config.rendering_type == "pbr":
            appe_descriptors_path = appe_descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(appe_descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["appe_descriptors"] = torch.load(appe_descriptors_path).to(
                self.device
            )
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing appearance descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                ref_feats = self.descriptor_model.compute_masked_patch_feature(
                    ref_imgs, ref_masks
                )
                self.ref_data["appe_descriptors"].append(ref_feats)

            self.ref_data["appe_descriptors"].stack()
            self.ref_data["appe_descriptors"] = self.ref_data["appe_descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["appe_descriptors"], appe_descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}, \
            Appearance descriptors shape: {self.ref_data['appe_descriptors'].shape}"
        )

    def set_reference_object_pointcloud(self):
        """
        Loading the pointclouds of reference objects: (N_object, N_pointcloud, 3)
        N_pointcloud: the number of points sampled from the reference object mesh.
        """
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects point cloud ...")

        start_time = time.time()
        pointcloud = BatchedData(None)
        pointcloud_path = osp.join(self.ref_dataset.template_dir, "pointcloud.pth")
        obj_pose_path = f"{self.ref_dataset.template_dir}/template_poses.npy"

        # Loading pointcloud pose
        if (
            os.path.exists(obj_pose_path)
            and not self.onboarding_config.reset_descriptors
        ):
            poses = (
                torch.tensor(np.load(obj_pose_path)).to(self.device).to(torch.float32)
            )  # N_all_template x 4 x 4
        else:
            template_poses = get_obj_poses_from_template_level(
                level=2, pose_distribution="all"
            )
            template_poses[:, :3, 3] *= 0.4
            poses = torch.tensor(template_poses).to(self.device).to(torch.float32)
            np.save(obj_pose_path, template_poses)

        self.ref_data["poses"] = poses[
            self.ref_dataset.index_templates, :, :
        ]  # N_template x 4 x 4
        if (
            os.path.exists(pointcloud_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["pointcloud"] = torch.load(
                pointcloud_path, map_location="cuda:0"
            ).to(self.device)
        else:
            mesh_path = osp.join(self.ref_dataset.root_dir, "models")
            if not os.path.exists(mesh_path):
                raise Exception("Can not find the mesh path.")
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Generating pointcloud dataset ...",
            ):
                # loading cad
                if self.dataset_name == "lmo":
                    all_pc_idx = [1, 5, 6, 8, 9, 10, 11, 12]
                    pc_id = all_pc_idx[idx]
                else:
                    pc_id = idx + 1
                mesh = trimesh.load_mesh(
                    os.path.join(mesh_path, f"obj_{(pc_id):06d}.ply")
                )
                model_points = (
                    mesh.sample(self.pointcloud_sample_num).astype(np.float32) / 1000.0
                )
                pointcloud.append(torch.tensor(model_points))

            pointcloud.stack()  # N_objects x N_pointcloud x 3
            self.ref_data["pointcloud"] = pointcloud.data.to(self.device)

            # save the precomputed features for future use
            torch.save(self.ref_data["pointcloud"], pointcloud_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Pointcloud shape: {self.ref_data['pointcloud'].shape}"
        )

    def best_template_pose(self, scores, pred_idx_objects):
        _, best_template_idxes = torch.max(scores, dim=-1)
        N_query, N_object = best_template_idxes.shape[0], best_template_idxes.shape[1]
        pred_idx_objects = pred_idx_objects[:, None].repeat(1, N_object)

        assert N_query == pred_idx_objects.shape[0], "Prediction num != Query num"

        best_template_idx = torch.gather(
            best_template_idxes, dim=1, index=pred_idx_objects
        )[:, 0]

        return best_template_idx

    def project_template_to_image(self, best_pose, pred_object_idx, batch, proposals):
        """
        Obtain the RT of the best template, then project the reference pointclouds to query image,
        getting the bbox of projected pointcloud from the image.
        """

        pose_R = self.ref_data["poses"][best_pose, 0:3, 0:3]  # N_query x 3 x 3
        select_pc = self.ref_data["pointcloud"][
            pred_object_idx, ...
        ]  # N_query x N_pointcloud x 3
        (N_query, N_pointcloud, _) = select_pc.shape

        # translate object_selected pointcloud by the selected best pose and camera coordinate
        posed_pc = torch.matmul(pose_R, select_pc.permute(0, 2, 1)).permute(0, 2, 1)
        translate = self.Calculate_the_query_translation(
            proposals,
            batch["depth"][0],
            batch["cam_intrinsic"][0],
            batch["depth_scale"],
        )
        posed_pc = posed_pc + translate[:, None, :].repeat(1, N_pointcloud, 1)

        # project the pointcloud to the image
        cam_instrinsic = (
            batch["cam_intrinsic"][0][None, ...].repeat(N_query, 1, 1).to(torch.float32)
        )
        image_homo = torch.bmm(cam_instrinsic, posed_pc.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        image_vu = (image_homo / image_homo[:, :, -1][:, :, None])[:, :, 0:2].to(
            torch.int
        )  # N_query x N_pointcloud x 2
        (imageH, imageW) = batch["depth"][0].shape
        image_vu[:, :, 0].clamp_(min=0, max=imageW - 1)
        image_vu[:, :, 1].clamp_(min=0, max=imageH - 1)

        return image_vu

    def Calculate_the_query_translation(
        self, proposal, depth, cam_intrinsic, depth_scale
    ):
        """
        Calculate the translation amount from the origin of the object coordinate system to the camera coordinate system.
        Cut out the depth using the provided mask and calculate the mean as the translation.
        proposal: N_query x imageH x imageW
        depth: imageH x imageW
        """
        shape = proposal.squeeze_(1).shape
        if len(shape) == 2:
            proposal = proposal.unsqueeze(0)
        (N_query, imageH, imageW) = proposal.shape
        masked_depth = proposal * (depth[None, ...].repeat(N_query, 1, 1))
        translate = depth_image_to_pointcloud_translate_torch(
            masked_depth, depth_scale, cam_intrinsic
        )
        return translate.to(torch.float32)

    def move_to_device(self):
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device
        # if there is predictor in the model, move it to device
        if hasattr(self.segmentor_model, "predictor"):
            self.segmentor_model.predictor.model = (
                self.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")

    def compute_semantic_score(self, proposal_decriptors):
        # compute matching scores for each proposals
        scores = self.matching_config.metric(
            proposal_decriptors, self.ref_data["descriptors"][:, : self.num_templates]
        )  # N_proposals x N_objects x N_templates
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":
            score_per_proposal_and_object = torch.topk(
                scores, k=min(5, self.num_templates), dim=-1
            )[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_query

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        semantic_score = score_per_proposal[idx_selected_proposals]

        # compute the best view of template
        flitered_scores = scores[idx_selected_proposals, ...]
        best_template = self.best_template_pose(flitered_scores, pred_idx_objects)

        return idx_selected_proposals, pred_idx_objects, semantic_score, best_template

    def compute_appearance_score(
        self, best_pose, pred_objects_idx, qurey_appe_descriptors
    ):
        """
        Based on the best template, calculate appearance similarity indicated by appearance score
        """
        con_idx = torch.concatenate(
            (pred_objects_idx[None, :], best_pose[None, :]), dim=0
        )
        ref_appe_descriptors = self.ref_data["appe_descriptors"][
            :, : self.num_templates
        ][
            con_idx[0, ...], con_idx[1, ...], ...
        ]  # N_query x N_patch x N_feature

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(
            qurey_appe_descriptors, ref_appe_descriptors
        )

        return appe_scores, ref_appe_descriptors

    def compute_geometric_score(
        self,
        image_uv,
        proposals,
        appe_descriptors,
        ref_aux_descriptor,
        visible_thred=0.5,
    ):
        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(
            appe_descriptors, ref_aux_descriptor, visible_thred
        )

        # IoU calculation
        y1x1 = torch.min(image_uv, dim=1).values
        y2x2 = torch.max(image_uv, dim=1).values
        xyxy = torch.concatenate((y1x1, y2x2), dim=-1)

        iou = compute_iou(xyxy, proposals.boxes)

        return iou, visible_ratio

    def test_step(self, batch, idx):
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}",
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.set_reference_object_pointcloud()
            self.proposal_stage_duration = []
            self.dino_stage_duration = []
            self.matching_stage_duration = []
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]).cpu().numpy().transpose(1, 2, 0)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np)

        # init detections with masks and boxes
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )

        # compute semantic descriptors and appearance descriptors for query proposals
        query_decriptors, query_appe_descriptors = self.descriptor_model(
            image_np, detections
        )
        proposal_stage_end_time = time.time()

        # matching descriptors
        dino_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.compute_semantic_score(query_decriptors)
        dino_stage_end_time = time.time()

        matching_stage_start_time = time.time()

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor = self.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_descriptors
        )

        # compute the geometric score
        image_uv = self.project_template_to_image(
            best_template, pred_idx_objects, batch, detections.masks
        )

        geometric_score, visible_ratio = self.compute_geometric_score(
            image_uv,
            detections,
            query_appe_descriptors,
            ref_aux_descriptor,
            visible_thred=self.visible_thred,
        )

        # final score
        final_score = (
            semantic_score + appe_scores + geometric_score * visible_ratio
        ) / (1 + 1 + visible_ratio)

        if self.weight_scores:
            final_score *= detections.seg_scores

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            (proposal_stage_end_time - proposal_stage_start_time)
            + (dino_stage_end_time - dino_stage_start_time)
            + (matching_stage_end_time - matching_stage_start_time)
        )

        self.proposal_stage_duration.append(
            proposal_stage_end_time - proposal_stage_start_time
        )
        self.dino_stage_duration.append(dino_stage_end_time - dino_stage_start_time)
        self.matching_stage_duration.append(
            matching_stage_end_time - matching_stage_start_time
        )

        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]

        # convert detections to coco format
        coco_result = detections.convert_to_coco_format(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            dataset_name=self.dataset_name,
        )
        for res in coco_result:
            self.final_results.append(res)

        return 0

    def test_epoch_end(self, outputs):
        detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
        save_json_bop23(detections_path, self.final_results)

        logging.info(f"Saved predictions to {detections_path}")
        # save log
        log_path = f"{self.log_dir}/{self.name_prediction_file}_time.txt"
        with open(log_path, "w") as f:
            f.write(
                f"Proposal stage duration: {np.mean(self.proposal_stage_duration)}\n"
            )
            f.write(f"Dino stage duration: {np.mean(self.dino_stage_duration)}\n")
            f.write(
                f"Matching stage duration: {np.mean(self.matching_stage_duration)}\n"
            )
        logging.info(
            f"proposal stage duration: {np.mean(self.proposal_stage_duration)}"
        )
        logging.info(f"dino stage duration: {np.mean(self.dino_stage_duration)}")
        logging.info(
            f"matching stage duration: {np.mean(self.matching_stage_duration)}"
        )


class prompted_segmentation(Instance_Segmentation_Model):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        visible_thred,
        pointcloud_sample_num,
        sim_thresh,
        sg_thresh,
        prompt_mode,
        score_mode,
        weight_scores,
        num_templates,
        **kwargs,
    ):
        super().__init__(
            segmentor_model,
            descriptor_model,
            onboarding_config,
            matching_config,
            post_processing_config,
            log_interval,
            log_dir,
            visible_thred,
            pointcloud_sample_num,
            weight_scores,
            **kwargs,
        )
        self.sim_thresh = sim_thresh
        self.sg_thresh = sg_thresh
        self.prompt_mode = prompt_mode
        self.score_mode = score_mode
        self.num_templates = num_templates

        if self.prompt_mode == "self_grounding":
            for _, module in self.descriptor_model.named_modules():
                if module.__class__.__name__ in [
                    "Attention",
                    "MemEffAttention",
                ]:  # Match by class name
                    setattr(module, "sg_threshold", self.sg_thresh)

    def test_step(self, batch, idx):
        if idx == 0:
            # os.makedirs(
            #     osp.join(
            #         self.log_dir,
            #         f"predictions/{self.dataset_name}/{self.name_prediction_file}",
            #     ),
            #     exist_ok=True,
            # )
            self.set_reference_objects()
            self.set_reference_object_pointcloud()
            self.set_last_token()
            self.set_grounding_info()

            self.proposal_stage_duration = []
            self.dino_stage_duration = []
            self.matching_stage_duration = []

            self.move_to_device()

            self.obj_templates_feats_selected = F.normalize(self.ref_data["appe_descriptors"], dim=-1)
            self.obj_templates_feats_selected = select_prototypes_cascade(
                self.obj_templates_feats_selected, 8, stages=4
            )

        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        proposal_stage_start_time = time.time()

        # if batch["scene_id"][0] != 13 and batch["frame_id"][0] != 470:
        #     return 0

        if self.prompt_mode == "normal":
            foreground_prompt_locations, foreground_prompt_map = (
                self.generate_foreground_prompt(batch, threshold=self.sim_thresh)
            )
        elif self.prompt_mode == "self_grounding":
            foreground_prompt_locations, foreground_prompt_map = (
                self.generate_foreground_prompt_sg(batch, threshold=self.sim_thresh)
            )
        elif self.prompt_mode == "grid":
            foreground_prompt_locations = get_grid_coordinates(
                size=(32, 32),
                device=self.device,
                with_offset=True,
            )
            foreground_prompt_locations = torch.stack(
                foreground_prompt_locations, dim=-1
            ).flatten(0, 1)
        else:
            raise NotImplementedError

        if self.prompt_mode != "grid":
            if len(foreground_prompt_locations) == 0:
                return 0

            foreground_prompt_locations = foreground_prompt_locations[:2048]

            # import torch.nn.functional as F
            # from torchvision import utils as vutils

            # foreground_prompt_map = foreground_prompt_map.permute(
            #     2, 0, 1
            # ).float()
            # foreground_prompt_map = F.interpolate(foreground_prompt_map.unsqueeze(0), size=(480, 640), mode='nearest')

            # vutils.save_image(
            #     foreground_prompt_map, f"mask_prompts_{self.sim_thresh}.png"
            # )
            # foreground_prompt_map[foreground_prompt_map == 0] = 0.2
            # result =  foreground_prompt_map * self.inv_rgb_transform(batch["image"])

            # vutils.save_image(
            #     result, f"positives_prompts_{self.sim_thresh}.png"
            # )

            # foreground_prompt_locations = foreground_prompt_locations.flatten(0,1)

            foreground_prompt_locations[..., 0] = (
                foreground_prompt_locations[..., 0] / self.descriptor_model.full_size[1]
            )
            foreground_prompt_locations[..., 1] = (
                foreground_prompt_locations[..., 1] / self.descriptor_model.full_size[0]
            )

        test_image_sized, point_prompt_scaled, _, _ = (
            self.segmentor_model.scale_image_prompt_to_dim(
                image=self.inv_rgb_transform(batch["image"][0]).unsqueeze(0),
                point_prompt=foreground_prompt_locations,
                # max_size=detector.input_size,
            )
        )
        _ = self.segmentor_model.encode_image(
            test_image_sized,
            original_image_size=batch["image"][0].shape[-2:],
        )
        seg_masks, seg_scores, stability_scores = (
            self.segmentor_model.segment_by_prompt(
                prompt=point_prompt_scaled,
                batch_size=64,
                score_threshould=0.88,
                stability_thresh=0.85,
            )
        )
        if len(seg_masks) == 0:
            return 0
        seg_masks = seg_masks > 0
        seg_masks = seg_masks.float()
        seg_scores = seg_scores.float()
        stability_scores = stability_scores.float()
        seg_boxes = self.segmentor_model.get_bounding_boxes_batch(seg_masks)
        keep_idxs = self.segmentor_model.batched_nms(
            boxes=seg_boxes.float(),
            scores=(seg_scores + stability_scores) / 2,
            idxs=torch.zeros(len(seg_masks)).to(self.device),  # categories
            iou_threshold=0.7,
        )
        seg_masks = seg_masks[keep_idxs]
        seg_scores = seg_scores[keep_idxs]
        seg_boxes = seg_boxes[keep_idxs]

        # init detections with masks and boxes
        detections = Detections({"masks": seg_masks.float(), "boxes": seg_boxes})
        detections.add_attribute("seg_scores", seg_scores)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )

        # compute semantic descriptors and appearance descriptors for query proposals
        try:
            query_decriptors, query_appe_descriptors = self.descriptor_model(
                batch["image"][0], detections
            )
        except:
            return 0

        proposal_stage_end_time = time.time()

        # matching descriptors
        dino_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.compute_semantic_score(query_decriptors)
        dino_stage_end_time = time.time()

        matching_stage_start_time = time.time()

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        if self.score_mode == "normal":
            appe_scores, ref_aux_descriptor = self.compute_appearance_score(
                best_template, pred_idx_objects, query_appe_descriptors
            )

            # compute the geometric score
            image_uv = self.project_template_to_image(
                best_template, pred_idx_objects, batch, detections.masks
            )

            geometric_score, visible_ratio = self.compute_geometric_score(
                image_uv,
                detections,
                query_appe_descriptors,
                ref_aux_descriptor,
                visible_thred=self.visible_thred,
            )
        elif self.score_mode == "self_grounding":
            if len(detections.masks.shape) == 4:
                detections.masks.squeeze_(1)
            grounding_info = self.ref_data["grounding_info"][pred_idx_objects]
            grounding_info = grounding_info.mean(dim=(0, 1)).unsqueeze(0)
            _, query_appe_descriptors = self.descriptor_model.forward_with_sg(
                batch["image"][0], detections, g_info=grounding_info
            )
            appe_scores, ref_aux_descriptor = self.compute_appearance_score_sg(
                best_template, pred_idx_objects, query_appe_descriptors
            )

            # compute the geometric score
            image_uv = self.project_template_to_image(
                best_template, pred_idx_objects, batch, detections.masks
            )

            geometric_score, visible_ratio = self.compute_geometric_score_sg(
                image_uv,
                detections,
                query_appe_descriptors,
                ref_aux_descriptor,
                visible_thred=self.visible_thred,
            )
        else:
            raise NotImplementedError

        final_score = (
            semantic_score + appe_scores + geometric_score * visible_ratio
        ) / (1 + 1 + visible_ratio)

        if self.weight_scores:
            final_score *= detections.seg_scores

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            (proposal_stage_end_time - proposal_stage_start_time)
            + (dino_stage_end_time - dino_stage_start_time)
            + (matching_stage_end_time - matching_stage_start_time)
        )
        self.proposal_stage_duration.append(
            proposal_stage_end_time - proposal_stage_start_time
        )
        self.dino_stage_duration.append(dino_stage_end_time - dino_stage_start_time)
        self.matching_stage_duration.append(
            matching_stage_end_time - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]

        # convert detections to coco format
        coco_result = detections.convert_to_coco_format(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            dataset_name=self.dataset_name,
        )
        for res in coco_result:
            self.final_results.append(res)
        return 0

    def compute_appearance_score(
        self, best_pose, pred_objects_idx, qurey_appe_descriptors
    ):
        """
        Based on the best template, calculate appearance similarity indicated by appearance score
        """
        con_idx = torch.concatenate(
            (pred_objects_idx[None, :], best_pose[None, :]), dim=0
        )
        ref_appe_descriptors = self.obj_templates_feats_selected[pred_objects_idx]

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(
            qurey_appe_descriptors, ref_appe_descriptors
        )

        return appe_scores, ref_appe_descriptors

    def generate_foreground_prompt(self, batch, threshold=0.4):
        test_image_desc = self.descriptor_model.encode_full_size(batch["image"])[0]

        obj_templates_feats = self.ref_data["last_token"]
        features_mask = obj_templates_feats.sum(dim=-1) > 0
        mask_sum = features_mask.sum(dim=-1)
        mask_sum[mask_sum == 0] = 1e-6
        obj_templates_feats = obj_templates_feats.sum(dim=-2) / mask_sum.unsqueeze(-1)
        obj_templates_feats[mask_sum == 0] = 0  # avoid nan
        obj_templates_feats = obj_templates_feats[:, : self.num_templates]
        obj_templates_feats = obj_templates_feats.mean(dim=(0, 1)).unsqueeze(0)

        test_image_desc /= torch.norm(test_image_desc, dim=-1, keepdim=True)
        obj_templates_feats /= torch.norm(obj_templates_feats, dim=-1, keepdim=True)

        scene_obj_sim = test_image_desc @ obj_templates_feats.t()

        scene_obj_sim = (scene_obj_sim - scene_obj_sim.min()) / (
            scene_obj_sim.max() - scene_obj_sim.min()
        )

        grid_prompt_locations = self.segmentor_model.generate_patch_grid_points(
            self.descriptor_model.output_spatial_size,
            self.descriptor_model.patch_size,
            device=self.device,
            corners=False,
        )
        foreground_prompt_map, foreground_prompt_locations = (
            self.segmentor_model.sim_2_point_prompts(
                scene_obj_sim=scene_obj_sim,
                grid_prompt_locations=grid_prompt_locations,
                spatial_size=self.descriptor_model.output_spatial_size,
                threshold=threshold,
            )
        )

        return foreground_prompt_locations, foreground_prompt_map

    def set_last_token(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data["last_token"] = BatchedData(None)
        last_token_path = osp.join(self.ref_dataset.template_dir, "last_token.pth")

        # Loading appearance descriptors
        if self.onboarding_config.rendering_type == "pbr":
            last_token_path = last_token_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(last_token_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["last_token"] = torch.load(last_token_path).to(self.device)
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing last tokens info ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                last_token = self.descriptor_model.compute_masked_patch_last_token(
                    ref_imgs, ref_masks
                )
                # last_token = last_token.mean(dim=0)
                self.ref_data["last_token"].append(last_token)

            self.ref_data["last_token"].stack()
            self.ref_data["last_token"] = self.ref_data["last_token"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["last_token"], last_token_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Query shape: {self.ref_data['last_token'].shape}, \
            Query descriptors shape: {self.ref_data['last_token'].shape}"
        )

    def generate_foreground_prompt_sg(self, batch, threshold=0.4):
        grounding_info = self.ref_data["grounding_info"][:, : self.num_templates]
        # average all the objects to form a foreground grounding info
        grounding_info = grounding_info.mean(dim=(0, 1)).unsqueeze(0)

        test_image_desc = self.descriptor_model.encode_full_size(
            batch["image"], g_info=grounding_info
        )[0]

        obj_templates_feats = self.ref_data["last_token"]
        features_mask = obj_templates_feats.sum(dim=-1) > 0
        mask_sum = features_mask.sum(dim=-1)
        mask_sum[mask_sum == 0] = 1e-6
        obj_templates_feats = obj_templates_feats.sum(dim=-2) / mask_sum.unsqueeze(-1)
        obj_templates_feats[mask_sum == 0] = 0  # avoid nan
        # obj_templates_feats = obj_templates_feats[mask_sum.any(dim=-1)]

        num_heads = self.descriptor_model.model.num_heads
        head_dim = obj_templates_feats.shape[-1] // num_heads
        o, t, c = obj_templates_feats.shape
        if grounding_info is not None:
            obj_templates_feats = obj_templates_feats.view(o, t, num_heads, head_dim)
            obj_templates_feats = F.normalize(obj_templates_feats, dim=-1)

        obj_templates_feats = obj_templates_feats.view(o, t, num_heads * head_dim)
        obj_templates_feats = obj_templates_feats[:, : self.num_templates]
        obj_templates_feats = obj_templates_feats.mean(dim=(0, 1)).unsqueeze(0)

        if grounding_info is not None:
            test_image_desc = test_image_desc.view(-1, num_heads, head_dim)
            obj_templates_feats = obj_templates_feats.view(-1, num_heads, head_dim)
            test_image_desc = F.normalize(test_image_desc, dim=-1)
            obj_templates_feats = F.normalize(obj_templates_feats, dim=-1)

        test_image_desc = test_image_desc.view(-1, num_heads * head_dim)
        obj_templates_feats = obj_templates_feats.view(-1, num_heads * head_dim)
        test_image_desc = F.normalize(test_image_desc, dim=-1)
        obj_templates_feats = F.normalize(obj_templates_feats, dim=-1)

        scene_obj_sim = test_image_desc @ obj_templates_feats.t()

        scene_obj_sim = (scene_obj_sim - scene_obj_sim.min()) / (
            scene_obj_sim.max() - scene_obj_sim.min()
        )

        grid_prompt_locations = self.segmentor_model.generate_patch_grid_points(
            self.descriptor_model.output_spatial_size,
            self.descriptor_model.patch_size,
            device=self.device,
            corners=False,
        )
        foreground_prompt_map, foreground_prompt_locations = (
            self.segmentor_model.sim_2_point_prompts(
                scene_obj_sim=scene_obj_sim,
                grid_prompt_locations=grid_prompt_locations,
                spatial_size=self.descriptor_model.output_spatial_size,
                threshold=threshold,
            )
        )

        return foreground_prompt_locations, foreground_prompt_map

    def set_grounding_info(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data["grounding_info"] = BatchedData(None)
        grounding_info_path = osp.join(
            self.ref_dataset.template_dir, "grounding_info.pth"
        )

        # Loading appearance descriptors
        if self.onboarding_config.rendering_type == "pbr":
            grounding_info_path = grounding_info_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(grounding_info_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["grounding_info"] = torch.load(grounding_info_path).to(
                self.device
            )
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing grounding info ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                ref_query = self.descriptor_model.compute_masked_patch_average_tokens(
                    ref_imgs, ref_masks
                )
                if len(ref_query) < 42:
                    average = ref_query.mean(dim=0).unsqueeze(0)
                    average = average.repeat(42 - len(ref_query), 1, 1, 1)
                    ref_query = torch.cat((ref_query, average), dim=0)
                # ref_query = ref_query.mean(dim=0)
                self.ref_data["grounding_info"].append(ref_query)

            self.ref_data["grounding_info"].stack()
            self.ref_data["grounding_info"] = self.ref_data["grounding_info"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["grounding_info"], grounding_info_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Query shape: {self.ref_data['grounding_info'].shape}, \
            Query descriptors shape: {self.ref_data['grounding_info'].shape}"
        )

    def compute_appearance_score_sg(
        self, best_pose, pred_objects_idx, qurey_appe_descriptors
    ):
        """
        Based on the best template, calculate appearance similarity indicated by appearance score
        """
        con_idx = torch.concatenate(
            (pred_objects_idx[None, :], best_pose[None, :]), dim=0
        )
        ref_appe_descriptors = self.ref_data["last_token"][
            con_idx[0, ...], con_idx[1, ...], ...
        ]  # N_query x N_patch x N_feature

        num_heads = self.descriptor_model.model.num_heads
        head_dim = qurey_appe_descriptors.shape[-1] // num_heads

        ref_appe_descriptors = ref_appe_descriptors.view(
            *ref_appe_descriptors.shape[:-1], num_heads, head_dim
        )
        ref_appe_descriptors = F.normalize(ref_appe_descriptors, dim=-1)
        ref_appe_descriptors = ref_appe_descriptors.view(
            *ref_appe_descriptors.shape[:-2], num_heads * head_dim
        )
        ref_appe_descriptors = F.normalize(ref_appe_descriptors, dim=-1)

        qurey_appe_descriptors = qurey_appe_descriptors.view(
            *qurey_appe_descriptors.shape[:-1], num_heads, head_dim
        )
        qurey_appe_descriptors = F.normalize(qurey_appe_descriptors, dim=-1)
        qurey_appe_descriptors = qurey_appe_descriptors.view(
            *qurey_appe_descriptors.shape[:-2], num_heads * head_dim
        )
        qurey_appe_descriptors = F.normalize(qurey_appe_descriptors, dim=-1)

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(
            qurey_appe_descriptors, ref_appe_descriptors
        )
        return appe_scores, ref_appe_descriptors

    def compute_geometric_score_sg(
        self,
        image_uv,
        proposals,
        qurey_appe_descriptors,
        ref_aux_descriptor,
        visible_thred=0.5,
    ):
        num_heads = self.descriptor_model.model.num_heads
        head_dim = qurey_appe_descriptors.shape[-1] // num_heads
        qurey_appe_descriptors = qurey_appe_descriptors.view(
            *qurey_appe_descriptors.shape[:-1], num_heads, head_dim
        )
        qurey_appe_descriptors = F.normalize(qurey_appe_descriptors, dim=-1)
        qurey_appe_descriptors = qurey_appe_descriptors.view(
            *qurey_appe_descriptors.shape[:-2], num_heads * head_dim
        )
        qurey_appe_descriptors = F.normalize(qurey_appe_descriptors, dim=-1)

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(
            qurey_appe_descriptors, ref_aux_descriptor, visible_thred
        )

        # IoU calculation
        y1x1 = torch.min(image_uv, dim=1).values
        y2x2 = torch.max(image_uv, dim=1).values
        xyxy = torch.concatenate((y1x1, y2x2), dim=-1)

        iou = compute_iou(xyxy, proposals.boxes)

        return iou, visible_ratio


class prompt_quality(prompted_segmentation):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        visible_thred,
        pointcloud_sample_num,
        sim_thresh,
        prompt_mode,
        score_mode,
        weight_scores,
        num_templates,
        **kwargs,
    ):
        super().__init__(
            segmentor_model,
            descriptor_model,
            onboarding_config,
            matching_config,
            post_processing_config,
            log_interval,
            log_dir,
            visible_thred,
            pointcloud_sample_num,
            sim_thresh,
            prompt_mode,
            score_mode,
            weight_scores,
            num_templates,
            **kwargs,
        )

        self.sg_quality = []
        self.grid_quality = []

    def test_step(self, batch, idx):
        if idx == 0:
            # os.makedirs(
            #     osp.join(
            #         self.log_dir,
            #         f"predictions/{self.dataset_name}/{self.name_prediction_file}",
            #     ),
            #     exist_ok=True,
            # )
            self.set_reference_objects()
            self.set_reference_object_pointcloud()
            self.set_last_token()
            self.set_grounding_info()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        # if batch["scene_id"][0] != 13 and batch["frame_id"][0] != 470:
        #     return 0

        if self.prompt_mode == "normal":
            foreground_prompt_locations, foreground_prompt_map = (
                self.generate_foreground_prompt(batch, threshold=self.sim_thresh)
            )
        elif self.prompt_mode == "self_grounding":
            foreground_prompt_locations, foreground_prompt_map = (
                self.generate_foreground_prompt_sg(batch, threshold=self.sim_thresh)
            )
        else:
            raise NotImplementedError

        if len(foreground_prompt_locations) == 0:
            return 0

        #  scale the prompt to image size
        foreground_prompt_locations[..., 0] = (
            foreground_prompt_locations[..., 0] / self.descriptor_model.full_size[1]
        ) * batch["image"].shape[-1]
        foreground_prompt_locations[..., 1] = (
            foreground_prompt_locations[..., 1] / self.descriptor_model.full_size[0]
        ) * batch["image"].shape[-2]

        grid_size = (32, 32)
        grid_prompt_locations = self.segmentor_model.generate_patch_grid_points(
            grid_size,
            14,
            device=self.device,
            corners=False,
        )
        grid_prompt_locations = grid_prompt_locations.flatten(0, 1)
        #  scale the prompt to image size
        grid_prompt_locations[..., 0] = (
            grid_prompt_locations[..., 0] / (14 * grid_size[0])
        ) * batch["image"].shape[-1]
        grid_prompt_locations[..., 1] = (
            grid_prompt_locations[..., 1] / (14 * grid_size[1])
        ) * batch["image"].shape[-2]

        mask_visib_path = os.path.join(batch["scene_path"][0], "mask_visib")
        foreground_masks = self.find_and_convert_images(
            mask_visib_path, str(batch["frame_id"][0].item()).zfill(6)
        )
        foreground_masks = foreground_masks.sum(dim=0).squeeze(0)

        # prompt locations to pixes coordinates
        foreground_prompt_locations = foreground_prompt_locations[:, [1, 0]].int()
        grid_prompt_locations = grid_prompt_locations[:, [1, 0]].int()

        prompt_foreground_map = foreground_masks[
            foreground_prompt_locations[:, 0], foreground_prompt_locations[:, 1]
        ]
        grid_foreground_map = foreground_masks[
            grid_prompt_locations[:, 0], grid_prompt_locations[:, 1]
        ]

        self.sg_quality.append(prompt_foreground_map.mean().item())
        self.grid_quality.append(grid_foreground_map.mean().item())

    def find_and_convert_images(self, directory, prefix):
        # Transform to convert images to PyTorch tensors
        transform = T.Compose(
            [
                T.ToTensor(),  # Convert image to tensor
            ]
        )

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
                    tensor = transform(img).to(self.device)
                    image_tensors.append(tensor)

        return torch.stack(image_tensors)

    def test_epoch_end(self, outputs):
        quality_path = f"{self.log_dir}/{self.dataset_name}_prompt_quality.csv"

        # average the list
        sg_quality = sum(self.sg_quality) / len(self.sg_quality)
        grid_quality = sum(self.grid_quality) / len(self.grid_quality)

        # save the quality lists to csv
        with open(quality_path, "w") as f:
            f.write("grid, sg\n")
            f.write(f"{grid_quality}, {sg_quality}\n")


class prompted_segmentation_slerp(Instance_Segmentation_Model):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        visible_thred,
        pointcloud_sample_num,
        sim_thresh,
        sg_thresh,
        prompt_mode,
        score_mode,
        weight_scores,
        num_templates,
        **kwargs,
    ):
        super().__init__(
            segmentor_model,
            descriptor_model,
            onboarding_config,
            matching_config,
            post_processing_config,
            log_interval,
            log_dir,
            visible_thred,
            pointcloud_sample_num,
            weight_scores,
            **kwargs,
        )
        self.sim_thresh = sim_thresh
        self.sg_thresh = sg_thresh
        self.prompt_mode = prompt_mode
        self.score_mode = score_mode
        self.num_templates = num_templates

        if self.prompt_mode == "self_grounding":
            for _, module in self.descriptor_model.named_modules():
                if module.__class__.__name__ in [
                    "Attention",
                    "MemEffAttention",
                ]:  # Match by class name
                    setattr(module, "sg_threshold", self.sg_thresh)

    def test_step(self, batch, idx):
        if idx == 0:
            # os.makedirs(
            #     osp.join(
            #         self.log_dir,
            #         f"predictions/{self.dataset_name}/{self.name_prediction_file}",
            #     ),
            #     exist_ok=True,
            # )
            self.set_reference_objects()
            self.set_reference_object_pointcloud()
            self.set_last_token()
            self.set_grounding_info()
            self.move_to_device()
            self.refined_obj_embeddings = self.generate_object_embeddings(
                self.prompt_mode == "self_grounding"
            )
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        proposal_stage_start_time = time.time()

        # if batch["scene_id"][0] != 13 and batch["frame_id"][0] != 470:
        #     return 0

        if self.prompt_mode == "normal":
            foreground_prompt_locations, foreground_prompt_map = (
                self.generate_foreground_prompt(
                    batch, self.refined_obj_embeddings, threshold=self.sim_thresh
                )
            )
        elif self.prompt_mode == "self_grounding":
            foreground_prompt_locations, foreground_prompt_map = (
                self.generate_foreground_prompt_sg(
                    batch, self.refined_obj_embeddings, threshold=self.sim_thresh
                )
            )
        else:
            raise NotImplementedError

        if len(foreground_prompt_locations) == 0:
            return 0

        if len(foreground_prompt_locations) > 1024:
            foreground_prompt_locations = foreground_prompt_locations[
                random.sample(range(len(foreground_prompt_locations)), 1024)
            ]

        # Visualize the prompt map
        # -------------------------------------------------------------------------------------------
        # import torch.nn.functional as F
        # from torchvision import utils as vutils

        # foreground_prompt_map = foreground_prompt_map.permute(2, 0, 1).float()
        # foreground_prompt_map = F.interpolate(
        #     foreground_prompt_map.unsqueeze(0), size=(480, 640), mode="nearest"
        # )

        # vutils.save_image(foreground_prompt_map, f"mask_prompts_{self.sim_thresh}.png")
        # foreground_prompt_map[foreground_prompt_map == 0] = 0.2
        # result = foreground_prompt_map * self.inv_rgb_transform(batch["image"])

        # vutils.save_image(
        #     result, f"positives_prompts_{self.dataset_name}_{self.sim_thresh}.png"
        # )
        # -------------------------------------------------------------------------------------------

        foreground_prompt_locations[..., 0] = (
            foreground_prompt_locations[..., 0] / self.descriptor_model.full_size[1]
        )
        foreground_prompt_locations[..., 1] = (
            foreground_prompt_locations[..., 1] / self.descriptor_model.full_size[0]
        )
        test_image_sized, point_prompt_scaled, _, _ = (
            self.segmentor_model.scale_image_prompt_to_dim(
                image=self.inv_rgb_transform(batch["image"][0]).unsqueeze(0),
                point_prompt=foreground_prompt_locations,
                # max_size=detector.input_size,
            )
        )
        _ = self.segmentor_model.encode_image(
            test_image_sized,
            original_image_size=batch["image"][0].shape[-2:],
        )
        seg_masks, seg_scores, stability_scores = (
            self.segmentor_model.segment_by_prompt(
                prompt=point_prompt_scaled,
                batch_size=64,
                score_threshould=0.88,
                stability_thresh=0.85,
            )
        )
        if len(seg_masks) == 0:
            return 0
        seg_masks = seg_masks > 0
        seg_masks = seg_masks.float()
        seg_scores = seg_scores.float()
        stability_scores = stability_scores.float()
        seg_boxes = self.segmentor_model.get_bounding_boxes_batch(seg_masks)
        keep_idxs = self.segmentor_model.batched_nms(
            boxes=seg_boxes.float(),
            scores=(seg_scores + stability_scores) / 2,
            idxs=torch.zeros(len(seg_masks)).to(self.device),  # categories
            iou_threshold=0.7,
        )
        seg_masks = seg_masks[keep_idxs]
        seg_scores = seg_scores[keep_idxs]
        seg_boxes = seg_boxes[keep_idxs]

        # init detections with masks and boxes
        detections = Detections({"masks": seg_masks.float(), "boxes": seg_boxes})
        detections.add_attribute("seg_scores", seg_scores)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )

        # compute semantic descriptors and appearance descriptors for query proposals
        try:
            query_decriptors, query_appe_descriptors = self.descriptor_model(
                batch["image"][0], detections
            )
        except:
            return 0

        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        if self.score_mode == "normal":
            appe_scores, ref_aux_descriptor = self.compute_appearance_score(
                best_template, pred_idx_objects, query_appe_descriptors
            )

            # compute the geometric score
            image_uv = self.project_template_to_image(
                best_template, pred_idx_objects, batch, detections.masks
            )

            geometric_score, visible_ratio = self.compute_geometric_score(
                image_uv,
                detections,
                query_appe_descriptors,
                ref_aux_descriptor,
                visible_thred=self.visible_thred,
            )
        elif self.score_mode == "self_grounding":
            if len(detections.masks.shape) == 4:
                detections.masks.squeeze_(1)
            grounding_info = self.ref_data["grounding_info"][pred_idx_objects]
            grounding_info = grounding_info.mean(dim=(0, 1)).unsqueeze(0)
            _, query_appe_descriptors = self.descriptor_model.forward_with_sg(
                batch["image"][0], detections, g_info=grounding_info
            )
            appe_scores, ref_aux_descriptor = self.compute_appearance_score_sg(
                best_template, pred_idx_objects, query_appe_descriptors
            )

            # compute the geometric score
            image_uv = self.project_template_to_image(
                best_template, pred_idx_objects, batch, detections.masks
            )

            geometric_score, visible_ratio = self.compute_geometric_score_sg(
                image_uv,
                detections,
                query_appe_descriptors,
                ref_aux_descriptor,
                visible_thred=self.visible_thred,
            )
        else:
            raise NotImplementedError

        final_score = (
            semantic_score + appe_scores + geometric_score * visible_ratio
        ) / (1 + 1 + visible_ratio)

        if self.weight_scores:
            final_score *= detections.seg_scores

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]

        # convert detections to coco format
        coco_result = detections.convert_to_coco_format(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            dataset_name=self.dataset_name,
        )
        for res in coco_result:
            self.final_results.append(res)
        return 0

    def slerp_batch(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Recursively apply SLERP over a list of normalized vectors.
        Args:
            vectors: Tensor of shape (k, d), assumed to be L2-normalized.
        Returns:
            A single normalized vector of shape (d,)
        """
        assert vectors.dim() == 2
        out = vectors[0]
        for i in range(1, vectors.shape[0]):
            a = out
            b = vectors[i]
            dot = (a * b).sum().clamp(-1.0, 1.0)
            theta = torch.acos(dot)
            if theta.abs() < 1e-5:
                out = a  # identical or nearly identical
            else:
                sin_theta = torch.sin(theta)
                t = 1.0 / (i + 1)
                out = (
                    torch.sin((1 - t) * theta) / sin_theta * a
                    + torch.sin(t * theta) / sin_theta * b
                )
                out = F.normalize(out, dim=0)
        return out

    def compute_common_token(
        self,
        all_tokens: torch.Tensor,
        topk: int = 10,
        chunk_size: int = 2048,
        is_sg=False,
    ) -> torch.Tensor:
        """
        Compute the common token by finding the most 'central' tokens across all object tokens,
        avoiding memory blowup by using chunked similarity.
        Args:
            all_tokens: (N, d) non-zero tokens
            topk: number of most common tokens to average
            chunk_size: how many tokens to compare at once
        Returns:
            common_embedding: (d,)
        """
        all_tokens = all_tokens[all_tokens.norm(dim=1) > 0]  # remove masked tokens
        if all_tokens.shape[0] == 0:
            return torch.zeros(all_tokens.shape[1], device=all_tokens.device)

        if is_sg:
            num_heads = self.descriptor_model.model.num_heads
            head_dim = all_tokens.shape[-1] // num_heads
            t, c = all_tokens.shape
            all_tokens_norm = all_tokens.view(t, num_heads, head_dim)
            all_tokens_norm = F.normalize(all_tokens, dim=-1)
            all_tokens_norm = all_tokens_norm.view(t, num_heads * head_dim)
        else:
            all_tokens_norm = F.normalize(all_tokens, dim=1)

        N = all_tokens.shape[0]
        avg_sim = torch.zeros(N, device=all_tokens.device)

        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk = all_tokens_norm[i:end]  # (chunk, d)
            sim = chunk @ all_tokens_norm.T  # (chunk, N)
            avg_sim[i:end] = sim.mean(dim=1)

        topk_idx = avg_sim.topk(min(topk, N)).indices
        selected = all_tokens[topk_idx]

        if is_sg:
            num_heads = self.descriptor_model.model.num_heads
            head_dim = selected.shape[-1] // num_heads
            t, c = selected.shape
            selected = selected.view(t, num_heads, head_dim)
            selected = F.normalize(selected, dim=-1)
            selected = selected.view(t, num_heads * head_dim)
        else:
            selected = F.normalize(selected, dim=1)

        return self.slerp_batch(selected)

    def compute_discriminative_embeddings(
        self, features: torch.Tensor, topk: int = 10, is_sg=False
    ):
        """
        Args:
            features: Tensor of shape (O, V, N, d), with background tokens = 0
            topk: Number of tokens to keep per object
        Returns:
            embeddings: Tensor of shape (O+1, d)
        """
        O, V, N, d = features.shape
        device = features.device

        # Flatten views and tokens: (O, V*N, d)
        obj_tokens = features.view(O, V * N, d)
        embeddings = []

        for o in range(O):
            this_tokens = obj_tokens[o]  # (V*N, d)
            mask = this_tokens.norm(dim=1) > 0  # non-zero tokens only
            valid_tokens = this_tokens[mask]  # (M, d)

            if valid_tokens.shape[0] == 0:
                # No valid tokens, fallback to zero vector
                z_o = torch.zeros(d, device=device)
            else:
                # Normalize
                if is_sg:
                    num_heads = self.descriptor_model.model.num_heads
                    head_dim = valid_tokens.shape[-1] // num_heads
                    t, c = valid_tokens.shape
                    valid_tokens_norm = valid_tokens.view(t, num_heads, head_dim)
                    valid_tokens_norm = F.normalize(valid_tokens_norm, dim=-1)
                    valid_tokens_norm = valid_tokens_norm.view(t, num_heads * head_dim)
                else:
                    valid_tokens_norm = F.normalize(valid_tokens, dim=1)

                # Intra-object similarity
                sim_intra = valid_tokens_norm @ valid_tokens_norm.T
                sim_intra_avg = sim_intra.mean(dim=1)

                # Inter-object tokens (excluding object o)
                other_tokens = torch.cat(
                    [
                        obj_tokens[i][obj_tokens[i].norm(dim=1) > 0]
                        for i in range(O)
                        if i != o
                    ],
                    dim=0,
                )  # ((O-1)*VN', d)

                if other_tokens.shape[0] == 0:
                    sim_inter_avg = torch.zeros_like(sim_intra_avg)
                else:
                    if is_sg:
                        num_heads = self.descriptor_model.model.num_heads
                        head_dim = other_tokens.shape[-1] // num_heads
                        t, c = other_tokens.shape
                        other_tokens_norm = other_tokens.view(t, num_heads, head_dim)
                        other_tokens_norm = F.normalize(other_tokens_norm, dim=-1)
                        other_tokens_norm = other_tokens_norm.view(
                            t, num_heads * head_dim
                        )
                    else:
                        other_tokens_norm = F.normalize(other_tokens, dim=1)
                    sim_inter = valid_tokens_norm @ other_tokens_norm.T
                    sim_inter_avg = sim_inter.mean(dim=1)

                # Score: more intra-similar, less inter-similar
                scores = sim_intra_avg - sim_inter_avg
                topk_idx = scores.topk(min(topk, valid_tokens.shape[0])).indices
                selected_tokens = valid_tokens[topk_idx]

                if is_sg:
                    num_heads = self.descriptor_model.model.num_heads
                    head_dim = selected_tokens.shape[-1] // num_heads
                    t, c = selected_tokens.shape
                    selected_tokens = selected_tokens.view(t, num_heads, head_dim)
                    selected_tokens = F.normalize(selected_tokens, dim=-1)
                    selected_tokens = selected_tokens.view(t, num_heads * head_dim)
                else:
                    selected_tokens = F.normalize(selected_tokens, dim=1)
                z_o = self.slerp_batch(selected_tokens)

            embeddings.append(z_o)

        # --------- Common token ---------
        all_tokens_flat = obj_tokens.view(-1, d)  # (O*VN, d)
        z_common = self.compute_common_token(all_tokens_flat, topk=topk, is_sg=is_sg)

        return torch.stack(embeddings + [z_common], dim=0)  # (O+1, d)

    def generate_object_embeddings(self, is_sg=False):
        obj_templates_feats = self.ref_data["last_token"]
        # features_mask = obj_templates_feats.sum(dim=-1) > 0
        # mask_sum = features_mask.sum(dim=-1)

        if is_sg:
            num_heads = self.descriptor_model.model.num_heads
            head_dim = obj_templates_feats.shape[-1] // num_heads
            o, t, wh, c = obj_templates_feats.shape
            obj_templates_feats = obj_templates_feats.view(
                o, t, wh, num_heads, head_dim
            )
            obj_templates_feats = F.normalize(obj_templates_feats, dim=-1)
            obj_templates_feats = obj_templates_feats.view(
                o, t, wh, num_heads * head_dim
            )
        else:
            obj_templates_feats /= torch.norm(obj_templates_feats, dim=-1, keepdim=True)

        obj_templates_feats = self.compute_discriminative_embeddings(
            obj_templates_feats, topk=10, is_sg=is_sg
        )
        return obj_templates_feats

    def generate_foreground_prompt(self, batch, obj_templates_feats, threshold=0.4):
        test_image_desc = self.descriptor_model.encode_full_size(batch["image"])[0]

        test_image_desc /= torch.norm(test_image_desc, dim=-1, keepdim=True)
        obj_templates_feats /= torch.norm(obj_templates_feats, dim=-1, keepdim=True)

        scene_obj_sim = test_image_desc @ obj_templates_feats.t()
        scene_obj_sim = scene_obj_sim.max(dim=-1)[0].unsqueeze(-1)

        scene_obj_sim = (scene_obj_sim - scene_obj_sim.min()) / (
            scene_obj_sim.max() - scene_obj_sim.min()
        )

        grid_prompt_locations = self.segmentor_model.generate_patch_grid_points(
            self.descriptor_model.output_spatial_size,
            self.descriptor_model.patch_size,
            device=self.device,
            corners=False,
        )
        foreground_prompt_map, foreground_prompt_locations = (
            self.segmentor_model.sim_2_point_prompts(
                scene_obj_sim=scene_obj_sim,
                grid_prompt_locations=grid_prompt_locations,
                spatial_size=self.descriptor_model.output_spatial_size,
                threshold=threshold,
            )
        )

        return foreground_prompt_locations, foreground_prompt_map

    def set_last_token(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data["last_token"] = BatchedData(None)
        last_token_path = osp.join(self.ref_dataset.template_dir, "last_token.pth")

        # Loading appearance descriptors
        if self.onboarding_config.rendering_type == "pbr":
            last_token_path = last_token_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(last_token_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["last_token"] = torch.load(last_token_path).to(self.device)
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing last tokens info ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                last_token = self.descriptor_model.compute_masked_patch_last_token(
                    ref_imgs, ref_masks
                )
                # last_token = last_token.mean(dim=0)
                self.ref_data["last_token"].append(last_token)

            self.ref_data["last_token"].stack()
            self.ref_data["last_token"] = self.ref_data["last_token"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["last_token"], last_token_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Query shape: {self.ref_data['last_token'].shape}, \
            Query descriptors shape: {self.ref_data['last_token'].shape}"
        )

    def generate_foreground_prompt_sg(self, batch, obj_templates_feats, threshold=0.4):
        grounding_info = self.ref_data["grounding_info"][:, : self.num_templates]
        # average all the objects to form a foreground grounding info
        grounding_info = grounding_info.mean(dim=(0, 1)).unsqueeze(0)

        test_image_desc = self.descriptor_model.encode_full_size(
            batch["image"], g_info=grounding_info
        )[0]

        num_heads = self.descriptor_model.model.num_heads
        head_dim = obj_templates_feats.shape[-1] // num_heads

        if grounding_info is not None:
            test_image_desc = test_image_desc.view(-1, num_heads, head_dim)
            obj_templates_feats = obj_templates_feats.view(-1, num_heads, head_dim)
            test_image_desc = F.normalize(test_image_desc, dim=-1)
            obj_templates_feats = F.normalize(obj_templates_feats, dim=-1)

        test_image_desc = test_image_desc.view(-1, num_heads * head_dim)
        obj_templates_feats = obj_templates_feats.view(-1, num_heads * head_dim)
        test_image_desc = F.normalize(test_image_desc, dim=-1)
        obj_templates_feats = F.normalize(obj_templates_feats, dim=-1)

        scene_obj_sim = test_image_desc @ obj_templates_feats.t()

        scene_obj_sim = (scene_obj_sim - scene_obj_sim.min()) / (
            scene_obj_sim.max() - scene_obj_sim.min()
        )

        grid_prompt_locations = self.segmentor_model.generate_patch_grid_points(
            self.descriptor_model.output_spatial_size,
            self.descriptor_model.patch_size,
            device=self.device,
            corners=False,
        )
        foreground_prompt_map, foreground_prompt_locations = (
            self.segmentor_model.sim_2_point_prompts(
                scene_obj_sim=scene_obj_sim,
                grid_prompt_locations=grid_prompt_locations,
                spatial_size=self.descriptor_model.output_spatial_size,
                threshold=threshold,
            )
        )

        return foreground_prompt_locations, foreground_prompt_map

    def set_grounding_info(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data["grounding_info"] = BatchedData(None)
        grounding_info_path = osp.join(
            self.ref_dataset.template_dir, "grounding_info.pth"
        )

        # Loading appearance descriptors
        if self.onboarding_config.rendering_type == "pbr":
            grounding_info_path = grounding_info_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(grounding_info_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["grounding_info"] = torch.load(grounding_info_path).to(
                self.device
            )
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing grounding info ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                ref_query = self.descriptor_model.compute_masked_patch_average_tokens(
                    ref_imgs, ref_masks
                )
                if len(ref_query) < 42:
                    average = ref_query.mean(dim=0).unsqueeze(0)
                    average = average.repeat(42 - len(ref_query), 1, 1, 1)
                    ref_query = torch.cat((ref_query, average), dim=0)
                # ref_query = ref_query.mean(dim=0)
                self.ref_data["grounding_info"].append(ref_query)

            self.ref_data["grounding_info"].stack()
            self.ref_data["grounding_info"] = self.ref_data["grounding_info"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["grounding_info"], grounding_info_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Query shape: {self.ref_data['grounding_info'].shape}, \
            Query descriptors shape: {self.ref_data['grounding_info'].shape}"
        )

    def compute_appearance_score_sg(
        self, best_pose, pred_objects_idx, qurey_appe_descriptors
    ):
        """
        Based on the best template, calculate appearance similarity indicated by appearance score
        """
        con_idx = torch.concatenate(
            (pred_objects_idx[None, :], best_pose[None, :]), dim=0
        )
        ref_appe_descriptors = self.ref_data["last_token"][
            con_idx[0, ...], con_idx[1, ...], ...
        ]  # N_query x N_patch x N_feature

        num_heads = self.descriptor_model.model.num_heads
        head_dim = qurey_appe_descriptors.shape[-1] // num_heads

        ref_appe_descriptors = ref_appe_descriptors.view(
            *ref_appe_descriptors.shape[:-1], num_heads, head_dim
        )
        ref_appe_descriptors = F.normalize(ref_appe_descriptors, dim=-1)
        ref_appe_descriptors = ref_appe_descriptors.view(
            *ref_appe_descriptors.shape[:-2], num_heads * head_dim
        )
        ref_appe_descriptors = F.normalize(ref_appe_descriptors, dim=-1)

        qurey_appe_descriptors = qurey_appe_descriptors.view(
            *qurey_appe_descriptors.shape[:-1], num_heads, head_dim
        )
        qurey_appe_descriptors = F.normalize(qurey_appe_descriptors, dim=-1)
        qurey_appe_descriptors = qurey_appe_descriptors.view(
            *qurey_appe_descriptors.shape[:-2], num_heads * head_dim
        )
        qurey_appe_descriptors = F.normalize(qurey_appe_descriptors, dim=-1)

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(
            qurey_appe_descriptors, ref_appe_descriptors
        )
        return appe_scores, ref_appe_descriptors

    def compute_geometric_score_sg(
        self,
        image_uv,
        proposals,
        qurey_appe_descriptors,
        ref_aux_descriptor,
        visible_thred=0.5,
    ):
        num_heads = self.descriptor_model.model.num_heads
        head_dim = qurey_appe_descriptors.shape[-1] // num_heads
        qurey_appe_descriptors = qurey_appe_descriptors.view(
            *qurey_appe_descriptors.shape[:-1], num_heads, head_dim
        )
        qurey_appe_descriptors = F.normalize(qurey_appe_descriptors, dim=-1)
        qurey_appe_descriptors = qurey_appe_descriptors.view(
            *qurey_appe_descriptors.shape[:-2], num_heads * head_dim
        )
        qurey_appe_descriptors = F.normalize(qurey_appe_descriptors, dim=-1)

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(
            qurey_appe_descriptors, ref_aux_descriptor, visible_thred
        )

        # IoU calculation
        y1x1 = torch.min(image_uv, dim=1).values
        y2x2 = torch.max(image_uv, dim=1).values
        xyxy = torch.concatenate((y1x1, y2x2), dim=-1)

        iou = compute_iou(xyxy, proposals.boxes)

        return iou, visible_ratio
