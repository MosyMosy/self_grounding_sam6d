import torch


# def sim_2_bbox_prompts(
#     scene_obj_sim,
#     foreground_prompt_map,
#     obj_point_size,
#     spatial_size,
#     patch_size,
#     obj_groups,
#     device,
#     count_threshold=1,
# ):
#     obj_idx = scene_obj_sim.argmax(dim=-1)
#     # obj_idx = obj_idx[obj_idx_score > 0.5]
#     obj_idx = obj_idx // obj_point_size

#     obj_idx = obj_idx.view(*spatial_size)

#     obj_group_id = obj_groups[obj_idx]
#     obj_group_id[foreground_prompt_map.squeeze(-1) == 0] = -1
#     labeled_group_id = ski.measure.label(
#         obj_group_id.cpu().numpy(), connectivity=2, background=-1
#     )
#     labeled_group_id = torch.from_numpy(labeled_group_id).to(device)

#     # set values with one repeated value to 0
#     unique_labels, label_counts = labeled_group_id.unique(return_counts=True)
#     for label, count in zip(unique_labels, label_counts):
#         if count <= count_threshold:
#             labeled_group_id[labeled_group_id == label] = 0

#     labeled_group_mask = labeled_image_to_masks(labeled_group_id)[1:]
#     labeled_group_bbox = get_bounding_boxes_batch(labeled_group_mask)

#     labeled_group_bbox *= patch_size
#     labeled_group_bbox[:, 1] += patch_size
#     labeled_group_bbox[:, 3] += patch_size

#     return labeled_group_id, labeled_group_bbox


def sim_2_point_prompts(
    scene_obj_sim, grid_prompt_locations, spatial_size, threshold=0.5
):
    if len(scene_obj_sim.shape) == 2:
        foreground_sim = scene_obj_sim.max(dim=1).values
    else:
        foreground_sim = scene_obj_sim
    foreground_prompt_map = foreground_sim.view(*spatial_size, -1) > threshold
    foreground_prompt_locations = grid_prompt_locations[
        foreground_prompt_map.squeeze(-1) == 1
    ]

    return foreground_prompt_map, foreground_prompt_locations


def sim_prompts_filter(
    scene_obj_sim, grid_prompt_locations, spatial_size, patch_size, threshold=0.5
):
    foreground_sim = scene_obj_sim.max(dim=1).values
    foreground_prompt_map = foreground_sim.view(*spatial_size, -1) > threshold

    foreground_prompt_filter = grid_prompt_locations // patch_size
    h_indices = foreground_prompt_filter[:, 1].int()
    w_indices = foreground_prompt_filter[:, 0].int()
    foreground_prompt_filter = foreground_prompt_map.squeeze(-1)[h_indices, w_indices]

    return foreground_prompt_map, grid_prompt_locations[foreground_prompt_filter]


def sim_2_point_prompts_2(
    scene_features, objs_features, grid_prompt_locations, spatial_size, threshold=0.5
):
    m, C = scene_features.shape
    n, p, C = objs_features.shape
    foreground_sim = scene_features @ objs_features.view(-1, C).t()
    # foreground_sim = scene_obj_sim.max(dim=1).values
    foreground_sim = foreground_sim > threshold
    foreground_sim = foreground_sim.view(m, n, p)
    foreground_sim = foreground_sim.sum(dim=-1)
    foreground_value, foreground_idx = foreground_sim.max(dim=-1)
    foreground_value = foreground_value.view(*spatial_size)
    foreground_idx = foreground_idx.view(*spatial_size)

    foreground_prompt_map = foreground_value > 0
    foreground_prompt_map = foreground_prompt_map
    foreground_prompt_locations = grid_prompt_locations[foreground_prompt_map == 1]

    return foreground_prompt_map.unsqueeze(-1), foreground_prompt_locations


def sim_prompts_filter_2(
    scene_features,
    objs_features,
    grid_prompt_locations,
    spatial_size,
    patch_size,
    threshold=0.5,
):
    m, C = scene_features.shape
    n, p, C = objs_features.shape
    foreground_sim = scene_features @ objs_features.view(-1, C).t()
    # foreground_sim = scene_obj_sim.max(dim=1).values
    foreground_sim = foreground_sim > threshold
    foreground_sim = foreground_sim.view(m, n, p)
    foreground_sim = foreground_sim.sum(dim=-1)
    foreground_value, foreground_idx = foreground_sim.max(dim=-1)
    foreground_value = foreground_value.view(*spatial_size)
    foreground_idx = foreground_idx.view(*spatial_size)

    foreground_prompt_map = foreground_value > 0

    foreground_prompt_filter = grid_prompt_locations // patch_size
    h_indices = foreground_prompt_filter[:, 1].int()
    w_indices = foreground_prompt_filter[:, 0].int()
    foreground_prompt_filter = foreground_prompt_map.squeeze(-1)[h_indices, w_indices]

    return foreground_prompt_map, grid_prompt_locations[foreground_prompt_filter]


def new_classification(
    scene_crops_feature, objs_sepaerate_features, k_view=10, k_class=1
):
    C = objs_sepaerate_features.shape[-1]

    scene_obj_sim = scene_crops_feature @ objs_sepaerate_features.view(-1, C).t()

    scene_obj_sim = scene_obj_sim.view(
        scene_crops_feature.shape[0],
        objs_sepaerate_features.shape[0],
        -1,
    )
    scene_obj_sim, top_template_idx = scene_obj_sim.topk(k_view, dim=-1)
    scene_obj_sim_mean = scene_obj_sim.mean(dim=-1)
    top_sim, top_obj_idx = scene_obj_sim_mean.max(dim=1)

    # top_sim, top_obj_idx = scene_obj_sim_mean.topk(k_class, dim=1)
    # if k_class == 1:
    #     top_obj_idx = top_obj_idx.squeeze(-1)
    #     top_sim = top_sim.squeeze(-1)

    # top_template_idx latest dim is sorted based on the max similarity
    best_selected_template_idx = top_template_idx[
        torch.arange(scene_crops_feature.shape[0]), top_obj_idx
    ]
    best_selected_template_idx = best_selected_template_idx[:, 0]

    return top_obj_idx, best_selected_template_idx, top_sim
    # print(f"Mean accuracy group: {mean_accuracey_group / len(test_ds)}")


def sim_2_point_prompts_feat_test(
    scene_features, objs_features, grid_prompt_locations, spatial_size, threshold=0.5
):
    m, C = scene_features.shape
    n, p, C = objs_features.shape
    foreground_sim = scene_features @ objs_features.view(-1, C).t()

    (
        estimated_obj_idx,
        best_selected_template_idx,
        estimated_score_cls,
    ) = new_classification(
        scene_features,
        objs_features,
    )

    # foreground_sim = scene_obj_sim.max(dim=1).values
    foreground_sim = foreground_sim > threshold
    foreground_sim = foreground_sim.view(m, n, p)
    foreground_sim = foreground_sim.sum(dim=-1)
    foreground_value, foreground_idx = foreground_sim.max(dim=-1)
    foreground_value = foreground_value.view(*spatial_size)
    foreground_idx = foreground_idx.view(*spatial_size)

    foreground_prompt_map = foreground_value > 0
    foreground_prompt_map = foreground_prompt_map
    foreground_prompt_locations = grid_prompt_locations[foreground_prompt_map == 1]

    return foreground_prompt_map.unsqueeze(-1), foreground_prompt_locations


def get_grid_coordinates(size, device, with_offset=False):
    H, W = size
    if with_offset:
        offset_i = 1 / (2 * W)
        offset_j = 1 / (2 * H)
    else:
        offset_i = 0
        offset_j = 0

    i = torch.linspace(offset_i, 1 - offset_i, W, device=device)
    j = torch.linspace(offset_j, 1 - offset_j, H, device=device)

    i, j = torch.meshgrid(i, j, indexing="ij")
    return i, j


def generate_patch_grid_points(size, patch_size, device, corners=False):
    i, j = get_grid_coordinates(size, device)
    i = i * patch_size * (size[1] - 1)
    j = j * patch_size * (size[0] - 1)
    if corners:
        i1, j1 = i + patch_size, j + patch_size
        i2, j2 = i, j + patch_size
        i3, j3 = i + patch_size, j

        i, j = (
            torch.stack([i, i1, i2, i3], dim=-1),
            torch.stack([j, j1, j2, j3], dim=-1),
        )
    else:
        # get the center of each patch
        i, j = i + patch_size // 2, j + patch_size // 2
    points = torch.stack([i, j], dim=-1)
    return points.transpose(0, 1)


def generate_grid_points(point_size, target_size, device, with_offset=True):
    i, j = get_grid_coordinates(point_size, device=device, with_offset=with_offset)
    # i, j = i / i.max(), j / j.max()
    i, j = i * target_size[1], j * target_size[0]
    points = torch.stack([i, j], dim=-1)
    return points
