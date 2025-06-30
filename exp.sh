# with sam
python run_inference.py dataset_name=icbin

# with fastsam
python run_inference.py dataset_name=icbin model=ISM_fastsam

# ==========================================================================================
for dataset in icbin ycbv tudl lmo tless itodd hb; do
	CUDA_VISIBLE_DEVICES=3 python run_inference.py \
	    model=ISM_sam_prompt \
	    prompt_mode=grid\
	    dataset_name=$dataset \
	    weight_scores=False \
	    model/segmentor_model=sam_2\
	    model.segmentor_model.multimask_output=true\
	    name_exp=SAM2_grid_multi_mask  
done


# ==========================================================================================
for dataset in icbin ycbv tudl lmo tless itodd hb; do
	CUDA_VISIBLE_DEVICES=2 python run_inference.py \
	    model=ISM_sam_prompt \
	    prompt_mode=grid\
	    dataset_name=$dataset \
	    weight_scores=False \
	    model/segmentor_model=sam_2\
	    model.segmentor_model.multimask_output=false\
	    name_exp=SAM2_grid_single_mask  
done


# ==========================================================================================
for dataset in icbin ycbv tudl lmo tless itodd hb; do
	CUDA_VISIBLE_DEVICES=2 python run_inference.py \
	    model=ISM_sam_prompt \
	    prompt_mode=self_grounding\
	    dataset_name=$dataset \
	    weight_scores=True \
	    model/segmentor_model=sam_main\
	    name_exp=SAM_self_grounding_single_mask  
done


# ==========================================================================================
for dataset in icbin ycbv tudl lmo tless itodd hb; do
	CUDA_VISIBLE_DEVICES=2 python run_inference.py \
	    model=ISM_sam_prompt \
	    prompt_mode=self_grounding\
	    dataset_name=$dataset \
	    weight_scores=True \
	    model/segmentor_model=sam_main\
	    name_exp=SAM_self_grounding_WA_selected_tokens
done

