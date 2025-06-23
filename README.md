# Prompt Generation for Model-Based Instance Segmentation of Unseen Objects


## Requirements
Create conda environment:

```
conda env create -f environment.yml
conda activate SG-SAM

pip install pip==23.3.2
pip install pytorch-lightning==1.8.1

# for using SAM
pip install git+https://github.com/facebookresearch/segment-anything.git


# for using fastSAM
pip install ultralytics==8.0.135
```


## Data Preparation

Please refer to [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Data)] for more details.


## Foundation Model Download

Download model weights of [Segmenting Anything](https://github.com/facebookresearch/segment-anything):
```
python download_sam.py
```

Download model weights of [Fast Segmenting Anything](https://github.com/CASIA-IVA-Lab/FastSAM):
```
python download_fastsam.py
```

Download model weights of ViT pre-trained by [DINOv2](https://github.com/facebookresearch/dinov2):
```
python download_dinov2.py
```


## Evaluation on BOP Datasets

To evaluate the model on BOP datasets, please run the following commands:

```
# Specify a specific GPU
export CUDA_VISIBLE_DEVICES=0

# For our MPG (all datasets):
for dataset in icbin ycbv tudl lmo tless itodd hb; do python run_inference.py model=ISM_sam_prompt prompt_mode=normal weight_scores=False name_exp=prompt dataset_name=$dataset; done

# For our MPG + SG (all datasets):
for dataset in icbin ycbv tudl lmo tless itodd hb; do python run_inference.py model=ISM_sam_prompt prompt_mode=self_grounding weight_scores=False name_exp=prompt dataset_name=$dataset; done

# For our MPG + SG + CW(all datasets):
for dataset in icbin ycbv tudl lmo tless itodd hb; do python run_inference.py model=ISM_sam_prompt prompt_mode=self_grounding weight_scores=True name_exp=prompt dataset_name=$dataset; done
```

To rebuild all the figures and analysis, use the scripts provided in the lab directory:

## Acknowledgments

This code is based on the [SAM6D](https://github.com/JiehongLin/SAM-6D) codebase. We appreciate all their efforts and thank them for sharing the code.

                                                              
