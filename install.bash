# Create the Conda environment
conda env create -f environment.yml

# Activate the environment
conda activate SG-SAM

pip install pip==23.3.2
pip install pytorch-lightning==1.8.1

# Clone the sam2 repository
cd model
git clone https://github.com/facebookresearch/sam2.git
cd sam2
conda run -n sam6d-ism pip install -e .
cd ../..