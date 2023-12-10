#!/bin/bash

set -e
sudo -u ec2-user -i <<'EOF'
conda activate pytorch_p310
pip install accelerate==0.22.0 bitsandbytes===0.41.1 diffusers==0.20.0 pytorch-fid==0.3.0 \
            pytorch_lightning==2.0.7 safetensors==0.3.3 segment-anything==1.0 supervision==0.8.0 \
            torch==2.0.1 transformers==4.32.0 wandb==0.15.8 wget==3.2 xformers==0.0.21
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/openai/CLIP.git
cd SageMaker
git clone https://github.com/christophschuhmann/improved-aesthetic-predictor.git
conda deactivate
EOF