#!/bin/bash
#SBATCH --partition=dev_gpu_4
#SBATCH --gres=gpu:1
#SBATCH --time=10
#SBATCH --output=/home/kit/stud/ucxam/dev/ArtXFashion/output.txt

python3 segmentation/cloth_segmentation.py main --config_path /home/kit/stud/ucxam/dev/ArtXFashion/segmentation/config.json
