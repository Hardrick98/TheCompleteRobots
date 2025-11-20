#!/bin/bash
#SBATCH --job-name=YOLO
#SBATCH --output="logs/yolo.out"
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --account=IscrC_SDG-GS
#SBATCH --time=1-00:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8
#SBATCH --dependency=afterany:26768944

module load anaconda3/2023.09-0
source activate /leonardo/home/userexternal/rcatalin/.conda/envs/Robots
python train_yolo.py
#python predict.py