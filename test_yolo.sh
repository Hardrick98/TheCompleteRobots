#!/bin/bash
#SBATCH --job-name=YOLO
#SBATCH --output="logs/yolo.out"
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --account=IscrC_SDG-GS
#SBATCH --time=00:05:00
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8

module load anaconda3/2023.09-0
source activate /leonardo/home/userexternal/rcatalin/.conda/envs/Robots
python predict.py