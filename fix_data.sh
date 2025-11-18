#!/bin/bash
#SBATCH --job-name=complete
#SBATCH --output="logs/fix.out"
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrC_SDG-GS
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # One task (process) total
#SBATCH --cpus-per-task=1                 # One CPU core per task
#SBATCH --time=04:00:00 
#SBATCH --mem=2G                          


module load anaconda3/2023.09-0
source activate /leonardo/home/userexternal/rcatalin/.conda/envs/Robots
python fix_annotations.py