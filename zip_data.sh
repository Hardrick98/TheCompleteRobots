#!/bin/bash
#SBATCH --job-name=ziparray
#SBATCH --output=logs/zip_%A_%a.out
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrC_SDG-GS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --array=0-9     # <-- 10 job paralleli (0..9)

# carico la cartella corrispondente all'indice dell'array
folder=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" folders.txt)

echo "Zipping folder: $folder"


zip -r "${folder}.zip" "$folder"

echo "Done zipping $folder"