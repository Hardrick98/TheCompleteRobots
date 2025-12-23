#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output="logs/chunk_fix_%a.out"
#SBATCH --array=0-1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --account=IscrC_SDG-GS
#SBATCH --time=1-00:00:00
#SBATCH --mem=40G
##SBATCH --cpus-per-gpu=4

echo "$(date)"
module load anaconda3/2023.09-0
source activate /leonardo/home/userexternal/rcatalin/.conda/envs/Robots
#module load ffmpeg/7.1-gcc-11.4.0
export PYOPENGL_PLATFORM=egl

DATASET="/leonardo_work/IscrC_SDG-GS/TheCompleteRobots/errors.csv"

# Numero di righe da processare per job
BATCH_SIZE=13

# Calcola le righe di inizio/fine per questo job
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE + 2 ))  # +2 per saltare intestazione
END=$(( START + BATCH_SIZE - 1 ))

echo "Processing lines $START to $END from dataset.csv"

# Leggi solo le righe in questo intervallo
awk -v s="$START" -v e="$END" 'NR>=s && NR<=e' "$DATASET" | while IFS=',' read -r idx interaction robot scene
do
    INTERACTION_PATH="/leonardo_scratch/large/userexternal/rcatalin/robot_dataset/$interaction"
    echo "[$(date)] Processing $interaction | Robot=$robot | Scene=$scene"

    # --- Esegui i tuoi script Python ---
    python -u retarget_motion.py --robot "$robot" --interaction "$INTERACTION_PATH"
    python -u compute_data.py --robot1 "$robot" --robot2 "$robot" --interaction "$INTERACTION_PATH"

    echo "Rendering videos for $robot..."
    for cam in exoR ego1R ego2R exoL ego1L ego2L; do
        python -u render.py \
            --interaction "$INTERACTION_PATH" \
            --robot1 "$robot" \
            --robot2 "$robot" \
            --scene "scenes/$scene" \
            --frames \
            --camera_mode "$cam"
    done

    echo "Computing Bounding Boxes..."
    python -u extract_bb.py --interaction "$INTERACTION_PATH" --robot1 "$robot" --robot2 "$robot" --green_screen --camera_mode exoR --bb_mode1
    python -u extract_bb.py --interaction "$INTERACTION_PATH" --robot1 "$robot" --robot2 "$robot" --green_screen --camera_mode exoR --bb_mode2
    python -u extract_bb.py --interaction "$INTERACTION_PATH" --robot1 "$robot" --robot2 "$robot" --green_screen --camera_mode exoL --bb_mode1
    python -u extract_bb.py --interaction "$INTERACTION_PATH" --robot1 "$robot" --robot2 "$robot" --green_screen --camera_mode exoL --bb_mode2
    python -u extract_bb.py --interaction "$INTERACTION_PATH" --robot1 "$robot" --robot2 "$robot" --green_screen --camera_mode ego1L --bb_mode2
    python -u extract_bb.py --interaction "$INTERACTION_PATH" --robot1 "$robot" --robot2 "$robot" --green_screen --camera_mode ego2L --bb_mode1
    python -u extract_bb.py --interaction "$INTERACTION_PATH" --robot1 "$robot" --robot2 "$robot" --green_screen --camera_mode ego1R --bb_mode2
    python -u extract_bb.py --interaction "$INTERACTION_PATH" --robot1 "$robot" --robot2 "$robot" --green_screen --camera_mode ego2R --bb_mode1


    echo "Script completed for $interaction ($robot, $scene)!"
done
echo "$(date)"
echo "Job completed!"



