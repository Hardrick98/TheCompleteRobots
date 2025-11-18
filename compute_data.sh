#!/bin/bash
#SBATCH --job-name=complete
#SBATCH --output="logs/last.out"
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:0
#SBATCH --account=IscrC_SDG-GS
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8

module load anaconda3/2023.09-0
source activate /leonardo/home/userexternal/rcatalin/.conda/envs/Robots
export PYOPENGL_PLATFORM=egl
#python retarget_motion.py --robot icub --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004
#python compute_data.py --robot1 icub --robot2 icub --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004
#
echo "Rendering videos..."
#
#python render.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --scene scenes/city.glb --frames --camera_mode exoR
#python render.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --scene scenes/city.glb --frames --camera_mode ego1R
python render.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --scene scenes/city.glb --frames --camera_mode ego2R
python render.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --scene scenes/city.glb --frames --camera_mode exoL
python render.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --scene scenes/city.glb --frames --camera_mode ego1L
python render.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --scene scenes/city.glb --frames --camera_mode ego2L

echo "Computing Bounding Boxes..."


python extract_bb.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --green_screen --camera_mode exoR --bb_mode1
python extract_bb.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --green_screen --camera_mode exoR --bb_mode2
python extract_bb.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --green_screen --camera_mode exoL --bb_mode1
python extract_bb.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --green_screen --camera_mode exoL --bb_mode2
python extract_bb.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --green_screen --camera_mode ego1L --bb_mode2
python extract_bb.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --green_screen --camera_mode ego2L --bb_mode1
python extract_bb.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --green_screen --camera_mode ego1R --bb_mode2
python extract_bb.py --interaction /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G011T006A031R004 --robot1 icub --robot2 icub --green_screen --camera_mode ego2R --bb_mode1




echo "Script completed!"