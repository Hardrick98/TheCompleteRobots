#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --output=logs/debug_videos.out
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrC_SDG-GS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --mem=8G

module load anaconda3/2023.09-0
source activate /leonardo/home/userexternal/rcatalin/.conda/envs/Robots


python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "nao" -c "exoR" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "nao" -c "exoL" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "nao" -c "ego1R" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "nao" -c "ego2R" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "nao" -c "ego1L" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "nao" -c "ego2L" &

wait


python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "g1" -c "exoR" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "g1" -c "exoL" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "g1" -c "ego1R" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "g1" -c "ego1L" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "g1" -c "ego2L" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "g1" -c "ego2R" &

wait

python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "pepper" -c "exoR" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "pepper" -c "exoL" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "pepper" -c "ego1R" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "pepper" -c "ego2R" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "pepper" -c "ego1L" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "pepper" -c "ego2L" &

wait

python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "atlas" -c "exoR" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "atlas" -c "exoL" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "atlas" -c "ego1R" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "atlas" -c "ego2R" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "atlas" -c "ego1L" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "atlas" -c "ego2L" &

wait

python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "icub" -c "exoR" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "icub" -c "exoL" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "icub" -c "ego1R" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "icub" -c "ego2R" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "icub" -c "ego1L" &
python debug_pose_annotation.py -i /leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G058T007A024R010 -r "icub" -c "ego2L" &

wait
echo "Done"