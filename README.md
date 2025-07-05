# The Complete Robots

## Install environment


```
conda env create -f environment.yml
```
then 

```
conda activate Robots
```


## To run ROBOT EQUALIZER

```
python robot_translator.py --robot ROBOT_NAME         # --help to see list of available robots
```

This code will print the kinematic associations.

## TO visualize the INVERSE KINEMATICS PROCESS run

Download SAMPLE here: https://drive.google.com/file/d/1TjfyUXMrarAcnr8UCNNuf0zAdzOPrxK_/view?usp=sharing

Download SMPL models here: https://drive.google.com/file/d/1eF2DCk7GhbSAYfC8eKVFCU27P4VeKNPV/view?usp=sharing

Put the models_smplx_v1_1 in the main folder and the sample pose wherever you want

```
python retarget.py --human_pose PATH_TO_SMPL_POSE --robot ROBOT_NAME
```


You should see in RED the human pose, in GREEN the robot target pose, with BLUE Crosses the reached pose in pyplot.
After that the MeshCat Visualizer will open on you local host.
Press and key and ENTER to end the program.