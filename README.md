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

Download SAMPLE here: https://drive.google.com/file/d/1CSDqJGh7zWs2xjXEa533vCPEcINhP3NH/view?usp=sharing
Download SMPL models here: https://drive.google.com/file/d/1eF2DCk7GhbSAYfC8eKVFCU27P4VeKNPV/view?usp=sharing

Put the models_smplx_v1_1 in the main folder and the sample pose wherever you want

```
python retarget.py --human_pose PATH_TO_SMPL_POSE --robot ROBOT_NAME
```

Press ENTER to visualize the human side by side to the robot.
<video width="300" autoplay loop muted playsinline>
<source src="videos/hug.mp4" type="video/mp4">
</video>
<p align="center">
  <img src="images/high_five.gif" width="300"/>
  <img src="images/wave.gif" width="300"/>

</p>