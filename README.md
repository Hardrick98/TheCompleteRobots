# The Complete Robots

## Install environment


```
conda env create -f environment.yml
```
then 

```
conda activate Robots
```


## TO find robot configurations


Download SMPL models here: https://drive.google.com/file/d/1eF2DCk7GhbSAYfC8eKVFCU27P4VeKNPV/view?usp=sharing

Put the models_smplx_v1_1 in the main folder.
You can use the interaction sample to test the code.

```
python retarget_motion.py --interaction PATH_TO_INTERACTION --robot ROBOT_NAME
```

If you want only one pose run:

```
python retarget_pose.py --interaction PATH_TO_INTERACTION --robot ROBOT_NAME
```

This will save robot configurations in the same folder.

To extract data like poses and collisions run:
```
python robot_interaction.py --interaction PATH_TO_INTERACTION --robot1 ROBOT1_NAME --robot2 ROBOT2_NAME --video 
```

To create visualization 

```
python cool_visualization.py --robot1 ROBOT1 --robot2 ROBOT2 --interaction PATH_TO_INTERACTION --scene SCENE --video --camera_mode CAMERA MODE
```



<p align="center">
  <img src="images/human_play.gif" width="300"/>
  <img src="images/robot_play.gif", width="300">
</p>

On the left humans on the right robots playing ROCK-PAPER-SCISSORS