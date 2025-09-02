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

To visualize and create a video run 
```
python robot_interaction.py --interaction PATH_TO_INTERACTION --robot1 ROBOT1_NAME --robot2 ROBOT2_NAME --video 
```
When the interactive window opens, adjust the camera as you like then press q. The interaction will start and the video will be recorded. Once the motion stops press q to end.

The output will be a video.mp4 file.

<p align="center">
  <img src="images/robot_play.gif" width="300"/>
  <img src="images/human_play.gif", width="300">
</p>
