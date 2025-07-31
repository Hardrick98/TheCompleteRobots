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
python robot_interaction.py --interaction PATH_TO_INTERACTION --robot ROBOT_NAME
```
After the visualization stops you can adjust the camera and press q to save camera parameters. Those will be loaded in the next execution.


The output will be a video.mp4 file.

<p align="center">
  <img src="images/wave.gif" width="600"/>
</p>