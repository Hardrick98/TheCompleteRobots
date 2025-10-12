# The Complete Robots

We are creating a dataset of robots interacting with each other. 

## Install environment


```
conda env create -f environment.yml
```
then 

```
conda activate Robots
```


## To use the code


Download SMPL-X models here: https://drive.google.com/file/d/1eF2DCk7GhbSAYfC8eKVFCU27P4VeKNPV/view?usp=sharing

Put the models_smplx_v1_1 in the main folder.

Download the Inter-X dataset https://github.com/liangxuy/Inter-X, and extract the motion folder wherever you want. That will be your PATH_TO_INTERACTION.


Retargeting pose from human to robot:

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

To create videos 

```
python cool_visualization.py --robot1 ROBOT1 --robot2 ROBOT2 --interaction PATH_TO_INTERACTION --scene SCENE --video --camera_mode CAMERA MODE
```



Alternatively you can use ./compute_data.sh INTERACTION, tp do it all at once.


## Sample of interaction:

<table align="center">
  <tr>
    <td align="center">
      <img src="images/nao_exoL.gif" width="400"/><br>
      <sub>Exo Camera L</sub>
    </td>
    <td align="center">
      <img src="images/nao_exoR.gif" width="400"/><br>
      <sub>Exo Camera R</sub>
    </td>
  </tr>
</table>


<table align="center">
  <tr>
    <td align="center">
      <img src="images/nao_ego1L.gif" width="400"/><br>
      <sub>Robot1 Camera L</sub>
    </td>
    <td align="center">
      <img src="images/nao_ego1R.gif" width="400"/><br>
      <sub>Robot1 Camera R</sub>
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center">
      <img src="images/nao_ego2L.gif" width="400"/><br>
      <sub>Robot2 Camera L</sub>
    </td>
    <td align="center">
      <img src="images/nao_ego2R.gif" width="400"/><br>
      <sub>Robot2 Camera R</sub>
    </td>
  </tr>
</table>


The displayed action is **Slap**.