from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import numpy as np
import torch.nn as nn
import torch   


if __name__ == "__main__":
    

    
    robot = load_robot("URDF/h1_2_handless.urdf")

    
    
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True) 
    viz.loadViewerModel()

    
    pose_dict = get_joints(robot)
    
    pin.forwardKinematics(robot.model, robot.data, robot.q0)
    
    robot_pose = []
    
    for joint_name, joint_id in pose_dict.items():
        T_world_joint = robot.data.oMi[joint_id]  # oMi = world to joint
        translation = T_world_joint.translation
        robot_pose.append(translation)
        
    
    robot_pose1 = np.array(robot_pose)
    print(robot_pose)
    
    array = np.load("/home/rcatalini/UH-1/output/Kick_something.npy", allow_pickle=True)
    #print(array.shape)
    #sample = dict(array[()])
    sample = array[0,:,:27]
    
    q = sample[10]
    
    forwardK(robot, q)
    robot_pose = []
    
    for joint_name, joint_id in pose_dict.items():
        T_world_joint = robot.data.oMi[joint_id]  # oMi = world to joint
        translation = T_world_joint.translation
        robot_pose.append(translation)
    
    robot_pose2 = np.array(robot_pose)
    
    
    print((robot_pose2-robot_pose1).sum(axis=1))
    
    robot = load_robot("URDF/h1_2_handless.urdf")
    
    q0 = robot.q0

#    for i in range(len(sample["dof_pos"])):
#        q = sample["dof_pos"][i]
#        forwardK(robot, q)
#        viz.display(q)
#        time.sleep(0.05) 

    """

    for i in range(len(sample)):
        q = sample[i]
        forwardK(robot, q)
        viz.display(q)
        time.sleep(0.05) 
    """
    
    
    #viz.clean()
    
    """
    with open("nao.json", "r") as f:
        nao_poses = json.load(f)[0]
    
    base_pose = nao_poses["base_pose"]
    q = robot.q0
    """
    
    #q[36] = q[28]
    #q[37] = q[29]
    #q[38] = q[30]
    #q[39] = q[31]
    #q[40] = q[32]    
    #q[28] = 0.0
    #q[29] = 0.0
    #q[30] = 0.0
    #q[31] = 0.0
    #q[32] = 0.0
    #q[33] = 0.0

    #pin.forwardKinematics(robot.model, robot.data, q)
    
    """
    input("PRESS ENTER TO CONTINUE") 
    
    


    pin.updateGeometryPlacements(robot.model, robot.data, robot.visual_model, robot.visual_data)

    joints = robot.visual_model.geometryObjects.tolist()
    joints_names = [joints[i].name for i in range(len(joints))]
   
    positions = []
    for i, name in enumerate(joints_names):
        placement = robot.visual_data.oMg[i]  # trasformazione SE3 (rispetto al mondo)
        pos = placement.translation           # posizione (x, y, z)
        positions.append(pos)
        if "Finger" not in name and "Thumb" not in name:
            print(f"{name}: pos = {pos}")

    positions = np.array(positions)
    
    max_pos = np.max(positions[:,2])
    min_pos = np.min(positions[:,2])
    
    print(max_pos)
    print(min_pos)
    print("Robot Height: ", max_pos-min_pos)
    """