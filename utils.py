import os
import pinocchio as pin

def load_robot(urdf_path): 

    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )
    print(f"URDF description successfully loaded in {robot}")
    
    print(f"Number of DoF: {len(robot.q0)}")
    
    return robot

def forwardK(robot, q):
    
    pin.forwardKinematics(robot.model, robot.data, q)
    
    print("Target pose reached")
    
def get_joints(robot):
    
    pose_dict = {}
    for joint in robot.model.joints:
        joint_id = joint.id
        if joint_id == 18446744073709551615:
            joint_id = 0
        joint_name = robot.model.names[joint_id]
        pos = robot.model.jointPlacements[joint_id]
        abs = robot.data.oMi[joint_id] 
        pose_dict[joint_name] = joint_id
    
    return pose_dict