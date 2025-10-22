from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    robot_list = [r.removesuffix(".urdf") for r in os.listdir("URDF") if r.endswith(".urdf") or r.endswith(".urdf")]
    
    parser = argparse.ArgumentParser(description="Visualize a humanoid robot model.")
    parser.add_argument(
        "--robot",
        type=str,
        default="nao",
        help="The robot to visualize.",
    )
    parser.add_argument("--visualize",
                        action="store_true",
                        help="Visualize the robot model in Meshcat.")
    parser.add_argument("--human_pose",
                        type=str,
                        help="Path to smpl human pose")
    args  = parser.parse_args()
    robot_name = args.robot.lower()    
    print(robot_name)
    try:
        robot = HumanoidRobot(f"URDF/{args.robot}.urdf")
    except Exception as e:
        print(f"Error loading robot {robot_name}: {e}")
        print("Available robots:")
        for r in robot_list:
            print(f"- {r}")
        exit(1)
    
    if args.visualize:
        viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz.initViewer(open=True) 
        viz.loadViewerModel()
        viz.display(robot.q0)
    pose_dict, keypoints = robot.get_joints(robot.q0)
    robot_joints, robot_limbs = robot.get_physical_joints()
    
    
    
    model = robot.model
    data = robot.data
    q0 = robot.q0  
    visual_model = robot.visual_model

    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    print(robot.joints)
    print(q0.shape)
    #print('root_joint',data.oMf[1].translation)
    #print('HipRoll',data.oMf[37].translation)
    #print('KneePitch',data.oMf[41].translation)
    #print('HipPitch',data.oMf[39].translation)
    #print('WheelB', data.oMf[73].translation)
    #print('WheelFL', data.oMf[75].translation)
    #print('WheelFR', data.oMf[77].translation)

    #compose_hand_mesh(model, visual_model, "RWristYaw")

    """
    dict = {}
    for visual in visual_model.geometryObjects:
        placement = visual.placement
        rot = placement.rotation
        t = placement.translation

        T = np.eye(4)
        T[:3,:3] = rot
        T[:3,3] = t

        dict[visual.name] = T

    import joblib
    print(dict)
    joblib.dump(dict, "icub_init.pkl")
    """



    viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True) 
    viz.loadViewerModel()

   
    viz.display(q0)
    plt.show()
    input("Press Enter to reset the visualization...")
    viz.reset()
    
    
    
    