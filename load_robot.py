from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mesh import compose_hand_mesh


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

    #compose_hand_mesh(model, visual_model, "RWristYaw")


    viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True) 
    viz.loadViewerModel()

   
    viz.display(q0)
    plt.show()
    input("Press Enter to reset the visualization...")
    viz.reset()
    
    
    