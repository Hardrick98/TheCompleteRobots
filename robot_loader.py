from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    robot_list = [r.removesuffix(".urdf") for r in os.listdir("/home/rcatalini/TheCompleteRobot/URDF") if r.endswith(".urdf") or r.endswith(".urdf")]
    
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
    args  = parser.parse_args()
    robot_name = args.robot.lower()    
    print(robot_name)
    try:
        robot = HumanoidRobot(f"/home/rcatalini/TheCompleteRobot/URDF/{args.robot}.urdf")
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
    pose_dict, keypoints = robot.get_joints(robot, robot.q0)
    joints, limbs = robot.get_physical_joints()
    
    
    
    #joints = np.concatenate((np.zeros((1,3)),joints), axis=0)
    
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', marker='o')
    for joint1_idx, joint2_idx in limbs:
        if all(len(joint) == 3 for joint in (joints[joint1_idx], joints[joint2_idx])):
            x_coords, y_coords, z_coords = zip(joints[joint1_idx], joints[joint2_idx])
            ax.plot(x_coords, y_coords, z_coords, c="red", linewidth=2)
    
    plt.show()
    
    
    human36m_joints = { 0: 'root', 1: 'rhip', 2: 'rkne', 3: 'rank', 4: 'lhip', 5: 'lkne', 6: 'lank',
    7: 'belly', 8: 'neck', 9: 'nose', 10: 'head', 11: 'lsho', 12: 'lelb', 13: 'lwri',
    14: 'rsho', 15: 'relb', 16: 'rwri'}
