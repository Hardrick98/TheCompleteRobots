from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from robotoid import Robotoid, HumanAction
from smplx import SMPLX
from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
from vedo import Plotter, Mesh
import os


if __name__ == "__main__":
    
    robot_list = [r.removesuffix(".urdf") for r in os.listdir("URDF") if r.endswith(".urdf") or r.endswith(".urdf")]
    
    parser = argparse.ArgumentParser(description="Retarget human to robot")
    parser.add_argument(
        "--robot",
        type=str,
        default="nao",
        help="The robot to visualize.",
    )
    parser.add_argument("--debug",
                        action="store_true",
                        help="Enter debug mode with plt")
    parser.add_argument("--interaction",
                        type=str,
                        help="Path to smpl human pose")
    parser.add_argument("--idx",
                        type=int, default=0,
                        help="Path to smpl human pose")
    parser.add_argument("--visualize",
                        action="store_true",
                        help="If to visualize video or not at the end")
    args  = parser.parse_args()
    robot_name = args.robot.lower() 
    idx = args.idx

    print(robot_name)
    try:
        robot = HumanoidRobot(f"URDF/{args.robot}.urdf")
    except Exception as e:
        print(f"Error loading robot {robot_name}: {e}")
        print("Available robots:")
        for r in robot_list:
            print(f"- {r}")
        exit(1)  
    
    
    model = robot.model
    data = robot.data
    q0 = robot.q0  

    smpl_model = SMPLX(
        model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
        gender='male', 
        batch_size=8
    ).to("cuda:0")

    wheeled = False
    if args.robot == "pepper":
        wheeled = True
    
    print(robot.get_frames())

    robotoid1 = Robotoid(robot, wheeled)
    robotoid2 = Robotoid(robot, wheeled)
    
    
    file1 = args.interaction + "/P2.npz"
    file2 = args.interaction + "/P1.npz"
    action1 = np.load(file1, allow_pickle=True)
    action2 = np.load(file2, allow_pickle=True)

    human_action1 = HumanAction(action1)
    human_action2 = HumanAction(action2)
    
    w_pos, w_ori = robotoid1.optimize(human_action1, idx=0)

    print(w_pos)
    print(w_ori)