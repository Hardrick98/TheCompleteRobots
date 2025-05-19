from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import numpy as np
import torch.nn as nn
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Display a robot in Meshcat")
    parser.add_argument("--urdf", type=str, default="nao.urdf", help="Path to the URDF file")
    args = parser.parse_args()
    
    robot = load_robot(f"URDF/{args.urdf}")

    
    
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True) 
    viz.loadViewerModel()

    viz.display(robot.q0)
    
    input("PRESS ENTER TO END")