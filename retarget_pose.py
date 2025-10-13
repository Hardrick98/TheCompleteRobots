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

    H = human_action1.get_joint_dict()
    q1 = robotoid1.retarget(human_action1, idx)[0]
    q2 = robotoid2.retarget(human_action2, idx)[0]


    if args.visualize:
            
        

        
        human_action = human_action2
       
        human_joints_seq, orientations_seq, translation_seq, global_orient_seq, human_meshes, directions_seq = human_action.get_attributes()  
        human_origin = translation_seq[0]
        human_mesh = human_meshes[0]

        viz = MeshcatVisualizer(robotoid1.model, robotoid1.collision_model, robotoid1.visual_model)
        viz.initViewer(open=False) 
        viz.loadViewerModel()
        viz.display(q1)
        input("Press Enter to reset the visualization...")
        viz.reset()
        
        visual_model = robot.visual_model   


        vp = Plotter(title="Human and Robot", axes=1, interactive=False)

        for visual in visual_model.geometryObjects:
            
            mesh_path = os.path.join(visual.meshPath.replace(".dae",".stl"))
            if not os.path.exists(mesh_path):
                print(f"Mesh not found: {mesh_path}")
                continue

            try:
                m = Mesh(mesh_path)
            except Exception as e:
                print(f"Error during loading of {mesh_path}: {e}")
                continue

            color = visual.meshColor
            m.color(color[:3])
            placement = data.oMf[visual.parentFrame]

            
            placement_world = placement.act(visual.placement)
            R = placement_world.rotation
            p = placement_world.translation


            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = p

            m.scale(visual.meshScale)
            m.apply_transform(T)

            vp += m


        M = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
        T = np.eye(4)
        T[:3, :3] = M
        human_origin[0] *= -1

        T[:3, 3] = -human_origin + (np.array([0,0.7,1]))
        human_mesh.apply_transform(T)
        vp += human_mesh
        vp.camera.SetPosition([3, 0, 1])        
        vp.camera.SetFocalPoint([0, 0, 0])      
        vp.camera.SetViewUp([0, 0, 1])         

        vp.show(axes=1, interactive=True)

