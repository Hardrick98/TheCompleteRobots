import pyrender
import numpy as np
from robotoid import Robotoid, HumanAction
from smplx import SMPLX
from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import trimesh
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
    parser.add_argument("--human",
                        type=int, default=1,
                        help="Decide which human  to visualize, could be 1 o 2")
    parser.add_argument("--visualize",
                        action="store_true",
                        help="If to visualize video or not at the end")
    args  = parser.parse_args()
    robot_name = args.robot.lower() 
    idx = args.idx
    
    print("Sequence index:", idx)
    print("Robot to retarget:", robot_name)
    
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
    
    
    file1 = args.interaction + "/P1.npz"
    file2 = args.interaction + "/P2.npz"
    action1 = np.load(file1, allow_pickle=True)
    action2 = np.load(file2, allow_pickle=True)

    human_action1 = HumanAction(action1)
    human_action2 = HumanAction(action2)

    H = human_action1.get_joint_dict()
    q1, error1 = robotoid1.retarget(human_action1, idx)
    q2, error2 = robotoid2.retarget(human_action2, idx)

    q1 = q1[0]
    q2 = q2[0]

    

    if args.visualize:

        if args.human == 2:
            robotoid = robotoid2
            q = q2
        else:
            robotoid = robotoid1
            q = q1

        pin.forwardKinematics(robotoid.model, robotoid.data, q)
        pin.updateFramePlacements(robotoid.model, robotoid.data)
            
        pyr_scene = pyrender.Scene(ambient_light=[0.5,0.5,0.5],bg_color=[255,255,255])

        if args.human == 2:
            human_action = human_action2
        else:
            human_action = human_action1
       
        human_joints_seq, orientations_seq, translation_seq, global_orient_seq, human_meshes, directions_seq = human_action.get_attributes()  
        human_origin = translation_seq[idx]
        human_mesh = human_meshes[idx]
        human_origin[0] *= -1

        """
        viz = MeshcatVisualizer(robotoid1.model, robotoid1.collision_model, robotoid1.visual_model)
        viz.initViewer(open=False) 
        viz.loadViewerModel()
        viz.display(q1)
        input("Press Enter to reset the visualization...")
        viz.reset()
        """

        visual_model = robot.visual_model   

        for visual in visual_model.geometryObjects:
            
            mesh_path = os.path.join(visual.meshPath)
            if not os.path.exists(mesh_path):
                print(f"Mesh not found: {mesh_path}")
                continue

            try:
                m = trimesh.load_mesh(mesh_path)
            except Exception as e:
                print(f"Error during loading of {mesh_path}: {e}")
                continue

            placement = data.oMf[visual.parentFrame]

            
            placement_world = placement.act(visual.placement)
            R = placement_world.rotation
            p = placement_world.translation


            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = p

            m.apply_scale(visual.meshScale)
            m.apply_transform(T)

            trans = human_origin + np.array([0,1,0])
            m.apply_translation(trans)


            pyr_mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
            node = pyr_scene.add(pyr_mesh)



        M = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
        T = np.eye(4)
        T[:3, :3] = M
    
        #T[:3, 3] = -human_origin
        human_mesh.apply_transform(T)


        pyr_mesh = pyrender.Mesh.from_trimesh(human_mesh, smooth=True)
        node = pyr_scene.add(pyr_mesh)
        pyrender.Viewer(pyr_scene, use_raymond_lighting=True) 

