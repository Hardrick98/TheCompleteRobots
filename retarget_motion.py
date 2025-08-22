from utils import *
from test_smpl import load_simple_all
import pinocchio as pin
import argparse
from vedo import Mesh, merge
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from robotoid import Robotoid, HumanAction
from smplx import SMPLX


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
                        type=int,
                        help="Path to smpl human pose")
    parser.add_argument("--visualize",
                        action="store_true",
                        help="If to visualize video or not at the end")
    args  = parser.parse_args()
    robot_name = args.robot.lower() 
    idx = args.idx

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
    
    print(wheeled)

    robotoid1 = Robotoid(robot, wheeled)
    robotoid2 = Robotoid(robot, wheeled)
    
    
    file1 = args.interaction + "/P2.npz"
    file2 = args.interaction + "/P1.npz"
    action1 = np.load(file1, allow_pickle=True)
    action2 = np.load(file2, allow_pickle=True)

    human_action1 = HumanAction(action1)
    human_action2 = HumanAction(action2)

    H = human_action1.get_joint_dict()
    joint_config1 = robotoid1.retarget(human_action1)
    joint_config2 = robotoid2.retarget(human_action2)


    path = file1.removesuffix(".npz")
    path = path[:-2] + robot_name + path[-1] + ".npy"
    np.save(path, joint_config1)
    path = file2.removesuffix(".npz")
    path = path[:-2] + robot_name + path[-1] + ".npy"
    np.save(path, joint_config2)

    

    if args.visualize:
            
        print("\nConstructing Visualization...")
        human_action = human_action2
       
        human_joints_seq, orientations_seq, translation_seq, global_orient_seq, human_meshes, directions_seq = human_action.get_attributes()  


        human_meshes_t = []
        robot_meshes = []

        

        for t in tqdm(range(len(joint_config1))):

            lowest_pos = 1

            q1 = joint_config1[t]
            pin.forwardKinematics(model, data, q1)
            pin.updateFramePlacements(model, data)

            visual_model = robot.visual_model   


            all_meshes = []
            for visual in visual_model.geometryObjects:
                mesh_path = os.path.join(visual.meshPath.replace(".dae", ".stl"))
                if not os.path.exists(mesh_path):
                    print(f"Mesh non trovata: {mesh_path}")
                    continue

                try:
                    m = Mesh(mesh_path)
                except Exception as e:
                    print(f"Errore nel caricare {mesh_path}: {e}")
                    continue

                color = visual.meshColor
                m.color(color[:3])  # RGB

                placement = data.oMf[visual.parentFrame]
                placement_world = placement.act(visual.placement)
                R = placement_world.rotation
                p = placement_world.translation

                if p[2] < lowest_pos:
                    lowest_pos = p[2]

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = p

                m.scale(visual.meshScale)
                m.apply_transform(T)

                all_meshes.append(m)


        

            robot_mesh = merge(all_meshes)



            M = np.array([
                [-1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ])
            T = np.eye(4)
            T[:3, :3] = M

            
            human_origin = translation_seq[t:t+1]
            human_origin[:,[1,2]]=human_origin[:,[2,1]]
            human_origin[:,0] *= -1

            interaction_pos = translation_seq[t:t+1]
            interaction_pos[:,[1,2]]=interaction_pos[:,[2,1]]
            interaction_pos[:,0] *= -1

            human_joints = human_joints_seq[t]
            human_joints[:,:] -= human_joints[:1,:]
            human_joints[:,0] *= -1
            human_joints[:,[1,2]] = human_joints[:,[2,1]]
            ground_pos = human_joints[H["RAnkle"]]
            diff = lowest_pos - ground_pos[2]

            distance = human_origin - interaction_pos
            T[:3, 3] = -human_origin + distance + np.array([0,0,diff])
            hm = human_meshes[t].apply_transform(T)

            robot_meshes.append(robot_mesh)
            human_meshes_t.append(hm)

        from test_smpl import animate_all_poses

        animate_all_poses(human_meshes_t, robot_meshes)


