from utils import *
from test_smpl import load_simple_all
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
from vedo import Plotter, Mesh, merge
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import matplotlib.pyplot as plt
from inverse_kinematics import InverseKinematicSolver
from robotoid import Robotoid
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
    parser.add_argument("--human_pose",
                        type=str,
                        help="Path to smpl human pose")
    parser.add_argument("--idx",
                        type=int,
                        help="Path to smpl human pose")
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
    

    pose_dict, robot_joints = robot.get_joints(robot.q0)
    _, robot_limbs = robot.get_physical_joints()
    
    
    
    model = robot.model
    data = robot.data
    q0 = robot.q0  

    smpl_model = SMPLX(
        model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
        gender='male', 
        batch_size=8
    ).to("cuda:0")

    robotoid = Robotoid(robot)
    F, R = robotoid.build()
    solver = InverseKinematicSolver(model,data)
        
    arr = np.load(args.human_pose, allow_pickle=True)
    
    joint_positions, orientations, translation, global_orient, human_meshes, directions_seq = load_simple_all(smpl_model, arr)    
    
    human_joints_seq = joint_positions.detach().cpu().numpy()
    orientations_seq = orientations.detach().cpu()
    translation_seq = translation.detach().cpu().numpy()
    global_orient_seq = global_orient.detach().cpu()

    print("\nFINDING CONFIGURATIONS...")

    joint_configurations = []
    sequence_num = joint_positions.shape[0]
    #sequence_num = 200
    for i in tqdm(range(sequence_num)):
    
        human_joints = human_joints_seq[i:i+1][0]
        orientations = orientations_seq[i:i+1][0]
        translation = translation_seq[i:i+1]
        global_orient = global_orient_seq[i:i+1]
        human_mesh = human_meshes[i]
        directions = directions_seq[i]


        translation[:,[1,2]] = translation[:,[2,1]]

        orientations = orientations.view(-1,3) 
        orientations = torch.cat((global_orient.view(-1,3),orientations),axis=0)
        
        links_positions = robot.get_links_positions(q0)
    
        directions = directions.detach().cpu().numpy()

    
        human_joints[:,:] -= human_joints[:1,:]
        human_joints[:,0] *= -1
        human_joints[:,[1,2]] = human_joints[:,[2,1]]
        directions[:,0] *= -1
        directions[:,[1,2]] = directions[:,[2,1]]
        
        


        H = {
        "root_joint":0,
        "LHip":1,
        "RHip":2,
        "spine1":3,
        "LKnee":4,
        "RKnee":5,
        "spine2":6,
        "LAnkle":7,
        "RAnkle":8,
        "spine3":9,
        "left_foot":10,
        "right_foot":11,
        "Neck":12,
        "left_collar":13,
        "right_collar":14,
        "Head":15,
        "LShoulder":16,
        "RShoulder":17,
        "LElbow":18,
        "RElbow":19,
        "LWrist":20,
        "RWrist":21}
        
        head_fixed = False
        if "Head" not in R:
            head_fixed = True
            R["Head"] = R["root_joint"]
            F["Head"] = F["root_joint"]
        
        robot_limbs = [(R["LHip"],R["LKnee"]), (R["LKnee"], R["LAnkle"]), (R["RHip"],R["RKnee"]), 
                    (R["RKnee"], R["RAnkle"]),(R["LShoulder"],R["LElbow"]), (R["LElbow"], R["LWrist"]), 
                    (R["RShoulder"],R["RElbow"]), (R["RElbow"], R["RWrist"]), (R["Head"], R["RShoulder"]), 
                    (R["Head"], R["LShoulder"]), (R["Head"], R["RHip"]),(R["Head"], R["LHip"])]
        
        robot_joints = scale_human_to_robot(R,F, robot_joints, H, human_joints, head_fixed)
    
        indices = [R["Head"],R["LHip"], R["LKnee"], R["LAnkle"], R["RHip"], R["RKnee"], R["RAnkle"],
                R["LShoulder"], R["LElbow"], R["LWrist"],
                R["RShoulder"], R["RElbow"], R["RWrist"]]
    
        
        links, links2 = robot.get_frames()
        joints = robot.joints

        


        
        target_positions = {
            F["LHip"] : robot_joints[R["LHip"]],
            F["RHip"] : robot_joints[R["RHip"]], 
            F["LElbow"]: robot_joints[R["LElbow"]],
            F["RElbow"]: robot_joints[R["RElbow"]],
            F["LWrist"]: robot_joints[R["LWrist"]], 
            F["RWrist"]: robot_joints[R["RWrist"]], 
            F["RKnee"]: robot_joints[R["RKnee"]], 
            F["LKnee"]: robot_joints[R["LKnee"]], 
            F["LAnkle"]: robot_joints[R["LAnkle"]], 
            F["RAnkle"]: robot_joints[R["RAnkle"]], 
            F["RShoulder"] : robot_joints[R["RShoulder"]],
            F["LShoulder"] : robot_joints[R["LShoulder"]],
            F["Head"]: robot_joints[R["Head"]],
        }
        
        joint_names = [k for k in target_positions.keys()]
        joint_ids = [joints[name]for name in joint_names]



        if head_fixed:
            target_positions.pop(F["Head"])

        
        target_orientations_global  = {
            F["RWrist"]: [directions[H["RWrist"]], [0,0,-1]], 
            F["LWrist"]: [directions[H["LWrist"]], [0,0,-1]],
            F["Head"]: [directions[H["Head"]], [1,0,0]]
        }
        
        frame_names = [k for k,v in target_orientations_global.items()]
        frame_ids = [model.getFrameId(f) for f in frame_names]

        
        solver.update(model,data,target_positions,target_orientations_global,joint_names, joint_ids, frame_names, frame_ids)
        
        if i==0:
            q1 = solver.inverse_kinematics(q0)
        else:
            q1 = solver.inverse_kinematics(q1)
        
        joint_configurations.append(q1)
        
        pin.forwardKinematics(model, data, q1)
        pin.updateFramePlacements(model, data)
            
    
print("\nCONSTRUCTING VISUALIZATION...")
"""
if args.human_pose[-5]=="1":
    args.human_pose[-5] == "2"
    interactor = args.human_pose
else:
    args.human_pose[-5] == "1"
    interactor = args.human_pose
"""
 
#arr = np.load(interactor, allow_pickle=True)

#joint_positions, orientations, translation, global_orient, human_meshes, directions_seq = load_simple_all(smpl_model, arr)    

#human_joints_seq = joint_positions.detach().cpu().numpy()
#orientations_seq = orientations.detach().cpu()
#translation_seq = translation.detach().cpu().numpy()
#global_orient_seq = global_orient.detach().cpu()


human_meshes_t = []
robot_meshes = []

for t in tqdm(range(len(joint_configurations))):

    q1 = joint_configurations[t]
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

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p

        m.scale(visual.meshScale[0])
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
    human_origin = translation_seq[t]
    human_origin[0] *= -1

    T[:3, 3] = -human_origin + (np.array([0,0.7,1]))
    hm = human_meshes[t].apply_transform(T)

    robot_meshes.append(robot_mesh)
    human_meshes_t.append(hm)

from test_smpl import animate_all_poses

animate_all_poses(human_meshes_t, robot_meshes)



    