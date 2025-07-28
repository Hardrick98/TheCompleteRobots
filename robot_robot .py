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


    
robot_list = [r.removesuffix(".urdf") for r in os.listdir("URDF") if r.endswith(".urdf") or r.endswith(".urdf")]

parser = argparse.ArgumentParser(description="Retarget human to robot")
parser.add_argument(
    "--robot",
    type=str,
    default="nao",
    help="The robot to visualize.",
)

parser.add_argument("--human_pose",
                    type=str,
                    help="Path to smpl human pose")
parser.add_argument("--video",
                        action="store_true",
                        help="If to record video or not at the end")
args  = parser.parse_args()
robot_name = args.robot.lower() 


try:
    robot1 = HumanoidRobot(f"URDF/{args.robot}.urdf")
    robot2 = HumanoidRobot(f"URDF/{args.robot}.urdf")
except Exception as e:
    print(f"Error loading robot {robot_name}: {e}")
    print("Available robots:")
    for r in robot_list:
        print(f"- {r}")
    exit(1)


pose_dict, robot_joints = robot1.get_joints(robot1.q0)
_, robot_limbs = robot1.get_physical_joints()



model = robot1.model
data = robot1.data
q0 = robot1.q0  

smpl_model = SMPLX(
    model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
    gender='male', 
    batch_size=8
).to("cuda:0")

robotoid = Robotoid(robot1)
F, R = robotoid.build()
solver = InverseKinematicSolver(model,data)


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




print("\nCONSTRUCTING VISUALIZATION...")



arr1 = np.load(f"{args.human_pose}/P1.npz", allow_pickle=True)
arr2 = np.load(f"{args.human_pose}/P2.npz", allow_pickle=True)

joint_positions1, _, translation1, _ , _ , _ = load_simple_all(smpl_model, arr1)
joint_positions2, _ , translation2, _ , _ , _ = load_simple_all(smpl_model, arr2)

smpl_model = 0

human1_js = joint_positions1.detach().cpu().numpy()
trans1 = translation1.detach().cpu().numpy()
human2_js = joint_positions2.detach().cpu().numpy()
trans2 = translation1.detach().cpu().numpy()

trans1[:,0] *= -1
trans1[:,[1,2]] = trans1[:,[2,1]]
human1_js[:,:,0] *= -1
human1_js[:,:,[1,2]] = human1_js[:,:,[2,1]]
human2_js[:,:,0] *= -1
human2_js[:,:,[1,2]] = human2_js[:,:,[2,1]]

joint_configurations1 = np.load(f"{args.human_pose}/R1.npy")
joint_configurations2 = np.load(f"{args.human_pose}/R2.npy")


robot2_meshes = []
robot1_meshes = []

 

for t in tqdm(range(len(joint_configurations1))):

    lowest_pos = 1

    q1 = joint_configurations1[t]
    pin.forwardKinematics(robot1.model, robot1.data, q1)
    pin.updateFramePlacements(robot1.model, robot1.data)

    visual_model = robot1.visual_model   

    robot_pos1 = []
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

        placement = robot1.data.oMf[visual.parentFrame]
        placement_world = placement.act(visual.placement)
        R = placement_world.rotation
        p = placement_world.translation

        robot_pos1.append(p)

        if p[2] < lowest_pos:
            lowest_pos = p[2]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p

        m.scale(visual.meshScale[0])
        m.apply_transform(T)

        all_meshes.append(m)


   

    robot_mesh1 = merge(all_meshes)


    q1 = joint_configurations2[t]
    pin.forwardKinematics(robot2.model, robot2.data, q1)
    pin.updateFramePlacements(robot2.model, robot2.data)

    visual_model = robot2.visual_model   


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

        placement = robot2.data.oMf[visual.parentFrame]
        placement_world = placement.act(visual.placement)
        R = placement_world.rotation
        p = placement_world.translation

        if p[2] < lowest_pos:
            lowest_pos = p[2]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p

        m.scale(visual.meshScale[0])
        m.apply_transform(T)

        all_meshes.append(m)


   
    robot_pos1 = np.vstack(robot_pos1)
    robot_mesh2 = merge(all_meshes)


    T = np.eye(4)

    distance_between_humans = human2_js[t][H["root_joint"]] - human1_js[t][H["root_joint"]]

    if t < 1:
    
        
        
        human_xmax = np.max(human1_js[t,:,0])
        human_xmin = np.min(human1_js[t,:,0])
        human_ymax = np.max(human1_js[t,:,1])
        human_ymin = np.min(human1_js[t,:,1])
        human_zmax = np.max(human1_js[t,:,2])
        human_zmin = np.min(human1_js[t,:,2])

        

        xhuman = human_xmax - human_xmin
        yhuman = human_ymax - human_ymin
        zhuman = human_zmax-human_zmin

        
        robot_xmax = np.max(robot_pos1[:,0])
        robot_xmin = np.min(robot_pos1[:,0])
        robot_ymax = np.max(robot_pos1[:,1])
        robot_ymin = np.min(robot_pos1[:,1])
        robot_zmax = np.max(robot_pos1[:,2])
        robot_zmin = np.min(robot_pos1[:,2])

        xrobot = robot_xmax - robot_xmin
        yrobot = robot_ymax - robot_ymin
        zrobot = robot_zmax - robot_zmin

    
        s = [xrobot/xhuman, yrobot/yhuman , 0]

    T[:3, 3] = trans1[t]

    robot_mesh1.apply_transform(T)

    distance = trans1[t] + distance_between_humans*s

    T[:3, 3] = distance
    
    robot_mesh2.apply_transform(T)

    robot1_meshes.append(robot_mesh1)
    robot2_meshes.append(robot_mesh2)


from test_smpl import animate_all_poses

animate_all_poses(robot1_meshes, robot2_meshes, video=args.video)



    