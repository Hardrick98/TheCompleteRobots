from utils import *
from human_interaction import load_simple_all
import joblib
import pinocchio as pin
import argparse
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import trimesh
from inverse_kinematics import InverseKinematicSolver
from robotoid import Robotoid
from smplx import SMPLX
from visual_utils import get_rotation_matrix
from trimesh.collision import CollisionManager
from visual_utils import preload_robot_meshes, calculate_scale_factors



robot_cameras_indexes = {"nao": [30,32], "g1":[34, 34], "atlas":[24,24], "pepper":[20,22], "icub":[90,90]}


robot_list = [r.removesuffix(".urdf") for r in os.listdir("URDF") if r.endswith(".urdf")]

parser = argparse.ArgumentParser(description="Retarget human to robot")
parser.add_argument("--robot1", type=str, default="nao", help="The robot1 to visualize.")
parser.add_argument("--robot2", type=str, default="nao", help="The robot2 to visualize.")
parser.add_argument("--interaction", type=str, help="Path to smpl human pose")
parser.add_argument("--video", action="store_true", help="If to record video or not at the end")

args = parser.parse_args()
robot_name1 = args.robot1.lower()
robot_name2 = args.robot2.lower()

wheeled = args.robot1.lower() == "pepper"

try:
    robot1 = HumanoidRobot(f"URDF/{args.robot1}.urdf")
except Exception as e:
    print(f"Error loading robot {robot_name1}: {e}")
    print("Available robots:")
    for r in robot_list:
        print(f"- {r}")
    exit(1)

try:
    robot2 = HumanoidRobot(f"URDF/{args.robot2}.urdf")
except Exception as e:
    print(f"Error loading robot {robot_name1}: {e}")
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
    model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',
    gender='male',
    batch_size=8
).to("cuda:0")

frames_list = robot1.body
print(frames_list)

robotoid = Robotoid(robot1, wheeled)
#F, R = robotoid.build()
solver = InverseKinematicSolver(model, data)

H = {
    "root_joint": 0,
    "LHip": 1,
    "RHip": 2,
    "spine1": 3,
    "LKnee": 4,
    "RKnee": 5,
    "spine2": 6,
    "LAnkle": 7,
    "RAnkle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "Neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "Head": 15,
    "LShoulder": 16,
    "RShoulder": 17,
    "LElbow": 18,
    "RElbow": 19,
    "LWrist": 20,
    "RWrist": 21
}

print("\nExtracting Data...")

robot_folder = f"{args.interaction}/{args.robot1}"

arr1 = np.load(f"{args.interaction}/P1.npz", allow_pickle=True)
arr2 = np.load(f"{args.interaction}/P2.npz", allow_pickle=True)

joint_positions1, _, translation1, _, _, _ = load_simple_all(smpl_model, arr1)
joint_positions2, _, translation2, _, _, _ = load_simple_all(smpl_model, arr2)
smpl_model = 0  # Free GPU

human1_js = joint_positions1.detach().cpu().numpy()
trans1 = translation1.detach().cpu().numpy()
human2_js = joint_positions2.detach().cpu().numpy()
trans2 = translation2.detach().cpu().numpy()

trans1[:, 0] *= -1
trans1[:, [1, 2]] = trans1[:, [2, 1]]
trans2[:, 0] *= -1
trans2[:, [1, 2]] = trans2[:, [2, 1]]
human1_js[:, :, 0] *= -1
human1_js[:, :, [1, 2]] = human1_js[:, :, [2, 1]]
human2_js[:, :, 0] *= -1
human2_js[:, :, [1, 2]] = human2_js[:, :, [2, 1]]

joint_configurations1 = np.load(f"{robot_folder}/{robot_name1}1_joints_config.npy")
joint_configurations2 = np.load(f"{robot_folder}/{robot_name2}2_joints_config.npy")

robot1_cache = preload_robot_meshes(robot1)
robot2_cache = preload_robot_meshes(robot2)

camera_base = np.eye(4)

n_frames = human1_js.shape[0]

manager1 = CollisionManager()
manager2 = CollisionManager()
names = []

for name, (base_mesh, placement, frame) in robot1_cache.items():
    manager1.add_object(name=f"{args.robot1}1_{name}", mesh=base_mesh)
    names.append(name)

for name, (base_mesh, placement, frame) in robot2_cache.items():
    manager2.add_object(name=f"{args.robot2}2_{name}", mesh=base_mesh)

cameras = {"ego1R":[],"ego2R":[], "ego1L":[], "ego2L":[]}

robot1_poses_all = []
robot2_poses_all = []

collision_list = []
temporal_max = np.zeros((n_frames,3))
temporal_min = np.zeros((n_frames,3))


for t in tqdm(range(len(joint_configurations1))):

    q1 = joint_configurations1[t]
    pin.forwardKinematics(robot1.model, robot1.data, q1)
    pin.updateFramePlacements(robot1.model, robot1.data)

    meshes1 = []
    robot_pos1 = []
    poses1 = []
    poses2 = []

    # Get positions and rotations

    for name, (base_mesh, placement, frame) in robot1_cache.items():
        placement_world = robot1.data.oMf[frame]
        R = placement_world.rotation
        p = placement_world.translation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        if args.robot1 == "icub":
            T = T @ placement
        robot_pos1.append(T[:3, 3])
        meshes1.append(base_mesh)
        poses1.append(T[None,:,:])
        
    
    poses1 = np.vstack(poses1)
    

    q2 = joint_configurations2[t]
    pin.forwardKinematics(robot2.model, robot2.data, q2)
    pin.updateFramePlacements(robot2.model, robot2.data)

    meshes2 = []
    robot_pos2 = []
    # Get positions and rotations
    for name, (base_mesh, placement, frame) in robot2_cache.items():
        placement_world = robot2.data.oMf[frame]
        R = placement_world.rotation
        p = placement_world.translation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        if args.robot2 == "icub":
            T = T @ placement
            if name[:-2] == "head":
                camera_base = placement
        robot_pos2.append(T[:3, 3])
        meshes2.append(base_mesh)
        poses2.append(T[None,:,:])
    


    poses2 = np.vstack(poses2)
    #Load translation in world
    t1 = trans1[t].copy()
    t2 = trans2[t].copy()


    if t == 0:      #calculate scaling factor
        s1, s2 = calculate_scale_factors(human1_js[t],human2_js[t], robot_pos1, robot_pos2)
  
    T1 = np.eye(4)
    min_z = np.min(human1_js[t,:,2]) 
    t1_s = t1.copy()
    t1_s[2] -= min_z           #make sure it is on the ground
    t1_s = t1_s * s1           #apply scaling
    T1[:3, 3] = t1_s
    
    bounds1 = []


    for i, mesh in enumerate(meshes1):
        T0 = poses1[i]
        m = mesh.copy()
        m.apply_transform(T1@T0)
        bounds1.append(m.bounds)
        manager1.set_transform(f"{args.robot1}1_{names[i]}", T1@T0)
        poses1[i] = T1@poses1[i]

    T2 = np.eye(4)
    min_z = np.min(human2_js[t,:,2])
    t2_s = t2.copy()
    t2_s[2] -= min_z  

    t2_s = t2_s * s2

    T2[:3, 3] = t2_s
    
    bounds2 = []

    for i, mesh in enumerate(meshes2):
        T0 = poses2[i]
        m = mesh.copy()
        m.apply_transform(T2@T0)
        bounds2.append(m.bounds)
        manager2.set_transform(f"{args.robot2}2_{names[i]}", T2@T0)
        poses2[i] = T2@poses2[i]

    bounds1 = np.array(bounds1)
    bounds2 = np.array(bounds2)
    minsR1 = bounds1[:,0].min(axis=0)
    maxsR1 = bounds1[:,1].max(axis=0)
    minsR2 = bounds2[:,0].min(axis=0)
    maxsR2 = bounds2[:,1].max(axis=0)



    maxs = np.max(np.concatenate((maxsR1[None,:], maxsR2[None,:]), axis=0), axis=0)
    mins = np.min(np.concatenate((minsR1[None,:], minsR2[None,:]), axis=0), axis=0)

    temporal_max[t] = maxs
    temporal_min[t] = mins


    collisions = manager1.in_collision_other(manager2, return_names=True)

    collision_list.append(collisions[1])

    robot1_poses_all.append(poses1[None,:,:,:])
    robot2_poses_all.append(poses2[None,:,:,:])

    robot1_camera_left = robot_cameras_indexes[args.robot1][0]
    robot1_camera_right = robot_cameras_indexes[args.robot1][1]
    robot2_camera_left = robot_cameras_indexes[args.robot2][0]
    robot2_camera_right = robot_cameras_indexes[args.robot2][1]

    if args.robot1 == "icub":
        R = np.eye(4)
        Rz = get_rotation_matrix(axis="z")
        R[:3,:3] = np.linalg.inv(Rz)
        R[:3,3] = np.array([0,-0.2,0.05]) #Valore che sarebbe da vedere bene
        camera_base = camera_base @ R
    if t==0:
        if args.robot1 == "g1":
            R = np.eye(4)
            Rx = get_rotation_matrix(theta=np.pi/4, axis="x")
            R[:3,:3] = np.linalg.inv(Rx)
            camera_base = camera_base @ R


    camera1L = get_camera_placement(robot1, robot1_camera_left, T1, robot_name= args.robot1, stereo="L", camera_base=camera_base)
    camera1R = get_camera_placement(robot1, robot1_camera_right, T1, robot_name= args.robot1, stereo="R",camera_base=camera_base)
    camera2L = get_camera_placement(robot2, robot2_camera_left, T2, robot_name= args.robot2, stereo="L", camera_base=camera_base)
    camera2R = get_camera_placement(robot2, robot2_camera_right, T2, robot_name= args.robot2, stereo="R", camera_base=camera_base)

    cameras["ego1L"].append(camera1L)
    cameras["ego1R"].append(camera1R)
    cameras["ego2L"].append(camera2L)
    cameras["ego2R"].append(camera2R)
    


robot1_poses = np.vstack(robot1_poses_all)
robot2_poses = np.vstack(robot2_poses_all)


maxs = np.max(temporal_max, axis=0)
mins = np.min(temporal_min, axis=0)

try:
    os.mkdir(f"{robot_folder}/data")
except:
    pass

final_dict = {"robot_1_poses":robot1_poses, "robot_2_poses":robot2_poses,"scales":(s1,s2), "ws_max":maxs, "ws_min":mins}

joblib.dump(final_dict, f"{robot_folder}/data/{args.robot1}_prep.pkl")
joblib.dump(collision_list, os.path.join(f"{robot_folder}/data",f"{args.robot1}_{args.robot2}_collisions.pkl"))
joblib.dump(cameras, os.path.join(f"{robot_folder}/data",f"{args.robot1}_cameras.pkl"))

print("Data successfully saved!")