from utils import *
from human_interaction import load_simple_all
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
from vedo import Plotter, Mesh, Video, Axes
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import matplotlib.pyplot as plt
from inverse_kinematics import InverseKinematicSolver
from robotoid import Robotoid
from smplx import SMPLX
import time
import joblib


def preload_robot_meshes(robot):
    frames = robot.body
    cache = {}
    for visual in robot.visual_model.geometryObjects:
        mesh_path = os.path.join(visual.meshPath.replace(".dae", ".stl"))
        if not os.path.exists(mesh_path):
            continue
        mesh = Mesh(mesh_path)
        mesh.color(visual.meshColor[:3])
        mesh.scale(visual.meshScale)
        cache[visual.name] = (mesh, visual.placement, frames[visual.name[:-2]])
  
    return cache


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

robotoid = Robotoid(robot1, wheeled)
F, R = robotoid.build()
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

print("\nCONSTRUCTING VISUALIZATION...")

arr1 = np.load(f"{args.interaction}/P1.npz", allow_pickle=True)
arr2 = np.load(f"{args.interaction}/P2.npz", allow_pickle=True)

joint_positions1, _, translation1, _, _, _ = load_simple_all(smpl_model, arr1)
joint_positions2, _, translation2, _, _, _ = load_simple_all(smpl_model, arr2)
smpl_model = 0  # libera GPU

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

joint_configurations1 = np.load(f"{args.interaction}/{robot_name1}1.npy")
joint_configurations2 = np.load(f"{args.interaction}/{robot_name2}2.npy")

robot1_cache = preload_robot_meshes(robot1)
robot2_cache = preload_robot_meshes(robot2)

vp = Plotter(interactive=True, axes=0, title="SMPLX Retargeting", bg='white')

if args.video:
    video = Video("video.mp4", duration=len(joint_configurations1)/60, fps=60)
else:
    video = None



prev_meshes1 = []
prev_meshes2 = []

for t in tqdm(range(len(joint_configurations1))):

    q1 = joint_configurations1[t]
    pin.forwardKinematics(robot1.model, robot1.data, q1)
    pin.updateFramePlacements(robot1.model, robot1.data)

    meshes1 = []
    robot_pos1 = []

    for name, (base_mesh, placement, frame) in robot1_cache.items():
        m = base_mesh.clone()
        placement_world = robot1.data.oMf[frame].act(placement)
        R = placement_world.rotation
        p = placement_world.translation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        m.apply_transform(T)
        m.color("blue")
        meshes1.append(m)
        robot_pos1.append(p)

    q2 = joint_configurations2[t]
    pin.forwardKinematics(robot2.model, robot2.data, q2)
    pin.updateFramePlacements(robot2.model, robot2.data)

    meshes2 = []
    robot_pos2 = []
    for name, (base_mesh, placement, parentFrame) in robot2_cache.items():
        m = base_mesh.clone()
        placement_world = robot2.data.oMf[parentFrame].act(placement)
        R = placement_world.rotation
        p = placement_world.translation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        m.apply_transform(T)
        m.color("red")
        meshes2.append(m)
        robot_pos2.append(p)

    t1 = trans1[t].copy()
    t2 = trans2[t].copy()

    if t == 0:
        robot_pos1 = np.vstack(robot_pos1)
        human_bounds = np.ptp(human1_js[t], axis=0)
        robot_bounds = np.ptp(robot_pos1, axis=0)
        s1 = robot_bounds / human_bounds  

        robot_pos2 = np.vstack(robot_pos2)
        human_bounds = np.ptp(human2_js[t], axis=0)
        robot_bounds = np.ptp(robot_pos2, axis=0)
        s2 = robot_bounds / human_bounds   
  
    T1 = np.eye(4)
    min_z = np.min(human1_js[t,:,2])
    t1_s = t1.copy()
    t1_s[2] -= min_z           #make sure it is on the ground
    t1_s = t1_s * s1
    T1[:3, 3] = t1_s
    
    for m in meshes1:
        m.apply_transform(T1)

    T2 = np.eye(4)
    min_z = np.min(human2_js[t,:,2])
    t2_s = t2.copy()
    t2_s[2] -= min_z  

    t2_s = t2_s * s2

    T2[:3, 3] = t2_s
    
    for m in meshes2:
        m.apply_transform(T2)
        
    axes_opts = dict(
        xtitle="X",
        ytitle="Y",
        ztitle="Z",
        xyplane_color="yellow",
        xyalpha=0.5,
        zrange=[0,2],
        xrange=[-1,1],
        yrange=[-1,1]
    )

    ax = Axes(vp, **axes_opts)
    if t == 0:
        vp.show(*meshes1, *meshes2, axes=ax, resetcam=True)

    else:
        vp.remove(*prev_meshes1, *prev_meshes2)
        vp.add(*meshes1, *meshes2)
        vp.render()

    if video:
        video.add_frame()

    time.sleep(0.01)

    prev_meshes1 = meshes1
    prev_meshes2 = meshes2

if video:
    video.close()

vp.interactive()
vp.close()
