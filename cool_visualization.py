from utils import *
from human_interaction import load_simple_all
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
from vedo import Video
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from inverse_kinematics import InverseKinematicSolver
from robotoid import Robotoid
from smplx import SMPLX
import pyrender
import trimesh
import imageio
import pywavefront


theta = np.pi / 2  # 90 gradi in radianti

Rx = np.array([
    [ np.cos(theta), 0, np.sin(theta)],
    [ 0,             1, 0            ],
    [-np.sin(theta), 0, np.cos(theta)]
])
Ry = np.array([
[1, 0,           0          ],
[0, np.cos(theta), -np.sin(theta)],
[0, np.sin(theta),  np.cos(theta)]
])

Rz = np.array([
[ np.cos(theta), -np.sin(theta), 0],
[ np.sin(theta),  np.cos(theta), 0],
[ 0,              0,             1]
])
    

# ------------------- funzione di preload -------------------
def preload_robot_meshes(robot):
    cache = {}
    for visual in robot.visual_model.geometryObjects:
        mesh_path = visual.meshPath
        if not os.path.exists(mesh_path):
            continue
        try:
            mesh = trimesh.load_mesh(mesh_path)
            mesh.apply_scale(visual.meshScale)
            cache[visual.name] = (mesh, visual.placement, visual.parentFrame)
        except Exception as e:
            print(f"Errore caricando mesh {mesh_path}: {e}")
            continue
    return cache

# ------------------- parser e robot -------------------
robot_list = [r.removesuffix(".urdf") for r in os.listdir("URDF") if r.endswith(".urdf")]

parser = argparse.ArgumentParser(description="Retarget human to robot")
parser.add_argument("--robot1", type=str, default="nao")
parser.add_argument("--robot2", type=str, default="nao")
parser.add_argument("--interaction", type=str)
parser.add_argument("--video", action="store_true")
parser.add_argument("--scene", type=str, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

robot_name1 = args.robot1.lower()
robot_name2 = args.robot2.lower()
wheeled = robot_name1 == "pepper"
camera_mode = "ego"
try:
    robot1 = HumanoidRobot(f"URDF/{args.robot1}.urdf")
except Exception as e:
    print(f"Error loading robot {robot_name1}: {e}")
    exit(1)

try:
    robot2 = HumanoidRobot(f"URDF/{args.robot2}.urdf")
except Exception as e:
    print(f"Error loading robot {robot_name2}: {e}")
    exit(1)


model = robot1.model
data = robot1.data
q0 = robot1.q0

smpl_model = SMPLX(
    model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',
    gender='male',
    batch_size=8
).to("cuda:0")

robotoid = Robotoid(robot1, wheeled)
F, R = robotoid.build()
solver = InverseKinematicSolver(model, data)

# ------------------- carica dati umani -------------------
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

# ------------------- preload robot meshes -------------------
robot1_cache = preload_robot_meshes(robot1)
robot2_cache = preload_robot_meshes(robot2)

# ------------------- setup pyrender -------------------
pyr_scene = pyrender.Scene(ambient_light=[0.5,0.5,0.5], bg_color=[0,255,0])
mesh_nodes1 = []
mesh_nodes2 = []

if args.scene != None:
    scene_mesh = trimesh.load_scene("LR.glb")
    T_center = np.eye(4)
    T_center[:3,3] = [12,-5.5,-5]
    scene_mesh.apply_transform(T_center)
    for node_name in scene_mesh.graph.nodes_geometry:
        # la tupla è (T, geom_name)
        T, geom_name = scene_mesh.graph[node_name]
        geom = scene_mesh.geometry[geom_name]
        pyr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
        T0 = np.eye(4)
        T0[:3,:3] = Ry

        pyr_scene.add(pyr_mesh, pose=T0@T)

    scene_mesh.matrix = T0

# aggiungi robot1
for name, (mesh, placement, parentFrame) in robot1_cache.items():
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    node = pyr_scene.add(pyr_mesh)
    mesh_nodes1.append((node, placement, parentFrame))

# aggiungi robot2
for name, (mesh, placement, parentFrame) in robot2_cache.items():
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    node = pyr_scene.add(pyr_mesh)
    mesh_nodes2.append((node, placement, parentFrame))

# luce
key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
pyr_scene.add(key_light, pose=np.eye(4))  

fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
pose_fill = np.eye(4)
pose_fill[:3,3] = [-2,2,1]  # spostala un po’
pyr_scene.add(fill_light, pose=pose_fill)


# luce di retro (back light)
back_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
pose_back = np.eye(4)
pose_back[:3,3] = [0,-3,1]
pyr_scene.add(back_light, pose=pose_back)

Rcam = np.eye(4)
Rcam[:3,3] = [-1,0,-1]
# camera
cam = pyrender.PerspectiveCamera(yfov=np.pi/2, aspectRatio=1280/720)
cam_node = pyr_scene.add(cam, pose=Rcam)

# renderer offscreen
if not args.debug:
    r = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)
frames = []

if not args.debug:
    n_frames = len(joint_configurations1)
else:
    n_frames = 1
# ------------------- loop frame -------------------
for t in tqdm(range(n_frames)):

    # --- robot1 ---
    q1 = joint_configurations1[t]
    pin.forwardKinematics(robot1.model, robot1.data, q1)
    pin.updateFramePlacements(robot1.model, robot1.data)



    robot_pos1 = []
    for node, placement, parentFrame in mesh_nodes1:
        placement_world = robot1.data.oMf[parentFrame].act(placement)
        T = np.eye(4)
        T[:3,:3] = placement_world.rotation
        T[:3,3] = placement_world.translation
        node.matrix = T 
        robot_pos1.append(placement_world.translation)

    # --- robot2 ---
    q2 = joint_configurations2[t]
    pin.forwardKinematics(robot2.model, robot2.data, q2)
    pin.updateFramePlacements(robot2.model, robot2.data)

    robot_pos2 = []
    for node, placement, parentFrame in mesh_nodes2:
        placement_world = robot2.data.oMf[parentFrame].act(placement)
        T = np.eye(4)
        T[:3,:3] = placement_world.rotation
        T[:3,3] = placement_world.translation
        node.matrix = T
        robot_pos2.append(placement_world.translation)

    # --- applica traslazioni/scalature dei robot rispetto umano ---
    t1_s = trans1[t].copy()
    t1_s[2] -= np.min(human1_js[t,:,2])
    if t == 0:
        robot_pos1_bounds = np.ptp(np.vstack(robot_pos1), axis=0)
        human_bounds1 = np.ptp(human1_js[t], axis=0)
        s1 = robot_pos1_bounds / human_bounds1
    t1_s *= s1
    T1 = np.eye(4)
    T1[:3,3] = t1_s
    for node, _, _ in mesh_nodes1:
        node.matrix = T1 @ node.matrix  # conserva le rotazioni originali
    
    

    t2_s = trans2[t].copy()
    t2_s[2] -= np.min(human2_js[t,:,2])
    if t == 0:
        robot_pos2_bounds = np.ptp(np.vstack(robot_pos2), axis=0)
        human_bounds2 = np.ptp(human2_js[t], axis=0)
        s2 = robot_pos2_bounds / human_bounds2
    t2_s *= s2
    T2 = np.eye(4)
    T2[:3,3] = t2_s
    #cam_node.matrix = T2 @cam_node.matrix
    if camera_mode == "exo":
        if t == 0:
            
            T0 = np.eye(4)
            T0[:3,:3] = Rz@Rz@Ry
            T0[:3,3] = (t1_s + t2_s)/2 + np.array([0,0.75,0])
            cam_node.matrix = T0
    else:
        camera_frame = 20
        camera_dir = robot2.data.oMf[camera_frame]
        T0 = np.eye(4)
        T0[:3,:3] = Rz@Ry
        T0[:3,3] = camera_dir.translation
        
        cam_node.matrix = T2 @ T0 


    for node, _, _ in mesh_nodes2:
        node.matrix = T2 @ node.matrix


    # --- render frame ---
    if not args.debug:
        color, _ = r.render(pyr_scene)
        frames.append(color)
if args.debug:
    pyrender.Viewer(pyr_scene, use_raymond_lighting=True) 
# ------------------- salva video -------------------

if not args.debug:
    r.delete()
if args.video:
    imageio.mimsave('robot_animation.mp4', frames, fps=30)
