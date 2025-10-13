from utils import *
import argparse
import joblib
import os
from tqdm import tqdm
import numpy as np
from robotoid import Robotoid
import pyrender
import trimesh
import imageio
from visual_utils import *

theta = np.pi / 2 

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
    

# ------------------- parser e robot -------------------
robot_list = [r.removesuffix(".urdf") for r in os.listdir("URDF") if r.endswith(".urdf")]

parser = argparse.ArgumentParser(description="Retarget human to robot")
parser.add_argument("--robot1", type=str, default="nao")
parser.add_argument("--robot2", type=str, default="nao")
parser.add_argument("--camera_mode", type=str, default="exo")
parser.add_argument("--interaction", type=str)
parser.add_argument("--video", action="store_true")
parser.add_argument("--scene", type=str, default=None)
parser.add_argument("--green_screen",action="store_true")
parser.add_argument("--bb_mode",action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

robot_name1 = args.robot1.lower()
robot_name2 = args.robot2.lower()
wheeled = robot_name1 == "pepper"
camera_mode = args.camera_mode
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


robotoid = Robotoid(robot1, wheeled)
F, R = robotoid.build()

human1_js = np.load(os.path.join(args.interaction,"data","human1_poses.npy"))
trans1 = np.load(os.path.join(args.interaction,"data","human1_trans.npy"))
human2_js = np.load(os.path.join(args.interaction,"data","human2_poses.npy"))
trans2 = np.load(os.path.join(args.interaction,"data","human2_trans.npy"))


robot1_poses= np.load(f"{args.interaction}/data/{robot_name1}1_poses.npy")
robot2_poses = np.load(f"{args.interaction}/data/{robot_name2}2_poses.npy")

# ------------------- preload robot meshes -------------------
robot1_cache = preload_robot_meshes(robot1)
robot2_cache = preload_robot_meshes(robot2)

cameras = joblib.load(os.path.join(f"{args.interaction}/data",f"{robot_name1}_cameras.pkl"))
# ------------------- setup pyrender -------------------


if args.green_screen == True:
    pyr_scene = pyrender.Scene(ambient_light=[0.5,0.5,0.5],bg_color=[0,255,0])
else:
    pyr_scene = pyrender.Scene(ambient_light=[0.5,0.5,0.5])


mesh_nodes1 = []
mesh_nodes2 = []


# loading robot1 meshes
for name, (mesh, placement, parentFrame) in robot1_cache.items():
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    node = pyr_scene.add(pyr_mesh)
    mesh_nodes1.append((node, placement, parentFrame))


# loading robot2 meshes
for name, (mesh, placement, parentFrame) in robot2_cache.items():
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    node = pyr_scene.add(pyr_mesh)
    mesh_nodes2.append((node, placement, parentFrame))



#SET BACKGROUND (IF PRESENT)

if args.scene != None:
    load_background(pyr_scene, args.scene)





#SET LIGHTS
set_lights(pyr_scene)



#INITIALIZE CAMERA

Rcam = np.eye(4)
#Rcam[:3,3] = [-1,0,-1]
# camera
cam = pyrender.PerspectiveCamera(yfov=np.pi/2, aspectRatio=1280/720)
cam_node = pyr_scene.add(cam, pose=Rcam)



# renderer offscreen
if not args.debug:
    r = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)
frames = []

if not args.debug:
    n_frames = robot1_poses.shape[0]
else:
    n_frames = 1

for t in tqdm(range(n_frames)):


    i = 0
    robot_pos1 = []
    for node,_,_ in mesh_nodes1:
        T = robot1_poses[t][i]
        node.matrix = T 
        robot_pos1.append(T[:3,3])
        i+= 1
    #robot_pos1 = [np.zeros((3,1)) for i in mesh_nodes2]
    
    robot_pos2 = []
    i = 0
    for node,_,_ in mesh_nodes2:
        T = robot2_poses[t][i]
        node.matrix = T 
        robot_pos2.append(T[:3,3])
        i += 1

    # --- scaling ---
    t1_s = trans1[t].copy()
    t1_s[2] -= np.min(human1_js[t,:,2])
    t2_s = trans2[t].copy()
    t2_s[2] -= np.min(human2_js[t,:,2])

    if t == 0:
        s1, s2 = calculate_scale_factors(human1_js[t],human2_js[t], robot_pos1, robot_pos2)
    
    t1_s *= s1 #scale translations
    T1 = np.eye(4)
    T1[:3,3] = t1_s
    for node, _, _ in mesh_nodes1:
        node.matrix = T1 @ node.matrix  # translate nodes in the world    

    
    t2_s *= s2
    T2 = np.eye(4)
    T2[:3,3] = t2_s
    for node, _, _ in mesh_nodes2:
        node.matrix = T2 @ node.matrix # translate nodes in the world 

    if "exo" in camera_mode:
        if t == 0:
            robot_pos1 = np.vstack(robot_pos1) + t1_s
            robot_pos2 = np.vstack(robot_pos2) + t2_s

            robot1_center = np.array([((robot_pos1[:,i].max() + robot_pos1[:,i].min())/2) for i in range(3)]) 
            robot2_center = np.array([((robot_pos2[:,i].max() + robot_pos2[:,i].min())/2) for i in range(3)])

            target = (robot1_center + robot2_center)/2
            direction = robot2_center - robot1_center
            direction[2] = 0
            direction /= np.linalg.norm(direction)
            ortho = np.array([-direction[1], -direction[0],0])
            
            distance = 0.1
            L = np.array([distance/ortho[0], 2*s1[2],0])
            R = np.array([-distance/ortho[0], 2*s1[2],0])
             
            if camera_mode == "exoR":
                camera_pos = target + R * ortho
            else:
                camera_pos = target + L * ortho


            cam_node.matrix = place_camera(camera_mode, camera_pos, target, t=t)
    else:

        cam_node.matrix = place_camera(camera_mode, cameras, target=None,  t=t)



    # --- render frame ---
    if not args.debug:
        color, _ = r.render(pyr_scene)
        color = color.copy()
        if args.bb_mode:
            
            import cv2
            not_green_idx = np.argwhere(np.all(color != [0,255,0], axis=-1))
            
            if not_green_idx.size != 0:
                min_y = np.min(not_green_idx[:,0])
                min_x = np.min(not_green_idx[:,1])
                max_y = np.max(not_green_idx[:,0])
                max_x = np.max(not_green_idx[:,1])
                cv2.rectangle(color,pt1=(min_x,min_y),pt2=(max_x,max_y), color=(255,0,0),thickness=2)


        frames.append(color)
if args.debug:
    pyrender.Viewer(pyr_scene, use_raymond_lighting=True) 
# ------------------- save video -------------------

if not args.debug:
    r.delete()
if args.video:
    imageio.mimsave(f'{args.robot1}_{args.robot2}_{args.camera_mode}.mp4', frames, fps=120)
