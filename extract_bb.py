from utils import *
import argparse
import joblib
import os
from tqdm import tqdm
import numpy as np
from robotoid import Robotoid
import pyrender
from scipy.spatial.transform import Rotation as Rot
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
parser.add_argument("--bb_mode1",action="store_true")
parser.add_argument("--bb_mode2",action="store_true")
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

robot_folder = f"{args.interaction}/{args.robot1}"
robot_folder = f"{args.interaction}/{args.robot1}"
data_dict = joblib.load(f"{robot_folder}/data/{args.robot1}_prep.pkl")


robot1_poses= data_dict["robot_1_poses"]
robot2_poses = data_dict["robot_2_poses"]
s1, s2 = data_dict["scales"]

if  not os.path.exists(f"{robot_folder}/{args.camera_mode}"):  
    os.makedirs(f"{robot_folder}/{args.camera_mode}")

if os.path.exists(f"{robot_folder}/data/{args.robot1}_1_data.pkl"):  
    data1 = joblib.load(f"{robot_folder}/data/{args.robot1}_1_data.pkl")
else:
    data1 = {}

if os.path.exists(f"{robot_folder}/data/{args.robot2}_2_data.pkl"):  
    data2 = joblib.load(f"{robot_folder}/data/{args.robot2}_2_data.pkl")
else:
    data2 = {}
    
if args.bb_mode1:
    if args.camera_mode not in data1.keys(): 
        data1[args.camera_mode] = {}
if args.bb_mode2:
    if args.camera_mode not in data2.keys():
        data2[args.camera_mode] = {}

# ------------------- preload robot meshes -------------------
robot1_cache = preload_robot_meshes(robot1)
robot2_cache = preload_robot_meshes(robot2)

cameras = joblib.load(os.path.join(f"{robot_folder}/data",f"{robot_name1}_cameras.pkl"))
# ------------------- setup pyrender -------------------
scene_data_path = os.path.join(f"{robot_folder}/data",f"scene_data.pkl")

if os.path.exists(scene_data_path):
    scene_data = joblib.load(scene_data_path)
    if "scene_point" in scene_data.keys():
        scene_point_index = scene_data["scene_point"] 
else:
    scene_data = {}
    scene_point_index = None


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


scene_point, scene_point_index = load_background_manual(pyr_scene, args.scene, scene_point_index=None, bb_mode= True)

    
if not os.path.exists(f"{robot_folder}/data/random_rotation.npy"):
    Rand_Rz =random_rotation()
    np.save(f"{robot_folder}/data/random_rotation.npy", Rand_Rz)
else:
    Rand_Rz = np.load(f"{robot_folder}/data/random_rotation.npy")


#SET LIGHTS
set_lights(pyr_scene)



#INITIALIZE CAMERA

Rcam = np.eye(4)
#Rcam[:3,3] = [-1,0,-1]
# camera
cam = pyrender.PerspectiveCamera(yfov=np.pi/2, aspectRatio=1280/720)

cam_node = pyr_scene.add(cam, pose=Rcam)

w = 1280
h = 720
aspect = 1280/720
yfov = np.pi/2

f_y = 0.5 * h / np.tan(yfov/2)
f_x = f_y
c_x = w / 2
c_y = h / 2
K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])



bounding_boxes = []

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
    
    robot_pos2 = []
    i = 0
    for node,_,_ in mesh_nodes2:
        T = robot2_poses[t][i]
        node.matrix = T 
        robot_pos2.append(T[:3,3])
        i += 1
    
    robot_pos2 = []
    robot_pos1 = []

        
    

    for node, _, _ in mesh_nodes1:
        Q = node.matrix 
        Q = scene_point@Rand_Rz@Q
        node.matrix = Q # translate nodes in the world 
        robot_pos1.append(Q[:3,3])


    for node, _, _ in mesh_nodes2:
        Q = node.matrix
        Q = scene_point@Rand_Rz@Q
        node.matrix = Q
        robot_pos2.append(Q[:3,3])

    

    robot_pos1 = np.vstack(robot_pos1)
    robot_pos2 = np.vstack(robot_pos2)
    if "exo" in camera_mode:
        if t == 0:

            robot1_center = np.mean(robot_pos1, axis=0)
            robot2_center = np.mean(robot_pos2, axis=0)

            
            target = (robot1_center + robot2_center) / 2.0

            direction = robot2_center - robot1_center
            direction[2] = 0
            direction /= np.linalg.norm(direction)

            rot_axis = np.array([0, 0, 1.0])  # ruota attorno all'asse Z
            rot = Rot.from_rotvec(rot_axis * np.pi/2).as_matrix()
            robot_direction = direction.copy()
            direction = rot @ direction
            

            up = np.array([0, 0, 1.0])


            # Parametri camera
            horizontal_offset = 0.3   
            vertical_offset = 0.15   
            distance_back = 2 * s1[2]      

            center_pos = target - direction * distance_back + np.array([0, 0, vertical_offset])

            if camera_mode == "exoL":
                camera_pos = center_pos - 0.5 * horizontal_offset * robot_direction
            else:
                camera_pos = center_pos + 0.5 * horizontal_offset * robot_direction
    

            E = place_camera(camera_mode, camera_pos, target, t=t)        
            cam_node.matrix = E
        


    else:

        E = place_camera(camera_mode, cameras, target=None,  t=t, random_rotation=Rand_Rz)
        cam_node.matrix = scene_point@ E

    if args.bb_mode1:
        for node, _, _ in mesh_nodes2:
            if node in pyr_scene.get_nodes():
                pyr_scene.remove_node(node)
    elif args.bb_mode2:
        for node, _, _ in mesh_nodes1:
            if node in pyr_scene.get_nodes():
                pyr_scene.remove_node(node)


    # --- render frame ---
    if not args.debug:
        color, _ = r.render(pyr_scene)
        color = color.copy()

        
        if args.bb_mode1 or args.bb_mode2:
            
            import cv2
            not_green_idx = np.argwhere(np.all(color != [0,255,0], axis=-1))
            
            if not_green_idx.size != 0:
                min_y = np.min(not_green_idx[:,0])
                min_x = np.min(not_green_idx[:,1])
                max_y = np.max(not_green_idx[:,0])
                max_x = np.max(not_green_idx[:,1])
                bounding_boxes.append(np.array([[min_x,min_y,max_x, max_y]]))
                cv2.rectangle(color,pt1=(min_x,min_y),pt2=(max_x,max_y), color=(255,0,0),thickness=2)
            else:
                bounding_boxes.append(np.array([[-1,-1,-1,-1]]))

        frames.append(color)
    
        # dopo il rendering, reinserisci
    if args.bb_mode1:
        for node, _, _ in mesh_nodes2:
            if node not in pyr_scene.get_nodes():
                pyr_scene.add_node(node)
    elif args.bb_mode2:
        for node, _, _ in mesh_nodes1:
            if node not in pyr_scene.get_nodes():
                pyr_scene.add_node(node)
            
if args.debug:

    pyrender.Viewer(pyr_scene, use_raymond_lighting=True) 
# ------------------- save video -------------------

if not args.debug:
    r.delete()
if args.video:
    imageio.mimsave(f'f"{robot_folder}/{args.robot1}_{args.robot2}_{args.camera_mode}.mp4', frames, fps=120)


if args.bb_mode1:
    data1[args.camera_mode]["bb2D"] = np.vstack(bounding_boxes)
if args.bb_mode2:
    
    data2[args.camera_mode]["bb2D"] = np.vstack(bounding_boxes)


joblib.dump(data1, f"{robot_folder}/data/{args.robot1}_1_data.pkl" )
joblib.dump(data2, f"{robot_folder}/data/{args.robot2}_2_data.pkl" )