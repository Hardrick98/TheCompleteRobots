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
from pose_masks import masks

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
[ np.cos(theta), -np.sin(theta), 0,0],
[ np.sin(theta),  np.cos(theta), 0,0],
[ 0,              0,             1,0],
[0,0,0,1]
])
    

# ------------------- parser e robot -------------------
robot_list = [r.removesuffix(".urdf") for r in os.listdir("URDF") if r.endswith(".urdf")]

parser = argparse.ArgumentParser(description="Retarget human to robot")
parser.add_argument("--robot1", type=str, default="nao")
parser.add_argument("--robot2", type=str, default="nao")
parser.add_argument("--camera_mode", type=str, default="exo")
parser.add_argument("--interaction", type=str)
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--scene", type=str, default=None)
parser.add_argument("--green_screen",action="store_true")
parser.add_argument("--frames",action="store_true")
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

human1_js = np.load(os.path.join(robot_folder,"data","human1_poses.npy"))
trans1 = np.load(os.path.join(robot_folder,"data","human1_trans.npy"))
human2_js = np.load(os.path.join(robot_folder,"data","human2_poses.npy"))
trans2 = np.load(os.path.join(robot_folder,"data","human2_trans.npy"))


robot1_poses= np.load(f"{robot_folder}/data/{robot1.name}_1_poses.npy")
robot2_poses = np.load(f"{robot_folder}/data/{robot2.name}_2_poses.npy")

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
    
if args.camera_mode not in data1.keys(): 
    data1[args.camera_mode] = {}
if args.camera_mode not in data2.keys():
        data2[args.camera_mode] = {}

# ------------------- preload robot meshes -------------------
robot1_cache = preload_robot_meshes(robot1)
robot2_cache = preload_robot_meshes(robot2)

cameras = joblib.load(os.path.join(f"{robot_folder}/data",f"{robot_name1}_cameras.pkl"))
# ------------------- setup pyrender -------------------

if args.frames:
    os.makedirs(f"{args.interaction}/{args.robot1}/{args.camera_mode}",exist_ok=True)

if args.green_screen == True:
    pyr_scene = pyrender.Scene(ambient_light=[0.5,0.5,0.5],bg_color=[0,255,0])
else:
    pyr_scene = pyrender.Scene(ambient_light=[0.5,0.5,0.5],bg_color=[135,206,235])


mesh_nodes1 = []
mesh_nodes2 = []


# loading robot1 meshes
for name, (mesh, placement, parentFrame) in robot1_cache.items():
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    node = pyr_scene.add(pyr_mesh)
    mesh_nodes1.append((node, mesh, placement, parentFrame))


# loading robot2 meshes
for name, (mesh, placement, parentFrame) in robot2_cache.items():
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    node = pyr_scene.add(pyr_mesh)
    mesh_nodes2.append((node, mesh, placement, parentFrame))



#SET BACKGROUND (IF PRESENT)

if args.scene != None:
    all_points, scene_point = load_background_debug(pyr_scene, args.scene)

print(scene_point)



#SET LIGHTS
set_lights(pyr_scene)




#INITIALIZE CAMERA

Rcam = np.eye(4)
cam = pyrender.PerspectiveCamera(yfov=np.pi/2, aspectRatio=1280/720)
cam_node = pyr_scene.add(cam, pose=Rcam)

w = 1280
h = 720
yfov = np.pi/2

f_y = 0.5 * h / np.tan(yfov/2)
f_x = f_y
c_x = w / 2
c_y = h / 2
K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])



camera_params = {"K": K, "E":[]}


poses1_2d = []
poses2_2d = []
poses1_3d = []
poses2_3d = []
poses1_3d_cam = []
poses2_3d_cam = []

# renderer offscreen
if not args.debug:
    r = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)
frames = []

if not args.debug:
    n_frames = robot1_poses.shape[0]
else:
    n_frames = 1
    
## Randomly rotate interaction
    
if not os.path.exists(f"{robot_folder}/data/random_rotation.npy"):
    Rand_Rz =random_rotation()
    np.save(f"{robot_folder}/data/random_rotation.npy", Rand_Rz)
else:
    Rand_Rz = np.load(f"{robot_folder}/data/random_rotation.npy")

temporal_max = np.zeros((n_frames,3))
temporal_min = np.zeros((n_frames,3))


faces = np.array([
    # bottom
    [0, 1, 2], [0, 2, 3],
    # top
    [4, 5, 6], [4, 6, 7],
    # front
    [0, 1, 5], [0, 5, 4],
    # back
    [3, 2, 6], [3, 6, 7],
    # left
    [0, 4, 7], [0, 7, 3],
    # right
    [1, 5, 6], [1, 6, 2]
])



for t in tqdm(range(n_frames)):
    i = 0
    robot_pos1 = []
    for node,mesh,_,_ in mesh_nodes1:
        P = robot1_poses[t][i]
        node.matrix = P 
        robot_pos1.append(P[:3,3])
        i+= 1
    
    robot_pos2 = []
    i = 0
    for node,mesh,_,_ in mesh_nodes2:
        P = robot2_poses[t][i]
        node.matrix = P 
        robot_pos2.append(P[:3,3])
        i += 1

    # --- scaling ---
    t1_s = trans1[t].copy()
    t1_s[2] -= np.min(human1_js[t,:,2])
    t2_s = trans2[t].copy()
    t2_s[2] -= np.min(human2_js[t,:,2])


    if t == 0:
        s1, s2 = calculate_scale_factors(human1_js[t],human2_js[t], robot_pos1, robot_pos2)
    
    robot_pos2 = []
    robot_pos1 = []

    meshes1 = []
    meshes2 = []
    
    t1_s *= s1 #scale translations
    T1 = np.eye(4)
    T1[:3,3] = t1_s
    for node,mesh, _, _ in mesh_nodes1:
        Q = T1 @ node.matrix 
        Q = scene_point@Rand_Rz@Q
        node.matrix = Q # translate nodes in the world 
        m = mesh.copy()
        m.apply_transform(node.matrix)
        meshes1.append(m)
        robot_pos1.append(Q[:3,3])

    
    t2_s *= s2
    T2 = np.eye(4)
    T2[:3,3] = t2_s
    for node,mesh, _, _ in mesh_nodes2:
        Q = T2@node.matrix
        Q = scene_point@Rand_Rz@Q
        node.matrix = Q
        m = mesh.copy()
        m.apply_transform(node.matrix)
        meshes2.append(m)
        robot_pos2.append(Q[:3,3])

    

    robot1_obj = trimesh.util.concatenate(meshes1).bounding_box.to_dict()
    robot2_obj = trimesh.util.concatenate(meshes2).bounding_box.to_dict()

    T = np.array(robot1_obj["transform"])
    ext = np.array(robot1_obj["extents"])
    minsR1 = T[:3, 3] - ext/2
    maxsR1 = T[:3, 3] + ext/2
    T = np.array(robot2_obj["transform"])
    ext = np.array(robot2_obj["extents"])
    center2 = T[:3, 3] + ext/2
    minsR2 = T[:3, 3] - ext/2
    maxsR2 = T[:3, 3] + ext/2

    maxs = np.max(np.concatenate((maxsR1[None,:], maxsR2[None,:]), axis=0), axis=0)
    mins = np.min(np.concatenate((minsR1[None,:], minsR2[None,:]), axis=0), axis=0)

    temporal_max[t] = maxs
    temporal_min[t] = mins


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
            camera_params["E"].append(E[None,:,:])
        
            cam_node.matrix = E
   

    # --- render frame ---
    if args.video != None or args.frames:



        color, _ = r.render(pyr_scene)
        frames.append(color)

        if args.frames:
            imageio.imwrite(f"{args.interaction}/{args.robot1}/{args.camera_mode}/frame_{t:05d}.png", color)
    

maxs = np.max(temporal_max, axis=0)
mins = np.min(temporal_min, axis=0)
#SORT A VALID POINT

robot_box = (mins, maxs)




if args.debug:
    pyrender.Viewer(pyr_scene, use_raymond_lighting=True) 
if not args.debug:
    r.delete()
if args.video != None:
    imageio.mimsave(f'{args.video}/cube.mp4', frames, fps=120)

