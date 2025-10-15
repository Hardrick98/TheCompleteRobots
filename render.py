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
parser.add_argument("--video", type=str, default=".")
parser.add_argument("--scene", type=str, default=None)
parser.add_argument("--green_screen",action="store_true")
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



masks = {"nao": [0, 1, 3, 6, 7, 9, 11, 12, 13, 16, 17, 19, 21, 22], "g1":[14,15,2,4,5,17,19,20,9,11,13,23,25,26]}

model = robot1.model
data = robot1.data
q0 = robot1.q0


robotoid = Robotoid(robot1, wheeled)
F, R = robotoid.build()



human1_js = np.load(os.path.join(args.interaction,"data","human1_poses.npy"))
trans1 = np.load(os.path.join(args.interaction,"data","human1_trans.npy"))
human2_js = np.load(os.path.join(args.interaction,"data","human2_poses.npy"))
trans2 = np.load(os.path.join(args.interaction,"data","human2_trans.npy"))


robot1_poses= np.load(f"{args.interaction}/data/{robot_name1}_1_poses.npy")
robot2_poses = np.load(f"{args.interaction}/data/{robot_name2}_2_poses.npy")

if os.path.exists(f"{args.interaction}/data/{args.robot1}_1_data.pkl"):  
    data1 = joblib.load(f"{args.interaction}/data/{args.robot1}_1_data.pkl")
else:
    data1 = {}

if os.path.exists(f"{args.interaction}/data/{args.robot2}_2_data.pkl"):  
    data2 = joblib.load(f"{args.interaction}/data/{args.robot2}_2_data.pkl")
else:
    data2 = {}
    
if args.camera_mode not in data1.keys(): 
    data1[args.camera_mode] = {}
if args.camera_mode not in data2.keys():
        data2[args.camera_mode] = {}

# ------------------- preload robot meshes -------------------
robot1_cache = preload_robot_meshes(robot1)
robot2_cache = preload_robot_meshes(robot2)

cameras = joblib.load(os.path.join(f"{args.interaction}/data",f"{robot_name1}_cameras.pkl"))
# ------------------- setup pyrender -------------------


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
    
if not os.path.exists(f"{args.interaction}/data/random_rotation.npy"):
    Rand_Rz =random_rotation()
    np.save(f"{args.interaction}/data/random_rotation.npy", Rand_Rz)
else:
    Rand_Rz = np.load(f"{args.interaction}/data/random_rotation.npy")

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

    # --- scaling ---
    t1_s = trans1[t].copy()
    t1_s[2] -= np.min(human1_js[t,:,2])
    t2_s = trans2[t].copy()
    t2_s[2] -= np.min(human2_js[t,:,2])


    if t == 0:
        s1, s2 = calculate_scale_factors(human1_js[t],human2_js[t], robot_pos1, robot_pos2)
    
    robot_pos2 = []
    robot_pos1 = []

        
    
    t1_s *= s1 #scale translations
    T1 = np.eye(4)
    T1[:3,3] = t1_s
    for node, _, _ in mesh_nodes1:
        Q = T1 @ node.matrix 
        Q = Rand_Rz@Q
        node.matrix = Q # translate nodes in the world 
        robot_pos1.append(Q[:3,3])

    
    t2_s *= s2
    T2 = np.eye(4)
    T2[:3,3] = t2_s
    for node, _, _ in mesh_nodes2:
        Q = T2@node.matrix
        Q = Rand_Rz@Q
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
            camera_params["E"].append(E[None,:,:])
        
            cam_node.matrix = E
        


    else:

        E = place_camera(camera_mode, cameras, target=None,  t=t)
        camera_params["E"].append(E[None,:,:])
        cam_node.matrix = E

    poses1_3d.append(robot_pos1)
    poses2_3d.append(robot_pos2)

    robot_pos1_exp = np.concatenate((robot_pos1, np.ones((robot_pos1.shape[0], 1))), axis=-1)

    p_camera = (np.linalg.inv(E) @ robot_pos1_exp.T).T  

    p_camera = p_camera[:, :3]  

    poses1_3d_cam.append(p_camera)

    pixels_h = (K @ p_camera.T).T 
    pose1_2d = pixels_h[:, :2] / pixels_h[:, 2:3]

    pose1_2d[:,0] = w - pose1_2d[:,0]


    robot_pos2_exp = np.concatenate((robot_pos2, np.ones((robot_pos2.shape[0], 1))), axis=-1)


    p_camera = (np.linalg.inv(E) @ robot_pos2_exp.T).T  

    p_camera = p_camera[:, :3]  

    poses2_3d_cam.append(p_camera)

    pixels_h = (K @ p_camera.T).T  
    pose2_2d = pixels_h[:, :2] / pixels_h[:, 2:3]

    pose2_2d[:,0] = w - pose2_2d[:,0]

    poses1_2d.append(pose1_2d)
    poses2_2d.append(pose2_2d)

    # --- render frame ---
    if not args.debug:
        color, _ = r.render(pyr_scene)
        color = color.copy()

        import cv2
        #for p in pose1_2d:
        #    cv2.circle(color,center=(int(p[0]),int(p[1])),radius=1,color=(255,0,0),thickness=2)
        #for p in pose2_2d:
        #    cv2.circle(color,center=(int(p[0]),int(p[1])),radius=1,color=(0,0,255),thickness=2)

        cv2.imwrite("test.png", color[:,:,::-1])
        frames.append(color)
    
if args.debug:

    pyrender.Viewer(pyr_scene, use_raymond_lighting=True) 
# ------------------- save video -------------------

if not args.debug:
    r.delete()
if args.video != None:
    imageio.mimsave(f'{args.video}/{args.robot1}_{args.camera_mode}.mp4', frames, fps=120)



data1[args.camera_mode]["pose2D_total"] = np.vstack(np.array(poses1_2d)[None,:,:])
data2[args.camera_mode]["pose2D_total"] = np.vstack(np.array(poses2_2d)[None,:,:])

data1[args.camera_mode]["pose2D"] = np.vstack(np.array(poses1_2d)[None,:,:])[:,masks[args.robot1],:]
data2[args.camera_mode]["pose2D"] = np.vstack(np.array(poses2_2d)[None,:,:])[:,masks[args.robot2],:]



if "world" not in data1.keys():
    data1["world"] = {}
    data1["world"]["pose3D"] = np.vstack(np.array(poses1_3d)[None,:,:])
    data2["world"] = {}
    data2["world"]["pose3D"] = np.vstack(np.array(poses2_3d)[None,:,:])


data1[args.camera_mode]["pose3D_total"]  = np.vstack(np.array(poses1_3d_cam)[None,:,:])
data2[args.camera_mode]["pose3D_total"]  = np.vstack(np.array(poses2_3d_cam)[None,:,:])

data1[args.camera_mode]["pose3D"]  = np.vstack(np.array(poses1_3d_cam)[None,:,:])[:,masks[args.robot1],:]
data2[args.camera_mode]["pose3D"]  = np.vstack(np.array(poses2_3d_cam)[None,:,:])[:,masks[args.robot2],:]

camera_params["E"] = np.vstack(camera_params["E"])

data1[args.camera_mode]["camera_params"] = camera_params
data2[args.camera_mode]["camera_params"] = camera_params


joblib.dump(data1, f"{args.interaction}/data/{args.robot1}_1_data.pkl" )
joblib.dump(data2, f"{args.interaction}/data/{args.robot2}_2_data.pkl" )