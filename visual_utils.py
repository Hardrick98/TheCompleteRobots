import numpy as np
import pyrender
import trimesh
import random
import os

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

def look_at(camera_pos, target):
    
    forward = (target - camera_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 0, 1]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rot = np.eye(4)
    
    rot[:3, :3] = np.vstack([right, up, -forward]).T
    rot[:3, 3] = camera_pos
    return rot

def place_camera(camera_mode, camera_poses, target, t):
    
    if "exo" in camera_mode:
        

            if camera_mode == "exoR":
                camera_pos=camera_poses 
            else:
                camera_pos=camera_poses 
            
            camera_pose = look_at(
                camera_pos=camera_pos,   # camera position
                target=target       # camera target
            )

        
            P = camera_pose
    
    else:
        F = np.eye(4)
        F[:3,:3] = np.linalg.inv(Rz)@Ry
        P = camera_poses[camera_mode][t] @ F


    return P


def set_lights(pyr_scene):
    
    # key light
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    pyr_scene.add(key_light, pose=np.eye(4))  


    # fill light
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    pose_fill = np.eye(4)
    pose_fill[:3,3] = [-2,2,1]
    pyr_scene.add(fill_light, pose=pose_fill)


    # back light
    back_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    pose_back = np.eye(4)
    pose_back[:3,3] = [0,-3,1]
    pyr_scene.add(back_light, pose=pose_back)

def calculate_scale_factors(human1, human2, robot1, robot2):
        robot_pos1_bounds = np.ptp(np.vstack(robot1), axis=0) #find lenghts on the three axis
        human_bounds1 = np.ptp(human1, axis=0)
        s1 = robot_pos1_bounds / human_bounds1
        robot_pos2_bounds = np.ptp(np.vstack(robot2), axis=0)
        human_bounds2 = np.ptp(human2, axis=0)
        s2 = robot_pos2_bounds / human_bounds2

        return s1,s2


placements = {"room": [[-1,-0.3,0]], "city":[[-1,-0.3,0]], "hospital":[[-1,0,0],[-1.8,0,0]], "estensi_light":[[-40,35,-0.1]]}

def random_rotation():
    
    theta = random.uniform(0, 2*np.pi)
    Rz = np.array([
    [ np.cos(theta), -np.sin(theta), 0,0],
    [ np.sin(theta),  np.cos(theta), 0,0],
    [ 0,              0,             1,0],
    [0,0,0,1]
    ])
    

            
    
    return Rz
    

def load_background(pyr_scene, scene_path):
    
    
    scene_mesh = trimesh.load_scene(scene_path)
    scene_name = scene_path.split("/")[-1].removesuffix(".glb")
    T_center = np.eye(4)
    T_center[:3,3] = placements[scene_name][0]
    scene_mesh.apply_transform(T_center)

    for node_name in scene_mesh.graph.nodes_geometry:
        T, geom_name = scene_mesh.graph[node_name]
        geom = scene_mesh.geometry[geom_name]
        pyr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
        T0 = np.eye(4)
        if scene_name == "estensi_light":
            pass
        else:
            T0[:3,:3] = Ry 
        pyr_scene.add(pyr_mesh, pose=T0@T)

    scene_mesh.matrix = T0

def preload_robot_meshes(robot):
    cache = {}
    frames = robot.body
    
    for visual in robot.visual_model.geometryObjects:
        mesh_path = visual.meshPath
        if not os.path.exists(mesh_path):
            continue
        try:
            mesh = trimesh.load_mesh(mesh_path)
            mesh.apply_scale(visual.meshScale)
            cache[visual.name] = (mesh, visual.placement, frames[visual.name[:-2]])
        except Exception as e:
            print(f"Errore caricando mesh {mesh_path}: {e}")
            continue
    return cache