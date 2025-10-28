import numpy as np
import pyrender
import trimesh
import random
import os


import joblib

def preload_robot_meshes(robot):
    cache = {}
    frames = robot.body

    if robot.name == "icub":
        init = joblib.load("icub_init.pkl")
    
    for visual in robot.visual_model.geometryObjects:
        mesh_path = visual.meshPath
        if not os.path.exists(mesh_path):
            continue
        try:
            mesh = trimesh.load_mesh(mesh_path)
            mesh.apply_scale(visual.meshScale)
            if robot.name == "icub":
                cache[visual.name] = (mesh, init[visual.name], frames[visual.name[:-2]])
            else:
                cache[visual.name] = (mesh, visual.placement, frames[visual.name[:-2]])
        except Exception as e:
            print(f"Errore caricando mesh {mesh_path}: {e}")
            continue
    return cache

def get_rotation_matrix(theta=np.pi/2, axis="x"):


    if axis == "x":

        Rx = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0,             1, 0            ],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        return Rx
    
    elif axis == "y":
        Ry = np.array([
        [1, 0,           0          ],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
        ])

        return Ry

    elif axis == "z":
        Rz = np.array([
    [ np.cos(theta), -np.sin(theta), 0],
    [ np.sin(theta),  np.cos(theta), 0],
    [ 0,              0,             1]
    ])
        return Rz

    else:
        return np.eye(3)
    

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

def place_camera(camera_mode, camera_poses, target, t, random_rotation=np.eye(4)):
    
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
        Rz = get_rotation_matrix(axis="z")
        Ry = get_rotation_matrix(axis="y")
        F[:3,:3] = np.linalg.inv(Rz)@Ry
        P = (random_rotation @ camera_poses[camera_mode][t]) @ F


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


placements = {"room": [[-1,-0.3,0]], 
              "room2":[[-1,-0.5,0]], 
              "city":[[-1,-0.3,0]], 
              "hospital":[[-1,0,0],[-1.8,0,0]], 
              "estensi_light":[[-40,35,-0.1]]}



correction_factors = {"room": 0.1, 
              "room2":0.4, 
              "city":7.5, 
              "hospital": 0.25, 
              "estensi_light":20}

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
            T0[:3,:3] = get_rotation_matrix(axis="y") 
        pyr_scene.add(pyr_mesh, pose=T0@T)

    scene_mesh.matrix = T0

    return np.eye(4)

import joblib




def load_background_new(pyr_scene, scene_path, robot_box, max_tries=100):
    import random, trimesh, numpy as np, pyrender

    scene_mesh = trimesh.load_scene(scene_path)
    scene_name = scene_path.split("/")[-1].removesuffix(".glb")

    T0 = np.eye(4)
    if scene_name != "estensi_light":
        T0[:3, :3] = get_rotation_matrix(axis="y")

    scene_mesh.apply_transform(T0)
    bbox = scene_mesh.bounding_box_oriented.to_dict()
    T = np.array(bbox["transform"])
    center = T[:3, 3]
    T_center = np.eye(4)
    T_center[:3, 3] = -center
    scene_mesh.apply_transform(T_center)

    bbox = scene_mesh.bounding_box_oriented.to_dict()
    ext = np.array(bbox["extents"])
    T = np.array(bbox["transform"])
    center = T[:3, 3]
    mins = center - ext / 2
    T_up = np.eye(4)
    T_up[:3, 3] = np.array([0, 0, -mins[2]])
    scene_mesh.apply_transform(T_up)

    bbox = scene_mesh.bounding_box_oriented.to_dict()
    scene_ext = np.array(bbox["extents"])
    T = np.array(bbox["transform"])
    scene_center = T[:3, 3]


    scene_mins = scene_center - scene_ext / 2
    scene_maxs = scene_center + scene_ext / 2

    transformed_geoms = []
    for node_name in scene_mesh.graph.nodes_geometry:
        node_T, geom_name = scene_mesh.graph[node_name]
        geom = scene_mesh.geometry[geom_name].copy()
        geom.apply_transform(node_T)  # porta ogni geom in world coords
        transformed_geoms.append(geom)

    combined = trimesh.util.concatenate(transformed_geoms)
    all_points = combined.vertices


    pyr_mesh = pyrender.Mesh.from_trimesh(combined, smooth=True)
    pyr_scene.add(pyr_mesh)


    tries = 0
    percent_inside = 100
    robot_min, robot_max = robot_box
    best_score = 100
    best_scene = scene_center

    floor_z = float(np.min(all_points[:, 2]))
    scene_mins = np.min(all_points,axis=0)
    scene_maxs = np.max(all_points,axis=0)

    robot_center = (robot_min + robot_max)/2
    offset = (robot_max-robot_min)
    print(offset)
    safe_mins = scene_mins 
    safe_maxs = scene_maxs - offset 
    
    while tries < max_tries:
        # Genera una posizione casuale del robot completamente dentro la scena
        # Tenendo conto delle dimensioni del robot

        print("safes", safe_mins, safe_maxs)
        #random.uniform(safe_mins[0], safe_maxs[0])
        #random.uniform(safe_mins[1], safe_maxs[1])



        scene_point = np.array([
            random.uniform(safe_mins[0], safe_maxs[0])-robot_min[0],
            random.uniform(safe_mins[1], safe_maxs[1])-robot_min[1],
            floor_z - robot_min[2] + correction_factors[scene_name]
        ])

        print("T",scene_point)

        robot_min, robot_max = robot_box
        maxs = robot_max + scene_point
        mins = robot_min + scene_point

        print("points", mins,maxs)
        # Verifica che non compenetri con l’ambiente
        percent_inside, scene_point = check_collision(maxs, mins, all_points, scene_point)

        if percent_inside < 0.3:
            break  # trovato un punto valido

        tries += 1



    if percent_inside < best_score:
        best_score = percent_inside
        best_scene = scene_point

    if percent_inside >= 0.3:
        print(f"⚠️ Nessuna posizione valida trovata dopo {tries} tentativi. {percent_inside}, {scene_point}")
        scene_point = best_scene
    else:
        print(f"✅ Posizione valida trovata dopo {tries} tentativi: {scene_point}")

    final = np.eye(4)
    final[:3, 3] = scene_point
    return final

   




def check_collision(maxs, mins, scene_points, scene_point):


    inside_mask = np.all((scene_points >= mins) & (scene_points <= maxs), axis=1)
    percent_inside = 100 * (inside_mask.sum() / scene_points.shape[0])
    return percent_inside, scene_point