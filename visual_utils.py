import numpy as np
import pyrender
import trimesh
import random
import os
from scipy.spatial import cKDTree
import joblib


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
              "baroque_room":[[-1,-0.5,0]], 
              "city":[[-1,-0.3,0]], 
              "hospital":[[-1,0,0],[-1.8,0,0]], 
              "estensi_hallway":[[-40,35,-0.1]],
              "estensi_room":[[0,0,0]]}


correction_factors = {"baroque_room":0.4, 
              "city":1, 
              "hospital": 0, 
              "office":0,
              "living_room":0.2}



def random_rotation():
    
    theta = random.uniform(0, 2*np.pi)
    Rz = np.array([
    [ np.cos(theta), -np.sin(theta), 0,0],
    [ np.sin(theta),  np.cos(theta), 0,0],
    [ 0,              0,             1,0],
    [0,0,0,1]
    ])
    

            
    
    return Rz
    


def fix_materials(mesh):
    # Se il materiale ha un'immagine con 1-2 canali, convertila o rimuovila
    mat = getattr(mesh.visual, "material", None)
    if mat is not None and hasattr(mat, "image"):
        img = mat.image
        if img is not None and img.ndim == 3 and img.shape[2] < 3:
            print(f"Removing unsupported texture with shape {img.shape}")
            mat.image = None
    return mesh

def load_background_manual(pyr_scene, scene_path):
    

    

    scene_positions = {"estensi_hallway":[[-40,35,-0.1], [-50,35,-0.1], [-30,35,-0.1]], "estensi_room":[[0,0,0.7],[-3,-4,0.7],[3,-4,0.7],[4,-5,0.7]], "city":[[21,9,-0.2],[-5,-5,0],[-10,43,-0.2],[50,43,-0.2],[-31,7,-0.2]]}

       
    scene_mesh = trimesh.load_scene(scene_path)



    scene_name = scene_path.split("/")[-1].removesuffix(".glb")
    index = random.randint(0, len(scene_positions[scene_name])-1)

    scene_point = np.array(scene_positions[scene_name][2])

    if scene_name == "estensi_room" or scene_name == "city":
        T0 = np.eye(4)
        T0[:3,:3] = get_rotation_matrix(axis="y") 
        scene_mesh.apply_transform(T0)

    scene_mesh.apply_translation(scene_point)

    

    vertices = []

    for node_name in scene_mesh.graph.nodes_geometry:
        T, geom_name = scene_mesh.graph[node_name]
        geom = scene_mesh.geometry[geom_name].copy()
        geom.apply_transform(T)

        vertices.append(geom.vertices)

        pyr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
        pyr_scene.add(pyr_mesh)

    F = np.eye(4)

    return F



def load_background_auto(pyr_scene, scene_path, robot_box, max_tries=2):
    import random, numpy as np, trimesh, pyrender
    
    robot_min, robot_max = robot_box
    robot_center = (robot_max+robot_min)/2
    offset = robot_max - robot_min


    scene_mesh = trimesh.load_scene(scene_path)
    scene_name = scene_path.split("/")[-1].removesuffix(".glb")

    T0 = np.eye(4)
    if scene_name == "estensi_hallway":
        pass
    elif scene_name == "baroque_room":
        T0[:3,:3] = get_rotation_matrix(theta=-np.pi/4,axis="z") @ get_rotation_matrix(axis="y") 
    else:
        T0[:3,:3] = get_rotation_matrix(axis="y") 

    scene_mesh.apply_transform(T0)

    bbox = scene_mesh.bounding_box
    center = bbox.centroid
    scene_mesh.apply_translation(robot_center-center)

    vertices = []

    for node_name in scene_mesh.graph.nodes_geometry:
        T, geom_name = scene_mesh.graph[node_name]
        geom = scene_mesh.geometry[geom_name].copy()
        geom.apply_transform(T)
        vertices.append(geom.vertices)

        pyr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
        pyr_scene.add(pyr_mesh)

    all_points = np.vstack(vertices)
    

    # Calcoli base scena
    scene_mins = np.min(all_points, axis=0)
    scene_maxs = np.max(all_points, axis=0)

    floor_z = float(scene_mins[2])
    # Limiti di posizionamento con margine
    padding = 1
    safe_mins = scene_mins + padding
    safe_maxs = scene_maxs - offset - padding

    # Ricerca posizione valida
    best_scene = np.array([0,0,0])
    best_coll = 100
    threshold = 0.1  # 2 cm: distanza minima accettabile dagli oggetti

    for tries in range(max_tries):
        scene_point = np.array([
            random.uniform(safe_mins[0], safe_maxs[0]),
            random.uniform(safe_mins[1], safe_maxs[1]),
            floor_z - robot_min[2] + correction_factors.get(scene_name, 0.0),
        ])

        mins = robot_min + scene_point
        maxs = robot_max + scene_point
        coll = check_collision(mins, maxs, all_points, scene_point)

        # Se è libero oltre la soglia → accettalo
        if coll < threshold:
            final = np.eye(4)
            final[:3, 3] = scene_point
            return final


        if coll  < best_coll:
            best_coll = coll
            best_scene = scene_point

    # Nessuna posizione libera trovata
    print(f"Nessuna posizione totalmente libera trovata dopo {max_tries} tentativi.")

    final = np.eye(4)
    final[:3, 3] = best_scene

    

    return final



def check_collision(maxs, mins, scene_points, scene_point, threshold=0.01):
    """
    Controlla se il bounding box del robot collide con la scena.
    Ritorna percentuale di punti del robot troppo vicini alla scena.
    """
    # Crea una griglia di punti nel volume del robot (approssimazione)
    xs = np.linspace(mins[0], maxs[0], 5)
    ys = np.linspace(mins[1], maxs[1], 5)
    zs = np.linspace(mins[2], maxs[2], 5)
    xx, yy, zz = np.meshgrid(xs, ys, zs)
    robot_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    # KDTree per velocizzare la ricerca della distanza minima
    tree = cKDTree(scene_points)
    dists, _ = tree.query(robot_points, k=1)
    num_collisions = np.sum(dists < threshold)

    percent_inside = 100 * num_collisions / robot_points.shape[0]
    print(percent_inside)
    return percent_inside

def check_collision(maxs, mins, scene_points, scene_point):


    inside_mask = np.all((scene_points >= mins) & (scene_points <= maxs), axis=1)
    percent_inside = 100 * (inside_mask.sum() / scene_points.shape[0])
    return percent_inside