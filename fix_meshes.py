from vedo import Mesh, show
import trimesh
import numpy as np

mesh = trimesh.load("/home/rick/TheCompleteRobots/meshes/h1_2_description/meshes/L_hand_base_link.STL")

T_swap_xy = np.eye(4)
T_swap_xy[:3, :3] = np.array([
    [0, 1, 0],  # X' = Y
    [-1, 0, 0],  # Y' = X
    [0, 0, 1],  # Z' = Z
])

# Applica la trasformazione
#mesh.apply_transform(T_swap_xy)

# (Opzionale) centra la mesh se necessario
#offset = -mesh.bounding_box
offset = np.array([0.0,0.06,0])

T_center = np.eye(4)
T_center[:3, 3] = offset
mesh.apply_transform(T_center)
mesh.export("/home/rick/TheCompleteRobots/meshes/h1_2_description/meshes/L_hand_base_link.STL")

show(mesh,axes=1)
