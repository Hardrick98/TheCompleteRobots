import trimesh
import numpy as np

# Carica la mesh della mano (in coordinate locali)
#mesh = trimesh.load('meshes/h1_2_description/meshes/L_hand_base_link.STL')  # o .dae, .obj...

mesh = trimesh.load("/home/rick/TheCompleteRobots/meshes/val_description/meshes/arms/palm_right.dae")

# Ottieni il bounding box axis-aligned (AABB)
bounds = mesh.bounds  # shape (2, 3) → min e max per x, y, z

# Calcola lunghezza (X), larghezza (Y), profondità (Z)
size = bounds[1] - bounds[0]
lunghezza, larghezza, profondità = size

print(f"Lunghezza (X): {lunghezza}")
print(f"Larghezza  (Y): {larghezza}")
print(f"Profondità (Z): {profondità}")
