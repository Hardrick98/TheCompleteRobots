import json, trimesh

with open("materials.json") as f:
    materials = json.load(f)

for link_name, rgba in materials.items():
    mesh = trimesh.load(f"meshes/g1_description/meshes_not_colored/{link_name}.STL")
    mesh.visual.vertex_colors = [int(c*255) for c in rgba]
    mesh.export(f"meshes/g1_description/meshes/{link_name}.dae")