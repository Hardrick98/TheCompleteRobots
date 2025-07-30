import bpy

# Rimuove oggetti di default
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Importa .dae
bpy.ops.wm.collada_import(filepath="head_camera.dae")

# Esporta in .stl
bpy.ops.export_mesh.stl(filepath="head_camera.stl")
