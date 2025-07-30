import trimesh
import numpy as np
import os


def compose_hand_mesh(model, visual_model, data, frame):

    frame_id = model.getFrameId(frame)
    frame = model.frames[frame_id]
    root_link_id = frame.parentJoint  

    desc_joints = []
    for j in range(model.njoints):
        cur = j
        while cur != 0:
            cur = model.parents[cur]
            if cur == root_link_id:
                desc_joints.append(j)
                break


    desc_joints = list(set(desc_joints + [root_link_id]))

    meshes = []
    for geom in visual_model.geometryObjects:
        if geom.parentJoint in desc_joints:
            meshes.append({
                "joint": model.names[geom.parentJoint],
                "meshPath": geom.meshPath,
                "scale": geom.meshScale,
                "frame_parent": geom.parentFrame
            })
    all_meshes = []

    
    for geom in visual_model.geometryObjects:
        if geom.parentJoint in desc_joints:
            mesh_path = geom.meshPath
            if not os.path.isfile(mesh_path):
                print(f"⚠️ File non trovato: {mesh_path}")
                continue
            # Carica mesh con Trimesh
            mesh = trimesh.load(mesh_path)
            placement = data.oMf[geom.parentFrame]
            placement_world = placement.act(geom.placement)
            R = placement_world.rotation
            p = placement_world.translation


            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = p

            mesh.apply_transform(T)
            all_meshes.append(mesh)
    if all_meshes:
        combined = trimesh.util.concatenate(all_meshes)
        bounds = combined.bounds  # shape (2, 3) → min e max per x, y, z

        # Calcola lunghezza (X), larghezza (Y), profondità (Z)
        size = bounds[1] - bounds[0]
        lunghezza, larghezza, profondità = size

        #combined.show()  # oppure: combined.export("combined_mesh.obj")
        return lunghezza, larghezza, profondità
    else:
        print("Nessuna mesh trovata")