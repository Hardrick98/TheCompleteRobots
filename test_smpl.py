import torch
from smplx import SMPL, SMPLX
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as Rot
arr = np.load("/datasets/HumanoidX/human_pose/youtube/ladder_setup_and_placement_training_clip_3.npz", allow_pickle=True)
from utils import compute_global_orientations_smplx


import trimesh



index_keypoints = [
    1, 4, 7,            # left hip, knee, ankle
    2, 5, 8,            # right hip, knee, ankle
    16, 18, 20,         # left shoulder, elbow, wrist
    17, 19, 21          # right shoulder, elbow, wrist
]



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_mesh_and_joints(vertices, joints, faces, directions, title="SMPLX Mesh + Joints"):

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Decima la mesh a ~20% dei vertici originali
    target_faces = 0.05
    mesh = mesh.simplify_quadric_decimation(target_faces)
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Mesh
    mesh = Poly3DCollection(vertices[faces], alpha=0.5)
    mesh.set_facecolor((0.8, 0.8, 1))
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    #ax.scatter(joints[:,0], joints[:,1], joints[:,2])
    
    # Joints
    print(len(directions))
    for i in [0,15,20,21]:
        ax.quiver(
            joints[i,0], joints[i,1],joints[i,2],                    # Punto di origine
            directions[i][0],              # Componente x della direzione
            directions[i][1],              # Componente y della direzione
            directions[i][2],              # Componente z della direzione
            length=1.0,
            linewidth = 3,# Lunghezza della freccia
            color=[0,0,i/22],
            normalize=True             # Normalizza la direzione
        )
    # Setup view
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=20, azim=-70)
    plt.show()

def load_simple(arr):
    smpl = arr["smpl"][()]
    global_orient = torch.from_numpy(smpl['global_orient'][0]).reshape(1, -1).to(torch.float32)
    body_pose_raw = torch.from_numpy(smpl['body_pose'][0][:21]).reshape(1, -1).to(torch.float32)
    body_pose = torch.from_numpy(smpl['body_pose'][0][:]).reshape(1, -1).to(torch.float32)
    transl        = torch.from_numpy(smpl['root_transl'][0]).reshape(1, -1).to(torch.float32)
    betas        = torch.from_numpy(smpl['betas'][0]).reshape(1, 10).to(torch.float32)

    # Carica il modello SMPL
    smpl_model = SMPLX(
        model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
        gender='male',  # oppure 'male', 'female'
        batch_size=1
    )


    rotvec = global_orient.numpy().flatten()
    global_rotation = torch.from_numpy(Rot.from_rotvec(rotvec).as_matrix()).float()
    

    
    v = torch.Tensor([0, 0, 1])
    direction = global_rotation @ v
    direction = direction / np.linalg.norm(direction)
    print("Direzione:", direction)

    orientations = compute_global_orientations_smplx(global_rotation, body_pose.view(-1,3).numpy())
    

    
    output = smpl_model(
        global_orient=global_orient,
        body_pose=body_pose_raw,
        betas=betas,
        transl=transl,
        return_verts=True  
    )

    mask = [i for i in range(0, output.vertices[0].shape[0],3)]
    mask2 = [i for i in range(0, smpl_model.faces[0].shape[0],3)]

    verts = output.vertices[0].detach().cpu().numpy()[:] # (N_verts, 3)
    joints = output.joints[0].detach().cpu().numpy()  # (N_joints, 3)
    faces = smpl_model.faces  # (N_faces, 3)
    
    directions = []
    for ori in orientations:
        direction = ori @ v
        direction = direction / np.linalg.norm(direction)
        directions.append(direction)
        

    print("Vertices shape:", verts.shape)
    print("Faces shape:", faces.shape)
    
    visualize_mesh_and_joints(verts, joints, faces, directions)
    
    return joints, body_pose_raw, direction

load_simple(arr)