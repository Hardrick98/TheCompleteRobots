import torch
from smplx import SMPL, SMPLX
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as Rot
arr = np.load("ab_exercises_ranked_best_to_worst_clip_1.npz", allow_pickle=True)







index_keypoints = [
    1, 4, 7,            # left hip, knee, ankle
    2, 5, 8,            # right hip, knee, ankle
    16, 18, 20,         # left shoulder, elbow, wrist
    17, 19, 21          # right shoulder, elbow, wrist
]



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_mesh_and_joints(vertices, joints, faces, direction, title="SMPLX Mesh + Joints"):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Mesh
    mesh = Poly3DCollection(vertices[faces], alpha=0.5)
    mesh.set_facecolor((0.8, 0.8, 1))
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    # Joints
    #ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=40)
    ax.quiver(
        joints[0,0], joints[0,1],joints[0,2],                    # Punto di origine
        direction[0],              # Componente x della direzione
        direction[1],              # Componente y della direzione
        direction[2],              # Componente z della direzione
        length=2.0,                # Lunghezza della freccia
        color='yellow',
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
    transl        = torch.from_numpy(smpl['root_transl'][0]).reshape(1, -1).to(torch.float32)
    betas        = torch.from_numpy(smpl['betas'][0]).reshape(1, 10).to(torch.float32)

    # Carica il modello SMPL
    smpl_model = SMPLX(
        model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
        gender='male',  # oppure 'male', 'female'
        batch_size=1
    )

    print("Global Orient:", global_orient)
    rotvec = global_orient.numpy().flatten()
    rotation = Rot.from_rotvec(rotvec)
    initial_vector = np.array([0, 0, 1])
    direction = rotation.apply(initial_vector)
    direction = direction / np.linalg.norm(direction)
    print("Direzione:", direction)


    
    output = smpl_model(
        global_orient=global_orient,
        body_pose=body_pose_raw,
        betas=betas,
        transl=transl,
        return_verts=True  # Se vuoi solo i keypoints
    )


    verts = output.vertices[0].detach().cpu().numpy()  # (N_verts, 3)
    joints = output.joints[0].detach().cpu().numpy()  # (N_joints, 3)
    faces = smpl_model.faces   # (N_faces, 3)
    
    visualize_mesh_and_joints(verts, joints, faces, direction)
    
    return joints, body_pose_raw, direction

load_simple(arr)