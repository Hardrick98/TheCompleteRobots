import torch
from smplx import SMPL, SMPLX
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import argparse
from utils import compute_global_orientations_smplx
from vedo import Mesh, Points, show, Arrow






def visualize_mesh_and_joints_vedo(vertices, joints, faces, directions, title="SMPLX Mesh + Joints"):
    # Mesh
    mesh = Mesh([vertices, faces])
    mesh.c('lightblue').alpha(0.5).lw(0.5)

    # Joints
    joints_points = Points(joints, r=12, c='red')  # r=radius, c=color

    print(len(directions))
    # Frecce delle direzioni
    arrows = []
    for i in [0, 15, 20,21]:  # esempio: bacino, mani, testa
        start = joints[i]
        end = start + 0.2 * directions[i].numpy()  # Scala la freccia
        arrow = Arrow(start, end, c='blue', s=0.001)  # s = scala della freccia
        arrows.append(arrow)

    show(mesh, joints_points, *arrows, axes=1, title=title)



def load_simple(arr, idx):
    smpl = arr["smpl"][()]
    global_orient = torch.from_numpy(smpl['global_orient'][idx]).reshape(1, -1).to(torch.float32)
    body_pose_raw = torch.from_numpy(smpl['body_pose'][idx][:21]).reshape(1, -1).to(torch.float32)
    body_pose = torch.from_numpy(smpl['body_pose'][idx][:]).reshape(1, -1).to(torch.float32)
    transl        = torch.from_numpy(smpl['root_transl'][idx]).reshape(1, -1).to(torch.float32)
    betas        = torch.from_numpy(smpl['betas'][idx]).reshape(1, 10).to(torch.float32)

    # Carica il modello SMPL
    smpl_model = SMPLX(
        model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
        gender='male', 
        batch_size=1
    )


    rotvec = global_orient.numpy().flatten()
    global_rotation = torch.from_numpy(Rot.from_rotvec(rotvec).as_matrix()).float()
    

    
    v = torch.Tensor([0, 0, 1])
    direction = global_rotation @ v
    direction = direction / np.linalg.norm(direction)
   

    orientations = compute_global_orientations_smplx(global_rotation, body_pose.view(-1,3).numpy())
    

    
    output = smpl_model(
        global_orient=global_orient,
        body_pose=body_pose_raw,
        betas=betas,
        transl=transl,
        return_verts=True  
    )



    verts = output.vertices[0].detach().cpu().numpy()[:] 
    joints = output.joints[0].detach().cpu().numpy() 
    faces = smpl_model.faces
    
    directions = []
    for ori in orientations[:-3]:
        direction = ori @ v
        direction = direction / np.linalg.norm(direction)
        directions.append(direction)
    
    v = torch.Tensor([0, -1, 0])

    for ori in orientations[-3:]:
        direction = ori @ v
        direction = direction / np.linalg.norm(direction)
        directions.append(direction)
    
    mesh = Mesh([verts, faces])
    mesh.c('lightblue').alpha(0.5).lw(0.5)
    
    #visualize_mesh_and_joints_vedo(verts, joints, faces, directions)
    
    return joints, body_pose.reshape(1, -1).to(torch.float32), transl.cpu().numpy(), global_orient.cpu(), mesh



def load_simple_interx(arr, idx):
    
    smpl = arr
    global_orient = torch.from_numpy(smpl['root_orient'][idx]).reshape(1, -1).to(torch.float32)
    body_pose_raw = torch.from_numpy(smpl['pose_body'][idx][:21]).reshape(1, -1).to(torch.float32)
    body_pose = torch.from_numpy(smpl['pose_body'][idx][:]).reshape(1, -1).to(torch.float32)
    transl        = torch.from_numpy(smpl['trans'][idx]).reshape(1, -1).to(torch.float32)
    betas        = torch.from_numpy(smpl['betas'][idx]).reshape(1, 10).to(torch.float32)

    # Carica il modello SMPL
    smpl_model = SMPLX(
        model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
        gender='male', 
        batch_size=1
    )


    rotvec = global_orient.numpy().flatten()
    global_rotation = torch.from_numpy(Rot.from_rotvec(rotvec).as_matrix()).float()
    

    
    v = torch.Tensor([0, 0, 1])
    direction = global_rotation @ v
    direction = direction / np.linalg.norm(direction)
   

    orientations = compute_global_orientations_smplx(global_rotation, body_pose.view(-1,3).numpy())
    

    
    output = smpl_model(
        global_orient=global_orient,
        body_pose=body_pose_raw,
        betas=betas,
        transl=transl,
        return_verts=True  
    )



    verts = output.vertices[0].detach().cpu().numpy()[:] 
    joints = output.joints[0].detach().cpu().numpy() 
    faces = smpl_model.faces
    
    directions = []
    for ori in orientations[:-3]:
        direction = ori @ v
        direction = direction / np.linalg.norm(direction)
        directions.append(direction)
    
    v = torch.Tensor([0, -1, 0])

    for ori in orientations[-3:]:
        direction = ori @ v
        direction = direction / np.linalg.norm(direction)
        directions.append(direction)
    
    mesh = Mesh([verts, faces])
    mesh.c('lightblue').alpha(0.5).lw(0.5)
    
    #visualize_mesh_and_joints_vedo(verts, joints, faces, directions)
    
    return joints, body_pose.reshape(1, -1).to(torch.float32), transl.cpu().numpy(), global_orient.cpu(), mesh






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file","-f",type=str)
    args = parser.parse_args()
    arr = np.load(args.file, allow_pickle=True)


    load_simple_interx(arr, 0)