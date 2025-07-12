import torch
from smplx import SMPL, SMPLX
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import argparse
from utils import compute_global_orientations_smplx, compute_global_orientations_batch
from vedo import Mesh, Points, show, Arrow, Plotter
from tqdm import tqdm





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



def load_simple_interx(smpl_model, arr, idx):
    
    device = "cuda:0"
    smpl = arr
    global_orient = torch.from_numpy(smpl['root_orient'][idx]).reshape(1, -1).to(torch.float32).to(device)
    body_pose_raw = torch.from_numpy(smpl['pose_body'][idx][:21]).reshape(1, -1).to(torch.float32).to(device)
    body_pose = torch.from_numpy(smpl['pose_body'][idx][:]).reshape(1, -1).to(torch.float32).to(device)
    transl        = torch.from_numpy(smpl['trans'][idx]).reshape(1, -1).to(torch.float32).to(device)
    betas        = torch.from_numpy(smpl['betas'][0]).reshape(1, 10).to(torch.float32).to(device)


    rotvec = global_orient.detach().cpu().numpy().flatten()
    global_rotation = torch.from_numpy(Rot.from_rotvec(rotvec).as_matrix()).float()
    

    
    v = torch.Tensor([0, 0, 1])
    direction = global_rotation @ v
    direction = direction / np.linalg.norm(direction)
   

    #orientations = compute_global_orientations_smplx(global_rotation, body_pose.cpu().view(-1,3).numpy())
    

    
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
    
    """
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
    """
    mesh = Mesh([verts, faces])
    #mesh.c('lightblue').alpha(0.5).lw(0.5)
    
    #visualize_mesh_and_joints_vedo(verts, joints, faces, directions)
    
    return joints, body_pose.cpu().reshape(1, -1).to(torch.float32), transl.cpu().numpy(), global_orient.cpu(), mesh,# directions


def load_simple_all(smpl_model, arr):
    
    device = "cuda:0"
    smpl = arr
    s = smpl['root_orient'].shape[0]
    global_orient = torch.from_numpy(smpl['root_orient']).reshape(s, 3).to(torch.float32).to(device)
    body_pose = torch.from_numpy(smpl['pose_body'][:]).reshape(s, -1, 3).to(torch.float32).to(device)
    transl        = torch.from_numpy(smpl['trans']).reshape(s, 3).to(torch.float32).to(device)
    betas        = torch.from_numpy(smpl['betas']).reshape(1, 10).repeat(s+8,1).to(torch.float32).to(device)


    rotvec = global_orient.detach().cpu().numpy()
    global_rotation = torch.from_numpy(Rot.from_rotvec(rotvec).as_matrix()).float().to(device).view(s,3,3)
    
    
    v1 = torch.Tensor([[0, 0, 1]]).repeat(s,1).transpose(1,0).to(device)
    v2 = torch.Tensor([[0, -1, 0]]).repeat(s,1).transpose(1,0).to(device)
    direction = global_rotation @ v1
    direction = direction / torch.linalg.norm(direction)
   

    orientations = compute_global_orientations_batch(global_rotation, body_pose)


    bs = 8
    pad = (bs - s % bs) % bs  # zero se s % bs == 0

    if pad > 0:
        global_orient = torch.cat([global_orient, torch.zeros((pad, 3), device=device)], dim=0)
        body_pose = torch.cat([body_pose, torch.zeros((pad, body_pose.shape[1], 3), device=device)], dim=0)
        transl = torch.cat([transl, torch.zeros((pad, 3), device=device)], dim=0)
        betas = torch.cat([betas, torch.zeros((pad, betas.shape[1]), device=device)], dim=0)

    vertices = []
    joints = []

    for i in range(0, global_orient.shape[0], bs):
        output = smpl_model(
            global_orient=global_orient[i:i+bs],
            body_pose=body_pose[i:i+bs],
            betas=betas[i:i+bs],
            transl=transl[i:i+bs],
            return_verts=True  
        )
        vertices.extend(output.vertices)  # estendi con tutti i batch
        joints.extend(output.joints)

    vertices = vertices[:s]
    joints = torch.stack(joints)[:s]

    faces = smpl_model.faces
    
    v1 = torch.Tensor([[0, 0, 1]]).transpose(1,0).to(device)
    v2 = torch.Tensor([[0, -1, 0]]).transpose(1,0).to(device)
    directions = []
    for ori2 in orientations:
        dir = []
        for ori in ori2[:-3]:
            direction = ori @ v1
            direction = direction / torch.linalg.norm(direction)
            dir.append(direction)
        

        for ori in ori2[-3:]:
            direction = ori @ v2
            direction = direction / torch.linalg.norm(direction)
            dir.append(direction)
        
        dir = torch.stack(dir)
        directions.append(dir)
    
    directions = torch.stack(directions).squeeze(-1)
    
    

    meshes = []
    
    for i in range(len(vertices)):
        meshes.append(Mesh([vertices[i].detach().cpu().numpy(), faces]))
    
    
    return joints, orientations, transl, global_orient, meshes, directions

import time

from vedo import Video

def animate_all_poses(pose1, pose2, delay = 0.01):
    vp = Plotter(interactive=False, axes=1, title="SMPLX Animation", bg='white')

    meshes1 = [m.clone().c('blue') for m in tqdm(pose1)]
    meshes2 = [m.clone().c('red') for m in tqdm(pose2)]
    framerate = 24
    video = Video(name="video.mp4", duration=len(pose1)/framerate, fps=framerate)
    vp.show(meshes1[0], meshes2[0], resetcam=True)
    
    vp.camera.SetPosition([5, 1, 1])        
    vp.camera.SetFocalPoint([0, 0, 0])      
    vp.camera.SetViewUp([0, 0, 1]) 
    vp.render()

    for i in range(1, len(meshes1)):
        # Usa `vp.remove()` per rimuovere solo i mesh vecchi
        vp.remove(meshes1[i-1], meshes2[i-1])

        vp.add(meshes1[i], meshes2[i])
        vp.render()
        video.add_frame()
        time.sleep(delay)
        

    video.close()
    vp.interactive().close()


if __name__ == "__main__":
    


    parser = argparse.ArgumentParser()
    parser.add_argument("--file","-f",type=str)
    args = parser.parse_args()
    arr = np.load(args.file, allow_pickle=True)
    arr2 = np.load("P2.npz")

    device = "cuda:0"
    #load_simple_interx(arr, 0)
    smpl_model = SMPLX(
        model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
        gender='male', 
        use_pca=False,
        batch_size=8
    ).to(device)
    
    joints, body_pose, transl, global_orient, pose1, directions = load_simple_all(smpl_model, arr)
    joints, body_pose, transl, global_orient, pose2, directions = load_simple_all(smpl_model, arr2)
    
    animate_all_poses(pose1, pose2)
    