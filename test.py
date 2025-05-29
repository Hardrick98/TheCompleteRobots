import numpy as np

arr = np.load("/datasets/HumanoidX/human_pose/youtube/red_dot_scary_maze_prank_on_my_son_for_his_reaction_shorts_clip_1.npz", allow_pickle=True)

smpl = arr["smpl"][()]


print(smpl["body_pose"].shape) #(77,23,3)
print(smpl["global_orient"].shape) # (77,3)
print(smpl["betas"].shape) # (77,10)
print(smpl["root_transl"].shape) # (77,3)


index_keypoints = [
    1, 4, 7,            # left hip, knee, ankle
    2, 5, 8,            # right hip, knee, ankle
    16, 18, 20,         # left shoulder, elbow, wrist
    17, 19, 21          # right shoulder, elbow, wrist
]


import torch
import smplx

global_orient = torch.from_numpy(smpl['global_orient'][0]).reshape(1, -1).to(torch.float32)
body_pose_raw = torch.from_numpy(smpl['body_pose'][0]).reshape(1, -1).to(torch.float32)
transl        = torch.from_numpy(smpl['root_transl'][0]).reshape(1, -1).to(torch.float32)
betas        = torch.from_numpy(smpl['betas'][0]).reshape(1, 10).to(torch.float32)

model = smplx.SMPL(model_path='SMPLX_NEUTRAL.npz', batch_size=1, device='cpu')
output = model(
    body_pose=body_pose_raw, 
    global_orient=global_orient,
    betas=betas,
    transl=transl
)
joints_3d = output.joints 
print(joints_3d.shape)  # (77, 22, 3)