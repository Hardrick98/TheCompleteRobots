import numpy as np
import pinocchio as pin
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import torch
from smplx import SMPLX

class HumanoidRobot:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.robot = self.load_robot(urdf_path)
        self.q0 = self.robot.q0
        self.model = self.robot.model
        self.data = self.robot.data
        self.pose_dict, self.abs_pose = self.get_joints(self.q0)
        self.collision_model = self.robot.collision_model
        self.visual_model = self.robot.visual_model
        self.body, self.joints = self.get_frames()
        

          # Initial configuration

    def forward_kinematics(self, q):
        self.forwardK(self.robot, q)
    
    def load_robot(self,urdf_path): 

        robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=["."],
            root_joint=pin.JointModelSpherical() 
        )
        print(f"URDF description successfully loaded in {robot}")
        
        print(f"Number of DoF: {len(robot.model.joints)}")
        
        return robot

    def forwardK(robot, q):
        
        pin.forwardKinematics(robot.model, robot.data, q)
    
        print("Target pose reached")
    
    def get_frames(self):
        body = {}
        joints = {}
        for i, frame in enumerate(self.model.frames):
            if frame.type == pin.FrameType.BODY:
                body[frame.name] = i
            if frame.type == pin.FrameType.JOINT:
                joints[frame.name] = i
            
        return body, joints
    
    def get_links_positions(self, q0):
       pin.forwardKinematics(self.model, self.data, q0)
       pin.updateFramePlacements(self.model, self.data)
       links_positions = []
       for name, id in self.body.items():
        pose = self.data.oMf[id].translation
        links_positions.append(pose)   
       
       return np.array(links_positions) 
    
    def get_joints(self, q0):
        
        pin.forwardKinematics(self.model, self.data, q0)
        abs_pose = []
        pose_dict = {}
        for joint in self.model.joints:
            joint_id = joint.id
            if joint_id == 18446744073709551615:
                joint_id = 0
            joint_name = self.model.names[joint_id]
            pos = self.model.jointPlacements[joint_id]
            abs = self.data.oMi[joint_id]
            if joint_id != 0:
                abs_pose.append(abs.translation)
                pose_dict[joint_name] = joint_id-1
        
        abs_pose = np.array(abs_pose)
        
        return pose_dict, abs_pose
    
    def get_parent_child_pairs(self):
        

        pairs = []
        for joint_id in range(1, self.model.njoints):  # salta il giunto di root (0)
            parent_id = self.model.parents[joint_id]
            child_name = self.model.names[joint_id]
            parent_name =self.model.names[parent_id]
            if parent_name != "universe":
                pairs.append((self.pose_dict[parent_name], self.pose_dict[child_name]))
        
        return pairs
    
    def get_physical_joints(self):
        # Converti ogni riga in tupla per poterla usare come chiave
        
        
        arr_view = [tuple(row) for row in self.abs_pose[1:]]
        
        # Raggruppa gli indici
        groups = defaultdict(list)
        for idx, triplet in enumerate(arr_view):
            groups[triplet].append(idx)
        
        joints = []
        for key, indices in groups.items():
            joints.append(indices)
        index = {}
        for i in range(len(joints)):
            for k in joints[i]:
                index[k] = i
        
        limbs = self.get_parent_child_pairs()
        new_limbs = []
        for limb in limbs:
            if limb[0] in index and limb[1] in index:
                if index[limb[0]] != index[limb[1]] and index[limb[0]]!=0 :
                    new_limbs.append((index[limb[0]], index[limb[1]]))
    
        
        filtered_joints = [self.abs_pose[i[0]] for i in joints]
        
        
        return np.vstack(filtered_joints), new_limbs
    
def rotate_human(human_joints):
    
        shoulder_vec = human_joints[14] - human_joints[11]
        shoulder_vec /= np.linalg.norm(shoulder_vec)

        target_vec = np.array([0, 1, 0])
        axis = np.cross(shoulder_vec, target_vec)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis /= axis_norm
            angle = np.arccos(np.clip(np.dot(shoulder_vec, target_vec), -1.0, 1.0))
        else:
            axis = np.array([0, 0, 1])
            angle = 0.0
        rot = R.from_rotvec(axis * angle)
        human_joints = rot.apply(human_joints)
        
        return human_joints
    



def load_simple(arr, index=0):
    smpl = arr["smpl"][()]
    global_orient = torch.from_numpy(smpl['global_orient'][index]).reshape(1, -1).to(torch.float32)
    body_pose_raw = torch.from_numpy(smpl['body_pose'][index])
    transl        = torch.from_numpy(smpl['root_transl'][index]).reshape(1, -1).to(torch.float32)
    betas        = torch.from_numpy(smpl['betas'][index]).reshape(1, 10).to(torch.float32)

    # Carica il modello SMPL
    smpl_model = SMPLX(
        model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
        gender='male',  # oppure 'male', 'female'
        batch_size=1
    )
    
    
    body_pose = body_pose_raw[:21].reshape(1, -1).to(torch.float32)

    output = smpl_model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl=transl,
        return_verts=True  # Se vuoi solo i keypoints
    )


    verts = output.vertices[0].detach().cpu().numpy()  # (N_verts, 3)
    joints = output.joints[0].detach().cpu().numpy()  # (N_joints, 3)
    faces = smpl_model.faces   # (N_faces, 3)

    return joints, body_pose_raw.reshape(1, -1).to(torch.float32), transl.cpu().numpy(), global_orient.cpu()



import kornia

def compute_global_orientations_smplx(global_orient, body_pose, change_ref = False):
   

    def rodrigues_kornia(rvecs: torch.Tensor) -> torch.Tensor:
        return kornia.geometry.axis_angle_to_rotation_matrix(rvecs)[:, :3, :3]  # (N, 3, 3)
    

    
    local_rotations = rodrigues_kornia(torch.from_numpy(body_pose))  # (22, 3, 3)
    
    
    
    parents = [-1,  # 0: pelvis (root)
               0,   # 1: left_hip
               0,   # 2: right_hip  
               0,   # 3: spine1
               1,   # 4: left_knee
               2,   # 5: right_knee
               3,   # 6: spine2
               4,   # 7: left_ankle
               5,   # 8: right_ankle
               6,   # 9: spine3
               7,   # 10: left_foot
               8,   # 11: right_foot
               9,   # 12: neck
               9,   # 13: left_collar
               9,   # 14: right_collar
               12,  # 15: head
               13,  # 16: left_shoulder
               14,  # 17: right_shoulder
               16,  # 18: left_elbow
               17,  # 19: right_elbow
               18,  # 20: left_wrist
               19]  # 21: right_wrist
    
    global_orientations = torch.zeros((22,3,3))
    global_orientations[0] = global_orient
    for i in range(1,22):
        parent_idx = parents[i]
        global_orientations[i] = global_orientations[parent_idx] @ local_rotations[i-1].float()

    M = torch.tensor([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ], dtype=torch.float32)

    if change_ref:

        global_orientations = torch.stack([
            M @ R for R in global_orientations
        ])
        
    return global_orientations


def get_smplx_global_orientations(global_orient, body_pose_raw, change_ref=False):
    
    global_rot_matrices = compute_global_orientations_smplx(global_orient, body_pose_raw, change_ref)
    
    return global_rot_matrices


def scale_human_to_robot(R, F, robot_joints, H, human_joints, head_fixed = False):

     
    
    hipH = np.linalg.norm(human_joints[H["LHip"]]-human_joints[H["root_joint"]])
    hipR = np.linalg.norm(robot_joints[R["LHip"]]-robot_joints[R["root_joint"]])
    
    
    spineH = np.linalg.norm(human_joints[H["Neck"]]-human_joints[H["root_joint"]])
    spineR = np.linalg.norm(robot_joints[R["Head"]]-robot_joints[R["root_joint"]])
    
    if not head_fixed:
        shoulH = np.linalg.norm(human_joints[H["LShoulder"]]-human_joints[H["Neck"]])
        shoulR = np.linalg.norm(robot_joints[R["LShoulder"]]-robot_joints[R["Head"]])
    else:
        shoulH = np.linalg.norm(human_joints[H["LShoulder"]]-human_joints[H["LHip"]])
        shoulR = np.linalg.norm(robot_joints[R["LShoulder"]]-robot_joints[R["LHip"]])
    
    femorH = np.linalg.norm(human_joints[H["LKnee"]]-human_joints[H["LHip"]])
    femorR = np.linalg.norm(robot_joints[R["LKnee"]]-robot_joints[R["LHip"]])
    
    tibiaH = np.linalg.norm(human_joints[H["LAnkle"]]-human_joints[H["LKnee"]])
    tibiaR = np.linalg.norm(robot_joints[R["LAnkle"]]-robot_joints[R["LKnee"]])

    upper_armH = np.linalg.norm(human_joints[H["LElbow"]]-human_joints[H["LShoulder"]])
    upper_armR = np.linalg.norm(robot_joints[R["LElbow"]]-robot_joints[R["LShoulder"]])

    forearmH = np.linalg.norm(human_joints[H["LWrist"]]-human_joints[H["LElbow"]])
    forearmR = np.linalg.norm(robot_joints[R["LWrist"]]-robot_joints[R["LElbow"]])
    
    
    s_upper_arm = upper_armR / upper_armH
    s_forearm = forearmR / forearmH
    s_spine = spineR / spineH
    s_shoulder = shoulR / shoulH
    s_hip = hipR / hipH
    s_femor = femorR / femorH
    s_tibia = tibiaR / tibiaH

    #Scaling
    if not head_fixed:
        robot_joints[R["Head"]] = robot_joints[R["root_joint"]] + (human_joints[H["Neck"]] - human_joints[H["root_joint"]]) * s_spine
        robot_joints[R["LShoulder"]] = robot_joints[R["Head"]] + (human_joints[H["LShoulder"]] - human_joints[H["Neck"]]) * s_shoulder
        robot_joints[R["RShoulder"]] = robot_joints[R["Head"]] + (human_joints[H["RShoulder"]] - human_joints[H["Neck"]]) * s_shoulder
    else:
        robot_joints[R["Head"]] = 0
        robot_joints[R["LShoulder"]] = robot_joints[R["LHip"]] + (human_joints[H["LShoulder"]] - human_joints[H["LHip"]]) * s_shoulder
        robot_joints[R["RShoulder"]] = robot_joints[R["RHip"]] + (human_joints[H["RShoulder"]] - human_joints[H["RHip"]]) * s_shoulder
    
   
    robot_joints[R["LHip"]] = robot_joints[R["root_joint"]] + (human_joints[H["LHip"]] - human_joints[H["root_joint"]]) * s_hip
    robot_joints[R["LKnee"]] = robot_joints[R["LHip"]] + (human_joints[H["LKnee"]] - human_joints[H["LHip"]]) * s_femor
    robot_joints[R["LAnkle"]] = robot_joints[R["LKnee"]] + (human_joints[H["LAnkle"]] - human_joints[H["LKnee"]]) * s_tibia
    
    robot_joints[R["RHip"]] = robot_joints[R["root_joint"]] + (human_joints[H["RHip"]] - human_joints[H["root_joint"]]) * s_hip
    robot_joints[R["RKnee"]] = robot_joints[R["RHip"]] + (human_joints[H["RKnee"]] - human_joints[H["RHip"]]) * s_femor
    robot_joints[R["RAnkle"]] = robot_joints[R["RKnee"]] + (human_joints[H["RAnkle"]] - human_joints[H["RKnee"]]) * s_tibia

    robot_joints[R["LElbow"]] = robot_joints[R["LShoulder"]] + (human_joints[H["LElbow"]] - human_joints[H["LShoulder"]]) * s_upper_arm
    robot_joints[R["LWrist"]] = robot_joints[R["LElbow"]] + (human_joints[H["LWrist"]] - human_joints[H["LElbow"]]) * s_forearm
    robot_joints[R["RElbow"]] = robot_joints[R["RShoulder"]] + (human_joints[H["RElbow"]] - human_joints[H["RShoulder"]]) * s_upper_arm
    robot_joints[R["RWrist"]] = robot_joints[R["RElbow"]] + (human_joints[H["RWrist"]] - human_joints[H["RElbow"]]) * s_forearm
    
    return robot_joints

def pyplot_arrows(ax, directions, human_joints, H):
    
    v = torch.tensor([0,0,1])

    #rotation = global_orientations_matrices[15].float()
    #direction = rotation.float() @ v.float()
    #direction = direction / torch.linalg.norm(direction)
    
    direction = directions[15]
        
    ax.quiver(
        human_joints[H["Head"]][0], 
        human_joints[H["Head"]][1],
        human_joints[H["Head"]][2],                    
        direction[0],              
        direction[1],              
        direction[2],            
        length=1.0,                
        color='purple',
        normalize=True           
    )
    
    
    #v = torch.tensor([0,1,0])
    #rotation = global_orientations_matrices[21].float()
    #direction = rotation.float() @ v.float()
    #direction_RHand = direction / torch.linalg.norm(direction)
    
    direction = directions[21]
    
        
    ax.quiver(
        human_joints[H["RWrist"]][0], 
        human_joints[H["RWrist"]][1],
        human_joints[H["RWrist"]][2],
        direction[0],              
        direction[1],              
        direction[2],              
        length=1.0,                
        color='orange',
        normalize=True             
    )
    
    
    #rotation = global_orientations_matrices[H["LWrist"]].float()
    #direction = rotation.float() @ v.float()
    #direction_LHand = direction / torch.linalg.norm(direction)
    
    direction = directions[20]
        
    ax.quiver(
        human_joints[H["LWrist"]][0], 
        human_joints[H["LWrist"]][1],
        human_joints[H["LWrist"]][2],
        direction[0],              
        direction[1],              
        direction[2],              
        length=1.0,                
        color='purple',
        normalize=True             
    )

import torch
import kornia

def compute_global_orientations_batch(global_orient: torch.Tensor, body_pose: torch.Tensor, change_ref: bool = False):
    """
    Args:
        global_orient: (N, 3, 3) root orientation matrices
        body_pose: (N, 21, 3) axis-angle body poses (excluding root)
        change_ref: if True, apply a reference frame change
    Returns:
        global_orientations: (N, 22, 3, 3)
    """
    def rodrigues_kornia(rvecs: torch.Tensor) -> torch.Tensor:
        return kornia.geometry.axis_angle_to_rotation_matrix(rvecs)[:, :3, :3]  # (N * 21, 3, 3)

    N = body_pose.shape[0]

    # Compute local rotations from axis-angle to rotation matrices
    local_rotations = rodrigues_kornia(body_pose.reshape(-1, 3)).reshape(N, 21, 3, 3)

    parents = torch.tensor([
        -1,  # 0: pelvis (root)
         0,  # 1: left_hip
         0,  # 2: right_hip  
         0,  # 3: spine1
         1,  # 4: left_knee
         2,  # 5: right_knee
         3,  # 6: spine2
         4,  # 7: left_ankle
         5,  # 8: right_ankle
         6,  # 9: spine3
         7,  # 10: left_foot
         8,  # 11: right_foot
         9,  # 12: neck
         9,  # 13: left_collar
         9,  # 14: right_collar
        12,  # 15: head
        13,  # 16: left_shoulder
        14,  # 17: right_shoulder
        16,  # 18: left_elbow
        17,  # 19: right_elbow
        18,  # 20: left_wrist
        19   # 21: right_wrist
    ], dtype=torch.long)

    global_orientations = torch.zeros((N, 22, 3, 3), dtype=torch.float32, device=global_orient.device)
    global_orientations[:, 0] = global_orient

    for i in range(1, 22):
        parent = parents[i]
        global_orientations[:, i] = torch.matmul(global_orientations[:, parent], local_rotations[:, i - 1])

    if change_ref:
        M = torch.tensor([
            [-1, 0, 0],
            [ 0, 0, 1],
            [ 0, 1, 0]
        ], dtype=torch.float32, device=global_orient.device)
        global_orientations = M @ global_orientations  # (3, 3) @ (N, 22, 3, 3) via broadcasting


    return global_orientations
