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
            root_joint=pin.JointModelFreeFlyer(),
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
            elif frame.type == pin.FrameType.JOINT:
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
    


def calculate_bone_rotation(parent_pos, child_pos, reference_vector=np.array([0, 1, 0])):
    """
    Calcola la matrice di rotazione per orientare un osso dalla posizione parent a child
    
    Args:
        parent_pos: posizione 3D del joint padre
        child_pos: posizione 3D del joint figlio  
        reference_vector: direzione di riferimento dell'osso nel modello (default: +Y)
    """
    # Vettore osso attuale
    bone_vector = child_pos - parent_pos
    bone_vector = bone_vector / np.linalg.norm(bone_vector)  # normalizza
    
    # Calcola la rotazione per allineare reference_vector con bone_vector
    # Usa la formula di Rodrigues o quaternioni
    
    # Metodo semplice con cross product
    v = np.cross(reference_vector, bone_vector)
    s = np.linalg.norm(v)
    c = np.dot(reference_vector, bone_vector)
    
    if s < 1e-6:  # Vettori paralleli
        if c > 0:
            return np.eye(3)  # Stessa direzione
        else:
            # Direzioni opposte - ruota di 180°
            return -np.eye(3)
    
    # Matrice di rotazione usando la formula di Rodrigues
    vx = np.array([[0, -v[2], v[1]], 
                   [v[2], 0, -v[0]], 
                   [-v[1], v[0], 0]])
    
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
    
    return R

def pose3d_to_rotations(joints_3d):
    """
    Converte una posa 3D Human3.6M in rotazioni per ogni joint
    
    Args:
        joints_3d: array (N, 3) con le posizioni 3D dei joint
        skeleton_hierarchy: dizionario che definisce parent-child relationships
    
    Returns:
        rotations: dizionario con le matrici di rotazione per ogni joint
    """
    rotations = {}
    
    # Definisci la gerarchia scheletrica di Human3.6M
    # (questo dipende dall'ordine specifico del tuo dataset)
    h36m_joints = [
        'hip',           # 0
        'right_hip',     # 1  
        'right_knee',    # 2
        'right_ankle',   # 3
        'left_hip',      # 4
        'left_knee',     # 5
        'left_ankle',    # 6
        'belly',         # 7
        'neck',        # 8
        'nose',     # 9
        'head',          # 10
        'left_shoulder', # 11
        'left_elbow',    # 12
        'left_wrist',    # 13
        'right_shoulder',# 14
        'right_elbow',   # 15
        'right_wrist'    # 16
    ]
    
    
    # Parent-child relationships per Human3.6M
    parent_child_pairs = [
        (0, 1),   # hip -> right_hip
        (1, 2),   # right_hip -> right_knee  
        (2, 3),   # right_knee -> right_ankle
        (0, 4),   # hip -> left_hip
        (4, 5),   # left_hip -> left_knee
        (5, 6),   # left_knee -> left_ankle
        (0, 7),   # hip -> spine
        (7, 8),   # spine -> thorax
        (8, 9),   # thorax -> neck_base
        (9, 10),  # neck_base -> head
        (8, 11),  # thorax -> left_shoulder
        (11, 12), # left_shoulder -> left_elbow
        (12, 13), # left_elbow -> left_wrist
        (8, 14),  # thorax -> right_shoulder
        (14, 15), # right_shoulder -> right_elbow
        (15, 16), # right_elbow -> right_wrist
    ]
    
    for parent_idx, child_idx in parent_child_pairs:
        parent_name = h36m_joints[parent_idx]
        child_name = h36m_joints[child_idx]
        
        parent_pos = joints_3d[parent_idx]
        child_pos = joints_3d[child_idx]
        
        # Calcola rotazione per questo osso
        rotation = calculate_bone_rotation(parent_pos, child_pos)
        rotations[parent_name] = rotation
    
    return rotations



def load_simple(arr):
    smpl = arr["smpl"][()]
    global_orient = torch.from_numpy(smpl['global_orient'][0]).reshape(1, -1).to(torch.float32)
    body_pose_raw = torch.from_numpy(smpl['body_pose'][0])
    transl        = torch.from_numpy(smpl['root_transl'][0]).reshape(1, -1).to(torch.float32)
    betas        = torch.from_numpy(smpl['betas'][0]).reshape(1, 10).to(torch.float32)

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

    return joints, body_pose_raw, transl.cpu().numpy()


"""
class Retargeting():
    
    def __init__(self):
        self.robot = HumanoidRobot("URDF/nao.urdf")
        self.robot.node_names = self.robot.kinematics.mjcf_data['node_names']
        self.robot.parent_indices = self.robot.kinematics.mjcf_data['parent_indices']
        self.robot.local_translation = self.robot.kinematics.mjcf_data['local_translation'].to(torch.float32).to(self.device)
        self.robot.local_rotation = self.robot.kinematics.mjcf_data['local_rotation'].to(torch.float32).to(self.device)
        self.robot.joints_range = self.robot.kinematics.mjcf_data['joints_range'].to(torch.float32).to(self.device)
        self.robot.num_joints = 27
        self.robot.num_bodies = 32

    def retarget(self):
        print('Total frames to retarget with:', self.data.num_frames)
        # initialize the data with the data input
        joint_pos = torch.zeros(1, 27, device=self.device, dtype=torch.float32, requires_grad=True)
        root_ori = torch.zeros(1, 3, device=self.device, dtype=torch.float32, requires_grad=True)
        root_trans = torch.zeros(1, 3, device=self.device, dtype=torch.float32, requires_grad=True)

        rotations = R.from_quat(self.data.root_orient).as_rotvec()
        root_ori_init = torch.from_numpy(rotations).to(self.device).to(torch.float32)
        root_trans_init = torch.from_numpy(self.data.root_translation).to(self.device).to(torch.float32)

        root_ori.data = root_ori_init
        root_trans.data = root_trans_init

        optimizer = torch.optim.Adam([joint_pos, root_trans], lr=0.005)

        for i in range(self.num_iterations):
            optimizer.zero_grad()   # Clear the gradients
            loss = self.loss(
                joint_pos=joint_pos,
                root_trans=root_trans,
                root_ori=root_ori
            )
            loss.backward()         # Compute gradients
            optimizer.step()        # Update parameters
            
            if i % 500 == 0:
                print(f"Iteration {i}: Loss = {loss.item()}, Loss per frame = {loss.item() / self.data.num_frames}")
        
        return joint_pos, root_trans, root_ori 

    def loss_smoothing(self, joint_pos):
        last = joint_pos[0: -2, :]
        this = joint_pos[1: -1, :]
        next = joint_pos[2:   , :]
        unsmooth = torch.abs(this - (last + next) * 0.5)
        return torch.sum(unsmooth)

    def loss(self, joint_pos, root_trans, root_ori, lamb=0.05):
        loss_smooth = self.loss_smoothing(joint_pos)
        loss_retarget = self.loss_retarget(joint_pos, root_trans, root_ori)
        return lamb * loss_smooth + loss_retarget



    def loss_retarget(self, joint_pos, root_trans, root_ori):
        '''
            keypoint: (batch_length, num_keypoints, 3)
            keypoint_gt: (batch_length, num_keypoints, 3)
        '''
        keypoint = self.forward(
            joint_pos=joint_pos,
            root_trans=root_trans,
            root_ori=root_ori
        )
        keypoint_gt = torch.from_numpy(self.data.keypoint_trans).to(self.device).to(torch.float32)

        error = torch.norm(keypoint - keypoint_gt, dim=-1, p=1)
        return torch.sum(error)

    def forward(self, joint_pos=None, root_trans=None, root_ori=None):
        '''
            Forward the kinematics model to get the keypoint translation
            Input: 
                joint_pos       (batch_length, num_joints)
            Output:
                keypoint_trans  (batch_length, num_keypoints, 3)
        '''
        joint_pos = self.set_clamp(joint_pos)
        

        # reshape the joint_pos to fit the input of the kinematics model
        # the output is the description of the rotations starting from the root body in axis-angle format
        pose_batch = self.joint_pos_to_pose_batch(joint_pos=joint_pos, root_ori=root_ori)

        output = self.robot.kinematics.fk_batch(
            pose=pose_batch,
            trans=root_trans.unsqueeze(0),
            convert_to_mat=True,
            return_full=False)
        
        output_trans = output['global_translation'][0, :, :, :]

        keypoint_trans = torch.zeros(batch_len, 12, 3, dtype=torch.float32, device=self.device)
        keypoint_trans[:, 0] = output_trans[:, 3]      # left_hip 
        keypoint_trans[:, 1] = output_trans[:, 4]      # left_knee
        keypoint_trans[:, 2] = output_trans[:, 5]      # left_ankle
        keypoint_trans[:, 3] = output_trans[:, 9]      # right_hip
        keypoint_trans[:, 4] = output_trans[:, 10]     # right_knee
        keypoint_trans[:, 5] = output_trans[:, 11]     # right_ankle
        keypoint_trans[:, 6] = output_trans[:, 16]     # left_shoulder
        keypoint_trans[:, 7] = output_trans[:, 19]     # left_elbow
        keypoint_trans[:, 8] = output_trans[:, 21]     # left_hand
        keypoint_trans[:, 9] = output_trans[:, 25]     # right_shoulder
        keypoint_trans[:, 10] = output_trans[:, 28]    # right_elbow
        keypoint_trans[:, 11] = output_trans[:, 30]    # right_hand        

        return keypoint_trans
"""