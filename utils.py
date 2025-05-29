import numpy as np
import pinocchio as pin
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

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