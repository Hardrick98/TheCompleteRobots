import numpy as np
import pinocchio as pin
from collections import defaultdict

class HumanoidRobot:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.robot = self.load_robot(urdf_path)
        self.q0 = self.robot.q0
        self.pose_dict, self.abs_pose = self.get_joints(self.robot, self.q0)
        self.model = self.robot.model
        self.data = self.robot.data
        self.collision_model = self.robot.collision_model
        self.visual_model = self.robot.visual_model
          # Initial configuration

    def forward_kinematics(self, q):
        self.forwardK(self.robot, q)

    def get_joints(self):
        return self.pose_dict
    
    def load_robot(self,urdf_path): 

        robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=["."],
            root_joint=None,
        )
        print(f"URDF description successfully loaded in {robot}")
        
        print(f"Number of DoF: {len(robot.model.joints)}")
        
        return robot

    def forwardK(robot, q):
        
        pin.forwardKinematics(robot.model, robot.data, q)
    
        print("Target pose reached")
    
    def get_joints(self,robot, q0):
        
        pin.forwardKinematics(robot.model, robot.data, q0)
        abs_pose = []
        pose_dict = {}
        for joint in robot.model.joints:
            joint_id = joint.id
            if joint_id == 18446744073709551615:
                joint_id = 0
            joint_name = robot.model.names[joint_id]
            pos = robot.model.jointPlacements[joint_id]
            abs = robot.data.oMi[joint_id]
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
        arr_view = [tuple(row) for row in self.abs_pose]
        
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
                if index[limb[0]] != index[limb[1]]:
                    new_limbs.append((index[limb[0]], index[limb[1]]))
        
        print(new_limbs)
        
        filtered_joints = [self.abs_pose[i[0]] for i in joints]
        
        
        return np.vstack(filtered_joints), new_limbs