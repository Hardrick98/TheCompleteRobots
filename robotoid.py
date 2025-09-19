from tqdm import tqdm
import numpy as np
from inverse_kinematics import InverseKinematicSolver
import os
from utils import compute_global_orientations_batch
from smplx import SMPLX
import torch
import pinocchio as pin
import trimesh
from vedo import Mesh
from scipy.spatial.transform import Rotation as Rot

class Robotoid:

    def __init__(self, robot, wheeled = False):
        
        self.pose_dict, self.robot_joints = robot.get_joints(robot.q0)
        #self.robot_joints, self.robot_limbs = robot.get_physical_joints()
        self.robot = robot
        self.q0 = robot.q0
        self.model = robot.model
        self.data = robot.data
        self.visual_model = robot.visual_model
        self.collision_model = robot.collision_model
        self.solver = InverseKinematicSolver(self.model,self.data)
        self.wheeled = wheeled
        self.N, self.J = self.build()
        self.links_positions = self.robot.get_links_positions(self.q0)
        self.cL, self.cR = self.find_palm_convention()
        self.head_fixed = False
        if "Head" not in self.N:
            self.head_fixed = True 
            self.J["Head"] = self.J["root_joint"]
            self.N["Head"] = self.N["root_joint"] 
        


    def find_palm_convention(self):
        """
        Determines the palm orientation convention used in the retargeting function.

        Returns:
            A tuple of two lists (cL, cR) representing direction vectors in the form [x', y', z'].
        """

        frame_id = self.model.getFrameId(self.N["RWrist"])
        frame = self.model.frames[frame_id]
        root_link_id = frame.parentJoint  

        desc_joints = []
        for j in range(self.model.njoints):
            cur = j
            while cur != 0:
                cur = self.model.parents[cur]
                if cur == root_link_id:
                    desc_joints.append(j)
                    break


        desc_joints = list(set(desc_joints + [root_link_id]))

        meshes = []
        for geom in self.visual_model.geometryObjects:
            if geom.parentJoint in desc_joints:
                meshes.append({
                    "joint": self.model.names[geom.parentJoint],
                    "meshPath": geom.meshPath,
                    "scale": geom.meshScale,
                    "frame_parent": geom.parentFrame
                })
        all_meshes = []

        
        for geom in self.visual_model.geometryObjects:
            if geom.parentJoint in desc_joints:
                mesh_path = geom.meshPath
                if not os.path.isfile(mesh_path):
                    print(f"File not found: {mesh_path}")
                    continue
                mesh = trimesh.load(mesh_path)
                placement = self.data.oMf[geom.parentFrame]
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
            x, y, z = size
            
            dir = np.argmin(np.array([x,y,z]))

            if dir == 0:
                print("Palm direction is x")
                cL = [1,0,0]
                cR = [1,0,0]
            elif dir == 1:
                print("Palm direction is y")
                cL = [0,-1,0]
                cR = [0,1,0]
            elif dir == 2:
                print("Palm direction is z")
                cL = [0,0,-1]
                cR = [0,0,-1]
            #combined.show()  # oppure: combined.export("combined_mesh.obj")
            return cL, cR
        else:
            print("Nessuna mesh trovata")
            return None, None

    def get_kinematic_chains(self,end_effectors, parent_child):
            """
            Returns:
                A list of kinematic chains
            """
            chains = []
            for ee in end_effectors:
                chain = []
                chain.append(ee)
                parent = parent_child[ee]
                while parent != "root":
                    chain.append(parent)
                    ee = parent
                    parent = parent_child[ee]
                chains.append(chain[::-1])
            
            return chains

    

    def longest_common_prefix(self,lists):
        if not lists:
            return []
        prefix = lists[0]
        for lst in lists[1:]:
            i = 0
            while i < len(prefix) and i < len(lst) and prefix[i] == lst[i]:
                i += 1
            prefix = prefix[:i]
        return prefix

    def reduce_lists(self, input_lists, max_groups=5):
        reduced = []
        used = [False] * len(input_lists)
        groups = []

        for i in range(len(input_lists)):
            if used[i]:
                continue
            group = [input_lists[i]]
            used[i] = True
            for j in range(i + 1, len(input_lists)):
                if not used[j]:
                    prefix = self.longest_common_prefix([input_lists[i], input_lists[j]])
                    if prefix and len(prefix) >= 3:
                        group.append(input_lists[j])
                        used[j] = True
            reduced.append(self.longest_common_prefix(group))
            groups.append(group)

        return reduced[:max_groups], groups
    

    def scale_human_to_robot(self, R, robot_joints, H, human_joints):
        
        """
        Scales and moves the robot joints to fit the human ones in the robot.

        Returns:
            A numpy array containing the robot joint target positions
        """
        
        spineH = np.linalg.norm(human_joints[H["Neck"]]-human_joints[H["root_joint"]])
        spineR = np.linalg.norm(robot_joints[R["Head"]]-robot_joints[R["root_joint"]])
        
        if not self.head_fixed or self.wheeled:
            shoulH = np.linalg.norm(human_joints[H["LShoulder"]]-human_joints[H["Neck"]])
            shoulR = np.linalg.norm(robot_joints[R["LShoulder"]]-robot_joints[R["Head"]])
        else:
            shoulH = np.linalg.norm(human_joints[H["LShoulder"]]-human_joints[H["LHip"]])
            shoulR = np.linalg.norm(robot_joints[R["LShoulder"]]-robot_joints[R["LHip"]])
        
        if not self.wheeled:
            
            hipH = np.linalg.norm(human_joints[H["LHip"]]-human_joints[H["root_joint"]])
            hipR = np.linalg.norm(robot_joints[R["LHip"]]-robot_joints[R["root_joint"]])

            femorH = np.linalg.norm(human_joints[H["LKnee"]]-human_joints[H["LHip"]])
            femorR = np.linalg.norm(robot_joints[R["LKnee"]]-robot_joints[R["LHip"]])
            
            tibiaH = np.linalg.norm(human_joints[H["LAnkle"]]-human_joints[H["LKnee"]])
            tibiaR = np.linalg.norm(robot_joints[R["LAnkle"]]-robot_joints[R["LKnee"]])

            s_hip = hipR / hipH
            s_femor = femorR / femorH
            s_tibia = tibiaR / tibiaH
        else:
            
            HHip = (human_joints[H["LHip"]] + human_joints[H["RHip"]])/2
            HKnee = (human_joints[H["LAnkle"]] + human_joints[H["RAnkle"]])/2
            hipH = np.linalg.norm(HHip-human_joints[H["root_joint"]])
            hipR = np.linalg.norm(robot_joints[R["Hip"]]-robot_joints[R["root_joint"]])

            femorH = np.linalg.norm(HKnee-HHip)
            femorR = np.linalg.norm(robot_joints[R["Knee"]]-robot_joints[R["Hip"]])
         
            
            s_hip = hipR / hipH
            s_femor = femorR / femorH
        

        upper_armH = np.linalg.norm(human_joints[H["LElbow"]]-human_joints[H["LShoulder"]])
        upper_armR = np.linalg.norm(robot_joints[R["LElbow"]]-robot_joints[R["LShoulder"]])

        forearmH = np.linalg.norm(human_joints[H["LWrist"]]-human_joints[H["LElbow"]])
        forearmR = np.linalg.norm(robot_joints[R["LWrist"]]-robot_joints[R["LElbow"]])
        
        
        s_upper_arm = upper_armR / upper_armH
        s_forearm = forearmR / forearmH
        s_spine = spineR / spineH
        s_shoulder = shoulR / shoulH


        #Scaling
        if not self.head_fixed:
            robot_joints[R["Head"]] = robot_joints[R["root_joint"]] + (human_joints[H["Neck"]] - human_joints[H["root_joint"]]) * s_spine
            robot_joints[R["LShoulder"]] = robot_joints[R["Head"]] + (human_joints[H["LShoulder"]] - human_joints[H["Neck"]]) * s_shoulder
            robot_joints[R["RShoulder"]] = robot_joints[R["Head"]] + (human_joints[H["RShoulder"]] - human_joints[H["Neck"]]) * s_shoulder
        else:
            robot_joints[R["Head"]] = 0
            robot_joints[R["LShoulder"]] = robot_joints[R["LHip"]] + (human_joints[H["LShoulder"]] - human_joints[H["LHip"]]) * s_shoulder
            robot_joints[R["RShoulder"]] = robot_joints[R["RHip"]] + (human_joints[H["RShoulder"]] - human_joints[H["RHip"]]) * s_shoulder
        
        
        if not self.wheeled:
            robot_joints[R["LHip"]] = robot_joints[R["root_joint"]] + (human_joints[H["LHip"]] - human_joints[H["root_joint"]]) * s_hip
            robot_joints[R["LKnee"]] = robot_joints[R["LHip"]] + (human_joints[H["LKnee"]] - human_joints[H["LHip"]]) * s_femor
            robot_joints[R["LAnkle"]] = robot_joints[R["LKnee"]] + (human_joints[H["LAnkle"]] - human_joints[H["LKnee"]]) * s_tibia
            robot_joints[R["RHip"]] = robot_joints[R["root_joint"]] + (human_joints[H["RHip"]] - human_joints[H["root_joint"]]) * s_hip
            robot_joints[R["RKnee"]] = robot_joints[R["RHip"]] + (human_joints[H["RKnee"]] - human_joints[H["RHip"]]) * s_femor
            robot_joints[R["RAnkle"]] = robot_joints[R["RKnee"]] + (human_joints[H["RAnkle"]] - human_joints[H["RKnee"]]) * s_tibia
        else:
            robot_joints[R["Knee"]] = robot_joints[R["Hip"]] + (HKnee - HHip) * s_femor
            robot_joints[R["Hip"]] = robot_joints[R["root_joint"]] + (HHip- human_joints[H["root_joint"]]) * s_hip


        robot_joints[R["LElbow"]] = robot_joints[R["LShoulder"]] + (human_joints[H["LElbow"]] - human_joints[H["LShoulder"]]) * s_upper_arm
        robot_joints[R["LWrist"]] = robot_joints[R["LElbow"]] + (human_joints[H["LWrist"]] - human_joints[H["LElbow"]]) * s_forearm
        
        robot_joints[R["RElbow"]] = robot_joints[R["RShoulder"]] + (human_joints[H["RElbow"]] - human_joints[H["RShoulder"]]) * s_upper_arm
        robot_joints[R["RWrist"]] = robot_joints[R["RElbow"]] + (human_joints[H["RWrist"]] - human_joints[H["RElbow"]]) * s_forearm
        
        return robot_joints


    
    def build(self):

        """
        Given the URDF file it builds the robotoid by associating each joint to the abstract structure.

        Returns:
            A robotoid
        """

        parent_child = {}
        for i, joint_name in enumerate(self.model.names):
            parent_idx = self.model.parents[i]
            if parent_idx == 0 and i != 0:
                parent_name = "root"
            else:
                parent_name = self.model.names[parent_idx]
            parent_child[joint_name] = parent_name

        print(parent_child)

        all_parents = set(self.model.parents)

        #GET END-EFFECTORS (JOINTS WITH NO CHILDREN)

        end_effectors = []
        for i, joint_name in enumerate(self.model.names):
            if i not in all_parents or i == 0:
                if i != 0:
                    end_effectors.append(joint_name)

        
        chains = self.get_kinematic_chains(end_effectors, parent_child)
        new_chains = []
        for chain in chains:
            new_chains.append([self.model.getJointId(joint) for joint in chain if self.model.getJointId(joint)!=1])

        
        #FIND KINEMATIC CHAINS
        if len(new_chains) > 5: #è probabile che ci siano più end-effector del necessario quindi riduci
            new_chains, groups = self.reduce_lists(new_chains, max_groups=5)
        

        ## HANDLING DUPLICATE ORIGIN JOINTS DIFFERENT FROM ROOT (ATLAS CASE)

        filter = []
        for chain in new_chains:
            for i in range(len(chain)):
                filter.append(chain[i])

        values_to_remove = set()
        for chain in new_chains:
            for i in range(len(chain)):
                if filter.count(chain[i]) > 1:
                    values_to_remove.add(chain[i])
                
        for chain in new_chains:
            for v in values_to_remove:
                if v in chain:
                    chain.remove(v)
        

        positions = []
        for chain in new_chains:
            positions.append(self.data.oMi[chain[0]].translation)

        positions = np.array(positions)
            
        #PULIRE CATENE INUTILI

        positions = positions[positions[:, 2].argsort()[::-1]]


        chains_new = []

        for chain in new_chains:
            positions = {}
            for j in (chain):
                positions[self.model.names[j]] = self.data.oMi[j].translation
            chains_new.append(positions)

        if len(chains_new) == 4:
            head = [{}]
            chains_new = head + chains_new
        
        ##CLUSTERIZE TO UNDERSTAND WHICH DEGREE OF FREEDOM IS RESPONSIBLE FOR A SPECIFIC LIMB

        robotoid = {}
        robotoid_labels = {}

        chain_value = 0
        for positions in chains_new:
            
            
            filtered_data = {k: v for k, v in positions.items()}

            keys = list(filtered_data.keys())
            points = np.array(list(filtered_data.values()))

            from sklearn.cluster import KMeans
            
            if points.shape[0] > 3:
                kmeans = KMeans(n_clusters=3, random_state=0).fit(points)
            else:
                if len(points) == 0:
                    if self.wheeled == True: 
                        robotoid[chain_value] = np.array([[-100,-100,-100]])
                        robotoid_labels[chain_value] = [[None]]
                        chain_value += 1
                    else:
                        print("Head not movable")
                        robotoid[chain_value] = np.array([[100,100,100]])
                        robotoid_labels[chain_value] = [[None]]
                        chain_value += 1
                    continue
                kmeans = KMeans(n_clusters=1,random_state=0).fit(points)

            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

            clusters = {}
            for label, key in zip(labels, keys):
                clusters.setdefault(label, []).append(key)
            

            centers = []
            centers_labels = []
            
            for i, c in enumerate(centroids):
                centers.append(c)
                centers_labels.append(clusters[i])
            
            robotoid[chain_value] = np.array(centers)
            robotoid_labels[chain_value] = centers_labels
                    
            chain_value += 1
            
        order = []
        for j in new_chains:
            for i in j:
                order.append(i)
        for chain_id in robotoid:
            centers = robotoid[chain_id]
            labels = robotoid_labels[chain_id]
            
            if labels[0][0] is None:
                continue

            sorted_indices = sorted(range(len(labels)), key=lambda i: order.index(self.model.getJointId(labels[i][0])))

            robotoid[chain_id] = centers[sorted_indices]
            robotoid_labels[chain_id] = [labels[i] for i in sorted_indices]



        



        sorted_chain_ids = sorted(robotoid.keys(), key=lambda i: robotoid[i][0, 2], reverse=True)

 

        robotoid = {i: robotoid[i] for i in sorted_chain_ids}
        robotoid_labels = {i: robotoid_labels[i] for i in sorted_chain_ids}

        new_robotoid = {i: v for i, (_, v) in enumerate(robotoid.items())}
        new_robotoid_labels = {i: v for i, (_, v) in enumerate(robotoid_labels.items())}
        


        robotoid = new_robotoid
        robotoid_labels = new_robotoid_labels 

        print("\n")
        print("Defined Chains:\n", robotoid_labels)
        print("\n")

        final = {}

        final["Head"] = robotoid_labels[0][0]

        if robotoid[1][0,1] > 0:
            final["LShoulder"] = robotoid_labels[1][0]
            final["LElbow"] = robotoid_labels[1][1]
            final["LWrist"] = robotoid_labels[1][2]
            final["RShoulder"] = robotoid_labels[2][0]
            final["RElbow"] = robotoid_labels[2][1]
            final["RWrist"] = robotoid_labels[2][2]
        else:
            final["LShoulder"] = robotoid_labels[2][0]
            final["LElbow"] = robotoid_labels[2][1]
            final["LWrist"] = robotoid_labels[2][2]
            final["RShoulder"] = robotoid_labels[1][0]
            final["RElbow"] = robotoid_labels[1][1]
            final["RWrist"] = robotoid_labels[1][2]
        
        if robotoid_labels[4][0][0] is not None:
            if robotoid[3][0,1] > 0:
                final["LHip"] = robotoid_labels[3][0]
                final["LKnee"] = robotoid_labels[3][1]
                final["LAnkle"] = robotoid_labels[3][2]
                final["RHip"] = robotoid_labels[4][0]
                final["RKnee"] = robotoid_labels[4][1]
                final["RAnkle"] = robotoid_labels[4][2]
            else:
                final["LHip"] = robotoid_labels[4][0]
                final["LKnee"] = robotoid_labels[4][1]
                final["LAnkle"] = robotoid_labels[4][2]
                final["RHip"] = robotoid_labels[3][0]
                final["RKnee"] = robotoid_labels[3][1]
                final["RAnkle"] = robotoid_labels[3][2]
        else:
            
         
            final["Hip"] = robotoid_labels[3][0][0:2]
            final["Knee"]  = [robotoid_labels[3][0][2]]

        final_reduced = {}
        final_values = {}
        for k,v in final.items():
            if v[0] is not None:
                final_reduced[k] = v[0]
                final_values[k] = self.model.getJointId(v[0])-1


        final_reduced["root_joint"] = "root_joint"
        final_values["root_joint"] = self.model.getJointId("root_joint") - 1

        return final_reduced, final_values

    
    def retarget(self, human_action, idx=None):
        """
        Given a human action this function retargets it to the robot.
        
        Returns:
            A numpy array with all the joint configurations for the action
        """  

        human_joints_seq, orientations_seq, translation_seq, global_orient_seq, _, directions_seq = human_action.get_attributes(idx)  
        H = human_action.get_joint_dict()


        print("\nFINDING CONFIGURATIONS...")

        joint_configurations = []
        sequence_num = human_joints_seq.shape[0]
        #sequence_num = 100

        for i in tqdm(range(sequence_num)):
        
            human_joints = human_joints_seq[i:i+1][0]
            orientations = orientations_seq[i:i+1][0]
            translation = translation_seq[i:i+1].copy()
            global_orient = global_orient_seq[i:i+1]
            directions = directions_seq[i]


            translation[:,[1,2]] = translation[:,[2,1]]

            orientations = orientations.view(-1,3) 
            orientations = torch.cat((global_orient.view(-1,3),orientations),axis=0)
                    
            directions = directions.detach().cpu().numpy()

        
            human_joints[:,:] -= human_joints[:1,:]
            human_joints[:,0] *= -1
            human_joints[:,[1,2]] = human_joints[:,[2,1]]
            directions[:,0] *= -1
            directions[:,[1,2]] = directions[:,[2,1]]


            
            self.robot_joints = self.scale_human_to_robot(self.J, self.robot_joints, H, human_joints)        
            
            joints = self.robot.joints
            
    

            if not self.wheeled:
                target_positions = {
                    self.N["LHip"] : self.robot_joints[self.J["LHip"]],
                    self.N["RHip"] : self.robot_joints[self.J["RHip"]], 
                    self.N["LElbow"]: self.robot_joints[self.J["LElbow"]],
                    self.N["RElbow"]: self.robot_joints[self.J["RElbow"]],
                    self.N["LWrist"]: self.robot_joints[self.J["LWrist"]], 
                    self.N["RWrist"]: self.robot_joints[self.J["RWrist"]], 
                    self.N["RKnee"]: self.robot_joints[self.J["RKnee"]], 
                    self.N["LKnee"]: self.robot_joints[self.J["LKnee"]], 
                    self.N["LAnkle"]: self.robot_joints[self.J["LAnkle"]], 
                    self.N["RAnkle"]: self.robot_joints[self.J["RAnkle"]], 
                    self.N["RShoulder"] : self.robot_joints[self.J["RShoulder"]],
                    self.N["LShoulder"] : self.robot_joints[self.J["LShoulder"]],
                    self.N["Head"]: self.robot_joints[self.J["Head"]],
                }
            else:
                
                target_positions = {
                    self.N["LElbow"]: self.robot_joints[self.J["LElbow"]],
                    self.N["RElbow"]: self.robot_joints[self.J["RElbow"]],
                    self.N["LWrist"]: self.robot_joints[self.J["LWrist"]], 
                    self.N["RWrist"]: self.robot_joints[self.J["RWrist"]], 
                    self.N["RShoulder"] : self.robot_joints[self.J["RShoulder"]],
                    self.N["LShoulder"] : self.robot_joints[self.J["LShoulder"]],
                    self.N["Head"]: self.robot_joints[self.J["Head"]],
                    self.N["Knee"]: self.robot_joints[self.J["Knee"]],
                    self.N["Hip"] : self.robot_joints[self.J["Hip"]]

                       
                    }
                
            
            if self.head_fixed:
                target_positions.pop(self.N["Head"])
            

            joint_names = [k for k in target_positions.keys()]
            joint_ids = [joints[name]for name in joint_names]

                

            target_orientations_global  = {
                self.N["RWrist"]: [directions[H["RWrist"]], self.cR], 
                self.N["LWrist"]: [directions[H["LWrist"]], self.cL],
                #F["LAnkle"]: [directions[H["LAnkle"]], [1,0,0]],
                #F["Head"]: [directions[H["Head"]], [1,0,0]]
    }
            
            frame_names = [k for k,v in target_orientations_global.items()]
            frame_ids = [self.model.getFrameId(f) for f in frame_names]

            
            self.solver.update(self.model,self.data,target_positions,target_orientations_global,joint_names, joint_ids, frame_names, frame_ids)
            
            if i==0:
                q1 = self.solver.inverse_kinematics(self.q0)
            else:
                q1 = self.solver.inverse_kinematics(q1)
            
            joint_configurations.append(q1)
            
            pin.forwardKinematics(self.model, self.data, q1)
            pin.updateFramePlacements(self.model, self.data)


        joint_configurations = np.vstack(joint_configurations)

        return joint_configurations
    






class HumanAction():

    def __init__(self, arr):
        

        self.device = "cuda:0"
        self.H = {
                "root_joint":0,
                "LHip":1,
                "RHip":2,
                "spine1":3,
                "LKnee":4,
                "RKnee":5,
                "spine2":6,
                "LAnkle":7,
                "RAnkle":8,
                "spine3":9,
                "left_foot":10,
                "right_foot":11,
                "Neck":12,
                "left_collar":13,
                "right_collar":14,
                "Head":15,
                "LShoulder":16,
                "RShoulder":17,
                "LElbow":18,
                "RElbow":19,
                "LWrist":20,
                "RWrist":21} 
         
        self.smpl_model = SMPLX(
            model_path='models_smplx_v1_1/models/smplx/SMPLX_MALE.npz',  # Deve contenere i file .pkl del modello
            gender='male', 
            use_pca=False,
            batch_size=8
        ).to(self.device)

        joint_positions, orientations, translation, global_orient, self.human_meshes, self.directions_seq = self.load_simple_all(arr)    
        self.human_joints_seq = joint_positions.detach().cpu().numpy()
        self.orientations_seq = orientations.detach().cpu()
        self.translation_seq = translation.detach().cpu().numpy()
        self.global_orient_seq = global_orient.detach().cpu()

    def get_attributes(self, idx=None):
        if idx is None:
            return self.human_joints_seq, self.orientations_seq, self.translation_seq, self.global_orient_seq, self.human_meshes, self.directions_seq 
        else:
            return self.human_joints_seq[idx:idx+1], self.orientations_seq[idx:idx+1], self.translation_seq[idx:idx+1], self.global_orient_seq[idx:idx+1], self.human_meshes[idx:idx+1], self.directions_seq[idx:idx+1]

    def get_joint_dict(self):
        return self.H


    def load_simple_all(self, arr):
    
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
            output = self.smpl_model(
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

        faces = self.smpl_model.faces
        
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

    