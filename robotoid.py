from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse





class Robotoid:

    def __init__(self, robot):
        
        self.pose_dict, _ = robot.get_joints(robot.q0)
        self.robot_joints, self.robot_limbs = robot.get_physical_joints()
        self.model = robot.model
        self.data = robot.data


    def get_kinematic_chains(self,end_effectors, parent_child):
            
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
    
    def build(self):

        parent_child = {}

        for i, joint_name in enumerate(self.model.names):
            parent_idx = self.model.parents[i]
            if parent_idx == 0 and i != 0:
                parent_name = "root"
            else:
                parent_name = self.model.names[parent_idx]
            parent_child[joint_name] = parent_name


        all_parents = set(self.model.parents)

        #GET END-EFFECTORS (JOINTS WITH NO CHILDREN)

        end_effectors = []
        for i, joint_name in enumerate(self.model.names):
            if i not in all_parents or i == 0:
                if i != 0:
                    end_effectors.append(joint_name)

        
        repeated = []

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


        final_reduced = {}
        final_values = {}
        for k,v in final.items():
            if v[0] is not None:
                final_reduced[k] = v[0]
                final_values[k] = self.model.getJointId(v[0])-1

        final_reduced["root_joint"] = "root_joint"
        final_values["root_joint"] = self.model.getJointId("root_joint") - 1


        return final_reduced, final_values
