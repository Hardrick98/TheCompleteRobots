from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize


if __name__ == "__main__":
    
    robot_list = [r.removesuffix(".urdf") for r in os.listdir("/home/rick/TheCompleteRobots/URDF") if r.endswith(".urdf") or r.endswith(".urdf")]
    
    parser = argparse.ArgumentParser(description="Visualize a humanoid robot model.")
    parser.add_argument(
        "--robot",
        type=str,
        default="nao",
        help="The robot to visualize.",
    )
    parser.add_argument("--visualize",
                        action="store_true",
                        help="Visualize the robot model in Meshcat.")
    args  = parser.parse_args()
    robot_name = args.robot.lower()    
    print(robot_name)
    try:
        robot = HumanoidRobot(f"/home/rick/TheCompleteRobots/URDF/{args.robot}.urdf")
    except Exception as e:
        print(f"Error loading robot {robot_name}: {e}")
        print("Available robots:")
        for r in robot_list:
            print(f"- {r}")
        exit(1)
    
    if args.visualize:
        viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz.initViewer(open=True) 
        viz.loadViewerModel()
        viz.display(robot.q0)
    pose_dict, keypoints = robot.get_joints(robot.q0)
    robot_joints, robot_limbs = robot.get_physical_joints()
    
    
    
    model = robot.model
    data = robot.data
    q0 = robot.q0  
    print("Robot joints:", pose_dict)
   
    
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(robot_joints[:, 0], robot_joints[:, 1], robot_joints[:, 2], c='r', marker='o')
    for joint1_idx, joint2_idx in robot_limbs:
        if all(len(joint) == 3 for joint in (robot_joints[joint1_idx], robot_joints[joint2_idx])):
            x_coords, y_coords, z_coords = zip(robot_joints[joint1_idx], robot_joints[joint2_idx])
            ax.plot(x_coords, y_coords, z_coords, c="red", linewidth=2)
    
### GET PARENT-CHILD RELATIONS

parent_child = {}

for i, joint_name in enumerate(robot.model.names):
    parent_idx = robot.model.parents[i]
    if parent_idx == 0 and i != 0:
        parent_name = "root"
    else:
        parent_name = robot.model.names[parent_idx]
    parent_child[joint_name] = parent_name

print(parent_child)


all_parents = set(model.parents)

#GET END-EFFECTORS (JOINTS WITH NO CHILDREN)

end_effectors = []
for i, joint_name in enumerate(model.names):
    if i not in all_parents or i == 0:
        if i != 0:
            end_effectors.append(joint_name)

print("End effectors trovati:")
for ee in end_effectors:
    print(f"- {ee}")

#FIND KINEMATIC CHAINS

def get_kinematic_chains(end_effectors, parent_child):
    
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

chains = get_kinematic_chains(end_effectors, parent_child)


new_chains = []
for chain in chains:
    
    new_chains.append([model.getJointId(joint) for joint in chain if model.getJointId(joint)!=1])


positions = []
for chain in new_chains:
    positions.append(data.oMi[chain[0]].translation)

positions = np.array(positions)
    
#PULIRE CATENE INUTILI

positions = positions[positions[:, 2].argsort()[::-1]]


chains_new = []

for chain in new_chains:
    positions = {}
    for j in (chain):
        positions[model.names[j]] = data.oMi[j].translation
    chains_new.append(positions)

##CLUSTERIZE TO UNDERSTAND WHICH DEGREE OF FREEDOM IS RESPONSIBLE FOR A SPECIFIC LIMB

robotoid = {}
robotoid_labels = {}

chain_value = 0
for positions in chains_new:
    
    print("\n\n")
    
    filtered_data = {k: v for k, v in positions.items()}

    keys = list(filtered_data.keys())
    points = np.array(list(filtered_data.values()))

    from sklearn.cluster import KMeans
    
    if points.shape[0] > 3:
        kmeans = KMeans(n_clusters=3, random_state=0).fit(points)
    else:
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
    


for chain_id in robotoid:
    # Prendi i centroidi e le label
    centers = robotoid[chain_id]
    labels = robotoid_labels[chain_id]
    
    # Ordina per la coordinata z ([:, 2])
    sorted_indices = np.argsort(centers[:, 2])[::-1] 
    
    # Applica l'ordinamento
    robotoid[chain_id] = centers[sorted_indices]
    robotoid_labels[chain_id] = [labels[i] for i in sorted_indices]

sorted_chain_ids = sorted(robotoid.keys(), key=lambda i: robotoid[i][0, 2], reverse=True)
robotoid = {i: robotoid[i] for i in sorted_chain_ids}
robotoid_labels = {i: robotoid_labels[i] for i in sorted_chain_ids}

new_robotoid = {i: v for i, (_, v) in enumerate(robotoid.items())}
new_robotoid_labels = {i: v for i, (_, v) in enumerate(robotoid_labels.items())}
   
robotoid = new_robotoid
robotoid_labels = new_robotoid_labels 
print(robotoid)
print(robotoid_labels)

final = {}

final["Head"] = robotoid_labels[0]

if robotoid[1][0,1] > 0:
    final["LShoulder"] = robotoid_labels[1][0]
    final["LElbow"] = robotoid_labels[1][1]
    final["LWrist"] = robotoid_labels[1][2]
    final["RShoulder"] = robotoid_labels[2][0]
    final["RElbow"] = robotoid_labels[2][1]
    final["LWrist"] = robotoid_labels[2][2]
else:
    final["LShoulder"] = robotoid_labels[2][0]
    final["LElbow"] = robotoid_labels[2][1]
    final["LWrist"] = robotoid_labels[2][2]
    final["RShoulder"] = robotoid_labels[1][0]
    final["RElbow"] = robotoid_labels[1][1]
    final["LWrist"] = robotoid_labels[1][2]
    
    
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
    
print(final)