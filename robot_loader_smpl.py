from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import qpsolvers
#from graph_net import HumanoidJointMapper
from scipy.optimize import minimize
from ik import tasks
import pink
from pink import solve_ik
from pink.tasks import FrameTask
import time
from loop_rate_limiters import RateLimiter


if __name__ == "__main__":
    
    robot_list = [r.removesuffix(".urdf") for r in os.listdir("URDF") if r.endswith(".urdf") or r.endswith(".urdf")]
    
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
        robot = HumanoidRobot(f"URDF/{args.robot}.urdf")
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
   
    """
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(robot_joints[:, 0], robot_joints[:, 1], robot_joints[:, 2], c='r', marker='o')
    for joint1_idx, joint2_idx in robot_limbs:
        if all(len(joint) == 3 for joint in (robot_joints[joint1_idx], robot_joints[joint2_idx])):
            x_coords, y_coords, z_coords = zip(robot_joints[joint1_idx], robot_joints[joint2_idx])
            ax.plot(x_coords, y_coords, z_coords, c="red", linewidth=2)
    """
    
    #LOAD SIMPLE
        
    #arr = np.load("/datasets/HumanoidX/human_pose/kinetics700/pADXOvpw1CM_000028_000038_clip_1.npz", allow_pickle=True)
    arr = np.load("/datasets/HumanoidX/human_pose/youtube/zoo_yoga_for_hampton_primary_clip_1.npz", allow_pickle=True)
    
    joint_positions, orientations, translation, global_orient = load_simple(arr, 20)    

    translation[:,[1,2]] = translation[:,[2,1]]

    index_keypoints = [
    1, 4, 7,            # left hip, knee, ankle
    2, 5, 8,            # right hip, knee, ankle
    16, 18, 20,         # left shoulder, elbow, wrist
    17, 19, 21          # right shoulder, elbow, wrist
    ]

    orientations = orientations.view(-1,3) 
     
    links_positions = robot.get_links_positions(q0)
   
        
    new = []
    
    robot_limbs = [(2, 3), (3, 4), (5, 6), (6, 7), (17, 18), (18, 19),
                   (20, 21), (21, 22), (0, 1), (0, 17), (1, 20), (0, 2), (1, 5)]
    

        
    
    joint_positions[:,:] -= joint_positions[:1,:]
    joint_positions[:,[1,2]] = joint_positions[:,[2,1]]
    human_joints = joint_positions
    
    
    
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(human_joints[:, 0], human_joints[:, 1], human_joints[:, 2], c='r', marker='o')

    """
    mapper = HumanoidJointMapper()
    (mapping, quality, similarity_matrix, human_names, robot_names, 
     unused_joints) = mapper.find_optimal_mapping(
        human_joints, human36m_limbs,
        robot_joints, robot_limbs, human_joint_names=human36m_names
    )

    mapper.visualize_mapping_with_links(
        human_joints, human36m_limbs,
        robot_joints, robot_limbs,
        mapping, quality, unused_joints,human_joint_names=human36m_names
        
    )
    
    mapper.get_detailed_report(mapping, quality, unused_joints)
    
    """



    #robot_joints = np.concatenate((np.zeros((1,3)),robot_joints))

    ## Link lenghts
    
   
    
    hipH = np.linalg.norm(human_joints[1]-human_joints[9])
    hipR = np.linalg.norm(robot_joints[2]-robot_joints[0])
    spineH = np.linalg.norm(human_joints[12]-human_joints[9])
    spineR = np.linalg.norm(robot_joints[1]-robot_joints[0])
    shoulH = np.linalg.norm(human_joints[16]-human_joints[12])
    shoulR = np.linalg.norm(robot_joints[5]-robot_joints[1])
    femorH = np.linalg.norm(human_joints[4]-human_joints[1])
    tibiaH = np.linalg.norm(human_joints[7]-human_joints[4])
    upper_armH = np.linalg.norm(human_joints[18]-human_joints[16])
    forearmH = np.linalg.norm(human_joints[20]-human_joints[18])
    femorR = np.linalg.norm(robot_joints[18]-robot_joints[17])
    tibiaR = np.linalg.norm(robot_joints[19]-robot_joints[18])
    upper_armR = np.linalg.norm(robot_joints[6]-robot_joints[5])
    forearmR = np.linalg.norm(robot_joints[7]-robot_joints[6])
    
    
    
    
    
    s_femor = femorR / femorH
    s_tibia = tibiaR / tibiaH
    s_upper_arm = upper_armR / upper_armH
    s_forearm = forearmR / forearmH
    s_spine = spineR / spineH
    s_shoulder = shoulR / shoulH
    s_hip = hipR / hipH


    #Scaling
    robot_joints[1] = robot_joints[0] + (human_joints[12]-human_joints[9]) * s_spine
    robot_joints[17] = robot_joints[0] + (human_joints[1]-human_joints[9]) * s_hip
    robot_joints[18] = robot_joints[17] + (human_joints[4] - human_joints[1]) * s_femor
    robot_joints[19] = robot_joints[18] + (human_joints[7] - human_joints[4]) * s_tibia
    robot_joints[2] = robot_joints[0] + (human_joints[2]-human_joints[9]) * s_hip
    robot_joints[3] = robot_joints[2] + (human_joints[5] - human_joints[2]) * s_femor
    robot_joints[4] = robot_joints[3] + (human_joints[8] - human_joints[5]) * s_tibia
    robot_joints[20] = robot_joints[1] + (human_joints[16]-human_joints[12]) * s_shoulder
    robot_joints[21] = robot_joints[20] + (human_joints[18] - human_joints[16]) * s_upper_arm
    robot_joints[22] = robot_joints[21] + (human_joints[20] - human_joints[18]) * s_forearm
    robot_joints[5] = robot_joints[1] + (human_joints[17]-human_joints[12]) * s_shoulder
    robot_joints[6] = robot_joints[5] + (human_joints[19] - human_joints[17]) * s_upper_arm
    robot_joints[7] = robot_joints[6] + (human_joints[21] - human_joints[19]) * s_forearm    

    

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(robot_joints[:, 0], robot_joints[:, 1], robot_joints[:, 2], c='g', marker='o')
    for joint1_idx, joint2_idx in robot_limbs:
        try:
            if all(len(joint) == 3 for joint in (robot_joints[joint1_idx], robot_joints[joint2_idx])):
                x_coords, y_coords, z_coords = zip(robot_joints[joint1_idx], robot_joints[joint2_idx])
                ax.plot(x_coords, y_coords, z_coords, c="green", linewidth=2)
        except:
            pass
    
    
    links = robot.body
    print("Robot frames:", links)
    
    
    
    frame_names = [ "RShoulder", "LShoulder", "LElbow", "RElbow", "l_wrist", "r_wrist", "RHip", "LHip","LTibia","RTibia", "l_ankle", "r_ankle", "Neck"]
    frame_ids = [links[name]for name in frame_names]
    
    target_positions = {
        #"torso" :robot_joints[0],
        "LHip": robot_joints[2],
        "RHip" : robot_joints[17],
        "LElbow": robot_joints[6], 
        "RElbow": robot_joints[21], 
        "l_wrist": robot_joints[7],  
        "r_wrist": robot_joints[22],
        "LTibia" : robot_joints[3],
        "RTibia" : robot_joints[18],
        "l_ankle": robot_joints[4],
        "r_ankle": robot_joints[19],
        "Neck" : robot_joints[1],
        "RShoulder": robot_joints[20],
        "LShoulder" : robot_joints[5]
    }

    target_orientations = {
        #"torso": orientations[9],
        "l_wrist": orientations[20],  
        "r_wrist": orientations[21],
        #"l_ankle": orientations[7],
        #"r_ankle": orientations[8],
        #"Neck" : orientations[12],
    }

    import scipy.spatial.transform

    def rotation_error(R, R_target):
        R_diff = R_target.T @ R
        rotvec = scipy.spatial.transform.Rotation.from_matrix(R_diff).as_rotvec()
        return np.linalg.norm(rotvec)**2  

        #tasks[name].set_target_position(pos.tolist())
    index_keypoints = [
    1, 4, 7,            # left hip, knee, ankle
    2, 5, 8,            # right hip, knee, ankle
    16, 18, 20,         # left shoulder, elbow, wrist
    17, 19, 21          # right shoulder, elbow, wrist
    ]
    
    
    global_orientations_matrices = get_smplx_global_orientations(global_orient, joint_positions)

    # 2. Mappa le orientazioni SMPL-X ai frame del tuo robot
    # Mapping da indici SMPL-X a nomi dei tuoi frame
    smplx_to_robot_mapping = {
        #9: "torso",      
        1: "LHip",       
        2: "RHip",       
        12: "Neck",       
        4: "LTibia",     # left_knee -> left_thigh
        5: "RTibia",     # right_knee -> right_thigh
        7: "l_ankle",    # left_ankle
        8: "r_ankle",    # right_ankle
        16: "LShoulder", # left_shoulder
        17: "RShoulder", # right_shoulder
        18: "LElbow",    # left_elbow
        19: "RElbow",    # right_elbow
        20: "l_wrist",   # left_wrist
        21: "r_wrist"    # right_wrist
    }

    # 3. Aggiorna target_orientations con le orientazioni globali
    target_orientations_global = {}
    for smplx_idx, robot_frame in smplx_to_robot_mapping.items():
        if robot_frame in target_orientations:  
            # Converti da matrice di rotazione a angle-axis per Pinocchio
            rot_matrix = global_orientations_matrices[smplx_idx].numpy()
            # Usa scipy o implementazione custom per convertire
            
            target_orientations_global[robot_frame] = rot_matrix
        
    
    print(target_orientations_global)
    
    def ik_cost(q, w_pos=1, w_ori=0.0001):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        cost_pos = 0.0
        cost_ori = 0.0
        
        for name, frame_id in zip(frame_names, frame_ids):
            oMf = data.oMf[frame_id]
            pos = oMf.translation
            ori = oMf.rotation#model.frames[frame_id].placement.rotation 
            target_pos = target_positions[name]
            cost_pos += np.linalg.norm(pos - target_pos)**2
            
            if name in target_orientations_global:
                
                target_ori = target_orientations_global[name]
                #target_ori = pin.exp3(target_ori.numpy())
                cost_ori += rotation_error(ori, target_ori)
                
        return w_pos * cost_pos + w_ori * cost_ori
    
    
    
    
    
    

    q_lower_limits = model.lowerPositionLimit
    q_upper_limits = model.upperPositionLimit
    bounds = []
    for i in range(model.nq):
        bounds.append((q_lower_limits[i], q_upper_limits[i]))

    

    res = minimize(ik_cost, q0, bounds=bounds, method='SLSQP', options={'maxiter': 1000, 'disp': True})
    

        
    q1 = np.array(res.x).reshape(-1)
    assert q1.shape[0] == model.nq
    
    
    joint_name = "RWristYaw"
    joint_id = model.getJointId(joint_name)
    idx = model.joints[joint_id].idx_q  # indice nel vettore q

    target_ori = target_orientations_global["r_wrist"]

    def rotation_error(R1, R2):
        R_diff = R1.T @ R2
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        return angle

    # Funzione costo: ottimizzi solo il valore q[idx]
    def cost(theta):
        q_tmp = q1.copy()
        q_tmp[idx] = theta[0] 
        print(q_tmp[idx])
        # solo il grado di libertà del giunto
        pin.forwardKinematics(model, data, q_tmp)
        pin.updateFramePlacements(model, data)
        R = data.oMi[joint_id].rotation
        return rotation_error(R, target_ori)

    # Ottimizza solo q1[idx]
    res = minimize(cost, q1[idx], method='SLSQP', options={'maxiter': 1000, 'disp': True})
    q1[idx] = res.x[0]  
    
    
    joint_name = "LWristYaw"
    joint_id = model.getJointId(joint_name)
    idx = model.joints[joint_id].idx_q  # indice nel vettore q

    target_ori = target_orientations_global["l_wrist"]

    def rotation_error(R1, R2):
        R_diff = R1.T @ R2
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        return angle

    # Funzione costo: ottimizzi solo il valore q[idx]
    def cost(theta):
        q_tmp = q1.copy()
        q_tmp[idx] = theta[0] 
        print(q_tmp[idx])
        # solo il grado di libertà del giunto
        pin.forwardKinematics(model, data, q_tmp)
        pin.updateFramePlacements(model, data)
        R = data.oMi[joint_id].rotation
        return rotation_error(R, target_ori)

    # Ottimizza solo q1[idx]
    res = minimize(cost, q1[idx], method='SLSQP', options={'maxiter': 1000, 'disp': True})
    q1[idx] = res.x[0]  
    
    
    pin.forwardKinematics(model, data, q1)
    pin.updateFramePlacements(model, data)

    final_positions = []

    for name, frame_id in zip(frame_names, frame_ids):
        final_positions.append(data.oMf[frame_id].translation)
    
    final_positions = np.array(final_positions)
    ax.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], c='b', marker='x')
    plt.show()
    
    viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True) 
    viz.loadViewerModel()
    
    print(q1.shape)
   
    viz.display(q1)
    plt.show()
    input("Press Enter to reset the visualization...")
    viz.reset()
    