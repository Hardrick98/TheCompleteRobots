from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from inverse_kinematics import InverseKinematicSolver
#from graph_net import HumanoidJointMapper
from scipy.optimize import minimize
from ik_pin import tasks
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
    parser.add_argument("--human_pose",
                        type=str,
                        help="Path to smpl human pose")
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
        
    arr = np.load(args.human_pose, allow_pickle=True)
    #arr = np.load("/datasets/HumanoidX/human_pose/youtube/q.npz", allow_pickle=True)
    
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
    
    
    links = robot.joints
    print("Robot frames:", links)
    
    
    
    #frame_names = [ "RShoulder", "LShoulder", "LElbow", "RElbow", "l_wrist", "r_wrist", "RHip", "LHip","LTibia","RTibia", "l_ankle", "r_ankle", "Neck"]
    frame_names = ['HeadYaw', 'LShoulderPitch', 'LElbowYaw', 'LWristYaw', 'RShoulderPitch', 'RElbowYaw', 
                   'RWristYaw', 'LHipYawPitch', 'LKneePitch','LAnklePitch', 'RHipYawPitch', 
                   'RKneePitch', 'RAnklePitch']
    frame_ids = [links[name]for name in frame_names]
    
    target_positions = {
        #"torso" :robot_joints[0],
        "LHipYawPitch": robot_joints[2],
        "RHipYawPitch" : robot_joints[17],
        "LElbowYaw": robot_joints[6], 
        "RElbowYaw": robot_joints[21], 
        "LWristYaw": robot_joints[7],  
        "RWristYaw": robot_joints[22],
        "LKneePitch" : robot_joints[3],
        "RKneePitch" : robot_joints[18],
        "LAnklePitch": robot_joints[4],
        "RAnklePitch": robot_joints[19],
        "HeadYaw" : robot_joints[1],
        "RShoulderPitch": robot_joints[20],
        "LShoulderPitch" : robot_joints[5]
    }

    print("Target positions:", target_positions)
    
    target_orientations = {
        #"torso": orientations[9],
        "LWristYaw": orientations[20],  
        "RWristYaw": orientations[21],
        #"LElbow": orientations[18],
        #"LShoulder": orientations[17]
        #"l_ankle": orientations[7],
        #"r_ankle": orientations[8],
        #"Neck" : orientations[12],
    }


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
        20: "LWristYaw",   # left_wrist
        21: "RWristYaw"    # right_wrist
    }

    # 3. Aggiorna target_orientations con le orientazioni globali
    target_orientations_global = {}
    for smplx_idx, robot_frame in smplx_to_robot_mapping.items():
        if robot_frame in target_orientations:  
            # Converti da matrice di rotazione a angle-axis per Pinocchio
            rot_matrix = global_orientations_matrices[smplx_idx].numpy()
            # Usa scipy o implementazione custom per convertire
            
            target_orientations_global[robot_frame] = rot_matrix
        

    
    solver = InverseKinematicSolver(model,data,target_positions,target_orientations_global,frame_names, frame_ids)
    
    q1 = solver.inverse_kinematics_position(q0)

    q1 = solver.end_effector_cost(q1, joint_name="RWristYaw", target_name="RWristYaw")
    q1 = solver.end_effector_cost(q1, joint_name="LWristYaw", target_name="LWristYaw")
    #q1 = solver.end_effector_cost(q1, joint_name="LElbowYaw", target_name="LElbow")
    
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
    
    
    
    ###Convert to WEBOTS pose
    
    DoF = ['HeadYaw', 'HeadPitch', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 
     'RWristYaw', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 
     'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll', 
     'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll']
    
    webots_pose = {}
    
    for joint_name in DoF:
        joint_id = model.getJointId(joint_name)
        idx = model.joints[joint_id].idx_q
    
        webots_pose[joint_name] = q1[idx]
        
    import json
    
    json_path = f"{robot_name}_pose.json"
    with open(json_path, 'w') as f:
        json.dump([webots_pose], f, indent=4)
    