from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from graph_net import HumanoidJointMapper
from scipy.optimize import minimize


if __name__ == "__main__":
    
    robot_list = [r.removesuffix(".urdf") for r in os.listdir("/home/rcatalini/TheCompleteRobot/URDF") if r.endswith(".urdf") or r.endswith(".urdf")]
    
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
        robot = HumanoidRobot(f"/home/rcatalini/TheCompleteRobot/URDF/{args.robot}.urdf")
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
    
    
    #LOAD SIMPLE
        
    arr = np.load("/datasets/HumanoidX/human_pose/youtube/red_dot_scary_maze_prank_on_my_son_for_his_reaction_shorts_clip_1.npz", allow_pickle=True)
    
    joint_positions, orientations = load_simple(arr)    

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
    
    
    print(robot_limbs)
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(links_positions[:, 0], links_positions[:, 1], links_positions[:, 2], c='r', marker='o')
    ax.scatter(robot_joints[:, 0], robot_joints[:, 1], robot_joints[:, 2], c='b', marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()
        
    
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
    
    
    #torsoH = np.linalg.norm(human_joints[11]-human_joints[8])
    #torsoR = np.linalg.norm(robot_joints[5]-robot_joints[1])
    #spineH = np.linalg.norm(human_joints[8]-human_joints[0])
    #spineR = np.linalg.norm(robot_joints[1]-robot_joints[0])
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
    #s_spine = spineR / spineH
    #s_torso = torsoR / torsoH


    #Scaling
    
    #robot_joints[1] = robot_joints[0] + (human_joints[8]-human_joints[0]) * s_spine
    #robot_joints[0] = robot_joints[17] + (human_joints[1]-human_joints[0]) *s_spine

    #robot_joints[5] = robot_joints[1] + (human_joints[11]-human_joints[8]) * s_torso
    #robot_joints[20] = robot_joints[1] + (human_joints[14]-human_joints[8]) * s_torso
    
    robot_joints[18] = robot_joints[17] + (human_joints[4] - human_joints[1]) * s_femor
    robot_joints[19] = robot_joints[18] + (human_joints[7] - human_joints[4]) * s_tibia
    robot_joints[3] = robot_joints[2] + (human_joints[5] - human_joints[2]) * s_femor
    robot_joints[4] = robot_joints[3] + (human_joints[8] - human_joints[5]) * s_tibia
    
    
    robot_joints[21] = robot_joints[20] + (human_joints[18] - human_joints[16]) * s_upper_arm
    robot_joints[22] = robot_joints[21] + (human_joints[20] - human_joints[18]) * s_forearm
    robot_joints[6] = robot_joints[5] + (human_joints[19] - human_joints[17]) * s_upper_arm
    robot_joints[7] = robot_joints[6] + (human_joints[21] - human_joints[19]) * s_forearm    

    
    print(robot_joints)

    
    #ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
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
    plt.show()
    
    links = robot.body
    print("Robot frames:", links)
    

    
    
    frame_names = ["l_wrist", "r_wrist", "LElbow", "RElbow", "LThigh","RThigh","l_ankle", "r_ankle", "RShoulder", "LShoulder"]
    frame_ids = [links[name]for name in frame_names]
    
    target_positions = {
        "LElbow": robot_joints[6], 
        "RElbow": robot_joints[21], 
        "l_wrist": robot_joints[7],  
        "r_wrist": robot_joints[22],
        "LThigh" : robot_joints[3],
        "RThigh" : robot_joints[18],
        "l_ankle": robot_joints[4],
        "r_ankle": robot_joints[19],
        #"Neck" : robot_joints[1],
        "RShoulder": robot_joints[20],
        "LShoulder" : robot_joints[5]
    }

    target_orientations = {
        "LElbow": orientations[18], 
        "RElbow": orientations[19], 
        "l_wrist": orientations[20],  
        "r_wrist": orientations[21],
        "LThigh" : orientations[4],
        "RThigh" : orientations[5],
        "l_ankle": orientations[7],
        "r_ankle": orientations[8],
        #"Neck" : robot_joints[1],
        "RShoulder": orientations[17],
        "LShoulder" : orientations[16]
    }
    
    index_keypoints = [
    1, 4, 7,            # left hip, knee, ankle
    2, 5, 8,            # right hip, knee, ankle
    16, 18, 20,         # left shoulder, elbow, wrist
    17, 19, 21          # right shoulder, elbow, wrist
    ]
    
    def ik_cost(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        cost = 0.0
        for name, frame_id in zip(frame_names, frame_ids):
            oMf = data.oMf[frame_id]  
            pos = oMf.translation
            ori = oMf.orientation
            target = target_positions[name]
            target_ori = target_orientations[name]
            cost += np.linalg.norm(pos - target)**2
        
        #pred_torso = data.oMf[torso_frame_id].rotation
        #pred_neck = data.oMf[neck_frame_id].rotation
        #rot_error = pin.log3(pred_torso.T @ target_torso) + pin.log3(pred_neck.T @ target_neck)  # Logaritmo della rotazione
        #orientation_cost = np.linalg.norm(rot_error)**2
        
        return cost #+ 10.0 * orientation_cost

    bounds = [(-np.pi, np.pi)] * model.nq  

    res = minimize(ik_cost, q0, bounds=bounds, method='L-BFGS-B')
    q1 = res.x  # co
    q1 = np.array(res.x).reshape(-1)
    assert q1.shape[0] == model.nq
    pin.forwardKinematics(model, data, q1)
    pin.updateFramePlacements(model, data)
    
    viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=False) 
    viz.loadViewerModel()
    viz.display(q1)
    input("Press Enter to reset the visualization...")
    viz.reset()
    