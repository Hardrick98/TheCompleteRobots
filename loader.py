from utils import *
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from inverse_kinematics import InverseKinematicSolver
from robotoid import Robotoid



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
        viz.initViewer(open=False) 
        viz.loadViewerModel()
        viz.display(robot.q0)
    pose_dict, robot_joints = robot.get_joints(robot.q0)
    _, robot_limbs = robot.get_physical_joints()
    
    
    
    model = robot.model
    data = robot.data
    q0 = robot.q0  

    
    #LOAD SIMPLE
        
    arr = np.load(args.human_pose, allow_pickle=True)
    
    
    joint_positions, orientations, translation, global_orient = load_simple(arr, 0)    

    translation[:,[1,2]] = translation[:,[2,1]]

    orientations = orientations.view(-1,3) 
     
    links_positions = robot.get_links_positions(q0)
   
        
    new = []        
    
    joint_positions[:,:] -= joint_positions[:1,:]
    joint_positions[:,[1,2]] = joint_positions[:,[2,1]]
    human_joints = joint_positions
    
    
    
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(human_joints[:, 0], human_joints[:, 1], human_joints[:, 2], c='r', marker='o')



    H = {
    "pelvis":0,
    "LHip":1,
    "RHip":2,
    "spine1":3,
    "LKnee":4,
    "RKnee":5,
    "spine2":6,
    "LAnkle":7,
    "RAnkle":8,
    "root_joint":9,
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

    robotoid = Robotoid(robot)
    F, R = robotoid.build()

        
    robot_limbs = [(R["LHip"],R["LKnee"]), (R["LKnee"], R["LAnkle"]), (R["RHip"],R["RKnee"]), 
                   (R["RKnee"], R["RAnkle"]),(R["LShoulder"],R["LElbow"]), (R["LElbow"], R["LWrist"]), 
                   (R["RShoulder"],R["RElbow"]), (R["RElbow"], R["RWrist"]), (R["Head"], R["RShoulder"]), 
                   (R["Head"], R["LShoulder"]), (R["Head"], R["RHip"]),(R["Head"], R["LHip"])]
    
    
    hipH = np.linalg.norm(human_joints[H["LHip"]]-human_joints[H["root_joint"]])
    hipR = np.linalg.norm(robot_joints[R["LHip"]]-robot_joints[R["root_joint"]])
    
    spineH = np.linalg.norm(human_joints[H["Neck"]]-human_joints[H["root_joint"]])
    spineR = np.linalg.norm(robot_joints[R["Head"]]-robot_joints[R["root_joint"]])
    
    shoulH = np.linalg.norm(human_joints[H["LShoulder"]]-human_joints[H["Neck"]])
    shoulR = np.linalg.norm(robot_joints[R["LShoulder"]]-robot_joints[R["Head"]])
    
    femorH = np.linalg.norm(human_joints[H["LKnee"]]-human_joints[H["LHip"]])
    femorR = np.linalg.norm(robot_joints[R["LKnee"]]-robot_joints[R["LHip"]])
    
    tibiaH = np.linalg.norm(human_joints[H["LAnkle"]]-human_joints[H["LKnee"]])
    tibiaR = np.linalg.norm(robot_joints[R["LAnkle"]]-robot_joints[R["LKnee"]])

    upper_armH = np.linalg.norm(human_joints[H["LElbow"]]-human_joints[H["LShoulder"]])
    upper_armR = np.linalg.norm(robot_joints[R["LElbow"]]-robot_joints[R["LShoulder"]])

    forearmH = np.linalg.norm(human_joints[H["LWrist"]]-human_joints[H["LElbow"]])
    forearmR = np.linalg.norm(robot_joints[R["LWrist"]]-robot_joints[R["LElbow"]])
    
    
    
    
    
    s_femor = femorR / femorH
    s_tibia = tibiaR / tibiaH
    s_upper_arm = upper_armR / upper_armH
    s_forearm = forearmR / forearmH
    s_spine = spineR / spineH
    s_shoulder = shoulR / shoulH
    s_hip = hipR / hipH


    #Scaling
    robot_joints[R["Head"]] = robot_joints[R["root_joint"]] + (human_joints[H["Neck"]] - human_joints[H["root_joint"]]) * s_spine
    robot_joints[R["RHip"]] = robot_joints[R["root_joint"]] + (human_joints[H["RHip"]] - human_joints[H["root_joint"]]) * s_hip
    robot_joints[R["RKnee"]] = robot_joints[R["RHip"]] + (human_joints[H["RKnee"]] - human_joints[H["RHip"]]) * s_femor
    robot_joints[R["RAnkle"]] = robot_joints[R["RKnee"]] + (human_joints[H["RAnkle"]] - human_joints[H["RKnee"]]) * s_tibia
    robot_joints[R["LShoulder"]] = robot_joints[R["Head"]] + (human_joints[H["LShoulder"]] - human_joints[H["Neck"]]) * s_shoulder
    robot_joints[R["LElbow"]] = robot_joints[R["LShoulder"]] + (human_joints[H["LElbow"]] - human_joints[H["LShoulder"]]) * s_upper_arm
    robot_joints[R["LWrist"]] = robot_joints[R["LElbow"]] + (human_joints[H["LWrist"]] - human_joints[H["LElbow"]]) * s_forearm
    robot_joints[R["LHip"]] = robot_joints[R["root_joint"]] + (human_joints[H["LHip"]] - human_joints[H["root_joint"]]) * s_hip
    robot_joints[R["LKnee"]] = robot_joints[R["LHip"]] + (human_joints[H["LKnee"]] - human_joints[H["LHip"]]) * s_femor
    robot_joints[R["LAnkle"]] = robot_joints[R["LKnee"]] + (human_joints[H["LAnkle"]] - human_joints[H["LKnee"]]) * s_tibia
    robot_joints[R["RShoulder"]] = robot_joints[R["Head"]] + (human_joints[H["RShoulder"]] - human_joints[H["Neck"]]) * s_shoulder
    robot_joints[R["RElbow"]] = robot_joints[R["RShoulder"]] + (human_joints[H["RElbow"]] - human_joints[H["RShoulder"]]) * s_upper_arm
    robot_joints[R["RWrist"]] = robot_joints[R["RElbow"]] + (human_joints[H["RWrist"]] - human_joints[H["RElbow"]]) * s_forearm
  
    indices = [R["Head"],R["LHip"], R["LKnee"], R["LAnkle"], R["RHip"], R["RKnee"], R["RAnkle"],
               R["LShoulder"], R["LElbow"], R["LWrist"],
               R["RShoulder"], R["RElbow"], R["RWrist"]]
  
    print("Robot joints after scaling:", robot_joints[indices])
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(robot_joints[indices, 0], robot_joints[indices, 1], robot_joints[indices, 2], c='g', marker='o')
    
    for joint1_idx, joint2_idx in robot_limbs:
        try:
            if all(len(joint) == 3 for joint in (robot_joints[joint1_idx], robot_joints[joint2_idx])):
                x_coords, y_coords, z_coords = zip(robot_joints[joint1_idx], robot_joints[joint2_idx])
                ax.plot(x_coords, y_coords, z_coords, c="green", linewidth=2)
        except:
            pass
    
    
    links = robot.joints  
    
    
    frame_names = [v for k,v in F.items() if v != "root_joint"]
    frame_ids = [links[name]for name in frame_names]

    
    target_positions = {
        #F["root_joint"]: robot_joints[R["root_joint"]],
        F["LHip"] : robot_joints[R["RHip"]],
        F["RHip"] : robot_joints[R["LHip"]], 
        F["LElbow"]: robot_joints[R["RElbow"]],
        F["RElbow"]: robot_joints[R["LElbow"]],
        F["LWrist"]: robot_joints[R["RWrist"]], 
        F["RWrist"]: robot_joints[R["LWrist"]], 
        F["RKnee"]: robot_joints[R["LKnee"]], 
        F["LKnee"]: robot_joints[R["RKnee"]], 
        F["LAnkle"]: robot_joints[R["RAnkle"]], 
        F["RAnkle"]: robot_joints[R["LAnkle"]], 
        F["Head"]: robot_joints[R["Head"]],
        F["RShoulder"] : robot_joints[R["LShoulder"]],
        F["LShoulder"] : robot_joints[R["RShoulder"]],
    }

    target_orientations = {
        #F["root_joint"]: orientations[H["pelvis"]],
        F["LWrist"]: orientations[H["RWrist"]],  
        F["RWrist"]: orientations[H["LWrist"]],
        #F["LShoulder"]: orientations[H["LShoulder"]],
        #F["LAnkle"]: orientations[H["LAnkle"]],
        #F["RAnkle"]: orientations[H["RAnkle"]],
        #F["Head"] : orientations[H["Neck"]],
    }

    index_keypoints = [
    1, 4, 7,            # left hip, knee, ankle
    2, 5, 8,            # right hip, knee, ankle
    16, 18, 20,         # left shoulder, elbow, wrist
    17, 19, 21          # right shoulder, elbow, wrist
    ]
    
    theta = -np.pi / 2  # -90 gradi
    R = torch.Tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])

    from scipy.spatial.transform import Rotation as Rot

    
    rotvec = global_orient.numpy().flatten()
    rotation = torch.from_numpy(Rot.from_rotvec(rotvec).as_matrix()).float()
    print(rotation)
    initial_vector = np.array([0, 0, 1])
    v = torch.tensor(initial_vector)
    rotation = R @ rotation
    direction = rotation @ v.float()
    direction = direction / torch.linalg.norm(direction)
    print("Direzione:", direction)


        
    global_orient = rotation.numpy() @ pin.exp3(global_orient.numpy().flatten())
    global_orientations_matrices = get_smplx_global_orientations(torch.from_numpy(global_orient).unsqueeze(0), joint_positions)


    smplx_to_robot_mapping = {
        9: F["root_joint"],      
        1: F["LHip"],       
        2: F["RHip"],       
        12: F["Head"],       
        4: F["LKnee"],     # left_knee -> left_thigh
        5: F["RKnee"],     # right_knee -> right_thigh
        7: F["LAnkle"],    # left_ankle
        8: F["RAnkle"],    # right_ankle
        16: F["LShoulder"], # left_shoulder
        17: F["RShoulder"], # right_shoulder
        18: F["LElbow"],    # left_elbow
        19: F["RElbow"],    # right_elbow
        20: F["LWrist"],   # left_wrist
        21: F["RWrist"]    # right_wrist
    }

    # 3. Aggiorna target_orientations con le orientazioni globali
    target_orientations_global = {}
    for smplx_idx, robot_frame in smplx_to_robot_mapping.items():
        if robot_frame in target_orientations:  
            rot_matrix = global_orientations_matrices[smplx_idx].numpy()
            target_orientations_global[robot_frame] = rot_matrix
    

    
    solver = InverseKinematicSolver(model,data,target_positions,target_orientations_global,frame_names, frame_ids)
    
    q1 = solver.inverse_kinematics_position(q0)

    #q1 = solver.end_effector_cost(q1, joint_name="RWrist", target_name="RWrist")gl
    #q1 = solver.end_effector_cost(q1, joint_name="LWrist", target_name="LWrist")
    #q1 = solver.end_effector_cost(q1, joint_name="LElbowYaw", target_name="LElbow")
    
    pin.forwardKinematics(model, data, q1)
    pin.updateFramePlacements(model, data)

    final_positions = []

    for name, frame_id in zip(frame_names, frame_ids):
        final_positions.append(data.oMf[frame_id].translation)
    
    final_positions = np.array(final_positions)
    
    ax.quiver(
        0, 0,0,                    # Punto di origine
        direction[0],              # Componente x della direzione
        direction[1],              # Componente y della direzione
        direction[2],              # Componente z della direzione
        length=2.0,                # Lunghezza della freccia
        color='yellow',
        normalize=True             # Normalizza la direzione
    )
    
    ax.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], c='b', marker='x')
    ax.view_init(azim=0, elev=0)
    
    viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True) 
    viz.loadViewerModel()
    
   
    viz.display(q1)
    plt.show()
    
    input("Press Enter to reset the visualization...")
    viz.reset()
    