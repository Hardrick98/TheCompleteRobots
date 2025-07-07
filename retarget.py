from utils import *
from test_smpl import load_simple
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import argparse
import os
from scipy.spatial.transform import Rotation as Rot
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

        
    arr = np.load(args.human_pose, allow_pickle=True)
    
    
    joint_positions, orientations, translation, global_orient, human_mesh = load_simple(arr, 0)    

    translation[:,[1,2]] = translation[:,[2,1]]

    orientations = orientations.view(-1,3) 
     
    links_positions = robot.get_links_positions(q0)
   
                
    human_origin = translation[0]
    joint_positions[:,:] -= joint_positions[:1,:]
    joint_positions[:,0] *= -1
    joint_positions[:,[1,2]] = joint_positions[:,[2,1]]
    human_joints = joint_positions
    
    
    
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(human_joints[:22, 0], human_joints[:22, 1], human_joints[:22, 2], c='r', marker='o')



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
    
    head_fixed = False
    if "Head" not in R:
        head_fixed = True
        R["Head"] = R["root_joint"]
        F["Head"] = F["root_joint"]
    
    robot_limbs = [(R["LHip"],R["LKnee"]), (R["LKnee"], R["LAnkle"]), (R["RHip"],R["RKnee"]), 
                (R["RKnee"], R["RAnkle"]),(R["LShoulder"],R["LElbow"]), (R["LElbow"], R["LWrist"]), 
                (R["RShoulder"],R["RElbow"]), (R["RElbow"], R["RWrist"]), (R["Head"], R["RShoulder"]), 
                (R["Head"], R["LShoulder"]), (R["Head"], R["RHip"]),(R["Head"], R["LHip"])]
    
    robot_joints = scale_human_to_robot(R,F, robot_joints, H, human_joints)
  
    indices = [R["Head"],R["LHip"], R["LKnee"], R["LAnkle"], R["RHip"], R["RKnee"], R["RAnkle"],
               R["LShoulder"], R["LElbow"], R["LWrist"],
               R["RShoulder"], R["RElbow"], R["RWrist"]]
  
    ax.scatter(robot_joints[indices, 0], robot_joints[indices, 1], robot_joints[indices, 2], c='g', marker='o')
    
    for joint1_idx, joint2_idx in robot_limbs:
        try:
            if all(len(joint) == 3 for joint in (robot_joints[joint1_idx], robot_joints[joint2_idx])):
                x_coords, y_coords, z_coords = zip(robot_joints[joint1_idx], robot_joints[joint2_idx])
                ax.plot(x_coords, y_coords, z_coords, c="green", linewidth=2)
        except:
            pass
    
    
    links = robot.get_frames()
    joints = robot.joints
    print(links)
    
    joint_names = [v for k,v in F.items() if v != "root_joint"]
    joint_ids = [joints[name]for name in joint_names]

    frame_names = ["RHand","LHand"]
    frame_ids = [143,83]
    
    target_positions = {
        F["LHip"] : robot_joints[R["LHip"]],
        F["RHip"] : robot_joints[R["RHip"]], 
        F["LElbow"]: robot_joints[R["LElbow"]],
        F["RElbow"]: robot_joints[R["RElbow"]],
        F["LWrist"]: robot_joints[R["LWrist"]], 
        F["RWrist"]: robot_joints[R["RWrist"]], 
        F["RKnee"]: robot_joints[R["RKnee"]], 
        F["LKnee"]: robot_joints[R["LKnee"]], 
        F["LAnkle"]: robot_joints[R["LAnkle"]], 
        F["RAnkle"]: robot_joints[R["RAnkle"]], 
        F["RShoulder"] : robot_joints[R["RShoulder"]],
        F["LShoulder"] : robot_joints[R["LShoulder"]],
        F["Head"]: robot_joints[R["Head"]],
    }

    if head_fixed:
        target_positions.pop(F["Head"])

    target_orientations = {
        #"RHand": orientations[H["RWrist"]],  
        #"LHand": orientations[H["LWrist"]],
        #F["LElbow"]: orientations[H["LElbow"]],
        #F["RElbow"]: orientations[H["RElbow"]],
        #F["RAnkle"]: orientations[H["right_foot"]],
        #F["LKnee"] : orientations[H["LKnee"]],
    }

    
    rotvec = global_orient.numpy().flatten()
    global_rotation = torch.from_numpy(Rot.from_rotvec(rotvec).as_matrix()).float()
    
    

        
    global_orientations_matrices = get_smplx_global_orientations(global_rotation.double(), orientations.numpy(), change_ref = True)




    smplx_to_robot_mapping = {   
        1: F["LHip"],       
        2: F["RHip"],       
        12: F["Head"],       
        4: F["LKnee"],     # left_knee -> left_thigh
        5: F["RKnee"],     # right_knee -> right_thigh
        10: F["LAnkle"],    # left_ankle
        11: F["RAnkle"],    # right_ankle
        16: F["LShoulder"], # left_shoulder
        17: F["RShoulder"], # right_shoulder
        18: F["LElbow"],    # left_elbow
        19: F["RElbow"],    # right_elbow
        20: "LHand",   # left_wrist
        21: "RHand"    # right_wrist
    }
    
    pyplot_arrows(ax, global_orientations_matrices, human_joints, H)
    


    
    target_orientations_global = {}
    for smplx_idx, robot_frame in smplx_to_robot_mapping.items():
        if robot_frame in target_orientations:  
            rot_matrix = global_orientations_matrices[smplx_idx].numpy()
            target_orientations_global[robot_frame] = rot_matrix
    
    
    
    solver = InverseKinematicSolver(model,data,target_positions,target_orientations_global,joint_names, joint_ids, frame_names, frame_ids)
    

    q1 = solver.inverse_kinematics_position(q0)
    
    """
    q2 = solver.inverse_kinematics_orientation(q0)       

    print(q1)
    for joint_name, _ in target_orientations.items():
        print(joint_name)
        joint_id = model.getJointId(joint_name)
        idx = model.joints[joint_id].idx_q 
        q1[idx] = q2[idx]
    print(q1)
    """
    pin.forwardKinematics(model, data, q1)
    pin.updateFramePlacements(model, data)

    
    viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True) 
    viz.loadViewerModel()
    viz.display(q1)
    #plt.show()
    input("Press Enter to reset the visualization...")
    viz.reset()
    
    
    visual_model = robot.visual_model

    from vedo import Plotter, Mesh
    


    vp = Plotter(title="NAO Robot", axes=1, interactive=False)

    for visual in visual_model.geometryObjects:
        
        mesh_path = os.path.join(visual.meshPath.replace(".dae",".stl"))
        if not os.path.exists(mesh_path):
            print(f"Mesh non trovata: {mesh_path}")
            continue

        try:
            m = Mesh(mesh_path)
        except Exception as e:
            print(f"Errore nel caricare {mesh_path}: {e}")
            continue

        color = visual.meshColor
        m.color(color[:3])
        placement = data.oMf[visual.parentFrame]

        import pinocchio as pin
        placement_world = placement.act(visual.placement)
        R = placement_world.rotation
        p = placement_world.translation

        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p

        m.scale(visual.meshScale[0])
        m.apply_transform(T)

        vp += m


        M = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    T = np.eye(4)
    T[:3, :3] = M
    human_origin[0] *= -1

    T[:3, 3] = -human_origin + (np.array([0.7,0,1]))
    human_mesh.apply_transform(T)
    vp += human_mesh
    vp.camera.SetPosition([0, 3, 1])        
    vp.camera.SetFocalPoint([0, 0, 0])      
    vp.camera.SetViewUp([0, 0, 1])         

    vp.show(axes=1, interactive=True)

    