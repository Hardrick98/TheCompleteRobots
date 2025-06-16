from pink.tasks import FrameTask
import numpy as np
from utils import *


tasks = {
    "torso": FrameTask(
        "torso",
        position_cost=1.0,              # [cost] / [m]
        orientation_cost=0.0,           # [cost] / [rad]
    ),
    "RShoulder": FrameTask(
        "RShoulder",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "LShoulder": FrameTask(
        "LShoulder",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "LElbow": FrameTask(
        "LElbow",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "RElbow": FrameTask(
        "RElbow",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "l_wrist": FrameTask(
        "l_wrist",
        position_cost=4.0,
        orientation_cost=0.0,
    ),
    "r_wrist": FrameTask(
        "r_wrist",
        position_cost=4.0,
        orientation_cost=0.0,
    ),
    "RPelvis": FrameTask(
        "RPelvis",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "LPelvis": FrameTask(
        "LPelvis",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "LThigh": FrameTask(
        "LThigh",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "RThigh": FrameTask(
        "RThigh",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "l_ankle": FrameTask(
        "l_ankle",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "r_ankle": FrameTask(
        "r_ankle",
        position_cost=1.0,
        orientation_cost=0.0,
    ),
    "Neck": FrameTask(
        "Neck",
        position_cost=1.0,
        orientation_cost=0.0,
    )
}

"""    
def transform_to_world_orientations(robot, configuration, target_orientations, frame_names):
# Aggiorna cinematica
    pin.forwardKinematics(robot.model, robot.data, robot.q0)
    pin.updateFramePlacements(robot.model, robot.data)
    
    world_orientations = {}
    
    for name in frame_names:
        if name in target_orientations:
            # Ottieni orientazione del frame nel world
            frame_id = robot.model.getFrameId(name)
            world_M_frame = robot.data.oMf[frame_id]
            frame_R_world = world_M_frame.rotation
            
            # Converti orientazione locale
            local_axis_angle = target_orientations[name].numpy()
            local_R = pin.exp3(local_axis_angle)
            
            # Componi le rotazioni
            world_R = frame_R_world @ local_R
            
            world_orientations[name] = world_R
    
    return world_orientations


target_orientations = transform_to_world_orientations(robot, robot.q0, target_orientations, frame_names)


configuration = pink.Configuration(robot.model, robot.data, robot.q0)
for body, task in tasks.items():
    R = target_orientations[body]
    #R = pin.exp3(axis_angle)       # Convert axis-angle to rotation matrix
    #quat = pin.Quaternion(R) 
    if type(task) is FrameTask:
        task.set_target(configuration.get_transform_frame_to_world(body)*pin.SE3(
    R, target_positions[body])
)

    # Select QP solver
solver = qpsolvers.available_solvers[0]
if "daqp" in qpsolvers.available_solvers:
    solver = "daqp"

t = 0.0  # [s]
dt = 6e-3
total_time = 20.0

# Loop di simulazione
for t in np.arange(0.0, total_time, dt):
    # Risolvi IK (ora molto più semplice!)
    velocity = solve_ik(
        configuration, 
        tasks.values(), 
        dt, 
        solver=solver, 
        safety_break=False
    )
    
    # Integra
    configuration.integrate_inplace(velocity, dt)
    pin.updateFramePlacements(robot.model, robot.data)
    print(configuration.q)
    viz.display(configuration.q)        
    time.sleep(dt)
viz.display(configuration.q)
"""