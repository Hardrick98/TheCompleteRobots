from controller import Robot, Motor
import sys
import json

f = open('/home/rcatalini/Napoli/nao.json')
action = json.load(f)[0]

robot = Robot()

timestep = int(robot.getBasicTimeStep())


joint_names = []


TIME_STEP = 16

def my_step():
    if robot.step(TIME_STEP) == -1:
        robot.cleanup()
        sys.exit(0)
  

def get_motor_names(robot):
    
    num_devices = robot.getNumberOfDevices()
    
    motor_names = []
    for i in range(num_devices):
        device = robot.getDeviceByIndex(i)
        if device.getNodeType() == 57:      
            motor_names.append(device.getName())
        
    return motor_names

joint_names = get_motor_names(robot)


def get_limits(joint_names):
    
    positions = {}
    limits = {}
    for name in joint_names:
    
        motor = robot.getDevice(name)
        positions[name] = motor.getTargetPosition()
        min_pos = motor.getMinPosition()
        max_pos = motor.getMaxPosition()
        
        limits[name] = [min_pos,max_pos]
        
    return positions, limits
 

    
    
positions, limits = get_limits(joint_names)

motors = {}
for name in joint_names:
    motor = robot.getDevice(name)
    motors[name] = motor

robot.step(timestep)


n_steps_to_achieve_target = 1000 // TIME_STEP  


def move_to_pose_simple(pose_dict, duration=1.0):
    for name, target in pose_dict.items():
        if name in motors:
            motor = motors[name]
            motor.setVelocity(1.0)
            motor.setPosition(target)
    
    steps = int(duration * 1000 // TIME_STEP)
    for _ in range(steps):
        my_step()


move_to_pose_simple(action["base_pose"])
print("Movement executed successfully")
    

        
        
        


