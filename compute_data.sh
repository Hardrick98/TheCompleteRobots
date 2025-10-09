python retarget_motion.py --robot nao --interaction /datasets/Inter-X/motions/$1
python robot_interaction.py --robot1 nao --robot2 nao --interaction /datasets/Inter-X/motions/$1
python cool_visualization.py --interaction /datasets/Inter-X/motions/$1 --robot1 nao --robot2 nao --scene scenes/room.glb --video --camera_mode exo
python cool_visualization.py --interaction /datasets/Inter-X/motions/$1 --robot1 nao --robot2 nao --scene scenes/room.glb --video --camera_mode ego1
python cool_visualization.py --interaction /datasets/Inter-X/motions/$1 --robot1 nao --robot2 nao --scene scenes/room.glb --video --camera_mode ego2

echo "Script completed!"