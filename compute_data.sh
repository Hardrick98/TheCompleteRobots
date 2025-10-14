echo "Starting Retargeting"

python retarget_motion.py --robot nao --interaction $1
python robot_interaction.py --robot1 nao --robot2 nao --interaction $1

echo "Rendering videos..."

python render.py --interaction $1 --robot1 nao --robot2 nao --scene scenes/room.glb --video --camera_mode exoR
python render.py --interaction $1 --robot1 nao --robot2 nao --scene scenes/room.glb --video --camera_mode ego1R
python render.py --interaction $1 --robot1 nao --robot2 nao --scene scenes/room.glb --video --camera_mode ego2R
python render.py --interaction $1 --robot1 nao --robot2 nao --scene scenes/room.glb --video --camera_mode exoL
python render.py --interaction $1 --robot1 nao --robot2 nao --scene scenes/room.glb --video --camera_mode ego1L
python render.py --interaction $1 --robot1 nao --robot2 nao --scene scenes/room.glb --video --camera_mode ego2L

echo "Computing Bounding Boxes..."


python extract_bb.py --interaction $1 --robot1 nao --robot2 nao --green_screen --camera_mode exoR --bb_mode1
python extract_bb.py --interaction $1 --robot1 nao --robot2 nao --green_screen --camera_mode exoR --bb_mode2
python extract_bb.py --interaction $1 --robot1 nao --robot2 nao --green_screen --camera_mode exoL --bb_mode1
python extract_bb.py --interaction $1 --robot1 nao --robot2 nao --green_screen --camera_mode exoL --bb_mode2
python extract_bb.py --interaction $1 --robot1 nao --robot2 nao --green_screen --camera_mode ego1L --bb_mode1
python extract_bb.py --interaction $1 --robot1 nao --robot2 nao --green_screen --camera_mode ego2L --bb_mode2
python extract_bb.py --interaction $1 --robot1 nao --robot2 nao --green_screen --camera_mode ego1R --bb_mode1
python extract_bb.py --interaction $1 --robot1 nao --robot2 nao --green_screen --camera_mode ego2R --bb_mode2




echo "Script completed!"