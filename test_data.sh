echo "Starting Retargeting"

python retarget_motion.py --robot $3 --interaction $1
python compute_data.py --robot1 $3 --robot2 $3 --interaction $1

echo "Rendering videos..."

python render.py --interaction $1 --robot1 $3 --robot2 $3 --scene $2 --video $1 --camera_mode $4
