import json
import numpy as np


with open('json/GT_khr3hv.json', 'r') as f:
    gt = json.load(f)

for llm in ["chat", "claude", "gemini", "deepseek"]:
    
    with open(f'json/{llm}_khr3hv.json', 'r') as f:
        pred = json.load(f)
        
    pose_type = "superhero"
        
    gt_pose = gt[0][pose_type]
    pred_pose =  pred[0][pose_type]

    gt_pose_array = []
    pred_pose_array = []

    for k, v in gt_pose.items():
        gt_pose_array.append(v)
        pred_pose_array.append(pred_pose[k])
        
    gt_pose = np.array(gt_pose_array)
    pred_pose = np.array(pred_pose_array)

    total_error = np.sum(np.abs(gt_pose - pred_pose), axis=0)

    #indices = np.nonzero(gt_pose) 
    #total_error = np.sum(gt_pose[indices] - pred_pose[indices], axis=0)

    print(f"{llm}: ", total_error)