import json
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Compute metrics for LLMs')
parser.add_argument('--pose', type=str, default='chat', help='LLM to compute metrics for')
args = parser.parse_args()

with open('json/GT_atlas.json', 'r') as f:
    gt = json.load(f)

for llm in ["chat", "claude", "gemini", "deepseek"]:
    
    with open(f'json/{llm}_atlas.json', 'r') as f:
        pred = json.load(f)
        
    pose_type = args.pose
        
    gt_pose = gt[0][pose_type]
    pred_pose =  pred[0][pose_type]

    gt_pose_array = []
    pred_pose_array = []

    for k, v in gt_pose.items():
        gt_pose_array.append(v)
        
        try:
            if v == 0:
                pred_pose_array.append(0)
            else:
                pred_pose_array.append(pred_pose[k])
        except:
            pred_pose_array.append(0)
    gt_pose = np.array(gt_pose_array)
    pred_pose = np.array(pred_pose_array)

    total_error = np.sum(np.abs(gt_pose - pred_pose), axis=0)

    #indices = np.nonzero(gt_pose) 
    #total_error = np.sum(gt_pose[indices] - pred_pose[indices], axis=0)

    print(f"{llm}: ", total_error)