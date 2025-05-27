import json
import numpy as np
import pandas as pd


    
data = {"Robot":[],"LLM":[],"Pose":[],"Avg1":[], "Avg2":[],"Sum1":[], "Sum2":[]}
data = {"Robot":[],"LLM":[],"Pose":[], "GT_count":[], "Pred_count":[]}

for robot in ["nao", "atlas","h12","khr3hv"]:
    with open(f'json/GT_{robot}.json', 'r') as f:
        gt = json.load(f)
    for p in ["T_pose","salute","superhero","sit"]:
        for llm in ["chat", "claude", "gemini", "deepseek"]:
            
            with open(f'json/{llm}_{robot}.json', 'r') as f:
                pred = json.load(f)
            
            pose_type = p
            gt_pose = gt[0][pose_type]
            pred_pose =  pred[0][pose_type]

            gt_pose_array = []
            pred_pose_array = []
            pred_pose_array2 = []
                
            for k, v in gt_pose.items():
                gt_pose_array.append(v)
                if k in pred_pose:
                    pred_value = pred_pose[k]
                    pred_pose_array.append(pred_value) #Considers all the joints
                else:
                    pred_pose_array.append(0)
                    
                   
            gt_pose = np.array(gt_pose_array)
            pred_pose = np.array(pred_pose_array)
            
            mask = (gt_pose != 0)

            if np.sum(mask) == 0:
                total_error_mean = 0
                total_error_sum = 0
            else:
                total_error_sum = np.sum(np.abs(gt_pose[mask] - pred_pose[mask]))
                total_error_mean = np.mean(np.abs(gt_pose[mask] - pred_pose[mask]))
            
            mask = (gt_pose != 0) | ((gt_pose == 0) & (pred_pose != 0)) #active or moved joints 
            if np.sum(mask) == 0:
                total_error_sum2 = 0
                total_error_mean2 = 0
            else:
                total_error_mean2 = np.mean(np.abs(gt_pose[mask] - pred_pose[mask]))
                total_error_sum2 = np.sum(np.abs(gt_pose[mask] - pred_pose[mask]))
            
            data["Robot"].append(robot)
            data["Pose"].append(p)
            data["LLM"].append(llm)
            data["Avg1"].append(total_error_mean)
            data["Avg2"].append(total_error_mean2)           
            data["Sum1"].append(total_error_sum)
            data["Sum2"].append(total_error_sum2) 
         

pd.DataFrame(data).to_csv("analysis_joints.csv", index=False)
        
    
    