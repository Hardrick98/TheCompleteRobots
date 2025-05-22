import json
import numpy as np
import pandas as pd


    
data = {"Robot":[],"LLM":[],"Pose":[],"Error":[]}

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
                
            for k, v in gt_pose.items():
                gt_pose_array.append(v)
                if k in pred_pose:
                    pred_value = pred_pose[k]
                    pred_pose_array.append(pred_value if v != 0 else 0)
                else:
                    pred_pose_array.append(0)
                   
                
            
            gt_pose = np.array(gt_pose_array)
            pred_pose = np.array(pred_pose_array)

            total_error = np.sum(np.abs(gt_pose - pred_pose), axis=0)
            
            data["Robot"].append(robot)
            data["Pose"].append(p)
            data["LLM"].append(llm)
            data["Error"].append(total_error)
            


pd.DataFrame(data).to_csv("analysis.csv", index=False)
        
    
    