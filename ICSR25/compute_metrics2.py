import json
import numpy as np
import pandas as pd


    
data = {"Robot":[],"LLM":[],"Pose":[],"Avg1":[], "Avg2":[],"Sum1":[], "Sum2":[]}
data = {"Robot":[],"LLM":[],"Pose":[], "Error":[], "Joint":[]}

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
                if k in pred_pose and v!= 0:
                    pred_value = pred_pose[k]
                    pred_pose_array.append(pred_value) #Considers all the joints

                    error = np.abs(v - pred_value)
                    data["Robot"].append(robot)
                    data["Pose"].append(p)
                    data["LLM"].append(llm)
                    data["Joint"].append(k)
                    data["Error"].append(error)           
                    
                    
                    
                   

            
         

pd.DataFrame(data).to_csv("analysis_joints.csv", index=False)
        
    
    