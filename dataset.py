import pandas as pd 
import os
import random
import joblib

scenes = [i for i in os.listdir("scenes/") if i.endswith(".glb")]
robots = ["nao", "pepper", "g1", "atlas", "icub"]

labels = ["Hug", "Hit", "Kick", "Push", "Pull", "Slap", "Pat on back", "Step on foot", "Link arms", "Pat on cheek","Touch head","Wave", 
"Point finger at","Chase", "Rock-paper-scissors", "Bend", "Chat", "Thumb up","Imitate","Look back"]

interactions = []

actions = joblib.load("actions.pkl")



for l in labels:    
    i = 0
    while i < 10:
        label_action = actions[l]
        n = random.randint(0, len(label_action)-1)
        interactions.append(label_action[n])
        label_action.pop(n)
        i+= 1

num_samples = len(interactions)*len(robots)

scenes_shuffled = [j for _ in range((num_samples//len(scenes))+1) for j in scenes]
scenes_shuffled.pop(-1)


robot_new = [r for _ in range(num_samples // len(robots)) for r in robots]

interactions_final = [i for i in interactions for _ in range(len(robots))]


random.shuffle(scenes_shuffled)


table = {"interaction": interactions_final, "robot":robot_new, "scenes":scenes_shuffled}

df = pd.DataFrame(table)
df.to_csv("dataset.csv")