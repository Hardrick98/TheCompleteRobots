import os
import joblib
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

i = "/leonardo_scratch/large/userexternal/rcatalin/robot_dataset/G024T000A000R001/"
for r in ["g1", "nao", "pepper","atlas","icub"]:

    for c in ["exoR", "exoL", "ego1R", "ego2R", "ego1L", "ego2L"]:

        
        data1 = joblib.load(f"{i}/{r}/data/{r}_1_data.pkl")
        poses1 = data1[c]['pose2D']
        poses1_3d = data1[c]['pose3D']

        mask1 = (poses1_3d[:,:,2] > 0)
        poses1[mask1] = [-1,-1]

        data2 = joblib.load(f"{i}/{r}/data/{r}_2_data.pkl")
        poses2 = data2[c]['pose2D']
        poses2_3d = data2[c]['pose3D']

        mask2 = (poses2_3d[:,:,2] > 0)
        poses2[mask2] = [-1,-1]

        data1[c]['pose2D'] = poses1
        data2[c]['pose2D'] = poses2

        joblib.dump(data1, f"{i}/{r}/data/{r}_1_data.pkl")
        joblib.dump(data2, f"{i}/{r}/data/{r}_2_data.pkl")


