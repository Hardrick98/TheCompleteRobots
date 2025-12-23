import joblib
import os
from tqdm import tqdm


def fix(data):

    for c in ["exoR","exoL","ego1R","ego2R","ego1L","ego2L"]:
        poses1_2d = data[c]['pose2D']
        poses1_3d = data[c]['pose3D']
        poses1_2d_total = data[c]['pose2D_total']
        poses1_3d_total = data[c]['pose3D_total']

        poses1_2d[:,5,:] = poses1_2d_total[:,16,:]
        poses1_2d[:,11,:] = poses1_2d_total[:,24,:]
        poses1_3d[:,5,:] = poses1_3d_total[:,16,:]
        poses1_3d[:,11,:] = poses1_3d_total[:,24,:]

        data[c]['pose2D'] = poses1_2d
        data[c]['pose3D'] = poses1_3d

    return data

PATH = "/leonardo_scratch/large/userexternal/rcatalin/robot_dataset"

for i in tqdm(os.listdir("/leonardo_scratch/large/userexternal/rcatalin/robot_dataset")):
    
    data1 = joblib.load(f"{PATH}/{i}/icub/data/icub_1_data.pkl")
    data1 = fix(data1)
    data1 = joblib.dump(data1, f"{PATH}/{i}/icub/data/icub_1_data.pkl")

    data2 = joblib.load(f"{PATH}/{i}/icub/data/icub_2_data.pkl")
    data2 = fix(data2)
    data2 = joblib.dump(data2, f"{PATH}/{i}/icub/data/icub_2_data.pkl")

print("Done!")