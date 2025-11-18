import os
import joblib

path = "/leonardo_scratch/large/userexternal/rcatalin/robot_dataset/"
interaction = os.listdir(path)

for i in interaction:

    interaction_path = f"{path}/{i}/"
    if "pepper" in os.listdir(interaction_path):
        robot_list = ["g1", "nao", "pepper","atlas","icub"]
    else:
        robot_list = ["g1", "nao","atlas","icub"]

    for r in robot_list:

        for c in ["exoR", "exoL", "ego1R", "ego2R", "ego1L", "ego2L"]:

            
            data1 = joblib.load(f"{interaction_path}/{r}/data/{r}_1_data.pkl")
            poses1 = data1[c]['pose2D']
            poses1_3d = data1[c]['pose3D']

            mask1 = (poses1_3d[:,:,2] > 0)

            poses1 = poses1.copy()
            poses1[mask1, :] = -1

            data2 = joblib.load(f"{interaction_path}/{r}/data/{r}_2_data.pkl")
            poses2 = data2[c]['pose2D']
            poses2_3d = data2[c]['pose3D']

            mask2 = (poses2_3d[:,:,2] > 0)
            poses2 = poses2.copy()
            poses2[mask2, :] = -1

            data1[c]['pose2D'] = poses1
            data2[c]['pose2D'] = poses2

            joblib.dump(data1, f"{interaction_path}/{r}/data/{r}_1_data.pkl")
            joblib.dump(data2, f"{interaction_path}/{r}/data/{r}_2_data.pkl")


