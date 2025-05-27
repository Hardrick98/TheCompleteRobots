import numpy as np
from URDF.utils import load_robot

array = np.load("/datasets/HumanoidX/humanoid_action/kinetics700/p8g4XYpFTu0_000018_000028_clip_1_sample.npy", allow_pickle=True)

sample = dict(array[()])

print(sample.keys())

print(sample["dof_pos"][0])

