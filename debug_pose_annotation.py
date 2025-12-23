import cv2
import os
import joblib
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str)
parser.add_argument("-c", type=str)
parser.add_argument("-r", type=str)
args = parser.parse_args()


collisions = joblib.load(f"{args.i}/{args.r}/data/{args.r}_{args.r}_collisions.pkl")

data1 = joblib.load(f"{args.i}/{args.r}/data/{args.r}_1_data.pkl")
poses1 = data1[args.c]['pose2D']
poses1_3d = data1[args.c]['pose3D']

mask1 = (poses1_3d[:,:,2] > 0)
poses1[mask1] = [-1,-1]

if "ego1" not in args.c:
    boxes1 = data1[args.c]['bb2D']

data2 = joblib.load(f"{args.i}/{args.r}/data/{args.r}_2_data.pkl")

poses2 = data2[args.c]['pose2D']
poses2_3d = data2[args.c]['pose3D']

mask2 = (poses2_3d[:,:,2] > 0)
poses2[mask2] = [-1,-1]


if "ego2" not in args.c:
    boxes2 = data2[args.c]['bb2D']


frames = [f"{args.i}/{args.r}/{args.c}/{j}" for j in os.listdir(f"{args.i}/{args.r}/{args.c}")]

frames.sort()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"{args.r}_{args.c}.mp4", fourcc, 120, (1280, 720))
for i in range(len(frames)):
        
    image = cv2.imread(frames[i])

    if len(collisions[i])>0:
        cv2.putText(image, text="COLLISION", org=(500,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,255), thickness=4)

    if "ego1" not in args.c:
        
        for p in poses1[i]:
            cv2.circle(image,center=(int(p[0]),int(p[1])),radius=2,color=(0,0,255),thickness=2)
        box1 = boxes1[i,:]
        cv2.rectangle(image,pt1=(int(box1[0]), int(box1[1])),pt2=(int(box1[2]), int(box1[3])),color=(255,0,0), thickness=2)

    if "ego2" not in args.c:
        for p in poses2[i]:
            cv2.circle(image,center=(int(p[0]),int(p[1])),radius=2,color=(0,0,255),thickness=2)
        box2 = boxes2[i,:]
        cv2.rectangle(image,pt1=(int(box2[0]), int(box2[1])),pt2=(int(box2[2]), int(box2[3])),color=(0,255,0), thickness=2)


    out.write(image)


 

out.release()

