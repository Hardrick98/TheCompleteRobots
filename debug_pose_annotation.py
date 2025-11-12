import cv2
import os
import joblib
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str)
parser.add_argument("-c", type=str)
parser.add_argument("-r", type=str)
args = parser.parse_args()


collisions = joblib.load(f"{args.i}/{args.r}/data/{args.r}_{args.r}_collisions.pkl")

data1 = joblib.load(f"{args.i}/{args.r}/data/{args.r}_1_data.pkl")
poses1 = data1[args.c]['pose2D']

if "ego1" not in args.c:
    boxes1 = data1[args.c]['bb2D']

data2 = joblib.load(f"{args.i}/{args.r}/data/{args.r}_2_data.pkl")

poses2 = data2[args.c]['pose2D']
if "ego2" not in args.c:
    boxes2 = data2[args.c]['bb2D']


frames = [f"{args.i}/{args.r}/{args.c}/{j}" for j in os.listdir(f"{args.i}/{args.r}/{args.c}")]

frames.sort()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"{args.r}_{args.c}.mp4", fourcc, 120, (1280, 720))
for i in tqdm(range(len(frames))):
        
    image = cv2.imread(frames[i])

    if len(collisions[i])>0:
        cv2.putText(image, text="COLLISION", org=(500,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,255), thickness=4)

    for p in poses1[i]:
        cv2.circle(image,center=(int(p[0]),int(p[1])),radius=2,color=(0,0,255),thickness=2)
    
    if "ego1" not in args.c:
        box1 = boxes1[i,:]
        cv2.rectangle(image,pt1=(int(box1[0]), int(box1[1])),pt2=(int(box1[2]), int(box1[3])),color=(255,0,0), thickness=2)


    for p in poses2[i]:
        cv2.circle(image,center=(int(p[0]),int(p[1])),radius=2,color=(0,0,255),thickness=2)
    
    if "ego2" not in args.c:
        box2 = boxes2[i,:]
        cv2.rectangle(image,pt1=(int(box2[0]), int(box2[1])),pt2=(int(box2[2]), int(box2[3])),color=(0,255,0), thickness=2)


    out.write(image)
 

out.release()

