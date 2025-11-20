import os
import joblib
import shutil
from tqdm import tqdm



def xyxy_to_xywhn(bbox):
    
    size = [1280,720]
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    xc = (x1 + x2)/2
    yc = (y1 + y2)/2

    x = xc / size[0]
    y = yc / size[1]
    width = width / size[0]
    height = height / size[1]

    return x, y, width, height

path = "/leonardo_scratch/large/userexternal/rcatalin/robot_dataset2/"
interactions = os.listdir(path)

for interaction in tqdm(interactions):

    interaction_path = f"{path}/{interaction}/"
    robot_list = [f for f in os.listdir(interaction_path) if not f.endswith('.npz')]


    for r in robot_list:

        

            
        data1 = joblib.load(f"{interaction_path}/{r}/data/{r}_2_data.pkl")
        data2 = joblib.load(f"{interaction_path}/{r}/data/{r}_1_data.pkl")
        
        images = os.listdir(f"{interaction_path}/{r}/exoR")
        images = [img for img in images if int(img.removeprefix("frame_").removesuffix(".png"))%60==0]
        images = sorted(images,key=lambda x: int(x.removeprefix("frame_").removesuffix(".png")))

        bboxes1 = data1['exoR']['bb2D']
        bboxes2 = data2['exoR']['bb2D']

        for i in range(len(images)):

           image_path = os.path.join(f"{interaction_path}/{r}/exoR",images[i])
           image_name = f"{interaction}_{r}_exoR_{images[i]}"
           image_dst_path = os.path.join(f"/leonardo_scratch/large/userexternal/rcatalin/robot_detection/images/{image_name}")

           shutil.copy(image_path,image_dst_path)
           num = int(images[i].removeprefix("frame_").removesuffix(".png"))
           bbox = bboxes1[num]     
           if bbox[0] == -1:
               continue
           x1, y1, x2, y2 = bbox
           x,y,w,h = xyxy_to_xywhn(bbox)
           label1 = f"0 {x} {y} {w} {h}\n"

           bbox = bboxes2[num]     
           if bbox[0] == -1:
               continue
           x1, y1, x2, y2 = bbox
           x,y,w,h = xyxy_to_xywhn(bbox)
           label2 = f"0 {x} {y} {w} {h}\n"

           label = label1 + label2

           txt_name = image_name.replace('.png','.txt')
           txt_dst_path = os.path.join(f"/leonardo_scratch/large/userexternal/rcatalin/robot_detection/labels/{txt_name}")

           with open(txt_dst_path,'w') as f: 
               f.write(label)
            
           


        """""
        bboxes = data1['ego1R']['bb2D']

        for i in range(len(images)):

           image_path = os.path.join(f"{interaction_path}/{r}/ego1R",images[i])
           image_name = f"{interaction}_{r}_ego1R_{images[i]}"
           image_dst_path = os.path.join(f"/leonardo_scratch/large/userexternal/rcatalin/robot_detection/images/{image_name}")

           shutil.copy(image_path,image_dst_path)
           num = int(images[i].removeprefix("frame_").removesuffix(".png"))
           bbox = bboxes[num]     
           if bbox[0] == -1:
               continue
           x1, y1, x2, y2 = bbox
           x,y,w,h = xyxy_to_xywhn(bbox)
           label = f"0 {x} {y} {w} {h}\n"
           txt_name = image_name.replace('.png','.txt')
           txt_dst_path = os.path.join(f"/leonardo_scratch/large/userexternal/rcatalin/robot_detection/labels/{txt_name}")

           with open(txt_dst_path,'w') as f: 
               f.write(label)
            
           f.close()
        



        data2 = joblib.load(f"{interaction_path}/{r}/data/{r}_1_data.pkl")

        images = os.listdir(f"{interaction_path}/{r}/ego2R")
        images = [img for img in images if int(img.removeprefix("frame_").removesuffix(".png"))%60==0]
        images = sorted(images,key=lambda x: int(x.removeprefix("frame_").removesuffix(".png")))

        bboxes = data2['ego2R']['bb2D']

        for i in range(len(images)):

           image_path = os.path.join(f"{interaction_path}/{r}/ego2R",images[i])
           image_name = f"{interaction}_{r}_ego2R_{images[i]}"
           image_dst_path = os.path.join(f"/leonardo_scratch/large/userexternal/rcatalin/robot_detection/images/{image_name}")

           num = int(images[i].removeprefix("frame_").removesuffix(".png"))

           shutil.copy(image_path,image_dst_path)

           bbox = bboxes[num]     
           if bbox[0] == -1:
               continue
           x1, y1, x2, y2 = bbox
           x,y,w,h = xyxy_to_xywhn(bbox)
           label = f"0 {x} {y} {w} {h}\n"
           txt_name = image_name.replace('.png','.txt')
           txt_dst_path = os.path.join(f"/leonardo_scratch/large/userexternal/rcatalin/robot_detection/labels/{txt_name}")

           with open(txt_dst_path,'w') as f: 
               f.write(label)
            
           f.close()

            
        """


import os

main_path = '/leonardo_scratch/large/userexternal/rcatalin/robot_detection/'
images_path = os.path.join(main_path, 'images')
labels_path = os.path.join(main_path, 'labels')


labels = [f for f in os.listdir(labels_path)
          if os.path.isfile(os.path.join(labels_path, f)) and f.endswith('.txt')]
import random
random.shuffle(labels)

split_idx = int(0.8 * len(labels))
train_labels = labels[:split_idx]
val_labels = labels[split_idx:]

for i in train_labels:
    os.rename(os.path.join(labels_path,i), os.path.join(labels_path,"train",i))
    i = i.removesuffix('.txt') + '.png'
    os.rename(os.path.join(images_path,i), os.path.join(images_path,"train",i))
    

for i in val_labels:
    os.rename(os.path.join(labels_path,i), os.path.join(labels_path,"val",i))
    i = i.removesuffix('.txt') + '.png'
    os.rename(os.path.join(images_path,i), os.path.join(images_path,"val",i))

print("Done!")

"""

import cv2
image = "/leonardo_scratch/large/userexternal/rcatalin/robot_detection/images/G008T001A005R000_atlas_ego1R_frame_00000.png"
label = "/leonardo_scratch/large/userexternal/rcatalin/robot_detection/labels/G008T001A005R000_atlas_ego1R_frame_00000.txt"

img = cv2.imread(image)
with open(label,'r') as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        cls = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        img_h, img_w, _ = img.shape

        x1 = int((x - w/2) * img_w)
        y1 = int((y - h/2) * img_h)
        x2 = int((x + w/2) * img_w)
        y2 = int((y + h/2) * img_h)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imwrite("output.png", img)
"""