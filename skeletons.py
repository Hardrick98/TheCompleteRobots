import numpy as np
import matplotlib.pyplot as plt
##Skeletons different annotations

dwpose_joints = {0:"nose", 1:"neck", 2:"rshoulder",3:"relbow",4:"rwrist",5:"lshoulder",6:"lelbow",
                   7:"lwrist",8:"rhip",9:"rknee",10:"rankle",11:"lhip",12:"lknee",13:"lankle",
                   14:"reye", 15:"leye", 16:"rear", 17:"lear"}

openpose17_joints = {"neck": 1, "nose": 0, "rshoulder": 2, "relbow": 3, "rhand" : 4, "lshoulder": 5, "lelbow": 6,
        "lhand":7, "rhip": 8, "rknee": 9, "rankle": 10, "lhip": 11, "lknee": 12, "lankle": 13, "reye":14,
         "leye":15, "rear": 16, "lear":17 }

human36m_joints = { 0: 'root', 1: 'rhip', 2: 'rkne', 3: 'rank', 4: 'lhip', 5: 'lkne', 6: 'lank',
    7: 'belly', 8: 'neck', 9: 'nose', 10: 'head', 11: 'lsho', 12: 'lelb', 13: 'lwri',
    14: 'rsho', 15: 'relb', 16: 'rwri'}

coco19_joints = {"neck": 0, "nose": 1, "pelv": 2, "lshoulder": 3, "lelbow": 4 ,"lwrist" : 5, "lhip": 6, "lknee": 7, 
                 "lankle":8, "rshoulder":9, "relbow":10, "rwrist":11, "rhip": 12, "rknee": 13, "rankle": 14, 
                 "leye":15, "lear":16, "reye": 17, "rear":18 }

egoexo_joints = {"nose": 0, "lshoulder": 5, "lelbow": 7 ,"lwrist" : 9, "lhip": 11, "lknee": 13, 
                 "lankle":15, "rshoulder":6, "relbow":8, "rwrist":10, "rhip": 12, "rknee": 14, "rankle": 16, 
                 "leye":1, "lear":3, "reye": 2, "rear": 4}

coco19_limbs = [[ 0 , 1], [0,3], [ 0 , 9], [ 0, 2], [1,15] ,[1,17] ,[3 ,4] , [4, 5],[ 6 , 7] ,
[ 6 , 2] , [ 7 , 8] , [ 9 ,10] , [10 ,11] , [12, 13] ,[12 , 2] , [13 ,14]
, [15 ,16] ,[15 ,17], [17 ,18] ]

kinect_limbs = [(18,19), (17,18), (16,17), (16,0), (0,12), (14,15), (13,14), (12, 13), (0,1), (1,20), (20,2), (2, 3), 
          (20, 4), (4, 5), (5,6), (20, 8), (8,9), (9,10), (10,11)]

kinect_joints = {
  "SPINEBASE": 0,
  "SPINEMID": 1,
  "NECK": 2,
  "HEAD": 3,
  "SHOULDERLEFT": 4,
  "ELBOWLEFT": 5,
  "WRISTLEFT": 6,
  "HANDLEFT": 7,
  "SHOULDERRIGHT": 8,
  "ELBOWRIGHT": 9,
  "WRISTRIGHT": 10,
  "HANDRIGHT": 11,
  "HIPLEFT": 12,
  "KNEELEFT": 13,
  "ANKLELEFT": 14,
  "FOOTLEFT": 15,
  "HIPRIGHT": 16,
  "KNEERIGHT": 17,
  "ANKLERIGHT": 18,
  "FOOTRIGHT": 19,
  "SPINESHOULDER": 20,
  "HANDTIPLEFT": 21,
  "THUMBLEFT": 22,
  "HANDTIPRIGHT": 23,
  "THUMBRIGHT": 24
}

dwpose_limbs = [
        (0, 1, "blue"), (1, 2, "blue"), (1, 5, "blue"), (5, 6, "green"), (6, 7, "green"), 
        (2, 3, "red"), (3, 4, "red"), (1, 8, "blue"), (1, 11, "blue"), (11, 12, "yellow"), 
        (12, 13, "yellow"), (0, 14, "black"), (8, 9, "purple"), (9, 10, "purple"), (0, 15, "black")] 
#(14,16,"black"),(15,17,"black")

human36m_limbs = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11),
    (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
]

egoexo_limbs = [[15, 13],[13, 11],[16, 14],[14, 12],[11, 12],[5, 11],[6, 12],
                [5, 6],[5, 7],[6, 8],[7, 9],[8, 10],[0, 1],[0, 2],
                [1, 3],[2, 4],[3, 5],[4, 6]]

skeletons = {"kinect":[kinect_joints,kinect_limbs],"openpose17": [openpose17_joints,None], "human36m": [human36m_joints,human36m_limbs], "dwpose": [dwpose_joints,dwpose_limbs], "coco19": [coco19_joints,coco19_limbs], "egoexo": [egoexo_joints, egoexo_limbs]  }


def plot_skeleton(pose, pose_estimator, default_color='blue', gaze=None):
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    color = "blue"
    skeleton = np.copy(pose)
    skeleton[:,1] = -skeleton[:,1]
    limbs = skeletons[pose_estimator][1]
    joints = skeletons[pose_estimator][0]
    for joint1_idx, joint2_idx in limbs:
        if all(len(joint) == 3 for joint in (skeleton[joint1_idx], skeleton[joint2_idx])):
            x_coords, y_coords, z_coords = zip(skeleton[joint1_idx], skeleton[joint2_idx])
            ax.plot(x_coords, z_coords, y_coords, c=color if default_color is None else default_color, linewidth=2)
    for joint in skeleton:
        ax.scatter(joint[0], joint[2], joint[1], c=color if default_color is None else default_color)

    if gaze == True:
        
        gaze = skeleton[-1]
        reye = joints["reye"]
        leye = joints["leye"]
        
        mid_eyes = (skeleton[reye]+skeleton[leye])/2
       
        
        x_coords = [gaze[0], mid_eyes[0]]
        y_coords = [gaze[1], mid_eyes[1]]
        z_coords = [gaze[2], mid_eyes[2]]
        
        gaze_color = None
        
        if default_color == "red":
            gaze_color = 'tomato'
        elif default_color == "green":
            gaze_color = 'lime'
        elif default_color == "blue":
            gaze_color = 'cyan'
        
        if gaze_color != None:
            ax.plot(x_coords, y_coords, z_coords, c=gaze_color, linewidth=2)
            ax.scatter(gaze[0],gaze[2],gaze[1], c=gaze_color)
            
        
        