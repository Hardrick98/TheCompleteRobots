from ultralytics import YOLO


#model = YOLO("/leonardo_work/IscrC_SDG-GS/TheCompleteRobots/runs/detect/train/weights/last.pt")  # pretrained YOLO11n model

model = YOLO("yolo11n.pt")

# Run batched inference on a list of images
results = model(["G041T005A012R010_atlas_exoR_frame_00000.png"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename="result.jpg")  # save to disk