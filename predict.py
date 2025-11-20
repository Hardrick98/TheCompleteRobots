from ultralytics import YOLO


model = YOLO("/leonardo_work/IscrC_SDG-GS/TheCompleteRobots/runs/detect/train/weights/last.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["nao.jpg", "nao2.jpg", "g1.jpg", "naos.png"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename="result.jpg")  # save to disk