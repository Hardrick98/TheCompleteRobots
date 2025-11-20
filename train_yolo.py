from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

results = model.train(data="robot_detection.yaml", epochs=30, imgsz=640)