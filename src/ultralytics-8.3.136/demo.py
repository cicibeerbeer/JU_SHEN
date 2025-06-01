from ultralytics import YOLO


model = YOLO("./yolov8n.pt", task="detect") 

results = model(source="./ultralytics/assets/bus.jpg",save = True,conf=0.05)