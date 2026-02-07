from ultralytics import YOLO

model = YOLO("yolov8n.pt")
names = model.names  # dict id->name
for k in sorted(names.keys()):
    print(k, names[k])
