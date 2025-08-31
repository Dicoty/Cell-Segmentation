from ultralytics import YOLO

model = YOLO('yolov11n.yaml', task='segment')

data = "C:\somefiles\weights_and_dataset\yolo_anno\data.yaml"
results = model.train(data='data.yaml', epochs=100, batch=8, imgsz=640)