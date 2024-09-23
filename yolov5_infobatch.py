from ultralytics import YOLO

# Load a model
model = YOLO("yolov5.yaml")  # build a new model from scratch

# Use the model
model.train(data="coco.yaml", infobatch=True, epochs=100, imgsz=640, hyp='yolov5_training.yaml')  # train the model
metrics = model.val()