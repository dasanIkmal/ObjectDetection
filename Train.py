from ultralytics import YOLO

# Load a model
model = YOLO("YOLOv8x.yaml")  # build a new model from scratch

# Use the model
model.train(data="config.yml", epochs=500)  # train the model