#from yolo_attnopt.attention_optimization import setup_optimized_yolo_environ
#setup_optimized_yolo_environ(optimize_level=2)

#---- Start using YOLO ----
import os
from ultralytics import YOLO

# YOLO params
image_path = 'assets/ivan-rohovchenko-iNPI5VlSt4o-unsplash.jpg'
imgsz = 2560
yolo_modelname = 'yolo11x.pt'
device = 'cuda:0'

# Load and initialize YOLO model
model = YOLO(yolo_modelname)
model = model.to(device)
assert os.path.exists(image_path)
results = model(image_path, imgsz=imgsz, save=True)
