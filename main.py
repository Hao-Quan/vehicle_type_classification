# from ultralytics import YOLO
#
# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
#
# # Train the model
# model.train(data='coco128.yaml', epochs=100, imgsz=640)
#

from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8x.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("/media/hao/My Passport/dataset/Veri-Wild/images/00001/000002.jpg")
# im1 = Image.open("/media/hao/My Passport/dataset/blimp/test_images/vehicles/test_1/0000.jpg")
#
results = model.predict(source=im1, save=True)  # save plotted images

# from roboflow import Roboflow
# rf = Roboflow(api_key="ivO9dgiYc3AQpHvRWBHC")
# project = rf.workspace("politecnico-di-milano-iuz9a").project("vehicle-classification-eo8bn")
# dataset = project.version(1).download("folder")

# results = model.train(
#    data='config.yaml',
#    # data=dataset,
#    imgsz=640,
#    epochs=10,
#    batch=1,
#    name='yolov8n_custom')

a = 1

# # from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
#
# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])
