from ultralytics import YOLO
from roboflow import Roboflow

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

def train():
    # from roboflow import Roboflow
    # rf = Roboflow(api_key="ivO9dgiYc3AQpHvRWBHC")
    # project = rf.workspace("politecnico-di-milano-iuz9a").project("vehicle_det_debug")
    # dataset = project.version(1).download("yolov8")

    model = YOLO("models/yolov8n.pt")

    results = model.train(
       # data='/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/vehicle_det_debug-1/data.yaml',
       # data='/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/yolov8_resized_dataset_debug/data.yaml',
       # data='/data/veri-wild/veri-wild1_debug/vehicle_det_debug-1/data.yaml',
        data='/data/veri-wild/veri-wild1/yolov8_resized_dataset/data.yaml',
       # data=dataset,
       imgsz=640,
       epochs=100,
       #batch=8,
       name='yolov8n_vehicle')

def val():
    #local
    # model = YOLO('runs/detect/yolov8n_vehicle14/weights/best.pt')  # load a custom model
    # remote debug
    model = YOLO('runs/detect/yolov8n_vehicle6/weights/best.pt')  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

def predict():
    # im = cv2.imread("1.jpg")
    # res_plotted = im.plot()
    # cv2.imshow("result", im)
    # cv2.waitKey(0)

    # im = cv2.imread("/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1/images/00273/002980.jpg")
    # im = cv2.imread("/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1/images/00001/000001.jpg")
    # print(im.shape)

    # local
    # model = YOLO('runs/detect/yolov8n_vehicle14/weights/best.pt')
    # results = model.predict(show=True, save=True, save_txt=True, source="datasets/0616.jpg")
    # results = model.predict(show=True, save=True, save_txt=True, source="datasets/1.jpg")
    # remote
    model = YOLO('runs/detect/yolov8n_vehicle6/weights/best.pt')  # load a custom model
    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/0616.jpg")
    results = model.predict(show=False, save=True, save_txt=True, source="datasets/1.jpg")



    # results = model(im)
    # print(results[0].probs)  # cls prob, (num_class, )

    # res = model(img)
    # res_plotted = results[0].plot()
    # cv2.imshow("result", res_plotted)

    a = 1


if __name__ == "__main__":
    train()
    # val()
    # predict()