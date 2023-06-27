from ultralytics import YOLO
from roboflow import Roboflow

# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

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

def train_classification():
    model = YOLO("models/yolov8n-cls.pt")

    #local - veriwild
    # results = model.train(
    #    data='/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/yolov8_cla_resized_mapping_dataset_debug',
    #    # data='/data/veri-wild/veri-wild1/yolov8_resized_dataset/data.yaml',
    #    imgsz=640,
    #    epochs=10,
    #    name='yolov8n_vehicle_cla')

    # remote  - veriwild
    # results = model.train(
    #     # data='/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/yolov8_cla_resized_mapping_dataset_debug',
    #     data='/data/veri-wild/veri-wild1/yolov8_cla_resized_mapping_dataset',
    #     imgsz=640,
    #     epochs=100,
    #     name='yolov8n_vehicle_cla')

    # local - compcars
    results = model.train(
       data='/media/hao/Seagate Basic/dataset/compcars/compcars_torchvision_debug/data/image_splitted',
       # data='/data/veri-wild/veri-wild1/yolov8_resized_dataset/data.yaml',
       imgsz=640,
       epochs=2,
       patience = 2,
       batch = 4,
       name='yolov8n_compcars_cla')

    # remote  - compcars
    results = model.train(
        data='/data/compcars/compcars_torchvision/data/image_splitted',
        imgsz=640,
        epochs=50,
        patience = 15,
        batch = 32,
        name='yolov8n_compcars_cla')

def val():
    #local
    # model = YOLO('runs/detect/yolov8n_vehicle14/weights/best.pt')  # load a custom model
    # remote debug
    model = YOLO('runs/detect/yolov8n_vehicle/weights/best.pt')  # load a custom model

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
    model = YOLO('runs/detect/yolov8n_vehicle/weights/best.pt')  # load a custom model
    # results = model.predict(show=True, save=True, save_txt=True, source="datasets/0616.jpg")


    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/1.png")
    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/2.png")
    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/3.png")
    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/4.png")

    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/0000/1.png")
    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/0000/2.png")
    results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/0000/3.png")


    # results = model(im)
    # print(results[0].probs)  # cls prob, (num_class, )

    # res = model(img)
    # res_plotted = results[0].plot()
    # cv2.imshow("result", res_plotted)

    a = 1

def predict_classification():
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
    model = YOLO('runs/classify/yolov8n_vehicle_cla/weights/best.pt')  # load a custom model
    # results = model.predict(show=True, save=True, save_txt=True, source="datasets/0616.jpg")


    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/1.png")
    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/2.png")
    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/3.png")
    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/4.png")

    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/0000/1.png")
    # results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/0000/2.png")
    results = model.predict(show=False, save=True, save_txt=True, source="datasets/blimp/0000/2.png")


    # results = model(im)
    # print(results[0].probs)  # cls prob, (num_class, )

    # res = model(img)
    # res_plotted = results[0].plot()
    # cv2.imshow("result", res_plotted)

    a = 1


def convert():
    model = YOLO('runs/detect/yolov8n_vehicle/weights/best.pt')
    # model.export(format='onnx')
    model.export(format='TensorRT', device=0)
    a = 1


if __name__ == "__main__":
    '''YOLOV8 det + classification model'''
    # train()
    # val()
    # predict()
    # convert()

    '''YOLOV8 ONLY classification model'''
    train_classification()
    # predict_classification()