import onnx
import onnxruntime
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    model =     torch.load("yolov8n.pt")
    model_yolov8n_cla = torch.load("runs/classify/yolov8n_compcars_cla/weights/best.pt")

    my_model = YOLO("runs/classify/yolov8n_compcars_cla/weights/best.pt")

    for name, param in my_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    a = 1


    # onnx_model = onnx.load("models/onnx/yolov8_det_cla_best.onnx")
    # onnx.checker.check_model(onnx_model)
    # a = 1
    #
    # ort_session = onnxruntime.InferenceSession("models/onnx/yolov8_det_cla_best.onnx")
    #
    # # compute ONNX Runtime output prediction
    #
    # imgx = Image.open("datasets/blimp/0000/1.png")
    # x = np.asarray(imgx)
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # ort_outs = ort_session.run(None, ort_inputs)
    #
    # a = 1
    #
    # # compare ONNX Runtime and PyTorch results
    # # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
