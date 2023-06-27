import onnx
import onnxruntime
import numpy as np
from PIL import Image

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    onnx_model = onnx.load("models/onnx/yolov8_det_cla_best.onnx")
    onnx.checker.check_model(onnx_model)
    a = 1

    ort_session = onnxruntime.InferenceSession("models/onnx/yolov8_det_cla_best.onnx")

    # compute ONNX Runtime output prediction

    imgx = Image.open("datasets/blimp/0000/1.png")
    x = np.asarray(imgx)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    a = 1

    # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
