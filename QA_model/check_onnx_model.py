import sys
import onnx
filename = 'model/onnx/KLRL-QA.onnx'
model = onnx.load(filename)
print(onnx.checker.check_model(model))