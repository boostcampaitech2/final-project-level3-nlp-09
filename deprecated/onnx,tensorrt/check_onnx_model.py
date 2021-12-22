import sys
import onnx
import onnxruntime as ort 
filename = 'model/onnx/KLRL-QA.onnx'
model = onnx.load(filename)
print(onnx.checker.check_model(model))
print(onnx.helper.printable_graph(model.graph))

sess_ort = ort.InferenceSession(filename) 
res = sess_ort.run(output_names=[output_tensor.name], input_feed={input_tensor.name: img}) 
print("the expected result is \"7\"") 
print("the digit is classified as \"%s\" in ONNXRruntime"%np.argmax(res))