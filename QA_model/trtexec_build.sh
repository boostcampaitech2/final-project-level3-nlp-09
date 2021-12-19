# trtexec --onnx=./model/onnx_cuda/onnx_cuda.onnx --workspace=8000 \
# --maxBatch=12 --saveEngine=./export/best.trt \
# --best --verbose 

trtexec --onnx=./model/onnx/onnx_model.onnx --workspace=8000 \
--maxBatch=12 --saveEngine=./export/fp16_test.trt \
--fp16 --verbose

# trtexec --onnx=./model/onnx/KLRL-QA.onnx --workspace=8000 \
# --maxBatch=12 --saveEngine=./export/int8.trt \
# --best --verbose
#   --fp16                      Enable fp16 precision, in addition to fp32 (default = disabled)
#   --int8                      Enable int8 precision, in addition to fp32 (default = disabled)
#   --best