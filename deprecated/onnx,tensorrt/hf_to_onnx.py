import torch
import onnx
from transformers import AutoConfig, AutoModelForQuestionAnswering
import numpy as np

model_name= './model/checkpoint-28500'  # pytorch_model.bin이 저장된 위치
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name, from_tf=bool(".ckpt" in model_name), config=config,
).cuda()


input_ids = torch.as_tensor(np.ones([1, 512]), dtype=torch.long).cuda()
attention_mask = torch.as_tensor(np.ones([1, 512]), dtype=torch.long).cuda()
#token_type_ids = torch.as_tensor(np.ones([1, 512]), dtype=torch.long).cuda() # roberta 계열의 경우 token_type_ids 없음

torch.onnx.export(
    model=model,
    args = (input_ids, attention_mask),# token_type_ids),
    f="./onnx_model.onnx",
    input_names=['input_ids', 'attention_mask'],#, 'token_type_ids'],
    output_names=['outputs'],
    opset_version=11,
    export_params=True,
)
onnx_model = onnx.load("onnx_model.onnx")