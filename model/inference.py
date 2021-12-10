
import pandas as pd
import numpy as np
import random
import os

from sklearn.model_selection import train_test_split
from torch import optim
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import AdamW, Trainer, TrainingArguments

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_metric

# with open('./basketball.txt', encoding= 'utf-8') as f:
#     sample_passage= f.load()
# print(sample_passage)

sample_question= '이것은 한 팀에 5명이 플레이하나요?'

model_name= 'klue/roberta-large'
config= AutoConfig.from_pretrained(model_name)
tokenizer= AutoTokenizer.from_pretrained(model_name)
model= AutoModelForSequenceClassification.from_pretrained(model_name)


best_state_dict= torch.load(os.path.join(f'/opt/ml/boolean_project/best_model/checkpoint-1000/pytorch_model.bin'))
model.load_state_dict(best_state_dict)

tokenized_data= tokenizer(
    sample_question,
    sample_passage,
    truncation= 'only_second',
    max_length= 512,
    padding= 'max_length',
    return_tensors= 'pt'
    )

print(tokenized_data)