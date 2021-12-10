
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

sample_passage= open('./basketball.txt', encoding= 'utf-8').read()
# print(sample_passage)
sample_question= ['농구은 11명이서 플레이하나요?',
                '배고픈가요?',
                '농구는 졸린가요?',
                '농구는 4명인가요?',
                '짜장면인가요?',
                '농구은 미국에서 처음 시작되었나요?',
                '농구은 농구인가요?']
# sample_question= '이것은 한 팀에 5명이 플레이하나요?'
# sample_question=['돈가스는 음식인가요?',
#                 '돈가스은 과일인가요?',
#                 '돈가스는 고기인가요?',
#                 '돈가스은 고기인가요?',
#                 '돈가스는 서양 음식인가요?',
#                 '돈가스는 일본 음식인가요?']
model_name= 'klue/roberta-large'
config= AutoConfig.from_pretrained(model_name)
tokenizer= AutoTokenizer.from_pretrained(model_name)
model= AutoModelForSequenceClassification.from_pretrained(model_name)


# best_state_dict= torch.load(os.path.join(f'/opt/ml/boolean_project/model_epoch_4.pt'))
# model.load_state_dict(best_state_dict)
model= torch.load(os.path.join(f'/opt/ml/boolean_project/model_epoch_4.pt')).cpu()

for i in range(len(sample_question)):
    tokenized_data= tokenizer(
        sample_question[i],
        sample_passage,
        truncation= 'only_second',
        max_length= 512,
        padding= 'max_length',
        stride= 128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False, 
        return_tensors= 'pt'
        )

    result_list= []
    prob_list= []
    for j in range(len(tokenized_data['input_ids'])):

        outputs= model(
            input_ids= tokenized_data['input_ids'][j].unsqueeze(0),
            attention_mask= tokenized_data['attention_mask'][j].unsqueeze(0)
        )
        logits= outputs['logits']
        logits= F.softmax(logits, dim= -1)
        logits= logits.detach().cpu().numpy()
        prob= logits
        
        result= np.argmax(logits, axis= -1)
        prob_list.append(prob.tolist())
        result_list.append(result.tolist())
    print(f'prob list : {prob_list}')
    print(f'result list : {result_list}')
