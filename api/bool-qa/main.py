from typing import List, Optional
import traceback, os

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np

data = {
        'category': '', 
        'context': '', 
        'answer': ''
    }

try:
    api_server_testing = os.environ['API_SERVER_TESTING']
except:
    api_server_testing = 'release'

if api_server_testing == 'release':
    model_name= 'rockmiin/ko-boolq-model'
    model= AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer= AutoTokenizer.from_pretrained(model_name)
    print(f'boolQA model loaded!')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_context(dataset_path: str, category: str, context_name: str):
    """
    dataset_path: 스무고개 질문지가 담긴 directory 이름
    return: 
        category, context, answer
    """
    assert os.path.isdir(dataset_path)
    assert os.path.isdir(os.path.join(dataset_path, category))
    print(os.path.join(dataset_path, category, context_name + '.txt'))
    assert os.path.exists(os.path.join(dataset_path, category, context_name + '.txt'))

    target_path = os.path.join(dataset_path, category, context_name + '.txt')
    with open(target_path, 'r', encoding='utf-8') as file:
        contexts = file.readlines()

    context = ' '.join(contexts)
    return context

_context = ''

@app.post("/set_category_boolq")
async def set_category(item: Request):
    """
    Context 카테고리 설정
    item:
        Input: {
            'category': 'some category', 
            'context_name': 'some context_name'
        }

        Return:
            {'result': 'Done!'}
            {'Error': err}
    """
    
    if not api_server_testing == 'TESTING':
        try:
            global _context
            item_dict = await item.json()
            context = get_context('data/contexts', item_dict['category'], item_dict['context_name'])
            _context = context
            return {'result': 'Done!'}
        except:
            err = traceback.format_exc()
            print(err)
            return {'Error': err}
    else:
        return {'result': 'Done!'}
    
# Get inference result
@app.post("/chat_boolq")
async def get_inference(item: Request):
    """
    Question에 대한 boolQA-base MRC 모델의 결과 추론
    item:
        {'data': 'some question'}
    """
    # model_name= 'rockmiin/ko-boolq-model'
    # model= AutoModelForSequenceClassification.from_pretrained(model_name)
    # tokenizer= AutoModelForSequenceClassification.from_pretrained(model_name)
    if not api_server_testing == 'TESTING':
        try:
            global _context

            item_dict = await item.json()
            question = item_dict['data']


            prob_list= []
            result_list= []
            print(f'device: {device}')
            print(question)
            print(type(_context))

            tokenized_text= tokenizer(
                question,
                _context,
                truncation= 'only_second',
                max_length= 256,
                stride= 64,
                padding= 'max_length',
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=False, 
                return_tensors= 'pt'
            )
            # print(tokenized_text)
            print(len(tokenized_text['input_ids']))
            result_list= {0: 0, 1: 0, 2: 0}

            outputs= model(
                input_ids= tokenized_text['input_ids'].to(device),
                attention_mask= tokenized_text['attention_mask'].to(device)
            )['logits']

            logits= outputs
            # logits= F.softmax(logits, dim= -1)
            logits= logits.detach().cpu().numpy()

            prob= logits
                
            result= np.argmax(logits, axis= -1)
            prob_list.append(prob.tolist())
            result_list[result.tolist()[0]]+=1
            # print(prob_list)
            # print(result_list)

            if result_list[1] != 0:
                answer= "네 맞습니다."
            elif result_list[0] != 0:
                answer= "아니오. 틀립니다."
            else:
                answer= "잘 모르겠습니다."


            return {'result': answer}

        except:
            err = traceback.format_exc()
            print(err)
            return {'result':err}
    else:
        return 'Test String입니다!!!!!!'


