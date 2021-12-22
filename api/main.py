from typing import List, Optional
import traceback, os

import pika
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from aiofile import AIOFile, Writer, Reader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from QA_model.inference import ExtractivedQAMdoel

import random
import numpy as np
import time

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
    inf = ExtractivedQAMdoel('QA_model/data/contexts')

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


@app.get("/set_category")
async def get_inferene():
    """
    Context 카테고리 설정
    item:
        Return: {
            'category': 'some category', 
            'context': 'some context', 
            'answer': 'title of context'
        }
    """
    data = {
        'category': '', 
        'context': '', 
        'answer': ''
    }
    if not api_server_testing == 'TESTING':
        try:
            # item_dict = await item.json()
            # category = item_dict['category']
            res = inf.set_context()
            data['category'], data['context'], data['answer'] = res
            return res
        except:
            err = traceback.format_exc()
            print(err)
            return {'Error':err}
    else:
        return data

# Get inference result
@app.post("/chat")
async def get_inference(item: Request):
    """
    Question에 대한 Extractived-base MRC 모델의 결과 추론
    item:
        {'data': 'some question'}
    """
    if not api_server_testing == 'TESTING':
        try:
            item_dict = await item.json()
            question = item_dict['data']
            inf.set_question(question)
            inf.prepare_dataset()
            generated = inf.run_mrc()

            return {'result': generated}
        except:
            err = traceback.format_exc()
            print(err)
            return {'Error':err}
    else:
        return 'Test String입니다!!!!!!'


def get_random_context(dataset_path: str):
    """
    dataset_path: 스무고개 질문지가 담긴 directory 이름
    return: 
        category, context, answer
    """
    assert os.path.isdir(dataset_path)
    random.seed(time.time())

    category_candidates= []

    # Append candidates of category
    with os.scandir(dataset_path) as entries:
        for entry in entries:
            if entry.is_dir():
                category_candidates.append(entry)

    # Select category
    path = random.choice(category_candidates)
    category = path.name

    candidates = []
    # Append candidates of contexts
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.name.endswith('txt') and entry.is_file():
                candidates.append(entry)

    # Select context!
    target = random.choice(candidates)
    with open(target.path, 'r', encoding='utf-8') as file:
        contexts = file.readlines()

    context = ' '.join(contexts)
    answer = target.name[:-4]
    return category, context, answer

@app.get("/set_category_boolq")
async def get_inferenec():
    """
    Context 카테고리 설정
    item:
        Return: {
            'category': 'some category', 
            'context': 'some context', 
            'answer': 'title of context'
        }
    """
    
    if not api_server_testing == 'TESTING':
        try:
            # item_dict = await item.json()
            # category = item_dict['category']
            res= get_random_context('BoolQA_model/data/context')
            # res= {'category': 'mbti', 'context': 'ESTP', 'answer': 'NO'}
            data['category'], data['context'], data['answer'] = res
            return res
        except:
            err = traceback.format_exc()
            print(err)
            return {'Error':err}
    else:
        return data
    
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
            item_dict = await item.json()
            question = item_dict['data']

            path= 'BoolQA_model'
            file_path= os.path.join(path, 'data', 'context', 'mbti', 'ESTP.txt')

            with open(file_path, 'r') as f:
                passage= f.read()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            prob_list= []
            result_list= []
            print(f'device: {device}')
            print(question)
            print(type(passage))

            tokenized_text= tokenizer(
                question,
                passage,
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

            model.eval()
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


