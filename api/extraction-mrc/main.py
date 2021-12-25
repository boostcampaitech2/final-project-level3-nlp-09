from typing import List, Optional
import traceback, os

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from QA_model.inference import ExtractivedQAMdoel

import numpy as np

try:
    api_server_testing = os.environ['API_SERVER_TESTING']
except:
    api_server_testing = 'release'

if api_server_testing == 'release':
    inf = ExtractivedQAMdoel('QA_model/data/contexts')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


@app.post("/set_category")
async def set_category(item: Request):
    """
    Context 카테고리 설정
    item:
        Input: {
            'category': 'some category', 
            'context_name': 'some context_name', 
        }

        Return:
            {'result': 'Done!'}
            {'Error': err}
    """
    if not api_server_testing == 'TESTING':
        try:
            item_dict = await item.json()
            res = inf.set_context(item_dict['category'], item_dict['context_name'])
            return {'result': 'Done!'}
        except:
            err = traceback.format_exc()
            print(err)
            return {'Error': err}
    else:
        return {'result': 'Done!'}

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
