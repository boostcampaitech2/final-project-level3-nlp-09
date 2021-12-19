import os
import random
import time
import string
from typing import List, Optional

import pika
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from aiofile import AIOFile, Writer, Reader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
from QA_model.inference import QAInference

inf = QAInference()
inf.set_context()

# upload_directory = os.environ['UPLOAD_PATH']
# access_token = os.environ['ACCESS_TOKEN']
# result_suffix = os.environ['RESULT_SUFFIX']
# array_suffix = os.environ['ARRAY_SUFFIX']
# queue_name = os.environ['QUEUE_NAME']
# rabbitmq_host = os.environ['RABBITMQ_HOST']

# try:
#     os.mkdir(upload_directory)
#     print(os.getcwd())
# except Exception as e:
#     print(e)


def random_string_with_time(length: int):
    return ''.join(
        random.choices(string.ascii_lowercase + string.digits, k=length)
    ) + '-' + str(int(time.time()))


async def store_file(uploaded_file: UploadFile, destination: string):
    async with AIOFile(os.path.join(upload_directory, destination), 'wb') as f:
        writer = Writer(f)
        while True:
            chunk = await uploaded_file.read(8192)
            if not chunk:
                break
            await writer(chunk)
    return destination


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


# Get inference result
@app.post("/chat")
async def get_inferenec(item: Request):
    item_dict = await item.json()
    text = item_dict['data']
    inf.set_question(text)
    inf.set_dataset()
    generated = inf.run_mrc()

    return {'result': generated}

