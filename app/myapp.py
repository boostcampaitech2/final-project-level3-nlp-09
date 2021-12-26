
from flask import Flask, render_template, request
# from transformers import AutoModel, AutoConfig, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import requests

import aiohttp, json
import asyncio


app = Flask(__name__)
@app.route("/")
def home(): 
	return render_template("test.html")

@app.route("/set_category_boolq")
async def set_category():
    category = request.args.get('category')
    answer = request.args.get('answer')
    print(f'category={category}, answer={answer}')
    data = {'category': category, 'context_name': answer}

    async with aiohttp.ClientSession() as session:
        url = 'http://14.49.45.219:9091/set_category_boolq'
        async with session.post(url, json=data) as resp:
            return ((await resp.json())['result'])

@app.route("/set_category")
async def set_category_boolq():
    category = request.args.get('category')
    answer = request.args.get('answer')
    print(f'category={category}, answer={answer}')
    data = {'category': category, 'context_name': answer}

    async with aiohttp.ClientSession() as session:
        url = 'http://14.49.45.219:9090/set_category'
        async with session.post(url, json=data) as resp:
            return ((await resp.json())['result'])

@app.route("/get_hint")
async def get_bot_response():
    userText = request.args.get('msg')
    print(userText)
    data = {'data':userText}
    async with aiohttp.ClientSession() as session:
        url = 'http://14.49.45.219:9090/chat'
        async with session.post(url, json=data) as resp:
            return ((await resp.json())['result'])

@app.route("/get_boolq")
async def get_boolqa_bot_response():
    userText = request.args.get('msg')
    print(userText)
    data = {'data':userText}
    async with aiohttp.ClientSession() as session:
        url = 'http://14.49.45.219:9091/chat_boolq'
        async with session.post(url, json=data) as resp:
            return ((await resp.json())['result'])

@app.route("/get_feedback")
async def get_user_feedback():
    answer_keyword = request.args.get('answer_keyword')
    answers = request.args.get('answers')
    question = request.args.get('question')
    data = {'answer_keyword': answer_keyword.split(','), 'answers': list(map(int, answers.split(','))),'question':question.split(',')}
    async with aiohttp.ClientSession() as session:
        url = 'http://14.49.45.219:9091/get_feedback'
        async with session.post(url, json=data) as resp:
            return data#((await resp.json())['result'])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)