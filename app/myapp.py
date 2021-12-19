
from flask import Flask, render_template, request
# from transformers import AutoModel, AutoConfig, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import requests

import aiohttp
import asyncio


app = Flask(__name__)
@app.route("/")
def home(): 
	return render_template("home.html")

@app.route("/get")
async def get_bot_response():
    userText = request.args.get('msg')
    print(userText)
    data = {'data':userText}
    async with aiohttp.ClientSession() as session:
        url = 'http://0.0.0.0:9090/chat'
        async with session.post(url, json=data) as resp:
            return ((await resp.json())['result'])

if __name__ == "__main__":
    # model_name= 'gpt2'
    # tokenizer= GPT2Tokenizer.from_pretrained(model_name)
    # model= GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    app.run(host='0.0.0.0', port=5000)