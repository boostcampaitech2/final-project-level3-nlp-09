
from flask import Flask, render_template, request
from transformers import AutoModel, AutoConfig, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
@app.route("/")
def home(): 
	return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    input_ids= tokenizer.encode(userText, return_tensors= 'pt')
    predict= model.generate(
        input_ids, 
        max_length=50, 
        num_beams=5, 
        no_repeat_ngram_size=2, 
        early_stopping=True)

    result= tokenizer.decode(predict[0], skip_special_tokens= True)
    return result

if __name__ == "__main__":
    model_name= 'gpt2'
    tokenizer= GPT2Tokenizer.from_pretrained(model_name)
    model= GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    app.run()