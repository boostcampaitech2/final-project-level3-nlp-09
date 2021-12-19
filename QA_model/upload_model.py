from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model = AutoModelForQuestionAnswering.from_pretrained('./models/klue/roberta-large5/checkpoint-28500/')
model.push_to_hub("ko-mrc-model")

tokenizer = AutoTokenizer.from_pretrained('./models/klue/roberta-large5/checkpoint-28500/')
tokenizer.push_to_hub("ko-mrc-model")