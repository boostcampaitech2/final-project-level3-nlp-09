from datasets import load_dataset
import time

for i in range(100):
    dataset = load_dataset("minji/test")
    print(dataset['train'][0])
    time.sleep


from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
# argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.

tokenizer = AutoTokenizer.from_pretrained("NaDy/ko-mrc-model")
model = AutoModelForQuestionAnswering.from_pretrained("NaDy/ko-mrc-model")