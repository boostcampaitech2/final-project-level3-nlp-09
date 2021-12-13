# final-project-level3-nlp-09  
final-project-level3-nlp-09 created by GitHub Classroom  
* This is feat/front branch  
    * cd front  
    * python myapp.py  

## Model
### Inference
```sh
python ./QA_model/inference.py --output_dir ./outputs/test_dataset/ --dataset_name ./QA_model/model/text_dict.json --model_name_or_path ./QA_model/model/checkpoint-28500 --do_predict

python ./QA_model/onnx_inference.py --output_dir ./outputs/test_dataset/ --dataset_name ./QA_model/model/text_dict.json --model_name_or_path ./QA_model/model/checkpoint-28500 --do_predict

streamlit run ./QA_model/streamlit_inf.py -- --output_dir ./outputs/test_dataset/ --dataset_name ./QA_model/model/text_dict.json --model_name_or_path ./QA_model/model/checkpoint-28500 --do_predict
```
### Run servers
```
$ docker-compose up -d --build rabbitmq api app
```
### Run web
```
$ API_URI=xxxx docker-compose up -d --build web
```
### Run worker (cuda environment)
```
/worker $ python3 -m pip install -r ./requirements.txt
/worker $ FILE_SERVER=http://{API_HOST}:8080 \
  RABBITMQ_HOST={RABBIT_MQ_HOST} \
  QUEUE_NAME=IMAGE-PROCESS \
  RESULT_SUFFIX=_result \
  ACCESS_TOKEN=TOKEN_FOR_DIRECT_UPLOAD \
  DOWNLOAD_PATH=./files \
  python3 main.py
```

### API server
[API document](api/README.md)

# twenty-questions

# pt to onnx
```sh
python QA_model/convert_graph_to_onnx.py --pipeline question-answering --framework pt --model ./QA_model/model/checkpoint-28500  --quantize ./QA_model/model/onnx/KLRL-QA.onnx
```

# 훈련
python train.py 

# inference
TBD...

# pt file link
[구글 드라이브](https://drive.google.com/drive/folders/1zXe4xHqX7kxOZIVjb73NW0rCZ3G7uUAX?usp=sharing)

```sh
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI" -O model.zip && rm -rf ~/cookies.txt
```
