# final-project-level3-nlp-09  
final-project-level3-nlp-09 created by GitHub Classroom  
* This is feat/front branch  
    * cd front  
    * python myapp.py  

## Model
### Inference
```sh
python ./api/inference.py --output_dir ./outputs/test_dataset/ --dataset_name ./api/model/text_dict.json --model_name_or_path api/model/checkpoint-28500 --do_predict
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
스무고개

# 훈련
python train.py 

# inference
TBD...

# pt file link
[구글 드라이브](https://drive.google.com/drive/folders/1zXe4xHqX7kxOZIVjb73NW0rCZ3G7uUAX?usp=sharing)

```sh
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16f1Qc7t5uvJaDzjXsa53a9xo2kcgcLbp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16f1Qc7t5uvJaDzjXsa53a9xo2kcgcLbp" -O model.zip && rm -rf ~/cookies.txt
```
