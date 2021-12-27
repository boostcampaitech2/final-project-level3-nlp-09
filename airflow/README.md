# final-project-level3-nlp-09  
final-project-level3-nlp-09 created by GitHub Classroom  
* This is feat/front branch  
    * cd front  
    * python myapp.py  

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

