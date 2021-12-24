# Twenty-Questions Game

## 🙋‍♂️팀원 소개
|김다영|김다인|박성호|박재형|서동건|정민지|최석민|
| :---: | :---: | :---: | :---: | :---: | :---: | :---:
| <a href="https://github.com/keemdy" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/68893924?v=4" width="80%" height="80%"> | <a href="https://github.com/danny980521" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/77524474?v=4" width="80%" height="80%">| <a href="https://github.com/naem1023" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/11407756?v=4" width="80%" height="80%"> | <a href="https://github.com/Jay-Ppark" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/29303223?v=4" width="80%" height="80%">|  <a href="https://github.com/donggunseo" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/43330160?v=4" width="80%" height="80%">|<a href="https://github.com/minji-o-j" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/45448731?v=4" width="80%" height="80%">| <a href="https://github.com/RockMiin" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/52374789?v=4" width="80%" height="80%">|
|EDA|DL 모델링, <br>Data 전처리,<br> 발표|Data 전처리,<br> 맞춤법과 띄어쓰기를<br> 이용한 feature 생성|EDA,<br> 조사를 이용한<br> feature 생성|DL, ML 모델링| <!--***여기에 각자 역할 적어!-->

# Model Overview

## Extraction-based MRC model
![](img/Reader.png)
- Get Answer of Question extracted from context


## BoolQA Model
- 

# Product Overview
![](img/Project-Overview.jpg)
- API Product
  - Dockerizing two API server.
  - Provide differenet CUDA environments for two model.
- Web Product
  - Dockerizing flask web server.
  - Asynchronous connection for two API server.
- CI/CD
  - Airflow
    - Retrain Boolean QA Model.
    - Evaluate retrained model.
    - Upload model to Huggingface Hub.
  - Github Action
    - Build docker image of API product
    - Execute Github Runner to deploy new docker image.
---
# Airflow

[Airflow documnet](airflow/README.md)


# API server
[API document](api/README.md)

## API Demo
```sh
# Boolean QA Model Demo
$ ./bm-demo.sh
# Extraction-based QA Model Demo
$ ./em-demo.sh
```
## API Docker Build
```
$ sudo ./build.sh
```
# Front-end
## Front-end Demo
```sh
$ cd app
$ pip install -r requirements.txt
$ python myapp.py
```
## Front-end Docker Build
```sh
$ sudo docker-compose up -d --build app
```
# Model
## Train, Inference environments
```sh
$ pip install -r BoolQA_model/requiremnets.txt
$ pip install -r QA_model/requiremnets.txt
```
## Infrenece testing of Boolean QA model
```sh
$ python 
```
## Train Extraction-based model
```sh
$ python train.py
```
## Infrenece testing of Extraction-based model
```sh
$ python test_extraction_qa_inference.py
```
