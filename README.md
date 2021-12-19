# twenty-questions
# Env Setting
## Train, Inference env
```sh
pip install -r requiremnets
```
## TensorRT, ONNX CUDA container
README.md의 모든 항목들 중 (cuda env)가 붙은 항목들은 해당 환경으로 실행
### Container setting
- Base or Recommendation 선택해 컨테이너 생성
```sh
$ docker pull nvcr.io/nvidia/tensorrt:21.11-py3

$ cd QA_model
# Base
$ docker run --gpus all -it --name cudaenv -v `pwd`:/workspace/app nvcr.io/nvidia/tensorrt:21.11-py3

docker run --gpus all -it -v `pwd`:/workspace/app nvcr.io/nvidia/pytorch:21.11-py3

# Recommendation for pytorch
$ docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v `pwd`:/workspace/app nvcr.io/nvidia/tensorrt:21.11-py3
```
# final-project-level3-nlp-09  
final-project-level3-nlp-09 created by GitHub Classroom  
* This is feat/front branch  
    * cd front  
    * boot 디렉안에 1안 index.html, 2안 test1, 3안 test2
    * python myapp.py  

### Run
```sh
# If container is stopped
$ docker start cudaenv
$ docker attach cudaenv
```
---
## Airflow

[Airflow documnet](airflow/README.md)


# API server
[API document](api/README.md)

## Build
```
$ docker-compose up -d --build rabbitmq api app
```
# Model
## Train
```sh
python train.py 
```
## Convert Pytorch to ONNX(cuda env)
```sh
$ cd app
$ python convert_graph_to_onnx.py --pipeline question-answering --framework pt --model ./model/checkpoint-28500  --quantize ./model/onnx/KLRL-QA.onnx
```

## Convert ONNX to TensorRT(cuda env)
```sh
$ chmod +x trtexec_build.sh
$ ./trtexec_build.sh
```
inference를 하고 싶다면.. `inference.py` 의 argument를 확인하고 실행
(이 때 사용하고 싶은 모델을 save_model 폴더에서 찾아 argument로 입력해줘야함.)

```python
python inference.py
```




# pt file link
[구글 드라이브](https://drive.google.com/drive/folders/1zXe4xHqX7kxOZIVjb73NW0rCZ3G7uUAX?usp=sharing)

```sh
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI" -O model.zip && rm -rf ~/cookies.txt
```
