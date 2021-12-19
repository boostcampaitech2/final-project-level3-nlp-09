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
### Run
```sh
# If container is stopped
$ docker start cudaenv
$ docker attach cudaenv
```