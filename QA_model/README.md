# Env
## CUDA container
```sh
docker run -it --name "onnx_cuda11.04" --gpus "device=1" -p 32100:22 -p 32200:8888 nvidia/cuda:11.4.2-base-ubuntu20.04 /bin/bash
```