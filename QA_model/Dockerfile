FROM nvcr.io/nvidia/pytorch:21.11-py3
# CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN pip3 install -r /app/requirements.txt