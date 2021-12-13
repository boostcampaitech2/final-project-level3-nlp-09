FROM nvidia/cuda:11.4.2-base-ubuntu20.04
CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

ADD / /app

RUN pip3 install -r /app/requirements.txt