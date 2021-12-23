FROM nvcr.io/nvidia/pytorch:21.11-py3

# RUN nvidia-smi

# ENV API_SERVER_TESTING TESTING
ADD ./ /work
# RUN ls -l /work

RUN pip3 install -r /work/api/bool-qa/requirements.txt --no-cache-dir

WORKDIR /work/api/bool-qa

# CMD uvicorn main:app --host 0.0.0.0 --port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9091"]

EXPOSE 9091
