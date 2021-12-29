FROM nvcr.io/nvidia/pytorch:21.11-py3

# RUN nvidia-smi

# ENV API_SERVER_TESTING TESTING
ADD ./ /extraction
# RUN ls -l /work

RUN pip3 install -r /extraction/requirements.txt --no-cache-dir

WORKDIR /extraction

# CMD uvicorn main:app --host 0.0.0.0 --port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9090"]

EXPOSE 9090
