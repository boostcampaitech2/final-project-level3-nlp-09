FROM nvcr.io/nvidia/pytorch:21.11-py3

# RUN nvidia-smi

# ENV API_SERVER_TESTING TESTING
ADD ./ /work
# RUN ls -l /work
RUN cp -r /work/QA_model /work/api/extraction-mrc

RUN pip3 install -r /work/api/extraction-mrc/requirements.txt --no-cache-dir

WORKDIR /work/api/extraction-mrc

# CMD uvicorn main:app --host 0.0.0.0 --port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9090"]

EXPOSE 9090
