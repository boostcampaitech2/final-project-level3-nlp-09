rm -rf api/extraction-mrc/QA_model
cp -r QA_model api/extraction-mrc/
cd api/extraction-mrc
uvicorn main:app --host 0.0.0.0 --port 9090