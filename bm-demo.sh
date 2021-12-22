rm -rf api/bool-qa/data
cp -r QA_model/data api/bool-qa/data
cd api/bool-qa
uvicorn main:app --host 0.0.0.0 --port 9091