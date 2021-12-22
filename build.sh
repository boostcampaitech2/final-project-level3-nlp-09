rm -rf api/extraction-mrc/QA_model
cp -r QA_model api/extraction-mrc/

rm -rf api/bool-qa/data
cp -r QA_model/data api/bool-qa/data

# Build Extracion-based MRC model container
docker stop em-api-server
docker rm em-api-server
docker build --tag em-api ./api/extraction-mrc
docker run --gpus all -d -p 9090:9090 --name em-api-server em-api:latest

# Build Boolean QA model container
docker stop bm-api-server
docker rm bm-api-server
docker build --tag bm-api ./api/bool-qa
docker run --gpus all -d -p 9091:9091 --name bm-api-server bm-api:latest