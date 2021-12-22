# Air flow
## Generate fernet key
Copy & paste to docker-compose.yaml.

```sh
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
## Build
```sh
docker-compose -f docker-compose-CeleryExecutor.yml up -d
```
## Set admin user
Set admin user, if admin user doesn't exit.

```sh
airflow users create \
    --username admin \
    --firstname Chungchoon \
    --lastname Hindi \
    --role Admin \
    --email relilau00@gmail.com
```