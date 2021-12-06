# Python bentoml for Machine Learning model API serving
Python bentoML API serving example for serving machine learning model with API

### Info & reference
> Blog post (Description)
    - https://lsjsj92.tistory.com/621  

> refer
    - https://github.com/bentoml/BentoML


## Description

- bentoml_process.py
    - Packing data with BentoML classifier
- classifier.py
    - BentoML classifier
    - Make BentoML model API environment
- main.py
    - Main file on this process
    - start with is_keras argument 
        - 1 : use tensorflow(keras)
        - 0 : non use tensorflow(keras), use scikit learn
    - Execute titanic modeling
    - Execute BentoML Packing
- titanic.py
    - Main of Titniac process
    - data load -> preprocess -> ML/DL modeling -> return model
- model.py
    - Machine Learning or Deep Learning Modeling part
    - Machine Learning : use scikit-learn library or lightgbm
    - Deep Learning : use tensorflow library ( keras )
- preprocess.py
    - Preprocess part
    - Preprocess titanic data
- config.py
    - Config part
- dataio.py
    - Data io part
    - Get data