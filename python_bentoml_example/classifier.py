from typing import List
import pandas as pd
import numpy as np

from bentoml import env, artifacts, api, BentoService
from bentoml.types import JsonSerializable
from bentoml.adapters import DataframeInput, JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact, JSONArtifact
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
import bentoml
from bentoml.artifact import PytorchModelArtifact
from bentoml.handlers import ImageHandler
"""
@env(
    requirements_txt_file="./requirements.txt"
)
@artifacts([TensorflowSavedModelArtifact('tf_model'), PickleArtifact('mapping')])
class TitanicTFClassifier(BentoService):
    def __init__(self):
        super().__init__()
        self.columns_list = ['Sex', 'Age_band', 'Pclass']

    def mapping_df(self, df):
        df['Sex'] = df['Sex'].map(self.artifacts.mapping)
        return df

    @api(
        input = DataframeInput(),
        batch = True
    )
    def predict(self, df: pd.DataFrame):
        df.columns = self.columns_list
        print(df.head())
        print(self.artifacts.mapping)
        df = self.mapping_df(df)
        return self.artifacts.tf_model(df)
"""

'''
# Json input으로 받을 경우는 아래와 같이 진행하면 됌
@env(
    requirements_txt_file="./requirements.txt"
)
@artifacts([KerasModelArtifact('tf_model'), PickleArtifact('mapping')])
class TitanicTFClassifier(BentoService):
    # DataFrame에서는 ['column']으로 접근하지만 
    def mapping_json_value(self, data):
        return self.artifacts.mapping.get(data)
    @api(
        input = JsonInput(),
        batch = True
    )
    def predict(self, json_list: List[JsonSerializable]):
        results = []
        for i in json_list:
            i['Sex'] = self.mapping_json_value(i['Sex'])
            predict_data = np.array([value for _, value in i.items()]).reshape(1, -1)
            results.append( self.artifacts.tf_model.predict( predict_data )[0])
        return results
'''

# we state all the dependencies the service requires here
@bentoml.env(pip_dependencies=['torch', 'torchvision'])

# Defines the artifact that is used to deserialize the model
@bentoml.artifacts([PytorchModelArtifact('model')])
class QuestionAnsweringLM(bentoml.BentoService):

    # the actual api definition. Requires a ImageHandler to 
    # accept the incomeing img.
    @bentoml.api(ImageHandler)
    def predict(self, img):
        
        # convert the image to pillow image for PyTorch
        img = Image.fromarray(img)
        # perform the transformations, returns a tensor (3, 224, 224)
        img = transform(img)

        # Use eval mode for evaluation.
        self.artifacts.model.eval()
        # Performs forward prop
        outputs = self.artifacts.model(img.unsqueeze(0))
        # find the index with the highest probability
        _, idxs = outputs.topk(1)
        idx = idxs.squeeze().item()
        # return the prediction
        return classes[idx]