from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.models import Model

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch

class TitanicModeling:
    def __init__(self):
        pass

    def run_sklearn_modeling(self, X, y):
        rf_model = self._get_rf_model()
        lgbm_model = self._get_lgbm_model()
        rf_model.fit(X, y)
        lgbm_model.fit(X, y)
        print('rf_model Score : ', rf_model.score(X, y))
        print('lgbm_model Score : ', lgbm_model.score(X, y))
        return rf_model, lgbm_model

    def run_keras_modeling(self, X, y):
        # model = self._get_keras_model()
        # model.fit(X, y)
        # predictions = model.predict(X)
        # print('keras prediction : ', predictions[:5])
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                    pad_token='<pad>', mask_token='<mask>') 

        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

        return model

    def _get_rf_model(self):
        return RandomForestClassifier(n_estimators=100, max_depth=5)

    def _get_lgbm_model(self):
        return LGBMClassifier(n_estimators=150)

    def _get_keras_model(self):
        inp = Input(shape=(3, ), name='inp_layer')
        dense_layer_1 = Dense(32, activation='relu', name="dense_1")
        dense_layer_2 = Dense(16, activation='relu', name="dense_2")
        predict_layer = Dense(1, activation = 'sigmoid', name='predict_layer')

        dense_vector_1 = dense_layer_1(inp)
        dense_vector_2 = dense_layer_2(dense_vector_1)
        predict_vector = predict_layer(dense_vector_2)

        model = Model(inputs=inp, outputs=predict_vector)
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
        return model




        

