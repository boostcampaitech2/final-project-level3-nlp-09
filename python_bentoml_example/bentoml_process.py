from config import EnvConfig
from classifier import TitanicSKlearnClassifier, TitanicTFClassifier

class BentoML:
    def run_bentoml(self, model1, model2, is_keras):
        if is_keras:
            classifier_service = TitanicTFClassifier()
            classifier_service.pack('mapping', EnvConfig().get_gender_mapping_code())
            classifier_service.pack('tf_model', model1)
        else:
            classifier_service = TitanicSKlearnClassifier()
            classifier_service.pack('mapping', EnvConfig().get_gender_mapping_code())
            classifier_service.pack('rf_model', model1)
            classifier_service.pack('lgbm_model', model2)

        saved_path = classifier_service.save()
        print("save path : ", saved_path)

        return saved_path