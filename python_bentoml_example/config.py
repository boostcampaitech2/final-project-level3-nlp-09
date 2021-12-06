import os


class PathConfig:
    def __init__(self):
        self.project_path = os.getcwd()
        #self.parent_path = "/".join(self.project_path.split("/")[:-1])
        self.titanic_path = f"{self.project_path}/data/titanic"


class EnvConfig:
    def get_gender_mapping_code(self):
        gender_mapping_info = {
            'male' : 0,
            'female' : 1,
        }

        return gender_mapping_info
