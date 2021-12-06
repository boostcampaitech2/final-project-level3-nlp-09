import argparse
import sys

from titanic import TitanicMain
from bentoml_process import BentoML

def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        '--is_keras', type=str,
        help="please input 1 or 0"
    )

    args = argument_parser.parse_args()
    try:
        is_keras = _str2bool(args.is_keras)
    except argparse.ArgumentTypeError as E:
        print("ERROR!! please input is_keras 0 or 1")
        sys.exit()
    titanic = TitanicMain()
    bento_ml = BentoML()

    if is_keras:
        model = titanic.run(is_keras)
        save_path = bento_ml.run_bentoml(model, None, is_keras)
    else:
        rf_model, lgbm_model = titanic.run(is_keras)
        save_path = bento_ml.run_bentoml(rf_model, lgbm_model, is_keras)

    print("save path")
    print(save_path)
    

