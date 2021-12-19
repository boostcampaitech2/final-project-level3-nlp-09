import logging
import sys
from typing import Callable, List, Dict, NoReturn, Tuple

import numpy as np
import pandas as pd

from datasets import (
    load_metric,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from QA_model.arguments import (
    ModelArguments,
    DataTrainingArguments,
)

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
import pickle

parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments)
        )
for p in parser._actions:
    if '--output_dir' in p.option_strings[0]:
        p.required = False
        print(p)
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

import os
if not os.path.exists('./args'):
    os.mkdir('./args')
with open('./args/model_args.pkl', 'wb') as file:
    pickle.dump(model_args, file, pickle.HIGHEST_PROTOCOL)
with open('./args/data_args.pkl', 'wb') as file:
    pickle.dump(data_args, file, pickle.HIGHEST_PROTOCOL)
with open('./args/training_args.pkl', 'wb') as file:
    pickle.dump(training_args, file, pickle.HIGHEST_PROTOCOL)
