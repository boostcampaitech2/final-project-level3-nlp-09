"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


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

from QA_model.utils_qa import postprocess_qa_predictions_inf, check_no_error
from QA_model.trainer_qa import QuestionAnsweringTrainer

from QA_model.random_context import get_context

class ModelArguments:
    def __init__(self):
        self.model_name_or_path = 'NaDy/ko-mrc-model'
        self.config_name = None
        self.tokenizer_name = None

class DataTrainingArguments:
    def __init__(self):
        self.overwrite_cache = False
        self.preprocessing_num_workers = 4
        self.max_seq_length = 384
        self.pad_to_max_length = False
        self.doc_stride = 128
        self.max_answer_length = 50
        

class ExtractivedQAMdoel:
    def __init__(self, dataset_path):
        # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
        # --help flag 를 실행시켜서 확인할 수 도 있습니다.
        self.dataset_path = dataset_path

        self.model_args = ModelArguments()
        self.data_args = DataTrainingArguments()
        self.training_args = TrainingArguments(
            output_dir = './outputs/one_question',
            do_predict = True,
            fp16=True
        )
        
        print(f"model is from {self.model_args.model_name_or_path}")
        
        # 모델을 초기화하기 전에 난수를 고정합니다.
        set_seed(self.training_args.seed)
        
        # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
        # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
        config = AutoConfig.from_pretrained(
            self.model_args.config_name
            if self.model_args.config_name
            else self.model_args.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            use_fast=True,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=config,
        )
        self.trainer = None

        self.set_context('행성', '금성')
        self.set_question('이것의 위치는?')
        self.prepare_dataset()
        self.run_mrc()

    def set_context(self, category, context_name):
        self.context = get_context(self.dataset_path, category, context_name)
        self.category = category
        self.answer = context_name
        return self.context

    def set_question(self, question):
        self.question = question

    def prepare_dataset(self):
        d = {'context' : [self.context], 'question' : [self.question], 'id' : ['mrc-1']}
        df = pd.DataFrame(data=d)
        
        f = Features(
            {
                "context": Value(dtype="string", id=None),  # 바꿈!
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        self.datasets = DatasetDict({'validation' : Dataset.from_pandas(df, features=f)})

    def run_mrc(self):
        return self._run_mrc(self.data_args,
            self.training_args,
            self.model_args,
            self.datasets,
            self.tokenizer,
            self.model
        )

    def _run_mrc(
        self,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        model_args: ModelArguments,
        datasets: DatasetDict,
        tokenizer,
        model,
    ) -> NoReturn:

        # eval 혹은 prediction에서만 사용함
        column_names = datasets["validation"].column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]

        # Padding에 대한 옵션을 설정합니다.
        # (question|context) 혹은 (context|question)로 세팅 가능합니다.
        pad_on_right = tokenizer.padding_side == "right"

        # 오류가 있는지 확인합니다.
        last_checkpoint, max_seq_length = check_no_error(
            data_args, training_args, datasets, tokenizer
        )
        # 싹 바꿈!
        def prepare_validation_features(examples):
            # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
            # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=False,  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                padding="max_length" if data_args.pad_to_max_length else False,
            )

            # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
            # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # sequence id를 설정합니다 (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                # 하나의 example이 여러개의 span을 가질 수 있습니다.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
            return tokenized_examples

        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Data collator
        # flag가 True이면 이미 max length로 padding된 상태입니다.
        # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )

        # Post-processing:
        def post_processing_function(
            examples,
            features,
            predictions: Tuple[np.ndarray, np.ndarray],
            training_args: TrainingArguments,
        ) -> EvalPrediction:
            # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
            # 바꿈!
            predictions = postprocess_qa_predictions_inf(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=data_args.max_answer_length,
                output_dir=training_args.output_dir,
            )
            # Metric을 구할 수 있도록 Format을 맞춰줍니다.
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]
            if training_args.do_predict:
                return formatted_predictions


        metric = load_metric("squad")
        def compute_metrics(p: EvalPrediction) -> Dict:
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        print("init trainer...")
        if self.trainer is None:
            # Trainer 초기화
            self.trainer = QuestionAnsweringTrainer(
                model=model,
                args=training_args,
                train_dataset=None,
                eval_dataset=eval_dataset,
                eval_examples=datasets["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                post_process_function=post_processing_function,
                compute_metrics=compute_metrics,
            )

        #### eval dataset & eval example - predictions.json 생성됨
        if training_args.do_predict:
            predictions = self.trainer.predict(
                test_dataset=eval_dataset, test_examples=datasets["validation"]
            )
            print(predictions)
            # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
            print(
                "No metric can be presented because there is no correct answer given. Job done!"
            )
            text = predictions[0]['prediction_text'].replace(self.answer, '[정답]')
            
            return text