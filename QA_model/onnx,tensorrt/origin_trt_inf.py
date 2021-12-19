''' NOTE:
    코드 참고
     - https://github.com/NVIDIA/TensorRT/blob/main/demo/BERT/notebooks/Q-and-A.ipynb


'''

import tensorrt as trt
import argparse
import pycuda.driver as cuda
import pycuda.autoinit
import collections
import pandas as pd
import time
import numpy as np

from tqdm import tqdm
from transformers import (set_seed,)
from random_context import get_random_context
from datasets import (
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)
from helpers import data_processing as dp
from helpers import tokenization as tokenization

class QAInference:
    def __init__(self, args):
        self.args = args

        # roberta
        self.max_seq_length = 384
        self.doc_stride = 128
        self.max_query_length = 64
        self.vocab_file = "./model/fine-tuned/vocab.txt"
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

        # 난수를 고정합니다.
        set_seed(args.seed)

    def inference_FP16(self, trt_context, d_inputs, h_output, d_output, features, tokens): 
        #global h_output
        context = trt_context
        
        _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
                "NetworkOutput",
                ["start_logits", "end_logits", "feature_index"])
        networkOutputs = []

        eval_time_elapsed = 0
        for feature_index, feature in enumerate(features):
            # Copy inputs
            print('feature.input_ids', feature.input_ids)
            input_ids_batch = np.repeat(np.expand_dims(feature.input_ids, 0), 1, axis=0)
            segment_ids_batch = np.repeat(np.expand_dims(feature.segment_ids, 0), 1, axis=0)
            # input_mask_batch = np.repeat(np.expand_dims(feature.input_mask, 0), 1, axis=0)

            input_ids = cuda.register_host_memory(np.ascontiguousarray(input_ids_batch.ravel()))
            segment_ids = cuda.register_host_memory(np.ascontiguousarray(segment_ids_batch.ravel()))
            # input_mask = cuda.register_host_memory(np.ascontiguousarray(input_mask_batch.ravel()))

            eval_start_time = time.time()
            cuda.memcpy_htod_async(d_inputs[0], input_ids, self.stream)
            cuda.memcpy_htod_async(d_inputs[1], segment_ids, self.stream)
            # cuda.memcpy_htod_async(d_inputs[2], input_mask, self.stream)

            # Run inference
            trt_context.execute_async_v2(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=self.stream.handle)
            # Synchronize the stream
            self.stream.synchronize()
            eval_time_elapsed += (time.time() - eval_start_time)

            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(h_output, d_output, self.stream)
            self.stream.synchronize()

            for index, batch in enumerate(h_output):
                # print(h_output)
                # Data Post-processing
                networkOutputs.append(_NetworkOutput(
                    start_logits = np.array(batch.squeeze()[:, 0]),
                    end_logits = np.array(batch.squeeze()[:, 1]),
                    feature_index = feature_index
                    ))

        eval_time_elapsed /= len(features)

        # The total number of n-best predictions to generate in the nbest_predictions.json output file
        n_best_size = 20

        # The maximum length of an answer that can be generated. This is needed 
        #  because the start and end predictions are not conditioned on one another
        max_answer_length = 30

        prediction, nbest_json, scores_diff_json = dp.get_predictions(tokens, features,
            networkOutputs, n_best_size, max_answer_length)

        return eval_time_elapsed, prediction, nbest_json

    def load_trt_model(self):
        tensorrt_file_name = self.args.tensorrt_file_name
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        f = open(tensorrt_file_name, "rb")
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        self.engine_context = engine.create_execution_context()
        print(f"*** load model: {tensorrt_file_name}")
        f.close()

        # We always use batch size 1.
        input_shape = (1, self.max_seq_length)  # (1,384)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize  # 384*4
        
        # Allocate device memory for inputs.
        self.d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(2)]

        # Specify input shapes. These must be within the min/max bounds of the active profile (0th profile in this case)
        # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
        for binding in range(2):
            self.engine_context.set_binding_shape(binding, input_shape)
        assert self.engine_context.all_binding_shapes_specified

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        self.h_output = cuda.pagelocked_empty(tuple(self.engine_context.get_binding_shape(2)), dtype=np.float32)
        print(self.h_output)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()



    # 문서를 랜덤으로 가져옵니다.
    def set_context(self):
        self.context, _ = get_random_context(self.args.question_dataset_name)

    # 질문을 받습니다.
    def set_question(self, question):
        self.question = question
        # print(f'*** 입력된 질문: {self.question}')

    def set_dataset(self):
        # print('*** set dataset 실행')
        assert self.context != "", "지문을 가져오지 못했습니다."
        assert self.question != "", "입력된 질문이 없습니다"

        d = {"context": [self.context], "question": [self.question], "id": ["mrc-1"]}
        df = pd.DataFrame(data=d)
        f = Features(
            {
                "context": Sequence(feature=Value(dtype="string", id=None)),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        datadict_df =  DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        self.val_datasets = datadict_df['validation']

    def question_features(self, tokens, question):
        # Extract features from the paragraph and question
        return dp.convert_example_to_features(tokens, question, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)

    def print_single_query(eval_time_elapsed, prediction, nbest_json):
        print("Answer: '{}'".format(prediction))
        print("With probability: {:.2f}%".format(nbest_json[0]['probability'] * 100.0))
        
    def answers(self):
        N_RUN = self.args.n_runs
        inference_time_arr = [] 
        for _ in tqdm(range(N_RUN)):
            doc_tokens = dp.convert_doc_tokens(self.context)
            features = self.question_features(doc_tokens, self.question)
            # print(self.engine_context)
            # print(self.d_inputs)
            
            eval_time_elapsed, prediction, nbest_json = self.inference_FP16(self.engine_context, self.d_inputs, self.h_output, self.d_output, features, doc_tokens)
            inference_time_arr.append(eval_time_elapsed)

        self.print_single_query(eval_time_elapsed, prediction, nbest_json)
        print("Average inference time (over {} runs): {:.2f} ms".format(N_RUN, 1000*np.mean(inference_time_arr))) 



if __name__ == "__main__":
    # parser 정의
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensorrt_file_name",
        type=str,
        default="./export/fp16_test.trt",
        help="파일 경로를 설정합니다. 파일은 .trt 형식이거나 .engine 형식이어야합니다.",
    )
    parser.add_argument(
        "--question_dataset_name",
        type=str,
        default="./model/data/text_dict.json",
        help="문제를 낼 json파일을 가져옵니다.",
    )
    parser.add_argument("--seed",type=int,default=42,help = '난수 고정용')
    parser.add_argument('--n_runs',type = int, default=1, help = '몇번 실행?')
    args = parser.parse_args()

    # class 객체 생성
    inf = QAInference(args)

    inf.load_trt_model()

    inf.set_context()
    inf.set_question('이것의 이름은?')
    inf.set_dataset()
    inf.answers()