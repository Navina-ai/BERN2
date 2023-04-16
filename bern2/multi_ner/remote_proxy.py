import json
import os
from torch import Tensor
import boto3
import numpy as np

from bern2.metrics import metrics


class TritonModelProxy:
    def __init__(self, model_name, batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size

    def __call__(self, *args, **kwargs):
        with metrics.timer(f"{os.getenv('RunEnv')}.temp_debug.inference.bern2.model_call.duration"):
            runtime_sm_client = boto3.client(service_name="sagemaker-runtime",
                                             aws_access_key_id=os.environ['AwsAccessKey'],
                                             aws_secret_access_key=os.environ['AwsSecretAccessKey'])
            input_ids = kwargs['input_ids']
            attention_mask = kwargs['attention_mask']
            i = 0
            res = list()

            while i < len(input_ids):
                ids_ = input_ids[i:i + self.batch_size]
                mask_ = attention_mask[i:i + self.batch_size]
                i += self.batch_size
                payload = {
                    "inputs": [{'name': 'input_ids', 'shape': ids_.shape, "datatype": "INT64", "data": ids_.tolist()},
                               {'name': 'attention_mask', 'shape': mask_.shape, "datatype": "INT64",
                                "data": mask_.tolist()}]}
                with metrics.timer(f"{os.getenv('RunEnv')}.temp_debug.inference.bern2.invoke_endpoint.duration"):
                    endpoint_response = runtime_sm_client.invoke_endpoint(EndpointName=self.model_name,
                                                                          ContentType='application/octet-stream',
                                                                          Body=json.dumps(payload))
                parsed_response = json.loads(endpoint_response["Body"].read().decode("utf8"))['outputs'][0]
                parsed_data = np.array(parsed_response['data']).reshape(parsed_response['shape'])
                res.append(parsed_data)

            return Tensor(np.concatenate(res))
    def to(self, *args, **kwargs):
        return self

    def eval(self, *args, **kwargs):
        pass
