import json
import os

import boto3
import numpy as np

class TritonModelProxy:
    def __init__(self, model_name, batch_size=32):
        self.model_name=model_name
        self.batch_size=batch_size
    def __call__(self, *args, **kwargs):
        runtime_sm_client = boto3.client(service_name="sagemaker-runtime",
                                         aws_access_key_id=os.environ['AwsAccessKey'],
                                         aws_secret_access_key=os.environ['AwsSecretAccessKey'])
        input_ids = kwargs['input_ids']
        attention_mask = kwargs['attention_mask']
        i = 0
        res = list()

        while i < len(input_ids):
            ids_ = input_ids[i:i + batch_size]
            mask_ = attention_mask[i:i + batch_size]
            i += batch_size
            payload = {
                "inputs": [{'name': 'input_ids', 'shape': ids_.shape, "datatype": "INT64", "data": ids_.tolist()},
                           {'name': 'attention_mask', 'shape': mask_.shape, "datatype": "INT64",
                            "data": mask_.tolist()}]}
            endpoint_response = runtime_sm_client.invoke_endpoint(EndpointName=model_name,
                                                                  ContentType='application/octet-stream',
                                                                  Body=json.dumps(payload))
            parsed_response = json.loads(endpoint_response["Body"].read().decode("utf8"))['outputs'][0]
            parsed_data = np.array(parsed_response['data']).reshape(parsed_response['shape'])
            res.append(parsed_data)

        return np.concatenate(res)
    def to(self, *args, **kwargs):
        return self
    def eval(self, *args, **kwargs):
        pass
