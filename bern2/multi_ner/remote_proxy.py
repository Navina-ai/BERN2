import json
import os

import boto3
import numpy as np


def create_remote_inference_proxy(model_name='bern2', batch_size=4):
    def call_remote_inference(**inputs):
        runtime_sm_client = boto3.client(service_name="sagemaker-runtime", aws_access_key_id=os.environ['AwsAccessKeyId'], aws_secret_access_key=os.environ['AwsSecretAccessKey'])
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        i = 0
        res = list()

        while i < len(input_ids):
            ids_ = input_ids[i:i + batch_size]
            mask_ = attention_mask[i:i + batch_size]
            i += batch_size
            payload = {"inputs": [{'name': 'input_ids', 'shape': ids_.shape, "datatype": "INT64", "data": ids_.tolist()},
                                  {'name': 'attention_mask', 'shape': mask_.shape, "datatype": "INT64", "data": mask_.tolist()}]}
            endpoint_response = runtime_sm_client.invoke_endpoint(EndpointName=model_name, ContentType='application/octet-stream', Body=json.dumps(payload))
            parsed_response = json.loads(endpoint_response["Body"].read().decode("utf8"))['outputs'][0]
            parsed_data = np.array(parsed_response['data']).reshape(parsed_response['shape'])
            res.append(parsed_data)

        return np.concatenate(res)

    return call_remote_inference
