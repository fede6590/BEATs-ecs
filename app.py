import json
import os
import boto3
from time import time, sleep
from uuid import uuid4

from main import predict, logger, load_model

region_name = 'us-east-1'
s3 = boto3.client('s3')
sqs_client = boto3.client('sqs', region_name=region_name)

location = 'model.pt'
model = None
queue_in = os.environ.get('SC_SQS_INPUT')
queue_out = os.environ.get('SC_SQS_OUTPUT')


def download_audio(event):
    input_bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    temp_file = os.path.join("/tmp/", file_key.replace('/', '_'))
    s3.download_file(input_bucket_name, file_key, temp_file)
    return temp_file, input_bucket_name, file_key


def delete_audio(temp_file):
    if os.path.isfile(temp_file):
        try:
            os.remove(temp_file)
        except Exception as e:
            logger.error(f'Error: {str(e)}')
    else:
        logger.error(f"'{temp_file}' not accesible")


def send_message_to_queue(queue, message, bucket):
    sqs_client.send_message(
        QueueUrl=queue,
        MessageBody=message,
        MessageGroupId=bucket
        )


def inference():
    global model
    if model is None:
        t0 = time()
        model = load_model(location)
        logger.info(f'Model ready ({round(time() - t0, 3)}s)')
        del t0

    while True:
        try:
            response = sqs_client.receive_message(QueueUrl=queue_in, MaxNumberOfMessages=1, WaitTimeSeconds=1)
            if 'Messages' in response:
                for message in response['Messages']:
                    data = json.loads(message['Body'])
                    if 'Records' in data:
                        t0 = time()
                        audio_path, bucket, file_key = download_audio(data)
                        label, code, prob = predict(model, audio_path)

                        response = {
                            'classification_sound_id': f'{code}',
                            'classification_sound_description': label,
                            'classification_sound_probability': prob,
                            'records_s3_object_key': file_key,
                            'request_id': str(uuid4()),
                        }

                        response = json.dumps(response)
                        receipt_handle = message['ReceiptHandle']
                        sqs_client.delete_message(QueueUrl=queue_in, ReceiptHandle=receipt_handle)
                        send_message_to_queue(queue_out, response, bucket)
                        logger.info(f'Inference time: {round(time() - t0, 3)}s - Response: {response}')
                        delete_audio(audio_path)
                        del t0

        except Exception as e:
            logger.error(f'Error: {str(e)}')

        sleep(1)


if __name__ == "__main__":
    inference()
