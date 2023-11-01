import json
import os
import boto3
from time import time, sleep
from uuid import uuid4

from main import predict, logger, load_model


queue_in = os.environ['SC_SQS_INPUT']
queue_out = os.environ['SC_SQS_OUTPUT']


region_name = 'us-east-1'
s3 = boto3.client('s3')
sqs_client = boto3.client('sqs', region_name=region_name)

location = 'model.pt'
model = None


def download_audio(event):
    input_bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    local_input_temp_file = "/tmp/" + file_key.replace('/', '-')
    logger.info(f"File_name: {local_input_temp_file}")
    s3.download_file(input_bucket_name, file_key, local_input_temp_file)
    audio_path = local_input_temp_file
    return audio_path, file_key


def delete_audio(audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    else:
        logger.error(f"'{audio_path}' doesn't exists")


def send_message_to_queue(message, file_key):
    sqs_client.send_message(
        QueueUrl=queue_out,
        MessageBody=message,
        MessageGroupId='ecs-sound-classifier-response',
        MessageDeduplicationId=file_key
    )


def inference():
    global model
    if model is None:
        model = load_model(location)
    while True:
        try:
            response = sqs_client.receive_message(QueueUrl=queue_in, MaxNumberOfMessages=1, WaitTimeSeconds=1)
            if 'Messages' in response:
                for message in response['Messages']:
                    t0 = time()
                    data = json.loads(message['Body'])
                    audio_path, file_key = download_audio(data)
                    label, code = predict(model, audio_path)
                    delete_audio(audio_path)

                    response = {
                        'classification_sound_id': f'{code}',
                        'classification_sound_description': label,
                        'records_s3_object_key': file_key,
                        'request_id': str(uuid4()),
                    }

                    response = json.dumps(response)
                    receipt_handle = message['ReceiptHandle']
                    sqs_client.delete_message(QueueUrl=queue_in, ReceiptHandle=receipt_handle)
                    send_message_to_queue(response, file_key)
                    logger.info(f'Pipeline accomplished: {round(time() - t0, 3)} s')

        except Exception as e:
            logger.error(f'Error: {str(e)}')

        sleep(1)


if __name__ == "__main__":
    inference()
