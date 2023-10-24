# Python Built-Ins:
import logging
import sys
import json
import os
import uuid

# External Dependencies:
import torch
import torchaudio
import boto3
import time

# Local Dependencies:
from BEATs import BEATs, BEATsConfig
# AWS 
region_name = 'us-east-1'
s3 = boto3.client('s3')
sqs_client = boto3.client('sqs', region_name=region_name)
# Logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Enviromental Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
MODEL_PATH = "model_weights.pt"


def model_load(model_path):
    global model
    if model is not None:
        logger.info("Model already loaded")
    else:
        logger.info("Loading Model")
        checkpoint = torch.load(model_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    return model

def download_audio(event):
    logger.info("Download Audio")
    input_bucket_name  = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    local_input_temp_file = "/tmp/" + file_key.replace('/','-')
    logger.info(f"File_name: {local_input_temp_file}")
    s3.download_file(input_bucket_name, file_key, local_input_temp_file)
    audio_path = local_input_temp_file
    return audio_path, file_key

def delete_audio(audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    else:
        logger.info(f"El archivo '{audio_path}' no existe y no se puede eliminar.")

def pre_process(audio_path):
    torchaudio.set_audio_backend("soundfile")
    waveform, original_sr = torchaudio.load(audio_path)
    resampled_waveform = torchaudio.transforms.Resample(original_sr, 16000)(waveform)

    return resampled_waveform

def get_label(label_pred):
    # Get final label
    index_list = label_pred[1][0].tolist() 

    for value in range(len(index_list)):
          if index_list[value] in [20, 404, 520, 151, 515, 522, 429, 199, 50, 433, 344, 34, 413, 244, 155, 245, 242]:
            return "Conversacion", 202
          elif index_list[value] in [284, 19, 473, 498, 395, 81, 431, 62, 410]:
            return "Bebe Llorando", 200
          elif index_list[value] in [323, 149, 339, 480, 488, 400, 150, 157]:
            return "Ladrido", 201
          elif index_list[value] in [335, 221, 336, 277]:
            return "Maullido", 202
          elif value == 4:
            return "No value", 100

def send_message_to_queue(message, file_key):
    response = sqs_client.send_message(
        QueueUrl='https://sqs.us-east-1.amazonaws.com/410677554255/ecs-output.fifo',
        MessageBody=message,
        MessageGroupId='ecs-sound-classifier-response',
        MessageDeduplicationId = file_key

    )

def predict(data):
    # Load model
    model = model_load(MODEL_PATH)
    try:
        with torch.no_grad():
            prediction = model.extract_features(data, padding_mask=None)[0]
        label_pred = prediction.topk(k=5)
        label, code = get_label(label_pred)
        return label, code
    except Exception as e:
        logger.error(f"Error en la inferencia: {str(e)}")
        return json.dumps({'error': 'Ocurrió un error en la inferencia'}), 500

def inference():
    logger.info(f"Queing")

    queue_url = "https://sqs.us-east-1.amazonaws.com/410677554255/s3-queue-ecs"
    while True:
        try:
            # Recupera mensajes de la cola SQS
            response = sqs_client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=1)

            # Verifica si se recibió un mensaje
            if 'Messages' in response:
                for message in response['Messages']:
                    tiempo_inicio = time.time()
                    # Procesa el mensaje
                    data = json.loads(message['Body'])
                    audio_path, file_key = download_audio(data)
                    data = pre_process(audio_path)
                    label, code = predict(data)
                    delete_audio(audio_path)

                    response = {
                        'classification_sound_id': f'{code}',
                        'classification_sound_description': label,
                        'records_s3_object_key': file_key,
                        'request_id' : str(uuid.uuid4()),
                    }

                    # Cargar audio en la cola
                    response = json.dumps(response)
                    tiempo_fin = time.time()
                    tiempo_transcurrido = tiempo_fin - tiempo_inicio
                    logger.info(f"Tiempo: {tiempo_transcurrido}- Inferencia: {response}")
                    receipt_handle = message['ReceiptHandle']
                    sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                    send_message_to_queue(response, file_key)

        except Exception as e:
            # Maneja errores, como excepciones de red
            print(f'Ocurrió un error: {str(e)}')
        
    
    
    



if __name__ == "__main__":
    # Lógica para configurar y ejecutar el proceso de inference
    inference()