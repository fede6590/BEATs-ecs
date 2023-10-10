import streamlit as st
import logging
# import os
import sys
import boto3
import torch

from inference import predict
from model.BEATs import BEATs, BEATsConfig

# Logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# # S3 BUCKET
# BUCKET = os.environ['BUCKET']
# KEY = os.environ['KEY']

# Global variables
s3 = boto3.client('s3')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
location = 'model.pt'
model = None


# def download_model(bucket, key):
#     location = f'/tmp/{os.path.basename(key)}'
#     if not os.path.isfile(location):
#         logger.info(f'Downloading {key} from {bucket} bucket to {location}')
#         try:
#             s3_resource = boto3.resource('s3')
#             s3_resource.Object(bucket, key).download_file(location)
#             logger.info(f"Model downloaded to {location}")
#         except Exception as e:
#             logger.error("An error occurred while downloading model: %s", e)
#     else:
#         logger.info("Model already downloaded")
#     return location


def load_model(location):
    global model
    if model is None:
        logger.info(f"Loading model to {device}...")
        checkpoint = torch.load(location)
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        logger.info("Model already loaded")
    return model.to(device)


def main():
    # global location
    # if location is None:
    #     location = download_model(BUCKET, KEY)

    model = load_model(location)

    st.title("Audio Clasification with BEATs")
    uploaded_file = st.file_uploader("Upload an audio file...", type=["wav"])

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        prediction = predict(model, audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        st.write(f"Prediction: {prediction}")


if __name__ == '__main__':
    main()
