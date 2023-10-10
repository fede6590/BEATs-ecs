import torch
import torchaudio
import logging
import sys
import io

from app import device


# Logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def pre_process(audio_bytes):
    wf, sr = torchaudio.load(io.BytesIO(audio_bytes))
    logger.info(f"Sample rate = {sr}")
    if sr != 16000:
        logger.info("Resampling...")
        wf = torchaudio.transforms.Resample(sr, 16000)(wf)
    return wf.to(device)


def get_label(label_pred):
    indices_list = label_pred[1][0].tolist()
    for value in indices_list:
        if value in [20, 404, 520, 151, 515, 522, 429, 199, 50, 433, 344, 34, 413, 244, 155, 245, 242]:
            return "Speech"
        elif value in [284, 19, 473, 498, 395, 81, 431, 62, 410]:
            return "Baby Crying"
        elif value in [323, 149, 339, 480, 488, 400, 150, 157]:
            return "Dog"
        elif value in [335, 221, 336, 277]:
            return "Cat"
        else:
            return "No Value"


def predict(model, audio_bytes):
    try:
        data = pre_process(audio_bytes)
        logger.info("Data ready")

        with torch.no_grad():
            logger.info("Sending to model...")
            pred = model.extract_features(data, padding_mask=None)[0]
        logger.info("Inference done")

        label_pred = pred.topk(k=5)
        label = get_label(label_pred)
        logger.info(f"Label: {label}")

        return {
            'statusCode': 200,
            'class': label,
        }
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return {
            'statusCode': 500,
            'class': None
        }
