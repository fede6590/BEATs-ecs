import torch
import torchaudio
import time
import logging
import streamlit as st

from model.BEATs import BEATs, BEATsConfig


class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        log_message = self.format(record)
        st.text(log_message)  # You can also use st.write


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(StreamlitLogHandler())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


def load_model(location):
    global model
    if model is None:
        checkpoint = torch.load(location)
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        logger.info(f"Model loaded to {device}")
    else:
        logger.info(f"Model already on {device}")
    return model.to(device)


def pre_process(wav_file, sr0):
    wf, sr = torchaudio.load(wav_file)
    if sr != sr0:
        wf = torchaudio.transforms.Resample(sr, 16000)(wf)
        logger.info(f"Resampled from {sr} to {sr0}")
    return wf.to(device)


def get_label(label_pred):
    index_list = label_pred[1][0].tolist() 
    for value in range(len(index_list)):
        if index_list[value] in [20, 404, 520, 151, 515, 522, 429, 199, 50, 433, 344, 34, 413, 244, 155, 245, 242]:
            return "Speech", 202
        elif index_list[value] in [284, 19, 473, 498, 395, 81, 431, 62, 410]:
            return "Crying baby", 200
        elif index_list[value] in [323, 149, 339, 480, 488, 400, 150, 157]:
            return "Dog", 201
        elif index_list[value] in [335, 221, 336, 277]:
            return "Cat", 202
        elif value == 4:
            return "No value", 100


def predict(audio_path):
    model = load_model('model.pt')
    try:
        data = pre_process(audio_path, 16000)  # Sample Rate = 16kHz
        logger.info("Data ready")

        with torch.no_grad():
            logger.info("Sending to model...")
            t0 = time.time()
            pred = model.extract_features(data, padding_mask=None)[0]
        logger.info(f"Inference accomplished: {round(time.time() - t0, 3)} s")

        label_pred = pred.topk(k=1)
        label, code = get_label(label_pred)
        logger.info(f"Label: {label} (code: {code})")

        return label, code

    except Exception as e:
        logger.error("An error occurred: %s", e)
        return f'ERROR ({e})'
