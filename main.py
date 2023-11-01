import torch
import torchaudio
import time
import logging
import sys

from model.BEATs import BEATs, BEATsConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(location):
    checkpoint = torch.load(location)
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model.to(device)


def pre_process(wav_file, sr0):
    wf, sr = torchaudio.load(wav_file)
    if sr != sr0:
        wf = torchaudio.transforms.Resample(sr, 16000)(wf)
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


def predict(model, audio_path):
    try:
        data = pre_process(audio_path, 16000)  # Sample Rate = 16kHz

        with torch.no_grad():
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
