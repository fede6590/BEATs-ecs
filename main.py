import torch
import torchaudio
import logging
import sys
import os

from model.BEATs import BEATs, BEATsConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k = os.environ.get("TOPK", 5)
thresh = os.environ.get('THRESH', .5)


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
    index_list = label_pred[1]
    for i, code in enumerate(index_list, start=1):
        if code in [20, 404, 520, 151, 515, 522, 429, 199, 50, 433, 344, 34, 413, 244, 155, 245, 242]:
            return "Speech", 202, label_pred[0][i-1]
        elif code in [284, 19, 473, 498, 395, 81, 431, 62, 410]:
            return "Crying baby", 200, label_pred[0][i-1]
        elif code in [323, 149, 339, 480, 488, 400, 150, 157]:
            return "Dog", 201, label_pred[0][i-1]
        elif code in [335, 221, 336, 277]:
            return "Cat", 203, label_pred[0][i-1]
        elif i == len(index_list):
            return "No value", 100, 0


def post_process(pred, k, thresh):
    topk_pred = pred.topk(k=k)
    mask = (topk_pred.values >= thresh)
    if True not in mask:
        mask[0][0] = True
    probs = topk_pred.values[mask].tolist()
    preds = topk_pred.indices[mask].tolist()
    return [probs, preds]


def predict(model, audio_path):
    try:
        data = pre_process(audio_path, 16000)  # Sample Rate = 16kHz
        with torch.no_grad():
            pred = model.extract_features(data, padding_mask=None)[0]
        label_pred = post_process(pred, k=k, thresh=thresh)
        label, code, prob = get_label(label_pred)
        return label, code, round(prob, 2)

    except Exception as e:
        logger.error("An error occurred: %s", e)
        return f'ERROR ({e})'
