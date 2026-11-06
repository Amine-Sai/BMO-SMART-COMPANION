import os
import pathlib
import onnxruntime
import torch

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
GALLERY_DIR = BASE_DIR / "gallery"
DATA_DIR = BASE_DIR / "data"

for _dir in (MODELS_DIR, GALLERY_DIR, DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAM_INDEX = int(os.environ.get("BMO_CAM", "0"))
OBJ_CONF = float(os.environ.get("BMO_OBJ_CONF", "0.40"))
FACE_TAU = float(os.environ.get("BMO_FACE_TAU", "0.45"))
SIGN_CONF = float(os.environ.get("BMO_SIGN_CONF", "0.90"))
SEQ_LEN = 30

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
SIGN_MODEL_PATH = MODELS_DIR / "sign.pt"
YOLO_MODEL_PATH = MODELS_DIR / "yolo26s.pt"


def get_onnx_providers() -> list[str]:
    available = onnxruntime.get_available_providers()
    if DEVICE == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
