import ctypes
import glob
import json
import os
import pathlib
import torch

from bmo_contracts.topics import (
    VISION_CMD,
    VISION_CMD_RESULT,
    VISION_CONTEXT,
    VISION_EVENTS,
)

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
GALLERY_DIR = BASE_DIR / "gallery"
OBJECT_GALLERY_DIR = GALLERY_DIR / "objects"
DATA_DIR = BASE_DIR / "data"

for _dir in (MODELS_DIR, GALLERY_DIR, OBJECT_GALLERY_DIR, DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAM_INDEX = int(os.environ.get("BMO_CAM", "0"))
MAX_FPS = int(os.environ.get("BMO_MAX_FPS", "60"))
OBJ_FRAME_STRIDE = int(os.environ.get("BMO_OBJ_STRIDE", "3"))
OBJ_CONF = float(os.environ.get("BMO_OBJ_CONF", "0.50"))
FACE_TAU = float(os.environ.get("BMO_FACE_TAU", "0.45"))
OBJECT_TAU = float(os.environ.get("BMO_OBJECT_TAU", "0.65"))
SIGN_CONF = float(os.environ.get("BMO_SIGN_CONF", "0.90"))
SEQ_LEN = 30

# Object detection model & exterior vocabulary configuration
OBJ_MODEL_TYPE = os.environ.get("BMO_OBJ_MODEL", "world").lower()
OBJECT_CLASSES_PATH = BASE_DIR / "everyday_objects.json"


def load_object_classes(filepath: pathlib.Path | str | None = None) -> list[str]:
    """Load everyday object detection classes from exterior JSON file."""
    path = pathlib.Path(filepath) if filepath is not None else OBJECT_CLASSES_PATH
    if not path.is_file():
        return ["person"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            classes = [str(c).strip() for c in data if str(c).strip()]
        elif isinstance(data, dict):
            if "classes" in data and isinstance(data["classes"], list):
                classes = [str(c).strip() for c in data["classes"] if str(c).strip()]
            else:
                classes = []
                for val in data.values():
                    if isinstance(val, list):
                        classes.extend(str(c).strip() for c in val if str(c).strip())
        else:
            classes = []
        return classes or ["person"]
    except Exception as exc:
        print(f"[config] Warning loading {path}: {exc}")
        return ["person"]


DEFAULT_OBJECT_CLASSES = load_object_classes()

# Lazy mode & Short-term memory settings
LAZY_MODE_DEFAULT = os.environ.get("BMO_LAZY", "1").lower() in ("1", "true", "yes")
MEMORY_WINDOW_SEC = float(os.environ.get("BMO_MEM_WINDOW", "60.0"))
AUTO_ENROLL_SHOTS = int(os.environ.get("BMO_ENROLL_SHOTS", "8"))

# MQTT broker settings
MQTT_HOST = os.environ.get("BMO_MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("BMO_MQTT_PORT", "1883"))
MQTT_TOPIC_CMD = os.environ.get("BMO_MQTT_TOPIC_CMD", VISION_CMD)
MQTT_TOPIC_EVENT = os.environ.get("BMO_MQTT_TOPIC_EVENT", VISION_EVENTS)
MQTT_TOPIC_CTX = os.environ.get("BMO_MQTT_TOPIC_CTX", VISION_CONTEXT)
MQTT_TOPIC_RESP = os.environ.get("BMO_MQTT_TOPIC_RESP", VISION_CMD_RESULT)

PALM_MODEL_URL = (
    "https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx"
)
PALM_MODEL_PATH = MODELS_DIR / "palm_detection.onnx"
HANDPOSE_MODEL_URL = "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx"
HANDPOSE_MODEL_PATH = MODELS_DIR / "handpose_estimation.onnx"
SIGN_MODEL_PATH = MODELS_DIR / "sign.pt"
YOLO_MODEL_PATH = MODELS_DIR / "yolo26s.pt"
YOLO_WORLD_MODEL_PATH = MODELS_DIR / "yolov8s-worldv2.pt"


def _preload_cuda_dlls() -> None:
    """Preload libcurand so ONNX Runtime CUDA provider resolves dependencies."""
    if DEVICE != "cuda":
        return
    curand_patterns = [
        str(
            BASE_DIR.parent
            / ".venv"
            / "lib"
            / "python*"
            / "site-packages"
            / "nvidia"
            / "curand"
            / "lib"
            / "libcurand.so*"
        ),
        os.path.expanduser(
            "~/.local/share/mise/installs/python/*/lib/python*/site-packages/nvidia/curand/lib/libcurand.so*"
        ),
    ]
    for pat in curand_patterns:
        for p in glob.glob(pat):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass


_preload_cuda_dlls()
import onnxruntime  # noqa: E402


def get_onnx_providers() -> list[str]:
    """Return available ONNX Runtime execution providers prioritizing CUDA."""
    available = onnxruntime.get_available_providers()
    if DEVICE == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
